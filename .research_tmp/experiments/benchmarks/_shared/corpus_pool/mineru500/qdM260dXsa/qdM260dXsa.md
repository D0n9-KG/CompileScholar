# Cross-Domain Policy Adaptation via Value-Guided Data Filtering

Kang Xu $^{1}$ $^{2*}$ Chenjia Bai $^{2\dagger}$ Xiaoteng Ma $^{3}$ Dong Wang $^{2}$ Bin Zhao $^{2}$ $^{4}$

Zhen Wang $^{4}$ Xuelong Li $^{2\ 4}$ Wei Li $^{1\dagger}$

$^{1}$ Fudan University $^{2}$ Shanghai Artificial Intelligence Laboratory $^{3}$ Tsinghua University $^{4}$ Northwestern Polytechnical University

# Abstract

Generalizing policies across different domains with dynamics mismatch poses a significant challenge in reinforcement learning. For example, a robot learns the policy in a simulator, but when it is deployed in the real world, the dynamics of the environment may be different. Given the source and target domain with dynamics mismatch, we consider the online dynamics adaptation problem, in which case the agent can access sufficient source domain data while online interactions with the target domain are limited. Existing research has attempted to solve the problem from the dynamics discrepancy perspective. In this work, we reveal the limitations of these methods and explore the problem from the value difference perspective via a novel insight on the value consistency across domains. Specifically, we present the Value-Guided Data Filtering (VGDF) algorithm, which selectively shares transitions from the source domain based on the proximity of paired value targets across the two domains. Empirical results on various environments with kinematic and morphology shifts demonstrate that our method achieves superior performance compared to prior approaches.

# 1 Introduction

Reinforcement Learning (RL) has demonstrated the ability to train highly effective policies with complex behaviors through extensive interactions with the environment $[62, 59, 2]$ . However, in many situations, extensive interactions are infeasible due to the data collection costs and the potential safety hazards associated with domains such as robotics $[33]$ and medical treatments $[54]$ . To address the issue, one approach is to interact with a surrogate environment, such as a simulator, and then transfer the learned policy to the original domain. However, an unbiased simulator may be unavailable due to the complex system dynamics or unexpected disturbances in the target scenario, leading to a dynamics mismatch. Such a mismatch is crucial for the sim-to-real problem in robotics $[1, 38, 51]$ and may cause performance degradation of the learned policy in the target domain. In this work, we focus on the dynamics adaptation problem, where we aim to train a well-performing policy for the target domain, given the source domain with the dynamics mismatch.

Recent research has tackled the adaptation over dynamics mismatch through various techniques, such as domain randomization $[56, 53, 45]$ , system identification $[77]$ , or simulator calibration $[8]$ , that require domain knowledge or privileged access to the physical system. Other methods have explored

![](images/542e94743b2a9796d5ab8ce3937c3cea4bbd45e4a70b50d66be58c117403b0e6.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Source Domain"] -->|Train| B["RL"]
    C["Target Domain"] -->|Deployment| B
    B --> D["(a) Dynamics Randomization/System Identification/Meta RL"]
```
</details>

![](images/1f4dd7dc68b5a0e7be5e9c7f273f8e341191cf3941400b4a1a1a7594c05386f0.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Expert/Offline Data"] -->|Train| B["Source Domain"]
    B --> C["RL"]
    C --> D["Target Domain"]
    D --> E["Deployment"]
    style A fill:#4CAF50,stroke:#333
    style B fill:#2196F3,stroke:#333
    style C fill:#2196F3,stroke:#333
    style D fill:#2196F3,stroke:#333
    style E fill:#2196F3,stroke:#333
```
</details>

![](images/8018ba192053530f5fb8140ca566cb79804e4bc420e1c90cdd7f59988288e647.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Target Domain"] -->|Calibration| B["RL"]
    B -->|Train| C["Manipulable Source Domain"]
    C -->|Deployment| A
    A -->|Limited data| B
    style A fill:#99CCFF,stroke:#333
    style B fill:#99CCFF,stroke:#333
    style C fill:#99CCFF,stroke:#333
```
</details>

![](images/286735a26f552e34f592099e8e2f6c46a32bed4ad896bf653b1f72ab0bb5d0e1.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Source Domain"] -->|Train| B["RL"]
    C["Source Domain Offline Data"] -->|Train| B
    D["Target Domain"] -->|Deployment| B
    B -->|Limited Online Interactions| D
```
</details>

Figure 1: Semantic illustration of main settings for dynamics adaptation problem. Methods in the first three categories require different assumptions, such as a wide range of source domains, demonstrations from the target domain, or a manipulable simulator. We focus on a more general setting, online dynamics adaptation, only requiring limited online interactions with the target domain.

the adaptation problem in specific scenarios, such as those with expert demonstrations $[41, 32]$ or offline datasets $[42, 49]$ , while the effectiveness of these methods heavily depends on the optimality of demonstrations or the quality of the datasets. In contrast to these works, we consider a more general setting called online dynamics adaptation, where the agent can access sufficient source domain data and a limited number of online interactions with the target domain. We compare the settings for the dynamics adaptation problem in Figure 1.

To address the online dynamics adaptation problem, prior works mainly focus on the single-step dynamics discrepancy and practically eliminating the gap via different ways $[17, 14]$ . However, we empirically demonstrate the limitation of the methods through a motivation example, suggesting their effectiveness heavily relies on strong assumptions about the transferability of paired domains. Theoretically, we formulate the performance bound of the learned policy with respect to the dynamics discrepancy term, which provides an explicit interpretation of the results. To address the problem, we focus on the value discrepancy between paired transitions across domains, motivated by the key idea: the transitions with consistent value targets can be seen as equivalent for policy adaptation. Based on the insight, we proposed a simple yet efficient algorithm called Value-Guided Data Filtering (VGDF) for online dynamics adaptation via selective data sharing. Specifically, we use a learned target domain dynamics model to obtain paired transitions based on the source domain state-action pair. The transitions are shared from the source to the target domain only if the value targets of the imagined target domain transition and that of the source domain transition are close. Compared to previous methods that utilize the single-step dynamics gap, our method measures value discrepancies to capture long-term differences between two domains for better adaptation.

Our contributions can be summarized as follows: 1) We reveal the limitations of prior dynamics-based methods and propose the value discrepancy perspective with theoretical analysis. 2) To provide a practical instantiation, we propose VGDF for online dynamics adaptation via selective data sharing. 3) We extend VGDF to a more practical setting with an offline source domain dataset and propose a variant algorithm motivated by novel theoretical results. 4) We empirically demonstrate the superior performance of our method given significant dynamics shifts, including kinematics and morphology mismatch, compared to previous methods.

# 2 Related Work

Domain adaptation in RL. Different from domain adaptation in supervised learning where different domains correspond to distinct data distributions $[34]$ , different domains in RL can differ in observation space $[26]$ , transition dynamics $[56, 77, 17]$ , embodiment $[79, 43]$ , or reward functions $[16, 81, 57]$ . In this work, we focus on domain adaptation with dynamics discrepancies. Prior works utilizing meta RL $[76, 48, 55]$ , domain randomization $[56, 53, 45]$ , and system identification $[80, 77, 15, 74]$ all assume the access to the distribution of training environments and rely on the hypothesis that the source and target domains are drawn from the same distribution. Another line of work has proposed to handle domain adaptation given expert demonstrations from the target domain $[41, 32, 27]$ . These approaches align the state visitation distributions of the trained policy in the source domain to the distribution of the expert demonstrations in the target domain through state-action correspondences $[79]$ or imitation learning $[28, 21, 72]$ . However, near-optimal demonstrations can be challenging to acquire in some tasks. More recent works have explored the dynamics adaptation given an offline dataset collected in the target domain $[42, 49]$ , while the performance of

the trained policy depends on the quality of the dataset $[50]$ . Orthogonal to these settings, we focus on a general paradigm where a relatively small number of online interactions with the target domain are accessible.

Online dynamics adaptation. Given limited online interactions with the target domain, several works calibrate the dynamics of the source domain by adjusting the physical parameters of the simulator $[8, 58, 15, 47]$ , while they assume the access of a manipulable simulator. Action transformation methods correct the transitions collected in the source domain by learning dynamics models of the two domains $[25, 14, 78]$ . However, the learned model can be inaccurate, which results in model exploitation and performance degradation $[30, 31]$ . Furthermore, the work that compensates the dynamics gap by modifying the reward function $[17]$ is practical only if the policy that performs well in both domains exists. Instead, we do not assume the dynamics-agnostic policy exists and demonstrate the effectiveness of our method when such an assumption does not hold.

Knowledge transfer in RL. Knowledge transfer has been proposed to reuse the knowledge from other tasks to boost the training for the current task $[69, 37]$ . The transferred knowledge can be modules (e.g., policy) $[52, 9, 4]$ , representations $[5]$ , and experiences $[29, 39, 75, 68]$ . Our method is related to works transferring experiences. However, prior works focus on transferring between tasks with different reward functions instead of dynamics. When the dynamics changes, the direct adoption of commonly used temporal difference error $[63]$ or advantage function $[60]$ in previous works $[29, 39, 68]$ would be inappropriate due to the shifted transition probabilities across domains. In contrast, we introduce novel measurements to evaluate the usefulness of the source domain transitions to tackle the dynamics shift problem specifically.

Theories on learning with dynamics mismatch. The performance guarantee of a policy trained with imaginary transitions from an inaccurate dynamics model has been analyzed in prior Dynastyle $[64, 65, 67]$ model-based RL algorithms $[44, 30, 61]$ . The theoretical results inspire us to formulate performance guarantees in the context of dynamics adaptation.

# 3 Preliminaries and Problem Statement

We consider two infinite-horizon Markov Decision Processes (MDP) $\mathcal{M}_{src} := (\mathcal{S}, \mathcal{A}, P_{src}, r, \gamma, \rho_0)$ and $\mathcal{M}_{tar} := (\mathcal{S}, \mathcal{A}, P_{tar}, r, \gamma, \rho_0)$ for the source domain and the target domain, respectively. The two domains share the same state space S, action space A, reward function $r : S \times A \to R$ with range $[0, r_{\max}]$ , discount factor $\gamma \in [0, 1)$ , and the initial state distribution $\rho_0 : S \to [0, 1]$ . The two domains differ on the transition probabilities, i.e., $P_{src}(s'|s, a)$ and $P_{tar}(s'|s, a)$ .

We define the probability that a policy $\pi$ encounters state s at the time step t in MDP M as $\mathrm{P}_{\mathcal{M},t}^{\pi}(s)$ . We denote the normalized probability that a policy $\pi$ encounters state s in M as $\nu_{\mathcal{M}}^{\pi}(s):=(1-\gamma)\sum_{t=0}^{\infty}\gamma^{t}\mathrm{P}_{\mathcal{M},t}^{\pi}(s)$ , and the normalized probability that a policy encounters state-action pair $(s,a)$ in M is $\rho_{\mathcal{M}}^{\pi}(s,a):=(1-\gamma)\sum_{t=0}^{\infty}\gamma^{t}\mathrm{P}_{\mathcal{M},t}^{\pi}(s)\pi(a|s)$ . The performance of a policy $\pi$ in M as is formally defined as $\eta_{\mathcal{M}}(\pi):=\mathbb{E}_{s,a\sim\rho_{\mathcal{M}}^{\pi}}[r(s,a)]$ .

We focus on the online dynamics adaptation problem where limited online interactions with the target domain are accessible, which can be defined as follows:

Definition 3.1. (Online Dynamics Adaptation) Given source domain $M_{src}$ and target domain $M_{tar}$ with different dynamics, we assume sufficient data from the source domain (online or offline) and a relatively small number of online interactions with $M_{tar}$ (e.g., $\Gamma := \frac{\# \text{ source domain data}}{\# \text{ target domain data}} = 10$ ), hoping to obtain a near-optimal policy $\pi$ concerning the target domain $M_{tar}$ .

The prior work [17] also focuses on the online dynamics adaptation problem with online source domain interactions. The proposed algorithm DARC estimates the dynamics discrepancy via learned domain classifiers and further introduces a reward correction (i.e., $\Delta r(s,a,s') \approx \log(P_{tar}(s'|s,a)/P_{src}(s'|s,a)))$ to optimize policy together with the task reward $r(i.e., r(s,a) + \Delta r(s,a,s'))$ , discouraging the agent from dynamics-inconsistent behaviors in the source domain.

# 4 Guaranteeing Policy Performance from a Value Discrepancy Perspective

In this section, we will first present an example demonstrating the limitation of the prior method considering the dynamics discrepancy. Following that, we provide a theoretical analysis of the

![](images/ac0ba54ce27c7f54e9f993674434d7faba7a16c6640954691761e04f8d2d7c42.jpg)  
(a)

![](images/5204191019a42dab5bc5f96d818428b4cf862826a42f1e6ecc5958ff5d8a9d3f.jpg)

<details>
<summary>heatmap</summary>

| Domain       | State Visitation |
| ------------ | ---------------- |
| Source Domain | High             |
| Target Domain | Medium to Low    |
| Ours         | High             |
| DARC         | Low              |
</details>

(b)

![](images/f0770afe6497fda03aed0d10e09aa92a60f0fa70a92824ed899e47c9fda68a70.jpg)

<details>
<summary>text_image</summary>

Learned Q Table
Ours
DARC
</details>

(c)   
Figure 2: The illustrations and results of the motivation experiment. (a) Illustration of the source and target domains in the grid world environment. The red dot and green square represent the agent and goal, respectively. (b) Visualization of the state visitation in both domains. The darker color suggests higher visitation probabilities. Our method guides the agent to reach regions with high target domain values while the agent trained by DARC is stuck in the room. (c) Visualization of the learned Q tables. Four triangles represent four actions; the darker color suggests a higher value estimation. Our method learns the optimal Q table whose greedy policy leads the agent to the goal of the target domain, while DARC fails due to pessimistic values of the crucial state-action pairs with dynamics mismatch.

dynamics-based method to provide an interpretation of the experiment results. Finally, we introduce a novel perspective on value discrepancies across domains for the online dynamics adaptation problem.

# 4.1 Motivation Example

We start with a 2D grid world task shown in Figure 2 (a), where the agent represented by the red dot needs to navigate to the green square representing the goal. We design source and target domains with different layouts and train a policy to reach the goal successfully in the target domain. We investigate the performance of DARC [17] that trains the policy with dynamics-guided reward correction and our proposed method (Section 5), using tabular Q-learning [73] as the backbone for all methods. Detailed environment settings are shown in Appendix D.

As the empirical state visitations and the learned Q tables show in Figure 2, DARC is stuck in the room and fails to obtain near-optimal Q-values, leading to poor performance. Specifically, we circle out four positions where specific actions will lead to the states with a dynamics mismatch concerning the two domains. Due to the introduced reward correction on the source domain transitions with dynamics mismatch, DARC learns overly pessimistic value estimations of particular state-action pairs, which hinders the agent from the optimal trajectory concerning the target domain. However, the values of the following inconsistent states, induced by the particular state-action pairs, are not significantly different concerning the target domain. The value difference quantifies the discrepancy of the long-term behaviors rather than single-step dynamics. Motivated by the value discrepancy perspective, our proposed method (Section 5.1) demonstrates superior performance.

# 4.2 Theoretical Interpretations and Value Discrepancy Perspective

To provide rigorous interpretations for the results, we derive a performance guarantee for the dynamics-guided methods, which mainly build on the theories proposed in prior methods $[30, 17]$ .

Theorem 4.1. (Performance bound controlled by dynamics discrepancy.) Denote the source domain and target domain with different dynamics as $M_{src}$ and $M_{tar}$ , respectively. We have the performance difference of any policy $\pi$ evaluated under $M_{src}$ and $M_{tar}$ be bounded as below,

$$
\eta_ {\mathcal {M} _ {t a r}} (\pi) \geq \eta_ {\mathcal {M} _ {s r c}} (\pi) - \frac {2 \gamma r _ {\max}}{(1 - \gamma) ^ {2}} \cdot \underbrace {\mathbb {E} _ {\rho_ {s r c} ^ {\pi}} \left[ D _ {\mathrm{TV}} \left(P _ {s r c} (\cdot | s , a) \| P _ {t a r} (\cdot | s , a)\right) \right]} _ {(a) d y n a m i c s d i s c r e p a n c y}. \tag {1}
$$

The proof of Theorem 4.1 is given in Appendix B. We observe that the derived performance bound in (1) is controlled by the dynamics discrepancy term (a). Intuitively, the performance difference would be minor when the dynamics discrepancy between the two domains is negligible. DARC [17] applies

the Pinsker's inequality [13] and derives the following form:

$$
\begin{array}{l} \eta_ {\mathcal {M} _ {t a r}} (\pi) \geq \eta_ {\mathcal {M} _ {s r c}} (\pi) - \frac {\gamma r _ {\max}}{(1 - \gamma) ^ {2}} \cdot \sqrt {2 \mathbb {E} _ {\rho_ {s r c} ^ {\pi}} [ D _ {\mathrm{KL}} (P _ {s r c} (\cdot | s , a) \| P _ {t a r} (\cdot | s , a)) ]} \\ = \eta_ {\mathcal {M} _ {s r c}} (\pi) + \frac {\gamma r _ {\max}}{(1 - \gamma) ^ {2}} \cdot \sqrt {2 \mathbb {E} _ {\rho_ {s r c} ^ {\pi} , P _ {s r c}} [ \log (P _ {t a r} (s ^ {\prime} | s , a) / P _ {s r c} (s ^ {\prime} | s , a)) ]}. \tag {2} \\ \end{array}
$$

Based on the result in (2), DARC optimizes the policy by converting the second term in RHS to a reward correction (i.e., $\Delta r := \log(P_{tar}(s'|s, a)/P_{src}(s'|s, a)))$ , leading to the dynamics discrepancy-based adaptation. However, given the transition from the source domain (i.e., $P_{src}(s'|s, a) \approx 1$ ), the reward correction will lead to significant penalty (i.e., $\log(P_{tar}(s'|s, a)/P_{src}(s'|s, a)) \ll 0$ ) if the likelihood estimation of the transition concerning the target domain is low (i.e., $P_{tar}(s'|s, a) \approx 0$ ). Consequently, the value estimation of the transition with dynamics mismatch tends to be overly pessimistic as shown in Figure 2 (c), which hinders learning an effective policy concerning the target domain.

Instead of myopically considering the single-step dynamics mismatch, we claim that the transitions with significant dynamics mismatch can be equivalent concerning the value estimations that evaluate the long-term behaviors. Due to the dynamics shift across domains, a state-action pair $(i.e., (s, a))$ would lead to two different next-states $(i.e., s'_{src}, s'_{tar})$ , the paired transitions are nearly equivalent for temporal different learning if the induced value estimations are close $(i.e., |V(s'_{src}) - V(s'_{tar})| \leq \epsilon)$ . Motivated by this, we derive a performance guarantee from the value difference perspective.

Theorem 4.2. (Performance bound controlled by value difference.) Denote source domain and target domain as $M_{src}$ and $M_{tar}$ , respectively. We have the performance guarantee of any policy $\pi$ over the two MDPs:

$$
\eta_ {\mathcal {M} _ {t a r}} (\pi) \geq \eta_ {\mathcal {M} _ {s r c}} (\pi) - \frac {\gamma}{1 - \gamma} \cdot \underbrace {\mathbb {E} _ {\rho_ {\mathcal {M} _ {s r c}} ^ {\pi}} \left[ \left| \mathbb {E} _ {P _ {s r c}} \left[ V _ {\mathcal {M} _ {t a r}} ^ {\pi} (s ^ {\prime}) \right] - \mathbb {E} _ {P _ {t a r}} \left[ V _ {\mathcal {M} _ {t a r}} ^ {\pi} (s ^ {\prime}) \right] \right| \right]} _ {(a): v a l u e d i f f e r e n c e}. \tag {3}
$$

The proof of Theorem 4.2 is given in Appendix B. The value difference term provides a novel perspective: the performance can be guaranteed if the transitions from the source domain lead to consistent value targets in the target domain. The result further highlights the value consistency perspective for the online dynamics adaptation problem.

# 5 Value-Guided Data Filtering

In this section, we propose Value-Guided Data Filtering (VGDF), a simple yet efficient algorithm for online domain adaptation via selective data sharing. Then we introduce the setting with offline source domain data and a variant of VGDF based on novel theoretical results. The pseudocodes are shown in Appendix A, and the illustration of VGDF is shown in Figure 3.

# 5.1 Dynamics Adaptation by Selective Data Sharing

Inspired by the performance bound proposed in Theorem 4.2, we can guarantee the policy performance by controlling the value difference term in (3). As discussed in Section 4.2, the paired transitions concerning two domains, induced by the same state-action pair, can be regarded as equivalent for temporal difference learning when the corresponding values are close. Thus, we propose to select source domain transitions with minor value discrepancies for dynamics adaptation.

To select rational transitions from the source domain, we need to compare the value differences of paired transitions based on the same source domain state-action pair $(s_{src}, a_{src})$ . Formally, given a state-action pair $(s_{src}, a_{src})$ from the source domain, our objective is to estimate whether the value-difference between $s'_{tar}$ and $s'_{src}$ is sufficiently small, i.e.,

$$
\Delta \left(s _ {s r c}, a _ {s r c}\right) := \mathbb {1} \left(\left| V _ {\mathcal {M} _ {t a r}} ^ {\pi} \left(s _ {t a r} ^ {\prime}\right) - V _ {\mathcal {M} _ {t a r}} ^ {\pi} \left(s _ {s r c} ^ {\prime}\right) \right| \leq \epsilon\right), \tag {4}
$$

where $s_{tar}' \sim P_{tar}(\cdot | s_{src}, a_{src})$ , $s_{src}' \sim P_{src}(\cdot | s_{src}, a_{src})$ , 1 denotes the indicator function and $\epsilon$ can be a predefined threshold.

To obtain $\Delta(s_{src}, a_{src})$ , we need to perform policy evaluation over the states to obtain the value estimations given the paired next states (i.e., $s_{src}', s_{tar}'$ ), as formulated in Eq. (4). Monte Carlo (MC)

![](images/f679b68ed0ac3221c8e294ab2138d847c64828354a229da87d004c4ecbafd5d4.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Source Domain"] --> B["D_src"]
    C["Target Domain"] --> D["D_tar"]
    B --> E["Data Filtering"]
    D --> E
    E --> F["RL"]
    G["D_src"] --> H["(s,a,r,s'_src)"]
    H --> I["Rejection Sampling"]
    J["(s,a)"] --> K["{T_θ}^M"]
    K --> L["{(s'_tar)}^M"]
    L --> M["{(Q_tar)}^M"]
    M --> N["1(Λ ≥ Λξ%)"]
    O["P[· |Mean((Q_tar^π)^M),Var([Q_tar^π)^M)"]] --> N
    N --> P["Histogram Curve"]
```
</details>

Figure 3: Semantic illustration of VGDF. We tackle online dynamics adaptation by selectively sharing the source domain data, and the RL denotes any off-the-shelf off-policy RL algorithm.

evaluation can provide unbiased values by rolling the policy starting from specific states [66]. However, since the environment is not manipulable, we cannot perform MC evaluation from arbitrary states. Thus, we propose to use an estimated value function for policy evaluation. In this work, we adopt the Fitted Q Evaluation (FQE) [46] that is widely used in off-policy RL algorithms [40, 23, 24]. Specifically, we utilize a learned Q function $Q_{\theta}: S \times A \to \mathbb{R}$ for evaluation.

Furthermore, one problem is that the corresponding target domain next state $s_{tar}'$ induced by $(s_{src}, a_{src})$ is unavailable in practice. To achieve this, we train a dynamics model with the collected data from the target domain. Following prior works [36, 10], we employ an ensemble of Gaussian dynamics models $\{T_{\phi_i}(s'|s, a)\}_{i=1}^M$ , in an attempt to capture the epistemic uncertainty due to the insufficient target domain samples. Given the source domain state-action pair $(s_{src}, a_{src})$ , we generate an ensemble of fictitious states and obtain the corresponding values for each state-action pair, which we call fictitious value ensemble (FVE) $\mathcal{Q}_{tar}^\pi(s_{src}, a_{src})$ :

$$
\mathcal {Q} _ {t a r} ^ {\pi} (s _ {s r c}, a _ {s r c}) := \left\{Q _ {\theta} (s _ {i} ^ {\prime}, a _ {i} ^ {\prime}) | _ {s _ {i} ^ {\prime} \sim T _ {\phi_ {i}} (\cdot | s _ {s r c}, a _ {s r c}), a _ {i} ^ {\prime} \sim \pi (\cdot | s _ {i} ^ {\prime})} \right\} _ {i = 1} ^ {M}. \tag {5}
$$

In practice, the choice of $\epsilon$ in Eq. (4) is also nontrivial due to task-specific scales of the values and the non-stationary value function during training. We replace the absolute value difference with the likelihood estimation to address the problem. Specifically, we construct a Gaussian distribution with the mean and variance of FVE denoted as $\mathcal{N}(\text{Mean}(\mathcal{Q}_{tar}^{\pi}(s_{src},a_{src})),\text{Var}(\mathcal{Q}_{tar}^{\pi}(s_{src},a_{src})))$ . Estimating the value of the source domain state as $V_{tar}^{\pi}(s_{src}^{\prime}):= Q_{\theta}(s_{src}^{\prime},a_{src}^{\prime})|_{a_{src}^{\prime}\sim\pi(\cdot|s_{src}^{\prime})}$ , we introduce Fictitious Value Proximity (FVP) representing the likelihood of the source domain state value in the distribution:

$$
\Lambda (s _ {s r c}, a _ {s r c}, s _ {s r c} ^ {\prime}) := \mathbb {P} (V _ {t a r} ^ {\pi} (s _ {s r c} ^ {\prime}) \mid \mathrm{Mean} (\mathcal {Q} _ {t a r} ^ {\pi} (s _ {s r c}, a _ {s r c})), \mathrm{Var} (\mathcal {Q} _ {t a r} ^ {\pi} (s _ {s r c}, a _ {s r c}))). \tag {6}
$$

Based on the likelihood estimation, we utilize the rejection sampling to select fixed percentage data (i.e., 25%) with the highest likelihood from a batch of source domain transitions at each training iteration. Specifically, we train the value function by optimizing the following objective:

$$
\theta \gets \underset {\theta} {\arg \min} \frac {1}{2} \mathbb {E} _ {(s, a, r, s ^ {\prime}) \sim D _ {t a r}} \left[ (Q _ {\theta} - \mathcal {T} Q _ {\theta}) ^ {2} \right] + \frac {1}{2} \mathbb {E} _ {(s, a, r, s ^ {\prime}) \sim D _ {s r c}} \left[ \omega (s, a, s ^ {\prime}) (Q _ {\theta} - \mathcal {T} Q _ {\theta}) ^ {2} \right],
$$

where $\omega(s,a,s^{\prime}):=1\left(\Lambda(s,a,s^{\prime})>\Lambda_{\xi\%}\right)$ . (7)

$\Lambda_{\xi\%}$ is the top $\xi$ -quantile likelihood estimation of the minibatch sampled from source domain data, $\mathcal{T}$ represents the Bellman operator, and $D_{src}, D_{tar}$ denote replay buffers of two domains.

Consider the case when the agent can perform online interactions with the source domain, the training data mostly comes from the source domain, while we aim to train a policy for the target domain. Hence, exploring the source domain is essential to collect transitions that might be high-value concerning the target domain. Thus, we introduce an exploration policy $\pi^{E}$ that maximizes the approximate upper confidence bound of the Q-value, i.e., $\pi^{E} \leftarrow \arg\max_{\pi^{E}} E_{s \sim D_{tar} \cup D_{src}} \left[ Q_{\mathrm{UB}}(s, a) |_{a \sim \pi^{\mathrm{E}}(\cdot | s)} \right]$ , where $Q_{\mathrm{UB}}(s, a) := \max \left\{ Q_{\theta_i}(s, a) \right\}_{i=1}^2$ under the implementation with SAC [24] backbone. Importantly, the exploration policy $\pi^{E}$ is separate from the main policy $\pi$ learned via vanilla SAC. $\pi^{E}$ and $\pi$ are used for data collection in the source domain and target domain, respectively. The optimistic data collection technique has been proposed for advanced exploration [11] while we utilize the technique in online dynamics adaptation setting.

# 5.2 Adaptation with Offline Dataset of Source Domain

So far, we have discussed the setting where the agent can interact with the source domain to collect data actively. Nonetheless, simultaneous online access to the source and target domain might sometimes be impractical. In order to address the limitation, we aim to extend our method to the setting we refer to as Offline Source with Online Target, in which the agent can access a source domain offline dataset and a relatively small number of online interactions with the target domain.

To adapt VGDF to such a setting, we propose a novel theoretical result of the performance guarantee:

Theorem 5.1. Under the setting with offline source domain dataset D whose empirical estimation of the data collection policy is $\pi_{D}(a|s):=\frac{\sum_{D}\mathbb{1}(s,a)}{\sum_{D}\mathbb{1}(s)}$ , let $M_{src}$ and $M_{tar}$ denote the source and target domain, respectively. We have the performance guarantee of any policy $\pi$ over the two MDPs:

$$
\eta_ {\mathcal {M} _ {t a r}} (\pi) \geq \eta_ {\mathcal {M} _ {s r c}} (\pi) - \frac {4 r _ {\max}}{(1 - \gamma) ^ {2}} \underbrace {\mathbb {E} _ {\rho_ {\mathcal {M} _ {s r c}} ^ {\pi_ {D}} , P _ {s r c}} [ D _ {T V} (\pi_ {D} | | \pi) ]} _ {(a): p o l i c y r e g u l a r i z a t i o n} - \frac {1}{1 - \gamma} \underbrace {\mathbb {E} _ {\rho_ {\mathcal {M} _ {s r c}} ^ {\pi_ {D}}} \left[ \left| \zeta (s , a) \right| \right]} _ {(b): v a l u e d i f f e r e n c e}, \tag {8}
$$

where $\zeta(s, a) := \mathbb{E}_{P_{src}, \pi} \left[ Q_{\mathcal{M}_{tar}}^{\pi}(s', a') \right] - \mathbb{E}_{P_{tar}, \pi} \left[ Q_{\mathcal{M}_{tar}}^{\pi}(s', a') \right]$ .

The proof of Theorem 5.1 is given in Appendix B. This theorem highlights the importance of policy regularization and value difference for achieving desirable performance. It is worth noting that the policy regularization term can shed light on the impact of behavior cloning, which has been proven effective for offline RL [22]. Additionally, the value difference term has a similar structure to that of Theorem 3. Thus, we propose a variant called $VGDF + BC$ that combines behavior cloning loss with the original selective data sharing scheme. The pseudocode is shown in Algorithm 2, Appendix A.

# 6 Experiments

In this section, we present empirical investigations of our approach. We examine the effectiveness of our method in scenarios with various dynamics shifts, including kinematic change and morphology change. Furthermore, we provide ablation studies and qualitative analysis of our method. Details of environment settings and the implementation are shown in Appendix D and Appendix E, respectively. Additional results are in Appendix F.

# 6.1 Adaptation Performance Evaluation

To systematically investigate the adaptation performance of the methods, we construct two types of dynamics shift scenarios, including kinematic shift and morphology shifts, for four environments (HalfCheetah, Ant, Walker, Hopper) from Gym Mujoco $[71, 7]$ . We use the original environment as the source domain across all experiments. To simulate kinematic shifts, we limit the rotation angle range of specific joints to simulate the broken joint scenario. As for morphology shifts, we modify the size of specific limbs while the number of limbs keeps unchanged to ensure the state/action space consistent across domains. Full details of the environment settings are deferred to Appendix D.

We compare our algorithm with four baselines: (i) DARC [17] trains the domain classifiers to compensate the agent with an extra reward for seeking dynamics-consistent behaviors; (ii) GARAT [14] trains the policy with an adversarial imitation reward in the grounded source domain via action transformation [25]; (iii) IW Clip (Importance Weighting Clip) performs importance-weighted bellman updates for source domain samples. The importance weights $(i.e., P_{tar}(s'|s,a)/P_{src}(s'|s,a))$ are approximated by the domain classifiers proposed in DARC, and we clip the weight to $[10^{-4},1]$ to stabilize training; (iv) Finetune uses the $10^{5}$ target domain transitions to finetune the policy trained in the source domain with 1M samples. Furthermore, Zero-shot shows the performance of directly transferring the learned policy in the source domain to the target domain, and Oracle demonstrates the performance of the policy trained in the target domain from scratch with 1M transitions. We run all algorithms with the same five random seeds. The implementation details are given in Appendix E.1.

As the results in Figure 4 show, our method outperforms GARAT and IW Clip in all environments. DARC demonstrates competitive performance only in the first two environments, while it does not work in other environments. We believe that the assumption of DARC does not hold in the failure cases due to the significant dynamics mismatch. GARAT fails in almost all environments, which we

![](images/94bc1eaca0e0576d512e4e765de7dfa70d10056f7894aefd0d28bdb0398ac71c.jpg)  
Ours DARC GARAT IW Clip Oracle Zero-shot Finetune

Figure 4: Adaptation performance in the target domain with kinematic mismatch (Top) or morphology mismatch (Bottom). Solid curves are average returns over five runs with different random seeds, and shaded areas indicate one standard deviation. We use data ratio $\Gamma = 10$ , which indicates all algorithms perform $10^{6}$ online interactions with the source domain except Oracle.

believe is caused by the impractical action transformation from inaccurate dynamics models. The performance of Zero-shot suggests that the policies trained in the source domains barely work in the target domains due to dynamics mismatch. Finetune achieves promising results and outperforms our method in two of eight environments. We believe that the temporally-extended behaviors of the pre-trained policy benefit learning in the downstream tasks with the assistance of efficient exploration. Nonetheless, our method is the only one that outperforms or matches the asymptotic performance of Oracle in four out of eight environments.

# 6.2 Ablation Studies

To investigate the impact of design components in our method, we perform ablation analysis on the ratio of transitions $\Gamma$ , data selection ratio $\xi\%$ , and the optimistic exploration.

Data ratio $\Gamma$ . We employ different ratios of transitions from the source domain versus those from the target domain ( $\Gamma = 5, 10, 20$ ) for variants of our algorithm. The results shown in Figure 5 demonstrate that the performance of our algorithm improves with more source domain transitions when the number of target domain transitions is the same. This finding indicates that VGDF can fully exploit the reusable

![](images/ecc5a08de90979fe6feead5fbe4e8d0720ac6fdeb254efee53bc6c6652e055d3.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (x10^5) | Return (Pink Line) | Return (Yellow Line) | Return (Blue Line) |
| ------------------------------ | ------------------ | -------------------- | ------------------ |
| 0.0                            | 0                  | 0                    | 0                  |
| 0.2                            | ~3000              | ~1500                | ~1000              |
| 0.4                            | ~4500              | ~2500                | ~1500              |
| 0.6                            | ~5000              | ~3500                | ~2000              |
| 0.8                            | ~5000              | ~4000                | ~2500              |
| 1.0                            | ~5000              | ~4500                | ~3000              |
| 1.2                            | ~5000              | ~5000                | ~3500              |
</details>

![](images/a7a79054ed65fd8ab9ff6ea5ca94316136b447ec86b12be0bf644c62f22e4ff0.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (×10⁵) | Γ = 5   | Γ = 10  | Γ = 20  |
| ------------------------------ | ------- | ------- | ------- |
| 0.0                            | 0       | 0       | 0       |
| 0.2                            | ~500    | ~750    | ~1000   |
| 0.4                            | ~1500   | ~1750   | ~2500   |
| 0.6                            | ~2250   | ~2500   | ~3500   |
| 0.8                            | ~3000   | ~3250   | ~3750   |
| 1.0                            | ~3500   | ~3750   | ~4000   |
| 1.2                            | ~4000   | ~4000   | ~4250   |
</details>

Figure 5: Effect of transition ratio $\Gamma$ .

source domain transitions to enhance the training efficiency concerning the target domain.

![](images/2e6b181d26da1c259680a3a35cf3409c32a56d4c746c5d42145330e56a3a179e.jpg)

<details>
<summary>bar</summary>

| Category         | HalfCheetah 10% | HalfCheetah 25% | HalfCheetah 50% | Hopper 10% | Hopper 25% | Hopper 50% | Hopper 75% | Hopper Mix | Hopper Oracle |
| ---------------- | --------------- | --------------- | --------------- | ---------- | ---------- | ---------- | ---------- | ---------- | -------------- |
| Broken back thigh | 4.5             | 4.8             | 5.2             | 2.8        | 2.9        | 3.0        | 3.1        | 2.2        | 2.7            |
| No legs          | 3.0             | 3.5             | 4.0             | 2.5        | 2.6        | 2.7        | 2.8        | 0.8        | 2.4            |
| Broken joints    | -               | -               | -               | -          | -          | -          | -          | -          | -              |
| Big head         | -               | -               | -               | -          | -          | -          | -          | -          | -              |
</details>

Figure 6: Effect of data selection ratios $\xi\%$ .

![](images/afdbf6162088ea8a14531d0af81a91ee39443a7838415330e8e601845de9961d.jpg)  
Figure 7: Effect of the optimistic exploration technique (i.e., $\pi^{E}$ ).

Data selection ratio $\xi\%$ . We employ different data ratios (10%, 25%, 50%, 75%) for the variants of our algorithm. Furthermore, we propose a baseline algorithm $Mix$ that learns with all source domain samples without selection ( $\omega(s, a, s') \equiv 1$ in Eq. (7)). The results, shown in Figure 6, indicate

Table 1: Results in the offline source online target setting. We evaluate the algorithms via the performance of the learned policy in the target domain and report the mean and std of the results across five runs with different random seeds. 

<table><tr><td></td><td>Offline only</td><td>Symmetric sampling</td><td>H2O</td><td>VGDF + BC</td></tr><tr><td>halfcheetah - broken back thigh</td><td>1128 ± 156</td><td>2439 ± 390</td><td>5761 ±148</td><td>4834 ± 250</td></tr><tr><td>halfcheetah - no thighs</td><td>361 ± 39</td><td>2211 ± 77</td><td>3023 ± 77</td><td>3910 ±160</td></tr><tr><td>hopper - broken hips</td><td>155 ± 19</td><td>2607 ± 181</td><td>2435 ± 325</td><td>2785 ±75</td></tr><tr><td>hopper - short feet</td><td>399 ± 5</td><td>2144 ± 509</td><td>868 ± 73</td><td>3060 ±60</td></tr><tr><td>walker - broken right thigh</td><td>1453 ± 412</td><td>709 ± 128</td><td>3743 ±50</td><td>3000 ± 388</td></tr><tr><td>walker - no right thigh</td><td>975 ± 131</td><td>872 ± 301</td><td>2600 ± 355</td><td>3293 ±306</td></tr></table>

that our algorithm performs robustly under various ratios within a specific range (e.g., $\xi\% \leq 50\%$ ). Surprisingly, Mix performs exceptionally well in environments with kinematic mismatches but fails in scenarios with morphology shifts. We attribute this to the less significant dynamics shift induced by kinematic changes compared to morphology changes.

Optimistic data collection. To validate the effect of the optimistic exploration $\pi^{E}$ , we introduce a variant of our method without $\pi^{E}$ . The results are shown in Figure 7. Removing the optimistic exploration technique results in performance degradation in three out of four environments concerning the sample efficiency, validating the effectiveness of the exploration policy.

# 6.3 Performance under Offline Source with Online Target

In this subsection, we extend our method to the setting with a source domain offline dataset and limited online interactions with the target domain, investigating the performance of our method without online access to the source domain. We use the D4RL medium datasets $[20]$ of three environments (i.e., HalfCheetah, Walker, Hopper) for evaluation. We compare the proposed VGDF + BC (Section 5.2) with the following baselines: Offline only that directly transfers the offline learned policy via CQL $[35]$ to the target domain; Symmetric sampling $[3]$ that samples 50% of the data from the target domain replay buffer and the remaining 50% from the source domain offline dataset for each training step; H2O $[49]$ that penalizes the Q function learning on source domain transitions with the estimated dynamics gap via learned classifiers. All algorithms have limited interactions with the target domain to $10^{5}$ steps. The experimental details are shown in Appendix E.2. The results shown in Table 1 demonstrate that our method outperforms the other methods in four out of six environments, indicating that filtering the source domain data with the value consistency paradigm is effective in the offline-online setting.

# 6.4 Quantifying Dynamics Mismatch via Fictitious Value Proximity

Although the empirical results suggest that our method can adapt the policy in the face of various dynamics shifts, the degree of the dynamics mismatch can only be evaluated via the adaptation performance rather than be quantified directly. Here, we propose quantifying the dynamics shifts via the proposed Fictitious Value Proximity (FVP) (Section 5.1).

We approximate the FVP in Eq. (5) by calculating the average likelihood of a batch of samples from the source domain by $\mathbb{E}[\Lambda(s,a,s^{\prime})]\approx\frac{1}{B}\sum_{(s,a,s^{\prime})}\hat{\Lambda}(s,a,s^{\prime})$ . We show the approximated FVP in Ant environments with kinematic or morphology shifts in Figure 8. We observe a significant gap between the FVP values of the paired domains, which suggests the target domain with the morphology shifts is "closer" to the source domain than the target domain with the kinematic shifts with respect to the value difference. FVP measured by value differences quantifies the long-term effect on the expected return. Such a measurement can be regarded as a way to quantify the domain discrepancies.

![](images/86f32451b0c925ee6c31e31c0c8123ac7afa26054c427f7b427a292bf290e4c9.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (×10³) | Kinematic | Morphology |
| ------------------------------ | --------- | ---------- |
| 0.0                            | 0.00      | 0.15       |
| 0.2                            | 0.05      | 0.20       |
| 0.4                            | 0.10      | 0.25       |
| 0.6                            | 0.15      | 0.30       |
| 0.8                            | 0.20      | 0.35       |
| 1.0                            | 0.25      | 0.40       |
</details>

Figure 8: Quantification analysis of the approximated FVP in Ant environments.

# 7 Conclusion

This work addresses the online dynamics adaptation problem by proposing VGDF that selectively shares the source domain transitions from a value consistency paradigm. Starting from the motivation example, we reveal the limitation of the prior dynamics-based method. Then we introduce a novel value discrepancy perspective with theoretical analysis, motivated by the insight that paired transitions with consistent value targets can be regarded as equivalent for training. Practically, we propose VGDF and the variant for the offline source domain setting. Empirical studies demonstrate the effectiveness of our method under significant dynamics gaps, including kinematics shifts and morphology shifts.

Limitation and future directions. One limitation of our method is the reliance on the ensemble dynamics models. However, the recent work estimating the epistemic uncertainty with a single model $[19]$ could be applicable. Furthermore, value-aware model learning $[18]$ may improve our method by training dynamics models with accurate value predictions of the generated samples. Exploring the effectiveness of value consistency for generalizing across reward functions can be another direction for future research. Finally, validating the effectiveness of the data sharing method in the Sim2Real problem would contribute to the robotics community. The online interaction with the reality system could be risky, recent works $[6, 70]$ can be integrated for safe online interactions.

# Acknowledgments

This work is supported by the National Natural Science Foundation of China (Grant No.62306242), the National Key R&D Program of China (Grant No.2022ZD0160100), Shanghai Artificial Intelligence Laboratory, Shanghai Municipal Science and Technology Major Project (No.2021SHZDZX0103), Scientific Research Development Center in Higher Education Institutions by the Ministry of Education, China (No.2021ITA10013), Shanghai Engineering Research Center of AI and Robotics, Engineering Research Center of AI and Robotics, Ministry of Education, China. We would like to thank the anonymous reviewers for their valuable suggestions.

# References

[1] Ilge Akkaya, Marcin Andrychowicz, Maciek Chociej, Mateusz Litwin, Bob McGrew, Arthur Petron, Alex Paino, Matthias Plappert, Glenn Powell, Raphael Ribas, et al. Solving rubik's cube with a robot hand. arXiv preprint arXiv:1910.07113, 2019.   
[2] Chenjia Bai, Lingxiao Wang, Lei Han, Jianye Hao, Animesh Garg, Peng Liu, and Zhaoran Wang. Principled exploration via optimistic bootstrapping and backward induction. In International Conference on Machine Learning, pages 577–587. PMLR, 2021.   
[3] Philip J Ball, Laura Smith, Ilya Kostrikov, and Sergey Levine. Efficient online reinforcement learning with offline data. arXiv preprint arXiv:2302.02948, 2023.   
[4] Mohammadamin Barekatain, Ryo Yonetani, and Masashi Hamaya. Multipolar: multi-source policy aggregation for transfer reinforcement learning between diverse environmental dynamics. In Proceedings of the Twenty-Ninth International Conference on International Joint Conferences on Artificial Intelligence, pages 3108–3116, 2021.   
[5] Andre Barreto, Diana Borsa, John Quan, Tom Schaul, David Silver, Matteo Hessel, Daniel Mankowitz, Augustin Zidek, and Remi Munos. Transfer in deep reinforcement learning using successor features and generalised policy improvement. In International Conference on Machine Learning, pages 501–510. PMLR, 2018.   
[6] Homanga Bharadhwaj, Aviral Kumar, Nicholas Rhinehart, Sergey Levine, Florian Shkurti, and Animesh Garg. Conservative safety critics for exploration. In International Conference on Learning Representations, 2020.   
[7] Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym. arXiv preprint arXiv:1606.01540, 2016.   
[8] Yevgen Chebotar, Ankur Handa, Viktor Makoviychuk, Miles Macklin, Jan Issac, Nathan Ratliff, and Dieter Fox. Closing the sim-to-real loop: Adapting simulation randomization with real

world experience. In 2019 International Conference on Robotics and Automation (ICRA), pages 8973–8979. IEEE, 2019.   
[9] Ching-An Cheng, Andrey Kolobov, and Alekh Agarwal. Policy improvement via imitation of multiple oracles. Advances in Neural Information Processing Systems, 33:5587–5598, 2020.   
[10] Kurtland Chua, Roberto Calandra, Rowan McAllister, and Sergey Levine. Deep reinforcement learning in a handful of trials using probabilistic dynamics models. Advances in neural information processing systems, 31, 2018.   
[11] Kamil Ciosek, Quan Vuong, Robert Loftin, and Katja Hofmann. Better exploration with optimistic actor critic. Advances in Neural Information Processing Systems, 32, 2019.   
[12] Erwin Coumans and Yunfei Bai. Pybullet, a python module for physics simulation for games, robotics and machine learning. http://pybullet.org, 2016–2021.   
[13] Imre Csiszár and János Körner. Information theory: coding theorems for discrete memoryless systems. Cambridge University Press, 2011.   
[14] Siddharth Desai, Ishan Durugkar, Haresh Karnan, Garrett Warnell, Josiah Hanna, and Peter Stone. An imitation from observation approach to transfer learning with dynamics mismatch. Advances in Neural Information Processing Systems, 33:3917–3929, 2020.   
[15] Yuqing Du, Olivia Watkins, Trevor Darrell, Pieter Abbeel, and Deepak Pathak. Auto-tuned sim-to-real transfer. In 2021 IEEE International Conference on Robotics and Automation (ICRA), pages 1290–1296. IEEE, 2021.   
[16] Yan Duan, John Schulman, Xi Chen, Peter L Bartlett, Ilya Sutskever, and Pieter Abbeel. RL $^{2}$ : Fast reinforcement learning via slow reinforcement learning. arXiv preprint arXiv:1611.02779, 2016.   
[17] Benjamin Eysenbach, Shreyas Chaudhari, Swapnil Asawa, Sergey Levine, and Ruslan Salakhutdinov. Off-dynamics reinforcement learning: Training for transfer with domain classifiers. In International Conference on Learning Representations, 2020.   
[18] Amir-massoud Farahmand, Andre Barreto, and Daniel Nikovski. Value-aware loss function for model-based reinforcement learning. In Artificial Intelligence and Statistics, pages 1486–1494. PMLR, 2017.   
[19] Angelos Filos, Eszter Vértes, Zita Marinho, Gregory Farquhar, Diana Borsa, Abram Friesen, Feryal Behbahani, Tom Schaul, André Barreto, and Simon Osindero. Model-value inconsistency as a signal for epistemic uncertainty. arXiv preprint arXiv:2112.04153, 2021.   
[20] Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine. D4rl: Datasets for deep data-driven reinforcement learning. arXiv preprint arXiv:2004.07219, 2020.   
[21] Justin Fu, Katie Luo, and Sergey Levine. Learning robust rewards with adversarial inverse reinforcement learning. In International Conference on Learning Representations, 2018.   
[22] Scott Fujimoto and Shixiang Shane Gu. A minimalist approach to offline reinforcement learning. Advances in neural information processing systems, 34:20132–20145, 2021.   
[23] Scott Fujimoto, Herke Hoof, and David Meger. Addressing function approximation error in actor-critic methods. In International conference on machine learning, pages 1587–1596. PMLR, 2018.   
[24] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pages 1861–1870. PMLR, 2018.   
[25] Josiah P Hanna and Peter Stone. Grounded action transformation for robot learning in simulation. In Thirty-first AAAI conference on artificial intelligence, 2017.

[26] Nicklas Hansen, Rishabh Jangir, Yu Sun, Guillem Alenyà, Pieter Abbeel, Alexei A Efros, Lerrel Pinto, and Xiaolong Wang. Self-supervised policy adaptation during deployment. In International Conference on Learning Representations, 2020.   
[27] Donald Hejna, Lerrel Pinto, and Pieter Abbeel. Hierarchically decoupled imitation for morphological transfer. In International Conference on Machine Learning, pages 4159–4171. PMLR, 2020.   
[28] Jonathan Ho and Stefano Ermon. Generative adversarial imitation learning. Advances in neural information processing systems, 29, 2016.   
[29] David Isele and Akansel Cosgun. Selective experience replay for lifelong learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018.   
[30] Michael Janner, Justin Fu, Marvin Zhang, and Sergey Levine. When to trust your model: Model-based policy optimization. Advances in Neural Information Processing Systems, 32, 2019.   
[31] Rahul Kidambi, Aravind Rajeswaran, Praneeth Netrapalli, and Thorsten Joachims. Morel: Model-based offline reinforcement learning. Advances in neural information processing systems, 33:21810–21823, 2020.   
[32] Kuno Kim, Yihong Gu, Jiaming Song, Shengjia Zhao, and Stefano Ermon. Domain adaptive imitation learning. In International Conference on Machine Learning, pages 5286–5295. PMLR, 2020.   
[33] Jens Kober, J Andrew Bagnell, and Jan Peters. Reinforcement learning in robotics: A survey. The International Journal of Robotics Research, 32(11):1238–1274, 2013.   
[34] Wouter M Kouw and Marco Loog. A review of domain adaptation without target labels. IEEE transactions on pattern analysis and machine intelligence, 43(3):766–785, 2019.   
[35] Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. Conservative q-learning for offline reinforcement learning. Advances in Neural Information Processing Systems, 33:1179–1191, 2020.   
[36] Thanard Kurutach, Ignasi Clavera, Yan Duan, Aviv Tamar, and Pieter Abbeel. Model-ensemble trust-region policy optimization. In International Conference on Learning Representations, 2018.   
[37] Alessandro Lazaric. Transfer in reinforcement learning: a framework and a survey. In Reinforcement Learning, pages 143–173. Springer, 2012.   
[38] Joonho Lee, Jemin Hwangbo, Lorenz Wellhausen, Vladlen Koltun, and Marco Hutter. Learning quadrupedal locomotion over challenging terrain. Science robotics, 5(47):eabc5986, 2020.   
[39] Alexander Li, Lerrel Pinto, and Pieter Abbeel. Generalized hindsight for reinforcement learning. Advances in neural information processing systems, 33:7754–7767, 2020.   
[40] Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. In ICLR (Poster), 2016.   
[41] Fangchen Liu, Zhan Ling, Tongzhou Mu, and Hao Su. State alignment-based imitation learning. In International Conference on Learning Representations, 2019.   
[42] Jinxin Liu, Zhang Hongyin, and Donglin Wang. Dara: Dynamics-aware reward augmentation in offline reinforcement learning. In International Conference on Learning Representations, 2021.   
[43] Xingyu Liu, Deepak Pathak, and Kris M Kitani. Revolver: Continuous evolutionary models for robot-to-robot policy transfer. arXiv preprint arXiv:2202.05244, 2022.   
[44] Yuping Luo, Huazhe Xu, Yuanzhi Li, Yuandong Tian, Trevor Darrell, and Tengyu Ma. Algorithmic framework for model-based deep reinforcement learning with theoretical guarantees. In International Conference on Learning Representations, 2018.

[45] Bhairav Mehta, Manfred Diaz, Florian Golemo, Christopher J Pal, and Liam Paull. Active domain randomization. In Conference on Robot Learning, pages 1162–1176. PMLR, 2020.   
[46] Rémi Munos and Csaba Szepesvári. Finite-time bounds for fitted value iteration. Journal of Machine Learning Research, 9(5), 2008.   
[47] Fabio Muratore, Christian Eilers, Michael Gienger, and Jan Peters. Data-efficient domain randomization with bayesian optimization. IEEE Robotics and Automation Letters, 6(2):911–918, 2021.   
[48] Anusha Nagabandi, Ignasi Clavera, Simin Liu, Ronald S Fearing, Pieter Abbeel, Sergey Levine, and Chelsea Finn. Learning to adapt in dynamic, real-world environments through meta-reinforcement learning. In International Conference on Learning Representations, 2018.   
[49] Haoyi Niu, Shubham Sharma, Yiwen Qiu, Ming Li, Guyue Zhou, Jianming Hu, and Xianyuan Zhan. When to trust your simulator: Dynamics-aware hybrid offline-and-online reinforcement learning. arXiv preprint arXiv:2206.13464, 2022.   
[50] Georg Ostrovski, Pablo Samuel Castro, and Will Dabney. The difficulty of passive learning in deep reinforcement learning. Advances in Neural Information Processing Systems, 34:23283-23295, 2021.   
[51] Michael O'Connell, Guanya Shi, Xichen Shi, Kamyar Azizzadenesheli, Anima Anandkumar, Yisong Yue, and Soon-Jo Chung. Neural-fly enables rapid learning for agile flight in strong winds. Science Robotics, 7(66):eabm6597, 2022.   
[52] Emilio Parisotto, Lei Jimmy Ba, and Ruslan Salakhutdinov. Actor-mimic: Deep multitask and transfer reinforcement learning. In ICLR (Poster), 2016.   
[53] Xue Bin Peng, Marcin Andrychowicz, Wojciech Zaremba, and Pieter Abbeel. Sim-to-real transfer of robotic control with dynamics randomization. In 2018 IEEE international conference on robotics and automation (ICRA), pages 3803–3810. IEEE, 2018.   
[54] Aniruddh Raghu, Matthieu Komorowski, Leo Anthony Celi, Peter Szolovits, and Marzyeh Ghassemi. Continuous state-space models for optimal sepsis treatment: a deep reinforcement learning approach. In Machine Learning for Healthcare Conference, pages 147–163. PMLR, 2017.   
[55] Roberta Raileanu, Max Goldstein, Arthur Szlam, and Rob Fergus. Fast adaptation to new environments via policy-dynamics value functions. In Proceedings of the 37th International Conference on Machine Learning, pages 7920–7931, 2020.   
[56] Aravind Rajeswaran, Sarvjeet Ghotra, Balaraman Ravindran, and Sergey Levine. Epopt: Learning robust neural network policies using model ensembles. arXiv preprint arXiv:1610.01283, 2016.   
[57] Kate Rakelly, Aurick Zhou, Chelsea Finn, Sergey Levine, and Deirdre Quillen. Efficient off-policy meta-reinforcement learning via probabilistic context variables. In International conference on machine learning, pages 5331–5340. PMLR, 2019.   
[58] Fabio Ramos, Rafael Carvalhaes Possas, and Dieter Fox. Bayessim: adaptive domain randomization via probabilistic inference for robotics simulators. arXiv preprint arXiv:1906.01728, 2019.   
[59] Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, et al. Mastering atari, go, chess and shogi by planning with a learned model. Nature, 588(7839):604–609, 2020.   
[60] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.   
[61] Jian Shen, Han Zhao, Weinan Zhang, and Yong Yu. Model-based policy optimization with unsupervised model adaptation. Advances in Neural Information Processing Systems, 33:2823-2834, 2020.

[62] David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, et al. Mastering chess and shogi by self-play with a general reinforcement learning algorithm. arXiv preprint arXiv:1712.01815, 2017.   
[63] Richard S Sutton. Learning to predict by the methods of temporal differences. Machine learning, 3(1):9–44, 1988.   
[64] Richard S Sutton. Integrated architectures for learning, planning, and reacting based on approximating dynamic programming. In Machine learning proceedings 1990, pages 216–224. Elsevier, 1990.   
[65] Richard S Sutton. Dyna, an integrated architecture for learning, planning, and reacting. ACM Sigart Bulletin, 2(4):160–163, 1991.   
[66] Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT press, 2018.   
[67] Richard S Sutton, Csaba Szepesvári, Alborz Geramifard, and Michael Bowling. Dyna-style planning with linear function approximation and prioritized sweeping. In Proceedings of the Twenty-Fourth Conference on Uncertainty in Artificial Intelligence, pages 528–536, 2008.   
[68] Yunzhe Tao, Sahika Genc, Jonathan Chung, Tao Sun, and Sunil Mallya. Repaint: Knowledge transfer in deep reinforcement learning. In International Conference on Machine Learning, pages 10141–10152. PMLR, 2021.   
[69] Matthew E Taylor and Peter Stone. Transfer learning for reinforcement learning domains: A survey. Journal of Machine Learning Research, 10(7), 2009.   
[70] Garrett Thomas, Yuping Luo, and Tengyu Ma. Safe reinforcement learning by imagining the near future. Advances in Neural Information Processing Systems, 34:13859–13869, 2021.   
[71] Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ international conference on intelligent robots and systems, pages 5026–5033. IEEE, 2012.   
[72] Faraz Torabi, Garrett Warnell, and Peter Stone. Generative adversarial imitation from observation. arXiv preprint arXiv:1807.06158, 2018.   
[73] Christopher JCH Watkins and Peter Dayan. Q-learning. Machine learning, 8(3):279–292, 1992.   
[74] Annie Xie, Shagun Sodhani, Chelsea Finn, Joelle Pineau, and Amy Zhang. Robust policy learning over multiple uncertainty sets. arXiv preprint arXiv:2202.07013, 2022.   
[75] Tianhe Yu, Aviral Kumar, Yevgen Chebotar, Karol Hausman, Sergey Levine, and Chelsea Finn. Conservative data sharing for multi-task offline reinforcement learning. Advances in Neural Information Processing Systems, 34:11501–11516, 2021.   
[76] Wenhao Yu, C Karen Liu, and Greg Turk. Policy transfer with strategy optimization. In International Conference on Learning Representations, 2018.   
[77] Wenhao Yu, Jie Tan, C Karen Liu, and Greg Turk. Preparing for the unknown: Learning a universal policy with online system identification. arXiv preprint arXiv:1702.02453, 2017.   
[78] Grace Zhang, Linghan Zhong, Youngwoon Lee, and Joseph J Lim. Policy transfer across visual and dynamics domain gaps via iterative grounding. arXiv preprint arXiv:2107.00339, 2021.   
[79] Qiang Zhang, Tete Xiao, Alexei A Efros, Lerrel Pinto, and Xiaolong Wang. Learning cross-domain correspondence for control with dynamics cycle-consistency. In International Conference on Learning Representations, 2020.   
[80] Wenxuan Zhou, Lerrel Pinto, and Abhinav Gupta. Environment probing interaction policies. In International Conference on Learning Representations, 2018.   
[81] Luisa Zintgraf, Kyriacos Shiarlis, Maximilian Igl, Sebastian Schulze, Yarin Gal, Katja Hofmann, and Shimon Whiteson. Varibad: A very good method for bayes-adaptive deep rl via meta-learning. In International Conference on Learning Representations, 2019.

# A Algorithm Description

The pseudocode of VGDF is presented in Algorithm 1. We utilize SAC [24] as our backbone algorithm. We employ a fixed entropy temperature coefficient in all experiments, demonstrating sufficient empirical performance. The training of the dynamics model ensemble follows prior works [10, 30] with the MLE loss. The calculation of the Fictitious Value Proximity follows Eq. (6) proposed in Section 5.1. Furthermore, the pseudocode of VGDF + BC is presented in Algorithm 2. We introduce the value-normalized tradeoff between the behavior cloning loss and the policy gradient following the prior work [22].

Algorithm 1 Value-Guided Data Filtering (VGDF)   
Input: Source domain $\mathcal{M}_{src}$ , target domain $\mathcal{M}_{tar}$ , and transition ratio $\Gamma (= 10)$ (source vs. target). Initialization: Policy $\pi$ , exploration policy $\pi^{\mathrm{E}}$ , value functions $\{Q_{\theta_i}\}_{i=1,2}$ , replay buffers $\{D_{src}, D_{tar}\}$ , dynamics model ensemble $\{T_{\phi_i}\}_{i=1}^M$ , data selection ratio $\xi$ , batch size $B$ , entropy temperature coefficient $\lambda$ .

1: for $t = 1, 2, \ldots$ do
2: # Interact with the source domain
3: Sample transition $(s_{src}, a_{src}, r_{src}, s'_{src})$ using $\pi^{\mathrm{E}}$ in $\mathcal{M}_{src}$ 4: $D_{src} \leftarrow D_{src} \cup (s_{src}, a_{src}, r_{src}, s'_{src})$ 5: # Interact with the target domain
6: if $t \% \Gamma == 0$ then
7: Sample transition $(s_{tar}, a_{tar}, r_{tar}, s'_{tar})$ using $\pi$ in $\mathcal{M}_{tar}$ 8: $D_{tar} \leftarrow D_{tar} \cup (s_{tar}, a_{tar}, r_{tar}, s'_{tar})$ 9: end if
10: Optimize dynamics ensemble $\{T_{\phi_i}\}_{i=1}^M$ with $D_{tar}$ via Eq. (13)
11: Sample $b_{src} := \{(s, a, r, s')\}_{src}^B$ from $D_{src}$ 12: Sample $b_{tar} := \{(s, a, r, s')\}_{tar}^B$ from $D_{tar}$ 13: Obtain Fictitious Value Proximity (FVP) $\{\Lambda(s, a, s')\}^B$ via Eq. (6) for transitions in $b_{src}$ 14: Obtain FVP quantile $\Lambda_{\xi \%}$ of $\{\Lambda(s, a, s')\}^B$ 15: # Optimize value function with data filtering
16: $\theta_{i=1,2} \leftarrow \arg \min_{\theta_i} \frac{1}{2B} \sum_{b_{tar}} \left[ (Q_{\theta_i} - TQ_{\theta_i})^2 \right] + \frac{1}{[2B \cdot \xi \%]} \sum_{b_{src}} \left[ 1 (\Lambda(s, a, s') > \Lambda_{\xi \%})(Q_{\theta_i} - TQ_{\theta_i})^2 \right]$ 17: # Optimize policies
18: $\pi^{\mathrm{E}} \leftarrow \arg \max_{\pi^{\mathrm{E}}} \frac{1}{2B} \sum_{b_{tar} \cup b_{src}} \left[ \max \{Q_{\theta_1}(s, a), Q_{\theta_2}(s, a)\} |_{a \sim \pi^{\mathrm{E}}(\cdot |s)} + \lambda H[\pi^{\mathrm{E}}] \right]$ 19: $\pi^{-} \leftarrow \arg \max_{\pi^{-}} \frac{1}{2B} \sum_{b_{tar} \cup b_{src}} \left[ \min \{Q_{\theta_1}(s, a), Q_{\theta_2}(s, a)\} |_{a \sim \pi(\cdot |s)} + \lambda H[\pi] \right]$ 20: end for

# B Proofs of the Performance Guarantees

This section presents the proof of our main results. Specifically, we propose that the value discrepancy can be leveraged for the performance guarantee across different domains Lemma C.3. In Theorem B.1, we convert the performance bound induced by the value discrepancy into a novel form for the offline source domain setting.

Theorem B.1. (Performance bound controlled by dynamics discrepancy.) Denote the source domain and target domain with different dynamics as $M_{src}$ and $M_{tar}$ , respectively. We have the

Algorithm 2 Value-Guided Data Filtering + Behavior Cloning (VGDF + BC)

Input: Source domain offline dataset $D_{src}$ , target domain $M_{tar}$ , max interaction steps with the target domain $T_{max}$ , and transition ratio $\Gamma\left(:=\frac{|D_{src}|}{T_{max}}=10\right)$ (source vs. target).

Initialization: Policy $\pi$ , value functions $\{Q_{\theta_{i}}\}_{i=1,2}^{max}$ , target domain replay buffer $D_{tar}$ , dynamics model ensemble $\{T_{\phi_{i}}\}_{i=1}^{M}$ , data selection ratio $\xi$ , batch size B, entropy temperature coefficient $\lambda$ , train repeat K, behavior cloning constant $\alpha$ .

1: for $t = 1, 2, \ldots, T_{max}$ do
2: # Interact with the target domain
3: Sample transition ( $s_{tar}, a_{tar}, r_{tar}, s'_{tar}$ ) using $\pi$ in $M_{tar}$ 4: $D_{tar} \leftarrow D_{tar} \cup (s_{tar}, a_{tar}, r_{tar}, s'_{tar})$ 5: # Repeat training for K times per step
6: for $k = 1, 2, \ldots, K$ do
7: Optimize dynamics ensemble $\{T_{\phi_i}\}_{i=1}^M$ with $D_{tar}$ via Eq. (13)
8: Sample $b_{src} := \{(s, a, r, s')\}_{src}^B$ from $D_{src}$ 9: Sample $b_{tar} := \{(s, a, r, s')\}_{tar}^B$ from $D_{tar}$ 10: Obtain Fictitious Value Proximity (FVP) $\{\Lambda(s, a, s')\}^B$ via Eq. (6) for transitions in $b_{src}$ 11: Obtain FVP quantile $\Lambda_{\xi\%}$ of $\{\Lambda(s, a, s')\}^B$ 12: # Optimize value function with data filtering
13: $\theta_{i=1,2} \leftarrow \arg\min_{\theta_i} \frac{1}{2B} \sum_{b_{tar}} \left[ (Q_{\theta_i} - TQ_{\theta_i})^2 \right] + \frac{1}{\lfloor 2B \cdot \xi\% \rfloor} \sum_{b_{src}} \left[ 1 (\Lambda(s, a, s') > \Lambda_{\xi\%})(Q_{\theta_i} - TQ_{\theta_i})^2 \right]$ 14: # Optimize policy with behavior cloning regularization
15: $\beta = \alpha / \left\{ \frac{1}{2B} \sum_{b_{tar} \cup b_{src}} \left[ \left| \min\{Q_{\theta_1}(s, a), Q_{\theta_2}(s, a)\}_{a \sim \pi(\cdot | s)} \right| \right] \right\}$ 16: $\pi \leftarrow \arg\max_{\pi} \frac{\beta}{2B} \sum_{b_{tar} \cup b_{src}} \left[ \min\{Q_{\theta_1}(s, a), Q_{\theta_2}(s, a)\}_{a \sim \pi(\cdot | s)} + \lambda H[\pi] - \frac{1}{B} \sum_{(s, a) \sim b_{src}} \left[ (\pi(s) - a)^2 \right] \right]$ 17: end for
18: end for

performance difference of any policy $\pi$ evaluated under $M_{src}$ and $M_{tar}$ be bounded as below,

$$
\eta_ {\mathcal {M} _ {t a r}} (\pi) \geq \eta_ {\mathcal {M} _ {s r c}} (\pi) - \frac {2 \gamma r _ {\max}}{(1 - \gamma) ^ {2}} \cdot \mathbb {E} _ {\rho_ {s r c} ^ {\pi}} \left[ D _ {\mathrm{TV}} \left(P _ {s r c} (\cdot | s, a) \| P _ {t a r} (\cdot | s, a)\right) \right].
$$

Proof. We have

$$
\begin{array}{l} \eta_ {s r c} (\pi) - \eta_ {t a r} (\pi) = \frac {\gamma}{1 - \gamma} \mathbb {E} _ {\rho_ {s r c} ^ {\pi} (s, a)} \left[ \int_ {s ^ {\prime}} P _ {s r c} (s ^ {\prime} | s, a) V _ {t a r} ^ {\pi} (s ^ {\prime}) - \int_ {s ^ {\prime}} P _ {t a r} (s ^ {\prime} | s, a) V _ {t a r} ^ {\pi} (s ^ {\prime}) d s ^ {\prime} \right] (\text {Lemma C.1}) \\ = \frac {\gamma}{1 - \gamma} \mathbb {E} _ {\rho_ {s r c} ^ {\pi} (s, a)} \left[ \int_ {s ^ {\prime}} (P _ {s r c} (s ^ {\prime} | s, a) - P _ {t a r} (s ^ {\prime} | s, a)) V _ {t a r} ^ {\pi} (s ^ {\prime}) d s ^ {\prime} \right] \\ \leq \frac {\gamma}{1 - \gamma} \mathbb {E} _ {\rho_ {s r c} ^ {\pi} (s, a)} \left[ \int_ {s ^ {\prime}} | (P _ {s r c} (s ^ {\prime} | s, a) - P _ {t a r} (s ^ {\prime} | s, a)) V _ {t a r} ^ {\pi} (s ^ {\prime}) | d s ^ {\prime} \right] \\ \leq \frac {\gamma}{1 - \gamma} \cdot \frac {r _ {\max}}{1 - \gamma} \mathbb {E} _ {\rho_ {s r c} ^ {\pi} (s, a)} \left[ \int_ {s ^ {\prime}} | P _ {s r c} (s ^ {\prime} | s, a) - P _ {t a r} (s ^ {\prime} | s, a) | d s ^ {\prime} \right] \\ = \frac {2 \gamma r _ {\max}}{(1 - \gamma) ^ {2}} \mathbb {E} _ {\rho_ {s r c} ^ {\pi} (s, a)} \left[ D _ {\mathrm{TV}} \left(P _ {s r c} (\cdot | s, a) \| P _ {t a r} (\cdot | s, a)\right) \right]. \tag {9} \\ \end{array}
$$

□

Theorem B.2. (Performance bound controlled by value difference.) Denote the source domain and target domain as $M_{src}$ and $M_{tar}$ , respectively. We have the performance guarantee of any policy $\pi$ over the two MDPs:

$$
\eta_ {\mathcal {M} _ {t a r}} (\pi) \geq \eta_ {\mathcal {M} _ {s r c}} (\pi) - \frac {\gamma}{1 - \gamma} \cdot \mathbb {E} _ {\rho_ {\mathcal {M} _ {s r c}} ^ {\pi}} \Bigg [ \left| \mathbb {E} _ {P _ {s r c}} \left[ V _ {\mathcal {M} _ {t a r}} ^ {\pi} (s ^ {\prime}) \right] - \mathbb {E} _ {P _ {t a r}} \left[ V _ {\mathcal {M} _ {t a r}} ^ {\pi} (s ^ {\prime}) \right] \right| \Bigg ].
$$

Proof. We have

$$
\begin{array}{l} \eta_ {s r c} (\pi) - \eta_ {t a r} (\pi) = \frac {\gamma}{1 - \gamma} \mathbb {E} _ {\rho_ {s r c} ^ {\pi} (s, a)} \left[ \int_ {s ^ {\prime}} P _ {s r c} (s ^ {\prime} | s, a) V _ {\mathcal {M} _ {t a r}} ^ {\pi} (s ^ {\prime}) - \int_ {s ^ {\prime}} P _ {t a r} (s ^ {\prime} | s, a) V _ {\mathcal {M} _ {t a r}} ^ {\pi} (s ^ {\prime}) d s ^ {\prime} \right] (\text {Lemma C.1}) \\ = \frac {\gamma}{1 - \gamma} \cdot \mathbb {E} _ {\rho_ {\mathcal {M} _ {s r c}} ^ {\pi}} \left[ \mathbb {E} _ {P _ {s r c}} \left[ V _ {\mathcal {M} _ {t a r}} ^ {\pi} (s ^ {\prime}) \right] - \mathbb {E} _ {P _ {t a r}} \left[ V _ {\mathcal {M} _ {t a r}} ^ {\pi} (s ^ {\prime}) \right] \right] \\ \leq \frac {\gamma}{1 - \gamma} \cdot \mathbb {E} _ {\rho_ {\mathcal {M} _ {s r c}} ^ {\pi}} \left[ \left| \mathbb {E} _ {P _ {s r c}} \left[ V _ {\mathcal {M} _ {t a r}} ^ {\pi} (s ^ {\prime}) \right] - \mathbb {E} _ {P _ {t a r}} \left[ V _ {\mathcal {M} _ {t a r}} ^ {\pi} (s ^ {\prime}) \right] \right| \right] \\ \end{array}
$$

![](images/f4a0341c99912f8889099490ddae3c6ca5a1542a8fd41b6f6f9418bc89694343.jpg)

Theorem B.3. Under the setting with offline source domain dataset D whose empirical estimation of the data collection policy is $\pi_{D}(a|s):=\frac{\sum_{D}\mathbb{1}(s,a)}{\sum_{D}\mathbb{1}(s)}$ , let $M_{src}$ and $M_{tar}$ denote the source and target domain, respectively. We have the performance guarantee of any policy $\pi$ over the two MDPs:

$$
\eta_ {\mathcal {M} _ {t a r}} (\pi) \geq \eta_ {\mathcal {M} _ {s r c}} (\pi) - \frac {4 r _ {\max}}{(1 - \gamma) ^ {2}} \mathbb {E} _ {\rho_ {\mathcal {M} _ {s r c}} ^ {\pi_ {D}}, P _ {s r c}} [ D _ {T V} (\pi_ {D} | | \pi) ] - \frac {1}{1 - \gamma} \mathbb {E} _ {\rho_ {\mathcal {M} _ {s r c}} ^ {\pi_ {D}}} \left[ | \zeta (s, a) | \right], \tag {10}
$$

where $\zeta(s, a) := \mathbb{E}_{P_{src}, \pi} \left[ Q_{\mathcal{M}_{tar}}^{\pi}(s', a') \right] - \mathbb{E}_{P_{tar}, \pi} \left[ Q_{\mathcal{M}_{tar}}^{\pi}(s', a') \right]$ .

Proof. We have

$$
\eta_ {\mathcal {M} _ {t a r}} (\pi) - \eta_ {\mathcal {M} _ {s r c}} (\pi) = \underbrace {\left(\eta_ {\mathcal {M} _ {s r c}} (\pi_ {D}) - \eta_ {\mathcal {M} _ {s r c}} (\pi)\right)} _ {(a)} - \underbrace {\left(\eta_ {\mathcal {M} _ {s r c}} (\pi_ {D}) - \eta_ {\mathcal {M} _ {t a r}} (\pi)\right)} _ {(b)}.
$$

We have

$$
\begin{array}{l} \eta_{\mathcal{M}_{src}}(\pi_{D}) - \eta_{\mathcal{M}_{src}}(\pi)\geq -\frac{1}{1 - \gamma}\mathbb{E}_{\substack{s,a\sim \rho_{\mathcal{M}_{src}}^{\pi_{D}}\\ s^{\prime}\sim P_{src}(\cdot |s,a)}}\left[  \left|\mathbb{E}_{a^{\prime}\sim \pi_{D}(\cdot |s^{\prime})}\left[Q_{\mathcal{M}_{src}}^{\pi}(s^{\prime},a^{\prime})\right] - \mathbb{E}_{a^{\prime}\sim \pi (\cdot |s^{\prime})}\left[Q_{\mathcal{M}_{src}}^{\pi}(s^{\prime},a^{\prime})\right]\right| \right] \\ = -\frac{1}{1 - \gamma}\mathbb{E}_{\substack{s,a\sim \rho_{\mathcal{M}_{src}}^{\pi_{D}}\\ s^{\prime}\sim P_{src}(\cdot |s,a)}}\left[\left|\sum_{\mathcal{A}}\left(\pi_{D}(a^{\prime}|s^{\prime}) - \pi (a^{\prime}|s^{\prime})\right)Q_{\mathcal{M}_{src}}^{\pi}(s^{\prime},a^{\prime})\right|\right] \\ \geq -\frac{1}{1 - \gamma}\mathbb{E}_{\substack{s,a\sim \rho_{\mathcal{M}_{src}}^{\pi_{D}}\\ s^{\prime}\sim P_{src}(\cdot |s,a)}}\left[\left|\sum_{\mathcal{A}}\left(\pi_{D}(a^{\prime}|s^{\prime}) - \pi (a^{\prime}|s^{\prime})\right)\frac{r_{\max}}{1 - \gamma}\right|\right] \\ \geq -\frac{r_{\max}}{(1 - \gamma)^{2}}\mathbb{E}_{\substack{s,a\sim \rho_{\mathcal{M}_{src}}^{\pi_{D}}\\ s^{\prime}\sim P_{src}(\cdot |s,a)}}\left[\sum_{\mathcal{A}}|\pi_{D}(a^{\prime}|s^{\prime}) - \pi (a^{\prime}|s^{\prime})|\right] \\ = -\frac{2r_{\max}}{(1 - \gamma)^{2}}\mathbb{E}_{\substack{s,a\sim \rho_{\mathcal{M}_{src}}^{\pi_{D}}\\ s^{\prime}\sim P_{src}(\cdot |s,a)}}\left[D_{TV}\left(\pi_{D}(\cdot |s^{\prime})\parallel \pi (\cdot |s^{\prime})\right)\right], \\ \end{array}
$$

and

$$
\begin{array}{l} - \left(\eta_ {\mathcal {M} _ {s r c}} (\pi_ {D}) - \eta_ {\mathcal {M} _ {t a r}} (\pi)\right) \\ = - \frac {1}{1 - \gamma} \mathbb {E} _ {s, a \sim \rho_ {\mathcal {M} _ {s r c}} ^ {\pi_ {D}}} \left[ \mathcal {G} _ {\mathcal {M} _ {1}, \mathcal {M} _ {2}} ^ {\pi_ {1}, \pi_ {2}} (s, a) \right] \tag {LemmaC.2} \\ \geq -\frac{2r_{max}}{(1 - \gamma)^{2}}\mathbb{E}_{\substack{s,a\sim \rho_{\mathcal{M}_{src}}^{\pi_{D}}\\ s^{\prime}\sim P_{src}(\cdot |s,a)}}[D_{TV}(\pi_{D}(\cdot |s^{\prime})  \|   \pi (\cdot |s^{\prime}))] \\ - \frac {1}{1 - \gamma} \mathbb {E} _ {s, a \sim \rho_ {\mathcal {M} _ {s r c}} ^ {\pi D}} \left[ \left| \mathbb {E} _ {s ^ {\prime}, a ^ {\prime} \sim P _ {s r c}, \pi} \left[ Q _ {\mathcal {M} _ {t a r}} ^ {\pi} (s ^ {\prime}, a ^ {\prime}) \right] - \mathbb {E} _ {s ^ {\prime}, a ^ {\prime} \sim P _ {t a r}, \pi} \left[ Q _ {\mathcal {M} _ {t a r}} ^ {\pi} (s ^ {\prime}, a ^ {\prime}) \right] \right| \right].   (\text {Lemma C.3}) \\ \end{array}
$$

Combining the two inequalities above completes the proof.

# C Proofs of Lemmas

This section provides proof of several lemmas used for our theoretical results. The first lemma is adopted from $[44]$ , and the proof is essentially the same as the original paper. Lemma C.2 and Lemma C.3 support the derivation of the performance difference bound in Theorem B.3.

Lemma C.1. (Telescoping Lemma, Lemma 4.3 in [44].) Let $\mathcal{M}_1 := (\mathcal{S}, \mathcal{A}, P_1, r, \gamma)$ and $\mathcal{M}_2 := (\mathcal{S}, \mathcal{A}, P_2, r, \gamma)$ be two MDPs with different dynamics $P_1$ and $P_2$ . Given a policy $\pi$ , let

$$
\mathcal {G} _ {\mathcal {M} _ {1}, \mathcal {M} _ {2}} ^ {\pi} (s, a) := \mathbb {E} _ {s ^ {\prime} \sim P _ {1}} \left[ V _ {\mathcal {M} _ {2}} ^ {\pi} (s ^ {\prime}) \right] - \mathbb {E} _ {s ^ {\prime} \sim P _ {2}} \left[ V _ {\mathcal {M} _ {2}} ^ {\pi} (s ^ {\prime}) \right],
$$

we have

$$
\eta_ {\mathcal {M} _ {1}} (\pi) - \eta_ {\mathcal {M} _ {2}} (\pi) = \frac {\gamma}{(1 - \gamma)} \mathbb {E} _ {s, a \sim \rho_ {\mathcal {M} _ {1}} ^ {\pi}} \left[ \mathcal {G} _ {\mathcal {M} _ {1}, \mathcal {M} _ {2}} ^ {\pi} (s, a) \right].
$$

Proof. Define $W_{j}$ as the expected return when executing $\pi$ on $\mathcal{M}_1$ for the first $j$ steps, then switching to $\pi$ and $\mathcal{M}_2$ for the remainder. That is

$$
W_{j}:= \sum_{t = 0}^{\infty}\gamma^{t}\mathbb{E}_{\substack{t <   j:s_{t},a_{t}\sim P_{1},\pi \\ t\geq j:s_{t},a_{t}\sim P_{2},\pi_{2}}}[r(s_{t},a_{t})]\\ ] = \mathbb{E}_{\substack{t <   j:s_{t},a_{t}\sim P_{1},\pi \\ t\geq j:s_{t},a_{t}\sim P_{2},\pi}}\left[\sum_{t = 0}^{\infty}\gamma^{t}r(s_{t},a_{t})\right].
$$

Then we have

$$
W _ {0} = \mathbb {E} _ {s, a \sim \rho_ {\mathcal {M} _ {2}, \pi}} [ r (s _ {t}, a _ {t}) ] = \eta_ {\mathcal {M} _ {2}} (\pi),
$$

$$
\text { and } W _ {\infty} = \mathbb {E} _ {s, a \sim \rho_ {\mathcal {M} _ {1}, \pi}} [ r (s _ {t}, a _ {t}) ] = \eta_ {\mathcal {M} _ {1}} (\pi).
$$

Thus we can obtain

$$
\eta_ {\mathcal {M} _ {1}} (\pi) - \eta_ {\mathcal {M} _ {2}} (\pi) = \sum_ {j = 0} ^ {\infty} (W _ {j + 1} - W _ {j}). \tag {11}
$$

Convert $W_{j}$ and $W_{j + 1}$ as following:

$$
W _ {j} = R _ {j} + \mathbb {E} _ {s _ {j}, a _ {j} \sim P _ {1}, \pi} \left[ \mathbb {E} _ {s _ {j + 1} \sim P _ {2}} \left[ \gamma^ {j + 1} V _ {\mathcal {M} _ {2}} ^ {\pi} (s _ {j + 1}) \right] \right]
$$

$$
W _ {j + 1} = R _ {j} + \mathbb {E} _ {s _ {j}, a _ {j} \sim P _ {1}, \pi} \left[ \mathbb {E} _ {s _ {j + 1} \sim P _ {1}} \left[ \gamma^ {j + 1} V _ {\mathcal {M} _ {2}} ^ {\pi} (s _ {j + 1}) \right] \right]
$$

Plug back to Eq.11 and we obtain

$$
\begin{array}{l} \eta_ {\mathcal {M} _ {1}} (\pi) - \eta_ {\mathcal {M} _ {2}} (\pi) = \sum_ {j = 0} ^ {\infty} (W _ {j + 1} - W _ {j}) \\ = \sum_ {j = 0} ^ {\infty} \gamma^ {j + 1} \mathbb {E} _ {s, a \sim \mathbb {P} _ {\mathcal {M} _ {1}, j} ^ {\pi}} \left[ \mathbb {E} _ {s ^ {\prime} \sim P _ {1}} \left[ V _ {\mathcal {M} _ {2}} ^ {\pi} (s ^ {\prime}) \right] - \mathbb {E} _ {s ^ {\prime} \sim P _ {2}} \left[ V _ {\mathcal {M} _ {2}} ^ {\pi} (s ^ {\prime}) \right] \right] \\ = \frac {\gamma}{(1 - \gamma)} \mathbb {E} _ {s, a \sim \rho_ {\mathcal {M} _ {1}} ^ {\pi}} \left[ \mathbb {E} _ {s ^ {\prime} \sim P _ {1}} \left[ V _ {\mathcal {M} _ {2}} ^ {\pi} (s ^ {\prime}) \right] - \mathbb {E} _ {s ^ {\prime} \sim P _ {2}} \left[ V _ {\mathcal {M} _ {2}} ^ {\pi} (s ^ {\prime}) \right] \right] \\ = \frac {\gamma}{(1 - \gamma)} \mathbb {E} _ {s, a \sim \rho_ {\mathcal {M} _ {1}} ^ {\pi}} \left[ \mathcal {G} _ {\mathcal {M} _ {1}, \mathcal {M} _ {2}} ^ {\pi} (s, a) \right]. \\ \end{array}
$$

![](images/2f260e088a249de96aca7ac4bd6cf708cf4aeaa04ded66eb03f85967a26ae6e8.jpg)

Lemma C.2. (Extension of Telescoping Lemma.) Let $\mathcal{M}_{1} := (\mathcal{S}, \mathcal{A}, P_{1}, r, \gamma)$ and $\mathcal{M}_{2} := (\mathcal{S}, \mathcal{A}, P_{2}, r, \gamma)$ be two MDPs with different dynamics $P_{1}$ and $P_{2}$ . Given two policies $\pi_{1}, \pi_{2}$ , let

$$
\mathcal {G} _ {\mathcal {M} _ {1}, \mathcal {M} _ {2}} ^ {\pi_ {1}, \pi_ {2}} (s, a) := \mathbb {E} _ {s ^ {\prime}, a ^ {\prime} \sim P _ {1}, \pi_ {1}} \left[ Q _ {\mathcal {M} _ {2}} ^ {\pi_ {2}} (s ^ {\prime}, a ^ {\prime}) \right] - \mathbb {E} _ {s ^ {\prime}, a ^ {\prime} \sim P _ {2}, \pi_ {2}} \left[ Q _ {\mathcal {M} _ {2}} ^ {\pi_ {2}} (s ^ {\prime}, a ^ {\prime}) \right],
$$

we have

$$
\eta_ {\mathcal {M} _ {1}} (\pi_ {1}) - \eta_ {\mathcal {M} _ {2}} (\pi_ {2}) = \frac {1}{(1 - \gamma)} \mathbb {E} _ {s, a \sim \rho_ {\mathcal {M} _ {1}} ^ {\pi_ {1}}} \left[ \mathcal {G} _ {\mathcal {M} _ {1}, \mathcal {M} _ {2}} ^ {\pi_ {1}, \pi_ {2}} (s, a) \right].
$$

Proof. Define $W_{j}$ as the expected return when executing $\pi_1$ on $\mathcal{M}_1$ for the first $j$ steps, then switching to $\pi_2$ and $\mathcal{M}_2$ for the remainder. That is

$$
W_{j}:= \sum_{t = 0}^{\infty}\gamma^{t}\mathbb{E}_{\substack{t <   j:s_{t},a_{t}\sim P_{1},\pi_{1}\\ t\geq j:s_{t},a_{t}\sim P_{2},\pi_{2}}}[r(s_{t},a_{t})] = \mathbb{E}_{\substack{t <   j:s_{t},a_{t}\sim P_{1},\pi_{1}\\ t\geq j:s_{t},a_{t}\sim P_{2},\pi_{2}}}\left[\sum_{t = 0}^{\infty}\gamma^{t}r(s_{t},a_{t})\right].
$$

Then we have

$$
W _ {0} = \mathbb {E} _ {s, a \sim \rho_ {\mathcal {M} _ {2}, \pi_ {2}}} [ r (s _ {t}, a _ {t}) ] = \eta_ {\mathcal {M} _ {2}} (\pi_ {2}),
$$

$$
\text { and } W _ {\infty} = \mathbb {E} _ {s, a \sim \rho_ {\mathcal {M} _ {1}, \pi_ {1}}} [ r (s _ {t}, a _ {t}) ] = \eta_ {\mathcal {M} _ {2}} (\pi_ {1}).
$$

Thus we can obtain

$$
\eta_ {\mathcal {M} _ {1}} (\pi_ {1}) - \eta_ {\mathcal {M} _ {2}} (\pi_ {2}) = \sum_ {j = 0} ^ {\infty} (W _ {j + 1} - W _ {j}). \tag {12}
$$

Convert $W_{j}$ and $W_{j + 1}$ as following:

$$
W _ {j} = R _ {j} + \mathbb {E} _ {s _ {j}, a _ {j} \sim P _ {1}, \pi_ {1}} \left[ \mathbb {E} _ {s _ {j + 1}, a _ {j + 1} \sim P _ {2}, \pi_ {2}} \left[ \gamma^ {j + 1} Q _ {\mathcal {M} _ {2}} ^ {\pi_ {2}} (s _ {j + 1}, a _ {j + 1}) \right] \right]
$$

$$
W _ {j + 1} = R _ {j} + \mathbb {E} _ {s _ {j}, a _ {j} \sim P _ {1}, \pi_ {1}} \left[ \mathbb {E} _ {s _ {j + 1}, a _ {j + 1} \sim P _ {1}, \pi_ {1}} \left[ \gamma^ {j + 1} Q _ {\mathcal {M} _ {2}} ^ {\pi_ {2}} (s _ {j + 1}, a _ {j + 1}) \right] \right]
$$

Plug back to Eq.12 and we obtain

$$
\eta_ {\mathcal {M} _ {1}} (\pi_ {1}) - \eta_ {\mathcal {M} _ {2}} (\pi_ {2}) = \sum_ {j = 0} ^ {\infty} (W _ {j + 1} - W _ {j})
$$

$$
= \sum_ {j = 0} ^ {\infty} \gamma^ {j + 1} \mathbb {E} _ {s, a \sim \mathbb {P} _ {\mathcal {M} _ {1}, j} ^ {\pi_ {1}}} \left[ \mathbb {E} _ {s ^ {\prime}, a ^ {\prime} \sim P _ {1}, \pi_ {1}} \left[ Q _ {\mathcal {M} _ {2}} ^ {\pi_ {2}} (s ^ {\prime}, a ^ {\prime}) \right] - \mathbb {E} _ {s ^ {\prime}, a ^ {\prime} \sim P _ {2}, \pi_ {2}} \left[ Q _ {\mathcal {M} _ {2}} ^ {\pi_ {2}} (s ^ {\prime}, a ^ {\prime}) \right] \right]
$$

$$
= \frac {\gamma}{(1 - \gamma)} \mathbb {E} _ {s, a \sim \rho_ {\mathcal {M} _ {1}} ^ {\pi_ {1}}} \left[ \mathbb {E} _ {s ^ {\prime}, a ^ {\prime} \sim P _ {1}, \pi_ {1}} \left[ Q _ {\mathcal {M} _ {2}} ^ {\pi_ {2}} (s ^ {\prime}, a ^ {\prime}) \right] - \mathbb {E} _ {s ^ {\prime}, a ^ {\prime} \sim P _ {2}, \pi_ {2}} \left[ Q _ {\mathcal {M} _ {2}} ^ {\pi_ {2}} (s ^ {\prime}, a ^ {\prime}) \right] \right]
$$

$$
= \frac {\gamma}{(1 - \gamma)} \mathbb {E} _ {s, a \sim \rho_ {\mathcal {M} _ {1}} ^ {\pi_ {1}}} \left[ \mathcal {G} _ {\mathcal {M} _ {1}, \mathcal {M} _ {2}} ^ {\pi_ {1}, \pi_ {2}} (s, a) \right].
$$

Lemma C.3. (Bound of $\mathcal{G}_{\mathcal{M}_1,\mathcal{M}_2}^{\pi_1,\pi_2}(s,a)$ .) Let

$$
\mathcal {G} _ {\mathcal {M} _ {1}, \mathcal {M} _ {2}} ^ {\pi_ {1}, \pi_ {2}} (s, a) := \mathbb {E} _ {s ^ {\prime}, a ^ {\prime} \sim P _ {1}, \pi_ {1}} \left[ Q _ {\mathcal {M} _ {2}} ^ {\pi_ {2}} (s ^ {\prime}, a ^ {\prime}) \right] - \mathbb {E} _ {s ^ {\prime}, a ^ {\prime} \sim P _ {2}, \pi_ {2}} \left[ Q _ {\mathcal {M} _ {2}} ^ {\pi_ {2}} (s ^ {\prime}, a ^ {\prime}) \right],
$$

we have

$$
\begin{array}{l} \mathcal {G} _ {\mathcal {M} _ {1}, \mathcal {M} _ {2}} ^ {\pi_ {1}, \pi_ {2}} (s, a) \leq \frac {2 r _ {\max}}{1 - \gamma} \mathbb {E} _ {s ^ {\prime} \sim P _ {1}} \left[ D _ {T V} (\pi_ {1} (\cdot | s ^ {\prime}) \| \pi_ {2} (\cdot | s ^ {\prime})) \right] \\ + \left| \mathbb {E} _ {s ^ {\prime}, a ^ {\prime} \sim P _ {1}, \pi_ {2}} \left[ Q _ {\mathcal {M} _ {2}} ^ {\pi_ {2}} (s ^ {\prime}, a ^ {\prime}) \right] - \mathbb {E} _ {s ^ {\prime}, a ^ {\prime} \sim P _ {2}, \pi_ {2}} \left[ Q _ {\mathcal {M} _ {2}} ^ {\pi_ {2}} (s ^ {\prime}, a ^ {\prime}) \right] \right|. \\ \end{array}
$$

Proof. We have

$$
\begin{array}{l} \mathcal {G} _ {\mathcal {M} _ {1}, \mathcal {M} _ {2}} ^ {\pi_ {1}, \pi_ {2}} (s, a) := \mathbb {E} _ {s ^ {\prime}, a ^ {\prime} \sim P _ {1}, \pi_ {1}} \left[ Q _ {\mathcal {M} _ {2}} ^ {\pi_ {2}} (s ^ {\prime}, a ^ {\prime}) \right] - \mathbb {E} _ {s ^ {\prime}, a ^ {\prime} \sim P _ {2}, \pi_ {2}} \left[ Q _ {\mathcal {M} _ {2}} ^ {\pi_ {2}} (s ^ {\prime}, a ^ {\prime}) \right] \\ = \underbrace {\mathbb {E} _ {s ^ {\prime} , a ^ {\prime} \sim P _ {1} , \pi_ {1}} \left[ Q _ {\mathcal {M} _ {2}} ^ {\pi_ {2}} (s ^ {\prime} , a ^ {\prime}) \right] - \mathbb {E} _ {s ^ {\prime} , a ^ {\prime} \sim P _ {1} , \pi_ {2}} \left[ Q _ {\mathcal {M} _ {2}} ^ {\pi_ {2}} (s ^ {\prime} , a ^ {\prime}) \right]} _ {(a)} \\ + \underbrace {\mathbb {E} _ {s ^ {\prime} , a ^ {\prime} \sim P _ {1} , \pi_ {2}} \left[ Q _ {\mathcal {M} _ {2}} ^ {\pi_ {2}} (s ^ {\prime} , a ^ {\prime}) \right] - \mathbb {E} _ {s ^ {\prime} , a ^ {\prime} \sim P _ {2} , \pi_ {2}} \left[ Q _ {\mathcal {M} _ {2}} ^ {\pi_ {2}} (s ^ {\prime} , a ^ {\prime}) \right]} _ {(b)}. \\ \end{array}
$$

For $(a)$ , we have

$$
\begin{array}{l} (a) = \mathbb {E} _ {s ^ {\prime} \sim P _ {1}} \left[ \sum_ {a ^ {\prime}} \pi_ {1} (a ^ {\prime} | s ^ {\prime}) Q _ {\mathcal {M} _ {2}} ^ {\pi_ {2}} (s ^ {\prime}, a ^ {\prime}) - \pi_ {2} (a ^ {\prime} | s ^ {\prime}) Q _ {\mathcal {M} _ {2}} ^ {\pi_ {2}} (s ^ {\prime}, a ^ {\prime}) \right] \\ \leq \mathbb {E} _ {s ^ {\prime} \sim P _ {1}} \left[ \sum_ {a ^ {\prime}} | \pi_ {1} (a ^ {\prime} | s ^ {\prime}) - \pi_ {2} (a ^ {\prime} | s ^ {\prime}) | \frac {r _ {\max}}{1 - \gamma} \right] \\ = \frac {r _ {\max}}{1 - \gamma} \mathbb {E} _ {s ^ {\prime} \sim P _ {1}} \left[ \sum_ {a ^ {\prime}} \left| \pi_ {1} \left(a ^ {\prime} \mid s ^ {\prime}\right) - \pi_ {2} \left(a ^ {\prime} \mid s ^ {\prime}\right) \right| \right] \\ = \frac {2 r _ {\max}}{1 - \gamma} \mathbb {E} _ {s ^ {\prime} \sim P _ {1}} \left[ D _ {T V} \left(\pi_ {1} (\cdot | s ^ {\prime}) \parallel \pi_ {2} (\cdot | s ^ {\prime})\right) \right]. \\ \end{array}
$$

For $(b)$ , we have

$$
\begin{array}{l} (b) = \mathbb {E} _ {s ^ {\prime}, a ^ {\prime} \sim P _ {1}, \pi_ {2}} \left[ Q _ {\mathcal {M} _ {2}} ^ {\pi_ {2}} (s ^ {\prime}, a ^ {\prime}) \right] - \mathbb {E} _ {s ^ {\prime}, a ^ {\prime} \sim P _ {2}, \pi_ {2}} \left[ Q _ {\mathcal {M} _ {2}} ^ {\pi_ {2}} (s ^ {\prime}, a ^ {\prime}) \right] \\ \leq \left| \mathbb {E} _ {s ^ {\prime}, a ^ {\prime} \sim P _ {1}, \pi_ {2}} \left[ Q _ {\mathcal {M} _ {2}} ^ {\pi_ {2}} (s ^ {\prime}, a ^ {\prime}) \right] - \mathbb {E} _ {s ^ {\prime}, a ^ {\prime} \sim P _ {2}, \pi_ {2}} \left[ Q _ {\mathcal {M} _ {2}} ^ {\pi_ {2}} (s ^ {\prime}, a ^ {\prime}) \right] \right|. \\ \end{array}
$$

Adding these two bounds together yields the desired result.

![](images/7a4ba4c7e815f8529c237607322464bd9e04686269028ad715a82fff59da9e7f.jpg)

# D Detailed Environment Setting

# D.1 Grid World

In the grid world environment, the agent obtains the X-Y coordination as the state and executes one of the four actions (Up, Down, Left, Right) at each time step. A non-zero reward 1.0 is provided only if the agent reaches the goal. Each episode terminates when the agent reaches the goal or the episode length of 256 is reached. The source domain and the target domain of the grid world are shown in Figure 9. For each algorithm, the agent interacts with the source and target domains for $5e^{5}$ and $5e^{4}$ steps, respectively.

![](images/45008b693a09b8148fda96d47daddeacc3bb76fbc856e1b367baacf0a7852ab1.jpg)

<details>
<summary>natural_image</summary>

Simple geometric shape with a red dot and a green square, no text or symbols present.
</details>

![](images/d4e4b695c19286c0a70989ada2294891b8c0df09c0a7c0433104e9eecf710ddc.jpg)

<details>
<summary>natural_image</summary>

Abstract geometric shape with a gray L-shaped path and a green square, no text or symbols present.
</details>

Figure 9: The source domain (Left) and the target domain (Right) of the grid world environments.

# D.2 Mujoco Environments

To investigate the performance of the algorithm thoroughly, we design eight environments based on four Mujoco $[71]$ benchmarks from Gym $[7]$ including HalfCheetah-v2, Ant-v4, Walker2D-v2, and Hopper-v2. For each benchmark, we propose two variants with kinematic shift or morphology shift. We run all experiments with the original environment as the source domain and the variation environment as the target domain. Detailed modifications of the environments are shown below, and the illustration of the environments is shown in Figure 10. For algorithms that access interactions with both domains, the agent interacts with the source and target domains for $10^{6}$ and $10^{5}$ steps, respectively.

Detailed modifications of the environments with kinematic shifts are shown below:

HalfCheetah - broken back thigh: We modify the rotation range of the joint on the thigh of the back leg from $[-0.52, 1.05]$ to $[-0.0052, 0.0105]$ .

![](images/589ef696fd0c6db1bf88109dbfd3642a6df99391518d7c2cde9c07cd953f685e.jpg)

<details>
<summary>text_image</summary>

Source Domains
Target Domains with Kinematic shifts
Target Domains with Morphology shifts
</details>

Figure 10: Illustration of all environments, including all source domains (Top), all target domains with kinematic shifts (Middle), and all target domains with morphology shifts (Bottom).

Ant - broken hips: We modify the rotation range of the joints on the hip of leg 1 and leg 2 from $[-30, 30]$ to $[-0.3, 0.3]$ .

Walker - broken right foot: We modify the rotation range of the joint on the foot of the right leg from $[-45, 45]$ to $[-0.45, 0.45]$ .

Hopper - broken joints: We modify the rotation range of the joint on the head from $[-150, 0]$ to $[-0.15, 0]$ and the joint on foot from $[-45, 45]$ to $[-18, 18]$ .

Detailed modifications of the environments with morphology shifts are shown below:

HalfCheetah - no thighs: We modify the size of both thighs. Detailed modifications of the xml file are:

```txt
<geom fromto="0 0 0 -0.0001 0 -0.0001" name="bthigh" size="0.046" type="capsule"/>
<body name="bshin" pos="-0.0001 0 -0.0001"> 
```

Ant - short feet: We modify the size of feet on leg 1 and leg 2. Detailed modifications of the xml file are:

```xml
<geom fromto="0.0 0.0 0.0 0.1 0.1 0.0" name="left_ankle_geom" size="0.08" type="capsule"/>
<geom fromto="0.0 0.0 0.0 -0.1 0.1 0.0" name="right_ankle_geom" size="0.08" type="capsule"/> 
```

Walker - no right thigh: We modify the size of thigh on the right leg. Detailed modifications of the xml file are:

```xml
<body name="thigh" pos="0 0 1.05">
    <joint axis="0 -1 0" name="thigh_joint" pos="0 0 1.05" range="-150 0" type="hinge"/>
    <geom friction="0.9" fromto="0 0 1.05 0 0 1.045" name="thigh_geom" size="0.05" type="capsule"/>
    <body name="leg" pos="0 0 0.35">
    <joint axis="0 -1 0" name="leg_joint" pos="0 0 1.045" range="-150 0" type="hinge"/>
    <geom friction="0.9" fromto="0 0 1.045 0 0 0.3" name="leg_geom" size="0.04" type="capsule"/>
    <body name="foot" pos="0.2 0 0"> 
```

```xml
<joint axis="0 -1 0" name="foot_joint" pos="0 0 0.3" range=" -45 45" type="hinge"/>
<geom friction="0.9" fromto="-0.0 0 0.3 0.2 0 0.3" name=" foot_geom" size="0.06" type="capsule"/>
</body>
</body>
</body> 
```

Hopper - big head: We modify the size of the head. Detailed modifications of the xml file are:

```xml
<geom friction="0.9" fromto="0 0 1.45 0 0 1.05" name="torso_geom" size="0.125" type="capsule"/> 
```

# E Algorithms and Implementation Details

# E.1 Implementation Details

The details of our algorithm and baseline methods are specified as follows:

SAC: We first specify the implementation of the shared backbone algorithm SAC utilized in all algorithms. The policy and the value function are two-layer MLP with 256 hidden units using ReLU activation. The learning rate is $3e^{-4}$ . Discount $\gamma$ is set as 0.99 in all environments. The temperature coefficient is fixed as 0.2. The batch size is 128. The smoothing coefficient of the target networks is 0.005. The training delay of the policy is set as 2. The replay buffer size is $1e^{6}$ .

VGDF: We use a five-layer MLP with 200 units as the dynamics model using Swish activation following prior works [10, 30]. The ensemble size is 7. We set the data selection ratio $\xi\%$ as $25\%$ in the experiments shown in Section 6.1. For each probabilistic dynamics model $T_{\phi_i}(s_{t+1}, r_t | s_t, a_t) = \mathcal{N}(\mu_{\phi_i}(s_t, a_t), \Sigma_{\phi_i}(s_t, a_t))$ , $i = 1, \ldots, M$ , we train the model by maximizing the objective:

$$
J (\phi_ {i}) := \mathbb {E} _ {(s _ {t}, a _ {t}, r _ {t}, s _ {t + 1}) \sim D _ {t a r}} \left[ \left[ \mu_ {\phi_ {i}} (s _ {t}, a _ {t}) - \right. \right.
$$

$$
\left. \left(s _ {t + 1}, r _ {t}\right) \right] ^ {\top} \Sigma_ {\phi_ {i}} ^ {- 1} (s _ {t}, a _ {t}) \left[ \mu_ {\phi_ {i}} (s _ {t}, a _ {t}) - (s _ {t + 1}, r _ {t}) \right] + \log \det \Sigma_ {\phi_ {i}} (s _ {t}, a _ {t}) ]. \tag {13}
$$

The exploration policy is a two-layer MLP with 256 hidden units. We warm-start the algorithm by utilizing samples from both domains without selection for the first 1e5 steps in the source domain.

DARC: We follow the default configurations of the public implementation (https://github.com/google-research/google-research/tree/master/darc). The domain classifiers $q_{\psi_{SAS}}(s_t, a_t, s_{t+1})$ , $q_{\psi_{SA}}(s_t, a_t)$ are trained by maximizing the cross-entropy losses:

$$
\begin{array}{l} J (\psi_ {S A S}) := \mathbb {E} _ {(s _ {t}, a _ {t}, s _ {t + 1}) \sim D _ {t a r}} [ \log q _ {\psi_ {S A S}} (t a r | s _ {t}, a _ {t}, s _ {t + 1}) ] \\ + \mathbb {E} _ {(s _ {t}, a _ {t}, s _ {t + 1}) \sim D _ {s r c}} \left[ \log (1 - q _ {\psi_ {S A S}} (t a r | s _ {t}, a _ {t}, s _ {t + 1})) \right], \\ J (\psi_ {S A}) := \mathbb {E} _ {(s _ {t}, a _ {t}) \sim D _ {t a r}} \left[ \log q _ {\psi_ {S A}} (t a r | s _ {t}, a _ {t}) \right] + \mathbb {E} _ {(s _ {t}, a _ {t}) \sim D _ {s r c}} \left[ \log (1 - q _ {\psi_ {S A}} (t a r | s _ {t}, a _ {t})) \right]. \\ \end{array}
$$

Following the original implementation, we use the standard Gaussian noise for the domain classifier training. During training, a reward correction $\Delta r(s_{t}, a_{t})$ is augmented to the original reward $r(s_{t}, a_{t})$ of each source domain transition, i.e. $\tilde{r}(s_{t}, a_{t}) := r(s_{t}, a_{t}) + \Delta r(s_{t}, a_{t})$ . The reward correction is calculated by:

$$
\Delta r (s _ {t}, a _ {t}) := \log \frac {q _ {\psi_ {S A S}} (t a r | s , a , s ^ {\prime})}{q _ {\psi_ {S A S}} (s r c | s , a , s ^ {\prime})} \frac {q _ {\psi_ {S A}} (s r c | s , a)}{q _ {\psi_ {S A}} (t a r | s , a)}.
$$

We warm-start the algorithm by training with samples from both domains for the first $10^{5}$ steps following the original implementation.

GARAT: We use the author implementation with default configurations (Supplemental in https://proceedings.neurips.cc/paper/2020/hash/28f248e9279ac845995c4e9f8af35c2b-Abstract.html). We add the XML files of our customized environments to rl\_gat/envs/assets/ folder. We limit the extra interactions with the grounded source environments as $10^{5}$ for fair comparisons with other algorithms.

Table 2: Hyperparameters. "-" denotes the hyperparameter is not used in the algorithm. "←" denotes the same choice as the algorithm in the first column. 

<table><tr><td>Hyperparameters</td><td>VGDF</td><td>DARC</td><td>GARAT</td><td>IW Clip</td><td>Finetune</td></tr><tr><td>Hidden layers (Policy)</td><td>2</td><td>←</td><td>←</td><td>←</td><td>←</td></tr><tr><td>Hidden units per layer (Policy)</td><td>256</td><td>←</td><td>←</td><td>←</td><td>←</td></tr><tr><td>Hidden layers (Value)</td><td>2</td><td>←</td><td>←</td><td>←</td><td>←</td></tr><tr><td>Hidden units per layer (Value)</td><td>256</td><td>←</td><td>←</td><td>←</td><td>←</td></tr><tr><td>Hidden layers (Classifier)</td><td>-</td><td>2</td><td>-</td><td>2</td><td>-</td></tr><tr><td>Hidden units per layer (Classifier)</td><td>-</td><td>256</td><td>-</td><td>256</td><td>-</td></tr><tr><td>Hidden layers (Dynamics model)</td><td>5</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Hidden units per layer (Dynamics model)</td><td>200</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Ensemble size</td><td>7</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Learning rate</td><td> $3e^{-4}$ </td><td>←</td><td>←</td><td>←</td><td>←</td></tr><tr><td>Batch size</td><td>128</td><td>←</td><td>←</td><td>←</td><td>←</td></tr><tr><td>Fixed temperature coefficient</td><td>0.2</td><td>←</td><td>←</td><td>←</td><td>←</td></tr><tr><td>Target smoothing coefficient</td><td>0.005</td><td>←</td><td>←</td><td>←</td><td>←</td></tr><tr><td>Policy training delay</td><td>2</td><td>←</td><td>←</td><td>←</td><td>←</td></tr><tr><td>Buffer size</td><td> $1e^6$ </td><td>←</td><td>←</td><td>←</td><td>←</td></tr><tr><td>Data selection ratio  $\xi \%$ </td><td>25%</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Warm-start steps</td><td> $1e^5$ </td><td> $1e^5$ </td><td>-</td><td> $1e^5$ </td><td>-</td></tr><tr><td>Importance weight clipping range</td><td>-</td><td>-</td><td>-</td><td> $[1e^{-4}, 1]$ </td><td>-</td></tr><tr><td>Interactions with grounded src environment</td><td>-</td><td>-</td><td> $1e^5$ </td><td>-</td><td>-</td></tr></table>

Importance Weighting Clip (IW Clip): We use the domain classifiers same as DARC to calculate the importance weight $w(s, a, s')$ . The importance weighting is calculated by:

$$
w (s, a, s ^ {\prime}) := \frac {P _ {t a r} (s ^ {\prime} | s , a)}{P _ {s r c} (s ^ {\prime} | s , a)} \approx \frac {q _ {\psi_ {S A S}} (t a r | s , a , s ^ {\prime})}{q _ {\psi_ {S A S}} (s r c | s , a , s ^ {\prime})} \frac {q _ {\psi_ {S A}} (s r c | s , a)}{q _ {\psi_ {S A}} (t a r | s , a)},
$$

where $q_{\psi_{SAS}}$ and $q_{\psi_{SA}}$ are the domain classifiers proposed in [17]. We use the importance weighing to reweight the value training with source domain samples. Specifically,

$$
\theta \leftarrow \arg \min _ {\theta} \frac {1}{2} \mathbb {E} _ {(s, a, r, s ^ {\prime}) \sim D _ {s r c}} \left[ w (s, a, s ^ {\prime}) (Q _ {\theta} - \mathcal {T} Q _ {\theta}) ^ {2} \right].
$$

To stabilize training, we clip the importance weight between $[1e^{-4}, 1]$ , same as the prior work [49].

Finetune: We first train a policy in the source domain with $10^{6}$ steps. Then we transfer the policy to the target domain and further train the policy for $10^{5}$ steps.

The detailed hyperparameters of all algorithms are listed in Table. 2, and we use the same hyperparameters across all environments.

# E.2 Implementation Details of the Offline-Online Experiments

To evaluate the performance of our algorithm in the offline source online target setting, we use medium datasets from D4RL $[20]$ for three environments (i.e., HalfCheetah, Hopper, Walker). We use the same source domain offline dataset for each environment's two different target domains. For the algorithms performing online learning using offline data (i.e., Symmetric sampling, H2O, VGDF + BC), we perform the online interactions with the target domain for $10^{5}$ steps and use $10^{6}$ source domain transitions, the training is repeated for 10 times per step in the target domain. The details of the methods are specified as follows:

Offline only: We directly transfer the policy learned through CQL $[35]$ with the source domain offline dataset. For the CQL implementation, we follow the suggested configurations in a public CQL implementation (https://github.com/tinkoff-ai/CORL). We perform training for $10^{6}$ steps with the offline dataset and report the zero-shot performance of the learned policy in the target domain.

Symmetric sampling [3]: We perform the value function training by combining CQL optimization (with offline transitions) and SAC optimization (with online transitions). For each training step, we

sample 50% of the data from the target domain replay buffer and the remaining 50% from the source domain offline dataset. The CQL and SAC loss is computed with the corresponding transitions.

H2O [49]: We follow the original implementation that learns the classifiers to estimate the dynamics discrepancy across domains and perform the clipped importance weighting on the CQL loss on the source domain data. Same as Symmetric sampling, we repeat the training for 10 times per step in the target domain.

VGDF + BC: We adapt VGDF to the Offline-Online setting by simply integrating the behavior cloning loss following $(10)$ . The training is repeated for 10 times per step with the target domain the same as the baseline methods. For the trade-off between the policy gradient and behavior cloning, we use the value-normalized regularization following the $TD3 + BC$ $[22]$ work and set the constant $\alpha$ as 5. Furthermore, we remove the exploration policy proposed in Section 5.1 since the online access to the source domain is no longer available in the offline-online setting.

# F Additional Experiment Results

# F.1 Quantifying Dynamics Shifts via FVP

In this section, we investigate whether the estimation of the value differences can quantify the difference across domains. Specifically, in different target domains of the same source domain, we demonstrate the estimation of FVP in two target domains. As the results show in Figure 11, the FVP differs in environments with different dynamics shifts (Kinematic or morphology). We observe that the FVP values in two target domains gradually approach each other in three out of four environments (HalfCheetah, Walker, Hopper), while the values in Ant remain relatively stationary. Furthermore, the FVP values in target domains with kinematic shifts are lower than those with morphology shifts across all four environments, which could result from the mismatched state space due to the limited joint ranges of robots in the target domain. Given the differences across different environments, we believe the FVP estimation could be used to quantify the domain differences.

![](images/548539251e1deab7c92ebe9ca452eb8b95ac283589722d7891226878253a6fe5.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (×10⁵) | Kinematic | Morphology |
| ------------------------------ | --------- | ---------- |
| 0.0                            | 0.1       | 0.08       |
| 0.2                            | 0.15      | 0.2        |
| 0.4                            | 0.1       | 0.25       |
| 0.6                            | 0.1       | 0.2        |
| 0.8                            | 0.1       | 0.15       |
| 1.0                            | 0.1       | 0.1        |
</details>

![](images/a4f2c3644856ca13087084454ab5df80f24bb76acef3f7ed062d466a641c1233.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (×10⁵) | Red Line Value | Blue Line Value |
| ------------------------------ | -------------- | --------------- |
| 0.0                            | ~0.5           | ~0.1            |
| 0.2                            | ~0.6           | ~0.2            |
| 0.4                            | ~0.7           | ~0.3            |
| 0.6                            | ~0.8           | ~0.4            |
| 0.8                            | ~0.9           | ~0.5            |
| 1.0                            | ~0.8           | ~0.6            |
</details>

![](images/c73934eaf37fb45b17159612f052823cc1872647e2eb7ab571a2e425676da1bb.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (×10⁵) | Series 1 | Series 2 |
| ------------------------------ | -------- | -------- |
| 0.0                            | 0.0      | 0.0      |
| 0.2                            | 0.3      | 0.2      |
| 0.4                            | 0.4      | 0.3      |
| 0.6                            | 0.5      | 0.4      |
| 0.8                            | 0.6      | 0.5      |
| 1.0                            | 0.7      | 0.6      |
</details>

![](images/299f52a33de7c9593e9d6995f1e85933010ff6a26039ed52b02a88c30a7db22d.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (×10⁵) | Red Line Value | Blue Line Value |
| ------------------------------ | -------------- | --------------- |
| 0.0                            | ~0.8           | ~0.2            |
| 0.2                            | ~0.6           | ~0.1            |
| 0.4                            | ~0.7           | ~0.1            |
| 0.6                            | ~0.5           | ~0.2            |
| 0.8                            | ~0.6           | ~0.3            |
| 1.0                            | ~0.7           | ~0.4            |
</details>

Figure 11: Quantification analysis of the approximated FVP in all environments with different dynamics shifts. The dots are averaged values, and the error bars indicate the standard error across five runs.

# F.2 Sensitivity to Ensemble Size

![](images/11cf900e5de1c33b4b8c8bd9bfc88553df82fcbf8ae80ace1691f67f677485b7.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (x10^5) | M = 7   | M = 5   | M = 3   |
| ------------------------------ | ------- | ------- | ------- |
| 0.0                            | 0       | 0       | 0       |
| 0.2                            | 1500    | 1000    | 1200    |
| 0.4                            | 2500    | 1500    | 2000    |
| 0.6                            | 3500    | 1800    | 2500    |
| 0.8                            | 4500    | 2000    | 3000    |
| 1.0                            | 5000    | 2200    | 3200    |
</details>

![](images/b8cf08d9341f85cc7a57cdb8617aaeecad4f4692e9751473005226b7333a57d7.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (×10⁵) | Return (Line 1) | Return (Line 2) | Return (Line 3) |
| ----------------------------- | --------------- | --------------- | --------------- |
| 0.0                           | 0               | 0               | 0               |
| 0.2                           | 500             | 600             | 700             |
| 0.4                           | 1000            | 1100            | 1200            |
| 0.6                           | 1500            | 1600            | 1700            |
| 0.8                           | 2000            | 2100            | 2200            |
| 1.0                           | 2500            | 2600            | 2700            |
</details>

![](images/e85c50ba05075af245700ce2ec701e5817852e0f48be6e2b228fc06dbd7c2e6e.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (×10⁵) | Return (Line 1) | Return (Line 2) | Return (Line 3) |
| ----------------------------- | --------------- | --------------- | --------------- |
| 0.0                           | 0               | 0               | 0               |
| 0.2                           | 1000            | 800             | 600             |
| 0.4                           | 2000            | 1800            | 1600            |
| 0.6                           | 2500            | 2400            | 2300            |
| 0.8                           | 2700            | 2600            | 2500            |
| 1.0                           | 2800            | 2700            | 2600            |
</details>

![](images/69f88b51e32f505fd5f1e8893376dc7b4053b58e8978a1641a1f1470da128b58.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (x10^5) | Return (Blue) | Return (Red) | Return (Yellow) |
| ------------------------------- | ------------- | ------------ | --------------- |
| 0.0                             | 0             | 0            | 0               |
| 0.2                             | 1500          | 1000         | 800             |
| 0.4                             | 2500          | 2000         | 1800            |
| 0.6                             | 3000          | 2500         | 2200            |
| 0.8                             | 3100          | 2800         | 2600            |
| 1.0                             | 3200          | 3100         | 3000            |
</details>

Figure 12: Performance of the variants with different ensemble size values M. The results validate that a smaller ensemble size is sufficient to achieve competitive asymptotic performance compared to the variant with a large ensemble size in most environments.

We have introduced the dynamics model ensemble to capture the epistemic uncertainty induced by the limited samples from the target domain. However, training the ensemble of the dynamics model takes extra computation resources. Unlike prior works in model-based RL $[30, 61]$ that utilize the generated samples for training, we measure the value difference with the help of the generated samples. Therefore, we aim to investigate whether a smaller ensemble size is sufficient to achieve competitive asymptotic performance. Here we set the ensemble size as different values (M = 7 in the original implementation) and run experiments in four environments. As the results show in Figure 12, variants with a small ensemble size (e.g., M = 3 or M = 5) can achieve identical asymptotic performance compared to the variant with a large ensemble size (e.g., M = 7) in three out of four environments.

# F.3 What about Importance Weighting via FVP instead of Rejection Sampling?

![](images/cf5f70847ad472c5db369ab61b375c7e2d28f0034a823c9215bf35fb64d1c8cd.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (x10^5) | Rejection Sampling Return | Importance Weighting via FVP Return |
| ------------------------------- | -------------------------- | ----------------------------------- |
| 0                               | 0                          | 0                                   |
| 1000                            | ~2000                      | ~1000                               |
| 2000                            | ~3500                      | ~1500                               |
| 3000                            | ~4500                      | ~2000                               |
| 4000                            | ~5000                      | ~2200                               |
| 5000                            | ~5200                      | ~2300                               |
</details>

![](images/57b7f285c17206718929a954c3a3d7685824709f349e179f8047531f702783e4.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (×10⁵) | Return (Blue Line) | Return (Yellow Line) |
| ------------------------------ | ------------------ | -------------------- |
| 500                            | ~800               | ~700                 |
| 1000                           | ~1200              | ~1000                |
| 1500                           | ~1600              | ~1300                |
| 2000                           | ~2000              | ~1600                |
| 2500                           | ~2400              | ~1900                |
| 3000                           | ~2800              | ~2200                |
| 3500                           | ~3200              | ~2500                |
</details>

![](images/46e93bdc4d4ed45eecf43c0fa8b583665ad9f53c5ef86c35ac7cad8f900b0687.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (×10⁵) | Return (Blue Line) | Return (Yellow Line) |
| ----------------------------- | ------------------ | -------------------- |
| 0                             | 500                | 500                  |
| 1000                          | 1000               | 800                  |
| 2000                          | 1500               | 1200                 |
| 3000                          | 2000               | 1800                 |
| 4000                          | 2500               | 2200                 |
| 5000                          | 3000               | 2500                 |
</details>

![](images/bd57a4454a906323b248ca0a5093f3abd4936093eda11d156a59e1d1fb4e3f10.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (×10⁵) | Return (Blue Line) | Return (Orange Line) |
| ------------------------------ | ------------------ | -------------------- |
| 0                              | 0                  | 0                    |
| 500                            | ~500               | ~700                 |
| 1000                           | ~1000              | ~1200                |
| 1500                           | ~1500              | ~1800                |
| 2000                           | ~2000              | ~2200                |
| 2500                           | ~2500              | ~2600                |
| 3000                           | ~2800              | ~2900                |
</details>

![](images/ef476f5cf870331705e881e1a91d9db0819bd8925aca14a6c713a1e8c6239e0a.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (×10⁵) | Return (Blue Line) | Return (Orange Line) |
| ------------------------------ | ------------------ | -------------------- |
| 0.0                            | 0                  | 0                    |
| 0.2                            | ~1000              | ~800                 |
| 0.4                            | ~1500              | ~1300                |
| 0.6                            | ~2000              | ~1800                |
| 0.8                            | ~2500              | ~2200                |
| 1.0                            | ~3200              | ~2800                |
</details>

![](images/1aeee5dab721389550e6bb541c0928322d98946ce934f2c046d352ff7379887a.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (×10⁵) | Return (Blue Line) | Return (Yellow Line) |
| ------------------------------ | ------------------ | -------------------- |
| 0.0                            | ~700               | ~700                 |
| 0.2                            | ~1200              | ~1100                |
| 0.4                            | ~1600              | ~1400                |
| 0.6                            | ~2000              | ~1800                |
| 0.8                            | ~2400              | ~2100                |
| 1.0                            | ~2800              | ~2400                |
</details>

![](images/3dbecbc6dd701e4d841d36b19fe82cd9a223fed3682a4f23c02cade955d452bc.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (×10⁵) | Return (Blue Line) | Return (Yellow Line) |
| ----------------------------- | ------------------ | -------------------- |
| 0.0                           | 0                  | 0                    |
| 0.2                           | ~500               | ~400                 |
| 0.4                           | ~1500              | ~1200                |
| 0.6                           | ~2500              | ~2000                |
| 0.8                           | ~3000              | ~2500                |
| 1.0                           | ~3200              | ~2800                |
</details>

![](images/6805e06c1eb9967995a2b85b9b8f3bc994050ca47897350cca62bb9096195f63.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (x10^5) | Return (Blue Line) | Return (Orange Line) |
| ------------------------------- | ------------------ | -------------------- |
| 0.0                             | 0                  | 0                    |
| 0.2                             | ~1000              | ~500                 |
| 0.4                             | ~2500              | ~1500                |
| 0.6                             | ~3000              | ~2500                |
| 0.8                             | ~3100              | ~2700                |
| 1.0                             | ~3200              | ~2800                |
</details>

Figure 13: Performance of the variants with rejection sampling or importance weighting technique. The results demonstrate that the original algorithm using rejection sampling outperforms the variant using importance weighting via FVP in almost all environments.

In the case of data selection based on the estimated FVP (fictitious value proximity in Eq. (6), one may wonder about using importance weighting via the FVP rather than rejection sampling, which might be sample-inefficient due to the discarded partial data. Here we implement a variant of our algorithm that performs importance weighting with the estimated fictitious value proximity. Specifically, we train the value functions following:

$$
\theta_ {i = 1, 2} \leftarrow \arg \min _ {\theta_ {i}} \frac {1}{2 B} \sum_ {\{(s, a, r, s ^ {\prime}) \} _ {t a r} ^ {B}} \left[ (Q _ {\theta_ {i}} - \mathcal {T} Q _ {\theta_ {i}}) ^ {2} \right] +
$$

$$
\frac {1}{2 B} \sum_ {\{(s, a, r, s ^ {\prime}) \} _ {s r c} ^ {B}} \left[ \frac {\Lambda (s , a , s ^ {\prime})}{\sum_ {\{s , a , s ^ {\prime} \} ^ {B}} \Lambda (s , a , s ^ {\prime})} (Q _ {\theta_ {i}} - \mathcal {T} Q _ {\theta_ {i}}) ^ {2} \right].
$$

We compare the variant with the original algorithm using rejection sampling in all eight environments and demonstrate the results in Figure 13. The original algorithm using rejection sampling outperforms the variant with importance weighting in almost all environments. The accuracy of the value proximity depends on the generated state and the value function. Thus, the estimation of FVP could be biased due to the inaccurate dynamics models and value functions in the early training stage, in which case naively utilizing the source domain samples weighted by the FVP can harm the policy performance concerning the target domain. In contrast, rejection sampling that only utilizes a small portion of source domain samples alleviates the negative effect of the source domain samples.

# F.4 What about Data Filtering via Value instead of FVP?

Prior works have examined sharing data across tasks with different reward functions rather than dynamics [75]. To investigate whether selectively sharing data with a high Q value can address the

![](images/3d947e459df343fe9efd59a605fe1cd7f71d8252d0930a938327d8980b177dd8.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (x10^5) | Filtering via FVP | Filtering via Value |
| ------------------------------ | ----------------- | ------------------- |
| 0                              | 0                 | 0                   |
| 100                            | 1000              | 800                 |
| 200                            | 2000              | 1800                |
| 300                            | 3000              | 2800                |
| 400                            | 3500              | 3300                |
| 500                            | 4000              | 3800                |
| 600                            | 4500              | 4200                |
| 700                            | 4800              | 4500                |
| 800                            | 5000              | 4800                |
| 900                            | 5200              | 5000                |
| 1000                           | 5500              | 5200                |
</details>

![](images/57f999f10be1853d6887688baad6feb7e2b077724a5133ed173041c39c39a0b4.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (×10⁵) | Return (Blue Line) | Return (Yellow Line) |
| ------------------------------ | ------------------ | -------------------- |
| 500                            | ~800               | ~800                 |
| 1000                           | ~1200              | ~1200                |
| 1500                           | ~1600              | ~1600                |
| 2000                           | ~2000              | ~2000                |
| 2500                           | ~2400              | ~2400                |
| 3000                           | ~2800              | ~2800                |
| 3500                           | ~3200              | ~3200                |
</details>

![](images/1b7b44fd9a191b6554b6a210752763246cb176be6ecd22a62c9d044b43188470.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (×10⁵) | Return (Blue Line) | Return (Orange Line) |
| ----------------------------- | ------------------ | -------------------- |
| 0                             | 500                | 500                  |
| 500                           | 1000               | 750                  |
| 1000                          | 1500               | 1000                 |
| 1500                          | 2000               | 1250                 |
| 2000                          | 2500               | 1500                 |
| 2500                          | 3000               | 1750                 |
| 3000                          | 3500               | 2000                 |
</details>

![](images/01d897a471826faf5b9c739abe55664fda31fe9dcfc44087c9955431ea6be69f.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (×10⁵) | Return (Blue Line) | Return (Orange Line) |
| ------------------------------ | ------------------ | -------------------- |
| 0                              | 0                  | 0                    |
| 500                            | 500                | 500                  |
| 1000                           | 1000               | 1000                 |
| 1500                           | 1500               | 1500                 |
| 2000                           | 2000               | 2000                 |
| 2500                           | 2500               | 2500                 |
| 3000                           | 2750               | 2750                 |
| 3500                           | 2800               | 2800                 |
| 4000                           | 2850               | 2850                 |
| 4500                           | 2900               | 2900                 |
| 5000                           | 2950               | 2950                 |
</details>

![](images/d5cbdc2c632d89ef1f30db96ccbf57f93772f92d6de774b3d8ce0b4fa87c663a.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (×10⁵) | Return (Blue Line) | Return (Yellow Line) |
| ------------------------------ | ------------------ | -------------------- |
| 0.0                            | 0                  | 0                    |
| 0.2                            | ~500               | ~400                 |
| 0.4                            | ~1000              | ~800                 |
| 0.6                            | ~1500              | ~1200                |
| 0.8                            | ~2000              | ~1600                |
| 1.0                            | ~3000              | ~2500                |
</details>

![](images/f20a07f42cd08cf1876640b7b8c213395ff262021de60e8840489e34adac9d00.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (×10⁵) | Return (Blue Line) | Return (Yellow Line) |
| ------------------------------ | ------------------ | -------------------- |
| 0.0                            | 500                | 500                  |
| 0.2                            | 1000               | 1200                 |
| 0.4                            | 1500               | 1800                 |
| 0.6                            | 2000               | 2400                 |
| 0.8                            | 2500               | 3000                 |
| 1.0                            | 3000               | 3500                 |
</details>

![](images/70e0844aa38784a96ec0d0ac2a7645a265ea6dd1068216f0bbdc64440ac8bc1d.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (×10⁵) | Return (Blue Line) | Return (Yellow Line) |
| ----------------------------- | ------------------ | -------------------- |
| 0.0                           | ~200               | ~200                 |
| 0.2                           | ~800               | ~700                 |
| 0.4                           | ~1600              | ~1500                |
| 0.6                           | ~2400              | ~2300                |
| 0.8                           | ~3000              | ~2900                |
| 1.0                           | ~3400              | ~3300                |
</details>

![](images/b621166310f7962578a0e6ffbd398c65c5185e3c8c2250e4525732a0ab45e0fc.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (x10^5) | Return (Blue Line) | Return (Yellow Line) |
| ------------------------------ | ------------------ | -------------------- |
| 0.0                            | 0                  | 0                    |
| 0.2                            | ~1000              | ~500                 |
| 0.4                            | ~2500              | ~1000                |
| 0.6                            | ~3000              | ~1500                |
| 0.8                            | ~3200              | ~2000                |
| 1.0                            | ~3300              | ~2500                |
</details>

Figure 14: Performance of the variants that employ data filtering based on Value or FVP. The results demonstrate that the original algorithm outperforms the variant using data filtering via Value in four of eight environments.

![](images/2de4a5cfaa8f4e6b7ca661f03e7a84130cbaeec4c1a264c3d49455016a44afae.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (×10⁵) | VGDF Return | DGDF Return |
| ----------------------------- | ----------- | ----------- |
| 0.0                           | 500         | 500         |
| 0.2                           | 1000        | 750         |
| 0.4                           | 1500        | 1250        |
| 0.6                           | 2000        | 1750        |
| 0.8                           | 2500        | 2250        |
| 1.0                           | 3000        | 2500        |
</details>

![](images/938f0a7493fa7d8b9d433155aedaab5e38ecba0ac6265cc667719558c410a42b.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (×10⁵) | Return (Blue Line) | Return (Yellow Line) |
| ----------------------------- | ------------------ | -------------------- |
| 0.0                           | 0                  | 0                    |
| 0.2                           | ~1000              | ~800                 |
| 0.4                           | ~2000              | ~1500                |
| 0.6                           | ~2500              | ~2000                |
| 0.8                           | ~3000              | ~2500                |
| 1.0                           | ~3500              | ~3000                |
</details>

![](images/2c111fde733e5710830942ef901cca3559e1b270932dec059f192cc0d53ad9b7.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (×10⁵) | Return (Blue Line) | Return (Yellow Line) |
| ------------------------------ | ------------------ | -------------------- |
| 0.0                            | ~200               | ~800                 |
| 0.2                            | ~1200              | ~400                 |
| 0.4                            | ~2500              | ~200                 |
| 0.6                            | ~2800              | ~150                 |
| 0.8                            | ~2700              | ~100                 |
| 1.0                            | ~2900              | ~150                 |
</details>

![](images/0d0f14d57ade53da9504a3eb69923ac66369a72fc254019636b199e2a3337892.jpg)

<details>
<summary>line</summary>

| Steps in Target Domain (×10⁵) | Return (Blue Line) | Return (Orange Line) |
| ----------------------------- | ------------------ | -------------------- |
| 0.0                           | 0                  | 0                    |
| 0.2                           | ~500               | ~500                 |
| 0.4                           | ~2500              | ~2000                |
| 0.6                           | ~3000              | ~2800                |
| 0.8                           | ~3100              | ~3000                |
| 1.0                           | ~3100              | ~3100                |
</details>

Figure 15: Comparison with the variant performing data filtering based on estimated dynamics discrepancies. The results demonstrate that the original algorithm outperforms the variant using data filtering via Value in four of eight environments, validating the effect of the value consistency.

online dynamics adaptation problem, we propose a variant of our algorithm that shares partial data with a relatively high Q value from the source domain. Specifically, we train the value functions following:

$$
\begin{array}{l} \theta_ {i = 1, 2} \leftarrow \arg \min _ {\theta_ {i}} \frac {1}{2 B} \sum_ {\{(s, a, r, s ^ {\prime}) \} _ {t a r} ^ {B}} \left[ (Q _ {\theta_ {i}} - \mathcal {T} Q _ {\theta_ {i}}) ^ {2} \right] + \\ \frac{1}{\left\lfloor 2B\cdot\xi\% \right\rfloor}\sum_{\left\{(s,a,r,s^{\prime})\right\}_{src}^{B}}\left[ \mathbb{1}\left(Q_{\theta_{i}}(s,a) > Q_{\xi \%}\right)(Q_{\theta_{i}} - \mathcal{T}Q_{\theta_{i}})^{2}\right], \\ \end{array}
$$

where $Q_{\xi\%}$ is the top $\xi$ -quantile Q value of a batch of source domain samples. We set $\xi\%$ as 25%, the same as our implementation. We compare the variant with the original algorithm in all eight environments and demonstrate the results in Figure 14. The results demonstrate that the original algorithm outperforms the variant using data filtering via value in four of eight environments. Due to the dynamics mismatch, a state-action pair from the source domain will lead to inconsistent states concerning two domains. Therefore, directly utilizing the transitions with high Q value without considering the consistency of the next state would provide a counterfactual value target for the state-action pair, which can result in an improper value estimation for learning.

# F.5 Comparison with Dynamics-guided Data Filtering

To investigate the effect of value consistency, we perform the ablation study by comparing VGDF to a variant that shares partial data based on dynamics discrepancies, i.e., Dynamics-guided Data Filtering (DGDF). Specifically, we estimate the dynamics discrepancy via the learned classifiers

Table 3: Results in PyBullet environments. We evaluate the algorithms via the performance of the learned policy in the target domain and report the mean and std of the results across five runs with different random seeds. (# source, # target) denotes the number of source domain data versus the number of target domain data. HC and HP denote HalfCheetah and Hopper, respectively. 

<table><tr><td></td><td>DARC</td><td>Finetune</td><td>VGDF</td><td>DARC</td><td>Finetune</td><td>VGDF</td></tr><tr><td>(# source, # target)</td><td>200k, 20k</td><td>1M, 20k</td><td>200k, 20k</td><td>1M, 100k</td><td>1M, 100k</td><td>1M, 100k</td></tr><tr><td>PyBullet - HC</td><td>304 ± 211</td><td>653 ± 51</td><td>770 ±203</td><td>679 ± 131</td><td>678 ± 38</td><td>808 ±89</td></tr><tr><td>PyBullet - HP</td><td>73 ± 20</td><td>240 ± 189</td><td>957 ±39</td><td>99 ± 20</td><td>869 ± 32</td><td>1006 ±2</td></tr></table>

following the prior works $[17, 49]$ . Same as VGDF, we share the source domain transitions whose estimated dynamics difference is smaller than the quantile value. We set the selection ratios as 25%, the same as our implementation. The results demonstrate that the original algorithm outperforms the variant in three out of four environments, validating the superior effect of the value consistency compared to the dynamics discrepancy.

# F.6 Extended Results in Pybullet Environments

To investigate the generality of VGDF, we perform additional experiments in PyBullet-HalfCheetah and PyBullet-Hopper from PyBullet environments $[12]$ which utilize Bullet as the physical engine instead of Mujoco. We first provide the details of the dynamics gap in the environments. In both environments, we regard the original environments as the source domains. In PyBullet-Hopper, we devised the target domain by increasing the torso size from 0.05 to 0.15, to simulate the morphology change. In PyBullet-HalfCheetah, we constrain the joint range of the front thigh from $[-1.5, 0.8]$ to $[-1.5, 0.4]$ , and the joint range of the front shin from $[-1.2, 1.1]$ to $[-1.2, 0.1]$ , to simulate the broken joint scenario that is widely used in related works.

The results are shown in the Table 3, and we report the performance of all algorithms concerning different numbers of target domain samples. All results are averaged across five runs with different seeds. The results demonstrate that VGDF consistently outperforms baselines given different number of target domain data, demonstrating the generality of our method.