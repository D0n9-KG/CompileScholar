# SEABO: A SIMPLE SEARCH-BASED METHOD FOR OFFLINE IMITATION LEARNING

Jiafei Lyu $^{1}$ , \*Xiaoteng Ma $^{2}$ , Le Wan $^{3}$ , Runze Liu $^{1}$ , Xiu Li $^{1,\dagger}$ , Zongqing Lu $^{4,\dagger}$

$^{1}$ Tsinghua Shenzhen International Graduate School, Tsinghua University

$^{2}$ Department of Automation, Tsinghua University, $^{3}$ IEG, Tencent

$^{4}$ School of Computer Science, Peking University

lvjf20@mails.tsinghua.edu.cn, li.xiu@sz.tsinghua.edu.cn, zongqing.lu@pku.edu.cn

# ABSTRACT

Offline reinforcement learning (RL) has attracted much attention due to its ability in learning from static offline datasets and eliminating the need of interacting with the environment. Nevertheless, the success of offline RL relies heavily on the offline transitions annotated with reward labels. In practice, we often need to hand-craft the reward function, which is sometimes difficult, labor-intensive, or inefficient. To tackle this challenge, we set our focus on the offline imitation learning (IL) setting, and aim at getting a reward function based on the expert data and unlabeled data. To that end, we propose a simple yet effective search-based offline IL method, tagged SEABO. SEABO allocates a larger reward to the transition that is close to its closest neighbor in the expert demonstration, and a smaller reward otherwise, all in an unsupervised learning manner. Experimental results on a variety of D4RL datasets indicate that SEABO can achieve competitive performance to offline RL algorithms with ground-truth rewards, given only a single expert trajectory, and can outperform prior reward learning and offline IL methods across many tasks. Moreover, we demonstrate that SEABO also works well if the expert demonstrations contain only observations. Our code is publicly available at https://github.com/dmksjfl/SEABO.

# 1 INTRODUCTION

In recent years, reinforcement learning (RL) (Sutton & Barto, 2018) has made prominent achievements in fields like video games (Mnih et al., 2015; Schrittwieser et al., 2020), robotics (Kober et al., 2013), nuclear fusion control (Degrave et al., 2022), etc. It is known that RL is a reward-oriented learning paradigm. Online RL algorithms typically require an interactive environment for data collection and improve the policy through trial-and-error. However, continual online interactions are usually expensive, time-consuming, or even dangerous in many practical applications. Offline RL (Lange et al., 2012; Levine et al., 2020), instead, resorts to learning optimal policies from previously gathered datasets, which are composed of trajectories containing observations, actions, and rewards.

A bare fact is that reward engineering is often difficult, expensive, and labor-intensive. It is also hard to specify or abstract a good reward function given some rules. To overcome this challenge in the offline setting, there are generally two methods. First, one can train the policy via the behavior cloning (BC) algorithm (Pomerleau, 1988), but its performance is heavily determined by the performance of the data-collecting policy (a.k.a., the behavior policy). Second, one can learn a reward function from some expert demonstrations and assign rewards to the unlabeled data in the dataset. Then, the policy can be optimized by leveraging the reward. This is also known as offline imitation learning (offline IL). Note that in many real-world tasks, acquiring a few expert demonstrations is easy (e.g., ask a human expert to operate the system) and affordable.

However, it turns out that, similar to offline RL, offline IL also suffers from distribution shift issue (Kim et al., 2022b; DeMoss et al., 2023), where the learned policy deviates from the data-collecting policy, leading to poor performance during evaluation. Prior works concerning on distribution cor-

![](images/80883f7d12e610e4a5b0819e970178e4fbef0a654760455c5568648ee591132b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["expert sample"] --> B["KD tree"]
    C["expert samples"] --> B
    D["query samples"] --> B
    B --> E["reward"]
    E --> F["offline RL"]
    G["unlabeled transition"] --> H["query"]
    I["expert demonstration"] --> J["SEABO"]
    J --> E
```
</details>

Figure 1: Left: The key idea behind SEABO. We assign larger rewards to transitions that are closer to the expert demonstration, and smaller rewards otherwise. The dotted lines connect the query samples with their nearest neighbors along the demonstration. Right: Illustration of the SEABO framework. Given an expert demonstration, we first construct a KD-tree and then feed the unlabeled samples into the tree to query their nearest neighbors. We use the resulting distance to calculate the reward label. Then one can adopt any existing offline RL algorithm to train on the labeled dataset.

rection estimation (DICE family) address this by enforcing the learned policy to be close to the behavior policy via a distribution divergence measure (e.g., f-divergence (Ghasemipour et al., 2019; Ke et al., 2019)). However, such distribution matching schemes can incur training instability (Ma et al., 2022) and over-conservatism (Yu et al., 2023), and they often involve training task-specific discriminators. On the other hand, some works seek to decouple the processes of reward annotation and policy optimization (Zolna et al., 2020; Luo et al., 2023). However, they involve solving complex optimal transport problems or contrasting expert states and unlabeled trajectory states.

In this paper, we propose a simple yet effective alternative, SEArch-Based method for Offline imitation learning, namely SEABO, that leverages search algorithms to acquire reward signals in an unsupervised learning manner. As illustrated in Figure 1 (left), we hypothesize that the transition is near-optimal if it lies close to the expert trajectory, hence larger reward ought to be assigned to it, and vice versa. To that end, we propose to determine whether the sample approaches the expert trajectory via measuring the distance between the query sample and its nearest neighbor in the expert trajectory. In practice, as depicted in Figure 1 (right), SEABO first builds a KD-tree upon expert demonstrations. Then for each unlabeled sample in the dataset, we query the tree to find its nearest neighbor, and measure their distance. If the distance is small (i.e., close to expert trajectory), a large reward will be annotated, while if the distance is large (i.e., stray away from the expert trajectory), the assigned reward is low. SEABO is efficient and easy to implement. It can be combined with any existing offline RL algorithm to acquire a meaningful policy from the static offline dataset.

Empirical results on the D4RL (Fu et al., 2020) datasets show that SEABO can enable the offline RL algorithm to achieve competitive or even better performance against its performance under ground-truth rewards with only one expert trajectory. SEABO also beats recent strong reward annotation methods and imitation learning baselines on many datasets. Furthermore, we also demonstrate that SEABO can learn effectively when the expert demonstrations are composed of pure observations.

# 2 PRELIMINARY

We formulate the interaction between the environment and policy as a Markov Decision Process (MDP) specified by the tuple $\langle S, A, p, r, \gamma, p_0 \rangle$ , where $S$ is the state space, $A$ is the action space, $p: S \times A \mapsto S$ is the transition dynamics, $r: S \times A \mapsto \mathbb{R}$ is the scalar reward signal, $\gamma \in [0,1]$ is the discount factor, $p_0$ is the initial state distribution. A policy $\pi(a|s)$ outputs the action based on the state $s$ . We assume that the underlying MDP has a finite horizon. The goal of RL is to maximize the discounted future reward: $J(\pi) = \mathbb{E}_{s_0 \sim p_0} \mathbb{E}_{a \sim \pi, s_{t+1} \sim p(\cdot | s_t, a_t)}[\sum_{t=0}^{T-1} \gamma^t r(s_t, a_t)]$ . Whereas, in many scenarios, it is often hard for one to get the reward signals. It is more common that the unlabeled trajectory, $\tau = \{s_0, a_0, \ldots, s_t, a_t, \ldots, s_T\}$ , is collected. This poses veritable challenges for applying offline RL algorithms.

In this paper, we focus on the offline IL setting. We assume that we have access to the dataset of expert demonstrations $\mathcal{D}_{e}=\{\tau_{e}^{(i)}\}_{i=1}^{M}$ , and a dataset of unlabeled data $D_{u}=\{\tau_{u}^{(i)}\}_{i=1}^{N}$ , where M and N are the sizes of the expert dataset and unlabeled dataset, respectively. The unlabeled trajectories are gathered by some unknown behavior policy $\mu$ . Note that we allow the expert demonstrations to either contain actions or do not contain actions. We aim at attaining the reward function by extracting information from the expert trajectories and unlabeled trajectories, and assigning rewards to the unlabeled datasets, without any interactions with the environment. Then we can train the policy using any offline RL algorithm.

# 3 RELATED WORK

Offline Reinforcement Learning. In offline RL (Lange et al., 2012; Levine et al., 2020), the agent is not permitted to interact with the environment, and can only learn policies from previously gathered dataset $\mathcal{D} = \{(s_i, a_i, r_i, s_{i+1})\}_{i=1}^N$ , where $N$ is the dataset size. Existing work on offline RL can be generally categorized into model-based (Yu et al., 2020; 2021; Kidambi et al., 2020; Lyu et al., 2022b; Rigter et al., 2022; Lu et al., 2022a; Chen et al., 2021; Janner et al., 2021; Uehara & Sun, 2022; Zhang et al., 2023) and model-free approaches (Fujimoto et al., 2019; Fujimoto & Gu, 2021; Kumar et al., 2020; Kostrikov et al., 2022; Lyu et al., 2022c;a; Cheng et al., 2022; Zhou et al., 2020; Ran et al., 2023; Bai et al., 2022; Yang et al., 2024). The success of these methods rely heavily on the requirement that the datasets must contain annotated reward signals.

Imitation Learning. Imitation Learning (IL) considers optimizing the behavior of the agent given some expert demonstrations, and no reward is needed. The primary goal of IL is to mimic the behavior of the expert demonstrator. Behavior cloning (BC) (Pomerleau, 1988) directly performs supervised regression or maximum-likelihood on expert demonstrations. Yet, BC can suffer from compounding error and may result in performance collapse upon unseen states (Ross et al., 2011). Another line of work, inverse reinforcement learning (IRL) (Arora & Doshi, 2021), first learns a reward function using expert demonstrations, and then utilizes this reward function to train policies with RL algorithms. Typical IRL algorithms include adversarial methods (Ho & Ermon, 2016; Jeon et al., 2018; Kostrikov et al., 2019; Baram et al., 2017), maximum-entropy approaches (Ziebart et al., 2008; Boularias et al., 2011), normalizing flows (Freund et al., 2023), etc. However, these methods often require abundant online transitions to train a good policy. Imitation learning without online interactions, which is the focus of our work, is hence attractive and remains an active area. There are many advances in offline IL, such as applying online IRL algorithms in the offline setting (Zolna et al., 2020; Yue et al., 2023), using energy-based methods (Jarrett et al., 2020), weighting the BC loss with the output of the trained discriminator (Xu et al., 2022), etc. Among them, DICE (Nachum et al., 2019) family receives much attention. Methods like ValueDICE (Kostrikov et al., 2020), DemoDICE (Kim et al., 2022b), and LobsDICE (Kim et al., 2022a) can consistently drub BC in the offline setting. Notably, a recent work, OTR (Luo et al., 2023), acquires the reward function in the offline setting via optimal transport. OTR decouples the processes of reward learning and policy optimization. Still, OTR needs to solve complex optimal transport problems. We, instead, explore to get the reward function via a search-based method.

Search Algorithms. Search algorithms (Korf, 1999) are critical components in artificial intelligence. Typical search algorithms include brute-force search algorithms (Dijkstra, 1959; Stickel & Tyson, 1985; Korf, 1985; Taylor & Korf, 1993), heuristic search approaches (Doran & Michie, 1966; Hart et al., 1968; Pohl, 1970; Edelkamp & Schrödl, 2011), etc. In this paper, we resort to the simple search approach, KD-tree (Bentley, 1975), for capturing the nearest neighbors of the unlabeled data in the expert demonstrations.

# 4 OFFLINE IMITATION LEARNING VIA SEARCH-BASED METHOD

In this section, we formally present our novel approach for offline imitation learning, SEArch-Based Offline imitation learning (SEABO). We begin by analyzing the common formulation adopted in distribution matching IL methods (Ho & Ermon, 2016; Kim et al., 2020; Kostrikov et al., 2020), which attempt to match the state-action distribution of the agent $p_{\pi}$ and the expert $p_e$ , often by means of minimizing some $f$ -divergence measurement $D_f$ : $\arg \min_{\pi} D_f(p_{\pi} \| p_e)$ . Though these methods have promising results, they usually require training task-specific discriminators and suffer

from training instability (Wang et al., 2020; Ma et al., 2022). A natural question arises, can we get the reward signals without training neural networks?

Instead of measuring the distribution of states or state-action pairs, we want to determine the optimality of a single transition. Our idea is quite straightforward, the closer the transition is to the expert trajectory, the more optimal this transition is. The agent ought to pay more attention to those optimal transitions. This motivates us to measure how close the unlabeled transition is to the expert trajectories. We propose to achieve this by finding the nearest neighbor of the query transition in the expert demonstrations, and then measuring their distance (e.g., Euclidean distance). If the distance is large, then the transition is away from the expert demonstration. While if the distance is small, it indicates that the transition is near-optimal, or even is exact expert data if the distance approaches 0. Intuitively, this distance can be interpreted as a reward signal.

To that end, we construct a function dubbed NearestNeighbor(demo, query\_sample) that returns the nearest neighbor of the query sample in the expert demonstrations. Suppose the expert trajectories are made up of state-action pairs, then for the query sample $(s, a, s')$ , we have:

$$
(\tilde {s} _ {e}, \tilde {a} _ {e}, \tilde {s} _ {e} ^ {\prime}) = \text { NearestNeighbor } (\mathcal {D} _ {e}, (s, a, s ^ {\prime})). \tag {1}
$$

Then we measure their deviation using some distance measurement D:

$$
d = D ((\tilde {s} _ {e}, \tilde {a} _ {e}, \tilde {s} _ {e} ^ {\prime}), (s, a, s ^ {\prime})). \tag {2}
$$

Afterward, following prior work (Cohen et al., 2022; Freund et al., 2023; Dadashi et al., 2021; Luo et al., 2023), we get the rewards via a squashing function: $r = \alpha \exp(-\beta \times d)$ , where $\alpha$ and $\beta$ are hyperparameters that control the scale of the reward and the impact of the distance, respectively.

Algorithm 1 SEArch-Based Offline Imitation Learning (SEABO)   
1: Require: expert demonstrations $D_{e}$ , unlabeled dataset $D_{u}$ 2: Initialize $D_{label} \leftarrow \emptyset$ . Given distance measurement D
3: for $(s, a, s')$ in $D_{u}$ do
4: Find its nearest neighbor, $(\tilde{s}_{e}, \tilde{a}_{e}, \tilde{s}_{e}') = \text{NearestNeighbor}(\mathcal{D}_{e}, (s, a, s'))$ 5: Measure the distance: $d = D((\tilde{s}_{e}, \tilde{a}_{e}, \tilde{s}_{e}', (s, a, s'))$ 6: Get the reward signal via Equation 3
7: $D_{label} \leftarrow D_{label} \cup (s, a, r, s')$ 8: end for

We name the resulting method SEABO, and list its pseudo-code in Algorithm 1. For practical implementation of SEABO, we leverage KD-tree (Bentley, 1975) for searching the nearest neighbors, and adopt Euclidean distance (Torabi et al., 2019) as the distance measurement for simplicity (i.e., the default setting of KD-tree). We also slightly modify the aforementioned formula of the reward function to make it better adapt to different tasks with one set of hyperparameters, which gives:

$$
r = \alpha \exp \left(- \frac {\beta \times d}{| \mathcal {A} |}\right), \tag {3}
$$

where $|A|$ is the dimension of the action space. Note that this technique is also adopted in Dadashi et al. (2021). We choose to use $(s, a, s')$ to query since the magnitude of states and actions may be different. One possible solution is to query the demonstrations via $(\xi \times s, a), \xi \in \mathbb{R}^{+}$ , but it introduces an additional hyperparameter that may need to be tuned per dataset. We empirically find that involving $s'$ in the query sample can ensure good performance across many tasks. The above procedure (as specified in Figure 1 (right)) also applies when the expert demonstrations contain only observations, because it is feasible that we find the nearest neighbors using only observations.

SEABO enjoys many advantages over prior reward learning methods or offline imitation learning algorithms. First, SEABO does not require any additional processing upon the offline dataset $^{1}$ . The unlabeled dataset can have different trajectory lengths, and the unlabeled trajectories can be fragmented, or even scattered, since SEABO computes the rewards only using the single transition instead of the entire trajectory. Second, SEABO does not require training reward models or discriminators, hence getting rid of the issues of training instability and hyperparameter tuning of the neural networks. Third, SEABO is easy to implement and can be combined with any offline RL algorithm.

![](images/0509b97671c9341a5545e00d6babf77d897053910a289161425757dc70701567.jpg)

<details>
<summary>line</summary>

| reward | density |
| ------ | ------- |
| -10    | 0.00    |
| 0      | 0.00    |
| 5      | 0.23    |
| 10     | 0.25    |
| 15     | 0.00    |
| 20     | 0.00    |
</details>

![](images/d82c396b8c83c478aa1112162026315a171ca7e5188f754c2d5639de9afb5213.jpg)

<details>
<summary>line</summary>

| reward | density |
| ------ | ------- |
| -0.5   | 0.0     |
| 0.0    | 3.0     |
| 0.5    | 2.0     |
| 1.0    | 0.0     |
| 1.5    | 0.0     |
</details>

![](images/0e514b2c7d4ea5dce56ccc771e71baf582499274b60f84cc2f7ada8185467daf.jpg)

<details>
<summary>line</summary>

| reward | density |
| ------ | ------- |
| -5.0   | 0.0     |
| -2.5   | 0.0     |
| 0.0    | 0.0     |
| 2.5    | 0.4     |
| 5.0    | 0.2     |
| 7.5    | 0.0     |
| 10.0   | 0.0     |
</details>

![](images/b23bcd8794584099caf3bac096b065f37d7b1496cafb498fa8456a9557cf8617.jpg)

<details>
<summary>line</summary>

| reward | density |
| ------ | ------- |
| 0.0    | 0.0     |
| 0.5    | 3.0     |
| 1.0    | 0.0     |
| 1.5    | 0.0     |
</details>

Figure 2: Density plots of ground-truth rewards and rewards acquired by SEABO. Note that oracle indicates the ground-truth rewards are plotted.

To show the effectiveness of our method, we plot the distribution of ground-truth rewards (oracle) and rewards given by SEABO. We choose two datasets, halfcheetah-medium-expert-v2 and hopper-medium-v2 from D4RL (Fu et al., 2020) as examples, and use $\alpha = 1, \beta = 0.5$ , which is the same as our hyperparameter setup in Section 5. The results are summarized in Figure 2. We find that the reward distributions of SEABO resemble those of oracle. Notably, SEABO successfully gives two peaks in halfcheetah-medium-expert, indicating that it can distinguish samples of different qualities. These reveal that SEABO can serve as a good reward labeler, which validates its combination with off-the-shelf offline RL algorithms.

# 5 EXPERIMENTS

In this section, we empirically evaluate SEABO on D4RL datasets. We are targeted at examining, given only one single expert demonstration, whether SEABO can make different base offline RL algorithms recover or beat their performance with ground-truth rewards across varied tasks. We are also interested in exploring how SEABO competes against prior reward learning and offline imitation learning methods. We further investigate whether SEABO can work well if the expert demonstrations are composed of pure observations. Moreover, we check how different choices of search algorithms affect the performance of SEABO.

We discard reward signals in the D4RL datasets to form unlabeled datasets. For expert demonstrations, we follow Luo et al. (2023) and utilize the trajectory with the highest return in the raw dataset for ease of evaluation. One can also use a separate expert trajectory. All of the experiments in this paper are run for 1M gradient steps over five different random seeds, and the results are averaged at the final gradient step. We report the mean performance in conjunction with the corresponding standard deviation. We adopt the same squashing function for tasks under the same domain. Unless specified, we use the number of expert demonstrations K = 1 for evaluation. It is worth noting that SEABO is computationally efficient since there is only a single expert trajectory, and the time complexity of KD-tree gives $\mathcal{O}(d_{f} \log |\mathcal{D}_{e}|)$ , where $d_{f}$ is the feature dimension size. It takes SEABO about 1 minute to annotate 1 million unlabeled transitions using merely CPUs. Hence, we believe the overall computation overhead from SEABO is minor and tolerable. We defer the experimental details and hyperparameter setup for all of our experiments to Appendix A.

# 5.1 MAIN RESULTS

SEABO upon different base algorithms. We first explore whether SEABO can aid different offline RL algorithms. We verify this by incorporating SEABO with two popular offline RL algorithms, TD3\_BC (Fujimoto & Gu, 2021) and IQL (Kostrikov et al., 2022). We conduct experiments on 9 medium-level (medium, medium-replay, medium-expert) D4RL MuJoCo locomotion “-v2” datasets (halfcheetah, hopper, walker2d) and summarize the results in Table 1. One can see that IQL+SEABO beats IQL with ground-truth rewards on 6 out of 9 datasets, and TD3\_BC+SEABO outperforms TD3\_BC with raw rewards on 5 out of 9 datasets. On other datasets, SEABO can achieve competitive performance against the oracle performance. The overall scores of SEABO with IQL and TD3\_BC exceed those of ground-truth rewards. This evidence indicates that SEABO can generate high-quality rewards and benefit different offline RL algorithms.

SEABO competes against baselines. To better illustrate the effectiveness of SEABO, we compare IQL+SEABO against the following strong reward learning and offline IL baselines: ORIL (Zolna et al., 2020), which learns the reward function by contrasting the expert demonstrations with the

Table 1: Results of SEABO upon different base algorithms. $\mu_{max}$ denotes the normalized return of the highest return trajectory in the specific dataset, IQL and TD3\_BC indicate that they are trained upon the ground-truth reward labels, while +SEABO indicates the algorithm is trained on the reward signals provided by SEABO. The normalized average scores at the final 10 episodes of evaluations are reported, along with standard deviations. We bold the mean score and highlight the cell if SEABO outperforms algorithms trained on ground-truth rewards. 

<table><tr><td>Task Name</td><td> $\mu_{\text{max}}$ </td><td>IQL</td><td>IQL+SEABO</td><td>TD3_BC</td><td>TD3_BC+SEABO</td></tr><tr><td>halfcheetah-medium</td><td>45.0</td><td>47.4±0.2</td><td>44.8±0.3</td><td>48.0±0.7</td><td>45.9±0.3</td></tr><tr><td>hopper-medium</td><td>99.5</td><td>66.2±5.7</td><td>80.9±3.2</td><td>60.7±12.5</td><td>76.1±4.2</td></tr><tr><td>walker2d-medium</td><td>92.0</td><td>78.3±8.7</td><td>80.9±0.6</td><td>83.7±5.3</td><td>76.6±0.4</td></tr><tr><td>halfcheetah-medium-replay</td><td>42.4</td><td>44.2±1.2</td><td>42.3±0.1</td><td>44.4±0.8</td><td>43.0±0.4</td></tr><tr><td>hopper-medium-replay</td><td>98.6</td><td>94.7±8.6</td><td>92.7±2.9</td><td>64.8±25.5</td><td>96.3±3.0</td></tr><tr><td>walker2d-medium-replay</td><td>89.9</td><td>73.8±7.1</td><td>74.0±2.7</td><td>87.4±8.4</td><td>73.1±2.2</td></tr><tr><td>halfcheetah-medium-expert</td><td>92.8</td><td>86.7±5.3</td><td>89.3±2.5</td><td>93.5±2.0</td><td>95.7±0.4</td></tr><tr><td>hopper-medium-expert</td><td>116.0</td><td>91.5±14.3</td><td>97.5±5.8</td><td>100.2±20.0</td><td>107.1±3.3</td></tr><tr><td>walker2d-medium-expert</td><td>109.0</td><td>109.6±1.0</td><td>110.9±0.2</td><td>109.5±0.5</td><td>109.7±0.2</td></tr><tr><td>Total Score</td><td>785.2</td><td>692.4</td><td>713.3</td><td>692.3</td><td>723.5</td></tr></table>

Table 2: Comparison of SEABO against some recent baselines. We report the mean normalized scores and the corresponding standard deviations. We bold and highlight the mean score cell if it is close to or beats IQL trained on the raw rewards. 

<table><tr><td>Task Name</td><td>BC</td><td>10%BC</td><td>IQL</td><td>IQL+ORIL</td><td>IQL+UDS</td><td>IQL+OTR</td><td>IQL+SEABO</td></tr><tr><td>halfcheetah-medium</td><td>42.6</td><td>42.5</td><td> $47.4 \pm 0.2$ </td><td> $49.0 \pm 0.2$ </td><td> $42.4 \pm 0.3$ </td><td> $43.2 \pm 0.2$ </td><td> $44.8 \pm 0.3$ </td></tr><tr><td>hopper-medium</td><td>52.9</td><td>56.9</td><td> $66.2 \pm 5.7$ </td><td> $47.0 \pm 4.0$ </td><td> $54.5 \pm 3.0$ </td><td> $74.2 \pm 5.1$ </td><td> $80.9 \pm 3.2$ </td></tr><tr><td>walker2d-medium</td><td>75.3</td><td>75.0</td><td> $78.3 \pm 8.7$ </td><td> $61.9 \pm 6.6$ </td><td> $68.9 \pm 6.2$ </td><td> $78.7 \pm 2.2$ </td><td> $80.9 \pm 0.6$ </td></tr><tr><td>halfcheetah-medium-replay</td><td>36.6</td><td>40.6</td><td> $44.2 \pm 1.2$ </td><td> $44.1 \pm 0.6$ </td><td> $37.9 \pm 2.4$ </td><td> $41.8 \pm 0.3$ </td><td> $42.3 \pm 0.1$ </td></tr><tr><td>hopper-medium-replay</td><td>18.1</td><td>75.9</td><td> $94.7 \pm 8.6$ </td><td> $82.4 \pm 1.7$ </td><td> $49.3 \pm 22.7$ </td><td> $85.4 \pm 0.8$ </td><td> $92.7 \pm 2.9$ </td></tr><tr><td>walker2d-medium-replay</td><td>26.0</td><td>62.5</td><td> $73.8 \pm 7.1$ </td><td> $76.3 \pm 4.9$ </td><td> $17.7 \pm 9.6$ </td><td> $67.2 \pm 6.0$ </td><td> $74.0 \pm 2.7$ </td></tr><tr><td>halfcheetah-medium-expert</td><td>55.2</td><td>92.9</td><td> $86.7 \pm 5.3$ </td><td> $87.5 \pm 3.9$ </td><td> $63.0 \pm 5.7$ </td><td> $87.4 \pm 4.4$ </td><td> $89.3 \pm 2.5$ </td></tr><tr><td>hopper-medium-expert</td><td>52.5</td><td>110.9</td><td> $91.5 \pm 14.3$ </td><td> $29.7 \pm 22.2$ </td><td> $53.9 \pm 2.5$ </td><td> $88.4 \pm 12.6$ </td><td> $97.5 \pm 5.8$ </td></tr><tr><td>walker2d-medium-expert</td><td>107.5</td><td>109.0</td><td> $109.6 \pm 1.0$ </td><td> $110.6 \pm 0.6$ </td><td> $107.5 \pm 1.7$ </td><td> $109.5 \pm 0.3$ </td><td> $110.9 \pm 0.2$ </td></tr><tr><td>Total Score</td><td>466.7</td><td>666.2</td><td>692.4</td><td>588.5</td><td>495.1</td><td>675.8</td><td>713.3</td></tr></table>

unlabeled trajectories; UDS (Yu et al., 2022), which keeps the rewards in the expert demonstrations and simply assigns minimum rewards to the unlabeled data; OTR (Luo et al., 2023), which learns a reward function via using the optimal transport to get the optimal alignment between the expert demonstrations and unlabeled trajectories. For a fair comparison, all these methods adopt IQL as the base algorithm. We additionally compare against BC, and 10%BC (Chen et al., 2021). We take the results of IQL+ORIL and IQL+UDS directly from the OTR paper. As OTR computes rewards using pure observations (and SEABO uses $(s, a, s')$ to query the reward), we modify its way of solving optimal coupling by involving actions, and run IQL+OTR on these datasets with its official codebase. We summarize the comparison results in Table 2. It can be found that, though methods like ORIL and OTR can lead to competitive or better performance on some of the datasets than IQL trained with raw rewards, SEABO beats them on numerous tasks. Meanwhile, SEABO is the only method that can surpass IQL with ground-truth rewards in terms of the total score.

SEABO evaluation on wider datasets. We further evaluate IQL+SEABO on two challenging domains from D4RL, AntMaze and Adroit. We run IQL with ground-truth rewards to obtain the IQL performance. We take the results of IQL+OTR from its paper directly. Table 3 demonstrates the detailed comparison results. We find that IQL+SEABO beats IQL and IQL+OTR on 5 out of 6 datasets on AntMaze, and outperforms baselines on 6 out of 8 datasets on Adroit, often by a large margin. IQL+SEABO incurs a performance improvement of 6.0% and 32.0% beyond IQL with vanilla rewards on AntMaze and Adroit tasks, respectively. These indicate that SEABO with one single expert trajectory can handle datasets with diverse behavior, and work as a good and promising proxy to the hand-crafted rewards.

Table 3: Experimental results on the AntMaze-v0 and Adroit-v0 domains. SEABO and OTR use IQL as the base algorithm. IQL denotes that IQL uses the ground-truth reward for policy learning. We report the mean normalized scores and the corresponding standard deviations. We bold and highlight the best mean score cell. 

<table><tr><td>Task Name</td><td>IQL</td><td>IQL+OTR</td><td>IQL+SEABO</td></tr><tr><td>umaze</td><td>87.5±2.6</td><td>83.4±3.3</td><td>90.0±1.8</td></tr><tr><td>umaze-diverse</td><td>62.2±13.8</td><td>68.9±13.6</td><td>66.2±7.2</td></tr><tr><td>medium-diverse</td><td>70.0±10.9</td><td>70.4±4.8</td><td>72.2±4.1</td></tr><tr><td>medium-play</td><td>71.2±7.3</td><td>70.5±6.6</td><td>71.6±5.4</td></tr><tr><td>large-diverse</td><td>47.5±9.5</td><td>45.5±6.2</td><td>50.0±6.8</td></tr><tr><td>large-play</td><td>39.6±5.8</td><td>45.3±6.9</td><td>50.8±8.7</td></tr><tr><td>Total Score</td><td>378.0</td><td>384.0</td><td>400.8</td></tr></table>

<table><tr><td>Task Name</td><td>IQL</td><td>IQL+OTR</td><td>IQL+SEABO</td></tr><tr><td>pen-human</td><td>70.7±8.6</td><td>66.8±21.2</td><td>94.3±12.0</td></tr><tr><td>pen-cloned</td><td>37.2±7.3</td><td>46.9±20.9</td><td>48.7±15.3</td></tr><tr><td>door-human</td><td>3.3±1.3</td><td>5.9±2.7</td><td>5.1±2.0</td></tr><tr><td>door-cloned</td><td>1.6±0.5</td><td>0.0±0.0</td><td>0.4±0.8</td></tr><tr><td>relocate-human</td><td>0.1±0.0</td><td>0.1±0.1</td><td>0.4±0.5</td></tr><tr><td>relocate-cloned</td><td>-0.2±0.0</td><td>-0.2±0.0</td><td>-0.2±0.0</td></tr><tr><td>hammer-human</td><td>1.6±0.6</td><td>1.8±1.4</td><td>2.7±1.8</td></tr><tr><td>hammer-cloned</td><td>2.1±1.0</td><td>0.9±0.3</td><td>2.2±0.8</td></tr><tr><td>Total Score</td><td>116.4</td><td>122.2</td><td>153.6</td></tr></table>

Table 4: Comparison of SEABO against imitation learning algorithms. We use IQL as the base algorithm for SEABO and PWIL. PWIL-action means that we concatenate state and action to compute rewards in PWIL. We report the mean performance at the final 10 episodes of evaluation for each algorithm, ± captures the standard deviation. We highlight the best mean score cell. 

<table><tr><td>Task Name</td><td>SQIL</td><td>DemoDICE</td><td>SMODICE</td><td>PWIL-action</td><td>SEABO</td></tr><tr><td>halfcheetah-medium</td><td>31.3±1.8</td><td>42.5±1.7</td><td>41.7±1.0</td><td>44.4±0.2</td><td>44.8±0.3</td></tr><tr><td>hopper-medium</td><td>44.7±20.1</td><td>55.1±3.3</td><td>56.3±2.3</td><td>60.4±1.8</td><td>80.9±3.2</td></tr><tr><td>walker2d-medium</td><td>59.6±7.5</td><td>73.4±2.6</td><td>13.3±9.2</td><td>72.6±6.3</td><td>80.9±0.6</td></tr><tr><td>halfcheetah-medium-replay</td><td>29.3±2.2</td><td>38.1±2.7</td><td>38.7±2.4</td><td>42.6±0.5</td><td>42.3±0.1</td></tr><tr><td>hopper-medium-replay</td><td>45.2±23.1</td><td>39.0±15.4</td><td>44.3±19.7</td><td>94.0±7.0</td><td>92.7±2.9</td></tr><tr><td>walker2d-medium-replay</td><td>36.3±13.2</td><td>52.2±13.1</td><td>44.6±23.4</td><td>41.9±6.0</td><td>74.0±2.7</td></tr><tr><td>halfcheetah-medium-expert</td><td>40.1±6.4</td><td>85.8±5.7</td><td>87.9±5.8</td><td>89.5±3.6</td><td>89.3±2.5</td></tr><tr><td>hopper-medium-expert</td><td>49.8±5.8</td><td>92.3±14.2</td><td>76.0±8.6</td><td>70.9±35.1</td><td>97.5±5.8</td></tr><tr><td>walker2d-medium-expert</td><td>35.9±22.2</td><td>106.9±1.9</td><td>47.8±31.1</td><td>109.8±0.2</td><td>110.9±0.2</td></tr><tr><td>Total Score</td><td>372.2</td><td>585.3</td><td>450.6</td><td>626.1</td><td>713.3</td></tr></table>

# 5.2 COMPARISON AGAINST OFFLINE IL ALGORITHMS

To further show the advantages of SEABO, we additionally compare it against recent strong offline imitation learning approaches, including DemoDICE (Kim et al., 2022b) and SMODICE (Ma et al., 2022). We also convert two online IL algorithms into the offline setting, SQIL (Reddy et al., 2020) and PWIL (Dadashi et al., 2021), where we replace the base algorithm in SQIL with TD3\_BC and utilize IQL as the base algorithm for PWIL. All algorithms are run using their official implementations under the identical experimental setting as SEABO (i.e., one single expert demonstration). For a fair comparison, we involve actions when training discriminators in SMODICE and measuring the distance in PWIL. We use IQL as the base algorithm for SEABO. The empirical results in Table 4 show that IQL+SEABO achieves the best performance on 6 out of 9 datasets, and has the highest total score (surpassing the second highest one by $13.9\%$ ). Though SEABO underperforms PWIL on some datasets, it significantly beats PWIL on tasks like hopper-medium-v2. Note that SMODICE behaves poorly on many tasks, which is also observed in Li et al. (2023).

# 5.3 STATE-ONLY REGIMES

We now examine how SEABO behaves when the expert demonstrations consist of only observations, i.e., $D_{e} = \{\tau_{e}^{i}\}_{i=1}^{M}$ , where M is the size of the demonstration and $\tau = \{s_{0}, s_{1}, \ldots, s_{T}\}$ . In principle, SEABO can also calculate rewards by querying the KD-tree with only states, $(\tilde{s}_{e}, \tilde{s}_{e}^{\prime}) = \text{NearestNeighbor}(\mathcal{D}_{e}, (s, s^{\prime}))$ . The distance can then be calculated with some distance metric D, $d = D((\tilde{s}_{e}, \tilde{s}_{e}^{\prime}), (s, s^{\prime}))$ , and the rewards can be computed accordingly, via Equation 3. For baselines, since DemoDICE and ValueDICE are inapplicable to state-only regimes (Zhu et al., 2021), we compare against LobsDICE (Kim et al., 2022a), which is a state-of-the-art offline IL algorithm that learns from expert observations. We also involve SMODICE, PWIL, and OTR for comparison, and train them using only expert observations. All baselines are run with their official implementation.

Table 5: Experimental results on the state-only regime. SEABO, PWIL, and OTR utilize IQL as the base offline RL algorithm. PWIL-state denotes that PWIL only uses observations to compute rewards. The results are averaged over the final 10 evaluations, and $\pm$ captures the standard deviation. We highlight the cell with the best mean performance. 

<table><tr><td>Task Name</td><td>SMODICE</td><td>LobsDICE</td><td>PWIL-state</td><td>OTR</td><td>SEABO</td></tr><tr><td>halfcheetah-medium</td><td>41.1±2.1</td><td>41.5±1.8</td><td>0.1±0.6</td><td>43.3±0.2</td><td>45.0±0.2</td></tr><tr><td>hopper-medium</td><td>56.5±1.8</td><td>56.9±1.4</td><td>1.4±0.5</td><td>78.7±5.5</td><td>74.7±5.2</td></tr><tr><td>walker2d-medium</td><td>15.5±18.6</td><td>69.3±5.4</td><td>0.2±0.2</td><td>79.4±1.4</td><td>81.3±1.3</td></tr><tr><td>halfcheetah-medium-replay</td><td>39.2±3.1</td><td>39.9±3.1</td><td>-2.4±0.2</td><td>41.3±0.6</td><td>42.4±0.6</td></tr><tr><td>hopper-medium-replay</td><td>55.3±21.4</td><td>41.6±16.8</td><td>0.7±0.2</td><td>84.8±2.6</td><td>88.0±0.7</td></tr><tr><td>walker2d-medium-replay</td><td>37.8±10.2</td><td>33.2±7.0</td><td>-0.2±0.2</td><td>66.0±6.7</td><td>76.4±3.0</td></tr><tr><td>halfcheetah-medium-expert</td><td>88.0±4.0</td><td>89.4±3.2</td><td>0.0±1.0</td><td>89.6±3.0</td><td>91.8±1.5</td></tr><tr><td>hopper-medium-expert</td><td>75.1±11.7</td><td>53.4±3.2</td><td>2.7±2.1</td><td>93.2±20.6</td><td>97.5±6.4</td></tr><tr><td>walker2d-medium-expert</td><td>32.3±14.7</td><td>106.6±2.7</td><td>0.2±0.3</td><td>109.3±0.8</td><td>110.5±0.3</td></tr><tr><td>Total Score</td><td>440.8</td><td>531.8</td><td>2.7</td><td>685.6</td><td>707.6</td></tr></table>

Table 6: Comparison of different choices of search algorithms in SEABO. We report the mean normalized scores with standard deviations. We highlight the best mean score cell except for IQL. 

<table><tr><td>Task Name</td><td>IQL</td><td>SEABO (KD-tree)</td><td>SEABO (Ball-tree)</td><td>SEABO (HNSW)</td></tr><tr><td>halfcheetah-medium</td><td>47.4±0.2</td><td>44.8±0.3</td><td>44.9±0.3</td><td>42.1±0.6</td></tr><tr><td>hopper-medium</td><td>66.2±5.7</td><td>80.9±3.2</td><td>80.7±3.7</td><td>47.2±2.9</td></tr><tr><td>walker2d-medium</td><td>78.3±8.7</td><td>80.9±0.6</td><td>80.8±0.6</td><td>30.7±19.9</td></tr><tr><td>halfcheetah-medium-replay</td><td>44.2±1.2</td><td>42.3±0.1</td><td>42.5±0.3</td><td>26.9±4.2</td></tr><tr><td>hopper-medium-replay</td><td>94.7±8.6</td><td>92.7±2.9</td><td>92.1±2.3</td><td>25.8±7.5</td></tr><tr><td>walker2d-medium-replay</td><td>73.8±7.1</td><td>74.0±2.7</td><td>74.3±2.0</td><td>29.1±10.1</td></tr><tr><td>halfcheetah-medium-expert</td><td>86.7±5.3</td><td>89.3±2.5</td><td>89.2±2.4</td><td>34.5±2.2</td></tr><tr><td>hopper-medium-expert</td><td>91.5±14.3</td><td>97.5±5.8</td><td>96.7±6.2</td><td>41.5±7.7</td></tr><tr><td>walker2d-medium-expert</td><td>109.6±1.0</td><td>110.9±0.2</td><td>110.9±0.1</td><td>108.6±0.8</td></tr><tr><td>Total Score</td><td>692.4</td><td>713.3</td><td>712.1</td><td>386.4</td></tr></table>

tions and single expert demonstration. The results in Table 5 suggest that SEABO outperforms other methods on 8 out of 9 tasks, achieving a total score of 707.6, while LobsDICE and OTR only have a total score of 531.8 and 685.6, respectively. It indicates that SEABO can work quite well regardless of whether the expert demonstrations contain actions, further demonstrating the advantages of SEABO. Note that the failure of PWIL in state-only regimes is also reported in Luo et al. (2023).

# 5.4 COMPARISON OF DIFFERENT SEARCH ALGORITHMS

The most critical component in SEABO is the nearest neighbor search algorithm. It is interesting to check how SEABO performs under different search algorithms. To that end, we build SEABO on top of Ball-tree (Omohundro, 1989; Liu et al., 2006), and HNSW (Hierarchical Navigable Small World graphs, Malkov & Yashunin (2018)). These are widely applied nearest neighbor algorithms, where Ball-tree partitions regions via hyper-spheres and HNSW is a fully graph-based search structure. We allow the single expert demonstration to involve actions (i.e., query with $(s, a, s')$ ), and run all of the variants of SEABO using the same set of hyperparameters for a fair comparison. Empirical results on 9 D4RL locomotion datasets are shown in Table 6. It is interesting to see that SEABO with Ball-tree is competitive with SEABO with KD-tree (their performance differences are minor), while SEABO with HNSW exhibits poor performance on many datasets. This means that the choice of the search algorithm counts in SEABO, and simply employing KD-tree can already guarantee good performance. Please see more discussions in Appendix C.

# 5.5 PARAMETER STUDY

It is vital to examine how sensitive SEABO is to the introduced hyperparameters. Due to the space limit, we can only report part of the experiments here and defer more experiments to Appendix B.3.

![](images/69f51e9e1eacdfb053ee5b9d2614a20c336f83ba0aa37c0b001fb5c61f6a9d2e.jpg)

Figure 3: Parameter study on the reward scale. The shaded region denotes the standard deviation.   
![](images/4f3ce469e8d8ee434b1267b6580f4104085843974f7c1bc93906cfa5f7a839db.jpg)

<details>
<summary>line</summary>

| Model | β = 0.1 | β = 0.5 | β = 1 | β = 5 |
| --- | --- | --- | --- | --- |
| halfcheetah-medium-replay-v2 | ~42 | ~43 | ~44 | ~38 |
| hopper-medium-v2 | ~70 | ~75 | ~80 | ~65 |
</details>

(a) Weighting coefficient $\beta$ .

![](images/39dd59ac66449dff6ab27a9432aecb5445b6ba33da396d69ebe7f8e7898f177e.jpg)

<details>
<summary>line</summary>

| Model | N | Normalized Score at 0.2 Million Steps | Normalized Score at 0.4 Million Steps | Normalized Score at 0.6 Million Steps | Normalized Score at 0.8 Million Steps | Normalized Score at 1.0 Million Steps |
|-------|---|----------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|
| halfcheetah-medium-replay-v2 | 1 | ~42 | ~43 | ~43 | ~43 | ~43 |
| halfcheetah-medium-replay-v2 | 5 | ~42 | ~43 | ~43 | ~43 | ~43 |
| halfcheetah-medium-replay-v2 | 10 | ~42 | ~43 | ~43 | ~43 | ~43 |
| hopper-medium-v2 | 1 | ~65 | ~70 | ~75 | ~78 | ~78 |
| hopper-medium-v2 | 5 | ~60 | ~65 | ~70 | ~75 | ~75 |
| hopper-medium-v2 | 10 | ~65 | ~70 | ~75 | ~80 | ~80 |
</details>

(b) Number of neighbors $N$ .   
Figure 4: Parameter study of (a) weighting coefficient $\beta$ , (b) number of neighbors N. The shaded region captures the standard deviation.

Reward scale $\alpha$ . $\alpha$ controls the scale of the resulting rewards. To check its influence, we conduct experiments on three datasets from D4RL locomotion tasks and sweep $\alpha$ across $\{1, 5, 10\}$ . Results in Figure 3 demonstrate that the best $\alpha$ may depend on the dataset while a smaller $\alpha$ is preferred.

Weighting coefficient $\beta$ . $\beta$ is probably the most critical hyperparameter which decides the scale of the distance. In Figure 4(a), we vary $\beta$ across $\{0.1, 0.5, 1, 5\}$ , and find that the performance drops with too small or too large $\beta$ . It seems that $\beta = 0.5$ or $\beta = 1$ can achieve a good trade-off.

Number of neighbors N. To see whether the number of neighbors N matters, we run IQL+SEABO with $N \in \{1, 5, 10\}$ . Results in Figure 4(b) show that SEABO is robust to this hyperparameter.

Number of expert demonstrations K. We investigate whether increasing the number of expert demonstrations can further boost the performance of SEABO and baselines by running experiments of these methods on 9 MuJoCo locomotion tasks. We report the aggregate performance (i.e., total score) in Table 7. One can see that all methods enjoy performance improvement when K = 10, while none of them can outperform SEABO (there still exists a large performance gap).

Table 7: Comparison of SEABO against baseline algorithms under different amounts of expert demonstrations. We report the aggregate performances and bold the best one. 

<table><tr><td># demo</td><td>DemoDICE</td><td>IQL+ORIL</td><td>IQL+UDS</td><td>IQL+OTR</td><td>IQL+PWIL</td><td>IQL+SEABO</td></tr><tr><td> $K = 1$ </td><td>585.3</td><td>588.5</td><td>495.1</td><td>685.6</td><td>626.1</td><td>713.3</td></tr><tr><td> $K = 10$ </td><td>589.3</td><td>618.3</td><td>575.8</td><td>694.2</td><td>638.0</td><td>716.1</td></tr></table>

# 6 CONCLUSION

In this paper, we propose a novel search-based offline imitation learning method, dubbed SEABO, that annotates the unlabeled offline trajectories in an unsupervised learning manner. SEABO builds a KD-tree using the expert demonstration(s), and searches the nearest neighbors of the query sample. We then measure their distance and output the reward signal via a squashing function. SEABO is easy to implement and can be incorporated with any offline RL algorithm. Experiments on D4RL datasets show that SEABO can incur competitive or even better offline policies than pre-defined reward functions. SEABO can also function well if the expert demonstrations are made up of only observations. For future work, it is interesting to apply SEABO in visual offline RL datasets (e.g., Lu et al. (2022b)), or adapt SEABO to cross-domain offline imitation learning tasks.

# ACKNOWLEDGEMENTS

This work was supported by the STI 2030-Major Projects under Grant 2021ZD0201404 and NSFC under Grant 62250068. The authors would like to thank the anonymous reviewers for their valuable comments and advice.

# REFERENCES

Saurabh Arora and Prashant Doshi. A Survey of Inverse Reinforcement Learning: Challenges, Methods and Progress. Artificial Intelligence, 297:103500, 2021.   
Chenjia Bai, Lingxiao Wang, Zhuoran Yang, Zhi-Hong Deng, Animesh Garg, Peng Liu, and Zhao-ran Wang. Pessimistic Bootstrapping for Uncertainty-Driven Offline Reinforcement Learning. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=Y4cs1Z3HnqL.   
Nir Baram, Oron Anschel, Itai Caspi, and Shie Mannor. End-to-End Differentiable Adversarial Imitation Learning. In International Conference on Machine Learning, 2017.   
Jon Louis Bentley. Multidimensional binary search trees used for associative searching. Communications of the ACM, 18(9):509–517, 1975.   
Abdeslam Boularias, Jens Kober, and Jan Peters. Relative Entropy Inverse Reinforcement Learning. In International Conference on Artificial Intelligence and Statistics, 2011.   
Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, P. Abbeel, A. Srinivas, and Igor Mordatch. Decision Transformer: Reinforcement Learning via Sequence Modeling. ArXiv, abs/2106.01345, 2021.   
Ching-An Cheng, Tengyang Xie, Nan Jiang, and Alekh Agarwal. Adversarially Trained Actor Critic for Offline Reinforcement Learning. In Proceedings of the 39th International Conference on Machine Learning, 2022. URL https://proceedings.mlr.press/v162/cheng22b.html.   
Kamil Ciosek. Imitation Learning by Reinforcement Learning. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=1zwleytEpYx.   
Samuel Cohen, Brandon Amos, Marc Peter Deisenroth, Mikael Henaff, Eugene Vinitsky, and Denis Yarats. Imitation Learning from Pixel Observations for Continuous Control, 2022. URL https://openreview.net/forum?id=JLbXkHkLCG6.   
Robert Dadashi, Leonard Hussenot, Matthieu Geist, and Olivier Pietquin. Primal Wasserstein Imitation Learning. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=TtYSU29zgR.   
Jonas Degrave, Federico Felici, Jonas Buchli, Michael Neunert, Brendan Tracey, Francesco Carpanese, Timo Ewalds, Roland Hafner, Abbas Abdolmaleki, Diego de Las Casas, et al. Magnetic control of tokamak plasmas through deep reinforcement learning. Nature, 602(7897):414–419, 2022.   
Branton DeMoss, Paul Duckworth, Nick Hawes, and Ingmar Posner. DITTO: Offline Imitation Learning with World Models. ArXiv, abs/2302.03086, 2023.   
Edsger W. Dijkstra. A note on two problems in connexion with graphs. Numerische Mathematik, 1:269–271, 1959.   
James E Doran and Donald Michie. Experiments with the graph traverser program. Proceedings of the Royal Society of London. Series A. Mathematical and Physical Sciences, 294(1437):235–259, 1966.   
Stefan Edelkamp and Stefan Schrödl. Heuristic search: theory and applications. Elsevier, 2011.

Gideon Joseph Freund, Elad Sarafian, and Sarit Kraus. A Coupled Flow Approach to Imitation Learning. In International Conference on Machine Learning, 2023.   
Justin Fu, Aviral Kumar, Ofir Nachum, G. Tucker, and Sergey Levine. D4RL: Datasets for Deep Data-Driven Reinforcement Learning. ArXiv, abs/2004.07219, 2020.   
Scott Fujimoto and Shixiang Shane Gu. A Minimalist Approach to Offline Reinforcement Learning. In Advances in Neural Information Processing Systems, 2021.   
Scott Fujimoto, David Meger, and Doina Precup. Off-Policy Deep Reinforcement Learning without Exploration. In International Conference on Machine Learning, 2019.   
Seyed Kamyar Seyed Ghasemipour, Richard S. Zemel, and Shixiang Shane Gu. A Divergence Minimization Perspective on Imitation Learning Methods. In Conference on Robot Learning, 2019.   
Abhishek Gupta, Vikash Kumar, Corey Lynch, Sergey Levine, and Karol Hausman. Relay Policy Learning: Solving Long-Horizon Tasks via Imitation and Reinforcement Learning. ArXiv, abs/1910.11956, 2019.   
Peter E Hart, Nils J Nilsson, and Bertram Raphael. A formal basis for the heuristic determination of minimum cost paths. IEEE transactions on Systems Science and Cybernetics, 4(2):100–107, 1968.   
Minho Heo, Youngwoon Lee, Doohyun Lee, and Joseph J. Lim. FurnitureBench: Reproducible Real-World Benchmark for Long-Horizon Complex Manipulation. ArXiv, abs/2305.12821, 2023.   
Jonathan Ho and Stefano Ermon. Generative Adversarial Imitation Learning. In Advances in Neural Information Processing Systems, 2016.   
Matthew W. Hoffman, Bobak Shahriari, John Aslanides, Gabriel Barth-Maron, Feryal M. P. Behbahani, Tamara Norman, Abbas Abdolmaleki, Albin Cassirer, Fan Yang, Kate Baumli, Sarah Henderson, Alexander Novikov, Sergio Gomez Colmenarejo, Serkan Cabi, Caglar Gulcehre, Tom Le Paine, Andrew Cowie, Ziyun Wang, Bilal Piot, and Nando de Freitas. Acme: A Research Framework for Distributed Reinforcement Learning. ArXiv, abs/2006.00979, 2020. URL https://api.semanticscholar.org/CorpusID:219176679.   
Michael Janner, Qiyang Li, and Sergey Levine. Offline Reinforcement Learning as One Big Sequence Modeling Problem. In Advances in Neural Information Processing Systems, 2021.   
Daniel Jarrett, Ioana Bica, and Mihaela van der Schaar. Strictly Batch Imitation Learning by Energy-based Distribution Matching. In Advances in neural information processing systems, 2020.   
Wonseok Jeon, Seokin Seo, and Kee-Eung Kim. A Bayesian Approach to Generative Adversarial Imitation Learning. In Neural Information Processing Systems, 2018.   
Liyiming Ke, Matt Barnes, Wen Sun, Gilwoo Lee, Sanjiban Choudhury, and Siddhartha S. Srinivasa. Imitation Learning as f-Divergence Minimization. In Workshop on the Algorithmic Foundations of Robotics, 2019.   
Rahul Kidambi, Aravind Rajeswaran, Praneeth Netrapalli, and Thorsten Joachims. MOReL: Model-Based Offline Reinforcement Learning. In Advances in Neural Information Processing Systems, 2020.   
Geon-Hyeong Kim, Jongmin Lee, Youngsoo Jang, Hongseok Yang, and Kee-Eung Kim. Lobs-DICE: Offline Learning from Observation via Stationary Distribution Correction Estimation. In Advances in Neural Information Processing Systems, 2022a. URL https://openreview.net/forum?id=8U5J6zK\_MtV.   
Geon-Hyeong Kim, Seokin Seo, Jongmin Lee, Wonseok Jeon, HyeongJoo Hwang, Hongseok Yang, and Kee-Eung Kim. DemoDICE: Offline Imitation Learning with Supplementary Imperfect Demonstrations. In International Conference on Learning Representations, 2022b. URL https://openreview.net/forum?id=BrPdX1bDZkQ.

Kuno Kim, Akshat Jindal, Yang Song, Jiaming Song, Yanan Sui, and Stefano Ermon. Imitation with Neural Density Models. In Neural Information Processing Systems, 2020.   
Diederik P. Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization. In International Conference on Learning Representation, 2015.   
Jens Kober, J Andrew Bagnell, and Jan Peters. Reinforcement learning in robotics: A survey. The International Journal of Robotics Research, 32(11):1238–1274, 2013.   
Richard E Korf. Depth-first iterative-deepening: An optimal admissible tree search. Artificial intelligence, 27(1):97–109, 1985.   
Richard E. Korf. Artificial Intelligence Search Algorithms. In Algorithms and Theory of Computation Handbook, 1999.   
Ilya Kostrikov, Kumar Krishna Agrawal, Debidatta Dwibedi, Sergey Levine, and Jonathan Tompson. Discriminator-Actor-Critic: Addressing Sample Inefficiency and Reward Bias in Adversarial Imitation Learning. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=Hk4fpoA5Km.   
Ilya Kostrikov, Ofir Nachum, and Jonathan Tompson. Imitation Learning via Off-Policy Distribution Matching. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=Hyg-JC4FDr.   
Ilya Kostrikov, Ashvin Nair, and Sergey Levine. Offline Reinforcement Learning with Implicit Q-Learning. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=68n2s9ZJWF8.   
Aviral Kumar, Aurick Zhou, G. Tucker, and Sergey Levine. Conservative Q-Learning for Offline Reinforcement Learning. In Advances in Neural Information Processing Systems, 2020.   
Sascha Lange, Thomas Gabel, and Martin A. Riedmiller. Batch Reinforcement Learning. In Reinforcement Learning, 2012.   
Youngwoon Lee, Edward S. Hu, Zhengyu Yang, Alexander Yin, and Joseph J. Lim. IKEA Furniture Assembly Environment for Long-Horizon Complex Manipulation Tasks. 2021 IEEE International Conference on Robotics and Automation (ICRA), pp. 6343–6349, 2019.   
Youngwoon Lee, Joseph J. Lim, Anima Anandkumar, and Yuke Zhu. Adversarial Skill Chaining for Long-Horizon Robot Manipulation via Terminal State Regularization. In Conference on Robot Learning, 2021.   
Sergey Levine, Aviral Kumar, G. Tucker, and Justin Fu. Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems. ArXiv, abs/2005.01643, 2020.   
Anqi Li, Byron Boots, and Ching-An Cheng. MAHALO: Unifying Offline Reinforcement Learning and Imitation Learning from Observations. In International Conference on Machine Learning, 2023.   
Ting Liu, Andrew W Moore, Alexander Gray, and Claire Cardie. New Algorithms for Efficient High-Dimensional Nonparametric Classification. Journal of Machine Learning Research, 7(6), 2006.   
Cong Lu, Philip Ball, Jack Parker-Holder, Michael Osborne, and Stephen J. Roberts. Revisiting Design Choices in Offline Model Based Reinforcement Learning. In International Conference on Learning Representations, 2022a. URL https://openreview.net/forum?id=zz9hXVhf40.   
Cong Lu, Philip J Ball, Tim GJ Rudner, Jack Parker-Holder, Michael A Osborne, and Yee Whye Teh. Challenges and Opportunities in Offline Reinforcement Learning from Visual Observations. arXiv preprint arXiv:2206.04779, 2022b.

Yicheng Luo, zhengyao jiang, Samuel Cohen, Edward Grefenstette, and Marc Peter Deisenroth. Optimal Transport for Offline Imitation Learning. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=MhuFzFsrfvH.   
Jiafei Lyu, aicheng Gong, Le Wan, Zongqing Lu, and Xiu Li. State Advantage Weighting for Offline RL. In 3rd Offline RL Workshop: Offline RL as a "Launchpad", 2022a. URL https://openreview.net/forum?id=2rOD\_UQfvl.   
Jiafei Lyu, Xiu Li, and Zongqing Lu. Double Check Your State Before Trusting It: Confidence-Aware Bidirectional Offline Model-Based Imagination. In Advances in Neural Information Processing Systems, 2022b. URL https://openreview.net/forum?id=3e3IQMLDSLP.   
Jiafei Lyu, Xiaoteng Ma, Xiu Li, and Zongqing Lu. Mildly Conservative Q-Learning for Offline Reinforcement Learning. In Advances in Neural Information Processing Systems, 2022c. URL https://openreview.net/forum?id=VYYf6S67pQc.   
Yecheng Jason Ma, Andrew Shen, Dinesh Jayaraman, and Osbert Bastani. Versatile Offline Imitation from Observations and Examples via Regularized State-Occupancy Matching. In International Conference on Machine Learning, 2022.   
Yu A Malkov and Dmitry A Yashunin. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE transactions on pattern analysis and machine intelligence, 42(4):824–836, 2018.   
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin A. Riedmiller, Andreas Kirkeby Fidjeland, Georg Ostrovski, Stig Petersen, Charlie Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. Human-level Control through Deep Reinforcement Learning. Nature, 518:529–533, 2015. URL https://api.semanticscholar.org/CorpusID:205242740.   
Ofir Nachum, Yinlam Chow, Bo Dai, and Lihong Li. DualDICE: Behavior-Agnostic Estimation of Discounted Stationary Distribution Corrections. In Advances in neural information processing systems, 2019.   
Stephen M Omohundro. Five balltree construction algorithms. Technical report, International Computer Science Institute, 1989.   
Jyothish Pari, Nur Muhammad (Mahi) Shafiullah, Sridhar Pandian Arunachalam, and Lerrel Pinto. The surprising effectiveness of representation learning for visual imitation. ArXiv, abs/2112.01511, 2021.   
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Neural Information Processing Systems, 2019.   
F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830, 2011.   
Ira Pohl. Heuristic search viewed as path finding in a graph. Artificial intelligence, 1(3-4):193–204, 1970.   
Dean A Pomerleau. Alvinn: An autonomous land vehicle in a neural network. In Advances in Neural Information Processing Systems, 1988.   
Yuhang Ran, Yichen Li, Fuxiang Zhang, Zongzhang Zhang, and Yang Yu. Policy Regularization with Dataset Constraint for Offline Reinforcement Learning. In International Conference on Machine Learning, 2023.

Siddharth Reddy, Anca D. Dragan, and Sergey Levine. {SQIL}: Imitation Learning via Reinforcement Learning with Sparse Rewards. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=S1xKd24twB.   
Marc Rigter, Bruno Lacerda, and Nick Hawes. RAMBO-RL: Robust Adversarial Model-Based Offline Reinforcement Learning. In Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=nrksGSRT7kX.   
Stéphane Ross, Geoffrey J. Gordon, and J. Andrew Bagnell. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. In International Conference on Artificial Intelligence and Statistics, 2011.   
Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, et al. Mastering atari, go, chess and shogi by planning with a learned model. Nature, 588(7839):604–609, 2020.   
Mark E Stickel and Mabry Tyson. An analysis of consecutively bounded depth-first search with applications in automated deduction. In IJCAI, pp. 1073–1075, 1985.   
Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT press, 2018.   
Larry A. Taylor and Richard E. Korf. Pruning Duplicate Nodes in Depth-First Search. In AAAI Conference on Artificial Intelligence, 1993.   
Emanuel Todorov, Tom Erez, and Yuval Tassa. MuJoCo: A Physics Engine for Model-based Control. IEEE/RSJ International Conference on Intelligent Robots and Systems, 2012.   
Faraz Torabi, Garrett Warnell, and Peter Stone. Recent Advances in Imitation Learning from Observation. In International Joint Conference on Artificial Intelligence, 2019.   
Masatoshi Uehara and Wen Sun. Pessimistic Model-based Offline Reinforcement Learning under Partial Coverage. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=tyrJsbKAe6.   
Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, Ilhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17:261–272, 2020.   
Ruohan Wang, Carlo Ciliberto, Pierluigi Vito Amadori, and Y. Demiris. Support-weighted Adversarial Imitation Learning. ArXiv, abs/2002.08803, 2020.   
Haoran Xu, Xianyuan Zhan, Honglei Yin, and Huiling Qin. Discriminator-Weighted Offline Imitation Learning from Suboptimal Demonstrations. In International Conference on Machine Learning, 2022.   
Kai Yang, Jian Tao, Jiafei Lyu, and Xiu Li. Exploration and anti-exploration with distributional random network distillation. ArXiv, abs/2401.09750, 2024.   
Lantao Yu, Tianhe Yu, Jiaming Song, Willie Neiswanger, and Stefano Ermon. Offline Imitation Learning with Suboptimal Demonstrations via Relaxed Distribution Matching. In AAAI Conference on Artificial Intelligence, 2023.   
Tianhe Yu, Garrett Thomas, Lantao Yu, Stefano Ermon, James Y. Zou, Sergey Levine, Chelsea Finn, and Tengyu Ma. MOPO: Model-based Offline Policy Optimization. In Advances in Neural Information Processing Systems, 2020.   
Tianhe Yu, Aviral Kumar, Rafael Rafailov, Aravind Rajeswaran, Sergey Levine, and Chelsea Finn. COMBO: Conservative Offline Model-Based Policy Optimization. In Advances in Neural Information Processing Systems, 2021.

Tianhe Yu, Aviral Kumar, Yevgen Chebotar, Karol Hausman, Chelsea Finn, and Sergey Levine. How to Leverage Unlabeled Data in Offline Reinforcement Learning. In International Conference on Machine Learning, 2022.   
Sheng Yue, Guanbo Wang, Wei Shao, Zhaofeng Zhang, Sen Lin, Ju Ren, and Junshan Zhang. CLARE: Conservative Model-Based Reward Learning for Offline Inverse Reinforcement Learning. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=5aT4ganOd98.   
Junjie Zhang, Jiafei Lyu, Xiaoteng Ma, Jiangpeng Yan, Jun Yang, Le Wan, and Xiu Li. Uncertainty-driven Trajectory Truncation for Model-based Offline Reinforcement Learning. ArXiv, abs/2304.04660, 2023.   
Wenxuan Zhou, Sujay Bajracharya, and David Held. PLAS: Latent Action Space for Offline Reinforcement Learning. In Conference on Robot Learning, 2020.   
Zhuangdi Zhu, Kaixiang Lin, Bo Dai, and Jiayu Zhou. Off-Policy Imitation Learning from Observations. In Neural Information Processing Systems, 2021.   
Brian D. Ziebart, Andrew L. Maas, J. Andrew Bagnell, and Anind K. Dey. Maximum Entropy Inverse Reinforcement Learning. In AAAI Conference on Artificial Intelligence, 2008.   
Konrad Zolna, Alexander Novikov, Ksenia Konyushkova, Caglar Gulcehre, Ziyun Wang, Yusuf Aytar, Misha Denil, Nando de Freitas, and Scott E. Reed. Offline Learning from Demonstrations and Unlabeled Experience. ArXiv, abs/2011.13885, 2020.

# A HYPERPARAMETER SETUP

In this section, we detail the hyperparameter setup utilized in our experiments. We conduct experiments on 9 MuJoCo locomotion “-v2” medium-level datasets, 6 AntMaze “-v0” datasets, and 8 Adroit “-v0” datasets, yielding a total of 23 tasks. We list the hyperparameter setup for IQL and TD3\_BC on MuJoCo locomotion tasks in Table 8. We keep the hyperparameter setup of the base offline RL algorithms unchanged for both IQL and TD3\_BC. For IQL, we do not rescale the rewards in the datasets by $1000/\text{max\_return}-\text{min\_return}$ , as we have an additional hyperparameter $\alpha$ to control the reward scale. In practice, we find minor performance differences if we rescale the rewards. We generally utilize the same formula of squashing function for most of the datasets, except that we set $\beta = 1$ in hopper-medium-replay-v2, and $\alpha = 10$ , $\beta = 0.1$ in hopper-medium-expert-v2 for better performance. Note that using $\alpha = 1$ , $\beta = 0.5$ on these tasks can also produce a good performance (e.g., setting $\alpha = 1$ , $\beta = 0.5$ on hopper-medium-replay-v2 leads to an average performance of 87.2, still outperforming strong baselines like OTR), while slightly modifying the hyperparameter setup can result in better performance. We divide the scaled distance by the action dimension of the task to strike a balance between different tasks (as we use one set of hyperparameters). This is also adopted in PWIL paper (Dadashi et al., 2021). For TD3\_BC, we use the same type of squashing function as IQL on the locomotion tasks, with $\alpha = 1$ , $\beta = 0.5$ , except that we use $\alpha = 10$ for walker2d-medium-v2 and walker2d-medium-replay-v2 for slightly better performance. We use the official implementation of TD3\_BC (https://github.com/sfujim/TD3\_BC) and adopt the PyTorch (Paszke et al. (2019)) version of IQL for evaluation.

Table 8: Hyperparameter setup of SEABO on locomotion tasks, with IQL and TD3\_BC as the base offline RL algorithms. 

<table><tr><td></td><td>Hyperparameter</td><td>Value</td></tr><tr><td rowspan="8">Shared Configurations</td><td>Hidden layers</td><td>(256, 256)</td></tr><tr><td>Discount factor</td><td>0.99</td></tr><tr><td>Actor learning rate</td><td> $3 \times 10^{-4}$ </td></tr><tr><td>Critic learning rate</td><td> $3 \times 10^{-4}$ </td></tr><tr><td>Batch size</td><td>256</td></tr><tr><td>Optimizer</td><td>Adam (Kingma &amp; Ba, 2015)</td></tr><tr><td>Target update rate</td><td> $5 \times 10^{-3}$ </td></tr><tr><td>Activation function</td><td>ReLU</td></tr><tr><td rowspan="3">IQL</td><td>Value learning rate</td><td> $3 \times 10^{-4}$ </td></tr><tr><td>Temperature</td><td>3.0</td></tr><tr><td>Expectile</td><td>0.7</td></tr><tr><td rowspan="4">TD3_BC</td><td>Policy noise</td><td>0.2</td></tr><tr><td>Policy noise clipping</td><td>(-0.5, 0.5)</td></tr><tr><td>Policy update frequency</td><td>2</td></tr><tr><td>Normalization weight</td><td>2.5</td></tr><tr><td rowspan="4">SEABO</td><td>Squashing function</td><td> $r = \exp(-\frac{0.5 \times d}{|A|})$ </td></tr><tr><td>Distance measurement</td><td>Euclidean distance</td></tr><tr><td>Number of neighbors</td><td>1</td></tr><tr><td>Number of expert demonstrations</td><td>1</td></tr></table>

We summarize the hyperparameter setup of SEABO (using IQL as the underlying algorithm) on the AntMaze domain and Adroit domain in Table 9 and Table 10, respectively. We only list the different hyperparameters in these tables and the other hyperparameters follow those presented in Table 8. Note that we filter the highest return trajectory as the expert demonstration in the Adroit domain, while selecting the goal-reached trajectory as the expert demonstration in the AntMaze domain, which is also adopted in OTR paper (Luo et al., 2023). We adopt a comparatively large $\beta = 5$ on AntMaze tasks. We also follow the IQL paper (Kostrikov et al., 2022) to subtract 1 from the rewards, which we find can result in better performance. For Adroit tasks, we remove the action dimension in the squashing function, since these tasks have large action space dimensions. If one insists on involving $|\mathcal{A}|$ , a much larger $\beta$ than 0.5 is then necessary to mitigate its influence. We find

Table 9: Hyperparameter setup of SEABO on AntMaze tasks, with IQL as the base offline RL algorithm. 

<table><tr><td></td><td>Hyperparameter</td><td>Value</td></tr><tr><td rowspan="2">IQL</td><td>Temperature</td><td>10.0</td></tr><tr><td>Expectile</td><td>0.9</td></tr><tr><td>SEABO</td><td>Squashing function</td><td> $r = \exp(-\frac{5 \times d}{|\mathcal{A}|}) - 1$ </td></tr></table>

Table 10: Hyperparameter setup of SEABO on Adroit tasks, with IQL as the base offline RL algorithm. 

<table><tr><td></td><td>Hyperparameter</td><td>Value</td></tr><tr><td rowspan="3">IQL</td><td>Temperature</td><td>0.5</td></tr><tr><td>Expectile</td><td>0.7</td></tr><tr><td>Actor dropout rate</td><td>0.1</td></tr><tr><td>SEABO</td><td>Squashing function</td><td> $r = \exp(-0.5 \times d)$ </td></tr></table>

that simply removing $|A|$ can ensure quite good performance on all of the evaluated Adroit datasets. Note that OTR (Luo et al., 2023) also adopts different forms of squashing functions for different domains. We query with $(s, s')$ for Adroit tasks and $(s, a, s')$ for other domains.

To acquire expert demonstrations, we use the trajectory with the highest return as expert demonstrations on MuJoCo locomotion tasks and Adroit tasks, and filter the goal-reached trajectory in AntMaze tasks. For all of the baseline reward learning and offline imitation learning algorithms, we follow this setting and run them with their official codebases $^{2}$ over five different random seeds. We use the PWIL implementation from Acme (Hoffman et al., 2020) $^{3}$ .

In SEABO, we use the KD-tree implementation from the scipy library (Virtanen et al., 2020), i.e., scipy.spatial.KDTree. We set the number of nearest neighbors N = 1, and keep other default hyperparameters in KD-tree. Note that we can directly get the desired distance by querying the KD-tree. For Ball-tree, we use its implementation in the scikit-learn package (Pedregosa et al., 2011), i.e., sklearn.neighbors.BallTree. We also keep its original hyperparameters unchanged. For HNSW, we use its implementation in hnswlib $^{4}$ . We use the suggested hyperparameter setting in its GitHub page and set ef\_construction=200 (which defines a construction time/accuracy trade-off) and M=16 (which defines the maximum number of outgoing connections in the graph). All these search algorithms adopt the Euclidean distance as the distance measurement.

In our experiments, we use MuJoCo 2.0 (Todorov et al., 2012) with Gym version 0.18.3, PyTorch (Paszke et al., 2019) version 1.8. We use the normalized score metric recommended in the D4RL paper (Fu et al., 2020), where 0 corresponds to a random policy, and 100 corresponds to an expert policy. Formally, suppose we get the average return J by deploying the learned policy in the test environment, the normalized score gives:

$$
\text { Normalized   score } = \frac {J - J _ {r}}{J _ {e} - J _ {r}} \times 1 0 0, \tag {4}
$$

where $J_{r}$ is the return of a random policy, and $J_{e}$ is the return of an expert policy.

# B MISSING EXPERIMENTAL RESULTS

In this section, we present the missing experimental results from the main text due to the space limit.

Table 11: Comparison of SEABO against baseline algorithms under 10 expert demonstrations. We use IQL as the base algorithm for SEABO, PWIL, and OTR. We report the mean performance at the final 10 evaluations for each algorithm, and ± captures the standard deviation. 

<table><tr><td>Task Name</td><td>IQL</td><td>PWIL-state</td><td>PWIL-action</td><td>OTR-state</td><td>OTR-action</td><td>SEABO</td></tr><tr><td>halfcheetah-medium</td><td>47.4±0.2</td><td>1.6±1.2</td><td>47.5±0.2</td><td>43.1±0.3</td><td>43.4±0.3</td><td>44.4±0.2</td></tr><tr><td>hopper-medium</td><td>66.2±5.7</td><td>2.1±1.3</td><td>70.4±4.2</td><td>80.0±5.2</td><td>75.4±4.6</td><td>81.4±3.5</td></tr><tr><td>walker2d-medium</td><td>78.3±8.7</td><td>0.9±1.3</td><td>81.9±1.0</td><td>79.2±1.3</td><td>79.7±1.2</td><td>81.1±0.7</td></tr><tr><td>halfcheetah-medium-replay</td><td>44.2±1.2</td><td>-2.3±0.5</td><td>44.6±1.1</td><td>41.6±0.3</td><td>41.9±0.3</td><td>43.9±0.2</td></tr><tr><td>hopper-medium-replay</td><td>94.7±8.6</td><td>1.4±1.2</td><td>89.7±4.9</td><td>84.4±1.8</td><td>85.3±1.1</td><td>86.4±1.4</td></tr><tr><td>walker2d-medium-replay</td><td>73.8±7.1</td><td>-0.1±0.2</td><td>72.2±10.6</td><td>71.8±3.8</td><td>69.1±4.6</td><td>78.0±0.7</td></tr><tr><td>halfcheetah-medium-expert</td><td>86.7±5.3</td><td>-0.3±1.5</td><td>88.6±4.3</td><td>87.9±3.4</td><td>88.3±5.1</td><td>90.5±2.5</td></tr><tr><td>hopper-medium-expert</td><td>91.5±14.3</td><td>1.5±0.6</td><td>32.9±25.0</td><td>96.6±21.5</td><td>86.6±22.9</td><td>100.0±7.0</td></tr><tr><td>walker2d-medium-expert</td><td>109.6±1.0</td><td>1.0±1.9</td><td>110.2±0.2</td><td>109.6±0.5</td><td>109.2±0.5</td><td>110.4±0.6</td></tr><tr><td>Total Score</td><td>692.4</td><td>5.8</td><td>638.0</td><td>694.2</td><td>678.9</td><td>716.1</td></tr></table>

# B.1 NUMERICAL COMPARISON UNDER TEN EXPERT DEMONSTRATIONS

In Section 5.5, we present the comparison results of SEABO and baseline reward learning and offline IL algorithms under different numbers of expert demonstrations $K \in \{1, 10\}$ . However, we only report the aggregate performance (i.e., the total score) on the 9 MuJoCo locomotion medium-level tasks (medium, medium-replay, medium-expert) in Table 7. To make the comparison clearer, we present the detailed normalized scores of these methods under $K = 10$ on different datasets in Table 11, where we mainly compare SEABO against different variants of PWIL and OTR. SEABO computes the rewards with actions involved in the single expert demonstration here.

The results reveal that SEABO outperforms baseline methods on 5 out of 9 datasets and is competitive with baselines on the rest of the datasets. SEABO achieves a total score of 716.1, surpassing the second best method (OTR-state) by $3.2\%$ . Though we observe that PWIL-action beats SEABO on datasets like halfcheetah-medium-v2, it can perform poorly on datasets like hopper-medium-expert-v2. We also note that the performance of PWIL deteriorates in the state-only regimes, i.e., learning from pure expert observations. This phenomenon is also reported in Dadashi et al. (2021); Luo et al. (2023). SEABO, instead, is flexible and can be applied regardless of whether the expert demonstrations contain actions.

Furthermore, we show in Table 12 the results of IQL+SEABO on AntMaze and Adroit datasets when 10 expert demonstrations are provided. We compare IQL+SEABO against IQL+OTR and IQL with raw rewards (denoted as IQL). The results demonstrate that SEABO can recover the performance of the offline RL algorithm with ground-truth rewards and sometimes yield better performance. This advantage is agnostic to the number of expert demonstrations K.

IQL+SEABO matches the performance of IQL+OTR on many AntMaze tasks and outperforms IQL+OTR on 6 out of 8 datasets from the Adroit domain. On both the AntMaze domain and Adroit domain, OTR underperforms SEABO in terms of the total score. One may notice that the performance of IQL+SEABO decreases with more expert demonstrations, mainly on the Adroit datasets. This is caused by the performance drop on pen-human-v0, which dominate the total score (the magnitude of its score is much larger than those of other datasets). One can also observe that the performance of IQL+OTR declines on many Adroit tasks, given 10 expert demonstrations (see Table 7 in Luo et al. (2023)). Still, IQL+SEABO exhibits strong performance across numerous datasets.

# B.2 COMPARISON OF TD3\_BC+OTR AND TD3\_BC+SEABO

Since the majority of the experiments in the main text are conducted using IQL as the base offline RL algorithm, it is interesting to see how SEABO competes against baseline methods with another offline RL algorithm as the base method. To that end, we choose TD3\_BC and incorporate it with the strong baseline method, OTR. We follow the experimental setting utilized in the main text, filter a single expert demonstration with the highest return in the offline dataset, and deem it as the expert demonstration. We run TD3\_BC+SEABO and TD3\_BC+OTR on 9 D4RL MuJoCo locomotion datasets. We follow our experimental setup specified in Appendix A, and use the default hyperpa-

Table 12: Experimental results of SEABO on the AntMaze-v0 and Adroit-v0 domains with 10 expert demonstrations. SEABO and OTR use IQL as the base algorithm. The average normalized scores along with the corresponding standard deviations are reported. We bold and highlight the best mean score cell. 

<table><tr><td>Task Name</td><td>IQL</td><td>IQL+OTR</td><td>IQL+SEABO</td></tr><tr><td>umaze</td><td>87.5±2.6</td><td>88.7±3.5</td><td>87.6±2.0</td></tr><tr><td>umaze-diverse</td><td>62.2±13.8</td><td>64.4±18.2</td><td>70.0±9.5</td></tr><tr><td>medium-diverse</td><td>70.0±10.9</td><td>70.5±6.9</td><td>70.2±5.4</td></tr><tr><td>medium-play</td><td>71.2±7.3</td><td>72.7±6.2</td><td>72.8±1.6</td></tr><tr><td>large-diverse</td><td>47.5±9.5</td><td>50.7±6.9</td><td>50.0±7.9</td></tr><tr><td>large-play</td><td>39.6±5.8</td><td>51.2±7.1</td><td>48.6±9.8</td></tr><tr><td>Total Score</td><td>378.0</td><td>398.2</td><td>399.2</td></tr></table>

<table><tr><td>Task Name</td><td>IQL</td><td>IQL+OTR</td><td>IQL+SEABO</td></tr><tr><td>pen-human</td><td>70.7±8.6</td><td>69.4±21.5</td><td>85.8±16.1</td></tr><tr><td>pen-cloned</td><td>37.2±7.3</td><td>42.7±25.0</td><td>49.2±12.2</td></tr><tr><td>door-human</td><td>3.3±1.3</td><td>4.2±2.1</td><td>6.8±5.6</td></tr><tr><td>door-cloned</td><td>1.6±0.5</td><td>0.0±0.0</td><td>0.1±0.1</td></tr><tr><td>relocate-human</td><td>0.1±0.0</td><td>0.1±0.1</td><td>0.1±0.1</td></tr><tr><td>relocate-cloned</td><td>-0.2±0.0</td><td>-0.2±0.0</td><td>-0.2±0.0</td></tr><tr><td>hammer-human</td><td>1.6±0.6</td><td>1.4±0.2</td><td>1.7±0.3</td></tr><tr><td>hammer-cloned</td><td>2.1±1.0</td><td>1.3±0.7</td><td>1.7±0.5</td></tr><tr><td>Total Score</td><td>116.4</td><td>118.9</td><td>145.2</td></tr></table>

Table 13: Comparison of SEABO against OTR using TD3\_BC as the base algorithm. We report the average normalized scores and their standard deviations. We bold and highlight the mean score cell except for TD3\_BC. We adopt one single expert demonstration for OTR and SEABO. 

<table><tr><td>Task Name</td><td>BC</td><td>10%BC</td><td>TD3_BC</td><td>TD3_BC+OTR</td><td>TD3_BC+SEABO</td></tr><tr><td>halfcheetah-medium</td><td>42.6</td><td>42.5</td><td> $48.0 \pm 0.7$ </td><td> $42.6 \pm 1.0$ </td><td> $45.9 \pm 0.3$ </td></tr><tr><td>hopper-medium</td><td>52.9</td><td>56.9</td><td> $60.7 \pm 12.5$ </td><td> $66.4 \pm 10.3$ </td><td> $76.1 \pm 4.2$ </td></tr><tr><td>walker2d-medium</td><td>75.3</td><td>75.0</td><td> $83.7 \pm 5.3$ </td><td> $76.9 \pm 5.4$ </td><td> $76.6 \pm 0.4$ </td></tr><tr><td>halfcheetah-medium-replay</td><td>36.6</td><td>40.6</td><td> $44.4 \pm 0.8$ </td><td> $39.4 \pm 1.3$ </td><td> $43.0 \pm 0.4$ </td></tr><tr><td>hopper-medium-replay</td><td>18.1</td><td>75.9</td><td> $64.8 \pm 25.5$ </td><td> $74.9 \pm 28.8$ </td><td> $96.3 \pm 3.0$ </td></tr><tr><td>walker2d-medium-replay</td><td>26.0</td><td>62.5</td><td> $87.4 \pm 8.4$ </td><td> $69.7 \pm 16.4$ </td><td> $73.1 \pm 2.2$ </td></tr><tr><td>halfcheetah-medium-expert</td><td>55.2</td><td>92.9</td><td> $93.5 \pm 2.0$ </td><td> $74.8 \pm 20.1$ </td><td> $95.7 \pm 0.4$ </td></tr><tr><td>hopper-medium-expert</td><td>52.5</td><td>110.9</td><td> $100.2 \pm 20.0$ </td><td> $103.2 \pm 13.9$ </td><td> $107.1 \pm 3.3$ </td></tr><tr><td>walker2d-medium-expert</td><td>107.5</td><td>109.0</td><td> $109.5 \pm 0.5$ </td><td> $109.0 \pm 0.6$ </td><td> $109.7 \pm 0.2$ </td></tr><tr><td>Total Score</td><td>466.7</td><td>666.2</td><td>692.3</td><td>656.9</td><td>723.5</td></tr></table>

rameter setup of OTR suggested by the authors. We summarize the comparison results in Table 13. It turns out that TD3\_BC+SEABO outperforms TD3\_BC+OTR on 8 out of 9 datasets, often by a large margin, surpassing it by $10.1\%$ in terms of the total score. TD3\_BC+SEABO is the only algorithm that even beats TD3\_BC learned with raw rewards in total score. We observe that the standard deviation of TD3\_BC+OTR is large on datasets like halfcheetah-medium-expert-v2, while the standard deviation of TD3\_BC+SEABO is much smaller. This evidence indicates that SEABO is superior to OTR when acting as the reward labeler, and can consistently aid different base offline RL algorithms recover its performance under ground-truth rewards or achieve better performance.

# B.3 HYPERPARAMETER SENSITIVITY

In Section 5.5, we are only able to attach the results on a small proportion of datasets from D4RL, e.g., halfcheetah-medium-replay-v2 due to the space limit. In this part, we include wider experimental results in terms of the reward scale $\alpha$ , weighting coefficient $\beta$ , and number of neighbors N. Again, we use IQL as the base offline RL algorithm for SEABO. The expert demonstrations utilized here contain actions. We follow the hyperparameter setup specified in Section A.

Reward scale $\alpha$ . The reward scale $\alpha$ controls the magnitude of the computed rewards. In Figure 3 of the main text, we find that a smaller $\alpha$ seems to be better (especially on hopper-medium-v2). We further conduct experiments on three additional tasks, halfcheetah-medium-expert-v2, hopper-medium-replay-v2, and walker2d-medium-v2 by varying $\alpha \in \{1, 5, 10\}$ . The results are shown in Figure 5, where we actually do not find much performance difference of $\alpha$ on these three tasks. That indicates that IQL+SEABO is robust to $\alpha$ on most of the datasets. In practice, one can simply set $\alpha = 1$ , which we find can already yield very good performance on MuJoCo tasks, AntMaze tasks, and Adroit tasks.

Weighting coefficient $\beta$ . As commented in the main text, the weighting coefficient $\beta$ is perhaps the most important hyperparameter in SEABO, since it controls the weights of the measured distance and this may have a significant influence on the final rewards. For a specific domain,

![](images/f6ea595d5d3fe3aeae06da39134995d43a5141d4fd316e11589f89716f7bcd53.jpg)

Figure 5: Additional experiments on the influence of $\alpha$ . The shaded region captures the standard deviation. All other hyperparameters are kept unchanged except $\alpha$ .   
![](images/4ca5d7348a2ab72084aca5b34007793cb6612a964c708bea20a9c594fd90b0f3.jpg)  
Figure 6: Additional experiments on the effect of $\beta$ . We choose three additional datasets from D4RL, and plot their mean normalized score curve. The shaded area denotes the standard deviation.

we mostly adopt a fixed $\beta$ as we do not want to bother tuning this hyperparameter. However, we believe it is vital to examine how $\beta$ influences the performance of SEABO in wider experiments. We additionally conduct several experiments on halfcheetah-medium-expert-v2, hopper-medium-replay-v2, walker2d-medium-replay-v2 from D4RL locomotion tasks. We sweep $\beta$ across $\{0.1, 0.5, 1, 5\}$ , and summarize the results in Figure 6. It can be clearly seen that a large $\beta$ results in poor performance on halfcheetah-medium-expert-v2 and walker2d-medium-replay-v2, while setting $\beta = 5$ results in the best performance on hopper-medium-replay-v2. In the hyperparameter setup part, we state that we set $\beta = 1$ on hopper-medium-replay-v2 due to the fact that SEABO is comparatively stable with $\beta\{0.5, 1\}$ . We do not doubt that the best $\beta$ is task-dependent, and one can get higher performance by carefully tuning this hyperparameter. However, we empirically show that using a fixed $\beta$ is also feasible, and we believe this is appealing since the users can get rid of the work of tedious hyperparameter search.

Number of neighbors N. The number of neighbors N is a hyperparameter introduced in the nearest neighbor algorithms. For all of our main experiments, we simply adopt N = 1, i.e., searching for the nearest neighbor. In Figure 4(b), we see that SEABO is robust to this hyperparameter. To examine whether this conclusion applies to a wider range of datasets, we conduct experiments on three additional datasets, halfcheetah-medium-v2, halfcheetah-medium-expert-v2, and walker2d-medium-replay-v2. The results are summarized in Figure 7, where we also observe that SEABO is robust to this hyperparameter, indicating the effectiveness and generality of SEABO.

# B.4 PERFORMANCE OF SEABO UNDER LONG-HORIZON MANIPULATION TASKS

In this part, we investigate how SEABO behaves under long-horizon manipulation tasks. To that end, we evaluate SEABO in Kitchen datasets (Fu et al., 2020). The kitchen environment (Gupta et al., 2019) consists of a 9 DoF Franka robot interacting with a kitchen scene that includes an openable microwave, four turnable oven burners, an oven light switch, a freely movable kettle, two hinged cabinets, and a sliding cabinet door. In kitchen, the robot may need to manipulate different components, e.g., it may need to open the microwave, move the kettle, turn on the light, and slide open the cabinet (precision is required). We run IQL+SEABO on three kitchen datasets using the author-recommended hyperparameters of IQL on the kitchen environment. We set reward

![](images/dd4ad68f189afcde883c409cf1520bb6136ccf990a3d75d80531f61a6f423277.jpg)

Figure 7: Additional experiments on examining the influence of the number of neighbors in KD-tree. The shaded region represents the standard deviation.   
Table 14: Comparison of SEABO against baselines in the Kitchen tasks. We report the average normalized scores and the corresponding standard deviations. We bold and highlight the best mean score cell. 

<table><tr><td>Task Name</td><td>BC</td><td>CQL</td><td>IQL</td><td>IQL+SEABO</td></tr><tr><td>kitchen-complete-v0</td><td>65.0</td><td>43.8</td><td>62.5</td><td> $67.5 \pm 4.2$ </td></tr><tr><td>kitchen-partial-v0</td><td>38.0</td><td>49.8</td><td>46.3</td><td> $71.0 \pm 4.1$ </td></tr><tr><td>kitchen-mixed-v0</td><td>51.5</td><td>51.0</td><td>51.0</td><td> $55.0 \pm 3.5$ </td></tr><tr><td>Average Score</td><td>51.5</td><td>48.2</td><td>53.3</td><td>64.5</td></tr></table>

scale $\alpha = 1$ , coefficient $\beta = 0.5$ for SEABO. We compare IQL+SEABO against some baselines taken from the IQL paper and summarize the results in Table 14. We find that SEABO exhibits superior performance, surpassing IQL with raw rewards by 21.0%. We believe these results show that SEABO can aid some long-horizon manipulation tasks.

However, we experimentally find that SEABO does not exhibit strong performance for some tasks that require high precision, e.g., the IKEA Furniture assembly benchmark (Lee et al., 2019; 2021; Heo et al., 2023). We leave the open problem of how to enable SEABO to successfully address such benchmarks a future work.

# B.5 LEARNING CURVES

In this section, we provide the detailed training curves of IQL+SEABO on the locomotion tasks, AntMaze tasks, and Adroit tasks. We also provide learning curves of TD3\_BC+SEABO on locomotion tasks. We summarize the results of IQL+SEABO on D4RL MuJoCo locomotion tasks in Figure 8, the performance of IQL+SEABO on AntMaze tasks in Figure 9, and the curves of IQL+SEABO on Adroit tasks in Figure 10. The results of TD3\_BC+SEABO are depicted in Figure 11.

From all these results, we find that both IQL+SEABO and TD3\_BC+SEABO have stable and strong performance on the evaluated tasks, indicating the advantages of our method.

# C DISCUSSIONS ON DIFFERENT SEARCH ALGORITHMS

The success of SEABO can be largely attributed to the adopted search algorithm (i.e., KD-tree). In Section 5.4 of the main text, we compare different design choices for the underlying search algorithm. It is not surprising to find that Ball-tree results in a similar performance as KD-tree, as Ball-tree shares many similarities with KD-tree. However, we find that HNSW incurs quite poor performance on many datasets using its default hyperparameter setup (see Appendix A). HNSW builds a multi-layer structure made up of a hierarchical set of proximity graphs for nested subsets of the stored elements while employing a heuristic for selecting proximity graph neighbors. HNSW is a graph-based search algorithm. Based on the empirical results in Table 6 in the main text, we find that HNSW leads to quite poor performance for the base offline RL algorithm, only achieving competitive performance against KD-tree on halfcheetah-medium-v2 and walker2d-medium-expert-v2.

![](images/4c21bafcb75fd15bcb7799acda7731cd45e217e56a93f51a424bf2ece0d50ee3.jpg)

Figure 8: Full learning curves of IQL+SEABO on D4RL MuJoCo datasets. We plot the average performance and the shaded region captures the standard deviation.   
![](images/cf99ba49ca6413fbb4ea4f2e6b51c337dd0292c47695c3e5bfaa5b21033897a4.jpg)  
Figure 9: Full learning curves of IQL+SEABO on AntMaze tasks. The mean performance in conjunction with the standard deviations are plotted.

![](images/c7017372c47c2654f19e6a5298dae2748e1b0ce5e39511af4a9289be11249c9b.jpg)

<details>
<summary>line</summary>

| Gradient Steps | Normalized score |
| -------------- | ---------------- |
| 0.0            | 90               |
| 0.2            | 100              |
| 0.4            | 95               |
| 0.6            | 100              |
| 0.8            | 95               |
| 1.0            | 100              |
</details>

![](images/c3c3176777a38504344de8c9195511897b8f267b9fb809bbe793c8212930cdc9.jpg)

<details>
<summary>line</summary>

| Gradient Steps | IQL+SEABO |
| -------------- | --------- |
| 0.0            | 50        |
| 0.2            | 55        |
| 0.4            | 50        |
| 0.6            | 55        |
| 0.8            | 50        |
| 1.0            | 55        |
</details>

![](images/3baeb858900429b762efaf6b1af4c172240b101623b3529c404166bac35d44a3.jpg)

<details>
<summary>line</summary>

| Gradient Steps | IQL+SEABO |
| -------------- | --------- |
| 0.0            | 7.0       |
| 0.1            | 8.0       |
| 0.2            | 6.0       |
| 0.3            | 5.0       |
| 0.4            | 6.0       |
| 0.5            | 5.0       |
| 0.6            | 7.0       |
| 0.7            | 6.0       |
| 0.8            | 5.0       |
| 0.9            | 6.0       |
| 1.0            | 5.0       |
</details>

![](images/61b5c2c4f6619a484dc2cf146e1283764de91bd0d7e40fe0fff8562d017420ba.jpg)

<details>
<summary>line</summary>

| Gradient Steps | IQL+SEABO |
| -------------- | --------- |
| 0.0            | 0.0       |
| 0.1            | 1.5       |
| 0.2            | 0.8       |
| 0.3            | 0.5       |
| 0.4            | 0.3       |
| 0.5            | 0.2       |
| 0.6            | 0.1       |
| 0.7            | 0.05      |
| 0.8            | 0.02      |
| 0.9            | 0.01      |
| 1.0            | 0.0       |
</details>

![](images/cafa263aa5f5e5234be0c4cab00cb084a2d666c4cdd980fabab12a72561bf385.jpg)

<details>
<summary>line</summary>

| Gradient Steps | Normalized score |
| -------------- | ---------------- |
| 0.0            | 0.25             |
| 0.2            | 0.20             |
| 0.4            | 0.15             |
| 0.6            | 0.10             |
| 0.8            | 0.05             |
| 1.0            | 0.00             |
</details>

![](images/932b4d364fd52c88caa41f902a266b8b21c03749f39d34375fb8176ea2954b70.jpg)

<details>
<summary>line</summary>

| Gradient Steps | IQL+SEABO |
| -------------- | --------- |
| 0.0            | -0.24     |
| 0.2            | -0.23     |
| 0.4            | -0.24     |
| 0.6            | -0.25     |
| 0.8            | -0.24     |
| 1.0            | -0.25     |
</details>

![](images/d037a9e845229ccd8db39965a39cfc435b894bf9f1792e91667ff6884b37f822.jpg)

<details>
<summary>line</summary>

| Gradient Steps | IQL+SEABO |
| -------------- | --------- |
| 0.0            | 1.5       |
| 0.2            | 2.0       |
| 0.4            | 2.5       |
| 0.6            | 2.0       |
| 0.8            | 1.8       |
| 1.0            | 1.7       |
</details>

![](images/73392031b4d2360643b71d800736595878123a102c329dd73dca4e7c1d6d46c7.jpg)

<details>
<summary>line</summary>

| Gradient Steps | IQL+SEABO |
| -------------- | --------- |
| 0.0            | 2.0       |
| 0.2            | 3.0       |
| 0.4            | 5.0       |
| 0.6            | 4.0       |
| 0.8            | 7.0       |
| 1.0            | 3.0       |
</details>

Figure 10: Full learning curves of IQL+SEABO on Adroit datasets. We report the mean performance along with its standard deviation.

![](images/ed6c18910f4e2a0cad93866e7f17192d37a4b98e145f3ccc87d2dbe94a3ba6d3.jpg)

<details>
<summary>line</summary>

| Gradient Steps | Normalized score |
| -------------- | ---------------- |
| 0.0            | 35.0             |
| 0.2            | 45.0             |
| 0.4            | 45.0             |
| 0.6            | 45.0             |
| 0.8            | 45.0             |
| 1.0            | 45.0             |
</details>

![](images/c69cd480a848af038b601f7dffb4bca5e49de23dc585357f17557cefb078c46a.jpg)

<details>
<summary>line</summary>

| Gradient Steps | TD3_BC+SEABO |
| -------------- | ------------- |
| 0.0            | 10            |
| 0.2            | 40            |
| 0.4            | 42            |
| 0.6            | 42            |
| 0.8            | 42            |
| 1.0            | 42            |
</details>

![](images/813730d4f274811ee57c83b1fa2f8dde209a0c477a96f2e311933cf544517d06.jpg)

<details>
<summary>line</summary>

| Gradient Steps | TD3_BC+SEABO |
| -------------- | ------------ |
| 0.0            | 20           |
| 0.2            | 85           |
| 0.4            | 90           |
| 0.6            | 95           |
| 0.8            | 97           |
| 1.0            | 98           |
</details>

![](images/29823f445a44e6244cc2e1e6ae8bdd6118a88cdddfc4c73c3a6685324e826890.jpg)

<details>
<summary>line</summary>

| Gradient Steps | Normalized score |
| -------------- | ---------------- |
| 0.0            | 50               |
| 0.2            | 65               |
| 0.4            | 70               |
| 0.6            | 75               |
| 0.8            | 72               |
| 1.0            | 75               |
</details>

![](images/543e93552f8e1e834ce124637524631e6ea510fac3576bf878ac635ebf8c46db.jpg)

<details>
<summary>line</summary>

| Gradient Steps | TD3_BC+SEABO |
| -------------- | ------------- |
| 0.0            | 20            |
| 0.2            | 80            |
| 0.4            | 90            |
| 0.6            | 95            |
| 0.8            | 98            |
| 1.0            | 97            |
</details>

![](images/0a9746be0371f8498a8ef9c9cae0ffa953f93b05a99ba9e036a3f39ad94652b3.jpg)

<details>
<summary>line</summary>

| Gradient Steps | TD3_BC+SEABO |
| -------------- | ------------ |
| 0.0            | 60           |
| 0.2            | 90           |
| 0.4            | 100          |
| 0.6            | 105          |
| 0.8            | 108          |
| 1.0            | 110          |
</details>

![](images/84641963e7c4d1d48b7876a024cbeb6a868c9a928a601a3c95eff684227a7af9.jpg)

<details>
<summary>line</summary>

| Gradient Steps | Normalized score |
| -------------- | ---------------- |
| 0.0            | 68.0             |
| 0.1            | 75.0             |
| 0.2            | 76.0             |
| 0.3            | 77.0             |
| 0.4            | 76.5             |
| 0.5            | 77.0             |
| 0.6            | 76.5             |
| 0.7            | 77.0             |
| 0.8            | 76.5             |
| 0.9            | 77.0             |
| 1.0            | 76.5             |
</details>

![](images/04faabca93baad64f631aed987749c76389e7ee5f32def551ca0ed3809cac50c.jpg)

<details>
<summary>line</summary>

| Gradient Steps | TD3_BC+SEABO |
| -------------- | ------------- |
| 0.0            | 20            |
| 0.2            | 60            |
| 0.4            | 70            |
| 0.6            | 75            |
| 0.8            | 78            |
| 1.0            | 75            |
</details>

![](images/8a1f80b78f26f42e2bfe915a454ec55efac19397285bb46397aadb0af9a22f5a.jpg)

<details>
<summary>line</summary>

| Gradient Steps | TD3_BC+SEABO |
| -------------- | ------------ |
| 0.0            | 80           |
| 0.2            | 110          |
| 0.4            | 110          |
| 0.6            | 110          |
| 0.8            | 110          |
| 1.0            | 110          |
</details>

Figure 11: Full learning curves of TD3\_BC+SEABO on D4RL MuJoCo datasets. The average performance as well as its statistical significance is depicted.

In this subsection, we try to understand why HNSW fails through some empirical evidence. We choose some subsets, halfcheetah-medium-v2, halfcheetah-medium-expert-v2, hopper-medium-replay-v2, hopper-medium-expert-v2, walker2d-medium-v2, and walker2d-medium-replay-v2, from D4RL MuJoCo datasets and plot the reward density of ground-truth rewards, rewards computed using KD-tree, and rewards acquired via HNSW. We summarize the results in Figure 12. It is clear that SEABO with KD-tree can produce a similar reward structure as the ground-truth reward distribution, while SEABO with HNSW tends to assign large rewards to only a small proportion of samples and small rewards to the majority of transitions. We believe this explains the unsatisfying performance of IQL+SEABO with HNSW as the base search algorithm, indicating that a graph-based search mechanism may not be suitable for D4RL datasets. Another possible explanation is that the hyperparameters of HNSW need to be tuned to adapt to different tasks. We do not doubt that a careful tuning of hyperparameters (e.g., the maximum number of outgoing connections in the graph, the number of neighbors, etc.) has the potential of making SEABO with HNSW work in D4RL datasets. However, we do not think it is necessary to do that considering the fact that adopting KD-tree with its default hyperparameters can already result in quite good performance across different datasets. Hence, it is recommended that one uses KD-tree (or Ball-tree) as the base search algorithm.

# D COMPUTE INFRASTRUCTURE

In Table 15, we list the compute infrastructure that we use to run all of the algorithms.

Table 15: Compute infrastructure. 

<table><tr><td>CPU</td><td>GPU</td><td>Memory</td></tr><tr><td>AMD EPYC 7452</td><td>RTX3090×8</td><td>288GB</td></tr></table>

# E LIMITATIONS

Despite the simplicity and effectiveness of our proposed algorithm, SEABO, we have to admit honestly that there may exist some potential limitations. First, SEABO is slightly sensitive to the weighting coefficient $\beta$ on some datasets (not all datasets), and one may need to manually tune it so as to find the best-suited hyperparameter setup for a specific task. While based on our empirical results, one can find the best $\beta \in \{0.5, 1, 5\}$ using grid search, It is not difficult to conduct experiments since SEABO is computationally efficient (and can be applied with only CPUs). Second, it may take more time for SEABO to annotate the unlabeled trajectories with visual input, as images are hard to process. Whereas, we can preprocess the visual images using some pre-trained image encoder (e.g., ImageNet pretrain models) to obtain low-dimensional representations of the high-dimensional image. Note that we build KD-tree upon expert demonstrations which usually contain a small amount of expert transitions. Thus, it should not be time-consuming to annotate the visual trajectories.

We hope this work can provide some new insights to the community and inspire future work on offline imitation learning.

# F ADDITIONAL REWARD PLOTS ON ADROIT AND ANTMAZE TASKS

In this section, we provide reward distribution plots of the ground truth rewards, rewards obtained by SEABO, and rewards output from HNSW on some Adroit-v0 and AntMaze-v0 tasks, hammer-human-v0, hammer-cloned-v0, door-human-v0, door-cloned-v0, antmaze-uamze-v0, and antmaze-medium-diverse-v0. Note that we provide the histogram plot of rewards in antmaze-medium-diverse-v0 as most of the samples in this datasets have quite similar reward signals, making it difficult to draw the density plot. We summarize the results in Figure 13. It can be seen that with KD-tree, SEABO outputs similar reward density as vanilla rewards (e.g., SEABO successfully gives three peaks in hammer-human-v0 and door-human-v0).

![](images/413963f9c3ffeaa3a42df4eae2af3f5e3111eb5c5592ba918f71e8866d633efb.jpg)  
Figure 12: Density plot comparison of ground-truth rewards and rewards acquired by different search algorithms. The right two columns show reward distributions of two SEABO variants.

![](images/e3163db6162bfa5a933e75a3b2263f19b701bd4ce24a90f0c931e1d7d9bc5310.jpg)  
Figure 13: Density plot comparison of ground-truth rewards and rewards acquired by different search algorithms. The results are on selected datasets from Adroit and AntMaze tasks.

# G DISCUSSIONS ON SEABO AND ILR

There are some previous studies that use nearest neighbor-based methods for imitation learning, e.g., Pari et al. (2021). Among them, the most relevant to our work is Ciosek (2022). In this section, we discuss the connections and differences between our method and prior work, ILR (Ciosek, 2022), which can be summarized below:

\- The motivations are varied. The practical reward formula in ILR is given by $r = 1 - \min_{(s', a') \in D} d_{l_2}((s, a), (s', a'))^2$ , which is a relaxation of its theoretical version. There exists a gap between the theory and the resulting reward formula. The authors claim that the relaxation is an upper bound on the scaled theoretical reward and interpret $L = \min_{(s', a') \in D} d_{l_2}((s, a), (s', a'))^2$ as the $l_2$ -diameter of the state-action space. The primary goal of doing so is to reduce imitation learning to RL with a stationary reward for deterministic experts. However, the motivation of SEABO is that we would like to determine the optimality of the single transition (instead of examining whether the transition comes from the expert trajectory or performing relaxation to the rewards). We assume that the transition is near-optimal if it lies close to the expert trajectory. Hence, we assign a larger reward to the transition if it is close to the expert trajectory and a smaller reward otherwise. Meanwhile, SEABO does not require that the expert is deterministic (and also does not require that the environment is deterministic). We aim to adopt SEABO to annotate unlabeled samples in the dataset and train off-the-shelf offline RL algorithms.

\- The methods are different but connected. The reward formula adopted in ILR is a special case of SEABO with Euclidean distance. SEABO does not interpret $L$ as the diameter of the state-action space. SEABO can adopt $N$ nearest neighbors and use their average distance to compute the reward (ILR simply finds the smallest Euclidean distance between sample $(s, a)$ and the expert trajectory). Meanwhile, SEABO is not restricted to Euclidean distance. Our procedure is, that we first find the nearest neighbor of the query sample, and then utilize some distance measurements (different distance measurements can be used here) to decide the distance between the query sample and its nearest neighbor, and finally get the reward by adopting a squashing function. Furthermore, SEABO strongly relies on the nearest neighbor methods (e.g., KD-Tree), and one can use different types of nearest neighbor algorithms in SEABO, while ILR does not emphasize search algorithms. Note that different search algorithms with different hyperparameter setups can result in different final rewards. For example, in scipy.spatial.KDTree.query, setting eps larger than 0 enables approximate nearest neighbors search and ensures that the $k$ -th returned value is no further than $(1 + \text{eps})$ times the distance to the real $k$ -th nearest neighbor. This may incur different results from ILR even under Euclidean distance. Moreover, SEABO can also work in state-only regimes, which is both a more general and challenging setting, while ILR strongly relies on the assumption that state-action pairs are present in the expert trajectory in its theory and practical implementation. Finally, one can query with $(s, a, s')$ , $(s, a)$ or $(s, s')$ in SEABO (ILR is limited to $(s, a)$ ), and SEABO adopts a different choice of squashing function.

\- The settings are varied. SEABO is targeted at the offline imitation learning setting while ILR addresses the online setting. It also turns out that the experiment setup (e.g., number of expert trajectories) is different between SEABO and ILR.