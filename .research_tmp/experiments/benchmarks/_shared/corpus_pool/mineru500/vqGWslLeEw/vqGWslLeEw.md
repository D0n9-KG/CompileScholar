# Revisiting the Minimalist Approach to Offline Reinforcement Learning

Denis Tarasov Vladislav Kurenkov Alexander Nikulin Sergey Kolesnikov Tinkoff

{den.tarasov, v.kurenkov, a.p.nikulin, s.s.kolesnikov}@tinkoff.ai

# Abstract

Recent years have witnessed significant advancements in offline reinforcement learning (RL), resulting in the development of numerous algorithms with varying degrees of complexity. While these algorithms have led to noteworthy improvements, many incorporate seemingly minor design choices that impact their effectiveness beyond core algorithmic advances. However, the effect of these design choices on established baselines remains understudied. In this work, we aim to bridge this gap by conducting a retrospective analysis of recent works in offline RL and propose ReBRAC, a minimalistic algorithm that integrates such design elements built on top of the TD3+BC method. We evaluate ReBRAC on 51 datasets with both proprioceptive and visual state spaces using D4RL and V-D4RL benchmarks, demonstrating its state-of-the-art performance among ensemble-free methods in both offline and offline-to-online settings. To further illustrate the efficacy of these design choices, we perform a large-scale ablation study and hyperparameter sensitivity analysis on the scale of thousands of experiments. $^{1}$

# 1 Introduction

Interest of the reinforcement learning (RL) community in the offline setting has led to a myriad of new algorithms specifically tailored to learning highly performant policies without the ability to interact with an environment (Levine et al., 2020; Prudencio et al., 2022). Yet, similar to the advances in online RL (Engstrom et al., 2020; Henderson et al., 2018), many of those algorithms come with an added complexity – design and implementation choices beyond core algorithmic innovations, requiring a delicate effort in reproduction, hyperparameter tuning, and causal attribution of performance gains.

Indeed, the issue of complexity was already raised in the offline RL community by Fujimoto & Gu (2021); the authors highlighted veiled design and implementation-level adjustments (e.g., different architectures or actor pre-training) and then demonstrated how a simple behavioral cloning regularization added to the TD3 (Fujimoto et al., 2018) constitutes a strong baseline in the offline setting. This minimalistic and uncluttered algorithm, TD3+BC, has become a de-facto standard baseline to be compared against. Indeed, most new algorithms juxtapose against it and claim significant gains over (Akimov et al., 2022; An et al., 2021; Nikulin et al., 2023; Wu et al., 2022; Chen et al., 2022b; Ghasemipour et al., 2022). However, application of newly emerged design and implementation choices to this baseline is still missing.

In this work, we build upon Fujimoto & Gu (2021) line of research and ask: what is the extent to which newly emerged minor design choices can advance the minimalistic offline RL algorithm? The answer is illustrated in Figure 1: we propose an extension to TD3+BC, ReBRAC (Section 3), that simply adds on recently appeared design decisions upon it. We test our algorithm on both proprioceptive and

![](images/0d90f1cd69ff148ab7df4f558c07f18bd976a0022d5d2e5f4800f9f4a43e0b67.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["MSE and Q normalization (TD3+BC)"] --> B["ReBRAC"]
    C["Deeper Networks (CQL)"] --> B
    D["Actor and critic penalty (TD3-CVAE)"] --> B
    E["Larger batches and LayerNorm (LB-SAC)"] --> B
    F["Adjusted discount factor (SPOT)"] --> B
```
</details>

(a)

![](images/2f87affd90ebdd87e886016d8c8712d9831bec78b0c3142a7a49d1fc5df31751.jpg)

<details>
<summary>line</summary>

| D4RL Normalized Score (τ) | ReBRAC | TD3 + BC | IQL | SAC-RND |
| ------------------------- | ------ | -------- | --- | ------- |
| 0                         | 1.00   | 1.00     | 1.00 | 1.00    |
| 25                        | 0.95   | 0.85     | 0.90 | 0.80    |
| 50                        | 0.85   | 0.75     | 0.80 | 0.70    |
| 75                        | 0.75   | 0.65     | 0.70 | 0.60    |
| 100                       | 0.60   | 0.50     | 0.55 | 0.45    |
| 125                       | 0.30   | 0.25     | 0.30 | 0.20    |
| 150                       | 0.10   | 0.10     | 0.10 | 0.10    |
</details>

(b)

![](images/42747fb28facc5ba8cc92d6ee6f42b1f474af62f7c852ef843ebb77d49b64873.jpg)

<details>
<summary>bar</summary>

| Algorithm X | Algorithm Y |
| :--- | :--- |
| ReBRAC | TD3+BC |
| ReBRAC | SAC-RND |
| ReBRAC | IQL |
</details>

(c)   
Figure 1: (a) The schema of our approach ReBRAC (b) Performance profiles (c) Probability of improvement. The curves (Agarwal et al., 2021) are for D4RL benchmark spanning all Gym-MuJoCo, AntMaze, and Adroit datasets (Fu et al., 2020).

visual state space problems using D4RL (Fu et al., 2020) and V-D4RL (Lu et al., 2022) benchmarks (Section 4) demonstrating its state-of-the-art performance across ensemble-free methods. Moreover, our approach demonstrates state-of-the-art performance in offline-to-offline setup on D4RL datasets (Section 4.3) while not being specifically designed for this setup. To further highlight the efficacy of the proposed modifications, we then conduct a large-scale ablation study (Section 4.4). We hope the described approach can serve as a strong baseline under different hyperparameter search budgets (Section 4.6), further accentuating the importance of seemingly minor design choices introduced along with core algorithmic innovations.

# 2 Preliminaries

# 2.1 Offline Reinforcement Learning

A standard Reinforcement Learning problem is defined as a Markov Decision Process (MDP) with the tuple $\{S, A, P, R, \gamma\}$ , where $S \subset \mathbb{R}^n$ is the state space, $A \subset \mathbb{R}^m$ is the action space, $P: S \times A \to S$ is the transition function, $R: S \times A \to \mathbb{R}$ is the reward function, and $\gamma \in (0,1)$ is the discount factor. The ultimate objective is to find a policy $\pi(a|s)$ that maximizes the cumulative discounted return $\mathbb{E}_{\pi} \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t)$ . This policy improves by interacting with the environment, observing states, and taking actions that provide rewards.

In offline RL, policies cannot interact with the environment and can only access a static transaction dataset D collected by one or more other policies. This setting presents new challenges, such as estimating values for state-action pairs not included in the dataset while exploration is unavailable (Levine et al., 2020).

# 2.2 Behavior Regularized Actor-Critic

Behavior Regularized Actor-Critic (BRAC) is an offline RL framework introduced in Wu et al. (2019). The core idea behind BRAC is that actor-critic algorithms can be penalized in two ways to solve offline RL tasks: actor penalization and critic penalization. In this framework, the actor objective is represented as in Equation (1), and the critic objective as in Equation (2), where F is a divergence function between dataset actions and policy actions distributions. The differences from a vanilla actor-critic are highlighted in blue.

$$
\pi = \underset {\pi} {\operatorname{argmax}} \mathbb {E} _ {(s, a) \sim D} \left[ Q _ {\theta} (s, \pi (s)) - \alpha \cdot F (\pi (s), a) \right] \tag {1}
$$

$$
\theta = \underset {\theta} {\operatorname{argmin}} \mathbb {E} _ {\substack {(s, a, r, s ^ {\prime}, \hat {a} ^ {\prime}) \sim D \\ a ^ {\prime} \sim \pi (s ^ {\prime})}} \left[ \left(Q _ {\theta} (s, a) - \left(r + \gamma \left(Q _ {\overline {{\theta}}} \left(s ^ {\prime}, a ^ {\prime}\right) - \alpha \cdot F \left(a ^ {\prime}, \hat {a} ^ {\prime}\right)\right)\right)\right) ^ {2} \right] \tag{2}
$$

In the original work, various choices of $F$ were evaluated when used as the regularization term for the actor or critic. The authors tested KL divergence, Kernel MMD, and Wasserstein distance but did not observe any consistent advantage. Finally, it is essential to note that, originally, both regularizations coefficients had the same weight.

Subsequently, TD3+BC (Fujimoto & Gu, 2021) was introduced, utilizing Mean Squared Error (MSE) as the regularization term $F$ for the actor. TD3+BC is considered to be the minimalist approach to offline RL as it modifies existing RL algorithms by simply adding behavior cloning term into actor loss which is easy to implement and does not bring any significant computational overhead.

# 3 ReBRAC: Distilling Key Design Choices

In this section, we describe the proposed method along with the discussion of the new design choices met in the offline RL literature (Table 1). Our approach is a more general version of BRAC built on top of the TD3+BC (Fujimoto & Gu, 2021) algorithm with different modifications in design while keeping it simple (Figure 1). We refer to our method as Revisited BRAC (ReBRAC).

Table 1: Adoption of implementation and design choices beyond core algorithmic advancements in some recently introduced algorithms. 

<table><tr><td>Modification</td><td>TD3+BC</td><td>CQL</td><td>EDAC</td><td>MSG</td><td>CNF</td><td>LB-SAC</td><td>SAC-RND</td></tr><tr><td>Deeper networks</td><td>✕</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Larger batches</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Layer Normalization</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td><td>√</td><td>√</td></tr><tr><td>Decoupled penalization</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td><td>√</td></tr><tr><td>Adjusted discount factor</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td><td>√</td></tr></table>

Deeper Networks The use of deeper neural networks has been a critical factor in the success of many Deep Learning models, with model quality generally increasing as depth increases, provided there is enough data to support this scaling (Kaplan et al., 2020). Similarly, recent studies in RL (Neumann & Gros, 2022; Sinha et al., 2020) and offline RL (Lee et al., 2022; Kumar et al., 2022) have demonstrated the importance of depth in achieving high performance. Although most offline RL algorithms are based on SAC (Haarnoja et al., 2018) or TD3 (Fujimoto et al., 2018), which by default employ two hidden layers, recent work (Kumar et al., 2020) uses three hidden layers for SAC instead, which appears to be an important change (Fujimoto & Gu, 2021). The change in network size has been adopted by later works (An et al., 2021; Yang et al., 2022; Zhuang et al., 2023; Nikulin et al., 2023) and may be one of the critical modifications that improve final performance.

The original BRAC and TD3+BC algorithms used only two hidden layers for their actor and critic networks, while most state-of-the-art solutions use deeper networks. Specifically, three hidden layers have become a common choice for recent offline RL algorithms. In ReBRAC, we follow this trend and use three hidden layers for the actor and critic networks. Additionally, we provide an ablation study in Section 4.5 to investigate the effect of the number of layers on ReBRAC's performance.

LayerNorm LayerNorm (Ba et al., 2016) is a widely used technique in deep learning that helps improve network convergence. In Hiraoka et al. (2021), authors add dropout and LayerNorm to different RL algorithms, notably boosting their performance. This technique is also applied in Smith et al. (2022), and it appears that boost is achieved primarily because of LayerNorm. Specifically, in offline RL, various studies have tested the effect of normalizations between layers (Bhatt et al., 2019; Kumar et al., 2022; Nikulin et al., 2022, 2023). A parallel study by Ball et al. (2023) empirically shows that LayerNorm helps to prevent catastrophic value extrapolation for the Q function when using offline datasets in online RL. Following these works, in our approach, we also apply LayerNorm between each layer of the critic networks.

Larger Batches Another technique to accelerate neural network convergence is large batch optimization (You et al., 2017, 2019). While studying batch sizes larger than 256 is limited, some prior works have used them. For instance, the convergence of SAC-N was accelerated in Nikulin et al. (2022). More recently proposed algorithms also use larger batches for training, although without providing detailed analyses (Akimov et al., 2022; Nikulin et al., 2023).

The usage of large batches in offline RL is still understudied, and its benefits and limitations are not fully understood. Our experiments show that in some domains, using large batches can lead to significant performance improvements, while in others, it might not have a notable impact or

even drop the performance (see Table 8). We increased the batch size to 1024 samples and scaled the learning rate for D4RL Gym-MuJoCo tasks, following the approach proposed by Nikulin et al. (2022).

Actor and critic penalty decoupling The original BRAC framework proposed penalizing the actor and the critic with the same magnitude. Most of the previous algorithms restrict only actor (Fujimoto & Gu, 2021; Wu et al., 2022) or only critic (Kumar et al., 2020; An et al., 2021; Ghasemipour et al., 2022). TD3-CVAE (Rezaeifar et al., 2022) penalizes both using the same coefficients, while another study, SAC-RND (Nikulin et al., 2023), shows that decoupling the penalization in offline RL is beneficial for algorithm performance, although ablations on using only one of the penalties are missing.

Our method allows simultaneous penalization of actor and critic with decoupled parameters. Inspired by TD3+BC (Fujimoto & Gu, 2021), Mean Squared Error (MSE) is used as a divergence function F, which we found simple and effective. The actor objective is shown in Equation (3), and the critic objective is shown in Equation (4). Differences from the original actor-critic are highlighted in red. Following TD3+BC (Fujimoto & Gu, 2021), the Q function is normalized to make the algorithm less sensitive to regularization parameters. Nonetheless, we forego the utilization of state normalization, as initially suggested in TD3 + BC, driven by our intention to execute the algorithm online and the observation that this adjustment typically results in negligible impact. Since our approach principally builds upon TD3+BC, the differences in their performances should be considered the most important ones and do not appear only because of the additional hyperparameters search.

$$
\pi = \underset {\pi} {\operatorname{argmax}} \mathbb {E} _ {(s, a) \sim D} \left[ Q _ {\theta} (s, \pi (s)) - \beta_ {1} \cdot (\pi (s) - a) ^ {2} \right] \tag {3}
$$

$$
\theta = \underset {\theta} {\operatorname{argmin}} \mathbb {E} _ {\substack {(s, a, r, s ^ {\prime}, \hat {a} ^ {\prime}) \sim D \\ a ^ {\prime} \sim \pi (s ^ {\prime})}} \left[ \left(Q _ {\theta} (s, a) - \left(r + \gamma \left(Q _ {\overline {{\theta}}} \left(s ^ {\prime}, a ^ {\prime}\right) - \beta_ {2} \cdot \left(a ^ {\prime} - \hat {a} ^ {\prime}\right) ^ {2}\right)\right)\right) ^ {2} \right] \tag{4}
$$

Discount factor $\gamma$ value change The choice of discount factor is an important aspect in solving RL problems (Jiang et al., 2015). A recent study (Hu et al., 2022) suggests that decreasing the default value of $\gamma$ from 0.99 may lead to better results in offline RL settings. In contrast, in SPOT (Wu et al., 2022), the authors increased the value of $\gamma$ up to 0.995 when fine-tuning AntMaze tasks with sparse rewards, which resulted in state-of-the-art solutions. Similarly, in the offline setting SAC-RND (Nikulin et al., 2023), increasing $\gamma$ also achieved high performance on the same domain. The choice of increased $\gamma$ for AntMaze tasks was motivated by the sparse reward, i.e., a low $\gamma$ value may not propagate the training signal well. However, further ablations are needed to understand if the change in the parameter is directly responsible for the improved performance. In our experiments, we also find that increasing $\gamma$ from the default value of 0.99 to 0.999 is vital for improved performance on this set of tasks (see Table 8).

# 4 Experiments

# 4.1 Evaluation on offline D4RL

We evaluate the proposed approach on three sets of D4RL tasks: Gym-MuJoCo, AntMaze, and Adroit. For each domain, we consider all of the available datasets. We compare our results to several ensemble-free baselines, including TD3+BC (Fujimoto & Gu, 2021), IQL (Kostrikov et al., 2021), CQL (Kumar et al., 2020) and SAC-RND (Nikulin et al., 2023).

The majority of the hyperparameters are adopted from TD3+BC, while $\beta_{1}$ and $\beta_{2}$ parameters from Equation (3) and Equation (4) are tuned. We examine the sensitivity to these parameters in Section 4.6. For a complete overview of the experimental setup and details, see Appendix A.

Following Wu et al. (2022), we tune hyperparameters over four seeds (referred to as training seeds) and evaluate the best parameters over ten new seeds (referred to as unseen training seeds), reporting the average performance of the last checkpoints for D4RL tasks. On V-D4RL, we use two and five seeds, respectively. This helps to avoid overfitting during hyperparameters search and outputs more just and reproducible results. For a fair comparison, we tune TD3+BC and IQL following the same protocol. We also rerun SAC-RND on Gym-MuJoCo and AntMaze tasks and tune it for the Adroit

Table 2: Average normalized score over the final evaluation and ten unseen training seeds on Gym-MuJoCo tasks. CQL scores were taken from An et al. (2021). The symbol ± represents the standard deviation across the seeds. To make a fair comparison against TD3+BC and IQL, we extensively tuned their hyperparameters. 

<table><tr><td>Task Name</td><td>TD3+BC</td><td>IQL</td><td>CQL</td><td>SAC-RND</td><td>ReBRAC, our</td></tr><tr><td>halfcheetah-random</td><td> $\underline{30.9} \pm 0.4$ </td><td> $19.5 \pm 0.8$ </td><td> $\underline{31.1} \pm 3.5$ </td><td> $27.6 \pm 2.1$ </td><td> $29.5 \pm 1.5$ </td></tr><tr><td>halfcheetah-medium</td><td> $\underline{54.7} \pm 0.9$ </td><td> $50.0 \pm 0.2$ </td><td> $46.9 \pm 0.4$ </td><td> $\underline{66.4} \pm 1.4$ </td><td> $\underline{65.6} \pm 1.0$ </td></tr><tr><td>halfcheetah-expert</td><td> $93.4 \pm 0.4$ </td><td> $95.5 \pm 2.1$ </td><td> $97.3 \pm 1.1$ </td><td> $\underline{102.6} \pm 4.2$ </td><td> $\underline{105.9} \pm 1.7$ </td></tr><tr><td>halfcheetah-medium-expert</td><td> $89.1 \pm 5.6$ </td><td> $92.7 \pm 2.8$ </td><td> $95.0 \pm 1.4$ </td><td> $\underline{108.1} \pm 1.5$ </td><td> $\underline{101.1} \pm 5.2$ </td></tr><tr><td>halfcheetah-medium-replay</td><td> $45.0 \pm 1.1$ </td><td> $42.1 \pm 3.6$ </td><td> $45.3 \pm 0.3$ </td><td> $\underline{51.2} \pm 3.2$ </td><td> $\underline{51.0} \pm 0.8$ </td></tr><tr><td>halfcheetah-full-replay</td><td> $75.0 \pm 2.5$ </td><td> $75.0 \pm 0.7$ </td><td> $76.9 \pm 0.9$ </td><td> $\underline{81.2} \pm 1.3$ </td><td> $\underline{82.1} \pm 1.1$ </td></tr><tr><td>hopper-random</td><td> $8.5 \pm 0.7$ </td><td> $\underline{10.1} \pm 5.9$ </td><td> $5.3 \pm 0.6$ </td><td> $\underline{19.6} \pm 12.4$ </td><td> $8.1 \pm 2.4$ </td></tr><tr><td>hopper-medium</td><td> $60.9 \pm 7.6$ </td><td> $\underline{65.2} \pm 4.2$ </td><td> $61.9 \pm 6.4$ </td><td> $91.1 \pm 10.1$ </td><td> $\underline{102.0} \pm 1.0$ </td></tr><tr><td>hopper-expert</td><td> $\underline{109.6} \pm 3.7$ </td><td> $108.8 \pm 3.1$ </td><td> $106.5 \pm 9.1$ </td><td> $\underline{109.8} \pm 0.5$ </td><td> $100.1 \pm 8.3$ </td></tr><tr><td>hopper-medium-expert</td><td> $87.8 \pm 10.5$ </td><td> $85.5 \pm 29.7$ </td><td> $96.9 \pm 15.1$ </td><td> $\underline{109.8} \pm 0.6$ </td><td> $\underline{107.0} \pm 6.4$ </td></tr><tr><td>hopper-medium-replay</td><td> $55.1 \pm 31.7$ </td><td> $89.6 \pm 13.2$ </td><td> $86.3 \pm 7.3$ </td><td> $\underline{97.2} \pm 9.0$ </td><td> $\underline{98.1} \pm 5.3$ </td></tr><tr><td>hopper-full-replay</td><td> $97.9 \pm 17.5$ </td><td> $104.4 \pm 10.8$ </td><td> $101.9 \pm 0.6$ </td><td> $\underline{107.4} \pm 0.8$ </td><td> $\underline{107.1} \pm 0.4$ </td></tr><tr><td>walker2d-random</td><td> $2.0 \pm 3.6$ </td><td> $11.3 \pm 7.0$ </td><td> $5.1 \pm 1.7$ </td><td> $\underline{18.7} \pm 6.9$ </td><td> $\underline{18.4} \pm 4.5$ </td></tr><tr><td>walker2d-medium</td><td> $77.7 \pm 2.9$ </td><td> $80.7 \pm 3.4$ </td><td> $79.5 \pm 3.2$ </td><td> $\underline{92.7} \pm 1.2$ </td><td> $\underline{82.5} \pm 3.6$ </td></tr><tr><td>walker2d-expert</td><td> $\underline{110.0} \pm 0.6$ </td><td> $96.9 \pm 32.3$ </td><td> $109.3 \pm 0.1$ </td><td> $104.5 \pm 22.8$ </td><td> $\underline{112.3} \pm 0.2$ </td></tr><tr><td>walker2d-medium-expert</td><td> $110.4 \pm 0.6$ </td><td> $\underline{112.1} \pm 0.5$ </td><td> $109.1 \pm 0.2$ </td><td> $104.6 \pm 11.2$ </td><td> $\underline{111.6} \pm 0.3$ </td></tr><tr><td>walker2d-medium-replay</td><td> $68.0 \pm 19.2$ </td><td> $75.4 \pm 9.3$ </td><td> $76.8 \pm 10.0$ </td><td> $\underline{89.4} \pm 3.8$ </td><td> $\underline{77.3} \pm 7.9$ </td></tr><tr><td>walker2d-full-replay</td><td> $90.3 \pm 5.4$ </td><td> $97.5 \pm 1.4$ </td><td> $94.2 \pm 1.9$ </td><td> $\underline{105.3} \pm 3.2$ </td><td> $\underline{102.2} \pm 1.7$ </td></tr><tr><td>Average</td><td>70.3</td><td>72.9</td><td>73.6</td><td>82.6</td><td>81.2</td></tr></table>

Table 3: Average normalized score over the final evaluation and ten unseen training seeds on AntMaze tasks. CQL scores were taken from Ghasemipour et al. (2022). The symbol ± represents the standard deviation across the seeds. To make a fair comparison against TD3+BC and IQL, we extensively tuned their hyperparameters. 

<table><tr><td>Task Name</td><td>TD3+BC</td><td>IQL</td><td>CQL</td><td>SAC-RND</td><td>ReBRAC, our</td></tr><tr><td>antmaze-umaze</td><td>66.3 ± 6.2</td><td>83.3 ± 4.5</td><td>74.0</td><td>97.0 ± 1.5</td><td>97.8 ± 1.0</td></tr><tr><td>antmaze-umaze-diverse</td><td>53.8 ± 8.5</td><td>70.6 ± 3.7</td><td>84.0</td><td>66.0 ± 25.0</td><td>88.3 ± 13.0</td></tr><tr><td>antmaze-medium-play</td><td>26.5 ± 18.4</td><td>64.6 ± 4.9</td><td>61.2</td><td>38.5 ± 29.4</td><td>84.0 ± 4.2</td></tr><tr><td>antmaze-medium-diverse</td><td>25.9 ± 15.3</td><td>61.7 ± 6.1</td><td>53.7</td><td>74.7 ± 10.7</td><td>76.3 ± 13.5</td></tr><tr><td>antmaze-large-play</td><td>0.0 ± 0.0</td><td>42.5 ± 6.5</td><td>15.8</td><td>43.9 ± 29.2</td><td>60.4 ± 26.1</td></tr><tr><td>antmaze-large-diverse</td><td>0.0 ± 0.0</td><td>27.6 ± 7.8</td><td>14.9</td><td>45.7 ± 28.5</td><td>54.4 ± 25.1</td></tr><tr><td>Average</td><td>28.7</td><td>58.3</td><td>50.6</td><td>60.9</td><td>76.8</td></tr></table>

domain. In other cases, we report results from previous works, meaning that scores for other methods can be lower if evaluated under our protocol.

The results of our tests on D4RL's Gym-MuJoCo, AntMaze, and Adroit tasks are available in Table 2, Table 3, Table 4, respectively. The mean-wise best results among algorithms are highlighted with bold, and the second best performance is underlined. Our approach, ReBRAC, achieves state-of-the-art results on Gym-MuJoCo, AntMaze, and Adroit tasks outperforming all baselines on average, except SAC-RND on Gym-MuJoCo tasks, which is slightly better. Performance profiles and probability of improvement (Agarwal et al., 2021) in Figure 1b and Figure 1c also demonstrate that ReBRAC is competitive when compared to the algorithms that we contrast against. Our method is also comparable to ensemble-based approaches (see Appendix C for additional comparisons).

# 4.2 Evaluation on offline V-D4RL

In addition to testing ReBRAC on D4RL, we evaluated its performance on V-D4RL benchmark (Lu et al., 2022). Our motivation for doing so was the fact that scores on D4RL Gym-MuJoCo tasks have saturated in recent years, and even after the introduction of ensemble-based offline RL methods

Table 4: Average normalized score over the final evaluation and ten unseen training seeds on Adroit tasks. BC and CQL scores were taken from Yang et al. (2022). The symbol ± represents the standard deviation across the seeds. To make a fair comparison against TD3+BC and IQL, we extensively tuned their hyperparameters. 

<table><tr><td>Task Name</td><td>BC</td><td>TD3+BC</td><td>IQL</td><td>CQL</td><td>SAC-RND</td><td>ReBRAC, our</td></tr><tr><td>pen-human</td><td>34.4</td><td>81.8 ± 14.9</td><td>81.5 ± 17.5</td><td>37.5</td><td>5.6 ± 5.8</td><td>103.5 ± 14.1</td></tr><tr><td>pen-cloned</td><td>56.9</td><td>61.4 ± 19.3</td><td>77.2 ± 17.7</td><td>39.2</td><td>2.5 ± 6.1</td><td>91.8 ± 21.7</td></tr><tr><td>pen-expert</td><td>85.1</td><td>146.0 ± 7.3</td><td>133.6 ± 16.0</td><td>107.0</td><td>45.4 ± 22.9</td><td>154.1 ± 5.4</td></tr><tr><td>door-human</td><td>0.5</td><td>-0.1 ± 0.0</td><td>3.1 ± 2.0</td><td>9.9</td><td>0.0 ± 0.1</td><td>0.0 ± 0.0</td></tr><tr><td>door-cloned</td><td>-0.1</td><td>0.1 ± 0.6</td><td>0.8 ± 1.0</td><td>0.4</td><td>0.2 ± 0.8</td><td>1.1 ± 2.6</td></tr><tr><td>door-expert</td><td>34.9</td><td>84.6 ± 44.5</td><td>105.3 ± 2.8</td><td>101.5</td><td>73.6 ± 26.7</td><td>104.6 ± 2.4</td></tr><tr><td>hammer-human</td><td>1.5</td><td>0.4 ± 0.4</td><td>2.5 ± 1.9</td><td>4.4</td><td>-0.1 ± 0.1</td><td>0.2 ± 0.2</td></tr><tr><td>hammer-cloned</td><td>0.8</td><td>0.8 ± 0.7</td><td>1.1 ± 0.5</td><td>2.1</td><td>0.1 ± 0.4</td><td>6.7 ± 3.7</td></tr><tr><td>hammer-expert</td><td>125.6</td><td>117.0 ± 30.9</td><td>129.6 ± 0.5</td><td>86.7</td><td>24.8 ± 39.4</td><td>133.8 ± 0.7</td></tr><tr><td>relocate-human</td><td>0.0</td><td>-0.2 ± 0.0</td><td>0.1 ± 0.1</td><td>0.2</td><td>0.0 ± 0.0</td><td>0.0 ± 0.0</td></tr><tr><td>relocate-cloned</td><td>-0.1</td><td>-0.1 ± 0.1</td><td>0.2 ± 0.4</td><td>-0.1</td><td>0.0 ± 0.0</td><td>0.9 ± 1.6</td></tr><tr><td>relocate-expert</td><td>101.3</td><td>107.3 ± 1.6</td><td>106.5 ± 2.5</td><td>95.0</td><td>3.4 ± 4.5</td><td>106.6 ± 3.2</td></tr><tr><td>Average w/o expert</td><td>11.7</td><td>18.0</td><td>20.8</td><td>11.7</td><td>1.0</td><td>25.5</td></tr><tr><td>Average</td><td>36.7</td><td>49.9</td><td>53.4</td><td>40.3</td><td>12.9</td><td>58.6</td></tr></table>

Table 5: Average normalized score over the final evaluation and five unseen training seeds on V-D4RL tasks. The score is mapped from a range of $[0, 1000]$ to $[0, 100]$ . The symbol $\pm$ represents the standard deviation across the seeds. 

<table><tr><td colspan="2">Environment</td><td>Offline DV2</td><td>DrQ+BC</td><td>CQL</td><td>BC</td><td>LOMPO</td><td>ReBRAC, our</td></tr><tr><td rowspan="5">walker-walk</td><td>random</td><td> $28.7 \pm 13.0$ </td><td> $5.5 \pm 0.9$ </td><td> $14.4 \pm 12.4$ </td><td> $2.0 \pm 0.2$ </td><td> $21.9 \pm 8.1$ </td><td> $15.9 \pm 2.3$ </td></tr><tr><td>mixed</td><td> $56.5 \pm 18.1$ </td><td> $28.7 \pm 6.9$ </td><td> $11.4 \pm 12.4$ </td><td> $16.5 \pm 4.3$ </td><td> $34.7 \pm 19.7$ </td><td> $41.6 \pm 8.0$ </td></tr><tr><td>medium</td><td> $34.1 \pm 19.7$ </td><td> $46.8 \pm 2.3$ </td><td> $14.8 \pm 16.1$ </td><td> $40.9 \pm 3.1$ </td><td> $43.4 \pm 11.1$ </td><td> $52.5 \pm 3.2$ </td></tr><tr><td>medexp</td><td> $43.9 \pm 34.4$ </td><td> $86.4 \pm 5.6$ </td><td> $56.4 \pm 38.4$ </td><td> $47.7 \pm 3.9$ </td><td> $39.2 \pm 19.5$ </td><td> $92.7 \pm 1.3$ </td></tr><tr><td>expert</td><td> $4.8 \pm 0.6$ </td><td> $68.4 \pm 7.5$ </td><td> $89.6 \pm 6.0$ </td><td> $91.5 \pm 3.9$ </td><td> $5.3 \pm 7.7$ </td><td> $81.4 \pm 10.0$ </td></tr><tr><td rowspan="5">cheetah-run</td><td>random</td><td> $31.7 \pm 2.7$ </td><td> $5.8 \pm 0.6$ </td><td> $5.9 \pm 8.4$ </td><td> $0.0 \pm 0.0$ </td><td> $11.4 \pm 5.1$ </td><td> $12.9 \pm 2.2$ </td></tr><tr><td>mixed</td><td> $61.6 \pm 1.0$ </td><td> $44.8 \pm 3.6$ </td><td> $10.7 \pm 12.8$ </td><td> $25.0 \pm 3.6$ </td><td> $36.3 \pm 13.6$ </td><td> $46.8 \pm 0.7$ </td></tr><tr><td>medium</td><td> $17.2 \pm 3.5$ </td><td> $53.0 \pm 3.0$ </td><td> $40.9 \pm 5.1$ </td><td> $51.6 \pm 1.4$ </td><td> $16.4 \pm 8.3$ </td><td> $59.0 \pm 0.7$ </td></tr><tr><td>medexp</td><td> $10.4 \pm 3.5$ </td><td> $50.6 \pm 8.2$ </td><td> $20.9 \pm 5.5$ </td><td> $57.5 \pm 6.3$ </td><td> $11.9 \pm 1.9$ </td><td> $58.3 \pm 11.7$ </td></tr><tr><td>expert</td><td> $10.9 \pm 3.2$ </td><td> $34.5 \pm 8.3$ </td><td> $61.5 \pm 4.3$ </td><td> $67.4 \pm 6.8$ </td><td> $14.0 \pm 3.8$ </td><td> $35.6 \pm 5.3$ </td></tr><tr><td rowspan="5">humanoid-walk</td><td>random</td><td> $0.1 \pm 0.0$ </td><td> $0.1 \pm 0.0$ </td><td> $0.2 \pm 0.1$ </td><td> $0.1 \pm 0.0$ </td><td> $0.1 \pm 0.0$ </td><td> $0.1 \pm 0.0$ </td></tr><tr><td>mixed</td><td> $0.2 \pm 0.1$ </td><td> $15.9 \pm 3.8$ </td><td> $0.1 \pm 0.0$ </td><td> $18.8 \pm 4.2$ </td><td> $0.2 \pm 0.0$ </td><td> $16.0 \pm 2.7$ </td></tr><tr><td>medium</td><td> $0.2 \pm 0.1$ </td><td> $6.2 \pm 2.4$ </td><td> $0.1 \pm 0.0$ </td><td> $13.5 \pm 4.1$ </td><td> $0.1 \pm 0.0$ </td><td> $9.0 \pm 2.3$ </td></tr><tr><td>medexp</td><td> $0.1 \pm 0.0$ </td><td> $7.0 \pm 2.3$ </td><td> $0.1 \pm 0.0$ </td><td> $17.2 \pm 4.7$ </td><td> $0.2 \pm 0.0$ </td><td> $7.8 \pm 2.4$ </td></tr><tr><td>expert</td><td> $0.2 \pm 0.1$ </td><td> $2.7 \pm 0.9$ </td><td> $1.6 \pm 0.5$ </td><td> $6.1 \pm 3.7$ </td><td> $0.1 \pm 0.0$ </td><td> $2.9 \pm 0.9$ </td></tr><tr><td colspan="2">Average</td><td>20.0</td><td>30.4</td><td>21.9</td><td>30.3</td><td>15.6</td><td>35.5</td></tr></table>

by An et al. (2021), there has been no notable progress on these tasks. On the other hand, V-D4RL provides a similar set of problems, with datasets collected in the same way as in D4RL but with the agent's observations now being images from the environment.

We tested our algorithm on all available single-task datasets without distractors and compared it to the baselines from the original V-D4RL work (Lu et al., 2022). The results are reported in Table 5. Our proposed approach achieves state-of-the-art or close-to-state-of-the-art results on most of the tasks, and it is the only method that, on average, performs notably better than naive Behavioral Cloning.

# 4.3 Evaluation on offline-to-online D4RL

The evaluation of offline-to-online performance is a pivotal aspect for reinforcement learning (RL) algorithms, particularly in light of recent developments. In this context, we conducted additional tests on ReBRAC, as it stands out as a promising algorithm for several compelling reasons.

First and foremost, ReBRAC demonstrates a remarkable proficiency following offline pre-training. Secondly, our algorithm shares notable similarities with TD3+BC, a method that has exhibited effectiveness in online fine-tuning as observed by Beeson et al. (Beeson & Montana, 2022)

For the sake of simplicity, we opted to disable critic penalization during the online fine-tuning. Furthermore, we linearly decay the actor's penalty to half of its initial value, following the approach described by Beeson & Montana (2022). Notably, no hyperparameter tuning was performed in this process.

To evaluate our approach in the offline-to-online setting, we followed the methodology outlined by Tarasov et al. (2022). In our comparative analysis, we consider the following algorithms: TD3+BC (Fujimoto & Gu, 2021), IQL (Kostrikov et al., 2021), CQL (Kumar et al., 2020), SPOT (Wu et al., 2022), and Cal-CQL (Nakamoto et al., 2023). The scores after the offline stage and online tuning, are reported in Table 6. We also provide finetuning cumulative regret proposed by Nakamoto et al. (2023) in Table 7.

Table 6: Normalized performance after offline pretraining and online finetuning on D4RL. Baselines scores except TD3+BC are taken from Tarasov et al. (2022). ReBRAC and TD3+BC scores are averaged over ten random seeds, and all others are averaged over four as in Tarasov et al. (2022). 

<table><tr><td>Task Name</td><td>TD3 + BC</td><td>IQL</td><td>SPOT</td><td>Cal-QL</td><td>ReBRAC, our</td></tr><tr><td>antmaze-u-v2</td><td>66.8 → 91.4</td><td>77.00 → 96.50</td><td>91.00 → 99.50</td><td>76.75 → 99.75</td><td>97.8 → 99.8</td></tr><tr><td>antmaze-u-d-v2</td><td>59.1 → 48.4</td><td>59.50 → 63.75</td><td>36.25 → 95.00</td><td>32.00 → 98.50</td><td>85.7 → 98.1</td></tr><tr><td>antmaze-m-p-v2</td><td>59.2 → 94.8</td><td>71.75 → 89.75</td><td>67.25 → 97.25</td><td>71.75 → 98.75</td><td>78.4 → 97.7</td></tr><tr><td>antmaze-m-d-v2</td><td>62.6 → 94.1</td><td>64.25 → 92.25</td><td>73.75 → 94.50</td><td>62.00 → 98.25</td><td>78.6 → 98.5</td></tr><tr><td>antmaze-l-p-v2</td><td>21.5 → 0.1</td><td>38.50 → 64.50</td><td>31.50 → 87.00</td><td>31.75 → 97.25</td><td>47.0 → 39.5</td></tr><tr><td>antmaze-l-d-v2</td><td>9.5 → 0.4</td><td>26.75 → 64.25</td><td>17.50 → 81.00</td><td>44.00 → 91.50</td><td>66.7 → 77.6</td></tr><tr><td>AntMaze avg</td><td>46.4 → 54.8 (+8.4)</td><td>56.29 → 78.50 (+22.21)</td><td>52.88 → 92.38 (+39.50)</td><td>53.04 → 97.33 (+24.29)</td><td>75.7 → 85.2 (+9.5)</td></tr><tr><td>pen-c-v1</td><td>86.1 → 110.3</td><td>84.19 → 102.02</td><td>6.19 → 43.63</td><td>-2.66 → -2.68</td><td>91.8 → 152.0</td></tr><tr><td>door-c-v1</td><td>0.0 → 3.4</td><td>1.19 → 20.34</td><td>-0.21 → 0.02</td><td>-0.33 → -0.33</td><td>0.4 → 104.9</td></tr><tr><td>hammer-c-v1</td><td>2.4 → 11.6</td><td>1.35 → 57.27</td><td>3.97 → 3.73</td><td>0.25 → 0.17</td><td>4.1 → 131.2</td></tr><tr><td>relocate-c-v1</td><td>-0.1 → 0.1</td><td>0.04 → 0.32</td><td>-0.24 → -0.15</td><td>-0.31 → -0.31</td><td>0.0 → 12.3</td></tr><tr><td>Adroit Avg</td><td>22.1 → 31.3 (+9.2)</td><td>21.69 → 44.99 (+23.3)</td><td>2.43 → 11.81 (+9.38)</td><td>-0.76 → -0.79 (-0.03)</td><td>24.0 → 100.1 (+76.1)</td></tr><tr><td>Total avg</td><td>36.7 → 45.4 (+8.7)</td><td>42.45 → 65.10 (+22.65)</td><td>32.70 → 60.15 (+27.45)</td><td>31.52 → 58.08 (+26.56)</td><td>55.0 → 91.1 (+36.1)</td></tr></table>

Table 7: Cumulative regret of online finetuning calculated as 1 - average success rate. Baselines scores except TD3+BC are taken from Tarasov et al. (2022). ReBRAC and TD3+BC regrets are averaged over ten random seeds, and all others are averaged over four as in Tarasov et al. (2022). 

<table><tr><td>Task Name</td><td>TD3 + BC</td><td>CQL</td><td>IQL</td><td>SPOT</td><td>Cal-QL</td><td>ReBRAC, our</td></tr><tr><td>antmaze-umaze-v2</td><td> $0.09 \pm 0.08$ </td><td> $0.02 \pm 0.00$ </td><td> $0.07 \pm 0.00$ </td><td> $0.02 \pm 0.00$ </td><td> $0.01 \pm 0.00$ </td><td> $\mathbf{0.00} \pm 0.00$ </td></tr><tr><td>antmaze-umaze-diverse-v2</td><td> $0.47 \pm 0.16$ </td><td> $0.09 \pm 0.01$ </td><td> $0.43 \pm 0.11$ </td><td> $0.22 \pm 0.07$ </td><td> $\mathbf{0.05} \pm 0.01$ </td><td> $0.06 \pm 0.13$ </td></tr><tr><td>antmaze-medium-play-v2</td><td> $0.12 \pm 0.05$ </td><td> $0.08 \pm 0.01$ </td><td> $0.09 \pm 0.01$ </td><td> $0.06 \pm 0.00$ </td><td> $0.04 \pm 0.01$ </td><td> $\mathbf{0.03} \pm 0.01$ </td></tr><tr><td>antmaze-medium-diverse-v2</td><td> $0.09 \pm 0.02$ </td><td> $0.08 \pm 0.00$ </td><td> $0.10 \pm 0.01$ </td><td> $0.05 \pm 0.01$ </td><td> $0.04 \pm 0.01$ </td><td> $\mathbf{0.02} \pm 0.00$ </td></tr><tr><td>antmaze-large-play-v2</td><td> $0.99 \pm 0.00$ </td><td> $0.21 \pm 0.02$ </td><td> $0.34 \pm 0.05$ </td><td> $0.29 \pm 0.07$ </td><td> $\mathbf{0.13} \pm 0.02$ </td><td> $0.36 \pm 0.30$ </td></tr><tr><td>antmaze-large-diverse-v2</td><td> $0.99 \pm 0.01$ </td><td> $0.21 \pm 0.03$ </td><td> $0.41 \pm 0.03$ </td><td> $0.23 \pm 0.08$ </td><td> $0.13 \pm 0.02$ </td><td> $\mathbf{0.10} \pm 0.07$ </td></tr><tr><td>AntMaze avg</td><td>0.45</td><td>0.11</td><td>0.24</td><td>0.15</td><td>0.07</td><td>0.09</td></tr><tr><td>pen-cloned-v1</td><td> $0.30 \pm 0.10$ </td><td> $0.97 \pm 0.00$ </td><td> $0.37 \pm 0.01$ </td><td> $0.58 \pm 0.02$ </td><td> $0.98 \pm 0.01$ </td><td> $\mathbf{0.08} \pm 0.00$ </td></tr><tr><td>door-cloned-v1</td><td> $0.97 \pm 0.03$ </td><td> $1.00 \pm 0.00$ </td><td> $0.83 \pm 0.03$ </td><td> $0.99 \pm 0.01$ </td><td> $1.00 \pm 0.00$ </td><td> $\mathbf{0.26} \pm 0.10$ </td></tr><tr><td>hammer-cloned-v1</td><td> $0.92 \pm 0.14$ </td><td> $1.00 \pm 0.00$ </td><td> $0.65 \pm 0.10$ </td><td> $0.98 \pm 0.01$ </td><td> $1.00 \pm 0.00$ </td><td> $\mathbf{0.13} \pm 0.02$ </td></tr><tr><td>relocate-cloned-v1</td><td> $0.99 \pm 0.01$ </td><td> $1.00 \pm 0.00$ </td><td> $1.00 \pm 0.00$ </td><td> $1.00 \pm 0.00$ </td><td> $1.00 \pm 0.00$ </td><td> $\mathbf{0.85} \pm 0.07$ </td></tr><tr><td>Adroit avg</td><td>0.79</td><td>0.99</td><td>0.71</td><td>0.89</td><td>0.99</td><td>0.33</td></tr><tr><td>Total avg</td><td>0.59</td><td>0.47</td><td>0.43</td><td>0.44</td><td>0.44</td><td>0.18</td></tr></table>

ReBRAC exhibits competitive performance, surpassing four out of six AntMaze datasets and achieving state-of-the-art results in terms of final scores on Adroit tasks. On average, ReBRAC outperforms its closest competitor, Cal-CQL, which was specifically designed for the offline-to-online problem in the concurrent work (Nakamoto et al., 2023).

Regarding regret, ReBRAC outperforms all other algorithms, with Cal-QL being the sole exception, showing slightly better results on average only in the AntMaze domain.

# 4.4 Ablating Design Choices

To better understand the source of improved performance, we conducted an ablation study on the modifications made to the algorithm. Results can be found in Table 8. Additional ablation studies

for all datasets can be found in Appendix G. One modification at a time was disabled, while all other changes were retained, including layer normalization in the critic network, additional linear layers in the actor and critic networks, adding an MSE penalty to the critic and actor loss. In the case of AntMaze, we also attempted to use the default $\gamma$ value instead of the increased one. To further demonstrate the efficacy of our modifications, we also ran our implementation as equivalent to the original TD3+BC, with all changes disabled and hyperparameters were taken from the original paper. This experiment serves to show that the improved scores are due to the proposed changes in the algorithm and not just different implementations. Furthermore, we searched for the regularization parameter for our implementation of TD3+BC to demonstrate that tuning this parameter is not the sole source of improvement. Moreover, we tested the TD3 + BC by adding each of the ReBRAC's modifications independently to demonstrate that each individual modifications is not sufficient for achieving performance of ReBRAC where the combination of modification appear.

Additionally, we run ablation which to validate the importance of decoupling by disabling it and searching the best penalty parameter value from the set of all previously used values listed in the Appendix B.

Table 8: ReBRAC's design choices ablations: each modification was disabled while keeping all the others. For brevity, we report the mean of average normalized scores over four unseen training seeds across domains. We also include tuned results for TD3+BC to highlight that the improvement does not come from hyperparameter search. For dataset-specific results, please refer to Appendix G. 

<table><tr><td>Ablation</td><td>Gym-MuJoCo</td><td>AntMaze</td><td>Adroit</td><td>All</td></tr><tr><td>TD3+BC, paper</td><td>-</td><td>27.3</td><td>0.0</td><td>-</td></tr><tr><td>TD3+BC, our</td><td>63.4</td><td>18.5</td><td>52.3</td><td>52.2</td></tr><tr><td>TD3+BC, tuned</td><td>71.8 (-10.9%)</td><td>27.9 (-62.9%)</td><td>53.5 (-25.9%)</td><td>58.3 (-19.2%)</td></tr><tr><td>TD3+BC w/ γ change</td><td>-</td><td>17.5 (-76.7%)</td><td>-</td><td>-</td></tr><tr><td>TD3+BC w/ LN</td><td>71.4 (-11.4%)</td><td>35.6 (-52.7%)</td><td>55.6 (-4.1%)</td><td>60.2 (-16.6%)</td></tr><tr><td>TD3+BC w/ large batch</td><td>14.4 (-82.1%)</td><td>0.0 (-100.0%)</td><td>1.6 (-97.2%)</td><td>7.9 (-89.0%)</td></tr><tr><td>TD3+BC w/ layer</td><td>71.2 (-11.6%)</td><td>44.1 (-41.4%)</td><td>56.4 (-2.7%)</td><td>61.9 (-14.2%)</td></tr><tr><td>ReBRAC w/o large batch</td><td>75.9 (-5.8%)</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ReBRAC w/ large batch</td><td>-</td><td>41.0 (-45.6%)</td><td>55.4 (-4.6%)</td><td>-</td></tr><tr><td>ReBRAC w/o γ change</td><td>-</td><td>21.0 (-72.1%)</td><td>-</td><td>-</td></tr><tr><td>ReBRAC w/o LN</td><td>59.2 (-26.5%)</td><td>0.0 (-100.0%)</td><td>25.1 (-56.7%)</td><td>38.0 (-47.3%)</td></tr><tr><td>ReBRAC w/o layer</td><td>78.5 (-2.6%)</td><td>18.1 (-75.9%)</td><td>59.0 (+1.7%)</td><td>61.9 (-14.2%)</td></tr><tr><td>ReBRAC w/o actor penalty</td><td>22.8 (-71.7%)</td><td>0.1 (-99.8%)</td><td>0.0 (-100.0%)</td><td>11.4 (-84.2%)</td></tr><tr><td>ReBRAC w/o critic penalty</td><td>81.1 (+0.6%)</td><td>72.2 (-4.1%)</td><td>56.9 (-1.8%)</td><td>71.5 (-0.9%)</td></tr><tr><td>ReBRAC w/o decoupling</td><td>79.8 (-0.9%)</td><td>76.9 (+2.1%)</td><td>56.7 (-2.2%)</td><td>71.6 (-0.8%)</td></tr><tr><td>ReBRAC</td><td>80.6</td><td>75.3</td><td>58.0</td><td>72.2</td></tr></table>

The ablation results show that ReBRAC outperforms TD3+BC not because of different implementations or actor regularization parameter choice. All domains suffer when the LayerNorm is disabled, leading to the halved average performance overall. Removing additional layers leads to a notable drop in AntMaze tasks, while on Gym-MuJoCo decrease is small, and on Adroit tasks, we can see a slight boost. The algorithm fails to learn most tasks when the actor penalty is disabled. Notably, the critic penalty plays a minor role in improving the performance on most of the problems as well as the decoupling penalties. Using standard batch size on Gym-MuJoCo tasks significantly decreases final scores, while using the increased discount factor for AntMaze is crucial for obtaining state-of-the-art performance.

Based on the conducted ablations studies, the proposed configuration of design choices leads to the best performance on average. Note that tuning these choices for each task independently makes it possible to get even higher scores than we report in Section 4.1. We also can change the number of hidden layers in the networks, leading to better performance, see Section 4.5. But in real life, algorithm evaluation might be costly, so we limit ourselves to tuning only regularization parameters.

# 4.5 Stacking Even More Layers

As ablations show, the depth of the network plays an important role when solving AntMaze and HalfCheetah tasks (see Appendix G). We conduct additional experiments to check how the perfor-

mance depends on the network depth in more detail. Ball et al. (2023) used networks of depth four when solving AntMaze tasks. Our goal is to find the point where the performance saturates. For this, we run our algorithm on AntMaze tasks increasing the number of layers to six while keeping other parameters unchanged. We attempt to increase actor and critic separately and at once. We also decreased each network's size by one layer similarly. Results can be found in Figure 2.

Several conclusions can be drawn from the results. First, further increments in the number of layers can lead to better results on the AntMaze domain when layers are scaled up to five. For six layers, performance drops or does not improve. Second, decreasing the critic's size leads to the worst performance on most datasets. Lastly, there is no clear pattern on how the performance changes even within a single domain. The drop on six layers is the only common feature that can be seen. On average, four critic layers and three actor layers were the best. Changing only the actor's network is more stable on average.

![](images/2b44279c4c857a3e6790066ea81ee7b91add4c644cdb1dc65e9d550299eaffc9.jpg)  
Figure 2: Impact of networks' depth on the final performance for the AntMaze tasks. Scores are averaged over four unseen training seeds. Shaded areas represent one standard deviation across seeds. These graphics demonstrate that one can achieve marginally better scores by tuning the number of layers for certain tasks.

# 4.6 Penalization Sensitivity Analysis

Following Kurenkov & Kolesnikov (2022), we demonstrate the sensitivity of ReBRAC to the choice of $\beta_{1}$ and $\beta_{2}$ hyperparameters under uniform policy selection on D4RL tasks using Expected Online Performance (EOP) and comparing it to the TD3+BC and IQL. EOP shows the best performance expected depending on the number of policies that can be deployed for online evaluation. Results are demonstrated in Table 9. As one can see, approximately ten policies are required for ReBRAC to attain ensemble-free state-of-the-art performance. ReBRAC's EOP is higher for any number of online policies when compared to TD3+BC and better than IQL on all domains when the evaluation budget is larger than two policies. See Appendix F for EOP separated by tasks.

Table 9: Expected Online Performance (Kurenkov & Kolesnikov, 2022) under uniform policy selection aggregated over D4RL domains across four training seeds. This demonstrates the sensitivity to the choice of hyperparameters given a certain budget for online evaluation. For dataset-specific results, please see Appendix F. 

<table><tr><td>Domain</td><td>Algorithm</td><td>1 policy</td><td>2 policies</td><td>3 policies</td><td>5 policies</td><td>10 policies</td><td>15 policies</td><td>20 policies</td></tr><tr><td rowspan="3">Gym-MuJoCo</td><td>TD3+BC</td><td> $49.8 \pm 21.4$ </td><td> $61.0 \pm 14.5$ </td><td> $65.3 \pm 9.3$ </td><td> $67.8 \pm 3.9$ </td><td>-</td><td>-</td><td>-</td></tr><tr><td>IQL</td><td> $65.0 \pm 9.1$ </td><td> $69.9 \pm 5.6$ </td><td> $71.7 \pm 3.5$ </td><td> $72.9 \pm 1.7$ </td><td> $73.6 \pm 0.8$ </td><td> $73.8 \pm 0.7$ </td><td> $74.0 \pm 0.6$ </td></tr><tr><td>ReBRAC</td><td> $62.0 \pm 17.1$ </td><td> $70.6 \pm 9.9$ </td><td> $73.3 \pm 5.5$ </td><td> $74.8 \pm 2.1$ </td><td> $75.6 \pm 0.8$ </td><td> $75.8 \pm 0.6$ </td><td> $76.0 \pm 0.5$ </td></tr><tr><td rowspan="3">AntMaze</td><td>TD3+BC</td><td> $6.9 \pm 7.0$ </td><td> $10.7 \pm 6.8$ </td><td> $13.0 \pm 6.0$ </td><td> $15.5 \pm 4.6$ </td><td>-</td><td>-</td><td>-</td></tr><tr><td>IQL</td><td> $29.8 \pm 15.5$ </td><td> $38.0 \pm 15.4$ </td><td> $43.1 \pm 13.8$ </td><td> $48.7 \pm 10.2$ </td><td> $53.2 \pm 4.4$ </td><td> $54.3 \pm 2.1$ </td><td> $54.7 \pm 1.2$ </td></tr><tr><td>ReBRAC</td><td> $67.9 \pm 10.0$ </td><td> $73.6 \pm 7.4$ </td><td> $76.1 \pm 5.5$ </td><td> $78.3 \pm 3.4$ </td><td> $79.9 \pm 1.7$ </td><td> $80.4 \pm 1.1$ </td><td>-</td></tr><tr><td rowspan="3">Adroit</td><td>TD3+BC</td><td> $23.6 \pm 19.9$ </td><td> $34.6 \pm 17.7$ </td><td> $40.6 \pm 14.5$ </td><td> $46.4 \pm 9.8$ </td><td>-</td><td>-</td><td>-</td></tr><tr><td>IQL</td><td> $53.1 \pm 0.7$ </td><td> $53.5 \pm 0.6$ </td><td> $53.7 \pm 0.5$ </td><td> $53.9 \pm 0.3$ </td><td> $54.1 \pm 0.2$ </td><td> $54.2 \pm 0.2$ </td><td> $54.2 \pm 0.1$ </td></tr><tr><td>ReBRAC</td><td> $44.1 \pm 18.4$ </td><td> $53.2 \pm 10.9$ </td><td> $56.1 \pm 6.1$ </td><td> $57.8 \pm 2.3$ </td><td> $58.6 \pm 0.9$ </td><td> $58.9 \pm 0.7$ </td><td> $59.1 \pm 0.6$ </td></tr></table>

# 5 Related Work

Ensemble-free offline RL methods. In recent years, many offline reinforcement learning algorithms were developed. TD3+BC (Fujimoto & Gu, 2021) represents a minimalist approach to offline RL, which incorporates a Behavioral Cloning component into the actor loss, enabling online actor-critic algorithms to operate in an offline setting. CQL (Kumar et al., 2020) drives the critic network to assign lower values to out-of-distribution state-action pairs and higher values to in-distribution pairs. IQL (Kostrikov et al., 2021) proposes a method for learning a policy without sampling out-of-distribution actions.

Despite this, more sophisticated methods may be necessary to achieve state-of-the-art results in an ensemble-free setup. For instance, Chen et al. (2022b); Akimov et al. (2022) pre-train different forms of encoders for actions, then optimize the actor to predict actions in the latent space. SPOT (Wu et al., 2022) pre-trains Variational Autoencoder and uses its uncertainty to penalize actor for sampling OOD actions while SAC-RND (Nikulin et al., 2023) applies Random Network Distillation and penalizes actor and critic.

Ensemble-based offline RL methods. A significant number of works in offline reinforcement learning have also leveraged ensemble methods for uncertainty estimation. The recently introduced SAC-N (An et al., 2021) algorithm outperformed all previous approaches on D4RL Gym-MuJoCo tasks; however, it necessitated large ensembles for some tasks, such as the hopper task, which required an ensemble size of 500 and imposed a significant computational burden. To mitigate this, the EDAC algorithm was introduced in the same work, which utilized ensemble diversification to reduce the ensemble size from 500 to 50. Despite the reduction, the ensemble size remains substantial compared to ensemble-free alternatives. It is worth mentioning that neither SAC-N nor EDAC is capable of solving the complex AntMaze tasks (Tarasov et al., 2022).

Another state-of-the-art algorithm in the Gym-MuJoCo tasks is RORL (Yang et al., 2022), which is a modification of SAC-N that makes the Q function more robust and smooth by perturbing state-action pairs with the use of out-of-distribution actions. RORL also requires an ensemble size of up to 20. On the other hand, MSG (Ghasemipour et al., 2022) utilizes independent targets for each ensemble member and achieves good performance on the Gym-MuJoCo tasks with an ensemble size of four but requires 64 ensemble members to achieve state-of-the-art performance on the AntMaze tasks.

Design choices ablations. Ablations of different design choices on established baselines are very limited, especially in offline RL. It is only shown by Fujimoto & Gu (2021) that CQL performs poorly if proposed non-algorithmic differences are eliminated. A parallel study (Ball et al., 2023) shows that some of the considered modifications (LayerNorm and networks depth) are important when used in online RL with offline data setting, which is different from pure offline.

# 6 Conclusion, Limitations, and Future work

In this work, we revisit recent advancements in the offline RL field over the last two years and incorporate a modest set of improvements to a previously established minimalistic TD3+BC baseline. Our experiments demonstrate that despite these limited updates, we can achieve more than competitive results on offline and offline-to-online D4RL and offline V-D4RL benchmarks under different hyperparameter budgets.

Despite the noteworthy results, our work is limited to one approach and a subset of possible design changes. It is imperative to explore the potential impact of these modifications on other offline RL methods (e.g., IQL, CQL, MSG) and to investigate other design choices used in offline RL, e.g., learning rate schedules, dropout (like in IQL), wider networks, or selection between stochastic and deterministic policies.

# References

Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, and Marc G Bellemare. Deep reinforcement learning at the edge of the statistical precipice. Advances in Neural Information Processing Systems, 2021.

Dmitriy Akimov, Vladislav Kurenkov, Alexander Nikulin, Denis Tarasov, and Sergey Kolesnikov. Let offline rl flow: Training conservative agents in the latent space of normalizing flows. arXiv preprint arXiv:2211.11096, 2022.   
Gaon An, Seungyong Moon, Jang-Hyun Kim, and Hyun Oh Song. Uncertainty-based offline reinforcement learning with diversified q-ensemble. Advances in neural information processing systems, 34:7436–7447, 2021.   
Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.   
Philip J Ball, Laura Smith, Ilya Kostrikov, and Sergey Levine. Efficient online reinforcement learning with offline data. arXiv preprint arXiv:2302.02948, 2023.   
Alex Beeson and Giovanni Montana. Improving td3-bc: Relaxed policy constraint for offline learning and stable online fine-tuning. arXiv preprint arXiv:2211.11802, 2022.   
Aditya Bhatt, Max Argus, Artemij Amiranashvili, and Thomas Brox. Crossnorm: Normalization for off-policy td reinforcement learning. arXiv preprint arXiv:1902.05605, 2019.   
Xi Chen, Ali Ghadirzadeh, Tianhe Yu, Yuan Gao, Jianhao Wang, Wenzhe Li, Bin Liang, Chelsea Finn, and Chongjie Zhang. Latent-variable advantage-weighted policy optimization for offline rl. arXiv preprint arXiv:2203.08949, 2022a.   
Xi Chen, Ali Ghadirzadeh, Tianhe Yu, Jianhao Wang, Alex Yuan Gao, Wenzhe Li, Liang Bin, Chelsea Finn, and Chongjie Zhang. Lapo: Latent-variable advantage-weighted policy optimization for offline reinforcement learning. Advances in Neural Information Processing Systems, 35:36902–36913, 2022b.   
Logan Engstrom, Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Firdaus Janoos, Larry Rudolph, and Aleksander Madry. Implementation matters in deep policy gradients: A case study on ppo and trpo. arXiv preprint arXiv:2005.12729, 2020.   
Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine. D4rl: Datasets for deep data-driven reinforcement learning. arXiv preprint arXiv:2004.07219, 2020.   
Scott Fujimoto and Shixiang Shane Gu. A minimalist approach to offline reinforcement learning. Advances in neural information processing systems, 34:20132–20145, 2021.   
Scott Fujimoto, Herke Hoof, and David Meger. Addressing function approximation error in actor-critic methods. In International conference on machine learning, pp. 1587–1596. PMLR, 2018.   
Seyed Kamyar Seyed Ghasemipour, Shixiang Shane Gu, and Ofir Nachum. Why so pessimistic? estimating uncertainties for offline rl through ensembles, and why their independence matters. arXiv preprint arXiv:2205.13703, 2022.   
Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pp. 1861–1870. PMLR, 2018.   
Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, and David Meger. Deep reinforcement learning that matters. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.   
Takuya Hiraoka, Takahisa Imagawa, Taisei Hashimoto, Takashi Onishi, and Yoshimasa Tsuruoka. Dropout q-functions for doubly efficient reinforcement learning. arXiv preprint arXiv:2110.02034, 2021.   
Hao Hu, Yiqin Yang, Qianchuan Zhao, and Chongjie Zhang. On the role of discount factor in offline reinforcement learning. In International Conference on Machine Learning, pp. 9072–9098. PMLR, 2022.   
Nan Jiang, Alex Kulesza, Satinder Singh, and Richard Lewis. The dependence of effective planning horizon on model accuracy. In Proceedings of the 2015 International Conference on Autonomous Agents and Multiagent Systems, pp. 1181–1189, 2015.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.   
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.   
Ilya Kostrikov, Ashvin Nair, and Sergey Levine. Offline reinforcement learning with implicit q-learning. arXiv preprint arXiv:2110.06169, 2021.   
Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. Conservative q-learning for offline reinforcement learning. Advances in Neural Information Processing Systems, 33:1179–1191, 2020.   
Aviral Kumar, Rishabh Agarwal, Xinyang Geng, George Tucker, and Sergey Levine. Offline q-learning on diverse multi-task data both scales and generalizes. arXiv preprint arXiv:2211.15144, 2022.   
Vladislav Kurenkov and Sergey Kolesnikov. Showing your offline reinforcement learning work: Online evaluation budget matters. In International Conference on Machine Learning, pp. 11729–11752. PMLR, 2022.   
Kuang-Huei Lee, Ofir Nachum, Mengjiao Yang, Lisa Lee, Daniel Freeman, Winnie Xu, Sergio Guadarrama, Ian Fischer, Eric Jang, Henryk Michalewski, et al. Multi-game decision transformers. arXiv preprint arXiv:2205.15241, 2022.   
Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643, 2020.   
Cong Lu, Philip J Ball, Tim GJ Rudner, Jack Parker-Holder, Michael A Osborne, and Yee Whye Teh. Challenges and opportunities in offline reinforcement learning from visual observations. arXiv preprint arXiv:2206.04779, 2022.   
Mitsuhiko Nakamoto, Yuexiang Zhai, Anikait Singh, Max Sobol Mark, Yi Ma, Chelsea Finn, Aviral Kumar, and Sergey Levine. Cal-ql: Calibrated offline rl pre-training for efficient online fine-tuning. arXiv preprint arXiv:2303.05479, 2023.   
Oren Neumann and Claudius Gros. Scaling laws for a multi-agent reinforcement learning model. arXiv preprint arXiv:2210.00849, 2022.   
Alexander Nikulin, Vladislav Kurenkov, Denis Tarasov, Dmitry Akimov, and Sergey Kolesnikov. Q-ensemble for offline rl: Don't scale the ensemble, scale the batch size. arXiv preprint arXiv:2211.11092, 2022.   
Alexander Nikulin, Vladislav Kurenkov, Denis Tarasov, and Sergey Kolesnikov. Anti-exploration by random network distillation. arXiv preprint arXiv:2301.13616, 2023.   
Rafael Figueiredo Prudencio, Marcos ROA Maximo, and Esther Luna Colombini. A survey on offline reinforcement learning: Taxonomy, review, and open problems. arXiv preprint arXiv:2203.01387, 2022.   
Shideh Rezaeifar, Robert Dadashi, Nino Vieillard, Léonard Hussenot, Olivier Bachem, Olivier Pietquin, and Matthieu Geist. Offline reinforcement learning as anti-exploration. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pp. 8106–8114, 2022.   
Samarth Sinha, Homanga Bharadhwaj, Aravind Srinivas, and Animesh Garg. D2rl: Deep dense architectures in reinforcement learning. arXiv preprint arXiv:2010.09163, 2020.   
Laura Smith, Ilya Kostrikov, and Sergey Levine. A walk in the park: Learning to walk in 20 minutes with model-free reinforcement learning. arXiv preprint arXiv:2208.07860, 2022.   
Denis Tarasov, Alexander Nikulin, Dmitry Akimov, Vladislav Kurenkov, and Sergey Kolesnikov. Corl: Research-oriented deep offline reinforcement learning library. arXiv preprint arXiv:2210.07105, 2022.

Jialong Wu, Haixu Wu, Zihan Qiu, Jianmin Wang, and Mingsheng Long. Supported policy optimization for offline reinforcement learning. arXiv preprint arXiv:2202.06239, 2022.   
Yifan Wu, George Tucker, and Ofir Nachum. Behavior regularized offline reinforcement learning. arXiv preprint arXiv:1911.11361, 2019.   
Rui Yang, Chenjia Bai, Xiaoteng Ma, Zhaoran Wang, Chongjie Zhang, and Lei Han. Rorl: Robust offline reinforcement learning via conservative smoothing. arXiv preprint arXiv:2206.02829, 2022.   
Yang You, Igor Gitman, and Boris Ginsburg. Large batch training of convolutional networks. arXiv preprint arXiv:1708.03888, 2017.   
Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, and Cho-Jui Hsieh. Large batch optimization for deep learning: Training bert in 76 minutes. arXiv preprint arXiv:1904.00962, 2019.   
Zifeng Zhuang, Kun LEI, Jinxin Liu, Donglin Wang, and Yilang Guo. Behavior proximal policy optimization. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=3c13LptpIph.

# A Experimental Details

In order to generate the results presented in Table 2 Table 3 and Table 4, we conducted a hyperparameter search and selected the best results from the final evaluations for each dataset. Our algorithm was implemented using JAX for the D4RL benchmark. For V-D4RL, we implement our approach using PyTorch adopting the TD3+BC implementation from Clean Offline RL (Tarasov et al., 2022). The experiments were conducted on V100 and A100 GPUs.

Gym-MuJoCo and Adroit tasks. Our study utilized the latest version of the datasets – v2 for Gym-MuJoCo and v1 for Adroit. The agents were trained for one million steps and evaluated over ten episodes.

For ReBRAC, we fine-tuned the $\beta_{1}$ parameter for the actor, which was selected from 0.001, 0.01, 0.05, 0.1. Similarly, the $\beta_{2}$ parameter for the critic was selected from a range of 0, 0.001, 0.01, 0.1, 0.5. The selected best parameters for each dataset are reported in Table 11.

For TD3+BC here and in the AntMaze domain, we use the same grid used in ReBRAC for actor regularization parameter $\alpha$ and add the default value of 0.4.

For IQL here and in the AntMaze domain, we selected $\beta$ value from a range of 0.5, 1, 3, 6, 10 and IQL $\tau$ value from a range of 0.5, 0.7, 0.9, 0.95. We used the implementation from Clean Offline RL (Tarasov et al., 2022) and kept other parameters unchanged.

For SAC-RND in Adroit domain we tune $\beta_{1}$ (actor parameter) in the range of 0.5, 1.0, 2.5, 5.0, 10.0 and $\beta_{2}$ (critic parameter) in the range of 0.01, 0.1, 1.0, 5.0, 10.0.

AntMaze tasks. In our work, we utilized v2 of the datasets. It's worth noting that previous studies have reported results using v0 datasets, which were found to contain numerous issues $^{2}$ . Each agent was trained for 1 million steps and evaluated over 100 episodes. Following Chen et al. (2022a), we modified the reward function by multiplying it by 100.

For ReBRAC, the $\beta_{1}$ (actor) and $\beta_{2}$ (critic) hyperparameters were carefully selected from the respective ranges of 0.0005, 0.001, 0.002, 0.003 and 0, 0.0001, 0.0005, 0.001. In addition, the actor and critic learning rates were optimized from 0.0001, 0.0002, 0.0003, 0.0005 and 0.0003, 0.0005, 0.001, respectively. The optimal hyperparameters for each dataset are presented in Table 11.

We also modified the $\gamma$ value for ReBRAC when addressing these tasks, driven by the following motivation. The length of the episodes in AntMaze can be as long as 1000 steps, while the reward is sparse and can only be obtained at the end of the episode. As a result, the discount for the reward with the default $\gamma$ can be as low as $0.99^{1000} = 4 \cdot 10^{-5}$ , which is extremely low for signal propagation, even when multiplying the reward by 100. By increasing $\gamma$ to 0.999, the minimum discount value becomes $0.999^{1000} = 0.36$ , which is more favorable for signal propagation.

V-D4RL. We used single-task datasets without distraction with a resolution of $84 \times 84$ pixels. For ReBRAC $\beta_{1}$ (actor) parameter was selected from the range of $\{0.03, 0.1, 0.3, 1.0\}$ and $\beta_{2}$ (critic) parameter from the range of $\{0.0, 0.001, 0.005, 0.01, 0.1\}$ .

Offline-to-offline. We used the same parameters for the offline-to-online setup with the only difference of setting $\beta_{2}$ to zero and lineary decaying $\beta_{1}$ to half of it's initial value during the online stage.

# B Hyperparameters

# B.1 ReBRAC

Table 10: ReBRAC's general hyperparameters. 

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>optimizer</td><td>Adam Kingma &amp; Ba (2014)</td></tr><tr><td>batch size</td><td>1024 on Gym-MuJoCo, 256 on other</td></tr><tr><td>learning rate (all networks)</td><td>1e-3 on Gym-MuJoCo, 3e-4 on Adroit and V-D4RL, 1e-4 on Antmaze</td></tr><tr><td>tau ( $\tau$ )</td><td>5e-3</td></tr><tr><td>hidden dim (all networks)</td><td>256</td></tr><tr><td>num hidden layers (all networks)</td><td>3</td></tr><tr><td>gamma ( $\gamma$ )</td><td>0.999 on AntMaze, 0.99 on other</td></tr><tr><td>nonlinearity</td><td>ReLU</td></tr></table>

Table 11: ReBRAC's best hyperparameters used in D4RL benchmark. 

<table><tr><td>Task Name</td><td> $\beta_1$  (actor)</td><td> $\beta_2$  (critic)</td></tr><tr><td>halfcheetah-random</td><td>0.001</td><td>0.1</td></tr><tr><td>halfcheetah-medium</td><td>0.001</td><td>0.01</td></tr><tr><td>halfcheetah-expert</td><td>0.01</td><td>0.01</td></tr><tr><td>halfcheetah-medium-expert</td><td>0.01</td><td>0.1</td></tr><tr><td>halfcheetah-medium-replay</td><td>0.01</td><td>0.001</td></tr><tr><td>halfcheetah-full-replay</td><td>0.001</td><td>0.1</td></tr><tr><td>hopper-random</td><td>0.001</td><td>0.01</td></tr><tr><td>hopper-medium</td><td>0.01</td><td>0.001</td></tr><tr><td>hopper-expert</td><td>0.1</td><td>0.001</td></tr><tr><td>hopper-medium-expert</td><td>0.1</td><td>0.01</td></tr><tr><td>hopper-medium-replay</td><td>0.05</td><td>0.5</td></tr><tr><td>hopper-full-replay</td><td>0.01</td><td>0.01</td></tr><tr><td>walker2d-random</td><td>0.01</td><td>0.0</td></tr><tr><td>walker2d-medium</td><td>0.05</td><td>0.1</td></tr><tr><td>walker2d-expert</td><td>0.01</td><td>0.5</td></tr><tr><td>walker2d-medium-expert</td><td>0.01</td><td>0.01</td></tr><tr><td>walker2d-medium-replay</td><td>0.05</td><td>0.01</td></tr><tr><td>walker2d-full-replay</td><td>0.01</td><td>0.01</td></tr><tr><td>antmaze-umaze</td><td>0.003</td><td>0.002</td></tr><tr><td>antmaze-umaze-diverse</td><td>0.003</td><td>0.001</td></tr><tr><td>antmaze-medium-play</td><td>0.001</td><td>0.0005</td></tr><tr><td>antmaze-medium-diverse</td><td>0.001</td><td>0.0</td></tr><tr><td>antmaze-large-play</td><td>0.002</td><td>0.001</td></tr><tr><td>antmaze-large-diverse</td><td>0.002</td><td>0.002</td></tr><tr><td>pen-human</td><td>0.1</td><td>0.5</td></tr><tr><td>pen-cloned</td><td>0.05</td><td>0.5</td></tr><tr><td>pen-expert</td><td>0.01</td><td>0.01</td></tr><tr><td>door-human</td><td>0.1</td><td>0.1</td></tr><tr><td>door-cloned</td><td>0.01</td><td>0.1</td></tr><tr><td>door-expert</td><td>0.05</td><td>0.01</td></tr><tr><td>hammer-human</td><td>0.01</td><td>0.5</td></tr><tr><td>hammer-cloned</td><td>0.1</td><td>0.5</td></tr><tr><td>hammer-expert</td><td>0.01</td><td>0.01</td></tr><tr><td>relocate-human</td><td>0.1</td><td>0.01</td></tr><tr><td>relocate-cloned</td><td>0.1</td><td>0.01</td></tr><tr><td>relocate-expert</td><td>0.05</td><td>0.01</td></tr></table>

Table 12: ReBRAC's best hyperparameters used in V-D4RL benchmark. 

<table><tr><td>Task Name</td><td> $\beta_1$  (actor)</td><td> $\beta_2$  (critic)</td></tr><tr><td>walker-walk-random</td><td>0.03</td><td>0.1</td></tr><tr><td>walker-walk-medium</td><td>0.03</td><td>0.005</td></tr><tr><td>walker-walk-expert</td><td>0.1</td><td>0.01</td></tr><tr><td>walker-walk-medium-expert</td><td>0.3</td><td>0.005</td></tr><tr><td>walker-walk-medium-replay</td><td>0.3</td><td>0.01</td></tr><tr><td>cheetah-run-random</td><td>0.1</td><td>0.01</td></tr><tr><td>cheetah-run-medium</td><td>0.1</td><td>0.1</td></tr><tr><td>cheetah-run-expert</td><td>0.01</td><td>0.01</td></tr><tr><td>cheetah-run-medium-expert</td><td>1.0</td><td>0.001</td></tr><tr><td>cheetah-run-medium-replay</td><td>0.03</td><td>0.1</td></tr><tr><td>humanoid-walk-random</td><td>1.0</td><td>0.01</td></tr><tr><td>humanoid-walk-medium</td><td>1.0</td><td>0.005</td></tr><tr><td>humanoid-walk-expert</td><td>1.0</td><td>0.1</td></tr><tr><td>humanoid-walk-medium-expert</td><td>1.0</td><td>0.005</td></tr><tr><td>humanoid-walk-medium-replay</td><td>1.0</td><td>0.001</td></tr></table>

# B.2 IQL

Table 13: IQL's best hyperparameters used in D4RL benchmark. 

<table><tr><td>Task Name</td><td> $\beta$ </td><td>IQL  $\tau$ </td></tr><tr><td>halfcheetah-random</td><td>3.0</td><td>0.95</td></tr><tr><td>halfcheetah-medium</td><td>3.0</td><td>0.95</td></tr><tr><td>halfcheetah-expert</td><td>6.0</td><td>0.9</td></tr><tr><td>halfcheetah-medium-expert</td><td>3.0</td><td>0.7</td></tr><tr><td>halfcheetah-medium-replay</td><td>3.0</td><td>0.95</td></tr><tr><td>halfcheetah-full-replay</td><td>1.0</td><td>0.7</td></tr><tr><td>hopper-random</td><td>1.0</td><td>0.95</td></tr><tr><td>hopper-medium</td><td>3.0</td><td>0.7</td></tr><tr><td>hopper-expert</td><td>3.0</td><td>0.5</td></tr><tr><td>hopper-medium-expert</td><td>6.0</td><td>0.7</td></tr><tr><td>hopper-medium-replay</td><td>6.0</td><td>0.7</td></tr><tr><td>hopper-full-replay</td><td>10.0</td><td>0.9</td></tr><tr><td>walker2d-random</td><td>0.5</td><td>0.9</td></tr><tr><td>walker2d-medium</td><td>6.0</td><td>0.5</td></tr><tr><td>walker2d-expert</td><td>6.0</td><td>0.9</td></tr><tr><td>walker2d-medium-expert</td><td>1.0</td><td>0.5</td></tr><tr><td>walker2d-medium-replay</td><td>0.5</td><td>0.7</td></tr><tr><td>walker2d-full-replay</td><td>1.0</td><td>0.7</td></tr><tr><td>antmaze-umaze</td><td>10.0</td><td>0.7</td></tr><tr><td>antmaze-umaze-diverse</td><td>10.0</td><td>0.95</td></tr><tr><td>antmaze-medium-play</td><td>6.0</td><td>0.9</td></tr><tr><td>antmaze-medium-diverse</td><td>6.0</td><td>0.9</td></tr><tr><td>antmaze-large-play</td><td>10.0</td><td>0.9</td></tr><tr><td>antmaze-large-diverse</td><td>6.0</td><td>0.9</td></tr><tr><td>pen-human</td><td>1.0</td><td>0.95</td></tr><tr><td>pen-cloned</td><td>10.0</td><td>0.9</td></tr><tr><td>pen-expert</td><td>10.0</td><td>0.8</td></tr><tr><td>door-human</td><td>0.5</td><td>0.9</td></tr><tr><td>door-cloned</td><td>6.0</td><td>0.7</td></tr><tr><td>door-expert</td><td>0.5</td><td>0.7</td></tr><tr><td>hammer-human</td><td>3.0</td><td>0.9</td></tr><tr><td>hammer-cloned</td><td>6.0</td><td>0.7</td></tr><tr><td>hammer-expert</td><td>0.5</td><td>0.95</td></tr><tr><td>relocate-human</td><td>1.0</td><td>0.95</td></tr><tr><td>relocate-cloned</td><td>6.0</td><td>0.9</td></tr><tr><td>relocate-expert</td><td>10.0</td><td>0.9</td></tr></table>

# B.3 TD3+BC

Table 14: TD3+BC's best hyperparameters used in D4RL benchmark. 

<table><tr><td>Task Name</td><td> $\alpha$ </td></tr><tr><td>halfcheetah-random</td><td>0.001</td></tr><tr><td>halfcheetah-medium</td><td>0.01</td></tr><tr><td>halfcheetah-expert</td><td>0.4</td></tr><tr><td>halfcheetah-medium-expert</td><td>0.1</td></tr><tr><td>halfcheetah-medium-replay</td><td>0.05</td></tr><tr><td>halfcheetah-full-replay</td><td>0.01</td></tr><tr><td>hopper-random</td><td>0.4</td></tr><tr><td>hopper-medium</td><td>0.05</td></tr><tr><td>hopper-expert</td><td>0.1</td></tr><tr><td>hopper-medium-expert</td><td>0.1</td></tr><tr><td>hopper-medium-replay</td><td>0.4</td></tr><tr><td>hopper-full-replay</td><td>0.01</td></tr><tr><td>walker2d-random</td><td>0.001</td></tr><tr><td>walker2d-medium</td><td>0.4</td></tr><tr><td>walker2d-expert</td><td>0.05</td></tr><tr><td>walker2d-medium-expert</td><td>0.1</td></tr><tr><td>walker2d-medium-replay</td><td>0.1</td></tr><tr><td>walker2d-full-replay</td><td>0.1</td></tr><tr><td>antmaze-umaze</td><td>0.4</td></tr><tr><td>antmaze-umaze-diverse</td><td>0.4</td></tr><tr><td>antmaze-medium-play</td><td>0.003</td></tr><tr><td>antmaze-medium-diverse</td><td>0.003</td></tr><tr><td>antmaze-large-play</td><td>0.003</td></tr><tr><td>antmaze-large-diverse</td><td>0.003</td></tr><tr><td>pen-human</td><td>0.1</td></tr><tr><td>pen-cloned</td><td>0.4</td></tr><tr><td>pen-expert</td><td>0.4</td></tr><tr><td>door-human</td><td>0.1</td></tr><tr><td>door-cloned</td><td>0.4</td></tr><tr><td>door-expert</td><td>0.1</td></tr><tr><td>hammer-human</td><td>0.4</td></tr><tr><td>hammer-cloned</td><td>0.4</td></tr><tr><td>hammer-expert</td><td>0.4</td></tr><tr><td>relocate-human</td><td>0.1</td></tr><tr><td>relocate-cloned</td><td>0.1</td></tr><tr><td>relocate-expert</td><td>0.4</td></tr></table>

# B.4 SAC-RND

Table 15: SAC-RND's best hyperparameters used in D4RL Adroit tasks. 

<table><tr><td>Task Name</td><td> $\beta_1$  (actor)</td><td> $\beta_2$  (critic)</td></tr><tr><td>pen-human</td><td>1.0</td><td>10.0</td></tr><tr><td>pen-cloned</td><td>2.5</td><td>0.01</td></tr><tr><td>pen-expert</td><td>10.0</td><td>5.0</td></tr><tr><td>door-human</td><td>5.0</td><td>0.01</td></tr><tr><td>door-cloned</td><td>5.0</td><td>1.0</td></tr><tr><td>door-expert</td><td>10.0</td><td>10.0</td></tr><tr><td>hammer-human</td><td>10.0</td><td>0.01</td></tr><tr><td>hammer-cloned</td><td>1.0</td><td>1.0</td></tr><tr><td>hammer-expert</td><td>2.5</td><td>10.0</td></tr><tr><td>relocate-human</td><td>5.0</td><td>0.01</td></tr><tr><td>relocate-cloned</td><td>5.0</td><td>1.0</td></tr><tr><td>relocate-expert</td><td>10.0</td><td>10.0</td></tr></table>

# C Comparison to Ensemble-based Methods

Comparison of ReBRAC with the ensemble-based methods is presented in Table 16, Table 17, and Table 18. We add the following ensemble-based methods: RORL for each domain (Yang et al., 2022), SAC-N/EDAC (An et al., 2021) for the Gym-MuJoCo and Adroit tasks $^{3}$ and MSG (Ghasemipour et al., 2022) for AntMaze tasks $^{4}$ . The mean-wise best results among algorithms are highlighted with bold, and the second-best performance is underlined. Our approach, ReBRAC, shows competitive results on the Gym-MuJoCo datasets. On AntMaze tasks, ReBRAC achieves state-of-the-art results among ensemble-free algorithms and a good score compared to ensemble-based algorithms. And on Adroit tasks, our approach outperforms both families of algorithms.

Table 16: ReBRAC evaluation on the Gym domain. We report the final normalized score averaged over 10 unseen training seeds on v2 datasets. CQL, SAC-N and EDAC scores are taken from An et al. (2021). RORL scores are taken from Yang et al. (2022). 

<table><tr><td colspan="5">Ensemble-free</td><td colspan="4">Ensemble-based</td></tr><tr><td>Task Name</td><td>TD3+BC</td><td>IQL</td><td>CQL</td><td>SAC-RND</td><td>SAC-N</td><td>EDAC</td><td>RORL</td><td>ReBRAC, our</td></tr><tr><td>halfcheetah-random</td><td> $\underline{30.9} \pm 0.4$ </td><td> $19.5 \pm 0.8$ </td><td> $\underline{31.1} \pm 3.5$ </td><td> $27.6 \pm 2.1$ </td><td> $28.0 \pm 0.9$ </td><td> $28.4 \pm 1.0$ </td><td> $28.5 \pm 0.8$ </td><td> $29.5 \pm 1.5$ </td></tr><tr><td>halfcheetah-medium</td><td> $\underline{54.7} \pm 0.9$ </td><td> $50.0 \pm 0.2$ </td><td> $46.9 \pm 0.4$ </td><td> $66.4 \pm 1.4$ </td><td> $\underline{67.5} \pm 1.2$ </td><td> $65.9 \pm 0.6$ </td><td> $\underline{66.8} \pm 0.7$ </td><td> $65.6 \pm 1.0$ </td></tr><tr><td>halfcheetah-expert</td><td> $93.4 \pm 0.4$ </td><td> $95.5 \pm 2.1$ </td><td> $97.3 \pm 1.1$ </td><td> $102.6 \pm 4.2$ </td><td> $105.2 \pm 2.6$ </td><td> $\underline{106.8} \pm 3.4$ </td><td> $105.2 \pm 0.7$ </td><td> $\underline{105.9} \pm 1.7$ </td></tr><tr><td>halfcheetah-medium-expert</td><td> $89.1 \pm 5.6$ </td><td> $92.7 \pm 2.8$ </td><td> $95.0 \pm 1.4$ </td><td> $\underline{108.1} \pm 1.5$ </td><td> $107.1 \pm 2.0$ </td><td> $106.3 \pm 1.9$ </td><td> $\underline{107.8} \pm 1.1$ </td><td> $101.1 \pm 5.2$ </td></tr><tr><td>halfcheetah-medium-replay</td><td> $45.0 \pm 1.1$ </td><td> $42.1 \pm 3.6$ </td><td> $45.3 \pm 0.3$ </td><td> $51.2 \pm 3.2$ </td><td> $\underline{63.9} \pm 0.8$ </td><td> $61.3 \pm 1.9$ </td><td> $\underline{61.9} \pm 1.5$ </td><td> $51.0 \pm 0.8$ </td></tr><tr><td>halfcheetah-full-replay</td><td> $75.0 \pm 2.5$ </td><td> $75.0 \pm 0.7$ </td><td> $76.9 \pm 0.9$ </td><td> $81.2 \pm 1.3$ </td><td> $\underline{84.5} \pm 1.2$ </td><td> $\underline{84.6} \pm 0.9$ </td><td>-</td><td> $82.1 \pm 1.1$ </td></tr><tr><td>hopper-random</td><td> $8.5 \pm 0.6$ </td><td> $10.1 \pm 5.9$ </td><td> $5.3 \pm 0.6$ </td><td> $19.6 \pm 12.4$ </td><td> $\underline{31.3} \pm 0.0$ </td><td> $25.3 \pm 10.4$ </td><td> $\underline{31.4} \pm 0.1$ </td><td> $8.1 \pm 2.4$ </td></tr><tr><td>hopper-medium</td><td> $60.9 \pm 7.6$ </td><td> $65.2 \pm 4.2$ </td><td> $61.9 \pm 6.4$ </td><td> $91.1 \pm 10.1$ </td><td> $100.3 \pm 0.3$ </td><td> $\underline{101.6} \pm 0.6$ </td><td> $\underline{104.8} \pm 0.1$ </td><td> $102.0 \pm 1.0$ </td></tr><tr><td>hopper-expert</td><td> $109.6 \pm 3.7$ </td><td> $108.8 \pm 3.1$ </td><td> $106.5 \pm 9.1$ </td><td> $109.8 \pm 0.5$ </td><td> $\underline{110.3} \pm 0.3$ </td><td> $110.1 \pm 0.1$ </td><td> $\underline{112.8} \pm 0.2$ </td><td> $100.1 \pm 8.3$ </td></tr><tr><td>hopper-medium-expert</td><td> $87.8 \pm 10.5$ </td><td> $85.5 \pm 29.7$ </td><td> $96.9 \pm 15.1$ </td><td> $109.8 \pm 0.6$ </td><td> $110.1 \pm 0.3$ </td><td> $\underline{110.7} \pm 0.1$ </td><td> $\underline{112.7} \pm 0.2$ </td><td> $107.0 \pm 6.4$ </td></tr><tr><td>hopper-medium-replay</td><td> $55.1 \pm 31.7$ </td><td> $89.6 \pm 13.2$ </td><td> $86.3 \pm 7.3$ </td><td> $97.2 \pm 9.0$ </td><td> $\underline{101.8} \pm 0.5$ </td><td> $101.0 \pm 0.5$ </td><td> $\underline{102.8} \pm 0.5$ </td><td> $98.1 \pm 5.3$ </td></tr><tr><td>hopper-full-replay</td><td> $97.9 \pm 17.5$ </td><td> $104.4 \pm 10.8$ </td><td> $101.9 \pm 0.6$ </td><td> $\underline{107.4} \pm 0.8$ </td><td> $\underline{102.9} \pm 0.3$ </td><td> $105.4 \pm 0.7$ </td><td>-</td><td> $\underline{107.1} \pm 0.4$ </td></tr><tr><td>walker2d-random</td><td> $2.0 \pm 3.6$ </td><td> $11.3 \pm 7.0$ </td><td> $5.1 \pm 1.7$ </td><td> $18.7 \pm 6.9$ </td><td> $\underline{21.7} \pm 0.0$ </td><td> $16.6 \pm 7.0$ </td><td> $\underline{21.4} \pm 0.2$ </td><td> $18.1 \pm 4.5$ </td></tr><tr><td>walker2d-medium</td><td> $77.7 \pm 2.9$ </td><td> $80.7 \pm 3.4$ </td><td> $79.5 \pm 3.2$ </td><td> $\underline{92.7} \pm 1.2$ </td><td> $87.9 \pm 0.2$ </td><td> $92.5 \pm 0.8$ </td><td> $\underline{102.4} \pm 1.4$ </td><td> $82.5 \pm 3.6$ </td></tr><tr><td>walker2d-expert</td><td> $110.0 \pm 0.6$ </td><td> $96.9 \pm 32.3$ </td><td> $109.3 \pm 0.1$ </td><td> $104.5 \pm 22.8$ </td><td> $107.4 \pm 2.4$ </td><td> $\underline{115.1} \pm 1.9$ </td><td> $\underline{115.4} \pm 0.5$ </td><td> $112.3 \pm 0.2$ </td></tr><tr><td>walker2d-medium-expert</td><td> $110.4 \pm 0.6$ </td><td> $112.1 \pm 0.5$ </td><td> $109.1 \pm 0.2$ </td><td> $104.6 \pm 11.2$ </td><td> $\underline{116.7} \pm 0.4$ </td><td> $114.7 \pm 0.9$ </td><td> $\underline{121.2} \pm 1.5$ </td><td> $111.6 \pm 0.3$ </td></tr><tr><td>walker2d-medium-replay</td><td> $68.0 \pm 19.2$ </td><td> $75.4 \pm 9.3$ </td><td> $76.8 \pm 10.0$ </td><td> $\underline{89.4} \pm 3.8$ </td><td> $78.7 \pm 0.7$ </td><td> $87.1 \pm 2.4$ </td><td> $\underline{90.4} \pm 0.5$ </td><td> $77.3 \pm 7.9$ </td></tr><tr><td>walker2d-full-replay</td><td> $90.3 \pm 5.4$ </td><td> $97.5 \pm 1.4$ </td><td> $94.2 \pm 1.9$ </td><td> $\underline{105.3} \pm 3.2$ </td><td> $94.6 \pm 0.5$ </td><td> $99.8 \pm 0.7$ </td><td>-</td><td> $\underline{102.2} \pm 1.7$ </td></tr><tr><td>Average w/o full-replay</td><td>66.8</td><td>70.1</td><td>70.1</td><td>79.5</td><td>82.4</td><td>82.9</td><td>85.7</td><td>78.0</td></tr><tr><td>Average</td><td>70.3</td><td>72.9</td><td>73.6</td><td>82.6</td><td>84.4</td><td>85.2</td><td>-</td><td>81.2</td></tr></table>

Table 17: ReBRAC evaluation on AntMaze domain. We report the final normalized score averaged over 10 unseen training seeds on v2 datasets. CQL scores are taken from Ghasemipour et al. (2022). RORL scores are taken from Yang et al. (2022). 

<table><tr><td colspan="5">Ensemble-free</td><td colspan="3">Ensemble-based</td></tr><tr><td>Task Name</td><td>TD3+BC</td><td>IQL</td><td>CQL</td><td>SAC-RND</td><td>RORL</td><td>MSG</td><td>ReBRAC, our</td></tr><tr><td>antmaze-umaze</td><td> $66.3 \pm 6.2$ </td><td> $83.3 \pm 4.5$ </td><td>74.0</td><td> $97.0 \pm 1.5$ </td><td> $97.7 \pm 1.9$ </td><td> $97.9 \pm 1.3$ </td><td> $97.8 \pm 1.0$ </td></tr><tr><td>antmaze-umaze-diverse</td><td> $53.8 \pm 8.5$ </td><td> $70.6 \pm 3.7$ </td><td>84.0</td><td> $66.0 \pm 25.0$ </td><td> $90.7 \pm 2.9$ </td><td> $79.3 \pm 3.0$ </td><td> $88.3 \pm 13.0$ </td></tr><tr><td>antmaze-medium-play</td><td> $26.5 \pm 18.4$ </td><td> $64.6 \pm 4.9$ </td><td>61.2</td><td> $38.5 \pm 29.4$ </td><td> $76.3 \pm 2.5$ </td><td> $85.9 \pm 3.9$ </td><td> $84.0 \pm 4.2$ </td></tr><tr><td>antmaze-medium-diverse</td><td> $25.9 \pm 15.3$ </td><td> $61.7 \pm 6.1$ </td><td>53.7</td><td> $74.7 \pm 10.7$ </td><td> $69.3 \pm 3.3$ </td><td> $84.6 \pm 5.2$ </td><td> $76.3 \pm 13.5$ </td></tr><tr><td>antmaze-large-play</td><td> $0.0 \pm 0.0$ </td><td> $42.5 \pm 6.5$ </td><td>15.8</td><td> $43.9 \pm 29.2$ </td><td> $16.3 \pm 11.1$ </td><td> $64.3 \pm 12.7$ </td><td> $60.4 \pm 26.1$ </td></tr><tr><td>antmaze-large-diverse</td><td> $0.0 \pm 0.0$ </td><td> $27.6 \pm 7.8$ </td><td>14.9</td><td> $45.7 \pm 28.5$ </td><td> $41.0 \pm 10.7$ </td><td> $71.3 \pm 5.3$ </td><td> $54.4 \pm 25.1$ </td></tr><tr><td>Average</td><td>28.7</td><td>58.3</td><td>50.6</td><td>60.9</td><td>65.2</td><td>80.5</td><td>76.8</td></tr></table>

Table 18: ReBRAC evaluation on Adroit domain. We report the final normalized score averaged over 10 unseen training seeds on v1 datasets. BC, CQL, EDAC and RORL scores are taken from Yang et al. (2022). 

<table><tr><td colspan="6">Ensemble-free</td><td colspan="3">Ensemble-based</td></tr><tr><td>Task Name</td><td>BC</td><td>TD3+BC</td><td>IQL</td><td>CQL</td><td>SAC-RND</td><td>RORL</td><td>EDAC</td><td>ReBRAC, our</td></tr><tr><td>pen-human</td><td>34.4</td><td>81.8 ± 14.9</td><td>81.5 ± 17.5</td><td>37.5</td><td>5.6 ± 5.8</td><td>33.7 ± 7.6</td><td>51.2 ± 8.6</td><td>103.5 ± 14.1</td></tr><tr><td>pen-cloned</td><td>56.9</td><td>61.4 ± 19.3</td><td>77.2 ± 17.7</td><td>39.2</td><td>2.5 ± 6.1</td><td>35.7 ± 35.7</td><td>68.2 ± 7.3</td><td>91.8 ± 21.7</td></tr><tr><td>pen-expert</td><td>85.1</td><td>146.0 ± 7.3</td><td>133.6 ± 16.0</td><td>107.0</td><td>45.4 ± 22.9</td><td>130.3 ± 4.2</td><td>122.8 ± 14.1</td><td>154.1 ± 5.4</td></tr><tr><td>door-human</td><td>0.5</td><td>-0.1 ± 0.0</td><td>3.1 ± 2.0</td><td>9.9</td><td>0.0 ± 0.0</td><td>3.7 ± 0.7</td><td>10.7 ± 6.8</td><td>0.0 ± 0.1</td></tr><tr><td>door-cloned</td><td>-0.1</td><td>0.1 ± 0.6</td><td>0.8 ± 1.0</td><td>0.4</td><td>0.2 ± 0.8</td><td>-0.1 ± 0.1</td><td>9.6 ± 8.3</td><td>1.1 ± 2.6</td></tr><tr><td>door-expert</td><td>34.9</td><td>84.6 ± 44.5</td><td>105.3 ± 2.8</td><td>101.5</td><td>73.6 ± 26.7</td><td>104.9 ± 0.9</td><td>-0.3 ± 0.1</td><td>104.6 ± 2.4</td></tr><tr><td>hammer-human</td><td>1.5</td><td>0.4 ± 0.4</td><td>2.5 ± 1.9</td><td>4.4</td><td>-0.1 ± 0.1</td><td>2.3 ± 2.3</td><td>0.8 ± 0.4</td><td>0.2 ± 0.2</td></tr><tr><td>hammer-cloned</td><td>0.8</td><td>0.8 ± 0.7</td><td>1.1 ± 0.5</td><td>2.1</td><td>0.1 ± 0.4</td><td>1.7 ± 1.7</td><td>0.3 ± 0.0</td><td>6.7 ± 3.7</td></tr><tr><td>hammer-expert</td><td>125.6</td><td>117.0 ± 30.9</td><td>129.6 ± 0.5</td><td>86.7</td><td>24.8 ± 39.4</td><td>132.2 ± 0.7</td><td>0.2 ± 0.0</td><td>133.8 ± 0.7</td></tr><tr><td>relocate-human</td><td>0.0</td><td>-0.2 ± 0.0</td><td>0.1 ± 0.1</td><td>0.2</td><td>0.0 ± 0.0</td><td>0.0 ± 0.0</td><td>0.1 ± 0.1</td><td>0.0 ± 0.0</td></tr><tr><td>relocate-cloned</td><td>-0.1</td><td>-0.1 ± 0.1</td><td>0.2 ± 0.4</td><td>-0.1</td><td>0.0 ± 0.0</td><td>0.0 ± 0.0</td><td>0.0 ± 0.0</td><td>0.9 ± 1.6</td></tr><tr><td>relocate-expert</td><td>101.3</td><td>107.3 ± 1.6</td><td>106.5 ± 2.5</td><td>95.0</td><td>3.4 ± 4.5</td><td>47.8 ± 13.5</td><td>-0.3 ± 0.0</td><td>106.6 ± 3.2</td></tr><tr><td>Average w/o expert</td><td>11.7</td><td>18.0</td><td>20.8</td><td>11.7</td><td>1.0</td><td>9.6</td><td>17.4</td><td>25.5</td></tr><tr><td>Average</td><td>36.7</td><td>49.9</td><td>53.4</td><td>40.3</td><td>12.9</td><td>41.0</td><td>21.9</td><td>58.6</td></tr></table>

# D Feature Normalization

Table 19: Average normalized score over the final evaluation and ten unseen training seeds on D4RL tasks for 2 types of normalization: LayerNorm (LN) and Feature Norm (FN). The symbol ± represents the standard deviation across the seeds. For both variatns we tune hyperparameters using the same grid. 

<table><tr><td>Task Name</td><td>ReBRAC + LN</td><td>ReBRAC + FN</td></tr><tr><td>halfcheetah-random</td><td>29.5 ± 1.5</td><td>31.4 ± 2.7</td></tr><tr><td>halfcheetah-medium</td><td>65.6 ± 1.0</td><td>66.1 ± 1.2</td></tr><tr><td>halfcheetah-expert</td><td>105.9 ± 1.7</td><td>104.1 ± 3.7</td></tr><tr><td>halfcheetah-medium-expert</td><td>101.1 ± 5.2</td><td>100.9 ± 4.7</td></tr><tr><td>halfcheetah-medium-replay</td><td>51.0 ± 0.8</td><td>54.7 ± 1.0</td></tr><tr><td>halfcheetah-full-replay</td><td>82.1 ± 1.1</td><td>81.5 ± 1.6</td></tr><tr><td>hopper-random</td><td>8.1 ± 2.4</td><td>8.2 ± 2.2</td></tr><tr><td>hopper-medium</td><td>102.0 ± 1.0</td><td>102.4 ± 0.2</td></tr><tr><td>hopper-expert</td><td>100.1 ± 8.3</td><td>99.7 ± 11.7</td></tr><tr><td>hopper-medium-expert</td><td>107.0 ± 6.4</td><td>107.7 ± 6.4</td></tr><tr><td>hopper-medium-replay</td><td>98.1 ± 5.3</td><td>91.0 ± 15.3</td></tr><tr><td>hopper-full-replay</td><td>107.1 ± 0.4</td><td>106.7 ± 0.5</td></tr><tr><td>walker2d-random</td><td>18.4 ± 4.5</td><td>0.0 ± 0.7</td></tr><tr><td>walker2d-medium</td><td>82.5 ± 3.6</td><td>81.8 ± 4.7</td></tr><tr><td>walker2d-expert</td><td>112.3 ± 0.2</td><td>110.7 ± 3.2</td></tr><tr><td>walker2d-medium-expert</td><td>111.6 ± 0.3</td><td>100.6 ± 34.3</td></tr><tr><td>walker2d-medium-replay</td><td>77.3 ± 7.9</td><td>80.2 ± 8.1</td></tr><tr><td>walker2d-full-replay</td><td>102.2 ± 1.7</td><td>101.2 ± 2.7</td></tr><tr><td>Gym-MuJoCo average</td><td>81.2</td><td>79.3</td></tr><tr><td>antmaze-umaze</td><td>97.8 ± 1.0</td><td>96.8 ± 1.6</td></tr><tr><td>antmaze-umaze-diverse</td><td>88.3 ± 13.0</td><td>88.5 ± 7.5</td></tr><tr><td>antmaze-medium-play</td><td>84.0 ± 4.2</td><td>84.1 ± 10.1</td></tr><tr><td>antmaze-medium-diverse</td><td>76.3 ± 13.5</td><td>75.6 ± 13.7</td></tr><tr><td>antmaze-large-play</td><td>60.4 ± 26.1</td><td>55.0 ± 30.0</td></tr><tr><td>antmaze-large-diverse</td><td>54.4 ± 25.1</td><td>66.4 ± 7.4</td></tr><tr><td>AntMaze average</td><td>76.8</td><td>77.7</td></tr><tr><td>pen-human</td><td>103.5 ± 14.1</td><td>107.0 ± 13.8</td></tr><tr><td>pen-cloned</td><td>91.8 ± 21.7</td><td>84.9 ± 20.1</td></tr><tr><td>pen-expert</td><td>154.1 ± 5.4</td><td>151.6 ± 4.7</td></tr><tr><td>door-human</td><td>0.0 ± 0.1</td><td>0.0 ± 0.0</td></tr><tr><td>door-cloned</td><td>1.1 ± 2.6</td><td>0.1 ± 0.1</td></tr><tr><td>door-expert</td><td>104.6 ± 2.4</td><td>105.1 ± 1.4</td></tr><tr><td>hammer-human</td><td>0.2 ± 0.2</td><td>0.2 ± 0.1</td></tr><tr><td>hammer-cloned</td><td>6.7 ± 3.7</td><td>10.1 ± 9.5</td></tr><tr><td>hammer-expert</td><td>133.8 ± 0.7</td><td>133.4 ± 1.5</td></tr><tr><td>relocate-human</td><td>0.0 ± 0.0</td><td>0.0 ± 0.0</td></tr><tr><td>relocate-cloned</td><td>0.9 ± 1.6</td><td>1.2 ± 1.9</td></tr><tr><td>relocate-expert</td><td>106.6 ± 3.2</td><td>108.7 ± 3.0</td></tr><tr><td>Adroit average w/o expert</td><td>25.5</td><td>25.4</td></tr><tr><td>Adroit average</td><td>58.6</td><td>58.5</td></tr></table>

# E Computational costs

Table 20: Computational costs for algorithms in Table 2. 

<table><tr><td>Algorithm</td><td>Number of runs</td><td>Approximate hours per run</td></tr><tr><td>TD3+BC, tuning</td><td>360</td><td>0.3</td></tr><tr><td>IQL, tuning</td><td>1440</td><td>1.8</td></tr><tr><td>ReBRAC, tuning</td><td>1440</td><td>0.4</td></tr><tr><td>TD3+BC, eval</td><td>180</td><td>0.2</td></tr><tr><td>IQL, eval</td><td>180</td><td>1.8</td></tr><tr><td>SAC-RND, eval</td><td>180</td><td>1.8</td></tr><tr><td>ReBRAC, eval</td><td>180</td><td>0.4</td></tr><tr><td>Sum</td><td>3960</td><td>4032.0</td></tr></table>

Table 21: Computational costs for algorithms in Table 3 and Table 17. 

<table><tr><td>Algorithm</td><td>Number of runs</td><td>Approximate hours per run</td></tr><tr><td>TD3+BC, tuning</td><td>96</td><td>0.5</td></tr><tr><td>IQL, tuning</td><td>480</td><td>2.1</td></tr><tr><td>ReBRAC, tuning</td><td>384</td><td>0.6</td></tr><tr><td>TD3+BC, eval</td><td>60</td><td>0.5</td></tr><tr><td>IQL, eval</td><td>60</td><td>2.0</td></tr><tr><td>SAC-RND, eval</td><td>60</td><td>2.9</td></tr><tr><td>MSG, eval</td><td>60</td><td>5.1</td></tr><tr><td>ReBRAC, eval</td><td>60</td><td>0.4</td></tr><tr><td>Sum</td><td>1260</td><td>1940.4</td></tr></table>

Table 22: Computational costs for algorithms in Table 4. 

<table><tr><td>Algorithm</td><td>Number of runs</td><td>Approximate hours per run</td></tr><tr><td>TD3+BC, tuning</td><td>240</td><td>0.3</td></tr><tr><td>IQL, tuning</td><td>960</td><td>1.8</td></tr><tr><td>SAC-RND, tuning</td><td>1200</td><td>1.1</td></tr><tr><td>ReBRAC, tuning</td><td>960</td><td>0.3</td></tr><tr><td>TD3+BC, eval</td><td>120</td><td>0.2</td></tr><tr><td>IQL, eval</td><td>120</td><td>1.9</td></tr><tr><td>SAC-RND, eval</td><td>120</td><td>1.1</td></tr><tr><td>ReBRAC, eval</td><td>120</td><td>0.3</td></tr><tr><td>Sum</td><td>3840</td><td>3828.0</td></tr></table>

Table 23: Computational costs for algorithms in Table 5. 

<table><tr><td>Algorithm</td><td>Number of runs</td><td>Approximate hours per run</td></tr><tr><td>ReBRAC, tuning</td><td>600</td><td>10.6</td></tr><tr><td>ReBRAC, eval</td><td>75</td><td>10.5</td></tr><tr><td>Sum</td><td>675</td><td>7147.5</td></tr></table>

Table 24: Computational costs for algorithms in Table 8 and Figure 2. 

<table><tr><td>Algorithm</td><td>Number of runs</td><td>Approximate hours per run</td></tr><tr><td>ReBRAC, ablations eval</td><td>1104</td><td>1.4</td></tr><tr><td>Sum</td><td>1104</td><td>1545.6</td></tr></table>

# F Expected Online Performance

Table 25: TD3+BC, IQL and ReBRAC Expected Online Performance under uniform policy selection on HalfCheetah tasks. 

<table><tr><td></td><td colspan="3">random</td><td colspan="3">medium</td><td colspan="3">expert</td><td colspan="3">medium-expert</td><td colspan="3">medium-replay</td><td colspan="3">full-replay</td></tr><tr><td>Policies</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td></tr><tr><td>1</td><td>14.6 ± 9.3</td><td>10.2 ± 6.8</td><td>17.6 ± 8.2</td><td>48.0 ± 5.8</td><td>48.0 ± 1.3</td><td>56.1 ± 6.3</td><td>59.5 ± 40.5</td><td>93.9 ± 4.2</td><td>90.7 ± 21.5</td><td>68.1 ± 31.4</td><td>87.7 ± 5.5</td><td>97.7 ± 6.8</td><td>34.7 ± 14.2</td><td>43.4 ± 1.3</td><td>47.7 ± 3.0</td><td>67.7 ± 12.5</td><td>73.1 ± 1.9</td><td>78.7 ± 3.3</td></tr><tr><td>2</td><td>19.8 ± 8.0</td><td>14.1 ± 5.9</td><td>22.2 ± 7.0</td><td>51.1 ± 5.4</td><td>48.8 ± 1.1</td><td>59.6 ± 5.8</td><td>80.4 ± 28.1</td><td>95.6 ± 1.5</td><td>100.8 ± 11.4</td><td>83.7 ± 18.1</td><td>90.8 ± 3.7</td><td>101.2 ± 3.8</td><td>41.5 ± 7.7</td><td>44.2 ± 0.8</td><td>49.4 ± 2.6</td><td>73.8 ± 7.0</td><td>74.1 ± 1.1</td><td>80.5 ± 2.6</td></tr><tr><td>3</td><td>22.5 ± 6.8</td><td>16.1 ± 4.6</td><td>24.6 ± 5.9</td><td>52.9 ± 5.0</td><td>49.1 ± 0.9</td><td>61.6 ± 4.9</td><td>88.2 ± 18.1</td><td>96.0 ± 0.8</td><td>103.6 ± 6.0</td><td>88.5 ± 10.1</td><td>92.0 ± 2.7</td><td>102.4 ± 2.4</td><td>43.6 ± 4.2</td><td>44.4 ± 0.6</td><td>50.3 ± 2.1</td><td>75.8 ± 4.2</td><td>74.5 ± 0.8</td><td>81.4 ± 2.0</td></tr><tr><td>4</td><td>24.2 ± 5.7</td><td>17.3 ± 3.9</td><td>26.9 ± 4.9</td><td>54.0 ± 4.7</td><td>49.4 ± 0.7</td><td>62.8 ± 4.2</td><td>91.3 ± 11.5</td><td>96.7 ± 0.9</td><td>104.7 ± 3.5</td><td>90.3 ± 5.9</td><td>92.7 ± 0.2</td><td>102.9 ± 1.9</td><td>44.5 ± 2.6</td><td>44.6 ± 0.4</td><td>50.8 ± 1.6</td><td>76.7 ± 3.1</td><td>74.7 ± 0.7</td><td>81.0 ± 1.6</td></tr><tr><td>5</td><td>25.3 ± 4.8</td><td>17.9 ± 2.8</td><td>27.0 ± 4.1</td><td>54.9 ± 4.3</td><td>49.5 ± 0.6</td><td>63.6 ± 3.5</td><td>92.7 ± 7.3</td><td>96.3 ± 0.4</td><td>105.2 ± 2.5</td><td>91.0 ± 3.6</td><td>93.2 ± 1.8</td><td>103.3 ± 4.6</td><td>44.9 ± 1.8</td><td>44.6 ± 0.4</td><td>51.1 ± 1.3</td><td>77.3 ± 2.5</td><td>74.9 ± 0.6</td><td>82.2 ± 1.2</td></tr><tr><td>6</td><td>18.4 ± 2.3</td><td>27.7 ± 3.4</td><td></td><td>-</td><td>49.6 ± 0.5</td><td>64.1 ± 3.0</td><td></td><td>96.3 ± 0.3</td><td>105.6 ± 2.1</td><td></td><td>93.5 ± 1.5</td><td>103.5 ± 1.5</td><td></td><td>44.7 ± 0.3</td><td>51.3 ± 1.1</td><td></td><td>75.0 ± 0.5</td><td>82.3 ± 1.0</td></tr><tr><td>7</td><td>-</td><td>18.7 ± 1.8</td><td>28.1 ± 2.9</td><td></td><td>-49.7 ± 0.5</td><td>64.5 ± 2.6</td><td></td><td>96.3 ± 0.3</td><td>105.9 ± 1.9</td><td></td><td>93.7 ± 1.3</td><td>103.7 ± 1.4</td><td></td><td>44.8 ± 0.3</td><td>51.4 ± 0.9</td><td></td><td>75.0 ± 0.5</td><td>82.5 ± 0.8</td></tr><tr><td>8</td><td>-</td><td>18.9 ± 1.5</td><td>28.5 ± 2.5</td><td></td><td>-49.7 ± 0.4</td><td>64.8 ± 2.3</td><td></td><td>96.4 ± 0.3</td><td>106.1 ± 1.7</td><td></td><td>93.8 ± 1.1</td><td>103.9 ± 1.4</td><td></td><td>44.8 ± 0.2</td><td>51.5 ± 0.7</td><td></td><td>75.1 ± 0.5</td><td>82.6 ± 0.7</td></tr><tr><td>9</td><td>-</td><td>19.0 ± 1.3</td><td>28.7 ± 2.1</td><td></td><td>-49.8 ± 0.4</td><td>65.0 ± 2.0</td><td></td><td>96.4 ± 0.2</td><td>106.3 ± 1.6</td><td></td><td>93.9 ± 1.0</td><td>104.0 ± 1.3</td><td></td><td>44.8 ± 0.2</td><td>51.6 ± 0.7</td><td></td><td>75.1 ± 0.4</td><td>82.6 ± 0.6</td></tr><tr><td>10</td><td>-</td><td>19.1 ± 1.1</td><td>28.9 ± 1.8</td><td></td><td>-49.8 ± 0.4</td><td>65.2 ± 1.7</td><td></td><td>96.5 ± 0.2</td><td>106.4 ± 1.5</td><td></td><td>94.0 ± 0.9</td><td>104.1 ± 1.3</td><td></td><td>44.8 ± 0.2</td><td>51.6 ± 0.6</td><td></td><td>75.2 ± 0.4</td><td>82.7 ± 0.6</td></tr><tr><td>11</td><td>-</td><td>19.2 ± 0.9</td><td>29.0 ± 1.6</td><td></td><td>-49.9 ± 0.4</td><td>65.3 ± 1.5</td><td></td><td>96.5 ± 0.2</td><td>106.6 ± 1.3</td><td></td><td>94.1 ± 0.8</td><td>104.2 ± 1.3</td><td></td><td>44.8 ± 0.2</td><td>51.7 ± 0.6</td><td></td><td>75.2 ± 0.4</td><td>82.7 ± 0.5</td></tr><tr><td>12</td><td>-</td><td>19.3 ± 0.8</td><td>29.2 ± 1.4</td><td></td><td>-49.9 ± 0.3</td><td>65.4 ± 1.3</td><td></td><td>96.5 ± 0.2</td><td>106.7 ± 1.2</td><td></td><td>94.2 ± 0.7</td><td>104.3 ± 1.3</td><td></td><td>44.9 ± 0.2</td><td>51.7 ± 0.5</td><td></td><td>75.2 ± 0.4</td><td>82.8 ± 0.4</td></tr><tr><td>13</td><td>-</td><td>19.4 ± 0.7</td><td>29.3 ± 1.2</td><td></td><td>-49.9 ± 0.3</td><td>65.5 ± 1.1</td><td></td><td>96.5 ± 0.2</td><td>106.8 ± 1.1</td><td></td><td>94.2 ± 0.7</td><td>104.3 ± 1.3</td><td></td><td>44.9 ± 0.2</td><td>51.7 ± 0.5</td><td></td><td>75.3 ± 0.3</td><td>82.8 ± 0.4</td></tr><tr><td>14</td><td>-</td><td>19.4 ± 0.6</td><td>29.3 ± 1.1</td><td></td><td>-49.9 ± 0.3</td><td>65.5 ± 1.0</td><td></td><td>96.5 ± 0.2</td><td>106.8 ± 1.1</td><td></td><td>94.3 ± 0.6</td><td>104.4 ± 1.2</td><td></td><td>44.9 ± 0.1</td><td>51.8 ± 0.5</td><td></td><td>75.3 ± 0.3</td><td>82.8 ± 0.4</td></tr><tr><td>15</td><td>-</td><td>19.5 ± 0.5</td><td>29.4 ± 1.0</td><td></td><td>-49.9 ± 0.3</td><td>65.6 ± 0.9</td><td></td><td>96.5 ± 0.2</td><td>106.9 ± 1.0</td><td></td><td>94.3 ± 0.6</td><td>104.5 ± 1.2</td><td></td><td>44.9 ± 0.1</td><td>51.8 ± 0.5</td><td></td><td>75.3 ± 0.3</td><td>82.8 ± 0.3</td></tr><tr><td>16</td><td>-</td><td>19.5 ± 0.3</td><td>29.4 ± 0.9</td><td></td><td>50.0 ± 0.3</td><td>65.6 ± 0.8</td><td></td><td>96.5 ± 0.1</td><td>107.0 ± 0.9</td><td></td><td>94.4 ± 0.5</td><td>104.6 ± 1.2</td><td></td><td>44.9 ± 0.1</td><td>51.8 ± 0.5</td><td></td><td>75.3 ± 0.3</td><td>82.9 ± 0.3</td></tr><tr><td>17</td><td>-</td><td>19.5 ± 0.4</td><td>29.5 ± 0.9</td><td></td><td>50.0 ± 0.3</td><td>65.7 ± 0.7</td><td></td><td>96.6 ± 0.1</td><td>107.0 ± 0.8</td><td></td><td>94.4 ± 0.5</td><td>104.6 ± 1.2</td><td></td><td>44.9 ± 0.1</td><td>51.9 ± 0.5</td><td></td><td>75.3 ± 0.3</td><td>82.9 ± 0.3</td></tr><tr><td>18</td><td>-</td><td>19.6 ± 0.4</td><td>29.5 ± 0.8</td><td></td><td>50.0 ± 0.3</td><td>65.7 ± 0.6</td><td></td><td>96.6 ± 0.1</td><td>107.1 ± 0.8</td><td></td><td>94.4 ± 0.5</td><td>104.7 ± 1.2</td><td></td><td>44.9 ± 0.1</td><td>51.9 ± 0.5</td><td></td><td>75.3 ± 0.3</td><td>82.9 ± 0.2</td></tr><tr><td>19</td><td>-</td><td>19.5 ± 0.4</td><td>29.6 ± 0.8</td><td></td><td>50.0 ± 0.3</td><td>65.7 ± 0.5</td><td></td><td>96.6 ± 0.1</td><td>107.1 ± 0.7</td><td></td><td>94.4 ± 0.5</td><td>104.7 ± 1.1</td><td></td><td>44.9 ± 0.1</td><td>51.9 ± 0.4</td><td></td><td>75.4 ± 0.3</td><td>82.9 ± 0.2</td></tr><tr><td>20</td><td>-</td><td>19.5 ± 0.3</td><td>29.6 ± 0.7</td><td></td><td>-</td><td>65.7 ± 0.5</td><td></td><td>-</td><td>107.1 ± 0.7</td><td></td><td>-</td><td>104.8 ± 1.1</td><td></td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr></table>

![](images/51f0ae24eb3c348fbaa24595c66cfe17a3c0bf5e2be4adb1aaf21ba95df6e8d2.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 15.0   | 12.0     | 10.0 |
| 2                         | 20.0   | 16.0     | 14.0 |
| 3                         | 25.0   | 20.0     | 17.0 |
| 4                         | 27.0   | 23.0     | 18.0 |
| 5                         | 28.0   | 25.0     | 19.0 |
| 6                         | 28.5   | 26.0     | 19.5 |
| 7                         | 29.0   | 27.0     | 20.0 |
| 8                         | 29.5   | 27.5     | 20.5 |
| 9                         | 29.8   | 28.0     | 21.0 |
| 10                        | 30.0   | 28.5     | 21.5 |
| 11                        | 30.2   | 29.0     | 22.0 |
| 12                        | 30.3   | 29.2     | 22.5 |
| 13                        | 30.4   | 29.3     | 23.0 |
| 14                        | 30.5   | 29.4     | 23.5 |
| 15                        | 30.6   | 29.5     | 24.0 |
| 16                        | 30.7   | 29.6     | 24.5 |
| 17                        | 30.8   | 29.7     | 25.0 |
| 18                        | 30.9   | 29.8     | 25.5 |
| 19                        | 31.0   | 29.9     | 26.0 |
| 20                        | 31.1   | 30.0     | 26.5 |
</details>

(a) halfcheetah-random EOP.

![](images/41b137fa867ff6ab7f2812b357ab50eff77e4d3f0a30a8e9d358d0e07f368df2.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 55     | 48       | 47  |
| 2                         | 60     | 50       | 48  |
| 3                         | 62     | 52       | 49  |
| 4                         | 63     | 53       | 49  |
| 5                         | 64     | 54       | 49  |
| 6                         | 64.5   | 54.5     | 49.5|
| 7                         | 65     | 55       | 49.5|
| 8                         | 65.5   | 55.5     | 49.5|
| 9                         | 65.5   | 55.5     | 49.5|
| 10                        | 65.5   | 55.5     | 49.5|
| 11                        | 65.5   | 55.5     | 49.5|
| 12                        | 65.5   | 55.5     | 49.5|
| 13                        | 65.5   | 55.5     | 49.5|
| 14                        | 65.5   | 55.5     | 49.5|
| 15                        | 65.5   | 55.5     | 49.5|
| 16                        | 65.5   | 55.5     | 49.5|
| 17                        | 65.5   | 55.5     | 49.5|
| 18                        | 65.5   | 55.5     | 49.5|
| 19                        | 65.5   | 55.5     | 49.5|
| 20                        | 65.5   | 55.5     | 49.5|
</details>

(b) halfcheetah-medium EOP.

![](images/34bb9adf8fd0674d1c8d51d3fef47d4da8fff387d65e559e0c7dc62806765b4a.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 90     | 60       | 90  |
| 2                         | 95     | 70       | 90  |
| 3                         | 98     | 80       | 90  |
| 4                         | 99     | 85       | 90  |
| 5                         | 100    | 90       | 90  |
| 6                         | 100    | 90       | 90  |
| 7                         | 100    | 90       | 90  |
| 8                         | 100    | 90       | 90  |
| 9                         | 100    | 90       | 90  |
| 10                        | 100    | 90       | 90  |
| 11                        | 100    | 90       | 90  |
| 12                        | 100    | 90       | 90  |
| 13                        | 100    | 90       | 90  |
| 14                        | 100    | 90       | 90  |
| 15                        | 100    | 90       | 90  |
| 16                        | 100    | 90       | 90  |
| 17                        | 100    | 90       | 90  |
| 18                        | 100    | 90       | 90  |
| 19                        | 100    | 90       | 90  |
| 20                        | 100    | 90       | 90  |
</details>

(c) halfcheetah-expert EOP.

![](images/dffbf1f2b55458a397145aa1cdacc8711ac97d218e5ed200a83f6e38560bda58.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 95     | 70       | 85  |
| 2                         | 96     | 80       | 88  |
| 3                         | 97     | 85       | 90  |
| 4                         | 97     | 88       | 91  |
| 5                         | 97     | 90       | 92  |
| 6                         | 97     | 91       | 92  |
| 7                         | 97     | 92       | 92  |
| 8                         | 97     | 92       | 92  |
| 9                         | 97     | 92       | 92  |
| 10                        | 97     | 92       | 92  |
| 11                        | 97     | 92       | 92  |
| 12                        | 97     | 92       | 92  |
| 13                        | 97     | 92       | 92  |
| 14                        | 97     | 92       | 92  |
| 15                        | 97     | 92       | 92  |
| 16                        | 97     | 92       | 92  |
| 17                        | 97     | 92       | 92  |
| 18                        | 97     | 92       | 92  |
| 19                        | 97     | 92       | 92  |
| 20                        | 97     | 92       | 92  |
</details>

(d) halfcheetah-medium-expert EOP.

![](images/f8aa99512650cf964cb94344fb4a8246840ab9a62af868a2be23b9950db79629.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 48     | 35       | 43  |
| 2                         | 49     | 40       | 43  |
| 3                         | 50     | 42       | 43  |
| 4                         | 50     | 43       | 43  |
| 5                         | 50     | 44       | 43  |
| 6                         | 50     | 44       | 43  |
| 7                         | 50     | 44       | 43  |
| 8                         | 50     | 44       | 43  |
| 9                         | 50     | 44       | 43  |
| 10                        | 50     | 44       | 43  |
| 11                        | 50     | 44       | 43  |
| 12                        | 50     | 44       | 43  |
| 13                        | 50     | 44       | 43  |
| 14                        | 50     | 44       | 43  |
| 15                        | 50     | 44       | 43  |
| 16                        | 50     | 44       | 43  |
| 17                        | 50     | 44       | 43  |
| 18                        | 50     | 44       | 43  |
| 19                        | 50     | 44       | 43  |
| 20                        | 50     | 44       | 43  |
</details>

(e) halfcheetah-medium-replay EOP.

![](images/222ae1f64136ea44e5653610278b415fec4b51bfac44e9952c790f29ed5ed377.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| -------------------------- | ------ | -------- | --- |
| 1                          | 78     | 68       | 72  |
| 2                          | 80     | 70       | 73  |
| 3                          | 81     | 72       | 74  |
| 4                          | 82     | 74       | 75  |
| 5                          | 83     | 75       | 75  |
| 6                          | 83     | 75       | 75  |
| 7                          | 83     | 75       | 75  |
| 8                          | 83     | 75       | 75  |
| 9                          | 83     | 75       | 75  |
| 10                         | 83     | 75       | 75  |
| 11                         | 83     | 75       | 75  |
| 12                         | 83     | 75       | 75  |
| 13                         | 83     | 75       | 75  |
| 14                         | 83     | 75       | 75  |
| 15                         | 83     | 75       | 75  |
| 16                         | 83     | 75       | 75  |
| 17                         | 83     | 75       | 75  |
| 18                         | 83     | 75       | 75  |
| 19                         | 83     | 75       | 75  |
| 20                         | 83     | 75       | 75  |
</details>

(f) halfcheetah-full-replay EOP.   
Figure 3: TD3+BC, IQL and ReBRAC visualised Expected Online Performance under uniform policy selection on HalfCheetah tasks.

Table 26: TD3+BC, IQL and ReBRAC Expected Online Performance under uniform policy selection on Hopper tasks. 

<table><tr><td></td><td colspan="3">random</td><td colspan="3">medium</td><td colspan="3">expert</td><td colspan="3">medium-expert</td><td colspan="3">medium-replay</td><td colspan="3">full-replay</td></tr><tr><td>Policies</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td></tr><tr><td>1</td><td>8.3 ± 4.5</td><td>7.5 ± 1.3</td><td>7.5 ± 0.9</td><td>39.8 ± 33.0</td><td>59.0 ± 4.9</td><td>69.5 ± 32.6</td><td>72.2 ± 47.1</td><td>96.6 ± 17.3</td><td>58.3 ± 40.5</td><td>55.0 ± 45.5</td><td>83.3 ± 28.4</td><td>58.7 ± 39.8</td><td>62.5 ± 14.9</td><td>63.8 ± 28.3</td><td>67.2 ± 28.4</td><td>68.7 ± 27.6</td><td>94.5 ± 20.8</td><td>96.7 ± 17.8</td></tr><tr><td>2</td><td>10.8 ± 4.1</td><td>8.1 ± 1.4</td><td>8.0 ± 0.5</td><td>57.7 ± 27.1</td><td>61.8 ± 32.8</td><td>69.1 ± 31.8</td><td>96.1 ± 10.8</td><td>105.8 ± 0.5</td><td>80.3 ± 30.8</td><td>79.3 ± 36.3</td><td>98.2 ± 17.0</td><td>70.7 ± 10.2</td><td>78.8 ± 19.5</td><td>82.2 ± 21.8</td><td>83.7 ± 25.7</td><td>104.0 ± 19.9</td><td>104.2 ± 16.2</td><td>104.2 ± 16.2</td></tr><tr><td>3</td><td>12.2 ± 3.7</td><td>8.4 ± 1.6</td><td>8.1 ± 0.4</td><td>66.5 ± 20.8</td><td>63.1 ± 3.2</td><td>92.8 ± 12.8</td><td>105.0 ± 20.6</td><td>109.0 ± 6.4</td><td>90.8 ± 22.1</td><td>90.9 ± 26.5</td><td>104.1 ± 11.8</td><td>91.8 ± 23.8</td><td>74.0 ± 7.1</td><td>85.2 ± 14.2</td><td>90.1 ± 16.6</td><td>92.2 ± 22.0</td><td>106.2 ± 4.7</td><td>106.2 ± 3.4</td></tr><tr><td>4</td><td>13.1 ± 3.4</td><td>8.7 ± 1.7</td><td>8.2 ± 0.3</td><td>71.4 ± 16.3</td><td>63.9 ± 2.9</td><td>95.9 ± 9.3</td><td>108.4 ± 13.0</td><td>110.3 ± 4.1</td><td>95.9 ± 16.0</td><td>96.8 ± 19.3</td><td>106.7 ± 8.3</td><td>97.5 ± 18.2</td><td>75.6 ± 5.3</td><td>88.6 ± 11.5</td><td>93.9 ± 11.8</td><td>97.4 ± 18.4</td><td>107.0 ± 2.5</td><td>106.9 ± 1.9</td></tr><tr><td>5</td><td>13.7 ± 3.1</td><td>8.9 ± 1.3</td><td>8.3 ± 0.2</td><td>74.4 ± 13.3</td><td>64.5 ± 2.7</td><td>97.7 ± 7.3</td><td>109.8 ± 8.2</td><td>111.0 ± 11.8</td><td>98.1 ± 11.8</td><td>100.2 ± 14.6</td><td>108.1 ± 6.1</td><td>97.0 ± 7.5</td><td>76.5 ± 4.0</td><td>90.9 ± 10.1</td><td>99.0 ± 10.1</td><td>100.8 ± 15.3</td><td>107.1 ± 10.3</td><td>107.1 ± 13.1</td></tr><tr><td>6</td><td>9.2 ± 1.8</td><td>8.3 ± 0.2</td><td></td><td></td><td>64.9 ± 2.4</td><td>98.8 ± 5.9</td><td></td><td>111.2 ± 2.0</td><td>100.3 ± 8.9</td><td></td><td>108.9 ± 4.5</td><td>103.2 ± 11.4</td><td></td><td>92.3 ± 8.7</td><td>97.0 ± 6.1</td><td></td><td>107.5 ± 1.1</td><td>107.4 ± 0.9</td></tr><tr><td>7</td><td>9.4 ± 1.7</td><td>8.3 ± 0.2</td><td></td><td></td><td>65.3 ± 2.1</td><td>99.6 ± 4.9</td><td></td><td>111.2 ± 1.5</td><td>101.5 ± 6.9</td><td></td><td>109.8 ± 2.8</td><td>104.8 ± 9.4</td><td></td><td>92.3 ± 8.7</td><td>97.0 ± 6.1</td><td></td><td>107.6 ± 0.7</td><td>107.5 ± 0.7</td></tr><tr><td>8</td><td>9.5 ± 1.8</td><td>8.4 ± 0.2</td><td></td><td></td><td>65.5 ± 2.0</td><td>100.2 ± 4.2</td><td></td><td>111.8 ± 1.2</td><td>102.1 ± 5.5</td><td></td><td>109.8 ± 2.8</td><td>105.8 ± 7.8</td><td></td><td>94.4 ± 6.8</td><td>98.1 ± 3.4</td><td></td><td>107.8 ± 0.8</td><td>107.5 ± 0.5</td></tr><tr><td>9</td><td>9.7 ± 1.8</td><td>8.4 ± 0.2</td><td></td><td></td><td>65.8 ± 1.8</td><td>100.6 ± 3.6</td><td></td><td>111.9 ± 1.0</td><td>102.7 ± 4.6</td><td></td><td>110.1 ± 2.4</td><td>106.6 ± 8.7</td><td></td><td>95.0 ± 6.0</td><td>98.4 ± 2.6</td><td></td><td>107.8 ± 0.7</td><td>107.5 ± 0.6</td></tr><tr><td>10</td><td>9.9 ± 1.8</td><td>8.4 ± 0.2</td><td></td><td></td><td>65.9 ± 1.7</td><td>100.9 ± 3.1</td><td></td><td>112.0 ± 0.9</td><td>103.1 ± 3.9</td><td></td><td>110.3 ± 2.2</td><td>107.3 ± 5.8</td><td></td><td>95.6 ± 5.3</td><td>98.6 ± 2.3</td><td></td><td>107.9 ± 0.7</td><td>107.6 ± 0.4</td></tr><tr><td>11</td><td>10.0 ± 1.8</td><td>8.4 ± 0.1</td><td></td><td></td><td>66.1 ± 1.6</td><td>101.2 ± 2.7</td><td></td><td>112.0 ± 0.8</td><td>103.4 ± 3.5</td><td></td><td>110.4 ± 2.0</td><td>107.7 ± 5.1</td><td></td><td>96.0 ± 4.7</td><td>98.7 ± 1.8</td><td></td><td>108.0 ± 0.6</td><td>107.6 ± 0.3</td></tr><tr><td>12</td><td>10.1 ± 1.8</td><td>8.4 ± 0.1</td><td></td><td></td><td>66.3 ± 1.6</td><td>101.3 ± 2.3</td><td></td><td>112.2 ± 0.7</td><td>103.7 ± 3.2</td><td></td><td>110.6 ± 1.9</td><td>108.1 ± 8.5</td><td></td><td>96.3 ± 4.1</td><td>98.0 ± 2.5</td><td></td><td>108.0 ± 0.3</td><td>107.6 ± 0.3</td></tr><tr><td>13</td><td>10.2 ± 1.7</td><td>8.4 ± 0.1</td><td></td><td></td><td>66.3 ± 1.4</td><td>101.5 ± 2.0</td><td></td><td>112.2 ± 0.7</td><td>103.9 ± 2.9</td><td></td><td>110.7 ± 1.8</td><td>108.4 ± 4.0</td><td></td><td>96.5 ± 3.6</td><td>99.0 ± 1.4</td><td></td><td>108.1 ± 0.5</td><td>107.7 ± 0.2</td></tr><tr><td>14</td><td>10.3 ± 1.7</td><td>8.4 ± 0.1</td><td></td><td></td><td>66.4 ± 1.7</td><td>101.6 ± 1.8</td><td></td><td>112.2 ± 0.7</td><td>104.1 ± 2.8</td><td></td><td>110.8 ± 1.8</td><td>108.7 ± 3.6</td><td></td><td>96.5 ± 3.6</td><td>99.0 ± 1.2</td><td></td><td>108.1 ± 0.2</td><td>107.7 ± 0.2</td></tr><tr><td>15</td><td>10.4 ± 1.7</td><td>8.4 ± 0.1</td><td></td><td></td><td>66.5 ± 1.3</td><td>101.7 ± 1.6</td><td></td><td>112.2 ± 0.6</td><td>104.3 ± 2.6</td><td></td><td>110.9 ± 1.7</td><td>108.9 ± 3.3</td><td></td><td>96.9 ± 2.8</td><td>99.1 ± 1.1</td><td></td><td>108.1 ± 0.5</td><td>107.7 ± 0.2</td></tr><tr><td>16</td><td>10.5 ± 1.6</td><td>8.5 ± 0.1</td><td></td><td></td><td>66.6 ± 1.2</td><td>101.8 ± 1.4</td><td></td><td>112.2 ± 0.6</td><td>104.6 ± 2.5</td><td></td><td>111.7 ± 1.0</td><td>109.0 ± 2.4</td><td></td><td>97.0 ± 2.5</td><td>99.2 ± 1.0</td><td></td><td>108.2 ± 0.4</td><td>107.9 ± 0.1</td></tr><tr><td>17</td><td>10.6 ± 1.6</td><td>8.5 ± 0.1</td><td></td><td></td><td>66.6 ± 1.2</td><td>101.9 ± 1.3</td><td></td><td>112.3 ± 0.5</td><td>104.6 ± 2.4</td><td></td><td>111.1 ± 1.7</td><td>109.3 ± 2.7</td><td></td><td>97.1 ± 2.2</td><td>99.3 ± 1.0</td><td></td><td>108.2 ± 0.4</td><td>107.7 ± 0.1</td></tr><tr><td>18</td><td>-</td><td>10.7 ± 1.5</td><td>8.5 ± 0.1</td><td></td><td>66.7 ± 1.1</td><td>101.9 ± 1.2</td><td></td><td>112.3 ± 0.5</td><td>104.7 ± 2.3</td><td></td><td>111.2 ± 1.7</td><td>109.4 ± 2.5</td><td></td><td>97.2 ± 1.9</td><td>99.3 ± 1.0</td><td></td><td>108.2 ± 0.4</td><td>107.7 ± 0.1</td></tr><tr><td>19</td><td>-</td><td>10.8 ± 1.5</td><td>8.5 ± 0.1</td><td></td><td>66.8 ± 1.2</td><td>102.0 ± 1.1</td><td></td><td>112.3 ± 0.6</td><td>104.8 ± 2.1</td><td></td><td>111.2 ± 1.6</td><td>109.5 ± 2.3</td><td></td><td>97.3 ± 2.7</td><td>99.4 ± 0.9</td><td></td><td>108.2 ± 0.2</td><td>107.7 ± 0.1</td></tr><tr><td>20</td><td>-</td><td>10.8 ± 1.5</td><td>8.5 ± 0.1</td><td></td><td>66.8 ± 1.0</td><td>102.0 ± 1.0</td><td></td><td>112.4 ± 0.5</td><td>104.9 ± 2.1</td><td></td><td>111.3 ± 1.6</td><td>109.6 ± 2.1</td><td></td><td>97.3 ± 1.5</td><td>99.4 ± 0.9</td><td></td><td>108.2 ± 0.3</td><td>107.7 ± 0.1</td></tr></table>

![](images/1c6c04e94bba1060eca856923afe3d8e9cd9ef7638f121fbc447e560cef9904f.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 7.5    | 10.0     | 7.5 |
| 2                         | 7.5    | 12.0     | 8.0 |
| 3                         | 7.5    | 13.0     | 8.5 |
| 4                         | 7.5    | 14.0     | 9.0 |
| 5                         | 7.5    | 14.5     | 9.5 |
| 6                         | 7.5    | 14.5     | 10.0 |
| 7                         | 7.5    | 14.5     | 10.5 |
| 8                         | 7.5    | 14.5     | 11.0 |
| 9                         | 7.5    | 14.5     | 11.5 |
| 10                        | 7.5    | 14.5     | 12.0 |
| 11                        | 7.5    | 14.5     | 12.5 |
| 12                        | 7.5    | 14.5     | 13.0 |
| 13                        | 7.5    | 14.5     | 13.5 |
| 14                        | 7.5    | 14.5     | 14.0 |
| 15                        | 7.5    | 14.5     | 14.5 |
| 16                        | 7.5    | 14.5     | 15.0 |
| 17                        | 7.5    | 14.5     | 15.5 |
| 18                        | 7.5    | 14.5     | 16.0 |
| 19                        | 7.5    | 14.5     | 16.5 |
| 20                        | 7.5    | 14.5     | 17.0 |
</details>

(a) hopper-random EOP.

![](images/a658488d649d7ac72f34846b17e7a306a1dd30edc666f680b44da7a5bc59e3c6.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 70     | 40       | 60  |
| 2                         | 85     | 60       | 60  |
| 3                         | 90     | 70       | 60  |
| 4                         | 95     | 75       | 60  |
| 5                         | 98     | 78       | 60  |
| 6                         | 99     | 80       | 60  |
| 7                         | 99.5   | 82       | 60  |
| 8                         | 99.8   | 83       | 60  |
| 9                         | 99.9   | 84       | 60  |
| 10                        | 99.95  | 85       | 60  |
| 11                        | 99.98  | 86       | 60  |
| 12                        | 99.99  | 87       | 60  |
| 13                        | 99.995 | 88       | 60  |
| 14                        | 99.998 | 89       | 60  |
| 15                        | 99.999 | 90       | 60  |
| 16                        | 99.9995| 91       | 60  |
| 17                        | 99.9998| 92       | 60  |
| 18                        | 99.9999| 93       | 60  |
| 19                        | 99.99995|100    | 60 |
| 20                        | 100    |        |         |
</details>

(b) hopper-medium EOP.

![](images/fc418414cfe5cc71bad77b17642ab8b4edd6acba21a0d3d72e7364752b551ef4.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 60     | 70       | 95  |
| 2                         | 80     | 90       | 105 |
| 3                         | 90     | 100      | 110 |
| 4                         | 95     | 105      | 110 |
| 5                         | 95     | 105      | 110 |
| 6                         | 95     | 105      | 110 |
| 7                         | 95     | 105      | 110 |
| 8                         | 95     | 105      | 110 |
| 9                         | 95     | 105      | 110 |
| 10                        | 95     | 105      | 110 |
| 11                        | 95     | 105      | 110 |
| 12                        | 95     | 105      | 110 |
| 13                        | 95     | 105      | 110 |
| 14                        | 95     | 105      | 110 |
| 15                        | 95     | 105      | 110 |
| 16                        | 95     | 105      | 110 |
| 17                        | 95     | 105      | 110 |
| 18                        | 95     | 105      | 110 |
| 19                        | 95     | 105      | 110 |
| 20                        | 95     | 105      | 110 |
</details>

(c) hopper-expert EOP.

![](images/c949bb78317f0c110b8c21869cc81221cc594701862669dc205240e18138af16.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 60     | 55       | 85  |
| 2                         | 75     | 70       | 95  |
| 3                         | 85     | 80       | 100 |
| 4                         | 90     | 85       | 102 |
| 5                         | 92     | 88       | 104 |
| 6                         | 94     | 90       | 105 |
| 7                         | 95     | 92       | 106 |
| 8                         | 96     | 94       | 107 |
| 9                         | 97     | 95       | 108 |
| 10                        | 98     | 96       | 109 |
| 11                        | 98.5   | 97       | 109.5 |
| 12                        | 99     | 98       | 110 |
| 13                        | 99.5   | 98.5     | 110.5 |
| 14                        | 100    | 99       | 111 |
| 15                        | 100.5  | 99.5     | 111.5 |
| 16                        | 101    | 100      | 112 |
| 17                        | 101.5  | 100.5    | 112.5 |
| 18                        | 102    | 101      | 113 |
| 19                        | 102.5  | 101.5    | 113.5 |
| 20                        | 103    | 102      | 114 |
</details>

(d) hopper-medium-expert EOP.

![](images/753c780fcc64cbb18054cc23de50f4a6e001870d53b26d6304a532926b5b611b.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 65     | 65       | 65  |
| 2                         | 70     | 70       | 70  |
| 3                         | 75     | 75       | 75  |
| 4                         | 80     | 80       | 80  |
| 5                         | 85     | 85       | 85  |
| 6                         | 90     | 90       | 90  |
| 7                         | 92     | 92       | 92  |
| 8                         | 94     | 94       | 94  |
| 9                         | 95     | 95       | 95  |
| 10                        | 96     | 96       | 96  |
| 11                        | 97     | 97       | 97  |
| 12                        | 97     | 97       | 97  |
| 13                        | 98     | 98       | 98  |
| 14                        | 98     | 98       | 98  |
| 15                        | 98     | 98       | 98  |
| 16                        | 98     | 98       | 98  |
| 17                        | 98     | 98       | 98  |
| 18                        | 98     | 98       | 98  |
| 19                        | 98     | 98       | 98  |
| 20                        | 98     | 98       | 98  |
</details>

(e) hopper-medium-replay EOP.

![](images/90f6e2d999bcd449829a64b245201dc2592ff4540acecfdf9e9c8dbf0b0ed120.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 95     | 70       | 95  |
| 2                         | 100    | 85       | 100 |
| 3                         | 100    | 95       | 100 |
| 4                         | 100    | 100      | 100 |
| 5                         | 100    | 100      | 100 |
| 6                         | 100    | 100      | 100 |
| 7                         | 100    | 100      | 100 |
| 8                         | 100    | 100      | 100 |
| 9                         | 100    | 100      | 100 |
| 10                        | 100    | 100      | 100 |
| 11                        | 100    | 100      | 100 |
| 12                        | 100    | 100      | 100 |
| 13                        | 100    | 100      | 100 |
| 14                        | 100    | 100      | 100 |
| 15                        | 100    | 100      | 100 |
| 16                        | 100    | 100      | 100 |
| 17                        | 100    | 100      | 100 |
| 18                        | 100    | 100      | 100 |
| 19                        | 100    | 100      | 100 |
| 20                        | 100    | 100      | 100 |
</details>

(f) hopper-full-replay EOP.   
Figure 4: TD3+BC, IQL and ReBRAC visualised Expected Online Performance under uniform policy selection on Hopper tasks.

Table 27: TD3+BC, IQL and ReBRAC Expected Online Performance under uniform policy selection on Walker2d tasks. 

<table><tr><td></td><td colspan="3">random</td><td colspan="3">medium</td><td colspan="3">expert</td><td colspan="3">medium-expert</td><td colspan="3">medium-replay</td><td colspan="3">full-replay</td></tr><tr><td>Policies</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td></tr><tr><td>1</td><td>3.2 ± 1.0</td><td>6.3 ± 2.9</td><td>7.9 ± 6.5</td><td>41.3 ± 28.6</td><td>65.3 ± 17.8</td><td>54.1 ± 34.5</td><td>67.0 ± 52.4</td><td>110.3 ± 4.0</td><td>84.2 ± 45.7</td><td>70.9 ± 46.5</td><td>103.8 ± 12.2</td><td>83.2 ± 46.9</td><td>36.4 ± 25.2</td><td>51.9 ± 28.3</td><td>54.5 ± 26.0</td><td>78.9 ± 15.4</td><td>71.8 ± 28.7</td><td>86.1 ± 21.5</td></tr><tr><td>2</td><td>3.7 ± 0.8</td><td>7.8 ± 2.8</td><td>11.4 ± 6.6</td><td>57.0 ± 23.4</td><td>75.0 ± 13.0</td><td>72.5 ± 23.4</td><td>92.9 ± 39.0</td><td>112.1 ± 1.8</td><td>104.3 ± 25.4</td><td>94.8 ± 32.2</td><td>109.6 ± 6.4</td><td>103.8 ± 26.2</td><td>50.0 ± 19.0</td><td>67.5 ± 21.0</td><td>68.8 ± 18.7</td><td>87.1 ± 10.5</td><td>87.1 ± 17.9</td><td>96.2 ± 10.2</td></tr><tr><td>3</td><td>4.0 ± 0.7</td><td>8.8 ± 2.6</td><td>13.6 ± 6.3</td><td>64.7 ± 17.7</td><td>79.0 ± 8.7</td><td>79.6 ± 14.9</td><td>103.2 ± 26.0</td><td>112.7 ± 1.2</td><td>109.5 ± 13.0</td><td>103.7 ± 20.6</td><td>111.2 ± 3.3</td><td>109.2 ± 13.4</td><td>56.1 ± 13.8</td><td>74.2 ± 14.8</td><td>74.8 ± 13.1</td><td>90.3 ± 6.8</td><td>92.6 ± 11.1</td><td>99.0 ± 5.7</td></tr><tr><td>4</td><td>4.2 ± 0.6</td><td>9.4 ± 2.3</td><td>15.1 ± 15.9</td><td>68.9 ± 13.4</td><td>80.5 ± 5.8</td><td>82.4 ± 9.5</td><td>107.4 ± 16.8</td><td>113.0 ± 1.0</td><td>111.6 ± 0.6</td><td>107.3 ± 13.1</td><td>111.2 ± 3.0</td><td>110.1 ± 6.8</td><td>59.2 ± 10.5</td><td>77.4 ± 4.4</td><td>77.6 ± 5.2</td><td>91.6 ± 4.4</td><td>94.9 ± 7.2</td><td>100.9 ± 3.5</td></tr><tr><td>5</td><td>4.3 ± 0.4</td><td>9.9 ± 2.1</td><td>16.3 ± 5.4</td><td>71.3 ± 10.4</td><td>81.6 ± 3.9</td><td>83.9 ± 6.2</td><td>109.1 ± 10.7</td><td>113.2 ± 0.8</td><td>111.5 ± 3.4</td><td>108.8 ± 8.3</td><td>111.9 ± 0.9</td><td>111.2 ± 3.5</td><td>61.0 ± 8.5</td><td>79.1 ± 7.4</td><td>79.4 ± 6.5</td><td>92.3 ± 3.0</td><td>96.0 ± 4.8</td><td>101.0 ± 3.2</td></tr><tr><td></td><td>6</td><td>10.2 ± 1.9</td><td>17.2 ± 5.0</td><td></td><td>82.0 ± 2.7</td><td>84.6 ± 4.1</td><td></td><td>113.3 ± 0.7</td><td>111.7 ± 1.8</td><td></td><td>112.0 ± 0.6</td><td>111.4 ± 1.9</td><td></td><td>80.1 ± 5.4</td><td>80.0 ± 4.7</td><td></td><td>96.7 ± 3.4</td><td>101.5 ± 2.8</td></tr><tr><td>7</td><td></td><td>10.5 ± 1.7</td><td>17.9 ± 4.6</td><td></td><td>82.1 ± 1.9</td><td>85.0 ± 2.8</td><td></td><td>113.4 ± 0.6</td><td>111.8 ± 1.1</td><td></td><td>112.1 ± 0.4</td><td>111.5 ± 1.1</td><td></td><td>80.0 ± 4.1</td><td>80.5 ± 3.6</td><td></td><td>97.0 ± 2.5</td><td>101.9 ± 2.5</td></tr><tr><td>8</td><td></td><td>10.7 ± 1.6</td><td>18.4 ± 4.2</td><td></td><td>82.3 ± 1.3</td><td>85.2 ± 2.0</td><td></td><td>113.5 ± 0.6</td><td>111.9 ± 0.7</td><td></td><td>112.1 ± 0.3</td><td>111.6 ± 0.8</td><td></td><td>81.1 ± 3.2</td><td>80.9 ± 2.9</td><td></td><td>97.3 ± 1.9</td><td>102.1 ± 2.3</td></tr><tr><td>9</td><td></td><td>10.9 ± 1.4</td><td>18.9 ± 3.8</td><td></td><td>82.4 ± 0.9</td><td>85.3 ± 1.5</td><td></td><td>113.6 ± 0.5</td><td>111.9 ± 0.6</td><td></td><td>112.2 ± 0.3</td><td>111.7 ± 0.6</td><td></td><td>81.4 ± 2.6</td><td>81.2 ± 2.4</td><td></td><td>97.4 ± 1.5</td><td>102.4 ± 2.0</td></tr><tr><td>10</td><td></td><td>11.0 ± 1.3</td><td>19.3 ± 4.5</td><td></td><td>82.4 ± 0.7</td><td>85.2 ± 1.2</td><td></td><td>113.6 ± 0.5</td><td>112.0 ± 0.5</td><td></td><td>112.2 ± 0.3</td><td>111.7 ± 0.5</td><td></td><td>81.0 ± 2.1</td><td>81.4 ± 2.1</td><td></td><td>97.5 ± 1.8</td><td>102.1 ± 1.9</td></tr><tr><td>11</td><td></td><td>11.1 ± 1.3</td><td>19.6 ± 3.2</td><td></td><td>82.4 ± 0.5</td><td>85.5 ± 1.0</td><td></td><td>113.6 ± 0.4</td><td>112.0 ± 0.4</td><td></td><td>112.2 ± 0.3</td><td>111.7 ± 0.4</td><td></td><td>81.8 ± 1.9</td><td>81.5 ± 1.9</td><td></td><td>97.6 ± 1.0</td><td>102.7 ± 1.6</td></tr><tr><td>12</td><td></td><td>11.2 ± 1.2</td><td>19.8 ± 3.0</td><td></td><td>82.3 ± 0.3</td><td>85.6 ± 0.8</td><td></td><td>113.6 ± 0.4</td><td>112.0 ± 0.4</td><td></td><td>112.2 ± 0.3</td><td>111.7 ± 0.4</td><td></td><td>81.7 ± 1.7</td><td>81.7 ± 1.8</td><td></td><td>97.7 ± 0.9</td><td>102.0 ± 1.5</td></tr><tr><td>13</td><td></td><td>11.3 ± 1.1</td><td>20.1 ± 2.8</td><td></td><td>82.5 ± 0.3</td><td>85.6 ± 0.8</td><td></td><td>113.7 ± 0.4</td><td>112.0 ± 0.3</td><td></td><td>112.3 ± 0.2</td><td>111.8 ± 0.3</td><td></td><td>82.0 ± 1.5</td><td>81.8 ± 1.7</td><td></td><td>97.8 ± 0.7</td><td>102.9 ± 1.3</td></tr><tr><td>14</td><td></td><td>11.4 ± 1.0</td><td>20.2 ± 2.6</td><td></td><td>82.5 ± 0.2</td><td>85.7 ± 0.7</td><td></td><td>113.7 ± 0.3</td><td>112.1 ± 0.3</td><td></td><td>112.3 ± 0.2</td><td>111.8 ± 0.3</td><td></td><td>82.1 ± 1.4</td><td>81.9 ± 1.6</td><td></td><td>97.8 ± 0.7</td><td>103.0 ± 1.2</td></tr><tr><td>15</td><td></td><td>11.5 ± 1.0</td><td>20.4 ± 2.4</td><td></td><td>82.5 ± 0.1</td><td>85.7 ± 0.7</td><td></td><td>113.7 ± 0.3</td><td>112.1 ± 0.2</td><td></td><td>112.3 ± 0.2</td><td>111.8 ± 0.3</td><td></td><td>82.3 ± 1.3</td><td>82.0 ± 1.6</td><td></td><td>97.9 ± 0.6</td><td>102.0 ± 1.1</td></tr><tr><td>16</td><td></td><td>11.5 ± 0.9</td><td>20.6 ± 2.2</td><td></td><td>82.5 ± 0.1</td><td>85.8 ± 0.6</td><td></td><td>113.8 ± 0.3</td><td>112.1 ± 0.2</td><td></td><td>112.3 ± 0.2</td><td>111.8 ± 0.2</td><td></td><td>82.3 ± 1.2</td><td>82.1 ± 1.5</td><td></td><td>97.9 ± 0.5</td><td>103.1 ± 0.5</td></tr><tr><td>17</td><td></td><td>11.6 ± 0.9</td><td>20.7 ± 2.1</td><td></td><td>82.5 ± 0.1</td><td>85.8 ± 0.6</td><td></td><td>113.8 ± 0.3</td><td>112.1 ± 0.2</td><td></td><td>112.3 ± 0.2</td><td>111.9 ± 0.2</td><td></td><td>82.4 ± 1.2</td><td>82.2 ± 1.4</td><td></td><td>97.9 ± 0.5</td><td>103.1 ± 0.8</td></tr><tr><td>18</td><td></td><td>11.6 ± 0.8</td><td>20.8 ± 1.9</td><td></td><td>82.5 ± 0.1</td><td>85.8 ± 0.6</td><td></td><td>113.8 ± 0.3</td><td>112.1 ± 0.2</td><td></td><td>112.3 ± 0.2</td><td>111.9 ± 0.2</td><td></td><td>82.4 ± 1.1</td><td>82.3 ± 1.4</td><td></td><td>97.9 ± 0.5</td><td>102.8 ± 0.8</td></tr><tr><td>19</td><td></td><td>11.7 ± 0.8</td><td>20.9 ± 1.8</td><td></td><td>82.5 ± 0.1</td><td>85.9 ± 0.6</td><td></td><td>113.8 ± 0.2</td><td>112.1 ± 0.1</td><td></td><td>112.3 ± 0.2</td><td>111.9 ± 0.2</td><td></td><td>82.5 ± 1.0</td><td>82.3 ± 1.3</td><td></td><td>98.0 ± 0.4</td><td>103.2 ± 0.7</td></tr><tr><td>20</td><td></td><td>11.7 ± 0.7</td><td>21.0 ± 1.7</td><td></td><td>82.5 ± 0.1</td><td>85.9 ± 0.5</td><td></td><td>113.8 ± 0.2</td><td>112.1 ± 0.1</td><td></td><td>112.3 ± 0.2</td><td>111.9 ± 0.2</td><td></td><td>82.5 ± 1.0</td><td>82.4 ± 1.3</td><td></td><td>98.0 ± 0.4</td><td>103.2 ± 0.6</td></tr></table>

![](images/8fd587b53af709044cb805794ee5a7447dea7f69696ea19a71b3d3291dbd0834.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| -------------------------- | ------ | -------- | --- |
| 1                          | 8      | 4        | 6   |
| 2                          | 10     | 4        | 7   |
| 3                          | 12     | 4        | 8   |
| 4                          | 14     | 4        | 9   |
| 5                          | 16     | 4        | 10  |
| 6                          | 17     | 4        | 10  |
| 7                          | 18     | 4        | 10  |
| 8                          | 19     | 4        | 10  |
| 9                          | 20     | 4        | 10  |
| 10                         | 20     | 4        | 10  |
| 11                         | 20     | 4        | 10  |
| 12                         | 20     | 4        | 10  |
| 13                         | 20     | 4        | 10  |
| 14                         | 20     | 4        | 10  |
| 15                         | 20     | 4        | 10  |
| 16                         | 20     | 4        | 10  |
| 17                         | 20     | 4        | 10  |
| 18                         | 20     | 4        | 10  |
| 19                         | 20     | 4        | 10  |
| 20                         | 20     | 4        | 10  |
</details>

(a) walker2d-random EOP.

![](images/34079821efc0011d871043a9b7c40a9e35094a06fbe50a26eb5f7050da256677.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 55     | 40       | 65  |
| 2                         | 70     | 55       | 75  |
| 3                         | 75     | 65       | 80  |
| 4                         | 80     | 70       | 82  |
| 5                         | 82     | 72       | 83  |
| 6                         | 83     | 73       | 84  |
| 7                         | 84     | 74       | 85  |
| 8                         | 85     | 75       | 85  |
| 9                         | 85     | 75       | 85  |
| 10                        | 85     | 75       | 85  |
| 11                        | 85     | 75       | 85  |
| 12                        | 85     | 75       | 85  |
| 13                        | 85     | 75       | 85  |
| 14                        | 85     | 75       | 85  |
| 15                        | 85     | 75       | 85  |
| 16                        | 85     | 75       | 85  |
| 17                        | 85     | 75       | 85  |
| 18                        | 85     | 75       | 85  |
| 19                        | 85     | 75       | 85  |
| 20                        | 85     | 75       | 85  |
</details>

(b) walker2d-medium EOP.

![](images/ec6620e17ae397f56f0630a2878442f67654438497b2bc161a8000a1f4819256.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 80     | 60       | 110 |
| 2                         | 100    | 90       | 110 |
| 3                         | 105    | 100      | 110 |
| 4                         | 105    | 105      | 110 |
| 5                         | 105    | 105      | 110 |
| 6                         | 105    | 105      | 110 |
| 7                         | 105    | 105      | 110 |
| 8                         | 105    | 105      | 110 |
| 9                         | 105    | 105      | 110 |
| 10                        | 105    | 105      | 110 |
| 11                        | 105    | 105      | 110 |
| 12                        | 105    | 105      | 110 |
| 13                        | 105    | 105      | 110 |
| 14                        | 105    | 105      | 110 |
| 15                        | 105    | 105      | 110 |
| 16                        | 105    | 105      | 110 |
| 17                        | 105    | 105      | 110 |
| 18                        | 105    | 105      | 110 |
| 19                        | 105    | 105      | 110 |
| 20                        | 105    | 105      | 110 |
</details>

(c) walker2d-expert EOP.

![](images/82bf18ae5b90f5c301bb9d6f46fe463c6021fc1d552d2f4461dbddb687655731.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 80     | 70       | 100 |
| 2                         | 90     | 80       | 105 |
| 3                         | 100    | 90       | 110 |
| 4                         | 105    | 100      | 110 |
| 5                         | 105    | 105      | 110 |
| 6                         | 105    | 105      | 110 |
| 7                         | 105    | 105      | 110 |
| 8                         | 105    | 105      | 110 |
| 9                         | 105    | 105      | 110 |
| 10                        | 105    | 105      | 110 |
| 11                        | 105    | 105      | 110 |
| 12                        | 105    | 105      | 110 |
| 13                        | 105    | 105      | 110 |
| 14                        | 105    | 105      | 110 |
| 15                        | 105    | 105      | 110 |
| 16                        | 105    | 105      | 110 |
| 17                        | 105    | 105      | 110 |
| 18                        | 105    | 105      | 110 |
| 19                        | 105    | 105      | 110 |
| 20                        | 105    | 105      | 110 |
</details>

(d) walker2d-medium-expert EOP.

![](images/6a794c23b1eadb38bb358a3449728d69c1b3482f524371ac046c642faea91eca.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| -------------------------- | ------ | -------- | --- |
| 1                          | 50     | 40       | 55  |
| 2                          | 55     | 50       | 60  |
| 3                          | 60     | 55       | 65  |
| 4                          | 65     | 60       | 70  |
| 5                          | 70     | 65       | 75  |
| 6                          | 75     | 70       | 78  |
| 7                          | 78     | 75       | 80  |
| 8                          | 80     | 78       | 82  |
| 9                          | 82     | 80       | 83  |
| 10                         | 83     | 82       | 84  |
| 11                         | 84     | 83       | 85  |
| 12                         | 85     | 84       | 85  |
| 13                         | 85     | 85       | 85  |
| 14                         | 85     | 85       | 85  |
| 15                         | 85     | 85       | 85  |
| 16                         | 85     | 85       | 85  |
| 17                         | 85     | 85       | 85  |
| 18                         | 85     | 85       | 85  |
| 19                         | 85     | 85       | 85  |
| 20                         | 85     | 85       | 85  |
</details>

(e) walker2d-medium-replay EOP.

![](images/238af4363519be08e4dcd2028c885860df0c54fad010b98a0a962ec4788663f2.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 85     | 75       | 70  |
| 2                         | 95     | 85       | 80  |
| 3                         | 98     | 90       | 85  |
| 4                         | 99     | 92       | 88  |
| 5                         | 100    | 93       | 90  |
| 6                         | 100    | 94       | 91  |
| 7                         | 100    | 95       | 92  |
| 8                         | 100    | 95       | 93  |
| 9                         | 100    | 96       | 94  |
| 10                        | 100    | 96       | 95  |
| 11                        | 100    | 97       | 96  |
| 12                        | 100    | 97       | 96  |
| 13                        | 100    | 98       | 97  |
| 14                        | 100    | 98       | 97  |
| 15                        | 100    | 98       | 98  |
| 16                        | 100    | 98       | 98  |
| 17                        | 100    | 98       | 98  |
| 18                        | 100    | 98       | 98  |
| 19                        | 100    | 98       | 98  |
| 20                        | 100    | 98       | 98  |
</details>

(f) walker2d-full-replay EOP.   
Figure 5: TD3+BC, IQL and ReBRAC visualised Expected Online Performance under uniform policy selection on Walker2d tasks.

Table 28: TD3+BC, IQL and ReBRAC Expected Online Performance under uniform policy selection on AntMaze tasks. 

<table><tr><td></td><td colspan="3">umaze</td><td colspan="3">medium-play</td><td colspan="3">large-play</td><td colspan="3">umaze-diverse</td><td colspan="3">medium-diverse</td><td colspan="3">large-diverse</td></tr><tr><td>Policies</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td></tr><tr><td>1</td><td>12.4 ± 24.8</td><td>64.3 ± 12.0</td><td>87.5 ± 10.9</td><td>7.5 ± 11.7</td><td>22.7 ± 29.5</td><td>75.0 ± 14.8</td><td>0.0 ± 0.0</td><td>10.9 ± 16.2</td><td>52.7 ± 21.4</td><td>9.6 ± 19.2</td><td>52.8 ± 11.1</td><td>70.4 ± 16.2</td><td>11.9 ± 13.8</td><td>21.3 ± 26.8</td><td>65.3 ± 26.3</td><td>0.2 ± 0.2</td><td>6.7 ± 10.3</td><td>56.8 ± 17.0</td></tr><tr><td>2</td><td>22.3 ± 29.8</td><td>71.1 ± 9.2</td><td>93.2 ± 6.2</td><td>13.0 ± 13.3</td><td>37.5 ± 30.9</td><td>82.7 ± 8.9</td><td>0.0 ± 0.0</td><td>18.6 ± 17.8</td><td>64.1 ± 13.0</td><td>17.3 ± 23.0</td><td>58.7 ± 11.3</td><td>79.3 ± 11.2</td><td>19.2 ± 14.0</td><td>35.0 ± 27.7</td><td>79.2 ± 16.0</td><td>0.3 ± 0.2</td><td>11.6 ± 11.6</td><td>66.2 ± 11.3</td></tr><tr><td>3</td><td>30.3 ± 31.0</td><td>74.3 ± 7.5</td><td>95.1 ± 3.7</td><td>17.0 ± 13.2</td><td>47.4 ± 28.5</td><td>85.4 ± 5.7</td><td>0.0 ± 0.0</td><td>24.0 ± 16.9</td><td>68.5 ± 9.3</td><td>23.4 ± 24.0</td><td>62.5 ± 10.4</td><td>83.1 ± 8.4</td><td>23.8 ± 12.7</td><td>44.0 ± 25.1</td><td>84.2 ± 9.8</td><td>0.4 ± 0.2</td><td>15.2 ± 11.4</td><td>70.1 ± 8.5</td></tr><tr><td>4</td><td>36.6 ± 30.5</td><td>76.2 ± 6.5</td><td>96.0 ± 2.4</td><td>20.0 ± 12.6</td><td>54.1 ± 25.0</td><td>86.7 ± 4.2</td><td>0.0 ± 0.0</td><td>27.9 ± 55.3</td><td>78.3 ± 7.3</td><td>28.3 ± 23.6</td><td>65.1 ± 75.3</td><td>85.7 ± 7.8</td><td>26.9 ± 11.1</td><td>50.0 ± 16.7</td><td>86.4 ± 6.6</td><td>0.4 ± 0.2</td><td>17.9 ± 10.6</td><td>72.5 ± 7.8</td></tr><tr><td>5</td><td>41.7 ± 29.1</td><td>77.5 ± 5.9</td><td>96.4 ± 1.8</td><td>22.2 ± 11.7</td><td>58.7 ± 21.4</td><td>87.5 ± 3.5</td><td>0.0 ± 0.0</td><td>30.8 ± 13.6</td><td>72.2 ± 6.0</td><td>32.3 ± 22.5</td><td>67.0 ± 8.2</td><td>86.5 ± 6.3</td><td>29.0 ± 9.7</td><td>54.0 ± 18.4</td><td>87.6 ± 4.9</td><td>0.5 ± 0.1</td><td>19.9 ± 9.7</td><td>73.5 ± 6.2</td></tr><tr><td>6</td><td>-</td><td>78.4 ± 5.4</td><td>96.7 ± 1.4</td><td>-</td><td>61.9 ± 18.2</td><td>88.3 ± 3.2</td><td>-</td><td>32.8 ± 11.8</td><td>73.1 ± 4.9</td><td>-</td><td>68.4 ± 7.3</td><td>87.6 ± 5.7</td><td>-</td><td>56.8 ± 15.4</td><td>88.4 ± 3.9</td><td>-</td><td>21.5 ± 8.8</td><td>74.5 ± 5.7</td></tr><tr><td>7</td><td>-</td><td>79.2 ± 5.9</td><td>96.9 ± 1.1</td><td>-</td><td>64.1 ± 15.3</td><td>88.5 ± 2.9</td><td>-</td><td>34.3 ± 10.3</td><td>73.7 ± 4.1</td><td>-</td><td>69.4 ± 12.4</td><td>88.4 ± 5.3</td><td>-</td><td>58.8 ± 13.0</td><td>88.9 ± 3.2</td><td>-</td><td>22.7 ± 8.0</td><td>75.3 ± 5.4</td></tr><tr><td>8</td><td>-</td><td>79.8 ± 4.6</td><td>97.0 ± 0.9</td><td>-</td><td>65.7 ± 12.8</td><td>88.9 ± 2.7</td><td>-</td><td>35.5 ± 8.9</td><td>74.2 ± 3.5</td><td>-</td><td>70.2 ± 5.7</td><td>89.0 ± 4.9</td><td>-</td><td>60.1 ± 10.7</td><td>89.3 ± 2.8</td><td>-</td><td>23.7 ± 7.2</td><td>75.9 ± 5.1</td></tr><tr><td>9</td><td>-</td><td>80.3 ± 4.3</td><td>97.1 ± 0.7</td><td>-</td><td>66.8 ± 10.7</td><td>89.2 ± 2.5</td><td>-</td><td>36.3 ± 7.7</td><td>74.5 ± 2.9</td><td>-</td><td>70.8 ± 5.1</td><td>89.5 ± 4.5</td><td>-</td><td>61.1 ± 9.0</td><td>89.6 ± 2.4</td><td>-</td><td>24.4 ± 6.5</td><td>76.5 ± 4.9</td></tr><tr><td>10</td><td>-</td><td>80.7 ± 4.0</td><td>97.1 ± 0.6</td><td>-</td><td>67.6 ± 9.0</td><td>89.4 ± 2.4</td><td>-</td><td>37.0 ± 6.7</td><td>74.8 ± 2.5</td><td>-</td><td>71.3 ± 4.6</td><td>89.9 ± 4.2</td><td>-</td><td>61.8 ± 9.5</td><td>89.8 ± 2.1</td><td>-</td><td>25.1 ± 5.9</td><td>76.9 ± 4.7</td></tr><tr><td>11</td><td>-</td><td>81.1 ± 3.8</td><td>97.2 ± 0.6</td><td>-</td><td>68.2 ± 7.5</td><td>89.6 ± 2.2</td><td>-</td><td>37.5 ± 5.9</td><td>75.0 ± 2.1</td><td>-</td><td>71.7 ± 4.2</td><td>90.3 ± 3.9</td><td>-</td><td>62.4 ± 6.3</td><td>90.0 ± 1.9</td><td>-</td><td>25.6 ± 5.4</td><td>77.3 ± 4.6</td></tr><tr><td>12</td><td>-</td><td>81.4 ± 3.5</td><td>97.2 ± 0.5</td><td>-</td><td>68.7 ± 6.3</td><td>90.0 ± 2.1</td><td>-</td><td>37.9 ± 5.2</td><td>75.1 ± 1.8</td><td>-</td><td>72.0 ± 3.8</td><td>90.6 ± 3.6</td><td>-</td><td>62.8 ± 5.3</td><td>90.1 ± 1.6</td><td>-</td><td>26.0 ± 4.9</td><td>77.4 ± 4.4</td></tr><tr><td>13</td><td>-</td><td>81.6 ± 3.3</td><td>97.3 ± 0.4</td><td>-</td><td>69.0 ± 5.2</td><td>90.0 ± 2.0</td><td>-</td><td>38.2 ± 4.6</td><td>75.2 ± 1.6</td><td>-</td><td>72.3 ± 3.5</td><td>90.9 ± 3.3</td><td>-</td><td>63.0 ± 4.5</td><td>90.3 ± 1.5</td><td>-</td><td>26.4 ± 4.5</td><td>78.0 ± 4.3</td></tr><tr><td>14</td><td>-</td><td>81.9 ± 3.1</td><td>97.3 ± 0.4</td><td>-</td><td>69.2 ± 4.4</td><td>90.1 ± 1.9</td><td>-</td><td>38.5 ± 4.2</td><td>75.3 ± 1.4</td><td>-</td><td>72.6 ± 3.2</td><td>91.1 ± 3.1</td><td>-</td><td>63.3 ± 6.3</td><td>90.4 ± 1.3</td><td>-</td><td>26.7 ± 4.2</td><td>78.3 ± 4.1</td></tr><tr><td>15</td><td>-</td><td>82.1 ± 3.0</td><td>97.3 ± 0.4</td><td>-</td><td>69.4 ± 3.7</td><td>90.2 ± 1.8</td><td>-</td><td>38.8 ± 3.8</td><td>75.4 ± 1.2</td><td>-</td><td>72.8 ± 3.0</td><td>91.3 ± 2.9</td><td>-</td><td>63.5 ± 3.2</td><td>90.4 ± 1.2</td><td>-</td><td>27.0 ± 3.9</td><td>78.5 ± 4.0</td></tr><tr><td>16</td><td>-</td><td>82.2 ± 2.8</td><td>97.3 ± 0.3</td><td>-</td><td>69.5 ± 3.1</td><td>90.3 ± 1.7</td><td>-</td><td>39.0 ± 3.5</td><td>75.4 ± 1.1</td><td>-</td><td>72.9 ± 2.8</td><td>91.4 ± 2.7</td><td>-</td><td>63.6 ± 2.8</td><td>90.5 ± 1.1</td><td>-</td><td>27.2 ± 3.6</td><td>78.7 ± 3.9</td></tr><tr><td>17</td><td>-</td><td>82.4 ± 2.7</td><td>-</td><td>-</td><td>69.6 ± 2.6</td><td>-</td><td>-</td><td>39.1 ± 3.3</td><td>-</td><td>-</td><td>73.1 ± 2.7</td><td>-</td><td>-</td><td>63.7 ± 2.4</td><td>-</td><td>-</td><td>27.4 ± 3.4</td><td>-</td></tr><tr><td>18</td><td>-</td><td>82.5 ± 2.5</td><td>-</td><td>-</td><td>69.7 ± 2.2</td><td>-</td><td>-</td><td>39.3 ± 3.1</td><td>-</td><td>-</td><td>73.2 ± 2.5</td><td>-</td><td>-</td><td>63.8 ± 2.1</td><td>-</td><td>-</td><td>27.6 ± 3.2</td><td>-</td></tr><tr><td>19</td><td>-</td><td>82.7 ± 2.4</td><td>-</td><td>-</td><td>69.8 ± 1.9</td><td>-</td><td>-</td><td>39.4 ± 2.9</td><td>-</td><td>-</td><td>73.4 ± 2.4</td><td>-</td><td>-</td><td>63.9 ± 1.8</td><td>-</td><td>-</td><td>27.7 ± 3.0</td><td>-</td></tr><tr><td>20</td><td>-</td><td>82.8 ± 2.3</td><td>-</td><td>-</td><td>69.8 ± 1.6</td><td>-</td><td>-</td><td>39.6 ± 2.8</td><td>-</td><td>-</td><td>73.5 ± 2.3</td><td>-</td><td>-</td><td>64.0 ± 1.6</td><td>-</td><td>-</td><td>27.9 ± 2.9</td><td>-</td></tr></table>

![](images/1fc5e42e39c691bb338ba9e3b81c84c21f37ca22b69d265429137661b1bfdfa6.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 90     | 15       | 65  |
| 2                         | 92     | 25       | 70  |
| 3                         | 93     | 35       | 75  |
| 4                         | 94     | 40       | 78  |
| 5                         | 95     | 45       | 80  |
| 6                         | 95     | 50       | 82  |
| 7                         | 96     | 55       | 83  |
| 8                         | 96     | 60       | 84  |
| 9                         | 96     | 65       | 85  |
| 10                        | 96     | 70       | 86  |
| 11                        | 96     | 75       | 87  |
| 12                        | 96     | 80       | 88  |
| 13                        | 96     | 85       | 89  |
| 14                        | 96     | 90       | 90  |
| 15                        | 96     | 95       | 91  |
| 16                        | 96     | 95       | 92  |
| 17                        | 96     | 95       | 93  |
| 18                        | 96     | 95       | 94  |
| 19                        | 96     | 95       | 95  |
| 20                        | 96     | 95       | 95  |
</details>

(a) antmaze-umaze EOP.

![](images/674a9e38519bbf1296f70b243c50f30829d456c1d8b7936cea38e79204520fac.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 75     | 10       | 20  |
| 2                         | 80     | 15       | 30  |
| 3                         | 82     | 20       | 40  |
| 4                         | 83     | 25       | 50  |
| 5                         | 84     | 30       | 60  |
| 6                         | 85     | 35       | 65  |
| 7                         | 85     | 40       | 68  |
| 8                         | 85     | 45       | 70  |
| 9                         | 85     | 50       | 72  |
| 10                        | 85     | 55       | 73  |
| 11                        | 85     | 60       | 74  |
| 12                        | 85     | 65       | 75  |
| 13                        | 85     | 70       | 76  |
| 14                        | 85     | 75       | 77  |
| 15                        | 85     | 80       | 78  |
| 16                        | 85     | 85       | 79  |
| 17                        | 85     | 90       | 80  |
| 18                        | 85     | 95       | 80  |
| 19                        | 85     | 100      | 80  |
| 20                        | 85     | 105      | 80  |
</details>

(b) antmaze-medium-play EOP.

![](images/60d3e78d735bb6239679d597fc8a5d335051a5d4ae98c7a743773d74db9e96e4.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 60     | 0        | 10  |
| 2                         | 70     | 0        | 20  |
| 3                         | 75     | 0        | 25  |
| 4                         | 78     | 0        | 30  |
| 5                         | 80     | 0        | 35  |
| 6                         | 80     | 0        | 38  |
| 7                         | 80     | 0        | 40  |
| 8                         | 80     | 0        | 40  |
| 9                         | 80     | 0        | 40  |
| 10                        | 80     | 0        | 40  |
| 11                        | 80     | 0        | 40  |
| 12                        | 80     | 0        | 40  |
| 13                        | 80     | 0        | 40  |
| 14                        | 80     | 0        | 40  |
| 15                        | 80     | 0        | 40  |
| 16                        | 80     | 0        | 40  |
| 17                        | 80     | 0        | 40  |
| 18                        | 80     | 0        | 40  |
| 19                        | 80     | 0        | 40  |
| 20                        | 80     | 0        | 40  |
</details>

(c) antmaze-large-play EOP.

![](images/4b9af3703cad049926a167f43fbdcca34d571e133f18423a343b1d87773092ee.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 70     | 10       | 50  |
| 2                         | 75     | 15       | 55  |
| 3                         | 80     | 20       | 60  |
| 4                         | 82     | 25       | 62  |
| 5                         | 83     | 30       | 63  |
| 6                         | 84     | 35       | 64  |
| 7                         | 85     | 40       | 65  |
| 8                         | 86     | 45       | 66  |
| 9                         | 87     | 50       | 67  |
| 10                        | 88     | 55       | 68  |
| 11                        | 89     | 60       | 69  |
| 12                        | 90     | 65       | 70  |
| 13                        | 91     | 70       | 71  |
| 14                        | 92     | 75       | 72  |
| 15                        | 93     | 80       | 73  |
| 16                        | 94     | 85       | 74  |
| 17                        | 95     | 90       | 75  |
| 18                        | 96     | 95       | 76  |
| 19                        | 97     | 100      | 77  |
| 20                        | 98     | 105      | 78  |
</details>

(d) antmaze-umaze-diverse EOP.

![](images/0f977e73f0e8473d8276ebdf46a098e21be29ec7d3cda94398f3099c12bfe737.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 70     | 15       | 20  |
| 2                         | 80     | 20       | 30  |
| 3                         | 85     | 25       | 40  |
| 4                         | 88     | 30       | 50  |
| 5                         | 90     | 35       | 55  |
| 6                         | 92     | 40       | 60  |
| 7                         | 93     | 45       | 62  |
| 8                         | 94     | 50       | 63  |
| 9                         | 95     | 55       | 64  |
| 10                        | 95     | 60       | 65  |
| 11                        | 95     | 62       | 65  |
| 12                        | 95     | 63       | 65  |
| 13                        | 95     | 64       | 65  |
| 14                        | 95     | 65       | 65  |
| 15                        | 95     | 65       | 65  |
| 16                        | 95     | 65       | 65  |
| 17                        | 95     | 65       | 65  |
| 18                        | 95     | 65       | 65  |
| 19                        | 95     | 65       | 65  |
| 20                        | 95     | 65       | 65  |
</details>

(e) antmaze-medium-diverse EOP.

![](images/29cf9a9e00b7d6abe2cea4d51dee477beddda0ff94bb96ab88f920c4502dc941.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 60     | 0        | 10  |
| 2                         | 70     | 0        | 15  |
| 3                         | 75     | 0        | 20  |
| 4                         | 78     | 0        | 22  |
| 5                         | 80     | 0        | 23  |
| 6                         | 81     | 0        | 24  |
| 7                         | 82     | 0        | 25  |
| 8                         | 83     | 0        | 26  |
| 9                         | 84     | 0        | 27  |
| 10                        | 85     | 0        | 28  |
| 11                        | 86     | 0        | 29  |
| 12                        | 87     | 0        | 30  |
| 13                        | 88     | 0        | 31  |
| 14                        | 89     | 0        | 32  |
| 15                        | 90     | 0        | 33  |
| 16                        | 91     | 0        | 34  |
| 17                        | 92     | 0        | 35  |
| 18                        | 93     | 0        | 36  |
| 19                        | 94     | 0        | 37  |
| 20                        | 95     | 0        | 38  |
</details>

(f) antmaze-large-diverse EOP.   
Figure 6: TD3+BC, IQL and ReBRAC visualised Expected Online Performance under uniform policy selection on AntMaze tasks.

Table 29: TD3+BC, IQL and ReBRAC Expected Online Performance under uniform policy selection on Pen tasks. 

<table><tr><td colspan="4">Human</td><td colspan="3">cloned</td><td colspan="3">expert</td></tr><tr><td>Policies</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td></tr><tr><td>1</td><td> $42.9 \pm 32.0$ </td><td> $87.1 \pm 4.1$ </td><td> $69.9 \pm 28.2$ </td><td> $33.6 \pm 23.2$ </td><td> $73.8 \pm 5.8$ </td><td> $65.8 \pm 32.7$ </td><td> $73.9 \pm 65.2$ </td><td> $130.1 \pm 2.8$ </td><td> $136.9 \pm 25.4$ </td></tr><tr><td>2</td><td> $60.4 \pm 25.4$ </td><td> $89.4 \pm 2.9$ </td><td> $85.8 \pm 20.7$ </td><td> $44.9 \pm 25.4$ </td><td> $76.7 \pm 4.8$ </td><td> $83.6 \pm 21.8$ </td><td> $108.9 \pm 52.9$ </td><td> $131.7 \pm 2.1$ </td><td> $149.0 \pm 13.6$ </td></tr><tr><td>3</td><td> $68.7 \pm 18.9$ </td><td> $90.3 \pm 2.5$ </td><td> $92.8 \pm 14.8$ </td><td> $52.6 \pm 25.2$ </td><td> $78.0 \pm 4.9$ </td><td> $90.8 \pm 15.4$ </td><td> $125.6 \pm 38.7$ </td><td> $132.4 \pm 1.8$ </td><td> $152.4 \pm 7.0$ </td></tr><tr><td>4</td><td> $73.0 \pm 14.2$ </td><td> $90.9 \pm 2.2$ </td><td> $96.3 \pm 10.9$ </td><td> $58.1 \pm 24.1$ </td><td> $79.0 \pm 5.2$ </td><td> $94.6 \pm 12.2$ </td><td> $134.1 \pm 27.8$ </td><td> $132.8 \pm 1.6$ </td><td> $153.5 \pm 3.8$ </td></tr><tr><td>5</td><td> $75.6 \pm 11.2$ </td><td> $91.4 \pm 2.0$ </td><td> $98.4 \pm 8.3$ </td><td> $62.3 \pm 22.6$ </td><td> $79.7 \pm 5.4$ </td><td> $97.0 \pm 10.4$ </td><td> $138.5 \pm 20.0$ </td><td> $133.1 \pm 1.5$ </td><td> $154.1 \pm 2.2$ </td></tr><tr><td>6</td><td>-</td><td> $91.7 \pm 1.8$ </td><td> $99.7 \pm 6.5$ </td><td>-</td><td> $80.3 \pm 5.5$ </td><td> $98.7 \pm 9.3$ </td><td>-</td><td> $133.4 \pm 1.4$ </td><td> $154.3 \pm 1.5$ </td></tr><tr><td>7</td><td>-</td><td> $92.0 \pm 1.7$ </td><td> $100.5 \pm 5.2$ </td><td>-</td><td> $80.9 \pm 5.6$ </td><td> $100.0 \pm 8.4$ </td><td>-</td><td> $133.6 \pm 1.3$ </td><td> $154.5 \pm 1.1$ </td></tr><tr><td>8</td><td>-</td><td> $92.2 \pm 1.6$ </td><td> $101.1 \pm 4.4$ </td><td>-</td><td> $81.4 \pm 5.7$ </td><td> $101.1 \pm 7.6$ </td><td>-</td><td> $133.7 \pm 1.2$ </td><td> $154.7 \pm 0.9$ </td></tr><tr><td>9</td><td>-</td><td> $92.3 \pm 1.4$ </td><td> $101.6 \pm 3.7$ </td><td>-</td><td> $81.9 \pm 5.8$ </td><td> $101.9 \pm 7.0$ </td><td>-</td><td> $133.9 \pm 1.1$ </td><td> $154.8 \pm 0.8$ </td></tr><tr><td>10</td><td>-</td><td> $92.5 \pm 1.3$ </td><td> $101.9 \pm 3.3$ </td><td>-</td><td> $82.3 \pm 5.9$ </td><td> $102.6 \pm 6.4$ </td><td>-</td><td> $134.0 \pm 1.0$ </td><td> $154.8 \pm 0.7$ </td></tr><tr><td>11</td><td>-</td><td> $92.6 \pm 1.2$ </td><td> $102.2 \pm 2.9$ </td><td>-</td><td> $82.7 \pm 5.9$ </td><td> $103.1 \pm 5.9$ </td><td>-</td><td> $134.1 \pm 0.9$ </td><td> $154.9 \pm 0.6$ </td></tr><tr><td>12</td><td>-</td><td> $92.7 \pm 1.2$ </td><td> $102.4 \pm 2.6$ </td><td>-</td><td> $83.0 \pm 5.9$ </td><td> $103.6 \pm 5.4$ </td><td>-</td><td> $134.1 \pm 0.9$ </td><td> $154.9 \pm 0.6$ </td></tr><tr><td>13</td><td>-</td><td> $92.8 \pm 1.1$ </td><td> $102.6 \pm 2.4$ </td><td>-</td><td> $83.4 \pm 5.9$ </td><td> $104.0 \pm 5.0$ </td><td>-</td><td> $134.2 \pm 0.8$ </td><td> $155.0 \pm 0.5$ </td></tr><tr><td>14</td><td>-</td><td> $92.9 \pm 1.0$ </td><td> $102.8 \pm 2.2$ </td><td>-</td><td> $83.7 \pm 5.9$ </td><td> $104.4 \pm 4.6$ </td><td>-</td><td> $134.2 \pm 0.7$ </td><td> $155.0 \pm 0.5$ </td></tr><tr><td>15</td><td>-</td><td> $92.9 \pm 1.0$ </td><td> $102.9 \pm 2.0$ </td><td>-</td><td> $84.0 \pm 5.8$ </td><td> $104.7 \pm 4.3$ </td><td>-</td><td> $134.3 \pm 0.7$ </td><td> $155.0 \pm 0.4$ </td></tr><tr><td>16</td><td>-</td><td> $93.0 \pm 0.9$ </td><td> $103.0 \pm 1.8$ </td><td>-</td><td> $84.3 \pm 5.8$ </td><td> $104.9 \pm 4.0$ </td><td>-</td><td> $134.3 \pm 0.6$ </td><td> $155.1 \pm 0.4$ </td></tr><tr><td>17</td><td>-</td><td> $93.0 \pm 0.9$ </td><td> $103.1 \pm 1.7$ </td><td>-</td><td> $84.6 \pm 5.7$ </td><td> $105.1 \pm 3.8$ </td><td>-</td><td> $134.4 \pm 0.6$ </td><td> $155.1 \pm 0.4$ </td></tr><tr><td>18</td><td>-</td><td> $93.1 \pm 0.8$ </td><td> $103.2 \pm 1.6$ </td><td>-</td><td> $84.8 \pm 5.7$ </td><td> $105.3 \pm 3.5$ </td><td>-</td><td> $134.4 \pm 0.6$ </td><td> $155.1 \pm 0.4$ </td></tr><tr><td>19</td><td>-</td><td> $93.1 \pm 0.8$ </td><td> $103.3 \pm 1.4$ </td><td>-</td><td> $85.0 \pm 5.6$ </td><td> $105.5 \pm 3.3$ </td><td>-</td><td> $134.4 \pm 0.5$ </td><td> $155.1 \pm 0.4$ </td></tr><tr><td>20</td><td>-</td><td> $93.2 \pm 0.8$ </td><td> $103.3 \pm 1.3$ </td><td>-</td><td> $85.3 \pm 5.5$ </td><td> $105.7 \pm 3.1$ </td><td>-</td><td> $134.5 \pm 0.5$ </td><td> $155.1 \pm 0.4$ </td></tr></table>

![](images/44936c9e04df45abb6ad5e563cbee3e53563bc3e7d002bbd9a517dcf74e0eef0.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| -------------------------- | ------ | -------- | --- |
| 1                          | 70     | 40       | 85  |
| 2                          | 85     | 60       | 88  |
| 3                          | 90     | 70       | 90  |
| 4                          | 95     | 75       | 92  |
| 5                          | 98     | 80       | 93  |
| 6                          | 99     | 85       | 94  |
| 7                          | 100    | 90       | 95  |
| 8                          | 100    | 92       | 95  |
| 9                          | 100    | 93       | 95  |
| 10                         | 100    | 94       | 95  |
| 11                         | 100    | 95       | 95  |
| 12                         | 100    | 95       | 95  |
| 13                         | 100    | 95       | 95  |
| 14                         | 100    | 95       | 95  |
| 15                         | 100    | 95       | 95  |
| 16                         | 100    | 95       | 95  |
| 17                         | 100    | 95       | 95  |
| 18                         | 100    | 95       | 95  |
| 19                         | 100    | 95       | 95  |
| 20                         | 100    | 95       | 95  |
</details>

(a) pen-human EOP.

![](images/d527071adcee7591210774c78db7d4d8dcde60f9ebb2a1e2622de705ca953e17.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD1 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 65     | 35       | 75  |
| 2                         | 80     | 45       | 80  |
| 3                         | 90     | 55       | 82  |
| 4                         | 95     | 60       | 83  |
| 5                         | 98     | 65       | 84  |
| 6                         | 99     | 70       | 85  |
| 7                         | 100    | 72       | 86  |
| 8                         | 100    | 75       | 87  |
| 9                         | 100    | 78       | 88  |
| 10                        | 100    | 80       | 89  |
| 11                        | 100    | 82       | 90  |
| 12                        | 100    | 83       | 91  |
| 13                        | 100    | 84       | 92  |
| 14                        | 100    | 85       | 93  |
| 15                        | 100    | 86       | 94  |
| 16                        | 100    | 87       | 95  |
| 17                        | 100    | 88       | 96  |
| 18                        | 100    | 89       | 97  |
| 19                        | 100    | 90       | 98  |
| 20                        | 100    | 91       | 99  |
</details>

(b) pen-cloned EOP.

![](images/cea489297a234a19cdf6d11726b00f34ee322055562a543db64f6da45908699c.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 140    | 80       | 130 |
| 2                         | 145    | 100      | 135 |
| 3                         | 148    | 120      | 138 |
| 4                         | 150    | 130      | 140 |
| 5                         | 152    | 135      | 142 |
| 6                         | 153    | 138      | 143 |
| 7                         | 154    | 140      | 144 |
| 8                         | 155    | 142      | 145 |
| 9                         | 156    | 143      | 146 |
| 10                        | 157    | 144      | 147 |
| 11                        | 158    | 145      | 148 |
| 12                        | 159    | 146      | 149 |
| 13                        | 160    | 147      | 150 |
| 14                        | 160    | 148      | 150 |
| 15                        | 160    | 149      | 150 |
| 16                        | 160    | 150      | 150 |
| 17                        | 160    | 150      | 150 |
| 18                        | 160    | 150      | 150 |
| 19                        | 160    | 150      | 150 |
| 20                        | 160    | 150      | 150 |
</details>

(c) pen-expert EOP.   
Figure 7: TD3+BC, IQL and ReBRAC visualised Expected Online Performance under uniform policy selection on Pen tasks.

Table 30: TD3+BC, IQL and ReBRAC Expected Online Performance under uniform policy selection on Door tasks. 

<table><tr><td colspan="4">human</td><td colspan="3">cloned</td><td colspan="3">expert</td></tr><tr><td>Policies</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td></tr><tr><td>1</td><td>-0.2 ± 0.1</td><td>4.4 ± 1.2</td><td>-0.1 ± 0.1</td><td>-0.1 ± 0.3</td><td>1.6 ± 0.8</td><td>0.3 ± 0.9</td><td>50.7 ± 46.3</td><td>102.1 ± 5.7</td><td>75.4 ± 43.0</td></tr><tr><td>2</td><td>-0.1 ± 0.1</td><td>5.0 ± 1.0</td><td>-0.0 ± 0.1</td><td>0.0 ± 0.3</td><td>2.0 ± 0.6</td><td>0.6 ± 1.2</td><td>75.6 ± 39.5</td><td>104.7 ± 2.5</td><td>96.1 ± 25.0</td></tr><tr><td>3</td><td>-0.1 ± 0.1</td><td>5.4 ± 1.0</td><td>-0.0 ± 0.0</td><td>0.1 ± 0.3</td><td>2.2 ± 0.5</td><td>0.8 ± 1.4</td><td>88.2 ± 30.4</td><td>105.4 ± 1.3</td><td>102.3 ± 13.5</td></tr><tr><td>4</td><td>-0.1 ± 0.1</td><td>5.6 ± 0.9</td><td>-0.0 ± 0.0</td><td>0.2 ± 0.3</td><td>2.4 ± 0.4</td><td>1.0 ± 1.6</td><td>94.9 ± 22.9</td><td>105.7 ± 0.8</td><td>104.4 ± 7.3</td></tr><tr><td>5</td><td>-0.1 ± 0.1</td><td>5.8 ± 0.9</td><td>-0.0 ± 0.0</td><td>0.2 ± 0.3</td><td>2.5 ± 0.4</td><td>1.2 ± 1.7</td><td>98.6 ± 17.2</td><td>105.8 ± 0.6</td><td>105.2 ± 4.1</td></tr><tr><td>6</td><td>-</td><td>5.9 ± 0.9</td><td>0.0 ± 0.0</td><td>-</td><td>2.5 ± 0.3</td><td>1.3 ± 1.8</td><td>-</td><td>105.9 ± 0.5</td><td>105.6 ± 2.4</td></tr><tr><td>7</td><td>-</td><td>6.0 ± 0.9</td><td>0.0 ± 0.0</td><td>-</td><td>2.6 ± 0.3</td><td>1.5 ± 1.8</td><td>-</td><td>106.0 ± 0.5</td><td>105.8 ± 1.5</td></tr><tr><td>8</td><td>-</td><td>6.1 ± 0.8</td><td>0.0 ± 0.0</td><td>-</td><td>2.6 ± 0.3</td><td>1.6 ± 1.9</td><td>-</td><td>106.1 ± 0.4</td><td>105.9 ± 1.0</td></tr><tr><td>9</td><td>-</td><td>6.2 ± 0.8</td><td>0.0 ± 0.0</td><td>-</td><td>2.6 ± 0.2</td><td>1.8 ± 1.9</td><td>-</td><td>106.1 ± 0.4</td><td>105.9 ± 0.7</td></tr><tr><td>10</td><td>-</td><td>6.3 ± 0.8</td><td>0.0 ± 0.0</td><td>-</td><td>2.7 ± 0.2</td><td>1.9 ± 1.9</td><td>-</td><td>106.1 ± 0.4</td><td>106.0 ± 0.5</td></tr><tr><td>11</td><td>-</td><td>6.4 ± 0.8</td><td>0.0 ± 0.0</td><td>-</td><td>2.7 ± 0.2</td><td>2.0 ± 2.0</td><td>-</td><td>106.2 ± 0.4</td><td>106.0 ± 0.4</td></tr><tr><td>12</td><td>-</td><td>6.4 ± 0.7</td><td>0.0 ± 0.0</td><td>-</td><td>2.7 ± 0.2</td><td>2.2 ± 2.0</td><td>-</td><td>106.2 ± 0.3</td><td>106.0 ± 0.3</td></tr><tr><td>13</td><td>-</td><td>6.5 ± 0.7</td><td>0.0 ± 0.0</td><td>-</td><td>2.7 ± 0.2</td><td>2.3 ± 2.0</td><td>-</td><td>106.2 ± 0.3</td><td>106.0 ± 0.2</td></tr><tr><td>14</td><td>-</td><td>6.5 ± 0.7</td><td>0.0 ± 0.0</td><td>-</td><td>2.7 ± 0.2</td><td>2.4 ± 2.0</td><td>-</td><td>106.2 ± 0.3</td><td>106.0 ± 0.2</td></tr><tr><td>15</td><td>-</td><td>6.6 ± 0.7</td><td>0.0 ± 0.0</td><td>-</td><td>2.7 ± 0.2</td><td>2.5 ± 2.0</td><td>-</td><td>106.3 ± 0.3</td><td>106.0 ± 0.2</td></tr><tr><td>16</td><td>-</td><td>6.6 ± 0.6</td><td>0.0 ± 0.0</td><td>-</td><td>2.7 ± 0.2</td><td>2.6 ± 1.9</td><td>-</td><td>106.3 ± 0.3</td><td>106.1 ± 0.2</td></tr><tr><td>17</td><td>-</td><td>6.6 ± 0.6</td><td>0.0 ± 0.0</td><td>-</td><td>2.8 ± 0.2</td><td>2.7 ± 1.9</td><td>-</td><td>106.3 ± 0.3</td><td>106.1 ± 0.2</td></tr><tr><td>18</td><td>-</td><td>6.7 ± 0.6</td><td>0.0 ± 0.0</td><td>-</td><td>2.8 ± 0.2</td><td>2.7 ± 1.9</td><td>-</td><td>106.3 ± 0.3</td><td>106.1 ± 0.1</td></tr><tr><td>19</td><td>-</td><td>6.7 ± 0.6</td><td>0.0 ± 0.0</td><td>-</td><td>2.8 ± 0.2</td><td>2.8 ± 1.9</td><td>-</td><td>106.3 ± 0.3</td><td>106.1 ± 0.1</td></tr><tr><td>20</td><td>-</td><td>6.7 ± 0.5</td><td>0.0 ± 0.0</td><td>-</td><td>2.8 ± 0.2</td><td>2.9 ± 1.9</td><td>-</td><td>106.3 ± 0.2</td><td>106.1 ± 0.1</td></tr></table>

![](images/e269396bb2977889a5e66a8eb5ece9d1c33cd77980c3b7e6b932d0475cff40eb.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 0.0    | 0.0      | 4.5 |
| 2                         | 0.0    | 0.0      | 5.0 |
| 3                         | 0.0    | 0.0      | 5.5 |
| 4                         | 0.0    | 0.0      | 6.0 |
| 5                         | 0.0    | 0.0      | 6.2 |
| 6                         | 0.0    | 0.0      | 6.3 |
| 7                         | 0.0    | 0.0      | 6.4 |
| 8                         | 0.0    | 0.0      | 6.5 |
| 9                         | 0.0    | 0.0      | 6.6 |
| 10                        | 0.0    | 0.0      | 6.7 |
| 11                        | 0.0    | 0.0      | 6.8 |
| 12                        | 0.0    | 0.0      | 6.9 |
| 13                        | 0.0    | 0.0      | 7.0 |
| 14                        | 0.0    | 0.0      | 7.1 |
| 15                        | 0.0    | 0.0      | 7.2 |
| 16                        | 0.0    | 0.0      | 7.3 |
| 17                        | 0.0    | 0.0      | 7.4 |
| 18                        | 0.0    | 0.0      | 7.5 |
| 19                        | 0.0    | 0.0      | 7.6 |
| 20                        | 0.0    | 0.0      | 7.7 |
</details>

(a) door-human EOP.

![](images/a2d843c3eb3934d559cf59437732201d1177bf2f0d4571c4711cc7e553a25537.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 0.5    | 0.2      | 1.8 |
| 2                         | 0.7    | 0.3      | 2.2 |
| 3                         | 0.9    | 0.4      | 2.5 |
| 4                         | 1.1    | 0.5      | 2.7 |
| 5                         | 1.3    | 0.6      | 2.8 |
| 6                         | 1.5    | 0.7      | 2.9 |
| 7                         | 1.7    | 0.8      | 2.9 |
| 8                         | 1.9    | 0.9      | 2.9 |
| 9                         | 2.1    | 1.0      | 2.9 |
| 10                        | 2.3    | 1.1      | 2.9 |
| 11                        | 2.5    | 1.2      | 2.9 |
| 12                        | 2.7    | 1.3      | 2.9 |
| 13                        | 2.9    | 1.4      | 2.9 |
| 14                        | 3.0    | 1.5      | 2.9 |
| 15                        | 3.1    | 1.6      | 2.9 |
| 16                        | 3.2    | 1.7      | 2.9 |
| 17                        | 3.3    | 1.8      | 2.9 |
| 18                        | 3.4    | 1.9      | 2.9 |
| 19                        | 3.5    | 2.0      | 2.9 |
| 20                        | 3.6    | 2.1      | 2.9 |
</details>

(b) door-cloned EOP.

![](images/0df8e698c22822f71948df916769a10ddbbb934933ed3d93c90d77a7c504e4eb.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 80     | 60       | 100 |
| 2                         | 100    | 80       | 100 |
| 3                         | 105    | 90       | 100 |
| 4                         | 105    | 95       | 100 |
| 5                         | 105    | 100      | 100 |
| 6                         | 105    | 100      | 100 |
| 7                         | 105    | 100      | 100 |
| 8                         | 105    | 100      | 100 |
| 9                         | 105    | 100      | 100 |
| 10                        | 105    | 100      | 100 |
| 11                        | 105    | 100      | 100 |
| 12                        | 105    | 100      | 100 |
| 13                        | 105    | 100      | 100 |
| 14                        | 105    | 100      | 100 |
| 15                        | 105    | 100      | 100 |
| 16                        | 105    | 100      | 100 |
| 17                        | 105    | 100      | 100 |
| 18                        | 105    | 100      | 100 |
| 19                        | 105    | 100      | 100 |
| 20                        | 105    | 100      | 100 |
</details>

(c) door-expert EOP.   
Figure 8: TD3+BC, IQL and ReBRAC visualised Expected Online Performance under uniform policy selection on Door tasks.

Table 31: TD3+BC, IQL and ReBRAC Expected Online Performance under uniform policy selection on Hammer tasks. 

<table><tr><td colspan="4">human</td><td colspan="3">cloned</td><td colspan="3">expert</td></tr><tr><td>Policies</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td></tr><tr><td>1</td><td>0.2 ± 0.0</td><td>1.6 ± 0.4</td><td>0.3 ± 0.2</td><td>1.1 ± 0.8</td><td>1.9 ± 1.1</td><td>4.5 ± 5.5</td><td>60.2 ± 55.4</td><td>128.7 ± 1.1</td><td>101.9 ± 49.9</td></tr><tr><td>2</td><td>0.3 ± 0.0</td><td>1.8 ± 0.4</td><td>0.3 ± 0.2</td><td>1.6 ± 0.8</td><td>2.5 ± 1.1</td><td>7.3 ± 6.1</td><td>90.0 ± 48.4</td><td>129.3 ± 0.9</td><td>124.3 ± 26.8</td></tr><tr><td>3</td><td>0.3 ± 0.0</td><td>1.9 ± 0.4</td><td>0.4 ± 0.2</td><td>1.8 ± 0.7</td><td>2.8 ± 1.0</td><td>9.2 ± 6.2</td><td>105.7 ± 38.1</td><td>129.6 ± 0.7</td><td>130.1 ± 13.6</td></tr><tr><td>4</td><td>0.3 ± 0.0</td><td>2.0 ± 0.4</td><td>0.4 ± 0.2</td><td>2.0 ± 0.6</td><td>3.1 ± 0.9</td><td>10.6 ± 6.2</td><td>114.3 ± 29.3</td><td>129.7 ± 0.5</td><td>131.9 ± 7.0</td></tr><tr><td>5</td><td>0.3 ± 0.0</td><td>2.1 ± 0.4</td><td>0.5 ± 0.2</td><td>2.1 ± 0.6</td><td>3.3 ± 0.9</td><td>11.7 ± 6.2</td><td>119.2 ± 22.4</td><td>129.8 ± 0.4</td><td>132.6 ± 3.8</td></tr><tr><td>6</td><td>-</td><td>2.1 ± 0.4</td><td>0.5 ± 0.2</td><td>-</td><td>3.4 ± 0.8</td><td>12.6 ± 6.1</td><td>-</td><td>129.9 ± 0.3</td><td>133.0 ± 2.4</td></tr><tr><td>7</td><td>-</td><td>2.2 ± 0.4</td><td>0.5 ± 0.2</td><td>-</td><td>3.5 ± 0.7</td><td>13.3 ± 6.0</td><td>-</td><td>129.9 ± 0.2</td><td>133.2 ± 1.7</td></tr><tr><td>8</td><td>-</td><td>2.2 ± 0.4</td><td>0.5 ± 0.2</td><td>-</td><td>3.6 ± 0.7</td><td>14.0 ± 5.9</td><td>-</td><td>129.9 ± 0.2</td><td>133.4 ± 1.4</td></tr><tr><td>9</td><td>-</td><td>2.3 ± 0.3</td><td>0.6 ± 0.2</td><td>-</td><td>3.7 ± 0.6</td><td>14.6 ± 5.8</td><td>-</td><td>130.0 ± 0.1</td><td>133.5 ± 1.2</td></tr><tr><td>10</td><td>-</td><td>2.3 ± 0.3</td><td>0.6 ± 0.2</td><td>-</td><td>3.7 ± 0.6</td><td>15.1 ± 5.7</td><td>-</td><td>130.0 ± 0.1</td><td>133.6 ± 1.1</td></tr><tr><td>11</td><td>-</td><td>2.3 ± 0.3</td><td>0.6 ± 0.1</td><td>-</td><td>3.8 ± 0.5</td><td>15.5 ± 5.6</td><td>-</td><td>130.0 ± 0.1</td><td>133.7 ± 1.0</td></tr><tr><td>12</td><td>-</td><td>2.3 ± 0.3</td><td>0.6 ± 0.1</td><td>-</td><td>3.8 ± 0.5</td><td>16.0 ± 5.4</td><td>-</td><td>130.0 ± 0.1</td><td>133.8 ± 0.9</td></tr><tr><td>13</td><td>-</td><td>2.4 ± 0.3</td><td>0.6 ± 0.1</td><td>-</td><td>3.9 ± 0.5</td><td>16.3 ± 5.3</td><td>-</td><td>130.0 ± 0.1</td><td>133.9 ± 0.8</td></tr><tr><td>14</td><td>-</td><td>2.4 ± 0.3</td><td>0.6 ± 0.1</td><td>-</td><td>3.9 ± 0.4</td><td>16.7 ± 5.2</td><td>-</td><td>130.0 ± 0.1</td><td>133.9 ± 0.7</td></tr><tr><td>15</td><td>-</td><td>2.4 ± 0.3</td><td>0.6 ± 0.1</td><td>-</td><td>3.9 ± 0.4</td><td>17.0 ± 5.1</td><td>-</td><td>130.0 ± 0.1</td><td>134.0 ± 0.6</td></tr><tr><td>16</td><td>-</td><td>2.4 ± 0.3</td><td>0.6 ± 0.1</td><td>-</td><td>4.0 ± 0.4</td><td>17.3 ± 4.9</td><td>-</td><td>130.0 ± 0.1</td><td>134.0 ± 0.6</td></tr><tr><td>17</td><td>-</td><td>2.4 ± 0.2</td><td>0.6 ± 0.1</td><td>-</td><td>4.0 ± 0.4</td><td>17.5 ± 4.8</td><td>-</td><td>130.0 ± 0.0</td><td>134.0 ± 0.5</td></tr><tr><td>18</td><td>-</td><td>2.4 ± 0.2</td><td>0.6 ± 0.1</td><td>-</td><td>4.0 ± 0.3</td><td>17.8 ± 4.7</td><td>-</td><td>130.0 ± 0.0</td><td>134.0 ± 0.5</td></tr><tr><td>19</td><td>-</td><td>2.5 ± 0.2</td><td>0.6 ± 0.1</td><td>-</td><td>4.0 ± 0.3</td><td>18.0 ± 4.6</td><td>-</td><td>130.0 ± 0.0</td><td>134.1 ± 0.4</td></tr><tr><td>20</td><td>-</td><td>2.5 ± 0.2</td><td>0.7 ± 0.1</td><td>-</td><td>4.0 ± 0.3</td><td>18.2 ± 4.5</td><td>-</td><td>130.0 ± 0.0</td><td>134.1 ± 0.4</td></tr></table>

![](images/5c88eb976f384ef43723f9a645c6e0e622fb0970b5aaa861da9df0d2d5788a18.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 0.3    | 0.2      | 1.5 |
| 2                         | 0.4    | 0.2      | 1.7 |
| 3                         | 0.45   | 0.2      | 1.8 |
| 4                         | 0.5    | 0.2      | 1.9 |
| 5                         | 0.55   | 0.2      | 2.0 |
| 6                         | 0.6    | 0.2      | 2.1 |
| 7                         | 0.65   | 0.2      | 2.2 |
| 8                         | 0.7    | 0.2      | 2.3 |
| 9                         | 0.75   | 0.2      | 2.4 |
| 10                        | 0.8    | 0.2      | 2.5 |
| 11                        | 0.85   | 0.2      | 2.55 |
| 12                        | 0.9    | 0.2      | 2.6 |
| 13                        | 0.95   | 0.2      | 2.65 |
| 14                        | 1.0    | 0.2      | 2.7 |
| 15                        | 1.05   | 0.2      | 2.75 |
| 16                        | 1.1    | 0.2      | 2.8 |
| 17                        | 1.15   | 0.2      | 2.85 |
| 18                        | 1.2    | 0.2      | 2.9 |
| 19                        | 1.25   | 0.2      | 2.95 |
| 20                        | 1.3    | 0.2      | 3.0 |
</details>

(a) hammer-human EOP.

![](images/cea6e094530802021722d7f69050ae243ee46138b68acb3ad4b44ef1b8819ba7.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 5      | 2        | 2   |
| 2                         | 7      | 2        | 2   |
| 3                         | 9      | 2        | 2   |
| 4                         | 11     | 2        | 2   |
| 5                         | 13     | 2        | 2   |
| 6                         | 15     | 2        | 2   |
| 7                         | 16     | 2        | 2   |
| 8                         | 17     | 2        | 2   |
| 9                         | 18     | 2        | 2   |
| 10                        | 19     | 2        | 2   |
| 11                        | 20     | 2        | 2   |
| 12                        | 20     | 2        | 2   |
| 13                        | 20     | 2        | 2   |
| 14                        | 20     | 2        | 2   |
| 15                        | 20     | 2        | 2   |
| 16                        | 20     | 2        | 2   |
| 17                        | 20     | 2        | 2   |
| 18                        | 20     | 2        | 2   |
| 19                        | 20     | 2        | 2   |
| 20                        | 20     | 2        | 2   |
</details>

(b) hammer-cloned EOP.

![](images/2196c6a47d1a523fb73a086c4963e2283b613f3e4ccf78be4018e21a574031a9.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 100    | 60       | 120 |
| 2                         | 120    | 80       | 120 |
| 3                         | 120    | 100      | 120 |
| 4                         | 120    | 110      | 120 |
| 5                         | 120    | 115      | 120 |
| 6                         | 120    | 120      | 120 |
| 7                         | 120    | 120      | 120 |
| 8                         | 120    | 120      | 120 |
| 9                         | 120    | 120      | 120 |
| 10                        | 120    | 120      | 120 |
| 11                        | 120    | 120      | 120 |
| 12                        | 120    | 120      | 120 |
| 13                        | 120    | 120      | 120 |
| 14                        | 120    | 120      | 120 |
| 15                        | 120    | 120      | 120 |
| 16                        | 120    | 120      | 120 |
| 17                        | 120    | 120      | 120 |
| 18                        | 120    | 120      | 120 |
| 19                        | 120    | 120      | 120 |
| 20                        | 120    | 120      | 120 |
</details>

(c) hammer-expert EOP.   
Figure 9: TD3+BC, IQL and ReBRAC visualised Expected Online Performance under uniform policy selection on Hammer tasks.

Table 32: TD3+BC, IQL and ReBRAC Expected Online Performance under uniform policy selection on tasks. 

<table><tr><td colspan="4">human</td><td colspan="3">cloned</td><td colspan="3">expert</td></tr><tr><td>Policies</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td><td>TD3+BC</td><td>IQL</td><td>ReBRAC</td></tr><tr><td>1</td><td>-0.2 ± 0.1</td><td>0.2 ± 0.2</td><td>-0.1 ± 0.1</td><td>-0.2 ± 0.1</td><td>-0.0 ± 0.1</td><td>0.5 ± 0.8</td><td>21.4 ± 43.2</td><td>106.0 ± 1.4</td><td>73.5 ± 44.3</td></tr><tr><td>2</td><td>-0.2 ± 0.1</td><td>0.2 ± 0.2</td><td>-0.1 ± 0.1</td><td>-0.2 ± 0.1</td><td>0.0 ± 0.1</td><td>0.9 ± 0.9</td><td>38.8 ± 51.9</td><td>106.8 ± 1.0</td><td>96.0 ± 27.4</td></tr><tr><td>3</td><td>-0.1 ± 0.1</td><td>0.3 ± 0.2</td><td>-0.0 ± 0.0</td><td>-0.2 ± 0.1</td><td>0.1 ± 0.1</td><td>1.2 ± 0.9</td><td>52.6 ± 54.0</td><td>107.2 ± 0.8</td><td>103.7 ± 15.9</td></tr><tr><td>4</td><td>-0.1 ± 0.1</td><td>0.3 ± 0.2</td><td>-0.0 ± 0.0</td><td>-0.1 ± 0.1</td><td>0.1 ± 0.1</td><td>1.4 ± 0.9</td><td>63.7 ± 53.1</td><td>107.4 ± 0.7</td><td>106.7 ± 9.4</td></tr><tr><td>5</td><td>-0.1 ± 0.1</td><td>0.4 ± 0.2</td><td>-0.0 ± 0.0</td><td>-0.1 ± 0.0</td><td>0.1 ± 0.1</td><td>1.6 ± 0.9</td><td>72.5 ± 50.7</td><td>107.5 ± 0.6</td><td>107.9 ± 5.8</td></tr><tr><td>6</td><td>-</td><td>0.4 ± 0.2</td><td>-0.0 ± 0.0</td><td>-</td><td>0.1 ± 0.1</td><td>1.7 ± 0.8</td><td>-</td><td>107.6 ± 0.5</td><td>108.6 ± 3.8</td></tr><tr><td>7</td><td>-</td><td>0.4 ± 0.2</td><td>-0.0 ± 0.0</td><td>-</td><td>0.1 ± 0.1</td><td>1.8 ± 0.8</td><td>-</td><td>107.7 ± 0.5</td><td>108.9 ± 2.6</td></tr><tr><td>8</td><td>-</td><td>0.4 ± 0.2</td><td>0.0 ± 0.0</td><td>-</td><td>0.1 ± 0.1</td><td>1.9 ± 0.8</td><td>-</td><td>107.7 ± 0.5</td><td>109.2 ± 1.9</td></tr><tr><td>9</td><td>-</td><td>0.5 ± 0.2</td><td>0.0 ± 0.0</td><td>-</td><td>0.1 ± 0.1</td><td>2.0 ± 0.7</td><td>-</td><td>107.8 ± 0.4</td><td>109.3 ± 1.5</td></tr><tr><td>10</td><td>-</td><td>0.5 ± 0.2</td><td>0.0 ± 0.0</td><td>-</td><td>0.1 ± 0.1</td><td>2.1 ± 0.7</td><td>-</td><td>107.8 ± 0.4</td><td>109.4 ± 1.3</td></tr><tr><td>11</td><td>-</td><td>0.5 ± 0.2</td><td>0.0 ± 0.0</td><td>-</td><td>0.2 ± 0.1</td><td>2.1 ± 0.7</td><td>-</td><td>107.8 ± 0.4</td><td>109.5 ± 1.1</td></tr><tr><td>12</td><td>-</td><td>0.5 ± 0.2</td><td>0.0 ± 0.0</td><td>-</td><td>0.2 ± 0.1</td><td>2.2 ± 0.6</td><td>-</td><td>107.9 ± 0.4</td><td>109.6 ± 1.0</td></tr><tr><td>13</td><td>-</td><td>0.5 ± 0.2</td><td>0.0 ± 0.0</td><td>-</td><td>0.2 ± 0.1</td><td>2.2 ± 0.6</td><td>-</td><td>107.9 ± 0.4</td><td>109.7 ± 0.9</td></tr><tr><td>14</td><td>-</td><td>0.5 ± 0.2</td><td>0.0 ± 0.0</td><td>-</td><td>0.2 ± 0.1</td><td>2.3 ± 0.6</td><td>-</td><td>107.9 ± 0.4</td><td>109.8 ± 0.9</td></tr><tr><td>15</td><td>-</td><td>0.5 ± 0.2</td><td>0.0 ± 0.0</td><td>-</td><td>0.2 ± 0.1</td><td>2.3 ± 0.6</td><td>-</td><td>107.9 ± 0.4</td><td>109.8 ± 0.8</td></tr><tr><td>16</td><td>-</td><td>0.5 ± 0.2</td><td>0.0 ± 0.0</td><td>-</td><td>0.2 ± 0.1</td><td>2.3 ± 0.5</td><td>-</td><td>108.0 ± 0.4</td><td>109.9 ± 0.8</td></tr><tr><td>17</td><td>-</td><td>0.6 ± 0.2</td><td>0.0 ± 0.0</td><td>-</td><td>0.2 ± 0.1</td><td>2.4 ± 0.5</td><td>-</td><td>108.0 ± 0.4</td><td>109.9 ± 0.8</td></tr><tr><td>18</td><td>-</td><td>0.6 ± 0.2</td><td>0.0 ± 0.0</td><td>-</td><td>0.2 ± 0.1</td><td>2.4 ± 0.5</td><td>-</td><td>108.0 ± 0.3</td><td>109.9 ± 0.8</td></tr><tr><td>19</td><td>-</td><td>0.6 ± 0.2</td><td>0.0 ± 0.0</td><td>-</td><td>0.2 ± 0.1</td><td>2.4 ± 0.5</td><td>-</td><td>108.0 ± 0.3</td><td>110.0 ± 0.7</td></tr><tr><td>20</td><td>-</td><td>0.6 ± 0.1</td><td>0.0 ± 0.0</td><td>-</td><td>0.2 ± 0.1</td><td>2.5 ± 0.4</td><td>-</td><td>108.0 ± 0.3</td><td>110.0 ± 0.7</td></tr></table>

![](images/e0029bf419cfa838f7ef685de74c80b8a8024c1b18fefe4e068b6c474c17fa1e.jpg)  
(a) relocate-human EOP.

![](images/085890e57e63ba20cfb4357dd7164a164200ecdd4983a71ad88221fb0b6393c7.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 0.5    | -0.1     | 0.1 |
| 2                         | 1.0    | -0.1     | 0.1 |
| 3                         | 1.5    | -0.1     | 0.1 |
| 4                         | 1.8    | -0.1     | 0.1 |
| 5                         | 2.0    | -0.1     | 0.1 |
| 6                         | 2.1    | -0.1     | 0.1 |
| 7                         | 2.2    | -0.1     | 0.1 |
| 8                         | 2.3    | -0.1     | 0.1 |
| 9                         | 2.4    | -0.1     | 0.1 |
| 10                        | 2.5    | -0.1     | 0.1 |
| 11                        | 2.6    | -0.1     | 0.1 |
| 12                        | 2.7    | -0.1     | 0.1 |
| 13                        | 2.8    | -0.1     | 0.1 |
| 14                        | 2.9    | -0.1     | 0.1 |
| 15                        | 3.0    | -0.1     | 0.1 |
| 16                        | 3.1    | -0.1     | 0.1 |
| 17                        | 3.2    | -0.1     | 0.1 |
| 18                        | 3.3    | -0.1     | 0.1 |
| 19                        | 3.4    | -0.1     | 0.1 |
| 20                        | 3.5    | -0.1     | 0.1 |
</details>

(b) relocate-cloned EOP.

![](images/7134fc1d00aeaa018aab3e363b7473f110b3b6637c0ba4b76ff4384b11af337b.jpg)

<details>
<summary>line</summary>

| Policies evaluated online | ReBRAC | TD3 + BC | IQL |
| ------------------------- | ------ | -------- | --- |
| 1                         | 80     | 20       | 100 |
| 2                         | 100    | 40       | 100 |
| 3                         | 100    | 60       | 100 |
| 4                         | 100    | 70       | 100 |
| 5                         | 100    | 75       | 100 |
| 6                         | 100    | 80       | 100 |
| 7                         | 100    | 85       | 100 |
| 8                         | 100    | 90       | 100 |
| 9                         | 100    | 95       | 100 |
| 10                        | 100    | 100      | 100 |
| 11                        | 100    | 100      | 100 |
| 12                        | 100    | 100      | 100 |
| 13                        | 100    | 100      | 100 |
| 14                        | 100    | 100      | 100 |
| 15                        | 100    | 100      | 100 |
| 16                        | 100    | 100      | 100 |
| 17                        | 100    | 100      | 100 |
| 18                        | 100    | 100      | 100 |
| 19                        | 100    | 100      | 100 |
| 20                        | 100    | 100      | 100 |
</details>

(c) relocate-expert EOP.   
Figure 10: TD3+BC, IQL and ReBRAC visualised Expected Online Performance under uniform policy selection on Relocate tasks.

# G D4RL tasks ablation

Table 33: ReBRAC ablations for halfcheetah tasks. We report final normalized score averaged over 4 unseen training seeds. 

<table><tr><td>Ablation</td><td>random</td><td>medium</td><td>expert</td><td>medium-expert</td><td>medium-replay</td><td>full-replay</td><td>Average</td></tr><tr><td>TD3+BC, paper</td><td>11.0 ± 1.1</td><td>48.3 ± 0.3</td><td>96.7 ± 1.1</td><td>90.7 ± 4.3</td><td>44.6 ± 0.5</td><td>-</td><td>-</td></tr><tr><td>TD3+BC, our</td><td>2.2 ± 0.0</td><td>44.6 ± 0.4</td><td>93.8 ± 0.1</td><td>91.9 ± 2.3</td><td>40.5 ± 1.6</td><td>69.3 ± 0.7</td><td>57.0</td></tr><tr><td>TD3+BC, tuned</td><td>30.1 ± 1.4 (+0.7%)</td><td>55.4 ± 0.9 (-15.2%)</td><td>95.5 ± 0.5 (-9.6%)</td><td>91.9 ± 2.3 (-11.4%)</td><td>45.1 ± 1.7 (-11.0%)</td><td>74.1 ± 2.9 (-9.5%)</td><td>65.3 (-10.3%)</td></tr><tr><td>ReBRAC w/o LN</td><td>32.0 ± 1.5 (+7.0%)</td><td>64.3 ± 4.1 (-1.5%)</td><td>61.7 ± 20.4 (-41.5%)</td><td>86.7 ± 0.9 (-16.3%)</td><td>52.8 ± 2.4 (+4.1%)</td><td>82.1 ± 2.1 (+0.2%)</td><td>63.2 (-13.1%)</td></tr><tr><td>ReBRAC w/o layer</td><td>27.8 ± 3.4 (-7.0%)</td><td>65.0 ± 1.6 (-0.4%)</td><td>74.4 ± 26.7 (-29.5%)</td><td>86.7 ± 8.8 (-16.3%)</td><td>50.4 ± 0.7 (-0.5%)</td><td>80.9 ± 1.1 (-1.2%)</td><td>64.1 (-11.9%)</td></tr><tr><td>ReBRAC w/o actor penalty</td><td>31.8 ± 4.1 (+6.3%)</td><td>64.5 ± 0.7 (-1.2%)</td><td>4.3 ± 4.3 (-95.9%)</td><td>71.6 ± 12.3 (-30.9%)</td><td>38.0 ± 27.2 (-25.0%)</td><td>59.3 ± 41.3 (-27.5%)</td><td>44.9 (-38.3%)</td></tr><tr><td>ReBRAC w/o critic penalty</td><td>28.1 ± 1.6 (-6.0%)</td><td>65.7 ± 1.4 (+0.6%)</td><td>104.2 ± 5.9 (-1.3%)</td><td>100.5 ± 3.1 (-3.0%)</td><td>50.7 ± 0.1 (0.0%)</td><td>81.7 ± 1.2 (-0.2%)</td><td>71.8 (-1.3%)</td></tr><tr><td>ReBRAC w/o large batch</td><td>21.0 ± 15.7 (-29.7%)</td><td>65.8 ± 0.7 (+0.7%)</td><td>62.6 ± 24.3 (-40.7%)</td><td>85.2 ± 7.3 (-17.8%)</td><td>50.7 ± 1.1 (0.0%)</td><td>81.9 ± 1.4 (0.0%)</td><td>61.2 (-15.9%)</td></tr><tr><td>ReBRAC</td><td>29.9 ± 1.2</td><td>65.3 ± 1.1</td><td>105.6 ± 1.5</td><td>103.7 ± 3.9</td><td>50.7 ± 0.6</td><td>81.9 ± 1.4</td><td>72.8</td></tr></table>

Table 34: ReBRAC ablations for hopper tasks. We report final normalized score averaged over 4 unseen training seeds. 

<table><tr><td>Ablation</td><td>random</td><td>medium</td><td>expert</td><td>medium-expert</td><td>medium-replay</td><td>full-replay</td><td>Average</td></tr><tr><td>TD3+BC, paper</td><td> $8.5 \pm 0.6$ </td><td> $59.3 \pm 4.2$ </td><td> $107.8 \pm 7.0$ </td><td> $98.0 \pm 9.4$ </td><td> $60.9 \pm 18.8$ </td><td>-</td><td>-</td></tr><tr><td>TD3+BC, our</td><td> $10.3 \pm 1.8$ </td><td> $53.2 \pm 2.2$ </td><td> $108.7 \pm 5.3$ </td><td> $75.8 \pm 8.9$ </td><td> $64.5 \pm 24.9$ </td><td> $49.9 \pm 9.6$ </td><td> $60.4$ </td></tr><tr><td>TD3+BC, tuned</td><td> $10.3 \pm 1.8 (+51.5\%)$ </td><td> $57.6 \pm 6.8 (-43.2\%)$ </td><td> $110.7 \pm 2.1 (+21.1\%)$ </td><td> $106.2 \pm 2.5 (-3.4\%)$ </td><td> $64.5 \pm 24.9 (-30.8\%)$ </td><td> $106.2 \pm 2.2 (-0.6\%)$ </td><td> $75.9 (-10.6\%)$ </td></tr><tr><td>ReBRAC w/o LN</td><td> $12.2 \pm 13.3 (+79.4\%)$ </td><td> $1.0 \pm 0.6 (-99.0\%)$ </td><td> $111.1 \pm 1.0 (+21.6\%)$ </td><td> $112.4 \pm 0.7 (+2.3\%)$ </td><td> $57.4 \pm 25.0 (-38.5\%)$ </td><td> $107.2 \pm 2.0 (+0.4\%)$ </td><td> $66.8 (-21.3\%)$ </td></tr><tr><td>ReBRAC w/o layer</td><td> $8.8 \pm 0.6 (+29.4\%)$ </td><td> $101.8 \pm 0.2 (+0.4\%)$ </td><td> $103.7 \pm 5.1 (+13.5\%)$ </td><td> $104.1 \pm 7.7 (-5.3\%)$ </td><td> $97.5 \pm 3.5 (+4.5\%)$ </td><td> $106.5 \pm 0.3 (-0.3\%)$ </td><td> $87.0 (+2.5\%)$ </td></tr><tr><td>ReBRAC w/o actor penalty</td><td> $7.5 \pm 4.6 (+10.3\%)$ </td><td> $1.7 \pm 1.2 (-98.3\%)$ </td><td> $1.1 \pm 0.5 (-98.8\%)$ </td><td> $1.6 \pm 1.6 (-98.5\%)$ </td><td> $24.4 \pm 8.7 (-73.8\%)$ </td><td> $27.7 \pm 23.4 (-74.1\%)$ </td><td> $10.6 (-87.5\%)$ </td></tr><tr><td>ReBRAC w/o critic penalty</td><td> $7.4 \pm 1.1 (+8.8\%)$ </td><td> $102.3 \pm 0.5 (+0.9\%)$ </td><td> $103.4 \pm 8.6 (+13.1\%)$ </td><td> $111.2 \pm 0.7 (+1.2\%)$ </td><td> $83.1 \pm 30.9 (-10.9\%)$ </td><td> $107.5 \pm 0.1 (+0.7\%)$ </td><td> $85.8 (+1.1\%)$ </td></tr><tr><td>ReBRAC w/o large batch</td><td> $8.6 \pm 0.5 (+26.5\%)$ </td><td> $98.9 \pm 5.2 (-2.5\%)$ </td><td> $98.8 \pm 13.4 (+8.1\%)$ </td><td> $107.8 \pm 2.9 (-1.9\%)$ </td><td> $96.2 \pm 7.6 (+3.1\%)$ </td><td> $106.6 \pm 0.2 (-0.2\%)$ </td><td> $86.1 (+1.4\%)$ </td></tr><tr><td>ReBRAC</td><td> $6.8 \pm 3.4$ </td><td> $101.4 \pm 1.5$ </td><td> $91.4 \pm 4.7$ </td><td> $109.9 \pm 3.0$ </td><td> $93.3 \pm 7.5$ </td><td> $106.8 \pm 0.6$ </td><td> $84.9$ </td></tr></table>

Table 35: ReBRAC ablations for walker2d tasks. We report final normalized score averaged over 4 unseen training seeds. 

<table><tr><td>Ablation</td><td>random</td><td>medium</td><td>expert</td><td>medium-expert</td><td>medium-replay</td><td>full-replay</td><td>Average</td></tr><tr><td>TD3+BC, paper</td><td> $1.6 \pm 1.7$ </td><td> $83.7 \pm 2.1$ </td><td> $110.2 \pm 0.3$ </td><td> $110.1 \pm 0.5$ </td><td> $81.8 \pm 5.5$ </td><td>-</td><td>-</td></tr><tr><td>TD3+BC, our</td><td> $4.5 \pm 2.2$ </td><td> $77.1 \pm 1.9$ </td><td> $109.1 \pm 0.5$ </td><td> $108.9 \pm 0.3$ </td><td> $50.9 \pm 13.7$ </td><td> $86.7 \pm 5.1$ </td><td> $72.8$ </td></tr><tr><td>TD3+BC, tuned</td><td> $4.5 \pm 2.2$  (-78.8%)</td><td> $77.1 \pm 1.9$  (-5.5%)</td><td> $110.1 \pm 0.1$  (-2.0%)</td><td> $110.2 \pm 0.7$  (-1.3%)</td><td> $58.8 \pm 28.5$  (-22.2%)</td><td> $89.4 \pm 8.2$  (-13.0%)</td><td> $75.0$  (-10.9%)</td></tr><tr><td>ReBRAC w/o LN</td><td> $1.3 \pm 1.5$  (-93.9%)</td><td> $84.3 \pm 2.5$  (+3.3%)</td><td> $8.3 \pm 3.1$  (-92.6%)</td><td> $52.7 \pm 53.9$  (-52.8%)</td><td> $78.9 \pm 8.4$  (+4.4%)</td><td> $61.1 \pm 46.6$  (-40.6%)</td><td> $47.7$  (-43.3%)</td></tr><tr><td>ReBRAC w/o layer</td><td> $11.4 \pm 11.9$  (-46.5%)</td><td> $86.2 \pm 0.9$  (+5.6%)</td><td> $112.1 \pm 0.1$  (-0.2%)</td><td> $111.9 \pm 0.2$  (+0.2%)</td><td> $83.9 \pm 5.4$  (+11.0%)</td><td> $101.8 \pm 0.9$  (-1.0%)</td><td> $84.5$  (+0.4%)</td></tr><tr><td>ReBRAC w/o actor penalty</td><td> $1.1 \pm 0.9$  (-94.8%)</td><td> $1.7 \pm 2.1$  (-97.9%)</td><td> $0.9 \pm 1.1$  (-99.2%)</td><td> $0.8 \pm 1.3$  (-99.3%)</td><td> $8.9 \pm 5.7$  (-88.2%)</td><td> $64.2 \pm 29.5$  (-37.5%)</td><td> $12.9$  (-84.7%)</td></tr><tr><td>ReBRAC w/o critic penalty</td><td> $19.8 \pm 3.6$  (-7.0%)</td><td> $81.6 \pm 9.2$  (0.0%)</td><td> $112.0 \pm 0.1$  (-0.3%)</td><td> $111.6 \pm 0.3$  (-0.1%)</td><td> $87.0 \pm 4.5$  (+15.1%)</td><td> $103.5 \pm 1.5$  (+0.7%)</td><td> $85.9$  (+2.0%)</td></tr><tr><td>ReBRAC w/o large batch</td><td> $5.6 \pm 0.2$  (-73.7%)</td><td> $84.8 \pm 1.0$  (+3.9%)</td><td> $112.2 \pm 0.2$  (-0.1%)</td><td> $110.9 \pm 0.2$  (-0.7%)</td><td> $71.7 \pm 20.2$  (-5.2%)</td><td> $97.7 \pm 5.8$  (-5.0%)</td><td> $80.4$  (-4.5%)</td></tr><tr><td>ReBRAC</td><td> $21.3 \pm 0.8$ </td><td> $81.6 \pm 3.9$ </td><td> $112.3 \pm 0.0$ </td><td> $111.7 \pm 0.3$ </td><td> $75.6 \pm 10.3$ </td><td> $102.8 \pm 0.9$ </td><td> $84.2$ </td></tr></table>

Table 36: ReBRAC ablations for AntMaze tasks. We report final normalized score averaged over 4 unseen training seeds. 

<table><tr><td>Ablation</td><td>umaze</td><td>umaze-diverse</td><td>medium-play</td><td>medium-diverse</td><td>large-play</td><td>large-diverse</td><td>Average</td></tr><tr><td>TD3+BC, paper</td><td>78.6</td><td>71.4</td><td>10.6</td><td>3.0</td><td>0.2</td><td>0.0</td><td>27.3</td></tr><tr><td>TD3+BC, our</td><td> $62.0 \pm 2.4$ </td><td> $48.0 \pm 11.6$ </td><td> $0.0 \pm 0.0$ </td><td> $0.5 \pm 1$ </td><td> $0.0 \pm 0.0$ </td><td> $0.5 \pm 0.5$ </td><td>18.5</td></tr><tr><td>TD3+BC, tuned</td><td> $62.0 \pm 2.4$  (-36.8%)</td><td> $48.0 \pm 11.6$  (-42.9%)</td><td> $39.0 \pm 21.7$  (-54.8%)</td><td> $18.5 \pm 17.7$  (-72.1%)</td><td> $0.2 \pm 0.5$  (-99.6%)</td><td> $0.0 \pm 1.0$  (-100.0%)</td><td>27.9 (-62.9%)</td></tr><tr><td>ReBRAC w/o γ change</td><td> $0.0 \pm 0.0$  (-100.0%)</td><td> $90.7 \pm 3.2$  (+7.7%)</td><td> $1.0 \pm 0.0$  (-98.8%)</td><td> $0.2 \pm 0.5$  (-99.7%)</td><td> $19.3 \pm 18.5$  (-58.0%)</td><td> $15.0 \pm 8.0$  (-79.0%)</td><td>21.0 (-72.1%)</td></tr><tr><td>ReBRAC w/o LN</td><td> $0.0 \pm 0.0$  (-100%)</td><td> $0.0 \pm 0.0$  (-100%)</td><td> $0.0 \pm 0.0$  (-100%)</td><td> $0.0 \pm 0.0$  (-100%)</td><td> $0.0 \pm 0.0$  (-100%)</td><td> $0.0 \pm 0.0$  (-100%)</td><td>0.0 (-100%)</td></tr><tr><td>ReBRAC w/o layer</td><td> $31.0 \pm 45.4$  (-68.4%)</td><td> $61.7 \pm 25.3$  (-26.7%)</td><td> $0.0 \pm 0.0$  (-100.0%)</td><td> $16.0 \pm 32.0$  (-75.8%)</td><td> $0.0 \pm 0.0$  (-100.0%)</td><td> $0.0 \pm 0.0$  (-100.0%)</td><td>18.1 (-76.0%)</td></tr><tr><td>ReBRAC w/o actor penalty</td><td> $1.0 \pm 1.1$  (-99.0%)</td><td> $0.0 \pm 0.0$  (-100%)</td><td> $0.0 \pm 0.0$  (-100%)</td><td> $0.0 \pm 0.0$  (-100%)</td><td> $0.0 \pm 0.0$  (-100%)</td><td> $0.0 \pm 0.0$  (-100%)</td><td>01 (-99.9%)</td></tr><tr><td>ReBRAC w/o critic penalty</td><td> $98.2 \pm 1.5$  (0.0%)</td><td> $78.0 \pm 26.3$  (-7.4%)</td><td> $86.2 \pm 2.6$  (0.0%)</td><td> $57.5 \pm 24.2$  (-13.1%)</td><td> $56.7 \pm 32.9$  (+23.3%)</td><td> $57.0 \pm 16.4$  (-20.3%)</td><td>72.2 (-4.1%)</td></tr><tr><td>ReBRAC w large batch</td><td> $60.7 \pm 31.3$  (-38.2%)</td><td> $68.5 \pm 17.9$  (-18.6%)</td><td> $43.9 \pm 49.9$  (-49.1%)</td><td> $34.0 \pm 40.6$  (-48.6%)</td><td> $39.2 \pm 45.9$  (-14.8%)</td><td> $0.0 \pm 0.0$  (-100.0%)</td><td>41.0 (-45.6%)</td></tr><tr><td>ReBRAC</td><td> $98.2 \pm 0.9$ </td><td> $84.2 \pm 18.5$ </td><td> $86.2 \pm 4.7$ </td><td> $66.2 \pm 16.3$ </td><td> $46.0 \pm 40.0$ </td><td> $71.5 \pm 12.3$ </td><td>75.3</td></tr></table>

Table 37: ReBRAC ablations for pen tasks. We report final normalized score averaged over 4 unseen training seeds. 

<table><tr><td>Ablation</td><td>human</td><td>cloned</td><td>expert</td><td>Average</td></tr><tr><td>TD3+BC, paper</td><td>0.0</td><td>0.0</td><td>0.3</td><td>0.0</td></tr><tr><td>TD3+BC, our</td><td>65.9 ± 24.6</td><td>78.1 ± 5.7</td><td>144.9 ± 7.5</td><td>96.3</td></tr><tr><td>TD3+BC, tuned</td><td>77.6 ± 18.5 (-23.9%)</td><td>78.1 ± 5.7 (-8.5%)</td><td>144.9 ± 7.5 (-10.3%)</td><td>100.2 (-12.5%)</td></tr><tr><td>ReBRAC w/o LN</td><td>78.6 ± 14.8 (-22.9%)</td><td>21.3 ± 11.0 (-75.1%)</td><td>86.7 ± 59.8 (-44.6%)</td><td>62.1 (-45.8%)</td></tr><tr><td>ReBRAC w/o layer</td><td>89.1 ± 14.7 (-12.6%)</td><td>106.7 ± 13.9 (+24.9%)</td><td>147.2 ± 5.7 (-6.0%)</td><td>114.3 (-0.3%)</td></tr><tr><td>ReBRAC w/o actor penalty</td><td>-0.5 ± 1.3 (-100.5%)</td><td>0.6 ± 1.6 (-99.3%)</td><td>0.0 ± 3.6 (-100.0%)</td><td>0.0 (-100.0%)</td></tr><tr><td>ReBRAC w/o critic penalty</td><td>99.9 ± 6.1 (-2.1%)</td><td>75.0 ± 16.7 (-12.2%)</td><td>154.6 ± 1.8 (-1.3%)</td><td>109.8 (-4.2%)</td></tr><tr><td>ReBRAC w large batch</td><td>67.2 ± 9.0 (-34.1%)</td><td>83.2 ± 23.3 (-2.6%)</td><td>155.0 ± 6.8 (-1.0%)</td><td>101.8 (-11.2%)</td></tr><tr><td>ReBRAC</td><td>102.0 ± 10.8</td><td>85.4 ± 24.2</td><td>156.6 ± 1.4</td><td>114.6</td></tr></table>

Table 38: ReBRAC ablations for door tasks. We report final normalized score averaged over 4 unseen training seeds. 

<table><tr><td>Ablation</td><td>human</td><td>cloned</td><td>expert</td><td>Average</td></tr><tr><td>TD3+BC, paper</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>TD3+BC, our</td><td>0.0 ± 0.1</td><td>0.4 ± 1.0</td><td>102.5 ± 2.9</td><td>34.3</td></tr><tr><td>TD3+BC, tuned</td><td>0.0 ± 0.1 (-)</td><td>0.4 ± 1.0 (+100.0%)</td><td>105.8 ± 0.3 (+0.8%)</td><td>35.4 (+1.1%)</td></tr><tr><td>ReBRAC w/o LN</td><td>-0.1 ± 0.0 (-)</td><td>-0.3 ± 0.0 (-250.0%)</td><td>106.0 ± 0.8 (+1.0%)</td><td>35.1 (+0.3%)</td></tr><tr><td>ReBRAC w/o layer</td><td>0.0 ± 0.0 (-)</td><td>0.1 ± 0.5 (-50.0%)</td><td>104.4 ± 2.3 (-0.5%)</td><td>34.8 (-0.6%)</td></tr><tr><td>ReBRAC w/o actor penalty</td><td>-0.1 ± 0.1 (-)</td><td>0.0 ± 0.0 (-100.0%)</td><td>0.0 ± 0.2 (-100.0%)</td><td>0.0 (-100.0%)</td></tr><tr><td>ReBRAC w/o critic penalty</td><td>0.0 ± 0.0 (-)</td><td>0.1 ± 0.0 (-50.0%)</td><td>106.1 ± 0.3 (+1.1%)</td><td>35.4 (+1.1%)</td></tr><tr><td>ReBRAC w large batch</td><td>-0.1 ± 0.1 (-)</td><td>0.1 ± 0.3 (-50.0%)</td><td>106.1 ± 0.1 (+1.1%)</td><td>35.3 (+0.9%)</td></tr><tr><td>ReBRAC</td><td>0.0 ± 0.0</td><td>0.2 ± 0.3</td><td>104.9 ± 2.2</td><td>35.0</td></tr></table>

Table 39: ReBRAC ablations for hammer tasks. We report final normalized score averaged over 4 unseen training seeds. 

<table><tr><td>Ablation</td><td>human</td><td>cloned</td><td>expert</td><td>Average</td></tr><tr><td>TD3+BC, paper</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>TD3+BC, our</td><td>0.3 ± 0.4</td><td>1.1 ± 1.1</td><td>127.0 ± 0.4</td><td>42.8</td></tr><tr><td>TD3+BC, tuned</td><td>0.3 ± 0.4 (+50.0%)</td><td>1.1 ± 1.1 (-80.0%)</td><td>127.0 ± 0.4 (-5.3%)</td><td>42.8 (-8.1%)</td></tr><tr><td>ReBRAC w/o LN</td><td>0.2 ± 0.0 (0.0%)</td><td>1.0 ± 1.0 (-81.8%)</td><td>9.9 ± 19.1 (-92.6%)</td><td>3.6 (-92.3%)</td></tr><tr><td>ReBRAC w/o layer</td><td>0.1 ± 0.0 (-50.0%)</td><td>21.3 ± 19.7 (+287.3%)</td><td>133.1 ± 0.5 (-0.8%)</td><td>51.5 (+10.5%)</td></tr><tr><td>ReBRAC w/o actor penalty</td><td>0.0 ± 0.0 (-100.0%)</td><td>0.0 ± 0.1 (-100.0%)</td><td>0.0 ± 0.1 (-100.0%)</td><td>0.0 (-100.0%)</td></tr><tr><td>ReBRAC w/o critic penalty</td><td>0.1 ± 0.1 (-50.0%)</td><td>1.9 ± 0.7 (-65.5%)</td><td>134.1 ± 0.2 (-0.1%)</td><td>45.3 (-2.8%)</td></tr><tr><td>ReBRAC w large batch</td><td>0.3 ± 0.8 (+50.0%)</td><td>10.6 ± 14.0 (+92.7%)</td><td>133.4 ± 0.5 (-0.6%)</td><td>48.1 (+3.2%)</td></tr><tr><td>ReBRAC</td><td>0.2 ± 0.2</td><td>5.5 ± 2.5</td><td>134.2 ± 0.4</td><td>46.6</td></tr></table>

Table 40: ReBRAC ablations for relocate tasks. We report final normalized score averaged over 4 unseen training seeds. 

<table><tr><td>Ablation</td><td>human</td><td>cloned</td><td>expert</td><td>Average</td></tr><tr><td>TD3+BC, paper</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>TD3+BC, our</td><td>0.0 ± 0.0</td><td>-0.1 ± 0.0</td><td>107.9 ± 0.6</td><td>35.9</td></tr><tr><td>TD3+BC, tuned</td><td>0.0 ± 0.0 (-)</td><td>-0.1 ± 0.0 (-105.3%)</td><td>107.9 ± 0.6 (+1.2%)</td><td>35.9 (-0.5%)</td></tr><tr><td>ReBRAC w/o LN</td><td>-0.2 ± 0.0 (-)</td><td>0.0 ± 0.3 (-100.0%)</td><td>-0.1 ± 0.0 (-100.1%)</td><td>-0.1 (-100.3%)</td></tr><tr><td>ReBRAC w/o layer</td><td>0.1 ± 0.3 (-)</td><td>1.7 ± 2.1 (-10.5%)</td><td>105.0 ± 3.1 (-1.5%)</td><td>35.6 (-1.4%)</td></tr><tr><td>ReBRAC w/o actor penalty</td><td>-0.1 ± 0.0 (-)</td><td>0.0 ± 0.0 (-100.0%)</td><td>-0.1 ± 0.1 (-100.1%)</td><td>0.0 (-100.0%)</td></tr><tr><td>ReBRAC w/o critic penalty</td><td>0.0 ± 0.1 (-)</td><td>1.9 ± 1.9 (0.0%)</td><td>109.6 ± 1.2 (+2.8%)</td><td>37.1 (+2.8%)</td></tr><tr><td>ReBRAC w large batch</td><td>0.0 ± 0.0 (-)</td><td>0.1 ± 0.2 (-94.7%)</td><td>109.6 ± 0.9 (+2.8%)</td><td>36.5 (+1.1%)</td></tr><tr><td>ReBRAC</td><td>0.0 ± 0.0</td><td>1.9 ± 2.3</td><td>106.6 ± 3.1</td><td>36.1</td></tr></table>