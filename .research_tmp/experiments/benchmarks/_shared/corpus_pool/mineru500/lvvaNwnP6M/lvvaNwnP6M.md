# H-InDex: Visual Reinforcement Learning with Hand-Informed Representations for Dexterous Manipulation

Yanjie Ze $^{12}$ Yuyao Liu $^{3*}$ Ruizhe Shi $^{3*}$ Jiaxin Qin $^{4}$ Zhecheng Yuan $^{31}$ Jiashun Wang $^{5}$ Huazhe Xu $^{316}$

$^{1}$ Shanghai Qi Zhi Institute $^{2}$ Shanghai Jiao Tong University $^{3}$ Tsinghua University, IIIS $^{4}$ Renmin University of China $^{5}$ Carnegie Mellon University $^{6}$ Shanghai AI Lab $^{*}$ Equal contribution. Order is decided by coin flip.

yanjieze.com/H-InDex

# Abstract

Human hands possess remarkable dexterity and have long served as a source of inspiration for robotic manipulation. In this work, we propose a human Hand-Informed visual representation learning framework to solve difficult Dexterous manipulation tasks (H-InDex) with reinforcement learning. Our framework consists of three stages: (i) pre-training representations with 3D human hand pose estimation, (ii) offline adapting representations with self-supervised keypoint detection, and (iii) reinforcement learning with exponential moving average BatchNorm. The last two stages only modify 0.36% parameters of the pre-trained representation in total, ensuring the knowledge from pre-training is maintained to the full extent. We empirically study 12 challenging dexterous manipulation tasks and find that H-InDex largely surpasses strong baseline methods and the recent visual foundation models for motor control. Code is available at yanjieze.com/H-InDex.

# 1 Introduction

Humans can adeptly tackle intricate and novel dexterous manipulation tasks. However, multifingered robotic hands still struggle to achieve such dexterity efficiently. Recent progress in representation learning for visuomotor tasks has proved that pre-trained universal representations may accelerate robot learning manipulation tasks $[17, 18, 27, 31]$ . In light of previous success, the similar morphology between human hands and robotic hands begs the question: can robotic hands leverage representations learned from human hands for achieving dexterity?

In this paper, we propose Hand-Informed visual reinforcement learning framework for Dexterous manipulation (H-InDex) that uses and adapts visual representations from human hands to boost robotic hand dexterity. Our framework consists of three stages:

Normalized Average Score   
![](images/592b4de48b729ec1a3bc95ae35be6feb53b5aa955e5b3f171bc9766116e64969.jpg)

<details>
<summary>bar</summary>

| Model | Score |
| :--- | :--- |
| H-InDex (ResNet-50) | 98.2 |
| VC-1 (ViT-B) | 81.4 |
| RRL (ResNet-50) | 72.8 |
| MVP (ViT-S) | 80.1 |
| R3M (ResNet-50) | 35.7 |
* Model params: ResNet-50 (24 M), ViT-B (86 M), ViT-S (22 M)
</details>

Figure 1: Normalized average score for our algorithm H-InDex and the baselines (VC-1 [17], MVP [31], R3M [18], and RRL [27]).

- Stage 1: Pre-training representations with 3D human hand pose estimation, where we adopt the feature encoder from an off-the-shelf 3D hand pose estimator FrankMocap [25].   
- Stage 2: Offline adapting representations with self-supervised keypoint detection, where we freeze the convolutional layers in the pre-trained representation and only finetune the affine

transformations in BatchNorm layers (0.18% parameters of the entire model). Such minimal modification of the pre-trained representations ensures that human dexterity is retained to the maximum extent and adapts the human hand representations into the target robotic domain.

\- Stage 3: Reinforcement learning with exponential moving average (EMA) BatchNorm and the adapted representations. EMA operates to dynamically update the mean and variance in BatchNorm layers, to further make the model adapt to the progressive learning stages.

In contrast to previous works that also learn representations from human videos $[17, 18, 31]$ , there are two major benefits of our framework: i) H-InDex explicitly learn human dexterity by forcing the model to predict the 3D hand pose instead of predicting or discriminating pixels unsupervisedly using masked auto-encoding $[9]$ or time contrastive learning $[26]$ ; ii) H-InDex directly adopts the off-the-shelf visual model that is designed to capture human hands rather than training large models on large-scale datasets for specific robotic tasks. These two points combined demonstrate a new cost-effective way to solve robotic tasks such as dexterous manipulation by leveraging existing visual models that are originally and only designed for human understanding.

To show the effectiveness of H-InDex, we experiment on 12 challenging visual dexterous manipulation tasks from Adroit [24] and DexMV [22]. We mainly report episode returns instead of success rates to better show how well the robots solve the tasks. In comparison with several strong visual foundation models for motor control, H-InDex largely surpasses all of them as shown in Figure 1.

To summarize, our contributions are three-fold:

\- We propose a novel visual reinforcement learning framework called H-InDex to utilize rich human hand information efficiently for dexterous manipulation.

\- We show the effectiveness of our framework on 12 challenging visual dexterous manipulation tasks, comparing with recent strong foundation models such as VC-1 [17].

\- Our study has offered valuable insights into the application of pre-trained models for dexterous manipulation, by exploring the direct application of a 3D human hand pose estimation model, originating from the vision community.

# 2 Related Work

Visual reinforcement learning for dexterous manipulation. Recent research has explored the use of deep reinforcement learning (RL) for solving dexterous manipulation tasks $[7,21,24,27,30]$ . For example, Rajeswaran et al. $[24]$ investigated the use of vector state information as input to the RL algorithm. Despite the success, assuming access to the ground-truth state limits its possibility to be deployed in the real world. RRL $[27]$ finds that ImageNet pre-trained ResNets $[10]$ are surprisingly effective in achieving dexterity with visual observations. Under the umbrella of visual RL, MoDem $[7]$ leverages a learned dynamics model to solve the tasks with good utilization of demonstrations. Furthermore, VRL3 $[30]$ utilizes offline RL to pre-train the visual representations and the policies in an end-to-end manner. In this work, H-InDex is designed to focus on visual representations while leaving the policy, training framework, and reward signals unchanged. As a result, H-InDex offers an orthogonal and complementary approach to prior efforts in this area.

Foundation models for visuo-motor control. Given the diversity of robotic tasks and computational constraints, there is a growing interest in developing a single visual foundation model that can serve as a general feature extractor. Such a model would enable the processing of high-dimensional visual observations into compact vectors, providing a promising approach for efficient and effective control of a wide range of robotic systems $[8, 17–19, 27, 31, 33, 34]$ . Among them, R3M $[18]$ pre-trains a ResNet-50 on Ego4D $[6]$ dataset and evaluates on several robotic manipulation tasks with imitation learning. MVP $[23, 31]$ pre-trains vision transformers $[4]$ with Masked AutoEncoder (MAE) $[9]$ on internet-scale data, achieving strong results on dexterous manipulation tasks. Similarly, a very recent foundation model VC-1 $[17]$ explores the scaling up of MAE for motor control and achieves consistently strong results across a wide range of benchmarks. However, it should be noted that VC-1 and R3M only employ IL to solve dexterous manipulation tasks, making it unclear whether these models are suitable for the setting of reinforcement learning, where agents need to trade off between exploration and exploitation. GNFactor $[34]$ distills the 2D foundation models into 3D space, but their agents are limited to address the gripper-based manipulation problems.

Learning dexterity from videos. A growing body of recent research aims to leverage human manipulation videos for improving visuomotor control tasks $[17, 20, 22, 28, 29, 31]$ . A line of works focuses on directly extracting human hand poses from videos and employing RL/IL to train on the retargeted robot joint positions, such as DexMV $[22]$ , Robotic telekinesis $[29]$ , VideoDex $[28]$ , and Imitate Video $[20]$ . In contrast to these approaches, our work explores a representation learning approach for leveraging online human-object interaction videos without explicit pose estimation to improve dexterity in robotic manipulation, sharing a similar motivation as MVP $[31]$ and VC-1 $[17]$ .

# 3 Preliminaries

Formulation. We model the problem as a Markov Decision Process (MDP) $\mathcal{M} = \langle \mathcal{S},\mathcal{A},\mathcal{T},\mathcal{R},\gamma \rangle$ , where $\mathbf{s}\in \mathcal{S}$ are states, $\mathbf{a}\in \mathcal{A}$ are actions, $\mathcal{T}:\mathcal{S}\times \mathcal{A}\mapsto \mathcal{S}$ is a transition function, $r\in \mathcal{R}$ are rewards, and $\gamma \in [0,1)$ is a discount factor. The agent's goal is to learn a policy $\pi_{\theta}$ that maximizes discounted cumulative rewards on $\mathcal{M}$ , i.e., $\max_{\theta}\mathbb{E}_{\pi_{\theta}}\left[\sum_{t = 0}^{\infty}\gamma^{t}r_{t}\right]$ , while using as few interactions with the environment as possible, referred as sample efficiency.

In this work, we focus on visual RL for dexterous manipulation tasks, where actions are high-dimensional $(\mathbf{a} \in \mathcal{A}^{30})$ and ground-truth states s are generally unknown, approximated by image observations $o \in O$ together with robot proprioceptive sensory information $q \in Q$ , i.e., $\mathbf{s} = (\mathbf{o}, \mathbf{q})$ . To better address the hard exploration problem in high-dimensional control [7, 24, 27, 30], we assume access to a limited number of expert demonstrations $D_{expert} = \{D_1, D_2, \cdots, D_N\}$ .

Demo Augmented Policy Gradient (DAPG) [24] is a model-free policy gradient method that utilizes given demonstrations to augment the policy and trains the policy with natural policy gradient (NPG) [13]. DAPG mainly consists of two stages:

(1) Pre-training the policy with behavior cloning, which is to solve the following maximum-likelihood problem:

$$
\underset {\theta} {\text { maximize }} \sum_ {(s, a) \in \mathcal {D} _ {\text { expert }}} \ln \pi_ {\theta} (a \mid s). \tag {1}
$$

(2) RL finetuning with the demo augmented loss, which is to add an additional term to the gradient,

$$
g _ {\text { aug }} = \underbrace {\sum_ {(s , a) \in \mathcal {D} _ {\pi}} \nabla_ {\theta} \ln \pi_ {\theta} (a \mid s) A ^ {\pi} (s , a)} _ {\text { original   gradient }} + \underbrace {\sum_ {(s , a) \in \mathcal {D} _ {\text { expert }}} \nabla_ {\theta} \ln \pi_ {\theta} (a \mid s) w (s , a)} _ {\text { demo   augmented   gradient }}, \tag {2}
$$

where $A^{\pi}(s,a)$ is the advantage function, $D_{\pi}$ represents the dataset obtained by executing policy $\pi_{\theta}$ on the MDP, and $w(s,a)$ is a weighting function.

# 4 Method

In this work, our goal is to achieve sample efficient visual reinforcement learning agents in dexterous manipulation tasks by incorporating human hand dexterity into visual representations. To this end, we propose Hand-Informed visual reinforcement learning for Dexterous manipulation (H-InDex), a simple yet effective learning framework to address the contact-rich dexterous manipulation problems effectively in limited interactions. The overview of our method is provided in Figure 2. H-InDex consists of three stages: 1) a representation pre-training stage where we pre-train the visual representations with the 3D human hand pose estimation task, aiming to make visual representations understand human hand dexterity from diverse natural videos; 2) a representation offline adaptation stage where we adapt only 0.18% parameters in the pre-trained representation with the self-supervised keypoint objective with in-domain data; 3) a reinforcement learning stage where the visual representation is frozen and we utilize the exponential moving average operation to update the mean and variance in BatchNorm of the visual representations.

Stage 1: Representation pre-training. We start by pre-training visual representations with monocular 3D human hand pose estimation, which is a well-established human hand understanding task in the computer vision community with large-scale annotated datasets available. Together with the datasets, there are a plethora of open-sourced models from which we use an off-the-shelf model FrankMocap [25]. FrankMocap is a whole-body pose estimation system with the hand module trained

(1) Pre-train representation with 3D human hand pose estimation   
![](images/ec08873d66e77769394894999d2695c77bd05ffe77b447cc8d574ee12e4ce443.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Input Image"] --> B["image encoder"]
    B --> C["conv"]
    B --> D["B N"]
    B --> E["conv"]
    B --> F["B N"]
    B --> G["avg food"]
    C --> H["compact repr."]
    H --> I["hand decoder"]
    I --> J["Hand Recognition Icon"]
```
</details>

(2) Offline adapt representation with self-supervised keypoint detection   
![](images/05574ca8553d4f5578f580cf7786b87f474c40e4d0b28a389490902c48d3bfdd.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["src view"] --> B["image encoder"]
    B --> C["feature map(7x7)"]
    C --> D["keypoint decoder"]
    D --> E["reconstruct tgt view"]
```
</details>

![](images/f7492c13dc16301d8b8b0fdced6b4ff1401e802671c61819328afa7912b03557.jpg)

<details>
<summary>line</summary>

| Time | Human | Adaptive |
|------|-------|----------|
| 0    | 0     | 0        |
| 1    | 0.5   | 0.2      |
| 2    | 1.0   | 0.8      |
| 3    | 1.5   | 1.5      |
| 4    | 2.0   | 2.5      |
| 5    | 2.5   | 3.0      |
| 6    | 3.0   | 3.5      |
| 7    | 3.5   | 4.0      |
| 8    | 4.0   | 4.5      |
| 9    | 4.5   | 5.0      |
| 10   | 5.0   | 5.5      |
| 11   | 5.5   | 6.0      |
| 12   | 6.0   | 6.5      |
| 13   | 6.5   | 7.0      |
| 14   | 7.0   | 7.5      |
| 15   | 7.5   | 8.0      |
| 16   | 8.0   | 8.5      |
| 17   | 8.5   | 9.0      |
| 18   | 9.0   | 9.5      |
| 19   | 9.5   | 10.0     |
| 20   | 10.0  | 10.5     |
| 21   | 10.5  | 11.0     |
| 22   | 11.0  | 11.5     |
| 23   | 11.5  | 12.0     |
| 24   | 12.0  | 12.5     |
| 25   | 12.5  | 13.0     |
| 26   | 13.0  | 13.5     |
| 27   | 13.5  | 14.0     |
| 28   | 14.0  | 14.5     |
| 29   | 14.5  | 15.0     |
| 30   | 15.0  | 15.5     |
| 31   | 15.5  | 16.0     |
| 32   | 16.0  | 16.5     |
| 33   | 16.5  | 17.0     |
| 34   | 17.0  | 17.5     |
| 35   | 17.5  | 18.0     |
| 36   | 18.0  | 18.5     |
| 37   | 18.5  | 19.0     |
| 38   | 19.0  | 19.5     |
| 39   | 19.5  | 20.0     |
| 40   | 20.0  | 20.5     |
| 41   | 20.5  | 21.0     |
| 42   | 21.0  | 21.5     |
| 43   | 21.5  | 22.0     |
| 44   | 22.0  | 22.5     |
| 45   | 22.5  | 23.0     |
| 46   | 23.0  | 23.5     |
| 47   | 23.5  | 24.0     |
| 48   | 24.0  | 24.5     |
| 49   | 24.5  | 25.0     |
| 50   | 25.0  | 25.5     |
| ... (after label) | ... (after label) | ... (after label) |
The image contains two lines: one for 'Human' and one for 'Adapt affine transformation', but the latter line is not explicitly labeled in the image but is mentioned in the legend as 'Adapt affine transformation'. The text 'human' appears to be part of the chart title and is not explicitly labeled in the image.
</details>

Adapt running mean and var low return $\rightarrow$ ... $\rightarrow$ high return   
![](images/cc2f49e415969bbc5e739ac7250ba1df66888f6e7dcc82aa81a2a5d074d8f2b5.jpg)

![](images/5377c70a881d868bbc46dc3fed5c07c1f611dc85032fbeaaff8da09200ccae6b.jpg)

(3) Reinforcement learning with compact representation and EMA BatchNorm   
![](images/020aa722b7b395d616b6f8a747ce3a9ca3b3a5bf01f848df9101134df1811052.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Input Image Encoder"] --> B["image encoder"]
    B --> C["compact repr."]
    C --> D["policy"]
    D --> E_Lisbonation["Action"]
```
</details>

Figure 2: The overview of H-InDex. H-InDex consists of three stages: 1) representation pre-training, 2) representation offline adaptation, and 3) reinforcement learning.

on 6 diverse hand datasets, totaling 400k samples. We adopt the ResNet-50 [10] feature encoder in the hand module to extract visual representations.

The use of a pre-trained model from the 3D hand pose estimation task $[25]$ shares the intuition with recent works on foundation models for motor control $[16–19, 31]$ : learning representations from human manipulation videos. However, the use of the pre-trained hand model offers two distinct advantages that are not typically found in other approaches: i) the model explicitly predicts the hand-related information from diverse videos, forcing it to learn the interaction and the movement of human hands; ii) the model can be borrowed from vision community without any extra cost to re-train a foundation model.

Stage 2: Representation offline adaptation. In the previous stage, we only pre-train visual representations that are suitable for human-centric images, neglecting the morphology and structure gap between robot hands and human hands. To bridge the gap without losing the information learned in the pre-training stage, we adopt a self-supervised keypoint detection objective $[11, 14, 15]$ to only finetune the affine transformations in the BatchNorm layers of the pre-trained model, which occupy only 0.18% of the entire model parameters. While finetuning only a small portion of parameters, it empirically outperforms both a frozen model and a fully finetuned model. We hypothesize that this is because the BatchNorm finetuning bridges the gap and mitigates catastrophic forgetting caused by finetuning $[1]$ .

We now describe the self-supervised keypoint objective. Given a target image $I_{t}$ and a source image $I_{s}$ sampled from a video, we aim to reconstruct $I_{t}$ with the appearance feature of $I_{s}$ and the keypoint feature of $I_{t}$ . Denote our pre-trained visual representation as $h_{\theta}$ , the keypoint feature extractor as $\mathcal{K}_{\psi}$ , the appearance feature extractor as $\mathcal{F}_{\phi}$ , and the image decoder as $\mathcal{G}_{\omega}$ . First, we extract a semantic

![](images/6849de442da3f7ac1aa48fe7c4eb9fdc2a092473faad3a279e65c360adf17d75.jpg)

<details>
<summary>text_image</summary>

Hammer
Door
Pen
Pour
Place Inside
Relocate Objects
Start
End
</details>

Figure 3: Visualization of our six kinds of dexterous manipulation tasks and one sampled trajectory. We depict both the initial configuration and the goal. Videos of trajectories for all tasks are available on our website yanjieze.com/H-InDex.

feature map $h_{\theta}(I_{t})$ from the target image $I_{t}$ and then get the keypoint feature $\mathcal{K}_{\psi}(h_{\theta}(I_{t}))$ . At the same time, we extract the appearance feature $\mathcal{F}_{\phi}(I_{s})$ from the source image $I_{s}$ . We then try to reconstruct the target image $I_{t}$ by decoding the concatenated keypoint feature and the appearance feature as $I_{t}^{\prime} = \mathcal{G}_{\omega}(\mathcal{K}_{\psi}(h_{\theta}(I_{t})), \mathcal{F}_{\phi}(I_{s}))$ . Our final supervision is the perceptual loss [12] $L_{percep}$ ,

$$
\mathcal {L} _ {\text { keypoint }} = \mathcal {L} _ {\text { percep }} (I _ {t}, I _ {t} ^ {\prime}) = \| \Lambda (I _ {t}) - \Lambda (\mathcal {G} _ {\omega} (\mathcal {K} _ {\psi} (h _ {\theta} (I _ {t})), \mathcal {F} _ {\phi} (I _ {s}))) \| _ {2} ^ {2}, \tag {3}
$$

where $\Lambda$ is the semantic feature prediction function in [12].

Stage 3: Reinforcement learning. During the reinforcement learning stage, the distribution of observations is continually changing. For example, in the early learning stage, the observations are usually random explorations, while at the end of the learning stage, most of the observations are converged trajectories. Such a property of reinforcement learning requires the internal statistics of neural networks to move slowly towards the current observation distribution. Therefore, we utilize the exponential moving average (EMA) operation to dynamically update the statistics (i.e., the running mean and the running variance) in BatchNorm layers.

Formally, for the input x that has k dimensions, i.e., $x = \{x^{(1)}, \cdots, x^{(k)}\}$ , we update the running mean $\mu^{(i)}$ and the running variance $(\sigma^{(i)})^{2}$ in BatchNorm layers with the following equation,

$$
\mu^ {(i)} \leftarrow (1 - m) \cdot \mu^ {(i)} + m \cdot \mathbb {E} [ x ^ {(i)} ], \tag {4}
$$

$$
\left(\sigma^ {(i)}\right) ^ {2} \leftarrow (1 - m) \cdot \left(\sigma^ {(i)}\right) ^ {2} + m \cdot \operatorname{Var} [ x ^ {(i)} ], \tag {5}
$$

for $i = 1, \cdots, k$ , where m is the momentum. When m is set to 0, our EMA BatchNorm layers revert back to the original layers, ensuring that the modification does not have negative impacts at the very least. Finally, all these three stages collectively contribute to our final method H-InDex. We remain implementation details in Appendix A.

# 5 Experiments

In this work, we delve into the application of visual reinforcement learning to address dexterous manipulation tasks, with a particular emphasis on the visual representation aspect. We evaluate the effectiveness of our proposed framework, H-InDex, across various tasks and elucidate the significance of each component in achieving the final results. Of particular importance is the integration of prior knowledge pertaining to human hand dexterity into our framework.

![](images/f3fe6f19242a4aad5a803088fc47d9c922fceb773668cea5bea73475e140be91.jpg)

<details>
<summary>line</summary>

| Episode | Red Line | Yellow Line | Green Line | Purple Line | Blue Line |
| ------- | -------- | ----------- | ---------- | ----------- | --------- |
| 0.0     | 0        | 0           | 0          | 0           | 0         |
| 0.5     | 5000     | 3000        | 2000       | 1000        | 500       |
| 1.0     | 10000    | 6000        | 4000       | 2000        | 1000      |
| 1.5     | 13000    | 8000        | 6000       | 3000        | 1500      |
| 2.0     | 15000    | 10000       | 7000       | 4000        | 2000      |
</details>

![](images/73903978739d2a66e58eac37cb85a5f128f530a2a313844d47819b89109dc83e.jpg)

<details>
<summary>line</summary>

| x    | Red   | Blue  | Green | Yellow |
| ---- | ----- | ----- | ----- | ------ |
| 0    | 0     | 0     | 0     | 0      |
| 1    | 1500  | 2000  | 1000  | 2500   |
| 2    | 2000  | 2500  | 1500  | 2750   |
| 3    | 2500  | 2750  | 2000  | 2875   |
| 4    | 2750  | 2875  | 2250  | 2937   |
</details>

![](images/2266742991e8aa0fd57e77933b4fce1eb60953adc3043dd272b406acd313d919.jpg)

<details>
<summary>line</summary>

| x    | Red   | Yellow | Green | Blue  |
| ---- | ----- | ------ | ----- | ----- |
| 0    | 1500  | 1500   | 1500  | 1500  |
| 1    | 2000  | 2000   | 2000  | 1700  |
| 2    | 2500  | 2500   | 2500  | 1800  |
| 3    | 3000  | 3000   | 3000  | 1900  |
| 4    | 3500  | 3500   | 3500  | 2000  |
| 5    | 3750  | 3750   | 3750  | 2100  |
| 6    | 3875  | 3875   | 3875  | 2200  |
| 7    | 3937.5| 3937.5 | 3937.5| 2250  |
| 8    | 4000  | 4000   | 4000  | 2300  |
</details>

![](images/242863e8648545ca770b484046b8e9cedcea4e393e9bd4ca188f813ee0a947bd.jpg)

<details>
<summary>line</summary>

| x    | Red   | Yellow | Green | Blue  |
| ---- | ----- | ------ | ----- | ----- |
| 0    | 5000  | 0      | 0     | 0     |
| 2    | 10000 | 5000   | 2500  | 1000  |
| 4    | 15000 | 10000  | 7500  | 3750  |
| 6    | 17500 | 12500  | 10000 | 7500  |
| 8    | 20000 | 15000  | 12500 | 12500 |
</details>

![](images/d55c1ef50055855003867b338fbda61da5739db6739a65e003e1ff9ae1bf2dd7.jpg)

<details>
<summary>line</summary>

| Episode | Episode Return (Red) | Episode Return (Orange) | Episode Return (Green) |
| ------- | -------------------- | ----------------------- | ---------------------- |
| 0       | 0                    | 0                       | 0                      |
| 1       | ~100                 | ~90                     | ~70                    |
| 2       | ~150                 | ~130                    | ~90                    |
| 3       | ~180                 | ~160                    | ~110                   |
| 4       | ~200                 | ~180                    | ~130                   |
| 5       | ~220                 | ~200                    | ~150                   |
| 6       | ~230                 | ~210                    | ~160                   |
| 7       | ~240                 | ~220                    | ~170                   |
| 8       | ~250                 | ~230                    | ~180                   |
</details>

![](images/5e9e66214a864061f74f8aa1ef4f2bee4cedc296d5e38632c9072b5fde82a858.jpg)

<details>
<summary>line</summary>

| x  | Red   | Yellow | Purple | Green | Blue  |
|----|-------|--------|--------|-------|-------|
| 0  | 0     | 0      | 0      | 0     | 0     |
| 1  | 300   | 200    | 150    | 100   | 50    |
| 2  | 500   | 350    | 250    | 150   | 100   |
| 3  | 650   | 450    | 350    | 200   | 150   |
| 4  | 800   | 550    | 450    | 250   | 200   |
| 5  | 900   | 650    | 550    | 300   | 250   |
| 6  | 1000  | 750    | 650    | 350   | 300   |
| 7  | 1100  | 850    | 750    | 400   | 350   |
| 8  | 1200  | 950    | 850    | 450   | 400   |
</details>

![](images/ac7ec8684aac210ad9323840093e3d6a975d1dcc2f0dcae66f695414e83c1b5a.jpg)

<details>
<summary>line</summary>

| x    | Red   | Green | Purple | Yellow | Blue  |
| ---- | ----- | ----- | ------ | ------ | ----- |
| 0.0  | 400   | 400   | 400    | 400    | 400   |
| 0.5  | 1200  | 800   | 600    | 500    | 450   |
| 1.0  | 1300  | 1000  | 800    | 600    | 500   |
| 1.5  | 1400  | 1200  | 1000   | 700    | 550   |
| 2.0  | 1500  | 1300  | 1100   | 800    | 600   |
</details>

![](images/162d3635cbd77869af434550b1d2438d373d8524aba13501e2098195ffce5d31.jpg)

<details>
<summary>line</summary>

| x    | Red   | Purple | Green | Blue  | Yellow |
| ---- | ----- | ------ | ----- | ----- | ------ |
| 0    | 0     | 0      | 0     | 0     | 0      |
| 1    | 250   | 200    | 150   | 100   | 50     |
| 2    | 500   | 400    | 350   | 250   | 200    |
| 3    | 750   | 600    | 550   | 400   | 300    |
| 4    | 1000  | 800    | 750   | 550   | 450    |
| 5    | 1250  | 1000   | 950   | 750   | 650    |
| 6    | 1500  | 1250   | 1150  | 950   | 850    |
</details>

![](images/ab32fdc34c351a8b6557887d5a977caf063bd9a684866877a8f2c57e4abde770.jpg)

<details>
<summary>line</summary>

| Environment Steps (×10⁶) | Episode Return (Red) | Episode Return (Yellow) | Episode Return (Green) | Episode Return (Purple) |
| ------------------------ | -------------------- | ----------------------- | ---------------------- | ----------------------- |
| 0                        | 1000                 | 1000                    | 1000                   | 1000                    |
| 2                        | 1500                 | 1750                    | 1600                   | 1250                    |
| 4                        | 2000                 | 2250                    | 1800                   | 1500                    |
| 6                        | 2250                 | 2500                    | 2000                   | 1750                    |
| 8                        | 2500                 | 2750                    | 2250                   | 2000                    |
</details>

![](images/b9114d9b5638b3ab59e017b1166238954e6e54792f2c4f68b5612541b3b398ca.jpg)

<details>
<summary>line</summary>

| Environment Steps (×10⁶) | Line 1 | Line 2 | Line 3 | Line 4 | Line 5 |
| ------------------------ | ------ | ------ | ------ | ------ | ------ |
| 0                        | 500    | 500    | 500    | 500    | 500    |
| 1                        | 1200   | 1300   | 1400   | 1500   | 1600   |
| 2                        | 1600   | 1700   | 1800   | 1900   | 2000   |
| 3                        | 1800   | 1900   | 2000   | 2100   | 2200   |
| 4                        | 2000   | 2100   | 2200   | 2300   | 2400   |
| 5                        | 2100   | 2200   | 2300   | 2400   | 2500   |
| 6                        | 2200   | 2300   | 2400   | 2500   | 2600   |
</details>

![](images/ab53e124c5f304a1c54003abe8dc032976ecc6a1a7d2d4a75e8d201151686297.jpg)

<details>
<summary>line</summary>

| Environment Steps (x 10^6) | Red Line | Green Line | Yellow Line | Blue Line |
| -------------------------- | -------- | ---------- | ----------- | --------- |
| 0                          | 1000     | 1000       | 500         | 200       |
| 2                          | 1500     | 1400       | 700         | 300       |
| 4                          | 1800     | 1700       | 900         | 400       |
| 6                          | 2000     | 1900       | 1100        | 500       |
| 8                          | 2100     | 2000       | 1200        | 600       |
</details>

![](images/9b5326de37d701a3beee300bc1047e984dae37453596762a44d8b203de50453c.jpg)

<details>
<summary>line</summary>

| Environment Steps (×10⁶) | Red Line | Green Line | Yellow Line | Purple Line | Blue Line |
| ------------------------ | -------- | ---------- | ----------- | ----------- | --------- |
| 0                        | 1000     | 1000       | 1000        | 100         | 100       |
| 1                        | 1800     | 1700       | 1600        | 800         | 600       |
| 2                        | 2200     | 2100       | 2000        | 1000        | 800       |
| 3                        | 2400     | 2300       | 2200        | 1200        | 1000      |
| 4                        | 2500     | 2400       | 2300        | 1400        | 1200      |
</details>

Figure 4: Episode return for 12 challenging dexterous manipulation tasks. We compare H-InDex with four strong visual representations for motor control, i.e., VC-1 [17], MVP [31], R3M [18], and RRL [27]. Mean of 3 seeds with seed number 0, 1, 2. Shaded area indicates 95% confidence intervals.

# 5.1 Experiment Setup

We evaluate H-InDex on 12 challenging visual dexterous manipulation tasks from Adroit [24] and DexMV [22] respectively, including Hammer, Door, Pen, Pour, Place Inside, and Relocate YCB Objects (7 different objects [3]). Visualization of each task is given in Figure 3 and detailed descriptions are provided in Appendix B. This selection of tasks is the most extensive compared to previous works e.g., DAPG (4 tasks) and DexMV (7 tasks), thereby showcasing the robustness and versatility of H-InDex. The tasks were performed with varying numbers of steps based on their level of complexity. We mainly report the cumulative rewards to show the speed of task completion. The dimension of image observations $\mathbf{o} \in \mathcal{O}$ is $3 \times 224 \times 224$ across all methods. We run experiments on an RTX 3090 GPU; each seed takes roughly 12 hours. Due to the limitation on computation resources, we choose to run 3 seeds for each group of experiments and consistently use 3 seeds with seed numbers 0, 1, 2 to ensure reproducibility. We also observe that H-InDex enjoys a slight variance between seeds, while baselines tend to have a larger variance.

# 5.2 Main Experiments

To demonstrate the effectiveness of H-InDex, we evaluate diverse recent strong visual representations for motor control, including (i) VC-1 $[17]^{1}$ , which trains masked auto-encoders over 5.6M images with over 10,000 GPU-hours and we use the ViT-B $[4]$ model (86M parameters); (ii) MVP $[23, 31]$ , which also uses masked auto-encoders for pre-training and we use the ViT-S $[4]$ model (22M parameters); (iii) R3M $[18]$ , which pre-trains a ResNet-50 (22M parameters) with time contrastive learning and video-language alignment (iv) RRL $[27]$ , which uses the ResNet-50 pre-trained on the ImageNet classification task directly. Due to task differences, we normalize the cumulative rewards based on the highest rewards achieved and present the average scores in Figure 1. We also report the learning curves in Figure 4. We then detail our observations below.

H-InDex emerges as the dominant representation. Across 12 tasks, H-InDex outperforms the recent state-of-the-art representation VC-1 by a 16.8% absolute improvement. Furthermore, H-InDex surpasses RRL, the original state-of-the-art representation in Adroit, by 25.4%. Analyzing

the learning curves, H-InDex demonstrates superior sample efficiency in 10 out of the 12 tasks. In only two tasks, namely relocate mug and relocate mustard bottle, VC-1 exhibits a slight advantage over H-InDex.

ConvNets v.s. ViTs. Among representations utilizing the ResNet-50 architecture (i.e., H-InDex, R3M, RRL), only H-InDex showcases obvious advantages over ViT-based representations. This suggests that with appropriate domain knowledge, ConvNets can still outperform ViTs. Additionally, we notice that ConvNets and ViTs excel in different tasks. For instance, in relocate tomato soup can, VC-1 and MVP achieve returns that are only half of what H-InDex and RRL accomplish. However, in relocate mug, VC-1 performs well. These observations highlight the task-dependent nature of the strengths exhibited by ConvNets and ViTs.

# 5.3 The Effectiveness of 3D Human Hand Prior

The significance of transferring the 3D human hand knowledge into dexterous manipulation is non-negligible. To demonstrate the utility of such 3D human hand prior, we compare our vanilla pre-trained representation i.e., the feature extractor from the FrankMocap hand module [25] (denoted as FrankMocap Hand) with other 4 representative pre-trained models: (i) FrankMocap Body, which is the body estimation module from FrankMocap [25], pre-trained with 3D body pose estimation, (ii) AlphaPose [5], which is a widely-used robust 2D human pose estimation algorithm, (iii) R3M [18], which is pre-trained with time contrastive learning [26] and language-video alignment on Ego4D [6], and (iv) RRL [27], which directly uses the ResNet-50 pre-trained on the ImageNet classification task. All the models use a ResNet-50 architecture and do not use any adaptation, ensuring the fairness of our comparison. We also put H-InDex as the best results achieved for comparison. Results are shown in Figure 5.

We now detail our observations below:

FrankMocap hand v.s. RRL/R3M. Our vanilla representation has significantly outperformed both RRL and R3M without the need for any adaptation.

FrankMocap hand v.s. 3D/2D body-centric representations. Our vanilla 3D hand representation, FrankMocap Hand, demonstrates superior sample efficiency compared to FrankMocap Body and significantly outperforms AlphaPose. This confirms our hypothesis that the 3D human hand prior is more advantageous than both the 3D and 2D human body prior. It is worth noting that AlphaPose, being a whole-body 2D pose estimation model, is also capable of estimating 2D hand poses. The fact that our 3D hand prior outperforms the 2D hand prior in this context further supports its effectiveness. We hypothesize this is because 2D pose estimation does not require deep spatial r H-InDex v.s. FrankMocap hand. Our vanilla hand representation already surpasses all other pre-trained ConvNets in performance. However, by applying our adaptation technique, we can further enhance the sample efficiency of the hand representation, underscoring the significance of adapting the model with in-domain data in a proper way.

![](images/21ddabe0d855687334e699d340bdf9a411ff7c43170553ea3a75e4e043266f3b.jpg)

<details>
<summary>line</summary>

| Environment Steps (×10⁶) | H-InDex | FrankMocap Hand | FrankMocap Body | AlphaPose | R3M | RRL |
| ------------------------ | ------- | --------------- | --------------- | --------- | --- | --- |
| 0.0                      | 0       | 0               | 0               | 0         | 0   | 0   |
| 0.5                      | ~5000   | ~3000           | ~2000           | ~100      | ~50 | ~100 |
| 1.0                      | ~10000  | ~7000           | ~6000           | ~200      | ~100 | ~400 |
| 1.5                      | ~14000  | ~11000          | ~9000           | ~300      | ~150 | ~600 |
| 2.0                      | ~15000  | ~13000          | ~10000          | ~400      | ~200 | ~800 |
</details>

Figure 5: Compare vanilla pre-trained representations with H-InDex.

# 5.4 Ablations

To validate the rationale behind the design choices of H-InDex, we performed a comprehensive set of ablation experiments.

Effects of each stage. Figure 6a provides insights into the contributions of each stage towards the overall efficiency of H-InDex. We refer to RRL as w/o Stage 1,2,3. Significantly, Stage 1 exhibits the most notable enhancement, underscoring the efficacy of human dexterity. Moreover, Stage 2 and Stage 3 also contribute appreciable advancements. In addition, the value of momentum (m) in Stage 3 has a significant influence, as illustrated in Figure 6b. To determine the optimal value, we performed a grid search over $m \in \{0, 0.1, 0.01, 0.001\}$ . This analysis highlights the importance of selecting an appropriate momentum value for achieving optimal performance in Stage 3 of H-InDex.

![](images/2d380be0cb36141aca0ed8190dde0643b56f0d39a2a2fccf5a94075e66162871.jpg)

<details>
<summary>line</summary>

| Environment Steps (×10⁶) | Stage 1 | Stage 1+2 | Stage 1+2+3 | w/o Stage 1,2,3 |
| ------------------------- | ------- | --------- | ----------- | --------------- |
| 0.0                       | 0       | 0         | 0           | 0               |
| 0.5                       | ~2500   | ~2500     | ~2500       | ~2500           |
| 1.0                       | ~7500   | ~7500     | ~7500       | ~7500           |
| 1.5                       | ~12500  | ~12500    | ~12500      | ~7500           |
| 2.0                       | ~15000  | ~15000    | ~15000      | ~7500           |
</details>

(a) Ablation on the effectiveness of our three stages.

![](images/7918a05ab208e5ede94a70dd5a6bdcf349af091723fdae897a63c4ff85a55dd9.jpg)

<details>
<summary>line</summary>

| Environment Steps (×10⁶) | m=0    | m=0.1  | m=0.01 | m=0.001 |
| ------------------------- | ------ | ------ | ------ | ------- |
| 0.0                       | 0      | 0      | 0      | 0       |
| 0.5                       | ~2500  | ~3000  | ~2000  | ~1500   |
| 1.0                       | ~7500  | ~8500  | ~6000  | ~7000   |
| 1.5                       | ~12500 | ~13500 | ~9500  | ~11500  |
| 2.0                       | ~15000 | ~15500 | ~12500 | ~14500  |
</details>

(b) Ablation on momentum $m \in \{0, 0.1, 0.01, 0.001\}$ .

![](images/37a9bc88594d7bea950e047b0e77f3eafc00101167ad115070661dcef0ac07ed.jpg)

<details>
<summary>line</summary>

| Environment Steps (×10⁶) | Stage 2: adapt 0.18% params (ours) | Stage 2: adapt 100% params |
| ------------------------ | ---------------------------------- | -------------------------- |
| 0.0                      | 0                                  | 0                          |
| 0.5                      | ~3000                              | ~1000                      |
| 1.0                      | ~7000                              | ~4000                      |
| 1.5                      | ~12000                             | ~7000                      |
| 2.0                      | ~15000                             | ~10000                     |
</details>

(c) Compare our minimal adaptation in Stage 2 and the full adaptation.

Figure 6: Ablation experiments. We ablate each component of H-InDex and show that each individual part effectively combines to contribute to the overall effectiveness of H-InDex.   
![](images/179e630a3443a0a66a4c1b5ae0b9c88abd8c6b2d58886d9c33d005d8e8b9fd1d.jpg)

<details>
<summary>line</summary>

| Episode | Red Line Value | Blue Line Value |
| ------- | -------------- | --------------- |
| 0.0     | 0              | 0               |
| 0.5     | ~3000          | ~1000           |
| 1.0     | ~8000          | ~4000           |
| 1.5     | ~12000         | ~7000           |
| 2.0     | ~15000         | ~11000          |
</details>

![](images/707c0cbb45770eb3cd7ea9023522168b129d5a446fda66fff8588bbaee790727.jpg)

<details>
<summary>line</summary>

| Episode | Episode Return (Red Line) | Episode Return (Blue Line) |
| ------- | ------------------------- | -------------------------- |
| 0       | 0                         | 0                          |
| 1       | ~75                       | ~60                        |
| 2       | ~125                      | ~90                        |
| 3       | ~175                      | ~130                       |
| 4       | ~200                      | ~160                       |
| 5       | ~225                      | ~180                       |
| 6       | ~240                      | ~200                       |
| 7       | ~250                      | ~210                       |
| 8       | ~255                      | ~220                       |
</details>

![](images/8219becff79cb326a17952b3a2b3cd6ca77a32d50a7da832752bcff7d43ef31d.jpg)

<details>
<summary>line</summary>

| Environment Steps (x10^6) | Episode Return (Red Line) | Episode Return (Blue Line) |
| -------------------------- | ------------------------- | -------------------------- |
| 0                          | 1300                      | 1000                       |
| 2                          | 1800                      | 1050                       |
| 4                          | 2100                      | 1100                       |
| 6                          | 2250                      | 1150                       |
| 8                          | 2300                      | 1200                       |
</details>

![](images/1a2f2d133d667dc235c4b6b3324dfc0661d14442839524bea1f2cde0d0cd84df.jpg)

<details>
<summary>line</summary>

| Step | Door Value |
| ---- | ---------- |
| 0    | 0          |
| 1    | 1000       |
| 2    | 2000       |
| 3    | 2750       |
| 4    | 2900       |
</details>

![](images/38652fed1c7b909d3ecd175de3bdaa2957c99a42648774a2287e44a19b6350de.jpg)

<details>
<summary>line</summary>

| x    | Red Line | Blue Line |
| ---- | -------- | --------- |
| 0    | 0        | 0         |
| 2    | 500      | 0         |
| 4    | 750      | 0         |
| 6    | 1000     | 250       |
| 8    | 1250     | 250       |
</details>

![](images/36dd13d0132ea9469300cffb8f64d116e112bc8f98edaabf30a17577c59e4e96.jpg)

<details>
<summary>line</summary>

| Environment Steps (×10⁶) | Red Line Value | Blue Line Value |
| ------------------------ | -------------- | --------------- |
| 0                        | 500            | 500             |
| 1                        | 1200           | 1100            |
| 2                        | 1600           | 1500            |
| 3                        | 1800           | 1700            |
| 4                        | 2000           | 1900            |
| 5                        | 2100           | 2000            |
| 6                        | 2200           | 2100            |
</details>

Stage 2: adapt 0.18% params (ours)

![](images/0e8c4cac14dd1d9790262863c8a81d0d1f2e62908745a1752f3623ddd37c23f2.jpg)

<details>
<summary>line</summary>

| x    | Red Line | Blue Line |
| ---- | -------- | --------- |
| 0    | 1500     | 200       |
| 1    | 1800     | 250       |
| 2    | 2200     | 300       |
| 3    | 2600     | 350       |
| 4    | 3000     | 400       |
| 5    | 3300     | 450       |
| 6    | 3500     | 500       |
</details>

![](images/3b4d22f9d71515d37d9cc1c27778d23083df9a389863d08f22b40a1aa929fdd4.jpg)

<details>
<summary>line</summary>

| x    | Red Line | Blue Line |
| ---- | -------- | --------- |
| 0.0  | 400      | 400       |
| 0.5  | 1200     | 600       |
| 1.0  | 1300     | 700       |
| 1.5  | 1400     | 800       |
| 2.0  | 1500     | 900       |
</details>

![](images/bb5c411721393726de48c490a3a091281febecb05e279ce51acab9a36e159ab9.jpg)

<details>
<summary>line</summary>

| Environment Steps (×10⁶) | Red Line Value | Blue Line Value |
| ------------------------ | -------------- | --------------- |
| 0                        | 800            | 800             |
| 2                        | 1500           | 900             |
| 4                        | 1800           | 1100            |
| 6                        | 2000           | 1400            |
| 8                        | 2100           | 1700            |
</details>

— Stage 2: adapt 100% params

![](images/3dca69d4ef550467276c8f8fe90d345064b16cfd075d06bfe06b8b2f851f9b50.jpg)

<details>
<summary>line</summary>

| x    | Red Line | Blue Line |
| ---- | -------- | --------- |
| 0    | 5000     | 0         |
| 1    | 10000    | 5000      |
| 2    | 15000    | 10000     |
| 3    | 17500    | 15000     |
| 4    | 19000    | 20000     |
| 5    | 20000    | 25000     |
| 6    | 21000    | 30000     |
| 7    | 22000    | 35000     |
| 8    | 23000    | 40000     |
</details>

![](images/0b2b3f0fb42f77d733425114a606286bf72aa0cab015cc7d8bb9fad22f822984.jpg)

<details>
<summary>line</summary>

| Step | Blue Line | Red Line |
|------|-----------|----------|
| 0    | 0         | 0        |
| 1    | 750       | 250      |
| 2    | 1000      | 500      |
| 3    | 1250      | 750      |
| 4    | 1375      | 1000     |
| 5    | 1475      | 1250     |
| 6    | 1500      | 1250     |
</details>

![](images/1d2b6dbee878f3a7aab159fc62bd8499511844f5fb70e9ff9e3049d4d0dfaea3.jpg)

<details>
<summary>line</summary>

| Environment Steps (×10⁶) | Red Line Value | Blue Line Value |
| ------------------------ | -------------- | --------------- |
| 0                        | 900            | 900             |
| 1                        | 1800           | 900             |
| 2                        | 2100           | 900             |
| 3                        | 2250           | 900             |
| 4                        | 2300           | 900             |
</details>

Figure 7: Ablation on Stage 2 (adapting 0.18% parameters or adapting 100% parameters). We observe that simply finetuning the entire visual representation would lead to sub-optimal results. Instead, H-InDex only adapts the parameters in BatchNorm layers and effectively solves all the tasks.   
Figure 9 provides more ablation results on Stage 3, further supporting the necessity of updating the BatchNorm layers during the training of RL agents.

Adapting 100% parameters v.s. adapting 0.18% parameters in Stage 2. In Stage 2 of H-InDex, we intentionally chose to adapt only 0.18% of the parameters (the affine transformations in BatchNorm layers) in the pre-trained representation. This decision was made to address a specific concern, as depicted in Figure 6c and Figure 7. By altering only the setting for Stage 2 while keeping all other factors constant, we observed that across all tasks, adapting all parameters is not more advantageous. This phenomenon may be attributed to the fact that freezing and finetuning partial parameters help mitigate catastrophic forgetting, which is often caused by full finetuning [1]. In Figure 8, we also show the necessity of our Stage 2. We could conclude from results that correctly finetuning the visual representation is one key to the stable convergence, and not correctly finetuning the model, such as finetuning all the parameters, could be even worse than the frozen model.

Robust visual generalization. One concern of our hand representation is its generalization ability, compared to the vision model pre-trained on large-scale datasets, such as VC-1 $[17]$ . Therefore, we change the background of the training scene to various novel backgrounds, as shown in Figure 12 (see Appendix C) and evaluate VC-1 and H-InDex on the task relocate potted meat can. The

![](images/9e9033f12aea37dae0007b36d2b5f8e7bdd0e0d0435ef7c57cbad5737d953def.jpg)

<details>
<summary>line</summary>

| Episode | Red Line Value | Green Line Value |
| ------- | -------------- | ---------------- |
| 0.0     | 0              | 0                |
| 0.5     | ~3000          | ~1500            |
| 1.0     | ~7000          | ~4000            |
| 1.5     | ~12000         | ~8000            |
| 2.0     | ~15000         | ~13000           |
</details>

![](images/c4993cd374fc4d80a60cd2ec7d30d13fbf39c5756d05cfe8324297d5a4fc73c7.jpg)

<details>
<summary>line</summary>

| x    | Red Line | Green Line |
| ---- | -------- | ---------- |
| 0    | 0        | 0          |
| 1    | 1000     | 1500       |
| 2    | 2000     | 2200       |
| 3    | 2700     | 2800       |
| 4    | 2900     | 2900       |
</details>

![](images/cdbd5e8c8593f11aa2dc4563ab469acd83b00457ef09862522ae931bab11c3dd.jpg)

<details>
<summary>line</summary>

| x    | Red Line | Green Line |
| ---- | -------- | ---------- |
| 0    | 1500     | 1700       |
| 1    | 1800     | 1900       |
| 2    | 2200     | 2100       |
| 3    | 2600     | 2400       |
| 4    | 2900     | 2700       |
| 5    | 3100     | 2900       |
| 6    | 3300     | 3100       |
</details>

![](images/4f1fdd3841611cf89ff30c4304b11db9bf647d46ed73eac559eacbe0c79eb25c.jpg)

<details>
<summary>line</summary>

| x    | Red Line | Green Line |
| ---- | -------- | ---------- |
| 0    | 5000     | 0          |
| 1    | 10000    | 5000       |
| 2    | 15000    | 10000      |
| 3    | 17500    | 12500      |
| 4    | 18750    | 15000      |
| 5    | 19375    | 16250      |
| 6    | 19788    | 17500      |
| 7    | 19939    | 18750      |
| 8    | 20000    | 19375      |
</details>

![](images/b3d436ff2eadce455b818b1fe91cc4c70640862b2aa15b74d4c1c5d994bcf48c.jpg)

<details>
<summary>line</summary>

| Environment Steps (×10⁶) | Episode Return (Red Line) | Episode Return (Green Line) |
| ------------------------- | ------------------------- | --------------------------- |
| 0                         | 0                         | 0                           |
| 2                         | ~100                      | ~80                         |
| 4                         | ~180                      | ~160                        |
| 6                         | ~230                      | ~210                        |
| 8                         | ~250                      | ~240                        |
</details>

![](images/969f17e70873330e91bb8c42f5a7cad5306a062f048abc1e98a72f30574c615a.jpg)

<details>
<summary>line</summary>

| Environment Steps (×10⁶) | Red Line Value | Green Line Value |
| ------------------------ | -------------- | ---------------- |
| 0.0                      | 400            | 400              |
| 0.5                      | 1200           | 1100             |
| 1.0                      | 1350           | 1250             |
| 1.5                      | 1450           | 1300             |
| 2.0                      | 1500           | 1350             |
</details>

![](images/f6045e5080db92ef9055c76567da9b2be5ae56403b2194fa9349a5d171374d72.jpg)

<details>
<summary>line</summary>

| Environment Steps (×10⁶) | Red Line Value | Green Line Value |
| ------------------------ | -------------- | ---------------- |
| 0                        | 0              | 0                |
| 1                        | ~300           | ~250             |
| 2                        | ~600           | ~500             |
| 3                        | ~900           | ~700             |
| 4                        | ~1100          | ~800             |
| 5                        | ~1250          | ~900             |
| 6                        | ~1350          | ~950             |
</details>

![](images/207e72f46df0649c0ccc291a53c62c7a566e491978868251136deb09ede6cefc.jpg)

<details>
<summary>line</summary>

| Environment Steps (×10⁶) | Red Line Value | Green Line Value |
| ------------------------ | -------------- | ---------------- |
| 0                        | 1250           | 1250             |
| 2                        | 1750           | 1500             |
| 4                        | 2000           | 1600             |
| 6                        | 2250           | 1750             |
| 8                        | 2300           | 1900             |
</details>

Stage 2: adapt 0.18% params (ours) w/o. Stage 2

Figure 8: Ablation on Stage 2 (with or without adaptation). We also conduct more experiments to show the necessity of Stage 2. We could observe a consistent improvement across these tasks by applying Stage 2, which only adapts 0.18% parameters of the visual representation.   
![](images/dba70f87c9ba08233aad91a7878d4eb44c720bbc80737eedb25cb7066a304597.jpg)

<details>
<summary>line</summary>

| Episode | Red Line Value | Blue Line Value |
| ------- | -------------- | --------------- |
| 0.0     | 0              | 0               |
| 0.5     | ~3000          | ~1000           |
| 1.0     | ~10000         | ~8000           |
| 1.5     | ~14000         | ~12000          |
| 2.0     | ~15000         | ~14000          |
</details>

![](images/a98f70021ba9a61778c2ed1915bf165987b1b0974852cac3e7f238ddc2c14c7c.jpg)

<details>
<summary>line</summary>

| x  | Red Line | Blue Line |
|----|----------|-----------|
| 0  | 0        | 0         |
| 1  | 50       | 40        |
| 2  | 100      | 90        |
| 3  | 150      | 140       |
| 4  | 200      | 180       |
| 5  | 220      | 200       |
| 6  | 240      | 210       |
| 7  | 250      | 220       |
| 8  | 260      | 230       |
</details>

![](images/d74cdbf5a320ed41032befccc87c9d30610672c8de522b13ca44e681439065cd.jpg)

<details>
<summary>line</summary>

| x    | Red Line | Purple Line |
| ---- | -------- | ----------- |
| 0    | 0        | 0           |
| 2    | 600      | 100         |
| 4    | 800      | 200         |
| 6    | 1000     | 300         |
| 8    | 1100     | 400         |
</details>

![](images/88f01decb41b7c6a46e89f5a0fba1d17e1e03f14e1fbac6e3546fce5b7babf04.jpg)

<details>
<summary>line</summary>

| x    | Red Line | Blue Line |
| ---- | -------- | --------- |
| 0.0  | 400      | 400       |
| 0.5  | 1200     | 800       |
| 1.0  | 1300     | 900       |
| 1.5  | 1400     | 1000      |
| 2.0  | 1500     | 1100      |
</details>

![](images/877ceb3f899f9b788ad8369af6fcd1633a684563e946db6cb0a83e4176f46346.jpg)

<details>
<summary>line</summary>

| Step | Red Line Value | Blue Line Value |
|------|----------------|-----------------|
| 0    | 0              | 0               |
| 1    | ~300           | ~250            |
| 2    | ~600           | ~500            |
| 3    | ~900           | ~700            |
| 4    | ~1100          | ~850            |
| 5    | ~1200          | ~950            |
| 6    | ~1300          | ~1000           |
</details>

![](images/e9115e4921b71c4a991580d5fde9f8a66c5eef56064b329fa2dc7f50600ccee6.jpg)

<details>
<summary>line</summary>

| Step | Stage 3: m>0 | Stage 3: m |
| ---- | ------------ | ---------- |
| 0    | 500          | 500        |
| 1    | 1000         | 700        |
| 2    | 1500         | 800        |
| 3    | 1800         | 850        |
| 4    | 2000         | 900        |
| 5    | 2100         | 950        |
| 6    | 2150         | 1000       |
</details>

![](images/9b54d9c8cabd3060704f9ecc6f35f971c8ecd44dd4459cf439299763c7e63630.jpg)

<details>
<summary>line</summary>

| x | Red Line | Blue Line |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | 1000 | 800 |
| 2 | 1500 | 1200 |
| 3 | 1700 | 1400 |
| 4 | 1900 | 1600 |
| 5 | 2000 | 1700 |
| 6 | 2100 | 1800 |
| 7 | 2150 | 1900 |
| 8 | 2200 | 2000 |
</details>

Figure 9: Ablation on Stage 3 (momentum m > 0 or m = 0). We observe that our Stage 3 contributes greatly to some specific tasks, such as relocate mustard bottle, and for some tasks like hammer, tuning this parameter only results in a slightly faster convergence. For tasks not shown here, we all use m = 0, since H-InDex with Stage 1 and Stage 2 has been strong enough.

results given in Table 3 (see Appendix C) show that H-InDex could handle the changed background better than VC-1. We also see the consistent performance drop across all scenes, emphasizing the importance of visual generalization.

Visualization of self-supervised keypoint detection in Stage 2. In Stage 2, a self-supervised keypoint detection objective is employed to fine-tune a minimal percentage (0.18%) of parameters in the pre-trained model. The visualization results, as shown in Figure 10, demonstrate the successful detection of keypoints. This observation highlights the pre-trained model's capability to effectively allocate attention to the hand and objects depicted in the images, even with the adaptation of only a small subset of parameters.

Visualization of affine transformations adaptation in Stage 2. The significance of Stage 2 in H-InDex is evident from Figure 6a. To gain deeper insights into this phenomenon, we visualize the distribution of the adapted parameters, specifically the affine transformations in BatchNorm layers. We accomplish this by fitting a Gaussian distribution. Figure 11 presents the visualization results, highlighting an interesting trend. For the shallow layers, the distributions of the adapted models closely resemble those of the pre-trained models. However, as we move deeper into the layers, noticeable differences emerge. We attribute this disparity to the fact that dissimilarities between human hands and robot hands extend beyond low-level features like color and texture. Instead,

![](images/c8c9bd042599d6ea7a4d97a716c49a0b4c38dc26484c72b5fb55b04ef2f69bb8.jpg)

<details>
<summary>text_image</summary>

Hammer
Pen
Pour
Relocate Objects
</details>

Figure 10: Visualization of our self-supervised keypoint detection. We select four tasks here and mark the detected keypoints on images with $\star$ red stars. We observe that these keypoints consistently mark the dynamic regions of images. Full videos are available on yanjieze.com/H-InDex.

![](images/7eb0edb4bb77a213f43d088e9856e4ac2566cf30ee6bf052f7f8f30bc5187f92.jpg)

<details>
<summary>line</summary>

| Layer | Human | Robot |
|-------|-------|-------|
| Layer1 BN1 | 0.0 | 0.0 |
| Layer1 BN2 | 0.2 | 0.2 |
| Layer1 BN3 | 0.25 | 0.25 |
| Layer2 BN1 | 0.0 | 0.0 |
| Layer2 BN2 | 0.2 | 0.2 |
| Layer2 BN3 | 0.25 | 0.25 |
| Layer3 BN1 | 0.2 | 0.2 |
| Layer3 BN2 | 0.25 | 0.25 |
| Layer3 BN3 | 0.2 | 0.2 |
| Layer4 BN1 | 0.0 | 0.0 |
| Layer4 BN2 | 0.2 | 0.2 |
| Layer4 BN3 | 0.25 | 0.25 |
</details>

Figure 11: Visualization of affine transformation adaptation in Stage 2. We fit a Gaussian distribution for parameters of affine transformations. We omit the X-axis and Y-axis here for simplicity, Human represents the original pre-trained model and robot represents the adapted representation in different tasks. It is observed that deep layers are subjected to large distribution shifts.

they encompass higher-level features such as dynamics and structure $[2, 32, 35]$ . This observation underscores the importance of our adaptation approach, as it effectively addresses the variations in both low-level and high-level features, facilitating the success of H-InDex.

# 6 Conclusion

In this study, we introduce H-InDex, a visual reinforcement learning framework that leverages hand-informed visual representations to tackle complex dexterous manipulation tasks effectively. H-InDex outperforms other recent state-of-the-art representations in a range of 12 tasks, including six kinds of manipulation skills. The effectiveness of H-InDex can be attributed to its three-stage approach, wherein Stage 1 incorporates a pre-trained 3D human hand representation and Stage 2 and Stage 3 focus on careful in-domain adaptation with only 0.36% parameters updated. These stages collectively contribute to the successful preservation and utilization of the human hand prior knowledge.

It is also important to acknowledge some limitations of our work. We did not investigate the generalization capabilities of H-InDex, particularly in scenarios involving the grasping of novel objects. Our future work aims to address this limitation and enhance H-InDex's generalization capabilities for real-world applications. In addition, we find that our Stage 3 is surprisingly effective in some specific tasks like relocate mustard bottle, while we have not given a theoretical understanding of such phenomena. We consider this problem as a possible future direction.

# Acknowledgment

This work is supported by National Key R&D Program of China (2022ZD0161700).

# References

[1] Craig Atkinson, Brendan McCane, Lech Szymanski, and Anthony Robins. Pseudo-rehearsal: Achieving deep reinforcement learning without catastrophic forgetting. Neurocomputing, 2021. 4, 8   
[2] David Bau, Bolei Zhou, Aditya Khosla, Aude Oliva, and Antonio Torralba. Network dissection: Quantifying interpretability of deep visual representations. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6541–6549, 2017. 10   
[3] Berk Calli, Aaron Walsman, Arjun Singh, Siddhartha Srinivasa, Pieter Abbeel, and Aaron M Dollar. Benchmarking in manipulation research: The ycb object and model set and benchmarking protocols. arXiv, 2015. 6, 14   
[4] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv, 2020. 2, 6   
[5] Hao-Shu Fang, Jiefeng Li, Hongyang Tang, Chao Xu, Haoyi Zhu, Yuliang Xiu, Yong-Lu Li, and Cewu Lu. Alphapose: Whole-body regional multi-person pose estimation and tracking in real-time. PAMI, 2022. 7   
[6] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In CVPR, 2022. 2, 7   
[7] Nicklas Hansen, Yixin Lin, Hao Su, Xiaolong Wang, Vikash Kumar, and Aravind Rajeswaran. Modem: Accelerating visual model-based reinforcement learning with demonstrations. ICLR, 2023. 2, 3   
[8] Nicklas Hansen, Zhecheng Yuan, Yanjie Ze, Tongzhou Mu, Aravind Rajeswaran, Hao Su, Huazhe Xu, and Xiaolong Wang. On pre-training for visuo-motor control: Revisiting a learning-from-scratch baseline. ICML, 2023. 2   
[9] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, 2022. 2   
[10] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016. 2, 4, 13   
[11] Tomas Jakab, Ankush Gupta, Hakan Bilen, and Andrea Vedaldi. Unsupervised learning of object landmarks through conditional image generation. NeurIPS, 2018. 4, 13   
[12] Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In ECCV, 2016. 5   
[13] Sham M Kakade. A natural policy gradient. NeurIPS, 2001. 3   
[14] Tejas D Kulkarni, Ankush Gupta, Catalin Ionescu, Sebastian Borgeaud, Malcolm Reynolds, Andrew Zisserman, and Volodymyr Mnih. Unsupervised learning of object keypoints for perception and control. NeurIPS, 2019. 4, 13   
[15] Yizhuo Li, Miao Hao, Zonglin Di, Nitesh Bharadwaj Gundavarapu, and Xiaolong Wang. Test-time personalization with a transformer for human pose estimation. NeurIPS, 2021. 4, 13   
[16] Yecheng Jason Ma, Shagun Sodhani, Dinesh Jayaraman, Osbert Bastani, Vikash Kumar, and Amy Zhang. Vip: Towards universal visual reward and representation via value-implicit pre-training. ICLR, 2023. 4   
[17] Arjun Majumdar, Karmesh Yadav, Sergio Arnaud, Yecheng Jason Ma, Claire Chen, Sneha Silwal, Aryan Jain, Vincent-Pierre Berges, Pieter Abbeel, Jitendra Malik, Dhruv Batra, Yixin Lin, Oleksandr Maksymets, Aravind Rajeswaran, and Franziska Meier. Where are we in the search for an artificial visual cortex for embodied intelligence? 2023. 1, 2, 3, 4, 6, 8, 14, 15, 16   
[18] Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, and Abhinav Gupta. R3m: A universal visual representation for robot manipulation. arXiv, 2022. 1, 2, 4, 6, 7, 15   
[19] Simone Parisi, Aravind Rajeswaran, Senthil Purushwalkam, and Abhinav Gupta. The unsurprising effectiveness of pre-trained vision models for control. ICML, 2022. 2, 4   
[20] Austin Patel, Andrew Wang, Ilija Radosavovic, and Jitendra Malik. Learning to imitate object interactions from internet videos. arXiv, 2022. 3

[21] Ivaylo Popov, Nicolas Heess, Timothy Lillicrap, Roland Hafner, Gabriel Barth-Maron, Matej Vecerik, Thomas Lampe, Yuval Tassa, Tom Erez, and Martin Riedmiller. Data-efficient deep reinforcement learning for dexterous manipulation. arXiv, 2017. 2   
[22] Yuzhe Qin, Yueh-Hua Wu, Shaowei Liu, Hanwen Jiang, Ruihan Yang, Yang Fu, and Xiaolong Wang. Dexmv: Imitation learning for dexterous manipulation from human videos. In ECCV, 2022. 2, 3, 6, 13   
[23] Ilija Radosavovic, Tete Xiao, Stephen James, Pieter Abbeel, Jitendra Malik, and Trevor Darrell. Real-world robot learning with masked visual pre-training. In CoRL, 2022. 2, 6   
[24] Aravind Rajeswaran, Vikash Kumar, Abhishek Gupta, Giulia Vezzani, John Schulman, Emanuel Todorov, and Sergey Levine. Learning complex dexterous manipulation with deep reinforcement learning and demonstrations. RSS, 2018. 2, 3, 6, 13   
[25] Yu Rong, Takaaki Shiratori, and Hanbyul Joo. Frankmocap: Fast monocular 3d hand and body motion capture by regression and integration. arXiv, 2020. 1, 3, 4, 7   
[26] Pierre Sermanet, Corey Lynch, Yevgen Chebotar, Jasmine Hsu, Eric Jang, Stefan Schaal, Sergey Levine, and Google Brain. Time-contrastive networks: Self-supervised learning from video. In ICRA, 2018. 2, 7   
[27] Rutav Shah and Vikash Kumar. Rrl: Resnet as representation for reinforcement learning. ICML, 2021. 1, 2, 3, 6, 7, 13, 15, 16   
[28] Kenneth Shaw, Shikhar Bahl, and Deepak Pathak. Videodex: Learning dexterity from internet videos. CoRL, 2022. 3   
[29] Aravind Sivakumar, Kenneth Shaw, and Deepak Pathak. Robotic telekinesis: learning a robotic hand imitator by watching humans on youtube. RSS, 2022. 3   
[30] Che Wang, Xufang Luo, Keith Ross, and Dongsheng Li. Vrl3: A data-driven framework for visual deep reinforcement learning. arXiv, 2022. 2, 3   
[31] Tete Xiao, Ilija Radosavovic, Trevor Darrell, and Jitendra Malik. Masked visual pre-training for motor control. arXiv, 2022. 1, 2, 3, 4, 6, 15   
[32] Jason Yosinski, Jeff Clune, Anh Nguyen, Thomas Fuchs, and Hod Lipson. Understanding neural networks through deep visualization. arXiv preprint arXiv:1506.06579, 2015. 10   
[33] Yanjie Ze, Nicklas Hansen, Yinbo Chen, Mohit Jain, and Xiaolong Wang. Visual reinforcement learning with self-supervised 3d representations. RA-L, 2023. 2   
[34] Yanjie Ze, Ge Yan, Yueh-Hua Wu, Annabella Macaluso, Yuying Ge, Jianglong Ye, Nicklas Hansen, Li Erran Li, and Xiaolong Wang. Gnfactor: Multi-task real robot learning with generalizable neural feature fields. CoRL, 2023. 2   
[35] Matthew D Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In ECCV, 2014. 10

# Appendix

# A Implementation Details

Codebase. Our major codebase is built upon the official implementation of RRL $[27]$ , which is publicly available on https://github.com/facebookresearch/RRL and includes the Adroit manipulation tasks $[24]$ . The DexMV $[22]$ tasks are from the official code https://github.com/yzqin/dexmv-sim. All the visual representations in our work are also available online, including RRL (pre-trained ResNet-50, provided in PyTorch officially), R3M (https://github.com/facebookresearch/r3m), MVP (https://github.com/ir413/mvp), VC-1 (https://github.com/facebookresearch/eai-vc), and FrankMocap (https://github.com/facebookresearch/frankmocap). This ensures the good reproducibility of our work. Our official code is released on https://github.com/YanjieZe/H-InDex.

Network architecture for H-InDex. The architecture employed by H-InDex is based on ResNet-50 [10], referred to as $h_{\theta}$ . In the initial stage (Stage 1), $h_{\theta}$ takes as input a $224 \times 224$ RGB image and processes it to generate a compact vector of size 2048. In Stage 2, we modify $h_{\theta}$ by removing the average pooling layer in the final layer, resulting in the image being decoded into a feature map with dimensions $7 \times 7 \times 2048$ . Moving on to Stage 3, $h_{\theta}$ once again produces a compact vector of size 2048, while simultaneously updating the statistics within the BatchNorm layers using the exponential moving average operation.

Implementation details for Stage 2. Our implementation strictly follows the previous work that also uses the self-supervised keypoint detection as objective $[11,14,15]$ . We give a PyTorch-style overview of the learning pipeline below and refer to $[11]$ for more implementation details. Notably, the visual representation $h_{\theta}$ (24M) contains the majority of parameters, while all other modules in the pipeline maintain a parameter count ranging from 1M to 3M. We use 50 demonstration videos as training data for each task and train 100k iterations to ensure convergence with learning rate $1 \times 10^{-4}$ . One of our core contributions is to only adapt the parameters in BatchNorm layers in $h_{\theta}$ , and we emphasize that the learning objective is not our contribution, as it has been well explored in $[11,14,15]$ .

```python
for _ in range(num_iters):
    # sample data
    source_view, target_view = next(data_iter) # 3x224x224

    # self-supervised keypoint-based reconstruction
    # h_theta is our visual representation
    feature_map = h_theta(target_view) # -> 7x7x2048
    keypoint_feat = keypoint_encoder(feature_map) # -> 30x56x56
    keypoint_feat = up_sampler(keypoint_feature) # -> 256x28x28
    appearance_feat = appearance_encoder(source_view) # -> 256x28x28
    target_view_recon = image_decoder([keypoint_feat, appearance_feat]) # -> 3x224x224

    # compute loss
    loss = perceptual_loss(target_view, target_view_recon)

    # compute gradient and update model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 
```

# B Task Descriptions

In this section, we briefly introduce our tasks. We use an Adroit dexterous hand for manipulation tasks. The task design follows Adroit $[24]$ and DexMV $[22]$ . Visualizations of task trajectories are available at yanjieze.com/H-InDex.

Hammer (Adroit). It requires the robot hand to pick up the hammer on the table and use the hammer to hit the nail.

Door (Adroit). It requires the robot hand to open the door on the table.

Pen (Adroit). It requires the robot hand to orient the pen to the target orientation.

Pour (DexMV). It requires the robot hand to reach the mug and pour the particles inside into a container.

Place inside (DexMV). It requires the robot hand to place the object on the table into the mug.

Relocate YCB objects [3] (DexMV). It requires the robot hand to pick up the object on the table to the target location. The objects in our tasks include foam brick, box, mug, mustard bottle, tomato soup can, and potted meat can.

# C Visual Generalization

One concern of our hand representation is its generalization ability, compared to the vision model pre-trained on large-scale datasets, such as VC-1 $[17]$ . Therefore, we change the background of the training scene to various novel backgrounds, as shown in Figure 12 and evaluate VC-1 and H-InDex on the task relocate potted meat can. The results given in Table 3 show that H-InDex could handle the changed background better than VC-1. We also see the consistent performance drop across all scenes, emphasizing the importance of visual generalization.

![](images/12a10236b1302f536bd39e0ef28e0cbdaf6018de0af69db97c15af18976f4bad.jpg)

<details>
<summary>natural_image</summary>

Grid of 3D-rendered panels with icons of a robotic hand and green objects, no text or symbols present
</details>

Figure 12: Various backgrounds for visual generalization. The first image shows the training scene and the rest 9 images show the novel scene.

Table 1: Scores for generalization to unseen backgrounds on relocate potted meat can task. We evaluate VC-1 and H-InDex with 20 episodes for each seed. 

<table><tr><td>Scene ID / Method</td><td>VC-1 [17]</td><td>H-InDex</td></tr><tr><td>Origin</td><td>2391.74±602.83</td><td>2240.37±85.45</td></tr><tr><td>1</td><td>896.28±1006.55</td><td>915.95±922.65</td></tr><tr><td>2</td><td>603.26±920.48</td><td>771.28±793.51</td></tr><tr><td>3</td><td>451.36±839.45</td><td>578.42±764.09</td></tr><tr><td>4</td><td>360.21±772.64</td><td>472.66±715.03</td></tr><tr><td>5</td><td>300.02±718.05</td><td>393.32±676.41</td></tr><tr><td>6</td><td>256.80±673.16</td><td>340.07±639.68</td></tr><tr><td>7</td><td>224.21±635.56</td><td>298.20±608.54</td></tr><tr><td>8</td><td>226.59±610.58</td><td>265.82±581.03</td></tr><tr><td>9</td><td>214.30±581.76</td><td>239.60±556.80</td></tr><tr><td>Average</td><td>392.56</td><td>475.04</td></tr></table>

# D Main Experiments (ConvNets Only)

In our primary experimental analysis, we conduct a comprehensive comparison of five visual representations, with three of them being ConvNets, including our method. Figure 13 presents an isolated

demonstration of the comparison among the ConvNets. Notably, our method H-InDex exhibits superior performance in comparison to the other ConvNets.

![](images/9881ef035df435a272de7cf12aeefa78813107c277abced5cab447659c1c9e11.jpg)  
Figure 13: Episode return for 12 challenging dexterous manipulation tasks. Mean of 3 seeds with seed number 0, 1, 2. Shaded area indicates 95% CIs.

# E Success Rates in Main Experiments

We present the success rates of our six task categories as in Table 3. Regarding the hammer task, it is evident that both H-InDex and VC-1 exhibit success rates near 100%. However, a notable disparity arises when considering episode returns, indicating the varying degrees of task execution proficiency even among successful agents.

Table 2: Success rates for main experiments. Highest success rates for each task are marked with bold fonts. The success rates only reflect whether the task is finished but do not reflect how fast the task is finished. H-InDex still dominates other methods. 

<table><tr><td>Task name / Method</td><td>RRL [27]</td><td>R3M [18]</td><td>MVP [31]</td><td>VC-1 [17]</td><td>H-InDex</td></tr><tr><td>Hammer (2M)</td><td> $89 \pm 15$ </td><td> $24 \pm 21$ </td><td> $83 \pm 11$ </td><td> $97 \pm 3$ </td><td> $100 \pm 0$ </td></tr><tr><td>Door (4M)</td><td> $92 \pm 1$ </td><td> $99 \pm 2$ </td><td> $100 \pm 0$ </td><td> $99 \pm 2$ </td><td> $96 \pm 5$ </td></tr><tr><td>Pen (8M)</td><td> $78 \pm 4$ </td><td> $58 \pm 6$ </td><td> $80 \pm 4$ </td><td> $81 \pm 2$ </td><td> $90 \pm 2$ </td></tr><tr><td>Pour (8M)</td><td> $38 \pm 33$ </td><td> $0 \pm 0$ </td><td> $23 \pm 38$ </td><td> $67 \pm 29$ </td><td> $99 \pm 2$ </td></tr><tr><td>Place inside (6M)</td><td> $70 \pm 50$ </td><td> $1 \pm 2$ </td><td> $99 \pm 2$ </td><td> $98 \pm 4$ </td><td> $97 \pm 6$ </td></tr><tr><td>Relocate large clamp (8M)</td><td> $33 \pm 31$ </td><td> $0 \pm 0$ </td><td> $47 \pm 27$ </td><td> $23 \pm 21$ </td><td> $50 \pm 45$ </td></tr><tr><td>Relocate foam brick (2M)</td><td> $87 \pm 11$ </td><td> $42 \pm 37$ </td><td> $48 \pm 46$ </td><td> $44 \pm 49$ </td><td> $86 \pm 10$ </td></tr><tr><td>Relocate box (6M)</td><td> $94 \pm 5$ </td><td> $45 \pm 24$ </td><td> $48 \pm 50$ </td><td> $49 \pm 50$ </td><td> $85 \pm 14$ </td></tr><tr><td>Relocate mug (2M)</td><td> $100 \pm 0$ </td><td> $82 \pm 2$ </td><td> $54 \pm 51$ </td><td> $74 \pm 44$ </td><td> $100 \pm 0$ </td></tr><tr><td>Relocate mustard bottle (2M)</td><td> $100 \pm 0$ </td><td> $82 \pm 8$ </td><td> $100 \pm 0$ </td><td> $99 \pm 2$ </td><td> $99 \pm 2$ </td></tr><tr><td>Relocate tomato soup can (2M)</td><td> $97 \pm 3$ </td><td> $18 \pm 31$ </td><td> $6 \pm 10$ </td><td> $30 \pm 52$ </td><td> $99 \pm 2$ </td></tr><tr><td>Relocate potted meat can (2M)</td><td> $97 \pm 6$ </td><td> $56 \pm 15$ </td><td> $69 \pm 48$ </td><td> $88 \pm 21$ </td><td> $94 \pm 5$ </td></tr><tr><td>Average</td><td>81.3</td><td>42.3</td><td>63.1</td><td>70.8</td><td>91.3</td></tr></table>

Table 3: Success rates and scores for generalization to unseen backgrounds on relocate potted meat can task. We evaluate VC-1 and H-InDex with 20 episodes for each seed. 

<table><tr><td>Scene ID / Method</td><td>VC-1 [17] (success rate)</td><td>VC-1 [17] (score)</td><td>H-InDex (success rate)</td><td>H-InDex (score)</td></tr><tr><td>1</td><td>38±43</td><td>896.28±1006.55</td><td>48±48</td><td>915.95±922.65</td></tr><tr><td>2</td><td>26±39</td><td>603.26±920.48</td><td>32±46</td><td>771.28±793.51</td></tr><tr><td>3</td><td>19±36</td><td>451.36±839.45</td><td>24±42</td><td>578.42±764.09</td></tr><tr><td>4</td><td>15±33</td><td>360.21±772.64</td><td>19±39</td><td>472.66±715.03</td></tr><tr><td>5</td><td>13±31</td><td>300.02±718.05</td><td>16±36</td><td>393.32±676.41</td></tr><tr><td>6</td><td>11±29</td><td>256.80±673.16</td><td>14±34</td><td>340.07±639.68</td></tr><tr><td>7</td><td>10±27</td><td>224.21±635.56</td><td>12±32</td><td>298.20±608.54</td></tr><tr><td>8</td><td>10±26</td><td>226.59±610.58</td><td>11±30</td><td>265.82±581.03</td></tr><tr><td>9</td><td>9±25</td><td>214.30±581.76</td><td>10±29</td><td>239.60±556.80</td></tr><tr><td>Average</td><td>16.8</td><td>392.56</td><td>20.7</td><td>475.04</td></tr></table>

# F Hyperparameters

We categorize hyperparameters into task-specific ones (Table 4) and task-agnostic ones (Table 5). Across all baselines, all the hyperparameters are shared except the momentum m, which is only used in our algorithm. All the hyperparameters for policy learning are the same as RRL [27]. This ensures the comparison between different representations is fair.

Our exploration of the momentum m in Table 4 has been limited to a specific set of values, namely $\{0, 0.1, 0.01, 0.001\}$ , through the use of a grid search technique, due to the limitation on computation resources. It is observed that carefully tuning m could take more benefits.

Table 4: Task-specific hyperparameters. 

<table><tr><td>Task name / Variable</td><td>Momentum m</td><td>Demonstrations</td><td>Training steps (M)</td><td>Episode length</td></tr><tr><td>Hammer</td><td>0.1</td><td>25</td><td>2</td><td>200</td></tr><tr><td>Door</td><td>0.0</td><td>25</td><td>4</td><td>200</td></tr><tr><td>Pen</td><td>0.0</td><td>25</td><td>6</td><td>100</td></tr><tr><td>Pour</td><td>0.0</td><td>50</td><td>8</td><td>200</td></tr><tr><td>Place inside</td><td>0.001</td><td>50</td><td>8</td><td>200</td></tr><tr><td>Relocate large clamp</td><td>0.01</td><td>50</td><td>8</td><td>100</td></tr><tr><td>Relocate foam brick</td><td>0.01</td><td>25</td><td>2</td><td>100</td></tr><tr><td>Relocate box</td><td>0.001</td><td>25</td><td>6</td><td>100</td></tr><tr><td>Relocate mug</td><td>0.0</td><td>25</td><td>8</td><td>100</td></tr><tr><td>Relocate mustard bottle</td><td>0.001</td><td>25</td><td>6</td><td>100</td></tr><tr><td>Relocate tomato soup can</td><td>0.01</td><td>25</td><td>8</td><td>100</td></tr><tr><td>Relocate potted meat can</td><td>0.0</td><td>25</td><td>4</td><td>100</td></tr></table>

Table 5: Task-agnostic hyperparameters. 

<table><tr><td>Variable</td><td>Value</td></tr><tr><td>Dimension of image observations</td><td>224 × 224 × 3</td></tr><tr><td>Dimension of robot states</td><td>30</td></tr><tr><td>Dimension of actions</td><td>30</td></tr><tr><td>Hidden dimensions of policy π</td><td>256, 256</td></tr><tr><td>BC learning rate</td><td>0.001</td></tr><tr><td>BC epochs</td><td>5</td></tr><tr><td>BC batch size</td><td>32</td></tr><tr><td>RL learning rate</td><td>0.001</td></tr><tr><td>Number of trajectories for one step</td><td>100</td></tr><tr><td>VF batch size</td><td>64</td></tr><tr><td>VF epochs</td><td>2</td></tr><tr><td>RL step size</td><td>0.05</td></tr><tr><td>RL gamma</td><td>0.995</td></tr><tr><td>RL gae</td><td>0.97</td></tr></table>