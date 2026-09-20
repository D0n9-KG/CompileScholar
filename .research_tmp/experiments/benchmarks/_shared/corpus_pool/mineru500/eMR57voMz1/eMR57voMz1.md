# Diversify & Conquer: Outcome-directed Curriculum RL via Out-of-Distribution Disagreement

Daesol Cho, Seungjae Lee, and H. Jin Kim

Seoul National University

Automation and Systems Research Institute (ASRI)

Artificial Intelligence Institute of Seoul National University (AIIS)

dscho1234@snu.ac.kr, ysz0301@snu.ac.kr, hjinkim@snu.ac.kr

# Abstract

Reinforcement learning (RL) often faces the challenges of uninformed search problems where the agent should explore without access to the domain knowledge such as characteristics of the environment or external rewards. To tackle these challenges, this work proposes a new approach for curriculum RL called Diversify for Disagreement & Conquer (D2C). Unlike previous curriculum learning methods, D2C requires only a few examples of desired outcomes and works in any environment, regardless of its geometry or the distribution of the desired outcome examples. The proposed method performs diversification of the goal-conditional classifiers to identify similarities between visited and desired outcome states and ensures that the classifiers disagree on states from out-of-distribution, which enables quantifying the unexplored region and designing an arbitrary goal-conditioned intrinsic reward signal in a simple and intuitive way. The proposed method then employs bipartite matching to define a curriculum learning objective that produces a sequence of well-adjusted intermediate goals, which enable the agent to automatically explore and conquer the unexplored region. We present experimental results demonstrating that D2C outperforms prior curriculum RL methods in both quantitative and qualitative aspects, even with the arbitrarily distributed desired outcome examples.

# 1 Introduction

Reinforcement learning (RL) has great potential for the automated learning of behaviors, but the process of learning individual useful behavior can be time-consuming due to the significant amount of experience required for the agent. Furthermore, in its general usage, RL often involves solving a challenging uninformed search problem, where informative rewards or desired behaviors are rarely observed. While there are some techniques that can alleviate the exploration burden such as reward-shaping $[30]$ or preference-based reward $[21, 6]$ , they often require significant domain knowledge or human intervention. This makes it difficult to utilize RL directly for problems that require challenging exploration, especially in many domains of practical significance. Thus, it is becoming increasingly crucial to tackle these challenges from the algorithmic level by developing RL agents that can learn autonomously with minimal supervision.

One potential approach is a curriculum learning algorithm. It involves proposing a carefully designed sequence of curriculum tasks or goals for the agent to accomplish, with each step building upon the previous one to gradually progress the curriculum. By doing so, it enables the agent to automatically explore the environment and improve the capability in a structured way. Previous studies primarily involve a mechanism to adjust the curriculum distribution by maximizing its entropy within the explored region $[35]$ , or taking into account the difficulty level $[43, 11]$ , or learning progress of the agent $[36]$ . However, efficient desired outcome-directed exploration is not available under these

![](images/d8399ad4bde5362afe786fcbd94dd544f0e0da88c850085dedd1a339266dfba2.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Explored states (label 0)"] --> B["MLP layers"]
    C["Desired outcome states (label 1)"] --> B
    D["Dsource"] --> B
    E["Dtarget"] --> B
    B --> F["Diversify with multiple valid classifiers"]
    F --> G["Unlabeled target data"]
    F --> H["Label 0 source data"]
    F --> I["Unlabeled target data"]
    G --> J["Label 1 source data"]
    H --> K["Label 0 source data"]
    I --> L["Unlabeled target data"]
    J --> M["Quantify unexplored areas via disagreement"]
    K --> M
    L --> M
    M --> N["Curriculum proposal"]
    style N fill:#f9f9f9,stroke:#333
    style M fill:#e6f7ff,stroke:#333
```
</details>

Figure 1: D2C trains a set of classifiers with labeled source data while diversifying their outputs on unlabeled target data (red: predicted label 0, blue: predicted label 1). Then, it proposes curriculum goals based on the diversified classifier's disagreement and similarity-to-desired outcome.

frameworks as these approaches do not have converging curriculum objectives, resulting in a naive search for unseen states.

Providing the agent with examples of successful outcomes can make the RL problem more manageable, as opposed to letting the agent explore without a particular objective and with an undirected reward. Such examples can offer considerable guidance on how to achieve a task if the agent can estimate the similarities between the desired outcome examples and visited states. In the realm of desired outcome-directed RL, previous studies try to maximize the probability of reaching desired outcomes $[12, 41, 8]$ . However, these approaches do not have tools for explicitly quantifying an unexplored region, leading the agent to settle for reaching an initially discovered desired outcome example, rather than continuing to explore further to find another example. Other works try to minimize the distance between the generated curriculum distribution and the desired outcome distribution to propose intermediate task goals $[37, 18]$ . But, these approaches have mainly been confined to problems that do not involve significant exploration challenges because they rely on the assumption that the Euclidean distance metric can represent the geodesic interpolation between distributions. This assumption is not universally applicable to all environmental geometries, making these methods less versatile than desired.

Therefore, it is necessary to develop an algorithm that enables the agent to automatically perform the outcome-directed exploration by generating a sequence of curriculum goals, which can be applied to arbitrary geometry and distribution of the given desired outcome examples. To do so, we propose Diversify for Disagreement & Conquer (D2C), which only requires desired outcome examples and does not require prior domain knowledge such as 1) the geometry of the environment 2) or the number of modes or distribution of the desired outcomes, or 3) external reward from the environment. Specifically, D2C involves diversifying the goal-conditioned classifiers to identify the similarities between the visited states and the desired outcome states. This is accomplished by ensuring that the outputs of each classifier disagree on unseen states, which not only allows determining the unexplored frontier region but also provides an arbitrary goal-conditioned intrinsic reward. Based on such conditional classifiers, we propose to employ bipartite matching to define a straightforward and easy-to-comprehend curriculum learning objective, which produces a range of well-adjusted curriculum goals that interpolate between the initial state distribution and arbitrarily distributed desired outcome states for enabling the agent to conquer the unexplored region.

To sum up, our work makes the following key contributions.

- We propose a new outcome-directed curriculum RL method that only needs a few arbitrarily distributed desired outcome examples and eliminates the need for external reward.   
- To the best of our knowledge, D2C is the first algorithm for curriculum RL that allows for automatic progress in any environment without being restricted by its geometry or the desired outcome distribution by proposing diversified conditional classifiers.   
- In various goal-conditioned RL experiments, our method consistently outperforms the previous curriculum RL methods through precisely calibrated guidance toward the desired outcome states in both quantitative and qualitative aspects.

# 2 Related Works

Despite various attempts to improve exploration in RL, it remains a difficult issue that has not been fully solved. Some previous works for exploration have suggested methods based on information theoretical approaches $[9, 39, 50, 20, 17, 27]$ , maximizing the state visitation distribution's entropy $[47, 25, 26]$ , counting the state visitation $[3, 31]$ , utilizing similarity or curiosity $[33, 45]$ , and quantifying uncertainty through prediction models $[4, 34]$ . Other approaches provide a curriculum to allow the agent to explore the environment through intermediate tasks. Curricula are usually created by adjusting the distribution of goals to cover new and unexplored areas. It is achieved by considering an auxiliary objective such as entropy $[35]$ or disagreement between the model ensembles $[49, 15, 28]$ or difficulty level of the curriculum $[11, 43]$ , regret $[16]$ , and learning progress $[36]$ . However, these methods only focus on visiting diverse frontier states or do not provide a mechanism to converge toward the desired outcome distribution. In contrast, our method enables more efficient outcome-directed exploration via curriculum proposal with an objective to converge rather than simply exploring various frontier states, only requiring a few desired outcome examples.

Assuming access to desired outcome samples or distribution, some prior methods try to accomplish the desired outcome states by maximizing the probability of reaching these states $[12, 41, 8]$ . However, they lack a mechanism for quantifying an under-explored region and synthesizing the knowledge acquired from the agent's experiences into versatile policies that can accomplish novel test goals. Some algorithms generate curricula as an interpolation between the distribution of desired target tasks and auxiliary tasks $[37, 18]$ , but they still rely on the Euclidean distance metric, which is insufficient to handle arbitrary geometric structures. There exists a work that addresses geometry-agnostic curriculum generation using the given desired outcome states, similar to our approach $[5]$ . But, it requires carefully tuned Wasserstein distance estimation that depends on the optimality of the agent. It leads to inconsistent estimation before the convergence, resulting in numerically unstable training, while our method does not have such dependence. Also, it adopts a meta-learning-based technique $[10, 23]$ that requires gradient computation at every optimization iteration, while our method only requires a single neural network inference, leading to much faster curriculum optimization.

A core idea behind our method is utilizing classifiers to learn a diverse set of hypotheses that minimize the loss on source inputs but make differing predictions on target inputs $[32, 22]$ . It is related to ensemble methods $[7, 19, 14]$ that aggregate the multiple functions' predictions, but the proposed method is distinct in terms of directly optimizing on an underspecified target dataset for enhancing diversity. Even though this diversifying strategy for target data is typically considered from the perspective of out-of-distribution robustness in conditions of distribution shift $[29, 24, 38]$ or domain adaptation $[44, 46, 42]$ , we demonstrate how it can be applied for classifiers to quantify the similarity between the visited states and desired outcome states, and discuss its links to exploration and curriculum generation in RL.

# 3 Preliminary

We consider the Markov decision process (MDP) $\mathcal{M} = (\mathcal{S},\mathcal{G},\mathcal{A},\mathcal{P},\gamma)$ , where $\mathcal{S}$ indicates the state space, $\mathcal{G}$ the goal space, $\mathcal{A}$ the action space, $\mathcal{P}(s^{\prime}|s,a)$ the transition dynamics, and $\gamma$ the discount factor. In our framework, the MDP is not provided a reward function and we consider a setting where only the desired outcome examples $\{g_k^+\}_{k=1}^K$ from the desired outcome distribution $p^+(g)$ are given. Thus, our method utilizes an intrinsic reward $r:\mathcal{S}\times\mathcal{G}\times\mathcal{A}\to\mathbb{R}$ . Also, we represent the curriculum distribution obtained by our method as $p^c(s)$ .

# 3.1 Acquiring knowledge from underspecified data

For diversification-based curriculum RL, we train a classification model $y = f(x)$ in a supervised learning setting where $x \in X$ is input, and $y \in Y$ is the corresponding label. The model f is trained with a labeled source dataset $\mathcal{D}_{\mathrm{S}} \sim p_{\mathrm{S}}(x, y)$ that includes both the given desired outcome examples (y = 1) and the visited states in the replay buffer B of the RL agent (y = 0). We assume that the desired outcome distribution can be modeled as a mixture of outcome distribution, $p^{+}(x) = \sum_{o \in \mathbb{O}} w_{o} p_{o}(x)$ , where each $o \in O$ corresponds to a specific outcome distribution $p_{o}(x)$ . The selection of model f from hypothesis class $f \in F$ is achieved by minimizing the predictive risk $\mathbb{E}_{p_{\mathrm{S}}(x, y)}[\mathcal{L}(f(x), y)]$ , where L is a standard classification loss.

Although $f$ demonstrates good generalization on previously unseen data acquired from the source distribution $p_{\mathrm{S}}(x,y)$ , it is ambiguous to evaluate the model $f$ in distribution shift conditions such as when querying target data obtained from an out-of-distribution (e.g. a state lies in an unexplored region). This is because there could be many potential models $f$ that can minimize the predictive risk. To formalize this intuition, we introduce the concept of an $\varepsilon$ -optimal set defined as $\mathcal{F}^{\varepsilon} := \{f \in \mathcal{F} \mid \mathcal{L}_p(f) \leq \varepsilon\}$ [22], where $\mathcal{L}_p$ represents the risk associated with a distribution $p$ , and $\varepsilon \geq 0$ .

The definition of the $\varepsilon$ -optimal set implies that the predictions of any two models $f_{1}, f_{2}$ in the set are almost identical on $p_{\mathrm{S}}(x, y)$ when $\varepsilon$ is small. But, using $F^{\varepsilon}$ as it is has a few drawbacks: 1) there is no criterion to prefer any particular hypotheses of $F^{\varepsilon}$ over another, and 2) the trained model f might not be suitable for evaluating the target data distribution as it does not have a mechanism to quantitatively discriminate unseen target data from the labeled source data.

To address this point, we utilize an unlabeled target dataset $\mathcal{D}_{\mathrm{T}} \sim p_{\mathrm{T}}(x)$ for comparing the functions within $F^{\varepsilon}$ by examining how their predictions differ on $D_{T}$ . Since $D_{T}$ can be viewed as indicating the directions of functional change that are most crucial to the automatic exploration of the RL agent, we set $p_{\mathrm{T}}(x)$ as the uniform distribution between the lower and upper bound of the state space to include all possible candidate states to visit. Knowing the state space's bounds is a commonly utilized assumption in many curriculum learning works [37, 18, 5] and it does not require being aware of the dynamically feasible areas. Then, our objective is to identify a set of classification models that perform well in $p_{\mathrm{S}}(x,y)$ and disagree in $p_{\mathrm{T}}(x)$ to recognize the unseen state from the unexplored region by quantifying the similarity between the observed states in B and desired outcome examples. Such a function will lie within $F^{\varepsilon}$ of $p_{\mathrm{S}}(x,y)$ , and the model leverages $D_{T}$ to identify a diverse set of functions within the near-optimal set.

# 4 Method

For an automatic exploration toward the desired outcome distribution via calibrated guidance of the curriculum, the proposed D2C suggests curriculum goals via diversified classifiers that disagree in the unexplored region and enables the agent to conquer this area by exploring through the goal-conditioned shaped intrinsic reward. It allows the agent to make progress without any prior domain knowledge of the environment such as obstacles or distribution of the desired outcome states and advances the curriculum towards the desired outcome distribution $p^{+}(g)$ .

# 4.1 Diversification for disagreement on underspecified data

For the automatic exploration of the RL agent with the curriculum proposal, quantification of whether a queried state is already explored or not is required. To obtain such a quantification, we diversify a collection of classifier functions by comparing predictions for the target dataset while minimizing the training error as briefly described in Section 3.1. The intuition behind this is that diversifying predictions will produce functions that disagree with data in the ambiguous region [22].

Specifically, we use a multi-headed neural network with N heads to train diverse multiple functions. Each head i produces a prediction for an input x represented as $f_{i}(x)$ . To ensure that every head has a low predictive risk on $p_{\mathrm{S}}(x,y)$ , we minimize the cross-entropy loss for each head using $\mathcal{L}_{\mathrm{xent}}(f_{i}) = \mathbb{E}_{x,y \sim \mathcal{D}_{\mathrm{S}}}[\mathcal{C}\mathcal{E}(f_{i}(x),y)]$ . Ideally, it is desirable for each function to rely on distinctive predictive features of the input to encourage differing predictions. Therefore, we train each pair of heads to generate statistically independent predictions, which implies disagreement in predictions. It could be achieved by minimizing the mutual information between each pair of heads:

$$
\mathcal {L} _ {\mathrm{MI}} \left(f _ {i}, f _ {j}\right) = \mathbb {E} _ {x \sim \mathcal {D} _ {\mathrm{T}}} \left[ D _ {\mathrm{KL}} \left(p \left(f _ {i} (x), f _ {j} (x)\right) \| p \left(f _ {i} (x)\right) \otimes p \left(f _ {j} (x)\right)\right) \right] \tag {1}
$$

where the input data is obtained from target dataset $D_{T}$ . For implementation, we compute empirical estimates of the joint distribution $p(f_{i}(x), f_{j}(x))$ and the product of the marginal distributions $p(f_{i}(x)) \otimes p(f_{j}(x))$ , which can be computed using libraries developed for deep learning [22].

In summary, the overall objective for classifier diversification is represented with a hyperparameter $\lambda$ :

$$
\sum_ {i} \mathcal {L} _ {\text {xent}} (f _ {i}) + \lambda \sum_ {i \neq j} \mathcal {L} _ {\mathrm{MI}} (f _ {i}, f _ {j}) \tag {2}
$$

![](images/8686680250e5f3596148e63f833aafdcd26e1604d21210c7f826895b5d11e437.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph (1) Non-conditional
        direction TB
        A["Curriculum candidate: s1, s2, g1+, s5, s3, s4"] --> B["Select s3, s4"]
        C["Agent: s1, s2, g1+, s5, s3, s4"] --> D["Select s3, s4"]
        E["Desired outcome: g2+, s5, s3, s4"] --> F["Select s3, s4"]
    end

    subgraph (3) Conditional
        direction TB
        G["Curriculum candidate: s1, s2, g1+, s5, s3, s4"] --> H["Select s1, s3"]
        I["Agent: s1, s2, g1+, s5, s3, s4"] --> J["Select s1, s3"]
        K["Desired outcome: g2+, s5, s3, s4"] --> L["Select s1, s3"]
    end

    style (1) Non-conditional fill:#f9f9f9,stroke:#333
    style (3) Conditional fill:#f9f9f9,stroke:#333
```
</details>

Figure 2: Overview of the curriculum proposal when we select two candidates for curriculum goals of $g_{i}^{+}$ by bipartite matching. (1) The curriculum goals are proposed according to the quantification of the similarity-to-desired outcome ( $s_{i}$ with a high probability will be selected). (2) If the classifier is not in the conditional form, it cannot distinguish the source of the desired outcome example $g_{i}^{+}$ , resulting in collapsed curriculum proposals, (3) while the conditional classifier enables non-collapsed curriculum proposals even when the agent achieves a specific desired outcome (e.g. $g_{2}^{+}$ ) first.

# 4.2 Quantifying unexplored regions by conditional classifiers

From this section, we slightly abuse the notation s instead of x for the RL setting. Considering the diversification process in Section 4.1, we can quantify how much a queried state s is similar to the desired outcome example from $p^{+}(g)$ . Specifically, we can define a pseudo probability of the queried point s by averaging predictions of each head:

$$
p _ {\text { pseudo }} (y = 1 | s) := \frac {1}{N} \sum_ {i = 1} ^ {N} f _ {i} (s) \tag {3}
$$

When the queried state s is in proximity to the data within $p_{S}$ (with the desired outcome example labeled as 1 and states from the replay buffer B labeled as 0), it becomes challenging for the classifiers to give a high probability to labels that substantially differ from the neighboring data, since each classifier $f_{i}$ is trained to minimize the loss for the source data. On the other hand, if the queried state s is significantly different from the data in $p_{S}$ , each prediction head will output different values since the classifiers are trained to diversify the predictions on the target data (uniform distribution on the state space) to make their prediction values disagree as much as possible. For example, in the case of two heads, one predicts 1 and the other predicts 0 for the same queried target data, resulting in the pseudo probability of 0.5. This could be interpreted as a quantification of how much disagreement exists between classifiers, or uncertainty of the queried data, which can be utilized as an unexplored region-aware classification (Figure 1).

Thus, we can consider a curriculum learning objective for interpolating the curriculum distribution from the initial state distribution to $p^{+}(g)$ represented by the following cross-entropy loss:

$$
\mathcal {L} _ {\text { curr }} = \mathbb {E} _ {s \sim \mathcal {B}, g ^ {+} \sim p ^ {+} (g)} \left[ \mathcal {C E} (p _ {\text { pseudo }} (y = 1 | s); y = p _ {\text { pseudo }} (y = 1 | g ^ {+})) \right] \tag {4}
$$

Intuitively, before discovering the desired outcome examples in $p^{+}(g)$ , the curriculum goal candidate $s \sim B$ that minimizes Eq (4) is proposed in the frontier of the explored regions where classifiers $f_{i}$ disagree. And, as the agent explores and discovers the desired outcome examples, the curriculum candidate is updated to converge to $p^{+}(g)$ to minimize the discrepancy between the predicted labels of $g^{+}$ and s.

However, it is not applicable for a case when the desired outcome examples are spread over multimodal distribution or arbitrarily because the trained classifiers $f_{i}$ do not distinguish which $p_{o}$ the desired outcome example is obtained from (Figure 2). In other words, $f_{i}$ will predict a value close to

1 for any desired outcome example $g^{+}$ , and the loss in Eq (4) will be close to 0. As the curriculum goal candidates are obtained from the replay buffer $\mathcal{B}$ , the curriculum optimization may collapse if the agent achieves one of the desired outcome distributions $(p_{o})$ earlier, which is not desirable for achieving all the desired outcome examples regardless of its distribution.

Since we assume that we do not know which $p_{o}$ the desired outcome example is obtained from, nor the number of modes (o) of the desired outcome distributions, we propose to utilize a conditional classifier to address this point while satisfying the assumption. Specifically, we define goal-conditioned classifiers, where each classifier takes input s and is conditioned on g, and these classifiers are trained to minimize the following modified objective of Eq (2):

$$
\begin{array}{l} \mathbb {E} _ {g \sim \mathcal {D} _ {\mathrm{G}}} \left[ \mathbb {E} _ {s \sim \mathcal {B}} \left[ \sum_ {i} \mathcal {L} _ {\text {xent}} (f _ {i} (s; g), y = 0) \right] + \mathbb {E} _ {\varepsilon} \left[ \sum_ {i} \mathcal {L} _ {\text {xent}} (f _ {i} (g + \varepsilon ; g), y = 1) \right] \right. \tag {5} \\ \left. + \lambda \mathbb {E} _ {s \sim \mathcal {D} _ {\tau}} \left[ \sum_ {i \neq j} \mathcal {L} _ {\mathrm{MI}} \left(f _ {i} (s; g), f _ {j} (s; g)\right) \right] \right] \\ \end{array}
$$

where $\varepsilon$ is small noise (from uniform distribution around zero or standard normal distribution with a small variance) for numerical stability, and $D_{G}$ can be either $D_{T}$ or $\mathcal{B} \cup p^{+}(g)$ for training arbitrary goal-conditioned & unexplored region-aware classifiers. We found that there is no significant difference between these choices. Then, the pseudo probability can be represented as $p_{\text{pseudo}}(y = 1|s; g) := \frac{1}{N} \sum_{i=1}^{N} f_{i}(s; g)$ and we can address the arbitrarily distributed desired outcome examples without curriculum collapse (Figure 2). This proposed conditional classifier-based quantification is one of the key differences from the previous similar outcome-directed RL methods [23, 5]. Because the previous works require computing gradients of thousands of data for meta-learning-based network inference, while our method only requires a single feedforward inference without backpropagation which leads to fast computation.

# 4.3 Curriculum optimization via bipartite matching

As we assume that we have access to desired outcome examples from $p^{+}(g)$ instead of their explicit distribution, we can approximate it using the sampled set $\hat{p}^{+}(g)$ ( $|\hat{p}^{+}(g)| = K$ ). Then, the problem is formulated by the combinatorial setting that requires finding the curriculum goal candidate set $\hat{p}^{c}(s)$ that will be assigned to each sample of $\hat{p}^{+}(g)$ , and it can be solved via bipartite matching. With curriculum goal candidates and desired outcome examples, the curriculum learning objective is represented as follows:

$$
\min _ {\hat {p} ^ {c} (s): | \hat {p} ^ {c} (s) | = K} \sum_ {s _ {i} \in \hat {p} ^ {c} (s), g _ {i} ^ {+} \in \hat {p} ^ {+} (g)} w (s _ {i}, g _ {i} ^ {+}) \tag {6}
$$

$$
w (s _ {i}, g _ {i} ^ {+}) := \mathcal {C E} (p _ {\text { pseudo }} (y = 1 | s _ {i}; g _ {i} ^ {+}); y = p _ {\text { pseudo }} (y = 1 | g _ {i} ^ {+}; g _ {i} ^ {+})) \tag {7}
$$

The intuition behind this objective is similar to Eq (4), but it is different in terms of considering conditional quantification, which enables addressing arbitrarily distributed desired outcome examples. Then, we can create a bipartite graph $\mathbf{G}$ with edge costs $w$ by considering the sets of nodes $\mathbf{V}_a$ and $\mathbf{V}_b$ , which represent achieved states in replay buffer $\mathcal{B}$ and $\hat{p}^{+}(g)$ , respectively. We define the bipartite graph $\mathbf{G}(\{\mathbf{V}_a, \mathbf{V}_b\}, \mathbf{E})$ with edge weights $\mathbf{E}(\cdot, \cdot) = -w(\cdot, \cdot)$ . To solve this bipartite matching problem, we employ the Minimum Cost Maximum Flow algorithm [1, 37] to find $K$ edges with the minimum cost $w$ . The entire curriculum RL process is shown in Algorithm 1 in Appendix B.

# 4.4 Conditional classifier-based intrinsic reward

As we have trained conditional classifiers and defined the pseudo probability, we can additionally use this value as a shaped intrinsic reward for enabling the agent to solve the uninformed search problem. As the pseudo probability outputs 0 for the state in the replay buffer B, 1 for the desired outcome state, and a value between 0 and 1 for the state where the classifiers disagree (meaning unexplored region), we can define the goal-conditioned intrinsic reward as $r = p_{\text{pseudo}}(y = 1|s; g)$ . We use this intrinsic reward in all of our experiments. The ablation study for this reward is detailed in Section 5.2.

Table 1: Comparison of our work with the prior curriculum RL methods in conceptual aspects. 

<table><tr><td></td><td>Conditional quantification</td><td>Target dist. of curriculum</td><td>Arbitrary desired outcome dist.</td><td>Geometry-agnostic</td><td>Without external reward</td></tr><tr><td>HGG</td><td>✘</td><td> $p^{+}(g)$ </td><td>√</td><td>✘</td><td>✘</td></tr><tr><td>CURROT</td><td>✘</td><td>uniform or  $p^{+}(g)$ </td><td>√</td><td>✘</td><td>✘</td></tr><tr><td>PLR</td><td>✘</td><td>✘</td><td>✘</td><td>√</td><td>✘</td></tr><tr><td>VDS</td><td>✘</td><td>✘</td><td>✘</td><td>√</td><td>✘</td></tr><tr><td>ALP-GMM</td><td>✘</td><td>✘</td><td>✘</td><td>√</td><td>✘</td></tr><tr><td>OUTPACE</td><td>✘</td><td> $p^{+}(g)$ </td><td>✘</td><td>√</td><td>√</td></tr><tr><td>Ours</td><td>√</td><td> $p^{+}(g)$ </td><td>√</td><td>√</td><td>√</td></tr></table>

![](images/890b999471815d9309316422fc31401885e1b9b5647227882fcf18b6d2e01822.jpg)

<details>
<summary>natural_image</summary>

Abstract pixelated pattern with colorful clusters on a checkered background (no text or symbols)
</details>

![](images/c61d2016d596a1e0425358dcb96693296eb247fba8455cfdd4f33ca4e8d9831f.jpg)

<details>
<summary>natural_image</summary>

Pixelated abstract shape with checkered background and rainbow gradient (no text or symbols)
</details>

![](images/e9e66fe6615615899bbbd40e006db034c468532d515b5e15bef698f539baa62a.jpg)

<details>
<summary>natural_image</summary>

Abstract pixelated graphic with pink and blue clusters on a black-and-white checkerboard background (no text or symbols)
</details>

![](images/b7232212b018a583627161ab896fb12493eee841b1709443a6a23180b95a7280.jpg)

![](images/5caf308853b01ec7b53cfff39fd76d46528936e8237cab20db14e7215854a6d0.jpg)

<details>
<summary>natural_image</summary>

Abstract geometric pattern with blue, yellow, and pink pixelated shapes forming a maze-like structure (no text or symbols)
</details>

(a) Ours

![](images/b38519b5572de81968a7cbf00c9d507eedaf15719072b637f3d416d01b3d4da9.jpg)

<details>
<summary>natural_image</summary>

Pixelated abstract pattern with blue and orange shapes, no text or symbols present
</details>

(b) OUTPACE

![](images/139e96de2947af625edbac4a5e8e267f1fb94d6623682adea00590265114f04e.jpg)

<details>
<summary>natural_image</summary>

Pixelated maze diagram with colored dots and arrows, no text or symbols present
</details>

(c) HGG

![](images/d25a01940d12fb48a382d6743c604b16a98742138c97e108eb383270d8dc9c85.jpg)  
Figure 3: Curriculum goal visualization of the proposed method and baselines. First row: Ant Locomotion, Second row: Spiral-Maze.

# 5 Experiment

We conduct experiments on 6 environments that have multi-modal desired outcome distribution to validate our proposed method. We use various maze environments (Point Complex-Maze, Medium-Maze, Spiral-Maze) to validate our curriculum proposal capability, which is not limited to specific geometries. Additionally, we evaluate our method on more complex dynamics or other domains such as the Ant-Locomotion and Sawyer-Peg Push, Pick&Place with obstacle environments to demonstrate its effectiveness in domains beyond the navigation. (Refer to Appendix A for more details.)

We compare our method with several previous curriculum generation approaches, each of which possesses the following characteristics. OUTPACE [5] prioritizes goals that are considered uncertain and temporally distant from the initial state distribution through meta-learning-based uncertainty quantification and Wasserstein-distance-based temporal distance approximation. HGG [37] aims to reduce the distance between the desired outcome state and curriculum distributions, using a value function bias and the Euclidean distance metric. CURROT [18] interpolates between the desired outcome state and curriculum distribution while considering the agent's current capability using the Wasserstein distance. VDS [49] proposes epistemic uncertainty-based goals by utilizing the value function ensembles. ALP-GMM [36] models absolute learning progress score by a GMM. PLR [16] prioritizes increasingly challenging tasks by ranking task levels based on TD errors. We summarize the conceptual comparison between our method and baselines in Table 1.

# 5.1 Experimental results

First, to validate the quality of curriculum goal interpolation from the initial state distribution to the desired outcome distribution, we qualitatively and quantitatively evaluate the curriculum progress achieved during the training by optimizing Eq (6). For qualitative evaluation, we visualize the curriculum goals obtained by the proposed method and other baselines (Figure 3). The proposed method shows calibrated guidance toward the desired outcome states, even with the multi-modal distribution. But, HGG shows an inappropriate curriculum proposal due to the Euclidean distance metric, and OUTPACE shows curriculum goals collapsed only toward a single direction because it utilizes the Wasserstein distance for biasing curriculum goals into the temporally distant region. Once the agent explores a temporally far region in a specific direction earlier, the curriculum proposal is collapsed toward this area. In contrast, our method does not have such dependence, which enables

![](images/b8978ce2abe33196ec9ccabd4242bd104bea72314ac3224b7d3a06dfaaf2e6c7.jpg)

<details>
<summary>line</summary>

| steps (k) | distance (green) | distance (red) | distance (blue) | distance (orange) | distance (purple) |
| --------- | ---------------- | -------------- | --------------- | ----------------- | ----------------- |
| 0         | 14               | 5              | 10              | 5                 | 5                 |
| 250       | 20               | 5              | 5               | 5                 | 5                 |
| 500       | 15               | 5              | 5               | 5                 | 5                 |
| 750       | 15               | 5              | 5               | 5                 | 5                 |
| 1000      | 15               | 5              | 5               | 5                 | 5                 |
| 1250      | 15               | 5              | 5               | 5                 | 5                 |
| 1500      | 15               | 5              | 5               | 5                 | 5                 |
| 1750      | 15               | 5              | 5               | 5                 | 5                 |
| 2000      | 15               | 5              | 5               | 5                 | 5                 |
</details>

(a) Complex-Maze

![](images/5bb8898da8cfb8d19727ac7aea5dd16fe77b72342bb343215c92e6067e380bf4.jpg)

<details>
<summary>line</summary>

| steps (k) | distance (green) | distance (red) | distance (purple) | distance (gray) | distance (black) |
| --------- | ---------------- | -------------- | ----------------- | --------------- | ---------------- |
| 0         | 20               | 5              | 5                 | 5               | 5                |
| 250       | 20               | 5              | 5                 | 5               | 5                |
| 500       | 15               | 5              | 5                 | 5               | 5                |
| 750       | 15               | 5              | 5                 | 5               | 5                |
| 1000      | 15               | 5              | 5                 | 5               | 5                |
| 1250      | 15               | 5              | 5                 | 5               | 5                |
| 1500      | 15               | 5              | 5                 | 5               | 5                |
| 1750      | 15               | 5              | 5                 | 5               | 5                |
| 2000      | 15               | 5              | 5                 | 5               | 5                |
</details>

(b) Medium-Maze

![](images/e907734d3d489dc848c0bddd7cd122f899c873fdf5224e6d3664da2a63fe448b.jpg)

<details>
<summary>line</summary>

| steps (k) | distance (line 1) | distance (line 2) | distance (line 3) | distance (line 4) | distance (line 5) |
| --------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| 0         | 15                | 10                | 5                 | 5                 | 5                 |
| 250       | 25                | 20                | 5                 | 5                 | 5                 |
| 500       | 0                 | 20                | 5                 | 5                 | 5                 |
| 750       | 0                 | 20                | 5                 | 5                 | 5                 |
| 1000      | 0                 | 20                | 5                 | 5                 | 5                 |
| 1250      | 0                 | 20                | 5                 | 5                 | 5                 |
| 1500      | 0                 | 20                | 5                 | 5                 | 5                 |
| 1750      | 0                 | 20                | 5                 | 5                 | 5                 |
| 2000      | 0                 | 20                | 5                 | 5                 | 5                 |
</details>

(c) Spiral-Maze

![](images/60569677785ce25d0db708ac1c3cc2857b9ea914ceda665433110823a5066899.jpg)

<details>
<summary>line</summary>

| steps (k) | distance (line 1) | distance (line 2) | distance (line 3) | distance (line 4) | distance (line 5) | distance (line 6) | distance (line 7) |
| --------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| 0         | 10                | 8                 | 6                 | 4                 | 2                 | 1                 | 0                 |
| 500       | 12                | 8                 | 6                 | 4                 | 2                 | 1                 | 0                 |
| 1000      | 10                | 8                 | 6                 | 4                 | 2                 | 1                 | 0                 |
| 1500      | 8                 | 8                 | 6                 | 4                 | 2                 | 1                 | 0                 |
| 2000      | 8                 | 8                 | 6                 | 4                 | 2                 | 1                 | 0                 |
| 2500      | 8                 | 8                 | 6                 | 4                 | 2                 | 1                 | 0                 |
| 3000      | 8                 | 8                 | 6                 | 4                 | 2                 | 1                 | 0                 |
</details>

(d) Ant Locomotion

![](images/fd3d3f08fe1f50da643490dbd86f26c0e086b3283cacc037c96df70ca72e4d79.jpg)

<details>
<summary>line</summary>

| steps (k) | distance (line 1) | distance (line 2) | distance (line 3) | distance (line 4) | distance (line 5) | distance (line 6) | distance (line 7) |
| --------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| 0         | 0.6               | 0.5               | 0.4               | 0.3               | 0.2               | 0.1               | 0.1               |
| 500       | 0.4               | 0.3               | 0.2               | 0.2               | 0.2               | 0.1               | 0.1               |
| 1000      | 0.3               | 0.2               | 0.2               | 0.2               | 0.2               | 0.1               | 0.1               |
| 1500      | 0.2               | 0.2               | 0.2               | 0.2               | 0.2               | 0.1               | 0.1               |
| 2000      | 0.2               | 0.2               | 0.2               | 0.2               | 0.2               | 0.1               | 0.1               |
| 2500      | 0.2               | 0.2               | 0.2               | 0.2               | 0.2               | 0.1               | 0.1               |
| 3000      | 0.2               | 0.2               | 0.2               | 0.2               | 0.2               | 0.1               | 0.1               |
</details>

(e) Sawyer Push

![](images/4ea2bc259bfd2cb5c455a29cf28a88cd8e825527a93920b5cd5584fa0c45e765.jpg)

<details>
<summary>line</summary>

| steps (k) | distance (line 1) | distance (line 2) | distance (line 3) | distance (line 4) | distance (line 5) |
| --------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| 0         | 0.6               | 0.5               | 0.4               | 0.3               | 0.2               |
| 500       | 0.5               | 0.4               | 0.3               | 0.2               | 0.1               |
| 1000      | 0.5               | 0.3               | 0.2               | 0.1               | 0.05              |
| 1500      | 0.5               | 0.2               | 0.15              | 0.1               | 0.05              |
| 2000      | 0.5               | 0.2               | 0.15              | 0.1               | 0.05              |
| 2500      | 0.5               | 0.2               | 0.15              | 0.1               | 0.05              |
| 3000      | 0.5               | 0.2               | 0.15              | 0.1               | 0.05              |
</details>

(f) Sawyer Pick&Place

![](images/ab8c993d783eb6afbb2f7cfef9d342e5315da9ea8a5df819baa6f5f9d26f5d55.jpg)  
Figure 4: The mean distance between the curriculum goals and final goals (Lower is better). The shaded area represents a standard deviation across 5 seeds. The increases of our method at initial steps in some environments are attributed to the geometry of the environments.

![](images/848330196c90d01d079a89a5aafa7a0307dea8abb8ff4529a9a42038b753cbb0.jpg)

<details>
<summary>line</summary>

| steps (k) | Success_rate |
| --------- | ------------ |
| 0         | 0.0          |
| 250       | 0.9          |
| 500       | 0.95         |
| 750       | 0.98         |
| 1000      | 0.99         |
| 1250      | 0.995        |
| 1500      | 0.998        |
| 1750      | 0.999        |
| 2000      | 1.0          |
</details>

(a) Complex-Maze

![](images/00ff18fb10334532f75d1d25666a901ef48e0d68343367607e0a5822c16a79cb.jpg)

<details>
<summary>line</summary>

| steps (k) | success_rate |
| --------- | ------------ |
| 0         | 0.0          |
| 250       | 0.8          |
| 500       | 1.0          |
| 750       | 1.0          |
| 1000      | 1.0          |
| 1250      | 1.0          |
| 1500      | 1.0          |
| 1750      | 1.0          |
| 2000      | 1.0          |
</details>

(b) Medium-Maze

![](images/ef5427fe76806127f458f87e047830d0d79e15506c52ac64820ca20f2eee16b9.jpg)

<details>
<summary>line</summary>

| steps (k) | success_rate |
| --------- | ------------ |
| 0         | 0.0          |
| 250       | 0.0          |
| 500       | 0.8          |
| 750       | 1.0          |
| 1000      | 1.0          |
| 1250      | 1.0          |
| 1500      | 1.0          |
| 1750      | 1.0          |
| 2000      | 1.0          |
</details>

(c) Spiral-Maze

![](images/f8a9959ff8016b4c920965d8f77613b5f33476cc88ae9c2860cf239b8ecaf976.jpg)

<details>
<summary>line</summary>

| steps (k) | Success_rate |
| --------- | ------------ |
| 0         | 0.0          |
| 500       | 0.0          |
| 1000      | 0.4          |
| 1500      | 0.9          |
| 2000      | 0.95         |
| 2500      | 0.98         |
| 3000      | 1.0          |
</details>

(d) Ant Locomotion

![](images/b86a1c245c128d0509247317455ae78bf2768e3c84fa3549177610c6912b9c02.jpg)

<details>
<summary>line</summary>

| steps (k) | success_rate (blue) | success_rate (orange) | success_rate (green) | success_rate (red) |
| --------- | ------------------- | --------------------- | -------------------- | ------------------ |
| 0         | 0.0                 | 0.0                   | 0.0                  | 0.0                |
| 500       | 0.4                 | 0.3                   | 0.2                  | 0.1                |
| 1000      | 0.7                 | 0.6                   | 0.3                  | 0.1                |
| 1500      | 0.9                 | 0.7                   | 0.3                  | 0.1                |
| 2000      | 0.95                | 0.75                  | 0.3                  | 0.1                |
| 2500      | 0.98                | 0.8                   | 0.3                  | 0.1                |
| 3000      | 1.0                 | 0.85                  | 0.3                  | 0.1                |
</details>

(e) Sawyer Push

![](images/eb17ceb416aaa544e76ebe105ad146233b3d1104f51b4490cc4bcdf499d92585.jpg)

<details>
<summary>line</summary>

| steps (k) | success_rate (line 1) | success_rate (line 2) | success_rate (line 3) |
| --------- | --------------------- | --------------------- | --------------------- |
| 0         | 0.0                   | 0.0                   | 0.0                   |
| 500       | 0.2                   | 0.1                   | 0.05                  |
| 1000      | 0.6                   | 0.4                   | 0.2                   |
| 1500      | 0.8                   | 0.6                   | 0.3                   |
| 2000      | 0.9                   | 0.7                   | 0.4                   |
| 2500      | 0.95                  | 0.8                   | 0.5                   |
| 3000      | 1.0                   | 0.9                   | 0.6                   |
</details>

(f) Sawyer Pick&Place   
Figure 5: Evaluation success rates, using the same seeds and legends as shown in Figure 4. Note that some baselines are not visible as they coincide with a success rate of zero.

our method to always propose curriculum goals in any uncertain direction quantified by disagreement between the diversified classifiers.

We also plot the average distance from the proposed curriculum goals to the desired outcome states for quantitative evaluation (Figure 4). As we evaluated with multi-modal desired outcome distribution, the distance cannot be measured by naively averaging the distance between the randomly chosen desired outcome states and curriculum goals. Thus, we measure the distance by bipartite matching with the $l_{2}$ distance metric, which computes the distance between the desired outcome states and their assigned curriculum goals. As shown in Figure 4, only the proposed method consistently shows superior interpolation results from the initial state distribution to desired outcome distribution, while other baselines show some progress only in a simple geometric structure due to the Euclidean distance metric, or get stuck in some local optimum area due to the lack of or insufficient unexplored region quantification. We also plot the performance of the outcome-directed RL in Figure 5. As expected by the average distance measurement in Figure 4, the proposed method is the only one that consistently and quickly achieves the desired outcome states through the guidance of calibrated curriculum goals, indicating the advantage of the proposed diversified conditional classifiers.

![](images/8e75fa5be5c681fdee89e99053e9257885f3666eb1ad96397e79f0a5f8745d73.jpg)

<details>
<summary>line</summary>

| steps (k) | λ = 1(Default) | λ = 0.1 | λ = 0.5 | λ = 2 |
| --------- | -------------- | ------- | ------- | ----- |
| 0         | 0.0            | 0.0     | 0.0     | 0.0   |
| 250       | 0.0            | 0.0     | 0.0     | 0.0   |
| 500       | 0.8            | 0.6     | 0.9     | 0.95  |
| 750       | 1.0            | 0.8     | 1.0     | 1.0   |
| 1000      | 1.0            | 0.9     | 1.0     | 1.0   |
| 1250      | 1.0            | 0.95    | 1.0     | 1.0   |
| 1500      | 1.0            | 0.98    | 1.0     | 1.0   |
| 1750      | 1.0            | 0.99    | 1.0     | 1.0   |
| 2000      | 1.0            | 1.0     | 1.0     | 1.0   |
</details>

![](images/ff4fcb5a9bd9d8080b954731b496a89c8307dac100b4bbe67b96918ae246b52b.jpg)

<details>
<summary>line</summary>

| steps (k) | head=2(Default) | head=4 | head=6 |
| --------- | ---------------- | ------ | ------ |
| 0         | 0.0              | 0.0    | 0.0    |
| 250       | 0.0              | 0.0    | 0.0    |
| 500       | 0.8              | 0.7    | 0.6    |
| 750       | 0.95             | 0.9    | 0.85   |
| 1000      | 1.0              | 1.0    | 1.0    |
| 1250      | 1.0              | 1.0    | 1.0    |
| 1500      | 1.0              | 1.0    | 1.0    |
| 1750      | 1.0              | 1.0    | 1.0    |
| 2000      | 1.0              | 1.0    | 1.0    |
</details>

![](images/325c1fbcc15f602f158fe2f47cb6222c8fa6406cbaa0722dfeed9d18af57f259.jpg)

<details>
<summary>line</summary>

| steps (k) | Ours | Ours w/o IntrinsicReward | Ours w/o Curriculum |
| --------- | ---- | ------------------------ | ------------------- |
| 0         | 0.0  | 0.0                      | 0.0                 |
| 250       | 0.2  | 0.1                      | 0.1                 |
| 500       | 0.8  | 0.7                      | 0.3                 |
| 750       | 0.95 | 0.85                     | 0.4                 |
| 1000      | 0.98 | 0.9                      | 0.5                 |
| 1250      | 0.99 | 0.95                     | 0.6                 |
| 1500      | 0.99 | 0.98                     | 0.65                |
| 1750      | 0.99 | 0.99                     | 0.68                |
| 2000      | 0.99 | 0.99                     | 0.7                 |
</details>

![](images/ec24b4e4f09003e73678bdb4721ec7a21a7939a9e20fdc26ada911fa0815b404.jpg)

<details>
<summary>line</summary>

| steps (k) | λ = 1(Default) | λ = 0.1 | λ = 0.5 | λ = 2 |
| --------- | --------------- | ------- | ------- | ----- |
| 0         | 0.0             | 0.0     | 0.0     | 0.0   |
| 500       | 0.4             | 0.3     | 0.2     | 0.1   |
| 1000      | 0.7             | 0.6     | 0.5     | 0.4   |
| 1500      | 0.9             | 0.8     | 0.7     | 0.6   |
| 2000      | 0.95            | 0.9     | 0.85    | 0.8   |
| 2500      | 0.98            | 0.95    | 0.9     | 0.85  |
| 3000      | 1.0             | 0.98    | 0.95    | 0.9   |
</details>

(a) Auxiliary loss weight $\lambda$

![](images/0e191dd9d196f40d873274fab479406a879842dcb22dc8eb627da442c18019f1.jpg)

<details>
<summary>line</summary>

| steps (k) | head=2(Default) | head=4 | head=6 |
| --------- | ---------------- | ------ | ------ |
| 0         | 0.0              | 0.0    | 0.0    |
| 500       | 0.3              | 0.3    | 0.3    |
| 1000      | 0.7              | 0.7    | 0.7    |
| 1500      | 0.9              | 0.9    | 0.9    |
| 2000      | 1.0              | 1.0    | 1.0    |
| 2500      | 1.0              | 1.0    | 1.0    |
| 3000      | 1.0              | 1.0    | 1.0    |
</details>

(b) Number of prediction heads

![](images/160acacc4d59d4bce76896b085c16e29ec047176fa1eefd3371d3b89b81b110f.jpg)

<details>
<summary>line</summary>

| steps (k) | Ours | Ours w/o IntrinsicReward | Ours w/o Curriculum |
| --------- | ---- | ------------------------ | ------------------- |
| 0         | 0.0  | 0.0                      | 0.0                 |
| 500       | 0.2  | 0.1                      | 0.1                 |
| 1000      | 0.6  | 0.2                      | 0.7                 |
| 1500      | 0.8  | 0.3                      | 0.9                 |
| 2000      | 0.9  | 0.3                      | 0.9                 |
| 2500      | 0.95 | 0.3                      | 0.95                |
| 3000      | 1.0  | 0.3                      | 1.0                 |
</details>

(c) Intrinsic reward & Curriculum   
Figure 6: Ablation study in terms of the episode success rates. First row: Spiral-Maze. Second row: Sawyer Push. The shaded area represents the standard deviation across 5 seeds.

# 5.2 Ablation study

Number of prediction heads & Auxiliary loss weight $\lambda$ . To investigate the sensitivity to the hyperparameters of our method, we experimented with the different values of the auxiliary loss' weight $\lambda$ and with the different number of prediction heads of the conditional classifiers. As shown in Figure 6a, 6b, the results are slightly dependent on the environment's characteristics such as the presence of object interaction or complex geometry. However, the overall performance trend is not significantly affected by the number of prediction heads and $\lambda$ , except in the extreme case, which supports the superiority of our proposed method. More qualitative/quantitative analyses and other ablation studies are included in Appendix C.

Reward type & Curriculum. We evaluate the effectiveness of the proposed intrinsic reward and curriculum proposal by conducting two ablation experiments. First, we replace the intrinsic reward with a sparse reward (Ours w/o IntrinsicReward), which is typically used in goal-conditioned RL problems. Second, we conduct experiments without the curriculum proposal while utilizing the intrinsic reward (Ours w/o Curriculum). As shown in Figure 6c, there is performance degradation without using the proposed intrinsic reward in an environment with complex dynamics since an informative reward signal is crucial. In addition, the absence of a curriculum proposal leads to performance degradation in an environment with complex geometry as it requires carefully crafted guidance for exploration. These results highlight the importance of both proposed components, i.e. intrinsic reward and curriculum proposal.

# 6 Conclusion

We propose D2C that 1) performs a classifier diversification process to distinguish the unexplored region from the explored area and desired outcome example and 2) conquers the unexplored region by proposing the curriculum goal and the shaped intrinsic reward. It enables the agent to automatically progress toward the desired outcome states without prior knowledge of the environment. We demonstrate that our method outperforms the previous methods in terms of sample efficiency and geometry-agnostic, desired-outcome-distribution-agnostic curriculum progress, both quantitatively and qualitatively.

Limitation & Broader impacts. Despite the promising results, there is a limitation in the scalability of the proposed method since we use small noise to augment the conditioned goal for numerical stability, making it difficult to scale to high-dimensional inputs such as images. Thus, addressing this point would be an interesting future research direction to develop a more generally applicable method. Also, our work is subject to the potential negative societal impacts of RL, but we do not expect to encounter any additional negative impacts specific to this work.

# 7 Acknowledgement

This work was supported by Korea Research Institute for defense Technology Planning and advancement (KRIT) Grant funded by Defense Acquisition Program Administration(DAPA) (No. KRIT-CT-23-003, Development of AI researchers based on deep reinforcement learning and establishment of virtual combat experiment environment).

# References

[1] R K Ahuja, T L Magnanti, and J B Orlin. Network Flows: Theory, Algorithms, and Applications. Prentice Hall, Englewood Cliffs, NJ, 1st edition, 1993.   
[2] Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, Pieter Abbeel, and Wojciech Zaremba. Hindsight experience replay. arXiv preprint arXiv:1707.01495, 2017.   
[3] Marc Bellemare, Sriram Srinivasan, Georg Ostrovski, Tom Schaul, David Saxton, and Remi Munos. Unifying count-based exploration and intrinsic motivation. Advances in neural information processing systems, 29, 2016.   
[4] Yuri Burda, Harrison Edwards, Amos Storkey, and Oleg Klimov. Exploration by random network distillation. arXiv preprint arXiv:1810.12894, 2018.   
[5] Daesol Cho, Seungjae Lee, and H Jin Kim. Outcome-directed reinforcement learning by uncertainty & temporal distance-aware curriculum goal generation. arXiv preprint arXiv:2301.11741, 2023.   
[6] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.   
[7] Thomas G Dietterich. Ensemble methods in machine learning. In Multiple Classifier Systems: First International Workshop, MCS 2000 Cagliari, Italy, June 21–23, 2000 Proceedings 1, pages 1–15. Springer, 2000.   
[8] Ben Eysenbach, Sergey Levine, and Russ R Salakhutdinov. Replacing rewards with examples: Example-based policy search via recursive classification. Advances in Neural Information Processing Systems, 34:11541–11552, 2021.   
[9] Benjamin Eysenbach, Abhishek Gupta, Julian Ibarz, and Sergey Levine. Diversity is all you need: Learning skills without a reward function. arXiv preprint arXiv:1802.06070, 2018.   
[10] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In International Conference on Machine Learning, pages 1126–1135. PMLR, 2017.   
[11] Carlos Florensa, David Held, Xinyang Geng, and Pieter Abbeel. Automatic goal generation for reinforcement learning agents. In International conference on machine learning, pages 1515–1528. PMLR, 2018.   
[12] Justin Fu, Avi Singh, Dibya Ghosh, Larry Yang, and Sergey Levine. Variational inverse control with events: A general framework for data-driven reward definition. Advances in neural information processing systems, 31, 2018.   
[13] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pages 1861–1870. PMLR, 2018.   
[14] Lars Kai Hansen and Peter Salamon. Neural network ensembles. IEEE transactions on pattern analysis and machine intelligence, 12(10):993–1001, 1990.   
[15] Edward S Hu, Richard Chang, Oleh Rybkin, and Dinesh Jayaraman. Planning goals for exploration. arXiv preprint arXiv:2303.13002, 2023.   
[16] Minqi Jiang, Edward Grefenstette, and Tim Rocktäschel. Prioritized level replay. In International Conference on Machine Learning, pages 4940–4950. PMLR, 2021.   
[17] Jaekyeom Kim, Seohong Park, and Gunhee Kim. Unsupervised skill discovery with bottleneck option learning. arXiv preprint arXiv:2106.14305, 2021.   
[18] Pascal Klink, Haoyi Yang, Carlo D'Eramo, Jan Peters, and Joni Pajarinen. Curriculum reinforcement learning via constrained optimal transport. In International Conference on Machine Learning, pages 11341–11358. PMLR, 2022.   
[19] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in neural information processing systems, 30, 2017.

[20] Michael Laskin, Hao Liu, Xue Bin Peng, Denis Yarats, Aravind Rajeswaran, and Pieter Abbeel. Cic: Contrastive intrinsic control for unsupervised skill discovery. arXiv preprint arXiv:2202.00161, 2022.   
[21] Kimin Lee, Laura Smith, and Pieter Abbeel. Pebble: Feedback-efficient interactive reinforcement learning via relabeling experience and unsupervised pre-training. arXiv preprint arXiv:2106.05091, 2021.   
[22] Yoonho Lee, Huaxiu Yao, and Chelsea Finn. Diversify and disambiguate: Learning from underspecified data. arXiv preprint arXiv:2202.03418, 2022.   
[23] Kevin Li, Abhishek Gupta, Ashwin Reddy, Vitchyr H Pong, Aurick Zhou, Justin Yu, and Sergey Levine. Mural: Meta-learning uncertainty-aware rewards for outcome-driven reinforcement learning. In International Conference on Machine Learning, pages 6346–6356. PMLR, 2021.   
[24] Evan Z Liu, Behzad Haghgoo, Annie S Chen, Aditi Raghunathan, Pang Wei Koh, Shiori Sagawa, Percy Liang, and Chelsea Finn. Just train twice: Improving group robustness without training group information. In International Conference on Machine Learning, pages 6781–6792. PMLR, 2021.   
[25] Hao Liu and Pieter Abbeel. Aps: Active pretraining with successor features. In International Conference on Machine Learning, pages 6736–6747. PMLR, 2021.   
[26] Hao Liu and Pieter Abbeel. Behavior from the void: Unsupervised active pre-training. Advances in Neural Information Processing Systems, 34:18459–18473, 2021.   
[27] Pietro Mazzaglia, Tim Verbelen, Bart Dhoedt, Alexandre Lacoste, and Sai Rajeswar. Choreographer: Learning and adapting skills in imagination. arXiv preprint arXiv:2211.13350, 2022.   
[28] Russell Mendonca, Oleh Rybkin, Kostas Daniilidis, Danijar Hafner, and Deepak Pathak. Discovering and achieving goals via world models. Advances in Neural Information Processing Systems, 34:24379–24391, 2021.   
[29] Junhyun Nam, Hyuntak Cha, Sungsoo Ahn, Jaeho Lee, and Jinwoo Shin. Learning from failure: De-biasing classifier from biased classifier. Advances in Neural Information Processing Systems, 33:20673–20684, 2020.   
[30] Andrew Y Ng, Daishi Harada, and Stuart Russell. Policy invariance under reward transformations: Theory and application to reward shaping. In IcmI, volume 99, pages 278–287. Citeseer, 1999.   
[31] Georg Ostrovski, Marc G Bellemare, Aäron Oord, and Rémi Munos. Count-based exploration with neural density models. In International conference on machine learning, pages 2721–2730. PMLR, 2017.   
[32] Matteo Pagliardini, Martin Jaggi, François Fleuret, and Sai Praneeth Karimireddy. Agree to disagree: Diversity through disagreement for better transferability. arXiv preprint arXiv:2202.04414, 2022.   
[33] Deepak Pathak, Pulkit Agrawal, Alexei A Efros, and Trevor Darrell. Curiosity-driven exploration by self-supervised prediction. In International conference on machine learning, pages 2778–2787. PMLR, 2017.   
[34] Deepak Pathak, Dhiraj Gandhi, and Abhinav Gupta. Self-supervised exploration via disagreement. In International conference on machine learning, pages 5062–5071. PMLR, 2019.   
[35] Vitchyr H Pong, Murtaza Dalal, Steven Lin, Ashvin Nair, Shikhar Bahl, and Sergey Levine. Skew-fit: State-covering self-supervised reinforcement learning. arXiv preprint arXiv:1903.03698, 2019.   
[36] Rémy Portelas, Cédric Colas, Katja Hofmann, and Pierre-Yves Oudeyer. Teacher algorithms for curriculum learning of deep rl in continuously parameterized environments. In Conference on Robot Learning, pages 835–853. PMLR, 2020.   
[37] Zhizhou Ren, Kefan Dong, Yuan Zhou, Qiang Liu, and Jian Peng. Exploration via hindsight goal generation. Advances in Neural Information Processing Systems, 32, 2019.   
[38] Shiori Sagawa, Pang Wei Koh, Tatsunori B Hashimoto, and Percy Liang. Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. arXiv preprint arXiv:1911.08731, 2019.   
[39] Archit Sharma, Shixiang Gu, Sergey Levine, Vikash Kumar, and Karol Hausman. Dynamics-aware unsupervised discovery of skills. arXiv preprint arXiv:1907.01657, 2019.   
[40] Archit Sharma, Kelvin Xu, Nikhil Sardana, Abhishek Gupta, Karol Hausman, Sergey Levine, and Chelsea Finn. Autonomous reinforcement learning: Benchmarking and formalism. arXiv preprint arXiv:2112.09605, 2021.

[41] Avi Singh, Larry Yang, Kristian Hartikainen, Chelsea Finn, and Sergey Levine. End-to-end robotic reinforcement learning without reward engineering. arXiv preprint arXiv:1904.07854, 2019.   
[42] Kihyuk Sohn, David Berthelot, Nicholas Carlini, Zizhao Zhang, Han Zhang, Colin A Raffel, Ekin Dogus Cubuk, Alexey Kurakin, and Chun-Liang Li. Fixmatch: Simplifying semi-supervised learning with consistency and confidence. Advances in neural information processing systems, 33:596–608, 2020.   
[43] Sainbayar Sukhbaatar, Zeming Lin, Ilya Kostrikov, Gabriel Synnaeve, Arthur Szlam, and Rob Fergus. Intrinsic motivation and automatic curricula via asymmetric self-play. arXiv preprint arXiv:1703.05407, 2017.   
[44] Baochen Sun, Jiashi Feng, and Kate Saenko. Correlation alignment for unsupervised domain adaptation. Domain adaptation in computer vision applications, pages 153–171, 2017.   
[45] David Warde-Farley, Tom Van de Wiele, Tejas Kulkarni, Catalin Ionescu, Steven Hansen, and Volodymyr Mnih. Unsupervised control through non-parametric discriminative rewards. arXiv preprint arXiv:1811.11359, 2018.   
[46] Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc V Le. Self-training with noisy student improves imagenet classification. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10687–10698, 2020.   
[47] Denis Yarats, Rob Fergus, Alessandro Lazaric, and Lerrel Pinto. Reinforcement learning with prototypical representations. In International Conference on Machine Learning, pages 11920–11931. PMLR, 2021.   
[48] Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, and Sergey Levine. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In Conference on Robot Learning, pages 1094–1100. PMLR, 2020.   
[49] Yunzhi Zhang, Pieter Abbeel, and Lerrel Pinto. Automatic curriculum learning through value disagreement. Advances in Neural Information Processing Systems, 33:7648–7659, 2020.   
[50] Rui Zhao, Yang Gao, Pieter Abbeel, Volker Tresp, and Wei Xu. Mutual information state intrinsic control. arXiv preprint arXiv:2103.08107, 2021.

# A Training & Experiments details

# A.1 Training details

Baselines. The baseline curriculum RL algorithms are trained as follows,

- OUTPACE [5]: We follow the default setting in the original implementation from https://github.com/jayLEE0301/outpace\_official.   
- HGG [37]: We follow the default setting in the original implementation from https://github.com/Stilwell-Git/Hindsight-Goal-Generation.   
- CURROT [18]: We follow the default setting in the original implementation from https://github.com/psclklnk/currot.   
- PLR [16], VDS [49], ALP-GMM [36]: We follow the default setting in implementation from https://github.com/psclklnk/currot.

D2C and all the baselines are trained by SAC $[13]$ with the sparse reward except for the OUTPACE which uses an intrinsic reward based on Wasserstein distance with a time-step metric.

Training details. We used NVIDIA A5000 GPU and AMD Ryzen Threadripper 3960X for training, and each experiment took about 1\~2 days for training. We used small noise from a uniform distribution with an environment-specific noise scale (Table 3) for augmenting the conditioned goal in Eq (5). Also, we used the mapping $\phi(\cdot)$ that abstracts the state space into the goal space when we use the diversified conditional classifiers (i.e. $f_{i}(\phi(s);g)$ ). For example, $\phi(\cdot)$ abstracts the proprioceptive states (e.g. xy position of the agent) in navigation tasks, and abstracts the object-centric states (e.g. xyz position of the object) in robotic manipulation tasks.

Table 2: Hyperparameters for D2C 

<table><tr><td>critic hidden dim</td><td>512</td><td>discount factor  $\gamma$ </td><td>0.99</td></tr><tr><td>critic hidden depth</td><td>3</td><td>batch size</td><td>512</td></tr><tr><td>critic target  $\tau$ </td><td>0.01</td><td>init temperature  $\alpha_{\text{init}}$  of SAC</td><td>0.3</td></tr><tr><td>Critic target update frequency</td><td>2</td><td>replay buffer  $\mathcal{B}$  size</td><td>3e6</td></tr><tr><td>actor hidden dim</td><td>512</td><td>learning rate for  $f_i$ </td><td>1e-3</td></tr><tr><td>actor hidden depth</td><td>3</td><td>learning rate for Critic &amp; Actor</td><td>1e-4</td></tr><tr><td>actor update frequency</td><td>2</td><td>optimizer</td><td>adam</td></tr></table>

Table 3: Default env-specific hyperparameters for D2C 

<table><tr><td>Env name</td><td># of heads</td><td> $\lambda$ </td><td> $\epsilon$ </td><td> $f_i$  update freq (step)</td><td> $f_i$  # of iteration per update</td><td>max episode horizon</td></tr><tr><td>Complex-Maze</td><td>2</td><td>1</td><td>0.5</td><td>2000</td><td>16</td><td>100</td></tr><tr><td>Medium-Maze</td><td>2</td><td>1</td><td>0.5</td><td>2000</td><td>16</td><td>100</td></tr><tr><td>Spiral-Maze</td><td>2</td><td>1</td><td>0.5</td><td>2000</td><td>16</td><td>100</td></tr><tr><td>Ant Locomotion</td><td>2</td><td>2</td><td>1.0</td><td>4500</td><td>16</td><td>300</td></tr><tr><td>Sawyer-Peg-Push</td><td>2</td><td>1</td><td>0.025</td><td>3000</td><td>16</td><td>200</td></tr><tr><td>Sawyer-Peg-Pick&amp;Place</td><td>2</td><td>1</td><td>0.025</td><td>3000</td><td>16</td><td>200</td></tr></table>

# A.2 Environment details

- Complex-Maze: The observation consists of the $xy$ position, angle, velocity, and angular velocity of the 'point'. The action space consists of the velocity and angular velocity of the 'point'. The initial state of the agent is $[0, 0]$ and the desired outcome states are obtained from the default goal points $[8, 16], [-8, -16], [16, -8], [-16, 8]$ . The size of the map is $36 \times 36$ .   
- Medium-Maze: It is the same as the Complex-Maze environment except that the desired outcome states are obtained from the default goal points [16, 16], [-16, -16], [16, -16], [-16, 16].

![](images/e2036ba1e180c511e7d11fb0ca46362442b6d6a86641a8e061a6494de7e8c11c.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Start"] --> B{Decision}
    B -->|Yes| C["Path 1"]
    B -->|No| D["Path 2"]
    C --> E["End"]
    D --> E
    style A fill:#fff,stroke:#000
    style E fill:#fff,stroke:#000
```
</details>

(a) Complex-Maze

![](images/1cc4823be325560830c2c8ce60e44987e81e53f8fb95ae8d40e39d91999b0d7d.jpg)

<details>
<summary>natural_image</summary>

Four square frames arranged in a cross pattern on a checkered blue background with yellow stars at corners (no text or symbols)
</details>

(b) Medium-Maze

![](images/011c89faf352f9d0b00f04096707e75e290a11223ba869e0de85e502fa250ccf.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Start"] --> B{Path}
    B --> C["End"]
    style A fill:#f9f,stroke:#333
    style C fill:#bbf,stroke:#333
```
</details>

(c) Spiral-Maze

![](images/bf3a932ff019fe8325cc629dcc08fb9c2e3264442cbfeb4d3fb43a4fa6ce8476.jpg)

<details>
<summary>text_image</summary>

Diagram showing a red spiral path on a black-and-white checkerboard background with yellow stars, resembling a maze or puzzle.
</details>

(d) Ant Locomotion

![](images/4ae435cb275f2bb5715d6d849d1c133697a72fd9bd47aa7f434230898c09bb87.jpg)

<details>
<summary>natural_image</summary>

3D rendering of a red robotic arm interacting with a wooden table, showing motion arrows and star symbols (no text or labels)
</details>

(e) Sawyer Push

![](images/dd75fe3b5621ce16053bd118e816a1064dcc4793728cd05b1650d518aa9b12ea.jpg)

<details>
<summary>natural_image</summary>

3D rendering of a red robotic arm interacting with a wooden table, showing motion arrows and star markers (no text or symbols)
</details>

(f) Sawyer Pick&Place   
Figure 7: Environments used for evaluation: yellow stars indicate the desired outcome examples. (a)-(c) the agent should navigate various maze environments with multi-modal desired outcome distribution. (d) the ant locomotion environment with multi-modal desired outcome distribution. (e) the robot has to push or pick & place a peg to the multi-modal desired locations while avoiding an obstacle at the center of the table.

- Spiral-Maze: The observation space and actions space and initial state of the agent are the same as in the Complex-Maze environment. The desired outcome states are obtained from the default goal points [12, 16], $[-12, -16]$ . The size of the map is $28 \times 36$ .   
- Ant Locomotion: The observation consists of the $xyz$ position, $xyz$ velocity, joint angle, and joint angular velocity of the 'ant'. The action space consists of the torque applied on the rotor of the 'ant'. The initial state of the agent is $[0, 0]$ and the desired outcome states are obtained from the default goal points $[4, 8], [-4, -8]$ . The size of the map is $12 \times 20$ .   
- Sawyer-Peg-Push: The observation consists of the $xyz$ position of the end-effector, the object, and the gripper's state. The action space consists of the $xyz$ position of the end-effector and gripper open/close control. The initial state of the object is [0.4, 0.8, 0.02] and the desired outcome states are obtained from the default goal points [-0.3, 0.4, 0.02], [-0.3, 0.8, 0.02], [0.4, 0.4, 0.02]. The wall is located at the center of the table. Thus, the robot arm should detour the wall to reach the desired goal states. We referred to the metaworld [48] and EARL [40] environments.   
- Sawyer-Peg-Pick&Place: It is the same as the Sawyer-Peg-Push environment except that the desired outcome states are obtained from the default goal points $[-0.3, 0.4, 0.2]$ , $[-0.3, 0.8, 0.2]$ , $[0.4, 0.4, 0.2]$ , and the wall is located at the center of the table, fully blocking a path for pushing. Thus, the robot arm should pick and move the object over the wall to reach the desired goal states.

Algorithm 1 Overview of D2C algorithm   
1: Input: desired outcome examples $\hat{p}^{+}(g)$ , total training episodes N, Env, environment horizon H, actor $\pi$ , critic Q, replay buffer B
2: for iteration=1,2,...,N do
3: $\hat{p}^{c} \leftarrow$ sample K curriculum goals that minimize Eq (6). We refer to HGG [37] for solving the bipartite matching problem.
4: for i=1,2,...,K do
5: Env.reset()
6: $g \leftarrow \hat{p}^{c}$ 7: for t=0,1,...,H-1 do
8: if achieved g then
9: $g \leftarrow$ random goal (randomly sample a few states near $s_{t}$ and measure $p_{pseudo}$ . Then select a state with the highest value of $p_{pseudo}$ .)
10: end if
11: $a_{t} \leftarrow \pi(\cdot|s_{t}, g)$ 12: $s_{t+1} \leftarrow \text{Env.step}(a_{t})$ 13: end for
14: $B \leftarrow B \cup \{s_{0}, a_{0}, g, s_{1}\ldots\}$ 15: end for
16: for i=0,1,...,M do
17: Sample a minibatch b from B and replace the original reward with intrinsic reward in Section 4.4 (We used relabeling technique based on [2]).
18: Train $\pi$ and Q with b via SAC [13].
19: Sample another minibatch $b'$ from $D_{S} \sim \{(\mathcal{B}, y = 0), (\mathcal{D}_{G}, y = 1)\}$ and $D_{T} \sim U$ .
20: Train $f_{i}$ with $b'$ via Eq. (5)
21: end for
22: end for

# C More experimental results

# C.1 Full results of the main script

We included the full results of the main script in this section. We include the visualization of the proposed curriculum goals in all environments in Figure 8. The visualization results of the Sawyer-Peg-Pick&Place are not included as it shares the same map with the Sawyer-Peg-Push environment.

![](images/b95ed0004ac63aa19802893832e58516bb20419931785809de7834ba2bdc9b5d.jpg)  
Figure 8: Curriculum goal visualization of the proposed method and baselines in all environments. First row: Complex-Maze. Second row: Medium-Maze. Third row: Spiral-Maze. Fourth row: Ant Locomotion. Fifth row: Sawyer Push. Sixth row: Sawyer Pick & Place.

# C.2 Additional ablation study results

Full ablation study results of the main script. We conducted ablation studies described in our main script in all environments. Figure 9 shows the average distance from the proposed curriculum goals to the desired final goal states along the training steps, and Figure 10 shows the episode success rates along the training steps. As we can see in these figures, we could obtain consistent analysis with the results in the main script in most of the environments.

![](images/a7f30ba092cc35f90a5300e6fdaa48bca3e2aa7db87b892a0a1a6eaf40957064.jpg)

<details>
<summary>line</summary>

| steps (k) | λ = 1(Default) | λ = 0.1 | λ = 0.5 | λ = 2 |
| --------- | -------------- | ------- | ------- | ----- |
| 0         | 12.0           | 12.0    | 12.0    | 12.0  |
| 250       | 0.5            | 1.0     | 1.0     | 1.0   |
| 500       | 0.3            | 1.2     | 1.2     | 1.2   |
| 750       | 0.2            | 1.5     | 1.5     | 1.5   |
| 1000      | 0.1            | 2.0     | 2.0     | 2.0   |
| 1250      | 0.1            | 2.5     | 2.5     | 2.5   |
| 1500      | 0.1            | 3.0     | 3.0     | 3.0   |
| 1750      | 0.1            | 4.0     | 4.0     | 4.0   |
| 2000      | 0.1            | 4.5     | 4.5     | 4.5   |
</details>

![](images/698c50cd1663613cf0330d41f6504e7182586c0a02684a58b1956e1ea00436ac.jpg)

<details>
<summary>line</summary>

| steps (k) | head=2(Default) | head=4 | head=6 |
| --------- | ---------------- | ------ | ------ |
| 0         | 12.0             | 12.0   | 12.0   |
| 250       | 1.0              | 3.0    | 1.0    |
| 500       | 0.5              | 2.0    | 1.5    |
| 750       | 0.5              | 1.5    | 2.0    |
| 1000      | 0.5              | 1.0    | 3.0    |
| 1250      | 0.5              | 1.0    | 4.0    |
| 1500      | 0.5              | 1.0    | 5.0    |
| 1750      | 0.5              | 1.0    | 5.5    |
| 2000      | 0.5              | 1.0    | 6.0    |
</details>

![](images/2d7cb9f96e308e439994391fbd56f29256501f22b96ba536f2abb5429c77f92e.jpg)

<details>
<summary>line</summary>

| steps (k) | Ours | Ours w/o IntrinsicReward |
| --------- | ---- | ------------------------ |
| 0         | 15.0 | 15.0                     |
| 250       | 1.0  | 2.0                      |
| 500       | 0.5  | 0.8                      |
| 750       | 0.3  | 0.5                      |
| 1000      | 0.2  | 0.3                      |
| 1250      | 0.1  | 0.2                      |
| 1500      | 0.1  | 0.1                      |
| 1750      | 0.1  | 0.1                      |
| 2000      | 0.1  | 0.1                      |
</details>

![](images/8efb2df28bc06c6db28bcc7eb037d45d73fac7aa62224ee4d805039003ad729d.jpg)

<details>
<summary>line</summary>

| steps (k) | λ = 1(Default) | λ = 0.1 | λ = 0.5 | λ = 2 |
| --------- | --------------- | ------- | ------- | ----- |
| 0         | 17.5            | 17.5    | 17.5    | 17.5  |
| 250       | 0.0             | 0.0     | 0.0     | 0.0   |
| 500       | 0.0             | 0.0     | 0.0     | 0.0   |
| 750       | 0.0             | 0.0     | 0.0     | 0.0   |
| 1000      | 0.0             | 0.0     | 0.0     | 0.0   |
| 1250      | 0.0             | 0.0     | 0.0     | 0.0   |
| 1500      | 0.0             | 0.0     | 0.0     | 0.0   |
| 1750      | 0.0             | 0.0     | 0.0     | 0.0   |
| 2000      | 0.0             | 0.0     | 0.0     | 0.0   |
</details>

![](images/e355d60f52ffdefc7d2eb672fe485d10678300cfa839f80097222f4ed8b219ad.jpg)

<details>
<summary>line</summary>

| steps (k) | head=2(Default) | head=4 | head=6 |
| --------- | ---------------- | ------ | ------ |
| 0         | 12.0             | 12.0   | 12.0   |
| 250       | 0.5              | 0.5    | 0.5    |
| 500       | 0.3              | 0.3    | 0.3    |
| 750       | 0.2              | 0.2    | 0.2    |
| 1000      | 0.1              | 0.1    | 0.1    |
| 1250      | 0.1              | 0.1    | 0.1    |
| 1500      | 0.1              | 0.1    | 0.1    |
| 1750      | 0.1              | 0.1    | 0.1    |
| 2000      | 0.1              | 0.1    | 0.1    |
</details>

![](images/01c15cf3e46a1ee3c21fa386c50c7527cc0d0e8ca3a5c7908da291738db99b72.jpg)

<details>
<summary>line</summary>

| steps (k) | Ours | Ours w/o IntrinsicReward |
| --------- | ---- | ------------------------ |
| 0         | 17.5 | 17.5                     |
| 250       | 0.0  | 0.0                      |
| 500       | 0.0  | 0.0                      |
| 750       | 0.0  | 0.0                      |
| 1000      | 0.0  | 0.0                      |
| 1250      | 0.0  | 0.0                      |
| 1500      | 0.0  | 0.0                      |
| 1750      | 0.0  | 0.0                      |
| 2000      | 0.0  | 0.0                      |
</details>

![](images/e415bf5d24f1fd8c13281f323db6aa6877427f3caf85dd947bc3723cc14c05d3.jpg)

<details>
<summary>line</summary>

| steps (k) | λ = 1(Default) | λ = 0.1 | λ = 0.5 | λ = 2 |
| --------- | --------------- | ------- | ------- | ----- |
| 0         | 15              | 5       | 15      | 15    |
| 250       | 25              | 5       | 25      | 25    |
| 500       | 10              | 5       | 10      | 10    |
| 750       | 5               | 10      | 5       | 5     |
| 1000      | 10              | 15      | 10      | 10    |
| 1250      | 15              | 20      | 15      | 15    |
| 1500      | 20              | 25      | 20      | 20    |
| 1750      | 25              | 30      | 25      | 25    |
| 2000      | 30              | 35      | 30      | 30    |
</details>

![](images/07ea14063c386dffe0d3a97a61a1334c79d18a34a05dfddb5b3b3d377fde3822.jpg)

<details>
<summary>line</summary>

| steps (k) | head=2(Default) | head=4 | head=6 |
| --------- | ---------------- | ------ | ------ |
| 0         | 10               | 10     | 10     |
| 250       | 25               | 25     | 25     |
| 500       | 15               | 15     | 15     |
| 750       | 5                | 5      | 5      |
| 1000      | 0                | 0      | 0      |
| 1250      | 0                | 0      | 0      |
| 1500      | 0                | 0      | 0      |
| 1750      | 0                | 0      | 0      |
| 2000      | 0                | 0      | 0      |
</details>

![](images/1973f9a98c1fa71d2eea2deb13e487e09906100e1828022dac8579a5c9d72375.jpg)

<details>
<summary>line</summary>

| steps (k) | Ours | Ours w/o IntrinsicReward |
| --------- | ---- | ------------------------ |
| 0         | 15   | 18                       |
| 250       | 25   | 26                       |
| 500       | 10   | 12                       |
| 750       | 2    | 3                        |
| 1000      | 0    | 0                        |
| 1250      | 0    | 0                        |
| 1500      | 0    | 0                        |
| 1750      | 0    | 0                        |
| 2000      | 0    | 0                        |
</details>

![](images/555de0273644b27f355cb41aa698a7a7bd66e31a51f39e2587b01ce6f6cc69b6.jpg)

<details>
<summary>line</summary>

| steps (k) | λ = 2(Default) | λ = 0.1 | λ = 0.5 | λ = 1 | λ = 4 |
| --------- | --------------- | ------- | ------- | ----- | ----- |
| 0         | 10.0            | 10.0    | 10.0    | 10.0  | 10.0  |
| 500       | 8.0             | 9.0     | 8.5     | 7.5   | 8.0   |
| 1000      | 6.0             | 8.5     | 8.0     | 6.5   | 7.5   |
| 1500      | 4.0             | 8.0     | 7.5     | 5.5   | 7.0   |
| 2000      | 2.0             | 7.5     | 7.0     | 4.5   | 6.5   |
| 2500      | 1.0             | 7.0     | 6.5     | 3.5   | 6.0   |
| 3000      | 0.5             | 6.5     | 6.0     | 2.5   | 5.5   |
</details>

![](images/133b4d1fcc07986eb55969dd3bc260a90d4d22370e504101a166387725c50738.jpg)

<details>
<summary>line</summary>

| steps (k) | head=2(Default) | head=4 | head=6 |
| --------- | ---------------- | ------ | ------ |
| 0         | 9.0              | 9.0    | 9.0    |
| 500       | 7.0              | 8.5    | 8.0    |
| 1000      | 1.0              | 8.0    | 7.5    |
| 1500      | 0.5              | 7.5    | 7.0    |
| 2000      | 0.5              | 7.0    | 6.5    |
| 2500      | 0.5              | 6.5    | 6.0    |
| 3000      | 0.5              | 6.0    | 5.5    |
</details>

![](images/d3e2d97e0c419441171b34f84b4cac98290acb6fe119040c97d2ffc06d41f002.jpg)

<details>
<summary>line</summary>

| steps (k) | Ours | Ours w/o IntrinsicReward |
| --------- | ---- | ------------------------ |
| 0         | 10   | 10                       |
| 500       | 5    | 9                        |
| 1000      | 8    | 9                        |
| 1500      | 1    | 9                        |
| 2000      | 1    | 9                        |
| 2500      | 1    | 9                        |
| 3000      | 1    | 9                        |
</details>

![](images/b290aded20ea6bc3091d20c9c4d7a8eecf57ac28dc293a1eaa83d0e2a4788750.jpg)

<details>
<summary>line</summary>

| steps (k) | λ = 1(Default) | λ = 0.1 | λ = 0.5 | λ = 2 |
| --------- | --------------- | ------- | ------- | ----- |
| 0         | 0.6             | 0.6     | 0.6     | 0.6   |
| 500       | 0.4             | 0.3     | 0.5     | 0.4   |
| 1000      | 0.2             | 0.2     | 0.3     | 0.3   |
| 1500      | 0.1             | 0.1     | 0.1     | 0.1   |
| 2000      | 0.1             | 0.1     | 0.1     | 0.1   |
| 2500      | 0.1             | 0.1     | 0.1     | 0.1   |
| 3000      | 0.1             | 0.1     | 0.1     | 0.1   |
</details>

![](images/8c7fc558b9656be583f02faf0c5ef47b2b1b0b2c070685efb478125e87c52b05.jpg)

<details>
<summary>line</summary>

| steps (k) | head=2(Default) | head=4 | head=6 |
| --------- | --------------- | ------ | ------ |
| 0         | 0.6             | 0.6    | 0.6    |
| 500       | 0.4             | 0.3    | 0.5    |
| 1000      | 0.2             | 0.1    | 0.3    |
| 1500      | 0.1             | 0.1    | 0.1    |
| 2000      | 0.1             | 0.1    | 0.1    |
| 2500      | 0.1             | 0.1    | 0.1    |
| 3000      | 0.1             | 0.1    | 0.1    |
</details>

![](images/87b2fccbc031f49750e7d0acf790a0ec9f6cb27a10902b05fc76f10b04fd8582.jpg)

<details>
<summary>line</summary>

| steps (k) | Ours  | Ours w/o IntrinsicReward |
| --------- | ----- | ------------------------ |
| 0         | 0.6   | 0.6                      |
| 500       | 0.4   | 0.5                      |
| 1000      | 0.2   | 0.4                      |
| 1500      | 0.1   | 0.3                      |
| 2000      | 0.1   | 0.25                     |
| 2500      | 0.1   | 0.2                      |
| 3000      | 0.1   | 0.2                      |
</details>

![](images/4723ad451dcbfb36eba9b4d2c25de40771aaa1886dcb200844333f78adfa0964.jpg)

<details>
<summary>line</summary>

| steps (k) | λ = 1(Default) | λ = 0.1 | λ = 0.5 | λ = 2 |
| --------- | --------------- | ------- | ------- | ----- |
| 0         | 0.6             | 0.6     | 0.6     | 0.6   |
| 500       | 0.4             | 0.4     | 0.4     | 0.4   |
| 1000      | 0.2             | 0.2     | 0.2     | 0.2   |
| 1500      | 0.1             | 0.1     | 0.1     | 0.1   |
| 2000      | 0.05            | 0.05    | 0.05    | 0.05  |
| 2500      | 0.02            | 0.02    | 0.02    | 0.02  |
| 3000      | 0.01            | 0.01    | 0.01    | 0.01  |
</details>

![](images/a8a531082765f6651eff75d78181fe7c300c8dc144229ca340edc9c8de5d9be7.jpg)

<details>
<summary>line</summary>

| steps (k) | head=2(Default) | head=4 | head=6 |
| --------- | ---------------- | ------ | ------ |
| 0         | 0.6              | 0.6    | 0.6    |
| 500       | 0.3              | 0.3    | 0.3    |
| 1000      | 0.1              | 0.1    | 0.1    |
| 1500      | 0.05             | 0.05   | 0.05   |
| 2000      | 0.02             | 0.02   | 0.02   |
| 2500      | 0.01             | 0.01   | 0.01   |
| 3000      | 0.01             | 0.01   | 0.01   |
</details>

![](images/146431633b05b1be880ed30a40ba4da3dfee49f638ef15bceb4a68b62bc7b155.jpg)

<details>
<summary>line</summary>

| steps (k) | Ours | Ours w/o IntrinsicReward |
| --------- | ---- | ------------------------ |
| 0         | 0.6  | 0.6                      |
| 500       | 0.3  | 0.4                      |
| 1000      | 0.1  | 0.4                      |
| 1500      | 0.0  | 0.4                      |
| 2000      | 0.0  | 0.4                      |
| 2500      | 0.0  | 0.4                      |
| 3000      | 0.0  | 0.4                      |
</details>

(a) Auxiliary loss weight $\lambda$   
(b) Number of prediction heads   
(c) Intrinsic reward & Curriculum   
Figure 9: Ablation study in terms of the distance from the proposed curriculum goals to the desired final goal states (Lower is better). There are no results for the ablation study without a curriculum proposal since there are no curriculum goals to measure the distance from the desired final goal states. First row: Complex-Maze. Second row: Medium-Maze. Third row: Spiral-Maze. Fourth row: Ant Locomotion. Fifth row: Sawyer Push. Sixth row: Sawyer Pick & Place. The shaded area represents a standard deviation across 5 seeds.

![](images/0afa855cf895f9adff00a2af0633f312edc8684a43a6fe847a7173bba80b150e.jpg)

<details>
<summary>line</summary>

| steps (k) | λ = 1(Default) | λ = 0.1 | λ = 0.5 | λ = 2 |
| --------- | --------------- | ------- | ------- | ----- |
| 0         | 0.0             | 0.0     | 0.0     | 0.0   |
| 250       | 0.9             | 0.85    | 0.85    | 0.8   |
| 500       | 0.95            | 0.9     | 0.9     | 0.85  |
| 750       | 0.95            | 0.9     | 0.9     | 0.85  |
| 1000      | 0.95            | 0.9     | 0.9     | 0.85  |
| 1250      | 0.95            | 0.9     | 0.9     | 0.85  |
| 1500      | 0.95            | 0.9     | 0.9     | 0.85  |
| 1750      | 0.95            | 0.9     | 0.9     | 0.85  |
| 2000      | 0.95            | 0.9     | 0.9     | 0.85  |
</details>

![](images/64b026ef0be937d2316571ab29c2e1e17ab63e8743e33296f29ece6ed9058e02.jpg)

<details>
<summary>line</summary>

| steps (k) | head=2(Default) | head=4 | head=6 |
| --------- | ---------------- | ------ | ------ |
| 0         | 0.0              | 0.0    | 0.0    |
| 250       | 0.9              | 0.85   | 0.75   |
| 500       | 0.95             | 0.9    | 0.8    |
| 750       | 0.95             | 0.9    | 0.8    |
| 1000      | 0.95             | 0.9    | 0.8    |
| 1250      | 0.95             | 0.9    | 0.8    |
| 1500      | 0.95             | 0.9    | 0.8    |
| 1750      | 0.95             | 0.9    | 0.8    |
| 2000      | 0.95             | 0.9    | 0.8    |
</details>

![](images/9e55348c84114962fbfbb06a254643275ff82ca879a3d878af1eab3bc8a455cb.jpg)

<details>
<summary>line</summary>

| steps (k) | Ours | Ours w/o IntrinsicReward | Ours w/o Curriculum |
| --------- | ---- | ------------------------ | ------------------- |
| 0         | 0.0  | 0.0                      | 0.0                 |
| 250       | 0.9  | 0.85                     | 0.8                 |
| 500       | 0.95 | 0.9                      | 0.9                 |
| 750       | 0.98 | 0.95                     | 0.95                |
| 1000      | 0.99 | 0.98                     | 0.98                |
| 1250      | 0.995| 0.99                     | 0.99                |
| 1500      | 0.998| 0.995                    | 0.995               |
| 1750      | 0.999| 0.998                    | 0.998               |
| 2000      | 1.0  | 1.0                      | 1.0                 |
</details>

![](images/67eb53036b2f3aef5743a4d0769e957c159fd1a0e9c6a124deebf7c686ddb151.jpg)

<details>
<summary>line</summary>

| steps (k) | λ = 1(Default) | λ = 0.1 | λ = 0.5 | λ = 2 |
| --------- | --------------- | ------- | ------- | ----- |
| 0         | 0.3             | 0.3     | 0.3     | 0.3   |
| 250       | 0.9             | 0.9     | 0.9     | 0.9   |
| 500       | 0.95            | 0.95    | 0.95    | 0.95  |
| 750       | 0.8             | 0.8     | 0.8     | 0.8   |
| 1000      | 0.9             | 0.9     | 0.9     | 0.9   |
| 1250      | 0.95            | 0.95    | 0.95    | 0.95  |
| 1500      | 0.95            | 0.95    | 0.95    | 0.95  |
| 1750      | 0.95            | 0.95    | 0.95    | 0.95  |
| 2000      | 0.95            | 0.95    | 0.95    | 0.95  |
</details>

![](images/cc3001627aaf9fc2e15b62fbd04d461454f9d2ad18d7f26b369ded96f0bdde11.jpg)

<details>
<summary>line</summary>

| steps (k) | head=2(Default) | head=4 | head=6 |
| --------- | --------------- | ------ | ------ |
| 0         | 0.0             | 0.0    | 0.0    |
| 250       | 0.8             | 0.8    | 0.8    |
| 500       | 0.9             | 0.9    | 0.9    |
| 750       | 0.95            | 0.95   | 0.95   |
| 1000      | 0.98            | 0.98   | 0.98   |
| 1250      | 0.99            | 0.99   | 0.99   |
| 1500      | 0.995           | 0.995  | 0.995  |
| 1750      | 0.998           | 0.998  | 0.998  |
| 2000      | 1.0             | 1.0    | 1.0    |
</details>

![](images/25ffa47661a8dc931cc14762dc97918be3b530a8e987cdeba176d8ea8a7420f0.jpg)

<details>
<summary>line</summary>

| steps (k) | Ours | Ours w/o IntrinsicReward | Ours w/o Curriculum |
| --------- | ---- | ------------------------ | ------------------- |
| 0         | 0.4  | 0.4                      | 0.4                 |
| 250       | 0.9  | 0.9                      | 0.9                 |
| 500       | 0.95 | 0.95                     | 0.95                |
| 750       | 0.98 | 0.98                     | 0.98                |
| 1000      | 0.99 | 0.99                     | 0.99                |
| 1250      | 0.99 | 0.99                     | 0.99                |
| 1500      | 0.99 | 0.99                     | 0.99                |
| 1750      | 0.99 | 0.99                     | 0.99                |
| 2000      | 0.99 | 0.99                     | 0.99                |
</details>

![](images/d27336c82c52cd34ca0763046f77d16cb41ee9bebffc5cb548e8a327c247de44.jpg)

<details>
<summary>line</summary>

| steps (k) | λ = 1(Default) | λ = 0.1 | λ = 0.5 | λ = 2 |
| --------- | --------------- | ------- | ------- | ----- |
| 0         | 0.0             | 0.0     | 0.0     | 0.0   |
| 250       | 0.0             | 0.0     | 0.0     | 0.0   |
| 500       | 0.8             | 0.8     | 0.8     | 0.8   |
| 750       | 1.0             | 1.0     | 1.0     | 1.0   |
| 1000      | 1.0             | 0.4     | 1.0     | 1.0   |
| 1250      | 1.0             | 0.6     | 1.0     | 1.0   |
| 1500      | 1.0             | 0.6     | 1.0     | 1.0   |
| 1750      | 1.0             | 0.6     | 1.0     | 1.0   |
| 2000      | 1.0             | 0.6     | 1.0     | 1.0   |
</details>

![](images/c61479da84f49f85ca6616137eff8135fbabe95863c4a02e3179fb252c8cbb67.jpg)

<details>
<summary>line</summary>

| steps (k) | head=2(Default) | head=4 | head=6 |
| --------- | --------------- | ------ | ------ |
| 0         | 0.0             | 0.0    | 0.0    |
| 250       | 0.0             | 0.0    | 0.0    |
| 500       | 0.8             | 0.7    | 0.6    |
| 750       | 0.95            | 0.9    | 0.85   |
| 1000      | 1.0             | 1.0    | 1.0    |
| 1250      | 1.0             | 1.0    | 1.0    |
| 1500      | 1.0             | 1.0    | 1.0    |
| 1750      | 1.0             | 1.0    | 1.0    |
| 2000      | 1.0             | 1.0    | 1.0    |
</details>

![](images/96303362437674665e71bcddbb152cc907339f2ba203cf209d277dae5396f50c.jpg)

<details>
<summary>line</summary>

| steps (k) | Ours | Ours w/o IntrinsicReward | Ours w/o Curriculum |
| --------- | ---- | ------------------------ | ------------------- |
| 0         | 0.0  | 0.0                      | 0.0                 |
| 250       | 0.2  | 0.2                      | 0.2                 |
| 500       | 0.8  | 0.8                      | 0.4                 |
| 750       | 1.0  | 1.0                      | 0.6                 |
| 1000      | 1.0  | 1.0                      | 0.6                 |
| 1250      | 1.0  | 1.0                      | 0.6                 |
| 1500      | 1.0  | 1.0                      | 0.6                 |
| 1750      | 1.0  | 1.0                      | 0.6                 |
| 2000      | 1.0  | 1.0                      | 0.6                 |
</details>

![](images/a0a6aac6418928d4303562b3fa770e7e2f267b098347445ba8a8ff3ba1a5f4da.jpg)

<details>
<summary>line</summary>

| steps (k) | λ = Z(Default) | λ = 0.1 | λ = 0.5 | λ = 1 | λ = 4 |
| --------- | --------------- | ------- | ------- | ----- | ----- |
| 0         | 0.0             | 0.0     | 0.0     | 0.0   | 0.0   |
| 500       | 0.0             | 0.0     | 0.0     | 0.0   | 0.0   |
| 1000      | 0.6             | 0.3     | 0.2     | 0.1   | 0.1   |
| 1500      | 0.9             | 0.5     | 0.3     | 0.2   | 0.3   |
| 2000      | 1.0             | 0.7     | 0.4     | 0.3   | 0.5   |
| 2500      | 1.0             | 0.8     | 0.5     | 0.4   | 0.6   |
| 3000      | 1.0             | 0.9     | 0.6     | 0.5   | 0.7   |
</details>

![](images/bbfdb449aca7666585d97026441aa0da35c46552693455c7e87cd531c4424195.jpg)

<details>
<summary>line</summary>

| steps (k) | head=2(Default) | head=4 | head=6 |
| --------- | --------------- | ------ | ------ |
| 0         | 0.0             | 0.0    | 0.0    |
| 500       | 0.0             | 0.0    | 0.0    |
| 1000      | 0.5             | 0.0    | 0.0    |
| 1500      | 1.0             | 0.2    | 0.3    |
| 2000      | 1.0             | 0.3    | 0.4    |
| 2500      | 1.0             | 0.4    | 0.5    |
| 3000      | 1.0             | 0.5    | 0.6    |
</details>

![](images/afbe7a4cd6618a9c39c2a4ece93c2302a99aeab7e4468fafd22dcdb8bb97bd15.jpg)

<details>
<summary>line</summary>

| steps (k) | Ours | Ours w/o IntrinsicReward | Ours w/o Curriculum |
| --------- | ---- | ------------------------ | ------------------- |
| 0         | 0.0  | 0.0                      | 0.0                 |
| 500       | 0.0  | 0.0                      | 0.0                 |
| 1000      | 0.8  | 0.0                      | 0.0                 |
| 1500      | 1.0  | 0.0                      | 0.2                 |
| 2000      | 1.0  | 0.0                      | 0.4                 |
| 2500      | 1.0  | 0.0                      | 0.6                 |
| 3000      | 1.0  | 0.0                      | 0.6                 |
</details>

![](images/55126f5df6a200119b1b14e7882a7b3bde1a4accba339361fdda8248c6ac46d4.jpg)

<details>
<summary>line</summary>

| steps (k) | λ = 1(Default) | λ = 0.1 | λ = 0.5 | λ = 2 |
| --------- | --------------- | ------- | ------- | ----- |
| 0         | 0.0             | 0.0     | 0.0     | 0.0   |
| 500       | 0.3             | 0.25    | 0.2     | 0.15  |
| 1000      | 0.6             | 0.55    | 0.5     | 0.45  |
| 1500      | 0.8             | 0.75    | 0.7     | 0.65  |
| 2000      | 0.9             | 0.85    | 0.8     | 0.75  |
| 2500      | 0.95            | 0.9     | 0.85    | 0.8   |
| 3000      | 1.0             | 0.95    | 0.9     | 0.85  |
</details>

![](images/10ff22f49039e9e20f303ffb6a8126813bb318f52eb96b0edd04a3e614241edd.jpg)

<details>
<summary>line</summary>

| steps (k) | head=2(Default) | head=4 | head=6 |
| --------- | ---------------- | ------ | ------ |
| 0         | 0.0              | 0.0    | 0.0    |
| 500       | 0.3              | 0.3    | 0.2    |
| 1000      | 0.7              | 0.7    | 0.6    |
| 1500      | 0.9              | 0.9    | 0.8    |
| 2000      | 0.95             | 0.95   | 0.9    |
| 2500      | 0.98             | 0.98   | 0.95   |
| 3000      | 1.0              | 1.0    | 1.0    |
</details>

![](images/17dcbc11ccdc3aa595ff9890a10b84c99b317f723d88f6bc6d29612d253ba31f.jpg)

<details>
<summary>line</summary>

| steps (k) | Ours | Ours w/o IntrinsicReward | Ours w/o Curriculum |
| --------- | ---- | ------------------------ | ------------------- |
| 0         | 0.0  | 0.0                      | 0.0                 |
| 500       | 0.2  | 0.1                      | 0.1                 |
| 1000      | 0.6  | 0.3                      | 0.7                 |
| 1500      | 0.8  | 0.3                      | 0.9                 |
| 2000      | 0.9  | 0.3                      | 0.9                 |
| 2500      | 0.95 | 0.3                      | 0.9                 |
| 3000      | 1.0  | 0.3                      | 0.9                 |
</details>

![](images/4778ac261783a5be5124dc3d7eb8655843739893ce379de19cad609b91c61ef6.jpg)

<details>
<summary>line</summary>

| steps (k) | λ = 1(Default) | λ = 0.1 | λ = 0.5 | λ = 2 |
| --------- | --------------- | ------- | ------- | ----- |
| 0         | 0.0             | 0.0     | 0.0     | 0.0   |
| 500       | 0.2             | 0.1     | 0.1     | 0.1   |
| 1000      | 0.4             | 0.3     | 0.3     | 0.3   |
| 1500      | 0.6             | 0.5     | 0.5     | 0.5   |
| 2000      | 0.8             | 0.7     | 0.7     | 0.7   |
| 2500      | 0.9             | 0.8     | 0.8     | 0.8   |
| 3000      | 1.0             | 0.9     | 0.9     | 0.9   |
</details>

(a) Auxiliary loss weight $\lambda$

![](images/6c04c9dc3bd9294df8e799c8b1c71d72c1b9c80640930952d95de1e90d5f5206.jpg)

<details>
<summary>line</summary>

| steps (k) | head=2(Default) | head=4 | head=6 |
| --------- | ---------------- | ------ | ------ |
| 0         | 0.0              | 0.0    | 0.0    |
| 500       | 0.1              | 0.1    | 0.1    |
| 1000      | 0.4              | 0.4    | 0.4    |
| 1500      | 0.7              | 0.7    | 0.7    |
| 2000      | 0.9              | 0.9    | 0.9    |
| 2500      | 1.0              | 1.0    | 1.0    |
| 3000      | 1.0              | 1.0    | 1.0    |
</details>

(b) Number of prediction heads

![](images/16bd4d3cb0e9509685f0f54a11e1c5a8baf6b662bed850c02cdcad07a0090811.jpg)

<details>
<summary>line</summary>

| steps (k) | Ours | Ours w/o IntrinsicReward | Ours w/o Curriculum |
| --------- | ---- | ------------------------ | ------------------- |
| 0         | 0.0  | 0.0                      | 0.0                 |
| 500       | 0.0  | 0.0                      | 0.0                 |
| 1000      | 0.6  | 0.0                      | 0.8                 |
| 1500      | 0.8  | 0.1                      | 0.9                 |
| 2000      | 0.9  | 0.1                      | 0.95                |
| 2500      | 0.95 | 0.1                      | 0.98                |
| 3000      | 1.0  | 0.2                      | 1.0                 |
</details>

(c) Intrinsic reward & Curriculum   
Figure 10: Ablation study in terms of the episode success rate. First row: Complex-Maze. Second row: Medium-Maze. Third row: Spiral-Maze. Fourth row: Ant Locomotion. Fifth row: Sawyer Push. Sixth row: Sawyer Pick & Place. The shaded area represents a standard deviation across 5 seeds.

Curriculum learning objective type. We conduct additional experiments to validate whether reflecting the temporal distance in a curriculum learning objective (Eq (7)) is required since there are a few works that estimate the temporal distance from the initial state distribution to propose the curriculum goals in a temporally distant region or explore based on this temporal information [5, 37]. To reflect the temporal distance in the cost function (Eq (7)), we modify it as $w(s_i, g_i^+) := \mathcal{C}\mathcal{E}(p_{\text{pseudo}}(y = 1|s_i; g_i^+); y = p_{\text{pseudo}}(y = 1|g_i^+; g_i^+)) - V^\pi(s_0, \phi(s_i)) (\phi(\cdot)$ is goal space mapping) since the value function itself implicitly represents the temporal distance if we use the sparse reward or custom-defined reward similar to the sparse one. In this case, our proposed intrinsic reward outputs 1 for the desired goal and 0 for the explored states, and it works similarly to the sparse one.

We experimented with this modified curriculum learning objective (+Value), and the results are shown in Figure 11, 12. It shows that there is no significant difference, which supports the superiority of our method in that our method achieves state-of-the-art results without considering additional temporal distance information.

![](images/0382c664a76df5c0a2861871d9e374d3cd9898e14a1cbaec1bf35f6b20f14bb7.jpg)

<details>
<summary>line</summary>

| steps (k) | Ours | Ours + Value |
| --------- | ---- | ------------ |
| 0         | 12   | 12           |
| 250       | 0    | 0            |
| 500       | 0    | 0            |
| 750       | 0    | 0            |
| 1000      | 0    | 0            |
| 1250      | 0    | 0            |
| 1500      | 0    | 0            |
| 1750      | 0    | 0            |
| 2000      | 0    | 0            |
</details>

(a) Complex-Maze

![](images/d887ec36665d120055bfa57f407a5e0c0eab913902aec06755fe5eb601ae852a.jpg)

<details>
<summary>line</summary>

| steps (k) | Ours | Ours + Value |
| --------- | ---- | ------------ |
| 0         | 12   | 8            |
| 250       | 0    | 0            |
| 500       | 0    | 0            |
| 750       | 0    | 0            |
| 1000      | 0    | 0            |
| 1250      | 0    | 0            |
| 1500      | 0    | 0            |
| 1750      | 0    | 0            |
| 2000      | 0    | 0            |
</details>

(b) Medium-Maze

![](images/923e176df8547d825b3e074127080134c996a4457b85f3016f10bdb54940f98c.jpg)

<details>
<summary>line</summary>

| steps (k) | Ours | Ours + Value |
| --------- | ---- | ------------ |
| 0         | 15   | 15           |
| 250       | 25   | 26           |
| 500       | 10   | 10           |
| 750       | 0    | 0            |
| 1000      | 0    | 0            |
| 1250      | 0    | 0            |
| 1500      | 0    | 0            |
| 1750      | 0    | 0            |
| 2000      | 0    | 0            |
</details>

(c) Spiral-Maze

![](images/2e9f97ea93a345e34b65fa43e91942268cf9626d5455f7fa440e9b6fa18c212e.jpg)

<details>
<summary>line</summary>

| steps (k) | Ours | Ours + Value |
| --------- | ---- | ------------ |
| 0         | 10   | 10           |
| 500       | 6    | 7            |
| 1000      | 2    | 4            |
| 1500      | 1    | 1            |
| 2000      | 1    | 1            |
| 2500      | 1    | 1            |
| 3000      | 1    | 1            |
</details>

(d) Ant Locomotion

![](images/f60f5992957297d6197f1312f083c9e71e76052fee8ea87136a5d8bc2f905e8a.jpg)

<details>
<summary>line</summary>

| steps (k) | Ours  | Ours + Value |
| --------- | ----- | ------------ |
| 0         | 0.6   | 0.6          |
| 500       | 0.4   | 0.45         |
| 1000      | 0.2   | 0.25         |
| 1500      | 0.1   | 0.1          |
| 2000      | 0.05  | 0.05         |
| 2500      | 0.05  | 0.05         |
| 3000      | 0.05  | 0.05         |
</details>

(e) Sawyer Push

![](images/6dcb130ca2e858575a2796cbe23ce70d73204ed3fa43f005ac19f928f93b0353.jpg)

<details>
<summary>line</summary>

| steps (k) | Ours  | Ours + Value |
| --------- | ----- | ------------ |
| 0         | 0.6   | 0.6          |
| 500       | 0.4   | 0.4          |
| 1000      | 0.2   | 0.2          |
| 1500      | 0.1   | 0.1          |
| 2000      | 0.05  | 0.05         |
| 2500      | 0.02  | 0.02         |
| 3000      | 0.01  | 0.01         |
</details>

(f) Sawyer Pick&Place

Figure 11: Ablation study in terms of the average distance from the curriculum goals to the final goals (Lower is better). +Value means that we additionally consider the value function bias in the curriculum learning objective to reflect the temporal distance from the initial state distribution.   
![](images/8c537605dd478f890ed4b65e545c2cd0b37386de4d2bd881d2913eae46cdf207.jpg)

<details>
<summary>line</summary>

| steps (k) | Ours | Ours + Value |
| --------- | ---- | ------------ |
| 0         | 0.0  | 0.0          |
| 250       | 0.9  | 0.9          |
| 500       | 1.0  | 1.0          |
| 750       | 1.0  | 1.0          |
| 1000      | 1.0  | 1.0          |
| 1250      | 1.0  | 1.0          |
| 1500      | 1.0  | 1.0          |
| 1750      | 1.0  | 1.0          |
| 2000      | 1.0  | 1.0          |
</details>

(a) Complex-Maze

![](images/9bc984c36ca3e076a3de16213aedf2ce6e3d885c438b8c7e5e558253a178968c.jpg)

<details>
<summary>line</summary>

| steps (k) | Ours | Ours + Value |
| --------- | ---- | ------------ |
| 0         | 0.0  | 0.0          |
| 250       | 0.8  | 0.7          |
| 500       | 0.95 | 0.9          |
| 750       | 0.98 | 0.95         |
| 1000      | 0.97 | 0.96         |
| 1250      | 0.98 | 0.97         |
| 1500      | 0.99 | 0.98         |
| 1750      | 0.99 | 0.99         |
| 2000      | 1.0  | 1.0          |
</details>

(b) Medium-Maze

![](images/659baf5c401d2df6899c9cf31652200e5d05a607ff28e96f76a4525f75a1048d.jpg)

<details>
<summary>line</summary>

| steps (k) | Ours | Ours + Value |
| --------- | ---- | ------------ |
| 0         | 0.0  | 0.0          |
| 250       | 0.0  | 0.0          |
| 500       | 0.8  | 0.8          |
| 750       | 0.95 | 0.95         |
| 1000      | 0.98 | 0.98         |
| 1250      | 0.99 | 0.99         |
| 1500      | 0.99 | 0.99         |
| 1750      | 0.99 | 0.99         |
| 2000      | 0.99 | 0.99         |
</details>

(c) Spiral-Maze

![](images/74f61ec87df59a91c8e84854e0f40d0bbf903b24bfd840d841e765847dbb3c25.jpg)

<details>
<summary>line</summary>

| steps (k) | Ours | Ours + Value |
| --------- | ---- | ------------ |
| 0         | 0.0  | 0.0          |
| 500       | 0.2  | 0.1          |
| 1000      | 0.6  | 0.4          |
| 1500      | 0.9  | 0.8          |
| 2000      | 1.0  | 0.95         |
| 2500      | 1.0  | 1.0          |
| 3000      | 1.0  | 1.0          |
</details>

(d) Ant Locomotion

![](images/3ac813136f940141a86202a18313dc58332e72842e245b985f8833ababc8a80b.jpg)

<details>
<summary>line</summary>

| steps (k) | Ours | Ours + Value |
| --------- | ---- | ------------ |
| 0         | 0.0  | 0.0          |
| 500       | 0.2  | 0.15         |
| 1000      | 0.6  | 0.5          |
| 1500      | 0.85 | 0.8          |
| 2000      | 0.95 | 0.9          |
| 2500      | 0.98 | 0.97         |
| 3000      | 1.0  | 1.0          |
</details>

(e) Sawyer Push

![](images/7e2d9147f8ea4a82869b67bef488d001225687984171e697983ac5cb1fb6aae6.jpg)

<details>
<summary>line</summary>

| steps (k) | Ours | Ours + Value |
| --------- | ---- | ------------ |
| 0         | 0.0  | 0.0          |
| 500       | 0.1  | 0.05         |
| 1000      | 0.4  | 0.2          |
| 1500      | 0.7  | 0.5          |
| 2000      | 0.9  | 0.8          |
| 2500      | 0.95 | 0.9          |
| 3000      | 1.0  | 1.0          |
</details>

(f) Sawyer Pick&Place   
Figure 12: Ablation study in terms of the episode success rates. +Value means that we additionally consider the value function bias in the curriculum learning objective to reflect the temporal distance from the initial state distribution.

Choice of goal candidates in training conditional classifiers. As mentioned in the main script, we also experimented with different choices of the goal candidates when we train the conditional classifiers (Eq (5)). The default setting is $D_{G} = D_{T}$ , and we also experimented with $\mathcal{D}_{\mathrm{G}} = \mathcal{B} \cup p^{+}(g)$ . The results are shown in Figure 13, 14. It shows that there is no significant difference, which means we can even make the problem setting more strict by conditioning the classifier only with the visited states and the given desired outcome examples.

![](images/b3c7bbabebe0f7d58725de31cd3e10855ae39c44e613e89b1cf6103b089a54c8.jpg)

<details>
<summary>line</summary>

| steps (k) | distance (D₀ = Dₜ) | distance (D₀ = B ∪ p⁺(g)) |
| --------- | ------------------- | -------------------------- |
| 0         | 12.0                | 12.0                       |
| 250       | 0.5                 | 0.5                        |
| 500       | 0.1                 | 0.1                        |
| 750       | 0.05                | 0.05                       |
| 1000      | 0.02                | 0.02                       |
| 1250      | 0.01                | 0.01                       |
| 1500      | 0.005               | 0.005                      |
| 1750      | 0.002               | 0.002                      |
| 2000      | 0.001               | 0.001                      |
</details>

(a) Complex-Maze

![](images/7b76ef6e31896d428ae4dfc512377c8e3ad1fb19aa6c7ed008b3f2a43bb54212.jpg)

<details>
<summary>line</summary>

| steps (k) | distance (D_G = D_T) | distance (D_G = B ∪ p^*(g)) |
| --------- | --------------------- | ---------------------------- |
| 0         | 12.0                  | 12.0                         |
| 250       | 0.5                   | 0.5                          |
| 500       | 0.3                   | 0.3                          |
| 750       | 0.2                   | 0.2                          |
| 1000      | 0.1                   | 0.1                          |
| 1250      | 0.1                   | 0.1                          |
| 1500      | 0.1                   | 0.1                          |
| 1750      | 0.1                   | 0.1                          |
| 2000      | 0.1                   | 0.1                          |
</details>

(b) Medium-Maze

![](images/37f791ad94f807b5cebcf056bcb411ccb875dc8ff533841d167a0092f3d5ea51.jpg)

<details>
<summary>line</summary>

| steps (k) | distance (D₀ = Dₜ) | distance (D₀ = B ∪ p*(g)) |
| --------- | ------------------- | -------------------------- |
| 0         | 15                  | 18                         |
| 250       | 25                  | 24                         |
| 500       | 10                  | 5                          |
| 750       | 0                   | 0                          |
| 1000      | 0                   | 0                          |
| 1250      | 0                   | 0                          |
| 1500      | 0                   | 0                          |
| 1750      | 0                   | 0                          |
| 2000      | 0                   | 0                          |
</details>

(c) Spiral-Maze

![](images/7e6a1cf67b38f2186969d3329d19b3eec1227711d66a6129797341e0cad3a007.jpg)

<details>
<summary>line</summary>

| steps (k) | distance (D_G = D_T) | distance (D_G = B ∪ p^+(g)) |
| --------- | -------------------- | --------------------------- |
| 0         | 10.0                 | 10.0                        |
| 500       | 6.0                  | 7.0                         |
| 1000      | 3.0                  | 4.0                         |
| 1500      | 1.0                  | 2.0                         |
| 2000      | 0.5                  | 1.5                         |
| 2500      | 0.3                  | 1.2                         |
| 3000      | 0.2                  | 1.0                         |
</details>

(d) Ant Locomotion

![](images/3af52b7160be98c8c6c3158ee8df9dab3442004b6b38cfeac09a075f949cdd47.jpg)

<details>
<summary>line</summary>

| steps (k) | distance (D₀ = Dₜ) | distance (D₀ = B ∪ p⁺(g)) |
| --------- | ------------------- | -------------------------- |
| 0         | 0.6                 | 0.5                        |
| 500       | 0.4                 | 0.3                        |
| 1000      | 0.2                 | 0.1                        |
| 1500      | 0.1                 | 0.05                       |
| 2000      | 0.05                | 0.02                       |
| 2500      | 0.02                | 0.01                       |
| 3000      | 0.01                | 0.01                       |
</details>

(e) Sawyer Push

![](images/9a9d52b55c5975562bb9417e413bc1b9c88ce0ad1c675ca464f596172eb0d02f.jpg)

<details>
<summary>line</summary>

| steps (k) | D_G = D_T | D_G = B ∪ p^+(g) |
| --------- | --------- | ---------------- |
| 0         | 0.6       | 0.2              |
| 500       | 0.4       | 0.5              |
| 1000      | 0.3       | 0.3              |
| 1500      | 0.1       | 0.1              |
| 2000      | 0.0       | 0.0              |
| 2500      | 0.0       | 0.0              |
| 3000      | 0.0       | 0.0              |
</details>

(f) Sawyer Pick&Place

Figure 13: Ablation study in terms of the average distance from the curriculum goals to the final goals (Lower is better). There are no significant differences between the choice of goal candidates to train the conditional classifiers.   
![](images/67e9e954164293860a2c92030eeef6edf2e1514bea5fbbba9f329d93f8f6ee32.jpg)

<details>
<summary>line</summary>

| steps (k) | success_rate (D_G = D_T) | success_rate (D_G = B ∪ p⁺(g)) |
| --------- | ------------------------ | ------------------------------- |
| 0         | 0.0                      | 0.0                             |
| 250       | 0.9                      | 0.9                             |
| 500       | 1.0                      | 1.0                             |
| 750       | 1.0                      | 1.0                             |
| 1000      | 1.0                      | 1.0                             |
| 1250      | 1.0                      | 1.0                             |
| 1500      | 1.0                      | 1.0                             |
| 1750      | 1.0                      | 1.0                             |
| 2000      | 1.0                      | 1.0                             |
</details>

(a) Complex-Maze

![](images/0c8675600abc513cac2ce9364a5dc1c8132e6aeb4248022c4f106bc96176dbbc.jpg)

<details>
<summary>line</summary>

| steps (k) | D_G = D_T | D_G = B ∪ p^+(g) |
| --------- | --------- | ---------------- |
| 0         | 0.4       | 0.4              |
| 250       | 0.9       | 0.9              |
| 500       | 0.95      | 0.95             |
| 750       | 0.9       | 0.9              |
| 1000      | 0.95      | 0.95             |
| 1250      | 0.95      | 0.95             |
| 1500      | 0.95      | 0.95             |
| 1750      | 0.95      | 0.95             |
| 2000      | 0.95      | 0.95             |
</details>

(b) Medium-Maze

![](images/08386332eddea50392c7fcbff637bb12f26566d739cb67486d8553136f614cca.jpg)

<details>
<summary>line</summary>

| steps (k) | D₀ = Dₜ | D₀ = B ∪ ρ⁺(g) |
| --------- | ------- | -------------- |
| 0         | 0.0     | 0.0            |
| 250       | 0.0     | 0.0            |
| 500       | 0.8     | 0.9            |
| 750       | 0.95    | 0.98           |
| 1000      | 0.98    | 0.99           |
| 1250      | 0.99    | 0.995          |
| 1500      | 0.995   | 0.998          |
| 1750      | 0.998   | 0.999          |
| 2000      | 0.999   | 1.0            |
</details>

(c) Spiral-Maze

![](images/42b3e13ae609f3f2e9e882e4ef5c65e7414e8de1fa26e261956b97bdf11dcdb1.jpg)

<details>
<summary>line</summary>

| steps (k) | success rate (D₀ = Dₜ) | success rate (D₀ = B ∪ p⁺(g)) |
| --------- | ------------------------ | ------------------------------- |
| 0         | 0.0                      | 0.0                             |
| 500       | 0.0                      | 0.0                             |
| 1000      | 0.5                      | 0.4                             |
| 1500      | 0.9                      | 0.7                             |
| 2000      | 1.0                      | 0.8                             |
| 2500      | 1.0                      | 0.9                             |
| 3000      | 1.0                      | 1.0                             |
</details>

(d) Ant Locomotion

![](images/3d297b6150d98f5aab98d45f826d7c9e2da9b488e079ee966ef7f4282ab40e6f.jpg)

<details>
<summary>line</summary>

| steps (k) | success_rate_DG = D_T | success_rate_DG = B ∪ p⁺(g) |
| --------- | --------------------- | ---------------------------- |
| 0         | 0.0                   | 0.0                          |
| 500       | 0.2                   | 0.1                          |
| 1000      | 0.6                   | 0.5                          |
| 1500      | 0.9                   | 0.8                          |
| 2000      | 1.0                   | 0.9                          |
| 2500      | 1.0                   | 1.0                          |
| 3000      | 1.0                   | 1.0                          |
</details>

(e) Sawyer Push

![](images/6793d7eb8bf20b52f56c2fe43ff421f19bd02bafe1d3573d1bbecbc24357c003.jpg)

<details>
<summary>line</summary>

| steps (k) | success_rate (D₀ = Dₜ) | success_rate (D₀ = B ∪ p⁺(g)) |
| --------- | ------------------------ | ------------------------------- |
| 0         | 0.0                      | 0.0                             |
| 500       | 0.0                      | 0.0                             |
| 1000      | 0.4                      | 0.3                             |
| 1500      | 0.8                      | 0.7                             |
| 2000      | 0.9                      | 0.85                            |
| 2500      | 0.95                     | 0.9                             |
| 3000      | 1.0                      | 0.95                            |
</details>

(f) Sawyer Pick&Place   
Figure 14: Ablation study in terms of the episode success rates. There are no significant differences between the choice of goal candidates to train the conditional classifiers.