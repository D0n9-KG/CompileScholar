# Bootstrapped Training of Score-Conditioned Generator for Offline Design of Biological Sequences

Minsu Kim $^{1}$ Federico Berto $^{1}$ Sungsoo Ahn $^{2}$ Jinkyoo Park $^{1}$

$^{1}$ Korea Advanced Institute of Science and Technology (KAIST)

$^{2}$ Pohang University of Science and Technology (POSTECH)

{min-su, fberto, jinkyoo.park}@kaist.ac.kr

{sungsoo.ahn}@postech.ac.kr

# Abstract

We study the problem of optimizing biological sequences, e.g., proteins, DNA, and RNA, to maximize a black-box score function that is only evaluated in an offline dataset. We propose a novel solution, bootstrapped training of score-conditioned generator (BOOTGEN) algorithm. Our algorithm repeats a two-stage process. In the first stage, our algorithm trains the biological sequence generator with rank-based weights to enhance the accuracy of sequence generation based on high scores. The subsequent stage involves bootstrapping, which augments the training dataset with self-generated data labeled by a proxy score function. Our key idea is to align the score-based generation with a proxy score function, which distills the knowledge of the proxy score function to the generator. After training, we aggregate samples from multiple bootstrapped generators and proxies to produce a diverse design. Extensive experiments show that our method outperforms competitive baselines on biological sequential design tasks. We provide reproducible source code: https://github.com/kaist-silab/bootgen.

# 1 Introduction

The automatic design of biological sequences, e.g., DNA, RNA, and proteins, with a specific property, e.g., high binding affinity, is a vital task within the field of biotechnology $[5, 54, 41, 33]$ . To solve this problem, researchers have developed algorithms to optimize a biological sequence to maximize a score function $[40, 9, 10, 2, 29]$ . Here, the main challenge is the expensive evaluation of the score function that requires experiments in a laboratory setting or clinical trials.

To resolve this issue, recent works have investigated offline model-based optimization $[32, 21, 44, 52, 12, 45, MBO]$ . Given an offline dataset of biological sequences paired with scores, offline MBO algorithms train a proxy for the score function, e.g., a deep neural network (DNN), and maximize the proxy function without querying the true score function. Therefore, such offline MBO algorithms bypass the expense of iteratively querying the true score function whenever a new solution is proposed. However, even optimizing such a proxy function is challenging due to the vast search space over the biological sequences.

On the one hand, several works $[21, 44, 52, 12]$ considered applying gradient-based maximization of the proxy function. However, when the proxy function is parameterized using a DNN, these methods often generate solutions where the true score is low despite the high proxy score. This is due to the fragility of DNNs against adversarial optimization of inputs $[52, 44, 21]$ . Furthermore, the gradient-based methods additionally require reformulating biological sequence optimization as a continuous optimization, e.g., continuous relaxation $[21, 44, 12]$ or mapping discrete designs to a continuous latent space $[52]$ .

On the other hand, one may consider training deep generative models to learn a distribution over high-scoring designs $[32, 29]$ . They learn to generate solutions from scratch, which amortizes optimization over the design space.

To be specific, Kumar and Levine $[32]$ suggests learning an inverse map from a score to a solution with a focus on generating high-scoring solutions. Next, Jain et al. $[29]$ proposed training a generative flow network $[7, GFN]$ as the generative distribution of high-scoring solutions.

Contribution In this paper, we propose a bootstrapped training of score-conditioned generator (BOOTGEN) for the offline design of biological sequences. Our key idea is to enhance the score-conditioned generator by suggesting a variation of the classical ensemble strategy of bootstrapping and aggregating. We train multiple generators using bootstrapped datasets from training and combine them with proxy models to create a reliable and diverse sampling solution.

In the bootstrapped training, we aim to align a score-conditioned generator with a proxy function by bootstrapping the training dataset of the generator. To be specific, we repeat multiple stages of (1) training the conditional generator on the training dataset with a focus on high-scoring sequences and (2) augmenting the training dataset using sequences that are sampled from the generator and labeled using the proxy function. Intuitively, our framework improves the score-to-sequence mapping (generator) to be consistent with the sequence-to-score mapping (proxy function), which is typically more accurate.

When training the score-conditioned generator, we assign high rank-based weights $[46]$ to high-scoring sequences. Sequences that are highly ranked among the training dataset are more frequently sampled to train the generator. This leads to shifting the training distribution towards an accurate generation of high-scoring samples. Compared with the value-based weighting scheme previously proposed by Kumar and Levine $[32]$ , the rank-based weighting scheme is more robust to the change of training dataset from bootstrapping.

To further boost the performance of our algorithm, we propose two post-processing processes after the training: filtering and diversity aggregation (DA). The filtering process aims to filter samples from generators using the proxy function to gather samples with cross-agreement between the proxy and generator. On the other hand, DA collects sub-samples from multiple generators and combines them into complete samples. DA enables diverse decision-making with reduced variance in generating quality, as it collects samples from multiple bootstrapped generators.

We perform extensive experiments on six offline biological design tasks: green fluorescent protein design $[54, GFP]$ , DNA optimization for expression level on an untranslated region $[41, UTR]$ , transcription factor binding $[5, TFBind8]$ , and RNA optimization for binding to three types of transcription factors $[33, RNA-Binding]$ . Our BOOTGEN demonstrates superior performance, surpassing the 100th percentile score and 50th percentile score of the design-bench baselines $[45]$ , a generative flow network (GFN)-based work $[29]$ and bidirectional learning method (BDI) $[12]$ . Furthermore, we additionally verify the superior performance of BOOTGEN in various design scenarios, particularly when given a few opportunities to propose solutions.

# 2 Related Works

# 2.1 Automatic Design of Biological Sequences

Researchers have investigated machine learning methods to automatically design biological sequences, e.g., Bayesian optimization $[50, 6, 36, 38, 43]$ , evolutionary methods $[3, 8, 18, 4, 40]$ , model-based reinforcement learning $[2]$ , and generative methods $[9, 32, 20, 29, 19, 11, 14]$ . These methods aim to optimize biological sequence (e.g., protein, RNA, and DNA) for maximizing the target objective of binding activity and folding, which has crucial application in drug discovery and health care $[29]$ .

# 2.2 Offline Model-based Optimization

Offline model-based optimization (MBO) aims to find the design x that maximizes a score function $f(\boldsymbol{x})$ , using only pre-collected offline data. The most common approach is to use gradient-based optimization on differentiable proxy models trained on an offline dataset [49, 44, 52, 21, 12]. How-

![](images/7a77745690c99515f119ac6974862141508c45396d6c814266814fbcc8b017b2.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Step A: Rank-based Weighted Training"] --> B["Training Dataset"]
    B --> C["Density y"]
    C --> D["Reweighting"]
    D --> E["Density y"]
    E --> F["Reweighted Dataset"]
    F --> G["Generator pθ(x|y)"]
    G --> H["Generator pθ(x|y†)"]
    H --> I["Proxy fφ(x)"]
    I --> J["Labelling"]
    J --> K["Generator pθ(x|y†)"]
    K --> L["Step B: Bootstrapping"]
    L --> M["Top K"]
    M --> N["Kernel Top K"]
    N --> O["Bootstrapping"]
    O --> P["Augmented Dataset"]
    P --> Q["Density y"]
    Q --> R["High Score y†"]
    R --> S["L"]
    S --> T["Output"]
```
</details>

Figure 2.1: Illustration of the bootstrapped training process for learning score-conditioned generator.

ever, they can lead to poor scores due to the non-smoothness of the proxy landscape. To address this issue, Trabucco et al. [44] proposed conservative objective models (COMs), which use adversarial training to create a smooth proxy. Yu et al. [52] suggests the direct imposition of a Gaussian prior on the proxy model to create a smooth landscape and model adaptation to perform robust estimation on the specific set of candidate design space. While these methods are effective for high-dimensional continuous design tasks, their performance on discrete spaces is often inferior to classical methods, e.g., gradient ascent [45].

# 2.3 Bootstrapping

Bootstrapping means maximizing the utilization of existing resources. In a narrow sense, it refers to a statistical method where the original dataset is repeatedly sampled to create various datasets $[25]$ . In a broader sense, it also encompasses concepts frequently used in machine learning to improve machine learning scenarios where the label is expensive, e.g., self-training and semi-supervised learning $[37, 1, 22]$ . These methods utilize an iterative training scheme to augment the dataset with self-labeled samples with high confidence.

The bootstrapping strategy at machine learning showed great success in various domains, e.g., fully-labeled classification $[51]$ , self-supervised learning $[17]$ and offline reinforcement learning $[48]$ . We introduce a novel bootstrapping strategy utilizing score-conditioned generators and apply it to offline biological sequence design, addressing the challenge of working with a limited amount of poor-quality offline datasets.

# 2.4 Design by Conditional Generation

Conditional generation is a promising method with several high-impact applications, e.g., class-conditional image generation $[35]$ , language-to-image generation $[39]$ , reinforcement learning $[16]$ . With the success of conditional generation, several studies proposed to use it for design tasks, e.g., molecule and biological sequence design.

Hottung et al. [27] proposed an instance-conditioned variational auto-encoder [31] for routing problems, which can generate near-optimal routing paths conditioned on routing instances. Igashov et al. [28] suggested a conditional diffusion model to generate 3D molecules given their fragments. Specifically, the molecular fragments are injected into the latent space of the diffusion model, and the diffusion model generates links between fragments to make the 3D molecular compound.

# 3 Bootstrapped Training of Score-Conditioned Generator (BOOTGEN)

Problem definition We are interested in optimizing a biological sequence x to maximize a given score function $f(\boldsymbol{x})$ . We consider an offline setting where, during optimization, we do not have access to the score function $f(\boldsymbol{x})$ . Instead, we optimize the biological sequences using a static dataset $\mathcal{D} = \{(\boldsymbol{x}_{n}, y_{n})\}_{n=1}^{N}$ consisting of offline queries $y_{n} = f(\boldsymbol{x}_{n})$ to the score function. Finally, we consider evaluating a set of sequences $\{\boldsymbol{x}_{m}\}_{m=1}^{M}$ as an output of offline design algorithms.

Overview of BOOTGEN We first provide a high-level description of our bootstrapped training of score-conditioned generator, coined BOOTGEN. Our key idea is to align the score-conditioned generation with a proxy model via bootstrapped training (i.e., we train the generator on sequences labeled using the proxy model) and aggregate the decisions over multiple generators and proxies for reliable and diverse sampling of solutions.

Algorithm 1 Bootrapped Training of Score-conditioned generators   
1: Input: Offline dataset $\mathcal{D} = \{\boldsymbol{x}_n, y_n\}_{n=1}^N$ .
2: Update $\phi$ to minimize $\sum_{(\boldsymbol{x},y) \in \mathcal{D}} (f_\phi(\boldsymbol{x}) - y)^2$ .
3: for $j = 1, \ldots, N_{\text{gen}}$ do
4: Initialize $\mathcal{D}_{\text{tr}} \leftarrow \mathcal{D}$ .
5: for $i = 1, \ldots, I$ do
6: Update $\theta_j$ to maximize $\sum_{(\boldsymbol{x},y) \in \mathcal{D}_{\text{tr}}} w(y, \mathcal{D}_{\text{tr}}) \log p_{\theta_j}(\boldsymbol{x}|y)$ .
7: Sample $\boldsymbol{x}_\ell^* \sim p_{\theta_j}(\boldsymbol{x}|y^\dagger)$ for $\ell = 1, \ldots, L$ .
8: Set $y_\ell^* \leftarrow f_\phi(x_\ell^*)$ for $\ell = 1, \ldots, L$ .
9: Set $\mathcal{D}_{\text{aug}}$ as top- $K$ scoring samples in $\{\boldsymbol{x}_\ell^*, y_\ell^*\}_{\ell=1}^L$ .
10: Set $\mathcal{D}_{\text{tr}} \leftarrow \mathcal{D}_{\text{tr}} \cup \mathcal{D}_{\text{aug}}$ .
11: end for
12: end for
13: Output: trained score-conditioned generators $p_{\theta_1}(\boldsymbol{x}|y), ..., p_{\theta_{N_{\text{gen}}}}(\boldsymbol{x}|y)$ .

Before BOOTGEN training, we pre-train proxy score function $f_{\phi}(\boldsymbol{x}) \approx f(\boldsymbol{x})$ only leveraging offline dataset D. After that, our BOOTGEN first initializes a training dataset $D_{tr}$ as the offline dataset D and then repeats the following steps:

A. BootGen optimizes the score-conditioned generator $p_{\theta}(\boldsymbol{x}|y)$ using the training dataset $D_{tr}$ . During training, it assigns rank-based weights to each sequence for the generator to focus on high-scoring samples.   
B. BOOTGEN bootstraps the training dataset $D_{tr}$ using samples from the generator $p_{\theta}(\boldsymbol{x}|y^{\dagger})$ conditioned on the desired score $y^{\dagger}$ . It uses a proxy $f_{\phi}(\boldsymbol{x})$ of the score function to label the new samples.

After BOOTGEN training for multiple score-conditioned generators $p_{\theta_{1}}(\boldsymbol{x}|y), \ldots, p_{\theta_{n}}(\boldsymbol{x}|y)$ , we aggregate samples from the generators with filtering of proxy score function $f_{\phi}$ to generate diverse and reliable samples. We provide the pseudo-code of the overall procedure in Algorithm 1 for training and Algorithm 2 for generating solutions from the trained model.

# 3.1 Rank-based Weighted Training

Here, we introduce our framework to train the score-conditioned generator. Our algorithm aims to train the score-conditioned generator with more focus on generating high-scoring designs. Such a goal is helpful for bootstrapping and evaluation of our framework, where we query the generator conditioned on a high score.

Given a training dataset $D_{tr}$ , our BOOTGEN minimizes the following loss function:

$$
\mathcal {L} (\theta) := - \sum_ {(\boldsymbol {x}, y) \in \mathcal {D} _ {\mathrm{tr}}} w (y, \mathcal {D} _ {\mathrm{tr}}) \log p _ {\theta} (\boldsymbol {x} | y), \quad w (y, \mathcal {D} _ {\mathrm{tr}}) = \frac {(k | \mathcal {D} _ {\mathrm{tr}} | + \operatorname{rank} (y , \mathcal {D} _ {\mathrm{tr}})) ^ {- 1}}{\sum_ {(\boldsymbol {x} , y) \in \mathcal {D} _ {\mathrm{tr}}} (k | \mathcal {D} _ {\mathrm{tr}} | + \operatorname{rank} (y , \mathcal {D} _ {\mathrm{tr}})) ^ {- 1}}.
$$

where $w(y, \mathcal{D}_{\mathrm{tr}})$ is the score-wise rank-based weight [46]. Here, k is a weight-shifting factor, and $\operatorname{rank}(y, \mathcal{D}_{\mathrm{tr}})$ denotes the relative ranking of a score y with respect to the set of scores in the dataset $D_{tr}$ . We note that a small weight-shifting factor k assigns high weights to high-scoring samples.

For mini-batch training of the score-conditioned generator, we approximate the loss function $\mathcal{L}(\theta)$ via sampling with probability $w(y, \mathcal{D}_{\mathrm{tr}})$ for each sample $(\boldsymbol{x}, y)$ .

We note that Tripp et al. [46] proposed the rank-based weighting scheme for training unconditional generators to solve online design problems. At a high level, the weighting scheme guides the generator to focus more on generating high-scoring samples. Compared to weights that are proportional to scores [32], using the rank-based weights promotes the training to be more robust against outliers, e.g., samples with abnormally high weights. To be specific, the weighting factor $w(y, \mathcal{D})$ is less affected by outliers due to its upper bound that is achieved when $\text{rank}(y, \mathcal{D}) = 1$ .

Algorithm 2 Aggregation Strategy for Sample Generation   
1: Input: Trained score-conditioned generators $p_{\theta_{1}}(\boldsymbol{x}|y), \ldots, p_{\theta_{N_{\text{gen}}}}(\boldsymbol{x}|y)$ , and trained proxy function $f_{\phi}(\boldsymbol{x})$ .
2: Initialize $D_{samples} \leftarrow \emptyset$ .
3: for $i = 1, \ldots, N_{gen}$ do
4: Sample $x_{m}^{*} \sim p_{\theta_{i}}(x|y^{\dagger})$ for $m \in [M]$ .
5: Set $y_{m}^{*} \leftarrow f_{\phi}(x_{m}^{*})$ for $m \in [M]$ .
6: Set $D_{sub-samples}$ as Top-K scoring samples in $\{x_{m}^{*}, y_{m}^{*}\}_{m=1}^{M}$ . ▷ Filtering
7: Set $D_{samples} \leftarrow D_{samples} \cup D_{sub-samples}$ ▷ Diversity Aggregation
8: end for
9: Output: $D_{samples}$ .

# 3.2 Bootstrapping

Next, we introduce our bootstrapping strategy to augment a training dataset with high-scoring samples that are collected from the score-conditioned generator and labeled using a proxy model. Our key idea is to enlarge the dataset so that the score-conditioned generation is consistent with predictions of the proxy model, in particular for the high-scoring samples. This enables self-training by utilizing the extrapolation capabilities of the generator and allows the proxy model to transfer its knowledge to the score-conditioned generation process.

We first generate a set of samples $x_{1}^{*},\ldots,x_{L}^{*}$ from the generator $p_{\theta}(\boldsymbol{x}|y^{\dagger})$ conditioned on the desired score $y^{\dagger}$ Then we compute the corresponding labels $y_{1}^{*},\ldots,y_{L}^{*}$ using the proxy model, i.e., we set $y_{\ell}=f_{\phi}(\boldsymbol{x}_{\ell})$ for $\ell=1,\ldots,L$ . Finally, we augment the training dataset using the set of top-K samples $D_{aug}$ with respect to the proxy model, i.e., we set $D_{tr}\cup D_{aug}$ as the new training dataset $D_{tr}$ .

# 3.3 Aggregation Strategy for Sample Generation

Here, we introduce additional post-hoc aggregation strategies that can be used to further boost the quality of samples from our generator. See Algorithm 2 for a detailed process.

Filtering We follow Kumar and Levine [32] to exploit the knowledge of the proxy function for filtering high-scoring samples from the generator. To be specific, when evaluating our model, we sample a set of candidate solutions and select the top samples with respect to the proxy function.

Diverse aggregation To enhance the diversity of candidate samples while maintaining reliable generating performances with low variance, we gather cross-aggregated samples from multiple score-conditioned generators. These generators are independently trained using our proposed bootstrapped training approach. Since each bootstrapped training process introduces high randomness due to varying training datasets, combining the generative spaces of multiple generators yields a more diverse space compared to a single generator.

Moreover, this process helps reduce the variance in generating quality. By creating ensemble candidate samples from multiple generators, we ensure stability and mitigate the risk of potential failure cases caused by adversarial samples. These samples may receive high scores from the proxy function but have low actual scores. This approach resembles the classical ensemble strategy known as “bagging,” which aggregates noisy bootstrapped samples from decision trees to reduce variances.

# 4 Experiments

We present experimental results on six representative biological sequence design tasks to verify the effectiveness of the proposed method. We also conduct ablation studies to verify the effectiveness of each component in our method. For training, we use a single GPU of NVIDIA A100, where the training time of one generator is approximately 10 minutes.

Table 4.1: Experimental results on 100th percentile scores. The mean and standard deviation are reported for 8 independent solution generations. $\mathcal{D}(\text{best})$ indicate the maximum score of the offline dataset. The best-scored value is marked in bold. 

<table><tr><td>Method</td><td>RNA-A</td><td>RNA-B</td><td>RNA-C</td><td>TFBind8</td><td>GFP</td><td>UTR</td><td>Avg.</td></tr><tr><td> $\mathcal{D}$  (best)</td><td>0.120</td><td>0.122</td><td>0.125</td><td>0.439</td><td>0.789</td><td>0.593</td><td>0.365</td></tr><tr><td>REINFORCE [45]</td><td> $0.462 \pm 0.080$ </td><td> $0.437 \pm 0.033$ </td><td> $0.463 \pm 0.043$ </td><td> $0.936 \pm 0.041$ </td><td> $\textbf{0.865} \pm 0.003$ </td><td> $0.685 \pm 0.012$ </td><td>0.643</td></tr><tr><td>CMA-ES [24]</td><td> $0.841 \pm 0.058$ </td><td> $0.822 \pm 0.046$ </td><td> $0.803 \pm 0.039$ </td><td> $0.904 \pm 0.040$ </td><td> $0.055 \pm 0.003$ </td><td> $0.737 \pm 0.013$ </td><td>0.694</td></tr><tr><td>BO-qEI [50]</td><td> $0.724 \pm 0.055$ </td><td> $0.729 \pm 0.038$ </td><td> $0.707 \pm 0.034$ </td><td> $0.798 \pm 0.083$ </td><td> $0.254 \pm 0.352$ </td><td> $0.684 \pm 0.000$ </td><td>0.649</td></tr><tr><td>CbAS [9]</td><td> $0.541 \pm 0.042$ </td><td> $0.647 \pm 0.057$ </td><td> $0.644 \pm 0.071$ </td><td> $0.913 \pm 0.025$ </td><td> $\textbf{0.865} \pm 0.004$ </td><td> $0.692 \pm 0.008$ </td><td>0.717</td></tr><tr><td>Auto. CbAS [20]</td><td> $0.524 \pm 0.055$ </td><td> $0.562 \pm 0.031$ </td><td> $0.495 \pm 0.048$ </td><td> $0.890 \pm 0.050$ </td><td> $\textbf{0.865} \pm 0.003$ </td><td> $0.693 \pm 0.009$ </td><td>0.672</td></tr><tr><td>MIN [32]</td><td> $0.376 \pm 0.039$ </td><td> $0.374 \pm 0.041$ </td><td> $0.404 \pm 0.047$ </td><td> $0.892 \pm 0.060$ </td><td> $\textbf{0.865} \pm 0.001$ </td><td> $0.691 \pm 0.011$ </td><td>0.600</td></tr><tr><td>Grad [45]</td><td> $0.821 \pm 0.048$ </td><td> $0.720 \pm 0.047$ </td><td> $0.688 \pm 0.035$ </td><td> $0.965 \pm 0.030$ </td><td> $0.862 \pm 0.003$ </td><td> $0.682 \pm 0.013$ </td><td>0.792</td></tr><tr><td>COMs [44]</td><td> $0.403 \pm 0.062$ </td><td> $0.393 \pm 0.076$ </td><td> $0.494 \pm 0.098$ </td><td> $0.945 \pm 0.033$ </td><td> $0.861 \pm 0.009$ </td><td> $0.699 \pm 0.011$ </td><td>0.633</td></tr><tr><td>AdaLead [42]</td><td> $0.691 \pm 0.059$ </td><td> $0.630 \pm 0.062$ </td><td> $0.605 \pm 0.055$ </td><td> $0.962 \pm 0.024$ </td><td> $0.841 \pm 0.014$ </td><td> $0.631 \pm 0.010$ </td><td>0.727</td></tr><tr><td>GFN-AL [29]</td><td> $0.630 \pm 0.054$ </td><td> $0.677 \pm 0.079$ </td><td> $0.623 \pm 0.045$ </td><td> $0.956 \pm 0.018$ </td><td> $0.059 \pm 0.006$ </td><td> $0.695 \pm 0.021$ </td><td>0.607</td></tr><tr><td>BDI [12]</td><td> $0.700 \pm 0.000$ </td><td> $0.560 \pm 0.000$ </td><td> $0.632 \pm 0.000$ </td><td> $0.973 \pm 0.000$ </td><td> $0.864 \pm 0.000$ </td><td> $0.667 \pm 0.000$ </td><td>0.733</td></tr><tr><td>BOOTGEN</td><td> $\textbf{0.902} \pm 0.039$ </td><td> $\textbf{0.931} \pm 0.055$ </td><td> $\textbf{0.831} \pm 0.044$ </td><td> $\textbf{0.979} \pm 0.001$ </td><td> $\textbf{0.865} \pm 0.000$ </td><td> $\textbf{0.865} \pm 0.000$ </td><td> $\textbf{0.895}$ </td></tr></table>

Table 4.2: Experimental results on 50th percentile scores. The mean and standard deviation are reported for 8 independent solution generations. $\mathcal{D}(\text{best})$ indicate the maximum score of the offline dataset. The best-scored value is marked in bold.

<table><tr><td>Method</td><td>RNA-A</td><td>RNA-B</td><td>RNA-C</td><td>TFBind8</td><td>GFP</td><td>UTR</td><td>Avg.</td></tr><tr><td> $\mathcal{D}$  (best)</td><td>0.120</td><td>0.122</td><td>0.125</td><td>0.439</td><td>0.789</td><td>0.593</td><td>0.365</td></tr><tr><td>REINFORCE [45]</td><td>0.159 ± 0.011</td><td>0.162 ± 0.007</td><td>0.177 ± 0.011</td><td>0.450 ± 0.017</td><td>0.845 ± 0.003</td><td>0.575 ± 0.018</td><td>0.395</td></tr><tr><td>CMA-ES [24]</td><td>0.558 ± 0.012</td><td>0.531 ± 0.010</td><td>0.535 ± 0.012</td><td>0.526 ± 0.017</td><td>0.047 ± 0.000</td><td>0.497 ± 0.009</td><td>0.449</td></tr><tr><td>BO-qEI [50]</td><td>0.389 ± 0.009</td><td>0.397 ± 0.015</td><td>0.391 ± 0.012</td><td>0.439 ± 0.000</td><td>0.246 ± 0.341</td><td>0.571 ± 0.000</td><td>0.406</td></tr><tr><td>CbAS [9]</td><td>0.246 ± 0.008</td><td>0.267 ± 0.021</td><td>0.281 ± 0.015</td><td>0.467 ± 0.008</td><td>0.852 ± 0.004</td><td>0.566 ± 0.018</td><td>0.447</td></tr><tr><td>Auto. CbAS [20]</td><td>0.241 ± 0.022</td><td>0.237 ± 0.009</td><td>0.193 ± 0.007</td><td>0.413 ± 0.012</td><td>0.847 ± 0.003</td><td>0.563 ± 0.019</td><td>0.420</td></tr><tr><td>MIN [32]</td><td>0.146 ± 0.009</td><td>0.143 ± 0.007</td><td>0.174 ± 0.007</td><td>0.417 ± 0.012</td><td>0.830 ± 0.011</td><td>0.586 ± 0.000</td><td>0.383</td></tr><tr><td>Grad [45]</td><td>0.473 ± 0.025</td><td>0.462 ± 0.016</td><td>0.393 ± 0.017</td><td>0.513 ± 0.007</td><td>0.763 ± 0.181</td><td>0.611 ± 0.000</td><td>0.531</td></tr><tr><td>COMs [44]</td><td>0.172 ± 0.026</td><td>0.184 ± 0.039</td><td>0.228 ± 0.061</td><td>0.512 ± 0.051</td><td>0.737 ± 0.262</td><td>0.608 ± 0.000</td><td>0.407</td></tr><tr><td>AdaLead [42]</td><td>0.407 ± 0.018</td><td>0.353 ± 0.029</td><td>0.326 ± 0.019</td><td>0.485 ± 0.013</td><td>0.186 ± 0.216</td><td>0.592 ± 0.002</td><td>0.392</td></tr><tr><td>GFN-AL [29]</td><td>0.312 ± 0.013</td><td>0.300 ± 0.012</td><td>0.324 ± 0.009</td><td>0.538 ± 0.045</td><td>0.051 ± 0.003</td><td>0.597 ± 0.021</td><td>0.354</td></tr><tr><td>BDI [12]</td><td>0.411 ± 0.000</td><td>0.308 ± 0.000</td><td>0.345 ± 0.000</td><td>0.595 ± 0.000</td><td>0.837 ± 0.010</td><td>0.527 ± 0.000</td><td>0.504</td></tr><tr><td>BOOTGEN</td><td>0.707 ± 0.005</td><td>0.717 ± 0.006</td><td>0.596 ± 0.006</td><td>0.833 ± 0.007</td><td>0.853 ± 0.017</td><td>0.701 ± 0.004</td><td>0.731</td></tr></table>

# 4.1 Experimental Setting

Tasks. We evaluate an offline design algorithm by (1) training it on an offline dataset and (2) using it to generate 128 samples for high scores. We measure the 50th percentile and 100th percentile scores of the generated samples. All the results are measured using eight independent random seeds.

We consider six biological sequence design tasks: green fluorescent protein (GFP), DNA optimization for expression level on untranslated region (UTR), DNA optimization tasks for transcription factor binding (TFBind8), and three RNA optimization tasks for transcription factor binding (RNA-Binding-A, RNA-Binding-B, and RNA-Binding-C). The scores of the biological sequences range in [0, 1]. We report the statistics of the offline datasets used for each task in Table A.1. We also provide a detailed description of the tasks in Appendix A.1.

Baselines We compare our BOOTGEN with the following baselines: gradient ascent with respect to a proxy score model $[45, Grad.], REINFORCE$ $[49]$ , Bayesian optimization quasi-expected-improvement $[50, BO-qEI]$ , covariance matrix adaptation evolution strategy $[24, CMA-ES]$ , conditioning by adaptive sampling $[9, CbAS]$ , autofocused CbAS $[20, Auto. CbAS]$ , model inversion network $[32, MIN]$ , where these are in the official design bench $[45]$ . We compare with additional baselines of conservative objective models $[44, COMs]$ , generative flow network for active learning $[29, GFN-AL]$ and bidirectional learning $[12, BDI]$ .

Implementation We parameterize the conditional distribution $p_{\theta}(x_{t}|\boldsymbol{x}_{1:t-1},y)$ using a 2-layer long short-term memory [26, LSTM] network with 512 hidden dimensions. The condition y is injected into the LSTM using a linear projection layer. We parameterize the proxy model using a multi-layer perceptron (MLP) with 2048 hidden dimensions and a sigmoid activation function. Our parameterization is consistent across all the tasks. We provide a detailed description of the hyperparameters in Appendix A. We also note the importance of the desired score $y^{\dagger}$ to condition during bootstrapping and evaluation. In this regard, we set it as the maximum score that is achievable for the given problem, i.e., we set $y^{\dagger}=1$ . We assume that such a value is known following [12].

![](images/733d9d7a13e97bde27d38e74f4b1aff7c903c6c362be808d292b6eca7084cd93.jpg)

<details>
<summary>line</summary>

| Number of evaluations (K) | BootGen | Grad. | BDI | CbAS | CMA-ES |
| ------------------------- | ------- | ----- | --- | ---- | ------ |
| 0                         | 0.6     | 0.4   | 0.6 | 0.3  | 0.6    |
| 20                        | 0.8     | 0.6   | 0.7 | 0.4  | 0.7    |
| 40                        | 0.85    | 0.7   | 0.75| 0.5  | 0.8    |
| 60                        | 0.85    | 0.75  | 0.75| 0.55 | 0.8    |
| 80                        | 0.85    | 0.75  | 0.75| 0.55 | 0.8    |
| 100                       | 0.85    | 0.75  | 0.75| 0.55 | 0.8    |
| 120                       | 0.85    | 0.75  | 0.75| 0.55 | 0.8    |
</details>

(a) RNA-Binding-A

![](images/8a1c5d6a3331312fc1cf56301161eeb8eac08f3e89dbe14b1539042c70b146c0.jpg)

<details>
<summary>line</summary>

| Number of evaluations (K) | BootGen | Grad. | BDI  | CbAS | CMA-ES |
| ------------------------- | ------- | ----- | ---- | ---- | ------ |
| 0                         | 0.3     | 0.4   | 0.5  | 0.3  | 0.5    |
| 20                        | 0.8     | 0.7   | 0.6  | 0.4  | 0.7    |
| 40                        | 0.85    | 0.8   | 0.65 | 0.5  | 0.8    |
| 60                        | 0.85    | 0.8   | 0.65 | 0.5  | 0.8    |
| 80                        | 0.85    | 0.8   | 0.65 | 0.5  | 0.8    |
| 100                       | 0.85    | 0.8   | 0.7  | 0.5  | 0.8    |
| 120                       | 0.85    | 0.8   | 0.7  | 0.5  | 0.8    |
| 140                       | 0.85    | 0.8   | 0.7  | 0.5  | 0.8    |
| 160                       | 0.85    | 0.8   | 0.7  | 0.5  | 0.8    |
| 180                       | 0.85    | 0.8   | 0.7  | 0.5  | 0.8    |
| 200                       | 0.85    | 0.8   | 0.7  | 0.5  | 0.8    |
| 220                       | 0.85    | 0.8   | 0.7  | 0.5  | 0.8    |
| 240                       | 0.85    | 0.8   | 0.7  | 0.5  | 0.8    |
| 260                       | 0.85    | 0.8   | 0.7  | 0.5  | 0.8    |
| 280                       | 0.85    | 0.8   | 0.7  | 0.5  | 0.8    |
| 300                       | 0.85    | 0.8   | 0.7  | 0.5  | 0.8    |
| 320                       | 0.85    | 0.8   | 0.7  | 0.5  | 0.8    |
| 340                       | 0.85    | 0.8   | 0.7  | 0.5  | 0.8    |
| 360                       | 0.85    | 0.8   | 0.7  | 0.5  | 0.8    |
| 380                       | 0.85    | 0.8   | 0.7  | 0.5  | 0.8    |
| 400                       | 0.85    | 0.8   | 0.7  | 0.5  | 0.8    |
| 420                       | 0.85    | 0.8   | 0.7  | 0.5  | 0.8    |
| 440                       | 0.85    | 0.8   | 0.7  | 0.5  | 0.8    |
| 460                       | 0.85    | 0.8   | 0.7  | 0.5  | 0.8    |
| 480                       | 0.85    | 0.8   | 0.7  | 0.5  | 0.8    |
| 500                       | 0.85    | 0.8   | 0.7  | 0.5  | 0.8    |
| Note: The Storm values are not explicitly provided in the code, so they are calculated based on the original data source used to generate the results from the code execution of the code source in the code source library.
</details>

(b) RNA-Binding-B

![](images/3cf55dd6320487105f3f0c2368ed3726f2977b4c7b7b1b400be0af1ff7182700.jpg)

<details>
<summary>line</summary>

| Number of evaluations (K) | BootGen | Grad. | BDI | CbAS | CMA-ES |
| ------------------------- | ------- | ----- | --- | ---- | ------ |
| 0                         | 0.6     | 0.4   | 0.6 | 0.3  | 0.6    |
| 20                        | 0.8     | 0.7   | 0.7 | 0.4  | 0.8    |
| 40                        | 0.85    | 0.8   | 0.7 | 0.5  | 0.85   |
| 60                        | 0.85    | 0.8   | 0.7 | 0.5  | 0.85   |
| 80                        | 0.85    | 0.8   | 0.7 | 0.5  | 0.85   |
| 100                       | 0.85    | 0.8   | 0.7 | 0.5  | 0.85   |
| 120                       | 0.85    | 0.8   | 0.7 | 0.5  | 0.85   |
</details>

(c) RNA-Binding-C

![](images/8109619efa242b082271296c09fce566fa601efdcff76f5b167924f806a64cd6.jpg)

<details>
<summary>line</summary>

| Number of evaluations (K) | BootGen | Grad. | BDI | ChAS | CMA-ES |
| ------------------------- | ------- | ----- | --- | ---- | ------ |
| 0                         | 0.875   | 0.875 | 0.875 | 0.875 | 0.875  |
| 20                        | 0.975   | 0.925 | 0.925 | 0.825 | 0.825  |
| 40                        | 0.975   | 0.950 | 0.925 | 0.850 | 0.850  |
| 60                        | 0.975   | 0.950 | 0.925 | 0.875 | 0.875  |
| 80                        | 0.975   | 0.950 | 0.925 | 0.900 | 0.900  |
| 100                       | 0.975   | 0.950 | 0.925 | 0.925 | 0.925  |
| 120                       | 0.975   | 0.950 | 0.925 | 0.925 | 0.925  |
</details>

(d) TFBind8

![](images/e8b9bc0552cfd94638a43704f227b9b0d73c17479845f9127d39d65b47398e8f.jpg)

<details>
<summary>line</summary>

| Number of evaluations (K) | BootGen | Grad. | BDI   | ChAS  | CMA-ES |
| ------------------------- | ------- | ----- | ----- | ----- | ------ |
| 0                         | 0.856   | 0.856 | 0.856 | 0.856 | 0.856  |
| 20                        | 0.864   | 0.860 | 0.858 | 0.864 | 0.864  |
| 40                        | 0.864   | 0.860 | 0.860 | 0.864 | 0.864  |
| 60                        | 0.864   | 0.860 | 0.862 | 0.864 | 0.864  |
| 80                        | 0.864   | 0.860 | 0.862 | 0.864 | 0.864  |
| 100                       | 0.864   | 0.860 | 0.862 | 0.864 | 0.864  |
| 120                       | 0.864   | 0.860 | 0.862 | 0.864 | 0.864  |
</details>

(e) GFP

![](images/7c7b3d1fd61d7c3b29ab0afc6735bb4463de88a94b25c91dde8c5ea68452ee35.jpg)

<details>
<summary>line</summary>

| Number of evaluations (K) | BootGen | Grad. | BDI | CbAS | CMA-ES |
| ------------------------- | ------- | ----- | --- | ---- | ------ |
| 0                         | 0.50    | 0.50  | 0.50 | 0.50 | 0.50   |
| 20                        | 0.75    | 0.65  | 0.65 | 0.65 | 0.70   |
| 40                        | 0.85    | 0.68  | 0.68 | 0.68 | 0.73   |
| 60                        | 0.85    | 0.69  | 0.69 | 0.69 | 0.74   |
| 80                        | 0.85    | 0.69  | 0.69 | 0.69 | 0.74   |
| 100                       | 0.85    | 0.69  | 0.69 | 0.69 | 0.74   |
| 120                       | 0.85    | 0.69  | 0.69 | 0.69 | 0.74   |
</details>

(f) UTR   
Figure 4.1: Evaluation-performance graph to compare with representative offline biological design baselines. The number of evaluations $K \in [1, 128]$ stands for the number of candidate designs to be evaluated by the Oracle score function. The average value and standard deviation error bar for 8 independent runs are reported. Our method outperforms other baselines at every task for almost all K.

# 4.2 Performance Evaluation

In Table 4.1 and Table 4.2, we report the performance of our BOOTGEN along with other baselines. One can observe how our BOOTGEN consistently outperforms the considered baselines across all six tasks. In particular, one can observe how our BOOTGEN achieves large gains in 50th percentile metrics. This highlights how our algorithm is able to create a reliable set of candidates.

For TFbind8, which has a relatively small search space $(4^{8})$ , having high performances on the 100th percentile is relatively easy. Indeed, the classical method of CMA-ES and Grad. gave pretty good performances. However, for the 50th percentile score, a metric for measuring the method's reliability, BDI outperformed previous baselines by a large margin. Our method outperformed even BDI and achieved an overwhelming score.

For higher dimensional tasks of UTR, even the 50th percentile score of BOOTGEN outperforms the 100th percentile score of other baselines by a large margin. We note that our bootstrapping strategies and aggregation strategy greatly contributed to improving performances on UTR. For additional tasks of RNA, we achieved the best score for both the 50th percentile and the 100th percentile. This result verifies that our method is task-expandable.

# 4.3 Varying the Evaluation Budget

In real-world scenarios, there may be situations where only a few samples can be evaluated due to the expensive score function. For example, in an extreme scenario, for the clinical trial of a new protein drug, there may be only one or two chances to be evaluated. As we measure the 50th percentile and 100th percentile score among 128 samples following the design-bench [45] at Tables 4.1 and 4.2, we also provide a 100th percentile score report at the fewer samples from 1 sample to the 128 samples to evaluate the model's robustness on the low-budget evaluation scenarios.

To account for this, we also provide a budget-performance graph that compares the performance of our model to the baselines using different numbers of evaluations. This allows us to observe the trade-off between performance and the number of samples generated. Note that we select baselines as the Top 5 methods in terms of average percentile 100 scores reported at Table 4.1.

Table 4.3: Experimental results on 100th percentile scores (100th Per.), 50th percentile scores (50th Per.), average score (Avg. Score), diversity, and novelty, among 128 samples of UTR task. The mean and standard deviation of 8 independent runs for producing 128 samples is reported. The best-scored value is marked in bold. The lowest standard deviation is marked as the underline. The DA stands for the diverse aggregation strategy. 

<table><tr><td>Methods</td><td>100th Per.</td><td>50th Per.</td><td>Avg. Score</td><td>Diversity</td><td>Novelty</td></tr><tr><td>MIN [32]</td><td> $0.691 \pm 0.011$ </td><td> $0.587 \pm 0.012$ </td><td> $0.554 \pm 0.010$ </td><td> $28.53 \pm 0.095$ </td><td> $18.32 \pm 0.091$ </td></tr><tr><td>CMA-ES [24]</td><td> $0.746 \pm 0.018$ </td><td> $0.498 \pm 0.012$ </td><td> $0.520 \pm 0.013$ </td><td> $24.69 \pm 0.150$ </td><td> $19.95 \pm 0.925$ </td></tr><tr><td>Grad. [45]</td><td> $0.682 \pm 0.013$ </td><td> $0.513 \pm 0.007$ </td><td> $0.521 \pm 0.006$ </td><td> $25.63 \pm 0.615$ </td><td> $16.89 \pm 0.426$ </td></tr><tr><td>GFN-AL [29]</td><td> $0.700 \pm 0.015$ </td><td> $0.602 \pm 0.014$ </td><td> $0.580 \pm 0.014$ </td><td> $30.89 \pm 1.220$ </td><td> $20.25 \pm 2.272$ </td></tr><tr><td>BOOTGEN w/o DA.</td><td> $0.729 \pm 0.074$ </td><td> $0.672 \pm 0.082$ </td><td> $0.652 \pm 0.081$ </td><td> $17.83 \pm 5.378$ </td><td> $20.49 \pm 1.904$ </td></tr><tr><td>BOOTGEN w/ DA. (ours)</td><td> $\mathbf{0.858} \pm \underline{\mathbf{0.003}}$ </td><td> $\mathbf{0.701} \pm \underline{\mathbf{0.004}}$ </td><td> $\mathbf{0.698} \pm \underline{\mathbf{0.001}}$ </td><td> $\mathbf{31.57} \pm \underline{\mathbf{0.073}}$ </td><td> $\mathbf{21.40} \pm \underline{\mathbf{0.057}}$ </td></tr></table>

![](images/8d3887336025cd5c00b938c7d585d7c8f7e6b9cf380cf07eb9867bd835bbbde2.jpg)

<details>
<summary>scatter</summary>

| Model       | Diversity | Avg. Score |
|-------------|-----------|------------|
| BootGen     | 31.5      | 0.70       |
| GFN-AL      | 30.5      | 0.59       |
| Grad.       | 29.5      | 0.58       |
| CMA-ES      | 28.5      | 0.57       |
| CbAS        | 29.0      | 0.56       |
| Auto. CbAS  | 28.0      | 0.55       |
| MIN         | 28.5      | 0.54       |
| REINFORCE   | 27.5      | 0.53       |
| BDI         | 26.0      | 0.48       |
</details>

![](images/814945b770c217c6d8c7bbb6ce02f174f251f4136333c7bbbd377c2fc870f3e9.jpg)

<details>
<summary>scatter</summary>

| Model       | Novelty | Avg. Score |
|-------------|---------|------------|
| BootGen     | 21.5    | 0.70       |
| GFN-AL      | 22.0    | 0.59       |
| Grad.       | 18.0    | 0.53       |
| CMA-ES      | 20.0    | 0.54       |
| CbAS        | 18.5    | 0.55       |
| Auto. CbAS  | 18.5    | 0.56       |
| MIN         | 18.5    | 0.57       |
| REINFORCE   | 18.5    | 0.51       |
| BDI         | 18.5    | 0.48       |
</details>

Figure 4.2: Multi-objectivity comparison of diversity and novelty on the average score for the UTR task. Each datapoint for 8 independent runs is depicted.

As shown in Fig. 4.1, our method outperforms every baseline for almost every evaluation budget. For the UTR task, our performance on a single evaluation budget gives a better score than the other baselines' scores when they have a budget of 128 evaluations. For RNA tasks, our method with an approximate budget of 30 achieves superior performance compared to other methods with a budget of 128. These results show that our method is the most reliable as its performance is most robust when the evaluation budget is limited.

# 4.4 Average Score with Diversity

For biological sequence design, measuring the diversity and novelty of the generated sequence is also crucial $[29]$ . Following the evaluation metric of $[29]$ we compare the performance of models in terms of diversity and novelty.

Here is measurement of diversity for sampled design dataset $D = \{x_{1}, \ldots, x_{M}\}$ from generator which is average of Levenshtein distance [23], denoted by $d(\boldsymbol{x}_{i}, \boldsymbol{x}_{j})$ , between arbitrary two biological sequences $x_{i}, x_{j}$ from the generated design candidates D:

$$
\text { Diversity } (\mathcal {D}) := \frac {1}{| \mathcal {D} | (| \mathcal {D} | - 1)} \sum_ {\boldsymbol {x} \in \mathcal {D}} \sum_ {\boldsymbol {s} \in \mathcal {D} \setminus \{\boldsymbol {x} \}} d (\boldsymbol {x}, \boldsymbol {s}).
$$

Next, we measure the minimum distance from the offline dataset $D_{offline}$ which measures the novelty of generated design candidates D as:

$$
\text { Novelty } (\mathcal {D}, \mathcal {D} _ {\text { offline }}) = \frac {1}{| \mathcal {D} |} \sum_ {\boldsymbol {x} \in \mathcal {D}} \min _ {\boldsymbol {s} \in \mathcal {D} _ {\text { offline }}} d (\boldsymbol {x}, \boldsymbol {s}).
$$

Our method surpasses all baselines, including GFN-AL $[29]$ , in the UTR task, as evidenced by the Pareto frontier depicted in Table 4.3 and Fig. 4.2. Given the highly dimensional nature of the UTR task and its expansive search space, the discovery of novel and diverse candidates appears to be directly related to their average score. This implies that extensive exploration of the high-dimensional space is crucial for improving scores in the UTR task.

Table 4.4: Ablation study for BOOTGEN. The average score among 128 samples is reported. We make 8 independent runs to produce 128 samples where the mean and the standard deviation are reported. For every method, an aggregation strategy is applied by default. The best-scored value is marked in bold. The lowest standard deviation is underlined. The RR stands for rank-based reweighting, the B stands for bootstrapping, and the F stands for filtering. 

<table><tr><td>Components</td><td>RNA-A</td><td>RNA-B</td><td>RNA-C</td><td>TFbind8</td><td>UTR</td><td>GFP</td></tr><tr><td> $\emptyset$ </td><td>0.388 ± 0.007</td><td>0.350 ± 0.008</td><td>0.394 ± 0.010</td><td>0.579 ± 0.010</td><td>0.549 ± 0.009</td><td>0.457 ± 0.044</td></tr><tr><td>{RR}</td><td>0.483 ± 0.006</td><td>0.468 ± 0.008</td><td>0.441 ± 0.010</td><td>0.662 ± 0.009</td><td>0.586 ± 0.008</td><td>0.281 ± 0.031</td></tr><tr><td>{RR, B}</td><td>0.408 ± 0.009</td><td>0.379 ± 0.009</td><td>0.417 ± 0.006</td><td>0.666 ± 0.009</td><td>0.689 ± 0.003</td><td>0.470 ± 0.034</td></tr><tr><td>{RR, F}</td><td>0.576 ± 0.005</td><td>0.586 ± 0.004</td><td>0.536 ± 0.007</td><td>0.833 ± 0.004</td><td>0.621 ± 0.003</td><td>0.783 ± 0.011</td></tr><tr><td>{RR, F, B}</td><td>0.607 ± 0.009</td><td>0.612 ± 0.005</td><td>0.554 ± 0.007</td><td>0.840 ± 0.004</td><td>0.698 ± 0.001</td><td>0.804 ± 0.002</td></tr></table>

It is worth noting that GFN-AL, which is specifically designed to generate diverse, high-quality samples through an explorative policy, secures a second place for diversity. Although GFN-AL occasionally exhibits better diversity than our method and achieves a second-place average score, it consistently delivers poor average scores in the GFP and RNA tasks Table 4.2. This drawback can be attributed to its high explorative policy, which necessitates focused exploration in narrow regions. In contrast, BOOTGEN consistently produces reliable scores across all tasks Table 4.2. For a comprehensive comparison with GFN-AL, please refer to the additional experiments presented in Appendix C.

Diverse aggregation strategy Our diverse aggregation (DA) strategy significantly enhances diversity, novelty, and score variance, as demonstrated in Table 4.3. This is especially beneficial for the UTR task, which necessitates extensive exploration of a vast solution space, posing a substantial risk to the bootstrapped training process. In this context, certain bootstrapped generators may yield exceedingly high scores, while others may produce low scores due to random exploration scenarios. By employing DA, we combine multiple generators to generate candidate samples, thereby greatly stabilizing the quality of the bootstrapped generator.

# 4.5 Ablation study

The effectiveness of our components, namely rank-based reweighting (RR), bootstrapping (B), and filtering (F), in improving performance is evident in Table 4.4. Across all tasks, these components consistently contribute to performance enhancements. The bootstrapping process is particularly more beneficial for high-dimensional tasks like UTR and GFP. This correlation is intuitive since high-dimensional tasks require a larger amount of data for effective exploration. The bootstrapped training dataset augmentation facilitates this search process by leveraging proxy knowledge. Additionally, the filtering technique proves to be powerful in improving scores. As we observed from the diverse aggregation and filtering, the ensemble strategy greatly enhances score-conditioned generators.

# 5 Future Works

Enhancing proxy robustness While our bootstrapping method shows promise for offline bio-sequential design tasks, it has inherent technical limitations. The assumption that the generator produces superior data to the training dataset may backfire if the generator samples have poor quality designs and the proxy used is inaccurate. While the current aggregation strategy effectively manages this risk, we can address this limitation by utilizing robust learning methods of proxies such as conservative proxies modeling $[44]$ , robust model adaptation techniques $[52]$ , parallel mentoring proxies $[13]$ , and importance-aware co-teaching of proxies $[53]$ for further improvement.

Enhancing architecture of BOOTGEN Our approach primarily employs a straightforward architectural framework, with a primary emphasis on validating its algorithmic structures in the context of offline biological sequence design. To enhance the practical utility of our method, it will be advantageous to incorporate established and robust architectural paradigms, exemplified in works such as $[11]$ and $[14]$ , into the framework of our method. One promising avenue for achieving this integration is the incorporation of pre-trained protein language models (pLMs) $[34, 15]$ , akin to those expounded upon in $[14]$ .

# 6 Conclusion

This study introduces a novel approach to stabilize and enhance score-conditioned generators for offline biological sequence design, incorporating the classical concepts of bootstrapping and aggregation. Our novel method, named BOOTGEN, consistently outperformed all baselines across six offline biological sequence design tasks, encompassing RNA, DNA, and protein optimization. Our strategy of bootstrapping and aggregation yielded remarkable improvements in achieving high scores, generating diverse samples, and minimizing performance variance.

# Acknowledgements

We thank all the valuable comments and suggestions from anonymous reviewers who helped us improve and refine our paper. This work was supported by the Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korean government(MSIT)(2022-0-01032, Development of Collective Collaboration Intelligence Framework for Internet of Autonomous Things).

# References

[1] M.-R. Amini and P. Gallinari. Semi-supervised logistic regression. In ECAI, volume 2, page 11, 2002.   
[2] C. Angermueller, D. Dohan, D. Belanger, R. Deshpande, K. Murphy, and L. Colwell. Model-based reinforcement learning for biological sequence design. In International conference on learning representations, 2019.   
[3] F. H. Arnold. Design by directed evolution. Accounts of chemical research, 31(3):125–131, 1998.   
[4] F. H. Arnold. Directed evolution: bringing new chemistry to life. Angewandte Chemie International Edition, 57(16):4143–4148, 2018.   
[5] L. A. Barrera, A. Vedenko, J. V. Kurland, J. M. Rogers, S. S. Gisselbrecht, E. J. Rossin, J. Woodard, L. Mariani, K. H. Kock, S. Inukai, et al. Survey of variation in human transcription factors reveals prevalent dna binding changes. Science, 351(6280):1450–1454, 2016.   
[6] D. Belanger, S. Vora, Z. Mariet, R. Deshpande, D. Dohan, C. Angermueller, K. Murphy, O. Chapelle, and L. Colwell. Biological sequences design using batched bayesian optimization. 2019.   
[7] E. Bengio, M. Jain, M. Korablyov, D. Precup, and Y. Bengio. Flow network based generative models for non-iterative diverse candidate generation. Advances in Neural Information Processing Systems, 34:27381–27394, 2021.   
[8] J. D. Bloom and F. H. Arnold. In the light of directed evolution: pathways of adaptive protein evolution. Proceedings of the National Academy of Sciences, 106(supplement\_1):9995–10000, 2009.   
[9] D. Brookes, H. Park, and J. Listgarten. Conditioning by adaptive sampling for robust design. In International conference on machine learning, pages 773–782. PMLR, 2019.   
[10] D. H. Brookes and J. Listgarten. Design by adaptive sampling. arXiv preprint arXiv:1810.03714, 2018.   
[11] A. Chan, A. Madani, B. Krause, and N. Naik. Deep extrapolation for attribute-enhanced generation. Advances in Neural Information Processing Systems, 34:14084–14096, 2021.   
[12] C. Chen, Y. Zhang, J. Fu, X. Liu, and M. Coates. Bidirectional learning for offline infinite-width model-based optimization. In Advances in Neural Information Processing Systems, 2022.

[13] C. Chen, C. Beckham, Z. Liu, X. Liu, and C. Pal. Parallel-mentoring for offline model-based optimization. arXiv preprint arXiv:2309.11592, 2023.   
[14] C. Chen, Y. Zhang, X. Liu, and M. Coates. Bidirectional learning for offline model-based biological sequence design. arXiv preprint arXiv:2301.02931, 2023.   
[15] C. Chen, J. Zhou, F. Wang, X. Liu, and D. Dou. Structure-aware protein self-supervised learning. Bioinformatics, 2023.   
[16] L. Chen, K. Lu, A. Rajeswaran, K. Lee, A. Grover, M. Laskin, P. Abbeel, A. Srinivas, and I. Mordatch. Decision transformer: Reinforcement learning via sequence modeling. Advances in neural information processing systems, 34:15084–15097, 2021.   
[17] T. Chen, S. Kornblith, K. Swersky, M. Norouzi, and G. E. Hinton. Big self-supervised models are strong semi-supervised learners. Advances in neural information processing systems, 33:22243–22255, 2020.   
[18] P. A. Dalby. Strategy and success for the directed evolution of enzymes. Current opinion in structural biology, 21(4):473–480, 2011.   
[19] C. Ekbote, M. Jain, P. Das, and Y. Bengio. Consistent training via energy-based gflownets for modeling discrete joint distributions. arXiv preprint arXiv:2211.00568, 2022.   
[20] C. Fannjiang and J. Listgarten. Autofocused oracles for model-based design. Advances in Neural Information Processing Systems, 33:12945–12956, 2020.   
[21] J. Fu and S. Levine. Offline model-based optimization via normalized maximum likelihood estimation. arXiv preprint arXiv:2102.07970, 2021.   
[22] Y. Grandvalet and Y. Bengio. Semi-supervised learning by entropy minimization. Advances in neural information processing systems, 17, 2004.   
[23] R. Haldar and D. Mukhopadhyay. Levenshtein distance technique in dictionary lookup methods: An improved approach. arXiv preprint arXiv:1101.1232, 2011.   
[24] N. Hansen. The CMA evolution strategy: a comparing review. Towards a new evolutionary computation, pages 75-102, 2006.   
[25] T. Hesterberg. Bootstrap. Wiley Interdisciplinary Reviews: Computational Statistics, 3(6):497–526, 2011.   
[26] S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural computation, 9(8):1735-1780, 1997.   
[27] A. Hottung, B. Bhandari, and K. Tierney. Learning a latent search space for routing problems using variational autoencoders. In International Conference on Learning Representations, 2020.   
[28] I. Igashov, H. Stärk, C. Vignac, V. G. Satorras, P. Frossard, M. Welling, M. Bronstein, and B. Correia. Equivariant 3d-conditional diffusion models for molecular linker design. arXiv preprint arXiv:2210.05274, 2022.   
[29] M. Jain, E. Bengio, A. Hernandez-Garcia, J. Rector-Brooks, B. F. Dossou, C. A. Ekbote, J. Fu, T. Zhang, M. Kilgour, D. Zhang, et al. Biological sequence design with gflownets. In International Conference on Machine Learning, pages 9786–9801. PMLR, 2022.   
[30] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.   
[31] D. P. Kingma and M. Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.   
[32] A. Kumar and S. Levine. Model inversion networks for model-based optimization. Advances in Neural Information Processing Systems, 33:5126–5137, 2020.

[33] R. Lorenz, S. H. Bernhart, C. Höner zu Siederdissen, H. Tafer, C. Flamm, P. F. Stadler, and I. L. Hofacker. Viennarna package 2.0. Algorithms for molecular biology, 6(1):1–14, 2011.   
[34] A. Madani, B. McCann, N. Naik, N. S. Keskar, N. Anand, R. R. Eguchi, P.-S. Huang, and R. Socher. Progen: Language modeling for protein generation. arXiv preprint arXiv:2004.03497, 2020.   
[35] M. Mirza and S. Osindero. Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784, 2014.   
[36] H. Moss, D. Leslie, D. Beck, J. Gonzalez, and P. Rayson. Boss: Bayesian optimization over string spaces. Advances in neural information processing systems, 33:15476-15486, 2020.   
[37] K. Nigam and R. Ghani. Analyzing the effectiveness and applicability of co-training. In Proceedings of the ninth international conference on Information and knowledge management, pages 86–93, 2000.   
[38] E. O. Pyzer-Knapp. Bayesian optimization for accelerated drug discovery. IBM Journal of Research and Development, 62(6):2–1, 2018.   
[39] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.   
[40] Z. Ren, J. Li, F. Ding, Y. Zhou, J. Ma, and J. Peng. Proximal exploration for model-guided protein sequence design. In K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvari, G. Niu, and S. Sabato, editors, Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 18520–18536. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr.press/v162/ren22a.html.   
[41] P. J. Sample, B. Wang, D. W. Reid, V. Presnyak, I. J. McFadyen, D. R. Morris, and G. Seelig. Human 5' utr design and variant effect prediction from a massively parallel translation assay. Nature biotechnology, 37(7):803–809, 2019.   
[42] S. Sinai, R. Wang, A. Whatley, S. Slocum, E. Locane, and E. D. Kelsic. Adalead: A simple and robust adaptive greedy search algorithm for sequence design. arXiv preprint arXiv:2010.02141, 2020.   
[43] K. Terayama, M. Sumita, R. Tamura, and K. Tsuda. Black-box optimization for automated discovery. Accounts of Chemical Research, 54(6):1334–1346, 2021.   
[44] B. Trabucco, A. Kumar, X. Geng, and S. Levine. Conservative objective models for effective offline model-based optimization. In International Conference on Machine Learning, pages 10358–10368. PMLR, 2021.   
[45] B. Trabucco, X. Geng, A. Kumar, and S. Levine. Design-bench: Benchmarks for data-driven offline model-based optimization. arXiv preprint arXiv:2202.08450, 2022.   
[46] A. Tripp, E. Daxberger, and J. M. Hernández-Lobato. Sample-efficient optimization in the latent space of deep generative models via weighted retraining. Advances in Neural Information Processing Systems, 33:11259–11272, 2020.   
[47] H. Wang, A. Sakhadeo, A. M. White, J. M. Bell, V. Liu, X. Zhao, P. Liu, T. Kozuno, A. Fyshe, and M. White. No more pesky hyperparameters: Offline hyperparameter tuning for RL. Transactions on Machine Learning Research, 2022. URL https://openreview.net/forum?id=Ai0Ui3440V.   
[48] K. Wang, H. Zhao, X. Luo, K. Ren, W. Zhang, and D. Li. Bootstrapped transformer for offline reinforcement learning. arXiv preprint arXiv:2206.08569, 2022.   
[49] R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3):229-256, 1992.   
[50] J. T. Wilson, R. Moriconi, F. Hutter, and M. P. Deisenroth. The reparameterization trick for acquisition functions. arXiv preprint arXiv:1712.00424, 2017.

[51] Q. Xie, M.-T. Luong, E. Hovy, and Q. V. Le. Self-training with noisy student improves imagenet classification. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10687–10698, 2020.   
[52] S. Yu, S. Ahn, L. Song, and J. Shin. Roma: Robust model adaptation for offline model-based optimization. Advances in Neural Information Processing Systems, 34:4619–4631, 2021.   
[53] Y. Yuan, C. Chen, Z. Liu, W. Neiswanger, and X. Liu. Importance-aware co-teaching for offline model-based optimization. arXiv preprint arXiv:2309.11600, 2023.   
[54] M. Zimmer. Green fluorescent protein (GFP): applications, structure, and related photophysical behavior. Chemical reviews, 102(3):759–782, 2002.

# A Additional Experimental Settings

Table A.1: Details of the offline datasets. We let $|\mathcal{X}|$ and $|\mathcal{D}|$ denote the sizes of the search space and the offline dataset, respectively. 

<table><tr><td></td><td>Seq. Length</td><td>Vocab size</td><td> $|\mathcal{X}|$ </td><td> $|\mathcal{D}|$ </td></tr><tr><td>GFP</td><td>20</td><td>237</td><td> $20^{237}$ </td><td>5,000</td></tr><tr><td>UTR</td><td>50</td><td>4</td><td> $50^4$ </td><td>140,000</td></tr><tr><td>TFBind8</td><td>8</td><td>8</td><td> $4^8$ </td><td>32,898</td></tr><tr><td>RNA-Binding</td><td>14</td><td>4</td><td> $4^{14}$ </td><td>5,000</td></tr></table>

# A.1 Datasets

- GFP [41] is a task to optimize a protein sequence of length 237 consisting of one of 20 amino acids, i.e., the search space is $20^{237}$ . Its objective is to find a protein with high fluorescence. Following Trabucco et al. [45], we prepare the offline dataset using 5000 samples with 50 to 60 percentile scores in the original data.   
- UTR [41] is a task to optimize a DNA sequence of length 50 consisting of one of four nucleobases: adenine (A), guanine (G), cytosine (C), thymine (T). Its objective is to maximize the expression level of the corresponding 5'UTR region. For the construction of the offline dataset $\mathcal{D}$ , following Trabucco et al. [45], we provide samples with scores under the 50th percentile data of 140,000 examples.   
- TFBind8 [5] is a task that optimizes DNA similar to the UTR. The objective is to find a length 8 sequence to maximize the binding activity with human transcription factors. For the offline dataset $\mathcal{D}$ , we provide under 50th percentile data of 32,898 examples following Trabucco et al. [45].   
- RNA-Binding [33] is a task that optimizes RNA, a sequence that contains four vocab words of nucleobases: adenine (A), uracil (U), cytosine (C), and guanine (G). The objective is to find a length 14 sequence to maximize the binding activity with the target transcription factor. We present three target transcriptions of RNA termed RNA-Binding-A (for L14 RNA1), RNA-Binding-B (for L14 RNA2), and RNA-Binding-C (for L14 RNA3). We provide under 0.12 scored data for the offline dataset $\mathcal{D}$ among randomly generated 5,000 sequences using open-source code $^{2}$ .

# A.2 Implementation of Baselines

This section provides a detailed implementation of baselines of offline biological sequence design.

Baselines from Design Bench [45]. Most baselines are from the offline model-based optimization (MBO) benchmark called design-bench [45]. The design bench contains biological sequence tasks of the GFP, UTR, and TFbind8, where it contains baselines of REINFORCE, CMA-ES [24], BO-qEI [50], CbAS [9], Auto. Cbas [20], MIN [32], gradient ascent (Grad.), and COMS [44]. We reproduce them by following the official source code $^{3}$ . For the RNA tasks, we follow hyperparameters of TFBind8 as the number of vocab are same as 4, and the sequence length is similar where the TFBind8 has length 8 and RNA tasks have length 14 as our method follows the same.

BDI [12]. For BDI, we follow hyperparameter setting at the paper [12] and implementation at the opensource code $^{4}$ . For RNA tasks, we follow the hyperparameter for TFBind8 tasks, as our method follows the same.

GFN-AL [29]. For GFN-AL we follow hyperparameters setting at the paper [29] and implementation on open-source code $^{5}$ . Because they only reported the TFbind8 and the GFP tasks, we use the hyperparameter of the GFP for the UTR tasks hyperparameter of the TFBind8 for RNA tasks, as our method follows the same.

# A.3 Hyperparameters

Table A.2: Hyperparameters. I denotes the number of bootstrapping iterations after pretraining, $I'$ denotes the number of pertaining iterations, M represents the number of samples used in inference, K stands for the number of filtered samples among M candidates, and $N_{gen}$ refers to the number of aggregated generators in the experiments. 

<table><tr><td> $I$ </td><td> $I'$ </td><td> $M$ </td><td> $K$ </td><td> $N_{\text{gen}}$ </td></tr><tr><td>2,500</td><td>12,500</td><td>1,280</td><td>128</td><td>8</td></tr></table>

Training. We give consistency hyperparameters for all tasks except the learning rate. We set the generator's learning rate to $10^{-5}$ for short-length tasks (lengths 8 and 14) of TFBind and RNA tasks and $5 \times 10^{-5}$ for longer-length tasks (lengths 50 and 237) of UTR and GFP. We trained the generator with 12,500 steps before bootstrapping. Bootstrapping is applied with 2,500 in additional steps. The batch size of training is 256. We set the weighting parameter $k = 10^{-2}$ . Note that we early-stopped the generator iteration of GFP with the 3,000 step based on monitoring the calibration model of Appendix B. For bootstrapping, the generator samples 2 candidates every 5 steps. For Top-K sampling at the bootstrapping, we sample with $L = 1,000$ and select the Top 2 samples to augment the training dataset.

Testing. For filtering, we generated M = 1,280 candidate samples and collected the Top-K samples where K = 128 based on the proxy score. For diverse aggregation, we collect K = 16 samples from 8 generators, making a total of 128 samples.

Proxy model. For the proxy model, we applied a weight regularization of $10^{-4}$ , set the learning rate to $10^{-4}$ , and used a dropout rate of 0.1. We used early stopping with a tolerance of 5 and a train/validate ratio of 9:1 following Jain et al. [29]. We used the Adam optimizer [30] for the training generator, proxy, and calibration model.

# B Calibration Model

![](images/bc3eb3e8715a941df549b259acb434cbbbf35d89251691e074da8da5a96591a4.jpg)  
100th Percentile 50th Percentile --- Calibration model

Figure B.1: Calibration model's tendency.

Tuning the hyperparameters of offline design algorithms is challenging due to the lack of access to the true score function. Therefore, existing works have proposed various strategies to circumvent this issue, e.g., choosing a hyperparameter that is transferrable between different tasks $[45]$ or tuning the hyperparameter based on training statistics $[52]$ .

In this work, we leverage the calibration function. Inspired by Wang et al. [47], we train the calibration function on the offline dataset to approximate the true score function similar to the proxy function. Then we use the calibration function to select a score-conditioned model that achieves higher performance with respect to the calibration function. We also choose the number of training steps and early stopping points using the same criterion.

As shown in Fig. B.1, the calibration model accurately predicts early stopping points as the GFP task is unstable and has a narrow high score region which gives a high chance to be overfitted into the low-scored region (Table 4.1 shows that score of GFP is highly polarized). By using the calibration function, we can simply choose an early stopping point for GFP. Note we simply leverage the proxy model as a calibration model with an exact sample training scheme and hyperparameters.

# C Diversity and Novelty Comparison with GFN-AL [29]

Table C.1: Experimental results on 100th percentile scores (100th Per.), 50th percentile scores (50th Per.), average score (Avg. Score), diversity, and novelty, among 128 samples of low dimensional tasks comparing with GFN-AL. The mean and standard deviation of 8 independent runs for producing 128 samples is reported. The best-scored value is marked in bold. 

<table><tr><td></td><td>Methods</td><td>100th Per.</td><td>50th Per.</td><td>Avg. Score</td><td>Diversity</td><td>Novelty</td></tr><tr><td rowspan="3">RNA-A</td><td>GFN-AL</td><td> $0.630 \pm 0.054$ </td><td> $0.312 \pm 0.013$ </td><td> $0.320 \pm 0.010$ </td><td> $8.858 \pm 0.045$ </td><td> $4.269 \pm 0.130$ </td></tr><tr><td>BOOTGEN</td><td> $\mathbf{0.898} \pm 0.039$ </td><td> $\mathbf{0.694} \pm 0.009$ </td><td> $\mathbf{0.699} \pm 0.008$ </td><td> $5.694 \pm 0.008$ </td><td> $\mathbf{7.509} \pm 0.049$ </td></tr><tr><td> $\text{BOOTGEN}^{\dagger}$ </td><td> $0.750 \pm 0.041$ </td><td> $0.382 \pm 0.014$ </td><td> $0.399 \pm 0.008$ </td><td> $\mathbf{8.917} \pm 0.078$ </td><td> $4.957 \pm 0.052$ </td></tr><tr><td rowspan="3">RNA-B</td><td>GFN-AL</td><td> $0.677 \pm 0.080$ </td><td> $0.300 \pm 0.012$ </td><td> $0.311 \pm 0.011$ </td><td> $8.846 \pm 0.050$ </td><td> $4.342 \pm 0.128$ </td></tr><tr><td>BOOTGEN</td><td> $\mathbf{0.886} \pm 0.028$ </td><td> $\mathbf{0.689} \pm 0.007$ </td><td> $\mathbf{0.693} \pm 0.007$ </td><td> $5.192 \pm 0.073$ </td><td> $\mathbf{7.981} \pm 0.036$ </td></tr><tr><td> $\text{BOOTGEN}^{\dagger}$ </td><td> $0.686 \pm 0.052$ </td><td> $0.355 \pm 0.018$ </td><td> $0.371 \pm 0.017$ </td><td> $\mathbf{8.929} \pm 0.121$ </td><td> $4.967 \pm 0.132$ </td></tr><tr><td rowspan="3">RNA-C</td><td>GFN-AL</td><td> $0.623 \pm 0.045$ </td><td> $0.324 \pm 0.010$ </td><td> $0.333 \pm 0.010$ </td><td> $8.831 \pm 0.046$ </td><td> $4.151 \pm 0.088$ </td></tr><tr><td>BOOTGEN</td><td> $\mathbf{0.837} \pm 0.045$ </td><td> $\mathbf{0.598} \pm 0.006$ </td><td> $\mathbf{0.606} \pm 0.006$ </td><td> $4.451 \pm 0.071$ </td><td> $\mathbf{7.243} \pm 0.051$ </td></tr><tr><td> $\text{BOOTGEN}^{\dagger}$ </td><td> $0.651 \pm 0.056$ </td><td> $0.370 \pm 0.011$ </td><td> $0.376 \pm 0.013$ </td><td> $\mathbf{8.913} \pm 0.087$ </td><td> $4.597 \pm 0.073$ </td></tr><tr><td rowspan="3">TFBind8</td><td>GFN-AL</td><td> $0.951 \pm 0.026$ </td><td> $0.537 \pm 0.055$ </td><td> $0.575 \pm 0.037$ </td><td> $5.001 \pm 0.178$ </td><td> $0.778 \pm 0.143$ </td></tr><tr><td>BOOTGEN</td><td> $\mathbf{0.977} \pm 0.004$ </td><td> $\mathbf{0.848} \pm 0.010$ </td><td> $\mathbf{0.839} \pm 0.009$ </td><td> $3.118 \pm 0.045$ </td><td> $\mathbf{1.802} \pm 0.025$ </td></tr><tr><td> $\text{BOOTGEN}^{\dagger}$ </td><td> $0.970 \pm 0.018$ </td><td> $0.613 \pm 0.017$ </td><td> $0.627 \pm 0.014$ </td><td> $\mathbf{5.048} \pm 0.039$ </td><td> $0.965 \pm 0.033$ </td></tr></table>

Table C.2: Experimental results on 100th percentile scores (100th Per.), 50th percentile scores (50th Per.), average score (Avg. Score), diversity, and novelty, among 128 samples of 6 six biological sequential tasks comparing with GFN-AL. The mean and standard deviation of 8 independent runs for producing 128 samples is reported. The best-scored value is marked in bold. The ‘Random’ stands for uniform random generator. 

<table><tr><td></td><td>Methods</td><td>100th Per.</td><td>50th Per.</td><td>Avg. Score</td><td>Diversity</td><td>Novelty</td></tr><tr><td rowspan="3">GFP</td><td>Random</td><td> $0.053 \pm 0.000$ </td><td> $0.051 \pm 0.000$ </td><td> $0.051 \pm 0.000$ </td><td> $\mathbf{219.840} \pm 0.207$ </td><td> $\mathbf{216.960} \pm 0.330$ </td></tr><tr><td>GFN-AL</td><td> $0.057 \pm 0.001$ </td><td> $0.051 \pm 0.004$ </td><td> $0.052 \pm 0.004$ </td><td> $130.113 \pm 41.202$ </td><td> $208.610 \pm 46.271$ </td></tr><tr><td>BOOTGEN</td><td> $\mathbf{0.865} \pm 0.000$ </td><td> $\mathbf{0.854} \pm 0.002$ </td><td> $\mathbf{0.813} \pm 0.011$ </td><td> $7.969 \pm 0.460$ </td><td> $2.801 \pm 0.163$ </td></tr></table>

Building upon the § 4.4 findings of the UTR, we present further multi-objective experimental results for the remaining 5 tasks, comparing them closely with the GFN-AL [29] approach. The GFN-AL model aims to achieve extensive exploration by prioritizing sample diversity and novelty, leading to the generation of diverse, high-quality biological sequences. Nevertheless, the diversity measure occasionally introduces a trade-off between sample scores, particularly when certain tasks exhibit a narrow score landscape, resulting in only a limited number of samples with high scores.

We conducted a detailed analysis to shed light on the relationship between score metrics (average, 100th percentile, 50th percentile) and diversity metrics (diversity and novelty). The results, presented in Table C.2, demonstrate that our proposed method, BOOTGEN, outperforms GFN-AL in terms of score performance. However, it is noteworthy that GFN-AL exhibits high diversity, particularly in the case of GFP. On the contrary, GFN-AL generates extremely low scores for GFP, almost comparable to those produced by a uniform random generator. This observation indicates that the GFP task possesses a narrow score landscape, making it relatively easy to generate diverse yet low-scoring samples.

For the TFbind8 and RNA tasks, GFN-AL achieves high diversity but a low novelty. This suggests that GFN-AL struggles to discover samples beyond the scope of the offline training dataset, resulting in less novel but diverse samples with low scores. In contrast, BOOTGEN successfully identifies high-scoring and novel samples. Consequently, in this scenario, we consider high diversity coupled with low novelty and score to be somewhat meaningless, as such results can also be achieved by a random generator.

To substantiate our claim regarding diversity, we present experimental results of an enhanced diversity version of BOOTGEN. In order to achieve increased diversity, BOOTGEN makes certain sacrifices in terms of score performance. One approach we employ is interpolation with a uniform random sequence generator. Specifically, we combine our generator with the uniform random gen-

erator to generate random samples in a portion of the sequence (we make 3/4 samples from the random generator and 1/4 from BOOTGEN). Additionally, we can filter out low-diversity sequences without requiring score evaluation, thereby generating a more diverse set of samples by referring code of GFN-AL [29]. To this end, we introduce the diversity-improved version of our method, denoted as $\mathrm{BOOTGEN}^{\dagger}$ . It is important to note that $\mathrm{BOOTGEN}^{\dagger}$ sacrifices score performance, as diversity and score are inherent trade-offs, and it focuses primarily on diversity to provide a more direct comparison with GFN-AL by manually adjusting diversity.

As shown in Table C.2, BOOTGEN $^{\dagger}$ exhibits similar diversity levels in RNA-A, RNA-B, RNA-C, and TFBind8, while achieving higher score metrics and novelty. We attribute these results to GFN's underfitting issue, as it fails to adequately fit within the high-scoring region of the score landscape, particularly for the high-dimensional tasks of UTR and GFP.

# D Rank-based weighting vs. Value-based weighting

![](images/56d9aa38b3e9a20fb6f85f29df600d085e71682993747ad51935aeed8734b06e.jpg)

<details>
<summary>line</summary>

| Iteration | NW    | VW (T=0.1) | VW (T=0.3) | VW (T=0.7) | VW (T=1.0) | RW (ours) |
| --------- | ----- | ---------- | ---------- | ---------- | ---------- | --------- |
| 0         | 0.65  | 0.65       | 0.65       | 0.65       | 0.65       | 0.65      |
| 2000      | 0.78  | 0.75       | 0.77       | 0.79       | 0.78       | 0.80      |
| 4000      | 0.80  | 0.76       | 0.78       | 0.81       | 0.80       | 0.83      |
| 6000      | 0.81  | 0.77       | 0.79       | 0.82       | 0.81       | 0.85      |
| 8000      | 0.81  | 0.77       | 0.79       | 0.82       | 0.81       | 0.86      |
| 10000     | 0.81  | 0.77       | 0.79       | 0.82       | 0.81       | 0.86      |
| 12000     | 0.81  | 0.77       | 0.79       | 0.82       | 0.81       | 0.86      |
| 14000     | 0.81  | 0.77       | 0.79       | 0.82       | 0.81       | 0.86      |
</details>

Figure D.1: Comparison of rank-based weighting (RW) and value-based weighting (VW) methods. The NW represents the case where no weighting is applied to the training distribution. In the VW case, we explored different weighting temperatures, $T \in \{0.1, 0.3, 0.7, 1.0\}$ . The 50th percentile scores of TFBind8 are reported, and the results include a bootstrapping procedure applied from iteration 12,500 to 15,000.

We verify the contribution of the rank-based weighting (RW) scheme compared to ours with no weighting (NW) and the existing value-based weighting (VW) proposed by Kumar and Levine [32]. To implement VW, we set the sample-wise weight proportional to $\exp(|y - y^{*}|/T)$ , where $y^{*}$ is the maximum score in the training dataset and $T \in \{0.1, 0.3, 0.7, 1.0\}$ is a hyperparameter. As shown in Fig. D.1, the results indicate that RW outperforms both NW and VW. This validates our design choice for BOOTGEN.