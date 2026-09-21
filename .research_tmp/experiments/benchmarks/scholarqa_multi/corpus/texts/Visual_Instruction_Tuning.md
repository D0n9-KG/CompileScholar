# Adaptive Task Balancing for Visual Instruction Tuning via Inter-Task Contribution and Intra-Task Difficulty

Yanqi Dai*

yanqidai@ruc.edu.cn

Renmin University of China

Beijing, China

Yong Wang

wangyong.lz@alibaba-inc.com

AMAP, Alibaba Group

Beijing, China

Zebin You

zebin@ruc.edu.cn

Renmin University of China

Beijing, China

Dong Jing

jingdong98@ruc.edu.cn

Renmin University of China

Beijing, China

Xiangxiang Chu

chuxiangxiang.cxx@alibaba-inc.com

AMAP, Alibaba Group

Beijing, China

Zhiwu Lu†

luzhiwu@ruc.edu.cn

Renmin University of China

Beijing, China

# Abstract

Visual instruction tuning is a key training stage of large multimodal models. However, when learning multiple visual tasks simultaneously, this approach often results in suboptimal and imbalanced overall performance due to latent knowledge conflicts across tasks. To mitigate this issue, we propose a novel Adaptive Task Balancing approach tailored for visual instruction tuning (VisATB).

Specifically, we measure two critical dimensions for visual task balancing based on validation performance: (1) Inter-Task Contribution, the mechanism where learning one task enhances the performance on others owing to shared knowledge across tasks, and (2) Intra-Task Difficulty, which denotes the inherent learning difficulty of a single task. Furthermore, we propose prioritizing three categories of tasks with greater weight: those that offer substantial contributions to others, those that receive minimal contributions from others, and those that present high learning difficulties.

Among these three task weighting strategies, the first and third focus on improving overall performance, and the second targets the mitigation of performance imbalance. Extensive experiments on three benchmarks demonstrate that our VisATB approach consistently achieves superior and more balanced overall performance in visual instruction tuning. The data, code, and models are available at YanqiDai/VisATB.

# CCS Concepts

- Computing methodologies $\rightarrow$ Natural language generation; Scene understanding; Object detection; Object recognition.

# Keywords

LMMs, Visual Instruction Tuning, Task Balancing

# ACM Reference Format:

Yanqi Dai, Yong Wang, Zebin You, Dong Jing, Xiangxiang Chu, and Zhiwu Lu. 2026. Adaptive Task Balancing for Visual Instruction Tuning via InterTask Contribution and Intra-Task Difficulty. In Proceedings of the ACM Web

*Part of this work was done during Yanqi Dai's internship at AMAP, Alibaba Group.  
† Corresponding author.

![](dt=2026-05-27/ht=14/5649a1afcbe6c96a0dbab9d35be4e2e688f06ec6d7bfa3ca03529e2f56e9a4c1.jpg)

This work is licensed under a Creative Commons Attribution 4.0 International License.

WW'26, Dubai, United Arab Emirates

© 2026 Copyright held by the owner/author(s).

ACM ISBN 979-8-4007-2307-0/2026/04

https://doi.org/10.1145/3774904.3792371

![](dt=2026-05-27/ht=14/6f641abaaa4976fd02d262aedf52d7532b7821da0efad98ac7f30575356e38ad.jpg)

Task: Detailed Image Captioning

The image captures a dynamic scene in the vast, deep blue ocean. Two surfers, clad in wetsuits, are the main subjects of this image. The surfer closest to us, donned in a black wetsuit, is skillfully riding a wave. ...

Overlapping Knowledge Domains

Task: Visual Question Answering

How many surfers are in the picture? 2 What color is the wetsuits? Black

What are formed? Waves

![](dt=2026-05-27/ht=14/22c2a6550114240eed151ac5aeb24816aaef04b8f1011450c90c11ecf83af8d3.jpg)

Conference 2026 (WWW '26), April 13-17, 2026, Dubai, United Arab Emirates. ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3774904.3792371

# 1 Introduction

Large multimodal models (LMMs) [2, 32, 54] have garnered significant attention for their capability to understand and reason across both visual and textual modalities. A pivotal advancement in this field is visual instruction tuning [33], which integrates visual encoders with large language models (LLMs) using visual instruction-following data and specialized alignment modules. This innovative technique extends the robust, general-purpose capabilities of LLMs to the visual modality, substantially enhancing both the efficiency

arXiv:2403.04343v3 [cs.AI] 21 Jan 2026

and effectiveness of LMM training. Many approaches, such as the LLaVA series [23, 31-33], have achieved remarkable results through visual instruction tuning.

To endow LMMs with diverse visual abilities, instruction-following data from multiple visual tasks are frequently combined indiscriminately for visual instruction tuning [5]. However, this approach faces a critical challenge: different tasks necessitate task-specific latent knowledge. For instance, the image captioning task demands comprehensive scene understanding and holistic description generation, whereas the visual grounding task emphasizes fine-grained localization of specific textual phrases.

Consequently, simultaneous training across multiple tasks imposes a substantial learning burden on the model due to task interference, potentially resulting in suboptimal performance compared to training on each task individually [8, 14]. Manual specification of data mixing ratios or task weights heavily relies on extensive ablation studies and expert knowledge, rendering such approaches both resource-intensive and difficult to generalize across different model architectures and tasks.

To mitigate this issue, we perform a rigorous analysis of the task relationship and identify two key concepts: inter-task contribution and intra-task difficulty. First, as illustrated in Figure 1a, we observe that tasks often share overlapping knowledge domains, facilitating knowledge transfer that boosts performance in related tasks. The extent of these overlaps varies across tasks, leading to differing degrees of inter-task contributions. Second, Figure 1b reflects that different tasks demonstrate distinct performance improvement trajectories as training data increases.

Specifically, tasks that achieve near-optimal performance with limited training data are regarded as relatively simple, whereas tasks that require extensive training data to reach optimal performance exhibit higher inherent learning difficulty.

Moreover, we introduce a novel Adaptive Task Balancing approach for visual instruction tuning (VisATB) based on the above two critical perspectives, integrating three task weighting strategies, each with its unique characteristics, as presented in Figure 2. In the preparation stage, we measure the inter-task contribution of one task to another task by training a model on one task and evaluating its normalized validation performance on the other task.

Additionally, to quantify the intra-task difficulty of a target task, we estimate the normalized validation performance gap between a model trained on a mini subset of the task and one trained on the entire dataset or a sufficiently large subset. Subsequently, in the task weight calculation stage, we recommend assigning greater weight to tasks that (1) offer substantial contribution to others, (2) receive minimal contribution from others, and (3) present high learning difficulties.

Among these three task weighting strategies, the first and third focus on improving overall performance, while the second aims to mitigate performance imbalance across tasks. VisATB integrates them to achieve a more robust and balanced overall performance. In the final training stage, we propose a Visual Instruction Task Weighting (VITW) paradigm tailored for visual instruction tuning, where losses are assigned task-specific weights and averaged at the token level. Building upon this paradigm, the final model is trained using the entire dataset of all tasks and the integrated task weight.

Our
contributions can be summarized as follows:

# 2 Method

In this section, we first introduce the Visual Instruction Task Weighting (VITW) paradigm tailored for visual instruction tuning. Building upon this paradigm, we analyze two crucial dimensions for visual task balancing: inter-task contribution and intra-task difficulty, and accordingly propose three task weighting strategies. Finally, we integrate these strategies to formulate the Adaptive Task Balancing (VisATB) approach.

# 2.1 Visual Instruction Task Weighting

Typically, data from various instruction-following tasks are mixed indiscriminately for visual instruction tuning. In this process, the training loss is computed as the average cross-entropy loss across all valid tokens, as expressed in the following equation:

$$
L = \frac {\sum_ {i = 1} ^ {N} \sum_ {j = 1} ^ {S _ {i}} \sum_ {k = 1} ^ {T _ {i j}} - \log (p \left(t _ {i j k}\right))}{\sum_ {i = 1} ^ {N} \sum_ {j = 1} ^ {S _ {i}} T _ {i j}}, \tag {1}
$$

where $N$ denotes the task number, $S_{i}$ indicates the sample number of Task $i$ , $T_{ij}$ represents the valid token number in the $j$ -th sample for Task $i$ , and $t_{ijk}$ signifies the $k$ -th valid token in the $j$ -th sample for Task $i$ . However, the traditional task weighting paradigm, which computes the total loss as a weighted average of individual task losses, is incompatible in this context. Differences in sequence length and sample size across tasks result in varying numbers of valid tokens for each task. Therefore, computing each task loss and averaging across all tasks introduces implicit weight to the losses of valid tokens, leading to biased learning across tasks.

To address this issue, we design a Visual Instruction Task Weighting (VITW) paradigm tailored for visual instruction tuning. The training loss of VITW is calculated as:

$$
L _ {\mathrm {V I T W}} = \frac {\sum_ {i = 1} ^ {N} \sum_ {j = 1} ^ {S _ {i}} \sum_ {k = 1} ^ {T _ {i j}} - \lambda_ {i} \log (p (t _ {i j k}))}{\sum_ {i = 1} ^ {N} \sum_ {j = 1} ^ {S _ {i}} \lambda_ {i} T _ {i j}}, \tag {2}
$$

where $\lambda_{i}$ signifies the weight of Task $i$ . The losses of valid tokens are assigned task-specific weight and averaged at the token level, rather than at the task level, to ensure an equitable consideration for each valid token. It is a robust foundation for VisATB and has the potential to inform future research in visual instruction tuning.

# 2.2 Inter-Task Contribution Balancing

Although the focal points of different tasks may vary in visual instruction tuning, a central shared objective exists: enhancing the capability of understanding and reasoning about visual information. As presented in Figure 1a, the detailed image captioning

WWW '26, April 13-17, 2026, Dubai, United Arab Emirates

Yanqi Dai et al.

![](dt=2026-05-27/ht=14/54d2e8545ad3f22fabe38eef7fc2ad27c6126458094ea859573b096f6b4b578f.jpg)

![](dt=2026-05-27/ht=14/b1d8da2a213661282ada5169e976273775afbd8b960c38760a216ec494950a0c.jpg)

data from ShareGPT4V [7] and the visual question answering data from VQAv2 [15] contain shared information for the same image, such as color, quantity, and category, exemplifying the overlapping knowledge domains among tasks. Therefore, learning one task can potentially enhance performance on others, through the mechanism we define as inter-task contribution, supported by the results in Section 3. The extent of inter-task contribution varies according to the degree of overlap in knowledge domains among tasks.

In practice, the inter-task contribution of Task $i$ to Task $j$ can be quantified as the normalized validation performance on Task $j$ of a model trained on Task $i$ , as follows:

$$
C _ {i \rightarrow j} = \frac {V _ {j} (i + \operatorname* {m i n i}) - V _ {j} (\operatorname* {m i n i})}{V _ {j} (j + \operatorname* {m i n i}) - V _ {j} (\operatorname* {m i n i})}, \tag {3}
$$

where $V_{j}(i + \mathrm{mini})$ denotes the performance on Task $j$ of a model trained on the entire dataset or a large enough subset of Task $i$ and mini subsets of all other tasks, and $V_{j}(\mathrm{mini})$ indicates the performance on Task $j$ of a model trained on mini subsets of all tasks. The large enough subsets are randomly sampled from the entire datasets, while the mini subsets are randomly sampled from these large enough subsets. To ensure fairness across tasks, each of these two sampling rates remains consistent across all tasks and is independent of the other.

Notably, the mini subsets are added to the training data to ensure the model understands the instruction demands of all tasks. In the formula, $V_{j}(\mathrm{mini})$ is subtracted from both the numerator and the denominator, which mitigates the influence of incorporating mini subsets into the training set on the validation performance on Task $j$ .

Based on the precise quantification of the inter-task contribution, we propose two novel task weighting strategies for inter-task contribution balancing:

(1) Task-Outward Contribution Balancing: We describe the task-outward contribution, $C_{\mathrm{out}}$ , as the average inter-task contribution of a single task to all other tasks. This concept denotes the

extent to which one task contributes to the performance of all other tasks. Tasks with higher $C_{\mathrm{out}}$ are more beneficial for overall training. Therefore, we propose assigning greater weight to tasks with higher $C_{\mathrm{out}}$ to improve collective performance across all tasks. Specifically, the task weight, $\lambda_{\mathrm{out}}$ , for task-outward contribution balancing is computed as:

$$
\lambda_ {\mathbf {o u t}} = N \times \operatorname {s o f t m a x} \left(\frac {C _ {\mathbf {o u t}}}{T}\right), \text {w h e r e} C _ {\mathrm {o u t}, i} = \frac {\sum_ {j \neq i} C _ {i \rightarrow j}}{N - 1}. \tag {4}
$$

Here, $C_{\mathrm{out},i}$ represents the task-outward contribution of Task $i$ , $C_{\mathrm{out}}$ denotes the task-outward contribution vector of all tasks, and $T$ is the temperature hyperparameter.

(2) Task-Inward Contribution Balancing: Conversely, we characterize the task-inward contribution, $C_{\mathrm{in}}$ , as the average intertask contribution from all other tasks to a single task. This concept signifies the degree to which the performance of one task benefits from all other tasks. In comparison to tasks that benefit from more collaborative training, those with lower $C_{\mathrm{in}}$ are more likely to exhibit reduced performance. Therefore, we propose assigning greater weight to tasks with lower $C_{\mathrm{in}}$ to mitigate performance imbalance. Specifically, the task weight, $\lambda_{\mathrm{in}}$ , for task-inward contribution balancing is calculated as:

$$
\lambda_ {\mathbf {i n}} = N \times \operatorname {s o f t m a x} \left(- \frac {C _ {\mathbf {i n}}}{T}\right), \text {w h e r e} C _ {\mathrm {i n}, i} = \frac {\sum_ {j \neq i} C _ {j \rightarrow i}}{N - 1}. \tag {5}
$$

Here, $C_{\mathrm{in},i}$ denotes the task-inward contribution of Task $i$ , $C_{\mathrm{in}}$ signifies the task-inward contribution vector of all tasks, and $T$ is the temperature hyperparameter.

# 2.3 Intra-Task Difficulty Balancing

Moreover, the inherent learning difficulty, termed intra-task difficulty, also varies substantially across tasks in visual instruction tuning. As presented in Figure 1b, different tasks exhibit distinct performance improvement trajectories w.r.t. increasing training

Adaptive Task Balancing for Visual Instruction Tuning via Inter-Task Contribution and Intra-Task Difficulty

WWW '26, April 13-17, 2026, Dubai, United Arab Emirates

data amount. Tasks that achieve near-optimal performance with limited training data are considered to have lower intra-task difficulties, whereas tasks that require extensive training data to reach optimal p
erformance demonstrate higher intra-task difficulties. For example, the tasks shown in Figure 1b are arranged in ascending order of intra-task difficulty as follows: visual question answering, detailed image captioning, and visual grounding.

In practice, the intra-task difficulty of Task $i$ can be measured as the normalized validation performance gap on Task $i$ between a model trained on a mini subset of Task $i$ and one trained on the entire dataset or a large enough subset of the same task. Additionally, we repurpose the additional models trained for inter-task contribution balancing to reduce time costs. Consequently, the intra-task difficulty of Task $i$ can be calculated as follows:

$$
D _ {i} = 1 - \frac {V _ {i} (\text {m i n i})}{V _ {i} (i + \text {m i n i})}, \tag {6}
$$

where $V_{i}(i + \mathrm{mini})$ indicates the performance on Task $i$ of a model trained on the entire dataset or a large enough subset of Task $i$ and mini subsets of all other tasks, and $V_{i}(\mathrm{mini})$ represents the performance on Task $i$ of a model trained on mini subsets of all tasks. Importantly, the impact of mini subsets from other tasks on the Task $i$ validation performance is negligible compared to the impact of its own mini subset. Therefore, model repurposing can significantly reduce additional time costs while maintaining minimal error in the computation of intra-task difficulties.

Due to the varying degrees of intra-task difficulties across tasks, learning all tasks equally may result in underfitting on more challenging tasks, even when simpler ones are overfitted. Therefore, we propose assigning greater weight to tasks with higher $D$ . Specifically, the task weight, $\lambda_{\mathbf{D}}$ , for intra-task difficulty balancing is computed as follows:

$$
\lambda_ {\mathbf {D}} = N \times \operatorname {s o f t m a x} \left(\frac {D}{T}\right), \tag {7}
$$

where $D$ denotes the intra-task difficulty vector of all tasks, and $T$ is the temperature hyperparameter.

# 2.4 VisATB: Adaptive Task Balancing

Finally, these three aforementioned task weighting strategies are integrated to formulate our VisATB approach. The specific task weight, $\lambda_{\mathrm{VisATB}}$ , for adaptive task balancing is computed as:

$$
\lambda_ {\text {V i s A T B}} = \alpha_ {\text {o u t}} \lambda_ {\text {o u t}} + \alpha_ {\text {i n}} \lambda_ {\text {i n}} + \alpha_ {\mathrm {D}} \lambda_ {\mathrm {D}}, \tag {8}
$$

where $\alpha_{\mathrm{out}}$ , $\alpha_{\mathrm{in}}$ , and $\alpha_{\mathrm{D}}$ denote the proportional coefficients, satisfying the constraint $\alpha_{\mathrm{out}} + \alpha_{\mathrm{in}} + \alpha_{\mathrm{D}} = 1$ .

In summary, tasks that offer substantial inter-task contributions to others and present high intra-task difficulties are assigned greater weight to achieve superior overall performance, while tasks that receive minimal inter-task contributions from others are assigned greater weight to mitigate performance imbalance.

# 3 Experiments

# 3.1 Experimental Setup

**Benchmarks.** We train and evaluate LMMs using three diverse multimodal benchmarks: a M3IT Benchmark [25] comprising 17 training tasks and 1.2 million training samples, an Academic Benchmark [31] including 8 training tasks, and a Chat Benchmark [33]

encompassing 3 training tasks. Moreover, we assess models on 7 unseen zero-shot tasks in the Academic Benchmark. A detailed description of the tasks and data preparation for these benchmarks is provided in Appendix A.

Compared Methods. We compare VisATB against the following baselines and methods: (1) Single-Task Learning (STL), where models are trained and tested on each single task; (2) Equal Weighting (EW), the most common approach, which minimizes the loss in Equation 1; (3) Task-Level Loss Aggregation (TLA), which calculates the loss within each task and then averages across all tasks; (4) Random Loss Weighting (RLW) [28]; (5) Dynamic Weight Average (DWA) [35]; and (6) Improvable Gap Balancing (IGBv1) [10]. Methods (4)-(6) are traditional task weighting methods, as described in Section 4.

We leverage VITW to adapt them for visual instruction tuning. Additionally, gradient-based task weighting methods are excluded for comparison due to the substantial computational cost of aggregating gradients from large-scale model parameters.

Evaluation Metrics. We first report the common evaluation metrics for each visual task. Furthermore, we introduce two overall metrics: $\Delta I\%$ , average per-task improvement, and $\Delta E\%$ , average per-task error, in test performance on fine-tuned tasks compared to the STL baseline, which are calculated as follows:

$$
\Delta I \% = \frac{1}{N}\sum_{i = 1}^{N}I_{i},\Delta E \% = \frac{1}{N}\sum_{i = 1}^{N}\max (0, - I_{i}),
$$

$$
\text {w h e r e} I _ {i} = \frac {1}{K _ {i}} \sum_ {j = 1} ^ {K _ {i}} (- 1) ^ {\delta_ {i j}} \frac {M _ {\mathrm {e} , i j} - M _ {\mathrm {b} , i j}}{M _ {\mathrm {b} , i j}}. \tag {9}
$$

Here, $N$ represents the task number, $K_{i}$ signifies the metric number for Task $i$ , and $I_{i}$ denotes the test performance improvement on Task $i$ . $M_{\mathrm{e},ij}$ and $M_{\mathrm{b},ij}$ are the values on the $j$ -th metric for Task $i$ of the models trained by the evaluated method and the STL baseline. $\delta_{ij}$ is an indicator function, where $\delta_{ij} = 0$ if a higher value is better on the $j$ -th metric for Task $i$ , and $\delta_{ij} = 1$ otherwise.

$\Delta I\%$ reflects the extent of overall performance improvement, while $\Delta E\%$ signifies the degree of performance imbalance. Moreover, to evaluate the generalizability of methods, we introduce $\Delta I_{zero}\%$ , average per-task improvement in test performance on zero-shot tasks compared to EW. The calculation of $\Delta I_{zero}\%$ is similar to that of $\Delta I\%$ , except the STL baseline is replaced with the EW method.

Implementation Details. In the main experiments, we train the pretrained LLaVA-v1.5-7B model on eight NVIDIA A100 GPUs using the same settings as Liu et al. [31]. This choice is driven by LLaVA's well-established architecture and the accessibility of its pretrained models. In the $\mathrm{M}^3\mathrm{IT}$ Benchmark, tasks are categorized into 5 groups following Li et al. [25]. We consider each task group as a whole and perform balancing at the group level. The temperature $T$ is set as 1.0 in the $\mathrm{M}^3\mathrm{IT}$ Benchmark, and 0.5 in the Academic Benchmark and the Chat Benchmark. The proportional coefficients of three task weighting strategies are consistently set as: $\alpha_{\mathrm{out}} = 0.25$ , $\alpha_{\mathrm{in}} = 0.25$ , and $\alpha_{\mathrm{D}} = 0.5$ .

To calculate inter-task contributions and intra-task difficulties, the entire datasets and 1/32nd mini subsets of tasks are utilized in the Academic Benchmark, which is guided by the training loss decline pattern, as detailed in Appendix B. In the Chat Benchmark, there are no specific constraints on the output format. Consequently,

WWW '26, April 13-17, 2026, Dubai, United Arab Emirates

Yanqi Dai et al.

Table 1: Comparative results on the ${\mathrm{M}}^{3}\mathrm{{IT}}$ Benchmark. ${\Delta I}\%$ and ${\Delta E}\%$ are the average per-task improvement and error on fine-tuned tasks compared to the STL baseline.

![](dt=2026-05-27/ht=14/c641d3bce9984a1c9a8118229efe95ac48d7628469d00f3126d577f289642a4e.jpg)

<table><tr><td rowspan="3">Methods</td><td colspan="3">Image Captioning</td><td rowspan="2">GOI</td><td rowspan="2">Text</td><td colspan="4">Classification</td><td rowspan="3">Moch</td><td rowspan="2">Shap</td><td colspan="2">VQA</td><td colspan="2">Reasoning</td><td colspan="2">Generation</td><td colspan="2">Overall</td><td></td></tr><tr><td>COCO</td><td>TCap</td><td>PCap</td><td>INet</td><td>ITM</td><td>SVE</td><td></td><td>OCR</td><td>GQA</td><td>SQA</td><td>CLE</td><td>NL</td><td>VisD</td><td>M30k</td><t
d>ΔI%↑</td><td>ΔE%↓</td></tr><tr><td colspan="3">CIDEr↑</td><td colspan="5">EM↑</td><td></td><td colspan="2">EM↑</td><td colspan="2">EM↑</td><td colspan="2">EM↑</td><td>EM↑</td><td>GLEU↑</td><td></td><td></td></tr><tr><td>STL</td><td>1.024</td><td>0.751</td><td>0.179</td><td>90.7</td><td>100</td><td>98.0</td><td>98.9</td><td>81.2</td><td>84.6</td><td>63.7</td><td>56.5</td><td>48.2</td><td>73.4</td><td>57.4</td><td>72.4</td><td>42.8</td><td>0.358</td><td></td><td></td><td></td></tr><tr><td>EW</td><td>1.022</td><td>0.747</td><td>0.250</td><td>91.2</td><td>100</td><td>98.0</td><td>99.3</td><td>81.8</td><td>85.6</td><td>66.0</td><td>54.9</td><td>52.4</td><td>79.6</td><td>59.9</td><td>71.1</td><td>42.5</td><td>0.351</td><td>3.51</td><td>0.49</td><td></td></tr><tr><td>TLA</td><td>1.001</td><td>0.739</td><td>0.219</td><td>92.1</td><td>100</td><td>97.7</td><td>99.2</td><td>81.6</td><td>85.6</td><td>65.3</td><td>55.5</td><td>49.5</td><td>76.5</td><td>60.4</td><td>69.0</td><td>42.2</td><td>0.342</td><td>1.43</td><td>0.97</td><td></td></tr><tr><td>RLW</td><td>1.016</td><td>0.748</td><td>0.245</td><td>90.8</td><td>100</td><td>97.4</td><td>99.2</td><td>81.8</td><td>79.8</td><td>64.5</td><td>54.5</td><td>50.3</td><td>76.4</td><td>55.1</td><td>69.2</td><td>41.6</td><td>0.340</td><td>1.21</td><td>1.59</td><td></td></tr><tr><td>DWA</td><td>1.021</td><td>0.757</td><td>0.243</td><td>89.8</td><td>100</td><td>97.5</td><td>99.3</td><td>81.5</td><td>79.0</td><td>67.5</td><td>55.5</td><td>53.4</td><td>80.6</td><td>59.9</td><td>70.9</td><td>42.4</td><td>0.351</td><td>3.08</td><td>0.91</td><td></td></tr><tr><td>IGBv1</td><td>1.006</td><td>0.727</td><td>0.227</td><td>89.7</td><td>100</td><td>97.1</td><td>99.1</td><td>76.1</td><td>76.4</td><td>63.9</td><td>50.6</td><td>45.4</td><td>72.2</td><td>52.8</td><td>63.0</td><td>40.8</td><td>0.343</td><td>-2.55</td><td>4.16</td><td></td></tr><tr><td>VisATB</td><td>1.019</td><td>0.756</td><td>0.258</td><td>91.7</td><td>100</td><td>97.9</td><td>99.4</td><td>82.2</td><td>86.1</td><td>68.0</td><td>55.1</td><td>52.7</td><td>81.1</td><td>61.8</td><td>71.6</td><td>42.7</td><td>0.351</td><td>4.52</td><td>0.39</td><td></td></tr></table>

Table 2: The task weights calculated in VisATB on the $\mathbf{M}^3\mathbf{IT}$ Benchmark.

![](dt=2026-05-27/ht=14/fd65aa97df94412b187f7d5d6b5513e95ceb81bc8081b08c45582f962c2bb8c5.jpg)

<table><tr><td>Task Weights</td><td>Cap.</td><td>Cls.</td><td>VQA</td><td>Reas.</td><td>Gen.</td></tr><tr><td>λout</td><td>0.9072</td><td>0.9978</td><td>1.1010</td><td>0.9927</td><td>1.0013</td></tr><tr><td>λin</td><td>0.9626</td><td>0.7375</td><td>1.2822</td><td>1.0286</td><td>0.9891</td></tr><tr><td>λD</td><td>1.1831</td><td>0.9421</td><td>0.9383</td><td>0.9967</td><td>0.9399</td></tr><tr><td>λVisATB</td><td>1.0590</td><td>0.9048</td><td>1.0649</td><td>1.0037</td><td>0.9675</td></tr></table>

only the entire datasets of tasks are required, without the need for mini subsets, simplifying the form of VisATB, as detailed in Appendix C. In the $\mathrm{M}^3\mathrm{IT}$ Benchmark, the training data size of each task group is substantial; thus, we use 1/4th (large enough) subsets and 1/32nd mini subsets of tasks. In practice, guided by experimental analysis in Appendix G, we suggest that subsets containing more than 10k samples and trained for over 100 steps are sufficiently large to ensure effective training of VisATB.

# 3.2 Evaluation on the $\mathbf{M}^3\mathrm{IT}$ Benchmark

Effectiveness of VisATB. The comparative results of the fine-tuned tasks on the $\mathbf{M}^3\mathrm{IT}$ Benchmark are presented in Table 1. Utilizing the same pretrained model and training data, VisATB achieves the optimal performance in both $\Delta I\%$ and $\Delta E\%$ . Specifically, VisATB attains the best performance on 11 out of 17 tasks and exhibits nearoptimal performance on the remaining tasks, which demonstrates the effectiveness of VisATB in both improving overall performance and mitigating performance imbalance across diverse visual tasks. Notably, in the reasoning group, VisATB substantially outperforms all baselines, such as 81.1 on SQA and 61.8 on CLE, highlighting its potential strength in complex reasoning tasks.

Additionally, the specific task weights calculated in VisATB are presented in Table 2. In the classification task group, although the tasks are assigned relatively small weights (0.9048), VisATB still demonstrates strong performance, attaining the optimal results on 4 out of 6 tasks and ranking second-best on the remaining tasks. This counterintuitive finding suggests an important insight: for certain relatively simple tasks, directly increasing their weight may not lead to further performance improvements due to potential overfitting or saturation effects. Instead, our approach of prioritizing tasks that contribute more significantly to these simple ones (as reflected in high task-outward contributions) has the potential to transcend

individual task performance limits by facilitating the acquisition of new, transferable knowledge across the task space.

Validity of VITW. We compare TLA with EW to systematically evaluate the validity of our VITW paradigm in visual instruction tuning. The results provide compelling evidence for the necessity of token-level loss aggregation. Specifically, TLA performs substantially worse than EW in both overall metrics: it achieves only $1.43\%$ in $\Delta I\%$ compared to $3.51\%$ of EW, and exhibits a more pronounced performance imbalance with a $\Delta E\%$ of $0.97\%$ versus $0.49\%$ of EW.

As discussed in Section 2.1, this performance degradation stems from the implicit weight bias introduced by TLA, which is inversely proportional to the number of valid tokens in each task. For example, the image captioning task group, which naturally involves longer textual descriptions and thus a larger number of valid tokens per sample, receives lower implicit weight in the optimization process, thereby leading to the poorest performance.

These findings underscore the validity of our VITW paradigm, which ensures equitable treatment of all valid tokens by computing weighted losses at the token level before aggregation.

Limitation of Traditional Task Weighting Methods. Directly applying traditional task weighting methods, including RLW, DWA, and IGBv1, in visual instruction tuning yields markedly inferior performance. These results underscore a fundamental limitation of traditional methods, which balance tasks solely based on training losses, when applied to visual instruction tuning: training losses fail to serve as reliable indicators of actual task learning progress and generalization capacity in LMMs. In contrast, the validation performance-based measurement in VisATB provides a more accurate reflection of the model's actual capabilities and learning trajectories across tasks, leading to superior task weight determination and significant performance improvements.

Quantitative Analysis of the Time Cost of VisATB. An important practical consideration for any task balancing method is its computational overhead. We provide a comprehensive analysis of the additional training time required by VisATB. The extra training time of VisATB is approximately $(R_{\text{large}} + N \times R_{\text{mini}})$ times the training duration of the final model, where $N$ denotes the task number, $R_{\text{large}}$ and $R_{\text{mini}}$ represent the sampling rates of large enough subsets and mini subsets. In the M<sup>3</sup>IT Benchmark, with $N = 5$ , $R_{\text{large}} = 1/4$ , and $R_{\text{mini}} = 1/32$ , the additional training time is about $0.25 + 5 \times 0.03 = 0.40$ times that of the final model.

Adaptive Task Balancing for Visual Instruction Tuning via Inter-Task Contribution and Intra-Task Difficulty

WWW '26, April 13-17, 2026, Dubai, United Arab Emirates

Table 3: Comparative results of the fine-tuned tasks on the Academi
c Benchmark. $\Delta I\%$ and $\Delta E\%$ are the average per-task improvement and error on fine-tuned tasks compared to the STL baseline.

![](dt=2026-05-27/ht=14/643a2537adda44db9b8911e1c3f18115f055b45113efa869cfc4e4132f7ff127.jpg)

<table><tr><td rowspan="2">Methods</td><td rowspan="2">ShareGPT4V
CIDEr↑</td><td rowspan="2">Ref-caption
CIDEr↑</td><td rowspan="2">VQAv2
EM↑</td><td rowspan="2">GQA
EM↑</td><td rowspan="2">ChartQA
EM↑</td><td rowspan="2">OCRVQA
EM↑</td><td rowspan="2">Ref-bbox
IoU↑</td><td colspan="2">Overall</td></tr><tr><td>ΔI%↑</td><td>ΔE%↓</td></tr><tr><td>STL</td><td>0.1285</td><td>0.4658</td><td>77.73</td><td>61.23</td><td>17.76</td><td>68.22</td><td>51.58</td><td></td><td></td></tr><tr><td>EW</td><td>0.1411</td><td>0.5591</td><td>78.27</td><td>62.20</td><td>19.60</td><td>67.73</td><td>61.63</td><td>8.75</td><td>0.10</td></tr><tr><td>TLA</td><td>0.1144</td><td>0.5770</td><td>77.72</td><td>60.42</td><td>22.36</td><td>67.80</td><td>56.58</td><td>6.65</td><td>1.85</td></tr><tr><td>RLW</td><td>0.1388</td><td>0.5571</td><td>77.28</td><td>60.61</td><td>18.20</td><td>66.73</td><td>55.86</td><td>4.95</td><td>0.54</td></tr><tr><td>DWA</td><td>0.1225</td><td>0.5470</td><td>78.28</td><td>61.82</td><td>19.88</td><td>67.87</td><td>61.12</td><td>6.34</td><td>0.74</td></tr><tr><td>IGBv1</td><td>0.1349</td><td>0.4824</td><td>77.00</td><td>60.92</td><td>17.20</td><td>65.96</td><td>55.47</td><td>0.17</td><td>1.13</td></tr><tr><td>VisATB</td><td>0.1437</td><td>0.5724</td><td>77.99</td><td>61.81</td><td>20.16</td><td>67.48</td><td>67.38</td><td>11.29</td><td>0.15</td></tr></table>

Table 4: Comparative results of the zero-shot tasks on the Academic Benchmark. $\Delta {I}_{\text{zero }}\%$ is the average per-task improvement on zero-shot tasks compared to the EW method.

![](dt=2026-05-27/ht=14/161a25ae6982d8e54ba60c6bea4f5a3a159ac949d39e507ff0846a059dc7e152.jpg)

<table><tr><td>Methods</td><td>TextVQA↑</td><td>POPE↑</td><td>MME↑</td><td>SQA↑</td><td>MMBench↑</td><td>SEEDI↑</td><td>MM-Vet↑</td><td>Overall (ΔIzero%↑)</td></tr><tr><td>EW</td><td>53.96</td><td>86.83</td><td>1524.41</td><td>60.74</td><td>48.54</td><td>54.13</td><td>29.00</td><td>0.00</td></tr><tr><td>TLA</td><td>50.74</td><td>86.94</td><td>1426.51</td><td>55.86</td><td>52.92</td><td>46.67</td><td>28.70</td><td>-3.73</td></tr><tr><td>RLW</td><td>52.74</td><td>86.29</td><td>1504.06</td><td>59.99</td><td>52.58</td><td>56.54</td><td>28.70</td><td>0.90</td></tr><tr><td>DWA</td><td>54.05</td><td>86.76</td><td>1484.26</td><td>60.53</td><td>49.31</td><td>54.75</td><td>28.90</td><td>-0.07</td></tr><tr><td>IGBv1</td><td>52.09</td><td>86.86</td><td>1497.02</td><td>57.79</td><td>39.95</td><td>47.26</td><td>29.30</td><td>-5.63</td></tr><tr><td>VisATB</td><td>53.87</td><td>86.73</td><td>1501.86</td><td>61.07</td><td>51.46</td><td>58.05</td><td>29.30</td><td>1.87</td></tr></table>

However, this overhead exhibits favorable scaling properties:

Consequently, VisATB can effectively function across diverse scales and scenarios without substantially increasing complexity or incurring prohibitive time costs, making it a practical solution for real-world visual instruction tuning applications.

# 3.3 Evaluation on the Academic Benchmark

The comparative results of the fine-tuned tasks on the Academic Benchmark are summarized in Table 3. Overall, VisATB achieves the highest $\Delta I\%$ while maintaining a near-minimal $\Delta E\%$ , indicating both enhanced overall performance and mitigated task imbalance. Compared with EW, VisATB delivers substantial gains on Ref-bbox, Ref-caption, ChartQA, and ShareGPT4V, while preserving competitive results on the remaining tasks. These results further

demonstrate the effectiveness of VisATB. Additionally, TLA and all traditional task weighting methods perform worse than EW and VisATB in both $\Delta I\%$ and $\Delta E\%$ , further underscoring the validity of VITW and the limitation of traditional methods.

Generalization on Zero-Shot Tasks. The comparative results of the zero-shot tasks on the Academic Benchmark are presented in Table 4, including single-word QA (TextVQA, POPE, MME), multiple-choice QA (SQA, MMBench, SEED $^1$ ), and open-ended QA (MM-Vet). Overall, VisATB achieves the optimal $\Delta I_{\text{zero}}\%$ . Compared to EW, VisATB exhibits significantly superior performance on multi-choice QA tasks while maintaining comparable results on other tasks. Importantly, the training corpus contains no visual multiple-choice QA-type data at all.

Nevertheless, VisATB substantially enhances the model's zero-shot generalization to such tasks, demonstrating its ability to induce transferable capabilities beyond the supervised domains. Additionally, TLA demonstrates markedly lower values than EW in $\Delta I_{\text{zero}}\%$ , further supporting the validity of VITW.

Visual Analysis of VisATB. The inter-task contributions and intra-task difficulties of VisATB are visualized in Figure 3, and the resulting task weights are detailed in Table 5. As shown in Figure 3a, the heatmap of inter-task contributions demonstrates substantial heterogeneity across task pairs: certain tasks (e.g., Ref-caption and VQAv2) provide strong positive contributions to others, while some (e.g., OCRVQA) receive minimal benefit from others.

Meanwhile, Figure 3b illustrates that intra-task difficulties vary significantly: tasks like grounding and captioning are notably harder to learn compared to simpler tasks such as basic visual QA. This highlights the necessity of considering both inter-task contributions and intra-task difficulties for effective task balancing, and the effectiveness of VisATB to capture these differences across tasks.

WWW '26, April 13-17, 2026, Dubai, United Arab Emirates

Yanqi Dai et al.

![](dt=2026-05-27/ht=14/a139e09e341e417989aadf92c77c4bbc45f2a503ee3d071b6ce2091d8246abb7.jpg)

![](dt=2026-05-27/ht=14/5109c9ce87673723b46229374c1c9447f0ecfe9e128883de8c8c8e02b994cdf0.jpg)

Table 5: The task weights calculated in VisATB on the Academic Benchmark.

![](dt=2026-05-27/ht=14/46d47a7841f2193c31b3dbb686380dbb4e208c45022d4a335b84f47721983f0a.jpg)

<table><tr><td>Task Weights</td><td>ShareGPT4V</td><td>Ref-caption</td><td>VQAv2</td><td>GQA</td><td>ChartQA</td><td>OCRVQA</td><td>Ref-bbox</td></tr><tr><td>λout</td><td>1.0223</td><td>1.0782</td><td>1.0888</td><td>0.9815</td><td>0.8592</td><td>0.9588</td><td>1.0112</td></tr><tr><td>λin</td><td>0.9727</td><td>0.8722</td><td>1.0345</td><td>0.9456</td><td>0.9649</td><td>1.1532</td><td>1.0569</td></tr><tr><td>λD</td><td>0.7195</td><td>1.1361</td><td>0.5598</td><td>0.6666</td><td>1.0920</td><td>0.5794</td><td>2.2466</td></tr><tr><td>λVisATB</td><td>0.8585</td><td>1.0557</td><td>0.8107</td><td>0.8151</td><td>1.0020</td><td>0.8177</td><td>1.6403</td></tr></table>

Table 6: Comparative results on the Academic Benchmark using various pretrained models.

![](dt=2026-05-27/ht=14/9d49962a2bfce5d3c1eaf28daab06b9ad227eb5c842c9b0febc9c6bb3f993843.jpg)

<table><tr><td>Models</td><td>Methods</td><td>ΔI%↑</td><td>ΔE%↓</td><td>ΔIzero%↑</td></tr><tr><td rowspan="2">LLaVA-v1.5-13B</td><td>EW</td><td>4.55</td><td>0.66</td><td>0.00</td></tr><tr><td>VisATB</td><td>6.89</td><td>0.29</td><td>0.38</td></tr><tr><td rowspan="2">Qwen2-VL-2B</td><td>EW</td><td>-2.68</td><td>4.08</td><td>0.00</td></tr><tr><td>VisATB</td><td>-1.32</td><td>3.70</td><td>1.00</td></tr></table>

Model Independence of VisATB. To verify the general effectiveness of our approach independent of specif
ic models, we further fine-tune the pretrained LLaVA-v1.5-13B [31] and Qwen2-VL-2B [50] models on the Academic Benchmark. The overall results are presented in Table 6, with detailed results and settings in Appendix D. Across both scales and architectures, VisATB consistently improves $\Delta I\%$ and $\Delta I_{\mathrm{zero}}\%$ while reducing $\Delta E\%$ compared to EW, demonstrating its robustness to backbone variation.

# 3.4 Evaluation on the Chat Benchmark

The comparative results of the fine-tuned tasks on the Chat Benchmark are presented in Table 7, which includes three loosely structured tasks: general conversation (Conv.), detailed description (Detail.), and complex reasoning (Complex.). This setting reduces overt format conflicts across tasks and serves to assess whether VisATB still offers benefits when structural incompatibilities are minimal. Overall, VisATB outperforms all methods except TLA in $\Delta I\%$ , while maintaining the lowest value in $\Delta E\%$ . Compared to EW, VisATB yields substantial gains on Conv. (+7.4 absolute, 52.2 vs. 44.8) and performs slightly better on Detail. (60.7 vs. 59.6), while sustaining the state-of-the-art Complex. performance (83.4). These results

Table 7: Comparative results on the Chat Benchmark. $\Delta I\%$ and $\Delta E\%$ are the average per-task improvement and error on fine-tuned tasks compared to the STL baseline.

![](dt=2026-05-27/ht=14/46a8c0eb31e6c648467d450aa8fdf5fd1dfff5c08fd5db9df78cdc04c42ea8cc.jpg)

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Conv.↑</td><td rowspan="2">Detail.↑</td><td rowspan="2">Complex.↑</td><td colspan="2">Overall</td></tr><tr><td>ΔI%↑</td><td>ΔE%↓</td></tr><tr><td>STL</td><td>43.6</td><td>49.3</td><td>80.9</td><td></td><td></td></tr><tr><td>EW</td><td>44.8</td><td>59.6</td><td>83.4</td><td>8.91</td><td>0.00</td></tr><tr><td>TLA</td><td>57.2</td><td>61.9</td><td>79.2</td><td>18.22</td><td>0.70</td></tr><tr><td>RLW</td><td>46.0</td><td>56.7</td><td>82.9</td><td>7.66</td><td>0.00</td></tr><tr><td>DWA</td><td>46.1</td><td>65.9</td><td>82.8</td><td>13.92</td><td>0.00</td></tr><tr><td>IGBv1</td><td>39.9</td><td>64.2</td><td>81.7</td><td>7.58</td><td>2.83</td></tr><tr><td>VisATB</td><td>52.2</td><td>60.7</td><td>83.4</td><td>15.31</td><td>0.00</td></tr></table>

indicate that in a chat-style setting, VisATB can still enhance underfitting tasks without compromising the performance of well-performing tasks.

Analysis of TLA and Traditional Methods. TLA achieves the best $\Delta I\%$ but at the cost of an increase in $\Delta E\%$ (0.70 vs. 0.00 of EW). Because the implicit weight introduced by TLA is unregulated, it is unstable across various scenarios and prone to performance imbalance. Additionally, traditional task weighting methods exhibit inferior performance in $\Delta I\%$ , except for DWA, which outperforms EW but still falls short of VisATB. These findings further indicate the necessity of our VITW paradigm and the insufficiency of traditional methods for visual instruction tuning.

# 3.5 Ablation Studies

As presented in Table 8 (with full results provided in Appendix E), we ablate our VisATB on the Academic Benchmark from three perspectives: task weighting strategies, temperatures, and calculation

Adaptive Task Balancing for Visual Instruction Tuning via Inter-Task Contribution and Intra-Task Difficulty

WWW '26, April 13-17, 2026, Dubai, United Arab Emirates

Table 8: Ablation results on the Academic Benchmark. $\alpha = [\alpha_{\mathrm{out}},\alpha_{\mathrm{in}},\alpha_{\mathrm{D}}]$ is the vector of proportional coefficients for the three task weighting strategies; $T$ is the temperature hyperparameter; and 'precise/real Diff' denotes the use of the precise or real calculation approach for intra-task difficulty.

![](dt=2026-05-27/ht=14/830b2a3b77e11f8ec94e688a3b51b5d9b35e0f69dc82a9823787524f90f5a1c9.jpg)

<table><tr><td>Methods</td><td>ΔI%↑</td><td>ΔE%↓</td><td>ΔIzero%↑</td></tr><tr><td>EW</td><td>8.75</td><td>0.10</td><td>0.00</td></tr><tr><td>VisATB (α=[1,0,0])</td><td>9.67</td><td>0.10</td><td>1.36</td></tr><tr><td>VisATB (α=[0,1,0])</td><td>8.45</td><td>0.05</td><td>-3.96</td></tr><tr><td>VisATB (α=[0,0,1])</td><td>12.21</td><td>0.30</td><td>0.63</td></tr><tr><td>VisATB (α=[0.50,0.50,0])</td><td>8.58</td><td>0.06</td><td>2.32</td></tr><tr><td>VisATB (α=[0.33,0.33,0.33])</td><td>8.77</td><td>0.32</td><td>3.04</td></tr><tr><td>VisATB (α=[0.25,0.25,0.50])</td><td>11.29</td><td>0.15</td><td>1.87</td></tr><tr><td>VisATB (T=2.0)</td><td>9.92</td><td>0.12</td><td>0.50</td></tr><tr><td>VisATB (T=1.0)</td><td>10.24</td><td>0.11</td><td>0.52</td></tr><tr><td>VisATB (T=0.5)</td><td>11.29</td><td>0.15</td><td>1.87</td></tr><tr><td>VisATB (precise Diff)</td><td>10.75</td><td>0.16</td><td>2.61</td></tr><tr><td>VisATB (real Diff)</td><td>11.29</td><td>0.15</td><td>1.87</td></tr></table>

approaches for the intra-task difficulty. The compared methods include: EW; VisATB $(\alpha = [\alpha_{\mathrm{out}},\alpha_{\mathrm{in}},\alpha_{\mathrm{D}}])$ , where the proportional coefficients for the three task weighting strategies are set to different values; VisATB $(T = 2.0 / 1.0 / 0.5)$ , where the temperature $T$ is set as 2.0, 1.0 or 0.5; and VisATB (precise/real Diff), where the precise or real calculation approach for intra-task difficulty is used. The precise calculation approach trains the additional models to precisely quantify the intra-task difficulty, as detailed in Appendix F, while the real calculation approach repurposes the models trained for inter-task contribution balancing to reduce the time cost.

Task Weighting Strategies. We first isolate each strategy component of VisATB. As mentioned in Section 2.4, $\lambda_{\mathrm{out}}$ and $\lambda_{\mathrm{D}}$ focus on improving overall performance, while $\lambda_{\mathrm{in}}$ aims to mitigate performance imbalance. Compared to EW, activating only the $\lambda_{\mathrm{out}}$ weight (VisATB $(\alpha = [1,0,0])$ ) raises $\Delta I\%$ from 8.75 to 9.67 and improves $\Delta I_{\mathrm{zero}}\%$ to 1.36, while keeping $\Delta E\%$ unchanged at 0.10.

Utilizing only the $\lambda_{\mathrm{in}}$ weight (VisATB $(\alpha = [0,1,0])$ ) effectively reduces $\Delta E\%$ to 0.05, albeit with a slight drop in $\Delta I\%$ and $\Delta I_{\mathrm{zero}}\%$ . Relying solely on the $\lambda_{\mathrm{D}}$ weight (VisATB $(\alpha = [0,0,1])$ ) produces the largest boost in $\Delta I\% = 12.21$ and a moderate improvement in $\Delta I_{\mathrm{zero}}\%$ but at the cost of a slight increase in $\Delta E\%$ . These underscore the effectiveness of all three task weighting strategies, each focusing on distinct yet complementary aspects.

When combining these task weighting strategies, the proportional coefficients can be adjusted to achieve more favorable Pareto trade-offs. Specifically, VisATB $(\alpha = [0.50, 0.50, 0])$ integrates two inter-task contribution balancing strategies, resulting in a balance between $\Delta I\%$ and $\Delta E\%$ , while also enhancing $\Delta I_{\mathrm{zero}}\%$ . Moreover, VisATB $(\alpha = [0.33, 0.33, 0.33])$ combines all three strategies, leading to improvements in $\Delta I\%$ and $\Delta I_{\mathrm{zero}}\%$ .

To further enhance overall performance, VisATB $(\alpha = [0.25, 0.25, 0.50])$ slightly increases the value of $\alpha_{\mathrm{D}}$ , achieving significantly higher $\Delta I\%$ and $\Delta I_{\mathrm{zero}}\%$ than EW, while also maintaining nearly the lowest $\Delta E\%$ . Notably, we observe that combining strategies generally leads to superior performance

in $\Delta I_{\mathrm{zero}}\%$ than employing any single strategy. This finding demonstrates the effectiveness of comprehensively considering multiple aspects of task balancing for generalization to unseen zero-shot tasks.

Temperatures. VisATB increasingl
y outperforms EW in both $\Delta I\%$ and $\Delta I_{\mathrm{zero}}\%$ as $T$ decreases, while exhibiting a slightly higher $\Delta E\%$ at lower values of $T$ . As the temperature in the softmax functions of Equations 4, 5, and 7 decreases, the weight distribution becomes progressively sharper. If the sharpness in task weight is excessively high, tasks with too small weights may inevitably underperform, leading to a slight performance imbalance. In practice, we recommend using the lowest temperature $T$ that ensures all task weights remain within the range of 0.5 to 2.0 to avoid over-balancing.

Calculation Approaches for Intra-Task Difficulty. As discussed in Section 2.3, the objective of the real approach is to reduce the time cost with minimal error. VisATB (precise Diff) and VisATB (real Diff) exhibit comparable performance levels, with VisATB (real Diff) even showing a slight advantage in $\Delta I\%$ and $\Delta E\%$ . Meanwhile, the real calculation approach enables a reduction of approximately $(R_{\text{large}} + R_{\text{mini}})$ times the training duration of the final model, where $R_{\text{large}}$ and $R_{\text{mini}}$ represent the sampling rates of large enough subsets and mini subsets, respectively. This observation underscores the efficacy of our real calculation approach.

# 4 Related Work

Visual Instruction Tuning. Instruction tuning is first proposed in NLP, enabling LLMs to follow textual instructions and accomplish various tasks [51]. Moreover, to extend the powerful abilities of LLMs into the multimodal domain, Liu et al. [33] introduce visual instruction tuning. This innovative technique integrates LLMs with visual encoders using visual instruction-following data and alignment modules.

Subsequently, a series of improved approaches demonstrate robust performance on visual tasks, respectively focusing on model structures [5, 8, 9, 14, 27, 44, 59], training settings [31, 53], and training data [7, 24, 49, 56, 57]. Particularly, to mitigate visual task conflicts, Dai et al. [9] adaptively adjust sampling probabilities based on task data sizes, and several recent studies design the mixture of LoRA experts structure [8, 14, 44]. In this paper, we propose tackling this challenge from an alternative perspective by adaptive task weighting.

Task Weighting. Adaptive task weighting is commonly employed in CV, which assigns task weights based on losses or gradients to balance the joint training process of tasks [3, 6, 20, 30, 34, 41, 43]. For example, Lin et al. [28] assign task weight randomly; Liu et al. [35] prefer tasks with lower loss decline rates; and Dai et al. [10] favor tasks with higher improvable gaps.

# 5 Conclusion

In this paper, we introduce an Adaptive Task Balancing approach for visual instruction tuning (VisATB). Specifically, we design a token-level Visual Instruction Task Weighting (VITW) paradigm. Building upon this paradigm, we analyze two crucial dimensions for visual task balancing: inter-task contribution and intra-task difficulty. Accordingly, we propose three distinct yet complementary task weighting strategies. Extensive experiments demonstrate that VisATB outperforms existing methods, achieving a more robust and balanced overall performance.

WWW '26, April 13-17, 2026, Dubai, United Arab Emirates

Yanqi Dai et al.

# Acknowledgments

This work was supported in part by National Natural Science Foundation of China (62376274, 62437002).

# References

Adaptive Task Balancing for Visual Instruction Tuning via Inter-Task Contribution and Intra-Task Difficulty

WWW '26, April 13-17, 2026, Dubai, United Arab Emirates

Table 9: Training data sizes and response format instructions for the fine-tuned tasks on the Academic Benchmark. The total training data size is $475\mathrm{k}$ .

![](dt=2026-05-27/ht=14/04a8b99c2503d47bef1c4364faab8664a14e847636f8ea55030880c4cfa1c8e3.jpg)

<table><tr><td>Tasks</td><td>Sizes</td><td>Response Format Instructions</td></tr><tr><td>ShareGPT</td><td>41k</td><td>-</td></tr><tr><td>ShareGPT4V</td><td>98k</td><td></td></tr><tr><td>Ref-caption</td><td>41k</td><td>Provide a short description for this region.</td></tr><tr><td>VQAv2</td><td>83k</td><td rowspan="2">Answer the question using a single word or phrase.</td></tr><tr><td>GQA</td><td>72k</td></tr><tr><td>ChartQA</td><td>18k</td><td></td></tr><tr><td>OCRVQA</td><td>80k</td><td></td></tr><tr><td>Ref-bbox</td><td>41k</td><td>Provide the bounding box coordinate of the region this sentence describes.</td></tr></table>

Table 10: Response format instructions for the zero-shot test tasks on the Academic Benchmark.

![](dt=2026-05-27/ht=14/7a23d0d9c0d1daa1bfdf56cd499e3b9efd1b5834578633536ab9620698924d08.jpg)

<table><tr><td>Tasks</td><td>Response Format Instructions</td></tr><tr><td>TextVQA</td><td rowspan="3">Answer the question using a single word or phrase.</td></tr><tr><td>POPE</td></tr><tr><td>MME</td></tr><tr><td>SQAI</td><td rowspan="3">Answer with the option&#x27;s letter from the given choices directly.</td></tr><tr><td>MMBench</td></tr><tr><td>SEEDI</td></tr><tr><td>MM-Vet</td><td>-</td></tr><tr><td>LLaVA-Bench</td><td></td></tr></table>

# A Task Information and Data Preparation

We train and evaluate LMMs on the following benchmarks:

$\mathbf{M}^3\mathrm{IT}$ Benchmark. The tasks in the $\mathbf{M}^3\mathrm{IT}$ Benchmark are carefully selected from the $\mathbf{M}^3\mathrm{IT}$ dataset [25], a large-scale multimodal instruction tuning dataset. All curated tasks have training and validation sets. If no test set is provided, the validation set is randomly divided into two equal parts, one for validation and the other for testing. Following the task clustering of Li et al. [25], the tasks are categorized into five distinct groups. The task grouping and task list in the group are shown as follows:

Chat Benchmark. The tasks in the Chat Benchmark are introduced by Liu et al. [33], including general conversation (Conv.), detailed description (Detail.), and complex reasoning (Complex.). We utilize LLaVA-Bench-COCO for validation and LLaVA $^{\text{W}}$ for testing.

Academic Benchmark. The tasks in the Academic Benchmark encompass ShareGPT [1], ShareGPT4V [7], Ref-caption, Ref-bbox [19, 38], VQAv2 [15], GQA [16], ChartQA [39], and OCRVQA [40]. Among these, the Ref-caption task involves generating captions for image regions defined by bounding boxes, while the Ref-bbox task aims to predict the bounding boxes corresponding to the described image regions. The testB set of Kazemzadeh et al. [19], the test-dev set of Goyal et al. [15], the test-dev-balanced set of Hudson and Manning [16], and the test sets of other tasks are used for testing their corresponding tasks. The weight of ShareGPT is set as 1.0.

Furthermore, we present 7 zero-shot metrics employed by Liu et al. [31] to evaluate the generalization of methods: TextVQA [46], POPE [26], MME [13], ScienceQA (SQA) [37], MMBench [36], SEED-Bench-IMG (SEED $^{I}$ ) [22], and MM-Vet [55].

The training data sizes and response format instructions for fine-tuned tasks are depicted in Table 9, while the response format instructions for zero-shot test tasks are presented in Table 10. Moreover, we employ multiple data processing and splitting strategies to reduce the computational cost and ensure the evaluation reliability:

(1) For ShareGPT4V, data is randomly partitioned, with 2k allocated to a validation set, 2k to a test set, and the remainder reserved for training.

WWW '26, April 13-17, 2026, Dubai, United Arab Emirates

Yanqi Dai et al.

Table 11: Complete results of the fine-tuned tasks on the Academic Benchmark using various pretrained models. $\Delta I\%$ and $\Delta E\%$ are the average per-task improvement and error on fine-tuned tasks compared to the STL baseline.

![](image)
l_text/v0/result=success/type=image/dt=2026-05-27/ht=14//8d092cebdc5383e2ed80b20287c1106381489dee3a83f63ee400f03dc16c62a5.jpg)

<table><tr><td rowspan="2">Models</td><td rowspan="2">Methods</td><td rowspan="2">ShareGPT4V
CIDEr↑</td><td rowspan="2">Ref-caption
CIDEr↑</td><td rowspan="2">VQAv2
EM↑</td><td rowspan="2">GQA
EM↑</td><td rowspan="2">ChartQA
EM↑</td><td rowspan="2">OCRVQA
EM↑</td><td rowspan="2">Ref-bbox
IoU↑</td><td colspan="2">Overall</td></tr><tr><td>ΔI%↑</td><td>ΔE%↓</td></tr><tr><td rowspan="3">LLaVA-v1.5-13B</td><td>STL</td><td>0.1407</td><td>0.5329</td><td>78.78</td><td>62.54</td><td>18.96</td><td>69.86</td><td>65.50</td><td></td><td></td></tr><tr><td>EW</td><td>0.1351</td><td>0.6008</td><td>79.51</td><td>63.05</td><td>22.28</td><td>69.41</td><td>68.44</td><td>4.55</td><td>0.66</td></tr><tr><td>VisATB</td><td>0.1391</td><td>0.6224</td><td>79.33</td><td>63.00</td><td>22.76</td><td>69.22</td><td>73.35</td><td>6.89</td><td>0.29</td></tr><tr><td rowspan="3">Qwen2-VL-2B</td><td>STL</td><td>0.1370</td><td>0.8402</td><td>81.11</td><td>64.55</td><td>64.36</td><td>73.40</td><td>27.89</td><td></td><td></td></tr><tr><td>EW</td><td>0.1305</td><td>0.6658</td><td>80.22</td><td>64.41</td><td>63.36</td><td>73.28</td><td>30.62</td><td>-2.68</td><td>4.08</td></tr><tr><td>VisATB</td><td>0.1480</td><td>0.6456</td><td>80.24</td><td>64.23</td><td>64.04</td><td>72.93</td><td>30.46</td><td>-1.23</td><td>3.70</td></tr></table>

Table 12: Complete results of the zero-shot tasks on the Academic Benchmark using various pretrained models. $\Delta I_{\mathrm{zero}}\%$ is the average per-task improvement in test performance on zero-shot tasks compared to the EW method.

![](dt=2026-05-27/ht=14/2cf4c97c62220d8b18019d9c5ea1297c98f81f6b7a59d4bdd88b5a0ddd1bffba.jpg)

<table><tr><td>Models</td><td>Methods</td><td>TextVQA↑</td><td>POPE↑</td><td>MME↑</td><td>SQA↑</td><td>MMBench↑</td><td>SEED1↑</td><td>MM-Vet↑</td><td>Overall (ΔIzero%↑)</td></tr><tr><td rowspan="2">LLaVA-v1.5-13B</td><td>EW</td><td>57.52</td><td>86.50</td><td>1556.48</td><td>64.18</td><td>58.25</td><td>64.56</td><td>35.40</td><td>0.00</td></tr><tr><td>VisATB</td><td>57.06</td><td>86.47</td><td>1567.57</td><td>65.90</td><td>57.65</td><td>64.58</td><td>35.80</td><td>0.38</td></tr><tr><td rowspan="2">Qwen2-VL-2B</td><td>EW</td><td>73.45</td><td>88.41</td><td>1438.46</td><td>52.49</td><td>70.45</td><td>74.08</td><td>35.70</td><td>0.00</td></tr><tr><td>VisATB</td><td>72.74</td><td>88.41</td><td>1445.53</td><td>64.35</td><td>69.85</td><td>73.69</td><td>30.80</td><td>1.00</td></tr></table>

![](dt=2026-05-27/ht=14/003d33815684bb9d94da61143b1f13e84141bb699e9775a6bc10527c163ab4f3.jpg)

# B The Sampling Rate for Mini Subsets

As discussed in Section 2.2, mini subsets enable the model to understand the instruction demands of all tasks. Due to the remarkable few-shot learning capabilities of large models, incorporating a small amount of training data can significantly enhance the model's adherence to the task instructions. In our experiments on the Academic Benchmark, the mini subset from each task is obtained by randomly

sampling 1/32nd of the entire dataset from that task. As shown in Figure 4, this sampling ratio is informed by the decline pattern of training loss, which exhibits a rapid decrease during the initial $100+$ steps, followed by a gradual reduction until reaching the final $3,700+$ steps. This pattern suggests that the model effectively grasps the instruction demands in the initial phase, with the subsequent phase dedicated mainly to acquiring the knowledge embedded within the data. Practically, we recommend ensuring that the mini subset from each task contains at least 1,000 data points, and the total number of training steps for all mini subsets exceeds 100.

# C The Simpler Form of VisATB in the Chat Benchmark

In the Chat Benchmark, there are no specific constraints on the output format. Consequently, only the entire datasets of tasks are required, without the need for mini subsets, simplifying the form of VisATB. Specifically, the inter-task contribution of Task $i$ to Task $j$ can be calculated as:

$$
C _ {i \rightarrow j} = \frac {V _ {j} (i) - V _ {j} (\text {b a s e})}{V _ {j} (j) - V _ {j} (\text {b a s e})}, \tag {10}
$$

where $V_{j}(i)$ represents the validation performance on Task $j$ of a model trained on Task $i$ , $V_{j}(j)$ denotes the validation performance on Task $j$ of a model trained on Task $j$ itself, and $V_{j}$ (base) signifies the validation performance on Task $j$ of a pretrained base model. Additionally, the intra-task difficulty of Task $i$ can be computed as:

$$
D _ {i} = 1 - \frac {V _ {i} (\operatorname* {m i n i} _ {i})}{V _ {i} (i)}, \tag {11}
$$

where $V_{i}(\mathrm{mini}_{i})$ denotes the validation performance on Task $i$ of a model trained on the mini subset of Task $i$ , and $V_{i}(i)$ signifies the validation performance on Task $i$ of a model trained on Task $i$ itself.

Adaptive Task Balancing for Visual Instruction Tuning via Inter-Task Contribution and Intra-Task Difficulty

WWW '26, April 13-17, 2026, Dubai, United Arab Emirates

Table 13: Detailed results of ablation studies for the fine-tuned tasks on the Academic Benchmark.

![](dt=2026-05-27/ht=14/eb322f219221393fd80822b0f63a58d7d1ac996b8572262aa1d97dc0895137fb.jpg)

<table><tr><td rowspan="2">Methods</td><td rowspan="2">ShareGPT4V
CIDEr↑</td><td rowspan="2">Ref-caption
CIDEr↑</td><td rowspan="2">VQAv2
EM↑</td><td rowspan="2">GQA
EM↑</td><td rowspan="2">ChartQA
EM↑</td><td rowspan="2">OCRVQA
EM↑</td><td rowspan="2">Ref-bbox
IoU↑</td><td colspan="2">Overall</td></tr><tr><td>ΔI%↑</td><td>ΔE%↓</td></tr><tr><td>EW</td><td>0.1411</td><td>0.5591</td><td>78.27</td><td>62.20</td><td>19.60</td><td>67.73</td><td>61.63</td><td>8.75</td><td>0.10</td></tr><tr><td>VisATB (α=[1,0,0])</td><td>0.1448</td><td>0.5763</td><td>78.34</td><td>62.16</td><td>19.52</td><td>67.72</td><td>61.81</td><td>9.67</td><td>0.10</td></tr><tr><td>VisATB (α=[0,1,0])</td><td>0.1333</td><td>0.5520</td><td>78.25</td><td>62.12</td><td>20.20</td><td>68.00</td><td>62.61</td><td>8.45</td><td>0.05</td></tr><tr><td>VisATB (α=[0,0,1])</td><td>0.1455</td><td>0.5706</td><td>77.46</td><td>61.30</td><td>20.08</td><td>67.04</td><td>71.52</td><td>12.21</td><td>0.30</td></tr><tr><td>VisATB (α=[0.50,0.50,0])</td><td>0.1340</td><td>0.5626</td><td>78.39</td><td>62.27</td><td>20.04</td><td>67.92</td><td>61.92</td><td>8.58</td><td>0.06</td></tr><tr><td>VisATB (α=[0.33,0.33,0.33])</td><td>0.1321</td><td>0.5591</td><td>78.12</td><td>62.08</td><td>19.68</td><td>66.68</td><td>66.08</td><td>8.77</td><td>0.32</td></tr><tr><td>VisATB (α=[0.25,0.25,0.50])</td><td>0.1437</td><td>0.5724</td><td>77.99</td><td>61.81</td><td>20.16</td><td>67.48</td><td>67.38</td><td>11.29</td><td>0.15</td></tr><tr><td>VisATB (T=2.0)</td><td>0.1433</td><td>0.5642</td><td>78.10</td><td>62.08</td><td>20.16</td><td>67.67</td><td>63.06</td><td>9.92</td><td>0.12</td></tr><tr><td>VisATB (T=1.0)</td><td>0.1369</td><td>0.5752</td><td>78.15</td><td>62.09</td><td>20.32</td><td>67.71</td><td>65.00</td><td>10.24</td><td>0.11</td></tr><tr><td>VisATB (T=0.5)</td><td>0.1437</td><td>0.5724</td><td>77.99</td><td>61.81</td><td>20.16</td><td>67.48</td><td>67.38</td><td>11.29</td><td>0.15</td></tr><tr><td>VisATB (precise Diff)</td><td>0.1345</td><td>0.5604</td><td>78.00</td><td>61.87</td><td>21.04</td><td>67.46</td><td>67.83</td><td>10.75</td><td>0.16</td></tr><tr><td>VisATB (real Diff)</td><td>0.1437</td><td>0.5724</td><td>77.99</td><td>61.81</td><td>20.16</td><td>67.48</td><td>67.38</td><td>11.29</td><td>0.15</td></tr></table>

Table 14: Detailed results of ablation studies for the zero-shot tasks on the Academic Benchmark.

![](image)
/mineru_full_text/v0/result=success/type=image/dt=2026-05-27/ht=14//5531fc6b1f2bdc0b9ae46cabc391cc24c44a3d2a714fa6eb9355b3a2ef581201.jpg)

<table><tr><td>Methods</td><td>TextVQA↑</td><td>POPE↑</td><td>MME↑</td><td>SQA↑</td><td>MMBench↑</td><td>SEED1↑</td><td>MM-Vet↑</td><td>Overall (ΔIzero%↑)</td></tr><tr><td>EW</td><td>53.96</td><td>86.83</td><td>1524.41</td><td>60.74</td><td>48.54</td><td>54.13</td><td>29.00</td><td>0.00</td></tr><tr><td>VisATB (α=[1,0,0])</td><td>54.40</td><td>86.87</td><td>1499.87</td><td>60.86</td><td>51.03</td><td>56.79</td><td>29.00</td><td>1.36</td></tr><tr><td>VisATB (α=[0,1,0])</td><td>54.19</td><td>86.49</td><td>1511.93</td><td>59.59</td><td>39.09</td><td>48.14</td><td>30.60</td><td>-3.96</td></tr><tr><td>VisATB (α=[0,0,1])</td><td>53.87</td><td>86.68</td><td>1487.54</td><td>60.74</td><td>49.91</td><td>55.55</td><td>29.50</td><td>0.63</td></tr><tr><td>VisATB (α=[0.50,0.50,0])</td><td>54.73</td><td>86.53</td><td>1518.02</td><td>60.79</td><td>51.89</td><td>57.83</td><td>29.50</td><td>2.32</td></tr><tr><td>VisATB (α=[0.33,0.33,0.33])</td><td>54.07</td><td>86.47</td><td>1514.76</td><td>61.26</td><td>52.92</td><td>57.39</td><td>30.80</td><td>3.04</td></tr><tr><td>VisATB (α=[0.25,0.25,0.50])</td><td>53.87</td><td>86.73</td><td>1501.86</td><td>61.07</td><td>51.46</td><td>58.05</td><td>29.30</td><td>1.87</td></tr><tr><td>VisATB (T=2.0)</td><td>54.07</td><td>86.66</td><td>1498.19</td><td>61.02</td><td>49.66</td><td>54.89</td><td>29.30</td><td>0.50</td></tr><tr><td>VisATB (T=1.0)</td><td>54.13</td><td>86.45</td><td>1499.72</td><td>60.29</td><td>49.48</td><td>56.21</td><td>29.10</td><td>0.52</td></tr><tr><td>VisATB (T=0.5)</td><td>53.87</td><td>86.73</td><td>1501.86</td><td>61.07</td><td>51.46</td><td>58.05</td><td>29.30</td><td>1.87</td></tr><tr><td>VisATB (precise Diff)</td><td>54.25</td><td>86.77</td><td>1504.47</td><td>60.83</td><td>51.72</td><td>57.47</td><td>30.80</td><td>2.61</td></tr><tr><td>VisATB (real Diff)</td><td>53.87</td><td>86.73</td><td>1501.86</td><td>61.07</td><td>51.46</td><td>58.05</td><td>29.30</td><td>1.87</td></tr></table>

# D Detailed Results and Settings on Various Pretrained Models

The detailed results on the Academic Benchmark using various pretrained models are shown in Tables 11 and 12. The temperature $T$ is set as 0.5 on LLaVA-v1.5-13B and 1.0 on Qwen2-VL-2B.

# E Detailed Ablation Results

The detailed ablation results on the Academic Benchmark are presented in Tables 13 and 14.

# F The Precise Calculation Approach for Intra-Task Difficulty

In the precise calculation approach for intra-task difficulty, the intra-task difficulty of Task $i$ can be calculated as follows:

$$
D _ {i} = 1 - \frac {V _ {i} (\operatorname* {m i n i} _ {i})}{V _ {i} (i)}, \tag {12}
$$

where $V_{i}(\mathrm{min}_{i})$ denotes the validation performance on Task $i$ of a model trained on the mini subset of Task $i$ , and $V_{i}(i)$ signifies the validation performance on Task $i$ of a model trained on Task $i$ itself.

# G The Sampling Rate for Sufficiently Large Subsets

As validated by experimental results in Sections 3.2 and 3.3, 1/4th subsets in the $\mathrm{M}^3\mathrm{IT}$ Benchmark and the entire datasets in the Academic benchmark are sufficient for VisATB to accurately measure inter-task contribution and intra-task difficulty. Specifically, the 1/4th subset of VQA in the $\mathrm{M}^3\mathrm{IT}$ Benchmark contains 14k samples, while the entire dataset of ChartQA in the Academic benchmark comprises 18k samples. Therefore, we recommend that subsets containing more than 10k samples and trained for over 100 steps are sufficiently large to ensure effective training of VisATB.

WWW '26, April 13-17, 2026, Dubai, United Arab Emirates

Yanqi Dai et al.