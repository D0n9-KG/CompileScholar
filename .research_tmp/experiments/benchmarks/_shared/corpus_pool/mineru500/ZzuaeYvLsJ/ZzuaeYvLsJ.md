# Deep Fuzzy Multi-view Learning for Reliable Classification

Siyuan Duan $^{1}$ Yuan Sun $^{1}$ Dezhong Peng $^{123}$ Guiduo Duan $^{4}$ Xi Peng $^{1}$ Peng Hu $^{1}$

# Abstract

Multi-view learning methods primarily focus on enhancing decision accuracy but often neglect the uncertainty arising from the intrinsic drawbacks of data, such as noise, conflicts, etc. To address this issue, several trusted multi-view learning approaches based on the Evidential Theory have been proposed to capture uncertainty in multi-view data. However, their performance is highly sensitive to conflicting views, and their uncertainty estimates, which depend on the total evidence and the number of categories, often underestimate uncertainty for conflicting multi-view instances due to the neglect of inherent conflicts between belief masses. To accurately classify conflicting multi-view instances and precisely estimate their intrinsic uncertainty, we present a novel Deep Fuzzy Multi-View Learning (FUML) method. Specifically, FUML leverages Fuzzy Set Theory to model the outputs of a classification neural network as fuzzy memberships, incorporating both possibility and necessity measures to quantify category credibility. A tailored loss function is then proposed to optimize the category credibility. To further enhance uncertainty estimation, we propose an entropy-based uncertainty estimation method leveraging category credibility. Additionally, we develop a Dual Reliable Multiview Fusion (DRF) strategy that accounts for both view-specific uncertainty and inter-view conflict to mitigate the influence of conflicting views in multi-view fusion. Extensive experiments demonstrate that our FUML achieves state-of-the-art performance in terms of both accuracy and reliability.

![](images/37b183fc3ec9cc15d8ce307c591cf8f80e159deda051791364077245f65261fd.jpg)

<details>
<summary>text_image</summary>

RGB view
Text view
...This
Bathroom is
really clean
...
Conflicting
Depth view
</details>

(a) Visualization of the conflicting multi-view instance

![](images/7f8d268a5aee13a43e396474c0df81cca793fa8b22f363131dd967c8ec23cb0b.jpg)

<details>
<summary>bar</summary>

| Room     | Belief mass | Uncertainty |
| -------- | ----------- | ----------- |
| Bedroom  | 0.8         | 0.2         |
| Bathroom | 0.6         | 0.1         |
</details>

(b) EDL-based TMVC

![](images/8fd3308964ac5fd4aca5f382d89a5bb14428cd6535d763aa58fa1cc84d203838.jpg)

<details>
<summary>bar</summary>

| Room | Membership | Uncertainty |
|---|---|---|
| Bedroom | 0.6 | 0.1 |
| Bathroom | 0.9 | 0.8 |
</details>

(c) Ours   
Figure 1. (a) Visualization of the conflicting multi-view instance: the depth view is related to the “Bedroom” category, while the other views show conflicting information, such as “Bathroom.” (b) EDL-based TMVC methods are sensitive to such conflicting multi-view instances. On one hand, because they neglect the global conflict between views in multi-view fusion, classification errors are often made. On the other hand, their uncertainty estimation is only related to the total evidence and the number of categories. For conflicting multi-view instances, as long as the total evidence is large, the uncertainty is seriously underestimated. (c) In our method, both global conflict and uncertainty are considered during fusion, allowing the conflicting multi-view instances to be classified correctly. Additionally, this method can estimate decision uncertainty more accurately.

# 1. Introduction

Multi-view/modal data encapsulates comprehensive information from various modalities, sources, and other perspectives (Yan et al., 2021; Qin et al., 2024; He et al., 2024). Multi-view classification (MVC) aims to utilize the consistency and complementary nature of these data to achieve more accurate classification. With the explosive growth of multi-view data in fields such as video surveillance (Wang et al., 2024a), medical detection (Yang et al., 2024; Nasarian et al., 2024), and autonomous driving (Hong et al., 2024), MVC has garnered great attention from both academia and industry in recent years. Despite the promising performance of existing MVC methods (Yang et al., 2019; Han et al., 2022a; Lin et al., 2022; Mittal et al., 2022; Zhang et al.,

2023), these approaches predominantly prioritize classification accuracy while often neglecting decision uncertainty. This could lead to unreliable decisions, limiting their applicability in reliability-critical scenarios.

To address this limitation, a series of trusted multi-view classification (TMVC) methods (Geng et al., 2021; Han et al., 2020; Liu et al., 2022; Xie et al., 2023; Liu et al., 2023; Xu et al., 2024b) based on Evidential Deep Learning (EDL) (Sensoy et al., 2018; Bao et al., 2021) have been proposed to estimate the uncertainty. These methods provide classification predictions alongside the corresponding uncertainty (inverted to reliability). However, they typically assume strict alignment across different views, which is often unrealistic due to noise, misalignment, or conflicting information in real-world scenarios. More intuitively, a case of multi-view user-generated content is illustrated in Figure 1 (a), where the RGB, text, and depth views exhibit conflicting categorical information. Such conflicts pose significant challenges to EDL-based TMVC methods as shown in Figure 1 (b), remarkably degrading their performance. Specifically, these methods face two key issues: i) They heavily rely on Dempster-Shafer theory (DST) (Shafer, 1992) for multi-view fusion, which fails to account for global conflicts among views overly emphasizes dominant evidence (Xiao, 2019b; Shang et al., 2021), often leading to misclassification. ii) Their uncertainty estimation relies solely on total evidence and the number of categories, neglecting inherent conflicts between belief masses, leading to inaccurate uncertainty quantification for conflicting multi-view instances.

To address the aforementioned problems, this paper presents a novel framework, Deep Fuzzy Multi-View Learning (FUML), grounded in Fuzzy Set Theory (Zadeh, 1965), which provides more precise decisions along with the corresponding accurate uncertainty estimates. Fuzzy Set Theory manages inherent uncertainty and fuzziness in data by introducing gradual membership between 0 and 1, allowing a sample to belong to multiple categories to varying degrees, thereby enabling effective uncertainty quantization and modeling. Based on this principle, as shown in Figure 2, we model the outputs of the classification network as fuzzy memberships corresponding to each category, representing the extent to which a sample belongs to each category. However, memberships alone provide only a possibility measure, i.e., the likelihood of belonging to a category, without indicating the between-class relationship (called necessity) that the sample does not belong to other categories. To integrate both aspects, we introduce category credibility, optimized via the proposed category credibility learning loss. Furthermore, we propose an entropy-based uncertainty estimation method that leverages category credibility to enhance uncertainty quantification. To mitigate the impact of conflicting views in multi-view fusion, we develop a Dual-reliable Multi-view Fusion (DRF) strategy, which considers both view-specific decision uncertainty and inter-view conflicts. Unlike existing uncertainty-aware fusion techniques in EDL-based TMVC that operate sequentially, our DRF employs a global one-time fusion strategy that reduces the influence of high-uncertainty and high-conflict views, ensuring more robust multi-view classification. The main contributions of this work are summarized as follows:

- We reveal and address the conflict sensitivity issue in existing EDL-based TMVC methods, proposing FUML, a novel framework based on Fuzzy Set Theory for enhancing classification and uncertainty estimation.   
- We develop a Dual-reliable Multi-view Fusion (DRF) strategy that effectively reduces the impact of conflicting views, embracing more robust multi-view classification.   
- We propose an entropy-based uncertainty qualification technique, enabling more accurate uncertainty estimation for conflicting multi-view instances.   
- We conduct extensive experiments comparing our FUML against 13 state-of-the-art MVC baselines on eight widely-used benchmarks, demonstrating superior accuracy, reliability, and robustness.

# 2. Related Work

Multi-view Learning. Studies have demonstrated that multi-view learning (MVL) significantly enhances performance across various tasks. Among them, the CCA-based multi-view methods are representative (Chaudhuri et al., 2009; Rupnik & Shawe-Taylor, 2010). With the advancements in deep learning (Yan et al., 2020), some deep MVL methods (Han et al., 2022a; Lin et al., 2022; Zhang et al., 2023; Cao et al., 2024; Qu et al., 2024; Wang et al., 2024b; Chen et al., 2024; Bi & Dornaika, 2024; Xu et al., 2025) emerged. However, most of them focus on improving accuracy, ignoring reliability, limiting their applicability in reliability-critical domains. To achieve reliable decisions, a range of trusted multi-view classification (TMVC) methods (Geng et al., 2021; Han et al., 2020; Liu et al., 2022; 2023; Xie et al., 2023; Liu et al., 2023; Xu et al., 2024a;b; Wang et al., 2024c; Yue et al., 2025; Wang et al., 2025) based on Evidential Deep Learning (EDL) (Sensoy et al., 2018; Gao et al., 2024) and Dempster-Shafer theory (DST) (Shafer, 1992) are proposed. Among them, Trusted Multi-view Classification (TMC) (Han et al., 2020) and Enhanced TMC (ETMC) (Han et al., 2022b) assume that multi-view data is complete; they dynamically integrate different views at the evidence level. However, the arbitrary view missing is widely present in practice. To solve this, Uncertainty-induced Incomplete Multi-View Data Classification (UIMC) (Xie et al., 2023) is proposed. Neverthe

less, UIMC assumes views are strictly aligned, while multiview data often contains low-quality conflicting instances. To address this, Evidential Conflictive Multiview Learning (ECML) (Xu et al., 2024a) designs a conflict opinion aggregation strategy and achieves reliable results for conflicting instances. However, ECML overly relies on the latter combined views, making its final decision vulnerable to conflicting views. Given this, we suggest jumping out of the Evidence Theory, and re-examining TMVC based on the Fuzzy Set Theory (Zadeh, 1965; Liu & Liu, 2010) to achieve a more accurate and robust TMVC.

Uncertainty-aware Deep Learning. Although deep learning has achieved great success in many tasks, it is difficult to provide reliable uncertainty estimates, which is crucial for reliable models (Wen et al., 2023; Chen et al., 2023). To solve this, early works endowed Deep Neural Networks (DNNs) with uncertainty by using distributions instead of deterministic weight parameters (Gal & Ghahramani, 2015; Molchanov et al., 2017; Lakshminarayanan et al., 2017), but they often suffer from high computational costs. The recently proposed test-time augmentation (Lyzhov et al., 2020) estimates uncertainty at test time, but it still needs multiple inferences. In contrast, Evidential Deep Learning (EDL) (Sensoy et al., 2018; Qin et al., 2022; Li et al., 2025) directly infers uncertainty from network outputs. Recently, researchers have extended EDL to the field of multi-view learning and pioneered a series of methods (Geng et al., 2021; Han et al., 2020; Liu et al., 2022; Xie et al., 2023; Liu et al., 2023; Xu et al., 2024a;b). Although these methods achieve promising uncertainty estimates, their uncertainty depends only on the number of categories and the total evidence, and the uncertainty of conflicting multi-view instance is often underestimated. In this paper, we draw on the Fuzzy Set Theory (Zadeh, 1965; Liu & Liu, 2010), which provides a more nuanced perspective that combines possibility and necessity measures to capture uncertainty more accurately.

# 3. The Proposed Method

# 3.1. Problem Definition

For a clear presentation, we first introduce the following notations. Given N training inputs $\{X_{n}\}_{n=1}^{N}$ with V views, i.e., $X = \{x^{v}\}_{v=1}^{V}$ , and the corresponding labels $\{y_{n}\}_{n=1}^{N}$ . The goal of trusted multi-view classification is to learn a model $f : \{x^{v}\}_{v=1}^{V} \to y$ that accurately predicts the label for an unseen sample by effectively integrating information from all available views and provide the corresponding decision uncertainty. The main challenge lies in utilizing the consistent and complementary information from each view while managing conflicting views, ultimately improving overall classification performance and providing accurate uncertainty estimation.

# 3.2. Deep Fuzzy Multi-view Learning

# 3.2.1. CATEGORY CREDIBILITY MODELING

The core idea of Fuzzy Set Theory is to allow samples to belong to a set to a certain degree, rather than either absolutely belonging or not belonging (Zadeh, 1965). Based on this, fuzzy systems can effectively handle the uncertainties and ambiguities inherent in real-world data (Das et al., 2020; Wu et al., 2025). According to Fuzzy Set Theory, membership quantifies the degree to which a sample belongs to a fuzzy set. Similarly, the output probabilities of a classification network, ranging from 0 to 1, represent the possibility of a sample belonging to each category, with higher values indicating a greater possibility of classification into that category. This parallel enables us to model the classifier's prediction for a category as a membership for that category. Therefore, for a given sample $\mathbf{x}_i^v$ , the memberships across all categories can be expressed as $m_{i1}^v, m_{i2}^v, \ldots, m_{iK}^v$ , where $K$ represents the total number of categories.

Nevertheless, the memberships only provide the possibility measure which represents the likelihood of belonging to a category, not the necessity measure—the certainty that the sample does not belong to other categories (Liu & Liu, 2010; Duan et al., 2025). To complement membership by quantifying these certainties, we introduce the concept of necessity:

$$
e _ {i k} ^ {v} = 1 - \max \{m _ {i l} ^ {v} \mid l \neq k \},   k = 1,..., K, \tag {1}
$$

where $\max\{m_{il}^{v} \mid l \neq k\}$ is the highest membership for the other categories $\{l\}_{l \neq k}$ . By integrating both possibility measure and necessity measure, we can obtain more comprehensive information. Therefore, we define the category credibility as the arithmetic mean of the possibility measure and the necessity measure:

Definition 3.1. Let $\mathbf{m}_i^v = [m_{i1}^v,m_{i2}^v,\dots,m_{iK}^v]$ be the vector of memberships of the $i$ -th sample in $v$ -th view, and $\forall m_{ik}^{v}\in [0,1], k = 1,2,\dots,K$ . Then, the category credibility of the $i$ -th sample to the $k$ -th category is defined by

$$
c _ {i k} ^ {v} = \frac {1}{2} (m _ {i k} ^ {v} + 1 - \max \{m _ {i l} ^ {v} \mid l \neq k \}), k = 1, 2,..., K, \tag {2}
$$

which can be arranged as $\mathbf{c}_i^v = [c_{i1}^v,c_{i2}^v,\dots,c_{iK}^v ]\in \mathbb{R}^K$

# 3.2.2. CATEGORY CREDIBILITY LEARNING

To map the logits of a neural network as memberships, first, $L_{p}$ -normalization is applied to the logits to limit the value in the range of [-1,1]. Subsequently, an activation function (i.e., ReLU) is used to yield output values in the range of [0,1]. These values can be modeled as memberships for the corresponding category. To be specific, the calculation formula is as follows:

$$
\mathbf {m} _ {i} ^ {v} = \operatorname{ReLU} \left(\frac {\mathbf {a} _ {i} ^ {v}}{\left| \left| \mathbf {a} _ {i} ^ {v} \right| \right| _ {p}}\right), \tag {3}
$$

![](images/d60e27db4c81ce80a87d7cbabe6cbdc10a173a9a263dfd1ee30a173f77a67a67.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["View 1: x¹"] --> B["f¹(·)"]
    B --> C["m¹"]
    D["View 2: x²"] --> E["f²(·)"]
    E --> F["m²"]
    G["..."]
    H["View V: xV"] --> I["fV(·)"]
    I --> J["mV"]
    K["m¹"] --> L["Conflict"]
    M["m²"] --> L
    N["mV"] --> L
    L --> O["o¹"]
    L --> P["o²"]
    L --> Q["..."]
    L --> R["oV"]
    L --> S["u¹"]
    L --> T["u²"]
    L --> U["uV"]
    O --> V["w¹"]
    P --> V
    Q --> V
    R --> V
    S --> V
    T --> V
    U --> V
    V --> W["m¹"]
    V --> X["m²"]
    V --> Y["mV"]
    style A fill:#f9f,stroke:#333
    style D fill:#f9f,stroke:#333
    style H fill:#f9f,stroke:#333
    style K fill:#ccf,stroke:#333
    style L fill:#cfc,stroke:#333
    style O fill:#fcc,stroke:#333
    style P fill:#fcc,stroke:#333
    style Q fill:#fcc,stroke:#333
    style R fill:#fcc,stroke:#333
    style S fill:#fcc,stroke:#333
    style V fill:#fcc,stroke:#333
    style W fill:#cff,stroke:#333
```
</details>

Figure 2. Illustration of FUML. Firstly, view-specific DNNs ( $\{f^v(\cdot)\}_{v=1}^V$ ) collect memberships ( $\{\mathbf{m}^v\}_{v=1}^V$ ) from multi-view instances ( $\{\mathcal{X}^v\}_{v=1}^V$ ), which could be termed as a possibility for each category. Secondly, the uncertainty of each view ( $\{u^v\}_{v=1}^V$ ) and the conflicts ( $\{o^v\}_{v=1}^V$ ) between views are calculated based on these memberships. Thirdly, the weights $\{w^v\}_{v=1}^V$ of each view could be calculated and are used to aggregate memberships from all views, thereby realizing Dual-reliable Multi-view Fusion (DRF). Finally, the aggregated memberships are used to construct trusted classification results, where the decision uncertainty will increase if aggregated memberships are conflicting.

where $a_{i}^{v}$ denotes the logits of a neural network. The corresponding category credibility $c_{i}^{v}$ can also be derived by Equation (2).

To achieve discriminative learning, each sample should exhibit the highest possible category credibility for its matched category while maintaining the lowest possible category credibility for all other categories. Intuitively, this could be achieved by directly aligning the category credibility $c_{i}^{v}$ with the corresponding one-hot labels $y_{i}^{v}$ , i.e., minimizing mean squared error ( $||c_{i}^{v}-y_{i}^{v}||_{2}$ ) or cross-entropy loss ( $-y_{i}^{v}\cdot\log(c_{i}^{v})-(1-y_{i}^{v})\cdot\log(1-c_{i}^{v})$ ). However, both approaches risk over-optimizing the necessity ( $e_{i}^{v}$ ) for unmatched categories, resulting in the neural network converging to a local optimum. The reason is as follows: When $y_{ik}^{v}=0$ , $m_{ik}^{v}$ would tend to 0, and the necessity $e_{ik}^{v}=1-\max\{m_{il}^{v}\mid l\neq k\}$ would also approach 0, driving $m_{il}^{v}$ towards 1. This is problematic because $m_{il}^{v}$ should approach 0 when $y_{il}^{v}=0$ , rather than 1. To tackle this issue, we propose category credibility learning loss to optimize the category credibility, thereby guiding the model toward the correct optimization:

$$
\mathcal {L} _ {c c l} = \frac {1}{N _ {b}} \sum_ {i = 1} ^ {N _ {b}} - \mathbf {y} _ {i} ^ {v} \cdot \log (\mathbf {r} _ {i} ^ {v}) - (1 - \mathbf {y} _ {i} ^ {v}) \cdot \log (1 - \mathbf {r} _ {i} ^ {v}), \tag {4}
$$

where $N_{b}$ is the batch size, $\mathbf{r}_i^v = \phi^{tr}(\mathbf{m}_i^v,\mathbf{y}_i^v) = [r_{i1}^v,r_{i2}^v,\dots,r_{iK}^v ]\in \mathbb{R}^K$ represents category credibility during training, and

$$
r _ {i k} ^ {v} = \left\{ \begin{array}{l l} \frac {m _ {i k} ^ {v} + 1 - \max \left\{m _ {i l} ^ {v} \mid l \neq k \right\}}{2}, & \text { if } y _ {i k} ^ {v} = 1, \\ \frac {m _ {i k} ^ {v} + 1 - m _ {i l} ^ {v}}{2}, & \text { if } y _ {i k} ^ {v} = 0, l = \underset {k} {\arg \max} y _ {i k} ^ {v}, \end{array} \right. \tag {5}
$$

where k = 1, 2, ..., K. From Equation (4), it can be seen that this loss function ensure $m_{ik}^{v}$ approaches 1 for matched categories where $y_{ik}^{v}=1$ and approaches 0 for unmatched category where $y_{ik}^{v}=0$ , by leveraging label information. To be specific, when $y_{ik}^{v}=0$ and $y_{il}^{v}=1$ (where l = $\arg\max_{k}y_{ik}^{v}$ ), we expect the membership of the matched category to be greater than that of any unmatched categories after training. For matched categories, the necessity during training should be calculated as $1-\max\{m_{il}^{v}\mid l\neq k\}$ , driving $\max\{m_{il}^{v}\mid l\neq k\}$ toward 0. For unmatched categories, the necessity during training should be calculated as $1-m_{il}^{v}$ , forcing $m_{il}^{v}$ to approach 1. Therefore, this approach effectively guides the necessity and category credibility optimization, avoiding over-optimization and ensuring correct model convergence.

# 3.2.3. CONFLICTIVE MULTI-VIEW FUSION

Environmental factors, such as sensor failure, adverse weather conditions, and data communication issues, often introduce noisy and unaligned views in multi-view data, i.e., create conflicting views (Xiao, 2019a; Zhang et al., 2024). Addressing these issues is essential for enhancing the precision and robustness of multi-view classification. Noisy views generally exhibit high uncertainty, complicating accurate decision-making and potentially leading to erroneous decisions. Therefore, the influence of noisy views should be minimized in multi-view fusion to prioritize the contributions of cleaner views. In contrast, unaligned views tend to generate highly conflicting but low-uncertainty decisions. Reducing their misleading impact on the final decision and limiting their influence in the fusion process is also crucial. Therefore, below, we first define the uncertainty and then the conflict, and finally use them to build a multi-view fusion strategy that can resist conflicting views.

Uncertainty Inference. Although the category credibility reflects the uncertainty of a single predicted category, it fails to capture the uncertainty of the entire prediction outcome. To overcome this limitation, we take into account the category credibility of all categories to calculate decision uncertainty. To be specific, inspired by Shannon's entropy, which measures uncertainty arising from information deficiency, we define uncertainty as follows:

Definition 3.2. Let $c_{i}^{v} = [c_{i1}^{v}, c_{i2}^{v}, ..., c_{iK}^{v}]$ be the vector of category credibility of the i-th sample in v-th view, and $\forall c_{ik}^{v} \in [0, 1]$ , $k = 1, 2, ..., K$ . Then, uncertainty is defined by

$$
\begin{array}{l} u _ {i} ^ {v} = \frac {\sum_ {k = 1} ^ {K} H (c _ {i k} ^ {v})}{K \cdot \ln 2} \\ = \frac {\sum_ {k = 1} ^ {K} - c _ {i k} ^ {v} \cdot l n (c _ {i k} ^ {v}) - (1 - c _ {i k} ^ {v}) \cdot l n (1 - c _ {i k} ^ {v})}{K \cdot \ln 2}, \tag {6} \\ \end{array}
$$

where K is the number of categories and $H(c_{ik}^{v})$ is the entropy of category credibility $c_{ik}^{v}$ . This uncertainty lies within the range [0, 1], where higher values denote greater uncertainty.

Conflict Inference. The above-defined uncertainty does not allow for the assessment of inconsistencies between views. To address this, below, we define conflict:

Definition 3.3. Let $\{m_{i}^{v}\}_{v=1}^{V}$ represent the set of multiview membership vectors for the i-th sample. The conflict of the v-th view relative to other views is defined as:

$$
o _ {i} ^ {v} = \frac {1}{V - 1} \sum_ {j \neq v} ^ {V} \left(1 - \frac {\mathbf {m} _ {i} ^ {v} \cdot \mathbf {m} _ {i} ^ {j}}{| | \mathbf {m} _ {i} ^ {v} | | \cdot | | \mathbf {m} _ {i} ^ {j} | |}\right). \tag {7}
$$

Since the range of all elements in $m_{i}^{v}$ is in the range [0,1], the conflict $o_{i}^{v}$ is in the range [0,1], where higher values denote greater conflict.

Views with relatively high conflict are considered unaligned with other views, whereas views with relatively low conflict are precisely the ones that should participate in the fusion. Next, we utilize view-specific uncertainty and the above conflict between views to propose a Dual-reliable Multiview Fusion, which means both noisy views and unaligned views can be reliably fused.

Dual-reliable Multi-view Fusion (DRF). In multi-view decision-level fusion, we hope to fuse views with low uncertainty and low conflict with other views. Following Definition 3.2 and Definition 3.3, we could fuse the final memberships $m_{i}^{a}$ from different views as follows:

$$
w _ {i} ^ {v} = \frac {g \left(\left(1 - u _ {i} ^ {v}\right) \left(1 - o _ {i} ^ {v}\right)\right)}{\sum_ {v = 1} ^ {V} g \left(\left(1 - u _ {i} ^ {v}\right) \left(1 - o _ {i} ^ {v}\right)\right)}, \tag {8}
$$

$$
\mathbf {m} _ {i} ^ {a} = \sum_ {v = 1} ^ {V} w _ {i} ^ {v} \cdot \mathbf {m} _ {i} ^ {v},
$$

where $g(\cdot)$ is a monotonically increasing function. In this work, we use the exponential function $exp(\cdot)$ as $g(\cdot)$ . Note that, during training, the uncertainty is calculated based on the category credibility during training, i.e., Equation (5). According to DRF, we could get the final memberships of each category and thus infer the overall uncertainty by Equation (6).

# 3.2.4. LOSS FUNCTION

To ensure that all views can simultaneously form reasonable decisions and thus improve the overall performance, we use a multi-task strategy with the following overall loss function:

$$
\mathcal {L} _ {\text { total }} = \mathcal {L} _ {c c l} (\mathbf {r} ^ {a}, \mathbf {y}) + \sum_ {v = 1} ^ {V} \mathcal {L} _ {c c l} (\mathbf {r} ^ {v}, \mathbf {y}). \tag {9}
$$

The pseudo-code of our FUML can be found in the Appendix B.3.

# 3.3. Discussion and Analyses

In this section, we analyze the advantages of FUML, especially the conflicting view fusion. The following propositions provide the theoretical analysis to support the conclusions. The proofs are shown in the Appendix A.

Proposition 3.4. For the i-th multi-view instance, when a clean view 1 with uncertainty $u_{i}^{1}$ is fused with a conflicting view 2 caused by noise with uncertainty $u_{i}^{2}$ , the fused uncertainty $u_{i}^{a} > u_{i}^{1}$ .

Proposition 3.5. For the $i$ -th multi-view instance, when a clean view 1 with uncertainty $u_{i}^{1}$ is fused with a conflicting view 2 caused by misalignment with uncertainty $u_{i}^{2}$ , the fused uncertainty $u_{i}^{a} > u_{i}^{1}$ and $u_{i}^{a} > u_{i}^{2}$ .

Based on Proposition 3.4 and Proposition 3.5, during the multi-view fusion stage, our FUML achieves more accurate uncertainty estimation when fusing with a conflicting view.

Intuitive explanation of the effectiveness of FUML. Without loss of generality, we assume that view $X^{A}$ is clean, view $X^{B}$ is noisy due to unknown environmental factors or sensor failure, and view $X^{C}$ is misaligned with other views for similar reasons. At this time, $X^{A}$ aligns with the distribution of clean training data, whereas view $X^{B}$ deviates significantly from it. Accordingly, we have $u^{A} \leqslant u^{B}$ and $o^{A} \leqslant o^{B}$ , leading to $w^{A} \geqslant w^{B}$ . Therefore, in our FUML framework, the multi-view decision will tend to rely more on the high-quality view $X^{A}$ than on the noisy view $X^{B}$ . In addition, although the uncertainty of view $X^{C}$ is not as high as that of view $X^{B}$ , its relative view $X^{C}$ has a higher conflict with other views. Accordingly, we have $u^{A} \approx u^{C}$ and $o^{A} \leqslant o^{C}$ thus $w^{A} \geqslant w^{C}$ . Therefore, for our method, the multi-view decision will tend to rely more on the low-conflict view $X^{A}$ than the view $X^{C}$ . As for the

weight between views $X^{B}$ and view $X^{C}$ , it is related to the comprehensive consideration of its uncertainty and conflict. By dynamically determining the fusion weights of each view, FUML effectively mitigates the influence of noisy and misaligned views, i.e., conflicting views, embracing robust classification for conflicting multi-view instances.

# 4. Experiments

# 4.1. Experimental Setup

We briefly present the experimental setup here, including the experimental datasets and comparison methods. Please refer to Appendix B for more detailed setup. The code of our FUML is available here $^{1}$ .

Datasets. To validate the effectiveness of the proposed FUML, we conduct experiments on eight public datasets: Handwritten (HW) $^{2}$ , MSRC-V1 (MSRC) (Winn & Jojic, 2005), NUS-WIDE-OBJ (NUSOBJ) $^{3}$ , Fashion-MV (Fashion) (Wang et al., 2023), Scene15 (Scene) $^{4}$ , LandUse (Yang & Newsam, 2010), Leaves100 (Leaves) $^{5}$ , and PIE $^{6}$ . The training set and the test set are split in a ratio of 8:2. The detailed information is shown in Table 1.

To create a test set with conflicting instances, following the methodology outlined in (Xu et al., 2024a), we apply two transformations: (1) We add Gaussian noise with different standard deviations to some test instances. (2) We alter the information in a random view for a subset of instances, making the view's label inconsistent with the true label.

Table 1. A summary of datasets used for evaluation. 

<table><tr><td>DATASET</td><td>SIZE</td><td>CATEGORIES</td><td>DIMENSIONALITY</td></tr><tr><td>HW</td><td>2000</td><td>10</td><td>240; 76; 216; 47; 64; 6</td></tr><tr><td>MSRC</td><td>210</td><td>7</td><td>1302; 48; 512; 100; 256; 210</td></tr><tr><td>NUSOBJ</td><td>30000</td><td>31</td><td>65; 226; 145; 74; 129</td></tr><tr><td>FASHION</td><td>10000</td><td>10</td><td>784; 784; 784</td></tr><tr><td>SCENE</td><td>4485</td><td>15</td><td>20; 59; 40</td></tr><tr><td>LANDUSE</td><td>2100</td><td>21</td><td>20; 59; 40</td></tr><tr><td>LEAVES</td><td>1600</td><td>100</td><td>64; 64; 64</td></tr><tr><td>PIE</td><td>680</td><td>68</td><td>484; 256; 279</td></tr></table>

Evaluation metrics. Owing to the inherent randomness, we report the mean accuracy and standard deviation across 10 different random seeds. Additionally, the improvements over the best-performing baseline are also reported.

Compared methods. For a comprehensive comparison, we adopted the following baselines: (1) The untrusted

baselines, i.e., can't provide decision uncertainty, include: DFTMC (Han et al., 2022a), DCP(CV&CG) (Lin et al., 2022), QMF (Zhang et al., 2023), and PDF (Cao et al., 2024). (2) The trusted baselines include: DUA-Nets (Geng et al., 2021), TMC (Han et al., 2020), ETMC (Han et al., 2022b), TMDL-OA (Liu et al., 2022), UIMC (Xie et al., 2023), ECML (Xu et al., 2024a), TMNR (Xu et al., 2024b), and CCML (Liu et al., 2024).

# 4.2. Comparison with State-of-the-Art Methods

To evaluate the performance of our FUML, we apply multi-view classification on eight datasets over 10 different random seeds. The experimental results of the normal and conflicting test sets are shown in Table 2 and Table 3, respectively. Note that DFTMC does not converge on the NUSOBJ and Fashion datasets, so we can't report its results and mark with ‘-’. The following key observations can be made from these results: (1) On the normal test sets, FUML outperforms all baselines on all datasets. For instance, on the Scene, Leaves, and PIE datasets, FUML achieves an accuracy improvement of 1.62%, 1.59%, and 1.47% compared to the second-best baselines. (2) When the performance is compared on the conflicting test sets, all methods exhibit a noticeable drop in accuracy. However, FUML consistently achieves superior performance, with particularly larger improvements on the Scene, LandUse, and Leaves datasets (4.83%, 7.31%, and 14.60%, respectively). This could be attributed to the proposed dual-reliable multi-view fusion, which improves the robustness to conflicting multi-view instances by reducing the weights of noisy and unaligned views during fusion, as verified by the ablation studies in Section 4.5. More comprehensive conflicting multi-view classification results and analysis can be found in Appendix C.1.

# 4.3. Uncertainty Effectiveness Analysis

To validate the effectiveness of our FUML in estimating uncertainty for conflicting multi-view instances, we compare it with two typical EDL-based TMVC methods, i.e., ETMC and ECML, using the uncertainty density map. The results, shown in Figure 3, reveal the following observations: (1) Compared to the normal test set, the uncertainty of ETMC and ECML barely changed with the addition of conflicting views, and even decreased on the LandUse dataset. However, as can be seen from Table 2 and Table 3, for ETMC and ECML, the addition of conflicting views greatly reduces the classification accuracy. Therefore, their uncertainty estimates for conflicting multi-view instances are inaccurate. (2) In contrast, the uncertainty estimated by FUML is notably higher for conflicting test sets than for normal ones, demonstrating the accuracy of FUML in estimating uncertainty since it can facilitate discrimination between normal and conflicting instances. The corresponding quantitative

Table 2. Accuracy (%) performance on normal test sets. The best and the second-best results are highlighted in boldface and underlined respectively. The means and standard deviations over ten runs are reported. The methods marked with '★' are trusted. 

<table><tr><td>METHODS.</td><td>HW</td><td>MSRC</td><td>NUSOBJ</td><td>FASHION</td><td>SCENE</td><td>LANDUSE</td><td>LEAVES</td><td>PIE</td></tr><tr><td>DFTMC</td><td> $98.75 \pm 0.39$ </td><td> $96.90 \pm 2.14$ </td><td>-</td><td>-</td><td> $63.10 \pm 3.60$ </td><td> $34.95 \pm 1.69$ </td><td> $69.92 \pm 2.54$ </td><td> $91.40 \pm 3.50$ </td></tr><tr><td>DCP-CV</td><td> $98.75 \pm 0.59$ </td><td> $92.86 \pm 2.61$ </td><td> $32.19 \pm 9.48$ </td><td> $97.96 \pm 0.16$ </td><td> $76.70 \pm 2.15$ </td><td> $71.71 \pm 2.09$ </td><td> $95.62 \pm 1.38$ </td><td> $86.32 \pm 4.87$ </td></tr><tr><td>DCP-CG</td><td> $99.00 \pm 0.47$ </td><td> $95.24 \pm 3.69$ </td><td> $43.65 \pm 1.10$ </td><td> $98.11 \pm 0.23$ </td><td> $77.79 \pm 1.73$ </td><td> $75.74 \pm 0.98$ </td><td> $98.19 \pm 0.46$ </td><td> $90.59 \pm 1.99$ </td></tr><tr><td>QMF</td><td> $98.72 \pm 0.48$ </td><td> $97.86 \pm 1.28$ </td><td> $45.41 \pm 0.43$ </td><td> $98.93 \pm 0.32$ </td><td> $68.58 \pm 1.49$ </td><td> $47.86 \pm 2.55$ </td><td> $95.69 \pm 1.25$ </td><td> $92.06 \pm 1.64$ </td></tr><tr><td>PDF</td><td> $98.40 \pm 0.37$ </td><td> $97.14 \pm 1.78$ </td><td> $46.78 \pm 0.33$ </td><td> $98.95 \pm 0.19$ </td><td> $70.25 \pm 1.21$ </td><td> $45.17 \pm 2.66$ </td><td> $98.03 \pm 0.71$ </td><td> $92.57 \pm 1.66$ </td></tr><tr><td> $DUA-NETS^{\star}$ </td><td> $98.10 \pm 0.32$ </td><td> $84.67 \pm 3.03$ </td><td> $27.75 \pm 0.00$ </td><td> $91.08 \pm 0.17$ </td><td> $65.01 \pm 1.55$ </td><td> $45.24 \pm 1.85$ </td><td> $90.31 \pm 1.25$ </td><td> $90.56 \pm 0.47$ </td></tr><tr><td> $TMC^{\star}$ </td><td> $98.51 \pm 0.15$ </td><td> $91.70 \pm 2.70$ </td><td> $38.77 \pm 0.81$ </td><td> $95.40 \pm 0.40$ </td><td> $67.71 \pm 0.30$ </td><td> $31.69 \pm 3.93$ </td><td> $86.81 \pm 2.20$ </td><td> $91.85 \pm 0.23$ </td></tr><tr><td> $ETMC^{\star}$ </td><td> $98.75 \pm 0.00$ </td><td> $92.86 \pm 3.01$ </td><td> $44.23 \pm 0.76$ </td><td> $96.21 \pm 0.36$ </td><td> $71.61 \pm 0.28$ </td><td> $43.52 \pm 3.19$ </td><td> $91.44 \pm 2.39$ </td><td> $93.75 \pm 1.08$ </td></tr><tr><td> $TMDL-OA^{\star}$ </td><td> $98.55 \pm 0.45$ </td><td> $95.00 \pm 1.67$ </td><td> $27.88 \pm 0.67$ </td><td> $86.52 \pm 0.04$ </td><td> $75.57 \pm 0.02$ </td><td> $25.02 \pm 2.10$ </td><td> $75.28 \pm 3.57$ </td><td> $92.33 \pm 0.36$ </td></tr><tr><td> $UIMC^{\star}$ </td><td> $98.25 \pm 0.00$ </td><td> $98.81 \pm 1.19$ </td><td> $43.42 \pm 0.12$ </td><td> $98.13 \pm 0.13$ </td><td> $77.70 \pm 0.00$ </td><td> $57.95 \pm 0.61$ </td><td> $95.31 \pm 0.71$ </td><td> $91.69 \pm 2.16$ </td></tr><tr><td> $ECML^{\star}$ </td><td> $98.72 \pm 0.39$ </td><td> $94.05 \pm 1.60$ </td><td> $42.62 \pm 0.42$ </td><td> $97.93 \pm 0.35$ </td><td> $76.19 \pm 0.12$ </td><td> $60.10 \pm 2.01$ </td><td> $92.53 \pm 1.94$ </td><td> $94.71 \pm 0.02$ </td></tr><tr><td> $TMNR^{\star}$ </td><td> $97.20 \pm 0.63$ </td><td> $94.05 \pm 3.24$ </td><td> $34.52 \pm 0.85$ </td><td> $94.10 \pm 0.50$ </td><td> $68.10 \pm 1.15$ </td><td> $27.38 \pm 1.88$ </td><td> $90.13 \pm 1.53$ </td><td> $89.53 \pm 1.89$ </td></tr><tr><td> $CCML^{\star}$ </td><td> $97.60 \pm 0.62$ </td><td> $96.90 \pm 2.39$ </td><td> $41.43 \pm 0.71$ </td><td> $95.16 \pm 0.41$ </td><td> $73.87 \pm 1.83$ </td><td> $60.86 \pm 1.93$ </td><td> $97.72 \pm 0.92$ </td><td> $93.97 \pm 1.67$ </td></tr><tr><td> $FUML^{\star}$ </td><td> $99.20 \pm 0.36$ </td><td> $99.76 \pm 0.75$ </td><td> $48.23 \pm 0.42$ </td><td> $98.96 \pm 0.25$ </td><td> $79.41 \pm 1.34$ </td><td> $76.71 \pm 0.46$ </td><td> $99.78 \pm 0.27$ </td><td> $96.18 \pm 1.24$ </td></tr><tr><td>IMPROVE</td><td> $\Delta 0.20$ </td><td> $\Delta 0.95$ </td><td> $\Delta 1.45$ </td><td> $\Delta 0.01$ </td><td> $\Delta 1.62$ </td><td> $\Delta 0.97$ </td><td> $\Delta 1.59$ </td><td> $\Delta 1.47$ </td></tr></table>

Table 3. Accuracy (%) performance on conflicting test sets. The best and the second-best results are highlighted in boldface and underlined respectively. The means and standard deviations over ten runs are reported. The methods marked with '★' are trusted. 

<table><tr><td>METHODS</td><td>HW</td><td>MSRC</td><td>NUSOBJ</td><td>FASHION</td><td>SCENE</td><td>LANDUSE</td><td>LEAVES</td><td>PIE</td></tr><tr><td>DFTMC</td><td>53.65±20.07</td><td>60.24±23.45</td><td>-</td><td>-</td><td>36.01±2.78</td><td>7.88±0.94</td><td>1.10±0.12</td><td>3.97±0.82</td></tr><tr><td>DCP-CV</td><td>98.20±0.56</td><td>84.76±7.00</td><td>28.10±7.80</td><td>92.72±2.41</td><td>66.22±2.12</td><td>59.98±1.93</td><td>76.94±1.36</td><td>67.06±2.15</td></tr><tr><td>DCP-CG</td><td>98.70±0.64</td><td>90.00±1.78</td><td>38.61±1.29</td><td>90.38±2.17</td><td>66.44±0.32</td><td>61.83±2.48</td><td>79.06±1.22</td><td>69.56±3.77</td></tr><tr><td>QMF</td><td>97.52±0.86</td><td>95.95±1.52</td><td>42.72±0.67</td><td>92.69±0.78</td><td>59.53±1.63</td><td>40.17±2.67</td><td>77.47±1.46</td><td>82.50±2.81</td></tr><tr><td>PDF</td><td>94.35±1.21</td><td>94.52±3.02</td><td>43.57±0.36</td><td>90.73±0.53</td><td>58.75±1.03</td><td>39.40±1.94</td><td>76.34±1.26</td><td>74.93±2.76</td></tr><tr><td> $DUA-NETS^{\star}$ </td><td>87.16±0.34</td><td>78.57±4.45</td><td>25.64±0.25</td><td>83.03±0.18</td><td>26.18±1.31</td><td>37.22±0.56</td><td>65.62±2.19</td><td>56.45±1.75</td></tr><tr><td> $TMC^{\star}$ </td><td>92.76±0.15</td><td>86.20±4.90</td><td>36.00±0.78</td><td>84.76±0.78</td><td>42.27±1.61</td><td>19.67±1.88</td><td>70.25±2.55</td><td>61.65±1.03</td></tr><tr><td> $ETMC^{\star}$ </td><td>93.85±1.26</td><td>87.14±4.54</td><td>40.45±0.81</td><td>86.48±1.05</td><td>56.90±1.70</td><td>36.05±2.50</td><td>74.19±1.74</td><td>73.82±4.77</td></tr><tr><td> $TMDL-OA^{\star}$ </td><td>92.45±0.05</td><td>84.52±2.20</td><td>27.02±0.75</td><td>74.55±0.07</td><td>48.42±1.02</td><td>21.71±1.83</td><td>62.28±3.70</td><td>68.16±0.34</td></tr><tr><td> $UIMC^{\star}$ </td><td>97.72±0.18</td><td>96.43±1.19</td><td>41.72±0.31</td><td>89.71±0.25</td><td>67.88±0.48</td><td>50.43±0.46</td><td>79.84±0.92</td><td>70.66±2.04</td></tr><tr><td> $ECML^{\star}$ </td><td>94.52±0.79</td><td>90.00±2.78</td><td>39.89±0.59</td><td>84.02±0.51</td><td>56.97±0.52</td><td>50.31±1.81</td><td>74.88±1.89</td><td>84.00±0.14</td></tr><tr><td> $TMNR^{\star}$ </td><td>92.78±1.01</td><td>90.71±4.19</td><td>30.88±0.58</td><td>85.76±0.81</td><td>60.00±1.43</td><td>23.95±1.92</td><td>74.09±1.99</td><td>80.59±3.26</td></tr><tr><td> $CCML^{\star}$ </td><td>93.22±1.09</td><td>94.29±2.18</td><td>37.38±0.65</td><td>83.84±1.01</td><td>62.08±1.34</td><td>52.48±2.74</td><td>78.87±2.31</td><td>83.24±2.79</td></tr><tr><td> $FUML^{\star}$ </td><td>98.78±0.36</td><td>98.81±1.60</td><td>47.08±0.32</td><td>96.68±0.32</td><td>72.71±1.75</td><td>69.14±2.43</td><td>94.44±1.18</td><td>88.01±2.53</td></tr><tr><td>IMPROVE</td><td>△ 0.08</td><td>△ 2.38</td><td>△ 3.51</td><td>△ 3.96</td><td>△ 4.83</td><td>△ 7.31</td><td>△ 14.60</td><td>△ 4.01</td></tr></table>

![](images/9853ba04d6dd99018c86b83e921c6f634be4fc1e5f2ad6a769a2027029e5e869.jpg)

<details>
<summary>line</summary>

| Uncertainty | Normal | Conflicting |
| ----------- | ------ | ----------- |
| 0.0         | 20.0   | 20.0        |
| 0.2         | 5.0    | 5.0         |
| 0.4         | 2.0    | 2.0         |
| 0.6         | 1.0    | 1.0         |
| 0.8         | 0.5    | 0.5         |
| 1.0         | 0.2    | 0.2         |
</details>

(a) ETMC (Fashion)

![](images/ad80f323035557e8511de8654479b94c182aced9e5c250a7f93c31af282bf70a.jpg)

<details>
<summary>line</summary>

| Uncertainty | Normal | Conflicting |
| ----------- | ------ | ----------- |
| 0.0         | 0.0    | 0.0         |
| 0.1         | 3.5    | 4.0         |
| 0.2         | 1.5    | 1.8         |
| 0.3         | 1.2    | 1.5         |
| 0.4         | 1.0    | 1.2         |
| 0.5         | 0.8    | 0.9         |
| 0.6         | 0.6    | 0.7         |
| 0.7         | 0.4    | 0.5         |
| 0.8         | 0.2    | 0.3         |
| 0.9         | 0.1    | 0.1         |
| 1.0         | 0.0    | 0.0         |
</details>

(b) ECML (Fashion)

![](images/1b9ce15492dd5d64b9ad4bab208ce069ad72d9167e447b92bc48d7d3c5e6db56.jpg)

<details>
<summary>area</summary>

| Uncertainty | Normal Density | Conflicting Density |
|-------------|----------------|---------------------|
| 0.0         | 4.5            | 1.2                 |
| 0.2         | 0.5            | 0.8                 |
| 0.4         | 0.3            | 0.6                 |
| 0.6         | 1.0            | 1.5                 |
| 0.8         | 0.2            | 2.2                 |
| 1.0         | 0.1            | 0.5                 |
</details>

(c) FUML (Fashion)

![](images/b2835d62b77823b1289c46d93671f7a98566124b3b6cf0894c404e7a7f8c8bd3.jpg)

<details>
<summary>line</summary>

| Uncertainty | Normal | Conflicting |
| ----------- | ------ | ----------- |
| 0.0         | 0.0    | 0.0         |
| 0.2         | 0.5    | 0.5         |
| 0.4         | 0.5    | 0.5         |
| 0.6         | 1.0    | 1.0         |
| 0.8         | 2.5    | 2.5         |
| 1.0         | 3.0    | 3.0         |
</details>

(d) ETMC (LandUse)

![](images/a7e375fcf813202dc175a58a3414f72e43b9d858e96b6f9db2ff80d02138e6be.jpg)

<details>
<summary>line</summary>

| Uncertainty | Normal | Conflicting |
| ----------- | ------ | ----------- |
| 0.0         | 0.0    | 0.0         |
| 0.2         | 0.0    | 0.0         |
| 0.4         | 0.1    | 0.1         |
| 0.6         | 0.3    | 0.3         |
| 0.8         | 1.0    | 1.0         |
| 1.0         | 4.0    | 4.0         |
</details>

(e) ECML (LandUse)

![](images/3a8b5ddd89d9b1063c8ccc629b9d1657883a84c20a42432c15dea598294cc308.jpg)

<details>
<summary>area</summary>

| Uncertainty | Normal Density | Conflicting Density |
|-------------|----------------|---------------------|
| 0.0         | 0.0            | 0.0                 |
| 0.2         | 0.5            | 0.2                 |
| 0.4         | 1.0            | 0.5                 |
| 0.6         | 1.5            | 1.0                 |
| 0.8         | 2.0            | 2.5                 |
| 1.0         | 0.5            | 0.0                 |
</details>

(f) FUML (LandUse)   
Figure 3. Density of uncertainty on the normal and conflicting test sets of the Fashion and LandUse datasets.

![](images/d90902d780dbdac87e512becb77cbd8e13e0ffeb5ea7e5a10820757de5c9829c.jpg)

<details>
<summary>line</summary>

| Uncertainty threshold | HW    | MSRC  | NUSOBj | Fashion | Scene | LandUse | Leaves | PIE   |
| --------------------- | ----- | ----- | ------ | ------- | ----- | ------- | ------ | ----- |
| 0.2                   | 1.0   | 1.0   | 1.0    | 1.0     | 1.0   | 1.0     | 1.0    | 1.0   |
| 0.4                   | 1.0   | 1.0   | 0.95   | 1.0     | 0.98  | 1.0     | 1.0    | 1.0   |
| 0.6                   | 1.0   | 1.0   | 0.85   | 1.0     | 0.95  | 0.98    | 1.0    | 1.0   |
| 0.8                   | 1.0   | 1.0   | 0.7    | 1.0     | 0.9   | 0.95    | 1.0    | 1.0   |
| 1.0                   | 1.0   | 1.0   | 0.5    | 1.0     | 0.8   | 0.9     | 1.0    | 1.0   |
</details>

(a) Normal test set

![](images/0c3c2b4ea8faca8df8552bd928cbe268c723ddafc6b2f7b6bd1ddc71bd122da5.jpg)

<details>
<summary>line</summary>

| Uncertainty threshold | HW    | MSRC  | NUSOBJ | Fashion | Scene | LandUse | Leaves | PIE   |
| --------------------- | ----- | ----- | ------ | ------- | ----- | ------- | ------ | ----- |
| 0.2                   | 1.0   | 1.0   | 1.0    | 1.0     | 1.0   | 1.0     | 1.0    | 1.0   |
| 0.4                   | 1.0   | 1.0   | 0.95   | 0.98    | 0.97  | 0.96    | 0.95   | 0.98  |
| 0.6                   | 1.0   | 1.0   | 0.85   | 0.95    | 0.92  | 0.90    | 0.92   | 0.95  |
| 0.8                   | 1.0   | 1.0   | 0.75   | 0.92    | 0.85  | 0.85    | 0.88   | 0.92  |
| 1.0                   | 1.0   | 1.0   | 0.5    | 0.9     | 0.75  | 0.75    | 0.75   | 0.85  |
</details>

(b) Conflicting test set

Figure 4. Accuracy with uncertainty thresholding.   
![](images/08898805a2ac2e23bc7012a30bc7462d7722f36bdc5329903f6f93476f209b31.jpg)

<details>
<summary>line</summary>

| Epoch | View 1 | View 2 | View 3 | Multi-view |
|-------|--------|--------|--------|------------|
| 0     | 0.14   | 0.14   | 0.14   | 0.02       |
| 100   | 0.09   | 0.09   | 0.09   | 0.02       |
| 200   | 0.09   | 0.09   | 0.09   | 0.02       |
| 300   | 0.10   | 0.10   | 0.10   | 0.02       |
| 400   | 0.10   | 0.10   | 0.10   | 0.02       |
| 500   | 0.10   | 0.10   | 0.10   | 0.02       |
</details>

(a) Fashion

![](images/3c395fdc5a3058de11a93b79b6290d214434209d3d4176c3180aeafc55dc76df.jpg)

<details>
<summary>line</summary>

| Epoch | View 1 | View 2 | View 3 | Multi-view |
|-------|--------|--------|--------|------------|
| 0     | 0.9    | 0.6    | 0.8    | 0.6        |
| 100   | 0.6    | 0.3    | 0.5    | 0.3        |
| 200   | 0.55   | 0.3    | 0.5    | 0.25       |
| 300   | 0.55   | 0.3    | 0.5    | 0.25       |
| 400   | 0.55   | 0.3    | 0.5    | 0.25       |
| 500   | 0.55   | 0.3    | 0.5    | 0.25       |
</details>

(b) LandUse   
Figure 5. Prediction error with different epochs.

results can be found in Appendix C.2.

Additionally, to observe the trend of classification accuracy of FUML as the uncertainty threshold varies, we plot Figure 4. It illustrates that FUML achieves significantly more accurate predictions as the prediction uncertainty decreases for both normal and conflicting test sets on all eight datasets. This demonstrates that our model's output, i.e., classification results and the corresponding uncertainty, supports making trusted decisions.

# 4.4. Multi-view Fusion Effectiveness Evaluation

To evaluate the effectiveness of our FUML for multi-view fusion, we compare the prediction error of multi-view learning results (depicted as a red line, labeled “Multi-view”) with the prediction error of each single-view learning result on the Fashion and LandUse datasets. As shown in Figure 5, the prediction error of the multi-view is consistently lower than that of any single-view in the proposed method, demonstrating that it effectively reduces prediction error by integrating multiple views to achieve more accurate results. Results for the other six datasets are provided in Appendix C.3.

# 4.5. Ablation Study

To demonstrate the effectiveness of each component of our FUML, we perform an ablation study on the conflicting test set of the Fashion and LandUse datasets with mean accuracy (Acc.), mean precision (Prec.), and mean F-score over 10 seeds. To be specific, we calculate metrics for each label and find their unweighted mean for precision and F-score. In addition, for simplicity, in this section, we represent $\mathcal{L}_{fcl}(\mathbf{r}^{a},\mathbf{y})$ as $L_{ccl}^{a}$ and represent $\sum_{v=1}^{V}\mathcal{L}_{fcl}(\mathbf{r}^{v},\mathbf{y})$ as $L_{ccl}^{v}$ . The results, presented in Table 4, reveal: (1) After removing $L_{ccl}^{a}$ or $L_{ccl}^{v}$ , all indicators decline to varying degrees, which shows that all components in the total loss function are indispensable. (2) When performing feature-level fusion, i.e., concatenating (Concat) all features and only using one DNN for prediction, performance declines significantly. (3) Compared to the arithmetic mean (Avg), our DRF fusion demonstrates superior performance. In summary, all components of FUML are indispensable. Additional ablation experimental results on the normal test set can be found in Appendix C.4.

Table 4. Ablation study on the conflicting test sets of the Fashion and LandUse datasets with all metrics in percentages (%). 

<table><tr><td colspan="3">METHOD</td><td colspan="3">FASHION</td><td colspan="3">LANDUSE</td></tr><tr><td> $\mathcal{L}_{ccl}^{a}$ </td><td> $\mathcal{L}_{ccl}^{v}$ </td><td>RULE</td><td>ACC. ↑</td><td>PREC. ↑</td><td>F-SCORE ↑</td><td>ACC. ↑</td><td>PREC. ↑</td><td>F-SCORE ↑</td></tr><tr><td>√</td><td>×</td><td>DRF</td><td>96.33</td><td>96.31</td><td>96.30</td><td>68.00</td><td>68.72</td><td>67.53</td></tr><tr><td>×</td><td>√</td><td>DRF</td><td>96.32</td><td>96.32</td><td>96.30</td><td>67.76</td><td>68.41</td><td>67.33</td></tr><tr><td>√</td><td>×</td><td>CONCAT</td><td>88.45</td><td>88.28</td><td>88.75</td><td>59.69</td><td>60.54</td><td>58.54</td></tr><tr><td>√</td><td>√</td><td>AVG</td><td>96.15</td><td>96.13</td><td>96.13</td><td>67.71</td><td>68.22</td><td>67.42</td></tr><tr><td>√</td><td>√</td><td>DRF</td><td>96.68</td><td>96.68</td><td>96.68</td><td>69.14</td><td>70.21</td><td>69.19</td></tr></table>

![](images/1663788612671fa2a427e34b075f8230518a6151b754898a7e0fae75933e1d19.jpg)  
(a) Original

![](images/56e4fc093eef85540df0ac33a38234faaeb3b0a775fba04403b8e108a45ef560.jpg)  
(b) $u \leqslant 0.8$

![](images/6649c90d9f0b5fbec9eea415654b2387caa68b7b2cdb3e14f99a7b2198071fa1.jpg)  
(c) $u \leqslant 0.6$   
Figure 6. Visualization of aggregated memberships on the training set of LandUse dataset by t-SNE (Van der Maaten & Hinton, 2008). Samples belonging to the same category are rendered with the same color. (a) Display the results of fused memberships. (b)-(c) demonstrate the results of fused memberships after the samples with uncertainty greater than 0.8 and 0.6 are removed, respectively.

# 4.6. t-SNE Visualization and Analysis.

We employ the t-SNE approach to embed the fused memberships of the training set from the LandUse dataset into a two-dimensional visualization plane, as shown in Figure 6. The results demonstrate: (1) Distinct categories occupy different Spaces, and are well distinguished, indicating that FUML learns discriminative information. (2) After filtering out samples with uncertainty greater than 0.8 and 0.6, the category boundaries gradually become more distinct. These improvements are attributed to the effective uncertainty estimation capability of FUML.

# 5. Conclusion

This paper presents the Deep Fuzzy Multi-view Learning method (FUML), a novel multi-view classification framework designed to accurately classify conflicting multi-view instances and precisely estimate their intrinsic uncertainty. Based on the Fuzzy Set Theory, our FUML models the outputs of classification neural networks as a set of fuzzy memberships and quantifies category credibility by incorporating both possibility and necessity measures. To optimize category credibility, we propose a category credibility learning loss. In addition, we propose a Dual-reliable Fusion (DRF) strategy, which assigns weights based on view-specific uncertainty and inter-view conflict that effectively mitigates the influence of conflicting views. Extensive experiments on eight public datasets demonstrate FUML's superiority over 13 state-of-the-art methods in terms of accuracy, robustness, and reliability, particularly in challenging scenarios with conflicting views.

# Acknowledgements

This work was supported in part by the National Key R&D Program of China 2024YFB4710604; in part by NSFC under Grant 62472295, 62176171, 62372315, and U24B20174; in part by the Fundamental Research Funds for the Central Universities under Grant CJ202303, and CJ202403; in part by Sichuan Science and Technology Planning Project under Grant 2024ZDZX0004, 2024NSFTD0049, 2024ZHCG0005, and 24NSFTD0130; in part by TCL science and technology innovation fund; in part by System of Systems and Artificial Intelligence Laboratory pioneer fund grant; and in part by the Chengdu Science and Technology Project under Grant 2023-XT00-00004-GX.

# Impact Statement

This paper presents work to advance the field of multi-view learning in machine learning. Our goal is to construct a trusted and reliable multi-view classification method to boost the accuracy and credibility of joint decisions in multi-view systems, lowering the potential classification errors and inaccurate uncertainty estimates of prediction. However, due to the data bias in open environments, there is a possibility of inevitable error when applying our method in real-world applications.

# References

Bao, W., Yu, Q., and Kong, Y. Evidential deep learning for open set action recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 13349–13358, 2021.   
Bi, J. and Dornaika, F. Sample-weighted fused graph-based

semi-supervised learning on multi-view data. Information Fusion, 104:102175, 2024.

Cao, B., Xia, Y., Ding, Y., Zhang, C., and Hu, Q. Predictive dynamic fusion. International Conference on Machine Learning, 2024.

Chaudhuri, K., Kakade, S. M., Livescu, K., and Sridharan, K. Multi-view clustering via canonical correlation analysis. In Proceedings of the 26th Annual International Conference on Machine Learning, pp. 129–136, 2009.

Chen, M., Gao, J., and Xu, C. Uncertainty-aware dual-evidential learning for weakly-supervised temporal action localization. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023.

Chen, Z., Lou, K., Liu, Z., Li, Y., Luo, Y., and Zhao, L. Joint long and short span self-attention network for multi-view classification. Expert Systems with Applications, 235:121152, 2024.

Das, R., Sen, S., and Maulik, U. A survey on fuzzy deep neural networks. ACM Computing Surveys (CSUR), 53(3):1–25, 2020.

Duan, S., Sun, Y., Peng, D., Liu, Z., Song, X., and Hu, P. Fuzzy multimodal learning for trusted cross-modal retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

Gal, Y. and Ghahramani, Z. Bayesian convolutional neural networks with bernoulli approximate variational inference. arXiv preprint arXiv:1506.02158, 2015.

Gao, J., Chen, M., Xiang, L., and Xu, C. A comprehensive survey on evidential deep learning and its applications. arXiv preprint arXiv:2409.04720, 2024.

Geng, Y., Han, Z., Zhang, C., and Hu, Q. Uncertainty-aware multi-view representation learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 7545–7553, 2021.

Han, Z., Zhang, C., Fu, H., and Zhou, J. T. Trusted multiview classification. In International Conference on Learning Representations, 2020.

Han, Z., Yang, F., Huang, J., Zhang, C., and Yao, J. Multimodal dynamics: Dynamical fusion for trustworthy multimodal classification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20707–20717, 2022a.

Han, Z., Zhang, C., Fu, H., and Zhou, J. T. Trusted multiview classification with dynamic evidential fusion. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(2):2551–2566, 2022b.

He, C., Zhu, H., Hu, P., and Peng, X. Robust variational contrastive learning for partially view-unaligned clustering. In Proceedings of the 32nd ACM International Conference on Multimedia, pp. 4167–4176, 2024.   
Hong, Z., Lin, Q., and Hu, B. Knowledge distillation-based edge-decision hierarchies for interactive behavior-aware planning in autonomous driving system. IEEE Transactions on Intelligent Transportation Systems, 2024.   
Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. In International Conference on Learning Representations, 2015.   
Lakshminarayanan, B., Pritzel, A., and Blundell, C. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in Neural Information Processing Systems, 30, 2017.   
Li, Y., Zhen, L., Sun, Y., Peng, D., Peng, X., and Hu, P. Deep evidential hashing for trustworthy cross-modal retrieval. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 18566–18574, 2025.   
Lin, Y., Gou, Y., Liu, X., Bai, J., Lv, J., and Peng, X. Dual contrastive prediction for incomplete multi-view representation learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(4):4447–4461, 2022.   
Liu, B. and Liu, B. Uncertainty theory. Springer, 2010.   
Liu, W., Yue, X., Chen, Y., and Denoeux, T. Trusted multiview deep learning with opinion aggregation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pp. 7585–7593, 2022.   
Liu, W., Chen, Y., Yue, X., Zhang, C., and Xie, S. Safe multi-view deep classification. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pp. 8870–8878, 2023.   
Liu, Y., Liu, L., Xu, C., Song, X., Guan, Z., and Zhao, W. Dynamic evidence decoupling for trusted multi-view learning. In Proceedings of the 32nd ACM International Conference on Multimedia, pp. 7269–7277, 2024.   
Lyzhov, A., Molchanova, Y., Ashukha, A., Molchanov, D., and Vetrov, D. Greedy policy search: A simple baseline for learnable test-time augmentation. In Conference on Uncertainty in Artificial Intelligence, pp. 1308–1317. PMLR, 2020.   
Madry, A., Makelov, A., Schmidt, L., Tsipras, D., and Vladu, A. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations, 2018.

Mittal, A., Dahiya, K., Malani, S., Ramaswamy, J., Kuruvilla, S., Ajmera, J., Chang, K.-h., Agarwal, S., Kar, P., and Varma, M. Multi-modal extreme classification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12393–12402, 2022.   
Molchanov, D., Ashukha, A., and Vetrov, D. Variational dropout sparsifies deep neural networks. In International Conference on Machine Learning, pp. 2498–2507. PMLR, 2017.   
Nasarian, E., Alizadehsani, R., Acharya, U. R., and Tsui, K.-L. Designing interpretable ml system to enhance trust in healthcare: A systematic review to proposed responsible clinician-ai-collaboration framework. Information Fusion, pp. 102412, 2024.   
Nie, F., Li, J., Li, X., et al. Self-weighted multiview clustering with multiple graphs. In IJCAI, pp. 2564–2570, 2017.   
Qin, Y., Peng, D., Peng, X., Wang, X., and Hu, P. Deep evidential learning with noisy correspondence for cross-modal retrieval. In Proceedings of the 30th ACM International Conference on Multimedia, pp. 4948–4956, 2022.   
Qin, Y., Chen, Y., Peng, D., Peng, X., Zhou, J. T., and Hu, P. Noisy-correspondence learning for text-to-image person re-identification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 27197–27206, 2024.   
Qu, J., Yang, Y., Dong, W., and Yang, Y. Lds2ae: Local diffusion shared-specific autoencoder for multimodal remote sensing image classification with arbitrary missing modalities. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 14731–14739, 2024.   
Rupnik, J. and Shawe-Taylor, J. Multi-view canonical correlation analysis. In Conference on Data Mining and Data Warehouses (SiKDD 2010), volume 473, pp. 1–4, 2010.   
Sensoy, M., Kaplan, L., and Kandemir, M. Evidential deep learning to quantify classification uncertainty. Advances in Neural Information Processing Systems, 31, 2018.   
Shafer, G. Dempster-shafer theory. Encyclopedia of artificial intelligence, 1:330–331, 1992.   
Shang, Q., Li, H., Deng, Y., and Cheong, K. H. Compound credibility for conflicting evidence combination: An autoencoder-k-means approach. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 52(9):5602–5610, 2021.   
Van der Maaten, L. and Hinton, G. Visualizing data using t-sne. Journal of Machine Learning Research, 9(11), 2008.

Wang, C., Zhu, W., Gao, B.-B., Gan, Z., Zhang, J., Gu, Z., Qian, S., Chen, M., and Ma, L. Real-iad: A real-world multi-view dataset for benchmarking versatile industrial anomaly detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22883–22892, 2024a.   
Wang, R., Sun, H., Ma, Y., Xi, X., and Yin, Y. Metaviewer: Towards a unified multi-view representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11590–11599, 2023.   
Wang, X., Wang, Y., Ke, G., Wang, Y., and Hong, X. Knowledge distillation-driven semi-supervised multi-view classification. Information Fusion, 103:102098, 2024b.   
Wang, X., Wang, Y., Wang, Y., Huang, A., and Liu, J. Trusted semi-supervised multi-view classification with contrastive learning. IEEE Transactions on Multimedia, 2024c.   
Wang, X., Duan, S., Li, Q., Duan, G., Sun, Y., and Peng, D. Reliable disentanglement multi-view learning against view adversarial attacks. arXiv preprint arXiv:2505.04046, 2025.   
Wen, J., Liu, C., Deng, S., Liu, Y., Fei, L., Yan, K., and Xu, Y. Deep double incomplete multi-view multi-label learning with incomplete labels and missing views. IEEE Transactions on Neural Networks and Learning Systems, 2023.   
Winn, J. and Jojic, N. Locus: Learning object classes with unsupervised segmentation. In Tenth IEEE International Conference on Computer Vision (ICCV'05) Volume 1, volume 1, pp. 756–763. IEEE, 2005.   
Wu, W., Duan, S., Sun, Y., Yu, Y., Liu, D., and Peng, D. Deep fuzzy physics-informed neural networks for forward and inverse pde problems. Neural Networks, 181:106750, 2025.   
Xiao, F. Multi-sensor data fusion based on the belief divergence measure of evidences and the belief entropy. Information Fusion, 46:23–32, 2019a. ISSN 1566-2535.   
Xiao, F. Multi-sensor data fusion based on the belief divergence measure of evidences and the belief entropy. Information Fusion, 46:23–32, 2019b.   
Xie, M., Han, Z., Zhang, C., Bai, Y., and Hu, Q. Exploring and exploiting uncertainty for incomplete multi-view classification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19873–19882, 2023.   
Xu, C., Si, J., Guan, Z., Zhao, W., Wu, Y., and Gao, X. Reliable conflictive multi-view learning. In Proceedings

of the AAAI Conference on Artificial Intelligence, volume 38, pp. 16129–16137, 2024a.   
Xu, C., Zhang, Y., Guan, Z., and Zhao, W. Trusted multiview learning with label noise. In Larson, K. (ed.), Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI-24, pp. 5263–5271. International Joint Conferences on Artificial Intelligence Organization, 8 2024b. Main Track.   
Xu, S., Sun, Y., Li, X., Duan, S., Ren, Z., Liu, Z., and Peng, D. Noisy label calibration for multi-view classification. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 21797–21805, 2025.   
Yan, R., Xie, L., Tang, J., Shu, X., and Tian, Q. Higcin: Hierarchical graph-based cross inference network for group activity recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(6):6955–6968, 2020.   
Yan, X., Hu, S., Mao, Y., Ye, Y., and Yu, H. Deep multiview learning methods: A review. Neurocomputing, 448:106–129, 2021. ISSN 0925-2312.   
Yang, M., Deng, C., and Nie, F. Adaptive-weighting discriminative regression for multi-view classification. Pattern Recognition, 88:236–245, 2019.   
Yang, Y. and Newsam, S. Bag-of-visual-words and spatial extensions for land-use classification. In Proceedings of the 18th SIGSPATIAL International Conference on Advances in Geographic Information Systems, pp. 270–279, 2010.   
Yang, Z., Zhang, J., Wang, G., Kalra, M. K., and Yan, P. Cardiovascular disease detection from multi-view chest x-rays with bi-mamba. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 134–144. Springer, 2024.   
Yue, X., Dong, Z., Chen, Y., and Xie, S. Evidential dissonance measure in robust multi-view classification to resist adversarial attack. Information Fusion, 113:102605, 2025.   
Zadeh, L. A. Fuzzy sets. Information and Control, 8(3):338–353, 1965.   
Zhang, Q., Wu, H., Zhang, C., Hu, Q., Fu, H., Zhou, J. T., and Peng, X. Provable dynamic fusion for low-quality multimodal data. In International Conference on Machine Learning, pp. 41753–41769. PMLR, 2023.   
Zhang, Q., Wei, Y., Han, Z., Fu, H., Peng, X., Deng, C., Hu, Q., Xu, C., Wen, J., Hu, D., et al. Multimodal fusion on low-quality data: A comprehensive survey. arXiv preprint arXiv:2404.18947, 2024.

# APPENDIX

This document provides mathematical proofs, additional experimental details, additional experimental results, and limitations to support the paper:

- In Appendix A, we provide mathematical proofs for all propositions presented in Section 3.3.   
- In Appendix B, we present comprehensive experimental documentation:

– Datasets Details in Appendix B.1   
– Baselines Details in Appendix B.2   
- Implementation Details in Appendix B.3

\- In Appendix C, we conduct extensive supplementary analyses through nine investigations:

- Additional Conflicting Multi-view Classification Results and Analysis in Appendix C.1   
– Additional Uncertainty Effectiveness Analysis in Appendix C.2   
- Additional Multi-view Fusion Effectiveness Analysis in Appendix C.3   
- Additional Ablation Study in Appendix C.4   
– Identification of Out-of-distribution in Appendix C.5   
- Parameter Analysis in Appendix C.6   
- Challenges of DST in ECML in Appendix C.7   
- Conflict Visualization in Appendix C.8   
- Adversarial Noise Effect Analysis in Appendix C.9

\- In Appendix D, we discuss the limitations of our FUML.

# A. Proofs

# A.1. Proof of Proposition 3.4

Proof. Without loss of generality, for the i-th multi-view instance, let $m_{i}^{1} = \left[m_{i1}^{1}, m_{i2}^{1}, ..., m_{iK}^{1}\right]$ denote the sorted memberships of a clean view $X_{i}^{1}$ , i.e., $m_{i1}^{1} \geqslant m_{i2}^{1} \geqslant, ..., m_{iK}^{1}$ , and $m_{i}^{2} = \left[m_{i1}^{2}, m_{i2}^{2}, ..., m_{iK}^{2}\right]$ denote sorted memberships of another noisy view $X_{i}^{2}$ . We assume that $m_{i1}^{1} - m_{ik}^{1} \geqslant m_{i1}^{2} - m_{ik}^{2}, k = 2, 3, ..., K$ . Therefore, we have

$$
c _ {i 1} ^ {1} = \frac {m _ {i 1} ^ {1} + 1 - m _ {i 2} ^ {1}}{2},
$$

$$
c _ {i k} ^ {1} = \frac {m _ {i k} ^ {1} + 1 - m _ {i 1} ^ {1}}{2}, k = 2, 3,..., K, \tag {10}
$$

$$
u _ {i} ^ {1} = \frac {H (\frac {m _ {i 1} ^ {1} + 1 - m _ {i 2} ^ {1}}{2}) + \sum_ {k = 2} ^ {K} H (\frac {m _ {i k} ^ {1} + 1 - m _ {i 1} ^ {1}}{2})}{K \cdot \ln 2},
$$

where $H(t) = -t \cdot \ln(t) - (1 - t) \cdot \ln(1 - t)$ . Because $t = 0.5$ is the symmetry axis of $H(t)$ , we have

$$
u _ {i} ^ {1} = \frac {H (\frac {m _ {i 1} ^ {1} - m _ {i 2} ^ {1} + 1}{2}) + \sum_ {k = 2} ^ {K} H (\frac {m _ {i 1} ^ {1} - m _ {i k} ^ {1} + 1}{2})}{K \cdot \ln 2} \tag {11}
$$

After fusing view $\mathcal{X}_i^1$ and $\mathcal{X}_i^2$ , we have

$$
\mathbf {m} _ {i k} ^ {a} = w _ {i} ^ {1} \cdot \mathbf {m} _ {i k} ^ {1} + w _ {i} ^ {2} \cdot \mathbf {m} _ {i k} ^ {2}, k = 2, 3, \dots , K, \tag {12}
$$

where $m_{i}^{a}$ denotes the fused memberships, and $w_{i}^{1} > 0$ and $w_{i}^{2} > 0$ represent the weights of view $X_{i}^{1}$ and $X_{i}^{2}$ , respectively. Therefore, we have

$$
\begin{array}{l} m _ {i 1} ^ {a} - m _ {i 2} ^ {a} = w _ {i} ^ {1} \cdot \left(m _ {i 1} ^ {1} - m _ {i 2} ^ {1}\right) + w _ {i} ^ {2} \cdot \left(m _ {i 1} ^ {2} - m _ {i 2} ^ {2}\right) <   m _ {i 1} ^ {1} - m _ {i 2} ^ {1}, \tag {13} \\ m _ {i 1} ^ {a} - m _ {i k} ^ {a} = w _ {i} ^ {1} \cdot (m _ {i 1} ^ {1} - m _ {i k} ^ {1}) + w _ {i} ^ {2} \cdot (m _ {i 1} ^ {2} - m _ {i k} ^ {2}) <   m _ {i 1} ^ {1} - m _ {i k} ^ {1}, k = 2, 3,..., K, \\ \end{array}
$$

and

$$
H \left(\frac {m _ {i 1} ^ {a} - m _ {i 2} ^ {a} + 1}{2}\right) > H \left(\frac {m _ {i 1} ^ {1} - m _ {i 2} ^ {1} + 1}{2}\right), \tag {14}
$$

$$
H (\frac {m _ {i 1} ^ {a} - m _ {i k} ^ {a} + 1}{2}) > H (\frac {m _ {i 1} ^ {1} - m _ {i k} ^ {1} + 1}{2}), k = 2, 3,..., K.
$$

Therefore, we have

$$
u _ {i} ^ {a} = \frac {H (\frac {m _ {i 1} ^ {a} - m _ {i 2} ^ {a} + 1}{2}) + \sum_ {k = 2} ^ {K} H (\frac {m _ {i 1} ^ {a} - m _ {i k} ^ {a} + 1}{2})}{K \cdot \ln 2} > \frac {H (\frac {m _ {i 1} ^ {1} - m _ {i 2} ^ {1} + 1}{2}) + \sum_ {k = 2} ^ {K} H (\frac {m _ {i 1} ^ {1} - m _ {i k} ^ {1} + 1}{2})}{K \cdot \ln 2} = u _ {i} ^ {1}. \tag {15}
$$

# A.2. Proof of Proposition 3.5

Proof. Without loss of generality, for the i-th multi-view instance, let $m_{i}^{1} = \left[m_{i1}^{1}, m_{i2}^{1}, ..., m_{iK}^{1}\right]$ denote the memberships of a clean view $X_{i}^{1}$ , and $m_{i}^{2} = \left[m_{i1}^{2}, m_{i2}^{2}, ..., m_{iK}^{2}\right]$ denote the memberships of another clean but misaligned view $X_{i}^{2}$ . We assume that the view $X_{i}^{1}$ is of the p-th category and the view $X_{i}^{2}$ is of the q-th category. Therefore, we can assume that

$$
m _ {i p} ^ {1} \gg m _ {i h} ^ {1}, \forall h \neq p,
$$

$$
m _ {i q} ^ {2} \gg m _ {i d} ^ {2}, \forall d \neq q, \tag {16}
$$

$$
m _ {i p} ^ {1} \approx m _ {i q} ^ {2}.
$$

After that, for view $X_{i}^{1}$ we have

$$
c _ {i p} ^ {1} = \frac {m _ {i p} ^ {1} + 1 - \max \left\{m _ {i h} ^ {1} \mid h \neq p \right\}}{2}, \tag {17}
$$

$$
c _ {i h} ^ {1} = \frac {m _ {i h} ^ {1} + 1 - m _ {i p} ^ {1}}{2}.
$$

Then,

$$
u _ {i} ^ {1} = \frac {H (c _ {i p} ^ {1}) + \sum_ {h \neq p} H (c _ {i h} ^ {1})}{K \cdot \ln 2} = \frac {H (\frac {m _ {i p} ^ {1} + 1 - \max \{m _ {i h} ^ {1} | h \neq p \}}{2}) + \sum_ {k \neq p} H (\frac {m _ {i h} ^ {1} + 1 - m _ {i p} ^ {1}}{2})}{K \cdot \ln 2}, \tag {18}
$$

where $H(t) = -t \cdot \ln(t) - (1 - t) \cdot \ln(1 - t)$ . Because $t = 0.5$ is the symmetry axis of $H(t)$ , we have

$$
u _ {i} ^ {1} = \frac {H (\frac {m _ {i p} ^ {1} + 1 - \max \{m _ {i h} ^ {1} \mid h \neq p \}}{2}) + \sum_ {h \neq p} H (\frac {m _ {i p} ^ {1} + 1 - m _ {i h} ^ {1}}{2})}{K \cdot \ln 2}, \tag {19}
$$

Meanwhile, for view $X_{i}^{2}$ , we have

$$
u _ {i} ^ {2} = \frac {H (\frac {m _ {i q} ^ {2} + 1 - \max \{m _ {i h} ^ {2} \mid d \neq q \}}{2}) + \sum_ {d \neq q} H (\frac {m _ {i q} ^ {2} + 1 - m _ {i d} ^ {2}}{2})}{K \cdot \ln 2}. \tag {20}
$$

After fusing view $\mathcal{X}_i^1$ and $\mathcal{X}_i^2$ , because $m_{ip}^{1}\approx m_{iq}^{2}\gg m_{iq}^{1},m_{ip}^{2},m_{ik}^{1},m_{ik}^{2},k\neq p$ and $k\neq q$ , we have

$$
u _ {i} ^ {a} = \frac {H (\frac {m _ {i p} ^ {a} + 1 - m _ {i q} ^ {a}}{2}) + H (\frac {m _ {i q} ^ {a} + 1 - m _ {i p} ^ {a}}{2}) + \sum_ {k \neq p , k \neq q} H (\frac {m _ {i k} ^ {a} + 1 - m _ {i p} ^ {a}}{2})}{K \cdot \ln 2}, \tag {21}
$$

$$
\approx \frac {2 \cdot \ln 2 + \sum_ {k \neq p , k \neq q} H (\frac {m _ {i p} ^ {a} + 1 - m _ {i k} ^ {a}}{2})}{K \cdot \ln 2}
$$

Next, we prove that $u_{i}^{a} > u_{i}^{1}$ . Specifically, we have

$$
m _ {i p} ^ {a} - m _ {i k} ^ {a} = w _ {i} ^ {1} \cdot (m _ {i p} ^ {1} - m _ {i k} ^ {1}) + w _ {i} ^ {2} \cdot (m _ {i p} ^ {2} - m _ {i k} ^ {2}), \tag {22}
$$

where $w_{i}^{1} \in (0,1)$ and $w_{i}^{2} \in (0,1)$ represent the weights of view $X_{i}^{1}$ and $X_{i}^{2}$ , respectively. In addition, we have $w_{i}^{1} + w_{i}^{2} = 1$ . Because $m_{ip}^{1} > m_{iq}^{2} \gg m_{iq}^{1}, m_{ip}^{2}, m_{ik}^{1}, m_{ik}^{2}, k \neq p$ and $k \neq q$ , we can deduce that

$$
m _ {i p} ^ {2} - m _ {i k} ^ {2} <   m _ {i p} ^ {a} - m _ {i k} ^ {a} <   m _ {i p} ^ {1} - m _ {i k} ^ {1}, \forall k \neq p. \tag {23}
$$

Therefore,

$$
H (\frac {m _ {i p} ^ {1} + 1 - m _ {i k} ^ {1}}{2}) <   H (\frac {m _ {i p} ^ {a} + 1 - m _ {i k} ^ {a}}{2}). \tag {24}
$$

Then, we can deduce that

$$
H (\frac {m _ {i p} ^ {1} + 1 - \max \{m _ {i k} ^ {1} | k \neq p \}}{2}) \ll H (\frac {1}{2}) = \ln 2. \tag {25}
$$

Combine Equations (19), (21), (24) and (25), we can deduce that

$$
u _ {i} ^ {1} = \frac {H (\frac {m _ {i p} ^ {1} + 1 - \max \{m _ {i h} ^ {1} | h \neq p \}}{2}) + \sum_ {h \neq p} H (\frac {m _ {i p} ^ {1} + 1 - m _ {i h} ^ {1}}{2})}{K \cdot \ln 2} <   \frac {2 \cdot \ln 2 + \sum_ {k \neq p , k \neq q} H (\frac {m _ {i p} ^ {a} + 1 - m _ {i k} ^ {a}}{2})}{K \cdot \ln 2} = u _ {i} ^ {a}. \tag {26}
$$

Similarly, $u^a > u^2$ can also be easily proved.

![](images/e9bb880e2e93c1e9f71313550b3807427f26e50ee2936b2802628af5050087ae.jpg)

# B. Experimental Details

# B.1. Datasets Details

The multi-view data used in this paper include:

- HandWritten (HW) $^{7}$ comprises 2000 instances of handwritten numerals ranging from '0' to '9', with 200 patterns per class, represented using six feature sets.   
- MSRC-V1 (MSRC) (Winn & Jojic, 2005) contains 210 images. Each image includes 7 classes. Following (Nie et al., 2017), we extract five features, including CM, HOG, GIST, CENTRIST feature, and LBP.   
- NUS-WIDE-OBJECT $^{8}$ (NUSOBJ) consists of 30,000 images of 31 classes. Each instance is described as 5 views, including Color Histogram, block-wise Color Moments, Color Correlogram, Edge Direction Histogram, and Wavelet Texture.   
- Fashion-MV (Fashion) (Wang et al., 2023) is an image dataset that contains 10 categories with a total of 30,000 fashion products. It has three views, each consisting of 10,000 grayscale images sampled from the same category.   
- Scene15 (Scene) $^{9}$ includes 4485 images from 15 indoor and outdoor scene categories, with features extracted using GIST, PHOG, and LBP.   
- LandUse (Yang & Newsam, 2010) contains 2100 satellite images with 3 views and 21 categories.   
- Leaves100 (Leaves) $^{10}$ consists of 1600 leaf samples from 100 plant species. We extracted shape descriptors, fine-scale edges, and texture histograms as 3 views.   
- $\mathbf{PIE}^{11}$ contains 680 instances belonging to 68 classes, with intensity, LBP, and Gabor as 3 views.

# B.2. Baselines Details

We compare the proposed FUML with the following baselines:

(1) The untrusted baselines include:

- DFTMC (Dynamical Fusion for Trustworthy Multimodal Classification) (Han et al., 2022a) captures both feature and modality informativeness, proposing a dynamical fusion network for trustworthy multimodal classification.   
- DCP (Dual Contrastive Prediction) (Lin et al., 2022) Provides an information-theoretic framework integrating consistency learning and data recovery, imputing missing views by minimizing conditional entropy through dual prediction.   
- QMF (Quality-aware Multimodal Fusion) (Zhang et al., 2023) improves classification accuracy and model robustness by a provably robust multimodal fusion method.   
- PDF (Predictive Dynamic Fusion) (Cao et al., 2024) reveals the multimodal fusion from a generalization perspective, improving reliability and stability.

(2) The trusted baselines include:

- DUA-Nets (Dynamic Uncertainty-Aware Networks) (Geng et al., 2021) utilizes reversal networks to integrate intrinsic information from different views into a unified representation.   
- TMC (Trusted Multi-view Classification) (Han et al., 2020) pioneers addressing the uncertainty estimation problem in multi-view classification and producing trusted classification results.   
- ETMC (Enhanced Trusted Multi-view Classification) (Han et al., 2022b) extends TMC by incorporating a pseudo view, enabling comprehensive interaction among different views.   
- TMDL-OA (Trusted Multi-View Deep Learning with Opinion Aggregation) (Liu et al., 2022) proposes a consistency measure loss to achieve trustworthy learning results.   
- UIMC (Uncertainty-induced Incomplete Multi-View Data Classification) (Xie et al., 2023) uses uncertainty-based imputation and evidence-based fusion for reliable classification of incomplete multi-view data.   
- ECML (Evidential Conflictive Multiview Learning) (Xu et al., 2024a) is the SOTA method for conflict multi-view classification, which proposes a new opinion aggregation strategy.   
- TMNR (Trusted Multi-view Learning with Label Noise) (Xu et al., 2024b) is a reliable multi-view learning model under the guidance of noisy labels.   
- CCML (Consistent and Complementary-aware trusted Multi-view Learning) (Liu et al., 2024) solves the problem of data semantic fuzziness in multi-view learning by dynamically decoupling consistency and complementary evidence, thus improving the accuracy and reliability of classification.

# B.3. Implementation Details

All experiments are implemented in PyTorch and are carried out on NVIDIA Tesla V100S. During the training phase, our FUML uses Adam (Kingma & Ba, 2015) with $\beta_{1} = 0.9$ , $\beta_{2} = 0.999$ , a weight decay of 0.0001, and a maximum of 500 epochs. The $p$ in Equation (3) is set to 3. For the NUSOBJ and Fashion datasets, the learning rate is set to 0.0002 and the batch size to 400, while for the remaining six datasets, the learning rate is set to 0.001 and the batch size to 100. The pseudo-code of FUML is shown in Algorithm 1. In addition, for a fair comparison, we replace the backbone networks of QMF (Zhang et al., 2023) and PDF (Cao et al., 2024) with the same fully connected layer as FUML while preserving their core models and loss functions. For other baselines, we follow the settings in their source code.

Algorithm 1 FUML algorithm   
/*Training*/
Input: the multi-view training data $\{\{x_{n}^{v}\}_{v=1}^{V}, y_{n}\}_{n=1}^{N}$ , batch size $N_{b}$ , maximal epoch number $N_{e}$ , learning rate $\eta$ , and the multi-view model $\{f^{v}(\cdot, \theta^{v})\}_{v=1}^{V}$ .
Output: optimized network parameters $\{\theta^{v}\}_{v=1}^{V}$ .
for $1, 2, \cdots, N_{e}$ do
    Randomly select $N_{b}$ samples from every view to construct a multi-view mini-batch.
    Calculate the output $\{\{a_{j}^{v}\}_{j=1}^{N_{b}}\}_{v=1}^{V}$ for all samples of the mini-batch by using their corresponding model $\{f^{v}(\cdot, \theta^{v})\}_{v=1}^{V}$ .
    Calculate view-specific memberships $\{m^{v}\}_{v=1}^{V}$ by Equation (3).
    Calculate credibility degrees during training $\{r^{v}\}_{v=1}^{V}$ by Equation (5).
    Calculate category credibility during training by Equation (5) and aggregate memberships by Equation (8).
    Compute $L_{total}$ according to Equation (9) on minibatch.
    Update FUML parameters $\{\theta^{i}\}_{i=1}^{V}$ using gradient descent algorithm with learning rate $\eta$ .
end for
/*Testing*/
Calculate the view-specific memberships by the trained model.
Calculate category credibility by Equation (2) and fuse memberships by Equation (8).

# C. Additional Experiments

# C.1. Additional Conflicting Multi-view Classification Results and Analysis

To further assess the performance of our FUML in conflicting multi-view classification, we compare it against the three best-performing untrusted multi-view classification methods, i.e., DCP-CG, QMF, and PDF, as well as the three best-performing trusted multi-view classification methods, i.e., UIMC, ECML, and CCML. Each method is evaluated over 10 runs, and the mean values along with standard deviations are reported.

Table 9 presents the experimental results where Gaussian noise with a mean of 0 and variances of 1, 5, and 10 is randomly added to half of the views in half of the test sets. Similarly, Table 10 illustrates the results of adding Gaussian noise with a mean of 0 and a variance of 1 to half of the views in randomly selected 10%, 20%, 30%, 40%, and 50% of the test set. Finally, Table 11 displays the experimental results of introducing unaligned views to half of the views in randomly selected 10%, 20%, 30%, 40%, and 50% of the test sets. These results indicate that our FUML surpasses nearly all comparison methods across various conflict settings, highlighting its effectiveness in conflicting multi-view classification.

# C.2. Additional Uncertainty Effectiveness Analysis

In Section 4.4 and Figure 3, we provide qualitative results on uncertainty estimation, and to complement them, we report here the quantitative results. Results are shown in Table 5, which demonstrate that our FUML is more effective than ETMC and ECML in uncertainty estimation.

Table 5. Quantitative results of uncertainty estimation. The normal test sets serve as in-distribution, while the conflicting test sets serve as out-of-distribution. The evaluation metric is FPR95, where lower values indicate better performance. The best results are highlighted in boldface. 

<table><tr><td>DATASET\METHOD</td><td>ETMC</td><td>ECML</td><td>FUML (OURS)</td></tr><tr><td>FASHION</td><td>0.930</td><td>0.967</td><td>0.510</td></tr><tr><td>LANDUSE</td><td>0.964</td><td>0.945</td><td>0.886</td></tr></table>

# C.3. Additional Multi-view Fusion Effectiveness Analysis

In this section, we present additional experimental results for analyzing the effectiveness of multi-view fusion. As shown in Figure 7, the prediction error of the multi-view approach is consistently lower than that of any single view in the proposed method. These results confirm that our method effectively reduces prediction error by integrating multiple views to achieve more accurate results, aligning with the observations reported in Section 4.4.

![](images/54609f93d9665a108e56efd92d72be38951f3cef0c4b331798cce646339deadd.jpg)

<details>
<summary>line</summary>

| Epoch | View 1 | View 2 | View 3 | View 4 | View 5 | View 6 | Multi-view |
|-------|--------|--------|--------|--------|--------|--------|------------|
| 0     | 0.3    | 0.3    | 0.3    | 0.3    | 0.3    | 0.3    | 0.0        |
| 100   | 0.3    | 0.15   | 0.1    | 0.15   | 0.1    | 0.1    | 0.0        |
| 200   | 0.3    | 0.15   | 0.1    | 0.15   | 0.1    | 0.1    | 0.0        |
| 300   | 0.3    | 0.15   | 0.1    | 0.15   | 0.1    | 0.1    | 0.0        |
| 400   | 0.3    | 0.15   | 0.1    | 0.15   | 0.1    | 0.1    | 0.0        |
| 500   | 0.5    | 0.15   | 0.1    | 0.15   | 0.1    | 0.1    | 0.0        |
</details>

(a) HW

![](images/4d245f863f20603c26d645d94a9772c13d961430598d308f9ea75da61f4cea98.jpg)

<details>
<summary>line</summary>

| Epoch | View 1 | View 2 | View 3 | View 4 | View 5 | View 6 | Multi-view |
|-------|--------|--------|--------|--------|--------|--------|------------|
| 0     | 0.8    | 0.8    | 0.8    | 0.8    | 0.8    | 0.8    | 0.8        |
| 100   | 0.4    | 0.4    | 0.2    | 0.2    | 0.2    | 0.2    | 0.0        |
| 200   | 0.4    | 0.4    | 0.2    | 0.2    | 0.2    | 0.2    | 0.0        |
| 300   | 0.4    | 0.4    | 0.2    | 0.2    | 0.2    | 0.2    | 0.0        |
| 400   | 0.4    | 0.4    | 0.2    | 0.2    | 0.2    | 0.2    | 0.0        |
| 500   | 0.4    | 0.4    | 0.2    | 0.2    | 0.2    | 0.2    | 0.0        |
</details>

(b) MSRC

![](images/602df614876ff2bb1a65bf60dd20df6a4f30a9f966e40531f9b2f2157633f407.jpg)

<details>
<summary>line</summary>

| Epoch | View 1 | View 2 | View 3 | View 4 | View 5 | Multi-view |
|-------|--------|--------|--------|--------|--------|------------|
| 0     | 0.725  | 0.700  | 0.680  | 0.660  | 0.640  | 0.590      |
| 100   | 0.690  | 0.670  | 0.660  | 0.675  | 0.630  | 0.525      |
| 200   | 0.685  | 0.675  | 0.670  | 0.680  | 0.635  | 0.525      |
| 300   | 0.680  | 0.675  | 0.675  | 0.685  | 0.635  | 0.525      |
| 400   | 0.685  | 0.675  | 0.675  | 0.685  | 0.635  | 0.525      |
| 500   | 0.690  | 0.675  | 0.675  | 0.685  | 0.635  | 0.525      |
</details>

(c) NUSOBJ

![](images/5286e6d0130992f2dab7588c2a6dda452814b02ecbac3ada36083dcc75c28f2a.jpg)

<details>
<summary>line</summary>

| Epoch | View 1 | View 2 | View 3 | Multi-view |
|-------|--------|--------|--------|------------|
| 0     | 0.14   | 0.14   | 0.14   | 0.02       |
| 100   | 0.09   | 0.09   | 0.09   | 0.01       |
| 200   | 0.09   | 0.09   | 0.09   | 0.01       |
| 300   | 0.10   | 0.10   | 0.10   | 0.01       |
| 400   | 0.10   | 0.10   | 0.10   | 0.01       |
| 500   | 0.10   | 0.10   | 0.10   | 0.01       |
</details>

(d) Fashion

![](images/dbf6c6d31e2ca2a700d118fdc442902c169f8db3f07f46d3550159f09c400df3.jpg)

<details>
<summary>line</summary>

| Epoch | View 1 | View 2 | View 3 | Multi-view |
|-------|--------|--------|--------|------------|
| 0     | 0.7    | 0.7    | 0.7    | 0.3        |
| 100   | 0.45   | 0.25   | 0.5    | 0.2        |
| 200   | 0.4    | 0.25   | 0.45   | 0.2        |
| 300   | 0.4    | 0.25   | 0.45   | 0.2        |
| 400   | 0.4    | 0.25   | 0.45   | 0.2        |
| 500   | 0.4    | 0.25   | 0.45   | 0.2        |
</details>

(e) Scene

![](images/1b58fb0308c073c59c722a27e020a26dd73bb1c123c04c0ab35ebf581b3b6307.jpg)

<details>
<summary>line</summary>

| Epoch | View 1 | View 2 | View 3 | Multi-view |
|-------|--------|--------|--------|------------|
| 0     | 0.9    | 0.85   | 0.85   | 0.6        |
| 100   | 0.6    | 0.35   | 0.55   | 0.3        |
| 200   | 0.55   | 0.3    | 0.5    | 0.25       |
| 300   | 0.55   | 0.3    | 0.5    | 0.25       |
| 400   | 0.55   | 0.3    | 0.5    | 0.25       |
| 500   | 0.55   | 0.3    | 0.5    | 0.25       |
</details>

(f) LandUse

![](images/7b071af6f805c25d35ecea54fd0d1fa75f308e684c3f963e50381c9c78f1ea28.jpg)

<details>
<summary>line</summary>

| Epoch | View 1 | View 2 | View 3 | Multi-view |
|-------|--------|--------|--------|------------|
| 0     | 0.8    | 1.0    | 0.8    | 0.2        |
| 100   | 0.2    | 0.4    | 0.2    | 0.0        |
| 200   | 0.2    | 0.4    | 0.2    | 0.0        |
| 300   | 0.2    | 0.4    | 0.2    | 0.0        |
| 400   | 0.2    | 0.4    | 0.2    | 0.0        |
| 500   | 0.2    | 0.4    | 0.2    | 0.0        |
</details>

(g) Leaves

![](images/5af81a2c2db0ba1bef333954efaeb585696c6b56ceaebb5d917a47077b7c1129.jpg)

<details>
<summary>line</summary>

| Epoch | View 1 | View 2 | View 3 | Multi-view |
|-------|--------|--------|--------|------------|
| 0     | 0.6    | 0.9    | 0.7    | 0.6        |
| 50    | 0.1    | 0.85   | 0.1    | 0.05       |
| 100   | 0.1    | 0.85   | 0.1    | 0.05       |
| 150   | 0.1    | 0.85   | 0.1    | 0.05       |
| 200   | 0.1    | 0.85   | 0.2    | 0.05       |
| 250   | 0.1    | 0.85   | 0.1    | 0.05       |
| 300   | 0.1    | 0.85   | 0.1    | 0.05       |
| 350   | 0.1    | 0.85   | 0.1    | 0.05       |
| 400   | 0.1    | 0.85   | 0.1    | 0.05       |
| 450   | 0.1    | 0.85   | 0.1    | 0.05       |
| 500   | 0.1    | 0.85   | 0.1    | 0.05       |
</details>

(h) PIE   
Figure 7. Prediction error with different epochs.

# C.4. Additional Ablation Study

In this section, we report the additional ablation experimental results: (1) Results in Table 6 show that all components contribute positively to performance. Compared to Concat (i.e., concatenate all view features) and Avg (i.e., $\mathbf{m}_i^a = (\sum_{v=1}^{V} \mathbf{m}_i^v)/V$ ), DRF is more effective on conflicting test sets than on normal test sets. (2) To evaluate the role of necessity, we removed the necessity in FUML and only used conflicts in the fusion process. Results in Table 7 prove that necessity can't be removed. (3) To further evaluate the role of uncertainty and conflict in DRF, we conduct more detailed ablation experiments. Results in Table 8 show that removing uncertainty ( $u$ ) or conflict ( $c$ ) in DRF leads to performance degradation, indicating the effectiveness of considering both uncertainty and conflict.

Table 6. Classification accuracy (ACC), Precision (Prec.), and F-score of FUML with different combination rules on the normal and conflicting test sets. All metrics are expressed as percentages (%). The best results are highlighted in boldface. 

<table><tr><td colspan="3">METHOD\DATASET</td><td colspan="3">FASHION (NORMAL)</td><td colspan="3">LANDUSE (NORMAL)</td><td colspan="3">FASHION (CONFLICTING)</td><td colspan="3">LANDUSE (CONFLICTING)</td></tr><tr><td> $\mathcal{L}_{ccl}^{a}$ </td><td> $\mathcal{L}_{ccl}^{v}$ </td><td>RULE</td><td>ACC↑</td><td>PREC.↑</td><td>F-SCORE↑</td><td>ACC↑</td><td>PREC.↑</td><td>F-SCORE↑</td><td>ACC↑</td><td>PREC.↑</td><td>F-SCORE↑</td><td>ACC↑</td><td>PREC.↑</td><td>F-SCORE↑</td></tr><tr><td>√</td><td>×</td><td>DRF</td><td>98.66</td><td>98.66</td><td>98.48</td><td>76.29</td><td>76.89</td><td>76.09</td><td>96.33</td><td>96.31</td><td>96.30</td><td>68.00</td><td>68.72</td><td>67.53</td></tr><tr><td>×</td><td>√</td><td>DRF</td><td>98.73</td><td>98.73</td><td>98.73</td><td>75.71</td><td>76.14</td><td>75.24</td><td>96.32</td><td>96.32</td><td>96.30</td><td>67.76</td><td>68.41</td><td>67.33</td></tr><tr><td>√</td><td>×</td><td>CONCAT</td><td>95.71</td><td>95.71</td><td>95.70</td><td>72.26</td><td>72.44</td><td>71.55</td><td>88.45</td><td>88.28</td><td>88.75</td><td>59.69</td><td>60.54</td><td>58.54</td></tr><tr><td>√</td><td>√</td><td>AVG</td><td>98.50</td><td>98.51</td><td>98.51</td><td>76.19</td><td>77.14</td><td>76.07</td><td>96.15</td><td>96.13</td><td>96.13</td><td>67.71</td><td>68.22</td><td>67.42</td></tr><tr><td>√</td><td>√</td><td>DRF</td><td>98.96</td><td>98.97</td><td>98.96</td><td>76.71</td><td>77.57</td><td>76.48</td><td>96.68</td><td>96.68</td><td>96.68</td><td>69.14</td><td>70.21</td><td>69.19</td></tr></table>

Table 7. Classification accuracy (%) on the normal and conflicting test set of the Fashion and LandUse datasets. The means and standard deviations over ten runs are reported. The best results are highlighted in boldface. 

<table><tr><td>METHOD\DATASET</td><td>FASHION (NORMAL)</td><td>LANDUSE (NORMAL)</td><td>FASHION (CONFLICTING)</td><td>LANDUSE (CONFLICTING)</td></tr><tr><td>w/o NECESSITY</td><td>97.70±0.41</td><td>44.64±3.65</td><td>94.91±0.47</td><td>39.00±3.29</td></tr><tr><td>OURS</td><td>98.96±0.25</td><td>76.71±0.46</td><td>96.68±0.32</td><td>69.14±2.43</td></tr></table>

Table 8. Classification accuracy (%) on the conflicting test set of the Fashion and LandUse datasets. The means and standard deviations over ten runs are reported. The best results are highlighted in boldface. 

<table><tr><td>METHOD\DATASET</td><td>FASHION</td><td>LANDUSE</td></tr><tr><td>AVG</td><td> $96.15 \pm 0.22$ </td><td> $67.71 \pm 2.30$ </td></tr><tr><td> $1 - u_{i}^{v}$ </td><td> $96.27 \pm 0.22$ </td><td> $68.14 \pm 2.47$ </td></tr><tr><td> $1 - c_{i}^{v}$ </td><td> $96.43 \pm 0.32$ </td><td> $68.50 \pm 2.02$ </td></tr><tr><td>OURS</td><td> $96.68 \pm 0.32$ </td><td> $69.14 \pm 2.28$ </td></tr></table>

# C.5. Identification of Out-of-distribution

This section presents out-of-distribution (OOD) detection results for FUML across all datasets. To validate the effectiveness of our FUML as a trusted method in data noise identification, we add Gaussian noise with fixed standard deviation ( $\delta = 0.1, 0.5, 1.0$ ) to all of the test samples in the test sets, creating OOD samples. In contrast, the remaining data were treated as in-distribution (ID) samples. The results are shown in Figure 8, indicating that ID samples exhibit consistently lower uncertainty relative to OOD samples across all eight datasets. Moreover, OOD samples with greater Gaussian noise deviation generally demonstrate higher uncertainty. Additionally, datasets with higher prediction accuracy (e.g., HW, MSRC, Fashion, Leaves, and PIE) exhibit lower overall uncertainty, whereas datasets with lower prediction accuracy, such as NUSOJB, Scene, and LandUse, tend to display higher uncertainty. These results demonstrate that FUML effectively measures uncertainty, thereby ensuring the reliability of the model's decisions.

![](images/9b7636f4d10d14b38b87223d5208f5acbbceeffa039ce5299d31dddf99f92820.jpg)

<details>
<summary>line</summary>

| Uncertainty | in-distribution | OOD(δ = 0.1) | OOD(δ = 0.5) | OOD(δ = 1.0) |
| ----------- | --------------- | ------------ | ------------ | ------------ |
| 0.0         | 0.0             | 0.0          | 0.0          | 0.0          |
| 0.2         | 2.5             | 2.0          | 1.5          | 1.0          |
| 0.4         | 3.5             | 3.0          | 1.5          | 1.0          |
| 0.6         | 2.0             | 1.5          | 1.5          | 1.5          |
| 0.8         | 1.0             | 1.0          | 1.5          | 2.0          |
| 1.0         | 0.5             | 0.5          | 1.0          | 1.5          |
</details>

(a) HW

![](images/ebe2c512c25bb317b8780668cd8a888c15ff5daf61df6c8076f7c053c4d1bd99.jpg)

<details>
<summary>line</summary>

| Uncertainty | in-distribution | OOD(δ = 0.1) | OOD(δ = 0.5) | OOD(δ = 1.0) |
| ----------- | --------------- | ------------ | ------------ | ------------ |
| 0.0         | 0.0             | 0.0          | 0.0          | 0.0          |
| 0.2         | 1.5             | 1.2          | 0.8          | 0.6          |
| 0.4         | 1.8             | 1.5          | 1.2          | 0.9          |
| 0.6         | 1.6             | 1.8          | 1.6          | 1.2          |
| 0.8         | 1.2             | 1.5          | 1.8          | 1.8          |
| 1.0         | 0.8             | 0.5          | 1.0          | 2.0          |
</details>

(b) MSRC

![](images/c265e7058a21691ca87fb07b050a6ec2b265931d20ba8191028b356271db1059.jpg)

<details>
<summary>line</summary>

| Uncertainty | in-distribution | OOD(δ = 0.1) | OOD(δ = 0.5) | OOD(δ = 1.0) |
| ----------- | --------------- | ------------ | ------------ | ------------ |
| 0.0         | 0.0             | 0.0          | 0.0          | 0.0          |
| 0.2         | 0.0             | 0.0          | 0.0          | 0.0          |
| 0.4         | 0.5             | 0.3          | 0.2          | 0.1          |
| 0.6         | 1.5             | 1.0          | 0.8          | 0.5          |
| 0.8         | 3.0             | 2.5          | 2.0          | 1.5          |
| 1.0         | 6.0             | 5.5          | 5.0          | 4.5          |
</details>

(c) NUSOBJ

![](images/ee861c12abd84bcbe589ed60d431c4f4d67474417ef488b740a7a51850c7261e.jpg)

<details>
<summary>area</summary>

| Uncertainty | in-distribution | OOD(δ = 0.1) | OOD(δ = 0.5) | OOD(δ = 1.0) |
| ----------- | --------------- | ------------ | ------------ | ------------ |
| 0.0         | 4.0             | 2.5          | 0.0          | 0.0          |
| 0.2         | 2.5             | 1.5          | 0.5          | 0.0          |
| 0.4         | 1.0             | 1.0          | 1.0          | 0.5          |
| 0.6         | 0.5             | 0.5          | 1.5          | 1.0          |
| 0.8         | 0.0             | 0.0          | 2.0          | 2.5          |
| 1.0         | 0.0             | 0.0          | 0.0          | 3.0          |
</details>

(d) Fashion

![](images/36f181518962d812f3f8f5ff9d492d7a23efd66f2b7f5975b7f8bf4d218d5c94.jpg)

<details>
<summary>area</summary>

| Uncertainty | in-distribution | OOD(δ = 0.1) | OOD(δ = 0.5) | OOD(δ = 1.0) |
| ----------- | --------------- | ------------ | ------------ | ------------ |
| 0.0         | 0.0             | 0.0          | 0.0          | 0.0          |
| 0.2         | 1.0             | 0.2          | 0.1          | 0.1          |
| 0.4         | 1.5             | 0.5          | 0.3          | 0.4          |
| 0.6         | 2.0             | 1.0          | 0.8          | 1.2          |
| 0.8         | 2.5             | 2.0          | 1.5          | 2.0          |
| 1.0         | 3.5             | 3.0          | 3.5          | 3.5          |
</details>

(e) Scene

![](images/ec1a875a1dad256b6df0e2205fc6595259655081761c78c907d28e071812a266.jpg)

<details>
<summary>line</summary>

| Uncertainty | in-distribution | OOD(δ = 0.1) | OOD(δ = 0.5) | OOD(δ = 1.0) |
| ----------- | --------------- | ------------ | ------------ | ------------ |
| 0.0         | 0.0             | 0.0          | 0.0          | 0.0          |
| 0.2         | 0.5             | 0.4          | 0.3          | 0.2          |
| 0.4         | 0.8             | 0.7          | 0.6          | 0.5          |
| 0.6         | 1.5             | 1.4          | 1.3          | 1.2          |
| 0.8         | 2.2             | 2.5          | 2.7          | 2.8          |
| 1.0         | 1.0             | 1.2          | 1.4          | 1.5          |
</details>

(f) LandUse

![](images/b3260f2568fbd4e1a4d817020636b6ed71ff3ef4890d6005fa57a6b784ecde71.jpg)

<details>
<summary>area</summary>

| Uncertainty | in-distribution | OOD(δ = 0.1) | OOD(δ = 0.5) | OOD(δ = 1.0) |
| ----------- | --------------- | ------------ | ------------ | ------------ |
| 0.0         | 0.0             | 0.0          | 0.0          | 0.0          |
| 0.2         | 0.0             | 0.0          | 0.0          | 0.0          |
| 0.4         | 2.5             | 0.0          | 0.0          | 0.0          |
| 0.6         | 2.5             | 1.0          | 0.0          | 0.0          |
| 0.8         | 1.0             | 3.0          | 1.0          | 1.0          |
| 1.0         | 17.5            | 5.0          | 17.5         | 17.5         |
</details>

(g) Leaves

![](images/27b97ab96c8c2f87f121160dad0f56749e5c7260d3cd0209b22570f9a9b0b359.jpg)

<details>
<summary>line</summary>

| Uncertainty | Normal | OOD(δ = 0.1) | OOD(δ = 0.5) | OOD(δ = 1) |
| ----------- | ------ | ------------ | ------------ | ---------- |
| 0.0         | 1.0    | 1.0          | 0.0          | 0.0        |
| 0.2         | 3.0    | 2.0          | 0.0          | 0.0        |
| 0.4         | 1.0    | 1.0          | 0.0          | 0.0        |
| 0.6         | 0.5    | 0.5          | 0.5          | 0.5        |
| 0.8         | 0.2    | 0.2          | 3.0          | 4.0        |
| 1.0         | 0.1    | 0.1          | 1.0          | 6.0        |
</details>

(h) PIE   
Figure 8. Density of uncertainty.

# C.6. Parameter Analysis

To investigate the parameter sensitivity of our method, we plot the accuracy, precision, and F-score of multi-view classification versus different p on the test sets of all the datasets as shown in Figure 9. The results show that in all datasets, the accuracy, precision, and F-score all increase first and then decrease with the increase of p. In general, performance is best when p is within 2-5. In this paper, we set p to 3 on all datasets. If we adjust the p value in our model, we believe our FUML will improve over what is reported in the Table 2 and Table 3.

![](images/c2134cf9b379c239a23407524a8195e847a78690f5dda8e7513bfa2f0ecfe705.jpg)

<details>
<summary>line</summary>

| p   | Accuracy | Precision | F-score |
| --- | -------- | --------- | ------- |
| 1   | 0.9902   | 0.9920    | 0.9901  |
| 2   | 0.9920   | 0.9918    | 0.9920  |
| 3   | 0.9920   | 0.9918    | 0.9918  |
| 4   | 0.9920   | 0.9918    | 0.9918  |
| 5   | 0.9920   | 0.9920    | 0.9911  |
| 6   | 0.9925   | 0.9925    | 0.9928  |
| 7   | 0.9930   | 0.9930    | 0.9925  |
| 8   | 0.9915   | 0.9915    | 0.9915  |
| 9   | 0.9920   | 0.9920    | 0.9918  |
| 10  | 0.9915   | 0.9915    | 0.9915  |
</details>

(a) HW

![](images/a21e42159dedc122e2b5f3a87b8f2fb3a199bcc9b29aad47397ebd450730c8cc.jpg)

<details>
<summary>line</summary>

| p  | Accuracy | Precision | F-score |
|----|----------|-----------|---------|
| 1  | 0.995    | 0.996     | 0.992   |
| 2  | 0.998    | 0.998     | 0.994   |
| 3  | 0.998    | 0.998     | 0.994   |
| 4  | 0.993    | 0.995     | 0.991   |
| 5  | 0.993    | 0.994     | 0.990   |
| 6  | 0.993    | 0.995     | 0.991   |
| 7  | 0.993    | 0.995     | 0.991   |
| 8  | 0.993    | 0.994     | 0.989   |
| 9  | 0.993    | 0.993     | 0.989   |
| 10 | 0.993    | 0.993     | 0.988   |
</details>

(b) MSRC

![](images/99d9a192c9876609de71ae8cf0562c38e22d78e337b8e007947ccbc2341c9d12.jpg)

<details>
<summary>line</summary>

| p  | Accuracy | Precision | F-score |
|----|----------|-----------|---------|
| 2  | 0.45     | 0.32      | 0.23    |
| 4  | 0.47     | 0.46      | 0.29    |
| 6  | 0.47     | 0.47      | 0.29    |
| 8  | 0.47     | 0.45      | 0.29    |
| 10 | 0.47     | 0.43      | 0.28    |
</details>

(c) NUSOBJ

![](images/4baa1c4d74d44619921f9ad111abb27a564197c6d10644b310eccb8c876e2f93.jpg)

<details>
<summary>line</summary>

| p  | Accuracy | Precision | F-score |
|----|----------|-----------|---------|
| 1  | 0.982    | 0.982     | 0.982   |
| 2  | 0.989    | 0.989     | 0.989   |
| 3  | 0.989    | 0.989     | 0.989   |
| 4  | 0.989    | 0.989     | 0.989   |
| 5  | 0.989    | 0.989     | 0.989   |
| 6  | 0.989    | 0.989     | 0.989   |
| 7  | 0.989    | 0.989     | 0.989   |
| 8  | 0.989    | 0.989     | 0.989   |
| 9  | 0.989    | 0.989     | 0.989   |
| 10 | 0.989    | 0.989     | 0.989   |
</details>

(d) Fashion

![](images/9b132f9081f169a936df3ae43703e0ca21cf2d2dd18dfdd5be4e455741c75c66.jpg)

<details>
<summary>line</summary>

| p  | Accuracy | Precision | F-score |
|----|----------|-----------|---------|
| 1  | 0.76     | 0.76      | 0.74    |
| 2  | 0.78     | 0.78      | 0.77    |
| 3  | 0.79     | 0.79      | 0.78    |
| 4  | 0.79     | 0.79      | 0.78    |
| 5  | 0.79     | 0.79      | 0.78    |
| 6  | 0.79     | 0.79      | 0.78    |
| 7  | 0.79     | 0.79      | 0.78    |
| 8  | 0.79     | 0.79      | 0.78    |
| 9  | 0.78     | 0.78      | 0.77    |
| 10 | 0.78     | 0.79      | 0.77    |
</details>

(e) Scene

![](images/a2e92a8b88733967fa54c2ef74d34e5adc87e8558d80dbff1a8f904cb388219a.jpg)

<details>
<summary>line</summary>

| p  | Accuracy | Precision | F-score |
|----|----------|-----------|---------|
| 1  | 0.70     | 0.70      | 0.69    |
| 2  | 0.76     | 0.76      | 0.74    |
| 3  | 0.77     | 0.77      | 0.76    |
| 4  | 0.78     | 0.77      | 0.76    |
| 5  | 0.78     | 0.77      | 0.77    |
| 6  | 0.77     | 0.77      | 0.76    |
| 7  | 0.76     | 0.76      | 0.76    |
| 8  | 0.76     | 0.76      | 0.75    |
| 9  | 0.76     | 0.76      | 0.75    |
| 10 | 0.76     | 0.76      | 0.75    |
</details>

(f) LandUse

![](images/a8ad62a5d44c8268ade3ad7964dd575c540a008bfac439fb81bacc630eafe576.jpg)

<details>
<summary>line</summary>

| p   | Accuracy | Precision | F-score |
| --- | -------- | --------- | ------- |
| 1   | 0.70     | 0.70      | 0.70    |
| 2   | 0.99     | 0.99      | 0.99    |
| 3   | 0.99     | 0.99      | 0.99    |
| 4   | 0.99     | 0.99      | 0.99    |
| 5   | 0.99     | 0.99      | 0.99    |
| 6   | 0.98     | 0.98      | 0.98    |
| 7   | 0.98     | 0.98      | 0.98    |
| 8   | 0.97     | 0.97      | 0.97    |
| 9   | 0.97     | 0.97      | 0.97    |
| 10  | 0.97     | 0.97      | 0.97    |
</details>

(g) Leaves

![](images/3bf75afb41fe1c450652a1153463e2b0f588325a05192b4c0f977a60551b6eab.jpg)

<details>
<summary>line</summary>

| p  | Accuracy | Precision | F-score |
|----|----------|-----------|---------|
| 2  | 0.95     | 0.95      | 0.93    |
| 4  | 0.96     | 0.95      | 0.95    |
| 6  | 0.96     | 0.95      | 0.94    |
| 8  | 0.96     | 0.95      | 0.94    |
| 10 | 0.94     | 0.93      | 0.93    |
</details>

(h) PIE   
Figure 9. The influence of $p$ .

# C.7. Challenges of DST in ECML

In evidential deep learning, the outputs of the classification neural network are modeled as evidence. Here, we define the evidence vector for v-th view as $e^{v} = [e_{1}^{v}, ..., e_{K}^{v}]$ . The parameter $\alpha_{k}^{v} = [\alpha_{1}^{v}, ..., \alpha_{K}^{v}]$ of the Dirichlet distribution is induced from evidence, i.e., $\alpha_{k}^{v} = e_{k}^{v} + 1$ . Then, the belief mass $b_{k}^{v}$ and the uncertainty $u^{v}$ are computed as

$$
b _ {k} ^ {v} = \frac {e _ {k} ^ {v}}{S ^ {v}} = \frac {\alpha_ {k} ^ {v} - 1}{S ^ {v}}, \text { and } u ^ {v} = \frac {K}{S ^ {v}}, \tag {27}
$$

where $S^{v} = \sum_{i=1}^{K}(e_{i}^{v} +) = \sum_{i=1}^{K}\alpha_{i}^{v}$ is the Dirichlet strength. Next, we analyze why the decision-level fusion in ECML (Xu et al., 2024a) is order-dependent. In the ECML framework, it is established that fusing two opinions $(\boldsymbol{w} = (\boldsymbol{b}, \boldsymbol{u}, \boldsymbol{a}))$ could be mathematically represented as averaging two pieces of evidence, using the following formula:

$$
\boldsymbol {w} = \boldsymbol {w} ^ {1} \underline {{\diamond}} \boldsymbol {w} ^ {2} \underline {{\diamond}} \dots \underline {{\diamond}} \boldsymbol {w} ^ {V}. \tag {28}
$$

This formula is implemented in the code snippet:

evidence\_a = evidences[0]

for v in range(num\_views):

evidence\_a = (evidences[i] + evidence\_a) / 2

ECML essentially applies the D-S combination rule by sequentially merging evidence from different views. However, it has a significant limitation: its decision-level fusion is order-dependent, where the later decision (here referring to evidence) strongly influences the final decision. As a result, ECML is sensitive to the order of conflicting views, making the final decision less reliable when contradictions arise between views. The proposed FUML uses uncertainty and conflict to calculate weights and simultaneously fuses the decision of multiple views at the decision-level fusion, thus avoiding this problem. The results are shown in the Table 12, confirming this.

Table 9. We add Gaussian noise on random $50\%$ modalities of $50\%$ random samples in the test set and $\delta$ presents the standard deviation. The means and standard deviations of classification accuracies $(\%)$ over ten runs are reported. The best results are highlighted in boldface. 

<table><tr><td>DATASET</td><td>METHOD</td><td> $\delta = 0$ </td><td> $\delta = 1$ </td><td> $\delta = 5$ </td><td> $\delta = {10}$ </td></tr><tr><td rowspan="7">HW</td><td>DCP-CG</td><td>99.00±0.47</td><td>98.02±0.52</td><td>96.40±1.28</td><td>93.40±1.12</td></tr><tr><td>QMF</td><td>98.72±0.48</td><td>94.85±0.99</td><td>77.32±1.46</td><td>71.23±1.56</td></tr><tr><td>PDF</td><td>98.40±0.37</td><td>95.40±0.98</td><td>78.92±1.65</td><td>73.55±1.68</td></tr><tr><td>UIMC</td><td>98.25±0.00</td><td>97.22±0.13</td><td>94.70±0.24</td><td>93.42±0.23</td></tr><tr><td>ECML</td><td>98.72±0.39</td><td>89.55±1.75</td><td>73.03±1.51</td><td>70.18±1.21</td></tr><tr><td>CCML</td><td>97.60±0.63</td><td>94.05±1.32</td><td>72.17±1.51</td><td>66.85±1.80</td></tr><tr><td>OURS</td><td>99.20±0.36</td><td>96.78±0.82</td><td>92.38±1.53</td><td>91.72±1.72</td></tr><tr><td rowspan="7">MSRC</td><td>DCP-CG</td><td>95.24±3.69</td><td>79.05±2.78</td><td>75.24±3.23</td><td>74.14±3.43</td></tr><tr><td>QMF</td><td>97.86±1.28</td><td>93.57±2.62</td><td>73.33±0.60</td><td>68.57±5.19</td></tr><tr><td>PDF</td><td>97.14±1.78</td><td>96.43±2.44</td><td>80.71±3.27</td><td>73.81±4.64</td></tr><tr><td>UIMC</td><td>98.81±1.19</td><td>97.21±1.29</td><td>94.71±1.78</td><td>93.29±1.90</td></tr><tr><td>ECML</td><td>94.05±1.60</td><td>83.33±3.53</td><td>68.33±3.38</td><td>65.95±3.85</td></tr><tr><td>CCML</td><td>96.90±2.39</td><td>91.90±1.90</td><td>73.33±7.59</td><td>66.43±6.16</td></tr><tr><td>OURS</td><td>99.76±0.75</td><td>98.10±2.33</td><td>95.00±2.49</td><td>94.29±3.05</td></tr><tr><td rowspan="7">NUSOBJ</td><td>DCP-CG</td><td>43.65±1.10</td><td>29.11±1.28</td><td>28.73±1.13</td><td>28.67±1.12</td></tr><tr><td>QMF</td><td>45.41±0.43</td><td>32.94±0.31</td><td>30.62±0.39</td><td>30.25±0.43</td></tr><tr><td>PDF</td><td>46.78±0.33</td><td>34.15±0.32</td><td>31.94±0.33</td><td>31.69±0.35</td></tr><tr><td>UIMC</td><td>43.42±0.12</td><td>40.97±0.12</td><td>36.67±0.13</td><td>35.85±0.19</td></tr><tr><td>ECML</td><td>42.62±0.42</td><td>31.59±0.65</td><td>30.33±0.53</td><td>29.91±0.48</td></tr><tr><td>CCML</td><td>41.43±0.71</td><td>29.96±0.87</td><td>28.01±0.67</td><td>27.75±0.65</td></tr><tr><td>OURS</td><td>48.23±0.42</td><td>41.49±0.49</td><td>41.20±0.54</td><td>41.16±0.55</td></tr><tr><td rowspan="7">FASHION</td><td>DCP-CG</td><td>98.11±0.23</td><td>86.77±2.10</td><td>82.45±2.67</td><td>82.24±2.66</td></tr><tr><td>QMF</td><td>98.93±0.32</td><td>93.05±0.48</td><td>71.88±0.72</td><td>69.94±0.69</td></tr><tr><td>PDF</td><td>98.95±0.19</td><td>93.16±0.59</td><td>79.83±0.81</td><td>73.28±0.70</td></tr><tr><td>UIMC</td><td>98.13±0.13</td><td>95.57±0.12</td><td>93.04±0.21</td><td>92.49±0.18</td></tr><tr><td>ECML</td><td>97.93±0.35</td><td>91.00±0.70</td><td>76.76±0.74</td><td>73.16±0.67</td></tr><tr><td>CCML</td><td>95.16±0.41</td><td>91.39±0.43</td><td>78.22±0.84</td><td>72.49±0.87</td></tr><tr><td>OURS</td><td>98.96±0.25</td><td>95.22±0.32</td><td>90.66±0.85</td><td>89.82±0.91</td></tr><tr><td rowspan="7">SCENE</td><td>DCP-CG</td><td>77.79±1.73</td><td>62.52±1.22</td><td>55.50±0.60</td><td>54.89±0.52</td></tr><tr><td>QMF</td><td>68.58±1.49</td><td>52.68±1.63</td><td>48.46±1.57</td><td>47.90±1.89</td></tr><tr><td>PDF</td><td>70.25±1.21</td><td>55.25±1.39</td><td>50.49±1.54</td><td>49.78±1.61</td></tr><tr><td>UIMC</td><td>77.70±0.00</td><td>72.32±0.56</td><td>66.35±0.65</td><td>65.63±0.72</td></tr><tr><td>ECML</td><td>76.19±0.12</td><td>58.08±1.34</td><td>57.66±1.86</td><td>57.09±1.91</td></tr><tr><td>CCML</td><td>73.87±1.83</td><td>51.43±2.10</td><td>47.87±1.60</td><td>47.50±1.66</td></tr><tr><td>OURS</td><td>79.41±1.34</td><td>70.14±2.27</td><td>69.11±2.24</td><td>68.97±2.25</td></tr><tr><td rowspan="7">LANDUSE</td><td>DCP-CG</td><td>75.74±0.98</td><td>55.05±1.66</td><td>53.76±1.08</td><td>53.57±0.93</td></tr><tr><td>QMF</td><td>47.86±2.55</td><td>35.36±2.65</td><td>33.38±2.56</td><td>33.43±2.75</td></tr><tr><td>PDF</td><td>45.17±2.66</td><td>36.21±2.17</td><td>34.64±1.74</td><td>34.31±1.71</td></tr><tr><td>UIMC</td><td>57.95±0.61</td><td>53.26±0.56</td><td>47.93±0.59</td><td>47.10±0.55</td></tr><tr><td>ECML</td><td>60.10±2.01</td><td>39.81±1.27</td><td>37.61±2.11</td><td>37.14±2.13</td></tr><tr><td>CCML</td><td>60.86±1.93</td><td>45.76±1.75</td><td>42.95±1.63</td><td>42.71±1.72</td></tr><tr><td>OURS</td><td>76.71±0.46</td><td>63.86±1.55</td><td>63.50±1.41</td><td>63.48±1.47</td></tr><tr><td rowspan="7">LEAVES</td><td>DCP-CG</td><td>98.19±0.46</td><td>65.38±1.22</td><td>65.31±1.20</td><td>65.21±1.31</td></tr><tr><td>QMF</td><td>95.69±1.25</td><td>74.16±2.08</td><td>64.75±2.06</td><td>64.06±2.20</td></tr><tr><td>PDF</td><td>98.03±0.71</td><td>77.62±2.42</td><td>68.34±1.85</td><td>66.94±2.07</td></tr><tr><td>UIMC</td><td>95.31±0.71</td><td>91.66±0.97</td><td>82.94±0.97</td><td>82.50±0.93</td></tr><tr><td>ECML</td><td>92.53±1.94</td><td>72.94±1.89</td><td>65.78±1.99</td><td>64.84±2.18</td></tr><tr><td>CCML</td><td>97.72±0.92</td><td>60.03±1.99</td><td>56.91±2.34</td><td>56.69±2.50</td></tr><tr><td>OURS</td><td>99.78±0.27</td><td>93.69±1.17</td><td>92.94±1.36</td><td>92.84±1.40</td></tr><tr><td rowspan="7">PIE</td><td>DCP-CG</td><td>90.59±1.99</td><td>61.18±3.40</td><td>61.18±3.40</td><td>61.18±3.40</td></tr><tr><td>QMF</td><td>92.06±1.64</td><td>75.51±2.61</td><td>63.64±3.05</td><td>62.79±3.11</td></tr><tr><td>PDF</td><td>92.57±1.66</td><td>77.35±3.29</td><td>65.81±3.67</td><td>63.90±2.95</td></tr><tr><td>UIMC</td><td>91.69±2.16</td><td>89.25±2.22</td><td>84.81±2.03</td><td>83.85±2.01</td></tr><tr><td>ECML</td><td>94.71±0.02</td><td>74.63±3.21</td><td>64.85±3.48</td><td>63.31±3.52</td></tr><tr><td>CCML</td><td>93.97±1.67</td><td>75.44±3.53</td><td>63.97±2.90</td><td>62.57±3.06</td></tr><tr><td>OURS</td><td>96.18±1.24</td><td>89.41±1.68</td><td>88.38±2.07</td><td>88.38±2.02</td></tr></table>

Table 10. We add Gaussian noise with standard deviation 1 on 50% modalities in different proportions of the test sets. The means and standard deviations of classification accuracies (%) over ten runs are reported. The best results are highlighted in boldface. 

<table><tr><td>DATASET</td><td>METHOD</td><td>0%</td><td>10%</td><td>20%</td><td>30%</td><td>40%</td><td>50%</td></tr><tr><td rowspan="7">HW</td><td>DCP-CG</td><td>99.00±0.47</td><td>98.45±0.37</td><td>98.45±0.37</td><td>98.25±0.34</td><td>98.15±0.45</td><td>98.02±0.52</td></tr><tr><td>QMF</td><td>98.72±0.48</td><td>97.58±0.79</td><td>96.75±0.72</td><td>96.20±0.84</td><td>95.67±0.90</td><td>94.85±0.99</td></tr><tr><td>PDF</td><td>98.40±0.37</td><td>97.82±0.55</td><td>97.22±0.64</td><td>96.60±0.90</td><td>96.02±0.97</td><td>95.40±0.98</td></tr><tr><td>UIMC</td><td>98.25±0.00</td><td>98.15±0.12</td><td>97.95±0.12</td><td>97.95±0.12</td><td>97.92±0.12</td><td>97.22±0.13</td></tr><tr><td>ECML</td><td>98.72±0.39</td><td>96.18±0.58</td><td>93.92±0.81</td><td>92.30±0.76</td><td>91.05±1.09</td><td>89.55±1.75</td></tr><tr><td>CCML</td><td>97.60±0.63</td><td>96.85±0.71</td><td>96.05±0.99</td><td>95.35±1.02</td><td>94.82±1.28</td><td>94.05±1.32</td></tr><tr><td>OURS</td><td>99.20±0.36</td><td>98.62±0.45</td><td>98.15±0.53</td><td>97.82±0.63</td><td>97.38±0.60</td><td>96.78±0.82</td></tr><tr><td rowspan="7">MSRC</td><td>DCP-CG</td><td>95.24±3.69</td><td>91.43±2.43</td><td>88.10±2.61</td><td>84.29±2.86</td><td>82.86±3.16</td><td>79.05±2.78</td></tr><tr><td>QMF</td><td>97.86±1.28</td><td>96.90±1.86</td><td>95.95±2.14</td><td>95.24±2.13</td><td>94.76±2.08</td><td>93.57±2.62</td></tr><tr><td>PDF</td><td>97.14±1.78</td><td>96.90±1.86</td><td>96.90±2.39</td><td>96.90±2.14</td><td>96.9±2.14</td><td>96.43±2.44</td></tr><tr><td>UIMC</td><td>98.81±1.19</td><td>98.05±1.17</td><td>98.05±1.17</td><td>97.81±1.19</td><td>97.81±1.19</td><td>97.21±1.29</td></tr><tr><td>ECML</td><td>94.05±1.60</td><td>91.90±2.43</td><td>89.52±2.86</td><td>88.10±3.53</td><td>86.43±3.38</td><td>83.33±3.53</td></tr><tr><td>CCML</td><td>96.90±2.39</td><td>94.05±3.06</td><td>93.57±3.2</td><td>93.57±3.2</td><td>92.62±2.49</td><td>91.90±1.90</td></tr><tr><td>OURS</td><td>99.76±0.75</td><td>99.76±0.71</td><td>99.29±1.09</td><td>98.81±1.19</td><td>98.81±1.19</td><td>98.10±2.33</td></tr><tr><td rowspan="7">NUSOBJ</td><td>DCP-CG</td><td>43.65±1.10</td><td>41.09±0.75</td><td>38.22±0.56</td><td>35.43±0.52</td><td>32.62±0.56</td><td>29.11±1.28</td></tr><tr><td>QMF</td><td>45.41±0.43</td><td>42.71±0.42</td><td>40.12±0.43</td><td>37.27±0.31</td><td>34.32±0.38</td><td>32.94±0.31</td></tr><tr><td>PDF</td><td>46.78±0.33</td><td>45.94±0.49</td><td>43.08±0.48</td><td>40.09±0.23</td><td>37.10±0.26</td><td>34.15±0.32</td></tr><tr><td>UIMC</td><td>43.42±0.12</td><td>42.95±0.13</td><td>42.33±0.15</td><td>41.81±0.14</td><td>41.27±0.17</td><td>40.97±0.12</td></tr><tr><td>ECML</td><td>42.62±0.42</td><td>40.62±0.89</td><td>38.50±0.84</td><td>37.09±0.66</td><td>35.06±0.82</td><td>31.59±0.65</td></tr><tr><td>CCML</td><td>41.43±0.71</td><td>38.76±0.64</td><td>36.57±0.79</td><td>34.33±0.67</td><td>32.07±0.77</td><td>29.96±0.87</td></tr><tr><td>OURS</td><td>48.23±0.42</td><td>46.66±0.75</td><td>45.39±0.68</td><td>44.10±0.58</td><td>42.78±0.56</td><td>41.49±0.49</td></tr><tr><td rowspan="7">FASHION</td><td>DCP-CG</td><td>98.11±0.23</td><td>95.42±0.55</td><td>93.33±0.89</td><td>91.09±1.29</td><td>88.87±1.75</td><td>86.77±2.10</td></tr><tr><td>QMF</td><td>98.93±0.32</td><td>96.84±0.51</td><td>95.67±0.56</td><td>94.71±0.45</td><td>93.78±0.51</td><td>93.05±0.48</td></tr><tr><td>PDF</td><td>98.95±0.19</td><td>97.57±0.36</td><td>96.64±0.43</td><td>95.82±0.46</td><td>95.14±0.49</td><td>93.16±0.59</td></tr><tr><td>UIMC</td><td>98.13±0.13</td><td>97.57±0.12</td><td>97.42±0.12</td><td>97.20±0.10</td><td>96.97±0.12</td><td>95.57±0.12</td></tr><tr><td>ECML</td><td>97.93±0.35</td><td>96.44±0.42</td><td>94.91±0.48</td><td>93.48±0.51</td><td>92.14±0.59</td><td>91.00±0.70</td></tr><tr><td>CCML</td><td>95.16±0.41</td><td>94.47±0.55</td><td>93.58±0.48</td><td>92.78±0.52</td><td>92.07±0.48</td><td>91.39±0.43</td></tr><tr><td>OURS</td><td>98.96±0.25</td><td>98.22±0.27</td><td>97.43±0.28</td><td>96.75±0.33</td><td>95.94±0.28</td><td>95.22±0.32</td></tr><tr><td rowspan="7">SCENE</td><td>DCP-CG</td><td>77.79±1.73</td><td>74.85±0.83</td><td>71.37±0.28</td><td>68.38±0.52</td><td>65.66±0.96</td><td>62.52±1.22</td></tr><tr><td>QMF</td><td>68.58±1.49</td><td>65.32±1.60</td><td>62.15±1.44</td><td>59.31±1.25</td><td>55.89±1.45</td><td>52.68±1.63</td></tr><tr><td>PDF</td><td>70.25±1.21</td><td>67.28±1.26</td><td>64.37±1.20</td><td>61.46±0.94</td><td>58.25±0.95</td><td>55.25±1.39</td></tr><tr><td>UIMC</td><td>77.70±0.00</td><td>76.10±0.54</td><td>74.95±0.56</td><td>73.43±0.54</td><td>72.93±0.54</td><td>72.32±0.56</td></tr><tr><td>ECML</td><td>76.19±0.12</td><td>69.94±1.51</td><td>66.73±1.46</td><td>63.92±1.44</td><td>60.81±1.50</td><td>58.08±1.34</td></tr><tr><td>CCML</td><td>73.87±1.83</td><td>64.56±1.64</td><td>61.19±1.49</td><td>58.06±1.67</td><td>54.65±1.78</td><td>51.43±2.10</td></tr><tr><td>OURS</td><td>79.41±1.34</td><td>77.53±1.83</td><td>75.47±1.95</td><td>73.77±2.13</td><td>72.04±2.09</td><td>70.14±2.27</td></tr><tr><td rowspan="7">LANDUSE</td><td>DCP-CG</td><td>75.74±0.98</td><td>72.29±1.95</td><td>68.10±1.24</td><td>64.29±1.33</td><td>60.00±0.96</td><td>55.05±1.66</td></tr><tr><td>QMF</td><td>47.86±2.55</td><td>45.31±2.75</td><td>42.83±2.60</td><td>40.69±2.63</td><td>38.12±2.67</td><td>35.36±2.65</td></tr><tr><td>PDF</td><td>45.17±2.66</td><td>44.76±2.64</td><td>42.79±2.54</td><td>40.69±2.14</td><td>38.62±2.13</td><td>36.21±2.17</td></tr><tr><td>UIMC</td><td>57.95±0.61</td><td>57.05±0.47</td><td>55.81±0.47</td><td>55.52±0.44</td><td>54.71±0.44</td><td>53.26±0.56</td></tr><tr><td>ECML</td><td>60.10±2.01</td><td>49.62±2.36</td><td>46.90±2.23</td><td>45.10±1.96</td><td>42.62±1.73</td><td>39.81±1.27</td></tr><tr><td>CCML</td><td>60.86±1.93</td><td>57.67±1.89</td><td>54.86±2.45</td><td>52.05±2.49</td><td>48.67±2.36</td><td>45.76±1.75</td></tr><tr><td>OURS</td><td>76.71±0.46</td><td>73.21±1.83</td><td>71.07±1.83</td><td>68.88±1.71</td><td>66.38±1.83</td><td>63.86±1.55</td></tr><tr><td rowspan="7">LEAVES</td><td>DCP-CG</td><td>98.19±0.46</td><td>91.06±0.70</td><td>84.25±1.06</td><td>77.81±1.06</td><td>71.56±1.27</td><td>65.38±1.22</td></tr><tr><td>QMF</td><td>95.69±1.25</td><td>91.31±1.20</td><td>87.03±1.72</td><td>82.37±1.97</td><td>78.34±2.43</td><td>74.16±2.08</td></tr><tr><td>PDF</td><td>98.03±0.71</td><td>94.09±1.13</td><td>89.91±1.58</td><td>85.62±2.22</td><td>81.41±2.75</td><td>77.62±2.42</td></tr><tr><td>UIMC</td><td>95.31±0.71</td><td>94.41±0.83</td><td>93.81±0.80</td><td>93.03±0.77</td><td>92.03±0.82</td><td>91.66±0.97</td></tr><tr><td>ECML</td><td>92.53±1.94</td><td>88.69±1.18</td><td>84.03±1.65</td><td>79.62±1.30</td><td>75.34±1.89</td><td>72.94±1.89</td></tr><tr><td>CCML</td><td>97.72±0.92</td><td>79.66±1.84</td><td>74.66±1.57</td><td>69.72±1.37</td><td>65.13±1.82</td><td>60.03±1.99</td></tr><tr><td>OURS</td><td>99.78±0.27</td><td>98.59±0.64</td><td>97.56±0.80</td><td>96.12±1.02</td><td>94.88±1.24</td><td>93.69±1.17</td></tr><tr><td rowspan="7">PIE</td><td>DCP-CG</td><td>90.59±1.99</td><td>83.24±3.73</td><td>77.50±3.34</td><td>71.32±2.98</td><td>66.18±2.94</td><td>61.18±3.40</td></tr><tr><td>QMF</td><td>92.06±1.64</td><td>88.75±1.95</td><td>85.22±2.04</td><td>81.62±2.42</td><td>78.46±3.19</td><td>75.51±2.61</td></tr><tr><td>PDF</td><td>92.57±1.66</td><td>89.71±1.83</td><td>86.32±2.19</td><td>83.38±2.55</td><td>80.07±3.13</td><td>77.35±3.29</td></tr><tr><td>UIMC</td><td>91.69±2.16</td><td>90.99±2.16</td><td>90.49±1.93</td><td>90.40±2.08</td><td>90.18±2.25</td><td>89.25±2.22</td></tr><tr><td>ECML</td><td>94.71±0.02</td><td>87.06±2.77</td><td>84.49±2.64</td><td>82.12±2.33</td><td>79.49±2.43</td><td>76.63±3.21</td></tr><tr><td>CCML</td><td>93.97±1.67</td><td>88.53±2.08</td><td>84.56±1.89</td><td>81.62±2.85</td><td>78.31±3.48</td><td>75.44±3.53</td></tr><tr><td>OURS</td><td>96.18±1.24</td><td>93.97±1.27</td><td>92.72±1.59</td><td>91.32±1.57</td><td>90.07±1.84</td><td>89.41±1.68</td></tr></table>

Table 11. We add unaligned views on 50% modalities of the test sets in different proportions. The means and standard deviations of classification accuracies (%) over ten runs are reported. The best results are highlighted in boldface. 

<table><tr><td>DATASET</td><td>METHOD</td><td>0%</td><td>10%</td><td>20%</td><td>30%</td><td>40%</td><td>50%</td></tr><tr><td rowspan="7">HW</td><td>DCP-CG</td><td>99.00±0.47</td><td>98.45±0.37</td><td>98.35±0.34</td><td>98.30±0.29</td><td>98.05±0.33</td><td>97.95±0.43</td></tr><tr><td>QMF</td><td>98.72±0.48</td><td>98.42±0.46</td><td>98.38±0.44</td><td>98.30±0.42</td><td>98.12±0.46</td><td>97.78±0.59</td></tr><tr><td>PDF</td><td>98.40±0.37</td><td>97.50±0.43</td><td>96.72±0.55</td><td>95.55±0.58</td><td>94.50±0.80</td><td>93.60±0.71</td></tr><tr><td>UIMC</td><td>98.25±0.00</td><td>98.20±0.19</td><td>97.88±0.13</td><td>97.72±0.11</td><td>97.72±0.11</td><td>97.72±0.11</td></tr><tr><td>ECML</td><td>98.72±0.39</td><td>97.48±0.71</td><td>96.92±0.81</td><td>96.02±0.72</td><td>95.45±0.95</td><td>94.60±0.77</td></tr><tr><td>CCML</td><td>97.60±0.63</td><td>96.23±0.75</td><td>95.40±1.02</td><td>94.40±1.17</td><td>93.40±1.41</td><td>92.25±1.53</td></tr><tr><td>OURS</td><td>99.20±0.36</td><td>99.13±0.28</td><td>99.10±0.28</td><td>99.05±0.33</td><td>99.05±0.31</td><td>99.10±0.32</td></tr><tr><td rowspan="7">MSRC</td><td>DCP-CG</td><td>95.24±3.69</td><td>94.71±3.50</td><td>94.71±2.78</td><td>93.29±3.56</td><td>93.29±3.56</td><td>92.33±4.10</td></tr><tr><td>QMF</td><td>97.86±1.28</td><td>97.14±1.78</td><td>96.67±2.18</td><td>95.95±2.14</td><td>95.48±1.98</td><td>95.00±2.70</td></tr><tr><td>PDF</td><td>97.14±1.78</td><td>95.71±1.78</td><td>94.76±2.08</td><td>92.86±2.61</td><td>92.38±3.16</td><td>91.43±3.23</td></tr><tr><td>UIMC</td><td>98.81±1.19</td><td>98.05±1.17</td><td>98.05±1.17</td><td>96.57±1.17</td><td>96.57±1.17</td><td>91.90±1.17</td></tr><tr><td>ECML</td><td>94.05±1.60</td><td>93.57±3.02</td><td>93.33±2.78</td><td>92.62±2.70</td><td>90.95±2.97</td><td>89.29±2.44</td></tr><tr><td>CCML</td><td>96.90±2.39</td><td>94.29±2.86</td><td>94.05±3.24</td><td>92.38±3.66</td><td>91.90±3.40</td><td>91.67±3.41</td></tr><tr><td>OURS</td><td>99.76±0.75</td><td>99.76±0.71</td><td>99.76±0.71</td><td>99.76±0.71</td><td>99.76±0.71</td><td>99.76±0.71</td></tr><tr><td rowspan="7">NUSOBJ</td><td>DCP-CG</td><td>43.65±1.10</td><td>43.22±0.95</td><td>42.56±1.05</td><td>41.96±1.01</td><td>41.42±1.04</td><td>40.86±1.08</td></tr><tr><td>QMF</td><td>45.41±0.43</td><td>45.22±0.40</td><td>45.15±0.43</td><td>44.94±0.55</td><td>44.76±0.58</td><td>44.72±0.59</td></tr><tr><td>PDF</td><td>46.78±0.33</td><td>46.50±0.29</td><td>46.24±0.44</td><td>45.92±0.46</td><td>45.58±0.49</td><td>45.35±0.46</td></tr><tr><td>UIMC</td><td>43.42±0.12</td><td>43.14±0.15</td><td>42.76±0.19</td><td>42.62±0.12</td><td>42.21±0.17</td><td>41.78±0.14</td></tr><tr><td>ECML</td><td>42.62±0.42</td><td>41.92±0.59</td><td>41.29±0.57</td><td>41.03±0.58</td><td>40.71±0.58</td><td>40.53±0.52</td></tr><tr><td>CCML</td><td>41.43±0.71</td><td>40.40±0.42</td><td>39.98±0.55</td><td>39.54±0.57</td><td>38.92±0.64</td><td>38.43±0.70</td></tr><tr><td>OURS</td><td>48.23±0.42</td><td>47.78±0.58</td><td>47.62±0.50</td><td>47.40±0.54</td><td>47.07±0.54</td><td>46.86±0.49</td></tr><tr><td rowspan="7">FASHION</td><td>DCP-CG</td><td>98.11±0.23</td><td>96.00±1.15</td><td>94.71±1.56</td><td>93.01±2.06</td><td>91.33±2.58</td><td>89.93±3.00</td></tr><tr><td>QMF</td><td>98.93±0.32</td><td>97.77±0.32</td><td>96.63±0.24</td><td>95.48±0.33</td><td>94.34±0.42</td><td>93.16±0.39</td></tr><tr><td>PDF</td><td>98.95±0.19</td><td>97.17±0.23</td><td>95.50±0.36</td><td>93.77±0.54</td><td>91.84±0.92</td><td>89.68±1.03</td></tr><tr><td>UIMC</td><td>98.13±0.13</td><td>96.12±0.19</td><td>93.75±0.23</td><td>91.85±0.21</td><td>89.90±0.27</td><td>88.12±0.33</td></tr><tr><td>ECML</td><td>97.93±0.35</td><td>94.43±0.32</td><td>91.20±0.39</td><td>87.91±0.51</td><td>84.58±0.60</td><td>81.49±0.68</td></tr><tr><td>CCML</td><td>95.16±0.41</td><td>92.55±0.32</td><td>89.76±0.24</td><td>86.81±0.62</td><td>84.08±0.85</td><td>81.45±1.07</td></tr><tr><td>OURS</td><td>98.96±0.25</td><td>98.57±0.21</td><td>98.18±0.24</td><td>97.78±0.19</td><td>97.32±0.21</td><td>96.88±0.23</td></tr><tr><td rowspan="7">SCENE</td><td>DCP-CG</td><td>77.79±1.73</td><td>74.27±0.64</td><td>72.33±0.69</td><td>69.90±1.29</td><td>67.56±1.56</td><td>65.04±1.51</td></tr><tr><td>QMF</td><td>68.58±1.49</td><td>66.83±1.30</td><td>65.22±1.11</td><td>63.57±1.01</td><td>61.67±0.97</td><td>60.07±0.99</td></tr><tr><td>PDF</td><td>70.25±1.21</td><td>67.87±1.17</td><td>65.54±1.12</td><td>63.14±0.98</td><td>60.66±1.00</td><td>58.13±1.36</td></tr><tr><td>UIMC</td><td>77.70±0.00</td><td>74.86±0.46</td><td>72.85±0.45</td><td>70.79±0.45</td><td>68.74±0.43</td><td>66.12±0.45</td></tr><tr><td>ECML</td><td>76.19±0.12</td><td>71.43±1.31</td><td>69.86±1.16</td><td>68.26±1.17</td><td>66.73±1.16</td><td>65.37±1.22</td></tr><tr><td>CCML</td><td>73.87±1.83</td><td>65.60±1.60</td><td>63.31±1.55</td><td>61.43±1.51</td><td>59.16±1.57</td><td>57.05±1.53</td></tr><tr><td>OURS</td><td>79.41±1.34</td><td>78.19±1.43</td><td>76.98±1.52</td><td>75.80±1.39</td><td>74.68±1.30</td><td>73.99±1.15</td></tr><tr><td rowspan="7">LANDUSE</td><td>DCP-CG</td><td>75.74±0.98</td><td>73.33±2.03</td><td>69.57±1.12</td><td>66.71±1.56</td><td>63.00±1.76</td><td>59.95±1.69</td></tr><tr><td>QMF</td><td>47.86±2.55</td><td>46.60±2.61</td><td>45.10±2.62</td><td>44.02±2.56</td><td>42.67±2.18</td><td>41.38±2.12</td></tr><tr><td>PDF</td><td>45.17±2.66</td><td>44.50±2.49</td><td>42.62±2.33</td><td>40.64±2.31</td><td>38.88±1.95</td><td>37.07±2.12</td></tr><tr><td>UIMC</td><td>57.95±0.61</td><td>56.40±0.58</td><td>53.45±0.56</td><td>51.83±0.41</td><td>49.36±0.41</td><td>47.30±0.63</td></tr><tr><td>ECML</td><td>60.10±2.01</td><td>52.17±1.18</td><td>50.79±1.58</td><td>49.62±1.69</td><td>48.00±1.38</td><td>46.76±1.89</td></tr><tr><td>CCML</td><td>60.86±1.93</td><td>59.29±1.87</td><td>57.80±2.29</td><td>55.48±2.96</td><td>54.17±2.70</td><td>53.21±3.07</td></tr><tr><td>OURS</td><td>76.71±0.46</td><td>74.52±1.91</td><td>73.38±1.81</td><td>72.10±1.74</td><td>70.98±1.61</td><td>69.93±1.47</td></tr><tr><td rowspan="7">LEAVES</td><td>DCP-CG</td><td>98.19±0.46</td><td>95.06±1.69</td><td>91.19±1.30</td><td>88.00±1.84</td><td>84.69±2.19</td><td>81.56±2.01</td></tr><tr><td>QMF</td><td>95.69±1.25</td><td>91.50±1.32</td><td>87.66±1.71</td><td>83.25±1.81</td><td>79.06±2.11</td><td>75.38±2.24</td></tr><tr><td>PDF</td><td>98.03±0.71</td><td>92.78±1.21</td><td>87.75±1.36</td><td>82.34±1.02</td><td>77.56±0.94</td><td>72.56±0.84</td></tr><tr><td>UIMC</td><td>95.31±0.71</td><td>89.72±0.65</td><td>86.47±0.74</td><td>83.06±0.87</td><td>80.41±0.94</td><td>77.09±0.97</td></tr><tr><td>ECML</td><td>92.53±1.94</td><td>89.84±1.95</td><td>86.41±1.84</td><td>83.09±1.82</td><td>79.62±2.09</td><td>76.19±2.25</td></tr><tr><td>CCML</td><td>97.72±0.92</td><td>81.09±2.22</td><td>78.06±2.01</td><td>74.59±2.01</td><td>71.34±1.69</td><td>68.12±1.71</td></tr><tr><td>OURS</td><td>99.78±0.27</td><td>98.94±0.72</td><td>97.84±1.14</td><td>96.72±1.55</td><td>95.81±1.51</td><td>94.69±1.26</td></tr><tr><td rowspan="7">PIE</td><td>DCP-CG</td><td>90.59±1.99</td><td>85.15±3.06</td><td>81.03±2.05</td><td>76.76±1.58</td><td>72.65±2.43</td><td>68.68±1.36</td></tr><tr><td>QMF</td><td>92.06±1.64</td><td>90.51±1.81</td><td>88.82±2.37</td><td>86.40±2.68</td><td>84.34±2.61</td><td>81.99±2.98</td></tr><tr><td>PDF</td><td>92.57±1.66</td><td>89.19±1.44</td><td>85.96±2.07</td><td>82.06±1.89</td><td>79.12±2.44</td><td>75.00±2.81</td></tr><tr><td>UIMC</td><td>91.69±2.16</td><td>85.96±2.40</td><td>82.72±2.33</td><td>77.06±2.79</td><td>73.09±2.72</td><td>68.24±2.82</td></tr><tr><td>ECML</td><td>94.71±0.02</td><td>88.75±2.51</td><td>86.69±2.47</td><td>84.56±2.28</td><td>82.50±1.97</td><td>80.15±2.55</td></tr><tr><td>CCML</td><td>93.97±1.67</td><td>90.22±1.86</td><td>88.24±1.64</td><td>86.03±1.77</td><td>84.26±1.62</td><td>82.65±1.95</td></tr><tr><td>OURS</td><td>96.18±1.24</td><td>94.93±1.37</td><td>93.38±1.95</td><td>91.47±1.92</td><td>89.85±2.13</td><td>87.94±2.38</td></tr></table>

Table 12. Classification accuracy (%) of ECML and our FUML with varying noise views on the PIE dataset. ECML fuses views from the first view, making it sensitive to the order of the noisy views. 

<table><tr><td>METHODS</td><td>NOISE VIEW</td><td>1+2+3</td><td>1+3+2</td><td>2+1+3</td><td>2+3+1</td><td>3+1+2</td><td>3+2+1</td><td> $\Delta\%$ </td></tr><tr><td rowspan="3">ECML</td><td>1</td><td>87.13</td><td>81.69</td><td>87.13</td><td>76.91</td><td>81.76</td><td>77.21</td><td>10.22</td></tr><tr><td>2</td><td>91.18</td><td>82.13</td><td>90.96</td><td>91.25</td><td>82.06</td><td>91.25</td><td>9.19</td></tr><tr><td>3</td><td>86.76</td><td>88.97</td><td>86.54</td><td>90.81</td><td>88.90</td><td>91.10</td><td>4.56</td></tr><tr><td rowspan="3">OURS</td><td>1</td><td colspan="6">93.24</td><td>0</td></tr><tr><td>2</td><td colspan="6">96.18</td><td>0</td></tr><tr><td>3</td><td colspan="6">92.65</td><td>0</td></tr></table>

# C.8. Conflict Visualization

Figure 10 presents the conflict on the HandWritten dataset with six views. To introduce conflicts, we modify the content in the third view to other categories, resulting in misalignment with the other views. The Figure 10 (a) and (b) depict the conflict of normal and conflicting instances, respectively. The results show that FUML effectively captures and quantifies conflict between views, further validating its reliability.

![](images/367b18dafa7ee5964b340feae14268e19664a62d46c3955d0b3886612e91ed78.jpg)

<details>
<summary>heatmap</summary>

| View | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| 1 | 0.00 | 0.31 | 0.00 | 0.27 | 0.00 | 0.28 |
| 2 | 0.31 | 0.00 | 0.31 | 0.00 | 0.31 | 0.01 |
| 3 | 0.00 | 0.31 | 0.00 | 0.27 | 0.00 | 0.28 |
| 4 | 0.27 | 0.00 | 0.27 | 0.00 | 0.27 | 0.01 |
| 5 | 0.00 | 0.31 | 0.00 | 0.27 | 0.00 | 0.28 |
| 6 | 0.28 | 0.01 | 0.28 | 0.01 | 0.28 | 0.00 |
</details>

(a) Normal instance

![](images/818ae39660cf435c2c0ddd14a57d430ceb3fd78db3fd83353e77b47b7f2580fc.jpg)

<details>
<summary>heatmap</summary>

| View | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| 1 | 0.00 | 0.31 | 1.00 | 0.27 | 0.00 | 0.28 |
| 2 | 0.31 | 0.00 | 0.98 | 0.00 | 0.31 | 0.01 |
| 3 | 1.00 | 0.98 | 0.00 | 1.00 | 1.00 | 1.00 |
| 4 | 0.27 | 0.00 | 1.00 | 0.00 | 0.27 | 0.01 |
| 5 | 0.00 | 0.31 | 1.00 | 0.27 | 0.00 | 0.28 |
| 6 | 0.28 | 0.01 | 1.00 | 0.01 | 0.28 | 0.00 |
</details>

(b) Conflicting instance   
Figure 10. Conflict visualization.

# C.9. Adversarial Noise Effect Analysis

In this section, we present additional experimental results under adversarial noise attacks. First, to evaluate the performance of our FUML under adversarial noise, we add projected gradient descent (PGD) adversarial noise attacks (Madry et al., 2018) with different maximum perturbation magnitudes (eps) to the test set of the Fashion dataset. The results in Table 13 show FUML's superior resistance to adversarial noise attacks. Second, to further evaluate the effectiveness of the uncertainty estimation mechanism of FUML under adversarial noise, we perform an OOD task on the Fashion dataset, using normal test sets as ID and PGD-attacked sets (eps=0.10) as OOD. Evaluated by FPR95 (lower is better), FUML achieved 0.68, outperforming ETMC and ECML (both 1.00). To sum up, under adversarial noise, the proposed FUML is superior in both classification accuracy and uncertainty estimation.

# D. Limitations

Even though the proposed FUML outperforms existing multi-view classification methods in terms of both performance and reliability, there are still some potential limitations. For instance, FUML's multi-view fusion weights incorporate both uncertainty and conflict through multiplication. Although extensive experiments have demonstrated the effectiveness of this

Table 13. Classification accuracy (%) of PDF, ECML, and our FUML under PGD adversarial noise attacks with different eps on the Fashion dataset. The best results are highlighted in boldface. 

<table><tr><td>METHOD\EPS</td><td>0</td><td>0.05</td><td>0.10</td></tr><tr><td>PDF</td><td>98.95 ± 0.19</td><td>22.07 ± 0.90</td><td>13.54 ± 0.89</td></tr><tr><td>ECML</td><td>97.93 ± 0.35</td><td>52.58 ± 0.51</td><td>42.74 ± 1.35</td></tr><tr><td>OURS</td><td>98.96±0.25</td><td>94.45±0.18</td><td>93.40±0.19</td></tr></table>

technique, it lacks corresponding theoretical guarantees. Therefore, it is crucial to explore new multi-view fusion techniques based on uncertainty and conflict from a theoretical standpoint. We hope that our study can serve as a valuable baseline for future research in multi-view learning and uncertainty estimation.