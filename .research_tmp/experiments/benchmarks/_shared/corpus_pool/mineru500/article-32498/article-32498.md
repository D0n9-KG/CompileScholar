# AIF-SFDA: Autonomous Information Filter-driven Source-Free Domain Adaptation for Medical Image Segmentation

Haojin Li $^{1,2}$ , Heng Li $^{1,*}$ , Jianyu Chen $^{1,2}$ , Rihan Zhong $^{1,2}$ , Ke Niu $^{3}$ , Huazhu Fu $^{4}$ , Jiang Liu $^{1,2,*}$

$^{1}$ Research Institute of Trustworthy Adaptive Systems, Southern University of Science and Technology

$^{2}$ Department of Computer Science and Engineering, Southern University of Science and Technology $^{3}$ Beijing Information Science & Technology University

$^{4}$ Institute of High Performance Computing, Agency for Science, Technology and Research

# Abstract

Decoupling domain-variant information (DVI) from domain-invariant information (DII) serves as a prominent strategy for mitigating domain shifts in the practical implementation of deep learning algorithms. However, in medical settings, concerns surrounding data collection and privacy often restrict access to both training and test data, hindering the empirical decoupling of information by existing methods. To tackle this issue, we propose an Autonomous Information Filter-driven Source-free Domain Adaptation (AIF-SFDA) algorithm, which leverages a frequency-based learnable information filter to autonomously decouple DVI and DII. Information Bottleneck (IB) and Self-supervision (SS) are incorporated to optimize the learnable frequency filter. The IB governs the information flow within the filter to diminish redundant DVI, while SS preserves DII in alignment with the specific task and image modality. Thus, the autonomous information filter can overcome domain shifts relying solely on target data. A series of experiments covering various medical image modalities and segmentation tasks were conducted to demonstrate the benefits of AIF-SFDA through comparisons with leading algorithms and ablation studies. The code is available at https://github.com/JingHuaMan/AIF-SFDA.

# Introduction

In recent years, there has been considerable advancement in the field of medical image segmentation methods based on deep learning (DL) (Li et al. 2024a). However, real-world scenarios frequently involve open datasets where the test data (target domain) is unseen and likely exhibits domain shifts in comparison to the training data (source domain) (Guan and Liu 2021), attributed to practical variations in acquisition devices, patient demographics, image quality, and other variables. These domain shifts can significantly affect the performance of segmentation models on target domains (Li et al. 2023a). Therefore, effectively transferring segmentation models from the source domain to the target domains is both logical and essential to enhance the utilization of DL algorithms.

![](images/61bbbc117af530b7038a1d65d657ac00f0289edfc676c2ed58744acd52cf91d0.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Autonomous Filtering"] --> B["IB-constrained DVI Reduction"]
    A --> C["SS-constrained DII Preservation"]
    B --> D["Information Filter"]
    C --> D
    D --> E["Domain Variant Information (DVI)"]
    D --> F["Domain Invariant Information (DII)"]
```
</details>

Figure 1: To develop an autonomous information filter in SFDA scenarios, we utilize IB-constrained mutual information constraint to reduce DVI in the image information while preserving DII through SS-constrained guidance.

To address this issue, Unsupervised Domain Adaptation (UDA) has been proposed to generalize models by leveraging labeled source data in conjunction with unlabeled target data. A prominent strategy (Yang and Soatto 2020; Liu et al. 2021) within UDA revolves around decoupling information into domain-variant and domain-invariant information (DVI & DII), followed by the compression of DVI and the enhancement of DII to enable robust inference on novel data.

Specifically, configurable information filters based on frequency filtering are applied to process images or features, selecting DVI and DII from various frequency components.

In these type of approach, filter configurations are typically derived by identifying characteristics and commonalities between the source and target domains in the frequency spectrum. Configurations can be obtained by empirically comparing frequency features (Liu et al. 2023) or through autonomous optimization guided by task-related losses (Lin et al. 2023). Nevertheless, jointly accessing both source and target domains leads to concerns involving data collection and privacy, often unacceptable in various practical contexts, especially medical scenarios.

Accordingly, source-free domain adaptation (SFDA) (Li et al. 2024b) becomes imperative to enable the adaptation of pre-trained models solely using unlabeled target data. However, challenges emerge in decoupling DVI and DII in SFDA settings. 1) The absence of labeled source data in SFDA re-

sults in a lack of guidance for decoupling DVI and DII. 2) The preservation of DII is complicated when relying solely on unseen and unlabeled target data. 3) Frequency filter-driven algorithms often rely on empirical filtering configuration, which is impractical in SFDA.

To facilitate SFDA in medical image segmentation, we propose an Autonomous Information Filter (AIF-SFDA) autonomously aimed at decoupling DVI and DII for adaptation during the inference phase. The AIF-SFDA enables learnable frequency filters through IB and SS, autonomously reducing DVI and preserving DII relying solely on target data. Specifically, the IB regulates information flow within the filter to eliminate redundant DVI, while SS is derived from confidence-aware pseudo-labeling and consistency constraints to guide DII extraction. Our main contributions can be summarized as follows:

- We propose a source-free domain adaptation algorithm for medical image segmentation termed AIF-SFDA to adaptively decouple DVI and DII using learnable frequency filters.   
- An autonomous information filter is constructed based on learnable frequency filters to reduce DVI and preserve DII, exclusively leveraging target data.   
- The IB and SS are implemented to regulate the learnable frequency filters, enabling the adaptive decoupling of domain information for SFDA.   
- Cross-domain experiments were conducted across diverse medical image modalities and segmentation tasks to assess the efficacy of AIF-SFDA through comparisons with state-of-the-art algorithms and ablation studies.

# Related Work

# Source-free Domain Adaptation

UDA algorithms have been designed to tackle domain shifts effectively by utilizing both the source and target data concurrently. However, due to concerns related to data access and privacy, SFDA has emerged as an alternative approach. SFDA allows the transfer of knowledge from a pre-trained source model to unlabeled target data without the need to access the source domain (Li et al. 2024b). SFDA methods can be broadly categorized into data-based and model-based approaches. Data-based methods process target data using knowledge from the source model to reduce discrepancies with the source domain, such as selecting target data to generate surrogate source data (Ye et al. 2021) or using image translation to adapt target domain images to a source-like style (Yang et al. 2022). Model-based methods primarily involve self-supervised tasks on intermediate features and segmentation outputs, such as contrastive learning (Zhang et al. 2024), pseudo-label supervision (Li et al. 2023b), or regularization constraints like entropy minimization (Fleuret et al. 2021).

# Frequency-based Domain Information Decoupling

Frequency domain methods established by operations such as DCT and DFT enable data transformation between the spatial and frequency domains. Past studies on transfer learning have demonstrated that different frequency components exhibit varying domain-related characteristics, thus facilitating domain information decoupling (Xu et al. 2021; Liu et al. 2023). A common approach involves empirically designing filters based on the a priori knowledge contained in labeled source data. For instance, splitting DII and DVI domains by high/low frequencies with a fixed threshold (Yang and Soatto 2020; Liu et al. 2021; Li et al. 2022) or selecting the most suitable components for the downstream task through spectral analysis (Huang et al. 2021). These assumptions are typically derived from the labeled source data. Another approach involves using adaptive filters with learnable parameters to guide DII extraction through supervised task-related loss (Lin et al. 2023). However, these methods that explicitly include supervised learning are not directly applicable in the SFDA scenario. Therefore, it is necessary to design a domain decoupling mechanism that does not require access to source data.

# Information Bottleneck in Deep learning

IB theory is an information-theoretic approach (Tishby, Pereira, and Bialek 2000), aimed at obtaining compact data representations by reducing task-irrelevant parts of the data. For a certain model, IB achieves this by minimizing the mutual information (MI) between the input and intermediate variables, while maximizing MI between the intermediate variables and the output. Recently, IB theory has been used to provide interpretable analyses for DL methods due to its clear mathematical framework. (Tishby and Zaslavsky 2015) viewed information extraction in multilayer networks as deriving the minimum sufficient statistic, while (Kawaguchi et al. 2023) showed that IB can control the generalization error of DL methods. Studies like (Alemi et al. 2016) explicitly utilize the IB principle, implementing a variational approximation with a variational network and showing high generalization performance. In this work, IB is applied to the information filter process to reduce DVI in the filtered image through MI constraints, thereby aiding the extraction of DII.

# Method

# Overview

To enhance the cross-domain performance of the segmentation model, SFDA consists of two distinct stages. In the source domain pre-training stage, given a source dataset $X_{S} = \{(x_{i}^{s},y_{i}^{s}),i = 1,\dots ,N_{S}\}$ that includes images and corresponding labels, a source model $g(\cdot)$ is well-trained to obtain the parameter $\theta_o$ , i.e., $\theta_o = \arg \min_{\theta_o}\frac{1}{N_S}\sum_{i = 1}^{N^s}l_s(g(x_i^s;\theta_o),y_i^s)$ , where $l_{s}$ denotes a certain supervised segmentation loss. In the target domain adaptation stage, given an image-only unlabelled target dataset $X_{T} = \{(x_{i}^{t}),i = 1,\dots ,N_{T}\}$ as well as the source model, it is necessary to improve the segmentation model's generalizability on the target data in the absence of direct access to source data.

The overall architecture of the AIF-SFDA algorithm we designed is shown in Figure 2. To effectively boost the domain information decoupling through image transformation,

![](images/d132f069658c8eb0184a25a606e4579a2cc49f7e9452b0f08370e46758e8a757.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["(a) Frequency-based Information Filter"] --> B["Attention Module, θf"]
    B --> C["TCI"]
    C --> D["x^t"]
    D --> E["Shared Encoder, θe"]
    E --> F["z^t"]
    F --> G["Teacher Decoder, θt, EMA from θs"]
    G --> H["p^t"]
    I["(b) IB-constrained DVI Reduction"] --> J["MI Minimization, ℒMI→θf"]
    J --> K["SS-constrained DII Preservation"]
    K --> L["Consistency Maximization, ℒCon→θe"]
    L --> M["Confidence-Aware PL Sup. ℒPL→θf, θe, θs"]
    M --> N["p^s"]
    O["DVI"] --> P["DII"]
    P --> Q["Loss"]
    R["x^f"] --> S["Shared Encoder, θe"]
    S --> T["z^s"]
    T --> U["Student Decoder, θs"]
```
</details>

Figure 2: The architecture of our proposed AIF-SFDA.

we incorporate a frequency-based information filter $f(\cdot)$ that autonomously decouples DVI and DII in the target domain image $x^{t}$ according to task type and instance characteristics. To guarantee that the information filter works robustly on the target data, IB-constrained DVI reduction is firstly achieved by optimizing mutual information minimization loss $L_{MI}$ based on the image features $z^{s}$ and $z^{t}$ before and after filtering. Furthermore, SS-constrained DII is preserved through confidence-aware pseudo-label supervision loss $L_{PL}$ and adversarial feature consistency loss $L_{Con}$ , ultimately enhancing the generalization performance of the segmentation model.

# Frequency-based Information Filter

To adaptively compare and select domain variant and invariant information, an adaptive filter should possess two properties: 1) learnable parameters that can be easily optimized, and 2) the ability to adaptively process image information based on the input image. To achieve these two goals, our proposed adaptive filtering can be briefly described as applying spatial attention to the spectrum obtained after the DCT transformation. Denote the 2D DCT process as $\mathcal{F}(\cdot)$ , and the basis functions are:

$$
B _ {u, v} ^ {i, j} = \cos (\frac {\pi (2 i + 1) u}{2 H}) \cos (\frac {\pi (2 j + 1) v}{2 W}). \tag {1}
$$

For each channel of $x^{t}$ , perform the 2D DCT process (assuming $x^{t}$ is a grayscale image):

$$
\mathcal {F} (x ^ {t}) _ {u, v} = \sum_ {i = 0} ^ {H - 1} \sum_ {j = 0} ^ {W - 1} x _ {i, j} ^ {t} B _ {u, v} ^ {i, j}, \tag {2}
$$

where $(u,v)$ are the indices on the spectrum, $\mathcal{F}(x^t)\in \mathbb{R}^{H,W}$ . Then, input $\mathcal{F}(x^t)$ into the attention module $M_{\theta_f}(\cdot)$ to obtain the attention map. The adaptive filtering process is formulated as:

$$
x ^ {f} = f _ {\theta_ {f}} (x ^ {t}) = \mathcal {F} ^ {- 1} (M _ {\theta_ {f}} (\mathcal {F} (x)) \odot \mathcal {F} (x)), \tag {3}
$$

where $\mathcal{F}^{-1}(\cdot)$ denotes the inverse DCT transform, and $\odot$ represents the Hadamard product. Through the above process, a filter capable of self-adjusting to select and remove information based on the input image is established.

# Domain Information Decoupling driven by Adaptive Filtering

IB-constrained DVI Reduction In the flow of image segmentation algorithms that incorporate an information filter, $x^{f}$ can be interpreted as an intermediate variable. Referring to the common practice of IB theory, we use information-theoretic methods to constrain the feature embeddings of $x^{t}$ and $x^{f}$ in order to modulate the adaptive filter. This ensures that $x^{f}$ approximates as closely as possible the task-relevant minimal sufficient statistics of $x^{t}$ , thereby reducing the unwanted DVI. This process is described mathematically by IB as:

$$
\min _ {p (x ^ {f} | x ^ {t})} [ I (x ^ {f}; x ^ {t}) - \beta I (\hat {y}; x ^ {f}) ], \tag {4}
$$

where $\beta$ is a Lagrange multiplier.

Considering the difficulty of quantifying the latter term in Eq. 4, we constrain only the former term. Since the computational complexity is limited by the high dimension of the data if MI is computed directly among images, we constrain it indirectly by the MI between the corresponding feature embeddings of $x^{t}$ and $x^{f}$ outputted by a shared encoder with parameter $\theta_{e}$ . In other words, we aim to reduce $I(z^{s};z^{t})$ . According to the definition in (Cheng et al. 2020), $I(z^{s};z^{t})$ has the following upper bound when $p(z^{t}|z^{s})$ is known:

$$
\begin{array}{l} I (z ^ {s}; z ^ {t}) \leq \mathbb {E} _ {p (z ^ {s}, z ^ {t})} \left[ \log p (z ^ {t} \mid z ^ {s}) \right] \\ - \mathbb {E} _ {p (z ^ {s}) p (z ^ {t})} \left[ \log p (z ^ {t} \mid z ^ {s}) \right]. \tag {5} \\ \end{array}
$$

However, since $p(z^{t}|z^{s})$ is intractable, we approximate it with a variational distribution $q_{\theta_{q}}(z^{t}|z^{s})$ . When the approximation is good, we can reduce $I(z^{s};z^{t})$ by optimizing the

following loss function:

$$
\mathcal {L} _ {M I} = \frac {1}{N} \sum_ {i = 1} ^ {N} [ \log q _ {\theta_ {q}} (z _ {i} ^ {t} | z _ {i} ^ {s}) - \frac {1}{N} \sum_ {j = 1} ^ {N} \log q _ {\theta_ {q}} (z _ {j} ^ {t} | z _ {i} ^ {s}) ], \tag {6}
$$

To ensure that $q(z^{t}|z^{s};\theta_{q})$ can approximate $p(z^{t}|z^{s})$ , we need to include the negative log-likelihood loss function:

$$
\mathcal {L} _ {L i} = - \frac {1}{N} \sum_ {i = 1} ^ {N} \log q _ {\theta_ {q}} (z ^ {t} | z ^ {s}), \tag {7}
$$

The proposed IB-constrained DVI Reduction prompts $x^{f}$ to remove the useless information in $x^{t}$ , but it does not guarantee the retention of the DII in the original data. Therefore, we need an information preservation mechanism to avoid the loss of meaningful information.

SS-constrained DII Preservation To preserve DII in domain information decoupling, AIF-SFDA employs self-supervised learning by fully utilizing unlabelled target data, using pseudo-label (PL) supervision and feature consistency constraints.

PL self-supervision helps the information filter learn the most task-specific DII, while also enabling the segmentation model to adapt to changes in the filtered image. A teacher-student architecture is employed to generate PLs while retaining source domain knowledge, with the teacher decoder parameterized by $\theta_{t}$ and the student decoder by $\theta_{s}$ . Denote the outputs of teacher and student decoders as $p^{t}$ and $p^{s}$ , the PL is $y^{t} = \arg\max_{c} p_{c}^{t}$ , and the confidence associated with the pseudo-label is $\phi(y^{t}) = \max_{c} p_{c}^{t}$ .

To address the interference of low-confidence pixels in PLs on the optimization of the information filter and segmentation models, we implement a PL filtering mechanism based on a confidence threshold. This approach aims to mitigate the impact of potentially incorrect pseudo-labels. The PL supervised loss function is defined as follows:

$$
\mathcal {L} _ {P L} = \frac {1}{H W} \sum_ {h, w} ^ {H, W} \mathbb {1} [ \phi (\hat {y} _ {h, w} ^ {t}) > \tau ] l _ {c e} (\hat {y} _ {h, w} ^ {t}, p _ {h, w} ^ {s}), \tag {8}
$$

where $\mathbb{1}(\cdot)$ denotes the indicator function, $l_{ce}(\cdot,\cdot)$ represents pixel-wise cross entropy, and $\tau$ is the confidence threshold as a hyperparameter.

The distance between feature embeddings $z^{s}$ and $z^{t}$ should be maximized for the information filter, while it should be minimized for the segmentation model, as both embeddings contain the same task-specific semantic information. This dual requirement motivates the design of feature consistency constraints for optimizing the segmentation model encoder, which enhances both the information extraction of the segmentation model and the adversarial optimization of the information filter.

We use the commonly used cosine similarity to implement the consistency constraint:

$$
\mathcal {L} _ {C o n} = \frac {\left\langle z ^ {s} , z ^ {t} \right\rangle}{\left| \left| z ^ {s} \right| \right| \left| \left| z ^ {t} \right| \right|}, \tag {9}
$$

where $\langle\cdot,\cdot\rangle$ and $\|\cdot\|$ denotes inner product and L2 norm respectively.

Algorithm 1: The training procedures of AIF-SFDA

Input: Target dataset $X_{T}$ , source model $g_{\theta_{o}}$ , information filter $f_{\theta_{f}}$ , variational distribution $q_{\theta_{q}}$ , max training iteration number N

1: $\theta_{e},\theta_{t}\gets \theta_{o}$ ▷ Copy to teacher model   
2: $\theta_{e},\theta_{s}\gets \theta_{o}$ ▷ Copy to student model   
3: for iter k = 1 to N do   
4: $x^{s}\sim X_{T}.$ ▷ Sample target data   
5: $x^{f}\gets f_{\theta_{f}}(x^{t}).$ $\triangleright$ Eq. 3   
6: $z^t, p^t \leftarrow g_{\theta_e, \theta_t}(x^t)$ . ▷ Teacher model process $x^t$   
7: $z^{s}, p^{s} \leftarrow g_{\theta_{e}, \theta_{s}}(x^{f})$ . $\triangleright$ Student model process $x^{f}$   
8: Compute $L_{MI}$ , $L_{Li}$ , $L_{Con}$ based on $z^{t}$ and $z^{s}$ .   
9: Compute $\mathcal{L}_{PL}$ based on $p^t$ and $p^s$ .   
10: Update $\theta_{f}$ by $\mathcal{L}_{PL}$ and $\mathcal{L}_{MI}$ . ▷ Eq. 10   
11: Update $\theta_{e},\theta_{s},\theta_{q}$ by $\mathcal{L}_{PL},\mathcal{L}_{Li}$ and $\mathcal{L}_{Con}$ . ▷ Eq. 11   
12: Update $\theta_{t}$ based on $\theta_{s}$ through EMA. ▷ Eq. 12   
13: end for

# Source-Free Domain Adaptation

The complete process of the proposed AIF-SFDA is outlined in Algorithm 1. The optimization process is divided into two steps. First, the information filter is optimized using pseudo-label self-supervision combined with IB-constrained DVI reduction, enabling the filter to autonomously extract DII and eliminate DVI:

$$
\min _ {\theta_ {f}} [ \mathcal {L} _ {P L} + \alpha_ {1} \mathcal {L} _ {M I} ]. \tag {10}
$$

Secondly, we optimize the parameters of the variational distributions in the student model and the MI constraints to assist in the optimization of the information filter and improve model generalization by learning the DII in the filtered image, i.e.:

$$
\min _ {\theta_ {e}, \theta_ {s}, \theta_ {q}} \left[ \mathcal {L} _ {P L} + \alpha_ {2} \mathcal {L} _ {L i} + \alpha_ {3} \mathcal {L} _ {C o n} \right], \tag {11}
$$

where $\alpha_{1}$ , $\alpha_{2}$ , and $\alpha_{3}$ are balancing hyperparameters. After each optimization iteration, we optimize the teacher decoder using the exponential moving average (EMA) based on the student decoder parameter:

$$
\theta_ {t} \leftarrow \eta \theta_ {t} + (1 - \eta) \theta_ {s}, \tag {12}
$$

where $\eta$ is a coefficient ranging between [0, 1].

# Experiments

# Experimental Settings

Datasets and Metrics The datasets involved in this work are shown in Table 1. In the fundus photography retinal vessel segmentation task, all datasets used are publicly available, with DRIVE (Staal et al. 2004) serving as the source domain and AVRDB (Niemeijer et al. 2011), CHASEDB1 (Owen et al. 2009), DRHAGIS (Holm et al. 2017), LESAV (Fraz et al. 2014), STARE (Hoover, Kouznetsova, and Goldbaum 2000) as the target domains. For the ultrasound joint cartilage segmentation task, the private datasets A, B, and C were provided by Southern University of Science and

Technology Hospital, with dataset A used as the source domain. All datasets were randomly divided into training and test sets in a 1:1 ratio. We employed two common segmentation metrics to evaluate the performance of each algorithm: the Dice Similarity Coefficient (DSC) and the Intersection-over-Union (IoU), where higher DSC and IoU indicate better segmentation results.

Implementation Details In the source domain pre-training stage, the segmentation model selects parameters at the optimal epoch based on the test set according to the early stopping mechanism. In the target domain adaptation stage, the model uses the parameters from the last epoch for performance evaluation on the test set. During the training process of both stages, we employ the Adam optimizer with an initial learning rate of 0.001 and a batch size of 2. The model is trained for 400 epochs in the retinal vessel segmentation task and 20 epochs in the ultrasound cartilage segmentation task, with the learning rate uniformly reduced to 0 in the latter half of the epochs. We use a naive U-net (Ronneberger, Fischer, and Brox 2015) as the segmentor, and the attention module $M$ in the information filter is implemented using a three-layer lightweight U-net. The variational distribution model $q$ employs a multivariate Gaussian distribution, parameterized using two 2-layer multi-layer perceptrons, each with a hidden size of 1024. The pseudo-label screening threshold $\tau$ is set to 0.8. The balancing coefficients $\alpha_{1}, \alpha_{2}$ , and $\alpha_{3}$ are set to 0.5, 1, and 1, respectively. The EMA coefficient $\eta$ is set to 0.9995.

# Comparison with State-of-the-Art Methods

In the comparison experiments, we selected nine SOTA segmentation baselines, including two vanilla segmentation algorithms: Rolling-Unet (Liu et al. 2024) and DTM-Former (Wang et al. 2024), three UDA algorithms: DAMAN (Mukherjee et al. 2022), CS-CADA (Gu et al. 2022) and MAAL (Zhou et al. 2023), and four SFDA algorithms: SFODA (Niloy, Bhaumik, and Woo 2024), UPL-SFDA (Wu et al. 2023), UBNA (Klingner et al. 2022) and TSFCT (Li et al. 2023b).

Result for Retinal Vessel Segmentation Table 2 presents the comparative results for the fundus vessel segmentation task, encompassing both the source domain model and the cross-domain performance of each baseline model, alongside our proposed AIF-SFDA. As observed, most baselines exhibit superior generalization on the target domain relative to the source model that solely employs U-net. In general, UDA and SFDA algorithms outperform naive segmentation

<table><tr><td>Task</td><td>Dataset</td><td>Volume</td></tr><tr><td>Retinal Vessel</td><td>DRIVE*, AVRDB, CHASEDB1, DRHAGIS, LES-AV, STARE</td><td>40, 100, 28, 40, 22, 20</td></tr><tr><td>Joint Cartilage</td><td>A*, B, C</td><td>956, 982, 750</td></tr></table>

\* DRIVE and A are used as source domains in the two tasks, respectively.

Table 1: Datasets and their volumes used in this work. methods due to their specialized design for cross-domain segmentation. Among the UDA algorithms, CS-CADA outperforms DAMAN and MAAL, indicating that its approach could be well-suited for the vessel segmentation task. While most SFDA methods enhance cross-domain segmentation performance, with SFODA achieving over a 6% DSC improvement across all datasets compared to the source model, TSFCT also demonstrates negative adaptation, underscoring the inherent challenges of adapting without access to source data. Notably, AIF-SFDA outperforms all baselines in generalizability, thereby validating the efficacy of the proposed autonomous information filtering mechanism.

We show a qualitative comparison of fundus vessel segmentation experiments in Figure 3. It can be seen that the domain differences between the fundus image datasets are mainly in the overall brightness of the images and task-irrelevant noise, which makes the baseline methods prone to ignoring small vessels in regions with uneven brightness and to misclassifying non-vessel noise pixels as false positives. Our proposed AIF-SFDA effectively exploits the frequency domain properties shared by vessel pixels, making the foreground pixels more conspicuous by processing the image with the autonomous information filter, thus improving the accuracy of small vessel segmentation and reducing the interference caused by unseen image noise in the target domains.

Result for Joint Cartilage Segmentation To conduct a more extensive comparison of medical images across multiple modalities, the segmentation results of the ultrasound joint cartilage segmentation task are shown in Table 3. It can be found that the DSC of the source domain model on C is worse than the performance on dataset B, and both UPL-SFDA and TSFCT exhibit negative adaptation, possibly because C differs more from the source domain than B. Notably, TSFCT and SFODA achieve the best DSC in the baseline on B and C, respectively, suggesting that pseudo-labeled SFDA may perform better with smaller domain shifts, while feature-based algorithms are more effective with larger shifts. AIF-SFDA, which integrates both techniques, achieves the optimal DSC on both datasets.

Qualitative results are shown in Figure 4. Generalization errors in cartilage segmentation often arise from contrast and luminance differences between ultrasound datasets, leading to poor continuity in segmentation. In dataset B, uneven image brightness hindered baseline methods from detecting end cartilage pixels, while the darker images in dataset C resulted in higher false negatives for SFDA baselines except for SFODA and UBNA. AIF-SFDA, using its information filter, mitigates the effects of blurring and low luminance, contributing to its strong generalization performance.

# Ablation Study

Ablation Study of Modules Table 4 shows the ablation experiments performed. We combine the MI minimization constraint (MI Min.) in IB-constrained DVI Reduction, the confidence-aware pseudo-label selection mechanism in SS-constrained DII Preservation (PL Sel.), and feature consistency constraints in SS-constrained DII Preservation (Cons.)

<table><tr><td rowspan="2">Algorithm</td><td rowspan="2">SF*</td><td colspan="2">AVRDB</td><td colspan="2">CHASEDB1</td><td colspan="2">DRHAGIS</td><td colspan="2">LESAV</td><td colspan="2">STARE</td></tr><tr><td>DSC↑</td><td>IoU↑</td><td>DSC↑</td><td>IoU↑</td><td>DSC↑</td><td>IoU↑</td><td>DSC↑</td><td>IoU↑</td><td>DSC↑</td><td>IoU↑</td></tr><tr><td>Source</td><td>/</td><td>54.34</td><td>39.47</td><td>52.36</td><td>37.87</td><td>54.65</td><td>39.48</td><td>58.12</td><td>42.18</td><td>58.53</td><td>42.69</td></tr><tr><td>Rolling-Unet</td><td>/</td><td>59.64</td><td>43.23</td><td>59.95</td><td>43.08</td><td>61.91</td><td>44.91</td><td>63.44</td><td>46.51</td><td>64.28</td><td>47.52</td></tr><tr><td>DTMFormer</td><td>/</td><td>60.73</td><td>44.54</td><td>60.69</td><td>44.33</td><td>62.22</td><td>45.76</td><td>63.84</td><td>47.45</td><td>64.64</td><td>48.41</td></tr><tr><td>CS-CADA</td><td>X</td><td>65.16</td><td>48.58</td><td>64.15</td><td>47.48</td><td>69.94</td><td>55.56</td><td>75.22</td><td>60.48</td><td>74.71</td><td>60.26</td></tr><tr><td>DAMAN</td><td>X</td><td>62.97</td><td>46.96</td><td>61.62</td><td>44.54</td><td>63.55</td><td>48.06</td><td>74.24</td><td>59.35</td><td>68.28</td><td>55.09</td></tr><tr><td>MAAL</td><td>X</td><td>60.80</td><td>47.73</td><td>61.80</td><td>45.72</td><td>68.88</td><td>52.60</td><td>77.56</td><td>63.73</td><td>70.95</td><td>58.97</td></tr><tr><td>SFODA</td><td>√</td><td>64.69</td><td>48.79</td><td>63.88</td><td>46.87</td><td>66.25</td><td>51.03</td><td>76.83</td><td>62.69</td><td>74.69</td><td>61.65</td></tr><tr><td>UPL-SFDA</td><td>√</td><td>61.31</td><td>43.57</td><td>62.88</td><td>45.81</td><td>66.64</td><td>51.49</td><td>76.59</td><td>62.45</td><td>75.33</td><td>62.20</td></tr><tr><td>UBNA</td><td>√</td><td>61.56</td><td>43.83</td><td>63.35</td><td>46.28</td><td>66.37</td><td>51.17</td><td>76.86</td><td>62.71</td><td>74.61</td><td>61.54</td></tr><tr><td>TSFCT</td><td>√</td><td>47.36</td><td>31.72</td><td>53.28</td><td>36.45</td><td>68.44</td><td>53.52</td><td>75.67</td><td>61.01</td><td>70.19</td><td>56.00</td></tr><tr><td>AIF-SFDA</td><td>√</td><td>66.22</td><td>49.85</td><td>64.44</td><td>47.49</td><td>69.99</td><td>55.16</td><td>78.08</td><td>63.15</td><td>76.68</td><td>63.01</td></tr></table>

\* Here we denote vanilla, UDA, and SFDA segmentation algorithms by /, X, and √, respectively.

Table 2: Comparison results on retinal vessel segmentation datasets, DSC (%) and IoU (%). 

<table><tr><td rowspan="2">Algorithm</td><td colspan="2">B</td><td colspan="2">C</td></tr><tr><td>DSC↑</td><td>IoU↑</td><td>DSC↑</td><td>IoU↑</td></tr><tr><td>Source</td><td>63.43</td><td>57.98</td><td>51.57</td><td>52.20</td></tr><tr><td>SFODA</td><td>66.69</td><td>60.11</td><td>55.14</td><td>55.21</td></tr><tr><td>UPL-SFDA</td><td>64.08</td><td>60.44</td><td>49.75</td><td>44.86</td></tr><tr><td>UBNA</td><td>65.39</td><td>60.57</td><td>54.48</td><td>53.36</td></tr><tr><td>TSFCT</td><td>67.03</td><td>57.13</td><td>51.02</td><td>47.27</td></tr><tr><td>AIF-SFDA</td><td>69.14</td><td>62.07</td><td>55.16</td><td>55.70</td></tr></table>

Table 3: Comparison results on ultrasound cartilage datasets, DSC (%) and IoU (%)

<table><tr><td>MI Min.</td><td>PL Sel.</td><td>Cons.</td><td>DSC↑</td><td>IoU↑</td></tr><tr><td></td><td></td><td></td><td>58.62</td><td>42.79</td></tr><tr><td>√</td><td></td><td></td><td>62.34</td><td>46.22</td></tr><tr><td>√</td><td></td><td>√</td><td>65.25</td><td>49.08</td></tr><tr><td>√</td><td>√</td><td></td><td>63.22</td><td>46.88</td></tr><tr><td>√</td><td>√</td><td>√</td><td>66.22</td><td>49.85</td></tr></table>

Table 4: Ablation Study on AVRDB.

sequentially to the model to validate the contribution of each module in AIF-SFDA to the target domain adaptation. The results indicate that the MI minimization constraint significantly contributes to enhancing the generalization of AIF-SFDA, confirming the effectiveness of selectively reducing DVI for information decoupling. Additionally, the incorporation of a confidence-aware pseudo-label selection mechanism markedly improves the stability of algorithm adaptation, while the feature consistency constraint enhances the model's feature extraction capability. The results of the ablation experiments prove the rationality of each module's settings.

Information Filter in Various Tasks When faced with different segmentation tasks, we expect the information filter to extract the frequency domain components that best fit the DII for different segmentation objectives. To verify this, we included an extra fundus optic disc (OD) segmentation experiment (see Technical Appendix for experimental details and quantitative analysis). Figure 5 shows the performance differences of the information filter guided by different segmentation tasks, all within the modality of fundus photography. It is evident that the filter map focuses on the middle and high-frequency regions for vessels with more pronounced high-frequency characteristics, while for the OD segmentation task, the filter map focuses on the middle and low frequencies. This demonstrates the flexibility of the information filter in AIF-SFDA, which can self-adjust according to the task type and instance characteristics.

Comparison with Fixed-setting Filters To demonstrate the importance of adaptive filtering for domain information decoupling, we replaced the information filter in AIF-SFDA with fixed frequency domain filters, removing the MI minimization loss for $\theta_{f}$ optimization. As shown in Figure 6, following (Li et al. 2023a), we used high-pass frequency filters with thresholds ranging from 0.01 to 0.1 for vessel segmentation on AVRDB. Fixed filter decoupling usually classifies components within the same frequency band as the same type of domain information and lacks the ability to autonomously adjust the filtering process based on the image, which prevents achieving optimal configuration. The learnable autonomous information filter in AIF-SFDA, compared to the fixed filter, increases vessel pixel edge gradients and effectively prevents artifact generation, enhancing cross-domain segmentation performance.

# Conclusion

In this paper, we present an Autonomous Information Filter driven Source-free Domain Adaptation (AIF-SFDA) algorithm for medical image segmentation tasks. The method employs a frequency-based information filter to autonomously eliminate DVI from images through mutual information minimization based on information bottleneck theory and guides DII extraction through unsupervised task-relevant loss, thereby facilitating target domain adaptation. The results of cross-domain experiments on various medical image segmentation tasks demonstrate that AIF-SFDA outperforms existing SFDA methods.

![](images/c9ee9fe62647a0187f980b8fbca6485663a0a9313e3b90287025a49b9e143412.jpg)

<details>
<summary>treemap</summary>

| Sample     | Source | Rolling-Unet | DTMFormer | CSCADA | DAMAN | MAAL | SFODA | UPL-SFDA | UBNA | TSFCT | AIF-SFDA (ours) |
|------------|--------|--------------|-----------|--------|-------|------|-------|----------|------|-------|-----------------|
| AVRDB      |        |              |           |        |       |      |       |          |      |       |                 |
| CHASEDB1   |        |              |           |        |       |      |       |          |      |       |                 |
| DRHAGIS    |        |              |           |        |       |      |       |          |      |       |                 |
| LES-AV     |        |              |           |        |       |      |       |          |      |       |                 |
| STARE      |        |              |           |        |       |      |       |          |      |       |                 |
</details>

Figure 3: Qualitative results for retinal vessel segmentation, where true positive pixels are colored in magenta, false positive pixels in red, and false negative pixels in blue.

![](images/a50533fcb8fee426b835f83839d33223be3ab8d94bdff1a66e02cd9f8bb386f0.jpg)

<details>
<summary>text_image</summary>

Ground Truth SFODA UPL-SFDA UBNA TSFCT AIF-SFDA
(ours)
B
C
</details>

Figure 4: Qualitative results for cartilage segmentation.

# Acknowledgments

This work was supported in part by the Basic Research Fund in Shenzhen Natural Science Foundation (Grant No. JCYJ20240813095112017), National Natural Science Foundation of China (Grant No. 62401246, 82272086), National Key R&D Program of China (Grant No. 2024YFE0198100), and Shenzhen Medical Research Fund (Grant No.D2402014).

# References

Alemi, A. A.; Fischer, I.; Dillon, J. V.; and Murphy, K. 2016. Deep variational information bottleneck. arXiv preprint arXiv:1612.00410.   
Cheng, P.; Hao, W.; Dai, S.; Liu, J.; Gan, Z.; and Carin, L. 2020. Club: A contrastive log-ratio upper bound of mutual information. In International conference on machine learning, 1779–1788. PMLR.   
Fleuret, F.; et al. 2021. Uncertainty reduction for model adaptation in semantic segmentation. In Proceedings of

![](images/fe1d1a63fb863a861f285e267a0423796e826f328ca32ecb469a315f83ae2ebc.jpg)  
Figure 5: Information filter processing for different tasks. (a) and (b) show the filtering process for vessel and optic disc (OD) segmentation, respectively. (I. original image, II. filtered image, III. filter map, IV. ground truth)

the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 9613–9623.

Fraz, M. M.; Barman, S. A.; Remagnino, P.; Hoppe, A.; Rudnicka, A. R.; and Owen, C. G. 2014. Ensemble classification-based vessel segmentation in retinal images using a novel vesselness enhancement method. Annals of the British Machine Vision Association, 2014: 32–32.

Gu, R.; Zhang, J.; Wang, G.; Lei, W.; Song, T.; Zhang, X.;

![](images/64c2cdc0a128dd1eea9ab3388adad615c0d655d6c71ccc88b554fafe36dcea1d.jpg)

<details>
<summary>bar</summary>

| Input Size | DSC (%) | Edge Gradient |
| ---------- | ------- | ------------- |
| Raw        | 54      | 90            |
| Fixed threshold increases (DSC) | 59      | 170           |
| AIF-SFDA (DSC) | 67      | 230           |
</details>

Figure 6: Comparison of fixed and adaptive filters on AVRDB. Left y-axis: DSC for vessel segmentation, indicating cross-domain performance. Right y-axis: Edge gradient magnitude around vessel pixels, indicating boundary distinctness. The filtered images outputted by fixed and adaptive filters are also visualized.

Li, K.; and Zhang, S. 2022. Contrastive semi-supervised learning for domain adaptive segmentation across similar anatomical structures. IEEE Transactions on Medical Imaging, 42(1): 245–256.   
Guan, H.; and Liu, M. 2021. Domain adaptation for medical image analysis: a survey. IEEE Transactions on Biomedical Engineering, 69(3): 1173–1185.   
Holm, S.; Russell, G.; Nourrit, V.; and McLoughlin, N. 2017. DR HAGIS—a fundus image database for the automatic extraction of retinal surface vessels from diabetic patients. Journal of Medical Imaging, 4(1): 014503–014503.   
Hoover, A.; Kouznetsova, V.; and Goldbaum, M. 2000. Locating blood vessels in retinal images by piecewise threshold probing of a matched filter response. IEEE Transactions on Medical Imaging, 19(3): 203–210.   
Huang, J.; Guan, D.; Xiao, A.; and Lu, S. 2021. Fsdr: Frequency space domain randomization for domain generalization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 6891–6902.   
Kawaguchi, K.; Deng, Z.; Ji, X.; and Huang, J. 2023. How does information bottleneck help deep learning? In International Conference on Machine Learning, 16049–16096. PMLR.   
Klingner, M.; Termöhlen, J.-A.; Ritterbach, J.; and Fingscheidt, T. 2022. Unsupervised batchnorm adaptation (ubna): A domain adaptation method for semantic segmentation without using source domain representations. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 210–220.   
Li, B.; Li, H.; Zhang, Y.; Li, H.; Chen, J.; Pan, F.; Chen, J.; and Liu, J. 2024a. FD-SDG: Frequency Dropout Based Single Source Domain Generalization Framework for Retinal Vessel Segmentation. In International Conference on Intelligent Computing, 393–404. Springer.   
Li, H.; Li, H.; Qiu, Z.; Hu, Y.; and Liu, J. 2022. Domain Adaptive Retinal Vessel Segmentation Guided by High-frequency Component. In International Workshop on Ophthalmic Medical Image Analysis, 115–124. Springer.

Li, H.; Li, H.; Zhao, W.; Fu, H.; Su, X.; Hu, Y.; and Liu, J. 2023a. Frequency-mixed single-source domain generalization for medical image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, 127–136. Springer.   
Li, J.; Yu, Z.; Du, Z.; Zhu, L.; and Shen, H. T. 2024b. A comprehensive survey on source-free domain adaptation. IEEE Transactions on Pattern Analysis and Machine Intelligence.   
Li, Z.; Li, C.; Luo, X.; Zhou, Y.; Zhu, J.; Xu, C.; Yang, M.; Wu, Y.; and Chen, Y. 2023b. Toward source-free cross tissues histopathological cell segmentation via target-specific finetuning. IEEE Transactions on Medical Imaging, 42(9):2666–2677.   
Lin, S.; Zhang, Z.; Huang, Z.; Lu, Y.; Lan, C.; Chu, P.; You, Q.; Wang, J.; Liu, Z.; Parulkar, A.; et al. 2023. Deep frequency filtering for domain generalization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 11797–11807.   
Liu, Q.; Chen, C.; Qin, J.; Dou, Q.; and Heng, P.-A. 2021. Feddg: Federated domain generalization on medical image segmentation via episodic learning in continuous frequency space. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 1013–1023.   
Liu, S.; Yin, S.; Qu, L.; and Wang, M. 2023. Reducing domain gap in frequency and spatial domain for cross-modality domain adaptation on medical image segmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, 1719–1727.   
Liu, Y.; Zhu, H.; Liu, M.; Yu, H.; Chen, Z.; and Gao, J. 2024. Rolling-Unet: Revitalizing MLP's Ability to Efficiently Extract Long-Distance Dependencies for Medical Image Segmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 3819–3827.   
Mukherjee, S.; Sarkar, R.; Manich, M.; Labruyere, E.; and Olivo-Marin, J.-C. 2022. Domain adapted multitask learning for segmenting amoeboid cells in microscopy. IEEE Transactions on Medical Imaging, 42(1): 42–54.   
Niemeijer, M.; Xu, X.; Dumitrescu, A.; Gupta, P.; van Ginneken, B.; Folk, J.; and Abràmoff, M. D. 2011. Automatic measurement of the arteriolar-to-venular width ratio in digital color fundus photographs. IEEE Transactions on Medical Imaging, 30(11): 1941–1950.   
Niloy, F. F.; Bhaumik, K. K.; and Woo, S. S. 2024. Source-Free Online Domain Adaptive Semantic Segmentation of Satellite Images Under Image Degradation. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 7275–7279. IEEE.   
Owen, C.; Rudnicka, A.; Mullen, R.; Barman, S.; Monekosso, D.; Whincup, P.; Ng, J.; and Paterson, C. 2009. Measuring retinal vessel tortuosity in 10-year-old children: validation of the computer-assisted image analysis of the retina (CAIAR) program. Investigative ophthalmology & visual science, 50(5): 2004–2010.   
Ronneberger, O.; Fischer, P.; and Brox, T. 2015. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference,

Munich, Germany, October 5-9, 2015, proceedings, part III 18, 234–241. Springer.   
Staal, J.; Abramoff, M.; Niemeijer, M.; Viergever, M.; and van Ginneken, B. 2004. Ridge-based vessel segmentation in color images of the retina. IEEE transactions on medical imaging, 23(4): 501–509.   
Tishby, N.; Pereira, F. C.; and Bialek, W. 2000. The information bottleneck method. arXiv preprint physics/0004057.   
Tishby, N.; and Zaslavsky, N. 2015. Deep learning and the information bottleneck principle. In 2015 ieee information theory workshop (itw), 1–5. IEEE.   
Wang, Z.; Lin, X.; Wu, N.; Yu, L.; Cheng, K.-T.; and Yan, Z. 2024. DTMFormer: Dynamic Token Merging for Boosting Transformer-Based Medical Image Segmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 5814–5822.   
Wu, J.; Wang, G.; Gu, R.; Lu, T.; Chen, Y.; Zhu, W.; Vercauteren, T.; Ourselin, S.; and Zhang, S. 2023. Upl-sfda: Uncertainty-aware pseudo label guided source-free domain adaptation for medical image segmentation. IEEE transactions on medical imaging.   
Xu, Q.; Zhang, R.; Zhang, Y.; Wang, Y.; and Tian, Q. 2021. A fourier-based framework for domain generalization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 14383–14392.   
Yang, C.; Guo, X.; Chen, Z.; and Yuan, Y. 2022. Source free domain adaptation for medical image segmentation with fourier style mining. Medical Image Analysis, 79: 102457.   
Yang, Y.; and Soatto, S. 2020. Fda: Fourier domain adaptation for semantic segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 4085–4095.   
Ye, M.; Zhang, J.; Ouyang, J.; and Yuan, D. 2021. Source data-free unsupervised domain adaptation for semantic segmentation. In Proceedings of the 29th ACM international conference on multimedia, 2233–2242.   
Zhang, H.; Su, Y.; Xu, X.; and Jia, K. 2024. Improving the generalization of segmentation foundation model under distribution shift via weakly supervised adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 23385–23395.   
Zhou, W.; Ji, J.; Cui, W.; Wang, Y.; and Yi, Y. 2023. Unsupervised Domain Adaptation Fundus Image Segmentation via Multi-scale Adaptive Adversarial Learning. IEEE Journal of Biomedical and Health Informatics.