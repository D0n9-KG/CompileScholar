# Style Adaptation and Uncertainty Estimation for Multi-Source Blended-Target Domain Adaptation

Yuwu Lu, $^{*}$ Haoyu Huang, and Xue Hu

School of Artificial Intelligence, South China Normal University

{luyuwu2008, hyhuang99, hx1430940232}@163.com

# Abstract

Blended-target domain adaptation (BTDA), which implicitly mixes multiple sub-target domains into a fine domain, has attracted more attention in recent years. Most previously developed BTDA approaches focus on utilizing a single source domain, which makes it difficult to obtain sufficient feature information for learning domain-invariant representations. Furthermore, different feature distributions derived from different domains may increase the uncertainty of models. To overcome these issues, we propose a style adaptation and uncertainty estimation (SAUE) approach for multi-source blended-target domain adaptation (MBDA). Specifically, we exploit the extra knowledge acquired from the blended-target domain, where a similarity factor is adopted to select more useful target style information for augmenting the source features. Then, to mitigate the negative impact of the domain-specific attributes, we devise a function to estimate and mitigate uncertainty in category prediction. Finally, we construct a simple and lightweight adversarial learning strategy for MBDA, effectively aligning multi-source and blended-target domains without the requirements of domain labels of the target domains. Extensive experiments conducted on several challenging DA benchmarks, including the ImageCLEF-DA, Office-Home, VisDA 2017, and DomainNet datasets, demonstrate the superiority of our method over the state-of-the-art (SOTA) approaches.

# 1 Introduction

Domain adaptation (DA), whose objective is to transfer knowledge from one or more well-labeled source domains to a non-labeled target domain, has been studied in recent years $[1, 2, 3, 4, 5, 6]$ , including object classification $[1, 2]$ , semantic segmentation $[3, 4]$ , and object detection $[5]$ . However, most DA approaches are based on a setting that has single source domain and single target domain $[1, 2]$ . In the real world, unlabeled target domains are usually drawn from different distributions. Therefore, most of the existing single target-based DA approaches may not be the best answer to address domain shifts in reality.

Fortunately, an increasing number of researchers have focused on the above-mentioned issues, and multi-target domain adaptation (MTDA) $[7, 8, 9]$ has been studied. MTDA aims to learn a model that can simultaneously utilize information from single source domain and multiple target domains and then perform well in multiple target domains. For instance, HGAN $[10]$ adopts a heterogeneous graph attention network to explore the relations among multiple target domain features. In $[11]$ , multiple expert models employed the corresponding source-target domain pair-groups and were then aligned by a student model. Although the existing MTDA approaches have made certain progress, the standard MTDA is still facing challenges because massive amounts of unlabeled data drawn from different distributions are commonly used in real-world settings. It is time-consuming and expensive to divide massive data into multiple corresponding distributions (target domains). For example, for

Table 1: Comparison about different settings of DA. 

<table><tr><td>Settings</td><td>Source domain number</td><td>Target domain number</td><td>Domain labels</td></tr><tr><td>UDA/SSDA</td><td>single</td><td>single</td><td>√</td></tr><tr><td>MSDA</td><td>multiple</td><td>single</td><td>√</td></tr><tr><td>MTDA</td><td>single</td><td>multiple</td><td>√</td></tr><tr><td>BTDA</td><td>single</td><td>multiple</td><td>×</td></tr><tr><td>MMDA</td><td>multiple</td><td>multiple</td><td>√</td></tr><tr><td>MBDA</td><td>multiple</td><td>multiple</td><td>×</td></tr></table>

encrypted data stored in a cloud server, due to privacy protection, models cannot directly know the origins of these data (domain labels), which are drawn from different distributions and are blended into a large target domain. Based on the above-mentioned case, blended-target DA (BTDA), which is a more beneficial scenario in real-world settings, has been proposed $[12]$ .

Current BTDA approaches $[12, 13]$ are mainly based on three points: 1) the adaptation process contains single source domain and multiple target domains. 2) During the adaptation process, the model cannot access both the domain labels and category labels of the target domains. 3) In blended-target domain, the category labels of each sub-target domain may not follow the same distribution. Therefore, simply utilizing MTDA or SSDA (single source domain adaptation) methods to handle the BTDA task may lead to negative transfer because the domain labels of target domains are unseen, and the category feature space is a hybrid space. As the first deep learning work focused on the BTDA scenario, AMEAN $[12]$ employs two adversarial learning steps and utilizes meta-learning to minimize the domain gap between the source domain and the blended-target domain. However, insufficient information obtained from single source domain makes models difficult to align the distributions of multiple target domains. Moreover, the presence of different distributions in the blended-target domain may aggravate negative transfer. Recently, multi-source domain adaptation (MSDA) $[14, 15, 16]$ has produced impressive results. MSDA approaches can utilize more feature information from extra source domains to learn domain-invariant representations, effectively solving negative transfer. However, as far as we know, no related works have been proposed to utilize more feature information from multiple source domains for BTDA.

In this paper, to further exploit feature information from multiple domains, we pay attention to the BTDA in the case of multiple source domains, i.e., Multi-Source Blended-Target Domain Adaptation (MBDA). The comparisons of different DA settings are illustrated in Table 1. At the same time, a style adaptation and uncertainty estimation (SAUE) method is proposed for MBDA. Different from previous works, we utilize the style information of the blended-target domain to enhance source domain features, thus building a better representation space. Specifically, we first simultaneously extract the source and target style information and then calculate the similarity factors between the source and target style information. The similarity factors are served as the weighted matrix during the feature augmentation process. Based on style adaptation, we further analyze the uncertainty in the classification model and adopt a loss function to eliminate the uncertainty introduced by the multi-source domains. In addition, as the domain labels of the blended-target domain are unavailable in MBDA, we construct an adversarial learning scheme for MBDA without the requirement of domain labels of the target domains.

The main contributions of this work are summarized as follows:

\- An approach SAUE is proposed to explore information from multiple source domains for BTDA. As far as we know, SAUE is the first work that was proposed for MBDA, which can utilize more feature information from extra source domains to learn domain-invariant representations.

\- To further exploit the style information in the blended-target domain, we propose a similarity-based style adaptation strategy for MBDA, which selects target styles to enhance the source representation space.

\- We propose an uncertainty estimation technique to enhance the robustness of our method, which exploits valuable knowledge from multiple source domains. In addition, we construct a specific adversarial learning strategy for MBDA, which aligns domains without the requirement of domain labels.

# 2 Related Works

Single-source and Single-target DA (SSDA). The objective of SSDA is to learn domain-invariant representations through the relations between the source and target domains. Based on this objective, researchers have carried out widespread researches and achieved certain progress $[17, 18, 19, 20, 21, 22, 23]$ . The current SSDA methods are mainly divided into two types: distance metric-based approaches $[17, 18, 19]$ and adversarial learning-based approaches $[20, 21, 22, 23]$ . Distance metric-based methods learn domain-invariant representations through feature discrepancy matching by using a distance metric function. DAN $[18]$ utilizes multi-kernel maximum mean discrepancy (MMD) to measure the discrepancy between the source and target domains and then minimizes the discrepancy to learn domain-invariant representations. CAF $[19]$ utilizes sliced Wasserstein distance (SWD) $[24]$ to measure domain discrepancy. Motivated by the GANs $[25, 26]$ , adversarial learning-based SSDA methods have also been widely researched $[20, 21, 22, 23]$ . Adversarial learning methods perform min-max two-player games between the category classifier and domain discriminator to learn domain-invariant representations. DANN $[20]$ , which was the earliest work in adversarial learning-based SSDA, successfully achieves domain-level adaptation via a gradient reverse layer. Different from DANN, SCDA $[22]$ and DALN $[23]$ remove the discriminator from their networks and model the adversarial relation between the feature extractor and category classifier. Although the above-mentioned approaches have achieved great success, due to the limitations of single source domain features and single target domain features, current SSDA methods still face some challenges in real applications.

Multiple Domains DA. The motivation of multiple domains DA is to explore more useful knowledge from multiple domains for the tasks. Many researchers have focused on multiple-domain DA and proposed many excellent methods $[13, 14, 15, 8]$ , including MSDA $[14, 15, 27, 28]$ , MTDA $[8]$ , BTDA $[13, 12]$ , and MMDA (multi-source and multi-target DA) $[29]$ . M3SDA $[14]$ utilizes moment matching to align domain distribution. DCA $[15]$ further extracts the multiview features from multiple source domains and then utilizes multiple classifiers and pseudo-label learning strategy to align distributions. Meanwhile, in MTDA, CGCT $[8]$ utilizes feature aggregation and curriculum learning to learn the pseudo-labels of multiple target domains. AMDA $[29]$ constructs multiple domain discriminators and utilizes attention mechanism to address the MMDA issue.

Recently, a more realistic DA scenario, BTDA, has been studied $[12, 13]$ . For example, MCDA $[13]$ utilizes the mutual condition to learn domain-invariant representations, which has achieved great progress in BTDA. However, single source domain in BTDA cannot provide sufficient feature information for aligning the source and blended-target domains. Furthermore, the unseen domain labels of the target domains also aggravate the challenges. Therefore, we consider multiple source domains in BTDA and utilize the style information of the target domains to optimize the representations of source features and minimize the model uncertainty, thereby obtaining a better transfer.

# 3 Method

# 3.1 Preliminary

In MBDA, we have $M$ labeled source domains $\mathcal{S} = \{\mathcal{S}_m\}_{m=1}^M$ that are drawn from distributions $\{P_{\mathcal{S}_m}\}_{m=1}^M$ . $\mathcal{S}_m = \{x_i^{\mathcal{S}_m}, y_i^{\mathcal{S}_m}\}_{i=1}^{|\mathcal{S}_m|}$ , where $x_i^{\mathcal{S}_m} \in \mathbb{R}^d$ denotes the $i$ -th source sample from the $m$ -th source domain and $y_i^{\mathcal{S}_m}$ is the corresponding category label, and $d$ denotes the number of dimensions. Meanwhile, we have an unlabeled blended-target domain $\mathcal{T}$ that consists of $N$ sub-target domains $\{\mathcal{T}_n\}_{n=1}^N$ , and $\mathcal{T} = \{x_j^{\mathcal{T}}\}_{j=1}^{|\mathcal{T}|}$ . The distributions of sub-target domains are $\{P_{\mathcal{T}_n}\}_{n=1}^N$ . Therefore, the distribution of blended-target domain $P_{\mathcal{T}}$ is the mixture of sub-target domains, i.e., $P_{\mathcal{T}} = \sum_{n=1}^N \pi_n P_{\mathcal{T}_n}$ , where $\pi \in [0,1]$ and $\sum_{n=1}^N \pi_n = 1$ . Each of the source and target domains shares the same category space. The objective of MBDA is to train a model that utilizes multiple source domain features and performs well on the blended-target domain. Different from MTDA, the target domain labels are unseen in MBDA. In addition, the analysis in [12] demonstrated that directly utilizing DA methods to address BTDA transfer tasks may cause increased uncertainty and negative transfer. Therefore, we utilize the style information of the blended-target domain to augment source features and minimize the uncertainty of the model. Furthermore, the adversarial learning strategy in our method without the requirement of domain labels of the sub-target domains is suitable for MBDA setting. Figure 1 illustrates the overall architecture of SAUE.

![](images/0799669a13bb40d045e9bf781554699d48348771f301d9e66dfb6276dd915ad8.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph_Feature_Extractor["Feature Extractor with Style Adaptation"]
        A["Source Domains S1 ... SM"] --> B["Convolution Layer ZiSm"]
        C["Blended Target Domain T1 ... TN"] --> D["Convolution Layer zjT"]
        B --> E["Normalization (μiSm, σimSm)"]
        D --> F["Adapt (μi', σjT)"]
        E --> G["Similarity Factors λiSm"]
        F --> H["Confuse (μi, σi)"]
        G --> I["ZiSmT"]
        H --> J["ZiSmT"]
        I --> K["Style Adaptation"]
        J --> L["Style Adaptation"]
        K --> M["Convolution Layer"]
        L --> M
    end

    subgraph_Classifier["Classifier and Uncertainty Estimation"]
        N["Lcls"] --> O["D(pi^Sm | id_i^Sm) KL divergence"]
        O --> P["L_unc"]
        P --> Q["||·||_N ||·||_N"]
        Q --> R["L_nwd"]
        R --> S["+"]
        S --> T["-"]
        T --> U["GRL"]
    end

    B --> V["Wasserstein Distance (μi', σj)"]
    V --> W["Adapt"]
    W --> X["Similarity Factors λiSm"]
    X --> Y["Confuse (μi, σi)"]
    Y --> Z["ZiSmT"]
    Z --> AA["ZiSmT"]
    AA --> AB["Style Adaptation"]
    AB --> AC["Feature Extractor with Style Adaptation"]
    AC --> AD["Source Dataflow"]
    AD --> AE["Target Dataflow"]
    AE --> AF["Backpropagation"]
    AF --> AG["GRL Gradient Reverse Layer"]
    AG --> AH["Gradient Reverse Layer"]
```
</details>

Figure 1: Overview of the proposed SAUE approach. First, the style information of the blended-target domain is utilized to augment the source features through the similarity factors. Second, we calculate the uncertainty of the model and optimize prediction uncertainty via the Dirichlet distribution. Finally, the adversarial learning strategy without discriminator effectively guides the SAUE process to adapt the blended-target domain without the requirement of domain labels of sub-target domains.

# 3.2 Style Adaptation from Blended-Target Domain

Since the principal parts of features from different domains remain domain-invariant, the domain-specific parts, which mainly contain style information, are the main discrepancies between different domains. In addition, the target feature distributions in MBDA are confused, which may cause model degradation. Therefore, we try to utilize the style information of blended-target domain to augment source features, which brings source features closer to target features. Previous work [13] has demonstrated that low-level features of deep neural networks (DNNs) mainly represent style information. Some works [13, 30] have utilized the channel-wise mean and variance of the low-level features to represent the style information of input samples. Thus, for sample $x_{i}^{\mathcal{S}_{m}}$ which from the $m$ -th source domain, let its low-level feature be $z_{i}^{\mathcal{S}_{m}} \in \mathbb{R}^{d \times H_{\mathcal{S}_{m}} \times W_{\mathcal{S}_{m}}}$ , where $d$ denotes the channel and $H_{\mathcal{S}_{m}}$ , $W_{\mathcal{S}_{m}}$ denote the height and width of sample $x_{i}^{\mathcal{S}_{m}}$ . The channel-wise mean and variance of the low-level feature $z_{i}^{\mathcal{S}_{m}}$ can be defined as follows:

$$
\mu_ {i} ^ {\mathcal {S} _ {m}} = \frac {1}{H _ {\mathcal {S} _ {m}} W _ {\mathcal {S} _ {m}}} \sum_ {h = 1} ^ {H _ {\mathcal {S} _ {m}}} \sum_ {w = 1} ^ {W _ {\mathcal {S} _ {m}}} z _ {i} ^ {\mathcal {S} _ {m}}, \sigma_ {i} ^ {\mathcal {S} _ {m}} = \sqrt {\frac {1}{H _ {\mathcal {S} _ {m}} W _ {\mathcal {S} _ {m}}} \sum_ {h = 1} ^ {H _ {\mathcal {S} _ {m}}} \sum_ {w = 1} ^ {W _ {\mathcal {S} _ {m}}} (z _ {i} ^ {\mathcal {S} _ {m}} - \mu_ {i} ^ {\mathcal {S} _ {m}}) ^ {2}}. \tag {1}
$$

Low-level features mainly represent style information, but different samples contain specific pieces of style information. Therefore, we adopt feature normalization technique to standardize the feature $z_{i}^{S_{m}}$ , and the normalized feature $\tilde{z}_{i}^{S_{m}}$ is defined as:

$$
\tilde {z} _ {i} ^ {\mathcal {S} _ {m}} = \frac {z _ {i} ^ {\mathcal {S} _ {m}} - \mu_ {i} ^ {\mathcal {S} _ {m}}}{\sigma_ {i} ^ {\mathcal {S} _ {m}} + \epsilon}, \tag {2}
$$

where $\epsilon$ is a small value used to avoid division by zero.

Then, the target features will be utilized to augment the normalized source features. Previous work [13] randomly augmented source features through target style information, which yielded limited performance. In our work, we select the target style information according to a weight factor. Specifically, we leverage the Wasserstein Distance [31] to measure the style distribution discrepancy $w_{i}^{S_{m}}$ between the source low-level feature $z_{i}^{S_{m}}$ and the target low-level feature $z_{j}^{\mathcal{T}}$ as follows:

$$
w _ {i} ^ {\mathcal {S} _ {m}} = \| \mu_ {i} ^ {\mathcal {S} _ {m}} - \mu_ {j} ^ {\mathcal {T}} \| + (\sigma_ {i} ^ {\mathcal {S} _ {m}}) ^ {2} + (\sigma_ {j} ^ {\mathcal {T}}) ^ {2} - 2 \sigma_ {i} ^ {\mathcal {S} _ {m}} \sigma_ {j} ^ {\mathcal {T}}. \tag {3}
$$

Then, we utilize $w_{i}^{S_{m}}$ to calculate the weight factor as follows:

$$
\lambda_ {i} ^ {\mathcal {S} _ {m}} = \frac {\exp (1 / (1 + w _ {i} ^ {\mathcal {S} _ {m}}))}{\sum_ {m = 1} ^ {M} \sum_ {i = 1} ^ {| \mathcal {S} _ {m} |} \exp (1 / (1 + w _ {i} ^ {\mathcal {S} _ {m}}))}. \tag {4}
$$

To ensure that the sum in Eq. (4) equals to 1, we utilize softmax operation to normalize each $\lambda_i^{S_m}$ . Then, we can obtain the weighted target style as follows:

$$
\mu_ {i} = \sum_ {m = 1} ^ {M} \sum_ {i = 1} ^ {| \mathcal {S} _ {m} |} \lambda_ {i} ^ {\mathcal {S} _ {m}} \mu_ {j} ^ {\mathcal {T}}, \sigma_ {i} = \sum_ {m = 1} ^ {M} \sum_ {i = 1} ^ {| \mathcal {S} _ {m} |} \lambda_ {i} ^ {\mathcal {S} _ {m}} \sigma_ {j} ^ {\mathcal {T}}. \tag {5}
$$

Finally, the low-level source feature augmented by the blended-target style information can be calculated as follows:

$$
z _ {i} ^ {\mathcal {S} _ {m} \mathcal {T}} = \sigma_ {i} \tilde {z} _ {i} ^ {\mathcal {S} _ {m}} + \mu_ {i}. \tag {6}
$$

Different from previously developed image augmentation method [13], our method directly utilizes target information with weight factor to augment source features instead of generating specific images. The low-level feature $z_{i}^{S_{m}\mathcal{T}}$ augmented by diverse target styles further mitigates the impact of the domain-specific attributes.

# 3.3 Uncertainty Estimation and Elimination

Although multiple source domains provide additional supervised information for adaptation compared to a single source domain, they also introduce more domain-specific attributes. This can cause model degradation, especially when some source domains are significantly dissimilar to the blended-target domain due to the abundance of domain-specific attributes. Evidential model learning (EDL) [32] is an interpretable approach that fuses knowledge from multiple domains using the Dempster-Shafer Rule [33, 34], which is more beneficial to MBDA scenario. Thus, we utilize the Dirichlet-based evidential model [32] to estimate the uncertainty of our model during the training process. Specifically, for a sample $x_{i}$ , we have the predictions $p_i = C(G(x_i)) = [p_{i1}, p_{i2}, \ldots, p_{iK}]$ , where $C$ and $G$ denote the classifier and feature generator, respectively. The probability density function of $p_i$ is defined as follows:

$$
D \left(p _ {i} \mid \alpha_ {i}\right) = \left\{ \begin{array}{l l} \frac {1}{B \left(\alpha_ {i}\right)} \prod_ {k = 1} ^ {K} p _ {k} ^ {\alpha_ {i k} - 1} & \text { for } p \in U _ {K} \\ 0 & \text { otherwise } \end{array} , \right. \tag {7}
$$

where $U_{k} = \{p_{i} | \sum_{k=1}^{K} p_{ik} = 1  and  0 \leq p_{i1}, ..., p_{iK} \leq 1\}$ is the K-dimensional unit simplex and $\alpha_{i}$ is the parameter of the Dirichlet distribution. $B(\alpha_{i}) = \frac{\Gamma(\sum_{k=1}^{K} \alpha_{ik})}{\prod_{k=1}^{K} \Gamma(\alpha_{ik})}$ is the K-dimensional multinomial beta function, and $\Gamma(\cdot)$ denotes the gamma function.

Previous work [32] has demonstrated that DNNs can generate opinions for classification tasks as Dirichlet distributions. Therefore, given sample $x_{i}$ , for prediction of class c that generated by DNNs, the Dirichlet distributions connected with DNNs can be defined as follows:

$$
P (y = c \mid x _ {i}) = \frac {\alpha_ {i c}}{\sum_ {k = 1} ^ {K} \alpha_ {i k}} = \frac {C _ {c} (G \left(x _ {i}\right))}{\sum_ {k = 1} ^ {K} C _ {k} \left(G \left(x _ {i}\right)\right)} = \mathbb {E} [ D \left(p _ {i c} \mid \alpha_ {i}\right) ]. \tag {8}
$$

The derivation of Eq. (8) is provided in Appendix B.

In this work, for source sample $x_{i}^{S_{m}}$ , we utilize the prediction of the category classifier as the evidence vector, and the parameters of the corresponding Dirichlet distribution can be defined as $\alpha_{i}^{S_{m}} = C(G(x_{i}^{S_{m}})) + 1$ . To eliminate the uncertainty, we force the total evidence to zero when the model generates an incorrect prediction for the source sample, and the corresponding uniform Dirichlet distribution is $D(p_{i}^{S_{m}} | \langle 1, \ldots, 1 \rangle)$ . For implementation, we utilize the Kullback-Leibler (KL) divergence to reduce the impact of incorrectly classified source samples in our loss function, which is defined as follows:

$$
\mathcal {L} _ {u n c} (x ^ {\mathcal {S} _ {m}}) = \sum_ {i = 1} ^ {| \mathcal {S} _ {m} |} K L [ D (p _ {i} ^ {\mathcal {S} _ {m}} | \tilde {\alpha} _ {i} ^ {\mathcal {S} _ {m}}) \| D (p _ {i} ^ {\mathcal {S} _ {m}} | \langle 1, \dots , 1 \rangle) ], \tag {9}
$$

where $\tilde{\alpha}_i^{S_m} = y_i^{S_m} + (1 - y_i^{S_m})\odot \alpha_i^{S_m}$ denotes the Dirichlet parameter used to remove the true evidence from prediction $\alpha_{i}^{S_{m}}$ . Specifically, the KL divergence is:

$$
\begin{array}{l} K L [ D (p _ {i} ^ {\mathcal {S} _ {m}} | \tilde {\alpha} _ {i} ^ {\mathcal {S} _ {m}}) \| D (p _ {i} ^ {\mathcal {S} _ {m}} | \langle 1, \dots , 1 \rangle) ] \\ = \log \left[ \frac {\Gamma (\sum_ {k = 1} ^ {K} \tilde {\alpha} _ {i} ^ {\mathcal {S} _ {m}})}{\Gamma (K) \prod_ {k = 1} ^ {K} \Gamma (\tilde {\alpha} _ {i k})} \right] + \sum_ {k = 1} ^ {K} (\tilde {\alpha} _ {i k} - 1) \left[ \Psi (\tilde {\alpha} _ {i k}) - \Psi (\sum_ {j = 1} ^ {K} \tilde {\alpha} _ {i j}) \right], \tag {10} \\ \end{array}
$$

where $\Gamma (\cdot)$ and $\varPsi (\cdot)$ denotes the gamma function and digamma function, respectively.

# Algorithm 1 SAUE for MBDA

Input: Source domains $\{S_{m}\}_{m=1}^{M}$ and the corresponding data $\{x_{i}^{S_{m}}, y_{i}^{S_{m}}\}_{i}^{|S_{m}|}$ , blended-target domain data $\{x_{j}^{T}\}$ , hyper-parameters $\lambda_{e}$ and $\lambda_{d}$ , maximum iteration I, and mini-batch size B.

Output: Optimal feature generator G and category classifier C.

1: for i in 1:I do   
2: Randomly sample a mini-batch of $B$ source samples and target samples.   
3: Augment the source features by utilizing style adaptation, i.e., Eq. (6).   
4: Minimize the parameters of the feature generator and category classifier by $\mathcal{L}_{cls}$ .   
5: Optimize the uncertainty of model through $L_{unc}$ .   
6: Perform the min-max game between feature generator and classifier with $L_{d}$ :

$$
\min _ {G} \max _ {C} \mathcal {L} _ {d} (x ^ {\mathcal {S} _ {m}}, x ^ {\mathcal {T}}).
$$

# 3.4 Domain Adversarial Alignment without Domain Labels

Existing works $[20, 21, 22, 8]$ have demonstrated that adversarial learning strategy is beneficial in DA. However, most of the adversarial learning strategies in DA $[20, 21]$ usually request the domain labels of the source and target domains to train their discriminators, which cannot satisfy MBDA. Inspired by the intra- and inter-class correlation $[35]$ , we design an adversarial learning strategy for MBDA without discriminator and domain label requirements. Specifically, the category classifier is reused to discriminate which domain a feature originates from, with the guidance of the Nuclear norm $\|\cdot\|_{*}$ . We first measure the distribution difference between the source and blended-target domains through the nuclear-norm 1-Wasserstein discrepancy (NWD) $[23]$ and then utilize a gradient reverse layer (GRL) $[20]$ to maximize the discriminative loss of the classifier. Simultaneously, we minimize the feature generator to play the min-max game with the classifier through the NWD. First, the NWD loss can be defined as:

$$
\mathcal {L} _ {d} (x ^ {\mathcal {S} _ {m}}, x ^ {\mathcal {T}}) = \frac {1}{| \mathcal {S} _ {m} |} \sum_ {i = 1} ^ {| \mathcal {S} _ {m} |} \left\| C (G (x _ {i} ^ {\mathcal {S} _ {m}})) \right\| _ {*} - \frac {1}{| \mathcal {T} |} \sum_ {j = 1} ^ {| \mathcal {T} |} \left\| C (G (x _ {j} ^ {\mathcal {T}})) \right\| _ {*}. \tag {11}
$$

Then, the adversarial learning strategy between feature generator and classifier is defined as follows:

$$
\min _ {G} \max _ {C} \mathcal {L} _ {d} (x ^ {\mathcal {S} _ {m}}, x ^ {\mathcal {T}}). \tag {12}
$$

# 3.5 Model Optimization and Theoretical Analysis

Overall Objective. The overall loss function that optimizes SAUE for MBDA is defined as:

$$
\min _ {G, C} \left\{\mathcal {L} _ {c l s} (x ^ {\mathcal {S} _ {m}}, x ^ {\mathcal {T}}) + \lambda_ {e} \mathcal {L} _ {u n c} (x ^ {\mathcal {S} _ {m}}) + \lambda_ {d} \max _ {C} \mathcal {L} _ {d} (x ^ {\mathcal {S} _ {m}}, x ^ {\mathcal {T}}) \right\}, \tag {13}
$$

where $\lambda_{e} = \min(1, e/\lambda_{e}^{\prime}) \in [0,1]$ is the annealing coefficient, which prevents $L_{unc}$ from overpenalizing the neural network to a uniform distribution in the early training epochs, and e is the current number of epochs and $\lambda_{e}^{\prime} = 40$ . $\lambda_{d}$ is a hyper-parameter which is initially set to $\lambda_{d} = 1$ as in [23]. $L_{cls}$ is the classification loss, which ensures that the category classifier can correctly classify samples. With the cross-entropy loss $L_{ce}$ , the classification loss $L_{cls}$ is defined as follows:

$$
\mathcal {L} _ {c l s} (x ^ {\mathcal {S} _ {m}}, y ^ {\mathcal {S} _ {m}}) = \sum_ {m = 1} ^ {M} \frac {1}{| \mathcal {S} _ {m} |} \sum_ {i = 1} ^ {| \mathcal {S} _ {m} |} \mathcal {L} _ {c e} (C (G (x _ {i} ^ {\mathcal {S} _ {m}})), y _ {i} ^ {\mathcal {S} _ {m}}). \tag {14}
$$

After adversarial training, our model can effectively adapt the blended-target domain without the requirement of domain labels of sub-target domains. The concrete steps of SAUE are shown in Algorithm 1.

Theoretical Analysis. Here, we utilize PAC-Bayesian theory $[36]$ to optimize our classification model with uncertainty estimation and elimination. The full-bound theorem motivated by previous work $[37]$ is illustrated in Theorem 1 and Lemma 1, and the proofs are provided in Appendix C.

Theorem 1 [37]. Suppose we have given the m-th source data distribution $P_{\mathcal{S}_m}$ , a hypothesis set $\mathcal{H}$ , and a prior distribution $\pi$ over the hypothesis space $\Theta$ . For any $\tau \in (0,1]$ and $\lambda > 0$ , with a probability at least $1 - \tau$ over the source samples $\mathcal{S}_m \sim P_{\mathcal{S}_m}$ , for all posteriors $\rho$ , we have:

$$
\mathbb {E} _ {\rho (\mathcal {H})} [ \mathcal {L} (\mathcal {H}) ] \leq \mathbb {E} _ {\rho (\mathcal {H})} [ \tilde {\mathcal {L}} _ {\mathcal {S} _ {m}} (\mathcal {H}) ] + \frac {1}{\lambda} \left[ K L (\rho \| \pi) + \log \frac {1}{\tau} + \Psi_ {\mathcal {S} _ {m}, \pi} (\lambda , n) \right], \tag {15}
$$

where $\Psi_{\mathcal{S}_m,\pi}(\lambda ,n) = \log \mathbb{E}_{\pi (\mathcal{H})}\mathbb{E}_{\mathcal{S}_m\sim P_{\mathcal{S}_m}}\left[e^{\lambda (\mathcal{L}(\mathcal{H}) - \mathcal{L}(\tilde{\mathcal{H}}))}\right].$

Lemma 1 [38]. The PAC-Bayes bound, involving constants $\tau$ and n, as introduced in Theorem 1, is minimized by the Bayesian posterior $p(\mathcal{H})$ , which represents the distribution over $\Theta$ .

During uncertainty estimation and elimination, just as in Eq. (9), we utilize $D(p_{i}^{\mathcal{S}_{m}}|\tilde{\alpha}_{i}^{\mathcal{S}_{m}})$ as the posterior distribution and $D(p_{i}^{\mathcal{S}_{m}}|\langle1,\ldots,1\rangle)$ as the prior distribution. Therefore, the upper bound of the classification model can be expressed as:

$$
\sum_ {m = 1} ^ {M} \frac {1}{| \mathcal {S} _ {m} |} \sum_ {i = 1} ^ {| \mathcal {S} _ {m} |} \left[ \mathcal {L} _ {c l s} + \frac {1}{\lambda} K L (D (p _ {i} ^ {\mathcal {S} _ {m}} | \tilde {\alpha} _ {i} ^ {\mathcal {S} _ {m}}) \| D (p _ {i} ^ {\mathcal {S} _ {m}} | \langle 1,..., 1 \rangle)) \right]. \tag {16}
$$

Generalization Bound. In this part, we prove why SAUE performs well on the blended-target domain via Lemma 2 and Theorem 2. The proofs and derivations are provided in Appendix D.

Lemma 2 [39]. Suppose we have given the probability measures $\nu_{\mathcal{S}_m}, \nu_\mathcal{T} \in \mathcal{P}(\mathcal{F})$ of the $m$ -th source feature $f_{\mathcal{S}_m}$ and the blended-target domain feature $f_\mathcal{T}$ , a hypothesis space $\Theta$ , and a subspace $\tilde{\mathcal{H}} \in \Theta$ . Let $\mathcal{F}$ denote a fixed representation space and $c(f_{\mathcal{S}_m}, f_\mathcal{T})$ denote the adaptation cost. For the ideal classifier $h' \in \tilde{\mathcal{H}}$ and any classifier $h \in \tilde{\mathcal{H}}$ with $f_{\mathcal{S}_m} \sim \nu_{\mathcal{S}_m}$ and $f_\mathcal{T} \sim \nu_\mathcal{T}$ , we have:

$$
\left| \epsilon_ {\mathcal {S} _ {m}} (h, h ^ {\prime}) - \epsilon_ {\mathcal {T}} (h, h ^ {\prime}) \right| \leq \frac {1}{2} d _ {\mathcal {H} \Delta \mathcal {H}} (\nu_ {\mathcal {S} _ {m}}, \nu_ {\mathcal {T}}), \tag {17}
$$

where $\epsilon_{S_{m}}$ and $\epsilon_{T}$ denote the error on the m-th source domain and the error on the blended-target domain respectively, and $\epsilon_{T} = \frac{1}{N} \sum_{j=1}^{N} \epsilon_{T_{j}}$ . $d_{H\Delta H}$ denotes the $H\Delta H$ -distance.

Theorem 2. Based on Lemma 2, with the error of the ideal joint hypothesis $\eta' = \epsilon_{S_m}(h') + \epsilon_T(h')$ which is a sufficiently small constant, for any $\delta \in (0,1)$ , with probability at least $1 - \delta$ , for every $h \in \mathcal{H}$ , $\epsilon_T(h)$ is bounded by the following terms:

$$
\epsilon_ {\mathcal {T}} (h) \leq \epsilon_ {\mathcal {S} _ {m}} (h) + \frac {1}{2} \hat {d} _ {\mathcal {H} \Delta \mathcal {H}} (\nu_ {\mathcal {S} _ {m}}, \nu_ {\mathcal {T}}) + 4 \sqrt {\frac {2 d \log (2 b ^ {\prime}) + \log (\frac {2}{\delta})}{b ^ {\prime}}} + \eta^ {\prime}, \tag {18}
$$

where $\eta' = \epsilon_{\mathcal{S}_{m}}(h') + \epsilon_{\mathcal{T}}(h')$ is the ideal error for the classifier, which is a sufficiently small constant. $b'$ is the size of unlabeled samples.

Therefore, the final objective of the MBDA classification task is to reduce the joint domain discrepancy term $\sum_{m=1}^{M} |\epsilon_{\mathcal{S}_m}(h, h') - \epsilon_{\mathcal{T}}(h, h')|$ .

# 4 Experiments

# 4.1 Datasets and Implementation Details

Datasets. Four standard benchmark datasets are used to validate the effectiveness of our proposed method. The ImageCLEF-DA $[40]$ contains 2,400 images and is divided into 4 domains: Bing (b), Caltech (c), ImageNet (i), and Pascal (p). Each domain has 12 categories, and every category has 50 images. The Office-Home $[41]$ also consists of 4 domains and 15,588 images belonging to 65 categories from four subdomains: Art (Ar), Clipart (Cl), Products (Pr), and Real world (Rw). The DomainNet $[14]$ is a large-scale dataset in DA that contains 0.6 million images of 345 categories from 6 domains: Clipart (C), Infograph (I), Painting (P), Quickdraw (Q), Real world (R), and Sketch (S). Following the protocol used in $[29]$ , we select 126 categories and 4 domains (C, P, R, and S) in our experiments. The VisDA 2017 $[42]$ dataset is a challenging dataset consists of 2 domains (Syn. and Rel.) and 7 categories.

Implementation Details. We utilize PyTorch framework $[43]$ to perform our experiments; the PyTorch version is 1.13.1 and CUDA version is 11.7. We use an ImageNet pre-trained ResNet $[44]$ , replacing the last FC layer with task-specific FC layers. All experiments are run on a single GeForce RTX-4090 GPU, and the batch size of both the source and blended-target domains are set to 32. The optimizer is Stochastic Gradient Descent (SGD) with a momentum parameter of 0.9 and a weight decay of 1e-3. The learning rate is 1e-3 and updated by the LambdaLR $[43]$ during the training process.

# 4.2 Comparisons to State-of-the-Art

To evaluate the effectiveness of our proposed method, we conduct extensive experiments and compare our approach with the state-of-the-art (SOTA) methods in terms of DA classification. The comparison methods include SSDA approaches, i.e., MCD [45], DAN [18], TSA [46], DALN [23], BIWAA [47], and SCDA [22]; MSDA methods, i.e., MDAN [48], DCTN [49], and DIDA [50]. MTDA/BTDA methods: MTDA-ITA [7] and MCDA [13]; and Multi-source Multi-target DA (MMDA), i.e., AMDA [29] and HTA [51]. The comparison results are presented in Tables 2-4, in which we select two domains as source domains and combine other two domains to form the blended-target domain. Note that these approaches do not totally match the MBDA setting. Therefore, we utilize the following rule for our comparison. For SSDA setting, one column denotes one SSDA task, such as R→C in Table 2. For MSDA methods that contain more than two source domains, we implement those methods according to their released codes, reset the source domain into two domains, such as R+S→C, and mark them with “\*”. Similarly, under the MTDA and BTDA settings, we reset the target into two domains and select the highest one in MTDA/BTDA task group that contains the same target domains, such as R→C+P and S→C+P in Table 2. For MMDA setting, two domains are sources, and the other domains are targets, such as R+S→C+P in Table 2. For better comparison, all results in Tables 2-4 are the averages of two target domains.

Results on the DomainNet are displayed in Table 2. Our SAUE method achieves SOTA performance in most of the experimental groups and attains the best performance in terms of average accuracy. Compared to the BTDA method MCDA in multi-source setting, our method achieves better performance because the style information of the target domain selected by the weight factor can enhance the source feature representations. Compared to MMDA method AMDA, although AMDA can access the domain labels of the target domains, our method still overpasses AMDA in terms of average classification accuracy (overpass 7.5%) and without the requirement of the domain labels of the target domains. Furthermore, both AMDA and our method are adversarial learning methods, and the comparison results further demonstrate the effectiveness of our adversarial learning strategy. These obtained improvements are mainly due to the uncertainty optimization process and the style information derived from target features.

Table 2: Accuracy (%) on the DomainNet for MBDA (ResNet-50). 

<table><tr><td rowspan="2">Method</td><td>R+S</td><td>S+P</td><td>P+R</td><td>C+S</td><td>R+C</td><td>C+P</td><td rowspan="2">Avg.</td></tr><tr><td>C+P</td><td>C+R</td><td>C+S</td><td>P+R</td><td>P+S</td><td>R+S</td></tr><tr><td>DANN[20] JMLR&#x27;16</td><td>31.4</td><td>39.7</td><td>26.8</td><td>29.3</td><td>31.3</td><td>31.2</td><td>31.6</td></tr><tr><td>DAN[18] TPAMI&#x27;19</td><td>32.8</td><td>40.6</td><td>28.2</td><td>29.8</td><td>31.5</td><td>32.0</td><td>32.5</td></tr><tr><td>MDAN[48] NeurIPS&#x27;18</td><td>54.5</td><td>59.0</td><td>45.0</td><td>58.8</td><td>51.7</td><td>61.0</td><td>54.5</td></tr><tr><td>MTDA[7] TIP&#x27;20</td><td>52.4</td><td>48.7</td><td>45.5</td><td>53.3</td><td>51.5</td><td>52.0</td><td>50.5</td></tr><tr><td>AMDA[29] TIP&#x27;21</td><td>65.8</td><td>67.8</td><td>56.7</td><td>65.1</td><td>58.9</td><td>66.4</td><td>63.4</td></tr><tr><td>DALN*[23] CVPR&#x27;22</td><td>61.2</td><td>69.2</td><td>64.1</td><td>63.5</td><td>59.3</td><td>64.8</td><td>63.7</td></tr><tr><td>MCDA*[13] AAAI&#x27;23</td><td>62.2</td><td>68.7</td><td>61.7</td><td>63.4</td><td>61.2</td><td>65.4</td><td>63.8</td></tr><tr><td>DGWA*[52] TMM&#x27;24</td><td>66.4</td><td>71.3</td><td>63.4</td><td>67.5</td><td>64.6</td><td>70.2</td><td>65.0</td></tr><tr><td>SAUE (Ours)</td><td>70.8</td><td>76.9</td><td>67.6</td><td>71.9</td><td>65.2</td><td>73.1</td><td>70.9</td></tr></table>

“\*” denotes that the results are obtained by the released code of the corresponding method. The best results are bolded.

Table 3: Accuracy (%) on the (a) Office-Home and the (b) ImageCLEF-DA for MBDA (ResNet-50).   
(a) Office-Home 

<table><tr><td rowspan="2">Method</td><td>Rw+Pr</td><td>Cl+Rw</td><td>Pr+Cl</td><td>Rw+Ar</td><td>Ar+Pr</td><td>Cl+Ar</td><td rowspan="2">Avg.</td></tr><tr><td>Ar+Cl</td><td>Ar+Pr</td><td>Ar+Rw</td><td>Cl+Pr</td><td>Cl+Rw</td><td>Pr+Rw</td></tr><tr><td>DANN[20] JMLR&#x27;16</td><td>53.5</td><td>61.9</td><td>53.5</td><td>55.6</td><td>57.1</td><td>60.1</td><td>57.6</td></tr><tr><td>DAN[18] TPAMI&#x27;19</td><td>53.4</td><td>60.1</td><td>52.2</td><td>54.3</td><td>52.2</td><td>58.7</td><td>56.3</td></tr><tr><td>MTDA[7] TIP&#x27;20</td><td>51.9</td><td>64.9</td><td>60.3</td><td>59.4</td><td>58.2</td><td>62.4</td><td>59.5</td></tr><tr><td>MDAN[48] NeurIPS&#x27;18</td><td>55.4</td><td>69.1</td><td>61.2</td><td>61.5</td><td>55.9</td><td>70.4</td><td>62.2</td></tr><tr><td>SCDA[22] CVPR&#x27;21</td><td>64.1</td><td>74.7</td><td>70.0</td><td>68.3</td><td>68.7</td><td>77.6</td><td>70.1</td></tr><tr><td>AMDA[29] TIP&#x27;21</td><td>61.4</td><td>77.0</td><td>72.3</td><td>67.4</td><td>64.9</td><td>77.4</td><td>70.0</td></tr><tr><td>HTA[51] Appl. Intell.&#x27;23</td><td>62.2</td><td>78.9</td><td>75.0</td><td>68.7</td><td>66.2</td><td>79.0</td><td>71.9</td></tr><tr><td>MCDA*[13] AAAI&#x27;23</td><td>63.6</td><td>74.9</td><td>70.0</td><td>68.7</td><td>68.1</td><td>78.1</td><td>70.6</td></tr><tr><td>DGWA[52] TMM&#x27;24</td><td>63.7</td><td>78.6</td><td>73.9</td><td>70.7</td><td>66.9</td><td>78.8</td><td>72.1</td></tr><tr><td>SAUE (Ours)</td><td>65.6</td><td>79.9</td><td>75.2</td><td>70.1</td><td>71.8</td><td>79.3</td><td>73.7</td></tr></table>

(b) ImageCLEF-DA

<table><tr><td rowspan="2">Method</td><td>i+p</td><td>p+c</td><td>c+i</td><td>b+p</td><td>i+b</td><td>b+c</td><td rowspan="2">Avg.</td></tr><tr><td>b+c</td><td>b+i</td><td>b+p</td><td>c+i</td><td>c+p</td><td>i+p</td></tr><tr><td>DANN[20] JMLR&#x27;16</td><td>76.4</td><td>72.4</td><td>69.1</td><td>87.9</td><td>82.9</td><td>79.3</td><td>77.9</td></tr><tr><td>DAN[18] TPAMI&#x27;19</td><td>78.3</td><td>74.8</td><td>70.3</td><td>91.5</td><td>85.0</td><td>78.8</td><td>79.8</td></tr><tr><td>CDAN*[21] NeurIPS&#x27;18</td><td>78.3</td><td>76.8</td><td>68.2</td><td>92.8</td><td>85.9</td><td>82.3</td><td>80.6</td></tr><tr><td>AMDA[29] TIP&#x27;21</td><td>78.8</td><td>77.3</td><td>71.7</td><td>92.3</td><td>85.2</td><td>83.8</td><td>81.5</td></tr><tr><td>SCDA*[22] CVPR&#x27;21</td><td>78.9</td><td>77.2</td><td>71.8</td><td>93.9</td><td>85.5</td><td>85.0</td><td>82.1</td></tr><tr><td>DIDA*[50] TIP&#x27;22</td><td>78.9</td><td>77.9</td><td>72.0</td><td>92.2</td><td>86.8</td><td>85.1</td><td>82.2</td></tr><tr><td>HTA[51] Appl. Intell.&#x27;23</td><td>79.3</td><td>78.2</td><td>72.3</td><td>92.8</td><td>85.6</td><td>84.9</td><td>82.2</td></tr><tr><td>MCDA*[13] AAAI&#x27;23</td><td>77.4</td><td>79.1</td><td>70.3</td><td>91.8</td><td>86.2</td><td>83.8</td><td>81.4</td></tr><tr><td>DGWA[52] TMM&#x27;24</td><td>79.7</td><td>79.1</td><td>72.7</td><td>93.8</td><td>86.0</td><td>84.5</td><td>82.7</td></tr><tr><td>SAUE (Ours)</td><td>80.8</td><td>80.2</td><td>73.8</td><td>94.9</td><td>88.6</td><td>87.2</td><td>84.3</td></tr></table>

“\*” denotes that the results are obtained by the released code of the corresponding method. The best results are bolded.

Results on the Office-Home are shown in Table 3a. The experimental results are compared with those of the SOTA methods, illustrating that our proposed method achieves dramatic improvements in most comparison groups and achieves the highest average classification accuracy (73.7%). Note that the Rw domain contains a total of 34,856 images, which is far more numerous than the other

Table 4: Accuracy (%) on the (a) default version of DomainNet dataset and (b) VisDA-2017 dataset (ResNet-101).   
(a) DomainNet   
(b) VisDA-2017 

<table><tr><td rowspan="2">Method</td><td>C+P+Q</td><td>C+P+R</td><td>C+P+S</td><td>C+Q+R</td><td>C+Q+S</td><td>C+R+S</td><td>P+Q+R</td><td>P+Q+S</td><td>P+R+S</td><td>Q+R+S</td><td rowspan="2">Avg.</td></tr><tr><td>R+S</td><td>Q+S</td><td>Q+R</td><td>P+S</td><td>P+R</td><td>P+Q</td><td>C+S</td><td>C+R</td><td>C+Q</td><td>C+P</td></tr><tr><td>MCDA</td><td>54.6</td><td>28.5</td><td>39.1</td><td>50.3</td><td>52.9</td><td>30.6</td><td>53.2</td><td>60.1</td><td>34.2</td><td>56.2</td><td>46.1</td></tr><tr><td>SCDA</td><td>54.2</td><td>31.0</td><td>40.5</td><td>51.8</td><td>56.2</td><td>30.4</td><td>54.9</td><td>59.3</td><td>36.0</td><td>57.0</td><td>47.1</td></tr><tr><td>DGWA</td><td>54.7</td><td>31.3</td><td>40.7</td><td>52.2</td><td>56.8</td><td>30.7</td><td>55.3</td><td>59.6</td><td>36.2</td><td>57.5</td><td>47.4</td></tr><tr><td>SAUE (Ours)</td><td>57.7</td><td>34.6</td><td>42.9</td><td>54.7</td><td>59.2</td><td>36.9</td><td>56.0</td><td>63.6</td><td>37.1</td><td>58.7</td><td>50.2</td></tr><tr><td rowspan="2">Method</td><td>R+S</td><td>Q+S</td><td>Q+R</td><td>P+S</td><td>P+R</td><td>P+Q</td><td>C+S</td><td>C+R</td><td>C+Q</td><td>C+P</td><td rowspan="2">Avg.</td></tr><tr><td>C+P+Q</td><td>C+P+R</td><td>C+P+S</td><td>C+Q+R</td><td>C+Q+S</td><td>C+R+S</td><td>P+Q+R</td><td>P+Q+S</td><td>P+R+S</td><td>Q+R+S</td></tr><tr><td>MCDA</td><td>40.0</td><td>54.6</td><td>50.2</td><td>33.4</td><td>41.2</td><td>52.9</td><td>45.3</td><td>40.2</td><td>48.3</td><td>41.2</td><td>44.7</td></tr><tr><td>SCDA</td><td>41.2</td><td>54.7</td><td>50.2</td><td>33.8</td><td>41.3</td><td>54.2</td><td>46.5</td><td>39.3</td><td>49.4</td><td>42.1</td><td>45.4</td></tr><tr><td>DGWA</td><td>41.5</td><td>55.1</td><td>50.7</td><td>34.3</td><td>42.6</td><td>54.7</td><td>46.9</td><td>39.7</td><td>49.8</td><td>42.6</td><td>45.8</td></tr><tr><td>SAUE (Ours)</td><td>43.3</td><td>57.7</td><td>53.1</td><td>37.7</td><td>44.6</td><td>57.5</td><td>50.5</td><td>41.2</td><td>51.3</td><td>45.3</td><td>48.2</td></tr></table>

<table><tr><td>Method</td><td>Syn.→Rel.</td></tr><tr><td>MCD</td><td>71.9</td></tr><tr><td>SWD</td><td>76.4</td></tr><tr><td>BNM</td><td>70.4</td></tr><tr><td>TSA</td><td>78.6</td></tr><tr><td>SCDA</td><td>79.7</td></tr><tr><td>DALN</td><td>80.6</td></tr><tr><td>DGWA</td><td>80.3</td></tr><tr><td>SAUE(ours)</td><td>81.5</td></tr></table>

three domains. Therefore, the adaptation task faces larger domain shifts and extremely unbalanced classes. The proposed method still achieves 2.8% improvements over AMDA and achieves dramatic improvements in the Rw+Pr→Ar+Cl and Rw+Ar→Pr+Cl tasks. These results occur because the proposed method decreases the impact of unbalanced classes by enhancing the feature representations and optimizing the prediction uncertainty.

Results on the ImageCLEF-DA are provided in Table 3b. Compared with the SOTA methods, our proposed method achieves an average accuracy of 84.3%, outperforming the existing approaches. Note that all four domains in ImageCLEF-DA contain 600 images. Therefore, the experimental results further demonstrate that our proposed method is effective when all the domains contain the same samples and classes.

Results on the Default Version of DomainNet and VisDA 2017. To evaluate the effectiveness of SAUE in different numbers of the source and target domains. We perform comparisons on the default version of the DomainNet dataset and the VisDA 2017 dataset, respectively. The default version of the DomainNet dataset consists of 5 domains, leading to the division of transfer tasks for MBDA into two categories: C+P+Q→R+S and R+S→C+P+Q. As shown in Table 4, SAUE outperforms the comparison methods across all transfer tasks, achieving the highest average classification accuracy. These results from large-scale datasets further demonstrate the superiority and flexibility of SAUE.

# 4.3 Experiment Analysis

![](images/1790fe4c1974e59d4133b7dd69197974c3e1c69a7e279a9210f3e5bb4341afbc.jpg)

<details>
<summary>heatmap</summary>

| λe   | 10   | 20   | 30   | 40   | 50   | 60   | 70   |
|------|------|------|------|------|------|------|------|
| 0.5  | 0.88 | 0.90 | 0.92 | 0.94 | 0.96 | 0.98 | 0.99 |
| 1.0  | 0.88 | 0.90 | 0.92 | 0.94 | 0.96 | 0.98 | 0.99 |
| 1.5  | 0.88 | 0.90 | 0.92 | 0.94 | 0.96 | 0.98 | 0.99 |
| 20   | 0.88 | 0.90 | 0.92 | 0.94 | 0.96 | 0.98 | 0.99 |
| 30   | 0.88 | 0.90 | 0.92 | 0.94 | 0.96 | 0.98 | 0.99 |
| 40   | 0.88 | 0.90 | 0.92 | 0.94 | 0.96 | 0.98 | 0.99 |
| 50   | 0.88 | 0.90 | 0.92 | 0.94 | 0.96 | 0.98 | 0.99 |
| 60   | 0.88 | 0.90 | 0.92 | 0.94 | 0.96 | 0.98 | 0.99 |
| 70   | 0.88 | 0.90 | 0.92 | 0.94 | 0.96 | 0.98 | 0.99 |
</details>

(a) b+i → c

![](images/b261bfc78a1b41d93d7cde8b9e32572b7158be999ae82764b6e484866938e196.jpg)

<details>
<summary>heatmap</summary>

| λσ   | 1.0  | 1.5  | 20   | 30   | 40   | 50   | 60   | 70   |
|------|------|------|------|------|------|------|------|------|
| 0.5  | 0.78 | 0.76 | 0.74 | 0.72 | 0.70 | 0.68 | 0.66 | 0.64 |
| 1.0  | 0.76 | 0.74 | 0.72 | 0.70 | 0.68 | 0.66 | 0.64 | 0.62 |
| 1.5  | 0.74 | 0.72 | 0.70 | 0.68 | 0.66 | 0.64 | 0.62 | 0.60 |
| 20   | 0.72 | 0.70 | 0.68 | 0.66 | 0.64 | 0.62 | 0.60 | 0.58 |
</details>

(b) b+i → p

Figure 2: The analysis of the SAUE parameters. 

<table><tr><td>Choices</td><td> $\lambda_{d}=0.1$ </td><td> $\lambda_{d}=0.5$ </td><td> $\lambda_{d}=1.0$ </td><td> $\lambda_{d}=1.5$ </td><td> $\lambda_{d}=2.0$ </td></tr><tr><td> $\lambda_{e}=10$ </td><td>96.5</td><td>96.7</td><td>96.8</td><td>96.6</td><td>96.5</td></tr><tr><td> $\lambda_{e}=20$ </td><td>96.8</td><td>97.0</td><td>97.1</td><td>97.0</td><td>96.9</td></tr><tr><td> $\lambda_{e}=40$ </td><td>97.0</td><td>97.2</td><td>97.3</td><td>97.2</td><td>97.2</td></tr><tr><td> $\lambda_{e}=60$ </td><td>94.7</td><td>94.9</td><td>95.2</td><td>95.1</td><td>94.9</td></tr><tr><td> $\lambda_{e}=80$ </td><td>88.7</td><td>93.3</td><td>93.4</td><td>93.2</td><td>89.8</td></tr></table>

(a) b+i → c

<table><tr><td>Choices</td><td> $\lambda_{d}=0.1$ </td><td> $\lambda_{d}=0.5$ </td><td> $\lambda_{d}=1.0$ </td><td> $\lambda_{d}=1.5$ </td><td> $\lambda_{d}=2.0$ </td></tr><tr><td> $\lambda_{e}=10$ </td><td>77.5</td><td>77.6</td><td>78.1</td><td>77.9</td><td>77.4</td></tr><tr><td> $\lambda_{e}=20$ </td><td>77.7</td><td>78.1</td><td>78.2</td><td>78.0</td><td>77.9</td></tr><tr><td> $\lambda_{e}=40$ </td><td>78.2</td><td>78.3</td><td>78.3</td><td>78.2</td><td>78.1</td></tr><tr><td> $\lambda_{e}=60$ </td><td>77.7</td><td>77.9</td><td>78.0</td><td>77.8</td><td>75.9</td></tr><tr><td> $\lambda_{e}=80$ </td><td>76.2</td><td>77.3</td><td>77.9</td><td>76.2</td><td>74.8</td></tr></table>

(b) b+i → p   
Table 5: The detailed numerical results corresponding to the relevant tasks.

Sensitivity Analysis. We evaluate the model's performance under different hyperparameter choices. Note that the hyperparameters in our method are the adversarial learning balance parameter $\lambda_{d}$ and annealing parameter $\lambda_{e}$ . As shown in Figure 2, we test different parameter groups to analyze the parameter sensitivity of our method, where $\lambda_{d} = \{0.1, 0.5, 1.0, 1.5, 2.0\}$ and $\lambda_{e} = \{10, 20, 40, 60, 80\}$ . The corresponding numerical results of Figure 2 are illustrated in Table 5. In Figure 2, our method is not sensitive to $\lambda_{d}$ and the best parameter choice is $\lambda_{d} = 1.0$ , but very sensitive to $\lambda_{e}$ (with $\lambda_{e} = 40$

working best); if $\lambda_{e}$ too small, the model will suffer from tremendous degradation due to the model is over-penalized by uncertainty loss.

Ablation Study. As listed in Table 6, we conduct ablation experiments to demonstrate the effectiveness of the style adaptation module and the loss function of the uncertainty optimization process. We test three experimental groups in the DomainNet dataset, including that: 1) remove the style adaptation (SA) module, 2) remove the uncertainty loss $L_{unc}$ , and 3) remove both of the above-mentioned items.

Table 6: Ablation study of SAUE on the Domain-Net. 

<table><tr><td rowspan="2">Source Target</td><td>R+S</td><td>S+P</td><td>P+R</td><td>C+S</td><td>R+C</td><td>C+P</td><td rowspan="2">Avg.</td></tr><tr><td>C+P</td><td>C+R</td><td>C+S</td><td>P+R</td><td>P+S</td><td>R+S</td></tr><tr><td>w/o SA</td><td>68.4</td><td>72.1</td><td>64.2</td><td>70.0</td><td>60.8</td><td>70.1</td><td>67.6</td></tr><tr><td>w/o  $\mathcal{L}_{unc}$ </td><td>69.7</td><td>75.0</td><td>65.8</td><td>71.2</td><td>64.8</td><td>72.6</td><td>69.9</td></tr><tr><td>w/o both</td><td>65.1</td><td>71.9</td><td>63.8</td><td>68.8</td><td>60.2</td><td>69.3</td><td>66.5</td></tr><tr><td>w/o WD</td><td>70.2</td><td>77.0</td><td>66.3</td><td>71.6</td><td>64.1</td><td>71.5</td><td>70.1</td></tr><tr><td>SAUE</td><td>70.8</td><td>76.9</td><td>67.6</td><td>71.9</td><td>65.2</td><td>73.1</td><td>70.9</td></tr></table>

The results illustrate that both style adaptation and prediction uncertainty optimization are useful for MBDA. Furthermore, we explore different style adaptation techniques. We directly change the Wasserstein Distance (WD) to randomly selected style information and report the obtained results in the fourth row of middle part of Table 6. Compared with the random augmentation methods, our proposed method can better select the style information through weight factors, which enhances the feature representations of the source domains and reduces the impact of domain shifts. More experiment analysis is provided in Appendix E.

Table 7: Comparison about the SA module on the DomainNet dataset with different backbones. 

<table><tr><td rowspan="2">Setting</td><td>C+P+Q</td><td>C+P+R</td><td>C+P+S</td><td>C+Q+R</td><td>C+Q+S</td><td>C+R+S</td><td>P+Q+R</td><td>P+Q+S</td><td>P+R+S</td><td>Q+R+S</td><td rowspan="2">Avg.</td></tr><tr><td>R+S</td><td>Q+S</td><td>Q+R</td><td>P+S</td><td>P+R</td><td>P+Q</td><td>C+S</td><td>C+R</td><td>C+Q</td><td>C+P</td></tr><tr><td>without SA (ResNet-50)</td><td>51.2</td><td>28.6</td><td>36.7</td><td>49.6</td><td>52.9</td><td>30.2</td><td>51.0</td><td>57.8</td><td>30.3</td><td>51.1</td><td>43.9</td></tr><tr><td>with SA (ResNet-50)</td><td>53.9</td><td>30.7</td><td>39.8</td><td>51.3</td><td>55.7</td><td>33.7</td><td>52.8</td><td>60.4</td><td>33.7</td><td>55.3</td><td>46.7 (+2.8)</td></tr><tr><td>without SA (ResNet-101)</td><td>56.1</td><td>33.7</td><td>41.3</td><td>53.1</td><td>57.3</td><td>35.1</td><td>54.8</td><td>61.2</td><td>36.4</td><td>56.7</td><td>48.6</td></tr><tr><td>with SA (ResNet-101)</td><td>57.7</td><td>34.6</td><td>42.9</td><td>54.7</td><td>59.2</td><td>36.9</td><td>56.0</td><td>63.6</td><td>37.1</td><td>58.7</td><td>50.2 (+1.6)</td></tr><tr><td rowspan="2">Setting</td><td>R+S</td><td>Q+S</td><td>Q+R</td><td>P+S</td><td>P+R</td><td>P+Q</td><td>C+S</td><td>C+R</td><td>C+Q</td><td>C+P</td><td rowspan="2">Avg.</td></tr><tr><td>C+P+Q</td><td>C+P+R</td><td>C+P+S</td><td>C+Q+R</td><td>C+Q+S</td><td>C+R+S</td><td>P+Q+R</td><td>P+Q+S</td><td>P+R+S</td><td>Q+R+S</td></tr><tr><td>without SA (ResNet-50)</td><td>39.5</td><td>53.2</td><td>47.3</td><td>33.2</td><td>40.1</td><td>52.4</td><td>45.8</td><td>38.2</td><td>45.3</td><td>40.5</td><td>43.5</td></tr><tr><td>with SA (ResNet-50)</td><td>41.1</td><td>55.3</td><td>49.8</td><td>35.6</td><td>42.5</td><td>54.7</td><td>48.3</td><td>39.5</td><td>47.7</td><td>43.2</td><td>45.8 (+2.3)</td></tr><tr><td>without SA (ResNet-101)</td><td>42.1</td><td>56.5</td><td>51.6</td><td>35.2</td><td>43.8</td><td>57.0</td><td>49.4</td><td>39.9</td><td>50.7</td><td>43.1</td><td>46.9</td></tr><tr><td>with SA (ResNet-101)</td><td>43.3</td><td>57.7</td><td>53.1</td><td>37.7</td><td>44.6</td><td>57.5</td><td>50.5</td><td>41.2</td><td>51.3</td><td>45.3</td><td>48.2 (+1.3)</td></tr></table>

Effectiveness of the SA Module with Different Backbones. We have analyzed the performance of our method using the ResNet-50 and ResNet-101 backbones in the default version of the DomainNet dataset in Table 7. The experimental results show that our method achieves significant performance gains when utilizing powerful backbones (ResNet-101). We compared the performance gains of the style adaptation module with standard backbone (ResNet-50) and powerful backbone (ResNet-101). The results indicate that the performance gains (+2.8% and +2.3%) of the style adaptation module when integrated into the ResNet-50 backbone surpass those (+1.6% and +1.3%) when integrated into the ResNet-101 backbone.

Effectiveness of the SAUE with ResNet and ViT backbones. We further evaluate the model's performance using different backbones, as shown in Table 8. When using ViT as the backbone, SAUE outperforms its performance with the ResNet-50 backbone. This performance gain is primarily due to the larger number of tunable parameters in ViT-B/16, demonstrating that our method effectively leverages these additional parameters to exploit transferable knowledge from multiple domains.

Table 8: Comparison about different backbones on the Office-Home dataset. 

<table><tr><td rowspan="2">Method</td><td>Rw+Pr</td><td>Cl+Rw</td><td>Pr+Cl</td><td>Rw+Ar</td><td>Ar+Pr</td><td>Cl+Ar</td><td rowspan="2">Avg.</td></tr><tr><td>Ar+Cl</td><td>Ar+Pr</td><td>Ar+Rw</td><td>Cl+Pr</td><td>Cl+Rw</td><td>Pr+Rw</td></tr><tr><td>ResNet-50</td><td>65.6</td><td>79.9</td><td>75.2</td><td>70.1</td><td>71.8</td><td>79.3</td><td>73.7</td></tr><tr><td>ViT-B/16</td><td>69.2</td><td>83.1</td><td>79.1</td><td>73.6</td><td>74.5</td><td>83.7</td><td>77.2</td></tr></table>

# 5 Conclusion

In this paper, we propose a SAUE approach for MBDA, which utilizes information from multiple source domains to adapt a blended-target domain. In particular, the style adaptation process utilizes similarity factors to select target style information to enhance the representations of the source features. The uncertainty estimation procedure utilizes the Dirichlet distribution to estimate the uncertainty of the model and then adopts the KL divergence measure to optimize the prediction uncertainty. The discriminator-free adversarial learning strategy is beneficial for MBDA. Extensive experimental results demonstrate the superior performance of SAUE to that of the competing methods.

# Acknowledgment

This work was supported by the National Natural Science Foundation of China (Grant No. 62176162) and the Guangdong Basic and Applied Basic Research Foundation (2023A1515012875, 2022A1515140099).

# References

[1] Rui Wang, Zuxuan Wu, Zejia Weng, Jingjing Chen, Guo-Jun Qi, and Yu-Gang Jiang. Cross-domain contrastive learning for unsupervised domain adaptation. IEEE TMM, 25:1665–1673, 2023.   
[2] Zhongyi Han, Haoliang Sun, and Yilong Yin. Learning transferable parameters for unsupervised domain adaptation. IEEE TIP, 31:6424–6439, 2022.   
[3] Yiting Cheng, Fangyun Wei, Jianmin Bao, Dong Chen, and Wenqiang Zhang. Adpl: Adaptive dual path learning for domain adaptation of semantic segmentation. IEEE TPAMI, 45(8):9339–9356, 2023.   
[4] Munan Ning, Donghuan Lu, Yujia Xie, Dongdong Chen, Dong Wei, Yefeng Zheng, Yonghong Tian, Shuicheng Yan, and Li Yuan. Madav2: Advanced multi-anchor based active domain adaptation segmentation. IEEE TPAMI, 45(8):13553–13566, 2023.   
[5] Poojan Oza, Vishwanath A. Sindagi, Vibashan Vishnukumar Sharmini, and Vishal M. Patel. Unsupervised domain adaptation of object detectors: A survey. IEEE TPAMI, 2023.   
[6] Jinjing Zhu, Haotian Bai, and Lin Wang. Patch-mix transformer for unsupervised domain adaptation: A game perspective. In CVPR, pages 3561-3571, 2023.   
[7] Behnam Gholami, Pritish Sahu, Ognjen Rudovic, Konstantinos Bousmalis, and Vladimir Pavlovic. Unsupervised multi-target domain adaptation: An information theoretic approach. IEEE TIP, 29:3993–4002, 2020.   
[8] Subhankar Roy, Evgeny Krivosheev, Zhun Zhong, Nicu Sebe, and Elisa Ricci. Curriculum graph co-teaching for multi-target domain adaptation. In CVPR, pages 5347-5357, 2021.   
[9] Jiazhong Zhou, Qing Tian, and Zhanghu Lu. Progressive decoupled target-into-source multi-target domain adaptation. INS, 634:140–156, 2023.   
[10] Xu Yang, Cheng Deng, Tongliang Liu, and Dacheng Tao. Heterogeneous graph attention network for unsupervised multiple-target domain adaptation. IEEE TPAMI, 44(4):1992–2003, 2022.   
[11] Takashi Isobe, Xu Jia, Shuaijun Chen, Jianzhong He, Yongjie Shi, Jianzhuang Liu, Huchuan Lu, and Shengjin Wang. Multi-target domain adaptation with collaborative consistency learning. In CVPR, pages 8183–8193, 2021.   
[12] Ziliang Chen, Jingyu Zhuang, Xiaodan Liang, and Liang Lin. Blending-target domain adaptation by adversarial meta-adaptation networks. In CVPR, pages 2248–2258, 2019.   
[13] Pengcheng Xu, Boyu Wang, and Charles Ling. Class overwhelms: Mutual conditional blended-target domain adaptation. In AAAI, pages 3036-3044, 2023.   
[14] Xingchao Peng, Qinxun Bai, Xide Xia, Zijun Huang, Kate Saenko, and Bo Wang. Moment matching for multi-source domain adaptation. In CVPR, pages 1406–1415, 2019.   
[15] Keqiuyin Li, Jie Lu, Hua Zuo, and Guangquan Zhang. Dynamic classifier alignment for unsupervised multi-source domain adaptation. IEEE TKDE, 35(5):4727–4740, 2023.   
[16] Keqiuyin Li, Jie Lu, Hua Zuo, and Guangquan Zhang. Multi-source contribution learning for domain adaptation. IEEE TNNLS, 33(10):5293–5307, 2021.   
[17] Mingsheng Long, Han Zhu, Jianmin Wang, and Michael I. Jordan. Unsupervised domain adaptation with residual transfer networks. In NeurIPS, pages 136–144, 2016.   
[18] Mingsheng Long, Yue Cao, Zhangjie Cao, Jianmin Wang, and Michael I. Jordan. Transferable representation learning with deep adaptation networks. IEEE TPAMI, 41(12):3071–3085, 2019.   
[19] Binhui Xie, Shuang Li, Fangrui Lv, Chi Harold Liu, Guoren Wang, and Dapeng Wu. A collaborative alignment framework of transferable knowledge extraction for unsupervised domain adaptation. IEEE TKDE, 35(7):6518–6533, 2023.

[20] Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, Francois Laviolette, Mario Marchand, and Victor Lempitsky. Domain-adversarial training of neural networks. JMLR, 17(1):2096–2030, 2016.   
[21] Mingsheng Long, Zhangjie Cao, Jianmin Wang, and Michael I. Jordan. Conditional adversarial domain adaptation. In NeurIPS, pages 1640–1650, 2018.   
[22] Shuang Li, Mixue Xie, Fangrui Lv, Chi Harold Liu, Jian Liang, Chen Qin, and Wei Li. Semantic concentration for domain adaptation. In ICCV, pages 9102–9112, 2021.   
[23] Lin Chen, Huaian Chen, Zhixiang Wei, Xin Jin, Xiao Tan, Yi Jin, and Enhong Chen. Reusing the task-specific classifier as a discriminator: discriminator-free adversarial domain adaptation. In CVPR, pages 7171–7181, 2022.   
[24] Chen-Yu Lee, Tanmay Batra, Mohammad Haris Baig, and Daniel Ulbricht. Sliced wasserstein discrepancy for unsupervised domain adaptation. In CVPR, pages 10285-10295, 2019.   
[25] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In NeurIPS, pages 2672–2680, 2014.   
[26] Antonia Creswell, Tom White, Vincent Dumoulin, Kai Arulkumaran, Biswa Sengupta, and Anil A. Bharath. Generative adversarial networks: An overview. IEEE MSP, 35(1):53–65, 2018.   
[27] Yongchun Zhu, Fuzhen Zhuang, and Deqing Wang. Aligning domain-specific distribution and classifier for cross-domain classification from multiple sources. In AAAI, pages 5989-5996, 2019.   
[28] Haozhe Feng, Zhaoyang You, Minghao Chen, Tianye Zhang, Minfeng Zhu, Fei Wu, Chao Wu, and Wei Chen. Kd3a: Unsupervised multi-source decentralized domain adaptation via knowledge distillation. In ICML, pages 3274–3283, 2021.   
[29] Yuxi Wang, Zhaoxiang Zhang, Wangli Hao, and Chunfeng Song. Attention guided multiple source and target domain adaptation. IEEE TIP, 30:892–906, 2021.   
[30] Kaiyang Zhou, Yongxin Yang, Yu Qiao, and Tao Xiang. Mixstyle neural networks for domain generalization and adaptation. IJCV, 132(3):822–836, 2024.   
[31] SS Vallender. Calculation of the wasserstein distance between probability distributions on the line. Theory Probab. its Appl., 18(4):784–786, 1974.   
[32] Murat Sensoy, Lance Kaplan, and Melih Kandemir. Evidential deep learning to quantify classification uncertainty. In NeurIPS, pages 3179-3189, 2018.   
[33] Zongbo Han, Changqing Zhang, Huazhu Fu, and Joey Tianyi Zhou. Trusted multi-view classification with dynamic evidential fusion. IEEE TPAMI, 45(2):2551–2566, 2022.   
[34] Zheyao Gao, Yuanye Liu, Fuping Wu, Nannan Shi, Yuxin Shi, and Xiahai Zhuang. A reliable and interpretable framework of multi-view learning for liver fibrosis staging. In MICCAI, pages 178–188. Springer, 2023.   
[35] Ying Jin, Ximei Wang, Mingsheng Long, and Jianmin Wang. Minimum class confusion for versatile domain adaptation. In ECCV, pages 464-480. Springer, 2020.   
[36] David A. McAllester. Some pac-bayesian theorems. Mach. Learn., 37(3):355-363, 1999.   
[37] Pierre Alquier, James Ridgway, and Nicolas Chopin. On the properties of variational approximations of gibbs posteriors. JMLR, 17(1):8374–8414, 2016.   
[38] Pascal Germain, Francis Bach, Alexandre Lacoste, and Simon Lacoste-Julien. Pac-bayesian theory meets bayesian inference. In NeurIPS, pages 1884–1892, 2016.   
[39] Shai Ben-David, John Blitzer, Koby Crammer, and Fernando Pereira. Analysis of representations for domain adaptation. In NeurIPS, pages 137-144, 2007.   
[40] Barbara Caputo, Henning Müller, Jesus Martinez-Gomez, Mauricio Villegas, Burak Acar, Novi Patricia, Neda Marvasti, Suzan Üsküdarlı, Roberto Paredes, and Miguel Cazorla. Imageclef 2014: Overview and analysis of the results. In ICCLEF, pages 192–211, 2014.   
[41] Hemanth Venkateswara, Jose Eusebio, Shayok Chakraborty, and Sethuraman Panchanathan. Deep hashing network for unsupervised domain adaptation. In CVPR, pages 5018-5027, 2017.

[42] Xingchao Peng, Ben Usman, Neela Kaushik, Judy Hoffman, Dequan Wang, and Kate Saenko. Visda: The visual domain adaptation challenge. arXiv preprint, 2017. arXiv:1710.06924.   
[43] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, and Luca Antiga. Pytorch: An imperative style, high-performance deep learning library. In NeurIPS, pages 8026–8037, 2019.   
[44] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, pages 770-778, 2016.   
[45] Kuniaki Saito, Kohei Watanabe, Yoshitaka Ushiku, and Tatsuya Harada. Maximum classifier discrepancy for unsupervised domain adaptation. In CVPR, pages 3723-3732, 2018.   
[46] Shuang Li, Mixue Xie, Kaixiong Gong, Chi Harold Liu, Yulin Wang, and Wei Li. Transferable semantic augmentation for domain adaptation. In CVPR, pages 11516-11525, 2021.   
[47] Thomas Westfechtel, Hao-Wei Yeh, Qier Meng, Yusuke Mukuta, and Tatsuya Harada. Backprop induced feature weighting for adversarial domain adaptation with iterative label distribution alignment. In WACV, pages 392–401, 2023.   
[48] Han Zhao, Shanghang Zhang, Guanhang Wu, José M. F. Moura, Joao P. Costeira, and Geoffrey J. Gordon. Adversarial multiple source domain adaptation. In NeurIPS, pages 8559–8570, 2018.   
[49] Ruijia Xu, Ziliang Chen, Wangmeng Zuo, Junjie Yan, and Liang Lin. Deep cocktail network: Multi-source unsupervised domain adaptation with category shift. In CVPR, pages 3964–3973, 2018.   
[50] Zhongying Deng, Kaiyang Zhou, Da Li, Junjun He, Yi-Zhe Song, and Tao Xiang. Dynamic instance domain adaptation. IEEE TIP, 31:4585–4597, 2022.   
[51] Zhuanghui Wu, Min Meng, Tianyou Liang, and Jigang Wu. Hierarchical triple-level alignment for multiple source and target domain adaptation. Appl. Intell., 53:3766–3782, 2023.   
[52] Yuwu Lu, Haoyu Huang, Biqing Zeng, Zhihui Lai, and Xuelong Li. Multi-source and multi-target domain adaptation based on dynamic generator with attention. IEEE TMM, 26:6891–6905, 2024.   
[53] Mixue Xie, Shuang Li, Rui Zhang, and Chi Harold Liu. Dirichlet-based uncertainty calibration for active domain adaptation. In ICLR, 2023.   
[54] Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. JMLR, 9:2579–2605, 2008.

# Appendix Contents

This supplementary material provides more details that are not presented in the main paper due to space limitations. The organization is as follows:

A Broader Impacts & Limitations   
B provides the derivations of $P(y = c|x_i) = \mathbb{E}[D(p_{ic}|\alpha_i)]$ .   
C provides the proof and derivations of Theorem 1.   
D provides the proof and derivations of the generalization bound.   
E provides additional experiment analysis on the ImageCLEF-DA dataset.

# A Broader Impacts & Limitations

Our work focuses on the problem of multi-source blended-target domain adaptation (MBDA), which aims to train a model that generalizes well on an unlabeled and distribution-confused target domain by leveraging multiple labeled source domains. The effectiveness of our method on several real-world datasets suggests that it can benefit relevant applications and communities dealing with domain shifts, such as encrypted data analysis, medical imaging, and autonomous driving. Nevertheless, we should also be cautious about potential failures of our method when encountering more significant distribution shifts, such as an increased number of source or target domains. In the future, we plan to incorporate more sub-domains into our experiments for further verifying the performance of our method.

# B Derivations of $P(y = c|x_i) = \mathbb{E}[D(p_{ic}|\alpha_i)]$

Given sample $x_{i}$ , for prediction of class c that generated by DNNs can be calculated as:

$$
\begin{array}{l} P (y = c | x _ {i}) \\ = \int \rho (y = c | p _ {i}) \rho (p _ {i} | x _ {i}) d p _ {i} \\ = \int p _ {i c} \cdot \rho (p _ {i} | x _ {i}) d p _ {i} \tag {1} \\ = \int \int \dots \int p _ {i c} \cdot \rho (p _ {i 1}, p _ {i 2, \ldots , p _ {i K}} | x _ {i}) d p _ {i 1} d p _ {i 2} \dots d p _ {i K} \\ = \int p _ {i c} \cdot \rho (p _ {i c} | x _ {i}) d p _ {i c}, \\ \end{array}
$$

where $p_{i}=C(G(x_{i}))=[p_{i1},p_{i2},\ldots,p_{iK}]$ and $p_{ic}$ is the c-th element of $p_{i}$ . Then, given $\rho(p_{i}|x_{i})\sim D(p_{i}|\alpha_{i})$ , we have $\rho(p_{ic}|x_{i})\sim Beta(p_{ic}|\alpha_{ic},\alpha_{i0}-\alpha_{ic})$ , where $\alpha_{i0}=\sum_{k=1}^{K}\alpha_{ik}$ . Thus, we further have:

$$
\rho (p _ {i c} | x _ {i}) = \frac {1}{B (\alpha_ {i c} , \alpha_ {i 0} - \alpha_ {i c})} p _ {i c} ^ {\alpha_ {i c} - 1} (1 - p _ {i c}) ^ {\alpha_ {i 0} - \alpha_ {i c} - 1}, \tag {2}
$$

where $B(\cdot, \cdot)$ is the $K$ -dimensional multinomial beta function, and $B(\alpha_{ic}, \alpha_{i0} - \alpha_{ic}) = \frac{\Gamma(\alpha_{ic})\Gamma(\alpha_{i0} - \alpha_{ic})}{\Gamma(\alpha_{ic} + \alpha_{i0} - \alpha_{ic})}$ , $\Gamma(\cdot)$ denotes the gamma function. Based on this, we can further obtain the following

derivation:

$$
\begin{array}{l} P (y = c | x _ {i}) = \int p _ {i c} \cdot \rho (p _ {i c} | x _ {i}) d p _ {i c} \\ = \int p _ {i c} \cdot \left[ \frac {1}{B (\alpha_ {i c} , \alpha_ {i 0} - \alpha_ {i c})} p _ {i c} ^ {\alpha_ {i c} - 1} (1 - p _ {i c}) ^ {\alpha_ {i 0} - \alpha_ {i c} - 1} \right] d p _ {i c} \\ = \frac {B (\alpha_ {i c} + 1 , \alpha_ {i 0} - \alpha_ {i c})}{B (\alpha_ {i c} , \alpha_ {i 0} - \alpha_ {i c})}. \\ \int \frac {1}{B \left(\alpha_ {i c} + 1 , \alpha_ {i 0} - \alpha_ {i c}\right)} p _ {i c} ^ {\alpha_ {i c}} \left(1 - p _ {i c}\right) ^ {\alpha_ {i 0} - \alpha_ {i c} - 1} d p _ {i c} \tag {3} \\ = \frac {B (\alpha_ {i c} + 1 , \alpha_ {i 0} - \alpha_ {i c})}{B (\alpha_ {i c} , \alpha_ {i 0} - \alpha_ {i c})} \cdot 1 \\ = \frac {\Gamma (\alpha_ {i c} + 1) \Gamma (\alpha_ {i 0})}{\Gamma (\alpha_ {i 0} + 1) \Gamma (\alpha_ {i c})} \\ = \frac {\alpha_ {i c} \Gamma (\alpha_ {i c}) \Gamma (\alpha_ {i 0})}{\alpha_ {i 0} \Gamma (\alpha_ {i 0}) \Gamma (\alpha_ {i c})} = \frac {\alpha_ {i c}}{\sum_ {k = 1} ^ {K} \alpha_ {i k}} = \frac {C _ {c} (G (x _ {i}))}{\sum_ {k = 1} ^ {K} C _ {k} (G (x _ {i}))} \\ = \mathbb {E} [ D (p _ {i c} | \alpha_ {i}) ]. \\ \end{array}
$$

In our work, the outputs of classifier are adopting the exponential function. Thus, following $[53]$ , the outputs of DNNs in SAUE can be viewed as the expectation of Dirichlet distribution.

# C Proof of Theorem 1

Theorem 1 [37]. Suppose we have given the m-th source data distribution $P_{S_{m}}$ , a hypothesis set H, and a prior distribution $\pi$ over the hypothesis space $\Theta$ . For any $\tau \in (0,1]$ and $\lambda > 0$ , with a probability at least $1 - \tau$ over the source samples $S_{m} \sim P_{S_{m}}^{n}$ , for all posteriors $\rho$ , we have:

$$
\mathbb {E} _ {\rho (\mathcal {H})} [ \mathcal {L} (\mathcal {H}) ] \leq \mathbb {E} _ {\rho (\mathcal {H})} [ \tilde {\mathcal {L}} _ {\mathcal {S} _ {m}} (\mathcal {H}) ] + \frac {1}{\lambda} \left[ K L (\rho \| \pi) + \log \frac {1}{\tau} + \Psi_ {\mathcal {S} _ {m}, \pi} (\lambda , n) \right], \tag {4}
$$

where $\Psi_{\mathcal{S}_m,\pi}(\lambda ,n) = \log \mathbb{E}_{\pi (\mathcal{H})}\mathbb{E}_{\mathcal{S}_m\sim P_{\mathcal{S}_m}^n}\left[e^{\lambda (\mathcal{L}(\mathcal{H}) - \mathcal{L}(\tilde{\mathcal{H}}))}\right].$

Lemma 1 [38]. The PAC-Bayes bound, involving constants $\tau$ and n, as introduced in Theorem 1, is minimized by the Bayesian posterior $p(\mathcal{H})$ , which represents the distribution over $\Theta$ .

Proof. The Donsker-Varadhan's change of measure states that for any measurable function $\phi : \Theta \to \mathbb{R}$ , we have:

$$
\mathbb {E} _ {\rho} (\mathcal {H}) \leq K L (\rho \| \pi) + \log \mathbb {E} _ {\pi (\mathcal {H})} [ e ^ {\phi (\mathcal {H})} ]. \tag {5}
$$

Thus, with $\phi(\mathcal{H}):=\lambda(L(\mathcal{H}-\hat{L}(\theta,\mathcal{S}_{m})\text{ and }\forall\rho\text{ over hypothesis space }\Theta,\text{ we have:}$

$$
\mathbb {E} _ {\rho} (\mathcal {H}) \left[ \lambda (L (\mathcal {H}) - \hat {L} (\mathcal {H}, \mathcal {S} _ {m})) \right] = \lambda \left(\mathbb {E} _ {\rho (\mathcal {H})} [ L (\mathcal {H}) ] - \mathbb {E} _ {\rho (\mathcal {H})} [ \hat {L} (\mathcal {H}, \mathcal {S} _ {m}) ]\right) \tag {6}
$$

$$
\leq K L (\rho \| \pi) + \log \mathbb {E} _ {\pi (\mathcal {H})} [ e ^ {\lambda (L (\mathcal {H}) - \hat {L} (\mathcal {H}, \mathcal {S} _ {m}))} ].
$$

For the non-negative random variable $\zeta_{\pi}(\mathcal{S}_{m}) := \mathbb{E}_{\pi(\mathcal{H})}[e^{\lambda(L(\mathcal{H}) - \tilde{L}(\mathcal{H},\mathcal{S}_{m}))}]$ , we apply Markov's inequality on it, and have:

$$
\mathbb {P} \left(\zeta \leq \frac {1}{\tau} \mathbb {E} _ {\mathcal {S} _ {m} \sim P _ {\mathcal {S} _ {m}} ^ {n}} \left[ \zeta_ {\pi} (\mathcal {S} _ {m}) \right]\right) \geq 1 - \tau . \tag {7}
$$

This implies that with probability at least $1 - \tau$ over the choice of $\mathcal{S}_m \sim P_{\mathcal{S}_m}^n$ , we have $\forall \rho$ over hypothesis space $\Theta$ :

$$
\mathbb {P} \left(\mathbb {E} _ {\rho (\mathcal {H})} [ \mathcal {L} (\mathcal {H}) ] \leq \mathbb {E} _ {\rho (\mathcal {H})} [ \hat {\mathcal {L}} _ {\mathcal {S} _ {m}} (\mathcal {H}) ] + \frac {1}{\lambda [ K L (\rho \| \pi) + \log \frac {1}{\tau} + \Psi_ {\mathcal {S} _ {m} , \pi} (\lambda , n) ]}\right) \geq 1 - \tau , \tag {8}
$$

where $\Psi_{\mathcal{S}_m,\pi}(\lambda ,n) = \log \mathbb{E}_{\pi (\mathcal{H})}\mathbb{E}_{\mathcal{S}_m\sim P_{\mathcal{S}_m}^n}\left[e^{\lambda (\mathcal{L}(\mathcal{H}) - \mathcal{L}(\tilde{\mathcal{H}}))}\right]$ , and we prove the statement of the Theorem 1.

# D Generalization Bound

Lemma 2 [39]. Suppose we have given the probability measures $\nu_{\mathcal{S}_m}, \nu_\mathcal{T} \in \mathcal{P}(\mathcal{F})$ of the $m$ -th source feature $f_{\mathcal{S}_m}$ and the blended-target domain feature $f_\mathcal{T}$ , a hypothesis space $\Theta$ , and a subspace $\tilde{\mathcal{H}} \in \Theta$ . Let $\mathcal{F}$ denote a fixed representation space and $c(f_{\mathcal{S}_m}, f_\mathcal{T})$ denote the adaptation cost. For the ideal classifier $h' \in \tilde{\mathcal{H}}$ and any classifier $h \in \tilde{\mathcal{H}}$ with $f_{\mathcal{S}_m} \sim \nu_{\mathcal{S}_m}$ and $f_\mathcal{T} \sim \nu_\mathcal{T}$ , we have:

$$
\left| \epsilon_ {\mathcal {S} _ {m}} (h, h ^ {\prime}) - \epsilon_ {\mathcal {T}} (h, h ^ {\prime}) \right| \leq \frac {1}{2} d _ {\mathcal {H} \Delta \mathcal {H}} (\nu_ {\mathcal {S} _ {m}}, \nu_ {\mathcal {T}}), \tag {9}
$$

where $\epsilon_{S_{m}}$ and $\epsilon_{T}$ denote the error on the m-th source domain and the error on the blended-target domain respectively, and $\epsilon_{T} = \frac{1}{N} \sum_{j=1}^{N} \epsilon_{T_{j}}$ . $d_{H\Delta H}$ denotes the $H\Delta H$ -distance.

Proof. By the definition of $\mathcal{H}\Delta \mathcal{H}$ -distance, we have:

$$
\begin{array}{l} d _ {\mathcal {H} \Delta \mathcal {H}} (\nu_ {\mathcal {S} _ {m}}, \nu_ {\mathcal {T}}) = 2 \sup _ {h, h ^ {\prime} \in \mathcal {H}} | \operatorname * {P r} _ {x \sim \nu_ {\mathcal {S} _ {m}}} [ h (x) \neq h ^ {\prime} (x) ] - \operatorname * {P r} _ {x \sim \nu_ {\mathcal {T}}} [ h (x) \neq h ^ {\prime} (x) ] | \\ = 2 \sup _ {h, h ^ {\prime} \in \mathcal {H}} | \epsilon_ {\mathcal {S} _ {m}} (h, h ^ {\prime}) - \epsilon_ {\mathcal {T}} (h, h ^ {\prime}) | \geq 2 | \epsilon_ {\mathcal {S} _ {m}} (h, h ^ {\prime}) - \epsilon_ {\mathcal {T}} (h, h ^ {\prime}) |. \tag {10} \\ \end{array}
$$

Theorem 2. Based on Lemma 2, with the error of the ideal joint hypothesis $\eta' = \epsilon_{\mathcal{S}_m}(h') + \epsilon_{\mathcal{T}}(h')$ which is a sufficiently small constant, for any $\delta \in (0,1)$ , with probability at least $1 - \delta$ , for every $h \in \mathcal{H}$ , $\epsilon_{\mathcal{T}}(h)$ is bounded by the following terms:

$$
\epsilon_ {\mathcal {T}} (h) \leq \epsilon_ {\mathcal {S} _ {m}} (h) + \frac {1}{2} \hat {d} _ {\mathcal {H} \Delta \mathcal {H}} (\nu_ {\mathcal {S} _ {m}}, \nu_ {\mathcal {T}}) + 4 \sqrt {\frac {2 d \log (2 b ^ {\prime}) + \log (\frac {2}{\delta})}{b ^ {\prime}}} + \eta^ {\prime}, \tag {11}
$$

where $\eta' = \epsilon_{\mathcal{S}_{m}}(h') + \epsilon_{\mathcal{T}}(h')$ is the ideal error for the classifier, which is a sufficiently small constant. $b'$ is the size of unlabeled samples.

Proof. From Lemma 2, we can obtain the following terms:

$$
\begin{array}{l} \epsilon_ {\mathcal {T}} (h) \leq \epsilon_ {\mathcal {T}} (h ^ {\prime}) + \epsilon_ {\mathcal {T}} (h, h ^ {\prime}) \\ \leq \epsilon_ {\mathcal {T}} (h ^ {\prime}) + \epsilon_ {\mathcal {S} _ {m}} (h, h ^ {\prime}) + | \epsilon_ {\mathcal {T}} (h, h ^ {\prime}) - \epsilon_ {\mathcal {S} _ {m}} (h, h ^ {\prime}) | \\ \leq \epsilon_ {\mathcal {T}} (h ^ {\prime}) + \epsilon_ {\mathcal {S} _ {m}} (h, h ^ {\prime}) + \frac {1}{2} d _ {\mathcal {H} \Delta \mathcal {H}} (\nu_ {\mathcal {S} _ {m}}, \nu_ {\mathcal {T}}) \\ \leq \epsilon_ {\mathcal {T}} (h ^ {\prime}) + \epsilon_ {\mathcal {S} _ {m}} (h) + \epsilon_ {\mathcal {S} _ {m}} (h ^ {\prime}) + \frac {1}{2} d _ {\mathcal {H} \Delta \mathcal {H}} (\nu_ {\mathcal {S} _ {m}}, \nu_ {\mathcal {T}}) \tag {12} \\ = \epsilon_ {\mathcal {S} _ {m}} (h) + \frac {1}{2} d _ {\mathcal {H} \Delta \mathcal {H}} (\nu_ {\mathcal {S} _ {m}}, \nu_ {\mathcal {T}}) + \eta^ {\prime} \\ \leq \epsilon_ {\mathcal {S} _ {m}} (h) + \frac {1}{2} d _ {\mathcal {H} \Delta \mathcal {H}} (\nu_ {\mathcal {S} _ {m}}, \nu_ {\mathcal {T}}) + 4 \sqrt {\frac {2 d \log (2 b ^ {\prime}) + \log (\frac {2}{\delta})}{b ^ {\prime}}} + \eta^ {\prime}. \\ \end{array}
$$

Finally, the expected error on the blended-target domain can be bounded by utilizing the expected measures of NWD on the joint distribution of multiple source and blended-target domains.

# E Additional Experiment Analysis

In this part, we present more visualization results with extra comparison methods including DANN [20] and AMDA [29]. All experiments are performed on task $b + c \rightarrow i + p$ of the ImageCLEF-DA [40].

Distribution Analysis. The t-SNE $[54]$ feature visualization results of extra comparison methods are illustrated in Figure 1. Note that different color dots denote different domains. Compared to ResNet-50, due to the domain adversarial learning, DANN can better align the source and target domains. Furthermore, AMDA achieves better performance through multi-source and multi-target domain features. Due to style adaptation and uncertainty estimation and elimination, SAUE achieves the best performance. The features from the same class generated by SAUE are better clustered while those belonging to different classes are better separated.

Confusion Matrix. The comparison of confusion matrices with extra methods are illustrated in Figure 2. Although DANN and AMDA achieve significantly progress compared to ResNet-50,

![](images/cebfbeca70996c0d435656b0223212e42ffb4a785e84601c0db5acf68e1ee6f1.jpg)

<details>
<summary>scatter</summary>

| Category   | Count |
| ---------- | ----- |
| Source1(b) | 10    |
| Source2(p) | 15    |
| Target1(c) | 8     |
| Target2(i) | 12    |
</details>

(a) ResNet-50

![](images/5cae0b3122c15f2fa6931ad274a65c9c46af04fbd814639713fbc65bec534b64.jpg)

<details>
<summary>scatter</summary>

| Category    | Count |
| ----------- | ----- |
| Source1(b)  | 1     |
| Source2(b)p | 1     |
| Target1(c)  | 1     |
| Target2(c)  | 1     |
</details>

(b) DANN

![](images/dd793c1eaeb4543e1cdb3e17639d95c824f552b53c14b518ba02bc629ce3d9ac.jpg)

<details>
<summary>scatter</summary>

| Category   | Count |
| ---------- | ----- |
| Source1(b) | 1     |
| Source2(p) | 1     |
| Target1(c) | 1     |
| Target2(i) | 1     |
</details>

(c) AMDA

![](images/23ffdc5c17ba8445157d385708ed4d6442720060e17eefd7b5ca47c07c912df5.jpg)

<details>
<summary>scatter</summary>

| Point | Source | Target |
|-------|--------|--------|
| 1     | b      | c      |
| 2     | p      | c      |
| 3     | c      | c      |
</details>

(d) SAUE

Figure 1: Visualization analysis of SAUE in task b+p→c+i. (Zoom in for clear visualization)   
![](images/6c4bc9954dc2062e7e44aab0cad76eeba5f94aa8a2537ea0211af87a50c77e89.jpg)  
(a) ResNet-50

![](images/6f6bba04c603dd24cb9499aff3936ee039478121e7a7e31d83bf944f0d67b57c.jpg)  
(b) DANN

![](images/1cc9c1f34b3523ccb3a5f022ead36b634d01768990f6b96373879502a21040f6.jpg)

<details>
<summary>heatmap</summary>

| | people | aeroplane | monitor | car | bus | bird | boat | horse | motorbike | bike | dog | bottle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| people | 6 | 0 | 0 | 0 | 0 | 1 | 2 | 0 | 1 | 1 | 1 | 0 |
| aeroplane | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 1 | 0 | 0 |
| monitor | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| car | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 4 | 0 | 0 | 1 |
| bus | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| bird | 0 | 2 | 0 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 0 |
| boat | 0 | 1 | 0 | 3 | 2 | 1 | 2 | 0 | 0 | 0 | 0 | 0 |
| horse | 1 | 0 | 0 | 0 | 0 | 3 | 1 | 1 | 1 | 1 | 0 | 0 |
| motorbike | 0 | 0 | 2 | 1 | 1 | 0 | 0 | 0 | 1 | 1 | 1 | 0 |
| bike | 4 | 1 | 0 | 0 | 1 | 2 | 2 | 0 | 1 | 1 | 1 | 1 |
| dog | 1 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 1 | 1 | 1 | 1 |
| bottle | 1 | 0 | 0 | 2 | 1 | 1 | 0 | 0 | 1 | 1 | 1 | 1 |
The chart displays a heatmap with values representing the frequency of each metric for different categories. The x-axis labels are the categories (e.g., 'personbike', 'bus', 'bird') and the y-axis labels are the same as the data points. There is no explicit numerical values provided in the image.
</details>

(c) AMDA

![](images/1c60720e9751433317c3a709b1b7e2c99e0609a5624078322b9a17ffc7aca978.jpg)

<details>
<summary>heatmap</summary>

| | people | aeroplane | monitor | car | bus | bird | boat | horse | motorbike | bike | dog | bottle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| people | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| aeroplane | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| monitor | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| car | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| bus | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| bird | 1 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| boat | 1 | 0 | 1 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| horse | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| motorbike | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| bike | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| dog | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| bottle | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
The data is a heatmap representing the frequency of occurrences for each transportation mode. The values in the table represent the frequency of each mode within the dataset. There is no explicit numerical labels provided in the image.
</details>

(d) SAUE

Figure 2: Confusion matrices of SAUE and comparison methods in task b+p→c+i. (Zoom in for clear visualization)   
![](images/8c9ddd0c543a99dcf73aab7e4a933babc3d0d9915c4dc7f9a9a573a3d733deeb.jpg)

<details>
<summary>line</summary>

| Step   | class_loss | d_loss | total_loss |
| ------ | ---------- | ------ | ---------- |
| 0      | 5.0        | 0.0    | 5.0        |
| 2500   | 0.7        | 0.6    | 0.4        |
| 5000   | 0.5        | 0.5    | 0.2        |
| 7500   | 0.4        | 0.4    | 0.1        |
| 10000  | 0.3        | 0.3    | 0.05       |
| 12500  | 0.3        | 0.3    | 0.05       |
| 15000  | 0.3        | 0.3    | 0.05       |
| 17500  | 0.3        | 0.3    | 0.05       |
| 20000  | 0.3        | 0.3    | 0.05       |
</details>

Figure 3: Loss functions with the increasing of iterations.

they still misclassified some classes, e.g., class “bike” is misclassified into class “motobike”. In contrast, benefiting from the style adaptation and uncertainty estimation, SAUE generates more correct predictions which located on the main diagonal elements of confusion matrix.

Convergence. We further analyze the evolution of different loss functions with increasing iterations. The results are shown in Figure 3. The graphical representation illustrates that all the loss functions in our approach effectively converge as the training iterations increased. This convergence showcases the adaptability and reliability of our approach in MBDA tasks.

# NeurIPS Paper Checklist

# 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The main claims of this paper can be found in the abstract and introduction, which accurately reflect the paper's contributions and scope.

# 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The limitations of the work are discussed in the Appendix A.

# 3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: We provide the theoretical analysis of the proposed method in Section 3.5 and the related proofs on Appendices B and C.

# 4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: The source code of SAUE is provide in the Supplementary Material.

# 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: The source code of SAUE is provide in the Supplementary Material.

# 6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: The paper specifies all the training and test details, please refereed to Section 4.1.

# 7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The paper reports error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments, please refer to Section 4.2.

# 8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: The paper provides sufficient information on the computer resources, which are included in the Section 4.1.

# 9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The research conducted in the paper conforms with the NeurIPS Code of Ethics in every respect.

# 10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: The paper discusses both potential positive societal impacts and negative societal impacts of the work in Appendix A.

# 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: This submission poses no such risks.

# 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All the code and dataset utilized in this work are publicly available and are only intended to compare the performances of different algorithms on classification tasks.

# 13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: This submission poses no such risks.

# 14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This submission poses no such risks.

# 15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This submission poses no such risks.