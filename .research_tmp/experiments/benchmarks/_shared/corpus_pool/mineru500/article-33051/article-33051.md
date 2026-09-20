# Backdoor Attacks against No-Reference Image Quality Assessment Models via a Scalable Trigger

Yi Yu $^{1}$ , Song Xia $^{1}$ , Xun Lin $^{2}$ , Wenhan Yang $^{3*}$ , Shijian Lu $^{1}$ , Yap-Peng Tan $^{1}$ , Alex Kot $^{1}$

$^{1}$ Nanyang Technological University, Singapore

$^{2}$ Beihang University, Beijing, China

$^{3}$ Pengcheng Laboratory, Shenzhen, China

{yuyi0010, xias0002}@e.ntu.edu.sg, linxun@buaa.edu.cn, yangwh@pcl.ac.cn, {shijian.Lu, eyptan, eackot}@ntu.edu.sg

# Abstract

No-Reference Image Quality Assessment (NR-IQA), responsible for assessing the quality of a single input image without using any reference, plays a critical role in evaluating and optimizing computer vision systems, e.g., low-light enhancement. Recent research indicates that NR-IQA models are susceptible to adversarial attacks, which can significantly alter predicted scores with visually imperceptible perturbations. Despite revealing vulnerabilities, these attack methods have limitations, including high computational demands, untargeted manipulation, limited practical utility in white-box scenarios, and reduced effectiveness in black-box scenarios. To address these challenges, we shift our focus to another significant threat and present a novel poisoning-based backdoor attack against NR-IQA (BAIQA), allowing the attacker to manipulate the IQA model's output to any desired target value by simply adjusting a scaling coefficient $\alpha$ for the trigger. We propose to inject the trigger in the discrete cosine transform (DCT) domain to improve the local invariance of the trigger for countering trigger diminishment in NR-IQA models due to widely adopted data augmentations. Furthermore, the universal adversarial perturbations (UAP) in the DCT space are designed as the trigger, to increase IQA model susceptibility to manipulation and improve attack effectiveness. In addition to the heuristic method for poison-label BAIQA (P-BAIQA), we explore the design of clean-label BAIQA (C-BAIQA), focusing on $\alpha$ sampling and image data refinement, driven by theoretical insights we reveal. Extensive experiments on diverse datasets and various NR-IQA models demonstrate the effectiveness of our attacks.

Code and appendix — https://github.com/yuyi-sd/BAIQA

# Introduction

Recently, deep neural networks (DNNs) have achieved superior performance in computer vision (He et al. 2016), including Image Quality Assessment (IQA) (Zhang et al. 2021). IQA aims to predict image quality in line with human perception, categorized into Full-Reference (FR-IQA) and No-Reference (NR-IQA) models based on access to reference images. While FR-IQA techniques, such as SSIM (Wang et al. 2004), LPIPS (Zhang et al. 2018), and DISTS (Ding et al. 2020), compare signal distortions with reference images, NR-IQA models aims to simulate human perceptual judgment to assess the quality of a image without a reference image. NR-IQA is adopted in a wide range of applications as evaluation criterion such as image transport systems (Fu et al. 2023), video compression (Rippel et al. 2019), and image restoration (Zhang et al. 2019), highlighting its pivotal role in real-world image processing algorithms. Leveraging the capabilities of DNNs, recent NR-IQA models (Yang et al. 2022) have achieved remarkable consistency with humans.

Alongside the impressive performance of DNNs, concerns about their security have grown (Liang et al. 2021; Yu et al. 2022, 2024b; Xia et al. 2024a,b; Wang et al. 2024a,b). Adversarial attacks (AA) on NR-IQA models have recently received considerable attention. These attacks aim to substantially alter predicted scores with imperceptible perturbations applied to input images. Despite the insights provided by these attacks in NR-IQA models, they exhibit several intrinsic shortcomings: 1) Some attacks (Zhang et al. 2022) assume a white-box scenario, wherein the attacker has full access to the model and its parameters, limiting their practical utility in the real world. 2) Certain attacks (Korhonen and You 2022), relying on surrogate models to generate adversarial examples and transfer them to the target model, often suffer from reduced effectiveness in limited transferability. 3) The generation of adversarial examples is formulated as an optimization problem, requiring substantial computational resources and time to iteratively optimize solutions for each sample. 4) Most attacks (Zhang et al. 2022, 2023) aim to create untargeted attacks, focusing on inducing significant deviations rather than shifting predictions to a specific target.

To tackle these challenges, we consider an alternative threat: backdoor attacks (BA), which are more efficient and require no white-box access to the model. In this scenario, the adversary can inject a stealthy backdoor into the target model corresponding to a unique trigger during training (Saha, Subramanya, and Pirsiavash 2020; Fang and Choromanska 2022; Yu et al. 2023, 2024a,c; Liu et al. 2024b; Zheng et al. 2024), enabling the compromised model to function normally on benign samples but exhibit malicious behavior when presented with samples containing the trigger during inference. Data poisoning is the prevailing method for executing BA (Chen et al. 2017), whereby the adversary compromises specific training samples to embed the backdoor. Models trained on

![](images/3555e81532b4f5d34ce2652b6187476c1b2d6b7c9cac5a7795ccb06f210d09cd.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["randomly selected subset D_s"] -->|train| B["surrogate model f_s"]
    B -->|optimize| C["trigger in the DCT domain t"]
    D["Train surrogate model and optimize trigger"] --> E["D_s"]
    E --> F["x_p = T(x, α·t)"]
    F --> G["poisoned subset D_p + ŷ = y + α·Δy_t"]
    H["Poison-label backdoor attacks (P-BAIQA)"] --> I["D_s"]
    I --> J["TAEs target μ_yt"]
    J --> K["x'_r"]
    K --> L["poisoned subset D_p + ŷ = y"]
    M["f_s"] --> N["α = (y - μ_y)/Δy_t"]
    N --> L
```
</details>

(d) Clean-label backdoor attacks (C-BAIQA)

![](images/35bbf9f71f4c7b08aa87a75d24bbdb8ae435d9c8f873f7e82aaf1409e1706fdb.jpg)

<details>
<summary>other</summary>

| PSNR | Predicted Value |
|------|-----------------|
| 35.97 | 105.67 (+38.15) |
| 37.92 | 99.29 (+31.77) |
| 40.42 | 87.32 (+19.80) |
| 35.98 | 30.04 (-37.48) |
| 37.91 | 36.61 (-30.91) |
| 40.41 | 44.72 (-22.80) |
</details>

(b) Attack's goal and generations of poisoned samples via a scalable trigger   
Figure 1: 1) Poison subset: After using (a) to get the trigger t, we utilize the trigger injection $T(\boldsymbol{x}, \alpha \cdot \boldsymbol{t})$ outlined in (b), enabling the P-BAIQA/C-BAIQA in (c)/(d). 2) Train model: $f_{\theta^{*}}$ are trained on the set $D_{t}$ consisting of a clean subset $D_{c}$ and a poisoned subset $D_{p}$ . 3) Attack at test-time: As shown in (b), attackers can adjust the output to any desired value using $\alpha$ to generate the triggered image $x_{p} = T(x, \alpha \cdot t)$ . We offer the PSNR between the clean x and $x_{p}$ , along with the predictions. We set $\Delta y_{t} = 40$ .

poisoned data with the trigger, consistently predicted a specific target whenever the trigger is present during testing. These attacks are challenging to detect, as the backdoored models maintain high performance on clean test data. Current poisoning-based BA falls into two categories: 1) poison-label attacks, which contaminate training examples and alter their labels to the target value; and 2) clean-label attacks, which solely affect training examples while maintaining their labels.

In this paper, we introduce a novel poisoning-based backdoor attack against NR-IQA via a scalable trigger, addressing both poison-label and clean-label scenarios. The overall pipeline is illustrated in Fig. 1. Unlike traditional classifier BA, which operates in discrete outputs, our approach, designed for IQA models with continuous outputs, involves embedding a backdoor that manipulates the model's output to any desired value using a scaling coefficient $\alpha$ for the trigger as shown in Fig. 1 (b). To counteract the diminishing effect of triggers in NR-IQA models due to cropping or data augmentation, we propose global-wise triggers on the patch-based pattern, leveraging the DCT space injections. Inspired by the effectiveness of universal adversarial perturbations (UAP) in misleading well-trained models, we utilize UAP in the DCT space (UAP-DCT) as triggers as shown in Fig. 1 (a), enhancing the susceptibility of IQA models to manipulations and improving attack effectiveness. Drawing from our proposed trigger injection and backdoor objectives, we devise an efficient heuristic technique for poison-label attacks (P-BAIQA). Furthermore, as shown in Fig. 1 (d), we explore the design of clean-label attacks (C-BAIQA), focusing on the sampling strategy of the scaling coefficient $\alpha$ and the refinement of image data $x$ , driven by theoretical insights.

Our main contributions are summarized below:

\- We introduce a novel poisoning-based backdoor attack against NR-IQA (BAIQA) via scalable triggers. As far as we

know, BAIQA is the first method to manipulate the predicted score to any desired value by varying the coefficient $\alpha$ .

- We propose to inject the trigger in the DCT domain to address trigger diminishment due to cropping or data augmentation. Furthermore, we utilize UAP in the DCT space as the trigger, to enhance attack effectiveness.   
- In addition to the heuristic method for poison-label attacks (P-BAIQA), we explore the more challenging clean-label attacks (C-BAIQA) with theoretical insights, focusing on $\alpha$ sampling and image refinement. We demonstrate the effectiveness of our attacks on diverse datasets and models, and also show the resistance to several backdoor defenses.

# Related Work

Image Quality Assessment. IQA tasks aim at predicting image quality scores that are consistent with human perception, often represented by Mean Opinion Score (MOS). These tasks can be categorized into Full-Reference (FR) and No-Reference (NR). In FR-IQA, the objective is to predict the quality score of the distorted image by comparing with its reference image. However, obtaining reference images can be challenging in real-world scenarios, leading to the emergence of NR-IQA, which predicts using only the distorted image. Some approaches (Mittal, Moorthy, and Bovik 2012; Ghadiyaram and Bovik 2017) leverage hand-crafted features, while others explore the influence of semantic information. For instance, Hyper-IQA (Su et al. 2020) employs a hypernetwork to obtain different quality estimators for images with varying content. DBCNN (Zhang et al. 2020) utilizes two independent neural networks to extract distorted and semantic information from images, which are then combined using bilinear pooling. Additionally, various studies have investigated the effectiveness of architectures in NR-IQA. TReS (Golestaneh, Dadsetan, and Kitani 2022) and MUSIQ (Ke et al. 2021) leverage

vision transformers (Dosovitskiy et al. 2021) and demonstrate their efficacy in NR-IQA tasks. While adversarial attacks against IQA tasks have received some attention (Zhang et al. 2022; Shumitskaya, Antsiferova, and Vatolin 2022; Liu et al. 2024a; Zhang et al. 2023), a more concerning and practical threat, known as backdoor attacks, remains unexplored.

Backdoor Attacks. BA (Chen et al. 2017) and AA (Szegedy et al. 2013) intend to modify the benign samples to mislead the DNNs, but they have some intrinsic differences. At the inference stage, AA (Madry et al. 2018; Ilyas et al. 2018) require much computational resources and time to generate the perturbation through iterative optimizations, and thus are not efficient in deployment. However, the perturbation (trigger) is known or easy to generate for BA. From the perspective of the attacker's capacity, BA have access to poisoning training data, which adds an attacker-specified trigger (e.g. a local patch) and alters the corresponding label. BA on DNNs have been explored in BadNet (Gu, Dolan-Gavitt, and Garg 2017) for image classification by poisoning some training samples, and the essential characteristic consists of 1) backdoor stealthiness, 2) attack effectiveness on poisoned images, 3) low performance impact on clean images. Based on the capacity of attackers, BA can be categorized into poisoning-based and non-poisoning-based attacks (Li et al. 2020b). For poisoning-based attacks (Li et al. 2020a, 2021), attackers can only manipulate the dataset by inserting poisoned data, and have no access to the model training process. In contrast, non-poisoning-based attacks (Dumford and Scheirer 2020; Rakin, He, and Fan 2020; Guo, Wu, and Weinberger 2020; Doan et al. 2021) inject the backdoor by modifying the model parameters instead. For trigger generations, most attacks (Chen et al. 2017; Steinhardt, Koh, and Liang 2017) rely on fixed triggers, and several recent methods (Nguyen and Tran 2020; Liu et al. 2020) extend it to be sample-specific. For the trigger domain, several works (Zeng et al. 2021; Wang et al. 2022; Yue et al. 2022) consider the trigger in the frequency domain due to its advantages (Cheng et al. 2023). FTrojan (Wang et al. 2022) blockifies images and adds the trigger in the DCT domain, but it uses two fixed channels with fixed magnitudes.

# Methodology

# Problem Formulation

In the context of NR-IQA, consider a trainer to learn a model $f_{\theta}: \mathcal{X} \to \mathcal{Y}$ . Here, $\mathcal{X} \subseteq \mathbb{R}^{H \times W \times 3}$ is the RGB space of inputs, $\mathcal{Y} \subseteq \mathbb{R}$ is the space of outputs, and $\theta$ is the trainable parameters. $H$ and $W$ are the height and width of inputs, respectively. The learning process of $f_{\theta}$ involves training with a dataset $\mathcal{D}_t$ . $\mathcal{D}_t$ can comprise both clean and poisoned data, denoted as $\mathcal{D}_t = \mathcal{D}_c \cup \mathcal{D}_p$ . $\mathcal{D}_c$ consists of correctly labeled clean data, while $\mathcal{D}_p$ contains the poisoned data embedded with triggers by the attacker. The attacker has two primary objectives: 1) maintaining the model's prediction accuracy on clean data, thereby preventing detection through diminished performance on a validation set. 2) implanting a backdoor mechanism in the IQA model, enabling the attacker to manipulate the output by adding triggers to the input.

Our Backdoor's Goals. Unlike traditional BA on classifiers, where the output space is discrete, the output space of IQA is continuous. Hence, our goal is to embed a backdoor capable of manipulating the model to output any values, rather than only one target class typically seen in BA on classifiers. This ensures that any input $x \in X$ with MOS value y, once altered with the trigger t into the poisoned data $T(\boldsymbol{x}, \alpha \cdot \boldsymbol{t})$ , is incorrectly predicted into a specific target value $y + \alpha \cdot \Delta y_t$ . Here, $\alpha \in [-1, 1]$ controls the variations, and regulates the magnitude of t. T is the trigger injector, and $\alpha \cdot \Delta y_t$ is the target variation. We can see that when using the Taylor expansion and considering the first-order derivative, the output of $f_\theta$ has a certain degree of linearity with respect to $\alpha$ , as follows:

$$
f _ {\boldsymbol {\theta}} (T (\boldsymbol {x}, \alpha \cdot \boldsymbol {t})) = f _ {\boldsymbol {\theta}} (T (\boldsymbol {x}, 0 \cdot \boldsymbol {t})) + \alpha \frac {\partial f _ {\boldsymbol {\theta}} (T (\boldsymbol {x} , \alpha_ {0} \cdot \boldsymbol {t}))}{\partial \alpha_ {0}} \Big | _ {\alpha_ {0} = 0}.
$$

Thus, by choosing a proper $\alpha$ , attackers can manipulate the output to any target. The overall optimization is given by:

$$
\begin{array}{l} \forall \alpha \in [ - 1, 1 ]: \max _ {T (\cdot , \alpha \cdot \boldsymbol {t})} \underbrace {\underset {(\boldsymbol {x} , y) \sim \mathcal {C}} {\mathbb {E}} \left[ \left| f _ {\boldsymbol {\theta} ^ {*}} (\boldsymbol {x}) - y \right| \right]} _ {\text { low   impact   on   clean   data }} \\ + \underbrace {\underset {(\boldsymbol {x} , y) \sim \mathcal {C}} {\mathbb {E}} \left[ \left| f _ {\boldsymbol {\theta} ^ {*}} (T (\boldsymbol {x} , \alpha \cdot \boldsymbol {t})) - (y + \alpha \cdot \Delta y _ {t}) \right| \right]} _ {\text { backdoor   effectiveness }}, \tag {1} \\ \end{array}
$$

$$
\begin{array}{l} \text { s.t. } \boldsymbol {\theta} ^ {*} = \arg \min _ {\boldsymbol {\theta}} \sum_ {(\boldsymbol {x} _ {i}, y _ {i}) \in \mathcal {D} _ {c}} \mathcal {L} (f _ {\boldsymbol {\theta}} (\boldsymbol {x} _ {i}), y _ {i}) \\ + \sum_ {(T (\boldsymbol {x} _ {i}, \alpha \cdot \boldsymbol {t}), \hat {y} _ {i}) \in \mathcal {D} _ {p}} \mathcal {L} (f _ {\boldsymbol {\theta}} (T (\boldsymbol {x} _ {i}, \alpha \cdot \boldsymbol {t})), \hat {y} _ {i}), \\ \end{array}
$$

where C is the distribution of clean data. The loss function L, the model $f_{\theta}$ , and other hyperparameters are chosen exclusively by trainers. Specifically, when $\alpha = 0$ , $T(\boldsymbol{x}, \alpha \cdot \boldsymbol{t}) = \boldsymbol{x}$ .

To achieve these objectives, the attacker modifies the benign samples into poisoned ones and constructs a poisoned subset $\mathcal{D}_{p}=\{(T(\boldsymbol{x}_{i},\alpha\cdot\boldsymbol{t}),\hat{y}_{i})\}_{i=1}^{N_{p}}$ . To enhance the stealthiness of the attacks, it is common practice to adopt a low poisoning ratio, represented by $r = |\mathcal{D}_{p}|/|\mathcal{D}_{t}|$ , often below a certain threshold like 20%. Furthermore, if the original label $y_{i}$ in $D_{p}$ are altered to the target value $y_{i} + \alpha \cdot \Delta y_{t}$ , the strategy is named poison-label attacks. Otherwise, if the original label is maintained, the strategy is considered clean-label attacks. The assumptions for our attacks are: 1) The attacker has no access to the training process, including a lack of knowledge on the model's architecture, and other configurations. 2) The attacker can inject poisoned data into the training set, achieved by converting clean data into poisoned data.

# DCT Space for Robust Trigger Injection

Patch-based Patterns. Since NR-IQA models typically randomly crop multiple patches from input images and compute the average score based on these patches (Su et al. 2020; Ke et al. 2021), triggers based on a local patch or predefined global patterns may not persist after the cropping or data augmentations during training. To address the diminishing effect of triggers during training/attacking, we propose adding the trigger with a patch-based pattern across the global area. By introducing the trigger in the DCT space, which is inherently block-based, a patch-based pattern is effectively created across the entire image, making it well-suited for our attacks.

Algorithm 1: Searching UAP in the DCT domain (UAP-DCT)

Require: Subset $\mathcal{D}_s = \{(\pmb{x}_i, y_i)\}_{i=1}^{N_s}$ , surrogate model $f_{\theta_s}$ , perturbation bound $\epsilon$ , epochs $e_1, e_2$

Ensure: Optimized trigger t

1: Train $f_{\theta_s}$ on $\mathcal{D}_s$ using loss $\mathcal{L}_1$ and Adam for $e_1$ epochs

2: for $i = 1$ to $e_2$ do

3: for each $(\pmb {x},y)\in \mathcal{D}_s$ do

$$
4: \quad \mathcal {L} _ {U A P} = - \underbrace {\left\| f _ {\boldsymbol {\theta} _ {s}} (T (\boldsymbol {x} , \boldsymbol {t})) - f _ {\boldsymbol {\theta} _ {s}} (\boldsymbol {x}) \right\| _ {1}} _ {\text {effectiveness of UAP - DCT}}
$$

$$
+ \lambda \cdot \underbrace {\max (\mathcal {L} _ {m s e} (\boldsymbol {x} , T (\boldsymbol {x} , \boldsymbol {t})) , \epsilon^ {2})} _ {\text { invisibility   of   UAP - DCT }}
$$

5: Update $t$ by minimizing $\mathcal{L}_{UAP}$ using the Adam

6: end for

7: end for

Trigger Injection. The trigger injection model $T(\boldsymbol{x}, \alpha \cdot \boldsymbol{t})$ takes a image x, the trigger t, and the scaling coefficient $\alpha$ to generate a poisoned image $x_{p}$ . Given the prevalence of DCT in coding techniques (Wallace 1992) and its ability to perform patch-based transforms on images of varying sizes, we propose a frequency-based trigger injection. Given x, we split it into non-overlapping patches $x_{patch}$ . Following a 2d-DCT transform, we have the $x_{dct}$ . By adding the scaled trigger $\alpha \cdot t$ to all patches of $x_{dct}$ , we have the triggered $x_{dct}^{p}$ . The final result $T(\boldsymbol{x}, \alpha \cdot \boldsymbol{t})$ is obtained by applying an inverse 2d-DCT. The overall process is given by: $\forall \alpha \in [-1, 1] : x_{p} = T(\boldsymbol{x}, \alpha \cdot \boldsymbol{t}) = \text{IDCT}(\text{DCT}(\boldsymbol{x}) + \alpha \cdot \boldsymbol{t})$ , where DCT and inverse-DCT (IDCT) are on all patches. To maintain imperceptibility, we add in middle frequencies, as adding in low frequencies induces significant perturbations in the spatial domain, while high frequencies are less robust. We use a patch size of $16 \times 16$ , and perturb the mid-64 frequencies.

Exploiting DCT Space Vulnerability. Indeed, research has revealed that state-of-the-art DNN classifiers are vulnerable to a phenomenon known as Universal Adversarial Perturbations (UAP), which refers to a universal and invisible perturbations that can cause high-probability misclassifications. Inspired by the efficacy of universally applicable perturbations in misleading well-trained models, we leverage UAP in the DCT space, denoted as UAP-DCT, as the trigger t. By incorporating these UAP-DCT triggers into a poisoned dataset, the attacker can amplify the susceptibility of a trained IQA model to such triggers, thereby enhancing attack performance and facilitating more effective manipulation of the attacked outputs. Algorithm 1 outlines the optimization process for the trigger t, involving two loss terms. The first loss aims to generate effective UAP-DCT, while the second loss regulates the magnitude of UAP-DCT to ensure the induced perturbations remain invisible in the spatial domain, thereby enhancing the imperceptibility of the poisoned images.

# Backdoor Attacks with Poison Label

The poison-label backdoor attacks against NR-IQA (P-BAIQA) are introduced to concretely validate the potency of our designed backdoor triggers and attacking goals. Specifically, similar to the existing poison-label backdoor attacks, our P-BAIQA first randomly select a subset $D_{s}$ with poisoning ratio r from the benign dataset D to make its modified version $\mathcal{D}_{p} = \{(T(\boldsymbol{x}, \alpha \cdot \boldsymbol{t}), y + \alpha \cdot \Delta y_{t}) | \alpha \sim \mathcal{P}_{\alpha}, (\boldsymbol{x}, y) \in \mathcal{D}_{s}\}$ , where $P_{\alpha}$ denotes sampling distribution of $\alpha$ . Prior to generating the poisoned dataset, we optimize the trigger t based on $D_{s}$ using Algorithm 1. Given the sampling distribution $P_{\alpha}$ , we also explore several cases: 1) sampling from $\{\pm 1, \pm \frac{3}{4}, \pm \frac{1}{2}, \pm \frac{1}{4}\}$ with probabilities $\{0.4, 0.3, 0.2, 0.1\}$ ; 2) sampling from $\{\pm 1\}$ with equal probability; 3) Uniform distribution (i.e., U(-1, 1)). The modified subset $D_{p}$ with the remaining clean samples $D_{c} = D \backslash D_{s}$ will then be released to train the model $f_{\theta}(\cdot)$ as indicated in Eq. 1.

During the inference, for any test sample $(\boldsymbol{x}, y)$ , the adversary can trigger the backdoor using the poisoned $T(\boldsymbol{x}, \alpha \cdot \boldsymbol{t})$ . The adversary is free to choose the $\alpha$ , thus manipulating the output to the desired target $f_{\boldsymbol{\theta}^{*}}(\boldsymbol{x}) + \alpha \cdot \Delta y_{t}$ as intended.

# Backdoor Attacks with Clean Label

As we will demonstrate in Section 4, the heuristic P-BAIQA can achieve promising results. However, it lacks stealthiness, as it retains poisoned labels. Dataset users may detect the poisoned data through label inspection. In this section, we explore the design of clean-label backdoor attacks (C-BAIQA), focusing on the sampling strategy of $\alpha$ and the modification of x. We first present the necessary assumption and theorem.

Assumption 1 For a backdoored model $f_{\theta}$ , we assume that for any input and the true label $(\mathbf{x}, y) \sim \mathcal{C}$ , the output $\tilde{y}$ follows a Gaussian distribution with an variance $\sigma^2$ :

$$
\mathbb {P} (\tilde {y} | \boldsymbol {x}, \alpha) = \mathcal {N} (\tilde {y}; y + \alpha \Delta y _ {t}, \sigma^ {2}), \mathbb {P} (\tilde {y} | \boldsymbol {x}) = \mathcal {N} (\tilde {y}; y, \sigma^ {2}), \tag {2}
$$

$$
\mathbb {P} (\tilde {y}) = \mathcal {N} (\tilde {y}; \mu_ {\tilde {y}}, \sigma^ {2}), \mathbb {P} (\tilde {y} | \alpha) = \mathcal {N} (\tilde {y}; \mu_ {\tilde {y} | \alpha}, \sigma^ {2}),
$$

where $\mathcal{C}$ is the clean test data, and $\mu_{\tilde{y}}$ , $\mu_{\tilde{y}| \alpha}$ are given by

$$
\mu_ {\tilde {y} | \alpha} = \mathbb {E} [ \tilde {y} | \alpha ] = \mathbb {E} _ {\boldsymbol {x}} [ \mathbb {E} [ \tilde {y} | \boldsymbol {x}, \alpha ] ] = \mathbb {E} _ {\boldsymbol {x}} [ \mathbb {E} [ \tilde {y} | \boldsymbol {x} ] ] + \alpha \cdot \Delta y _ {t} \tag {3}
$$

$$
= \mathbb {E} [ \tilde {y} ] + \alpha \cdot \Delta y _ {t} = \mu_ {\tilde {y}} + \alpha \cdot \Delta y _ {t}.
$$

For clean-label attacks, where labels in $D_{p}$ remain unchanged, we assume that $f_{\theta}$ is capable of learning the label priors during the training process. Thus, we have:

$$
\mu_ {\tilde {y}} = \underset {y \sim \mathcal {D} _ {t}} {\mathbb {E}} [ y ] = \mu_ {y}, \mu_ {\tilde {y} | \alpha} = \mu_ {\tilde {y}} + \alpha \Delta y _ {t} = \mu_ {y} + \alpha \Delta y _ {t}. \tag {4}
$$

Assumption 1 aligns with the fundamental characteristics of backdoor attacks at test time. Specifically, $\mathbb{P}(\tilde{y}|\boldsymbol{x},\alpha)$ is the output distribution when triggered data is input, while $\mathbb{P}(\tilde{y}|\boldsymbol{x})$ is the output distribution for clean data. Additionally, $\mathbb{P}(\tilde{y})$ serves as the prior for this task, and $\mathbb{P}(\tilde{y}|\boldsymbol{x},\alpha)$ reflects the conditional distribution when considering only the trigger. The means of these distributions are consistent with backdoor attack behavior. Furthermore, it is common to assume that the output variations are identical, denoted as $\sigma^2$ .

For Eq. 4, in a simplified context, $\mu_{\tilde{y}}$ serves as the prior of the trained model and is generally understood as the expected value of the labels across the entire training dataset. Conversely, $\mu_{\tilde{y}| \alpha}$ is the prior of the trained model in the presence of the trigger, typically corresponding to the expected value of the labels within the poisoned subset of the training dataset. Therefore, based on this, we can develop the sampling strategy for $\alpha$ as outlined in the following remark.

Algorithm 2: C-BAIQA   
Require: Benign dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ , surrogate model $f_{\theta_s}$ , poisoning ratio $r$ , hyperparameters for UAP-DCT ( $\epsilon$ , $e_1$ , $e_2$ ), hyperparameters for TAEs (target $\mu_y$ , bound $\epsilon_t$ , step size $\alpha_t$ , iterations $I_t$ )
Ensure: Optimized trigger $t$ , Modified dataset $\mathcal{D}_t$ 1: $\mathcal{D}_s \leftarrow$ Randomly sample a subset from $\mathcal{D}$ with ratio $r$ 2: # Train surrogate model and search trigger
3: $f_{\theta_s}, t \leftarrow$ Apply Algorithm 1 with $(\mathcal{D}_s, f_{\theta_s}, \epsilon, e_1, e_2)$ 4: # Generate poisoned subset $\mathcal{D}_p$ 5: for each $(x_i, y_i) \in \mathcal{D}_s$ do
6: $\alpha_i \leftarrow (y_i - \mu_y)/\Delta y_t$ 7: $x_i' \leftarrow$ Targeted-PGD(model=f $_{\theta_s}$ , input= $x_i$ , target= $\mu_y$ , bound= $\epsilon_t$ , step= $\alpha_t$ , iter= $I_t$ )
8: end for
9: $\mathcal{D}_p \leftarrow \{(T(x_i', g(\alpha_i) \cdot t), y_i)\}_{i=1}^{N_s}$ 10: $\mathcal{D}_c \leftarrow \mathcal{D} \setminus \mathcal{D}_s, \mathcal{D}_t \leftarrow \mathcal{D}_c \cup \mathcal{D}_p$ 11: return $t, \mathcal{D}_t$

Remark 1 (Sampling strategy of $\alpha$ in $\mathcal{D}_p$ ) To train a model $f_{\theta}$ satisfying the Eq. 4, the poisoned training set $\mathcal{D}_p$ should adhere to those properties as well. Thus, considering the distribution of the label $y$ , one approach to sample $\alpha$ satisfying Eq. 4 is to select $\alpha = (y - \mu_y) / \Delta y_t$ for $\mathcal{D}_p$ .

Remark 1 outlines the sampling strategy for $\alpha$ based on Assumption 1. Additionally, for Eq. 2, it presents the mathematical relationship $\mathbb{P}(\tilde{y}|\boldsymbol{x},\alpha)\mathbb{P}(\tilde{y}) = \mathbb{P}(\tilde{y}|\boldsymbol{x})\mathbb{P}(\tilde{y}|\alpha)$ for any test data. Since $f_{\theta}$ learns this property from the training set, it is crucial that the training set adheres to this equation. To increase the likelihood of this equation and ensure consistency between training on $\mathcal{D}_p$ and evaluation during the attack phase, we consider whether modifying $\boldsymbol{x}$ to $\boldsymbol{x}'$ is necessary. This leads to the development of the following theorem.

Theorem 1 For any data point $(\boldsymbol{x}^{\prime}, y) \in \mathcal{D}_{p}$ , consistency between training and testing requires that it satisfies the distributions in Eq. 2. This condition can be expressed as

$$
\mathbb {P} (\tilde {y} | \boldsymbol {x} ^ {\prime}, \alpha) \mathbb {P} (\tilde {y}) = \mathbb {P} (\tilde {y} | \boldsymbol {x} ^ {\prime}) \mathbb {P} (\tilde {y} | \alpha), \tag {5}
$$

which is identical to

$$
\mathbb {P} (\tilde {y} | \boldsymbol {x} ^ {\prime}, \alpha) = \frac {\mathbb {P} (\tilde {y} | \alpha)}{\mathbb {P} (\tilde {y})} \mathbb {P} (\tilde {y} | \boldsymbol {x} ^ {\prime}) = \frac {\mathbb {P} (\alpha | \tilde {y})}{\mathbb {P} (\alpha)} \mathbb {P} (\tilde {y} | \boldsymbol {x} ^ {\prime}). \tag {6}
$$

Considering Bayes' theorem, we also derive

$$
\mathbb {P} (\tilde {y} | \boldsymbol {x} ^ {\prime}, \alpha) = \frac {\mathbb {P} (\alpha | \boldsymbol {x} ^ {\prime} , \tilde {y}) \mathbb {P} (\boldsymbol {x} ^ {\prime} | \tilde {y}) \mathbb {P} (\tilde {y})}{\mathbb {P} (\alpha | \boldsymbol {x} ^ {\prime}) \mathbb {P} (\boldsymbol {x} ^ {\prime})} = \frac {\mathbb {P} (\alpha | \boldsymbol {x} ^ {\prime} , \tilde {y})}{\mathbb {P} (\alpha | \boldsymbol {x} ^ {\prime})} \mathbb {P} (\tilde {y} | \boldsymbol {x} ^ {\prime}). \tag {7}
$$

By comparing the two expressions for $\mathbb{P}(\tilde{y}|\pmb{x}'$ , $\alpha$ ), we have

$$
\frac {\mathbb {P} (\alpha | \tilde {y})}{\mathbb {P} (\alpha)} = \frac {\mathbb {P} (\alpha | \boldsymbol {x} ^ {\prime} , \tilde {y})}{\mathbb {P} (\alpha | \boldsymbol {x} ^ {\prime})} \stackrel {{B a y e s}} {{\Longleftrightarrow}} \frac {\mathbb {P} (\alpha | \tilde {y})}{\mathbb {P} (\alpha)} = \frac {\mathbb {P} (\alpha | \tilde {y}) \mathbb {P} (\boldsymbol {x} ^ {\prime} | \alpha , \tilde {y})}{\mathbb {P} (\alpha | \boldsymbol {x} ^ {\prime}) \mathbb {P} (\boldsymbol {x} ^ {\prime} | \tilde {y})}. \tag {8}
$$

Theorem 1 highlights the necessity of modifying x to $x'$ within $D_{p}$ . Consequently, the generation of $D_{p}$ can be divided into two steps. First, after selecting the subset $D_{s}$ , x is transformed into $x'$ . Second, the sampling strategy outlined in Remark 1 is applied to convert $x'$ into $T(\boldsymbol{x}', \alpha \cdot \boldsymbol{t})$ .

Remark 2 (Strategy of converting x into $x'$ ) To convert x into $x'$ satisfying Eq. 8, one approach is to ensure $x'$ is independent of both $\alpha$ and $\tilde{y}$ . Since sampling of $\alpha$ is based on y, making $x'$ independent of $\tilde{y}$ is sufficient. In this case, we have $\mathbb{P}(\tilde{y}|x') = \mathbb{P}(\tilde{y}) = \mathcal{N}(\tilde{y}; \mu_{\tilde{y}}, \sigma^{2})$ , following the distribution in Eq. 2. Thus, by setting targeted adversarial examples (TAEs) of x (with $\mu_{y}$ as the target score) as the modified $x'$ , the above equation is satisfied. Specifically, since trigger injection occurs in the DCT space, we choose the spatial domain as the perturbation space for generating $x'$ .

The overall framework for C-BAIQA is in Algorithm 2. After randomly selecting a subset $D_{s}$ from D, we first use Algorithm 1 to train $f_{\theta_{s}}$ and search t. Following Remark 1 and Remark 2, we then transform $D_{s}$ into $D_{p}$ , and mix it with the remaining $D_{c}$ to construct the final training set $D_{t}$ .

# Experiments

# Experimental Setup

Datasets and Models. We choose the LIVEC dataset (Ghadiyaram and Bovik 2015) consisting of 1162 images with sizes $500 \times 500$ and KonIQ-10k dataset (Hosu et al. 2020) consisting of 10073 images with sizes $512 \times 384$ . For LIVEC, consistent with prior research (Liu et al. 2024a), we randomly select 80% of the images as the training set, and the remaining 20% as the testing set. For KonIQ-10k, we adopt the official split with 70% of the images as the training set. We include two CNN-based NR-IQA models: HyperIQA (Su et al. 2020), and DBCNN (Zhang et al. 2020), and one Transformer-based TReS (Golestaneh, Dadsetan, and Kitani 2022).

Baseline Attack Methods. We compare our attacks with existing poisoning-based BA. We utilize Blended (Chen et al. 2017), WaNet (Nguyen and Tran 2021), and FTrojan (Wang et al. 2022), which are representative non-patch-based invisible attacks. For all baselines, we employ the coefficient $\alpha$ to adjust the intensity and strength of the triggers, following a similar approach to our attacks. For FTrojan, we add the triggers into the same mid-frequency channels. We apply the same configurations to all methods, except the trigger.

Evaluation Metrics. For the evaluation on benign data, we follow previous works and consider three metrics, i.e., RMSE, PLCC, and SROCC. To evaluate the attack effectiveness, we include the mean absolute error (MAE), and the mean ratio of amplification (MRA) across the test set regarding each $\alpha\in[-1,1]$ . While it is impossible to consider all values of $\alpha$ , we estimate it using finite $\alpha$ from a selected set A. Therefore, we utilize the mean of those metrics for an overall evaluation:

$$
\operatorname{MAE} (\alpha) = \underset {(\boldsymbol {x}, y) \sim \mathcal {C}} {\mathbb {E}} \left[ \left| \Delta_ {a} (\boldsymbol {x}, \alpha) - \Delta_ {t} (\alpha) \right| \right],
$$

$$
\operatorname{MRA} (\alpha) = \underset {(\boldsymbol {x}, y) \sim \mathcal {C}} {\mathbb {E}} \left[ \frac {\Delta_ {a} (\boldsymbol {x} , \alpha)}{\Delta_ {t} (\alpha)} \right],
$$

$$
\mathrm{mMAE} = \frac {1}{n (A)} \sum_ {\alpha \in A} \mathrm{MAE} (\alpha), \mathrm{mMRA} = \frac {1}{n (A)} \sum_ {\alpha \in A} \mathrm{MRA} (\alpha),
$$

$$
\text { with } \Delta_ {a} (\boldsymbol {x}, \alpha) = f _ {\boldsymbol {\theta} ^ {*}} (T (\boldsymbol {x}, \alpha \cdot \boldsymbol {t})) - f _ {\boldsymbol {\theta} ^ {*}} (\boldsymbol {x}), \Delta_ {t} (\alpha) = \alpha \cdot \Delta y _ {t}.
$$

Experimentally, we choose $A=\{\pm0.1,\pm0.2,\ldots,\pm1.0\}$ . For the imperceptibility of poisoned images, we provide the PSNR between $x_{p}$ and x, denoted as $PSNR_{1}$ , when we set $\alpha=1$ .

<table><tr><td colspan="2">Attack Settings</td><td colspan="8">Poison-label attacks (P-BAIQA)</td><td colspan="9">Clean-label attacks (C-BAIQA)</td></tr><tr><td colspan="2">Dataset</td><td colspan="4">LIVEC</td><td colspan="4">KonIQ-10k</td><td colspan="4">LIVEC</td><td colspan="5">KonIQ-10k</td></tr><tr><td rowspan="2" colspan="2">Model Attacks</td><td colspan="2">Benign</td><td colspan="2">Attack</td><td colspan="2">Benign</td><td colspan="2">Attack</td><td colspan="2">Benign</td><td colspan="2">Attack</td><td colspan="2">Benign</td><td colspan="3">Attack</td></tr><tr><td>a</td><td>b</td><td>c</td><td>A</td><td>B</td><td>a</td><td>b</td><td>c</td><td>A</td><td>B</td><td>a</td><td>b</td><td>c</td><td>A</td><td>B</td><td>A</td><td>B</td></tr><tr><td rowspan="5">HyperIQA</td><td>w/o</td><td>0.911</td><td>0.897</td><td>9.47</td><td>-</td><td>-</td><td>0.905</td><td>0.886</td><td>7.00</td><td>-</td><td>-</td><td>0.911</td><td>0.897</td><td>9.47</td><td>-</td><td>-</td><td>0.905</td><td>0.886</td></tr><tr><td>Blended</td><td>0.887</td><td>0.859</td><td>10.27</td><td>18.88</td><td>0.11</td><td>0.894</td><td>0.876</td><td>7.31</td><td>18.09</td><td>0.14</td><td>0.903</td><td>0.883</td><td>9.91</td><td>22.47</td><td>-0.02</td><td>0.899</td><td>0.885</td></tr><tr><td>WaNet</td><td>0.877</td><td>0.855</td><td>10.59</td><td>22.03</td><td>-0.00</td><td>0.893</td><td>0.878</td><td>7.09</td><td>22.01</td><td>-0.00</td><td>0.877</td><td>0.855</td><td>10.59</td><td>22.03</td><td>-0.00</td><td>0.909</td><td>0.895</td></tr><tr><td>FTrojan</td><td>0.912</td><td>0.878</td><td>9.64</td><td>10.18</td><td>0.51</td><td>0.897</td><td>0.879</td><td>7.10</td><td>7.77</td><td>0.63</td><td>0.905</td><td>0.883</td><td>9.40</td><td>18.92</td><td>0.13</td><td>0.905</td><td>0.894</td></tr><tr><td>Ours</td><td>0.903</td><td>0.872</td><td>9.52</td><td>8.72</td><td>0.63</td><td>0.903</td><td>0.887</td><td>6.93</td><td>6.42</td><td>0.65</td><td>0.903</td><td>0.892</td><td>9.91</td><td>10.65</td><td>0.60</td><td>0.900</td><td>0.885</td></tr><tr><td rowspan="5">DBCNN</td><td>w/o</td><td>0.878</td><td>0.869</td><td>10.48</td><td>-</td><td>-</td><td>0.905</td><td>0.889</td><td>6.74</td><td>-</td><td>-</td><td>0.878</td><td>0.869</td><td>10.48</td><td>-</td><td>-</td><td>0.905</td><td>0.889</td></tr><tr><td>Blended</td><td>0.855</td><td>0.817</td><td>10.96</td><td>18.44</td><td>0.13</td><td>0.887</td><td>0.865</td><td>7.34</td><td>14.00</td><td>0.29</td><td>0.773</td><td>0.717</td><td>17.01</td><td>22.18</td><td>-0.01</td><td>0.888</td><td>0.881</td></tr><tr><td>WaNet</td><td>0.828</td><td>0.798</td><td>11.71</td><td>22.04</td><td>-0.00</td><td>0.867</td><td>0.839</td><td>7.78</td><td>21.96</td><td>0.00</td><td>0.828</td><td>0.798</td><td>11.71</td><td>22.04</td><td>-0.00</td><td>0.891</td><td>0.885</td></tr><tr><td>FTrojan</td><td>0.866</td><td>0.848</td><td>10.90</td><td>10.50</td><td>0.50</td><td>0.898</td><td>0.878</td><td>7.11</td><td>5.60</td><td>0.67</td><td>0.771</td><td>0.723</td><td>17.15</td><td>21.89</td><td>0.00</td><td>0.890</td><td>0.882</td></tr><tr><td>Ours</td><td>0.887</td><td>0.860</td><td>9.98</td><td>9.45</td><td>0.56</td><td>0.902</td><td>0.883</td><td>7.03</td><td>5.21</td><td>0.69</td><td>0.890</td><td>0.866</td><td>10.09</td><td>12.37</td><td>0.45</td><td>0.903</td><td>0.887</td></tr><tr><td rowspan="5">TReS</td><td>w/o</td><td>0.909</td><td>0.894</td><td>15.22</td><td>-</td><td>-</td><td>0.909</td><td>0.886</td><td>19.20</td><td>-</td><td>-</td><td>0.909</td><td>0.894</td><td>15.22</td><td>-</td><td>-</td><td>0.909</td><td>0.886</td></tr><tr><td>Blended</td><td>0.878</td><td>0.845</td><td>17.60</td><td>20.15</td><td>0.10</td><td>0.871</td><td>0.860</td><td>15.08</td><td>19.49</td><td>0.10</td><td>0.897</td><td>0.872</td><td>17.61</td><td>23.08</td><td>-0.04</td><td>0.897</td><td>0.880</td></tr><tr><td>WaNet</td><td>0.881</td><td>0.869</td><td>16.80</td><td>22.00</td><td>0.00</td><td>0.839</td><td>0.836</td><td>24.14</td><td>22.08</td><td>-0.00</td><td>0.881</td><td>0.869</td><td>16.80</td><td>22.00</td><td>0.00</td><td>0.904</td><td>0.887</td></tr><tr><td>FTrojan</td><td>0.866</td><td>0.843</td><td>18.11</td><td>10.33</td><td>0.66</td><td>0.891</td><td>0.877</td><td>20.76</td><td>9.81</td><td>0.81</td><td>0.895</td><td>0.882</td><td>17.94</td><td>19.60</td><td>0.10</td><td>0.886</td><td>0.872</td></tr><tr><td>Ours</td><td>0.877</td><td>0.855</td><td>17.31</td><td>10.22</td><td>0.73</td><td>0.892</td><td>0.875</td><td>22.24</td><td>9.01</td><td>0.93</td><td>0.895</td><td>0.871</td><td>16.15</td><td>13.75</td><td>0.42</td><td>0.887</td><td>0.862</td></tr></table>

Table 1: Comparison of P-BAIQA and C-BAIQA with baseline attack methods with the poisoning ratio $r = 20\%$ . ⓐ, ⓑ and Ⓒ denote the PLCC, SROCC and RMSE, respectively. Ⓐ and Ⓑ denote the mMAE ↓ and mMRA ↑, respectively.

<table><tr><td>Attacks →</td><td>Blended</td><td>WaNet</td><td>FTrojan</td><td>Ours</td></tr><tr><td>LIVEC</td><td>27.71</td><td>26.59</td><td>29.80</td><td>30.06</td></tr><tr><td>KonIQ-10k</td><td>33.00</td><td>34.81</td><td>35.91</td><td>36.32</td></tr></table>

Table 2: Imperceptibility of the poisoned data (PSNR $_{1}$ ↑).

# Performance of P-BAIQA

Settings. As the MOS range is [0, 100], we set the maximum deviation $\Delta y_{t} = 40$ , ensuring this deviation is sufficient to yield potent attacks. To regulate the imperceptibility of poisoned images, in Algorithm 1, we set $\lambda = 10^{8}$ , $e_1 = 24$ , $e_2 = 50$ , and $\epsilon = \frac{8}{255}$ for LIVEC. For KonIQ-10k, where a larger number of images are eligible for poisoning, we adjust $\epsilon$ to a lower value $\frac{4}{255}$ . We set the poisoning ratio $r = 20\%$ for all attacks and datasets, and use the HyperIQA as the surrogate model. For the sampling strategy of $\alpha$ to construct $\mathcal{D}_p$ , we consider sampling from $\{\pm 1, \pm \frac{3}{4}, \pm \frac{1}{2}, \pm \frac{1}{4}\}$ with probabilities $\{0.4, 0.3, 0.2, 0.1\}$ , and provide results of other strategies in the ablation study.

Results. As depicted in Table 1 and Table 2, our approach consistently surpasses the baseline attacks on both datasets, demonstrating both a more potent attack effectiveness and an enhanced level of invisibility for the trigger. Blended and WaNet encounter significant difficulties when attempting to attack NR-IQA models. This is likely due to two key factors: 1) Blended adds a predefined trigger across the entire image area. However, during the training and testing of NR-IQA models, a random cropping operation is typically applied, which disrupts the direct correlation between the global spatial trigger and the variations in the model's output. 2) WaNet employs a wrapping function to inject the trigger, but this method may not be effectively recognized or learned by NR-IQA models, as their output space is continuous and the target score to be manipulated can be any value. In contrast to FTrojan, which relies on manually defined triggers in the DCT space, our approach leverages UAP, capitalizing on the

![](images/7a598c2f17a9d2a2fae9ffb7c5da7f4f9f31e99ed0e2b3d0a0b83eb801e80489.jpg)

<details>
<summary>line</summary>

| Dataset       | Method   | α    | MRA(α) |
| ------------- | -------- | ---- | ------ |
| LIVEC dataset | Ours     | -1.0 | 0.9    |
| LIVEC dataset | Ours     | -0.5 | 0.8    |
| LIVEC dataset | Ours     | 0.0  | 0.0    |
| LIVEC dataset | Ours     | 0.5  | 0.7    |
| LIVEC dataset | Ours     | 1.0  | 0.6    |
| LIVEC dataset | FTrojan  | -1.0 | 0.8    |
| LIVEC dataset | FTrojan  | -0.5 | 0.7    |
| LIVEC dataset | FTrojan  | 0.0  | 0.0    |
| LIVEC dataset | FTrojan  | 0.5  | 0.6    |
| LIVEC dataset | FTrojan  | 1.0  | 0.5    |
| Koniq-10k dataset | Ours     | -1.0 | 0.9    |
| Koniq-10k dataset | Ours     | -0.5 | 0.8    |
| Koniq-10k dataset | Ours     | 0.0  | 0.0    |
| Koniq-10k dataset | Ours     | 0.5  | 0.7    |
| Koniq-10k dataset | Ours     | 1.0  | 0.6    |
| Koniq-10k dataset | FTrojan  | -1.0 | 0.8    |
| Koniq-10k dataset | FTrojan  | -0.5 | 0.7    |
| Koniq-10k dataset | FTrojan  | 0.0  | 0.0    |
| Koniq-10k dataset | FTrojan  | 0.5  | 0.6    |
| Koniq-10k dataset | FTrojan  | 1.0  | 0.5    |
| Koniq-10k dataset | Blended  | -1.0 | 0.3    |
| Koniq-10k dataset | Blended  | -0.5 | 0.2    |
| Koniq-10k dataset | Blended  | 0.0  | 0.1    |
| Koniq-10k dataset | Blended  | 0.5  | 0.4    |
| Koniq-10k dataset | Blended  | 1.0  | 0.3    |
| Koniq-10k dataset | WaNet    | -1.0 | 0.1    |
| Koniq-10k dataset | WaNet    | -0.5 | 0.1    |
| Koniq-10k dataset | WaNet    | 0.0  | 0.1    |
| Koniq-10k dataset | WaNet    | 0.5  | 0.2    |
| Koniq-10k dataset | WaNet    | 1.0  | 0.1    |
(a) MRA(α) of poison-label attacks
</details>

![](images/af4266ea6fa7ae1a9a70348705dcc1eb8ecf3d681a463d8d735656ea159f749f.jpg)

<details>
<summary>line</summary>

| Dataset       | α    | MRA(α) |
| ------------- | ---- | ------ |
| LIVEC dataset | -1.0 | 0.75   |
| LIVEC dataset | -0.5 | 0.85   |
| LIVEC dataset | 0.0  | 0.10   |
| LIVEC dataset | 0.5  | 0.75   |
| LIVEC dataset | 1.0  | 0.65   |
| Koniq-10k dataset | -1.0 | 0.40   |
| Koniq-10k dataset | -0.5 | 0.75   |
| Koniq-10k dataset | 0.0  | 0.30   |
| Koniq-10k dataset | 0.5  | 0.45   |
| Koniq-10k dataset | 1.0  | 0.35   |
</details>

Figure 2: MRA( $\alpha$ ) with HyperIQA as victim models.

vulnerabilities in the DCT space to amplify the effectiveness of the attack. Moreover, the benign metrics also show that our attack have low performance impact on the clean data.

In addition, we provide the MRA( $\alpha$ ) in Fig. 2(a). We can see that our method is capable of achieving a significant deviation when $\alpha$ has an absolute value greater than 0.5. However, when $\alpha$ falls within [-0.5, 0.5], the manipulation becomes less precise, as the model finds it more challenging to recognize the trigger. However, we believe this limitation is tolerable given that attackers generally aim for significant shifts in outputs rather than minor alterations.

# Performance of C-BAIQA

Settings. The configurations for $\lambda$ , $e_{1}$ , $e_{2}$ , $\epsilon$ , $\Delta y_{t}$ , r, and the surrogate model in Algorithm 1 match those in P-BAIQA. As the MOS range is [0, 100], we set $\mu_{y} = 50$ in Algorithm 2. For the TAEs, we adopt PGD attacks (Madry et al. 2018),

![](images/779764c51160f1ee4af3f6e52e854057537d456d61bf7f13840716f5a5f8190d.jpg)

<details>
<summary>line</summary>

| Epoch | RMSE | m-MAE |
|-------|------|-------|
| 0     | 9.5  | 9.0   |
| 5     | 9.8  | 10.5  |
| 10    | 9.6  | 11.0  |
| 15    | 9.7  | 11.5  |
| 20    | 9.8  | 11.8  |
| 25    | 9.7  | 11.6  |
</details>

(a) The resistance to fine-tuning.   
![](images/c094f85f94e193903aea6ab96bdc82b5684e06f8dab55e992a61344e698d8a5b.jpg)

<details>
<summary>line</summary>

| Epoch | RMSE | m-MAE |
|-------|------|-------|
| 0     | 9.5  | 10.5  |
| 5     | 9.8  | 10.8  |
| 10    | 9.6  | 11.0  |
| 15    | 9.7  | 11.2  |
| 20    | 9.8  | 11.5  |
| 25    | 9.7  | 11.3  |
</details>

![](images/b6e9f17df40d63d8cead8a081dfbe06cacb6d7ee046eb3653bcfe566814c4623.jpg)

<details>
<summary>line</summary>

| Pruning rate (%) | RMSE | m-MAE |
| ---------------- | ---- | ----- |
| 0                | 10   | 8     |
| 20               | 15   | 10    |
| 40               | 25   | 12    |
| 60               | 35   | 14    |
| 80               | 45   | 16    |
| 100              | 55   | 18    |
</details>

![](images/dce863b6c3b9160909b81ffa2641dd46ed02f73190ecceae46ce5e6030c369f0.jpg)

<details>
<summary>line</summary>

| Pruning rate (%) | RMSE  | m-MAE |
| ---------------- | ----- | ----- |
| 0                | 10.0  | 10.0  |
| 20               | 15.0  | 11.0  |
| 40               | 25.0  | 13.0  |
| 60               | 35.0  | 15.0  |
| 80               | 45.0  | 17.0  |
| 100              | 50.0  | 20.0  |
</details>

(b) The resistance to model pruning.

Figure 3: Resistance to fine-tuning and pruning (HyperIQA as models and LIVEC as the dataset). 

<table><tr><td rowspan="3">Attack Type</td><td>Dataset →</td><td colspan="6">LIVEC</td><td colspan="6">KonIQ-10k</td></tr><tr><td rowspan="2">Model→Metrics→</td><td colspan="2">HyperIQA</td><td colspan="2">DBCNN</td><td colspan="2">TReS</td><td colspan="2">HyperIQA</td><td colspan="2">DBCNN</td><td colspan="2">TReS</td></tr><tr><td>mMAE</td><td>mMRA</td><td>mMAE</td><td>mMRA</td><td>mMAE</td><td>mMRA</td><td>mMAE</td><td>mMRA</td><td>mMAE</td><td>mMRA</td><td>mMAE</td><td>mMRA</td></tr><tr><td rowspan="3">P-BAIQA</td><td>1</td><td>7.825</td><td>0.7660</td><td>9.093</td><td>0.7900</td><td>11.241</td><td>0.8244</td><td>7.548</td><td>0.8540</td><td>6.741</td><td>0.8078</td><td>13.193</td><td>1.0916</td></tr><tr><td>2</td><td>9.485</td><td>0.5419</td><td>9.608</td><td>0.5756</td><td>11.831</td><td>0.6942</td><td>7.029</td><td>0.6413</td><td>5.510</td><td>0.6813</td><td>9.543</td><td>0.8308</td></tr><tr><td>Ours</td><td>8.719</td><td>0.6344</td><td>9.449</td><td>0.5611</td><td>10.223</td><td>0.7274</td><td>6.420</td><td>0.6487</td><td>5.208</td><td>0.6917</td><td>9.012</td><td>0.9291</td></tr><tr><td rowspan="3">C-BAIQA</td><td>3</td><td>12.689</td><td>0.4117</td><td>12.610</td><td>0.4195</td><td>14.597</td><td>0.3490</td><td>15.166</td><td>0.3002</td><td>14.277</td><td>0.3463</td><td>17.837</td><td>0.1714</td></tr><tr><td>4</td><td>20.777</td><td>0.0638</td><td>19.320</td><td>0.1257</td><td>19.089</td><td>0.1271</td><td>19.023</td><td>0.1924</td><td>16.388</td><td>0.3242</td><td>21.499</td><td>0.0249</td></tr><tr><td>Ours</td><td>10.653</td><td>0.5983</td><td>12.372</td><td>0.4525</td><td>13.746</td><td>0.4183</td><td>15.185</td><td>0.2859</td><td>13.970</td><td>0.3682</td><td>14.597</td><td>0.3490</td></tr></table>

Table 3: Ablation study of our attacks. ① and ② denote our P-BAIQA with $\alpha$ sampling from $\{\pm 1\}$ and U(-1, 1), respectively. ③ denotes our C-BAIQA without converting $x$ into $x'$ . ④ denotes our C-BAIQA by setting $\mu_y = 65$ for the sampling strategy of $\alpha$ .

with $\epsilon_{t}$ in terms of $\ell_{\infty}$ -norm (i.e., $\epsilon_{t} = 2/255$ for LIVEC and $\epsilon_{t} = 1/255$ for KonIQ-10k), $\alpha_{t} = 1/255$ , and $I_{t} = 20$ .

Results. From Table 1 and Table 2, none of the baseline methods successfully attack NR-IQA models. However, our C-BAIQA even achieves a comparable performance to our P-BAIQA on the LIVEC. On the KonIQ-10k, we observed a significant drop in the performance compared to P-BAIQA. This may be attributed to the fact that KonIQ-10k contains a larger number of clean data samples compared to LIVEC. Consequently, the NR-IQA models may have been less inclined to learn spurious correlations from the poisoned data and instead relied more on learning from the clean data. From Fig. 2 (b), we see a similar trend as the results of P-BAIQA.

# Resistance to Backdoor Defenses

We assess the resistance of our attacks to backdoor defenses. Given that strategies like Neural Cleanse (Wang et al. 2019), and STRIP (Gao et al. 2021) are tailored for defending against BA in classification, we delve into the resistance of our BAIQA against fine-tuning (Liu, Xie, and Srivastava 2017; Liu, Dolan-Gavitt, and Garg 2018) and model pruning (Liu, Dolan-Gavitt, and Garg 2018; Wu and Wang 2021). These are representative defenses that are applicable to our tasks. Further details on the setup are in the appendix.

As depicted in Fig. 3(a), our attacks are resist to fine-tuning. Notably, the mMAE for both P-BAIQA and C-BAIQA remain largely unaffected, experiencing only a marginal increase of approximately 2 after the whole fine-tuning. Furthermore, our attacks are resistant to model pruning, as evident from Fig. 3(b). Specifically, when applying a pruning rate of 40%, the mMAE values for P-BAIQA and C-BAIQA remain relatively stable, whereas the benign metric (RMSE) is significantly impacted, resulting in elevated values.

# Abaltion Study

Sampling of $\alpha$ for P-BAIQA. We explore three cases: 1) $\{\pm1, \pm\frac{3}{4}, \pm\frac{1}{2}, \pm\frac{1}{4}\}$ with probabilities $\{0.4, 0.3, 0.2, 0.1\}$ as ours; 2) $\{\pm1\}$ with equal probability; 3) Uniform distribution (i.e., U(-1, 1)). In Table 3, when the quantity of poisoned data is limited, e.g., on LIVEC, option 2) is a preferable choice. This preference stems from the fact that a higher absolute value of $\alpha$ results in a greater loss, thereby taking precedence during training, and making the model more effectively learn the correlation between the trigger and the attack target. When the amount of poisoned data is adequate, e.g., on KonIQ-10k, option 1), as adopted in P-BAIQA, provides more precise control regarding the mMAE during attacking. Value $\mu_{y}$ and strategy of converting x into $x'$ for C-BAIQA. From Table 3, the results of ③ show that the transformation of x into $x'$ enhances the attack's efficacy, consistent with Remark 2. Results of ③ show that values of $\mu_{y}$ that is closer to the mean MOS across the dataset yields improved attack effectiveness, which is in line with Remark 1.

# Conclusion

In this work, we introduce a novel poisoning-based backdoor attack against NR-IQA, that leverages the adjustment of a scaling coefficient $\alpha$ for the trigger to manipulate the model's output to desired target values. We propose embedding the trigger in the DCT domain, and incorporate UAP in the DCT space as the trigger, enhancing the IQA model's susceptibility to manipulation and improving the attack's efficacy. Our approach explores both poison-label and clean-label scenarios, the latter focusing on $\alpha$ sampling and image data refinement with theoretical insights. Extensive experiments on diverse datasets and NR-IQA models validate the effectiveness.

# Acknowledgments

This work was done at Rapid-Rich Object Search Lab, School of Electrical and Electronic Engineering, Nanyang Technological University. This research is supported in part by the Basic and Frontier Research Project of PCL, the Major Key Project of PCL, and Guangdong Basic and Applied Basic Research Foundation under Grant 2024A1515010454.

# References

Chen, X.; Liu, C.; Li, B.; Lu, K.; and Song, D. 2017. Targeted backdoor attacks on deep learning systems using data poisoning. arXiv preprint arXiv:1712.05526.   
Cheng, H.; Yang, S.; Zhou, J. T.; Guo, L.; and Wen, B. 2023. Frequency guidance matters in few-shot learning. In Proc. IEEE Int'l Conf. Computer Vision, 11814–11824.   
Ding, K.; Ma, K.; Wang, S.; and Simoncelli, E. P. 2020. Image quality assessment: Unifying structure and texture similarity. IEEE Trans. on Pattern Analysis and Machine Intelligence, 44(5): 2567–2581.   
Doan, K.; Lao, Y.; Zhao, W.; and Li, P. 2021. LIRA: Learnable, Imperceptible and Robust Backdoor Attacks. In Proc. IEEE Int'l Conf. Computer Vision, 11966–11976.   
Dosovitskiy, A.; Beyer, L.; Kolesnikov, A.; Weissenborn, D.; Zhai, X.; Unterthiner, T.; Dehghani, M.; Minderer, M.; Heigold, G.; Gelly, S.; et al. 2021. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In Proc. Int'l Conf. Learning Representations.   
Dumford, J.; and Scheirer, W. 2020. Backdooring convolutional neural networks via targeted weight perturbations. In Proc. IEEE Int'l Joint Conf. on Biometrics, 1–9.   
Fang, S.; and Choromanska, A. 2022. Backdoor attacks on the DNN interpretation system. In Proc. AAAI Conf. on Artificial Intelligence, volume 36, 561–570.   
Fu, H.; Liang, F.; Liang, J.; Li, B.; Zhang, G.; and Han, J. 2023. Asymmetric learned image compression with multi-scale residual block, importance scaling, and post-quantization filtering. IEEE Trans. on Circuits and Systems for Video Technology.   
Gao, Y.; Kim, Y.; Doan, B. G.; Zhang, Z.; Zhang, G.; Nepal, S.; Ranasinghe, D. C.; and Kim, H. 2021. Design and evaluation of a multi-domain trojan detection method on deep neural networks. IEEE Trans. on Dependable and Secure Computing, 19(4): 2349–2364.   
Ghadiyaram, D.; and Bovik, A. C. 2015. Massive online crowdsourced study of subjective and objective picture quality. IEEE Trans. on Image Processing, 25(1): 372–387.   
Ghadiyaram, D.; and Bovik, A. C. 2017. Perceptual quality prediction on authentically distorted images using a bag of features approach. Journal of vision, 17(1): 32–32.   
Golestaneh, S. A.; Dadsetan, S.; and Kitani, K. M. 2022. No-Reference Image Quality Assessment via Transformers, Relative Ranking, and Self-Consistency. In Proc. of the IEEE/CVF Winter Conf. on Applications of Computer Vision.   
Gu, T.; Dolan-Gavitt, B.; and Garg, S. 2017. Badnets: Identifying vulnerabilities in the machine learning model supply chain. arXiv preprint arXiv:1708.06733.

Guo, C.; Wu, R.; and Weinberger, K. Q. 2020. Trojannet: Embedding hidden trojan horse models in neural networks. arXiv preprint arXiv:2002.10078.   
He, K.; Zhang, X.; Ren, S.; and Sun, J. 2016. Deep residual learning for image recognition. In Proc. IEEE Int'l Conf. Computer Vision and Pattern Recognition, 770–778.   
Hosu, V.; Lin, H.; Sziranyi, T.; and Saupe, D. 2020. KonIQ-10k: An ecologically valid database for deep learning of blind image quality assessment. IEEE Trans. on Image Processing.   
Ilyas, A.; Engstrom, L.; Athalye, A.; and Lin, J. 2018. Black-box adversarial attacks with limited queries and information. In Proc. Int'l Conf. Machine Learning, 2137–2146.   
Ke, J.; Wang, Q.; Wang, Y.; Milanfar, P.; and Yang, F. 2021. Musiq: Multi-scale image quality transformer. In Proc. IEEE Int'l Conf. Computer Vision, 5148–5157.   
Korhonen, J.; and You, J. 2022. Adversarial attacks against blind image quality assessment models. In Proc. of the 2nd Workshop on Quality of Experience in Visual Multimedia Applications.   
Li, S.; Xue, M.; Zhao, B. Z. H.; Zhu, H.; and Zhang, X. 2020a. Invisible backdoor attacks on deep neural networks via steganography and regularization. IEEE Trans. on Dependable and Secure Computing, 18(5): 2088–2105.   
Li, Y.; Li, Y.; Wu, B.; Li, L.; He, R.; and Lyu, S. 2021. Invisible backdoor attack with sample-specific triggers. In Proc. IEEE Int'l Conf. Computer Vision, 16463–16472.   
Li, Y.; Wu, B.; Jiang, Y.; Li, Z.; and Xia, S.-T. 2020b. Backdoor learning: A survey. arXiv preprint arXiv:2007.08745.   
Liang, K.; Zhang, J. Y.; Wang, B.; Yang, Z.; Koyejo, S.; and Li, B. 2021. Uncovering the connections between adversarial transferability and knowledge transferability. In Proc. Int'l Conf. Machine Learning.   
Liu, K.; Dolan-Gavitt, B.; and Garg, S. 2018. Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks. arXiv preprint arXiv:1805.12185.   
Liu, Y.; Ma, X.; Bailey, J.; and Lu, F. 2020. Reflection backdoor: A natural backdoor attack on deep neural networks. In European Conf. on Computer Vision, 182–199.   
Liu, Y.; Xie, Y.; and Srivastava, A. 2017. Neural trojans. In IEEE Int'l Conf. on Computer Design, 45–48.   
Liu, Y.; Yang, C.; Li, D.; Ding, J.; and Jiang, T. 2024a. Defense Against Adversarial Attacks on No-Reference Image Quality Models with Gradient Norm Regularization. arXiv preprint arXiv:2403.11397.   
Liu, Z.; Wang, T.; Huai, M.; and Miao, C. 2024b. Backdoor attacks via machine unlearning. In Proc. AAAI Conf. on Artificial Intelligence, volume 38, 14115–14123.   
Madry, A.; Makelov, A.; Schmidt, L.; Tsipras, D.; and Vladu, A. 2018. Towards Deep Learning Models Resistant to Adversarial Attacks. In Proc. Int'l Conf. Learning Representations. Mittal, A.; Moorthy, A. K.; and Bovik, A. C. 2012. No-reference image quality assessment in the spatial domain. IEEE Trans. on Image Processing, 21(12): 4695–4708.   
Nguyen, T. A.; and Tran, A. 2020. Input-aware dynamic backdoor attack. In Proc. Annual Conf. Neural Information Processing Systems, volume 33, 3454–3464.

Nguyen, T. A.; and Tran, A. T. 2021. WaNet - Imperceptible Warping-based Backdoor Attack. In Proc. Int'l Conf. Learning Representations.   
Rakin, A. S.; He, Z.; and Fan, D. 2020. Tbt: Targeted neural network attack with bit trojan. In Proc. IEEE Int'l Conf. Computer Vision and Pattern Recognition, 13198–13207.   
Rippel, O.; Nair, S.; Lew, C.; Branson, S.; Anderson, A. G.; and Bourdev, L. 2019. Learned video compression. In Proc. IEEE Int'l Conf. Computer Vision, 3454–3463.   
Saha, A.; Subramanya, A.; and Pirsiavash, H. 2020. Hidden trigger backdoor attacks. In Proc. AAAI Conf. on Artificial Intelligence, volume 34, 11957–11965.   
Shumitskaya, E.; Antsiferova, A.; and Vatolin, D. 2022. Universal perturbation attack on differentiable no-reference image-and video-quality metrics. In BMVC.   
Steinhardt, J.; Koh, P. W. W.; and Liang, P. S. 2017. Certified defenses for data poisoning attacks. In Proc. Annual Conf. Neural Information Processing Systems, volume 30.   
Su, S.; Yan, Q.; Zhu, Y.; Zhang, C.; Ge, X.; Sun, J.; and Zhang, Y. 2020. Blindly assess image quality in the wild guided by a self-adaptive hyper network. In Proc. IEEE Int'l Conf. Computer Vision and Pattern Recognition, 3667–3676.   
Szegedy, C.; Zaremba, W.; Sutskever, I.; Bruna, J.; Erhan, D.; Goodfellow, I.; and Fergus, R. 2013. Intriguing properties of neural networks. arXiv preprint arXiv:1312.6199.   
Wallace, G. K. 1992. The JPEG still picture compression standard. IEEE Trans. on Consumer Electronics, 38: 43–59.   
Wang, B.; Yao, Y.; Shan, S.; Li, H.; Viswanath, B.; Zheng, H.; and Zhao, B. Y. 2019. Neural cleanse: Identifying and mitigating backdoor attacks in neural networks. In IEEE Symposium on Security and Privacy, 707–723.   
Wang, T.; Yao, Y.; Xu, F.; An, S.; Tong, H.; and Wang, T. 2022. An invisible black-box backdoor attack through frequency domain. In European Conf. on Computer Vision.   
Wang, X.; Hu, S.; Zhang, Y.; Zhou, Z.; Zhang, L. Y.; Xu, P.; Wan, W.; and Jin, H. 2024a. ECLIPSE: Expunging clean-label indiscriminate poisons via sparse diffusion purification. In European Symposium on Research in Computer Security.   
Wang, X.; Li, M.; Liu, W.; Zhang, H.; Hu, S.; Zhang, Y.; Zhou, Z.; and Jin, H. 2024b. Unlearnable 3D point clouds: Class-wise transformation is all you need. In Proc. Annual Conf. Neural Information Processing Systems.   
Wang, Z.; Bovik, A. C.; Sheikh, H. R.; and Simoncelli, E. P. 2004. Image quality assessment: from error visibility to structural similarity. IEEE Trans. on Image Processing.   
Wu, D.; and Wang, Y. 2021. Adversarial neuron pruning purifies backdoored deep models. In Advances in Neural Information Processing Systems, volume 34, 16913–16925.   
Xia, S.; Yang, W.; Yu, Y.; Lin, X.; Ding, H.; DUAN, L.; and Jiang, X. 2024a. Transferable Adversarial Attacks on SAM and Its Downstream Models. In Proc. Annual Conf. Neural Information Processing Systems.   
Xia, S.; Yu, Y.; Jiang, X.; and Ding, H. 2024b. Mitigating the Curse of Dimensionality for Certified Robustness via Dual Randomized Smoothing. In Proc. Int'l Conf. Learning Representations.

Yang, S.; Wu, T.; Shi, S.; Lao, S.; Gong, Y.; Cao, M.; Wang, J.; and Yang, Y. 2022. MANIQA: Multi-dimension Attention Network for No-Reference Image Quality Assessment. arXiv preprint arXiv:2204.08958.

Yu, F.; Zeng, B.; Zhao, K.; Pang, Z.; and Wang, L. 2024a. Chronic Poisoning: Backdoor Attack against Split Learning. In Proc. AAAI Conf. on Artificial Intelligence.

Yu, Y.; Wang, Y.; Xia, S.; Yang, W.; Lu, S.; Tan, Y.-P.; and Kot, A. C. 2024b. Purify Unlearnable Examples via Rate-Constrained Variational Autoencoders. In Proc. Int'l Conf. Machine Learning.

Yu, Y.; Wang, Y.; Yang, W.; Guo, L.; Lu, S.; Duan, L.-Y.; Tan, Y.-P.; and Kot, A. C. 2024c. Robust and Transferable Backdoor Attacks Against Deep Image Compression With Selective Frequency Prior. IEEE Trans. on Pattern Analysis and Machine Intelligence.

Yu, Y.; Wang, Y.; Yang, W.; Lu, S.; Tan, Y.-P.; and Kot, A. C. 2023. Backdoor attacks against deep image compression via adaptive frequency trigger. In Proc. IEEE Int'l Conf. Computer Vision and Pattern Recognition, 12250–12259.

Yu, Y.; Yang, W.; Tan, Y.-P.; and Kot, A. C. 2022. Towards Robust Rain Removal Against Adversarial Attacks: A Comprehensive Benchmark Analysis and Beyond. In Proc. IEEE Int'l Conf. Computer Vision and Pattern Recognition.

Yue, C.; Lv, P.; Liang, R.; and Chen, K. 2022. Invisible backdoor attacks using data poisoning in the frequency domain. arXiv preprint arXiv:2207.04209.

Zeng, Y.; Park, W.; Mao, Z. M.; and Jia, R. 2021. Rethinking the backdoor attacks' triggers: A frequency perspective. In Proc. IEEE Int'l Conf. Computer Vision, 16473–16481.

Zhang, A.; Ran, Y.; Tang, W.; and Wang, Y.-G. 2023. Vulnerabilities in video quality assessment models: The challenge of adversarial attacks. In Proc. Annual Conf. Neural Information Processing Systems, volume 36.

Zhang, R.; Isola, P.; Efros, A. A.; Shechtman, E.; and Wang, O. 2018. The unreasonable effectiveness of deep features as a perceptual metric. In Proc. IEEE Int'l Conf. Computer Vision and Pattern Recognition, 586–595.

Zhang, W.; Li, D.; Min, X.; Zhai, G.; Guo, G.; Yang, X.; and Ma, K. 2022. Perceptual attacks of no-reference image quality models with human-in-the-loop. In Proc. Annual Conf. Neural Information Processing Systems.

Zhang, W.; Liu, Y.; Dong, C.; and Qiao, Y. 2019. Ranksrgan: Generative adversarial networks with ranker for image super-resolution. In Proc. IEEE Int'l Conf. Computer Vision.

Zhang, W.; Ma, K.; Yan, J.; Deng, D.; and Wang, Z. 2020. Blind Image Quality Assessment Using A Deep Bilinear Convolutional Neural Network. IEEE Trans. on Circuits and Systems for Video Technology, 30(1): 36–47.

Zhang, W.; Ma, K.; Zhai, G.; and Yang, X. 2021. Uncertainty-aware blind image quality assessment in the laboratory and wild. IEEE Trans. on Image Processing, 30: 3474–3486.

Zheng, Q.; Yu, Y.; Yang, S.; Liu, J.; Lam, K.-Y.; and Kot, A. 2024. Towards Physical World Backdoor Attacks against Skeleton Action Recognition. In European Conf. on Computer Vision.

# Appendix

# Impact Statement

The research presented in this paper, focusing on backdoor attacks against No-Reference Image Quality Assessment (NR-IQA) models, while valuable for understanding and defending against potential threats, also brings forth significant social implications. NR-IQA systems are crucial for a wide range of applications, including but not limited to medical imaging, autonomous driving, and surveillance systems. As such, vulnerabilities in these systems have the potential to cause harm at both individual and societal levels. Firstly, successful backdoor attacks on NR-IQA models can be leveraged for malicious purposes, such as altering the perception of image quality in critical applications. In medical imaging, this could lead to misdiagnosis or incorrect treatment plans. In autonomous driving, it could result in the system misinterpreting road conditions, potentially leading to accidents. In surveillance systems, attackers could potentially manipulate video feeds to evade detection. Secondly, the methods presented in this research demonstrate that current NR-IQA models are susceptible to manipulation, highlighting the need for greater attention and resources towards enhancing their security and robustness. As NR-IQA systems become more ubiquitous in society, their potential impact also increases, and it is essential that researchers and developers prioritize security considerations in their design and implementation. Finally, this research also raises awareness of the broader issue of adversarial machine learning, and the need for continued research in this area. As machine learning and AI technologies continue to proliferate, the potential for malicious use of these technologies also increases. By studying and defending against adversarial attacks, we can help ensure that these technologies are used responsibly and safely. In summary, while the research presented in this abstract provides valuable insights into vulnerabilities in NR-IQA models, it also highlights the need for greater attention towards the social implications of these vulnerabilities, and the need for continued research in adversarial machine learning to ensure the safe and responsible use of AI technologies.

# Formulations of RMSE, SROCC, and PLCC

In this section, we will introduce IQA-specific metrics RMSE, SROCC, and PLCC in Sec 4.1.

RMSE measures the difference between MOS values and predicted scores, which is represented as

$$
\mathrm{RMSE} = \sqrt {\frac {1}{N} \sum_ {i = 1} ^ {N} (y _ {i} - f _ {i}) ^ {2}}. \tag {9}
$$

In this equation, N is the number of images. $y_{i}$ and $f_{i}$ represent the MOS and predicted score of the $i^{th}$ image, respectively. The smaller the RMSE value is, the smaller the differences between the two groups of scores.

SROCC measures the correlation between MOS values and predicted scores to what extent the correlation can be described by a monotone function. The specific formulation is as follows:

$$
\mathrm{SROCC} = 1 - \frac {6 \sum_ {i = 1} ^ {N} d _ {i} ^ {2}}{N (N ^ {2} - 1)}, \tag {10}
$$

where $d_{i}$ denotes the difference between orders of the $i^{th}$ image in subjective and objective quality scores. The closer the SROCC value is to 1, the more consistent the ordering is between two groups of scores.

PLCC measures the linear correlation between MOS values and predicted scores, which is formulated as

$$
\mathrm{PLCC} = \frac {\sum_ {i = 1} ^ {N} (y _ {i} - \bar {y}) (f _ {i} - \bar {f})}{\sum_ {i = 1} ^ {N} (y _ {i} - \bar {y}) ^ {2} (f _ {i} - \bar {f}) ^ {2}}, \tag {11}
$$

$$
\bar {y} = \frac {1}{N} \sum_ {i = 1} ^ {N} y _ {i}, \bar {f} = \frac {1}{N} \sum_ {i = 1} ^ {N} f _ {i}.
$$

The closer the PLCC value is to 1, the higher the positive correlation between the two groups of scores.

# Hardware Setup

We conducted all the training, test, and attack on an NVIDIA GeForce RTX 3090 GPU with 24GB of memory.

# Standard Training of NR-IQA models

For the standard training of three NR-IQA models: HyperIQA (Su et al. 2020), DBCNN (Zhang et al. 2020), and TReS (Golestaneh, Dadsetan, and Kitani 2022), we follow their official code, and model architecture, i.e., ResNet-50 as the feature extractor for HyperIQA, VGG for DBCNN, and ResNet-18 for TReS.

When considering data augmentations and evaluation for testing, we establish a cropping size of $224 \times 224$ and utilize 25 patches per image. For all models, a batch size of 32 is adopted.

HyperIQA. Specifically for the HyperIQA model, we select the L1 loss function, employ the Adam optimizer, set an initial learning rate of 2e-5, a weight decay of 5e-4, and train for 24 epochs. Additionally, we incorporate a multistep learning rate scheduler with a step of 8 epochs and a gamma value of 0.1.

DBCNN. For the DBCNN model, we choose the MSE loss. The initial 8 epochs are dedicated to training the fully-connected layer alone, utilizing the SGD optimizer with a fixed learning rate of 1e-3 and a weight decay of 5e-4. In the subsequent 8 epochs, we switch to the Adam optimizer, maintaining a fixed learning rate of 1e-5 and the same weight decay.

TReS. Lastly, for the TReS model, we opt for the L1 loss, utilize the Adam optimizer, set an initial learning rate of 2e-5, a weight decay of 5e-4, and train for 12 epochs. A multistep learning rate scheduler with a step of 4 epochs and a gamma value of 0.1 is also employed.

# Detailed Setup of The Baseline Attacks

We have incorporated Blended (Chen et al. 2017) and WaNet (Nguyen and Tran 2021) into our work by utilizing the open-source Python toolbox, BackdoorBox (?). For Blended,

when testing on the LIVEC dataset, we randomly sample the noise from a uniform distribution U[0, 1] and set the weight to 0.1 when the scaling coefficient $\alpha = 1$ . Similarly, for WaNet on the LIVEC dataset, we adhere to the default settings, setting the uniform grid size to 4, and the strength to 0.4 when $\alpha = 1$ . Furthermore, we have conducted experiments with FTrojan (Wang et al. 2022) using the official codebase. In our experiments, we select the middle 64 frequencies to embed the trigger, and on the LIVEC dataset, we set the magnitude of the trigger to 66 when $\alpha = 1$ , which is well-aligned with our approach. When adjusting the strength of the trigger, we apply the $\alpha$ to the weight in Blended, the strength in WaNet, and the magnitude in FTrojan. This allows us to control the intensity or visibility of the trigger in each respective method. On the Koniq-10k dataset, for a fair comparison with our method, we have reduced the values of all strength-related hyperparameters for the baseline methods by half.

# Detailed Setup of The Backdoor Defenses

Settings for Fine-tuning. As an example for discussion, we perform experiments on the LIVEC dataset. We choose to fine-tune the whole network on a randomly selected benign subset. Specifically, for fine-tuning, we utilize 20% of the benign training samples, while maintaining all other training configurations in line with the standard training procedures for NR-IQA models.

Settings for Model Pruning. We conduct the experiments on the LIVEC dataset as an example for discussion. Following its default settings, we conduct channel pruning (?) on the output of the last convolutional layer of the feature extractor with 20% benign training samples. The pruning rate $\beta \in \{0\%, 5\%, \cdots, 95\%\}$ .

# More Details of Experimental Results

Main experiments. Herein, we present an extended analysis and results of the MAE( $\alpha$ ) for both P-BAIQA and C-BAIQA.

As depicted in Fig. 2 and 4, during clean label attacks, all baseline methods are unable to effectively compromise the NR-IQA models. This deficiency may be attributed to the limited representation capacity of the trigger. Notably, in the case of the Blended approach, it becomes apparent that the backdoored model primarily learns to diminish the output MOS scores regardless of the value of $\alpha$ .

Conversely, for poison label attacks, our proposed method significantly outperforms Blended and WaNet, and exhibits slightly superior performance compared to FTrojan.

In addition to mMAE and mMRA, which assess the attack's mean effectiveness, we provide the MRA( $\alpha$ ) in Fig. 2(a). We can see that our method is capable of achieving a significant deviation when $\alpha$ has an absolute value greater than 0.5. However, when $\alpha$ falls within [-0.5, 0.5], the manipulation becomes less precise, as the model finds it more challenging to recognize the trigger. However, we believe this limitation is tolerable given that attackers generally aim for significant shifts in outputs rather than minor alterations. Results of MAE( $\alpha$ ) and visualized results of poisoned images are in Appendix. Moreover, we provide the results of P-BAIQA with lower poisoning ratios in Table 4 in the ap-

![](images/f66fc17803b6ed1499f669cdee18266850b908ebc84f925868c21fe099048c05.jpg)

<details>
<summary>line</summary>

| Dataset       | Ours  | FTrojan | Blended | WaNet |
| ------------- | ----- | ------- | ------- | ----- |
| LIVEC dataset | 35.0  | 12.0    | 30.0    | 40.0  |
| KonIQ-10k dataset | 35.0  | 12.0    | 30.0    | 40.0  |
</details>

(a) MAE(α) of poison-label attacks

![](images/70c0746c961d240ea9c7b6ab0fdcaf5cc25d4c2ec769dc08b1e38655dd029a72.jpg)

<details>
<summary>line</summary>

| Dataset       | Ours  | FTrojan | Blended | WaNet | LIVEC |
| ------------- | ----- | ------- | ------- | ----- | ----- |
| LIVEC dataset | 35    | 30      | 38      | 32    | 40    |
| KonIQ-10k dataset | 30    | 25      | 35      | 28    | 32    |
</details>

(b) MAE(α) of clean-label attacks

Figure 4: MAE( $\alpha$ ) for both P-BAIQA and C-BAIQA (HyperIQA as victim models).   
![](images/ec091cbb919fb64bef97fd7960ea1df3be315d1331c6df24b1a4764e1f15b8d7.jpg)  
GT: 29.46
Pred (Ours): 34.62

![](images/c5349df68afab934d56e7ff853c551bd276347dbca360c0ffa27e078b6efd4d4.jpg)  
PSNR: 31.49
Pred:64.88 (+25.34)

![](images/768dbc6083303613f1d09143376dce5a8cdec958165309d678a162dea8c0f723.jpg)  
PSNR: 37.40
Pred:49.32 (+4.08)

![](images/34f459b558b8d110abf907b130059c4c9f59744adb432a8c840b740ca5857fcf.jpg)  
PSNR: 36.01
Pred:75.13 (+39.29)

![](images/ee43157a828d3b5db98b2d8ec5da045bb92c7ae70b5e57f8fcbaacb546bfb542.jpg)  
PSNR: 36.74
Pred:74.20 (+39.58)   
Figure 5: Visualized results of poison-label attacks on Koniq-10k. For all poisoned images, we adopt $\alpha = 1$ , and the deviations of the output are compared to the clean output of corresponding models.

pendix, and can observe that higher poisoning ratios indicate better attack effectiveness.

More visualized results. In Fig. 5 and 6, we offer visualizations that illustrate the comparative results of our method and the baseline approaches (we set $\alpha = 1$ for poisoned images, and the ideal deviations $\alpha \cdot \Delta y_{t} = 40$ ). For poison-label attacks, it is evident that our method outperforms the others, demonstrating superior attack performance while minimizing the impact on clean data. When considering clean-label attacks, it becomes apparent that the baseline methods are unable to effectively attack NR-IQA models. Specifically, Blended and FTrojan techniques struggle to achieve significant decreases in output scores, while WaNet exhibits almost no deviation in its performance.

Ablation study. Here, we present the MRA( $\alpha$ ) and MSE( $\alpha$ ) values for P-BAIQA and C-BAIQA, corresponding to the ablation study conducted in Section 5.1.

As shown in Fig. 7, on both datasets, while approach (1) introduces greater deviations in the output, its manipulation is not precise, often resulting in a higher MAE. In comparison, approach (2) performs worse in terms of both MRA and MAE.

Furthermore, Fig. 8 demonstrates that converting x to $x'$ effectively enhances the attack performance, as compared to approach (3). The results of approach (4) highlight the crucial role of the $\mu_{y}$ selection. Specifically, when $\mu_{y}$ is set to 65, the backdoored model learns to solely decrease the output MOS scores, regardless of the $\alpha$ value chosen for the trigger.

Table 4: Evaluation of P-BAIQA with different poisoning ratios. 

<table><tr><td rowspan="3">Attack Type</td><td colspan="2">Dataset →</td><td colspan="6">LIVEC</td><td colspan="6">KonIQ-10k</td></tr><tr><td rowspan="2">Model↓</td><td rowspan="2">Ratio r (%)↓</td><td colspan="3">Benign Metrics</td><td colspan="3">Attack Metrics</td><td colspan="3">Benign Metrics</td><td colspan="3">Attack Metrics</td></tr><tr><td>PLCC</td><td>SROCC</td><td>RMSE</td><td>PSNR $_1$ </td><td>mMAE</td><td>mMRA</td><td>PLCC</td><td>SROCC</td><td>RMSE</td><td>PSNR $_1$ </td><td>mMAE</td><td>mMRA</td></tr><tr><td rowspan="6">P-BAIQA</td><td rowspan="2">HyperIQA</td><td>5</td><td>0.9052</td><td>0.8839</td><td>9.580</td><td>30.06</td><td>15.023</td><td>0.2799</td><td>0.9011</td><td>0.8882</td><td>6.902</td><td>36.32</td><td>11.277</td><td>0.4021</td></tr><tr><td>10</td><td>0.9051</td><td>0.8921</td><td>9.705</td><td>30.06</td><td>13.896</td><td>0.3336</td><td>0.9071</td><td>0.8871</td><td>6.883</td><td>36.32</td><td>8.415</td><td>0.7494</td></tr><tr><td rowspan="2">DBCNN</td><td>5</td><td>0.8762</td><td>0.8542</td><td>10.348</td><td>30.06</td><td>13.271</td><td>0.3517</td><td>0.9005</td><td>0.8829</td><td>7.070</td><td>36.32</td><td>7.777</td><td>0.5846</td></tr><tr><td>10</td><td>0.8667</td><td>0.8624</td><td>10.573</td><td>30.06</td><td>11.297</td><td>0.4798</td><td>0.9009</td><td>0.8852</td><td>7.008</td><td>36.32</td><td>6.986</td><td>0.6077</td></tr><tr><td rowspan="2">TReS</td><td>5</td><td>0.8825</td><td>0.8543</td><td>17.879</td><td>30.06</td><td>19.978</td><td>0.0825</td><td>0.8945</td><td>0.8780</td><td>20.874</td><td>36.32</td><td>11.494</td><td>0.5029</td></tr><tr><td>10</td><td>0.8788</td><td>0.8574</td><td>16.941</td><td>30.06</td><td>14.619</td><td>0.3327</td><td>0.8994</td><td>0.8785</td><td>22.315</td><td>36.32</td><td>9.797</td><td>0.6772</td></tr></table>

![](images/4cf075cfdcd638f302ec4d571ebf46da6a007ae47ccfea891a7d64bb66d975a4.jpg)  
GT: 30.91
Pred (Ours): 39.34

![](images/d8bb5034e920ee5d11adaa026783e7e6fd25319b8c37e30c49002f1e46fb52ed.jpg)  
PSNR: 28.20
Pred:29.64 (-9.92)

![](images/037006a1c8461aab51b6d1e40daaa973ef09a761a1880b95eb03cb8af7afb8c4.jpg)  
PSNR: 35.22
Pred:39.67 (+0.75)

![](images/28391f6fbda0af0c22cb91a7ae54178d4d54bdf68c6fd5cc807f009798cf8921.jpg)  
PSNR: 30.20
Pred:79.95 (-3.34)

![](images/b1b6c6510eb14c34b77110d58e7effde7ed36337abd64f4032876aa091b05676.jpg)  
PSNR: 30.32
Pred: 71.14 (+31.80)   
Figure 6: Visualized results of clean-label attacks on LIVEC. For all poisoned images, we adopt $\alpha = 1$ , and the deviations of the output are compared to the clean output of corresponding models.

![](images/e979b01a03c558bc698317e78f0aeb03b4771c7d494c4ce647abed334497d227.jpg)

<details>
<summary>line</summary>

| Dataset       | α    | MRA  |
| ------------- | ---- | ---- |
| LIVEC dataset | -1.0 | 1.00 |
| LIVEC dataset | -0.5 | 0.75 |
| LIVEC dataset | 0.0  | 0.00 |
| LIVEC dataset | 0.5  | 0.75 |
| LIVEC dataset | 1.0  | 1.00 |
| Koniq-10k dataset | -1.0 | 1.00 |
| Koniq-10k dataset | -0.5 | 0.75 |
| Koniq-10k dataset | 0.0  | 0.00 |
| Koniq-10k dataset | 0.5  | 0.75 |
| Koniq-10k dataset | 1.0  | 1.00 |
</details>

(a) MRA(α) of poison-label attacks

![](images/95582bd627a3de85a3fc6fb12ae968af90fe13c72d0aef73c952a4b38c9a2405.jpg)

<details>
<summary>line</summary>

| α     | Ours (LIVEC dataset) | Ours (Koniq-10k dataset) | (1) (LIVEC dataset) | (1) (Koniq-10k dataset) | (2) (LIVEC dataset) | (2) (Koniq-10k dataset) |
|-------|----------------------|--------------------------|--------------------|------------------------|--------------------|------------------------|
| -1.0  | 10.0                 | 10.0                     | 10.0               | 10.0                   | 10.0               | 10.0                   |
| -0.5  | 8.0                  | 8.0                      | 8.0                | 8.0                    | 8.0                | 8.0                    |
| 0.0   | 4.0                  | 4.0                      | 4.0                | 4.0                    | 4.0                | 4.0                    |
| 0.5   | 6.0                  | 6.0                      | 6.0                | 6.0                    | 6.0                | 6.0                    |
| 1.0   | 12.0                 | 12.0                     | 12.0               | 12.0                   | 12.0               | 12.0                   |
</details>

(b) MAE( $\alpha$ ) of poison-label attacks   
Figure 7: Ablation study: MRA( $\alpha$ ) and MSE( $\alpha$ ) for P-BAIQA (HyperIQA as victim models).

![](images/d4173bd445f909d34484e06a74ce4041cc9ff0e6939b647a8ad7877cf0530408.jpg)

<details>
<summary>line</summary>

| α    | Ours (LIVEC) | Ours (Koniq-10k) | (3) (LIVEC) | (3) (Koniq-10k) | (4) (LIVEC) | (4) (Koniq-10k) |
|------|--------------|------------------|-------------|-----------------|-------------|-----------------|
| -1.0 | 0.75         | 0.75             | 0.75        | 0.75            | 0.75        | 0.75            |
| -0.5 | 0.70         | 0.70             | 0.70        | 0.70            | 0.70        | 0.70            |
| 0.0  | 0.65         | 0.65             | 0.65        | 0.65            | 0.65        | 0.65            |
| 0.5  | 0.75         | 0.75             | 0.75        | 0.75            | 0.75        | 0.75            |
| 1.0  | 0.75         | -0.50            | 0.75        | -0.50           | -0.50       | -0.50           |
</details>

(a) MRA(α) of clean-label attacks

![](images/851075d699214d84d170a3280ab83b827ea6072910498bd9a3e228f88aa96dea.jpg)

<details>
<summary>line</summary>

| Dataset       | Method | MAE  |
| ------------- | ------ | ---- |
| LIVEC dataset | Ours   | 20   |
| LIVEC dataset | (3)    | 25   |
| LIVEC dataset | (4)    | 30   |
| Koniq-10k dataset | Ours  | 20   |
| Koniq-10k dataset | (3)  | 25   |
| Koniq-10k dataset | (4)  | 30   |
</details>

(b) MAE( $\alpha$ ) of clean-label attacks   
Figure 8: Ablation study: MRA( $\alpha$ ) and MSE( $\alpha$ ) for C-BAIQA (HyperIQA as victim models).