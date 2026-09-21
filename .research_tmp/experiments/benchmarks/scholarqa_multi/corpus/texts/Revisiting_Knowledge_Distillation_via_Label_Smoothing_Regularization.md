# Revisiting Knowledge Distillation via Label Smoothing Regularization

Li Yuan $^{1}$ Francis EH Tay $^{1}$ Guilin Li $^{2}$ Tao Wang $^{1}$ Jiashi Feng $^{1}$

$^{1}$ National University of Singapore $^{2}$ Huawei Noah's Ark Lab {ylustcnus, twangnh}@gmail.com, {mpetayeh,elefjia}@nus.edu.sg, guilinli2@huawei.com

# Abstract

Knowledge Distillation (KD) aims to distill the knowledge of a cumbersome teacher model into a lightweight student model. Its success is generally attributed to the privileged information on similarities among categories provided by the teacher model, and in this sense, only strong teacher models are deployed to teach weaker students in practice. In this work, we challenge this common belief by following experimental observations: 1) beyond the acknowledgment that the teacher can improve the student, the student can also enhance the teacher significantly by reversing the KD procedure; 2) a poorly-trained teacher with much lower accuracy than the student can still improve the latter significantly. To explain these observations, we provide a theoretical analysis of the relationships between KD and label smoothing regularization. We prove that 1) KD is a type of learned label smoothing regularization and 2) label smoothing regularization provides a virtual teacher model for KD. From these results, we argue that the success of KD is not fully due to the similarity information between categories from teachers, but also to the regularization of soft targets, which is equally or even more important.

Based on these analyses, we further propose a novel Teacher-free Knowledge Distillation (Tf-KD) framework, where a student model learns from itself or manually-designed regularization distribution. The Tf-KD achieves comparable performance with normal KD from a superior teacher, which is well applied when a stronger teacher model is unavailable. Meanwhile, Tf-KD is generic and can be directly deployed for training deep neural networks. Without any extra computation cost, Tf-KD achieves up to 0.65% improvement on ImageNet over well-established baseline models, which is superior to label smoothing regularization.

# 1. Introduction

Knowledge Distillation (KD) [7] aims to transfer knowledge from one neural network (teacher) to another (student). Usually, the teacher model has a strong learning capacity with higher performance, which teaches a lower-capacity student model through providing “soft targets”. It is commonly believed that the soft targets of the teacher model can transfer “dark knowledge” containing privileged information on similarity among different categories $[7]$ to enhance the student model.

In this work, we first examine such a common belief through following exploratory experiments: 1) let student models teach teacher models by transferring soft targets of the students; (2) let poorly-trained teacher models with worse performance teach students. Based on the common belief, it is expected that the teacher model would not be enhanced significantly via training from the students and poorly-trained teachers would not enhance the students, as the weak student and poorly-trained teacher models cannot provide reliable similarity information between categories. However, after extensive experiments on various models and datasets, we observe contradictory results: the weak student can improve the teacher and the poorly-trained teacher can also enhance the student remarkably. Such intriguing results motivate us to interpret KD as a regularization term, and we re-examine knowledge distillation from the perspective of Label Smoothing Regularization (LSR) [16] that regularizes model training by replacing the one-hot labels with smoothed ones. $^{1}$

We then analyze theoretically the relationships between KD and LSR. For LSR, by splitting the smoothed label into two parts and examining the corresponding losses, we find the first part is the ordinary cross-entropy for ground-truth distribution (one-hot label) and outputs of model, and the second part corresponds to a virtual teacher model which provides a uniform distribution to teach the model. For KD, by combining the teacher's soft targets with the one-hot ground-truth label, we find that KD is a learned LSR where the smoothing distribution of KD is from a teacher model but the smoothing distribution of LSR is manually designed. In a nutshell, we find KD is a learned LSR and LSR is an ad-hoc KD. Such relationships can explain the

above counterintuitive results—the soft targets from weak student and poorly-trained teacher models can effectively regularize the model training, even though they lack strong similarity information between categories. We therefore argue that the similarity information between categories cannot fully explain the dark knowledge in KD, and the soft targets from the teacher model indeed provide effective regularization for the student model, which are equally or even more important.

Based on the analyses, we conjecture that with non-reliable or even zero similarity information between categories from the teacher model, KD may still well improve the student models. We thus propose a novel Teacher-free Knowledge Distillation (Tf-KD) framework with two implementations. The first one is to train the student model by itself (i.e., self-training), and the second is to manually design a target distribution as a virtual teacher model which has 100% accuracy. The first method is motivated by replacing the dark knowledge with predictions from the model itself, and the second method is inspired by the relationships between KD and LSR. We validate through extensive experiments that the two implementations of Tf-KD are both simple yet effective. Particularly, in the second implementation without similarity information in the virtual teacher, Tf-KD still achieves comparable performance with normal KD, which clearly justifies:

Dark knowledge does not just include the similarity between categories, but also imposes regularization on the student training.

Tf-KD well applies to scenarios where the student model is too strong to find teacher models or computational resource is limited for training teacher models. For example, if we take a cumbersome single model ResNeXt101-32×8d [18] as the student model (with 88.79M parameters and 16.51G FLOPs on ImageNet), it is hard or computationally expensive to train a stronger teacher model. We deploy our virtual teacher to teach this powerful student and achieve 0.48% improvement on ImageNet without any extra computation cost. Similarly, when taking a powerful single model ResNeXt29-8×64d with 34.53M parameters as a student model, our self-training implementation achieves more than 1.0% improvement on CIFAR100 (from 81.03% to 82.08%).

Our contributions are summarized as follows:

- By designing two exploratory experiments on teacher models of KD, we observe counterintuitive results, which motivate us to interpreted KD as a regularization method.   
- We then provide theoretical analysis to reveal the relationships between KD and label smoothing regularization.

\- We propose Teacher-free Knowledge Distillation (Tf-KD), which achieves comparable performance with normal knowledge distillation and superior performance to label smoothing regularization on ImageNet-2012.

# 2. Exploratory Experiments and Counterintuitive Observations

To examine the common belief on dark knowledge in KD, we conduct two exploratory experiments:

1) The standard knowledge distillation is to adopt a teacher to teach a weaker student. What if we reverse the operation? Based on the common belief, the teacher should not be improved significantly because the student is too weak to transfer effective knowledge.   
2) If we use a poorly-trained teacher which has much worse performance than the student to teach the student, it is assumed to bring no improvement to the latter. For example, if a poorly-trained teacher with only $10\%$ accuracy is adopted in an image classification task, the student would learn from its soft targets with $90\%$ error, thus the student should not be improved or even suffer worse performance.

We name the “student teach teacher” as Reversed Knowledge Distillation (Re-KD), and the “poorly-trained teacher teach student” as Defective Knowledge Distillation (De-KD) (Fig. 1). We conduct Re-KD and De-KD experiments on CIFAR10, CIFAR100 and Tiny-ImageNet datasets with a variety of neural networks. For fair comparisons, all experiments are conducted with the same settings and hyper-parameters are obtained by grid search from 70 epochs training (200 epochs in total). Detailed implementation and experiment settings are given in Supplementary Material.

# 2.1. Reversed Knowledge Distillation

We conduct Re-KD experiments on the three datasets respectively. CIFAR10 and CIFAR100 [9] contain natural RGB images of 32x32 pixels with 10 and 100 classes, respectively, and Tiny-ImageNet is a subset of ImageNet [3] with 200 classes, where each image is down-sized to 64x64 pixels. For generality of the experiments, we adopt 5-layer plain CNN, MobilenetV2 [15] and ShufflenetV2 [10] as student models and ResNet18, ResNet50 [6], DenseNet121 [8] and ResNeXt29-8×64d as teachers. The results of Re-KD on the three datasets are given in Tabs. 1 to 3.

In Tab. 1, the teacher models are improved significantly by learning from students, especially for teacher models ResNet18 and ResNet50. The two teachers obtain more than 1.1% improvement when taught by MobileNetV2 and

![](images/44e8b0f10e98b61120e9ba553c4bbef49b71d3c084c68d7d76f6cd54b5b9f12a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Teacher"] --> B["Student"]
    style A fill:#f9f,stroke:#333
    style B fill:#bbf,stroke:#333
    subgraph Teacher
        A1["Yellow Node"]
        A2["Yellow Node"]
        A3["Yellow Node"]
        A4["Yellow Node"]
        A5["Yellow Node"]
        A6["Yellow Node"]
        A7["Yellow Node"]
        A8["Yellow Node"]
        A9["Yellow Node"]
        A10["Yellow Node"]
        A11["Yellow Node"]
        A12["Yellow Node"]
        A13["Yellow Node"]
        A14["Yellow Node"]
        A15["Yellow Node"]
        A16["Yellow Node"]
        A17["Yellow Node"]
        A18["Yellow Node"]
        A19["Yellow Node"]
        A20["Yellow Node"]
        A21["Yellow Node"]
        A22["Yellow Node"]
        A23["Yellow Node"]
        A24["Yellow Node"]
        A25["Yellow Node"]
        A26["Yellow Node"]
        A27["Yellow Node"]
        A28["Yellow Node"]
        A29["Yellow Node"]
        A30["Yellow Node"]
        A31["Yellow Node"]
        A32["Yellow Node"]
        A33["Yellow Node"]
        A34["Yellow Node"]
        A35["Yellow Node"]
        A36["Yellow Node"]
        A37["Yellow Node"]
        A38["Yellow Node"]
        A39["Yellow Node"]
        A40["Yellow Node"]
        A41["Yellow Node"]
        A42["Yellow Node"]
        A43["Yellow Node"]
        A44["Yellow Node"]
        A45["Yellow Node"]
        A46["Yellow Node"]
        A47["Yellow Node"]
        A48["Yellow Node"]
        A49["Yellow Node"]
        A50["Yellow Node"]
    end
```
</details>

(a) Normal KD

![](images/be5412d2d9362b44139c6deb6b23a6c00db55e6d597328e31dc0c4c8b82ae2df.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Student"] --> B["Teacher"]
    style A fill:#f9f,stroke:#333
    style B fill:#bbf,stroke:#333
```
</details>

(b) Reversed KD

![](images/193641b7636fac8252809bf824800e55179751bbf0b75f6bce7b5bb8a7859f00.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Poorly-trained Teacher"] --> B["Student"]
    style A fill:#f9f,stroke:#333
    style B fill:#bbf,stroke:#333
```
</details>

(c) Defective KD   
Figure 1. (a) Normal KD framework. (b)(c) Diagrams of exploratory experiments we conduct.

ShuffleNetV2. We can also observe similar results on CIFAR10 and Tiny-ImageNet. When comparing Re-KD (S→T) with Normal KD (T→S), we can see in most cases, Normal KD achieves better results. It should be noted that Re-KD takes the teacher's accuracy as the baseline accuracy, which is much higher than that of Normal KD. However, in some cases, we can find Re-KD outperforms Normal KD. For instance, in Tab. 2 (3rd row), the student model (plain CNN) can only be improved by 0.31% when taught by MobileNetV2, but the teacher (MobileNetV2) can be improved by 0.92% by learning from the student. We have similar observations for ResNeXt29 and ResNet18 (4th row in Tab. 2).

We claim that while the standard knowledge distillation can improve the performance of students on all datasets, the superior teacher can also be enhanced significantly by learning from a weak student, as suggested through the Re-KD experiments.

# 2.2. Defective Knowledge Distillation

We conduct De-KD on CIFAR100 and Tiny-ImageNet. We adopt MobileNetV2 and ShuffleNetV2 as student models and ResNet18, ResNet50 and ResNeXt29 (8×64d) as teacher models. The poorly-trained teachers are trained by 1 epoch (ResNet18) or 50 epochs (ResNet50 and ResNeXt29), with very poor performance. For example, ResNet18 only obtains 15.48% accuracy on CIFAR100 and 9.41% accuracy on Tiny-ImageNet after trained with 1 epoch, and ResNet50 obtains 45.82% and 31.01% on CIFAR100 and Tiny-ImageNet, after trained with 50 epochs (200 epochs in total).

From De-KD experiment results on CIFAR100 in Tab. 4, we observe that the student can be greatly promoted even when distilled by a poorly-trained teacher. For instance, the MobileNetV2 and ShuffleNetV2 can be promoted by 2.27% and 1.48% when taught by the one-epoch-trained ResNet18 with only 15.48% accuracy (2nd row). For poorly-trained ResNeXt29 with 51.94% accuracy (4th row), we find ResNet18 can still be improved by 1.41%, and Mo bileNetV2 obtains 3.14% improvement. From the De-KD experiment results on Tiny-ImageNet in Tab. 4, we find ResNet18 with 9.14% accuracy can still enhance the teacher model MobileNetV2 by 1.16%. Other poorly-trained teachers are all able to enhance the students to some degree.

To better demonstrate the distillation accuracy of a student when taught by poorly-trained teachers with different levels of accuracy, we save 9 checkpoints of ResNet18 and ResNeXt29 in the normal training process. Taking these checkpoints as teacher models to teach MobileNetV2, we observe that MobileNetV2 can always be improved by poorly-trained ResNet18 or poorly-trained ResNeXt29 with different levels of accuracy (Fig. 2). So we can say while a poorly-trained teacher provides much more noisy logits to the student, the student can still be enhanced. The De-KD experiment results are also conflicted with the common belief.

The counterintuitive results of Re-KD and De-KD make us rethink the “dark knowledge” in KD, and we argue that it does not just contain the similarity information. Lacking enough similarity information, a model can still provide “dark knowledge” to enhance other models. To explain this, we make a reasonable assumption and view knowledge distillation as a model regularization, and investigate what is the additional information in the “dark knowledge” of a model. In the next, we will analyze the relationships between knowledge distillation and label smoothing regularization to explain the experimental results of Re-KD and De-KD.

# 3. Knowledge Distillation and Label Smoothing Regularization

We mathematically analyze the relationships between Knowledge Distillation (KD) and Label Smoothing Regularization (LSR), hoping to explain the intriguing results of exploratory experiments in Sec. 2. Given a neural network S to train, we first give loss function of LSR for S. For each training example x, S outputs the probability of each label

Table 1. Normal KD and Re-KD experiment results on CIFAR100. We report mean±std (in %) over 3 runs. The number in parenthesis means increased accuracy over baseline (T: teacher, S: student). 

<table><tr><td>Teacher: baseline</td><td>Student: baseline</td><td>Normal KD (T→S)</td><td>Re-KD (S→T)</td></tr><tr><td rowspan="2">ResNet18: 75.87</td><td>MobileNetV2: 68.38</td><td> $71.05 \pm 0.16 (+2.67)$ </td><td> $77.28 \pm 0.28 (+1.41)$ </td></tr><tr><td>ShuffleNetV2: 70.34</td><td> $72.05 \pm 0.13 (+1.71)$ </td><td> $77.35 \pm 0.32 (+1.48)$ </td></tr><tr><td rowspan="2">ResNet50: 78.16</td><td>MobileNetV2: 68.38</td><td> $71.04 \pm 0.20 (+2.66)$ </td><td> $79.30 \pm 0.11 (+1.14)$ </td></tr><tr><td>ShuffleNetV2: 70.34</td><td> $72.15 \pm 0.18 (+1.81)$ </td><td> $79.43 \pm 0.39 (+1.27)$ </td></tr><tr><td rowspan="2">DenseNet121: 79.04</td><td>MobileNetV2: 68.38</td><td> $71.29 \pm 0.23 (+2.91)$ </td><td> $79.55 \pm 0.11 (+0.51)$ </td></tr><tr><td>ShuffleNetV2: 70.34</td><td> $72.32 \pm 0.25 (+1.98)$ </td><td> $79.83 \pm 0.05 (+0.79)$ </td></tr><tr><td rowspan="2">ResNeXt29: 81.03</td><td>MobileNetV2: 68.38</td><td> $71.65 \pm 0.41 (+3.27)$ </td><td> $81.53 \pm 0.14 (+0.50)$ </td></tr><tr><td>ResNet18: 75.87</td><td> $77.84 \pm 0.15 (+1.97)$ </td><td> $81.62 \pm 0.22 (+0.59)$ </td></tr></table>

Table 2. Re-KD experiment results (accuracy, mean±std over 3 runs in %) on CIFAR10. 

<table><tr><td>Teacher: baseline</td><td>Student: baseline</td><td>Normal KD (T→S)</td><td>Re-KD (S→T)</td></tr><tr><td rowspan="2">ResNet18: 95.12</td><td>Plain CNN: 87.14</td><td> $87.67 \pm 0.17 (+0.53)$ </td><td> $95.33 \pm 0.12 (+0.21)$ </td></tr><tr><td>MobileNetV2: 90.98</td><td> $91.69 \pm 0.14 (+0.71)$ </td><td> $95.71 \pm 0.11 (+0.59)$ </td></tr><tr><td>MobileNetV2: 90.98</td><td>Plain CNN: 87.14</td><td> $87.45 \pm 0.18 (+0.31)$ </td><td> $91.81 \pm 0.23 (+0.92)$ </td></tr><tr><td>ResNeXt29: 95.76</td><td>ResNet18: 95.12</td><td> $95.80 \pm 0.13 (+0.68)$ </td><td> $96.49 \pm 0.15 (+0.73)$ </td></tr></table>

Table 3. Re-KD experiment results (accuracy, in %) on Tiny-ImageNet. 

<table><tr><td>Teacher: baseline</td><td>Student: baseline</td><td>Normal KD (T→S)</td><td>Re-KD (S→T)</td></tr><tr><td rowspan="2">ResNet18: 63.44</td><td>MobileNetV2: 55.06</td><td>56.70 (+1.64)</td><td>64.12 (+0.68)</td></tr><tr><td>ShuffleNetV2: 60.51</td><td>61.19 (+0.68)</td><td>64.35 (+0.91)</td></tr><tr><td rowspan="3">ResNet50: 67.47</td><td>MobileNetV2: 55.06</td><td>56.02 (+0.96)</td><td>67.68 (+0.21)</td></tr><tr><td>ShuffleNetV2: 60.51</td><td>60.79 (+0.28)</td><td>67.62 (+0.15)</td></tr><tr><td>ResNet18: 63.44</td><td>64.23 (+0.79)</td><td>67.89 (+0.42)</td></tr></table>

Table 4. De-KD accuracy (in %) on two datasets. Pt-Teacher is “Poorly-trained Teacher”. Refer to the “Normal KD” in Tabs. 1 to 3 for the accuracy of students taught by “fully-trained teacher”. 

<table><tr><td>Dataset</td><td>Pt-Teacher: baseline</td><td>Student: baseline</td><td>De-KD</td></tr><tr><td rowspan="8">CIFAR100</td><td rowspan="2">ResNet18: 15.48</td><td>MobileNetV2: 68.38</td><td> $70.65 \pm 0.35 (+2.27)$ </td></tr><tr><td>ShuffleNetV2: 70.34</td><td> $71.82 \pm 0.11 (+1.48)$ </td></tr><tr><td rowspan="3">ResNet50: 45.82</td><td>MobileNetV2: 68.38</td><td> $71.45 \pm 0.23 (+3.09)$ </td></tr><tr><td>ShuffleNetV2: 70.34</td><td> $72.11 \pm 0.09 (+1.77)$ </td></tr><tr><td>ResNet18: 75.87</td><td> $77.23 \pm 0.11 (+1.23)$ </td></tr><tr><td rowspan="3">ResNeXt29: 51.94</td><td>MobileNetV2: 68.38</td><td> $71.52 \pm 0.27 (+3.14)$ </td></tr><tr><td>ShuffleNetV2:70.34</td><td> $72.26 \pm 0.36 (+1.92)$ </td></tr><tr><td>ResNet18: 75.87</td><td> $77.28 \pm 0.17 (+1.41)$ </td></tr><tr><td rowspan="4">Tiny-ImageNet</td><td rowspan="2">ResNet18: 9.41</td><td>MobileNetV2: 55.06</td><td> $56.22 (+1.16)$ </td></tr><tr><td>ShuffleNetV2: 60.51</td><td> $60.66 (+0.15)$ </td></tr><tr><td rowspan="2">ResNet50: 31.01</td><td>MobileNetV2:55.06</td><td> $56.02 (+0.96)$ </td></tr><tr><td>ShuffleNetV2: 60.51</td><td> $61.09 (+0.58)$ </td></tr></table>

$k \in \{1...K\} : p(k|x) = softmax(z_k) = \frac{\exp(z_k)}{\sum_{i=1}^{K} \exp(z_i)}$ , where $z_i$ is the logit of the neural network $S$ . The ground truth distribution over the labels is $q(k|x)$ . We write $p(k|x)$ as $p(k)$ and $q(k|x)$ as $q(k)$ for simplicity. The model $S$ can be trained by minimizing the cross-entropy loss: $H(q,p) = -\sum_{k=1}^{K} q(k)\log(p(k))$ . For a single ground-truth label $y$ , the $q(y|x) = 1$ and $q(k|x) = 0$ for all $k \neq y$ .

In LSR, it minimizes the cross-entropy between modified label distribution $q'(k)$ and the network output $p(k)$ , where $q'(k)$ is the smoothed label distribution formulated as

$$
q ^ {\prime} (k) = (1 - \alpha) q (k) + \alpha u (k), \tag {1}
$$

which is a mixture of $q(k)$ and a fixed distribution $u(k)$ , with weight $\alpha$ . Usually, the $u(k)$ is uniform distribution as

![](images/a1e2f296e9a09ab0bcbab02fd080beb19f046a7f6e7e3b1adca0f468255e568f.jpg)

<details>
<summary>line</summary>

MobileNetV2 KD by ResNet18 with different accuracy
| Accuracy of Poorly-trained ResNet18 (%) | KD by ResNet18 | MobileNetV2 baseline |
|---|---|---|
| 15.48 | 70.5 | 68.4 |
| 25.55 | 70.5 | 68.4 |
| 35.61 | 70.9 | 68.4 |
| 45.68 | 70.9 | 68.4 |
| 55.74 | 71.0 | 68.4 |
| 65.81 | 70.5 | 68.4 |
| 75.87 | 71.3 | 68.4 |
</details>

(a) ResNet18

![](images/98d4f4fc771cce100cb227e299cb60e96b67d404319a3c2fc2b6b1fdab04eebe.jpg)

<details>
<summary>line</summary>

MobileNetV2 KD by ResNeXt29 with different accuracy
| Accuracy of Poorly-trained ResNeXt29 (%) | KD by ResNeXt29 | MobileNetV2 baseline |
|---|---|---|
| 14.33 | 70.8 | 68.4 |
| 25.33 | 70.6 | 68.4 |
| 36.33 | 71.4 | 68.4 |
| 47.33 | 71.5 | 68.4 |
| 58.33 | 71.6 | 68.4 |
| 69.33 | 71.7 | 68.4 |
| 80.33 | 71.8 | 68.4 |
</details>

(b) ResNeXt29   
Figure 2. MobileNetV2 taught by ResNet18 and ResNeXt29 with different accuracy on CIFAR100. MobileNetV2 is enhanced by different poorly-trained teachers compared with baseline (the red line). The final point of two blue lines is the result taught by “fully-trained teacher”.

$u(k) = 1/K$ . The cross-entropy loss $H(q', p)$ defined over the smoothed labels is

$$
\begin{array}{l} H (q ^ {\prime}, p) = - \sum_ {k = 1} ^ {K} q ^ {\prime} (k) \log p (k) = (1 - \alpha) H (q, p) + \alpha H (u, p) \\ = (1 - \alpha) H (q, p) + \alpha (D _ {K L} (u, p) + H (u)), \tag {2} \\ \end{array}
$$

where $D_{KL}$ is the Kullback-Leibler divergence (KL divergence) and $H(u)$ denotes the entropy of u and is a constant for the fixed uniform distribution $u(k)$ . Thus, the loss function of label smoothing to model S can be written as

$$
\mathcal {L} _ {L S} = (1 - \alpha) H (q, p) + \alpha D _ {K L} (u, p). \tag {3}
$$

For knowledge distillation, the teacher-student learning mechanism is applied to improve the performance of the student. We assume the student is the model S with output prediction $p(k)$ , and the output prediction of the teacher network is $p_{\tau}^{t}(k) = \text{softmax}(z_{k}^{t}) = \frac{\exp(z_{k}^{t}/\tau)}{\sum_{i=1}^{K} \exp(z_{i}^{t}/\tau)}$ , where $z^{t}$ is the output logits of the teacher network and $\tau$ is the temperature to soften $p^{t}(k)$ (written as $p_{\tau}^{t}(k)$ after softened). The idea behind knowledge distillation is to let the student (the model S) mimic the teacher by minimizing the cross-entropy loss and KL divergence between the predictions of student and teacher as

$$
\mathcal {L} _ {K D} = (1 - \alpha) H (q, p) + \alpha D _ {K L} (p _ {\tau} ^ {t}, p _ {\tau}). \tag {4}
$$

Comparing Eq. (3) and Eq. (4), we find the two loss functions have a similar form. The only difference is that the $p_{\tau}^{t}(k)$ in $D_{KL}(p_{\tau}^{t}, p_{\tau})$ is a distribution from a teacher model and $u(k)$ in $D_{KL}(u, p)$ is the pre-defined uniform distribution. From this view, we can consider KD as a special case of LSR where the smoothing distribution is learned but not pre-defined. On the other hand, if we view the regularization term $D_{KL}(u, p)$ as a virtual teacher model of knowledge distillation, this teacher model will give a uniform probability to all classes, meaning it has a random accuracy (1% accuracy for CIFAR100, 0.1% accuracy for ImageNet).

Since $D_{KL}(p_{\tau}^{t}, p_{\tau}) = H(p_{\tau}^{t}, p_{\tau}) - H(p_{\tau}^{t})$ , where the entropy $H(p_{\tau}^{t})$ is constant for a fixed teacher model, we can reformulate Eq. (4) to

$$
\begin{array}{l} L _ {K D} = (1 - \alpha) H (q, p) + \alpha (D _ {K L} (p _ {\tau} ^ {t}, p _ {\tau}) + H (p _ {\tau} ^ {t})) \\ = (1 - \alpha) H (q, p) + \alpha H (p _ {\tau} ^ {t}, p _ {\tau}). \tag {5} \\ \end{array}
$$

If we set the temperature $\tau = 1$ , we have $L_{KD} = H(\tilde{q}^t, p)$ , where $\tilde{q}^t$ is

$$
\tilde {q} ^ {t} (k) = (1 - \alpha) q (k) + \alpha p ^ {t} (k). \tag {6}
$$

If we compare Eq. (6) with Eq. (1), it is more clearly seen that KD is a special case of LSR. Moreover, the distribution $p^t(k)$ is a learned distribution (from a trained teacher) instead of a uniform distribution $u(k)$ . We visualize the output probability $p^t(k)$ of a teacher and compare it with label smoothing in Supplementary Material, and find with higher temperature $\tau$ , the $p^t(k)$ is more similar to the uniform distribution $u(k)$ of label smoothing.

Based on the comparison of the two loss functions, we summarize the relationships between knowledge distillation and label smoothing regularization as follows:

- Knowledge distillation is a learned label smoothing regularization, which has a similar function with the latter, i.e. regularizing the classifier layer of the model.   
- Label smoothing is an ad-hoc knowledge distillation, which can be revisited as a teacher model with random accuracy and temperature $\tau = 1$ .   
- With higher temperature, the distribution of teacher's soft targets in knowledge distillation is more similar to the uniform distribution of label smoothing.

Therefore, the experiment results of Re-KD and De-KD can be explained as the soft targets of the model in high temperature are closer to a uniform distribution of label smoothing, where the learned soft targets can provide model regularization for the teacher model. That is why a student can enhance the teacher and a poorly-trained teacher can still improve the student model.

# 4. Teacher-free Knowledge Distillation

As we above analyzed, the “dark knowledge” in the teacher model is more of a regularization term than the similarity information between categories. Intuitively, we consider replacing the output distribution of the teacher model with a simple one. We therefore propose a novel Teacher-free Knowledge Distillation (Tf-KD) framework with two implementations. Tf-KD is especially applicable to cases where a stronger teacher model is not available, or only limited computation resources are provided.

The first Tf-KD method is self-training knowledge distillation, denoted as $Tf-KD_{self}$ . As aforementioned, the teacher can be taught by a student and a poorly-trained teacher can also enhance the student. Hence when a stronger teacher model is not available, we propose to deploy “self-training”. It should be noted that the teacher in KD always means a stronger model. We name self-training as a teacher-free method because the model is not a teacher with stronger learning capacity than itself. Our $Tf-KD_{self}$ is similar to Born-again networks [4], but there are two differences. Our motivation (self-training/self-regularization) is different from Born-again networks; and our method uses soft targets of model self as regularization, while Born-again networks utilize an ensemble of student models to train itself iteratively. Specifically, we first train the student model in the normal way to obtain a pre-trained model, which is then used to provide soft label to train itself as in Eq. (4). Formally, given a model S, we denote its pretrained model as $S^{p}$ ; then we try to minimize the KL divergence of the logits between S and $S^{p}$ by $Tf-KD_{self}$ . The loss function of $Tf-KD_{self}$ to train model S is

$$
L _ {s e l f} = (1 - \alpha) H (q, p) + \alpha D _ {K L} (p _ {\tau} ^ {t}, p _ {\tau}), \tag {7}
$$

where $p, p_{\tau}^{t}$ are the output probability of $S$ and $S^{p}$ respectively, $\tau$ is the temperature and $\alpha$ is the weight.

The second implementation of our Tf-KD method is to manually design a teacher with 100% accuracy. In Sec. 3, we reveal LSR is a virtual teacher model with random accuracy. So, if we design a teacher with higher accuracy, we can assume it would bring more improvement to the student. We propose to combine KD and LSR to build a simple teacher model which will output distribution for classes as the following:

$$
p ^ {d} (k) = \left\{ \begin{array}{l l} a & \text { if   } k = c, \\ (1 - a) / (K - 1) & \text { if   } k \neq c, \end{array} \right. \tag {8}
$$

where K is the total number of classes, c is the correct label and a is the correct probability for the correct class. We always set $a \geq 0.9$ , so the probability of a correct class is much higher than that of an incorrect one, and the manually-designed teacher model has 100% accuracy for any dataset.

![](images/6fa56f878187bf604133526e4bdcbb44167e46ef14458553dbe3a66f64b20ebe.jpg)

<details>
<summary>bar</summary>

| class | manually designed teacher | u(k) of label smoothing |
|---|---|---|
| C1 | 0.095 | 0.100 |
| C2 | 0.095 | 0.100 |
| C3 | 0.095 | 0.100 |
| C4 | 0.095 | 0.100 |
| C5 | 0.095 | 0.100 |
| C6 | 0.150 | 0.100 |
| C7 | 0.095 | 0.100 |
| C8 | 0.095 | 0.100 |
| C9 | 0.095 | 0.100 |
| C10 | 0.095 | 0.100 |
</details>

Figure 3. Distribution of manually designed teacher (softened by $\tau = 20$ ) on 10-class dataset. C6 is the correct label. As a comparison, the orange bar is the uniform distribution of LSR.

We name this method as Teacher-free KD by manually-designed regularization, denoted as $Tf-KD_{reg}$ . The loss function is

$$
L _ {r e g} = (1 - \alpha) H (q, p) + \alpha D _ {K L} (p _ {\tau} ^ {d}, p _ {\tau}), \tag {9}
$$

where $\tau$ is the temperature to soften the manually-designed distribution $p^{d}$ (as $p_{\tau}^{d}$ after softening). We set a high temperature $\tau \geq 20$ to make this virtual teacher output a soft probability, in which way it gains the smoothing property as LSR. We visualize the distribution of the manually designed teacher in Fig. 3. As Fig. 3 shows, this manually designed teacher model outputs soft targets with 100% classification accuracy, and also has the smoothing property of label smoothing. But the Tf-KD $_{reg}$ is not an over-parameterized version of LSR because the temperature $\tau \gg 1$ , thus Eq. 9 will not be equal to Eq. 3 when we adjust the parameters $\alpha$ , a or $u(k)$ .

The two Teacher-free methods, Tf-KD $_{self}$ and Tf-KD $_{reg}$ , are very simple yet effective, as validated via extensive experiments in the next section.

# 5. Experiments on Tf-KD

In this section, we conduct experiments to evaluate Tf-KD $_{self}$ and Tf-KD $_{reg}$ on three datasets for image classification: CIFAR100, Tiny-ImageNet and ImageNet. For fair comparisons, all experiments are conducted with the same setting.

# 5.1. Experiments for Self-training

For our Tf-KD $_{self}$ and Normal KD, the hyperparameters (temperature $\tau$ and $\alpha$ ) are obtained by grid search from 70 epochs training (200 epochs), the values of hyper-parameters are given in Supplementary Material.

CIFAR100. On CIFAR100, we use baseline models including MobileNetV2, ShuffleNetV2, GoogLeNet, ResNet18, DenseNet121 and ResNeXt29(8×64d). The baselines are trained for 200 epochs, with batch size 128. The initial learning rate is 0.1 and then divided by 5 at the

Table 5. Accuracy improvement comparison (in %) on CIFAR100 (T: Teacher, R: ResNet, RX: ResNeXt, D: DenseNet). 

<table><tr><td>Model</td><td>Baseline</td><td>Tf-KDself</td><td>Normal KD [T]</td></tr><tr><td>MobileNetV2</td><td>68.38</td><td>70.96 (+2.58)</td><td>+2.67 [R18]</td></tr><tr><td>ShuffleNetV2</td><td>70.34</td><td>72.23 (+1.89)</td><td>+1.71 [R18]</td></tr><tr><td>ResNet18</td><td>75.87</td><td>77.10 (+1.23)</td><td>+1.19 [R50]</td></tr><tr><td>GoogLeNet</td><td>78.72</td><td>80.17 (+1.45)</td><td>+1.39 [RX29]</td></tr><tr><td>DenseNet121</td><td>79.04</td><td>80.26 (+1.22)</td><td>+1.15 [RX29]</td></tr><tr><td>ResNeXt29</td><td>81.03</td><td>82.08 (+1.05)</td><td>+1.12 [RX101]</td></tr></table>

60th, 120th, 160th epoch. We use SGD optimizer with the momentum of 0.9, and weight decay is set to 5e-4.

Tab. 5 shows the test accuracy of the six models. It can be seen that our Tf-KD $_{self}$ consistently outperforms the baselines. For example, as a powerful model with 34.52M parameters, ResNeXt29 improves itself by 1.05% with self-regularization. Even when compared to Normal KD with a superior teacher in Tab. 5 (4th column), our method achieves comparable performance (experiment settings for Tf-KD and Normal KD are the same and hyper-parameters are searched for both Tf-KD $_{self}$ and Normal KD). For example, with ResNet50 to teach ReseNet18, the student has a 1.19% improvement, but our method achieves 1.23% improvement without using any stronger teacher model. We also obtain similar results for MobileNetV2 by Tf-KD $_{self}$ in Fig. 4.

![](images/0498101eac0e43a1e5ee42f7f93d1a6c2fa735c624b2c7c30ebf953b1ac2b1c9.jpg)

<details>
<summary>line</summary>

| epoch | MobileNetV2 Self-training | MobileNetV2 taught by ResNet18 | MobileNetV2 baseline |
|-------|---------------------------|-------------------------------|----------------------|
| 0     | 25                        | 24                            | 26                   |
| 50    | 60                        | 58                            | 59                   |
| 100   | 68                        | 67                            | 66                   |
| 150   | 70                        | 69                            | 68                   |
| 200   | 70                        | 69                            | 68                   |
</details>

Figure 4. MobileNetV2 obtains similar improvement by self-regularization or taught by ResNet18.

Tiny-ImageNet. On Tiny-ImageNet, we use baseline models including MobileNetV2, ShuffleNetV2, ResNet50, DenseNet121. They are trained for 200 epochs with batch size $bn = 128$ for MobileNetV2, ShuffleNetV2 and $bn = 64$ for ResNet50, DenseNet121. The initial learning rate is $\eta = 0.1 * \frac{bn}{128}$ and then divided by 10 at the 60th, 120th, 160th epoch. We use SGD optimizer with momentum of 0.9, and weight decay is set to 5e-4. Tab. 6 shows the results of Tf-KD $_{self}$ on Tiny-ImageNet. It can be seen that Tf-KD $_{self}$ consistently improves the baseline models and achieves comparable improvement with Normal KD.

ImageNet. ImageNet-2012 is one of the largest datasets for object classification, with over 1.3m hand-annotated images. The baseline models we use on this dataset include ResNet18, ResNet50, DenseNet121, RexNeXt101 (32x8d), and we adopt official implementation of Pytorch to train them. We set batch size $bn = 512$ for ResNet18, ResNet50, DenseNet121, and $bn = 256$ for RexNeXt101. Following common experiment settings [5], the initial learning rate is $\eta = 0.1 * \frac{bn}{256}$ which is then divided by 10 at the 30th, 60th, 80th epoch in total 90 epochs. We use SGD optimizer with momentum of 0.9, and weight decay is 1e-4. Results are reported in Tab. 7. We can see that the self-training can further improve the baseline performance on ImageNet-2012. As a comparison, we also use DenseNet121 to teach ResNet18 on ImageNet, and ResNet18 obtains $0.56\%$ improvement, which is comparable with our Tf-KD $_{self}$ (Tab. 8).

Table 6. Tf-KD $_{self}$ experiment results on Tiny-ImageNet (in %). 

<table><tr><td>Model</td><td>Baseline</td><td>Tf-KDself</td><td>Normal KD [T]</td></tr><tr><td>MobileNetV2</td><td>55.06</td><td>56.77 (+1.71)</td><td>+1.64 [R18]</td></tr><tr><td>ShuffleNetV2</td><td>60.51</td><td>61.36 (+0.85)</td><td>+0.68 [R18]</td></tr><tr><td>ResNet50</td><td>67.47</td><td>68.18 (+0.71)</td><td>+0.76 [D121]</td></tr><tr><td>DenseNet121</td><td>68.15</td><td>68.29 (+0.14)</td><td>+0.16 [RX29]</td></tr></table>

Table 7. Tf-KD $_{self}$ experiment results on ImageNet (Top1 accuracy, in %). 

<table><tr><td>Model</td><td>Baseline</td><td>Tf-KDself</td></tr><tr><td>ResNet18</td><td>69.84</td><td>70.42 (+0.58)</td></tr><tr><td>ResNet50</td><td>75.77</td><td>76.41 (+0.64)</td></tr><tr><td>DenseNet121</td><td>75.28</td><td>75.72 (+0.44)</td></tr><tr><td>ResNeXt101</td><td>79.28</td><td>79.56 (+0.28)</td></tr></table>

Table 8. Comparison between Tf-KD $_{self}$ and Normal KD on ImageNet (Top1 accuracy, in %). 

<table><tr><td>Model</td><td>Baseline</td><td>Tf-KDself</td><td>Normal KD [T]</td></tr><tr><td>ResNet18</td><td>69.84</td><td>70.42 (+0.58)</td><td>70.40 (+0.56) [D121]</td></tr></table>

# 5.2. Experiments for Manually-designed Regularization

For all experiments of Tf-KD $_{reg}$ , we adopt the same implementation settings with Tf-KD $_{self}$ , except for using a virtual output distribution as a regularization term (Eq. (9)). For fair comparisons, experiment settings for Normal KD and Tf-KD $_{reg}$ are the same. See Supplementary Material for hyper-parameters of Tf-KD $_{reg}$ .

CIFAR100 and Tiny-ImageNet. For Tf-KD $_{reg}$ experiments on CIFAR100 and Tiny-ImageNet, we set the probability for correct classes as a = 0.99 (Eq. (8)). The temperature $\tau$ and $\alpha$ in Eq. (9) are different for different baseline models (see Supplementary Material). From Tab. 9 and Tab. 10, we can observe with no teacher used and just a regularization term added, Tf-KD $_{reg}$ achieves comparable performance with Normal KD on both CIFAR100 and Tiny-ImageNet.

ImageNet. For the Tf-KD $_{reg}$ on ImageNet, we adopt temperature $\tau = 20$ as normal knowledge distillation, and

Table 9. Tf-KD $_{reg}$ achieves comparable results with Normal KD on CIFAR100. 

<table><tr><td>Model</td><td>Baseline</td><td>Tf-KD $_{reg}$ </td><td>Normal KD [Teacher]</td><td>+ LSR</td></tr><tr><td>MobileNetV2</td><td>68.38</td><td>70.88 (+2.50)</td><td>71.05 (+2.67) [ResNet18]</td><td>69.32 (+0.94)</td></tr><tr><td>ShuffleNetV2</td><td>70.34</td><td>72.09 (+1.75)</td><td>72.05 (+1.71) [ResNet18]</td><td>70.83 (+0.49)</td></tr><tr><td>ResNet18</td><td>75.87</td><td>77.36 (+1.49)</td><td>77.19 (+1.32) [ResNet50]</td><td>77.26 (+1.39)</td></tr><tr><td>GoogLeNet</td><td>78.15</td><td>79.22 (+1.07)</td><td>78.84 (+0.99) [ResNeXt29]</td><td>79.07 (+0.92)</td></tr></table>

Table 10. Tf-KD $_{reg}$ experiment results on Tiny-ImageNet. 

<table><tr><td>Model</td><td>Baseline</td><td>Tf-KD $_{reg}$ </td><td>Normal KD [Teacher]</td><td>+ LSR</td></tr><tr><td>MobileNetV2</td><td>55.06</td><td>56.47 (+1.41)</td><td>56.53 (+1.47) [ResNet18]</td><td>56.24 (+1.18)</td></tr><tr><td>ShuffleNetV2</td><td>60.51</td><td>60.93 (+0.42)</td><td>61.19 (+0.68) [ResNet18]</td><td>60.66 (+0.11)</td></tr><tr><td>ResNet50</td><td>67.47</td><td>67.92 (+0.45)</td><td>68.15 (+0.68) [ResNeXt29]</td><td>67.63 (+0.16)</td></tr><tr><td>DenseNet121</td><td>68.15</td><td>68.37 (+0.18)</td><td>68.44 (+0.26) [ResNeXt29]</td><td>68.19 (+0.04)</td></tr></table>

$\alpha = 0.1$ as label smoothing regularization. The probability for correct classes in the manually-designed teacher is $a = 0.99$ (Eq. (9)). We test our Tf-KD $_{reg}$ with four baseline models: ResNet18, ResNet50, DenseNet121 and ResNeXt101 (32x8d). As a regularization term, the manually designed teacher achieves consistent improvement compared with baselines. For example, the proposed Tf-KD $_{reg}$ improves the top1 accuracy of ResNet50 by $0.65\%$ on ImageNet-2012 (Tab. 11). Even for a huge single model ResNeXt101 (32x8d) with 88.79M parameters, our method achieves $0.48\%$ improvement by using the manually designed teacher.

Table 11. Test accuracy improvement (in %) on ImageNet. 

<table><tr><td>Model</td><td>Baseline</td><td> $+\text{Tf-KD}_{reg}$ </td><td>+ LSR</td></tr><tr><td>ResNet18</td><td>69.84</td><td>70.24 (+0.40)</td><td>70.02 (+0.18)</td></tr><tr><td>ResNet50</td><td>75.77</td><td>76.42 (+0.65)</td><td>76.38 (+0.51)</td></tr><tr><td>DenseNet121</td><td>75.28</td><td>75.62 (+0.34)</td><td>75.24 (-0.04)</td></tr><tr><td>ResNeXt101</td><td>79.28</td><td>79.76 (+0.48)</td><td>79.67 (+0.39)</td></tr></table>

Comparing our two methods Tf-KD $_{self}$ and Tf-KD $_{reg}$ , we observe that Tf-KD $_{self}$ works better in small dataset (CIFAR100) while Tf-KD $_{reg}$ performs slightly better in large dataset (ImageNet).

Comparison with LSR The $Tf-KD_{reg}$ is motivated by LSR, which can be seen as a modification of LSR. This modification significantly improves the performance of neuron networks without extra computation cost. Same as LSR, $Tf-KD_{reg}$ can serve as a generic regularization method to normally train neural networks. We compare our $Tf-KD_{reg}$ with label smoothing on CIFAR100, Tiny-ImageNet and ImageNet. For fair comparisons, experiment settings for $Tf-KD_{reg}$ and LSR are the same. The results are shown in Tab. 9, 10 and 11. It can be seen that $Tf-KD_{reg}$ consistently outperforms LSR. Additionally, the formulation of $KDR_{man}$ is similar to LSR, but it is not an overparameterized version of label smoothing. We give detailed comparison between $Tf-KD_{reg}$ and LSR to show the difference in Supplementary Material.

# 6. Related Work

Knowledge Distillation Since [7] proposed knowledge distillation based on prior work [2], KD has been widely adopted or modified [14, 19, 20, 4, 1, 11, 17]. Different from existing works, our work challenges the common belief of knowledge distillation based on our designed exploratory experiments. A related work is deep mutual learning [21], which proposes to let an ensemble of student models to learn with each other by minimizing the KL Divergence of predictions. Comparatively, our work reveals the relationship between KD and label smoothing, and our proposed Tf-KD can serve as a general method for neural network training. Another related work is Born-again networks [4], which use similar method as Tf-KD $_{self}$ . The difference is that Born-again networks utilize an ensemble of students to train itself in the final step.

Label Smoothing Szegedy et al. [16] proposed LSR to replace the “hard labels” with smoothed labels, boosting performance of many tasks like image classification, language translation and speech recognition [13]. Recently, [12] empirically showed label smoothing can also help improve model calibration. In our work, we adopt label smoothing regularization to understand the regularization function of knowledge distillation.

# 7. Conclusion

In this work, we find through experiments and analyses that the “dark knowledge” of a teacher model is more of a regularization term than similarity information of categories. Based on the relationship between KD and LSR, we propose Teacher-free KD. Experiment results show our Tf-KD can achieve comparable results with Normal KD in image classification. Our work also suggests that, when it is hard to find a stronger teacher for a powerful model or computation resource is limited to train teacher models, the targeted model can still get enhanced by self-training or a manually-designed regularization term.

Acknowledgement Jiashi Feng was partially supported by AI.SG R-263-000-D97-490, NUS ECRA R-263-000-C87-133 and MOE Tier-II R-263-000-D17-112. Besides, we thank Dr.Jianan Li, Mr.Daquan Zhou and Mr.Yujun Shi for discussion during this work.

# References

[1] R. Anil, G. Pereyra, A. Passos, R. Ormandi, G. E. Dahl, and G. E. Hinton. Large scale distributed neural network training through online distillation. arXiv preprint arXiv:1804.03235, 2018.   
[2] J. Ba and R. Caruana. Do deep nets really need to be deep? In Advances in neural information processing systems, pages 2654-2662, 2014.   
[3] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.   
[4] T. Furlanello, Z. C. Lipton, M. Tschannen, L. Itti, and A. Anandkumar. Born again neural networks. arXiv preprint arXiv:1805.04770, 2018.   
[5] P. Goyal, P. Dollár, R. Girshick, P. Noordhuis, L. Wesolowski, A. Kyrola, A. Tulloch, Y. Jia, and K. He. Accurate, large minibatch sgd: Training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.   
[6] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.   
[7] G. Hinton, O. Vinyals, and J. Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.   
[8] G. Huang, Z. Liu, L. Van Der Maaten, and K. Q. Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4700–4708, 2017.   
[9] A. Krizhevsky, G. Hinton, et al. Learning multiple layers of features from tiny images. Technical report, Citeseer, 2009.   
[10] N. Ma, X. Zhang, H.-T. Zheng, and J. Sun. Shufflenet v2: Practical guidelines for efficient cnn architecture design. In Proceedings of the European Conference on Computer Vision (ECCV), pages 116–131, 2018.   
[11] S.-I. Mirzadeh, M. Farajtabar, A. Li, and H. Ghasemzadeh. Improved knowledge distillation via teacher assistant: Bridging the gap between student and teacher. arXiv preprint arXiv:1902.03393, 2019.   
[12] R. Müller, S. Kornblith, and G. Hinton. When does label smoothing help? arXiv preprint arXiv:1906.02629, 2019.   
[13] G. Pereyra, G. Tucker, J. Chorowski, L. Kaiser, and G. Hinton. Regularizing neural networks by penalizing confident output distributions. arXiv preprint arXiv:1701.06548, 2017.   
[14] A. Romero, N. Ballas, S. E. Kahou, A. Chassang, C. Gatta, and Y. Bengio. Fitnets: Hints for thin deep nets. arXiv preprint arXiv:1412.6550, 2014.   
[15] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4510–4520, 2018.   
[16] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2818–2826, 2016.   
[17] T. Wang, L. Yuan, X. Zhang, and J. Feng. Distilling object detectors with fine-grained feature imitation. In The

IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.   
[18] S. Xie, R. Girshick, P. Dollár, Z. Tu, and K. He. Aggregated residual transformations for deep neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1492–1500, 2017.   
[19] J. Yim, D. Joo, J. Bae, and J. Kim. A gift from knowledge distillation: Fast optimization, network minimization and transfer learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4133-4141, 2017.   
[20] R. Yu, A. Li, V. I. Morariu, and L. S. Davis. Visual relationship detection with internal and external linguistic knowledge distillation. In Proceedings of the IEEE International Conference on Computer Vision, pages 1974–1982, 2017.   
[21] Y. Zhang, T. Xiang, T. M. Hospedales, and H. Lu. Deep mutual learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4320-4328, 2018.