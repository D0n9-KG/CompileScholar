# What Knowledge Gets Distilled in Knowledge Distillation?

Utkarsh Ojha $^{*}$ Yuheng Li $^{*}$ Anirudh Sundara Rajan $^{*}$

Yingyu Liang Yong Jae Lee

University of Wisconsin-Madison

# Abstract

Knowledge distillation aims to transfer useful information from a teacher network to a student network, with the primary goal of improving the student's performance for the task at hand. Over the years, there has a been a deluge of novel techniques and use cases of knowledge distillation. Yet, despite the various improvements, there seems to be a glaring gap in the community's fundamental understanding of the process. Specifically, what is the knowledge that gets distilled in knowledge distillation? In other words, in what ways does the student become similar to the teacher? Does it start to localize objects in the same way? Does it get fooled by the same adversarial samples? Does its data invariance properties become similar? Our work presents a comprehensive study to try to answer these questions. We show that existing methods can indeed indirectly distill these properties beyond improving task performance. We further study why knowledge distillation might work this way, and show that our findings have practical implications as well.

# 1 Introduction

Knowledge distillation, first introduced in $[2, 13]$ , is a procedure of training neural networks in which the ‘knowledge’ of a teacher is transferred to a student. The thesis is that such a transfer (i) is possible and (ii) can help the student learn additional useful representations. The seminal work by $[13]$ demonstrated its effectiveness by making the student imitate the teacher’s class probability outputs for an image. This ushered in an era of knowledge distillation algorithms, including those that try to mimic intermediate features of the teacher $[29, 37, 14]$ , or preserve the relationship between samples as modeled by the teacher $[25, 36]$ , among others.

While thousands of papers have been published on different techniques and ways of using knowledge distillation, there appears to be a gap in our fundamental understanding of it. Yes, it is well-known that the student's performance on the task at hand can be improved with the help of a teacher. But what exactly is the so-called knowledge that gets distilled during the knowledge distillation process? For example, does distillation make the student look at similar regions as the teacher when classifying images? If one crafts an adversarial image to fool the teacher, is the student also more prone to getting fooled by it? If the teacher is invariant to a certain change in data, is that invariance also transferred to the student? Such questions have not been thoroughly answered in the existing literature.

This has become particularly relevant because there have been studies which present some surprising findings about the distillation process. [3] showed that performing knowledge distillation with a bigger teacher does not necessarily improve the student's performance over that with a smaller teacher, and thus raised questions about the effectiveness of the distillation procedure in such cases. [32] showed that the agreement between the teacher and distilled student's predictions on test images is

![](images/35e01338946685a0d16bc5ed0ea3332cd432d022a4bcceb5bcb3bb6b91558488.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Teacher"] -->|x| B["Student"]
    B -->|KL| C["Output"]
    style A fill:#ccc,stroke:#333
    style B fill:#ccc,stroke:#333
    style C fill:#fff,stroke:#333
```
</details>

![](images/45be1551d86d6353070d13b82bc71cc841ab78537776fa5dfed13c287bb8b213.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Teacher"] -->|f_t^(l)| B["Student"]
    B -->|f_s^(l)| C["x"]
    style A fill:#ccc,stroke:#333
    style B fill:#ccc,stroke:#333
    style C fill:#ccc,stroke:#333
    note right of B: ≈
    note left of C: Hint
```
</details>

![](images/d728f7d0d39afba8fd072b209839ca45cf8be3cfcaff79c480efb76ae381ae4f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Teacher"] -->|x| B["Student"]
    B -->|y| C["class(x) ≠ class(y)"]
    A -->|t(x)| D["CRD"]
    B -->|s(x)| E["CRD"]
    D <-->|≈| F["CRD"]
    E <-->|≠| G["CRD"]
```
</details>

Figure 1: Methods used in this work. (i) KL [13]: mimicking of class probabilities. (ii) Hint [29]: mimicking of features at an intermediate layer. (iii) CRD [34]: features from the student and teacher for the same image constitute a positive pair, and those from different classes make up a negative pair.

not that different to the agreement of those between the teacher and an independently trained student, raising further doubts about how knowledge distillation works, if it works at all.

In this work, we present a comprehensive study tackling the above questions. We analyze three popular knowledge distillation methods $[13, 29, 34]$ . Many of our findings are quite surprising. For example, by simply mimicking the teacher's output using the method of $[13]$ , the student can inherit many implicit properties of the teacher. It can gain the adversarial vulnerability that the teacher has. If the teacher is invariant to color, the student also improves its invariance to color. To understand why these properties get transferred without an explicit objective to do so, we study the distillation process through a geometric lens, where we think about the features from a teacher as relative positions of an instance (i.e., distances) from its decision boundary. Mimicking those features, we posit, can therefore help the student inherit the decision boundary and (consequently) the implicit properties of the teacher. We show that these findings have practical implications; e.g., an otherwise fair student can inherit biases from an unfair teacher. Hence, by shedding some light on the ‘dark knowledge’ $[12]$ , our goal is to dissect the distillation process better.

# 2 Related work

Model compression [2] first introduced the idea of knowledge distillation by compressing an ensemble of models into a smaller network. [13] took the concept forward for modern deep learning by training the student to mimic the teacher's output probabilities. Some works train the student to be similar to the teacher in the intermediate feature spaces [29, 37]. Others train the student to mimic the relationship between samples produced by the teacher [25, 35, 26], so that if two samples are close/far in the teacher's representation, they remain close/far in the student's representation. Contrastive learning has recently been shown to be an effective distillation objective in [34]. More recently, [1] present practical tips for performing knowledge distillation; e.g., providing the same view of the input to both the teacher and the student, and training the student long enough through distillation. Finally, the benefits of knowledge distillation have been observed even if the teacher and the student have the same architecture [5, 39]. For a more thorough survey of knowledge distillation, see [11]. In this work, we choose to study three state-of-the-art methods, each representative of the output-based, feature-based, and contrastive-based families of distillation approaches.

There have been a few papers that present some surprising results. [3] shows that a smaller & less accurate teacher is often times better than a bigger & more accurate teacher in increasing the distilled student's performance. More recent work shows that the agreement between the predictions of a teacher and student is not necessarily much higher than that between the teacher and an independent student [32]. There has been work done which tries to explain why distillation improves student's performance by trying linking it to the regularizing effect of soft labels [21, 40] or what the key ingredients are which help in student's optimization process [27, 15]. What we seek to understand in this work is different: we study different ways (beyond performance improvement) in which a student becomes similar to the teacher by inheriting its implicit properties.

# 3 Distillation methods studied

To ensure that our findings are general and cover a range of distillation techniques, we select standard methods representative of three families of distillation techniques: output-based [13], feature-based [29], and contrastive-based [34]. The objectives of these methods, described below, are combined with the cross entropy loss $\mathcal{L}_{\mathrm{CLS}}(\mathbf{z}_{\mathbf{s}},\mathbf{y}):=-\sum_{j=1}^{c}y_{j}\log\sigma_{j}(\mathbf{z}_{\mathbf{s}})$ , where y is the

![](images/01021eda9eef658077d0060a58b0992d16539453436e0afee469601f3c749c69.jpg)

<details>
<summary>natural_image</summary>

Illustration of a bird in flight with blue wings flying above two smaller birds, set against a colorful abstract background (no text or symbols)
</details>

Independent student

![](images/42697ac4cd563b1a40c6a2f1f77cb909624d44e4e3d723804d660c5e00a0bccb.jpg)

<details>
<summary>natural_image</summary>

Illustration of a bird in flight over a colorful, textured surface with a central blue-green circular pattern (no text or symbols)
</details>

Teacher

![](images/87377d21359e4d05d65f5f0e33bc5172a62cded16328d942603d3477088e9db2.jpg)

<details>
<summary>natural_image</summary>

Abstract artistic composition with blue and yellow hues, no text or symbols present
</details>

Distilled student

![](images/06c1e56122d43245fb3c8b87425d869a9f4cbada9854fb35dece9a974ccb2560.jpg)

<details>
<summary>bar_stacked</summary>

| Category | Value |
|---|---|
| Top 1 | 50.3 |
| Top 2 | 67.2 |
| Bottom 1 | 52.9 |
| Bottom 2 | 55.6 |
</details>

ResNet50 $\rightarrow$ ResNet18

![](images/fd2b826aedf877fd45b023ea536205a7f98763145dd076ca0526e2f88bdf47d0.jpg)

<details>
<summary>bar_stacked</summary>

| Category | Value |
|---|---|
| Top 1 | 50.2 |
| Top 2 | 59.2 |
| Bottom 1 | 52.9 |
| Bottom 2 | 67.6 |
</details>

VGG19 $\rightarrow$ VGG11   
Figure 2: Left: An example of how the distilled student can focus on similar regions as the teacher while classifying an image. Right: % where teacher's CAM is more similar to the distilled student's CAM than to the independent student's CAM. The red line indicates chance performance (50%).

ground-truth one-hot label vector, $\mathbf{z}_{\mathbf{s}}$ is the student's logit output, $\sigma_{j}(\mathbf{z}) = \exp (z_{j}) / \sum_{i}\exp (z_{i})$ is the softmax function, and $c$ is the number of classes.

(1) KL: [13] proposed to use the soft labels produced by the teacher as an additional target for the student to match, apart from the (hard) ground-truth labels. This is done by minimizing the KL-divergence between the predictive distributions of the student and the teacher:

$$
\mathcal {L} _ {\mathrm{KL}} \left(\mathbf {z} _ {\mathbf {s}}, \mathbf {z} _ {\mathbf {t}}\right) := - \tau^ {2} \sum_ {j = 1} ^ {c} \sigma_ {j} \left(\frac {\mathbf {z} _ {\mathbf {t}}}{\tau}\right) \log \sigma_ {j} \left(\frac {\mathbf {z} _ {\mathbf {s}}}{\tau}\right), \tag {1}
$$

where $z_{t}$ is the logit output of the teacher, and $\tau$ is a scaling temperature. The overall loss function is $\gamma\mathcal{L}_{\mathrm{CLS}} + \alpha\mathcal{L}_{\mathrm{KL}}$ , where $\gamma$ and $\alpha$ are balancing parameters. We refer to this method as KL.

(2) Hint: FitNets [29] makes the student's intermediate features $(\mathbf{f}_{\mathrm{s}})$ mimic those of the teacher's $(\mathbf{f}_{\mathrm{t}})$ for an image $x$ , at some layer $l$ . It first maps the student's features (with additional parameters $r$ ) to match the dimensions of the teacher's features, and then minimizes their mean-squared error:

$$
\mathcal {L} _ {\text { Hint }} (\mathbf {f} _ {\mathbf {s}} ^ {(1)}, \mathbf {f} _ {\mathbf {t}} ^ {(1)}) = \frac {1}{2} | | \mathbf {f} _ {\mathbf {t}} ^ {(1)} - r (\mathbf {f} _ {\mathbf {s}} ^ {(1)}) | | ^ {2} \tag {2}
$$

The overall loss is $\gamma \mathcal{L}_{\mathrm{CLS}} + \beta \mathcal{L}_{\mathrm{Hint}}$ , where $\gamma$ and $\beta$ are balancing parameters. [29] termed the teacher's intermediate representation as Hint, and we adopt this name.

(3) CRD: Contrastive representation distillation [34] proposed the following. Let $s(x)$ and $t(x)$ be the student's and teacher's penultimate feature representation for an image $x$ . If $x$ and $y$ are from different categories, then $s(x)$ and $t(x)$ should be similar (positive pair), and $s(x)$ and $t(y)$ should be dissimilar (negative pair). A key for better performance is drawing a large number of negative samples $N$ for each image, which is done using a contantly updated memory bank.

$$
\mathcal {L} _ {\mathrm{CRD}} = - \log h (s (x), t (x)) - \sum_ {j = 1} ^ {N} \log (1 - h (s (x), t (y _ {j}))) \tag {3}
$$

where $h(a,b)=(e^{a\cdot b/\tau})/(e^{a\cdot b/\tau}+\frac{N}{M})$ , M is the size of the training data, $\tau$ is a scaling temperature, and $\cdot$ is the dot product. We use CRD to refer to this method. All other implementation details (e.g., temperature for KL, layer index for Hint) can be found in appendix.

# 4 Experiments

We now discuss our experimental design. To reach conclusions that are generalizable across different architectures and datasets, and robust across independent runs, we experiment with a variety of teacher-student architectures, and tune the hyperparameters so that the distillation objective improves the test performance of the student compared to independent training. For each setting, we average the results over two independent runs. We report the top-1 accuracy of all the models in the appendix. The notation $Net_{1} \rightarrow Net_{2}$ indicates distilling the knowledge of $Net_{1}$ (teacher) into $Net_{2}$ (student).

# 4.1 Does localization knowledge get distilled?

We start by studying whether the localization properties of a teacher transfers to a student through knowledge distillation. For example, suppose that the teacher classifies an image as a cat by focusing

![](images/e639c08a7fbd2870fdd9a4546eff859a1e78c82aa07c4d2efd8a70f1cc1aa7ee.jpg)

<details>
<summary>bar</summary>

| Category | Fooling rate (%) |
| -------- | ---------------- |
| Ind      | 44.2             |
| KL       | 52               |
| Hint     | 48.3             |
| CRD      | 50.5             |
</details>

(a) ResNet50 → ResNet18

![](images/f5cdafe404a55dbd77cd983c162e89b079691d3e324fdd86d7438e3df85e2afa.jpg)

<details>
<summary>bar</summary>

| Category | Value |
|---|---|
| Ind | 62.3 |
| KL | 69.7 |
| Hint | 79.8 |
| CRD | 70.5 |
</details>

(b) VGG19 → VGG11

![](images/5dc96bf2fed56c0ff2b22906f1926641f4b695abfe204bc2d862b9b835ee601f.jpg)

<details>
<summary>bar</summary>

| Category | Value |
|---|---|
| Ind | 69 |
| KL | 70.5 |
| Hint | 70.7 |
| CRD | 70.6 |
</details>

(c) VGG19 → VGG11 (R18)

![](images/8a01106eff19d2deb28ca9a464e2186d90607c24562b558e64c6b3fca966d8cf.jpg)

<details>
<summary>bar</summary>

| Category | Value |
|---|---|
| Ind | 36.2 |
| KL | 43 |
| Hint | 47.7 |
| CRD | 49 |
</details>

(d) VGG19 → ResNet18

![](images/8cce51ccf60bd9761e856c6952364bbc9b1e627240708b7196b858a559416666.jpg)

<details>
<summary>bar</summary>

| Category | Value |
|---|---|
| Ind | 21.9 |
| KL | 22.3 |
| Hint | 23.5 |
| CRD | 24 |
</details>

(e) ViT → ResNet18   
Figure 3: The images which fool the teacher fool the distilled student more than the independent student in (a), (b) and (d), but not in (e). If the adversarial attack is generated using a foreign network (ResNet18, (c)), it fails to convincingly fool the distilled student more than the independent one.

on its face, and an independent student classifies it as a cat by focusing on its fur. After knowledge distillation, will the student focus more on the face when classifying the image as a cat?

Experimental setup: We train three models for ImageNet classification: a teacher, a distilled and an independent student. For each network, we obtain the class activation map (CAM) of the ground-truth class for each of random 5000 test images, using Grad-CAM [30], which visualizes the pixels that a classifier focuses on for a given class. We then compute how often (in terms of % of images) the teacher's CAM is more similar (using cosine similarity) to the distilled student's CAM than to the independent student's CAM. A score of $>50\%$ means that, on average, the distilled student's CAMs become more similar to those of the teacher than without distillation. As a sanity check, we also compute the same metric between the teacher's CAM and the CAMs of two independent students trained with different seeds (Ind), which should be equally distant from the teacher (score of $\sim 50\%$ ).

Results: Fig. 2 (right) shows the results for two configurations of teacher-student architectures: (i) ResNet50 → ResNet18 and (ii) VGG19 → VGG11. For KL and CRD, we observe a significant increase in similarity compared to random chance (as achieved by the Ind baseline). Hint also shows a consistent increase, although the magnitude is not as large.

Discussion: This experiment shows that having access to the teacher's class-probabilities, i.e. confidence for the ground-truth class, can give information on where the teacher is focusing on while making the classification decision. This corroborates, to some degree, the result obtained in the Grad-CAM paper [30], which showed that if the network is very confident of the presence of an object in an image, it focuses on a particular region (Fig. 1(c) in [30]), and when it is much less confident, it focuses on some other region (Fig. 7(d) in [30]). Fig. 2 (left) shows a sample test image and the corresponding CAMs produced by the independent student (left), teacher (middle), and distilled student (right) for the ground-truth class. The distilled student looks at similar regions as the teacher, and moves away from the regions it was looking at initially (independent student). So, regardless of the correctness of any network's CAM, our analysis shows that a distilled student's CAM does become similar to those of the teacher for all three distillation techniques, albeit with varying degrees.

# 4.2 Does adversarial vulnerability get distilled?

Next, we study the transfer of a different kind of property. If we design an adversarial image to fool the teacher, then will that same image fool the distilled student more than the independent student?

Experimental setup: We train a teacher, a distilled student, and an independent student for ImageNet classification. Given 5000 random test images, we convert each image I into an adversarial image $I^{adv}$ using iterative FGSM [10, 19] (see appendix for details), whose goal is to fool the teacher, so that teacher's prediction for $I^{adv}$ changes from its original prediction for I. Fooling rate is then defined as the fraction of adversarial images which succeed at this task. In our experiments, we use only this fraction of adversarial images which fool the teacher, and apply them to different students.

Results: We evaluate four configurations: (i) ResNet50 → ResNet18, (ii) VGG19 → VGG11, (iii) VGG19 → ResNet18, (iv) ViT (ViT-b-32) [4] → ResNet18. The fooling rate is \~85% for all the teachers. Fig. 3 shows the fooling rate (y-axis) when applying these successful adversarial images to different types of students (x-axis). We see that for ResNet50 → ResNet18, the ability to fool the independent student drops to 44.2%, which is expected since the adversarial images aren't designed for that student. In distilled students, we see an increase in the fooling rate relative to the independent one across all distillation methods (48%-52%). The trend holds for VGG19 → VGG11 and VGG19

![](images/0f15e108d6b413fd1b73b5e013b2256688b5ff2d75c42acd90f9501ced022843.jpg)

<details>
<summary>bar</summary>

| Category | Agreement (%) |
| -------- | ------------- |
| Ind      | 71.3          |
| KL       | 82.1          |
| KL*      | 74            |
| Hint     | 72.2          |
| Hint*CRD | 72.4          |
| CRD*     | 79.5          |
| T        | 71.3          |
| T*       | 92.5          |
| T*       | 78            |
</details>

![](images/d95442ea3ae54b28adb9dee9efe15034356b116c28a1d71aaf2e2a14865270d1.jpg)

<details>
<summary>line</summary>

| R50→R18 | CRD  | Hint | Ind  | KL   | T    |
| ------- | ---- | ---- | ---- | ---- | ---- |
| 0.3     | 72   | 62   | 60   | 76   | 90   |
| 0.4     | 68   | 58   | 56   | 72   | 86   |
| 0.5     | 62   | 52   | 50   | 66   | 82   |
| 0.6     | 50   | 42   | 40   | 54   | 78   |
</details>

![](images/16a225eab6dfe8e296794ccfb713db4e86e46455039f063a0017b3a868a4db26.jpg)

<details>
<summary>line</summary>

| Swin-T → R18 | CRD  | Hint | Ind  | KL   | T    |
| ------------ | ---- | ---- | ---- | ---- | ---- |
| 0.3          | 62   | 60   | 62   | 62   | 80   |
| 0.4          | 55   | 52   | 55   | 55   | 78   |
| 0.5          | 48   | 48   | 48   | 48   | 75   |
| 0.6          | 40   | 40   | 40   | 40   | 73   |
</details>

![](images/1e0bc921dae3a215ecae0c18112d2909c44a372cadde8502f70301723a655253.jpg)

<details>
<summary>line</summary>

| color strength | T    | Ind  | Hint-F2 | Hint-F3 | Hint-F4 | Hint-F5 | Hint-F6 | Hint-F7 |
| -------------- | ---- | ---- | ------- | ------- | ------- | ------- | ------- | ------- |
| 0.3            | 90   | 60   | 65      | 62      | 60      | 68      | 65      | 80      |
| 0.4            | 88   | 55   | 62      | 58      | 55      | 65      | 62      | 75      |
| 0.5            | 85   | 50   | 58      | 55      | 52      | 62      | 58      | 70      |
| 0.6            | 80   | 45   | 55      | 52      | 48      | 58      | 55      | 65      |
</details>

Figure 4: (a) Agreement between two images with different color properties. \* indicates distillation done by a teacher T\* not trained to be color invariant. (b, c) Agreement between two images having increasingly different color properties. (d) Effect of different layers used in Hint distillation.

$\rightarrow$ ResNet18; Fig. 3 (b, d). When distillation is done from a transformer to a CNN (ViT $\rightarrow$ ResNet18) the fooling rates remain similar for the independent and distilled students; Fig. 3 (e). Here, we don't see the student becoming similar to the teacher to the extent observed for a CNN $\rightarrow$ CNN distillation.

Discussion: This result is surprising. Iterative FGSM is a white box attack, which means that it has full access to the target model (teacher), including its weights and gradients. Thus, the adversarial examples are specifically crafted to fool the teacher. In contrast, the student never has direct access to the weights and gradients of the teacher, regardless of the distillation method. Yet, by simply trying to mimic the teacher's soft probabilities $(KL)$ or an intermediate feature layer (Hint, CRD), the student network inherits, to some extent, the particular way that a teacher is fooled.

We conduct an additional study to ensure that the reason the distilled student is getting fooled more is because it is being attacked specifically by its teacher's adversarial images; i.e., if those images are designed for some other network, would the difference in fooling rates still be high? We test this in the VGG19 → VGG11 setting. This time we generate adversarial images ( $I \rightarrow I^{adv}$ ) to fool an ImageNet pre-trained ResNet18 network, instead of VGG19 (the teacher). We then use those images to attack the same VGG11 students from the VGG19 → VGG11 setting. In Fig. 3 (c), we see that the fooling rates for the independent and distilled students remain similar. This indicates that distillation itself does not make the student more vulnerable to any adversarial attack, and instead, an increase in fooling rate can be attributed to the student's inheritance of the teacher's adversarial vulnerability.

# 4.3 Does invariance to data transformations get distilled?

We have studied whether the response properties on single images get transferred from the teacher to the student. Now suppose that the teacher is invariant to certain changes in data, either learned explicitly through data augmentation or implicitly due to architectural choices. Can such properties about changes in images get transferred during distillation?

Experimental setup: We study color invariance as a transferable property. We train three models for ImageNet classification: a teacher, a distilled and an independent student. While training the teacher, we add color jitter in addition to the standard augmentations (random crops, horizontal flips). Specifically, we alter an image's brightness, contrast, saturation, and hue, with magnitudes sampled uniformly in [0, 0.4] for the first three, and in [0, 0.2] for hue. This way, the teacher gets to see the same image with different color properties during training and can become color invariant. When training the student, we only use the standard augmentations without color jittering, and see whether such a distilled student can indirectly inherit the color invariance property through the teacher.

Results: We start with the ResNet50 → ResNet18 configuration. After training, we evaluate the models on 50k ImageNet validation images. For each image X, we construct its augmented version $X'$ by altering the color properties in the same way as done while training the teacher. Fig. 4 (a) depicts the agreement scores between X and $X'$ (y-axis) for different models (x-axis). The teacher (T), being trained to have that property, achieves a high score of 92.5%. The independent student, which is not trained to be color invariant, has a much lower score of 71.3%. But, when the color-invariant teacher is used to perform distillation using KL or CRD, the agreement scores of the students jump up to 82.1% and 79.5% respectively. To ensure that this increase is not due to some regularizing property of the distillation methods that has nothing to do with the teacher, we repeat the experiment, this time with a ResNet50 teacher (T\*) that is not trained with color augmentation. The new agreement scores for the distilled students (marked by \*; e.g., KL\*) drop considerably

![](images/3d606348be06bc8e017459b0a2280bde8c473735c325f10255423cc24db949c9.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Teacher"] -->|KD: natural images| B["Student"]
    B --> C["Similar on unseen domains?"]
    style A fill:#f9f,stroke:#333
    style B fill:#bbf,stroke:#333
    style C fill:#dfd,stroke:#333
```
</details>

![](images/e7dd11570e15b8d462150b1c660730170ab489412ee4cd3de5310f4998b9b65f.jpg)

<details>
<summary>bar</summary>

|        | Ind  | KL   | Hint | CRD  |
| ------ | ---- | ---- | ---- | ---- |
| sketch | 33   | 37   | 36   | 40   |
| stylized | 22   | 29   | 28   | 31   |
| silhouette | 26   | 31   | 29   | 34   |
| edge   | 13   | 31   | 10   | 37   |
| cue conflict | 23   | 29   | 28   | 31   |
</details>

(a) VGG19 → ResNet18

![](images/1750c3ec4fffa4e15c70a7c33c3b7d62f302d84e14d197f0c1434592fc01ff29.jpg)

<details>
<summary>bar</summary>

| Category | Ind | KL | Hint | CRD |
| :--- | :--- | :--- | :--- | :--- |
| sketch | 51 | 57 | 52 | 54 |
| stylized | 33 | 38 | 35 | 34 |
| silhouette | 50 | 58 | 54 | 50 |
| edge | 22 | 27 | 26 | 25 |
| cue conflict | 37 | 42 | 39 | 38 |
</details>

(b) Swin-Base $\rightarrow$ Swin-Tiny   
Figure 5: Left: Does knowledge transferred about one domain give knowledge about other unseen domains? Right: Consensus scores between teacher and student for images from unseen domains.

compared to the previous case. This is a strong indication that the student does inherit teacher-specific invariance properties during distillation.

In Fig. 4(b), we show that this trend in agreement scores (y-axis) holds even when the magnitude of change in brightness, contrast and saturation (x-axis) is increased to create $X'$ from $X$ . We repeat this experiment for Swin-tiny (a transformer) [22] → ResNet18 in Fig. 4(c). We again see some improvements in the agreement scores of the distilled students. This increase, however, is not as significant as that observed in the previous CNN → CNN setting. Throughout this work, we find that distilling the properties from a transformer teacher into a CNN student is difficult. The implicit biases introduced due to architectural differences between the teacher (transformer) and student (CNN) seem too big, as was studied in [28], to be overcome by current distillation methods.

We also note the ineffectiveness of Hint in distilling this knowledge. Our guess for this is the choice of l for distillation, which is typically set to be in the middle of the network as opposed to deeper layers where KL and CRD operate. So, we perform an ablation study for Hint, where the student mimics a different feature of the teacher each time; starting from a coarse feature of resolution $56 \times 56$ (F1) to the output scores (logits) of the teacher network (F7). We plot the agreement scores in Fig. 4(d), where we see that the score increases as we choose deeper features. Mimicking the deeper layers likely constrains the overall student more compared to a middle layer, since in the latter case, the rest of the student (beyond middle layer) could still function differently compared to the teacher.

Discussion: In sum, color invariance can be transferred during distillation. This is quite surprising since the teacher is not used in a way which would expose any of its invariance properties to the student. Remember that all the student has to do during distillation is to match the teacher's features for an image X; i.e., the student does not get to see its color augmented version, $X'$ , during training. If so, then why should it get to know how the teacher would have responded to $X'$ ?

# 4.4 Does knowledge about unseen domains get distilled?

So far, our analysis has revolved around the original ImageNet dataset, something that was used to perform the distillation itself. So, does the teacher only transfer its knowledge pertaining to this domain, or also of domains it has never seen (Fig. 5 left)?

Experimental setup: To test this, we first perform distillation using two settings (i) VGG19 → ResNet18 and (ii) Swin-Base $[22]$ → Swin-Tiny, where the training of the teacher, as well as distillation is done on ImageNet. During test time, we take an image from an unseen domain and see how frequently the student's and teacher's class predictions match for it (regardless of whether the predicted label is correct or incorrect), which we call the consensus score. For the unseen domains, we consider the five datasets proposed in $[7]$ : sketch, stylized, edge, silhouette, cue conflict. Images from these domains are originally from ImageNet, but have had their properties modified.

Results: Fig. 5 (right) shows the consensus scores between the teacher and the student (y-axis). We see a nearly consistent increase in consensus brought about by the distillation methods in both settings. The extent of this increase, however, differs among the domains. There are cases, for example, in VGG19 → ResNet18, where consensus over images from edge domain increases over 100% (12% → ≥30%) by some distillation methods. On the other hand, the increase can also be relatively modest in certain cases: e.g. Swin-Base → Swin-Tiny in stylized domain.

Discussion: [32] showed that the agreement in classification between the teacher and distilled student is not much different than that to an independent student on CIFAR. We find this to be true in our ImageNet classification experiments as well. That is why the increase in agreement observed for

![](images/7372f04d832dcc96848959f0d2f7b6519b8d36c4f461d3d6e8578f781f50755f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Teacher"] -->|Knowledge on three domains| B["Student"]
    B -->|Distilled on one domain| A
```
</details>

![](images/17fbbd969d91485f304426808bd1f51ce46ee6865551275d6e40dd3bdf0ca14a.jpg)

<details>
<summary>bar</summary>

| MNIST-Orig | Top-1 accuracy (%) |
| ---------- | ------------------ |
| Ind        | 99.2               |
| KL         | 98.9               |
| Hint       | 99.1               |
| CRD        | 99.1               |
</details>

![](images/473092b6a2a07b45aaa73a61a1614d90aeea46a2f0fa1ae75ee6ca866202ec7a.jpg)

<details>
<summary>bar</summary>

| MNIST-Color | Value |
| ----------- | ----- |
| Ind         | 63.6  |
| KL          | 89.4  |
| Hint        | 97.8  |
| CRD         | 67.8  |
</details>

![](images/0c09dcdb72d4506600a587df35bb2a58393d19992a8985869de8914efecc0303.jpg)

<details>
<summary>bar</summary>

|        | Value  |
| ------ | ------ |
| Ind    | 53.5   |
| KL     | 66.6   |
| Hint   | 69.7   |
| CRD    | 53.2   |
</details>

Figure 6: Left: The teacher is trained on three domains: MNIST-Orig, MNIST-Color, and MNIST-M. Distillation is done only on MNIST-Orig. Right: Test accuracy of the students. Note the increase in performance on MNIST-Color & MNIST-M domains by the distilled students.

unseen domains is surprising. After all, if the agreement is not increasing when images are from the seen domain, why should it increase when they are from an unseen domain? It is possible that there is more scope for increase in agreement in unseen domains vs seen domain. The consensus score between teacher and independent student for the seen domain is $\geq75\%$ (appendix), whereas for an unseen domain, e.g. sketch, it is $\leq40\%$ . So, it might not be that knowledge distillation does not work, as the authors wondered in [32], but its effect could be more prominent in certain situations.

# 4.5 Other studies

In appendix, we study additional aspects of a model such as shape/texture bias, invariance to random crops, and find that even these obscure properties can transfer from a teacher to the student. We also explore the following questions: If we can find alternative ways of increasing a student's performance (e.g. using crafted soft labels), will that student gain similar knowledge as a distilled student? If distillation cannot increase the student's performance, is there no knowledge transferred? Finally, we also present results on some more datasets like CIFAR100 [18], VLCS and PACS [20], showing that the phenomena of the implicit transfer of properties dueing knowledge distillation extends even to datasets beyond ImageNet and MNIST.

# 5 Applications

Beyond the exploratory angle of studying whether certain properties get transferred during distillation, the idea of a student becoming similar to a teacher in a broad sense has practical implications. We discuss a good and a bad example in this section.

# 5.1 The Good: the free ability of domain adaptation

Consider the following setup: the teacher is trained for a task by observing data from multiple domains $(\mathcal{D}_{1} \cup \mathcal{D}_{2})$ . It is then used to distill knowledge into a student on only $D_{1}$ . Apart from getting knowledge about $D_{1}$ , will the student also get knowledge about $D_{2}$ indirectly?

Experimental setup: We use MNIST digit recognition, and train the teacher on three domains: (i) MNIST-orig: original gray-scale images from MNIST, (ii) MNIST-Color: background of each image randomly colored, and (iii) MNIST-M [6]: MNIST digits pasted on random natural image patches. The student models are trained only on MNIST-orig and evaluated (top-1 accuracy) on all three domains. The network architecture is same for both the teacher and the student (see appendix).

Results: When the independent student is trained only on MNIST-orig, its performance drops on the unseen domains, which is expected due to domain shift. The distilled students (especially KL & Hint), however, are able to significantly improve their performance on both unseen domains; Fig. 6.

Discussion: This result shows distillation's practical benefits: once a teacher acquires an ability through computationally intensive training (e.g., training on multiple datasets), that ability can be distilled into a student, to a decent extent, through a much simpler process. The student sees the teacher's response to gray-scale images (MNIST-orig) that lack any color information. But that information helps the student to deal better with colored images (e.g., MNIST-Color), likely because the teacher has learned a domain-invariant representation (e.g., shape) which is distilled to the student.

# 5.2 The Bad: Students can inherit harmful biases from the teacher

Consider the problem of classifying gender from human faces. Imagine an independent student which performs the classification fairly across all races. The teacher, on the other hand, is biased against certain races, but is more accurate than the student on average. Will the student, which was originally fair, become unfair after mimicking the unfair teacher?

Experimental setup: We consider a ResNet20 → ResNet20 setting, and use FairFace dataset [16], which contains images of human faces from 7 different races with their gender labeled. From its training split, we create two different subsets ( $D_{s}$ and $D_{t}$ ) with the following objectives - (i) $D_{s}$ has a particular racial composition so that a model trained on it will perform fairly across all races during test time; (ii) $D_{t}$ 's composition is intended to make the model perform unfairly for certain races. The exact composition of $D_{s}$ and $D_{t}$ is given in the appendix. The teacher is trained on $D_{t}$ whereas the independent/distilled students are trained on $D_{s}$ . We use KL for distillation.

Results: We observe in the figure on the right that the independent student performs roughly fairly across all races, which is what we intended. The teacher is more accurate than the student, but performs relatively poorly on faces from Race 3 compared to others. After distillation, we observe that the student's overall accuracy improves, but the gain in accuracy is less for images from Race 3. So, when it is trained on $\mathcal{D}_s$ by itself, it behaves fairly. But when it mimics the teacher on $\mathcal{D}_s$ , it becomes unfair. This follows the observation made in $[23]$ , where the authors found that the performance gains during distillation might not be spread uniformly across different sub-categories.

![](images/2300ad14e445dab4280a861a4a0b899946efc1c0b401e8a7e9b9487465149570.jpg)

<details>
<summary>bar</summary>

| Race | Fair student | Distilled student | Unfair Teacher |
| :--- | :--- | :--- | :--- |
| Race 1 | 66 | 74 | 80 |
| Race 2 | 67 | 75 | 80 |
| Race 3 | 65 | 65 | 71 |
| Race 4 | 67 | 73 | 79 |
| Race 5 | 68 | 78 | 88 |
| Race 6 | 69 | 76 | 82 |
| Race 7 | 64 | 75 | 78 |
</details>

Discussion: The practical takeaway from the experiment is that knowledge distillation can bring forth behaviour which is considered socially problematic if it is viewed simply as a blackbox tool to increase a student's performance on test data, as it did in the previous example. Proper care must hence be taken regarding the transfer/amplification of unwanted biases in the student.

# 6 Why does knowledge distillation work in this way?

An Illustrative Example. Why should a teacher's response to an image contain such rich information about its implicit properties? We first intuitively explain this through a toy classification problem using $KL$ as the distillation objective. Fig. 7 (left) shows data points from two classes (red and blue). The teacher has access to the complete set, thereby learning to classify them appropriately (orange decision boundary). On the other hand, suppose that the training of the independent and distilled student is done on a subset of the points. The independent student can learn to classify them in different ways, where each decision boundary can be thought of as looking for different features, hence giving rise to different properties. But the distilled student is trying to mimic the teacher's class probabilities, which reflect the distance from the (orange) decision boundary of the teacher, visualized as circles around the points. Then the decision boundary of the distilled student should remain tangential to these circles so that it preserves the teacher's distances. So, this very simple case depicts that mimicking the class probabilities can force the student's decision boundary to resemble that of the teacher, which is not true for the independent student.

A mathematical formulation of the same case can be considered as follows. Let $D$ be a linearly separable dataset: $D \in \{(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n) | x_i \in \Re^m, y_i \in \{-1, 1\}\}$ . For an arbitrary linear classifier having weights $W \in \Re^{m \times 1}$ and biases $b \in \Re$ , the binary classification objective for the teacher and the independent student can be described as $\forall (x_i, y_i) \in D$ , $(W^T x_i + b)y_i > 0$ .

Since the dataset is linearly separable, both the teacher as well as the independent student can solve this system of inequalities in many ways. Let $\{W_t, b_t\}$ be the parameters of one particular teacher.

Next, we look at the objective of the distilled student, who tries to map $x_{i}$ to $z_{i} = W_{t}^{T}x_{i} + b_{t}$ , instead of $y_{i}$ . That is, the optimal solution satisfies: $\forall (x_i,z_i)\in D_{dist},W_s^T x_i + b_s = z_i$ . This system of linear equations will have a unique solution provided it has access to at least $m + 1$ training instances and that these instances are linearly independent. Since we know that the teacher's weights and

![](images/662d2a852d9078bd290aebd60d114a0c7ce8702084692fa4cfe1657a591fc73e.jpg)  
Figure 7: Left. Teacher learns on the entire dataset. Independent student without access to the full dataset might learn some spurious correlations. Distillation provides constraints to reconstruct the teacher's decision boundary. Right. Sampled decision boundaries for points from MNIST-Color visualized using [31]. The distilled student is better at reconstructing the teacher's decision boundary.

biases, $\{W_{t}, b_{t}\}$ , can satisfy this equation, we can conclude that if there exists a unique solution, then $W_{s} = W_{t}$ and $b_{s} = b_{t}$ . So, the distilled student, when solving the distillation objective, will recover the weights and biases of the teacher, and consequently its decision boundary.

Complex neural networks: Given that the distances of data points, i.e., decision boundary, can be preserved by the distilled student in the very simple case presented above, we now study whether they can be preserved in a more complex setting involving multi-layer neural networks as well.

There have been works [9, 31] which have studied this for the $KL$ objective: if the teacher and the students are all trained on the same dataset $\mathcal{D}$ , the decision boundary of the teacher will be more similar to the distilled student than the independent student for points from $\mathcal{D}$ . And while this result is useful, it is not clear whether (i) this can help explain why the student inherits teacher's knowledge about other domains, and (ii) whether this holds true for distillation objectives other than $KL$ .

To study this, we revisit the setup of Sec. 5.1, where the teacher was trained on three domains and distillation was performed on one (Fig. 6 left). We study what happens to the decision boundary similarity between the students and the teacher on all three domains. To compute that similarity between two neural networks, we use the method introduced in [31]. We randomly sample a triplet of data points from a given domain and create a plane that passes over that triplet. The networks that are being compared receive the same dataset of points in this plane, and we compute intersection over union between their predictions. We report the score averaged over 50 triplets.

<table><tr><td></td><td>Ind</td><td>KL</td><td>Hint</td><td>CRD</td></tr><tr><td>MNIST-Orig</td><td>0.9107</td><td>0.9341</td><td>0.9317</td><td>0.9062</td></tr><tr><td>MNIST-Color</td><td>0.4156</td><td>0.6711</td><td>0.7980</td><td>0.4733</td></tr><tr><td>MNIST-M</td><td>0.4691</td><td>0.5872</td><td>0.6541</td><td>0.4862</td></tr></table>

Table 1: Decision boundary similarity calculated between students and the teacher.

Results: Table 1 shows the similarity scores, where we see that the distilled students' decision boundary are almost always more similar to the teacher's than the independent student's. This is particularly evident when the scores are computed for domains not seen by the students; e.g., for MNIST-Color, the similarity score increases from 0.416 to 0.671 for the student distilled with KL.

Discussion: First, we can draw an analogy of this experimental setup with the toy example discussed before. The three MNIST related domains (for training the teacher) are similar to the overall set of blue+red points used to train the toy teacher (Fig. 7 left). The singular MNIST-orig domain (for training the students) is similar to the three points available to train the toy students. Now, what the results from Table 1 show is that the data points available for distillation can help determine the teacher's decision boundary even for points which the student did not have access to. So, similar to how the decision boundary estimated by the toy student can correctly estimate the distances of missing blue/red points, the neural network based student can also estimate the distances of points from unseen domains, e.g., MNIST-color, in a similar way as its teacher, thereby inheriting the teacher's behavior on these points. An example of this is given in Fig. 7 (right).

# 7 Limitations

While we have tried to study the distillation process in many settings, there do exist many more which remain unexplored. This is because the analysis conducted in our work has many axes of generalization: whether these conclusions hold (i) for other datasets (e.g. SUN scene classification dataset), (ii) in other kinds of tasks (object detection, semantic segmentation), (iii) in other kinds of architectures (modern CNNs, like ResNext), (iv) or for more recent distillation objectives, etc. Moreover, in Sec. 3.5 in the supplementary, we explore the transferability of shape/texture bias from teacher to student, and find some discrepancy in different distillation objective's abilities. Therefore, the conclusions drawn from our work will be more helpful if one can study all the combinations of these factors. Due to limitations in terms of space and computational requirements, however, we have not been able to study all of those combinations.

# 8 Conclusion

Knowledge distillation is a beautiful concept, but its success i.e, increase in student's accuracy, has often been explained by a transfer of dark knowledge from the teacher to the student. In this work, we have tried to shed some light on this dark knowledge. There are, however, additional open questions, which we did not tackle in this work: given the architectures of the teacher and student, is there a limit on how much knowledge can be transferred (e.g., issues with ViT → CNN)? If one wants to actively avoid transferring a certain property of the teacher into a student (Sec. 5.2), but wants to distill other useful properties, can we design an algorithm tailored for that? We hope this work also motivates other forms of investigation to have an even better understanding of the distillation process.

# Acknowledgement

This work was supported in part by NSF CAREER IIS2150012, and Institute of Information & communications Technology Planning & Evaluation(IITP) grant funded by the Korea government(MSIT) (No. 2022-0-00871, Development of AI Autonomy and Knowledge Enhancement for AI Agent Collaboration), Air Force Grant FA9550-18-1-0166, the National Science Foundation (NSF) Grants 2008559-IIS, 2023239-DMS, and CCF-2046710.

# References

[1] Lucas Beyer, Xiaohua Zhai, Amélie Royer, Larisa Markeeva, Rohan Anil, and Alexander Kolesnikov. Knowledge distillation: A good teacher is patient and consistent. In arXiv, 2021.   
[2] Cristian Bucila, Rich Caruana, and Alexandru Niculescu-Mizil. Model compression. In SIGKDD, 2006.   
[3] Jang Hyun Cho and Bharath Hariharan. On the efficacy of knowledge distillation. In ICCV, 2019.   
[4] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In arXiv, 2020.   
[5] Tommaso Furlanello, Zachary C. Lipton, Michael Tschannen, Laurent Itti, and Anima Anandkumar. Born again neural networks. In ICML, 2018.   
[6] Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario March, and Victor Lempitsky. Domain-adversarial training of neural networks. JMLR, 2016.   
[7] Robert Geirhos, Kantharaju Narayanappa, Benjamin Mitzkus, Tizian Thieringer, Matthias Bethge, Felix A Wichmann, and Wieland Brendel. Partial success in closing the gap between human and machine vision. In NeurIPS, 2021.   
[8] Robert Geirhos, Patricia Rubisch, Claudio Michaelis, Matthias Bethge, Felix A Wichmann, and Wieland Brendel. Imagenet-trained cnns are biased towards texture; increasing shape bias improves accuracy and robustness. In ICLR, 2019.

[9] Micah Goldblum, Liam Fowl, Soheil Feizi, and Tom Goldstein. Adversarially robust distillation. In Proceedings of the AAAI Conference on Artificial Intelligence, 2020.   
[10] Ian Goodfellow, Jon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. In ICLR, 2014.   
[11] Jianping Gou, Baosheng Yu, Stephen J. Maybank, and Dacheng Tao. Knowledge distillation: A survey. In arXiv, 2021.   
[12] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Dark knowledge. TTIC Distinguished lecture series, 2014.   
[13] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. In NeurIPS Deep Learning Workshop, 2014.   
[14] Zehao Huang and Naiyan Wang. Like what you like: Knowledge distill via neuron selectivity transfer. In arXiv, 2017.   
[15] Aditya K Menon, Ankit Singh Rawat, Sashank Reddi, Seungyeon Kim, and Sanjiv Kumar. A statistical perspective on distillation. In ICML, 2021.   
[16] Kimmo Karkkainen and Jungseock Joo. Fairface: Face attribute dataset for balanced race, gender, and age for bias measurement and mitigation. In WACV, 2021.   
[17] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In ICML, 2019.   
[18] Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. In Technical report, Citeseer, 2009.   
[19] Alexey Kurakin, Ian Goodfellow, and Sammy Bengio. Adversarial machine learning at scale. In arXiv, 2016.   
[20] Da Li, Yongxin Yang, Yi-Zhe Song, and Timothy M. Hospedales. Deeper, broader and artier domain generalization. In ICCV, 2017.   
[21] Yuan Li, Francis E.H.Tay, Guilin Li, Tao Wang, and Jiashi Feng. Revisiting knowledge distillation via label smoothing regularization. In CVPR, 2020.   
[22] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In ICCV, 2021.   
[23] Michal Lukasik, Srinadh Bhojanapalli, Aditya Krishna Menon, and Sanjiv Kumar. Teacher's pet: understanding and mitigating biases in distillation. In arXiv, 2021.   
[24] Rafael Müller, Simon Kornblith, and Geoffrey Hinton. When does label smoothing help? In NeurIPS, 2019.   
[25] Wonpyo Park, Dongju Kim, Yan Lu, and Minsu Cho. Relational knowledge distillation. In CVPR, 2019.   
[26] Baoyun Peng, Xiao Jin, Jiaheng Liu, Shunfeng Zhou, Yichao Wu, Yu Liu, Dongsheng Li, and Zhaoning Zhang. Correlation congruence for knowledge distillation. In ICCV, 2019.   
[27] Max Phuong and Christoph H. Lampert. Towards understanding knowledge distillation. In ICML, 2019.   
[28] Maithra Raghu, Thomas Unterthiner, Simon Kornblith, Chiyuan Zhang, and Alexey Dosovitskiy. Do vision transformers see like convolutional neural networks? In NeurIPS, 2021.   
[29] Adriana Romero, Nicholas Ballas, Samira Ebrahimi Kahau, Antoine Chassang, Carlo Gatta, and Yoshua Bengio. Fitnets: Hints for thin deep nets. In ICLR, 2015.   
[30] Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In IJCV, 2019.   
[31] Gowthami Somepalli, Liam Fowl, Arpit Bansal, Ping Yeh-Chiang, Yehuda Dar, Richard Baraniuk, Micah Goldblum, and Tom Goldstein. Can neural nets learn the same model twice? investigating reproducibility and double descent from the decision boundary perspective. arXiv preprint arXiv:2203.08124, 2022.   
[32] Samuel Stanton, Pavel Izmailov, Polina Kirichenko, Alexander A. Alemi, and Andrew Gordon Wilson. Does knowledge distillation really work? In NeurIPS, 2021.

[33] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In CVPR, 2016.   
[34] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive representation distillation. In ICLR, 2020.   
[35] Frederick Tung and Greg Mori. Similarity-preserving knowledge distillation. In ICCV, 2019.   
[36] Junho Yim, Donggyu Joo, Jihoon Bae, and Junmo Kim. A gift from knowledge distillation: Fast optimization, network minimization and transfer learning. In CVPR, 2017.   
[37] Sergey Zagoruyko and Nikos Komodakis. Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. In ICLR, 2017.   
[38] Richard Zhang. Making convolutional networks shift-invariant again. In ICML, 2019.   
[39] Ying Zhang, Tao Xiang, Timothy M. Hospedales, and Huchuan Lu. Deep mutual learning. In CVPR, 2018.   
[40] Zhilu Zhang and Mert Sabuncu. Self-distillation as instance-specific label smoothing. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 2184–2195. Curran Associates, Inc., 2020.

# Appendix

This document provides additional information complementing the main paper. First, we describe details pertaining to different distillation procedures used in Sec. 9. Then, in Sec. 10, we detail the iterative FGSM [19] used to create adversarial images. Following that, in Sec. 11, we perform more analyses to further dissect the distillation process, which corroborates our findings presented in the main paper. Finally, we present the top-1 accuracy of all the models, as well as the results shown in the main paper with their error bars, in Sec. 12. Additionally, we have provided scripts used for evaluation performed in Sec. 4.2 and 4.3; please see readme.txt.

# 9 Training details

ImageNet experiments: We first describe the hyper-parameters used for different distillation objectives.

- ResNet50 → ResNet18:   
- KL: $\gamma = 0.5$ , $\alpha = 0.5$   
- Hint: $\gamma = 1.0, \beta = 5.0$   
- CRD: $\gamma = 1.0$ , $\beta = 0.8$

\- VGG19 → VGG11:

\- KL: $\gamma = 1.0, \alpha = 0.2$

\- Hint: $\gamma = 1, \beta = 0.5$

\- CRD: $\gamma = 1, \beta = 0.8$

• VGG19 → ResNet18:

\- KL: $\gamma = 0.9$ , $\alpha = 0.1$

\- Hint: $\gamma = 1, \beta = 0.2$

\- CRD: $\gamma = 1, \beta = 1.2$

\- ViT → ResNet18:

\- KL: $\gamma = 1.0, \alpha = 0.2$

\- Hint: $\gamma = 1, \beta = 1$

\- CRD: $\gamma = 1, \beta = 0.2$

\- Swin-Base $\rightarrow$ Swin-Tiny:

\- KL: $\gamma = 0.1, \alpha = 0.9$

\- Hint: $\gamma = 1, \beta = 1$

\- CRD: $\gamma = 1, \beta = 0.8$

• ResNet50 (sty) → ResNet18:

\- KL (lower): $\gamma = 0.1$ , $\alpha = 0.9$

\- KL (higher): $\gamma = 0.9$ , $\alpha = 0.1$

\- Hint (lower): $\gamma = 1.0$ , $\beta = 0.2$

\- Hint (higher): $\gamma = 1.0$ , $\beta = 100.0$

\- CRD (lower): $\gamma = 1.0$ , $\beta = 0.8$

\- CRD (higher): $\gamma = 1.0$ , $\beta = 1.2$

• ResNet50 (col) → ResNet18:

\- KL: $\gamma = 0.5$ , $\alpha = 0.5$

\- Hint: $\gamma = 1.0$ , $\beta = 5.0$

\- CRD: $\gamma = 1.0$ , $\beta = 0.8$

\- ResNet50 → ResNet18 (w/o crop):

\- KL: $\gamma = 0.5$ , $\alpha = 0.5$

\- Hint: $\gamma = 1.0$ , $\beta = 0.2$

\- CRD: $\gamma = 1.0$ , $\beta = 0.8$

The temperature used in KL (Eq. 1 in main paper) is set to 4, and the temperature used in CRD (Eq. 3 in main paper) is set to 0.07. For CRD, the number of negative samples (N in Eq. 3) is set to 16384. For the other details, we follow the official PyTorch recommendations for training CNN-based classification models on ImageNet. $^{2}$ We train the independent students for 90 epochs, and all the distilled students for 100 epochs on ImageNet. For teacher models, we try to use those officially provided by PyTorch, whenever available. For all CNN teachers (except for stylized Res50 which is taken from here $^{3}$ ) and ViT, we take models from PyTorch torchvision model zoo. $^{4}$ For Swin transformer models, we follow the training process and pretrained models given by the authors. $^{5}$ We use one 3090 Ti for training ResNet18, and two 3090 Ti for training VGG11. Each experiment takes about 2-3 days. Four A6000 are used to train Swin-T, which takes around 5 days to train.

When performing distillation using Hint, we need to specify the intermediate layers at which the student will mimic the teacher. Following $[29]$ , we usually choose layers in the middle for that purpose. For ResNets, we choose feature after the second residual block, which has a resolution of $28 \times 28$ . For VGG11 and VGG19, we choose feature after 4th and 7th conv layer whose resolution is $56 \times 56$ . For Swin, we choose the feature coming after 'stage 2' (refer to Fig3 in $[22]$ ), which produces a feature of $28 \times 28$ resolution. In the case of ViT-B-32 → ResNet18, the intermediate layer for ResNet18 is chosen after the fourth residual block (right before average pooling), which produces a feature of $7 \times 7$ resolution. For ViT-B-32, we choose the last layer of the encoder backbone (right before classification head), which outputs a feature having 50 dimensions. Here, we remove the classification token feature and reshape the rest into a $7 \times 7$ representation.

Note that (i) ResNet50 (sty) denotes the ResNet50 teacher trained on Stylized ImageNet dataset, which is used in Section 4.5 in the main paper; (ii) ResNet50 (col) denotes the ResNet50 teacher trained with additional color augmentations, used in Section 4.3 (color-invariance experiment); (iii) ResNet18 (w/o crop) denotes the students trained without crop augmentations used in Section 4.3 (crop-invariance experiment). Finally, the further bifurcation in ResNet50 (sty) → ResNet18 i.e., lower vs higher, denotes the hyper-parameters used when we put a lower vs higher weight on the distillation loss component, relative to the cross-entropy loss.

MNIST experiments: The architecture of both the teacher and the student, as well as all the other training details (e.g. batch size, learning rate) is taken from the standard example given by PyTorch: Conv(32) → ReLU → Conv(64) → ReLU → MaxPool(2) → dropout(0.25) → Linear(9216, 128) → ReLU → dropout(0.5) → Linear(128, 10). $^{6}$ The distillation specific hyper-parameters are listed below:

- $KL$ : $\gamma = 0.1$ , $\alpha = 0.9$ , $\tau = 8$   
- Hint: $\gamma = 1.0$ , $\beta = 2.0$ , Conv(64) is chosen as the intermediate layer for both the teacher and the student.   
- $CRD$ : $\gamma = 1.0, \beta = 0.1, \tau = 0.1$ , no. of negative samples ( $N$ ) = 32.

# 10 Process of creating the adversarial images

In Section 4.2 of the main paper, we mentioned using Iterative-FGSM [10, 19] for converting a clean image $(I)$ to its adversarial form $(I^{adv})$ . Here, we describe that conversion process in detail. First, we pass the clean image through the target network (to be fooled). Then we compute the gradient of the loss function with respect to the image $(\nabla_I)$ , and then update the image in the opposite way, so as to maximize the loss $(J(I, y_{true}))$ . The update is bounded to be within a range $[I - \epsilon, I + \epsilon]$ , so that the change in the image is imperceptible. This whole process constitutes one step of FGSM, and the iterative version of this method does this for $k$ steps ( $k = 5$ in our case). The process can be depicted formally through Eq. 4, where $\alpha$ controls the step size:

$$
I _ {0} ^ {a d v} = I, \quad I _ {t + 1} ^ {a d v} = C l i p _ {I, \epsilon} \left\{I _ {t} ^ {a d v} + \alpha \operatorname{sign} \left(\nabla_ {X} J \left(I _ {N} ^ {a d v}, y _ {\text {true}}\right)\right) \right\} \tag {4}
$$

![](images/ecf30a8b54c151d6b6a676ff8742a3e0ca95f2370387ba135336989cd9268dea.jpg)

Figure 8: Visualizing the effect of data transformations. Top: Altering the color properties of an image (original) with increasing strengths. Middle: Taking random crops of an image (original) with different scale size. Bottom: Shifting the image left by different amounts. Color/crop invariance is studied in Sec. 4.3 of the main paper, and shift invariance is studied in Sec. 11.4.   
![](images/21cf314e66dea8bd4423384774b1e3dd20cf85ee0fa5ee9bae33789e860b3474.jpg)  
Figure 9: Centered kernel alignment (CKA) scores for various distillation settings. Left: Comparison of the teacher's representations with the independent and two distilled students (KL and Hint). Right: Comparison of the teacher (Swin-Tiny) with independent and distilled student (KL).

# 11 More analyses

# 11.1 Can distillation work even without increasing student's performance?

In the experiments discussed in the main paper, the distillation objective increases the performance of the student, compared to an independent student. However, it is possible that this does not happen, as was discussed in [3]. What do we conclude from that phenomenon? Is it that there is no knowledge transferred from the teacher to the student? In this section, we discuss such scenarios. We perform ResNet50 → ResNet18 distillation using all the distillation methods, using different hyper-parameter values ( $\alpha$ , $\beta$ , $\gamma$ in Equation 1 and 2 in main paper), and choose the distilled students that are no more accurate than the independent student. The top-1 accuracy of the models are: (i) $S_{Ind}$ : 70.03%, (ii) $S_{KL}$ : 69.23%, (iii) $S_{Hint}$ : 70.05% and (iv) $S_{CRD}$ :

![](images/8bb9350a7df5e1109999501340875a2ed0cba3800a7760e092e89295db147ad6.jpg)

<details>
<summary>bar</summary>

| Category | Fooling rate (%) |
| -------- | ---------------- |
| Ind      | 44.2             |
| KL       | 52.2             |
| Hint     | 61.8             |
| CRD      | 49.1             |
</details>

69.79%. Figure on the top shows the results of attacking these students using successful adversarial images crafted for ResNet50. Interestingly, the fooling rates for the distilled students are still higher compared to the independent student. So, while judging a distillation setup based on the increase in student's performance is fair, it is not that the knowledge distillation does not work if the student's performance is not increasing.

# 11.2 Can any soft label transfer a similar knowledge?

When performing distillation through $KL$ , the student has an additional target of soft labels from the teacher to match. In another line of work on 'label smoothing', converting the one-hot ground truth label into a softer version has also shown to improve a model's test performance [33, 24, 21, 40]. Could this mean that using any soft label, and not necessarily obtained through a teacher, can change a student's property e.g., color invariance to the same extent?

Experimental setup: We use ResNet18 as the student and train it for ImageNet classification using KL method. However, for each input image x, instead of $z_{t}$ (eq. 1, main paper) coming from an actual teacher, we generate the soft probabilities using x's ground-truth label y. We first add a random Gaussian noise with variance 0.2, and then perform the softmax operation with temperature 0.15 to convert it into a probability distribution. This probability vector then acts as the target for the student to match. We then evaluate the agreement score of this pseudo-distilled student for color invariance (similar to Figure 4(b) in main paper).

Results: We discuss three models, (i) the independent student (Ind): top-1 acc. = 70.04%, (ii) student distilled using color-invariant ResNet50 as the teacher (KL): top-1 acc. = 71.10%, and (iii) student distilled through the soft-labels without the teacher (KL\*): top-1 acc. = 70.49%. In the figure on the right, we see that while using soft-labels does marginally increase the agreement score of the student, it does not match the scores obtained by the students distilled with the actual color-invariant teacher. This reinforces the observation we made in section 4.3, that an increase in color invariance is primarily due to certain knowledge being inherited from the teacher.

![](images/654ba4646044b45a9f131c6420f60d2ef85c2b5369912331785257454c7b8119.jpg)

<details>
<summary>line</summary>

| Color Strength | Ind  | KL   | KL*  | T    |
| -------------- | ---- | ---- | ---- | ---- |
| 0.2            | 71   | 82   | 72   | 91   |
| 0.3            | 61   | 75   | 62   | 89   |
| 0.4            | 52   | 69   | 55   | 86   |
| 0.5            | 46   | 62   | 49   | 82   |
| 0.6            | 39   | 55   | 43   | 77   |
</details>

# 11.3 Does invariance to random crops transfer during knowledge distillation?

This section extends the study done in Sec. 4 of the main paper, but for another popular data augmentation technique: randomly resized crops.

Experimental setup (crop invariance): While training the teacher, we randomly crop the images as part of data augmentation (in addition to horizontal flips), with crop size between 8% to 100% of the image size. So, for example, the teacher can get to see a random 20% region of an image in one iteration, and a random 80% region of the same image in a different iteration. While training the students (independent or distilled), apart from horizontal flips, we only use center crop and do not show random crops of an image.

Results (crop invariance): During evaluation, we start with a test image X from the 50k val set. We then set a crop scale, e.g. 0.2, and generate two random crops $X_{1}$ and $X_{2}$ so that both cover a random 20% area of the original image X. Higher the crop scale, more image content will be common between the two crops. Then, we measure how frequently a model assigns the same class to $X_{1}$ and $X_{2}$ . Fig. 4(d) (main paper) shows the agreement scores for increasing crop scales, where we again observe that the students distilled through KL and CRD become more invariant to this operation. Student distilled through Hint, however, does not increase its invariance to random crops, just as it did not increase its invariance to color jittering to the same extent as other methods in Fig. 4(b) (main paper).

![](images/c512aa7711e7641ab52ef5417431bb8ff290137f40640182fb86752a8374785a.jpg)

<details>
<summary>line</summary>

| x    | CRD  | Hint | Ind  | KL   | T    |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 0.3  | 55   | 50   | 55   | 55   | 71   |
| 0.4  | 65   | 61   | 65   | 65   | 78   |
| 0.5  | 71   | 68   | 71   | 71   | 82   |
| 0.6  | 76   | 73   | 73   | 73   | 85   |
</details>

# 11.4 Does shift invariance transfer during knowledge distillation?

Section 4 (main paper) and 11.3 (appendix) discussed whether invariance to certain data transformations can transfer from a teacher to the student during knowledge distillation. Fig. 8 visualizes the effect of those transformations. Note that when we generate two random crops $(X_{1}, X_{2})$ of an image $(X)$ with a fixed scale (e.g. 0.4), the aspect ratio of the two crops can still be kept different, which is what we do in Fig. 8 (middle) and in the results shown in the previous section. If the aspect ratio is

kept the same between $X_{1}$ and $X_{2}$ , then one can study a more common property of neural networks: shift invariance i.e. whether the network's predictions remain same if we shift an image by certain pixels (either left/right/top/bottom). We study if this knowledge can be transferred from a teacher to the student during the distillation process.

Experimental setup: For the teacher, we choose a model which has been explicitly made to be shift-invariant. A recent work showed that a model's robustness to input shifts is related with the aliasing phenomenon, which refers to signal distorted with a small downsampling rate. To alleviate this issue and make CNNs shift invariant, [38] inserts low-pass filters into CNNs before downsampling. So, we use an anti-aliased ResNet50# as the teacher (# represents anti-aliased, same for the below). The student is the standard ResNet18 (without being anti-aliased). The distillation ResNet50# → ResNet18 is done on the standard ImageNet dataset. The shift invariance of a model is evaluated across the 50k validation images in ImageNet. We start with a test image X resized into 256x256 resolution. Then, we define the maximum shift we want in the resulting two images. If, for example, that value is 32, then we do a center crop of 256x256 followed by two random 224x224 crops to generate $X_{1}$ and $X_{2}$ , keeping the aspect ratio same for both. If, instead, we desire a maximum shift of only 8 between $X_{1}$ and $X_{2}$ , we would do a center crop of 232x232, followed by two random 224x224 crops. Then, we compute how frequently a model gives the same prediction for $X_{1}$ and $X_{2}$ , which is called the agreement score (same as section 4.3).

Results: In the figure on the right, we see the agreement scores of different models, and see that the agreement scores of the ResNet18 students distilled using KL and CRD increase relative to the independent ResNet18. Note that one can convert ResNet18 (the student) into its anti-aliased version as well by inserting low-pass filters [38]. The agreement score achieved by this student can be thought of as the upper-limit for a ResNet18 model, which we show by light green colored plot (denoted as Ind#). Given the results of section 4.3 (crop-invariance), this result is expected since invariance to image shifts (aspect ratio constant) is a subset of invariance to random crops (aspect ratio could be different). Again, we observe that Hint has difficulty in transferring this property.

![](images/5d02a623ed05eacd8ecf98d256726849f8f49ce69fff293c6754b111f0ad91af.jpg)

<details>
<summary>line</summary>

| x    | CRD   | Hint  | Ind   | Ind#  | KL    | T#    |
| ---- | ----- | ----- | ----- | ----- | ----- | ----- |
| 8    | 90.5  | 89.2  | 89.0  | 91.5  | 90.0  | 93.5  |
| 16   | 88.5  | 87.5  | 87.0  | 89.5  | 88.0  | 92.5  |
| 24   | 87.0  | 86.0  | 85.5  | 88.5  | 87.0  | 91.5  |
| 32   | 86.5  | 85.0  | 84.5  | 87.5  | 86.0  | 91.0  |
</details>

# 11.5 Does shape/texture bias get distilled?

The previous section dealt with knowledge about images from unseen domains, and the section before that discussed if certain invariances can be transferred. This section brings together those ideas to study an important property: shape/texture bias of neural networks. Prior work has shown that convolutional networks tend to overly rely on texture cues when categorizing images $[8]$ . Here we study the following: If the teacher is shape biased, and the default (independent) student more texture biased, does distillation increase the shape bias of the distilled student?

Experimental setup: We use the toolbox in $[7]$ to compute the shape vs. texture biases of a model. Shape bias is computed by using images with conflicting content and style information: e.g., an image with a shape (content) of a cat but texture (style) of an elephant. So, this particular image could have two correct decisions, a cat or an elephant. Using such images, the task is to see what fraction of correct decisions are based on shape vs. texture information. For the teacher, we choose a ResNet50\* trained on Stylized-ImageNet $[8]$ , where the image labels are kept the same, but the style is borrowed from arbitrary paintings. This way, the teacher has to focus more on shape information and consequently has a high shape bias of $\sim0.81$ . We choose ResNet18 as the student, as it has a lower shape bias of $\sim0.21$ . We then perform ResNet50\* → ResNet18 distillation on the standard ImageNet dataset; i.e., the student is trained without any stylized images, while the teacher is, and we evaluate whether the student inherits the shape bias of the teacher. We also conduct an experiment with a transformer teacher and CNN student: ViT → ResNet18. Since ViT have been shown to be inherently more shape-biased, we do not train the ViT teacher on Stylized-ImageNet, and instead train both it and the student on standard ImageNet.

Results: For each distillation method, we show two results: one with lower weight on the distillation loss ( $\downarrow$ ) and one with higher ( $\uparrow$ ). From (a) in the right figure, we see that both KL and CRD improve

the distilled student's shape bias, with a further jump obtained when using a higher weight, especially through $KL$ . Sec. 4 (main paper) already showed that the student can indirectly inherit color invariance properties of the teacher. But, it is still interesting to see that, with proper hyperparameters, the inherited knowledge includes more subtle properties, like texture invariance as well.

For ViT (shape bias = 0.615) → ResNet18, the shape bias of the distilled students do not change much (b). This follows a general trend where distilling knowledge from a transformer into a CNN turns out to be difficult. The implicit biases introduced due to architectural differences between the teacher and student, seem too big to be overcome by current distillation methods.

![](images/fba4a2cf6d955f4937bfc44616a6d7af05927b61d6e8155944e20354edf9afbc.jpg)

<details>
<summary>bar</summary>

| Method   | Shape bias |
| -------- | ---------- |
| Ind      | 0.21       |
| ↓KL↑     | 0.26       |
| ↓Hint↑   | 0.5        |
| ↓CRD↑   | 0.22       |
|         | 0.24       |
|         | 0.24       |
|         | 0.25       |
</details>

(a) ResNet50\* → ResNet18

![](images/6f30ecb29f6367dcbe334ad3301ae5fb0ac2eb50ded27e31d744656fb5073f4a.jpg)

<details>
<summary>bar</summary>

| Category | Value |
|---|---|
| Ind | 0.21 |
| KL | 0.2 |
| Hint | 0.22 |
| CRD | 0.21 |
</details>

(b) ViT → ResNet18

# 11.6 Distillation makes internal representations to become similar

We hypothesize the following: when mimicking the teacher at a particular layer, the student's intermediate representations before that layer become similar as well. That is, rather than predicting the activations in the target layer (e.g., output layer) in a very different way (e.g., the student classifying an image based on color features while the teacher classifies it based on shape), the student learns to behave more like the teacher throughout its network. However, the degree to which this happens depends both on which layer the student mimics, and how similar the student's architecture is to that of the teacher. To study these aspects, we use centered kernel alignment (CKA) [17], a popular method for measuring the similarity of two neural networks. Given two representations, $X \in \mathbb{R}^{n \times p_1}$ and $Y \in \mathbb{R}^{n \times p_2}$ of the same $n$ inputs, $\mathrm{CKA}(X, Y) \in [0, 1]$ indicates how similar (close to 1) or dissimilar (close to 0) the two are.

Experimental setup: We consider three settings: (i) ResNet50 → ResNet18 using KD; (ii) ResNet50 → ResNet18 using Hint (distillation after the default second convolutional stage); and (iii) Swin-tiny → ResNet18 using KD. For each setting, we consider representations from (roughly) corresponding locations in the network (e.g., after the last layer in each convolutional stage). Seven corresponding locations are chosen from the teacher and student (for ResNets, the same layers used in the Hint ablation study, Fig. 4d). We take 100 random images from the ImageNet validation set and compute their representations from those layers to construct a 7 x 7 similarity matrix. We compare the teacher to both the independent and distilled student to get two similarity matrices.

Results: Figure 9 shows the similarities between the teacher and the independent/distilled students. First, we see that the scores are higher between the corresponding feature representations (along the diagonal entries) of the distilled student and teacher networks for ResNet50 → ResNet18, with KD resulting in a more significant gain than Hint. Second, we see very similar and low overall scores (except for the target F7 layer) for the independent and distilled students for Swin-tiny → ResNet18. These support our hypothesis that the student learns similar intermediate representations as the teacher before the target layer, if the student and teacher's architectures are of the same family (e.g., both are ResNets). Moreover, mimicking the output class probabilities (KD) leads to the student learning more similar representations as those of the teacher than mimicking an earlier layer (Hint). Finally, when the architectures are very different (Swin-tiny and ResNet18), the intermediate representations do not become similar (despite a performance gain of the distilled student) because their inductive biases lead to different ways of learning the task. Overall, our analysis shows that there is a correlation between the degree to which a student inherits the teacher's general properties and learned representation similarities.

# 11.7 Results on additional datasets

Beyond the results on ImageNet and MNIST, we study if the phenomena of implicit knowledge transfer exists for some other datasets as well. First, we conduct the adversarial vulnerability experiment on CIFAR-100 $[18]$ (section 4.2 in the main paper). We report the results on three different teacher-student settings: (i) Wide ResNet 40-2 (WRN-40-2) -> ShuffleNetV1, (ii) VGG13 -> VGG8, (iii) ResNet50 -> MobileNetV2. Both the teacher and the students are trained on the training split of CIFAR-100, and tested on 5000 random images from the test split. The results shown below in Table 2 depict the fooling rates (in %) of different kinds of students when using adversarial images crafted for the teacher. We see that the fooling rates increase for distilled students, following a similar

<table><tr><td></td><td>WRN 40-2 → ShuffleNetV1</td><td>VGG-13 → VGG-8</td><td>ResNet50 → MobileNetV2</td></tr><tr><td>Ind</td><td>31.23</td><td>42.42</td><td>36.57</td></tr><tr><td>KL</td><td>48.62</td><td>51.87</td><td>43.32</td></tr><tr><td>Hint</td><td>62.63</td><td>49.79</td><td>43.91</td></tr><tr><td>CRD</td><td>49.41</td><td>54.68</td><td>46.07</td></tr></table>

Table 2: Aversarial vulnerability results on CIFAR-100 (analogous to Section 4.2 in the main paper.

<table><tr><td></td><td>Caltech 101 (unseen)</td><td>LabelMe (seen)</td><td>SUN09 (seen)</td><td>VOC 2007 (seen)</td></tr><tr><td>Ind</td><td>54.31</td><td>61.07</td><td>60.38</td><td>51.94</td></tr><tr><td>KL</td><td>71.83</td><td>60.73</td><td>61.52</td><td>53.12</td></tr><tr><td>Hint</td><td>67.95</td><td>60.84</td><td>60.34</td><td>52.76</td></tr><tr><td>CRD</td><td>61.66</td><td>63.57</td><td>55.31</td><td>53.12</td></tr></table>

Table 3: Domain adaptation results on VLCS results (analogous to Fig. 6). Teacher's knowledge about the unseen domain (Caltech 101) gets transferred, to some extent, into the distilled student.

pattern as in Sec. 4.2 - Fig. 3. So, adversarial vulnerability of the teacher does get distilled into the students trained with different distillation objectives.

Next, similar to the MNIST domain adaptation experiment in Section 5.1, we conduct the experiment on a more real-world domain. We consider two datasets: VLCS and PACS [20]. VLCS consists of images from four domains - VOC2007, LabelMe, Caltech-101, and SUN - where in each domain there are images belonging to five categories. PACS consists of images from four domains as well - sketch, photo, cartoon, art painting. Each domain consists of the same seven object categories. The teacher is trained on all the four domains, but the students (independent and distilled) are trained on three domains (images from one domain are never shown). The unseen domains are Caltech 101 and Photo when working with VLCS and PACS datasets respectively.

The goal is to see if mimicking the teacher on three domains also helps the student inherit teacher's information on the fourth (hidden) domain; i.e., do we see an improvement in the accuracy on that hidden domain for the distilled students, compared to an independent student. The results, depicting the classification accuracy of different models, are shown below in Tables 3 and 4. The distilled students' performance improves on the unseen domains in both the cases, simply by having access to teacher's responses on the other three domains. This is particularly pronounced when distillation is done using KL. For example, for VLCS, the performance on the unseen domain (Caltech 101) improves from 54.31 by the independent student to 71.83 by the KL distilled student.

<table><tr><td></td><td>Photo (unseen)</td><td>Sketch (seen)</td><td>Cartoon (seen)</td><td>Art (seen)</td></tr><tr><td>Ind</td><td>40.91</td><td>70.17</td><td>66.47</td><td>45.56</td></tr><tr><td>KL</td><td>49.93</td><td>72.82</td><td>69.69</td><td>46.64</td></tr><tr><td>Hint</td><td>48.34</td><td>71.13</td><td>68.08</td><td>45.89</td></tr><tr><td>CRD</td><td>44.49</td><td>71.52</td><td>67.14</td><td>47.86</td></tr></table>

Table 4: Domain adaptation results on PACS dataset (analogous to Table 3).

# 12 Supporting quantitative results

Finally, we report the performance of different models on ImageNet 50k validation set. Table 5 lists the top-1 accuracies of different models used in the main paper. Overall, we have tried to use the hyper-parameters which improve the distilled student's performance compared to the independent student. In every case, we use a single teacher to perform distillation into two students trained with different random seeds i.e. Teacher $\rightarrow$ Student $_{1}$ and Teacher $\rightarrow$ Student $_{2}$ , for each method. We then report the results shown in the main paper with their respective error bars, in Tables 6-13.

<table><tr><td></td><td>Teacher</td><td>Ind</td><td>KL</td><td>Hint</td><td>CRD</td></tr><tr><td>ResNet50 → ResNet18</td><td>76.13</td><td>70.04±0.01</td><td>70.98±0.01</td><td>70.56±0.16</td><td>70.73±0.02</td></tr><tr><td>VGG19 → VGG11</td><td>72.37</td><td>68.88±0.01</td><td>69.74±0.10</td><td>69.38±0.15</td><td>69.74±0.07</td></tr><tr><td>VGG19 → ResNet18</td><td>72.37</td><td>70.04±0.01</td><td>70.62±0.02</td><td>70.21±0.30</td><td>70.42±0.07</td></tr><tr><td>ViT → ResNet18</td><td>75.91</td><td>70.04±0.01</td><td>70.39±0.02</td><td>70.59±0.07</td><td>70.58±0.03</td></tr><tr><td>Swin-Base → Swin-Tiny</td><td>83.50</td><td>81.13±0.08</td><td>81.23±0.04</td><td>81.33±0.11</td><td>81.27 ±0.21</td></tr><tr><td>ResNet50 (sty) → ResNet18 ↑</td><td>60.18</td><td>70.04±0.01</td><td>61.45±0.07</td><td>68.82±0.12</td><td>69.56±0.07</td></tr><tr><td>ResNet50 (sty) → ResNet18 ↓</td><td>60.18</td><td>70.04±0.01</td><td>70.65±0.03</td><td>70.45±0.07</td><td>69.96±0.05</td></tr><tr><td>ResNet50 (col) → ResNet18</td><td>75.32</td><td>70.04±0.01</td><td>71.01±0.06</td><td>70.41±0.20</td><td>70.97±0.19</td></tr><tr><td>ResNet50 → ResNet18 (w/o crop)</td><td>76.13</td><td>64.84±0.02</td><td>68.75±0.01</td><td>64.81±0.14</td><td>67.41±0.07</td></tr></table>

Table 5: Top-1 accuracy (in %) of different models on 50k ImageNet validation images.

<table><tr><td></td><td>Teacher</td><td>Ind</td><td>KL</td><td>Hint</td><td>CRD</td></tr><tr><td>ResNet50 → ResNet18</td><td>84.82</td><td> $44.16 \pm 0.19$ </td><td> $51.98 \pm 2.44$ </td><td> $48.34 \pm 0.34$ </td><td> $50.46 \pm 0.29$ </td></tr><tr><td>VGG19 → VGG11</td><td>87.22</td><td> $62.29 \pm 0.36$ </td><td> $69.74 \pm 0.67$ </td><td> $79.78 \pm 0.08$ </td><td> $70.51 \pm 0.78$ </td></tr><tr><td>VGG19 → VGG11 (R18)</td><td>87.22</td><td> $69.02 \pm 0.48$ </td><td> $70.54 \pm 0.90$ </td><td> $70.68 \pm 0.62$ </td><td> $70.59 \pm 0.62$ </td></tr><tr><td>ViT → ResNet18</td><td>85.84</td><td> $21.93 \pm 0.24$ </td><td> $21.57 \pm 0.49$ </td><td> $23.34 \pm 0.14$ </td><td> $23.47 \pm 0.31$ </td></tr><tr><td>VGG19 → ResNet18</td><td>87.22</td><td> $36.19 \pm 0.01$ </td><td> $43.02 \pm 0.06$ </td><td> $47.68 \pm 0.47$ </td><td> $48.99 \pm 0.05$ </td></tr></table>

Table 6: Adversarial fooling rates (in %), corresponding to Figure 3 in the main paper.

<table><tr><td></td><td>ResNet50 (col)</td><td>ResNet50</td></tr><tr><td>Ind</td><td>71.27±0.21</td><td>71.27±0.21</td></tr><tr><td>KL</td><td>82.10±0.07</td><td>74.02±0.23</td></tr><tr><td>Hint</td><td>72.22±0.14</td><td>72.42±0.39</td></tr><tr><td>CRD</td><td>79.44±0.20</td><td>71.27±0.25</td></tr></table>

Table 7: Table corresponding to Figure 4(a) in main paper. Knowledge transfer about color information from two teachers: color invariant ResNet50 (T) and default ResNet50 (T\*).

<table><tr><td></td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.6</td></tr><tr><td>Ind</td><td> $60.77 \pm 0.10$ </td><td> $52.32 \pm 0.21$ </td><td> $45.55 \pm 0.20$ </td><td> $39.56 \pm 0.31$ </td></tr><tr><td>KL</td><td> $75.32 \pm 0.17$ </td><td> $68.56 \pm 0.36$ </td><td> $61.93 \pm 0.33$ </td><td> $55.15 \pm 0.37$ </td></tr><tr><td>Hint</td><td> $61.96 \pm 0.00$ </td><td> $53.34 \pm 0.41$ </td><td> $47.72 \pm 0.51$ </td><td> $42.00 \pm 0.53$ </td></tr><tr><td>CRD</td><td> $71.83 \pm 0.48$ </td><td> $64.31 \pm 0.14$ </td><td> $57.26 \pm 0.47$ </td><td> $49.90 \pm 0.48$ </td></tr></table>

Table 8: Table corresponding to Figure 4(b) in main paper. Illustration of knowledge transfer in, ResNet50 → ResNet18, if the two images have increasingly different color properties.

<table><tr><td></td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.6</td></tr><tr><td>Ind</td><td> $60.77 \pm 0.10$ </td><td> $52.32 \pm 0.21$ </td><td> $45.55 \pm 0.20$ </td><td> $39.56 \pm 0.31$ </td></tr><tr><td>KL</td><td> $62.49 \pm 0.09$ </td><td> $54.18 \pm 0.15$ </td><td> $47.68 \pm 0.50$ </td><td> $41.92 \pm 0.48$ </td></tr><tr><td>Hint</td><td> $61.68 \pm 0.04$ </td><td> $53.63 \pm 0.19$ </td><td> $47.37 \pm 0.32$ </td><td> $41.76 \pm 0.50$ </td></tr><tr><td>CRD</td><td> $60.85 \pm 0.65$ </td><td> $52.80 \pm 0.76$ </td><td> $46.91 \pm 1.29$ </td><td> $42.00 \pm 1.31$ </td></tr></table>

Table 9: Table corresponding to Figure 4(c) in main paper. Illustration of knowledge transfer in, Swin-Tiny → ResNet18, if the two images have increasingly different color properties.

<table><tr><td></td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.6</td></tr><tr><td>Ind</td><td> $50.30 \pm 0.21$ </td><td> $61.27 \pm 0.10$ </td><td> $68.20 \pm 0.50$ </td><td> $73.30 \pm 0.33$ </td></tr><tr><td>KL</td><td> $56.26 \pm 0.00$ </td><td> $66.06 \pm 0.02$ </td><td> $72.06 \pm 0.19$ </td><td> $76.79 \pm 0.04$ </td></tr><tr><td>Hint</td><td> $50.23 \pm 0.27$ </td><td> $61.14 \pm 0.11$ </td><td> $68.04 \pm 0.14$ </td><td> $73.36 \pm 0.08$ </td></tr><tr><td>CRD</td><td> $55.10 \pm 0.12$ </td><td> $65.36 \pm 0.43$ </td><td> $71.55 \pm 0.26$ </td><td> $76.57 \pm 0.43$ </td></tr></table>

Table 10: Table corresponding to Figure 4(d) in main paper. Illustration of knowledge transfer in, ResNet50 → ResNet18, if the two images are random crops of increasing scales.

<table><tr><td rowspan="2"></td><td colspan="6">VGG19 → ResNet18</td></tr><tr><td>sketch</td><td>stylized</td><td>silhouette</td><td>edge</td><td>cue conflict</td><td>ImageNet val</td></tr><tr><td>Ind</td><td>33.62±0.12</td><td>21.68±0.19</td><td>12.81±0.94</td><td>26.25±1.25</td><td>22.81±0.00</td><td>75.60±0.01</td></tr><tr><td>KL</td><td>37.56±0.31</td><td>28.81±0.44</td><td>31.25±5.00</td><td>31.25±5.00</td><td>29.30±2.03</td><td>77.21±0.06</td></tr><tr><td>Hint</td><td>37.18±1.19</td><td>27.19±0.19</td><td>10.00±3.75</td><td>29.69±2.19</td><td>27.73±1.25</td><td>76.49±0.09</td></tr><tr><td>CRD</td><td>40.50±0.37</td><td>30.75±0.50</td><td>37.81±2.19</td><td>35.00±1.25</td><td>30.93±0.08</td><td>78.36±0.06</td></tr></table>

<table><tr><td rowspan="2"></td><td colspan="6">Swin-Base → Swin-Tiny</td></tr><tr><td>sketch</td><td>stylized</td><td>silhouette</td><td>edge</td><td>cue conflict</td><td>ImageNet val</td></tr><tr><td>Ind</td><td> $51.37 \pm 0.37$ </td><td> $33.75 \pm 0.37$ </td><td> $22.50 \pm 1.25$ </td><td> $50.00 \pm 0.00$ </td><td> $37.26 \pm 0.47$ </td><td> $88.79 \pm 0.07$ </td></tr><tr><td>KL</td><td> $56.93 \pm 1.06$ </td><td> $38.43 \pm 0.68$ </td><td> $27.50 \pm 2.50$ </td><td> $57.81 \pm 1.56$ </td><td> $42.61 \pm 0.04$ </td><td> $89.39 \pm 0.05$ </td></tr><tr><td>Hint</td><td> $52.56 \pm 1.18$ </td><td> $35.18 \pm 0.19$ </td><td> $26.87 \pm 1.87$ </td><td> $54.37 \pm 2.50$ </td><td> $38.51 \pm 0.62$ </td><td> $89.03 \pm 0.17$ </td></tr><tr><td>CRD</td><td> $54.18 \pm 0.94$ </td><td> $34.75 \pm 1.50$ </td><td> $26.25 \pm 0.00$ </td><td> $50.62 \pm 0.00$ </td><td> $39.22 \pm 0.47$ </td><td> $88.97 \pm 0.01$ </td></tr></table>

Table 11: Consensus scores between teacher and the student, corresponding to Figure 5 in the paper. ImageNet val denotes the 50k images in the validation set of the seen domain (ImageNet).

<table><tr><td rowspan="2"></td><td colspan="2">ResNet50 (sty) → ResNet18</td><td rowspan="2">ViT → ResNet18</td></tr><tr><td>Lower</td><td>Higher</td></tr><tr><td>Ind</td><td>0.21±0.01</td><td>0.21±0.01</td><td>0.21±0.01</td></tr><tr><td>KL</td><td>0.26±0.01</td><td>0.50±0.00</td><td>0.20±0.01</td></tr><tr><td>Hint</td><td>0.22±0.01</td><td>0.24±0.02</td><td>0.22±0.00</td></tr><tr><td>CRD</td><td>0.24±0.00</td><td>0.25±0.01</td><td>0.21±0.00</td></tr></table>

Table 12: Shape bias scores of students, corresponding to the figure in Section 4.5 in the main paper.

<table><tr><td></td><td>MNIST-orig</td><td>MNIST-Color</td><td>MNIST-M</td></tr><tr><td>Ind</td><td>99.08±0.07</td><td>72.86±2.32</td><td>56.09±1.14</td></tr><tr><td>KL</td><td>98.90±0.01</td><td>91.76±1.00</td><td>67.92±1.07</td></tr><tr><td>Hint</td><td>99.10±0.06</td><td>97.05±0.05</td><td>64.06±0.93</td></tr><tr><td>CRD</td><td>99.00±0.10</td><td>83.98±0.88</td><td>60.36±0.23</td></tr></table>

Table 13: Top-1 accuracy of distilled models, corresponding to figure 6 in the main paper.

<table><tr><td></td><td> $\mathcal{D}_{s}$ </td><td> $\mathcal{D}_{t}$ </td></tr><tr><td>Race 1</td><td>600</td><td>4000</td></tr><tr><td>Race 2</td><td>50</td><td>4000</td></tr><tr><td>Race 3</td><td>2000</td><td>0</td></tr><tr><td>Race 4</td><td>200</td><td>4000</td></tr><tr><td>Race 5</td><td>0</td><td>4000</td></tr><tr><td>Race 6</td><td>200</td><td>4000</td></tr><tr><td>Race 7</td><td>800</td><td>4000</td></tr></table>

Table 14: Dataset composition of FairFace [16]. Different rows represent the number of training images used from each race.