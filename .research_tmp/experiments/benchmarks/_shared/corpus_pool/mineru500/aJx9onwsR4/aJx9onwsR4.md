# Predicting the Performance of Foundation Models via Agreement-on-the-Line

Rahul Saxena $^{*1}$ Taeyoun Kim $^{*1}$ Aman Mehra $^{*1}$ Christina Baek $^{1}$ Zico Kolter $^{1,2}$ Aditi Raghunathan $^{1}$

Carnegie Mellon University $^{1}$ , Bosch Center for AI $^{2}$ {rsaxena2, taeyoun3, amanmehr, kbaek, zkolter, raditi}@cs.cmu.edu

# Abstract

Estimating the out-of-distribution performance in regimes where labels are scarce is critical to safely deploy foundation models. Recently, it was shown that ensembles of neural networks observe the phenomena “agreement-on-the-line”, which can be leveraged to reliably predict OOD performance without labels. However, in contrast to classical neural networks that are trained on in-distribution data from scratch for numerous epochs, foundation models undergo minimal finetuning from heavily pretrained weights, which may reduce the ensemble diversity needed to observe agreement-on-the-line. In our work, we demonstrate that when lightly finetuning multiple runs from a single foundation model, the choice of randomness during training (linear head initialization, data ordering, and data subsetting) can lead to drastically different levels of agreement-on-the-line in the resulting ensemble. Surprisingly, only random head initialization is able to reliably induce agreement-on-the-line in finetuned foundation models across vision and language benchmarks. Second, we demonstrate that ensembles of multiple foundation models pretrained on different datasets but finetuned on the same task can also show agreement-on-the-line. In total, by careful construction of a diverse ensemble, we can utilize agreement-on-the-line-based methods to predict the OOD performance of foundation models with high precision.

# 1 Introduction

Foundation models (FM), or large models first pretrained on open world data then finetuned or prompted for a specific downstream task, have proven to be powerful solutions for many common machine learning problems. A notable trait about FMs is that they are far more robust to distribution shift than other deep learning approaches — across image and language benchmarks, they suffer a smaller performance degradation on out-of-distribution (OOD) data, that may vary substantially from the in-distribution (ID) finetuning data $[43, 42, 7, 60, 58, 14]$ . From clinical decision-making in different hospitals to navigating robots through unseen terrains, FMs are increasingly utilized for tasks prone to distribution shift. However, evaluating these models in OOD settings remains difficult: in many cases, acquiring labels for OOD data is costly and inefficient, while unlabeled OOD data is much easier to collect. Although the field has explored other means for estimating OOD accuracy without labeled data, they are not ideal for FMs. A reliable FM performance estimator has the following desirable properties. First, the method must be computationally efficient to account for FMs' large model size. Second, FMs are leveraged for many different tasks (e.g., classification, question-answering, regression), so the method should also be versatile across tasks. Third, as we will see, methods for finetuned FMs may require different model assumptions from neural networks trained from scratch.

![](images/3ce68daee01f9485e464d84dc4589990b950f7160f29e93e5da5e3959418fd65.jpg)

<details>
<summary>scatter</summary>

| ID | OOD | Accuracy | Agreement |
|----|-----|----------|-----------|
| 5  | 5   | 8        | 6         |
| 10 | 10  | 12       | 10        |
| 15 | 15  | 16       | 14        |
| 20 | 20  | 20       | 18        |
| 25 | 25  | 24       | 22        |
| 30 | 30  | 28       | 26        |
| 35 | 35  | 32       | 30        |
| 40 | 40  | 36       | 34        |
| 45 | 45  | 40       | 38        |
| 50 | 50  | 44       | 42        |
| 55 | 55  | 48       | 46        |
| 60 | 60  | 52       | 50        |
| 65 | 65  | 56       | 54        |
| 70 | 70  | 60       | 58        |
| 75 | 75  | 64       | 62        |
| 80 | 80  | 68       | 66        |
| 85 | 85  | 72       | 70        |
| 90 | 90  | 76       | 74        |
</details>

![](images/893a406c069141a3a949c7e48aa4291d232d35cbbff8adbcaf38f6d31313666e.jpg)

<details>
<summary>line</summary>

| ID  | Accuracy | Agreement |
| --- | -------- | --------- |
| 10  | 10       | 10        |
| 30  | 30       | 40        |
| 50  | 50       | 60        |
| 70  | 70       | 80        |
| 90  | 90       | 100       |
| 110 | 110      | 120       |
</details>

![](images/f72a48099b34110866ac8762f743b1d18c0c5e9647fb11ce4608e685d0073a80.jpg)

<details>
<summary>line</summary>

| ID  | Accuracy | Agreement |
| --- | -------- | --------- |
| 10  | 10       | 10        |
| 30  | 25       | 30        |
| 50  | 30       | 50        |
| 70  | 45       | 70        |
| 90  | 60       | 90        |
| 110 | 75       | 110       |
</details>

![](images/8aaf900b21694b0e656c337677faa3ad95d37b59759c23b4a91133a231ff8e7a.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  |
| --- | ---- |
| 30  | 30   |
| 40  | 40   |
| 50  | 50   |
| 60  | 60   |
| 70  | 70   |
| 80  | 80   |
| 90  | 90   |
</details>

![](images/07104ca304ed7de9ba718a0db900f8ad7cac820fc7c85f3b9b0f4dc488a8d763.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  |
| --- | ---- |
| 30  | 30   |
| 40  | 40   |
| 50  | 50   |
| 60  | 60   |
| 70  | 70   |
| 80  | 80   |
| 90  | 90   |
</details>

![](images/b8d11a84a27dc8ab00b382e095dd1fba0b3c8df6acae49ce1209269efb4d898a.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  |
| --- | ---- |
| 30  | 30   |
| 50  | 50   |
| 70  | 70   |
| 90  | 90   |
</details>

![](images/f7425f5752c33441dd3207916dcf1687ff8980d362701319fa4dad561c074bbe.jpg)

<details>
<summary>line</summary>

| ID  | F1 Score | Agreement |
| --- | -------- | --------- |
| 10  | 10       | 10        |
| 30  | 30       | 25        |
| 50  | 50       | 45        |
| 70  | 70       | 65        |
| 90  | 90       | 85        |
</details>

(a) Random Head

![](images/1fdf8f5b86825ab76dafe53a33adbe4a9eace00c14058648dfacba104fea8ab5.jpg)

<details>
<summary>line</summary>

| ID  | F1 Score | Agreement |
| --- | -------- | --------- |
| 10  | 10       | 10        |
| 30  | 25       | 28        |
| 50  | 40       | 45        |
| 70  | 55       | 60        |
| 90  | 70       | 80        |
</details>

(b) Data Ordering

![](images/938b4ab43583abc7ef7323705eab59026f6c98cafb16287737f61bc7f9ac4ddc.jpg)

<details>
<summary>line</summary>

| ID  | F1 Score | Agreement |
| --- | -------- | --------- |
| 10  | 10       | 10        |
| 20  | 20       | 20        |
| 30  | 30       | 30        |
| 40  | 40       | 40        |
| 50  | 50       | 50        |
| 60  | 60       | 60        |
| 70  | 70       | 70        |
| 80  | 80       | 80        |
| 90  | 90       | 90        |
</details>

(c) Data Subsetting   
Figure 1: The ID vs OOD lines for accuracy (orange) and agreement (blue) for various datasets and fine-tuned ensembles. Each blue dot corresponds to a member of the ensemble and represents the ID (x) and OOD (y) accuracy. Each orange dot corresponds to a pair of these members and represents the ID (x) and OOD (y) agreement. From CIFAR10 to CIFAR10C “Pixelate” in linear probed CLIP, MNLI to SNLI in full fine-tuned OPT, and SQuAD to SQuAD-Shifts “Amazon” in full fine-tuned GPT2, we observe that randomly initializing the head as the diversity source for generating ensembles (columns) shows the closest agreement linear fit to accuracy.

Recently, [2] proposed a promising method for estimating the OOD accuracy of deep networks using the agreement between pairs of these classifiers (i.e., how often two classifiers make the same prediction). For distribution shifts where models observe a strong linear correlation in ID versus OOD accuracy – a common phenomenon in vision and language benchmarks $[39, 1]$ – a strong linear correlation also holds for ID versus OOD agreement with extremely similar slopes and intercepts. These effects are referred to as accuracy-on-the-line (ACL) and agreement-on-the-line (AGL) respectively, and together they provide a simple method for estimating OOD accuracy via unlabeled data alone. Namely, without any OOD labels, we can instead measure the linear fit of ID versus OOD agreement as a proxy for the linear fit of ID versus OOD accuracy. With this linear fit, we can verify whether ACL holds by using the correlation strength of agreement's linear trend and estimating each model's OOD accuracy by linearly transforming ID accuracy. This simple approach has shown to reliably predict the OOD accuracy of models within a few percentage points across classification and question-answering tasks.

Unfortunately, while the method has several practical advantages, it is unclear whether finetuned FMs also observe the necessary AGL phenomena. Intuitively, a prerequisite to observing AGL is a diverse ensemble of classifiers. Since OOD accuracy falls below ID accuracy, if the linear trend in ID versus OOD agreement is to match that of accuracy, models must also agree much less OOD than ID. For this to happen, errors between any two models must be sufficiently decorrelated. [2] observes AGL in ensembles of neural networks trained for hundreds of epochs from scratch where it is conceivable that the stochasticity between training runs leads to large divergences in the weight space, and corresponding models have diverse OOD predictions. However, in the case of finetuned FMs, models are much closer in the weight space. FMs are often either linear probed over the same pretrained weights or full finetuned for a few epochs with a small learning rate and intuitively, such light finetuning may lead to models that “revert back” to their pretrained behavior to make highly correlated predictions OOD.

This raises the question: can we enforce AGL in this paradigm of lightly finetuning heavily pretrained models? In this work, we conduct an extensive study across several modalities, e.g., CLIP-based image classification and LLM-based question-answering, and training regiments, e.g., full finetuning and linear probing, to understand when AGL holds for finetuned FMs. We first investigate whether AGL appears in an ensemble of finetuned models from a single base FM. To collect a deep ensemble, the following sources of diversity can be injected into the finetuning process: 1) random initialization of the linear head; 2) random data ordering; and 3) random data subsetting. We find that not every source of diversity during fine-tuning on ID data manifests in sufficient diversity OOD, breaking the matching linear fits in ID versus OOD accuracy and agreement. Interestingly, finetuning models from different random initializations of the linear head consistently induces AGL across benchmarks. In contrast, neural networks trained from scratch observe AGL irrespective to these diversity sources.

Second, we show that finetuned models from multiple different base FMs can be leveraged for AGL-based performance estimation. As base FMs can be pretrained with different datasets, architectures, and training regiments, the linear trends in ID versus OOD accuracy and agreement may break altogether in such ensembles. Indeed previous works indicate that on vision tasks, FMs pretrained on different image corpora can have different levels of OOD robustness for the same ID performance $[17, 43, 51]$ . On the contrary, we find that on language tasks, FMs pretrained on different text corpora observe both AGL and ACL across question-answering and text classification tasks.

In total, we develop simple techniques for applying AGL-based performance estimation methods to predict the OOD performance of foundation models. We demonstrate that the AGL phenomenon is not limited to ensembles of neural networks trained from scratch. By simply finetuning FMs from random initializations of the linear head, we can observe the phenomena in FMs across a wide variety of tasks (classification, question-answering) and modalities (vision, language) and training procedures (linear probing, full finetuning). We find that AGL is the only method to accurately estimate the performance of finetuned FMs across all tasks, surpassing other performance estimation baselines by a significant margin as large as 20% mean absolute percentage error.

# 2 Background and related work

# 2.1 Setup

We are interested in evaluating models that map an input $x \in X$ to a discrete output $y \in Y$ . In particular, we finetune foundation models. For a base model B, let $f(B)$ denote a finetuned version of B. In this work, we consider a variety of foundation models: GPT2 [42], OPT [65], Llama2 [53], BERT [14], and CLIP [43].

Finetuning strategies. We have access to labeled data from some distribution $D_{ID}$ that we use for obtaining $f(B)$ from B. In this work, we consider the following standard finetuning procedures.

1. Linear probing (LP): Given features from the base model $B_{\theta}$ , we train a linear head v such that the final classifier maps the score $v^{\top}\mathsf{B}_{\theta}(x)$ to a predicted class. We randomly initialize v and update v via gradient steps on a suitable loss function. The base model parameters remain frozen. We refer to v as either a linear probe (classification), or span prediction head (question-answering) depending on the task.

2. Full finetuning (FFT): We update all parameters of the backbone $B_{\theta}$ and the linear head v using a small learning rate. When infeasible to update all parameters, we perform low-rank adaptation (LoRA) [25] to reduce the number of trainable parameters while still effectively updating the feature extractor $B_{\theta}$ . In this work, we do not distinguish between LoRA and FFT as they conceptually achieve the same effect, and seem to show similar empirical trends in our studies.

OOD performance estimation. Given access to a labeled validation set from $D_{ID}$ and unlabeled samples from a related but different distribution $D_{OOD}$ , our goal is to estimate performance on $D_{OOD}$ . We consider the standard performance metrics for various tasks: Accuracy $\ell_{0-1}: Y \mapsto [0,1]$ for classification, and Exact Match $\ell_{EM}: Y \mapsto [0,1]$ and Macro-averaged F1 score $\ell_{F1}: Y \mapsto [0,1]$ for question-answering. We use $\ell$ to denote the appropriate metric in the context.

# 2.2 Background on OOD accuracy estimation

There is rich literature on OOD performance estimation for deep networks, with a variety of proposed approaches. Initial works focused on upper bounding the degree of distribution shift through data and/or model dependent metrics, e.g., uniform convergence bounds using H-divergence $[4, 37, 11, 30]$ . However, these bounds tend to be loose for deep networks $[39]$ . The following works try to estimate the performance exactly.

For classification, [23, 22, 19, 16, 21] leverage the model's confidence to predict the OOD performance. Since deep models are typically overconfident, these models are first calibrated in-distribution by temperature scaling. Similar methods are uncertainty quantification works that directly calibrate models under distribution shift [62, 67, 41]. Confidence based methods are commonly utilized in practice, and favorable for foundation models as they are computationally light and model-agnostic. However, they often fail for large shifts [19] and are often well-defined for accuracy but not other common metrics like F1 score. These can be limiting factors for foundation models which are applied to a broad array of tasks. Still, as they are the most common estimation methods, we utilize them as the baselines in our work.

[49, 12, 13] also measure model behavior on known auxiliary tasks to understand model behavior under the distribution shift at hand. However, these approaches tend to be overfit to specific datasets or modalities. Similar to AGL, there are prediction methods that utilize information from ensembles. Oftentimes a separate “reference” ensemble is trained on some objective to predict the performance of a “target” model [10, 63, 8]. These methods have a higher computational cost than AGL. Although AGL also requires at least 3 models to compute agreement, these models only undergo generic finetuning. Thus, it is a better suited approach for evaluating foundation models, especially if off-the-shelf finetuned models are readily available, e.g., from Huggingface (see Section 4).

Overall, there is growing attention towards understanding the safety and reliability of foundation models. To understand the effective robustness of FMs under distribution shift, recent works focus on studying the “accuracy-on-the-line” phenomena $[39]$ (details in next subsection) and designing benchmarks that expose different failure modes of large models $[36, 54]$ . However, unsupervised OOD performance estimation is underexplored in this modern setting, in terms of new methods and the transferability of old methods to large pretrained models.

# 2.3 Accuracy and agreement on the line

We are interested in adapting the method “agreement-on-the-line” (AGL) [2] for OOD estimation as it obtains state-of-the-art performance estimation across several distribution shifts. AGL is based on an earlier observation called “accuracy-on-the-line” (ACL) — across common distribution shift benchmarks, there is a strong linear correlation between the ID and OOD performance of models [39, 45–47, 61, 51, 38]. ACL can also be observed in FMs for image classification, e.g., CIFAR10C [22], ImageNetV2 [46], FMoW-WILDS [28], and question-answering, e.g., SQuAD-Shifts [38]. However, ACL does not always hold, e.g., Camelyon-WILDS [39] and SearchQA [1].

While ACL is a striking phenomenon, it does not immediately provide a practical method to estimate OOD performance—computing the linear fit of ID versus OOD accuracy requires labeled samples from $D_{OOD}$ . Alternatively, we can estimate this linear trend exactly using only the agreement between neural networks [2]. Formally, given a pair of models $f_{1}$ and $f_{2}$ that map inputs to labels, accuracy

and agreement is defined as

$$
\operatorname{Acc} \left(f _ {i}\right) = \mathbb {E} _ {x, y \sim \mathcal {D}} [ \ell \left(f _ {i} (x), y\right) ], \quad \operatorname{Agr} \left(f _ {1}, f _ {2}\right) = \mathbb {E} _ {x, y \sim \mathcal {D}} [ \ell \left(f _ {1} (x), f _ {2} (x)\right) ], \tag {1}
$$

where $\ell$ is the appropriate performance metric of interest. While accuracy requires access to the ground truth labels y, agreement only requires access to unlabeled data and a pair of models. [2] observes that when ID versus OOD accuracy is strongly linearly correlated between neural networks, i.e., ACL, then the ID versus OOD agreement of pairs of these models also observe a strong linear correlation with the same linear slope and bias. Furthermore, when accuracies do not show a linear correlation, agreements also do not. This coupled phenomenon is dubbed “agreement-on-the-line” (AGL).

To use AGL for OOD performance estimation, one may obtain the slope and bias of the agreement line with unlabeled data, and then estimate the OOD performance by linearly transforming the ID validation performance. Specifically, with a collection of models $F = \{f_{1}, f_{2}, ..., f_{n}\}$ , AGL suggests that ID versus OOD accuracy observe a strong linear correlation if and only if ID versus OOD agreement observes a strong linear correlation and when they do, the slopes and biases match: $\forall f_{i}, f_{j} \in F$ where $i \neq j$

$$
\Phi^ {- 1} \left(\operatorname{Acc} _ {\mathrm{OOD}} \left(f _ {i}\right)\right) = a \cdot \Phi^ {- 1} \left(\operatorname{Acc} _ {\mathrm{ID}} \left(f _ {i}\right)\right) + b
$$

$$
\Updownarrow \tag {2}
$$

$$
\Phi^ {- 1} (\mathrm{Agr} _ {\mathrm{OOD}} (f _ {i}, f _ {j})) = a \cdot \Phi^ {- 1} (\mathrm{Agr} _ {\mathrm{ID}} (f _ {i}, f _ {j})) + b
$$

$\Phi^{-1}$ is the probit transform used to induce a better linear fit as used in [2, 39]. Provided access to $\text{Acc}_{\text{ID}}(f_i), \text{Agr}_{\text{ID}}(f_i, f_j), \text{Agr}_{\text{OOD}}(f_i, f_j) \forall i, j$ , we can estimate $\text{Acc}_{\text{OOD}}(f_i)$ for all $f_i \in \mathcal{F}$ . We refer the reader to [2] for formal AGL-based performance estimation algorithms (ALine-S and ALine-D), which we also provide in Appendix A.1.1.

# 3 Predicting OOD performance: single base foundation model

We first evaluate whether AGL appears in an ensemble of multiple finetuned runs of a single base foundation model. This would enable precise OOD performance estimates for each ensemble member. A practitioner may naively gather a finetuned ensemble by training a couple runs with different seeds or hyperparameters. However, an overriding concern is that even with some randomness in the finetuning process, linear probing or light full-finetuning over the same base model may lead to solutions with very correlated predictions. We extensively evaluate the following methods of introducing diversity into the finetuning process to see what approach (if any) can lead to AGL.

1. Random linear heads We initialize the last layer of the network (i.e., the linear head) randomly, instead of via some zero-shot or pre-specified manner.   
2. Data ordering We present the same training data to each model but shuffle the order of the data, i.e., model observes different minibatches.   
3. Data subsetting We i.i.d. sample p% subset of the data to train over. In the main body, we report models trained on independently sampled 10% of the training data, other proportions of 30% and 50% are reported in Appendix A.4.

We perturb one source of diversity at a time and study whether AGL occurs in each resulting model ensemble. For each setting, we also vary the number of training epochs to collect models with different ID performance, which is necessary to obtain a meaningful linear correlation in accuracy. The additional randomness induced by different training epochs does not affect the observations we make. We use at most four A6000's for all experiments except for linear probing where we use one RTX 8000.

# 3.1 VLM-based Image Classification

We first investigate the effect of diversity source on AGL behavior for vision benchmarks. For image classification, a common pipeline is to finetune over a CLIP [43] pretrained foundation model.

CLIP Linear Probing We finetune over OpenCLIP ViT-B/32 model trained on LAION-2B [26]. Given its well-established zero-shot capabilities, a popular method of finetuning CLIP is to simply employ linear probing on top of the CLIP representation. We take particular interest in evaluating the OOD performance of an ensemble of linear models trained on top of frozen base model representations.

Datasets We evaluate ensembles on synthetic corruptions (CIFAR10C, CIFAR100C, ImageNetC), dataset replication shifts (CIFAR10.1, ImageNetV2), style shifts (OfficeHome), geographical and temporal shifts (FMoW-WILDS, iWildCam-WILDS), and interlaboratory shifts in medicine (Camelyon17-WILDS). iWildCam-WILDS exhibits weak ACL and Camelyon17-WILDS doesn't exhibit any ACL [39]. We test on iWildCam-WILDS and Camelyon17-WILDS to verify AGL's negative condition, i.e., when the linear correlation does not exist in ID versus OOD accuracy, it also does not exist in agreement.

Table 1: We evaluate models on the following distribution shift benchmarks. 

<table><tr><td>ID</td><td>OOD</td></tr><tr><td>CIFAR10 [29]</td><td>CIFAR10C [22], CIFAR10.1 [45]</td></tr><tr><td>CIFAR100 [29]</td><td>CIFAR100C [22]</td></tr><tr><td>ImageNet [48]</td><td>ImageNetC [22], CIFAR10.1 [45]</td></tr><tr><td>FMoW ID [28]</td><td>FMoW OOD [28]</td></tr><tr><td>iWildCam ID [28]</td><td>iWildCam OOD [28]</td></tr><tr><td>Camelyon17 ID [28]</td><td>Camelyon17 OOD [28]</td></tr><tr><td>OfficeHome [56]</td><td>All (ID, OOD) pairings of domains Art, ClipArt, Product, Real World</td></tr><tr><td>MNLI [59]</td><td>MNLI-Mismatched [59], SNLI [6]</td></tr><tr><td>SQuAD [44]</td><td>SQuAD-Shifts [38]</td></tr></table>

![](images/107b90a6434493d7ae4665e81365c86ba0fbff8fc2c8a85eed5e03940647175b.jpg)

<details>
<summary>scatter</summary>

| ID | OOD | Accuracy |
|----|-----|----------|
| 5  | 2   | 3        |
| 10 | 8   | 7        |
| 15 | 12  | 10       |
| 20 | 16  | 14       |
| 25 | 20  | 18       |
| 30 | 24  | 22       |
| 35 | 28  | 26       |
| 40 | 32  | 30       |
| 45 | 36  | 34       |
| 50 | 40  | 38       |
| 55 | 44  | 42       |
| 60 | 48  | 46       |
| 65 | 52  | 50       |
| 70 | 56  | 54       |
| 75 | 60  | 58       |
| 80 | 64  | 62       |
| 85 | 68  | 66       |
| 90 | 72  | 70       |
</details>

![](images/592fb2dfd231e6cd2903ee081ad7a60b87ca0c88779058a625fef904db250058.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  |
| --- | ---- |
| 70  | 50   |
</details>

![](images/860dbcc5ee11818bef173d10191f4d5d5158a246e446831f1076f1299ecd69ec.jpg)

<details>
<summary>line</summary>

| ID  | OOD  |
| --- | ---- |
| 10  | 10   |
| 20  | 20   |
| 30  | 30   |
| 40  | 40   |
| 50  | 50   |
| 60  | 60   |
| 70  | 70   |
| 80  | 80   |
| 90  | 90   |
</details>

![](images/28969c92fcade519677ae8da7385cda1eeeb66e3cf0ef50199017eff5ae43302.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 10 | 10  |
| 30 | 30  |
| 50 | 50  |
| 70 | 70  |
| 90 | 90  |
</details>

Figure 2: In ensembles with diverse random initializations, ACL and AGL holds across benchmarks in linear probed CLIP models. Similar to [2], neither ACL nor AGL holds for the Camelyon17-WILDS

Results Across vision benchmarks, linear probed CLIP models observe ACL, i.e. there is a strong linear correlation in the ID versus OOD performance. Similarly, on the same datasets, we can observe a corresponding strong linear correlation in agreement across ensembles injected with diversity in linear head initialization, data ordering, and data subsetting (Figures 1). However, we find that only ensembles with diverse initialization leads to AGL where the linear trend of agreement and accuracy have a matching slope and bias (Figure 2). See Appendix A.3.2 for results on other datasets. In model ensembles obtained by data ordering and data subsetting, we observe a consistent trend where the agreement trend observe a much higher slope close to the diagonal y = x line. These results are not specific to linear probing alone. In full finetuned CLIP models, we also observe that random linear heads induce the most reliable AGL behavior (See Appendix A.3). Note that this setting is still notably different from [2] where models are heavily trained for tens to hundreds of epochs often with a large learning rate, which causes AGL behavior to be more robust to the source of diversity used to induce the ensemble (See Appendix A.3.4).

# 3.2 LLM-based Question-Answering and Text Classification

We conduct a similar systematic investigation of AGL in finetuned runs of a single base language model. Similar to CLIP linear probing, we find that AGL cannot be observed without random head initialization in language models evaluated on text classification and extractive question-answering tasks. While we mostly focus on tasks that require a linear head during finetuning, we also conduct a diversity study on generative tasks where the base model is finetuned directly in Appendix A.6.1.

Table 2: ALine-D MAPE (%) of different sources of diversity for CLIP linear probing and GPT2-Medium/OPT-125M full finetuning. We average the score over all corruptions for CIFAR10C. 

<table><tr><td>Source of Diversity</td><td>CIFAR10C</td><td>SQuAD-Shifts Amazon</td><td>SQuAD-Shifts Reddit</td><td>SNLI</td></tr><tr><td>Random Linear Heads</td><td>14.64</td><td>6.34</td><td>3.48</td><td>11.70</td></tr><tr><td>Data Ordering</td><td>37.01</td><td>10.30</td><td>9.59</td><td>15.40</td></tr><tr><td>Data Subsetting</td><td>35.85</td><td>16.21</td><td>13.94</td><td>15.50</td></tr></table>

Full Finetuned Language Models We evaluate over a collection of 450 full finetuned runs of several base FMs: GPT2-Medium [42] and OPT-125M [65]. Models are full finetuned for up to 20 epochs with a small learning rate ( $\leq 1e^{-6}$ ). Hyperparameters specifics can be found in Appendix A.2. We do not conduct a linear probing study for question-answering as it leads to poorly performing models. For text classification, we also conduct a linear probing study in Appendix A.5.

Datasets We test models on a text classification shift from MNLI [59] in the GLUE benchmark [57] to MNLI-Mismatched [59] and SNLI [6]. We also evaluate extractive question-answering models on the shift from SQuAD v1.1 [44] to SQuAD-Shifts [38].

Results We evaluate models on accuracy for text classification and F1 score for question-answering. Similar to our findings in CLIP, in both text classification and question-answering benchmarks, ensembles of full finetuned LLMs observe AGL when models are trained from different randomly initialized linear or span heads while data ordering and data subsetting observe an agreement trend closer to the diagonal y = x line (Figure 1 and Appendix A.5). We note that with full finetuning, the differences in AGL behavior between diversity sources are not as stark as with linearly probed models. In some sense, how model diversity is achieved becomes increasingly less important for observing AGL as the base model parameters also diverge, with ensembles of models heavily trained from scratch at the extreme [2].

# 3.3 Summary and Implications

Across image and language modalities, we demonstrate that ensembles of finetuned FMs can also observe agreement-on-the-line similar to heavily trained CNN's $[2]$ . In both domains and regardless of the fine-tuning strategy, e.g. FFT and LP, or metric, e.g. F1 and Accuracy, employed, the diversity induced via random head initialization yields AGL, while the diversity induced via data reordering or data subsetting does not. This phenomenon can be observed across hyperparameters (Appendix A.7) and different PEFT methods (Appendix A.8). With a single heavily pre-trained base FM, one may think that light finetuning leads to downstream models with highly correlated behavior under distribution shift. However, simply randomly initializing the linear head alone induces sufficiently decorrelated models for observing AGL. The diversity in the ensemble becomes important when predicting the OOD performance of models using downstream AGL-based methods. In Table 2, we show that AGL-based methods can only accurately predict the OOD performance of models in ensembles with diverse initialization, and cannot with data subsetting or ordering.

Furthermore, our findings contrast previous work that suggest AGL is a neural-network specific phenomenon $[2, 31]$ , unlike ACL which is model agnostic $[39]$ . Specifically, $[2]$ report that linear models trained on top of the flattened CIFAR10 images do not observe AGL. However, we find that, on top of CLIP features, linear models can exhibit AGL with random initialization. Previous work on the Generalization Disagreement Equality $[27]$ contend that data subsetting leads to the most diversity in model predictions for deep ensembles (neural networks, random forests). Specifically, in-distribution, the agreement rate between pairs of models was shown to equal their expected accuracy in ensembles obtained by data subsetting, while those that vary random initialization has slightly higher agreement $[27, 40]$ . On the other hand, in our problem setting of out-of-distribution datasets on FMs, we found that ensembles induced by different random initialization achieves AGL, while data ordering or subsetting cannot. Our setting is different from previous literature in two ways: (1) AGL studies the OOD agreement rate relative to their ID agreement, in contrast to the GDE phenomenon which only regards the models' ID agreement. We hypothesize that random initialization is much more important for observing the right levels of OOD agreement. (2) Models

![](images/39a97b28f5f785df2d7aadc5b0e705e4850a47fec93fbf9f7942d4beb0033f85.jpg)  
Agreement Llama GPT OPT

Figure 3: AGL can be observed between models finetuned from different base models (Llama, GPT, OPT) for the F1 score for question-answering shift (SQuAD to SQuAD-Shifts) and accuracy for text classification (MNLI-Matched to MNLI-Mismatched and SNLI). SQuAD-Shifts New Wiki, SQuAD-Shifts NYT, and MNLI Mismatched show little drop in OOD performance because the distribution shift is small compared to the corresponding ID dataset. Nonetheless, we observe that AGL holds regardless of the degree of distribution shift.

are only lightly fine-tuned or linearly probed, unlike deep ensembles trained from scratch. Diversity sources may behave differently in this circumstance.

# 4 Predicting OOD performance: multiple foundation models

Alternatively, we consider ensembling multiple base foundation models. First, ACL may not hold because the base models are heavily pretrained on different data corpuses. This may cause respective downstream models to have different ID versus OOD accuracy trends or “effective robustness” [17]. On vision tasks, for example, linear probing over CLIP, EfficientNet [50], ViT [15], and BYOL [20] observe varying robustness trends [43]. Second, even when ACL does hold, it is unclear whether the ensembles will also observe AGL. Here the problem is different from the single base model setting: any pair of foundation models finetuned from different base models may agree too little, or OOD agreement rate may vary across model pairs depending on the similarity of the pretraining corpus, breaking the linear correlation of agreement entirely. Yet, we observe that for language models and tasks, ensembles of finetuned FMs from a wide range of base models observe both ACL and AGL.

Models We finetune models from OPT-125M, OPT-350M, OPT-1.3B [65], GPT2, GPT2-Medium, GPT2-Large, GPT2-XL [42], GPT-Neo-135M [5], Llama2-7B [53], Alpaca-7B [52], and Vicuna-7B [9]. We fully finetune OPT and GPT models and LoRA finetune Llama, Alpaca, and Vicuna. These models are pretrained on different mixtures of BookCorpus [66], Stories [55], PILE [18], CCNews v2 corpus, and PushShift.io Reddit [3]. Alpaca and Vicuna are instruction-finetuned over Llama2.

# 4.1 Results

We investigate the AGL behavior of an ensemble of foundation models finetuned from diverse base models in Figure 3 for question-answering. First note that base LLM models pretrained on different text corpora lead to finetuned models that lie on the same linear trend in accuracy. Unlike the different accuracy trends observed by different vision foundation models [42], we suspect that the pretraining datasets for the language models in our study observe much more homogeneity. Second, the ID

Table 3: The MAPE (%) of predicting OOD performance using AGL-based ALine and other baseline methods. We collect a diverse ensemble by randomizing the linear initialization and including multiple base models. \*We filter out shifts with low correlation in agreement. 

<table><tr><td>OOD Dataset</td><td>ALine-D</td><td>ALine-S</td><td>Naive Agr</td><td>ATC</td><td>AC</td><td>DOC-Feat</td></tr><tr><td>CIFAR10C*</td><td>5.44</td><td>4.73</td><td>17.39</td><td>6.90</td><td>11.49</td><td>11.91</td></tr><tr><td>CIFAR10.1 v6</td><td>1.95</td><td>1.99</td><td>16.95</td><td>2.60</td><td>4.93</td><td>5.36</td></tr><tr><td>CIFAR100C*</td><td>7.17</td><td>6.79</td><td>17.66</td><td>8.18</td><td>17.58</td><td>14.96</td></tr><tr><td>ImageNetC*</td><td>15.03</td><td>14.17</td><td>32.27</td><td>15.90</td><td>22.83</td><td>13.42</td></tr><tr><td>ImageNetV2 MatchFreq</td><td>8.44</td><td>8.43</td><td>22.21</td><td>3.02</td><td>15.53</td><td>8.43</td></tr><tr><td>fMoW-WILDS</td><td>14.26</td><td>7.29</td><td>141.31</td><td>12.17</td><td>19.87</td><td>8.76</td></tr><tr><td>OfficeHome-Art</td><td>17.78</td><td>13.94</td><td>40.60</td><td>38.52</td><td>19.68</td><td>44.40</td></tr><tr><td>OfficeHome-ClipArt</td><td>14.04</td><td>11.70</td><td>33.44</td><td>33.81</td><td>14.97</td><td>28.77</td></tr><tr><td>OfficeHome-Product</td><td>14.64</td><td>11.78</td><td>39.88</td><td>84.18</td><td>63.05</td><td>75.44</td></tr><tr><td>OfficeHome-Real</td><td>12.65</td><td>10.18</td><td>36.20</td><td>27.72</td><td>21.85</td><td>28.85</td></tr><tr><td>SQuAD-Shifts Reddit</td><td>3.61</td><td>3.48</td><td>26.56</td><td>19.06</td><td>30.94</td><td>9.18</td></tr><tr><td>SQuAD-Shifts Amazon</td><td>3.61</td><td>4.93</td><td>26.46</td><td>24.35</td><td>34.93</td><td>7.31</td></tr><tr><td>SQuAD-Shifts NYT</td><td>1.64</td><td>1.75</td><td>23.46</td><td>4.01</td><td>25.96</td><td>2.80</td></tr><tr><td>SQuAD-Shifts New Wiki</td><td>6.33</td><td>6.58</td><td>25.24</td><td>5.18</td><td>25.96</td><td>7.50</td></tr><tr><td>MNLI Mismatched</td><td>0.55</td><td>0.41</td><td>0.55</td><td>12.00</td><td>0.63</td><td>0.51</td></tr><tr><td>SNLI</td><td>2.90</td><td>2.10</td><td>2.50</td><td>6.30</td><td>3.80</td><td>8.40</td></tr></table>

versus OOD agreement between pairs of models in this ensemble, including those between different base foundation models, is also strongly correlated and the slope and intercept closely matches that of accuracy. In other words, ensembles of different base models also observe AGL without any special regularization for ensemble diversity. The same holds for generative QA tasks (Appendix A.6.2).

# 5 Estimating OOD Accuracy using AGL in Diverse Ensembles

By constructing a diverse ensemble of foundation models, we can leverage AGL to extract precise estimates of model performances under distribution shift. We construct ensembles by collecting models trained from randomly-initialized heads (Section 3) and different base models (Section 4). For image classification, our model collection consists just linear models over CLIP representations. For text classification and question-answering, we include GPT, OPT, and Llama models individually finetuned from differently initialized heads. In Table 3, we compare the Mean Absolute Percentage Error (MAPE) of AGL-based prediction algorithms, ALine-S and ALine-D [2], to other baselines.

We compare against confidence based methods ATC [19], AC [24] and DOC-Feat [21] and Naive Agreement which directly uses agreement between model pairs [27, 35]. For confidence based methods, we first temperature scale the models using ID validation data, and pick the lower error rate from the estimations obtained with and without temperature scaling. However, there are several limitations when naively applying confidence baselines to estimate performance on question-answering, as they estimate classification accuracy. First, there is no easy analogous formulation of confidence baselines for the F1 score, so we estimate the exact-match score instead for fair comparison. On the other hand, AGL can predict performance across metrics accuracy, F1, and exact-match. Second, extractive question-answering is a joint classification task where models predict both the start and end token index of the answer span in the context. More details for how we calibrate baselines for this setting is provided in Appendix A.1.2.

Because ALine-S and ALine-D only provide estimation guarantees where the coefficient of determination $R^{2}$ of the linear fit in agreement is strong ([2]), we filter out datasets with low $R^{2} \leq 0.95$ . These shifts include iWildCam-WILDS, Camelyon-WILDS, and a few corruptions in CIFAR10C, CIFAR100C, and ImageNetC. We evaluate ALine-S/D for these failure cases in Appendix A.1.3. Across datasets with a high $R^{2}$ in agreement, ALine-S and ALine-D provide precise OOD performance estimates in finetuned FMs, surpassing other baselines by a large margin. This is noteworthy, especially for shifts where the agreement line is significantly off y = x, further lending to the utility of this method. Furthermore, they perform better on the question-answering task SQuAD, with the next best confidence method achieving as large as 20% higher error.

# 6 Limitations

Estimating the out-of-distribution performance of foundation models has rapidly grown in importance, especially as these models are increasingly deployed in real-world use cases. Our work focuses on a promising method to enable deployers to reduce the harm of machine learning systems when they encounter OOD inputs. However, deployers should be careful to not use AGL as the only signal for OOD performance. The correlation between agreement and accuracy is not guaranteed to hold for all distribution shifts, so other metrics should additionally be used to monitor model performance. In particular for foundation models, we observe that careful choices during fine-tuning is required to observe AGL. In fact, if different pretrained checkpoints are evaluated zero-shot, without any fine-tuning, AGL is not able to reliably predict the performance of FMs (Appendix A.3.5). Furthermore, while we studied AGL closely for a wide array of classification/QA benchmarks, there remains other important downstream tasks such as long-form generation that we leave for future study. We also do not provide any theoretical guarantees to back our empirical findings, such as the importance of random initialization, which we leave for future work.

# 7 Conclusion

We develop methods for extending AGL to foundation models to enable OOD performance prediction in this emerging paradigm. We find that utilizing AGL for performance estimation requires a careful tuning of ensemble diversity. Unlike the original paradigm of AGL, where models observed tens or hundreds of epochs of training on the ID dataset, we find that randomness in specific optimization choices, especially linear head initialization, is crucial for foundation models. In fact, in contrast to $[2]$ , we find that linear models can also observe AGL, specifically in the CLIP representation space, suggesting that AGL may not be a neural network specific phenomena. Our conclusion on AGL also sheds light on the robustness of foundation models. First, our experiments show that light finetuning alone can corrupt models to have diverse behaviors. Next, in contrast to vision models, where previous works show different forms of pretraining lead to different slopes in the linear correlations $[43]$ , we find that all the language models we evaluate, e.g., OPT, GPT2, GPT2-Neo, Alpaca, Llama, and Vicuna lie on the same accuracy line. This is particularly intriguing because it goes against the common wisdom that the pretraining data influences the models' “effective robustness”. We leave these questions for future analysis.

# References

[1] Anas Awadalla, Mitchell Wortsman, Gabriel Ilharco, Sewon Min, Ian Magnusson, Hannaneh Hajishirzi, and Ludwig Schmidt. Exploring the landscape of distributional robustness for question answering models. arXiv preprint arXiv:2210.12517, 2022.   
[2] Christina Baek, Yiding Jiang, Aditi Raghunathan, and J Zico Kolter. Agreement-on-the-line: Predicting the performance of neural networks under distribution shift. Advances in Neural Information Processing Systems, 35:19274–19289, 2022.   
[3] Jason Baumgartner, Savvas Zannettou, Brian Keegan, Megan Squire, and Jeremy Blackburn. The pushshift reddit dataset. In Proceedings of the international AAAI conference on web and social media, volume 14, pages 830–839, 2020.   
[4] Shai Ben-David, John Blitzer, Koby Crammer, and Fernando Pereira. Analysis of representations for domain adaptation. Advances in neural information processing systems, 19, 2006.   
[5] Sid Black, Leo Gao, Phil Wang, Connor Leahy, and Stella Biderman. GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow, 2021. URL https://doi.org/10.5281/zenodo.5297715.   
[6] Samuel R Bowman, Gabor Angeli, Christopher Potts, and Christopher D Manning. A large annotated corpus for learning natural language inference. arXiv preprint arXiv:1508.05326, 2015.   
[7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

[8] Jiefeng Chen, Frederick Liu, Besim Avci, Xi Wu, Yingyu Liang, and Somesh Jha. Detecting errors and estimating accuracy on unlabeled data with self-training ensembles. arXiv preprint arXiv:2106.15728, 2021.   
[9] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality, March 2023. URL https://lmsys.org/blog/2023-03-30-vicuna/.   
[10] Ching-Yao Chuang, Antonio Torralba, and Stefanie Jegelka. Estimating generalization under distribution shifts via domain-invariant representations. arXiv preprint arXiv:2007.03511, 2020.   
[11] Corinna Cortes, Yishay Mansour, and Mehryar Mohri. Learning bounds for importance weighting. Advances in neural information processing systems, 23, 2010.   
[12] Weijian Deng and Liang Zheng. Are labels always necessary for classifier accuracy evaluation? In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15064-15073. IEEE Computer Society, 2021. doi: 10.1109/CVPR46437.2021.01482.   
[13] Weijian Deng, Stephen Gould, and Liang Zheng. What does rotation prediction tell us about classifier accuracy under varying testing environments? arXiv preprint arXiv:2106.05961, 2021.   
[14] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.   
[15] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.   
[16] Hady Elsahar and Matthias Gallé. To annotate or not? predicting performance drop under domain shift. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2163–2173, 2019.   
[17] Alex Fang, Gabriel Ilharco, Mitchell Wortsman, Yuhao Wan, Vaishaal Shankar, Achal Dave, and Ludwig Schmidt. Data determines distributional robustness in contrastive language image pre-training (clip). In International Conference on Machine Learning, pages 6216–6234. PMLR, 2022.   
[18] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. The pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020.   
[19] Saurabh Garg, Sivaraman Balakrishnan, Zachary C Lipton, Behnam Neyshabur, and Hanie Sedghi. Leveraging unlabeled data to predict out-of-distribution performance. International Conference on Learning Representations, 2022.   
[20] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural information processing systems, 33:21271–21284, 2020.   
[21] Devin Guillory, Vaishaal Shankar, Sayna Ebrahimi, Trevor Darrell, and Ludwig Schmidt. Predicting with confidence on unseen distributions. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1134–1144, 2021.   
[22] Dan Hendrycks and Thomas G. Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. In 7th International Conference on Learning Representations, ICLR, 2019.   
[23] Dan Hendrycks and Kevin Gimpel. A baseline for detecting misclassified and out-of-distribution examples in neural networks. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings, 2017.

[24] Dan Hendrycks and Kevin Gimpel. A baseline for detecting misclassified and out-of-distribution examples in neural networks. International Conference on Learning Representations, 2017.   
[25] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.   
[26] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, July 2021. URL https://doi.org/10.5281/zenodo.5143773. If you use this software, please cite it as below.   
[27] Yiding Jiang, Vaishnavh Nagarajan, Christina Baek, and J Zico Kolter. Assessing generalization of sgd via disagreement. International Conference on Learning Representations, 2022.   
[28] Pang Wei Koh, Shiori Sagawa, Henrik Marklund, Sang Michael Xie, Marvin Zhang, Akshay Balsubramani, Weihua Hu, Michihiro Yasunaga, Richard Lanas Phillips, Irena Gao, et al. Wilds: A benchmark of in-the-wild distribution shifts. In International Conference on Machine Learning, pages 5637–5664. PMLR, 2021.   
[29] Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. Technical Report, 2009.   
[30] Ilja Kuzborskij and Francesco Orabona. Stability and hypothesis transfer learning. In International Conference on Machine Learning, pages 942–950. PMLR, 2013.   
[31] Donghwan Lee, Behrad Moniri, Xinmeng Huang, Edgar Dobriban, and Hamed Hassani. Demystifying disagreement-on-the-line in high dimensions, 2023.   
[32] Patrick Lewis, Barlas Oğuz, Ruty Rinott, Sebastian Riedel, and Holger Schwenk. Mlqa: Evaluating cross-lingual extractive question answering. arXiv preprint arXiv:1910.07475, 2019.   
[33] Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin A Raffel. Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning. Advances in Neural Information Processing Systems, 35:1950–1965, 2022.   
[34] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.   
[35] Omid Madani, David Pennock, and Gary Flake. Co-validation: Using model disagreement on unlabeled data to validate classification algorithms. Advances in neural information processing systems, 17, 2004.   
[36] Andrey Malinin, Neil Band, Ganshin, Alexander, German Chesnokov, Yarin Gal, Mark J. F. Gales, Alexey Noskov, Andrey Ploskonosov, Liudmila Prokhorenkova, Ivan Provilkov, Vatsal Raina, Vyas Raina, Roginskiy, Denis, Mariya Shmatova, Panos Tigas, and Boris Yangel. Shifts: A dataset of real distributional shift across multiple large-scale tasks, 2022.   
[37] Yishay Mansour, Mehryar Mohri, and Afshin Rostamizadeh. Domain adaptation: Learning bounds and algorithms. arXiv preprint arXiv:0902.3430, 2009.   
[38] John Miller, Karl Krauth, Benjamin Recht, and Ludwig Schmidt. The effect of natural distribution shift on question answering models. In International conference on machine learning, pages 6905–6916. PMLR, 2020.   
[39] John P Miller, Rohan Taori, Aditi Raghunathan, Shiori Sagawa, Pang Wei Koh, Vaishaal Shankar, Percy Liang, Yair Carmon, and Ludwig Schmidt. Accuracy on the line: on the strong correlation between out-of-distribution and in-distribution generalization. In International Conference on Machine Learning, pages 7721–7735. PMLR, 2021.   
[40] Preetum Nakkiran and Yamini Bansal. Distributional generalization: A new kind of generalization. arXiv preprint arXiv:2009.08092, 2020.

[41] Aleksandr Podkopaev and Aaditya Ramdas. Distribution-free uncertainty quantification for classification under label shift. In Uncertainty in artificial intelligence, pages 844–853. PMLR, 2021.   
[42] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.   
[43] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.   
[44] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions for machine comprehension of text. arXiv preprint arXiv:1606.05250, 2016.   
[45] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do cifar-10 classifiers generalize to cifar-10? arXiv preprint arXiv:1806.00451, 2018.   
[46] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do imagenet classifiers generalize to imagenet? In International conference on machine learning, pages 5389–5400. PMLR, 2019.   
[47] Rebecca Roelofs, Vaishaal Shankar, Benjamin Recht, Sara Fridovich-Keil, Moritz Hardt, John Miller, and Ludwig Schmidt. A meta-analysis of overfitting in machine learning. Advances in Neural Information Processing Systems, 32, 2019.   
[48] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael S. Bernstein, Alexander C. Berg, and Li Fei-Fei. Imagenet large scale visual recognition challenge. CoRR, abs/1409.0575, 2014. URL http://arxiv.org/abs/1409.0575.   
[49] Sebastian Schelter, Tammo Rukat, and Felix Biessmann. Learning to validate the predictions of black box classifiers on unseen data. In Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data, page 1289–1299, New York, NY, USA, 2020. Association for Computing Machinery. ISBN 9781450367356.   
[50] Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning, pages 6105–6114. PMLR, 2019.   
[51] Rohan Taori, Achal Dave, Vaishaal Shankar, Nicholas Carlini, Benjamin Recht, and Ludwig Schmidt. Measuring robustness to natural distribution shifts in image classification. Advances in Neural Information Processing Systems, 33:18583–18599, 2020.   
[52] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford\_alpaca, 2023.   
[53] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023.

[54] Dustin Tran, Jeremiah Liu, Michael W. Dusenberry, Du Phan, Mark Collier, Jie Ren, Kehang Han, Zi Wang, Zelda Mariet, Huiyi Hu, Neil Band, Tim G. J. Rudner, Karan Singhal, Zachary Nado, Joost van Amersfoort, Andreas Kirsch, Rodolphe Jenatton, Nithum Thain, Honglin Yuan, Kelly Buchanan, Kevin Murphy, D. Sculley, Yarin Gal, Zoubin Ghahramani, Jasper Snoek, and Balaji Lakshminarayanan. Plex: Towards reliability using pretrained large model extensions, 2022.   
[55] Trieu H Trinh and Quoc V Le. A simple method for commonsense reasoning. arXiv preprint arXiv:1806.02847, 2018.   
[56] Hemanth Venkateswara, Jose Eusebio, Shayok Chakraborty, and Sethuraman Panchanathan. Deep hashing network for unsupervised domain adaptation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 5018–5027, 2017.   
[57] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In ICLR, 2018.   
[58] Dequan Wang, Xiaosong Wang, Lilong Wang, Mengzhang Li, Qian Da, Xiaoqiang Liu, Xiangyu Gao, Jun Shen, Junjun He, Tian Shen, et al. Medfmc: A real-world dataset and benchmark for foundation model adaptation in medical image classification. arXiv preprint arXiv:2306.09579, 2023.   
[59] Adina Williams, Nikita Nangia, and Samuel R. Bowman. A broad-coverage challenge corpus for sentence understanding through inference, 2018.   
[60] Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs, Raphael Gontijo Lopes, Hannaneh Hajishirzi, Ali Farhadi, Hongseok Namkoong, et al. Robust fine-tuning of zero-shot models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7959–7971, 2022.   
[61] Chhavi Yadav and Léon Bottou. Cold case: The lost mnist digits. Advances in neural information processing systems, 32, 2019.   
[62] Yaodong Yu, Stephen Bates, Yi Ma, and Michael Jordan. Robust calibration with multi-domain temperature scaling. Advances in Neural Information Processing Systems, 35:27510–27523, 2022.   
[63] Yaodong Yu, Zitong Yang, Alexander Wei, Yi Ma, and Jacob Steinhardt. Predicting out-of-distribution error with the projection norm, 2022.   
[64] Elad Ben Zaken, Shauli Ravfogel, and Yoav Goldberg. Bitfit: Simple parameter-efficient fine-tuning for transformer-based masked language-models. arXiv preprint arXiv:2106.10199, 2021.   
[65] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. Opt: Open pre-trained transformer language models, 2022.   
[66] Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In Proceedings of the IEEE international conference on computer vision, pages 19–27, 2015.   
[67] Yuli Zou, Weijian Deng, and Liang Zheng. Adaptive calibrator ensemble for model calibration under distribution shift. arXiv preprint arXiv:2303.05331, 2023.

# A Appendix

# Contents

A.1 Details Regarding OOD Performance Estimation Baselines ..... 16

A.1.1 AGL-Based Estimation Methods: ALine-S/D 16   
A.1.2 Temperature Scaling for Confidence-Based Estimation Methods ..... 17   
A.1.3 Comparison to ProjNorm 17   
A.1.4 Failure Datasets with Low Linear Correlation ..... 18

A.2 Finetuning Hyperparameters 19

A.2.1 Linear Probing over CLIP for Vision Tasks 19   
A.2.2 Full Finetuning GPT, OPT, BERT on Language Tasks 19

A.3 More Experiments on the Effect of Diversity Source with CLIP ..... 21

A.3.1 OfficeHome Linear Probing Diversity Experiments ..... 21   
A.3.2 AGL appears in Random Head CLIP Ensembles across Datasets ..... 22   
A.3.3 Diversity Matters in Full Finetuned CLIP Ensembles 26   
A.3.4 Any diverse ensemble display AGL in models heavily trained from scratch 27   
A.3.5 AGL in FMs under Zero-Shot and Few-Shot Settings 27   
A.3.6 Effect of Training Dataset Size 28

A.4 More Experiments on the Effect of Diversity Source for Extractive Question Answering 30

A.4.1 Single-Base Diversity Experiments Using OPT and BERT ..... 30   
A.4.2 Diversity in Full Finetuned GPT2 for Different Data Portions ..... 31   
A.4.3 Diversity in Full Finetuned OPT for Different Data Portions ..... 34   
A.4.4 Diversity in Full Finetuned BERT for Different Data Portions ..... 38

A.5 More Experiments on the Effect of Diversity Source for Text Classification ..... 42

A.5.1 Diversity Source for Linear Probing 42

A.6 Generative Question-Answering 43

A.6.1 Experiments on the effect of Diversity Source ..... 43   
A.6.2 Experiments starting from multiple foundation models ..... 45

A.7 Diversity under different learning rates and batch sizes 46   
A.8 Diversity Experiments for different PEFT methods 47

# A.1 Details Regarding OOD Performance Estimation Baselines

# A.1.1 AGL-Based Estimation Methods: ALine-S/D

ALine algorithms are the AGL-based performance estimation methods proposed in $[2]$ . When the AGL phenomenon occurs, i.e., models observe a strong linear correlation in both ID versus OOD agreement and accuracy with matching slopes and biases, algorithms ALine-S and ALine-D effectively apply the linear transformation calculated using agreements to map the ID performances to OOD performance estimates. We describe the algorithms in more detail below.

AGL Provided a collection of models $F = \{f_{1}, f_{2}, ..., f_{n}\}$ , AGL suggests that ID versus OOD accuracy observe a strong linear correlation if and only if ID versus OOD agreement observes a strong linear correlation and when they do, the slopes and biases match: $\forall f_{i}, f_{j} \in F$ where $i \neq j$

$$
\Phi^ {- 1} \left(\operatorname{Acc} _ {\mathrm{OOD}} \left(f _ {i}\right)\right) = a \cdot \Phi^ {- 1} \left(\operatorname{Acc} _ {\mathrm{ID}} \left(f _ {i}\right)\right) + b
$$

$\Updownarrow$ (3)

$$
\Phi^ {- 1} (\mathrm{Agr} _ {\mathrm{OOD}} (f _ {i}, f _ {j})) = a \cdot \Phi^ {- 1} (\mathrm{Agr} _ {\mathrm{ID}} (f _ {i}, f _ {j})) + b
$$

$\Phi^{-1}$ is the probit transform used to induce a better linear fit as used in [2] and [39]. Provided access to $\mathrm{Acc}_{\mathrm{ID}}(f_i), \mathrm{Agr}_{\mathrm{ID}}(f_i, f_j), \mathrm{Agr}_{\mathrm{OOD}}(f_i, f_j) \forall i, j$ , we'd like to estimate $\mathrm{Acc}_{\mathrm{OOD}}(f_i)$ for all $f_i \in \mathcal{F}$ .

ALine-S The algorithm ALine-S simply estimates the slope $a$ and bias $b$ of accuracy by computing the linear fit of agreement.

$$
\hat {a}, \hat {b} = \arg \min _ {a, b \in \mathbb {R}} \sum_ {i \neq j} \left(\Phi^ {- 1} (\hat {\mathrm{Agr}} _ {\mathrm{OOD}} (f _ {i}, f _ {j})) - a \cdot \Phi^ {- 1} (\hat {\mathrm{Agr}} _ {\mathrm{ID}} (f _ {i}, f _ {j})) - b\right) ^ {2} \tag {4}
$$

With $\hat{a}$ and $\hat{b}$ , we estimate $\operatorname{Acc}_{\operatorname{OOD}}(f_{i}) \approx \hat{a} \cdot \operatorname{Acc}_{\operatorname{ID}}(f_{i}) + \hat{b}$ . This method is called Aline-S.

ALine-D This method instead constructs the following system of linear equations. Provided the relation in Equation 2, one can derive that for any $f_{i}, f_{j} \in \mathcal{F}$ ,

$$
\frac {1}{2} \left(\Phi^ {- 1} (\mathrm{Acc} _ {\mathrm{OOD}} (f _ {i})) + \Phi^ {- 1} (\mathrm{Acc} _ {\mathrm{OOD}} (f _ {j}))\right)
$$

$$
\approx \Phi^ {- 1} (\mathrm{Agr} _ {\mathrm{OOD}} (f _ {i}, f _ {j})) + \hat {a} \cdot \left(\frac {1}{2} \Phi^ {- 1} (\mathrm{Acc} _ {\mathrm{ID}} (f _ {i})) + \frac {1}{2} \Phi^ {- 1} (\mathrm{Acc} _ {\mathrm{ID}} (f _ {j})) - \Phi^ {- 1} (\mathrm{Agr} _ {\mathrm{ID}} (f _ {i}, f _ {j}))\right) \tag {5}
$$

Treating $\operatorname{Acc}_{\operatorname{OOD}}(f_{i}) \forall i$ as unknown variables, note that the right hand side is known and we can construct a linear system of equations using all $\binom{n}{2}$ pairs of models. The algorithm employs least squares to solve this approximate system of linear equations.

# A.1.2 Temperature Scaling for Confidence-Based Estimation Methods

We compare ALine against confidence-based methods ATC [19], AC [24], and Doc-Feat [21]. These methods notably perform better after calibrating the models ID by temperature scaling.

Classification For classification tasks, we optimize a temperature T for each model f on the cross-entropy loss over the in-distribution validation data.

$$
\min _ {T} \sum_ {x, y} \mathsf {C E} (\sigma (f (x) \exp (T)), y) \tag {6}
$$

where $\sigma (\cdot)$ is the softmax.

Question-Answering For extractive question-answering tasks, the model has to predict two labels - the start and end token index $y = [y_s, y_e]$ of the context span that answers the question. For each model, we attach a span prediction head $v = [v_s, v_e] \in \mathbb{R}^{d \times 2}$ on top of the base $\mathrm{B}_{\theta}(x) \in \mathbb{R}^{d \times N}$ where $N$ is the token length of $x$ . $s(x) = v_s^\top \mathrm{B}_{\theta}(x)$ and $e(x) = v_e^\top \mathrm{B}_{\theta}(x)$ predict the start and end token index, respectively.

We're interested in evaluating question-answering models on the exact match (EM) objective,

$$
\mathsf {E M} (\hat {y}, y) = \mathbb {1} \left[ \hat {y} _ {s} = y _ {s} \right] \cdot \mathbb {1} \left[ \hat {y} _ {e} = y _ {e} \right] \tag {7}
$$

EM treats question-answering as a classification problem over $N \times N$ choices of start and end index pairs. This allows us to utilize confidence-based methods that are designed for classification tasks. We can calculate the model confidence for index pair $[i, j]$ as $\sigma(s(x))_{i} \cdot \sigma(e(x))_{j}$ .

We jointly optimize a separate temperature for the start and end logits $T_{s}$ and $T_{e}$ and we minimize the cross-entropy loss over the in-distribution validation data.

$$
\min _ {T} \sum_ {x, y} \mathsf {C E} (\sigma (s (x) \exp (T _ {s})) \sigma (e (x) \exp (T _ {e})) ^ {\top}, y) \tag {8}
$$

# A.1.3 Comparison to ProjNorm

In this section, we present a comparison with ProjNorm [63], a method that yields a score which is shown to be correlated with the OOD performance of the model. We study the same setting as presented in Section 4 where we estimate OOD performance of foundation models pretrained on different text corpora. Unlike AGL, ProjNorm doesn't provide an estimate of the OOD performance hence we compare the linear correlation between the predicted value and OOD performance. From Table 4 it can be seen that estimates from ALine-D are more strongly correlated with OOD performance than ProjNorm.

Table 4: Correlation coefficient between OOD Accuracy and Prediction for ALine-D and ProjNorm 

<table><tr><td>Method</td><td>SQuAD-Shifts Amazon</td><td>SQuAD-Shifts Reddit</td></tr><tr><td>ALine-D</td><td>0.98</td><td>0.98</td></tr><tr><td>ProjNorm</td><td>0.64</td><td>0.79</td></tr></table>

# A.1.4 Failure Datasets with Low Linear Correlation

In our comparison with baselines in Section 5, we filter out datasets with a low correlation coefficient $\leq 0.95$ in ID vs OOD agreement. When the linear correlation of agreement is weak, AGL tells us that the correlation is also low for accuracy, and AGL-based methods are not guaranteed to be reliable in such circumstances. We provide ID vs OOD accuracy and agreement scatter plots for all datasets in Appendix A.3.2.

Below, we separately provide the comparison with baselines for the excluded datasets. We generally find that the baseline ATC [19] is significantly better in circumstances where AGL-based methods are unreliable.

Table 5: The MAPE (%) of predicting OOD performance using AGL-based ALine and other baseline methods. We collect a diverse ensemble by randomizing the linear initialization and including multiple base models. 

<table><tr><td>OOD Dataset</td><td>Agreement  $R^{2}$ </td><td>ALine-D</td><td>ALine-S</td><td>Naive Agr</td><td>ATC</td><td>AC</td><td>DOC-Feat</td></tr><tr><td>CIFAR10C Gaussian Noise</td><td>0.85</td><td>65.59</td><td>56.67</td><td>41.85</td><td>42.77</td><td>74.60</td><td>75.40</td></tr><tr><td>CIFAR10C Glass Blur</td><td>0.89</td><td>44.24</td><td>44.97</td><td>31.24</td><td>33.58</td><td>79.45</td><td>80.36</td></tr><tr><td>CIFAR10C Shot Noise</td><td>0.89</td><td>45.02</td><td>37.60</td><td>27.70</td><td>28.25</td><td>49.23</td><td>49.90</td></tr><tr><td>CIFAR10C Speckle Noise</td><td>0.90</td><td>36.67</td><td>30.13</td><td>23.45</td><td>22.72</td><td>43.46</td><td>44.12</td></tr><tr><td>CIFAR100C Gaussian Noise</td><td>0.93</td><td>34.98</td><td>32.21</td><td>28.88</td><td>21.18</td><td>69.04</td><td>63.81</td></tr><tr><td>CIFAR100C Glass Blur</td><td>0.93</td><td>66.03</td><td>63.45</td><td>42.41</td><td>25.47</td><td>109.73</td><td>103.70</td></tr><tr><td>ImageNetC Gaussian Noise</td><td>0.89</td><td>50.10</td><td>47.26</td><td>73.47</td><td>13.21</td><td>54.86</td><td>42.66</td></tr><tr><td>ImageNetC Glass Blur</td><td>0.87</td><td>74.20</td><td>71.84</td><td>100.32</td><td>15.97</td><td>74.12</td><td>59.60</td></tr><tr><td>ImageNetC Impulse Noise</td><td>0.88</td><td>62.86</td><td>59.68</td><td>87.83</td><td>16.54</td><td>63.23</td><td>49.99</td></tr><tr><td>ImageNetC Shot Noise</td><td>0.88</td><td>53.89</td><td>51.12</td><td>78.37</td><td>14.66</td><td>57.47</td><td>44.58</td></tr><tr><td>iWildCam-WILDS</td><td>0.85</td><td>22.05</td><td>25.29</td><td>46.42</td><td>37.25</td><td>57.31</td><td>69.58</td></tr><tr><td>Camelyon17-WILDS</td><td>0.59</td><td>10.14</td><td>6.44</td><td>13.26</td><td>6.46</td><td>8.76</td><td>8.90</td></tr></table>

# A.2 Finetuning Hyperparameters

We state here the hyperparameters used to finetune the models for diversity experiments reported in Section 3.

# A.2.1 Linear Probing over CLIP for Vision Tasks

We train all linear probes using SGD. Models are trained for different timesteps to achieve an even distribution of ID accuracies.

Table 6: CLIP Linear Probing 

<table><tr><td>Dataset</td><td>Hyperparameters</td></tr><tr><td>CIFAR10</td><td>Learning Rate:  $5 \times 10^{-4}$ Batch Size: 1028</td></tr><tr><td>CIFAR100</td><td>Learning Rate:  $1 \times 10^{-3}$ Batch Size: 1028</td></tr><tr><td>ImageNet</td><td>Learning Rate:  $1 \times 10^{-1}$ Batch Size: 1028</td></tr><tr><td>OfficeHome</td><td>Learning Rate:  $1 \times 10^{-3}$ Batch Size: 200</td></tr><tr><td>FMoW-WILDS</td><td>Learning Rate:  $1 \times 10^{-3}$ Batch Size: 200</td></tr><tr><td>iWildCam-WILDS</td><td>Learning Rate:  $1 \times 10^{-3}$ Batch Size: 200</td></tr><tr><td>Camelyon17-WILDS</td><td>Learning Rate:  $1 \times 10^{-3}$ Batch Size: 200</td></tr></table>

# A.2.2 Full Finetuning GPT, OPT, BERT on Language Tasks

We use AdamW [34] to full finetune language models. We keep the learning rate small. Models are trained for different timesteps to achieve an even distribution of ID accuracies.

Table 7: Full Finetuning GPT2-Medium on SQuAD 

<table><tr><td>Source of Diversity</td><td>Hyperparameters</td></tr><tr><td>Initialization + Ordering</td><td>Learning rate:  $2 \times 10^{-7}$ Weight Decay:  $1 \times 10^{-5}$ Batch Size: 32Max Epochs: 20</td></tr><tr><td>Subsetting (10, 30 % of data)</td><td>Learning rate:  $6,4 \times 10^{-7}$ Weight Decay:  $1 \times 10^{-5}$ Batch Size:Max Epochs: 20</td></tr></table>

Table 8: Full Finetuning OPT-125M on SQuAD 

<table><tr><td>Source of Diversity</td><td>Hyperparameters</td></tr><tr><td>Initialization + Ordering</td><td>Learning rate:  $4 \times 10^{-7}$ Weight Decay:  $1 \times 10^{-5}$ Batch Size: 32Max Epochs: 20</td></tr><tr><td>Subsetting (10, 30, 50 % of data)</td><td>Learning rate:  $40, 12, 8 \times 10^{-7}$ Weight Decay:  $1 \times 10^{-5}$ Batch Size: 32Max Epochs: 20</td></tr></table>

Table 9: Full Finetuning BERT-Uncased on SQuAD 

<table><tr><td>Source of Diversity</td><td>Hyperparameters</td></tr><tr><td>Initialization + Ordering</td><td>Learning rate:  $2 \times 10^{-7}$ Weight Decay:  $1 \times 10^{-5}$ Batch Size: 32Max Epochs: 20</td></tr><tr><td>Subsetting (10, 30, 50 % of data)</td><td>Learning rate:  $20, 6, 4 \times 10^{-7}$ Weight Decay:  $1 \times 10^{-5}$ Batch Size: 32Max Epochs: 20</td></tr></table>

Table 10: Full Finetuning GPT2-Medium on MNLI 

<table><tr><td>Source of Diversity</td><td>Hyperparameters</td></tr><tr><td>Initialization + Ordering</td><td>Learning rate:  $5 \times 10^{-4}$ Weight Decay:  $1 \times 10^{-5}$ Batch Size: 128Max Epochs: 10</td></tr><tr><td>Subsetting (10% of data)</td><td>Learning rate:  $5 \times 10^{-3}$ Weight Decay:  $1 \times 10^{-5}$ Batch Size: 128Max Epochs: 10</td></tr></table>

Table 11: Full Finetuning OPT-125M on MNLI 

<table><tr><td>Source of Diversity</td><td>OPT-125MVaried</td><td>Fixed</td></tr><tr><td>Initialization + Ordering</td><td>Learning rate:  $1 \times 10^{-3}$ Weight Decay:  $1 \times 10^{-5}$ Batch Size: 128Max Epochs: 10</td><td></td></tr><tr><td>Subsetting (10% of data)</td><td>Learning rate:  $1 \times 10^{-2}$ Weight Decay:  $1 \times 10^{-5}$ Batch Size: 128Max Epochs: 10</td><td></td></tr></table>

# A.3 More Experiments on the Effect of Diversity Source with CLIP

# A.3.1 OfficeHome Linear Probing Diversity Experiments

In Section 3.1, we examine how diversity source impacts whether AGL is observed in linear probed CLIP models from CIFAR10 to CIFAR10C. Here, we perform the same experiment on Office-Home [56], which consists of 4 domains or image styles (“Art”, “Clip Art”, “Product”, and “Real World”) for 65 common objects. We train models on one domain and treat the remaining three domains as OOD. Similarly, only Random Initialization yields AGL or matching slopes in accuracy and agreement, and as a result, the corresponding MAPE of estimating the OOD performance of this diverse ensemble is the smallest.

![](images/b9c5b57b80ba5eb69d506a4d357cfdd8c78eb8596cd89330fa1441e12cbf0d1b.jpg)

Figure 4: ID vs OOD accuracy and agreement of linear probed CLIP models on OfficeHome Art (top row), Product (middle row), and Real World (bottom row). The figure title is the OOD domain.   
Table 12: The ALine-S MAPE(%) for ensembles trained on each domain of OfficeHome. 

<table><tr><td>OfficeHome ID domain</td><td>Source of Diversity</td><td>OOD Estimation MAPE(%)</td></tr><tr><td rowspan="3">Art</td><td>Random Linear Heads</td><td>14.09</td></tr><tr><td>Data Ordering</td><td>20.67</td></tr><tr><td>Data Subsetting</td><td>120.85</td></tr><tr><td rowspan="3">ClipArt</td><td>Random Linear Heads</td><td>12.23</td></tr><tr><td>Data Ordering</td><td>28.45</td></tr><tr><td>Data Subsetting</td><td>78.49</td></tr><tr><td rowspan="3">Product</td><td>Random Linear Heads</td><td>13.63</td></tr><tr><td>Data Ordering</td><td>110.45</td></tr><tr><td>Data Subsetting</td><td>92.97</td></tr><tr><td rowspan="3">Real</td><td>Random Linear Heads</td><td>9.95</td></tr><tr><td>Data Ordering</td><td>33.36</td></tr><tr><td>Data Subsetting</td><td>3078</td></tr></table>

# A.3.2 AGL appears in Random Head CLIP Ensembles across Datasets

We report the strength of AGL in linear probed CLIP models with randomly initialized linear heads across all datasets discussed in Section 3.2.

![](images/5f5156f52924421971f532346d0e668d9cafd252521b64b2e4d2d2e73d08ed6e.jpg)  
Figure 5: AGL and ACL for all C10C shifts with random head initialization finetuning.

![](images/e766fd3c73fd6d1ec9d5638b5640e5ba8d95a2a85ab66de26935c59746908d4f.jpg)

<details>
<summary>line</summary>

| ID | Accuracy | Agreement |
|----|----------|-----------|
| 10 | 10       | 10        |
| 20 | 25       | 25        |
| 30 | 35       | 35        |
| 40 | 45       | 45        |
| 50 | 55       | 55        |
| 60 | 65       | 65        |
| 70 | 75       | 75        |
| 80 | 85       | 85        |
| 90 | 90       | 90        |
</details>

![](images/47d0b72a78b94016138c16d19714bc491bdda80b172331dc839b46643fd8cdfc.jpg)

<details>
<summary>scatter</summary>

| ID | OOD | Type     |
|----|-----|----------|
| 10 | 10  | Accuracy |
| 20 | 20  | Agreement|
| 30 | 30  | Accuracy |
| 40 | 40  | Agreement|
| 50 | 50  | Accuracy |
| 60 | 60  | Agreement|
| 70 | 70  | Accuracy |
| 80 | 80  | Agreement|
| 90 | 90  | Accuracy |
</details>

Figure 6: AGL and ACL for the C10.1 shifts with random head initialization finetuning.

![](images/feea0ccb2f19444b1f8a10aad94c7541183d956b41e3d1dd3fb0bd02daf0bc5c.jpg)

<details>
<summary>line</summary>

| ID | OOD |
| --- | --- |
| 5 | 5 |
| 10 | 10 |
| 30 | 30 |
| 70 | 70 |
| 90 | 90 |
</details>

![](images/c34afd84d72523950b29886ae71e4b963000c925871854240ae45157fb4c17f5.jpg)

<details>
<summary>scatter</summary>

| ID | OOD | Type     |
|----|-----|----------|
| 5  | 3   | Accuracy |
| 10 | 10  | Agreement|
| 20 | 20  | Accuracy |
| 30 | 30  | Agreement|
| 40 | 40  | Accuracy |
| 50 | 50  | Agreement|
| 60 | 60  | Accuracy |
| 70 | 70  | Agreement|
| 80 | 80  | Accuracy |
| 90 | 90  | Agreement|
</details>

![](images/b80a8deaff97bfe62f61e69c347a2c1f25ce6512ee0d959507572f281a1ca836.jpg)

<details>
<summary>scatter</summary>

| ID | OOD | Accuracy |
|----|-----|----------|
| 5  | 3   | 3        |
| 10 | 10  | 10       |
| 20 | 20  | 20       |
| 30 | 30  | 30       |
| 40 | 40  | 40       |
| 50 | 50  | 50       |
| 60 | 60  | 60       |
| 70 | 70  | 70       |
| 80 | 80  | 80       |
| 90 | 90  | 90       |
</details>

![](images/0915048307311420a422f42b599021dfac0eea0fc503b4212f0b1aeec9baf02d.jpg)

<details>
<summary>scatter</summary>

| ID | OOD | Type     |
|----|-----|----------|
| 5  | 5   | Accuracy |
| 10 | 10  | Agreement|
| 20 | 20  | Accuracy |
| 30 | 30  | Agreement|
| 40 | 40  | Accuracy |
| 50 | 50  | Agreement|
| 60 | 60  | Accuracy |
| 70 | 70  | Agreement|
| 80 | 80  | Accuracy |
| 90 | 90  | Agreement|
</details>

![](images/bd6ddbc74340d2c1cdfcaf224f8393c7222f41478cc613261103b02dbf8ca9b6.jpg)

<details>
<summary>scatter</summary>

| ID | OOD | Accuracy |
|----|-----|----------|
| 5  | 5   | 5        |
| 10 | 10  | 10       |
| 20 | 20  | 20       |
| 30 | 30  | 30       |
| 40 | 40  | 40       |
| 50 | 50  | 50       |
| 60 | 60  | 60       |
| 70 | 70  | 70       |
| 80 | 80  | 80       |
| 90 | 90  | 90       |
</details>

![](images/a52f062830b15d3926dbb3557b6d1617acf74e91c30141c59b5491e9db957b46.jpg)

<details>
<summary>scatter</summary>

| ID | OOD | Type     |
|----|-----|----------|
| 5  | 5   | Accuracy |
| 10 | 10  | Agreement|
| 30 | 30  | Accuracy |
| 60 | 60  | Agreement|
| 70 | 70  | Accuracy |
| 80 | 80  | Agreement|
| 90 | 90  | Accuracy |
</details>

![](images/eeace1941e9f72b248c41f1a0fc92f58db70816b0fdecd721e545455fa77e5a0.jpg)

<details>
<summary>line</summary>

| ID | OOD (Accuracy) | OOD (Agreement) |
|----|----------------|-----------------|
| 0  | 0              | 0               |
| 5  | 5              | 5               |
| 10 | 10             | 10              |
| 15 | 15             | 15              |
| 20 | 20             | 20              |
| 25 | 25             | 25              |
| 30 | 30             | 30              |
| 35 | 35             | 35              |
| 40 | 40             | 40              |
| 45 | 45             | 45              |
| 50 | 50             | 50              |
| 55 | 55             | 55              |
| 60 | 60             | 60              |
| 65 | 65             | 65              |
| 70 | 70             | 70              |
| 75 | 75             | 75              |
| 80 | 80             | 80              |
| 85 | 85             | 85              |
| 90 | 90             | 90              |
</details>

![](images/44ffb15aeeca0a20386b23c7f7645ba450a9ef49ce65710460498622dca424ca.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 5  | 5   |
| 10 | 10  |
| 20 | 15  |
| 30 | 20  |
| 40 | 25  |
| 50 | 30  |
| 60 | 35  |
| 70 | 40  |
| 80 | 45  |
| 90 | 50  |
</details>

![](images/8ec23eaec4d06df9dfd0dbd41d92196416a2ab56917a6548a8a627de2df6463c.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  |
| --- | ---- |
| 5   | 3    |
| 10  | 8    |
| 20  | 12   |
| 30  | 15   |
| 40  | 20   |
| 50  | 25   |
| 60  | 30   |
| 70  | 35   |
| 80  | 40   |
| 90  | 45   |
</details>

![](images/f898b75073dd4db79f9607b092723c86f12fe6af68fc63f2a783db4919bc841f.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 5  | 5   |
| 10 | 10  |
| 15 | 15  |
| 20 | 20  |
| 25 | 25  |
| 30 | 30  |
| 35 | 35  |
| 40 | 40  |
| 45 | 45  |
| 50 | 50  |
| 55 | 55  |
| 60 | 60  |
| 65 | 65  |
| 70 | 70  |
| 75 | 75  |
| 80 | 80  |
| 85 | 85  |
| 90 | 90  |
</details>

![](images/47f8b59f60b598426649029f733d81a5a92764df81b5be9e36c33d43380f0aef.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
| --- | --- |
| 5 | 5 |
| 10 | 10 |
| 20 | 20 |
| 30 | 30 |
| 40 | 40 |
| 50 | 50 |
| 60 | 60 |
| 70 | 70 |
| 80 | 80 |
| 90 | 90 |
</details>

![](images/58e07acc6ddb250f773a00b9033dc9227f190335e2dc7bddabf26d048bbc1ad9.jpg)

<details>
<summary>line</summary>

| ID | OOD | Accuracy |
|----|-----|----------|
| 5  | 5   | 5        |
| 10 | 10  | 10       |
| 15 | 15  | 15       |
| 20 | 20  | 20       |
| 25 | 25  | 25       |
| 30 | 30  | 30       |
| 35 | 35  | 35       |
| 40 | 40  | 40       |
| 45 | 45  | 45       |
| 50 | 50  | 50       |
| 55 | 55  | 55       |
| 60 | 60  | 60       |
| 65 | 65  | 65       |
| 70 | 70  | 70       |
| 75 | 75  | 75       |
| 80 | 80  | 80       |
| 85 | 85  | 85       |
| 90 | 90  | 90       |
</details>

![](images/714b4dba040b0b0c660988df2bc1ca2a00dbbc4c80b00ee1c81975a189172024.jpg)

<details>
<summary>line</summary>

| ID  | Accuracy | Agreement |
| --- | -------- | --------- |
| 5   | 3        | 2         |
| 10  | 8        | 7         |
| 20  | 15       | 14        |
| 30  | 22       | 20        |
| 40  | 30       | 28        |
| 50  | 38       | 36        |
| 60  | 46       | 44        |
| 70  | 54       | 52        |
| 80  | 62       | 60        |
| 90  | 70       | 68        |
</details>

![](images/af30f3848c40bbb04c5e35ff3ba54ad03ad84013649360d4cd150809725e89ce.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  |
| --- | ---- |
| 5   | 5    |
| 10  | 10   |
| 20  | 20   |
| 30  | 30   |
| 40  | 40   |
| 50  | 50   |
| 60  | 60   |
| 70  | 70   |
| 80  | 80   |
| 90  | 90   |
</details>

![](images/d20b52ccec999de7ef22aa81954469c97c6307cdac3a5daf8f483d9b4001eb00.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 5  | 3   |
| 10 | 8   |
| 20 | 15  |
| 30 | 25  |
| 40 | 35  |
| 50 | 45  |
| 60 | 55  |
| 70 | 65  |
| 80 | 75  |
| 90 | 85  |
</details>

![](images/6a96a0fa7009d45c740935a5ec4dcd69735f7eb68de6b0e6f57d3ce52f96f1f5.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 5  | 5   |
| 10 | 10  |
| 20 | 20  |
| 30 | 30  |
| 40 | 40  |
| 50 | 50  |
| 60 | 60  |
| 70 | 70  |
| 80 | 80  |
| 90 | 90  |
</details>

![](images/e4c8518214ef939e4cdac21d851b2c89b0b0e4bd164b82b18ab03b15a9d7eac5.jpg)

<details>
<summary>scatter</summary>

| ID | OOD | Accuracy |
|----|-----|----------|
| 5  | 3   | 3        |
| 10 | 8   | 8        |
| 20 | 15  | 15       |
| 30 | 22  | 22       |
| 40 | 30  | 30       |
| 50 | 38  | 38       |
| 60 | 45  | 45       |
| 70 | 52  | 52       |
| 80 | 60  | 60       |
| 90 | 68  | 68       |
</details>

![](images/c2f2c07e3d1db1c490e9c36de5508b71d4b237fb3cfccb01bb06014ad977a113.jpg)

<details>
<summary>line</summary>

| ID | OOD |
| --- | --- |
| 10 | 5 |
| 20 | 10 |
| 30 | 15 |
| 40 | 20 |
| 50 | 25 |
| 60 | 30 |
| 70 | 35 |
| 80 | 40 |
| 90 | 45 |
</details>

![](images/77ad1aff459dc5a40dec18b96be5d5a43a8c5a17c404fb6685e61a26b287d706.jpg)

<details>
<summary>line</summary>

| ID | OOD (Accuracy) | OOD (Agreement) |
|----|----------------|-----------------|
| 5  | 3              | 2               |
| 10 | 10             | 8               |
| 15 | 15             | 12              |
| 20 | 20             | 16              |
| 25 | 25             | 20              |
| 30 | 30             | 24              |
| 35 | 35             | 28              |
| 40 | 40             | 32              |
| 45 | 45             | 36              |
| 50 | 50             | 40              |
| 55 | 55             | 44              |
| 60 | 60             | 48              |
| 65 | 65             | 52              |
| 70 | 70             | 56              |
| 75 | 75             | 60              |
| 80 | 80             | 64              |
| 85 | 85             | 68              |
| 90 | 90             | 72              |
</details>

Figure 7: AGL and ACL for the C100C shifts with random head initialization finetuning.

![](images/5085a8c04ce54b21ffff1be4376944ea23be844d01c9dc4ef8e8c1ed82d6de2b.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 70 | 65  |
</details>

![](images/80a213f8b568d42f0cc0c32cbf5ea486978a889b1f41ec48e6672a80b4cd975c.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 10 | 5   |
| 20 | 10  |
| 30 | 15  |
| 40 | 20  |
| 50 | 25  |
| 60 | 30  |
| 70 | 35  |
| 80 | 40  |
| 90 | 45  |
</details>

![](images/7f3d2ddacf9b8301e60ad5a652c13dff47d9bae2654bc478693fb48246409e96.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  |
| --- | ---- |
| 5   | 5    |
| 10  | 10   |
| 15  | 15   |
| 20  | 20   |
| 25  | 25   |
| 30  | 30   |
| 35  | 35   |
| 40  | 40   |
| 45  | 45   |
| 50  | 50   |
| 55  | 55   |
| 60  | 60   |
| 65  | 65   |
| 70  | 70   |
| 75  | 75   |
| 80  | 80   |
| 85  | 85   |
| 90  | 90   |
</details>

![](images/0390a59a32a5cb0b1ba4e665096b04fb0587e29a56918eff83e628d9c73e7163.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  |
| --- | ---- |
| 10  | 10   |
| 20  | 20   |
| 30  | 30   |
| 40  | 40   |
| 50  | 50   |
| 60  | 60   |
| 70  | 70   |
| 80  | 80   |
| 90  | 90   |
</details>

![](images/258817be1c337c2faaacc3fb713501f9b0df35ffe0d7edb5e8f4ab4371e8d295.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  |
| --- | ---- |
| 5   | 5    |
| 10  | 10   |
| 20  | 15   |
| 30  | 20   |
| 40  | 25   |
| 50  | 30   |
| 60  | 35   |
| 70  | 40   |
| 80  | 45   |
| 90  | 50   |
</details>

![](images/1b5644d3fca6a33bdfc71819529747b9173640e2857d9c9f0a17bf8ba26cac8d.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
| --- | --- |
| 5 | 5 |
| 10 | 10 |
| 20 | 20 |
| 30 | 30 |
| 40 | 40 |
| 50 | 50 |
| 60 | 60 |
| 70 | 70 |
| 80 | 80 |
| 90 | 90 |
</details>

Figure 8: AGL and ACL for the ImageNetC shifts with random head initialization finetuning.

![](images/ee231abe4fada87617fe4943f5399e52d408e6901fe394070c13fcc0cc566d9f.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 1  | 1   |
| 2  | 2   |
| 3  | 3   |
| 4  | 4   |
| 5  | 5   |
| 6  | 6   |
| 7  | 7   |
| 8  | 8   |
| 9  | 9   |
| 10 | 10  |
| 11 | 11  |
| 12 | 12  |
| 13 | 13  |
| 14 | 14  |
| 15 | 15  |
| 16 | 16  |
| 17 | 17  |
| 18 | 18  |
| 19 | 19  |
| 20 | 20  |
| 21 | 21  |
| 22 | 22  |
| 23 | 23  |
| 24 | 24  |
| 25 | 25  |
| 26 | 26  |
| 27 | 27  |
| 28 | 28  |
| 29 | 29  |
| 30 | 30  |
| 31 | 31  |
| 32 | 32  |
| 33 | 33  |
| 34 | 34  |
| 35 | 35  |
| 36 | 36  |
| 37 | 37  |
| 38 | 38  |
| 39 | 39  |
| 40 | 40  |
| 41 | 41  |
| 42 | 42  |
| 43 | 43  |
| 44 | 44  |
| 45 | 45  |
| 46 | 46  |
| 47 | 47  |
| 48 | 48  |
| 49 | 49  |
| 50 | 50  |
| 51 | 51  |
| 52 | 52  |
| 53 | 53  |
| 54 | 54  |
| 55 | 55  |
| 56 | 56  |
| 57 | 57  |
| 58 | 58  |
| 59 | 59  |
| 60 | 60  |
| 61 | 61  |
| 62 | 62  |
| 63 | 63  |
| 64 | 64  |
| 65 | 65  |
| 66 | 66  |
| 67 | 67  |
| 68 | 68  |
| 69 | 69  |
| 70 | 70  |
| 71 | 71  |
| 72 | 72  |
| 73 | 73  |
| 74 | 74  |
| 75 | 75  |
| 76 | 76  |
| 77 | 77  |
| 78 | 78  |
| 79 | 79  |
| 80 | 80  |
| Note: The actual values for Accuracy and Agreement are not provided in the code. I have used the same label 'OOD' as they are explicitly written in the chart.
</details>

![](images/aacee701429c9c44bcb964f19ebd0bfa8bacd0dd825fd2f621c7f4a663dae7eb.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 0  | 0   |
| 10 | 10  |
| 30 | 30  |
| 50 | 50  |
| 70 | 70  |
| 90 | 90  |
</details>

![](images/cd04b695a2c093245e0938cc635cdf9c18616a025277cb0064856d52ce17f6ef.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  |
| --- | ---- |
| 5   | 5    |
| 10  | 10   |
| 15  | 15   |
| 20  | 20   |
| 25  | 25   |
| 30  | 30   |
| 35  | 35   |
| 40  | 40   |
| 45  | 45   |
| 50  | 50   |
| 55  | 55   |
| 60  | 60   |
| 65  | 65   |
| 70  | 70   |
| 75  | 75   |
| 80  | 80   |
| 85  | 85   |
| 90  | 90   |
</details>

Figure 9: AGL and ACL for the ImageNet V2 shifts with random head initialization finetuning.

![](images/8b950af11c3c47f9a35e26181be8c4f031f6ab69009b0a09767a89157dd67ffd.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 10 | 10  |
| 20 | 25  |
| 30 | 40  |
| 40 | 50  |
| 50 | 60  |
| 60 | 70  |
| 70 | 80  |
| 80 | 90  |
| 90 | 95  |
</details>

![](images/2ccba60ea763401284fb59d7f84d1226c6f250dd383fa0157781e8f6cdbb883a.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
| --- | --- |
| 10 | 10 |
| 20 | 25 |
| 30 | 35 |
| 40 | 45 |
| 50 | 55 |
| 60 | 65 |
| 70 | 75 |
| 80 | 85 |
| 90 | 90 |
</details>

![](images/d93ad76b5f59a9d1003864a353aa5b43ea9ab2f9d49cd15f92efb52a0a1e2cf1.jpg)

<details>
<summary>scatter</summary>

| ID | OOD | Type     |
|----|-----|----------|
| 10 | 5   | Accuracy |
| 15 | 10  | Agreement |
| 20 | 15  | Accuracy |
| 25 | 20  | Agreement |
| 30 | 25  | Accuracy |
| 35 | 30  | Agreement |
| 40 | 35  | Accuracy |
| 45 | 40  | Agreement |
| 50 | 45  | Accuracy |
| 55 | 50  | Agreement |
| 60 | 55  | Accuracy |
| 65 | 60  | Agreement |
| 70 | 65  | Accuracy |
| 75 | 70  | Agreement |
| 80 | 75  | Accuracy |
| 85 | 80  | Agreement |
| 90 | 85  | Accuracy |
| 95 | 90  | Agreement |
</details>

Figure 10: AGL and ACL for 3 benchmarks from the WILDS dataset with random head initialization finetuning.

![](images/75b570cbc01c655ac36d08b8cde8ef9cc0d08a8c31999772993d78f9ab79c5db.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 10 | 5   |
| 20 | 10  |
| 30 | 15  |
| 40 | 20  |
| 50 | 25  |
| 60 | 30  |
| 70 | 35  |
| 80 | 40  |
| 90 | 45  |
</details>

![](images/bf6a3cb35501ff0a1c580dca13e3d3ba73d14f95cc3a031069a3d9c426af97a0.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  |
| --- | ---- |
| 10  | 5    |
| 20  | 10   |
| 30  | 15   |
| 40  | 20   |
| 50  | 25   |
| 60  | 30   |
| 70  | 35   |
| 80  | 40   |
| 90  | 45   |
</details>

![](images/3c42d4c5348a2a556523b0b4cb188cd68c941276b0c119ac8deb9630f9749c40.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 10 | 5   |
| 20 | 10  |
| 30 | 20  |
| 40 | 30  |
| 50 | 40  |
| 60 | 50  |
| 70 | 60  |
| 80 | 70  |
| 90 | 80  |
</details>

Figure 11: AGL and ACL for the OfficeHome ClipArt, Product, Real shifts with random head initialization finetuning over OfficeHome Art.

![](images/fe735b7997d21687eced525bc50e14634d80779c4e1155309907f1a5dbdd7198.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  |
| --- | ---- |
| 5   | 5    |
| 10  | 10   |
| 20  | 20   |
| 30  | 30   |
| 40  | 40   |
| 50  | 50   |
| 60  | 60   |
| 70  | 70   |
| 80  | 80   |
| 90  | 90   |
</details>

![](images/9900cca206203908bf4b11c74c0b9fa97180a8c51c0864bd82141bfead231f0f.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 10 | 5   |
| 20 | 10  |
| 30 | 15  |
| 40 | 20  |
| 50 | 25  |
| 60 | 30  |
| 70 | 35  |
| 80 | 40  |
| 90 | 45  |
</details>

![](images/3969b75501dd08087c566893a9a574c8c401b7b18af68d34f8d95cca9c916978.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  |
| --- | ---- |
| 5   | 5    |
| 10  | 10   |
| 15  | 15   |
| 20  | 20   |
| 25  | 25   |
| 30  | 30   |
| 35  | 35   |
| 40  | 40   |
| 45  | 45   |
| 50  | 50   |
| 55  | 55   |
| 60  | 60   |
| 65  | 65   |
| 70  | 70   |
| 75  | 75   |
| 80  | 80   |
| 85  | 85   |
| 90  | 90   |
</details>

Figure 12: ID vs OOD accuracy and agreement with OH ClipArt ID   
Figure 13: AGL and ACL for the OfficeHome Art, Product, Real shifts with random head initialization finetuning over OfficeHome ClipArt.

![](images/bcde30f4c7c661cf77b470dbc7aa030c9f720f197b0168c9cb042491b3681bd2.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  |
| --- | ---- |
| 10  | 5    |
| 20  | 8    |
| 30  | 10   |
| 40  | 12   |
| 50  | 15   |
| 60  | 20   |
| 70  | 25   |
| 80  | 30   |
| 90  | 35   |
</details>

![](images/5ffedfc2105e21e5eef2f3fef0469d184488b9aa5f3f45b7e810d57310c672c5.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  |
| --- | ---- |
| 10  | 8    |
| 20  | 12   |
| 30  | 16   |
| 40  | 20   |
| 50  | 24   |
| 60  | 28   |
| 70  | 32   |
| 80  | 36   |
| 90  | 40   |
</details>

![](images/0befbf15921c90114afcd43739dd7b60be9c61eb9c6fa4580358d1b5efb0eddf.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  |
| --- | ---- |
| 5   | 5    |
| 10  | 10   |
| 20  | 20   |
| 30  | 30   |
| 40  | 40   |
| 50  | 50   |
| 60  | 60   |
| 70  | 70   |
| 80  | 80   |
| 90  | 90   |
</details>

Figure 14: AGL and ACL for the OfficeHome ClipArt, Art, Real shifts with random head initialization finetuning over OfficeHome Product.

![](images/1903eb724df151ce6bd93775742b06a0795cbd25ca0617ab5a423cced699eb27.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
| --- | --- |
| 10 | 10 |
| 20 | 20 |
| 30 | 30 |
| 40 | 40 |
| 50 | 50 |
| 60 | 60 |
| 70 | 70 |
| 80 | 80 |
| 90 | 90 |
</details>

![](images/1b25d566e0f41e49e9b7bbf1d4896c2b70caaff07fb7b41bea647f5a3e2c6ff8.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 10 | 5   |
| 20 | 10  |
| 30 | 15  |
| 40 | 20  |
| 50 | 25  |
| 60 | 30  |
| 70 | 35  |
| 80 | 40  |
| 90 | 45  |
</details>

![](images/9546aa8a7a69d466c213d0dfb0ef31f52e91bc29357d83f478fa1c1c1b448873.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  |
| --- | ---- |
| 10  | 5    |
| 20  | 10   |
| 30  | 15   |
| 40  | 20   |
| 50  | 25   |
| 60  | 30   |
| 70  | 35   |
| 80  | 40   |
| 90  | 45   |
</details>

Figure 15: AGL and ACL for the OfficeHome Art, ClipArt, Product shifts with random head initialization finetuning over OfficeHome Real.

# A.3.3 Diversity Matters in Full Finetuned CLIP Ensembles

We show that light full finetuning over CLIP also observes similar effects of diversity source on the strength of AGL. We verify this on shifts from CIFAR10 to CIFAR10C.

![](images/75bbb0b52de98ee4272ccd5ece7df867fdbfb1b911794eb479f17677af7fafdf.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
| --- | --- |
| 10 | 5 |
| 20 | 15 |
| 30 | 25 |
| 40 | 35 |
| 50 | 45 |
| 60 | 55 |
| 70 | 65 |
| 80 | 75 |
| 90 | 85 |
</details>

![](images/70bffb7c24395e6e0a99cb6b9bec5ce3a51d59678beb3961a8e153c3ab839734.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  |
| --- | ---- |
| 10  | 10   |
| 20  | 20   |
| 30  | 30   |
| 40  | 40   |
| 50  | 50   |
| 60  | 60   |
| 70  | 70   |
| 80  | 80   |
| 90  | 90   |
| 100 | 100  |
</details>

![](images/7521bf1511a3d0c0ce9f2823299b6d59d01b0418fe7cc2cd92910e55e4f63fd0.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  |
| --- | ---- |
| 25  | 20   |
| 30  | 30   |
| 40  | 40   |
| 50  | 50   |
| 60  | 60   |
| 70  | 70   |
| 80  | 80   |
| 90  | 90   |
| 95  | 95   |
</details>

![](images/4cd1b03b13ca011480463e1d9d0fe504fe7e19576ef78290a1ffcacaf737258e.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 10 | 5   |
| 20 | 15  |
| 30 | 25  |
| 40 | 35  |
| 50 | 45  |
| 60 | 55  |
| 70 | 65  |
| 80 | 75  |
| 90 | 85  |
</details>

(a) Random Head

![](images/916123bbdfc8aacad71e3eb76cd4b5fca0f54b02162725b530fb29fcbd294e37.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  |
| --- | ---- |
| 10  | 10   |
| 20  | 25   |
| 30  | 35   |
| 40  | 45   |
| 50  | 55   |
| 60  | 65   |
| 70  | 75   |
| 80  | 85   |
| 90  | 95   |
| 100 | 100  |
</details>

(b) Data Ordering

![](images/7a51efd371f1169d62103c230f98570e430c187d85b832b8644dcf082643295b.jpg)

<details>
<summary>line</summary>

| ID  | Accuracy | Agreement |
| --- | -------- | --------- |
| 10  | 10       | 10        |
| 30  | 30       | 30        |
| 50  | 50       | 50        |
| 70  | 70       | 70        |
| 90  | 90       | 90        |
| 110 | 110      | 110       |
</details>

(c) Data Subsetting   
Figure 16: ID vs OOD accuracy and agreement of full finetuned CLIP models on shift from CIFAR10 to CIFAR10C “JPEG Compression” (top row) and “Pixelate” (bottom row) shifts. Only Random Initialization (Column a) yields AGL or matching slopes in accuracy and agreement.

Table 13: The average MAPE(%) of ALine estimates of OOD performance of full finetuned CLIP models across all 19 CIFAR10C shifts. As can be seen from Figure 4, only ensembles with diverse random initialization consistently results in smallest MAPE values. 

<table><tr><td>Source of Diversity</td><td>CIFAR10C MAPE(%)</td></tr><tr><td>Random Linear Heads</td><td>29.37</td></tr><tr><td>Data Ordering</td><td>69.98</td></tr><tr><td>Data Subsetting</td><td>33.57</td></tr></table>

# A.3.4 Any diverse ensemble display AGL in models heavily trained from scratch

On the other hand, we demonstrate that in models heavily trained from scratch, AGL can be observed irrespective of the diversity source. Consistent with the models in $[2]$ , we train ResNet18 on CIFAR10 from scratch, varying the different sources of randomness. These models are trained heavily with SGD with learning rate of $1 \times 10^{-2}$ , batch size 128, and weight decay of $1 \times 10^{-5}$ for up to 200 epochs. We do not use any data augmentation.

![](images/4555d7a0b989f3b4c7c08ca1f1d992e477065ae16b5957f00e994e90073afee8.jpg)  
Figure 17: Effect of Diversity Source on ResNet18 from CIFAR10 to CIFAR10C-Snow

# A.3.5 AGL in FMs under Zero-Shot and Few-Shot Settings

![](images/565bfae6ac2782387564c2b17800bcbd5225c649c93f4613595e16febb814d3d.jpg)

<details>
<summary>scatter</summary>

| SQuAD | SQuADshifts Reddit |
|-------|---------------------|
| 10    | 10                  |
| 30    | 30                  |
| 50    | 50                  |
| 70    | 70                  |
| 90    | 90                  |
</details>

![](images/20aef2d7f5905c56fcdb1b37ddaae254c2edb878e7c5ee651d470fd1b8c5912c.jpg)

<details>
<summary>scatter</summary>

| CIFAR10 | CIFAR10C Pixelate | Label     |
| ------- | ----------------- | --------- |
| ~10     | ~10               | Agr R20.988 |
| ~30     | ~30               | Acc R20.994 |
</details>

Figure 18: Zero-Shot and Few-Shot Settings. (Left) We plot the ID versus OOD zero-shot performances of OLMo7B checkpoints. We see that the linear correlation is weak in both F1 and F1-Agreement. This means that the effective robustness of base models vary widely during pretraining. (Right) We train linear probes over CLIP embeddings on few-shot CIFAR10 (10 examples per class) with random initialization. We similarly observe AGL and ACL with random initialization.

# A.3.6 Effect of Training Dataset Size

To rid of any confounding factors from data subsetting observing a smaller amount of data, we evaluate all randomness sources on different training dataset sizes. We track the effect of diversity in random initialization, data ordering, and data subsetting for different portions of the training data (100%, 50%, 30%, 10%). For each percentage x%, Random Initialization and Data Ordering ensembles are trained on the same randomly sampled x% proportion of the data while each model in the Data Subsetting ensemble observe different randomly sampled x% portions

![](images/66b5d4f92db970957f40dec1a4feab48bb26f76a1a39bf3bcd60ee4af9055835.jpg)

<details>
<summary>scatter</summary>

| ID | OOD | Accuracy | Agreement |
|----|-----|----------|-----------|
| 5  | 5   | 8        | 7         |
| 10 | 10  | 12       | 11        |
| 15 | 15  | 16       | 15        |
| 20 | 20  | 20       | 19        |
| 25 | 25  | 24       | 23        |
| 30 | 30  | 28       | 27        |
| 35 | 35  | 32       | 31        |
| 40 | 40  | 36       | 35        |
| 45 | 45  | 40       | 39        |
| 50 | 50  | 44       | 43        |
| 55 | 55  | 48       | 47        |
| 60 | 60  | 52       | 51        |
| 65 | 65  | 56       | 55        |
| 70 | 70  | 60       | 59        |
| 75 | 75  | 64       | 63        |
| 80 | 80  | 68       | 67        |
| 85 | 85  | 72       | 71        |
| 90 | 90  | 76       | 75        |
</details>

![](images/087946782d78520d0b8aa4b28a78364df2d976f6f6dbc55e47e664d21951c84d.jpg)

<details>
<summary>line</summary>

| ID  | Accuracy | Agreement |
| --- | -------- | --------- |
| 10  | 10       | 10        |
| 30  | 25       | 50        |
| 50  | 30       | 60        |
| 70  | 40       | 80        |
| 90  | 50       | 90        |
| 110 | 60       | 100       |
</details>

![](images/31d3b0e0653d358a667095988316ed7c8eba1eddaca9ec4cb76fd971d97ec922.jpg)

<details>
<summary>scatter</summary>

| ID | OOD | Type     |
|----|-----|----------|
| 10 | 5   | Accuracy |
| 20 | 15  | Agreement|
| 30 | 25  | Accuracy |
| 40 | 35  | Agreement|
| 50 | 45  | Accuracy |
| 60 | 55  | Agreement|
| 70 | 65  | Accuracy |
| 80 | 75  | Agreement|
| 90 | 85  | Accuracy |
</details>

![](images/4a383239bd9b02e061d310dbc857196dcd050232487f40bd2e4c2f286ef1e850.jpg)

<details>
<summary>line</summary>

| ID  | Accuracy | Agreement |
| --- | -------- | --------- |
| 10  | 10       | 10        |
| 30  | 30       | 40        |
| 50  | 50       | 60        |
| 70  | 70       | 80        |
| 90  | 90       | 100       |
| 110 | 110      | 120       |
</details>

(a) Random Head   
(b) Data ordering

Figure 19: ID vs OOD accuracy and agreement of linear probe CLIP models finetuned on 100% of CIFAR10 training data evaluated on the CIFAR10C “JPEG Compression” (top row) and “Pixelate” (bottom row) shifts   
![](images/71b3b19ca251471d43cf8147c94b8b226097cb675e5071634573b189c90471dc.jpg)

<details>
<summary>scatter</summary>

| ID | OOD | Type     |
|----|-----|----------|
| 10 | 10  | Accuracy |
| 20 | 20  | Agreement|
| 30 | 30  | Accuracy |
| 40 | 40  | Agreement|
| 50 | 50  | Accuracy |
| 60 | 60  | Agreement|
| 70 | 70  | Accuracy |
| 80 | 80  | Agreement|
| 90 | 90  | Accuracy |
</details>

![](images/5a9839ce7fb4ea33a12496f29071ab7c5b3097b20bcc1210736bb574421dff78.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  |
| --- | ---- |
| 10  | 10   |
| 20  | 20   |
| 30  | 30   |
| 40  | 40   |
| 50  | 50   |
| 60  | 60   |
| 70  | 70   |
| 80  | 80   |
| 90  | 90   |
| 100 | 100  |
</details>

![](images/c96e07d33f4d24343dfd96f7b618ea74cbf65db71b1a077b20454278baebe605.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  |
| --- | ---- |
| 10  | 10   |
| 20  | 20   |
| 30  | 30   |
| 40  | 40   |
| 50  | 50   |
| 60  | 60   |
| 70  | 70   |
| 80  | 80   |
| 90  | 90   |
| 100 | 100  |
</details>

![](images/3186e465fbb1824b3b5cea4a627e6355c9c410b15dcfe515ff11ac1481a2af3b.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 10 | 10  |
| 20 | 20  |
| 30 | 30  |
| 40 | 40  |
| 50 | 50  |
| 60 | 60  |
| 70 | 70  |
| 80 | 80  |
| 90 | 90  |
</details>

(a) Random Head

![](images/c54ea213b9f4a28665e8ca466330da8b18d19ff78fab7d21a6a40cd09cc4181f.jpg)

<details>
<summary>line</summary>

| ID  | OOD  |
| --- | ---- |
| 10  | 10   |
| 20  | 20   |
| 30  | 30   |
| 40  | 40   |
| 50  | 50   |
| 60  | 60   |
| 70  | 70   |
| 80  | 80   |
| 90  | 90   |
| 100 | 100  |
</details>

(b) Data ordering

![](images/bdb3a79b7a709c0433e32e6b87d60eff0a1372c59168bc792b46903048cd7afd.jpg)

<details>
<summary>line</summary>

| ID  | OOD (Accuracy) | OOD (Agreement) |
| --- | -------------- | --------------- |
| 10  | 15             | 15              |
| 30  | 30             | 40              |
| 50  | 45             | 60              |
| 70  | 60             | 80              |
| 90  | 75             | 95              |
| 110 | 90             | 105             |
</details>

(c) Data Subsetting   
Figure 20: ID vs OOD accuracy and agreement of linear probe CLIP models finetuned on 50% of CIFAR10 training data evaluated on the CIFAR10C “JPEG Compression” (top) and “Pixelate” (bottom) shifts

![](images/c5b7dfdf6b4c4c965012b5a420920cb886735ea093d203daef824a63ab808724.jpg)

<details>
<summary>scatter</summary>

| ID | OOD | Accuracy |
|----|-----|----------|
| 10 | 5   | 8        |
| 20 | 10  | 12       |
| 30 | 15  | 16       |
| 40 | 20  | 20       |
| 50 | 25  | 24       |
| 60 | 30  | 28       |
| 70 | 35  | 32       |
| 80 | 40  | 36       |
| 90 | 45  | 40       |
</details>

![](images/d253beac7828f225bd30690934b4fe2e7e5d318c322e0e51ab6b6a5cc9134d45.jpg)

![](images/7efdd92d1e0415447db8bfc53bc27a8b265c3992a96100654aabe58d7399558c.jpg)

<details>
<summary>line</summary>

| ID  | OOD (Accuracy) | OOD (Agreement) |
| --- | -------------- | --------------- |
| 10  | 10             | 10              |
| 30  | 25             | 30              |
| 50  | 40             | 50              |
| 70  | 55             | 70              |
| 90  | 70             | 90              |
| 110 | 85             | 105             |
</details>

![](images/7ff5f39bb51fa84ef4ecc0f4fe785824e7f57cccb2767beb1b031aec3c2a397e.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 10 | 5   |
| 20 | 15  |
| 30 | 25  |
| 40 | 35  |
| 50 | 45  |
| 60 | 55  |
| 70 | 65  |
| 80 | 75  |
| 90 | 85  |
</details>

(a) Random Head

![](images/a969d712d2d57dc927d1a8468ce69796b9c0e5a9ed119efb48077ce3e572389f.jpg)

<details>
<summary>line</summary>

| ID  | OOD (Accuracy) | OOD (Agreement) |
| --- | -------------- | --------------- |
| 10  | 10             | 10              |
| 20  | 25             | 30              |
| 30  | 35             | 45              |
| 40  | 45             | 60              |
| 50  | 55             | 70              |
| 60  | 65             | 80              |
| 70  | 75             | 90              |
| 80  | 85             | 100             |
| 90  | 95             | 110             |
| 100 | 105            | 120             |
</details>

(b) Data ordering

![](images/b64739e3d41b1684466bab80d19d4733b448b2a1ca3e4e5c5be8df4368d69f02.jpg)

<details>
<summary>line</summary>

| ID  | OOD  |
| --- | ---- |
| 10  | 10   |
| 30  | 30   |
| 50  | 50   |
| 70  | 70   |
| 90  | 90   |
</details>

(c) Data Subsetting   
Figure 21: ID vs OOD accuracy and agreement of linear probe CLIP models finetuned on 30% of the CIFAR10 training data evaluated on the CIFAR10C “JPEG Compression” (top) and “Pixelate” (bottom) shifts

![](images/66b3a93a7c049b409b3c5f11a3cdc93c2f463f58749dbc98ec33a09915f8de62.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
| --- | --- |
| 10 | 8 |
| 20 | 15 |
| 30 | 25 |
| 40 | 35 |
| 50 | 45 |
| 60 | 55 |
| 70 | 65 |
| 80 | 75 |
| 90 | 85 |
</details>

![](images/fe3497efc871a653a6f797a473fb8eefb34dbf422fcec949b47976318e0b2a18.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  | Type     |
| --- | ---- | -------- |
| 10  | 10   | Accuracy |
| 20  | 20   | Accuracy |
| 30  | 30   | Accuracy |
| 40  | 40   | Accuracy |
| 50  | 50   | Accuracy |
| 60  | 60   | Accuracy |
| 70  | 70   | Accuracy |
| 80  | 80   | Accuracy |
| 90  | 90   | Accuracy |
| 10  | 10   | Agreement |
| 20  | 20   | Agreement |
| 30  | 30   | Agreement |
| 40  | 40   | Agreement |
| 50  | 50   | Agreement |
| 60  | 60   | Agreement |
| 70  | 70   | Agreement |
| 80  | 80   | Agreement |
| 90  | 90   | Agreement |
| 10  | 10   | Agreement |
| 20  | 20   | Agreement |
| 30  | 30   | Agreement |
| 40  | 40   | Agreement |
| 50  | 50   | Agreement |
| 60  | 60   | Agreement |
| 70  | 70   | Agreement |
| 80  | 90   | Agreement |
| 90  | 100  | Agreement |
</details>

![](images/855113ce5157374765a46e8ac477d0363acc8049771f01166741d6aeb24ddbdd.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  |
| --- | ---- |
| 10  | 10   |
| 20  | 20   |
| 30  | 30   |
| 40  | 40   |
| 50  | 50   |
| 60  | 60   |
| 70  | 70   |
| 80  | 80   |
| 90  | 90   |
</details>

![](images/bd7f4dc45fbafbbc9318c98f6d3207f10edbcc03578675c707ddf1b7b5dbefb5.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 5  | 5   |
| 10 | 10  |
| 15 | 15  |
| 20 | 20  |
| 25 | 25  |
| 30 | 30  |
| 35 | 35  |
| 40 | 40  |
| 45 | 45  |
| 50 | 50  |
| 55 | 55  |
| 60 | 60  |
| 65 | 65  |
| 70 | 70  |
| 75 | 75  |
| 80 | 80  |
| 85 | 85  |
| 90 | 90  |
</details>

(a) Random Head

![](images/424dba06f9c5fd7d69361c5a961d6e6c219bacd031794355e2a03d9ea3e67637.jpg)

<details>
<summary>line</summary>

| ID  | Accuracy | Agreement |
| --- | -------- | --------- |
| 10  | 10       | 10        |
| 30  | 30       | 30        |
| 50  | 50       | 50        |
| 70  | 70       | 70        |
| 90  | 90       | 90        |
| 110 | 110      | 110       |
</details>

(b) Data Ordering

![](images/b767e89c18d411b506c142bbf3cda1f4b94f1c5c653295c3134e578830d192f4.jpg)

<details>
<summary>line</summary>

| ID  | OOD (Accuracy) | OOD (Agreement) |
| --- | -------------- | --------------- |
| 10  | 10             | 10              |
| 30  | 30             | 40              |
| 50  | 40             | 60              |
| 70  | 50             | 70              |
| 90  | 70             | 90              |
</details>

(c) Data Subsetting   
Figure 22: ID vs OOD accuracy and agreement of linear probe CLIP models finetuned on 10% of the CIFAR10 training data and evaluated on the CIFAR10C “JPEG Compression” (top) and “Pixelate” (bottom) shifts

# A.4 More Experiments on the Effect of Diversity Source for Extractive Question Answering

# A.4.1 Single-Base Diversity Experiments Using OPT and BERT

In Section 3.2, we report the effect of diversity source in full finetuned GPT2-Medium models for the extractive QA task SQuAD to SQuAD-Shifts. In this section, we also provide the same experiments on OPT-125M and BERT. As observed in GPT, using Random Heads yields the strongest AGL behavior and achieves the smallest ALine-D MAPE.

In Table 14 and 15, we report the MAPE of OOD performance estimation of models trained with Random Initialization and Data Ordering using 100% of training data and Data Subsetting with 10% of training data.

Table 14: The average MAPE (%) of ALine-D OOD performance estimates of OPT-125M models full finetuned on SQuAD. 

<table><tr><td>Source of Diversity</td><td>SQuAD-Shifts Amazon</td><td>SQuAD-Shifts Reddit</td></tr><tr><td>Random Head</td><td>6.54</td><td>5.43</td></tr><tr><td>Data Ordering</td><td>11.37</td><td>8.70</td></tr><tr><td>Data Subsetting</td><td>11.15</td><td>9.65</td></tr></table>

Table 15: The average MAPE (%) of ALine-D OOD performance estimates of BERT models full finetuned on SQuAD. 

<table><tr><td>Source of Diversity</td><td>SQuAD-Shifts Amazon</td><td>SQuAD-Shifts Reddit</td></tr><tr><td>Random Head</td><td>14.70</td><td>8.62</td></tr><tr><td>Data Ordering</td><td>16.55</td><td>9.16</td></tr><tr><td>Data Subsetting</td><td>22.13</td><td>18.64</td></tr></table>

# A.4.2 Diversity in Full Finetuned GPT2 for Different Data Portions

We track the effect of diversity in random initialization, data ordering, and data subsetting for different portions of the training data (100%, 50%, 30%, 10%). For each percentage x%, Random Initialization and Data Ordering ensembles are trained on the same randomly sampled x% proportion of the data while each model in the Data Subsetting ensemble observe different randomly sampled x% portions.

![](images/fb34817898d908dd32b5feb53a80f903b22c9b62b22663acb25c03e5354409a5.jpg)  
Figure 23: ID vs OOD accuracy and agreement of models finetuned on SQuAD from a single pretrained GPT2 model with 100% of the training data

![](images/34b60c923e10748343426576387f74f3896060ab2e242f26f685b1edc508fd18.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|-----------|
| 10 | 10       | 10        |
| 30 | 30       | 30        |
| 50 | 50       | 50        |
| 70 | 70       | 70        |
| 90 | 90       | 90        |
</details>

![](images/fa29282994df9da4ec83875bfb315d3dde72441c538fec7a5b9839365b732c20.jpg)

<details>
<summary>line</summary>

| ID  | F1 Score | Agreement |
| --- | -------- | --------- |
| 10  | 10       | 10        |
| 30  | 30       | 30        |
| 50  | 50       | 50        |
| 70  | 70       | 70        |
| 90  | 90       | 90        |
</details>

![](images/00633b808667faabce8b94c51a99e05f2cde2f36e668919243627e0e61b06a7b.jpg)

<details>
<summary>line</summary>

| ID  | F1 Score | Agreement |
| --- | -------- | --------- |
| 10  | 10       | 10        |
| 30  | 30       | 30        |
| 50  | 50       | 50        |
| 70  | 70       | 70        |
| 90  | 90       | 90        |
</details>

![](images/07a08dfdd2dd88be6ebef5b5f18317f74858096b6c857c35fdd91f0010db6a3c.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 10 | 10  |
| 30 | 30  |
| 50 | 50  |
| 70 | 70  |
| 90 | 90  |
</details>

![](images/fc6561beaca91d485c5376aba0c932c1f69978953d2ecfe2fa29070ebca27c8c.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|-----------|
| 10 | 10       | 10        |
| 30 | 30       | 30        |
| 50 | 50       | 50        |
| 70 | 70       | 70        |
| 90 | 90       | 90        |
</details>

![](images/dc7b62c8a8a09a04012a466b265eb6e173a1e407f4bbd7dbc3da4d023efbd7c8.jpg)

<details>
<summary>line</summary>

| ID  | F1 Score | Agreement |
| --- | -------- | --------- |
| 10  | 10       | 10        |
| 30  | 30       | 30        |
| 50  | 50       | 50        |
| 70  | 70       | 70        |
| 90  | 90       | 90        |
</details>

![](images/6e872481cdca63cecda0f04e061e02225e2ef7a7ae0ff5c62d3f95eb3b9db7a9.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 30 | 30       | 30      |
| 40 | 40       | 40      |
| 50 | 50       | 50      |
| 60 | 60       | 60      |
| 70 | 70       | 70      |
| 80 | 80       | 80      |
| 90 | 90       | 90      |
</details>

![](images/ce857abd834a692e3d4c17ed01b0bd555cce77de53b2b1d57a14b8b1acb6de12.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/068aa99e18e379f8661ea95722c3b729edfdb430e0a99795fda6dbd67ba370d5.jpg)

<details>
<summary>scatter</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/fb70c82c09132ef1e7a67e3535e2071488f32186e9a17925d816d64439f15d2c.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/cdb2945938df827a069eed9ae5678edcbd52af45e1f7a3f1701331c0ff9eb425.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/830b054ec0a365c70b19955587f31938c046ee2e4b2446470c89d8ae13b800f9.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

(a) Random Head   
(b) Data Ordering   
(c) Data Subsetting   
Figure 24: ID vs OOD accuracy and agreement of models finetuned on SQuAD from a single pretrained GPT2 model with 50% of the training data

![](images/25c61eafd0d42dfad6df75e20bebd801c5e28d5b84e0bc49e2d6e2408216437d.jpg)

<details>
<summary>line</summary>

| ID  | F1 Score | Agreement |
| --- | -------- | --------- |
| 10  | 10       | 10        |
| 20  | 25       | 25        |
| 30  | 35       | 35        |
| 40  | 45       | 45        |
| 50  | 55       | 55        |
| 60  | 65       | 65        |
| 70  | 75       | 75        |
| 80  | 85       | 85        |
| 90  | 95       | 95        |
</details>

![](images/b8dd79920a9aa3992acadf5f1ee7e6b90351583f42d9d9b357d908230bb81034.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 5.0      | 5.0     |
| 20 | 15.0     | 18.0    |
| 30 | 25.0     | 28.0    |
| 40 | 35.0     | 38.0    |
| 50 | 45.0     | 48.0    |
| 60 | 55.0     | 58.0    |
| 70 | 65.0     | 68.0    |
| 80 | 75.0     | 78.0    |
| 90 | 85.0     | 88.0    |
</details>

![](images/43bb9bf40eecb97312fb0648632f1ec66e9fcc3864923b2e3708a76e2752216c.jpg)

<details>
<summary>line</summary>

| ID  | F1 Score | Agreement |
| --- | -------- | --------- |
| 10  | 10       | 10        |
| 20  | 20       | 25        |
| 30  | 30       | 35        |
| 40  | 40       | 45        |
| 50  | 50       | 55        |
| 60  | 60       | 65        |
| 70  | 70       | 75        |
| 80  | 80       | 85        |
| 90  | 90       | 95        |
</details>

![](images/09463b9fafc76107bf2430db5ba7aa1fda96291fe5ece6c822242abcb8753765.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|-----------|
| 10 | 10       | 10        |
| 20 | 25       | 25        |
| 30 | 35       | 35        |
| 40 | 45       | 45        |
| 50 | 55       | 55        |
| 60 | 65       | 65        |
| 70 | 75       | 75        |
| 80 | 85       | 85        |
| 90 | 90       | 90        |
</details>

![](images/ea3777b8111116f88bd3d5fffa92b63a06d41269eb1fc6d68315e16197bb1b88.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 8.0      | 8.0     |
| 30 | 25.0     | 24.0    |
| 50 | 40.0     | 39.0    |
| 70 | 60.0     | 58.0    |
| 90 | 85.0     | 84.0    |
</details>

![](images/7239e9db61286c7e80d4e209b39a9375187f3a0eeb763608263a20ad0cab3854.jpg)

<details>
<summary>line</summary>

| ID  | F1 Score | Agreement |
| --- | -------- | --------- |
| 10  | 10       | 10        |
| 20  | 25       | 25        |
| 30  | 35       | 35        |
| 40  | 45       | 45        |
| 50  | 55       | 55        |
| 60  | 65       | 65        |
| 70  | 75       | 75        |
| 80  | 85       | 85        |
| 90  | 90       | 90        |
</details>

![](images/5a2a07adfa7e6fb1a18d1e583c29cf5c05491794668356384e8e9ec9b93ac066.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
| --- | -------- | --------- |
| 10 | 10 | 10 |
| 20 | 20 | 20 |
| 30 | 30 | 30 |
| 40 | 40 | 40 |
| 50 | 50 | 50 |
| 60 | 60 | 60 |
| 70 | 70 | 70 |
| 80 | 80 | 80 |
| 90 | 90 | 90 |
</details>

![](images/9eb2e018f85d79af338a929c7d57d0ec4cffa3901cd7c6c8826dd761c35d314b.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 8.0      | 8.0     |
| 20 | 20.0     | 20.0    |
| 30 | 30.0     | 30.0    |
| 40 | 40.0     | 40.0    |
| 50 | 50.0     | 50.0    |
| 60 | 60.0     | 60.0    |
| 70 | 70.0     | 70.0    |
| 80 | 80.0     | 80.0    |
| 90 | 90.0     | 90.0    |
</details>

![](images/a0c711c0538e2b719931cd02efd967e018ca8aa89815d8f54bd7f1bc843592d0.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 20 | 20       | 20      |
| 30 | 30       | 30      |
| 40 | 40       | 40      |
| 50 | 50       | 50      |
| 60 | 60       | 60      |
| 70 | 70       | 70      |
| 80 | 80       | 80      |
| 90 | 90       | 90      |
</details>

![](images/95864be25a933fea525689b8d9ae7fc461f4aaa9c938182d11812608fdaf7e10.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 20 | 25       | 25      |
| 30 | 35       | 35      |
| 40 | 45       | 45      |
| 50 | 55       | 55      |
| 60 | 65       | 65      |
| 70 | 75       | 75      |
| 80 | 85       | 85      |
| 90 | 90       | 90      |
</details>

![](images/7baa234d0dd7da82a3ca6f384fdf264b99b008a717f973ccb72531ea40078c80.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/622aa79d63b39fb846b9af89510a2fe229583cc6ae93de3e2af9d4d3cb01abe0.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 20 | 25       | 25      |
| 30 | 35       | 35      |
| 40 | 45       | 45      |
| 50 | 55       | 55      |
| 60 | 65       | 65      |
| 70 | 75       | 75      |
| 80 | 85       | 85      |
| 90 | 90       | 90      |
</details>

(a) Random Head   
(b) Data Ordering   
(c) Data Subsetting   
Figure 25: ID vs OOD accuracy and agreement of models finetuned on SQuAD from a single pretrained GPT2 model with 10% of the training data

# A.4.3 Diversity in Full Finetuned OPT for Different Data Portions

We track the effect of diversity in random initialization, data ordering, and data subsetting for different portions of the training data (100%, 50%, 30%, 10%). For each percentage x%, Random Initialization and Data Ordering ensembles are trained on the same randomly sampled x% proportion of the data while each model in the Data Subsetting ensemble observe different randomly sampled x% portions.

![](images/04af1ec2fa862c9599b545351c59c27f78f75c982a1744b34e6b24c535499a59.jpg)  
Figure 26: ID vs OOD accuracy and agreement of models finetuned on SQuAD from a single pretrained OPT-125m with 100% of the training data

![](images/9f1bc2d1e64dff26addf51f4775f70569402b257a98718216428f8e66b323014.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 10 | 10  |
| 30 | 30  |
| 50 | 50  |
| 70 | 70  |
| 90 | 90  |
</details>

![](images/eaaf78fcc8dc48f95d8900af18c5f0617cdeab2ed3a99a2edfe37c4f6379ab45.jpg)

<details>
<summary>line</summary>

| ID  | F1 Score | Agreement |
| --- | -------- | --------- |
| 10  | 10       | 10        |
| 30  | 30       | 30        |
| 50  | 50       | 50        |
| 70  | 70       | 70        |
| 90  | 90       | 90        |
</details>

![](images/7413670c4cfbff9d93220f077bdbd59d3654183475ce08088101ff27cafa3744.jpg)

<details>
<summary>line</summary>

| ID  | F1 Score | Agreement |
| --- | -------- | --------- |
| 10  | 10       | 10        |
| 30  | 30       | 30        |
| 50  | 50       | 50        |
| 70  | 70       | 70        |
| 90  | 90       | 90        |
</details>

![](images/871ceab1f081321da8abc72128160d3011c972d6cc5bdb1cd75169715c05258d.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 10 | 10  |
| 30 | 30  |
| 50 | 50  |
| 70 | 70  |
| 90 | 90  |
</details>

![](images/038273cf8038cdf3e7c9a9e132cfb65b4dad103f032e7404bb6bc21d8332c8cb.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/8b8ff3863cbdf75535144adb81f25849403f321574ce9a60544a4e82fc41892a.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|-----------|
| 10 | 10       | 10        |
| 30 | 30       | 30        |
| 50 | 50       | 50        |
| 70 | 70       | 70        |
| 90 | 90       | 90        |
</details>

![](images/a097ea53b5ef6ebbfa58e67f6ba860c08b60c8fb8b19836894e407cb0262f64c.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/92b32c0e39fd2549d00af86870559f1a05b48f705af3e18a66472f711d6fe6e5.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/72a386d7ba2faf99e617556aa4ee46e85c9ba41326fa9728b577289bbefc515e.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/7dd87a95043f9c5a8c1f17c760940f10eca31f9c726a8ebfda157c4c81f9c3ac.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/0dea8e99a484fe5b6130ebc5f63dfcbf8c5ff3947644954f90839ea9fad31c41.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/35dc1c0eca311e827cd199608c625258380e79fb00b63bf46500b83cccdaaef8.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

(a) Random Head   
(b) Data Ordering   
(c) Data Subsetting   
Figure 27: ID vs OOD accuracy and agreement of models finetuned on SQuAD from a single pretrained OPT-125m with 50% of the training data

![](images/25f3838b1628654bb27ff49e0ae33e2c6230384dbd5b7f5ceb944b75ab6291ac.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  |
| --- | ---- |
| 10  | 10   |
| 30  | 30   |
| 50  | 50   |
| 70  | 70   |
| 90  | 90   |
</details>

![](images/008fd404650e3fdc733bc76dddd4ac898ba1041cc8ae2e5f9e34f2c889960771.jpg)

<details>
<summary>line</summary>

| ID  | F1 Score | Agreement |
| --- | -------- | --------- |
| 10  | 10       | 10        |
| 30  | 30       | 30        |
| 50  | 50       | 50        |
| 70  | 70       | 70        |
| 90  | 90       | 90        |
</details>

![](images/250438087737d57ab7635a0707785310285f11ef795b94c62b69ddce82b85cd2.jpg)

<details>
<summary>line</summary>

| ID  | F1 Score | Agreement |
| --- | -------- | --------- |
| 10  | 10       | 10        |
| 30  | 30       | 30        |
| 50  | 50       | 50        |
| 70  | 70       | 70        |
| 90  | 90       | 90        |
</details>

![](images/a38e33b75490eef01fc990befed2b8a1203a486e0dbbe7320c4bd8e9be12dd1d.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 30 | 30  |
| 40 | 40  |
| 50 | 50  |
| 60 | 60  |
| 70 | 70  |
| 80 | 80  |
| 90 | 90  |
</details>

![](images/c526184af34a0bcf46b77d965616a3938c327ee395b08ffb8acc87caf6284d77.jpg)

<details>
<summary>line</summary>

| ID  | F1 Score | Agreement |
| --- | -------- | --------- |
| 10  | 10       | 10        |
| 30  | 30       | 30        |
| 50  | 50       | 50        |
| 70  | 70       | 70        |
| 90  | 90       | 90        |
</details>

![](images/a51e095f8c1ae6ef6b904138f15bd6ce2f4187510e85dc58c336a537cd97f887.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/3f5b656162ea151ff0d1b37791d8657ce8ef08cb4aac9d785a7c6a9d196d1d7d.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/9bfd105844498c0a66fc24844cf9843577de9312b749491df6a91785927cb643.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/631392bfb6aefe958c64fc47d2e035548d10cb845cb116b32c6eb6e953944197.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/91655d945e10bf7dc3427a37fd3789cf31bf04a3136d2a0bd4778b67df924f01.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/64a9c3c7689769fe3adbdd075542cc759044ed7bf1842f62e8cbe84fd80adb86.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/cd52985d05ebac81558e494a2603f256960dd1ff638794e342b8000f39a9c92f.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

(a) Random Head   
(b) Data Ordering   
(c) Data Subsetting   
Figure 28: ID vs OOD accuracy and agreement of models finetuned on SQuAD from a single pretrained OPT-125m with 30% of the training data

![](images/1a8ad69c51eeac47b455f78213f4ab46f79c9166659eccbe41f5a53117adf468.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|-----------|
| 10 | 10       | 10        |
| 20 | 25       | 25        |
| 30 | 35       | 35        |
| 40 | 45       | 45        |
| 50 | 55       | 55        |
| 60 | 65       | 65        |
| 70 | 75       | 75        |
| 80 | 85       | 85        |
| 90 | 95       | 95        |
</details>

![](images/f4fe4a0ad39a263b096e0654557d0b2724dcd02f88bbb1819820dfa0448ca890.jpg)

<details>
<summary>line</summary>

| ID  | F1 Score | Agreement |
| --- | -------- | --------- |
| 10  | 10       | 10        |
| 30  | 30       | 30        |
| 50  | 50       | 50        |
| 70  | 70       | 70        |
| 90  | 90       | 90        |
</details>

![](images/f58a2f42e34956226fd47a582b820d906df81b28f7807a4c6f30b656e3b0e31e.jpg)

<details>
<summary>line</summary>

| ID  | F1 Score | Agreement |
| --- | -------- | --------- |
| 10  | 10       | 10        |
| 30  | 30       | 30        |
| 50  | 50       | 50        |
| 70  | 70       | 70        |
| 90  | 90       | 90        |
</details>

![](images/083d2a206e0775756da5d5d1c6cd567cc82a8e319c46907e802498e9c7d986e1.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 10 | 10  |
| 20 | 20  |
| 30 | 30  |
| 40 | 40  |
| 50 | 50  |
| 60 | 60  |
| 70 | 70  |
| 80 | 80  |
| 90 | 90  |
</details>

![](images/cffd94665b3a080627bec945035664041bf8fa3a6b4fa8374b9c7dd9c0845ce0.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/4531aec754f24e155dccfe2807aca936c188ecb518ebd8afc1af6ba7dfc81b73.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/b755272dd8594449c4c558774a3d55b5f3849a7b956fa2e3f822898320650517.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 20 | 25       | 25      |
| 30 | 35       | 35      |
| 40 | 45       | 45      |
| 50 | 55       | 55      |
| 60 | 65       | 65      |
| 70 | 75       | 75      |
| 80 | 85       | 85      |
| 90 | 90       | 90      |
</details>

![](images/9f22ef85bbb80d2d2606e542dd2b83630b2080edecb90fd4c7643e4933ac6d54.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/dca8d075a35ce2424112f91edd88e17da7aadd9c7d8f0b8857a996df9c95db02.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/cbafd32cf5a56646bf0fd47119f5cb6d136b21ad5aa9807518d08102ccd9490f.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/188994aad71f23fadc4bd059126cde66089ef0944327fa8571f5ec04627793fa.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/4df606b025efc00824d7b3d625a53791e0b79506ce1e38471135df41e221edf8.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

(a) Random Head   
(b) Data Ordering   
(c) Data Subsetting   
Figure 29: ID vs OOD accuracy and agreement of models finetuned on SQuAD from a single pretrained OPT-125m with 10% of the training data

# A.4.4 Diversity in Full Finetuned BERT for Different Data Portions

We track the effect of diversity in random initialization, data ordering, and data subsetting for different portions of the training data (100%, 50%, 30%, 10%). For each percentage x%, Random Initialization and Data Ordering ensembles are trained on the same randomly sampled x% proportion of the data while each model in the Data Subsetting ensemble observe different randomly sampled x% portions.

![](images/574d7ea0c4c867e8058b84babddb4a36806b4d5660ed128052b9b2d7da8e26a2.jpg)  
Figure 30: ID vs OOD accuracy and agreement of models finetuned on SQuAD from a single pretrained BERT with 100% of the training data

![](images/6e38aa7ca30ed5670e845cb4a53d2978d0f9bc37b7c0e6f9429a451d3a55ff3d.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|-----------|
| 10 | 10       | 10        |
| 20 | 25       | 25        |
| 30 | 35       | 35        |
| 40 | 45       | 45        |
| 50 | 55       | 55        |
| 60 | 65       | 65        |
| 70 | 75       | 75        |
| 80 | 85       | 85        |
| 90 | 95       | 95        |
</details>

![](images/bca8ea296294221ed9151a1b90e03e752609c58de3ec5a8529a8368903156e70.jpg)

<details>
<summary>line</summary>

| ID  | F1 Score | Agreement |
| --- | -------- | --------- |
| 10  | 10       | 10        |
| 30  | 30       | 30        |
| 50  | 50       | 50        |
| 70  | 70       | 70        |
| 90  | 90       | 90        |
</details>

![](images/db51c3b50586d1961d683212b15413d5be290d2a2be461d0c58f78e359e0dd41.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|-----------|
| 10 | 10       | 10        |
| 20 | 25       | 25        |
| 30 | 35       | 35        |
| 40 | 45       | 45        |
| 50 | 55       | 55        |
| 60 | 65       | 65        |
| 70 | 75       | 75        |
| 80 | 85       | 85        |
| 90 | 90       | 90        |
</details>

![](images/9305246f0d5d91450c1a20b127c5adf908471ea80cbbb46781e2e0b840980baf.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|-----------|
| 10 | 10       | 10        |
| 20 | 25       | 25        |
| 30 | 35       | 35        |
| 40 | 45       | 45        |
| 50 | 55       | 55        |
| 60 | 65       | 65        |
| 70 | 75       | 75        |
| 80 | 85       | 85        |
| 90 | 90       | 90        |
</details>

![](images/c3b90f2c362acb42978610a6b3560e9255d9cb30ea118bb851a8f5c1488dec84.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/f89e4654734b3a8571f3bc94be271d13ee251686974e22f7104329a8bb4f9706.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 20 | 25       | 25      |
| 30 | 35       | 35      |
| 40 | 45       | 45      |
| 50 | 55       | 55      |
| 60 | 65       | 65      |
| 70 | 75       | 75      |
| 80 | 85       | 85      |
| 90 | 90       | 90      |
</details>

![](images/7bbd8ef9332e58e0a9e219196f58814677e1b8f1dc4e209f9a925b99e52b647b.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 20 | 25       | 25      |
| 30 | 35       | 35      |
| 40 | 45       | 45      |
| 50 | 55       | 55      |
| 60 | 65       | 65      |
| 70 | 75       | 75      |
| 80 | 85       | 85      |
| 90 | 90       | 90      |
</details>

![](images/0b57f5093ab6b69b11d626d607e606a1ed89b32e32dd089b74c10bc56cbc2fdc.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/06fc4e1a6b0956ad5cae67315b7b06ba50b9f135b57264b1c61d0ad27ef8454a.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|-----------|
| 10 | 10       | 10        |
| 20 | 25       | 25        |
| 30 | 35       | 35        |
| 40 | 45       | 45        |
| 50 | 55       | 55        |
| 60 | 65       | 65        |
| 70 | 75       | 75        |
| 80 | 85       | 85        |
| 90 | 90       | 90        |
</details>

![](images/85e96e525490f435ca72283947f776c611a50b99ac1748607c7654f709946c5b.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 20 | 25       | 25      |
| 30 | 35       | 30      |
| 40 | 45       | 40      |
| 50 | 55       | 50      |
| 60 | 65       | 60      |
| 70 | 75       | 70      |
| 80 | 85       | 80      |
| 90 | 90       | 90      |
</details>

![](images/27b9a217766d411564129cfefba82a146b56b1869bdeaf3189967fea550aba38.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/fe998ecbde28b9699581ec0d1780150f09e8abe58fb2e5b83f915d45e36854be.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 20 | 25       | 25      |
| 30 | 35       | 35      |
| 40 | 45       | 45      |
| 50 | 55       | 55      |
| 60 | 65       | 65      |
| 70 | 75       | 75      |
| 80 | 85       | 85      |
| 90 | 90       | 90      |
</details>

(a) Random Head   
(b) Data Ordering   
(c) Data Subsetting   
Figure 31: ID vs OOD accuracy and agreement of models finetuned on SQuAD from a single pretrained BERT with 50% of the training data

![](images/af55d493742bd4c65582e8bf6d82bd870010ecd00e46ae90e8869bfad58fc2b5.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|-----------|
| 10 | 10       | 10        |
| 20 | 25       | 25        |
| 30 | 35       | 35        |
| 40 | 45       | 45        |
| 50 | 55       | 55        |
| 60 | 65       | 65        |
| 70 | 75       | 75        |
| 80 | 85       | 85        |
| 90 | 90       | 90        |
</details>

![](images/7f87223bb01a9e23f6670a8109726c0f013557437a8e305ed2b6c5cfa8fc140d.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 20 | 25       | 25      |
| 30 | 35       | 35      |
| 40 | 45       | 45      |
| 50 | 55       | 55      |
| 60 | 65       | 65      |
| 70 | 75       | 75      |
| 80 | 85       | 85      |
| 90 | 90       | 90      |
</details>

![](images/bbc7805ccc44999a7b7c2cabd9f27bfc552c6227f347bf448e7642ddc55a7558.jpg)

<details>
<summary>line</summary>

| ID  | F1 Score | Agreement |
| --- | -------- | --------- |
| 10  | 10       | 10        |
| 20  | 25       | 25        |
| 30  | 35       | 35        |
| 40  | 45       | 45        |
| 50  | 55       | 55        |
| 60  | 65       | 65        |
| 70  | 75       | 75        |
| 80  | 85       | 85        |
| 90  | 90       | 90        |
</details>

![](images/0a2f33420008e7de4b51a448c57bca4fb1b5100bdc0db63ac706cfbca5760eeb.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|-----------|
| 10 | 10       | 10        |
| 20 | 25       | 25        |
| 30 | 35       | 35        |
| 40 | 45       | 45        |
| 50 | 55       | 55        |
| 60 | 65       | 65        |
| 70 | 75       | 75        |
| 80 | 85       | 85        |
| 90 | 95       | 95        |
</details>

![](images/f909c908f9b45093f01790bfe1807b9aa466c11b16417d113e7f31c850767c0a.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 20 | 25       | 25      |
| 30 | 35       | 35      |
| 40 | 45       | 45      |
| 50 | 55       | 55      |
| 60 | 65       | 65      |
| 70 | 75       | 75      |
| 80 | 85       | 85      |
| 90 | 90       | 90      |
</details>

![](images/5625fe66c7633d702df074bb68ec9c6dd3dd1c2f46331143ce6b4bf75d9b27ff.jpg)

<details>
<summary>line</summary>

| ID  | F1 Score | Agreement |
| --- | -------- | --------- |
| 10  | 10       | 10        |
| 20  | 25       | 25        |
| 30  | 35       | 35        |
| 40  | 45       | 45        |
| 50  | 55       | 55        |
| 60  | 65       | 65        |
| 70  | 75       | 75        |
| 80  | 85       | 85        |
| 90  | 90       | 90        |
</details>

![](images/308ef1cd18af3a06d5cb834e1b3ddd3027289f8177fb2afd223cfc0231fb0249.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
| --- | --- | --- |
| 10 | 10 | 10 |
| 20 | 25 | 25 |
| 30 | 35 | 35 |
| 40 | 45 | 45 |
| 50 | 55 | 55 |
| 60 | 65 | 65 |
| 70 | 75 | 75 |
| 80 | 85 | 85 |
| 90 | 90 | 90 |
</details>

![](images/355a9b12071292c2a11dfd215ffb9e5739dd3f9e36428c519f1ca0246085ddfa.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/3d16b045ef277736cb1e1a97439ad55614aa51746f3736fb7baa8c9bf9a37fa6.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|-----------|
| 10 | 10       | 10        |
| 20 | 25       | 25        |
| 30 | 35       | 35        |
| 40 | 45       | 45        |
| 50 | 55       | 55        |
| 60 | 65       | 65        |
| 70 | 75       | 75        |
| 80 | 85       | 85        |
| 90 | 90       | 90        |
</details>

![](images/5bfa8ad6105f2b95d5e5544ace37699bd35c9aba51699a6c5e9aa0fca16b8958.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/4f7a86f433b10ef2b87acfd2cabe7e18abc129c0733697d95e7471aa174a1dcc.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/df2e3b88a9f2a2ba8a2291534cf95265870b80145e66fba87cd94c60333bbb0a.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 20 | 25       | 25      |
| 30 | 35       | 35      |
| 40 | 45       | 45      |
| 50 | 55       | 55      |
| 60 | 65       | 65      |
| 70 | 75       | 75      |
| 80 | 85       | 85      |
| 90 | 90       | 90      |
</details>

(a) Random Head   
(b) Data Ordering   
(c) Data Subsetting   
Figure 32: ID vs OOD accuracy and agreement of models finetuned on SQuAD from a single pretrained BERT with 30% of the training data

![](images/895e051504c349d21fd6dded0198bf81991c1916ad09dc4af9f66310a4ff716d.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|-----------|
| 10 | 10       | 10        |
| 20 | 25       | 25        |
| 30 | 35       | 35        |
| 40 | 45       | 45        |
| 50 | 55       | 55        |
| 60 | 65       | 65        |
| 70 | 75       | 75        |
| 80 | 85       | 85        |
| 90 | 90       | 90        |
</details>

![](images/3db9a71acfae4d0df71f7b96dc36e475fdc29fd768db587cda10519af2638e34.jpg)

<details>
<summary>line</summary>

| ID  | F1 Score | Agreement |
| --- | -------- | --------- |
| 10  | 10       | 10        |
| 20  | 25       | 28        |
| 30  | 35       | 40        |
| 40  | 45       | 50        |
| 50  | 55       | 60        |
| 60  | 65       | 70        |
| 70  | 75       | 80        |
| 80  | 85       | 90        |
| 90  | 95       | 100       |
</details>

![](images/9d5568aa04b7572d335173f64374d36f2597da302b52b9e9f16fbf107fa13eb6.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|-----------|
| 10 | 10       | 10        |
| 20 | 25       | 25        |
| 30 | 35       | 35        |
| 40 | 45       | 45        |
| 50 | 55       | 55        |
| 60 | 65       | 65        |
| 70 | 75       | 75        |
| 80 | 85       | 85        |
| 90 | 90       | 90        |
</details>

![](images/60393f7b51daa72e3d07d372a43069be5b7295080966d43b6ba3a0bb95f8f700.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|-----------|
| 10 | 10       | 10        |
| 20 | 25       | 25        |
| 30 | 35       | 35        |
| 40 | 45       | 45        |
| 50 | 55       | 55        |
| 60 | 65       | 65        |
| 70 | 75       | 75        |
| 80 | 85       | 85        |
| 90 | 90       | 90        |
</details>

![](images/9bd544c9f9ef720f551b44f793e1fd5d51ef66882e371d6ee3e250b169a85edd.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 20 | 25       | 25      |
| 30 | 35       | 35      |
| 40 | 45       | 45      |
| 50 | 55       | 55      |
| 60 | 65       | 65      |
| 70 | 75       | 75      |
| 80 | 85       | 85      |
| 90 | 90       | 90      |
</details>

![](images/61d4b561c9d0d4dd5b3030b09a1f66bbd8f87ed69e1fc29ccac5aff92234e9e9.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|-----------|
| 10 | 10       | 10        |
| 20 | 25       | 25        |
| 30 | 35       | 35        |
| 40 | 45       | 45        |
| 50 | 55       | 55        |
| 60 | 65       | 65        |
| 70 | 75       | 75        |
| 80 | 85       | 85        |
| 90 | 90       | 90        |
</details>

![](images/6dd3a65424fc61e220273f8e0d981a8781b6b4a6569acabc22c052f770d91294.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 20 | 25       | 25      |
| 30 | 35       | 35      |
| 40 | 45       | 45      |
| 50 | 55       | 55      |
| 60 | 65       | 65      |
| 70 | 75       | 75      |
| 80 | 85       | 85      |
| 90 | 90       | 90      |
</details>

![](images/62925a48126da5ada002c1cd3e7b7acfa5088d253cf746db94597c9d1608ca29.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/4a2d3684494ccb40ae3a229fea4b556289106cdf4c9b75eacd878cd91bfdbe04.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 20 | 25       | 25      |
| 30 | 35       | 35      |
| 40 | 45       | 45      |
| 50 | 55       | 55      |
| 60 | 65       | 65      |
| 70 | 75       | 75      |
| 80 | 85       | 85      |
| 90 | 90       | 90      |
</details>

![](images/a89e99cb7af42809c502d6d279e0f2880062ae164656037ba8519dbff32e1cab.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 20 | 25       | 25      |
| 30 | 35       | 35      |
| 40 | 45       | 45      |
| 50 | 55       | 55      |
| 60 | 65       | 65      |
| 70 | 75       | 75      |
| 80 | 85       | 85      |
| 90 | 90       | 90      |
</details>

![](images/906674f2d63f8b71044f082307ff18b6fd757c3fbc28bca11a59a8da77c8efd5.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 30 | 30       | 30      |
| 50 | 50       | 50      |
| 70 | 70       | 70      |
| 90 | 90       | 90      |
</details>

![](images/490f848e90f074d31837949e7d0d270f8e041b3dac96a6e383e1b3fa6a14ebd9.jpg)

<details>
<summary>line</summary>

| ID | F1 Score | Agreement |
|----|----------|---------|
| 10 | 10       | 10      |
| 20 | 25       | 25      |
| 30 | 35       | 35      |
| 40 | 45       | 45      |
| 50 | 55       | 55      |
| 60 | 65       | 65      |
| 70 | 75       | 75      |
| 80 | 85       | 85      |
| 90 | 90       | 90      |
</details>

(a) Random Head   
(b) Data Ordering   
(c) Data Subsetting   
Figure 33: ID vs OOD accuracy and agreement of models finetuned on SQuAD from a single pretrained BERT with 10% of the training data

# A.5 More Experiments on the Effect of Diversity Source for Text Classification

# A.5.1 Diversity Source for Linear Probing

In Section 3.2, we reported the effect of diversity source for full finetuned OPT. Here, we demonstrate similar results on text classification shift from MNLI-matched to SNLI with linear probed GPT2-Medium (Figure 34) and OPT-125M (Figure 35). Similarly, random initialization observes the strongest AGL behavior. In Table 16, we report the average MAPE of OOD performance estimation using ALine across models.

![](images/4cda487ae15aca7f3ad7937eb0965e8517235c5e801bf9de47fd0ab2b0222c87.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  |
| --- | ---- |
| 30  | 30   |
| 40  | 40   |
| 50  | 50   |
| 60  | 60   |
| 70  | 70   |
| 80  | 80   |
| 90  | 90   |
</details>

(a) Random Head

![](images/64d6c811738349bb7f366b24d6c1e2fe3438f873dcbe9ac0f4fb636ef70ffc70.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  |
| --- | ---- |
| 10  | 10   |
| 30  | 30   |
| 50  | 50   |
| 70  | 70   |
| 90  | 90   |
</details>

(b) Data Ordering

![](images/11795a92abad5949efcf504cc84e7cae240b2db7f6096259e2ebc74b1711780b.jpg)

<details>
<summary>line</summary>

| ID  | OOD  |
| --- | ---- |
| 10  | 10   |
| 30  | 30   |
| 50  | 50   |
| 70  | 70   |
| 90  | 90   |
</details>

(c) Data Subsetting

Figure 34: ID vs OOD accuracy and agreement of models finetuned on MNLI from GPT2-Medium. Random Head and Data Ordering ensembles are trained on 100% of the training data while Data Subsetting ensemble is on 10%.   
![](images/dd51d2655402dd859020739e13faf900838902a73637ed816ce26a6a70421591.jpg)

<details>
<summary>line</summary>

| ID  | OOD  |
| --- | ---- |
| 10  | 10   |
| 30  | 30   |
| 50  | 50   |
| 70  | 70   |
| 90  | 90   |
</details>

(a) Random Head

![](images/38b7e52fe3eddbbf634d29b49e6d1f00ee33ff39fc915abafc2ac9d4dd74e662.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 50 | 40  |
| 55 | 45  |
| 60 | 50  |
| 65 | 55  |
| 70 | 60  |
| 75 | 65  |
| 80 | 70  |
| 85 | 75  |
| 90 | 80  |
</details>

(b) Data Ordering

![](images/5f30c5c919916b158f5bdc97980dfca7ce795ac310032b3f006d1d9cb6dca7f9.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  |
| --- | ---- |
| 10  | 10   |
| 30  | 30   |
| 50  | 50   |
| 70  | 70   |
| 90  | 90   |
</details>

(c) Data Subsetting   
Figure 35: ID vs OOD accuracy and agreement of models finetuned on MNLI from OPT-125M. Random Head and Data Ordering ensembles are trained on 100% of the training data while Data Subsetting ensemble is on 10%.

Table 16: The average MAPE (%) of ALine-D performance estimates of accuracy on SNLI. 

<table><tr><td>Model</td><td>GPT2-Medium</td><td>OPT-125M</td></tr><tr><td>Random Linear Heads</td><td>5.9</td><td>5.4</td></tr><tr><td>Data Ordering</td><td>6.7</td><td>12.4</td></tr><tr><td>Data Subsetting</td><td>5.3</td><td>8.6</td></tr></table>

# A.6 Generative Question-Answering

We also test whether AGL appears for generative question answering, where models generate the answer to the question instead of directly extracting a span from the provided context.

# A.6.1 Experiments on the effect of Diversity Source

For generative tasks, it is common to finetune over the base model directly instead of attaching a linear head on top. We again study different ways of achieving a diverse set of classifiers (e.g., Random Initialization, Data Ordering, Data Subsetting). However, we replace randomly initialization of a linear head instead with randomly initializing the non-zero initialized LoRA weights.

Experimental Details We use a base model GPT2 and finetune models using cross-entropy loss on the next token prediction objective over SQuAD. During training, we concatenate the context, question, and answer together, and we apply the next token prediction objective on just the answer tokens. We add a line break ( $n$ ) after the answer, which functions as a “stop token” that the models also have to predict after answering the question. We train with the AdamW optimizer with a learning rate of $1e-4$ , weight decay of $1e-2$ , and batch size of 16 up to a maximum of 4 epochs. As with extractive QA, we measure the F1 score of the model’s answer (its output before a line break).

Conclusion We track the effect of diversity for 100% of the training data in random initialization and data ordering, 50% of the training data in data subsetting. Figure 36 shows that all sources of diversity are sufficient to show AGL. Diversity is different from extractive question-answering because the search space of all tokens already provides enough diversity among models. We also note the MAE in Table 17 and MAPE in Table 18.

Table 17: ALine-S MAE (%) of different sources of diversity for generative question-answering on GPT2 

<table><tr><td>Source of Diversity</td><td>SQuAD-Shifts Amazon</td><td>SQuAD-Shifts Reddit</td><td>SQuAD-Shifts New-Wiki</td><td>SQuAD-Shifts NYT</td></tr><tr><td>Random Linear Heads</td><td>0.0354</td><td>0.0170</td><td>0.0129</td><td>0.0148</td></tr><tr><td>Data Ordering</td><td>0.0296</td><td>0.0177</td><td>0.0165</td><td>0.0152</td></tr><tr><td>Data Subsetting</td><td>0.0312</td><td>0.0205</td><td>0.0178</td><td>0.0166</td></tr></table>

Table 18: ALine-S MAPE (%) of different sources of diversity for generative question-answering on GPT2 

<table><tr><td>Source of Diversity</td><td>SQuAD-Shifts Amazon</td><td>SQuAD-Shifts Reddit</td><td>SQuAD-Shifts New-Wiki</td><td>SQuAD-Shifts NYT</td></tr><tr><td>Random Linear Heads</td><td>13.93</td><td>6.6</td><td>4.88</td><td>5.06</td></tr><tr><td>Data Ordering</td><td>12.05</td><td>6.57</td><td>5.91</td><td>5.41</td></tr><tr><td>Data Subsetting</td><td>16.43</td><td>10.08</td><td>7.47</td><td>8.32</td></tr></table>

![](images/18c3157e797ba0b5a7aab15828ff7370d1630934d8944dea6c448eef6752b61b.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 10 | 5   |
| 20 | 10  |
| 30 | 20  |
| 40 | 30  |
| 50 | 40  |
| 60 | 50  |
| 70 | 60  |
| 80 | 70  |
| 90 | 80  |
</details>

![](images/7d9a5e8c0c8f7f91a5442f06990498eadc6c5e5f877b6438c89f71e9fdb9546c.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 10 | 5   |
| 20 | 10  |
| 30 | 20  |
| 40 | 30  |
| 50 | 40  |
| 60 | 50  |
| 70 | 60  |
| 80 | 70  |
| 90 | 80  |
</details>

![](images/918822a44d88fc64efe1204653e7d4c4349507df81e25414f2d880e608bde824.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
| --- | --- |
| 10 | 5 |
| 20 | 10 |
| 30 | 20 |
| 40 | 30 |
| 50 | 40 |
| 60 | 50 |
| 70 | 60 |
| 80 | 70 |
| 90 | 80 |
</details>

![](images/63ca3f4e6ab23786a97f462819707857e2ebb29c891178cff10efa4ebf0a04e2.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 10 | 8   |
| 20 | 15  |
| 30 | 25  |
| 40 | 35  |
| 50 | 45  |
| 60 | 55  |
| 70 | 65  |
| 80 | 75  |
| 90 | 85  |
</details>

![](images/aed2e088b6fa7f33a32e37b2af68c6245bc0dc7834dea9e0116dd268ce2287f9.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 10 | 10  |
| 20 | 20  |
| 30 | 30  |
| 40 | 40  |
| 50 | 50  |
| 60 | 60  |
| 70 | 70  |
| 80 | 80  |
| 90 | 90  |
</details>

![](images/c8f681ac438bbb7800080bdc95f188c6d2bac21a542d91c67776f92478f7e19c.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 10 | 5   |
| 20 | 10  |
| 30 | 20  |
| 40 | 30  |
| 50 | 40  |
| 60 | 50  |
| 70 | 60  |
| 80 | 70  |
| 90 | 80  |
</details>

![](images/b022b1f0b8c800cc102087c205a6b542046a9452342035dc8486fa67f264487b.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 10 | 10  |
| 20 | 20  |
| 30 | 30  |
| 40 | 40  |
| 50 | 50  |
| 60 | 60  |
| 70 | 70  |
| 80 | 80  |
| 90 | 90  |
</details>

![](images/f9ee47f550c4acd140a106e9228a867be3a53c2a3a90f4dc123c41324f1a0730.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  |
| --- | ---- |
| 10  | 12   |
| 20  | 25   |
| 30  | 38   |
| 40  | 50   |
| 50  | 62   |
| 60  | 75   |
| 70  | 88   |
| 80  | 95   |
| 90  | 100  |
</details>

![](images/13d1dfaeb6ca27a646cc2d69486b52cbf6265ce69083a33d587969af3ab8424f.jpg)

<details>
<summary>scatter</summary>

| ID  | OOD  |
| --- | ---- |
| 5   | 5    |
| 10  | 10   |
| 15  | 15   |
| 20  | 20   |
| 25  | 25   |
| 30  | 30   |
| 35  | 35   |
| 40  | 40   |
| 45  | 45   |
| 50  | 50   |
| 55  | 55   |
| 60  | 60   |
| 65  | 65   |
| 70  | 70   |
| 75  | 75   |
| 80  | 80   |
| 85  | 85   |
| 90  | 90   |
</details>

![](images/4bb8e63bd34f8831aabafc56fe3e610cbc4ef4973df16e8d9c6cf57838e9023b.jpg)

<details>
<summary>line</summary>

| ID | Accuracy | Agreement |
|----|----------|-----------|
| 5  | 5        | 5         |
| 10 | 10       | 10        |
| 15 | 15       | 15        |
| 20 | 20       | 20        |
| 25 | 25       | 25        |
| 30 | 30       | 30        |
| 35 | 35       | 35        |
| 40 | 40       | 40        |
| 45 | 45       | 45        |
| 50 | 50       | 50        |
| 55 | 55       | 55        |
| 60 | 60       | 60        |
| 65 | 65       | 65        |
| 70 | 70       | 70        |
| 75 | 75       | 75        |
| 80 | 80       | 80        |
| 85 | 85       | 85        |
| 90 | 90       | 90        |
</details>

![](images/9e9f28f8df5df918492ddbfc0c15257d830859469720d04be88f75ff9c0ac1a4.jpg)

<details>
<summary>line</summary>

| ID | OOD (Accuracy) | OOD (Agreement) |
|----|----------------|-----------------|
| 5  | 5              | 5               |
| 10 | 10             | 10              |
| 30 | 30             | 30              |
| 70 | 70             | 70              |
| 90 | 90             | 90              |
</details>

![](images/f6c380bee22171df27d08419a8ae8a343d7978bfc2f3eb123efd355fd3947778.jpg)

<details>
<summary>line</summary>

| ID | OOD (Accuracy) | OOD (Agreement) |
|----|----------------|-----------------|
| 5  | 5              | 5               |
| 10 | 10             | 10              |
| 20 | 20             | 20              |
| 30 | 30             | 30              |
| 40 | 40             | 40              |
| 50 | 50             | 50              |
| 60 | 60             | 60              |
| 70 | 70             | 70              |
| 80 | 80             | 80              |
| 90 | 90             | 90              |
</details>

(a) Random Head   
(b) Data Ordering   
(c) Data Subsetting   
Figure 36: ID vs OOD F1 and agreement of generative models finetuned on SQuAD from a single pretrained GPT2

# A.6.2 Experiments starting from multiple foundation models

We also finetune base models of the GPT and OPT model families for generative question-answering using the same hyperparameters mentioned in the previous section. Similarly, we evaluate models using the F1 score. Figure 37 shows that AGL holds for all shifts in SQuAD-Shifts. Furthermore, it also holds for the mlqa-translate-test.es test split of the MLQA dataset [32], which is the English portion of English-Spanish translated MLQA questions.

![](images/b3bae433c55f0e43dd9c364f5865bed1b75ca86ab145c822fbba1c9f6b6b006b.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 5  | 8   |
| 10 | 12  |
| 15 | 16  |
| 20 | 20  |
| 25 | 24  |
| 30 | 28  |
| 35 | 32  |
| 40 | 36  |
| 45 | 40  |
| 50 | 44  |
| 55 | 48  |
| 60 | 52  |
| 65 | 56  |
| 70 | 60  |
</details>

![](images/972859c455f40cb8ddf4aae7aac7b615ee90faa22eccbdc7730be50e3eff55e4.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 5  | 5   |
| 10 | 10  |
| 20 | 20  |
| 30 | 30  |
| 40 | 40  |
| 50 | 50  |
| 60 | 60  |
| 70 | 70  |
</details>

![](images/a3f105cb1f78bb2096d09b9616bc0ad5a2b84da412a3e0dd421b83726e2e741e.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 5  | 5   |
| 10 | 10  |
| 20 | 20  |
| 30 | 30  |
| 40 | 40  |
| 50 | 50  |
| 60 | 60  |
| 70 | 70  |
</details>

![](images/2dc49f6fa8c07a4091cf802631b991124a42db4f854ba66a8a2e5e62b8a523df.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 5  | 8   |
| 10 | 10  |
| 20 | 15  |
| 30 | 20  |
| 40 | 25  |
| 50 | 30  |
| 60 | 35  |
| 70 | 40  |
</details>

![](images/7e2e666d1262ef1c3e6cfafb09cb6120ef91e35d7db8eddb22b2d18c1b4afad1.jpg)

<details>
<summary>scatter</summary>

| ID | OOD |
|----|-----|
| 5  | 5   |
| 10 | 10  |
| 20 | 20  |
| 30 | 30  |
| 40 | 40  |
| 50 | 50  |
| 60 | 60  |
| 70 | 70  |
</details>

![](images/341dd7c5c8477c131b68305b0cd6770768a22d73bc5c2507004304324c4b5f09.jpg)  
Figure 37: Generative models finetuned from different base models of the GPT and OPT family. AGL holds for all SQuAD-Shifts splits and MLQA Spanish split.

# A.7 Diversity under different learning rates and batch sizes

We test the robustness of our observation across learning rate and batch size for single base FM on CLIP embeddings. Figure 38 shows that AGL holds regardless of the learning rate or batch size used for fine-tuning. That is, AGL holds with random initialization of the head and does when for shuffling the data subset or randomizing the data ordering.

![](images/b8e81e40c88496d49e9e1e963ab86389d55fc63b1e0270509f394cf442cf13cf.jpg)  
Figure 38: Over CLIP embeddings, we test the diversity sources under different learning rates and batch sizes. While all models lie on the same ID versus OOD accuracy line (red), only the set of models trained with different random initialization (cyan) achieves ID versus OOD agreement with the same slope and bias as ID versus OOD accuracy.

# A.8 Diversity Experiments for different PEFT methods

We compare full fine-tuning with different PEFT methods and observe that AGL holds for randomly initialized heads and does not for data ordering or data subsetting. Figure 39 shows a single GPT2 trained with LoRA [25], IA3 [33], and BitFit [64]. The linear fit for the three PEFT methods aligns with full fine-tuning for randomly initialized heads.

![](images/33a5d2eabf2d0ae80096eab813a8ffd32bfe552f4f7b552146b14f2b4fc96d37.jpg)

<details>
<summary>line</summary>

| ID | LoRA | IA3 | BitFit | Full Fine-Tuning |
|----|------|-----|--------|------------------|
| 5  | 5    | 5   | 5      | 5                |
| 10 | 10   | 10  | 10     | 10               |
| 30 | 30   | 30  | 30     | 30               |
| 70 | 70   | 70  | 70     | 70               |
| 90 | 90   | 90  | 90     | 90               |
</details>

(a) Random Head

![](images/a1caa1711096eaee2e44b2446cfbbb4d22b537fa4c5e90cbd33e23824f5638d9.jpg)

<details>
<summary>line</summary>

| ID | LoRA | IA3 | BitFit | Full Fine-Tuning | ACC |
|----|------|-----|--------|------------------|-----|
| 5  | 5    | 5   | 5      | 5                | 5   |
| 10 | 10   | 10  | 10     | 10               | 10  |
| 30 | 30   | 30  | 30     | 30               | 30  |
| 50 | 50   | 50  | 50     | 50               | 50  |
| 70 | 70   | 70  | 70     | 70               | 70  |
| 90 | 90   | 90  | 90     | 90               | 90  |
</details>

(b) Data Ordering

![](images/9712fdd3ccdb02a9fe82d88cfe3a6b676d3077ee01475ca3c407903d1fe88bbe.jpg)

<details>
<summary>line</summary>

| ID  | LoRA | IA3  | BitFit | Full Fine-Tuning | ACC  |
| --- | ---- | ---- | ------ | ---------------- | ---- |
| 5   | 5    | 5    | 5      | 5                | 5    |
| 10  | 10   | 10   | 10     | 10               | 10   |
| 30  | 30   | 30   | 30     | 30               | 30   |
| 50  | 50   | 50   | 50     | 50               | 50   |
| 70  | 70   | 70   | 70     | 70               | 70   |
| 90  | 90   | 90   | 90     | 90               | 90   |
</details>

(c) Data Subsetting   
Figure 39: ID (SQuAD) vs OOD (SQuAD-Shifts Reddit) Agreement trend from a single GPT2 for different PEFT methods (LoRA, IA3, BitFit). The accuracy line of all methods combined is shown in dotted gray while the agreement lines of each method is drawn as separate colors.