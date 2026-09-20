# Thinking Racial Bias in Fair Forgery Detection: Models, Datasets and Evaluations

Decheng Liu $^{1*}$ , Zongqi Wang $^{2*}$ , Chunlei Peng $^{1\dagger}$ Nannan Wang $^{1}$ , Ruimin Hu $^{1}$ , Xinbo Gao $^{3}$

$^{1}$ Xidian University, Xi'an, China

$^{2}$ Tsinghua University, Beijing, China

$^{3}$ Chongqing University of Posts and Telecommunications, Chongqing, China

dchliu@xidian.edu.cn, zq-wang24@mails.tsinghua.edu.cn, clpeng@xidian.edu.cn, nnwang@xidian.edu.cn,

hrm1964@163.com, gaoxb@cqupt.edu.cn

# Abstract

Due to the successful development of deep image generation technology, forgery detection plays a more important role in social and economic security. Racial bias has not been explored thoroughly in the deep forgery detection field. In the paper, we first contribute a dedicated dataset called the Fair Forgery Detection (FairFD) dataset, where we prove the racial bias of public state-of-the-art (SOTA) methods. Different from existing forgery detection datasets, the self-constructed FairFD dataset contains a balanced racial ratio and diverse forgery generation images with the largest-scale subjects. Additionally, we identify the problems with naive fairness metrics when benchmarking forgery detection models. To comprehensively evaluate fairness, we design novel metrics including Approach Averaged Metric and Utility Regularized Metric, which can avoid deceptive results. We also present an effective and robust post-processing technique, Bias Pruning with Fair Activations (BPFA), which improves fairness without requiring retraining or weight updates. Extensive experiments conducted with 12 representative forgery detection models demonstrate the value of the proposed dataset and the reasonability of the designed fairness metrics. By applying the BPFA to the existing fairest detector, we achieve a new SOTA. Furthermore, we conduct more in-depth analyses to offer more insights to inspire researchers in the community. The source code is available at https://github.com/liudan193/Fairness-Benchmark-for-Face-Forgery-Detection.

# Introduction

Face forgery refers to the creation of fake images or videos of a person's face using conventional techniques or deep learning methods. These forgeries can be used to spread misinformation, commit fraud, or even blackmail people. There are numerous methods are proposed for detecting face forgery [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]. Although an increasing number of advanced face forgery detection technologies are being developed, the racial fairness of these detectors is consistently overlooked by researchers [15]. Detectors with severe racial bias can lead to significant social impact. These detectors might disproportionately label faces from a particular racial group as fake, thereby indicating discrimination towards this particular racial group. Therefore, when a detector is ready for deployment, evaluating and analyzing its fairness is a crucial process. However, although there is extensive available research about the fairness in machine learning to draw upon $[16, 17, 18, 19, 20]$ , evaluating the fairness in face forgery detection systems remains difficult. This is due to several distinct differences between face forgery detection and other deep learning tasks.

This work aims to fill the gap in research on racial fairness in face forgery detection by proposing an accurate, comprehensive and credible fairness evaluation system. To achieve this goal, we analyze the shortcomings of existing evaluation components (i.e. dataset and metric), and our corresponding solutions. (1) Dataset. Existing face forgery detection datasets have a limited number of subjects. We find performance fluctuations significantly across subjects, so individual fairness may overshadow group fairness, which will make the evaluation results inaccurate. It also can be found that different forgery approaches have different fairness levels, limited forgery approaches will lead to a non-comprehensive result. Otherwise, undefined ethnicity (faces from two ethnicities are swapped) will lead to an inaccurate result. (2) Metric. We also propose two issues (Bias Offset and Aggregation Distortion) that will cause deceptive results. Bias Offset arises because existing fairness metrics typically use overall average accuracy for calculations instead of assessing each forgery method separately. This way may obscure some biases as different forgery methods may have different privileged races. Aggregation Distortion arises because detectors often show significant utility variations across different forgery techniques. Even if two forgery methods exhibit the same bias, detectors with lower utility can be more unfair. Treating each forgery method as equal will lead to unreliable results.

To tackle these problems, we firstly introduce the Fair Forgery Detection (FairFD) dataset for racial bias evaluation, which contains the largest scale subjects, and incorporates diverse forgery approaches including Face Swapping: FaceSwap [21], SimSwap [22], Expression Reenactment: FastReen [23], DualReen [24], Face Editing: MaskGAN [25], StarGAN [26], StyleGAN [27], Diffusion-Based: SDSwap [28], DCFace [29], Face2Diffusion [30] and Transformer-Based: FSRT [31].

![](images/248c39cf0996afe9f32dab82b5860d79526439fbd9d76a09f5b2346ddc2949f5.jpg)

<details>
<summary>bar</summary>

| Evaluation Dataset | Caucasian | Asian | African | Indian |
| --- | --- | --- | --- | --- |
| Real Face | 0.85 | 0.65 | 1.05 | 0.75 |
| Face Swap | 0.90 | 0.95 | 0.70 | 0.60 |
| Expression Reenactment | 0.75 | 0.85 | 0.55 | 0.65 |
| Attribute Manipulation | 0.45 | 0.55 | 0.40 | 0.35 |
| Stable Diffusion | 0.25 | 0.20 | 0.30 | 0.35 |
The image contains a visual annotation pointing to a speech bubble, suggesting a contextual or thematic link between racial bias and fairness metrics. Below it is a table listing specific metrics (e.g., Naive Metric, Approach Averaged Metric) and their corresponding values.
</details>

Figure 1: Workflow of fairness evaluation in forgery detection. We first construct an evaluation dataset containing a large number of subjects, diverse forgery approaches, and racial balance. Subsequently, we obtain the test results of the forgery detector on each race and forgery method. Finally, we comprehensively evaluate the detector using three sets of 12 fairness metrics in total.

And the self-constructed FairFD dataset does not have any undefined ethnicity annotations. For the specific metric, we address the mentioned two issues by introducing the Approach Averaged Metric, which calculates fairness separately for each forgery approach and then aggregates them, and the Utility Regularized Metric, which uses the utility to regularize the fairness.

In addition to our evaluation system, we also introduce a new pruning approach called BPFA (Bias Pruning with Fair Activations). The designed BPFA leverages an innovative pruning metric to identify weights with the least impact on model utility while contributing most significantly to bias (e.g., racial bias). By pruning these weights, BPFA successfully enhances fairness without compromising utility. As a post-processing method, BPFA enhances fairness without any retraining. And it can be applied to any detector, including those already trained with existing fairness learning strategies, to further improve fairness performance. The workflow of fairness evaluation is illustrated in Figure 1. Sufficient experimental results prove that the proposed BPFA is an efficient, plug-and-play and robust pruning scheme, outperforming other baseline pruning methods by a significant margin.

The key contributions are summarized as follows:

\- To our knowledge, it is the early exploration to introduce a comprehensive racial bias evaluation benchmark for forgery detection, providing a large-scale dataset, fairness metrics, and evaluation protocols. We newly introduce the Fair Forgery Detection (FairFD) dataset for racial bias in forgery detection evaluation, which con-

tains the largest scale subjects, race-balanced ratio and incorporates diverse forgery approaches.

- We identify the bias offset and aggregation distortion problems with naive fairness metrics. Following, the novel Approach Averaged Metric and Utility Regularized Metric are designed to address the mentioned issues. Extensive experimental results demonstrate the limited fairness of existing SOTA methods and validate the value of our proposed metric.   
- We propose the Bias Pruning with Fair Activations (BPFA) method to improve the fairness of forgery detectors. Extensive experiments demonstrate the advantages of our method, particularly its efficiency, plug-and-play nature and robustness. By combining BPFA with the fairest detector, we achieve a new SOTA in racial fairness performance. Specifically, we offer in-depth analyses and insightful observations to advance the community.

# Related Work

# Fairness in Face Forgery Detection

Fairness Algorithm. Fairness in face forgery detection is a relatively novel topic. DAG-FDD [37] is first proposed to address fairness without demographic information by setting a probability threshold for minority groups to ensure low error rates for all groups meeting this threshold. DAW-FDD [37] utilizes demographic information to design losses to ensure similar performance across specified groups. PFGDFD [38] improves fairness by using disentanglement loss to separate demographic and forgery features.

<table><tr><td rowspan="2">Dataset</td><td colspan="5">Race Rate</td><td rowspan="2">Race Balance</td><td rowspan="2">Undefined Ethnicity</td><td rowspan="2">Subject Number</td><td rowspan="2">Approach</td><td rowspan="2">Real Img Number</td><td rowspan="2">Fake Img Number</td></tr><tr><td>Caucasian</td><td>Asian</td><td>Indian</td><td>African</td><td>Others</td></tr><tr><td>FF++ [1]</td><td>~43.9%</td><td>~16.8%</td><td>~3.2%</td><td>~3.8%</td><td>~32.3%</td><td>✕</td><td>Yes</td><td>~1000</td><td>1</td><td>73k</td><td>266k</td></tr><tr><td>UADFV [32]</td><td>97.96%</td><td>2.04%</td><td>0</td><td>0</td><td>0</td><td>✕</td><td>Yes</td><td>49</td><td>1</td><td>241</td><td>252</td></tr><tr><td>CelebDF-v2 [33]</td><td>88.10%</td><td>5.10%</td><td>0</td><td>6.80%</td><td>0</td><td>✕</td><td>Yes</td><td>59</td><td>1</td><td>225k</td><td>2,116k</td></tr><tr><td>DFDC [34]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>✕</td><td>Yes</td><td>960</td><td>8</td><td>488k</td><td>1,783k</td></tr><tr><td>DF-1.0 [35]</td><td>~25%</td><td>~25%</td><td>~25%</td><td>~25%</td><td>0</td><td>√</td><td>Yes</td><td>100</td><td>1</td><td colspan="2">total 17,600k</td></tr><tr><td>ForgeryNet [36]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>✕</td><td>Yes</td><td>5400</td><td>15</td><td>1438k</td><td>1457k</td></tr><tr><td>FairFD(ours)</td><td>~25%</td><td>~25%</td><td>~25%</td><td>~25%</td><td>0</td><td>√</td><td>No</td><td>11430</td><td>11</td><td>52k</td><td>572k</td></tr></table>

Table 1: Face Forgery Detection Dataset Comparison. Our dataset is race-balanced, with no undefined races, the maximized number of subjects and forgery approaches exhibit diversity.

Deepfake Dataset. We summarize the information of existing datasets in Table 1. We provide the proportions of each race, along with whether the datasets are race-balanced. Additionally, we present whether undefined ethnicity faces are included (i.e., faces from one race are replaced with another race. We also supply the number of subjects, the number of forgery approaches, and the total number of frames. DAG(W)-FDD [37] and PFGDFD [38] directly use several of the datasets mentioned above as test data. Our work reveals inherent limitations when using these datasets for evaluation. There is currently no suitable dataset to evaluate the fairness of forgery detection. We also give details of widely used face forgery datasets in the section "Face Forgery Detection Datasets" in Supp.

Fairness Metric. To evaluate racial fairness, what we require is group fairness metrics. There are various group fairness metrics, and the selection of a metric depends on the application context. In this work, we consider the commonly used metrics including DPD [16, 17], DEOdds [16], DEO [18], STD [39, 40, 41, 42, 43] and our proposed novel fairness metrics. The definitions of these metrics can be found in the section "Existing Fairness Metrics" in Supp.

# FairFD Dataset

# Limitations of Current Datasets

Limited Number of Subjects. The construction process of the existing face forgery detection datasets involves collecting videos, subsequently creating forgeries, and then extracting frames. As a result, there are typically a small number of subjects and each subject has a large number of frames in these datasets. The limited number of subjects makes it challenging to draw meaningful comparisons across groups. Furthermore, we conducted the verification experiment to analyze and prove it in the section "Number of Subjects is Limited" in Supp.

Lack of Diversity of Forgery Approaches. In Table 1, only DFDC and ForgeryNet employ a variety of forgery techniques. However, we find that different forgery methods have different fairness levels. We validate this point in the subsequent Figure 4. Thus, we should strive to diversify forgery methods as much as possible, enabling a more comprehensive evaluation of the system's fairness.

Undefined Attribute Annotation. For these identity-replaced forgery approaches, there is a possibility of faces from one ethnicity being replaced with those from another. In related work [44], this phenomenon is referred to as "undefined attribute annotation." Undefined attributes can also significantly lower the quality of the evaluation of racial fairness, which is ignored in existing face forgery detection datasets.

# FairFD Description

Considering the mentioned limitations in existing datasets, we introduce our dataset, FairFD, aiming to address these shortcomings. FairFD endeavors to overcome previous challenges and provide a more accurate, reliable and comprehensive benchmark for evaluating fairness in face forgery detection. Representative examples of ours are presented in Figure 1. The overview of FairFD can be shown in the Table 1. Subsequently, we delve into several pivotal facets of our dataset.

Our dataset is an image-level dataset, and for each image, there are 11 kinds of corresponding forgery images, i.e., Face Swapping: FaceSwap [21], SimSwap [22], Expression Reenactment: FastReen [23], DualReen [24], Face Editing: StarGAN [26], StyleGAN [27], MaskGAN [25], Diffusion-Based: SDSwap [28], DCFace [29], Face2Diffusion [30] and Transformer-Based: FSRT [31]. In addition to the forgery approach label, our approach also includes labels for four ethnicities (i.e., Caucasian, Asian, African, and Indian). Each ethnicity contains approximately 3000 subjects.

# Source Data Collection and Forgery Process

To align with our requirements, which include having racial labels, and containing a sufficient number of subjects, we use the RFW [39] dataset as pristine images. The RFW dataset comprises face images with four racial labels (i.e., Caucasian, Asian, African, and Indian), containing approximately 3000 subjects for each racial group, with a roughly equal distribution. Each subject has approximately $3 \sim 7$ images. All images in the RFW dataset have a resolution of $400 \times 400$ pixels. Besides, the images are carefully selected to maintain similar distributions in terms of age, gender, yaw angle, and pitch angle. See details in "Detailed Distribution of FairFD" in Supp. To reduce the human resources, we use RFW as the source data.

To achieve the goal of diversity, we choose various approaches and techniques. We classify the forgery methods

![](images/bd9dc55161a6d51cd72303af38640d6f687c10a8c1d3029b05a3a1afecb99159.jpg)

<details>
<summary>line</summary>

| Ethnicity | FA 1  | FA 2  | Avg   |
| --------- | ----- | ----- | ----- |
| Caucasian | 85    | 40    | 63    |
| Asian     | 75    | 50    | 63    |
| African   | 80    | 63    | 72    |
| Indian    | 65    | 50    | 58    |
</details>

Figure 2: A face forgery detection model exhibits different biases for different forgery approaches.

into face swap, expression reenactment, attribute manipulation and advanced forgery methods(stable diffusion and transformer). See details in the section "Classification of Forgery Approaches" in Supp. We reimplement these methods and apply them to the source data. Details configuration and process of forgery crafting can be found in the section "Forgery Crafting Process" in Supp.

# The Proposed Evaluation Metrics

Even though we have obtained a reliable dataset for evaluating deepfake detection's fairness, we still can not get a credible evaluation result due to existing fairness metrics having two flaws. Firstly, the bias offset may lead to an underestimation of racial bias. Secondly, aggregation distortion can result in biased evaluation outcomes favoring specific forgery approaches. Below, we introduce the two flaws and their corresponding solutions respectively.

We make corrections to four widely used metrics: DPD [16], DEOdds [16], DEO [18], and STD [39]. For clarity, we leverage DPD to introduce our new metric in the following discussions as an example. The following is the definition of DPD:

$$
D P D = \max _ {s, s ^ {\prime} \in \mathbb {S}, s \neq s ^ {\prime}} \left| P (\hat {Y} \mid S = s) - P (\hat {Y} \mid S = s ^ {\prime}) \right|, \tag {1}
$$

where $\hat{Y}$ is the predicted labels. S represents the set of sensitive attributes, $s \in S$ and S = {Caucasian, Asian, Indian, African}.

# Bias Offset Problem

Bias offset refers to bias that will be partially obscured due to the calculation process of existing fairness metrics. Existing fairness metrics do not calculate separately for each forgery method instead of the final averaged performance scores. However, this way may obscure certain biases, which we call bias offset. Taking an example, in Figure 2, face forgery detectors may exhibit different biases for various forgery approaches. We calculate AccGap (the maximum differences in accuracy) and STD (standard deviation) for each forgery approach. In this example, both Forgery Approach 1 (FA1) and Forgery Approach 2 (FA2) exhibit an AccGap greater than 0.2 and an STD greater than 0.07. However, for Forgery Approach 1(FA1), the performance of Caucasians is better than Asians, while for FA2, the performance of Asians is better than Caucasians. In this situation, when calculating the fairness score using the final averaged performance scores, bias is to some extent offset, resulting in a smaller bias score. AccGap is less than 0.2, and the STD is less than 0.07 calculated using average accuracy. We refer to this phenomenon as bias offset. A more reliable way is to calculate fairness metrics separately for each forgery method and average them. We call this novel strategy as Approach Averaged Metric.

$$
\begin{array}{l} \text { AADPD } = \frac {1}{| \mathbb {F} |} \sum_ {f \in \mathbb {F}} \max _ {s, s ^ {\prime} \in \mathbb {S}, s \neq s ^ {\prime}} \left| P (\hat {Y} \mid S = s, F = f) \right. \\ - P \left(\hat {Y} \mid S = s ^ {\prime}, F = f\right) |, \tag {2} \\ \end{array}
$$

where $\mathbb{F}$ donates real face and forgery approaches. $f\in \mathbb{F}$ and $\mathbb{F}$ is the set of forgery methods.

# Aggregation Distortion Problem

Various forgery approaches not only exhibit different fairness situations but also demonstrate distinct levels of performance. For example, in Figure. 3 and Figure. 4, this is clearly evident. We identify the aggregation distortion problem where even if we calculate fairness scores separately for each forgery method and average them together, the averaged result can achieve a distorted fairness score due to the performance difference.

Directly averaging fairness scores when employing common fairness metrics might lead to misleading conclusions. For instance, consider two approaches with the accuracy of 20% and 80% respectively. We assume that both approaches yield a bias of 10% if we employ DEO as the fairness metric. Then, we calculate a simple average, which is also 10%. This would lead us to focus solely on the absolute differences in error rates without taking into account the variations in baselines. If the racial biases are both calculated to be 10%, the forgery method with only a 20% accuracy would evidently be much more unfair. This oversimplified average fails to capture the substantial disparity in performance between the two methods. To address this issue, we propose a fixed version. For each forgery approach, we have:

$$
\begin{array}{l} \text { URDPD } = \frac {1}{| \mathbb {F} |} \sum_ {f \in \mathbb {F}} \max _ {s, s ^ {\prime} \in \mathbb {S}, s \neq s ^ {\prime}} | P (\hat {Y} | S = s, F = f) \\ - P \left(\hat {Y} \mid S = s ^ {\prime}, F = f\right) \bigg | / A C C _ {F = f}, \tag {3} \\ \end{array}
$$

where $ACC_{F = f}$ calculates the accuracy of given different forgery approaches.

After applying Eq. 3 to each forgery method, we calculate their results and then obtain the final fairness score by averaging them. We refer to this approach as Utility Regularized Metric. This nuanced method acknowledges the significance of each forgery approach, providing a more accurate and insightful evaluation of fairness in the context of the diverse

fairness and performance landscape. In summary, we recommend not relying on a single fairness metric but rather considering a combination of multiple metrics to collectively reflect the fairness of a detector.

# Bias Pruning with Fair Activations

In this section, we present the Bias Pruning with Fair Activations (BPFA) approach, which develops a novel pruning metric combining weights and the fairness of activations to determine weight importance. Then, we prune those with the lowest pruning scores based on a predefined pruning rate by layer. Here we utilize an unstructured pruning strategy. Noting that the proposed BPFA can be directly extended to process other biases except for racial bias.

Pruning Metric. Consider a convolutional layer weights W of shape $(C_{out}, C_{in}, S_{Ker}^{h}, S_{Ker}^{w})$ , where $C_{out}$ represents the number of output filters, and each filter has dimension $(C_{in}, S_{Ker}^{h}, S_{Ker}^{w})$ . For one data sample, the output of this layer is denoted as X with shape $(C_{out}, S_{out}^{h}, S_{out}^{w})$ . The L2 norm by filter of X is represented as $\|X\|_{2} \in R^{C_{out}}$ . We compute the average L2 norm across all samples from a specific race $s \in S$ , denoted by $Z^{s} = \|X\|_{2}^{s}$ . For each filter, we then calculate the standard deviation of these norms across all races, which serves as the bias for that filter:

$$
B I A S _ {i} = \operatorname{std} \left(\left\{Z _ {i} ^ {s} \right\} _ {s \in \mathbb {S}}\right), \tag {4}
$$

where $std(\cdot)$ denotes the standard deviation. This bias measures the variability of outputs across different races. The pruning score (PS) for each weight $W_{ijkm}$ in the convolutional layer at the position $(i,j,k,m)$ is then calculated by combining the weight with respect to the computed bias:

$$
P S _ {i j k m} = \frac {\left| W _ {i j k m} \right|}{B I A S _ {i}}. \tag {5}
$$

By comparing the pruning scores, we can identify and potentially remove weights that have the least impact on model utility but the greatest impact on bias. This allows us to reduce bias and improve fairness without sacrificing performance. Note that while our method is illustrated using convolutional layers as an example, it can be easily extended to linear layers.

# Benchmark Experiments

# Experimental Setup

Dataset. We use FF++ (c23) as our training set. Specifically, for each video, we select 32 frames, crop the facial region, and finally resize it to $256 \times 256$ . We utilize the preprocessed data provided by [45], which has already undergone the aforementioned operations. Our proposed new dataset serves as the testing set. As our dataset inherently consists of face images with backgrounds and bodies removed, there is no need for additional face cropping. Subsequently, we resize the images to $256 \times 256$ for inference. Note that we still provide the original dataset with a resolution of $400 \times 400$ for scenarios requiring higher resolution.

Algorithms. We summarize the face forgery detection algorithms in the section "Face Forgery Detection Algorithms Categories” in Supp. For a comprehensive and fair analysis, we select several representative algorithms. For spatial-based detectors, we choose Xception [1], RECCE [10], UCF [11], Capsule [5], FFD [8] and CORE [9]. For frequency-based detectors, we select F3Net [12], SPSL [13] and SRM [14]. For fairness-enhanced detectors, we select DAG [37](Xception as base model), DAW [37](Xception as base model) and PFGDFD [38](UCF as base model). In detail, these models are trained with the Adam optimization algorithm with a learning rate of 0.0002 and an epoch number of 10. The batch size is 32. And data augmentation methods including image compression, horizontal flip and rotation are applied. However, when applying these data augmentation methods to DAG, we find that its fairness level significantly deteriorated. For a fair comparison, we report below the results using data augmentation. Meanwhile, the results without data augmentation are presented in the section "Results without Data Augmentation" in Supp.

# Benchmarking Fairness of Face Forgery Detectors

Benchmark Results. The benchmark results (shown in Table 2) present a comprehensive evaluation of 12 face forgery detectors using various fairness metrics. We highlight the four fairest detectors using different colors. Based on the results, we draw the following significant observations: (1) Current face forgery detectors all exhibit a high degree of racial bias. The DPD metric of SPSL can achieve 0.0203 with the smallest racial bias. This indicates that the difference in the probability of classifying faces as fake between the most advantaged and disadvantaged groups is 2.03%. When separately calculating and averaging for each forgery method, the AADPD metric reaches 5.56%. On the other hand, for the least fair detector UCF, the difference in the probability of classifying faces as fake between the most advantaged and disadvantaged groups is 17.65%. This reminds researchers to address the racial bias in existing face forgery detection models. (2) Current face forgery detectors have racial bias variation. Comparing the least fair detector UCF with the most fair detector SPSL, the former's URDPD is 4.76 times that of the latter, URDEOdds is 3.58 times, URDEO 4.98 times, and URSTD is 4.48 times, showing significant gap in racial bias. Other detectors also exhibit varying degrees of racial bias. Furthermore, we observe that three frequency-based detectors, SPSL, F3Net, and SRM, consistently demonstrate a smaller racial bias across all fairness metrics. We conduct an in-depth investigation into this in the section "Analyses and Discussions" in Supp.

Detailed Utility Results. To present more detailed results, we present the AUC for each detector, each race, and each forgery method in Figure 3. Results show that different forgery methods exhibit varying levels of utility. This validates the advantage of Utility Regularized Metric.

Detailed Fairness Results. We present the standard deviation (STD) of the ACC for the four races in Figure 4 for each forgery method(including Real Face). Our findings reveal that different forgery methods exhibit varying levels of fairness, and different detectors rank the fairness of these forgery methods differently. This validates the advantage of the proposed Approach Averaged Metric.

<table><tr><td rowspan="2" colspan="2">Fairness Metric</td><td colspan="6">Spatial-based</td><td colspan="3">Frequency-based</td><td colspan="3">Fairness-enhanced</td></tr><tr><td>Xception</td><td>RECCE</td><td>UCF</td><td>Capsule</td><td>FFD</td><td>CORE</td><td>F3Net</td><td>SPSL</td><td>SRM</td><td>DAG</td><td>DAW</td><td>PFGDFD</td></tr><tr><td rowspan="4">Naive Metric</td><td>DPD↓</td><td>0.1810</td><td>0.1338</td><td>0.1765</td><td>0.0969</td><td>0.1099</td><td>0.0951</td><td>0.0674</td><td>0.0203</td><td>0.0990</td><td>0.1723</td><td>0.0513</td><td>0.0805</td></tr><tr><td>DEOdds↓</td><td>0.1666</td><td>0.1264</td><td>0.1495</td><td>0.0902</td><td>0.1005</td><td>0.0798</td><td>0.0763</td><td>0.0304</td><td>0.0714</td><td>0.2288</td><td>0.0593</td><td>0.1396</td></tr><tr><td>DEO↓</td><td>0.2088</td><td>0.1548</td><td>0.2014</td><td>0.1118</td><td>0.1242</td><td>0.1084</td><td>0.0801</td><td>0.0215</td><td>0.1090</td><td>0.2105</td><td>0.0611</td><td>0.1032</td></tr><tr><td>STD↓</td><td>0.0647</td><td>0.0474</td><td>0.0631</td><td>0.0343</td><td>0.0398</td><td>0.0342</td><td>0.0265</td><td>0.0080</td><td>0.0355</td><td>0.0636</td><td>0.0195</td><td>0.0328</td></tr><tr><td rowspan="4">Approach Averaged Metric</td><td>AADPD↓</td><td>0.2024</td><td>0.1572</td><td>0.2175</td><td>0.1323</td><td>0.1552</td><td>0.1147</td><td>0.1158</td><td>0.0556</td><td>0.1413</td><td>0.2201</td><td>0.0735</td><td>0.1393</td></tr><tr><td>AADEOdds↓</td><td>0.1669</td><td>0.1302</td><td>0.1630</td><td>0.1034</td><td>0.1196</td><td>0.0858</td><td>0.0961</td><td>0.0481</td><td>0.0925</td><td>0.2324</td><td>0.0662</td><td>0.1560</td></tr><tr><td>AADEO↓</td><td>0.2095</td><td>0.1626</td><td>0.2284</td><td>0.1381</td><td>0.1623</td><td>0.1205</td><td>0.1197</td><td>0.0571</td><td>0.1511</td><td>0.2177</td><td>0.0749</td><td>0.1360</td></tr><tr><td>AASTD↓</td><td>0.0750</td><td>0.0578</td><td>0.0809</td><td>0.0493</td><td>0.0576</td><td>0.0449</td><td>0.0448</td><td>0.0219</td><td>0.0531</td><td>0.0834</td><td>0.0283</td><td>0.0530</td></tr><tr><td rowspan="4">Utility Regularized Metric</td><td>URDPD↓</td><td>0.1357</td><td>0.1118</td><td>0.1523</td><td>0.0808</td><td>0.1037</td><td>0.0803</td><td>0.0806</td><td>0.0320</td><td>0.0904</td><td>0.1474</td><td>0.0555</td><td>0.0881</td></tr><tr><td>URDEOdds↓</td><td>0.1057</td><td>0.0852</td><td>0.1069</td><td>0.0639</td><td>0.0763</td><td>0.0567</td><td>0.0625</td><td>0.0299</td><td>0.0584</td><td>0.1445</td><td>0.0440</td><td>0.0986</td></tr><tr><td>URDEO↓</td><td>0.1417</td><td>0.1171</td><td>0.1614</td><td>0.0842</td><td>0.1092</td><td>0.0850</td><td>0.0842</td><td>0.0324</td><td>0.0968</td><td>0.1480</td><td>0.0578</td><td>0.0860</td></tr><tr><td>URSTD↓</td><td>0.0501</td><td>0.0410</td><td>0.0565</td><td>0.0301</td><td>0.0384</td><td>0.0313</td><td>0.0312</td><td>0.0126</td><td>0.0339</td><td>0.0559</td><td>0.0214</td><td>0.0335</td></tr><tr><td>Utility</td><td>AUC↑</td><td>0.6911</td><td>0.6897</td><td>0.7214</td><td>0.6815</td><td>0.7304</td><td>0.6864</td><td>0.6564</td><td>0.6763</td><td>0.7102</td><td>0.6672</td><td>0.6604</td><td>0.6302</td></tr></table>

Table 2: Bias evaluation on FairFD for 12 face forgery detectors using Naive Metrics, Approach Averaged Metrics, Utility Regularized Metrics. For each row, the best values are underlined and bolded, followed by the second-best values which are underlined, bolded, and italicized, the third-best values are bolded, and the fourth-best values are bolded and italicized.

![](images/642a4299129bf16d84caaa31dced7861a0ec0cb23add9c4d5e9143ae7ad761f3.jpg)  
F1: FaceSwap F2: SimSwap F3: FastReen F4: DualReen F5: MaskGAN F6: StarGAN F7: StyGAN F8: SDSwap F9: DCFace F10: F2D F11: FSRT

Figure 3: Detailed utility (AUC) for diverse races, forgery approaches, detectors.

# Evaluating BPFA

Baseline Algorithm We select two baseline methods for comparison. The first WEIG uses only the absolute values of the weights as the pruning score, and the second RoBA uses only the reciprocal of the bias of the activations. More details about these baselines are shown in Supp. For all three methods, we prune the parameters with the lowest pruning scores. These pruning baselines can be directly applied in existing SOTA forgery detection models.

Results Analysis The experimental results under optimal pruning rates for each detector and method are shown in Table 3. It can be found that the proposed method BPFA consistently outperforms all baseline methods, achieving supe-

rior fairness without compromising utility. Although WEIG generally preserves good utility and enhances fairness, its improvements in fairness are not as significant as those achieved by BPFA. On the other hand, RoBA exhibits highly unstable performance. It results in improving fairness but at the cost of significantly reduced utility, which can render the model nearly unusable in the forgery detection task. These results prove the superior performance of BPFA in enhancing both utility and fairness for forgery detection. It is encouraging to find that SPSL+BPFA achieves the new state-of-the-art performance. The only parameter in our method is the pruning rate. The ablation study for pruning rate is detailed in three tables in the section "Ablation Study on Prun-

<table><tr><td rowspan="2" colspan="2">Method</td><td colspan="4">Naive Metric↓</td><td colspan="4">Approach Averaged Metric↓</td><td colspan="4">Utility Regularized Metric↓</td><td colspan="2">Utility↑</td></tr><tr><td>DPD</td><td>DEOdds</td><td>DEO</td><td>STD</td><td>MA DPD</td><td>MA DEOdds</td><td>MA DEO</td><td>MA STD</td><td>UR DPD</td><td>UR DEOdds</td><td>UR DEO</td><td>UR STD</td><td>AUC</td><td>ACC</td></tr><tr><td rowspan="4">SPSL</td><td>Original</td><td>0.0203</td><td>0.0304</td><td>0.0215</td><td>0.0080</td><td>0.0556</td><td>0.0481</td><td>0.0571</td><td>0.0219</td><td>0.0320</td><td>0.0299</td><td>0.0324</td><td>0.0126</td><td>0.6763</td><td>0.7618</td></tr><tr><td>WEIG</td><td>0.0183</td><td>0.0258</td><td>0.0201</td><td>0.0072</td><td>0.0564</td><td>0.0451</td><td>0.0586</td><td>0.0219</td><td>0.0324</td><td>0.0277</td><td>0.0334</td><td>0.0126</td><td>0.6769</td><td>0.7615</td></tr><tr><td>RoBA</td><td>0.1128</td><td>0.1598</td><td>0.1395</td><td>0.0445</td><td>0.1462</td><td>0.1616</td><td>0.1432</td><td>0.0583</td><td>0.0893</td><td>0.1024</td><td>0.0867</td><td>0.0356</td><td>0.6331</td><td>0.7037</td></tr><tr><td>BPFA</td><td>0.0181</td><td>0.0209</td><td>0.0200</td><td>0.0072</td><td>0.0473</td><td>0.0357</td><td>0.0496</td><td>0.0182</td><td>0.0265</td><td>0.0218</td><td>0.0275</td><td>0.0102</td><td>0.6862</td><td>0.8055</td></tr><tr><td rowspan="4">FFD</td><td>Original</td><td>0.1099</td><td>0.1005</td><td>0.1242</td><td>0.0398</td><td>0.1552</td><td>0.1196</td><td>0.1623</td><td>0.0576</td><td>0.1037</td><td>0.0763</td><td>0.1092</td><td>0.0384</td><td>0.7304</td><td>0.5751</td></tr><tr><td>WEIG</td><td>0.1098</td><td>0.1003</td><td>0.1240</td><td>0.0398</td><td>0.1550</td><td>0.1194</td><td>0.1621</td><td>0.0576</td><td>0.1035</td><td>0.0761</td><td>0.1090</td><td>0.0384</td><td>0.7304</td><td>0.5751</td></tr><tr><td>RoBA</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.5967</td><td>-</td></tr><tr><td>BPFA</td><td>0.1096</td><td>0.0999</td><td>0.1237</td><td>0.0397</td><td>0.1546</td><td>0.1189</td><td>0.1617</td><td>0.0574</td><td>0.1032</td><td>0.0758</td><td>0.1087</td><td>0.0382</td><td>0.7305</td><td>0.5760</td></tr><tr><td rowspan="4">PFG-DFD</td><td>Original</td><td>0.0805</td><td>0.1396</td><td>0.1032</td><td>0.0328</td><td>0.1393</td><td>0.1560</td><td>0.1360</td><td>0.0530</td><td>0.0881</td><td>0.0986</td><td>0.0860</td><td>0.0335</td><td>0.6302</td><td>0.6019</td></tr><tr><td>WEIG</td><td>0.0789</td><td>0.1340</td><td>0.1012</td><td>0.0319</td><td>0.1349</td><td>0.1494</td><td>0.1320</td><td>0.0513</td><td>0.0853</td><td>0.0944</td><td>0.0835</td><td>0.0324</td><td>0.6298</td><td>0.6021</td></tr><tr><td>RoBA</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.5468</td><td>-</td></tr><tr><td>BPFA</td><td>0.0594</td><td>0.1337</td><td>0.0796</td><td>0.0238</td><td>0.1079</td><td>0.1442</td><td>0.1006</td><td>0.0411</td><td>0.0644</td><td>0.0969</td><td>0.0578</td><td>0.0245</td><td>0.6445</td><td>0.7415</td></tr></table>

Table 3: Experiments with different fairness pruning methods. We highlight the best method for each metric in bold. And we use '-' to indicate methods that cause severe performance degradation, rendering the detector unusable even setting a pruning rate as low as 0.1%.

![](images/bd7fa106b060cb88ca999407c8e7980c0e23d30fcb8e0a5ffa0d8f2980cd60b2.jpg)  
Figure 4: Fairness (STD) for different forgery approaches.

ing Rate" in Supp. Our findings indicate that different detectors and different methods have significantly different optimal pruning rates (which correspond to the least decrease in utility (or even improvement) while achieving the best average value across the 12 fairness metrics.

# Analyses and Discussions

Here we conduct deeper analyses and give some insights: (1) We train detectors using balanced training data, finding that while racial bias can be reduced, the cost of collecting balanced data is substantial; (2) We set the optimal classification threshold for each race and then use the resulting accuracy values to calculate fairness. The conclusions show that this method is highly cost-effective and significantly improves both utility and fairness simultaneously; (3) We analyze that frequency-based detectors exhibit superior fairness performance because of not utilizing race-sensitive informa-

tion, e.g. color. More analyses and details are shown in Supp.

# Conclusion

This paper early explores a comprehensive racial bias evaluation benchmark for forgery detection, which provides a newly self-construct dataset, fairness metric and unified protocols. We identify numerous disadvantages in existing datasets and fairness metrics, then propose a novel dataset FairFD dataset and two sets of fairness metrics to address these mentioned issues. Besides, we also propose a novel Bias Pruning with Fair Activations algorithm to improve the fairness performance without an extra training process. Emphatically, we evaluate the fairness of multiple existing face forgery detectors. The results indicate the racial bias in current detectors is generally high and prove the advantages of our proposed BPFA. Further analyses reveal some interesting insights into the emergence of racial bias. We hope the proposed benchmark can inspire more researchers to develop the field. In the future, we will explore a unified fairness metric for diverse biases in more kinds of datasets, and construct the video-level forgery detection datasets for more real applications. See social impact in Supp.

# Acknowledgments

This work was supported in part by the National Natural Science Foundation of China under Grant 62306227, Grant 62276198, Grant U22A2035, Grants U22A2096, Grant 62441601 and Grant 62036007; in part by the Fundamental Research Funds for the Central Universities under Grant ZYTS24142, Grant QTZX23083 and Grant QTZX23042; in part by the Key Research and Development Program of Shaanxi (Program No. 2023-YBGY-231); in part by Young Elite Scientists Sponsorship Program by CAST under Grant 2022QNRC001; in part by the Guangxi Natural Science Foundation Program under Grant 2021GXNSFDA075011;

in part by the Shaanxi Province Core Technology Research and Development Project under grant 2024QY2-GJHX-11; in part by Open Research Project of Key Laboratory of Artificial Intelligence Ministry of Education under Grant AI202401, in part by the Nanning Scientific Research and Technological Development Project 20231042; in part by the ‘111 Center’ (B16037).

# References

[1] Andreas Rossler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, and Matthias Nießner. Faceforensics++: Learning to detect manipulated facial images. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1–11, 2019.   
[2] Darius Afchar, Vincent Nozick, Junichi Yamagishi, and Isao Echizen. Mesonet: a compact facial video forgery detection network. In 2018 IEEE international workshop on information forensics and security (WIFS), pages 1–7. IEEE, 2018.   
[3] Sheng-Yu Wang, Oliver Wang, Richard Zhang, Andrew Owens, and Alexei A Efros. Cnn-generated images are surprisingly easy to spot... for now. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8695–8704, 2020.   
[4] Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning, pages 6105–6114. PMLR, 2019.   
[5] Huy H Nguyen, Junichi Yamagishi, and Isao Echizen. Capsule-forensics: Using capsule networks to detect forged images and videos. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 2307–2311. IEEE, 2019.   
[6] Yuezun Li and Siwei Lyu. Exposing deepfake videos by detecting face warping artifacts. arxiv 2018. arXiv preprint arXiv:1811.00656, 1811.   
[7] Lingzhi Li, Jianmin Bao, Ting Zhang, Hao Yang, Dong Chen, Fang Wen, and Baining Guo. Face x-ray for more general face forgery detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5001–5010, 2020.   
[8] Hao Dang, Feng Liu, Joel Stehouwer, Xiaoming Liu, and Anil K Jain. On the detection of digital face manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern recognition, pages 5781–5790, 2020.   
[9] Yunsheng Ni, Depu Meng, Changqian Yu, Chengbin Quan, Dongchun Ren, and Youjian Zhao. Core: Consistent representation learning for face forgery detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12–21, 2022.   
[10] Junyi Cao, Chao Ma, Taiping Yao, Shen Chen, Shouhong Ding, and Xiaokang Yang. End-to-end reconstruction-classification learning for face forgery

detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4113–4122, 2022.   
[11] Zhiyuan Yan, Yong Zhang, Yanbo Fan, and Baoyuan Wu. Ucf: Uncovering common features for generalizable deepfake detection. arXiv preprint arXiv:2304.13949, 2023.   
[12] Yuyang Qian, Guojun Yin, Lu Sheng, Zixuan Chen, and Jing Shao. Thinking in frequency: Face forgery detection by mining frequency-aware clues. In European conference on computer vision, pages 86–103. Springer, 2020.   
[13] Honggu Liu, Xiaodan Li, Wenbo Zhou, Yuefeng Chen, Yuan He, Hui Xue, Weiming Zhang, and Nenghai Yu. Spatial-phase shallow learning: rethinking face forgery detection in frequency domain. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 772–781, 2021.   
[14] Yuchen Luo, Yong Zhang, Junchi Yan, and Wei Liu. Generalizing face forgery detection with high-frequency features. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16317–16326, 2021.   
[15] Momina Masood, Mariam Nawaz, Khalid Mahmood Malik, Ali Javed, Aun Irtaza, and Hafiz Malik. Deep-fakes generation and detection: State-of-the-art, open challenges, countermeasures, and way forward. Applied intelligence, 53(4):3974–4026, 2023.   
[16] Alekh Agarwal, Alina Beygelzimer, Miroslav Dudík, John Langford, and Hanna Wallach. A reductions approach to fair classification. In International conference on machine learning, pages 60–69. PMLR, 2018.   
[17] Alekh Agarwal, Miroslav Dudík, and Zhiwei Steven Wu. Fair regression: Quantitative definitions and reduction-based algorithms. In International Conference on Machine Learning, pages 120–129. PMLR, 2019.   
[18] Moritz Hardt, Eric Price, and Nati Srebro. Equality of opportunity in supervised learning. Advances in neural information processing systems, 29, 2016.   
[19] Yu Tian, Min Shi, Yan Luo, Ava Kouhana, Tobias Elze, and Mengyu Wang. Harvard fairseg: A large-scale medical image segmentation dataset for fairness learning using segment anything model with fair error-bound scaling. In International Conference on Learning Representations (ICLR), 2024.   
[20] Xudong Shen, Chao Du, Tianyu Pang, Min Lin, Yongkang Wong, and Mohan Kankanhalli. Finetuning text-to-image diffusion models for fairness. In International Conference on Learning Representations (ICLR), 2024.   
[21] Marek Kowalski. Faceswap. https://github.com/MarekKowalski/FaceSwap, 2016.   
[22] Renwang Chen, Xuanhong Chen, Bingbing Ni, and Yanhao Ge. Simswap: An efficient framework for high fidelity face swapping. In Proceedings of the 28th ACM

International Conference on Multimedia, pages 2003-2011, 2020.   
[23] Egor Zakharov, Aleksei Ivakhnenko, Aliaksandra Shysheya, and Victor Lempitsky. Fast bi-layer neural synthesis of one-shot realistic head avatars. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XII 16, pages 524–540. Springer, 2020.   
[24] Gee-Sern Hsu, Chun-Hung Tsai, and Hung-Yi Wu. Dual-generator face reenactment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 642–650, 2022.   
[25] Cheng-Han Lee, Ziwei Liu, Lingyun Wu, and Ping Luo. Maskgan: Towards diverse and interactive facial image manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5549–5558, 2020.   
[26] Yunjey Choi, Minje Choi, Munyoung Kim, Jung-Woo Ha, Sunghun Kim, and Jaegul Choo. Stargan: Unified generative adversarial networks for multidomain image-to-image translation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8789–8797, 2018.   
[27] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401-4410, 2019.   
[28] Tran Xen. Diffusionfaceswap. https://github.com/glucauze/sd-webui-faceswaplab, 2023.   
[29] Minchul Kim, Feng Liu, Anil Jain, and Xiaoming Liu. Dcface: Synthetic face generation with dual condition diffusion model. In Proceedings of the ieee/cvf conference on computer vision and pattern recognition, pages 12715–12725, 2023.   
[30] Kaede Shiohara and Toshihiko Yamasaki. Face2diffusion for fast and editable face personalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6850–6859, 2024.   
[31] Andre Rochow, Max Schwarz, and Sven Behnke. Fsrt: Facial scene representation transformer for face reenactment from factorized appearance head-pose and facial expression features. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7716–7726, 2024.   
[32] Xin Yang, Yuezun Li, and Siwei Lyu. Exposing deep fakes using inconsistent head poses. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 8261-8265. IEEE, 2019.   
[33] Yuezun Li, Xin Yang, Pu Sun, Honggang Qi, and Siwei Lyu. Celeb-df: A large-scale challenging dataset for deepfake forensics. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3207–3216, 2020.

[34] Brian Dolhansky, Joanna Bitton, Ben Pflaum, Jikuo Lu, Russ Howes, Menglin Wang, and Cristian Canton Ferrer. The deepfake detection challenge (dfdc) dataset. arXiv preprint arXiv:2006.07397, 2020.   
[35] Liming Jiang, Ren Li, Wayne Wu, Chen Qian, and Chen Change Loy. Deeperforensics-1.0: A large-scale dataset for real-world face forgery detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2889–2898, 2020.   
[36] Yinan He, Bei Gan, Siyu Chen, Yichun Zhou, Guojun Yin, Luchuan Song, Lu Sheng, Jing Shao, and Ziwei Liu. Forgerynet: A versatile benchmark for comprehensive forgery analysis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4360–4369, 2021.   
[37] Yan Ju, Shu Hu, Shan Jia, George H Chen, and Siwei Lyu. Improving fairness in deepfake detection. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 4655–4665, 2024.   
[38] Li Lin, Xinan He, Yan Ju, Xin Wang, Feng Ding, and Shu Hu. Preserving fairness generalization in deepfake detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16815–16825, 2024.   
[39] Mei Wang, Weihong Deng, Jiani Hu, Xunqiang Tao, and Yaohai Huang. Racial faces in the wild: Reducing racial bias by information maximization adaptation network. In Proceedings of the ieee/cvf international conference on computer vision, pages 692–702, 2019.   
[40] Joseph P Robinson, Gennady Livitz, Yann Henon, Can Qin, Yun Fu, and Samson Timoner. Face recognition: too bias, or not too bias? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pages 0–1, 2020.   
[41] Sixue Gong, Xiaoming Liu, and Anil K Jain. Jointly de-biasing face recognition and demographic attribute estimation. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXIX 16, pages 330–347. Springer, 2020.   
[42] Jun Yu, Xinlong Hao, Haonian Xie, and Ye Yu. Fair face recognition using data balancing, enhancement and fusion. In Computer Vision–ECCV 2020 Workshops: Glasgow, UK, August 23–28, 2020, Proceedings, Part VI 16, pages 492–505. Springer, 2020.   
[43] Fu-En Wang, Chien-Yi Wang, Min Sun, and Shang-Hong Lai. Mixfairface: Towards ultimate fairness via mixfair adapter in face recognition. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 14531–14538, 2023.   
[44] Ying Xu, Philipp Terhörst, Kiran Raja, and Marius Pedersen. A comprehensive analysis of ai biases in deepfake detection with massively annotated databases. arXiv preprint arXiv:2208.05845, 2022.

[45] Zhiyuan Yan, Yong Zhang, Xinhang Yuan, Siwei Lyu, and Baoyuan Wu. Deepfakebench: A comprehensive benchmark of deepfake detection. arXiv preprint arXiv:2307.01426, 2023.   
[46] Loc Trinh and Yan Liu. An examination of fairness of ai models for deepfake detection. In Zhi-Hua Zhou, editor, Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21, pages 567–574. International Joint Conferences on Artificial Intelligence Organization, 8 2021. Main Track.   
[47] Zhifei Zhang, Yang Song, and Hairong Qi. Age progression/regression by conditional adversarial autoencoder. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2017.   
[48] Aakash Varma Nadimpalli and Ajita Rattani. Gbdf: Gender balanced deepfake dataset towards fair deepfake detection. In Pattern Recognition, Computer Vision, and Image Processing. ICPR 2022 International Workshops and Challenges: Montreal, QC, Canada, August 21–25, 2022, Proceedings, Part II, page 320–337, Berlin, Heidelberg, 2023. Springer-Verlag.   
[49] Ivan Perov, Daiheng Gao, Nikolay Chervoniy, Kunlin Liu, Sugasa Marangonda, Chris Umé, Mr Dpfks, Carl Shift Facenheim, Luis RP, Jian Jiang, et al. Deepfacelab: Integrated, flexible and extensible face-swapping framework. arXiv preprint arXiv:2005.05535, 2020.   
[50] Haiming Yu, Hao Zhu, Xiangju Lu, and Junhui Liu. Migrating face swap to mobile devices: a lightweight framework and a supervised training solution. In 2022 IEEE International Conference on Multimedia and Expo (ICME), pages 1–6. IEEE, 2022.   
[51] Wei Shen and Rujie Liu. Learning residual images for face attribute manipulation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4030-4038, 2017.   
[52] Jingxiang Sun, Xuan Wang, Yong Zhang, Xiaoyu Li, Qi Zhang, Yebin Liu, and Jue Wang. Fenerf: Face editing in neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7672–7682, 2022.   
[53] Yanbo Xu, Yueqin Yin, Liming Jiang, Qianyi Wu, Chengyao Zheng, Chen Change Loy, Bo Dai, and Wayne Wu. Transeditor: Transformer-based dual-space gan for highly controllable facial editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7683-7692, 2022.   
[54] Justus Thies, Michael Zollhöfer, Matthias Nießner, Levi Valgaerts, Marc Stamminger, and Christian Theobalt. Real-time expression transfer for facial reenactment. ACM Trans. Graph., 34(6):183–1, 2015.   
[55] Justus Thies, Michael Zollhofer, Marc Stamminger, Christian Theobalt, and Matthias Nießner. Face2face: Real-time face capture and reenactment of rgb videos.

In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2387-2395, 2016.   
[56] Wayne Wu, Yunxuan Zhang, Cheng Li, Chen Qian, and Chen Change Loy. Reenactgan: Learning to reenact faces via boundary transfer. In Proceedings of the European conference on computer vision (ECCV), pages 603–619, 2018.   
[57] David Gray Grant. Equalized odds is a requirement of algorithmic fairness. Synthese, 201(3):101, 2023.   
[58] Matt Tora. Deepfakes. https://github.com/deepfakes/faceswap, 2018.   
[59] Justus Thies, Michael Zollhöfer, and Matthias Nießner. Deferred neural rendering: Image synthesis using neural textures. Acm Transactions on Graphics (TOG), 38(4):1–12, 2019.   
[60] Zhaoyu Chen, Bo Li, Shuang Wu, Kaixun Jiang, Shouhong Ding, and Wenqiang Zhang. Content-based unrestricted adversarial attack. Advances in Neural Information Processing Systems, 36, 2024.   
[61] ZongQi Wang, Wenchao Xu, Haozhao Wang, and Nan Cheng. APD: Boosting adversarial transferability via perturbation dropout, 2024.   
[62] Tim Franzmeyer, Stephen Marcus McAleer, Joao F. Henriques, Jakob Nicolaus Foerster, Philip Torr, Adel Bibi, and Christian Schroeder de Witt. Illusory attacks: Detectability matters in adversarial attacks on sequential decision-makers. In The Twelfth International Conference on Learning Representations, 2024.   
[63] Zheng Xu, Yanxiang Zhang, Galen Andrew, Christopher A Choquette-Choo, Peter Kairouz, H Brendan McMahan, Jesse Rosenstock, and Yuanbo Zhang. Federated learning of gboard language models with differential privacy. arXiv preprint arXiv:2305.18465, 2023.   
[64] Dan Qiao and Yu-Xiang Wang. Offline reinforcement learning with differential privacy. Advances in Neural Information Processing Systems, 36, 2024.   
[65] Francesco Pittaluga and Bingbing Zhuang. Ldp-feat: Image features with local differential privacy. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17580–17590, 2023.   
[66] Mei Wang, Yaobin Zhang, and Weihong Deng. Meta balanced network for fair face recognition. IEEE transactions on pattern analysis and machine intelligence, 44(11):8433–8448, 2021.   
[67] Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for efficient neural network. Advances in neural information processing systems, 28, 2015.   
[68] Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In International Conference on Learning Representations, 2019.   
[69] Davis Blalock, Jose Javier Gonzalez Ortiz, Jonathan Frankle, and John Guttag. What is the state of neural network pruning? Proceedings of machine learning and systems, 2:129–146, 2020.

[70] Mingjie Sun, Zhuang Liu, Anna Bair, and J Zico Kolter. A simple and effective pruning approach for large language models. In The Twelfth International Conference on Learning Representations, 2024.   
[71] Kaede Shiohara and Toshihiko Yamasaki. Detecting deepfakes with self-blended images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18720–18729, 2022.

# Full Related Works

# Other Work on Fairness in Face Forgery Detection

The research on fairness in face forgery detection is still relatively limited and waits for further exploration. A preliminary study [46] investigates bias in three commonly used face forgery detectors. They created real face images by sampling from the RFW [39] and UTKFace [47]. Next, they generate fake faces by blending two faces. However, they do not explicitly consider any fairness metrics. This simple study lacks a reasonable evaluation system but still verifies face forgery detectors have a significant racial bias to some extent. Study in [44] annotates five popular deepfake detection datasets with age, gender, ethnicity, etc. Due to the racial imbalance of current datasets, they also propose a metric to deal with the unbalanced test dataset. Another study [48] create a gender-balanced dataset (GBDF) sampled from the FF++, Celeb-DF, and DF-1.0. This approach involves a limited number of subjects, which restricts the dataset's capability for fairness evaluations.

# Classification of Forgery Approaches

There are several approaches for creating fake faces. Here, we provide a brief overview of the classification of forgery methods:

1. Identity-replaced Forgery Approach refers to substituting the original identity in an image, e.g., FaceSwap.FaceSwap [21, 22, 49, 50] involves replacing the face of a person in a video or image with another person's face. The method usually uses deep learning algorithms to detect and extract the faces of the two people and then swaps them.

2. Identity-remained Forgery Approach retains the original identity and alters other facial attributes, e.g., Attribute Manipulation and Expression Reenactment. Attribute Manipulation [51, 26, 25, 52, 53] involves manipulating the attributes of a face, such as skin color, gender, and eye shape, while preserving the identity of the person. The method usually adopts a GAN. Expression Reenactment [54, 55, 56, 23, 24] involves transferring the facial expressions of one person to another person's face. The method usually detect and track the facial landmarks of the two people and then transfers the expression from the source face to the target face.

# Existing Fairness Metrics

Commonly, researchers use the performance metric difference between privileged groups and unprivileged groups as a fairness metric. Demographic Parity Difference (DPD) [16, [17] utilizes the difference in positive rate, which represents the proportion of data predicted to be positive, as its fairness metric. The difference in Equalized Odds (DE-Odds) [16] utilizes the average of the differences in true positive rate and false positive rate as its fairness metric. The difference in Equal Opportunity (DEO) [18] utilizes the difference in true positive rate solely as a fairness metric. In face recognition scenarios, the Standard Deviation (STD) of performance metrics across different groups is often used [39, 40, 41, 42, 43]. Equity-Scaled Segmentation Performance (ESSP) proposes to evaluate segmentation performance and group fairness simultaneously in medical image segmentation scenarios [19]. In the context of the generative model, the frequency of each group in the generated images is computed, and the average difference in frequency between each pair of groups is calculated as fairness metric [20]. Moreover, there are numerous works proposing more suitable metrics based on the application scenarios.

Due to the complexity of fairness evaluation, numerous fairness metrics are proposed from various perspectives to cater to different scenarios. Therefore, we need to leverage multiple fairness metrics to evaluate the fairness in face forgery detection comprehensively. Here, we present the four most commonly used fairness metrics and outline their respective applicable scenarios.

Demographic Parity Difference (DPD): In the context of face forgery detection, where label 1 represents fake face, DPD reflects the model's inclination to categorize faces of a specific race as fake. When people perceive the classification of faces as fake as a form of discrimination, we can leverage DPD to assess the extent of bias in the model.

$$
D P D = \max _ {s, s ^ {\prime} \in \mathbb {S}, s \neq s ^ {\prime}} \left| P (\hat {Y} \mid S = s) - P (\hat {Y} \mid S = s ^ {\prime}) \right|, \tag {6}
$$

where $\hat{Y}$ is the predicted labels. S represents the set of sensitive attributes, $S \in S$ and $S = \{Caucasian, Asian, Indian, African\}$ .

Difference in Equalized Odds (DEOdds): DPD may fail in certain situations $[57]$ . Considering a scenario where a face forgery detector classifies faces of African and Caucasian individuals as fake at a similar rate, but the model makes different types of errors for the two groups. Specifically, for African faces, the false positive rate is significantly higher than for Caucasian faces, while for Caucasian individuals, the false negative rate is higher. In such a case, we can use DEOdds.

$$
D E O d d s = \frac {1}{2} \sum_ {y = \{0, 1 \}} \max _ {s, s ^ {\prime} \in \mathbb {S}, s \neq s ^ {\prime}} \left| P (\hat {Y} | Y = y, S = s) - P (\hat {Y} | Y = y, S = s ^ {\prime}) \right|. \tag {7}
$$

Equal Opportunity (DEO): DEO has more relaxed conditions compared to DEOdds. When assessing the fairness between group A and group B, only the images of faces that are inherently fake need to be considered. DEO solely focuses on true positive rates and does not capture the overall classification differences.

$$
D E O = \max _ {s, s ^ {\prime} \in \mathbb {S}, s \neq s ^ {\prime}} \left| P (\hat {Y} | Y = 1, S = s) - P (\hat {Y} | Y = 1, S = s ^ {\prime}) \right|. \tag {8}
$$

Standard Deviation (STD): STD differs from metrics mentioned above. STD considers the overall variability rather than just the differences between the best and worst-performing ethnicities. We use accuracy as the performance metric.

$$
S T D = \operatorname{std} \left(\left\{\operatorname{acc} (S = s) _ {s \in \mathbb {S}} \right\}\right), \tag {9}
$$

where std is a function that calculates the standard deviation of given list. acc calculates the accuracy of a given race.

# Face Forgery Detection Datasets

Below we give simple description of some widely used face forgery datasets.

FaceForensics++(FF++) [1] is a forensics dataset that consists of 1000 original video sequences downloaded from the Internet(.,i.e., YouTube). The videos are manipulated with four automated face manipulation methods. For face swap, they use Deepfakes [58] and FaceSwap [21]. For expression reenactment, they use Face2Face [55] and NeuralTextures [59].

UADFV [32] contains videos of varying classes, with each video being classified as either real or fake. The dataset is relatively small, with only 98 videos, but it has been found to be convenient in terms of how the data is formatted.

CelebDF-v2 [33] is a comprehensive collection for deepfake forensics, comprising 590 real videos and 5,639 DeepFake videos featuring celebrities with high-quality. These videos were generated through an enhanced synthesis process, ensuring superior visual quality and better representing the DeepFake content prevalent on the internet.

Deepfake Detection Challenge(DFDC) [34] is created by Facebook in partnership with other industry leaders and academic experts. The dataset consists of 128,154 videos featuring 960 paid actors. The dataset was used in a Kaggle competition to create new and better models to detect manipulated media.

DeeperForensics-1.0(DF-1.0) [35] contains 60,000 videos and 17.6 million frames with 100 consented actors. The actors in DF-1.0 have four skin tones: white, black, yellow, brown, with roughly balanced ratio. Different from other datasets, DF-1.0 attach importance to high-quality and diversity of source face videos. Each actor has various poses, expressions, and illuminations. And DF-1.0 has more real videos than fake videos with a ratio of 5:1.

ForgeryNet [36] is a very large dataset for real-world face forgery detection, with 2.9 million images, 221,247 videos and 15 forgery approaches. And a pipeline of conducting various face forgery approaches are proposed.

# Face Forgery Detection Algorithms Categories

Current face forgery detectors can be roughly divided into two categories: Spatial-based, and Frequency-based.

Spatial-based method [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] is based on the spatial domain features(i.e., forgery clues) of the fake face. Such as RECCE [10] utilizes an encoder to reconstruct real face images, thereby exploring the differences between real and fake images in the spatial domain. The differences in the reconstruction of the two are then used as guidance to train a classifier for detecting deep-fakes. Frequency-based method $[12, 13, 14]$ identifies distinctions between real and fake faces in frequency domain, which are used to detect whether the face is forged. Such as F3Net $[12]$ utilizes frequency-aware decomposed image components and local frequency statistics to explore forgery patterns, enabling effective forgery detection.

# Fairness in Face Recognition

An increasing number of researchers are now paying attention to societal issues of artificial intelligence, including adversarial example $[60, 61, 62]$ , privacy protection $[63, 64, 65]$ , and fairness concerns.

The research on fairness in face forgery detection is similar to the study of fairness in face recognition. RFW $[39]$ is first introduced as a race balanced test dataset for face recognition. They also propose IMAN which uses a deep information maximization adaptation network to align global distribution to decrease race gap at domain-level, and learns the discriminative target representations at cluster level. Another balanced face recognition dataset BFW $[40]$ is proposed to evaluate the fairness of face recognition system both for gender and ethnic groups and this work also shows variations in the optimal scoring threshold for face-pairs across different subgroups.

The issue of racial bias in machine learning has garnered significant attention in multi fields. The research on fairness in face forgery detection is akin to the study of fairness in face recognition. RFW $[39]$ is first introduced as a balanced test dataset for ethnic group. IMAN $[39]$ uses a deep information maximization adaptation network to align global distribution to decrease race gap at domain-level, and learns the discriminative target representations at cluster level. Another balanced faces dataset BFW $[40]$ is proposed to evaluate the fairness of face recognition system both for gender and ethnic groups and this work also shows variations in the optimal scoring threshold for face-pairs across different subgroups. DebFace $[41]$ uses a de-biasing adversarial network to extract disentangled feature representations for both unbiased face recognition and demographics estimation and adopts adversarial learning to minimize correlation among feature factors so as to abate bias influence. $[42]$ involves multiple preprocessing methods to improve the dual-shot face detector, data re-sampling to balance the data distribution, and multiple data enhancement methods to increase accuracy performance and proposes a linear-combination strategy is adopted to benefit from multi-model fusion. Meta Balanced Network $[66]$ uses meta-learning algorithm to learn adaptive margins in large margin loss to mitigate the algorithmic bias in face recognition models. MixFairFace $[43]$ proposes MixFair Adapter to determine and reduce the identity bias of training samples.

# Model Pruning

Current pruning methods typically use two types of signals: the magnitude [67, 68, 69] of the neurons and the absolute value of the neuron activations [70]. Generally, the smaller these values, the less contribution they make to the model.

<table><tr><td></td><td>Caucasian</td><td>Asian</td><td>African</td><td>Indian</td></tr><tr><td>TPR</td><td>0.7764</td><td>0.7365</td><td>0.6289</td><td>0.6801</td></tr><tr><td>TNR</td><td>0.9502</td><td>0.9652</td><td>0.9314</td><td>0.9646</td></tr></table>

Table 4: Evaluation with TPR and TNR for each race on FF++ subset.

In our method, we combine the magnitude of the neurons and the bias of activations across different races as the pruning score. The resulting pruning score helps identify neurons that have a small contribution to model performance but a large contribution to racial bias.

# Details of Crafting Dataset

# Number of Subjects is Limited

We propose that the limited number of subjects makes it challenging to draw meaningful comparisons across groups. Here we do a verification experiment to explain and prove it. We utilize the FF++ dataset, retains only the Caucasian, Asian, African, and Indian subsets. The remaining dataset consists of a total of 677 subjects. For each race, 16 subjects are chosen as the test dataset, and the rest serve as the training dataset (9:1 approximately). We train an Xception model on the training dataset for deepfake detection and evaluate its performance on the test dataset. We follow the hyperparameters specified in [45] and train the model for 10 epochs.

In Table 4, we present the TPR and TNR of different ethnicities. And we can find that model performs worse on African for both TPR and TNR. It seems that this result indicates discrimination of the detectors towards African subset, but we argue that this discrimination is not credible. In Figure 5, we show the TPR and TNR on each subject. We can observe significant fluctuations in the model's performance across each subject. The presence of extreme results also indicates that subjects significantly influence the model's classification. Therefore, when the number of subjects is insufficient, it becomes challenging to capture the overall characteristics of a group, making it difficult to draw meaningful comparisons across race groups.

# Forgery Crafting Process

1) For face swap, we select FaceSwap [21] and SimSwap [22]. Although both methods result in a face-swapping effect, to ensure approach diversity, we select two distinct face-swapping approaches. FaceSwap [21] is a graphic-based face swap method. Whereas SimSwap [22] is a learning-based face swap method. 2) For expression reenactment, we select FastReen [23] and DualReen [24]. Both face swap and expression reenactment methods involve the use of source and target images to transfer faces or expressions from the target image to the source image. For each subject, we first designate it as the source subject. Next, within the same ethnic group, we randomly select another subject as the target subject, ensuring that the transfer occurs only within a single ethnic group as stated in section "Limitations of Current Datasets" in Supp. Once the source-target pairs are established, these pairs remain fixed in other identity-replaced forgery approaches. Considering that each subject may have multiple images, for each specific image, we randomly select one image from its appointed target subject for the transfer, ensuring diversity and variability in the dataset. 3) For attribute manipulation, we select MaskGAN [25]. MaskGAN [25] provides an official GUI program that allows manual manipulation of attributes by making use of face parsing. However, as we aim to automate the face forgery process, we randomly select a subset from the nose, glasses, left eye, right eye, left eyebrow, right eyebrow, left ear, right ear, mouth, upper lip, and lower lip. We then apply random dilate and erode operations to the selected parts, enlarging or reducing the chosen regions. Missing parts are filled with skin, resulting in the final manipulated outcome. 4) For much recent forgery methods, we select Diffusion-Based(SDSwap [28], DCFace [29], Face2Diffusion [30]) and Transformer-Based(FSRT [31]). These approaches use advanced technologies: Stable Diffusion and Transformer, resulting in more realistic generated faces. In summary, we select three types, a total of 11 forgery approaches, which achieve the diversity requirement.

# Detailed Distribution of FairFD

For the number of race categories, existing work almost all uses three or four races, and we follow previous works [43, 38, 37] with standard practice for convenience.

Moreover, to ensure an accurate evaluation, we keep distributions of other attributes (including gender, age, etc) similar across different racial groups. Since our goal is to simulate real-world racial fairness evaluation, in each race group, distributions of these attributes are simulated based on real-world distributions. These attributes (e.g., gender, age, etc.) do not introduce additional biases, as we derive our real face images from the existing, widely recognized racial bias evaluation dataset RFW [39]. It is worth noting that these attributes are not perfectly balanced within each racial group, as we focus on 'in-the-wild' faces, aiming to simulate real-world distributions for a more meaningful practical evaluation. Refer to Figure 6 for detailed distributions. We ensure similar distributions of these attributes across racial groups, ensuring that our dataset does not introduce additional bias.

# Analyses and Discussions

Balanced Training Dataset. Data imbalance across races is a prevalent source of bias. To assess the impact of imbalanced training sets on racial bias, we create a dataset that balances across different racial groups. The dataset is obtained by sampling from the FF++. Because FF++ contains a very few number of African and Indian subjects, we can only select 26 subjects from each race. This balanced dataset is employed as the training set. We adopt the same data preprocessing methods and training parameters as the racially imbalanced training set. We use real face, FaceSwap, SimSwap, FastReen, DualReen and MaskGAN of FairFD as our dataset. Notably, due to the reduced dataset size, the number of epochs is doubled to 20.

Table 5 presents the results. For convenience, we put the results of models trained on unbalanced and balanced

![](images/bf72755a509f22f92588aeb3ee8b09f42ad7b8f2ffb46db4f7fbe0c301a2ee9f.jpg)  
Figure 5: Evaluation with TPR and TNR for for each subject on FF++ subset.

![](images/cf2cec6fb163e150cc6d8c1f8d351567f2aa8e83874618c67c8bab008f58b753.jpg)  
(a) yaw

![](images/398816a913d4315785e80cb3aaaea2b71939cc2ba795654d1cde648f01ba1c2e.jpg)  
(b) pitch

![](images/75b57f4b569500283ae8cd3fa71eb3cc5f6a31ce31e2fd4af5ab14c564366671.jpg)  
(c) age

![](images/e07032e2f3ed527e385c0ee6e44e3edfeb2157aa290d584adf0d32e5bff814bc.jpg)  
(d) gender

![](images/0d6614d3532b5689502b6697dad1661bdf2c77c1b976329d35daac109fdec015.jpg)  
(e) pose gap

![](images/a41073deed37bd6e615d317dd48e3d4a1833ff227a089cbb782517558b0f8846.jpg)  
(f) age gap

Figure 6: The distribution of other attributes (aside from racial attributes) in FairFD. This figure is adapted from RFW [39]. 

<table><tr><td rowspan="2" colspan="2">Metrics</td><td colspan="2">Xception</td><td colspan="2">F3Net</td><td colspan="2">RECCE</td><td colspan="2">UCF</td></tr><tr><td>Unbalanced</td><td>Balanced</td><td>Unbalanced</td><td>Balanced</td><td>Unbalanced</td><td>Balanced</td><td>Unbalanced</td><td>Balanced</td></tr><tr><td rowspan="4">Naive Metric</td><td>DPD</td><td>0.1538</td><td>0.0852</td><td>0.0764</td><td>0.0817</td><td>0.1317</td><td>0.0836</td><td>0.1782</td><td>0.0908</td></tr><tr><td>DEOdds</td><td>0.1672</td><td>0.0837</td><td>0.0877</td><td>0.0630</td><td>0.1379</td><td>0.0663</td><td>0.1655</td><td>0.0674</td></tr><tr><td>DEO</td><td>0.2095</td><td>0.1076</td><td>0.1030</td><td>0.1027</td><td>0.1777</td><td>0.1027</td><td>0.2334</td><td>0.1104</td></tr><tr><td>STD</td><td>0.0563</td><td>0.0334</td><td>0.0278</td><td>0.0329</td><td>0.0473</td><td>0.0314</td><td>0.0635</td><td>0.0355</td></tr><tr><td rowspan="4">Approach Averaged Metric</td><td>AADPD</td><td>0.1957</td><td>0.1030</td><td>0.1102</td><td>0.0996</td><td>0.1644</td><td>0.1044</td><td>0.2107</td><td>0.0961</td></tr><tr><td>AADEOdds</td><td>0.1674</td><td>0.0857</td><td>0.0951</td><td>0.0691</td><td>0.1378</td><td>0.0746</td><td>0.1655</td><td>0.0674</td></tr><tr><td>AADEO</td><td>0.2099</td><td>0.1116</td><td>0.1178</td><td>0.1149</td><td>0.1777</td><td>0.1193</td><td>0.2334</td><td>0.1104</td></tr><tr><td>AASTD</td><td>0.0721</td><td>0.0412</td><td>0.0425</td><td>0.0393</td><td>0.0603</td><td>0.0405</td><td>0.0773</td><td>0.0374</td></tr><tr><td rowspan="4">Utility Regularized Metric</td><td>URDPD</td><td>0.1314</td><td>0.0741</td><td>0.0769</td><td>0.0770</td><td>0.1158</td><td>0.0733</td><td>0.1433</td><td>0.0723</td></tr><tr><td>URDEOdds</td><td>0.1069</td><td>0.0573</td><td>0.0624</td><td>0.0511</td><td>0.0908</td><td>0.0505</td><td>0.1069</td><td>0.0487</td></tr><tr><td>URDEO</td><td>0.1437</td><td>0.0824</td><td>0.0842</td><td>0.0900</td><td>0.1283</td><td>0.0847</td><td>0.1614</td><td>0.0842</td></tr><tr><td>URSTD</td><td>0.0482</td><td>0.0297</td><td>0.0295</td><td>0.0302</td><td>0.0423</td><td>0.0284</td><td>0.0523</td><td>0.0281</td></tr><tr><td>Utility</td><td>Accuracy</td><td>0.5552</td><td>0.3984</td><td>0.5266</td><td>0.3308</td><td>0.5011</td><td>0.4276</td><td>0.5557</td><td>0.3682</td></tr></table>

Table 5: Evaluations on the unbalanced and balanced training dataset.

datasets together. For both Naive Metric and Approach Averaged Metric, we observe that models trained on balanced datasets generally exhibit lower racial bias. Despite the sigsignificant improvement brought by a balanced dataset, these detectors still demonstrate a relatively high racial bias. It is crucial to note that, particularly for F3Net, not all metrics

![](images/ac7d9f4e403739fae4922231cfad64a7336b86793dca306dfc790f11cf46210d.jpg)

Figure 7: Probability score distribution of each race. 

<table><tr><td rowspan="2">Xception</td><td colspan="4">Utility(Accuracy) ↑</td><td colspan="2">Fairness ↓</td></tr><tr><td>Caucasian</td><td>Asian</td><td>African</td><td>Indian</td><td>STD</td><td>AccGap</td></tr><tr><td>Best Threshold</td><td>0.6530</td><td>0.7300</td><td>0.5380</td><td>0.6660</td><td>-</td><td>-</td></tr><tr><td>Real Face</td><td>0.7960/0.8964</td><td>0.7201/0.8813</td><td>0.8449/0.8767</td><td>0.7715/0.8888</td><td>0.0450/0.0075</td><td>0.1248/0.0197</td></tr><tr><td>FaceSwap</td><td>0.8910/0.8265</td><td>0.8949/0.7883</td><td>0.7668/0.7373</td><td>0.8968/0.8264</td><td>0.0552/0.0366</td><td>0.1300/0.0892</td></tr></table>

Table 6: Performance and fairness with 0.5 and optimal thresholds as threshold respectively. Left of '/' 0.5. Right of '/' is optimal threshold.

demonstrate improvement. These results indicate that training models on a balanced dataset cannot completely address the issue of racial bias, because there are other factors contributing to racial bias. For the Utility Regularized Metric, we note that the enhancement achieved through a balanced training dataset is less significant or may even diminish compared to the Naive Metric and Approach Averaged Metric. This is because this set of metrics is influenced by performance, and the reduced data volume results in a notable model performance drop. This highlights the advantage of the Utility Regularized Metric, i.e., it can reflect the model's performance.

Based on the aforementioned observations, we argue that utilizing a race-balanced training dataset may not be a recommended way. The fact that deliberately collecting such balanced data in real-world scenarios usually implies discarding a wealth of available imbalanced datasets. Existing datasets are also seldom racially balanced. Furthermore, employing a racially balanced training set does not guarantee effective mitigation of racial bias. In conclusion, training with race-balanced datasets poses challenges, considering the scarcity of such datasets in real-world scenarios and the limited effectiveness in addressing racial bias. Therefore, it is preferable to utilize some other fair learning methods that exhibit better trade-offs.

Setting Different Threshold for Each Race. The study in [40] suggests that in face recognition, the confidence score distributions vary among different race groups. So, they propose setting different thresholds for different races can enhance fairness as well as improve overall performance. We utilize only the real face and FaceSwap portions of FairFD. Firstly, we plot the probability score distribution for each race in Figure 7, revealing distinct differences in the confidence score distribution across different races. The Asian subset shows a higher frequency of high confidence scores, making it more prone to be classified as a fake face. Therefore, a larger threshold can be set for the Asian subset. On the other hand, the African subset exhibits a lower frequency of high probability scores, making it less likely to be classified as a fake face, allowing for a smaller threshold. Next, we set the optimal thresholds, which are values that maximize the overall performance for each racial subset. Table 6 presents the optimal threshold values, as well as the performance and fairness scores when using threshold 0.5 and the optimal thresholds. We observe that, under the condition of maximizing overall performance, there is a certain degree of reduction in racial bias. We use test data as "Balanced Training Dataset". Table 7 demonstrates racial bias using a fairness metric, showing that indeed, setting different thresholds can reduce racial bias. This conclusion aligns with the findings in [40] consistently.

Analysis of Frequency-Based Detector. In our previous findings, we observe that frequency-based methods exhibit lower racial bias compared to spatial-based methods. In this section, we provide a preliminary discussion using a subset of FairFD as "Balanced Training Dataset". Spatial-based methods focus on learning forgery clues in the spatial domain, mainly including color mismatch, textures, shapes, and blending boundaries [71]. On the other hand, frequency-based methods concentrate on learning forgery clues in the frequency domain, especially targeting high-frequency information related to blending boundaries, edges, and textures [12]. Frequency-based methods capture less color information, which is also crucial in distinguishing between different racial groups. Consequently, we make the assumption that frequency-based methods' racial biases are smaller due to these detectors learning less color information. Therefore, we convert the RGB images of our FairFD dataset into grayscale to eliminate color information but retain frequency domain information, and test multiple detectors. The results in Table 8 demonstrate that the racial bias of spatial-based detectors decreases, while the racial bias of frequency-based

<table><tr><td>Threshold</td><td>AADPD</td><td>AADEOdds</td><td>AADEO</td><td>AASTD</td><td>URDPD</td><td>URDEOdds</td><td>URDEO</td><td>URSTD</td></tr><tr><td>0.5</td><td>0.1274</td><td>0.1274</td><td>0.1300</td><td>0.0501</td><td>0.1551</td><td>0.1551</td><td>0.1507</td><td>0.0607</td></tr><tr><td>BEST [40]</td><td>0.0544</td><td>0.0544</td><td>0.0892</td><td>0.0220</td><td>0.0301</td><td>0.0301</td><td>0.0497</td><td>0.0122</td></tr></table>

Table 7: Fairness metric results at different thresholds.

<table><tr><td rowspan="2" colspan="2">Fairness Metric</td><td colspan="2">Xception</td><td colspan="2">F3Net</td><td colspan="2">RECCE</td><td colspan="2">UCF</td></tr><tr><td>RGB</td><td>Grayscale</td><td>RGB</td><td>Grayscale</td><td>RGB</td><td>Grayscale</td><td>RGB</td><td>Grayscale</td></tr><tr><td rowspan="4">Naive Metric</td><td>DPD</td><td>0.1538</td><td>0.1309</td><td>0.0764</td><td>0.0780</td><td>0.1317</td><td>0.0811</td><td>0.1782</td><td>0.1324</td></tr><tr><td>DEOdds</td><td>0.1672</td><td>0.1439</td><td>0.0877</td><td>0.0976</td><td>0.1379</td><td>0.0840</td><td>0.1655</td><td>0.1512</td></tr><tr><td>DEO</td><td>0.2095</td><td>0.1789</td><td>0.1030</td><td>0.1035</td><td>0.1777</td><td>0.1076</td><td>0.2334</td><td>0.1828</td></tr><tr><td>STD</td><td>0.0563</td><td>0.0470</td><td>0.0278</td><td>0.0283</td><td>0.0473</td><td>0.0295</td><td>0.0635</td><td>0.0475</td></tr><tr><td rowspan="4">Approach Averaged Metric</td><td>AADPD</td><td>0.1957</td><td>0.1673</td><td>0.1102</td><td>0.1147</td><td>0.1644</td><td>0.1097</td><td>0.2107</td><td>0.1722</td></tr><tr><td>AADEOdds</td><td>0.1674</td><td>0.1439</td><td>0.0951</td><td>0.1055</td><td>0.1378</td><td>0.0900</td><td>0.1655</td><td>0.1512</td></tr><tr><td>AADEO</td><td>0.2099</td><td>0.1789</td><td>0.1178</td><td>0.1193</td><td>0.1777</td><td>0.1195</td><td>0.2334</td><td>0.1828</td></tr><tr><td>AASTD</td><td>0.0721</td><td>0.0614</td><td>0.0425</td><td>0.0441</td><td>0.0603</td><td>0.0410</td><td>0.0773</td><td>0.0658</td></tr><tr><td rowspan="4">Utility Regularized Metric</td><td>URDPD</td><td>0.1314</td><td>0.1144</td><td>0.0769</td><td>0.0802</td><td>0.1158</td><td>0.0785</td><td>0.1433</td><td>0.1209</td></tr><tr><td>URDEOdds</td><td>0.1069</td><td>0.0935</td><td>0.0624</td><td>0.0687</td><td>0.0908</td><td>0.0603</td><td>0.1069</td><td>0.0986</td></tr><tr><td>URDEO</td><td>0.1437</td><td>0.1249</td><td>0.0842</td><td>0.0859</td><td>0.1283</td><td>0.0876</td><td>0.1614</td><td>0.1321</td></tr><tr><td>URSTD</td><td>0.0482</td><td>0.0419</td><td>0.0295</td><td>0.0309</td><td>0.0423</td><td>0.0294</td><td>0.0523</td><td>0.0461</td></tr></table>

Table 8: Comparison of fairness on RGB and Grayscale images.

methods does not decrease and the changes are small. Thus, color indeed appears to be a contributing factor that leads spatial-based methods to have higher racial bias compared to frequency-based methods. There is still much exploration to be done in the comparative study of frequency and spatial domains, which can significantly contribute to the development of methods for mitigating racial bias. We leave this avenue of research for future investigation.

# Visualization in Feature Space

The research on fairness in face forgery detection is similar to the study of fairness in face recognition(see section "Fairness in Face Recognition" in Supp for more details). As stated in [39], one reason for racial bias is that different subsets' features are totally separate. So they take the racial bias as a problem of domain gap and propose IMAN(information maximization adaptation network) to decrease this domain gap. We sample 500 samples for each ethnic group and use the well trained Xception model as the detector to obtain features for these samples. Then, we plot the t-SNE dimensionality reduction graphs for the feature spaces of the four ethnic subsets in Figure 8. Considering that different forgery approaches will cause distinct results, we plot for Real Face, FaceSwap and FaceReen respectively. Unlike in [39], we do not observe distinct separation between different subsets at feature level. Next, we use the MMD(Maximum Mean Discrepancy) to mathematically calculate the feature distances between the Caucasian subset and other ethnic groups. The results in Figure 9 demonstrate that although distances show difference, in comparison to the distances in [39], the distances here are all nearly close to zero. These results are because they consider the face recognition task in [39], so the models tend to learn distinctive features for each sub-

<table><tr><td rowspan="2">Fairness Metric</td><td colspan="3">Fairness-enhanced</td></tr><tr><td>DAG</td><td>DAW</td><td>PFGDFD</td></tr><tr><td>DPD↓</td><td>0.0316</td><td>0.0694</td><td>0.0709</td></tr><tr><td>DEOdds↓</td><td>0.0564</td><td>0.1057</td><td>0.1102</td></tr><tr><td>DEO↓</td><td>0.0402</td><td>0.0870</td><td>0.0893</td></tr><tr><td>STD↓</td><td>0.0122</td><td>0.0253</td><td>0.0280</td></tr><tr><td>AADPD↓</td><td>0.0569</td><td>0.0989</td><td>0.1127</td></tr><tr><td>AADEOdds↓</td><td>0.0641</td><td>0.1105</td><td>0.1210</td></tr><tr><td>AADEO↓</td><td>0.0555</td><td>0.0966</td><td>0.1110</td></tr><tr><td>AASTD↓</td><td>0.0225</td><td>0.0378</td><td>0.0436</td></tr><tr><td>URDPD↓</td><td>0.0322</td><td>0.0574</td><td>0.0688</td></tr><tr><td>URDEOdds↓</td><td>0.0447</td><td>0.0767</td><td>0.0763</td></tr><tr><td>URDEO↓</td><td>0.0297</td><td>0.0536</td><td>0.0673</td></tr><tr><td>URSTD↓</td><td>0.0127</td><td>0.0219</td><td>0.0266</td></tr><tr><td>AUC↑</td><td>0.6638</td><td>0.6121</td><td>0.6336</td></tr></table>

Table 9: Bias evaluation for three detectors trained without data augmentation.

ject, resulting in significant differences at the feature level. In contrast, for the task of face forgery detection, the detector does not exhibit the same tendency, leading to smaller differences in feature level. Therefore, attempting to enhance the fairness of face forgery detection from a feature-level perspective may not be feasible.

# Results without Data Augmentation

Results without data augmentation are shown in Table 9.

![](images/f004f9013ba3d8e807da3a1ed0a0e6aeacfc96a906f540b56f5e2d05a9164c53.jpg)

<details>
<summary>scatter</summary>

| Group     | Color  |
|-----------|--------|
| Caucasian | Green  |
| Asian     | Blue   |
| African   | Purple |
| Indian    | Red    |
</details>

![](images/92f5b6db937b765ff93657ca955c126812baa46bbdf0d339758d50510a78211f.jpg)

<details>
<summary>scatter</summary>

| Ethnicity | Count |
| --------- | ----- |
| Caucasian | 120   |
| Asian     | 115   |
| African   | 110   |
| Indian    | 105   |
</details>

![](images/660ab47f2728a119b7f7b6b4dce04677655c410cc3bc59e69186f997e9c7caa7.jpg)

<details>
<summary>scatter</summary>

| Ethnicity | Count |
| --------- | ----- |
| Caucasian | 120   |
| Asian     | 150   |
| African   | 130   |
| Indian    | 140   |
</details>

Figure 8: T-SNE visualization on Xception model.   
![](images/75fa71d5af949895ea491cf9f0896da7a98a35444a141e2621e4444314618bea.jpg)

<details>
<summary>bar</summary>

| Real Face | MMD     |
| --------- | ------- |
| Ca-Ca     | 0.0005  |
| Ca-As     | 0.004   |
| Ca-Af     | 0.010   |
| Ca-In     | 0.0025  |
</details>

![](images/ed36cdbd90636ff32d3ffdaefb7d010c6d8c1f1d0d779e40aa0d13d150857a51.jpg)

<details>
<summary>bar</summary>

| FaceSwap | MMD     |
| -------- | ------- |
| Ca-Ca    | 0.0002  |
| Ca-As    | 0.0065  |
| Ca-Af    | 0.0150  |
| Ca-In    | 0.0018  |
</details>

![](images/a114ba15e8e7b1d9e34b58df753a59719c26828c9db558d2d7b7a58e26d71c51.jpg)

<details>
<summary>bar</summary>

| Category | MMD |
|---|---|
| Ca-Ca | 0.0007 |
| Ca-As | 0.0083 |
| Ca-Af | 0.0084 |
| Ca-In | 0.0018 |
</details>

Figure 9: Maximum Mean Discrepancy (MMD) on Xception model.

# Baseline Algorithm

WEIG uses only the absolute values of the weights as the pruning score, i.e., the pruning score of WEIG is:

$$
P S _ {i j k m} ^ {W E I G} = \left| W _ {i j k m} \right|, \tag {10}
$$

RoBA uses only the reciprocal of the bias of the activations, i.e., the pruning score of RoBA is:

$$
P S _ {i j k m} ^ {R o B A} = \frac {1}{B I A S _ {i}}. \tag {11}
$$

# Ablation Study on Pruning Rate

We set the pruning rates to 0.1%, 0.4%, 0.1%, 1%, 4%, 7%, and 10%. At a pruning rate of 10%. A pruning rate of 10% is found to severely degrade utility in our experiments, so we do not conduct tests with higher pruning rates. The results are shown in the Table 10, Table 11 and Table 12. The '-' symbol indicates that the model could completely unable to identify forged images. From the results, we can observe that our proposed method (BPFA) and WEIG demonstrate excellent robustness to varying pruning rates, with BPFA achieving a higher level of fairness compared to WEIG. RoBA exhibits highly unstable performance, being significantly affected by the pruning rate. Across all the pruning rates we tested, RoBA fails to achieve both good fairness and utility simultaneously. Moreover, RoBA requires extremely low pruning rates to maintain its classification capability; otherwise, it completely loses its ability to classify.

# Benchmark Results (Rankings)

In Table 13 and Table 14, we present the ranking results of the benchmark, which allows for a clearer comparison of the rankings of the methods under different metrics.

<table><tr><td rowspan="3">Pruning Rate</td><td rowspan="3">Method</td><td colspan="4">Naive Metric↓</td><td colspan="4">Approach Averaged Metric↓</td><td colspan="4">Utility Regularized Metric↓</td><td colspan="2">Utility↑</td></tr><tr><td rowspan="2">DPD</td><td rowspan="2">DEOdds</td><td rowspan="2">DEO</td><td rowspan="2">STD</td><td>MA</td><td>MA</td><td>MA</td><td>MA</td><td>UR</td><td>UR</td><td>UR</td><td>UR</td><td rowspan="2">AUC</td><td rowspan="2">ACC</td></tr><tr><td>DPD</td><td>DEOdds</td><td>DEO</td><td>STD</td><td>DPD</td><td>DEOdds</td><td>DEO</td><td>STD</td></tr><tr><td>0</td><td>Original</td><td>0.0203</td><td>0.0304</td><td>0.0215</td><td>0.0080</td><td>0.0556</td><td>0.0481</td><td>0.0571</td><td>0.0219</td><td>0.0320</td><td>0.0299</td><td>0.0324</td><td>0.0126</td><td>0.6763</td><td>0.7618</td></tr><tr><td rowspan="3">0.1%</td><td>WEIG</td><td>0.0183</td><td>0.0258</td><td>0.0201</td><td>0.0072</td><td>0.0564</td><td>0.0451</td><td>0.0586</td><td>0.0219</td><td>0.0324</td><td>0.0277</td><td>0.0334</td><td>0.0126</td><td>0.6769</td><td>0.7615</td></tr><tr><td>RoBA</td><td>0.0188</td><td>0.0258</td><td>0.0214</td><td>0.0069</td><td>0.0602</td><td>0.0466</td><td>0.0629</td><td>0.0233</td><td>0.0458</td><td>0.0322</td><td>0.0486</td><td>0.0177</td><td>0.6608</td><td>0.3468</td></tr><tr><td>BPFA</td><td>0.0183</td><td>0.0258</td><td>0.0201</td><td>0.0072</td><td>0.0564</td><td>0.0451</td><td>0.0586</td><td>0.0219</td><td>0.0324</td><td>0.0277</td><td>0.0334</td><td>0.0126</td><td>0.6769</td><td>0.7615</td></tr><tr><td rowspan="3">0.4%</td><td>WEIG</td><td>0.0184</td><td>0.0259</td><td>0.0201</td><td>0.0072</td><td>0.0563</td><td>0.0451</td><td>0.0586</td><td>0.0219</td><td>0.0324</td><td>0.0277</td><td>0.0334</td><td>0.0126</td><td>0.6769</td><td>0.7615</td></tr><tr><td>RoBA</td><td>0.1128</td><td>0.1598</td><td>0.1395</td><td>0.0445</td><td>0.1462</td><td>0.1616</td><td>0.1432</td><td>0.0583</td><td>0.0893</td><td>0.1024</td><td>0.0867</td><td>0.0356</td><td>0.6331</td><td>0.7037</td></tr><tr><td>BPFA</td><td>0.0184</td><td>0.0259</td><td>0.0201</td><td>0.0072</td><td>0.0563</td><td>0.0451</td><td>0.0586</td><td>0.0219</td><td>0.0324</td><td>0.0277</td><td>0.0334</td><td>0.0126</td><td>0.6770</td><td>0.7613</td></tr><tr><td rowspan="3">0.7%</td><td>WEIG</td><td>0.0181</td><td>0.0259</td><td>0.0199</td><td>0.0071</td><td>0.0563</td><td>0.0452</td><td>0.0585</td><td>0.0219</td><td>0.0324</td><td>0.0278</td><td>0.0333</td><td>0.0126</td><td>0.6769</td><td>0.7613</td></tr><tr><td>RoBA</td><td>0.0210</td><td>0.0166</td><td>0.0234</td><td>0.0077</td><td>0.0268</td><td>0.0191</td><td>0.0283</td><td>0.0098</td><td>0.0224</td><td>0.0145</td><td>0.0240</td><td>0.0082</td><td>0.6853</td><td>0.1760</td></tr><tr><td>BPFA</td><td>0.0181</td><td>0.0259</td><td>0.0199</td><td>0.0071</td><td>0.0563</td><td>0.0452</td><td>0.0585</td><td>0.0219</td><td>0.0324</td><td>0.0278</td><td>0.0333</td><td>0.0126</td><td>0.6769</td><td>0.7615</td></tr><tr><td rowspan="3">1%</td><td>WEIG</td><td>0.0189</td><td>0.0296</td><td>0.0201</td><td>0.0075</td><td>0.0562</td><td>0.0484</td><td>0.0577</td><td>0.0220</td><td>0.0323</td><td>0.0300</td><td>0.0328</td><td>0.0126</td><td>0.6760</td><td>0.7603</td></tr><tr><td>RoBA</td><td>0.0605</td><td>0.0496</td><td>0.0686</td><td>0.0239</td><td>0.0777</td><td>0.0563</td><td>0.0819</td><td>0.0300</td><td>0.0616</td><td>0.0408</td><td>0.0658</td><td>0.0238</td><td>0.6913</td><td>0.3124</td></tr><tr><td>BPFA</td><td>0.0195</td><td>0.0295</td><td>0.0208</td><td>0.0079</td><td>0.0556</td><td>0.0477</td><td>0.0571</td><td>0.0218</td><td>0.0320</td><td>0.0296</td><td>0.0325</td><td>0.0125</td><td>0.6766</td><td>0.7611</td></tr><tr><td rowspan="3">4%</td><td>WEIG</td><td>0.0190</td><td>0.0277</td><td>0.0211</td><td>0.0074</td><td>0.0569</td><td>0.0467</td><td>0.0590</td><td>0.0221</td><td>0.0328</td><td>0.0288</td><td>0.0336</td><td>0.0128</td><td>0.6766</td><td>0.7596</td></tr><tr><td>RoBA</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>BPFA</td><td>0.0164</td><td>0.0242</td><td>0.0179</td><td>0.0063</td><td>0.0523</td><td>0.0424</td><td>0.0543</td><td>0.0203</td><td>0.0298</td><td>0.0262</td><td>0.0305</td><td>0.0116</td><td>0.6788</td><td>0.7813</td></tr><tr><td rowspan="3">7%</td><td>WEIG</td><td>0.0196</td><td>0.0274</td><td>0.0217</td><td>0.0076</td><td>0.0564</td><td>0.0459</td><td>0.0586</td><td>0.0220</td><td>0.0326</td><td>0.0282</td><td>0.0334</td><td>0.0127</td><td>0.6777</td><td>0.7587</td></tr><tr><td>RoBA</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>BPFA</td><td>0.0181</td><td>0.0209</td><td>0.0200</td><td>0.0072</td><td>0.0473</td><td>0.0357</td><td>0.0496</td><td>0.0182</td><td>0.0265</td><td>0.0218</td><td>0.0275</td><td>0.0102</td><td>0.6862</td><td>0.8055</td></tr><tr><td rowspan="3">10%</td><td>WEIG</td><td>0.0195</td><td>0.0268</td><td>0.0215</td><td>0.0076</td><td>0.0560</td><td>0.0451</td><td>0.0582</td><td>0.0219</td><td>0.0323</td><td>0.0277</td><td>0.0332</td><td>0.0126</td><td>0.6777</td><td>0.7599</td></tr><tr><td>RoBA</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>BPFA</td><td>0.0239</td><td>0.0178</td><td>0.0259</td><td>0.0096</td><td>0.0487</td><td>0.0310</td><td>0.0523</td><td>0.0196</td><td>0.0274</td><td>0.0181</td><td>0.0293</td><td>0.0110</td><td>0.6938</td><td>0.7900</td></tr><tr><td rowspan="3">Pruning Rate</td><td rowspan="3">Method</td><td colspan="4">Naive Metric</td><td colspan="4">Approach Averaged Metric</td><td colspan="4">Utility Regularized Metric</td><td colspan="2">Utility</td></tr><tr><td rowspan="2">DPD</td><td rowspan="2">DEOdds</td><td rowspan="2">DEO</td><td rowspan="2">STD</td><td>MA</td><td>MA</td><td>MA</td><td>MA</td><td>UR</td><td>UR</td><td>UR</td><td>UR</td><td rowspan="2">AUC</td><td rowspan="2">ACC</td></tr><tr><td>DPD</td><td>DEOdds</td><td>DEO</td><td>STD</td><td>DPD</td><td>DEOdds</td><td>DEO</td><td>STD</td></tr><tr><td>0</td><td>Original</td><td>0.1099</td><td>0.1005</td><td colspan="2">0.12420.0398</td><td>0.1552</td><td>0.1196</td><td colspan="2">0.16230.0576</td><td>0.1037</td><td>0.0763</td><td colspan="2">0.10920.0384</td><td colspan="2">0.73040.5751</td></tr><tr><td rowspan="3">0.1%</td><td>WEIG</td><td>0.1099</td><td>0.1006</td><td colspan="2">0.12410.0398</td><td>0.1552</td><td>0.1196</td><td colspan="2">0.16230.0576</td><td>0.1037</td><td>0.0763</td><td colspan="2">0.10910.0384</td><td colspan="2">0.73040.5750</td></tr><tr><td>RoBA</td><td>0.0369</td><td>0.0343</td><td colspan="2">0.04150.0151</td><td>0.0676</td><td>0.0492</td><td colspan="2">0.07130.0265</td><td>0.0566</td><td>0.0371</td><td colspan="2">0.06040.0221</td><td colspan="2">0.59670.2235</td></tr><tr><td>BPFA</td><td>0.1099</td><td>0.1006</td><td colspan="2">0.12410.0398</td><td>0.1552</td><td>0.1197</td><td colspan="2">0.16230.0576</td><td>0.1037</td><td>0.0763</td><td colspan="2">0.10920.0384</td><td colspan="2">0.73050.5751</td></tr><tr><td rowspan="3">0.4%</td><td>WEIG</td><td>0.1099</td><td>0.1006</td><td colspan="2">0.12420.0398</td><td>0.1552</td><td>0.1196</td><td colspan="2">0.16230.0576</td><td>0.1037</td><td>0.0763</td><td colspan="2">0.10910.0384</td><td colspan="2">0.73040.5751</td></tr><tr><td>RoBA</td><td>0.0582</td><td>0.0649</td><td colspan="2">0.06900.0223</td><td>0.0763</td><td>0.0693</td><td colspan="2">0.07770.0299</td><td>0.0607</td><td>0.0475</td><td colspan="2">0.06340.0238</td><td colspan="2">0.57060.2187</td></tr><tr><td>BPFA</td><td>0.1100</td><td>0.1003</td><td colspan="2">0.12420.0398</td><td>0.1552</td><td>0.1194</td><td colspan="2">0.16230.0576</td><td>0.1037</td><td>0.0762</td><td colspan="2">0.10920.0384</td><td colspan="2">0.73050.5751</td></tr><tr><td rowspan="3">0.7%</td><td>WEIG</td><td>0.1100</td><td>0.1005</td><td colspan="2">0.12420.0398</td><td>0.1553</td><td>0.1196</td><td colspan="2">0.16240.0577</td><td>0.1037</td><td>0.0763</td><td colspan="2">0.10920.0384</td><td colspan="2">0.73040.5750</td></tr><tr><td>RoBA</td><td>0.0467</td><td>0.0600</td><td colspan="2">0.05670.0191</td><td>0.0654</td><td>0.0645</td><td colspan="2">0.06560.0261</td><td>0.0390</td><td>0.0445</td><td colspan="2">0.03790.0156</td><td colspan="2">0.56900.7860</td></tr><tr><td>BPFA</td><td>0.1098</td><td>0.1000</td><td colspan="2">0.12390.0398</td><td>0.1547</td><td>0.1190</td><td colspan="2">0.16190.0575</td><td>0.1033</td><td>0.0759</td><td colspan="2">0.10880.0383</td><td colspan="2">0.73030.5757</td></tr><tr><td rowspan="3">1%</td><td>WEIG</td><td>0.1098</td><td>0.1003</td><td colspan="2">0.12400.0398</td><td>0.1550</td><td>0.1194</td><td colspan="2">0.16210.0576</td><td>0.1035</td><td>0.0761</td><td colspan="2">0.10900.0384</td><td colspan="2">0.73040.5751</td></tr><tr><td>RoBA</td><td>0.0400</td><td>0.0621</td><td colspan="2">0.05030.0155</td><td>0.0556</td><td>0.0639</td><td colspan="2">0.05390.0219</td><td>0.0327</td><td>0.0465</td><td colspan="2">0.02990.0128</td><td colspan="2">0.56700.8283</td></tr><tr><td>BPFA</td><td>0.1096</td><td>0.0999</td><td colspan="2">0.12370.0397</td><td>0.1546</td><td>0.1189</td><td colspan="2">0.16170.0574</td><td>0.1032</td><td>0.0758</td><td colspan="2">0.10870.0382</td><td colspan="2">0.73050.5760</td></tr><tr><td rowspan="3">4%</td><td>WEIG</td><td>0.1105</td><td>0.1008</td><td colspan="2">0.12490.0400</td><td>0.1560</td><td>0.1199</td><td colspan="2">0.16320.0579</td><td>0.1042</td><td>0.0765</td><td colspan="2">0.10970.0386</td><td colspan="2">0.73070.5752</td></tr><tr><td>RoBA</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>BPFA</td><td>0.1099</td><td>0.1016</td><td colspan="2">0.12440.0397</td><td>0.1551</td><td>0.1204</td><td colspan="2">0.16210.0575</td><td>0.1032</td><td>0.0765</td><td colspan="2">0.10850.0382</td><td colspan="2">0.73140.5812</td></tr><tr><td rowspan="3">7%</td><td>WEIG</td><td>0.1110</td><td>0.1026</td><td colspan="2">0.12600.0402</td><td>0.1578</td><td>0.1221</td><td colspan="2">0.16490.0585</td><td>0.1061</td><td>0.0781</td><td colspan="2">0.11170.0392</td><td colspan="2">0.73170.5678</td></tr><tr><td>RoBA</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>BPFA</td><td>0.1029</td><td>0.1153</td><td colspan="2">0.11680.0372</td><td>0.1562</td><td>0.1370</td><td colspan="2">0.16000.0582</td><td>0.1064</td><td>0.0871</td><td colspan="2">0.11030.0395</td><td colspan="2">0.71510.5413</td></tr><tr><td rowspan="3">10%</td><td>WEIG</td><td>0.1114</td><td>0.1012</td><td colspan="2">0.12640.0402</td><td>0.1579</td><td>0.1207</td><td colspan="2">0.16540.0586</td><td>0.1070</td><td>0.0777</td><td colspan="2">0.11290.0396</td><td colspan="2">0.73000.5574</td></tr><tr><td>RoBA</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>BPFA</td><td>0.0946</td><td>0.0822</td><td colspan="2">0.10710.0351</td><td>0.1453</td><td>0.1053</td><td colspan="2">0.15330.0542</td><td>0.1048</td><td>0.0711</td><td colspan="2">0.11150.0389</td><td colspan="2">0.73120.4728</td></tr><tr><td rowspan="3">Pruning Rate</td><td rowspan="3">Method</td><td colspan="4">Naive Metric↓</td><td colspan="4">Approach Averaged Metric↓</td><td colspan="4">Utility Regularized Metric↓</td><td colspan="2">Utility↑</td></tr><tr><td rowspan="2">DPD</td><td rowspan="2">DEOdds</td><td rowspan="2">DEO</td><td rowspan="2">STD</td><td>MA</td><td>MA</td><td>MA</td><td>MA</td><td>UR</td><td>UR</td><td>UR</td><td>UR</td><td rowspan="2">AUC</td><td rowspan="2">ACC</td></tr><tr><td>DPD</td><td>DEOdds</td><td>DEO</td><td>STD</td><td>DPD</td><td>DEOdds</td><td>DEO</td><td>STD</td></tr><tr><td>0</td><td>Original</td><td>0.0805</td><td>0.1396</td><td>0.1032</td><td>0.0328</td><td>0.1393</td><td>0.1560</td><td>0.1360</td><td>0.0530</td><td>0.0881</td><td>0.0986</td><td>0.0860</td><td>0.0335</td><td>0.6302</td><td>0.6019</td></tr><tr><td rowspan="3">0.1%</td><td>WEIG</td><td>0.0789</td><td>0.1340</td><td>0.1013</td><td>0.0319</td><td>0.1349</td><td>0.1494</td><td>0.1320</td><td>0.0513</td><td>0.0853</td><td>0.0945</td><td>0.0835</td><td>0.0324</td><td>0.6298</td><td>0.6021</td></tr><tr><td>RoBA</td><td>0.0216</td><td>0.0369</td><td>0.0278</td><td>0.0078</td><td>0.0316</td><td>0.0381</td><td>0.0303</td><td>0.0122</td><td>0.0181</td><td>0.0296</td><td>0.0158</td><td>0.0070</td><td>0.5468</td><td>0.8798</td></tr><tr><td>BPFA</td><td>0.0785</td><td>0.1392</td><td>0.1010</td><td>0.0320</td><td>0.1366</td><td>0.1552</td><td>0.1329</td><td>0.0521</td><td>0.0864</td><td>0.0982</td><td>0.0840</td><td>0.0329</td><td>0.6300</td><td>0.6039</td></tr><tr><td rowspan="3">0.4%</td><td>WEIG</td><td>0.0789</td><td>0.1340</td><td>0.1012</td><td>0.0319</td><td>0.1349</td><td>0.1494</td><td>0.1320</td><td>0.0513</td><td>0.0853</td><td>0.0944</td><td>0.0835</td><td>0.0324</td><td>0.6298</td><td>0.6021</td></tr><tr><td>RoBA</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>BPFA</td><td>0.0594</td><td>0.1337</td><td>0.0796</td><td>0.0238</td><td>0.1079</td><td>0.1442</td><td>0.1006</td><td>0.0411</td><td>0.0644</td><td>0.0969</td><td>0.0578</td><td>0.0245</td><td>0.6445</td><td>0.7415</td></tr><tr><td rowspan="3">0.7%</td><td>WEIG</td><td>0.0789</td><td>0.1340</td><td>0.1012</td><td>0.0319</td><td>0.1349</td><td>0.1494</td><td>0.1320</td><td>0.0513</td><td>0.0853</td><td>0.0945</td><td>0.0835</td><td>0.0324</td><td>0.6298</td><td>0.6021</td></tr><tr><td>RoBA</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>BPFA</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td rowspan="3">1%</td><td>WEIG</td><td>0.0789</td><td>0.1340</td><td>0.1012</td><td>0.0319</td><td>0.1349</td><td>0.1494</td><td>0.1320</td><td>0.0513</td><td>0.0853</td><td>0.0945</td><td>0.0835</td><td>0.0324</td><td>0.6298</td><td>0.621</td></tr><tr><td>RoBA</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>BPFA</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td rowspan="3">4%</td><td>WEIG</td><td>0.0786</td><td>0.1344</td><td>0.1010</td><td>0.0318</td><td>0.1348</td><td>0.1499</td><td>0.1318</td><td>0.0513</td><td>0.0852</td><td>0.0947</td><td>0.0833</td><td>0.0324</td><td>0.6299</td><td>0.6022</td></tr><tr><td>RoBA</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>BPFA</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td rowspan="3">7%</td><td>WEIG</td><td>0.0806</td><td>0.1359</td><td>0.1032</td><td>0.0324</td><td>0.1355</td><td>0.1505</td><td>0.1324</td><td>0.0514</td><td>0.0859</td><td>0.0950</td><td>0.0841</td><td>0.0326</td><td>0.6289</td><td>0.5959</td></tr><tr><td>RoBA</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>BPFA</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td rowspan="3">10%</td><td>WEIG</td><td>0.0837</td><td>0.1382</td><td>0.1067</td><td>0.0331</td><td>0.1370</td><td>0.1518</td><td>0.1340</td><td>0.0520</td><td>0.0870</td><td>0.0959</td><td>0.0852</td><td>0.0330</td><td>0.6282</td><td>0.5936</td></tr><tr><td>RoBA</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>BPFA</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr></table>

Table 10: Ablation study on pruning rate of SPSL. We use '-' to indicate that the model is completely unusable (AUC = 0.5).

Table 11: Ablation study on pruning rate of FFD. We use '-' to indicate that the model is completely unusable (AUC = 0.5).

Table 12: Ablation study on pruning rate of PFGDFD. We use '-' to indicate that the model is completely unusable (AUC = 0.5).

<table><tr><td>Metric</td><td>DPD</td><td>AADPD</td><td>URDPD</td><td>DEOdds</td><td>AADEOdds</td><td>URDEOdds</td></tr><tr><td>1</td><td>SPSL</td><td>SPSL</td><td>SPSL</td><td>SPSL</td><td>SPSL</td><td>SPSL</td></tr><tr><td>2</td><td>DAW</td><td>DAW</td><td>DAW</td><td>DAW</td><td>DAW</td><td>DAW</td></tr><tr><td>3</td><td>F3Net</td><td>CORE</td><td>CORE</td><td>SRM</td><td>CORE</td><td>CORE</td></tr><tr><td>4</td><td>PFGDFD</td><td>F3Net</td><td>F3Net</td><td>F3Net</td><td>SRM</td><td>SRM</td></tr><tr><td>5</td><td>CORE</td><td>Capsule</td><td>Capsule</td><td>CORE</td><td>F3Net</td><td>F3Net</td></tr><tr><td>6</td><td>Capsule</td><td>PFGDFD</td><td>PFGDFD</td><td>Capsule</td><td>Capsule</td><td>Capsule</td></tr><tr><td>7</td><td>SRM</td><td>SRM</td><td>SRM</td><td>FFD</td><td>FFD</td><td>FFD</td></tr><tr><td>8</td><td>FFD</td><td>FFD</td><td>FFD</td><td>RECCE</td><td>RECCE</td><td>RECCE</td></tr><tr><td>9</td><td>RECCE</td><td>RECCE</td><td>RECCE</td><td>PFGDFD</td><td>PFGDFD</td><td>PFGDFD</td></tr><tr><td>10</td><td>DAG</td><td>Xception</td><td>Xception</td><td>UCF</td><td>UCF</td><td>Xception</td></tr><tr><td>11</td><td>UCF</td><td>UCF</td><td>DAG</td><td>Xception</td><td>Xception</td><td>UCF</td></tr><tr><td>12</td><td>Xception</td><td>DA</td><td>UCF</td><td>DAG</td><td>DAG</td><td>DAG</td></tr><tr><td>Metric</td><td>DEO</td><td>AADEO</td><td>URDEO</td><td>STD</td><td>AASTD</td><td>URSTD</td></tr><tr><td>1</td><td>SPSL</td><td>SPSL</td><td>SPSL</td><td>SPSL</td><td>SPSL</td><td>SPSL</td></tr><tr><td>2</td><td>DAW</td><td>DAW</td><td>DAW</td><td>DAW</td><td>DAW</td><td>DAW</td></tr><tr><td>3</td><td>F3Net</td><td>F3Net</td><td>Capsule</td><td>F3Net</td><td>F3Net</td><td>Capsule</td></tr><tr><td>4</td><td>PFGDFD</td><td>CORE</td><td>F3Net</td><td>PFGDFD</td><td>CORE</td><td>F3Net</td></tr><tr><td>5</td><td>CORE</td><td>PFGDFD</td><td>CORE</td><td>CORE</td><td>Capsule</td><td>CORE</td></tr><tr><td>6</td><td>SRM</td><td>Capsule</td><td>PFGDFD</td><td>Capsule</td><td>PFGDFD</td><td>PFGDFD</td></tr><tr><td>7</td><td>Capsule</td><td>SRM</td><td>SRM</td><td>SRM</td><td>SRM</td><td>SRM</td></tr><tr><td>8</td><td>FFD</td><td>FFD</td><td>FFD</td><td>FFD</td><td>FFD</td><td>FFD</td></tr><tr><td>9</td><td>RECCE</td><td>RECCE</td><td>RECCE</td><td>RECCE</td><td>RECCE</td><td>RECCE</td></tr><tr><td>10</td><td>UCF</td><td>Xception</td><td>Xception</td><td>UCF</td><td>Xception</td><td>Xception</td></tr><tr><td>11</td><td>Xception</td><td>DAG</td><td>DAG</td><td>DAG</td><td>UCF</td><td>DAG</td></tr><tr><td>12</td><td>DAG</td><td>UCF</td><td>UCF</td><td>Xception</td><td>DAG</td><td>UCF</td></tr></table>

Table 13: Ranking results of DPD, DEOdds and their variants.

Table 14: Ranking results of DEO, STD and their variants.