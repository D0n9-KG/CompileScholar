# UniTSFace: Unified Threshold Integrated Sample-to-Sample Loss for Face Recognition

Qiufu Li $^{1,2,6,\#}$

liqiufu@szu.edu.cn

Xi Jia 1,2,3,#

x.jia.1@cs.bham.ac.uk

Jiancan Zhou 1,2,4,#

zhoujiancan@foxmail.com

Linlin Shen 1,2,6,\*

llshen@szu.edu.cn

Jinming Duan $^{3,5}$

j.duan@bham.ac.uk

$^{1}$ National Engineering Laboratory for Big Data System Computing Technology, Shenzhen University, China

$^{2}$ Computer Vision Institute, Shenzhen University, China

$^{3}$ School of Computer Science, University of Birmingham, UK

$^{4}$ Aqara, Lumi United Technology Co., Ltd, China

$^{5}$ Alan Turing Institute, UK

$^{6}$ SZU Branch, Shenzhen Institute of Artificial Intelligence and Robotics for Society, China

# Abstract

Sample-to-class-based face recognition models can not fully explore the cross-sample relationship among large amounts of facial images, while sample-to-sample-based models require sophisticated pairing processes for training. Furthermore, neither method satisfies the requirements of real-world face verification applications, which expect a unified threshold separating positive from negative facial pairs. In this paper, we propose a unified threshold integrated sample-to-sample based loss (USS loss), which features an explicit unified threshold for distinguishing positive from negative pairs. Inspired by our USS loss, we also derive the sample-to-sample based softmax and BCE losses, and discuss their relationship. Extensive evaluation on multiple benchmark datasets, including MFR, IJB-C, LFW, CFP-FP, AgeDB, and MegaFace, demonstrates that the proposed USS loss is highly efficient and can work seamlessly with sample-to-class-based losses. The embedded loss (USS and sample-to-class Softmax loss) overcomes the pitfalls of previous approaches and the trained facial model UniTSFace exhibits exceptional performance, outperforming state-of-the-art methods, such as CosFace, ArcFace, VPL, AnchorFace, and UNPG. Our code is available at https://github.com/CVI-SZU/UniTSFace.

# 1 Introduction

Modern deep facial recognition systems, involving an enormous number of facial images and identities, essentially rely on discriminative feature learning: the facial images from the same identity should be close while those from different identities should be distant in the feature space. That is, the similarity of a positive pair (two facial images from the same identity) is required to be larger than any negative pair (two facial images from different identities). In other words, a unified threshold is expected to distinguish positive from negative pairs.

The feature learning process of general deep face recognition models is either based on sample-to-class losses (such as Softmax loss $[27, 28]$ ) or sample-to-sample losses (such as contrastive $[4, 10, 25]$ )

and triplet loss [22]). Inspired by the success of large-scale image classification, the softmax loss and its extensions have become popular in deep face recognition systems. In the softmax loss, each weight vector can be regarded as a proxy of the corresponding class (face identity), the classification-based face recognition models are demanded to learn the class proxy and image features simultaneously. However, these models possess a significant drawback, namely a domain gap between the training and testing stages of face models. This gap arises from the reliance on limited identity proxies for computing and optimizing feature similarity in sample-to-class models. Conversely, in real-world scenarios, feature similarity is computed and compared across various samples from diverse facial identities. In other words, the sample-to-class-based classification strategy may not entirely explore the variances across samples (Problem 1). Therefore, its efficacy in accurately reflecting real-world scenarios is questionable. To tackle this drawback, VPL [8] considers a small variation around the class proxy, which implicitly introduces more samples during the optimization and improves the face recognition performance. However, while this small variation extends the capability of softmax loss, it cannot essentially represent all the samples in this class.

Face models based on sample-to-sample losses $[4, 10, 22, 3, 24]$ learn facial identity features and optimize feature similarities by comparing positive and negative samples, which is closer to real-world face recognition applications than sample-to-class losses. The majority of sample-to-sample losses aim to maximize inter-class discrepancy while minimizing intra-class distances, which may need a meticulous sampling/paring step for every mini-batch. Moreover, in face verification tasks, a single threshold is required to distinguish positive facial pairs from negative ones. Unfortunately, none of the aforementioned methods incorporate such an explicit constraint (Problem 2).

In this study, we commence our research by analyzing a reasonable, albeit naive, sample-to-sample loss, which enables us to straightforwardly investigate the variances across facial samples, thereby resolving Problem 1. To address Problem 2, we integrate a unified learnable threshold into the naive loss, resulting in a novel sample-to-sample loss function that we term the unified threshold integrated sample-to-sample (USS) loss. We provide a mathematical elaboration of the relationship between our USS loss and the sample-to-sample BCE loss and softmax loss, from the perspective of the naive sample-to-sample loss. To evaluate the effectiveness of the USS loss, we conduct experiments on various benchmark datasets and qualitatively demonstrate that it meets the requirements of real-world applications. Additionally, we find that incorporating a margin can easily enhance the USS loss. Our USS loss also works seamlessly with the sample-to-class based losses and shows significant improvements when they are combined together to train a face model called UniTSFace. To summarize, the contributions of this work are

- We introduce a unified threshold integrated sample-to-sample loss (USS) derived from a reasonable naive loss for face recognition. We also derive the sample-to-sample based softmax and binary-cross-entropy (BCE) losses from the naive loss and reveal the mathematical relationship among the softmax, BCE, and our USS losses.   
- We demonstrate that a unified threshold can be learned by adopting the proposed USS loss and quantitatively and qualitatively demonstrate that the learned threshold aligns with face verification expectations in experiments.   
- The proposed USS loss can be enhanced with an auxiliary margin and is compatible with existing sample-to-class based losses. We show that USS loss, when used jointly with sample-to-class based losses such as CosFace and ArcFace, leads to a continuous improvement.   
- Our UniTSFace outperforms state-of-the-art methods on the Megaface 1 dataset and ranks first place on the MFR ongoing challenge till the submission of this work (May 17 '23, academic track): http://iccv21-mfr.com/#/leaderboard/academic.

# 2 Related Works

Sample-to-Sample based Methods. DeepID2 [25] uses a contrastive loss[4, 10] to encourage the features learned from the same identity to be close while that learned from different identities are distant. FaceNet[22] constructs three-element tuples and minimizes the distance between an anchor and a positive sample and maximizes the distance between the anchor and a negative sample. The effectiveness of contrastive/triplet losses relies on the meticulous selection of pairs/triplets. Furthermore, neither method explicitly enforces that all positive sample-to-sample similarities are greater than negative similarities.

Sample-to-Class based Methods. While Softmax loss is frequently utilized in deep recognition models, it only promotes separability and does not learn discriminative features. Various approaches have been proposed to enhance the feature learning of softmax loss. Some methods $[32, 9, 34, 16]$ proposed extra constraints imposed on the softmax loss, such methods normally have a class proxy (class center/prototype) and maximize (minimize) the similarity (distances) between a facial feature and its corresponding class proxy $[32]$ , as well as maximize the distances between all class proxies $[16, 9, 34]$ . However, the training process of such methods needs careful balancing of the softmax loss and the extra constraints. Alternatively, some works directly improve the softmax by normalizing the facial features and adding margins between positive and negative sample-to-class pairs $[18, 29, 30, 7, 17]$ . While sample-to-class methods demonstrate excellent performance in deep face verification, they may not entirely explore the variability across various facial samples.

Hybrid (/combined) Methods. Circle loss [26] is one of the first works that discussed the two elemental learning paradigms, i.e., learning with class-level labels and pair-wise labels, in a unified framework. However, the two learning paradigms are respectively used in circle loss. Following [26], the combination of the two learning paradigms to solve the shortcoming of either method become popular, e.g., VPL[8], AnchorFace[15], and UNPG[13]. VPL extends the softmax loss by considering a small feature variation around the class proxy W. The embedding of a small feature variation cannot essentially represent all samples, VPL still presents a large gap between training a face recognition model and testing over the open sets. UNPG jointly optimizes the distance between a sample $x_{i}$ with negative class proxies $W_{j}$ and negative samples $x_{j}$ in the same softmax loss using batch processing, focusing on the negative pairs during training. AnchorFace, however, uses a combination of softmax loss and different sample-to-sample losses, such as TAR Loss and FAR Loss, to target specific testing protocols. In contrast, our USS loss is more general and compatible with existing sample-to-class losses. We demonstrate our USS loss can lead to a steady improvement when used with sample-to-class losses jointly in experiments.

# 3 Methods

Suppose $\mathcal{M}$ is a deep face model trained on a facial sample set $\mathcal{D} = \bigcup_{i=1}^{N} \mathcal{D}_i$ captured from $N$ subjects, where $\mathcal{D}_i$ denotes the subset containing the facial samples captured from the same subject $i$ . Then, we can get a feature set $\mathcal{F} = \bigcup_{i=1}^{N} \mathcal{F}_i = \bigcup_{i=1}^{N} \left\{\boldsymbol{x}^{(i)} = \mathcal{M}(\boldsymbol{X}^{(i)}) : \boldsymbol{X}^{(i)} \in \mathcal{D}_i\right\}$ , where $\boldsymbol{x}^{(i)}$ is the feature vector of the sample $\boldsymbol{X}^{(i)}$ and $\boldsymbol{X}^{(i)}$ denotes the sample captured from subject $i$ .

For any two samples $X, X_{*} \in D$ , we apply a bivariate operator $g(\boldsymbol{x}, \boldsymbol{x}_{*}) \in [-1, 1]$ denoting their feature similarity, where $\boldsymbol{x} = \mathcal{M}(\boldsymbol{X}), \boldsymbol{x}_{*} = \mathcal{M}(\boldsymbol{X}_{*})$ are features of $X, X_{*}$ . We term $g(\boldsymbol{x}, \boldsymbol{x}_{*})$ the positive sample-to-sample similarity if the two samples are captured from the same subject, while the negative sample-to-sample similarity if they are from different subjects. Then, for any sample $\boldsymbol{X}^{(i)} \in \mathcal{D}_{i}, \forall i$ , we define its positive and negative similarity sets as

$$
\boldsymbol {s} _ {\boldsymbol {X} ^ {(i)}} ^ {(\text { pos })} = \left\{g (\boldsymbol {x} ^ {(i)}, \boldsymbol {x} _ {*}): \boldsymbol {x} _ {*} \in \mathcal {F} _ {i} \right\}, \tag {1}
$$

$$
\boldsymbol {s} _ {\boldsymbol {X} ^ {(i)}} ^ {(\text { neg })} = \bigcup_ {\substack {j = 1 \\ j \neq i}} ^ {N} \left\{g (\boldsymbol {x} ^ {(i)}, \boldsymbol {x} _ {*}): \boldsymbol {x} _ {*} \in \mathcal {F} _ {j} \right\}, \tag{2}
$$

where $\boldsymbol{x}^{(i)}=\mathcal{M}(\boldsymbol{X}^{(i)})$ .

In real applications of face verification, a threshold $\hat{t} \in [-1, 1]$ is chosen to verify whether two samples X and $X_{*}$ are from the same subject or not. Specifically, the two facial samples X, $X_{*}$ are with the same identity if $g(\boldsymbol{x}, \boldsymbol{x}_{*}) \geq \hat{t}$ , while they are from two different identities when $g(\boldsymbol{x}, \boldsymbol{x}_{*}) < \hat{t}$ .

To be in line with face verification applications, during the training process of $\mathcal{M}$ , we expect a unified threshold $t$ such that any two samples $(\pmb{X}^{(i)} \in \mathcal{D}_i$ and $\pmb{X}_*^{(j)} \in \mathcal{D}_j)$ conform to the above rules. If they share the same identity (i.e., $i = j$ ), then $g(\pmb{x}^{(i)}, \pmb{x}_*^{(j)}) \geq t$ , in the case of $i \neq j$ , $g(\pmb{x}^{(i)}, \pmb{x}_*^{(j)}) < t$ , where $\pmb{x}^{(i)} = \mathcal{M}(\pmb{X}^{(i)})$ , $\pmb{x}_*^{(j)} = \mathcal{M}(\pmb{X}_*^{(j)})$ . The unified threshold $t$ satisfies

$$
\max \left(\bigcup_ {i = 1} ^ {N} \bigcup_ {\boldsymbol {X} ^ {(i)} \in \mathcal {D} _ {i}} \boldsymbol {s} _ {\boldsymbol {X} ^ {(i)}} ^ {(\text { neg })}\right) <   t \leq \min \left(\bigcup_ {i = 1} ^ {N} \bigcup_ {\boldsymbol {X} ^ {(i)} \in \mathcal {D} _ {i}} \boldsymbol {s} _ {\boldsymbol {X} ^ {(i)}} ^ {(\text { pos })}\right). \tag {3}
$$

However, the existing sample-to-sample losses $[4, 10, 22, 24]$ fail to explicitly include and learn the unified threshold. In this paper, by explicitly defining the unified threshold t, we design a unified threshold integrated sample-to-sample loss.

# 3.1 Unified Threshold Integraed Sample-to-Sample Loss (USS Loss)

Clearly, in the training of model M, it expects large positive sample-to-sample similarities but small negative ones, and then, for any sample $\boldsymbol{X}^{(i)}$ , a naive loss could be reasonably defined as,

$$
L _ {\text {NAIVE}} \left(\boldsymbol {X} ^ {(i)}\right) = - \frac {1}{| \mathcal {F} _ {i} |} \sum_ {\boldsymbol {x} \in \mathcal {F} _ {i}} \gamma g \left(\boldsymbol {x} ^ {(i)}, \boldsymbol {x}\right) + \frac {1}{| \mathcal {F} - \mathcal {F} _ {i} |} \sum_ {\substack {j = 1 \\ j \neq i}} ^ {N} \sum_ {\boldsymbol {x} \in \mathcal {F} _ {j}} \gamma g \left(\boldsymbol {x} ^ {(i)}, \boldsymbol {x}\right), \tag{4}
$$

where $\gamma$ is a scale factor, $|\mathcal{S}|$ denotes the element number of a set $\mathcal{S}$ , and $\mathcal{F} - \mathcal{F}_i$ denotes the complementary set of $\mathcal{F}_i$ in $\mathcal{F}$ , i.e., $\mathcal{F} - \mathcal{F}_i = \bigcup_{\substack{j=1 \\ j \neq i}}^N \mathcal{F}_j$ .

The loss in Eq. (4) computes the feature similarities of all positive sample-to-sample pairs and all negative ones for the sample $\boldsymbol{X}^{(i)}$ , which is not easily implemented in practice. In this paper, without loss of generality, we consider the feature similarities of one positive sample pair and N - 1 negative sample pairs for the sample $\boldsymbol{X}^{(i)}$ in the naive loss,

$$
L _ {\text { naive }} \left(\boldsymbol {X} ^ {(i)}\right) = - \gamma g \left(\boldsymbol {x} ^ {(i)}, \boldsymbol {x} _ {*} ^ {(i)}\right) + \frac {1}{N - 1} \sum_ {\substack {j = 1 \\ j \neq i}} ^ {N} \gamma g \left(\boldsymbol {x} ^ {(i)}, \boldsymbol {x} _ {*} ^ {(j)}\right), \tag{5}
$$

where $\boldsymbol{x}_{*}^{(i)} = \mathcal{M}(\boldsymbol{X}_{*}^{(i)})$ , $\boldsymbol{x}_{*}^{(j)} = \mathcal{M}(\boldsymbol{X}_{*}^{(j)})$ , and $X_{*}^{(i)}$ and $X_{*}^{(j)}$ are randomly taken from the subject i, j, with $j \neq i$ .

Using the inequality of arithmetic and geometric means $^{1}$ , we derive two inequalities about $L_{naive}$ ,

$$
L _ {\text { naive }} \left(\boldsymbol {X} ^ {(i)}\right) \leq 2 \log \left(1 + \frac {\exp \left(\sum_ {j = 1} ^ {N} \frac {\gamma g \left(\boldsymbol {x} ^ {(i)} , \boldsymbol {x} _ {*} ^ {(j)}\right)}{N - 1}\right)}{\exp \left(\gamma g \left(\boldsymbol {x} ^ {(i)} , \boldsymbol {x} _ {*} ^ {(i)}\right)\right)}\right) - 2 \log 2, \tag {6}
$$

$$
L _ {\text { naive }} \left(\boldsymbol {X} ^ {(i)}\right) \leq \frac {2}{N - 1} \sum_ {\substack {j = 1 \\ j \neq i}} ^ {N} \log \left(1 + \frac {\mathrm{e} ^ {\gamma g \left(\boldsymbol {x} ^ {(i)} , \boldsymbol {x} _ {*} ^ {(j)}\right)}}{\mathrm{e} ^ {\gamma g \left(\boldsymbol {x} ^ {(i)} , \boldsymbol {x} _ {*} ^ {(i)}\right)}}\right) - 2 \log 2. \tag{7}
$$

According to Eqs. (6), (7), and the unified threshold $t$ in Eq. (3), one can get

$$
\begin{array}{l} \frac {N}{2} L _ {\text { naive }} (\boldsymbol {X} ^ {(i)}) + N \log 2 \\ \leq \log \left(1 + \frac {\exp \left(\sum_ {j = 1} ^ {N} \frac {\gamma g \left(\boldsymbol {x} ^ {(i)} , \boldsymbol {x} _ {*} ^ {(j)}\right)}{N - 1}\right)}{\exp \left(\gamma g \left(\boldsymbol {x} ^ {(i)} , \boldsymbol {x} _ {*} ^ {(i)}\right)\right)}\right) + \sum_ {\substack {j = 1 \\ j \neq i}} ^ {N} \log \left(1 + \frac {\exp \left(\gamma g \left(\boldsymbol {x} ^ {(i)} , \boldsymbol {x} _ {*} ^ {(j)}\right)\right)}{\exp \left(\gamma g \left(\boldsymbol {x} ^ {(i)} , \boldsymbol {x} _ {*} ^ {(i)}\right)\right)}\right) (8) \\ \leq \log \left(1 + \frac {\exp \left(\sum_ {j = 1} ^ {N} \frac {\gamma t}{N - 1}\right)}{\exp \left(\gamma g \left(\boldsymbol {x} ^ {(i)} , \boldsymbol {x} _ {*} ^ {(i)}\right)\right)}\right) + \sum_ {\substack {j = 1 \\ j \neq i}} ^ {N} \log \left(1 + \frac {\exp \left(\gamma g \left(\boldsymbol {x} ^ {(i)} , \boldsymbol {x} _ {*} ^ {(j)}\right)\right)}{\exp (\gamma t)}\right) (9) \\ = \log \left(1 + \mathrm{e} ^ {- \gamma g (\boldsymbol {x} ^ {(i)}, \boldsymbol {x} _ {*} ^ {(i)}) + \gamma t}\right) + \sum_ {\substack {j = 1 \\ j \neq i}} ^ {N} \log \left(1 + \mathrm{e} ^ {\gamma g (\boldsymbol {x} ^ {(i)}, \boldsymbol {x} _ {*} ^ {(j)}) - \gamma t}\right). (10) \\ \end{array}
$$

We define the Unified threshold integrated Sample-to-Sample (USS) loss $L_{\mathrm{uss}}(\boldsymbol{X}^{(i)})$ as

$$
L _ {\mathrm{uss}} \left(\boldsymbol {X} ^ {(i)}\right) = \log \left(1 + \mathrm{e} ^ {- \gamma g \left(\boldsymbol {x} ^ {(i)}, \boldsymbol {x} _ {*} ^ {(i)}\right) + b}\right) + \sum_ {\substack {j = 1 \\ j \neq i}} ^ {N} \log \left(1 + \mathrm{e} ^ {\gamma g \left(\boldsymbol {x} ^ {(i)}, \boldsymbol {x} _ {*} ^ {(j)}\right) - b}\right) \tag{11}
$$

where $b = \gamma t$ is a constant to be learned (depicted in Fig. 1). The detailed derivations of the above inequalities are described in the supplementary (appendix).

We here analyze that the unified threshold $t$ could be learned. Suppose that the model $\mathcal{M}$ has been perfectly trained, and $L_{\mathrm{uss}}$ has reached its minimum point after the training, which means (i) the positive sample-to-sample similarity $g(\pmb{x}^{(i)},\pmb{x}_{*}^{(i)})$ tends to 1, and the negative ones $g(\pmb{x}^{(i)},\pmb{x}_{*}^{(j)})$ tends to -1; and (ii) $L_{\mathrm{uss}}$ reaches its stationary point in terms of variable $b$ . From (ii), one can deduce that

$$
0 = \frac {\partial L _ {\mathrm{uss}}}{\partial b} = \frac {\mathrm{e} ^ {- \gamma g (\boldsymbol {x} ^ {(i)} , \boldsymbol {x} _ {*} ^ {(i)}) + b}}{1 + \mathrm{e} ^ {- \gamma g (\boldsymbol {x} ^ {(i)} , \boldsymbol {x} _ {*} ^ {(i)}) + b}} - \sum_ {\substack {j = 1 \\ j \neq i}} ^ {N} \frac {\mathrm{e} ^ {\gamma g (\boldsymbol {x} ^ {(i)} , \boldsymbol {x} _ {*} ^ {(j)}) - b}}{1 + \mathrm{e} ^ {\gamma g (\boldsymbol {x} ^ {(i)} , \boldsymbol {x} _ {*} ^ {(j)}) - b}} \tag{12}
$$

$$
\stackrel {\text {(i)}}{=} \frac {\mathrm{e} ^ {- \gamma + b}}{1 + \mathrm{e} ^ {- \gamma + b}} - \sum_ {\substack {j = 1 \\ j \neq i}} ^ {N} \frac {\mathrm{e} ^ {- \gamma - b}}{1 + \mathrm{e} ^ {- \gamma - b}} \tag{13}
$$

$$
\Rightarrow b = \log \frac {(N - 2) \mathrm{e} ^ {- \gamma} + \sqrt {(N - 2) ^ {2} \mathrm{e} ^ {- 2 \gamma} + 4 (N - 1)}}{2}. \tag {14}
$$

If $N < \frac{e^{2\gamma} + 3}{2}$ , the final learned threshold $t = \frac{b}{\gamma}$ will locate between -1 and 1, which means now the leaned t has correctly separated the negative sample-to-sample similarities, $g(\boldsymbol{x}^{(i)}, \boldsymbol{x}_{*}^{(j)})$ , and the positive ones, $g(\boldsymbol{x}^{(i)}, \boldsymbol{x}_{*}^{(i)})$ . In the practice, following [30, 7], we set $\gamma = 64$ , then, according to the above analysis, the unified threshold t could be leaned if $N < 1.9 \times 10^{55}$ .

# 3.2 Sample-to-Sample Based Softmax and BCE Losses

Sample-to-class based softmax and BCE losses are widely applied in the image classification. For the study of face verification, we here deduce the sample-to-sample based softmax and BCE losses from the naive loss $L_{\mathrm{naive}}$ .

Similar to the deduction of USS loss, using the inequality of arithmetic and geometric means, we first present another two inequalities about $L_{naive}$ ,

$$
L _ {\text { naive }} \left(\boldsymbol {X} ^ {(i)}\right) \leq - \frac {N}{N - 1} \log \frac {\mathrm{e} ^ {\gamma g \left(\boldsymbol {x} ^ {(i)} , \boldsymbol {x} _ {*} ^ {(i)}\right)}}{\sum_ {j = 1} ^ {N} \mathrm{e} ^ {\gamma g \left(\boldsymbol {x} ^ {(i)} , \boldsymbol {x} _ {*} ^ {(j)}\right)}} - \frac {N \log N}{N - 1}, \tag {15}
$$

$$
\sum_ {i = 1} ^ {N} L _ {\text { naive }} \left(\boldsymbol {X} ^ {(i)}\right) \leq \frac {2}{N - 1} \sum_ {i = 1} ^ {N} \sum_ {\substack {j = 1 \\ j \neq i}} ^ {N} \log \left(1 + \frac {\mathrm{e} ^ {\gamma g \left(\boldsymbol {x} ^ {(i)} , \boldsymbol {x} _ {*} ^ {(j)}\right)}}{\mathrm{e} ^ {\gamma g \left(\boldsymbol {x} ^ {(j)} , \boldsymbol {x} _ {*} ^ {(j)}\right)}}\right) - 2 N \log 2. \tag{16}
$$

Softmax Loss. For sample $\pmb{X}^{(i)} \in \mathcal{D}_i$ with $\pmb{x}^{(i)} = \mathcal{M}(\pmb{X}^{(i)})$ , we define its sample-to-sample softmax loss as

$$
L _ {\text { soft }} (\boldsymbol {X} ^ {(i)}) = - \log \frac {\mathrm{e} ^ {\gamma g (\boldsymbol {x} ^ {(i)} , \boldsymbol {x} _ {*} ^ {(i)})}}{\sum_ {j = 1} ^ {N} \mathrm{e} ^ {\gamma g (\boldsymbol {x} ^ {(i)} , \boldsymbol {x} _ {*} ^ {(j)})}}. \tag {17}
$$

Then, according to Eq. (15), one can get

$$
L _ {\text { naive }} (\boldsymbol {X} ^ {(i)}) \leq \frac {N}{N - 1} L _ {\text { soft }} (\boldsymbol {X} ^ {(i)}) - \frac {N}{N - 1} \log N. \tag {18}
$$

Similar to the naive loss $L_{\mathrm{naive}}$ , the design of softmax loss $L_{\mathrm{soft}}$ does not consider the unified threshold among the sample-to-sample pairs.

BCE Loss. For all samples captured from subject i, we assume the existence of a threshold $t_{i}$ , which could separate their all positive sample-to-sample pairs and negative ones, i.e.,

$$
\max \left(\bigcup_ {\boldsymbol {X} ^ {(i)} \in \mathcal {D} _ {i}} \boldsymbol {s} _ {\boldsymbol {X} ^ {(i)}} ^ {(\text { neg })}\right) <   t _ {i} \leq \min \left(\bigcup_ {\boldsymbol {X} ^ {(i)} \in \mathcal {D} _ {i}} \boldsymbol {s} _ {\boldsymbol {X} ^ {(i)}} ^ {(\text { pos })}\right), \quad \forall i, \tag {19}
$$

then, according to Eqs. (16) and (6),

$$
\begin{array}{l} \frac {N}{2} \sum_ {i = 1} ^ {N} L _ {\text { naive }} (\boldsymbol {X} ^ {(i)}) + N ^ {2} \log 2 \\ \leq \sum_ {i = 1} ^ {N} \left[ \log \left(1 + \frac {\exp \left(\sum_ {\substack {j = 1 \\ j \neq i}} ^ {N} \frac {\gamma g \left(\boldsymbol {x} ^ {(i)} , \boldsymbol {x} _ {*} ^ {(j)}\right)}{N - 1}\right)}{\exp \left(\gamma g \left(\boldsymbol {x} ^ {(i)}, \boldsymbol {x} _ {*} ^ {(i)}\right)\right)}\right) + \sum_ {\substack {j = 1 \\ j \neq i}} ^ {N} \log \left(1 + \frac {\exp \left(\gamma g \left(\boldsymbol {x} ^ {(i)} , \boldsymbol {x} _ {*} ^ {(j)}\right)\right)}{\exp \left(\gamma g \left(\boldsymbol {x} ^ {(j)} , \boldsymbol {x} _ {*} ^ {(j)}\right)\right)}\right) \right] (20) \\ \leq \sum_ {i = 1} ^ {N} \left[ \log \left(1 + \mathrm{e} ^ {- \gamma g (\boldsymbol {x} ^ {(i)}, \boldsymbol {x} _ {*} ^ {(i)}) + \gamma t _ {i}}\right) + \sum_ {\substack {j = 1 \\ j \neq i}} ^ {N} \log \left(1 + \mathrm{e} ^ {\gamma g (\boldsymbol {x} ^ {(i)}, \boldsymbol {x} _ {*} ^ {(j)}) - \gamma t _ {j}}\right) \right]. (21) \\ \end{array}
$$

We define BCE loss for sample $X^{(i)}$ as

$$
L _ {\mathrm{bce}} \left(\boldsymbol {X} ^ {(i)}\right) = \log \left(1 + \mathrm{e} ^ {- \gamma g \left(\boldsymbol {x} ^ {(i)}, \boldsymbol {x} _ {*} ^ {(i)}\right) + b _ {i}}\right) + \sum_ {\substack {j = 1 \\ j \neq i}} ^ {N} \log \left(1 + \mathrm{e} ^ {\gamma g \left(\boldsymbol {x} ^ {(i)}, \boldsymbol {x} _ {*} ^ {(j)}\right) - b _ {j}}\right), \tag{22}
$$

where $b_{i} = \gamma t_{i}$ are parameters to be learned. Note that the $t_{i}$ can be different for different identities and therefore are not unified.

After the training of the model M, i.e., the threshold $t_{i} = \frac{b_{i}}{\gamma}$ were learned, then,

$$
\sum_ {i = 1} ^ {N} L _ {\text { naive }} (\boldsymbol {X} ^ {(i)}) \leq \frac {2}{N} \sum_ {i = 1} ^ {N} L _ {\text { bce }} (\boldsymbol {X} ^ {(i)}) - 2 N \log 2. \tag {23}
$$

# 3.3 Marginal Sample-to-Sample Based Losses

Our USS loss, as well as the deduced sample-to-sample based $L_{soft}$ and $L_{bce}$ , only encourage the separability between positive and negative sample pairs. To further improve the discriminative ability of such losses, i.e., to encourage the positive features to be distant from the negative ones, we further proposed the marginal extensions for such losses.

By introducing a margin on the feature similarities, we can have the marginal USS loss as:

$$
L _ {\mathrm{uss-m}} \left(\boldsymbol {X} ^ {(i)}\right) = \log \left(1 + \mathrm{e} ^ {- \gamma \left(g \left(\boldsymbol {x} ^ {(i)}, \boldsymbol {x} _ {*} ^ {(i)}\right) - m\right) + b}\right) + \sum_ {\substack {j = 1 \\ j \neq i}} ^ {N} \log \left(1 + \mathrm{e} ^ {\gamma g \left(\boldsymbol {x} ^ {(i)}, \boldsymbol {x} _ {*} ^ {(j)}\right) - b}\right) \tag{24}
$$

where the m is the introduced hyper-parameter on the margin. Proper adjusting of such a parameter can improve the recognition performance, as discussed in Sec. 4.3. Similarly, we can extend the vanilla $L_{soft}$ and $L_{bce}$ to the marginal version of $L_{soft-m}$ and $L_{bce-m}$ , the full equations are given in the supplementary materials (and appendix).

# 4 Experiments

# 4.1 Datasets and Evaluations

Datasets. We utilize four publicly available datasets for training, namely, CASIA-WebFace[33] (consisting of 0.5 million images of 10K identities), Glint360K[2] (comprising 17.1 million images of 360K identities), WebFace42M[35] (containing 42.5 million images of 2 million identities), and WebFace4M, which is a subset of WebFace42M with 4.2 million images of 0.2 million identities. For evaluating the face verification performance, the ICCV-2021 Masked Face Recognition Challenge (MFR Ongoing)[5] is adopted. The MFR ongoing testing protocol includes various popular benchmarks, such as LFW [11], CFP-FP [23], AgeDB [21], and IJB-C [20], along with its own MFR benchmarks, such as the Mask, Children, and Globalized Multi-Racial (GMR) test sets. The Mask set comprises 13.9K positive pairs and 96.9 million negative pairs (including 6.9K masked images and

13.9K non-masked images) of 6.9K identities. The Children set contains 157K images (totaling 1.7 million positive pairs and 24.7 billion negative pairs) of 14K identities. The Globalized Multi-Racial sets comprise 1.6 million images in total, consisting of 4.6 million positive pairs and 2.6 trillion negative pairs, and representing 242K identities across four races: African, Caucasian, South-Asian, and East-Asian. For face identification, the MegaFace Challenge 1[14] is employed as the test set, which comprises a gallery set with over 1 million images from 690K different identities, and a probe set with 3,530 images from 530 identities. It is worth noting that the MegaFace Challenge 1 also has a verification track, and the verification performance is included in the experiments.

Evaluation and Metrics. For the MFR Ongoing Challenge, the trained models are submitted to and evaluated by the online server. Specifically, we report 1:1 verification accuracy for LFW, CFP-FP, and AgeDB. We report True Accept Rate (TAR) at False Accept Rate (FAR) levels of 1e-4 and 1e-5 for IJB-C. We report TARs at FAR=1e-4 for the Mask and Children test sets, and TARs at FAR=1e-6 for the GMR test sets. For the MegaFace Challenge 1, we report Rank1 accuracy for identification and TAR at FAR=1e-6 for verification.

# 4.2 Implementation Details

Preprocessing. Firstly, we aligned all face images using the 5 landmarks detected by RetinaFace [6] and cropped the center $112 \times 112$ patch. We then normalized the cropped images by first subtracting 127.5 and then dividing 128. Finally, we augmented the training images with horizontal flipping.

Training. We adopt customized ResNets as our backbone following $[7]$ . We implement all models using Pytorch and train them using the SGD optimizer with a weight decay of 5e-4 and momentum of 0.9. For the face models on CASIA-WebFace, we train them over 28 epochs with a batch size of 512. The learning rate starts at 0.1 and is reduced by a factor of 10 at the $16^{th}$ and $24^{th}$ epoch. For both Glint360K and WebFace4M, we train the ResNets for 20 epochs using a batch size of 1024. The learning rate is initially set at 0.1, while a polynomial decay strategy (power=2) is applied for the learning rate schedule. In the case of WebFace42M, we train the ResNets for 20 epochs, using a larger batch size of 4096. The learning rate linearly warms up from 0 to 0.4 during the first epoch, followed by a polynomial decay (power=2) for the remaining 19 epochs. We include the detailed settings of all hyper-parameters used in Sec. 4 and Sec. 5 in the appendix for further reference.

Testing. For a given facial image, we extract two 512-dimensional features from the original image and its horizontally flipped counterpart. These features are then combined together as the final representation with element-wise addition. To assess the similarity between two images, we utilize the cosine similarity metric.

# 4.3 Ablation and Parameter Study

Effectiveness of unified threshold. We first demonstrate the necessity of learning a unified threshold by comparing our proposed $L_{uss}$ with two other sample-to-sample losses, $L_{soft}$ and $L_{bce}$ , in Table 1. We additionally report the performance of the model trained with $L_{naive}$ . The first to third rows respectively show the results from $L_{naive}$ , $L_{soft}$ and $L_{bce}$ , while the fourth row lists the performance of our $L_{uss}$ . We observe that $L_{soft}$ performs 4.68% better than our proposed $L_{uss}$ on IJB-C. However, our USS loss outperforms $L_{soft}$ on the other four datasets, particularly on MFR-All where it achieves a 12.09% higher TAR@FAR=1e-6. Compared to the $L_{bce}$ , the gains achieved by our $L_{uss}$ are more significant, with performance improvements of 29.52% on MFR-All, 33.01% in terms of TAR@FAR=1e-4 on IJB-C, 30.10% on CFP-FP, and 19.69% on AgeDB. The $L_{naive}$ , however, is difficult to train and the trained model barely learns any useful features. We note that the results are not cherry-picked. For a fair comparison, we adopted the same experimental setting when we trained the model using the different losses. During our experiments, we found that the model is easy to converge using $L_{soft}$ and $L_{uss}$ , and the training process is stable, which leads to more favorable results. The model, however, is difficult to train with $L_{naive}$ and $L_{bce}$ . We have dedicated many efforts to these two losses, but the convergence is still problematic.

<table><tr><td>Loss</td><td>MR-ALL</td><td>IJB-C</td><td>LFW</td><td>CFP</td><td>Age</td></tr><tr><td> $L_{\text{naive}}$ </td><td>0.0</td><td>0.35</td><td>50.0</td><td>50.0</td><td>50.0</td></tr><tr><td> $L_{\text{soft}}$ </td><td>26.34</td><td>76.88</td><td>98.80</td><td>95.50</td><td>93.48</td></tr><tr><td> $L_{\text{bce}}$ </td><td>8.91</td><td>39.19</td><td>92.60</td><td>66.41</td><td>74.36</td></tr><tr><td> $L_{\text{uss}}$ </td><td>38.43</td><td>72.20</td><td>99.40</td><td>96.51</td><td>94.05</td></tr><tr><td> $L_{\text{soft-m}}$ </td><td>38.11</td><td>83.60</td><td>99.18</td><td>96.32</td><td>94.18</td></tr><tr><td> $L_{\text{bce-m}}$ </td><td>9.01</td><td>45.08</td><td>92.48</td><td>65.34</td><td>76.30</td></tr><tr><td> $L_{\text{uss-m}}$ </td><td>42.55</td><td>83.92</td><td>99.46</td><td>96.81</td><td>94.28</td></tr></table>

Table 1: Ablation study of the proposed $L_{uss}$ .

When comparing the marginal softmax loss $L_{soft-m}$ , marginal BCE loss $L_{bce-m}$ with marginal USS loss $L_{uss-m}$ (last three rows), we observe that the marginal USS loss $L_{uss-m}$ achieves the highest performance across all five reported datasets, which further confirms the effectiveness of our proposed unified threshold strategy.

It is noteworthy that all the methods in this table are based on the same ResNet-50 network trained on the CASIA-WebFace dataset.

Qualitative study of unified threshold. To better understand the effects of unified threshold, in Fig. 1, we randomly select 20 identities (5,074 facial images) from the whole training dataset with N = 10572 subjects, and then construct 1 positive sample pair and N - 1 negative pairs for each of the selected images. For each of the 20 identities, we compute the optimal threshold to separate the positive pair and the hardest negative pair, i.e., the most similar negative pair.

![](images/7051077d3793993f4df90621a1cb67ca1e5b69c8f5a9c232bfd20f739d551965.jpg)

<details>
<summary>boxplot</summary>

| Group   | Min  | Q1   | Median | Q3   | Max  |
|---------|------|------|--------|------|------|
| L_soft  | 0.44 | 0.46 | 0.48   | 0.50 | 0.54 |
| L_bce   | 0.27 | 0.30 | 0.32   | 0.34 | 0.38 |
| L_uss   | 0.47 | 0.49 | 0.50   | 0.51 | 0.52 |
</details>

Figure 1: Threshold distributions of $L_{soft}$ , $L_{bce}$ , and $L_{uss}$ on 20 random selected identities in CASIA-WebFace. Each dot denotes the optimal threshold for each identity. The dashed blue line represents the threshold that was learned solely by $L_{uss}$ with $t = \frac{b}{\gamma} = 31.3344/64 = 0.4896$ in Eq. (11).

We respectively plot threshold distributions of the 20 identities for $L_{soft}$ , $L_{bce}$ , and $L_{uss}$ . In Fig. 1, each dot denotes the optimal threshold for each identity. The blue dashed line is the unified threshold $t = \frac{b}{\gamma} = 31.3344/64 = 0.4896$ exclusively learned by our $L_{uss}$ in Eq. (11). From this figure, we can clearly observe that our $L_{uss}$ has the most compact threshold distribution, while the threshold distributions from $L_{bce}$ and $L_{soft}$ are relatively loose. Most importantly, for $L_{uss}$ , the median threshold from the randomly sampled 20 identities is in line with the learned unified threshold 0.4896 from the face model trained with $L_{uss}$ , which proves the effects of including an explicit unified threshold.

Impact of Margins. The results in Table 1 (with m = 0.1) demonstrate that a margin improves the performance of $L_{uss}$ . However, the introduced margin m remains a hyperparameter that needs to be tuned. In Table 2, we investigate the effects of using different margins in $L_{uss-m}$ . Specifically, we experiment with four more settings that increase m to 0.2, 0.3, 0.4, and 0.5. We observe that i) the highest performance on MR-ALL, IJB-C, LFT, CFP-FP, and AgeDB are respectively achieved when m equals 0.2, 0.4, 0.1, 0.1, and 0.3; ii) the performance of the four marginal $L_{uss-m}$ losses are all higher than that of the original $L_{uss}$ (i.e., m = 0 in $L_{uss-m}$ ). Although the auxiliary margin improves the overall performance, the performance seems to saturate when m is set to 0.4. Note that in our later experiments, we use m = 0.1.

<table><tr><td>Margin</td><td>MR-ALL</td><td>IJB-C</td><td>LFW</td><td>CFP</td><td>Age</td></tr><tr><td>m=0.0</td><td>38.43</td><td>72.20</td><td>99.40</td><td>96.51</td><td>94.05</td></tr><tr><td>m=0.1</td><td>42.55</td><td>83.92</td><td>99.46</td><td>96.81</td><td>94.28</td></tr><tr><td>m=0.2</td><td>44.74</td><td>87.65</td><td>99.26</td><td>96.65</td><td>94.16</td></tr><tr><td>m=0.3</td><td>44.14</td><td>87.58</td><td>99.30</td><td>96.41</td><td>94.38</td></tr><tr><td>m=0.4</td><td>43.38</td><td>87.82</td><td>99.26</td><td>96.04</td><td>94.23</td></tr><tr><td>m=0.5</td><td>43.91</td><td>87.62</td><td>99.13</td><td>96.05</td><td>94.23</td></tr></table>

Table 2: Parameter study of margin in $L_{uss-m}$ .

<table><tr><td></td><td>MR-ALL</td><td>IJB-C</td><td>LFW</td><td>CFP</td><td>Age</td></tr><tr><td>USS</td><td>42.55</td><td>83.92</td><td>99.46</td><td>96.81</td><td>94.28</td></tr><tr><td>ArcFace</td><td>42.21</td><td>48.49</td><td>99.31</td><td>97.07</td><td>94.51</td></tr><tr><td>ArcFace+USS</td><td>48.92</td><td>89.56</td><td>99.40</td><td>97.22</td><td>95.20</td></tr><tr><td>CosFace</td><td>45.12</td><td>56.65</td><td>99.36</td><td>97.30</td><td>94.98</td></tr><tr><td>CosFace+USS</td><td>50.28</td><td>89.84</td><td>99.41</td><td>97.35</td><td>95.13</td></tr></table>

Table 3: Combination of USS and two sample-to-class based methods.

Compatibility. In Table 1 and Table 2, we have shown that the proposed $L_{uss}$ is superior to other sample-to-sample based losses and can be boosted by adding a proper margin. In Table 3, we investigate the effects of combining $L_{uss}$ with two marginal softmax losses, i.e., ArcFace and CosFace. We find that the model trained with the combined loss significantly outperforms the original ArcFace and CosFace models, as well as our USS loss. For example, the model trained with ArcFace + USS respectively outperforms ArcFace and USS with 6.71% and 6.37% gains on MR-ALL, and 41.07% and 5.64% gains on IJB-C. The model trained with CosFace + USS respectively outperforms CosFace and USS with 5.16% and 7.73% gains on MR-ALL, and 33.19% and 5.92% gains on IJB-C. Such significant improvements suggest the compatibility of our USS loss. In this work, we refer to the model trained using a combination of a sample-to-class method and our USS as UniTSFace. For our experiments, we choose to use CosFace as the sample-to-class method.

<table><tr><td>Method</td><td>P</td><td>R</td><td>Iden.</td><td>Veri.</td><td>Method</td><td>P</td><td>R</td><td>Iden.</td><td>Veri.</td></tr><tr><td>Softmax Loss [18]</td><td>S</td><td>✘</td><td>54.85</td><td>65.92</td><td>ArcFace [7]</td><td>L</td><td>✘</td><td>81.03</td><td>96.98</td></tr><tr><td>Triplet Loss [18, 22]</td><td>S</td><td>✘</td><td>64.79</td><td>78.32</td><td>CurricularFace [12]</td><td>L</td><td>✘</td><td>81.26</td><td>97.26</td></tr><tr><td>Contrastive [18, 25]</td><td>S</td><td>✘</td><td>65.21</td><td>78.86</td><td>CosFace [30]</td><td>L</td><td>✘</td><td>82.72</td><td>96.65</td></tr><tr><td>Center [18, 32]</td><td>S</td><td>✘</td><td>65.49</td><td>80.14</td><td>UniTSFace</td><td>L</td><td>✘</td><td>85.01</td><td>97.85</td></tr><tr><td>L-Softmax [18, 19]</td><td>S</td><td>✘</td><td>67.12</td><td>80.42</td><td>SphereFace2 [31]</td><td>L</td><td>✓</td><td>89.84</td><td>91.94</td></tr><tr><td>SphereFace [18]</td><td>S</td><td>✘</td><td>72.72</td><td>85.56</td><td>CosFace [7, 30]</td><td>L</td><td>✓</td><td>97.91</td><td>97.91</td></tr><tr><td>SphereFace+ [16]</td><td>S</td><td>✘</td><td>73.03</td><td>-</td><td>SphereFace [18]</td><td>L</td><td>✓</td><td>98.16</td><td>98.46</td></tr><tr><td>CosFace [30]</td><td>S</td><td>✘</td><td>77.11</td><td>89.88</td><td>ArcFace [7]</td><td>L</td><td>✓</td><td>98.35</td><td>98.48</td></tr><tr><td>ArcFace [7]</td><td>S</td><td>✘</td><td>77.50</td><td>92.34</td><td>Circle Loss [26]</td><td>L</td><td>✓</td><td>98.50</td><td>98.73</td></tr><tr><td>CurricularFace [12]</td><td>S</td><td>✘</td><td>77.65</td><td>92.91</td><td>CurricularFace [12]</td><td>L</td><td>✓</td><td>98.71</td><td>98.64</td></tr><tr><td>UniTSFace</td><td>S</td><td>✘</td><td>77.41</td><td>93.50</td><td>VPL [8]</td><td>L</td><td>✓</td><td>98.80</td><td>98.97</td></tr><tr><td>ArcFace [7]</td><td>S</td><td>✓</td><td>91.75</td><td>93.69</td><td>Partial FC [2]</td><td>L</td><td>✓</td><td>98.94</td><td>99.10</td></tr><tr><td>CurricularFace [12]</td><td>S</td><td>✓</td><td>92.48</td><td>94.55</td><td>UNPG [13]</td><td>L</td><td>✓</td><td>99.27</td><td>-</td></tr><tr><td>UniTSFace</td><td>S</td><td>✓</td><td>92.36</td><td>94.84</td><td>UniTSFace</td><td>L</td><td>✓</td><td>99.27</td><td>99.19</td></tr></table>

Table 4: Comparisons between different methods on the MegaFace Challenge 1. The letter P indicates the ‘Small’ or ‘Large’ protocols, R denotes whether the label refinement is used. All reported results, with the exception of our own, were directly taken from their respective papers.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Net. Data.</td><td colspan="7">MFR</td><td colspan="2">IJB-C</td><td colspan="3">Verification Acc.</td></tr><tr><td>Mask</td><td>Child.</td><td>Afri.</td><td>Cau.</td><td>S-A.</td><td>E-A.</td><td>MR-All</td><td>1e-4</td><td>1e-5</td><td>LFW</td><td>CFP</td><td>Age</td></tr><tr><td>Contrastive[4]</td><td></td><td>6.67</td><td>10.58</td><td>12.40</td><td>18.84</td><td>13.57</td><td>10.38</td><td>12.38</td><td>58.47</td><td>46.75</td><td>95.50</td><td>74.65</td><td>82.28</td></tr><tr><td>(N+1)-Tuplet [24]</td><td></td><td>26.56</td><td>28.23</td><td>39.16</td><td>50.92</td><td>47.71</td><td>24.06</td><td>38.11</td><td>83.60</td><td>73.93</td><td>99.18</td><td>96.32</td><td>94.18</td></tr><tr><td>ArcFace[7]</td><td></td><td>38.52</td><td>31.42</td><td>45.87</td><td>63.69</td><td>59.85</td><td>7.66</td><td>42.21</td><td>48.49</td><td>9.18</td><td>99.31</td><td>97.07</td><td>94.51</td></tr><tr><td>CosFace[30]</td><td></td><td>38.79</td><td>31.33</td><td>48.06</td><td>63.56</td><td>58.71</td><td>15.08</td><td>45.12</td><td>56.65</td><td>11.30</td><td>99.36</td><td>97.30</td><td>94.98</td></tr><tr><td>Sphere-Rv1[17]</td><td>R50</td><td>32.80</td><td>28.09</td><td>40.24</td><td>57.24</td><td>50.38</td><td>22.30</td><td>39.92</td><td>86.35</td><td>75.81</td><td>99.38</td><td>96.95</td><td>94.48</td></tr><tr><td>SphereFace2[31]</td><td>CASIA</td><td>35.40</td><td>30.55</td><td>46.65</td><td>62.69</td><td>56.23</td><td>26.65</td><td>44.20</td><td>88.41</td><td>79.18</td><td>99.46</td><td>97.42</td><td>94.96</td></tr><tr><td>VPL[8]</td><td></td><td>33.86</td><td>31.39</td><td>46.52</td><td>59.93</td><td>54.07</td><td>27.18</td><td>47.02</td><td>88.44</td><td>81.38</td><td>99.30</td><td>97.07</td><td>94.75</td></tr><tr><td>AnchorFace[15]</td><td></td><td>37.04</td><td>32.28</td><td>49.60</td><td>63.17</td><td>59.80</td><td>28.88</td><td>48.44</td><td>88.81</td><td>77.82</td><td>99.56</td><td>97.48</td><td>95.18</td></tr><tr><td>UNPG[13]</td><td></td><td>38.62</td><td>33.24</td><td>49.94</td><td>63.85</td><td>59.60</td><td>29.21</td><td>48.66</td><td>88.17</td><td>77.73</td><td>99.45</td><td>97.25</td><td>94.83</td></tr><tr><td>UniTSFace</td><td></td><td>37.98</td><td>31.73</td><td>51.45</td><td>64.89</td><td>59.73</td><td>29.56</td><td>50.28</td><td>89.84</td><td>82.64</td><td>99.41</td><td>97.35</td><td>95.13</td></tr><tr><td>Partial FC [1]</td><td>R50</td><td>72.28</td><td>-</td><td>84.86</td><td>91.57</td><td>88.57</td><td>67.52</td><td>86.85</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>UniTSFace</td><td>WF4M</td><td>75.93</td><td>72.00</td><td>88.17</td><td>93.68</td><td>91.40</td><td>70.55</td><td>89.65</td><td>97.03</td><td>95.18</td><td>99.80</td><td>99.04</td><td>97.93</td></tr><tr><td>Partial FC [1]</td><td>R200</td><td>91.87</td><td>-</td><td>97.79</td><td>98.70</td><td>98.54</td><td>89.52</td><td>97.70</td><td>97.97</td><td>96.93</td><td>99.83</td><td>99.51</td><td>98.70</td></tr><tr><td>UniTSFace</td><td>WF42M</td><td>92.87</td><td>93.51</td><td>98.35</td><td>99.03</td><td>98.99</td><td>90.76</td><td>98.16</td><td>97.99</td><td>97.00</td><td>99.83</td><td>99.47</td><td>98.71</td></tr></table>

Table 5: Comparisons between different methods on MFR-Ongoing.

# 5 Comparison with the State-Of-The-Art

MegaFace Challenge 1. In Table 4, we compare both the identification and verification performance of our UniTSFace with several state-of-the-art methods on MegaFace Challenge 1. These methods include sample-to-sample based Contrastive Loss [4, 10] and Triplet Loss [22], sample-to-class based CosFace[30] and ArcFace[7], as well as the hybrid/combined approaches: VPL[8] and UNPG[13].

To ensure a fair comparison, as per the official protocols, we compare our UniTSFace trained on the CASIA-WebFace with the models trained on 'Small' datasets, while UniTSFace trained on Glint360K is compared with the models trained on 'Large' datasets in Table 4. When label refinement [7] is not used, our UniTSFace achieves the highest accuracy among the compared models trained on 'Large' datasets, with an $85.01\%$ identification accuracy and a $97.85\%$ verification accuracy. When label refinement [7] is used, the accuracies can be further increased to $99.27\%$ and $99.19\%$ respectively. Though our UniTSFace is slightly lower than CurricularFace [12] in terms of identification performance on 'Small' datasets, UniTSFace achieves the highest accuracy on the verification track, regardless of whether label refinement is used or not. These results demonstrate the effectiveness of our UniTSFace in both identification and verification tasks.

MFR Ongoing Benchmarks. We then compare the proposed UniTSFace with Contrastive Loss[4, 10], $(\mathrm{N} + 1)$ -Tuplet Loss[24], CosFace[30], ArcFace[7], VPL[8], AnchorFace[15], and UNPG[13] on the MFR Ongoing benchmark. We re-implement these methods with the optimal hyper-parameters recommended in their original papers. All the compared models are trained with a ResNet-50 backbone and the CASIA-WebFace dataset. As reported in Table 5, our UniTSFace achieves clear improvement over other methods.

Additionally, we compare our UniTSFace with the recent Partial FC[1], which is the leading method on the MFR ongoing challenge. Following the settings of Partial FC[1], we train the proposed UniTSFace on the WebFace4M and WebFace42M datasets using two different architectures, i.e., ResNet-50 and ResNet-200. We can observe that, on average, our performance is better than that of the Partial FC. Till the submission of this work (May 17 '23), the proposed UniTSFace ranks first place on the academic track of the MFR-ongoing leaderboard.

# 6 Discussion and Conclusion

# Discussion.

1) Though the threshold range of USS is narrowed compared to other losses in Fig. 1, is it correct to claim the word “unified”? Firstly, our theoretical objective is to learn a unified threshold that satisfies Eq.(3) during the training of a facial model. Therefore, we first assume the existence of such a unified threshold, and then propose the unified threshold integrated sample-to-sample (USS) loss. We have proven through our analysis in Section 3.1 that, ideally, a model trained by USS could learn a unified threshold for the training dataset.

However, we must admit that achieving this ideal goal is subjective to the model capacity, training hyper-parameters, and even the training dataset itself, which are all independent from our USS loss. For example, if the backbone network only uses one single linear neural layer, our USS loss definitely cannot guarantee a unified threshold either. In Fig. 1, using the same backbone architecture and training hyper-parameters, i) our USS loss is able to achieve a more compact threshold distribution than the other losses, moreover, and ii) the learned threshold denoted by the blue dashed line lies around the median of the boxplot, these two observations suggest the superiority of imposing the unified threshold and are consistent with our expectations.

2) In the testing stage, how to determine the threshold? The threshold learned by the training stage cannot be directly used in testing. In the testing stage, the threshold is determined according to the specific testing criteria. For example, when reporting the 1:1 verification accuracy on LFW, CFP-FP, AgeDB, 10-fold validation is used. We first select the threshold that achieves the highest accuracy in the first 9 folds and then adopt this threshold to calculate the accuracy in the leave-out fold.

3) How does UniTSFace compare to other methods such as CosFace in terms of computational efficiency and memory usage? UniTSFace utilizes the ResNet architecture as its backbone and optimizes the parameters using the algorithmic average of the cosine-margin Softmax loss and the proposed USS loss. When compared to CosFace, the extra computational cost brought by USS loss is relatively small, as the computational consumption and memory usage mostly depend on the convolutional operations inherent to the selected network architecture and the resolutions of the input images. In experiments, we indeed found that the computational differences between these methods during both the testing and training stages are negligible.

# Conclusion.

We propose the USS loss for deep face recognition by explicitly defining a unified threshold that separates all positive sample-to-sample pairs from negative ones. Though this unified threshold learned in the training stage cannot be directly applied to the testing stage, the model trained by USS is desired to extract more discriminative features and subsequently improve the face recognition performance in various testing scenarios, which have been demonstrated with extensive experiments.

Furthermore, the proposed USS loss can be effortlessly extended to the Marginal USS loss and can also be seamlessly combined with other sample-to-class losses. In our experiments, we have combined the USS loss with ArcFace and CosFace methods, and the combined approaches consistently surpass their respective individual counterparts. We denote the fusion of the CosFace and USS losses as “UniTSFace” which we have compared with other sophisticated combinations such as VPL, UNPG, and AnchorFace. The experimental results on multiple benchmark datasets further suggest the superiority of our UniTSFace.

In conclusion, we believe our USS loss provides the research community with a much more versatile and effective solution for face recognition tasks.

# Acknowledgement

This work was supported by the National Natural Science Foundation of China under Grants 82261138629 and 62006156, Guangdong Basic and Applied Basic Research Foundation under Grants 2023A1515010688 and 2022A1515012125, and Shenzhen Municipal Science and Technology Innovation Council under Grant JCYJ20220531101412030.

# References

[1] Xiang An, Jiankang Deng, Jia Guo, Ziyong Feng, XuHan Zhu, Jing Yang, and Tongliang Liu. Killing two birds with one stone: Efficient and robust training of face recognition cnns by partial fc. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4042–4051, 2022.   
[2] Xiang An, Xuhan Zhu, Yuan Gao, Yang Xiao, Yongle Zhao, Ziyong Feng, Lan Wu, Bin Qin, Ming Zhang, Debing Zhang, et al. Partial fc: Training 10 million identities on a single machine. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1445–1449, 2021.   
[3] Weihua Chen, Xiaotang Chen, Jianguo Zhang, and Kaiqi Huang. Beyond triplet loss: a deep quadruplet network for person re-identification. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 403–412, 2017.   
[4] Sumit Chopra, Raia Hadsell, and Yann LeCun. Learning a similarity metric discriminatively, with application to face verification. In 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05), volume 1, pages 539–546. IEEE, 2005.   
[5] Jiankang Deng, Jia Guo, Xiang An, Zheng Zhu, and Stefanos Zafeiriou. Masked face recognition challenge: The insightface track report. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1437–1444, 2021.   
[6] Jiankang Deng, Jia Guo, Evangelos Ververas, Irene Kotsia, and Stefanos Zafeiriou. Retinaface: Single-shot multi-level face localisation in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5203–5212, 2020.   
[7] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4690–4699, 2019.   
[8] Jiankang Deng, Jia Guo, Jing Yang, Alexandros Lattas, and Stefanos Zafeiriou. Variational prototype learning for deep face recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11906–11915, 2021.   
[9] Yueqi Duan, Jiwen Lu, and Jie Zhou. Uniformface: Learning deep equidistributed representation for face recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.   
[10] Raia Hadsell, Sumit Chopra, and Yann LeCun. Dimensionality reduction by learning an invariant mapping. In 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06), volume 2, pages 1735–1742. IEEE, 2006.   
[11] Gary B Huang, Marwan Mattar, Tamara Berg, and Eric Learned-Miller. Labeled faces in the wild: A database for studying face recognition in unconstrained environments. In Workshop on faces in 'Real-Life' Images: detection, alignment, and recognition, 2008.   
[12] Yuge Huang, Yuhan Wang, Ying Tai, Xiaoming Liu, Pengcheng Shen, Shaoxin Li, Jilin Li, and Feiyue Huang. Curricularface: adaptive curriculum learning loss for deep face recognition. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5901–5910, 2020.   
[13] Junuk Jung, Seonhoon Lee, Heung-Seon Oh, Yongjun Park, Joochan Park, and Sungbin Son. Unified negative pair generation toward well-discriminative feature space for face recognition. arXiv preprint arXiv:2203.11593, 2022.   
[14] Ira Kemelmacher-Shlizerman, Steven M Seitz, Daniel Miller, and Evan Brossard. The megaface benchmark: 1 million faces for recognition at scale. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4873–4882, 2016.   
[15] Jiaheng Liu, Haoyu Qin, Yichao Wu, and Ding Liang. Anchorface: Boosting tar@ far for practical face recognition. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 1711–1719, 2022.   
[16] Weiyang Liu, Rongmei Lin, Zhen Liu, Lixin Liu, Zhiding Yu, Bo Dai, and Le Song. Learning towards minimum hyperspherical energy. Advances in neural information processing systems, 31, 2018.

[17] Weiyang Liu, Yandong Wen, Bhiksha Raj, Rita Singh, and Adrian Weller. Sphereface revived: Unifying hyperspherical face recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022.   
[18] Weiyang Liu, Yandong Wen, Zhiding Yu, Ming Li, Bhiksha Raj, and Le Song. Sphereface: Deep hypersphere embedding for face recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 212–220, 2017.   
[19] Weiyang Liu, Yandong Wen, Zhiding Yu, and Meng Yang. Large-margin softmax loss for convolutional neural networks. In ICML, volume 2, page 7, 2016.   
[20] Brianna Maze, Jocelyn Adams, James A Duncan, Nathan Kalka, Tim Miller, Charles Otto, Anil K Jain, W Tyler Niggel, Janet Anderson, Jordan Cheney, et al. Iarpa janus benchmark-c: Face dataset and protocol. In 2018 international conference on biometrics (ICB), pages 158–165. IEEE, 2018.   
[21] Stylianos Moschoglou, Athanasios Papaioannou, Christos Sagonas, Jiankang Deng, Irene Kotsia, and Stefanos Zafeiriou. Agedb: the first manually collected, in-the-wild age database. In proceedings of the IEEE conference on computer vision and pattern recognition workshops, pages 51–59, 2017.   
[22] Florian Schroff, Dmitry Kalenichenko, and James Philbin. Facenet: A unified embedding for face recognition and clustering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 815–823, 2015.   
[23] Soumyadip Sengupta, Jun-Cheng Chen, Carlos Castillo, Vishal M Patel, Rama Chellappa, and David W Jacobs. Frontal to profile face verification in the wild. In 2016 IEEE winter conference on applications of computer vision (WACV), pages 1–9. IEEE, 2016.   
[24] Kihyuk Sohn. Improved deep metric learning with multi-class n-pair loss objective. Advances in neural information processing systems, 29, 2016.   
[25] Yi Sun, Yuheng Chen, Xiaogang Wang, and Xiaoou Tang. Deep learning face representation by joint identification-verification. Advances in neural information processing systems, 27, 2014.   
[26] Yifan Sun, Changmao Cheng, Yuhan Zhang, Chi Zhang, Liang Zheng, Zhongdao Wang, and Yichen Wei. Circle loss: A unified perspective of pair similarity optimization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6398–6407, 2020.   
[27] Yi Sun, Xiaogang Wang, and Xiaoou Tang. Deep learning face representation from predicting 10,000 classes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1891–1898, 2014.   
[28] Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, and Lior Wolf. Deepface: Closing the gap to human-level performance in face verification. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1701–1708, 2014.   
[29] Feng Wang, Xiang Xiang, Jian Cheng, and Alan Loddon Yuille. Normface: L2 hypersphere embedding for face verification. In Proceedings of the 25th ACM international conference on Multimedia, pages 1041–1049, 2017.   
[30] Hao Wang, Yitong Wang, Zheng Zhou, Xing Ji, Dihong Gong, Jingchao Zhou, Zhifeng Li, and Wei Liu. Cosface: Large margin cosine loss for deep face recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5265–5274, 2018.   
[31] Yandong Wen, Weiyang Liu, Adrian Weller, Bhiksha Raj, and Rita Singh. Sphereface2: Binary classification is all you need for deep face recognition. In International Conference on Learning Representations, 2022.   
[32] Yandong Wen, Kaipeng Zhang, Zhifeng Li, and Yu Qiao. A discriminative feature learning approach for deep face recognition. In European conference on computer vision, pages 499–515. Springer, 2016.   
[33] Dong Yi, Zhen Lei, Shengcai Liao, and Stan Z Li. Learning face representation from scratch. arXiv preprint arXiv:1411.7923, 2014.   
[34] Kai Zhao, Jingyi Xu, and Ming-Ming Cheng. Regularface: Deep face recognition via exclusive regularization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019.   
[35] Zheng Zhu, Guan Huang, Jiankang Deng, Yun Ye, Junjie Huang, Xinze Chen, Jiagang Zhu, Tian Yang, Jiwen Lu, Dalong Du, et al. Webface260m: A benchmark unveiling the power of million-scale deep face recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10492–10502, 2021.