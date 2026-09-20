# ENHANCING TAIL PERFORMANCE IN EXTREME CLASSIFIERS BY LABEL VARIANCE REDUCTION

Anirudh Buvanesh\*, Rahul Chand\*, Jatin Prakash, Bhawna Paliwal, Mudit Dhawan Neelabh Madan, Deepesh Hada, Vidit Jain, Sonu Mehta, Yashoteja Prabhu Manish Gupta, Ramachandran Ramjee, Manik Varma

Microsoft

{t-abuvanesh, t-rahulchand, t-japrakash, bhawna, t-mdhawan
t-nmadan, deepeshhada, jainvidit, sonu.mehta, yprabhu
gmanish, ramjee, manik}@microsoft.com

# ABSTRACT

Extreme Classification (XC) architectures, which utilize a massive One-vs-All (OvA) classifier layer at the output, have demonstrated remarkable performance on problems with large label sets. Nonetheless, these architectures falter on tail labels with few representative samples. This phenomenon has been attributed to factors such as classifier over-fitting and missing label bias, and solutions involving regularization and loss re-calibration have been developed. This paper explores the impact of label variance - a previously unexamined factor - on the tail performance in extreme classifiers. It also develops a method to systematically reduce label variance in XC by transferring the knowledge from a specialized tail-robust teacher model to the OvA classifiers. For this purpose, it proposes a principled knowledge distillation framework, LEVER, which enhances the tail performance in extreme classifiers with formal guarantees on generalization. Comprehensive experiments are conducted on a diverse set of XC datasets, demonstrating that LEVER can enhance tail performance by around 5% and 6% points in PSP and coverage metrics, respectively, when integrated with leading extreme classifiers. Moreover, it establishes a new state-of-the-art when added to the top-performing Renée classifier. Extensive ablations and analyses substantiate the efficacy of our design choices. Another significant contribution is the release of two new XC datasets that are different from and more challenging than the available benchmark datasets, thereby encouraging more rigorous algorithmic evaluation in the future. Code for LEVER is available at: aka.ms/lever.

# 1 INTRODUCTION

Extreme Classification (XC) addresses tasks where a data point is mapped to the most relevant subset of labels from a large label space. Deep architectures that comprise a neural network encoder followed by a massive One-vs-All (OvA) classification layer at the output have become the de-facto standard for contemporary XC algorithms and have demonstrated remarkable results on several large-scale applications (Agrawal et al., 2013; Yadav et al., 2021; Chang et al., 2020; Beygelzimer et al., 2009; Babbar & Schölkopf, 2017). Despite this progress, such over-parameterized OvA classification layers has also been known to overfit and underperform on labels with limited representative samples, also known as the tail labels (Wei et al., 2021). As a result, bulk of such tail labels, which often provide niche and highly informative results for a test sample (Jain et al., 2016), are incorrectly classified thus diminishing their aggregate utility for a practical application.

The challenge of enhancing the tail performance of extreme OvA classifiers has been the focus of some recent studies. These investigations have identified multiple factors that contribute to the hardness of tail labels and proposed solutions to alleviate them. Some works have addressed the

concern of overfitting to data-scarce tail labels by constraining the capacity of tail classifiers through regularization tricks (Guo et al., 2019). A separate line of work has studied the effects of false negatives, also known as missing labels, on the tail performance and proposed to appropriately amend the classifier training loss through propensity-scoring techniques (Qaraei et al., 2021).

This paper brings to light another important yet previously unexamined factor behind the underperformance of tail OvA classifiers, namely label variance (Sec. 3). Typically, the ground truth of an XC dataset is constructed by approximating a complex label distribution that arises in a source application with a discrete sample of labels. For example, in a recommendation task, the ground truth is defined as the set of items clicked by each user within a specified time period. However, in general, the ground truth can vary from one data sampling period to another, as a user's interests can fluctuate with time. Similarly, in expert annotation based data, employing fewer experts to reduce annotation costs can introduce variance in the ground truth (aka. label variance) owing to inter-annotator disagreements. Large label variance is particularly harmful for the tail classifiers' performance as they have to rely on sparse ground truth and the approximation errors can have a drastically magnified effects with low sample counts.

In a recent work, Menon et al. (2021a) studied the problem of label variance in the context of multiclass classification and retrieval, and further note that a teacher-to-student knowledge distillation strategy can be used to improve the generalization performance of the student model. This paper borrows the basic ideas from Menon et al. (2021a) and extends them to the more challenging Extreme Classification setting through several key innovations. First, it theoretically formalizes the performance degradation in OvA classifiers owing to label variance, specifically quantifying the magnified effect on the tail classifiers. Second, whereas Menon et al. (2021a) assumes the pre-existence of a teacher, this paper learns its own Siamese-style teacher model that is optimized for tail performance, and further develops a principled knowledge distillation strategy to effectively teach the downstream OvA classifiers. The resulting approach, LEVER, is demonstrated to improve tail classifiers' performance by around 5% and 6% points in terms of PSP and coverage metrics, which also advances the state-of-the-art in XC.

Another independent contribution of this paper is the public release of two new datasets for algorithmic benchmarking in XC. Traditionally, performance in XC is mostly assessed on the public datasets available from (Bhatia et al., 2016). These datasets appear to share a common property that the data points associated with a label are fairly similar to each other in their semantic intents, making these datasets less challenging to learn. In contrast, the real-world applications of XC can be more diverse in their properties and complexity. To encourage more rigorous algorithmic evaluation, the new datasets are constructed with the property that a label can be associated with data points of vastly different intents. These datasets, termed as multi-intent datasets, are inspired by real applications, are more challenging, and can unlock exciting research problems in the future.

This paper makes the following key contributions: 1. Identifies the problem of label variance which adversely affects the performance of tail classifiers in XC. 2. Proposes a principled LEVER approach to mitigate the label variance effects on tail classifiers in XC (Sec. 3.2). 3. Develops an effective Siamese-style model as a tail teacher with LEVER (Sec. 3.3). 4. Conducts extensive experimentation using multiple state-of-the-art baselines and diverse benchmarks to demonstrate the utility and generality of the proposed approach (Sec. 5). 5. Releases two new multi-intent datasets for robust experimentation in XC (Sec. 4).

# 2 RELATED WORK

# 2.1 EXTREME CLASSIFICATION

Recent advancements in XC have leveraged deep network-based representations like LSTM (You et al., 2018), Transformer (Zhang et al., 2021; Jiang et al., 2021) or customized architectures (Dahiya et al., 2021b) to generate rich semantic representations of inputs. These are then assigned to appropriate labels via an OvA classifier layer. To facilitate efficient learning with large label sets, techniques such as multi-staged encoder refinement (Dahiya et al., 2021a; Zhang et al., 2021; Jiang et al., 2021), hierarchical label search, and hard-negative sampling (Dahiya et al., 2023a; 2021b; Zhang et al., 2021; Jiang et al., 2021; Mittal et al., 2021a) have been introduced. Furthermore, simultaneous training of the deep encoder and OvA classifiers has been demonstrated to boost per-

formance in leading XC approaches like DEXA (Dahiya et al., 2023b), ELIAS (Gupta et al., 2022), CascadeXML (Kharbanda et al., 2022) and Renée (Jain et al., 2023). However, despite these advancements, many of these approaches share a common limitation: a decline in performance for tail labels, which is the primary focus of this paper.

# 2.2 ENHANCING TAIL PERFORMANCE IN XC

Extreme classifiers have been observed to under-perform on tail labels with limited representative samples. This phenomenon has been attributed to various factors, and several approaches have been proposed to address them.

Over-fitting of OvA Classifiers: OvA classifiers, which employ a distinct classifier for each label, are massively parameterized in scenarios with large label sets. Consequently, they are susceptible to overfitting on tail labels with scarce representative samples. In response, various classifier regularization techniques have been introduced. For instance, ProXML (Babbar & Schölkopf, 2019) employs an L1-regularizer, and GLaS (Guo et al., 2019) uses a label-decorrelation based regularizer.

Bias due to Missing Labels: In XC datasets, which are often too large for exhaustive labeling, missing or false negative labels are a frequent issue. These missing labels introduce systematic biases into the ground truth and are known to significantly impact tail labels. Strategies to address tail labels typically involve estimating the missing propensities for labels first and then recalibrating the loss through simple weighting (Jain et al., 2016; Wei et al., 2021; Wydmuch et al., 2021; Schultheis et al., 2022). The phenomenon of missing label bias is distinct from that of label variance.

Data Scarcity in Tail Labels: XC datasets contain tail labels with a limited number of positive data samples. To mitigate this scarcity, data augmentation techniques like TAUG (Wei et al., 2021) and Gandalf (Kharbanda et al., 2024) have been proposed. However, these methods lack formal guarantees and do not perform consistently across different datasets as shown in this paper (Table 2). Another line of work leverages label-side features to improve the tail label prediction performance (Xiong et al., 2020; Dahiya et al., 2021a; 2023a; Jain et al., 2023). Approaches like NGAME (Dahiya et al., 2023a) share information between semantically similar labels by placing them close to each other in a dense embedding space using a Siamese encoder. However, these methods primarily focus on enhancing encoder robustness and do not explicitly address the quality of subsequent OvA classifiers. Our proposed model shares similarities with these approaches through its use of a Siamese teacher but distinguishes itself by learning a specialized teacher model suitable for distillation and developing a principled approach to improve tail OvA classifiers.

In addition to these known issues, this paper introduces label variance as an additional, but important, consideration pertaining to tail performance in XC. A closely related work is the study around uncertainty quantification in extreme classification (Jiang et al., 2023) because variance can intrinsically be viewed as an uncertainty measurement. But in this work, we attempt to mitigate variance rather than just estimate it. It is important to differentiate the label variance discussed here from the variance described in (Babbar & Schölkopf, 2019). The latter addresses variance from the perspective of lack of commonality between the features of train and test instances. In contrast, our focus on label variance pertains to inaccuracies in the ground truth relevance scores.

# 3 LEVER: LABEL VARIANCE REDUCTION IN EXTREME CLASSIFICATION

Label variance is a measure of approximation errors introduced in the ground truth of a dataset due to the discrete data sampling process. These errors can negatively impact the performance of trained classifiers, particularly those on the tail. This section introduces LEVER, a principled approach based on knowledge distillation designed to alleviate label variance and enhance the generalization capabilities of One-vs-All (OvA) classifiers. An effective teacher model for distillation based on a Siamese-style encoder is also proposed.

# 3.1 PRELIMINARIES

Extreme Classification (XC) maps a data point space $\mathcal{X}$ onto a label space represented as $\mathcal{Y} = \{0,1\}^L$ , where $L$ is the number of labels, potentially reaching into the millions. A deep extreme classification architecture typically includes a deep encoder $\mathcal{E}_{\theta}$ which generates a semantically rich

representation $\mathcal{E}_{\theta}(\mathbf{x})$ for any given input data point $x \in X$ . This is followed by a One-vs-All classifier layer $\{w_{l}\}_{l=1}^{L}$ which sorts the labels based on $\mathbf{w}_{l}^{\top}\mathcal{E}_{\theta}(\mathbf{x})$ scores and predicts the highest scoring labels as the most relevant ones for x.

Different strategies have been employed for training such a deep architecture including stagewise training where encoder and classifiers are optimized in two successive stages, and end-to-end training where both are optimized jointly. For this paper, we assume a stagewise training schedule. Furthermore, the focus will be primarily on the second stage of OvA classifier training during which encoder is assumed to be already trained and held fixed. As a result, each OvA classifier is trained independently of others which also simplifies the theoretical analysis. For brevity, we drop the encoder symbol $E_{\theta}$ and directly use x to refer to a data point's embedding from the encoder over which OvA classifiers are applied.

For a data point x, let $\mathbb{P}(Y(\mathbf{x})=\mathbf{y}|\mathbf{x}))\quad\forall\mathbf{y}\in\{0,1\}^{L}$ represent the true and complete distribution of label relevance which accurately captures the stochasticities inherent in the user preferences or annotator judgments. Note that this distribution sums up to 1 over all label subsets. Unfortunately, the full relevance distribution is seldom available and is instead approximated with a discrete sample of labels $y\sim\mathbb{P}(Y(\mathbf{x})=y|\mathbf{x})$ . The approximation error due to this sampling is captured by the following expression for label variance:

$$
\mathbb {V} _ {\mathbf {y} | \mathbf {x}} [ \mathbf {y} ] = \mathbb {E} _ {\mathbf {y} | \mathbf {x}} [ \mathbf {y} - \mathbb {E} [ \mathbf {y} ] ] ^ {2}
$$

$$
\mathbb {V} _ {y _ {l} | \mathbf {x}} [ y _ {l} ] = \mathbb {E} _ {y _ {l} | \mathbf {x}} [ y _ {l} - \mathbb {E} [ y _ {l} ] ] ^ {2} = \mathbb {P} (y _ {l} = 1 | \mathbf {x}) (1 - \mathbb {P} (y _ {l} = 1 | \mathbf {x})) \tag {1}
$$

The second expression denotes the variance in the marginal relevance of a label $l$ to point $\mathbf{x}$ , a term that is particularly useful in analyzing One-vs-All classifiers. A larger variance indicates that the imprecision in a sampled label is more.

To train the classifier for label l, we first construct a training set denoted as $D = \{x_{i}, y_{il}\}_{i=1}^{N}$ and solve a binary classification problem with $y_{il}$ as the target label for $x_{i}$ . For simplicity, we present the analysis for a single classifier, with the understanding that the same holds for all classifiers. To avoid confusion, we omit subscript l where it is not necessary. The binary classification objective minimizes the following empirical risk of classification:

$$
\hat {\mathbf {R}} = \min _ {\mathbf {w}} \frac {1}{N} \sum_ {i = 1} ^ {N} \mathcal {L} (y _ {i}, \mathbf {w} ^ {\top} \mathbf {x} _ {i})
$$

$$
\text { with, } \mathcal {L} (y, \mathbf {w} ^ {\top} \mathbf {x}) = C y f (1, \mathbf {w} ^ {\top} \mathbf {x}) + (1 - y) f (0, \mathbf {w} ^ {\top} \mathbf {x}) \tag {2}
$$

Here, f represents a convex classification surrogate such as hinge loss or logistic loss (Qaraei et al., 2021). Using a weight factor C > 1 is standard practice in imbalanced classification to appropriately balance the relative importance of positive and negative samples for a label. This is particularly important for a tail label with a few positives, denoted by number S where:

$$
\mathbb {E} _ {\mathbf {x}} [ p _ {x} ] \approx \frac {S}{N} \ll 1 \text {   where,   } p _ {x} = \mathbb {P} (y = 1 | \mathbf {x}) \tag {3}
$$

Following the standard practice (Kakade et al., 2008), we assume that the norms of the weight vector w and the input vector x are bounded by $\|w\|\leq W$ and $\|x\|\leq B$ respectively. Additionally, we assume that the function f exhibits Lipschitz continuity with a Lipschitz constant L.

The generalization performance of a trained classifier w is evaluated by its true population risk. A lower value of this risk indicates superior predictive capability:

$$
\mathbf {R} = \mathbb {E} _ {\mathbf {x}, y} [ \mathcal {L} (y, \mathbf {w} ^ {\top} \mathbf {x}) ] \tag {4}
$$

# 3.2 LEVER FRAMEWORK

The deviation between empirical and true risks formally measures a classifier's generalization gap, with smaller values indicating better test-time generalization. Following (Maurer & Pontil, 2009), we express the generalization gap in terms of data-dependent bounds based on label variance. Applying Bennett's inequality, as suggested in the reference, with simplifications relevant to the problem at hand, provides us with the following result. Note that all the proofs are available from the supplementary Sec. A.

Theorem 1. Let $\mathcal{M}_N$ be the uniform covering number (Menon et al., 2021a) corresponding to the classification loss $\mathcal{L}$ . Then, given the definitions established earlier, For any $\delta \in (0,1)$ , with probability at least $1 - \delta$ over sampling the data points $\{\mathbf{x}\}_{i=1}^N$ ,

$$
\mathbf {R} \leq \hat {\mathbf {R}} + \mathcal {O} \left(\sqrt {\mathbb {V} _ {\mathbf {x}} \left[ \mathcal {L} \left(p _ {x} , \mathbf {w} ^ {\top} \mathbf {x}\right) \right] + \mathbb {E} _ {\mathbf {x}} \left[ \mathbb {V} _ {y | \mathbf {x}} [ y | \mathbf {x} ] \right] (C L W B) ^ {2}} \sqrt {\frac {\log \left(\mathcal {M} _ {N} / \delta\right)}{N}} + \frac {\log \left(\mathcal {M} _ {N} / \delta\right)}{N}\right) \tag {5}
$$

where, $\mathbb{V}_{\mathbf{x}}\mathcal{L}(p_x,\mathbf{w}^\top \mathbf{x})$ and $\mathbb{V}_y[y|\mathbf{x}]$ are the variances in the loss function contributed by $\mathbf{x}$ , and conditional variance of $y$ respectively.

Lemma 1. Assuming the loss weighting factor C defined in Eq. 2 as $C = \frac{N}{S}$ , where S is the threshold defined in Eq. 3 and N is the number of training points, the variance term $V = E_{x}[V_{y}[y|x])(CLWB)^{2}$ in Theorem 1 is bounded by $\frac{N(LWB)^{2}}{S}$ .

Theorem 1 establishes a strong dependence between the classifier performance and the variance in labels $V_{y|x}[y|x]$ with larger values of the latter degrading the effectiveness of the trained classifiers. Furthermore, Lemma 1 shows that a smaller positive sample count S can amplify the adverse effect of label variance which makes the tail classifiers more prone to label variance-related degradation. Now, if we have access to precise estimates of marginal relevance, denoted by $p_{x} = E[y|x]$ , we can replace y with $p_{x}$ , effectively reducing the label variance term to 0. This forms the intuition behind LEVER which employs an additional teacher network to provide accurate estimates of $p_{x}$ .

In practice, however, obtaining a perfect teacher is infeasible both due to modeling and computational hardness issues. As a result, the ability to robustly leverage a partially biased teacher to improve the target student model is essential for the practical utility of LEVER. To enable this, we propose the following variant of LEVER where an imperfect teacher's relevance estimates are used for regularizing the original loss with discrete labels:

$$
\min _ {\mathbf {w}} \frac {\lambda}{N} \sum_ {i = 1} ^ {N} \mathcal {L} (y _ {i}, \mathbf {w} ^ {\top} \mathbf {x} _ {i}) + \frac {1 - \lambda}{N} \sum_ {i = 1} ^ {N} \mathcal {L} (\hat {p} _ {i}, \mathbf {w} ^ {\top} \mathbf {x} _ {i}) \tag {6}
$$

where $\hat{p}_{i}$ are the relevance estimates outputted by the teacher model, and $\lambda$ is a regularization hyperparameter. The above formulation aims to trade off variance errors due to $y_{i}$ with the bias errors due to $\hat{p}_{i}$ to attain the lowest overall generalization error. The following theorem shows that, for an appropriate choice of $\lambda$ , the risk of the resulting classifier is lower than when trained on either $y_{i}$ or $\hat{p}_{i}$ alone:

Theorem 2. Let $\mathbf{R},\hat{\mathbf{R}}$ be the population risk and empirical risk for a binary classification loss $\mathcal{L}$ . Let $\mathcal{M}_N$ be the uniform covering number (Menon et al., 2021a) corresponding to $\mathcal{L}$ . Also, let the teacher be imperfect with maximum possible error in relevance estimates bounded by $E = \| p_{\mathbf{x}} - \hat{p}_{\mathbf{x}}\|_{\infty}$ . Then, by solving the regularized optimization problem $\hat{\mathbf{R}}_s = \min_{\mathbf{w}}\frac{\lambda}{N}\sum_{i=1}^{N}\mathcal{L}(y_i,\mathbf{w}^\top \mathbf{x}_i) + \frac{1-\lambda}{N}\sum_{i=1}^{N}\mathcal{L}(\hat{p}_i,\mathbf{w}^\top \mathbf{x}_i)$ and setting $\lambda$ to minimize population risk- for any $\delta \in (0,1)$ , the following inequality holds with probability at least $1 - \delta$ over sampling the data points $\{\mathbf{x}\}_{i=1}^N$ under the assumption of a reasonably small teacher error $(E)$ :

$$
\lambda = \frac {c}{b} \sqrt {\frac {a}{b ^ {2} - c ^ {2}}} \quad ; \quad \mathbf {R} \leq \hat {\mathbf {R}} _ {s} + \sqrt {a - a \frac {c ^ {2}}{b ^ {2}}} + c + \frac {\log (\mathcal {M} _ {N} / \delta)}{N} \tag {7}
$$

$$
\text { where, } \quad a = V _ {x} \frac {\log (\mathcal {M} _ {N} / \delta)}{N}; \quad b = \sqrt {S \log (\mathcal {M} _ {N} / \delta)} \frac {C L W B}{N}; \quad c = E C L W B \tag {8}
$$

Note that when $c = 0$ , $\lambda = 0$ which is equivalent to training on pure teacher estimates. Also, when $0 < c \leq b$ , $\sqrt{a - a\frac{c^2}{b^2}} + c \leq \min \{\sqrt{a + b^2}, \sqrt{a} + c\}$ . In other words, the bound over population risk is tighter than when $\lambda = 0$ or $\lambda = 1$ . Therefore, trading off the teacher's bias with label variance by setting an appropriate $0 < \lambda < 1$ can lead to better generalization than pure training with either original ground truth or biased teacher estimates as label targets.

# 3.3 A SIAMESE-STYLE TEACHER FOR LEVER

Recent studies have shown that Siamese Networks, when used as input encoders, exhibit strong performance on tail labels (Dahiya et al., 2021a; 2023a; Jain et al., 2023). This success can be

attributed to the ability of Siamese encoders to leverage label correlations by utilizing label-side features. These features, often presented as descriptive text or structured graphs over labels, are commonly found in XC applications. In fact, most recent XC datasets have started to incorporate them (Bhatia et al., 2016). Consequently, this allows for the sharing of information between semantically similar labels, effectively addressing the problem of data scarcity in tail labels. It is important to note, however, that a standalone Siamese model is insufficient as it tends to under-fit data-rich head labels, thereby compromising overall prediction quality. This paper, therefore, proposes the use of Siamese Networks as teachers within the LEVER framework to enhance the tail performance of one-vs-all classifiers. By employing LEVER, we can improve the tail performance of one-vs-all classifiers without compromising their already excellent head accuracies.

A Siamese encoder, $E_{\theta}$ , is trained to map the features of data points, denoted as $\{x_{i}\}_{i=1}^{N}$ , and label features, represented as $\{z_{l}\}_{l=1}^{L}$ , into a common embedding space. The objective of this mapping is to ensure that labels relevant to a given data point are positioned closer in the embedding space, while those that are irrelevant are distanced. Typically, this is achieved by minimizing a triplet loss $[\mathbf{z}_{l}^{\top}\mathbf{x}_{k}-\mathbf{z}_{l}^{\top}\mathbf{x}_{i}+\Delta]_{+}$ , where k and l are a negative and a positive samples, respectively, for label l and $\Delta$ is a margin enforced for better generalization (Dahiya et al., 2021a; 2023a). However, the triplet-loss is not probabilistically calibrated and does not provide reliable marginal relevance targets for training a student. To address this, we leverage a logistic-loss based objective that is found to be well-calibrated:

$$
\min_ {\theta} \sum_ {l \in L} \sum_ {\substack {k \in X _ {-} \\ i \in X _ {+}}} \log (1 + e ^ {\mathbf {z} _ {l} ^ {\top} \mathbf {x} _ {k} - \mathbf {z} _ {l} ^ {\top} \mathbf {x} _ {i} + \Delta}) \tag{9}
$$

The following theorem demonstrates the calibration property of Eq. 9 assuming that the loss can be fully minimized, i.e., loss between each positive-negative pair is minimized.

Theorem 3. Consider a label z, and a pair of data points $x_{a}, x_{b}$ . Let $p_{a}, p_{b}$ be the probabilities that the label is relevant to points a, b respectively. Then, assuming that Eq. 9 is fully minimized, the expected loss in Eq. 9 is minimized for $p_{a} = 1/(1 + e^{-(\mathbf{z}^{\top}\mathbf{x}_{a}+c)}), p_{b} = 1/(1 + e^{-(\mathbf{z}^{\top}\mathbf{x}_{b}+c)})$ .

The above result shows a direct connection between the Siamese model's scores and relevance probabilities, which can be exploited as teacher targets. The parameter $c$ is a hyper-parameter, and it is fitted by cross-validation. While the above strategy provides well-calibrated scores, we empirically observe that simple score mapping strategies, such as $p_a = \frac{\cos\text{Sim}(z,x_a) + 1}{2}$ , where $\cos \text{Sim}$ represents the cosine similarity, also work equally well.

To make training tractable, we follow the negative mining strategy used in NGAME (Dahiya et al., 2023a). Motivated by recent works that under-sample (or oversample) model inputs (Menon et al., 2021b) to address dataset imbalance, we modify NGAME's point-wise sampling strategy to a label-wise approach, in which mini-batches are made from labels rather than points. This adjustment leads to the up-sampling of tail labels, thereby increasing their importance during training. Empirically, we find that a teacher trained via this strategy exhibits better tail performance. Subsequently, the one-vs-all (OvA) classifier distilled from this teacher outperforms the OvA classifier distilled from the Siamese teacher trained with point-wise sampling, both in precision (+0.37% on average in P@1) and in PSP (+1.7% on average in PSP@1), as detailed in Table 12 in the appendix.

# 4 CONTRIBUTED DATASETS

Motivation Performance evaluation of XC algorithms has largely relied on public benchmark datasets available from (Bhatia et al., 2016). In these datasets, the data points associated with a label tend to be fairly similar to each other in their semantic intents. We refer to these as single-intent datasets. For example, in LF-AmazonTitles-131K, the label “clothing for men” might be associated with “formal shirts for men” or “casual shirts for men”. In contrast, several real-world XC applications belong to a multi-intent setting where the label can be associated with data points of vastly different intents. For instance, in query auto-completion (Yadav et al., 2021) where the prefix of a search query needs to be mapped to its completing suffixes, a suffix “..book” might start with either “face..” or “note..” as prefix thus leading to completely different final queries. Such multi-intent datasets can be challenging for XC but are under-represented among existing benchmarks. Additionally, the datasets we release exhibit significant imbalances compared to existing benchmarks, with

7 – 11% of the labels accounting for 80% of the positive instances (refer Table 3). This imbalance poses multiple challenges. First, methods like GLaS (Guo et al., 2019) and Gandalf (Kharbanda et al., 2024), which depend on label correlations for regularization or data augmentation, struggle due to the sparse correlations among tail labels when imbalance is high (refer Table 2). Second, classifier-based methods may achieve high precision by focusing on the head labels, but this results in poor performance on tail metrics such as coverage (refer Table 2). We believe that the contributed datasets will promote further study into developing methods that are robust across various dataset settings.

Contributed datasets: Two new datasets, LF-AOL-270K and LF-WikiHierarchy-1M are curated. LF-AOL-270K involves the query auto-completion task of matching a query prefix with completing suffixes. It is curated from publicly available AOL search logs (Pass et al., 2006). LF-WikiHierarchy-1M involves the taxonomy completion task (Benaouicha et al., 2016) of matching a Wikipedia category to its parent categories (Zesch & Gurevych, 2007). This dataset is motivated by the real-world application of query-to-ad keyword matching where a keyword can subsume the intent of its query thus giving rise to hierarchical association structures. Complete dataset creation details and dataset statistics are provided in appendix Sec. B.3.

# 5 EXPERIMENTS AND RESULTS

Datasets: LEVER was evaluated on a diverse set of datasets, encompassing both full-text and short-text feature scenarios, as well as novel multi-intent datasets. Specifically, we utilized three full-text datasets (LF-Amazon-131K, LF-Wikipedia-500K, LF-WikiSeeAlso-320K), two short-text datasets (LF-AmazonTitles-131K, LF-AmazonTitles-1.3M), and two new multi-intent datasets (LF-WikiHierarchy-1M and LF-AOL-270K). For detailed dataset statistics, please refer to Table 3 in the appendix. Additionally, we evaluate LEVER on a large proprietary query-to-keyword matching dataset with 20M labels (refer Sec. B.2 in the appendix for more details).

Evaluation Metrics: To assess the test-time performance, standard evaluation metrics were used, namely precision@k (P@k, k=1, 3, and 5) and its propensity-weighted variant PSP@k (with k=1, 3, and 5). Detailed definitions for these metrics can be found in (Bhatia et al., 2016). Additionally, following the recommendations in (Schultheis et al., 2022), we also included coverage@k (C@k) as an important metric to evaluate the tail performance.

Baselines We applied LEVER to improve multiple strong OvA-based baselines, including CascadeXML (Kharbanda et al., 2022), ELIAS (Gupta et al., 2022), and Renée (Jain et al., 2023), for demonstrating its effectiveness and generality. We also compared LEVER to other competing tail-enhancement techniques including regularization-based methods such as GLaS (Guo et al., 2019) and L2-regularization, data augmentation methods like TAUG (Wei et al., 2021) and Gandalf (Kharbanda et al., 2024), and propensity weighting approaches such as Re-rank (Wei et al., 2021). For comprehensive details on model hyper-parameters, please refer to Sec. D in the appendix.

LEVER Implementation Details As discussed in Sec. 3, LEVER uses a Siamese teacher to obtain relevance estimates, $\hat{p}$ . Using the relevance estimates an augmented dataset $\mathcal{D}_{aug}$ is created by adding each label as a document, resulting in a dataset comprising $N + L$ documents and $L$ labels. Document and label embeddings from the Siamese teacher are then used to add $\tau_{l}$ nearest labels, and $\tau_{d}$ nearest documents for a particular label. In total, $\tau = \tau_{l} + \tau_{d}$ elements are added for each label. Empirically, we find that not adding documents ( $\tau_{d} = 0$ ) leads to performance similar to that of adding documents in most cases. For more details on LEVER's hyper-parameters refer Sec. D.7.1 in the appendix.

Performance on SOTA OvA methods Table 1 demonstrates LEVER's effectiveness when applied to leading classifier-based XC methods, including CascadeXML, ELIAS, and Renée. LEVER consistently improves P@1 and PSP@1 on average by $2\%$ and $5\%$ , respectively, across all base models and datasets. When applied to Renée, LEVER achieves new state-of-the-art, increasing PSP@1 by up to $5\%$ while maintaining comparable precision. Notably, LEVER proves highly effective on smaller datasets (LF-AmazonTitles-131K, LF-Amazon-131K), highlighting its importance when data is limited. Table 13 in appendix further illustrates LEVER's gains on a proprietary dataset containing 20M labels. Larger improvements in ELIAS and CascadeXML are attributed to these models not explicitly utilizing label features during training or initialization. In contrast, Renée, which uses

Table 1: LEVER can be applied to improve any OvA-based approach. When used with leading OvA approaches LEVER consistently boosts tail performance across all benchmarks, increasing PSP on average by 5.3% while maintaining comparable precision (1.4% gain on average). Coverage metrics (reported in Table 5 in the appendix) show similar trends with an average gain of 6.5%. 

<table><tr><td rowspan="2">Model</td><td colspan="6">LF-AmazonTitles-131K</td><td colspan="6">LF-Amazon-131K</td></tr><tr><td>P@1</td><td>P@3</td><td>P@5</td><td>PSP@1</td><td>PSP@3</td><td>PSP@5</td><td>P@1</td><td>P@3</td><td>P@5</td><td>PSP@1</td><td>PSP@3</td><td>PSP@5</td></tr><tr><td>ELIAS</td><td>37.28</td><td>25.18</td><td>18.14</td><td>28.95</td><td>34.45</td><td>39.08</td><td>43.03</td><td>29.27</td><td>21.20</td><td>33.49</td><td>40.80</td><td>46.76</td></tr><tr><td>ELIAS + LEVER</td><td>42.86</td><td>28.37</td><td>20.16</td><td>36.30</td><td>41.05</td><td>45.43</td><td>47.38</td><td>32.24</td><td>23.22</td><td>38.97</td><td>46.74</td><td>52.79</td></tr><tr><td>CascadeXML</td><td>36.28</td><td>24.88</td><td>18.18</td><td>26.50</td><td>33.21</td><td>38.81</td><td>43.76</td><td>29.75</td><td>21.58</td><td>34.05</td><td>41.69</td><td>47.96</td></tr><tr><td>CascadeXML + LEVER</td><td>43.58</td><td>28.79</td><td>20.63</td><td>36.24</td><td>41.83</td><td>46.95</td><td>48.24</td><td>32.82</td><td>23.73</td><td>39.09</td><td>47.55</td><td>54.18</td></tr><tr><td>Renée</td><td>46.05</td><td>30.81</td><td>22.04</td><td>38.47</td><td>44.87</td><td>50.33</td><td>48.05</td><td>32.33</td><td>23.26</td><td>39.32</td><td>47.10</td><td>53.51</td></tr><tr><td>Renée + LEVER</td><td>46.44</td><td>30.83</td><td>21.92</td><td>39.70</td><td>45.44</td><td>50.31</td><td>49.19</td><td>33.30</td><td>24.04</td><td>40.64</td><td>48.48</td><td>54.87</td></tr><tr><td rowspan="2"></td><td colspan="6">LF-Wikipedia-500K</td><td colspan="6">LF-AmazonTitles-1.3M</td></tr><tr><td>P@1</td><td>P@3</td><td>P@5</td><td>PSP@1</td><td>PSP@3</td><td>PSP@5</td><td>P@1</td><td>P@3</td><td>P@5</td><td>PSP@1</td><td>PSP@3</td><td>PSP@5</td></tr><tr><td>ELIAS</td><td>81.94</td><td>62.71</td><td>48.75</td><td>33.58</td><td>43.92</td><td>48.67</td><td>47.48</td><td>42.21</td><td>38.60</td><td>18.79</td><td>23.20</td><td>26.06</td></tr><tr><td>ELIAS + LEVER</td><td>82.44</td><td>63.88</td><td>50.03</td><td>36.94</td><td>49.28</td><td>55.03</td><td>48.91</td><td>43.17</td><td>39.28</td><td>23.68</td><td>27.43</td><td>29.72</td></tr><tr><td>CascadeXML</td><td>77.00</td><td>58.30</td><td>45.10</td><td>31.25</td><td>39.35</td><td>43.29</td><td>47.14</td><td>41.43</td><td>37.73</td><td>15.92</td><td>20.23</td><td>23.16</td></tr><tr><td>CascadeXML + LEVER</td><td>80.10</td><td>60.41</td><td>46.44</td><td>36.79</td><td>46.65</td><td>50.99</td><td>47.98</td><td>42.02</td><td>38.12</td><td>20.06</td><td>24.51</td><td>27.28</td></tr><tr><td>Renée</td><td>84.95</td><td>66.25</td><td>51.68</td><td>37.10</td><td>50.27</td><td>55.68</td><td>56.10</td><td>49.91</td><td>45.32</td><td>28.56</td><td>33.38</td><td>36.14</td></tr><tr><td>Renée + LEVER</td><td>85.02</td><td>66.37</td><td>51.98</td><td>42.93</td><td>55.00</td><td>60.29</td><td>56.01</td><td>49.43</td><td>44.85</td><td>33.55</td><td>36.82</td><td>38.81</td></tr><tr><td rowspan="2"></td><td colspan="6">LF-AOL-270K</td><td colspan="6">LF-WikiHierarchy-1M</td></tr><tr><td>P@1</td><td>P@3</td><td>P@5</td><td>PSP@1</td><td>PSP@3</td><td>PSP@5</td><td>P@1</td><td>P@3</td><td>P@5</td><td>PSP@1</td><td>PSP@3</td><td>PSP@5</td></tr><tr><td>ELIAS</td><td>40.83</td><td>22.33</td><td>14.91</td><td>13.29</td><td>21.46</td><td>25.22</td><td>95.27</td><td>94.25</td><td>92.45</td><td>17.15</td><td>24.41</td><td>30.01</td></tr><tr><td>ELIAS + LEVER</td><td>40.85</td><td>22.83</td><td>15.57</td><td>13.68</td><td>24.30</td><td>30.43</td><td>94.02</td><td>91.97</td><td>89.50</td><td>28.27</td><td>36.80</td><td>42.13</td></tr><tr><td>CascadeXML</td><td>41.20</td><td>22.12</td><td>14.82</td><td>12.58</td><td>19.53</td><td>23.19</td><td>94.88</td><td>93.69</td><td>91.79</td><td>16.03</td><td>22.87</td><td>28.17</td></tr><tr><td>CascadeXML + LEVER</td><td>39.41</td><td>21.78</td><td>14.99</td><td>11.96</td><td>21.30</td><td>27.59</td><td>94.77</td><td>93.54</td><td>91.56</td><td>20.14</td><td>27.49</td><td>33.01</td></tr><tr><td>Renée</td><td>40.97</td><td>23.34</td><td>15.85</td><td>14.76</td><td>26.45</td><td>32.19</td><td>95.01</td><td>93.99</td><td>92.24</td><td>19.69</td><td>27.36</td><td>33.20</td></tr><tr><td>Renée + LEVER</td><td>41.70</td><td>24.76</td><td>17.07</td><td>20.38</td><td>37.07</td><td>45.13</td><td>95.19</td><td>93.91</td><td>92.07</td><td>24.76</td><td>32.63</td><td>38.15</td></tr></table>

Table 2: Comparison of LEVER with other tail specific XC approaches. LEVER outperforms regularization and augmentation-based methods by an average of 4% in coverage and 3% in PSP. 

<table><tr><td rowspan="2"></td><td colspan="6">LF-AmazonTitles-131K</td><td colspan="6">LF-AOL-270K</td></tr><tr><td>C@1</td><td>C@3</td><td>C@5</td><td>PSP@1</td><td>PSP@3</td><td>PSP@5</td><td>C@1</td><td>C@3</td><td>C@5</td><td>PSP@1</td><td>PSP@3</td><td>PSP@5</td></tr><tr><td>Renée</td><td>31.31</td><td>53.50</td><td>61.03</td><td>38.47</td><td>44.87</td><td>50.33</td><td>12.40</td><td>29.77</td><td>36.53</td><td>14.76</td><td>26.45</td><td>32.19</td></tr><tr><td>Renée +TAUG</td><td>29.47</td><td>51.52</td><td>58.68</td><td>36.49</td><td>42.83</td><td>47.85</td><td>12.46</td><td>29.26</td><td>35.88</td><td>15.72</td><td>26.74</td><td>32.35</td></tr><tr><td>Renée + BoW</td><td>30.03</td><td>51.78</td><td>59.17</td><td>36.96</td><td>42.86</td><td>48.09</td><td>12.67</td><td>34.32</td><td>43.45</td><td>15.58</td><td>30.28</td><td>37.90</td></tr><tr><td>Renée + L2Reg</td><td>31.66</td><td>53.65</td><td>60.80</td><td>38.74</td><td>44.53</td><td>49.49</td><td>8.67</td><td>21.07</td><td>26.27</td><td>12.21</td><td>20.09</td><td>24.36</td></tr><tr><td>Renée + GLaS</td><td>31.90</td><td>54.02</td><td>61.15</td><td>38.74</td><td>44.53</td><td>49.49</td><td>12.36</td><td>29.41</td><td>36.06</td><td>14.67</td><td>26.11</td><td>36.75</td></tr><tr><td>Renée + Gandalf</td><td>33.17</td><td>55.36</td><td>62.22</td><td>40.49</td><td>45.83</td><td>50.96</td><td>12.63</td><td>29.82</td><td>36.31</td><td>15.10</td><td>26.64</td><td>32.17</td></tr><tr><td>Renée + LEVER</td><td>32.50</td><td>54.59</td><td>61.42</td><td>39.70</td><td>45.44</td><td>50.31</td><td>17.43</td><td>42.54</td><td>52.01</td><td>20.38</td><td>37.07</td><td>45.14</td></tr><tr><td></td><td colspan="6">LF-Wikipedia-500K</td><td colspan="6">LF-WikiHierarchy-1M</td></tr><tr><td>Renée</td><td>22.90</td><td>50.08</td><td>61.59</td><td>37.10</td><td>50.27</td><td>55.68</td><td>6.72</td><td>11.49</td><td>14.65</td><td>19.69</td><td>27.36</td><td>33.20</td></tr><tr><td>Renée + TAUG</td><td>19.88</td><td>44.74</td><td>56.13</td><td>33.76</td><td>46.54</td><td>52.16</td><td>3.59</td><td>7.19</td><td>9.94</td><td>16.95</td><td>24.06</td><td>29.69</td></tr><tr><td>Renée + BoW</td><td>22.92</td><td>49.64</td><td>61.40</td><td>36.66</td><td>49.79</td><td>55.55</td><td>7.84</td><td>14.77</td><td>18.39</td><td>24.25</td><td>31.10</td><td>36.30</td></tr><tr><td>Renée + L2Reg</td><td>26.52</td><td>53.95</td><td>65.14</td><td>39.55</td><td>52.42</td><td>57.43</td><td>5.61</td><td>9.96</td><td>12.98</td><td>18.56</td><td>25.90</td><td>31.49</td></tr><tr><td>Renée + GLaS</td><td>23.43</td><td>52.02</td><td>63.90</td><td>37.27</td><td>51.54</td><td>57.15</td><td>6.89</td><td>11.82</td><td>15.08</td><td>20.07</td><td>27.82</td><td>33.70</td></tr><tr><td>Renée + Gandalf</td><td>23.09</td><td>49.87</td><td>61.24</td><td>37.05</td><td>49.94</td><td>55.31</td><td>6.92</td><td>13.17</td><td>17.52</td><td>21.84</td><td>30.05</td><td>36.09</td></tr><tr><td>Renée + LEVER</td><td>29.46</td><td>58.53</td><td>70.29</td><td>42.93</td><td>55.00</td><td>60.29</td><td>9.32</td><td>16.41</td><td>20.29</td><td>24.76</td><td>32.63</td><td>38.15</td></tr></table>

the NGAME encoder for initialization, shows comparatively modest gains with LEVER. Moreover, Table 4 in the appendix illustrates the performance of LEVER when combined with XReg (Prabhu et al., 2020), an extension of Parabel, showcasing that LEVER can effectively combine with non-DNN-based methods too.

Comparison with Tail Extreme Classification Methods: In Table 2, we present a comparative analysis of Renée + LEVER against leading tail label-specialized methods. Note that these approaches can be easily integrated with OvA classifiers without any architectural modifications.

These methods can be broadly categorized into two classes: (1) regularization-based, such as GLaS and L2-regularization. GLaS promotes the proximity of classifiers for labels with similar ground truths, while L2-regularization introduces an additional L2 loss between tail expert label embeddings and label classifiers. (2) Augmentation-based, such as TAUG and Gandalf, which introduce additional training data for labels. Detailed comparisons with other prominent Extreme Classification methods, including XR-Transformer (Zhang et al., 2021), ELIAS, CascadeXML, NGAME, and ECLARE (Mittal et al., 2021b), are provided in Table 7 within the appendix. Our primary focus here is on tail label performance, hence we report PSP and coverage metrics.

LEVER consistently outperforms the second-best method by an average margin of 4% in coverage and 3% in PSP. Notably, on datasets characterized by significant skew and multi-intent scenarios, LEVER exhibits substantial gains in comparison to approaches like GLaS and Gandalf, which rely on ground truth data to model label correlations. For example, in the query completion task on the AOL dataset, the label “who wrote To Kill a Mockingbird” co-occurs with labels like “wholesale t-shirts” or “who am I” as they share the prefix “who”. Training classifiers with such diverse targets can lead to associations between dissimilar labels, hampering classifier training. Using Bag of Words (BoW) features from label text to model label connections alleviates the multi-intent and skew issue to some extent, as observed when Renée+ BoW performs better than Renée+ GLaS/Gandalf in LF-AOL-270K and LF-WikiHierarchy-1M. However, LEVER goes further by learning semantic associations between labels and documents through a tail-expert Siamese network, surpassing raw text-based methods.

Comparison with Siamese Teacher: Table 15 in the appendix compares LEVER against its corresponding Siamese encoder-based teacher. LEVER utilizes the teacher to improve OvA performance on the tail without degrading the classifier performance on the head labels. As a result, the student model in LEVER can surpass its own teacher in overall performance since it outperforms the Siamese teacher on the head labels while more-or-less equalizing on the tail.

Comparison with an ensemble of OvA classifier and tail-expert: To combine the strengths of OvA classifiers and encoder, another option might be to consider an ensemble model that uses predictions from the OvA model for head labels and the encoder predictions for the tail labels. Table 8 in the appendix compares LEVER with an ensemble of OvA (Renée) and Siamese Encoder. LEVER outperforms the ensemble on both precision and tail metrics. A more detailed discussion of this is provided in Sec. C.3 of the appendix.

Choice of expert encoder: LEVER utilizes a 6-layer DistilBert as an expert encoder. In Table 11 in the appendix we show results for two other light-weight encoders: a 3-layer MiniLM (Wang et al., 2020) and Astec Encoder (Dahiya et al., 2021b). We observe that a superior expert encoder leads to improved performance in both P and PSP.

Effect of varying $\tau$ : Table 21 and Figure 7 in appendix shows effect of varying $\tau$ on LEVER's performance. Increasing $\tau$ improves performance on tail, while it hurts head and torso labels.

LEVER Computational Cost: Since LEVER is a training time-only modification, it leaves the inference costs unchanged while increasing the training time on average by 3.1x. Table 23 in the appendix shows the training time for different models and datasets when combined with LEVER. Note that in ELIAS and CascadeXML, where the train times increase by a greater margin, the gains provided by LEVER are also higher (avg. +6.1% increase in PSP and +2% increase in P). Tables 24, 25 and 26, in the appendix show the break down of the train times for Renée, ELIAS, and CascadeXML respectively.

# 6 CONCLUSIONS

This paper presented a novel approach to address the challenges of tail performance in Extreme Classification (XC) by focusing on label variance, a previously unexplored factor. It proposed LEVER framework for leveraging a tail-robust teacher model to systematically reduce label variance, thereby enhancing the performance of one-vs-all classifiers. It further developed an effective instantiation of this framework using a specialized Siamese teacher model. Experimental results on various XC datasets demonstrated significant improvements in tail performance metrics when LEVER was integrated with leading extreme classifiers, and advanced the state-of-the-art in XC. Finally, this paper also released two new and multi-intent datasets for robust benchmarking in XC.

# REFERENCES

Rahul Agrawal, Archit Gupta, Yashoteja Prabhu, and Manik Varma. Multi-label learning with millions of labels: Recommending advertiser bid phrases for web pages. In Proceedings of the 22nd international conference on World Wide Web, pp. 13–24, 2013.   
Rohit Babbar and Bernhard Schölkopf. Dismec: Distributed sparse machines for extreme multi-label classification. In Proceedings of the Tenth ACM International Conference on Web Search and Data Mining, WSDM '17, pp. 721–729, New York, NY, USA, 2017. Association for Computing Machinery. ISBN 9781450346757. doi: 10.1145/3018661.3018741. URL https://doi.org/10.1145/3018661.3018741.   
Rohit Babbar and Bernhard Schölkopf. Data scarcity, robustness and extreme multi-label classification. Machine Learning, 108(8):1329–1351, 2019.   
Mohamed Benaouicha, Mohamed Ali Hadj Taieb, and Malek Ezzeddine. Derivation of "is a" taxonomy from wikipedia category graph. Eng. Appl. Artif. Intell., 50:265–286, 2016.   
Alina Beygelzimer, John Langford, Yuri Lifshits, Gregory Sorkin, and Alex Strehl. Conditional probability tree estimation analysis and algorithms. In Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence, pp. 51–58, 2009.   
Kush Bhatia, Kunal Dahiya, Himanshu Jain, Purushottam Kar, Anshul Mittal, Yashoteja Prabhu, and Manik Varma. The extreme classification repository: Multi-label datasets and code, 2016. URL http://manikvarma.org/downloads/XC/XMLRepository.html.   
Wei-Cheng Chang, Hsiang-Fu Yu, Kai Zhong, Yiming Yang, and Inderjit S Dhillon. Taming pretrained transformers for extreme multi-label text classification. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 3163–3171, 2020.   
Kunal Dahiya, Ananye Agarwal, Deepak Saini, K Gururaj, Jian Jiao, Amit Singh, Sumeet Agarwal, Purushottam Kar, and Manik Varma. Siamesexml: Siamese networks meet extreme classifiers with 100m labels. In International Conference on Machine Learning, pp. 2330–2340. PMLR, 2021a.   
Kunal Dahiya, Deepak Saini, Anshul Mittal, Ankush Shaw, Kushal Dave, Akshay Soni, Himanshu Jain, Sumeet Agarwal, and Manik Varma. Deepxml: A deep extreme multi-label learning framework applied to short text documents. In Proceedings of the 14th ACM International Conference on Web Search and Data Mining, pp. 31–39, 2021b.   
Kunal Dahiya, Nilesh Gupta, Deepak Saini, Akshay Soni, Yajun Wang, Kushal Dave, Jian Jiao, Gururaj K, Prasenjit Dey, Amit Singh, Deepesh Hada, Vidit Jain, Bhawna Paliwal, Anshul Mittal, Sonu Mehta, Ramachandran Ramjee, Sumeet Agarwal, Purushottam Kar, and Manik Varma. Ngame: Negative mining-aware mini-batching for extreme classification. In Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining, WSDM '23, pp. 258–266, New York, NY, USA, 2023a. Association for Computing Machinery. ISBN 9781450394079. doi: 10.1145/3539597.3570392. URL https://doi.org/10.1145/3539597.3570392.   
Kunal Dahiya, Sachin Yadav, Sushant Sondhi, Deepak Saini, Sonu Mehta, Jian Jiao, Sumeet Agarwal, Purushottam Kar, and Manik Varma. Deep encoders with auxiliary parameters for extreme classification. In In Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Long Beach, California, August 2023b. URL https://www.microsoft.com/en-us/research/publication/deep-encoders-with-auxiliary-parameters-for-extreme-classification/.   
Chuan Guo, Ali Mousavi, Xiang Wu, Daniel N Holtmann-Rice, Satyen Kale, Sashank Reddi, and Sanjiv Kumar. Breaking the glass ceiling for embedding-based classifiers for large output spaces. Advances in Neural Information Processing Systems, 32, 2019.   
Nilesh Gupta, Patrick Chen, Hsiang-Fu Yu, Cho-Jui Hsieh, and Inderjit Dhillon. Elias: End-to-end learning to index and search in large output spaces. Advances in Neural Information Processing Systems, 35:19798–19809, 2022.

Himanshu Jain, Yashoteja Prabhu, and Manik Varma. Extreme multi-label loss functions for recommendation, tagging, ranking & other missing label applications. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining, pp. 935–944, 2016.   
Vidit Jain, Jatin Prakash, Deepak Saini, Jian Jiao, Ramachandran Ramjee, and Manik Varma. Renee: End-to-end training of extreme classification models. In Proceedings of Machine Learning and Systems, pp. To appear, 2023.   
Jyun-Yu Jiang, Wei-Cheng Chang, Jiong Zhang, Cho-Jui Hsieh, and Hsiang-Fu Yu. Uncertainty quantification for extreme classification. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR '23, pp. 1649–1659, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9781450394086. doi:10.1145/3539618.3591780. URL https://doi.org/10.1145/3539618.3591780.   
Ting Jiang, Deqing Wang, Leilei Sun, Huayi Yang, Zhengyang Zhao, and Fuzhen Zhuang. Lightxml: Transformer with dynamic negative sampling for high-performance extreme multi-label text classification. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 7987–7994, 2021.   
Sham M Kakade, Karthik Sridharan, and Ambuj Tewari. On the complexity of linear prediction: Risk bounds, margin bounds, and regularization. In Advances in neural information processing systems. Curran Associates, Inc., 2008. URL https://proceedings.neurips.cc/paper\_files/paper/2008/file/5b69b9cb83065d403869739ae7f0995e-Paper.pdf.   
Siddhant Kharbanda, Atmadeep Banerjee, Erik Schultheis, and Rohit Babbar. Cascadexml: Rethinking transformers for end-to-end multi-resolution training in extreme multi-label classification. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 2074–2087. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper\_files/paper/2022/file/0e0157ce5ea15831072be4744cbd5334-Paper-Conference.pdf.   
Siddhant Kharbanda, Devaansh Gupta, Erik Schultheis, Atmadeep Banerjee, Vikas Verma, and Rohit Babbar. Gandalf: Learning label correlations in extreme multi-label classification via label features, 2024. URL https://openreview.net/forum?id=JuyFppXzh2.   
Gyuwan Kim. Subword language model for query auto-completion. arXiv preprint arXiv:1909.00599, 2019.   
Andreas Maurer and Massimiliano Pontil. Empirical bernstein bounds and sample variance penalization. arXiv preprint arXiv:0907.3740, 2009.   
Aditya K Menon, Ankit Singh Rawat, Sashank Reddi, Seungyeon Kim, and Sanjiv Kumar. A statistical perspective on distillation. In Proceedings of the 38th International Conference on Machine Learning, 2021a.   
Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh Rawat, Himanshu Jain, Andreas Veit, and Sanjiv Kumar. Long-tail learning via logit adjustment. In International Conference on Learning Representations, 2021b. URL https://openreview.net/forum?id=37nvvqkCo5.   
Bhaskar Mitra and Nick Craswell. Query auto-completion for rare prefixes. CIKM '15: Proceedings of the 24th ACM International on Conference on Information and Knowledge Management, 2015.   
Anshul Mittal, Kunal Dahiya, Sheshansh Agrawal, Deepak Saini, Sumeet Agarwal, Purushottam Kar, and Manik Varma. Decaf: Deep extreme classification with label features. In Proceedings of the 14th ACM International Conference on Web Search and Data Mining, pp. 49–57, 2021a.   
Anshul Mittal, Noveen Sachdeva, Sheshansh Agrawal, Sumeet Agarwal, Purushottam Kar, and Manik Varma. Eclare: Extreme classification with label graph correlations. In Proceedings of the Web Conference 2021, pp. 3721–3732, 2021b.   
Greg Pass, Abdur Chowdhury, and Cayley Torgeson. A picture of search. In Proceedings of the 1st international conference on Scalable information systems, 2006.

John Platt. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. Adv. Large Margin Classif., 10, 06 2000.   
Yashoteja Prabhu, Aditya Kusupati, Nilesh Gupta, and Manik Varma. Extreme regression for dynamic search advertising. In Proceedings of the 13th International Conference on Web Search and Data Mining, WSDM '20, pp. 456–464, New York, NY, USA, 2020. Association for Computing Machinery. ISBN 9781450368223. doi: 10.1145/3336191.3371768. URL https://doi.org/10.1145/3336191.3371768.   
Mohammadreza Qaraei, Erik Schultheis, Priyanshu Gupta, and Rohit Babbar. Convex surrogates for unbiased loss functions in extreme classification with missing labels. In Proceedings of the Web Conference, pp. 3711–3720, 2021.   
Erik Schultheis, Marek Wydmuch, Rohit Babbar, and Krzysztof Dembczynski. On missing labels, long-tails and propensities in extreme multi-label classification. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 1547–1557, 2022.   
Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. Minilm: Deep self-attention distillation for task-agnostic compression of pre-trained transformers, 2020.   
Tong Wei, Wei-Wei Tu, Yu-Feng Li, and Guo-Ping Yang. Towards robust prediction on tail labels. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, pp. 1812–1820, 2021.   
Marek Wydmuch, Kalina Jasinska-Kobus, Rohit Babbar, and Krzysztof Dembczynski. Propensity-scored probabilistic label trees. In Proceedings of the International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 2252–2256, 2021.   
Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N Bennett, Junaid Ahmed, and Arnold Overwijk. Approximate nearest neighbor negative contrastive learning for dense text retrieval. In International Conference on Learning Representations, 2020.   
Nishant Yadav, Rajat Sen, Daniel N Hill, Arya Mazumdar, and Inderjit S Dhillon. Session-aware query auto-completion using extreme multi-label ranking. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, pp. 3835–3844, 2021.   
Ronghui You, Suyang Dai, Zihan Zhang, Hiroshi Mamitsuka, and Shanfeng Zhu. Attentionxml: Extreme multi-label text classification with multi-label attention based recurrent neural networks. arXiv preprint arXiv:1811.01727, 137:138–187, 2018.   
Torsten Zesch and Iryna Gurevych. Analysis of the Wikipedia category graph for NLP applications. In Proceedings of the Second Workshop on TextGraphs: Graph-Based Algorithms for Natural Language Processing, pp. 1–8, Rochester, NY, USA, 2007. Association for Computational Linguistics. URL https://aclanthology.org/W07-0201.   
Jiong Zhang, Wei-Cheng Chang, Hsiang-Fu Yu, and Inderjit Dhillon. Fast multi-resolution transformer fine-tuning for extreme multi-label text classification. Advances in Neural Information Processing Systems, 34:7267–7280, 2021.   
Ruohong Zhang, Yau-Shian Wang, Yiming Yang, Donghan Yu, Tom Vu, and Likun Lei. Long-tailed extreme multi-label text classification by the retrieval of generated pseudo label descriptions. In Andreas Vlachos and Isabelle Augenstein (eds.), Findings of the Association for Computational Linguistics: EACL 2023, pp. 1092–1106, Dubrovnik, Croatia, May 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-eacl.81. URL https://aclanthology.org/2023.findings-eacl.81.

# CONTENTS

1 Introduction 1   
2 Related Work 2

2.1 Extreme Classification 2   
2.2 Enhancing Tail Performance in XC 3

3 LEVER: Label Variance Reduction in Extreme Classification 3

3.1 Preliminaries 3   
3.2 LEVER Framework 4   
3.3 A Siamese-Style Teacher for LEVER 5

4 Contributed Datasets 6   
5 Experiments and Results 7   
6 Conclusions 9

A Theoretical Proofs 14   
B Dataset details 17

B.1 Dataset Statistics 17   
B.2 QK-20M Dataset 17   
B.3 Multi-intent dataset preparation ..... 17

B.3.1 LF-AOL-270K 17   
B.3.2 LF-WikiHierarchy-1M 18

C Additional results 20

C.1 LEVER's performance on non-DNN methods ..... 20   
C.2 Comparison with SOTA and Tail XC methods 20   
C.3 Comparison with ensemble between tail Expert and OvA classifier ..... 20   
C.4 Effect of Re-ranking on LEVER and other Tail XC approaches ..... 22   
C.5 Ablations 22

D Model Details and Hyperparameters 25

D.1 Tail expert Siamese Encoder 25   
D.2 ELIAS 26   
D.3 CascadeXML 28   
D.4 Renée 29   
D.5 ReRank + TAUG 30   
D.6 Gandalf 30

D.7 LEVER 31

D.7.1 Hyperparameters 31

D.7.2 Training time 32

# A THEORETICAL PROOFS

Theorem 1. Let $\mathbf{R},\hat{\mathbf{R}}$ be the population risk and empirical risk for a binary classification loss $\mathcal{L}$ . Let $\mathcal{M}_N$ be the uniform covering number (Menon et al., 2021a) corresponding to $\mathcal{L}$ . Then, given the definitions established earlier, for any $\delta \in (0,1)$ , the following inequality holds with probability at least $1 - \delta$ over sampling the data points $\{\mathbf{x}\}_{i=1}^N$ :

$$
\mathbf {R} \leq \hat {\mathbf {R}} + \mathcal {O} \left(\sqrt {\mathbb {V} _ {\mathbf {x}} \left[ \mathcal {L} \left(p _ {x} , \mathbf {w} ^ {\top} \mathbf {x}\right) \right] + \mathbb {E} _ {\mathbf {x}} \left[ \mathbb {V} _ {y} [ y | \mathbf {x} ] \right] (C L W B) ^ {2} ]} \sqrt {\frac {\log \left(\mathcal {M} _ {N} / \delta\right)}{N}} + \frac {\log \left(\mathcal {M} _ {N} / \delta\right)}{N}\right) \tag {10}
$$

where, $\mathbb{V}_{\mathbf{x}}\mathcal{L}(p_{x},\mathbf{w}^{\top}\mathbf{x})$ and $V_{y}[y|\mathbf{x}]$ are the variances in the loss function contributed by sampling of data points $\{x\}_{i=1}^{N}$ , and conditional sampling of labels $\{y\}_{i=1}^{N}$ , respectively.

Proof. Applying Proposition 2. from (Menon et al., 2021a) to our setting gives the following initial result:

$$
\mathbf {R} \leq \hat {\mathbf {R}} + \mathcal {O} \left(\sqrt {\mathbb {V} _ {\mathbf {x} , y} \mathcal {L} (y , \mathbf {w} ^ {\top} \mathbf {x})} \sqrt {\log (\mathcal {M} _ {N} / \delta) / N} + \log (\mathcal {M} _ {N} / \delta) / N\right) \tag {11}
$$

The following simplifications can be made by leveraging basic probabilistic calculus:

$$
\mathbb {V} _ {\mathbf {x}, y} [ \mathcal {L} (y, \mathbf {w} ^ {\top} \mathbf {x}) ] = \mathbb {V} _ {\mathbf {x}} [ \mathbb {E} _ {y} [ \mathcal {L} (y, \mathbf {w} ^ {\top} \mathbf {x}) | \mathbf {x} ] ] + \mathbb {E} _ {\mathbf {x}} [ \mathbb {V} _ {y} [ \mathcal {L} (y, \mathbf {w} ^ {\top} \mathbf {x}) | \mathbf {x} ] ] (\text { by   law   of   total   variance })
$$

$$
\begin{array}{l} = \mathbb {V} _ {\mathbf {x}} [ \mathbb {E} _ {y} [ \mathcal {L} (y, \mathbf {w} ^ {\top} \mathbf {x}) | \mathbf {x} ] ] + \mathbb {E} _ {\mathbf {x}} [ \mathbb {V} _ {y} [ C y f (1, \mathbf {w} ^ {\top} \mathbf {x}) + (1 - y) f (0, \mathbf {w} ^ {\top} \mathbf {x}) | \mathbf {x} ] ] \\ = \mathbb {V} _ {\mathbf {x}} [ \mathcal {L} (p _ {x}, \mathbf {w} ^ {\top} \mathbf {x}) ] + \mathbb {E} _ {\mathbf {x}} [ \mathbb {V} _ {y} [ y | \mathbf {x} ] (C f (1, \mathbf {w} ^ {\top} \mathbf {x}) - f (0, \mathbf {w} ^ {\top} \mathbf {x})) ^ {2} ] \\ = \mathbb {V} _ {\mathbf {x}} [ \mathcal {L} (p _ {x}, \mathbf {w} ^ {\top} \mathbf {x}) ] + \mathbb {E} _ {\mathbf {x}} [ \mathbb {V} _ {y} [ y | \mathbf {x} ] d _ {x} ^ {2} ] \tag {12} \\ \end{array}
$$

$$
\text { where }, d _ {x} = (C f (1, \mathbf {w} ^ {\top} \mathbf {x}) - f (0, \mathbf {w} ^ {\top} \mathbf {x}))
$$

$$
d _ {x} ^ {2} = C ^ {2} f ^ {2} (1, \mathbf {w} ^ {\top} \mathbf {x}) + f ^ {2} (0, \mathbf {w} ^ {\top} \mathbf {x}) - 2 C. f (1, \mathbf {w} ^ {\top} \mathbf {x}). f (0, \mathbf {w} ^ {\top} \mathbf {x})
$$

Since $f \geq 0$ we get,

$$
d _ {x} ^ {2} \leq C ^ {2} f ^ {2} (1, \mathbf {w} ^ {\top} \mathbf {x}) + f ^ {2} (0, \mathbf {w} ^ {\top} \mathbf {x}) \tag {13}
$$

Using 13 in 12 gives,

$$
\mathbb {V} _ {\mathbf {x}, y} \left[ \mathcal {L} \left(y, \mathbf {w} ^ {\top} \mathbf {x}\right) \right] \leq \mathbb {V} _ {\mathbf {x}} \left[ \mathcal {L} \left(p _ {x}, \mathbf {w} ^ {\top} \mathbf {x}\right) \right] + \mathbb {E} _ {\mathbf {x}} \mathbb {V} _ {y} [ y | \mathbf {x} ] \left(C ^ {2} f ^ {2} \left(1, \mathbf {w} ^ {\top} \mathbf {x}\right) + f ^ {2} \left(0, \mathbf {w} ^ {\top} \mathbf {x}\right)\right) ] \tag {14}
$$

Assuming $f(0,0) = f(1,0) = f_0$ (a small constant), and applying the Lipschitz continuity of $f$ gives,

$$
\left| f \left(y, \mathbf {w} ^ {\top} \mathbf {x}\right) - f (y, 0) \right| \leq L \left| \mathbf {w} ^ {\top} \mathbf {x} \right| \leq L W B \quad \forall y \in \{0, 1 \}
$$

$$
f (y, \mathbf {w} ^ {\top} \mathbf {x}) \leq L W B + f _ {0} \tag {15}
$$

Using 15 in 14 gives,

$$
\mathbb {V} _ {\mathbf {x}, y} [ \mathcal {L} (y, \mathbf {w} ^ {\top} \mathbf {x}) ] \leq \mathbb {V} _ {\mathbf {x}} [ \mathcal {L} (p _ {x}, \mathbf {w} ^ {\top} \mathbf {x}) ] + \mathbb {E} _ {\mathbf {x}} \mathbb {V} _ {y} [ y | \mathbf {x} ] (C ^ {2} + 1) (L W B + f _ {0}) ^ {2}
$$

$$
\approx \mathcal {O} \left(\mathbb {V} _ {\mathbf {x}} \left[ \mathcal {L} \left(p _ {x}, \mathbf {w} ^ {\top} \mathbf {x}\right) \right] + \mathbb {E} _ {\mathbf {x}} \mathbb {V} _ {y} [ y | \mathbf {x} ] (C L W B) ^ {2}\right) \tag {16}
$$

Using 16 in 11 completes the proof,

$$
\mathbf {R} \leq \hat {\mathbf {R}} + \mathcal {O} \left(\sqrt {\mathbb {V} _ {\mathbf {x}} \left[ \mathcal {L} \left(p _ {x} , \mathbf {w} ^ {\top} \mathbf {x}\right) \right] + \mathbb {E} _ {\mathbf {x}} \mathbb {V} _ {y} [ y | \mathbf {x} ] (C L W B) ^ {2} ]} \sqrt {\frac {\log \left(\mathcal {M} _ {N} / \delta\right)}{N}} + \frac {\log \left(\mathcal {M} _ {N} / \delta\right)}{N}\right) \tag {17}
$$

□

Lemma 1. Assuming the loss weighting factor C is defined in Equation 2 as $C = \frac{N}{S}$ , where S is the threshold defined in Equation 3 and N is the number of training points, the variance term $\mathcal{V} = \mathbb{E}_{\mathbf{x}}[\mathbb{V}_{y}[y|\mathbf{x}])(CLWB)^{2}$ in Theorem 1 is bounded by $\frac{N(LWB)^{2}}{S}$ .

Proof. Recall from Section 3.1 that $p_x = \mathbb{P}(y = 1|\mathbf{x})$ and $\mathbb{E}_{\mathbf{x}}[p_x] \leq \frac{S}{N}$ . Using $\mathbb{V}_y[y|\mathbf{x}] = p_x(1 - p_x)$ gives,

$$
\mathcal {V} = \mathbb {E} _ {\mathbf {x}} [ p _ {x} (1 - p _ {x}) ] (N / S) ^ {2} (L W B) ^ {2} \leq \mathbb {E} _ {\mathbf {x}} [ p _ {x} ] \frac {(N L W B) ^ {2}}{S ^ {2}} \leq \frac {N (L W B) ^ {2}}{S} \tag {18}
$$

![](images/c378826deaa6314e4448f75a1d531ec6e6b44c263e59eebc7384d9f7d404bc1c.jpg)

Theorem 2. Let R, $\hat{R}$ be the population risk and empirical risk for a binary classification loss L. Let $M_{N}$ be the uniform covering number (Menon et al., 2021a) corresponding to L. Also, let the teacher be imperfect with maximum possible error in relevance estimates bounded by $E = \|p_{x} - \hat{p}_{x}\|_{\infty}$ . Then, solving the following regularized optimization problem:

$$
\hat {\mathbf {R}} _ {s} = \min _ {\mathbf {w}} \frac {\lambda}{N} \sum_ {i = 1} ^ {N} \mathcal {L} (y _ {i}, \mathbf {w} ^ {\top} \mathbf {x} _ {i}) + \frac {1 - \lambda}{N} \sum_ {i = 1} ^ {N} \mathcal {L} (\hat {p} _ {i}, \mathbf {w} ^ {\top} \mathbf {x} _ {i}) \tag {19}
$$

and setting $\lambda$ to minimize population risk will given the following bound for any $\delta\in(0,1)$ , the following inequality holds with probability at least $1-\delta$ over sampling the data points $\{x\}_{i=1}^{N}$ :

$$
\lambda = \frac {c}{b} \sqrt {\frac {a}{b ^ {2} - c ^ {2}}} \quad ; \quad \mathbf {R} \leq \hat {\mathbf {R}} _ {s} + \sqrt {a - a \frac {c ^ {2}}{b ^ {2}}} + c \tag {20}
$$

where, $a = V_{x} \frac{\log(\mathcal{M}_{N}/\delta)}{N}$ ; $b = \sqrt{S} CLWB \sqrt{\frac{\log(\mathcal{M}_{N}/\delta)}{N}}$ ; c = ECLWB (21)

Proof. Teacher tends to be imperfect with relevance estimates $\hat{p}_{x}$ . In this case, let us train the classifier using targets $s_{x} = \lambda y_{x} + (1 - \lambda)\hat{p}_{x}$ . Let the corresponding population and empirical risks when trained on $s_{x}$ be $R_{s}, \hat{R}_{s}$ respectively. Then, the following holds:

$$
\begin{array}{l} \mathbf {R} - \hat {\mathbf {R}} _ {s} = \mathbf {R} - \mathbf {R} _ {s} + \mathbf {R} _ {s} - \hat {\mathbf {R}} _ {s} \\ \leq \left\| \mathbf {R} - \mathbf {R} _ {s} \right\| + \left\| \mathbf {R} _ {s} - \hat {\mathbf {R}} _ {s} \right\| \\ \end{array}
$$

The first term can be bounded as follows:

$$
\begin{array}{l} \mathbf {R} - \mathbf {R} _ {s} = \mathbb {E} _ {\mathbf {x}} \mathbb {E} _ {y | \mathbf {x}} \mathcal {L} (y _ {x}, \mathbf {w} ^ {\top} \mathbf {x}) - \mathcal {L} (\lambda y _ {x} + (1 - \lambda) \hat {p} _ {x}, \mathbf {w} ^ {\top} \mathbf {x}) \\ = \mathbb {E} _ {\mathbf {x}} \mathcal {L} (p _ {x}, \mathbf {w} ^ {\top} \mathbf {x}) - \mathcal {L} (\lambda p _ {x} + (1 - \lambda) \hat {p} _ {x}, \mathbf {w} ^ {\top} \mathbf {x}) \\ \text { Assuming }, \mathcal {L} (y, \mathbf {w} ^ {\top} \mathbf {x}) = C y f (1, \mathbf {w} ^ {\top} \mathbf {x}) + (1 - y) f (0, \mathbf {w} ^ {\top} \mathbf {x}) \\ \mathbf {R} - \mathbf {R} _ {s} = \mathbb {E} _ {\mathbf {x}} [ (1 - \lambda) (p _ {x} - \hat {p} _ {x}) \big (C f (1, \mathbf {w} ^ {\top} \mathbf {x}) - f (0, \mathbf {w} ^ {\top} \mathbf {x}) \big) ] \\ \leq (1 - \lambda) \| p _ {x} - \hat {p} _ {x} \| _ {\infty} \max _ {\mathbf {x}} \left(C f (1, \mathbf {w} ^ {\top} \mathbf {x}) - f (0, \mathbf {w} ^ {\top} \mathbf {x})\right) \\ \leq (1 - \lambda) E (C + 1) (L W B + f _ {0}) \\ \end{array}
$$

$$
\begin{array}{l} \mathbf {R} - \mathbf {R} _ {s} = \mathbb {E} _ {\mathbf {x}} [ (1 - \lambda) (p _ {x} - \hat {p} _ {x}) \big (C f (1, \mathbf {w} ^ {\top} \mathbf {x}) - f (0, \mathbf {w} ^ {\top} \mathbf {x}) \big) ] \\ \leq (1 - \lambda) \| p _ {x} - \hat {p} _ {x} \| _ {\infty} \max _ {\mathbf {x}} \left(C f (1, \mathbf {w} ^ {\top} \mathbf {x}) - f (0, \mathbf {w} ^ {\top} \mathbf {x})\right) \\ \leq (1 - \lambda) E (C + 1) (L W B + f _ {0}) \\ \approx \mathcal {O} ((1 - \lambda) E C L W B) \tag {22} \\ \end{array}
$$

where $E$ is the upper bound over the error in the teacher's relevance estimates.

The second term $\|\mathbf{R}_{s}-\hat{\mathbf{R}}_{s}\|$ can be bounded by applying (A):

$$
\mathbf {R} _ {s} \leq \hat {\mathbf {R}} _ {s} + \mathcal {O} \Big (\sqrt {V _ {x} + \mathbb {E} _ {\mathbf {x}} [ \mathbb {V} _ {y} [ \lambda y + (1 - \lambda) \hat {p} _ {x} | \mathbf {x} ] ] (C L W B) ^ {2} ]} \sqrt {\frac {\log (\mathcal {M} _ {N} / \delta)}{N}} + \frac {\log (\mathcal {M} _ {N} / \delta)}{N} \Big)
$$

Now, $\mathbb{V}_y[\lambda y + (1 - \lambda)\hat{p}_x|\mathbf{x}] = \mathbb{V}_y[\lambda y|\mathbf{x}]$

$$
= \lambda^ {2} \mathbb {V} _ {y} [ y | \mathbf {x} ]
$$

$$
\mathbf {R} _ {s} \leq \hat {\mathbf {R}} _ {s} + \mathcal {O} \Big (\sqrt {V _ {x} + \lambda^ {2} \frac {S (C L W B) ^ {2}}{N}} ] \sqrt {\frac {\log (\mathcal {M} _ {N} / \delta)}{N}} + \frac {\log (\mathcal {M} _ {N} / \delta)}{N} \Big)
$$

where, $\frac{S}{N} \approx \mathbb{E}_x p_x \geq \mathbb{E}_x \mathbb{V}_y[y|\mathbf{x}]$ (23)

As a result:

$$
\mathbf {R} \leq \hat {\mathbf {R}} _ {s} + \mathcal {O} \left(\sqrt {V _ {x} + \lambda^ {2} \frac {S (C L W B) ^ {2}}{N}} ] \sqrt {\frac {\log \left(\mathcal {M} _ {N} / \delta\right)}{N}} + \frac {\log \left(\mathcal {M} _ {N} / \delta\right)}{N}\right) + \mathcal {O} ((1 - \lambda) E C L W B) \tag {24}
$$

As $\lambda$ is a regularization hyper-parameter, it value needs to be set so as to minimize the generalization error. Theoretically, this can be achieved by solving:

$$
\min _ {\lambda} \sqrt {a + \lambda^ {2} b ^ {2}} + (1 - \lambda) c
$$

where, $a = V_{x}\frac{\log(\mathcal{M}_{N} / \delta)}{N}$

$$
b = \sqrt {S \log (\mathcal {M} _ {N} / \delta)} \frac {C L W B}{N}
$$

$$
c = E C L W B \tag {25}
$$

Let's assume a reasonably small bias in teacher estimate. Specifically, let $c < b$ which means that the error due to teacher bias is relatively smaller than the error due to label variance.

Now, taking the derivative w.r.t $\lambda$ and setting it to 0, we get:

$$
\lambda = \frac {c}{b} \sqrt {\frac {a}{b ^ {2} - c ^ {2}}} \tag {26}
$$

$$
\sqrt {a + \lambda^ {2} b ^ {2}} + (1 - \lambda) c = \sqrt {a - a \frac {c ^ {2}}{b ^ {2}}} + c \tag {27}
$$

![](images/e97882dab85ce893010a9cd3069ecf80ee79a92694d0de3d6cc5de24b60ce4d5.jpg)

Theorem 3. Given a label z, and a pair of data points $x_{a}, x_{b}$ . Let $p_{a}, p_{b}$ be the probabilities that the label is relevant to points a, b respectively. Then, assuming that (9) is fully minimized, the expected loss in (9) is minimized for $p_{a} = 1/(1 + e^{-(\mathbf{z}^{\top}\mathbf{x}_{a}+c)}), p_{b} = 1/(1 + e^{-(\mathbf{z}^{\top}\mathbf{x}_{b}+c)})$

Proof. The expected loss between the triplet is given by:

$$
p _ {a} (1 - p _ {b}) \log (1 + e ^ {\mathbf {z} ^ {\top} \mathbf {x} _ {b} - \mathbf {z} ^ {\top} \mathbf {x} _ {a}}) + p _ {b} (1 - p _ {a}) \log (1 + e ^ {\mathbf {z} ^ {\top} \mathbf {x} _ {a} - \mathbf {z} ^ {\top} \mathbf {x} _ {b}})
$$

Assuming $\Delta = \mathbf{z}^{\top}\mathbf{x}_b - \mathbf{z}^{\top}\mathbf{x}_a$ and taking the gradient w.r.t $\mathbf{z}$ gives,

$$
\begin{array}{l} = p _ {a} (1 - p _ {b}) \frac {e ^ {\Delta} (\mathbf {x} _ {b} - \mathbf {x} _ {a})}{1 + e ^ {\Delta}} + p _ {b} (1 - p _ {a}) \frac {e ^ {- \Delta} (\mathbf {x} _ {a} - \mathbf {x} _ {b})}{1 + e ^ {- \Delta}} \\ = \frac {\mathbf {x} _ {b} - \mathbf {x} _ {a}}{1 + e ^ {\Delta}} \left(e ^ {\Delta} p _ {a} (1 - p _ {b}) - p _ {b} (1 - p _ {a})\right) \tag {28} \\ \end{array}
$$

Setting $p_a = 1 / (1 + e^{-(\mathbf{z}^\top \mathbf{x}_a + c)}), p_b = 1 / (1 + e^{-(\mathbf{z}^\top \mathbf{x}_b + c)})$ in 28 gives,

$$
\begin{array}{l} = \frac {\mathbf {x} _ {b} - \mathbf {x} _ {a}}{1 + e ^ {\Delta}} \Big (\frac {(e ^ {\mathbf {z} ^ {\top} \mathbf {x} _ {b} - \mathbf {z} ^ {\top} \mathbf {x} _ {a}}) (e ^ {- (\mathbf {z} ^ {\top} \mathbf {x} _ {b} + c)})}{(1 + e ^ {- (\mathbf {z} ^ {\top} \mathbf {x} _ {a} + c)}) (1 + e ^ {- (\mathbf {z} ^ {\top} \mathbf {x} _ {b} + c)})} - \frac {(e ^ {- (\mathbf {z} ^ {\top} \mathbf {x} _ {a} + c)})}{(1 + e ^ {- (\mathbf {z} ^ {\top} \mathbf {x} _ {a} + c)}) (1 + e ^ {- (\mathbf {z} ^ {\top} \mathbf {x} _ {b} + c)})} \Big) \\ = 0 \tag {29} \\ \end{array}
$$

From 29 we see that the derivative of the loss is 0 when $p_{a} = 1/(1 + e^{-(\mathbf{z}^{\top}\mathbf{x}_{a}+c)}), p_{b} = 1/(1 + e^{-(\mathbf{z}^{\top}\mathbf{x}_{b}+c)})$ thus minimizing the expected loss. Note that this calibration strategy is in line with posthoc calibration strategies discussed in Platt (2000), where a model is learned, and then a parametrized sigmoid function is fit to learn the relevance probabilities. □

# B DATASET DETAILS

# B.1 DATASET STATISTICS

Table 3 shows the statistics of benchmark datasets including the newly contributed multi-intent datasets.

Table 3: Dataset Statistics. Pos-80% is an imbalance metric (Schultheis et al., 2022) defined as minimum fraction of class labels that retain 80% of all positive labels in the dataset. Lower value corresponds to higher skew.

<table><tr><td></td><td>Dataset</td><td>Train Docs</td><td>Test Docs</td><td>Labels</td><td>Avg. Labels/Doc</td><td>Avg. Docs/Label</td><td>Pos-80%</td></tr><tr><td rowspan="5">Existing</td><td>LF-AmazonTitles-131K</td><td>294,805</td><td>134,835</td><td>131,073</td><td>2.29</td><td>5.15</td><td>47.5</td></tr><tr><td>LF-Amazon-131K</td><td>294,805</td><td>134,835</td><td>131,073</td><td>2.29</td><td>5.15</td><td>47.5</td></tr><tr><td>LF-WikiSeeAlso-320K</td><td>693,082</td><td>177,515</td><td>312,330</td><td>2.11</td><td>4.68</td><td>37.4</td></tr><tr><td>LF-Wikipedia-500K</td><td>1,813,391</td><td>783,743</td><td>501,070</td><td>4.77</td><td>24.75</td><td>25.1</td></tr><tr><td>LF-AmazonTitles-1.3M</td><td>2,248,619</td><td>970,237</td><td>1,305,265</td><td>22.20</td><td>38.24</td><td>28.9</td></tr><tr><td rowspan="2">New</td><td>LF-AOL-270K</td><td>3,922,479</td><td>519,352</td><td>272,825</td><td>2.01</td><td>28.83</td><td>11.6</td></tr><tr><td>LF-WikiHierarchy-1M</td><td>1,589,378</td><td>397,952</td><td>976,214</td><td>25.98</td><td>42.31</td><td>7.3</td></tr></table>

# B.2 QK-20M DATASET

Query Keyword (QK) matching is an essential element in applications such as sponsored search. In these applications, users express their intent by querying a search engine, while advertisers bid on relevant phrases from the same domain, referred to as keywords. The retrieval (or matching system) is responsible for matching user queries to relevant advertisements. To train a query-keyword matching system, we build a dataset using click logs from Bing. We begin by considering 20M popular advertiser bid phrases (or keywords), and then corresponding to each keyword we add relevant queries to the ground truth based on whether there was a user click on the keyword when the user searched for that particular query.

# B.3 MULTI-INTENT DATASET PREPARATION

# B.3.1 LF-AOL-270K

Task Description: Query auto-completion involves matching a query prefix to completing suffixes, e.g. given a prefix, ‘cheap nike s’ recommending suffix completions like ‘shoes’, ‘shirts’ etc. LF-AOL-270K is curated from AOL search logs (Pass et al., 2006) for the task of query auto-completion where (prefix, suffix) pairs are modeled as (doc, label) pairs. Retrieved suffixes from this task can be combined with user prefixes to get full query completions as proposed in (Mitra & Craswell, 2015).

Dataset generation: The dataset generation process involved three steps (i) Pre-processing, (ii) Prefix-suffix generation, and (iii) Post-processing.

Pre-processing: Queries in AOL search logs were de-duplicated and non-alphanumeric characters were removed. Queries with less than three characters were filtered since auto-completion is rarely required for those. Additionally, steps prescribed in (Kim, 2019) were followed for pre-processing and train-test splits creation.

Prefix-suffix generation: After pre-processing, a shortlist of the top 10M popular suffixes was derived from the train split, based on their frequency in queries. These suffixes are popular n-grams (word-level) up to 100 characters appearing at the end of queries. Sampling was done to ensure that each train query has at least one suffix from 10M suffix shortlist. Ground truth suffixes were added for sampled prefixes yielding 9.3M suffixes and 5.67M distinct prefixes (training points). Using the 10M suffix shortlist, the process of sampling prefixes was repeated in the test split, resulting in 460K suffixes.

Post-processing: Train-test leakage was avoided by removing all prefixes in the test set that appeared in the train set. The suffix (label) set is derived from the intersection of train and test suffixes to have a fixed label set. Finally, the dataset contains 272K labels (suffixes), 3.9M training points (prefixes), and 519K test points (test prefixes).

Code to create the dataset from raw AOL search logs is available here $^{1}$ .

# B.3.2 LF-WIKIHIERARCHY-1M

Task description: Taxonomy completion task involves matching a category with its generalized parent categories. LF-WikiHierarchy-1M uses Wikipedia categories to build a taxonomy completion task where documents are categories and labels are its parent categories. Articles in Wikipedia are assigned categories, which serve as semantic tags. (e.g. ‘FIFA World Cup 2022’ article has a category tag of ‘Football’). These categories are arranged in a taxonomy-like structure where each category is linked to zero or more parent categories. The parent of a category is its direct generalization, e.g. the category ‘Football’ has direct parent categories ‘Athletic Sports’, ‘Team sports’ and ‘Ball Games’. This taxonomy-like structure is called Wikipedia Category graph (WCG) and has been well studied in (Zesch & Gurevych, 2007; Benaouicha et al., 2016).

Dataset generation The dataset generation process involved four steps (i) Raw data collection, (ii) Pre-processing, (iii) Label set generation from WCG and (iv) Post-processing.

Raw data collection: The WCG is created by using the English Wikimedia dump as of 03/23 $^{2}$ . The dump contains the list of all Wikipedia categories and their links.

Pre-processing: To create the WCG we first filter out all meta categories used for Wikipedia maintenance, e.g. ‘Wikipedia missing topics’, ‘Wikipedia new articles’, ‘Categories for renaming’ etc. The complete list of filtered meta-categories are released as part of the code. Post filtering, the resulting WCG is a directed acyclic graph with 1,993,526 categories (nodes) and 5,781,016 edges. Each edge is a document-label pair.

Label set generation from WCG: The WCG in its current form only contains direct parents and misses out on potentially important ground truth information. For example, the category ‘Football’ will not have categories like ‘Sports’, ‘Athletic Sports’, and ‘Team activities’ as its labels as they are not its direct parents. On the other hand, adding all reachable nodes as labels leads to vague document-label pairs. For example, starting from ‘Football’ one can reach the category ‘Cosmopolitan mammals’ as follows: ‘Football’ → ‘Athletic sports’ → ‘Sports by type’ → ‘Sports’ → ‘Entertainment’ → ‘Human activities’ → ‘Humans’ → ‘Cosmopolitan mammals’.

To maximize the relevant ground truth document-label pairs while also avoiding wrong matches like ‘Football’ → ‘Cosmopolitan mammals’ we limit the traversal to a maximum depth of 3 which gave the optimal trade-off (i.e. maximize true positives while avoiding false positives). Thus, in the above example, only categories up to ‘Sports’ are added as labels. Please refer to Fig. 1 for more clarity. Subsequently, we get 1,987,330 documents and 976,214 labels with 51,643,812 edges between them. Note that both the number of documents and labels are less than the total number of categories (1,993,526). Some categories will not have a parent and therefore won’t be added as a document. Similarly, categories that are not parents of any category will not be added as labels. The final dataset is created by taking an 80%-20% random train-test split.

Post-processing: The dataset contains categories that occur as both documents and labels. For example, the category ‘Football’ occurs both as a label (for ‘Football clubs’, ‘History of Football’, etc), and as a document. Since a category will never have itself as a label we filter off pairs like

![](images/49b2ef1128545e35a33a2f2e60c78c0fc1433e91c8e9502523f903b7c3a1b48f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Team sports"] --> B["Physical Skill"]
    A --> C["Team activities"]
    A --> D["Athletic sports"]
    A --> E["Ball game"]
    F["Football"] --> D
    F --> E
    G["Teams"] --> H["Sports by type"]
    I["Games"] --> J["Games by type"]
    H --> I
    J --> K["Balls"]
    style A fill:#f9f,stroke:#333
    style F fill:#ccf,stroke:#333
    style G fill:#cfc,stroke:#333
    style I fill:#fcc,stroke:#333
    style J fill:#cff,stroke:#333
    style K fill:#ffc,stroke:#333
```
</details>

Figure 1: Snapshot of WCG graph starting from category ‘Football’. All categories (nodes) reachable from ‘Football’ till the depth of 3 are added to its ground truth label set.

![](images/79a4c00ceae723aea2ac2bba5f9bf5315a01141fbb03013c207fa5843df25647.jpg)

<details>
<summary>line</summary>

| Avg Training Points per Label | Encoder | OvA (Renée) |
| ----------------------------- | ------- | ----------- |
| 2^6                           | 2.3     | 2.6         |
| 2^4                           | 1.5     | 1.7         |
| 2^2                           | 1.8     | 2.0         |
| 2^4                           | 4.5     | 4.2         |
</details>

(a)

![](images/17ba7b11499e1d8feaa6254b9003221b853698b45d283d698d0accf145d0674f.jpg)

<details>
<summary>line</summary>

| Avg Training Points per Label | Encoder | OvA (Renée) |
| ----------------------------- | ------- | ----------- |
| 2^17                          | 0.5     | 2.7         |
| 2^13                          | 0.8     | 1.8         |
| 2^9                           | 0.7     | 1.4         |
| 2^5                           | 0.9     | 1.1         |
| 2^4                           | 4.5     | 2.2         |
</details>

(b)

![](images/a47fb71976c43883c34a7591b5818b17e7784834ce81b53724b65a820a78120c.jpg)

<details>
<summary>line</summary>

| Avg Training Points per Label | Encoder | OvA (Renée) |
| ----------------------------- | ------- | ----------- |
| 2^11                          | 6.8     | 8.3         |
| 2^9                           | 4.3     | 6.5         |
| 2^7                           | 3.5     | 4.5         |
| 2^5                           | 4.0     | 4.3         |
| 2^3                           | 4.8     | 4.2         |
| 2^1                           | 7.5     | 4.2         |
</details>

(c)

![](images/6b450cba2eff810d8737bd8bab556151b187e322c3dee584631d02ec96c1f3a6.jpg)

<details>
<summary>line</summary>

| Avg Training Points per Label | Encoder | OvA (Renée) |
| ----------------------------- | ------- | ----------- |
| 2^13                          | 2       | 17          |
| 2^10                          | 2       | 12          |
| 2^7                           | 6       | 8           |
| 2^4                           | 19      | 3           |
</details>

(d)   
Figure 2: P@5 comparison of Siamese Encoder (blue) and OvA Classifier Renée (orange) on homogeneous (LF-AmazonTitles-131K: Fig. 2a, LF-Wikipedia-500K: Fig. 2c) and heterogeneous datasets (LF-AOL-270K: Fig. 2b, LF-WikiHierarchy-1M: Fig. 2d). Labels are partitioned into equi-volume bins based on their frequencies along the X-axis. The difference in performance (on both head and tail) is wider for heterogeneous datasets.

‘Football’→‘Football’ during evaluation so as to not unfairly penalize Siamese-based models that rank such pairs at the top.

The LF-WikiHierarchy-1M dataset is available here $^{3}$

# C ADDITIONAL RESULTS

# C.1 LEVER'S PERFORMANCE ON NON-DNN METHODS

Table 4 illustrates the performance of LEVER when combined with XReg (Prabhu et al., 2020), an extension of Parabel, showcasing that LEVER can effectively combine with non-DNN-based methods.

Table 4: Performance Comparison of XReg and XReg + LEVER on LF-AOL-270K and LF-AmazonTitles-131K 

<table><tr><td>Dataset</td><td>Model</td><td>P@1</td><td>P@3</td><td>P@5</td><td>PSP@1</td><td>PSP@3</td><td>PSP@5</td><td>C@1</td><td>C@3</td><td>C@5</td></tr><tr><td rowspan="2">LF-AmazonTitles-131K</td><td>XReg</td><td>33.1</td><td>22.3</td><td>16.0</td><td>24.5</td><td>29.4</td><td>33.54</td><td>20.22</td><td>36.63</td><td>42.84</td></tr><tr><td>XReg + LEVER</td><td>38.0</td><td>24.7</td><td>17.6</td><td>31.4</td><td>35.1</td><td>39.1</td><td>25.84</td><td>43.47</td><td>49.68</td></tr><tr><td rowspan="2">LF-AOL-270K</td><td>XReg</td><td>27.0</td><td>14.3</td><td>9.9</td><td>7.0</td><td>11.0</td><td>14.1</td><td>4.18</td><td>10.58</td><td>14.44</td></tr><tr><td>XReg + LEVER</td><td>26.1</td><td>14.3</td><td>10.1</td><td>9.2</td><td>17.9</td><td>24.0</td><td>6.79</td><td>20.59</td><td>28.65</td></tr></table>

# C.2 COMPARISON WITH SOTA AND TAIL XC METHODS

Table 5 demonstrates the enhanced performance achieved by applying LEVER to top-performing Extreme Classification (XC) methods, including ELIAS, CascadeXML, and Renée. On average PSP metrics are boosted by 5%, Coverage improves by 6.5% and Precision improves by 1.4%.

Table 7 presents a comparison of LEVER with various OvA-based methods (XR-Transformer, ELIAS, and CascadeXML) and Siamese encoder methods (NGAME and ECLARE). It's worth noting that for WikiHierarchy-1M, OvA and Siamese approaches exhibit significant trade-offs between precision and tail metrics.

# C.3 COMPARISON WITH ENSEMBLE BETWEEN TAIL EXPERT AND OVA CLASSIFIER

In the ensemble model, for each data point, the encoder and OvA model provide a shortlist of top-k labels along with their prediction scores. These two shortlists (containing a total of up to 2k labels) need to be combined into a single shortlist of k labels by tie-breaking as elaborated below. First, the labels with a frequency more than the cut-off are considered from the OvA's shortlist. Similarly, the labels with a frequency less than the cut-off are considered from the encoder's shortlist. Cut-offs are derived on the basis of the cross-over points between Encoder and Renée in the decile wise plots shown in Fig. 2. Then, the two resulting shortlists are combined by considering the assigned label scores from both models and retaining only the k overall highest-scoring labels. Table 8 shows that LEVER clearly outperforms the ensemble model in 3 out of 4 datasets across all metrics. In the case of LF-WikiHierarchy-1M, the ensemble model shows gains in coverage metrics ( $\sim$ 4-5%), this comes at the expense of a significant loss in Precision ( $\sim$ 30%). Figure 3 compares the performance of LEVER with the ensemble model and here we see a clear dip in the torso deciles. To better understand why the ensemble curve doesn't exactly mimic the OvA curve before the cutoff and encoder curve after the cutoff, consider the following toy example:

Assume a dataset D with 8 labels which are partitioned into 3 deciles (head, torso, and tail deciles). Out of 8 labels, 3 belong to the head decile $(H_{1}, H_{2}, H_{3})$ , 2 belong to the torso decile $(O_{1}, O_{2})$ and the remaining 3 belong to the tail decile $(T_{1}, T_{2}, T_{3})$ . The cut-off threshold partitions the label set into 2 sets: (i) labels with frequency greater than cut-off: $(H_{1}, H_{2}, H_{3}, O_{2})$ and labels with frequency less than cut-off: $(O_{1}, T_{1}, T_{2}, T_{3})$ . Assume a data point d has ground truth labels: $(H_{1}, H_{2}, O_{1}, O_{2}, T_{1})$ .

Below we list the predictions of different models in the format of “label ID:model score”

Top-5 encoder predictions ( $T_{1}:0.8,T_{2}:0.6,T_{3}:0.4,O_{1}:0.2,O_{2}:0.1$ ).

Top-5 OvA predictions ( $H_{1}:0.7,H_{2}:0.5,H_{3}:0.3,O_{2}:0.2,O_{1}:0.1$ )

To compute the ensemble model predictions, we first restrict the predictions of the individual models based on the cutoff frequency, i.e. Encoder's predictions are restricted to $(O_{1}, T_{1}, T_{2}, T_{3})$ and OvA predictions are restricted to $(H_{1}, H_{2}, H_{3}, O_{2})$ . This gives the following filtered shortlists:

Table 5: Using LEVER with leading OvA approaches improves their tail label performance consistently across benchmarks, with an average gain of 5% in PSP and 6.5% in coverage (C), while maintaining comparable precision (P) with an average gain of 1.4%. 

<table><tr><td rowspan="2">Model</td><td colspan="9">LF-AmazonTitles-131K</td></tr><tr><td>P@1</td><td>P@3</td><td>P@5</td><td>PSP@1</td><td>PSP@3</td><td>PSP@5</td><td>C@1</td><td>C@3</td><td>C@5</td></tr><tr><td>ELIAS</td><td>37.28</td><td>25.18</td><td>18.14</td><td>28.95</td><td>34.45</td><td>39.08</td><td>23.73</td><td>42.36</td><td>49.06</td></tr><tr><td>ELIAS + LEVER</td><td>42.86</td><td>28.37</td><td>20.16</td><td>36.30</td><td>41.05</td><td>45.43</td><td>29.81</td><td>49.88</td><td>56.25</td></tr><tr><td>CascadeXML</td><td>36.28</td><td>24.88</td><td>18.18</td><td>26.50</td><td>33.21</td><td>38.81</td><td>21.38</td><td>40.58</td><td>48.46</td></tr><tr><td>CascadeXML + LEVER</td><td>43.58</td><td>28.79</td><td>20.63</td><td>36.24</td><td>41.83</td><td>46.95</td><td>29.43</td><td>50.61</td><td>57.90</td></tr><tr><td>Renée</td><td>46.05</td><td>30.81</td><td>22.04</td><td>38.47</td><td>44.87</td><td>50.33</td><td>31.31</td><td>53.50</td><td>61.03</td></tr><tr><td>Renée + LEVER</td><td>46.44</td><td>30.83</td><td>21.92</td><td>39.70</td><td>45.44</td><td>50.31</td><td>32.50</td><td>54.59</td><td>61.42</td></tr><tr><td></td><td colspan="9">LF-Amazon-131K</td></tr><tr><td>ELIAS</td><td>43.03</td><td>29.27</td><td>21.20</td><td>33.49</td><td>40.80</td><td>46.76</td><td>27.04</td><td>49.10</td><td>57.34</td></tr><tr><td>ELIAS + LEVER</td><td>47.38</td><td>32.24</td><td>23.22</td><td>38.97</td><td>46.74</td><td>52.79</td><td>31.47</td><td>55.27</td><td>63.40</td></tr><tr><td>CascadeXML</td><td>43.76</td><td>29.75</td><td>21.58</td><td>34.05</td><td>41.69</td><td>47.96</td><td>27.30</td><td>50.18</td><td>58.81</td></tr><tr><td>CascadeXML + LEVER</td><td>48.24</td><td>32.82</td><td>23.73</td><td>39.09</td><td>47.55</td><td>54.18</td><td>31.26</td><td>55.97</td><td>64.81</td></tr><tr><td>Renée</td><td>48.05</td><td>32.33</td><td>23.26</td><td>39.32</td><td>47.10</td><td>53.51</td><td>31.49</td><td>55.81</td><td>64.61</td></tr><tr><td>Renée + LEVER</td><td>49.19</td><td>33.30</td><td>24.04</td><td>40.64</td><td>48.48</td><td>54.87</td><td>32.39</td><td>56.81</td><td>65.20</td></tr><tr><td></td><td colspan="9">LF-WikiSeeAlso-320K</td></tr><tr><td>ELIAS</td><td>41.40</td><td>27.36</td><td>20.66</td><td>23.83</td><td>28.38</td><td>31.90</td><td>13.30</td><td>27.72</td><td>35.50</td></tr><tr><td>ELIAS + LEVER</td><td>45.99</td><td>30.28</td><td>22.78</td><td>30.00</td><td>34.16</td><td>37.52</td><td>16.56</td><td>33.06</td><td>41.34</td></tr><tr><td>CascadeXML</td><td>30.21</td><td>18.72</td><td>14.05</td><td>12.46</td><td>14.15</td><td>16.25</td><td>6.70</td><td>13.38</td><td>17.72</td></tr><tr><td>CascadeXML + LEVER</td><td>38.84</td><td>25.43</td><td>19.36</td><td>21.62</td><td>25.85</td><td>29.45</td><td>12.01</td><td>25.12</td><td>32.59</td></tr><tr><td>Renée</td><td>47.79</td><td>31.73</td><td>23.82</td><td>31.13</td><td>36.49</td><td>40.37</td><td>17.02</td><td>35.32</td><td>44.56</td></tr><tr><td>Renée + LEVER</td><td>47.89</td><td>31.52</td><td>23.53</td><td>32.44</td><td>37.45</td><td>40.99</td><td>17.78</td><td>36.33</td><td>45.31</td></tr><tr><td></td><td colspan="9">LF-Wikipedia-500K</td></tr><tr><td>ELIAS</td><td>81.94</td><td>62.71</td><td>48.75</td><td>33.58</td><td>43.92</td><td>48.67</td><td>19.62</td><td>41.30</td><td>51.36</td></tr><tr><td>ELIAS + LEVER</td><td>82.44</td><td>63.88</td><td>50.03</td><td>36.94</td><td>49.28</td><td>55.03</td><td>23.55</td><td>50.81</td><td>63.04</td></tr><tr><td>CascadeXML</td><td>77.00</td><td>58.30</td><td>45.10</td><td>31.25</td><td>39.35</td><td>43.29</td><td>15.78</td><td>33.07</td><td>41.46</td></tr><tr><td>CascadeXML + LEVER</td><td>80.10</td><td>60.41</td><td>46.44</td><td>36.79</td><td>46.65</td><td>50.99</td><td>23.99</td><td>49.16</td><td>60.13</td></tr><tr><td>Renée</td><td>84.95</td><td>66.25</td><td>51.68</td><td>37.10</td><td>50.27</td><td>55.68</td><td>22.90</td><td>50.08</td><td>61.59</td></tr><tr><td>Renée + LEVER</td><td>85.02</td><td>66.37</td><td>51.98</td><td>42.93</td><td>55.00</td><td>60.29</td><td>29.46</td><td>58.53</td><td>70.29</td></tr><tr><td></td><td colspan="9">LF-AOL-270K</td></tr><tr><td>ELIAS</td><td>40.83</td><td>22.33</td><td>14.91</td><td>13.29</td><td>21.46</td><td>25.22</td><td>10.46</td><td>22.85</td><td>27.06</td></tr><tr><td>ELIAS + LEVER</td><td>40.85</td><td>22.83</td><td>15.57</td><td>13.68</td><td>24.30</td><td>30.43</td><td>10.52</td><td>26.33</td><td>33.56</td></tr><tr><td>CascadeXML</td><td>41.20</td><td>22.12</td><td>14.82</td><td>12.58</td><td>19.53</td><td>23.19</td><td>8.73</td><td>19.47</td><td>23.47</td></tr><tr><td>CascadeXML + LEVER</td><td>39.41</td><td>21.78</td><td>14.99</td><td>11.96</td><td>21.30</td><td>27.59</td><td>7.86</td><td>22.11</td><td>29.89</td></tr><tr><td>Renée</td><td>40.97</td><td>23.34</td><td>15.85</td><td>14.76</td><td>26.45</td><td>32.19</td><td>12.40</td><td>29.77</td><td>36.53</td></tr><tr><td>Renée + LEVER</td><td>41.70</td><td>24.76</td><td>17.07</td><td>20.38</td><td>37.07</td><td>45.13</td><td>17.43</td><td>42.54</td><td>52.01</td></tr><tr><td></td><td colspan="9">LF-WikiHierarchy-1M</td></tr><tr><td>ELIAS</td><td>95.27</td><td>94.25</td><td>92.45</td><td>17.15</td><td>24.41</td><td>30.01</td><td>4.00</td><td>7.78</td><td>10.49</td></tr><tr><td>ELIAS + LEVER</td><td>94.02</td><td>91.97</td><td>89.50</td><td>28.27</td><td>36.80</td><td>42.13</td><td>10.78</td><td>18.88</td><td>23.03</td></tr><tr><td>CascadeXML</td><td>94.88</td><td>93.69</td><td>91.79</td><td>16.03</td><td>22.87</td><td>28.17</td><td>3.12</td><td>6.17</td><td>8.52</td></tr><tr><td>CascadeXML + LEVER</td><td>94.77</td><td>93.54</td><td>91.56</td><td>20.14</td><td>27.49</td><td>33.01</td><td>6.68</td><td>11.22</td><td>14.13</td></tr><tr><td>Renée</td><td>95.01</td><td>93.99</td><td>92.24</td><td>19.69</td><td>27.36</td><td>33.20</td><td>6.72</td><td>11.49</td><td>14.65</td></tr><tr><td>Renée + LEVER</td><td>95.19</td><td>93.90</td><td>92.07</td><td>24.76</td><td>32.63</td><td>38.15</td><td>9.32</td><td>16.14</td><td>20.29</td></tr><tr><td></td><td colspan="9">LF-AmazonTitles-1.3M</td></tr><tr><td>ELIAS</td><td>47.48</td><td>42.21</td><td>38.60</td><td>18.79</td><td>23.20</td><td>26.06</td><td>11.53</td><td>21.45</td><td>27.33</td></tr><tr><td>ELIAS + LEVER</td><td>48.91</td><td>43.17</td><td>39.28</td><td>23.68</td><td>27.43</td><td>29.72</td><td>15.10</td><td>26.65</td><td>32.84</td></tr><tr><td>CascadeXML</td><td>47.14</td><td>41.43</td><td>37.73</td><td>15.92</td><td>20.23</td><td>23.16</td><td>8.65</td><td>16.75</td><td>21.95</td></tr><tr><td>CascadeXML + LEVER</td><td>47.98</td><td>42.02</td><td>38.12</td><td>20.06</td><td>24.51</td><td>27.28</td><td>12.36</td><td>22.57</td><td>28.52</td></tr><tr><td>Renée</td><td>56.10</td><td>49.91</td><td>45.32</td><td>28.56</td><td>33.38</td><td>36.14</td><td>17.31</td><td>30.60</td><td>37.59</td></tr><tr><td>Renée + LEVER</td><td>56.01</td><td>49.43</td><td>44.85</td><td>33.55</td><td>36.82</td><td>38.81</td><td>21.03</td><td>35.70</td><td>42.78</td></tr></table>

Encoder: $(T_{1}:0.8,T_{2}:0.6,T_{3}:0.4,O_{1}:0.2)$

OvA: $(H_{1}:0.7,H_{2}:0.5,H_{3}:0.3,O_{2}:0.2)$

Next, we combine and sort the labels based on the scores from both the encoder and OvA as follows:

$(T_{1}:0.8,H_{1}:0.7,T_{2}:0.6,H_{2}:0.5,T_{3}:0.4,H_{3}:0.3,O_{2}:0.2,O_{1}:0.2)$

Finally, we retain only the top-5 highest scoring labels as our final ensemble predictions:

Table 6: Using LEVER with leading OvA approaches improves their tail performance consistently across benchmarks in Macro-F1 (+4.3% on avg.), Macro-precision (+4.1% on avg.), and Macro-Recall (+5.49% on avg.). Following Zhang et al. (2023), k values for datasets were chosen based on average labels per point for the dataset. We use k=3 for LF-AmazonTitles-131K, LF-Amazon-131K, LF-WikiSeeAlso-320K, LF-AOL-270K, k=5 for LF-Wikipedia-500K, k=25 for LF-AmazonTitles-1.3M and LF-WikiHierarchy-1M. 

<table><tr><td rowspan="2">Model</td><td colspan="3">LF-AmazonTitles-131K</td><td colspan="3">LF-Amazon-131K</td><td colspan="3">LF-Wikipedia-500K</td></tr><tr><td>F1@k</td><td>P@k</td><td>R@k</td><td>F1@k</td><td>P@k</td><td>R@k</td><td>F1@k</td><td>P@k</td><td>R@k</td></tr><tr><td>ELIAS</td><td>24.23</td><td>22.82</td><td>31.70</td><td>28.48</td><td>26.46</td><td>37.93</td><td>27.98</td><td>27.97</td><td>35.17</td></tr><tr><td>ELIAS + LEVER</td><td>29.53</td><td>27.70</td><td>38.21</td><td>33.36</td><td>31.16</td><td>43.35</td><td>34.12</td><td>33.88</td><td>44.68</td></tr><tr><td>CascadeXML</td><td>22.23</td><td>20.80</td><td>30.36</td><td>28.89</td><td>26.74</td><td>38.83</td><td>20.04</td><td>20.13</td><td>26.75</td></tr><tr><td>CascadeXML + LEVER</td><td>29.12</td><td>26.91</td><td>39.04</td><td>33.29</td><td>30.76</td><td>44.13</td><td>31.40</td><td>31.61</td><td>41.70</td></tr><tr><td>Renée</td><td>32.19</td><td>30.36</td><td>41.55</td><td>32.55</td><td>29.95</td><td>43.88</td><td>35.94</td><td>36.78</td><td>43.76</td></tr><tr><td>Renée + LEVER</td><td>33.35</td><td>31.78</td><td>42.45</td><td>35.69</td><td>34.05</td><td>45.04</td><td>40.54</td><td>40.71</td><td>51.36</td></tr><tr><td rowspan="2">Model</td><td colspan="3">LF-AmazonTitles-1.3M</td><td colspan="3">LF-AOL-270K</td><td colspan="3">LF-WikiHierarchy-1M</td></tr><tr><td>F1@k</td><td>P@k</td><td>R@k</td><td>F1@k</td><td>P@k</td><td>R@k</td><td>F1@k</td><td>P@k</td><td>R@k</td></tr><tr><td>ELIAS</td><td>16.43</td><td>15.73</td><td>24.68</td><td>10.69</td><td>9.45</td><td>16.18</td><td>20.71</td><td>23.26</td><td>22.43</td></tr><tr><td>ELIAS + LEVER</td><td>18.85</td><td>17.92</td><td>28.58</td><td>13.97</td><td>12.87</td><td>19.41</td><td>26.99</td><td>28.75</td><td>30.53</td></tr><tr><td>CascadeXML</td><td>13.58</td><td>13.23</td><td>22.06</td><td>9.00</td><td>8.25</td><td>13.46</td><td>17.73</td><td>20.91</td><td>19.22</td></tr><tr><td>CascadeXML + LEVER</td><td>16.02</td><td>15.07</td><td>27.02</td><td>11.59</td><td>11.03</td><td>15.74</td><td>21.17</td><td>24.36</td><td>22.85</td></tr><tr><td>Renée</td><td>24.79</td><td>24.16</td><td>35.42</td><td>17.70</td><td>16.88</td><td>22.53</td><td>24.48</td><td>27.59</td><td>26.04</td></tr><tr><td>Renée + LEVER</td><td>26.72</td><td>25.73</td><td>37.04</td><td>22.38</td><td>20.40</td><td>30.18</td><td>27.89</td><td>30.67</td><td>30.23</td></tr></table>

<table><tr><td>Model</td><td colspan="3">LF-WikiSeeAlso-320K</td></tr><tr><td></td><td>F1@k</td><td>P@k</td><td>R@k</td></tr><tr><td>ELIAS</td><td>17.67</td><td>16.83</td><td>22.46</td></tr><tr><td>ELIAS + LEVER</td><td>22.06</td><td>21.21</td><td>27.26</td></tr><tr><td>CascadeXML</td><td>5.86</td><td>5.20</td><td>9.78</td></tr><tr><td>CascadeXML + LEVER</td><td>14.81</td><td>13.90</td><td>20.00</td></tr><tr><td>Renée</td><td>23.28</td><td>22.17</td><td>29.30</td></tr><tr><td>Renée + LEVER</td><td>23.79</td><td>22.70</td><td>30.12</td></tr></table>

Ensemble predictions: $(T_{1}:0.8,H_{1}:0.7,T_{2}:0.6,H_{2}:0.5,T_{3}:0.4)$ Table 9 shows the contribution to P@5 for different models across the three deciles. Note that the example is in line with our observations in Fig. 3 where (i) Encoder performs better on tail deciles (blue curve), (ii) OvA models perform better on head deciles (orange), (iii) Ensemble (green) between Encoder and OvA models perform comparably to Encoders on tail deciles and OvA based models on head deciles but incurs significant losses in torso deciles, (iv) Performance of the ensemble model can be worse than the individual models (e.g. ensemble P@5 < OvA P@5 in toy example). If the Ensemble model were to dominate both Encoder and OvA models it should have achieved decile-wise contributions of $(2/5,2/5,1/5)$ which is not the case. On average, the torso labels are ranked relatively lower by both models since neither model specializes in them. Further, when combined using the proposed ensemble these labels get more aggressively down-voted.

# C.4 EFFECT OF RE-RANKING ON LEVER AND OTHER TAIL XC APPROACHES

Table 10 illustrates the impact of post-hoc reranking using inverse propensity scores, on LEVER and other Tail XC approaches. The application of reranking shows varying degrees of trade-offs between precision and tail metrics across different models. Notably, while LEVER attains superior performance in tail metrics for three out of four datasets, there is a trade-off in precision compared to other methods in the LF-WikiHierarchy-1M dataset.

# C.5 ABLATIONS

Effect of Teacher Model: Table 11 demonstrates the impact of employing various encoders as a tail expert. We conduct a comparison with two alternative encoders: (i) MiniLM, a 3-layer transformer

Table 7: Comparison between LEVER and leading OvA-based methods such as XR-Transformer, ELIAS, and CascadeXML, as well as Siamese encoder-based methods like NGAME and ECLARE. Note that for LF-WikiHierarchy-1M, OvA and Siamese-based methods display significant trade-offs between precision and tail metrics. Siamese-based methods score much higher in PSP numbers (+16 on average) but lag behind in precision (-14 on average) when compared to OvA-based methods. 

<table><tr><td></td><td></td><td>P@1</td><td>P@3</td><td>P@5</td><td>PSP@1</td><td>PSP@3</td><td>PSP@5</td><td>C@1</td><td>C@3</td><td>C@5</td></tr><tr><td></td><td></td><td colspan="9">LF-AmazonTitles-131K</td></tr><tr><td rowspan="6">SOTA XC Methods</td><td>XR-Transformer</td><td>38.10</td><td>25.57</td><td>18.32</td><td>28.86</td><td>34.85</td><td>39.59</td><td>20.24</td><td>40.70</td><td>48.87</td></tr><tr><td>ELIAS</td><td>37.28</td><td>25.18</td><td>18.14</td><td>28.95</td><td>34.45</td><td>39.08</td><td>23.73</td><td>42.36</td><td>49.06</td></tr><tr><td>CascadeXML</td><td>36.28</td><td>24.88</td><td>18.18</td><td>26.50</td><td>33.21</td><td>38.81</td><td>21.38</td><td>40.58</td><td>48.46</td></tr><tr><td>ECLARE</td><td>41.40</td><td>27.58</td><td>19.82</td><td>34.22</td><td>39.69</td><td>44.63</td><td>27.91</td><td>48.38</td><td>55.48</td></tr><tr><td>NGAME</td><td>46.58</td><td>30.41</td><td>21.49</td><td>39.54</td><td>44.77</td><td>49.59</td><td>32.35</td><td>53.78</td><td>60.73</td></tr><tr><td>Renée</td><td>46.05</td><td>30.81</td><td>22.04</td><td>38.47</td><td>44.87</td><td>50.33</td><td>31.31</td><td>53.50</td><td>61.03</td></tr><tr><td rowspan="6">Tail XC Methods</td><td>Renée +TAUG</td><td>44.34</td><td>29.73</td><td>21.15</td><td>36.49</td><td>42.83</td><td>47.85</td><td>29.47</td><td>51.52</td><td>58.68</td></tr><tr><td>Renée + BoW</td><td>42.95</td><td>29.18</td><td>21.03</td><td>36.96</td><td>42.86</td><td>48.09</td><td>30.03</td><td>51.78</td><td>59.17</td></tr><tr><td>Renée + L2Reg</td><td>45.19</td><td>29.92</td><td>21.29</td><td>38.47</td><td>44.23</td><td>49.24</td><td>31.66</td><td>53.65</td><td>60.80</td></tr><tr><td>Renée + GLaS</td><td>45.35</td><td>30.03</td><td>21.33</td><td>38.74</td><td>44.53</td><td>49.49</td><td>31.90</td><td>54.02</td><td>61.15</td></tr><tr><td>Renée + Gandalf</td><td>45.86</td><td>30.53</td><td>21.79</td><td>40.49</td><td>45.83</td><td>50.96</td><td>33.17</td><td>55.36</td><td>62.22</td></tr><tr><td>Renée + LEVER</td><td>46.44</td><td>30.83</td><td>21.92</td><td>39.70</td><td>45.44</td><td>50.31</td><td>32.50</td><td>54.59</td><td>61.42</td></tr><tr><td></td><td></td><td colspan="9">LF-AOL-270K</td></tr><tr><td rowspan="6">SOTA XC Methods</td><td>XR-Transformer</td><td>37.56</td><td>20.44</td><td>13.94</td><td>11.76</td><td>21.10</td><td>26.31</td><td>8.83</td><td>23.07</td><td>29.44</td></tr><tr><td>ELIAS</td><td>40.83</td><td>22.33</td><td>14.91</td><td>13.29</td><td>21.46</td><td>25.22</td><td>10.46</td><td>22.85</td><td>27.06</td></tr><tr><td>CascadeXML</td><td>41.20</td><td>22.12</td><td>14.82</td><td>12.58</td><td>19.53</td><td>23.19</td><td>8.73</td><td>19.47</td><td>23.74</td></tr><tr><td>ECLARE</td><td>28.53</td><td>16.18</td><td>11.55</td><td>10.11</td><td>18.69</td><td>24.58</td><td>7.41</td><td>20.59</td><td>28.11</td></tr><tr><td>NGAME</td><td>39.44</td><td>22.29</td><td>15.30</td><td>16.33</td><td>29.63</td><td>37.06</td><td>14.28</td><td>34.70</td><td>43.84</td></tr><tr><td>Renée</td><td>40.97</td><td>23.34</td><td>15.85</td><td>14.76</td><td>26.45</td><td>32.19</td><td>12.40</td><td>29.77</td><td>36.53</td></tr><tr><td rowspan="6">Tail XC Methods</td><td>Renée +TAUG</td><td>40.40</td><td>22.80</td><td>15.56</td><td>15.72</td><td>26.74</td><td>32.35</td><td>12.46</td><td>29.26</td><td>35.88</td></tr><tr><td>Renée + BoW</td><td>41.11</td><td>23.91</td><td>16.41</td><td>15.58</td><td>30.28</td><td>37.90</td><td>12.67</td><td>34.32</td><td>43.45</td></tr><tr><td>Renée + L2Reg</td><td>39.83</td><td>21.75</td><td>14.71</td><td>12.21</td><td>20.09</td><td>24.36</td><td>8.67</td><td>21.07</td><td>26.27</td></tr><tr><td>Renée + GLaS</td><td>40.91</td><td>23.25</td><td>15.78</td><td>14.67</td><td>26.11</td><td>31.75</td><td>12.36</td><td>29.41</td><td>36.06</td></tr><tr><td>Renée + Gandalf</td><td>40.63</td><td>23.01</td><td>15.58</td><td>15.10</td><td>26.64</td><td>32.17</td><td>12.63</td><td>29.82</td><td>36.31</td></tr><tr><td>Renée + LEVER</td><td>41.71</td><td>24.77</td><td>17.07</td><td>20.38</td><td>37.07</td><td>45.14</td><td>17.43</td><td>42.54</td><td>52.01</td></tr><tr><td></td><td></td><td colspan="9">LF-Wikipedia-500K</td></tr><tr><td rowspan="5">SOTA XC Methods</td><td>XR-Transformer</td><td>81.62</td><td>61.38</td><td>47.85</td><td>33.58</td><td>42.97</td><td>47.81</td><td>19.05</td><td>40.05</td><td>50.66</td></tr><tr><td>ELIAS</td><td>81.94</td><td>62.71</td><td>48.75</td><td>33.58</td><td>43.92</td><td>48.67</td><td>19.62</td><td>41.30</td><td>51.36</td></tr><tr><td>CascadeXML</td><td>77.00</td><td>58.3</td><td>45.10</td><td>31.25</td><td>39.35</td><td>43.29</td><td>15.78</td><td>33.07</td><td>41.46</td></tr><tr><td>NGAME</td><td>84.32</td><td>65.59</td><td>51.41</td><td>39.88</td><td>50.74</td><td>57.09</td><td>26.22</td><td>51.42</td><td>64.79</td></tr><tr><td>Renée</td><td>84.95</td><td>66.25</td><td>51.68</td><td>37.10</td><td>50.27</td><td>55.68</td><td>22.90</td><td>50.08</td><td>61.59</td></tr><tr><td rowspan="6">Tail XC Methods</td><td>Renée +TAUG</td><td>83.07</td><td>64.46</td><td>50.32</td><td>33.76</td><td>46.54</td><td>52.16</td><td>19.88</td><td>44.74</td><td>56.13</td></tr><tr><td>Renée + BoW</td><td>84.43</td><td>66.09</td><td>51.74</td><td>36.66</td><td>49.79</td><td>55.55</td><td>22.92</td><td>49.64</td><td>61.40</td></tr><tr><td>Renée + L2Reg</td><td>84.57</td><td>66.05</td><td>51.50</td><td>39.55</td><td>52.42</td><td>57.43</td><td>26.52</td><td>53.95</td><td>65.14</td></tr><tr><td>Renée + GLaS</td><td>84.85</td><td>66.63</td><td>52.09</td><td>37.27</td><td>51.54</td><td>57.15</td><td>23.43</td><td>52.02</td><td>63.90</td></tr><tr><td>Renée + Gandalf</td><td>84.59</td><td>66.07</td><td>51.63</td><td>37.05</td><td>49.94</td><td>55.31</td><td>23.09</td><td>49.87</td><td>61.24</td></tr><tr><td>Renée + LEVER</td><td>85.02</td><td>66.37</td><td>51.98</td><td>42.93</td><td>55.00</td><td>60.29</td><td>29.46</td><td>58.53</td><td>70.29</td></tr><tr><td></td><td></td><td colspan="9">LF-WikiHierarchy-1M</td></tr><tr><td rowspan="6">SOTA XC Methods</td><td>XR-Transformer</td><td>95.33</td><td>94.26</td><td>92.39</td><td>15.96</td><td>23.04</td><td>28.62</td><td>2.98</td><td>6.23</td><td>8.86</td></tr><tr><td>ELIAS</td><td>95.27</td><td>94.25</td><td>92.45</td><td>17.15</td><td>24.41</td><td>30.01</td><td>4.00</td><td>7.78</td><td>10.49</td></tr><tr><td>CascadeXML</td><td>94.88</td><td>93.69</td><td>91.79</td><td>16.03</td><td>22.87</td><td>28.17</td><td>3.12</td><td>6.17</td><td>8.52</td></tr><tr><td>ECLARE</td><td>90.95</td><td>89.14</td><td>86.90</td><td>15.70</td><td>22.41</td><td>27.65</td><td>2.57</td><td>5.94</td><td>9.30</td></tr><tr><td>NGAME</td><td>83.16</td><td>78.24</td><td>73.90</td><td>38.43</td><td>44.22</td><td>47.93</td><td>7.83</td><td>22.59</td><td>29.25</td></tr><tr><td>Renée</td><td>95.01</td><td>93.99</td><td>92.24</td><td>19.69</td><td>27.36</td><td>33.20</td><td>6.72</td><td>11.49</td><td>14.65</td></tr><tr><td rowspan="6">Tail XC Methods</td><td>Renée +TAUG</td><td>95.34</td><td>94.45</td><td>92.27</td><td>16.95</td><td>24.06</td><td>29.69</td><td>3.59</td><td>7.19</td><td>9.94</td></tr><tr><td>Renée + BoW</td><td>93.92</td><td>92.04</td><td>90.27</td><td>24.25</td><td>31.10</td><td>36.30</td><td>7.84</td><td>14.77</td><td>18.39</td></tr><tr><td>Renée + L2Reg</td><td>94.68</td><td>93.45</td><td>91.56</td><td>18.56</td><td>25.90</td><td>31.49</td><td>5.61</td><td>9.96</td><td>12.98</td></tr><tr><td>Renée + GLaS</td><td>95.01</td><td>93.98</td><td>92.26</td><td>20.07</td><td>27.82</td><td>33.70</td><td>6.89</td><td>11.82</td><td>15.08</td></tr><tr><td>Renée + Gandalf</td><td>93.01</td><td>90.85</td><td>88.16</td><td>21.84</td><td>30.05</td><td>36.09</td><td>6.92</td><td>13.17</td><td>17.52</td></tr><tr><td>Renée + LEVER</td><td>95.19</td><td>93.90</td><td>92.07</td><td>24.76</td><td>32.63</td><td>38.15</td><td>9.32</td><td>16.41</td><td>20.29</td></tr></table>

model, and (ii) Astec, which learns a projection matrix from sparse Bag of Words (BoW) features to a dense embedding space. The results highlight that the choice of a superior teacher substantially enhances the performance of LEVER.

Effect of sampling strategy: LEVER makes use of NGAME Module (Dahiya et al., 2023a) trained using mini-batches of labels instead of documents. The modification helps specialize the Siamese

Table 8: Comparison of LEVER with an ensemble of OvA and tail expert encoder. LEVER outperforms the ensemble consistently on all metrics for 3 out of 4 datasets. Note that for LF-WikiHierarchy-1M even though the ensemble improves coverage, the drop in precision is very large (31% on average). 

<table><tr><td rowspan="2"></td><td>P@1</td><td>P@3</td><td>P@5</td><td>PSP@1</td><td>PSP@3</td><td>PSP@5</td><td>C@1</td><td>C@3</td><td>C@5</td></tr><tr><td colspan="9">LF-AmazonTitles-131K</td></tr><tr><td>Ensemble</td><td>42.98</td><td>26.84</td><td>18.23</td><td>37.24</td><td>41.58</td><td>44.69</td><td>30.38</td><td>51.57</td><td>57.54</td></tr><tr><td>Renée + LEVER</td><td>46.44</td><td>30.83</td><td>21.92</td><td>39.70</td><td>45.44</td><td>50.31</td><td>32.82</td><td>55.11</td><td>61.94</td></tr><tr><td colspan="10">LF-AOL-270K</td></tr><tr><td>Ensemble</td><td>35.20</td><td>20.33</td><td>13.69</td><td>19.43</td><td>36.98</td><td>44.74</td><td>17.19</td><td>44.80</td><td>54.87</td></tr><tr><td>Renée + LEVER</td><td>41.71</td><td>24.77</td><td>17.07</td><td>20.38</td><td>37.07</td><td>45.14</td><td>17.43</td><td>42.54</td><td>52.01</td></tr><tr><td colspan="10">LF-Wikipedia-500K</td></tr><tr><td>Ensemble</td><td>82.55</td><td>61.96</td><td>46.65</td><td>39.82</td><td>51.29</td><td>55.76</td><td>25.72</td><td>58.46</td><td>72.17</td></tr><tr><td>Renée + LEVER</td><td>85.02</td><td>66.42</td><td>52.05</td><td>42.50</td><td>54.86</td><td>60.20</td><td>29.46</td><td>58.53</td><td>70.29</td></tr><tr><td colspan="10">LF-WikiHierarchy-1M</td></tr><tr><td>Ensemble</td><td>67.48</td><td>62.65</td><td>58.39</td><td>28.08</td><td>31.75</td><td>34.01</td><td>10.86</td><td>20.82</td><td>25.89</td></tr><tr><td>Renée + LEVER</td><td>95.19</td><td>93.90</td><td>92.07</td><td>24.79</td><td>32.74</td><td>38.29</td><td>9.08</td><td>16.12</td><td>20.02</td></tr></table>

Table 9: P@5 performance for different models across deciles. 

<table><tr><td>Model</td><td>Head Decile P@5</td><td>Torso Decile P@5</td><td>Tail Decile P@5</td><td>Overall P@5</td></tr><tr><td>Encoder</td><td>0/5</td><td>2/5</td><td>1/5</td><td>3/5</td></tr><tr><td>OvA</td><td>2/5</td><td>2/5</td><td>0/5</td><td>4/5</td></tr><tr><td>Ensemble</td><td>2/5</td><td>0/5</td><td>1/5</td><td>3/5</td></tr></table>

Table 10: Performance comparison of LEVER with other tail XC approaches LEVER outperforms other tail XC methods in tail metrics on 3 out of 4 datasets. while LEVER attains superior performance in tail metrics for three out of four datasets, there is a trade-off in precision compared to other methods in the LF-WikiHierarchy-1M dataset. 

<table><tr><td>Dataset</td><td>Model</td><td>P@1</td><td>P@3</td><td>P@5</td><td>Ps@1</td><td>Ps@3</td><td>Ps@5</td><td>C@1</td><td>C@3</td><td>C@5</td></tr><tr><td rowspan="6">LF-AmazonTitles-131K</td><td>Renée</td><td>46.05</td><td>30.81</td><td>22.04</td><td>38.47</td><td>44.87</td><td>50.33</td><td>31.31</td><td>53.50</td><td>61.03</td></tr><tr><td>+ Rerank</td><td>46.16</td><td>30.80</td><td>22.02</td><td>39.99</td><td>45.53</td><td>50.78</td><td>32.90</td><td>54.65</td><td>61.84</td></tr><tr><td>+ L2Reg + ReRank</td><td>44.89</td><td>29.71</td><td>21.14</td><td>39.99</td><td>44.58</td><td>49.29</td><td>33.18</td><td>54.48</td><td>61.19</td></tr><tr><td>+ GLaS + ReRank</td><td>45.06</td><td>29.82</td><td>21.17</td><td>40.18</td><td>44.83</td><td>49.49</td><td>33.36</td><td>54.78</td><td>61.48</td></tr><tr><td>+ Gandalf + ReRank</td><td>44.17</td><td>30.29</td><td>21.90</td><td>40.98</td><td>46.09</td><td>51.19</td><td>33.61</td><td>56.27</td><td>62.97</td></tr><tr><td>+ LEVER + ReRank</td><td>45.36</td><td>30.67</td><td>21.95</td><td>41.13</td><td>46.00</td><td>50.85</td><td>33.91</td><td>55.79</td><td>62.31</td></tr><tr><td rowspan="6">LF-AOL-270K</td><td>Renée</td><td>40.97</td><td>23.34</td><td>15.85</td><td>15.06</td><td>26.36</td><td>31.97</td><td>12.40</td><td>29.77</td><td>36.53</td></tr><tr><td>+ Rerank</td><td>41.53</td><td>24.11</td><td>16.44</td><td>20.21</td><td>31.11</td><td>37.24</td><td>20.27</td><td>36.13</td><td>43.01</td></tr><tr><td>+ L2Reg + ReRank</td><td>40.25</td><td>22.43</td><td>15.25</td><td>15.16</td><td>23.59</td><td>28.81</td><td>13.69</td><td>26.38</td><td>32.56</td></tr><tr><td>+ GLaS + ReRank</td><td>41.41</td><td>24.01</td><td>16.37</td><td>19.93</td><td>30.71</td><td>36.82</td><td>20.05</td><td>35.71</td><td>42.56</td></tr><tr><td>+ Gandalf + ReRank</td><td>40.87</td><td>23.49</td><td>15.94</td><td>20.70</td><td>30.68</td><td>36.22</td><td>20.61</td><td>35.47</td><td>41.65</td></tr><tr><td>+ LEVER + ReRank</td><td>39.60</td><td>24.23</td><td>16.92</td><td>28.20</td><td>41.58</td><td>49.33</td><td>27.40</td><td>48.95</td><td>57.62</td></tr><tr><td rowspan="6">LF-Wikipedia-500K</td><td>Renée</td><td>84.95</td><td>66.25</td><td>51.68</td><td>37.10</td><td>50.27</td><td>55.68</td><td>22.90</td><td>50.08</td><td>61.59</td></tr><tr><td>+ Rerank</td><td>79.28</td><td>63.56</td><td>50.80</td><td>53.44</td><td>56.16</td><td>59.06</td><td>41.58</td><td>59.52</td><td>67.17</td></tr><tr><td>+ L2Reg + ReRank</td><td>79.30</td><td>63.64</td><td>50.69</td><td>57.67</td><td>58.06</td><td>60.32</td><td>45.03</td><td>62.90</td><td>70.14</td></tr><tr><td>+ GLaS + ReRank</td><td>80.20</td><td>64.74</td><td>51.50</td><td>53.22</td><td>56.85</td><td>60.07</td><td>41.49</td><td>60.17</td><td>68.55</td></tr><tr><td>+ Gandalf + ReRank</td><td>80.58</td><td>64.55</td><td>51.29</td><td>51.03</td><td>55.09</td><td>58.36</td><td>39.07</td><td>58.00</td><td>66.23</td></tr><tr><td>+ LEVER + ReRank</td><td>75.34</td><td>62.07</td><td>50.26</td><td>59.15</td><td>60.29</td><td>62.95</td><td>47.24</td><td>68.40</td><td>75.94</td></tr><tr><td rowspan="6">LF-WikiHierarchy-1M</td><td>Renée</td><td>95.01</td><td>93.99</td><td>92.24</td><td>19.69</td><td>27.36</td><td>33.20</td><td>6.62</td><td>11.39</td><td>14.56</td></tr><tr><td>+ Rerank</td><td>89.95</td><td>89.94</td><td>88.86</td><td>44.15</td><td>52.89</td><td>58.47</td><td>18.15</td><td>30.53</td><td>35.16</td></tr><tr><td>+ L2Reg + ReRank</td><td>91.18</td><td>90.92</td><td>89.51</td><td>41.22</td><td>49.41</td><td>55.00</td><td>16.34</td><td>27.78</td><td>32.73</td></tr><tr><td>+ GLaS + ReRank</td><td>93.09</td><td>92.42</td><td>91.00</td><td>46.38</td><td>54.75</td><td>60.17</td><td>18.78</td><td>31.41</td><td>36.07</td></tr><tr><td>+ Gandalf + ReRank</td><td>86.03</td><td>83.04</td><td>80.88</td><td>51.77</td><td>57.79</td><td>61.37</td><td>20.21</td><td>34.82</td><td>39.73</td></tr><tr><td>+ LEVER + ReRank</td><td>86.27</td><td>84.51</td><td>83.69</td><td>52.17</td><td>58.92</td><td>63.59</td><td>19.60</td><td>34.84</td><td>40.58</td></tr></table>

encoder towards tail labels. Renée + LEVER $_{doc}$ denotes the model that uses NGAME encoder with mini-batches of documents to augment the training data. Table 12 shows the effect of sampling strategy by comparing Renée + LEVER $_{doc}$ and Renée + LEVER. Renée + LEVER outperforms Renée + LEVER $_{doc}$ by upto 2% in PSP while being comparable in precision.

![](images/7a24efbf1d79825d83fc7713e1cade083827c8448842e7af3427ab9947cac933.jpg)

<details>
<summary>line</summary>

| Avg Training Points per Label | Tail Expert Encoder | Renée | Ensemble (Encoder + Renée) | Renee + Lever |
| ----------------------------- | ------------------- | ----- | --------------------------- | ------------- |
| 2^11                          | 0.2                 | 8.3   | 8.1                         | 8.4           |
| 2^9                           | 0.8                 | 6.5   | 5.7                         | 6.2           |
| 2^7                           | 1.8                 | 4.8   | 3.9                         | 4.5           |
| 2^5                           | 3.5                 | 4.5   | 2.8                         | 4.3           |
| 2^3                           | 5.0                 | 4.2   | 4.0                         | 4.5           |
| 2^1                           | 8.5                 | 4.2   | 8.0                         | 7.2           |
</details>

Figure 3: Performance comparison of a Tail Expert Encoder (blue), an OvA Classifier (orange), an Ensemble of Expert Encoder and OvA Classifier (Green) and LEVER-based OvA Classifier (red) in the presence of label skew. Labels are partitioned into equi-volume bins based on their frequencies along the X-axis. OvA overfits to tail labels with few training points. Encoder leverages label metadata to improve on tail but underfits to head. Ensemble mode suffers on torso labels. LEVER combines the strengths of both OvA and Encoder to perform well on all labels. The macro prefix has been omitted for the sake of brevity.

Table 11: Comparison of different encoders: a 3-layer MiniLM and Astec, and their effects on LEVER performance. The Astec encoder learns a projection matrix that maps sparse Bag-of-Words features to a dense embedding space. A superior teacher leads to improved performance in both Precision and tail metrics, namely PSP and coverage. Note that for all we add the same number of neighbours for each label across all teachers. 

<table><tr><td rowspan="2"></td><td>P@1</td><td>P@3</td><td>P@5</td><td>PSP@1</td><td>PSP@3</td><td>PSP@5</td><td>C@1</td><td>C@3</td><td>C@5</td></tr><tr><td colspan="9">LF-AmazonTitles-131K</td></tr><tr><td>Astec Encoder</td><td>19.78</td><td>18.28</td><td>14.39</td><td>16.96</td><td>27.38</td><td>33.61</td><td>14.34</td><td>35.77</td><td>44.37</td></tr><tr><td>MinLM-L3 Encoder</td><td>23.86</td><td>21.65</td><td>16.82</td><td>20.22</td><td>32.44</td><td>39.26</td><td>17.20</td><td>41.80</td><td>50.82</td></tr><tr><td>DistilBERT-L6 Encoder</td><td>41.33</td><td>28.71</td><td>20.77</td><td>39.24</td><td>44.62</td><td>49.52</td><td>32.83</td><td>55.11</td><td>61.95</td></tr><tr><td>Renée</td><td>46.05</td><td>30.81</td><td>22.04</td><td>38.47</td><td>44.87</td><td>50.33</td><td>31.31</td><td>53.50</td><td>61.03</td></tr><tr><td>Renée + LEVER (Astec)</td><td>42.76</td><td>28.97</td><td>20.97</td><td>36.09</td><td>42.25</td><td>47.82</td><td>29.54</td><td>51.29</td><td>59.01</td></tr><tr><td>Renée + LEVER (MiniLM-L3)</td><td>45.26</td><td>30.40</td><td>21.82</td><td>38.28</td><td>44.67</td><td>50.09</td><td>31.22</td><td>53.73</td><td>61.11</td></tr><tr><td>Renée + LEVER (DistilBERT-L6)</td><td>46.44</td><td>30.83</td><td>21.92</td><td>39.70</td><td>45.44</td><td>50.31</td><td>32.82</td><td>55.11</td><td>61.94</td></tr><tr><td></td><td colspan="9">LF-AmazonTitles-1.3M</td></tr><tr><td>Astec Encoder</td><td>36.14</td><td>30.25</td><td>26.32</td><td>28.12</td><td>29.00</td><td>29.29</td><td>18.38</td><td>31.56</td><td>37.83</td></tr><tr><td>MiniLM-L3 Encoder</td><td>32.10</td><td>26.86</td><td>23.43</td><td>25.48</td><td>26.20</td><td>26.48</td><td>16.81</td><td>29.32</td><td>35.46</td></tr><tr><td>DistilBERT-L6 Encoder</td><td>42.27</td><td>36.16</td><td>31.63</td><td>35.62</td><td>38.11</td><td>38.87</td><td>22.37</td><td>38.93</td><td>46.98</td></tr><tr><td>Renée</td><td>56.10</td><td>49.91</td><td>45.32</td><td>28.56</td><td>33.38</td><td>36.14</td><td>17.61</td><td>30.60</td><td>37.59</td></tr><tr><td>Renée + LEVER (Astec)</td><td>49.30</td><td>43.12</td><td>39.26</td><td>30.46</td><td>33.83</td><td>35.86</td><td>18.39</td><td>33.10</td><td>40.71</td></tr><tr><td>Renée + LEVER (MiniLM-L3)</td><td>50.24</td><td>44.01</td><td>40.08</td><td>32.73</td><td>35.90</td><td>37.73</td><td>20.09</td><td>35.55</td><td>43.23</td></tr><tr><td>Renée + LEVER (DistilBERT-L6)</td><td>56.01</td><td>49.43</td><td>44.85</td><td>33.55</td><td>36.82</td><td>38.81</td><td>21.03</td><td>35.70</td><td>42.78</td></tr></table>

Effect of varying $\tau$ : The hyperparameter $\tau$ is tuned using a validation set that contains $5\%$ of the training data. The best value of $\tau$ obtained is then used to train LEVER on complete training data. Fig. 7 shows the effect of varying $\tau$ on LEVER's performance. It can be seen that increasing $\tau$ leads to better performance on tail labels, while it hurts the head or torso labels.

Effect of varying $\lambda$ : The hyperparameter $\lambda$ controls the importance between the two loss terms. Table 14 shows the effect of varying $\lambda$ , and Figures 4, 5 and 6 show their corresponding decile-wise plots.

# D MODEL DETAILS AND HYPERPARAMETERS

# D.1 TAIL EXPERT SIAMESE ENCODER

NGAME's (Dahiya et al., 2023a) hyperparameters include:

Table 12: Siamese teacher models, when trained with a document-wise sampling strategy (Siamese Encoder $_{doc}$ ) perform better in P@1 by 3% on average but are inferior in PSP@1 by 10% compared to label-wise trained teachers (Siamese Encoder $_{lbl}$ ). However, OvA classifiers, when distilled from Siamese Encoder $_{lbl}$ (Renée + LEVER $_{lbl}$ ), are comparable in Precision while more accurate in PSP@1 by 1.7% compared to OvA classifiers distilled from Siamese Encoder $_{doc}$ (Renée + LEVER $_{doc}$ ), thus highlighting the importance of a tail-specialized Siamese teacher model. 

<table><tr><td>Dataset</td><td>Model</td><td>P@1</td><td>P@3</td><td>P@5</td><td>PSP@1</td><td>PSP@3</td><td>PSP@5</td></tr><tr><td rowspan="4">LF-AmazonTitles-131K</td><td>Siamese  $Encoder_{doc}$ </td><td>43.13</td><td>28.99</td><td>20.73</td><td>38.72</td><td>43.93</td><td>48.81</td></tr><tr><td>Siamese  $Encoder_{lbl}$ </td><td>41.31</td><td>28.70</td><td>20.77</td><td>39.21</td><td>44.61</td><td>49.51</td></tr><tr><td>Renée +  $LEVER_{doc}$ </td><td>46.05</td><td>30.81</td><td>22.04</td><td>38.47</td><td>44.87</td><td>50.33</td></tr><tr><td>Renée +  $LEVER_{lbl}$ </td><td>46.44</td><td>30.83</td><td>21.92</td><td>39.70</td><td>45.44</td><td>50.31</td></tr><tr><td rowspan="4">LF-Wikipedia-500K</td><td>Siamese  $Encoder_{doc}$ </td><td>81.96</td><td>60.72</td><td>46.23</td><td>48.76</td><td>55.87</td><td>58.25</td></tr><tr><td>Siamese  $Encoder_{lbl}$ </td><td>67.82</td><td>45.66</td><td>34.31</td><td>60.77</td><td>57.26</td><td>57.20</td></tr><tr><td>Renée +  $LEVER_{doc}$ </td><td>85.09</td><td>65.94</td><td>51.69</td><td>40.17</td><td>52.65</td><td>58.18</td></tr><tr><td>Renée +  $LEVER_{lbl}$ </td><td>85.02</td><td>66.37</td><td>51.98</td><td>42.93</td><td>55.00</td><td>60.29</td></tr><tr><td rowspan="4">LF-WikiHierarchy-1M</td><td>Siamese  $Encoder_{doc}$ </td><td>59.08</td><td>53.96</td><td>49.58</td><td>50.82</td><td>50.20</td><td>50.16</td></tr><tr><td>Siamese  $Encoder_{lbl}$ </td><td>66.82</td><td>60.64</td><td>55.42</td><td>75.63</td><td>73.02</td><td>70.52</td></tr><tr><td>Renée +  $LEVER_{doc}$ </td><td>95.02</td><td>94.06</td><td>92.28</td><td>23.64</td><td>31.28</td><td>36.89</td></tr><tr><td>Renée +  $LEVER_{lbl}$ </td><td>95.19</td><td>93.90</td><td>92.07</td><td>24.79</td><td>32.74</td><td>38.29</td></tr><tr><td rowspan="4">LF-AmazonTitles-1.3M</td><td>Siamese  $Encoder_{doc}$ </td><td>45.83</td><td>39.94</td><td>35.48</td><td>33.04</td><td>35.64</td><td>36.80</td></tr><tr><td>Siamese  $Encoder_{lbl}$ </td><td>42.27</td><td>36.16</td><td>31.63</td><td>35.62</td><td>38.11</td><td>38.87</td></tr><tr><td>Renée +  $LEVER_{doc}$ </td><td>55.02</td><td>48.94</td><td>44.82</td><td>31.86</td><td>36.42</td><td>38.75</td></tr><tr><td>Renée +  $LEVER_{lbl}$ </td><td>56.01</td><td>49.43</td><td>44.85</td><td>33.55</td><td>36.82</td><td>38.81</td></tr></table>

Table 13: P and PSP Comparison of NGAME, Renée, and Renée + LEVER on QK-20M Dataset 

<table><tr><td></td><td>P@1</td><td>P@3</td><td>P@5</td><td>PSP@1</td><td>PSP@3</td><td>PSP@5</td></tr><tr><td>NGAME</td><td>69.94</td><td>52.72</td><td>44.81</td><td>48.24</td><td>55.63</td><td>58.71</td></tr><tr><td>Renée</td><td>72.14</td><td>54.87</td><td>47.02</td><td>50.75</td><td>58.90</td><td>62.56</td></tr><tr><td>Renée + LEVER</td><td>71.70</td><td>54.48</td><td>46.56</td><td>54.74</td><td>63.36</td><td>67.14</td></tr></table>

- cluster-sz: Mini-batches in NGAME are created from clusters of similar documents (or labels). To build a batch of $B$ documents (or labels) we pick $B / \text{cluster-sz}$ clusters.   
- cluster-freq: Denotes the frequency of refreshing the clusters using updated embeddings.   
- $\gamma$ : Denotes the margin enforced while training with contrastive loss.   
- lr: Learning rate for the encoder.   
- bsz: Denotes the size of mini-batches.   
- epochs: Denotes the number of epochs for which the NGAME module is trained.

To train the tail-expert NGAME module we closely follow the settings from (Dahiya et al., 2023a). NGAME utilizes a 6-layer DistilBERT architecture. Table 16 shows the hyperparameters used on benchmark as well as newly contributed datasets.

# D.2 ELIAS

ELIAS's (Gupta et al., 2022) hyperparameters include:

- $C$ : Denotes the number of clusters in the index graph.   
- $\alpha$ : Multiplicative hyperparameter that controls the effective number of clusters that can get activated for a given input get activated for a given input.   
- $\beta$ : Multiplicative hyperparameter that controls the effective number of labels that can get assigned to a particular cluster.   
- $\rho$ : Controls the row-wise sparsity of the adjacency matrix.   
- $\lambda_{elias}$ : Controls importance of classification loss $\mathcal{L}_c$ and shortlist loss $\mathcal{L}_s$ in the final loss.   
- $K$ : Denotes the shortlist size, label classifiers are only evaluated on top-K shortlisted labels.   
• b: Denotes the beam size.

Table 14: Effect of varying $\lambda$ when Renée is combined with LEVER. The equal weightage (0.5) gives the best performance. Increasing $\lambda$ weighs the hard labels more, resulting in performance that gets closer to the base classifier, i.e., PSP worsens, and Precision remains more or less unaffected. Decreasing $\lambda$ also helps only up to a certain point, i.e., $\lambda = 0.5$ ; we believe this is because our teacher is not perfect, and we strike a balance between hard and soft labels. Figures 4, 5 and 6 show the decile-wise plots corresponding to these values. Note that for LF-AmazonTitles-131K the effect of varying $\lambda$ is minimal. 

<table><tr><td colspan="10">LF-AmazonTitles-131K</td></tr><tr><td> $\lambda$ </td><td>P@1</td><td>P@3</td><td>P@5</td><td>PSP@1</td><td>PSP@3</td><td>PSP@5</td><td>C@1</td><td>C@3</td><td>C@5</td></tr><tr><td>0.33</td><td>46.15</td><td>30.76</td><td>21.87</td><td>39.7</td><td>45.33</td><td>50.13</td><td>32.56</td><td>54.48</td><td>61.24</td></tr><tr><td>0.50</td><td>46.44</td><td>30.83</td><td>21.92</td><td>39.70</td><td>45.44</td><td>50.31</td><td>32.82</td><td>55.11</td><td>61.94</td></tr><tr><td>0.66</td><td>46.57</td><td>30.87</td><td>21.93</td><td>39.59</td><td>45.46</td><td>50.36</td><td>32.31</td><td>54.49</td><td>61.48</td></tr><tr><td>0.80</td><td>46.58</td><td>30.88</td><td>21.92</td><td>39.37</td><td>45.39</td><td>50.33</td><td>32.06</td><td>54.38</td><td>61.42</td></tr><tr><td colspan="10">LF-Wikipedia-500K</td></tr><tr><td> $\lambda$ </td><td>P@1</td><td>P@3</td><td>P@5</td><td>PSP@1</td><td>PSP@3</td><td>PSP@5</td><td>C@1</td><td>C@3</td><td>C@5</td></tr><tr><td>0.33</td><td>84.66</td><td>65.94</td><td>51.54</td><td>40.51</td><td>53.4</td><td>58.75</td><td>26.88</td><td>56.01</td><td>67.94</td></tr><tr><td>0.50</td><td>85.02</td><td>66.42</td><td>52.05</td><td>42.50</td><td>54.86</td><td>60.20</td><td>29.46</td><td>58.53</td><td>70.29</td></tr><tr><td>0.66</td><td>84.96</td><td>66.51</td><td>52.18</td><td>39.34</td><td>53.53</td><td>59.46</td><td>25.66</td><td>55.78</td><td>68.33</td></tr><tr><td>0.80</td><td>84.84</td><td>66.42</td><td>52.14</td><td>38.66</td><td>53.09</td><td>59.16</td><td>24.96</td><td>55.08</td><td>67.79</td></tr><tr><td colspan="10">LF-AOL-270K</td></tr><tr><td> $\lambda$ </td><td>P@1</td><td>P@3</td><td>P@5</td><td>PSP@1</td><td>PSP@3</td><td>PSP@5</td><td>C@1</td><td>C@3</td><td>C@5</td></tr><tr><td>0.33</td><td>41.14</td><td>24</td><td>16.5</td><td>17.24</td><td>32.31</td><td>40.02</td><td>14.14</td><td>36.67</td><td>45.84</td></tr><tr><td>0.50</td><td>41.70</td><td>24.78</td><td>17.07</td><td>20.38</td><td>37.07</td><td>45.13</td><td>17.43</td><td>42.54</td><td>52.01</td></tr><tr><td>0.66</td><td>41.44</td><td>24.41</td><td>16.76</td><td>16.76</td><td>32.69</td><td>40.47</td><td>14.16</td><td>37.38</td><td>46.54</td></tr><tr><td>0.80</td><td>41.17</td><td>24.04</td><td>16.49</td><td>15.59</td><td>30.3</td><td>37.64</td><td>13.16</td><td>34.62</td><td>43.25</td></tr></table>

Table 15: Renée (OvA) and Siamese trained encoder exhibit different trade-offs on in precision and tail-metrics (PSP, Coverage). LEVER improves the tail performance of Renée (+5% on average in PSP and +3% on average in coverage) while retaining comparable precision. 

<table><tr><td></td><td>P@1</td><td>P@3</td><td>P@5</td><td>PSP@1</td><td>PSP@3</td><td>PSP@5</td><td>C@1</td><td>C@3</td><td>C@5</td></tr><tr><td colspan="10">LF-AmazonTitles-131K</td></tr><tr><td>Siamese Encoder</td><td>41.33</td><td>28.71</td><td>20.77</td><td>39.24</td><td>44.62</td><td>49.52</td><td>32.83</td><td>55.11</td><td>61.95</td></tr><tr><td>Renée</td><td>46.05</td><td>30.81</td><td>22.04</td><td>38.47</td><td>44.87</td><td>50.33</td><td>31.31</td><td>53.50</td><td>61.03</td></tr><tr><td>Renée + LEVER</td><td>46.44</td><td>30.83</td><td>21.92</td><td>39.70</td><td>45.44</td><td>50.31</td><td>32.50</td><td>54.59</td><td>61.42</td></tr><tr><td colspan="10">LF-AOL-270K</td></tr><tr><td>Siamese Encoder</td><td>23.24</td><td>15.67</td><td>11.68</td><td>25.41</td><td>36.24</td><td>43.43</td><td>27.49</td><td>47.45</td><td>55.58</td></tr><tr><td>Renée</td><td>40.97</td><td>23.34</td><td>15.85</td><td>14.76</td><td>26.45</td><td>32.19</td><td>12.40</td><td>29.77</td><td>36.53</td></tr><tr><td>Renée + LEVER</td><td>41.71</td><td>24.77</td><td>17.07</td><td>20.38</td><td>37.07</td><td>45.14</td><td>17.43</td><td>42.54</td><td>52.01</td></tr><tr><td colspan="10">LF-Wikipedia-500K</td></tr><tr><td>Siamese Encoder</td><td>67.81</td><td>45.65</td><td>34.31</td><td>60.76</td><td>57.25</td><td>57.20</td><td>48.76</td><td>71.41</td><td>78.59</td></tr><tr><td>Renée</td><td>84.95</td><td>66.25</td><td>51.68</td><td>37.10</td><td>50.27</td><td>55.68</td><td>22.90</td><td>50.08</td><td>61.59</td></tr><tr><td>Renée + LEVER</td><td>85.02</td><td>66.37</td><td>51.98</td><td>42.93</td><td>55.00</td><td>60.29</td><td>29.46</td><td>58.53</td><td>70.29</td></tr><tr><td colspan="10">LF-WikiHierarchy-1M</td></tr><tr><td>Siamese Encoder</td><td>66.82</td><td>60.64</td><td>55.42</td><td>75.63</td><td>73.02</td><td>70.54</td><td>21.82</td><td>42.93</td><td>50.62</td></tr><tr><td>Renée</td><td>95.01</td><td>93.99</td><td>92.24</td><td>19.69</td><td>27.36</td><td>33.20</td><td>6.72</td><td>11.49</td><td>14.65</td></tr><tr><td>Renée + LEVER</td><td>95.19</td><td>93.90</td><td>92.07</td><td>24.76</td><td>32.63</td><td>38.15</td><td>9.32</td><td>16.14</td><td>20.29</td></tr></table>

LF-Wikipedia-500K   
![](images/262cc927b888aa00875bdd0d03923044803a7e7e07d68c40cca14f2acea3616d.jpg)

<details>
<summary>line</summary>

| Avg Training Points per Label | λ = 0.33 | λ = 0.50 | λ = 0.66 | λ = 0.80 |
| ----------------------------- | -------- | -------- | -------- | -------- |
| 2^11                          | 8.3      | 8.4      | 8.3      | 8.3      |
| 2^9                           | 6.2      | 6.3      | 6.2      | 6.2      |
| 2^7                           | 4.7      | 4.6      | 4.7      | 4.7      |
| 2^5                           | 4.2      | 4.3      | 4.3      | 4.3      |
| 2^3                           | 4.1      | 4.2      | 4.1      | 4.1      |
| 2^1                           | 4.0      | 4.1      | 4.0      | 4.0      |
</details>

Figure 4: Effect of varying hyperparameter $\lambda$ on LEVER's head and tail performance on LF-Wikipedia-500K

LF-AmazonTitles-131K   
![](images/c7d1fef3a2c5fb0c70f9ad82364363fd6bc034c35bcf33cc082e65dafac472eb.jpg)

<details>
<summary>line</summary>

| Avg Training Points per Label | λ = 0.33 | λ = 0.50 | λ = 0.66 | λ = 0.80 |
| ----------------------------- | -------- | -------- | -------- | -------- |
| 2^6                           | 2.7      | 2.7      | 2.7      | 2.7      |
| 2^5                           | 1.8      | 1.8      | 1.8      | 1.8      |
| 2^4                           | 1.6      | 1.6      | 1.6      | 1.6      |
| 2^3                           | 1.6      | 1.6      | 1.6      | 1.6      |
| 2^2                           | 1.8      | 1.8      | 1.8      | 1.8      |
| 2^1                           | 4.2      | 4.2      | 4.2      | 4.2      |
</details>

Figure 5: Effect of varying hyperparameter $\lambda$ on LEVER's head and tail performance on LF-AmazonTitles-127K. Here the choice of $\lambda$ has minimal affect on the final performance of LEVER therefore all four plots are closely superimposed.

- epochs: Denotes the total number of epochs (i.e. including stage 1 and stage 2 training).   
- $LR_{\phi}, LR_W$ : Denotes the learning rate used for the transformer encoder and the rest of the model.   
- bsz: denotes the batch-size of the mini-batches used during training

We closely follow the setting used in (Gupta et al., 2022). ELIAS uses a 6-layer Distil-BERT encoder. Note that the NGAME encoder is only used to augment the ground truth with labels similar to a particular label, it is not used in any other way while training ELIAS. Table 17 shows the hyperparameters used on the benchmark as well as newly contributed datasets.

# D.3 CASCADEXML

CascadeXML's (Kharbanda et al., 2022) hyperparameters include:

![](images/de962ce5bf4518a0c869e183eb0b0c60cbd6fecd55c606d1687329cd908b29e9.jpg)

<details>
<summary>line</summary>

| Avg Training Points per Label | λ = 0.33 | λ = 0.50 | λ = 0.66 | λ = 0.80 |
| ----------------------------- | -------- | -------- | -------- | -------- |
| 2^19                          | 2.6      | 2.6      | 2.6      | 2.6      |
| 2^16                          | 2.4      | 2.4      | 2.4      | 2.4      |
| 2^13                          | 1.8      | 1.8      | 1.8      | 1.8      |
| 2^10                          | 1.5      | 1.5      | 1.5      | 1.5      |
| 2^7                           | 1.2      | 1.2      | 1.2      | 1.2      |
| 2^4                           | 1.4      | 1.4      | 1.4      | 1.4      |
| 2^8                           | 3.0      | 3.7      | 3.1      | 2.7      |
</details>

Figure 6: Effect of varying hyperparameter $\lambda$ on LEVER's head and tail performance on LF-AOL-270K

Table 16: Hyperparameters of tail-expert NGAME module. × indicates use of random mini-batches. 

<table><tr><td>Dataset</td><td>cluster-sz</td><td>cluster-freq</td><td> $\gamma$ </td><td>LR</td><td>bsz</td><td>Epochs</td></tr><tr><td>LF-AmazonTitles-131K</td><td>8</td><td>5</td><td>0.3</td><td> $2 \times 10^{-4}$ </td><td>1600</td><td>300</td></tr><tr><td>LF-Amazon-131K</td><td>512</td><td>5</td><td>0.3</td><td> $2 \times 10^{-4}$ </td><td>700</td><td>400</td></tr><tr><td>LF-AOL-270K</td><td> $\times$ </td><td> $\times$ </td><td>0.05</td><td> $2 \times 10^{-4}$ </td><td>3200</td><td>300</td></tr><tr><td>LF-WikiSeeAlso-320K</td><td>512</td><td>5</td><td>0.3</td><td> $2 \times 10^{-4}$ </td><td>1024</td><td>300</td></tr><tr><td>LF-Wikipedia-500K</td><td>16</td><td>5</td><td>0.3</td><td> $2 \times 10^{-4}$ </td><td>512</td><td>40</td></tr><tr><td>LF-WikiHierarchy-1M</td><td>1024</td><td>5</td><td>0.3</td><td> $2 \times 10^{-4}$ </td><td>6400</td><td>300</td></tr><tr><td>LF-AmazonTitles-1.3M</td><td>8</td><td>5</td><td>0.3</td><td> $2 \times 10^{-4}$ </td><td>1600</td><td>400</td></tr></table>

- Ep: Number of epochs CascadeXML is trained for.   
- bsz: Denotes the batch size used for training.   
- label resolution: Denotes the BERT layers and clustering size used at each resolution.   
- dropout: Dropout used at each resolution.   
- shortlist size: Cluster size used at each resolution.   
- $LR_{\phi}, LR_W$ : Denotes the learning rate used for the transformer encoder and weight vectors.

We closely follow the setting used in (Kharbanda et al., 2022). CascadeXML uses a 12-layer BERT encoder. Note that the NGAME encoder is only used to augment the ground truth with labels similar to a particular label, it is not used in any other way while training CascadeXML. Table 18 shows the hyperparameters used on benchmark as well as newly contributed datasets.

# D.4 RENÉE

Renée's (Jain et al., 2023) hyperparameters include:

- epochs: Denotes the total number of epochs for which Renée is trained.   
- dropout: Denotes the probability of randomly dropping the encoder outputs in order to regularise the network.   
- warmup: Warmup steps is the number of training iterations over which both the encoder and the classifier learning rates are linearly increased from 0 to the maximum value.

Table 17: Hyperparameters of ELIAS 

<table><tr><td>Dataset</td><td>C</td><td> $\alpha$ </td><td> $\beta$ </td><td> $\rho$ </td><td> $\lambda_{elias}$ </td><td>K</td><td>b</td><td>Epochs</td><td> $LR_{\phi}$ </td><td> $LR_{W}$ </td><td>bsz</td></tr><tr><td>LF-AmazonTitles-131K</td><td>2048</td><td>10</td><td>150</td><td>1000</td><td>0.05</td><td>2000</td><td>20</td><td>60</td><td> $1 \times 10^{-4}$ </td><td> $2 \times 10^{-2}$ </td><td>512</td></tr><tr><td>LF-Amazon-131K</td><td>2048</td><td>10</td><td>150</td><td>1000</td><td>0.05</td><td>2000</td><td>20</td><td>70</td><td> $7 \times 10^{-5}$ </td><td> $5 \times 10^{-3}$ </td><td>1024</td></tr><tr><td>LF-AOL-270K</td><td>4096</td><td>10</td><td>150</td><td>1000</td><td>0.05</td><td>2000</td><td>20</td><td>70</td><td> $3 \times 10^{-5}$ </td><td> $1 \times 10^{-3}$ </td><td>8192</td></tr><tr><td>LF-WikiSeeAlso-320K</td><td>4096</td><td>10</td><td>150</td><td>1000</td><td>0.05</td><td>2000</td><td>20</td><td>40</td><td> $5 \times 10^{-5}$ </td><td> $5 \times 10^{-3}$ </td><td>1024</td></tr><tr><td>LF-Wikipedia-500K</td><td>8192</td><td>10</td><td>150</td><td>1000</td><td>0.05</td><td>2000</td><td>20</td><td>40</td><td> $5 \times 10^{-5}$ </td><td> $5 \times 10^{-3}$ </td><td>256</td></tr><tr><td>LF-WikiHierarchy-1M</td><td>16384</td><td>10</td><td>150</td><td>1000</td><td>0.05</td><td>2000</td><td>20</td><td>30</td><td> $5 \times 10^{-5}$ </td><td> $5 \times 10^{-3}$ </td><td>1024</td></tr><tr><td>LF-AmazonTitles-1.3M</td><td>16384</td><td>10</td><td>150</td><td>1000</td><td>0.05</td><td>2000</td><td>20</td><td>40</td><td> $2 \times 10^{-5}$ </td><td> $1 \times 10^{-3}$ </td><td>1024</td></tr></table>

Table 18: Hyperparameters of CascadeXML 

<table><tr><td>Dataset</td><td>Ep</td><td>bsz</td><td>Label Resolution</td><td>Dropout</td><td>Shortlist-sz</td><td> $LR_{\phi}$ </td><td> $LR_W$ </td></tr><tr><td>LF-AmazonTitles-131K</td><td>15</td><td>64</td><td> $\{5,6\}:2^{10}-\{8\}:2^{13}-\{10\}:2^{16}-12:131073$ </td><td>0.2, 0.25, 0.35, 0.5</td><td> $2^{10}, 2^{10}, 2^{10}$ </td><td> $1e^{-4}$ </td><td> $1e^{-3}$ </td></tr><tr><td>LF-Amazon-131K</td><td>15</td><td>64</td><td> $\{5,6\}:2^{9}-\{8\}:2^{12}-\{10\}:2^{15}-12:131073$ </td><td>0.2, 0.25, 0.4, 0.5</td><td> $2^{6}, 2^{7}, 2^{8}$ </td><td> $1e^{-4}$ </td><td> $1e^{-3}$ </td></tr><tr><td>LF-AOL-270K</td><td>12</td><td>96</td><td> $\{5,6\}:2^{10}-\{8\}:2^{13}-\{10\}:2^{16}-12:272825$ </td><td>0.2, 0.25, 0.35, 0.5</td><td> $2^{10}, 2^{10}, 2^{10}$ </td><td> $1e^{-4}$ </td><td> $1e^{-3}$ </td></tr><tr><td>LF-WikiSeeAlso-320K</td><td>12</td><td>64</td><td> $\{5,6\}:2^{10}-\{8\}:2^{13}-\{10\}:2^{16}-12:312330$ </td><td>0.2, 0.25, 0.35, 0.5</td><td> $2^{10}, 2^{11}, 2^{12}$ </td><td> $1e^{-4}$ </td><td> $1e^{-3}$ </td></tr><tr><td>LF-Wikipedia-500K</td><td>12</td><td>256</td><td> $\{5,6\}:2^{10}-\{8\}:2^{13}-\{10\}:2^{16}-12:501070$ </td><td>0.2, 0.25, 0.35, 0.5</td><td> $2^{10}, 2^{10}, 2^{11}$ </td><td> $1e^{-4}$ </td><td> $1e^{-3}$ </td></tr><tr><td>LF-WikiHierarchy-1M</td><td>12</td><td>96</td><td> $\{5,6\}:2^{10}-\{8\}:2^{13}-\{10\}:2^{16}-12:976214$ </td><td>0.2, 0.25, 0.35, 0.5</td><td> $2^{10}, 2^{10}, 2^{10}$ </td><td> $1e^{-4}$ </td><td> $1e^{-3}$ </td></tr><tr><td>LF-AmazonTitles-1.3M</td><td>10</td><td>48</td><td> $\{7,8\}:2^{13}-\{10\}:2^{16}-12:1305265$ </td><td>0.2, 0.3, 0.4</td><td> $2^{10}, 2^{11}$ </td><td> $1e^{-4}$ </td><td> $1e^{-3}$ </td></tr></table>

- $LR_{\phi}, LR_W$ : Denotes the learning rate used for the transformer encoder and the classifier layer.   
- bsz: Denotes the batch size of the mini-batches used during training.   
- clf-wd: Weight decay for fully connected layer parameters.

Table 19: Hyperparameters of Renée 

<table><tr><td>Dataset</td><td>Epochs</td><td>Dropout</td><td>Warmup</td><td> $LR_{\phi}$ </td><td> $LR_W$ </td><td>bsz</td><td>clf-wd</td></tr><tr><td>LF-AmazonTitles-131K</td><td>100</td><td>0.85</td><td>5000</td><td> $1 \times 10^{-5}$ </td><td> $5 \times 10^{-2}$ </td><td>512</td><td> $1 \times 10^{-4}$ </td></tr><tr><td>LF-Amazon-131K</td><td>100</td><td>0.85</td><td>5000</td><td> $1 \times 10^{-5}$ </td><td> $5 \times 10^{-2}$ </td><td>512</td><td> $1 \times 10^{-4}$ </td></tr><tr><td>LF-AOL-270K</td><td>100</td><td>0.60</td><td>20000</td><td> $1 \times 10^{-6}$ </td><td> $1 \times 10^{-3}$ </td><td>1024</td><td> $1 \times 10^{-4}$ </td></tr><tr><td>LF-WikiSeeAlso-320K</td><td>100</td><td>0.75</td><td>5000</td><td> $2 \times 10^{-4}$ </td><td> $2 \times 10^{-1}$ </td><td>2048</td><td> $1 \times 10^{-4}$ </td></tr><tr><td>LF-Wikipedia-500K</td><td>100</td><td>0.70</td><td>5000</td><td> $5 \times 10^{-5}$ </td><td> $4 \times 10^{-3}$ </td><td>2048</td><td> $1 \times 10^{-4}$ </td></tr><tr><td>LF-WikiHierarchy-1M</td><td>100</td><td>0.70</td><td>20000</td><td> $1 \times 10^{-4}$ </td><td> $2 \times 10^{-3}$ </td><td>1024</td><td> $1 \times 10^{-2}$ </td></tr><tr><td>LF-AmazonTitles-1.3M</td><td>100</td><td>0.70</td><td>15000</td><td> $1 \times 10^{-6}$ </td><td> $1 \times 10^{-2}$ </td><td>1024</td><td> $1 \times 10^{-4}$ </td></tr></table>

We closely follow the setting used in (Jain et al., 2023). Renée uses a 6-layer Distil-BERT encoder. Table 19 shows the hyperparameters used on benchmark as well as newly contributed datasets.

# D.5 RERANK + TAUG

ReRank + TAUG (Wei et al., 2021) hyperparameters include:

- $\epsilon_{split}$ : Denotes the proportion of labels that will be considered as head labels. The original dataset $D$ containing $L$ labels is split into 2 datasets $D_h$ and $D_t$ . $D_h$ contains headmost $\epsilon_{split}L$ labels and their associated training points, while $D_t$ contains the remaining $L - \epsilon_{split}L$ labels along with their associated training points.   
- n-aug: Denotes the number of additional data points that will be generated for each data point in $D_{t}$ .   
- $p_{drop}$ : Denotes the probability of dropping a token from the data point.   
- $p_{swap}$ : Denotes the probability of swapping two randomly chosen tokens.   
- rerank-strategy: Denotes the multiplicative factor used to re-rank scores. We use the label inverse propensity factor to perform re-ranking.

# D.6 GANDALF

Gandalf's (Kharbanda et al., 2024) hyperparameters include:

Table 20: Hyperparameters of Re-rank + TAUG 

<table><tr><td>Dataset</td><td> $\epsilon_{\text{split}}$ </td><td>n-aug</td><td> $p_{\text{drop}}$ </td><td> $p_{\text{swap}}$ </td></tr><tr><td>LF-AmazonTitles-131K</td><td>0.90</td><td>8</td><td>0.30</td><td>0.30</td></tr><tr><td>LF-AOL-270K</td><td>0.65</td><td>6</td><td>0.20</td><td>0.20</td></tr><tr><td>LF-Wikipedia-500K</td><td>0.90</td><td>4</td><td>0.10</td><td>0.10</td></tr><tr><td>LF-WikiHierarchy-1M</td><td>0.90</td><td>4</td><td>0.20</td><td>0.20</td></tr></table>

\- threshold: Denotes the threshold used to filter out labels obtained from the normalized label correlation graph during augmentation.

We closely follow the settings used in (Kharbanda et al., 2024) and use threshold of 0.1 for all datasets.

# D.7 LEVER

# D.7.1 HYPERPARAMETERS

LEVER uses the parameter $\tau$ to control the number of entities (data points or labels) that are added for each label. The hyper-parameter $c$ from Theorem 3 was set to 0, as this value worked consistently well across datasets. Table 22 displays the values of $\tau$ , $\tau_{d}$ , and $\tau_{l}$ for both benchmark and newly contributed datasets. We observed that setting $\tau_{d} = 0$ yields satisfactory results for most cases, with the exception of LF-Wikipedia-500K, for which we set $\tau_{d} = 12$ . A value of $\lambda = 0.5$ was employed for all experiments. Table 21 illustrates the effect of varying $\tau$ , and Table 14 demonstrates the effect of varying $\lambda$ .

Table 21: Performance variation in P, PSP and Coverage when as $\tau$ is varied in LEVER for LF-Wikipedia-500K and LF-WikiHierarchy-1M. Figure 7 shows the decile-wise plots corresponding to these values. As more neighbours are added, the tail metrics (PSP and Coverage) improve while the Precision remains constant or slightly drops. 

<table><tr><td colspan="11">LF-Wikipedia-500K</td></tr><tr><td>τ</td><td>PPL</td><td>P@1</td><td>P@3</td><td>P@5</td><td>PSP@1</td><td>PSP@3</td><td>PSP@5</td><td>C@1</td><td>C@3</td><td>C@5</td></tr><tr><td>1</td><td>18.1</td><td>84.96</td><td>66.26</td><td>51.68</td><td>37.10</td><td>50.27</td><td>55.68</td><td>22.90</td><td>50.08</td><td>61.59</td></tr><tr><td>20</td><td>39.5</td><td>85.14</td><td>66.90</td><td>52.35</td><td>38.39</td><td>52.33</td><td>58.04</td><td>24.72</td><td>53.45</td><td>65.43</td></tr><tr><td>45</td><td>64.5</td><td>85.02</td><td>66.43</td><td>52.05</td><td>42.51</td><td>54.86</td><td>60.20</td><td>29.08</td><td>58.28</td><td>70.09</td></tr><tr><td colspan="11">LF-WikiHierarchy-1M</td></tr><tr><td>τ</td><td>PPL</td><td>P@1</td><td>P@3</td><td>P@5</td><td>PSP@1</td><td>PSP@3</td><td>PSP@5</td><td>C@1</td><td>C@3</td><td>C@5</td></tr><tr><td>1</td><td>43.3</td><td>95.01</td><td>93.99</td><td>92.24</td><td>19.69</td><td>27.36</td><td>33.20</td><td>6.62</td><td>11.39</td><td>14.56</td></tr><tr><td>4</td><td>44.8</td><td>95.19</td><td>93.90</td><td>92.07</td><td>24.79</td><td>32.74</td><td>38.29</td><td>9.08</td><td>16.12</td><td>20.02</td></tr><tr><td>8</td><td>47.7</td><td>94.97</td><td>93.18</td><td>90.98</td><td>26.37</td><td>34.99</td><td>40.51</td><td>10.08</td><td>18.34</td><td>22.74</td></tr></table>

![](images/870c55e24e607b2de4bfca2349f332668090674b035dbed949a533c0c6bdff53.jpg)

<details>
<summary>line</summary>

| Avg Training Points per Label | τ=45 | τ=20 | τ=1 |
| ----------------------------- | ---- | ---- | --- |
| 2^11                          | 8.5  | 8.4  | 8.3 |
| 2^9                           | 6.2  | 6.0  | 5.8 |
| 2^7                           | 4.5  | 4.3  | 4.6 |
| 2^5                           | 4.2  | 4.4  | 4.3 |
| 2^3                           | 4.1  | 4.5  | 4.2 |
| 2^1                           | 7.3  | 6.2  | 5.6 |
</details>

![](images/1819844e07f686956c65429dcf803f3d0abcae3fccfcd5116e181b618ebdb40d.jpg)

<details>
<summary>line</summary>

| Avg Training Points per Label | τ=8   | τ=4   | τ=1   |
| ----------------------------- | ----- | ----- | ----- |
| 2^13                          | 9.0   | 10.5  | 11.0  |
| 2^11                          | 8.0   | 9.5   | 10.5  |
| 2^9                           | 8.5   | 10.0  | 10.5  |
| 2^7                           | 10.5  | 9.5   | 10.0  |
| 2^5                           | 10.5  | 8.0   | 6.5   |
| 2^3                           | 8.0   | 5.5   | 3.0   |
</details>

Figure 7: Effect of varying hyperparameter $\tau$ on LEVER's head and tail performance on LF-Wikipedia-500K and LF-WikiHierarchy-1M.

![](images/aa4a656e7d2a2c808c0ba7edaac718f5a3756cb908225fbf4e9438cc944f558b.jpg)

<details>
<summary>line</summary>

| Avg Training Points per Label | ELIAS | ELIAS + LEVER | Renée | Renée + LEVER |
| ----------------------------- | ----- | ------------- | ----- | ------------- |
| 2^13                          | 18.0  | 6.5           | 11.0  | 10.0          |
| 2^10                          | 10.5  | 7.0           | 10.5  | 9.5           |
| 2^7                           | 8.0   | 12.0          | 9.0   | 10.0          |
| 2^4                           | 4.5   | 13.5          | 6.0   | 8.5           |
| 2^3                           | 2.0   | 8.5           | 3.0   | 6.0           |
</details>

Figure 8: In ELIAS there is a bigger drop in head performance as compared to Renée when LEVER is applied. This therefore translates to a much larger increase in ELIAS+LEVER's tail performance for this dataset, note that for ELIAS the performance improvement starts from the torso labels itself.

Table 22: LEVER's Hyperparameter $\tau$ on benchmark datasets 

<table><tr><td>Dataset</td><td> $\tau$ </td><td> $\tau_{l}$ </td><td> $\tau_{d}$ </td></tr><tr><td>LF-AmazonTitles-131K</td><td>15</td><td>15</td><td>0</td></tr><tr><td>LF-Amazon-131K</td><td>20</td><td>20</td><td>0</td></tr><tr><td>LF-AOL-270K</td><td>100</td><td>100</td><td>0</td></tr><tr><td>LF-WikiSeeAlso-320K</td><td>4</td><td>4</td><td>0</td></tr><tr><td>LF-Wikipedia-500K</td><td>75</td><td>60</td><td>15</td></tr><tr><td>LF-WikiHierarchy-1M</td><td>4</td><td>4</td><td>0</td></tr><tr><td>LF-AmazonTitles-1.3M</td><td>15</td><td>15</td><td>0</td></tr></table>

# D.7.2 TRAINING TIME

Table 23 shows the training time for different models and datasets when combined with LEVER. Note that in ELIAS and CascadeXML, where the train times increase by a greater margin, the gains provided by LEVER are also higher (avg. +6.1% increase in PSP and +2

Table 23: Training time (in hours) for different models on a single NVIDIA V100 GPU. The average training time increases by 3.1x, and in the worst case, by 8.9x. For LEVER counterparts, this includes the time for training the teacher model, generating soft labels, and using/training the OvA classifier.

<table><tr><td>Dataset</td><td>Renée</td><td>Renée+ LEVER</td><td>ELIAS</td><td>ELIAS+ LEVER</td><td>CascadeXML</td><td>CascadeXML+ LEVER</td></tr><tr><td>LF-AmazonTitles-131K</td><td>17.59</td><td>20.11</td><td>4.33</td><td>18.82</td><td>3.63</td><td>17.14</td></tr><tr><td>LF-Amazon-131K</td><td>42.77</td><td>46.51</td><td>19.44</td><td>65.62</td><td>4.60</td><td>41.31</td></tr><tr><td>LF-AOL-270K</td><td>136.22</td><td>138.20</td><td>60.67</td><td>171.20</td><td>42.12</td><td>152.00</td></tr><tr><td>LF-WikiSeeAlso-320K</td><td>86.42</td><td>95.19</td><td>25.33</td><td>111.46</td><td>12.40</td><td>88.52</td></tr><tr><td>LF-Wikipedia-500K</td><td>154.93</td><td>184.05</td><td>138.67</td><td>226.72</td><td>29.58</td><td>89.72</td></tr><tr><td>LF-WikiHierarchy-1M</td><td>31.44</td><td>40.15</td><td>24.00</td><td>61.17</td><td>9.85</td><td>35.91</td></tr><tr><td>LF-AmazonTitles-1.3M</td><td>154.39</td><td>186.45</td><td>40.00</td><td>158.23</td><td>70.00</td><td>202.93</td></tr><tr><td>Average Time Inc.</td><td></td><td>1.14x</td><td></td><td>3.29x</td><td></td><td>4.86x</td></tr></table>

Table 24, 25, 26 show the runtime break down when LEVER. Note that the training time of the Siamese Teacher is less than what is reported in (Dahiya et al., 2023a) as that includes the time taken to train both the NGAME Encoder and NGAME Classifier.

Table 24: Renée + LEVER training time (in hrs) on a single NVIDIA V100 GPU. Training LEVER involves three steps (a): Time to train the Siamese teacher, (b): Time to construct soft labels and (c): Time to train the Renée. 

<table><tr><td>Dataset</td><td>Siamese Teacher</td><td>Soft Labels</td><td>Renée</td><td>Total</td></tr><tr><td>LF-AmazonTitles-131K</td><td>11.82</td><td>0.07</td><td>8.22</td><td>20.11</td></tr><tr><td>LF-Amazon-131K</td><td>34.44</td><td>0.07</td><td>12.00</td><td>46.51</td></tr><tr><td>LF-AOL-270K</td><td>108.00</td><td>0.20</td><td>30.00</td><td>138.20</td></tr><tr><td>LF-WikiSeeAlso-320K</td><td>69.86</td><td>0.25</td><td>25.07</td><td>95.19</td></tr><tr><td>LF-Wikipedia-500K</td><td>50.26</td><td>0.45</td><td>133.33</td><td>184.05</td></tr><tr><td>LF-WikiHierarchy-1M</td><td>19.33</td><td>1.04</td><td>19.78</td><td>40.15</td></tr><tr><td>LF-AmazonTitles-1.3M</td><td>93.50</td><td>0.73</td><td>92.22</td><td>186.45</td></tr></table>

Table 25: ELIAS + LEVER training time (in hrs) on a single NVIDIA V100 GPU. Training LEVER involves three steps (a): Time to train the Siamese teacher, (b): Time to construct soft labels and (c): Time to train the ELIAS. 

<table><tr><td>Dataset</td><td>Siamese Teacher</td><td>Soft Labels</td><td>ELIAS</td><td>Total</td></tr><tr><td>LF-AmazonTitles-131K</td><td>11.82</td><td>0.07</td><td>6.93</td><td>18.82</td></tr><tr><td>LF-Amazon-131K</td><td>34.44</td><td>0.07</td><td>31.11</td><td>65.62</td></tr><tr><td>LF-AOL-270K</td><td>108.00</td><td>0.20</td><td>63.00</td><td>171.20</td></tr><tr><td>LF-WikiSeeAlso-320K</td><td>69.86</td><td>0.25</td><td>41.33</td><td>111.46</td></tr><tr><td>LF-Wikipedia-500K</td><td>50.26</td><td>0.45</td><td>176.00</td><td>226.72</td></tr><tr><td>LF-WikiHierarchy-1M</td><td>19.33</td><td>1.04</td><td>40.80</td><td>61.17</td></tr><tr><td>LF-AmazonTitles-1.3M</td><td>93.50</td><td>0.73</td><td>64.00</td><td>158.23</td></tr></table>

Table 26: CascadeXML + LEVER training time (in hrs) on a single NVIDIA V100 GPU. Training LEVER involves three steps (a): Time to train the Siamese teacher, (b): Time to construct soft labels and (c): Time to train the CascadeXML. 

<table><tr><td>Dataset</td><td>Siamese Teacher</td><td>Soft Labels</td><td>CascadeXML</td><td>Total</td></tr><tr><td>LF-AmazonTitles-131K</td><td>11.82</td><td>0.07</td><td>5.25</td><td>17.14</td></tr><tr><td>LF-Amazon-131K</td><td>34.44</td><td>0.07</td><td>6.80</td><td>41.31</td></tr><tr><td>LF-AOL-270K</td><td>108.00</td><td>0.20</td><td>43.80</td><td>152.00</td></tr><tr><td>LF-WikiSeeAlso-320K</td><td>69.86</td><td>0.25</td><td>18.40</td><td>88.52</td></tr><tr><td>LF-Wikipedia-500K</td><td>50.26</td><td>0.45</td><td>39.00</td><td>89.72</td></tr><tr><td>LF-WikiHierarchy-1M</td><td>19.33</td><td>1.04</td><td>15.54</td><td>35.91</td></tr><tr><td>LF-AmazonTitles-1.3M</td><td>93.50</td><td>0.73</td><td>108.70</td><td>202.93</td></tr></table>