# Membership Inference Attacks From First Principles

Nicholas Carlini $^{*1}$ Steve Chien $^{1}$ Milad Nasr $^{1,2}$ Shuang Song $^{1}$ Andreas Terzis $^{1}$ Florian Tramèr $^{1}$

$^{1}$ Google Research $^{2}$ University of Massachusetts Amherst

Abstract—A membership inference attack allows an adversary to query a trained machine learning model to predict whether or not a particular example was contained in the model's training dataset. These attacks are currently evaluated using average-case “accuracy” metrics that fail to characterize whether the attack can confidently identify any members of the training set. We argue that attacks should instead be evaluated by computing their true-positive rate at low (e.g., $\leq 0.1\%$ ) false-positive rates, and find most prior attacks perform poorly when evaluated in this way. To address this we develop a Likelihood Ratio Attack (LiRA) that carefully combines multiple ideas from the literature. Our attack is $10\times$ more powerful at low false-positive rates, and also strictly dominates prior attacks on existing metrics.

# I. INTRODUCTION

Neural networks are now trained on increasingly sensitive datasets, and so it is necessary to ensure that trained models are privacy-preserving. In order to empirically verify if a model is in fact private, membership inference attacks $[60]$ have become the de facto standard $[42, 63]$ because of their simplicity. A membership inference attack receives as input a trained model and an example from the data distribution, and predicts if that example was used to train the model.

Unfortunately as noted by recent work [44, 69], many prior membership inference attacks use an incomplete evaluation methodology that considers average-case success metrics (e.g., accuracy or ROC-AUC) that aggregate an attack's accuracy over an entire dataset and over all detection thresholds [6, 18, 26, 33–35, 45, 52, 54, 54–57, 61, 63, 66, 70]. However, privacy is not an average case metric, and should not be evaluated as such [65]. Thus, while existing membership inference attacks do appear effective when evaluated under this average-case methodology, we make the case they do not actually effectively measure the worst-case privacy of machine learning models.

Contributions. In this paper we re-examine the problem statement of membership inference attacks from first principles. We first argue that membership inference attacks should be evaluated by considering their true-positive rate (TPR) at low false-positive rates (FPR). This objective of designing methods around low false-positive rates is typical in many areas of computer security $[21, 27, 28, 31, 41, 49]$ , and for similar reasons it is the right metric here. If a membership inference attack can reliably violate the privacy of even just a few users in a sensitive dataset, it has succeeded. And conversely, an attack that only unreliably achieves high aggregate attack success rate should not be considered successful.

When evaluated this way, we find most prior attacks fail in the low false-positive rate regime. Furthermore, aggregate metrics (e.g., AUC) are often uncorrelated with low FP success rates. For example the attack of Yeom et al. [70] has a high accuracy (59.5%) yet fails completely at low FPRs, and the attack of Long et al. [36] has a much lower accuracy (53.5%) but achieves higher success rates at low FPRs.

![](images/a115be28b73ba85d2e6b45b58734fea3b56e90d33ea28fe8d5b34cbbf511116a.jpg)

<details>
<summary>line</summary>

| Method | Acc (%) |
| --- | --- |
| Ours. | 63.7 |
| Ye et al. | 60.0 |
| Sablayrolles et al. | 56.1 |
| Long et al. | 53.5 |
| Watson et al. | 59.1 |
| Shokri et al. | 59.5 |
| Song et al. | 59.5 |
| Yeom et al. | 59.5 |
| Jayaraman et al. | 59.0 |
</details>

Fig. 1: Comparing the true-positive rate vs. false-positive rate of prior membership inference attacks reveals a wide gap in effectiveness. An attack's average accuracy is not indicative of its performance at low FPRs. By extending on the most effective ideas, we improve membership inference attacks by $10 \times$ , for a non-overfit CIFAR-10 model (92% test accuracy).

We develop a Likelihood Ratio Attack (LiRA) that succeeds $10\times$ more often than prior work at low FPRs—but still strictly dominates prior attacks on aggregate metrics introduced previously. Our attack combines per-example difficulty scores [37, 56, 68] with a principled and well-calibrated Gaussian likelihood estimate. Figure 1 shows the success rate of our attack on a log-scale Receiver Operating Characteristic (ROC) curve [59], comparing the ratio of true-positives to false-positives. We perform an extensive experimental evaluation to understand each of the factors that contribute to our attack's success, and release our open source code. $^{1}$

Future work will need to re-examine many questions that have been studied using prior, much less effective, membership inference attacks. Attacks that use less information (e.g., label-only attacks $[6, 34, 54]$ ) may or may not achieve high success rate at low false-positive rates; algorithms previously seen as “private” because they resist prior attacks might be vulnerable to our new attack; and old defenses dismissed as ineffective might be able to defend against these new stronger attacks.

# II. BACKGROUND

We begin with a background that will be familiar to readers knowledgeable of machine learning privacy.

# A. Machine learning notation

A classification neural network $f_{\theta}: X \to [0,1]^{n}$ is a learned function that maps some input data sample $x \in X$ to an n-class probability distribution; we let $f(x)_{y}$ denote the probability of class y. Given a dataset D sampled from some underlying distribution D, we write $f_{\theta} \leftarrow \mathcal{T}(D)$ to denote that the neural network f parameterized with weights $\theta$ is learned by running the training algorithm T on the training set D. Neural networks are trained via stochastic gradient descent [32] to minimize some loss function $\ell$ :

$$
\theta_ {i + 1} \leftarrow \theta_ {i} - \eta \sum_ {(x, y) \in B} \nabla_ {\theta} \ell (f _ {\theta_ {i}} (x), y) \tag {1}
$$

Here, B is a batch of random training examples from D, and $\eta$ is the learning rate, a small constant. For classification tasks, the most common loss function is the cross-entropy loss:

$$
\ell (f _ {\theta} (x), y) = - \log (f _ {\theta} (x) _ {y}).
$$

When the weights $\theta$ are clear from context, we will simply write a trained model as f. At times it will be useful to view a model f as a function $f(x) = \sigma(z(x))$ , where $z : X \to R^{n}$ returns the feature outputs of the network, followed by a softmax normalization layer $\sigma(z) = [\frac{e^{z_{1}}}{\sum_{i} e^{z_{i}}}, \ldots, \frac{e^{z_{n}}}{\sum_{i} e^{z_{i}}}]$ .

Training neural networks that reach 100% training accuracy is easy—running the gradient descent from Equation 1 on any sufficiently sized neural network eventually achieves this goal [72]. The difficulty is in training models that generalize to an unseen test set $D_{test} \leftarrow D$ drawn from the same distribution. There are a number of techniques to increase the generalization ability of neural networks (augmentations [7, 67, 73], weight regularization [30], tuned learning rates [23, 38]). For the remainder of this paper, all models we train use state-of-the-art generalization-enhancing techniques. This makes our analysis much more realistic than prior work, which often uses models with $2-5\times$ higher error rates than our models.

# B. Training data privacy

Neural networks must not leak details of their training datasets, particularly when used in privacy-sensitive scenarios $[5, 13]$ . The field of training data privacy constructs attacks that leak data, develops techniques to prevent memorization, and measures the privacy of proposed defenses.

a) Privacy attacks: There are various forms of attacks on the privacy of training data. Training data extraction [4] is an explicit attack where an adversary recovers individual examples used to train the model. In contrast, model inversion attacks recover aggregate details of particular sub-classes instead of individual training examples [16]. Finally, property inference attacks aim at inferring non-trivial properties of the training dataset. For example, a classifier trained on bitcoin logs can reveal whether or not the machines that generated the logs were patched for Meltdown and Spectre [17].

We focus on a more fundamental attack that predicts if a particular example is part of a training dataset. First explored as tracing attacks $[11, 12, 22, 59]$ on medical datasets, they were extended to machine learning models as membership inference attacks $[60]$ . In these settings, being able to reliably (with high precision) identify a few users as being contained in sensitive medical datasets is itself a privacy violation $[22]$ —even if this is done with low recall. Further, membership inference attacks are the foundation of stronger extraction attacks $[3, 4]$ , and in order to be used in this way must again have exceptionally high precision.

b) Theory of memorization: The ability to perform membership inference is directly tied to a model's ability to memorize individual data points or labels. Zhang et al. [72] demonstrated that standard neural networks can memorize entirely randomly labeled datasets. A recent line of work initiated by Feldman [14] shows both theoretically and empirically that some amount of memorization may be necessary to achieve optimal generalization [2, 15].

c) Privacy-preserving training: The most widely deployed technique to make neural networks private is to make the learning process differentially private [10]. This can be done in various ways—for example by modifying the SGD algorithm [1, 64], or by aggregating results from a model ensemble [50]. Independent from differential privacy based defenses, there are other heuristic techniques (that is, without a formal proof of privacy) that have been developed to improve the privacy of machine learning models [26, 45]. Unfortunately, many of these have been shown to be vulnerable to more advanced forms of attack [6, 61].

d) Measuring training data privacy: Given a particular training scheme, a final direction of work aims to answer the question “how much privacy does this scheme offer?” Existing techniques often work by altering the training pipeline, either by injecting outlier canaries [3], or using poisoning to search for worst-case memorization [24, 47]. While these techniques give increasingly strong measurements of a trained model’s privacy, the fact that they require modifying the training pipeline creates an up-front cost to deployment. As a result, by far the most common technique used to audit machine learning models is to just use a membership inference attack. Existing membership inference attack libraries (see, e.g., Murakonda and Shokri [42], Song and Marn [63]) form the basis for most production privacy analysis [63], and it is therefore critical that they accurately assess the privacy of machine learning models.

# III. MEMBERSHIP INFERENCE ATTACKS

The objective of a membership inference attack (MIA) [60] is to predict if a specific training example was, or was not, used as training data in a particular model. This makes MIAs the simplest and most widely deployed attack for auditing training data privacy. It is thus important that they can reliably succeed at this task. This section formalizes the membership inference attack security game ( $§III-A$ ), and introduces our membership inference evaluation methodology ( $§III-B$ ).

# A. Definitions

We define membership inference via a standard security game inspired by Yeom et al. [70] and Jayaraman et al. [25].

Definition 1 (Membership inference security game). The game proceeds between a challenger C and an adversary A:

1) The challenger samples a training dataset $D \leftarrow \mathbb{D}$ and trains a model $f_{\theta} \leftarrow \mathcal{T}(D)$ on the dataset $D$ .   
2) The challenger flips a bit $b$ , and if $b = 0$ , samples a fresh challenge point from the distribution $(x, y) \leftarrow \mathbb{D}$ (such that $(x, y) \notin D$ ). Otherwise, the challenger selects a point from the training set $(x, y) \leftarrow {}^{\$} D$ .   
3) The challenger sends $(x,y)$ to the adversary.   
4) The adversary gets query access to the distribution $\mathbb{D}$ , and to the model $f_{\theta}$ , and outputs a bit $\hat{b} \leftarrow \mathcal{A}^{\mathbb{D},f}(x,y)$ .   
5) Output 1 if $\hat{b}=b$ , and 0 otherwise.

For simplicity, we will write $\mathcal{A}(x,y)$ to denote the adversary's prediction on the sample $(x,y)$ when the distribution $\mathbb{D}$ and model $f$ are clear from context.

Note that this game assumes that the adversary is given access to the underlying training data distribution D; while some attacks do not make use of this assumption [70], many attacks require query-access to the distribution in order to train “shadow models” [60] (as we will describe). The above game also assumes that the adversary is given access to both a training example and its ground-truth label.

Instead of outputting a “hard prediction”, all the attacks we consider output a continuous confidence score, which is then thresholded to yield a membership prediction. That is,

$$
\mathcal {A} (x, y) = \mathbb {1} [ \mathcal {A} ^ {\prime} (x, y) > \tau ]
$$

where 1 is the indicator function, $\tau$ is some tunable decision threshold, and $A'$ outputs a real-valued confidence score.

A first membership inference attack. For illustrative purposes, we begin by considering a very simple membership inference attack (due to Yeom et al. [70]). This attack relies on the observation that, because machine learning models are trained to minimize the loss of their training examples (see Equation 1), examples with lower loss are on average more likely to be members of the training data. Formally, the LOSS membership inference attack defines

$$
\mathcal {A} _ {\text { loss }} (x, y) = \mathbb {1} [ - \ell (f (x), y) > \tau ].
$$

# B. Evaluating membership inference attacks

Prior work lays out several strategies to determine the effectiveness of a membership inference attack, i.e., how to measure the adversary's success in Definition 1. We now show that existing evaluation methodologies fail to characterize whether an attack succeeds at confidently predicting membership. We thus propose a more suitable evaluation procedure.

As a running example for the remainder of this section, we train a standard CIFAR-10 [29] ResNet [19] to 92% test accuracy by training it on half of the dataset (i.e., 25,000 examples)—leaving another 25,000 examples for evaluation as non-members. While this dataset is not sensitive, it serves as a strong baseline for understanding properties of machine learning models in general. We train this model using standard techniques to reduce overfitting, including weight decay [30], train-time augmentations [7], and early stopping. As a result, this model has only a 8% train-test accuracy gap.

Balanced Attack Accuracy. The simplest method to evaluate attack efficacy is through a standard “accuracy” metric that measures how often an attack correctly predicts membership on a balanced dataset of members and non-members $[6, 18, 33, 46, 56, 60, 61, 66, 68, 70]$ .

Definition 2. The balanced attack accuracy of a membership inference attack A in Definition 1 is defined as

$$
\operatorname * {P r} _ {x, y, f, b} [ \mathcal {A} ^ {\mathbb {D}, f} (x, y) = b ].
$$

Even though balanced accuracy is used in many papers to evaluate membership inference attacks, we argue that this metric is inherently inadequate for multiple reasons:

- Balanced accuracy is symmetric. That is, the metric assigns equal cost to false-positives and to false-negatives. However, in practice, adversaries often only care about one of these two sources of errors. For example, when a membership inference attack is used in a training data extraction attack [4], false negatives are benign (some data will not be successfully extracted) whereas false-positives directly reduce the utility of the attack.   
- Balanced accuracy is an average-case metric, but this is not what matters in security. Consider comparing two attacks. Attack A perfectly targets a known subset of 0.1% of users, but succeeds with a random 50% chance on the rest. Attack B succeeds with 50.05% probability on any given user. On average, these two attacks have the same attack success rate (and thus the same balanced accuracy). However, the second attack is practically useless, while the first attack is exceptionally potent.

We now illustrate how exactly these issues arise for the simple LOSS attack described above. For our CIFAR-10 model, this attack's balanced accuracy is $60\%$ . This is (much) better than random guessing, and so one might reasonably conclude that the attack is useful and practically worrying.

However, this attack completely fails at confidently identifying any members! Let's examine for the moment the 1% of samples from the CIFAR-10 dataset with lowest losses $\ell(f(x), y)$ . These are the samples where the attack is most confident that they are members. Yet, on this subset, the attack is only correct 48% of the time (worse than random guessing). In contrast, for the 1% samples with highest loss (confident non-members), the attack is correct 100% of the time. Thus, the LOSS attack is actually a strong non-membership inference attack, and is practically useless at inferring membership. An attack with the symmetrical property (i.e., the attack confidently identifies members, but not non-members) is a much stronger attack on privacy, yet it achieves the same balanced accuracy.

ROC Analysis. Instead of the balanced accuracy, we should thus consider metrics that emphasize positive predictions (i.e., membership guesses) over negative (non-membership) predictions. A natural choice is to consider the tradeoff between the true-positive rate (TPR) and false-positive rate (FPR). Intuitively, an attack should maximize the true-positive rate (many members are identified), while incurring few false-positives (incorrect membership guesses). We prefer this to a precision/recall analysis because TPR/FPR is independent of the (often unknown) prevalence of members in the population.

The TPR/FPR tradeoff is fully characterized by the Receiver Operating Characteristic (ROC) curve, which compares the attack's TPR and FPR for all possible choices of the decision threshold $\tau$ . In Figure 2a, we show the ROC curve for the LOSS attack. The attack fails to achieve a TPR better than random chance at any FPR below $20\%$ —it is therefore ineffective at confidently breaching the privacy of its members.

![](images/f4d69f7350e52eb2c8d946f76272b9920a15cb44dd53b542539c3aba4ea120c5.jpg)

<details>
<summary>line</summary>

| False Positive Rate | True Positive Rate |
| ------------------- | ------------------ |
| 0.00                | 0.00               |
| 0.25                | 0.25               |
| 0.50                | 0.50               |
| 0.75                | 0.75               |
| 1.00                | 1.00               |
</details>

(a) linear scale

![](images/95cdf6f7ed214c7f7d9c91ad03343890fdd7176030842b6aedb3eaab1ca02967.jpg)

<details>
<summary>line</summary>

| False Positive Rate | True Positive Rate |
| ------------------- | ------------------ |
| 1e-5                | 1e-5               |
| 1e-4                | 1e-4               |
| 1e-3                | 1e-3               |
| 1e-2                | 1e-2               |
| 1e-1                | 1e-1               |
</details>

(b) log scale   
Fig. 2: ROC curve for the LOSS baseline membership inference attack, shown with both linear scaling (left), also and log-log scaling (right) to emphasize the low-FPR regime.

Prior papers that do report ROC curves summarize them by the AUC (Area Under the Curve) [20, 39, 43, 57, 68, 69]. However, as we can see from the curves above, the AUC is not an appropriate measure of an attack's efficacy, since the AUC averages over all false-positive rates, including high error rates that are irrelevant for a practical attack. The TPR of an attack when the FPR is above $50\%$ is not meaningfully useful, yet this regime accounts for more than half of its AUC score.

To illustrate, consider our hypothetical Attack A from earlier that confidently identifies 0.1% of members, but makes no confident predictions for any other samples. This attack perfectly breaches the privacy of some members, but has an AUC $\approx$ 51%—lower than the AUC of the weak LOSS attack.

True-Positive Rate at Low False-Positive Rates. Our recommended evaluation of membership inference attacks is thus to report an attack's true-positive rate at low false-positive rates.

Prior work occasionally reports true-positive rates at moderate false-positive rates (or reports precision/recall values that can be converted into TPR/FPR rates if the prevalence is known). For example, Shokri et al. [60] frequently reports that the “recall is almost 1” however there is a meaningful FPR difference between a recall of 1.0 and 0.999. Other works consistently report precision/recall values, but for equivalent false-positive rates between 3% and 40%, which we argue is too high to be practically meaningful.

In this paper, we argue for studying the extremely low false-positive regime. We do this by (1) reporting full ROC curves in logarithmic scale (see Figure 2b); and (2) optionally summarizing an attack's success rate by reporting its TPR at a fixed low FPR (e.g., $0.001\%$ or $0.1\%$ ). For example, the LOSS attack achieves a TPR of $0\%$ at an FPR of $0.1\%$ (worse than chance). While summarizing an attack's performance at a single choice of (low) FPR can be useful for quickly comparing attack configurations, we encourage future work to always also report full (log-scale) ROC curves as we do.

# IV. THE LIKELIHOOD RATIO ATTACK (LIRA)

# A. Membership inference as hypothesis testing

The game in Definition 1 requires the adversary to distinguish between two “worlds”: one where f is trained on a randomly sampled dataset that contains a target point $(x, y)$ , and one where f is not trained on $(x, y)$ . It is thus natural to see a membership inference attack as performing a hypothesis test to guess whether or not f was trained on $(x, y)$ .

We formalize this by considering two distributions over models: $\mathbb{Q}_{\mathrm{in}}(x,y) = \{f\gets \mathcal{T}(D\cup \{(x,y)\})\mid D\gets \mathbb{D}\}$ is the distribution of models trained on datasets containing $(x,y)$ , and then $\mathbb{Q}_{\mathrm{out}}(x,y) = \{f\gets \mathcal{T}(D\backslash \{(x,y)\})\mid D\gets \mathbb{D}\}$ . Given a model $f$ and a target example $(x,y)$ , the adversary's task is to perform a hypothesis test that predicts if $f$ was sampled either from $\mathbb{Q}_{\mathrm{in}}$ or if it was sampled from $\mathbb{Q}_{\mathrm{out}}$ [59].

We perform this hypothesis test according to the Neyman-Pearson lemma [48], which states that the best hypothesis test at a fixed false positive rate is obtained by thresholding the Likelihood-ratio Test between the two hypotheses:

$$
\Lambda (f; x, y) = \frac {p (f \mid \mathbb {Q} _ {\text { in }} (x , y))}{p (f \mid \mathbb {Q} _ {\text { out }} (x , y))}, \tag {2}
$$

where $p(f \mid \mathbb{Q}_b(x, y))$ is the probability density function over $f$ under the (fixed) distribution of model parameters $\mathbb{Q}_b(x, y)$ .

Unfortunately the above test is intractable: even the distributions $\mathbb{Q}_{\mathrm{in}}$ and $\mathbb{Q}_{\mathrm{out}}$ are not analytically known. To simplify the situation, we instead define $\tilde{\mathbb{Q}}_{\mathrm{in}}$ and $\tilde{\mathbb{Q}}_{\mathrm{out}}$ as the distributions of losses on $(x,y)$ for models either trained, or not trained, on this example. Then, we can replace both probabilities in Equation 2 with the easy-to-calculate quantity

$$
p (\ell (f (x), y) \mid \tilde {\mathbb {Q}} _ {\text { in / out }} (x, y)). \tag {3}
$$

This is now a likelihood test for a one-dimensional statistic, which can be efficiently computed with query access to $f$ .

Our attack follows the above intuition. We train several “shadow models” in order to directly estimate the distribution $\tilde{Q}_{in/out}$ . To minimize the number of shadow models necessary, we assume $\tilde{Q}_{in/out}$ is a Gaussian distribution, reducing our attack to estimating just four parameters: the mean and variance of each distribution. To run our inference attack on any model f, we can compute its loss on $\ell(f(x), y)$ , measure the likelihood of this loss under each of the distributions $\tilde{Q}_{in}$ and $\tilde{Q}_{out}$ , and return whichever is more likely.

![](images/62055e131125af2fb235d682097a74e955eb621570e67055ce5dac9bba081af0.jpg)

<details>
<summary>histogram</summary>

| Category             | member | non-member |
| -------------------- | ------ | ---------- |
| easy to fit / inlier | 100    | 50         |
| easy to fit / outlier | 20     | 150        |
| hard to fit / inlier | 50     | 30         |
| hard to fit / outlier | 100    | 200        |
</details>

Fig. 3: Some examples are easier to fit than others, and some have a larger separability between their losses when being a member of the training set or not. We train 1024 models on random subsets of CIFAR-10 and plot the losses for four examples when the example is a member of the training set ( $\tilde{\mathbb{Q}}_{\mathrm{in}}(x,y)$ , in red) or not ( $\tilde{\mathbb{Q}}_{\mathrm{out}}(x,y)$ , in blue).

# B. Memorization and per-example hardness

By casting membership inference as a Likelihood-ratio test, it becomes clear why the LOSS attack (and those that build on it) are ineffective: by directly thresholding the quantity $\ell(f(x), y)$ , this attack implicitly assumes that the losses of all examples are a priori on an equal scale, and that the inclusion or exclusion of one example will have a similar effect on the model as any other example. That is, if we measure $\ell(f(x), y) < \ell(f(x'), y')$ then the LOSS attack predicts that $(x, y)$ is more likely to be a member than $(x', y')$ —regardless of any other properties of these examples.

Feldman and Zhang [15] show that not all examples are equal: some examples (“outliers”) have an outsized effect on the learned model when inserted into a training dataset, compared to other (“inlier”) examples. To replicate their experiment, we choose a training dataset D and sample a random subset $D_{in} \subset D$ containing half of the dataset. We train a model on this dataset $f \leftarrow \mathcal{T}(D_{\text{in}})$ , and evaluate the loss on every example $(x, y) \in D$ , annotated by whether or not $(x, y)$ was in the training set $D_{in}$ . We repeat the above experiment hundreds of times, thereby empirically estimating the distributions $p(\ell(f(x), y) \mid \tilde{\mathbb{Q}}_{\text{in/out}}(x, y))$ by sampling.

Figure 3 plots histograms of model losses on four CIFAR-10 images when the image is contained in the model's training dataset (red) and when it is absent (blue). We chose these images to illustrate two different axes of variation. On the columns we compare “inliers” to “outliers”, as determined by the model’s loss when not trained on the example. The left column shows examples with low loss when omitted from the training set, those in the right column have high loss. On rows we compare how easy the examples are to fit. The examples in the top row have very low loss when trained on, while the examples in the bottom row have higher loss. Importantly, observe that these two dimensions do measure different quantities. An example can be an outlier but easy to fit (upper right), or an inlier but hard to fit (lower left).

![](images/deeb035068714664a09b13920fcba2364d82a4378cbfdc934961f89dfe9635ca.jpg)

<details>
<summary>histogram</summary>

| Category          | Bin Range | Frequency |
| ----------------- | --------- | --------- |
| confidence        | 0.0 - 1.0 | ~200      |
| CE loss           | 10^-7     | ~50       |
| logit scaling    | -20 to 20 | ~50       |
</details>

Fig. 4: The model's confidence, or its logarithm (the cross-entropy loss) are not normally distributed. Applying the logit function yields values that are approximately normal.

The goal of a membership inference adversary is to distinguish the two distributions in Figure 3 for a given example. This view illustrates the shortcomings of prior attacks (e.g., the LOSS attack): a global threshold on the observed loss $\ell(f(x), y)$ cannot distinguish between the different scenarios in Figure 3. The only confident assessment that such an attack can make is that examples with high loss are non-members. In contrast, the Likelihood-ratio test in Equation (2) considers the hardness of each example individually by modeling separate pairs of distributions $\tilde{\mathbb{Q}}_{\text{in}}, \tilde{\mathbb{Q}}_{\text{out}}$ for each example $(x, y)$ .

# C. Estimating the likelihood-ratio with parametric modeling

We directly turn this observation into a membership inference attack by computing per-example hardness scores [37, 56, 68, 69]. By training models on random samples of data from the distribution D, we obtain empirical estimates of the distributions $\tilde{Q}_{in}$ and $\tilde{Q}_{out}$ for any example $(x,y)$ . And from here, we can estimate the likelihood from Equation 3 to predict if an example is a member of the training dataset or not.

To improve performance at very low false-positive rates, instead of empirically modeling the distributions $\tilde{Q}_{in/out}$ directly from the data, we opt for a parametric and model $\tilde{Q}_{in/out}$ by Gaussian distributions. Parametric modeling has several significant benefits over nonparametric modeling.

- Parametric modeling requires training fewer shadow models to achieve the same generalization of nonparametric approaches. For example, we can match the recent (nonparametric) work of [69] with $400\times$ fewer models.   
- We can extend our attack to multivariate parametric models, allowing us to further improve attack success rate by querying the model multiple times ( $\S$ VI-C).

Doing this requires some care. Indeed, as can be seen in Figure 3, the model's cross-entropy loss is not well approximated by a normal distribution. First, the cross-entropy loss is on a logarithmic scale. If we take the negative exponent, $\exp(-\ell(f(x), y))$ , we instead obtain the model "confidence" $f(x)_y$ , which is bounded in the interval [0,1] and thus not normally distributed either (i.e., the confidences for outliers and inliers concentrate, respectively, around 0 and 1). We thus apply a logit scaling to the model's confidence,

$$
\phi (p) = \log \left(\frac {p}{1 - p}\right), \quad \text { for } p = f (x) _ {y}
$$

Algorithm 1 Our online Likelihood Ratio Attack (LiRA). We train shadow models on datasets with and without the target example, estimate mean and variance of the loss distributions, and compute a likelihood ratio test. (In our offline variant, we omit lines 5, 6, 10, and 12, and instead return the prediction by estimating a single-tailed distribution, as is shown in Equation (4).)

Require: model f, example $(x,y)$ , data distribution D
1: $conf_{in} = \{\}$ 2: $conf_{out} = \{\}$ 3: for N times do
4: $D_{attack} \leftarrow ^{S} D$ ▷ Sample a shadow dataset
5: $f_{in} \leftarrow \mathcal{T}(D_{attack} \cup \{(x,y)\})$ ▷ train IN model
6: $conf_{in} \leftarrow conf_{in} \cup \{\phi(f_{in}(x)_y)\}$ 7: $f_{out} \leftarrow \mathcal{T}(D_{attack} \setminus \{(x,y)\})$ ▷ train OUT model
8: $conf_{out} \leftarrow conf_{out} \cup \{\phi(f_{out}(x)_y)\}$ 9: end for
10: $\mu_{in} \leftarrow mean(conf_{in})$ 11: $\mu_{out} \leftarrow mean(conf_{out})$ 12: $\sigma_{in}^{2} \leftarrow var(conf_{in})$ 13: $\sigma_{out}^{2} \leftarrow var(conf_{out})$ 14: $conf_{obs} = \phi(f(x)_y)$ ▷ query target model
15: return $\Lambda = \frac{p(conf_{obs} | \mathcal{N}(\mu_{in}, \sigma_{in}^{2}))}{p(conf_{obs} | \mathcal{N}(\mu_{out}, \sigma_{out}^{2}))}$

to obtain a statistic in the range $(-\infty, \infty)$ that is (empirically) approximately normal. Figure 4 displays the distributions of model confidences, the negative log of the confidences (the cross-entropy loss), and the logit of the confidences. Only the logit approach is well approximated by a pair of Gaussians.

Our complete online attack (Algorithm 1). We first train N shadow models [60] on random samples from the data distribution D, so that half of these models are trained on the target point $(x, y)$ , and half are not (we call these respectively IN and OUT models for $(x, y)$ ). We then fit two Gaussians to the confidences of the IN and OUT models on $(x, y)$ (in logit scale). Finally, we query the confidence of the target model f on $(x, y)$ and output a parametric Likelihood-ratio test.

This attack is easily parallelized across multiple target points. Given a dataset $D \leftarrow D$ , we train shadow models on N subsets of D, chosen so that each target $(x, y) \in D$ appears in N/2 subsets. The same N shadow models can then be used to estimate the Likelihood-ratio test for all examples in D.

As an optimization, we can improve the attack by querying the target model on multiple points $x_{1}, x_{2}, \ldots, x_{m}$ obtained by applying standard data augmentations to the target point x (as previously observed in [6]). In this case, we fit m-dimensional spherical Gaussians $\mathcal{N}(\boldsymbol{\mu}_{\mathrm{in}}, \boldsymbol{\sigma}_{\mathrm{in}}^{2}I), \mathcal{N}(\boldsymbol{\mu}_{\mathrm{out}}, \boldsymbol{\sigma}_{\mathrm{out}}^{2}I)$ to the losses collected from querying the shadow models m times per example, and compute a standard likelihood-ratio test between two multivariate normal distributions.

Our offline attack. While our online attack is effective, it has a significant usability limitation: it requires the adversary train new models after they are told to infer the membership of the example $(x, y)$ . This requires training new machine learning models for every (batch of) membership inference queries, and is computationally expensive.

To improve the efficiency of our attack, we propose an offline attack algorithm that trains shadow models on randomly sampled datasets ahead of time, and never trains shadow models on the target points. For this attack, we remove lines 5,6,10 and 12 from Algorithm 1, and only estimate the mean $\mu_{\mathrm{out}}$ and variance $\sigma_{\mathrm{out}}^2$ of model confidences when the target example is not in the shadow models' training data. We then change the likelihood-ratio test in line 15 to a one-sided hypothesis test. That is, we measure the probability of observing a confidence as high as the target model's under the null-hypothesis that the target point $(x,y)$ is a non-member:

$$
\Lambda = 1 - \operatorname * {P r} [ Z > \phi (f (x) _ {y}) ], \text { where } Z \sim \mathcal {N} (\mu_ {\mathrm{out}}, \sigma_ {\mathrm{out}} ^ {2}). \tag {4}
$$

The larger the target model's confidence is compared to $\mu_{\mathrm{out}}$ , the higher the likelihood that the query sample is a member. Similar to our online attack, we improve the attack by querying on multiple augmentations and fitting a multivariate normal.

# V. ATTACK EVALUATION

We now investigate our offline and online attack variants in a thorough evaluation across datasets and ML techniques.

Again, we focus extensively on the low-false positive rate regime. This is the setting with the most practical consequences: for example, to extract training data $[4]$ it is far more important for attacks to have a low false positive rate than high average success, as false positives are fare more costly than false negatives. Similarly, de-identifying even a few users contained in a sensitive dataset is far more important than saying an average-case statement “most people are probably not contained in the sensitive dataset”.

We use both datasets traditionally used for membership inference attack evaluations, but also new datasets that are less typically used. In addition to the CIFAR-10 dataset introduced previously, we also consider three other datasets: CIFAR-100 [29] (another standard image classification task), ImageNet [9] (a standard challenging image classification task) and WikiText-103 [40] (a natural language processing text dataset). For CIFAR-100, we follow the same process as for CIFAR-10 and train a wide ResNet [71] to 60% accuracy on half of the dataset (25,000 examples). For ImageNet, we train a ResNet-50 on 50% of the dataset (roughly half a million examples). For WikiText-103, we use the GPT-2 tokenizer [53] to split the dataset into a million sentences and train a small GPT-2 [53] model on 50% of the dataset for 20 epochs to minimize the cross-entropy loss. Prior work has additionally performed experiments on two toy datasets that we do not believe are meaningful benchmarks for privacy because of their simplicity: Purchase and Texas (see [60] for details). $^{2}$

![](images/2d85ec32b6090f926b5a3387293933f40877bdff37bc00358004ec7cd5bc0ee1.jpg)

<details>
<summary>line</summary>

| Dataset     | AUC    |
| ----------- | ------ |
| CIFAR-100   | 0.925  |
| CIFAR-10    | 0.720  |
| ImageNet    | 0.765  |
| WikiText    | 0.715  |
</details>

Fig. 5: Success rate of our attack on CIFAR-10, CIFAR-100, ImageNet, and WikiText. All plots are generated with 256 shadow models, except ImageNet which uses 64.

For each dataset, the adversary trains N shadow models (N = 64 for ImageNet, and N = 256 otherwise) on training sets chosen so that each example $(x, y)$ is contained in exactly half of the shadow models' training sets (thus, for each example we have N/2 IN models, and N/2 OUT models). We use the entire dataset for this purpose, and thus the training sets of individual shadow models and the target model may partially overlap. This is a strong assumption, which we make here mainly due to the small size of some of the datasets we consider. In Section VI-D, we show that our attack works just as well when the adversary trains shadow models on datasets that are fully disjoint from the target model's training set.

For all datasets except ImageNet, we repeat each attack 10 times and report the attack success rates across all 10 attacks.

# A. Online attack evaluation

Figure 5 presents the main results of our online attack when evaluated on the four more complex of the datasets mentioned above (CIFAR-10, CIFAR-100, ImageNet, and WikiText-103). Even though these datasets are complex, it is relatively efficient to train most of these models—for example a CIFAR-10 or CIFAR-100 model takes just six minutes to train. Additional results for the Purchase and Texas dataset are given in the Appendix—these datasets are much simpler and while they are typically used for membership inference, we argue they are too simple to have generalizable lessons.

Our attack has true-positive rates ranging from $0.1\%$ to $10\%$ at a false-positive rate of $0.001\%$ . If we compare the three image datasets, consistent with prior works, we find that the attack's average success rate (i.e., the AUC) is correlated directly with the generalization gap of the trained model. All three models have perfect $100\%$ training accuracy, but the test accuracy of the CIFAR-10 model is $90\%$ , the ImageNet model is $65\%$ , and the CIFAR-100 model is $60\%$ . Yet, at low false-positives, the CIFAR-10 models are easier to attack than the ImageNet models, despite their better generalization.

![](images/2874d76338f8a167e4f62a1b820e399b21153609727f69ee77fc29b61c9ed5af.jpg)

<details>
<summary>line</summary>

| Model          | AUC    |
| -------------- | ------ |
| CIFAR-100      | 0.859  |
| CIFAR-10       | 0.674  |
| ImageNet       | 0.728  |
| WikiText-103   | 0.713  |
</details>

Fig. 6: Success rate of our offline attack on CIFAR-10, CIFAR-100, ImageNet, and WikiText. All plots are generated with 128 OUT shadow models, except ImageNet which uses 32. For each dataset, we also plot our online attack with the same number of shadow models (half IN, half OUT).

# B. Offline attack evaluation

Figure 6 evaluates our offline attack from Section IV-C, where the adversary performs the costly operations of training shadow models only before being handed the target query point $(x,y)$ . Our attack performs only slightly worse in this offline setting—at an FPR of $0.1\%$ , our offline attack's TPR is at most $20\%$ lower than that of our best online attack with the same number of shadow models.

# C. Re-evaluating prior membership inference attacks

In order to understand how our attack compares to prior work, we now re-evaluate prior attack techniques under our low-FPR objective. We study these attacks following the same evaluation protocol introduced above and on the same datasets (for WikiText, we omit a few entries for attacks that are not directly applicable to sequential language models).

A summary of our analysis is presented in Table I. We compare the efficacy of eight representative attacks from the literature. For each attack, we compute a full ROC curve and select a decision threshold that maximizes TPR at a given FPR. Surprisingly, we find that despite being published in 2019, the attack of Sablayrolles et al. [56] outperforms other attacks under our metric (often by an order of magnitude), even when compared to more recent attacks such as Jayaraman et al. [25] (PETS'21) and Song and Mittal [61] (USENIX'21).

Shadow models. One of the first membership inference attacks (due to Shokri et al. [60]) that improves on the baseline LOSS attack, introduced the idea of shadow models, but used in a simpler way than we have done here. Each shadow model $f_{i}$ (of a similar type to the target model $f$ ) is trained on random subsets $D_{i}$ of training data available to the adversary. The attack then trains a new neural network $g$ to predict an example's membership status. Given the pre-softmax features $f_{i}(x)$ and class label $y$ , the model $g$ predicts whether the data

<table><tr><td rowspan="2">Method</td><td rowspan="2">shadow models</td><td rowspan="2">multiple queries</td><td rowspan="2">class hardness</td><td rowspan="2">example hardness</td><td colspan="3">TPR @ 0.001% FPR</td><td colspan="3">TPR @ 0.1% FPR</td><td colspan="3">Balanced Accuracy</td></tr><tr><td>C-10</td><td>C-100</td><td>WT103</td><td>C-10</td><td>C-100</td><td>WT103</td><td>C-10</td><td>C-100</td><td>WT103</td></tr><tr><td>Yeom et al. [70]</td><td>○</td><td>○</td><td>○</td><td>○</td><td>0.0%</td><td>0.0%</td><td>0.00%</td><td>0.0%</td><td>0.0%</td><td>0.1%</td><td>59.4%</td><td>78.0%</td><td>50.0%</td></tr><tr><td>Shokri et al. [60]</td><td>●</td><td>○</td><td>●</td><td>○</td><td>0.0%</td><td>0.0%</td><td>-</td><td>0.3%</td><td>1.6%</td><td>-</td><td>59.6%</td><td>74.5%</td><td>-</td></tr><tr><td>Jayaraman et al. [25]</td><td>○</td><td>●</td><td>○</td><td>○</td><td>0.0%</td><td>0.0%</td><td>-</td><td>0.0%</td><td>0.0%</td><td>-</td><td>59.4%</td><td>76.9%</td><td>-</td></tr><tr><td>Song and Mittal [61]</td><td>●</td><td>○</td><td>●</td><td>○</td><td>0.0%</td><td>0.0%</td><td>-</td><td>0.1%</td><td>1.4%</td><td>-</td><td>59.5%</td><td>77.3%</td><td>-</td></tr><tr><td>Sablayrolles et al. [56]</td><td>●</td><td>○</td><td>●</td><td>●</td><td>0.1%</td><td>0.8%</td><td>0.01%</td><td>1.7%</td><td>7.4%</td><td>1.0%</td><td>56.3%</td><td>69.1%</td><td>65.7%</td></tr><tr><td>Long et al. [37]</td><td>●</td><td>○</td><td>●</td><td>●</td><td>0.0%</td><td>0.0%</td><td>-</td><td>2.2%</td><td>4.7%</td><td>-</td><td>53.5%</td><td>54.5%</td><td>-</td></tr><tr><td>Watson et al. [68]</td><td>●</td><td>○</td><td>●</td><td>●</td><td>0.1%</td><td>0.9%</td><td>0.02%</td><td>1.3%</td><td>5.4%</td><td>1.1%</td><td>59.1%</td><td>70.1%</td><td>65.4%</td></tr><tr><td>Ye et al. [69]</td><td>●</td><td>○</td><td>●</td><td>●</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>60.3%</td><td>76.9%</td><td>65.5%</td></tr><tr><td>Ours</td><td>●</td><td>●</td><td>●</td><td>●</td><td>2.2%</td><td>11.2%</td><td>0.09%</td><td>8.4%</td><td>27.6%</td><td>1.4%</td><td>63.8%</td><td>82.6%</td><td>65.6%</td></tr></table>

TABLE I: Comparison of prior membership inference attacks under the same settings for well-generalizing models on CIFAR-10, CIFAR-100, and WikiText-103 using 256 shadow models. Accuracy is only presented for completeness; we do not believe this is a meaningful metric for evaluating membership inference attacks. Full ROC curves are presented in Appendix A.

point $(x,y)$ was a member of the shadow training set $D_{i}$ . For a target model f and point $(x,y)$ , the attack then outputs $g(f(x),y)$ as a membership confidence score.

We implement this by training shadow models that randomly subsample half of the total dataset. The training set of the shadow models thus partially overlaps with the training set of the target model f. This is a stronger assumption than that made by Shokri et al. [60] and thus yields a slightly stronger attack. Despite being significantly more expensive than the LOSS attack due to the overhead of training many shadow models and then training a membership inference predictor on the output of the models, this attack does not perform significantly better at low false-positive rates.

Multiple queries. It is possible to improve attacks by making multiple queries to the model. Jayaraman et al. [25] do this with their MERLIN attack, that queries the target model $f$ multiple times on a sample $x$ perturbed with fresh Gaussian noise, and measures how the model's loss varies in the neighborhood of $x$ . However, even when querying the target model 100 times and carefully choosing the noise magnitude, we find that this attack does not improve the adversary's success at low false-positive rates.

Choquette-Choo et al. [6] suggest an alternate technique to increase attack accuracy when models are trained with data augmentations. In addition to querying the model on $f(x)$ , this attack also queries on augmentations of x that the model might have seen during training. This is the direct motivation for us making these additional queries, which as we will show in Section VI-C improves our attack success rate considerably.

Per-class hardness. Instead of using per-example hardness scores as we have done, a potentially simpler method would be to design just one scoring function $A_{y}^{\prime}$ per class y, by scaling the model's loss by a class-dependent value: $\mathcal{A}_{y}^{\prime}(x,y)=\mathcal{A}^{\prime}(x,y)-\tau_{y}$ . For example, in the ImageNet dataset [9] there are several hundred classes for various breeds of dogs, and so correctly classifying individual dog breeds tends to be harder than other broader classes. Interestingly, despite this intuition, in practice using per-class thresholds neither helps improve balanced attack accuracy nor attack success rates at low false-positive rates, although it does improve the AUC of attacks on CIFAR-10 and CIFAR-100 by 2%.

The attack of Song and Mittal [61] reported in Figure 1 and Table I combines per-class scores with additional techniques. Instead of working with the standard cross-entropy loss, this attack uses a modified entropy measure and trains shadow models to approximate the distributions of entropy values for members and non-members of each class. Given a model f and target sample $(x, y)$ , the attack computes a hypothesis test between the (per-class) member and non-member distributions (see [61]). Despite these additional techniques, this attack does not improve upon the baseline attack [60] at low FPRs.

Per-example hardness. As we do in our work, a final direction considers per-example hardness. Sablayrolles et al. [56] is the most direct influence for LiRA. Their attack, $\mathcal{A}^{\prime}(x,y)=\ell(f(x),y)-\tau_{x,y}$ , scales the loss by a per-example hardness threshold $\tau_{x,y}$ that is estimated by training shadow models. Instead of fitting Gaussians to the shadow models' outputs as we do, this paper takes a simpler nonparametric approach and sets the threshold near the midpoint $\tau_{x,y}=(\mu_{\mathrm{in}}(x,y)+\mu_{\mathrm{out}}(x,y))/2$ so as to maximize the attack accuracy; here $\mu_{in},\mu_{out}$ are the means computed as we do.

The recent work of Watson et al. [68] considers an offline variant of Sablayrolles et al. [56], that sets $\tau_{x,y} = \mu_{\mathrm{out}}(x,y)$ (i.e., each example's loss is calibrated by the average loss of shadow models not trained on this example).

Both Sablayrolles et al. and Watson et al. evaluate their attacks using average case metrics (balanced accuracy and AUC), and find that using per-example hardness thresholds can moderately improve upon past attacks. In our evaluation (Table I), we find that the balanced accuracy and AUC of their approaches are actually slightly lower than those of other simpler attacks. Yet, we find that per-example hardness-calibrated attacks reach a significantly better true-positive rate at low false-positive rates—and are thus much better attacks according to our suggested evaluation methodology.

![](images/beeda32aa7c29faab16fd63dcfa093f7e59b0cd545e0a76a09fc39edb2980cbd.jpg)

<details>
<summary>scatter</summary>

| Model           | Train Test Gap | TPR @ 0.1% FPR |
| --------------- | -------------- | -------------- |
| CNN1, CNN2, CNN4 | 0.05           | 0.001          |
| CNN8            | 0.15           | 0.01           |
| CNN32, CNN64    | 0.25           | 0.05           |
| WRN28-1         | 0.1            | 0.005          |
| WRN28-2         | 0.2            | 0.01           |
| WRN28-10        | 0.3            | 0.05           |
</details>

Fig. 7: Attack true-positive rate versus model train-test gap for a variety of CIFAR-10 models.

The discrepancy between the balanced accuracy and our recommended low false-positive metric is even more stark for the attack of Long et al. [37]. This attack also trains shadow models to estimate per-example hardness, but additionally filters out a fraction of outliers to which the attack should be applied, and then makes no confident guesses for non-outliers. This attack thus cannot achieve a high average accuracy, yet outperforms most prior attacks at low false-positive rates.

To expand, this attack [37] builds a graph of all examples $x$ , where an edge between $x$ and $x'$ is weighted by the cosine similarity between the features $z(x)$ and $z(x')$ . Our implementation of this attack selects the $10\%$ of outliers with the largest distance to their nearest neighbor in this graph. For each such outlier $(x,y)$ , the attack trains shadow models to numerically estimate the probability of observing a loss as high as $\ell(f(x),y)$ when $(x,y)$ is not a member.

The attack in the concurrent work of Ye et al. [69] is close in spirit to ours. They follow the same approach as our offline attack, by training multiple OUT models and then performing an exact one-sided hypothesis test. Specifically, to target an FPR of $\alpha$ , their attack sets each example's decision threshold so that an $\alpha$ -fraction of the measured OUT losses for that example lie below the threshold.

The critical difference between our attack and these prior attacks is that we use a more efficient parametric approach, that models the distribution of losses as Gaussians. Since Sablayrolles et al. [56] and Watson et al. [68] only measure the means of the distributions, the attacks are sub-optimal if different samples' loss distributions have very different scales and spreads (c.f. Figure 4). The attacks of Long et al. [37] and Ye et al. [69] take into account the full distribution of OUT losses, but have difficulties extrapolating to low FPRs due to the lack of a parametric assumption. By design, the exact test of Ye et al. [69] can at best target an FPR of $1/N$ with $N$ shadow models. It is thus inapplicable in the setting we consider here (256 shadow models, and a target FPR of $0.1\%$ ). Long et al. [37] extrapolate to the tails of the empirical loss distribution using cubic splines, which easily overfit and diverge outside of their support.

<table><tr><td>Attack Approach</td><td>TPR @ 0.1% FPR</td></tr><tr><td>LOSS attack [70]</td><td>0.0%</td></tr><tr><td>+ Logit scaling</td><td>0.1%</td></tr><tr><td>+ Multiple queries</td><td>0.1%</td></tr><tr><td>LOSS attack [70]</td><td>0.0%</td></tr><tr><td>+ Per-example thresholds ( $\tilde{Q}_{out}$  only) [68]</td><td>1.3%</td></tr><tr><td>+ Logit scaling</td><td>4.7%</td></tr><tr><td>+ Gaussian Likelihood</td><td>4.7%</td></tr><tr><td>+ Multiple queries (our offline attack)</td><td>7.1%</td></tr><tr><td>LOSS attack [70]</td><td>0.0%</td></tr><tr><td>+ Per-example thresholds ( $\tilde{Q}_{in}$  &amp;  $\tilde{Q}_{out}$ ) [56]</td><td>1.7%</td></tr><tr><td>+ Logit scaling</td><td>1.9%</td></tr><tr><td>+ Gaussian Likelihood</td><td>5.6%</td></tr><tr><td>+ Multiple queries (our online attack)</td><td>8.4%</td></tr></table>

TABLE II: By iteratively adding the main components of our attack we can interpolate between the simple LOSS threshold attack [70] and our full offline and online attacks.

# D. Membership inference and overfitting

To better understand the relationship between overfitting and vulnerability to membership inference attacks, Figure 7 plots various models' train-test gap (that is, their train accuracy minus their test accuracy) versus our attack's TPR at an FPR of $0.1\%$ . We train CNN models and Wide ResNets (WRN) of various sizes on CIFAR-10, with different optimizers and data augmentations (see Section VI-E for details). Each point represents one training configuration for the target model.

While there is an overall trend that overfit models (those with higher train-test gap) are more vulnerable to attack, we do find examples of models that have identical train-test gaps but are $100 \times$ more vulnerable to attack. In Figure 16 in the Appendix we further plot the attack TPR as a function of the test accuracy of these models. There, we observe a clear trend that more accurate models are more vulnerable to attack.

# VI. ABLATION STUDY

Our attack has a number of moving pieces that are connected in various ways; in this section we investigate how these pieces come together to reach such high accuracy at low false-positive rates. We exclusively use CIFAR-10 for these ablation studies as it is the most popular image classification dataset and is the hardest datasets we have considered;

A summary of our analysis is presented in Table II. The baseline LOSS attack achieves a true-positive rate of 0% at a false positive rate of 0.1% (as shown previously in Figure 2b). If we do not use per-example thresholds, this basic attack can only be marginally improved by properly scaling the loss and issuing multiple queries to the target model.

By incorporating per-example thresholds obtained by estimating the distributions $\tilde{Q}_{in}$ and $\tilde{Q}_{out}$ as in [56], the attack success rate increases to 1.7%—about one-order-of-magnitude better than chance. By ensuring that we appropriately re-scale the model losses (explored in detail in Section VI-A) and fitting the re-scaled losses with Gaussians (see Section VI-B), we increase the attack success rate by a factor of $3.3\times$ . Finally, we can nearly double the attack success rate by evaluating the

![](images/c85baea3e3fc44fea97536b09bcf4baf9f721fcca9ea319297fdaafbb2e6fd57.jpg)

<details>
<summary>line</summary>

| False Positive Rate | f(x)_y (confidence) | log(f(x)_y) (CE loss) | φ(f(x)_y) (logit scale, unstable) | φ(f(x)_y), (logit scale, stable) | z(x)_y (output feature) | z(x)_y - max(z(x)_y') (Hinge) |
| ------------------- | --------------------- | ---------------------- | ---------------------------------- | --------------------------------- | ------------------------ | ------------------------------ |
| 10^-5               | ~10^-5                | ~10^-5                 | ~10^-4                             | ~10^-2                            | ~10^-2                   | ~10^-2                         |
| 10^-4               | ~10^-3                | ~10^-3                 | ~10^-2                             | ~10^-1                            | ~10^-1                   | ~10^-1                         |
| 10^-3               | ~10^-2                | ~10^-2                 | ~10^-1                             | ~10^0                             | ~10^0                    | ~10^0                          |
| 10^-2               | ~10^-1                | ~10^-1                 | ~10^0                              | ~10^0                             | ~10^0                    | ~10^0                          |
| 10^-1               | ~10^0                 | ~10^0                  | ~10^0                              | ~10^0                             | ~10^0                    | ~10^0                          |
| 10^0                | ~10^0                 | ~10^0                  | ~10^0                              | ~10^0                             | ~10^0                    | ~10^0                          |
</details>

Fig. 8: The best scoring metrics ensure the output distribution is approximately Gaussian, and the worst metrics are not easily modeled with a standard distribution (see Figure 4).

target model on the same data augmentations as used during training, as we will show in Section VI-C.

We also perform the same ablation but with the offline variant of our attack. Here, if we start with the attack of Watson et al. [68] to reach a $1.3\%$ true-positive rate; adding logit scaling, Gaussian likelihood, and multiple queries yields an attack that is nearly as strong as our full attack (TPR of $7.1\%$ versus $8.4\%$ at an FPR of $0.1\%$ ).

# A. Logit scaling the loss function

The first step of our attack projects the model's confidences to a logit scale to ensure that the distributions that we work with are approximately normal. Figure 8 compares performance of our attack for various choices of statistics that we can fit using shadow models. Recall that we defined our neural network function $f(x)$ to denote the evaluation of the model along with a final softmax activation function; we use $z(x)$ to denote the pre-softmax activations of the neural network.

As expected, we find that using the model's confidence $f(x)_y \in [0,1]$ , or its logarithm (the cross-entropy loss), leads to poor performance of the attack since these statistics do not behave like Gaussians (recall from Figure 4).

Our logit rescaling performs best, but the exact numerical computation of the logit function $\phi(p) = \log\left(\frac{p}{1-p}\right)$ matters. We consider two mathematically equivalent variants:

$$
\phi_ {\text { unstable }} = \log (f (x) _ {y}) - \log (1 - f (x) _ {y})
$$

$$
\phi_ {\text { stable }} = \log (f (x) _ {y}) - \log \sum_ {y ^ {\prime} \neq y} f (x) _ {y ^ {\prime}}.
$$

We find that the second version is more stable in practice, when the model's confidence is very high, $f(x)_y \approx 1$ (we compute all logarithms as $\log (x + \epsilon)$ for a small $\epsilon > 0$ ). Note that this second stable variant requires access to the full vector of model confidences $f(x) \in [0,1]^n$ rather than just the confidence of the predicted class.

![](images/288348a53ce2cd32f7366bd7469a4e031ae1f21fbdeaf8207419bcb9ffa1b22f.jpg)

<details>
<summary>line</summary>

| Number of shadow models | Gaussian LRT with per-example variance | Gaussian LRT with global variance | non-parametric (Sablayrolles et al.) |
| ----------------------- | -------------------------------------- | ---------------------------------- | ------------------------------------ |
| 4                       | 0.00                                   | 0.04                               | 0.015                                |
| 8                       | 0.01                                   | 0.05                               | 0.017                                |
| 16                      | 0.03                                   | 0.06                               | 0.018                                |
| 32                      | 0.05                                   | 0.07                               | 0.019                                |
| 64                      | 0.07                                   | 0.075                              | 0.019                                |
| 128                     | 0.08                                   | 0.08                               | 0.019                                |
| 256                     | 0.085                                  | 0.08                               | 0.019                                |
| 512                     | 0.085                                  | 0.08                               | 0.019                                |
</details>

Fig. 9: Attack success rate increases as the number of shadow models increases, with the benefit eventually tapering off. When fewer than 64 models are used, it is better to estimate the variance of the model confidence as a global parameter instead of computing it on a per-example basis.

If the adversary can query the model to obtain the unnormalized features $z(x)$ (i.e., the outputs of the model's last layer before the softmax function), a hinge loss performs similarly

$$
\ell_ {\text { Hinge }} (x, y) = z (x) _ {y} - \max _ {y ^ {\prime} \neq y} z (x) _ {y ^ {\prime}}.
$$

To see why this is the case, observe that

$$
\begin{array}{l} \phi (f (x) _ {y}) = \log (f (x) _ {y}) - \log \sum_ {y ^ {\prime} \neq y} f (x) _ {y ^ {\prime}} \\ = z (x) _ {y} - \underset {y ^ {\prime} \neq y} {\text { LogSumExp }}   z (x) _ {y ^ {\prime}}  , \\ \end{array}
$$

where the LogSumExp function is a smooth approximation to the maximum function. When the features $z(x)$ are available, we recommend using the hinge loss as its computation is numerically simpler than that of the logit-scaled confidence.

We note that the different attack variants we consider here lead to orders-of-magnitude differences in attack performance at low false-positive rates—even though all variants achieve similar AUC scores (68–72%). This again highlights the importance of carefully designing attacks, and of measuring attack performance at low false-positive rates rather than on average across the entire ROC curve.

The choice of an appropriate loss function can also have a major impact on previous MIAs. For example, for the attack of Watson et al. [68] (which scales the model's loss by the mean loss of OUT models not trained on the example, $\mu_{\mathrm{out}}(x,y)$ ) applying logit scaling nearly quadruples the attack's true-positive rate at an FPR of $0.1\%$ (see Table II).

# B. Gaussian distribution fitting

Like other shadow models membership inference attacks [60], our attack requires that we train enough models to accurately estimate the distribution of losses. It is thus desirable to minimize the number of shadow models that are necessary. However, most prior works in Table I that rely on

<table><tr><td rowspan="2">Queries</td><td colspan="2">TPR @ FPR</td></tr><tr><td>0.1%</td><td>0.001%</td></tr><tr><td>1 (no augmentations)</td><td>5.6%</td><td>1.0%</td></tr><tr><td>2 (mirror)</td><td>7.5%</td><td>1.8%</td></tr><tr><td>18 (mirror + shifts)</td><td>8.4%</td><td>2.2%</td></tr><tr><td>162 (mirror + shifts)</td><td>8.4%</td><td>2.2%</td></tr></table>

TABLE III: Querying on augmented versions of the image doubles the true-positive rate at low false-positive rates, with most benefits given by just two queries.

shadow models do not analyze this tradeoff and report results only for a fixed number of shadow models [37, 56, 60].

Figure 9 displays our online attack's TPR at a fixed FPR of $0.1\%$ , as we vary the number of shadow models (half IN and half OUT). Training more than 64 shadow models provides diminishing benefits, but the attack deteriorates quickly with fewer models—due to the difficulty of fitting Gaussian distributions on a small number of data points.

With a small number of shadow models, we can improve the attack considerably by estimating the variances $\sigma_{\mathrm{in}}^2$ and $\sigma_{\mathrm{out}}^2$ of model confidences in Algorithm 1 globally rather than for each individual example. That is, we still estimate the means $\mu_{\mathrm{in}}$ and $\mu_{\mathrm{out}}$ separately for each example, but we estimate the variance $\sigma_{\mathrm{in}}^2$ (respectively $\sigma_{\mathrm{out}}^2$ ) over the shadow models' confidences on all training set members (respectively non-members).

For a small number of shadow models (< 64), estimating a global variance outperforms our general attack that estimates the variance for each example separately. For a larger number of models, our full attack is stronger: with 1024 shadow models for example, the TPR decreases from 8.4% to 7.9% by using a global variance.

# C. Number of queries

Models are typically trained to minimize their loss not only on the original training example, but also on augmented versions of the example. It therefore makes sense to perform membership inference attacks on the augmented versions of the example that may have been seen during training. Results of this analysis are presented in Table III. There are 162 potential augmentations of each training image for our CIFAR-10 model ( $2 \times 9 \times 9$ , computed by either horizontally flipping the image or not, and shifting the image by up to $\pm4$ pixels in each height or width). We find that querying on just 2 augmentations gives most of the benefit, with increasing to 18 queries performing identically to all 162 augmentations.

# D. Disjoint datasets

In our experiments so far, we trained both the target models and the adversary's shadow models by subsampling from a common dataset. That is, we use a large dataset $D_{\text{attack}}$ (e.g., the entire CIFAR-10 dataset) to train shadow models, and the target model's training set $D_{\text{train}}$ is some (unknown) subset of this dataset. This setup favors the attacker, as the training sets of shadow models and the target model can partially overlap. In a real attack, the adversary likely has access to a dataset $D_{\text{attack}}$ that is disjoint from the training set $D_{\text{train}}$ . We now show that this more realistic setup has only a minor influence on the attack's success rate.

![](images/cdb5c4076497156914e0ed7ba8a53c9a56c4429852303a48697df0b45b7e0d56.jpg)

<details>
<summary>line</summary>

| False Positive Rate | True Positive Rate (D_train ⊂ D_attack) | True Positive Rate (D_train ∩ D_attack = Ø) | True Positive Rate (D_train ≠ D_attack) |
| ------------------- | ---------------------------------------- | -------------------------------------------- | ---------------------------------------- |
| 10^-4               | ~0.01                                    | ~0.01                                        | ~0.005                                   |
| 10^-3               | ~0.05                                    | ~0.05                                        | ~0.02                                    |
| 10^-2               | ~0.1                                     | ~0.1                                         | ~0.05                                    |
| 10^-1               | ~0.5                                     | ~0.5                                         | ~0.2                                     |
| 10^0                | 1.0                                      | 1.0                                          | 1.0                                      |
</details>

Fig. 10: The attack's success rate on CINIC-10 remains unchanged when the training sets of shadow models are sampled from a dataset $D_{\text{attack}}$ that is disjoint from the target model's training set $D_{\text{train}}$ . The attack's performance does decrease when the two datasets are sampled from different distributions.

For this experiment, we use the CINIC-10 dataset $[8]$ . This dataset combines CIFAR-10 with an additional 210k images taken from ImageNet that correspond to classes contained in CIFAR-10 (e.g., bird/airplane/truck etc). We train a target model and 128 shadow models (OUT models only) each on 50,000 points. We compare three attack setups:

1) The shadow models' training sets are sampled from the full CINIC-10 dataset. This is the same setup as in all our previous experiments, where $D_{\mathrm{train}} \subset D_{\mathrm{attack}}$ .   
2) The shadow models' training sets have no overlap with the target model, i.e., $D_{\mathrm{train}} \cap D_{\mathrm{attack}} = \emptyset$ .   
3) The target model is trained on CIFAR-10, while the attacker trains shadow models on the ImageNet portion of CINIC-10. There is thus a distribution shift between the target model's dataset and the attacker's dataset.

Figure 10 shows that our attack's performance is not influenced by an overlap between the training sets of the target model and shadow models. The attack success is unchanged when the attacker uses a disjoint dataset. A distribution shift between the training sets of the target model and shadow models does reduce the attack's TPR. Surprisingly, the attack's AUC is much higher when there is a distribution shift—we leave an explanation of this phenomenon to future work.

# E. Mismatched training procedures

We now explore how our attack is affected if the attacker does not know the exact training procedure of the target model. We train models with various architectures, optimizers, and data augmentations to investigate the attack's performance when the adversary guesses each of these incorrectly. For each attack, we train 64 shadow models and use our online

![](images/b8ce929ea266458f67029d21efd8f9e7c632fc9b83aac7330e2df9168ac25dca.jpg)

<details>
<summary>scatter</summary>

| Target model architecture | CNN-16 | CNN-32 | CNN-64 | WRN28-1 | WRN28-2 | WRN28-10 |
| ------------------------- | ------ | ------ | ------ | ------- | ------- | -------- |
| TPR @ 0.1% FPR            | 0.01   | 0.05   | 0.1    | 0.05    | 0.05    | 0.05     |
</details>

(a) Vary model architecture.

![](images/a9142e168d0b668970975580145f32413b8cccfcaf89e122447c3724ffe89e32.jpg)

<details>
<summary>scatter</summary>

Shadow model optimizer
| Target model optimizer | TPR @ 0.1% FPR |
| :--- | :--- |
| SGD | 0.05 |
| SGDM | 0.08 |
| Adam | 0.06 |
</details>

(b) Vary training optimizer.

![](images/c90ce3f72095842c6ead794f1669452c53d3d79dc4b0b36101c420e135a02dd4.jpg)

<details>
<summary>scatter</summary>

| Target model augmentation | TPR @ 0.1% FPR | Shadow model augmentation |
| ------------------------- | -------------- | ------------------------ |
| None                      | 0.1            | None                     |
| None                      | 0.01           | +Mirror                  |
| None                      | 0.001          | +Shift                   |
| None                      | 0.0001         | +Cutout                  |
| +Mirror                   | 0.1            | None                     |
| +Mirror                   | 0.01           | +Mirror                  |
| +Mirror                   | 0.001          | +Shift                   |
| +Mirror                   | 0.0001         | +Cutout                  |
| +Shift                    | 0.1            | None                     |
| +Shift                    | 0.01           | +Mirror                  |
| +Shift                    | 0.001          | +Shift                   |
| +Shift                    | 0.0001         | +Cutout                  |
| +Cutout                   | 0.1            | None                     |
| +Cutout                   | 0.01           | +Mirror                  |
| +Cutout                   | 0.001          | +Shift                   |
| +Cutout                   | 0.0001         | +Cutout                  |
</details>

(c) Vary data augmentation.   
Fig. 11: Our attack succeeds when the adversary is uncertain of the target model's training setup. We vary the target model's architecture (a), the training optimizer (b) and the data augmentation (c), as well as the adversary's guess of each of these properties when training shadow models. The attack performs best when the adversary guesses correctly (black-lined markers).

attack variant with a global estimate of the variance (see Section VI-B). Figure 11 summarizes our results at a fixed FPR of 0.1%. Appendix Figures 22 to 24 have full ROC curves.

In Figure 11a, we vary the target model's architecture. We study three CNN models (with 16, 32 and 64 convolutional filters), and three Wide ResNets (WRN) with width 1, 2 and 10. All models are trained with SGD with momentum and with random augmentations. Our attack performs best when the attacker trains shadow models of the same architecture as the target model, but using a similar model (e.g., a WRN28-1 instead of a WRN28-2) has a minimal effect on the attack. Moreover, we find that for both the CNN and WRN model families, larger models are more vulnerable to attacks.

In Figure 11b we fix the architecture to a WRN28-10, and vary the training optimizer: SGD, SGDM (SGD with momentum) or Adam. For both the defender or the attacker, the choice of optimizer has minimal impact on the attack.

Finally, in Figure 11c we fix the architecture (WRN28-10) and optimizer (SGDM) and vary the data augmentation used for training: none, mirroring, mirroring + shifts, mirroring + shifts + cutout. The attacker's guess of the data augmentation is used both to train shadow models, and to create additional queries for the attack. We find that correctly guessing the target model's data augmentation has the highest impact on attack performance. Models trained with stronger augmentations are harder to attack, as these models are less overfit.

# VII. ADDITIONAL INVESTIGATIONS

We now pivot from evaluating our attack to using our attack as a tool to better understand memorization in real models ( $\S$ VII-A) and why memorization occurs ( $\S$ VII-B).

# A. Attacking real-world models

All our experiments so far have involved attacking models that we ourselves have trained. To ensure that we did not somehow train weakly accidentally private (or non-private) models, we now show that our attacks also succeed on existing pretrained state-of-the-art models. To this end, we load standard models pre-trained by Phan [51] on the complete CIFAR-10 training set (50,000 examples). We train 256 shadow models by using the same training code and subsampling 50,000 points at random from the entire CIFAR-10 dataset (60,000 examples). On average, we have 213 IN models and 43 OUT models per example. Figure 12 shows our attack's true-positive rate at a $0.1\%$ FPR for various canonical model architectures.

![](images/4a8cfcb798cff7b2646a1bb71ee28e97279c8608c49150fce1609f720e9e21b9.jpg)

<details>
<summary>scatter</summary>

| Target model architecture | VGG16 | ResNet18 | ResNet34 | ResNet50 | DenseNet121 | Inception-v3 | ResNet50 | MobileNet-v2 |
| ------------------------- | ----- | -------- | -------- | -------- | ----------- | ------------ | -------- | ------------ |
| VGG16                     | 0.05  | 0.03     | 0.02     | 0.02     | 0.02        | 0.02         | 0.02     | 0.02         |
| ResNet18                  | 0.03  | 0.04     | 0.03     | 0.03     | 0.03        | 0.03         | 0.03     | 0.03         |
| ResNet34                  | 0.02  | 0.02     | 0.02     | 0.02     | 0.02        | 0.02         | 0.02     | 0.02         |
| ResNet50                  | 0.02  | 0.02     | 0.02     | 0.02     | 0.02        | 0.02         | 0.02     | 0.02         |
| DenseNet121               | 0.02  | 0.02     | 0.02     | 0.02     | 0.02        | 0.02         | 0.02     | 0.02         |
| Inception-v3              | 0.01  | 0.01     | 0.01     | 0.01     | 0.01        | 0.01         | 0.01     | 0.01         |
| MobileNet-v2              | 0.01  | 0.01     | 0.01     | 0.01     | 0.01        | 0.01         | 0.01     | 0.01         |
</details>

Fig. 12: Our attack succeeds against real state-of-the-art CIFAR-10 models [51]. The attacker trains shadow models on a random subset of 50,000 points from the entire CIFAR-10 dataset. The attack performs best when the shadow models have the same architecture as the target model, but training different models still leads to a strong attack.

We consider two attack variants: (1) the adversary knows the target model's architecture and uses it to train the shadow models; (2) the shadow models use a different architecture than the target model. Since we only have 43 models to estimate the distribution $(\tilde{\mathbb{Q}}_{\mathrm{out}})$ , esitimating a global variance for all examples performs best. The results of this experiment are qualitatively similar to those in Section VI-E: (1) the model architecture has a small effect on the privacy leakage (e.g., the attack works better against a ResNet-18 than against a MobileNet-v2); (2) the attack works best when the shadow models share the same architecture as the target model, but it is robust to architecture mismatches. For example, attacking a ResNet-34 model with either ResNet-18 or ResNet-50 shadow models leads to a minor drop in attack success rate (from $5\%$ TPR to $4\%$ TPR).

![](images/03a21a9526c9489e845174b74c32ac0e0501d04d3a0ee0f812c506a5e5862781.jpg)

<details>
<summary>bar_stacked</summary>

| Privacy score | CIFAR-10, Correct Labels | CINIC-10, Correct Labels | CIFAR-100, Random Labels | CIFAR-10, Random Labels |
| ------------- | ------------------------ | ------------------------ | ------------------------ | ----------------------- |
| 0             | 450                      | 180                      | 0                        | 30                      |
| 1             | 100                      | 80                       | 50                       | 20                      |
| 2             | 0                        | 0                        | 120                      | 40                      |
| 3             | 0                        | 0                        | 130                      | 150                     |
| 4             | 0                        | 0                        | 0                        | 70                      |
| 5             | 0                        | 0                        | 0                        | 10                      |
</details>

Fig. 13: Out-of-distribution training examples are less private.

# B. Why are some examples less private?

While our average attack success rate is modest, the success rate at low false-positive rates can be very high. This suggests that there is a subset of examples that are easier to attack than others. While a full investigation of this is beyond the scope of our paper, we find an important factor behind why some samples are less private is that they are out-of-distribution.

To make this argument, we intentionally inject out-of-distribution examples into a model's training dataset and compare the difficulty of attacking these newly inserted samples versus typical examples. Specifically, we insert 1,000 examples from various out-of-distribution sources into the 50,000-example CIFAR-10 training dataset to form a new augmented 51,000 example dataset. We then train shadow models on this dataset, run our attack, and measure the distinguishability of distributions of losses for IN and OUT models for each of the 1,000 newly inserted examples (we use a simple measure of distance between distributions here, defined as $d = \frac{|\mu_{\text{in}} - \mu_{\text{out}}|}{\sigma_{\text{in}} + \sigma_{\text{out}}}$ ). Figure 13 plots the distribution of these “privacy scores” assigned to each example. As a baseline, in blue, we show the distribution of privacy scores for the standard CIFAR-10 dataset; these are tightly concentrated around 0.

Next we show the privacy scores of examples inserted from the CINIC-10 dataset, which are drawn from ImageNet. Due to this slight distribution shift, the CINIC-10 images have a larger privacy score on average: it is easier to detect their presence in the dataset because they are slightly out-of-distribution.

We can extend this further by inserting intentionally mis-labeled images that are extremely out-of-distribution. If we choose 1,000 images (shown in red) from the CIFAR-10 test set and assign new random labels to each image, then we get a much higher privacy score for these images. Finally, we interpolate between the extreme OOD setting of random (and thus incorrectly) labeled CIFAR-10 images and correctly-labeled CINIC-10 by inserting randomly labeled images from CIFAR-100 (shown in green). Because these images come from a disjoint class distribution, models will not typically be confident on their label one way or another unless they are seen during training. The privacy scores here fall in between correctly labeled CINIC-10 and incorrectly labeled CIFAR-10.

# VIII. CONCLUSION

As we have argued throughout this paper, membership inference attacks should focus on the problem of achieving high true-positive rates at low false-positive rates. Our attack presents one way to succeed at this goal. There are a number of different evaluation directions that we hope future work will explore under this direction.

Membership inference attacks as a privacy metric. Both researchers $[42]$ and practitioners $[63]$ use membership inference attacks to measure privacy of trained models. We argue that these metrics should use strong attacks (such as ours) in order to accurately measure privacy leakage. Future work using membership inference attacks should consider the low false-positive rate regime, to better understand if the privacy of even just a few users can be confidently breached.

Usability improvements to membership inference attacks. The key limitation of per-example membership inference attacks is that they require new hyperparameters that need to be learned from the data. While it is much more important that attacks are strong (even if slow) as opposed to fast (but weak), we hope that future work will improve the computational efficiency of our attack approach, in order to allow it to be deployed in more settings.

Improving other privacy attacks with our method. Membership inference attacks form the basis for many other privacy attack methods $[3, 4, 17]$ . Our membership inference method, in principle, should be able to directly improve these attacks.

Rethinking our current understanding of MIA results. The literature on membership inference attacks has answered a number of memorization questions. However, many (or even most) of these prior papers focused on the inadequate metric of average-case attack success rates, instead of on the low false-positive rate regime. As a result it will be necessary to re-investigate prior results from this perspective:

- Do previously-“broken” [26, 45] defenses prevent our attack? Prior defenses were only ever shown to be ineffective at preventing an adversary from succeeding on average—not confidently at low false-positive rates.   
- How does differential privacy interact with our improved attacks? We have preliminary evidence that vacuous guarantees might prevent our low-FPR attacks (Section A-A).   
- Are attacks with reduced capabilities possible? For example, label-only attacks [6, 34, 54] can match the balanced accuracy of shadow-model approaches. But do these attacks work at low false-positive rates?   
- Are attacks with extra capabilities more effective? Prior work has shown that access to gradient queries [46] or intermediate models [58] improves attack AUC. However, does this observation hold at low false-positive rates?

We hope that future work will be able to answer these questions, among many more, in order to better evaluate (and develop) techniques that preserve the privacy of training data. By developing attacks that succeed low false-positive rates, we can evaluate privacy not as a measurement of the average user, but of the most vulnerable.

# ACKNOWLEDGEMENTS

We are grateful to Thomas Steinke, Dave Evans, Reza Shokri, Sanghyun Hong, Alex Sablayrolles, Liwei Song, Matthias Lécuyer and the anonymous reviewers for comments on drafts of this paper.

# REFERENCES

[1] Martin Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. Deep learning with differential privacy. In Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security, page 308–318. ACM, 2016.   
[2] Gavin Brown, Mark Bun, Vitaly Feldman, Adam Smith, and Kunal Talwar. When is memorization of irrelevant training data necessary for high-accuracy learning? In Proceedings of the 53rd Annual ACM SIGACT Symposium on Theory of Computing, pages 123–132, 2021.   
[3] Nicholas Carlini, Chang Liu, Úlfar Erlingsson, Jernej Kos, and Dawn Song. The secret sharer: Evaluating and testing unintended memorization in neural networks. In 28th USENIX Security Symposium (USENIX Security 19), pages 267–284, 2019.   
[4] Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, et al. Extracting training data from large language models. In 30th USENIX Security Symposium (USENIX Security 21), 2021.   
[5] Mia Xu Chen, Benjamin N Lee, Gagan Bansal, Yuan Cao, Shuyuan Zhang, Justin Lu, Jackie Tsay, Yinan Wang, Andrew M Dai, Zhifeng Chen, et al. Gmail smart compose: Real-time assisted writing. In ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 2287–2295, 2019.   
[6] Christopher A Choquette-Choo, Florian Tramer, Nicholas Carlini, and Nicolas Papernot. Label-only membership inference attacks. In International Conference on Machine Learning, pages 1964–1974. PMLR, 2021.   
[7] Ekin D. Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, and Quoc V. Le. Autoaugment: Learning augmentation policies from data, 2018.   
[8] Luke N Darlow, Elliot J Crowley, Antreas Antoniou, and Amos J Storkey. CINIC-10 is not Imagenet or CIFAR-10. arXiv preprint arXiv:1810.03505, 2018.   
[9] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.   
[10] Cynthia Dwork and Aaron Roth. The algorithmic foundations of differential privacy. Found. Trends Theor. Comput. Sci., 9(3-4):211–407, 2014.   
[11] Cynthia Dwork, Adam Smith, Thomas Steinke, Jonathan Ullman, and Salil Vadhan. Robust traceability from trace amounts. In 2015 IEEE 56th Annual Symposium on Foundations of Computer Science, pages 650–669. IEEE, 2015.   
[12] Cynthia Dwork, Adam Smith, Thomas Steinke, and Jonathan Ullman. Exposed! a survey of attacks on private data. Annual Review of Statistics and Its Application, 4:61–84, 2017.   
[13] Andre Esteva, Brett Kuprel, Roberto A Novoa, Justin Ko, Susan M Swetter, Helen M Blau, and Sebastian Thrun. Dermatologist-level classification of skin cancer with deep neural networks. Nature, 542(7639):115–118, 2017.   
[14] Vitaly Feldman. Does learning require memorization? a short tale about a long tail. In Proceedings of the 52nd Annual ACM SIGACT Symposium on Theory of Computing, pages 954–959, 2020.

[15] Vitaly Feldman and Chiyuan Zhang. What neural networks memorize and why: Discovering the long tail via influence estimation. arXiv preprint arXiv:2008.03703, 2020.   
[16] Matt Fredrikson, Somesh Jha, and Thomas Ristenpart. Model inversion attacks that exploit confidence information and basic countermeasures. In Proceedings of the 22nd ACM SIGSAC Conference on Computer and Communications Security, pages 1322-1333, 2015.   
[17] Karan Ganju, Qi Wang, Wei Yang, Carl A Gunter, and Nikita Borisov. Property inference attacks on fully connected neural networks using permutation invariant representations. In Proceedings of the 2018 ACM SIGSAC conference on computer and communications security, pages 619–633, 2018.   
[18] Jamie Hayes, Luca Melis, George Danezis, and Emiliano De Cristofaro. LOGAN: Membership inference attacks against generative models. In Proceedings on Privacy Enhancing Technologies (PoPETs), pages 133–152. De Gruyter, 2019.   
[19] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition, 2015.   
[20] Xinlei He, Jinyuan Jia, Michael Backes, Neil Zhenqiang Gong, and Yang Zhang. Stealing links from graph neural networks. In 30th USENIX Security Symposium (USENIX Security 21), 2021.   
[21] Grant Ho, Aashish Sharma, Mobin Javed, Vern Paxson, and David Wagner. Detecting credential spearphishing in enterprise settings. In 26th USENIX Security Symposium (USENIX Security 17), pages 469–485, 2017.   
[22] Nils Homer, Szabolcs Szelinger, Margot Redman, David Duggan, Waibhav Tembe, Jill Muehling, John V Pearson, Dietrich A Stephan, Stanley F Nelson, and David W Craig. Resolving individuals contributing trace amounts of dna to highly complex mixtures using high-density snp genotyping microarrays. PLoS genetics, 4(8), 2008.   
[23] Robert A Jacobs. Increased rates of convergence through learning rate adaptation. Neural networks, 1(4):295-307, 1988.   
[24] Matthew Jagielski, Jonathan Ullman, and Alina Oprea. Auditing differentially private machine learning: How private is private SGD? arXiv preprint arXiv:2006.07709, 2020.   
[25] Bargav Jayaraman, Lingxiao Wang, David Evans, and Quanquan Gu. Revisiting membership inference under realistic assumptions. In Proceedings on Privacy Enhancing Technologies (PoPETs), 2021.   
[26] Jinyuan Jia, Ahmed Salem, Michael Backes, Yang Zhang, and Neil Zhenqiang Gong. Memguard: Defending against black-box membership inference attacks via adversarial examples. In Proceedings of the 2019 ACM SIGSAC Conference on Computer and Communications Security, pages 259–274, 2019.   
[27] Alex Kantchelian, Michael Carl Tschantz, Sadia Afroz, Brad Miller, Vaishaal Shankar, Rekha Bachwani, Anthony D Joseph, and J Doug Tygar. Better malware ground truth: Techniques for weighting anti-virus vendor labels. In Proceedings of the 8th ACM Workshop on Artificial Intelligence and Security, pages 45–56, 2015.   
[28] Zico Kolter and Marcus A Maloof. Learning to detect and classify malicious executables in the wild. Journal of Machine Learning Research, 7(12), 2006.   
[29] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images, 2009.   
[30] Anders Krogh and John A Hertz. A simple weight decay can improve generalization. In Advances in neural information processing systems, pages 950-957, 1992.   
[31] Aleksandar Lazarevic, Levent Ertoz, Vipin Kumar, Aysel Ozgur, and Jaideep Srivastava. A comparative study of anomaly detection schemes in network intrusion detection. In Proceedings of the 2003 SIAM international conference on data mining, pages 25–36. SIAM, 2003.   
[32] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Pro-

ceedings of the IEEE, 86(11):2278-2324, 1998.   
[33] Klas Leino and Matt Fredrikson. Stolen memories: Leveraging model memorization for calibrated white-box membership inference. arXiv preprint arXiv:1906.11798, 2019.   
[34] Zheng Li and Yang Zhang. Membership leakage in label-only exposures. arXiv preprint arXiv:2007.15528, 2020.   
[35] Yugeng Liu, Rui Wen, Xinlei He, Ahmed Salem, Zhikun Zhang, Michael Backes, Emiliano De Cristofaro, Mario Fritz, and Yang Zhang. ML-Doctor: Holistic risk assessment of inference attacks against machine learning models. arXiv preprint arXiv:2102.02551, 2021.   
[36] Yunhui Long, Vincent Bindschaedler, and Carl A Gunter. Towards measuring membership privacy. arXiv preprint arXiv:1712.09136, 2017.   
[37] Yunhui Long, Lei Wang, Diyue Bu, Vincent Bindschaedler, Xiaofeng Wang, Haixu Tang, Carl A Gunter, and Kai Chen. A pragmatic approach to membership inferences on machine learning models. In 2020 IEEE European Symposium on Security and Privacy (EuroS&P), pages 521–534. IEEE, 2020.   
[38] Ilya Loshchilov and Frank Hutter. SGDR: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983, 2016.   
[39] Luca Melis, Congzheng Song, Emiliano De Cristofaro, and Vitaly Shmatikov. Exploiting unintended feature leakage in collaborative learning. In 2019 IEEE Symposium on Security and Privacy (SP), pages 691–706. IEEE, 2019.   
[40] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. arXiv preprint arXiv:1609.07843, 2016.   
[41] Vangelis Metsis, Ion Androutsopoulos, and Georgios Paliouras. Spam filtering with naive Bayes—which naive Bayes? In CEAS, volume 17, pages 28–69, 2006.   
[42] Sasi Kumar Murakonda and Reza Shokri. ML Privacy Meter: Aiding regulatory compliance by quantifying the privacy risks of machine learning. arXiv preprint arXiv:2007.09339, 2020.   
[43] Sasi Kumar Murakonda, Reza Shokri, and George Theodorakopoulos. Ultimate power of inference attacks: Privacy risks of learning high-dimensional graphical models. arXiv e-prints, pages arXiv–1905, 2019.   
[44] Sasi Kumar Murakonda, Reza Shokri, and George Theodorakopoulos. Quantifying the privacy risks of learning high-dimensional graphical models. In International Conference on Artificial Intelligence and Statistics, pages 2287-2295. PMLR, 2021.   
[45] Milad Nasr, Reza Shokri, and Amir Houmansadr. Machine learning with membership privacy using adversarial regularization. In Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security, pages 634–646, 2018.   
[46] Milad Nasr, Reza Shokri, and Amir Houmansadr. Comprehensive privacy analysis of deep learning: Passive and active white-box inference attacks against centralized and federated learning. In 2019 IEEE symposium on security and privacy (SP), pages 739–753. IEEE, 2019.   
[47] Milad Nasr, Shuang Song, Abhradeep Thakurta, Nicolas Papernot, and Nicholas Carlini. Adversary instantiation: Lower bounds for differentially private machine learning. arXiv preprint arXiv:2101.04535, 2021.   
[48] Jerzy Neyman and Egon Sharpe Pearson. On the problem of the most efficient tests of statistical hypotheses. Philosophical Transactions of the Royal Society of London., 231(694-706):289–337, 1933.   
[49] Patrick Pantel and Dekang Lin. SpamCop: A spam classification & organization program. In Proceedings of AAAI-98 Workshop on Learning for Text Categorization, pages 95–98, 1998.   
[50] Nicolas Papernot, Shuang Song, Ilya Mironov, Ananth Raghunathan, Kunal Talwar, and Úlfar Erlingsson. Scalable private

learning with PATE. arXiv preprint arXiv:1802.08908, 2018.   
[51] Huy Phan. huyvnphan/pytorch\_cifar10, January 2021. URL https://doi.org/10.5281/zenodo.4431043.   
[52] Apostolos Pyrgelis, Carmela Troncoso, and Emiliano De Cristofaro. Knock knock, who's there? membership inference on aggregate location data. arXiv preprint arXiv:1708.06145, 2017.   
[53] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 2019.   
[54] Shadi Rahimian, Tribhuvanesh Orekondy, and Mario Fritz. Sampling attacks: Amplification of membership inference attacks by repeated queries. arXiv preprint arXiv:2009.00395, 2020.   
[55] Md Atiqur Rahman, Tanzila Rahman, Robert Laganière, Noman Mohammed, and Yang Wang. Membership inference attack against differentially private deep learning model. Trans. Data Priv., 11(1):61–79, 2018.   
[56] Alexandre Sablayrolles, Matthijs Douze, Cordelia Schmid, Yann Ollivier, and Hervé Jégou. White-box vs black-box: Bayes optimal strategies for membership inference. In International Conference on Machine Learning, pages 5558–5567. PMLR, 2019.   
[57] Ahmed Salem, Yang Zhang, Mathias Humbert, Pascal Berrang, Mario Fritz, and Michael Backes. ML-Leaks: Model and data independent membership inference attacks and defenses on machine learning models, 2018.   
[58] Ahmed Salem, Apratim Bhattacharya, Michael Backes, Mario Fritz, and Yang Zhang. Updates-leak: Data set inference and reconstruction attacks in online learning. In 29th USENIX Security Symposium (USENIX Security 20), pages 1291–1308, 2020.   
[59] Sriram Sankararaman, Guillaume Obozinski, Michael I Jordan, and Eran Halperin. Genomic privacy and limits of individual detection in a pool. Nature genetics, 41(9):965–967, 2009.   
[60] Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership inference attacks against machine learning models. arXiv preprint arXiv:1610.05820, 2016.   
[61] Liwei Song and Prateek Mittal. Systematic evaluation of privacy risks of machine learning models. In 30th USENIX Security Symposium (USENIX Security 21), 2021.   
[62] Liwei Song, Reza Shokri, and Prateek Mittal. Privacy risks of securing machine learning models against adversarial examples. In Proceedings of the 2019 ACM SIGSAC Conference on Computer and Communications Security, pages 241–257, 2019.   
[63] Shuang Song and David Marn. Introducing a new privacy testing library in tensorflow. https://blog.tensorflow.org/2020/06/introducing-new-privacy-testing-library.html, 2020.   
[64] Shuang Song, Kamalika Chaudhuri, and Anand D Sarwate. Stochastic gradient descent with differentially private updates. In 2013 IEEE Global Conference on Signal and Information Processing, pages 245–248. IEEE, 2013.   
[65] Thomas Steinke and Jonathan Ullman. The pitfalls of average-case differential privacy. DifferentialPrivacy.org, 07 2020. https://differentialprivacy.org/average-case-dp/.   
[66] Stacey Truex, Ling Liu, Mehmet Emre Gursoy, Lei Yu, and Wenqi Wei. Towards demystifying membership inference attacks. arXiv preprint arXiv:1807.09173, 2018.   
[67] David A Van Dyk and Xiao-Li Meng. The art of data augmentation. Journal of Computational and Graphical Statistics, 10(1):1–50, 2001.   
[68] Lauren Watson, Chuan Guo, Graham Cormode, and Alex Sablayrolles. On the importance of difficulty calibration in membership inference attacks. arXiv preprint arXiv:2111.08440, 2021.   
[69] Jiayuan Ye, Aadyaa Maddi, Sasi Kumar Murakonda, and Reza Shokri. Enhanced membership inference attacks against ma-

chine learning models. arXiv preprint arXiv:2111.09679, 2021.   
[70] Samuel Yeom, Irene Giacomelli, Matt Fredrikson, and Somesh Jha. Privacy risk in machine learning: Analyzing the connection to overfitting. In 2018 IEEE 31st Computer Security Foundations Symposium (CSF), pages 268–282. IEEE, 2018.   
[71] Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. arXiv preprint arXiv:1605.07146, 2016.   
[72] Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning (still) requires rethinking generalization. Communications of the ACM, 64(3):107–115, 2021.   
[73] Zhun Zhong, Liang Zheng, Guoliang Kang, Shaozi Li, and Yi Yang. Random erasing data augmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 13001–13008, 2020.

# APPENDIX A ADDITIONAL EXPERIMENTS

# A. Attacking DP-SGD

Machine learning with differential privacy $[1]$ is the main defence mechanism against privacy attacks including membership inference against machine learning models. Differential privacy provides an upper bound on the success of any membership inference attack. Recent works $[24, 47]$ thus used membership attacks to empirically audit differential privacy bounds, in particular those obtained from DP-SGD $[1]$ . In this work, we are interested in the effect of DP-SGD on the performance of our membership inference attack.

We consider different combinations of DP-SGD's noise multiplier and clipping norm parameters in our evaluation. Table IV summarizes the average accuracy of standard CNN models trained on CIFAR-10 with DP-SGD for different parameter sets. We evaluate the effectiveness of our membership inference attacks for these settings in Figure 14. Even just clipping the gradient norm without adding any noise reduces the performance of our attack significantly. However, small clipping norms can reduce the accuracy of the models as shown in Table IV.

TABLE IV: Accuracy of the models trained with DP-SGD on CIFAR10 with different noise parameters 

<table><tr><td>Noise Multiplier (σ)</td><td>C=10</td><td>C=5</td><td>C=1</td></tr><tr><td>0.0</td><td>84.0%</td><td>78.5%</td><td>61.3%</td></tr><tr><td>0.2</td><td>73.9%</td><td>77.1%</td><td>62.8%</td></tr><tr><td>0.8</td><td>36.9%</td><td>43.3%</td><td>61.3%</td></tr></table>

For higher clipping norms, adding very small amounts of noise (Figure 14-b) reduces the effectiveness of the membership inference attack to chance, while resulting in models with higher accuracy.

Training models with very small amounts of noise is an effective defense against our membership inference attack, despite resulting in very large provable DP bounds $\epsilon$ .

![](images/88103a4dd0a18d7a8e3755cd0d16e032ab84ab2af7de3abfe734d157f9ce2b66.jpg)

<details>
<summary>line</summary>

| False Positive Rate | True Positive Rate (σ=0.0,C=10) | True Positive Rate (σ=0.0,C=5) | True Positive Rate (σ=0.0,C=1) | True Positive Rate (DP upper for eps=1) |
| ------------------- | -------------------------------- | ------------------------------- | ------------------------------ | --------------------------------------- |
| 10⁻⁵                | ~10⁻³                            | ~10⁻³                           | ~10⁻⁴                          | ~10⁻³                                   |
| 10⁻⁴                | ~10⁻²                            | ~10⁻²                           | ~10⁻³                          | ~10⁻²                                   |
| 10⁻³                | ~10⁻¹                            | ~10⁻¹                           | ~10⁻²                          | ~10⁻¹                                   |
| 10⁻²                | ~10⁰                             | ~10⁰                            | ~10⁻¹                          | ~10⁰                                    |
| 10⁻¹                | ~10⁰                             | ~10⁰                            | ~10⁰                           | ~10⁰                                    |
| 10⁰                 | ~10⁰                             | ~10⁰                            | ~10⁰                           | ~10⁰                                    |
</details>

(a) $\epsilon = \infty$   
![](images/fafc614c2e780390e741a5fdb7001056b885aa7ba15c49f856edf9688d92503f.jpg)

<details>
<summary>line</summary>

| False Positive Rate | True Positive Rate (σ=0.2, C=10) | True Positive Rate (σ=0.2, C=5) | True Positive Rate (σ=0.2, C=1) | True Positive Rate (DP upper for eps=1) |
| ------------------- | ---------------------------------- | -------------------------------- | ------------------------------- | --------------------------------------- |
| 10⁻⁵                | ~10⁻⁴                              | ~10⁻⁴                            | ~10⁻⁴                           | ~10⁻⁴                                   |
| 10⁻⁴                | ~10⁻³                              | ~10⁻³                            | ~10⁻³                           | ~10⁻³                                   |
| 10⁻³                | ~10⁻²                              | ~10⁻²                            | ~10⁻²                           | ~10⁻²                                   |
| 10⁻²                | ~10⁻¹                              | ~10⁻¹                            | ~10⁻¹                           | ~10⁻¹                                   |
| 10⁻¹                | ~10⁰                               | ~10⁰                             | ~10⁰                            | ~10⁰                                    |
| 10⁰                 | ~10⁰                               | ~10⁰                             | ~10⁰                            | ~10⁰                                    |
</details>

(b) $\epsilon > 5000$   
![](images/896991625f66b1f28e31b85c86d7764e7881d0886c440558d20efab7b18f1077.jpg)

<details>
<summary>line</summary>

| False Positive Rate | True Positive Rate (σ=0.8,C=10) | True Positive Rate (σ=0.8,C=5) | True Positive Rate (σ=0.8,C=1) | True Positive Rate (DP upper for eps=1) |
| ------------------- | -------------------------------- | ------------------------------- | ------------------------------ | --------------------------------------- |
| 1e-5                | ~1e-5                            | ~1e-5                           | ~1e-5                          | ~1e-4                                   |
| 1e-4                | ~1e-4                            | ~1e-4                           | ~1e-4                          | ~1e-3                                   |
| 1e-3                | ~1e-3                            | ~1e-3                           | ~1e-3                          | ~1e-2                                   |
| 1e-2                | ~1e-2                            | ~1e-2                           | ~1e-2                          | ~1e-1                                   |
| 1e-1                | ~1e-1                            | ~1e-1                           | ~1e-1                          | ~1e0                                    |
| 1e0                 | 1e0                              | 1e0                             | 1e0                            | 1e0                                     |
</details>

(c) $\epsilon = 8$   
Fig. 14: Effectiveness of using DP-SGD against our attack with different privacy budgets.

# B. White-box Attacks

Previous works [46, 62] suggested that is possible to achieve better membership inference if the adversary has white-box access to the target model. In particular, previous works showed that using the norm of the model's gradient at a target point could increase the balanced accuracy of membership inference attacks. Figure 15 highlights the comparison between a white-box and a black-box adversary. The results show that using gradient norms will improve the overall AUC both for our online attack, as well as when using a global threshold as in the LOSS attack. However, at lower false-positive rates we do not observe any improvement of using gradient norms compared to just using model confidences.

![](images/42a083ffaf4e9adf339355ba05ce86aa83d9e8b5a558e99daa0dfafb6b40cfae.jpg)

<details>
<summary>line</summary>

| Method | AUC |
| --- | --- |
| Ours based on logits | 0.711 |
| Ours based on gradient-norm | 0.724 |
| Global threshold based on logits | 0.576 |
| Global threshold based on gradient norm | 0.601 |
</details>

Fig. 15: Comparison of the white-box attack using our approach to the black-box setting.

# APPENDIX B ADDITIONAL FIGURES AND TABLES

# A. Attack Performance versus Model Accuracy

In Section V-D, Figure 7 we plotted the relationship between a model's train-test gap and its vulnerability to membership inference attacks. In Figure 16, we look at the attack success rate as a function of the test accuracy of the same models. There is a clear trend where better models are more vulnerable to attacks. Prior work reported a similar phenomenon for data extraction attacks [3, 4].

![](images/ec806933293fb63d82ab53fa020740bd32486fd21c161bbc12bdb4eccb53f021.jpg)

<details>
<summary>scatter</summary>

| Model       | Test Accuracy | TPR @ 0.1% FPR |
|-------------|---------------|----------------|
| CNN1        | 0.4           | 0.001          |
| CNN2        | 0.5           | 0.001          |
| CNN4        | 0.6           | 0.001          |
| CNN8        | 0.7           | 0.01           |
| CNN16       | 0.8           | 0.1            |
| CNN32       | 0.9           | 0.1            |
| CNN64       | 0.8           | 0.1            |
| WRN28-1     | 0.8           | 0.01           |
| WRN28-2     | 0.8           | 0.01           |
| WRN28-10    | 0.9           | 0.1            |
</details>

Fig. 16: Attack true-positive rate versus model test accuracy.

# B. Full ROC Curves for Gaussian Distribution Fitting

In Figure 17, we show full (log-scale) ROC curves for the experiment in Section VI-B, where we explored the effect of varying the number of shadow models on the success rate of our online attack. We vary the number of shadow models from 4 to 256 and consider two attack variants: (1) fit Gaussians for each example by estimating the means $\mu_{in}$ , $\mu_{out}$ and variances $\sigma_{in}^{2}$ , $\sigma_{out}^{2}$ independently for each example; (2) estimate the means $\mu_{in}$ , $\mu_{out}$ for each example, but estimate global variances $\sigma_{in}^{2}$ , $\sigma_{out}^{2}$ . As we observed in Section VI-B, estimating per-example variances works poorly when the number of shadow models is small (< 64). With a global estimate of the variance, the attack performs nearly on par with our best attack with as little as 16 shadow models.

![](images/b5010f4693fd30b2f4cdbbfc59eea50ed3c7ad6c18d1b0842dff493a89b299e7.jpg)

<details>
<summary>line</summary>

| False Positive Rate | 256 models | 64 models | 16 models | 4 models | with global variance |
| ------------------- | ---------- | --------- | --------- | -------- | -------------------- |
| 10^-5               | ~10^-2     | ~10^-2    | ~10^-3    | ~10^-5   | ~10^-5               |
| 10^-4               | ~10^-1     | ~10^-1    | ~10^-2    | ~10^-4   | ~10^-4               |
| 10^-3               | ~10^0      | ~10^0     | ~10^-1    | ~10^-3   | ~10^-3               |
| 10^-2               | ~10^0      | ~10^0     | ~10^0     | ~10^-2   | ~10^-2               |
| 10^-1               | ~10^0      | ~10^0     | ~10^0     | ~10^-1   | ~10^-1               |
| 10^0                | ~10^0      | ~10^0     | ~10^0     | ~10^0    | ~10^0                |
</details>

Fig. 17: Effect of varying the number of models trained on attack success rates. It is always useful to estimate the mean per-example difficulty; however when only a few models are available, it is orders of magnitude more effective to assign all examples the same variance.

# C. Comparison to Prior Work on Additional Datasets

Similarly to Figure 1 for CIFAR-10, we compare our attack against prior membership inference attacks on additional datasets: CIFAR-100 in Figure 18, WikiText-103 in Figure 19, Texas in Figure 20 and Purchase in Figure 21.

![](images/513f5d9af690338c3d835759c8bd8be044376b71af8bece6eabd986721e72885.jpg)

<details>
<summary>line</summary>

| Method              | True Positive Rate at 10⁻⁵ | True Positive Rate at 10⁻⁴ | True Positive Rate at 10⁻³ | True Positive Rate at 10⁻² | True Positive Rate at 10⁻¹ | True Positive Rate at 10⁰ |
|---------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|
| Ours                | ~0.1                        | ~0.05                       | ~0.02                       | ~0.01                       | ~0.005                      | ~0.001                      |
| Sablayrolles et al.| ~0.01                       | ~0.005                      | ~0.002                      | ~0.001                      | ~0.0005                     | ~0.0001                     |
| Long et al.         | ~0.001                      | ~0.0005                     | ~0.0002                     | ~0.0001                     | ~0.00005                    | ~0.00001                    |
| Watson et al.       | ~0.001                      | ~0.0005                     | ~0.0002                     | ~0.0001                     | ~0.00005                    | ~0.00001                    |
| Shokri et al.       | ~0.001                      | ~0.0005                     | ~0.0002                     | ~0.0001                     | ~0.00005                    | ~0.00001                    |
| Song et al.         | ~0.001                      | ~0.0005                     | ~0.0002                     | ~0.0001                     | ~0.00005                    | ~0.00001                    |
| Yeom et al.         | ~0.001                      | ~0.0005                     | ~0.0002                     | ~0.0001                     | ~0.00005                    | ~0.00001                    |
| Jayaraman et al.    | ~0.001                      | ~0.0005                     | ~0.0002                     | ~0.0001                     | ~0.00005                    | ~0.00001                    |
</details>

Fig. 18: ROC curve of prior membership inference attacks, compared to our attack, on CIFAR-100.

![](images/15d08a20f8ab7d339db35de9bf501a86eda8596706375b8961ce780665ad16a5.jpg)

<details>
<summary>line</summary>

| False Positive Rate | Ours     | Sablayrolles et al. | Watson et al. | Yeom et al. |
| ------------------- | -------- | ------------------- | ------------- | ----------- |
| 1e-5                | 0.001    | 0.0001              | 0.0001        | 0.00001     |
| 1e-4                | 0.002    | 0.0002              | 0.0002        | 0.00002     |
| 1e-3                | 0.005    | 0.0005              | 0.0005        | 0.00005     |
| 1e-2                | 0.01     | 0.001               | 0.001         | 0.0001      |
| 1e-1                | 0.02     | 0.002               | 0.002         | 0.0002      |
| 1e+0                | 0.1      | 0.01                | 0.1           | 0.01        |
</details>

Fig. 19: ROC curve of prior membership inference attacks, compared to our attack, on WikiText-103. We omit prior attacks that rely on the model features $z(x)$ , as these attacks were not designed for sequential models.

![](images/466483f92cbb72db7e43ecba885a1bbd0b7dc35f1ce2d85e5464f4c48dea9696.jpg)

<details>
<summary>line</summary>

| Method              | False Positive Rate | True Positive Rate |
| ------------------- | ------------------- | ------------------ |
| Ours                | 10^-5               | 10^-1              |
| Ours                | 10^-4               | 10^-1              |
| Ours                | 10^-3               | 10^-1              |
| Ours                | 10^-2               | 10^-1              |
| Ours                | 10^-1               | 10^-1              |
| Ours                | 10^0                | 10^0               |
| Sablayrolles et al.| 10^-5               | 10^-1              |
| Sablayrolles et al.| 10^-4               | 10^-1              |
| Sablayrolles et al.| 10^-3               | 10^-1              |
| Sablayrolles et al.| 10^-2               | 10^-1              |
| Sablayrolles et al.| 10^-1               | 10^-1              |
| Sablayrolles et al.| 10^0                | 10^0               |
| Long et al.         | 10^-5               | 10^-5              |
| Long et al.         | 10^-4               | 10^-4              |
| Long et al.         | 10^-3               | 10^-3              |
| Long et al.         | 10^-2               | 10^-3              |
| Long et al.         | 10^-1               | 10^-3              |
| Long et al.         | 10^0                | 10^0               |
| Watson et al.       | 10^-5               | 10^-5              |
| Watson et al.       | 10^-4               | 10^-4              |
| Watson et al.       | 10^-3               | 10^-3              |
| Watson et al.       | 10^-2               | 10^-3              |
| Watson et al.       | 10^-1               | 10^-3              |
| Watson et al.       | 10^0                | 10^0               |
| Shokri et al.       | 10^-5               | 10^-5              |
| Shokri et al.       | 10^-4               | 10^-4              |
| Shokri et al.       | 10^-3               | 10^-3              |
| Shokri et al.       | 10^-2               | 10^-3              |
| Shokri et al.       | 10^-1               | 10^-3              |
| Shokri et al.       | 10^0                | 10^0               |
| Song et al.         | 10^-5               | 10^-5              |
| Song et al.         | 10^-4               | 10^-4              |
| Song et al.         | 10^-3               | 10^-3              |
| Song et al.         | 10^-2               | 10^-3              |
| Song et al.         | 10^-1               | 10^-3              |
| Song et al.         | 10^0                | 10^0               |
| Yeom et al.         | 10^-5               | 10^-5              |
| Yeom et al.         | 10^-4               | 10^-4              |
| Yeom et al.         | 10^-3               | 10^-3              |
| Yeom et al.         | 10^-2               | 10^-3              |
| Yeom et al.         | 10^-1               | 10^-3              |
| Yeom et al.         | 10^0                | 10^0               |
| Jayaraman et al.    | 10^-5               | 10^-5              |
| Jayaraman et al.    | 10^-4               | 10^-4              |
| Jayaraman et al.    | 10^-3               | 10^-3              |
| Jayaraman et al.    | 10^-2               | 10^-3              |
| Jayaraman et al.    | 10^-1               | 10^-3              |
| Jayaraman et al.    | 10^0                | 10^0               |
</details>

Fig. 20: ROC curve of prior membership inference attacks, compared to our attack, on the Texas dataset.

![](images/e7046bc4f3c0091b70a96a2bcf4f3e137928a69a9ce25d6e0579f7a52ebcc83b.jpg)

<details>
<summary>line</summary>

| Method              | False Positive Rate | True Positive Rate |
| ------------------- | ------------------- | ------------------ |
| Ours                | 1e-5                | 1e-3               |
| Sablayrolles et al.| 1e-4                | 1e-2               |
| Long et al.         | 1e-3                | 1e-1               |
| Watson et al.       | 1e-2                | 1e-1               |
| Shokri et al.       | 1e-1                | 1e-1               |
| Song et al.         | 1e-1                | 1e-1               |
| Yeom et al.         | 1e-1                | 1e-1               |
| Jayaraman et al.    | 1e-1                | 1e-1               |
</details>

Fig. 21: ROC curve of prior membership inference attacks, compared to our attack, on the Purchase dataset.

<table><tr><td>Attack Approach</td><td>TPR @ 0.1% FPR</td></tr><tr><td>LOSS attack [70]</td><td>0.0%</td></tr><tr><td>+ Logit scaling</td><td>0.1%</td></tr><tr><td>+ Multiple queries</td><td>0.1%</td></tr><tr><td>LOSS attack [70]</td><td>0.0%</td></tr><tr><td>+ Per-example thresholds ( $\tilde{\mathbb{Q}}_{\text{out}}$  only) [68]</td><td>5.2%</td></tr><tr><td>+ Logit scaling</td><td>14.7%</td></tr><tr><td>+ Gaussian Likelihood</td><td>18.9%</td></tr><tr><td>+ Multiple queries (our offline attack)</td><td>22.3%</td></tr><tr><td>LOSS attack [70]</td><td>0.0%</td></tr><tr><td>+ Per-example thresholds ( $\mathbb{Q}_{\text{in}}$  &amp;  $\mathbb{Q}_{\text{out}}$ ) [56]</td><td>7.4%</td></tr><tr><td>+ Logit scaling</td><td>2.8%</td></tr><tr><td>+ Gaussian Likelihood</td><td>24.1%</td></tr><tr><td>+ Multiple queries (our attack)</td><td>27.6%</td></tr></table>

TABLE V: Breakdown of how various components build up to obtain our best attacks on the CIFAR-100 dataset.

<table><tr><td>Attack Approach</td><td>TPR @ 0.1% FPR</td></tr><tr><td>LOSS attack [70]</td><td>0.1%</td></tr><tr><td>+ Logit scaling</td><td>0.1%</td></tr><tr><td>LOSS attack [70]</td><td>0.1%</td></tr><tr><td>+ Per-example thresholds ( $\tilde{\mathbb{Q}}_{\text{out}}$  only) [68]</td><td>1.1%</td></tr><tr><td>+ Logit scaling</td><td>1.1%</td></tr><tr><td>+ Gaussian Likelihood (our offline attack)</td><td>1.2%</td></tr><tr><td>LOSS attack [70]</td><td>0.1%</td></tr><tr><td>+ Per-example thresholds ( $\mathbb{Q}_{\text{in}}$  &amp;  $\mathbb{Q}_{\text{out}}$ ) [56]</td><td>1.0%</td></tr><tr><td>+ Logit scaling</td><td>1.0%</td></tr><tr><td>+ Gaussian Likelihood (our attack)</td><td>1.4%</td></tr></table>

TABLE VI: Breakdown of how various components build up to obtain our best attacks on the WikiText-103 dataset.

# D. Attack Ablations on Additional Datasets

Similarly to Table II for CIFAR-10, we now perform ablations on the different components of our attack for CIFAR-100 (Table V), WikiText-103 (Table VI), Texas (Table VII) and Purchase (Table VIII). Note that for WikiText-103, Texas and Purchase, we train models without any data augmentations and thus do not perform augmentations in the attack either.

As we observed in Section VI for CIFAR-10, a Gaussian Likelihood Test after logit scaling significantly boosts the performance of past attacks that rely on per-example thresholds, both in the offline case and in the online case.

In contrast to CIFAR-10, we observe that logit scaling on its own is often detrimental to the online attack of Sablayrolles et al. [56]. Similarly, we find that a Gaussian Likelihood Test on its own (i.e., without logit scaling) often hurts the attack performance. Thus, these two components necessarily have to be applied together to achieve a good attack performance.

# E. Full ROC Curves for Mismatched Training Procedures

In Figures 22 to 24, we plot full ROC curves for the experiments from Section VI-E, where the attacker has to

<table><tr><td>Attack Approach</td><td>TPR @ 0.1% FPR</td></tr><tr><td>LOSS attack [70]</td><td>0.1%</td></tr><tr><td>+ Logit scaling</td><td>0.1%</td></tr><tr><td>LOSS attack [70]</td><td>0.1%</td></tr><tr><td>+ Per-example thresholds ( $\tilde{\mathbb{Q}}_{\text{out only}}$ ) [68]</td><td>8.8%</td></tr><tr><td>+ Logit scaling</td><td>19.0%</td></tr><tr><td>+ Gaussian Likelihood (our offline attack)</td><td>24.6%</td></tr><tr><td>LOSS attack [70]</td><td>0.1%</td></tr><tr><td>+ Per-example thresholds ( $\mathbb{Q}_{\text{in}}$  &amp;  $\mathbb{Q}_{\text{out}}$ ) [56]</td><td>14.9%</td></tr><tr><td>+ Logit scaling</td><td>8.4%</td></tr><tr><td>+ Gaussian Likelihood (our attack)</td><td>33.2%</td></tr></table>

TABLE VII: Breakdown of how various components build up to obtain our best attacks on the Texas dataset.

<table><tr><td>Attack Approach</td><td>TPR @ 0.1% FPR</td></tr><tr><td>LOSS attack [70]</td><td>0.0%</td></tr><tr><td>+ Logit scaling</td><td>0.1%</td></tr><tr><td>LOSS attack [70]</td><td>0.0%</td></tr><tr><td>+ Per-example thresholds ( $\tilde{\mathbb{Q}}_{\text{out only}}$ ) [68]</td><td>1.5%</td></tr><tr><td>+ Logit scaling</td><td>1.3%</td></tr><tr><td>+ Gaussian Likelihood (our offline attack)</td><td>1.4%</td></tr><tr><td>LOSS attack [70]</td><td>0.0%</td></tr><tr><td>+ Per-example thresholds ( $\mathbb{Q}_{\text{in}}$  &amp;  $\mathbb{Q}_{\text{out}}$ ) [56]</td><td>2.7%</td></tr><tr><td>+ Logit scaling</td><td>0.2%</td></tr><tr><td>+ Gaussian Likelihood (our attack)</td><td>4.1%</td></tr></table>

TABLE VIII: Breakdown of how various components build up to obtain our best attacks on the Purchase dataset.

guess the architecture, optimizer and data augmentation used by the target model.

![](images/3267e9a37999e37e5c7f54701c43a2c0710e9256c3fcd3d9077c5282e81f157e.jpg)

<details>
<summary>line</summary>

| False Positive Rate | True Positive Rate (Line 1) | True Positive Rate (Line 2) | True Positive Rate (Line 3) | True Positive Rate (Line 4) | True Positive Rate (Line 5) |
| ------------------- | --------------------------- | --------------------------- | --------------------------- | --------------------------- | --------------------------- |
| 10⁻⁵                | ~10⁻³                       | ~10⁻³                       | ~10⁻³                       | ~10⁻³                       | ~10⁻³                       |
| 10⁻⁴                | ~10⁻²                       | ~10⁻²                       | ~10⁻²                       | ~10⁻²                       | ~10⁻²                       |
| 10⁻³                | ~10⁻¹                       | ~10⁻¹                       | ~10⁻¹                       | ~10⁻¹                       | ~10⁻¹                       |
| 10⁻²                | ~10⁰                        | ~10⁰                        | ~10⁰                        | ~10⁰                        | ~10⁰                        |
| 10⁻¹                | ~10⁰                        | ~10⁰                        | ~10⁰                        | ~10⁰                        | ~10⁰                        |
| 10⁰                 | ~10⁰                        | ~10⁰                        | ~10⁰                        | ~10⁰                        | ~10⁰                        |
</details>

(a) CNN-16 as target.

![](images/514967f4a3b9f878584ea2d5cf76656197eac02ee86a1adbbf7ea28e032b222f.jpg)

<details>
<summary>line</summary>

| False Positive Rate | True Positive Rate (Line 1) | True Positive Rate (Line 2) | True Positive Rate (Line 3) | True Positive Rate (Line 4) | True Positive Rate (Line 5) | True Positive Rate (Line 6) | True Positive Rate (Line 7) |
| ------------------- | --------------------------- | --------------------------- | --------------------------- | --------------------------- | --------------------------- | --------------------------- | --------------------------- |
| 10⁻⁵                | ~10⁻²                       | ~10⁻²                       | ~10⁻²                       | ~10⁻²                       | ~10⁻²                       | ~10⁻²                       | ~10⁻²                       |
| 10⁻⁴                | ~10⁻¹                       | ~10⁻¹                       | ~10⁻¹                       | ~10⁻¹                       | ~10⁻¹                       | ~10⁻¹                       | ~10⁻¹                       |
| 10⁻³                | ~10⁰                        | ~10⁰                        | ~10⁰                        | ~10⁰                        | ~10⁰                        | ~10⁰                        | ~10⁰                        |
| 10⁻²                | ~10⁰                        | ~10⁰                        | ~10⁰                        | ~10⁰                        | ~10⁰                        | ~10⁰                        | ~10⁰                        |
| 10⁻¹                | ~10⁰                        | ~10⁰                        | ~10⁰                        | ~10⁰                        | ~10⁰                        | ~10⁰                        | ~10⁰                        |
| 10⁰                 | ~10⁰                        | ~10⁰                        | ~10⁰                        | ~10⁰                        | ~10⁰                        | ~10⁰                        | ~10⁰                        |
</details>

(b) CNN-32 as target.

![](images/24b6e98e1196c5f69ce1b8500ab3702338cb6e77c34c415476f5a0d4e1dbce9c.jpg)

<details>
<summary>line</summary>

| False Positive Rate | True Positive Rate (Line 1) | True Positive Rate (Line 2) | True Positive Rate (Line 3) | True Positive Rate (Line 4) |
| ------------------- | --------------------------- | --------------------------- | --------------------------- | --------------------------- |
| 10⁻⁵                | ~10⁻²                       | ~10⁻²                       | ~10⁻²                       | ~10⁻³                       |
| 10⁻⁴                | ~10⁻¹                       | ~10⁻¹                       | ~10⁻¹                       | ~10⁻³                       |
| 10⁻³                | ~10⁰                        | ~10⁰                        | ~10⁰                        | ~10⁻²                       |
| 10⁻²                | ~10⁰                        | ~10⁰                        | ~10⁰                        | ~10⁻¹                       |
| 10⁻¹                | ~10⁰                        | ~10⁰                        | ~10⁰                        | ~10⁰                        |
| 10⁰                 | ~10⁰                        | ~10⁰                        | ~10⁰                        | ~10⁰                        |
</details>

(c) CNN-64 as target.

![](images/95b08f2316e1cfaad140aa022a614b32c48da5b813216ac173bee4fad2e8e99a.jpg)

<details>
<summary>line</summary>

| False Positive Rate | True Positive Rate (Red) | True Positive Rate (Blue) | True Positive Rate (Green) | True Positive Rate (Purple) | True Positive Rate (Brown) |
| ------------------- | ------------------------ | ------------------------- | -------------------------- | --------------------------- | -------------------------- |
| 10⁻⁵                | ~10⁻²                    | ~10⁻³                     | ~10⁻³                      | ~10⁻³                       | ~10⁻³                      |
| 10⁻⁴                | ~10⁻¹                    | ~10⁻³                     | ~10⁻²                      | ~10⁻²                       | ~10⁻²                      |
| 10⁻³                | ~10⁻¹                    | ~10⁻³                     | ~10⁻²                      | ~10⁻²                       | ~10⁻²                      |
| 10⁻²                | ~10⁻¹                    | ~10⁻³                     | ~10⁻²                      | ~10⁻²                       | ~10⁻²                      |
| 10⁻¹                | ~10⁻¹                    | ~10⁻³                     | ~10⁻²                      | ~10⁻²                       | ~10⁻²                      |
| 10⁰                 | 10⁰                      | 10⁰                       | 10⁰                        | 10⁰                         | 10⁰                        |
</details>

(d) WRN28-1 as target.

![](images/6f79cb802d34fe6088ef0ba02fed493d77824a0c24bbc22f75e17ed6793677ad.jpg)

<details>
<summary>line</summary>

| False Positive Rate | True Positive Rate (Line 1) | True Positive Rate (Line 2) | True Positive Rate (Line 3) | True Positive Rate (Line 4) |
| ------------------- | --------------------------- | --------------------------- | --------------------------- | --------------------------- |
| 1e-5                | ~1e-2                       | ~1e-2                       | ~1e-2                       | ~1e-4                       |
| 1e-4                | ~1e-2                       | ~1e-2                       | ~1e-2                       | ~1e-3                       |
| 1e-3                | ~1e-2                       | ~1e-2                       | ~1e-2                       | ~1e-3                       |
| 1e-2                | ~1e-2                       | ~1e-2                       | ~1e-2                       | ~1e-3                       |
| 1e-1                | ~1e-2                       | ~1e-2                       | ~1e-2                       | ~1e-3                       |
| 1e0                 | 1e0                         | 1e0                         | 1e0                         | 1e0                         |
</details>

(e) WRN28-2 as target.

![](images/5d75c7f0d5a4f09f4ed4580f15652208996d67521d677f8105094e21be432f87.jpg)

<details>
<summary>line</summary>

| False Positive Rate | CNN-16 | CNN-32 | CNN-64 | WRN28-1 | WRN28-2 | WRN28-10 |
| ------------------- | ------ | ------ | ------ | ------- | ------- | -------- |
| 10⁻⁵                | ~10⁻³  | ~10⁻³  | ~10⁻³  | ~10⁻³   | ~10⁻³   | ~10⁻³    |
| 10⁻⁴                | ~10⁻²  | ~10⁻²  | ~10⁻²  | ~10⁻²   | ~10⁻²   | ~10⁻²    |
| 10⁻³                | ~10⁻¹  | ~10⁻¹  | ~10⁻¹  | ~10⁻¹   | ~10⁻¹   | ~10⁻¹    |
| 10⁻²                | ~10⁰   | ~10⁰   | ~10⁰   | ~10⁰    | ~10⁰    | ~10⁰     |
| 10⁻¹                | ~10⁰   | ~10⁰   | ~10⁰   | ~10⁰    | ~10⁰    | ~10⁰     |
| 10⁰                 | ~10⁰   | ~10⁰   | ~10⁰   | ~10⁰    | ~10⁰    | ~10⁰     |
</details>

(f) WRN28-10 as target.   
Fig. 22: Different architectures with momentum optimizer and mirror & shift as augmentation.

![](images/2da55a928a1390094f19482bea279d93d29996beef073eb378eeec858dcec7c7.jpg)

<details>
<summary>line</summary>

| False Positive Rate | True Positive Rate |
| ------------------- | ------------------ |
| 10⁻⁵                | 10⁻²               |
| 10⁻⁴                | 10⁻¹               |
| 10⁻³                | 10⁻¹               |
| 10⁻²                | 10⁻¹               |
| 10⁻¹                | 10⁻¹               |
| 10⁰                 | 10⁰                |
</details>

(a) SGD as target.

![](images/bc2ce23f2b9b623c918cc706981916f32cf35a8d2ea6cf3662ce94b066b3ede6.jpg)

<details>
<summary>line</summary>

| False Positive Rate | True Positive Rate (Orange) | True Positive Rate (Blue) | True Positive Rate (Green) |
| ------------------- | --------------------------- | ------------------------- | -------------------------- |
| 10⁻⁵                | ~0.02                       | ~0.02                     | ~0.01                      |
| 10⁻⁴                | ~0.05                       | ~0.04                     | ~0.03                      |
| 10⁻³                | ~0.1                        | ~0.08                     | ~0.06                      |
| 10⁻²                | ~0.2                        | ~0.15                     | ~0.1                       |
| 10⁻¹                | ~0.5                        | ~0.3                      | ~0.2                       |
| 10⁰                 | ~1.0                        | ~1.0                      | ~1.0                       |
</details>

(b) Momentum as target.

![](images/64a4df441cee8a534201c5a2b48e2077c43d6da54da05c45ef951bfac522d8c0.jpg)

<details>
<summary>line</summary>

| False Positive Rate | SGD     | SGDM    | Adam    |
| ------------------- | ------- | ------- | ------- |
| 10⁻⁵                | ~10⁻³   | ~10⁻³   | ~10⁻²   |
| 10⁻⁴                | ~10⁻²   | ~10⁻²   | ~10⁻¹   |
| 10⁻³                | ~10⁻¹   | ~10⁻¹   | ~10⁰    |
| 10⁻²                | ~10⁰    | ~10⁰    | ~10⁰    |
| 10⁻¹                | ~10⁰    | ~10⁰    | ~10⁰    |
| 10⁰                 | ~10⁰    | ~10⁰    | ~10⁰    |
</details>

(c) Adam as target.   
Fig. 23: Different optimizers on WRN28-10 with mirror & shift as augmentation.

![](images/f4bd49588aaacee9e323167ec62269d6fc4a1423041feb70dad9d9ba28dba888.jpg)

<details>
<summary>line</summary>

| False Positive Rate | True Positive Rate (Line 1) | True Positive Rate (Line 2) | True Positive Rate (Line 3) | True Positive Rate (Line 4) |
| ------------------- | --------------------------- | --------------------------- | --------------------------- | --------------------------- |
| 10⁻⁵                | ~10⁻¹                       | ~10⁻³                       | ~10⁻⁴                       | ~10⁻⁵                       |
| 10⁻⁴                | ~10⁻¹                       | ~10⁻²                       | ~10⁻³                       | ~10⁻⁴                       |
| 10⁻³                | ~10⁻¹                       | ~10⁻²                       | ~10⁻³                       | ~10⁻³                       |
| 10⁻²                | ~10⁻¹                       | ~10⁻²                       | ~10⁻³                       | ~10⁻³                       |
| 10⁻¹                | ~10⁻¹                       | ~10⁻²                       | ~10⁻³                       | ~10⁻³                       |
| 10⁰                 | ~10⁰                        | ~10⁰                        | ~10⁰                        | ~10⁰                        |
</details>

(a) No augmentation.

![](images/b6e951c0bfbdb2cd7fcf66a7991985ead5458ddf6a9b91ba772cc8862b4e3341.jpg)

<details>
<summary>line</summary>

| False Positive Rate | True Positive Rate (Red) | True Positive Rate (Green) | True Positive Rate (Blue) | True Positive Rate (Orange) |
| ------------------- | ------------------------ | -------------------------- | ------------------------- | --------------------------- |
| 10⁻⁵                | ~10⁻³                    | ~10⁻³                      | ~10⁻²                     | ~10⁻¹                       |
| 10⁻⁴                | ~10⁻³                    | ~10⁻³                      | ~10⁻²                     | ~10⁻¹                       |
| 10⁻³                | ~10⁻³                    | ~10⁻³                      | ~10⁻²                     | ~10⁻¹                       |
| 10⁻²                | ~10⁻³                    | ~10⁻³                      | ~10⁻²                     | ~10⁻¹                       |
| 10⁻¹                | ~10⁻³                    | ~10⁻³                      | ~10⁻²                     | ~10⁻¹                       |
| 10⁰                 | ~10⁻³                    | ~10⁻³                      | ~10⁻²                     | ~10⁻¹                       |
</details>

(b) Mirror.

![](images/b80606aad3e6e82ce6e5eee1e24b2c98e348f2d853636d4e724d3dc1a347c888.jpg)

<details>
<summary>line</summary>

| False Positive Rate | True Positive Rate (Green) | True Positive Rate (Orange) | True Positive Rate (Red) | True Positive Rate (Blue) |
| ------------------- | -------------------------- | --------------------------- | ------------------------ | ------------------------- |
| 10⁻⁵                | ~10⁻²                      | ~10⁻³                       | ~10⁻³                    | ~10⁻⁴                     |
| 10⁻⁴                | ~10⁻¹                      | ~10⁻²                       | ~10⁻²                    | ~10⁻³                     |
| 10⁻³                | ~10⁰                       | ~10⁻¹                       | ~10⁻¹                    | ~10⁻²                     |
| 10⁻²                | ~10⁰                       | ~10⁰                        | ~10⁰                     | ~10⁻¹                     |
| 10⁻¹                | ~10⁰                       | ~10⁰                        | ~10⁰                     | ~10⁰                      |
| 10⁰                 | ~10⁰                       | ~10⁰                        | ~10⁰                     | ~10⁰                      |
</details>

(c) Mirror+Shift.

![](images/0ca00f7d642a2afc5a07dde5d647951ec7b57d9a41afb336b00743debcce5780.jpg)

<details>
<summary>line</summary>

| False Positive Rate | None     | +Mirror  | +Shift   | +Cutout  |
| ------------------- | -------- | -------- | -------- | -------- |
| 10⁻⁵                | 10⁻⁵     | 10⁻⁴     | 10⁻³     | 10⁻²     |
| 10⁻⁴                | 10⁻⁴     | 10⁻³     | 10⁻²     | 10⁻¹     |
| 10⁻³                | 10⁻³     | 10⁻²     | 10⁻¹     | 10⁰      |
| 10⁻²                | 10⁻²     | 10⁻¹     | 10⁰      | 10¹      |
| 10⁻¹                | 10⁻¹     | 10⁰      | 10¹      | 10²      |
| 10⁰                 | 10⁰      | 10¹      | 10²      | 10³      |
</details>

(d) Mirror+Shift+Cutout16.   
Fig. 24: Different augmentations on WRN28-10 with momentum optimizer.