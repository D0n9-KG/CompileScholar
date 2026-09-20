# What is Flagged in Uncertainty Quantification? Latent Density Models for Uncertainty Categorization

Hao Sun $^{\dagger,*}$ , Boris van Breugel $^{\dagger}$ , Jonathan Crabbé, Nabeel Seedat, Mihaela van der Schaar
Department of Applied Mathematics and Theoretical Physics
University of Cambridge

# Abstract

Uncertainty Quantification (UQ) is essential for creating trustworthy machine learning models. Recent years have seen a steep rise in UQ methods that can flag suspicious examples, however, it is often unclear what exactly these methods identify. In this work, we propose a framework for categorizing uncertain examples flagged by UQ methods in classification tasks. We introduce the confusion density matrix—a kernel-based approximation of the misclassification density—and use this to categorize suspicious examples identified by a given uncertainty method into three classes: out-of-distribution (OOD) examples, boundary (Bnd) examples, and examples in regions of high in-distribution misclassification (IDM). Through extensive experiments, we show that our framework provides a new and distinct perspective for assessing differences between uncertainty quantification methods, thereby forming a valuable assessment benchmark.

# 1 Introduction

Black-box parametric models like neural networks have achieved remarkably good performance on a variety of challenging tasks, yet many real-world applications with safety concerns—e.g. healthcare $[1]$ , finance $[2, 3]$ and autonomous driving $[4, 5]$ —necessitate reliability. These scenarios require trustworthy model predictions to avoid the high cost of erroneous decisions.

Uncertainty quantification $[6–10]$ addresses the challenge of trustworthy prediction through inspecting the confidence of a model, enabling intervention whenever uncertainty is too high. Usually, however, the cause of the uncertainty is not clear. In this work, we go beyond black-box UQ in the context classification tasks and aim to answer two questions:

1. How do we provide a more granular categorization of why uncertainty methods to identify certain predictions as suspicious?   
2. What kind of examples do different UQ methods tend to mark as suspicious?

Categorizing Uncertainty We propose a Density-based Approach for Uncertainty Categorization (DAUC): a model-agnostic framework that provides post-hoc categorization for model uncertainty. We introduce the confusion density matrix, which captures the predictive behaviors of a given model. Based on such a confusion density matrix, we categorize the model's uncertainty into three classes: (1) OOD uncertainty caused by OOD examples, i.e. test-time examples that resemble no training-time sample [11–14]. Such uncertainty can be manifested by a low density in the confusion density matrix; (2) Bnd uncertainty caused by neighboring decision boundaries, i.e., non-conformal predictions due to confusing training-time resemblances from different classes or inherent ambiguities making it challenging to classify the data [15–17]; Such uncertainty can be manifested by the density of

![](images/daaf1bb1b86f28de08e0d928f0feb85a4c1b598872348cc509018e7b6c7feae2.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Test Examples"] --> B["Model"]
    B --> C["Model Uncertainty"]
    C --> D["Trusted Predictions"]
    D --> E["Untrusted Predictions"]
    E --> F["Model's Latent Space Inspection"]
    F --> G["Interpretable Categories of Untrusted Predictions"]
    G --> H["Predict Class 1"]
    H --> I["Inspect"]
    I --> J["OOD Example"]
    I --> K["Boundary Example"]
    I --> L["Boundary & IDM Example"]
    I --> M["IDM Example"]
    H --> N["Predict Class 2"]
    N --> O["Untrusted Test Example"]
    O --> P["Improve Model Prediction"]
    P --> Q["Model"]
    Q --> R["Predictions"]
    R --> S["Improve Model Prediction"]
    S --> T["Model Uncertainty"]
    T --> U["Trusted Predictions"]
    U --> V["Untrusted Predictions"]
    V --> W["Model's Latent Space Inspection"]
    W --> X["Interpretable Categories of Untrusted Predictions"]
    X --> Y["Predict Class 1"]
    Y --> Z["Inspect"]
    Z --> AA["OOD Example"]
    Z --> AB["Boundary Example"]
    Z --> AC["Boundary & IDM Example"]
    Z --> AD["IDM Example"]
```
</details>

Figure 1: Given a prediction and UQ algorithm, our method divides flagged test time examples into different classes. Class OOD identifies outliers, which are mistrusted because they do not resemble training data; class IDM indicates examples lying in regions with high misclassification; class Bnd indicates examples that lie near the decision boundary.

diagonal elements in the confusion density matrix; (3) IDM uncertainty caused by imperfections in the model—as manifested by high misclassification at validation time—such that similar misclassification is to be expected during testing [18]. Such uncertainty can be manifested by the density of off-diagonal elements in the confusion density matrix. Figure 1 illustrates the different classes and in Section 4 we provide visualisations using real data. We show how DAUC can be used to benchmark a broad class of UQ methods, by categorizing what each method tends to flag as suspicious.

# Our contributions can be summarized as follows:

1. Formally, we propose the confusion density matrix—the heart of DAUC—that links the training time error, decision boundary ambiguity, and uncertainty with latent representation density.   
2. Practically, we leverage DAUC as a unified framework for uncertain example categorization. DAUC offers characterisation of uncertain examples at test time.   
3. Empirically, we use DAUC to benchmark existing UQ methods. We manifest different methods' sensitivity to different types of uncertain examples, this provides model insight and aids UQ method selection.

# 2 Related Work

Uncertainty Quantification Uncertainty quantification methods are used to assess the confidence in a model's predictions. In recent years the machine learning community has proposed many methods which broadly fall into the following categories: (1) Ensemble methods (i.e. Deep Ensembles [6]), which—while considered state of the art—have a high computational burden; (2) Approximate Bayesian methods (i.e. Stochastic Variational Inference [7–9]), however work by [19–21] suggests these methods do not yield high-quality uncertainty estimates; and (3) Dropout-based methods such as Monte Carlo Dropout [10], which are simpler as they do rely on estimating a posterior, however the quality of the uncertainty estimates is linked to the choice of parameters which need to be calibrated to match the level of uncertainty and avoid suboptimal performance [22]. For completeness we note that whilst conformal prediction [23] is another UQ method, the paradigm is different from the other aforementioned methods, as it returns predictive sets to satisfy coverage guarantees rather than a value-based measure of uncertainty.

In practice, the predictive uncertainty for a model prediction arises from the lack of relevant training data (epistemic uncertainty) or the inherent non-separable property of the data distribution (aleatoric uncertainty) $[24–26]$ . This distinction in types of uncertainty is crucial as it has been shown that samples with low epistemic uncertainty are more likely under the data distribution, hence motivating why epistemic uncertainty has been used for OOD/outlier detection $[10]$ . Moreover, ambiguous instances close to the decision boundary typically result in high aleatoric uncertainty $[27]$ .

This suggests that different sets of uncertain data points are associated with different types of uncertainties and, consequently, different types of misclassifications. Furthermore, it is to be expected that different uncertainty estimators are more able/prone to capturing some types of uncertainty than others. To understand these models better, we thus require a more granular definition to characterize

Table 1: Comparison with related work in UQ. We aim to provide a flexible framework for inspecting mistrusted examples identified by uncertainty estimators. Furthermore, this framework enables us to improve the prediction performance on a certain type of flagged uncertain class. 

<table><tr><td>Method</td><td>Model Structure</td><td>Uncertainty Estimation</td><td>Categorize Uncertainty</td><td>Improve Prediction</td><td>Examples</td></tr><tr><td>BNNs</td><td>Bayesian Layers</td><td>√</td><td>·</td><td>·</td><td>[8, 9]</td></tr><tr><td>GP</td><td>Gaussian Processes</td><td>√</td><td>·</td><td>·</td><td>[39]</td></tr><tr><td>MC-Dropout</td><td>Drop-Out Layers</td><td>√</td><td>·</td><td>·</td><td>[10]</td></tr><tr><td>Deep-Ensemble</td><td>Multiple Models</td><td>√</td><td>·</td><td>·</td><td>[6]</td></tr><tr><td>ICP</td><td>Model “Wrapper”</td><td>√</td><td>·</td><td>·</td><td>[23, 40],</td></tr><tr><td>Performance Prediction</td><td>Multiple Predictive Models</td><td>√</td><td>·</td><td>·</td><td>[37, 18]</td></tr><tr><td>DAUC</td><td>Assumption 1</td><td>√</td><td>√</td><td>√</td><td>(Ours)</td></tr></table>

uncertain data points. We employ three classes: outliers (OOD), boundary examples (Bnd) and examples of regions with high in-distribution misclassification (IDM), see Fig 1.

Out-of-distribution (OOD) detection Recall that UQ methods have been used to flag OOD examples. For completeness, we highlight that other alternative methods exist for OOD detection. For example, Lee et al. [28] detects OOD examples based on the Mahalanobis distance, whilst Ren et al. [29] uses the likelihood ratio between two generative models. Besides supervised learning, OOD is an essential topic in offline reinforcement learning [30–35]. Sun et al. [36] detects the OOD state in the context of RL using confidence intervals. We emphasize that DAUC's aim is broader—creating a unifying framework for categorizing multiple types of uncertainty—however, existing OOD methods could be used to replace DAUC's OOD detector.

Accuracy without Labels We contrast our work to the literature which aims to determine model accuracy without access to ground-truth labels. Methods such as $[37, 38]$ propose a secondary regression model as an accuracy predictor given a data sample. Ramalho and Miranda $[18]$ combine regression model with latent nearest neighbors for uncertain prediction. This is different from our setting which is focused on inspecting UQ methods on a sample level by characterizing it as an outlier, boundary, or IDM example. We contrast DAUC with related works in Table 1.

# 3 Categorizing Model Uncertainty via Latent Density

# 3.1 Preliminaries

We consider a typical classification setting where $X \subseteq R^{d_{X}}$ is the input space and $Y = [0,1]^{C}$ is the set of class probabilities, where $d_{X}$ is the dimension of input space and $C \in N^{*}$ is the number of classes. We are given a prediction model $f : X \to Y$ that maps $x \in X$ to class probabilities $f(x) \in Y$ . An uncertainty estimator $u : X \to [0,1]$ quantifies the uncertainty of the predicted outcomes. Given some threshold $\tau$ , the inference-time predictions can be separated into trusted predictions $\{x \in X | u(x) < \tau\}$ and untrusted predictions $\{x \in X | u(x) \geq \tau\}$ . We make the following assumption on the model architecture of f.

Assumption 1 (Model Architecture). Model f can be decomposed as $f = \varphi \circ l \circ g$ , where $g : X \to H \subseteq R^{d_H}$ is a feature extractor that maps the input space to a $d_H < d_X$ dimensional latent (or representation) space, $l : H \to R^C$ is a linear map between the latent space to the output space and $\varphi : R^C \to Y$ is a normalizing map that converts vectors into probabilities.

Remark 1. This assumption guarantees that the model is endowed with a lower-dimensional representation space. Most modern uncertainty estimation methods like MCD, Deep-Ensemble, BNN satisfy this assumption. In the following, we use such a space to categorize model uncertainty.

We assume that the model and the uncertainty estimator have been trained with a set of $N \in \mathbb{N}^*$ training examples $\mathcal{D}_{\mathrm{train}} = \{(x^n, y^n) \mid n \in [N]\}$ . At inference time the underlying model $f$ and uncertainty method $u$ predict class probabilities $f(x) \in \mathcal{Y}$ and uncertainty $u(x) \in [0,1]$ , respectively. We assign a class to this probability vector $f(x)$ with the map class: $\mathcal{Y} \to [C]$ that maps a probability vector $y$ to the class with maximal probability class $[y] = \arg \max_{c \in [C]} y_c$ . While uncertainty estimators flag examples to be trustworthy or not, those estimators do not provide a fine-grained reason for what a certain prediction should not be mistrusted. Our aim is to use the

model's predictions and representations of a corpus of labelled examples—which we will usually take to be the training $(\mathcal{D}_{\mathrm{train}})$ or validation $(\mathcal{D}_{\mathrm{val}})$ sets—to categorize inference-time uncertainty predictions. To that aim, we distinguish two general scenarios where a model's predictions should be considered with skepticism.

# 3.2 Flagging OOD Examples

There is a limit to model generalization. Uncertainty estimators should be skeptical when the input $x \in X$ differs significantly from input examples that the model was trained on. From an UQ perspective, the predictions for these examples are expected to be associated with a large epistemic uncertainty.

A natural approach to flagging these examples is to define a density $p(\cdot \mid \mathcal{D}_{\mathrm{train}}) : \mathcal{X} \to \mathbb{R}^{+}$ over the input space. This density should be such that $p(x \mid \mathcal{D}_{\mathrm{train}})$ is high whenever the example $x \in X$ resembles one or several examples from the training set $D_{train}$ . Conversely, a low value for $p(x \mid \mathcal{D}_{\mathrm{train}})$ indicates that the example x differs from the training examples. Of course, estimating the density $p(x \mid \mathcal{D}_{\mathrm{train}})$ is a nontrivial task. At this stage, it is worth noting that this density does not need to reflect the ground-truth data generating process underlying the training set $D_{train}$ . For the problem at hand, this density $p(x \mid \mathcal{D}_{\mathrm{train}})$ need only measure how close the example x is to the training data manifold. A common approach is to build a kernel density estimation with the training set $D_{train}$ . Further, we note that Assumption 1 provides a representation space H that was specifically learned for the classification task on $D_{train}$ . In Appendix A.1, we argue that this latent space is suitable for our kernel density estimation. This motivates the following definition for $p(\cdot \mid \mathcal{D}_{\mathrm{train}})$ .

Definition 1 (Latent Density). Let $f: \mathcal{X} \to \mathcal{Y}$ be a prediction model, let $g: \mathcal{X} \to \mathcal{H}$ be the feature extractor from Assumption 1 and let $\kappa: \mathcal{H} \times \mathcal{H} \to \mathbb{R}^+$ be a kernel function. The latent density $p(\cdot | \mathcal{D}): \mathcal{X} \to \mathbb{R}^+$ is defined over a dataset $\mathcal{D}$ as:

$$
p (x \mid \mathcal {D}) \equiv \frac {1}{N} \sum_ {\tilde {x} \in \mathcal {D}} \kappa [ g (x), g (\tilde {x}) ] \tag {1}
$$

Test examples with low training density are likely to be underfitted for the model — thus should not be trusted.

Definition 2 (OOD Score). The OOD Score $T_{\mathrm{OOD}}$ is defined as

$$
T _ {\mathrm{OOD}} (x) \equiv \frac {1}{p (x | \mathcal {D} _ {\text { train }})} \tag {2}
$$

For a test example $x \in \mathcal{D}_{\mathrm{test}}$ , if $T_{\mathrm{OOD}}(x|\mathcal{D}_{\mathrm{train}}) \geq \tau_{\mathrm{OOD}}$ the example is suspected to be an outlier with respect to the training set. We set $\tau_{\mathrm{OOD}} = \frac{1}{\min_{x' \in \mathcal{D}_{\mathrm{train}}} p(x'| \mathcal{D}_{\mathrm{train}})}$ , i.e. a new sample's training density is smaller than the minimal density of training examples.

# 3.3 Flagging IDM and Boundary Examples

Samples that are not considered outliers, yet are given high uncertainty scores, we divide up further into two non-exclusionary categories. The first category consists of points located near the boundary between two or more classes, the second consists of points that are located in regions of high misclassification.

For achieving this categorization, we will use a separate validation set $D_{val}$ . We first partition the validation examples according to their true and predicted label. More precisely, for each couple of classes $(c_{1}, c_{2}) \in [C]^{2}$ , we define the corpus $\mathcal{C}_{c_{1} \mapsto c_{2}} \equiv \{(x, y) \in \mathcal{D}_{\text{val}} \mid \text{class}[y] = c_{1} \land \text{class}[f(x)] = c_{2}\}$ of validation examples whose true class is $c_{1}$ and whose predicted class is $c_{2}$ . In the case where these two classes are different $c_{1} \neq c_{2}$ , this corresponds to a corpus of misclassified examples. Clearly, if some example $x \in X$ resembles one or several examples of those misclassification corpus, it is legitimate to be skeptical about the prediction $f(x)$ for this example. In fact, keeping track of the various misclassification corpora from the validation set allows us to have an idea of what the misclassification is likely to be.

In order to make this detection of suspicious examples quantitative, we will mirror the approach from Definition 1. Indeed, we can define a kernel density $p(\cdot \mid \mathcal{C}_{c_1 \mapsto c_2}) : \mathcal{X} \to \mathbb{R}^+$ for each corpus

$\mathcal{C}_{c_1 \mapsto c_2}$ . Again, this density will be such that $p(\cdot \mid \mathcal{C}_{c_1 \mapsto c_2})$ is high whenever the representation of the example $x \in \mathcal{X}$ resembles the representation of one or several examples from the corpus $\mathcal{C}_{c_1 \mapsto c_2}$ . If this corpus is a corpus of misclassified examples, this should trigger our skepticism about the model's prediction $f(x)$ . By aggregating the densities associated with each of these corpora, we arrive at the following definition.

Definition 3 (Confusion Density Matrix). Let $f: \mathcal{X} \to \mathcal{Y}$ be a prediction model, let $g: \mathcal{X} \to \mathcal{H}$ be the feature extractor from Assumption 1 and let $\kappa: \mathcal{H} \times \mathcal{H} \to \mathbb{R}^{+}$ be a kernel function. The confusion density matrix $P(\cdot \mid \mathcal{D}_{\mathrm{val}}): \mathcal{X} \to (\mathbb{R}^{+})^{C \times C}$ is defined as

$$
P _ {c _ {1}, c _ {2}} \left(x \mid \mathcal {D} _ {\text { val }}\right) \equiv p \left(x \mid \mathcal {C} _ {c _ {1} \mapsto c _ {2}}\right) = \frac {1}{\left| \mathcal {C} _ {c _ {1} \mapsto c _ {2}} \right|} \sum_ {\tilde {x} \in \mathcal {C} _ {c _ {1} \mapsto c _ {2}}} \kappa [ g (x), g (\tilde {x}) ], \forall (c _ {1}, c _ {2}) \in [ C ] ^ {2} \tag {3}
$$

Remark 2. The name confusion density is chosen to make a parallel with confusion matrices. Like confusion matrices, our confusion density indicates the likelihood of each couple $(c_{1}, c_{2})$ , where $c_{1}$ is the true class and $c_{2}$ is the predicted class. Unlike confusion matrices, our confusion density provides an instance-wise (i.e. for each $x \in X$ ) likelihood for each couple.

# 3.3.1 Bnd examples

By inspecting the confusion matrix, we can quantitatively distinguish two situations where the example $x \in \mathcal{X}$ is likely to be mistrusted. The first situation where high uncertainty arises, is when $g(x)$ is close to latent representations of validation examples that have been correctly assigned a label that differs from the predicted one class $[f(x)]$ . This typically happens when $g(x)$ is located close to a decision boundary in latent space. In our validation confusion matrix, this likelihood that $x$ is related to validation examples with different labels is reflected by the diagonal elements. This motivates the following definition.

Definition 4 (Boundary Score). Let $P(x \mid \mathcal{D}_{\mathrm{val}})$ be the confusion density matrix for an example $x \in \mathcal{X}$ with predicted class $\hat{c} = \operatorname{class}[f(x)]$ . We define the boundary score as the sum of each density of well-classified examples from a different class:

$$
T _ {\mathrm{Bnd}} (x) = \sum_ {c \neq \hat {c}} ^ {C} P _ {c, c} \left(x \mid \mathcal {D} _ {\mathrm{val}}\right). \tag {4}
$$

Points are identified as Bnd when $T_{Bnd} > \tau_{Bnd}$ —see Appendix A.2.

# 3.3.2 IDM examples

The second situation is the one previously mentioned: the latent representation $g(x)$ is close to latent representations of validation examples that have been misclassified. In our validation confusion matrix, this likelihood that x is related to misclassified validation examples is reflected by the off-diagonal elements. This motivates the following definition.

Definition 5 (IDM Score). Let $P(x \mid \mathcal{D}_{\mathrm{val}})$ be the confusion density matrix for an example $x \in \mathcal{X}$ . We define the IDM score as the sum of each density corresponding to a misclassification of the predicted class $\hat{c} = \text{class } f(x)$ in the confusion density matrix:

$$
T _ {\mathrm{IDM}} (x) = \sum_ {c \neq \hat {c}} ^ {C} P _ {c, \hat {c}} \left(x \mid \mathcal {D} _ {\text { val }}\right). \tag {5}
$$

Points are identified as IDM when $T_{IDM} > \tau_{IDM}$ . We choose $\tau_{IDM}$ such that the proportion of IDM points in the validation set equals the number of misclassified examples. Details for definitions and choices of thresholds are provided in Appendix A.2.

Remark 3. Note that the definitions of Bnd examples and IDM examples do not exclude each other, therefore, an uncertain example can be flagged as a Bnd example, an IDM example, or flagged as both Bnd and IDM (B&I). To make this distinction clear, we will refer to the disjoint classes as:

$$
\mathcal {S} _ {\text { Bnd }} = \{x | x \in \mathcal {D} _ {\text { test }}, T _ {\text { Bnd }} (x) > \tau_ {\text { Bnd }}, T _ {\text { IDM }} (x) \leq \tau_ {\text { IDM }} \}
$$

$$
\mathcal {S} _ {\mathrm{IDM}} = \left\{x \mid x \in \mathcal {D} _ {\text {test}}, T _ {\mathrm{Bnd}} (x) \leq \tau_ {\mathrm{Bnd}}, T _ {\mathrm{IDM}} (x) > \tau_ {\mathrm{IDM}} \right\}
$$

$$
\mathcal {S} _ {\mathrm{B} \& \mathrm{I}} = \left\{x \mid x \in \mathcal {D} _ {\text {test}}, T _ {\text {Bnd}} (x) > \tau_ {\text {Bnd}}, T _ {\text {IDM}} (x) > \tau_ {\text {IDM}} \right\}
$$

Test examples that are flagged by uncertainty method $u$ — yet do not meet any of the thresholds—are marked as Other.

In a nutshell DAUC uses the OOD, IDM and Bnd classes to categorize model uncertainty—see Table 2 for an overview. Better predictions may be possible for IDM samples, in case a different classifier is used. For samples that are also labelled as Bnd, fine-tuning the existing model—possibly after gathering more data—may be able to separate the different classes better. IDM samples that are not in the Bnd class are harder, and may only be classified correctly if a different latent representation is found, or an different training set is used. We explore the idea of improving the performance on uncertain examples in Appendix B. In Section 4.1 we explore the distinction between classes further.

Table 2: Summary of different uncertainty types 

<table><tr><td>Type</td><td>Definition</td><td>Description</td></tr><tr><td>OOD</td><td> $1/p(x|\mathcal{D}_{\text{train}}) > \tau_{\text{OOD}}$ </td><td>Samples that do not resemble the training data. Additional labelled data that covers this part of the input space is required to improve performance on these samples.</td></tr><tr><td>Bnd</td><td> $\sum_{c \neq \hat{c}} P_{c,c}(x|\mathcal{D}_{\text{val}}) > \tau_{\text{Bnd}}$ </td><td>Samples near the boundaries in the latent space. Predictions on these samples are sensitive to small changes in the predictor, and fine-tuning the prediction model may yield better predictions.</td></tr><tr><td>IDM</td><td> $\sum_{c \neq \hat{c}} P_{c,\hat{c}}(x|\mathcal{D}_{\text{val}}) > \tau_{\text{IDM}}$ </td><td>Samples that are likely to be misclassified, since similar examples were misclassified in the validation set.</td></tr></table>

# 4 Experiments

In this section, we demonstrate our proposed method with empirical studies. Specifically, we use two experiments as Proof-of-Concept, and two experiments as Use Cases. Specifically, in Sec. 4.1 we visualize the different classes of flagged examples on a modified Two-Moons dataset; in Sec. 4.2 we quantitatively assess DAUC's categorization accuracy on the Dirty-MNIST dataset [41]; in Sec. 4.3, we present a use case of DAUC—comparing existing uncertainty estimation benchmarks; in Sec. 4.4, we demonstrate another important use case of DAUC—improving uncertain predictions.

Our selection of the Dirty-MNIST dataset for empirical evaluation was motivated by the pursuit of better reproducibility. As an existing publicly available resource, Dirty-MNIST provides gold labels for boundary classes and OOD examples, making it particularly suitable for benchmarking the performance of DAUC.

Recognizing the importance of demonstrating the broader applicability of DAUC, we have extended our evaluation to include results on the Dirty-CIFAR dataset. Details of this additional evaluation are available in Appendix C.4, where we also describe how we created the dataset. This dataset will also be made publicly available. Additional empirical evidence that justifies DAUC is provided in Appendix C.

# 4.1 Visualizing DAUC with Two-Smiles

# 4.1.1 Experiment Settings

To highlight the different types of uncertainty, we create a modified version of the Two-Moons dataset, which we will call “Two-Smiles”. We use scikit-learn’s datasets package to generate 6000 two-moons examples with a noise rate of 0.1. In addition, we generate two Gaussian clusters of 150 examples centered at $(0,1.5)$ and $(1,-1)$ for training, validation and test, and mark them as the positive and negative class separately. The data is split into a training, validation and test set. We add an additional 1500 OOD examples to the test set, that are generated with a Gaussian distribution centered at $(2,2)$ and $(-1,-1.5)$ . Overall, the test set counts 4500 examples, out of which 1500 are positive examples, 1500 are negative examples and 1500 are OOD—see Figure 2 (a).

# 4.1.2 Results

We demonstrate the intuition behind the different types of uncertainty, by training a linear model to classify the Two-Smiles data—i.e. using a composition of an identity embedding function and linear classifier, see Assumption 1. Figure 2 (b) shows the test-time misclassified examples. Since the identity function does not linearly separate the two classes and outliers are unobserved at training

![](images/9e46f28d33542a480c0292c3344d409288d7d954ba3b1b5566286edb6b98542d.jpg)  
(a) Test Dataset

![](images/b967c844ecb34f7f8a45b24c7df8f6a3b9c8e1cfa388971e979c6644d89888f9.jpg)  
(b) Test Errors

![](images/8ac32ee8348bde9bdb7076ddbb163ffe2cda7e544b68f504d306f118ad11fe7a.jpg)  
(c) Flagged OOD

![](images/f731138152cda148d6d08bbe5a9138add6f290f99355f71daa25e76969427798.jpg)  
(d) Flagged Bnd

![](images/95bd4d73bafe1e615dc3eba85ed1902a88502ecca378486141f420c0b7adba4b.jpg)  
(e) Flagged IDM   
Figure 2: Visualization of our proposed framework on the Two-Smiles dataset with a linear model.

time, the classifier makes mistakes near class boundaries, in predicting the OOD examples, as well as the clusters at $(0,1.5)$ and $(1,-1)$ . In Figure 2 (c)-(e), we see that DAUC correctly flags the OOD examples, boundary examples and the IDM examples. Also note the distinction between $S_{B\&I}$ and $S_{IDM}$ . Slightly fine-tuning the prediction model may yield a decision boundary that separates the $S_{B\&I}$ samples from the incorrect class. On the other hand, this will not suffice for the $S_{IDM}$ examples—i.e. the clusters at $(0,1.5)$ and $(1,-1)$ —which require a significantly different representation/prediction model to be classified correctly.

# 4.2 Verifying DAUC with Dirty-MNIST:

# 4.2.1 Experiment Settings

We evaluate the effectiveness of DAUC on the Dirty-MNIST dataset $[41]$ . The training data set is composed of the vanilla MNIST dataset and Ambiguous-MNIST, containing ambiguous artificial examples—e.g., digits 4 that look like 9s. The test set is similar but also contains examples from Fashion-MNIST as OOD examples. The advantage of this dataset is that the different data types roughly correspond to the categories we want to detect, and as such we can verify whether DAUC's classification corresponds to these “ground-truth” labels. Specifically, we deem all examples from MNIST to be non-suspicious, associate Ambiguous-MNIST with class Bnd, and associate Fashion-MNIST with class OOD (There are 1,000 OOD examples out of the 11,000 test examples). See $[41]$ for more details on the dataset. We use a 3-layer CNN model with ReLU activation and MaxPooling as the backbone model in this section.

# 4.2.2 Results

![](images/2677ab28c3b34d1e7584435cb7ec4e51f493d9268e03b3d4d70342f0c03d7cd9.jpg)  
(a) OOD

![](images/eb4ec25e3eb8cf851041bfe0c51705f7364c318dd967471755348d52c3e1c089.jpg)  
(b) Bnd

![](images/488f715d3b1e9c35fae6761990321c111d7c31470647f5797ecd36f102e8f479.jpg)  
(c) IDM

![](images/193d803e76770ebd6c04066a83ad4b6e880082124dd6e1509a7e8ea27e3c65a6.jpg)  
(d) Flag Bnd

![](images/b26eba7f0779adc82924a042bc2c83fdd75d94f7d7173ad3224f48e3a4592a0c.jpg)  
(e) Flag IDM   
Figure 3: (a-c) Box-plots of different data classes and corresponding scores. All values are scaled by being divided by the maximal value. (d-e) Precision-Recall curves for different choices of thresholds in flagging Bnd examples and IDM examples.

Flagging Outliers We assume the ground-truth label of all Fashion-MNIST data is OOD. Comparing these to the classification by DAUC, we are able to quantitatively evaluate the performance of DAUC in identifying outliers. We compute precision, recall and F1-scores for different classes, see Figure 3 (a) and Table 3. All Fashion-MNIST examples are successfully flagged as OOD by DAUC. In Appendix C.3 we include more OOD experiments, including comparison to OOD detector baselines.

Table 3: Quantitative results on the Dirty-MNIST dataset. Our proposed method can flag all three classes of uncertain examples with high F1-Score. The results presented in the table are based on 8 repeated runs. 

<table><tr><td>Category</td><td>Precision</td><td>Recall</td><td>F1-Score</td></tr><tr><td>OOD</td><td> $1.000 \pm 0.000$ </td><td> $1.000 \pm 0.000$ </td><td> $1.000 \pm 0.000$ </td></tr><tr><td>Bnd</td><td> $0.959 \pm 0.008$ </td><td> $0.963 \pm 0.006$ </td><td> $0.961 \pm 0.003$ </td></tr><tr><td>IDM</td><td> $0.918 \pm 0.039$ </td><td> $0.932 \pm 0.024$ </td><td> $0.924 \pm 0.012$ </td></tr></table>

![](images/6237a9685ef5573000c579a1d4a7a5324bceb6880f15db9ab3278686cf50ef62.jpg)

<details>
<summary>text_image</summary>

9
7
8
outlier
9
OOD Examples
Class 7 Class 7 Class 8 Class 8 Class 9 Class 9
Bnd Examples
7 2 8 8 9 9
Class 7 Class 7 Class 8 Class 8 Class 9 Class 9
7 2 1 6 9 9
IDM Examples
Class 7 Class 7 Class 8 Class 8 Class 9 Class 9
7 9 6 9 9 9
</details>

Figure 4: Examples of different uncertainty classes $S_{OOD}$ , $S_{IDM}$ , $S_{Bnd}$ and $S_{B\&I}$ . For better visualization, we only plot some classes including the outliers. t-SNE [42] is leveraged in generating low-dim visualizations.

Flagging Bnd Examples We expect most of the boundary examples to belong to the Ambiguous-MNIST class, as these have been synthesised using a linear interpolation of two different digits in latent space [41]. Figure 3 (b) shows that DAUC's boundary scores are indeed significantly higher in Ambiguous-MNIST compared to vanilla MNIST. Figure 3 (d) shows the precision-recall curve of DAUC, created by varying threshold $\tau_{\mathrm{Bnd}}$ . Most boundary examples are correctly discovered under a wide range of threshold choices. This stability of uncertainty categorization is desirable, since $\tau_{\mathrm{Bnd}}$ is usually unknown exactly.

Flagging IDM Examples In order to quantitatively evaluate the performance of DAUC on flagging IDM examples, we use a previously unseen hold-out set, which is balanced to consist of 50% misclassified and 50% correctly classified examples. We label the former as test-time IDM examples and the latter as non-IDM examples, and compare this to DAUC's categorization. Figure 3c shows DAUC successfully assigns significantly higher IDM scores to the examples that are to-be misclassified.

Varying $\tau_{IDM}$ we create the precision-recall curve for the IDM class in Figure 3 (e), which is fairly stable w.r.t. the threshold $\tau_{IDM}$ . In practice, we recommend to use the prediction accuracy on the validation dataset for setting $\tau_{IDM}$ —see Appendix A.2.

Visualizing Different Classes Figure 4 shows examples from the $S_{OOD}$ , $S_{Bnd}$ , $S_{IDM}$ and $S_{B\&I}$ sets. The first row shows OOD examples from the Fashion-MNIST dataset. The second row shows boundary examples, most of which indeed resemble more than one class. The third row shows IDM examples, which DAUC thinks are likely to be misclassified since mistakes were made nearby on the validation set. Indeed, these examples look like they come from the “dirty” part of dirty-MNIST, and most digits are not clearly classifiable. The last row contains B&I examples, which exhibit both traits.

# 4.3 Benchmark Model Uncertainty Categorization

In this section, we demonstrate how DAUC categorizes existing UQ model uncertainty. We compare UQ methods MC-Dropout [10] (MCD), Deep-Ensemble [6] (DE) and Bayesian Neural Networks [9] (BNNs). These methods output predictions and uncertainty scores simultaneously, we follow the traditional approach to mark examples as uncertain or trusted according to their uncertain scores and specified thresholds. To demonstrate how DAUC categorizes all ranges of uncertain examples, we present results from top 5% to the least 5% uncertainty.

Figure 5 compares the proportion of different classes of flagged examples across the three UQ methods. The first and second rows show the total number and proportion of flagged examples for each class, respectively. There is a significant difference in what the different UQ methods identify. There is a significant difference between types of classes that the different UQ methods identify as discussed in the literature [19–21], which have shown that some UQ methods might not yield high quality uncertainty estimates due to the learning paradigm or sensitivity to parameters. Let us look at each column more carefully:

![](images/4adf8a76d73c59f739edb8594a48da5ffae04d596bd1045d85fe3ee6a3aeb9ca.jpg)

<details>
<summary>line</summary>

| Top Uncertainty | BNNs | MCD  | DE   |
| --------------- | ---- | ---- | ---- |
| 0               | 0    | 0    | 0    |
| 10              | 600  | 400  | 800  |
| 20              | 700  | 450  | 900  |
| 30              | 750  | 500  | 950  |
| 40              | 800  | 550  | 950  |
| 50              | 850  | 600  | 950  |
| 60              | 850  | 650  | 950  |
| 70              | 850  | 700  | 950  |
| 80              | 850  | 750  | 950  |
| 90              | 850  | 800  | 950  |
| 100             | 1000 | 1000 | 1000 |
</details>

![](images/d0815ec2c10a30d080781a35c20889b23a872b9d7facf4d957235b40a441e8db.jpg)

<details>
<summary>line</summary>

| Top Uncertainty | BNNs  | MCD   | DE    |
| --------------- | ----- | ----- | ----- |
| 0               | 0     | 0     | 0     |
| 10              | 500   | 500   | 500   |
| 20              | 1000  | 1000  | 1000  |
| 30              | 1500  | 1500  | 1500  |
| 40              | 1750  | 1750  | 1750  |
| 50              | 1875  | 1875  | 1875  |
| 60              | 2000  | 2000  | 2000  |
| 70              | 2000  | 2000  | 2000  |
| 80              | 2000  | 2000  | 2000  |
| 90              | 2000  | 2000  | 2000  |
</details>

![](images/36a9801a0eed74128fbd57a12d96afe073bdced3d681a5c102e45f66989a2ae8.jpg)

<details>
<summary>line</summary>

| Top Uncertainty | BNNs  | MCD   | DE    |
| --------------- | ----- | ----- | ----- |
| 0               | 0     | 0     | 0     |
| 10              | 500   | 500   | 500   |
| 20              | 1000  | 1000  | 1000  |
| 30              | 1500  | 1500  | 1500  |
| 40              | 1750  | 1750  | 1750  |
| 50              | 2000  | 2000  | 2000  |
| 60              | 2100  | 2100  | 2100  |
| 70              | 2150  | 2150  | 2150  |
| 80              | 2200  | 2200  | 2200  |
| 90              | 2250  | 2250  | 2250  |
</details>

![](images/5e797139983e39ae50bf611f9ce1ddc8c4eed0c2d7a486e3334cbd825027f468.jpg)

<details>
<summary>line</summary>

| Top Uncertainty | BNNs  | MCD   | DE    |
| --------------- | ----- | ----- | ----- |
| 0               | 0     | 0     | 0     |
| 10              | 500   | 500   | 500   |
| 20              | 1500  | 1500  | 1500  |
| 30              | 2500  | 2500  | 2500  |
| 40              | 3500  | 3500  | 3500  |
| 50              | 4500  | 4500  | 4500  |
| 60              | 5500  | 5500  | 5500  |
| 70              | 6500  | 6500  | 6500  |
| 80              | 7000  | 7000  | 7000  |
| 90              | 7500  | 7500  | 7500  |
</details>

![](images/87a63bdcf7538145e1205aad2ba364dc1e8a2cdabe3851af54e55f2b4abaee5b.jpg)

![](images/4ae25ccc083f563e42e8c70590d71ccf43eccbc1f2b6f7961e66c5d4795c4f1a.jpg)

<details>
<summary>line</summary>

| Top Uncertainty | BNNs  | MCD   | DE    |
| --------------- | ----- | ----- | ----- |
| 0               | 0.00  | 0.00  | 0.00  |
| 10              | 0.10  | 0.15  | 0.12  |
| 20              | 0.18  | 0.22  | 0.20  |
| 30              | 0.20  | 0.24  | 0.23  |
| 40              | 0.21  | 0.25  | 0.24  |
| 50              | 0.22  | 0.25  | 0.24  |
| 60              | 0.21  | 0.24  | 0.23  |
| 70              | 0.20  | 0.23  | 0.22  |
| 80              | 0.19  | 0.22  | 0.21  |
| 90              | 0.18  | 0.21  | 0.20  |
| 100             | 0.17  | 0.20  | 0.19  |
</details>

![](images/deb825edbdb1d2f8d1204a6231559f1b28d95f8008891aaac45562056ceb74e5.jpg)

<details>
<summary>line</summary>

| Top Uncertainty | BNNs  | MCD   | DE    |
| --------------- | ----- | ----- | ----- |
| 0               | 0.00  | 0.00  | 0.00  |
| 10              | 0.10  | 0.15  | 0.12  |
| 20              | 0.18  | 0.22  | 0.18  |
| 30              | 0.20  | 0.24  | 0.20  |
| 40              | 0.21  | 0.25  | 0.21  |
| 50              | 0.22  | 0.25  | 0.22  |
| 60              | 0.23  | 0.24  | 0.23  |
| 70              | 0.23  | 0.23  | 0.23  |
| 80              | 0.22  | 0.22  | 0.22  |
| 90              | 0.21  | 0.21  | 0.21  |
| 100             | 0.20  | 0.20  | 0.20  |
</details>

![](images/d758ab649aadd8b6bbfb82b715bd9717ab27ac3dd16858458b5af370b9cd4ac4.jpg)  
Figure 5: Results of applying our method in categorizing different uncertainty estimation methods. First row: comparisons on the numbers in different classes of examples. Second row: comparisons on the proportion of different classes of flagged examples to the total number of identified uncertain examples. Different methods tend to identify different certain types of uncertain examples. The results presented are based on 8 repeated runs with different random seeds.

1. DE tends to identify the OOD examples as the most uncertain examples. Specifically, looking at the bottom figure we see that the top 5% untrusted examples identified by DE are almost all OOD examples, which is not the case for the other UQ methods. By contrast, MCD is poor at identifying OOD examples; it flags some of the OOD samples as the most certain. This is explained by MCD's mechanism. Uncertainty is based on the difference in predictions across different drop-outs, however this could lead to outliers always having the same prediction—due to correlation between different nodes in the MCD model, extreme values of the OOD examples may always saturate the network and lead to the same prediction, even if some nodes are dropped out. DE is most apt at flagging the OOD class. This confirms the finding by [20] who showed that DE outperforms other methods under dataset shift—which is effectively what OOD represents.   
2. After the OOD examples have been flagged, the next most suspicious examples are the Bnd and IDM classes—see columns 2 and 3. The number of these examples increases almost linearly with the number of flagged examples, until at about 88% no more examples are flagged as IDM and Bnd. This behaviour is explained by the Vanilla MNIST examples—which are generally distinguishable and relatively easily classified correctly—accounting for about 15% of the test examples.   
3. As expected, the number of examples belonging to the Other class increases when more examples are flagged as uncertain. This makes sense, as the Other class indicates we cannot flag why the methods flagged these examples as uncertain, i.e. maybe these predictions should in fact be trusted.

# 4.4 Improving Uncertain Predictions

In this section, we explore the inverse direction, i.e. employing DAUC's categorization for creating better models. We elaborate the practical method in Appendix B. We experiment on UCI's Covtype, Digits and Spam dataset [43] with linear models (i.e. $g = Id$ ) and experiment on DMNIST with ResNet-18 learning the latent representation.

Our empirical studies (Figure 8, Appendix C) have shown that the B&I examples are generally hardest to classify, hence we demonstrate the use case of DAUC for improving the predictive performance on this class. We vary the proportion q of training samples that we discard before training new prediction model $f_{B\&I}$ —only saving the training samples that resemble the B&I dataset most—see Figure 6. We find that retraining the linear model with filtered training data according to Eq. 6 significantly improves the performance. We observe that performance increases approximately linearly proportional to q, until the amount of training data becomes too low. The latter depends on the dataset and model used.

![](images/ea130a829a9aa7d2b6e64ba3ff8d18611dab38bd798a68e261fa1f5a7b46bc2e.jpg)

<details>
<summary>line</summary>

| Quantile q | w/ Filtered Training Set | Baseline |
| ---------- | ------------------------ | -------- |
| 0.00       | 0.25                     | 0.25     |
| 0.10       | 0.30                     | 0.25     |
| 0.20       | 0.35                     | 0.25     |
| 0.30       | 0.40                     | 0.25     |
| 0.40       | 0.45                     | 0.25     |
| 0.50       | 0.48                     | 0.25     |
| 0.60       | 0.49                     | 0.25     |
| 0.70       | 0.49                     | 0.25     |
| 0.80       | 0.48                     | 0.25     |
| 0.90       | 0.47                     | 0.25     |
| 0.95       | 0.46                     | 0.25     |
</details>

![](images/a8b94e432dde8edbba632509f6bb7e29c24ebc9e58c41519373b6f36dbe23f52.jpg)

<details>
<summary>line</summary>

| Quantile q | w/ Filtered Training Set | Baseline |
| ---------- | ------------------------ | -------- |
| 0.00       | 0.0                      | 0.0      |
| 0.10       | 0.2                      | 0.0      |
| 0.20       | 0.3                      | 0.0      |
| 0.30       | 0.4                      | 0.0      |
| 0.40       | 0.5                      | 0.0      |
| 0.50       | 0.6                      | 0.0      |
| 0.60       | 0.7                      | 0.0      |
| 0.70       | 0.75                     | 0.0      |
| 0.80       | 0.8                      | 0.0      |
| 0.85       | 0.8                      | 0.0      |
</details>

![](images/47c2216a8a67a956f500a997399799bca49cb95e08f562c7f397d4b9f836202c.jpg)

<details>
<summary>line</summary>

| Quantile q | w/ Filtered Training Set | Baseline |
| ---------- | ------------------------ | -------- |
| 0.00       | 0.60                     | 0.60     |
| 0.10       | 0.68                     | 0.60     |
| 0.20       | 0.71                     | 0.60     |
| 0.30       | 0.71                     | 0.60     |
| 0.40       | 0.71                     | 0.60     |
| 0.50       | 0.71                     | 0.60     |
| 0.60       | 0.73                     | 0.60     |
| 0.70       | 0.75                     | 0.60     |
| 0.80       | 0.77                     | 0.60     |
| 0.90       | 0.75                     | 0.60     |
| 0.95       | 0.72                     | 0.60     |
</details>

![](images/c4d107d196d3c6c747029e8f419ec7be59330b6c2551294d347e5ecde6cadcd5.jpg)

<details>
<summary>line</summary>

| Quantile q | w/ Filtered Training Set | Baseline |
| ---------- | ------------------------ | -------- |
| 0.0        | 0.64                     | 0.64     |
| 0.1        | 0.65                     | 0.64     |
| 0.2        | 0.66                     | 0.64     |
| 0.3        | 0.67                     | 0.64     |
| 0.4        | 0.68                     | 0.64     |
| 0.5        | 0.68                     | 0.64     |
| 0.6        | 0.68                     | 0.64     |
| 0.7        | 0.67                     | 0.64     |
| 0.8        | 0.65                     | 0.64     |
| 0.9        | 0.62                     | 0.64     |
</details>

Figure 6: Improved uncertain predictions. Left to right: Covtype, Digits, Spam, DMNIST. Experiment are performed with different proportion of training samples discarded: e.g., with q = 0.0, all examples in the training set are used; while with q = 0.9, only top 10% examples most resembling the test data are used for training. The results presented are based on 10 repeated runs with different random seeds.

# 5 Conclusion and Future Work

We have proposed DAUC, a framework for model uncertainty categorization. DAUC categorizes uncertain examples identified by UQ benchmarks into three classes—OOD, Bnd and IDM. These classes correspond to different causes for the uncertainty and require different strategies for possibly better predictions. We have demonstrated the power of DAUC by inspecting three different UQ methods—highlighting that each one identifies different examples. We believe DAUC can aid the development and benchmarking of UQ methods, paving the way for more trustworthy ML models.

In future work, DAUC has great potential to be extended to more general tasks, such as the regression setting, and reinforcement learning setting, where uncertainty quantification is essential. The idea of separating the source of uncertainty improves not only exploration $[44]$ but also exploitation in the offline settings $[30–35]$ .

In the era of Large Language Models (LLMs) [45, 46], uncertainty quantification is essential in evaluating the task performance of LLMs [47], and holds great potential for AI alignment [48, 49] — as understanding the ability boundary of LLMs is essential, and identifying the suspicious outputs of LLMs can be potentially addressed by extending the framework of DAUC to the LLMs' setting.

# Acknowledgement

HS acknowledges and thanks the funding from the Office of Naval Research (ONR). We thank the van der Schaar lab members for reviewing the paper and sharpening the idea. We thank all anonymous reviewers, ACs, SACs, and PCs for their efforts and time in the reviewing process and in improving our paper.

# References

[1] Ravi Aggarwal, Viknesh Sounderajah, Guy Martin, Daniel SW Ting, Alan Karthikesalingam, Dominic King, Hutan Ashrafian, and Ara Darzi. Diagnostic accuracy of deep learning in medical imaging: a systematic review and meta-analysis. NPJ digital medicine, 4(1):1–23, 2021.   
[2] Alisa Kim, Y Yang, Stefan Lessmann, Tiejun Ma, M-C Sung, and Johnnie EV Johnson. Can deep learning predict risky retail investors? a case study in financial risk behavior forecasting. European Journal of Operational Research, 283(1):217–234, 2020.   
[3] Qianggang Ding, Sifan Wu, Hao Sun, Jiadong Guo, and Jian Guo. Hierarchical multi-scale gaussian transformer for stock movement prediction. In IJCAI, pages 4640–4646, 2020.   
[4] Jinkyu Kim and John Canny. Interpretable learning for self-driving cars by visualizing causal attention. In Proceedings of the IEEE international conference on computer vision, pages 2942–2950, 2017.   
[5] Jiankai Sun, Hao Sun, Tian Han, and Bolei Zhou. Neuro-symbolic program search for autonomous driving decision module design. In Jens Kober, Fabio Ramos, and Claire Tomlin, editors, Proceedings of the 2020 Conference on Robot Learning, volume 155 of Proceedings of Machine Learning Research, pages 21–30. PMLR, 16–18 Nov 2021. URL https://proceedings.mlr.press/v155/sun21a.html.   
[6] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. In Proceedings of the 31st International Conference on Neural Information Processing Systems, pages 6405–6416, 2017.   
[7] Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra. Weight uncertainty in neural network. In International Conference on Machine Learning, pages 1613–1622. PMLR, 2015.   
[8] Alex Graves. Practical variational inference for neural networks. Advances in neural information processing systems, 24, 2011.   
[9] Soumya Ghosh, Jiayu Yao, and Finale Doshi-Velez. Structured variational learning of bayesian neural networks with horseshoe priors. In International Conference on Machine Learning, pages 1744–1753. PMLR, 2018.   
[10] Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning, pages 1050–1059. PMLR, 2016.   
[11] Andrey Malinin and Mark Gales. Predictive uncertainty estimation via prior networks. arXiv preprint arXiv:1802.10501, 2018.   
[12] Joost Van Amersfoort, Lewis Smith, Yee Whye Teh, and Yarin Gal. Uncertainty estimation using a single deep deterministic neural network. In International Conference on Machine Learning, pages 9690–9700. PMLR, 2020.   
[13] Moloud Abdar, Farhad Pourpanah, Sadiq Hussain, Dana Rezazadegan, Li Liu, Mohammad Ghavamzadeh, Paul Fieguth, Xiaochun Cao, Abbas Khosravi, U Rajendra Acharya, et al. A review of uncertainty quantification in deep learning: Techniques, applications and challenges. Information Fusion, 2021.   
[14] Xuming Ran, Mingkun Xu, Lingrui Mei, Qi Xu, and Quanying Liu. Detecting out-of-distribution samples via variational auto-encoder with reliable uncertainty estimation. Neural Networks, 145:199–208, 2022.   
[15] Dimitris Tsipras, Shibani Santurkar, Logan Engstrom, Andrew Ilyas, and Aleksander Madry. From imagenet to image classification: Contextualizing progress on benchmarks. In International Conference on Machine Learning, pages 9625–9635. PMLR, 2020.

[16] Ikki Kishida and Hideki Nakayama. Empirical study of easy and hard examples in cnn training. In International Conference on Neural Information Processing, pages 179–188. Springer, 2019.   
[17] Rajmadhan Ekambaram, Dmitry B Goldgof, and Lawrence O Hall. Finding label noise examples in large scale datasets. In 2017 IEEE International Conference on Systems, Man, and Cybernetics (SMC), pages 2420–2424. IEEE, 2017.   
[18] Tiago Ramalho and Miguel Miranda. Density estimation in representation space to predict model uncertainty. In International Workshop on Engineering Dependable and Secure Machine Learning Systems, pages 84–96. Springer, 2020.   
[19] Jiayu Yao, Weiwei Pan, Soumya Ghosh, and Finale Doshi-Velez. Quality of uncertainty quantification for bayesian neural network inference. arXiv preprint arXiv:1906.09686, 2019.   
[20] Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D Sculley, Sebastian Nowozin, Joshua Dillon, Balaji Lakshminarayanan, and Jasper Snoek. Can you trust your model's uncertainty? evaluating predictive uncertainty under dataset shift. Advances in Neural Information Processing Systems, 32:13991–14002, 2019.   
[21] Andrew YK Foong, David R Burt, Yingzhen Li, and Richard E Turner. On the expressiveness of approximate inference in bayesian neural networks. arXiv preprint arXiv:1909.00719, 2019.   
[22] Francesco Verdoja and Ville Kyrki. Notes on the behavior of mc dropout. arXiv preprint arXiv:2008.02627, 2020.   
[23] Vladimir Vovk, Alexander Gammerman, and Glenn Shafer. Algorithmic learning in a random world. Springer Science & Business Media, 2005.   
[24] Alex Kendall and Yarin Gal. What uncertainties do we need in bayesian deep learning for computer vision? Advances in Neural Information Processing Systems, 30:5574–5584, 2017.   
[25] Moloud Abdar, Farhad Pourpanah, Sadiq Hussain, Dana Rezazadegan, Li Liu, Mohammad Ghavamzadeh, Paul Fieguth, Xiaochun Cao, Abbas Khosravi, U. Rajendra Acharya, Vladimir Makarenkov, and Saeid Nahavandi. A review of uncertainty quantification in deep learning: Techniques, applications and challenges. Information Fusion, 76:243–297, 2021. ISSN 1566-2535. doi: 10.1016/j.inffus.2021.05.008.   
[26] Eyke Hüllermeier and Willem Waegeman. Aleatoric and epistemic uncertainty in machine learning: an introduction to concepts and methods. Machine Learning 2021 110:3, 110(3):457–506, 2021. ISSN 1573-0565. doi: 10.1007/S10994-021-05946-3.   
[27] Lisa Schut, Oscar Key, Rory Mc Grath, Luca Costabello, Bogdan Sacaleanu, Yarin Gal, et al. Generating interpretable counterfactual explanations by implicit minimisation of epistemic and aleatoric uncertainties. In International Conference on Artificial Intelligence and Statistics, pages 1756–1764. PMLR, 2021.   
[28] Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin. A simple unified framework for detecting out-of-distribution samples and adversarial attacks. Advances in neural information processing systems, 31, 2018.   
[29] Jie Ren, Peter J Liu, Emily Fertig, Jasper Snoek, Ryan Poplin, Mark Depristo, Joshua Dillon, and Balaji Lakshminarayanan. Likelihood ratios for out-of-distribution detection. Advances in Neural Information Processing Systems, 32:14707–14718, 2019.   
[30] Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643, 2020.   
[31] Rui Yang, Yiming Lu, Wenzhe Li, Hao Sun, Meng Fang, Yali Du, Xiu Li, Lei Han, and Chongjie Zhang. Rethinking goal-conditioned supervised learning and its connection to offline rl. arXiv preprint arXiv:2202.04478, 2022.   
[32] Xiaoyu Wen, Xudong Yu, Rui Yang, Chenjia Bai, and Zhen Wang. Towards robust offline-to-online reinforcement learning via uncertainty and smoothness. arXiv preprint arXiv:2309.16973, 2023.

[33] Rui Yang, Chenjia Bai, Xiaoteng Ma, Zhaoran Wang, Chongjie Zhang, and Lei Han. Rorl: Robust offline reinforcement learning via conservative smoothing. Advances in Neural Information Processing Systems, 35:23851–23866, 2022.   
[34] Rui Yang, Lin Yong, Xiaoteng Ma, Hao Hu, Chongjie Zhang, and Tong Zhang. What is essential for unseen goal generalization of offline goal-conditioned rl? In International Conference on Machine Learning, pages 39543–39571. PMLR, 2023.   
[35] Zhihan Liu, Yufeng Zhang, Zuyue Fu, Zhuoran Yang, and Zhaoran Wang. Learning from demonstration: Provably efficient adversarial policy imitation with linear function approximation. In International Conference on Machine Learning, pages 14094–14138. PMLR, 2022.   
[36] Hao Sun, Alihan Hüyük, Daniel Jarrett, and Mihaela van der Schaar. Accountability in offline reinforcement learning: Explaining decisions with a corpus of examples. arXiv preprint arXiv:2310.07747, 2023.   
[37] Sindhu Ghanta, Sriram Subramanian, Lior Khermosh, Harshil Shah, Yakov Goldberg, Swaminathan Sundararaman, Drew Roselli, and Nisha Talagala. {MPP}: Model performance predictor. In 2019 {USENIX} Conference on Operational Machine Learning (OpML 19), pages 23–25, 2019.   
[38] Weijian Deng and Liang Zheng. Are labels always necessary for classifier accuracy evaluation? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15069–15078, 2021.   
[39] Carl Edward Rasmussen. Gaussian processes in machine learning. In Summer school on machine learning, pages 63–71. Springer, 2003.   
[40] Vineeth Balasubramanian, Shen-Shyang Ho, and Vladimir Vovk. Conformal prediction for reliable machine learning: theory, adaptations and applications. Newnes, 2014.   
[41] Jishnu Mukhoti, Andreas Kirsch, Joost van Amersfoort, Philip HS Torr, and Yarin Gal. Deterministic neural networks with appropriate inductive biases capture epistemic and aleatoric uncertainty. arXiv preprint arXiv:2102.11582, 2021.   
[42] Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9(11), 2008.   
[43] Dheeru Dua and Casey Graff. UCI machine learning repository, 2017. URL http://archive.ics.uci.edu/ml.   
[44] Daniel Jarrett, Corentin Tallec, Florent Altché, Thomas Mesnard, Rémi Munos, and Michal Valko. Curiosity in hindsight. arXiv preprint arXiv:2211.10515, 2022.   
[45] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.   
[46] R OpenAI. Gpt-4 technical report. arXiv, pages 2303-08774, 2023.   
[47] Hao Sun. Offline prompt evaluation and optimization with inverse reinforcement learning. arXiv preprint arXiv:2309.06553, 2023.   
[48] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.   
[49] Hao Sun. Reinforcement learning in the era of llms: What is essential? what is needed? an rl perspective on rlhf, prompting, and beyond. arXiv preprint arXiv:2310.06147, 2023.   
[50] Jonathan Crabbé, Zhaozhi Qian, Fergus Imrie, and Mihaela van der Schaar. Explaining latent representations with a corpus of examples. NeurIPS 2021, 2021.

[51] Bolin Gao and Lacra Pavel. On the properties of the softmax function with application in game theory and reinforcement learning. arXiv preprint arXiv:1704.00805, 2017.   
[52] Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.   
[53] Arnaud Van Looveren, Janis Klaise, Giovanni Vacanti, Oliver Cobb, Ashley Scillitoe, and Robert Samoilescu. Alibi detect: Algorithms for outlier, adversarial and drift detection, 2019. URL https://github.com/SeldonIO/alibi-detect.   
[54] Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou. Isolation forest. In 2008 eighth ieee international conference on data mining, pages 413–422. IEEE, 2008.   
[55] Simon J Sheather and Michael C Jones. A reliable data-based bandwidth selection method for kernel density estimation. Journal of the Royal Statistical Society: Series B (Methodological), 53(3):683–690, 1991.   
[56] David W Scott. Multivariate density estimation and visualization. In Handbook of computational statistics, pages 549–569. Springer, 2012.   
[57] Bernard W Silverman. Density estimation for statistics and data analysis. Routledge, 2018.   
[58] Soumya Ghosh, Q. Vera Liao, Karthikeyan Natesan Ramamurthy, Jiri Navratil, Prasanna Sattigeri, Kush R. Varshney, and Yunfeng Zhang. Uncertainty quantification 360: A holistic toolkit for quantifying and communicating the uncertainty of ai, 2021.   
[59] Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, et al. Scikit-learn: Machine learning in python. the Journal of machine Learning research, 12:2825–2830, 2011.

# A Missing Details

# A.1 Motivations for working with model latent space

In Section 3, we introduced the confusion density matrix that allows us to categorize suspicious examples at testing time. Crucially, this density matrix relies on kernel density estimations in the latent space H associated with the model f through Assumption 1. Why are we performing a kernel density estimation in latent space rather than in input space X? The answer is fairly straightforward: we want our density estimation to be coupled to the model and its predictions.

Let us now make this point more rigorous. Consider two input examples $x_{1}, x_{2} \in X$ . The model assigns a representations $g(x_{1}), g(x_{2}) \in \mathcal{H}$ and class probabilities $f(x_{1}), f(x_{2}) \in \mathcal{Y}$ . If we define our kernel $\kappa$ in latent space H, this often means $^{2}$ that $\kappa[g(x_{1}), g(x_{2})]$ grows as $\|g(x_{1}) - g(x_{2})\|_{\mathcal{H}}$ decreases. Hence, examples that are assigned a similar latent representation by the model f are related by the kernel. Since our whole discussion revolves around model predictions, we would like to guarantee that two examples related by the kernel are given similar predictions by the model f. In this way, we would be able to interpret a large kernel density $\kappa[g(x_{1}), g(x_{2})]$ as a hint that the predictions $f(x_{1})$ and $f(x_{2})$ are similar. We will now show that, under Assumption 1, such a guarantee exists. Similar to [50], we start by noting that

$$
\begin{array}{l} \left\| (l \circ g) \left(x _ {1}\right) - (l \circ g) \left(x _ {2}\right) \right\| _ {\mathbb {R} ^ {C}} = \left\| l [ g \left(x _ {1}\right) - g \left(x _ {2}\right) ] \right\| _ {\mathbb {R} ^ {C}} \\ \leq \| l \| _ {\mathrm{op}} \left\| g (x _ {1}) - g (x _ {2}) \right\| _ {\mathcal {H}}, \\ \end{array}
$$

where $\|\cdot\|_{R^{C}}$ is a norm on $R^{C}$ and $\|l\|_{op}$ is the operator norm of the linear map l. In order to extend this inequality to black-box predictions, we note that the normalizing map in Assumption 1 is often a Lipschitz function with Lipschitz constant $\lambda\in R$ . For instance, a Softmax function with inverse temperature constant $\lambda^{-1}$ is $\lambda$ -Lipschitz [51]. We use this fact to extend our inequality to predicted class probabilities:

$$
\| f (x _ {1}) - f (x _ {2}) \| _ {\mathcal {Y}} = \| (\varphi \circ l \circ g) (x _ {1}) - (\varphi \circ l \circ g) (x _ {2}) \| _ {\mathcal {Y}}
$$

$$
\leq \lambda \| (l \circ g) (x _ {1}) - (l \circ g) (x _ {2}) \| _ {\mathbb {R} ^ {C}}
$$

$$
\leq \lambda \| l \| _ {\mathrm{op}} \| g (x _ {1}) - g (x _ {2}) \| _ {\mathcal {H}}.
$$

This crucial inequality guarantees that examples $x_{1}, x_{2} \in X$ that are given a similar latent representation $g(x_{1}) \approx g(x_{2})$ will also be given a similar prediction $f(x_{1}) \approx f(x_{2})$ . In short: two examples that are related according to a kernel density defined in the model latent space H are guaranteed to have similar predictions. This is the motivation we wanted to support the definition of the kernel $\kappa$ in latent space.

An interesting question remains: is it possible to have similar guarantees if we define the kernel in input space? When we deal with deep models, the existence of adversarial examples indicates the opposite [52]. Indeed, if $x_{2}$ is an adversarial example with respect to $x_{1}$ , we have $x_{1} \approx x_{2}$ (and hence $\|x_{1} - x_{2}\|_{\mathcal{X}}$ small) with two predictions $f(x_{1})$ and $f(x_{2})$ that are significantly different. Therefore, defining the kernel $\kappa$ in input space might result in relating examples that are given a significantly different prediction by the model. For this reason, we believe that the latent space is more appropriate in our setting.

# A.2 Details: Flagging IDM and Bnd Examples with Thresholds

In order to understand uncertainty, it will be clearer to map those scores into binary classes with thresholds. In our experiments, we use empirical quantiles as thresholds. e.g., to label an example as IDM, we specify an empirical quantile number q, and calculate the corresponding threshold based on the order statistics of IDM Scores for test examples: $S_{\mathrm{IDM}}^{(1)}, \ldots, S_{\mathrm{IDM}}^{(|\mathcal{D}_{\mathrm{test}}|)}$ , where $S_{\mathrm{IDM}}^{(n)}$ denotes the n-th smallest IDM score out of $|D_{test}|$ testing-time examples. Then, the threshold given quantile number q is

$$
\tau_ {\mathrm{IDM}} (q) \equiv S _ {\mathrm{IDM}} ^ {(\lfloor | \mathcal {D} _ {\mathrm{test}} | \cdot q \rfloor)}.
$$

Similarly, we can define quantile-based threshold in flagging Bnd examples based on the order statistics of Bnd Scores for test examples, such that for given quantile q,

$$
\tau_ {\mathrm{Bnd}} (q) \equiv S _ {\mathrm{Bnd}} ^ {(\lfloor | \mathcal {D} _ {\mathrm{test}} | \cdot q \rfloor)}.
$$

Practically, a natural choice of q is to use the validation accuracy: when there are 1 - q examples misclassified in the validation set, we also expect the testing-time in distribution examples with the highest 1 - q to be marked as Bnd or IDM examples.

# B Improving Predicting Performance of Uncertain Examples

Knowing the category that a suspicious example belongs to, can we improve its prediction? For ease of exposition, we focus on improving predictions for $S_{B\&I}$ .

Let $p(x \mid S_{\mathrm{B\&I}})$ be the latent density be defined as in Definition 1. We can improve the prediction performance of the model on $S_{\mathrm{B\&I}}$ examples by focusing on the part of examples in the training set that are closely related to those suspicious examples. We propose to refine the training dataset $D_{train}$ by only keeping the examples that resembles the latent representations for the specific type of test-time suspicious examples, and train another model on this subset of the training data:

$$
\tilde {\mathcal {D}} _ {\text { train }} \equiv \{x \in \mathcal {D} _ {\text { train }} | p (x \mid \mathcal {S} _ {\mathrm{B} \& \mathrm{I}}) \geq \tau_ {\text { test }} \}, \tag {6}
$$

where $\tau_{test}$ is a threshold that can be adjusted to keep a prespecified proportion q of the related training data. Subsequently, new prediction model $f_{B\&I}$ is trained on $\hat{D}_{train}$ .

Orthogonal to ensemble methods that require multiple models trained independently, and improve overall prediction accuracy by bagging or boosting, our method is targeted at improving the model's performance on a specified subclass of test examples by finding the most relevant training examples. Our method is therefore more transparent and can be used in parallel with ensemble methods if needed.

Threshold $\tau_{\mathrm{test}}(q)$ For every training example $x \in D_{train}$ , we have the latent density $p(x|\mathcal{D}_{\mathrm{B}\&\mathrm{I}})$ over the B&I class of the test set. With their order statistics $p_{(1)}(x|\mathcal{D}_{\mathrm{B}\&\mathrm{I}}), \ldots, p_{(|\mathcal{D}_{\mathrm{train}}|)}(x|\mathcal{D}_{\mathrm{B}\&\mathrm{I}})$ . Given quantile number q, our empirical quantile based threshold $\tau_{test}$ is chosen as

$$
\tau_ {\mathrm{test}} (q) \equiv p _ {(\lfloor q \cdot | \mathcal {D} _ {\mathrm{train}} | \rfloor)} (x | \mathcal {D} _ {\mathrm{B} \& \mathrm{I}}).
$$

During the inverse training time, we train our model to predict those B&I class of test examples only with the training data with higher density than $\tau_{\mathrm{test}}(q)$ . We experiment with different choices of q in the experiment (Figure 6 in Sec. 4.4).

# C Additional Experiments

# C.1 Categorization of Uncertainty under Different Thresholds

In the main text, we provide results with $\tau_{Bnd} = \tau_{IDM} = 0.8$ , which approximates the accuracy on validation set—as a natural choice. In this section, we vary these thresholds and show in Figure 7 that changing those thresholds does not significantly alter the conclusions drawn above.

Figure 8 looks more closely into the top 25% uncertain examples for each method, and the accuracy on each of the uncertainty classes. As expected, the accuracy of the B&I examples is always lower than that of the trusted class, meaning that those examples are most challenging for the classifier. And the accuracy of flagged classes are always lower than the other class, verifying the proposed categorization of different classes.

# C.2 Inverse Direction: More Results

In the main text, we show the results on improving prediction performance on the B&I class with training example filtering (On the Covtype, Digits dataset). More results on other classes of examples are provided in this section.

We experiment on three UCI datasets: Covtype, Digits, and Spam. And experiment with three classes we defined in this work:

![](images/d0df60ef64adbbc0e7a67022fa920b94e4132cb15e7aa06d38dfe58a6af243c1.jpg)

![](images/c209edb62ff8e1e507d9cccb6e9ca0f4b85d16f9953691ed414a3a68a81eda26.jpg)

<details>
<summary>line</summary>

| Top Uncertainty | BNNs  | MCD   | DE    |
| --------------- | ----- | ----- | ----- |
| 0               | 0     | 0     | 0     |
| 10              | 500   | 500   | 500   |
| 20              | 1500  | 1500  | 1500  |
| 30              | 2500  | 2500  | 2500  |
| 40              | 3500  | 3500  | 3500  |
| 50              | 4500  | 4500  | 4500  |
| 60              | 5000  | 5000  | 5000  |
| 70              | 5200  | 5200  | 5200  |
| 80              | 5300  | 5300  | 5300  |
| 90              | 5400  | 5400  | 5400  |
| 100             | 5500  | 5500  | 5500  |
</details>

![](images/def60009299afc55e2f61b0583afd5922e1639f95c0521c5fc3b880c569786d0.jpg)

<details>
<summary>line</summary>

| Top Uncertainty | BNNs  | MCD   | DE    |
| --------------- | ----- | ----- | ----- |
| 0               | 0     | 0     | 0     |
| 10              | 500   | 500   | 500   |
| 20              | 1500  | 1500  | 1500  |
| 30              | 2500  | 2500  | 2500  |
| 40              | 3500  | 3500  | 3500  |
| 50              | 4500  | 4500  | 4500  |
| 60              | 5000  | 5000  | 5000  |
| 70              | 5250  | 5250  | 5250  |
| 80              | 5500  | 5500  | 5500  |
| 90              | 5750  | 5750  | 5750  |
| 100             | 6000  | 6000  | 6000  |
</details>

![](images/65eadcd6441ffed7c26de40ad1ba7d8aa217759884109459df844265ca2a57c7.jpg)

<details>
<summary>line</summary>

| Top Uncertainty | BNNs  | MCD   | DE    |
| --------------- | ----- | ----- | ----- |
| 0               | 0     | 0     | 0     |
| 10              | 100   | 150   | 50    |
| 20              | 300   | 400   | 150   |
| 30              | 600   | 700   | 300   |
| 40              | 900   | 1000  | 500   |
| 50              | 1200  | 1300  | 700   |
| 60              | 1500  | 1600  | 900   |
| 70              | 1800  | 1900  | 1100  |
| 80              | 2200  | 2300  | 1400  |
| 90              | 2800  | 3200  | 2200  |
| 100             | 3200  | 3400  | 2800  |
</details>

(a) $\tau_{\mathrm{BD}}(0.5),\tau_{\mathrm{IDM}}(0.5)$   
![](images/3905e16fe4f202fcac1a4eac9ecc6d78597eedb8f625d6136c105fabdc6a3cd1.jpg)

![](images/b5f8655092133d7c08092bded80053cf8a3eb7b82c2d33b0400c191b40357049.jpg)

<details>
<summary>line</summary>

| Top Uncertainty | BNNs  | MCD   | DE    |
| --------------- | ----- | ----- | ----- |
| 0               | 0     | 0     | 0     |
| 10              | 500   | 500   | 500   |
| 20              | 1500  | 1500  | 1500  |
| 30              | 2500  | 2500  | 2500  |
| 40              | 3500  | 3500  | 3500  |
| 50              | 4000  | 4000  | 4000  |
| 60              | 4200  | 4200  | 4200  |
| 70              | 4300  | 4300  | 4300  |
| 80              | 4400  | 4400  | 4400  |
| 90              | 4500  | 4500  | 4500  |
</details>

![](images/a434eb02b177bb8dac1910aac5bf36b1237ad168bab2cd996e188918e601b079.jpg)

<details>
<summary>line</summary>

| Top Uncertainty | BNNs  | MCD   | DE    |
| --------------- | ----- | ----- | ----- |
| 0               | 0     | 0     | 0     |
| 10              | 500   | 500   | 500   |
| 20              | 1500  | 1500  | 1500  |
| 30              | 2500  | 2500  | 2500  |
| 40              | 3500  | 3500  | 3500  |
| 50              | 4000  | 4000  | 4000  |
| 60              | 4200  | 4200  | 4200  |
| 70              | 4300  | 4300  | 4300  |
| 80              | 4400  | 4400  | 4400  |
| 90              | 4500  | 4500  | 4500  |
</details>

![](images/60b92a4a053a05d974c22a198df5df45e044c29028f3ae74dd51d97443ae30a7.jpg)

<details>
<summary>line</summary>

| Top Uncertainty | BNNs  | MCD   | DE    |
| --------------- | ----- | ----- | ----- |
| 0               | 0     | 0     | 0     |
| 10              | 500   | 500   | 500   |
| 20              | 1000  | 1000  | 1000  |
| 30              | 1500  | 1500  | 1500  |
| 40              | 2000  | 2000  | 2000  |
| 50              | 2500  | 2500  | 2500  |
| 60              | 3000  | 3000  | 3000  |
| 70              | 3500  | 3500  | 3500  |
| 80              | 4000  | 4000  | 4000  |
| 90              | 4500  | 4500  | 4500  |
</details>

(b) $\tau_{\mathrm{BD}}(0.6),\tau_{\mathrm{IDM}}(0.6)$   
![](images/2886e90b1e6a7adf5fec8e45bf72096205f49d90d74c1a356cb91c4dc7f2cec4.jpg)

![](images/8068eb076987232b63590420c9a7533e54f38b93c36c1a567fa555a8b436c2dc.jpg)

<details>
<summary>line</summary>

| Top Uncertainty | BNNs  | MCD   | DE    |
| --------------- | ----- | ----- | ----- |
| 0               | 0     | 0     | 0     |
| 10              | 500   | 500   | 500   |
| 20              | 1000  | 1000  | 1000  |
| 30              | 1500  | 1500  | 1500  |
| 40              | 2000  | 2000  | 2000  |
| 50              | 2500  | 2500  | 2500  |
| 60              | 2750  | 2750  | 2750  |
| 70              | 2900  | 2900  | 2900  |
| 80              | 3000  | 3000  | 3000  |
| 90              | 3100  | 3100  | 3100  |
</details>

![](images/6cc4588024c1e4a8d2a35fedc83cfc52a29846729c67c85a636ce10d9281d80a.jpg)

<details>
<summary>line</summary>

| Top Uncertainty | BNNs  | MCD   | DE    |
| --------------- | ----- | ----- | ----- |
| 0               | 0     | 0     | 0     |
| 10              | 500   | 500   | 500   |
| 20              | 1000  | 1000  | 1000  |
| 30              | 1500  | 1500  | 1500  |
| 40              | 2000  | 2000  | 2000  |
| 50              | 2500  | 2500  | 2500  |
| 60              | 2750  | 2750  | 2750  |
| 70              | 2900  | 2900  | 2900  |
| 80              | 3000  | 3000  | 3000  |
| 90              | 3100  | 3100  | 3100  |
</details>

![](images/b532ca39d7249f6b1dec5f01210f12a97e581e54c968dac52e6abc46e09f0b41.jpg)

<details>
<summary>line</summary>

| Top Uncertainty | BNNs  | MCD   | DE    |
| --------------- | ----- | ----- | ----- |
| 0               | 0     | 0     | 0     |
| 10              | 500   | 500   | 500   |
| 20              | 1000  | 1000  | 1000  |
| 30              | 1500  | 1500  | 1500  |
| 40              | 2000  | 2000  | 2000  |
| 50              | 2500  | 2500  | 2500  |
| 60              | 3000  | 3000  | 3000  |
| 70              | 3500  | 3500  | 3500  |
| 80              | 4500  | 4500  | 4500  |
| 90              | 5500  | 5500  | 5500  |
| 100             | 6000  | 6000  | 6000  |
</details>

(c) $\tau_{\mathrm{BD}}(0.7),\tau_{\mathrm{IDM}}(0.7)$   
![](images/44aeb6e463fb8ce12df0d133ff1488272fe96c66f14fca8ed91ac93691944593.jpg)

<details>
<summary>line</summary>

| Top Uncertainty | BNNs | MCD  | DE   |
| --------------- | ---- | ---- | ---- |
| 0               | 0    | 0    | 0    |
| 10              | 600  | 400  | 800  |
| 20              | 700  | 500  | 900  |
| 30              | 750  | 550  | 950  |
| 40              | 800  | 600  | 1000 |
| 50              | 850  | 650  | 1000 |
| 60              | 900  | 700  | 1000 |
| 70              | 950  | 750  | 1000 |
| 80              | 1000 | 800  | 1000 |
| 90              | 1000 | 1000 | 1000 |
</details>

![](images/1e8ce75d65f0e856af5fe5b11e3bed32dddf3d1f73d992b159ca48d8298c13bc.jpg)

<details>
<summary>line</summary>

| Top Uncertainty | BNNs  | MCD   | DE    |
| --------------- | ----- | ----- | ----- |
| 0               | 0     | 0     | 0     |
| 10              | 500   | 500   | 500   |
| 20              | 1000  | 1000  | 1000  |
| 30              | 1500  | 1500  | 1500  |
| 40              | 1750  | 1750  | 1750  |
| 50              | 1900  | 1900  | 1900  |
| 60              | 2000  | 2000  | 2000  |
| 70              | 2050  | 2050  | 2050  |
| 80              | 2100  | 2100  | 2100  |
| 90              | 2150  | 2150  | 2150  |
</details>

![](images/784ad917f9edacefc7df5cd7f900c8c8dd1ef9ec96634f8d6bc9a5e295997de0.jpg)

<details>
<summary>line</summary>

| Top Uncertainty | BNNs  | MCD   | DE    |
| --------------- | ----- | ----- | ----- |
| 0               | 0     | 0     | 0     |
| 10              | 500   | 500   | 500   |
| 20              | 1000  | 1000  | 1000  |
| 30              | 1500  | 1500  | 1500  |
| 40              | 1750  | 1750  | 1750  |
| 50              | 2000  | 2000  | 2000  |
| 60              | 2100  | 2100  | 2100  |
| 70              | 2150  | 2150  | 2150  |
| 80              | 2200  | 2200  | 2200  |
| 90              | 2250  | 2250  | 2250  |
| 100             | 2300  | 2300  | 2300  |
</details>

![](images/4f286b1a38e62472b53a479ee9e25f0e318c9df4b27f47fc2936136f298dfa85.jpg)

<details>
<summary>line</summary>

| Top Uncertainty | BNNs  | MCD   | DE    |
| --------------- | ----- | ----- | ----- |
| 0               | 0     | 0     | 0     |
| 10              | 500   | 500   | 500   |
| 20              | 1500  | 1500  | 1500  |
| 30              | 2500  | 2500  | 2500  |
| 40              | 3500  | 3500  | 3500  |
| 50              | 4500  | 4500  | 4500  |
| 60              | 5500  | 5500  | 5500  |
| 70              | 6500  | 6500  | 6500  |
| 80              | 7500  | 7500  | 7500  |
| 90              | 8500  | 8500  | 8500  |
| 100             | 9500  | 9500  | 9500  |
</details>

(d) $\tau_{\mathrm{BD}}(0.8),\tau_{\mathrm{IDM}}(0.8)$   
![](images/958d4444989165b357226d65b4652a902cb0b698680fcdaf306dd8a4b6aad4f3.jpg)

![](images/00223751e3d45b5011251f7e1f3b1f99a5c7fed0b77b451ff58be8569cea68b4.jpg)

<details>
<summary>line</summary>

| Top Uncertainty | BNNs | MCD  | DE   |
| --------------- | ---- | ---- | ---- |
| 0               | 0    | 0    | 0    |
| 10              | 100  | 150  | 200  |
| 20              | 300  | 400  | 500  |
| 30              | 500  | 600  | 700  |
| 40              | 700  | 800  | 900  |
| 50              | 850  | 950  | 1000 |
| 60              | 950  | 1050 | 1050 |
| 70              | 1000 | 1100 | 1100 |
| 80              | 1050 | 1150 | 1150 |
| 90              | 1100 | 1200 | 1200 |
</details>

![](images/0d8bb15dcac0fe11a6f5e1582a1655ba4548b3ff170924b475391f66d789db28.jpg)

<details>
<summary>line</summary>

| Top Uncertainty | BNNs  | MCD   | DE    |
| --------------- | ----- | ----- | ----- |
| 0               | 0     | 0     | 0     |
| 10              | 100   | 100   | 100   |
| 20              | 300   | 350   | 350   |
| 30              | 500   | 600   | 600   |
| 40              | 700   | 800   | 800   |
| 50              | 850   | 950   | 950   |
| 60              | 950   | 1050  | 1050  |
| 70              | 1000  | 1100  | 1100  |
| 80              | 1050  | 1150  | 1150  |
| 90              | 1100  | 1200  | 1200  |
| 100             | 1150  | 1250  | 1250  |
</details>

![](images/e230c0f640e51ac486b11b2fdf505c36f5bfe08e7fd4f9514680aa756df5e9d7.jpg)

<details>
<summary>line</summary>

| Top Uncertainty | BNNs  | MCD   | DE    |
| --------------- | ----- | ----- | ----- |
| 0               | 0     | 0     | 0     |
| 10              | 500   | 500   | 500   |
| 20              | 1500  | 1500  | 1500  |
| 30              | 2500  | 2500  | 2500  |
| 40              | 3500  | 3500  | 3500  |
| 50              | 4500  | 4500  | 4500  |
| 60              | 5500  | 5500  | 5500  |
| 70              | 6500  | 6500  | 6500  |
| 80              | 7500  | 7500  | 7500  |
| 90              | 8500  | 8500  | 8500  |
| 100             | 9500  | 9500  | 9500  |
</details>

(e) $\tau_{\mathrm{BD}}(0.9),\tau_{\mathrm{IDM}}(0.9)$   
Figure 7: Experiments on different choices of thresholds. The results presented are based on 8 repeated runs with different random seeds.

1. B&I class (Figure 9). As we have discussed in our main text, the prediction accuracy on the B&I class are always the lowest among all classes. By training with filtered examples in $\mathcal{D}_{\mathrm{train}}$ rather than the entire training set, the B&I class of examples can be classified with a remarkably improved accuracy.   
2. Bnd class (Figure 10). This class of examples are located at boundaries in the latent space of validation set, but not necessarily have been misclassified. Therefore, their performance baseline (training with the entire $D_{train}$ ) is relatively high. The improvement is clear but not as much as on the other two classes.   
3. IDM class (Figure 11). For this class of examples, similar mistakes have been made in the validation set, yet those examples are not necessarily located in the boundaries—the misclassification may be caused by ambiguity in decision boundary, imperfectness of either the model or the dataset. The primal prediction accuracy on this class of examples is lower than the Bnd class but higher than the B&I class, training with filtered $D_{train}$ also clearly improve the performance on this class of examples.

![](images/59348b97edf6054a4c7afe4e9dc6d7ad7de30b24e4fcba1a0a6aa67af69abe2f.jpg)

<details>
<summary>pie</summary>

| Category | Percentage (%) |
| :--- | :--- |
| OOD | 0.0 |
| Bnd | 62.63 |
| IDM | 56.55 |
| B&I | 52.07 |
| Other | 64.93 |
</details>

(a) BNNs

![](images/8bf5e46bc222ff5e188c5c16771b559b789789e622386431676b30519627b018.jpg)

<details>
<summary>pie</summary>

| Category | Percentage (%) |
| :--- | :--- |
| OOD | 0.0 |
| Bnd | 62.17 |
| IDM | 69.02 |
| B&I | 61.41 |
| Other | 74.42 |
</details>

(b) MCD

![](images/b0304c678722be2a13885dfacfcf9f0d692a6cd4d2663bf190498d0d0da723bf.jpg)

<details>
<summary>pie</summary>

| Category | Accuracy (%) |
| :--- | :--- |
| OOD | 0.0 |
| Bnd | 59.3 |
| IDM | 65.05 |
| B&I | 58.15 |
| Other | 68.99 |
</details>

(c) DE   
Figure 8: The top $25\%$ uncertain examples identified by different methods. Legend of each figure provide the accuracy and proportion of each class. As the classifier can not make correct predictions on the OOD examples, it's always better for uncertainty estimators to flag more OOD examples.

Table 4: DAUC is not the only choice in identifying OOD examples. On the Dirty-MNIST dataset, DAUC, Outlier-AE and the IForest can identify most outliers in the test dataset. (Given threshold = 1.0 for those two benchmark methods). 

<table><tr><td>Method</td><td>Precision</td><td>Recall</td><td>F1-Score</td></tr><tr><td>DAUC</td><td> $1.0000 \pm 0.0000$ </td><td> $1.0000 \pm 0.0000$ </td><td> $1.0000 \pm 0.0000$ </td></tr><tr><td>Outlier-AE</td><td> $1.0000 \pm 0.0000$ </td><td> $1.0000 \pm 0.0000$ </td><td> $1.0000 \pm 0.0000$ </td></tr><tr><td>IForest [54]</td><td> $0.9998 \pm 0.0004$ </td><td> $1.0000 \pm 0.0000$ </td><td> $0.9999 \pm 0.0002$ </td></tr></table>

# C.3 Alternative Approach in Flagging OOD

As we have mentioned in the main text, although DAUC has a unified framework in understanding all three types of uncertainty the uncertainty caused by OOD examples can also be identified by off-the-shelf algorithms. We compare DAUC to two existing outlier detection methods in Table 4, where all methods achieve good performance on the Dirty-MNIST dataset. Our implementation is based on Alibi Detect [53].

![](images/cbcfdc55b7b92a6370d7672db7a163e47e43a3d18368c3634509879c5bbc38fb.jpg)

<details>
<summary>line</summary>

| Quantile q | w/ Filtered Training Set | Baseline |
| ---------- | ------------------------ | -------- |
| 0.00       | 0.25                     | 0.25     |
| 0.10       | 0.30                     | 0.25     |
| 0.20       | 0.32                     | 0.25     |
| 0.30       | 0.34                     | 0.25     |
| 0.40       | 0.36                     | 0.25     |
| 0.50       | 0.38                     | 0.25     |
| 0.60       | 0.40                     | 0.25     |
| 0.70       | 0.42                     | 0.25     |
| 0.80       | 0.44                     | 0.25     |
| 0.90       | 0.43                     | 0.25     |
| 0.95       | 0.41                     | 0.25     |
</details>

(a) Covtype

![](images/3a58dfa80b4ca7e1d452e76f28a9acb96cbc315b2a009b3d3b53501666008722.jpg)

<details>
<summary>line</summary>

| Quantile q | w/ Filtered Training Set | Baseline |
| ---------- | ------------------------ | -------- |
| 0.00       | 0.0                      | 0.0      |
| 0.10       | 0.2                      | 0.0      |
| 0.20       | 0.3                      | 0.0      |
| 0.30       | 0.4                      | 0.0      |
| 0.40       | 0.5                      | 0.0      |
| 0.50       | 0.6                      | 0.0      |
| 0.60       | 0.7                      | 0.0      |
| 0.70       | 0.75                     | 0.0      |
| 0.80       | 0.8                      | 0.0      |
| 0.90       | 0.8                      | 0.0      |
| 0.95       | 0.75                     | 0.0      |
</details>

(b) Digits

![](images/c6c010858d1d731ceba6203a7834a40d5944ed4460c87606d87a15ee8a897c4c.jpg)

<details>
<summary>line</summary>

| Quantile q | w/ Filtered Training Set | Baseline |
| ---------- | ------------------------ | -------- |
| 0.00       | 0.60                     | 0.60     |
| 0.10       | 0.70                     | 0.60     |
| 0.20       | 0.72                     | 0.60     |
| 0.30       | 0.71                     | 0.60     |
| 0.40       | 0.71                     | 0.60     |
| 0.50       | 0.71                     | 0.60     |
| 0.60       | 0.73                     | 0.60     |
| 0.70       | 0.75                     | 0.60     |
| 0.80       | 0.77                     | 0.60     |
| 0.90       | 0.75                     | 0.60     |
| 0.95       | 0.72                     | 0.60     |
</details>

(c) Spam   
Figure 9: Experiments on the B&I class (reported in the main text). The results presented are based on 10 repeated runs with different random seeds.

# C.4 Experiments on Dirty-CIFAR-10

Dataset Discription In this experiment, we introduce a revised version of the CIFAR-10 dataset to test DAUC's scalability. Similar to the Dirty-MNIST dataset [41], we use linear combinations of the latent representation to construct the “boundary” class. In the original CIFAR-10 Dataset, each of the 10 classes of objects has 6000 training examples. We split the training set into training set (40%), validation set (40%) and test set (20%). To verify the performance of DAUC in detecting OOD examples, we randomly remove one of those 10 classes (denoted with class-i) during training

![](images/85f17db98bd008646914498e1d718eedba855a7a5949ee6e55ab01e4dbf7efd8.jpg)

<details>
<summary>line</summary>

| Quantile q | w/ Filtered Training Set | Baseline |
| ---------- | ------------------------ | -------- |
| 0.00       | 0.46                     | 0.46     |
| 0.10       | 0.47                     | 0.46     |
| 0.20       | 0.48                     | 0.46     |
| 0.30       | 0.49                     | 0.46     |
| 0.40       | 0.50                     | 0.46     |
| 0.50       | 0.51                     | 0.46     |
| 0.60       | 0.52                     | 0.46     |
| 0.70       | 0.53                     | 0.46     |
| 0.80       | 0.54                     | 0.46     |
| 0.90       | 0.53                     | 0.46     |
| 0.95       | 0.52                     | 0.46     |
</details>

(a) Covtype

![](images/657e5c74d93e486afa5564ac3f935b5196e6231ab0400f4644601f3a3c64584b.jpg)

<details>
<summary>line</summary>

| Quantile q | w/ Filtered Training Set | Baseline |
| ---------- | ------------------------ | -------- |
| 0.00       | 0.1                      | 0.1      |
| 0.10       | 0.2                      | 0.1      |
| 0.20       | 0.3                      | 0.1      |
| 0.30       | 0.4                      | 0.1      |
| 0.40       | 0.5                      | 0.1      |
| 0.50       | 0.6                      | 0.1      |
| 0.60       | 0.7                      | 0.1      |
| 0.70       | 0.75                     | 0.1      |
| 0.80       | 0.7                      | 0.1      |
| 0.90       | 0.6                      | 0.1      |
| 0.95       | 0.5                      | 0.1      |
</details>

(b) Digits

![](images/60d9eff57fba1cff45510f1b30dc6e3ea23226e783436eccf75c4483ff145f43.jpg)

<details>
<summary>line</summary>

| Quantile q | w/ Filtered Training Set | Baseline |
| ---------- | ------------------------ | -------- |
| 0.00       | 0.78                     | 0.78     |
| 0.10       | 0.80                     | 0.78     |
| 0.20       | 0.81                     | 0.78     |
| 0.30       | 0.81                     | 0.78     |
| 0.40       | 0.82                     | 0.78     |
| 0.50       | 0.82                     | 0.78     |
| 0.60       | 0.83                     | 0.78     |
| 0.70       | 0.83                     | 0.78     |
| 0.80       | 0.84                     | 0.78     |
| 0.90       | 0.83                     | 0.78     |
| 0.95       | 0.81                     | 0.78     |
</details>

(c) Spam

Figure 10: Experiments on the Bnd class. The results presented are based on 10 repeated runs with different random seeds.   
![](images/ce2f8aba3b1918d4c27d7f49be35f470c9d52e37179531037c2c9da83104e541.jpg)

<details>
<summary>line</summary>

| Quantile q | w/ Filtered Training Set | Baseline |
| ---------- | ------------------------ | -------- |
| 0.00       | 0.38                     | 0.38     |
| 0.10       | 0.40                     | 0.38     |
| 0.20       | 0.42                     | 0.38     |
| 0.30       | 0.44                     | 0.38     |
| 0.40       | 0.46                     | 0.38     |
| 0.50       | 0.47                     | 0.38     |
| 0.60       | 0.48                     | 0.38     |
| 0.70       | 0.47                     | 0.38     |
| 0.80       | 0.46                     | 0.38     |
| 0.90       | 0.45                     | 0.38     |
| 0.95       | 0.46                     | 0.38     |
</details>

(a) Covtype

![](images/b93b384032823e42d78c4b9d654708a347e4fec44a48bc5ebf92cdb5aae27e1a.jpg)

<details>
<summary>line</summary>

| Quantile q | w/ Filtered Training Set | Baseline |
| ---------- | ------------------------ | -------- |
| 0.00       | 0.0                      | 0.0      |
| 0.10       | 0.2                      | 0.0      |
| 0.20       | 0.3                      | 0.0      |
| 0.30       | 0.4                      | 0.0      |
| 0.40       | 0.5                      | 0.0      |
| 0.50       | 0.6                      | 0.0      |
| 0.60       | 0.7                      | 0.0      |
| 0.70       | 0.75                     | 0.0      |
| 0.80       | 0.78                     | 0.0      |
| 0.90       | 0.77                     | 0.0      |
| 0.95       | 0.76                     | 0.0      |
</details>

(b) Digits

![](images/c19b2ecc607179dad939236ce76580589001183daa3547f0b2b5f3daa967a2cd.jpg)

<details>
<summary>line</summary>

| Quantile q | w/ Filtered Training Set | Baseline |
| ---------- | ------------------------ | -------- |
| 0.00       | 0.68                     | 0.68     |
| 0.10       | 0.72                     | 0.68     |
| 0.20       | 0.74                     | 0.68     |
| 0.30       | 0.75                     | 0.68     |
| 0.40       | 0.76                     | 0.68     |
| 0.50       | 0.77                     | 0.68     |
| 0.60       | 0.78                     | 0.68     |
| 0.70       | 0.79                     | 0.68     |
| 0.80       | 0.80                     | 0.68     |
| 0.90       | 0.79                     | 0.68     |
| 0.95       | 0.78                     | 0.68     |
</details>

(c) Spam   
Figure 11: Experiments on the IDM class. The results presented are based on 10 repeated runs with different random seeds.

and manually concatenate OOD examples with the test dataset, with label i. In our experiment, we use 1000 MNIST digits as the OOD examples, with zero-padding to make those digits share the same input shape as the CIFAR-10 images. Combining those boundary examples, OOD examples and the vanilla CIFAR-10 examples, we get a new benchmark, dubbed as Dirty-CIFAR-10, for quantitative evaluation of DAUC.

Quantify the performance of DAUC on Dirty-CIFAR-10 Quantitatively, we evaluate the performance of DAUC in categorizing all three classes of uncertain examples. Results of averaged performance and standard deviations based on 8 repeated runs are provided in Table 5.

Table 5: Quantitative results on the Dirty-CIFAR-10 dataset. DAUC scales well and is able to categorize all three classes of uncertain examples. Results presented in the table are based on 8 repeated runs with different random seeds. 

<table><tr><td>Category</td><td>Precision</td><td>Recall</td><td>F1-Score</td></tr><tr><td>OOD</td><td>0.986 ± 0.003</td><td>0.959 ± 0.052</td><td>0.972 ± 0.027</td></tr><tr><td>Bnd</td><td>0.813 ± 0.002</td><td>0.975 ± 0.000</td><td>0.887 ± 0.001</td></tr><tr><td>IDM</td><td>0.688 ± 0.041</td><td>0.724 ± 0.017</td><td>0.705 ± 0.027</td></tr></table>

Categorize Uncertain Predictions on Dirty-CIFAR-10 Similar to Sec. 4.3 and Figure 5, we can categorize uncertain examples flagged by BNNs, MCD, and DE using DAUC—see Figure 12. We find that in the experiment with CIFAR-10, DE tends to discover more OOD examples as top uncertain examples. Differently, although BNNs flag fewer OOD examples as top-uncertain, they continuously discover those OOD examples and are able to find most of them for the top 50% uncertainty. On the contrary, MCD performs the worst among all three methods, similar to the result is drawn from the DMNIST experiment. On the other hand, while BNN is good at identifying OOD examples, it flags less uncertain examples in the Bnd and IDM classes. DE is the most apt at flagging both Bnd and

IDM examples and categorizes far fewer examples into the Other class. These observations are well aligned with the experiment results we had with DMNIST in Sec. 4.3, showing the scalability of DAUC to large-scale image datasets.

![](images/179029fb6766225b1d6a3d82b73027068001e20611ac614d37fe31d2bec6532b.jpg)

![](images/799d64fbfff94e527cb179a7543b63196046ef51aa98ac7b27bb1b817f89ddca.jpg)

<details>
<summary>line</summary>

| Top Uncertainty | BNNs  | MCD   | DE    |
| --------------- | ----- | ----- | ----- |
| 0               | 0     | 0     | 0     |
| 10              | 500   | 500   | 500   |
| 20              | 1000  | 1000  | 1000  |
| 30              | 1500  | 1500  | 1500  |
| 40              | 2000  | 2000  | 2000  |
| 50              | 2500  | 2500  | 2500  |
| 60              | 3000  | 3000  | 3000  |
| 70              | 3500  | 3500  | 3500  |
| 80              | 4000  | 4000  | 4000  |
| 90              | 4500  | 4500  | 4500  |
| 100             | 5000  | 5000  | 5000  |
</details>

![](images/90c9e6c60d2e8996d3b980e2355bd5060434629a84e75d0894f889c90e9ef848.jpg)

<details>
<summary>line</summary>

| Top Uncertainty | BNNs  | MCD   | DE    |
| --------------- | ----- | ----- | ----- |
| 0               | 0     | 0     | 0     |
| 10              | 500   | 500   | 500   |
| 20              | 1000  | 1000  | 1000  |
| 30              | 1500  | 1500  | 1500  |
| 40              | 1750  | 1750  | 1750  |
| 50              | 2000  | 2000  | 2000  |
| 60              | 2100  | 2100  | 2100  |
| 70              | 2150  | 2150  | 2150  |
| 80              | 2200  | 2200  | 2200  |
| 90              | 2250  | 2250  | 2250  |
</details>

![](images/f7e28665e6c7725af8a747e515ef226b3deb31824bafdeb5a76d57d43b93edb3.jpg)

<details>
<summary>line</summary>

| Top Uncertainty | BNNs  | MCD   | DE    |
| --------------- | ----- | ----- | ----- |
| 0               | 0     | 0     | 0     |
| 10              | 500   | 400   | 300   |
| 20              | 1500  | 1400  | 1200  |
| 30              | 2500  | 2400  | 2100  |
| 40              | 3500  | 3400  | 3100  |
| 50              | 4500  | 4400  | 4100  |
| 60              | 5500  | 5400  | 5100  |
| 70              | 6500  | 6400  | 6100  |
| 80              | 7500  | 7400  | 7100  |
| 90              | 8500  | 8400  | 8100  |
</details>

![](images/3055dab251db48532de406c5e4e87578d6e00ce16e7d30705543f61e3098967e.jpg)

<details>
<summary>line</summary>

| Top Uncertainty | BNNs  | MCD   | DE    |
| --------------- | ----- | ----- | ----- |
| 0               | 0.55  | 0.50  | 0.45  |
| 10              | 0.40  | 0.35  | 0.30  |
| 20              | 0.25  | 0.20  | 0.15  |
| 30              | 0.18  | 0.15  | 0.12  |
| 40              | 0.15  | 0.12  | 0.10  |
| 50              | 0.13  | 0.10  | 0.08  |
| 60              | 0.12  | 0.09  | 0.07  |
| 70              | 0.11  | 0.08  | 0.06  |
| 80              | 0.10  | 0.07  | 0.05  |
| 90              | 0.09  | 0.06  | 0.04  |
</details>

![](images/87a074718d3f3e7c6fd32592f2318274250abb4fcb5af32b625b3a8015778308.jpg)

<details>
<summary>line</summary>

| Top Uncertainty | BNNs  | MCD   | DE    |
| --------------- | ----- | ----- | ----- |
| 0               | 0.05  | 0.15  | 0.25  |
| 10              | 0.10  | 0.20  | 0.24  |
| 20              | 0.12  | 0.21  | 0.23  |
| 30              | 0.13  | 0.21  | 0.22  |
| 40              | 0.14  | 0.20  | 0.21  |
| 50              | 0.15  | 0.19  | 0.20  |
| 60              | 0.16  | 0.18  | 0.19  |
| 70              | 0.17  | 0.17  | 0.18  |
| 80              | 0.17  | 0.16  | 0.17  |
| 90              | 0.17  | 0.16  | 0.16  |
| 100             | 0.17  | 0.16  | 0.16  |
</details>

![](images/b8bfd0d920f9420eb89531721d301af01ac33e6cd660510222ccea74cd7e635c.jpg)

<details>
<summary>line</summary>

| Top Uncertainty | BNNs  | MCD   | DE    |
| --------------- | ----- | ----- | ----- |
| 0               | 0.125 | 0.125 | 0.125 |
| 10              | 0.150 | 0.160 | 0.180 |
| 20              | 0.175 | 0.190 | 0.210 |
| 30              | 0.190 | 0.205 | 0.225 |
| 40              | 0.200 | 0.215 | 0.235 |
| 50              | 0.210 | 0.220 | 0.240 |
| 60              | 0.215 | 0.225 | 0.245 |
| 70              | 0.220 | 0.230 | 0.250 |
| 80              | 0.225 | 0.235 | 0.255 |
| 90              | 0.230 | 0.240 | 0.260 |
</details>

![](images/755f78b9ce81b71dfd01590f9f027fdf93cdc614b29adcf42ec109e353a73864.jpg)

<details>
<summary>line</summary>

| Top Uncertainty | BNNs  | MCD   | DE    |
| --------------- | ----- | ----- | ----- |
| 0               | 0.35  | 0.25  | 0.15  |
| 10              | 0.45  | 0.40  | 0.25  |
| 20              | 0.48  | 0.45  | 0.35  |
| 30              | 0.50  | 0.48  | 0.40  |
| 40              | 0.52  | 0.50  | 0.45  |
| 50              | 0.53  | 0.52  | 0.48  |
| 60              | 0.54  | 0.53  | 0.50  |
| 70              | 0.55  | 0.54  | 0.52  |
| 80              | 0.56  | 0.55  | 0.54  |
| 90              | 0.57  | 0.56  | 0.56  |
</details>

Figure 12: Experiments on the CIFAR-10 dataset. Results of applying DAUC in categorizing different uncertainty estimation methods. First row: comparisons of the numbers in different classes of examples. Second row: comparisons on the proportion of different classes of flagged examples to the total number of identified uncertain examples. Different methods tend to identify different certain types of uncertain examples. The results presented are based on 8 repeated runs with different random seeds.

# D Implementation Details

# D.1 Code

Our code is available at https://github.com/vanderschaarlab/DAUC.

# D.2 Hyperparameters

# D.2.1 Bandwidth

In our experiments, we use (z-score) normalized latent representations and bandwidth 1.0. In the inverse direction, as the sample sizes are much smaller, a bandwidth of 0.01 is used as the recommended setting. There is a vast body of research on selecting a good bandwidth for Kernel Density Estimation models [55–57] and using these to adjust DAUC's bandwidth to a more informed choice may further improve performance.

# D.3 Inverse Direction: Quantile Threshold q

As depicted in Appendix A.2, a natural choice of q is to use the validation accuracy. We use this heuristic approach in our experiments for the inverse direction.

# D.4 Model Structure

In our experiments, we implement MCD and DE with 3-layer-CNNs with ReLU activation. Our experiments on BNNs are based on the IBM UQ360 software [58]. More details of the convolutional network structure are provided in Table 6.

# D.5 Implementation of Kernel Density Estimation and Repeat Runs

Our implementation of KDE models is based on sklearn's KDE package [59]. Gaussian kernels are used as default settings. We experiment with 8-10 random seeds and report the averaged results and standard deviations. In our experiments, we find using different kernels in density estimation provides highly correlated scores. We calculate the Spearman's $\rho$ correlation between scores DAUC gets over 5 runs with Gaussian, Tophat, and Exponential kernels under the same bandwidth. Changing the kernel brings highly correlated scores (all above 0.86) for DAUC and, hence, has a minor impact on

Table 6: Network Structure 

<table><tr><td>Layer</td><td>Unit</td><td>Activation</td><td>Pooling</td></tr><tr><td>Conv 1</td><td> $(1, 32, 3, 1, 1)$ </td><td>ReLU()</td><td>MaxPool2d(2)</td></tr><tr><td>Conv 2</td><td> $(32, 64, 3, 1, 1)$ </td><td>ReLU()</td><td>MaxPool2d(2)</td></tr><tr><td>Conv 3</td><td> $(64, 64, 3, 1, 1)$ </td><td>ReLU()</td><td>MaxPool2d(2)</td></tr><tr><td>FC</td><td> $(64 \times 3 \times 3, 40)$ </td><td>ReLU()</td><td>-</td></tr><tr><td>Out</td><td> $(40, N_{\text{Class}})$ </td><td>SoftMax()</td><td>-</td></tr></table>

DAUC's performance. We preferred KDE since the latent representation is relatively low-dimensional. We found that a low-dim latent space (e.g., 10) works well for all experiments (including CIFAR-10).

# D.6 Hardware

All results reported in our paper are conducted with a machine with 8 Tesla K80 GPUs and 32 Intel(R) E5-2640 CPUs. The computational cost is mainly in density estimation, and for low-dim representation space, such an estimation can be efficient: running time for DAUC on the Dirty-MNIST dataset with KDE is approximately 2 hours.

# Assumptions and Limitations

In this work, we introduced the confusion density matrix that allows us to categorize suspicious examples at testing time. Crucially, this density matrix relies on kernel density estimations in the latent space H associated with the model f through Assumption 1. We note this assumption generally holds for most modern uncertainty estimation methods.

While the core contribution of this work is to introduce the concept of confusion density matrix for uncertainty categorization, the density estimators leveraged in the latent space can be further improved. We leave this to future work.

# Broader Impact

While previous works on uncertainty quantification (UQ) focused on the discovery of uncertain examples, in this work, we propose a practical framework for categorizing uncertain examples that are flagged by UQ methods. We demonstrated that such a categorization can be used for UQ method selection — different UQ methods are good at figuring out different uncertainty sources. Moreover, we show that for the inverse direction, uncertainty categorization can improve model performance.

With our proposed framework, many real-world application scenarios can be potentially benefit. e.g., in Healthcare, a patient marked as uncertain that categorized as OOD — preferably identified by Deep Ensemble, as we have shown — should be treated carefully when applying regular medical experience; and an uncertain case marked as IDM — preferably identified by MCD — can be carefully compared with previous failure cases for a tailored and individualized medication.