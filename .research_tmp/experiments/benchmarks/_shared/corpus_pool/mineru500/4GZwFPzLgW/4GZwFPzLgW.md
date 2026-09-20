# Contextures: Representations from Contexts

Runtian Zhai $^{1}$ , Kai Yang $^{2}$ , Che-Ping Tsai $^{1}$ , Burak Varicı $^{1}$ , Zico Kolter $^{1}$ , Pradeep Ravikumar $^{1}$

$^{1}$ Carnegie Mellon University $^{2}$ Peking University

rzhai@alumni.cmu.edu, pradeepr@cs.cmu.edu

June 11, 2025

# Abstract

Despite the empirical success of foundation models, we do not have a systematic characterization of the representations that these models learn. In this paper, we establish the contexture theory. It shows that a large class of representation learning methods can be characterized as learning from the association between the input and a context variable. Specifically, we show that many popular methods aim to approximate the top-d singular functions of the expectation operator induced by the context, in which case we say that the representation learns the contexture. We demonstrate the generality of the contexture theory by proving that representation learning within various learning paradigms—supervised, self-supervised, and manifold learning—can all be studied from such a perspective. We also prove that the representations that learn the contexture are optimal on those tasks that are compatible with the context. One important implication of the contexture theory is that once the model is large enough to approximate the top singular functions, further scaling up the model size yields diminishing returns. Therefore, scaling is not all we need, and further improvement requires better contexts. To this end, we study how to evaluate the usefulness of a context without knowing the downstream tasks. We propose a metric and show by experiments that it correlates well with the actual performance of the encoder on many real datasets.

# 1 Introduction

Representation learning underpins the modern deep learning revolution, leading up to the remarkable recent successes of foundation models $[BHA^{+}21]$ . But a critical question that has remained unanswered to a satisfactory extent is: why are these models learning anything useful, or perhaps even what representations are these models learning? Unlike classical statistical learning theory, where there is no mystery regarding the statistical estimand, the very target itself is unclear in representation learning. For example, what are the representations that BERT $[DCLT19]$ —trained to do cloze tests—is learning, and why are they useful in understanding the sentiment of user reviews on Netflix? What representations do deep neural networks learn, and why are they useful if they cause neural collapse $[PHD20]$ , where deep representations could collapse to a few clusters? Do the many different self-supervised learning methods $[BIS^{+}23]$ all learn similar or disparate representations?

The responses to these questions are often muddled. Many analyses are conflated with the mystery of deep learning generalization—the ability of large neural networks to learn function approximations that generalize to unseen points. However, this is a very different problem from the mechanism of representation learning. Our focus is on what representation learning (also known as “pretraining” in the context of foundation models) aims to capture, and why it can be applied to tasks completely different from the objectives used to train the representations. Another way this question is muddled is by recourse to scaling. A popular viewpoint argues

that even if the encoder performs poorly on one task, increasing the model size while keeping everything else the same could allow better performance to “emerge” $[WTB^{+}22]$ . However, substantial evidence suggests that certain abilities cannot emerge solely from scaling. Additional training signals, such as alignment $[OWJ^{+}22]$ , are necessary.

The above questions are naturally interesting to learning theorists, but why should the broader machine learning community care about understanding the mechanism of representation learning, if empirical success seems to be always achievable with existing approaches by scaling up the model size, an empirical observation known as scaling laws $[KMH^{+}20]$ ? This is because sustainable success or progress is not always guaranteed. Ilya Sutskever recently remarked that “pretraining as we know it will end” $[Sut24]$ , largely because the current pretraining paradigm is producing diminishing returns. Understanding what representations are learned by foundation models is crucial for designing future generations of pretraining methods, and this is how this field can make scientific progress.

In this work, we establish the contexture theory, which provides a unified lens for inspecting a large class of representation learning methods. The central argument of the contexture theory is that representations are learned from the association between the input X and a context variable A. This framework is general enough to encompass a wide variety of learning paradigms, as we demonstrate in Section 3.

Now, suppose we are given a context variable A along with X, how should we learn the representation? In Section 4 we prove that the optimal method is to approximate the top singular functions of the expectation operator induced by the context, in which case we say that the encoder “learns the contexture”. Such an encoder is optimal as long as the task is known to be compatible with the context, and in Section 4 we define a quantitative measurement of such compatibility.

Our theory implies that the main consequence of making the model large is that the learned representation space will be brought closer to the space spanned by the top-d singular functions of the expectation operator, which we empirically verify in Section 4.2. Once the two spaces are close enough, further scaling will yield diminishing returns. We envision that future breakthroughs in pretraining require context scaling, where better contexts are learned from data, instead of heuristics.

Taking a first step towards context scaling, in Section 5 we study how to evaluate contexts. This is a prerequisite because if we cannot even determine which contexts are good, then there will be no way to create better contexts. The key takeaway from our analysis is that for a context to be useful (meaning that it can lead to good representations), the association between X and A should be moderate—neither too strong nor too weak. For example, if A = X, then their association is the strongest; if A is independent of X, then their association is the weakest. However, neither context is useful because A does not provide additional information about X. In Section 5.2, we propose a quantitative measurement of context usefulness that can be efficiently estimated and does not require knowledge of the downstream task. We also demonstrate empirically that the proposed metric correlates with the performance of the encoder on many real datasets.

In one sentence, the key contribution of this work is clarity on the target of a large class of representation learning methods—the singular functions of the expectation operator. However, we do not discuss the numerical aspect of approximating these functions, which requires an expressive model architecture and a good optimizer, and such analyses are left to future work.

# 1.1 Examples of Representation Learning Methods

Supervised learning is the simplest way to learn representations. For example, the representations of neural networks pretrained on ImageNet $[RDS^{+}15]$ were very popular in the early days of the deep learning boom [HAE16]. Specifically, one uses the output of an intermediate layer, typically the one before the last linear layer, as the representation. However, it has never been fully explained why the penultimate layer works so well on tasks very different from the pretraining task.

Self-supervised learning (SSL) is currently the most common way of learning representations. Two of the most widely used SSL categories are multi-view learning and denoising autoencoders. Multi-view learning

includes contrastive learning [OLV18, CKNH20] and non-contrastive learning $[GSA^{+}20, ZJM^{+}21, BPL22]$ . Denoising autoencoders have wide applications, including language $[DCLT19, RWC^{+}19]$ , vision $[HCX^{+}22]$ , videos $[GWDL23]$ , and more.

Manifold learning is a classical method that aims to capture the geometry of the data. Examples such as locally linear embedding (LLE) [RS00] and Laplacian eigenmaps [BN03] formulate manifold learning as node representation learning on a graph, where connected nodes should have similar embeddings.

# 1.2 Related Work

Understanding what representations an encoder learns has long been a hot research topic in machine learning. In the pre-deep-learning-boom era, early work on manifold learning revealed the connection between representation learning and approximating the top eigenfunctions of a kernel $[BN03, BDR^{+}04, CL06]$ . Moreover, using the eigenvectors of the graph Laplacian as node representations on graphs was a classical technique in graph applications $[BN02, ZGL03]$ .

Although deep learning has achieved phenomenal success, it is not as theoretically driven as traditional methods, which makes it difficult to analyze what representations it learns. Most early work experimentally studied the representations of neural networks with visualization tools [SVZ14, ZF14], or data manipulation $[ZBH^{+}17]$ . These experimental tools are still widely used today to analyze large language models [YN22, NCL $^{+}$ 23, NLW23].

On the theoretical understanding of representations, two lines of work are closely related to this paper. The first line studies representation alignment [KNLH19, HCWI24, FPM+24]. These papers mainly focus on comparing between two representations, while this work aims to evaluate a single representation and the context in which it is trained. Representation similarity has also been studied in neuroscience [KMB08]. The second line develops the spectral theory of self-supervised learning (SSL). SSL has achieved remarkable success in recent years [RWC+19, CKNH20, ZJM+21, BPL22, HCX+22, BHX+22, ODM+23, ADM+23]. See the SSL cookbook by [BIS+23] for a summary of these methods. [HWGM21, JHM23] related contrastive learning to the spectrum of the augmentation graph and the positive-pair kernel. Then, [ZLR+24] extended the spectral theory to all SSL methods, not just contrastive learning. This paper extends the spectral theory to an even broader scope than [ZLR+24]. Other theoretical work on SSL studies its training dynamics [DLS22, JVLT22, Tia22] and builds its connection to information theory [AS18, BL22, SZBK+23].

Another line of work for characterizing the representations aims to learn disentangled or causally associated representations $[HAP^{+}18, SLB^{+}21]$ . It is shown that such representations can be provably recovered, provided sufficient variables are present in the environments, either through auxiliary variables or interventions $[KKMH20, VAST24, BRR^{+}23, YXL^{+}24]$ . Some of these results further require stringent parametric assumptions.

Prior work has developed quite different theoretical frameworks for different pretraining methods, and each framework is not applicable to other settings. In this work, we provide a unified framework that spans the works discussed above and more—manifold learning, SSL based on augmentation, supervised learning, graph representation learning, etc. It further extends the spectral theory in $[ZLR^{+}24]$ to other paradigms beyond SSL.

# 2 Definitions and Examples

Let $\mathcal{X}$ be the input space. The goal of representation learning is to learn a feature encoder $\Phi : \mathcal{X} \to \mathbb{R}^d$ . We call $\Phi(x)$ the embedding of $x$ , and $d$ the embedding dimension. Let $P_{\mathcal{X}}$ be the data distribution. Throughout this work, we assume $P_{\mathcal{X}}$ to be fixed.

The central argument of the contexture theory is that representations are learned from the association between the input $X \in X$ and a context variable $A \in A$ . A is called the context space. Let $P^{+}(x, a)$ be the joint distribution of X and A, with marginal distributions $P_{X}$ and $P_{A}$ . Let $L^{2}(P_{X})$ be the $L^{2}$ function

space w.r.t. $P_{X}$ , with inner product $\langle f_{1}, f_{2} \rangle_{P_{X}} = \mathbb{E}_{X \sim P_{X}} [f_{1}(X) f_{2}(X)]$ and norm $\|f\|_{P_{X}} = \sqrt{\langle f, f \rangle_{P_{X}}}$ . Define $L^{2}(P_{A}), \langle \cdot, \cdot \rangle_{P_{A}}, \| \cdot \|_{P_{A}}$ for $P_{A}$ similarly.

# 2.1 Examples of Contexts

We first introduce some commonly used contexts, their corresponding $P^{+}$ , and how $P^{+}$ is provided in practice.

1. Labels are a common type of context in machine learning. They can take different forms, such as discrete categories in classification, continuous values in regression, or structured outputs like text sequences in image captioning. Labels may be obtained from human annotators or in pseudo-forms, such as clusters or teacher models. Typically, labels are provided as compatible pairs sampled from the joint distribution $P^{+}(x,a)$ .   
2. Random transformations generate different views of the same data point. Common transformations include adding random noise to inputs, as seen in diffusion models and denoising autoencoders, or randomly corrupting/masking inputs, as in SimCLR and masked autoencoder. These transformations are typically defined by domain experts and are specified through a predefined conditional distribution $P^{+}(a \mid x)$ .   
3. Graphs provide locality information about the inputs. The edge values are typically given, which represent the similarity between two inputs. We can have a continuous space version of such a graph kernel to capture local geometry, which in the limit can be related to differential operators such as the Laplace-Beltrami operator on a manifold. In this case, we have $\mathcal{A} = \mathcal{X}$ with the conditional distribution $P^{+}(a\mid x)$ proportional to the edge values between $x$ and $a$ . Please refer to Section 3.3 for a more detailed construction of $P^{+}(a\mid x)$ .   
4. Stochastic Features are potentially stochastic predefined or pretrained mappings from $\mathcal{X}$ to a vector space, which we denote by $a = \Omega(x)$ . This encompasses teacher models that provide stochastic pretrained feature encoders. In contrast to previous instances, here $P^{+}(a \mid x)$ is directly described via a reparameterization or structural equation for $a$ in terms of $x$ .

# 2.2 Induced Kernels and Expectation Operator

The joint distribution $P^{+}$ fully determines the association between X and A. It induces the positive-pair kernel [JHM23] and the dual kernel $[ZLR^{+}24]$ defined as follows.

Definition 1. The positive-pair kernel $k_A^+$ and its dual kernel $k_X^+$ are defined as

$$
k _ {A} ^ {+} (a, a ^ {\prime}) = \frac {P ^ {+} (a , a ^ {\prime})}{P _ {\mathcal {A}} (a) P _ {\mathcal {A}} (a ^ {\prime})} = \frac {\int P ^ {+} (a \mid x) P ^ {+} (a ^ {\prime} \mid x) d P _ {\mathcal {X}} (x)}{P _ {\mathcal {A}} (a) P _ {\mathcal {A}} (a ^ {\prime})};
$$

$$
k _ {X} ^ {+} (x, x ^ {\prime}) = \frac {P ^ {+} (x , x ^ {\prime})}{P _ {\mathcal {X}} (x) P _ {\mathcal {X}} (x ^ {\prime})} = \frac {\int P ^ {+} (x \mid a) P ^ {+} (x ^ {\prime} \mid a) d P _ {\mathcal {A}} (a)}{P _ {\mathcal {X}} (x) P _ {\mathcal {X}} (x ^ {\prime})}.
$$

These two kernels are density ratios between joints and marginals for $A$ and $X$ , respectively. The kernels captures how more likely $(a, a')$ or $(x, x')$ appear together than independently given $P^{+}$ . $P^{+}$ also induces the following expectation operator. Intuitively, given a function $g \in L^{2}(P_{\mathcal{A}})$ , the expectation operator computes the expectation of $g(A)$ conditioned on any given $x$ .

Definition 2. The expectation operator $T_{P+}: L^2(P_A) \to L^2(P_X)$ is defined as for all $g \in L^2(P_A)$ ,

$$
(T _ {P ^ {+}} g) (x) = \int g (a) P ^ {+} (a \mid x) d a = \mathbb {E} [ g (A) \mid x ].
$$

Its adjoint operator $T_{P^+}^* : L^2(P_{\mathcal{X}}) \to L^2(P_{\mathcal{A}})$ , which satisfies $\langle f, T_{P^+}g \rangle_{P_{\mathcal{X}}} = \langle T_{P^+}^* f, g \rangle_{P_{\mathcal{A}}} (\forall f \in L^2(P_{\mathcal{X}}), g \in L^2(P_{\mathcal{A}}))$ , is given by $(T_{P^+}^* f)(a) = \int f(x) \frac{P^+(a \mid x) P_{\mathcal{X}}(x)}{P_{\mathcal{A}}(a)} dx = \mathbb{E}[f(X) \mid a]$ .

Now we discuss the spectral properties of these operators. Define the kernel integral operator $T_{k_{A}^{+}}: L^{2}(P_{\mathcal{A}}) \to L^{2}(P_{\mathcal{A}})$ as $(T_{k_{A}^{+}}g)(a) = \int g(a') k_{A}^{+}(a, a') dP_{\mathcal{A}}(a')$ . Define the other operator $T_{k_{X}^{+}}: L^{2}(P_{\mathcal{X}}) \to L^{2}(P_{\mathcal{X}})$ similarly. It is easy to see that $T_{k_{A}^{+}} = T_{P^{+}}^{*} T_{P^{+}}$ , and $T_{k_{X}^{+}} = T_{P^{+}} T_{P^{+}}^{*}$ .

We call $\lambda\in R$ an eigenvalue of $T_{k_{A}^{+}}$ with eigenfunction $\nu\in L^{2}(P_{\mathcal{A}})$ , if $T_{k_{A}^{+}}\nu=\lambda\nu$ . Suppose $T_{k_{A}^{+}}$ is a Hilbert-Schmidt integral operator. Then, we can order its eigenvalues by $1=\lambda_{0}\geq\lambda_{1}\geq\cdots\geq0$ , and the corresponding eigenfunctions $\nu_{0},\nu_{1},\cdots$ form an orthonormal basis (ONB) of $L^{2}(P_{\mathcal{A}})$ . Here $\lambda_{i}\leq1$ because of Jensen's inequality, and $\nu_{0}\equiv1$ is always an eigenfunction of $T_{k_{A}^{+}}$ with $\lambda_{0}=1$ . Similarly, we can order the eigenvalues of $T_{k_{X}^{+}}$ by $1=\kappa_{0}\geq\kappa_{1}\geq\cdots\geq0$ , and the eigenfunctions $\mu_{0},\mu_{1},\cdots$ form an ONB of $L^{2}(P_{\mathcal{X}})$ , where $\mu_{0}\equiv1$ . We also have the following result.

Lemma 3 (Duality property, [ZLR $^{+}$ 24], Proposition 1). For all i, we have $\lambda_{i} = \kappa_{i} \in [0,1]$ . And if $\lambda_{i} > 0$ , then we have $\mu_{i} = \lambda_{i}^{-\frac{1}{2}} T_{P^{+}} \nu_{i}$ , and $\nu_{i} = \lambda_{i}^{-\frac{1}{2}} T_{P^{+}}^{*} \mu_{i}$ .

We call $s_{i} = \lambda_{i}^{\frac{1}{2}}$ a singular value of $T_{P^{+}}$ , associated with left singular function $\mu_{i} \in L^{2}(P_{\mathcal{X}})$ and right singular function $\nu_{i} \in L^{2}(P_{\mathcal{A}})$ . Since we choose $\mu_{0} \equiv 1$ and $\nu_{0} \equiv 1$ , all other $\mu_{i}$ (and $\nu_{i}$ ) must have zero mean because they are orthogonal to $\mu_{0}$ (and $\nu_{0}$ ). Using these singular functions, we can spectrally decompose $P^{+}$ as follows.

Lemma 4. The spectral decomposition of $P+$ is $P^{+}(x,a)=\sum_{i}s_{i}\mu_{i}(x)\nu_{i}(a)P_{\mathcal{X}}(x)P_{\mathcal{A}}(a)$ .

Proof. $\forall i,\left\langle \frac{P^{+}(x,a)}{P_{\mathcal{X}}(x)P_{\mathcal{A}}(a)},\nu_{i}\right\rangle_{P_{\mathcal{A}}} = \int P^{+}(a\mid x)\nu_{i}(a)da = (T_{P^{+}}\nu_{i})(x) = (\lambda_{i}^{\frac{1}{2}}\mu_{i})(x) = s_{i}\mu_{i}(x)$ . Since $(\nu_{i})_{i\geq 0}$ is an ONB, we have $\frac{P^{+}(x,a)}{P_{\mathcal{X}}(x)P_{\mathcal{A}}(a)} = \sum_{i = 0}^{\infty}s_{i}\mu_{i}(x)\nu_{i}(a)$ .

The first key result of the contexture theory is that the optimal d-dimensional representation should recover the linear space spanned by the top-d singular functions $\mu_{1},\cdots,\mu_{d}$ . We say that such a representation learns the contexture. Note that the constant function $\mu_{0}\equiv1$ is excluded, as it does not need to be learned—there is no benefit in allocating a dimension to encode something already universally present.

Definition 5. A d-dimensional encoder $\Phi = [\phi_1, \cdots, \phi_d]$ learns the contexture of $P^+$ , if there exists a set of top-d singular functions $\{\mu_1, \cdots, \mu_d\}$ of $T_{P^+}$ , such that $\operatorname{span}\{\phi_1, \cdots, \phi_d\} = \operatorname{span}\{\mu_1, \cdots, \mu_d\}$ . We also say that $\Phi$ extracts the top-d eigenspace of $T_{k_X^+}$ .

If the multiplicity of $s_{d}$ is more than 1, then $\Phi$ recovering the span of any top-d singular functions suffices. The intuition why learning the contexture is ideal is that such a representation keeps the most information (variance) of the context, which is analogous to principal component analysis (PCA) in the finite-dimensional case. Consider the case where X and A are both finite sets. Let $N = |X|$ and $M = |A|$ . Then, a function $f \in L^{2}(P_{\mathcal{X}})$ is a vector in $R^{N}$ , $g \in L^{2}(P_{\mathcal{A}})$ is a vector in $R^{M}$ , and $T_{P+}$ is essentially a matrix $T \in R^{N \times M}$ . Suppose we want to learn a d-dimensional embedding $E \in R^{N \times d}$ for the N samples in X, and it should preserve the information of T as much as possible, then what should we do? PCA states that we should use the top-d left singular vectors of T as E, which are equivalent to the top-d eigenvectors of $TT^{\top}$ , because they maximize the explained variance. Similarly, functional spaces are essentially infinite-dimensional vector spaces, so the d-dimensional embedding of X that preserves the most information of $T_{P+}$ consists of the top-d left singular functions of $T_{P+}$ , or equivalently the top-d eigenfunctions of $T_{k_{X}^{+}} = T_{P+} T_{P+}^{*}$ .

# 3 Learning the Contexture

In this section, we show that every example method in Section 1.1 does one of the following:

(i) Extracts the top-d eigenspace of $T_{k_{X}^{+}} = T_{P^{+}} T_{P^{+}}^{*}$ (learns the contexture of $P^{+}$ ), which according to Definition 5 is equivalent to recovering the span of $\mu_{1}, \cdots, \mu_{d}$ (excluding $\mu_{0} \equiv 1$ );

(ii) Extracts the top- $d$ eigenspace of $T_{P^+}\Lambda T_{P^+}^*$ , where $\Lambda$ is the integral operator of a kernel $k_{\Lambda}(a,a')$ , that is $(\Lambda g)(a) = \int g(a')k_{\Lambda}(a,a')dP_{\mathcal{A}}(a')$ . $k_{\Lambda}$ is called the loss kernel, which is defined by the loss function used in the objective. Since the constant function is not necessarily the top eigenfunction of $T_{P^+}\Lambda T_{P^+}^*$ , in this case, we do not exclude any eigenfunction.

Notation: For any $f \in L^{2}(P_{\mathcal{X}})$ , denote its mean by $\bar{f} = \mathbb{E}_{P_{\mathcal{X}}}[f(X)]$ , and its centered version by $\tilde{f} = f - \bar{f}$ . The same notation is used for multi-dimensional functions and random variables, as long as the distribution is clear from context. The covariance matrix of any $\Phi : X \to R^{d}$ , denoted by $\operatorname{Cov}_{P_{\mathcal{X}}}[ \Phi ]$ , is a $d \times d$ matrix C where $C[i,j] = \left\langle \tilde{\phi}_{i}, \tilde{\phi}_{j} \right\rangle_{P_{\mathcal{X}}}$ .

# 3.1 Supervised Learning: Label Context

In the supervised learning paradigm, the context variable A is the label of x, and $P^{+}(\cdot \mid x)$ is the label distribution of x. Consider minimizing the mean squared error (MSE):

$$
\mathcal {R} (\Phi) = \min _ {\boldsymbol {W} \in \mathbb {R} ^ {d _ {A} \times d}, \boldsymbol {b} \in R ^ {d _ {A}}} \mathbb {E} _ {(X, A) \sim P ^ {+}} \left[ \| \boldsymbol {W} \Phi (X) + \boldsymbol {b} - A \| _ {2} ^ {2} \right]. \tag {1}
$$

That is, $\Phi(X)$ is the output of the layer before the last linear layer in a neural network, and b denotes the bias. If b can be an arbitrary vector, then the linear layer is biased; if b = 0 is fixed, then the linear layer is unbiased.

First, we study classification tasks where $A$ is one-hot.

Theorem 6 (Proof in Appendix A.1). Let $A$ be a one-hot random vector. Suppose the linear layer is unbiased, that is $\pmb{b} = \mathbf{0}$ . Then, $\Phi^{*}$ minimizes $\mathcal{R}(\Phi)$ if and only if it extracts the top- $d$ eigenspace of $T_{P^+}\Lambda T_{P^+}^*$ , where $k_{\Lambda}(a,a') = \mathbb{I}[a = a']$ , or $(\Lambda g)(a) = g(a)P_{\mathcal{A}}(a)$ . If all classes have the same size, then the top- $d$ eigenfunctions of $T_{P^+}\Lambda T_{P^+}^*$ and $T_{P^+}T_{P^+}^*$ are the same.

This theorem works for randomized labels, where each x could belong to multiple classes with certain probabilities. The loss kernel $k_{\Lambda}$ is a consequence of class imbalance, and it puts more weights on larger classes. Indeed, in practice, smaller classes are harder to learn. To get rid of $\Lambda$ , we can use the class-balanced risk (also known as importance weighting [Shi00]):

$$
\mathcal {R} _ {\mathrm{bal}} (\Phi) = \min _ {\boldsymbol {W} \in \mathbb {R} ^ {d _ {A} \times d}, \boldsymbol {b} \in R ^ {d _ {A}}} \mathbb {E} _ {(X, A) \sim P ^ {+}} \left[ \frac {\| \boldsymbol {W} \Phi (X) + \boldsymbol {b} - A \| _ {2} ^ {2}}{\sqrt {P _ {\mathcal {A}} (A)}} \right].
$$

Theorem 7 (Proof in Appendix A.2). Under the setting of Theorem 6, suppose the linear layer is biased. Then, $\Phi^{*}$ minimizes $\mathcal{R}_{\mathrm{bal}}(\Phi)$ if and only if it learns the contexture of $P^{+}$ .

Interestingly, our result can partially explain neural collapse. [PHD20] empirically showed that when there are d classes of equal sizes and the label A is deterministic, a sufficiently trained deep representation will collapse to an equiangular tight frame (ETF) $\phi_{1},\cdots,\phi_{d}$ , which are defined as $\phi_{i}(x)=c(\mathbb{I}[x\text{ belongs to class }i]-d^{-1})$ for all $i\in[d]$ and some constant c. The span of $\phi_{1},\cdots,\phi_{d}$ is the same as the span of the top-d eigenfunctions of $T_{P^{+}}\Lambda T_{P^{+}}^{*}$ . However, our result cannot explain why $\phi_{1},\cdots,\phi_{d}$ converge to the exact functions as above—it only proves that they will span the same space. To explain this, one needs to analyze the training dynamics, which depend on the specific optimizer, such as gradient-based methods, while our results are independent of the optimizer.

When the classes have different sizes, it is easy to see that the dual kernel of $T_{P^{+}}\Lambda T_{P^{+}}^{*}$ is $k_{X}^{+}(x,x^{\prime})=\mathbb{I}[x$ and $x^{\prime}$ have the same label]. This is equivalent to the simplex-encoded labels interpolation (SELI) defined by [TKVB22, Definition 2], which generalizes neural collapse.

For regression where $A$ is an arbitrary Euclidean vector, using the same objective as Eqn. (1), we can prove the following result.

Theorem 8 (Proof in Appendix A.3). $\Phi^{*}$ minimizes Eqn. (1) if and only if $\Phi^{*}$ extracts the top- $d$ eigenspace of $T_{P^{+}}\Lambda T_{P^{+}}^{*}$ . If the linear layer is unbiased ( $\mathbf{b} = \mathbf{0}$ ), then $k_{\Lambda}(a, a') = \langle a, a' \rangle$ ; if it is biased ( $\mathbf{b}$ can be arbitrary), then $k_{\Lambda}(a, a') = \langle \tilde{a}, \tilde{a}' \rangle$ .

Remark. Kernel $k_{\Lambda}(a,a')=\langle a,a'\rangle$ is called the linear kernel on A, and $k_{\Lambda}(a,a')=\left\langle\tilde{a},\tilde{a}'\right\rangle$ is called the centered linear kernel w.r.t. $P_{A}$ . Theorem 6 is a special case of Theorem 8.

# 3.2 Self-supervised Learning (SSL): Transformation Context

Two major types of SSL are multi-view learning and denoising autoencoders. Multi-view learning independently samples two views $A, A^{+}$ from $P^{+}(\cdot|x)$ for every $x$ . $A^{+}$ is called a positive sample of $A$ . Then, one trains $\Psi: \mathcal{A} \to \mathbb{R}^d$ such that $\Psi(A) \approx \Psi(A^{+})$ . This $\Psi$ is an encoder on $\mathcal{A}$ instead of $\mathcal{X}$ , so at downstream we need to convert $\Psi(a)$ to $\Phi(x)$ , which is typically done via the average encoder:

$$
\Phi (x) = \left(T _ {P ^ {+}} \Psi\right) (x) = \int \Psi (a) d P ^ {+} (a \mid x).
$$

By Lemma 3, we have the following corollary.

Corollary 9. Let $s_{d} > 0$ . The average encoder $\Phi$ spanning the span of the left top-d singular functions of $T_{P^{+}}$ is equivalent to $\Psi$ spanning the span of the right top-d singular functions of $T_{P^{+}}$ .

Enforcing $\Psi(A)\approx\Psi(A^{+})$ alone leads to the degenerate solution where $\Psi$ gives the same embedding to all a. This is called the feature collapse problem. There are two solutions: contrastive learning and non-contrastive learning. Prior work by [HWGM21, JHM23, ZLR $^{+}$ 24] showed that the following spectral contrastive loss $L_{C}$ and spectral non-contrastive loss $L_{N}$ can learn the contexture of $P^{+}$ . Let $A^{+}$ be a positive sample of A drawn from the same x, and $A^{-}$ be a negative sample drawn from another x independently.

$$
\mathcal {L} _ {\mathrm{C}} = \mathbb {E} \left[ - \left\langle \tilde {\Psi} (A), \tilde {\Psi} (A ^ {+}) \right\rangle + \frac {1}{2} \left\langle \tilde {\Psi} (A), \tilde {\Psi} (A ^ {-}) \right\rangle^ {2} \right];
$$

$$
\mathcal {L} _ {\mathrm{N}} = \mathbb {E} \left[ \left\| \Psi (A) - \Psi (A ^ {+}) \right\| _ {2} ^ {2} \right] \text { s   .   t   . } \operatorname{Cov} _ {P _ {A}} [ \Psi ] = I,
$$

where the $(i,j)$ -th entry of $\operatorname{Cov}_{P_{\mathcal{A}}}\left[\Psi\right]$ is $\langle\psi_{i},\psi_{j}\rangle_{P_{\mathcal{A}}}$ , $i,j\in[d]$ . Minimizing $L_{N}$ is a constrained optimization problem. The constraint $\operatorname{Cov}_{P_{\mathcal{A}}}\left[\Psi\right]=I$ is called the orthonormality constraint. It makes sure that $\Psi$ must be rank-d, so that it cannot be a constant function on A.

Theorem 10 (Proof in Appendix A.4). $\Psi^{*}$ minimizes $\mathcal{L}_{\mathrm{C}}$ or $\mathcal{L}_{\mathrm{N}}$ if and only if $\tilde{\Phi}^{*} = T_{P+}\tilde{\Psi}^{*}$ learns the contexture.

For denoising autoencoders, suppose $\mathcal{X} \subseteq \mathbb{R}^{d_X}$ . Then, consider minimizing the following objective:

$$
\mathcal {R} (\Psi) = \min _ {\boldsymbol {W} \in \mathbb {R} ^ {d _ {X} \times d}, \boldsymbol {b} \in \mathbb {R} ^ {d _ {X}}} \underset {(X, A) \sim P ^ {+}} {\mathbb {E}} \left[ \| \boldsymbol {W} \Psi (A) + \boldsymbol {b} - X \| _ {2} ^ {2} \right]. \tag {2}
$$

Theorem 11. Let $\Psi^{*}$ be any minimizer of Eqn. (2). Then, $\tilde{\Psi}^{*}$ extracts the top- $d$ eigenspace of $T_{P^{+}}^{*}\Lambda T_{P^{+}}$ , where $\Lambda$ is the integral operator of $k_{\Lambda}(x,x') = \langle \tilde{x},\tilde{x}'\rangle$ if $\mathbf{b}$ can be an arbitrary vector, or $k_{\Lambda}(x,x') = \langle x,x'\rangle$ if $\mathbf{b} = \mathbf{0}$ . Consequently, $\tilde{\Phi}^{*} = T_{P^{+}}\tilde{\Psi}^{*}$ extracts the top- $d$ eigenspace of $T_{P^{+}}T_{P^{+}}^{*}\Lambda$ .

Proof. The proof is the same as Theorem 8.

# 3.3 Node Representation Learning: Graph Context

Let $\mathcal{G} = (\mathcal{V},\mathcal{E})$ be an undirected graph, where edge $(u,v)$ has a weight $w(u,v)$ such that $w(u,v) = w(v,u)\geq 0$ . Let the degree of node $u$ be $d(u) = \sum_{v\in \mathcal{V}}w(u,v)$ , and $d_{\mathrm{sum}} = \sum_v d(v)$ . Let $P_{\mathcal{X}}(u) = \frac{d(u)}{d_{\mathrm{sum}}}$ be a distribution on $\mathcal{V}$ , and $P_{w}(u,v) = \frac{w(u,v)}{d_{\mathrm{sum}}}$ be a distribution on $\mathcal{E}$ . Define $P^{+}(v|u) = \frac{w(u,v)}{d(u)}$ . Then, the following optimization problem with a similar orthonormality constraint can learn the contexture of $P^{+}$ .

$$
\underset {\Phi : \mathcal {X} \rightarrow \mathbb {R} ^ {d}} {\text { minimize }} \quad \frac {1}{2} \mathbb {E} _ {(u, v) \sim P _ {w}} \left[ \| \Phi (u) - \Phi (v) \| _ {2} ^ {2} \right] \quad \text { s.t. } \quad \operatorname{Cov} _ {P _ {\mathcal {X}}} [ \Phi ] = I. \tag {3}
$$

Theorem 12 (Proof in Appendix A.5). Let $\Phi^{*}$ be any solution to Eqn. (3) (so that for any constant $c$ , $\Phi^{*} + c$ is also a solution). Then, $\tilde{\Phi}^{*}$ learns the contexture of $P^{+}$ .

# 4 Optimality of the Contexture

So far, we have shown that commonly used learning objectives can learn the contexture. Now, we present a rigorous theory that addresses the question of why and when learning the contexture is optimal. Arguably, no representations can be good for all downstream tasks. However, we show that a feature encoder that learns the contexture is optimal for the class of tasks that are compatible with the context. This provides a quantitative characterization of when a task respects the human prior knowledge the context incorporates. Interestingly, as we detail in the sequel, this has intriguing implications for scaling laws.

# 4.1 Compatibility

The ultimate evaluation of an encoder is its performance on relevant downstream tasks. Most downstream tasks, such as prediction, clustering, and segmentation, can be associated with a target function $f^{*} \in L^{2}(P_{\mathcal{X}})$ . For example, multi-class classification can be associated with multiple one-vs-all labeling functions. Moreover, if we are fitting a linear predictor on top of $\Phi$ , then the mean and variance of $f^{*}$ do not matter because we can change the weight and bias of the predictor accordingly. Thus, we can assume that $f^{*}$ is normalized, that is, it has zero mean and unit variance.

We say that a context $P^{+}$ is useful for a task $f^{*}$ , if it can help us learn a good predictor for $f^{*}$ . Formally, suppose $A \sim P^{+}(\cdot \mid x)$ is a random corruption of x. Consider the scenario where we have a corrupted training set $\{(a_{i}, y_{i})\}$ , where $a_{i} \sim P^{+}(\cdot \mid x_{i})$ and $y_{i} = f^{*}(x_{i})$ . That is, we cannot see the original samples $x_{i}$ , but can only see the corrupted samples $a_{i}$ . To learn a predictor on this training set, we can train a predictor $\hat{g}: A \to Y$ , and then use $\hat{f} = \mathbb{E}[\hat{g}(A) \mid x]$ . At test time, given input x, we can draw $a \sim P^{+}(\cdot \mid x)$ and then output the average of $g^{*}(a)$ . For this procedure to work, two conditions are necessary:

- There exists $g^{*} \in L^{2}(P_{\mathcal{A}})$ s.t. $f^{*}(x) = \mathbb{E}[g^{*}(A) \mid x]$ .   
- This $g^{*}$ has a low $\operatorname{Var}[g^{*}(A) \mid x]$ on average over $x$ .

If $\operatorname{Var}[g^{*}(A) \mid x]$ is high, then $g^{*}(a)$ will be far away from $y = f^{*}(x)$ , so fitting $\hat{g}$ on $(a, y)$ will not work. Based on these insights, we define compatibility as follows.

Definition 13. The compatibility with $P^{+}$ of any non-zero $f \in L^{2}(P_{\mathcal{X}})$ is defined as

$$
\rho (f, P ^ {+}) = \max _ {g \in L ^ {2} (P _ {\mathcal {A}}), g \neq \mathbf {0}} \frac {\left<   \tilde {f} , T _ {P ^ {+}} g \right> _ {P _ {\mathcal {X}}}}{\left\| \tilde {f} \right\| _ {P _ {\mathcal {X}}} \| g \| _ {P _ {\mathcal {A}}}} \in [ 0, 1 ].
$$

For further insight, let $f = \sum_{i} u_{i} \mu_{i}$ and $g = \sum_{i} v_{i} \nu_{i}$ . Then, $\rho(f, P^{+}) = \max_{v_{i}} \frac{\sum_{i \geq 1} s_{i} u_{i} v_{i}}{\sqrt{\sum_{i \geq 1} u_{i}^{2}} \sqrt{\sum_{i} v_{i}^{2}}} = \sqrt{\frac{\sum_{i > 1} s_{i}^{2} u_{i}^{2}}{\sum_{i \geq 1} u_{i}^{2}}}$ by Cauchy-Schwarz inequality (the optimal $v_{i}$ satisfy $v_{0} = 0$ and $v_{i} \propto s_{i} u_{i}$ for $i \geq 1$ ). For any $\epsilon > 0$ , we define the class of $(1 - \epsilon)$ -compatible tasks as

$$
\mathcal {F} _ {\epsilon} (P ^ {+}) = \big \{f \in L ^ {2} (P _ {\mathcal {X}}): \mathbb {E} [ f ] = 0, \rho (f, P ^ {+}) \geq 1 - \epsilon \big \}.
$$

This class of tasks satisfies the two conditions, i.e. we can find a $g^{*}$ with low variance $\operatorname{Var}[g^{*}(A) \mid x]$ :

Theorem 14 (Proof in Appendix B.1). For any $f^{*} \in \mathcal{F}_{\epsilon}(P^{+})$ , there exists a $g^{*} \in L^{2}(P_{\mathcal{A}})$ such that $f^{*}(x) = \mathbb{E}[g^{*}(A) \mid x]$ , and $g^{*}$ satisfies

$$
\underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A, A ^ {\prime} \sim P ^ {+} (\cdot \mid X)} {\mathbb {E}} \Big [ (g ^ {*} (A) - g ^ {*} (A ^ {\prime})) ^ {2} \Big ] \leq 4 \epsilon \| g ^ {*} \| _ {P _ {\mathcal {A}}} ^ {2}.
$$

Now that we have a class of tasks compatible with $P^{+}$ , we evaluate $\Phi$ by its worst-case approximation error on $\mathcal{F}_{\epsilon}(P^{+})$ . The most common way to evaluate $\Phi$ is to fit a linear predictor on top, also called a linear

probe, which is the focus of our attention (other methods for using $\Phi$ include fitting a small neural network on top, using a kernel method, or using KNN). Specifically, the worst-case approximation error of $\Phi$ on $\mathcal{F} \subset L^{2}(P_{\mathcal{X}})$ is the maximum error of the optimal linear probe in estimating any function in F. In this work, we focus on the $L_{2}$ error.

Definition 15. Let $\mathcal{F} \subset L^{2}(P_{\mathcal{X}})$ be a function class where $f \in \mathcal{F} \Rightarrow \alpha f \in \mathcal{F}$ for all $\alpha \in \mathbb{R}$ . The worst-case approximation error of $\Phi: \mathcal{X} \to \mathbb{R}^{d}$ on $\mathcal{F}$ is defined as

$$
\operatorname{err} (\Phi ; \mathcal {F}) = \max _ {f \in \mathcal {F} (P ^ {+}), \| f \| _ {P _ {\mathcal {X}}} = 1} \operatorname{err} (\Phi , f), \quad w h e r e \quad \operatorname{err} (\Phi , f) = \min _ {\boldsymbol {w} \in \mathbb {R} ^ {d}, b \in \mathbb {R}} \left\| \boldsymbol {w} ^ {\top} \Phi + b - f \right\| _ {P _ {\mathcal {X}}} ^ {2}.
$$

The following key result shows that the $\Phi$ that minimizes $\operatorname{err}(\Phi; \mathcal{F}_{\epsilon}(P^{+}))$ over all d-dimensional encoders must recover the linear space spanned by the $\mu_{1}, \cdots, \mu_{d}$ . Here $\mu_{0}$ is excluded since the bias b in the linear predictor implicitly contains $\mu_{0}$ .

Theorem 16 (Proof in Appendix B.2). Suppose $1 - s_{1} \leq \epsilon \leq 1 - \sqrt{\frac{s_{1}^{2} + s_{2}^{2}}{2}}$ . For any d, among all $\Phi = [\phi_{1}, \cdots, \phi_{d}]$ where $\phi_{i} \in L^{2}(P_{\mathcal{X}})$ , $\Phi$ minimizes $\text{err}(\Phi; \mathcal{F}_{\epsilon}(P^{+}))$ if and only if it learns the contexture of $T_{P^{+}}$ . The error is given by

$$
\min _ {\Phi : \mathcal {X} \to \mathbb {R} ^ {d},   \phi_ {i} \in L ^ {2} (P _ {\mathcal {X}})} \operatorname{err} \bigl (\Phi ; \mathcal {F} _ {\epsilon} (P ^ {+}) \bigr) = \frac {s _ {1} ^ {2} - (1 - \epsilon) ^ {2}}{s _ {1} ^ {2} - s _ {d + 1} ^ {2}}.
$$

Conversely, for any $d$ -dimensional encoder $\Phi$ and any $\epsilon > 0$ , there exists $f \in L^2(P_{\mathcal{X}})$ such that $\rho(f, P^+) = 1 - \epsilon$ , and $\mathrm{err}(\Phi, f) \geq \frac{s_1^2 - (1 - \epsilon)^2}{s_1^2 - s_{d+1}^2}$ .

This result has two parts. First, we show that if $f^{*}$ is compatible ( $f^{*} \in \mathcal{F}_{\epsilon}(P^{+})$ ), the optimal encoder achieves low error on $f^{*}$ . Second, we ask what if $f^{*}$ is incompatible. We cannot claim that no $\Phi$ works for $f^{*}$ —if one knows $f^{*}$ a priori, then one can set $\phi_{1} = f^{*}$ to achieve zero error. Instead, we show that for any $\Phi$ , there exists an f with the same compatibility as $f^{*}$ for which $\Phi$ performs poorly. Therefore, compatibility reflects whether a context is suitable for a task.

Evaluating an arbitrary encoder. The above result bounds the approximation error of the encoder that learns the contexture. We can also bound the approximation error of an arbitrary encoder using the contexture theory. See Appendix C.

# 4.2 Implications for Scaling Laws

Scaling laws $[KMH^{+}20]$ state that the performance of deep neural models such as foundation models grow with their size. Moreover, models of different architectures learn highly aligned $[KNLH19]$ representations when scaled up. $[HCWI24]$ thus proposed the platonic representation hypothesis that scaling makes representations more aligned with an underlying reality, though they did not formally define this reality.

The contexture theory provides a new perspective on the role of scaling. The function class from which the feature encoders $\Phi$ are trained is a subset of $L^{2}(P_{\mathcal{X}})$ , and as the model gets larger, the function class approaches $L^{2}(P_{\mathcal{X}})$ . This suggests that scaling brings the learned representation closer to the span of the top-d singular functions of $T_{P^{+}}$ , explaining why big models learn aligned representations. The key difference is that contexts are designed by humans and thus are more subjective than the so-called “underlying reality”.

We substantiate this extrapolation with an experiment on the abalone dataset from OpenML. We use KNN $(K=30)$ as context, where $A=X$ , and $P^{+}$ maps x to one of its K nearest neighbors equiprobably. We compare two d-dimensional representations with d=128 learned in the following two ways: (i) Kernel PCA to obtain the exact top-d eigenfunctions of $T_{k_{X}^{+}}$ ; (ii) non-contrastive learning ( $L_{N}$ in Theorem 10) implemented with VICReg [BPL22]. For (ii), we use a fully-connected neural network with Tanh activation, skip connections, and AdamW optimizer [KB15, LH17]. We use the same number of training epochs for every

![](images/afb843e3fc6685028c0a486397134930a8bc79dd76218511433e01572afa179f.jpg)  
Figure 1: Alignment between the learned representation and the top- $d$ eigenfunctions of $T_{k_X^+}$ on the abalone dataset. Solid curves: CCA. Dashed curves: mutual KNN. Depth here means the number of hidden layers.

model. For each width and depth, we run the experiments 15 times with different random initializations, and report the average alignment. See Appendix E for more details.

We measure the correlation between the two representations using two metrics: the classical canonical-correlation analysis (CCA) metric $R_{CCA}^{2}$ , and the mutual KNN metric with 10 neighbors as used by [HCWI24]. We center and whiten the representations (making the covariance identity) when using the mutual KNN. CCA is invariant to all invertible linear transformations $\Phi$ , which is ideal because such transformations do not affect the performance of the downstream linear probe, since one can adjust W and b of the linear probe accordingly. [KNLH19] also proposed the linear CKA metric, but we do not use it because it is only invariant to orthogonal transformations on $\Phi$ .

Figure 1 plots the alignment between the exact top-d eigenfunctions and the learned deep representation while varying the depth and width of the neural network. We can see that when depth and width are chosen correctly, the CCA can be as high as 0.9, and the mutual KNN can be higher than 0.8. Note that these alignment metric values are very high. For example, in [HCWI24], the mutual KNN metric value is usually below 0.2. Hence, the representation learned by the neural network is highly aligned with the top-d eigenfunctions.

The top plot studies neural networks with increasing widths. We observe that when the neural network is relatively narrow, increasing the width improves alignment. However, once the neural network is sufficiently wide, further increasing the width may have a negative effect. For example, when the depth is 3, the alignment is the highest when the width is 512, and the alignment becomes lower when the network is wider than 512. Since increasing the width can only make the function class of $\Phi$ larger, this phenomenon is not due to the expressivity of the neural network. We hypothesize that it arises from optimization difficulty, that is larger models are harder to train effectively. Consequently, with the same number of pretraining steps, a larger

model will be farther away from the minima, leading to a reduced alignment.

The bottom plot studies neural networks with increasing depths, and a similar trend is observed. When the network is shallow, increasing the depth improves the alignment. However, once the network is sufficiently deep, further increasing the depth may have a negative effect. We also observe from the bottom plot that a width-512 network has higher alignment than widths 1024 and 2048. In addition, the alignment does not reach 1. This is expected as the model is non-convex, so the true optima (the precise top-d eigenspace) cannot be found by gradient-based methods.

In summary, we draw two conclusions from this experiment: (i) the representation learned by a large neural network is highly aligned with the top- $d$ eigenfunctions; (ii) once the neural network is wide and deep enough, further increasing its size does not improve the alignment higher, and may even degrade it. Hence, we put forward the following argument about the scaling law: Once the model is large enough such that $\Phi$ is already highly aligned with the top- $d$ eigenfunctions, further increasing the model size inevitably yields diminishing returns.

# 5 Context Evaluation

Creating useful contexts that produce better representations is a challenging open problem. In this section, we take a first step by studying when a context is useful and how to efficiently evaluate its usefulness. The key result is that the usefulness of a context is largely determined by the association level between X and A, and a useful context should have a moderate association. The association level affects the shape of the spectrum, that is the decay rate of the singular values. We propose a metric that only uses the singular values to evaluate the usefulness of a context, without knowledge of the downstream task. Then, we empirically verify that this metric has a strong correlation with the actual performance of the encoder on many real datasets. As such, the proposed metric can help practitioners to select among various pretraining methods or hyperparameter settings efficiently.

# 5.1 The Effect of Context Association

A useful context should provide sufficient training signals that are easy for the model to capture. If the association between X and A of a context is too weak, then the signals will be insufficient. If the association is too strong, then capturing the signals will be too hard. The association level affects the spectrum of the context—the stronger the association, the slower the decay of the singular values. As an illustration, Figure 3 (top) displays the spectra of contexts with weak, moderate, and strong association, from left to right.

Case 1: Weak association. Consider the extreme case where A is independent of X. This context is clearly useless because it provides no information. In this case, only the trivial singular function $\mu_{0} \equiv 1$ has a positive singular value; all the other singular values are 0. When X and A are nearly independent, $k_{X}^{+}(x, x')$ is very close to 1, which causes the singular values to decay too fast. Formally, we have:

Lemma 17 (Proof in Appendix F.2). When $|k_X^+(x, x') - 1| < \epsilon$ for all $x, x' \in \mathcal{X}$ , we have $\sum_{i > 0} s_i^2 < \epsilon$ .

In Appendix F, we empirically verify that low association leads to a small $|k_{X}^{+}(x,x^{\prime})-1|$ for all $x,x^{\prime}$ . In such a scenario, $\mathcal{F}_{\epsilon}(P^{+})$ is a very small set, so very few tasks are compatible with and can benefit from the context.

Case 2: Strong association. Contexts with very strong associations, whose singular values decay too slowly, are not useful. For example, the extreme case A = X is clearly useless. There are two reasons: (i) for upstream, slow decay implies that there will be very non-smooth singular functions with large singular values, which are difficult to learn; (ii) for downstream, a larger d is needed, as more singular functions have non-trivial contributions to the kernel, and it leads to a higher sample complexity. In Appendix F, we

![](images/af3a81fd11b6f70502d0797a12608c5e8abd37c16895b52372c56114b5decce1.jpg)  
Figure 2: Association level vs, prediction error $err_{d^{*}} = \min_{d} err_{d}$ when $\Phi$ is d-dimensional, and the decay rate $\lambda$ for the RBF and KNN contexts on credit-approval and breast-w datasets. A larger $-\gamma$ (or K/N) indicates a lower association level, while a smaller $-\gamma$ (or K/N) corresponds to a higher association level. Across all four figures, we observe a U-shaped trend: prediction error increases at both extremes of low and high association. The estimated decay rate $\lambda$ consistently captures this behavior, serving as a proxy for the level of association.

empirically verify that kernel $k_{X}^{+}$ has a high Lipschitz constant when the association is strong, meaning that the kernel is non-smooth and thus the singular functions are non-smooth.

Quantitative measurements for level of association. While mutual information captures mutual dependence between random variables, estimating it from samples remains a longstanding challenge as it requires the joint density function to be known [Pan03]. As an alternative approach, we propose to use the decay rate of singular values $(s_{i})_{i\geq0}$ as an indicator of the strength of association.

To estimate the decay rate $\lambda$ , we assume that the singular values decay exponentially and fit the regression model $s_{i}^{2} = \exp(-\lambda i)$ . When $\lambda$ is large, it indicates a fast decay rate and the context has a low association. Conversely, when $\lambda$ is small, it implies a slow decay rate and highly associated context. In Figure 2, we empirically demonstrate that the estimated decay rate closely reflects the level of association, and that both extremely strong and weak associations can lead to degraded empirical performance. In Appendix F, we observe similar trends for all 28 datasets and propose alternative metrics for detecting the level of associations.

# 5.2 Task-agnostic Evaluation of Contexts

A good measurement of context usefulness should be task-agnostic, because we would like the pretrained encoder to be transferable to a variety of tasks, which we might not know at pretrain time. Note that for any task-agnostic metric, one can adversarially create a task for which the metric fails, so there is no universal task-agnostic metric. However, a metric can still be very useful if it provides guidance for most real tasks. To this end, we set up the following desiderata.

1. Approximation-estimation trade-off. The prediction error of a linear model fit on the encoder can be decomposed as the sum of approximation error and estimation error. The former decreases with d, while the latter increases with d. The metric should reflect this trade-off.   
2. Efficient computation. Note that estimating the singular functions is as hard as training an encoder. Therefore, we propose to use a metric that only uses the singular values of $T_{P^{+}}$ , which can be more efficiently estimated.

With these desiderata in mind, we define our metric as

$$
\tau_ {d} = \frac {1}{1 - s _ {d + 1} ^ {2}} + \beta \frac {\sum_ {i = 1} ^ {d} s _ {i} ^ {2}}{\sum_ {i = 1} ^ {d _ {0}} s _ {i} ^ {2}}, \quad \tau = \min _ {d} \tau_ {d}, \tag {4}
$$

where $\beta > 0$ is a parameter, and $d_{0}$ is the maximum embedding dimension we consider. Typically $d_{0}$ ranges from 512 to 8192. We choose $\beta = 1$ and $d_{0} = 512$ in our experiments. $\tau_{d}$ is a proxy of the prediction error when the embedding dimension is d. Thus, the d that minimizes $\tau_{d}$ can be viewed as the optimal embedding dimension predicted by the metric, and $\tau$ evaluates the context when d is chosen optimally.

Derivation of the metric. Let the target function be $f^{*} = f_{0} + f_{1}$ , where $\langle f_{0}, f_{1} \rangle_{P_{X}} = 0$ , $f_{1}$ is compatible with the context, and $f_{0}$ is not compatible with the context. The prediction error can then be decomposed into three components:

(i) The approximation error of $f_{1}$   
(ii) The approximation error of $f_{0}$   
(iii) The estimation error

By Theorem 16, component (i) can be bounded by $\frac{s_1^2 - (1 - \epsilon)^2}{s_1^2 - s_{d + 1}^2}$ . In practice, $s_1$ is usually very close to 1, and we simplify this bound to the first term of Eqn. (4), up to a constant factor. For component (ii), stronger associations imply that more tasks are compatible with the context, reducing this approximation error. Thus, this component should be negatively correlated with $\sum_{i=1}^{d_0} s_i^2$ . Component (iii), the estimation error, increases with stronger associations, since higher association typically requires a larger $d$ , and thus greater sample complexity. Based on the results in [ZLR+24], this component can be essentially understood as positively correlated with $\sum_{i=1}^{d} s_i^2$ . The second term in Eqn. (4) combines the contributions from components (ii) and (iii), and is designed to be bounded by 1. This metric can be efficiently estimated. It only requires the top- $d_0$ eigenfunctions of $T_{k_X^+}$ , which can be estimated in $O(m^3)$ time using a random subset of $m = \Theta(d_0 \log d_0)$ samples. See Appendix D for details.

Next, we empirically examine $\tau_{d}$ on two datasets. First, we apply the metric to the abalone dataset and use KNN as the context, similar to Section 4.2. We adjust the association of the context by changing K. In particular, we choose K = 150 (weak), K = 30 (moderate) and K = 5 (strong). We obtain the exact eigenvalues and eigenfunctions of $T_{k_{X}^{+}}$ using kernel PCA. In Figure 3, we plot the spectra of the three contexts in the top row. Then, in the bottom row, we compare $\tau_{d}$ against the prediction error of the linear probe under different d. We can see that when the association is weak or moderate, $\tau_{d}$ first decreases and then increases, which tracks the actual error. However, when the association is too strong, $\tau_{d}$ monotonically decreases with d, and it cannot track the actual error.

Second, we apply the metric to the MNIST dataset. The context is random cropping with crop ratio $\alpha$ . We adjust the association of the context by changing $\alpha$ . In particular, we choose $\alpha = 0.5$ (weak), $\alpha = 0.2$ (moderate) and $\alpha = 0.05$ (strong). Since kernel PCA is not scalable to datasets as large as MNIST, we instead train a neural network. Specifically, we train a LeNet [LBBH98] using the non-contrastive learning objective ( $L_{N}$ in Theorem 10) and the AdamW optimizer. Then, we estimate the top eigenvalues using the method in Appendix D. The downstream task is a binary classification task—whether the digit is greater than 4. After pretraining, a linear probe is fit on top of $\Phi$ using ridge regression. The result is plotted in Figure 4.

From Figure 4, we can see that when the association is not too strong, $\tau_{d}$ first decreases and then increases, similar to Figure 3. However, on MNIST, the downstream error monotonically decreases with $d$ , unlike abalone. This disparity is due to the difference between the two downstream tasks. To demonstrate this, in Figure 5 we plot the cosine similarity between the target function $f^{*}$ and the estimated $i$ -th eigenfunction on the two datasets. We can see that the variance of $f^{*}$ on abalone is mostly concentrated on the top-5 eigenfunctions, with the first cosine similarity being almost 0.5. In contrast, the variance of $f^{*}$ on MNIST is more scattered, and the cosine similarity is still close to 0.1 for the 150-th eigenfunction. Consequently, having a large

![](images/82795ef58306b299663b26d2f027ec258cda59add1c7ad7baeac70fc295f1ff7.jpg)

<details>
<summary>line</summary>

| i   | s²_i |
| --- | ---- |
| 0   | 1.0  |
| 50  | 0.1  |
| 100 | 0.01 |
| 150 | 0.001|
| 200 | 0.0001|
</details>

![](images/43fbef99fcdea91b7c858d79089ad176a019228019c7f6d5de14dbb928b925a4.jpg)

<details>
<summary>line</summary>

| i   | s²_i |
| --- | ---- |
| 0   | 1.0  |
| 100 | 0.3  |
| 200 | 0.1  |
</details>

![](images/fddc6c4c8b26818243c01692b7a131e3bb75a0774a083ac8e3bc5502c665b570.jpg)

<details>
<summary>line</summary>

| i   | g_i^2 |
| --- | ----- |
| 0   | 1.0   |
| 100 | 0.8   |
| 200 | 0.7   |
</details>

![](images/d554ff3717d699e304ef6091a7abccc29ebf3ce93db2833b2794fecb6cee5175.jpg)

<details>
<summary>line</summary>

| d   | Solid Line | Dashed Line |
| --- | ---------- | ----------- |
| 0   | 0.8        | 0.6         |
| 50  | 0.45       | 0.5         |
| 100 | 0.45       | 0.55        |
| 150 | 0.45       | 0.57        |
| 200 | 0.45       | 0.6         |
| 250 | 0.45       | 0.6         |
</details>

(a) Weak association

![](images/25c423d85bccacd2d1f77652864047e59f7fee41002cf143891f990d9bdb89bf.jpg)

<details>
<summary>line</summary>

| d   | Black Line | Red Dashed Line |
| --- | ---------- | --------------- |
| 0   | 0.8        | 0.6             |
| 50  | 0.5        | 0.55            |
| 100 | 0.45       | 0.55            |
| 150 | 0.45       | 0.57            |
| 200 | 0.45       | 0.58            |
| 250 | 0.45       | 0.59            |
</details>

(b) Moderate association

![](images/c344d3baa09159f851eec484e18413fe623a1d0732d366e7bfb8425c60e502f1.jpg)

<details>
<summary>line</summary>

| d   | Red Line | Black Line |
| --- | -------- | ---------- |
| 0   | 0.7      | 0.8        |
| 50  | 0.55     | 0.75       |
| 100 | 0.53     | 0.7        |
| 150 | 0.55     | 0.65       |
| 200 | 0.58     | 0.6        |
| 250 | 0.6      | 0.58       |
</details>

(c) Strong association

Figure 3: Metric illustration on abalone. Top row: context spectra. Bottom row: black solid curves are $\tau_{d}$ divided by 6; red dashed curves are the actual downstream prediction error. We divide $\tau_{d}$ by 6 to fit it in the same plot.   
![](images/995412617d56072d2fee54c199a08e4b35d7bcaa6ce0b9c3bfd5d1f7af07cd63.jpg)

<details>
<summary>line</summary>

| i   | s²_i |
| --- | ---- |
| 0   | 1.0  |
| 50  | 0.2  |
| 100 | 0.05 |
| 150 | 0.02 |
| 200 | 0.01 |
</details>

![](images/2dc6ef517c9185039499c0a3c829ab5d62ade32db34b45faf0cdbe8d2cb11e0b.jpg)

<details>
<summary>line</summary>

| i   | s²_i |
| --- | ---- |
| 0   | 1.0  |
| 100 | 0.2  |
| 200 | 0.05 |
</details>

![](images/c955e8f9f616129a449f654468f9d8e322c320497562318ddffa9b6e2d569d9c.jpg)

<details>
<summary>line</summary>

| i   | s²_i |
| --- | ---- |
| 0   | 1.0  |
| 100 | 0.5  |
| 200 | 0.25 |
</details>

![](images/3a0bc1ec09473ca13c63100ce40ee640630f66379f92d46a32d6b5971f88148f.jpg)

<details>
<summary>line</summary>

| d   | Solid Line | Dashed Line |
| --- | ---------- | ----------- |
| 0   | 0.2        | 0.8         |
| 100 | 0.3        | 0.4         |
| 200 | 0.3        | 0.3         |
</details>

(a) Weak association

![](images/31c4d22f8ae81d874ffb992025890e8e72e2ace3671ed6c32e28de7aec790816.jpg)

<details>
<summary>line</summary>

| d   | Solid Line | Dashed Line |
| --- | ---------- | ----------- |
| 0   | 0.8        | 0.8         |
| 50  | 0.35       | 0.4         |
| 100 | 0.3        | 0.35        |
| 150 | 0.3        | 0.3         |
| 200 | 0.3        | 0.25        |
</details>

(b) Moderate association

![](images/c1ab474c944fb445bd9296defd791cdc541fceb565969799b7df35eafa0129a7.jpg)

<details>
<summary>line</summary>

| d   | Solid Line | Dashed Line |
| --- | ---------- | ----------- |
| 0   | 0.8        | 0.8         |
| 100 | 0.4        | 0.3         |
| 200 | 0.35       | 0.25        |
</details>

(c) Strong association   
Figure 4: Metric illustration on MNIST, similar to Figure 3.

d on abalone will have a little impact on the approximation error but will increase the estimation error significantly. On the other hand, having a larger d on MNIST will decrease the approximation error more than it increases the estimation error, which is why the total error monotonically decreases with d.

The takeaway from this experiment is that, while a context with moderate association is generally good, its effectiveness ultimately depends on the specific downstream task.

For example, on abalone the weakest context actually leads to the lowest error, because the variance of $f^{*}$ is concentrated on the top-5 eigenfunctions. On the other hand, on MNIST the strongest context leads to the lowest error, because the variance of $f^{*}$ is scattered among a lot of features, and a stronger association

![](images/a4187136cba3e63720b27d512f9be6a81402ca88453d1d703c8aa294990d4bd7.jpg)

<details>
<summary>line</summary>

| x    | Cosine similarity |
| ---- | ----------------- |
| 0    | 0.45              |
| 10   | 0.1               |
| 20   | 0.18              |
| 30   | 0.05              |
| 40   | 0.12              |
| 50   | 0.08              |
| 60   | 0.06              |
| 70   | 0.07              |
| 80   | 0.05              |
| 90   | 0.04              |
| 100  | 0.03              |
| 110  | 0.05              |
| 120  | 0.04              |
| 130  | 0.03              |
| 140  | 0.02              |
| 150  | 0.01              |
</details>

(a) abalone

![](images/8e3eaa87b541486bbde8b7dd05b6411c35a6def385a06b2276a9c08a58dc6a02.jpg)

<details>
<summary>line</summary>

| x    | y     |
| ---- | ----- |
| 0    | 0.2   |
| 10   | 0.3   |
| 20   | 0.1   |
| 30   | 0.2   |
| 40   | 0.1   |
| 50   | 0.2   |
| 60   | 0.1   |
| 70   | 0.1   |
| 80   | 0.1   |
| 90   | 0.1   |
| 100  | 0.1   |
| 110  | 0.1   |
| 120  | 0.1   |
| 130  | 0.1   |
| 140  | 0.1   |
| 150  | 0.1   |
</details>

(b) MNIST   
Figure 5: Comparison of the downstream task between abalone and MNIST. The y-axis is the cosine similarity between the downstream task and the i-th eigenfunction.

allows more features to be discovered. Hence, no evaluation metric would universally work for all contexts and downstream tasks, but a metric would still be useful if it correlates well with the actual error in most scenarios, and thus can provide insights into choosing the right context and the right hyperparameters, such as the mask or crop ratio.

# 5.3 Empirical Verification

In this section, we examine whether our metric correlates with the encoder's performance on real datasets. In practice, the prediction error is influenced by many factors. To create a setting where all factors but the context are controlled, we let the encoder be the exact top- $d$ singular functions obtained by kernel PCA.

Each dataset is randomly split into a pretrain set, a downstream labeled set, and a test set. The downstream linear predictor is fit via ridge regression. Hyperparameter grid search is conducted at both encoder learning and downstream stages. The evaluation metric is the mean squared error. Let $err_{d}$ be the actual prediction error when $\Phi$ is d-dimensional. We test d up to $d_{0}=512$ . Let $d^{*}$ be the one that minimizes $err_{d}$ . We use the following four types of contexts.

- RBF kernels: $k(x, a) = \exp(-\gamma \| x - a \|^2)$ . We define $P^+$ as $P^+(a \mid x) \propto k(x, a)$ for each $x$ .   
- KNN: $P^{+}(a \mid x) = K^{-1}$ if $a$ is a KNN of $x$ , else 0.   
- RBF $_{\text{mask}}$ : First, randomly mask 20% of the features, and then apply RBF kernels to the other features. Specifically, we randomly draw 50 masks, and use the average of $P^{+}$ over all masks as the context. We do not use masking alone because its association is too strong.   
- $\mathrm{KNN}_{\mathrm{mask}}$ : $20\%$ random masking and then apply KNN.

For each of these contexts, $A = X$ . For each type, we use 35 contexts by adjusting $\gamma$ for RBF kernels and K for KNN. By doing so, we adjust the association level between X and A. We make sure that contexts in every type range from very weak to very strong association.

In Table 1 we report the correlation between $\tau$ and $\mathrm{err}_{d^*}$ over all 140 contexts from the four types on 28 classification (Cls) and regression (Reg) datasets from OpenML [VvRBT13] that are widely used in machine learning research. The most common metric is the Pearson correlation, but it can only detect linear correlations, while the correlation between $\tau$ and $\mathrm{err}_{d^*}$ is not necessarily linear. Thus, we also report the distance correlation [SRB07], another common metric that can detect non-linear correlations, but it cannot tell if the correlation is positive or negative because it is always non-negative.

The median reported in the table shows that on more than half of the datasets, the Pearson correlation is over 0.5, which is generally considered a strong correlation. The distance correlation is even higher. As

Table 1: Correlation between $\tau$ and the actual error $err_{d^{*}}$ on all 4 types of contexts. 

<table><tr><td>Dataset</td><td>Size (↑)</td><td>#Feature</td><td>Type</td><td>Pearson</td><td>Distribution</td></tr><tr><td>credit-approval</td><td>690</td><td>15</td><td>Cls</td><td>0.583</td><td>0.683</td></tr><tr><td>breast-w</td><td>699</td><td>9</td><td>Cls</td><td>0.072</td><td>0.255</td></tr><tr><td>diabetes</td><td>768</td><td>8</td><td>Cls</td><td>0.737</td><td>0.740</td></tr><tr><td>solar_flare</td><td>1066</td><td>10</td><td>Reg</td><td>0.019</td><td>0.262</td></tr><tr><td>Moneyball</td><td>1232</td><td>14</td><td>Reg</td><td>0.680</td><td>0.650</td></tr><tr><td>yeast</td><td>1269</td><td>8</td><td>Cls</td><td>0.221</td><td>0.256</td></tr><tr><td>cmc</td><td>1473</td><td>9</td><td>Cls</td><td>0.867</td><td>0.860</td></tr><tr><td>Wine</td><td>1599</td><td>11</td><td>Reg</td><td>-0.084</td><td>0.212</td></tr><tr><td>scene</td><td>2407</td><td>299</td><td>Cls</td><td>0.608</td><td>0.685</td></tr><tr><td>dna</td><td>3186</td><td>180</td><td>Cls</td><td>0.881</td><td>0.843</td></tr><tr><td>splice</td><td>3190</td><td>60</td><td>Cls</td><td>0.831</td><td>0.801</td></tr><tr><td>kr-vs-kp</td><td>3196</td><td>36</td><td>Cls</td><td>0.543</td><td>0.512</td></tr><tr><td>abalone</td><td>4177</td><td>8</td><td>Reg</td><td>0.028</td><td>0.470</td></tr><tr><td>spambase</td><td>4601</td><td>57</td><td>Cls</td><td>0.775</td><td>0.858</td></tr><tr><td>colleges</td><td>7603</td><td>44</td><td>Reg</td><td>0.155</td><td>0.387</td></tr><tr><td>mushroom</td><td>8124</td><td>22</td><td>Cls</td><td>0.185</td><td>0.340</td></tr><tr><td>kin8nm</td><td>8192</td><td>8</td><td>Reg</td><td>0.805</td><td>0.760</td></tr><tr><td>pumadyn32nh</td><td>8192</td><td>32</td><td>Reg</td><td>0.938</td><td>0.961</td></tr><tr><td>cpu_activity</td><td>8192</td><td>21</td><td>Reg</td><td>0.709</td><td>0.825</td></tr><tr><td>SpeedDating</td><td>8378</td><td>120</td><td>Cls</td><td>0.590</td><td>0.656</td></tr><tr><td>grid_stability</td><td>10000</td><td>12</td><td>Reg</td><td>0.925</td><td>0.911</td></tr><tr><td>sulfur</td><td>10081</td><td>6</td><td>Reg</td><td>-0.180</td><td>0.487</td></tr><tr><td>brazilian_houses</td><td>10692</td><td>9</td><td>Reg</td><td>-0.290</td><td>0.563</td></tr><tr><td>fifa</td><td>19178</td><td>28</td><td>Reg</td><td>-0.349</td><td>0.663</td></tr><tr><td>superconductivity</td><td>21263</td><td>81</td><td>Reg</td><td>0.141</td><td>0.367</td></tr><tr><td>kings_county</td><td>21613</td><td>21</td><td>Reg</td><td>0.842</td><td>0.882</td></tr><tr><td>health_insurance</td><td>22272</td><td>11</td><td>Reg</td><td>0.601</td><td>0.749</td></tr><tr><td>cps88wages</td><td>28155</td><td>6</td><td>Reg</td><td>0.250</td><td>0.479</td></tr><tr><td rowspan="2" colspan="3"></td><td>Mean</td><td>0.431</td><td>0.611</td></tr><tr><td>Median</td><td>0.587</td><td>0.659</td></tr></table>

expected, the metric does not work on all datasets. For example, the Pearson correlation is very negative on brazilian\_houses and fifa.

To understand when our metric might fail, we further visualize the results by plotting $\tau$ against $err_{d^{*}}$ on five of the datasets in Figure 6. In this figure, plots (a), (b), and (c) are three success cases where a clear positive correlation can be observed, and plots (d) and (e) display two failure cases. Plot (d) shows a common failure case: if $\tau$ is very close to $2 = \beta + 1$ , meaning that the metric believes that the association is extremely weak or extremely strong, then the metric will predict that the context is bad. However, a generally bad context can still be good on some tasks. For example, a very weak context still works well on a task that only uses the top-5 singular functions of the context, as we have shown in Section 5.2. Therefore, it is advisable to abstain from using the metric when it is too close to $\beta + 1$ .

Plot (e) shows a case where the metric is generally good for every single context type but has poor cross-type behavior. Specifically, it fails to predict that KNN is worse than RBF on this dataset. This suggests

![](images/d81e1c0cc468dee66bf8b981ecace8eb098e396b64f3f5e7cfbdfd19a2eb24c5.jpg)

<details>
<summary>scatter</summary>

| τ    | err_d* |
|------|--------|
| 1.6  | 0.7    |
| 1.8  | 0.75   |
| 2.0  | 0.8    |
</details>

(a) diabetes

![](images/abc63154f05564c052e540df53481a40fe41d406836736043bca337b2b025f56.jpg)

<details>
<summary>scatter</summary>

| τ    | Series 1 | Series 2 | Series 3 | Series 4 |
|------|----------|----------|----------|----------|
| 1.2  | 0.6      | 0.6      | 0.6      | 0.6      |
| 1.5  | 0.7      | 0.7      | 0.7      | 0.7      |
| 1.8  | 0.9      | 0.9      | 0.9      | 0.9      |
</details>

(b) splice

![](images/5af1f5a73ca6bb2e87ba6d8df38898ccf56137d55587059a2569d151190fa123.jpg)

<details>
<summary>scatter</summary>

| τ    | y     |
| ---- | ----- |
| 1.6  | 0.25  |
| 1.7  | 0.28  |
| 1.8  | 0.30  |
| 1.9  | 0.35  |
| 2.0  | 0.40  |
</details>

(c) kings-county

![](images/0a624bb2ad76196c6eb1d71e323e4e461b5ca8e2c2cb8523e19ced9b8cfd5b13.jpg)

<details>
<summary>line</summary>

| τ    | err_d* |
| ---- | ------ |
| 1.85 | 0.03   |
| 1.9  | 0.03   |
| 1.95 | 0.03   |
| 2.0  | 0.06   |
</details>

(d) brazilian\_houses

![](images/f1ca86089007378475b8f12e14e054b988f018a2c2d0716d615953c095357fe1.jpg)

<details>
<summary>scatter</summary>

| τ    | y     | Series |
|------|-------|--------|
| 1.6  | 0.35  | Red    |
| 1.7  | 0.38  | Red    |
| 1.8  | 0.36  | Orange |
| 1.9  | 0.39  | Orange |
| 2.0  | 0.42  | Red    |
| 1.6  | 0.32  | Green  |
| 1.7  | 0.28  | Green  |
| 1.8  | 0.27  | Green  |
| 1.9  | 0.26  | Green  |
| 2.0  | 0.25  | Green  |
</details>

(e) fifa

![](images/e76e8026e6d968ff3d15d5146ce4b5078de74aa1ca9fde5162aae45bbe9f44c9.jpg)

<details>
<summary>text_image</summary>

○RBF
□KNN
▲RBF ★ Masking
★KNN ★ Masking
</details>

Figure 6: Scatter plots of $\tau$ versus $err_{d^{*}}$ . Dashed line: Linear fit.

that the metric might not be able to compare different types of contexts. For example, if two contexts of completely different types have similar spectra, then the metric will indicate that they are similarly useful. This is because the metric only depends on the spectrum. However, it could be possible that for a particular task, one context is good and the other is bad, and our metric cannot reflect this disparity.

Overall, although there does not exist a universal metric that works for all contexts and tasks, and our metric does have failure cases, the experiment results here provide empirical evidence that more often than not, the proposed metric correlates well with the actual prediction error of the downstream linear probe. Hence, the proposed metric is useful for choosing hyperparameters and comparing contexts in practice.

# 6 Conclusion

The contexture theory established in this paper characterizes the mechanism of representation learning by clarifying the target of representation learning—the top singular functions of the expectation operator induced by the contexture, that is the association between the input X and a context variable A. A representation that learns the contexture achieves the lowest worst-case approximation error on the class of tasks compatible with the context. We show that most representation learning approaches could be cast as learning the contexture, and empirically demonstrate that the representations learned by large neural networks are highly aligned with the top singular functions. We further analyze how to evaluate the usefulness of a context using its spectrum, and the key takeaway is that a good context should have a moderate association between X and A.

Our analysis has three limitations, which lead to three open problems. First, our analysis focused on the minimizers of the objectives. However, $[CKL^{+}21]$ showed that deep models trained by popular gradient methods do not find the minimizers, but instead oscillate around the edge of stability. The open problem is whether the representation is always aligned with the top-d singular functions as it is oscillating. Second, we did not discuss the impact of the inductive bias of the model architecture, such as the translation invariance

of convolutional neural networks. Such inductive biases can affect the context and, therefore, the encoder. We pose how to integrate the effect of these biases into our theory as an open problem. Third, our theory assumes that $P_{X}$ is fixed. In practice, however, there is always a data distribution shift from pretraining to downstream. Hence, the open problem is how to refine our theory to handle such distribution shifts.

# Acknowledgements

We thank Rattana Pukdee, Hugo Contant, Chenhao Zhang and Zihao Ye for their feedback on this paper. Kai Yang did this work at Carnegie Mellon University, under the PKU-CMU undergraduate research program. We acknowledge the support of NSF via IIS-2211907, ONR via N00014-23-1-2368, AFRL via FA8750-23-2-1015, and DARPA via HR00112020006.

# References

[ADM+23] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15619–15629, 2023.   
[AS18] Alessandro Achille and Stefano Soatto. Emergence of invariance and disentanglement in deep representations. Journal of Machine Learning Research, 19(50):1–34, 2018.   
[BA00] Gaston Baudat and Fatiha Anouar. Generalized discriminant analysis using a kernel approach. Neural computation, 12(10):2385-2404, 2000.   
[BDR $^{+}$ 04] Yoshua Bengio, Olivier Delalleau, Nicolas Le Roux, Jean-François Paiement, Pascal Vincent, and Marie Ouimet. Learning eigenfunctions links spectral embedding and kernel PCA. Neural computation, 16(10):2197–2219, 2004.   
[BHA $^{+}$ 21] Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. arXiv:2108.07258, 2021.   
[BHX $^{+}$ 22] Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, and Michael Auli. Data2vec: A general framework for self-supervised learning in speech, vision and language. In International conference on machine learning, pages 1298–1312. PMLR, 2022.   
[BIS $^{+}$ 23] Randall Balestriero, Mark Ibrahim, Vlad Sobal, Ari S. Morcos, Shashank Shekhar, Tom Goldstein, Florian Bordes, Adrien Bardes, Grégoire Mialon, Yuandong Tian, Avi Schwarzschild, Andrew Gordon Wilson, Jonas Geiping, Quentin Garrido, Pierre Fernandez, Amir Bar, Hamed Pirsiavash, Yann LeCun, and Micah Goldblum. A cookbook of self-supervised learning. arxiv:2304.12210, 2023.   
[BL22] Randall Balestriero and Yann LeCun. Contrastive and non-contrastive self-supervised learning recover global and local spectral embedding methods. Advances in Neural Information Processing Systems, 35:26671–26685, 2022.   
[BN02] Mikhail Belkin and Partha Niyogi. Using manifold structure for partially labeled classification. In Proc. Advances in Neural Information Processing Systems, Vancouver, BC, Canada, December 2002.   
[BN03] Mikhail Belkin and Partha Niyogi. Laplacian eigenmaps for dimensionality reduction and data representation. Neural computation, 15(6):1373-1396, 2003.

[BPL22] Adrien Bardes, Jean Ponce, and Yann LeCun. VICReg: Variance-invariance-covariance regularization for self-supervised learning. In International Conference on Learning Representations, 2022.   
[BRR $^{+}$ 23] Simon Buchholz, Goutham Rajendran, Elan Rosenfeld, Bryon Aragam, Bernhard Schölkopf, and Pradeep Ravikumar. Learning linear causal representations from interventions under general nonlinear mixing. In Proc. Advances in Neural Information Processing Systems, New Orleans, LA, December 2023.   
[CKL $^{+}$ 21] Jeremy Cohen, Simran Kaur, Yuanzhi Li, J Zico Kolter, and Ameet Talwalkar. Gradient descent on neural networks typically occurs at the edge of stability. In International Conference on Learning Representations, 2021.   
[CKNH20] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In Proc. International Conference on Machine Learning, July 2020.   
[CL06] Ronald R Coifman and Stéphane Lafon. Diffusion maps. Applied and computational harmonic analysis, 21(1):5–30, 2006.   
[DCLT19] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), pages 4171–4186. Association for Computational Linguistics, June 2019.   
[DLS22] Alexandru Damian, Jason Lee, and Mahdi Soltanolkotabi. Neural networks can learn representations with gradient descent. In Conference on Learning Theory, London, UK, July 2022.   
[FPM $^{+}$ 24] Marco Fumero, Marco Pegoraro, Valentino Maiorca, Francesco Locatello, and Emanuele Rodolà. Latent functional maps: a spectral framework for representation alignment. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 66178–66203. Curran Associates, Inc., 2024.   
[GSA $^{+}$ 20] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Remi Munos, and Michal Valko. Bootstrap your own latent - a new approach to self-supervised learning. In Proc. Advances in Neural Information Processing Systems, December 2020.   
[GWDL23] Agrim Gupta, Jiajun Wu, Jia Deng, and Fei-Fei Li. Siamese masked autoencoders. In Advances in Neural Information Processing Systems, New Orleans, LA, December 2023.   
[HAE16] Minyoung Huh, Pulkit Agrawal, and Alexei A Efros. What makes imagenet good for transfer learning? arXiv:1608.08614, 2016.   
[HAP $^{+}$ 18] Irina Higgins, David Amos, David Pfau, Sebastien Racaniere, Loic Matthey, Danilo Rezende, and Alexander Lerchner. Towards a definition of disentangled representations. arXiv:1812.02230, 2018.

[HCWI24] Minyoung Huh, Brian Cheung, Tongzhou Wang, and Phillip Isola. Position: The platonic representation hypothesis. In Proc. International Conference on Machine Learning, Vienna, Austria, July 2024.   
[HCX $^{+}$ 22] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, New Orleans, LA, June 2022.   
[HWGM21] Jeff Z HaoChen, Colin Wei, Adrien Gaidon, and Tengyu Ma. Provable guarantees for self-supervised deep learning with spectral contrastive loss. In Proc. Advances in Neural Information Processing Systems, December 2021.   
[JHM23] Daniel D. Johnson, Ayoub El Hanchi, and Chris J. Maddison. Contrastive learning can find an optimal basis for approximately view-invariant functions. In International Conference on Learning Representations, 2023.   
[JVLT22] Li Jing, Pascal Vincent, Yann LeCun, and Yuandong Tian. Understanding dimensional collapse in contrastive self-supervised learning. In International Conference on Learning Representations, 2022.   
[KB15] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Yoshua Bengio and Yann LeCun, editors, 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, 2015.   
[KKMH20] Ilyes Khemakhem, Diederik Kingma, Ricardo Monti, and Aapo Hyvärinen. Variational autoencoders and nonlinear ICA: A unifying framework. In Proc. International Conference on Artificial Intelligence and Statistics, August 2020.   
[KMB08] Nikolaus Kriegeskorte, Marieke Mur, and Peter A Bandettini. Representational similarity analysis-connecting the branches of systems neuroscience. Frontiers in systems neuroscience, 2:249, 2008.   
[KMH $^{+}$ 20] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv:2001.08361, 2020.   
[KNLH19] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In Proc. International Conference on Machine Learning, Long Beach, CA, June 2019.   
[LBBH98] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.   
[LH17] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.   
[LLM04] Qingshan Liu, Hanqing Lu, and Songde Ma. Improving kernel fisher discriminant analysis for face recognition. IEEE transactions on circuits and systems for video technology, 14(1):42–49, 2004.   
[MRW $^{+}$ 99] Sebastian Mika, Gunnar Ratsch, Jason Weston, Bernhard Scholkopf, and Klaus-Robert Mullers. Fisher discriminant analysis with kernels. In Neural networks for signal processing IX: Proceedings of the 1999 IEEE signal processing society workshop (cat. no. 98th8468), pages 41–48. Ieee, 1999.

[NCL $^{+}$ 23] Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. Progress measures for grokking via mechanistic interpretability. In The Eleventh International Conference on Learning Representations, 2023.   
[NLW23] Neel Nanda, Andrew Lee, and Martin Wattenberg. Emergent linear representations in world models of self-supervised sequence models. In Yonatan Belinkov, Sophie Hao, Jaap Jumelet, Najoung Kim, Arya McCarthy, and Hosein Mohebbi, editors, Proceedings of the 6th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, pages 16–30, Singapore, December 2023. Association for Computational Linguistics.   
[ODM $^{+}$ 23] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv:2304.07193, 2023.   
[OLV18] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv:1807.03748, 2018.   
[OWJ+22] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. In Proc. Advances in Neural Information Processing Systems, New Orleans, LA, December 2022.   
[Pan03] Liam Paninski. Estimation of entropy and mutual information. Neural computation, 15(6):1191-1253, 2003.   
[PHD20] Vardan Papyan, XY Han, and David L Donoho. Prevalence of neural collapse during the terminal phase of deep learning training. Proceedings of the National Academy of Sciences, 117(40):24652–24663, 2020.   
[RDS $^{+}$ 15] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International Journal of Computer Vision, 115:211–252, 2015.   
[RS00] Sam T Roweis and Lawrence K Saul. Nonlinear dimensionality reduction by locally linear embedding. Science, 290(5500):2323-2326, 2000.   
[RWC $^{+}$ 19] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.   
[Shi00] Hidetoshi Shimodaira. Improving predictive inference under covariate shift by weighting the log-likelihood function. Journal of Statistical Planning and Inference, 90(2):227–244, 2000.   
[SLB $^{+}$ 21] Bernhard Schölkopf, Francesco Locatello, Stefan Bauer, Nan Rosemary Ke, Nal Kalchbrenner, Anirudh Goyal, and Yoshua Bengio. Toward causal representation learning. Proceedings of the IEEE, 109(5):612–634, May 2021.   
[SRB07] Gábor J. Székely, Maria L. Rizzo, and Nail K. Bakirov. Measuring and testing dependence by correlation of distances. The Annals of Statistics, 35(6):2769–2794, 2007.   
[STWCK05] John Shawe-Taylor, Christopher KI Williams, Nello Cristianini, and Jaz Kandola. On the eigenspectrum of the gram matrix and the generalization error of kernel-PCA. IEEE Transactions on Information Theory, 51(7):2510–2522, 2005.   
[Sut24] Ilya Sutskever. Test of time award talk: Sequence to sequence learning with neural networks. Advances in Neural Information Processing Systems, 2024.

[SVZ14] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks: Visualising image classification models and saliency maps. In Yoshua Bengio and Yann LeCun, editors, 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Workshop Track Proceedings, 2014.   
[SZBK $^{+}$ 23] Ravid Shwartz-Ziv, Randall Balestriero, Kenji Kawaguchi, Tim GJ Rudner, and Yann LeCun. An information theory perspective on variance-invariance-covariance regularization. Advances in Neural Information Processing Systems, 36:33965–33998, 2023.   
[Tia22] Yuandong Tian. Understanding deep contrastive learning via coordinate-wise optimization. In Proc. Advances in Neural Information Processing Systems, New Orleans, LA, December 2022.   
[TKVB22] Christos Thrampoulidis, Ganesh Ramachandra Kini, Vala Vakilian, and Tina Behnia. Imbalance trouble: Revisiting neural-collapse geometry. In Proc. Advances in Neural Information Processing Systems, New Orleans, LA, December 2022.   
[VAST24] Burak Varıcı, Emre Acartürk, Karthikeyan Shanmugam, and Ali Tajer. General identifiability and achievability for causal representation learning. In Proc. International Conference on Artificial Intelligence and Statistics, Valencia, Spain, May 2024.   
[VvRBT13] Joaquin Vanschoren, Jan N. van Rijn, Bernd Bischl, and Luis Torgo. Openml: Networked science in machine learning. SIGKDD Explorations, 15(2):49–60, 2013.   
[WTB $^{+}$ 22] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. Emergent abilities of large language models. Transactions on Machine Learning Research, 2022. Survey Certification.   
[Yar18] Dmitry Yarotsky. Optimal approximation of continuous functions by very deep relu networks. In Conference on Learning Theory, pages 639-649, Stockholm, Sweden, July 2018.   
[YN22] Kayo Yin and Graham Neubig. Interpreting language models with contrastive explanations. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 184–198, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics.   
[YXL $^{+}$ 24] Dingling Yao, Danru Xu, Sébastien Lachapelle, Sara Magliacane, Perouz Taslakian, Georg Martius, Julius von Kügelgen, and Francesco Locatello. Multi-view causal representation learning with partial observability. In Proc. International Conference on Learning Representations, Vienna, Austria, May 2024.   
[ZBH $^{+}$ 17] Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. In International Conference on Learning Representations, 2017.   
[ZF14] Matthew D Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part I 13, pages 818–833. Springer, 2014.   
[ZGL03] Xiaojin Zhu, Zoubin Ghahramani, and John D Lafferty. Semi-supervised learning using gaussian fields and harmonic functions. In Proc. International Conference on Machine learning, pages 912–919, Washington, DC, USA, August 2003.

[ZJM $^{+}$ 21] Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, and Stéphane Deny. Barlow twins: Self-supervised learning via redundancy reduction. In Proc. International Conference on Machine Learning, pages 12310–12320, July 2021.   
[ZLR $^{+}$ 24] Runtian Zhai, Bingbin Liu, Andrej Risteski, Zico Kolter, and Pradeep Ravikumar. Understanding augmentation-based self-supervised representation learning via rkhs approximation and regression. In International Conference on Learning Representations, 2024.

# A Proofs for Section 3

# A.1 Proof of Theorem 6

Theorem 6. Let A be a one-hot random vector. Suppose the linear layer is unbiased, that is b = 0. Then, $\Phi^{*}$ minimizes $\mathcal{R}(\Phi)$ if and only if it extracts the top-d eigenspace of $T_{P^{+}}\Lambda T_{P^{+}}^{*}$ , where $k_{\Lambda}(a, a') = \mathbb{I}[a = a']$ , or $(\Lambda g)(a) = g(a)P_{\mathcal{A}}(a)$ . If all classes have the same size, then the top-d eigenfunctions of $T_{P^{+}}\Lambda T_{P^{+}}^{*}$ and $T_{P^{+}}T_{P^{+}}^{*}$ are the same.

The following lemma will be very useful in the proof.

Lemma 18. $T_{P^{+}}\Lambda T_{P^{+}}^{*}$ is the integral kernel operator of the following kernel

$$
k (x, x ^ {\prime}) = \iint k _ {\Lambda} (a, a ^ {\prime}) P ^ {+} (a | x) P ^ {+} (a ^ {\prime} | x ^ {\prime}) d a d a ^ {\prime}.
$$

Proof. By definition, we have

$$
(T _ {P ^ {+}} ^ {*} h) (a ^ {\prime}) = \int h (x ^ {\prime}) P ^ {+} (x ^ {\prime} | a ^ {\prime}) d x ^ {\prime}.
$$

Thus, we have

$$
\begin{array}{l} (\Lambda T _ {P ^ {+}} ^ {*} h) (a) = \int (T _ {P ^ {+}} ^ {*} h) (a ^ {\prime}) k _ {\Lambda} (a, a ^ {\prime}) P _ {\mathcal {A}} (a ^ {\prime}) d a ^ {\prime} \\ = \iint h (x ^ {\prime}) P ^ {+} (x ^ {\prime} | a ^ {\prime}) k _ {\Lambda} (a, a ^ {\prime}) P _ {\mathcal {A}} (a ^ {\prime}) d x ^ {\prime} d a ^ {\prime} \\ = \iint h (x ^ {\prime}) P ^ {+} (a ^ {\prime} | x ^ {\prime}) k _ {\Lambda} (a, a ^ {\prime}) P _ {\mathcal {X}} (x ^ {\prime}) d x ^ {\prime} d a ^ {\prime}. \\ \end{array}
$$

This implies that

$$
\begin{array}{l} (T _ {P ^ {+}} \Lambda T _ {P ^ {+}} ^ {*} h) (x) = \int (\Lambda T _ {P ^ {+}} ^ {*} h) (a) P ^ {+} (a | x) d a \\ = \iiint h (x ^ {\prime}) k _ {\Lambda} (a, a ^ {\prime}) P ^ {+} (a | x) P ^ {+} (a ^ {\prime} | x ^ {\prime}) P _ {\mathcal {X}} (x ^ {\prime}) d a d a ^ {\prime} d x ^ {\prime} \\ = \int h (x ^ {\prime}) k (x, x ^ {\prime}) P _ {\mathcal {X}} (x ^ {\prime}) d x ^ {\prime}, \\ \end{array}
$$

as desired.

Then, we finish the proof of Theorem 6.

Proof. For any fixed $\Phi$ , define

$$
\mathcal {R} (\Phi , \boldsymbol {W}) = \mathbb {E} _ {P ^ {+}} \Big [ \| A - \boldsymbol {W} \Phi (X) \| _ {2} ^ {2} \Big ] = \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A \sim P ^ {+} (\cdot | X)} {\mathbb {E}} \Big [ \| A - \boldsymbol {W} \Phi (X) \| _ {2} ^ {2} \Big ].
$$

Assuming, without loss of generality, that $\mathbb{E}_{X\sim P_{\mathcal{X}}}\left[\Phi_i\Phi_j\right] = \delta_{ij}$ ; otherwise one can perform Gram-Schmidt process on $\Phi_i$ and change the value of $W$ respectively. Thus, it amounts to minimizing

$$
\begin{array}{l} \mathcal {R} (\Phi , \boldsymbol {W}) = \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A \sim P ^ {+} (\cdot | X)} {\mathbb {E}} \left[ \| A - \boldsymbol {W} \Phi (X) \| _ {2} ^ {2} \right] \\ = \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \| \boldsymbol {W} \Phi (X) \| _ {2} ^ {2} - 2 \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A \sim P ^ {+} (\cdot | X)} {\mathbb {E}} \langle A, \boldsymbol {W} \Phi (X) \rangle + \underset {A \sim P _ {\mathcal {A}}} {\mathbb {E}} \| A \| _ {2} ^ {2} \\ = \| \boldsymbol {W} \| _ {F} ^ {2} - 2 \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A \sim P ^ {+} (\cdot | X)} {\mathbb {E}} \langle A, \boldsymbol {W} \Phi (X) \rangle + \underset {A \sim P _ {\mathcal {A}}} {\mathbb {E}} \| A \| _ {2} ^ {2}. \\ \end{array}
$$

Denote $\pmb{W} = (w_{ij})_{1\leq i\leq d_A,1\leq j\leq d}$ . We have

$$
\frac {\partial \mathcal {R}}{\partial w _ {i j}} = 2 w _ {i j} - 2 \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A \sim P ^ {+} (\cdot | X)} {\mathbb {E}} [ A _ {i} \Phi_ {j} (X) ],
$$

which implies that for a fixed $\Phi$ , the optimal W that minimizes this loss should satisfy

$$
w _ {i j} = \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A \sim P ^ {+} (\cdot | X)} {\mathbb {E}} [ A _ {i} \Phi_ {j} (X) ].
$$

Combining the minimizer of W with R and notice that $E_{A\sim P_{A}}\|A\|_{2}^{2}$ is a constant, it suffices to maximize

$$
\begin{array}{l} F (\Phi) = \sum_ {i, j} \left[ \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A \sim P ^ {+} (\cdot | X)} {\mathbb {E}} A _ {i} \Phi_ {j} (X) \right] ^ {2} \\ = \int \sum_ {j} \Phi_ {j} (x _ {1}) \Phi_ {j} (x _ {2}) \langle a _ {1}, a _ {2} \rangle P _ {\mathcal {X}} (x _ {1}) P ^ {+} (a _ {1} | x _ {1}) P _ {\mathcal {X}} (x _ {2}) P ^ {+} (a _ {2} | x _ {2}) d x _ {1} d a _ {1} d x _ {2} d a _ {2} \\ = \iint \sum_ {j} \Phi_ {j} (x _ {1}) \Phi_ {j} (x _ {2}) \hat {k} (x _ {1}, x _ {2}) P _ {\mathcal {X}} (x _ {1}) P _ {\mathcal {X}} (x _ {2}) d x _ {1} d x _ {2}, \\ \end{array}
$$

where

$$
\begin{array}{l} \hat {k} (x _ {1}, x _ {2}) = \iint \langle a _ {1}, a _ {2} \rangle P ^ {+} (a _ {1} | x _ {1}) P ^ {+} (a _ {2} | x _ {2}) d a _ {1} d a _ {2} \tag {5} \\ = \iint \mathbb {I} [ a _ {1} = a _ {2} ] P ^ {+} (a _ {1} | x _ {1}) P ^ {+} (a _ {2} | x _ {2}) d a _ {1} d a _ {2}. \\ \end{array}
$$

Thus $\Phi^{*}$ is a minimizer of $\mathcal{R}(\Phi)$ if $\Phi^{*}$ extracts the top- $d$ eigenfunctions of $\hat{k}(x_1, x_2)$ . Combining with Lemma 18 yields that $k_{\Lambda}(a, a') = \mathbb{I}[a = a']$ . Furthermore, we have $(\Lambda g)(a) = \int g(a') k_{\Lambda}(a, a') dP_{\mathcal{A}}(a') = g(a) P_{\mathcal{A}}(a)$ , as desired.

If all classes have the same size, we have $P_{\mathcal{A}}(a) \equiv c \in (0,1)$ where c is a constant. Thus $(\Lambda g)(a) = g(a)P_{\mathcal{A}}(a) = cg(a)$ , which implies that $T_{P^{+}}\Lambda T_{P^{+}}^{*} = cT_{P^{+}}T_{P^{+}}^{*}$ . This concludes that $T_{P^{+}}\Lambda T_{P^{+}}^{*}$ and $T_{P^{+}}T_{P^{+}}^{*}$ share the same top-d eigenfunctions.

# A.2 Proof of Theorem 7

Theorem 7. Under the setting of Theorem 6, suppose the linear layer is biased. Then, $\Phi^{*}$ minimizes $\mathcal{R}_{\mathrm{bal}}(\Phi)$ if and only if it learns the contexture of $P^{+}$ .

Proof. Let us first assume that the linear layer is unbiased, that is b = 0. For any fixed $\Phi$ , define

$$
\mathcal {R} (\Phi , \boldsymbol {W}) = \mathbb {E} _ {P ^ {+}} \left[ \frac {1}{\sqrt {P _ {\mathcal {A}} (A)}} \| A - \boldsymbol {W} \Phi (X) \| _ {2} ^ {2} \right] = \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A \sim P ^ {+} (\cdot | X)} {\mathbb {E}} \left[ \frac {1}{\sqrt {P _ {\mathcal {A}} (A)}} \| A - \boldsymbol {W} \Phi (X) \| _ {2} ^ {2} \right].
$$

Assuming, without loss of generality,

$$
\underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A \sim P ^ {+} (\cdot | X)} {\mathbb {E}} \left[ \frac {1}{\sqrt {P _ {\mathcal {A}} (A)}} \Phi_ {i} \Phi_ {j} \right] = \delta_ {i j};
$$

otherwise we can perform Gram-Schmidt process on $\Phi_{i}$ and change the value of W respectively. Thus it suffices to minimize

$$
\begin{array}{l} \mathcal {R} (\Phi , \boldsymbol {W}) = \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A \sim P ^ {+} (\cdot | X)} {\mathbb {E}} \left[ \frac {1}{\sqrt {P _ {\mathcal {A}} (A)}} \| A - \boldsymbol {W} \Phi (X) \| _ {2} ^ {2} \right] \\ = \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A \sim P ^ {+} (\cdot | X)} {\mathbb {E}} \left[ \frac {1}{\sqrt {P _ {\mathcal {A}} (A)}} \| \boldsymbol {W} \Phi (X) \| _ {2} ^ {2} \right] \\ - 2 \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A \sim P ^ {+} (\cdot | X)} {\mathbb {E}} \left\langle \frac {A}{\sqrt {P _ {\mathcal {A}} (A)}}, \boldsymbol {W} \Phi (X) \right\rangle + \underset {A \sim P _ {\mathcal {A}}} {\mathbb {E}} \left[ \frac {\| A \| _ {2} ^ {2}}{\sqrt {P _ {\mathcal {A}} (A)}} \right] \\ = \| \boldsymbol {W} \| _ {F} ^ {2} - 2 \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A \sim P ^ {+} (\cdot | X)} {\mathbb {E}} \left\langle \frac {A}{\sqrt {P _ {\mathcal {A}} (A)}}, \boldsymbol {W} \Phi (X) \right\rangle + \underset {A \sim P _ {\mathcal {A}}} {\mathbb {E}} \left[ \frac {\| A \| _ {2} ^ {2}}{\sqrt {P _ {\mathcal {A}} (A)}} \right]. \\ \end{array}
$$

Denote $\boldsymbol{W} = (w_{ij})_{1\leq i\leq d_A,1\leq j\leq d}$ . We have

$$
\frac {\partial \mathcal {R}}{\partial w _ {i j}} = 2 w _ {i j} - 2 \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A \sim P ^ {+} (\cdot | X)} {\mathbb {E}} \left[ \frac {A _ {i}}{\sqrt {P _ {\mathcal {A}} (A)}} \Phi_ {j} (X) \right],
$$

which implies that for a fixed $\Phi$ , the minimizer of W satisfies

$$
w _ {i j} = \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A \sim P ^ {+} (\cdot | X)} {\mathbb {E}} \left[ \frac {A _ {i}}{\sqrt {P _ {\mathcal {A}} (A)}} \Phi_ {j} (X) \right].
$$

Combining the minimizer of W with R, it suffices to maximize

$$
\mathcal {R} ^ {\prime} = \sum_ {i, j} \left[ \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A \sim P ^ {+} (\cdot | X)} {\mathbb {E}} \frac {A _ {i}}{\sqrt {P _ {\mathcal {A}} (A)}} \Phi_ {j} (X) \right] ^ {2} = \iint \sum_ {j} \Phi_ {j} (x _ {1}) \Phi_ {j} (x _ {2}) \hat {k} (x _ {1}, x _ {2}) P _ {\mathcal {X}} (x _ {1}) P _ {\mathcal {X}} (x _ {2}) d x _ {1} d x _ {2},
$$

where

$$
\begin{array}{l} \hat {k} (x _ {1}, x _ {2}) = \iint \frac {\langle a _ {1} , a _ {2} \rangle}{\sqrt {P _ {\mathcal {A}} (a _ {1}) P _ {\mathcal {A}} (a _ {2})}} P ^ {+} (a _ {1} | x _ {1}) P ^ {+} (a _ {2} | x _ {2}) d a _ {1} d a _ {2} \\ = \iint \frac {\mathbb {I} [ a _ {1} = a _ {2} ]}{\sqrt {P _ {\mathcal {A}} (a _ {1}) P _ {\mathcal {A}} (a _ {2})}} P ^ {+} (a _ {1} | x _ {1}) P ^ {+} (a _ {2} | x _ {2}) d a _ {1} d a _ {2} \\ = \int \frac {P ^ {+} (a | x _ {1}) P ^ {+} (a | x _ {2})}{P _ {\mathcal {A}} (a)} d y. \\ \end{array}
$$

Thus, $\Phi^{*}$ is a minimizer of $\mathcal{R}(\Phi)$ if $\Phi^{*}$ extracts the top-d eigenfunctions of $\hat{k}(x_{1}, x_{2})$ . Note that here the top-d eigenfunctions include $\mu_{0} \equiv 1$ . When we include the bias b in the linear layer, then this $\mu_{0}$ is covered by the bias term, so $\Phi$ will extract $\mu_{1}, \cdots, \mu_{d}$ .

# A.3 Proof of Theorem 8

Theorem 8. $\Phi^{*}$ minimizes Eqn. (1) if and only if $\Phi^{*}$ extracts the top- $d$ eigenspace of $T_{P^{+}}\Lambda T_{P^{+}}^{*}$ . If the linear layer is unbiased ( $\mathbf{b} = \mathbf{0}$ ), then $k_{\Lambda}(a, a') = \langle a, a' \rangle$ ; if it is biased ( $\mathbf{b}$ can be arbitrary), then $k_{\Lambda}(a, a') = \langle \tilde{a}, \tilde{a}' \rangle$ .

Proof. For the unbiased linear model, the proof is similar to that of Theorem 6. Combining Eqn. (5) and Lemma 18 yields the desired result.

Next, we consider a biased linear model. For a variable $z$ , we denote $\bar{z} = \mathbb{E}[Z]$ , and $\tilde{z} = z - \mathbb{E}[Z]$ as its centered version.

For any fixed $\Phi$ , define

$$
\begin{array}{l} \mathcal {R} (\Phi , \boldsymbol {W}, \boldsymbol {b}) = \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A \sim P ^ {+} (\cdot | X)} {\mathbb {E}} \left[ \| A - \boldsymbol {W} \Phi (X) - \boldsymbol {b} \| _ {2} ^ {2} \right] \\ = \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A \sim P ^ {+} (\cdot | X)} {\mathbb {E}} \left[ \| A - \boldsymbol {W} \Phi (X) - \boldsymbol {b} \| _ {2} ^ {2} \right] \\ = \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A \sim P ^ {+} (\cdot | X)} {\mathbb {E}} \left[ \left\| \tilde {A} - \boldsymbol {W} \tilde {\Phi} (X) - \hat {\boldsymbol {b}} \right\| _ {2} ^ {2} \right] \\ = \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A \sim P ^ {+} (\cdot | X)} {\mathbb {E}} \left[ \left\| \tilde {A} - \boldsymbol {W} \tilde {\Phi} (X) \right\| _ {2} ^ {2} \right] + \left\| \hat {\boldsymbol {b}} \right\| _ {2} ^ {2} \\ \end{array}
$$

where $\hat{\pmb{b}} = \pmb{W}\mathbb{E}_{X\sim P_{\mathcal{X}}}\big[\Phi (X)\big] - \mathbb{E}_{A\sim P_A}[A]$ + $\pmb{b}$ . Thus, for any fixed $\Phi, \pmb{W}$ , the optimal $\pmb{b} = \mathbb{E}_{A\sim P_A}[A] - \pmb{W}\mathbb{E}_{X\sim P_{\mathcal{X}}}\big[\Phi (X)\big]$ .

Assuming, without loss of generality, $E_{X\sim P_{X}}[\tilde{\Phi}_{i}\tilde{\Phi}_{j}] = \delta_{ij}$ ; otherwise we can perform Gram-Schmidt process on $\tilde{\Phi}_{i}$ and change the value of W respectively. Thus, it suffices to minimize

$$
\begin{array}{l} \hat {\mathcal {R}} (\Phi , \boldsymbol {W}) = \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A \sim P ^ {+} (\cdot | X)} {\mathbb {E}} \left[ \left\| \tilde {A} - \boldsymbol {W} \tilde {\Phi} (X) \right\| _ {2} ^ {2} \right] \\ = \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \left\| \boldsymbol {W} \tilde {\Phi} (X) \right\| _ {2} ^ {2} - 2 \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A \sim P ^ {+} (\cdot | X)} {\mathbb {E}} \left\langle \tilde {Y}, \boldsymbol {W} \tilde {\Phi} (X) \right\rangle + \underset {A \sim P _ {\mathcal {A}}} {\mathbb {E}} \left\| \tilde {A} \right\| _ {2} ^ {2} \\ = \| \boldsymbol {W} \| _ {F} ^ {2} - 2 \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A \sim P ^ {+} (\cdot | X)} {\mathbb {E}} \left\langle \tilde {A}, \boldsymbol {W} \tilde {\Phi} (X) \right\rangle + \underset {A \sim P _ {\mathcal {A}}} {\mathbb {E}} \left\| \tilde {A} \right\| _ {2} ^ {2}. \\ \end{array}
$$

Denote $\pmb{W} = (w_{ij})_{1\leq i\leq d_y,1\leq j\leq d}$ . We have

$$
\frac {\partial \hat {\mathcal {R}}}{\partial w _ {i j}} = 2 w _ {i j} - 2 \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A \sim P ^ {+} (\cdot | X)} {\mathbb {E}} \left[ \tilde {A} _ {i} \tilde {\Phi} _ {j} (X) \right],
$$

which implies that for a fixed $\Phi$ , the minimizer of W satisfies

$$
w _ {i j} = \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A \sim P ^ {+} (\cdot | X)} {\mathbb {E}} \left[ \tilde {A} _ {i} \tilde {\Phi} _ {j} (X) \right].
$$

Combining the minimizer of $\mathbf{W}$ with $\hat{\mathcal{R}}$ and notice that $\mathbb{E}_{A\sim P_A}\left\| \tilde{A}\right\| _2^2$ is a constant, it suffices to maximize

$$
\begin{array}{l} \hat {\mathcal {R}} ^ {\prime} = \sum_ {i, j} \left[ \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A \sim P ^ {+} (\cdot | X)} {\mathbb {E}} \tilde {A} _ {i} \tilde {\Phi} _ {j} (X) \right] ^ {2} \\ = \int \sum_ {j} \tilde {\Phi} _ {j} (x _ {1}) \tilde {\Phi} _ {j} (x _ {2}) \langle \tilde {a} _ {1}, \tilde {a} _ {2} \rangle P _ {\mathcal {X}} (x _ {1}) P ^ {+} (a _ {1} | x _ {1}) P _ {\mathcal {X}} (x _ {2}) P ^ {+} (a _ {2} | x _ {2}) d x _ {1} d a _ {1} d x _ {2} d a _ {2} \\ = \iint \sum_ {j} \tilde {\Phi} _ {j} (x _ {1}) \tilde {\Phi} _ {j} (x _ {2}) \hat {k} (x _ {1}, x _ {2}) P _ {\mathcal {X}} (x _ {1}) P _ {\mathcal {X}} (x _ {2}) d x _ {1} d x _ {2}, \\ \end{array}
$$

where

$$
\hat {k} (x _ {1}, x _ {2}) = \iint \langle \tilde {a} _ {1}, \tilde {a} _ {2} \rangle P ^ {+} (a _ {1} | x _ {1}) P ^ {+} (a _ {2} | x _ {2}) d a _ {1} d a _ {2}.
$$

Notice that

$$
\begin{array}{l} \iint \hat {k} (x _ {1}, x _ {2}) P _ {\mathcal {X}} (x _ {1}) P _ {\mathcal {X}} (x _ {2}) d x _ {1} d x _ {2} = \int \langle \tilde {a} _ {1}, \tilde {a} _ {2} \rangle P ^ {+} (x _ {1}, a _ {1}) P ^ {+} (x _ {2}, a _ {2}) d x _ {1} d a _ {1} d x _ {2} d a _ {2} \\ = \int \langle \tilde {a} _ {1}, \tilde {a} _ {2} \rangle P _ {\mathcal {A}} (a _ {1}) P _ {\mathcal {A}} (a _ {2}) d a _ {1} d a _ {2} = 0, \\ \end{array}
$$

thus $\Phi^{*}$ is a minimizer of $\mathcal{R}(\Phi)$ if $\tilde{\Phi}^{*}$ extracts the top- $d$ eigenfunctions of $\hat{k}(x_1, x_2)$ . Combining with Lemma 18 yields the desired results.

# A.4 Proof of Theorem 10

Theorem 10. $\Psi^{*}$ minimizes $\mathcal{L}_{\mathrm{C}}$ or $\mathcal{L}_{\mathrm{N}}$ if and only if $\tilde{\Phi}^{*} = T_{P^{+}}\tilde{\Psi}^{*}$ learns the contexture.

Proof. We prove the two cases as follows.

(i) The spectral contrastive loss is

$$
\mathcal {L} _ {\mathrm{C}} (\Psi) = \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A, A ^ {+} \sim P ^ {+} (\cdot | X)} {\mathbb {E}} \left[ - \Big \langle \tilde {\Psi} (A), \tilde {\Psi} (A ^ {+}) \Big \rangle + \frac {1}{2} \underset {A ^ {-} \sim P _ {\mathcal {A}}} {\mathbb {E}} \left[ \Big \langle \tilde {\Psi} (A), \tilde {\Psi} (A ^ {-}) \Big \rangle^ {2} \right] \right].
$$

Suppose $\psi_{i} = \sum_{j\geq 0}c_{ij}\nu_{j}$ where $\nu_{j}$ is the ONB of $L^2 (P_{\mathcal{A}})$ in Lemma 3. Since $\nu_{j}$ is the ONB of $L^2 (P_{\mathcal{A}})$ and $\nu_0\equiv 1$ , we can get for $j\geq 1$ , $\mathbb{E}_{P_{\mathcal{A}}}\big[\nu_j(a)\big] = \delta_{0,j} = 0$ . Thus we can get $\tilde{\psi}_i = \psi_i - \mathbb{E}[\psi_i] = \sum_{j\geq 1}c_{ij}\nu_j$ .

Denote matrix $\boldsymbol{C}=(c_{ij})_{1\leq i\leq d,j\geq1}$ , matrix $\boldsymbol{B}=(b_{ij}):=\boldsymbol{C}^{\top}\boldsymbol{C}$ , and matrix $\boldsymbol{D}=\mathrm{diag}(s_{1}^{2},s_{2}^{2},\cdots)$ where $s_{i}$ is the singular value of $T_{P^{+}}$ . We have

$$
\underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A, A ^ {+} \sim P ^ {+} (\cdot | X)} {\mathbb {E}} \left[ \left\langle \tilde {\Psi} (A), \tilde {\Psi} (A ^ {+}) \right\rangle \right]
$$

$$
= \iiint \left\langle \tilde {\Psi} (a), \tilde {\Psi} (a ^ {+}) \right\rangle P ^ {+} (a | x) P ^ {+} (a ^ {+} | x) P _ {\mathcal {X}} (x) d x d a d a ^ {+}
$$

$$
= \int \left\langle \int \tilde {\Psi} (a) P ^ {+} (a | x) d y, \int \tilde {\Psi} (a ^ {+}) P ^ {+} (a ^ {+} | x) d a ^ {+} \right\rangle p (x) d x
$$

$$
= \int \left\langle T _ {P ^ {+}} \tilde {\Psi} (x), T _ {P ^ {+}} \tilde {\Psi} (x) \right\rangle p (x) d x = \| T _ {P ^ {+}} \tilde {\Psi} \| _ {P _ {\mathcal {X}}} ^ {2}
$$

$$
= \sum_ {i} s _ {i} ^ {2} b _ {i i};
$$

and

$$
\underset {A, A ^ {-} \sim P _ {\mathcal {A}}} {\mathbb {E}} \left[ \left\langle \tilde {\Psi} (A), \tilde {\Psi} (A ^ {-}) \right\rangle^ {2} \right] = \iint \left[ \sum_ {i = 1} ^ {d} \tilde {\psi} _ {i} (a) \tilde {\psi} _ {i} (a ^ {-}) \right] ^ {2} d P _ {\mathcal {A}} (a) d P _ {\mathcal {A}} (a ^ {-})
$$

$$
= \sum_ {1 \leq i, j \leq d} \left[ \int \tilde {\psi} _ {i} (a) \tilde {\psi} _ {j} (a) d P _ {\mathcal {A}} (a) \right] ^ {2}
$$

$$
= \sum_ {i, j} b _ {i j} ^ {2}.
$$

Thus, we have

$$
\mathcal {L} _ {\mathrm{C}} (\Psi) = - \sum_ {i} s _ {i} ^ {2} b _ {i i} + \frac {1}{2} \sum_ {i, j} b _ {i j} ^ {2} = \| \boldsymbol {B} - \boldsymbol {D} \| _ {F} ^ {2} - \| \boldsymbol {D} \| _ {F} ^ {2}.
$$

So if suffices to minimize $\|B-D\|_{F}^{2}$ where $\operatorname{rank}(B) \leq d$ . By Eckart-Young-Mirsky Theorem, we know the minimizer of B is $B^{*} = \operatorname{diag}(s_{1}^{2}, \cdots, s_{d}^{2})$ . Thus, the minimizer of C should be $C^{*} = U \operatorname{diag}(s_{1}, \cdots, s_{d})$ where $U \in R^{d \times d}$ is an orthonormal matrix. This indicates the minimizer $\tilde{\Psi}^{*}$ extracts the top-d singular functions of $T_{P^{+}}$ , and $\tilde{\Phi}^{*}$ learns the contexture of $P^{+}$ .

(ii) The non-contrastive loss is

$$
\mathcal {L} _ {\mathrm{N}} (\Psi) = \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A, A ^ {+} \sim P ^ {+} (\cdot | X)} {\mathbb {E}} \Big [ - \Big \langle \tilde {\Psi} (A), \tilde {\Psi} (A ^ {+}) \Big \rangle \Big ];
$$

$$
\mathcal {L} _ {\mathrm{N}} ^ {\prime} (\Psi) = \underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A, A ^ {+} \sim P ^ {+} (\cdot | X)} {\mathbb {E}} \bigg [ \big \| \Psi (A) - \Psi (A ^ {+}) \big \| _ {2} ^ {2} \bigg ],
$$

where $\operatorname{Cov}_{P_{\mathcal{A}}}\left[\Psi\right]=I$ . Since for any $\Psi$ ,

$$
\mathcal {L} _ {\mathrm{N}} ^ {\prime} (\Psi) - \mathcal {L} _ {N} (\Psi) = 2 \underset {A \sim P _ {\mathcal {A}}} {\mathbb {E}} \left[ \left\| \tilde {\Psi} (A) \right\| _ {2} ^ {2} \right] = 2 d
$$

is a constant, thus $\Psi^{*}$ minimizes $\mathcal{L}_{\mathrm{N}}(\Psi) \iff \Psi^{*}$ minimizes $\mathcal{L}_{\mathrm{N}}^{\prime}(\Psi)$ .

Suppose $\psi_i = \sum_{j\geq 0}c_{ij}\nu_j$ where $\nu_{j}$ is the ONB of $L^2 (P_{\mathcal{A}})$ in Lemma 4. Since $\mathbb{E}_{P_A}[\nu_j(a)] = \delta_{0,j}$ , we can get $\tilde{\psi}_i = \psi_i - \mathbb{E}[\psi_i] = \sum_{j\geq 1}c_{ij}\nu_j$ .

We now consider the minimizer of $\mathcal{L}_{\mathrm{N}}(\Psi)$ . By the calculation in (i), we obtain

$$
\mathcal {L} _ {\mathrm{N}} (\Psi) = - \underset {X \sim P _ {\mathcal {X}} A, A ^ {+} \sim P ^ {+} (\cdot | X)} {\mathbb {E}} \mathbb {E} \left[ \left\langle \tilde {\Psi} (A), \tilde {\Psi} (A ^ {+}) \right\rangle \right] = - \| T _ {P ^ {+}} \tilde {\Psi} \| _ {P _ {\mathcal {X}}} ^ {2} = - \sum_ {i} s _ {i} ^ {2} b _ {i i}.
$$

By $\mathbb{E}_{P_{\mathcal{A}}}\left[\tilde{\psi}_i\tilde{\psi}_j\right] = \delta_{ij}$ , we have

$$
\sum_ {i} b _ {i i} = \sum_ {i, j} c _ {i j} ^ {2} = d.
$$

Since $\nu_{i}$ is an ONB of $L^{2}(P_{\mathcal{A}})$ , $\tilde{\psi}_{1},\cdots,\tilde{\psi}_{d}$ are orthogonal, we have

$$
b _ {i i} = \sum_ {j = 1} ^ {d} c _ {j i} ^ {2} = \sum_ {j = 1} ^ {d} \left\langle \tilde {\psi} _ {j}, \nu_ {i} \right\rangle_ {P _ {\mathcal {A}}} ^ {2} \leq \| \nu_ {i} \| _ {P _ {\mathcal {A}}} ^ {2} = 1. \tag {6}
$$

Thus, we conclude that

$$
\mathcal {L} _ {\mathrm{N}} (\Psi) + \sum_ {i = 1} ^ {d} s _ {i} ^ {2} = \sum_ {i = 1} ^ {d} s _ {i} ^ {2} (1 - b _ {i i}) - \sum_ {i > d} s _ {i} ^ {2} b _ {i i} \geq \sum_ {i = 1} ^ {d} s _ {d} ^ {2} (1 - b _ {i i}) - \sum_ {i > d} s _ {d} ^ {2} b _ {i i} = 0,
$$

which implies that $\mathcal{L}_{\mathrm{N}}(\Psi) \geq -\sum_{i=1}^{d} s_{i}^{2}$ . To attain equality, we will have $b_{ii} = 1$ for $i = 1, \cdots, d$ , and $b_{ii} = 0$ for $i \geq d + 1$ . By Eqn. (6), we can know $\Psi^{*}$ extracts the span of $\nu_{1}, \cdots, \nu_{d}$ , indicating that $\tilde{\Psi}^{*}$ extracts the top-d singular functions of $T_{P^{+}}$ and $\tilde{\Phi}^{*}$ learns the contexture of $P^{+}$ .

![](images/8b18457d82218cfeec8d5a842ecadfc5df3080fafda9d59b0041a3e03a299c5c.jpg)

# A.5 Proof of Theorem 12

Theorem 12. Let $\Phi^{*}$ be any solution to Eqn. (3) (so that for any constant $c$ , $\Phi^{*} + c$ is also a solution). Then, $\tilde{\Phi}^{*}$ learns the contexture of $P^{+}$ .

Proof. Without loss of generality, suppose $\bar{\Phi}=0$ . We have

$$
(T _ {P ^ {+}} f) (u) = \sum_ {v} f (v) \frac {w (u , v)}{d (u)}; \quad \langle T _ {P ^ {+}} f, g \rangle_ {P _ {\mathcal {X}}} = \sum_ {u, v} f (u) g (v) \frac {w (u , v)}{d _ {\mathrm{sum}}} = \langle f, T _ {P ^ {+}} g \rangle_ {P _ {\mathcal {X}}},
$$

which implies that $T_{P^{+}}$ is self-adjoint. Therefore, the eigenfunctions of $T_{P^{+}}$ are the same as those of $T_{P^{+}}^{*}T_{P^{+}}$ , with square root eigenvalues.

For the objective of Eqn. (3), we have

$$
\begin{array}{l} \frac {1}{2} \mathbb {E} _ {(u, v) \sim P _ {w}} \Big [ \| \Phi (u) - \Phi (v) \| _ {2} ^ {2} \Big ] = \underset {(u, v) \sim P _ {w}} {\mathbb {E}} \Big [ \| \Phi (u) \| _ {2} ^ {2} - \langle \Phi (u), \Phi (v) \rangle \Big ] \\ = \sum_ {i = 1} ^ {d} \left(\| \phi_ {i} \| _ {P _ {\mathcal {X}}} ^ {2} - \langle \phi_ {i}, T _ {P +} \phi_ {i} \rangle_ {P _ {\mathcal {X}}}\right) \\ = d - \sum_ {i = 1} ^ {d} \left\langle \phi_ {i}, T _ {P ^ {+}} \phi_ {i} \right\rangle_ {P _ {\mathcal {X}}}. \\ \end{array}
$$

Note that $(u,v)$ and $(v,u)$ can be drawn from $P_{w}$ with equal probability. We conclude that $\Phi$ extracts the top-d eigenfunctions of $T_{P^{+}}$ , which are the same as the top-d eigenfunctions of $T_{P^{+}}^{*}T_{P^{+}}$ , or the top-d singular functions of $T_{P^{+}}$ . This implies that $\tilde{\Phi}$ learns the contexture of $T_{P^{+}}$ .

# B Proofs for Section 4

# B.1 Proof of Theorem 14

Theorem 14. For any $f^{*} \in \mathcal{F}_{\epsilon}(P^{+})$ , there exists a $g^{*} \in L^{2}(P_{\mathcal{A}})$ such that $f^{*}(x) = \mathbb{E}[g^{*}(A) \mid x]$ , and $g^{*}$ satisfies

$$
\underset {X \sim P _ {\mathcal {X}}} {\mathbb {E}} \underset {A, A ^ {\prime} \sim P ^ {+} (\cdot \mid X)} {\mathbb {E}} \left[ \left(g ^ {*} (A) - g ^ {*} \left(A ^ {\prime}\right)\right) ^ {2} \right] \leq 4 \epsilon \| g ^ {*} \| _ {P _ {\mathcal {A}}} ^ {2}. \tag {7}
$$

Proof. Let $g^{*} = \sum s_{i}u_{i}\nu_{i}$ . We have already explained that if $f^{*} \in \mathcal{F}_{\epsilon}(P^{+})$ , i.e., $\mathbb{E}[f^{*}] = 0$ and $\rho(f^{*}, P^{+}) \geq 1 - \epsilon$ , then it must satisfy the condition w.r.t. $g^{*}$ :

$$
\frac {\left\langle f ^ {*} , T _ {P ^ {+}} g ^ {*} \right\rangle_ {P _ {\mathcal {X}}}}{\| f ^ {*} \| _ {P _ {\mathcal {X}}} \| g ^ {*} \| _ {P _ {\mathcal {A}}}} \geq 1 - \epsilon .
$$

For Eqn. (7), we have $P(A'|A = a) = \int P^{+}(A'|X = x)P^{+}(x|A' = a)dx$ , where $P^{+}(x|a) = \frac{P^{+}(a|x)P_{\mathcal{X}}(x)}{P_{\mathcal{A}}(a)}$ by Bayes rule. Then, using Definition 1 we have $P(A'|A = a) = k_{A}^{+}(a,a^{\prime})P_{\mathcal{A}}(a^{\prime})$ , which implies that

$$
\begin{array}{l} \mathbb {E} _ {X \sim P _ {\mathcal {X}}} \mathbb {E} _ {A, A ^ {\prime} \sim P ^ {+} (\cdot | X)} [ g ^ {*} (A) g ^ {*} (A ^ {\prime}) ] = \mathbb {E} _ {A \sim P _ {\mathcal {A}}} \mathbb {E} _ {A ^ {\prime} \sim P (\cdot | A)} [ g ^ {*} (A) g ^ {*} (A ^ {\prime}) ] \\ = \mathbb {E} _ {A} \bigg [ g ^ {*} (A) \int g ^ {*} (a ^ {\prime}) P (a ^ {\prime} | A) d a ^ {\prime} \bigg ] = \mathbb {E} _ {A} \bigg [ g ^ {*} (A) \int g ^ {*} (a ^ {\prime}) k _ {A} ^ {+} (a, a ^ {\prime}) P _ {\mathcal {A}} (a ^ {\prime}) d a ^ {\prime} \bigg ] = \Big \langle g ^ {*}, T _ {k _ {A} ^ {+}} g ^ {*} \Big \rangle_ {P _ {\mathcal {A}}}. \\ \end{array}
$$

Since $T_{k_A^+}g^* = T_{P^+}^*T_{P^+}g^* = \sum s_i^3 u_i\nu_i$ , Eqn. (7) is equivalent to $\sum (s_i^2 - s_i^4)u_i^2 \leq 2\epsilon \sum s_i^2 u_i^2$ . Meanwhile, we have $\sum s_i^2 u_i^2 \geq (1 - \epsilon)^2 \sum u_i^2 \geq (1 - 2\epsilon) \sum u_i^2$ . By Cauchy-Schwarz inequality, we have $(\sum s_i^4 u_i^2)(\sum u_i^2) \geq (\sum s_i^2 u_i^2)^2 \geq (1 - 2\epsilon)(\sum u_i^2)(\sum s_i^2 u_i^2)$ , which proves Eqn. (7).

# B.2 Proof of Theorem 16

Theorem 16. Suppose $1 - s_{1} \leq \epsilon \leq 1 - \sqrt{\frac{s_{1}^{2} + s_{2}^{2}}{2}}$ . For any d, among all $\Phi = [\phi_{1}, \cdots, \phi_{d}]$ where $\phi_{i} \in L^{2}(P_{\mathcal{X}})$ , $\Phi$ minimizes $\text{err}(\Phi; \mathcal{F}_{\epsilon}(P^{+}))$ if and only if it learns the contexture of $T_{P^{+}}$ . The error is given by

$$
\min _ {\Phi : \mathcal {X} \to \mathbb {R} ^ {d}, \phi_ {i} \in L ^ {2} (P _ {\mathcal {X}})} \operatorname{err} \bigl (\Phi ; \mathcal {F} _ {\epsilon} (P ^ {+}) \bigr) = \frac {s _ {1} ^ {2} - (1 - \epsilon) ^ {2}}{s _ {1} ^ {2} - s _ {d + 1} ^ {2}}.
$$

Conversely, for any $d$ -dimensional encoder $\Phi$ and any $\epsilon > 0$ , there exists $f \in L^2(P_{\mathcal{X}})$ such that $\rho(f, P^+) = 1 - \epsilon$ , and $\operatorname{err}(\Phi, f) \geq \frac{s_1^2 - (1 - \epsilon)^2}{s_1^2 - s_{d+1}^2}$ .

Proof. Necessity: Since $\text{span}(\Phi)$ is at most rank-d, thus there exists $f_{1} \in \text{span}\{\mu_{1}, \cdots, \mu_{d+1}\}$ with $\|f_{1}\|_{P_{\mathcal{X}}} = 1$ that is orthogonal to $\text{span}(\Phi)$ . Thus, there exists $f_{1}, f_{2} \in \text{span}\{\mu_{1}, \cdots, \mu_{d+1}\}$ with $\|f_{1}\|_{P_{\mathcal{X}}} = \|f_{2}\|_{P_{\mathcal{X}}} = 1$ , $f_{1}$ is orthogonal to $\text{span}(\Phi)$ and $f_{2} \in \text{span}(\Phi)$ (thus $f_{1} \perp f_{2}$ ), and $\mu_{1} \in \text{span}\{f_{1}, f_{2}\}$ . Suppose $\mu_{1} = \alpha_{1}f_{1} + \alpha_{2}f_{2}$ (without loss of generosity, assuming $\alpha_{1}, \alpha_{2} \in [0, 1]$ ) and denote $f_{0} = \alpha_{2}f_{1} - \alpha_{1}f_{2}$ . Then $\|f_{0}\|_{P_{\mathcal{X}}} = 1$ and $\langle\mu_{1}, f_{0}\rangle_{P_{\mathcal{X}}} = 0$ . Since $f_{1}, f_{2} \in \text{span}\{\mu_{1}, \cdots, \mu_{d+1}\}$ , we have $f_{0} \in \text{span}\{\mu_{2}, \cdots, \mu_{d+1}\}$ and thus $\mathbb{E}[f_{0}] = 0$ .

Consider $f = \beta_{1}\mu_{1} + \beta_{2}f_{0}$ where $\beta_1^2 +\beta_2^2 = 1, \beta_1,\beta_2\in [0,1]$ . Denote $f = \sum_{i\geq 1}u_i\mu_i$ , we can get $\sum_{i}u_{i}^{2} = 1$ and

$$
\beta_ {2} ^ {2} \leq \frac {s _ {1} ^ {2} - (1 - \epsilon) ^ {2}}{s _ {1} ^ {2} - s _ {d + 1} ^ {2}} \implies \sum_ {i > 1} s _ {i} ^ {2} u _ {i} ^ {2} \geq s _ {1} ^ {2} \beta_ {1} ^ {2} + s _ {d + 1} ^ {2} \beta_ {2} ^ {2} = s _ {1} ^ {2} - (s _ {1} ^ {2} - s _ {d + 1} ^ {2}) \beta_ {2} ^ {2} \geq (1 - \epsilon) ^ {2} \sum_ {i} u _ {i} ^ {2}.
$$

Obviously, $f \in \mathcal{F}(P^{+})$ . We have

$$
f = \beta_ {1} \mu_ {1} + \beta_ {2} f _ {0} = \beta_ {1} \left(\alpha_ {1} f _ {1} + \alpha_ {2} f _ {2}\right) + \beta_ {2} \left(\alpha_ {2} f _ {1} - \alpha_ {1} f _ {2}\right) = \left(\alpha_ {1} \beta_ {1} + \alpha_ {2} \beta_ {2}\right) f _ {1} + \left(\alpha_ {2} \beta_ {1} - \alpha_ {1} \beta_ {2}\right) f _ {2}.
$$

By the definition of $f_{1}, f_{2}$ we can know the approximation error for $f$ is $(\alpha_{1}\beta_{1} + \alpha_{2}\beta_{2})^{2}$ . We can show $F(\alpha_{1}) = \alpha_{1}\beta_{1} + \alpha_{2}\beta_{2} = \alpha_{1}\beta_{1} + \sqrt{1 - \alpha_{1}^{2}}\beta_{2}$ ( $\alpha_{1} \in [0,1]$ ) first increases then decreases when $\beta_{1}, \beta_{2} \in [0,1]$ . Thus $F(\alpha_{1})^{2} \geq \min\{F(0)^{2}, F(1)^{2}\} = \min\{\beta_{1}^{2}, \beta_{2}^{2}\}$ . Take $\beta_{2}^{2} = \frac{s_{1}^{2} - (1 - \epsilon)^{2}}{s_{1}^{2} - s_{d+1}^{2}} \leq \frac{1}{2}$ , we can get for $f$ , the approximation error is always at least $\frac{s_1^2 - (1 - \epsilon)^2}{s_1^2 - s_{d+1}^2}$ .

To attain equality, we must have $\sum_{i\geq1}s_{i}^{2}u_{i}^{2}=s_{1}^{2}\beta_{1}^{2}+s_{d+1}^{2}\beta_{2}^{2}$ . This implies that $f_{1}=\mu_{d+1}$ , indicating that $\operatorname{span}(\phi_{1},\cdots,\phi_{d})=\operatorname{span}(\mu_{1},\cdots,\mu_{d})$ . Thus $\Phi$ learns the contexture of $T_{P^{+}}$ .

Furthermore, denote $f_{0}=k_{2}\mu_{2}+\cdots+k_{d+1}\mu_{d+1}$ and consider $f=\beta_{1}\mu_{1}+\beta_{2}f_{0}$ where $\beta_{1}^{2}+\beta_{2}^{2}=1$ , $\beta_{1},\beta_{2}\in[0,1]$ . By the definition of $f_{0}$ and f, we have $\|f\|_{P_{X}}=1$ and $f=\beta_{1}\mu_{1}+\beta_{2}\mu_{2}=\beta_{1}\mu_{1}+\beta_{2}k_{2}\mu_{2}+\cdots+\beta_{2}k_{d+1}\mu_{d+1}$ . Thus

$$
\rho^ {2} (f, P ^ {+}) = s _ {1} ^ {2} \beta_ {1} ^ {2} + \beta_ {2} ^ {2} \sum_ {i = 2} ^ {d + 1} s _ {i} ^ {2} k _ {i} ^ {2} = s _ {1} ^ {2} - \left(s _ {1} ^ {2} - \sum_ {i = 2} ^ {d + 1} s _ {i} ^ {2} k _ {i} ^ {2}\right) \beta_ {2} ^ {2}.
$$

Take

$$
\beta_ {2} ^ {2} = \frac {s _ {1} ^ {2} - (1 - \epsilon) ^ {2}}{s _ {1} ^ {2} - (\sum_ {i = 2} ^ {d + 1} s _ {i} ^ {2} k _ {i} ^ {2})} \leq \frac {s _ {1} ^ {2} - (1 - \epsilon) ^ {2}}{s _ {1} ^ {2} - s _ {2} ^ {2}} \leq \frac {s _ {1} ^ {2} - \frac {s _ {1} ^ {2} + s _ {2} ^ {2}}{2}}{s _ {1} ^ {2} - s _ {2} ^ {2}} = \frac {1}{2},
$$

we have $\rho(f,P^{+})=1-\epsilon$ . Similarly, the approximation error for f is

$$
(\alpha_ {1} \beta_ {1} + \alpha_ {2} \beta_ {2}) ^ {2} \geq \min \{\beta_ {1} ^ {2}, \beta_ {2} ^ {2} \} = \beta_ {2} ^ {2} = \frac {s _ {1} ^ {2} - (1 - \epsilon) ^ {2}}{s _ {1} ^ {2} - (\sum_ {i = 2} ^ {d + 1} s _ {i} ^ {2} k _ {i} ^ {2})} \geq \frac {s _ {1} ^ {2} - (1 - \epsilon) ^ {2}}{s _ {1} ^ {2} - s _ {d + 1} ^ {2}}.
$$

Sufficiency: For any $f \in \mathcal{F}(P^{+})$ with $\|f\|_{P_{\chi}} = 1$ and $E[f] = 0$ , denote $f = \sum_{i \geq 1} u_i \mu_i$ where $\sum_{i \geq 1} u_i^2 = 1$ . Obviously we have $(1 - \epsilon)^2 \leq \sum_{i \geq 1} s_i^2 u_i^2 \leq 1$ . Notice that when $\text{span}(\phi_1, \cdots, \phi_d) = \text{span}(\mu_1, \cdots, \mu_d)$ since $\Phi$ learns the contexture of $T_{P^+}$ , the approximation of f will be $\sum_{i \geq d+1} u_i^2 := A$ . By the given conditions, we have

$$
(1 - \epsilon) ^ {2} \leq \sum_ {i \geq 1} s _ {i} ^ {2} u _ {i} ^ {2} \leq s _ {1} ^ {2} \sum_ {i = 1} ^ {d} u _ {i} ^ {2} + s _ {d + 1} ^ {2} \sum_ {i \geq d + 1} u _ {i} ^ {2} = s _ {1} ^ {2} - (s _ {1} ^ {2} - s _ {d + 1} ^ {2}) A,
$$

and this implies that

$$
A = \min _ {\boldsymbol {w} \in \mathbb {R} ^ {d}, b \in \mathbb {R}} \left\| \boldsymbol {w} ^ {\top} \Phi + b - f \right\| _ {P \mathcal {X}} ^ {2} \leq \frac {s _ {1} ^ {2} - (1 - \epsilon) ^ {2}}{s _ {1} ^ {2} - s _ {d + 1} ^ {2}}.
$$

When $u_1^2 = 1 - \frac{s_1^2 - (1 - \epsilon)^2}{s_1^2 - s_{d + 1}^2}, u_{d + 1}^2 = \frac{s_1^2 - (1 - \epsilon)^2}{s_1^2 - s_{d + 1}^2}$ , the equality holds. Thus, the approximation error reaches its lower bound when $\Phi$ learns the contexture of $T_{P+}$ .

# C Evaluating an Arbitrary Encoder

Given a context that is compatible with the task, the encoder that learns the contexture is optimal. Now, what about an arbitrary encoder $\Phi$ ? Is it possible to bound its worst-case approximation error on the class of compatible tasks? To derive such a bound, two key objects are necessary: the induced RKHS and the ratio trace. They were originally defined in $[ZLR^{+}24]$ for self-supervised learning, and here we extend them to a broader scope.

Denote the range of $T_{P^+}^*$ by $R(T_{P^+}^*) = \{T_{P^+}^* f \mid f \in L^2(P_X)\}$ .

Definition 19. The induced RKHS of $P^{+}$ , denoted by $H_{P^{+}}$ , is the Hilbert space $R(T_{P^{+}}^{*})$ with the inner product given by $\left\langle T_{P^{+}}^{*}f_{1},T_{P^{+}}^{*}f_{2}\right\rangle_{H_{P^{+}}}=\left\langle f_{1},f_{2}\right\rangle_{P_{X}}$ .

An alternative formula is that for any $h_{1}, h_{2} \in H_{P^{+}}$ where $h_{1} = \sum u_{i}\nu_{i}$ and $h_{2} = \sum v_{i}\nu_{i}$ , there is $\langle h_{1}, h_{2} \rangle_{\mathcal{H}_{P^{+}}} = \sum \frac{u_{i}v_{i}}{s_{i}^{2}}$ .

Proposition 20. The induced RKHS $H_{P+}$ has the following properties:

(i) $k_{A}^{+}$ is the reproducing kernel, such that $h(a) = \left\langle h,k_{A}^{+}(a,\cdot)\right\rangle_{\mathcal{H}_{P^{+}}}$ for all $h\in \mathcal{H}_{P^{+}}$

(ii) $\mathcal{H}_{P^{+}}$ is isometric to $\operatorname{span}\{\mu_i : s_i > 0\}$ , which is a subspace of $L^2(P_{\mathcal{X}})$ .

(iii) $f^{*} \in \mathcal{F}_{\epsilon}(P^{+})$ is equivalent to $h^{*} = T_{P^{+}}^{*}f^{*}$ satisfying the following isometry property:

$$
(1 - \epsilon) \left\| \tilde {h} ^ {*} \right\| _ {\mathcal {H} _ {P ^ {+}}} \leq \left\| \tilde {h} ^ {*} \right\| _ {P _ {\mathcal {A}}} \leq \left\| \tilde {h} ^ {*} \right\| _ {\mathcal {H} _ {P ^ {+}}}. \tag {8}
$$

Proof. For any $h \in \mathcal{H}_{P^+}$ where $h = T_{P^+}^* f$ and $f = \sum u_i \mu_i$ , we have

$$
\left<   h, k _ {A} ^ {+} (a, \cdot) \right> _ {\mathcal {H} _ {P ^ {+}}} = \left<   \sum s _ {i} u _ {i} \nu_ {i}, \sum s _ {i} ^ {2} \nu_ {i} (a) \nu_ {i} \right> _ {\mathcal {H} _ {P ^ {+}}} = \sum s _ {i} u _ {i} \nu_ {i} (a) = h (a),
$$

which proves (i). (ii) is obvious. Regarding (iii), recall that $f^{*} = \sum u_{i}\mu_{i}\in \mathcal{F}_{\epsilon}(P^{+})$ is equivalent to $\sum_{i\geq 1}s_i^2 u_i^2\geq (1 - \epsilon)^2\sum_{i\geq 1}u_i^2$ , and this is $\left\| \tilde{h} ^*\right\|_{P_A}\geq (1 - \epsilon)\left\| \tilde{h} ^*\right\|_{\mathcal{H}_{P^+}}$ . Meanwhile, it is obvious that $\left\| \tilde{h} ^*\right\|_{P_A}\leq \left\| \tilde{h} ^*\right\|_{\mathcal{H}_{P^+}}$ always holds.

Definition 21. Define covariance matrices $C_{\Phi} = \operatorname{Cov}_{P_{X}}[\Phi]$ , and $B_{\Phi} = \operatorname{Cov}_{P_{A}}[T_{P^{+}}^{*}\Phi]$ . If $C_{\Phi}$ is invertible, then the ratio trace of $\Phi$ w.r.t. $P^{+}$ is defined as $\operatorname{RT}(\Phi; P^{+}) = \operatorname{RT}(\phi_{1}, \cdots, \phi_{d}; P^{+}) := \operatorname{Tr}(C_{\Phi}^{-1}B_{\Phi})$ ; otherwise, let $\Phi' = [\phi_{i_{1}}, \cdots, \phi_{i_{t}}]$ be the maximal linearly independent subset of $[\phi_{1}, \cdots, \phi_{d}]$ , and define the ratio trace of $\Phi$ the same as the ratio trace of $\Phi'$ .

The ratio trace of any $\Phi$ essentially measures how well $\Phi$ is aligned with the contexture of $P^{+}$ . Multiplying $\Phi$ by any invertible matrix does not change its ratio trace. If $\Phi$ learns the contexture, then its ratio trace is $s_{1}^{2} + \cdots + s_{d}^{2}$ , which can be easily shown by setting $\phi_{i} = \mu_{i}$ . In fact, this is the maximum ratio trace of any d-dimensional encoder.

Lemma 22. Suppose $\phi_{1},\cdots,\phi_{d}$ are orthonormal and all have zero mean. Then, we have

$$
\| T _ {P ^ {+}} ^ {*} \phi_ {1} \| _ {P _ {\mathcal {A}}} ^ {2} + \dots + \| T _ {P ^ {+}} ^ {*} \phi_ {d} \| _ {P _ {\mathcal {A}}} ^ {2} \leq s _ {1} ^ {2} + \dots + s _ {d} ^ {2}.
$$

Proof. Let $\phi_{i}=\sum_{j\geq1}q_{ij}\mu_{j}$ for $i\in[d]$ . Then, $\boldsymbol{Q}=(q_{ij})$ is a matrix with d orthonormal rows and infinitely many columns. It is easy to see that the left-hand side is equal to $\operatorname{Tr}(\boldsymbol{Q}\boldsymbol{D}\boldsymbol{Q}^{\top})$ , where $D=\operatorname{diag}\{s_{1}^{2},s_{2}^{2},\cdots\}$ . Let $q_{j}$ be the j-th column of Q. For all $j\in[d]$ , there is $\sum_{i=1}^{j}q_{i}^{\top}q_{i}\leq j$ ; and for any j>d, there is $\sum_{i=1}^{j}q_{i}^{\top}q_{i}\leq d$ . Thus, using Abel transformation, we have

$$
\mathrm{Tr} (\boldsymbol {Q} \boldsymbol {D} \boldsymbol {Q} ^ {\top}) = \mathrm{Tr} (\boldsymbol {D} \boldsymbol {Q} ^ {\top} \boldsymbol {Q}) = \sum_ {j = 1} ^ {\infty} s _ {j} ^ {2} \boldsymbol {q} _ {j} ^ {\top} \boldsymbol {q} _ {j} = \sum_ {j = 1} ^ {\infty} \left(\sum_ {i = 1} ^ {j} \boldsymbol {q} _ {i} ^ {\top} \boldsymbol {q} _ {i}\right) (s _ {j} ^ {2} - s _ {j + 1} ^ {2}) \leq \sum_ {j = 1} ^ {d} s _ {j} ^ {2},
$$

as desired.

The ratio trace induces a key quantity in the approximation error bound called the trace gap, which reflects the gap between $\Phi$ and the top-d singular functions. The larger the trace gap is, the larger the approximation error will be. A simple definition is $s_{1}^{2} + \cdots + s_{d+1}^{2} - \mathrm{RT}(\Phi; P^{+})$ , whose lower bound $s_{d+1}^{2}$ can be achieved by the top-d singular functions, the optimal encoder. However, there is an issue with this definition. For example, consider an encoder with d = 1000. It learns the top-10 singular functions, but the other 990 dimensions are complete noise that has zero contribution to $\mathrm{RT}(\Phi; P^{+})$ . The approximation error of this encoder should be no higher than that of the top-10 singular functions, because adding more dimensions will never make the approximation error higher. However, if d becomes larger and $\mathrm{RT}(\Phi; P^{+})$ stays the same, then $s_{1}^{2} + \cdots + s_{d+1}^{2} - \mathrm{RT}(\Phi; P^{+})$ will become larger, so this quantity does not correlate with the approximation error in this scenario. The following definition fixes this issue.

Definition 23. For any linearly independent $f_{1}, \cdots, f_{d'} \in L^{2}(P_{\mathcal{X}})$ , denote $F = [f_{1}, \cdots, f_{d'}]$ , $C_{F} = \operatorname{Cov}_{P_{\mathcal{X}}}[F]$ , and $B_{F} = \operatorname{Cov}_{P_{A}}[F]$ . The trace gap of $\Phi$ w.r.t. $P^{+}$ is defined as

$$
\mathrm{TG} (\Phi ; P ^ {+}) := \inf _ {d ^ {\prime} \leq d} \inf _ {f _ {1}, \dots , f _ {d ^ {\prime}}} \big \{s _ {1} ^ {2} + \dots + s _ {d ^ {\prime} + 1} ^ {2} - \mathrm{Tr} (\boldsymbol {C} _ {F} ^ {- 1} \boldsymbol {B} _ {F}) \big \}.
$$

Obviously, this definition of trace gap is upper bounded by $s_{1}^{2} + \cdots + s_{d+1}^{2} - \mathrm{RT}(\Phi; P^{+})$ . It solves the issue in the previous example because having completely noisy dimensions does not affect the trace gap. The following result bounds the approximation error.

Theorem 24. Suppose $\mathrm{TG}(\Phi; P^{+}) < s_{1}^{2}$ , and $\epsilon > 1 - s_{1}$ . Then,

$$
\mathrm{err} (\Phi ; \mathcal {F} _ {\epsilon} (P ^ {+})) \leq \frac {s _ {1} ^ {2} - (1 - \epsilon) ^ {2} + s _ {1} \mathrm{TG} (\Phi ; P ^ {+})}{s _ {1} ^ {2} - \mathrm{TG} (\Phi ; P ^ {+}) ^ {2}}.
$$

Remark. This bound is fairly tight. If $\Phi$ learns the contexture, then by Theorem 16 we have $\text{err}(\Phi; \mathcal{F}_{\epsilon}(P^{+})) = \frac{s_{1}^{2} - (1 - \epsilon)^{2}}{s_{1}^{2} - s_{d+1}^{2}}$ , and $\text{TG}(\Phi; P^{+}) = s_{d+1}$ . Compared to this exact formula, the above upper bound only has an extra $s_{1}\text{TG}(\Phi; P^{+})$ term in the numerator.

Proof. Let $f_{1},\cdots,f_{d'}$ be the functions that minimize $s_{1}^{2}+\cdots+s_{d'+1}^{2}-\operatorname{Tr}(C_{F}^{-1}B_{F})$ . Without loss of generality, assume that $f_{1},\cdots,f_{d'}$ have zero mean and are orthonormal. Let $F=\operatorname{span}\{f_{1},\cdots,f_{d'}\}$ , and $H=\operatorname{span}\left\{T_{P^{+}}^{*}f_{1},\cdots,T_{P^{+}}^{*}f_{d'}\right\}$ . For any $f\in\mathcal{F}_{\epsilon}(P^{+})$ with $\|f\|_{P_{X}}=1$ , let $h=T_{P^{+}}^{*}f\in\mathcal{H}_{P^{+}}$ , and let $f_{F}$ be the projection of f onto F. Since $\operatorname{err}(\Phi;\mathcal{F}_{\epsilon}(P^{+}))$ is upper bounded by $\|f-f_{F}\|_{P_{X}}^{2}$ , it suffices to show that $\|f-f_{F}\|_{P_{X}}^{2}$ is upper bounded by the right-hand side.

Let $\alpha^2 = \| f_{\mathcal{F}}\|_{P_{\mathcal{X}}}^2$ , and $\beta^2 = \| f - f_{\mathcal{F}}\|_{P_{\mathcal{X}}}^2$ , where $\alpha$ and $\beta$ are non-negative. Then, $\alpha^2 + \beta^2 = \| f\|_{P_{\mathcal{X}}}^2 = 1 = \| h\|_{\mathcal{H}_{P^+}}^2$ . The isometry property says that $(1 - \epsilon)^2 (\alpha^2 +\beta^2)\leq \| h\|_{P_A}^2$ . Let $f - f_{\mathcal{F}} = \beta f_0$ where $\| f_0\|_{P_\mathcal{X}} = 1$ . Let $h_\mathcal{F} = T_{P^+}^* h_\mathcal{F}$ and $h_0 = T_{P^+}^* f_0$ . Then, we have $\| h_\mathcal{F}\|_{P_A}^2\leq s_1^2 \| f_\mathcal{F}\|_{P_\mathcal{X}}^2 = s_1^2 \alpha^2$ . Meanwhile, since $f_0$ is orthogonal to $f_{1},\dots ,f_{d^{\prime}}$ , by Lemma 22 we have $\left\| T_{P^+}^* f_0\right\|_{P_A}^2 +\left\| T_{P^+}^* f_1\right\|_{P_A}^2 +\dots +\left\| T_{P^+}^* f_{d'}\right\|_{P_A}^2\leq s_1^2 +\dots +s_{d' + 1}^2$ , which implies that $\left\| T_{P^+}^* f_0\right\|_{P_A}^2\leq s_1^2 +\dots +s_{d' + 1}^2 -\mathrm{Tr}(C_F^{-1}B_F^{-1})$ . Let $\tau = \mathrm{TG}(\Phi ;P^{+})$ . Then, we have

$$
\| h \| _ {P _ {\mathcal {A}}} ^ {2} = \| h _ {\mathcal {F}} + \beta h _ {0} \| _ {P _ {\mathcal {A}}} ^ {2} \leq \| h _ {\mathcal {F}} \| _ {P _ {\mathcal {A}}} ^ {2} + \beta^ {2} \| h _ {0} \| _ {P _ {\mathcal {A}}} ^ {2} + 2 \beta \| h _ {\mathcal {F}} \| _ {P _ {\mathcal {A}}} \| h _ {0} \| _ {P _ {\mathcal {A}}} \leq s _ {1} ^ {2} \alpha^ {2} + \tau^ {2} \beta^ {2} + 2 s _ {1} \tau \alpha \beta .
$$

Thus, we have $(1 - \epsilon)^{2}(\alpha^{2} + \beta^{2})\leq s_{1}^{2}\alpha^{2} + \tau^{2}\beta^{2} + 2s_{1}\tau \alpha \beta$ , which implies that $(s_1^2 -\tau^2)\beta^2\leq [s_1^2 -(1 - \epsilon)^2 ](\alpha^2 +$ $\beta^2) + 2s_1\tau \alpha \beta \leq [s_1^2 -(1 - \epsilon)^2 +s_1\tau ](\alpha^2 +\beta^2)$ , as desired.

Connection to Fisher discriminant analysis. Fisher discriminant analysis $[MRW^{+}99, BA00, LLM04]$ , or more generally linear discriminant analysis (LDA), is a classical method of learning linear classifiers in statistics. Here we show that Fisher discriminant analysis has a strong connection to the contexture theory. Suppose $X \subseteq R^{d_{X}}$ . Fisher discriminant analysis defines the following between-class covariance matrix $S_{B} \in R^{d_{X} \times d_{X}}$ and within-class covariance matrix $S_{W} \in R^{d_{X} \times d_{X}}$ :

$$
\boldsymbol {S} _ {B} = \iint \Big \{(\mathbb {E} [ X \mid A = a _ {1} ] - \mathbb {E} [ X \mid A = a _ {2} ]) (\mathbb {E} [ X \mid A = a _ {1} ] - \mathbb {E} [ X \mid A = a _ {2} ]) ^ {\top} \Big \};
$$

$$
\pmb {S} _ {W} = \int \mathbb {E} _ {P ^ {+}} \Big [ (X - \mathbb {E} [ X \mid A = a ]) (X - \mathbb {E} [ X \mid A = a ]) ^ {\top} \Big | A = a \Big ] d P _ {\mathcal {A}} (a).
$$

In the original formulation of Fisher discriminant analysis, A is the label of X. Here we extend it to a general context variable. Consider a linear encoder $\Phi(x) = Wx$ , where $W \in R^{d \times d_{X}}$ . Then, one solves the following optimization problem to find W:

$$
\underset {\boldsymbol {W} \in \mathbb {R} ^ {d \times d _ {\mathcal {X}}}} {\text {maximize}} J (\boldsymbol {W}) = \operatorname{Tr} \left[ \left(\boldsymbol {W} \boldsymbol {S} _ {B} \boldsymbol {W} ^ {\top}\right) \left(\boldsymbol {W} \boldsymbol {S} _ {W} \boldsymbol {W} ^ {\top}\right) ^ {- 1} \right] \quad \text {s.t.} \quad \boldsymbol {W} \boldsymbol {S} _ {W} \boldsymbol {W} ^ {\top} \text {is invertible.}
$$

Here, $J(\boldsymbol{W})$ is called the Fisher discriminant. Define $\Psi(a)=\mathbb{E}_{P^{+}}[\boldsymbol{W}X|A=a]$ . Then, we can see that

$$
\boldsymbol {W} \boldsymbol {S} _ {B} \boldsymbol {W} ^ {\top} = \iint (\Psi (a _ {1}) - \Psi (a _ {2})) (\Psi (a _ {1}) - \Psi (a _ {2})) ^ {\top} d P _ {\mathcal {A}} (a _ {1}) d P _ {\mathcal {A}} (a _ {2});
$$

$$
\pmb {W} \pmb {S} _ {W} \pmb {W} ^ {\top} = \int \mathbb {E} _ {P ^ {+}} \Big [ (\Phi (X) - \Psi (a)) (\Phi (X) - \Psi (a)) ^ {\top} \Big | A = a \Big ] d P _ {\mathcal {A}} (a).
$$

Let $C_{\Phi} = \mathbb{E}[\tilde{\Phi}(X)\tilde{\Phi}(X)^{\top}]$ and $B_{\Phi} = \mathbb{E}[\tilde{\Psi}(A)\tilde{\Psi}(A)^{\top}]$ . Then, we have

$$
\boldsymbol {W} \boldsymbol {S} _ {B} \boldsymbol {W} ^ {\top} = 2 \left\{\mathbb {E} \left[ \Psi (A) \Psi (A) ^ {\top} \right] - \bar {\Psi} \bar {\Psi} ^ {\top} \right\} = 2 \mathbb {E} \left[ \tilde {\Psi} (A) \tilde {\Psi} (A) ^ {\top} \right] = 2 \boldsymbol {B} _ {\Phi};
$$

$$
\boldsymbol {W} \boldsymbol {S} _ {W} \boldsymbol {W} ^ {\top} = \int \mathbb {E} _ {P ^ {+}} \left[ \Phi (X) \Phi (X) ^ {\top} - \Psi (a) \Psi (a) ^ {\top} \mid A = a \right] d P _ {\mathcal {A}} (a)
$$

$$
= \mathbb {E} \big [ \Phi (X) \Phi (X) ^ {\top} \big ] - \mathbb {E} \big [ \Psi (A) \Psi (A) ^ {\top} \big ]
$$

$$
= \mathbb {E} \left[ \tilde {\Phi} (X) \tilde {\Phi} (X) ^ {\top} \right] - \mathbb {E} \left[ \tilde {\Psi} (A) \tilde {\Psi} (A) ^ {\top} \right] = C _ {\Phi} - B _ {\Phi}.
$$

![](images/f6bc420b9ac727e69264dcd0735eff757f4475a518844e537917f78c1d263c91.jpg)

<details>
<summary>line</summary>

| i   | s²_i (blue solid) | s²_i (red dotted) | s²_i (orange solid) | s²_i (black solid) |
| --- | ----------------- | ----------------- | ------------------- | ------------------ |
| 0   | 1.0               | 1.0               | 1.0                 | 1.0                |
| 50  | ~0.2              | ~0.3              | ~0.1                | ~0.15              |
| 100 | ~0.1              | ~0.15             | ~0.05               | ~0.08              |
| 150 | ~0.05             | ~0.08             | ~0.03               | ~0.04              |
| 200 | ~0.03             | ~0.05             | ~0.02               | ~0.02              |
| 250 | ~0.02             | ~0.03             | ~0.01               | ~0.01              |
</details>

(a) abalone $(n = 4177)$

![](images/c0f1798c7c5b270746ce66c60dbcc4393b2dc3c90a0a97a68ca00282b782004e.jpg)

<details>
<summary>line</summary>

| x    | Blue Solid | Red Dashed |
| ---- | ---------- | ---------- |
| 0    | 1.0        | 1.0        |
| 100  | 0.2        | 0.4        |
| 200  | 0.05       | 0.2        |
</details>

(b) fifa (n = 19178)

![](images/eca9eeee5686836c431031460f319deef1aefe71560b758d29ab38862e4e0497.jpg)

<details>
<summary>line</summary>

| x    | Blue Line | Red Dashed Line | Orange Line |
| ---- | --------- | --------------- | ----------- |
| 0    | 1.0       | 1.0             | 1.0         |
| 50   | 0.6       | 0.7             | 0.5         |
| 100  | 0.3       | 0.5             | 0.2         |
| 150  | 0.2       | 0.4             | 0.1         |
| 200  | 0.1       | 0.3             | 0.05        |
| 250  | 0.05      | 0.2             | 0.02        |
</details>

(c) kings\_county (n = 21613)

![](images/935248683e6d2b9fc2eafee93103ac5daf26af7d5b6863dd96df7d8674e0410a.jpg)  
Figure 7: Estimating the eigenvalues using a random subset of m samples, from n total training samples.

Therefore, $J(\boldsymbol{W}) = 2 \operatorname{Tr}[(\boldsymbol{C}_{\Phi} - \boldsymbol{B}_{\Phi})^{-1}\boldsymbol{B}_{\Phi}]$ , which is very similar to the ratio trace defined in Definition 21. Recall that an encoder that learns the contexture maximizes the ratio trace. A well-known result is that $J(\boldsymbol{W})$ is maximized when W consists of the top-d eigenvectors of $S_{W}^{-1}S_{B}$ . Hence, Fisher discriminant analysis is almost equivalent to contexture learning under the constraint that the encoder must be linear.

# D Efficient Estimation of the Context Usefulness Metric

To estimate the metric $\tau_{d}$ defined in Eqn. (4), it suffices to estimate the top- $d_0$ eigenvalues of the context. This can be efficiently done with the following procedure:

(i) Train an encoder $\Phi$ whose output dimension is at least $d_0$ to learn the contexture with a random subset of $m$ samples.   
(ii) Estimate the covariance matrix $C_{\Phi} \in R^{d \times d} = \operatorname{Cov}_{P_{\mathcal{X}}}[\Phi]$ with Monte Carlo.   
(iii) Estimate $\pmb{B}_{\Phi} \in \mathbb{R}^{d \times d}$ , where $\pmb{B}_{\Phi}[i,j] = \left\langle \tilde{\phi}_i, T_{k_X^+} \tilde{\phi}_j \right\rangle_{P_{\mathcal{X}}}$ , with Monte Carlo.   
(iv) Solve the generalized eigenvalue problem $B_{\Phi}v = \lambda C_{\Phi}v$ . The eigenvalues $\lambda_{1} \geq \cdots \geq \lambda_{d_{0}}$ are estimates of $s_{1}^{2}, \cdots, s_{d_{0}}^{2}$ . Moreover, let $Q = [v_{1}, \cdots, v_{d}]$ where $(v_{i})$ are the orthonormal eigenvectors corresponding to $(\lambda_{i})$ . Let $\Phi^{*}$ be the normalized version of $\tilde{\Phi}Q$ such that each dimension is scaled to unit variance. Then, $\phi_{1}^{*}, \cdots, \phi_{d_{0}}^{*}$ are estimates of $\mu_{1}, \cdots, \mu_{d_{0}}$ .

If we only need to estimate the eigenvalues, then [STWCK05] showed that for any fixed d, the sum $s_{1}^{2} + \cdots + s_{d}^{2}$ can be estimated with low error using $\Theta(d)$ i.i.d. samples. By union bound, all $s_{1}^{2}, \cdots, s_{d_{0}}^{2}$ can be estimated with low error using $m = \Theta(d_{0} \log d_{0})$ i.i.d. samples. However, if we want to estimate the eigenfunctions as well, then usually we need to use the entire training set.

Let us demonstrate this method on 3 real datasets from OpenML [VvRBT13]: abalone, fifa, and kings\_county. We use KNN with $K = 60$ as context, where $\mathcal{A} = \mathcal{X}$ , and $P^{+}(x^{\prime}|x) = K^{-1}$ if $x^{\prime}$ is a $K$ -nearest neighbor of $x$ and 0 otherwise. For this context, we can exactly compute $k_{X}^{+}$ , and thus we can obtain the exact eigenvalues (ground truth) using kernel PCA. Meanwhile, we pretrain $\Phi$ with one of the variational objectives using a random subset of $m$ samples, and estimate the eigenvalues using the post-hoc approach. Then, we compare the estimation with the ground truth.

We use a 2-layer wide Tanh-activated neural network with embedding dimension d = 512 and hidden dimension 20,000 as $\Phi$ . We train the model through non-contrastive learning with the orthonormality constraint implemented by VICReg [BPL22], and AdamW [KB15, LH17] as the optimizer. We vary m and compare the estimated top- $d_{0}$ eigenvalues with the ground truth, where $d_{0} = 256$ . The estimated eigenvalues and the ground truth are plotted in Figure 7. From the plots, we observe that the eigenvalues estimated by our estimation method decay faster than the ground truth, even if the full dataset is used. We hypothesize that the main reason is that even though we use a very wide neural network, its function class is still a subset

<table><tr><td>Dataset</td><td>m = 100</td><td>m = 300</td><td>m = 600</td><td>m = 1000</td><td>m = 2000</td><td>Full dataset</td></tr><tr><td>abalone</td><td>0.157</td><td>0.124</td><td>0.088</td><td>0.104</td><td>0.110</td><td>0.088</td></tr><tr><td>fifa</td><td>0.218</td><td>0.151</td><td>0.137</td><td>0.134</td><td>0.133</td><td>0.131</td></tr><tr><td>kings_county</td><td>0.278</td><td>0.264</td><td>0.190</td><td>0.183</td><td>0.177</td><td>0.177</td></tr></table>

Table 2: Average estimation error of the top-256 eigenvalues.

of $L^{2}(P_{\mathcal{X}})$ . Consequently, the inductive bias of the model architecture has an impact on the encoder, and therefore the learned contexture can be viewed as a mixture of the inductive bias and the original KNN context. This mixture causes the eigenvalues to decay faster, which explains the observation in Figure 7. Another reason is related to optimization. Since the model is non-convex, gradient methods cannot find the minima of the objective.

The average estimation error of the top-256 eigenvalues is reported in Table 2. The error is defined as $\frac{1}{d_0}\sum_{i=1}^{d_0}|\hat{\lambda}_i - s_i^2|$ , where $\hat{\lambda}_i$ is the estimated eigenvalue. The table shows that when $m\in[600,1000]\approx[0.5d_0\log d_0,0.7d_0\log d_0]$ , the performance is comparable to using the full dataset, which verifies the theoretical result of [STWCK05]. The estimation error is not zero even if the full dataset is used due to the aforementioned reasons. In summary, the post-hoc method can estimate the eigenvalues using a small subset of samples, but the estimated eigenvalues decay faster than the ground truth.

# E Scaling Law Experiment Details

Here we provide a more detailed description of the experiment setting in Section 4.2.

Experiment overview. The purpose of this experiment is to examine whether a large neural network can learn the contexture well, and whether scaling up the model size makes the learned representation more aligned to the top-d eigenfunctions. We compare two encoders. The first encoder is obtained via kernel PCA on the dual kernel, so it consists of the exact top-d eigenfunctions. The second encoder is obtained via training a large neural network to optimize an objective that can learn the contexture. Then, we compute the representational alignment of these two encoders. The most classical metric is the canonical-correlation analysis (CCA) metric $R_{CCA}^{2}$ , which is invariant under invertible linear transformations to the encoders. [KNLH19] proposed a variant called linear CKA, which is only invariant under orthogonal transformations. In our setting, since we only care about the span of $\phi_{1},\cdots,\phi_{d}$ , we would like the metric to be invariant under all invertible transformations, which is why we use CCA. In addition, we also use the mutual KNN metric with 10 neighbors proposed by [HCWI24], which measures the intersection over union (IoU) of nearest neighbors between the two representations. This metric is not invariant under invertible linear transformations, so we whiten the two representations such that their covariance matrices are both identities.

Setup. We use the abalone dataset from OpenML, and split the dataset into a pretrain set, a downstream train set and a downstream test set by 70%-15%-15%. We use K-nearest neighbors (KNN) with K = 30 as the context. The embedding dimension is set to d = 128. For the second encoder, we train a fully-connected neural network with Tanh activation and skip connections for a sufficient number of steps with full-batch AdamW, and vary the depth and width of the network so that we can study their effect on the alignment. Here, “depth” refers to the number of hidden layers—for example, a 2-layer neural network has depth 1. For each width and depth, we run the experiments 15 times with different random initializations and report the average alignment.

In our experiments, we observe the dimension collapse problem [JVLT22]—if we set the output dimension of the neural network to be $d$ , then the rank of the learned representation will usually be less than $d$ , meaning that it can only extract the top- $d'$ eigenspace for some $d' < d$ . [JVLT22] proved that the training

dynamics of self-supervised learning can cause this problem, that is, a large neural network trained with a gradient method cannot find the exact minima, but will find a low-rank solution instead.

To fix this issue, we set the output dimension of the neural network to be $d_{1}=512>d$ . After we obtain the $d_{1}$ -dimensional encoder, similar to Appendix D we estimate the matrices $C_{\Phi}$ and $B_{\Phi}$ , and solve the generalized eigenvalue problem $B_{\Phi}v=\lambda C_{\Phi}v$ . Let $V=[v_{1},\cdots,v_{d}]\in R^{d_{1}\times d}$ be the top-d eigenvectors; then, we use $\tilde{\Phi}V$ as the d-dimensional representation. In other words, we use the 128 principal components of the 512-dimensional embedding.

# F More on Context Evaluation

In this section, we offer guidance for practitioners on identifying contexts with weak or strong associations with inputs. We then show that both excessively weak and overly strong associations degrade downstream performance and demonstrate that the proposed quantitative measurements accurately capture association strength in our controlled experiments. Moreover, we provide full experimental results that complements Table 1. Finally, we provide proofs for the lemmas in Section 5.1.

# F.1 Quantitative measurements for level of association

While mutual information captures mutual dependence between random variables, estimating it from samples remains a long-standing challenge as it requires the joint density function to be known [Pan03]. To address this, we propose alternative metrics that are computationally more tractable.

- Decay rate (all association): As shown in the top row of Figure 3, the decay rate of singular values $(s_i)_{i\geq 0}$ reflects the strength of association. To estimate the decay rate $\lambda$ , we assume the singular values decay exponentially and fit the regression model $s_i^2 = \exp (-\lambda i)$ . When $\lambda$ is large, it indicates a fast decay rate and context has low association. Conversely, when $\lambda$ is low, it implies a slow decay and highly associated context.   
- Expected kernel deviation (weak association): Since the kernel values $k_A^+(a, a')$ can be close to 1 for the contexts with low association, we propose using the expected absolute deviation from 1, i.e. $\mathbb{E}_{x, x' \sim P_X} \left[ |k_X^+(x, x') - 1| \right]$ , as a measure to indicate the weak association setting. Given samples $x_1, \cdots, x_n \in \mathcal{X}$ , we use Monte Carlo sampling to approximate it, i.e.

$$
\frac {1}{n ^ {2}} \sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {n} | k _ {X} ^ {+} (x _ {i}, x _ {j}) - 1 |.
$$

\- Lipschitz constant (strong association): We empirically measure the Lipschitz constant of $k_X^+$ for detecting contexts with strong association. Specifically, given samples $x_1, \cdots, x_n \in \mathcal{X}$ , we use the following estimation:

$$
L _ {k _ {X} ^ {+}} = \sup _ {x, y, z \in \mathcal {X}} \frac {| k _ {X} ^ {+} (z , x) - k _ {X} ^ {+} (z , y) |}{| | x - y | | _ {2}} \approx \max _ {1 \leq i <   j \leq n, 1 \leq k \leq n} \frac {| k _ {X} ^ {+} (x _ {k} , x _ {i}) - k _ {X} ^ {+} (x _ {k} , x _ {j}) |}{| | x _ {i} - x _ {j} | | _ {2}}.
$$

We note that the first metric requires estimating singular values $(s_{i}^{2})_{i\geq0}$ and the last two metrics rely on estimating kernel values $k_{X}^{+}(x,x^{\prime})$ . For estimating singular values, we employ the same technique as the task agnostic metric $\tau$ in Eqn. (4). Estimating kernel values, on the other hand, necessitates training an encoder $\Phi$ and approximating $k_{X}^{+}(x,x^{\prime})$ as $\Phi(x)^{\top}\Phi(x^{\prime})$ , which may require a large training set. Thus, decay rate measurements are preferred due to their simpler estimation process.

Additionally, for the non-smooth kernel $k_{X}^{+}$ , we have the following lemma showing that we may have the singular functions could be non-smooth and difficult to estimate.

Lemma 25 (Proof in Appendix F.3). Let the Lipschitz constant for the positive pair kernel $k_X^+$ be $L_{k_X^+} = \sup_{x_1,x_2,x_3\in \mathcal{X}}\frac{|k_X^+(x_3,x_1) - k_X^+(x_3,x_2)|}{||x_1 - x_2||_2}$ and the maximum Lipschitz constant of its eigenfunctions be $L_{\mu} = \max_{i\geq 1}\sup_{x_1,x_2\in \mathcal{X}}\frac{|\mu_i(x_1) - \mu_i(x_2)|}{||x_1 - x_2||_2}$ . Assume that all the eigenfunctions $\mu_i$ are bounded by $c$ , i.e. $\mu_i(x)\leq c$ for all $i > 0$ and $x\in \mathcal{X}$ . Then we have $L_{k_X^+}\leq cL_\mu \sum_{i = 1}s_i^2$ .

Assume that there exists a universal c that bounds all eigenfunctions. For a highly non-smooth kernel $k_{X}^{+}$ with high Lipschiz constant $L_{k_{X}^{+}}$ , the lemma implies that we have either (i) smooth singular functions with large $L_{\mu}$ and slow decay in singular values with small $\sum_{i=1} s_{i}^{2}$ , or (ii) non-smooth singular functions with large $L_{\mu}$ and fast decay in singular values with small $\sum_{i=1} s_{i}^{2}$ . For (i), we need a larger d to approximate the kernel well, which leads to a higher downstream sample complexity. For (ii), the function approximation by neural networks becomes more difficult for non-smooth functions [Yar18].

# F.1.1 Empirical Verification

Setup. We provide empirical evidence showing (1) downstream performance is worse for contexts with weak and strong associations, (2) the proposed quantitative measurements are positively correlated with the association level. To control the level of association, we use RBF kernels and KNN.

For the estimation of kernel value $k_{X}^{+}$ , we use $k_{X}^{+}(x,x^{\prime})=\int\frac{P^{+}(a|x)P^{+}(a|x^{\prime})}{P_{A}(a)}da$ since $P^{+}(a|x)$ can be efficiently computed for these contexts. The decay rate is estimated using a non-linear least squares approach to fit the regression model. For computing the expected kernel deviation, we utilize the entire training set. To estimate the Lipschitz constant, we restrict the sample size to n=1000 for computational efficiency. Other experimental setups are the same as in Section 5.3.

Results. Figure 8 and Figure 9 illustrate the relationship between association level and both the linear probe error $err_{d^{*}}$ and the decay rate $\lambda$ for KNN and RBF contexts, respectively. The results show that $err_{d^{*}}$ increases at both extremes, with most blue curves exhibiting a U-shape. This suggests that both weak and strong association levels lead to higher errors. Additionally, the red curve indicates that the decay rate $\lambda$ increases as the association level strengthens, highlighting a strong correlation between association level and spectral decay, which is effectively captured by the estimated decay rate.

We report the relationship between association level and both the expected kernel deviation and the Lipschitz constant $L_{k_{x}^{+}}$ in Figure 10 for the KNN context and Figure 11 for the RBF context. The results show that contexts with low association exhibit small expected kernel deviations, while those with high association have large Lipschitz constants. These findings align with our theoretical developments in Section 5.1.

# F.2 Proof of Lemma 17

Proof. Define $k_X^{+'}(x,x') = k_X^+(x,x') - 1 = \sum_{i > 0}s_i^2\mu_i(x)\mu_i(x')$ that is the positive pair kernel without the trivial mode $(s_0,\mu_0)$ , where the equality follows the definition of $T_{k_X^+}$ . We also denote the corresponding kernel integral operator as $(T_{k_X^{+'}}g)(x) = \int g(x')k_X^{+'}(x,x')dP_X(x')$ . Then we have

$$
\sum_ {i > 0} s _ {i} ^ {2} = \mathrm{Tr} (T _ {k _ {X} ^ {+ ^ {\prime}}}) = \int k _ {X} ^ {+ ^ {\prime}} (x, x ^ {\prime}) d P _ {\mathcal {X}} (x ^ {\prime}) <   \int \epsilon d P _ {\mathcal {X}} (x ^ {\prime}) = \epsilon .
$$

![](images/8cff8df70dbd5e00d45cbf3e4ff70db71530a9ffee8bb65493b0f1b358403cf8.jpg)

![](images/1eba46c9cbe83b923c6f8a64aca0bc495978d48791ad39cf193ba594476e0bd3.jpg)  
Figure 8: Association level vs $err_{d^{*}}$ and the decay rate $\lambda$ for the KNN context on the 28 datasets. A larger K/N indicates a lower association level, while a smaller K/N corresponds to a higher association level.

# F.3 Proof of Lemma 25

Proof.

$$
\begin{array}{l} L _ {k _ {X} ^ {+}} = \sup _ {x _ {1}, x _ {2}, x _ {3} \in \mathcal {X}} \frac {| k _ {X} ^ {+} (x _ {3} , x _ {1}) - k _ {X} ^ {+} (x _ {3} , x _ {2}) |}{\| x _ {1} - x _ {2} \| _ {2}} \\ = \sup _ {x _ {1}, x _ {2}, x _ {3} \in \mathcal {X}} \frac {| \sum_ {i \geq 0} s _ {i} ^ {2} \mu_ {i} (x _ {3}) \mu_ {i} (x _ {1}) - \sum_ {i \geq 0} s _ {i} ^ {2} \mu_ {i} (x _ {3}) \mu_ {i} (x _ {2}) |}{\| x _ {1} - x _ {2} \| _ {2}} \\ = \sup _ {x _ {1}, x _ {2}, x _ {3} \in \mathcal {X}} \frac {\left| \sum_ {i \geq 0} s _ {i} ^ {2} \mu_ {i} (x _ {3}) \left(\mu_ {i} (x _ {1}) - \mu_ {i} (x _ {2}) \right. \right|}{\left\| x _ {1} - x _ {2} \right\| _ {2}} \\ = \sup _ {x _ {1}, x _ {2}, x _ {3} \in \mathcal {X}} \frac {\left| \sum_ {i > 0} s _ {i} ^ {2} \mu_ {i} (x _ {3}) (\mu_ {i} (x _ {1}) - \mu_ {i} (x _ {2}) \right|}{\| x _ {1} - x _ {2} \| _ {2}} (\mu_ {0} \equiv 1) \\ \leq \sup _ {x _ {1}, x _ {2}, x _ {3} \in \mathcal {X}} \frac {\sum_ {i > 0} s _ {i} ^ {2} \mu_ {i} (x _ {3}) | \mu_ {i} (x _ {1}) - \mu_ {i} (x _ {2}) |}{\| x _ {1} - x _ {2} \| _ {2}} \\ \leq \sup _ {x _ {3} \in \mathcal {X}} \sum_ {i > 0} s _ {i} ^ {2} \mu_ {i} (x _ {3}) \sup _ {x _ {1}, x _ {2} \in \mathcal {X}} \frac {\left| \mu_ {i} (x _ {1}) - \mu_ {i} (x _ {2}) \right|}{\left\| x _ {1} - x _ {2} \right\| _ {2}} \\ \leq \sup _ {x _ {3} \in \mathcal {X}} \sum_ {i > 0} s _ {i} ^ {2} \mu_ {i} (x _ {3}) L _ {\mu} \\ \leq \sup _ {x _ {3} \in \mathcal {X}} c L _ {\mu} \sum_ {i > 0} s _ {i} ^ {2}. \\ \end{array}
$$

![](images/18665277fb85ad2d94b89f0f0a597bad6faecc000575c03fab94b1263be3f3ce.jpg)

Figure 9: Association level vs $err_{d^{*}}$ and the decay rate $\lambda$ for the RBF context on the 28 datasets. A larger $\gamma$ indicates a higher association level, while a smaller $\gamma$ corresponds to a higher association level.   
![](images/ee4da6a5d6711a8aab6aa1f761f1698dcd7003de9e7a0dff1143c8a6f9ca1919.jpg)  
Figure 10: Association level vs the kernel deviation $\mathbb{E}_{x,x^{\prime}\sim P_{\mathcal{X}}}|k_X^+(x,x') - 1|$ and the Lipschitz constant $L_{k_X^+}$ for the KNN context on the 28 datasets. A larger $K / N$ indicates a lower association level, while a smaller $K / N$ corresponds to a higher association level. We multiply $L_{k_X^+}$ by the input feature dimension $d_{feat}$ to normalize the $L_{2}$ distance in the denominator.

![](images/3825157bf13746b8385923603d3a4862d35f5feb92f376bad84c84ce61b73bd3.jpg)  
Figure 11: Association level vs the kernel deviation $\mathbb{E}_{x,x^{\prime}\sim P_{\mathcal{X}}}|k_X^+(x,x') - 1|$ and the Lipschitz constant $L_{k_X^+}$ for the RBF context on the 28 datasets. A larger $\gamma$ indicates a higher association level, while a smaller $\gamma$ corresponds to a higher association level. We multiply $L_{k_X^+}$ by the input feature dimension $d_{feat}$ to normalize the $L_2$ distance in the denominator.