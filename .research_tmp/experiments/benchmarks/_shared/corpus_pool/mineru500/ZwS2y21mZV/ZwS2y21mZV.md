# Approximation Rate of the Transformer Architecture for Sequence Modeling

Haotian Jiang \*

haotian.jiang@cnrsatcreate.sg

Qianxiao Li \*†

qianxiao@nus.edu.sg

# Abstract

The Transformer architecture is widely applied in sequence modeling applications, yet the theoretical understanding of its working principles remains limited. In this work, we investigate the approximation rate results for the Transformer architectures on general sequence to sequence target relationships. We begin by establishing a representation theorem for the target space and introduce a novel notion of complexity measures to construct approximation spaces. These measures encapsulate both pairwise and pointwise interactions among input tokens. Based on this framework, we derive an explicit Jackson-type approximation rate estimate for the Transformer. This rate sheds light on the underlying structural characteristics of the Transformer, thereby delineating the types of sequential relationships they excel in approximating. Notably, our findings on approximation rates facilitate a concrete comparison between the Transformer and traditional sequence modeling approaches, such as recurrent neural networks.

# 1 Introduction

The Transformer architecture, as introduced by Vaswani et al. [30] has become immensely popular in the field of sequence modeling. Variants such as BERT [8] and GPT [6] have achieved excellent performance, becoming the default choices for natural language processing (NLP) problems. Concurrently, Dosovitskiy et al. [11] successfully applied the Transformer to image classification problems by flattening the image into a sequence of patches. Despite its success across various fields of practical application, many theoretical questions remain unanswered. Among these, we focus on two essential questions in this work: firstly, the approximation rate of the Transformer on sequence modeling; secondly, the comparative advantages and disadvantages of the Transformer with recurrent neural networks (RNNs) on different temporal structures.

The concept of Jackson-type approximation rates is derived from the Jackson Theorem for polynomials $[15]$ , and is further elaborated in the work of DeVore $[9]$ for addressing general forward approximation problems. To illustrate this, consider classic polynomial approximation. It involves defining an appropriate approximation space, accompanied by specific complexity measures, precisely the Sobolev space. According to the Jackson theorem, a function with a small Sobolev norm can be efficiently approximated by a polynomial. We aim to establish similar approximation rate results for the Transformer. This identifies the type of targets that the Transformer can efficiently learn. We examine general non-linear sequence-to-sequence target relationships, extending the linear target form explored in previous studies of the approximation results for linear RNNs and linear temporal convolution networks $[24, 16]$ . We introduce a novel notion of complexity, establishing a concrete target space from which approximation rates can be deduced. This enables us to identify the types of sequential relationships that Transformer can approximate efficiently. Based on our theoretical analysis, we identify concrete classes of temporal structures where the Transformer outperforms

the RNNs and vice versa. Our main contributions are summarized as follows: 1. We develop Jackson-type approximation rate results for single-layer Transformer networks with one attention head. Our analysis reveals that the approximation capacity is governed by a low-rank structure within the pairwise coupling of the target's temporal features. Empirical validation confirms that the findings observed under theoretical settings also hold true in practical applications. 2. We conduct a comparative analysis between the Transformer and RNNs, aiming to identify specific types of temporal structures where one model excels or underperforms compared to the other.

# 2 Related work

We first review the approximation results of Transformer networks. The universal approximation property (UAP) of the Transformer architecture is first proved in Yun et al. [33], which is further extended to Transformers with sparse attention matrices [34]. The above universal results are developed for a deep Transformer structure. In contrast, Kajitsuka & Sato [17] applying a similar technique to prove one layer Transformers can achieve UAP by increasing the width. Additionally, Kratsios et al. [19], Edelman et al. [12], Luo et al. [25] considers the UAP of the Transformer in various different settings. Giannou et al. [13] considers a special setting regarding expressiveness, demonstrating that Transformers can represent any computer program. Beyond the UAP results, there have been developments in specific approximation rate results. Gurevych et al. [14] demonstrated the rate of the misclassification probability by considering the approximation of hierarchical composition functions, which are composed of sparse functions. This rate takes into account both the level of composition and the smoothness of the component functions. Bai et al. [1] and Wang & E [31] explore target relationships with certain special structures. Additionally, Takakura & Suzuki [29] developed approximation rates for a function space consisting of infinite-length sequence-to-sequence functions, which is characterized by the smoothness of the functions. Our approximation rate result steps further by considering temporal structures, shedding light on how the Transformer model handles temporal relationships. Apart from the approximation results, numerous intrinsic properties of the Transformer have been investigated. Dong et al. [10] and Bhojanapalli et al. [4] considers the rank structure of the attention matrices. Levine et al. [20] examine the correlation between the dependency of input variables and the depth of the model. The Transformer is a very flexible architecture, such that a special configuration of parameters can emulate other architectures. For example, Cordonnier et al. [7], Li et al. [21] showed that attention layers under certain assumptions can perform convolution operations. However, not all emulations are valid explanations of the working principles of the Transformer. In this context, definitions of complexity measures and the resulting Jackson-type approximation rate estimates provide more insight into the inner workings of the architecture. This is the focus of the current work.

# 3 Sequence modeling as an approximation problem

We first motivate the theoretical settings of Jackson-type approximation rate results by considering classic polynomial approximation. Then, we formulate the Transformer as an instance of such an approximation problem.

Motivation of Jackson-type Approximation Rates Consider two normed vector spaces, X and Y, designated as the input and output spaces, respectively. We define the target space $C \subset Y^{X}$ as a set of mappings from X to Y that we aim to approximate with simpler functions. The hypothesis space is denoted by $H = \bigcup_{m} H^{(m)}$ , where $H^{(m)}$ represents a sequence of hypothesis spaces. Here, m denotes the complexity or the approximation budget of these spaces. Hypothesis space H encompasses the candidate functions used to approximate targets in C. Let $\alpha$ be a constant, we introduce a complexity measure, denoted as $C^{(\alpha)} : C \to R$ , based on the structure of H. The complexity measure is used to construct an approximation space $C^{(\alpha)} := \{H \in C : C^{(\alpha)}(H) < \infty\}$ , which is usually dense in C. Then for any $H \in C^{(\alpha)}$ , the Jackson-type approximation rate is expressed as follows:

$$
\inf _ {\hat {H} \in \mathcal {H} ^ {(m)}} \left\| H - \hat {H} \right\| \leq E (C ^ {(\alpha)} (H), m). \tag {1}
$$

Here, the error bound $E(\cdot, m)$ decreases to zero as $m$ approaches infinity, and the rate of decay is usually called the approximation rate. In this context, $C^{(\alpha)}(H)$ quantifies the complexity of a target

H when approximated using H. A smaller value indicates that the target H can be more efficiently approximated with candidates from H. Consequently, the complexity measure discerns the types of targets that can be efficiently approximated within the given hypothesis space. Notably, different hypothesis spaces typically give rise to different complexity measures and approximation spaces. These variations characterize the approximation capabilities of the hypothesis spaces themselves.

Defining an appropriate approximation space $\mathcal{C}^{(\alpha)}$ is essential. Without specific structures, the general target space $\mathcal{C}$ does not provide any rate results. Opting for an approximation space with defined structures allows for the derivation of approximation rates. Moreover, the property that $\mathcal{C}^{(\alpha)}$ is dense in $\mathcal{C}$ ensures that the restriction to $\mathcal{C}^{(\alpha)}$ is not overly limiting, preserving the necessary expressiveness of the space. To illustrate these concepts, we consider polynomial approximation over the interval $[0,1]$ . In this case we set $\mathcal{X} = [0,1]$ and $\mathcal{Y} = \mathbb{R}$ . The target space $\mathcal{C} = C([0,1])$ is the set of continuous functions defined on $[0,1]$ . The hypothesis space comprises all polynomials, expressed as follows:

$$
\mathcal {H} = \bigcup_ {m \in \mathbb {N}} \mathcal {H} ^ {(m)} = \bigcup_ {m \in \mathbb {N}} \left\{\hat {H} (x) = \sum_ {k = 0} ^ {m - 1} a _ {k} x ^ {k}: a _ {k} \in \mathbb {R} \right\}.
$$

According to the Jackson theorem for [15], the Sobolev norm serves as an appropriate complexity measure, defined as $C^{(\alpha)}(H) = \max_{r=1\ldots\alpha} \| H^{(r)} \|_{\infty}$ . Let $\mathcal{C}^{(\alpha)}$ be the approximation space containing targets with finite complexity measures. Consequently, for $H \in \mathcal{C}^{(\alpha)}$ , we have the following approximate rate:

$$
\inf _ {\hat {H} \in \mathcal {H} ^ {(m)}} \| H - \hat {H} \| \leq \frac {c _ {\alpha}}{m ^ {\alpha}} C (H). \tag {2}
$$

Here, $c_{\alpha}$ is a constant depending only on $\alpha$ . This theorem implies that smooth functions with small Sobolev norms can be efficiently approximated by polynomials. Developing Jackson-type approximation rates for various sequence modeling hypothesis spaces is crucial for understanding their differences. Jackson-type results for RNNs, CNNs, and encoder-decoder hypothesis spaces have been established in Li et al. [24], Jiang et al. [16], Li et al. [23], where complexity measures such as decaying memory and sparsity were found to influence the approximation rates. In Section 4.1, we identify appropriate complexity measures regarding the Transformer and develop corresponding approximation rates. This enables us to discern the essential structures that facilitate efficient approximation using the Transformer. Furthermore, it allows us to understand how and when the Transformer architecture differs from traditional sequence modeling architecture RNNs, which we will discuss in Section 6.

Formulation of Sequence Modeling as Approximation Problems In sequence modeling, we seek to learn relationships between two sequences $x$ and $y$ . Mathematically, we consider an input sequence space

$$
\mathcal {X} = \left\{\boldsymbol {x}: x (s) \in [ 0, 1 ] ^ {d}, \text {   for   all   } s \in [ \tau ] \right\}. \tag {3}
$$

Here, $[\tau] := \{1, \ldots, \tau\}$ and $\tau$ denotes the maximum length of the input, and we focus on the finite setting, where $\tau < \infty$ . Corresponding to each input $x \in X$ is an output sequence y belonging to

$$
\mathcal {Y} = \left\{\boldsymbol {y}: y (s) \in \mathbb {R}, \text {   for   all   } s \in [ \tau ] \right\}. \tag {4}
$$

We use $H := \{H_{t}\}_{t=1}^{\tau}$ to denote the mapping between x and y, such that $y(t) = H_{t}(x)$ for each $t \in [\tau]$ . Define $C(\mathcal{X}, \mathcal{Y})$ to denote the space of continuous mappings between the input and output space. We may regard each $H_{t} : [0, 1]^{d \times \tau} \to R$ as a $\tau$ -variable function, where each variable is a vector in $[0, 1]^{d}$ . Next, we define the Transformer hypothesis space.

The Transformer Hypothesis Space. We consider the following Transformer block retaining most of the important components.

$$
\hat {H} _ {t} (\boldsymbol {x}) = \hat {F} \left(\sum_ {s = 1} ^ {\tau} \sigma [ (W _ {Q} \hat {h} (t)) ^ {\top} W _ {K} \hat {h} (\cdot) ] (s) \cdot W _ {V} \hat {h} (s)\right), \tag {5}
$$

where $\hat{h} = \hat{f}\circ x$ and $\hat{F}:\mathbb{R}^{m_v}\to \mathbb{R},\hat{f}:\mathbb{R}^d\to \mathbb{R}^n$ are two feed-forward networks. The parameter matrices have dimension $W_{Q},W_{K}\in \mathbb{R}^{m_{h}\times n},W_{V}\in \mathbb{R}^{m_{v}\times n}$ . The softmax function is denoted as $\sigma$ , such that $\sigma [\rho (t,\cdot)](s) = \frac{\exp(\rho(t,s))}{\sum_{s'}\exp(\rho(t,s'))}$ . We focus on a simplified architecture: a single-layer

Transformer with one head. In this work, layer normalization and residual connections are not taken into account. While this constitutes a simplified setting intended for theoretical analysis, it's worth noting that the phenomena observed under the theoretical settings also hold true in practical applications, as we have demonstrated in Section 5. The approximation budget of the Transformer depends on several components. We use $\hat{\mathcal{F}}^{(m_{\mathrm{FF}})}$ to denote the class of feed-forward networks used in the Transformer with budget $m_{\mathrm{FF}}$ , which is usually determined by the number of neurons and layers. Let $m = (n, m_h, m_v, m_{\mathrm{FF}})$ denote the overall approximation budget. We use $\mathcal{H}^{(m)}$ to denote the Transformer with complexity $m$ . Then, we define the Transformer hypothesis space by

$$
\mathcal {H} = \bigcup_ {m} \mathcal {H} ^ {(m)}, \quad \mathcal {H} ^ {(m)} = \left\{\hat {\boldsymbol {H}}: \hat {\boldsymbol {H}} \text {   satisfies   Equation   (5)   with   } m \right\}. \tag {6}
$$

# 4 Approximation results

Following the motivation of approximation problems for sequence modeling as discussed in Section 3, this section discusses the approximation rate results for the Transformer. Firstly, we introduce the notion of the permutation equivariance property of the Transformer and discuss the role of position encodings in eliminating it. Next, we define the target space and present the corresponding representation theorem. We then establish the complexity measures necessary to form the approximation space. Our main result Theorem 4.2 presents the approximation rate results.

Permutation Equivariance and Position Encodings Our objective is to approximate a target relationship H where each H belongs to $C(\mathcal{X}, \mathcal{Y})$ . It is important to note that without specific modifications, the Transformer inherently cannot approximate such targets due to its permutation equivariance properties. This property implies that permuting the temporal indices of the input sequence results in a corresponding permutation of the output sequence. More precisely, if p denotes a bijection on $[\tau]$ representing a permutation of $\tau$ objects, a sequence of functions H is considered permutation equivariant if for all bijections p and inputs $x \in X$ , the condition $\boldsymbol{H}(\boldsymbol{x} \circ p) = \boldsymbol{H}(\boldsymbol{x}) \circ p$ holds. The Transformer $\hat{H}$ within the hypothesis space H is indeed permutation equivariant (refer to Appendix A for details). Directly applying the Transformer, therefore, yields permutation equivariant hypotheses, which are inadequate for approximating general sequential relationships that lack this symmetry. In practical applications, incorporating position encodings is a widely adopted approach to counteract permutation equivariance [30]. Various methods exist for embedding positional information. The simplest approach is fixed encoding, which involves mapping $x(t)$ to a higher-dimensional space and then offsetting each $x(t)$ by distinct distances. Formally, with $A \in R^{d' \times d}$ where $d' \geq d$ and a constant or trainable $e(t) \in \mathbb{R}^{d'}$ , position encodings can be expressed as $x(t) \mapsto Ax(t) + e(t)$ . For general purposes, it's sufficient for the encoded input space $\mathcal{X}^{(E)}$ to satisfy the following condition:

$$
\mathcal {X} ^ {(E)} = \left\{\boldsymbol {x}: x (s) \in \mathcal {I} ^ {(s)} \subset \mathbb {R} ^ {d}, \text {   where   } \mathcal {I} ^ {(i)}, \mathcal {I} ^ {(j)} \text {   are   disjoint,   compact   } \forall i, j \in [ \tau ] \right\}. \tag {7}
$$

This ensures that for each input $\boldsymbol{x} = (x(1), \ldots, x(\tau))$ , all tokens $x(i)$ and $x(j)$ are distinct, meaning no two input sequences are temporal permutations of each other. Define the set $I = \bigcup I_s$ to be the range of the inputs. Moving forward, we assume that position encoding has been applied, allowing us to consider $\mathcal{X}^{(E)}$ as the input space. Consequently, we define the target space to be $\mathcal{C} = C(\mathcal{X}^{(E)}, \mathcal{Y})$ , which denotes the space of continuous mappings between $\mathcal{X}^{(E)}$ and Y.

# 4.1 Jackson-type approximation rate

To derive the Jackson-type approximation rate, it is necessary first to define the complexity measures to form an approximation space so that approximation rates can be obtained. We begin with the following representation theorem for the target space.

Representation of the target space C Given that the Transformer inherently captures both pairwise and pointwise relations among input tokens through attention and feed-forward components, respectively, we are motivated to establish the following representation theorem for the target space C. This theorem demonstrates that every target can be exactly expressed in terms of pairwise and pointwise relations.

Theorem 4.1 (Representation of the target space). Consider $d$ -dimensional, length $\tau$ input space $\mathcal{X}^{(E)}$ with position encoding added. Then, for any $\pmb{H} \in C(\mathcal{X}^{(E)}, \mathcal{Y})$ , there exists continuous functions $F \in C([0,1]^n, \mathbb{R})$ , $f \in C(\mathcal{I}, [0,1]^n)$ and $\rho \in C(\mathcal{I} \times \mathcal{I}, \mathbb{R})$ such that for all $t \in [\tau]$ we have

$$
H _ {t} (\boldsymbol {x}) = F \left(\sum_ {s = 1} ^ {\tau} \sigma [ \rho (x (t), x (\cdot)) ] (s) f (x (s))\right), \tag {8}
$$

where $n = 2\tau d + 1$ and $\sigma$ is the softmax function. The proof is presented in Appendix A.2.

We refer to $\rho$ as the temporal coupling component and $F$ and $f$ as the element-wise components. It's important to note that the functions $F$ , $f$ , and $\rho$ may not be uniquely determined. In Appendix B.1, we explore certain invariant properties associated with Equation 8. Additionally, we provide illustrative examples in Appendix B.1 where the target explicitly conforms to Equation 8. Leveraging this representation theorem, we define the following complexity measures for targets $\mathcal{H} \in \mathcal{C}$ .

Temporal Coupling Component Now, we discuss the complexity measures associated with the temporal coupling term $\rho(u,v)$ that is central to understanding the attention mechanism in the Transformer. We employ the proper orthogonal decomposition (POD) [3] to decompose the temporal coupling of $\rho$ . This approach can be viewed as an extension of matrix singular value decomposition (SVD) to functions of two variables. We have the following decomposition: $\rho(u,v) = \sum_{k=1}^{\infty} \sigma_k \phi_k(u) \psi_k(v)$ , where $\phi_k, \psi_k$ are orthonormal bases for $L^2(\mathcal{I})$ , and the singular values $\sigma_k \geq 0$ are arranged in descending order. The bases $\phi_k$ and $\psi_k$ are of optimal choices, ensuring that $\hat{\rho}(u,v)$ satisfies:

$$
\inf _ {\operatorname{rank} (\hat {\rho}) \leq r} \| \rho (u, v) - \hat {\rho} (u, v) \| _ {2} ^ {2} = \sum_ {k = r + 1} ^ {\infty} \sigma_ {k} ^ {2}, \tag {9}
$$

analogous to the Eckart-Young theorem for matrices, with the rank defined as the number of terms in the POD decomposition. This implies that the approximation quality of $\rho(u,v)$ can be measured by the decay rate of its singular values. This motivates our following definition of complexity measure regarding $\rho$ . Let $\alpha > 1/2$ be a constant, and $\{\sigma_{i}^{(\rho)}\}$ be singular values of $\rho$ under POD. We define the complexity measure of H by

$$
C _ {1} ^ {(\alpha)} (\boldsymbol {H}) = \inf _ {F, f, \rho} \inf \left\{c: \sigma_ {s} ^ {(\rho)} \leq c s ^ {- \alpha}, s \geq 1 \right\}, \tag {10}
$$

where the first infimum is taken over all $F, f, \rho$ such that Equation (8) holds. In particular, $C_1^{(g)}(\boldsymbol{H}) < \infty$ if it has a representation in the form (8) with $\rho$ having fast decaying singular values.

Element-wise Component Now, we introduce the complexity measure for approximating the element-wise components $F$ and $f$ . Let $\mathcal{F}^{(m_{\mathrm{FF}})}$ be a hypothesis space comprising feed-forward neural networks with a budget of $m_{\mathrm{FF}}$ , representing parameters such as width or depth. We assume the existence of $\beta > 0$ such that:

$$
\inf _ {\hat {f} \in \mathcal {F} ^ {(m _ {\mathrm{FF}})}} \left\| f - \hat {f} \right\| \leq \frac {C _ {\mathrm{FF}} ^ {(\beta)} (f)}{m _ {\mathrm{FF}} ^ {\beta}}. \tag {11}
$$

Here, $C_{FF}$ represents a complexity measure of f corresponding to its approximation by $\mathcal{F}(m_{\mathrm{FF}})$ . It is essential to emphasize that we are assuming the existence of pre-existing approximation rate results for the feed-forward component. For instance, in Barron [2], $\mathcal{F}(m_{\mathrm{FF}})$ is considered to be one-layer neural networks with sigmoidal activation, where $m_{FF}$ corresponds to the width of the network. By adopting this result, we have $\beta = 1/2$ and $C_{\mathrm{FF}}^{(\beta)}(f)$ is a moment of the Fourier transform of f. For shallow ReLU networks commonly used in Transformers, Klusowski & Barron [18] demonstrate $\beta = 1/2 + 1/n$ , where n is the input dimension of the neural network. We maintain the generality of Equation (11). This enables us to substitute any other relevant estimates from the mentioned references. This flexibility allows for a broader application of the complexity measure in different scenarios and settings. Next, we proceed to define the complexity measure for $H \in H$ . This measure considers all the components that need to be approximated using the feed-forward network.

$$
C _ {2} ^ {(\beta)} (\boldsymbol {H}, k) = \inf _ {F, f, \rho} \left(C _ {\mathrm{FF}} ^ {(\beta)} (F) + C _ {\mathrm{FF}} ^ {(\beta)} (f) + C _ {\mathrm{FF}} ^ {(\beta)} (\rho , k)\right), \tag {12}
$$

where $C_{\mathrm{FF}}^{(\beta)}(\rho,k)=\sum_{i=1}^{k}(C_{\mathrm{FF}}^{(\beta)}(\phi_{i})+C_{\mathrm{FF}}^{(\beta)}(\psi_{i}))$ considers the approximation of the approximation of the POD bases $\phi_{i}$ and $\psi_{i}$ for the target function $\rho$ . Notably, this complexity measure is dependent on a parameter k, which determines the number of bases we want to consider in the approximation of $\rho$ . Finally, we define a complexity measure considering the norm of the target.

$$
C _ {0} (\boldsymbol {H}) = \inf _ {F, f, \rho} \left\{K _ {F} \| f \| _ {\infty} \left(\sup _ {i} \| \psi_ {i} \| _ {\infty} + \| \phi_ {i} \| _ {\infty}\right), 1 \right\}, \tag {13}
$$

where $K_{F}$ is the Lipschitz constant of F and $\phi_{i}, \psi_{i}$ are POD bases of $\rho$ .

Approximation Rates Combining the complexity measures Equations (10) and (13) and Equation (12) discussed above, we define the approximation spaces, which consists of targets that have finite complexity measures:

$$
\mathcal {C} ^ {(\alpha , \beta)} = \{\boldsymbol {H} \in \mathcal {C}: C _ {0} (\boldsymbol {H}) + C _ {1} ^ {(\alpha)} (\boldsymbol {H}) + C _ {2} ^ {(\beta)} (\boldsymbol {H}, k) <   \infty , k \geq 1 \}. \tag {14}
$$

One can understand this as an analog of the classical Sobolev spaces for polynomial approximation but adapted to the Transformer hypothesis space. Note that $\mathcal{C}^{(\alpha, \beta)}$ is also dense in general continuous target space $C(\mathcal{X}^{(E)}, \mathcal{Y})$ when $n$ sufficiently large (See Appendix A.2). We are now ready to present the main result of this paper.

Theorem 4.2 (Jackson-type approximation rates for the Transformer). Consider sequences with a fixed length $\tau$ . Suppose the target $\pmb{H} \in \mathcal{C}^{(\alpha, \beta)}$ has a representation in the form Equation (8) with $F \in C([0,1]^{n'}, \mathbb{R})$ and $f \in C(\mathcal{I}, [0,1]^{n'})$ . Let the hidden dimension of the Transformer be $n = 2 * m_h + m_v$ , with $m_v \geq n'$ . Then, we have:

$$
\inf _ {\hat {\boldsymbol {H}} \in \mathcal {H} ^ {(m)}} \int_ {\mathcal {I}} \sum_ {t} ^ {\tau} \left| H _ {t} (\boldsymbol {x}) - \hat {H} _ {t} (\boldsymbol {x}) \right| d \boldsymbol {x} \leq \tau^ {2} C _ {0} (\boldsymbol {H}) \left(\frac {C _ {1} ^ {(\alpha)} (\boldsymbol {H})}{m _ {h} ^ {2 \alpha - 1}} + \frac {C _ {2} ^ {(\beta)} (\boldsymbol {H} , m _ {h})}{m _ {\mathrm{FF}} ^ {\beta}} \cdot (m _ {h}) ^ {\beta + 1}\right),
$$

where $m = (n, m_h, m_v, m_{\mathrm{FF}})$ is the approximation budget, and $C_0, C_1^{(\alpha)}C_2^{(\beta)}$ are complexity measures of $\mathbf{H}$ defined in (13), (10) and (12), respectively. The proof is presented in Appendix A.3.

Here, $m_{h}$ denotes the hidden dimension of the attention mechanism, essentially the size of the query and key vectors, while $m_{FF}$ represents the complexity measure of the pointwise feed-forward network employed in the Transformer. We first consider how the attention mechanism $\hat{\rho}(x(t), x(s))$ approximates the temporal coupling term $\rho$ . By setting $[W_{Q}]_{k} = e_{k}$ and $[W_{K}]_{k} = e_{k+m_{h}+1}$ , we can write $\hat{\rho}$ into $\hat{\rho}(x(t), x(s)) = (W_{Q}\hat{f}(x(t)))^{\top}W_{K}\hat{f}(x(s)) = \sum_{k=1}^{m_{h}}\hat{\phi}_{k}(x(t))\hat{\psi}_{k}(x(s))$ , where $\hat{\phi}_{k}, \hat{\psi}_{k} : I \to R$ for $k \in [m_{h}]$ are components of $\hat{f}$ and are represented by the feed-forward network. This suggests that the approximation of $\rho$ by $\hat{\rho}$ is a low-rank approximation as discussed in Equation (9). When we increase $m_{h}$ , the first term in the error rate that considers the POD decomposition decreases since there are more basis functions included. However, in scenarios where $m_{FF}$ remains unchanged, the second term in the error bound will increase. It is important to highlight that this error increment pertains only to the error bound, not to the best approximation error, which does not necessarily become worse when increasing the approximation budget. When $m_{h}$ increases, there are more basis functions that need to be approximated by the feed-forward components; thus, the error bound converges when both $m_{h} \to \infty$ and $m_{FF}^{\beta}/m_{h}^{\beta+1} \to \infty$ . In Appendix B.2, we provide a synthetic example to illustrate the above discussion.

The complexity measure $C_{2}(\cdot)$ accounts for the quality of approximation using feed-forward networks. On the other hand, $C_{1}(\cdot)$ is the most interesting part, which concerns the internal structure of the attention mechanism. It tells us that if a target can be written in form (8) with $\rho(u,v)$ , then it can be efficiently approximated with small $m_{h}$ if $\rho(u,v)$ has fast decaying singular values. This decay condition can be understood as effectively a low-rank condition on $(u,v)\mapsto\rho(u,v)$ , analogous to the familiar concept for low-rank approximations of matrices. These observations provide important insights into the structure, bias, and limitations of the Transformer.

# 5 Numerical Demonstrations

In this section, we present numerical examples to demonstrate our approximation rate results. We begin with synthetic examples, where we can specify the singular value decay patterns, thus validating

the Jackson-type approximation rates in Theorem 4.2. Then, we turn to a practical example involving a Vision Transformer (ViT) model applied to the CIFAR10 dataset, where we do not have direct access to the singular values. We will discuss methods to estimate the singular value decaying pattern.

# 5.1 Practical Example

We next analyze a practical example, focusing on the Vision Transformer (ViT) model with the CIFAR10 dataset. In this scenario, we do have direct access to the temporal coupling term $\rho$ of the target relationship. However, we demonstrate that as we train a sequence of models with increasing attention dimension $m_{h}$ , the singular values of $\hat{\rho}$ converges, implying the decaying pattern of $\rho$ . Our first step is to estimate the rank of $\hat{\rho}$ from sampled data. Subsequently, we examine the singular value decay pattern, and the error changes when altering the attention head size $m_{h}$ .

![](images/5530b1c24b5597c214e45670bf7d4a9a2e4af7d55534c2686c494c6a2144d939.jpg)

<details>
<summary>line</summary>

| k  | σ_k  |
|----|------|
| 0  | 2.5  |
| 1  | 1.0  |
| 2  | 0.7  |
| 3  | 0.5  |
| 4  | 0.4  |
| 5  | 0.3  |
| 6  | 0.25 |
| 7  | 0.2  |
| 8  | 0.15 |
| 9  | 0.1  |
| 10 | 0.08 |
| 11 | 0.05 |
| 12 | 0.03 |
| 13 | 0.02 |
| 14 | 0.01 |
</details>

(a) Estimated singular values

![](images/a30476e15614534a5acda2a2c551fb975a14e4aae626a5ddcade4ccfde2d7449.jpg)

<details>
<summary>line</summary>

| k   | m_h=1  | m_h=3  | m_h=50 | m_h=55 | m_h=64 |
| --- | ------ | ------ | ------ | ------ | ------ |
| 0   | 0.22   | 0.32   | 0.28   | 0.27   | 0.26   |
| 1   | 0.01   | 0.08   | 0.09   | 0.09   | 0.09   |
| 2   | 0.00   | 0.03   | 0.04   | 0.04   | 0.04   |
| 3   | 0.00   | 0.01   | 0.02   | 0.02   | 0.02   |
| 4   | 0.00   | 0.00   | 0.01   | 0.01   | 0.01   |
| 5   | 0.00   | 0.00   | 0.01   | 0.01   | 0.01   |
| 6   | 0.00   | 0.00   | 0.01   | 0.01   | 0.01   |
| 7   | 0.00   | 0.00   | 0.01   | 0.01   | 0.01   |
| 8   | 0.00   | 0.00   | 0.01   | 0.01   | 0.01   |
| 9   | 0.00   | 0.00   | 0.01   | 0.01   | 0.01   |
| 10  | 0.00   | 0.00   | 0.01   | 0.01   | 0.01   |
| 11  | 0.00   | 0.00   | 0.01   | 0.01   | 0.01   |
| 12  | 0.00   | 0.00   | 0.01   | 0.01   | 0.01   |
| 13  | 0.00   | 0.00   | 0.01   | 0.01   | 0.01   |
| 14  | 0.00   | 0.00   | 0.01   | 0.01   | 0.01   |
| 15  | 0.00   | 0.00   | 0.01   | 0.01   | 0.01   |
| 16  | 0.00   | 0.00   | 0.01   | 0.01   | 0.01   |
| 17  | 0.00   | 0.00   | 0.01   | 0.01   | 0.01   |
| 18  | 0.00   | 0.00   | 0.01   | 0.01   | 0.01   |
| 19  | 0.00   | 0.00   | 0.01   | 0.01   | 0.01   |
| 20+ | ~-     | ~-     | ~-     | ~-     | ~-     |
</details>

(b) Singular values for different $m_h$

![](images/f453cd08630bef0b5003cb9e500664ee6495f29879dbcc8b3938469d738de74a.jpg)

<details>
<summary>line</summary>

| mh  | Error  | Fitted |
| --- | ------ | ------ |
| 0   | 0.18   | 0.18   |
| 20  | 0.14   | 0.14   |
| 40  | 0.135  | 0.135  |
| 60  | 0.13   | 0.13   |
</details>

(c) Training Errors   
Figure 1: (a) is the estimated singular value of the attention matrix over a set of inputs for $m_{h} = 64$ . The violin plot shows the distribution of each singular value. (b) plots the estimated singular values for models with different $m_{h}$ . (c) plots the training error against $m_{h}$ .

Estimate the Rank of $\hat{\rho}$ Given a trained model, it is hard to directly compute the rank of $\hat{\rho}(u,v)$ . Instead, we examine the attention matrix $\hat{\rho}(\boldsymbol{x})\in\mathbb{R}^{\tau\times\tau}$ where $[\hat{\rho}(\boldsymbol{x})]_{t,s}=\rho(x(t),x(s))$ . This is essentially a sample from $\hat{\rho}$ . By analyzing the rank of these sampled matrices, we can estimate the rank of $\hat{\rho}$ . According to Braun [5], the singular values of the matrix $\hat{\rho}(\boldsymbol{x})$ exhibit the same decay pattern as those of $\hat{\rho}(u,v)$ . Consequently, we can approximate the singular values of $\hat{\rho}(u,v)$ by averaging the singular values of $\hat{\rho}(\boldsymbol{x})$ across various inputs. In Figure 1(a), we numerically estimate the singular values of $\hat{\rho}$ in the ViT-B\_16 model [11]. We observe that the singular values tend to be more concentrated, suggesting that we can effectively estimate the rank of $\hat{\rho}(u,v)$ by evaluating it at sampled inputs.

We next estimate the singular value decaying pattern of the temporal coupling term $\rho$ in the target. In Figure 1(b), we analyze the singular values of the $\hat{\rho}$ for ViT models with varying values of $m_{h}$ . We estimate the singular values by averaging over a set of inputs. We observe that as $m_{h}$ increases and reaches a sufficiently large value, the decaying pattern of the singular values starts to converge. As an example, Figure 1(b) plots the estimated singular values for the first head of the last layer for models with different $m_{h}$ . This convergence suggests that the rank of the attention matrix becomes representative of the actual rank of $\rho$ in the target. Consequently, it suggests the presence of a low-rank structure in real-world datasets. In Figure 1(c), we plot the training error as an estimation of the approximation error. The plot reveals that the error decreases as $m_{h}$ increases, following a power law decaying pattern $O\left(1/m_{h}^{0.27}\right)$ . This indicates that the target indeed exhibits a low-rank structure, and the pattern of error decay aligns with our approximation rate presented in Theorem 4.2. This illustrates that while our theorem is formulated based on a simplified scenario, the phenomenon of low rank is also observable in real-world datasets and models.

# 6 Comparison with RNN

Based on the approximation results in Theorem 4.2, this section presents a comparison between the Transformer and RNN. Our comparison centers on how each model is affected by the alterations in the temporal structures of sequential relationships. We primarily investigate two distinct temporal

structures: temporal ordering and temporal mixing, as they have varying impacts on the performance of each architecture. Our approach involves manipulating the temporal structures within the sequential relationships and evaluating how each architecture is affected by these changes. Proof are presented in Appendix D.2 and Appendix D.3.

# 6.1 Temporal Ordering Structure

We first analyze how the Transformer and RNN handle the change in temporal ordering of the sequential relationship. Empirically, it is observed that in certain contexts, the ordering of inputs does not significantly impact the relationships. For instance, in NLP applications, altering the word order in a sentence often does not drastically change its meaning. Similarly, in the ViT model, the arrangement of image patches typically does not substantially affect the outcome. However, in specific applications such as time series analysis, temporal ordering plays a crucial role, as the target relationships are governed by the ordering of the sequence. To alter the temporal order of target H, we apply a fixed permutation p and define the new target as $\tilde{H}_{t}(\boldsymbol{x} \circ p) = H_{t}(\boldsymbol{x})$ . This permutes the input but keeps the output unchanged, resulting in a change to the temporal ordering.

We start by considering the RNN, which is affected by the change in temporal ordering. As demonstrated in Li et al. [22], a linear RNN is represented by the form $\hat{H}_{t}(\boldsymbol{x}) = \sum_{s} c^{\top} e^{W s} U x(t - s)$ . When we employ it to approximate linear targets represented by $H_{t}(\boldsymbol{x}) = \sum_{s} \rho(s) x(t - s)$ , the complexity measures of the RNN $C_{\mathrm{RNN}}(\boldsymbol{H})$ is determined by both decay speed and magnitude of $\rho(s)$ . We use $C_{RNN}$ to denote the approximation space for the RNN. (See Appendix D.1). We next show that the RNN is affected by the change in temporal ordering.

Proposition 6.1. Let $H \in C_{RNN}$ and p be a fixed permutation, such that there exists $t'$ with $p(t') > t'$ . Suppose $\tilde{H}$ is defined by $\tilde{H}_{t}(\boldsymbol{x} \circ p) = H_{t}(\boldsymbol{x})$ . Then $\tilde{H} \notin C_{RNN}$ .

This proposition shows that the altered target $\tilde{H}$ no longer belongs to the approximation space for RNN. This lies in the fact that RNN can only handle causal targets, where $y(t)$ does not depend on future inputs. However, the permuted target $\tilde{x}$ is no longer causal, making the RNN incapable of learning such relationships. We next show that the Transformer, in contrast, remains unaffected by changes in temporal ordering.

Proposition 6.2. Let $\pmb{H} \in \mathcal{C}^{(\alpha, \beta)}$ and $p$ be a fixed permutation. Suppose $\tilde{\pmb{H}}$ is defined by $\tilde{H}_t(\pmb{x} \circ p) = H_t(\pmb{x})$ . Then $\tilde{\pmb{H}}$ have same complexity measures with $\pmb{H}$ for the complexity measures $C_0, C_1, C_2$ defined in Equation (13),(10) and (12).

This proposition shows that the altered target $\tilde{H}$ maintains the same complexity measures as the original target. This observation implies that the Transformer's approximation capability is not affected by alterations in temporal ordering. This point is further substantiated by our testing of the Transformer on real-world datasets, as illustrated in Table 1. We consider the ViT model on the CIFAR10 dataset and the base Transformer structure [30] on the WMT2014 English-German dataset. To alter the temporal ordering of the target relationship, we fix a permutation of indices denoted as $p$ and apply it to all inputs while keeping the output unchanged. The experimental results provide evidence that the performance of the Transformer is unaffected by the temporal ordering of the target relationships.

<table><tr><td></td><td>CIFAR10 (Acc.)</td><td>ENG-DE (BLUE)</td></tr><tr><td>Original</td><td>0.98</td><td>26.85</td></tr><tr><td>Altered</td><td>0.96</td><td>25.91</td></tr></table>

Table 1: Numerical results of the Transformer on original and altered targets. The altered target is constructed by permuting the entire input dataset while keeping the output unchanged.

# 6.2 Temporal Mixing Structure

In this section, we explore how mixing elements from different time indices can affect the performance of the Transformer and RNN. Temporal mixing refers to the idea of blending information from various time indices, often through operations like convolution, which can alter the temporal structure. To

illustrate, consider a linear relationship represented as $H_{t}(\boldsymbol{x}) = \sum_{s} \rho(s)x(t - s)$ . Now, imagine we apply a weighted sum of the input sequence x using a filter $\theta$ to get an altered input. We denote this operation as $\tilde{x} = \theta * x$ , where $(\theta * x)[t] = \sum_{s=0}^{l-1} \theta(s)x(t + s)$ . This mixes the information in the sequence from different time indices. In this case, we define the altered target as $\tilde{H}_{t}(\boldsymbol{x}) = H_{t}(\theta * \boldsymbol{x}) = \sum_{s} \tilde{\rho}(s)x(t - s)$ , where $\tilde{\rho} = \theta * \rho$ is the altered kernel (See Appendix D.3). This scenario often arises in signal processing and data analysis. For example, when dealing with a noisy input signal x, one approach is to apply a moving average filter to smooth it out. This filtering process involves mixing information from different time indices, which can significantly affect the behavior of target relationships. In the following sections, we will explore how such temporal mixing affects the temporal structures in sequential relationships. We begin by examining the linear RNN.

Proposition 6.3. Let $\pmb{H} \in \mathcal{C}_{RNN}$ associated with representation $\rho$ , such that $|\rho(s)| \leq e^{-\frac{s}{\gamma}}$ for some $\gamma > 0$ . Let $\theta$ be a length $l$ filter such that $\| \theta \|_1 \leq 1$ . Suppose $\tilde{\pmb{H}}$ is defined by $\tilde{H}_t(\pmb{x}) = H_t(\theta * \pmb{x})$ . Then we have both $C_{RNN}(\pmb{H}) \leq \gamma$ and $C_{RNN}(\tilde{\pmb{H}}) \leq \gamma$ , where $C_{RNN}$ is the complexity measure of the RNN (See Equation (75)).

This proposition shows that under temporal mixing $\theta$ with certain conditions, the complexity measure of the altered target $\tilde{H}$ does not worsen. As a result, the performance of the RNN is unaffected in such cases. However, performance of the Transformer can be impacted by temporal mixing in the target relationship. Consider $\boldsymbol{H} \in \mathcal{C}^{(\alpha,\beta)}$ and an altered target $\tilde{H}_{t}(\boldsymbol{x}) = H_{t}(\theta * \boldsymbol{x})$ . This alteration can affect the complexity measures for $\tilde{F}$ , $\tilde{f}$ and $\tilde{\rho}$ in $\tilde{H}$ . In Appendix D.3, we present an example where the rank of $\tilde{\rho}$ increases compared to $\rho$ , leading to a performance drop in the Transformer. Numerical results in Table 1 also indicate that temporal mixing influences the Transformer's approximation capability. Preprocessing the input to mitigate this temporal mixing could potentially enhance performance, we leave this as a future direction. The following numerical examples demonstrate the above discussions. We conduct numerical experiments to substantiate the discussions above. These experiments focus on a linear target relationship defined as $H_{t}(\boldsymbol{x}) = \sum_{s} e^{-s} x(t - s)$ . To manipulate the temporal ordering, we apply permute the function $e^{-s}$ . Additionally, for introducing temporal mixing, the input is convolved with a randomly generated filter. Both the RNN and the Transformer are employed to learn these targets. Detailed experimental settings are discussed in the Appendix C. The results are presented in Table 2. The bold font indicates a performance drop in the architecture under the corresponding modification of temporal structures. It is observed that the performance of the Transformer remains unaffected by changes in the temporal ordering structure; however, it is impacted by temporal mixing. In contrast, the RNN exhibits opposite behaviors. This highlights that neither architecture consistently outperforms the other, as they each adapt to different types of temporal structures.

<table><tr><td rowspan="2"></td><td colspan="2">Temporal Ordering</td><td colspan="2">Temporal Mixing</td></tr><tr><td>RNN</td><td>Trans.</td><td>RNN</td><td>Trans.</td></tr><tr><td>Original</td><td>1.02e-7</td><td>2.18e-5</td><td>1.02e-7</td><td>2.18e-5</td></tr><tr><td>Altered</td><td>3.57e-2</td><td>2.51e-5</td><td>1.58e-7</td><td>2.39e-4</td></tr></table>

Table 2: The table presents the MSE values for both RNN and the Transformer under different alterations of temporal structures.

# 7 Conclusion

In this paper, we have developed Jackson-type approximation rates for the Transformer in a simplified setting. An important outcome of our work is the identification of complexity measures and approximation space defined in Equation (14). This, in turn, enables us to derive explicit Jackson-type approximation rate results in Theorem 4.2. Our rate results suggest that the Transformer performs well when the temporal coupling of the target exhibits a low-rank pattern. The experiments presented in Section 5 showcase the existence of low-rank patterns in real-world applications. Furthermore, the comparisons with RNNs underscore the specific temporal structures that each model handles efficiently. Future research directions involve extending the analysis to multi-headed attention and deeper Transformers. Additionally, we aim to investigate the potential benefits of removing temporal mixing in the input to enhance the performance of the Transformer.

# Acknowledgement

This research is part of the programme DesCartes and is supported by the National Research Foundation, Prime Minister's Office, Singapore under its Campus for Research Excellence and Technological Enterprise (CREATE) programme. QL acknowledges support by the National Research Foundation, Singapore, under the NRF fellowship (project No. NRF-NRFF13-2021-0005).

# References

[1] Bai, Y., Chen, F., Wang, H., Xiong, C., and Mei, S. Transformers as Statisticians: Provable In-Context Learning with In-Context Algorithm Selection, July 2023.   
[2] Barron, A. R. Approximation and estimation bounds for artificial neural networks. Machine Learning, 14(1):115–133, January 1994. ISSN 1573-0565. doi: 10.1007/BF00993164.   
[3] Berkooz, G., Holmes, P., and Lumley, J. L. The Proper Orthogonal Decomposition in the Analysis of Turbulent Flows. Annual Review of Fluid Mechanics, 25(1):539–575, 1993. doi:10.1146/annurev.fl.25.010193.002543.   
[4] Bhojanapalli, S., Yun, C., Rawat, A. S., Reddi, S. J., and Kumar, S. Low-Rank Bottleneck in Multi-head Attention Models, February 2020.   
[5] Braun, M. Spectral properties of the kernel matrix and their relation to kernel methods in machine learning. 2005.   
[6] Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D. Language Models are Few-Shot Learners, July 2020.   
[7] Cordonnier, J.-B., Loukas, A., and Jaggi, M. On the Relationship between Self-Attention and Convolutional Layers. In arXiv:1911.03584 [Cs, Stat], January 2020.   
[8] Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805 [cs], May 2019.   
[9] DeVore, R. A. Nonlinear approximation. Acta Numerica, 7:51–150, January 1998. ISSN 1474-0508, 0962-4929. doi: 10.1017/S0962492900002816.   
[10] Dong, Y., Cordonnier, J.-B., and Loukas, A. Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth. arXiv:2103.03404 [cs], March 2021.   
[11] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In International Conference on Learning Representations, September 2020.   
[12] Edelman, B. L., Goel, S., Kakade, S., and Zhang, C. Inductive Biases and Variable Creation in Self-Attention Mechanisms, June 2022.   
[13] Giannou, A., Rajput, S., Sohn, J.-Y., Lee, K., Lee, J. D., and Papailiopoulos, D. Looped Transformers as Programmable Computers. In Proceedings of the 40th International Conference on Machine Learning, pp. 11398–11442. PMLR, July 2023.   
[14] Gurevych, I., Kohler, M., and Sahin, G. G. On the rate of convergence of a classifier based on a Transformer encoder, November 2021.   
[15] Jackson, D. The Theory of Approximation. American Mathematical Soc., December 1930. ISBN 978-0-8218-3892-1.   
[16] Jiang, H., Li, Z., and Li, Q. Approximation Theory of Convolutional Architectures for Time Series Modelling. In Proceedings of the 38th International Conference on Machine Learning, pp. 4961–4970. PMLR, July 2021.

[17] Kajitsuka, T. and Sato, I. Are Transformers with One Layer Self-Attention Using Low-Rank Weight Matrices Universal Approximators?, July 2023.   
[18] Klusowski, J. M. and Barron, A. R. Approximation by Combinations of ReLU and Squared ReLU Ridge Functions with $\ell^1$ and $\ell^0$ Controls, May 2018.   
[19] Kratsios, A., Zamanlooy, B., Liu, T., and Dokmanić, I. Universal Approximation Under Constraints is Possible with Transformers, February 2022.   
[20] Levine, Y., Wies, N., Sharir, O., Bata, H., and Shashua, A. Limits to Depth Efficiencies of Self-Attention. In Advances in Neural Information Processing Systems, volume 33, pp. 22640–22651. Curran Associates, Inc., 2020.   
[21] Li, S., Chen, X., He, D., and Hsieh, C.-J. Can Vision Transformers Perform Convolution? October 2021.   
[22] Li, Z., Han, J., E, W., and Li, Q. On the Curse of Memory in Recurrent Neural Networks: Approximation and Optimization Analysis. In International Conference on Learning Representations, September 2020.   
[23] Li, Z., Jiang, H., and Li, Q. On the approximation properties of recurrent encoder-decoder architectures. In International Conference on Learning Representations, September 2021.   
[24] Li, Z., Han, J., E, W., and Li, Q. Approximation and Optimization Theory for Linear Continuous-Time Recurrent Neural Networks. Journal of Machine Learning Research, 23(42):1–85, 2022. ISSN 1533-7928.   
[25] Luo, S., Li, S., Zheng, S., Liu, T.-Y., Wang, L., and He, D. Your Transformer May Not be as Powerful as You Expect, October 2022.   
[26] Ostrand, P. A. Dimension of metric spaces and Hilbert's problem 13. Bulletin of the American Mathematical Society, 71(4):619–622, July 1965. ISSN 0002-9904, 1936-881X.   
[27] Sanford, C., Hsu, D., and Telgarsky, M. Representational Strengths and Limitations of Transformers, June 2023.   
[28] Shen, Z., Yang, H., and Zhang, S. Optimal Approximation Rate of ReLU Networks in terms of Width and Depth. Journal de Mathématiques Pures et Appliquées, 157:101–135, January 2022. ISSN 00217824. doi: 10.1016/j.matpur.2021.07.009.   
[29] Takakura, S. and Suzuki, T. Approximation and Estimation Ability of Transformers for Sequence-to-Sequence Functions with Infinite Dimensional Input, May 2023.   
[30] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is All you Need. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.   
[31] Wang, M. and E, W. Understanding the Expressive Power and Mechanisms of Transformer for Sequence Modeling, July 2024.   
[32] Wu, H., Xu, J., Wang, J., and Long, M. Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting, January 2022.   
[33] Yun, C., Bhojanapalli, S., Rawat, A. S., Reddi, S. J., and Kumar, S. Are Transformers universal approximators of sequence-to-sequence functions? In arXiv:1912.10077 [Cs, Stat], February 2020.   
[34] Yun, C., Chang, Y.-W., Bhojanapalli, S., Rawat, A. S., Reddi, S. J., and Kumar, S. \$O(n)\$ Connections are Expressive Enough: Universal Approximability of Sparse Transformers, December 2020.   
[35] Zeng, A., Chen, M., Zhang, L., and Xu, Q. Are Transformers Effective for Time Series Forecasting?, August 2022.   
[36] Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., and Zhang, W. Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting, March 2021.

# A Proof of Theorems

# A.1 Permutation Equivariance of the Hypothesis Space H

Let $p:[\tau]\to[\tau]$ be a permutation. Let's consider an $\hat{H}\in\mathcal{H}$ with permuted inputs:

$$
\hat {H} _ {t} (\boldsymbol {x} \circ p) = \hat {F} \left(\sum_ {s = 1} ^ {\tau} \sigma [ (W _ {Q} \hat {f} (x (p (t)))) ^ {\top} W _ {K} \hat {f} (\cdot) ] (p (s)) \cdot W _ {V} \hat {f} (x (p (s)))\right), \tag {15}
$$

Since the permutation of index $s$ does not affect the sum, we have

$$
= \hat {F} \left(\sum_ {s = 1} ^ {\tau} \sigma [ (W _ {Q} \hat {f} (x (p (t)))) ^ {\top} W _ {K} \hat {f} (\cdot) ] (s) \cdot W _ {V} \hat {f} (x (s))\right), \tag {16}
$$

$$
= \hat {H} _ {p (t)} (\boldsymbol {x}). \tag {17}
$$

This shows that the Transformer hypothesis $\mathcal{H}$ is permutation equivariant.

# A.2 Density of the Target Space $\mathcal{C}$

In this section, we show that the target space $\mathcal{C}$ defined in Equation (8) is dense in the general continuous target space $C(\mathcal{X}^{(E)},\mathcal{Y})$ . This ensures that the space defined is general enough to approximate arbitrary continuous targets.

To begin with, we first introduce the following representation theorem for multivariate functions.

Theorem A.1 (Kolmogorov Representation Theorem). [26] Let $I_{1}, \ldots, I_{\tau}$ be compact d dimensional metric spaces. Then there are continuous functions $\psi_{qs}: I_{s} \to [0, 1]$ and continuous function $g_{q}: [0, \tau] \to R$ , such that any continuous function $f: \prod I_{i} \to R$ can be represented as

$$
f (x _ {1}, \dots , x _ {\tau}) = \sum_ {q = 0} ^ {2 \tau d} g _ {q} \left(\sum_ {s = 1} ^ {\tau} \psi_ {q s} (x _ {s})\right). \tag {18}
$$

This theorem states that any $\tau$ variable functions can be decomposed into superpositions of one variable function.

Now, we discuss how the pointwise functions on a sequence can apply different mappings on different time indices.

Proposition A.2. Suppose we have $\tau$ variables in disjoint domains, where $x_{1} \in I_{1}, \cdot, x_{\tau} \in I_{\tau}$ and $I_{1}, \ldots, I_{\tau}$ are all disjoint. We consider the following vector-valued function

$$
f (x _ {1}, \dots , x _ {\tau}) = (f _ {1} (x _ {1}), \dots , f _ {\tau} (x _ {\tau})), \tag {19}
$$

where $f_{s}:\mathcal{I}_{s}\to \mathbb{R}$ for $s\in [\tau]$ . We can indeed define a pointwise function $g:\bigcup \mathcal{I}_s\to \mathbb{R}$ to represent $f$ . Since all $\mathcal{I}_s$ are disjoint, we can define a piecewise function $g$ , where $g(x_s) = f_s(x_s)$ holds for all $s\in [\tau]$ .

This proposition demonstrates that for a sequence input x with appropriate positional encodings, where $x(i)$ and $x(j)$ belong to disjoint sets, a pointwise function can represent distinct mappings for these elements. We are now ready to present the following theorem.

Theorem A.3. Consider $d$ -dimensional, length $\tau$ input space $\mathcal{X}^{(E)}$ with position encoding added. Then, for any $\boldsymbol{H} \in C(\mathcal{X}^{(E)}, \mathcal{Y})$ , there exists continuous functions $F \in C([0,1]^n, \mathbb{R})$ , $f \in C(\mathcal{I}, [0,1]^n)$ and $\rho \in C(\mathcal{I} \times \mathcal{I}, \mathbb{R})$ such that for all $t \in [\tau]$ we have

$$
H _ {t} (\boldsymbol {x}) = F \left(\sum_ {s = 1} ^ {\tau} \sigma [ \rho (x (t), x (\cdot)) ] (s) f (x (s))\right), \tag {20}
$$

where $n = 2\tau d + 1$ and $\sigma$ is the softmax function.

Proof. Based on the representation for continuous function Theorem A.1, we can decompose $\pmb{H}$ into

$$
H _ {t} (\boldsymbol {x}) = \sum_ {q = 0} ^ {2 \tau d} g _ {q} ^ {(t)} \left(\sum_ {s = 1} ^ {\tau} \psi_ {q, s} (x _ {s})\right), \tag {21}
$$

where $\psi_{q,s}^{(\tau)}:\mathcal{I}_s\to [0,1]$ and $g_q^{(t)}:[0,\tau ]\to \mathbb{R}$ are continuous functions. We next construct $F,f$ and $\rho$ to make $\pmb{H}$ satisfy this form. Firstly, since $\mathcal{I}_s$ are disjoint we can define proper piecewise function $\rho$ such that $\sigma [\rho (x(t),\cdot)](x(s)) = \left\{ \begin{array}{ll}\frac{2}{\tau + 1} & t = s\\ \frac{1}{\tau + 1} & t\neq s \end{array} \right.$ , which simplifies Equation (8) to

$$
H _ {t} (\boldsymbol {x}) = F \left(\frac {1}{\tau + 1} \Big (f (x (t)) + \sum_ {s = 1} ^ {\tau} f (x (s)) \Big)\right). \tag {22}
$$

Next, based on Proposition A.2, we let the pointwise function $f: \mathcal{I} \to [0,1]^n$ to apply different mappings for each $x(s)$ , such that

$$
f: x (s) \mapsto \left(\hat {\psi} _ {0, s} (x _ {s}), \dots , \hat {\psi} _ {2 \tau d, s} (x _ {s}), b _ {s}\right), \tag {23}
$$

where $b_{s} \in [0,1]$ are different constants. We then have that

$$
f (x (t)) + \sum_ {s = 1} ^ {\tau} f (x (s)) = \left(\hat {\psi} _ {0, t} \left(x _ {t}\right) + \sum_ {s = 1} ^ {\tau} \hat {\psi} _ {0, s} \left(x _ {s}\right), \dots , \hat {\psi} _ {2 \tau d, t} \left(x _ {t}\right) + \sum_ {s = 1} ^ {\tau} \hat {\psi} _ {2 \tau d, s} \left(x _ {s}\right), b _ {t} + \sum_ {s = 1} ^ {\tau} b _ {s}\right). \tag {24}
$$

Here, $b_{t}$ performs as a shifting to make the range of Equation (24) disjoint for different t. Again, based on Proposition A.2, we can define F to have individual mappings for different t. We first define $F_{1} : [0, \tau + 1]^{n} \to [0, \tau]^{n}$ to be

$$
F _ {1}: u (t) \mapsto \left(u _ {1} (t) - \hat {\psi} _ {0, t} (x _ {t}), \dots , u _ {2 \tau d + 1} (t) - \hat {\psi} _ {2 \tau d, t} (x _ {t}), u _ {n} (t) - \sum_ {s = 1} ^ {\tau} b _ {s}\right), \tag {25}
$$

where $u_{i}$ denotes the $i$ -th dimension of $u$ . Next, define $F_{2}:[0,\tau]^{n}\to\mathbb{R}$ to be

$$
F _ {2}: u (t) \mapsto \sum_ {q = 0} ^ {2 \tau d} g _ {q} ^ {(t)} \left(u _ {q} (t)\right). \tag {26}
$$

Finally, let $F(u) = F_{2} \circ F_{1}((\tau + 1)u)$ . Substitute into Equation (22) we get that

$$
H _ {t} (\boldsymbol {x}) = \sum_ {q = 0} ^ {2 \tau d} g _ {q} ^ {(t)} \left(\sum_ {s = 1} ^ {\tau} \psi_ {q, s} (x _ {s})\right). \tag {27}
$$

![](images/b4b4f40d617aff2c79774e8102a34265d4b3388f8bde8090d5b76f998d6bd9a7.jpg)

This theorem shows that the target space C is, in fact, a representation of the general continuous target space $C(\mathcal{X}^{(E)},\mathcal{Y})$ . In particular, since I is a compact metric space, the complexity measures of each component $g_{q}^{(s)}$ and $\psi_{q,s}$ used in the construction of H have finite complexity measures $C_{0}, C_{1}$ and $C_{2}$ . This implies that the approximation space $\mathcal{C}^{(\alpha,\beta)}$ is also dense.

# A.3 Proof of Jackson-type approximation rate Theorem 4.2

Now, we present the proof of the Jackson-type approximation rates.

Lemma A.4. The following inequality will be used to prove the theorem.

$$
\left| a b - \hat {a} \hat {b} \right| \leq | a | | b - \hat {b} | + | \hat {b} | | a - \hat {a} | \tag {28}
$$

$$
\leq | a | | b - \hat {b} | + | b | | a - \hat {a} | + | a - \hat {a} | | b - \hat {b} | \tag {29}
$$

Proof. Proof of Theorem 4.2 Since there are various components in the model, we will consider each of them separately. Firstly let's consider the approximation of $F$ .

Let

$$
h _ {t} (\boldsymbol {x}) = \sum_ {s} ^ {\tau} \sigma (\rho (x (t), x (s))) f (x (s)) \tag {30}
$$

and

$$
\hat {h} _ {t} (\boldsymbol {x}) = \sum_ {s} ^ {\tau} \sigma (\hat {\rho} (x (t), x (s))) \hat {f} (x (s)) \tag {31}
$$

we have

$$
\left| F (h) - \hat {F} (\hat {h}) \right| = \left| F (h) - F (\hat {h}) + F (\hat {h}) - \hat {F} (\hat {h}) \right| \tag {32}
$$

$$
\leq K _ {F} \left| h - \hat {h} \right| + \left| F (\hat {h}) - \hat {F} (\hat {h}) \right|. \tag {33}
$$

Let's consider $\left|F(\hat{h}) - \hat{F}(\hat{h})\right|$ first. Since $\mathcal{I}$ is a compact domain and $\hat{h}$ is continuous, its range $\hat{h}(\mathcal{I})$ is a compact set. For all $x \in \hat{h}(\mathcal{I})$ we have $\left|F(\hat{h}(\boldsymbol{x})) - \hat{F}(\hat{h}(\boldsymbol{x}))\right| \leq \| F - \hat{F}\|_{L^{\infty}(\hat{h}(\mathcal{I}))}$ . This implies that

$$
\sum_ {t} ^ {\tau} \int_ {\mathcal {I}} \left| F (\hat {h} _ {t} (\boldsymbol {x})) - \hat {F} (\hat {h} _ {t} (\boldsymbol {x})) \right| d \boldsymbol {x} \leq \sum_ {t} ^ {\tau} \int_ {\mathcal {I}} d \boldsymbol {x} \cdot \left\| F - \hat {F} \right\| _ {L ^ {\infty} (\hat {h} (\mathcal {I}))} \tag {34}
$$

$$
= \tau \| \mathcal {I} | \left\| F - \hat {F} \right\| _ {L ^ {\infty} (\hat {h} (\mathcal {I}))} \tag {35}
$$

where $|\mathcal{I}|$ denotes the volume of $\mathcal{I}$ . Recall that $\hat{h}_t(\pmb{x}) = \sum_s^\tau \sigma (\hat{\rho} (x(t), x(s)))\hat{\rho} (x(s))$ , we may assume the parameters of $\hat{f}$ is bounded such that $\hat{f}(x(s)) \subset [0,1]$ . By the property of Softmax we have $\hat{h} (\mathcal{I}) \subset [0,1]$ , thus,

$$
\leq \tau | \mathcal {I} | \left\| F - \hat {F} \right\| _ {L ^ {\infty} ([ 0, 1 ])} \leq \tau \left\| F - \hat {F} \right\|. \tag {36}
$$

Next, let's consider the first term of (32),

$$
\left| h - \hat {h} \right| \leq \sum_ {s} ^ {\tau} \left| \sigma (\rho (x _ {t}, x _ {s})) f (x _ {s}) - \sigma (\hat {\rho} (x _ {t}, x _ {s})) \hat {f} (x _ {s}) \right| \tag {37}
$$

$$
\leq \sum_ {s} ^ {\tau} | f | \left| \sigma (\rho (x _ {t}, x _ {s})) - \sigma (\hat {\rho} (x _ {t}, x _ {s})) \right| + \left| \sigma (\hat {\rho} (x _ {t}, x _ {s})) \right| \left| f - \hat {f} \right| \tag {38}
$$

Since Softmax is Lipschitz continuous and bounded by 1, we have that

$$
\leq \sum_ {s} ^ {\tau} | f | | \rho - \hat {\rho} | + \sum_ {s} ^ {\tau} \left| f - \hat {f} \right|. \tag {39}
$$

The second term is an approximation with neural networks

$$
\int \sum_ {t, s} ^ {\tau} \left| f (x _ {s}) - \hat {f} (x _ {s}) \right| d \boldsymbol {x} \leq \tau^ {2} \left\| f - \hat {f} \right\|. \tag {40}
$$

Now we remain to derive a bound for $|\rho - \hat{\rho}|$ . Let $\tilde{\rho}$ be a $m_h$ term truncation of POD expansion of $\rho$ , then

$$
\left| \rho - \hat {\rho} \right| = \left| \rho - \tilde {\rho} + \tilde {\rho} - \hat {\rho} \right| \tag {41}
$$

$$
\leq | \rho - \tilde {\rho} | + | \tilde {\rho} - \hat {\rho} |. \tag {42}
$$

The first term is the error estimates using POD, which says that

$$
\int \sum_ {t, s} ^ {\tau} | \rho (x _ {t}, x _ {s}) - \tilde {\rho} (x _ {t}, x _ {s}) | ^ {2} d \boldsymbol {x} = \tau^ {2} \| \rho - \hat {\rho} \| _ {2} ^ {2} \tag {43}
$$

$$
\leq \tau^ {2} \sum_ {i = m _ {h} + 1} ^ {\infty} \sigma_ {i} ^ {2}. \tag {44}
$$

The second term is again approximation with neural works, which we have

$$
\sum_ {i = 1} ^ {m _ {h}} \left| \phi_ {i} \psi_ {i} - \hat {\phi} _ {i} \hat {\psi} _ {i} \right| \leq \sum_ {i = 1} ^ {m _ {h}} | \phi_ {i} | | \psi_ {i} - \hat {\psi} _ {i} | + | \psi_ {i} | \left| \phi_ {i} - \hat {\phi} _ {i} \right| + \left| \phi_ {i} - \hat {\phi} _ {i} \right| | \psi_ {i} - \hat {\psi} _ {i} | \tag {45}
$$

Thus, combining all the inequalities above, we have that

$$
\int \sum_ {t} ^ {\tau} \left| H _ {t} (\boldsymbol {x}) - \hat {H} _ {t} (\boldsymbol {x}) \right| d \boldsymbol {x} \leq \tag {46}
$$

$$
K _ {F} \sup | f | \tau^ {2} \left(\sum_ {i = m _ {h} + 1} ^ {\infty} \sigma_ {i} ^ {2} + \sum_ {i = 1} ^ {m _ {h}} \sup | \phi_ {i} | \| \psi_ {i} - \hat {\psi} _ {i} \| + \sup | \psi_ {i} | \| \phi_ {i} - \hat {\phi} _ {i} \| + \| \phi_ {i} - \hat {\phi} _ {i} \| \| \psi_ {i} - \hat {\psi} _ {i} \|\right) \tag {47}
$$

$$
+ \tau^ {2} \| F - \hat {F} \| + \tau^ {2} \| f - \hat {f} \| \tag {48}
$$

By substituting the complexity measure of H to R.H.S., we have that

$$
R. H. S. \leq \tau^ {2} \mathcal {C} _ {0} (\boldsymbol {H}) \left(\frac {C _ {1} ^ {(\alpha)} (\boldsymbol {H})}{m _ {h} ^ {2 \alpha - 1}} + \frac {C _ {2} ^ {(\beta)} (\boldsymbol {H})}{m _ {\mathrm{FF}} ^ {\beta}} \cdot (m _ {h}) ^ {\beta + 1}\right). \tag {49}
$$

![](images/7cd4651e3b311feaec63a86a71c9d9f580ec1785a293dc4a6a01f7d71aac3af5.jpg)

This completes the proof.

# B Extra discussions

# B.1 Numerical Examples of the Target Form Equation (8)

In this section, we provide numerical examples that follow the target form Equation (8):

$$
H _ {t} (\boldsymbol {x}) = F \left(\sum_ {s = 1} ^ {\tau} \sigma [ \rho (x (t), x (\cdot)) ] (s) f (x (s))\right).
$$

This form, in fact, is not a unique representation; in other words, for $\pmb{H}_1 = \pmb{H}_2$ we may not have $F_1 = F_2$ , $f_1 = f_2$ and $\rho_1 = \rho_2$ . However, we have the following propositions that characterize the invariant properties of the target form.

Proposition B.1. Suppose $H_{1}$ and $H_{2}$ are in the form of Equation (8), then the following properties hold.

1. If $\boldsymbol{H}_{1}(\boldsymbol{x}) = \boldsymbol{H}_{2}(\boldsymbol{x})$ holds for all x, then $F_{1} \circ f_{1} = F_{2} \circ f_{2}$ .   
2. Suppose $\sigma$ is hard-max function. For non constant $\mathbf{H}_1, \mathbf{H}_2 \in \tilde{\mathcal{C}}$ , if $\mathbf{H}_1(\mathbf{x}) = \mathbf{H}_2(\mathbf{x})$ holds for all $\mathbf{x}$ , then $\arg \max_{x(s)}[\rho_1(x(t), x(s))] = \arg \max_{x(s)}[\rho_2(x(t), x(s))]$ if the argmax is unique and $F_1 \circ f_1, F_2 \circ f_2$ are injections.   
3. Suppose $\sigma$ is softmax function. For non constant $\mathbf{H}_1, \mathbf{H}_2$ , if $\mathbf{H}_1(\mathbf{x}) = \mathbf{H}_2(\mathbf{x})$ holds for all $\mathbf{x}$ and $\rho_1, \rho_2$ are unbounded, then $\arg \max_{x(s)}[\rho_1(x(t), x(s))] = \arg \max_{x(s)}[\rho_2(x(t), x(s))]$ if the argmax is unique and $F_1 \circ f_1$ , $F_2 \circ f_2$ are injections.

Proof. By considering a constant input sequence $\pmb{x} = (x, \dots, x)$ , we get $F_1 \circ f_1(x) = F_2 \circ f_2(x)$ for all $x$ .

Suppose $\arg \max[\rho_1(x(t), x(s))] = c_1$ and $\arg \max[\rho_2(x(t), x(s))] = c_2$ , then $H_1(x) = H_2(x)$ implies $F_1 \circ \rho_1(c_1) = F_2 \circ \rho_2(c_2)$ . By the injection assumption, we have $c_1 = c_2$ , which implies the argmax are equal.

When the normalization is softmax, since $\rho$ is unbounded, we consider a sequence of inputs $\{\pmb{x}^{(i)}\}$ such that the $\rho(x^{(i)}(t), x^{(i)}(s))$ goes to infinity, which became the same as the previous hard-max case.

Although these properties need to satisfy certain conditions to be theoretically correct, we empirically found they generally hold true in real applications.

Next, we demonstrate these concepts through numerical examples by utilizing the Transformer model to learn different targets. We consider the following simplified version of a single layer Transformer

$$
\hat {H} _ {t} (\boldsymbol {x}) = \hat {F} \left(W _ {V} \hat {f} (\boldsymbol {x}) \cdot \sigma [ (W _ {Q} \hat {f} (\boldsymbol {x})) ^ {\top} W _ {K} \hat {f} (\boldsymbol {x}) ]\right). \tag {50}
$$

Here, $x \in R^{d \times \tau}$ is the input, $\hat{F}, \hat{f}$ are two nonlinear mappings applied column-wise, and $\sigma$ is the softmax function applied column-wise to the matrix.

Nearest point to a set In this example, we consider two sets of points $U \subset R^d$ and $V \subset R^d$ as inputs. For each point $u_t \in U$ , we aim to determine the nearest point from set $V$ . More specifically, we define an input sequence $x \in \mathcal{X} \subset \mathbb{R}^{2d \times \tau}$ in the form $x(t) = (u_t, v_t) \in R^{2d}$ , where $u_t$ and $v_t$ are two $d$ -dimensional points belonging to point set $U$ and $V$ , respectively. The output is defined as

$$
y (t) = \underset {v \in V} {\arg \min} | u (t) - v |. \tag {51}
$$

This can be rewritten as

$$
y (t) = \sum_ {s = 1} ^ {\tau} \sigma_ {H} [ - | u (t) - v (\cdot) | ] (s) v (s), \tag {52}
$$

where $\sigma_{H}: R^{\tau} \to R^{\tau}$ is the hard-max function. Note this equation conforms with the form in Equation (8) with $F \circ f$ begin identity and $\rho(u, v) = |u - v|$ . For the numerical example, we consider $U, V \subset R^{2}$ and each have 6 points, and train a Transformer model to learn this target and examine the learned model.

![](images/2afd63beb8e1e17b2f9f9e6c50bb15f75067dcbeb010ad8a8f72700128307a4a.jpg)

<details>
<summary>heatmap</summary>

| | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 0 | Dark Blue | Dark Blue | Light Green | Light Green | Dark Blue | Dark Blue |
| 1 | Dark Blue | Dark Blue | Light Green | Light Green | Medium Green | Medium Green |
| 2 | Dark Blue | Dark Blue | Light Green | Medium Green | Medium Green | Medium Green |
| 3 | Dark Blue | Dark Blue | Light Green | Medium Green | Medium Green | Medium Green |
| 4 | Dark Blue | Dark Blue | Light Green | Medium Green | Medium Green | Medium Green |
| 5 | Dark Blue | Dark Blue | Light Green | Medium Green | Medium Green | Medium Green |
</details>

(a) Attention matrix

![](images/d57832403782af3efc9f3e14c80def0355ab6657283c382e8d1c26613daa39fc.jpg)

(b) Graph recovered by the model   
![](images/a054938189b4afa703b4f341ad1c13c1d8f6f22a2675d6c2988f958d382fe412.jpg)

<details>
<summary>scatter</summary>

| x    | y    |
| ---- | ---- |
| 0.00 | 0.00 |
| 0.25 | 0.25 |
| 0.50 | 0.50 |
| 0.75 | 0.75 |
| 1.00 | 1.00 |
</details>

(c) Scatter plot of $\hat{F} \circ \hat{f}$   
Figure 2: For Figures (a) and (b) we examine a particular instance of the input x. Figure (a) plots the attention matrix A, while Figure (b) illustrates the learned relationships, with green points and red points representing points from set U and V, respectively. Figure (c) is the scatter plot of $F \circ f(\boldsymbol{x})$ for randomly generated inputs x.

In Figure 2(a), we plot the attention matrix A of a particular instance of the input x. For Figure 2(b), we visualize the point set U, V based on their coordinates and establish connections according

to the values in $A$ . Specifically, two points $u_{i}$ and $v_{j}$ are connected based on the value of $A_{ij}$ . This visualization reveals exactly that each green point in set $U$ is linked to the nearest point in $V$ , demonstrating the model's ability to recover the target relationship accurately. Furthermore, for the target form Equation (52), $F \circ f$ in this case is an identity function. As shown in Figure 2(c), $\hat{F} \circ \hat{f}$ is also found to learn an identity function. This observation is in accordance with Proposition B.1, despite the theoretical assumptions that may not fully apply.

Weighted average by weight In this scenario, we are given sequences of point masses as inputs. Our objective is to compute the center of gravity for each sequence. These point masses are in $\mathbb{R}^d$ space, each with a mass $m\in \mathbb{R}$ .

Consider a sequence containing $\tau$ points. We extend this to a sequence of $\tau + 1$ points by adding an extra point $x_{\mathrm{pred}} \notin \mathcal{X}$ at the beginning of the sequence, serving as a prediction token. Consequently, the input sequence is represented as:

$$
\boldsymbol {x} = (x _ {\text { pred }}, \binom {x _ {1}} {m _ {1}}, \dots , \binom {x _ {\tau}} {m _ {\tau}}). \tag {54}
$$

The corresponding output sequence is formulated as:

$$
\boldsymbol {y} = (| \bar {x} |, | x _ {1} |, \dots , | x _ {\tau} |), \tag {55}
$$

where $x_{\mathrm{pred}}$ is mapped to $\bar{x}$ , and the other points remain unchanged. $|\cdot|$ denotes the Euclidean norm. The center of gravity $\bar{x}$ is defined by the equation:

$$
\bar {x} := \sum_ {s = 1} ^ {\tau} \frac {m _ {s}}{\sum_ {s ^ {\prime}} m _ {s ^ {\prime}}} x _ {s}. \tag {56}
$$

We may consider several sequence segments of point masses, denoted as $x_{1}$ to $x_{n}$ , which may vary in length. These sequences are concatenated to form an input sequence x. Correspondingly, the output y concatenates $y_{1}$ to $y_{n}$ .

As an example, consider the following input-output pair:

$$
\boldsymbol {x} = (x _ {\text { pred }}, \binom {x _ {1}} {m _ {1}}, \binom {x _ {2}} {m _ {2}}, \binom {x _ {3}} {m _ {3}}, x _ {\text { pred }}, \binom {x _ {4}} {m _ {4}}, \binom {x _ {5}} {m _ {5}}). \tag {57}
$$

$$
\boldsymbol {y} = \left(\left| \frac {\sum_ {s = 1} ^ {3} m _ {s} x _ {s}}{\sum_ {s ^ {\prime} = 1} ^ {3} m _ {s ^ {\prime}}} \right|, | x _ {1} |, | x _ {2} |, | x _ {3} |, \left| \frac {\sum_ {s = 4} ^ {5} m _ {s} x _ {s}}{\sum_ {s ^ {\prime} = 4} ^ {5} m _ {s ^ {\prime}}} \right|, | x _ {4} |, | x _ {5} |\right). \tag {58}
$$

In this example, the output sequence y includes the norm of the computed centers of gravity for each concatenated segment of the input sequence x, where each segment is separated by the prediction token $x_{pred}$ . This can be formulated as target form Equation (8) such that

$$
y (t) = \left\{ \begin{array}{l l} \left\| \sum_ {s \in \mathcal {I} _ {x}} \frac {m _ {s}}{\sum_ {s ^ {\prime} \in \mathcal {I} _ {x}} m _ {s ^ {\prime}}} x (s) \right\| & x (t) = x _ {\text { pred }} \\ \left\| \sum_ {s = 1} ^ {\tau} \mathbf {1} _ {s = t} x (s) \right\| & \text { otherwise } \end{array} , \right. \tag {59}
$$

where $I_{x}$ denotes the index of the sequence segment corresponding to $x_{pred}$ . The weighted output here depends on the prediction token, such that it only takes the weighted average within the corresponding sequence. For numerical illustration, we consider input x of length 10, containing sequence segments of length at least 2. We apply the single-layer Transformer model to learn this target.

In Figure 3, we analyze an input x comprising two sequence segments, $x_{1}$ and $x_{2}$ , with lengths of 3 and 5, respectively. In this scenario, $x(0)$ and $x(5)$ are prediction tokens. The attention matrix behaves as expected: the outputs $y(0)$ and $y(5)$ are influenced predominantly by the points within their respective sequences, and the values are approximately the normalized weight. Furthermore, since the remaining outputs are the same as the inputs, the attention matrix values for these elements are confined to the diagonal, which means that the output at that point is only determined by itself.

Furthermore, in Figure 3(c) we observe that $\hat{F} \circ \hat{f}$ can successfully recover $F \circ f$ in the target which is a norm of the $\mathbb{R}^2$ vector.

![](images/b03081e2c7350c0c57c94049ac88dd3bc72810b20c5ed60db141eb806c86a679.jpg)

<details>
<summary>heatmap</summary>

| | 0 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| 0 | 0.0 | 0.2 | 0.3 | 0.4 | 0.5 |
| 2 | 0.1 | 0.3 | 0.4 | 0.5 | 0.6 |
| 4 | 0.2 | 0.4 | 0.5 | 0.6 | 0.7 |
| 6 | 0.3 | 0.5 | 0.6 | 0.7 | 0.8 |
| 8 | 0.4 | 0.6 | 0.7 | 0.8 | 0.9 |
</details>

(a) Attention matrix

![](images/2c95f931bfc099372d34c6f5b3eceb83b2b0b7815210b9509debe5b5079606e8.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    x1[" x₁ "] --> y1[" y₁ "]
    x2[" x₂ "] --> y2[" y₂ "]
    y1 --> y2
    y2 --> y1
    style x1 fill:#f9f,stroke:#333
    style x2 fill:#bbf,stroke:#333
```
</details>

(b) Graph recovered by the model   
![](images/51e55b465520530be64f30563107592ad7c203330e64e3936cb17d8b09a693be.jpg)

<details>
<summary>heatmap</summary>

|        | 0.0  | 0.2  | 0.4  |
| ------ | ---- | ---- | ---- |
| 0.0    | -0.4 | -0.2 | 0.0  |
| 0.2    | -0.2 | 0.0  | 0.2  |
| 0.4    | 0.0  | 0.2  | 0.4  |
</details>

(c) Contour plot of $\hat{F} \circ \hat{f}$   
Figure 3: Figure (a) plots the attention matrix A, while Figure (b) is the illustration of the learned relationship. In this instance, there are two sequences, $x_{1}$ and $x_{2}$ , each connected to their respective predictions. The color of the connecting lines represents the corresponding values in A. Figure (c) presents the contour plot of $F \circ f(\boldsymbol{x})$ , generated for a set of random inputs x.

Weighted average by selection The previous calculated the weighted average by specifying the weight of each point. Now, we consider another way to calculate a weighted average where we specify a subset of points we are going to take the average. Suppose we have a length $\tau$ sequence of $\mathbb{R}^d$ vectors $\pmb{v} \in \mathbb{R}^{d \times \tau}$ . We also have a sequence of vectors $s \in [\tau]^{k \times \tau}$ consisting of sequences of indices. We stack $\pmb{v}$ and $s$ to form the input sequence $x \in \mathbb{R}^{(d + k) \times \tau}$ . This extends the qSA task proposed in Sanford et al. [27].

The output $y(t)$ is defined as

$$
y (t) = \frac {1}{k} \sum_ {s \in \mathbf {s} (t)} v _ {s}. \tag {60}
$$

This can be written in the target form Equation (8) such that $F \circ f$ is identity. Depending on the frequency of $s$ appears in the index vector $s(t)$ , we have that

$$
\rho (x (t), x (s)) \in \left\{0, \frac {1}{k}, \frac {2}{k}, \dots , 1 \right\} \tag {61}
$$

As an example, we have the following input-output pairs.

$$
\boldsymbol {v} = (v _ {1}, v _ {2}, v _ {3}),
$$

$$
\boldsymbol {s} = \left(\left( \begin{array}{l} 1 \\ 1 \\ 2 \end{array} \right), \left( \begin{array}{l} 1 \\ 2 \\ 3 \end{array} \right), \left( \begin{array}{l} 3 \\ 3 \\ 3 \end{array} \right)\right). \tag {62}
$$

$$
\boldsymbol {y} = \frac {1}{3} \left(2 v _ {1} + v _ {2}, v _ {1} + v _ {2} + v _ {3}, 3 v _ {3}\right).
$$

In our numerical experiments, we set $\tau = 8$ , d = 2, and k = 3. This configuration implies that both the input and output sequences have a length of 8, and the output $y(t)$ is computed as the average of three inputs, as indices designated by $s(t)$ . For this particular setting we have $\rho \in \{0, \frac{1}{2}, \frac{2}{3}, 1\}$ . Again, we use a single-layer transformer model to learn this target.

In Figure 4 we examine a particular input, where

$$
\boldsymbol {s} = \left(\left( \begin{array}{l} 4 \\ 5 \\ 5 \end{array} \right), \left( \begin{array}{l} 4 \\ 4 \\ 4 \end{array} \right), \left( \begin{array}{l} 1 \\ 1 \\ 7 \end{array} \right), \left( \begin{array}{l} 5 \\ 5 \\ 5 \end{array} \right), \left( \begin{array}{l} 0 \\ 3 \\ 7 \end{array} \right), \left( \begin{array}{l} 0 \\ 6 \\ 7 \end{array} \right), \left( \begin{array}{l} 2 \\ 2 \\ 2 \end{array} \right), \left( \begin{array}{l} 2 \\ 3 \\ 4 \end{array} \right)\right). \tag {63}
$$

In Figure 4(a), we observe that the attention matrix accurately reconstructs $\rho$ , as demonstrated in Equation (61). Figure 4(b) illustrates the graph representation of this input-output correlation based on the matrix $A$ . In this graph, an output $i$ and an input $j$ are connected by the value of $A_{ij}$ . The figure highlights the model's capability to recover the relationship: Output $y(1), y(3), y(6)$ each rely on a singular input value, output $y(0), y(2)$ are calculated as the average of two values, and output $y(4), y(5), y(7)$ each represent an average over three values.

![](images/3697b3d5352e09e3d4fcba9349743b28a625dda38b68d4d42c39d83446688136.jpg)

<details>
<summary>heatmap</summary>

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
The values in the table represent the magnitudes of the color-coded cells in the grid. There is no explicit numerical data provided in the image.
</details>

(a) Attention matrix

![](images/10c63169d72798ef18add2bece9719514175dbbe3d688f647b6977e1055b2ec4.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input 6"] --> B["Output 5"]
    B --> C["Node 3"]
    C --> D["Output 4"]
    D --> E["Node 4"]
    E --> F["Output 2"]
    F --> G["Node 7"]
    G --> H["Output 1"]
    H --> I["Node 1"]
    I --> J["Output 2"]
    J --> K["Node 6"]
    K --> L["Output 4"]
    L --> M["Node 7"]
    M --> N["Output 4"]
    N --> O["Node 4"]
    O --> P["Output 2"]
    P --> Q["Node 6"]
    Q --> R["Output 4"]
    R --> S["Node 4"]
    S --> T["Output 2"]
    T --> U["Node 6"]
    U --> V["Output 4"]
    V --> W["Node 4"]
    W --> X["Output 2"]
    X --> Y["Node 6"]
    Y --> Z["Output 4"]
    Z --> AA["Node 4"]
    AA --> AB["Output 2"]
```
</details>

(b) Graph recovered by the model   
![](images/57ff64f7de3e637613b0660c756350c8f88ce1ab9372318d77dd1382c1a21d60.jpg)

<details>
<summary>line</summary>

| x    | y    |
| ---- | ---- |
| 0.00 | 0.00 |
| 0.25 | 0.25 |
| 0.50 | 0.50 |
| 0.75 | 0.75 |
| 1.00 | 1.00 |
</details>

(c) Contour plot of $\hat{F} \circ \hat{f}$   
Figure 4: Figure (a) plots the attention matrix $A$ , while Figure (b) illustrates the learned relationship. The points in different colors refer to the input and output, respectively. They are connected based on the value of $A$ . Figure (c) presents the scatter plot of $F \circ f(\pmb{x})$ , generated for a set of random inputs $\pmb{x}$ .

In summary, these examples demonstrate tasks that follow the form presented in Equation (8). Moreover, the numerical experiments validates Proposition B.1, where $F \circ f$ is always the same, and the attention matrix recovers the desired weights.

# B.2 Synthetic Examples

![](images/a5a102a73254cded080d3717e34d00a49acaef07f4a8cf02e78f3e26177941ba.jpg)

<details>
<summary>line</summary>

| mh | r = 2 | r = 6 | r = ∞ |
|----|-------|-------|-------|
| 2  | 1.0   | 1.5   | 1.8   |
| 4  | 0.0   | 0.8   | 1.4   |
| 6  | 0.0   | 0.0   | 1.0   |
| 8  | 0.0   | 0.0   | 0.8   |
| 10 | 0.0   | 0.0   | 0.7   |
</details>

(a) $\sigma_{k} = \left\{ \begin{array}{ll}k^{-0.55} & k\leq r\\ 0 & k > r \end{array} \right.$

![](images/3200594e8251cc0191b705900793eff9659dc3d8243cae704c0be35faacb6775.jpg)

<details>
<summary>line</summary>

| mh | r = 2 | r = 6 | r = ∞ |
| --- | --- | --- | --- |
| 2 | 1.0 | 1.3 | 1.5 |
| 4 | 0.0 | 0.5 | 0.7 |
| 6 | 0.0 | 0.0 | 0.4 |
| 8 | 0.0 | 0.0 | 0.3 |
| 10 | 0.0 | 0.0 | 0.2 |
</details>

(b) $\sigma_{k} = \left\{ \begin{array}{ll}k^{-1} & k\leq r\\ 0 & k > r \end{array} \right.$   
Figure 5: We consider two class of $\rho$ with different singular value decay rate $\alpha$ as indicated above. Here, $r$ denotes the rank of the target. For each (a) and (b), we consider three targets with $\rho$ having different ranks, where $r = 2,6,\infty$ . The figure plots the training error against $m_h$ . Each colored line corresponds to a target with rank $r$ as indicated in the legend. The grey dotted line plots $m_h^{-(2\alpha -1)}$ .

The Jackson-type approximation rate in Theorem 4.2 has the following prediction. For sequence relationships admitting a representation Equation (8), if its rank $r$ is finite, then perfect approximation is possible as long as $m_h \geq r$ with $m_{\mathrm{FF}}$ large enough. If $r = \infty$ , then the approximation error is determined by decay of $\sigma_k$ . Concretely, if $\sigma_k \sim k^{-\alpha}$ , then the error decays like $m_h^{-(2\alpha - 1)} + \text{constant}$ , provided $m_{\mathrm{FF}}$ is sufficiently large. We numerically verify the prediction by constructing a set of targets using Equation (8). In this specific example, we set both $F$ and $f$ as the identity function. The temporal coupling term $\rho(u, v)$ is formulated using an orthonormal bases where $\phi_i, \psi_i \in \{\sqrt{2} \sin 2i\pi x : i \geq 1\}$ , accompanied by singular values with various decaying patterns. Subsequently, we trained the Transformer as defined in Equation (5) to learn these targets. Figure 5 plots the training error against $m_h$ , where (a) and (b) correspond to targets with different singular value decay patterns. We also consider different rank $r$ of the targets as plotted using lines with different colors. The error decay rates of (a) and (b) are different, which follows the decay rate determined by that of the singular values. Furthermore, for a fixed $m_h$ , the error increases with the target's rank. The experiment results are consistent with our estimates in Theorem 4.2.

# C Experiment settings

In this section, we summarize the settings for the numerical experiments.

Experiments in Appendix B.2 These experiments consider the synthetic targets with simplified Transformer models as defined in Equation (5). The target relationship is constructed using Equation (8):

$$
H _ {t} (\boldsymbol {x}) = F \left(\sum_ {s = 1} ^ {\tau} \sigma [ \rho (x (t), x (\cdot)) ] (s) f (x (s))\right). \tag {64}
$$

We consider scalar inputs where $d = 1$ and $F(x) = f(x) = x$ are identity functions. $\rho(u, v)$ is constructed using the POD decomposition:

$$
\rho (u, v) = \sum_ {i = 1} ^ {r} \sigma_ {i} \phi_ {i} (u) \psi_ {i} (v). \tag {65}
$$

We consider orthonormal bases $\phi_{i},\psi_{i}\in\{\sqrt{2}\sin2i\pi x:i\geq1\}$ . By doing this, we can specify $\sigma_{i}$ , which are then exactly the singular values. The input sequence of length 16 is generated using a uniform distribution on [0,1]. We use Transformers as defined in Equation (5) to learn these targets. The feed-forward part is constructed using a dense network with a width of 128 and a depth of 3 to ensure it has enough expressiveness. Moreover, we have $n=d_{v}=32$ and $m_{h}$ , which range from 1 to 16, to construct a model with different ranks. We use PyTorch default initialization and use normal training procedures with Adam Optimizer. We train enough epochs to ensure the loss does not decrease so that we can use the training error to estimate the approximation error.

Experiments in Section 5.1 In this example, we tested the ViT model on the CIFAR10 dataset. We consider the baseline model ViT\_B16 [11]. This model configuration utilizes 12 heads, and each head is of dimension 64. In Figure 1(a), we plot the singular values distribution of the attention matrix (before Softmax normalization) from the first attention head. We then vary the size of each attention head from 1 to 64 while keeping other configurations unchanged. We follow the training procedure as described in [11] to train the modified models.

# Experiments in Section 6

Comparison of the Transformer and RNNs We consider the following linear target relationship:

$$
H _ {t} (\boldsymbol {x}) = \sum_ {t = 0} ^ {t} \rho (s) x (t - s), \tag {66}
$$

with kernel $\rho(s) = \exp(-s)$ . The input sequence of length 32 is generated using a uniform distribution on [0,1]. For the RNN, we utilize a one-layer vanilla RNN architecture with 128 hidden units, employing a linear activation function. For the Transformer, we use the simplified structure as presented in Equation (8) with position encoding added. For parameter settings, we have $n = d_{v} = m_{h} = 32$ . The feed-forward part is constructed using a dense network with a width 128 and a depth 3.

To change the temporal ordering, we permute the input while keeping the output unchanged, this leads to a change in the temporal ordering of the target relationship. For all the input sequences, we exchange the first 10 values with the last 10 values, resulting in a change in the underlying temporal structure. For the temporal mixing operation, we use a randomly generated length 5 filter and the operation \* described in Section 6.2.

In Table 2, it is also interesting to note that for this target, the RNN performs better than the Transformer. This is potentially because temporal ordering is inherently important in this target, and the Transformer is not good at capturing such relationships. Some variants of the Transformer, such as Informer [36] and Autoformer [32], have been proposed specifically to address temporal relationships, particularly in sequence forecasting tasks. However, empirical results presented in Zeng et al. [35] indicate that these variants still cannot effectively capture temporal relationships.

As a comparison, we consider a different type of linear relationship where the temporal ordering is unimportant. We randomly generate the filter such that $\rho(s) \sim \mathcal{U}_{[0,1]}$ . This does not have temporal

ordering since if we permute $\rho$ , the distribution of it does not change. The results are shown in Table 3. We can observe that it is hard for RNN to learn this target. But the Transformer can learn this target, and a permutation of the input does not affect its performance.

<table><tr><td></td><td>RNN</td><td>Trans.</td></tr><tr><td>Original</td><td>5.58</td><td>5.43e-4</td></tr><tr><td>Permuted</td><td>5.24</td><td>5.36e-4</td></tr></table>

Table 3: The table presents the MSE values for both RNN and the Transformer datasets without temporal ordering.

Real-world datasets for the Transformer In Table 1, we test the performance of the Transformer on a permuted dataset considering real-world examples. For the ViT experiment, we use the ViT\_B16 model. For the WMT2014 English-German dataset, we use the original Transformer model as proposed in [30]. For both experiments, when testing on the permuted dataset, we initialize the model with parameters that are trained on the original dataset. To generate the permuted dataset, we fix a permutation such that the first 10 elements are shifted to the end of the sequence. This operation is applied to all the input sequences. Note that the permutation of the ViT dataset is before the addition of position encoding and after the convolution embedding.

# D Detailed Dissuasion of Section 6

In this section, we provide additional discussions related to Section 6. We first review the approximation results for the RNN, the we provide proofs for the propositions presented in Section 6.

# D.1 Introduce the approximation results for the RNN

We here review the results of approximation results of RNN presented in Li et al. [23]. Here, we consider input and output space defined by

$$
\mathcal {X} = \left\{\boldsymbol {x}: x (s) \in [ 0, 1 ] ^ {d}, s \in \mathbb {N} _ {\geq 0} \right\} \tag {67}
$$

and

$$
\mathcal {Y} = \left\{\boldsymbol {y}: y (s) \in \mathbb {R}, s \in \mathbb {N} _ {\geq 0} \right\}. \tag {68}
$$

There are infinite sequences with time indices starting from zero.

We consider the following dynamics for the linear RNN:

$$
h (t) = W h (t - 1) + U x (t) \tag {69}
$$

$$
y (t) = c ^ {\top} h (t), \tag {70}
$$

where $h \in \mathbb{R}^n$ with $h(0) = 0$ is the hidden state of the RNN. $W \in \mathbb{R}^{n \times n}$ , $U \in \mathbb{R}^{d \times n}$ and $c^\top \in \mathbb{R}^{n \times 1}$ are parameters. The approximation budget of the model is $n$ , which is the size of its hidden state. We can explicitly solve this dynamic where we get

$$
\hat {H} _ {t} (\boldsymbol {x}) = \sum_ {s = 0} ^ {t} c ^ {\top} W ^ {s} U x (t - s). \tag {71}
$$

Here, we assume the eigenvalue of $W$ has a negative real part to ensure it is stable when $s$ increases.

For the target, we consider a linear relationship represented by

$$
H _ {t} (\boldsymbol {x}) = \sum_ {s = 0} ^ {t} \rho (s) x (t - s), \tag {72}
$$

where $\| \rho(s) \| \leq \infty$ is a unique representation of $H$ . Thus, the approximation capability of the RNN is characterized by the properties of $\rho$ .

Note that $\hat{\rho}(s) = c^{\top}W^{s}U$ in the RNN exhibits an exponential decaying pattern. Thus, the target must also have a similar decay pattern to achieve a good approximation. We have precisely defined the following complexity measures for the RNN.

$$
C _ {1} (\boldsymbol {H}) = 1 + \inf \left\{\beta > 0: \lim _ {t \rightarrow \infty} e ^ {\frac {t}{\beta}} | \rho (t) | = 0 \right\}. \tag {73}
$$

This measures the decaying speed of the RNN, we assume $\rho$ exhibits an exponential decay, with decaying speed measured by $C_1$ . The next complexity considers the norm of $\rho$ :

$$
C _ {2} (\boldsymbol {H}) = \sup \left\{e ^ {\frac {t}{C _ {1} (\boldsymbol {H})}} | \rho (t) |: t \geq 0 \right\}. \tag {74}
$$

Combined together, we define the complexity measure for RNN as

$$
C _ {\mathrm{RNN}} = C _ {1} (\boldsymbol {H}) \cdot C _ {2} (\boldsymbol {H}) \tag {75}
$$

This considers the magnitude of $\rho$ , $C_2$ is small if $\rho$ does not have a large value far from the origin. We define the RNN approximation space to be targets that have finite complexity measures

$$
\mathcal {C} _ {\mathrm{RNN}} = \{\boldsymbol {H} \text {   satisfies   Equation   (72) }: C _ {\mathrm{RNN}} <   \infty \}. \tag {76}
$$

The approximation rate of the RNN follows $\frac{d \cdot C_{RNN}}{m}$ .

# D.2 Discussion of Section 6.1

Proof of Proposition 6.1 We now present the proof for Proposition 6.1

Proof. Let $p: \mathbb{N}_{\geq 0} \to \mathbb{N}_{\geq 0}$ be a permutation on the time indices. We consider an altered target

$$
\tilde {H} _ {t} (\boldsymbol {x} \circ p) = H _ {t} (\boldsymbol {x}) \tag {77}
$$

We then have

$$
\tilde {H} _ {t} (\boldsymbol {x} \circ p) = \sum_ {s = 0} ^ {t} \tilde {\rho} (s) x (p (t - s)) \tag {78}
$$

Let's consider the most simple case where $p$ only permute two time indices such that $p(t_1) = t_2$ and $t_2 > t_1$ . We consider the output at $t_1$

$$
H _ {t _ {1}} (\boldsymbol {x}) = \sum_ {s = 0} ^ {t _ {1}} \rho (s) x (t _ {1} - s) \tag {79}
$$

$$
= \sum_ {s = 1} ^ {t _ {1}} \rho (s) x (t _ {1} - s) + \rho (0) x (t _ {1}). \tag {80}
$$

However, for $\tilde{H}_{t_1}(\pmb{x})$ we have

$$
\tilde {H} _ {t _ {1}} (\boldsymbol {x} \circ p) = \sum_ {s = 0} ^ {t _ {1}} \tilde {\rho} (s) x (p (t _ {1} - s)) \tag {81}
$$

$$
= \sum_ {s = 1} ^ {t _ {1}} \tilde {\rho} (s) x (t _ {1} - s) + \tilde {\rho} (0) x (t _ {2}). \tag {82}
$$

Note the in this case the output at $t_{1}$ depends on a future value of input $x(t_{2})$ , this means that $\tilde{H}$ is no longer causal. Consequently, $\tilde{H}$ cannot be written in the form of Equation (72), thus not belong to the RNN approximation space. ☐

# Proof of Proposition 6.2 We next consider the Proposition 6.2

Proof. For a fixed permutation $p: \tau \to \tau$ , we consider $\tilde{H}_t(\boldsymbol{x} \circ p) = H_t(\boldsymbol{x})$ . Suppose the altered target has the following form:

$$
\tilde {H} _ {t} (\boldsymbol {x} \circ p) = \tilde {F} \left(\sum_ {s = 1} ^ {\tau} \sigma [ \tilde {\rho} (x (p (t)), x (\cdot)) ] (p (s)) \tilde {f} (p (x (s)))\right), \tag {83}
$$

we next show that $\tilde{F},\tilde{f}$ and $\tilde{\rho}$ has same complexity measures as $F,f$ and $\rho$ . Firstly, we note that we can remove the permutation on index $s$ because the ordering of a summation does not affect the result.

$$
\tilde {H} _ {t} (\boldsymbol {x} \circ p) = \tilde {F} \left(\sum_ {s = 1} ^ {\tau} \sigma [ \tilde {\rho} (x (p (t)), x (\cdot)) ] (s) \tilde {f} (s))\right). \tag {84}
$$

We now consider the POD expansion of $\tilde{\rho}$ where

$$
\tilde {\rho} (x (p (t)), x (s)) = \sum_ {i = 1} ^ {r} \sigma_ {i} \tilde {\phi} _ {i} (x (p (t))) \tilde {\psi} _ {i} (x (s)). \tag {85}
$$

Recall that our input space is defined as Equation (7). Where $x_{s} \in I_{s}$ such that $I_{i}$ and $I_{j}$ are closed disjoint sets. Thus, for a pointwise function $\phi$ , we can write it into the following piecewise form

$$
\phi (x) = \phi^ {(t)} (x) \quad x \in \mathcal {I} _ {t}. \tag {86}
$$

Let $\tilde{\phi}_i(x) = \phi_i^{(p(t))}(x)$ for $x\in \mathcal{I}_t$ . This is essentially the permutation of the disjoint pieces of $\tilde{\phi}_i$ . Thus we have

$$
\tilde {\phi} _ {i} (x (p (t))) = \phi_ {i} ^ {(t)} (x (t)). \tag {87}
$$

Consequently, we have $\tilde{\rho}(x(p(t)), x(s)) = \rho(x(t), x(s))$ . Finally let $\tilde{F} = F$ and $\tilde{f} = f$ we achieved $\tilde{H}_t(\boldsymbol{x} \circ p) = H_t(\boldsymbol{x})$ . Since $\tilde{F}$ and $\tilde{f}$ remain unchanged, we only need to consider the complexity measure for $\rho$ . We note that the rank of the POD is unaffected. We remain with $C_{\mathrm{FF}}(\tilde{\phi}_i)$ . The function $\tilde{\phi}_i$ is a permutation of pieces of the piecewise function $\phi_i$ . We next discuss the kind of approximation schemes that $C_{\mathrm{FF}}(\tilde{\phi}_i)$ is unchanged under this operation.

Since the set I is disconnect, we apply Tietze extension theorem to extend $\phi_{i}$ to its convex hull, denoted as $\Phi_{i}$ , such that $\Phi_{i}(x)=\phi_{i}(x)$ for $x\in I$ and $\sup\{|\Phi_{i}(x)|\}=\sup\{| \phi_{i}(x)|\}$ .

We first consider the polynomial approximation as we introduced in Section 3. In this case, we have $C_{\mathrm{FF}}(\phi_i) = \max_{r=1 \ldots \alpha} \| \phi_i \|_\infty$ . We next show that $C_{\mathrm{FF}}(\tilde{\phi}_i) = C_{\mathrm{FF}}(\phi_i)$ . Since $\mathcal{I}$ is closed and $\tilde{\phi}$ is continuous, we apply Tietze extension theorem to extend $\tilde{\phi}_i$ to $\mathbb{R}^d$ denoted as $\tilde{\Phi}_i$ , such that $\tilde{\Phi}_i(x) = \tilde{\phi}_i(x)$ for $x \in \mathcal{I}$ and $\sup \{|\tilde{\Phi}_i(x)|\} = \sup \{|\tilde{\phi}_i(x)|\}$ . This implies that $C_{\mathrm{FF}}(\tilde{\phi}_i) = C_{\mathrm{FF}}(\phi_i)$ . Thus, in the case of polynomial approximation, the complexity measures remain unchanged.

Next, we consider the approximation using the ReLu network. We assume $\phi_{i}$ to be Hölder continuous such that

$$
\left| f (x) - f (y) \right| \leq C \| x - y \| ^ {\alpha}. \tag {88}
$$

The Tietze extension theorem states we can extend it to $\Phi_{i}$ with the same C and $\alpha$ .

As shown in Shen et al. [28], the approximation capability of a ReLU network on a Hölder function depends on C and $\alpha$ . Where functions with small C, $\alpha$ can be approximated more effectively. Since $\tilde{\phi}_{i}$ is a permutation of disconnected pieces of $\phi_{i}$ , its Hölder constant remains unchanged. Thus, its $\tilde{\Phi}_{i}$ has same Hölder constant as $\Phi_{i}$ , which implies the complexity measure is the same.

![](images/84bed005a8ab6806c4fee80a34da6592fcf7b8b44e36b6247ed881ab7124bd0b.jpg)

In the above proof, we considered two different approximation schemes where the complexity measures of $\tilde{H}$ remain unchanged.

# D.3 Discussion for Section 6.2

Proof of Proposition 6.3 We first discuss Proposition 6.3. Consider a filter $\theta$ which is a length $l$ filter with $\| \theta \|_1 \leq 1$ . The altered target is defined by

$$
\tilde {H} _ {t} (\boldsymbol {x}) = H _ {t} (\theta * \boldsymbol {x}) = \sum_ {s = 0} ^ {t} \rho [ s ] (\theta * x) [ t - s ]. \tag {89}
$$

Rearrange the summation, and we get

$$
\begin{array}{l} \tilde {H} _ {t} (\boldsymbol {x}) = \sum_ {s = 0} ^ {t} \rho [ s ] (\theta * x) [ t - s ] (90) \\ = \sum_ {s = 0} ^ {t} \rho (s) \sum_ {s ^ {\prime} = 0} ^ {l - 1} \theta [ s ^ {\prime} ] x (t - s + s ^ {\prime}) (91) \\ = \sum_ {s = 0} ^ {t} \sum_ {s ^ {\prime} = 0} ^ {l - 1} \theta [ s ^ {\prime} ] \rho (s) x (t - s + s ^ {\prime}) (92) \\ = \sum_ {s = 0} ^ {t} \sum_ {s ^ {\prime} = 0} ^ {l - 1} \theta [ s ^ {\prime} ] \rho [ s + s ^ {\prime} ] x [ t - s ] (93) \\ = \sum_ {s = 0} ^ {t} (\theta * \rho) [ s ] x [ t - s ] (94) \\ = \sum_ {s = 0} ^ {t} \tilde {\rho} [ s ] x [ t - s ]. (95) \\ \end{array}
$$

Thus, we have the altered kernel denoted as $\tilde{\rho} = \theta * \rho$ .

Proof. Firstly, for $|\rho(s)| \leq \exp(-\frac{s}{\beta})$ , by the definition of $C_1$ and $C_2$ we have $C_1(\boldsymbol{H}) \leq \beta$ and $C_2(\boldsymbol{H}) \leq 1$ .

By assumption we have $\rho$ exhibits exponential decay such that $|\rho(s)| \leq e^{-\beta s}$ for some $\beta$ . We then have

$$
| \tilde {\rho} (s) | = | (\theta * \rho) [ t ] | = \sum_ {s ^ {\prime} = 0} ^ {l - 1} | \theta (s ^ {\prime}) \rho (s + s ^ {\prime}) | \leq e ^ {- \beta s} \| \theta \| _ {1} = e ^ {- \beta s}, \tag {96}
$$

which shows that $\tilde{\rho}[t]$ is also bounded by $e^{-\beta s}$ . This implies that $C_{1}(\tilde{\boldsymbol{H}}) \leq \beta$ and $C_{2}(\tilde{\boldsymbol{H}}) \leq 1$ .

![](images/452e83aed4ec2a120f2c5b5614e1ef5e7bd8fe29e33444ba4df994efdeeb09c6.jpg)

Transformer Affected by Temporal Mixing Finally, we present an example where a temporal mixing in the input will affect the target's rank. For simplicity we consider the case where $d = 1$ and $\tau = 2$ , and $x_{1}, x_{2} \in [0,1]$ . Note that here, we do not assume $x_{1}$ and $x_{2}$ belong to disjoint intervals for simplicity. For target in the form of Equation (8), we consider a temporal coupling term $\rho$ defined as

$$
\rho (x _ {1}, x _ {2}) = 1 + \sqrt {2} \sin (2 \pi x _ {1}) \cdot \sqrt {2} \sin (2 \pi x _ {2}). \tag {97}
$$

In this case, $\rho(u,v)$ is of rank 2 with both singular values equal to 1. Consider a new input $\tilde{\boldsymbol{x}} = (x_1,x_2 + x_1)$ with temporal mixing. We then have

$$
\rho (x _ {1}, x _ {2} + x _ {1}) = 1 + \sqrt {2} \sin (2 \pi x _ {1}) \cdot \sqrt {2} \sin (2 \pi (x _ {1}, x)). \tag {98}
$$

We numerically estimate the singular values of $\tilde{\rho}$ to get $\sigma_{1}=1.265$ , $\sigma_{2}=0.5$ and $\sigma_{3}=0.4$ . The rank of $\tilde{\rho}$ increases, and there are also extra bases included. Here, we only consider the rank $\rho$ from the same representation, while we do not exclude the possibility that there exist other representations that may have another rank pattern. Indeed, analyzing the temporal mixing in the Transformer is non-trivial; we only provided an example where the rank increases. Conversely, a suitable deconvolution process can also decrease the complexity measure of the target. We leave it as a future direction for analysis.

# NeurIPS Paper Checklist

# 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction accurately reflect the paper's contributions and scope.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.   
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.   
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.   
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

# 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We made simplification and assumptions for theoretical analysis. However, we performed empirical experiments to demonstrate the implication of the theoretical results also holds true in general settings.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.   
- The authors are encouraged to create a separate "Limitations" section in their paper.   
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.   
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.   
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.   
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.   
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.   
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

# 3. Theory Assumptions and Proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

# Answer: [Yes]

Justification: All the theoretical settings and assumptions are presented. Complete proofs of theorems are presented in the appendices.

# Guidelines:

- The answer NA means that the paper does not include theoretical results.   
- All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.   
- All assumptions should be clearly stated or referenced in the statement of any theorems.   
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.   
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.   
- Theorems and Lemmas that the proof relies upon should be properly referenced.

# 4. Experimental Result Reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

# Answer: [Yes]

Justification: We conducted numerical experiments to demonstrate our theoretical results, settings are presented in the appendices.

# Guidelines:

- The answer NA means that the paper does not include experiments.   
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.   
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.   
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general, releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.   
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example   
(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.   
(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.   
(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).   
(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

# 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [NA]

Justification: We conducted numerical experiments, with settings clearly presented in the paper.

# Guidelines:

- The answer NA means that paper does not include experiments requiring code.   
- Please see the NeurIPS code and data submission guidelines (https://nips.cc/public/guides/CodeSubmissionPolicy) for more details.   
- While we encourage the release of code and data, we understand that this might not be possible, so “No” is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).   
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https://nips.cc/public/guides/CodeSubmissionPolicy) for more details.   
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.   
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.   
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).   
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

# 6. Experimental Setting/Details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: Detailed experiment settings are presented in the appendices.

# Guidelines:

- The answer NA means that the paper does not include experiments.   
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.   
- The full details can be provided either with the code, in appendix, or as supplemental material.

# 7. Experiment Statistical Significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: Our numeric demonstrations does not require error bars.

# Guidelines:

- The answer NA means that the paper does not include experiments.   
- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.   
- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)   
- The assumptions made should be given (e.g., Normally distributed errors).   
- It should be clear whether the error bar is the standard deviation or the standard error of the mean.   
- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a $96\%$ CI, if the hypothesis of Normality of errors is not verified.   
- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).   
- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

# 8. Experiments Compute Resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [NA]

Justification: Our paper does not contain computational extensive experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.   
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.   
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.   
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

# 9. Code Of Ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The Code of Ethics is followed.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.   
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.   
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

# 10. Broader Impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Our results are theoretical, and there is no societal impact.

Guidelines:

- The answer NA means that there is no societal impact of the work performed.   
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.   
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.   
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.   
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

# 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: Our paper have no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.   
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.   
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.   
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

# 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: Our paper does not use existing assets.

Guidelines:

- The answer NA means that the paper does not use existing assets.   
- The authors should cite the original paper that produced the code package or dataset.   
- The authors should state which version of the asset is used and, if possible, include a URL.   
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.   
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.   
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.   
- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

\- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

# 13. New Assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: Our paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.   
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.   
- The paper should discuss whether and how consent was obtained from people whose asset is used.   
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

# 14. Crowdsourcing and Research with Human Subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Our paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.   
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.   
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

# 15. Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: Our paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.   
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.   
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.   
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.