# GPEX, A Framework For Interpreting Artificial Neural Networks

Amir Akbarnejad

Department of Computing Science

University of Alberta

Edmonton, AB, Canada

ah8@ualberta.ca

Gilbert Bigras

Department of Laboratory Medicine and Pathology

University of Alberta

Edmonton, AB, Canada

Gilbert.Bigras@albertaprecisionlabs.ca

Nilanjan Ray

Department of Computing Science

University of Alberta

Edmonton, AB, Canada

nray1@ualberta.ca

# Abstract

The analogy between Gaussian processes (GPs) and deep artificial neural networks (ANNs) has received a lot of interest, and has shown promise to unbox the blackbox of deep ANNs. Existing theoretical works put strict assumptions on the ANN (e.g. requiring all intermediate layers to be wide, or using specific activation functions). Accommodating those theoretical assumptions is hard in recent deep architectures, and those theoretical conditions need refinement as new deep architectures emerge. In this paper we derive an evidence lower-bound that encourages the GP's posterior to match the ANN's output without any requirement on the ANN. Using our method we find out that on 5 datasets, only a subset of those theoretical assumptions are sufficient. Indeed, in our experiments we used a normal ResNet-18 or feed-forward backbone with a single wide layer in the end. One limitation of training GPs is the lack of scalability with respect to the number of inducing points. We use novel computational techniques that allow us to train GPs with hundreds of thousands of inducing points and with GPU acceleration. As shown in our experiments, doing so has been essential to get a close match between the GPs and the ANNs on 5 datasets. We implement our method as a publicly available tool called GPEX: https://github.com/amirakbarnejad/gpex. On 5 datasets (4 image datasets, and 1 biological dataset) and ANNs with 2 types of functionality (classifier or attention-mechanism) we were able to find GPs whose outputs closely match those of the corresponding ANNs. After matching the GPs to the ANNs, we used the GPs' kernel functions to explain the ANNs' decisions. We provide more than 200 explanations (around 30 explanations in the paper and the rest in the supplementary) which are highly interpretable by humans and show the ability of the obtained GPs to unbox the ANNs' decisions.

# 1 Introduction

Artificial neural networks (ANNs) are widely adopted in machine learning. Despite their benefits, ANNs are known to be black-box to humans, meaning that their inner mechanism for making predictions is not necessarily interpretable/explainable to humans. ANN's black-box property impedes its deployment in safety-critical applications like medical imaging or autonomous driving, and makes them hard-to-troubleshoot for machine learning researchers.

Attribution-based explanation methods like LIME[31], SHAP[21] and most gradient-based explanation methods like DeepLIFT [5] presume a linear surrogate model. Given a test instance $x_{test}$ , this simpler surrogate model is encouraged to have the same output "locally" around $x_{test}$ . Because of this "local assumptions", explanations from these methods might be unreliable, and can be easily manipulated by an adversary model [11][28]. Moreover, these models may produce discordant explanations for a fixed model and test instance [16].

Considering Gaussian processes (GPs) [26] as the explainer model is beneficial, because: 1. Gaussian processes are highly interpretable. 2. Researchers have long known that GP's posterior has the potential to match an ANN's output "globally". More precisely, given an ANN and some requirements on it [24][8], there might exist a GP whose posterior matches the ANN's output all over the input-space $\mathbb{X}$ (as opposed to the local explanation models for which the match happens only locally around a test instance $x_{test} \in \mathbb{X}$ ). Not many explainer models can globally match the ANN's output. Among gradient-based methods, with the best of our knowledge only Integrated Gradients [33] has a weak sense of ANN's global behaviour over the input space. Having some conditions on an ANN, representer point selection [37] finds a "globally faithfull" explainer model that, similar to GPs, works with a kernel function. As we will elaborate upon in Sec. 4.3 and Sec. S6 of the supplementary, the GP's kernel that we find in this paper is superior due to a technical point in the formulation of representer point selection [37]. All in all, using GPs to explain ANNs is quite promising and has advantages over other approaches to explain ANNs.

The contributions of this paper are as follows:

- Theoretical results on ANN-GP analogy impose some restrictions on ANNs under which the ANN will be equivalent to a GP. These conditions are too restrictive for recently used deep architectures. Moreover, those theoretical conditions need refinement as new deep architectures emerge. In this paper we derive an ELBO for training GPs which encourages GP's posterior to match ANN's output. Our formulation and method doesn't impose any restriction on the ANN and the method used to train it.   
- Using our method, we empirically show that on 5 datasets (4 image datasets, and 1 biological dataset) and ANNs with 2 types of functionality (classifier or attention-mechanism) the ANN needs to fulfill only a subset of those theoretical conditions. Indeed, in our experiments we used a normal ResNet-18 or feed-forward backbone with a single wide layer in the end.   
- Scalability is a major issue in training GPs. To address this issue, we adopted computational techniques recently used for fast spectral clustering [13] as well as a novel method to learn the GPs using mini-batches of inducing points and training instances. These computational techniques allow us to train GPs with hundreds of thousands of inducing points. According to our analysis, increasing the inducing points has been essential to get a good match between the trained GPs and ANNs. Indeed, without many inducing points GPs posterior cannot be a complex function (a function with many ups and downs [36]) and fails to match ANNs' output.   
- With the best of our knowledge, our work is the first method that performs knowledge distillation between GPs and ANNs.   
- We implement our method as a public python library called GPEX (Gaussian Processes for EXplaining ANNs). GPEX takes in an arbitrary PyTorch module, and replaces any ANN submodule of choice by GPs. Our package makes use of GPU-acceleration, and enables effortless application of GPs without getting users involved in details of the inference procedure. GPEX can be used by machine learning researchers to interpret/troubleshoot their artificial neural networks. Moreover, GPEX can be used by researchers working on the theoretical side of ANN-GP analogy to empirically test their hypotheses.

# 2 Proposed Method

# 2.1 Notation

In this article the function $g(.)$ always denotes an ANN. The kernel of a Gaussian process is denoted by the double-input function $\mathcal{K}(.,.)$ . We assume the kernel similarity between two instances $x_{i}$ and $x_{j}$ is equal to $f(\boldsymbol{x}_{i})^{T}f(\boldsymbol{x}_{j})$ , where $f(.)$ maps the input-space to the kernel space. In this paper u (resp. v) denotes a vector in the kernel-space (resp. the posterior mean) of a GP. In some sense u and v denote

![](images/d3f16442fc140772495de8d1e4253896fef3c26da1238a54d94b77f4c8e7bf43.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["general feed-forward module"] --> B["x"]
    B --> C["ANN"]
    C --> D["v"]
    D --> E["y"]
    F["Observed Data"] --> G["GP's mean"]
    G --> H["GP's mean ± covariance"]
    H --> I["-1.0 to 3.0"]
    I --> J["-3.0 to 0.0"]
    J --> K["0.0 to 3.0"]
    K --> L["1.0 to 3.0"]
    L --> M["2.0 to 3.0"]
    M --> N["3.0 to 3.0"]
```
</details>

Figure 1: a) A general feed-forward pipeline, with an ANN sub-module to be explained by GPEX. b) Typical behaviour of Gaussian process posterior given a set of observed values.

the input and the output of a GP, respectively. We have that: $\mathcal{K}(\pmb{x}_i,\pmb{x}_j) = f(\pmb{x}_i)^T f(\pmb{x}_j) = \pmb{u}_i^T\pmb{u}_j$ . The number of GPs is equal to the number of the outputs from the ANN. In other words, we consider one GP per scalar output-head from the ANN. We use index $\ell$ to specify the $\ell$ -th GP as follows: $\mathcal{K}_{\ell}(\pmb{x}_i,\pmb{x}_j) = f_{\ell}(\pmb{x}_i)^T f_{\ell}(\pmb{x}_j) = \pmb{u}_i^{(\ell)T}\pmb{u}_j^{(\ell)}$ . We parameterize the $\ell$ -th GP by a set of $M$ inducing points $\{(\tilde{\pmb{u}}_m^{(\ell)},\tilde{v}_m^{(\ell)})\}_{m=1}^M$ . The tilde in $(\tilde{\pmb{u}}_m^{(\ell)},\tilde{v}_m^{(\ell)})$ indicates that $\tilde{\pmb{u}}$ is one of the $M$ inducing points in the kernel space. However, $\pmb{u}$ (without tilde) can be an arbitrary point in the continuous kernel-space.

# 2.2 The Proposed Framework

To make our framework as general as possible, we consider a general feed-forward pipeline that contains an ANN as a submodule. In Fig. 1a the bigger square illustrates the general module. The input-output of the general pipeline are denoted in Fig. 1a by X and Y. The general pipeline has at least one ANN submodule to be explained by GPEX. Fig. 1a illustrates this ANN by the small blue rectangle within the general pipeline. The input-output of the ANN are denoted in Fig. 1a by x and v. Note that X and Y can be anything, including without any limitation, a set of vectors, labels, and meta-information. However, input-output of the ANN (i.e. x and y) are required to be in tensor format. The exact requirements are provided in the online documentation for GPEX. Moreover, the general module can have other arbitrary submodules, which are depicted by the blue clouds. The relations between the submodules, as illustrated by the dotted-lines in Fig. 1a, can also be quite general. Our probabilistic formulation only needs access to the conditional distributions $p(\mathbf{x}|\mathcal{X})$ and $p(\mathcal{Y}|\mathbf{x},\mathcal{X})$ . Similarly, the proposed GPEX is completely agnostic about the general pipeline and it only requires the ANN's input-output to be in the tensor format. Given a PyTorch module, the proposed GPEX tool automatically grabs the distributions $p(\mathbf{x}|\mathcal{X})$ and $p(\mathcal{Y}|\mathbf{x},\mathcal{X})$ from the main module it is given.

The inducing points $\{\tilde{\boldsymbol{u}}_m^{(\ell)},\tilde{v}_m^\ell \}_{m = 1}^M$ parameterize the $\ell$ -th GP. Note that $\tilde{\boldsymbol{u}}_m^{(\ell)} = f_\ell (\tilde{\boldsymbol{x}}_m)$ . A feature point like $\boldsymbol{x}$ is first mapped to the kernel-space as $\boldsymbol{u}^{(\ell)} = f_{\ell}(\boldsymbol{x})$ . Note that the kernel functions $\{f_{\ell}(.)\}_{\ell = 1}^{L}$ are implemented as separate neural networks, or for the sake of efficiency as a single neural network backbone with $L$ different heads. Afterwards, the GP's posterior on $\boldsymbol{x}$ depends on the kernel similarities between $\boldsymbol{u}^{(\ell)}$ and the inducing points $\{\tilde{\boldsymbol{u}}_m^{(\ell)}\}_{m = 1}^M$ . More precisely, the posterior of the $\ell$ -th GP on $\boldsymbol{x}$ is a random variable $v^{(\ell)}$ whose distribution is as follows [26]:

$$
p \left(v ^ {(\ell)} \mid \boldsymbol {u} ^ {(\ell)}, \tilde {\mathbf {u}} _ {1: M} ^ {(\ell)}, \tilde {v} _ {1: M} ^ {(\ell)}\right) = \mathcal {N} \left(v ^ {(\ell)}; \mu_ {v} \left(\boldsymbol {u} ^ {(\ell)}, \tilde {\mathbf {u}} _ {1: M} ^ {(\ell)}, \tilde {v} _ {1: M} ^ {(\ell)}\right), c o v _ {v} \left(\boldsymbol {u} ^ {(\ell)}, \tilde {\mathbf {u}} _ {1: M} ^ {(\ell)}, \tilde {v} _ {1: M} ^ {(\ell)}\right)\right), \tag {1}
$$

where $\mu_v(.,.,.)$ and $cov_v(.,.,.)$ are the mean and covariance of a GP's posterior computed as:

$$
\mu_ {v} \left(\boldsymbol {u} ^ {(\ell)}, \tilde {\mathbf {u}} _ {1: M} ^ {(\ell)}, \tilde {v} _ {1: M} ^ {(\ell)}\right) = \mathcal {K} \left(\boldsymbol {u} ^ {(\ell)}, \tilde {\mathbf {u}} _ {1: M} ^ {(\ell)}\right) \left[ \mathcal {K} \left(\tilde {\mathbf {u}} _ {1: M} ^ {(\ell)}, \tilde {\mathbf {u}} _ {1: M} ^ {(\ell)}\right) + \sigma_ {g p} ^ {2} \mathbf {I} _ {M \times M} \right] ^ {- 1} \tilde {v} _ {1: M} ^ {(\ell)} \tag {2}
$$

and

$$
c o v _ {v} \left(\boldsymbol {u} ^ {(\ell)}, \tilde {\mathbf {u}} _ {1: M} ^ {(\ell)}, \tilde {v} _ {1: M} ^ {(\ell)}\right) = \mathcal {K} \left(\boldsymbol {u} ^ {(\ell)}, \boldsymbol {u} ^ {(\ell)}\right) -
$$

$$
\mathcal {K} (\boldsymbol {u} ^ {(\ell)}, \tilde {\boldsymbol {u}} _ {1: M} ^ {(\ell)}) \times \left[ \mathcal {K} (\tilde {\boldsymbol {u}} _ {1: M} ^ {(\ell)}, \tilde {\boldsymbol {u}} _ {1: M} ^ {(\ell)}) + \sigma_ {g p} ^ {2} \mathbf {I} _ {M \times M} \right] ^ {- 1} \mathcal {K} (\tilde {\boldsymbol {u}} _ {1: M} ^ {(\ell)}, \boldsymbol {u} ^ {(\ell)}). \tag {3}
$$

As the variables $\{v_m^{(\ell)}\}_{m = 1}^M$ and $v$ are latent or hidden, we train the model parameters by optimizing a variational lower-bound. We consider the following variational distributions:

$$
q _ {1} (v ^ {(\ell)} \mid \boldsymbol {x}) = \mathcal {N} \big (v ^ {(\ell)}; g _ {\ell} (\boldsymbol {x}), \sigma_ {g} ^ {2} \big), \quad q _ {2} \big (\tilde {v} _ {m} ^ {(\ell)} \big) = \mathcal {N} \big (\tilde {v} _ {m} ^ {(\ell)}; \varphi_ {m} ^ {(\ell)}, \sigma_ {\varphi} ^ {2} \big). \tag {4}
$$

In Eq. 4, the function $g_{\ell}(.)$ is the $\ell$ -th output from the ANN. Note that as the set of hidden variables $\{\tilde{v}_m^{(\ell)}\}_{m=1}^M$ is finite, we have parameterized their variational distribution by a finite set of numbers $\{\varphi_m^{(\ell)}\}_{m=1}^M$ . However, as the variables $x$ can vary arbitrarily in the feature space, the variable $u^{(\ell)}$ varies arbitrarily in the kernel space. Therefore, the set of values $v^{(\ell)}$ may be infinite. Accordingly, the variational distribution for $v^{(\ell)}$ is conditioned on $x$ and is parameterized by the ANN $g(.)$ .

# 2.3 The Derived Evidence Lower-Bound (ELBO)

Due to space limitation, the derivation of the lower-bound is moved to Sec. S1 of the supplementary material. In this section we only introduce the derived ELBO and discuss how it relates the GP, the ANN and the training cost of the main module in an intuitive way. The ELBO terms containing the GP parameters (i.e. the parameters of the kernel function $f(.))$ is denoted by $L_{gp}$ . According to Eq. S9 of the supplementary material $L_{gp}$ is as follows:

$$
\begin{array}{l} \mathcal {L} _ {g p} = - \frac {1}{2} \mathbb {E} _ {\sim q} \big [ \sum_ {\ell = 1} ^ {L} \frac {(\mu_ {v} (\pmb {u} ^ {(\ell)} , \tilde {\pmb {u}} _ {1 : M} ^ {(\ell)} , \tilde {v} _ {1 : M} ^ {(\ell)}) - g _ {\ell} (\pmb {x})) ^ {2} + \sigma_ {g} ^ {2}}{c o v _ {v} (\pmb {u} ^ {(\ell)} , \tilde {\pmb {u}} _ {1 : M} ^ {(\ell)} , \tilde {v} _ {1 : M} ^ {(\ell)})} \big ] \\ - \frac {1}{2} \mathbb {E} _ {\sim q} \left[ \sum_ {\ell = 1} ^ {L} \log \left(\frac {\operatorname{cov} _ {v} \left(\boldsymbol {u} ^ {(\ell)} , \tilde {\boldsymbol {u}} _ {1 : M} ^ {(\ell)} , \tilde {v} _ {1 : M} ^ {(\ell)}\right)}{\sigma_ {g} ^ {2}}\right) \right] + (\text { const. }), \tag {5} \\ \end{array}
$$

where $q(.)$ is the variational distribution that factorizes to the $q_{1}(.)$ and $q_{2}(.)$ distributions defined in Eq. 4. In the first term of Eq. 5, the numerator encourages the GP and the ANN to have the same output. More precisely, for a feature point x we can compute the corresponding point in the kernel space as $\boldsymbol{u}^{(\ell)} = f_{\ell}(\boldsymbol{x})$ and then compute the GP's posterior mean based on kernel similarities between u and the inducing points to get the GP's mean $\mu_{v}$ . In Eq. 5 the GP's mean $\mu_{v}$ is encouraged to match the ANN's output $g_{\ell}(\boldsymbol{x})$ . In Eq. 5, because of the denominator of the first term, the ANN-GP similarity is not encouraged uniformly over the feature-space. Wherever the GP's uncertainty is low, the term $cov_{v}(\boldsymbol{u}^{(\ell)}, \tilde{\mathbf{u}}_{1:M}^{(\ell)}, \tilde{v}_{1:M}^{(\ell)})$ in the denominator becomes small. Therefore, the GP's mean is highly encouraged to match the ANN's output. On the other hand, in regions where the GP's uncertainty is high, the GP-ANN analogy is less encouraged. This formulation is quite intuitive according to the behaviour of Gaussian processes. Fig. 1b illustrates the posterior of a GP with radial-basis kernel for a given set of observations. In regions like $[3, \infty)$ and $(-\infty, -4]$ there are no nearby observed data. Therefore, in these regions the GP is highly uncertain and the blue uncertainty margin is thick in such regions. Intuitively, our derived ELBO in Eq. 5 encourages the GP-ANN analogy only when GP's uncertainty is low and gives less importance to regions similar to $[3, \infty)$ and $(-\infty, -4]$ in Fig. 1b. Note that this formulation makes no difference for the ANN as ANNs are known to be global approximators. However, this formulation makes a difference when training the GP, because the GP is not required to match the ANN in regions where there are no similar training instances. The ELBO terms containing the ANN parameters is denoted by $L_{ann}$ . According to Sec. S1.2 of the supplementary material, $L_{ann}$ is as follows:

$$
\mathcal {L} _ {a n n} = - \frac {1}{2} \mathbb {E} _ {\sim q} \left[ \sum_ {\ell = 1} ^ {L} \frac {\left(\mu_ {v} \left(\boldsymbol {u} ^ {(\ell)} , \tilde {\mathbf {u}} _ {1 : M} ^ {(\ell)} , \tilde {v} _ {1 : M} ^ {(\ell)}\right) - g _ {\ell} (\boldsymbol {x})\right) ^ {2}}{c o v _ {v} \left(\boldsymbol {u} ^ {(\ell)} , \tilde {\mathbf {u}} _ {1 : M} ^ {(\ell)} , \tilde {v} _ {1 : M} ^ {(\ell)}\right)} \right] + \mathbb {E} _ {\sim q} \left[ \log p (\mathcal {Y} | \boldsymbol {y}, \mathcal {X}) \right]. \tag {6}
$$

In the above objective the first term encourages the ANN to have the same output as the GP. Similar to the objective of Eq. 5, the denominator of the first term gives more weight to ANN-GP analogy when GP's uncertainty is low. In the right-hand-side of Eq. 6, the second term is the likelihood of the pipeline's output(s), i.e. Y in Fig. 1a. This term can be, e.g., the cross-entropy loss when Y contains class scores in a classification problem, or the mean-squared error when Y is the predicted value for a regression problem, or a combination of those costs in a multi-task setting.

# 3 Algorithm

We consider a separate Gaussian process for each output head of an ANN. In other words, given an ANN we have as many GPs as the number of the ANN's output heads. To explain an ANN, we find the explainer GPs by optimizing the objective in Eq. 5 w.r.t. to the kernel mappings $\{f_{\ell}(.)\}_{\ell = 1}^{L}$ . To do so, we need to have $\mu_v$ which in turn means we need to have all kernel-space representations

Algorithm 1 Method Optim\_KernMappings   
Input: Input instance x and inducing instance $\tilde{x}$ , list of matrices U, list of vectors V.
Output: Kernel-space mappings $[f_{1}(.),...,f_{L}(.)]$ .
Initialisation : loss ← 0.
1: $\mu$ , cov ← forward_GP(x, $\tilde{x}$ , U, V) //feed x to GPs, "forward_GP" is Alg.S1 in supplementary.
2: $\mu_{ann} \leftarrow g(\mathbf{x})$ //feed x to ANN.
3: for $\ell = 1$ to L do
4: loss ← loss + $\frac{(\mu[\ell]-\mu_{ann}[\ell])^{2}+\sigma_{g}^{2}}{cov[\ell]} + \log(cov[\ell])$ . //Eq.5.
5: end for
6: $\delta \leftarrow \frac{\partial loss}{\partial params([f_{1}(.),...,f_{L}(.)])}$ . //the gradient of loss.
7: params([f_{1},...,f_{L})] ← params([f_{1},...,f_{L})] - lr × $\delta$ //update the parameters.
8: lr ← updated learning rate
9: return [f_{1}(.),...,f_{L}(.)]

Algorithm 2 Method Explain\_ANN   
Input: Training dataset ds_train, and the inducing dataset ds_inducing.
Output: Updated kernel-space mappings $[f_{1}(.),...,f_{L}(.)]$ , and the other GP parameters U and V.
Initialisation : U, V ← Init_GPparams(ds_inducing) //Alg.S3 in supplementary.
1: for iter = 1 to max_iter do
2:    x ← randselect(ds_train).
3: $\tilde{x} \leftarrow$ randselect(ds_inducing)
4: $[f_{1}(.),...,f_{L}(.)] \leftarrow$ Optim_KernMapings(x, $\tilde{x}$ , U, V).
5: $\tilde{x} \leftarrow$ randselect(ds_inducing).
6:    for $\ell = 1$ to L do
7:    //update kernel-space representations.
8:    U[ $\ell$ ][ $\tilde{x}.index$ ] ← f $_{\ell}$ ( $\tilde{x}$ )
9:    end for
10: end for
11: return $[f_{1}(.),...,f_{L}(.)]$ , U, V

$\{\tilde{\boldsymbol{u}}_{m}^{(\ell)}\}_{m=1}^{M}$ . However, it is computationally prohibitive to feed thousands of inducing instances to the kernel mappings as $\tilde{\boldsymbol{u}}_{m}^{(\ell)} = f_{\ell}(\tilde{\boldsymbol{x}}_{\boldsymbol{m}})$ for $m \in \{1, 2, ..., M\}$ in each gradient-descent iteration. On the other hand, as the kernel-space mappings $\{f_{\ell}(.)\}_{\ell=1}^{L}$ keep changing during training, we need to somehow track how the inducing points $\{\tilde{\boldsymbol{u}}_{m}^{(\ell)}\}_{m=1}^{M}$ change during training. To this end, we put the kernel-space representations of the inducing points in matrices denoted by U. During training, these matrices are repeatedly updated by feeding mini-batches of inducing instances to the kernel-mappings.

Alg. 2 optimizes the objective of Eq. 5 w.r.t. the kernel mappings $\{f_{\ell}(.)\}_{\ell=1}^{L}$ . First, a single training instance $x$ and a single inducing point $\tilde{x}$ are selected (line 2-3). Afterwards, the procedure of Alg. 1 is called to update the kernel mappings (line 4 of Alg. 2). To update the kernel-mappings, the GP posterior is computed via the matrices U (line 1 of Alg.1). The "forward\_GP" procedure (called in line 1 of Alg. 1) is provided in Alg. S1 of the supplementary, and uses the matrices U to compute GP's posterior. Only the rows of U that correspond to the selected inducing point $\tilde{x}$ are computed using the kernel-mappings, so that the gradient w.r.t. the kernel-mappings can be computed in the backward pass (lines 5-6 of Alg. S1 in the supplementary). Finally, the matrices U are updated (lines 6-8 of Alg. 2). Due to the lack of space, the routines "forward\_GP" and "Init\_GPparams" and more details are moved to Sec. S2 of the supplementary. Of course instead of a single training/inducing instance, we used a mini-batch of multiple training/inducing instances.

One difficulty of training GPs is the matrix inversion of Eqs.2 and 3, which has $\mathcal{O}(M^3)$ complexity using standard matrix inversion methods. To address this issue, we adopted computational techniques recently used for fast spectral clustering [13]. Let $\mathbf{A}$ be an arbitrary $M\times D$ matrix where $M >> D$ . Moreover, let $\mathbf{b}$ be a $M$ -dimensional vector and let $\sigma$ be a scalar. The computational techniques [13] allow us to efficiently compute: $(\mathbf{A}\mathbf{A}^T +\sigma^2\mathbf{I}_{M\times M})^{-1}\mathbf{b}$ . (Note the similar terms in the right hand side of Eqs. 2 and 3.) The idea is that $\mathbf{A}\mathbf{A}^T$ is of rank $D$ . Therefore, from linear algebra it follows

that $(\mathbf{A}\mathbf{A}^{T} + \sigma^{2}\mathbf{I}_{M\times M})^{-1}$ has M - D eigen-values all of which are equal to $\sigma^{-2}$ . Therefore, in the space of those eigen-vectors, the transformation on b is simply a scaling by $\sigma^{-2}$ . The details and more computational techniques are provided in Sec. S2.1 of the supplementary. These computational techniques allow us to efficiently compute the GP-posterior for hundreds of thousands of inducing points in each gradient descent iteration.

A note on the used datasets in Alg. 2: According to our analysis of Sec. S7 in the supplementary material, "ds\_inducing" should be as large as possible so the GP posteriors can be flexible enough to match the ANNs. Therefore a good practice is to include all training instances (without data augmentation) in "ds\_inducing". By doing so, the following issue arises. An instance from "ds\_train" like x is an augmented version of an inducing instance $\tilde{x}$ . Because x and $\tilde{x}$ are close, their kernel-space representations $f(\boldsymbol{x})$ and $f(\tilde{\boldsymbol{x}})$ also become close regardless of parameters of $f(.).$ Consequently, regardless of $f(.), GP's$ posterior mean will be roughly equal for both x and $\tilde{x}$ . Indeed, in this case Alg. 2 fails to find the kernel mappings $\{f_{\ell}(.)\}_{\ell=1}^{L}$ . To avoid this issue, we sample x in line 2 of Alg. 2 as follows: $x_{1}$ and $x_{2}$ are randomly selected from "ds\_train", and $\alpha \sim uniform(-1,2)$ , and $x = \alpha x_{1} + (1 - \alpha)x_{2}$ . The rest of Alg. 2 after line 2 is run as before.

# 4 Experiments

We conducted several experiments on four publicly available datasets: MNIST [9], Cifar10 [19], Kather [15], and DogsWolves [34]. For MNIST [9] and Cifar10 [19] we used the standard split to training and test sets provided by the datasets. For Kather [15] and DogsWolves [34] we randomly selected 70% and 80% of instances as our training set. The exact parameter settings for running Alg. 2 are elaborated upon in Sec. S5 of the supplementary. We trained the ANNs as usual rather than using Eq. 6, because our proposed GPEX should be applicable to ANNs which are trained as usual.

# 4.1 Measuring Faithfulness of GPs to ANNs

We trained a separate convolutional neural network (CNN) on each dataset to perform the classification task. For MNIST [9], Cifar10 [19], and Kather [15] we used a ResNet-18 [12] backbone followed by some fully connected layers. DogsWolves [34] is a relatively small dataset, and very deep architectures like ResNet [12] overfit to training set. Therefore, we used a convolutional backbone which is suggested in the dataset website [34]. For all datasets, we set the width (i.e. the number of neurons) of the second last fully-connected layer to 1024. Because according to theoretical results on GP-ANN analogy, the second last layer of ANN should be wide. We used an implementation of ResNet [12] which is publicly available online [2]. We trained the pipelines for 20, 200, 20, and 20 epochs on MNIST [9], Cifar10 [19], Kather [15], and DogsWolves [34], respectively. For Cifar10 [19], we used the exact optimizer suggested by [2]. For other datasets we used an Adam [17] optimizer with a learning-rate of 0.0001. The test accuracies of the models are equal to $99.56\%$ , $95.43\%$ , $96.80\%$ , and $80.50\%$ on MNIST [9], Cifar10 [19], Kather [15], and DogsWolves [34], respectively. We also applied our proposed GPEX to a state-of-the-art cell-embedding method called scArches [20]. We ran a tutorial notebook [32] and applied GPEX to the decoder whose job is to predict expression of some genes given scArches [20] cell embeddings. More details are provided in our public github repository (repository link is provided in page 1).

We explained each classifier ANN using our proposed GPEX framework (i.e. Alg.2). As discussed in Sec. 3, given an ANN we have as many kernel-spaces (and as many GPs) as the number of ANN's output heads. The exact parameter settings and practical considerations for training the GPs is elaborated upon in Sec. S5 of the supplementary material. To measure the faithfulness of GPs to ANNs, we compute the Pearson correlation coefficient for each ANN head and the mean of the corresponding GP posterior on unseen test instances. The results are provided in Fig. 3. In Fig. 3, the first five groups of bars (i.e. the groups labeled as Cifar10 (classifier), MNIST (classifier), Kather (classifier), DogsWolves (classifier), and scArches (classifier)) correspond to applying the proposed GPEX to the five classifier ANNs trained on the four datasets and scArches embeddings. According to Fig. 3, our trained GPs almost perfectly match the corresponding ANNs. Only for DogsWolves [34], as illustrated by the 4-th bar group in Fig. 3, the correlation coefficients are lower compared to other datasets. We hypothesize that this is because the DogsWolves dataset [34] has very few images. GP posterior mean can be changed only by moving the inducing points in the kernel-space.

![](images/ef57877681fad5a180b4f83f31f8ae0734c294aebe6374038f039462b9591512.jpg)

<details>
<summary>text_image</summary>

1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1
2 2 2 2 2 2 2 2 2 2 2
2 2 2 2 2 2 2 2 2 2 2
2 2 2 2 2 2 2 2 2 2 2
4 4 4 4 4 4 4 4 4 4 4
4 4 4 4 4 4 4 4 4 4
4 4 4 4 4 4 4 4 4 4
5 5 5 5 5 5 5 5 5 5
5 5 5 5 5 5 5 5 5 5
7 7 7 7 7 7 7 7 7
7 7 7 7 7 7 7 7 7
3 3 3 3 3 3 3
8 y & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & &
</details>

(a)   
![](images/68ce77780705a03cd78107f5132c56b1d332e19e6c078d579a9b7214c7d1d17d.jpg)  
(b)

![](images/c4a5142db2b9966ee540081be7afaf5b3ad952f5370e2653cc246e6dc05c7624.jpg)

<details>
<summary>natural_image</summary>

Grid of 70+ images showing various horse and horseback animals in various poses, colors, and expressions (no text or symbols visible)
</details>

(c)   
![](images/d2983cac6110efd87c253146e15e1be5d7c6ddfab1962f18df0c6c4b2092c374.jpg)

<details>
<summary>natural_image</summary>

Grid of 20 small images showing various medical or scientific views with no visible text, numbers, or symbols.
</details>

![](images/2498a2701ab837ec01a532902de2ce6ce888e039ed4e8ba2a01fd867c1681835.jpg)

<details>
<summary>natural_image</summary>

Grid of 20 small images showing various subjects and medical imaging modalities, no visible text or symbols
</details>

(d)   
Figure 2: (a,c,d) Sample explanations for MNIST, Cifar10, and DogsWolves. In each row a test instance is shown in the first column, and the 10 nearest neighbours (in the kernel-space of the GP that corresponds to the output-head with maximum value at the test instance) is shown in columns 2-11. (b) Evaluating our proposed method, representer point selection [37], and influence functions [18] in dataset debugging task.

Therefore, when very few inducing points are available GP posterior mean is less flexible [36]. This is consistent with our parameter analysis in Sec. S7 of supplementary material.

In Fig. 1a we discussed that GPEX is not only able to explain a classifier ANN, but it can explain any ANN which is a subcomponent of any feed-forward pipeline. To evaluate this ability, we trained three classifiers with an attention mechanism [22]. Each classifier has two ResNet-18 [12] backbones: one extracts a volumetric map containing deep features, and the other produces a spatial attention mask. For each attention backbone, we set the width of the second last layer to 1024, followed by a linear layer and sigmoid activation. We applied our proposed GPEX (i.e. Alg.2) to each classifier, but this time the ANN to be explained (i.e. the box called "ANN" in Fig. 1a) is set to be the attention submodule. Note that each attention backbone produces a spatial attention mask of size $h$ by $w$ . We think of each attention backbone as an ANN which has $h \times w$ output heads. We trained three classifier pipelines with attention mechanism on Cifar10 [19], MNIST [9], and Kather [15] with the same training procedure as previous part. In Fig. 3, 6-th, 7-th, and 8-th bar groups show the correlation coefficients between the attention backbones and the corresponding GPs on unseen test instances. According to Fig. 3, our proposed GPEX is able find GPs which are faithful to attention subcomponents of the classifier pipelines. Note that we didn't include all attention heads, because

![](images/0bfc549f36b5b82f4e8e75fbb4cca0d19672fb9ed766a7667c8fec4f47b5a4d8.jpg)

<details>
<summary>bar</summary>

| Dataset | Feature Set | Correlation coefficient |
| :--- | :--- | :--- |
| Cifar10 (classifier) | Ibad1 | 0.97 |
| Cifar10 (classifier) | Ibad2 | 0.98 |
| Cifar10 (classifier) | Ibad3 | 0.96 |
| Cifar10 (classifier) | Ibad4 | 0.95 |
| Cifar10 (classifier) | Ibad5 | 0.97 |
| Cifar10 (classifier) | Ibad6 | 0.98 |
| Cifar10 (classifier) | Ibad7 | 0.98 |
| Cifar10 (classifier) | Ibad8 | 0.98 |
| Cifar10 (classifier) | Ibad9 | 0.98 |
| Cifar10 (classifier) | Ibad10 | 0.98 |
| MNIST (classifier) | Ibad1 | 0.98 |
| MNIST (classifier) | Ibad2 | 0.98 |
| MNIST (classifier) | Ibad3 | 0.98 |
| MNIST (classifier) | Ibad4 | 0.98 |
| MNIST (classifier) | Ibad5 | 0.98 |
| MNIST (classifier) | Ibad6 | 0.98 |
| MNIST (classifier) | Ibad7 | 0.98 |
| MNIST (classifier) | Ibad8 | 0.98 |
| MNIST (classifier) | Ibad9 | 0.98 |
| MNIST (classifier) | Ibad10 | 0.98 |
| Kather (classifier) | Ibad1 | 0.99 |
| Kather (classifier) | Ibad2 | 0.99 |
| Kather (classifier) | Ibad3 | 0.98 |
| Kather (classifier) | Ibad4 | 0.98 |
| Kather (classifier) | Ibad5 | 0.98 |
| Kather (classifier) | Ibad6 | 0.98 |
| Kather (classifier) | Ibad7 | 0.98 |
| Kather (classifier) | Ibad8 | 0.98 |
| Kather (classifier) | Ibad9 | 0.98 |
| Kather (classifier) | Ibad10 | 0.98 |
| DogsWolves (classifier) (classifier) | Ibad1 | 0.88 |
| DogsWolves (classifier) (classifier) | Ibad2 | 0.83 |
| scArches (classifier) (classifier) | Ibad1 | 0.98 |
| scArches (classifier) (classifier) | Ibad2 | 0.94 |
| scArches (classifier) (classifier) | Ibad3 | 0.97 |
| scArches (classifier) (classifier) | Ibad4 | 0.96 |
| scArches (classifier) (classifier) | Ibad5 | 0.96 |
| scArches (classifier) (classifier) | Ibad6 | 0.96 |
| scArches (classifier) (classifier) | Ibad7 | 0.96 |
| scArches (classifier) (classifier) | Ibad8 | 0.96 |
| scArches (classifier) (classifier) | Ibad9 | 0.96 |
| scArches (classifier) (classifier) | Ibad10 | 0.96 |
| Cifar10 (attention) | Ibad15 | 0.97 |
| Cifar10 (attention) | Ibad17 | 0.97 |
| Cifar10 (attention) | Ibad18 | 0.97 |
| Cifar10 (attention) | Ibad19 | 0.97 |
| Cifar10 (attention) | Ibad20 | 0.97 |
| Cifar10 (attention) | Ibad21 | 0.97 |
| Cifar10 (attention) | Ibad22 | 0.97 |
| Cifar10 (attention) | Ibad23 | 0.97 |
| Cifar10 (attention) | Ibad24 | 0.97 |
| Cifar10 (attention) | Ibad25 | 0.97 |
| Cifar10 (attention) | Ibad26 | 0.97 |
| Cifar10 (attention) | Ibad27 | 0.97 |
| Cifar10 (attention) | Ibad28 | 0.97 |
| Cifar10 (attention) | Ibad29 | 0.97 |
| Cifar10 (attention) | Ibad30 | 0.97 |
| Cifar10 (attention) | Ibad31 | 0.97 |
| Cifar10 (attention) | Ibad32 | 0.97 |
| Cifar10 (attention) | Ibad33 | 0.97 |
| Cifar10 (attention) | Ibad34 | 0.97 |
| Cifar10 (attention) | Ibad35 | 0.97 |
| Cifar10 (attention) | Ibad36 | 0.97 |
| Cifar10 (attention) | Ibad37 | 0.97 |
| Cifar10 (attention) | Ibad38 | 0.97 |
| Cifar10 (attention) | Ibad39 | 0.97 |
| Cifar10 (attention) | Ibad40 | 0.97 |
| Cifar10 (attention) | Ibad41 | 0.97 |
| Cifar10 (attention) | Ibad42 | 0.97 |
| Cifar10 (attention) | Ibad43 | 0.97 |
| Cifar10 (attention) | Ibad44 | 0.97 |
| Cifar10 (attention) | Ibad45 | 0.97 |
| Cifar10 (attention) | Ibad46 | 0.97 |
| Cifar10 (attention) | Ibad47 | 0.97 |
| Cifar10 (attention) | Ibad48 | 0.97 |
| Cifar10 (attention) | Ibad49 | 0.97 |
| Cifar10 (attention) | Ibad50 | 0.97 |
| MNIST (attention) [Ibad15] [Ibad16] [Ibad17] [Ibad18] [Ibad19] [Ibad20] [Ibad21] [Ibad22] [Ibad23] [Ibad24] [Ibad25] [Ibad26] [Ibad27] [Ibad28] [Ibad29] [Ibad30] [Ibad31] [Ibad32] [Ibad33] [Ibad34] [Ibad35] [Ibad36] [Ibad37] [Ibad38] [Ibad39] [Ibad40] [Ibad41] [Ibad42] [Ibad43] [Ibad44] [Ibad45] [Ibad46] [Ibad47] [Ibad48] [Ibad49] [Ibad50] [Ibad51] [Ibad52] [Ibad53] [Ibad54] [Ibad55] [Ibad56] [Ibad57] [Ibad58] [Ibad59] [Ibad60] [Ibad61] [Ibad62] [Ibad63] [Ibad64] [Ibad65] [Ibad66] [Ibad67] [Ibad68] [Ibad69] [Ibad70] [Ibad71] [Ibad72] [Ibad73] [Ibad74] [Ibad75] [Ibad76] [Ibad77] [Ibad78] [Ibad79] [Ibad80] [Ibad81] [Ibad82] [Ibad83] [Ibad84] [Ibad85] [Ibad86] [Ibad87] [Ibad88] [Ibad89] [Ibad90] [Ibad100]
The correlation coefficients are not explicitly labeled as they are estimated based on the provided code format.
</details>

Figure 3: Faithfulness of GPs to ANNs measured by the Pearson correlation coefficient.

some pixels in attention masks are always off. In Sec. S3 of the supplementary material we have included more information and insights about the faithfulness of GPs to ANNs.

# 4.2 Explaining ANNs' Decisions

In Sec. 4.1 we trained four CNN classifiers on Cifar10 [19], MNIST [9], Kather [15], and DogsWolves [34], respectively. Afterwards, we applied our proposed explanation method to each CNN classifier. In this section, we are going to explain the decisions made by the classifiers via the obtained GPs found by Alg. 2. We explain the decision made for a test instance like $\boldsymbol{x}_{test}$ as follows. We consider the GP and the kernel-space that correspond to the ANN's head with maximum value (i.e. the ANN's head that relates to the predicted label). Consequently, among the instances in the inducing dataset, we find the 10 closest instances to $\boldsymbol{x}_{test}$ , like $\{\boldsymbol{x}_{i1}, \boldsymbol{x}_{i2}, ..., \boldsymbol{x}_{i10}\}$ . Intuitively the ANN has labeled $\boldsymbol{x}_{test}$ in that way because it has found $\boldsymbol{x}_{test}$ to be similar to $\{\boldsymbol{x}_{i1}, \boldsymbol{x}_{i2}, ..., \boldsymbol{x}_{i10}\}$ .

For MNIST digit classification, some test instances and nearest neighbours in training set are shown in Fig. 2a. In this figure each row corresponds to a test instance. The first column depicts the test instance itself and columns 2 to 11 depict the 10 nearest neighbours. For example, in Fig. 2a the image in row3-col1 depicts a test instance $\boldsymbol{x}_{test}$ and the images in row3, cols2-11 depict the nearest neighbours $\{\boldsymbol{x}_{i1}, \boldsymbol{x}_{i2}, ..., \boldsymbol{x}_{i10}\}$ . According to rows 1 and 2 of Fig. 2a, the classifier has labeled the two images as digit 1 because it has found 1 digits with similar inclinations in the training set (in Fig. 2a in row 1 all digits are vertical but in row 2 all digits are inclined). We see the model has also taken the inclination into account for the test instances of rows 7, 8, 15, 16, and 17 of Fig. 2a. In Fig. 2a, according to rows 3, 4, and 5 the test instances are classified as digit 2 because 2 digits with similar styles are found in the training set. We see the model has also taken the style into account for the test instances of rows 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, and 17 of Fig. 2a. For instance, the test instance in row 6 of Fig. 2a is a 4 digit with a short stand and the two nearest neighbours are alike. Or for the test instances in rows 13, 14, and 15 of Fig. 2a the test instances have incomplete circles in the same way as their nearest neighbours. More explanations are provided in the supplementary material in Sec. S4.

Fig. 2c illustrates some sample explanations for Cifar10 [19]. Like before, each row corresponds to a test instance, the first column depicts the test instance itself and columns 2 to 11 depict the 10 nearest neighbours. In Fig. 2c, the test instances of rows 1, 2, 3, 4, and 5 are captured from horses' heads from closeby, and the nearest neighbours are alike. However, in rows 6, 7, 8, 9, 10, and 11 of Fig. 2c the test images are taken from faraway and the found similar training images are also taken from faraway. Intuitively, as the classifier is not aware of 3D geometry, it finds training images which are captured from the same distance. In rows 9, 10, and 11 of Fig. 2c, we see that the testing images contain riders. Similarly, the nearest neighbours also tend to have riders. Therefore, in rows 9, 10, and 11 of Fig. 2c the model has made use of the riders or other context information to classify the test instances as horse. More explanations are provided in the supplementary material in Sec. S4.

Besides finding the nearest neighbours, we provide CAM-like [38] explanations as to why $x_{test}$ and an instance like $x_{ij}, 1 \leq j \leq 10$ are considered similar by the model (according to the procedure of Sec. S2.2 in the supplementary material). Fig.2d illustrates some sample explanations for DogsWolves [34] dataset. In row 1 of Fig. 2d, the first column depicts the test instance itself and columns 2 to 11 depict the 10 nearest neighbours. The second and third rows highlight the pixels

that contribute the most to the similarities. The second and third rows highlight the pixels of $x_{test}$ and $\{x_{i1}, x_{i2}, ..., x_{i10}\}$ respectively. According to row 3 of Fig. 2d, the pink object next to the dog's leg has contributed the most to the similarities. According to row 2 of Fig. 2d regions like the baby in column 3, the dog colar or costume in columns 4, 5, and 6, human finger in column 9, and the background in columns 10 and 11 have contributed the most to their similarity to the test instance. These are patterns that usually happen for dogs images. Indeed, since the training set has been small (1600 images), to detect dogs the model is making use of patterns that normally exist in indoor scenes and do not normally appear in wolves images. We see a similar pattern for the test instance in row 6 of Fig. 2d and also several explanations in Sec.S4 of the supplementary.

# 4.3 Comparing GPEX to Representer-Point Selection and Influence Functions

In Sec. S6 of the supplementary material we qualitatively compare GPEX explanations to those of representer point selection [37]. According to the experiments and detailed discussions of Sec. S6 in the supplementary, the GP's kernel that we find in this paper is superior due to a technical point in the formulation of representer point selection [37]. Besides the analysis of Sec. S6, we compared our proposed GPEX with representer point selection [37] and influence functions [18] in dataset debugging task. In these experiments we only selected images from Cifar10 [19] that are labeled as either automobile or horse. To corrupt the labels, we randomly selected $45\%$ of training instances and changed their labels. Afterwards, we trained a classifier CNN with ResNet18 [12] backbone with the same training procedure explained in Sec. 4.1. In dataset debugging task, training instances are shown to a user in some order. After seeing an instance, the user checks the label of the instance and corrects it if needed. One can use explanation methods to bring the corrupted labels to the user's attention more quickly. Given an explanation method, we repeatedly select a test instance which is misclassified by the model. Afterwards, we show to the user the closest training instance (of course among the training instances which are not yet shown to the user). We repeat this process for test instances in turn until all training instances are shown to the user. We compared our proposed GPEX to representer point selection [37] and influence functions [18] in dataset debugging task. We used an implementation of influence functions [18] based on LiSSA [4] with 10 steps for each instance. The implementation is publicly available [1]. For representer point selection [37] we used the implementation by authors which is publicly available [3]. The result is shown in Fig. 2b. According to the plot on the left in Fig. 2b, when correcting the dataset by GPEX, the model accuracy becomes close to $90\%$ after showing about 4000 instances to user. But when using representer point selection [37] or influence functions [18], this happens when the user has seen about 7000 training instances. With noisy labels model training becomes unstable. Therefore, in the plot on the left of Fig. 2b we repeat the training 5 times and we report the standard errors by the lines in top of the bars. According to the plot on the right of Fig. 2b, after showing a fixed number of training instances to the user, when using the proposed GPEX more corrupted labels are shown to the user. Indeed, GPEX brings the corrupted labels to the user's attention quicker than representer point selection [37] does. Interestingly, according to the plot in the right hand side of Fig. 2b influence functions [18] is quicker at spotting incorrect labels, but the instances found by our proposed method are more effective in increasing the accuracy quicker.

# 5 Related Work

The first theoretical connection between ANNs and GPs was that under some conditions, a random single-layer neural network converges to the mean of a Gaussian process $[23]$ as the width of that single layer goes to infinity. This connection was later proven for ANNs with many layers $[8]$ , and for ANNs trained with gradient descent $[14]$ . The theoretical requirements are usually too restrictive. For example, $[8]$ requires all intermediate layers to be wide and also requires the dataset to be countable (so data-augmentations like color-jitter are not allowed). Or $[24]$ requires the ANN to be trained with MSE loss and requires all intermediate layers to be wide. In this paper we do not presume any conditions on the ANN and simply distill knowledge from a neural network to some GPs. Of note, those theoretical conditions may facilitate knowledge distillation and improve the Pearson correlation coefficient between the ANNs and the GPs obtained by our method.

Scalability is a major issue when training GPs, and including a few inducing points may limit the flexibility of GP's posterior [36]. Here we review some previous methods to tackle the computational challenges of training GPs. SV-DKL [27] derives a lower-bound for training a GP with a deep

kernel. In this method, a grid of inducing points are considered in the kernel-space (like the vectors $\{(\tilde{\boldsymbol{u}}_m^{(\ell)},\tilde{v}_m^{(\ell)})\}_{m = 1}^M$ with the notation of this paper). Afterwards, each input instance is firstly mapped to the kernel-space and the output is computed based on similarities to the grid points in the kernel-space. Since the GP posterior is computed via the grid points, SV-DKL [27] is scalable. But unfortunately the number of grid points cannot be increased to above 1000 even for Cifar10 [19] and with a RTX 3090 GPU. Therefore, this may limit the flexibility of the GP's posterior [36].

A more recent framework called GPytorch $[29]$ provides GPU acceleration. However, its computational complexity is quadratic in number of inducing points. Other approaches to improve scalability of GPs include: considering structured kernel matrices $[7]$ , kernel interpolation $[35]$ , and imposing grid-structure on including points $[27]$ . Stack of Gaussian processes are shown to be connected to ANNs $[10][30][24]$ . By stacking kernels, GP kernels work on intermediate representations and therefore are not necessarily interpretable to humans. But in our method the GPs' kernels work directly on the input-space itself. Knowledge distillation (KD) is closely related to this work. With the best of our knowledge and according to the authors, $[6]$ is the first work that applies KD to GPs. But the distinction of our work is that we distill knowledge from ANN to GP, as opposed to the self-distillation of $[6]$ that distills knowledge from a GP to another GP.

Limitations and Outlook: In this work we used Eq. 5 to distill knowledge from ANN to GP. One may use Eq. 6 to distill knowledge from GP to ANN in order to, e.g., transfer GP's good generalization to the ANN. Our method scales very well, and Alg. 2 runs without memory/computational issues even on imagenet with more than 1M inducing points (i.e. images) and Resnet-18 [12] when a few output-heads are selected, but on imagenet we failed to match the GPs to ANN in a 2-3 day runtime. The issue is that the U matrices have to be updated very often (the update of line 8 of Alg. 2) so that GPs' kernels are updated according to an accurate estimate of kernel-space representations. Otherwise the convergence may not happen especially for millions of inducing points and a small batch-size (as required for, e.g., CNNs). We used control-variate [25], but one may use more advanced heuristics [35] to achieve convergence for datasets like imagenet and with a reasonable computation time. In this paper we analyzed the effect of number of inducing points, the width of the second last layer, and the number of epochs for which the ANN is trained. One can use the proposed tool to answer other questions, like, is the GP kernel required to have more parameters than the ANN itself? May it so happen that a test instance is equally close to hundreds of training instances thereby limiting a human's ability to understand ANNs decision? Is the uncertainty provided by the GP correlated to the understandability of the explanations to humans or to the ANN's failures?

# 6 Acknowledgements

The experiments of this paper were enabled in part by the Digital Research Alliance of Canada. This work was supported in part by the NSERC Discovery Grant.

# References

[1] A simple PyTorch implementation of influence functions. https://github.com/alstonlo/torch-influence. [Online; accessed 19-Oct-2023].   
[2] An implementation for ResNet in pytorch. https://github.com/kuangliu/pytorch-cifar. [Online; accessed 19-Dec-2021].   
[3] Implementaiton of representer point selection. https://github.com/chihkuanyeh/Representer\_Point\_Selection. [Online; accessed 19-Oct-2023].   
[4] N. Agarwal, B. Bullins, and E. Hazan. Second-order stochastic optimization for machine learning in linear time. The Journal of Machine Learning Research, 18(1):4148–4187, 2017.   
[5] Avanti Shrikumar et al. Learning important features through propagating activation differences. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 3145–3153. PMLR, 06–11 Aug 2017.   
[6] K. Borup and L. N. Andersen. Self-distillation for gaussian process regression and classification. arXiv preprint arXiv:2304.02641, 2023.

[7] M. K. Cohen, S. Daulton, and M. A. Osborne. Log-linear-time gaussian processes using binary tree kernels. In A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho, editors, Advances in Neural Information Processing Systems, 2022.   
[8] A. G. de G. Matthews, J. Hron, M. Rowland, R. E. Turner, and Z. Ghahramani. Gaussian process behaviour in wide deep neural networks. In International Conference on Learning Representations, 2018.   
[9] L. Deng. The mnist database of handwritten digit images for machine learning research. IEEE Signal Processing Magazine, 29(6):141–142, 2012.   
[10] Y. Gal and Z. Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning, pages 1050–1059. PMLR, 2016.   
[11] A. Ghorbani, A. Abid, and J. Zou. Interpretation of neural networks is fragile. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01):3681–3688, Jul. 2019.   
[12] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016.   
[13] L. He, N. Ray, Y. Guan, and H. Zhang. Fast large-scale spectral clustering via explicit feature mapping. IEEE Transactions on Cybernetics, 49(3):1058–1071, 2019.   
[14] Jaehoon Lee et al. Wide neural networks of any depth evolve as linear models under gradient descent. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.   
[15] J. N. Kather, C.-A. Weis, F. Bianconi, S. M. Melchers, L. R. Schad, T. Gaiser, A. Marx, and F. G. Zöllner. Multi-class texture analysis in colorectal cancer histology. Scientific Reports, 6, 2016.   
[16] A. Khakzar, P. Khorsandi, R. Nobahari, and N. Navab. Do explanations explain? model knows best. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10244–10253, June 2022.   
[17] D. P. Kingma and J. Ba. Adam: A method for stochastic optimization, 2017.   
[18] P. W. Koh and P. Liang. Understanding black-box predictions via influence functions. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1885–1894. PMLR, 06–11 Aug 2017.   
[19] A. Krizhevsky. Learning multiple layers of features from tiny images. pages 32-33, 2009.   
[20] M. Lotfollahi, M. Naghipourfar, M. D. Luecken, M. Khajavi, M. Büttner, M. Wagenstetter, Ž. Avsec, A. Gayoso, N. Yosef, M. Interlandi, et al. Mapping single-cell data to reference atlases by transfer learning. Nature Biotechnology, pages 1–10, 2021.   
[21] S. M. Lundberg and S.-I. Lee. A unified approach to interpreting model predictions. In Advances in Neural Information Processing Systems 30, pages 4765-4774. Curran Associates, Inc., 2017.   
[22] Meng-Hao Guo et al. Attention mechanisms in computer vision: A survey. CoRR, abs/2111.07624, 2021.   
[23] R. Neal. Bayesian Learning for Neural Networks. Lecture Notes in Statistics. Springer New York, 2012.   
[24] R. Novak, L. Xiao, J. Hron, J. Lee, A. Alemi, J. Sohl-dickstein, and S. Schoenholz. Neural tangents: Fast and easy infinite neural networks in python. 2020.   
[25] J. Paisley, D. Blei, and M. Jordan. Variational bayesian inference with stochastic search. arXiv preprint arXiv:1206.6430, 2012.   
[26] C. E. Rasmussen and C. K. I. Williams. Gaussian Processes for Machine Learning (Adaptive Computation and Machine Learning). The MIT Press, 2005.   
[27] A. Wilson et al. Stochastic variational deep kernel learning. In Advances in Neural Information Processing Systems, volume 29. Curran Associates, Inc., 2016.   
[28] D. Slack et al. Fooling lime and shap: Adversarial attacks on post hoc explanation methods. In Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society, AIES '20, page 180–186, New York, NY, USA, 2020. Association for Computing Machinery.   
[29] J. Gardner et al. Gpytorch: Blackbox matrix-matrix gaussian process inference with gpu acceleration. Advances in Neural Information Processing Systems, 2018-December:7576–7586, 2018.

[30] V. Dutordoir et al. Deep neural networks as point estimates for deep gaussian processes. Advances in Neural Information Processing Systems, 34:9443-9455, 2021.   
[31] M. T. Ribeiro, S. Singh, and C. Guestrin. "why should I trust you?": Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, San Francisco, CA, USA, August 13-17, 2016, pages 1135–1144, 2016.   
[32] scArches CVAE notebook. https://docs.scarches.org/en/latest/expimap\_surgery\_pipeline\_basic.html.   
[33] M. Sundararajan, A. Taly, and Q. Yan. Axiomatic attribution for deep networks. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 3319–3328. PMLR, 06–11 Aug 2017.   
[34] H. Vutukuri. Dogs vs Wolves Classification of Dogs and Wolves. https://www.kaggle.com/harishvutukuri/dogs-vs-wolves, 2019. [Online; accessed 19-Dec-2021].   
[35] A. Wilson and H. Nickisch. Kernel interpolation for scalable structured gaussian processes (kiss-gp). In International conference on machine learning, pages 1775–1784. PMLR, 2015.   
[36] Y. Bengio's post on gp vs ann https://qr.ae/pvqZn7.   
[37] C.-K. Yeh, J. Kim, I. E.-H. Yen, and P. K. Ravikumar. Representer point selection for explaining deep neural networks. Advances in neural information processing systems, 31, 2018.   
[38] B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, and A. Torralba. Learning deep features for discriminative localization. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016.

# Supplementary Material for GPEX, A Framework For Interpreting Artificial Neural Networks

Amir Akbarnejad, Gilbert Bigras, Nilanjan Ray

![](images/033c012f0a65c1b417cd9e33932d854f4c0bac079d638d4ae5ebfbfaa2b76c51.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Input Layer
        Xn["x_n"] --> Xn2["x_n"]
        Xn2 --> Un2["u_n"]
        Un2 --> Vn["v_n"]
        Vn --> Yn["Y_n"]
        Yn --> ∞[∞]
    end

    subgraph Hidden Layer
        Xm["x̃_m"] --> Um["ū_m"]
        Um --> Vm["Ṽ_m"]
        Vm --> GPd["GP_d"]
        GPd --> Dv["D_v"]
    end

    Xn2 -.->|g| Vn
    Un2 -.->|f| Vn
    Vn -.-> Yn
    Xm -.-> Um
    Um -.-> Vm
    GPd -.-> Vm
    Dv -.-> Vm
    style Input Layer fill:#f9f,stroke:#333
    style Hidden Layer fill:#bbf,stroke:#333
    style Input Layer fill:#dfd,stroke:#333
    style Hidden Layer fill:#dfd,stroke:#333
    style Input Layer fill:#dfd,stroke:#333
    style Hidden Layer fill:#dfd,stroke:#333
    style Input Layer fill:#dfd,stroke:#333
    style Hidden Layer fill:#dfd,stroke:#333
```
</details>

Fig. S1: The proposed framework as a probabilistic graphical model.

# S1 DERIVING THE VARIATIONAL LOWER-BOUND

In this section we derive the variational lower-bound introduced in Sec.2.3 of the main article. We firstly introduce Lemmas 1 and 2 as they appear in our derivations.

Lemma 1. The KL-divergence between two normal distributions $\mathcal{N}_1(. ; \boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1)$ and $\mathcal{N}_2(. ; \boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)$ can be computed as follows:

$$
\begin{array}{l} K L \left(\mathcal {N} _ {1} | | \mathcal {N} _ {2}\right) = \frac {1}{2} \left(\log \left(\frac {| \boldsymbol {\Sigma} _ {2} |}{| \boldsymbol {\Sigma} _ {1} |}\right) - D + t r a c e \left\{\boldsymbol {\Sigma} _ {2} ^ {- 1} \boldsymbol {\Sigma} _ {1} \right\} \right. \\ \left. + \left(\boldsymbol {\mu} _ {2} - \boldsymbol {\mu} _ {1}\right) ^ {T} \boldsymbol {\Sigma} _ {2} ^ {- 1} \left(\boldsymbol {\mu} _ {2} - \boldsymbol {\mu} _ {1}\right)\right). \tag {S1} \\ \end{array}
$$

Lemma 2. Let $p_1$ and $p_2$ be two normal distributions:

$$
p _ {1} (x) = \mathcal {N} \bigl (x; \mu_ {1}, \sigma_ {1} ^ {2} \bigr),
$$

$$
p _ {2} (x) = \mathcal {N} \bigl (x; \mu_ {2}, \sigma_ {2} ^ {2} \bigr).
$$

We have that

$$
\begin{array}{l} \mathbb {E} _ {x \sim p _ {2}} \left[ \log p _ {1} (x; \mu_ {1}, \sigma_ {1} ^ {2}) \right] = \\ - \frac {\left(\mu_ {1} - \mu_ {2}\right) ^ {2} + \sigma_ {2} ^ {2}}{2 \sigma_ {1} ^ {2}} - \frac {1}{2} \log \left(\sigma_ {1} ^ {2}\right) - \frac {1}{2} \log (2 \pi). \tag {S2} \\ \end{array}
$$

Fig.S1 illustrates the framework as a probabilistic graphical model. A general feed-forward pipeline takes in a set of input(s) X and produces a set of output(s) Y. The general pipeline is required to have at least one ANN as a submodule. The ANN submodule is required to take in only one input x and to produce only one output v, where x and v are tensors of arbitrary sizes. As illustrated in Fig.S1, the ANN's input x can depend arbitrarily on some other intermediate variables in the pipeline. This relation is modeled by the conditional distribution $p(\mathbf{x}_{n}|Parent(\mathbf{x}_{n}))$ where $Parent(\mathbf{x}_{n})$ is the set of all variables which are connected to $x_{n}$ . Similarly, as illustrated in Fig.S1 the pipeline's output Y can arbitrarily depend on some intermediate variables in the pipeline. This relation is modeled by the conditional distribution $p(\mathcal{Y}_{n}|Parent(\mathcal{Y}_{n}))$ . In Fig.S1 the lower boxes are the inducing points and other variables that determine the GPs' posterior. More precisely, in Fig.S1 $\{\tilde{x}_{m}\}_{m=1}^{M}$ are some inducing points (e.g. some training images). Vectors in the kernel space are denoted by $\tilde{u}$ and u. Moreover, the observed values are denoted by v and $\tilde{v}$ . Informally, u and v denote the input/output of the GPs. When referring to one of the M inducing points a "tilde" is used (as $(\tilde{u},\tilde{v})$ ), however $(u,v)$ corresponds to a point that can be anywhere in the kernel-space.

The inducing instances $\{\tilde{x}_m\}_{m=1}^M$ are mapped to the kernel-spaces by the kernel mappings $\{f_1(.),...,f_L(.)\}$ . In Fig.S1 the variables $\{\tilde{u}_m\}_{m=1}^M$ are the kernel-space representations of the inducing points $\{\tilde{x}_m\}_{m=1}^M$ . Moreover, $\{\tilde{v}_m\}_{m=1}^M$ are the GP's output values at the inducing points. Given an instance $x_n$ , it is firstly fed to the kernel mappings $\{f_1(.),...,f_L(.)\}$ and the kernel-space representations $u_n$ are obtained. Afterwards, the GPs' outputs on $u_n$ depend on $u_n$ as well as all other inducing points because the inducing points actually determine the GPs' posterior on all kernel-space points including $u_n$ . Therefore, in Fig.S1 the variable $v_n$ is not only connected to $u_n$ but it is also connected to the box at the bottom (i.e. all inducing points and other variables associated with them).

As usual, the variational lower-bound is equal to

$$
\mathcal {L} = \mathbb {E} _ {\sim q} [ \log p (\text { all   variables }) ] - \mathbb {E} _ {\sim q} [ \log q (\text { hidden   variables }) ]. \tag {S3}
$$

The likelihood of all variables in Eq.S3 factorizes as the product of conditional distributions of each variable given

its parents. Therefore

$$
p (\text { all   variables }) = \prod_ {\text { variable } t} p (t | \text { Parent } (t)). \tag {S4}
$$

In Eq.S4 only some conditional distributions appear in our derivations which are discussed at the following.

- The variable $x_{n}$ : the ANN's input $x_{n}$ can depend arbitrarily on some other intermediate variables in the pipeline. In our derivations we leave this conditional distribution as $p(\boldsymbol{x}_{n}|Parent(\boldsymbol{x}_{n}))$ .   
- The variable $u_{n}$ : Given a training instance $x_{n}$ , the kernel-space representations $u_{n}$ are deterministically obtained by feeding the instance to the kernel-mappings $[f_{1}(.),...,f_{L}(.)]$ .   
- The variable $\pmb{v}_n$ : The ANN's output is required to depend only on the input, so

$$
p \left(\boldsymbol {v} _ {n} \mid \text { Parent } (\boldsymbol {v} _ {n})\right) = p \left(\boldsymbol {v} _ {n} \mid \boldsymbol {u} _ {n}, \boldsymbol {x} _ {n}, \{\tilde {\boldsymbol {x}} _ {m}, \tilde {\boldsymbol {u}} _ {m}, \tilde {v} _ {m} \} _ {m = 1} ^ {M}\right). \tag {S5}
$$

The above distribution is actually the GPs' posterior at $u_{n}$ (i.e. the normal distribution of Eq.1 of the main article).

- The variable $\tilde{\boldsymbol{x}}_m$ : the inducing point $\tilde{\boldsymbol{x}}_m$ can depend arbitrarily on some other intermediate variables in the pipeline. In our derivations we leave this conditional distribution as $p(\tilde{\boldsymbol{x}}_m|Parent(\tilde{\boldsymbol{x}}_m))$ .   
- The variable $\tilde{\boldsymbol{u}}_m$ : Given an inducing point $\tilde{\boldsymbol{x}}_m$ , the kernel-space representations $\tilde{\boldsymbol{u}}_m$ are deterministically obtained by feeding the inducing point $\tilde{\boldsymbol{x}}_m$ to the kernel-mappings $[f_1(.), ..., f_L(.)]$ .   
- The variables $\hat{\boldsymbol{v}}_m$ : Given the kernel-space representations $\{\tilde{\boldsymbol{u}}_m^{(\ell)}\}_{m=1}^M$ , the variables $\{\tilde{v}_1^{(\ell)}, ..., \tilde{v}_M^{(\ell)}\}$ follow a $M$ -dimensional Gaussian distribution with zero mean and a covariance matrix determined by the GP prior covariance among the variables $\{\tilde{\boldsymbol{u}}_m^{(\ell)}\}_{m=1}^M$ .   
- The variable $\mathcal{Y}_n$ : the pipeline's output $\mathcal{Y}$ can arbitrarily depend on some intermediate variables in the pipeline. In our derivations we leave this conditional distribution as $p(\mathcal{Y}_n|Parent(\mathcal{Y}_n))$ .

According to Eq.S4, the likelihood of all variables factorizes as

$$
\begin{array}{l} p (\text { all   variables }) = \prod_ {\text { variable } t} p (t | \text { Parent } (t)) \\ = \left(\prod_ {n} p (\boldsymbol {x} _ {n} | P a r e n t (\boldsymbol {x} _ {n}))\right) \times \left(\prod_ {n} p (\boldsymbol {u} _ {n} | \boldsymbol {x} _ {n})\right) \times \\ \big (\prod_ {n} \prod_ {\ell} p (v _ {n} ^ {(\ell)} | \boldsymbol {u} _ {n}, \boldsymbol {x} _ {n}, \{\tilde {\boldsymbol {x}} _ {m}, \tilde {\boldsymbol {u}} _ {m}, \tilde {v} _ {m} \} _ {m = 1} ^ {M}) \big) \times \\ \big (\prod_ {m} p \left(\tilde {\boldsymbol {x}} _ {m} \mid \text {Parent} \left(\tilde {\boldsymbol {x}} _ {m}\right)\right) \times \big (\prod_ {m} p \left(\tilde {\boldsymbol {u}} _ {m} \mid \tilde {\boldsymbol {x}} _ {m}\right) \big) \times \\ \big (\prod_ {\ell} p (\tilde {\boldsymbol {v}} _ {1: M} ^ {(\ell)} | \mathbf {0}, \mathcal {K} _ {p r i o r} (\tilde {\boldsymbol {u}} _ {1: M} ^ {(\ell)}, \tilde {\boldsymbol {u}} _ {1: M} ^ {(\ell)})) \big) \times \\ \left(\prod_ {n} p \left(\mathcal {Y} _ {n} \mid \text { Parent } \left(\mathcal {Y} _ {n}\right)\right)\right) \times \left(\prod_ {\text { other   vars } t} p (t \mid \text { Parent } (t))\right). \tag {S6} \\ \end{array}
$$

Now we derive the lower-bound $\mathcal{L}$ with respect to each parameter separately.

# S1.1 Deriving the Lower-bound With Respect to the Kernel-mappings

In the right-hand-side of Eq.S6 only the following terms are dependant on the kernel-mappings $[f_{1}(.),...,f_{L}(.)]$ :

$$
\begin{array}{l} \big [ \prod_ {m} p (\tilde {\boldsymbol {u}} _ {m} | \tilde {\boldsymbol {x}} _ {m}) \times \prod_ {\ell} p (\tilde {v} _ {m} ^ {(\ell)} | \mathbf {0}, \mathcal {K} _ {p r i o r} (\tilde {\boldsymbol {u}} _ {1: M} ^ {(\ell)}, \tilde {\boldsymbol {u}} _ {1: M} ^ {(\ell)})) \big ] \times \\ \left[ \prod_ {n} p (\boldsymbol {u} _ {n} | \boldsymbol {x} _ {n}) \times \prod_ {\ell} p (v _ {n} ^ {(\ell)} | \boldsymbol {u} _ {n}, \boldsymbol {x} _ {n}, \{\tilde {\boldsymbol {x}} _ {m}, \tilde {\boldsymbol {u}} _ {m}, \tilde {v} _ {m} \} _ {m = 1} ^ {M}) \right]. \tag {S7} \\ \end{array}
$$

Note that in the above equation the terms $p(\tilde{\boldsymbol{u}}_m|\tilde{\boldsymbol{x}}_m)$ and $p(\boldsymbol{u}_n|\boldsymbol{x}_n)$ are equal to 1 because $\tilde{\boldsymbol{u}}_m$ and $\boldsymbol{u}_n$ are deterministically obtained from $\tilde{\boldsymbol{x}}_m$ and $\boldsymbol{x}_n$ . Therefore, in Eq.S3 the terms containing the kernel mappings $[f_1(.),\dots,f_L(.)]$ are as follows:

$$
\begin{array}{l} \mathcal {L} _ {f} = \mathbb {E} _ {\sim q} \big [ \sum_ {\ell} \log p (v ^ {(\ell)} | \boldsymbol {u}, \boldsymbol {x}, \{\tilde {\boldsymbol {x}} _ {m}, \tilde {\boldsymbol {u}} _ {m}, \tilde {v} _ {m} \} _ {m = 1} ^ {M}) \big ] + \\ \sum_ {\ell} \mathbb {E} _ {\sim q} \big [ \log p (\tilde {\pmb {v}} _ {1: M} ^ {(\ell)} | \pmb {0}, \mathcal {K} _ {p r i o r} (\tilde {\pmb {u}} _ {1: M} ^ {(\ell)}, \tilde {\pmb {u}} _ {1: M} ^ {(\ell)})) \big ] - \\ \sum_ {\ell} \mathbb {E} _ {\sim q} \big [ \log q _ {2} (\tilde {\pmb {v}} _ {1: M} ^ {(\ell)}) \big ] \\ = \mathbb {E} _ {\sim q} \big [ \sum_ {\ell} \log p (v ^ {(\ell)} | \pmb {u}, \pmb {x}, \{\tilde {\pmb {x}} _ {m}, \tilde {\pmb {u}} _ {m}, \tilde {v} _ {m} \} _ {m = 1} ^ {M}) \big ] - \\ \sum_ {\ell = 1} \mathbb {E} _ {\sim q} \left[ K L \left(q _ {2} \left(\tilde {\boldsymbol {v}} _ {1: M} ^ {(\ell)}\right) \mid \mid p \left(\tilde {\boldsymbol {v}} _ {1: M} ^ {(\ell)} \mid \mathbf {0}, \mathcal {K} _ {\text { prior }} \left(\tilde {\boldsymbol {u}} _ {1: M} ^ {(\ell)}, \tilde {\boldsymbol {u}} _ {1: M} ^ {(\ell)}\right)\right)\right) \right]. \tag {S8} \\ \end{array}
$$

We simplify the two terms on the right-hand-side of Eq.S8. The first term is the expected log-likelihood of a Gaussian distribution (i.e. the conditional log-likelihood of $\tilde{v}^{\ell}$ as in Eq.1 of the main article). Also the variational distribution $q(.)$ is Gaussian. Therefore, we can use Lemma.2 to simplify the first term:

$$
\begin{array}{l} \mathbb {E} _ {\sim q} \big [ \sum_ {\ell = 1} ^ {L} \log p (v ^ {(\ell)} | \boldsymbol {u}, \boldsymbol {x}, \{\tilde {\boldsymbol {x}} _ {m}, \tilde {\boldsymbol {u}} _ {m}, \tilde {v} _ {m} \} _ {m = 1} ^ {M}) \big ] = \\ \sum_ {\ell = 1} ^ {L} \mathbb {E} _ {\sim q} \big [ \log p (v ^ {(\ell)} | \boldsymbol {u}, \boldsymbol {x}, \{\tilde {\boldsymbol {x}} _ {m}, \tilde {\boldsymbol {u}} _ {m}, \tilde {v} _ {m} \} _ {m = 1} ^ {M}) \big ] = \\ \sum_ {\ell = 1} ^ {L} \left[ - \frac {\left(\mu_ {v} (\boldsymbol {u} ^ {(\ell)} , \tilde {\mathbf {u}} _ {1 : M} ^ {(\ell)} , \tilde {v} _ {1 : M} ^ {(\ell)}) - g _ {\ell} (\boldsymbol {x})\right) ^ {2} + \sigma_ {g} ^ {2}}{c o v _ {v} (\boldsymbol {u} ^ {(\ell)} , \tilde {\mathbf {u}} _ {1 : M} ^ {(\ell)} , \tilde {v} _ {1 : M} ^ {(\ell)})} \right. \tag {S9} \\ - \frac {1}{2} \log \left(c o v _ {v} (\pmb {\mathscr {u}} ^ {(\ell)}, \tilde {\pmb {\mathscr {u}}} _ {1: M} ^ {(\ell)}, \tilde {v} _ {1: M} ^ {(\ell)})\right) \\ \left. - \frac {1}{2} \log (2 \pi) \right]. \\ \end{array}
$$

Note that the two terms of Eq.S9 are the two terms which were presented and discussed in Eq.5 of the main article.

Now we simplify the KL-term on the right-hand-side of

Eq.S8. According to Lemma.1 we have that

$$
\begin{array}{l} K L \left(q _ {2} \left(\tilde {\boldsymbol {v}} _ {1: M} ^ {(\ell)}\right) \mid \mid p \left(\tilde {\boldsymbol {v}} _ {1: M} ^ {(\ell)} \mid \mathbf {0}, \mathcal {K} _ {\text { prior }} \left(\tilde {\boldsymbol {u}} _ {1: M} ^ {(\ell)}, \tilde {\boldsymbol {u}} _ {1: M} ^ {(\ell)}\right)\right)\right) = \\ + 0. 5 \big (\log (\frac {\sigma_ {g p} ^ {2}}{\sigma_ {\varphi} ^ {2}}) \big) \\ - 0. 5 M \\ + \frac {\sigma_ {\varphi} ^ {2}}{\sigma_ {g p} ^ {2}} \\ + \frac {\boldsymbol {\varphi} _ {1 : M} ^ {(\ell) ^ {T}} \boldsymbol {\varphi} _ {1 : M} ^ {(\ell)}}{\sigma_ {g p} ^ {2}}, \\ \end{array}
$$

(S10)

where $\varphi$ are the variational parameters of $q_{2}(.)$ as in Eq.4 of the main article. Therefore, the KL-term of Eq.S8 is a constant with respect to the kernel mappings $[f_{1}(.),...,f_{L}(.)]$ and can be discarded. All in all, the lower-bound for optimizing the kernel-mappings is equal to the right-hand-side of Eq.S9 which was introduced and discussed in Sec.2.3. of the main article.

# S1.2 Deriving the Lower-bound With Respect to the ANN Parameters

According to Eq.4 of the main article, in our formulation the ANN's parameters appear as some variational parameters. Therefore, the likelihood of all variables (Eq.S6) does not generally depend on the ANN's parameters. But according to the general ELBO formulation in Eq.S3 the ELBO $\mathcal{L}$ depends on ANN's parameters, because when computing the expectation the variables are drawn from the variational distribution $q(.)$ . We estimated the ELBO of Eq.S3 by the average over few samples. More precisely, given a training instance $x$ , we firstly computed the kernel-space representations as:

$$
\boldsymbol {u} ^ {(\ell)} = f _ {\ell} (\boldsymbol {x}), 1 \leq \ell \leq L. \tag {S11}
$$

Afterwards, we used the reparametrization trick for Eq.1 of the main article to draw a sample for $\boldsymbol{v}^{(\ell)}$ as follows:

$$
\begin{array}{l} z _ {q 2} ^ {(\ell)} \sim \mathcal {N} (0, 1), \\ v ^ {(\ell)} \sim \mu_ {v} (\boldsymbol {u} ^ {(\ell)}, \tilde {\mathbf {u}} _ {1: M} ^ {(\ell)}, \tilde {v} _ {1: M} ^ {(\ell)}) + z _ {q 2} ^ {(\ell)} c o v _ {v} (\boldsymbol {u} ^ {(\ell)}, \tilde {\mathbf {u}} _ {1: M} ^ {(\ell)}, \tilde {v} _ {1: M} ^ {(\ell)}), \tag {S12} \\ \end{array}
$$

where $\mu_{v}(.,.,.)$ and $cov_{v}(.,.,.)$ are defined in Eqs.2 and 3 of the main article. Moreover, we continue the forward pass of the original pipeline to get a sample Y. Having drawn x, u, v, and Y from the variational distribution, we estimate the ELBO of Eq.S3 by these samples.

$$
\begin{array}{l} \mathcal {L} = \\ \mathbb {E} _ {\sim q} \big [ \log p (\text { all   variables }) \big ] - \mathbb {E} _ {\sim q} \big [ \log q (\text { hidden   variables }) \big ] \\ \approx \log p (\text { all   variables }) \Big | _ {\boldsymbol {x}, \boldsymbol {u}, v, \mathcal {Y}} - \sum_ {m} ^ {M} \sum_ {\ell} ^ {L} \mathbb {E} _ {\sim q _ {2}} \left[ \log q _ {2} \left(\tilde {v} _ {m} ^ {(\ell)}\right) \right] \tag {S13} \\ \end{array}
$$

In the above equation, the second term on the right-hand-side is the entropy of a normal distribution and it only depends on the variance of the $q_{2}$ distribution. As we let the variance of $q_{2}$ be fixed ( $\sigma_{g}^{2}$ in Eq.4 of the main article), the second term is a constant. Therefore,

$$
\mathcal {L} \approx \log p (\text { all   variables }) \bigg | _ {\boldsymbol {x}, \boldsymbol {u}, v, \mathcal {Y}}. \tag {S14}
$$

Among the likelihood term on the right-hand-side of Eq.S6 the conditional distribution of all variables before $u_{n}$ (e.g. $x_{n}$ and $X_{n}$ ) are independent of the ANN's parameters (i.e. the parameters of the function $g(\cdot)$ ). On the other hand, for all variables that appear after $u_{n}$ , the conditional distribution depends on the ANN's parameters. Indeed, according to Eq.S14

$$
\begin{array}{l} \mathcal {L} _ {a n n} \approx \left[ \sum_ {\ell = 1} ^ {L} \log p (v ^ {(\ell)} | \boldsymbol {u}, \boldsymbol {x}, \{\tilde {\boldsymbol {x}} _ {m}, \tilde {\boldsymbol {u}} _ {m}, \tilde {v} _ {m} \} _ {m = 1} ^ {M}) \right] \Big | _ {\boldsymbol {x}, \boldsymbol {v}} + \\ \left. \log p (\mathcal {Y} | P a r e n t (\mathcal {Y})) \right| _ {\boldsymbol {x}, \boldsymbol {v}, \mathcal {Y}} + \\ \left(\sum_ {\text {other vars after} \boldsymbol {u} _ {n}} \log p (t | \text {Parent} (t))) \right| _ {\boldsymbol {x}, \boldsymbol {v}, \mathcal {Y}}. \tag {S15} \\ \end{array}
$$

In the above equation, the first term on the right-hand-side is the log-likelihood of the normal distribution of Eq.1:

$$
\begin{array}{l} \log p (v ^ {(\ell)} | \boldsymbol {u}, \boldsymbol {x}, \{\tilde {\boldsymbol {x}} _ {m}, \tilde {\boldsymbol {u}} _ {m}, \tilde {v} _ {m} \} _ {m = 1} ^ {M}) = \\ - \frac {1}{2} \left[ \sum_ {\ell = 1} ^ {L} \frac {\left(\mu_ {v} \left(\boldsymbol {u} ^ {(\ell)} , \tilde {\mathbf {u}} _ {1 : M} ^ {(\ell)} , \tilde {v} _ {1 : M} ^ {(\ell)}\right) - g _ {\ell} (\boldsymbol {x})\right) ^ {2}}{c o v _ {v} \left(\boldsymbol {u} ^ {(\ell)} , \tilde {\mathbf {u}} _ {1 : M} ^ {(\ell)} , \tilde {v} _ {1 : M} ^ {(\ell)}\right)} \right] \tag {S16} \\ + \left(\text { some   terms   independent   from } g (.)\right). \\ \end{array}
$$

In Eq.S15 the term $p(\mathcal{Y}|Parent(\mathcal{Y}))$ is the likelihood of the output(s) of the whole pipeline as illustrated by Fig.1a of the main article, given the ANN's output and all other intermediate variables on which the final output $\mathcal{Y}$ depends. This likelihood turns out to be equivalent to commonly-used losses like the cross-entropy loss or the mean-squared loss. Here we elaborate upon how this happens. Let the task be a classification, and let $\hat{\mathcal{Y}} \in \mathbb{R}^L$ be the pipeline's output. The final model prediction $\mathcal{Y}$ is done as follows:

$$
\mathcal {Y} \sim \mathcal {C a t e g o r i c a l} (\hat {\mathcal {Y}} _ {K},..., \hat {\mathcal {Y}} _ {K}) \tag {S17}
$$

Therefore we have that

$$
p \big (\mathcal {Y} | P a r e n t (\mathcal {Y}) \big) = (\hat {\mathcal {Y}} _ {1}) ^ {I [ \mathcal {Y} = = 1 ]} \times ... \times (\hat {\mathcal {Y}} _ {K}) ^ {I [ \mathcal {Y} = = K ]}, \tag {S18}
$$

where $I[\cdot]$ is the indicator function. So, we have that

$$
\begin{array}{l} \log p (\mathcal {Y} \mid \text {Parent} (\mathcal {Y})) = \tag {S19} \\ I [ \mathcal {Y} = = 1 ] \log (\hat {\mathcal {Y}} _ {1}) + \dots + I [ \mathcal {Y} = = K ] \log (\hat {\mathcal {Y}} _ {K}). \\ \end{array}
$$

Therefore, when the pipeline is for classification, $\log p(\mathcal{Y}|\mathbf{v},\text{etc.})$ will be equal to the cross-entropy loss. This conclusion was introduced and discussed in Eq.6 of the main article. We can draw similar conclusions when the pipeline is for other tasks like regression, or even a combination of tasks.

In the general pipeline of Fig.S1, if all stages after v are deterministic (of course except the final stage which is probabilistic like Eq.S17), the third term on the right-hand-side of Eq.S15 becomes 1. Therefore, the right-hand-side of Eq.S15 is equal to Eq.6 of the main article. As we discussed in Sec.2.3 of the main article, $L_{ann}$ has two terms: the first terms encourages the GP-ANN analogy and the second term seeks to lower the task-loss.

Algorithm S1 Method Forward\_GP   
Input: Input instance x and inducing instance $\tilde{x}$ , list of matrices U, list of vectors V.

Output: List of GP posterior means $\mu$ , and covariances cov. Initialisation : $\mu = \text{list}(L)$ , $\text{cov} = \text{list}(L)$ .

1: for $\ell = 1$ to L do

2: $u = f_{\ell}(\boldsymbol{x}) // \text{map } \boldsymbol{x}$ to the kernel space of the $\ell$ -th GP.

3: $U_{\ell} \leftarrow U[L] // \text{get the inducing points of the } \ell$ -th GP.

4: $V_{\ell} \leftarrow V[L] // \text{observed values at the inducing points}$ .

5: if training then

6: $U_{\ell}[\tilde{x}.index] \leftarrow f_{\ell}(\tilde{x}) // \text{to pass gradient w.r.t. } f_{\ell}(.)$ 7: end if

8: $\mu[\ell] \leftarrow \boldsymbol{u}^{T}U_{\ell}^{T}\left(U_{\ell}U_{\ell}^{T} + \sigma_{gp}^{2}I\right)^{-1}V_{\ell}$ .

9: $cov[\ell] \leftarrow \boldsymbol{u}^{T}\boldsymbol{u} - \boldsymbol{u}^{T}U_{\ell}^{T}\left(U_{\ell}U_{\ell}^{T} + \sigma_{gp}^{2}I\right)^{-1}U_{\ell}\boldsymbol{u}$ .

10: end for

11: return $\mu$ and cov

Algorithm S2 Method Optim\_KernMappings   
Input: Input instance x and inducing instance $\tilde{x}$ , list of matrices U, list of vectors V.

Output: Kernel-space mappings $[f_{1}(.),...,f_{L}(.)]$ .

Note the important modifications to Alg.S2 which are explained in Sec.S5.

Initialisation : loss ← 0.

1: $\mu$ , cov ← forward_GP(x, $\tilde{x}$ , U, V) // feed x to GPs.

2: $\mu_{ann} \leftarrow g(\mathbf{x}) // feed x to ANN$ .

3: for $\ell = 1$ to L do

4: loss ← loss + $\frac{(\mu[\ell]-\mu_{ann}[\ell])^{2}+\sigma_{g}^{2}}{cov[\ell]} + \log(cov[\ell])$ . // Eq.5.

5: end for

6: $\delta \leftarrow \frac{\partial loss}{\partial params([f_{1}(.),...,f_{L}(.)])}$ . // the gradient of loss.

7: $params([f_{1},...,f_{L})] \leftarrow params([f_{1},...,f_{L}) - lr \times \delta // update the parameters.$ 8: $lr \leftarrow updated learning rate$ 9: return $[f_{1}(.),...,f_{L}(.)]$

# S1.3 Deriving the Lower-bound With Respect to $q_{2}(.)$ Parameters

In Eq.4 of the main article we considered the variational parameters $\{\varphi_m^{(\ell)}\}_{m=1}^M$ for the hidden variables $\{\tilde{v}_m^{(\ell)}\}_{m=1}^M$ . The ELBO of Eq.S3 can be optimized with respect to $\{\varphi_m^{(\ell)}\}_{m=1}^M$ as well. But we noticed that optimizing $\{\varphi_m^{(\ell)}\}_{m=1}^M$ is computationally unstable. Therefore, we set $\{\varphi_m^{(\ell)}\}_{m=1}^M$ according to the following rule:

$$
\begin{array}{l} \varphi_ {m} ^ {(\ell)} = g _ {\ell} (\tilde {\boldsymbol {x}} _ {m}), \\ 1 \leq m \leq M, 1 \leq \ell \leq L. \\ \end{array}
$$

We set $\{\varphi_m^{(\ell)}\}_{m = 1}^M$ as above because $\tilde{v}_m^{(\ell)}$ is simply the $\ell$ -th GP posterior mean at the inducing point $\tilde{x}_m$ . To make the GP's posterior mean equal to the ANN's output, $\tilde{v}_{\ell}^{(m)}$ should be equal to the ANN's (i.e. $g(.)'$ s) output at the $m$ -th inducing point.

# S2 ALGORITHM DETAILS

During training, to compute GP's posterior we firstly need to have the $M$ inducing points $\{(\tilde{\boldsymbol{u}}_m^{(\ell)},\tilde{v}_m^{(\ell)})\}_{m = 1}^M$

Algorithm S3 Method Init\_GPparams   
Input: Dataset of inducing points $[\tilde{x}_{1},..., \tilde{x}_{M}]$ .
Output: List of matrices U, list of vectors V.
Initialisation : U = list(L), V = list(L).
1: for $\ell = 1$ to L do
2: $\mathbf{V}[\ell] \leftarrow [g(\tilde{\mathbf{x}}_{1})[\ell], ..., g(\tilde{\mathbf{x}}_{M})[\ell])]$ .
3: end for
4: for $\ell = 1$ to L do
5: $\mathbf{U}[\ell] \leftarrow [f_{\ell}(\tilde{\mathbf{x}}_{1}), ..., f_{\ell}(\tilde{\mathbf{x}}_{M})]$ .
6: end for
7: return U and V

Algorithm S4 Method Explain\_ANN   
Input: Training dataset ds_train, and the inducing dataset ds_inducing.
Output: Kernel-space mappings $[f_{1}(.),...,f_{L}(.)]$ , and the other GP parameters U and V.
Initialisation : U, V ← Init_GPparams(ds_inducing).
1: for iter = 1 to max_iter do
2:    x ← randselect(ds_train).
3: $\tilde{x} \leftarrow$ randselect(ds_inducing)
4: $[f_{1}(.),...,f_{L}(.)] \leftarrow$ Optim_KernMapings(x, $\tilde{x}$ , U, V).
5: $\tilde{x} \leftarrow$ randselect(ds_inducing).
6:    for $\ell = 1$ to L do
7:    //update kernel-space representations.
8:    U[ $\ell$ ][ $\tilde{x}.index$ ] ← f $_{\ell}$ ( $\tilde{x}$ )
9:    end for
10: end for
11: return $[f_{1}(.),...,f_{L}(.)]$ , U, V

It is computationally prohibitive to repeatedly update $\{\tilde{\boldsymbol{u}}_m^{(\ell)}\}_{m = 1}^M$ by mapping all $M$ instances to the kernel space as $\tilde{\boldsymbol{u}}_m^{(\ell)} = f_\ell (\tilde{\boldsymbol{x}}_m)$ . On the other hand, as the kernel-space mappings $\{f_\ell (.)\}_{\ell = 1}^L$ keep changing during training, we need to somehow track how the inducing points $\{\tilde{\boldsymbol{u}}_m^{(\ell)}\}_{m = 1}^M$ change during training. To this end, we consider a matrix whose $m$ -th row contains the value of $f_\ell (\tilde{\boldsymbol{x}}_m)$ at some point during training, where $\tilde{\boldsymbol{x}}_m$ is the $m$ -th inducing instance. During training, we keep updating the rows of this matrix by feeding mini-batches of instances to $f_\ell (.)$ . Note that we have as many GPs as the number of ANN's output heads. Therefore, for each GP we consider a separate matrix containing the representations of the inducing instances in the $\ell$ -th kernel space. In Algs.S1, S2, S3, and S4 the variable $\mathbf{U}$ is a list containing all of the aforementioned matrices. To explain a given ANN, we let the ANN to be fixed and we only train the GPs' parameters. This procedure is explained in Alg.S4. In each iteration, the kernel-mappings are updated according to the objective function of Eq.5 (line 3 of Alg.S2). Afterwards, to make the matrices in $\mathbf{U}$ track the changes in $[f_1(.),...,f_L(.)]$ , we map an inducing instance (or a mini-batch of inducing instances) to the kernel spaces, and we update the corresponding matrices and rows in $\mathbf{U}$ according to the newly obtained kernel-space representations. Updating $\mathbf{U}$ is done in line 8 of Alg.S4. The method in Alg.S1 computes the GPs' posterior means and covariances at any instance like $\pmb{x}$ , given the observed inducing points as specified by $\mathbf{U}$ and $\mathbf{V}$ . Note that this

Algorithm S5 Method Efficiently\_Compute\_AATinvb   
Input: Matrix A of size $M \times D$ , vector b of size $M \times 1$ , and positive scalar $\sigma$ .

Output: The vector output = $(\mathbf{A}\mathbf{A}^{T} + \sigma^{2}\mathbf{I})^{-1}\mathbf{b}$ .

1: $\tilde{\mathbf{E}}, \tilde{\boldsymbol{\lambda}} \leftarrow eigendecomp(\mathbf{A}^{T}\mathbf{A} + \sigma^{2}\mathbf{I})$ .

2: $[\tilde{\boldsymbol{e}}_{1}, ..., \tilde{\boldsymbol{e}}_{D}] \leftarrow \tilde{\mathbf{E}}$ 3: $[\tilde{\lambda}_{1}, ..., \tilde{\lambda}_{D}] \leftarrow \tilde{\boldsymbol{\lambda}}$ 4: $[\boldsymbol{e}_{1}, ..., \boldsymbol{e}_{D}] \leftarrow [\mathbf{A}\tilde{\boldsymbol{e}}_{1}, ..., \mathbf{A}\tilde{\boldsymbol{e}}_{D}]$ 5: $[\lambda_{1}, ..., \lambda_{D}] \leftarrow [\tilde{\lambda}_{1}, ..., \lambda_{D}]$ 6: $E \leftarrow [\boldsymbol{e}_{1}, ..., \boldsymbol{e}_{D}]$ 7: $\Lambda \leftarrow diagonal(\frac{1}{\lambda_{1} + \sigma^{2}}, ..., \frac{1}{\lambda_{D} + \sigma^{2}})$ 8: output $\leftarrow E\Lambda E^{T}\mathbf{b} + \frac{1}{\sigma^{2}}(\mathbf{b} - EE^{T}\mathbf{b})$ //according to //Eq.S21 in supplementary material

9: return output

method returns two outputs, because a GP's posterior at x is a normal distribution described by its mean and variance. In Alg.S1 lines 8 and 9 correspond to the equations of GP posterior (i.e. Eqs. 1 and 2 of the main article). The method in Alg.S1 is used both during training and testing. During training, this method is called whenever ANN's output and GP's posterior are encouraged to be close. During training, according to line 6 of Alg.S1 only the matrix row(s) corresponding to the fed inducing instance(s) are the result of mapping the inducing instance(s) via the kernel-mapping, and all other rows are kept fixed. Line 6 of Alg.S1 allows for computing the gradient of loss with respect to kernel-mappings $[f_1(.),...,f_L(.)]$ . During testing we call Alg.S1 to get the GP's posterior at a test instance like $x_{test}$ . Alg.S3 initializes the GP parameters U and V. For the $\ell$ -th GP, the vector V[ $\ell$ ] is initialized to the $\ell$ -th output head of the ANN at all inducing images. In Alg.S3, the vector V[ $\ell$ ] is initialized in line 2. Moreover, for the $\ell$ -th GP the matrix U[ $\ell$ ] is initialized by mapping all inducing instances to the $\ell$ -th kernel-space via the mapping $f_\ell(.)$ . In Alg.S3 the matrix U[ $\ell$ ] is initialized in line 5. The method in Alg.S3 is called only once before training the GP. For instance, when explaining an ANN in Alg.S4, the initialisation is done once at the beginning of the procedure.

# S2.1 Efficiently Computing Gaussian Process Posterior

Let A be an arbitrary $M \times D$ matrix where $M >> D$ . Moreover, let b be a M-dimensional vector and let $\sigma$ be a scalar. The computational techniques [10] allow us to efficiently compute:

$$
\left(\mathbf {A} \mathbf {A} ^ {T} + \sigma^ {2} \boldsymbol {I} _ {M \times M}\right) ^ {- 1} \boldsymbol {b}.
$$

The idea is that $AA^{T}$ and therefore its inverse are of rank D. Therefore, $(\mathbf{AA}^{T})^{-1}$ has D non-zero eigenvalues like $\{\lambda_{1},\ldots,\lambda_{D}\}$ and the rest of its eigenvalues are zero. Let the corresponding eigenvectors be $\{e_{1},\ldots,e_{D}\}$ . To compute $(\mathbf{AA}^{T})^{-1}\mathbf{b}$ we can simply project b to the D-dimensional space of the eigenvectors. By doing so, we avoid the $\mathcal{O}(M^{3})$ computational complexity. Let $\{\lambda_{1},\ldots,\lambda_{D}\}$ be the non-zero eigenvalues of $AA^{T}$ and let $\{e_{1},\ldots,e_{D}\}$ be the corresponding eigenvectors. From linear algebra, it follows that for $AA^{T}+\sigma^{2}I_{M\times M}$ the eigenvalues and the eigenvectors are

$\{\lambda_{1}+\sigma^{2},\ldots,\lambda_{D}+\sigma^{2},\sigma^{2},\ldots,\sigma^{2}\}$ and $\{e_{1},\ldots,e_{D}\}$ , respectively. Note that M-D eigenvectors are added all of which are equal to $\sigma^{2}$ . Similarly, from linear algebra it follows that for the inverse of $AA^{T}+\sigma^{2}I_{M\times M}$ the eigenvalues and eigenvectors are $\left\{\frac{1}{\lambda_{1}+\sigma^{2}},\ldots,\frac{1}{\lambda_{D}+\sigma^{2}},\frac{1}{\sigma^{2}},\ldots,\frac{1}{\sigma^{2}}\right\}$ and $\{e_{1},\ldots,e_{D},e_{D+1},\ldots,e_{M}\}$ respectively. Note that although there are M eigenvectors, only the first D eigenvectors appear in our computations. More precisely, let $E\in R^{M\times D}$ be a matrix whose columns are $\{e_{1},\ldots,e_{D}\}$ . Let $\Lambda$ be a diagonal matrix whose diagonal is formed by $\left\{\frac{1}{\lambda_{1}+\sigma^{2}},\ldots,\frac{1}{\lambda_{D}+\sigma^{2}}\right\}$ . In the space of the D eigenvectors the linear transformation on any vector like b is equal to $E\Lambda E^{T}b$ , meaning that multiplication by $E^{T}$ transforms b to the space of the D eigenvectors, multiplication by $\Lambda$ performs the transformation in that space, and multiplication by E transforms the result back to the original space. The $(M-D)$ eigenvalues that correspond to the rest of the eigenvectors are all the same and are equal to $\frac{1}{\sigma^{2}}$ . Therefore, there is no need to project b to the space of the $(M-D)$ eigenvectors because the linear transformation in that space is simply a scaling by $\frac{1}{\sigma^{2}}$ . All in all, we have that

$$
\left(\mathbf {A} \mathbf {A} ^ {T} + \sigma^ {2} \mathbf {I} _ {M \times M}\right) ^ {- 1} \boldsymbol {b} = \mathbf {E} \boldsymbol {\Lambda} \mathbf {E} ^ {T} \boldsymbol {b} + \frac {1}{\sigma^ {2}} (\boldsymbol {b} - \mathbf {E} \mathbf {E} ^ {T} \boldsymbol {b}). \tag {S21}
$$

Complexity of computing the right-hand-side of Eq.S21 is way lower than the $\mathcal{O}(M^{3})$ requirement of the standard matrix inversion. We borrowed more computational ideas from the work on fast spectral clustering [10]. To compute the first D eigenvalues and eigenvectors of $AA^{T}$ , we worked with the D-by-D matrix $A^{T}A$ rather than the M-by-M matrix $AA^{T}$ (recall that D << M), because given the eigenvalues and eigenvectors of $A^{T}A$ , those of $AA^{T}$ are easily computable [10]. The procedure is explained in Alg.S5. In Alg.S5, lines 1-3 compute the eigenvalues/vectors of the matrix $A^{T}A$ . Afterwards, lines 4 and 5 compute the first D eigenvalues/vectors of $AA^{T}$ using those of $A^{T}A$ . Finally, line 8 computes $(\mathbf{AA}^{T} + \sigma^{2}\mathbf{I})^{-1}\mathbf{b}$ according to the right-hand-side of Eq.S21. To make the computations faster, we made use of the following equation $AA^{T} = \sum_{m} A[m, :]A[m, :]^{T}$ , where $A[m, :]$ is the m-th row of the matrix A. Thanks to this equation, we compute $AA^{T}$ only once at the beginning of the training. Afterwards, as each mini-batch alters only some rows of A, we update the previously computed $AA^{T}$ by considering only the effect of the modified rows.

# S2.2 Computing Pixel Contributions to the Similarity

We first explain the idea of CAM [34], afterwards we modify it for the architectures of our kernel modules. Let the kernel mapping $f(.)$ be a convolutional neural network that produces a volumetric map of size $C \times H \times W$ followed by a spatial average pooling that produces the $C$ -dimensional vector in the kernel-space. In this case, $\mathcal{K}(\pmb{x}_1, \pmb{x}_2)$ is as follows:

$$
\begin{array}{l} \mathcal {K} (\boldsymbol {x} _ {1}, \boldsymbol {x} _ {2}) = f (\boldsymbol {x} _ {1}) ^ {T} f (\boldsymbol {x} _ {2}) \\ = \left(\sum_ {i = 1} ^ {H} \sum_ {j = 1} ^ {W} z _ {i j} ^ {(1)}\right) ^ {T} \left(\sum_ {k = 1} ^ {H} \sum_ {\ell = 1} ^ {W} z _ {k \ell} ^ {(2)}\right) \tag {S22} \\ = \sum_ {i = 1} ^ {H} \sum_ {j = 1} ^ {W} \sum_ {k = 1} ^ {H} \sum_ {\ell = 1} ^ {W} \left(\boldsymbol {z} _ {i j} ^ {(1) ^ {T}} \boldsymbol {z} _ {k \ell} ^ {(2)}\right), \\ \end{array}
$$

where $\boldsymbol{z}^{(1)}$ and $\boldsymbol{z}^{(2)}$ are the volumetric maps of size $C \times H \times W$ and the indices $(i, j)$ and $(k, \ell)$ index the spatial locations over the volumetric maps. The last term in Eq.S22 shows that the total similarity $\mathcal{K}(\boldsymbol{x}_{1}, \boldsymbol{x}_{2})$ is the sum of the contributions from each pair of positions $(i, j)$ on $x_{1}$ and $(k, \ell)$ on $x_{2}$ . To compute the contribution of a specific location like $(i, j)$ on $x_{1}$ , we sum up the contributions of $(i, j)$ on $x_{1}$ and all possible locations $\{(k, \ell)\}_{k=1\ell=1}^{H\quad W}$ on $x_{2}$ .

The kernel-mappings that we used have a slightly different architecture than a volumetric map followed by spatial average pooling. Our kernel mappings produce a volumetric map of size $C \times H \times W$ followed by a spatial average pooling that produces a C-dimensional vector. Afterwards, the resulting vector is divided by its $\ell_{2}$ -norm to produce a vector of norm 1. Consequently, this vector of norm 1 is fed to a leaky ReLU layer that produces the final kernel-space representation $f(\boldsymbol{x})$ . For this architecture the pixel contributions can be computed according to an equation similar to Eq.S22 as follows. Our kernel mappings produce the volumetric map z of size $C \times H \times W$ followed by a spatial average pooling that produces the C-dimensional vector a:

$$
\boldsymbol {a} = \sum_ {i = 1} ^ {H} \sum_ {j = 1} ^ {W} z _ {i j}. \tag {S23}
$$

Afterwards, the resulting vector is divided by its $\ell_2$ -norm to produce the vector $b$ of norm 1:

$$
\boldsymbol {b} = [ \frac {a _ {1}}{| | \boldsymbol {a} | | _ {2}}, \dots , \frac {a _ {C}}{| | \boldsymbol {a} | | _ {2}} ]. \tag {S24}
$$

Consequently, this vector of norm 1 is fed to a leaky ReLU layer that produces the final kernel-space representation $f(\pmb{x})$ :

$$
f (\boldsymbol {x}) = \text { leakyReLU } (\boldsymbol {b}). \tag {S25}
$$

We begin with simplifying Eq.S25. The leaky ReLU activation function multiplies the input by a constant and this constant depends on the sign of the input. Therefore, applying the leaky ReLU activation is equivalent to multiplication by a diagonal matrix $\Lambda$ . Therefore,

$$
f (\boldsymbol {x}) = \boldsymbol {\Lambda} \boldsymbol {b}. \tag {S26}
$$

Let $x_{1}$ and $x_{2}$ be two images, and $z^{(1)}$ and $z^{(2)}$ be the corresponding volumetric maps. We have that

$$
\boldsymbol {a} ^ {(1)} = \sum_ {i = 1} ^ {H} \sum_ {j = 1} ^ {W} z _ {i j} ^ {(1)}, \tag {S27}
$$

$$
\boldsymbol {a} ^ {(2)} = \sum_ {k = 1} ^ {H} \sum_ {\ell = 1} ^ {W} \boldsymbol {z} _ {k \ell} ^ {(2)}.
$$

And

$$
\boldsymbol {b} ^ {(1)} = \left[ \frac {a _ {1} ^ {(1)}}{\left| \left| \boldsymbol {a} ^ {(1)} \right| \right| _ {2}}, \dots , \frac {a _ {C} ^ {(1)}}{\left| \left| \boldsymbol {a} ^ {(1)} \right| \right| _ {2}} \right], \tag {S28}
$$

$$
\boldsymbol {b} ^ {(2)} = [ \frac {a _ {1} ^ {(2)}}{| | \boldsymbol {a} ^ {(2)} | | _ {2}}, \dots , \frac {a _ {C} ^ {(2)}}{| | \boldsymbol {a} ^ {(2)} | | _ {2}} ].
$$

And

$$
f (\boldsymbol {x} ^ {(1)}) = \boldsymbol {\Lambda} ^ {(1)} \boldsymbol {b} ^ {(1)}, \tag {S29}
$$

$$
f (\boldsymbol {x} ^ {(2)}) = \boldsymbol {\Lambda} ^ {(2)} \boldsymbol {b} ^ {(2)}.
$$

Now we simplify the similarity $\mathcal{K}(\pmb{x}_1,\pmb{x}_2)$ :

$$
\begin{array}{l} \mathcal {K} (\boldsymbol {x} _ {1}, \boldsymbol {x} _ {2}) = \left(\boldsymbol {\Lambda} ^ {(1)} \boldsymbol {b} ^ {(1)}\right) ^ {T} \left(\boldsymbol {\Lambda} ^ {(2)} \boldsymbol {b} ^ {(2)}\right) \\ = \left(\boldsymbol {\Lambda} ^ {(1) ^ {T}} \boldsymbol {\Lambda} ^ {(2)}\right) \left(\boldsymbol {b} ^ {(1) ^ {T}} \boldsymbol {b} ^ {(2)}\right) \\ = \frac {\left(\boldsymbol {\Lambda} ^ {(1) ^ {T}} \boldsymbol {\Lambda} ^ {(2)}\right)}{| | \boldsymbol {a} ^ {(1)} | | _ {2} | | \boldsymbol {a} ^ {(2)} | | _ {2}} \left(\sum_ {i = 1} ^ {H} \sum_ {j = 1} ^ {W} \boldsymbol {z} _ {i j} ^ {(1)}\right) ^ {T} \left(\sum_ {k = 1} ^ {H} \sum_ {\ell = 1} ^ {W} \boldsymbol {z} _ {k \ell} ^ {(2)}\right) \\ = \frac {\left(\boldsymbol {\Lambda} ^ {(1) ^ {T}} \boldsymbol {\Lambda} ^ {(2)}\right)}{\left| \left| \boldsymbol {a} ^ {(1)} \right| \right| _ {2} \left| \left| \boldsymbol {a} ^ {(2)} \right| \right| _ {2}} \sum_ {i = 1} ^ {H} \sum_ {j = 1} ^ {W} \sum_ {k = 1} ^ {H} \sum_ {\ell = 1} ^ {W} \left(\boldsymbol {z} _ {i j} ^ {(1) ^ {T}} \boldsymbol {z} _ {k \ell} ^ {(2)}\right). \tag {S30} \\ \end{array}
$$

Indeed, as the used architecture for kernel-mappings is slightly different than producing a volumetric map followed by spatial average pooling, instead of Eq.S22, we used Eq.S30 that we derived above.

# S3 EXAMINING FAITHFULNESS OF GPs TO ANNs

In Sec.4.1. of the main article, we examined the faithfulness of the found GPs to their corresponding ANNs. In this section we provide more information and insights about the analogy between the GPs found by our proposed GPEX and their corresponding ANNs. Figs.S2, S4, and S6 illustrate the scatter plots of ANN-GP outputs on Cifar10 [15], MNIST [6], and Kather [12], respectively. These scatter plots are obtained on the testing set which has been invisible to the proposed GPEX. Note that in Figs.S2, S4, and S6 each ANN's output head and its corresponding GP have a separate scatter plot.

In the main article, we discussed that our proposed GPEX is applicable to any subcomponent of a pipeline. To verify this, in Sec.4.1. of the main article we applied the proposed GPEX to attention subcomponents of classifier pipelines. Here we provide more information about the faithfulness of the found GPs to the attention subcomponents. Figs.S3, S5, and S7 illustrate the scatter plots for attention submodules and their corresponding GPs.

For Cifar10 [15] in Fig.S3, each attention mask is $3 \times 3$ and we have 9 scatter plots. According to Fig.S3, in attention masks some output heads like head 1, head 2, and head 3 do not turn on for any instnace (the values change around -2, and sigmoid of -2 is a small number). Therefore, in Fig.3 of the main article we have excluded the attention heads which are always off. Similarly, for MNIST [6] and Kather [12] we see some attention heads are always off in Figs.S5 and S7, and we have excluded those heads in Fig.3 of the main article.

So far we reported correlation coefficients (Fig.3 of the main article) and scatter plots (Figs.S2, S3, S4, S5, S6, S7) to examine the faithfulness of GPs to their corresponding ANNs. To get more insights, we selected mini-batches of testing instances and fed each mini-batch to both ANN and corresponding GPs. The output from ANN (and similarly GPs) is a matrix of shape $batchsize \times D_{v}$ , where $D_{v}$ is the number of output heads from the ANN. Ideally, we should get two identical $batchsize \times D_{v}$ matrices for each mini-batch, because the GPs are supposed to be faithfull to ANNs. Figs. S59, S60, S61, and S62 illustrate the heatmaps for four randomly fed mini-batches from Cifar10 [15], MNIST [6], Kather [12], and DogsWolves [30], respectively. According

to Figs. S59, S60, S61, and S62 the outputs from GPs almost match those from their corresponding ANNs. In Figs. S59, S60, S61, and S62 the red rectangles show the test instances for which the GP's decision (i.e. the class with the highest score) does not match the ANN's decision. According to Figs. S59, S60, S61, and S62 the disagreement between GPs prediction and ANN prediction mostly happens when either some output activations are very close to one another or all activations are close to zero. This is consistent with the scatter plots of Figs. S2, S4, and S6 in which the scatters are slightly dispersed for intermediate values. Tab. S1 reports the test accuracy of the ANNs and their corresponding GPs. We see that GPs' accuracies are slightly lower than those of the corresponding ANNs. Figs. S59, S60, S61, and S62 provide insights about how this small disagreement can be potentially solved in future research by, e.g., preventing the ANN from having near-zero activations or having output heads which are very close to one another. We repeated the experiment with 5 different random splits and reported the results in Fig. S67. According to Fig. S67 the correlation coefficients are high for different training/testing splits.

We repeated the experiment of Figs. S59, S60, S61, and S62 for the attention submodules and corresponding GPs. Reults are provided in Figs. S63, S64, and S65. According to Figs. S63, S64, and S65 our proposed GPEX has found GPs which are faithful to the attention submodules.

# S4 EXPLAINING ANNs' DECISIONS

In Sec.4.2 of the main article we applied our proposed method (i.e. Alg.S4) to some ANN classifiers. Afterwards, we explained the decisions made by the ANNs via the GPs and the kernel-spaces that our proposed GPEX has found. Here we are going to provide more explanations for ANNs' decisions on more testing instances.

We explain the decision made for a test instance like $x_{test}$ as follows. We consider the GP and the kernel-space that correspond to the ANN's head with maximum value (i.e. the ANN's head that relates to the predicted label). Consequently, among the instances in the inducing dataset, we find the 10 closest instances to $x_{test}$ , like $\{x_{i1}, x_{i2}, ..., x_{i10}\}$ . Intuitively the ANN has labeled $x_{test}$ in that way because it has found $x_{test}$ to be similar to $\{x_{i1}, x_{i2}, ..., x_{i10}\}$ . Besides finding the nearest neighbours, we provide explanation as to why $x_{test}$ and an instance like $x_{ij}, 1 \leq j \leq 10$ are considered similar by the model. The procedure is explained in Sec.S2.2.

For MNIST digit classification, some test instances and nearest neighbours in training set are shown in Figs.S8, S9, S10, and S11. In these figures each row corresponds to a test instance. The first column depicts the test instance itself and columns 2 to 11 depict the 10 nearest neighbours. According to rows 2 and 3 of Fig.S8, the classifier has labeled the two images as digit 1 because it has found 1 digits with similar inclinations in the training set. We see the model has also taken the inclination into account for the test instances of rows 8 and 9 of Fig.S8 and rows 1, 2, and 3 of Fig.S11. In Fig.S8, according to rows 4, 5, and 6 the test instances are classified as digit 2 because 2 digits with similar styles are found in the training set. We see the model has also taken the style into account for the test instances of rows 7, 8, 9, 10, 11 of Fig.S8 and rows 1, 2, 3, 4, 5, 6, 7, and 8 of Fig.S9. For instance, the test instance in row 1 of Fig.S9 is a 4 digit with a short tail and the two nearest neighbours are alike. Or for the test instances in rows 5, 6, 7, and 8 of Fig.S10 the test instances have incomplete circles in the same way as their nearest neighbours.

Figs.S12, S13, S14, S15, S16, S17, S18, and S19 illustrate sample explanations for similarities. For instance row 1 of Fig.S12 illustrates a test instance as well as the 10 nearest neighbours. The second row of Fig.S12 highlights to what degree each region of each nearest neighbour contributes to its similarity to the test instance. The third row of Fig.S12 illustrates to what degree each region of the test instance contributes to its similarity to each of the nearest neighbours. For example, according to rows 1, 2, and 3 of Fig.S17 the cross pattern of the 8 digits have had a significant contribution to their similarities. For MNIST [6], more similarity explanations are provided in Figs.S12, S13, S14, S15, S16, S17, S18, and S19.

Figs.S36, S37, S38, S39, S40, S41, S42, S43, S44, S45, S46, S47, S48, and S49 illustrate some sample explanations for Cifar10 [15]. Like before, each row corresponds to a test instance, the first column depicts the test instance itself and columns 2 to 11 depict the 10 nearest neighbours. In rows 8, 9, 10, and 11 of Fig.S44 and rows 1 and 2 of Fig.S45, the test instances are captured from horses' heads from closeby, and the nearest neighbours are alike. However, in rows 3, 4, 5, 6, 7, 8, and 9 of Fig.S45 the test images are taken from faraway and the found similar training images are also taken from faraway. Intuitively, as the classifier is not aware of 3D geometry, it finds training images which are captured from the same distance. We constantly observe this pattern in more explanations: row 6, 7, 8, 9, 10, and 11 in Fig.S39, all rows of Fig.S40, rows 1, 2, 6, 7, 8, 9, 10 and 11 of Fig.S41, rows 1, 7, 8, 9, 10, and 11 of Fig.S42, rows 1, 2, 3, 4, 5, 6, 7, and 8 of Fig.S43, all rows of Fig.S45 and rows 1-10 of Fig.S46.

Animal faces tend to be recognized by similar faces. We see this pattern in rows 2, 3, 4, 5 and 6 of Fig.S40, rows 6, 7, 8, and 9 of Fig.S41, rows 7 and 8 of Fig.S43, rows 8, 9, 10, and 11 of Fig.S44 and rows 1, 2, 10, and 11 of Fig.S45. To classify airplanes, the model has taken into account the inclination. For instance, in Fig.S36 the model has taken into account whether the airplane is taking off (rows 1, 8, 9, 10, and 11 of Fig.S36), flying straight (rows 2 and 4 of Fig.S36) or is inclined downwards (rows 3, 5, 6 and 7 of Fig.S36). Furthermore, the bat-like airplanes are recognized by the model because similar bat-like airplanes are found in the training set, as we see in rows 1, 2, 3, 4, 5, 6 and 7 of Fig.S37. Cessnas are often classified by finding cessnas in the training set, as we see in rows 8, 9 and 10 of Fig.S37 and row 1 of Fig.S38.

Since the classifier has no knowledge about 3D geometry, it tends to find training instances which are captured from the same angle as the test instance, as we see in rows 6, 7, 8, 9, 10 and 11 of Fig.S39, rows 7, 8, 9, 10 and 11 of Fig.S42, rows 9, 10 and 11 of Fig.S43, rows 1, 2, 3, 4, 5, 6 and 7 of Fig.S44, row 11 of Fig.S46, all rows of Fig.S47, and rows 1, 2, 3, 4, 5, 6, 7, and 8 of Fig.S48. In rows 3, 4, and 5 of Fig.S41 it seems the model takes into account the ostrich-like shape of the animal. In rows 2, 3, 4, and 6 of Fig.S42 the horns seem to have an effect. In rows 6, 7, 8, and 9 of Fig.S45, we see

the model have made use of the riders to classify the test instances as horse. According to rows 1, 2, 3, 4, 5, 6, 7, 8, 9, and 10 of Fig.S46, the model distinguishes between medium sized ships and huge cargo ships. To classify firefighter trucks, model tends to find similar firefighter trucks in the training set, as we see in rows 10 and 11 of Fig.S47, and rows 1, 2, 3, and 4 of Fig.S48. For some testing instances, the model finds training instances which are almost identical to the test instance, as we see in rows 2 and 5 of Fig.S40, row 7 of Fig.S42, row 8 of Fig.S43, and row 8 of Fig.S48.

In rows 2, 4, 5, 6, 7, 8, 9, 10, and 11 of Fig.S38 it seems the classifier has taken into account the blue background. We used the proposed GPEX to explain as to why some testing instances get misclassified. Rows 9, 10, and 11 of Fig.S48 and all rows of Fig.S49 illustrate some instances which are misclassified. For instance in row 10 of Fig.S48 the test image shows an airplane, but the model has classified it as a cat, because it is similar to the cat faces shown in columns 2 to 11 (can you find the cat face in the airplane image?). In row 11 of Fig.S48, the car is classified as truck partially because it very similar to the truck at column 2. In row 2 of Fig.S49, the deer is classified as horse partially because it is very similar to the training image shown in column 2. In row 3 of Fig.S49, we hypothesize the dog is classified as cat because the model has taken into account the cyan and red colors in the background. In this case, adding dog images with cyan and red background may make the model classify this test instance correctly. In rows 5 and 6 of Fig.S49, the model correctly understands the test images are similar to some faces from other animals, but it fails to find similar frog faces in the training set. In this case, adding more images from frog faces may solve this issue. In row 7 of Fig.S49 the horse is classified as airplane, because the model thinks the horse image is similar to some airplane training images which are taking off. Interestingly, the jumping frog in column 4 has been considered similar to the horse image. It seems having inclined edges (due to taking off, jupming) has contributed to the similarities, and therefore the model has incorrectly classified the horse as airplane.

For the DogsWolves dataset [30] the explanations are provided in Figs.S25-S35. According to rows 10, 11, and 12 of Fig.S29, the red ball in the dog's mouth (as highlighted in row 12 of Fig.S29) has the most contribution to the similarities. According to row 2 of Fig.S29, patterns like human hand in column 4 or woody or pink background in columns 8, 10, and 11 are highlighted in nearest neighbours while in the test insntace (row 3 of Fig.S29) the red ball at the bottom right is highlighted. Our explanations consistently show that the model detects dogs by any pattern that rarely appear in a wolf image. For instance in rows 4-6 of Fig.S29, according to row 4 humans in columns 3, 9, and 11, and dog collars or costumes in columns 4, 5, 6, and 10, and the brick wall in the test instance (row 6 of Fig.S29) are used by the model. According to rows 9, 12, and 15 of Fig.S29, the flowers, the red ball in the dogs mouth, and the children are used by the model, respectively. According to rows 3, 6, 9, 12, and 15 of Fig.S30, the red rope, the dog's color, red patterns, brown background and brown background are used by the model, respectively. According to rows 3, 6, 9, 12, and 15 of Fig.S31, brown background, human, brown background, the red wallet, and the pink ball are used by the model, respectively. According to rows 3, 6, 9, 12, and 15 of Fig.S32, the child, pink pillow, brown color, orange background, and red blood are used by the model, respectively. Note that in Fig.S32 the last two instances (rows 10-15) are misclassified. In Fig.S33 all test instances get misclassified. According to rows 3, 6, 9, 12, and 15 of Fig.S33, colorful background, the red object attached to the wolf, background, white background, and dark-green background are used by the model, respectively. Figs.S34 and S35 illustrate more explanations. For instance, according to row 6 of Fig.S34 and row 12 of Fig.S35, the test instances are misclassified due to their dark background. Moreover, according to rows 3, 6, and 15 of Fig.S35, the test instances are misclassified due to their background. All in all, our explanations reveal that for the DogsWolves dataset [30] the model makes use of potentially incorrect clues to label instances. This is not surprising because the dataset has only 2000 images.

For Kather dataset [12], some explanations are shown in Figs.S20, S21, S22, S23, and S24. Like before, in Figs.S20 and S21 each row corresponds to a test instance, the first column depicts the test instance itself and columns 2 to 11 depict the 10 nearest neighbours. In row 1 of Fig.S20, the test image is classified as fat tissue. According to rows 1, 2, and 3 of Fig.S22, the similarity is due to the wire mesh formed by cellular membranes described by our expert pathologist. Row 13 of Fig.S22 shows cancer-associated stroma which is classified correctly. All 10 nearest neighbours are also cancer-associated stroma. Distinguishing between cancer-associated stroma and normal smooth muscle is a challenging task even for expert pathologists, and they often look similar. According to rows 13, 14, and 15 of Fig.S22, the model cares about both the stroma and nuclei. In row 7 of Fig.S22, the test image is correctly classified as lymphocytes. For a pathologist they represent scattered well defined round structures. According to rows 7, 8, and 9 of Fig.S22, the model considers all regions which matches the way pathologists recognize lymphocytes. In rows 1, 2 and 3 of Fig.S23 and rows 1, 2, and 3 of Fig.S24, for the two test instances the model takes into account nuclei which is not the same way that a pathologists would classify the images. We hypothesize that for the model it is easier to extract features from nuclei than to consider the context information. Because even small changes in nuclei is easily measurable by the model while it is not easily noticeable by human eyes. The test image in row 7 of Fig.S24 gets misclassified. According to rows 7, 8, and 9 of Fig.S24 the artificial white holes are considered as glandular lumens by the model and that explains why the test instance gets misclassified. The test image in row 10 of Fig.S23 gets misclassified. According to rows 10, 11, and 12 of Fig.S23, the test image is smooth muscle. But it contains artifactual white spaces (retractions) like the found similar training instances. This make the model think the test image is similar to debris images that contain artifactual white spaces. For Kather dataset [12], more sample explanations are provide in Figs.S22, S23, and S24.

# S5 PRACTICAL DETAILS AND PARAMETER SETTINGS

In this section we discuss some practical details which we have not yet discussed in this paper. Moreover, we provide the exact parameter settings that we used throughout our experiments. As explained in Sec.3 of the main article, there are L kernel mappings that we denoted by $[f_{1}(.), f_{2}(.), ..., f_{L}(.)]$ . One can implement this kernel mappings by, e.g., considering L independent CNNs. However, doing so dramatically increases the computation cost. Therefore, we modeled the L mappings by a common ResNet-50 [9] backbone. After the common backbone, we placed L branches. Each branch has two convolutional layers followed by global spatial average pooling that produce a vector. Each branch ends with an L2 normalizer layer (that sets the L2-norm of the vector to 1) followed by a leaky-ReLU layer. During our experiments, we noticed that the L2-normalization layer and the final leaky-ReLU layer are essential. Without the L2 normalization layer, the vectors in the kernel-space can have arbitrarily-small or arbitrarily-big elements, and this makes the training unstable. We included the last leaky-ReLU layer, because according to GP posterior mean formula, vectors in the kernel-space go through a linear transformation. Therefore, without the last leaky-ReLU layer, the pipeline would have two consecutive linear layers. Throughout our experiments, we set the output of each branch (i.e. vectors in the kernel-space of each GP) to be 20-dimensional.

As illustrated by Fig.S66 (to be discussed in Sec.S7), we need to make the inducing dataset as large as possible. Therefore, throughout our experiments we selected the whole training dataset as the inducing dataset. Unlike training instances, we didn't apply data-augmentation on inducing instances. By doing so, the training dataset and the inducing dataset will have very similar instances. This causes a difficulty that we are going to discuss in this part. The kernel-mappings $[f_1(.),...,f_L(.)]$ are trained according to Alg.S4. After selecting an instance like $x$ from the training dataset, $x$ is actually the augmented version of an inducing instance like $\tilde{x}_m$ . Indeed, we have that $x = DataAug(\tilde{x}_m)$ . Because $x$ and $\tilde{x}_m$ are very similar, they will be very close to one another in the kernel-spaces regardless of what parameters $[f_1(.),...,f_L(.)]$ have. Therefore, regardless of the kernel-mappings, the GP-mean will match the ANN value at $x$ , and there will be no training signal for the kernel-mappings $[f_1(.),...,f_L(.)]$ . Note that in this case the GPs match the ANNs only on training instances, and the analogy does not generalize to testing instances. To avoid this issue, we optimized the GP-ANN analogy (i.e. the objective in Eq.5 of the main article) on instances like $\lambda x_i + (1 - \lambda)x_j$ , where $x_i$ and $x_j$ are two instances randomly selected from the training set and $\lambda$ is a scalar uniformly selected from $[-1,2]$ .

When applying our proposed GPEX we used Adam optimizer $[14]$ . Although the AMSGrad version of this optimizer is often recommended, for our proposed GPEX we noticed the Adam optimizer $[14]$ without AMSGrad works the best. For explaining classifier ANNs, we used a learning-rate of 0.0001 while for explaining the attention submodules we used a learning rate of 0.00001. On a RTX3090 GPU, the experiments took around 3 days for the 4 image datasets and around 2 hours for the biological dataset. We ran Alg.S4 for 200 epochs. Afterwards, we ran line 8 of Alg. S4 for the inducing dataset. Afterwards, we continued Alg.S4 for 200 more epochs and repeating line 8 of Alg.S4 10 times instead of once.

# S6 QUALITATIVE COMPARISION OF GPEX AND REPRESENTER POINT SELECTION

We qualitatively compared the explanations of our proposed GPEX to those of representer point selection $[33]$ . The results are provided in Figs.S50, S51, S52, S53, S54, S55, S56, and S57. In each triple, the first row shows the test instance and the 10 nearest neighbours found by our proposed GPEX. The second row shows the 10 nearest neighbours selected by representer point selection $[33]$ . The third row shows the 10 nearest neighbours according to the kernel-space of representer point selection $[33]$ . Representer point selection $[33]$ assigns an importance weight to each training instance. Therefore, some training instances tend to appear as nearest neighbours regardless of what the testing instance is. We see this behaviour in rows 2, 5, 8, 11, and 14 of Figs.S50-S57. However, for our proposed GPEX the nearest neighbours can freely change for different test instances. We see this behaviour in rows 1, 4, 7, 10, and 13 of Figs.S50-S57. If we ignore the importance weights in representer point selection $[33]$ , the aforementioned issue in that method happens less frequently, as we see in rows 3, 6, 9, 12, and 15 of Figs.S50-S57. However, the issue is that without the importance weights, the explainer model in representer point selection $[33]$ will not be faithful to the ANN itself.

# S7 PARAMETER ANALYSIS

To analyze the effect of the number of inducing points (i.e. the variable M in Sec. S2) we applied the proposed GPEX to the classifier CNN that we trained on Cifar10 dataset $[15]$ in Sec. 4.1 of the main manuscript. This time, instead of considering all training instances as the inducing dataset, we randomly selected some training instances. In Fig. S66, the horizontal axis shows the size of the inducing dataset. For each size, we repeated the experiment 5 times (i.e. split 1-5 in Fig. S66). According to Fig. S66, to obtain GPs which are faithful to ANNs one needs to have a lot of inducing points. This highlights the importance of the scalability techniques that we used (the computational techniques are elaborated upon in Sec. S2.1 of supplementary material). Another intriguing point in Fig. S66 is that if we are to select a few training images as inducing points, the correlation coefficients highly depend on which instances are selected. More precisely, Fig. S66 suggests that one may be able to reach high correlation coefficients by selecting a few inducing points from the training set in a subtle way.

So far we analyzed the effect of the size of inducing dataset. Here we analyze two other important factors: the width of the second last layer and number of epochs for which the ANN has been trained. On Cifar10 [15] we trained ANNs with different number of neurons in the second last layer and we analyzed the ANN at different checkpoints during training (10, 50, 100, 150, and 200 epochs). The result is shown in Fig.S58. According to Fig.S58, increasing the

width of the second last layer increases the correlation coefficients. However, as illustrated by Fig.S58, the proposed GPEX can achieve almost perfect match even when the second last layer of the ANN is not wide. Moreover, according to Fig.S58, our proposed GPEX can reach high correlation coefficients even when the ANN's parameters are not a local minimum of the classification loss. This empirical results show that most theoretical results like requiring all layers of the ANN to be wide [5], or requiring the ANN to be optimized on a loss [20] may not be necessary.

![](images/f4c2ed1f655b04a37f2881fc3186485443ecad2a12bfd6bbf46a8de45794578a.jpg)

Fig. S2: Scatters for Cifar10 (classifier).   
![](images/c2f714aac5671014876c30cda0ef440c769599834e313c1352f81bd6cf93b5b8.jpg)

<details>
<summary>scatter</summary>

| GP output (head 1) | ANN output (head 1) |
| ------------------ | ------------------- |
| -2.20              | -2.20               |
| -2.18              | -2.21               |
| -2.16              | -2.22               |
| -2.14              | -2.23               |
| -2.12              | -2.24               |
| -2.10              | -2.24               |
</details>

![](images/3bb3c9a67f528b33effebbd287b701a87a0497bc2e2d82b75822395b6e2a0de5.jpg)

<details>
<summary>scatter</summary>

| GP output (head 2) | ANN output (head 2) |
| ------------------ | ------------------- |
| -2.30              | -2.100              |
| -2.25              | -2.150              |
| -2.20              | -2.200              |
| -2.15              | -2.250              |
| -2.10              | -2.200              |
| -2.05              | -2.150              |
</details>

![](images/b925524df684e55aafc3fe56dc6ba17765ed6d30da931b6ed824163e0fb5d50e.jpg)

<details>
<summary>scatter</summary>

| GP output (head 3) | ANN output (head 3) |
| ------------------ | ------------------- |
| -2.20              | -2.225              |
| -2.15              | -2.175              |
| -2.10              | -2.150              |
| -2.05              | -2.125              |
| -2.00              | -2.100              |
| -1.95              | -2.075              |
| -1.90              | -2.050              |
| -1.85              | -2.025              |
| -1.80              | -2.000              |
| -1.75              | -1.975              |
| -1.70              | -1.950              |
| -1.65              | -1.925              |
| -1.60              | -1.900              |
| -1.55              | -1.875              |
| -1.50              | -1.850              |
| -1.45              | -1.825              |
| -1.40              | -1.800              |
| -1.35              | -1.775              |
| -1.30              | -1.750              |
| -1.25              | -1.725              |
| -1.20              | -1.700              |
| -1.15              | -1.675              |
| -1.10              | -1.650              |
| -1.05              | -1.625              |
| -1.00              | -1.600              |
| -0.95              | -1.575              |
| -0.90              | -1.550              |
| -0.85              | -1.525              |
| -0.80              | -1.500              |
| -0.75              | -1.475              |
| -0.70              | -1.450              |
| -0.65              | -1.425              |
| -0.60              | -1.400              |
| -0.55              | -1.375              |
| -0.50              | -1.350              |
| -0.45              | -1.325              |
| -0.40              | -1.300              |
| -0.35              | -1.275              |
| -0.30              | -1.250              |
| -0.25              | -1.225              |
| -0.20              | -1.200              |
| -0.15              | -1.175              |
| -0.10              | -1.150              |
| -0.05              | -1.125              |
| 0.00               | -1.100              |
</details>

![](images/07dc0ea06d649bda7fcefcce0aabe96d2477e77872edc5484a38508b4100f189.jpg)

<details>
<summary>scatter</summary>

| GP output (head 4) | ANN output (head 4) |
| ------------------ | ------------------- |
| -2.26              | -2.26               |
| -2.24              | -2.24               |
| -2.22              | -2.22               |
| -2.20              | -2.20               |
| -2.18              | -2.18               |
| -2.16              | -2.16               |
| -2.14              | -2.14               |
| -2.12              | -2.12               |
</details>

![](images/38850ccf9030837f84b3f0c751d8079e40c07b412f132d27fce2a7def6291b9b.jpg)

<details>
<summary>scatter</summary>

| GP output (head s) | ANN output (head s) |
| ------------------ | ------------------- |
| -2.24              | -2.18               |
| -2.20              | -2.20               |
| -2.18              | -2.22               |
| -2.16              | -2.24               |
| -2.14              | -2.26               |
| -2.12              | -2.24               |
| -2.10              | -2.22               |
</details>

![](images/445042f2c70716e2c0d596fa2140b5ba6945bbae3a9c94a58427ee8f2ac5b676.jpg)

<details>
<summary>scatter</summary>

| GP output (head 6) | ANN output (head 6) |
| ------------------ | ------------------- |
| -2.0               | -2.0                |
| -1.5               | -1.5                |
| -1.0               | -1.0                |
| -0.5               | -0.5                |
| 0.0                | 0.0                 |
| 0.5                | 0.5                 |
| 1.0                | 1.0                 |
| 1.5                | 1.5                 |
| 2.0                | 2.0                 |
</details>

![](images/eab2e82c256785ac06d571b0eadeb5e8f26bf9a93eb3c7a258014334831dc085.jpg)

<details>
<summary>scatter</summary>

| GP output (head 7) | ANN output (head 7) |
| ------------------ | ------------------- |
| -2.0               | -2.0                |
| -1.5               | -1.5                |
| -1.0               | -1.0                |
| -0.5               | -0.5                |
| 0.0                | 0.0                 |
| 0.5                | 0.5                 |
| 1.0                | 1.0                 |
| 1.5                | 1.5                 |
| 2.0                | 2.0                 |
</details>

![](images/e29325b42e96779cfe3ec6974782bc01b6f96f1a79ad315d254548a9a7fb5687.jpg)

<details>
<summary>scatter</summary>

| GP output (head 8) | ANN output (head 8) |
| ------------------ | ------------------- |
| -2.30              | -2.30               |
| -2.25              | -2.25               |
| -2.20              | -2.20               |
| -2.15              | -2.15               |
| -2.10              | -2.10               |
</details>

![](images/e9c5acef607ee7aaec859e7a7391814239d390b8952a4fbdaf81381a85fb1eab.jpg)

<details>
<summary>scatter</summary>

| GP output (head 9) | ANN output (head 9) |
| ------------------ | ------------------- |
| -2.2               | -2.2                |
| -2.1               | -2.0                |
| -2.0               | -1.9                |
| -1.9               | -1.9                |
</details>

Fig. S3: Scatters for Cifar10 (attention).

![](images/5adc3ec9d0d0efc9b63a206ee529b1f2f060e0b80a334bfc3d188423ebe94865.jpg)

<details>
<summary>scatter</summary>

| GP output (head 1) | NWS output (head 1) |
| ------------------ | ------------------- |
| -50                | -40                 |
| -30                | -20                 |
| -10                | -5                  |
| 0                  | 0                   |
| 10                 | 5                   |
| 20                 | 10                  |
| 30                 | 15                  |
</details>

![](images/27ad7ca32ec7a45b189a026aef424b3af72908c79c54c038e27a50d6334318f0.jpg)

<details>
<summary>scatter</summary>

| GP output (head 2) | ANN output (head 2) |
| ------------------ | ------------------- |
| -10                | -25                 |
| -5                 | -10                 |
| 0                  | 0                   |
| 5                  | 5                   |
| 10                 | 10                  |
| 15                 | 15                  |
| 20                 | 20                  |
</details>

![](images/c856e62cc67499325c9b4cbe962747cdc07ec0be17e77b91ec79c5944b6dfea7.jpg)

<details>
<summary>scatter</summary>

| CP output (head 3) | ANN output (head 3) |
| ------------------ | ------------------- |
| -20                | -15                 |
| -10                | -10                 |
| 0                  | 0                   |
| 10                 | 5                   |
| 20                 | 10                  |
| 30                 | 15                  |
</details>

![](images/3d002599e3ac6689780280649513a51b27e96a147174b952aea1d1412c7b32d7.jpg)

<details>
<summary>scatter</summary>

| CP output (head 4) | NNN output (head 4) |
| ------------------ | ------------------- |
| -30                | -25                 |
| -20                | -15                 |
| -10                | -5                  |
| 0                  | 0                   |
| 10                 | 5                   |
| 20                 | 10                  |
| 30                 | 15                  |
</details>

![](images/9f1e1248e7fea585ca8cbeee28c91d76322f0cafa327e34cb3f0e1f7f634291f.jpg)

<details>
<summary>scatter</summary>

| GP output (head s) | NNN output (head s) |
| ------------------ | ------------------- |
| -60                | -40                 |
| -40                | -20                 |
| -20                | 0                   |
| 0                  | 10                  |
| 20                 | 20                  |
| 40                 | 30                  |
</details>

![](images/12144edc0df8d5c3b3f95e83fea42c1ce4ad0f8b444bcf1da95fcf0c88cd1f05.jpg)

<details>
<summary>scatter</summary>

| GP output (head 5) | MNN output (head 6) |
| ------------------ | ------------------- |
| -30                | -25                 |
| -20                | -15                 |
| -10                | -5                  |
| 0                  | 0                   |
| 10                 | 5                   |
| 20                 | 10                  |
</details>

![](images/ac934ec149c639553600200fc0f73e1f1af944011be34193b9295f2700e043a0.jpg)

<details>
<summary>scatter</summary>

| GP output (head 1) | ANN output (head 1) |
| ------------------ | ------------------- |
| -60                | -40                 |
| -40                | -20                 |
| -20                | 0                   |
| 0                  | 10                  |
| 20                 | 30                  |
</details>

![](images/39794147348581c26e0c350f4e84ef619410453af1247e36f66bb2b50167e514.jpg)

<details>
<summary>scatter</summary>

| GP output (head B) | MNN output (head B) |
| ------------------ | ------------------- |
| -60                | -60                 |
| -40                | -40                 |
| -20                | -20                 |
| 0                  | 0                   |
| 20                 | 20                  |
| 40                 | 40                  |
</details>

![](images/3747241c214369a17c00b7cffa677ceda2b3231c6958e6b0a2f82e63573fccec.jpg)

<details>
<summary>scatter</summary>

| GP output (head 9) | MNN output (head 9) |
| ------------------ | ------------------- |
| -20                | -20                 |
| -10                | -10                 |
| 0                  | 0                   |
| 10                 | 10                  |
| 20                 | 15                  |
</details>

![](images/8ea70e6a622884bf3c5ce62d1de8817e044d8510a689a4159744a238663a0c8b.jpg)

<details>
<summary>scatter</summary>

| CP output (head 10) | ANN output (head 10) |
| ------------------- | -------------------- |
| -35                 | -30                  |
| -25                 | -20                  |
| -15                 | -10                  |
| -5                  | 0                    |
| 5                   | 5                    |
| 10                  | 10                   |
| 15                  | 15                   |
| 20                  | 20                   |
| 25                  | 25                   |
| 30                  | 30                   |
</details>

Fig. S4: Scatters for MNIST (classifier).

![](images/e245bc9a543585296db440045d0385aac0f2027b3acfff769d8f4602bd26cb51.jpg)  
Fig. S5: Scatters for MNIST (attention).

![](images/f1fdeebe8910ff7137ca3f0ea68b550c1dbca8fc6dd62c6575e14a8b6c572a31.jpg)

<details>
<summary>scatter</summary>

| GP output (head 1) | ANN output (head 1) |
| ------------------ | ------------------- |
| -75                | -90                 |
| -50                | -40                 |
| -25                | -20                 |
| 0                  | 0                   |
| 25                 | 25                  |
| 50                 | 50                  |
</details>

![](images/be9b498dffd96fbd97786004906989b8b473e300021584e82f9672495952382d.jpg)

<details>
<summary>scatter</summary>

| GP output (head 2) | ANN output (head 2) |
| ------------------ | ------------------- |
| -60                | -50                 |
| -40                | -30                 |
| -20                | -10                 |
| 0                  | 0                   |
| 20                 | 10                  |
</details>

![](images/8bdcad98abf0a8fe47c9a2ccc3bcaeaeda3605578cb54eea845232ca5919cff4.jpg)

<details>
<summary>scatter</summary>

| GP output (head 3) | ANN output (head 3) |
| ------------------ | ------------------- |
| -10                | -10                 |
| 0                  | 0                   |
| 10                 | 20                  |
| 15                 | 40                  |
</details>

![](images/4382bf9c046af53338d64248097f94110c98d9ea571fd5885b80c55dc2361c19.jpg)

<details>
<summary>scatter</summary>

| GP output (head 4) | ANN output (head 4) |
| ------------------ | ------------------- |
| -70                | -70                 |
| -60                | -65                 |
| -50                | -55                 |
| -40                | -45                 |
| -30                | -35                 |
| -20                | -25                 |
| -10                | -15                 |
| 0                  | -5                  |
| 10                 | 5                   |
| 20                 | 15                  |
| 30                 | 25                  |
| 40                 | 35                  |
</details>

![](images/080c6f94560ef3cf53e62e69545498a0c76dfb3925bc4f9e119aa3f66f383df3.jpg)

<details>
<summary>scatter</summary>

| GP output (head 5) | ANN output (head 5) |
| ------------------ | ------------------- |
| -75                | -100                |
| -25                | -40                 |
| 0                  | 0                   |
</details>

![](images/9040592f09d88dc03b8949921f4c715ed024ac3dcd2eecad32e255e16963f2a6.jpg)

<details>
<summary>scatter</summary>

| GP output (head 6) | ANN output (head 6) |
| ------------------ | ------------------- |
| -30                | -25                 |
| -20                | -15                 |
| -10                | -5                  |
| 0                  | 0                   |
| 10                 | 5                   |
| 20                 | 10                  |
| 30                 | 15                  |
| 40                 | 20                  |
| 50                 | 25                  |
| 60                 | 30                  |
</details>

![](images/71b5f06095a3c2fdec212e50dabe2e881b9790c6e83606d4f414c3bf3a3392d7.jpg)

<details>
<summary>scatter</summary>

| GP output (head 7) | ANN output (head 7) |
| ------------------ | ------------------- |
| -60                | -80                 |
| -20                | -40                 |
| 0                  | 0                   |
| 20                 | 10                  |
</details>

![](images/d899ccaa68ccd459ca87b7080e2ff3b4c249f748c7b4e4c3fe609b653ef0df3c.jpg)

<details>
<summary>scatter</summary>

| GP output (head 8) | ANN output (head 8) |
| ------------------ | ------------------- |
| -40                | -35                 |
| -30                | -25                 |
| -20                | -15                 |
| -10                | -5                  |
| 0                  | 0                   |
| 10                 | 5                   |
| 20                 | 10                  |
</details>

![](images/b1b2c94a09db9a310114b4e76e886be0d56f55d89e30a3e9a167b7029d3edd46.jpg)

<details>
<summary>scatter</summary>

| GP output (head 9) | ANN output (head 9) |
| ------------------ | ------------------- |
| -50                | -50                 |
| -40                | -50                 |
| -30                | -25                 |
| -20                | -20                 |
| -10                | -10                 |
| 0                  | 0                   |
| 10                 | 10                  |
| 20                 | 10                  |
</details>

Fig. S6: Scatters for Kather dataset (classifier).

![](images/817330f6bb98dbddea6d5eaa40df444538df8fb63a402538c325e2142bd1754c.jpg)  
Fig. S7: Scatters for Kather dataset (attention).

![](images/f8ab5c461a2237021aea27899d06dd6d294d00d0a1f2382247a40dbf681a11c3.jpg)  
Fig. S8: Explanations for MNIST (set 1).

![](images/a965986e08ce1b7fbe5c1c0f6e32429ef33532c4e626bb1e32dcd86dd624ee2b.jpg)  
Fig. S9: Explanations for MNIST (set 2).

![](images/273e444baa708a91c371163637ad4559e066218afd9e20294d49c0ea4d9d46c1.jpg)  
Fig. S10: Explanations for MNIST (set 3).

![](images/74950c418b6b66a5fe9a2da68dba93e6a2cb7491cd0247f78cb3d53c43a8041c.jpg)  
Fig. S11: Explanations for MNIST (set 4).

![](images/6d07d7dc123b5f0909991068aea0fcbec7c340e2324aa821a0aa794bf500b9cc.jpg)  
Fig. S12: Explanations for MNIST (set 5).

![](images/01d113798766311c3230f11ad437a5f0db908f33460c0038b34d35addd3c04b9.jpg)  
Fig. S13: Explanations for MNIST (set 6).

![](images/2894d3c3b5eff8235ad4903be778284e412c97ad5d8d910fdaf208b38c64a584.jpg)  
Fig. S14: Explanations for MNIST (set 7).

![](images/e8e4c9935f5134144a21bacafdced8b705ca4a4a23c8be79e6246caa7e55dd06.jpg)  
Fig. S15: Explanations for MNIST (set 8).

![](images/a9fb8da57e268c58158f75827f5fad553c99f25d6313ab295e806b3e78896066.jpg)  
Fig. S16: Explanations for MNIST (set 9).

![](images/f132505f5346e540afd1e68e9ed9e982d1a33e64cb1227192f5ef7d56219b8f1.jpg)  
Fig. S17: Explanations for MNIST (set 10).

![](images/facf1ce482cc27ba63ad61773e5bb9c7e8ebde89eed9d88abbde414e7a9032f0.jpg)  
Fig. S18: Explanations for MNIST (set 11).

![](images/a84ba9620a4c557d4dbe4d73f02969eb0c4d83ef6637cd058fc2b83f13baf7ec.jpg)  
Fig. S19: Explanations for MNIST (set 12).

![](images/7e434d90ca927b4224dbd7c02fd4e17772c081db14741d5be921566fb847b428.jpg)  
Fig. S20: Explanations for Kather dataset (set 1).

![](images/47c29136602f7a89f4e16711d36fc85e8a1819f23786f7f0cbb0f0199350c379.jpg)  
Fig. S21: Explanations for Kather dataset (set 2).

![](images/7f8cf5f981946ab72051abac1024cac70ab7461cb7ae6552bed626c2272da593.jpg)  
Fig. S22: Explanations for Kather dataset (set 3).

![](images/b311982a499545ffba9e5942d19c2d4cc7cf0229060b45a2d5a4066cba2c0ca4.jpg)  
Fig. S23: Explanations for Kather dataset (set 4).

![](images/2f4bbb6e56323af304a9be325440405e64bbe1d690b9ab38a6cad650e98b5074.jpg)  
Fig. S24: Explanations for Kather dataset (set 5).

![](images/8dabea6e2c9439426b960cc13437cdec731652144850e454409a1c61debe1e18.jpg)  
Fig. S25: Explanations for DogsWolves (set 1).

![](images/228d6fc264d0804a3597b9e82ae00c27be2fcc6810c25368fbeb94b7200e8d74.jpg)  
Fig. S26: Explanations for DogsWolves (set 2).

![](images/746f54379559214833cb1fb4f48e4731a058364f81dd08f0713255c9b94071a9.jpg)  
Fig. S27: Explanations for DogsWolves (set 3).

![](images/62eed5640a34aa8fe36e8139266083e60247a84c167d562b53f27a92b58eedb8.jpg)  
Fig. S28: Explanations for DogsWolves (set 4).

![](images/291da8281a0a75d094261f3f90ef6d1db68d39d6a1c0a7f2a2acad0b76c949aa.jpg)  
Fig. S29: Explanations for DogsWolves (set 5).

![](images/fadffd06d4532431e5d39f194cbf06c6d5ed0c90f9cd1c11addb30bb323e52cb.jpg)  
Fig. S30: Explanations for DogsWolves (set 6).

![](images/546f1e6c025d2cae9f3642b39f46ae65bcc140a4ba9707ee9a1c7608ec69a004.jpg)  
Fig. S31: Explanations for DogsWolves (set 7).

![](images/6fec3e53fc53b5f9a707911780e4ca50f01d05f2b7397341ea6ba0a5b80fd77c.jpg)  
Fig. S32: Explanations for DogsWolves (set 8).

![](images/271e598c6316befc3480869d0cc98c5019f09ca3a727d5a55c6124e91475dc29.jpg)  
Fig. S33: Explanations for DogsWolves (set 9).

![](images/ac04e61a6f67529a2ba06e5a67f6ef3dbd78ad9c73d5927715d9a789fad5a6cd.jpg)  
Fig. S34: Explanations for DogsWolves (set 10).

![](images/55d631f7d20602b6793b9d0e26fcce65b856e4e8f46e668086386efd0f1c96a4.jpg)  
Fig. S35: Explanations for DogsWolves (set 11).

![](images/8155b79224fa95b4153aebff32e218de30e3c3aba625d0c89d05e640d7b070a5.jpg)  
Fig. S36: Explanations for Cifar10 (set 1).

![](images/627d55bd9c0c63c27728f320b4979640db321d28938e64818da985113898f29a.jpg)  
Fig. S37: Explanations for Cifar10 (set 2).

![](images/ea288032f4ac7de37bb875f05910faf3157f4f3ca5cb8d76d318d8383be492da.jpg)  
Fig. S38: Explanations for Cifar10 (set 3).

![](images/52d2eb5f9c4f04d71c4da2359a9e99fa99124eebaa144459f34177517eea7cf1.jpg)  
Fig. S39: Explanations for Cifar10 (set 4).

![](images/ecd3d3cd4d28ce1fb28735c3b2baad30e5c86d3b4eb45cb8b53b955b17d420b9.jpg)  
Fig. S40: Explanations for Cifar10 (set 5).

![](images/ff8a85a72c28baf45f8aa6b1ce853da90938e5a6cb38c16849d8e268c1b27446.jpg)  
Fig. S41: Explanations for Cifar10 (set 6).

![](images/5e8e31414a07f22c510c60a105076678c10664a7640f93be2eb95a5d233be664.jpg)  
Fig. S42: Explanations for Cifar10 (set 7).

![](images/ec99a03c90a9bbe02a41fc3879ecbc43caafae14852b88549ce3efc74c162184.jpg)  
Fig. S43: Explanations for Cifar10 (set 8).

![](images/12f2e1a5ef4e7c32012cb5f376c938c560b2c234cbaaf6782903fbdf7a94a077.jpg)  
Fig. S44: Explanations for Cifar10 (set 9).

![](images/1684d91af376525cc30430500f5e44c777c31e20ef49d747d741af071e3da0fb.jpg)  
Fig. S45: Explanations for Cifar10 (set 10).

![](images/ff464981e48b647fd94a7425f6c17d97b33bf143bb603deb023c0c98d46b887b.jpg)  
Fig. S46: Explanations for Cifar10 (set 11).

![](images/c45156de2da259eaab7c29d846cebd965a45b83c7ddcf0e70dc68fee95838f1f.jpg)  
Fig. S47: Explanations for Cifar10 (set 12).

![](images/1e2ad8371a8434380af9da4eb51f2e65f3ffb9e5446c39cf622e289d5732a8f5.jpg)  
Fig. S48: Explanations for Cifar10 (set 13).

![](images/8a1842c4021a2b81e251a4f0608cc26ddde5416f3fb8f21286a53f03e0353ca4.jpg)  
Fig. S49: Explanations for Cifar10 (set 14).

![](images/89292f3afa0331c5fd9d4ffd66d192271ead6a678a9fa1310ed5d0e955d4f9de.jpg)  
Fig. S50: Comparing the proposed GPEX with representer point selection [33] (set 1).

![](images/89c181c6d580c16d813953d4fb61772be2e8fcb3e61d8bccf65dc8dc024b6873.jpg)  
Fig. S51: Comparing the proposed GPEX with representer point selection [33] (set 2).

![](images/b31eb996827e411fc9f7c0f7c290f6c608fbc86f027ded24f7fb482c7c819615.jpg)  
Fig. S52: Comparing the proposed GPEX with representer point selection [33] (set 3).

![](images/4b8f2912e853388e9acfb29871960342170004edacdeb8fc4f51dbb1630aeaf1.jpg)  
Fig. S53: Comparing the proposed GPEX with representer point selection [33] (set 4).

![](images/e8b58e8c8d8567ed14135ba81453fed08656acb9b39a8efb0f52d7a81118d26a.jpg)  
Fig. S54: Comparing the proposed GPEX with representer point selection [33] (set 5).

![](images/89328a75293f46f6b628515a1046d16ad3d99e624d7419a940e9b16f0b340bdb.jpg)  
Fig. S55: Comparing the proposed GPEX with representer point selection [33] (set 6).

![](images/163f782de19dbcbdd36bea239a5aab78d81810e33462f5c9e153212880b47eca.jpg)  
Fig. S56: Comparing the proposed GPEX with representer point selection [33] (set 7).

![](images/cb1d2b2d943a3741da48cf0d520404c3bff20c27dc589ab9300909967a193127.jpg)  
Fig. S57: Comparing the proposed GPEX with representer point selection [33] (set 8).

![](images/b9e19c22bd2c029dc9d88d1151535e0aecd7d93790c72671343f714e2224b7d8.jpg)

<details>
<summary>line</summary>

| width of the second last layer of ANN | 10 epochs | 50 epochs | 100 epochs | 150 epochs | 200 epochs |
| ------------------------------------- | --------- | --------- | ---------- | ---------- | ---------- |
| 8                                     | 0.007     | 0.013     | 0.016      | 0.020      | 0.024      |
| 64                                    | 0.008     | 0.014     | 0.017      | 0.021      | 0.023      |
| 128                                   | 0.009     | 0.015     | 0.018      | 0.022      | 0.021      |
| 256                                   | 0.010     | 0.016     | 0.019      | 0.023      | 0.022      |
| 512                                   | 0.011     | 0.017     | 0.020      | 0.024      | 0.023      |
| 1024                                  | 0.012     | 0.018     | 0.021      | 0.025      | 0.026      |
| 2048                                  | 0.013     | 0.019     | 0.022      | 0.026      | 0.024      |
</details>

head 1

![](images/312f89b0cf95dd70f54dda3092afe3b6e4ec47663277ff4b175f0dd52bbf1c05.jpg)

<details>
<summary>line</summary>

| width of the second last layer of ANN | 10 epochs | 50 epochs | 100 epochs | 150 epochs | 200 epochs |
| ------------------------------------- | --------- | --------- | ---------- | ---------- | ---------- |
| 8                                     | 0.006     | 0.009     | 0.008      | 0.011      | 0.014      |
| 64                                    | 0.005     | 0.007     | 0.007      | 0.008      | 0.013      |
| 128                                   | 0.005     | 0.006     | 0.006      | 0.007      | 0.011      |
| 256                                   | 0.005     | 0.007     | 0.005      | 0.008      | 0.011      |
| 512                                   | 0.004     | 0.010     | 0.011      | 0.011      | 0.014      |
| 1024                                  | 0.005     | 0.007     | 0.008      | 0.011      | 0.011      |
| 2488                                  | 0.003     | 0.006     | 0.006      | 0.011      | 0.014      |
</details>

head 2

![](images/75907165acb0e43941bf0b11944c06ccb7339ff05d3caff3c2e0e1136bc8a16d.jpg)

<details>
<summary>line</summary>

| width of the second last layer of ANN | 10 epochs | 50 epochs | 100 epochs | 150 epochs | 200 epochs |
| ------------------------------------- | --------- | --------- | ---------- | ---------- | ---------- |
| 8                                     | 0.010     | 0.015     | 0.015      | 0.022      | 0.030      |
| 64                                    | 0.011     | 0.015     | 0.015      | 0.024      | 0.030      |
| 128                                   | 0.012     | 0.015     | 0.015      | 0.024      | 0.030      |
| 256                                   | 0.011     | 0.019     | 0.015      | 0.024      | 0.030      |
| 512                                   | 0.011     | 0.018     | 0.027      | 0.022      | 0.030      |
| 1024                                  | 0.011     | 0.018     | 0.019      | 0.019      | 0.028      |
| 2048                                  | 0.011     | 0.017     | 0.019      | 0.022      | 0.028      |
</details>

head 3

![](images/683dbc220f4a61c0ace01b45052fd4cf1f4cd7c95c574dff943f254b0a7c9477.jpg)

<details>
<summary>line</summary>

| width of the second last layer of ANN | 10 epochs | 50 epochs | 100 epochs | 150 epochs | 200 epochs |
| ------------------------------------- | --------- | --------- | ---------- | ---------- | ---------- |
| 8                                     | 0.007     | 0.011     | 0.012      | 0.03       | 0.012      |
| 64                                    | 0.008     | 0.012     | 0.013      | 0.02       | 0.01        |
| 128                                   | 0.009     | 0.013     | 0.014      | 0.025      | 0.012      |
| 256                                   | 0.008     | 0.014     | 0.015      | 0.025      | 0.01       |
| 512                                   | 0.007     | 0.01          | 0.018      | 0.025      | 0.01       |
| 1024                                  | 0.008     | 0.02      | 0.018      | 0.025      | 0.012      |
| 2488                                  | 0.007     | 0.01      | 0.01       | 0.018      | 0.01       |
</details>

head 4

![](images/b67bfdd917bbda7a5c6b17da2a42b4883d80a2fc2d02e8e9e69d744f35640399.jpg)

<details>
<summary>line</summary>

| width of the second last layer of ANN | 10 epochs | 50 epochs | 100 epochs | 150 epochs | 200 epochs |
| ------------------------------------- | --------- | --------- | ---------- | ---------- | ---------- |
| 284                                   | 0.0055    | 0.0075    | 0.0100     | 0.0150     | 0.0200     |
| 64                                    | 0.0060    | 0.0125    | 0.0125     | 0.0175     | 0.0200     |
| 128                                   | 0.0075    | 0.0125    | 0.0125     | 0.0150     | 0.0200     |
| 266                                   | 0.0075    | 0.0125    | 0.0125     | 0.0175     | 0.0175     |
| 512                                   | 0.0075    | 0.0125    | 0.0125     | 0.0175     | 0.0150     |
| 1024                                  | 0.0075    | 0.0125    | 0.0125     | 0.0175     | 0.0175     |
| 2488                                  | 0.0075    | 0.0125    | 0.0125     | 0.0175     | 0.0175     |
</details>

head 5

![](images/37ad29ec25c828d531ec5ff3c8d8e7bdc05b4efa75cfe1b2c11b1fc01dad24b3.jpg)

<details>
<summary>line</summary>

| width of the second last layer of ANN | 10 epochs | 50 epochs | 100 epochs | 150 epochs | 200 epochs |
| ------------------------------------- | --------- | --------- | ---------- | ---------- | ---------- |
| 8                                     | 0.006     | 0.010     | 0.015      | 0.022      | 0.024      |
| 64                                    | 0.007     | 0.011     | 0.014      | 0.018      | 0.025      |
| 128                                   | 0.008     | 0.012     | 0.013      | 0.019      | 0.026      |
| 256                                   | 0.009     | 0.013     | 0.012      | 0.018      | 0.027      |
| 512                                   | 0.010     | 0.014     | 0.013      | 0.016      | 0.022      |
| 1024                                  | 0.011     | 0.015     | 0.014      | 0.019      | 0.026      |
| 2048                                  | 0.012     | 0.016     | 0.015      | 0.014      | 0.023      |
</details>

head 6

![](images/f5c8c5046532efae54cf55cbdb3aec0adac83181bfa9c804eeed469d20f7b8d7.jpg)

<details>
<summary>line</summary>

| width of the second last layer of ANN | 10 epochs | 50 epochs | 100 epochs | 150 epochs | 200 epochs |
| ------------------------------------- | --------- | --------- | ---------- | ---------- | ---------- |
| 84                                    | 0.0095    | 0.0095    | 0.0135     | 0.0135     | 0.0175     |
| 64                                    | 0.0075    | 0.0135    | 0.0135     | 0.0135     | 0.0175     |
| 128                                   | 0.0125    | 0.0135    | 0.0145     | 0.0175     | 0.0175     |
| 296                                   | 0.0125    | 0.0135    | 0.0135     | 0.0175     | 0.0175     |
| 512                                   | 0.0125    | 0.0145    | 0.0135     | 0.0135     | 0.0175     |
| 1024                                  | 0.0125    | 0.0145    | 0.0135     | 0.0135     | 0.0175     |
| 2048                                  | 0.0125    | 0.0145    | 0.0135     | 0.0135     | 0.0175     |
</details>

head 7

![](images/5f7a65688f8b988de0ba8567007dfe23f08e0b5fc2c9f1d668b7ed5e01695ecb.jpg)

<details>
<summary>line</summary>

| width of the second last layer of ANN | 10 epochs | 50 epochs | 100 epochs | 150 epochs | 200 epochs |
| ------------------------------------- | --------- | --------- | ---------- | ---------- | ---------- |
| 8                                     | 0.006     | 0.007     | 0.012      | 0.013      | 0.017      |
| 64                                    | 0.006     | 0.008     | 0.011      | 0.012      | 0.015      |
| 128                                   | 0.007     | 0.009     | 0.011      | 0.013      | 0.015      |
| 256                                   | 0.007     | 0.008     | 0.011      | 0.013      | 0.014      |
| 512                                   | 0.006     | 0.013     | 0.011      | 0.012      | 0.015      |
| 1024                                  | 0.006     | 0.013     | 0.011      | 0.012      | 0.013      |
| 2488                                  | 0.002     | 0.008     | 0.012      | 0.012      | 0.015      |
</details>

head 8

![](images/bfd09a7f92fa9280f415e0ab332bf41eb90458bcd9639428a9504f22a79621ec.jpg)

<details>
<summary>line</summary>

| width of the second last layer of ANN | 10 epochs | 50 epochs | 100 epochs | 150 epochs | 200 epochs |
| ------------------------------------- | --------- | --------- | ---------- | ---------- | ---------- |
| 8                                     | 0.007     | 0.008     | 0.010      | 0.012      | 0.015      |
| 64                                    | 0.005     | 0.009     | 0.009      | 0.011      | 0.013      |
| 128                                   | 0.006     | 0.010     | 0.011      | 0.012      | 0.014      |
| 256                                   | 0.005     | 0.011     | 0.013      | 0.013      | 0.015      |
| 512                                   | 0.004     | 0.012     | 0.016      | 0.012      | 0.015      |
| 1024                                  | 0.005     | 0.013     | 0.012      | 0.013      | 0.016      |
| 2048                                  | 0.004     | 0.014     | 0.012      | 0.012      | 0.016      |
</details>

head 9

![](images/be51ccb0392426e205b172d1b77e716b25c1fc3ae0e113a48b6bde0ce46f3d45.jpg)

<details>
<summary>line</summary>

| width of the second last layer of ANN | 10 epochs | 50 epochs | 100 epochs | 150 epochs | 200 epochs |
| ------------------------------------- | --------- | --------- | ---------- | ---------- | ---------- |
| 8                                     | 0.0045    | 0.0095    | 0.011      | 0.015      | 0.015      |
| 64                                    | 0.005     | 0.006     | 0.007      | 0.01       | 0.016      |
| 128                                   | 0.004     | 0.006     | 0.007      | 0.009      | 0.012      |
| 256                                   | 0.005     | 0.006     | 0.007      | 0.01       | 0.015      |
| 512                                   | 0.004     | 0.007     | 0.008      | 0.01       | 0.015      |
| 1024                                  | 0.008     | 0.01      | 0.008      | 0.01       | 0.013      |
| 2488                                  | 0.002     | 0.01      | 0.008      | 0.01       | 0.013      |
</details>

head 10   
Fig. S58: Parameter analysis of Sec.S7.

<table><tr><td></td><td>Cifar10 [15]</td><td>MNIST [6]</td><td>Kather [12]</td><td>DogsWolves [30]</td></tr><tr><td>ANN accuracy</td><td>95.43</td><td>99.56</td><td>96.80</td><td>80.50</td></tr><tr><td>GPs accuracy</td><td>92.26</td><td>99.41</td><td>93.60</td><td>78.75</td></tr></table>

TABLE S1: Accuracies of ANN classifiers versus the accuracies of the explainer GPs on four datasets.

![](images/ce6a84c7e8b21b26ba1e79721b55ec87144fdea555e7fc0e435429ac091b11e6.jpg)

Fig. S59: Comparing GP and ANN outputs for four batches of Cifar10 dataset [33]. The red rectangles highlight the instnaces for which the predictions of GP and ANN (i.e. the class with maximum score) are different.   
![](images/c321635017527d02e0c319af0f348afa378ef2318e17fa466b68ef3c7a47ac75.jpg)

<details>
<summary>heatmap</summary>

| GPs output (batchsize x D) | ANN output (batchsize x D) |
| :--- | :--- |
| -28 | -30 |
| -27 | -29 |
| -26 | -28 |
| -25 | -27 |
| -24 | -26 |
| -23 | -25 |
| -22 | -24 |
| -21 | -23 |
| -20 | -22 |
| -19 | -21 |
| -18 | -20 |
| -17 | -19 |
| -16 | -18 |
| -15 | -17 |
| -14 | -16 |
| -13 | -15 |
| -12 | -14 |
| -11 | -13 |
| -10 | -12 |
| -9 | -11 |
| -8 | -10 |
| -7 | -9 |
| -6 | -8 |
| -5 | -7 |
| -4 | -6 |
| -3 | -5 |
| -2 | -4 |
| -1 | -3 |
| 0 | -2 |
| 1 | -1 |
| 2 | 0 |
| 3 | 1 |
| 4 | 2 |
| 5 | 3 |
| 6 | 4 |
| 7 | 5 |
| 8 | 6 |
| 9 | 7 |
| 10 | 8 |
| 11 | 9 |
| 12 | 10 |
| 13 | 11 |
| 14 | 12 |
| 15 | 13 |
| 16 | 14 |
| 17 | 15 |
| 18 | 16 |
| 19 | 17 |
| 20 | 18 |
| 21 | 19 |
| 22 | 20 |
| 23 | 21 |
| 24 | 22 |
| 25 | 23 |
| 26 | 24 |
| 27 | 25 |
| 28 | 26 |
| 29 | 27 |
| 30 | 28 |
| 31 | 29 |
| 32 | 30 |
| 33 | 31 |
| 34 | 32 |
| 35 | 33 |
| 36 | 34 |
| 37 | 35 |
| 38 | 36 |
| 39 | 37 |
| 40 | 38 |
| 41 | 39 |
| 42 | 40 |
| 43 | 41 |
| 44 | 42 |
| 45 | 43 |
| 46 | 44 |
| 47 | 45 |
| 48 | 46 |
| 49 | 47 |
| 50 | 48 |
| 51 | 49 |
| 52 | 50 |
| 53 | 51 |
| 54 | 52 |
| 55 | 53 |
| 56 | 54 |
| 57 | 55 |
| 58 | 56 |
| 59 | 57 |
| 60 | 58 |
| 61 | 59 |
| 62 | 60 |
| 63 | 61 |
| 64 | 62 |
| 65 | 63 |
| 66 | 64 |
| 67 | 65 |
| 68 | 66 |
| 69 | 67 |
| 70 | 68 |
| 71 | 69 |
| 72 | 70 |
| 73 | 71 |
| 74 | 72 |
| 75 | 73 |
| 76 | 74 |
| 77 | 75 |
| 78 | 76 |
| 79 | 77 |
| 80 | 78 |
| 81 | 79 |
| 82 | 80 |
| 83 | 81 |
| 84 | 82 |
| 85 | 83 |
| 86 | 84 |
| 87 | 85 |
| 88 | 86 |
| 89 | 87 |
| 90 | 88 |
| 91 | 89 |
| 92 | 90 |
| 93 | 91 |
| 94 | 92 |
| 95 | 93 |
| 96 | 94 |
| 97 | 95 |
| 98 | 96 |
| 99 | 97 |
| Note: The heatmap values are estimated based on the grid layout of the data. The color scale ranges from dark purple (low) to deep red (high). There is only one data point in the heatmap.
</details>

batch 1

![](images/15428f15ec21de4c79ed114a787ca6ee22d0ff560fe7f4d8b10a6877e477f839.jpg)

<details>
<summary>heatmap</summary>

| GPs output (batchsize x D) | ANN output (batchsize x D) |
| -------------------------- | -------------------------- |
| -10                        | -10                        |
| -5                         | -5                         |
| 0                          | 0                          |
| 5                          | 5                          |
| 10                         | 10                         |
| 15                         | 15                         |
| 20                         | 20                         |
| 25                         | 25                         |
| 30                         | 30                         |
</details>

batch 2

![](images/69673d2e3444431ebad4f27e8ad29fa7ada59a401c4dfc737461d434dca3a800.jpg)

<details>
<summary>heatmap</summary>

| GPs output (batchsize x D) | ANN output (batchsize x D) |
| -------------------------- | -------------------------- |
| -30                        | -30                        |
| -20                        | -20                        |
| -10                        | -10                        |
| 0                          | 0                          |
| 10                         | 10                         |
| 20                         | 20                         |
| 30                         | 30                         |
</details>

batch 3

![](images/06e9487d68060d7cc8de915102500afba286592add325ddcf5f5417a68edfad1.jpg)

<details>
<summary>heatmap</summary>

| GPs output (batchsize x D) | ANN output (batchsize x D) |
| -------------------------- | ------------------------- |
| -30                        | -30                       |
| -29                        | -29                       |
| -28                        | -28                       |
| -27                        | -27                       |
| -26                        | -26                       |
| -25                        | -25                       |
| -24                        | -24                       |
| -23                        | -23                       |
| -22                        | -22                       |
| -21                        | -21                       |
| -20                        | -20                       |
| -19                        | -19                       |
| -18                        | -18                       |
| -17                        | -17                       |
| -16                        | -16                       |
| -15                        | -15                       |
| -14                        | -14                       |
| -13                        | -13                       |
| -12                        | -12                       |
| -11                        | -11                       |
| -10                        | -10                       |
| -9                         | -9                        |
| -8                         | -8                        |
| -7                         | -7                        |
| -6                         | -6                        |
| -5                         | -5                        |
| -4                         | -4                        |
| -3                         | -3                        |
| -2                         | -2                        |
| -1                         | -1                        |
| 0                          | 0                         |
| 1                          | 1                         |
| 2                          | 2                         |
| 3                          | 3                         |
| 4                          | 4                         |
| 5                          | 5                         |
| 6                          | 6                         |
| 7                          | 7                         |
| 8                          | 8                         |
| 9                          | 9                         |
| 10                         | 10                        |
| 11                         | 11                        |
| 12                         | 12                        |
| 13                         | 13                        |
| 14                         | 14                        |
| 15                         | 15                        |
| 16                         | 16                        |
| 17                         | 17                        |
| 18                         | 18                        |
| 19                         | 19                        |
| 20                         | 20                        |
| 21                         | 21                        |
| 22                         | 22                        |
| 23                         | 23                        |
| 24                         | 24                        |
| 25                         | 25                        |
| 26                         | 26                        |
| 27                         | 27                        |
| 28                         | 28                        |
| 29                         | 29                        |
| 30                         | 30                        |
</details>

batch 4   
Fig. S60: Comparing GP and ANN outputs for four batches of MNIST dataset [32]. The red rectangles highlight the instnaces for which the predictions of GP and ANN (i.e. the class with maximum score) are different.

![](images/5a8e23ad7a5d77e6729699e546adda51006e2b742db8eb781de803b88c38bfa7.jpg)

<details>
<summary>heatmap</summary>

| GPs output (batchsize x D) | ANN output (batchsize x D) |
| -------------------------- | -------------------------- |
| -32.5                      | -7.5                       |
| -30.0                      | -5.0                       |
| -28.5                      | -2.5                       |
| -27.0                      | 0.0                        |
| -25.5                      | 2.5                        |
| -24.0                      | 5.0                        |
| -22.5                      | 7.5                        |
| -21.0                      | 10.0                       |
| -19.5                      | 12.5                       |
| -18.0                      | 15.0                       |
| -16.5                      | 17.5                       |
| -15.0                      | 20.0                       |
| -13.5                      | 22.5                       |
| -12.0                      | 25.0                       |
| -10.5                      | 27.5                       |
| -9.0                       | 30.0                       |
| -7.5                       | 32.5                       |
| -6.0                       | 35.0                       |
| -4.5                       | 37.5                       |
| -3.0                       | 40.0                       |
| -1.5                       | 42.5                       |
| 0.0                        | 45.0                       |
| 1.5                        | 47.5                       |
| 3.0                        | 50.0                       |
| 4.5                        | 52.5                       |
| 6.0                        | 55.0                       |
| 7.5                        | 57.5                       |
| 9.0                        | 60.0                       |
| 10.5                       | 62.5                       |
| 12.0                       | 65.0                       |
| 13.5                       | 67.5                       |
| 15.0                       | 70.0                       |
| 16.5                       | 72.5                       |
| 18.0                       | 75.0                       |
| 19.5                       | 77.5                       |
| 21.0                       | 80.0                       |
| 22.5                       | 82.5                       |
| 24.0                       | 85.0                       |
| 25.5                       | 87.5                       |
| 27.0                       | 90.0                       |
| 28.5                       | 92.5                       |
| 30.0                       | 95.0                       |
| 31.5                       | 97.5                       |
| 33.0                       | 100.0                      |
| 34.5                       | 102.5                      |
| 36.0                       | 105.0                      |
| 37.5                       | 107.5                      |
| 39.0                       | 110.0                      |
| 40.5                       | 112.5                      |
| 42.0                       | 115.0                      |
| 43.5                       | 117.5                      |
| 45.0                       | 120.0                      |
| 46.5                       | 122.5                      |
| 48.0                       | 125.0                      |
| 49.5                       | 127.5                      |
| 51.0                       | 130.0                      |
| 52.5                       | 132.5                      |
| 54.0                       | 135.0                      |
| 55.5                       | 137.5                      |
| 57.0                       | 140.0                      |
| 58.5                       | 142.5                      |
| 60.0                       | 145.0                      |
| 61.5                       | 147.5                      |
| 63.0                       | 150.0                      |
| 64.5                       | 152.5                      |
| 66.0                       | 155.0                      |
| 67.5                       | 157.5                      |
| 69.0                       | 160.0                      |
| 70.5                       | 162.5                      |
| 72.0                       | 165.0                      |
| 73.5                       | 167.5                      |
| 75.0                       | 170.0                      |
| 76.5                       | 172.5                      |
| 78.0                       | 175.0                      |
| 79.5                       | 177.5                      |
| 81.0                       | 180.0                      |
| 82.5                       | 182.5                      |
| 84.0                       | 185.0                      |
| 85.5                       | 187.5                      |
| 87.0                       | 190.0                      |
| 88.5                       | 192.5                      |
| 90.0                       | 195.0                      |
| 91.5                       | 197.5                      |
| 93.0                       | 200.0                      |
| 94.5                       | 202.5                      |
| 96.0                       | 205.0                      |
| 97.5                       | 207.5                      |
| 99.0                       | 210.0                      |
| Note: The heatmap values are not explicitly labeled in the code image but correspond to the original data points used in the heatmap chart.
</details>

batch 1

![](images/ea2c2c167484363f930c3e836042f63eb0264971ee33b0577cc407cbffaaf125.jpg)

<details>
<summary>heatmap</summary>

| Batch Size | GPs Output (batchsize x D) | ANN Output (batchsize x D) |
| ---------- | -------------------------- | -------------------------- |
| 1          | -3                         | -3                         |
| 1          | -2                         | -2                         |
| 1          | -1                         | -1                         |
| 1          | 0                          | 0                          |
| 1          | 1                          | 1                          |
| 1          | 2                          | 2                          |
| 1          | 3                          | 3                          |
| 1          | 4                          | 4                          |
| 1          | 5                          | 5                          |
| 1          | 6                          | 6                          |
| 1          | 7                          | 7                          |
| 1          | 8                          | 8                          |
| 1          | 9                          | 9                          |
| 1          | 10                         | 10                         |
| 1          | 11                         | 11                         |
| 1          | 12                         | 12                         |
| 1          | 13                         | 13                         |
| 1          | 14                         | 14                         |
| 1          | 15                         | 15                         |
| 2          | -3                         | -3                         |
| 2          | -2                         | -2                         |
| 2          | -1                         | -1                         |
| 2          | 0                          | 0                          |
| 2          | 1                          | 1                          |
| 2          | 2                          | 2                          |
| 2          | 3                          | 3                          |
| 2          | 4                          | 4                          |
| 2          | 5                          | 5                          |
| 2          | 6                          | 6                          |
| 2          | 7                          | 7                          |
| 2          | 8                          | 8                          |
| 2          | 9                          | 9                          |
| 2          | 10                         | 10                         |
| 2          | 11                         | 11                         |
| 2          | 12                         | 12                         |
| 2          | 13                         | 13                         |
| 2          | 14                         | 14                         |
| 2          | 15                         | 15                         |
| 2          | 16                         | 16                         |
| 2          | 17                         | 17                         |
| 2          | 18                         | 18                         |
| 2          | 19                         | 19                         |
| 2          | 20                         | 20                         |
| 2          | 21                         | 21                         |
| 2          | 22                         | 22                         |
| 2          | 23                         | 23                         |
| 2          | 24                         | 24                         |
| 2          | 25                         | 25                         |
| 2          | 26                         | 26                         |
| 2          | 27                         | 27                         |
| 2          | 28                         | 28                         |
| 2          | 29                         | 29                         |
| 2          | 30                         | 30                         |
| 2          | 31                         | 31                         |
| 2          | 32                         | 32                         |
| 2          | 33                         | 33                         |
| 2          | 34                         | 34                         |
| 2          | 35                         | 35                         |
| 2          | 36                         | 36                         |
| 2          | 37                         | 37                         |
| 2          | 38                         | 38                         |
| 2          | 39                         | 39                         |
| 2          | 40                         | 40                         |
| 2          | 41                         | 41                         |
| 2          | 42                         | 42                         |
| 2          | 43                         | 43                         |
| 2          | 44                         | 44                         |
| 2          | 45                         | 45                         |
| 2          | 46                         | 46                         |
| 2          | 47                         | 47                         |
| 2          | 48                         | 48                         |
| 2          | 49                         | 49                         |
| 2          | 50                         | 50                         |
| ...        |                             |                            |
| ...        |                             |                            |
| ...        |                             |                            |
| ...        |                             |                            |
| ...        |                             |                            |
| ...        |                             |                            |
| ...        |                             |                            |
| ...        |                             |                            |
| ...        |                             |                            |
| ...        |                             |                            |
| ...        |                             |                            |
| ...        |                             |                            |
| ...        |                             |                            |
| ...    |                             |                            |
| ...        |                             |                            |
| ...        |                             |                            |
| ...        |                             |                            |
| ...        |                             |                            |
| ...        |                             |                            |
| ...        |                             |                            |
| ...        |                             |                            |
| ...        |                             |                            |
| ...        |                             |                            |
| ...        |                             |                            |
| ...        |                             |                            |
| ...        |                             | C: ~0.5, ~0.6, ~0.7, ~0.8, ~0.9, ~1.0, ~1.1, ~1.2, ~1.3, ~1.4, ~1.5, ~1.6, ~1.7, ~1.8, ~1.9, ~2.0, ~2.1, ~2.2, ~2.3, ~2.4, ~2.5, ~2.6, ~2.7, ~2.8, ~2.9, ~3.0, ~3.1, ~3.2, ~3.3, ~3.4, ~3.5, ~3.6, ~3.7, ~3.8, ~3.9, ~4.0, ~4.1, ~4.2, ~4.3, ~4.4, ~4.5, ~4.6, ~4.7, ~4.8, ~4.9, ~5.0, ~5.1, ~5.2, ~5.3, ~5.4, ~5.5, ~5.6, ~5.7, ~5.8, ~5.9, ~6.0, ~6.1, ~6.2, ~6.3, ~6.4, ~6.5, ~6.6, ~6.7, ~6.8, ~6.9, ~7.0, ~7.1, ~7.2, ~7.3, ~7.4, ~7.5, ~7.6, ~7.7, ~7.8, ~7.9, ~8.0, ~8.1, ~8.2, ~8.3, ~8.4, ~8.5, ~8.6, ~8.7, ~8.8, ~8.9, ~9.0, ~9.1, ~9.2, ~9.3, ~9.4, ~9.5, ~9.6, ~9.7, ~9.8, ~9.9, ~10.0, ~9.9, ~100     .      .            .              .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .             .           .              ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..            ..           ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..            ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..              ..             :                      |
</details>

batch 2

![](images/73b1531367034c7a004dbb70ddd6cbbd3a639087e06f85c77aec78281a1a089b.jpg)

<details>
<summary>heatmap</summary>

| Batch Size | GPs Output (batchsize x D) | ANN Output (batchsize x D) |
| ---------- | -------------------------- | -------------------------- |
| 1          | -25                        | -5                         |
| 1          | -10                        | 0                          |
| 1          | 0                          | 5                          |
| 1          | 10                         | 10                         |
| 1          | 20                         | 5                          |
| 1          | 30                         | 0                          |
| 1          | 40                         | -5                         |
| 1          | 50                         | -10                        |
| 1          | 60                         | -5                         |
| 1          | 70                         | 0                          |
| 1          | 80                         | 5                          |
| 1          | 90                         | 10                         |
| 1          | 100                        | 5                          |
| 2          | -25                        | -5                         |
| 2          | -10                        | 0                          |
| 2          | 0                          | 5                          |
| 2          | 10                         | 10                         |
| 2          | 20                         | 5                          |
| 2          | 30                         | 0                          |
| 2          | 40                         | -5                         |
| 2          | 50                         | -10                        |
| 2          | 60                         | -5                         |
| 2          | 70                         | 0                          |
| 2          | 80                         | 5                          |
| 2          | 90                         | 10                         |
| 2          | 100                        | 5                          |
| 2          | 110                        | 0                          |
| 2          | 120                        | -5                         |
| 2          | 130                        | -10                        |
| 2          | 140                        | -5                         |
| 2          | 150                        | 0                          |
| 2          | 160                        | 5                          |
| 2          | 170                        | 10                         |
| 2          | 180                        | 5                          |
| 2          | 190                        | 0                          |
| 2          | 200                        | -5                         |
| 3          | -25                        | -5                         |
| 3          | -10                        | 0                          |
| 3          | 0                          | 5                          |
| 3          | 10                         | 10                         |
| 3          | 20                         | 5                          |
| 3          | 30                         | 0                          |
| 3          | 40                         | -5                         |
| 3          | 50                         | -10                        |
| 3          | 60                         | -5                         |
| 3          | 70                         | 0                          |
| 3          | 80                         | 5                          |
| 3          | 90                         | 10                         |
| 3          | 100                        | 5                          |
| 3          | 110                        | 0                          |
| 3          | 120                        | -5                         |
| 3          | 130                        | -10                        |
| 3          | 140                        | -5                         |
| 3          | 150                        | 0                          |
| 3          | 160                        | 5                          |
| 3          | 170                        | 10                         |
| 3          | 180                        | 5                          |
| 3          | 190                        | 0                          |
| 3          | 200                        | -5                         |
| ...        | ...                        | ...                        |
| ...        | ...                        | ...                        |
| ...        | ...                        | ...                        |
| ...        | ...                        | ...                        |
| ...        | ...                        | ...                        |
| ...        | ...                        | ...                        |
| ...        | ...                        | ...                        |
| ...        | ...                        | ...                        |
| ...        | ...                        | ...                        |
| ...        | ...                        | ...                        |
| ...        | ...                        | ... (with label "ANN output (batchsize x D)" in the heatmap) |
| ...        | ...                        | ... (with label "ANN output (batchsize x D)" in the heatmap) |
| ...        | ...                        | ... (with label "ANN output (batchsize x D)" in the heatmap) |
| ...        | ...                        | ... (with label "ANN output (batchsize x D)" in the heatmap) |
| ...        | ...                        | ... (with label "ANN output (batchsize x D)" in the heatmap)
</details>

batch 3

![](images/bb7007bb2afce6ece21b6f758e14a1626c06c139d17ec011a0a3ec1a7affa420.jpg)

<details>
<summary>heatmap</summary>

| Batch Size | GPS Output (batchsize x D) | ANN Output (batchsize x D) |
| ---------- | -------------------------- | -------------------------- |
| 1          | -5                         | -5                         |
| 2          | -5                         | -5                         |
| 3          | -5                         | -5                         |
| 4          | -5                         | -5                         |
| 5          | -5                         | -5                         |
| 6          | -5                         | -5                         |
| 7          | -5                         | -5                         |
| 8          | -5                         | -5                         |
| 9          | -5                         | -5                         |
| 10         | -5                         | -5                         |
| 11         | -5                         | -5                         |
| 12         | -5                         | -5                         |
| 13         | -5                         | -5                         |
| 14         | -5                         | -5                         |
| 15         | -5                         | -5                         |
| 16         | -5                         | -5                         |
| 17         | -5                         | -5                         |
| 18         | -5                         | -5                         |
| 19         | -5                         | -5                         |
| 20         | -5                         | -5                         |
| 21         | -5                         | -5                         |
| 22         | -5                         | -5                         |
| 23         | -5                         | -5                         |
| 24         | -5                         | -5                         |
| 25         | -5                         | -5                         |
| 26         | -5                         | -5                         |
| 27         | -5                         | -5                         |
| 28         | -5                         | -5                         |
| 29         | -5                         | -5                         |
| 30         | -5                         | -5                         |
| 31         | -5                         | -5                         |
| 32         | -5                         | -5                         |
| 33         | -5                         | -5                         |
| 34         | -5                         | -5                         |
| 35         | -5                         | -5                         |
| 36         | -5                         | -5                         |
| 37         | -5                         | -5                         |
| 38         | -5                         | -5                         |
| 39         | -5                         | -5                         |
| 40         | -5                         | -5                         |
| 41         | -5                         | -5                         |
| 42         | -5                         | -5                         |
| 43         | -5                         | -5                         |
| 44         | -5                         | -5                         |
| 45         | -5                         | -5                         |
| 46         | -5                         | -5                         |
| 47         | -5                         | -5                         |
| 48         | -5                         | -5                         |
| 49         | -5                         | -5                         |
| 50         | -5                         | -5                         |
| 51         | -5                         | -5                         |
| 52         | -5                         | -5                         |
| 53         | -5                         | -5                         |
| 54         | -5                         | -5                         |
| 55         | -5                         | -5                         |
| 56         | -5                         | -5                         |
| 57         | -5                         | -5                         |
| 58         | -5                         | -5                         |
| 59         | -5                         | -5                         |
| 60         | -5                         | -5                         |
| 61         | -5                         | -5                         |
| 62         | -5                         | -5                         |
| 63         | -5                         | -5                         |
| 64         | -5                         | -5                         |
| 65         | -5                         | -5                         |
| 66         | -5                         | -5                         |
| 67         | -5                         | -5                         |
| 68         | -5                         | -5                         |
| 69         | -5                         | -5                         |
| 70         | -5                         | -5                         |
| 71         | -5                         | -5                         |
| 72         | -5                         | -5                         |
| 73         | -5                         | -5                         |
| 74         | -5                         | -5                         |
| 75         | -5                         | -5                         |
| 76         | -5                         | -5                         |
| 77         | -5                         | -5                         |
| 78         | -5                         | -5                         |
| 79         | -5                         | -5                         |
| 80         | -5                         | -5                         |
| 81         | -5                         | -5                         |
| 82         | -5                         | -5                         |
| 83         | -5                         | -5                         |
| 84         | -5                         | -5                         |
| 85         | -5                         | -5                         |
| 86         | -5                         | -5                         |
| 87         | -5                         | -5                         |
| 88         | -5                         | -5                         |
| 89         | -5                         | -5                         |
| 90         | -5                         | -5                         |
| 91         | -5                         | -5                         |
| 92         | -5                         | -5                         |
| 93         | -5                         | -5                         |
| 94         | -5                         | -5                         |
| 95         | -5                         | -5                         |
| 96         | -5                         | -5                         |
| 97         | -5                         | -5                         |
| 98         | -5                         | -5                         |
| 99         | -5                         | -5                         |
| 100        | -10                        | 10                        |
| 101        | 10                        | 10                        |
| 102        | 10                        | 10                        |
| 103        | 10                        | 10                        |
| 104        | 10                        | 10                        |
| 105        | 10                        | 10                        |
| 106        | 10                        | 10                        |
| 107        | 10                        | 10                        |
| 108        | 10                        | 10                        |
| 109        | 10                        | 10                        |
| 110        | 10                        | 10                        |
| 111        | 10                        | 10                        |
| 112        | 10                        | 10                        |
| 113        | 10                        | 10                        |
| 114        | 10                        | 10                        |
| 115        | 10                        | 10                        |
| 116        | 10                        | 10                        |
| 117        | 10                        | 10                        |
| 118        | 10                        | 10                        |
| 119        | 10                        | 10                        |
| 120        | 10                        | 10                        |
| 121        | 10                        | 10                        |
| 122        | 10                        | 10                        |
| 123        | 10                        | 10                        |
| 124        | 10                        | 10                        |
| 125        | 10                        | 10                        |
| 126        | 10                        | 10                        |
| 127        | 10                        | 10                        |
| 128        | 10                        | 10                        |
| 129        | 10                        | 10                        |
| 130        | 10                        | 10                        |
| 131        | 10                        | 10                        |
| 132        | 10                        | 10                        |
| 133        | 10                        | 10                        |
| 134        | 10                        | 10                        |
| 135        | 10                        | 10                        |
| 136        | 10                        | 10                        |
| 137        | 10                        | 10                        |
| 138        | 10                        | 10                        |
| 139        | 10                        | 10                        |
| 140        | nan                          | nan                        |
The heatmap visualizes the distribution of GPS and ANN outputs based on batch size x D. Values are estimated based on the chart type. The data is presented in a grid format with rows and columns specified in the code. The color scale ranges from dark blue (low) to light blue (high).
</details>

batch 4   
Fig. S61: Comparing GP and ANN outputs for four batches of Kather dataset [34]. The red rectangles highlight the instnaces for which the predictions of GP and ANN (i.e. the class with maximum score) are different.

![](images/42626e5b2ba0e0e71a11c9b50d514da67038fe3f86356c31fe2d0d5ba8220b15.jpg)

Fig. S62: Comparing GP and ANN outputs for four batches of DogsWolves dataset [35]. The red rectangles highlight the instnaces for which the predictions of GP and ANN (i.e. the class with maximum score) are different.   
![](images/4919a0b7576cd0812dc250979969ce5031a0d861f8c04042933f8f53ae892776.jpg)

Fig. S63: Comparing GP and ANN (attention submodule) outputs for 3 batches of Cifar10 dataset [15].   
![](images/8f94f2f505328a3cbe58d36a409bcc4ad79cc16a4df2dd68556d6a25a66f69df.jpg)

Fig. S64: Comparing GP and ANN (attention submodule) outputs for 3 batches of MNIST dataset [6].   
![](images/96db3bff0829432c13ef1a337c8dfeb96c0be9816683973df9cd18d90bad8826.jpg)  
Fig. S65: Comparing GP and ANN (attention submodule) outputs for 3 batches of Kather dataset [12].

![](images/6e8f39a84210d43650bb313502ff2d32bd8be8864b25aa69584c5606b3d1e523.jpg)

<details>
<summary>scatter</summary>

| the size of the inducing dataset | correlation coefficient between GP and ANN | split |
| --- | --- | --- |
| 2^8 | ~0.75 | split 1 |
| 2^9 | ~0.85 | split 1 |
| 2^10 | ~0.90 | split 1 |
| 2^11 | ~0.92 | split 1 |
| 2^12 | ~0.94 | split 1 |
| 2^13 | ~0.95 | split 1 |
| 2^14 | ~0.96 | split 1 |
| 2^15 | ~0.97 | split 1 |
| 50000 | ~0.98 | split 1 |
| 2^8 | ~0.80 | split 2 |
| 2^9 | ~0.82 | split 2 |
| 2^10 | ~0.78 | split 2 |
| 2^11 | ~0.85 | split 2 |
| 2^12 | ~0.90 | split 2 |
| 2^13 | ~0.93 | split 2 |
| 2^14 | ~0.94 | split 2 |
| 2^15 | ~0.95 | split 2 |
| 50000 | ~0.96 | split 2 |
| 2^8 | ~0.85 | split 3 |
| 2^9 | ~0.88 | split 3 |
| 2^10 | ~0.91 | split 3 |
| 2^11 | ~0.93 | split 3 |
| 2^12 | ~0.95 | split 3 |
| 2^13 | ~0.96 | split 3 |
| 2^14 | ~0.97 | split 3 |
| 2^15 | ~0.98 | split 3 |
| 50000 | ~0.98 | split 3 |
| 2^8 | ~0.78 | split 4 |
| 2^9 | ~0.84 | split 4 |
| 2^10 | ~0.87 | split 4 |
| 2^11 | ~0.90 | split 4 |
| 2^12 | ~0.93 | split 4 |
| 2^13 | ~0.95 | split 4 |
| 2^14 | ~0.96 | split 4 |
| 2^15 | ~0.97 | split 4 |
| 50000 | ~0.97 | split 4 |
| 2^8 | ~0.76 | split 5 |
| 2^9 | ~0.83 | split 5 |
| 2^10 | ~0.86 | split 5 |
| 2^11 | ~0.88 | split 5 |
| 2^12 | ~0.91 | split 5 |
| 2^13 | ~0.93 | split 5 |
| 2^14 | ~0.94 | split 5 |
| 2^15 | ~0.95 | split 5 |
| 50000 | ~0.95 | split 5 |
</details>

Fig. S66: Analyzing the effect of the size of inducing dataset.

![](images/a88ec655cb568d997400755e74a36d16c3c9027b7c7804e04db5276940e1b1a7.jpg)

![](images/8f32fafe970f89be1808b27d63a259a518535f0edf3a219c9c1539c1dd13b39e.jpg)

<details>
<summary>boxplot</summary>

| Kather | AUC     |
|--------|---------|
| head 1 | 99.0    |
| head 2 | 98.8    |
| head 3 | 98.5    |
| head 4 | 98.7    |
| head 5 | 98.3    |
| head 6 | 99.1    |
| head 7 | 98.5    |
| head 8 | 97.0    |
| head 9 | 98.6    |
</details>

![](images/7c9e8bb07cecacbab0f0c1c9e31af581763997abdb229624a75a15ccb14fbb0d.jpg)

<details>
<summary>boxplot</summary>

| Head   | AUC Value |
|--------|-----------|
| head 1 | 97.0      |
| head 2 | 98.5      |
| head 3 | 96.0      |
| head 4 | 94.0      |
| head 5 | 96.5      |
| head 6 | 93.5      |
| head 7 | 97.0      |
| head 8 | 97.5      |
| head 9 | 97.0      |
| head 10| 97.5      |
</details>

Fig. S67: Correlation coefficients for 5 splits.