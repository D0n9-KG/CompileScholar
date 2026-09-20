# A Restoration Network as an Implicit Prior

Yuyang Hu $^{1}$ , Mauricio Delbracio $^{2}$ ,

Peyman Milanfar $^{2}$ , Ulugbek S. Kamilov $^{1,2}$

$^{1}$ Washington University in St. Louis, $^{2}$ Google Research

{h.yuyang, kamilov}@wustl.edu, {mdelbra,milanfar}@google.com

# Abstract

Image denoisers have been shown to be powerful priors for solving inverse problems in imaging. In this work, we introduce a generalization of these methods that allows any image restoration network to be used as an implicit prior. The proposed method uses priors specified by deep neural networks pre-trained as general restoration operators. The method provides a principled approach for adapting state-of-the-art restoration models for other inverse problems. Our theoretical result analyzes its convergence to a stationary point of a global functional associated with the restoration operator. Numerical results show that the method using a super-resolution prior achieves state-of-the-art performance both quantitatively and qualitatively. Overall, this work offers a step forward for solving inverse problems by enabling the use of powerful pre-trained restoration models as priors.

# 1 Introduction

Many problems in computational imaging, biomedical imaging, and computer vision can be formulated as inverse problems, where the goal is to recover a high-quality images from its low-quality observations. Imaging inverse problems are generally ill-posed, thus necessitating the use of prior models on the unknown images for accurate inference. While the literature on prior modeling of images is vast, current methods are primarily based on deep learning (DL), where a deep model is trained to map observations to images (Lucas et al., 2018; McCann et al., 2017; Ongie et al., 2020).

Image denoisers have become popular for specifying image priors for solving inverse problems (Venkatakrishnan et al., 2013; Romano et al., 2017; Kadkhodaie & Simoncelli, 2021; Kamilov et al., 2023). Pre-trained denoisers provide a convenient proxy for image priors that does not require the description of the full density of natural images. The combination of state-of-the-art (SOTA) deep denoisers with measurement models has been shown to be effective in a number of inverse problems, including image super-resolution, deblurring, inpainting, microscopy, and medical imaging (Metzler et al., 2018; Zhang et al., 2017b; Meinhardt et al., 2017; Dong et al., 2019; Zhang et al., 2019; Wei et al., 2020; Zhang et al., 2022) (see also the recent reviews Ahmad et al. (2020); Kamilov et al. (2023)). This success has led to active research on novel methods based on denoiser priors, their theoretical analyses, statistical interpretations, as well as connections to related approaches such as score matching and diffusion models (Chan et al., 2017; Romano et al., 2017; Buzzard et al., 2018; Reehorst & Schniter, 2019; Sun et al., 2019; Sun et al., 2019; Ryu et al., 2019; Xu et al., 2020; Liu et al., 2021; Cohen et al., 2021a; Hurault et al., 2022a,b; Laumont et al., 2022; Gan et al., 2023).

Despite the rich literature on the topic, the prior work has narrowly focused on leveraging the statistical properties of denoisers. There is little work on extending the formalism and theory to priors specified using other types of image restoration operators, such as, for example, deep image super-resolution models. Such extensions would enable new algorithms that can leverage SOTA pre-trained restoration networks for solving other inverse problems. In this paper, we address this gap by developing the Deep Restoration Priors (DRP) methodology that provides a principled approach for using restoration operators as priors. We show that when the restoration operator is a minimum mean-squared error (MMSE) estimator, DRP can be interpreted as minimizing a composite objective function that includes log of the density of the degraded

image as the regularizer. Our interpretation extends the recent formalism based on using MMSE denoisers as priors (Bigdeli et al., 2017; Xu et al., 2020; Kadkhodaie & Simoncelli, 2021; Laumont et al., 2022; Gan et al., 2023). We present a theoretical convergence analysis of DRP to a stationary point of the objective function under a set of clearly specified assumptions. We show the practical relevance of DRP by solving several inverse problems by using a super-resolution network as a prior. Our numerical results show the potential of DRP to adapt the super-resolution model to act as an effective prior that can outperform image denoisers. This work thus addresses a gap in the current literature by providing a new principled framework for using pre-trained restoration models as priors for inverse problems.

All proofs and some details that have been omitted for space appear in the appendix.

# 2 Background

Inverse Problems. Many imaging problems can be formulated as inverse problems that seek to recover an unknown image $x \in R^{n}$ from from its corrupted observation

$$
\boldsymbol {y} = \boldsymbol {A} \boldsymbol {x} + \boldsymbol {e}, \tag {1}
$$

where $A \in R^{m \times n}$ is a measurement operator and $e \in R^{m}$ is the noise. A common strategy for addressing inverse problems involves formulating them as an optimization problem

$$
\widehat {\boldsymbol {x}} \in \underset {\boldsymbol {x} \in \mathbb {R} ^ {n}} {\arg \min} f (\boldsymbol {x}) \quad \text { with } \quad f (\boldsymbol {x}) = g (\boldsymbol {x}) + h (\boldsymbol {x}), \tag {2}
$$

where g is the data-fidelity term that measures the fidelity to the observation y and h is the regularizer that incorporates prior knowledge on x. For example, common functionals in imaging inverse problems are the least-squares data-fidelity term $g(\boldsymbol{x}) = \frac{1}{2} \| \boldsymbol{A}\boldsymbol{x} - \boldsymbol{y} \|_{2}^{2}$ and the total variation (TV) regularizer $h(\boldsymbol{x}) = \tau \| \boldsymbol{D}\boldsymbol{x} \|_{1}$ , where D is the image gradient, and $\tau > 0$ a regularization parameter.

Deep Learning. DL is extensively used for solving imaging inverse problems (McCann et al., 2017; Lucas et al., 2018; Ongie et al., 2020). Instead of explicitly defining a regularizer, DL methods often train convolutional neural networks (CNNs) to map the observations to the desired images (Wang et al., 2016; Jin et al., 2017; Kang et al., 2017; Chen et al., 2017; Delbracio et al., 2021; Delbracio & Milanfar, 2023). Model-based DL (MBDL) is a widely-used sub-family of DL algorithms that integrate physical measurement models with priors specified using CNNs (see reviews by Ongie et al. (2020); Monga et al. (2021)). The literature of MBDL is vast, but some well-known examples include plug-and-play priors (PnP), regularization by denoising (RED), deep unfolding (DU), compressed sensing using generative models (CSGM), and deep equilibrium models (DEQ) (Bora et al., 2017; Romano et al., 2017; Zhang & Ghanem, 2018; Hauptmann et al., 2018; Gilton et al., 2021; Liu et al., 2022). These approaches come with different trade-offs in terms of imaging performance, computational and memory complexity, flexibility, need for supervision, and theoretical understanding.

Denoisers as Priors. PnP (Venkatakrishnan et al., 2013; Sreehari et al., 2016) is one of the most popular MBDL approaches for inverse problems based on using deep denoisers as imaging priors (see recent reviews by Ahmad et al. (2020); Kamilov et al. (2023)). For example, the proximal-gradient method variant of PnP can be written as (Hurault et al., 2022a)

$$
\boldsymbol {x} ^ {k} \leftarrow \operatorname{prox} _ {\gamma g} (\boldsymbol {z} ^ {k}) \quad \text {with} \quad \boldsymbol {z} ^ {k} \leftarrow \boldsymbol {x} ^ {k - 1} - \gamma \tau (\boldsymbol {x} ^ {k - 1} - D _ {\sigma} (\boldsymbol {x} ^ {k - 1})), \tag {3}
$$

where $D_{\sigma}$ is a denoiser with a parameter $\sigma > 0$ for controlling its strength, $\tau > 0$ is a regularization parameter, and $\gamma > 0$ is a step-size. The theoretical convergence of PnP methods has been established for convex functions g using monotone operator theory (Sreehari et al., 2016; Sun et al., 2019; Ryu et al., 2019), as well as for nonconvex functions based on interpreting the denoiser as a MMSE estimator (Xu et al., 2020) or ensuring that the term $(I - D_{\sigma})$ in (3) corresponds to a gradient $\nabla h$ of a function h parameterized by a deep neural network (Hurault et al., 2022a,b; Cohen et al., 2021a). Many variants of PnP have been developed over the past few years (Romano et al., 2017; Metzler et al., 2018; Zhang et al., 2017b; Meinhardt et al.,

2017; Dong et al., 2019; Zhang et al., 2019; Wei et al., 2020), which has motivated an extensive research on its theoretical properties (Chan et al., 2017; Buzzard et al., 2018; Ryu et al., 2019; Sun et al., 2019; Tirer & Giryes, 2019; Teodoro et al., 2019; Xu et al., 2020; Sun et al., 2021; Cohen et al., 2021b; Hurault et al., 2022a; Laumont et al., 2022; Hurault et al., 2022b; Gan et al., 2023).

This work is most related to two recent PnP-inspired methods using restoration operators instead of denoisers (Zhang et al., 2019; Liu et al., 2020). Deep plug-and-play super-resolution (DPSR) (Zhang et al., 2019) was proposed to perform image super-resolution under arbitrary blur kernels by using a bicubic super-resolver as a prior. Regularization by artifact removal (RARE) (Liu et al., 2020) was proposed to use CNNs pre-trained directly on subsampled and noisy Fourier data as priors for magnetic resonance imaging (MRI). These prior methods did not leverage statistical interpretations of the restoration operators to provide a theoretical analysis for the corresponding PnP variants.

It is also worth highlighting the work of Gribonval and colleagues on theoretically exploring the relationship between MMSE restoration operators and proximal operators (Gribonval, 2011; Gribonval & Machart, 2013; Gribonval & Nikolova, 2021). Some of the observations and intuition in that prior line of work is useful for the theoretical analysis of the proposed DRP methodology.

Our contribution. (1) Our first contribution is the new method DRP for solving inverse problems using the prior implicit in a pre-trained deep restoration network. Our method is as a major extension of recent methods (Bigdeli et al., 2017; Xu et al., 2020; Kadkhodaie & Simoncelli, 2021; Gan et al., 2023) from denoisers to more general restoration operators. (2) Our second contribution is a new theory that characterizes the solution and convergence of DRP under priors associated with the MMSE restoration operators. Our theory is general in the sense that it allows for nonsmooth data-fidelity terms and expansive restoration models. (3) Our third contribution is the implementation of DRP using the popular SwinIR (Liang et al., 2021) super-resolution model as a prior for two distinct inverse problems, namely deblurring and super-resolution. Our implementation that shows the potential of using restoration models to achieve SOTA performance.

# 3 Deep Restoration Prior

Image denoisers are currently extensively used as priors for solving inverse problems. We extend this approach by proposing the following method that uses a more general restoration operator.

Algorithm 1 Deep Restoration Priors (DRP)   
1: input: Initial value $x^{0} \in R^{n}$ and parameters $\gamma, \tau > 0$ 2: for $k = 1, 2, 3, \ldots$ do
3: $z^{k} \leftarrow x^{k-1} - \gamma \tau G(x^{k-1})$ where $G(x) := x - R(Hx)$ 4: $x^{k} \leftarrow \text{sprox}_{\gamma g}(z^{k})$ 5: end for

The prior in Algorithm 1 is implemented in Line 3 using a deep model $R : R^{p} \to R^{n}$ pre-trained to solve the following restoration problem

$$
\boldsymbol {s} = \mathbf {H} \boldsymbol {x} + \boldsymbol {n} \quad \text { with } \quad \boldsymbol {x} \sim p _ {\boldsymbol {x}}, \quad \boldsymbol {n} \sim \mathcal {N} (0, \sigma^ {2} \mathbf {I}), \tag {4}
$$

where $\mathbf{H} \in \mathbb{R}^{p \times n}$ is a degradation operator, such as blur or downscaling, and $\boldsymbol{n} \in \mathbb{R}^p$ is the additive white Gaussian noise (AWGN) of variance $\sigma^2$ . The density $p_{\boldsymbol{x}}$ is the prior distribution of the desired class of images. Note that the restoration problem (4) is only used for training $\mathsf{R}$ and doesn't have to correspond to the inverse problem in (1) we are seeking to solve. When $\mathbf{H} = \mathbf{I}$ , the restoration operator $\mathsf{R}$ reduces to an AWGN denoiser used in the traditional PnP methods (Romano et al., 2017; Kadkhodaie & Simoncelli, 2021; Hurault et al., 2022a). The goal of DRP is to leverage a pre-trained restoration network $\mathsf{R}$ to gain access to the prior.

The measurement consistency is implemented in Line 4 using the scaled proximal operator

$$
\operatorname{sprox} _ {\gamma g} (\boldsymbol {z}) := \operatorname{prox} _ {\gamma g} ^ {\mathbf {H} ^ {\top} \mathbf {H}} (\boldsymbol {z}) = \underset {\boldsymbol {x} \in \mathbb {R} ^ {n}} {\arg \min} \left\{\frac {1}{2} \| \boldsymbol {x} - \boldsymbol {z} \| _ {\mathbf {H} ^ {\top} \mathbf {H}} ^ {2} + \gamma g (\boldsymbol {x}) \right\}, \tag {5}
$$

where $\|v\|_{H^{T}H} := v^{T}H^{T}Hv$ denotes the weighted Euclidean seminorm of a vector v. When $H^{T}H$ is positive definite and g is convex, the functional being minimized in (5) is strictly convex, which directly implies that the solution is unique. On the other hand, when g is not convex or $H^{T}H$ is positive semidefinite, there might be multiple solutions and the scaled proximal operator simply returns one of the solutions. It is also worth noting that (5) has an efficient solution when g is the least-squares data-fidelity term (see for example the discussion in Kamilov et al. (2023) on efficient implementations of proximal operators of least-squares).

The fixed points of Algorithm 1 can be characterized for subdifferentiable g (see Chapter 3 in Beck (2017) for a discussion on subdifferentiability). When DRP converges, it converges to vectors $x^{*} \in R^{n}$ that satisfy (see formal analysis in Appendix A.1)

$$
\mathbf {0} \in \partial g (\boldsymbol {x} ^ {*}) + \tau \mathbf {H} ^ {\top} \mathbf {H G} (\boldsymbol {x} ^ {*}) \tag {6}
$$

where $\partial g$ is the subdifferential of g and G is defined in Line 3 of Algorithm 1. As discussed in the next section, under additional assumptions, one can associate the fixed points of DRP with the stationary points of a composite objective function $f = g + h$ for some regularizer h.

# 4 Convergence Analysis of DRP

In this section, we present a theoretical analysis of DRP. We first provide a more insightful interpretation of its solutions for restoration models that compute MMSE estimators of (4). We then discuss the convergence of the iterates generated by DRP. Our analysis will require several assumptions that act as sufficient conditions for our theoretical results.

We will consider restoration models that perform MMSE estimation of $x \in R^{n}$ for the problem (4)

$$
R (\boldsymbol {s}) = \mathbb {E} [ \boldsymbol {x} | \boldsymbol {s} ] = \int \boldsymbol {x} p _ {\boldsymbol {x} | \boldsymbol {s}} (\boldsymbol {x}; \boldsymbol {s}) d \boldsymbol {x} = \int \boldsymbol {x} \frac {p _ {\boldsymbol {s} | \boldsymbol {x}} (\boldsymbol {s} ; \boldsymbol {x}) p _ {\boldsymbol {x}} (\boldsymbol {x})}{p _ {\boldsymbol {s}} (\boldsymbol {s})} d \boldsymbol {x}. \tag {7}
$$

where we used the probability density of the observation $s \in R^{p}$

$$
p _ {\boldsymbol {s}} (\boldsymbol {s}) = \int p _ {\boldsymbol {s} | \boldsymbol {x}} (\boldsymbol {s}; \boldsymbol {x}) p _ {\boldsymbol {x}} (\boldsymbol {x}) \mathrm{d} \boldsymbol {x} = \int G _ {\sigma} (\boldsymbol {s} - \mathbf {H} \boldsymbol {x}) p _ {\boldsymbol {x}} (\boldsymbol {x}) \mathrm{d} \boldsymbol {x}. \tag {8}
$$

The function $G_{\sigma}$ in (8) denotes the Gaussian density function with the standard deviation $\sigma > 0$ .

Assumption 1. The prior density $p_{x}$ is non-degenerate over $R^{n}$ .

As a reminder, a probability density $p_{\pmb{x}}$ is degenerate over $\mathbb{R}^n$ , if it is supported on a space of lower dimensions than $n$ . Our goal is to establish an explicit link between the MMSE restoration operator (7) and the following regularizer

$$
h (\boldsymbol {x}) = - \tau \sigma^ {2} \log p _ {\boldsymbol {s}} (\mathbf {H} \boldsymbol {x}), \quad \boldsymbol {x} \in \mathbb {R} ^ {n}, \tag {9}
$$

where $\tau$ is the parameter in Algorithm 1, $p_{s}$ is the density of the observation (8), and $\sigma^{2}$ is the AWGN level used for training the restoration network. We adopt Assumption 1 to have a more intuitive mathematical exposition, but one can in principle generalize the link between MMSE operators and regularization beyond non-degenerate priors (Gribonval & Machart, 2013). It is also worth observing that the function h is infinitely continuously differentiable, since it is obtained by integrating $p_{x}$ with a Gaussian density $G_{\sigma}$ (Gribonval, 2011; Gribonval & Machart, 2013).

Assumption 2. The scaled proximal operator $\mathsf{sprox}_{\gamma g}$ is well-defined in the sense that there exists a solution to the problem (5) for any $z \in \mathbb{R}^n$ . The function $g$ is subdifferentiable over $\mathbb{R}^n$ .

This mild assumption is necessary for us to be able to run our method. There are multiple ways to ensure that the scaled proximal operator is well defined. For example, $sprox_{\gamma g}$ is always well-defined for any g that is proper, closed, and convex (Parikh & Boyd, 2014). This directly makes DRP applicable with the popular least-squares data-fidelity term $g(\boldsymbol{x}) = \frac{1}{2}\|\boldsymbol{y} - \boldsymbol{A}\boldsymbol{x}\|_{2}^{2}$ . One can relax the assumption of convexity by

considering $g$ that is proper, closed, and coercive, in which case $\mathsf{sprox}_{\gamma g}$ will have a solution (see for example Chapter 6 of Beck (2017)). Note that we do not require the solution to (5) to be unique; it is sufficient for $\mathsf{sprox}_{\gamma g}$ to return one of the solutions.

We are now ready to theoretically characterize the solutions of DRP.

Theorem 1. Let R be the MMSE restoration operator (7) corresponding to the restoration problem (4) under Assumptions 1-3. Then, any fixed-point $x^{*} \in R^{n}$ of DRP satisfies

$$
\mathbf {0} \in \partial g (\pmb {x} ^ {*}) + \nabla h (\pmb {x} ^ {*}),
$$

where $h$ is given in (9).

The proof of the theorem is provided in the appendix and generalizes the well-known Tweedie's formula (Robbins, 1956; Miyasawa, 1961; Gribonval, 2011) to restoration operators. The theorem implies that the solutions of DRP satisfy the first-order conditions for the objective function $f = g + h$ . If $g$ is a negative log-likelihood $p_{\mathbf{y}|\mathbf{x}}$ , then the fixed-points of DRP can be interpreted as maximum-a-posteriori probability (MAP) solutions corresponding to the prior density $p_{s}$ . The density $p_{s}$ is related to the true prior $p_{x}$ through eq. (8), which implies that DRP has access to the prior $p_{x}$ through the restoration operator $\mathsf{R}$ via density $p_{s}$ . As $\mathbf{H} \to \mathbf{I}$ and $\sigma \to 0$ , the density $p_{s}$ approaches the prior distribution $p_{x}$ .

The convergence analysis of DRP will require additional assumptions.

Assumption 3. The data-fidelity term g and the implicit regularizer h are bounded from below.

This assumption implies that there exists $f^{*} > -\infty$ such that $f(\boldsymbol{x}) \geq f^{*}$ for all $x \in R^{n}$ .

Assumption 4. The function h has a Lipschitz continuous gradient with constant L > 0. The degradation operator associated with the restoration network is such that $\lambda \succeq H^{T}H \succeq \mu > 0$ .

This assumption is related to the implicit prior associated with a restoration model and is necessary to ensure the monotonic reduction of the objective f by the DRP iterates. As stated under eq. (9), the function h is infinitely continuously differentiable. We additionally adopt the standard optimization assumption that $\nabla h$ is Lipschitz continuous (Nesterov, 2004). It is also worth noting that the positive definiteness of $H^{T}H$ in Assumption 4 is a relaxation of the traditional PnP assumption that the prior is a denoiser, which makes our theoretical analysis a significant extension of the prior work (Bigdeli et al., 2017; Xu et al., 2020; Kadkhodaie & Simoncelli, 2021; Gan et al., 2023).

We are now ready to state the following results.

Theorem 2. Run DRP for for $t \geq 1$ iterations under Assumptions 1-4 using a step-size $\gamma = \mu / (\alpha L)$ with $\alpha > 1$ . Then, for each iteration $1 \leq k \leq t$ , there exists $\boldsymbol{w}(\boldsymbol{x}^k) \in \partial f(\boldsymbol{x}^k)$ such that

$$
\min _ {1 \leq k \leq t} \| \boldsymbol {w} (\boldsymbol {x} ^ {k}) \| _ {2} ^ {2} \leq \frac {1}{t} \sum_ {k = 1} ^ {t} \| \boldsymbol {w} (\boldsymbol {x} ^ {k}) \| _ {2} ^ {2} \leq \frac {C (f (\boldsymbol {x} ^ {0}) - f ^ {*})}{t},
$$

where $C > 0$ is an iteration independent constant.

The exact expression for the constant C is given in the proof. Theorem 2 shows that the iterates generated by DRP satisfy $\boldsymbol{w}(\boldsymbol{x}^{k}) \to \mathbf{0}$ as $t \to \infty$ . Theorems 1 and 2 do not explicitly require convexity or smoothness of g, and non-expansiveness of R. They can thus be viewed as a major generalization of the existing theory from denoisers to more general restoration operators.

# 5 Numerical Results

We now numerically validate DRP on several distinct inverse problems. Due to space limitations in the main paper, we have included several additional numerical results in the appendix.

![](images/117768ce225a9726d67e0e5ee36c1148a35f18dc2700beccd8b46129ba1efd48.jpg)  
Figure 1: Illustration of the convergence behaviour of DRP for image deblurring and single image super resolution on the Set3c dataset. (a)-(b): Deblurring with Gaussian blur kernels of standard deviations 1.6 and 2.0. (c)-(d): $2 \times$ and $3 \times$ super resolution with the Gaussian blur kernel of standard deviation 2.0. Average distance $\| x^{k} - x^{k-1}\|_{2}^{2}$ and PSNR relative to the groundtruth are plotted, with shaded areas indicating the standard deviation of these metrics across all test images.

We consider two inverse problems of form $y = Ax + e$ : (a) Image Deblurring and (b) Single Image Super Resolution (SISR). For both problems, we assume that e is the additive white Gaussian noise (AWGN). We adopt the traditional $\ell_{2}$ -norm loss as the data-fidelity term in (2) for both problems. We use the Peak Signal-to-Noise Ratio (PSNR) for quantitative performance evaluation.

In the main manuscript, we compare DRP with several variants of denoiser-based methods, including SD-RED (Romano et al., 2017), PnP-ADMM (Chan et al., 2017), IRCNN (Zhang et al., 2017b), and DPIR (Zhang et al., 2022). SD-RED and PnP-ADMM refer to the steepest-descent variant of RED and the ADMM variant of PnP, both of which incorporate AWGN denoisers based on DnCNN (Zhang et al., 2017a). IRCNN and DPIR are based on half-quadratic splitting (HQS) iterations that use the IRCNN and the DRUNet denoisers, respectively.

In the appendix, we present several additional comparisons, namely: (a) evaluation of the performance of DRP on the task of image denoising; (b) additional comparison of DRP with the recent provably convergent variant of PnP called gradient-step plug-and-play (GS-PnP) (Hurault et al., 2022a); (c) comparison of DRP with the diffusion posterior sampling (DPS) (Chung et al., 2023) method that uses a denoising diffusion model as a prior; and (d) illustration of the improvement of DRP using SwinIR as a prior over the direct application of SwinIR on SR using the Gaussian kernel.

# 5.1 Swin Transformer based Super Resolution Prior

Super Resolution Network Architecture. We pre-trained a $q \times$ super resolution model $R_{q}$ using the SwinIR (Liang et al., 2021) architecture based on Swin Transformer. Our training dataset comprised both the DIV2K (Agustsson & Timofte, 2017) and Flick2K (Lim et al., 2017) dataset, containing 3450 color images in total. During training, we applied $q \times$ bicubic downsampling to the input images with AWGN characterized by standard deviation $\sigma$ randomly chosen in [0, 10/255]. We used three SwinIR SR models, each trained for different down-sampling factors: $2 \times$ , $3 \times$ and $4 \times$ .

Prior Refinement Strategy for the Super Resolution prior. Theorem 1 suggests that as $H \rightarrow I$ , the prior in DRP converges to $p_{x}$ . This process can be approximated for SwinIR by controlling the down-sampling factor q of the SR restoration prior $\mathsf{R}_{q}(\cdot)$ . We observed through our numerical experiments that gradual reduction of q leads to less reconstruction artifacts and enhanced fine details. We will denote the approach of gradually reducing q as prior refinement strategy. We initially set q to a larger down-sampling factor, which acts as a more aggressive prior; we then reduce q to a smaller value leading to preservation of finer details. This strategy is conceptually analogous to the gradual reduction of $\sigma$ in the denoiser in the SOTA PnP methods such as DPIR (Zhang et al., 2022).

<table><tr><td>Kernel</td><td>Datasets</td><td>SD-RED</td><td>PnP-ADMM</td><td>IRCNN+</td><td>DPIR</td><td>DRP</td></tr><tr><td rowspan="4"><img src="images/3b89ccc87b4da9a33eecbd9dcb8ab854699f696d4291022353426e01902a8660.jpg"/></td><td>Set3c</td><td>27.14</td><td>29.11</td><td>28.14</td><td>29.53</td><td>30.69</td></tr><tr><td>Set5</td><td>29.78</td><td>32.31</td><td>29.46</td><td>32.38</td><td>32.79</td></tr><tr><td>CBSD68</td><td>25.78</td><td>28.90</td><td>26.86</td><td>28.86</td><td>29.10</td></tr><tr><td>McMaster</td><td>29.69</td><td>32.20</td><td>29.15</td><td>32.42</td><td>32.79</td></tr><tr><td rowspan="4"><img src="images/bee916a6cf6736f7921dac98f868c277d1df33e0eb48bfb00e74af89c2343f1e.jpg"/></td><td>Set3c</td><td>25.83</td><td>27.05</td><td>26.58</td><td>27.52</td><td>27.89</td></tr><tr><td>Set5</td><td>28.13</td><td>30.77</td><td>28.75</td><td>30.94</td><td>31.04</td></tr><tr><td>CBSD68</td><td>24.43</td><td>27.45</td><td>25.97</td><td>27.52</td><td>27.46</td></tr><tr><td>McMaster</td><td>28.71</td><td>30.50</td><td>28.27</td><td>30.78</td><td>30.79</td></tr></table>

Table 1: PSNR (dB) of DRP and several SOTA methods for solving inverse problems using denoisers on image deblurring with the Gaussian blur kernels of standard deviation 1.6 and 2.0 on Set3c, Set5, CBSD68 and McMaster datasets. The best and second best results are highlighted. Note how DRP can outperform SOTA PnP methods that use denoisers as priors.

# 5.2 Image Deblurring

Image deblurring is based on the degradation operator of the form $\mathbf{A} = \mathbf{K}$ , where $\mathbf{K}$ is a convolution with the blur kernel $\mathbf{k}$ . We consider image deblurring using two $25 \times 25$ Gaussian kernels (with the standard deviations 1.6 and 2) used in Zhang et al. (2019), and the AWGN vector $e$ corresponding to noise level of 2.55/255. The restoration model used as a prior in DRP is SwinIR introduced in Section 5.1, so that the operation $\mathbf{H}$ corresponds to the standard bicubic downsampling. The scaled proximal operator $\mathsf{sprox}_{\lambda g}$ in (5) with data-fidelity term $g(\mathbf{x}) = \frac{1}{2}\| \mathbf{y} - \mathbf{K}\mathbf{x}\| _2^2$ can be written as

$$
\operatorname{sprox} _ {\gamma g} (\boldsymbol {z}) = \left(\boldsymbol {K} ^ {\top} \boldsymbol {K} + \gamma \mathbf {H} ^ {\top} \mathbf {H}\right) ^ {- 1} \left[ \boldsymbol {K} ^ {\top} \boldsymbol {y} + \gamma \mathbf {H} ^ {\top} \mathbf {H} \boldsymbol {z} \right]. \tag {10}
$$

We adopt a standard approach of using a few iterations of the conjugate gradient (CG) method (see for example Aggarwal et al. (2019)) to implement the scaled proximal operator (10) by avoiding the direct inversion of $(\mathbf{K}^{\mathrm{T}}\mathbf{K} + \gamma\mathbf{H}^{\mathrm{T}}\mathbf{H})$ . In each DRP iteration, we run three steps of a CG solver, starting from a warm initialization from the previous DRP iteration. We fine-turned the hyper-parameter $\gamma$ , $\tau$ and SR restoration prior rate q to achieve the highest PSNR value on the Set5 dataset and then apply the same configuration to the other three datasets.

Figure 1 (a)-(b) illustrates the convergence behaviour of DRP on the Set3c dataset for two blur kernels. Table 1 presents the quantitative evaluation of the reconstruction performance on two different blur kernels, showing that DRP outperforms the baseline methods across four widely-used datasets. Figure 2 visually illustrates the reconstructed results on the same two blur kernels. Note how DRP can reconstruct the fine details of the tiger and starfish, as highlighted within the zoom-in boxes, while all the other baseline methods yield either oversmoothed reconstructions or noticeable artifacts. These results show that DRP can leverage SwinIR as an implicit prior, which not only ensures stable convergence, but also leads to competitive performance when compared to denoisers priors.

Figure 3 illustrates the impact of the prior-refinement strategy described in Section 5.1. We compare three settings: (i) use of only $3\times$ prior, (ii) use of only $2\times$ prior, and (iii) use of the prior-refinement strategy to leverage both $3\times$ and $2\times$ priors. The subfigure on the left shows the convergence of DRP for each configuration, while the ones on the right show the final imaging quality. Note how the reduction of q leads to better performance, which is analogous to what was observed with the reduction of $\sigma$ in the SOTA PnP methods (Zhang et al., 2022).

# 5.3 Single Image Super Resolution

We apply DRP using the bicubic SwinIR prior to Single Image Super Resolution (SISR) task. The measurement operator in SISR can be written as $\mathbf{A} = \mathbf{SK}$ , where $\mathbf{K}$ is convolution with the blur kernel $\mathbf{k}$ and $\mathbf{S}$ performs

![](images/c25d1a6bc6586aa06720a7e9bfc036359da2dfac3ebc4e62a56cdab7590a4df7.jpg)  
Figure 2: Visual comparison of DRP with several well-known methods on Gaussian deblurring of color images. The top row shows results for a blur kernel with a standard deviation (std) of 1.6, while the bottom row shows results for another blur kernel with std = 2. The squares at the bottom-left corner of blurry images show the blur kernels. Each image is labeled by its PSNR in dB with respect to the original image. The visual differences are highlighted in the bottom-right corner. Note how DRP using restoration prior improves over SOTA methods based on denoiser priors.

standard $d$ -fold down-sampling with $d^2 = n / m$ . The scaled proximal operator $\mathsf{sprox}_{\lambda g}$ in (5) with data-fidelity term $g(\pmb{x}) = \frac{1}{2} \| \pmb{y} - \pmb{S}\pmb{K}\pmb{x}\| _2^2$ can be write as:

$$
\operatorname{sprox} _ {\gamma g} (\boldsymbol {z}) = \left(\boldsymbol {K} ^ {\top} \boldsymbol {S} ^ {\top} \boldsymbol {S} \boldsymbol {K} + \gamma \mathbf {H} ^ {\top} \mathbf {H}\right) ^ {- 1} \left[ \boldsymbol {K} ^ {\top} \boldsymbol {S} ^ {\top} \boldsymbol {y} + \gamma \mathbf {H} ^ {\top} \mathbf {H} \boldsymbol {z} \right], \tag {11}
$$

where H is the bicubic downsampling operator. Similarly to deblurring in Section 5.2, we use CG to efficiently compute (11). We adjust the hyper-parameter $\gamma$ , $\tau$ , and the SR restoration prior factor q for the best PSNR performance on Set5, and then use these parameters on the remaining datasets.

We evaluate super-resolution performance across two $25 \times 25$ Gaussian blur kernels, each with distinct standard deviations (1.6 and 2.0), and for two distinct downsampling factors ( $2\times$ and $3\times$ ), incorporating an AWGN vector e corresponding to noise level of 2.55/255.

Figure 1 (c)-(d) illustrates the convergence behaviour of DRP on the Set3c dataset for $2 \times$ and $3 \times$ SISR. Figure 4 shows the visual reconstruction results for the same downsampling factors. Table 2 summarizes the PSNR values achieved by DRP relative to other baseline methods when applied to different blur kernel and downsampling factors on four commonly used datasets.

It is worth highlighting that the SwinIR model used in DRP was pre-trained for the bicubic super-resolution task. Consequently, the direct application of the pre-trained SwinIR to the setting considered in this section leads to the suboptimal performance due to mismatch between the kernels used. See Appendix B.4 to see how DRP improves over the direct application of SwinIR.

# 6 Conclusion

The work presented in this paper proposes a new DRP method for solving imaging inverse problems by using pre-trained restoration operators as priors, presents its theoretical analysis in terms of convergence, and applies the method to two well-known inverse problems. The proposed method and its theoretical analysis extend the recent work using denoisers as priors by considering more general restoration operators. The numerical validation of DRP shows the improvements due to the use of learned SOTA super-resolution models.

![](images/a422bb6441ae6569cec30d6b25e001937d61397e09d81886f1b3ecdd455cefdd.jpg)

<details>
<summary>line</summary>

| Method       | q    | Value (dB) |
| ------------ | ---- | ---------- |
| PSNR         | 2 and q = 3 | 28.12      |
| PSNR         | q = 2        | 30.28      |
| PSNR         | q = 3        | 30.58      |
| Ground Truth | -    | -          |
</details>

Figure 3: Illustration of the impact of different SR factors in the prior used within DRP for image deblurring. We show three scenarios: (i) using only $3\times$ prior, (ii) using only $2\times$ prior, and (iii) the use of the prior refinement strategy, which combines both the $2\times$ and $3\times$ priors. Left: Convergence of PSNR against the iteration number for all three configurations. Right: Visual illustration of the final image for each setting. The visual difference is highlighted by the red arrow in the zoom-in box. Note how the reduction of q can lead to about 0.3 dB improvement in the final performance.

![](images/7cb6de9bbea10c6245b85803ea7fc72967ebecce9c80ae361bf7edfa876c3b96.jpg)

<details>
<summary>text_image</summary>

2× LR Image
PnP-ADMM
IRCNN+
DPIR
DRP (Ours)
Ground Truth
96
27.30
96
27.10
96
27.47
96
28.03
96
PSNR
96
3× LR Image
PnP-ADMM
IRCNN+
DPIR
DRP (Ours)
Ground Truth
25.69
25.30
25.75
26.46
PSNR
</details>

Figure 4: Visual comparison of DRP and several well-known methods on single image super resolution. The top row displays performances for $2 \times SR$ , while the bottom row showcases results for $3 \times SR$ . The lower-left corner of each low-resolution image shows the blur kernels. Each image is labeled by its PSNR in dB with respect to the original image. The visual differences are highlighted by the boxes in the bottom-right corner. Note the excellent performance of the proposed DRP method using the SwinIR prior both visually and in terms of PSNR.

One conclusion of this work is the potential effectiveness of going beyond priors specified by traditional denoisers.

# Limitations

The work presented in this paper comes with several limitations. The proposed DRP method uses pre-trained restoration models as priors, which means that its performance is inherently limited by the quality of the pre-trained model. As shown in this paper, pre-trained restoration models provide a convenient, principled, and flexible mechanism to specify priors; yet, they are inherently self-supervised and their empirical performance can thus be suboptimal compared to priors trained in a supervised fashion for a specific inverse problem. Our theory is based on the assumption that the restoration prior used for inference performs MMSE estimation. While this assumption is reasonable for deep networks trained using the MSE loss, it is not directly applicable to denoisers trained using other common loss functions, such as the $\ell_{1}$ -norm or SSIM. Finally, as is common with most theoretical work, our theoretical conclusions only hold when our assumptions are satisfied, which

<table><tr><td>SR</td><td>Kernel</td><td>Datasets</td><td>SD-RED</td><td>PnP-ADMM</td><td>IRCNN+</td><td>DPIR</td><td>DRP</td></tr><tr><td rowspan="8">2×</td><td rowspan="4"><img src="images/892173e62c8447dad8a5f7cb2a7d717bb7562539c64536778bbaac2809233184.jpg"/></td><td>Set3c</td><td>27.01</td><td>27.88</td><td>27.48</td><td>28.18</td><td>29.26</td></tr><tr><td>Set5</td><td>28.98</td><td>31.41</td><td>29.47</td><td>31.42</td><td>31.47</td></tr><tr><td>CBSD68</td><td>26.11</td><td>28.00</td><td>26.66</td><td>27.97</td><td>28.12</td></tr><tr><td>McMaster</td><td>28.70</td><td>30.98</td><td>29.11</td><td>31.16</td><td>31.39</td></tr><tr><td rowspan="4"><img src="images/8603fcce353aba117691d56b9ed4f334b0b28dd9606af463e1c6f8f00201b527.jpg"/></td><td>Set3c</td><td>25.20</td><td>25.86</td><td>25.92</td><td>26.80</td><td>27.41</td></tr><tr><td>Set5</td><td>28.57</td><td>30.06</td><td>28.91</td><td>30.36</td><td>30.42</td></tr><tr><td>CBSD68</td><td>25.77</td><td>26.88</td><td>26.06</td><td>26.98</td><td>26.98</td></tr><tr><td>McMaster</td><td>28.15</td><td>29.53</td><td>28.41</td><td>29.87</td><td>30.03</td></tr><tr><td rowspan="8">3×</td><td rowspan="4"><img src="images/a916cb91724da050d8e5e296dd9a2fd7c55ddd4f610d6c3fe2bceec9843c1e17.jpg"/></td><td>Set3c</td><td>25.50</td><td>25.85</td><td>25.72</td><td>26.64</td><td>27.77</td></tr><tr><td>Set5</td><td>28.75</td><td>30.09</td><td>29.14</td><td>30.39</td><td>30.83</td></tr><tr><td>CBSD68</td><td>25.69</td><td>26.78</td><td>26.01</td><td>26.80</td><td>27.18</td></tr><tr><td>McMaster</td><td>28.38</td><td>29.52</td><td>28.53</td><td>29.82</td><td>29.92</td></tr><tr><td rowspan="4"><img src="images/5ddbc2a9461d5b9ddde7983842374b5df9ea05b34f1143a21cbaa795fa103ec8.jpg"/></td><td>Set3c</td><td>24.55</td><td>24.87</td><td>24.87</td><td>25.84</td><td>26.84</td></tr><tr><td>Set5</td><td>28.19</td><td>29.26</td><td>28.37</td><td>29.70</td><td>29.88</td></tr><tr><td>CBSD68</td><td>25.40</td><td>26.28</td><td>25.56</td><td>26.39</td><td>26.60</td></tr><tr><td>McMaster</td><td>27.79</td><td>28.72</td><td>27.85</td><td>29.11</td><td>29.47</td></tr></table>

Table 2: PSNR (dB) comparison of DRP and several baselines for SISR on Set3c, Set5, CBSD68 and McMaster datasets. The best and second best results are highlighted. Note the excellent quantitative performance of DRP, which suggests the potential of using general restoration models as priors.

might limit their applicability in certain settings. Our future work will continue investigating ways to extend our theory by exploring alternative strategies for relaxing our assumptions.

# References

H. K. Aggarwal, M. P. Mani, and M. Jacob. Modl: Model-based deep learning architecture for inverse problems. IEEE Trans. Med. Imag., 38(2):394–405, Feb. 2019. ISSN 1558-254X. doi: 10.1109/TMI.2018.2865356.   
E. Agustsson and R. Timofte. Ntire 2017 challenge on single image super-resolution: Dataset and study. In Proc. IEEE Conf. Comput. Vis. and Pattern Recognit. (CVPR) workshops, July 2017.   
R. Ahmad, C. A. Bouman, G. T. Buzzard, S. Chan, S. Liu, E. T. Reehorst, and P. Schniter. Plug-and-play methods for magnetic resonance imaging: Using denoisers for image recovery. IEEE Sig. Process. Mag., 37(1):105–116, 2020.   
A. Beck. First-Order Methods in Optimization. MOS-SIAM Series on Optimization. SIAM, 2017.   
S. A. Bigdeli, M. Zwicker, P. Favaro, and M. Jin. Deep mean-shift priors for image restoration. Proc. Adv. Neural Inf. Process. Syst., 30, 2017.   
A. Bora, A. Jalal, E. Price, and A. G. Dimakis. Compressed sensing using generative priors. In Int. Conf. Mach. Learn., pp. 537–546, Sydney, Australia, August 2017.   
G. T. Buzzard, S. H. Chan, S. Sreehari, and C. A. Bouman. Plug-and-play unplugged: Optimization free reconstruction using consensus equilibrium. SIAM J. Imaging Sci., 11(3):2001–2020, Sep. 2018.   
S. H. Chan, X. Wang, and O. A. Elgendy. Plug-and-play ADMM for image restoration: Fixed-point convergence and applications. IEEE Trans. Comp. Imag., 3(1):84–98, Mar. 2017.   
H. Chen, Y. Zhang, M. K. Kalra, F. Lin, Y. Chen, P. Liao, J. Zhou, and G. Wang. Low-dose CT with a residual encoder-decoder convolutional neural network. IEEE Trans. Med. Imag., 36(12):2524–2535, Dec. 2017.

H. Chung, J. Kim, M. T. McCann, M. L. K., and J. C. Ye. Diffusion posterior sampling for general noisy inverse problems. In Int. Conf. on Learn. Represent., 2023. URL https://openreview.net/forum?id=OnD9zGAGTOk.   
R. Cohen, Y. Blau, D. Freedman, and E. Rivlin. It has potential: Gradient-driven denoisers for convergent solutions to inverse problems. In Proc. Adv. Neural Inf. Process. Syst. 34, 2021a.   
R. Cohen, M. Elad, and P. Milanfar. Regularization by denoising via fixed-point projection (red-pro). SIAM Journal on Imaging Sciences, 14(3):1374–1406, 2021b.   
M. Delbracio and P. Milanfar. Inversion by direct iteration: An alternative to denoising diffusion for image restoration. Trans. on Mach. Learn. Research, 2023. ISSN 2835-8856. URL https://openreview.net/forum?id=VmyFF51L3F. Featured Certification.   
M. Delbracio, H. Talebei, and P. Milanfar. Projected distribution loss for image enhancement. In 2021 Int. Conf. on Comput. Photography (ICCP), pp. 1–12. IEEE, 2021.   
W. Dong, P. Wang, W. Yin, G. Shi, F. Wu, and X. Lu. Denoising prior driven deep neural network for image restoration. IEEE Trans. Pattern Anal. Mach. Intell., 41(10):2305–2318, Oct 2019.   
W. Gan, S. Shoushtari, Y. Hu, J. Liu, H. An, and U. S. Kamilov. Block coordinate plug-and-play methods for blind inverse problems. In Proc. Adv. Neural Inf. Process. Syst. 36, 2023.   
D. Gilton, G. Ongie, and R. Willett. Deep equilibrium architectures for inverse problems in imaging. IEEE Trans. Comput. Imag., 7:1123-1133, 2021.   
R. Gribonval. Should penalized least squares regression be interpreted as maximum a posteriori estimation? IEEE Trans. Signal Process., 59(5):2405–2410, May 2011.   
R. Gribonval and P. Machart. Reconciling “priors” & “priors” without prejudice? In Proc. Adv. Neural Inf. Process. Syst. 26, pp. 2193–2201, Lake Tahoe, NV, USA, December 5-10, 2013.   
R. Gribonval and M. Nikolova. On Bayesian estimation and proximity operators. Appl. Comput. Harmon. Anal., 50:49–72, January 2021.   
A. Hauptmann, F. Lucka, M. Betcke, N. Huynh, J. Adler, B. Cox, P. Beard, S. Ourselin, and S. Arridge. Model-based learning for accelerated, limited-view 3-d photoacoustic tomography. IEEE Trans. Med. Imag., 37(6):1382–1393, 2018.   
S. Hurault, A. Leclaire, and N. Papadakis. Gradient step denoiser for convergent plug-and-play. In Int. Conf. on Learn. Represent., Kigali, Rwanda, May 1-5, 2022a.   
S. Hurault, A. Leclaire, and N. Papadakis. Proximal denoiser for convergent plug-and-play optimization with nonconvex regularization. In Int. Conf. Mach. Learn., pp. 9483–9505, Kigali, Rwanda, 2022b. PMLR.   
K. H. Jin, M. T. McCann, E. Froustey, and M. Unser. Deep convolutional neural network for inverse problems in imaging. IEEE Trans. Image Process., 26(9):4509–4522, Sep. 2017. doi: 10.1109/TIP.2017.2713099.   
Z. Kadkhodaie and E. P. Simoncelli. Stochastic solutions for linear inverse problems using the prior implicit in a denoiser. In Proc. Adv. Neural Inf. Process. Syst. 34, pp. 13242–13254, December 6-14, 2021.   
U. S. Kamilov, C. A. Bouman, G. T. Buzzard, and B. Wohlberg. Plug-and-play methods for integrating physical and learned models in computational imaging. IEEE Signal Process. Mag., 40(1):85–97, January 2023.   
E. Kang, J. Min, and J. C. Ye. A deep convolutional neural network using directional wavelets for low-dose x-ray CT reconstruction. Med. Phys., 44(10):e360–e375, 2017.

R. Laumont, V. De Bortoli, A. Almansa, J. Delon, A. Durmus, and M. Pereyra. Bayesian imaging using plug & play priors: When Langevin meets Tweedie. SIAM J. Imaging Sci., 15(2):701–737, 2022.   
J. Liang, J. Cao, G. Sun, K. Zhang, L. Van G., and R. Timofte. Swinir: Image restoration using swin transformer. In Proc. IEEE Int. Conf. Comp. Vis. (ICCV), pp. 1833–1844, 2021.   
B. Lim, S. Son, H. Kim, S. Nah, and K. Mu L. Enhanced deep residual networks for single image super-resolution. In Proc. IEEE Conf. Comput. Vis. and Pattern Recognit. (CVPR) workshops, pp. 136–144, 2017.   
J. Liu, Y. Sun, C. Eldeniz, W. Gan, H. An, and U. S. Kamilov. RARE: Image reconstruction using deep priors learned without ground truth. IEEE J. Sel. Topics Signal Process., 14(6):1088–1099, Oct. 2020.   
J. Liu, S. Asif, B. Wohlberg, and U. S. Kamilov. Recovery analysis for plug-and-play priors using the restricted eigenvalue condition. In Proc. Adv. Neural Inf. Process. Syst. 34, pp. 5921–5933, December 6-14, 2021.   
J. Liu, X. Xu, W. Gan, S. Shoushtari, and U. S. Kamilov. Online deep equilibrium learning for regularization by denoising. In Proc. Adv. Neural Inf. Process. Syst., New Orleans, LA, 2022.   
A. Lucas, M. Iliadis, R. Molina, and A. K. Katsaggelos. Using deep neural networks for inverse problems in imaging: Beyond analytical methods. IEEE Signal Process. Mag., 35(1):20–36, January 2018.   
M. T. McCann, K. H. Jin, and M. Unser. Convolutional neural networks for inverse problems in imaging: A review. IEEE Signal Process. Mag., 34(6):85–95, 2017.   
T. Meinhardt, M. Moeller, C. Hazirbas, and D. Cremers. Learning proximal operators: Using denoising networks for regularizing inverse imaging problems. In Proc. IEEE Int. Conf. Comp. Vis., pp. 1799–1808, Venice, Italy, Oct. 22-29, 2017.   
C. Metzler, P. Schniter, A. Veeraraghavan, and R. Baraniuk. prDeep: Robust phase retrieval with a flexible deep network. In Proc. 36th Int. Conf. Mach. Learn., pp. 3501–3510, Stockholmsmässan, Stockholm Sweden, Jul. 10–15 2018.   
K. Miyasawa. An empirical bayes estimator of the mean of a normal population. Bull. Inst. Internat. Statist, 38:1–2, 1961.   
V. Monga, Y. Li, and Y. C. Eldar. Algorithm unrolling: Interpretable, efficient deep learning for signal and image processing. IEEE Signal Process. Mag., 38(2):18–44, March 2021.   
Y. Nesterov. Introductory Lectures on Convex Optimization: A Basic Course. Kluwer Academic Publishers, 2004.   
G. Ongie, A. Jalal, C. A. Metzler, R. G. Baraniuk, A. G. Dimakis, and R. Willett. Deep learning techniques for inverse problems in imaging. IEEE J. Sel. Areas Inf. Theory, 1(1):39–56, May 2020.   
N. Parikh and S. Boyd. Proximal algorithms. Foundations and Trends in Optimization, 1(3):123-231, 2014.   
E. T. Reehorst and P. Schniter. Regularization by denoising: Clarifications and new interpretations. IEEE Trans. Comput. Imag., 5(1):52–67, March 2019.   
H. Robbins. An empirical Bayes approach to statistics. Proc. Third Berkeley Symp. on Math. Statist. and Prob., Vol. 1 (Univ. of Calif. Press, 1956), pp. 157–163, 1956.   
Y. Romano, M. Elad, and P. Milanfar. The little engine that could: Regularization by denoising (RED). SIAM J. Imaging Sci., 10(4):1804–1844, 2017.   
E. K. Ryu, J. Liu, S. Wang, X. Chen, Z. Wang, and W. Yin. Plug-and-play methods provably converge with properly trained denoisers. In Proc. 36th Int. Conf. Mach. Learn., volume 97, pp. 5546–5557, Long Beach, CA, USA, Jun. 09–15 2019.

S. Sreehari, S. V. Venkatakrishnan, B. Wohlberg, G. T. Buzzard, L. F. Drummy, J. P. Simmons, and C. A. Bouman. Plug-and-play priors for bright field electron tomography and sparse interpolation. IEEE Trans. Comput. Imaging, 2(4):408–423, Dec. 2016.   
Y. Sun, B. Wohlberg, and U. S. Kamilov. An online plug-and-play algorithm for regularized image reconstruction. IEEE Trans. Comput. Imag., 5(3):395–408, September 2019.   
Y. Sun, S. Xu, Y. Li, L. Tian, B. Wohlberg, and U. S. Kamilov. Regularized Fourier ptychography using an online plug-and-play algorithm. In Proc. IEEE Int. Conf. Acoustics, Speech and Signal Process. (ICASSP), pp. 7665–7669, Brighton, UK, May 12-17, 2019. doi: 10.1109/ICASSP.2019.8683057.   
Y. Sun, Z. Wu, B. Wohlberg, and U. S. Kamilov. Scalable plug-and-play ADMM with convergence guarantees. IEEE Trans. Comput. Imag., 7:849–863, July 2021.   
A. M. Teodoro, J. M. Bioucas-Dias, and M. A. T. Figueiredo. A convergent image fusion algorithm using scene-adapted Gaussian-mixture-based denoising. IEEE Trans. Image Process., 28(1):451–463, January 2019.   
T. Tirer and R. Giryes. Image restoration by iterative denoising and backward projections. IEEE Trans. Image Process., 28(3):1220–1234, Mar. 2019.   
S. V. Venkatakrishnan, C. A. Bouman, and B. Wohlberg. Plug-and-play priors for model based reconstruction. In Proc. IEEE Global Conf. Signal Process. and Inf. Process., pp. 945–948, Austin, TX, USA, Dec. 3-5, 2013.   
S. Wang, Z. Su, L. Ying, X. Peng, S. Zhu, F. Liang, D. Feng, and D. Liang. Accelerating magnetic resonance imaging via deep learning. In Proc. Int. Symp. Biomedical Imaging, pp. 514–517, April 2016. doi: 10.1109/ISBI.2016.7493320.   
Y. Wang, J. Yu, and J. Zhang. Zero-shot image restoration using denoising diffusion null-space model. arXiv preprint arXiv:2212.00490, 2022.   
K. Wei, A. Aviles-Rivero, J. Liang, Y. Fu, C.-B. Schönlieb, and H. Huang. Tuning-free plug-and-play proximal algorithm for inverse imaging problems. In Proc. 37th Int. Conf. Mach. Learn., 2020.   
X. Xu, Y. Sun, J. Liu, B. Wohlberg, and U. S. Kamilov. Provable convergence of plug-and-play priors with mmse denoisers. IEEE Signal Process. Lett., 27:1280–1284, 2020.   
J. Zhang and B. Ghanem. ISTA-Net: Interpretable optimization-inspired deep network for image compressive sensing. In Proc. IEEE Conf. Comput. Vision Pattern Recognit., pp. 1828–1837, 2018.   
K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang. Beyond a Gaussian denoiser: Residual learning of deep CNN for image denoising. IEEE Trans. Image Process., 26(7):3142–3155, Jul. 2017a.   
K. Zhang, W. Zuo, S. Gu, and L. Zhang. Learning deep CNN denoiser prior for image restoration. In Proc. IEEE Conf. Comput. Vis. and Pattern Recognit., pp. 3929–3938, Honolulu, USA, July 21-26, 2017b.   
K. Zhang, W. Zuo, and L. Zhang. Deep plug-and-play super-resolution for arbitrary blur kernels. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pp. 1671–1681, Long Beach, CA, USA, June 16-20, 2019.   
K. Zhang, Y. Li, W. Zuo, L. Zhang, L. Van Gool, and R. Timofte. Plug-and-play image restoration with deep denoiser prior. IEEE Trans. Patt. Anal. and Machine Intell., 44(10):6360–6376, October 2022.   
Y. Zhu, K. Zhang, J. Liang, J. Cao, B. Wen, R. Timofte, and L. Van G. Denoising diffusion models for plug-and-play image restoration. In Proc. IEEE Conf. Comput. Vis. and Pattern Recognit. (CVPR), pp. 1219–1229, 2023.

# A Theoretical Analysis of DRP

# A.1 Proof of Theorem 1

Theorem. Let R be the MMSE restoration operator (7) corresponding to the restoration problem (4) under Assumptions 1-3. Then, any fixed-point $x^{*} \in R^{n}$ of DRP satisfies

$$
\mathbf {0} \in \partial g (\boldsymbol {x} ^ {*}) + \nabla h (\boldsymbol {x} ^ {*}),
$$

where $h$ is given in (9).

Proof. First note that any fixed point $x^{*} \in R^{n}$ of the DRP method can be expressed as

$$
\boldsymbol {x} ^ {*} = \operatorname{sprox} _ {\gamma g} (\boldsymbol {x} ^ {*} - \gamma \tau G (\boldsymbol {x} ^ {*})) = \underset {\boldsymbol {x} \in \mathbb {R} ^ {n}} {\arg \min} \left\{\frac {1}{2} \| \boldsymbol {x} - (\boldsymbol {x} ^ {*} - \gamma \tau G (\boldsymbol {x} ^ {*})) \| _ {\mathbf {H} ^ {\intercal} \mathbf {H}} ^ {2} + \gamma g (\boldsymbol {x}) \right\}, \tag {12}
$$

where we used the definition of the scaled proximal operator. From the optimality conditions of the scaled proximal operator, we then get

$$
\mathbf {0} \in \partial g (\boldsymbol {x} ^ {*}) + \tau \mathbf {H} ^ {\top} \mathbf {H} \mathsf {G} (\boldsymbol {x} ^ {*}). \tag {13}
$$

On the other hand, the gradient of $p_{s}$ , defined in (8), can be expressed as

$$
\nabla_ {\boldsymbol {s}} p _ {\boldsymbol {s}} (\boldsymbol {s}) = \int \left(\frac {1}{\sigma^ {2}} (\mathbf {H} \boldsymbol {x} - \boldsymbol {s})\right) G _ {\sigma} (\boldsymbol {s} - \mathbf {H} \boldsymbol {x}) p _ {\boldsymbol {x}} (\boldsymbol {x}) d \boldsymbol {x} = \frac {1}{\sigma^ {2}} (\mathbf {H R} (\boldsymbol {s}) - \boldsymbol {s} p _ {\boldsymbol {s}} (\boldsymbol {s})), \tag {14}
$$

where we used the gradient of the Gaussian density with respect to s and the definition of the MMSE restoration operator in eq. (7). By rearranging the terms, we obtain the following relationship

$$
\mathbf {H} \mathrm{R} (\boldsymbol {s}) - \boldsymbol {s} = \sigma^ {2} \nabla \log p _ {\boldsymbol {s}} (\boldsymbol {s}), \quad \boldsymbol {s} \in \mathbb {R} ^ {p}. \tag {15}
$$

By using the definitions $\mathsf{G}(\pmb{x}) = \pmb{x} - \mathsf{R}(\mathbf{H}\pmb{x})$ and $h(\pmb{x}) = -\tau \sigma^2\log p_s(\mathbf{H}\pmb{x})$ , with $\pmb{x} \in \mathbb{R}^n$ , in (15), we obtain the following generalization of the well-known Tweedie's formula

$$
\mathbf {H} ^ {\top} \mathbf {H G} (\boldsymbol {x}) = \mathbf {H} ^ {\top} \mathbf {H} \left(\boldsymbol {x} - \mathrm{R} (\mathbf {H} \boldsymbol {x})\right) = - \sigma^ {2} \nabla \log p _ {\boldsymbol {s}} (\mathbf {H} \boldsymbol {x}) = \frac {1}{\tau} \nabla h (\boldsymbol {x}). \tag {16}
$$

By combining(12) and (16), we directly obtain the desired result

$$
\mathbf {0} \in \partial g (\boldsymbol {x} ^ {*}) + \nabla h (\boldsymbol {x} ^ {*}).
$$

# A.2 Proof of Theorem 2

Theorem. Run DRP for for $t \geq 1$ iterations under Assumptions 1-4 using a step-size $\gamma = \mu / (\alpha L)$ with $\alpha > 1$ . Then, for each iteration $1 \leq k \leq t$ , there exists $\boldsymbol{w}(\boldsymbol{x}^{k}) \in \partial f(\boldsymbol{x}^{k})$ such that

$$
\min _ {1 \leq k \leq t} \| \boldsymbol {w} (\boldsymbol {x} ^ {k}) \| _ {2} ^ {2} \leq \frac {1}{t} \sum_ {k = 1} ^ {t} \| \boldsymbol {w} (\boldsymbol {x} ^ {k}) \| _ {2} ^ {2} \leq \frac {C (f (\boldsymbol {x} ^ {0}) - f ^ {*})}{t},
$$

where $C > 0$ is an iteration independent constant.

Proof. Consider the iteration $k \geq 1$ of DRP

$$
\boldsymbol {x} ^ {k} = \operatorname{sprox} _ {\gamma g} \left(\boldsymbol {x} ^ {k - 1} - \gamma \tau \mathsf {G} (\boldsymbol {x} ^ {k - 1})\right) \quad \text {with} \quad \mathsf {G} := \boldsymbol {x} - \mathsf {R} (\mathbf {H} \boldsymbol {x}),
$$

where R is the MMSE restoration operator specified in (7). This implies that $x^{k}$ minimizes

$$
\varphi (\boldsymbol {x}) := \frac {1}{2 \gamma} (\boldsymbol {x} - \boldsymbol {x} ^ {k - 1}) ^ {\top} \mathbf {H} ^ {\top} \mathbf {H} (\boldsymbol {x} - \boldsymbol {x} ^ {k - 1}) + \left[ \tau \mathbf {H} ^ {\top} \mathbf {H G} (\boldsymbol {x} ^ {k - 1}) \right] ^ {\top} (\boldsymbol {x} - \boldsymbol {x} ^ {k - 1}) + g (\boldsymbol {x})
$$

$$
= \frac {1}{2 \gamma} (\boldsymbol {x} - \boldsymbol {x} ^ {k - 1}) ^ {\top} \mathbf {H} ^ {\top} \mathbf {H} (\boldsymbol {x} - \boldsymbol {x} ^ {k - 1}) + \nabla h (\boldsymbol {x} ^ {k - 1}) ^ {\top} (\boldsymbol {x} - \boldsymbol {x} ^ {k - 1}) + g (\boldsymbol {x}),
$$

where in the second inequality we used eq. (16) from the proof in Appendix A.1. By evaluating $\varphi$ at $\pmb{x}^k$ and $\pmb{x}^{k - 1}$ , we obtain the following useful inequality

$$
g (\boldsymbol {x} ^ {k}) \leq g (\boldsymbol {x} ^ {k - 1}) - \frac {1}{2 \gamma} (\boldsymbol {x} ^ {k} - \boldsymbol {x} ^ {k - 1}) ^ {\top} \mathbf {H} ^ {\top} \mathbf {H} (\boldsymbol {x} ^ {k} - \boldsymbol {x} ^ {k - 1}) - \nabla h (\boldsymbol {x} ^ {k - 1}) ^ {\top} (\boldsymbol {x} ^ {k} - \boldsymbol {x} ^ {k - 1}). \tag {17}
$$

On the other hand, from the L-Lipschitz continuity of $\nabla h$ , we have the following bound

$$
h (\boldsymbol {x} ^ {k}) \leq h (\boldsymbol {x} ^ {k - 1}) + \nabla h (\boldsymbol {x} ^ {k - 1}) ^ {\top} (\boldsymbol {x} ^ {k} - \boldsymbol {x} ^ {k - 1}) + \frac {L}{2} \| \boldsymbol {x} ^ {k} - \boldsymbol {x} ^ {k - 1} \| _ {2} ^ {2}. \tag {18}
$$

By combining eqs. (17) and (18), we obtain

$$
f (\boldsymbol {x} ^ {k}) \leq f (\boldsymbol {x} ^ {k - 1}) - \frac {1}{2} (\boldsymbol {x} ^ {k} - \boldsymbol {x} ^ {k - 1}) ^ {\top} \left[ \frac {1}{\gamma} \mathbf {H} ^ {\top} \mathbf {H} - L \mathbf {I} \right] (\boldsymbol {x} ^ {k} - \boldsymbol {x} ^ {k - 1})
$$

$$
\leq f (\boldsymbol {x} ^ {k - 1}) - (\alpha - 1) \frac {L}{2} \| \boldsymbol {x} ^ {k} - \boldsymbol {x} ^ {k - 1} \| _ {2} ^ {2}, \tag {19}
$$

where we used $\gamma=\mu/(\alpha L)$ with $\alpha>1$ and $\mu>0$ defined in Assumption 4.

On the other hand, from the optimality conditions for $\varphi$ , we also have

$$
\mathbf {0} \in \mathbf {H} ^ {\top} \mathbf {H} (\boldsymbol {x} ^ {k} - \boldsymbol {x} ^ {k - 1} + \gamma \tau \mathsf {G} (\boldsymbol {x} ^ {k - 1})) + \gamma \partial g (\boldsymbol {x} ^ {k})
$$

$$
\Leftrightarrow \quad \frac {1}{\gamma} \mathbf {H} ^ {\top} \mathbf {H} (\boldsymbol {x} ^ {k} - \boldsymbol {x} ^ {k - 1}) \in \partial g (\boldsymbol {x} ^ {k}) + \nabla h (\boldsymbol {x} ^ {k - 1}),
$$

where we used eq. (16) from Appendix A.1. This directly implies that the following inclusion

$$
\boldsymbol {w} (\boldsymbol {x} ^ {k}) := \frac {1}{\gamma} \mathbf {H} ^ {\top} \mathbf {H} (\boldsymbol {x} ^ {k} - \boldsymbol {x} ^ {k - 1}) + \nabla h (\boldsymbol {x} ^ {k}) - \nabla h (\boldsymbol {x} ^ {k - 1}) \in \partial f (\boldsymbol {x} ^ {k})
$$

The norm of the subgradient $\boldsymbol{w}(\boldsymbol{x}^{k})$ can be bounded as follows

$$
\| \boldsymbol {w} (\boldsymbol {x} ^ {k}) \| _ {2} \leq \frac {1}{\gamma} \| \mathbf {H} ^ {\top} \mathbf {H} (\boldsymbol {x} ^ {k} - \boldsymbol {x} ^ {k - 1}) \| _ {2} + \| \nabla h (\boldsymbol {x} ^ {k}) - \nabla h (\boldsymbol {x} ^ {k - 1}) \| _ {2}
$$

$$
\leq L \left(\alpha (\lambda / \mu) + 1\right) \| \boldsymbol {x} ^ {k} - \boldsymbol {x} ^ {k - 1} \| _ {2}, \tag {20}
$$

where we used the Lipschitz constant of $\nabla h$ , $\gamma = \mu / (\alpha L)$ , and $\lambda \geq \mu > 0$ defined in Assumption 4.

By combining eqs. (19) and (20), we obtain the following inequality

$$
\left\| \boldsymbol {w} \left(\boldsymbol {x} ^ {k}\right) \right\| _ {2} ^ {2} \leq A _ {1} \left\| \boldsymbol {x} ^ {k} - \boldsymbol {x} ^ {k - 1} \right\| _ {2} ^ {2} \leq A _ {2} \left(f \left(\boldsymbol {x} ^ {k - 1}\right) - f \left(\boldsymbol {x} ^ {k}\right)\right), \tag {21}
$$

where $A_{1} := L^{2}(\alpha(\lambda/\mu) + 1)^{2} > 0$ and $A_{2} := 2A_{1}/(L(\alpha - 1)) > 0$ . Hence, by averaging over $t \geq 1$ iterations, we can directly get the desired result

$$
\min _ {1 \leq k \leq t} \| \boldsymbol {w} (\boldsymbol {x} ^ {k}) \| _ {2} ^ {2} \leq \frac {1}{t} \sum_ {k = 1} ^ {t} \| \boldsymbol {w} (\boldsymbol {x} ^ {k}) \| _ {2} ^ {2} \leq \frac {A _ {2} (f (\boldsymbol {x} ^ {0}) - f ^ {*})}{t}. \tag {22}
$$

This implies that $\pmb{w}(\pmb{x}^k)\to \mathbf{0}$ as $t\to \infty$

![](images/6aad61c2f3054aba53a9f357b3dcde67a81a757c3ea6caeb71f4b2058827fa9a.jpg)

# B Additional Numerical Results

# B.1 Image denoising

In this subsection, we show the performance of DRP on Gaussian image denoising. The corresponding measurement model is $y = x + e$ , where e is AWGN with the standard deviation $\sigma$ and x is the unknown clean image. We use the same SwinIR SR model, introduced in 5.1, as the prior for denoising. The degradation model in the SwinIR prior is the operation H corresponding to bicubic downsampling. The scaled proximal operator $sprox_{\lambda g}$ in (5) with data-fidelity term $g(\boldsymbol{x}) = \frac{1}{2} \| \boldsymbol{y} - \boldsymbol{x} \|_2^2$ can be written as

$$
\operatorname{sprox} _ {\gamma g} (\boldsymbol {z}) := \left(\mathbf {I} + \gamma \mathbf {H} ^ {\top} \mathbf {H}\right) ^ {- 1} [ \boldsymbol {y} + \gamma \mathbf {H} ^ {\top} \mathbf {H} \boldsymbol {z} ], \tag {23}
$$

which can be efficiently implemented using CG, as in Section 5.2.

We compare DRP with one of SOTA denoising model DRUNet (Zhang et al., 2019) on noise level ( $\sigma = 0.1$ ). Figure 5 and Figure 6 illustrate the visual performance of DRP on the Set5 and CBSD68 datasets, respectively. Figure 7 further explores the impact of using different SR factors q as priors, elucidating how these choices influence the visual quality of denoising.

![](images/21f9ddfe3d793a465024d75ac16d6d00d383f758a263b97df6be17fd87fcc02a.jpg)  
Figure 5: Illustration of denoising results of DRP on Set5 dataset with noise level $\sigma = 0.1$ . Each image is labeled by its PSNR (dB) with respect to the original image.

![](images/f20163c28854b720b2ab7d298357c9fa80b541760272a4a338c0b68c482bbf11.jpg)

<details>
<summary>text_image</summary>

σ = 0.1
DRUNet
DPR (Ours)
Ground Truth
PSNR
20.05
31.97
29.79
20.84
31.53
29.22
20.10
32.20
30.05
</details>

Figure 6: Illustration of denoising results of DRP on CBSD68 dataset with noise level $\sigma = 0.1$ . Each image is labeled by its PSNR (dB) with respect to the original image.

![](images/7b2435f5c2273dc0a93eae76838cc431a0b8aa4fef1596b7b48ebd387fbcba47.jpg)

<details>
<summary>text_image</summary>

σ = 0.1
2×
3×
4×
Ground Truth
20.85
32.39
31.66
30.51
20.33
29.42
27.05
25.68
</details>

Figure 7: Illustration of denoising results of DRP on Set5 dataset with three SR level prior (2×, 3× and 4×). Each image is labeled by its PSNR (dB) with respect to the original image.

# B.2 Comparison With GS-PnP

In this subsection, we will compare DRP with the recent gradient-step denoiser PnP method (GS-PnP) (Hurault et al., 2022a). These comparisons were not included in the main paper due to space, but are provided here for completeness. GS-PnP provides comparable performance on image deblurring and single image super resolution as DPIR (Zhang et al., 2019), but comes with theoretical convergence guarantees.

Table 3 shows that DRP outperforms both DPIR and GS-PnP on image deblurring in most settings in terms of PSNR. Similarly, Table 4 shows that DRP can achieve better SISR performance in terms of PSNR compared to both methods. Figure 8 provides additional visual results on SISR showing that DRP can recover intricate details and sharpen features.

<table><tr><td>Kernel</td><td>Datasets</td><td>GS-PnP</td><td>DPIR</td><td>DRP</td></tr><tr><td rowspan="2"><img src="images/e0e901c8ef587bd6c093b4387cb6a61f4e6408e324093b3b4f14af9671d3c418.jpg"/></td><td>Set3c</td><td>29.53</td><td>29.78</td><td>30.69</td></tr><tr><td>CBSD68</td><td>28.86</td><td>28.70</td><td>29.10</td></tr><tr><td rowspan="2"><img src="images/20919fb6f0667b8a39cb9650253fd60b63e13eeb1f7133ec429931fbcc8a6039.jpg"/></td><td>Set3c</td><td>27.52</td><td>27.32</td><td>27.89</td></tr><tr><td>CBSD68</td><td>27.44</td><td>27.52</td><td>27.46</td></tr></table>

Table 3: PSNR performance of GS-PnP, DPIR, and DRP for image deblurring on Set3c and CBSD68 datasets on two blur kernels. The best and second best results are highlighted.

<table><tr><td>SR</td><td>Kernels</td><td>Datasets</td><td>GS-PnP</td><td>DPIR</td><td>DRP</td></tr><tr><td rowspan="4">2×</td><td rowspan="2"><img src="images/92037de50b0960a27c561a40ff375e005ea3a9cf5882bc4e98b8c0546b375f2b.jpg"/></td><td>Set3c</td><td>28.23</td><td> $\underline{28.18}$ </td><td>29.26</td></tr><tr><td>CBSD68</td><td>28.03</td><td> $\underline{27.97}$ </td><td>28.12</td></tr><tr><td rowspan="2"><img src="images/398cec6c5a3a72f3800a8f9d5bacce0a452f25e585f8dff55fa7f42442cbed65.jpg"/></td><td>Set3c</td><td>26.19</td><td> $\underline{26.80}$ </td><td>27.41</td></tr><tr><td>CBSD68</td><td>26.79</td><td> $\underline{26.98}$ </td><td>26.98</td></tr><tr><td rowspan="4">3×</td><td rowspan="2"><img src="images/280afeb4ba856fe68b2407108e988df804f4d25a86f090a72715c13e2cab465d.jpg"/></td><td>Set3c</td><td>26.20</td><td> $\underline{26.64}$ </td><td>27.77</td></tr><tr><td>CBSD68</td><td>26.77</td><td> $\underline{26.80}$ </td><td>27.18</td></tr><tr><td rowspan="2"><img src="images/398cec6c5a3a72f3800a8f9d5bacce0a452f25e585f8dff55fa7f42442cbed65.jpg"/></td><td>Set3c</td><td>25.18</td><td> $\underline{25.84}$ </td><td>26.84</td></tr><tr><td>CBSD68</td><td>26.30</td><td> $\underline{26.39}$ </td><td>26.60</td></tr></table>

Table 4: PSNR performance of GS-PnP, DPIR, and DRP for 2× and 3× SISR on the Set3c and CBSD68 datasets, using two blur kernels. The best and second best results are highlighted.

![](images/70842cbf4c37dff1d2f87791733a447607d5f48a8076a1cd99f818fb4fe8f4a2.jpg)  
Figure 8: Illustration of SISR results of DRP compared with two SOTA denoiser based PnP method DPIR (Zhang et al., 2019) and GS-PnP (Hurault et al., 2022a). The top row displays the performance for $2 \times$ SISR task, while the bottom row showcases results for $3 \times$ SISR task. Each image is labeled by its PSNR (dB) with respect to the original image, and the visual difference is highlighted by the boxes in the left bottom corner.

# B.3 Comparison with Diffusion Posterior Sampling

There is a growing interest in using denoisers within diffusion models for solving inverse problems (Zhu et al., 2023; Wang et al., 2022; Chung et al., 2023). One of the most wiedely-adopted diffusion model in this context is the diffusion posterior sampling (DPS) method from (Chung et al., 2023), which integrates pre-trained denoisers and measurement models for posterior sampling. One may argue that DPS is related to PnP due to the use of image denoisers as priors. In this section, we present results comparing DRP with DPS for deblurring human faces. We used the public implementation of DPS on the GitHub page that uses the prior specifically trained on human face image dataset (Chung et al., 2023). DRP uses the same SwinIR model trained on general image datasets (see Section 5.1). DPS and DRP are related but very different classes of methods. While DPS seeks to use denoisers to generate perceptually realistic solutions to inverse problems, DRP enables the adaptation of pre-trained restoration models as priors for solving other inverse problems.

Table 5 presents PSNR results obtained by DPS and DRP for human face deblurring. While we omitted the visual results from the paper for the privacy reasons, we will be happy to provide them if requested by the reviewers. Overall, DPS achieves more perceptually realistic images, while DRP achieves higher PSNR and more closely matches the ground truth images. This is not surprising when considering the generative nature of DPS. A similar observation is available in the original DPS publication, which reported better PSNR and SSIM performance of PnP-ADMM relative to DPS on SISR and deblurring (see Appendix E in (Chung et al., 2023)).

<table><tr><td>Kernel</td><td>DPS</td><td>DRP</td></tr><tr><td><img src="images/b0075f49e26e9fdfdaf5e251ff27b600dd69e5078f0693a3a73227354f6fd4ae.jpg"/></td><td>29.61</td><td>34.61</td></tr><tr><td><img src="images/c97107117ce8fa1e8685ad78e9e33d1198b5e9a9ac7b76fc0b5906c0914b9133.jpg"/></td><td>28.80</td><td>33.05</td></tr></table>

Table 5: PSNR performance of DPS and DRP for image deblurring on three sample images from FFHQ validation set (provided in the DPS GitHub project) with two blur kernels. The best results are highlighted.

# B.4 Performance of Bicubic SwinIR on a mismatched SISR task

In this section, our aim is to bring a noteworthy point: the SwinIR SR prior we employed in our DRP method are specifically trained for bicubic SR task. Consequently, a direct application of SwinIR to SISR task could potentially yield sub-optimal results. This observation implies that our DRP algorithm have the capacity to using the mismatched restoration model as an implicit prior, effectively adapting it for general image restoration tasks.

Figure 9 presents a visual comparison on set3c datasets, accompanied by PSNR values in relation to the groundtruth image. As shown in the figure, diectly using SwinIR trained for bicubic SR task can not handle SISR task, while using it within our GPoP algorithm as a prior can lead to SOTA performance.

![](images/76742df9f6cd3331819855b7fd0a2208c9ffdbb6dc99bbe4decafd1800479e67.jpg)  
Figure 9: Illustration of $2\times$ SISR results of DRP compared with directly using bicubic SR SwinIR. Each image is labeled by its PSNR (dB) with respect to the original image.