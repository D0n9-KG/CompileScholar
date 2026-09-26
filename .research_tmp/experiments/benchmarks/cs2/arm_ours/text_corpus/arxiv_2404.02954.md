Deep Generative Models through the Lens of the Manifold Hypothesis: A Survey and New Connections 
 
 
 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: CC BY 4.0
 
 
arXiv:2404.02954v2 [cs.LG] 25 Sep 2024 
 
 

# Deep Generative Models through the Lens of the Manifold Hypothesis: A Survey and New Connections

 
 
 Gabriel Loaiza-Ganem gabriel@layer6.ai 
 
 Affiliation: Layer 6 AI 
 
    
 Brendan Leigh Ross brendan@layer6.ai 
 
 Affiliation: Layer 6 AI 
 
    
 Rasa Hosseinzadeh rasa@layer6.ai 
 
 Affiliation: Layer 6 AI 
 
    
 Anthony L. Caterini anthony@layer6.ai 
 
 Affiliation: Layer 6 AI 
 
    
 Jesse C. Cresswell jesse@layer6.ai 
 
 Affiliation: Layer 6 AI 
 

 Abstract 
 
 In recent years there has been increased interest in understanding the interplay between deep generative models (DGMs) and the manifold hypothesis. Research in this area focuses on understanding the reasons why commonly-used DGMs succeed or fail at learning distributions supported on unknown low-dimensional manifolds, as well as developing new models explicitly designed to account for manifold-supported data. This manifold lens provides both clarity as to why some DGMs (e.g. diffusion models and some generative adversarial networks) empirically surpass others (e.g. likelihood-based models such as variational autoencoders, normalizing flows, or energy-based models) at sample generation, and guidance for devising more performant DGMs. We carry out the first survey of DGMs viewed through this lens, making two novel contributions along the way. First, we formally establish that numerical instability of likelihoods in high ambient dimensions is unavoidable when modelling data with low intrinsic dimension. We then show that DGMs on learned representations of autoencoders can be interpreted as approximately minimizing Wasserstein distance: this result, which applies to latent diffusion models, helps justify their outstanding empirical results. The manifold lens provides a rich perspective from which to understand DGMs, and we aim to make this perspective more accessible and widespread.

 
 
 
 
 
 Reviewed on OpenReview: https://openreview.net/forum?id=a90WpmSi0I 

 
 

 
 

## Section 1 Introduction

 
 Learning the distribution that gave rise to observed data in ℝ D \mathbb{R}^{D} has long been a central problem in statistics and machine learning ( Lehmann Casella, 2006 ; Murphy, 2012 ) . In the deep learning era, there has been a tremendous amount of research aimed at leveraging neural networks to solve this task, bringing forth deep generative models (DGMs).
This effort has paid off, with state-of-the-art approaches such as diffusion models ( Sohl-Dickstein et al., 2015 ; Ho et al., 2020 ; Song et al., 2021b ) and their latent variants ( Rombach et al., 2022 ) achieving remarkable empirical success ( Ramesh et al., 2022 ; Nichol et al., 2022 ; Saharia et al., 2022 ) .
A natural question is why these latest models outperform previous DGMs, or more generally:

 What makes a deep generative model good? 

 To answer this fundamental question, a line of research has emerged with the goal of understanding DGMs through their relationship with the manifold hypothesis , and using the obtained insights to drive empirical improvements. In its simplest form, this crucial hypothesis states that high-dimensional data of interest often lies on an unknown d ∗ d^{\ast} -dimensional submanifold ℳ \mathcal{M} of ℝ D \mathbb{R}^{D} , with d ∗ D d^{\ast} D . Studying the behaviour of DGMs when the underlying data-generating distribution is supported on an unknown low-dimensional submanifold of ℝ D \mathbb{R}^{D} has already proven fruitful: for example, it is precisely those models with the capacity to learn low-dimensional manifolds (such as diffusion models, latent or otherwise) that tend to work better in practice. Similarly, various pathologies of DGMs – such as mode collapse in variational autoencoders ( Kingma Welling, 2014 ; Rezende et al., 2014 ; Dai Wipf, 2019 ) , and numerical instabilities in the score function of diffusion models ( Pidstrigach, 2022 ; Lu et al., 2023 ) or in normalizing flows ( Dinh et al., 2015 ; Dinh et al., 2017 ; Cornish et al., 2020 ; Behrmann et al., 2021 ) , among others – can be explained as a failure to properly account for manifold structure within data.
Despite the existence of various DGM review papers ( Kobyzev et al., 2020 ; Papamakarios et al., 2021 ; Bond-Taylor et al., 2022 ; Yang et al., 2023 ) , to the best of our knowledge none takes this “manifold lens”. Here, we present the first survey of DGMs from this viewpoint.

 
 
 The relevance and usefulness of studying DGMs through this lens hinges on a critical assumption: namely, that the manifold hypothesis actually holds. It is thus relevant to justify this hypothesis. There is a plethora of arguments supporting the existence of low-dimensional structure in most high-dimensional data of interest, which we summarize below:

 
 
 
 • 
 
 Intuition  Informally, saying that a subset ℳ \mathcal{M} of ℝ D \mathbb{R}^{D} is a low-dimensional submanifold is a mathematical way of capturing two important properties: a sense of sparsity in ambient space ( ℳ \mathcal{M} has volume, or Lebesgue measure, 0 0 in ℝ D \mathbb{R}^{D} ), and a notion of smoothness. These are properties that one can commonly expect of high-dimensional data of interest. For example, consider natural images in [ 0 , 1 ] D [0,1]^{D} (assume they have been scaled to this range), where D D corresponds to the number of pixels. Imagine sampling uniformly from [ 0 , 1 ] D [0,1]^{D} until a human deems the resulting sample to be a natural image. For all intents and purposes, such an image would never be sampled since natural images are “sparse” in their ambient space. Images are also “smooth” in that, given a natural image, one can always conceive of slightly deforming it in such a way that the result remains a natural image. One can also intuitively understand a d ∗ d^{\ast} -dimensional submanifold ℳ \mathcal{M} of ℝ D \mathbb{R}^{D} as a (smooth in some way) subset of intrinsic dimension d ∗ d^{\ast} , which can be thought of as the number of factors of variation needed to characterize a point in ℳ \mathcal{M} .
Again, most machine learning researchers or practitioners who have worked with natural images will have the tacit understanding that far fewer dimensions than the ambient dimension , D D , are needed to describe an image: for example, images can be successfully synthesized from low-dimensional latent variables ( Rombach et al., 2022 ; Sauer et al., 2023 ) , and they can be effectively compressed without affecting how humans perceive them ( Wallace, 1992 ; Townsend et al., 2019 ; Ruan et al., 2021 ) .
Through this line of thinking, the manifold hypothesis has been a motivating concept in the field of machine learning from its infancy. Some of the first autoencoders ( Kramer, 1991 ) aimed to account for data with low intrinsic dimension. Early unsupervised algorithms such as Boltzmann machines ( Ackley et al., 1985 ) were conceived as ways of learning the constraints that govern complex data distributions. Indeed, the manifold hypothesis is one of the core intuitions behind why neural networks are so successful at learning low-dimensional representations in the first place ( Bengio et al., 2013 ) .

 

 • 
 
 Theory  The manifold hypothesis helps explain the success of deep learning through more than just intuition. For example, under standard assumptions, the sample complexity of kernel density estimation in ℝ D \mathbb{R}^{D} is exponential in D D ( Wand Jones, 1994 ) . This well-known result is a manifestation of the curse of dimensionality, and highlights the fundamental hardness of learning arbitrary high-dimensional distributions. Yet DGMs succeed at this task, suggesting that these standard assumptions are too loose and do not take into account relevant structure present in data of interest. Making the assumption that the data lies on a submanifold of ℝ D \mathbb{R}^{D} is a sensible way of incorporating more structure, and is consistent with theory: the sample complexity of kernel density estimation actually scales exponentially with intrinsic dimensionality ( Ozakin Gray, 2009 ; Berenfeld Hoffmann, 2021 ) – even if the data is concentrated around a manifold rather than exactly on one ( Divol, 2022 ) . These results suggest that the task of learning distributions when d ∗ d^{\ast} is much smaller than D D is fundamentally more tractable than when there is no low-dimensional submanifold (i.e. d ∗ = D d^{\ast}=D ). This observation is not unique to density estimation: the difficulty of classification and manifold learning are also known to scale with intrinsic – rather than ambient – dimension ( Narayanan Niyogi, 2009 ; Narayanan Mitter, 2010 ) , making them effectively impossible for high-dimensional data without additional structure.
These results are a sign that manifold structure in data is the reason why deep learning manages to avoid the curse of dimensionality. The triumph of modern algorithms on these tasks thus provides strong implicit justification for the manifold hypothesis.

 

 • 
 
 Empiricism  There are various works which use existing intrinsic dimension estimators ( Levina Bickel, 2004 ; MacKay Ghahramani, 2005 ; Johnsson et al., 2014 ; Facco et al., 2017 ; Bac et al., 2021 ) , or develop their own, and apply them on commonly-used image datasets ( Pope et al., 2021 ; Tempczyk et al., 2022 ; Zheng et al., 2022 ; Brown et al., 2023 ) . These works unanimously estimate the intrinsic dimension of images to be orders of magnitude smaller than their ambient dimension, and similar studies have been carried out on physics datasets with analogous findings ( Cresswell et al., 2022 ) . All these works provide explicit evidence supporting the manifold hypothesis.

 

 
 
 
 Our goal in this survey is to present an accessible, yet mathematically precise, view of DGMs from the perspective of the manifold hypothesis. First, Section 2 characterizes the setup we will consider throughout, and provides a consistent set of notation and terminology with which to describe DGMs. Section 3 lays out relevant background, covering manifold learning and divergences between probability distributions – with a special focus on which ones provide an adequate minimization objective when manifolds are involved. Section 4 describes popular manifold-unaware DGMs – i.e. models which do not account for the manifold hypothesis – and in Section 4.1.1 we provide our first novel result, showing that high-dimensional likelihood-based models are unavoidably bound to suffer numerical instability when the manifold hypothesis holds. Section 5 covers manifold-aware DGMs – i.e. models which do account for the manifold hypothesis. We include both popular DGMs which happen to be manifold-aware and models which were explicitly designed to account for manifold structure. In Section 5.3.1 we provide a new perspective on two-step models , one of the predominant paradigms of manifold-aware DGMs: we show that, in addition to their common interpretation as jointly learning a manifold and a distribution, they minimize a (potentially regularized) upper bound – which can become tight at optimality – of the Wasserstein distance against the true data-generating distribution.
In Section 6 we cover discrete DGMs, before concluding and discussing directions for future research in Section 7 .

 
 
 

## Section 2 Notation and Setup

 
 In this section we present the notation and setup that we will use throughout our work. We try to deviate as little as possible from standard notation, but we nonetheless prioritize precision and consistency across models.
While knowledge of measure theory and differential geometry is needed to understand some technical details of how DGMs relate to manifolds, the core ideas, methods, and intuitions of this area do not require mathematics beyond what is typically known by machine learning researchers. Our intention in this survey is thus to remain accessible to readers with no background in these topics.

 
 
 
 Advanced topics  In the interest of mathematically-inclined readers, we use grey boxes like this one throughout the manuscript to present content which does require familiarity with topics such as measure theory ( Billingsley, 2012 ) , topology ( Munkres, 2014 ) , or differential geometry ( Lee, 2012 ; Lee, 2018 ) . Providing a primer covering all the relevant material from these topics would be prohibitively lengthy and thus falls outside the scope of our work. Nonetheless, we do include a short summary of weak convergence of probability measures in Appendix A , as this is a particularly important tool from measure theory allowing us to formalize the intuition that a model learns its target distribution throughout training – even when this target is supported on a low-dimensional manifold. The material within these grey boxes is self-contained and is not necessary to understand the rest of our survey. 
 
 
 

### Section 2.1 Notation

 
 Ambient and latent spaces 

 
 We denote the D D -dimensional ambient space as 𝒳 \mathcal{X} , where depending on the particular model being discussed, 𝒳 \mathcal{X} could be ℝ D \mathbb{R}^{D} or [ 0 , 1 ] D [0,1]^{D} . Many models use a latent space, which we denote as 𝒵 \mathcal{Z} , where 𝒵 = ℝ d \mathcal{Z}=\mathbb{R}^{d} . In most cases the latent space is low-dimensional, i.e. d D d D , but in some instances this need not hold. We use lower-case letters x ∈ 𝒳 x\in\mathcal{X} and z ∈ 𝒵 z\in\mathcal{Z} to denote points, and upper-case letters X ∈ 𝒳 X\in\mathcal{X} and Z ∈ 𝒵 Z\in\mathcal{Z} for random variables, on these respective spaces.

 
 
 
 Encoders and decoders 

 
 It will often be the case that we consider an encoder and a decoder between the aforementioned spaces, which we denote as f : 𝒳 → 𝒵 f:\mathcal{X}\rightarrow\mathcal{Z} and g : 𝒵 → 𝒳 g:\mathcal{Z}\rightarrow\mathcal{X} , respectively.

 
 
 
 Probability 

 
 We use the letters p p and q q to denote densities; an exception being the Gaussian density, which we denote as 𝒩 ⁡ ( x , μ , Σ ) \mathcal{N}(x;\mu,\Sigma) when evaluated at x x , where μ \mu and Σ \Sigma correspond to its mean and covariance matrix, respectively. We write X ∼ p X\sim p to indicate that X X is distributed according to p p . Since we will often need various densities, we use superindices to identify them, 1 1 
 1 
 
 
 
 That is, we do not use the common overloading of notation where p ⁡ ( x ) p(x) and p ⁡ ( z ) p(z) refer to different densities: in our notation these would correspond to the same density evaluated at two different points. e.g. p X {p}^{X} and p Z {p}^{Z} will denote densities on 𝒳 \mathcal{X} and 𝒵 \mathcal{Z} , respectively. In particular, we write the true data-generating density on 𝒳 \mathcal{X} as p ∗ X {p}^{X}_{\ast} .
For a space 𝒮 \mathcal{S} (e.g. 𝒵 \mathcal{Z} or 𝒳 \mathcal{X} ), we denote the set of all probability distributions on 𝒮 \mathcal{S} as Δ ⁡ ( 𝒮 ) \Delta(\mathcal{S}) . We use p ⊛ q p\circledast q to denote the convolution between the densities p p and q q , i.e. if X 1 ∼ p X_{1}\sim p and X 2 ∼ q X_{2}\sim q are independent, p ⊛ q p\circledast q is the density of X 1 + X 2 X_{1}+X_{2} . We denote expectations with 𝔼 \mathbb{E} , and use a subindex to specify what density the expectation is taken with respect to, e.g. 𝔼 X ∼ p ∗ X ​ [ ⋅ ] \mathbb{E}_{X\sim{p}^{X}_{\ast}}[\cdot] .

 
 
 
 Network parameters 

 
 All DGMs leverage neural networks, in one way or another, to define a density p θ X {p}^{X}_{\theta} parameterized by the learnable parameters θ \theta of the neural network(s). We will often abuse notation and use θ \theta to parameterize all the generative components of the DGM; e.g. some models involve a decoder g θ g_{\theta} and a prior p θ Z {p}^{Z}_{\theta} on 𝒵 \mathcal{Z} , and we frequently subindex both components with θ \theta even if they do not share parameters. In some instances it will be necessary to distinguish between generative parameters, in which case we will explicitly write θ = ( θ 1 , θ 2 ) \theta=(\theta_{1},\theta_{2}) , using e.g. g θ 1 g_{\theta_{1}} and p θ 2 Z {p}^{Z}_{\theta_{2}} instead of g θ g_{\theta} and p θ Z {p}^{Z}_{\theta} , respectively. We will sometimes overload notation by using subindices to denote a sequence of model parameters ( θ t ) t = 1 ∞ (\theta_{t})_{t=1}^{\infty} ; the meaning of parameter subindices will always be clear from context. We use ϕ \phi for all auxiliary parameters; i.e. those that are not needed for generation. For example, an encoder f ϕ f_{\phi} could have been learned alongside p θ Z {p}^{Z}_{\theta} and g θ g_{\theta} , but it might not be needed to sample from the model. We use θ ∗ {\theta^{*}} and ϕ ∗ {\phi^{*}} to denote optimal values of these parameters (with respect to the loss being optimized for the particular model being discussed).

 
 
 
 Calculus 

 
 We denote derivatives (gradients/Jacobians) with ∇ \nabla , and use a subindex to indicate which variable the differentiation is with respect to. For example, ∇ x f ϕ ​ ( x ) \nabla_{x}f_{\phi}(x) and ∇ z g θ ​ ( z ) \nabla_{z}g_{\theta}(z) respectively denote the Jacobians of the encoder and decoder with respect to their inputs (not their parameters), evaluated at x x and z z ; and ∇ θ ​ log ​ p θ X ​ ( x ) \nabla_{\theta}\log{p}^{X}_{\theta}(x) denotes, for a given x x , the gradient of the log density of the model with respect to its parameters, evaluated at θ \theta .

 
 
 
 Linear algebra 

 
 We denote the identity matrix as I I , with a corresponding subindex to indicate dimension. For example, I D I_{D} corresponds to the D × D D\times D identity matrix. For a matrix J J , we use det J \det J , tr ⁡ J \tr J , and J ⊤ J^{\top} to denote its determinant, trace, and transpose, respectively. We denote the ℓ 1 \ell_{1} and ℓ 2 \ell_{2} norms in Euclidean space as ∥ ⋅ ∥ 1 \|\cdot\|_{1} and ∥ ⋅ ∥ 2 \|\cdot\|_{2} , respectively.

 
 
 
 Formal notation  In order to formally discuss probability distributions on manifolds, we need the language of measure theory. All the measures we will consider are defined over 𝒳 \mathcal{X} or 𝒵 \mathcal{Z} , with their respective Borel σ \sigma -algebras. We remind the reader that, since they are Euclidean, 𝒳 \mathcal{X} and 𝒵 \mathcal{Z} are not arbitrary probability spaces. For a subset A A of 𝒳 \mathcal{X} , we denote its closure in 𝒳 \mathcal{X} as cl 𝒳 ⁡ ( A ) \cl_{\mathcal{X}}(A) .
We denote measures with the same letters as densities, but with uppercase blackboard style, i.e. ℙ \mathbb{P} and ℚ \mathbb{Q} – two exceptions being the Lebesgue and Gaussian measures, which we denote as λ \lambda and 𝒩 ⁡ ( μ , Σ ) \mathcal{N}(\mu,\Sigma) , respectively. We use a subindex on λ \lambda to indicate its ambient dimension; e.g. λ D \lambda_{D} denotes the D D -dimensional Lebesgue measure. We write ℙ ≪ ℚ \mathbb{P}\ll\mathbb{Q} to indicate that ℙ \mathbb{P} is absolutely continuous with respect to ℚ \mathbb{Q} . We use # \# as a subscript to denote pushforward measures; e.g. for a measurable function g : 𝒵 → 𝒳 g:\mathcal{Z}\rightarrow\mathcal{X} and a probability measure ℙ Z \mathbb{P}^{Z} on 𝒵 \mathcal{Z} , g # ​ ℙ Z g_{\#}\mathbb{P}^{Z} is the pushforward probability measure (on 𝒳 \mathcal{X} ) of ℙ Z \mathbb{P}^{Z} through g g . We will write the true data-generating distribution corresponding to p ∗ X {p}^{X}_{\ast} as ℙ ∗ X \mathbb{P}^{X}_{*} , and the model distribution corresponding to p θ X {p}^{X}_{\theta} as ℙ θ X \mathbb{P}^{X}_{\theta} . Finally, we use → 𝜔 \xrightarrow{\omega} to indicate weak convergence of probability measures. 
 
 
 
 
 
 
 
 
 
 
 
 Figure 1: (a) Depiction of a full-dimensional density (i.e. a density in the “usual sense”) p X {p}^{X} on ℝ D \mathbb{R}^{D} , with D = 2 D=2 . The probability assigned by p X {p}^{X} to a region A A of ℝ D \mathbb{R}^{D} is its integral over the region, i.e. ∬ A p X ​ ( x 1 , x 2 ) ​ d ​ x 1 ​ d ​ x 2 \iint_{A}{p}^{X}(x_{1},x_{2}){\textnormal{d}}x_{1}{\textnormal{d}}x_{2} . (b) When the density p ∗ X {p}^{X}_{\ast} is instead supported on a d ∗ d^{\ast} -dimensional manifold ℳ \mathcal{M} embedded in ℝ D \mathbb{R}^{D} (here, d ∗ = 1 d^{\ast}=1 and ℳ \mathcal{M} is a curve), the integral evaluates to zero, and thus p ∗ X {p}^{X}_{\ast} is not a density in the “usual sense”. (c) Formally, in order to recover the probability assigned to A A by the manifold-supported density p ∗ X {p}^{X}_{\ast} , the density must be integrated only over ℳ \mathcal{M} using a volume form ( d vol ℳ {\textnormal{d}}\textrm{vol}_{\mathcal{M}} ) on the manifold, ∫ A ∩ ℳ p ∗ X ​ d vol ℳ \int_{A\cap\mathcal{M}}{p}^{X}_{\ast}{\textnormal{d}}\textrm{vol}_{\mathcal{M}} – which in this case simply corresponds to a line integral. 
 
 
 
 

### Section 2.2 Setup

 
 The main goal of all the models p θ X {p}^{X}_{\theta} considered in this paper is to learn p ∗ X {p}^{X}_{\ast} . Two main assumptions, which most of the works presented throughout our survey follow, are commonly made when attempting to understand DGMs through the manifold lens:

 
 
 
 • 
 
 Manifold support  As mentioned in the introduction, there is a substantial body of work supporting the existence of low-dimensional structure in high-dimensional data of interest such as images. One way of mathematically expressing this structure is by taking a literal interpretation of the manifold hypothesis; that is, assuming that p ∗ X {p}^{X}_{\ast} is supported on a d ∗ d^{\ast} -dimensional submanifold ℳ \mathcal{M} of 𝒳 \mathcal{X} , where 0 d ∗ D 0 d^{\ast} D and both ℳ \mathcal{M} and d ∗ d^{\ast} are unknown. Several aspects of this assumption warrant additional discussion. ( i ) (i) p ∗ X {p}^{X}_{\ast} being manifold-supported implies that it is not a full-dimensional density (i.e. with respect to the D D -dimensional Lebesgue measure) and it is thus not a density in the “usual sense”. See Figure 1 for an explanation. ( i ​ i ) (ii) This assumption is a choice about how to represent low-dimensional structure, and can be relaxed in various ways. For example, one could instead assume that the support of p ∗ X {p}^{X}_{\ast} has varying intrinsic dimension ( Brown et al., 2023 ) , that it has singularities ( Von Rohrscheidt Rieck, 2023 ; Wang Wang, 2024 ) , or that p ∗ X {p}^{X}_{\ast} simply concentrates most of its mass around a manifold – rather than all of it ( Divol, 2022 ; Berenfeld et al., 2024 ) . ( i ​ i ​ i ) (iii) The assumption that p ∗ X {p}^{X}_{\ast} is supported exactly on a manifold remains nonetheless very useful, even if we believe it to be “slightly off”; it serves as a first step towards understanding the interplay between DGMs and the low-dimensional structure of the data they are trained on, and as we will see, insights arising from this assumption explain various empirical observations. ( i ​ v ) (iv) The requirement that d ∗ 0 d^{\ast} 0 simply rules out p ∗ X {p}^{X}_{\ast} being a probability mass function, which are better modelled with discrete DGMs. Our main focus is thus on continuous models, although we discuss their discrete counterparts in Section 6 .

 

 • 
 
 Nonparametric regime  We use the term nonparametric regime to refer to the setting where we assume access to an infinite amount of data, arbitrarily flexible models, and exact optimization.
More specifically, training any of the DGMs that we will consider requires minimizing a loss that depends on model parameters and which involves an expectation with respect to p ∗ X {p}^{X}_{\ast} ; doing so is challenging for various reasons. ( i ) (i) In practice, expectations with respect to p ∗ X {p}^{X}_{\ast} cannot be computed, and are thus approximated through empirical averages over the dataset at hand. Writing the losses using expectations with respect to p ∗ X {p}^{X}_{\ast} is thus assuming access to infinite data. ( i ​ i ) (ii) Attempting to reason about optimal parameter values for any given neural network architecture quickly becomes essentially impossible in all but trivial cases. One way to circumvent this issue is to assume that all the neural networks involved, or any density model p θ X {p}^{X}_{\theta} used, can represent any continuous function, or continuous density, respectively. This assumption of arbitrary flexibility is of course motivated by universal approximation properties of neural networks ( Hornik, 1991 ; Koehler et al., 2021 ; Puthawala et al., 2022 ) . ( i ​ i ​ i ) (iii) Stochastic gradient-based optimization ( Robbins Monro, 1951 ) over mini-batches of the non-convex loss is used in practice ( Kingma Ba, 2015 ) , which is not guaranteed to recover a global optimum. Once again to facilitate analysis, it is convenient to assume that all the optimization problems can be solved exactly.

 

 
 
 
 Overall, even though these assumptions are optimistic, they remain popular as they allow for mathematical analysis of DGMs. The nonparametric regime also provides a necessary condition for DGMs to learn p ∗ X {p}^{X}_{\ast} in practice; if a model fails to capture p ∗ X {p}^{X}_{\ast} even in this idealistic regime – which as we will see happens surprisingly often for commonly-used DGMs – then there is no hope that the model can empirically recover p ∗ X {p}^{X}_{\ast} in a realistic setting.

 
 
 
 Formal setup  Throughout our work, ℳ \mathcal{M} will be a d ∗ d^{\ast} -dimensional embedded submanifold of ℝ D \mathbb{R}^{D} . We previously mentioned that we will assume ℙ ∗ X \mathbb{P}^{X}_{*} is supported on ℳ \mathcal{M} , and that it admits a density p ∗ X {p}^{X}_{\ast} ; in this grey box we formalize the meaning of these statements, the latter of which can be formalized either through measure theory or differential geometry (although we will not require this formal understanding of p ∗ X {p}^{X}_{\ast} , we nonetheless include it for completeness): 
 
 • 
 
 Formalizing manifold support  Intuitively, the support of a distribution is the “smallest set of probability 1 1 ”. To formally capture this intuition, the support supp ⁡ ( ℙ ) \supp(\mathbb{P}) of a distribution ℙ \mathbb{P} on 𝒳 \mathcal{X} is defined as ( Bogachev, 2007 , Section 2 of Chapter 7, page 77) 
 

 
 
 supp ⁡ ( ℙ ) ≔ ⋂ C ∈ 𝒞 ⁡ ( ℙ ) C , \supp(\mathbb{P})\coloneqq\bigcap_{C\in\mathcal{C}(\mathbb{P})}C, 
 
 (1) 
 
 where 𝒞 ⁡ ( ℙ ) \mathcal{C}(\mathbb{P}) is the collection of closed (in 𝒳 \mathcal{X} ) sets C C such that ℙ ⁡ ( C ) = 1 \mathbb{P}(C)=1 . It immediately follows that the support of a distribution is always a closed set in its ambient space. In general, ℳ \mathcal{M} need not be closed in 𝒳 \mathcal{X} , in which case it would be impossible for ℙ ∗ X \mathbb{P}^{X}_{*} to be supported on ℳ \mathcal{M} . We nonetheless abuse language throughout our survey (both in the main text and in these formal boxes) and say ℙ ∗ X \mathbb{P}^{X}_{*} is supported on ℳ \mathcal{M} when we actually mean that ℙ ∗ X ​ ( ℳ ) = 1 \mathbb{P}^{X}_{*}(\mathcal{M})=1 and that supp ⁡ ( ℙ ∗ X ) = cl 𝒳 ⁡ ( ℳ ) \supp(\mathbb{P}^{X}_{*})=\cl_{\mathcal{X}}(\mathcal{M}) . 
 
 
 • 
 
 Formalizing manifold-supported densities through measure theory  The manifold ℳ \mathcal{M} can always be equipped with a Riemannian metric which it inherits from 𝒳 \mathcal{X} , making ℳ \mathcal{M} into a Riemannian manifold. Riemannian manifolds admit a unique measure (defined over their Borel sets), λ ℳ \lambda_{\mathcal{M}} , which plays an analogous role to that of λ D \lambda_{D} on ℝ D \mathbb{R}^{D} . The measure λ ℳ \lambda_{\mathcal{M}} is called the Riemannian measure (or sometimes the Lebesgue measure) of ℳ \mathcal{M} . We refer readers interested in Riemannian measures to the treatments by Dieudonné (1973, Section 22 of Chapter 16) and Pennec (2006) . Then ℙ ∗ X \mathbb{P}^{X}_{*} can be restricted to ℳ \mathcal{M} , resulting in the probability measure ℙ X ∗ | ℳ \mathbb{P}^{X}_{*}\arrowvert_{\mathcal{M}} defined over the Borel sets of ℳ \mathcal{M} given by 
 

 
 
 ℙ ∗ X | ℳ ( B ) ≔ ℙ ∗ X ( B ) \mathbb{P}^{X}_{*}\arrowvert_{\mathcal{M}}(B)\coloneqq\mathbb{P}^{X}_{*}(B) 
 
 (2) 
 
 for any Borel set B B of ℳ \mathcal{M} , and p ∗ X {p}^{X}_{\ast} can then be defined as the density (i.e. Radon-Nikodym derivative) – assuming it exists – of this measure with respect to the corresponding Riemannian measure, i.e. 
 

 
 
 p ∗ X ≔ d ℙ X ∗ | ℳ d ​ λ ℳ . {p}^{X}_{\ast}\coloneqq\dfrac{{\textnormal{d}}\mathbb{P}^{X}_{*}\arrowvert_{\mathcal{M}}}{{\textnormal{d}}\lambda_{\mathcal{M}}}. 
 
 (3) 
 
 In other words p ∗ X {p}^{X}_{\ast} is such that, for any Borel set A A of 𝒳 \mathcal{X} , 
 

 
 
 ℙ ∗ X ​ ( A ) = ∫ A ∩ ℳ p ∗ X ​ ( x ) ​ d ​ λ ℳ ​ ( x ) . \mathbb{P}^{X}_{*}(A)=\int_{A\cap\mathcal{M}}{p}^{X}_{\ast}(x){\textnormal{d}}\lambda_{\mathcal{M}}(x). 
 
 (4) 
 
 
 • 
 
 Formalizing manifold-supported densities through differential geometry  When the Riemannian manifold ℳ \mathcal{M} is orientable, the manifold admits a Riemannian volume form d vol ℳ {\textnormal{d}}\textrm{vol}_{\mathcal{M}} , and p ∗ X {p}^{X}_{\ast} is defined as the function – assuming it exists – having the property that 
 

 
 
 ℙ ∗ X ​ ( U ) = ∫ U ∩ ℳ p ∗ X ​ d vol ℳ \mathbb{P}^{X}_{*}(U)=\int_{U\cap\mathcal{M}}{p}^{X}_{\ast}{\textnormal{d}}\textrm{vol}_{\mathcal{M}} 
 
 (5) 
 
 for every open set U U of 𝒳 \mathcal{X} , i.e. d vol ℳ {\textnormal{d}}\textrm{vol}_{\mathcal{M}} “plays the role” of the Riemannian measure λ ℳ \lambda_{\mathcal{M}} in Equation 4 . The technical tool allowing us to establish a correspondence between these two views of p ∗ X {p}^{X}_{\ast} is known as the Riesz-Markov-Kakutani theorem ( Rudin, 1987 , Theorem 2.14) – which is sometimes also referred to as the Riesz representation theorem and should not be confused with a different theorem about Hilbert spaces bearing the same name. The Riesz-Markov-Kakutani theorem allows us to assign a unique measure to d vol ℳ {\textnormal{d}}\textrm{vol}_{\mathcal{M}} , namely λ ℳ \lambda_{\mathcal{M}} , such that integrating continuous compactly-supported functions against them is equivalent. In this sense, λ ℳ \lambda_{\mathcal{M}} is the natural measure to “extend” d vol ℳ {\textnormal{d}}\textrm{vol}_{\mathcal{M}} . We point out that when ℳ \mathcal{M} is non-orientable, even though d vol M {\textnormal{d}}\textrm{vol}_{M} is not defined, integration on manifolds can still be carried out through the above measure-theoretic formulation. 
 
 
 
 
 
 
 

## Section 3 Background

 
 In this section we cover background topics and standard tools which we make use of throughout the survey. Expert readers may wish to skip to Section 4 .

 
 

### Section 3.1 Deep Generative Models on Known Manifolds

 
 This work focuses on data governed by the manifold hypothesis, wherein the dataset of interest is constrained to an unknown d ∗ d^{\ast} -dimensional submanifold of 𝒳 \mathcal{X} . However, for some datasets, the manifold is known a priori , and the challenge lies in designing a generative model which can learn densities within the manifold.
Generative modelling on known manifolds is a distinct task from our focus in this survey, but it is closely related, and we thus briefly summarize work on this topic below.

 
 
 Gemici et al. (2016) first identified deep generative modelling on known manifolds as a problem of interest and showed that the change-of-variables formula ( Section 3.3 ) used to train normalizing flows ( Section 4.1.3 ) can be generalized to account for manifold structure. However, naïvely applying this idea can be numerically unstable, so past work has specifically focused on developing this approach further for manifolds such as tori, spheres, and hyperbolic spaces ( Rezende et al., 2020 ; Bose et al., 2020 ; Sorrenson et al., 2023 ) . Other works have designed generative models which preserve the symmetries of data on certain manifolds (Lie groups) of interest in the natural sciences ( Kanwar et al., 2020 ; Boyda et al., 2021 ; Katsman et al., 2021 ) .
Using ordinary differential equations or stochastic differential equations to model data on known manifolds has also been proven to be effective ( Mathieu Nickel, 2020 ; Rozen et al., 2021 ; De Bortoli et al., 2022 ; Ben-Hamu et al., 2022 ; Lou et al., 2023 ; Chen Lipman, 2024 ) ,
with the advantage that these approaches can be defined independently of any parameterization of the manifold. Bonet et al. (2024) recently proposed an approach based on optimal transport ( Section 3.5 ) for generative modelling on known manifolds.

 
 
 

### Section 3.2 Manifold Learning

 
 Learning distributions whose support is an unknown manifold ℳ \mathcal{M} implies learning ℳ \mathcal{M} as well, at least implicitly. The field of manifold learning is thus closely related to the main topic of our work. The term “manifold learning” is often treated synonymously with dimensionality reduction, and refers to methods whose goal is to provide a useful representation of high-dimensional data by transforming it into a lower-dimensional space. That representation may provide information about the data such as its intrinsic dimension, yield a useful visualization in two or three dimensions, or serve as a simplified starting point for downstream supervised learning tasks. Generally, manifold learning methods fall into three categories: spectral methods ( Pearson, 1901 ; Kruskal, 1964 ; Beals et al., 1968 ; Schölkopf et al., 1998 ; Roweis Saul, 2000 ; Tenenbaum et al., 2000 ) which rely on the eigenvectors and eigenvalues of some matrix related to the data; probabilistic methods ( Tipping Bishop, 1999 ; van der Maaten Hinton, 2008 ; McInnes et al., 2018 ) , which treat datapoints as high-dimensional random vectors whose relevant information is contained in some low-dimensional latent variables; and bottleneck methods ( Rumelhart et al., 1988 ; Kramer, 1991 ; Tishby et al., 2000 ; Kingma Welling, 2014 ; Alemi et al., 2017 ) , which rely on passing information through a low-dimensional bottleneck representation, often using neural networks. We refer the reader to the work of Ghojogh et al. (2023) for a comprehensive review of manifold learning using this tripartite classification.

 
 
 The focus on manifold learning in this work is mostly on bottleneck methods such as autoencoders ( Rumelhart et al., 1988 ) – and their many variants – whose core idea is to train an encoder-decoder pair of neural networks to ensure reconstruction, for example through a squared ℓ 2 \ell_{2} loss: 2 2 
 2 
 
 
 
 Note that autoencoders are not by themselves generative models, but we nonetheless parameterize g g with θ \theta (which we use for generative parameters) for consistency with other decoders in the rest of the paper. 

 

 
 | 
 min θ , ϕ ⁡ 𝔼 X ∼ p ∗ X ​ [ ‖ X − g θ ​ ( f ϕ ​ ( X ) ) ‖ 2 2 ] . \min_{\theta,\phi}\mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[\|X-g_{\theta}\left(f_{\phi}(X)\right)\|_{2}^{2}\right]. | 
 | 
 (6) | 
 

 Here, the “bottleneck” refers to the d d -dimensional encoder output f ϕ ​ ( X ) f_{\phi}(X) , which the decoder g θ g_{\theta} uses to reconstruct X X . Since p ∗ X {p}^{X}_{\ast} is supported on ℳ \mathcal{M} , the objective in Equation 6 aims to learn the manifold in the sense that it encourages perfect reconstructions on it, or more formally, if x ∈ ℳ x\in\mathcal{M} , then x = g θ ∗ ​ ( f ϕ ∗ ​ ( x ) ) x=g_{\theta^{*}}(f_{\phi^{*}}(x)) . We highlight that perfect reconstructions need not always be achievable – even under the nonparametric regime ( Section 2.2 ) – due to topological constraints. We discuss this point, which we will revisit in Section 5.4 , in the next grey box.

 
 
 We finish this section with the observation that, despite what the term “manifold learning” might suggest, autoencoders do not by themselves formally learn ℳ \mathcal{M} – even when perfect reconstructions are achieved. One might expect a perfectly trained autoencoder to characterize ℳ \mathcal{M} as the set of possible decoder outputs. However, the encoder may not make use of the entire latent space, leaving some points that would not be seen by the decoder during training. As a result, points z ∈ 𝒵 ∖ f ϕ ∗ ​ ( ℳ ) z\in\mathcal{Z}\setminus f_{\phi^{\ast}}(\mathcal{M}) might be decoded outside of ℳ \mathcal{M} , as illustrated in Figure 2 . Alternatively, one might expect an autoencoder to characterize ℳ \mathcal{M} as the set of points which are perfectly reconstructed. However, some points outside the manifold can in principle still be perfectly reconstructed, as also illustrated in Figure 2 . Despite these observations, assuming perfect reconstructions, the set f ϕ ∗ ​ ( ℳ ) f_{\phi^{\ast}}(\mathcal{M}) together with the decoder g θ ∗ g_{\theta^{\ast}} jointly characterize ℳ \mathcal{M} , since g θ ∗ ​ ( f ϕ ∗ ​ ( ℳ ) ) = ℳ g_{\theta^{\ast}}(f_{\phi^{\ast}}(\mathcal{M}))=\mathcal{M} ; we will revisit this point in Section 5.3 to show that training a DGM on data encoded by f ϕ ∗ f_{\phi^{*}} can characterize ℳ \mathcal{M} .

 
 
 
 
 When are perfect reconstructions achievable? 

 
 Ideally, the loss in Equation 6 achieves a value of 0 0 at optimality, which would directly imply that x = g θ ∗ ​ ( f ϕ ∗ ​ ( x ) ) , ℙ ∗ X ​ -almost-surely x=g_{\theta^{*}}(f_{\phi^{*}}(x)),\mathbb{P}^{X}_{*}\text{-almost-surely} . Under mild regularity conditions ( Loaiza-Ganem et al., 2022a ) , this in turn implies that x = g θ ∗ ​ ( f ϕ ∗ ​ ( x ) ) x=g_{\theta^{*}}(f_{\phi^{*}}(x)) for all x ∈ ℳ x\in\mathcal{M} , in which case we say that the encoder-decoder pair ( OPEN f ϕ ∗ , g θ ∗ ) f_{\phi^{*}},g_{\theta^{*}}) reconstructs ℳ \mathcal{M} perfectly. When this condition is satisfied, the restriction f ϕ ∗ | ℳ f_{\phi^{*}}|_{\mathcal{M}} is a (topological) embedding of ℳ \mathcal{M} into 𝒵 \mathcal{Z} , as is evidenced by the existence of its continuous left-inverse, g θ ∗ g_{\theta^{*}} . In other words, the existence of some continuous function f f that embeds ℳ \mathcal{M} into 𝒵 \mathcal{Z} is a necessary condition to achieve perfect reconstructions and thus to learn ℳ \mathcal{M} . 
 
 
 
 In general, however, such an f f may not exist. For example, as we will see in Section 5.3 , it is sometimes desirable to set d d , the dimensionality of 𝒵 \mathcal{Z} , to be equal to d ∗ d^{\ast} , the dimensionality of ℳ \mathcal{M} . This precludes the existence of f f for many manifolds ℳ \mathcal{M} , such as if ℳ \mathcal{M} is a d ∗ d^{\ast} -dimensional sphere, for which no embedding f : ℳ → ℝ d f:\mathcal{M}\to\mathbb{R}^{d} is possible when d = d ∗ d=d^{\ast} . In cases where no plausible embedding exists, even networks ( f ϕ , g θ ) (f_{\phi},g_{\theta}) which come close to perfectly reconstructing ℳ \mathcal{M} will incur numerical instability ( Cornish et al., 2020 ) . In some other cases, it is possible to resolve these topological issues by increasing d d . For instance, a dimensionality of d = 2 ​ d ∗ + 1 d=2d^{\ast}+1 is enough to topologically embed any manifold of dimension d ∗ d^{\ast} in ℝ d \mathbb{R}^{d} ( Hurewicz Wallman, 1948 , Theorem V 3) . 
 
 
 
 
 
 Figure 2: Illustration of why autoencoders, by themselves, do not characterize ℳ \mathcal{M} even if they achieve perfect reconstructions on it. The illustrative point z ∈ 𝒵 ∖ f ϕ ∗ ​ ( ℳ ) z\in\mathcal{Z}\setminus f_{\phi^{\ast}}(\mathcal{M}) is such that x = g θ ∗ ​ ( z ) ∉ ℳ x=g_{\theta^{\ast}}(z)\notin\mathcal{M} , so that the set of possible decoder outputs does not match ℳ \mathcal{M} , i.e. g θ ∗ ​ ( 𝒵 ) ≠ ℳ g_{\theta^{\ast}}(\mathcal{Z})\neq\mathcal{M} – even though ℳ \mathcal{M} is contained in g θ ∗ ​ ( 𝒵 ) g_{\theta^{\ast}}(\mathcal{Z}) due to the assumption of perfect reconstructions. Additionally, in this example x ∈ 𝒳 ∖ ℳ x\in\mathcal{X}\setminus\mathcal{M} is perfectly reconstructed, so that the set of perfectly reconstructed points does not match ℳ \mathcal{M} , i.e. { x ∈ 𝒳 ∣ x = g θ ∗ ​ ( f ϕ ∗ ​ ( x ) ) } ≠ ℳ \{x\in\mathcal{X}\mid x=g_{\theta^{\ast}}(f_{\phi^{\ast}}(x))\}\neq\mathcal{M} – even though ℳ \mathcal{M} is contained in this set whenever x = g θ ∗ ​ ( f ϕ ∗ ​ ( x ) ) x=g_{\theta^{\ast}}(f_{\phi^{\ast}}(x)) for every x ∈ ℳ x\in\mathcal{M} . 
 
 
 

### Section 3.3 The Change-of-Variables Formula

 
 It will often be the case that we have a density p Z {p}^{Z} on 𝒵 \mathcal{Z} along with a decoder g : 𝒵 → 𝒳 g:\mathcal{Z}\rightarrow\mathcal{X} . Together, these two components implicitly define the distribution of X = g ⁡ ( Z ) X=g(Z) , where Z ∼ p Z Z\sim{p}^{Z} , and it will often be of interest to explicitly evaluate the density p X {p}^{X} of X X (formally, p X {p}^{X} is the pushforward density of p Z {p}^{Z} through g g ).
The suitable tool is the change-of-variables formula, whose simplest form states that, when 𝒵 = 𝒳 = ℝ D \mathcal{Z}=\mathcal{X}=\mathbb{R}^{D} , if g g is a diffeomorphism (i.e. a continuously differentiable function with a continuously differentiable inverse), then

 

 
 | 
 p X ​ ( x ) \displaystyle{p}^{X}(x) | 
 = p Z ​ ( z ) ​ | det ∇ z g ​ ( z ) | − 1 \displaystyle={p}^{Z}(z)\left|\det\nabla_{z}g(z)\right|^{-1} | 
 | 
 (7) | 
 
 
 | 
 | 
 = p Z ​ ( f ⁡ ( x ) ) ​ | det ∇ x f ​ ( x ) | , \displaystyle={p}^{Z}\left(f(x)\right)\left|\det\nabla_{x}f(x)\right|, | 
 | 
 (8) | 
 

 where f = g − 1 f=g^{-1} and z = f ⁡ ( x ) z=f(x) , so that ∇ z g ​ ( z ) ∈ ℝ D × D \nabla_{z}g(z)\in\mathbb{R}^{D\times D} and ∇ x f ​ ( x ) ∈ ℝ D × D \nabla_{x}f(x)\in\mathbb{R}^{D\times D} .
An extension of this formula that will also be of use applies to the case where 𝒵 = ℝ d \mathcal{Z}=\mathbb{R}^{d} with d ≤ D d\leq D . When d D d D , g : 𝒵 → 𝒳 g:\mathcal{Z}\rightarrow\mathcal{X} cannot be a diffeomorphism, but if it is injective it can be a diffeomorphism onto its image, g ⁡ ( 𝒵 ) g(\mathcal{Z}) , in which case the density of X X is now given by

 

 
 | 
 p X ​ ( x ) = p Z ​ ( z ) ​ | det ( ∇ z g ​ ( z ) ⊤ ​ ∇ z g ​ ( z ) ) | − 1 2 , {p}^{X}(x)={p}^{Z}(z)\left|\det\left(\nabla_{z}g(z)^{\top}\nabla_{z}g(z)\right)\right|^{-\frac{1}{2}}, | 
 | 
 (9) | 
 

 where again z = f ⁡ ( x ) z=f(x) , but now f : g ⁡ ( 𝒵 ) → 𝒵 f:g(\mathcal{Z})\rightarrow\mathcal{Z} is the left inverse of g g (i.e. z ′ = f ⁡ ( g ⁡ ( z ′ ) ) z^{\prime}=f(g(z^{\prime})) for all z ′ ∈ 𝒵 z^{\prime}\in\mathcal{Z} ), and ∇ z g ​ ( z ) ∈ ℝ D × d \nabla_{z}g(z)\in\mathbb{R}^{D\times d} .

 
 
 Several remarks about these formulas are worth making. ( i ) (i) Computationally, it is often the case that Equation 8 is used, rather than Equation 7 , as Equation 8 requires only a forward pass through the encoder f f (as well as computing its Jacobian determinant), whereas Equation 7 requires an additional forward computation through the decoder g g ; ( i ​ i ) (ii) Equation 9 , which is referred to as the injective change-of-variables formula, reduces to Equation 7 in the case where d = D d=D , since the determinant distributes over products of square matrices; and ( i ​ i ​ i ) (iii) when d D d D , p X {p}^{X} in Equation 9 is a manifold-supported density because it is only defined on a submanifold, g ⁡ ( 𝒵 ) g(\mathcal{Z}) , of 𝒳 \mathcal{X} , much like p ∗ X {p}^{X}_{\ast} which is only defined on ℳ \mathcal{M} . We refer the reader to the work of Köthe (2023) for a review of the uses the change-of-variables formula has within DGMs.

 
 
 
 Note that a more formal way of describing the change-of-variables formula is through the language of pushforward measures, where we have a measure ℙ Z \mathbb{P}^{Z} on 𝒵 \mathcal{Z} admitting a density p Z {p}^{Z} with respect to λ d \lambda_{d} , along with the measurable map g : 𝒵 → 𝒳 g:\mathcal{Z}\rightarrow\mathcal{X} . In the case of Equation 7 and Equation 8 , p X {p}^{X} corresponds to the density of g # ​ ℙ Z g_{\#}\mathbb{P}^{Z} with respect to λ D \lambda_{D} ; i.e. p X = d ​ g # ​ ℙ Z / d ​ λ D {p}^{X}={\textnormal{d}}g_{\#}\mathbb{P}^{Z}/{\textnormal{d}}\lambda_{D} . In the case of Equation 9 , when g g is a smooth embedding, p X {p}^{X} is now a density with respect to the Riemannian measure on g ⁡ ( 𝒵 ) g(\mathcal{Z}) , i.e. p X = d ​ g # ​ ℙ Z / d ​ λ g ⁡ ( 𝒵 ) {p}^{X}={\textnormal{d}}g_{\#}\mathbb{P}^{Z}/{\textnormal{d}}\lambda_{g(\mathcal{Z})} , where g ⁡ ( 𝒵 ) g(\mathcal{Z}) is treated as an embedded submanifold of 𝒳 \mathcal{X} . 
 
 
 
 

### Section 3.4 Failures of KL Divergence

 
 Despite the KL divergence being widely used throughout machine learning, it is most commonly used with the implicit assumption that the two involved densities are densities in the “same sense” (i.e. when they both admit the same dominating measure, see the grey box below). This assumption fails in the manifold setting when p θ X {p}^{X}_{\theta} is full-dimensional, since p ∗ X {p}^{X}_{\ast} is manifold-supported. We thus find it useful to provide the formal definition of the KL divergence in the grey box below, along with a discussion. In summary, the usual formula for computing KL divergence,

 

 
 | 
 𝕂 𝕃 ( p ∗ X ∥ p θ X ) = 𝔼 X ∼ p ∗ X [ log p ∗ X ​ ( X ) p θ X ​ ( X ) ] , \mathbb{KL}\left({p}^{X}_{\ast}\,\|\,{p}^{X}_{\theta}\right)=\mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[\log\dfrac{{p}^{X}_{\ast}(X)}{{p}^{X}_{\theta}(X)}\right], | 
 | 
 (10) | 
 

 is only valid when p ∗ X {p}^{X}_{\ast} and p θ X {p}^{X}_{\theta} are such that for every subset A A of 𝒳 \mathcal{X} that is assigned probability 0 0 by p θ X {p}^{X}_{\theta} , the density p ∗ X {p}^{X}_{\ast} also assigns probability 0 0 to A A . Whenever this property does not hold, 𝕂 𝕃 ( p ∗ X ∥ p θ X ) \mathbb{KL}({p}^{X}_{\ast}\,\|\,{p}^{X}_{\theta}) is defined as infinity. It follows that in the manifold setting, 𝕂 𝕃 ( p ∗ X ∥ p θ X ) = ∞ = 𝕂 𝕃 ( p θ X ∥ p ∗ X ) \mathbb{KL}({p}^{X}_{\ast}\,\|\,{p}^{X}_{\theta})=\infty=\mathbb{KL}({p}^{X}_{\theta}\,\|\,{p}^{X}_{\ast}) when p θ X {p}^{X}_{\theta} is full-dimensional, as illustrated in Figure 3 . It also follows that 𝕂 𝕃 ( p ∗ X ∥ p θ X ) = ∞ \mathbb{KL}({p}^{X}_{\ast}\,\|\,{p}^{X}_{\theta})=\infty even if p θ X {p}^{X}_{\theta} is supported on a d ∗ d^{\ast} -dimensional manifold – as long as ℳ \mathcal{M} is not contained in the support of p θ X {p}^{X}_{\theta} – as illustrated in Figure 3 .

 
 
 KL divergence and maximum-likelihood 

 
 The most common way of attempting to minimize the KL divergence between the true distribution and the model is through maximum-likelihood:

 

 
 | 
 max θ ⁡ 𝔼 X ∼ p ∗ X ​ [ log ⁡ p θ X ​ ( X ) ] . \max_{\theta}\mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[\log{p}^{X}_{\theta}(X)\right]. | 
 | 
 (11) | 
 

 When the KL divergence between p ∗ X {p}^{X}_{\ast} and p θ X {p}^{X}_{\theta} is not trivially equal to infinity, it can be written as

 

 
 | 
 𝕂 𝕃 ( p ∗ X ∥ p θ X ) \displaystyle\mathbb{KL}({p}^{X}_{\ast}\,\|\,{p}^{X}_{\theta}) | 
 = ∫ log ⁡ ( p ∗ X ​ ( x ) p θ X ​ ( x ) ) ​ p ∗ X ​ ( x ) ​ d ​ x = ∫ p ∗ X ​ ( x ) ​ log ​ p ∗ X ​ ( x ) ​ d ​ x − ∫ p ∗ X ​ ( x ) ​ log ​ p θ X ​ ( x ) ​ d ​ x \displaystyle=\displaystyle\int\log\left(\dfrac{{p}^{X}_{\ast}(x)}{{p}^{X}_{\theta}(x)}\right){p}^{X}_{\ast}(x){\textnormal{d}}x=\int{p}^{X}_{\ast}(x)\log{p}^{X}_{\ast}(x){\textnormal{d}}x-\int{p}^{X}_{\ast}(x)\log{p}^{X}_{\theta}(x){\textnormal{d}}x | 
 | 
 (12) | 
 
 
 | 
 | 
 = 𝔼 X ∼ p ∗ X ​ [ log ⁡ p ∗ X ​ ( X ) ] − 𝔼 X ∼ p ∗ X ​ [ log ⁡ p θ X ​ ( X ) ] . \displaystyle=\mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[\log{p}^{X}_{\ast}(X)\right]-\mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[\log{p}^{X}_{\theta}(X)\right]. | 
 | 
 (13) | 
 

 Since 𝔼 X ∼ p ∗ X ​ [ log ⁡ p ∗ X ​ ( X ) ] \mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[\log{p}^{X}_{\ast}(X)\right] does not depend on θ \theta , this common derivation shows that as long as | 𝔼 X ∼ p ∗ X ​ [ log ⁡ p ∗ X ​ ( X ) ] | ∞ |\mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[\log{p}^{X}_{\ast}(X)\right]| \infty , maximum-likelihood optimization is equivalent to minimizing KL divergence. However, a key step in this derivation (the first equality) is the assumption that the KL divergence between p ∗ X {p}^{X}_{\ast} and p θ X {p}^{X}_{\theta} is not trivially infinite. As previously mentioned, in the manifold setting we will generally have 𝕂 𝕃 ( p ∗ X ∥ p θ X ) = ∞ \mathbb{KL}({p}^{X}_{\ast}\,\|\,{p}^{X}_{\theta})=\infty . It follows that in this setting, maximum-likelihood is not equivalent to KL divergence minimization, a point that we will later revisit.

 
 
 
 
 
 
 
 
 Figure 3: Illustration of why KL divergences can be infinite in the manifold setting. (a) p θ X {p}^{X}_{\theta} has full-dimensional support (light red region), while p ∗ X {p}^{X}_{\ast} is supported on a lower-dimensional manifold ℳ \mathcal{M} (blue curve). The model p θ X {p}^{X}_{\theta} assigns probability 0 0 to A A , i.e. ∫ A p θ X ​ d ​ x = 0 \int_{A}{p}^{X}_{\theta}{\textnormal{d}}x=0 , because the region A A has zero volume in 𝒳 \mathcal{X} . However, p ∗ X {p}^{X}_{\ast} does not, since ∫ A ∩ ℳ p ∗ X ​ d vol ℳ 0 \int_{A\cap\mathcal{M}}{p}^{X}_{\ast}{\textnormal{d}}\textrm{vol}_{\mathcal{M}} 0 . We conclude that 𝕂 𝕃 ( p ∗ X ∥ p θ X ) = ∞ \mathbb{KL}\left({p}^{X}_{\ast}\,\|\,{p}^{X}_{\theta}\right)=\infty . Meanwhile, ∫ B ∩ ℳ p ∗ X ​ d vol ℳ = 0 \int_{B\cap\mathcal{M}}{p}^{X}_{\ast}{\textnormal{d}}\textrm{vol}_{\mathcal{M}}=0 because B ∩ ℳ = ∅ B\cap\mathcal{M}=\emptyset , yet we have ∫ B p θ X ​ d ​ x 0 \int_{B}{p}^{X}_{\theta}{\textnormal{d}}x 0 , entailing that 𝕂 𝕃 ( p θ X ∥ p ∗ X ) = ∞ \mathbb{KL}\left({p}^{X}_{\theta}\,\|\,{p}^{X}_{\ast}\right)=\infty . (b) Analogous example where now p θ X {p}^{X}_{\theta} and p ∗ X {p}^{X}_{\ast} are both supported on low-dimensional manifolds. Since ℳ \mathcal{M} is not contained in the support of p θ X {p}^{X}_{\theta} , there exists a set A A to which p θ X {p}^{X}_{\theta} assigns probability 0 0 despite having positive probability under p ∗ X {p}^{X}_{\ast} , so that 𝕂 𝕃 ( p ∗ X ∥ p θ X ) = ∞ \mathbb{KL}\left({p}^{X}_{\ast}\,\|\,{p}^{X}_{\theta}\right)=\infty . 
 
 
 
 Formally, the KL divergence between two probability measures ℙ \mathbb{P} and ℚ \mathbb{Q} , 𝕂 𝕃 ( ℙ ∥ ℚ ) \mathbb{KL}(\mathbb{P}\,\|\,\mathbb{Q}) , is defined as 
 

 
 
 𝕂 𝕃 ( ℙ ∥ ℚ ) ≔ { ∫ log ⁡ ( d ​ ℙ d ​ ℚ ​ ( x ) ) ​ d ​ ℙ ​ ( x ) ,  if  ​ d ​ ℙ d ​ ℚ ​  exists ∞ ,  otherwise , \mathbb{KL}(\mathbb{P}\,\|\,\mathbb{Q})\coloneqq\begin{cases}\displaystyle\int\log\left(\dfrac{{\textnormal{d}}\mathbb{P}}{{\textnormal{d}}\mathbb{Q}}(x)\right){\textnormal{d}}\mathbb{P}(x),\text{ if }\dfrac{{\textnormal{d}}\mathbb{P}}{{\textnormal{d}}\mathbb{Q}}\text{ exists}\\
\infty,\text{ otherwise}\end{cases}, 
 
 (14) 
 
 where d ​ ℙ / d ​ ℚ {\textnormal{d}}\mathbb{P}/{\textnormal{d}}\mathbb{Q} denotes the Radon-Nikodym derivative of ℙ \mathbb{P} with respect to ℚ \mathbb{Q} . By the Radon-Nikodym theorem, d ​ ℙ / d ​ ℚ {\textnormal{d}}\mathbb{P}/{\textnormal{d}}\mathbb{Q} exists if and only if ℙ ≪ ℚ \mathbb{P}\ll\mathbb{Q} . Finally, when both ℙ \mathbb{P} and ℚ \mathbb{Q} are dominated by the same measure η \eta – i.e. ℙ ≪ η \mathbb{P}\ll\eta and ℚ ≪ η \mathbb{Q}\ll\eta – with corresponding densities p p and q q with respect to η \eta , the KL divergence between them simplifies to 
 

 
 
 𝕂 𝕃 ( ℙ ∥ ℚ ) = 𝕂 𝕃 ( p ∥ q ) = ∫ log ( p ⁡ ( x ) q ⁡ ( x ) ) p ( x ) d η ( x ) , \mathbb{KL}(\mathbb{P}\,\|\,\mathbb{Q})=\mathbb{KL}(p\,\|\,q)=\displaystyle\int\log\left(\dfrac{p(x)}{q(x)}\right)p(x){\textnormal{d}}\eta(x), 
 
 (15) 
 
 which recovers the commonly-used expressions for KL divergence ( Equation 13 ) whenever η \eta is either the Lebesgue measure or the counting measure. 
 
 
 
 Figure 4: The optimal transport problem can be visualized as the minimum cost of “transporting” the density p p over to the density q q . Picturing p p and q q as piles of dirt, each dirt particle from p p must be moved so that it becomes part of q q . Moving dirt from x x to y y incurs a cost given by c ⁡ ( x , y ) c(x,y) . The joint distribution γ \gamma of ( X , Y ) (X,Y) can be thought of as specifying the “transport plan”: the constraint that its X X -marginal matches p p ensures the starting pile of dirt is p p ; the constraint that its Y Y -marginal matches q q ensures the final pile of dirt is q q ; and its ( Y | X = x ) (Y|X=x) -conditional – illustrated with the black arrows in the figure – specifies how the dirt at x x from p p is (potentially stochastically) allocated to dirt from q q . The most efficient plan possible for shifting all the dirt has an overall cost 𝕎 c ​ ( p , q ) \mathbb{W}^{c}(p,q) . This analogy explains why the Wasserstein-1 distance is sometimes called the earth mover’s distance. 
 
 
 
 

### Section 3.5 Wasserstein Distances

 
 In Section 3.4 , we summarized why 𝕂 𝕃 ( p ∗ X ∥ p θ X ) \mathbb{KL}({p}^{X}_{\ast}\,\|\,{p}^{X}_{\theta}) does not provide a useful notion of divergence between p ∗ X {p}^{X}_{\ast} and p θ X {p}^{X}_{\theta} in the manifold setting, and the same is true of many other common divergences between distributions (see the discussion in Section 4.2 ).
Wasserstein distances, which are based on the optimal transport problem ( Villani, 2009 ; Peyré Cuturi, 2019 ) , provide a distance between distributions that remains meaningful even in the manifold setting. Despite the fact that accurately estimating Wasserstein distances is challenging ( Arora et al., 2017 ) , DGMs based on minimizing these distances tend to work very well in practice (e.g. Section 5.2.1 and Section 5.3.1 ).

 
 
 The optimal transport problem between two densities p p and q q on 𝒳 \mathcal{X} is given by

 

 
 | 
 𝕎 c ​ ( p , q ) ≔ inf γ ∈ Π ⁡ ( p , q ) 𝔼 ( X , Y ) ∼ γ ​ [ c ⁡ ( X , Y ) ] , \mathbb{W}^{c}(p,q)\coloneqq\inf_{\gamma\in\Pi(p,q)}\mathbb{E}_{(X,Y)\sim\gamma}[c(X,Y)], | 
 | 
 (16) | 
 

 where c : 𝒳 × 𝒳 → ℝ c:\mathcal{X}\times\mathcal{X}\rightarrow\mathbb{R} is called the cost function, and Π ⁡ ( p , q ) \Pi(p,q) is the set of distributions on 𝒳 × 𝒳 \mathcal{X}\times\mathcal{X} whose marginals match p p and q q , respectively.
Intuitively, the optimal transport problem can be understood as the cost (as measured by c c ) of “transporting” p p to q q , as illustrated in Figure 4 .
When c c is given by the ℓ 1 \ell_{1} distance (i.e. c ⁡ ( x , y ) = ‖ x − y ‖ 1 c(x,y)=\|x-y\|_{1} ) 𝕎 c \mathbb{W}^{c} is called the Wasserstein-1 distance, and is denoted as 𝕎 1 \mathbb{W}_{1} . The 𝕎 1 \mathbb{W}_{1} metric admits the following well-known dual formulation:

 

 
 | 
 𝕎 1 ​ ( p , q ) = sup h ∈ ℋ 𝔼 X ∼ p ​ [ h ⁡ ( X ) ] − 𝔼 X ∼ q ​ [ h ⁡ ( X ) ] , \mathbb{W}_{1}(p,q)=\sup_{h\in\mathcal{H}}\mathbb{E}_{X\sim p}[h(X)]-\mathbb{E}_{X\sim q}[h(X)], | 
 | 
 (17) | 
 

 where ℋ ≔ { h : 𝒳 → ℝ ∣ h  is Lipschitz and  Lip ( h ) ≤ 1 } \mathcal{H}\coloneqq\{h:\mathcal{X}\rightarrow\mathbb{R}\mid h\text{ is Lipschitz and }\text{Lip}(h)\leq 1\} , and Lip ​ ( h ) \text{Lip}(h) denotes the Lipschitz constant of h h . Analogously, when c c is given by the squared ℓ 2 \ell_{2} distance (i.e. c ⁡ ( x , y ) = ‖ x − y ‖ 2 2 c(x,y)=\|x-y\|_{2}^{2} ) 𝕎 c \sqrt{\mathbb{W}^{c}} is called the Wasserstein-2 distance, and is denoted as 𝕎 2 \mathbb{W}_{2} . The Wasserstein distances 𝕎 1 ​ ( p ∗ X , p θ X ) \mathbb{W}_{1}({p}^{X}_{\ast},{p}^{X}_{\theta}) and 𝕎 2 ​ ( p ∗ X , p θ X ) \mathbb{W}_{2}({p}^{X}_{\ast},{p}^{X}_{\theta}) remain meaningfully defined even in the manifold setting (formally, this is because they metrize weak convergence, which we discuss in the grey box below), and thus provide sensible optimization objectives. For 𝕎 1 \mathbb{W}_{1} , this property can be informally understood through Equation 17 , which essentially says that two distributions are close in Wasserstein distance if no Lipschitz function can discriminate between them. Intuitively, if no such function can discern between p ∗ X {p}^{X}_{\ast} and p θ X {p}^{X}_{\theta} , then they must be “truly” close, even if one is manifold-supported and the other full-dimensional (or if both are supported on non-overlapping manifolds).

 
 
 
 First, we point out that the optimal transport problem in Equation 16 applies to arbitrary probability measures ℙ \mathbb{P} and ℚ \mathbb{Q} on 𝒳 \mathcal{X} , not only densities: 
 

 
 
 𝕎 c ​ ( ℙ , ℚ ) ≔ inf Γ ∈ Π ⁡ ( ℙ , ℚ ) 𝔼 ( X , Y ) ∼ γ ​ [ c ⁡ ( X , Y ) ] , \mathbb{W}^{c}(\mathbb{P},\mathbb{Q})\coloneqq\inf_{\Gamma\in\Pi(\mathbb{P},\mathbb{Q})}\mathbb{E}_{(X,Y)\sim\gamma}[c(X,Y)], 
 
 (18) 
 
 where Π ( ℙ , ℚ ) ≔ { Γ ∈ Δ ( 𝒳 × 𝒳 ) ∣ Γ ( A × 𝒳 ) = ℙ ( A ) \Pi(\mathbb{P},\mathbb{Q})\coloneqq\{\Gamma\in\Delta(\mathcal{X}\times\mathcal{X})\mid\Gamma(A\times\mathcal{X})=\mathbb{P}(A) and Γ ⁡ ( 𝒳 × A ) = ℚ ⁡ ( A ) \Gamma(\mathcal{X}\times A)=\mathbb{Q}(A) for every measurable set A ⊂ 𝒳 } A\subset\mathcal{X}\} , and c : 𝒳 × 𝒳 → ℝ c:\mathcal{X}\times\mathcal{X}\rightarrow\mathbb{R} is measurable. 
 
 If c c is the ℓ 1 \ell_{1} (or ℓ 2 \ell_{2} ) distance, then convergence in 𝕎 c \mathbb{W}^{c} is equivalent to convergence in distribution plus convergence in first (or second) moments ( Villani, 2009 , Theorem 6.9) .
In particular, this implies that if 𝒳 \mathcal{X} is bounded (and c c is either the ℓ 1 \ell_{1} or ℓ 2 \ell_{2} distance), then 𝕎 c \mathbb{W}^{c} metrizes weak convergence, meaning that given a sequence ( ℙ θ t X ) t = 1 ∞ (\mathbb{P}^{X}_{\theta_{t}})_{t=1}^{\infty} of probability measures, 𝕎 c ​ ( ℙ θ t X , ℙ ∗ X ) → 0 \mathbb{W}^{c}(\mathbb{P}^{X}_{\theta_{t}},\mathbb{P}^{X}_{*})\rightarrow 0 as t → ∞ t\rightarrow\infty if and only if ℙ θ t X → 𝜔 ℙ ∗ X \mathbb{P}^{X}_{\theta_{t}}\xrightarrow{\omega}\mathbb{P}^{X}_{*} as t → ∞ t\rightarrow\infty . Arjovsky et al. (2017) identified that metrizing weak convergence is a desirable property in an optimization objective for training DGMs, as it ensures that “getting closer and closer” to the target distribution is properly quantified, even in the presence of dimensionality mismatch (see Appendix A for a more detailed discussion of this point).
 
 Note that the KL divergence does not metrize weak convergence. Let us illustrate why this is problematic through an example by letting ℙ ∗ X σ = ℙ ∗ X ⊛ 𝒩 ⁡ ( 0 , σ 2 ​ I D ) \mathbb{P}^{X_{\sigma}}_{\ast}=\mathbb{P}^{X}_{*}\circledast\mathcal{N}(0,\sigma^{2}I_{D}) , where ℙ ∗ X \mathbb{P}^{X}_{*} is supported on ℳ \mathcal{M} . As σ → 0 + \sigma\rightarrow 0^{+} , ℙ ∗ X σ \mathbb{P}^{X_{\sigma}}_{\ast} gets closer to ℙ ∗ X \mathbb{P}^{X}_{*} , yet this is not reflected in the KL divergence, since 𝕂 𝕃 ( ℙ ∗ X ∥ ℙ ∗ X σ ) = ∞ \mathbb{KL}(\mathbb{P}^{X}_{*}\,\|\,\mathbb{P}^{X_{\sigma}}_{\ast})=\infty for every σ 0 \sigma 0 (this is because ℙ ∗ X ≪ ℙ ∗ X σ \mathbb{P}^{X}_{*}\ll\mathbb{P}^{X_{\sigma}}_{\ast} does not hold). Thus 𝕂 𝕃 ( ℙ ∗ X ∥ ℙ ∗ X σ ) → ∞ \mathbb{KL}(\mathbb{P}^{X}_{*}\,\|\,\mathbb{P}^{X_{\sigma}}_{\ast})\rightarrow\infty as σ → 0 + \sigma\rightarrow 0^{+} . This means that, despite 𝕂 𝕃 ( ℙ ∗ X ∥ ℙ ∗ X σ ) \mathbb{KL}(\mathbb{P}^{X}_{*}\,\|\,\mathbb{P}^{X_{\sigma}}_{\ast}) being minimized at σ = 0 \sigma=0 , the KL divergence provides no learning signal, and the same holds for the reverse KL. On the other hand, 𝕎 c ​ ( ℙ ∗ X , ℙ ∗ X σ ) → 0 \mathbb{W}^{c}(\mathbb{P}^{X}_{*},\mathbb{P}^{X_{\sigma}}_{\ast})\rightarrow 0 as σ → 0 + \sigma\rightarrow 0^{+} , provided that 𝕎 c \mathbb{W}^{c} metrizes weak convergence. 
 
 
 
 

### Section 3.6 Maximum Mean Discrepancy

 
 Similarly to Wasserstein distances ( Section 3.5 ), the maximum mean discrepancy ( Gretton et al., 2006 , MMD;) provides a notion of distance between probability distributions which remains mathematically meaningful in the manifold setting.
The MMD between two probability densities p p and q q on 𝒳 \mathcal{X} , 𝕄 ​ 𝕄 ​ 𝔻 k ​ ( p , q ) \mathbb{MMD}_{k}(p,q) , is given by

 

 
 | 
 𝕄 ​ 𝕄 ​ 𝔻 k ​ ( p , q ) ≔ ( 𝔼 X , X ′ ∼ p ​ [ k ⁡ ( X , X ′ ) ] − 2 ​ 𝔼 X ∼ p , Y ∼ q ​ [ k ⁡ ( X , Y ) ] + 𝔼 Y , Y ′ ∼ q ​ [ k ⁡ ( Y , Y ′ ) ] ) 1 2 , \mathbb{MMD}_{k}(p,q)\coloneqq\Big(\mathbb{E}_{X,X^{\prime}\sim p}[k(X,X^{\prime})]-2\mathbb{E}_{X\sim p,Y\sim q}[k(X,Y)]+\mathbb{E}_{Y,Y^{\prime}\sim q}[k(Y,Y^{\prime})]\Big)^{\frac{1}{2}}, | 
 | 
 (19) | 
 

 where X , X ′ , Y , Y ′ X,X^{\prime},Y,Y^{\prime} are independent, and k : 𝒳 × 𝒳 → ℝ k:\mathcal{X}\times\mathcal{X}\rightarrow\mathbb{R} is a symmetric positive semi-definite kernel (i.e. a function having the property that, for any n ∈ ℕ n\in\mathbb{N} and x 1 , … , x n ∈ 𝒳 x_{1},\dots,x_{n}\in\mathcal{X} , the n × n n\times n matrix K K given by K i ​ j = k ⁡ ( x i , x j ) K_{ij}=k(x_{i},x_{j}) is symmetric positive semi-definite), which is set as a hyperparameter.

 
 
 The MMD has several desirable mathematical properties. ( i ) (i) Under some regularity conditions which are satisfied by many commonly-used kernels, the MMD is a metric in the space of probability distributions over 𝒳 \mathcal{X} . ( i ​ i ) (ii) 𝕄 ​ 𝕄 ​ 𝔻 k 2 ​ ( p , q ) \mathbb{MMD}^{2}_{k}(p,q) can be straightforwardly estimated in an unbiased manner through Monte Carlo sampling, making it particularly amenable to gradient-based optimization. ( i ​ i ​ i ) (iii) Conditions on k k which make 𝕄 ​ 𝕄 ​ 𝔻 k \mathbb{MMD}_{k} meaningfully defined in the manifold setting (formally, conditions under which MMD metrizes weak convergence) are known ( Simon-Gabriel Schölkopf, 2018 ; Simon-Gabriel et al., 2023 ) . Provided that 𝒳 \mathcal{X} is compact (which is the case for images in [ 0 , 1 ] D [0,1]^{D} ), these conditions hold for most commonly-used kernels, meaning that MMD can be used to compare distributions regardless of their support.

 
 
 
 

## Section 4 Manifold-Unaware Deep Generative Models

 
 In this section we describe popular deep generative modelling frameworks which were not developed with the manifold setting in mind, and discuss their inability to learn p ∗ X {p}^{X}_{\ast} in this setting.

 
 

### Section 4.1 The Problem with Likelihood-Based Approaches: Manifold Overfitting

 
 Likelihood-based deep generative models are a broad and popular class of models, which includes variational autoencoders ( Kingma Welling, 2014 ; Rezende et al., 2014 ) , normalizing flows ( Dinh et al., 2015 ; Dinh et al., 2017 ) , energy-based models ( Xie et al., 2016 ; Du Mordatch, 2019 ) , continuous autoregressive models ( Uria et al., 2013 ) , and more ( Bond-Taylor et al., 2022 ) .
At a high-level, these models leverage neural networks to construct a full-dimensional density p θ X {p}^{X}_{\theta} . The models are trained by maximizing, sometimes approximately, the log-likelihood:

 

 
 | 
 max θ ⁡ 𝔼 X ∼ p ∗ X ​ [ log ⁡ p θ X ​ ( X ) ] . \max_{\theta}\mathbb{E}_{X\sim{p}^{X}_{\ast}}[\log{p}^{X}_{\theta}(X)]. | 
 | 
 (20) | 
 

 When the underlying density p ∗ X {p}^{X}_{\ast} is full-dimensional, this objective is equivalent to minimizing the KL divergence between p ∗ X {p}^{X}_{\ast} and p θ X {p}^{X}_{\theta} ( Equation 13 ). However, in our setting of interest p ∗ X {p}^{X}_{\ast} is manifold-supported, and as mentioned in Section 3.4 , this equivalence breaks down, leading to the natural question: what happens if the likelihood is optimized when p θ X {p}^{X}_{\theta} is full-dimensional but p ∗ X {p}^{X}_{\ast} is not?

 
 
 
 
 Figure 5: Illustration of manifold overfitting, where the 1 1 -dimensional p ∗ X {p}^{X}_{\ast} (shades of blue) along a curve ℳ \mathcal{M} in 2 2 -dimensional ambient space is improperly approximated. Each row shows a sequence of full-dimensional densities p θ t X {p}^{X}_{\theta_{t}} (red surfaces) having the property that their likelihood diverges to infinity on all of ℳ \mathcal{M} , yet each sequence approximates a different manifold-supported density p † X p_{\dagger}^{X} on ℳ \mathcal{M} : the top sequence will recover a bimodal distribution on ℳ \mathcal{M} and the bottom sequence a trimodal one, despite p ∗ X {p}^{X}_{\ast} being unimodal. 
 
 
 The first consequence of this dimensionality misspecification is that the log-likelihood does not admit a maximum as it can be made arbitrarily large. To see this, consider a sequence of full-dimensional models ( p θ t X ) t = 0 ∞ ({p}^{X}_{\theta_{t}})_{t=0}^{\infty} which concentrate more and more mass around ℳ \mathcal{M} during training, as depicted in Figure 5 . If ℳ \mathcal{M} was full-dimensional, it would be impossible to have p θ t X ​ ( x ) → ∞ {p}^{X}_{\theta_{t}}(x)\rightarrow\infty as t → ∞ t\rightarrow\infty for all x ∈ ℳ x\in\mathcal{M} , as doing so would quickly violate the requirement that the densities integrate to 1 1 . However, when ℳ \mathcal{M} is low-dimensional, it is “infinitely thin” in ℝ D \mathbb{R}^{D} , and thus the model densities can be made to diverge to infinity along the entire manifold. This phenomenon is illustrated twice in Figure 5 .

 
 
 At a first glance, the fact that the likelihood does not admit a maximum might seem inconsequential, as one might hope that as long as 𝔼 X ∼ p ∗ X ​ [ log ⁡ p θ t X ​ ( X ) ] → ∞ \mathbb{E}_{X\sim{p}^{X}_{\ast}}[\log{p}^{X}_{\theta_{t}}(X)]\rightarrow\infty as t → ∞ t\rightarrow\infty , then p ∗ X {p}^{X}_{\ast} is still being learned. However, this is not the case, and the reason is once again illustrated in Figure 5 : there are many ways in which the likelihood can diverge to infinity.
 Loaiza-Ganem et al. (2022a) formalized this intuition by proving that under mild regularity conditions, for any manifold-supported density p † X p_{\dagger}^{X} on ℳ \mathcal{M} , there always exists a sequence of full-dimensional densities which simultaneously ( i ) (i) becomes arbitrarily large on the entire manifold, in turn maximizing likelihood, yet ( i ​ i ) (ii) approximates p † X p_{\dagger}^{X} rather than the true data-generating density p ∗ X {p}^{X}_{\ast} . The latter condition is formalized using weak convergence in the grey box below, but can be intuitively understood as saying that samples from p θ t X {p}^{X}_{\theta_{t}} and p † X p_{\dagger}^{X} become indistinguishable as t → ∞ t\rightarrow\infty . 3 3 
 3 
 
 
 
 Note that approximating p † X p_{\dagger}^{X} does not imply that p θ t X ​ ( x ) → p † X ​ ( x ) {p}^{X}_{\theta_{t}}(x)\rightarrow p_{\dagger}^{X}(x) as t → ∞ t\rightarrow\infty for all x x because p θ t X {p}^{X}_{\theta_{t}} is full-dimensional, whereas p † X p_{\dagger}^{X} is manifold-supported. 

 
 
 An immediate consequence of this result is that maximum-likelihood is an ill-posed objective in the manifold setting, as it simply encourages models to concentrate mass around ℳ \mathcal{M} with no concern for the distribution within it. Loaiza-Ganem et al. (2022a) thus call this behaviour manifold overfitting . Several consequences of manifold overfitting are worth discussing. ( i ) (i) Manifold overfitting does not imply that model densities diverge to infinity on all of ℳ \mathcal{M} ; as long as these densities diverge to infinity on a subset of non-zero probability under p ∗ X {p}^{X}_{\ast} and do not converge to zero on the rest of ℳ \mathcal{M} , the log-likelihood will still be “maximized”, i.e. 𝔼 X ∼ p ∗ X ​ [ log ⁡ p θ t X ​ ( X ) ] → ∞ \mathbb{E}_{X\sim{p}^{X}_{\ast}}[\log p_{\theta_{t}}^{X}(X)]\rightarrow\infty as t → ∞ t\rightarrow\infty . Analogously, densities could diverge to infinity on a superset of ℳ \mathcal{M} – such as a manifold of dimension higher than d ∗ d^{\ast} but lower than D D ( Koehler et al., 2022 ) . Similarly, the log-likelihood can be made to diverge to infinity in such a way that the sequence of models p θ t X p_{\theta_{t}}^{X} does not learn any distribution p † X p_{\dagger}^{X} ( Loaiza-Ganem et al., 2022a ) . In other words, the behaviour of models which “maximize” likelihood in the manifold setting can be pathological beyond p θ t X ​ ( x ) {p}^{X}_{\theta_{t}}(x) diverging to infinity if and only if x ∈ ℳ x\in\mathcal{M} . ( i ​ i ) (ii) One might be hopeful that in practice these pathological scenarios are avoided through the optimization dynamics of gradient descent so that p ∗ X {p}^{X}_{\ast} is properly learned, yet this is not the case ( Koehler et al., 2022 ) . ( i ​ i ​ i ) (iii) Manifold overfitting cannot be detected by using test likelihoods: as long as the test data is generated from p ∗ X {p}^{X}_{\ast} , then it lies on ℳ \mathcal{M} with probability 1 1 , and thus test likelihoods are also subject to degenerate behaviour. This observation highlights that, in the manifold setting, test log-likelihoods should be avoided as a DGM evaluation metric, and that sample-based metrics ( Heusel et al., 2017 ; Borji, 2019 ; Stein et al., 2023 ) should be favoured instead. This unreliability of test log-likelihoods is consistent with the fact that they are not always correlated with sample quality when modelling images ( Theis et al., 2016 ) .

 
 
 
 The manifold overfitting result of Loaiza-Ganem et al. (2022a) can be more formally stated as saying that, under some regularity conditions and provided that d ∗ D d^{\ast} D , for any distribution ℙ † X \mathbb{P}_{\dagger}^{X} on 𝒳 \mathcal{X} supported on ℳ \mathcal{M} , there exists a sequence of distributions ( ℙ θ t X ) t = 1 ∞ (\mathbb{P}_{\theta_{t}}^{X})_{t=1}^{\infty} such that: 
 
 • 
 
 ℙ θ t X \mathbb{P}_{\theta_{t}}^{X} is full-dimensional, i.e. ℙ θ t X ≪ λ D \mathbb{P}_{\theta_{t}}^{X}\ll\lambda_{D} , for every t t . 
 
 • 
 
 For every x ∈ ℳ x\in\mathcal{M} , it holds that p θ t X ​ ( x ) → ∞ p_{\theta_{t}}^{X}(x)\rightarrow\infty as t → ∞ t\rightarrow\infty , where p θ t X p_{\theta_{t}}^{X} is a density of ℙ θ t X \mathbb{P}_{\theta_{t}}^{X} with respect to λ D \lambda_{D} . 
 
 • 
 
 For every x ∈ 𝒳 ∖ cl 𝒳 ⁡ ( ℳ ) x\in\mathcal{X}\setminus\cl_{\mathcal{X}}(\mathcal{M}) , it holds that p θ t X ​ ( x ) → 0 p_{\theta_{t}}^{X}(x)\rightarrow 0 as t → ∞ t\rightarrow\infty . 
 
 • 
 
 ℙ θ t X → 𝜔 ℙ † X \mathbb{P}_{\theta_{t}}^{X}\xrightarrow{\omega}\mathbb{P}_{\dagger}^{X} as t → ∞ t\rightarrow\infty . 
 
 
 Loaiza-Ganem et al. (2022a) proved this result under the assumption that ℳ \mathcal{M} is analytic. Despite their other regularity conditions being very mild, this is a strong assumption. However, as we now argue, the result actually holds for arbitrary smooth submanifolds ℳ \mathcal{M} of 𝒳 \mathcal{X} . Gray (1974, Theorem 3.1) proved a result which immediately implies that, if ℳ \mathcal{M} is an analytic Riemannian manifold, then for x ∈ ℳ x\in\mathcal{M} , as ε → 0 \varepsilon\rightarrow 0 , 
 

 
 
 λ ℳ ​ ( B ε ℳ ​ ( x ) ) = v ⁡ ( d ∗ ) ​ ε d ∗ ​ ( 1 + 𝒪 ⁡ ( ϵ 2 ) ) , \lambda_{\mathcal{M}}\left(B^{\mathcal{M}}_{\varepsilon}(x)\right)=v(d^{\ast})\,\varepsilon^{d^{\ast}}\left(1+\mathcal{O}(\epsilon^{2})\right), 
 
 (21) 
 
 where λ ℳ \lambda_{\mathcal{M}} is the Riemannian measure on ℳ \mathcal{M} , B ε ℳ ​ ( x ) B_{\varepsilon}^{\mathcal{M}}(x) denotes a geodesic ball in ℳ \mathcal{M} of radius ε \varepsilon centred at x x , and v ⁡ ( d ∗ ) v(d^{\ast}) is the volume of a d ∗ d^{\ast} -dimensional Euclidean ball of radius 1 1 . Loaiza-Ganem et al. (2022a) used the assumption that ℳ \mathcal{M} is analytic only to apply Equation 21 by evoking the result of Gray (1974) . However, Equation 21 is known to hold for arbitrary smooth Riemannian manifolds ( Gallot et al., 2004 , Theorem 3.98) . It immediately follows that the result of Loaiza-Ganem et al. (2022a) indeed holds for arbitrary smooth submanifolds ℳ \mathcal{M} of 𝒳 \mathcal{X} , even if they are not analytic. 
 
 
 

#### Section 4.1.1 The Unavoidable Numerical Instability of High-Dimensional Likelihoods

 
 The manifold overfitting result of Loaiza-Ganem et al. (2022a) described in Section 4.1 establishes that maximum-likelihood is an ill-posed objective for high-dimensional densities in the manifold setting. Before continuing our review of existing work, we point out that their result does not rule out the possibility of somehow addressing the pathological behaviour of maximum-likelihood, for example by adding a regularizer. Here we prove that it is actually impossible to do so, by showing that for any “infinitely thin” subset M M of 𝒳 \mathcal{X} (of which ℳ \mathcal{M} is an example, but here we do not require M M to be a manifold), any density p † X {p}^{X}_{\dagger} supported on M M , and any sequence of D D -dimensional models p θ t X {p}^{X}_{\theta_{t}} which learn p † X {p}^{X}_{\dagger} , the following holds: ( i ) (i) for any x ∈ 𝒳 x\in\mathcal{X} outside of M M , p θ t X ​ ( x ) {p}^{X}_{\theta_{t}}(x) gets arbitrarily close to 0 0 as t → ∞ t\rightarrow\infty ; and ( i ​ i ) (ii) for any x ∈ M x\in M and any L 0 L 0 , for large enough t t it holds that p θ t X ​ ( x ′ ) L {p}^{X}_{\theta_{t}}(x^{\prime}) L for some x ′ ∈ 𝒳 x^{\prime}\in\mathcal{X} arbitrarily close to x x . We formally state our theorem and include a technical discussion in the grey box below.

 
 
 Technicalities aside, our result shows that likelihoods become arbitrarily close to 0 0 outside M M , and that they become arbitrarily large on it (or arbitrarily close to it). The problem here is twofold: likelihoods are unstable not only because they become arbitrarily large around M M , but also because they must change very rapidly to approach 0 0 outside of it. In particular, this implies that if p θ t X {p}^{X}_{\theta_{t}} is Lipschitz, the corresponding Lipschitz constant must blow up as t → ∞ t\rightarrow\infty .

 
 
 One way to interpret our result is as a “soft generalization” of the manifold overfitting result of Loaiza-Ganem et al. (2022a) ; whereas they show that for any target p † X {p}^{X}_{\dagger} there exists a sequence of models which approximates it while exploding on ℳ \mathcal{M} and converging to 0 0 elsewhere, we show that any sequence recovering p † X {p}^{X}_{\dagger} will exhibit similar pathological behaviour on M M . Two implications of our result are worth discussing:

 
 
 
 • 
 
 Numerical instability of likelihood evaluation  Our theorem applies even if the models were not trained through maximum-likelihood, so that if the target distribution is correctly recovered through any means, density evaluation will remain numerically unstable – even when training itself does not involve likelihoods and is numerically stable. To see this, simply apply our theorem with p † X = p ∗ X {p}^{X}_{\dagger}={p}^{X}_{\ast} , which immediately yields that any sequence of D D -dimensional models which learn p ∗ X {p}^{X}_{\ast} will do so with numerically unstable likelihoods. We can gain intuition as to why this should indeed be the case through Figure 5 : the only way for the red surfaces ( p θ t X {p}^{X}_{\theta_{t}} ) to recover the density on the blue curve ( p ∗ X {p}^{X}_{\ast} ) is by spiking to infinity around it, and by not assigning mass elsewhere.

 

 • 
 
 Unfixability of maximum-likelihood  Another consequence of our result is that maximum-likelihood cannot be “fixed”; for example, any regularizer added to it which ensures that p ∗ X {p}^{X}_{\ast} is learned (rather than some arbitrary p † X {p}^{X}_{\dagger} ) would not circumvent the aforementioned numerical instabilities – provided it does not obviate the need to compute likelihoods (or any surrogates used) during training (e.g. by cancelling out the log-likelihood, at which point it would not fit the description of a regularizer anymore). Analogously, any regularizer or architecture guaranteeing numerical stability of likelihoods would be such that p ∗ X {p}^{X}_{\ast} is not learned. In other words, our result ensures that learning p ∗ X {p}^{X}_{\ast} and having numerically stable likelihoods cannot happen simultaneously.

 

 
 
 
 
 1 Likelihood Instability of Deep Generative Models. 
 
 Let M ⊂ 𝒳 M\subset\mathcal{X} be a Borel set such that λ D ​ ( cl 𝒳 ⁡ ( M ) ) = 0 \lambda_{D}(\cl_{\mathcal{X}}(M))=0 , and let ℙ † X \mathbb{P}^{X}_{\dagger} be a probability measure on 𝒳 \mathcal{X} such that ℙ † X ​ ( M ) = 1 \mathbb{P}^{X}_{\dagger}(M)=1 and supp ⁡ ( ℙ † X ) = cl 𝒳 ⁡ ( M ) \supp(\mathbb{P}^{X}_{\dagger})=\cl_{\mathcal{X}}(M) . Let ( ℙ θ t X ) t = 1 ∞ (\mathbb{P}^{X}_{\theta_{t}})_{t=1}^{\infty} be a sequence of probability measures on 𝒳 \mathcal{X} such that ℙ θ t X → 𝜔 ℙ † X \mathbb{P}^{X}_{\theta_{t}}\xrightarrow{\omega}\mathbb{P}^{X}_{\dagger} as t → ∞ t\rightarrow\infty and ℙ θ t X ≪ λ D \mathbb{P}^{X}_{\theta_{t}}\ll\lambda_{D} , with corresponding densities p θ t X {p}^{X}_{\theta_{t}} . Then: 
 
 • 
 
 lim inf t → ∞ p θ t X ​ ( x ) = 0 \displaystyle\liminf_{t\rightarrow\infty}{p}^{X}_{\theta_{t}}(x)=0 , λ D \lambda_{D} -almost-everywhere on 𝒳 ∖ cl 𝒳 ⁡ ( M ) \mathcal{X}\setminus\cl_{\mathcal{X}}(M) . 
 
 • 
 
 sup x ′ ∈ B ε ​ ( x ) p θ t X ​ ( x ′ ) → ∞ \displaystyle\sup_{x^{\prime}\in B_{\varepsilon}(x)}{p}^{X}_{\theta_{t}}(x^{\prime})\rightarrow\infty as t → ∞ t\rightarrow\infty for every x ∈ cl 𝒳 ⁡ ( M ) x\in\cl_{\mathcal{X}}(M) and every ε 0 \varepsilon 0 , where B ε ​ ( x ) ≔ { x ′ ∈ 𝒳 ∣ ‖ x ′ − x ‖ 2 ε } B_{\varepsilon}(x)\coloneqq\{x^{\prime}\in\mathcal{X}\mid\|x^{\prime}-x\|_{2} \varepsilon\} . 
 
 
 
 
 
 Proof. 
 
 See Appendix B.1 . ∎ 
 
 
 
 We now make some relevant observations about the Likelihood Instability Theorem . ( i ) (i) Note that we only require the closure of M M to have Lebesgue measure 0 0 , so it need not be a manifold. Our result thus applies in settings beyond the standard manifold hypothesis, such as when M M is given by a union of manifolds ( Brown et al., 2023 ) , or by a non-manifold set with singularities ( Von Rohrscheidt Rieck, 2023 ; Wang Wang, 2024 ) . ( i ​ i ) (ii) lim inf t → ∞ p θ t X ​ ( x ) \liminf_{t\rightarrow\infty}{p}^{X}_{\theta_{t}}(x) cannot in general be replaced by lim t → ∞ p θ t X ​ ( x ) \lim_{t\rightarrow\infty}{p}^{X}_{\theta_{t}}(x) since the limit need not exist, but as an immediate corollary, if the limit exists, then it must be 0 0 λ D \lambda_{D} -almost-everywhere on 𝒳 ∖ cl 𝒳 ⁡ ( M ) \mathcal{X}\setminus\cl_{\mathcal{X}}(M) . ( i ​ i ​ i ) (iii) We also point out that sup x ′ ∈ B ε ​ ( x ) p θ t X ​ ( x ′ ) \sup_{x^{\prime}\in B_{\varepsilon}(x)}{p}^{X}_{\theta_{t}}(x^{\prime}) cannot be in general replaced by p θ t X ​ ( x ) {p}^{X}_{\theta_{t}}(x) either, despite our conclusion holding for every ε 0 \varepsilon 0 . Intuitively, this is because for any given x ∈ M x\in M the divergence to infinity of the density might not happen at x x , but rather on a sequence converging to it: we provide an illustrative example in Appendix B.1 . ( i ​ v ) (iv) We do emphasize that despite not concluding that p θ t X ​ ( x ) → 0 {p}^{X}_{\theta_{t}}(x)\rightarrow 0 outside of M M nor that p θ t X ​ ( x ) → ∞ {p}^{X}_{\theta_{t}}(x)\rightarrow\infty in M M , our result does unequivocally ensure numerical instability of the involved densities. 
 
 
 
 

#### Section 4.1.2 Variational Autoencoders

 
 Variational autoencoders ( Kingma Welling, 2014 ; Rezende et al., 2014 , VAEs;) are a class of likelihood-based models.
The continuous VAEs that we consider here specify a fixed prior density p Z {p}^{Z} (commonly a standard Gaussian) on 𝒵 \mathcal{Z} , along with a learnable conditional full-dimensional likelihood p θ X | Z p^{X|Z}_{\theta} on 𝒳 \mathcal{X} , which is often a parameterized Gaussian,

 

 
 | 
 p θ X | Z ​ ( x | z ) = 𝒩 ⁡ ( x , g θ ​ ( z ) , Σ θ X | Z ​ ( z ) ) , p^{X|Z}_{\theta}(x|z)=\mathcal{N}\left(x;g_{\theta}(z),\Sigma_{\theta}^{X|Z}(z)\right), | 
 | 
 (22) | 
 

 where Σ θ X | Z : 𝒵 → ℝ D × D \Sigma_{\theta}^{X|Z}:\mathcal{Z}\rightarrow\mathbb{R}^{D\times D} is symmetric positive definite, often given by γ ​ I D \gamma I_{D} , where γ 0 \gamma 0 is treated as a free parameter rather than the output of a neural network. The conditional likelihood p θ X | Z ( ⋅ | z ) p^{X|Z}_{\theta}(\cdot|z) in Equation 22 can be understood as a stochastic decoder, whose mean is given by the deterministic decoder g θ ​ ( z ) g_{\theta}(z) .
Together, the prior and the conditional likelihood implicitly define the marginal likelihood over data:

 

 
 | 
 p θ X ​ ( x ) = ∫ p Z ​ ( z ) ​ p θ X | Z ​ ( x | z ) ​ d ​ z . {p}^{X}_{\theta}(x)=\int{p}^{Z}(z)p^{X|Z}_{\theta}(x|z){\textnormal{d}}z. | 
 | 
 (23) | 
 

 Since computing the marginal likelihood involves an intractable integral, a variational posterior density q ϕ Z | X q^{Z|X}_{\phi} on 𝒵 \mathcal{Z} is introduced, and the following objective, called the evidence lower bound (ELBO), is jointly maximized over θ \theta and ϕ \phi :

 

 
 | 
 ℰ p ∗ X ​ ( θ , ϕ ) \displaystyle\mathcal{E}_{{p}^{X}_{\ast}}(\theta,\phi) | 
 ≔ 𝔼 X ∼ p ∗ X [ 𝔼 Z ∼ q Z | X ϕ ( ⋅ | X ) [ log p θ X | Z ( X | Z ) ] − 𝕂 𝕃 ( q ϕ Z | X ( ⋅ | X ) ∥ p Z ) ] \displaystyle\coloneqq\mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[\mathbb{E}_{Z\sim q^{Z|X}_{\phi}(\cdot|X)}[\log p^{X|Z}_{\theta}(X|Z)]-\mathbb{KL}\left(q^{Z|X}_{\phi}(\cdot|X)\,\Big\|\,{p}^{Z}\right)\right] | 
 | 
 (24) | 
 
 
 | 
 | 
 = 𝔼 X ∼ p ∗ X [ log p θ X ( X ) − 𝕂 𝕃 ( q ϕ Z | X ( ⋅ | X ) ∥ p θ Z | X ( ⋅ | X ) ) ] ≤ 𝔼 X ∼ p ∗ X [ log p θ X ( X ) ] , \displaystyle=\mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[\log{p}^{X}_{\theta}(X)-\mathbb{KL}\left(q^{Z|X}_{\phi}(\cdot|X)\,\Big\|\,p^{Z|X}_{\theta}(\cdot|X)\right)\right]\leq\mathbb{E}_{X\sim{p}^{X}_{\ast}}[\log{p}^{X}_{\theta}(X)], | 
 | 
 (25) | 
 

 where p θ Z | X p^{Z|X}_{\theta} denotes the true posterior density, which is implicitly defined by the prior p Z {p}^{Z} and the conditional likelihood p θ X | Z p_{\theta}^{X|Z} . Note that Equation 24 is used as the optimization objective, since the true posterior density cannot be tractably evaluated. While not directly usable as an objective, Equation 25 shows that the ELBO lower-bounds the log-likelihood 𝔼 X ∼ p ∗ X ​ [ log ⁡ p θ X ​ ( X ) ] \mathbb{E}_{X\sim{p}^{X}_{\ast}}[\log{p}^{X}_{\theta}(X)] , and differs from it only by the error incurred by q ϕ Z | X q^{Z|X}_{\phi} to approximate the true posterior: this is often used as a justification for using the ELBO as an objective, as it simultaneously encourages learning θ \theta through maximum-likelihood, and ϕ \phi so that q ϕ Z | X q^{Z|X}_{\phi} matches the true posterior. It is common to also specify q ϕ Z | X q^{Z|X}_{\phi} as a Gaussian,

 

 
 | 
 q ϕ Z | X ​ ( z | x ) = 𝒩 ⁡ ( z , f ϕ ​ ( x ) , Σ ϕ Z | X ​ ( x ) ) , q^{Z|X}_{\phi}(z|x)=\mathcal{N}\left(z;f_{\phi}(x),\Sigma_{\phi}^{Z|X}(x)\right), | 
 | 
 (26) | 
 

 where Σ ϕ Z | X : 𝒳 → ℝ d × d \Sigma_{\phi}^{Z|X}:\mathcal{X}\rightarrow\mathbb{R}^{d\times d} is also symmetric positive definite, and often given by a diagonal matrix with positive entries along its diagonal. In an analogous manner to the conditional likelihood, the variational posterior q ϕ Z | X ( ⋅ | x ) q_{\phi}^{Z|X}(\cdot|x) in Equation 26 can be interpreted as a stochastic encoder, whose mean is given by the deterministic encoder f ϕ ​ ( x ) f_{\phi}(x) .

 
 
 An issue which commonly affects VAEs is posterior collapse ( Chen et al., 2017 ; Wang et al., 2021 ) , where the learned variational posterior q ϕ ∗ Z | X q_{\phi^{\ast}}^{Z|X} partially collapses to the prior p Z {p}^{Z} . We will shortly explain posterior collapse in VAEs through the manifold lens, and thus we briefly summarize the phenomenon here: in the case where Σ ϕ ∗ Z | X \Sigma_{\phi^{\ast}}^{Z|X} is taken as a diagonal matrix, a subset of the diagonal entries of Σ ϕ ∗ Z | X ​ ( x ) \Sigma_{\phi^{\ast}}^{Z|X}(x) collapses to 1 1 for x ∈ ℳ x\in\mathcal{M} , and the corresponding entries of f ϕ ∗ ​ ( x ) f_{\phi^{\ast}}(x) collapse to 0 0 , matching the standard Gaussian prior p Z {p}^{Z} ; whereas the remaining diagonal entries of Σ ϕ ∗ Z | X ​ ( x ) \Sigma_{\phi^{\ast}}^{Z|X}(x) are extremely close to 0 0 , essentially losing stochasticity along these coordinates. In other words, VAEs tend to only “use” a subset of the coordinates of their latent space 𝒵 \mathcal{Z} to obtain data samples and default the rest to the prior.

 
 
 VAEs through the lens of the manifold hypothesis 

 
 Despite the autoencoder-like structure of VAEs that leverages low-dimensional representations, VAEs as presented above are full-dimensional models. This is a direct consequence of the choice of p θ X | Z p^{X|Z}_{\theta} , which always assigns strictly positive density to all of 𝒳 \mathcal{X} , i.e. p θ X | Z ​ ( x | z ) 0 p^{X|Z}_{\theta}(x|z) 0 for all x ∈ 𝒳 x\in\mathcal{X} and z ∈ 𝒵 z\in\mathcal{Z} . This in turn implies that p θ X ​ ( x ) 0 {p}^{X}_{\theta}(x) 0 for all x ∈ 𝒳 x\in\mathcal{X} , so that the model density is not supported on a low-dimensional manifold. Thus, since the ELBO is maximized as a proxy for the log-likelihood, intuitively VAEs should be subject to manifold overfitting. Dai Wipf (2019) formally show that this is indeed the case by proving that, subject to some regularity conditions and assuming that d ≥ d ∗ d\geq d^{\ast} , for any manifold-supported density p † X p_{\dagger}^{X} on ℳ \mathcal{M} , there exists a sequence of VAE models parameterized by ( θ t , ϕ t ) t = 1 ∞ (\theta_{t},\phi_{t})_{t=1}^{\infty} such that: ( i ) (i) ℰ p ∗ X ​ ( θ t , ϕ t ) → ∞ \mathcal{E}_{{p}^{X}_{\ast}}(\theta_{t},\phi_{t})\rightarrow\infty as t → ∞ t\rightarrow\infty ; ( i ​ i ) (ii) the VAE models p θ t X p_{\theta_{t}}^{X} learn p † X p_{\dagger}^{X} instead of p ∗ X {p}^{X}_{\ast} ; and ( i ​ i ​ i ) (iii) 𝕂 𝕃 ( q ϕ t Z | X ( ⋅ | x ) ∥ p θ t Z | X ( ⋅ | x ) ) → 0 \mathbb{KL}(q^{Z|X}_{\phi_{t}}(\cdot|x)\,\|\,p_{\theta_{t}}^{Z|X}(\cdot|x))\rightarrow 0 as t → ∞ t\rightarrow\infty for every x ∈ ℳ x\in\mathcal{M} . Although this result predates the manifold overfitting result of Loaiza-Ganem et al. (2022a) from Section 4.1 , it can be understood as saying that maximizing the ELBO instead of the log-likelihood does not prevent manifold overfitting. It is also worth noting that while the work of Loaiza-Ganem et al. (2022a) extends the result from Dai Wipf (2019) to non-VAE models and VAE models with flexible variational posteriors ( Rezende Mohamed, 2015 ; Kingma et al., 2016 ; van den Berg et al., 2018 ; Caterini et al., 2021a ) – since if the variational posterior is flexible enough, optimizing the ELBO becomes equivalent to maximizing the log-likelihood in the nonparametric regime ( Section 2.2 ) – it does not immediately imply that manifold overfitting can happen when q ϕ Z | X q^{Z|X}_{\phi} is Gaussian, which Dai Wipf (2019) do prove.

 
 
 Well-known training instabilities of VAEs can be understood as consequences of manifold overfitting. For example, as previously mentioned, it is common to use Σ θ X | Z ​ ( z ) = γ ​ I D \Sigma^{X|Z}_{\theta}(z)=\gamma I_{D} for every z ∈ 𝒵 z\in\mathcal{Z} , where γ \gamma is a free parameter ( γ \gamma is also often treated as a non-learnable hyperparameter instead). This purposely simplistic choice is made for the sake of training stability, e.g. parameterizing Σ θ X | Z \Sigma^{X|Z}_{\theta} as a diagonal matrix whose non-zero entries are given by a neural network can easily result in divergent training ( Lin et al., 2019 ; Rybkin et al., 2021 ) ; the added flexibility of the stochastic decoder of this VAE can be understood as making it more prone to experience manifold overfitting. We also highlight that, when modelling images, the best empirically performing VAEs do not use a Gaussian conditional likelihood p θ X | Z p_{\theta}^{X|Z} . Instead, they treat pixels as discrete and use a categorical p θ X | Z p_{\theta}^{X|Z} ( Vahdat Kautz, 2020 ) . Mathematically, these discrete models are not afflicted by manifold overfitting (we discuss why in Section 6 ), which helps explain why they outperform VAEs with continuous conditional likelihoods.

 
 
 Dai Wipf (2019) provide further insights into the interplay between VAEs and the manifold hypothesis beyond manifold overfitting. In particular, they also show that under appropriate conditions, Gaussian VAEs such as the ones presented above achieve perfect reconstructions, in the sense that encoding and then decoding any x ∈ ℳ x\in\mathcal{M} recovers x x .
This result justifies using VAEs as autoencoders despite the fact that they suffer from manifold overfitting. Additionally, Dai Wipf (2019) also show that only d ∗ d^{\ast} latent dimensions are needed to achieve these perfect reconstructions, suggesting that the posterior over the remaining d − d ∗ d-d^{\ast} latent dimensions defaults to the standard Gaussian prior p Z {p}^{Z} due to the KL term in Equation 24 . In other words, the manifold lens helps elucidate why posterior collapse happens.

 
 
 
 We now formalize the discussion on the results of Dai Wipf (2019) , which assume Gaussian VAEs as described above, with the decoder covariance given by Σ θ X | Z ​ ( z ) = γ ​ I D \Sigma_{\theta}^{X|Z}(z)=\gamma I_{D} , where γ 0 \gamma 0 is a free parameter that does not depend on z z . They also assume throughout that ℳ \mathcal{M} is diffeomorphic to ℝ d ∗ \mathbb{R}^{d^{\ast}} , which is a much stronger assumption than required by Loaiza-Ganem et al. (2022a) .
 
 The first result of Dai Wipf (2019) assumes some regularity conditions, that d ∗ D d^{\ast} D ,
that d ≥ d ∗ d\geq d^{\ast} , and that γ \gamma is learnable (i.e. it is part of θ \theta ). The result then states that for any distribution ℙ † X \mathbb{P}_{\dagger}^{X} on 𝒳 \mathcal{X} supported on ℳ \mathcal{M} , there exist a sequence of VAE models parameterized by ( θ t , ϕ t ) t = 1 ∞ (\theta_{t},\phi_{t})_{t=1}^{\infty} such that: 
 
 • 
 
 For every x ∈ ℳ x\in\mathcal{M} , it holds that p θ t X ​ ( x ) → ∞ p_{\theta_{t}}^{X}(x)\rightarrow\infty and 𝕂 𝕃 ( q ϕ t Z | X ( ⋅ | x ) ∥ p θ t Z | X ( ⋅ | x ) ) → 0 \mathbb{KL}\left(q^{Z|X}_{\phi_{t}}(\cdot|x)\,\Big\|\,p_{\theta_{t}}^{Z|X}(\cdot|x)\right)\rightarrow 0 as t → ∞ t\rightarrow\infty . Note that together, these two limits imply not only that 𝔼 X ∼ ℙ ∗ X ​ [ log ⁡ p θ t X ​ ( X ) ] → ∞ \mathbb{E}_{X\sim\mathbb{P}^{X}_{*}}[\log p_{\theta_{t}}^{X}(X)]\rightarrow\infty , but also that ℰ p ∗ X ​ ( θ t , ϕ t ) → ∞ \mathcal{E}_{{p}^{X}_{\ast}}(\theta_{t},\phi_{t})\rightarrow\infty as t → ∞ t\rightarrow\infty . 
 
 • 
 
 ℙ θ t X → 𝜔 ℙ † X \mathbb{P}_{\theta_{t}}^{X}\xrightarrow{\omega}\mathbb{P}_{\dagger}^{X} as t → ∞ t\rightarrow\infty , where ℙ θ t X \mathbb{P}_{\theta_{t}}^{X} is the distribution corresponding to the model density p θ t X p_{\theta_{t}}^{X} . 
 
 
 
 As previously mentioned, Dai Wipf (2019) also show that Gaussian VAEs achieve perfect reconstructions, and they link this behaviour to posterior collapse. To do so, they first consider γ 0 \gamma 0 as a fixed hyperparameter instead of being learnable, and write the corresponding ELBO as ℰ p ∗ X ​ ( θ , ϕ , γ ) \mathcal{E}_{{p}^{X}_{\ast}}(\theta,\phi;\gamma) , with corresponding maximizers θ ∗ ​ ( γ ) \theta^{\ast}(\gamma) and ϕ ∗ ​ ( γ ) \phi^{\ast}(\gamma) . They then prove that, if d ≥ d ∗ d\geq d^{\ast} and under similar regularity conditions to the result above, for every γ 0 \gamma 0 there exists γ ′ ∈ ( 0 , γ ) \gamma^{\prime}\in(0,\gamma) such that: 
 

 
 
 ℰ p ∗ X ​ ( θ ∗ ​ ( γ ′ ) , ϕ ∗ ​ ( γ ′ ) , γ ′ ) ℰ p ∗ X ​ ( θ ∗ ​ ( γ ) , ϕ ∗ ​ ( γ ) , γ ) . \mathcal{E}_{{p}^{X}_{\ast}}\left(\theta^{\ast}(\gamma^{\prime}),\phi^{\ast}(\gamma^{\prime});\gamma^{\prime}\right) \mathcal{E}_{{p}^{X}_{\ast}}\left(\theta^{\ast}(\gamma),\phi^{\ast}(\gamma);\gamma\right). 
 
 (27) 
 
 In particular this suggests that, when γ \gamma is learnable, it must converge to 0 0 to make the ELBO diverge to infinity. An intuitive way to understand this is as saying that, for a small enough decoder variance γ \gamma , it is preferable to maximize the log ⁡ p θ X | Z ​ ( X | Z ) \log p_{\theta}^{X|Z}(X|Z) term in Equation 24 – which in this case boils down to an ℓ 2 \ell_{2} reconstruction error weighted by 1 / ( 2 ​ γ ) 1/(2\gamma) – instead of minimizing the KL term. Dai Wipf (2019) then leverage this result to show that VAEs achieve perfect reconstructions in the sense that 
 

 
 
 lim γ → 0 g θ ∗ ​ ( γ ) ​ ( f ϕ ∗ ​ ( γ ) ​ ( x ) ) = x , ℙ ∗ X ​ -almost-surely . \lim_{\gamma\rightarrow 0}g_{\theta^{\ast}(\gamma)}\left(f_{\phi^{\ast}(\gamma)}(x)\right)=x,\quad\mathbb{P}^{X}_{*}\text{-almost-surely}. 
 
 (28) 
 
 Finally, since ℙ ∗ X \mathbb{P}^{X}_{*} is supported on a d ∗ d^{\ast} -dimensional manifold ℳ \mathcal{M} , perfectly reconstructing a point x ∈ ℳ x\in\mathcal{M} requires d ∗ d^{\ast} dimensions, and not the full d d of the latent space 𝒵 \mathcal{Z} : this means that d ∗ d^{\ast} latent dimensions are used to achieve perfect reconstructions, and thus the respective approximate posterior variances are sent to 0 0 ; whereas the remaining d − d ∗ d-d^{\ast} dimensions in the approximate posterior default to a standard Gaussian to minimize the KL term in the ELBO. This explanation of posterior collapse was suggested by Dai Wipf (2019) , and it was further formalized by Zheng et al. (2022) . 
 
 
 
 
 

#### Section 4.1.3 Normalizing Flows

 
 Normalizing flows ( Dinh et al., 2015 ; Dinh et al., 2017 ; Papamakarios et al., 2017 ; Kingma Dhariwal, 2018 ; Durkan et al., 2019 ; Kobyzev et al., 2020 ; Papamakarios et al., 2021 , NFs;) are a class of DGMs that leverage the change-of-variables formula to enable maximum-likelihood training. NFs construct a bijective neural network g θ : 𝒵 → 𝒳 g_{\theta}:\mathcal{Z}\rightarrow\mathcal{X} , where 𝒵 = 𝒳 = ℝ D \mathcal{Z}=\mathcal{X}=\mathbb{R}^{D} , such that both g θ g_{\theta} and its inverse f θ f_{\theta} are differentiable. 4 4 
 4 
 
 
 
 Note that in NFs, the “encoder” f θ f_{\theta} is uniquely determined by the “decoder” g θ g_{\theta} . Thus, the “encoder” does not require auxiliary parameters ϕ \phi , which is why we parameterize it with the generative parameters θ \theta . We do nonetheless highlight that moving away from this restriction by parameterizing the encoder and decoder networks separately and making them learn to invert each other has been attempted ( Draxler et al., 2024 ) . A prior density p Z {p}^{Z} is specified (often a standard Gaussian), and sampling from the model p θ X {p}^{X}_{\theta} is achieved through X = g θ ​ ( Z ) X=g_{\theta}(Z) where Z ∼ p Z Z\sim{p}^{Z} (formally, p θ X {p}^{X}_{\theta} is the pushforward of p Z {p}^{Z} through g θ g_{\theta} ). The change-of-variables formula from Equation 8 provides the maximum-likelihood objective:

 

 
 | 
 max θ ⁡ 𝔼 X ∼ p ∗ X ​ [ log ⁡ p Z ​ ( f θ ​ ( X ) ) + log ⁡ | det ∇ x f θ ​ ( X ) | ] . \max_{\theta}\mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[\log{p}^{Z}\left(f_{\theta}(X)\right)+\log\left|\det\nabla_{x}f_{\theta}(X)\right|\right]. | 
 | 
 (29) | 
 

 Constructing the Jacobian ∇ x f θ ​ ( x ) \nabla_{x}f_{\theta}(x) through automatic differentiation to compute the log-likelihood and then backpropagate with respect to θ \theta is computationally prohibitive. To circumvent this issue, NFs are constructed in such a way that ensures that det ∇ x f θ ​ ( x ) \det\nabla_{x}f_{\theta}(x) can be efficiently evaluated in closed-form (or at least approximated), thus enabling gradient optimization with respect to θ \theta .

 
 
 A relevant variant of NFs are continuous NFs, where f θ f_{\theta} is defined implicitly through an ordinary differential equation (ODE) rather than explicitly constructed ( Chen et al., 2018 ; Salman et al., 2018 ; Grathwohl et al., 2019 ) . More specifically, an auxiliary neural network v θ : 𝒳 × [ 0 , T ] → 𝒳 v_{\theta}:\mathcal{X}\times[0,T]\rightarrow\mathcal{X} is used to specify the ODE:

 

 
 | 
 d ​ x t = v θ ​ ( x t , t ) ​ d ​ t , x 0 ∈ 𝒳 . \displaystyle\begin{split} {\textnormal{d}}x_{t}=v_{\theta}(x_{t},t){\textnormal{d}}t,\\
 x_{0}\in\mathcal{X}.\end{split} | 
 | 
 (30) | 
 

 Under standard regularity conditions this ODE has a unique and smooth solution on [ 0 , T ] [0,T] ( Khalil, 2002 ) , 5 5 
 5 
 
 
 
 The most notable of these conditions is v θ v_{\theta} being Lipschitz in t t (with the Lipschitz constant not depending on x x ), which will become relevant when we discuss diffusion models in Section 5.1.2 . i.e. it characterizes the trajectory ( x t ) t ∈ [ 0 , T ] (x_{t})_{t\in[0,T]} , and thus implicitly defines a mapping from the initial condition to the final point in the trajectory, namely

 

 
 | 
 f θ : 𝒳 → 𝒵 , x 0 ↦ x T , \displaystyle\begin{split}f_{\theta}:\mathcal{X} \rightarrow\mathcal{Z},\\
x_{0} \mapsto x_{T},\end{split} | 
 | 
 (31) | 
 

 where again 𝒵 = 𝒳 \mathcal{Z}=\mathcal{X} .
Furthermore, under the same conditions that guarantee a unique solution to Equation 30 , f θ f_{\theta} is invertible and its inverse g θ g_{\theta} can be computed by solving the reverse ODE

 

 
 | 
 d ​ y t = − v θ ​ ( y t , T − t ) ​ d ​ t , y 0 = x T ∈ 𝒵 , \displaystyle\begin{split} {\textnormal{d}}y_{t}=-v_{\theta}(y_{t},T-t){\textnormal{d}}t,\\
 y_{0}=x_{T}\in\mathcal{Z},\end{split} | 
 | 
 (32) | 
 

 whose solution is the reversed trajectory ( y t ) t ∈ [ 0 , T ] = ( x T − t ) t ∈ [ 0 , T ] (y_{t})_{t\in[0,T]}=(x_{T-t})_{t\in[0,T]} , so that g θ g_{\theta} corresponds to the map

 

 
 | 
 g θ : 𝒵 → 𝒳 , y 0 ↦ y T . \displaystyle\begin{split}g_{\theta}:\mathcal{Z} \rightarrow\mathcal{X},\\
y_{0} \mapsto y_{T}.\end{split} | 
 | 
 (33) | 
 

 
 
 Like standard NFs, continuous NFs are sampled from by first obtaining Y 0 ∼ p Z Y_{0}\sim{p}^{Z} , and then computing Y T = g θ ​ ( Y 0 ) Y_{T}=g_{\theta}(Y_{0}) – which is now done by numerically solving Equation 32 initialized at y 0 = Y 0 y_{0}=Y_{0} . In this case the log det \log\det term in the change-of-variables formula takes the form ( Chen et al., 2018 ) 

 

 
 | 
 log det ∇ x 0 f θ ( x 0 ) = ∫ 0 T tr ( ∇ x t v θ ( x t , t ) ) d t , \log\det\nabla_{x_{0}}f_{\theta}(x_{0})=\int_{0}^{T}\tr\left(\nabla_{x_{t}}v_{\theta}(x_{t},t)\right){\textnormal{d}}t, | 
 | 
 (34) | 
 

 thus enabling maximum-likelihood training of continuous NFs through

 

 
 | 
 max θ ⁡ 𝔼 X 0 ∼ p ∗ X ​ [ log ⁡ p Z ​ ( f θ ​ ( X 0 ) ) + ∫ 0 T tr ⁡ ( ∇ x t v θ ​ ( X t , t ) ) ​ d ​ t ] , \max_{\theta}\mathbb{E}_{X_{0}\sim{p}^{X}_{\ast}}\left[\log{p}^{Z}\left(f_{\theta}(X_{0})\right)+\int_{0}^{T}\tr\left(\nabla_{x_{t}}v_{\theta}(X_{t},t)\right){\textnormal{d}}t\right], | 
 | 
 (35) | 
 

 where X t X_{t} corresponds to x t x_{t} when Equation 30 is initialized at x 0 = X 0 ∼ p ∗ X x_{0}=X_{0}\sim{p}^{X}_{\ast} .
While the computations used to train continuous NFs through Equation 35 are significantly different than those used for standard NFs, both models are fundamentally doing the same thing: modelling the data as the distribution obtained by mapping a simple distribution p Z {p}^{Z} such as a Gaussian through a bijective function g θ g_{\theta} (defined explicitly or implicitly), and training the model via maximum-likelihood.

 
 
 Normalizing flows through the lens of the manifold hypothesis 

 
 By construction, NFs – continuous or not – are full-dimensional models and are thus susceptible to manifold overfitting ( Section 4.1 ). There are however other pathologies associated with using NFs in the manifold setting. For example, Cornish et al. (2020) show that if the supports of p Z {p}^{Z} and p ∗ X {p}^{X}_{\ast} are not homeomorphic, 6 6 
 6 
 
 
 
 Recall that two spaces are homeomorphic when there exists a homeomorphism – i.e. a continuous invertible function with continuous inverse – between them. then any normalizing flow which approximates p ∗ X {p}^{X}_{\ast} must have exploding bi-Lipschitz constant. In the manifold setting, the support of p ∗ X {p}^{X}_{\ast} is the d ∗ d^{\ast} -dimensional manifold ℳ \mathcal{M} , which is not homeomorphic to the support of p Z {p}^{Z} , i.e. ℝ D \mathbb{R}^{D} . The exploding bi-Lipschitz constant implied by the result of Cornish et al. (2020) entails that, if NFs converge to a distribution on a low-dimensional manifold (whether this is p ∗ X {p}^{X}_{\ast} or some other distribution), they must do so in a numerically unstable way, which is completely consistent with our result in Section 4.1.1 . These theoretical insights are borne out in practice; for example, Behrmann et al. (2021) show that trained NFs are numerically non-invertible. While perhaps initially surprising, this phenomenon is neatly explained by considering NFs through the lens of the manifold hypothesis, which we return to in Section 5.3.3 .

 
 
 
 

#### Section 4.1.4 Energy-Based Models

 
 Energy-based models ( Xie et al., 2016 ; Du Mordatch, 2019 , EBMs;) are likelihood-based DGMs which construct a density p θ X {p}^{X}_{\theta} by specifying it up to proportionality. More specifically, a neural network E θ : 𝒳 → ℝ E_{\theta}:\mathcal{X}\rightarrow\mathbb{R} , called the energy function, is used to define p θ X {p}^{X}_{\theta} through

 

 
 | 
 p θ X ​ ( x ) ∝ e − E θ ​ ( x ) , {p}^{X}_{\theta}(x)\propto e^{-E_{\theta}(x)}, | 
 | 
 (36) | 
 

 where p θ X {p}^{X}_{\theta} is assumed to be well-defined (i.e. its normalizing constant is presumed finite: ∫ 𝒳 e − E θ ​ ( x ) ​ d ​ x ∞ \int_{\mathcal{X}}e^{-E_{\theta}(x)}{\textnormal{d}}x \infty ). While the likelihood of EBMs is unavailable due to the normalizing constant of p θ X {p}^{X}_{\theta} being intractable, the observation that

 

 
 | 
 ∇ θ ​ log ​ p θ X ​ ( x ) = 𝔼 X ∼ p θ X ​ [ ∇ θ E θ ​ ( X ) ] − ∇ θ E θ ​ ( x ) \nabla_{\theta}\log{p}^{X}_{\theta}(x)=\mathbb{E}_{X\sim{p}^{X}_{\theta}}\left[\nabla_{\theta}E_{\theta}(X)\right]-\nabla_{\theta}E_{\theta}(x) | 
 | 
 (37) | 
 

 enables gradient optimization of the log-likelihood, since

 

 
 | 
 ∇ θ 𝔼 X ∼ p ∗ X ​ [ log ⁡ p θ X ​ ( X ) ] = 𝔼 X ∼ p θ X ​ [ ∇ θ E θ ​ ( X ) ] − 𝔼 X ∼ p ∗ X ​ [ ∇ θ E θ ​ ( X ) ] , \nabla_{\theta}\mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[\log{p}^{X}_{\theta}(X)\right]=\mathbb{E}_{X\sim{p}^{X}_{\theta}}\left[\nabla_{\theta}E_{\theta}(X)\right]-\mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[\nabla_{\theta}E_{\theta}(X)\right], | 
 | 
 (38) | 
 

 and the expectation with respect to p θ X {p}^{X}_{\theta} can be estimated through Markov chain Monte Carlo (MCMC) methods such as Langevin dynamics ( Welling Teh, 2011 ) . In practice, the log-likelihood is maximized by implementing the loss

 

 
 | 
 min θ ⁡ 𝔼 X ∼ p ∗ X ​ [ E θ ​ ( X ) ] − 𝔼 X ∼ p θ ′ X ​ [ E θ ​ ( X ) ] , \min_{\theta}\mathbb{E}_{X\sim{p}^{X}_{\ast}}[E_{\theta}(X)]-\mathbb{E}_{X\sim{p}^{X}_{\theta^{\prime}}}[E_{\theta}(X)], | 
 | 
 (39) | 
 

 where θ ′ = 𝚜𝚝𝚘𝚙𝚐𝚛𝚊𝚍 ⁡ ( θ ) \theta^{\prime}=\mathtt{stopgrad}(\theta) , as it provides the correct gradient with respect to θ \theta . 7 7 
 7 
 
 
 
 𝚜𝚝𝚘𝚙𝚐𝚛𝚊𝚍 \mathtt{stopgrad} is a computational operator, commonly available in automatic differentiation libraries such as PyTorch ( Paszke et al., 2019 ) , which leaves the forward pass unchanged ( θ ′ = θ \theta^{\prime}=\theta ) while ignoring gradients when differentiating ( ∇ θ θ ′ = 0 \nabla_{\theta}\theta^{\prime}=0 ), i.e. 𝔼 X ∼ p θ ′ X ​ [ E θ ​ ( X ) ] = 𝔼 X ∼ p θ X ​ [ E θ ​ ( X ) ] \mathbb{E}_{X\sim p_{\theta^{\prime}}^{X}}[E_{\theta}(X)]=\mathbb{E}_{X\sim{p}^{X}_{\theta}}[E_{\theta}(X)] , yet ∇ θ 𝔼 X ∼ p θ ′ X ​ [ E θ ​ ( X ) ] = 𝔼 X ∼ p θ X ​ [ ∇ θ E θ ​ ( X ) ] \nabla_{\theta}\mathbb{E}_{X\sim p_{\theta^{\prime}}^{X}}[E_{\theta}(X)]=\mathbb{E}_{X\sim{p}^{X}_{\theta}}[\nabla_{\theta}E_{\theta}(X)] even though ∇ θ 𝔼 X ∼ p θ X ​ [ E θ ​ ( X ) ] \nabla_{\theta}\mathbb{E}_{X\sim{p}^{X}_{\theta}}[E_{\theta}(X)] is not in general equal to 𝔼 X ∼ p θ X ​ [ ∇ θ E θ ​ ( X ) ] \mathbb{E}_{X\sim{p}^{X}_{\theta}}[\nabla_{\theta}E_{\theta}(X)] . 

 
 
 Energy-based models through the lens of the manifold hypothesis 

 
 Since by construction p θ X ​ ( x ) 0 {p}^{X}_{\theta}(x) 0 for all x ∈ 𝒳 x\in\mathcal{X} , EBMs are full-dimensional models and are thus susceptible to manifold overfitting ( Section 4.1 ). EBM training and sampling are known to be difficult. A common trick to sample from a trained EBM is to initialize MCMC chains not from noise, but from previous chains held in a replay buffer. The buffer is a set of MCMC samples from chains that have been advanced by the EBM throughout the entire training process ( Du Mordatch, 2019 ; Grathwohl et al., 2020 ) . Alternatively, a mixture of historical training checkpoints can be used for sampling ( Du Mordatch, 2019 ) . These tricks are necessary because EBMs are prone to mode collapse, especially in high-dimensional ambient spaces ( Arbel et al., 2021 ; Loaiza-Ganem et al., 2022a ) . This mode collapse behaviour is often blamed on Langevin dynamics and the multimodality of the target distribution, but can be further explained by the large or unstable gradients in the density landscape of a model that has undergone manifold overfitting.

 
 
 One particularly interesting exception is the normalized autoencoder proposed by Yoon et al. (2021) , which employs the reconstruction error of an autoencoder to define the energy function; i.e.

 

 
 | 
 E θ ​ ( x ) = ‖ x − g θ ​ ( f θ ​ ( x ) ) ‖ 2 2 T , E_{\theta}(x)=\dfrac{\|x-g_{\theta}(f_{\theta}(x))\|_{2}^{2}}{T}, | 
 | 
 (40) | 
 

 where T 0 T 0 is a hyperparameter. 8 8 
 8 
 
 
 
 Note that the distribution of normalized autoencoders depends on the encoder f θ f_{\theta} , which is why we parameterize it with θ \theta instead of auxiliary parameters ϕ \phi ; this encoder need not share parameters with the decoder g θ g_{\theta} . Much like variational autoencoders ( Section 4.1.2 ), despite using an autoencoder-like structure, normalized autoencoders remain full-dimensional models; this is true of EBMs regardless of the choice of energy function. An interesting observation, which to the best of our knowledge has not been previously made, is that the energy function in Equation 40 is lower-bounded by 0 0 , which in turn implies that e − E θ t ​ ( x ) e^{-E_{\theta_{t}}(x)} is upper-bounded, meaning that p θ t X ​ ( x ) p^{X}_{\theta_{t}}(x) cannot be sent to infinity by making E θ t ​ ( x ) E_{\theta_{t}}(x) arbitrarily negative for x ∈ ℳ x\in\mathcal{M} . In particular, manifold overfitting can only occur when the normalizing constant ∫ 𝒳 e − E θ t ​ ( x ) ​ d ​ x \int_{\mathcal{X}}e^{-E_{\theta_{t}}(x)}{\textnormal{d}}x goes to 0 0 .
While it is of course possible for this to happen with arbitrarily flexible networks, we hypothesize that EBMs with lower-bounded energy functions might have an inductive bias which helps them avoid manifold overfitting in practice, albeit likely at the cost of generative quality. This might explain their empirical success at density-based out-of-distribution detection ( Yoon et al., 2023 ) .

 
 
 
 
 

### Section 4.2 Generative Adversarial Networks

 
 Generative adversarial networks ( Goodfellow et al., 2014 ; Radford et al., 2015 , GANs;) use a neural network g θ : 𝒵 → 𝒳 g_{\theta}:\mathcal{Z}\rightarrow\mathcal{X} called the generator, along with a latent distribution p Z {p}^{Z} (usually a standard Gaussian), to specify the model distribution p θ X {p}^{X}_{\theta} . Similarly to normalizing flows ( Section 4.1.3 ), samples X = g θ ​ ( Z ) X=g_{\theta}(Z) from a GAN are obtained by sampling Z ∼ p Z Z\sim{p}^{Z} and transforming the result through g θ g_{\theta} (as in NFs, p θ X {p}^{X}_{\theta} is formally given by the pushforward of p Z {p}^{Z} through g θ g_{\theta} ), although unlike NFs, d = D d=D is not required, and rather d D d D is the standard choice for GANs, so that g θ g_{\theta} need not be invertible. GANs are not likelihood-based models, and in order to train g θ g_{\theta} , they introduce a binary classifier h ϕ : 𝒳 → ( 0 , 1 ) h_{\phi}:\mathcal{X}\rightarrow(0,1) , which is trained to distinguish between real samples X ∼ p ∗ X X\sim{p}^{X}_{\ast} and generated samples X ∼ p θ X X\sim{p}^{X}_{\theta} . The generator g θ g_{\theta} is trained alongside h ϕ h_{\phi} , so as to make the classifier unable to successfully differentiate between real and generated samples:

 

 
 | 
 min θ ⁡ max ϕ ​ 𝔼 X ∼ p ∗ X ​ [ log ⁡ h ϕ ​ ( X ) ] + 𝔼 Z ∼ p Z ⁡ [ log ⁡ ( 1 − h ϕ ​ ( g θ ​ ( Z ) ) ) ] . \min_{\theta}\max_{\phi}\Exp_{X\sim{p}^{X}_{\ast}}\left[\log h_{\phi}(X)\right]+\Exp_{Z\sim{p}^{Z}}\left[\log\left(1-h_{\phi}(g_{\theta}(Z))\right)\right]. | 
 | 
 (41) | 
 

 Assuming that the classifier is arbitrarily flexible, it can be shown that the above objective is equivalent to minimizing the Jensen-Shannon divergence, 𝕁 ​ 𝕊 \mathbb{JS} , between the true distribution and the model,

 

 
 | 
 min θ 𝕁 𝕊 ( p ∗ X ∥ p θ X ) , \min_{\theta}\mathbb{JS}\left({p}^{X}_{\ast}\,\|\,{p}^{X}_{\theta}\right), | 
 | 
 (42) | 
 

 where 𝕁 𝕊 ( p ∥ q ) ≔ 1 2 𝕂 𝕃 ( p ∥ 1 2 p + 1 2 q ) + 1 2 𝕂 𝕃 ( q ∥ 1 2 p + 1 2 q ) \mathbb{JS}(p\,\|\,q)\coloneqq\frac{1}{2}\mathbb{KL}(p\,\|\,\frac{1}{2}p+\frac{1}{2}q)+\frac{1}{2}\mathbb{KL}(q\,\|\,\frac{1}{2}p+\frac{1}{2}q) . This result was first shown by

 
 
 Goodfellow et al. (2014) under the unstated assumption that p θ X {p}^{X}_{\theta} and p ∗ X {p}^{X}_{\ast} are densities supported on the same manifold for all θ \theta . While this assumption is unrealistic in the manifold setting and should not be expected to hold in practice, the proof provided by Goodfellow et al. (2014) is “correct in spirit”, and was later formalized and generalized by Donahue et al. (2017) .

 
 
 Generative adversarial networks through the lens of the manifold hypothesis 

 
 The Jensen-Shannon divergence 𝕁 𝕊 ( p ∥ q ) \mathbb{JS}(p\,\|\,q) is meaningfully defined even when 𝕂 𝕃 ( p ∥ q ) = ∞ \mathbb{KL}(p\,\|\,q)=\infty . At a first glance this might suggest that optimizing the Jensen-Shannon divergence circumvents issues such as manifold overfitting ( Section 4.1 ), which arise from attempting to minimize the KL divergence. However, the gradients of the Jensen-Shannon divergence 𝕁 𝕊 ( p ∗ X ∥ p θ X ) \mathbb{JS}({p}^{X}_{\ast}\,\|\,{p}^{X}_{\theta}) with respect to model parameters θ \theta will be 0 0 whenever the supports of p ∗ X {p}^{X}_{\ast} and p θ X {p}^{X}_{\theta} do not overlap (formally, the Jensen-Shannon divergence does not metrize weak convergence). This property makes gradient optimization futile whenever the support of the GAN has no overlap with the underlying data manifold, which Arjovsky et al. (2017) identify as a cause of training instabilities for GANs. The Jensen-Shannon divergence is a particular instance of an f f -divergence ( Polyanskiy Wu, 2022 ) , and there is work generalizing GANs to minimize f f -divergences ( Nowozin et al., 2016 ) . Yet, Arjovsky et al. (2017) also show that various other f f -divergences, such as the total variation distance and KL divergence, suffer from similar pathologies as the Jensen-Shannon divergence when there is mismatch between the supports of p θ X {p}^{X}_{\theta} and p ∗ X {p}^{X}_{\ast} . In other words, although GANs do not suffer from manifold overfitting, they can still struggle to model manifold-supported data. It is however worth highlighting that the manifold-related woes of GANs are fundamentally different than those of likelihood-based models: the former use a proper low-dimensional model (whenever d D d D ), and the resulting problems are due only to the optimization objective; whereas the latter are full-dimensional models, and are thus misspecified. Still, GANs can remain topologically misspecified, e.g. when ℳ \mathcal{M} is disconnected ( Section 5.4.3 ), but again, this is an inherently different situation than the dimensional misspecification of likelihood-based models.

 
 
 
 

### Section 4.3 Score Matching

 
 Score matching ( Hyvärinen, 2005 ) is a method to learn full-dimensional densities p ∗ X {p}^{X}_{\ast} . The main idea is to learn the (Stein) score function, ∇ x ​ log ​ p ∗ X \nabla_{x}\log{p}^{X}_{\ast} , rather than p ∗ X {p}^{X}_{\ast} itself. 9 9 
 9 
 
 
 
 While in machine learning ∇ x ​ log ​ p θ X \nabla_{x}\log{p}^{X}_{\theta} is often called the score function of a model, in the statistics literature the score function refers to ∇ θ ​ log ​ p θ X \nabla_{\theta}\log{p}^{X}_{\theta} , whereas ∇ x ​ log ​ p θ X \nabla_{x}\log{p}^{X}_{\theta} is called the Stein score. In order to achieve this, a model p θ X {p}^{X}_{\theta} is implicitly characterized by s θ X : 𝒳 → 𝒳 s_{\theta}^{X}:\mathcal{X}\rightarrow\mathcal{X} , whose goal is to approximate the unknown true score function. The Fisher divergence, which is sometimes referred to as the Fisher information distance ( DasGupta, 2008 ) , and which is defined as

 

 
 | 
 𝔽 ⁡ ( p , q ) ≔ 𝔼 X ∼ p ​ [ ‖ ∇ x ​ log ​ q ​ ( X ) − ∇ x ​ log ​ p ​ ( X ) ‖ 2 2 ] , \mathbb{F}(p,q)\coloneqq\mathbb{E}_{X\sim p}\left[\|\nabla_{x}\log q(X)-\nabla_{x}\log p(X)\|_{2}^{2}\right], | 
 | 
 (43) | 
 

 is leveraged for this goal. Ideally the model would be trained by minimizing 𝔽 ⁡ ( p ∗ X , p θ X ) \mathbb{F}({p}^{X}_{\ast},{p}^{X}_{\theta}) as

 

 
 | 
 min θ ⁡ 𝔼 X ∼ p ∗ X ​ [ ‖ s θ X ​ ( X ) − ∇ x ​ log ​ p ∗ X ​ ( X ) ‖ 2 2 ] , \min_{\theta}\mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[\|s_{\theta}^{X}(X)-\nabla_{x}\log{p}^{X}_{\ast}(X)\|_{2}^{2}\right], | 
 | 
 (44) | 
 

 but naïvely doing so requires evaluating the unknown ∇ x ​ log ​ p ∗ X \nabla_{x}\log{p}^{X}_{\ast} . Hyvärinen (2005) showed that, under mild regularity conditions,

 

 
 | 
 𝔽 ⁡ ( p , q ) = 𝔼 X ∼ p ​ [ ‖ ∇ x ​ log ​ q ​ ( X ) ‖ 2 2 + 2 ​ tr ​ ∇ x 2 ​ log ⁡ q ⁡ ( X ) ] + c ⁡ ( p ) , \mathbb{F}(p,q)=\mathbb{E}_{X\sim p}\left[\|\nabla_{x}\log q(X)\|_{2}^{2}+2\tr\nabla_{x}^{2}\log q(X)\right]+c(p), | 
 | 
 (45) | 
 

 where c ⁡ ( p ) c(p) is a term which depends only on p p . Since c ⁡ ( p ∗ X ) c({p}^{X}_{\ast}) is a constant with respect to θ \theta , the objective in Equation 44 is thus equivalent to

 

 
 | 
 min θ ⁡ 𝔼 X ∼ p ∗ X ​ [ ‖ s θ X ​ ( X ) ‖ 2 2 + 2 ​ tr ⁡ ∇ x s θ X ​ ( X ) ] , \min_{\theta}\mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[\|s_{\theta}^{X}(X)\|_{2}^{2}+2\tr\nabla_{x}s_{\theta}^{X}(X)\right], | 
 | 
 (46) | 
 

 which can actually be minimized.
Once a model is trained, Markov chain Monte Carlo methods such as Langevin dynamics can be used to sample from it,
similarly to energy-based models ( Section 4.1.4 ).

 
 
 Score matching through the lens of the manifold hypothesis 

 
 As mentioned above, score matching is derived under the assumption that the underlying data distribution p ∗ X {p}^{X}_{\ast} is full-dimensional. While score matching has been extended to known manifolds ( Mardia et al., 2016 ) , we are not aware of any work theoretically studying dimensional mispecification within score matching in an analogous manner to how Loaiza-Ganem et al. (2022a) characterize manifold overfitting ( Section 4.1 ) within likelihood-based models.
Nonetheless, we should intuitively expect score matching to fail under the manifold setting due to this dimensional misspecification. To see why this is the case, we begin by noting that p θ X {p}^{X}_{\theta} is indeed full-dimensional since s θ X s_{\theta}^{X} takes inputs from all of 𝒳 \mathcal{X} rather than just ℳ \mathcal{M} ( s θ X s_{\theta}^{X} is evaluated at potentially any point in 𝒳 \mathcal{X} during sampling when using procedures such as Langevin dynamics). The score functions s θ X s_{\theta}^{X} and ∇ x ​ log ​ p ∗ X \nabla_{x}\log{p}^{X}_{\ast} are thus different types of objects – the former is a full-dimensional score function and the latter is a manifold-supported one. Comparing the values of dimensionally-mismatched densities is not meaningful, and the comparison remains equally meaningless between the corresponding score functions. Consequently, there is no reason to expect Equation 44 to succeed at matching p θ X {p}^{X}_{\theta} to p ∗ X {p}^{X}_{\ast} in the presence of dimensional misspecification.
This issue was identified by Song Ermon (2019) , who empirically confirm that score matching struggles in the manifold setting.

 
 
 
 
 

## Section 5 Manifold-Aware Deep Generative Models

 
 As covered throughout Section 4 , many commonly-used DGMs struggle to learn distributions on unknown manifolds. There are various (not always mutually exclusive) approaches that enable manifold-awareness, including judiciously adding noise to the target distribution; using support-agnostic optimization objectives (e.g. those which metrize weak convergence); and two-step models, which carry out generative modelling on a low-dimensional latent space and then map back to data space. We review these approaches in Section 5.1 , Section 5.2 , and Section 5.3 , respectively. In Section 5.3.1 we show that ( i ) (i) two-step models can be interpreted as (potentially regularized) minimizers of an upper bound of the Wasserstein distance, thus establishing a link between these different approaches for achieving manifold-awareness, and that ( i ​ i ) (ii) the upper bound becomes tight at optimality whenever an autoencoder can achieve perfect reconstructions. Finally, in Section 5.4 we cover methods which make an explicit attempt at properly capturing the topology of ℳ \mathcal{M} . We take a lax interpretation of manifold-awareness throughout, and discuss not only DGMs which are formally manifold-aware, but also those which, while mathematically manifold-unaware, leverage some inductive bias towards manifold-awareness.

 
 

### Section 5.1 Manifold-Awareness by Adding Noise

 
 When manifold-unawareness arises due to the mismatch between the dimension of the model and that of the true distribution – as is the case for likelihood-based models ( Section 4.1 ) – adding noise to the training data seems like a natural solution; this can make the target distribution full-dimensional (e.g. by convolving the true distribution with a Gaussian) and thus hopefully avoids manifold-related problems. Indeed, dequantization – i.e. the practice of adding noise to data that was discretized so as to be able to fit a continuous density model – is very common ( Theis et al., 2016 ; Dinh et al., 2017 ; Ho et al., 2019 ) , and can be further justified as a way to avoid manifold overfitting. Unfortunately, it has been shown that just adding Gaussian noise is not enough to empirically avoid manifold overfitting ( Zhang et al., 2020a ; Loaiza-Ganem et al., 2022a ; Loaiza-Ganem et al., 2022b ) . Even though the theoretical conditions for manifold overfitting do not hold anymore, the new (noisy) target density will be extremely peaked around ℳ \mathcal{M} ( Section 4.1.1 ), and thus still numerically exposed to manifold-related woes. This observation is consistent with known convergence rates at which DGMs trained on noisy data recover p ∗ X {p}^{X}_{\ast} ( Chae et al., 2023 ) . The lesson here is that adding noise can enable DGMs to learn unknown manifolds, but the noise has to be added carefully. Various methods doing so have been proposed, which we now review.

 
 

#### Section 5.1.1 Denoising Score Matching

 
 As previously mentioned, Song Ermon (2019) showed that score matching ( Section 4.3 ) struggles to model manifold-supported data, and they thus advocate for adding noise and using denoising score matching ( Vincent, 2011 ) instead. In denoising score matching, the target distribution is not p ∗ X {p}^{X}_{\ast} anymore, but rather the distribution p ∗ X σ p^{X_{\sigma}}_{\ast} obtained by adding independent Gaussian noise 𝒩 ⁡ ( ⋅ , 0 , σ 2 ​ I D ) \mathcal{N}(\ \cdot\ ;0,\sigma^{2}I_{D}) to samples from p ∗ X {p}^{X}_{\ast} , where σ 2 \sigma^{2} is a hyperparameter. More formally, p ∗ X σ ≔ p ∗ X ⊛ 𝒩 ⁡ ( ⋅ , 0 , σ 2 ​ I D ) p^{X_{\sigma}}_{\ast}\coloneqq{p}^{X}_{\ast}\circledast\mathcal{N}(\ \cdot\ ;0,\sigma^{2}I_{D}) . Importantly, adding full-dimensional Gaussian noise ensures that p ∗ X σ p^{X_{\sigma}}_{\ast} is always full-dimensional, regardless of the support of p ∗ X {p}^{X}_{\ast} . Score matching can then be applied to learn a network s θ X s_{\theta}^{X} to approximate ∇ x σ ​ log ​ p ∗ X σ \nabla_{x_{\sigma}}\log p^{X_{\sigma}}_{\ast} through Equation 46 (with p ∗ X {p}^{X}_{\ast} replaced by p ∗ X σ p_{\ast}^{X_{\sigma}} ). However, Vincent (2011) shows that this objective is equivalent to

 

 
 | 
 min θ 𝔼 X 0 ∼ p ∗ X [ 𝔼 X σ ∼ p ∗ X σ | X 0 ( ⋅ | X 0 ) [ ∥ s θ X ( X σ ) − ∇ x σ log p ∗ X σ | X 0 ( X σ | X 0 ) ∥ 2 2 ] ] , \min_{\theta}\mathbb{E}_{X_{0}\sim{p}^{X}_{\ast}}\left[\mathbb{E}_{X_{\sigma}\sim p_{\ast}^{X_{\sigma}|X_{0}}(\cdot|X_{0})}\left[\|s_{\theta}^{X}(X_{\sigma})-\nabla_{x_{\sigma}}\log p_{\ast}^{X_{\sigma}|X_{0}}(X_{\sigma}|X_{0})\|_{2}^{2}\right]\right], | 
 | 
 (47) | 
 

 where p ∗ X σ | X 0 ​ ( x σ | x 0 ) = 𝒩 ⁡ ( x σ , x 0 , σ 2 ​ I D ) p_{\ast}^{X_{\sigma}|X_{0}}(x_{\sigma}|x_{0})=\mathcal{N}(x_{\sigma};x_{0},\sigma^{2}I_{D}) is the density of noisy data (denoted X σ X_{\sigma} ) given the (un-noised) datapoint X 0 = x 0 X_{0}=x_{0} . Equation 47 is much easier to optimize than the usual score matching objective ( Equation 46 ), since there is no need to backpropagate through the trace of the Jacobian of s θ X s_{\theta}^{X} .
 Saremi Hyvärinen (2019) use the loss function in Equation 47 to learn an energy-based model ( Section 4.1.4 ) on the noised-out data, but derive the loss from the perspective of empirical Bayes ( Robbins, 1956 ) ; their approach can in principle be applied to other noising processes, but only the Gaussian case is implemented in their work.

 
 
 Despite mathematically avoiding manifold-related pathologies, denoising score matching as presented above faces a tradeoff; setting σ \sigma to a very small value means that the target density p ∗ X σ p_{\ast}^{X_{\sigma}} is closer to the actual manifold-supported data density p ∗ X {p}^{X}_{\ast} , but doing so also means the target density is highly peaked around ℳ \mathcal{M} and thus might be harder to properly learn ( Section 4.1.1 ). As a way of being able to use small amounts of noise while still efficiently learning the resulting distribution, Song Ermon (2019) propose to use various noise levels. More specifically, they consider fixed noise levels 0 σ 1 σ 2 ⋯ σ T 0 \sigma_{1} \sigma_{2} \dots \sigma_{T} , and modify the score function to take the noise level as input; i.e. s θ X : 𝒳 × [ σ 1 , σ T ] → 𝒳 s_{\theta}^{X}:\mathcal{X}\times[\sigma_{1},\sigma_{T}]\rightarrow\mathcal{X} is now such that it aims to approximate the score function at all the corresponding noise levels: s θ X ​ ( ⋅ , σ t ) ≈ ∇ x σ t ​ log ​ p ∗ X σ t s_{\theta}^{X}(\ \cdot\ ,\sigma_{t})\approx\nabla_{x_{\sigma_{t}}}\log p^{X_{\sigma_{t}}}_{\ast} for t = 1 , … , T t=1,\dots,T . This new score function is trained with a weighted sum of the corresponding denoising score matching objectives,

 

 
 | 
 min θ ∑ t = 1 T w ( t ) 𝔼 X 0 ∼ p ∗ X [ 𝔼 X σ t ∼ p ∗ X σ t | X 0 ( ⋅ | X 0 ) [ ∥ s θ X ( X σ t , σ t ) − ∇ x σ t log p ∗ X σ t | X 0 ( X σ t | X 0 ) ∥ 2 2 ] ] , \min_{\theta}\sum_{t=1}^{T}w(t)\mathbb{E}_{X_{0}\sim{p}^{X}_{\ast}}\left[\mathbb{E}_{X_{\sigma_{t}}\sim p_{\ast}^{X_{\sigma_{t}}|X_{0}}(\cdot|X_{0})}\left[\|s_{\theta}^{X}(X_{\sigma_{t}},\sigma_{t})-\nabla_{x_{\sigma_{t}}}\log p^{X_{\sigma_{t}}|X_{0}}_{\ast}(X_{\sigma_{t}}|X_{0})\|_{2}^{2}\right]\right], | 
 | 
 (48) | 
 

 where w ⁡ ( t ) 0 w(t) 0 is a pre-specified weight coefficient which aims to keep the T T terms in the sum at roughly equal magnitudes. The intuition behind using varying noise levels is twofold. ( i ) (i) Learning the score function for larger values of σ \sigma is easier, and thanks to parameter sharing ( θ \theta is the same for all noise levels), doing so is helpful for learning the score function for small values of σ \sigma .
 ( i ​ i ) (ii) Once the model is trained, different noise levels are also used within an annealed sampling scheme. s θ ∗ X ​ ( ⋅ , σ T ) s_{\theta^{\ast}}^{X}(\ \cdot\ ,\sigma_{T}) is used alongside Markov chain Monte Carlo to generate a sample, which is then used to initialize another Markov chain that now uses s θ ∗ X ​ ( ⋅ , σ T − 1 ) s_{\theta^{\ast}}^{X}(\ \cdot\ ,\sigma_{T-1}) ; this process is repeated until s θ ∗ X ​ ( ⋅ , σ 1 ) s_{\theta^{\ast}}^{X}(\ \cdot\ ,\sigma_{1}) is used – and works much better than only using s θ ∗ X ​ ( ⋅ , σ 1 ) s_{\theta^{\ast}}^{X}(\ \cdot\ ,\sigma_{1}) . Although this scheme produces approximate samples from p ∗ X σ 1 p_{\ast}^{X_{\sigma_{1}}} rather than from p ∗ X {p}^{X}_{\ast} , as long as σ 1 \sigma_{1} is small enough, the difference is negligible in practice.

 
 
 

#### Section 5.1.2 Score-Based Diffusion Models

 
 Song et al. (2021b) proposed score-based diffusion models as an extension of denoising score matching ( Section 5.1.1 ) where there is a continuum of noise levels. Formally, they achieve this by constructing ( X t ) t ∈ [ 0 , T ] (X_{t})_{t\in[0,T]} , where X t ∈ 𝒳 X_{t}\in\mathcal{X} for every t ∈ [ 0 , T ] t\in[0,T] , as an Ornstein–Uhlenbeck process given by the Itô stochastic differential equation (SDE), 10 10 
 10 
 
 
 
 Readers unfamiliar with SDEs can understand Equation 49 through its Euler-Maruyama discretization: split [ 0 , T ] [0,T] into n n sub-intervals of equal length Δ t , n = T / n \Delta_{t,n}=T/n , sample X 0 , n ∼ p ∗ X X_{0,n}\sim{p}^{X}_{\ast} , and set X t k + 1 , n = X t k , n − β ⁡ ( t k ) 2 ​ X t k , n ​ Δ t , n + β ⁡ ( t k ) ​ Δ B t k , n X_{t_{k+1},n}=X_{t_{k},n}-\tfrac{\beta(t_{k})}{2}X_{t_{k},n}\Delta_{t,n}+\sqrt{\beta(t_{k})}\Delta_{B_{t_{k}},n} for k = 0 , … , n − 1 k=0,\dots,n-1 , where t k = k ​ Δ t , n t_{k}=k\Delta_{t,n} , and where Δ B t k , n = B t k + 1 − B t k ∼ 𝒩 ⁡ ( ⋅ , 0 , Δ t , n ​ I D ) \Delta_{B_{t_{k}},n}=B_{t_{k+1}}-B_{t_{k}}\sim\mathcal{N}(\ \cdot\ ;0,\Delta_{t,n}I_{D}) are independent. This procedure characterizes X t , n X_{t,n} at the times t k t_{k} for k = 0 , … , n k=0,\dots,n , and linearly interpolating between them yields a continuous stochastic process ( X t , n ) t ∈ [ 0 , T ] (X_{t,n})_{t\in[0,T]} , the limit of which as n → ∞ n\rightarrow\infty corresponds to the process specified by the SDE. 

 

 
 | 
 d ​ X t = − β ⁡ ( t ) 2 ​ X t ​ d ​ t + β ⁡ ( t ) ​ d ​ B t , X 0 ∼ p ∗ X , \displaystyle\begin{split} {\textnormal{d}}X_{t}=-\dfrac{\beta(t)}{2}X_{t}{\textnormal{d}}t+\sqrt{\beta(t)}{\textnormal{d}}B_{t},\\
 X_{0}\sim{p}^{X}_{\ast},\end{split} | 
 | 
 (49) | 
 

 where β : [ 0 , T ] → ℝ + \beta:[0,T]\rightarrow\mathbb{R}_{+} is a hyperparameter (often an affine function, i.e. β ⁡ ( t ) = β min + ( β max − β min ) ​ t / T \beta(t)=\beta_{\text{min}}+(\beta_{\text{max}}-\beta_{\text{min}})t/T , where 0 β min β max 0 \beta_{\text{min}} \beta_{\text{max}} ), and ( B t ) t ∈ [ 0 , T ] (B_{t})_{t\in[0,T]} denotes a D D -dimensional Brownian motion. Other choices of SDE are possible, but we focus on the one above – which is often referred to as a variance preserving SDE – since it is assumed in some of the theoretical results that we will shortly discuss. Under mild regularity conditions, the SDE admits a unique solution ( Øksendal, 2003 ) , thus characterizing the density p ∗ X t p_{\ast}^{X_{t}} of X t X_{t} for every t ∈ [ 0 , T ] t\in[0,T] , and prescribes how to progressively transform the data density p ∗ X 0 = p ∗ X p_{\ast}^{X_{0}}={p}^{X}_{\ast} into the noisier density p ∗ X T p_{\ast}^{X_{T}} . Note that, due to the added Gaussian noise (from the Brownian motion in Equation 49 ), p ∗ X t p^{X_{t}}_{\ast} is a full-dimensional density for every t ∈ ( 0 , T ] t\in(0,T] regardless of the support of p ∗ X {p}^{X}_{\ast} .

 
 
 Reversing ( X t ) t ∈ [ 0 , T ] (X_{t})_{t\in[0,T]} provides a way to transform samples from p ∗ X T p_{\ast}^{X_{T}} into samples from p ∗ X {p}^{X}_{\ast} . The reverse process ( Y t ) t ∈ [ 0 , T ] ≔ ( X T − t ) t ∈ [ 0 , T ] (Y_{t})_{t\in[0,T]}\coloneqq(X_{T-t})_{t\in[0,T]} also obeys an SDE ( Anderson, 1982 ; Haussmann Pardoux, 1986 ) : 11 11 
 11 
 
 
 
 Note that the Brownian motions in Equation 49 and Equation 50 are not in general the same Brownian motion (they just have the same distribution), but we do not differentiate between them for notational simplicity.
Note also that Equation 50 differs from the corresponding equation in ( Song et al., 2021b ) since d ​ t {\textnormal{d}}t in our notation corresponds to − d ​ t -{\textnormal{d}}t in theirs. 

 

 
 | 
 d ​ Y t = β ⁡ ( T − t ) ​ ( Y t 2 + ∇ y t ​ log ​ p ∗ X T − t ​ ( Y t ) ) ​ d ​ t + β ⁡ ( T − t ) ​ d ​ B t , Y 0 ∼ p ∗ X T . \displaystyle\begin{split} {\textnormal{d}}Y_{t}=\beta(T-t)\left(\dfrac{Y_{t}}{2}+\nabla_{y_{t}}\log p_{\ast}^{X_{T-t}}(Y_{t})\right){\textnormal{d}}t+\sqrt{\beta(T-t)}{\textnormal{d}}B_{t},\\
 Y_{0}\sim p_{\ast}^{X_{T}}.\end{split} | 
 | 
 (50) | 
 

 The main idea of score-based diffusion models is to leverage this reverse SDE to build a generative model. In order to achieve this, some approximations are needed. First, since p ∗ X T p_{\ast}^{X_{T}} is not known exactly, Equation 50 is initialized at a known distribution p ∗ X ∞ p_{\ast}^{X_{\infty}} . Formally, ( Y t ) t ∈ [ 0 , T ] (Y_{t})_{t\in[0,T]} is approximated by ( Y ~ t ) t ∈ [ 0 , T ] (\tilde{Y}_{t})_{t\in[0,T]} , where

 

 
 | 
 d ​ Y ~ t = β ⁡ ( T − t ) ​ ( Y ~ t 2 + ∇ y ~ t ​ log ​ p ∗ X T − t ​ ( Y ~ t ) ) ​ d ​ t + β ⁡ ( T − t ) ​ d ​ B t , Y ~ 0 ∼ p ∗ X ∞ . \displaystyle\begin{split} {\textnormal{d}}\tilde{Y}_{t}=\beta(T-t)\left(\dfrac{\tilde{Y}_{t}}{2}+\nabla_{\tilde{y}_{t}}\log p_{\ast}^{X_{T-t}}(\tilde{Y}_{t})\right){\textnormal{d}}t+\sqrt{\beta(T-t)}{\textnormal{d}}B_{t},\\
 \tilde{Y}_{0}\sim p_{\ast}^{X_{\infty}}.\end{split} | 
 | 
 (51) | 
 

 We denote the density of Y ~ t \tilde{Y}_{t} as p ~ X T − t \tilde{p}^{X_{T-t}} , and will shortly explain how p ∗ X ∞ p_{\ast}^{X_{\infty}} is chosen so as to be close to p ∗ X T p_{\ast}^{X_{T}} . The score ∇ y ~ t ​ log ​ p ∗ X T − t ​ ( Y ~ t ) \nabla_{\tilde{y}_{t}}\log p_{\ast}^{X_{T-t}}(\tilde{Y}_{t}) is also unknown, and thus must be approximated as well. Score-based diffusion models leverage neural networks to construct s θ X : 𝒳 × ( 0 , T ] → 𝒳 s_{\theta}^{X}:\mathcal{X}\times(0,T]\rightarrow\mathcal{X} with the goal of approximating this function, i.e. s θ X ​ ( x , t ) ≈ ∇ x ​ log ​ p ∗ X t ​ ( x ) s_{\theta}^{X}(x,t)\approx\nabla_{x}\log p_{\ast}^{X_{t}}(x) for all x ∈ 𝒳 x\in\mathcal{X} and t ∈ ( 0 , T ] t\in(0,T] . We will also soon explain how this network is trained, but for a given s θ X s_{\theta}^{X} , ( Y ~ t ) t ∈ [ 0 , T ] (\tilde{Y}_{t})_{t\in[0,T]} is approximated by ( Y ^ t ) t ∈ [ 0 , T ] (\hat{Y}_{t})_{t\in[0,T]} , where

 

 
 | 
 d ​ Y ^ t = β ⁡ ( T − t ) ​ ( Y ^ t 2 + s θ X ​ ( Y ^ t , T − t ) ) ​ d ​ t + β ⁡ ( T − t ) ​ d ​ B t , Y ^ 0 ∼ p ∗ X ∞ . \displaystyle\begin{split} {\textnormal{d}}\hat{Y}_{t}=\beta(T-t)\left(\dfrac{\hat{Y}_{t}}{2}+s_{\theta}^{X}(\hat{Y}_{t},T-t)\right){\textnormal{d}}t+\sqrt{\beta(T-t)}{\textnormal{d}}B_{t},\\
 \hat{Y}_{0}\sim p_{\ast}^{X_{\infty}}.\end{split} | 
 | 
 (52) | 
 

 We denote the density of Y ^ t \hat{Y}_{t} as p ^ θ X T − t \hat{p}_{\theta}^{X_{T-t}} . Ideally, a model sample would be obtained by perfectly solving Equation 52 . In practice this SDE must be discretized and a numerical solver must be used. The model distribution p θ X {p}^{X}_{\theta} is thus given by the approximate solution of Equation 52 at time T T .

 
 
 In summary, diffusion models aim to solve Equation 50 , since perfectly doing so would yield samples from p ∗ X {p}^{X}_{\ast} , but this is impossible and three sources of error have to be introduced to approximately solve this equation. ( i ) (i) p ∗ X T p^{X_{T}}_{\ast} is unknown, and is thus approximated by p ∗ X ∞ p^{X_{\infty}}_{\ast} ; ( i ​ i ) (ii) the true score ∇ x ​ log ​ p ∗ X T − t ​ ( x ) \nabla_{x}\log p^{X_{T-t}}_{\ast}(x) is also unknown, and is thus approximated by s θ X ​ ( x , T − t ) s_{\theta}^{X}(x,T-t) ; and ( i ​ i ​ i ) (iii) the resulting SDE in Equation 52 must be solved numerically, inducing discretization error.

 
 
 We have not yet discussed how diffusion models are trained. Before doing so, we point out that Equation 49 has the known transition kernel

 

 
 | 
 p ∗ X t | X 0 ​ ( x t | x 0 ) = 𝒩 ⁡ ( x t , 1 − σ t 2 ​ x 0 , σ t 2 ​ I D ) , p^{X_{t}|X_{0}}_{\ast}(x_{t}|x_{0})=\mathcal{N}\left(x_{t};\sqrt{1-\sigma_{t}^{2}}x_{0},\sigma_{t}^{2}I_{D}\right), | 
 | 
 (53) | 
 

 where p ∗ X t | X 0 ( ⋅ | x 0 ) p^{X_{t}|X_{0}}_{\ast}(\cdot|x_{0}) is the conditional density of X t X_{t} given that X 0 = x 0 X_{0}=x_{0} , and where

 

 
 | 
 σ t 2 = 1 − e − ∫ 0 t β ( s ) d s . \sigma_{t}^{2}=1-e^{-\int_{0}^{t}\beta(s){\textnormal{d}}s}. | 
 | 
 (54) | 
 

 Thanks to Equation 53 , ∇ x t ​ log ​ p ∗ X t | X 0 ​ ( x t | x 0 ) \nabla_{x_{t}}\log p^{X_{t}|X_{0}}_{\ast}(x_{t}|x_{0}) can be evaluated, and sampling from p ∗ X t | X 0 ( ⋅ | x 0 ) p^{X_{t}|X_{0}}_{\ast}(\cdot|x_{0}) is very straightforward. Together, these points imply that denoising score matching ( Section 5.1.1 ) provides a tractable objective for training diffusion models: 12 12 
 12 
 
 
 
 While the target conditional densities in Equation 48 and Equation 55 – i.e. 𝒩 ⁡ ( x σ t , x 0 , σ t 2 ​ I D ) \mathcal{N}(x_{\sigma_{t}};x_{0},\sigma^{2}_{t}I_{D}) and 𝒩 ⁡ ( x t , 1 − σ t 2 ​ x 0 , σ t 2 ​ I D ) \mathcal{N}(x_{t};\sqrt{1-\sigma^{2}_{t}}x_{0},\sigma^{2}_{t}I_{D}) , respectively – differ in that the mean of the latter is scaled by 1 − σ t 2 \sqrt{1-\sigma^{2}_{t}} , the result of Vincent (2011) which justifies denoising score matching can be easily adapted to this scaled setting, so that Equation 55 is minimized when s θ X ​ ( ⋅ , t ) s_{\theta}^{X}(\ \cdot\ ,t) matches the true score function ∇ x t ​ log ​ p ∗ X t \nabla_{x_{t}}\log p^{X_{t}}_{\ast} . 

 

 
 | 
 min θ ∫ 0 T w ( t ) 𝔼 X 0 ∼ p ∗ X [ 𝔼 X t ∼ p X t | X 0 ∗ ( ⋅ | X 0 ) [ ∥ s θ X ( X t , t ) − ∇ x t log p ∗ X t | X 0 ( X t | X 0 ) ∥ 2 2 ] ] d t , \min_{\theta}\int_{0}^{T}w(t)\mathbb{E}_{X_{0}\sim{p}^{X}_{\ast}}\left[\mathbb{E}_{X_{t}\sim p^{X_{t}|X_{0}}_{\ast}(\cdot|X_{0})}\left[\|s_{\theta}^{X}(X_{t},t)-\nabla_{x_{t}}\log p^{X_{t}|X_{0}}_{\ast}(X_{t}|X_{0})\|_{2}^{2}\right]\right]{\textnormal{d}}t, | 
 | 
 (55) | 
 

 where w : [ 0 , T ] → ℝ + w:[0,T]\rightarrow\mathbb{R}_{+} is a weighting function (set as a hyperparameter).

 
 
 Additionally, as long as β \beta is such that σ T 2 → 1 \sigma_{T}^{2}\rightarrow 1 as T → ∞ T\rightarrow\infty , Equation 53 also implies that p ∗ X T | X 0 ( ⋅ | x 0 ) p^{X_{T}|X_{0}}_{\ast}(\cdot|x_{0}) stops depending on x 0 x_{0} in the sense that it converges to 𝒩 ⁡ ( ⋅ , 0 , I D ) \mathcal{N}(\ \cdot\ ;0,I_{D}) as T → ∞ T\rightarrow\infty : this observation provides an avenue for approximately sampling from p ∗ X T p^{X_{T}}_{\ast} , namely by setting p ∗ X ∞ p^{X_{\infty}}_{\ast} to a standard Gaussian.

 
 
 Finally, Song et al. (2021b) also show that the SDE in Equation 49 is intimately linked to the ODE

 

 
 | 
 d ​ x t = − β ⁡ ( t ) 2 ​ ( x t + ∇ x t ​ log ​ p ∗ X t ​ ( x t ) ) ​ d ​ t , x 0 ∈ 𝒳 . \displaystyle\begin{split} {\textnormal{d}}x_{t}=-\dfrac{\beta(t)}{2}\left(x_{t}+\nabla_{x_{t}}\log p_{\ast}^{X_{t}}(x_{t})\right){\textnormal{d}}t,\\
 x_{0}\in\mathcal{X}.\end{split} | 
 | 
 (56) | 
 

 These equations are related in that, under some regularity conditions, if the ODE is initialized at x 0 = X 0 ∼ p ∗ X x_{0}=X_{0}\sim{p}^{X}_{\ast} , then x T x_{T} will have the same distribution as X T X_{T} . Equation 56 can of course not be solved because the true score function is unknown, but it can be approximated by replacing it with the learned score function, resulting in the new ODE:

 

 
 | 
 d ​ x ^ t = − β ⁡ ( t ) 2 ​ ( x ^ t + s θ ∗ X ​ ( x ^ t , t ) ) ​ d ​ t , x ^ 0 ∈ 𝒳 . \displaystyle\begin{split} {\textnormal{d}}\hat{x}_{t}=-\dfrac{\beta(t)}{2}\left(\hat{x}_{t}+s_{\theta^{\ast}}^{X}(\hat{x}_{t},t)\right){\textnormal{d}}t,\\
 \hat{x}_{0}\in\mathcal{X}.\end{split} | 
 | 
 (57) | 
 

 This equation allows us to interpret diffusion models as continuous normalizing flows ( Section 4.1.3 ): v θ ∗ ​ ( x , t ) v_{\theta^{\ast}}(x,t) in Equation 30 is given by − β ⁡ ( t ) 2 ​ ( x + s θ ∗ X ​ ( x , t ) ) -\tfrac{\beta(t)}{2}(x+s_{\theta^{\ast}}^{X}(x,t)) . The connection between diffusion models and continuous NFs has two relevant consequences. ( i ) (i) It allows for an alternative way of sampling from them: instead of solving Equation 52 , the ODE in Equation 57 can be reversed in time as in Equation 32 to obtain

 

 
 | 
 d ​ y ^ t = β ⁡ ( T − t ) 2 ​ ( y ^ t + s θ ∗ X ​ ( y ^ t , T − t ) ) ​ d ​ t , y ^ 0 ∈ 𝒳 . \displaystyle\begin{split} {\textnormal{d}}\hat{y}_{t}=\dfrac{\beta(T-t)}{2}\left(\hat{y}_{t}+s_{\theta^{\ast}}^{X}(\hat{y}_{t},T-t)\right){\textnormal{d}}t,\\
 \hat{y}_{0}\in\mathcal{X}.\end{split} | 
 | 
 (58) | 
 

 Initializing this ODE at y ^ 0 = Y ^ 0 ∼ p ∗ X ∞ \hat{y}_{0}=\hat{Y}_{0}\sim p_{\ast}^{X_{\infty}} (which plays the role of p Z {p}^{Z} in continuous NFs) and solving it will then result in samples from p ∗ X {p}^{X}_{\ast} if the score function was properly learned, and provided that Equation 57 and Equation 58 admit unique solutions which are inverses of each other. ( i ​ i ) (ii) The connection to continuous NFs is also used to justify using the change-of-variables formula ( Equation 34 ) for evaluating the density p ^ θ X 0 \hat{p}^{X_{0}}_{\theta} implicitly defined by s θ X s_{\theta}^{X} .

 
 
 Diffusion models through the lens of the manifold hypothesis 

 
 There are deep connections between diffusion models and manifolds. First, score-based diffusion models are linked to maximum-likelihood. For example, Sohl-Dickstein et al. (2015) and Ho et al. (2020) formulate diffusion models not through SDEs, but through variational inference. In this formulation, the time interval [ 0 , T ] [0,T] is discretized, X 0 X_{0} still corresponds to data, and X t X_{t} for t 0 t 0 is treated as a latent variable. Then, the (discretized) forward process from Equation 49 corresponds to a fixed variational approximation to the posterior distribution (of latents given data), and the backward process from Equation 52 provides a likelihood term (of data given latents). The resulting model, which is reminiscent of a variational autoencoder ( Section 4.1.2 ), can be trained either by maximizing an ELBO (similar to Equation 24 ) or a reweighted version of it. Song et al. (2021a) establish another connection between score-based diffusion models and maximum-likelihood: under the assumption that p ∗ X {p}^{X}_{\ast} is a full-dimensional density (which does not hold in the manifold setting) and some other regularity conditions, the denoising score matching objective from Equation 55 , with a specific choice of weighting function w w , becomes equivalent to minimizing an upper bound of 𝕂 𝕃 ( p ∗ X ∥ p ^ θ X 0 ) \mathbb{KL}({p}^{X}_{\ast}\,\|\,\hat{p}_{\theta}^{X_{0}}) which becomes tight at optimality. 13 13 
 13 
 
 
 
 It is worthwhile to highlight that Kwon et al. (2022) proved a similar result with Wasserstein distance ( Section 3.5 ), namely that the denoising score matching objective used to train diffusion models provides, up to scaling and constant factors, an upper bound of the 𝕎 2 \mathbb{W}_{2} distance between the model and p ∗ X {p}^{X}_{\ast} which becomes tight at optimality. Unfortunately, Kwon et al. (2022) also assume p ∗ X {p}^{X}_{\ast} is full-dimensional, and thus their result cannot be used to justify the manifold-awareness of diffusion models. In other words, when p ∗ X {p}^{X}_{\ast} is full-dimensional and for a particular choice of w w , diffusion models are likelihood-based models (provided that one ignores the approximation error between p ∗ X T p^{X_{T}}_{\ast} and p ∗ X ∞ p^{X_{\infty}}_{\ast} , and the discretization error, but intuitively both of these errors can be made small by choosing a large T T and making the discretization sufficiently fine, respectively).

 
 
 Naïvely, both of these views of diffusion models suggest that they are a likelihood-based method and thus susceptible to manifold overfitting ( Section 4.1 ).
However, a competing intuition is that the denoising carried out through the backward SDE ( Equation 50 or its approximation Equation 52 ) effectively projects noisy data onto ℳ \mathcal{M} by removing the noise ( Kadkhodaie Simoncelli, 2021 ) . Fortunately, it is the latter intuition that turns out to be correct: note that diffusion models do not only aim to learn the true manifold-supported data density p ∗ X = p ∗ X 0 {p}^{X}_{\ast}=p^{X_{0}}_{\ast} , but also its noisy full-dimensional versions p ∗ X t p^{X_{t}}_{\ast} for every t ∈ ( 0 , T ] t\in(0,T] .
Indeed, the objective in Equation 55 recovers the score function for all t ∈ ( 0 , T ] t\in(0,T] , and stopping the reverse process from Equation 50 at time T − t T-t instead of T T results in a sample from p ∗ X t p^{X_{t}}_{\ast} . Since p ∗ X t p^{X_{t}}_{\ast} is full-dimensional for t 0 t 0 , we should not expect maximum-likelihood to fail at learning it, and by continuity of the solutions of Equation 50 , we should expect p ∗ X 0 p^{X_{0}}_{\ast} to be properly learned, even under the manifold setting . Pidstrigach (2022) formalizes this intuition, proving that under mild assumptions, p ~ X 0 \tilde{p}^{X_{0}} and p ∗ X {p}^{X}_{\ast} have the same support, and that 𝕂 𝕃 ( p ∗ X ∥ p ~ X 0 ) ≤ 𝕂 𝕃 ( p ∗ X T ∥ p ∗ X ∞ ) \mathbb{KL}({p}^{X}_{\ast}\,\|\,\tilde{p}^{X_{0}})\leq\mathbb{KL}(p^{X_{T}}_{\ast}\,\|\,p^{X_{\infty}}_{\ast}) . 14 14 
 14 
 
 
 
 Note that even though p ∗ X {p}^{X}_{\ast} and p ~ X 0 \tilde{p}^{X_{0}} are not full-dimensional densities, the KL divergence between them is meaningfully defined because they have the same support ( Section 3.4 ). 
Since p ~ X 0 \tilde{p}^{X_{0}} is the distribution of the model under the assumptions of no discretization error and having perfectly recovered the true score function (justified by the nonparametric regime assumption from Section 2.2 ), the fact that 𝕂 𝕃 ( p ∗ X T ∥ p ∗ X ∞ ) → 0 \mathbb{KL}(p^{X_{T}}_{\ast}\,\|\,p^{X_{\infty}}_{\ast})\rightarrow 0 as T → ∞ T\rightarrow\infty then implies that perfectly trained score-based diffusion models properly learn p ∗ X {p}^{X}_{\ast} .
 Pidstrigach (2022) also shows that if the score function is well approximated, in the sense that ‖ s θ ∗ X ​ ( x , t ) − ∇ x ​ log ​ p ∗ X t ​ ( x ) ‖ 2 \|s_{\theta^{\ast}}^{X}(x,t)-\nabla_{x}\log p^{X_{t}}_{\ast}(x)\|_{2} is upper-bounded (with the bound not depending on x x nor t t ), then p ^ θ ∗ X 0 \hat{p}_{\theta^{\ast}}^{X_{0}} has the same support as p ∗ X {p}^{X}_{\ast} , i.e. the model (assuming no discretization error) has the same support as the data.
Although this result does not guarantee that diffusion models recover their target distribution, it does ensure that they recover the correct manifold, even if there is some error in the learned score function. De Bortoli (2022) further refined these results, finding an upper bound for 𝕎 1 ​ ( p ∗ X , p θ ∗ X ) \mathbb{W}_{1}({p}^{X}_{\ast},{p}^{X}_{\theta^{\ast}}) under reasonable assumptions (recall that p θ X {p}^{X}_{\theta} corresponds to the approximate solution of Equation 52 at time T T ). This upper bound depends on T T , the error between the true and modelled score functions, and the step size of the discretization used to solve Equation 52 ; the upper bound goes to 0 0 as these quantities go to ∞ \infty , 0 0 , and 0 0 at appropriate rates, respectively. All of this entails that score-based diffusion models can learn distributions on unknown manifolds.

 
 
 
 
 
 
 
 
 Figure 6: (a) Informal illustration of why the score function explodes. Consider Y t = x Y_{t}=x for some fixed x x outside of ℳ \mathcal{M} as t t increases from 0 0 to T T . Since diffusion models learn manifolds, Y T Y_{T} must be in ℳ \mathcal{M} . Thus, when t t gets “infinitesimally” close to T T , the diffusion must push x x onto ℳ \mathcal{M} by moving it in the direction of the drift term in Equation 50 , i.e. μ ⁡ ( Y t , t ) ≔ β ⁡ ( T − t ) ​ ( Y t / 2 + ∇ y t ​ log ​ p ∗ X T − t ​ ( Y t ) ) \mu(Y_{t},t)\coloneqq\beta(T-t)(Y_{t}/2+\nabla_{y_{t}}\log p_{\ast}^{X_{T-t}}(Y_{t})) , but using only an “infinitesimally” small step size. Since x x is not “infinitesimally” close to ℳ \mathcal{M} , it needs to move a “non-infinitesimal” amount in an “infinitesimal” time step: the only way in which this can happen is if the norm of the drift, and thus of the score as well, explodes to infinity. (b) In practice, the norm of the score function diverges not only for fixed points ( ‖ s θ ∗ X ​ ( x , T − t ) ‖ 2 → ∞ \|s_{\theta^{\ast}}^{X}(x,T-t)\|_{2}\rightarrow\infty ), but also along generated trajectories ( ‖ s θ ∗ X ​ ( Y t , T − t ) ‖ 2 → ∞ \|s_{\theta^{\ast}}^{X}(Y_{t},T-t)\|_{2}\rightarrow\infty ). 
 
 
 Pidstrigach (2022) also argues that to properly learn the manifold the score ∇ x ​ log ​ p ∗ X T − t ​ ( x ) \nabla_{x}\log p^{X_{T-t}}_{\ast}(x) must explode to infinity at time T T , i.e. ‖ ∇ x ​ log ​ p ∗ X T − t ​ ( x ) ‖ 2 → ∞ \|\nabla_{x}\log p^{X_{T-t}}_{\ast}(x)\|_{2}\rightarrow\infty as t → T − t\rightarrow T^{-} for every x x outside of ℳ \mathcal{M} . This behaviour, which we illustrate in Figure 6 , is easy to understand intuitively: Equation 50 results in Y T ∼ p ∗ X Y_{T}\sim{p}^{X}_{\ast} , which must be in ℳ \mathcal{M} because, as discussed in the previous paragraph, diffusion models learn manifolds.
Now, fix x x outside of ℳ \mathcal{M} .
Since Y t Y_{t} is full-dimensional for every t T t T , it follows that x x is in its support, so that x x is a possible value for Y t Y_{t} . This means that when t t is “infinitesimally close” to T T , the norm of the drift in Equation 50 , i.e. β ⁡ ( T − t ) ​ ‖ x / 2 + ∇ x ​ log ​ p ∗ X T − t ​ ( x ) ‖ 2 \beta(T-t)\|x/2+\nabla_{x}\log p^{X_{T-t}}_{\ast}(x)\|_{2} , must be “infinitely large” to move x x to ℳ \mathcal{M} , since x x is outside of ℳ \mathcal{M} . In other words, the score must explode to infinity. It follows that if s θ ∗ X ​ ( x , T − t ) s_{\theta^{\ast}}^{X}(x,T-t) closely matches ∇ x ​ log ​ p ∗ X T − t ​ ( x ) \nabla_{x}\log p^{X_{T-t}}_{\ast}(x) , it must also diverge as t → T − t\rightarrow T^{-} .
 Lu et al. (2023) formalize this intuition, showing that under some regularity conditions, not only does the score function explode when d ∗ D d^{\ast} D , but also that it does not when d ∗ = D d^{\ast}=D . This phenomenon justifies an often used parameterization of score functions: rather than directly taking s θ X s_{\theta}^{X} as a neural network (which would be continuous on the compact interval [ 0 , T ] [0,T] and would thus achieve its maximum, making it impossible for the function to diverge and approximate the true score function), s θ X s_{\theta}^{X} is taken as s θ X ​ ( x , t ) = s ^ θ X ​ ( x , t ) / σ t s_{\theta}^{X}(x,t)=\hat{s}_{\theta}^{X}(x,t)/\sigma_{t} , where s ^ θ X \hat{s}_{\theta}^{X} is the neural network.
Empirically, it has been observed that diffusion models produce trajectories with exploding score functions ( Figure 6 ), i.e. that ‖ s θ ∗ X ​ ( Y t , T − t ) ‖ 2 → ∞ \|s^{X}_{\theta^{\ast}}(Y_{t},T-t)\|_{2}\rightarrow\infty as t → T − t\rightarrow T^{-} ( Kim et al., 2022 ) , which explains the common practice of not running Equation 52 until time T T , and instead stopping at time T − ε T-\varepsilon for some small ε 0 \varepsilon 0 ( Vahdat et al., 2021 ) .
The results of Pidstrigach (2022) and Lu et al. (2023) showing that the score is unbounded in the manifold setting thus strongly hint at why these trajectories have exploding scores. 15 15 
 15 
 
 
 
 Note that the result that ‖ s θ ∗ X ​ ( x , T − t ) ‖ 2 → ∞ \|s^{X}_{\theta^{\ast}}(x,T-t)\|_{2}\rightarrow\infty as t → T − t\rightarrow T^{-} for every x x outside of ℳ \mathcal{M} does not formally imply that ‖ s θ ∗ X ​ ( Y t , T − t ) ‖ 2 → ∞ \|s^{X}_{\theta^{\ast}}(Y_{t},T-t)\|_{2}\rightarrow\infty with probability 1 1 , even if Y t Y_{t} is outside of ℳ \mathcal{M} for t T t T , since Y t Y_{t} converges to Y T ∈ ℳ Y_{T}\in\mathcal{M} . Nevertheless the score function exploding at fixed points shows it is unbounded, which is necessary for the trajectories to blow up as well. 

 
 
 We also highlight that there is no contradiction between diffusion models being able to learn distributions on unknown manifolds and the fact that they can be interpreted as continuous normalizing flows (which cannot do so because of manifold overfitting, see Section 4.1 ); the former are trained through denoising score matching instead of maximum-likelihood. We note as well that the function v θ ∗ ​ ( x , t ) = − β ⁡ ( t ) 2 ​ ( x + s θ ∗ X ​ ( x , t ) ) v_{\theta^{\ast}}(x,t)=-\tfrac{\beta(t)}{2}(x+s_{\theta^{\ast}}^{X}(x,t)) which allows us to interpret diffusion models as continuous NFs through the ODE in Equation 30 is not Lipschitz in t t when the score function blows up to infinity at time 0 0 . In turn, there is no immediate guarantee that Equation 57 has a unique solution, so that diffusion models need not be properly defined as continuous NFs; this is of course consistent with the numerical instabilities which render their likelihoods unreliable ( Section 4.1.1 and Section 4.1.3 ).

 
 
 Overall, we believe this view of score-based diffusion models through the manifold setting is particularly illuminating, as it justifies their strong empirical performance (unlike many other popular models, diffusion models can properly learn manifold-supported distributions) and a popular parameterization of the score function, while also explaining its exploding behaviour at time 0 0 .

 
 
 Lastly, we highlight that the progressive noising of data employed by diffusion models ( Equation 49 , or a discretized version) has been adopted in contexts beyond denoising score matching: Xiao et al. (2022) combine it with adversarial training ( Section 4.2 and Section 5.2.1 ), and Tran et al. (2023) leverage it for manifold-aware maximum-likelihood training of DGMs.

 
 
 
 Here we briefly formalize some of the conclusions of Pidstrigach (2022) discussed above. The statement that p ~ X 0 \tilde{p}^{X_{0}} and p ∗ X {p}^{X}_{\ast} have the same support is formalized as ℙ ~ X 0 ≪ ℙ ∗ X \tilde{\mathbb{P}}^{X_{0}}\ll\mathbb{P}^{X}_{*} and ℙ ∗ X ≪ ℙ ~ X 0 \mathbb{P}^{X}_{*}\ll\tilde{\mathbb{P}}^{X_{0}} , where ℙ ~ X 0 \tilde{\mathbb{P}}^{X_{0}} is the probability measure corresponding to p ~ X 0 \tilde{p}^{X_{0}} . Note that this statement implies that ℙ ~ X 0 \tilde{\mathbb{P}}^{X_{0}} and ℙ ∗ X \mathbb{P}^{X}_{*} must have the same sets of measure 0 0 , and thus of measure 1 1 as well, and since the support of a distribution depends only on the sets to which it assigns probability 1 1 ( Equation 1 ), these two probability measures must have the same support. Similarly, the statement that p ^ θ ∗ X 0 \hat{p}^{X_{0}}_{\theta^{\ast}} and p ∗ X {p}^{X}_{\ast} have the same support is formalized as ℙ ^ θ ∗ X 0 ≪ ℙ ∗ X \hat{\mathbb{P}}^{X_{0}}_{\theta^{\ast}}\ll\mathbb{P}^{X}_{*} and ℙ ∗ X ≪ ℙ ^ θ ∗ X 0 \mathbb{P}^{X}_{*}\ll\hat{\mathbb{P}}^{X_{0}}_{\theta^{\ast}} , where ℙ ^ θ ∗ X 0 \hat{\mathbb{P}}^{X_{0}}_{\theta^{\ast}} is the probability measure corresponding to p ^ θ ∗ X 0 \hat{p}^{X_{0}}_{\theta^{\ast}} . 
 
 
 
 
 

#### Section 5.1.3 Conditional Flow Matching

 
 Continuous normalizing flows ( Section 4.1.3 ) as defined through Equation 30 were originally trained through maximum-likelihood. As discussed in Section 5.1.2 , diffusion models can be interpreted as providing an alternative training objective for continuous NFs.
Conditional flow matching ( Liu et al., 2023 ; Albergo Vanden-Eijnden, 2023 ; Lipman et al., 2023 , CFM;) provides additional alternatives, the main variant of which we cover here.

 
 
 Recall that diffusion models parameterize and learn a vector field ( s θ X s_{\theta}^{X} , approximating the score function) by attempting to regress against the true score,

 

 
 | 
 min ⁡ ∫ 0 T θ ⁡ w ⁡ ( t ) ​ 𝔼 X t ∼ p ∗ X t ​ [ ‖ s θ X ​ ( X t , t ) − ∇ x t ​ log ​ p ∗ X t ​ ( X t ) ‖ 2 2 ] ​ d ​ t , \min_{\theta}\int_{0}^{T}w(t)\mathbb{E}_{X_{t}\sim p_{\ast}^{X_{t}}}[\|s_{\theta}^{X}(X_{t},t)-\nabla_{x_{t}}\log p_{\ast}^{X_{t}}(X_{t})\|_{2}^{2}]{\textnormal{d}}t, | 
 | 
 (59) | 
 

 where we will assume w ⁡ ( t ) = 1 w(t)=1 to better highlight their similarities to CFM.
As discussed in Section 4.3 and Section 5.1.1 , directly optimizing Equation 59 cannot be done in practice because p ∗ X t p_{\ast}^{X_{t}} is unknown. However, by conditioning on X 0 X_{0} , the equivalent – yet tractable – objective in Equation 55 is obtained.

 
 
 Consider a “true” vector field v ∗ : 𝒳 × [ 0 , T ] → 𝒳 v_{\ast}:\mathcal{X}\times[0,T]\rightarrow\mathcal{X} in the sense that its corresponding forward ordinary differential equation ( Equation 30 ) maps samples from p ∗ X {p}^{X}_{\ast} at time t = 0 t=0 to samples from p Z {p}^{Z} at t = T t=T . Note that even when such a vector field exists, it is not unique.
Similarly to diffusion models, CFM learns a vector field v θ v_{\theta} by attempting to regress against v ∗ v_{\ast} , i.e.

 

 
 | 
 min ⁡ ∫ 0 T θ ⁡ 𝔼 X t ∼ p ∗ X t ​ [ ‖ v θ ​ ( X t , t ) − v ∗ ​ ( X t , t ) ‖ 2 2 ] ​ d ​ t , \min_{\theta}\int_{0}^{T}\mathbb{E}_{X_{t}\sim p_{\ast}^{X_{t}}}[\|v_{\theta}(X_{t},t)-v_{\ast}(X_{t},t)\|_{2}^{2}]{\textnormal{d}}t, | 
 | 
 (60) | 
 

 where p ∗ X t p_{\ast}^{X_{t}} now corresponds to the density of x t x_{t} if d ​ x t = v ∗ ​ ( x t , t ) ​ d ​ t {\textnormal{d}}x_{t}=v_{\ast}(x_{t},t){\textnormal{d}}t is initialized at x 0 = X 0 ∼ p ∗ X x_{0}=X_{0}\sim{p}^{X}_{\ast} .
Again, Equation 60 cannot be directly optimized because v ∗ v_{\ast} , and hence also p ∗ X t p_{\ast}^{X_{t}} , are unknown. Conditioning is once again the solution, except now conditioning is done on both X 0 X_{0} and X T X_{T} which allows for explicit construction of a target vector field that maps X 0 X_{0} to X T X_{T} . There are many such vector fields, but the simplest is the vector field ( X T − X 0 ) / T (X_{T}-X_{0})/T ( Tong et al., 2024 ) . Since this vector field is time-independent, sampling from the corresponding p ∗ X t p_{\ast}^{X_{t}} becomes easy as one can take X t = ( ( T − t ) ​ X 0 + t ​ X T ) / T X_{t}=((T-t)X_{0}+tX_{T})/T , where X 0 ∼ p ∗ X X_{0}\sim{p}^{X}_{\ast} and X T ∼ p Z X_{T}\sim{p}^{Z} are independently sampled. In summary, v θ v_{\theta} is trained through

 

 
 | 
 min ⁡ ∫ 0 T θ ⁡ 𝔼 X 0 ∼ p ∗ X , X T ∼ p Z ​ [ ‖ v θ ​ ( ( T − t ) ​ X 0 + t ​ X T T , t ) − X T − X 0 T ‖ 2 2 ] ​ d ​ t . \min_{\theta}\int_{0}^{T}\mathbb{E}_{X_{0}\sim{p}^{X}_{\ast},X_{T}\sim{p}^{Z}}\left[\left\|v_{\theta}\left(\dfrac{(T-t)X_{0}+tX_{T}}{T},t\right)-\dfrac{X_{T}-X_{0}}{T}\right\|_{2}^{2}\right]{\textnormal{d}}t. | 
 | 
 (61) | 
 

 Remarkably, when p ∗ X {p}^{X}_{\ast} and p Z {p}^{Z} are both full-dimensional, and under some other regularity conditions, the conditional optimization problem in Equation 61 turns out to be equivalent to the unconditional optimization problem in Equation 60 for a particular v ∗ v_{\ast} . Hence, under these conditions, the vector field v θ ∗ v_{\theta^{\ast}} optimized under Equation 61 can still be used for generation with the reverse ODE in Equation 32 .

 
 
 Conditional flow matching through the lens of the manifold hypothesis 

 
 We begin by pointing out that Lipman et al. (2023) add a small amount of noise to the data while applying CFM, which enforces full-dimensionality of the target density. However, if no noise is added, the known correctness guarantees for CFM break down when p ∗ X {p}^{X}_{\ast} is supported on a low-dimensional manifold. Kingma Gao (2023) proved that, surprisingly, the objectives in Equation 55 for diffusion models and Equation 61 for CFM are equivalent given appropriate hyperparameter choices. Since diffusion models are manifold-aware, at first glance this result might seem to imply that CFM must also learn distributions on unknown manifolds. However, this does not follow from the result of Kingma Gao (2023) : the manifold-awareness of diffusion models is guaranteed when the backward stochastic differential equation ( Equation 52 ) is used to sample from the model, whereas CFM always uses an ODE ( Equation 32 ). 16 16 
 16 
 
 
 
 In turn, the result of Kingma Gao (2023) does ensure that if the vector field learned through CFM is converted back to a score function and used alongside the backward SDE for sampling, then the resulting model will be manifold-aware. However, this is not how CFM is sampled from in practice. Note also that any result ensuring that Equation 58 can indeed produce samples from p ∗ X {p}^{X}_{\ast} in the manifold setting would in turn guarantee the manifold-awareness of CFM. To the best of our knowledge, there is currently no formal analysis of CFM under the manifold hypothesis analogous to that of Pidstrigach (2022) or De Bortoli (2022) for diffusion models. 17 17 
 17 
 
 
 
 The closest analysis we are aware of is due to Gao Zhu (2024) , who provide a Wasserstein distance bound for the ODE sampler. However, this bound requires assumptions which are incompatible with the manifold setting, and thus does not ensure manifold-awareness. This stands in contrast with the analogous bound for the SDE sampler by De Bortoli (2022) , which does guarantee manifold-awareness. Nonetheless, due to its similarities with diffusion models, we conjecture that CFM is indeed manifold-aware, which would be consistent with its strong empirical performance. We do however point out that if CFM successfully learns its target distribution in the manifold setting, then it must experience the numerical instabilities of likelihood evaluation that we established in Section 4.1.1 , even if these instabilities do not manifest themselves during training since likelihoods do not appear in Equation 61 . Finally, we also highlight the work of Kapusniak et al. (2024) , who point out that X t X_{t} linearly interpolating between data X 0 X_{0} and noise X T X_{T} has some undesirable properties; they thus propose a modification of CFM with the goal of ensuring that the interpolations remain close to geodesics in ℳ \mathcal{M} , which they show leads to empirical improvements.

 
 
 
 

#### Section 5.1.4 Noisy Normalizing Flows

 
 Several models have been proposed based on the idea of adding noise to the data, training a normalizing flow ( Section 4.1.3 ) on this noisy data, and then having a “deflation” procedure whose goal is to sample from p ∗ X {p}^{X}_{\ast} rather than its learned noisy version. One such model is the denoising NF of Horvat Pfister (2021) , which itself is heavily based on the theoretical results of Horvat Pfister (2023) . 18 18 
 18 
 
 
 
 Horvat Pfister (2023) was released first on arXiv despite having a later publication date than Horvat Pfister (2021) . 
Motivated by the inherent struggles of flow-based architectures on manifold-supported data (described in Section 4.1 and Section 4.1.3 ), denoising NFs attempt to model an “inflated” version of the data distribution with added noise, and then provide conditions under which this noise can be “deflated” to recover the true on-manifold density.
Similar to denoising score matching ( Section 5.1.1 ), the inflation step of Horvat Pfister (2021) consists of simply adding full dimensional Gaussian noise to the data, and building a density model for p ∗ X σ ≔ p ∗ X ⊛ 𝒩 ⁡ ( ⋅ , 0 , σ 2 ​ I D ) p^{X_{\sigma}}_{\ast}\coloneqq{p}^{X}_{\ast}\circledast\mathcal{N}(\,\cdot\,;0,\sigma^{2}I_{D}) , where σ 0 \sigma 0 is a hyperparameter.
On the other hand, Horvat Pfister (2023) discuss a more theoretically-grounded inflation step, where ( D − d ∗ ) (D-d^{\ast}) -dimensional Gaussian noise is added to X ∼ p ∗ X X\sim{p}^{X}_{\ast} along the
 normal space of ℳ \mathcal{M} at X X . 19 19 
 19 
 
 
 
 “Normal” in the geometric sense, not the Gaussian sense. 
While determining the normal space is only possible for known manifolds, the authors argue that when d ∗ d^{\ast} is much smaller than D D , using full-dimensional Gaussian noise is a good approximation of ( D − d ∗ ) (D-d^{\ast}) -dimensional noise in the normal space.
We will shortly review the deflation step.

 
 
 To model p ∗ X σ p^{X_{\sigma}}_{\ast} , Horvat Pfister (2021) use a full-dimensional DGM p θ X σ p^{X_{\sigma}}_{\theta} .
Their specific choice is a D D -dimensional NF, albeit with a non-standard learnable latent distribution p θ Z {p}^{Z}_{\theta} . For some hyperparameter d d meant to approximate d ∗ d^{\ast} , the first d d latent coordinates are themselves modelled by a d d -dimensional NF, and the remaining D − d D-d latent coordinates use a zero-mean Gaussian with covariance σ 2 ​ I D − d \sigma^{2}I_{D-d} , matching the noise level of the inflation step above.
The overall likelihood for their full-dimensional model is thus

 

 
 | 
 p θ X σ ​ ( x ) = p θ Z 1 ​ ( z 1 ) ​ p Z 2 ​ ( z 2 ) ​ | det ∇ x f θ ​ ( x ) | , p^{X_{\sigma}}_{\theta}(x)=p_{\theta}^{Z_{1}}(z_{1})p^{Z_{2}}(z_{2})\left|\det\nabla_{x}f_{\theta}(x)\right|, | 
 | 
 (62) | 
 

 where: f θ = g θ − 1 f_{\theta}=g_{\theta}^{-1} with g θ g_{\theta} being the D D -dimensional NF, z = ( z 1 , z 2 ) = f θ ​ ( x ) z=(z_{1},z_{2})=f_{\theta}(x) with z 1 ∈ ℝ d z_{1}\in\mathbb{R}^{d} and z 2 ∈ ℝ D − d z_{2}\in\mathbb{R}^{D-d} , and the latent density p θ Z {p}^{Z}_{\theta} is given by p θ Z ​ ( z ) = p θ Z 1 ​ ( z 1 ) ​ p θ Z 2 ​ ( z 2 ) {p}^{Z}_{\theta}(z)=p_{\theta}^{Z_{1}}(z_{1})p_{\theta}^{Z_{2}}(z_{2}) , where p θ Z 1 p_{\theta}^{Z_{1}} is the density corresponding to the d d -dimensional NF model, and p Z 2 ​ ( ⋅ ) = 𝒩 ⁡ ( ⋅ , 0 , σ 2 ​ I D − d ) p^{Z_{2}}(\cdot)=\mathcal{N}(\,\cdot\,;0,\sigma^{2}I_{D-d}) with no trainable parameters.
The idea is that the first d d latent coordinates are noise-insensitive – i.e. they are meant to denoise the data – while the remaining D − d D-d coordinates are noise-sensitive. The model p θ X σ p^{X_{\sigma}}_{\theta} defined above has no explicit manifold-awareness.
 Horvat Pfister (2021) thus add an injective flow construction (reviewed further in Section 5.3.3 ) into the mix, defining the actual (denoised) manifold-supported generative process p θ X {p}^{X}_{\theta} by first sampling Z 1 ∼ p θ Z 1 Z_{1}\sim p^{Z_{1}}_{\theta} and then setting X = g θ ​ ( ( Z 1 , 0 ) ) X=g_{\theta}((Z_{1},0)) , where 0 ∈ ℝ D − d 0\in\mathbb{R}^{D-d} and ( ⋅ , ⋅ ) (\cdot,\cdot) denotes concatenation so that ( Z 1 , 0 ) ∈ ℝ D (Z_{1},0)\in\mathbb{R}^{D} . Horvat Pfister (2021) refer to using the manifold-supported p θ X {p}^{X}_{\theta} rather than the full-dimensional p θ X σ p^{X_{\sigma}}_{\theta} as “deflating” p θ X σ p^{X_{\sigma}}_{\theta} .

 
 
 Horvat Pfister (2021) also follow injective flows ( Equation 97 ) in adding a reconstruction term as a regularizer to encourage manifold-awareness of the overall objective, which can be written as

 

 
 | 
 max θ ⁡ 𝔼 X σ ∼ p ∗ X σ ​ [ log ⁡ p θ X σ ​ ( X σ ) ] − β ​ 𝔼 X σ ∼ p ∗ X σ ​ [ ‖ X σ − g θ ​ ( ( f θ ​ ( X σ ) 1 , 0 ) ) ‖ 2 2 ] , \max_{\theta}\Exp_{X_{\sigma}\sim p_{\ast}^{X_{\sigma}}}\left[\log p_{\theta}^{X_{\sigma}}(X_{\sigma})\right]-\beta\Exp_{X_{\sigma}\sim p_{\ast}^{X_{\sigma}}}\left[\|X_{\sigma}-g_{\theta}\left(\left(f_{\theta}(X_{\sigma})_{1},0\right)\right)\|_{2}^{2}\right], | 
 | 
 (63) | 
 

 where β 0 \beta 0 is a hyperparameter, f θ ​ ( X σ ) 1 ∈ ℝ d f_{\theta}(X_{\sigma})_{1}\in\mathbb{R}^{d} corresponds to the first d d coordinates of f θ ​ ( X σ ) f_{\theta}(X_{\sigma}) , and once again 0 ∈ ℝ D − d 0\in\mathbb{R}^{D-d} . The regularizer encourages g θ g_{\theta} to not need Z 2 = f θ ​ ( X σ ) 2 Z_{2}=f_{\theta}(X_{\sigma})_{2} (i.e. the last D − d D-d coordinates of f θ ​ ( X σ ) f_{\theta}(X_{\sigma}) ) to perfectly reconstruct X σ X_{\sigma} : this can be intuitively understood as providing an inductive bias which promotes capturing the added noise only through Z 2 Z_{2} , thus justifying the use of p θ X {p}^{X}_{\theta} instead of p θ X σ p^{X_{\sigma}}_{\theta} .

 
 
 We now discuss the “deflation” aspect of this approach.
 Horvat Pfister (2023) prove that a trained denoising normalizing flow p θ ∗ X {p}^{X}_{\theta^{\ast}} recovers p ∗ X {p}^{X}_{\ast} under the following conditions: ( i ) (i) the noise added to X ∼ p ∗ X X\sim{p}^{X}_{\ast} is only added along the ( D − d ∗ ) (D-d^{\ast}) -dimensional normal space to the true manifold ℳ \mathcal{M} at X X , ( i ​ i ) (ii) the noise parameter σ \sigma is sufficiently small, and ( i ​ i ​ i ) (iii) the manifold ℳ \mathcal{M} is “sufficiently smooth and disentangled” (this condition is properly formalized by Horvat Pfister (2023) ).

 
 
 Condition ( i ) (i) is not satisfied by the denoising normalizing flow, although Horvat Pfister (2021) argue that when D D is much larger than d ∗ d^{\ast} , 𝒩 ⁡ ( ⋅ , 0 , σ 2 ​ I D ) \mathcal{N}(\,\cdot\,;0,\sigma^{2}I_{D}) is a good approximation for the ( D − d ∗ ) − (D-d^{\ast})- dimensional Gaussian noise in the normal space; it is also worth reiterating that it would not be possible to add noise in the normal space without knowing the true manifold exactly in the first place.
Condition ( i ​ i ​ i ) (iii) is also worth discussing: it is entirely unclear if natural data observed “in the wild” is sufficiently smooth and disentangled,
so we may not have any guarantee of retrieving the true manifold-supported data distribution even in the nonparametric regime. In summary, despite denoising NFs being an elegant approach, their manifold-awareness can only be ensured under potentially strong and hard-to-verify assumptions.

 
 
 Postels et al. (2022) proposed a very similar approach to denoising NFs, except the latent density used to define p θ X σ p_{\theta}^{X_{\sigma}} is fixed as a standard Gaussian, no regularizer is used during training, and the “deflation” step is done by first sampling X σ ∼ p θ ∗ X σ X_{\sigma}\sim p_{\theta^{\ast}}^{X_{\sigma}} and then solving

 

 
 | 
 X = arg ​ max x ​ log ​ p θ ∗ X σ ​ ( x ) − β ​ ‖ x − X σ ‖ 2 2 . X=\argmax_{x}\,\log p_{\theta^{\ast}}^{X_{\sigma}}(x)-\beta\|x-X_{\sigma}\|_{2}^{2}. | 
 | 
 (64) | 
 

 Although Postels et al. (2022) do not rigorously justify their method, the intuition behind this “deflation” step is sensible; since p θ ∗ X σ p_{\theta^{\ast}}^{X_{\sigma}} should spike around ℳ \mathcal{M} when σ \sigma is small enough, Equation 64 can be informally interpreted as pulling X σ X_{\sigma} closer to ℳ \mathcal{M} .

 
 
 Finally, Kim et al. (2020) use a continuum of noise levels in a manner reminiscent of diffusion models ( Section 5.1.2 ). More specifically, they first construct a conditional normalizing flow g θ : 𝒵 × [ 0 , σ max ] → 𝒳 g_{\theta}:\mathcal{Z}\times[0,\sigma_{\text{max}}]\rightarrow\mathcal{X} , where σ max 0 \sigma_{\text{max}} 0 is a hyperparameter. For every σ ∈ [ 0 , σ max ] \sigma\in[0,\sigma_{\text{max}}] , the flow g θ ​ ( ⋅ , σ ) g_{\theta}(\cdot,\sigma) must be a diffeomorphism, whose inverse we denote as f θ ​ ( ⋅ , σ ) f_{\theta}(\cdot,\sigma) . They then condition the NF on the amount of added noise and train through (conditional) maximum-likelihood:

 

 
 | 
 max ⁡ ∫ 0 σ max θ ⁡ 𝔼 X σ ∼ p ∗ X σ ​ [ log ⁡ p Z ​ ( f θ ​ ( X σ , σ ) ) + log ⁡ | det ∇ x σ f θ ​ ( X σ , σ ) | ] ​ d ​ σ . \max_{\theta}\int_{0}^{\sigma_{\text{max}}}\mathbb{E}_{X_{\sigma}\sim p_{\ast}^{X_{\sigma}}}\left[\log{p}^{Z}\left(f_{\theta}(X_{\sigma},\sigma)\right)+\log\left|\det\nabla_{x_{\sigma}}f_{\theta}(X_{\sigma},\sigma)\right|\right]{\textnormal{d}}\sigma. | 
 | 
 (65) | 
 

 Since the conditional NF is trained so that X σ = g θ ∗ ​ ( Z , σ ) X_{\sigma}=g_{\theta^{\ast}}(Z,\sigma) , where Z ∼ p Z Z\sim{p}^{Z} , is distributed according to p ∗ X σ p_{\ast}^{X_{\sigma}} , the “deflation” step here simply consists of using σ = 0 \sigma=0 when sampling from the model, i.e. X = g θ ∗ ​ ( Z , 0 ) X=g_{\theta^{\ast}}(Z,0) .

 
 
 Finally, we highlight that although the three methods presented above entail fitting full-dimensional likelihood-based models to full-dimensional target densities, the target densities consist of the convolution of manifold-supported densities with small amounts of noise. As a result, these methods are still exposed to the numerical pathologies described in Section 4.1.1 .

 
 
 

#### Section 5.1.5 Spread Divergences

 
 As previously mentioned, attempting to minimize KL divergence through maximum-likelihood in the manifold setting can result in manifold overfitting ( Section 4.1 ), and while adding a small amount of noise to the data can circumvent the problem in theory, it might not in practice. Zhang et al. (2020a) propose to not only add noise to the data, but to add the same amount of noise to the model as well. They formalize this idea by introducing spread divergences. Here we will focus exclusively on the spread KL divergence and Gaussian noise, but highlight that the same ideas can be applied to other divergences and (potentially learnable) noise distributions. Formally, for a fixed σ 0 \sigma 0 , the spread KL divergence 𝕂 𝕃 σ ( p ∥ q ) \mathbb{KL}_{\sigma}(p\,\|\,q) between probability densities p p and q q is given by

 

 
 | 
 𝕂 𝕃 σ ( p ∥ q ) ≔ 𝕂 𝕃 ( p σ ∥ q σ ) , \mathbb{KL}_{\sigma}(p\,\|\,q)\coloneqq\mathbb{KL}\left(p_{\sigma}\,\|\,q_{\sigma}\right), | 
 | 
 (66) | 
 

 where p σ ≔ p ⊛ 𝒩 ⁡ ( ⋅ , 0 , σ 2 ​ I D ) p_{\sigma}\coloneqq p\circledast\mathcal{N}(\ \cdot\ ;0,\sigma^{2}I_{D}) and q σ ≔ q ⊛ 𝒩 ⁡ ( ⋅ , 0 , σ 2 ​ I D ) q_{\sigma}\coloneqq q\circledast\mathcal{N}(\ \cdot\ ;0,\sigma^{2}I_{D}) , i.e. the spread KL divergence is the KL divergence between noisy versions of p p and q q . The spread KL divergence has two important properties: 𝕂 𝕃 σ ( p ∥ q ) ≥ 0 \mathbb{KL}_{\sigma}(p\,\|\,q)\geq 0 , with equality if and only if p = q p=q ; and it is always meaningfully defined, since the noisy versions of the original distributions are always full-dimensional ( Section 3.4 ). Together, these properties imply that the spread KL divergence provides a mathematically sensible objective to train generative models under the manifold setting.

 
 
 While the noisy model p θ X ⊛ 𝒩 ⁡ ( ⋅ , 0 , σ 2 ​ I D ) {p}^{X}_{\theta}\circledast\mathcal{N}(\ \cdot\ ;0,\sigma^{2}I_{D}) always has a full-dimensional density, this density cannot in general be evaluated, and thus minimizing 𝕂 𝕃 σ ( p ∗ X ∥ p θ X ) \mathbb{KL}_{\sigma}({p}^{X}_{\ast}\,\|\,{p}^{X}_{\theta}) is not immediately trivial.
 Zhang et al. (2020a) propose a very similar model to a variational autoencoder ( Section 4.1.2 ), called the δ \delta -VAE, where the conditional distribution of X X given Z Z is now a point mass at g θ ​ ( Z ) g_{\theta}(Z) , instead of a Gaussian as in Equation 22 . In other words, this model is like a Gaussian VAE, except no additional noise is added to g θ ​ ( Z ) g_{\theta}(Z) , i.e. all the noise in this model comes from the low-dimensional Z Z .
Note that, unlike VAEs, this model is not full-dimensional. Much like VAEs are trained to minimize an upper bound of 𝕂 𝕃 ( p ∗ X ∥ p θ X ) \mathbb{KL}({p}^{X}_{\ast}\,\|\,{p}^{X}_{\theta}) (or equivalently, maximizing the lower bound to the log-likelihood from Equation 24 ), δ \delta -VAEs minimize an upper bound of 𝕂 𝕃 σ ( p ∗ X ∥ p θ X ) \mathbb{KL}_{\sigma}({p}^{X}_{\ast}\,\|\,{p}^{X}_{\theta}) , which as we will see ends up amounting to a simple modification to standard VAE training. Importantly, even if the supports of p θ X {p}^{X}_{\theta} and p ∗ X {p}^{X}_{\ast} do not overlap, their spread KL divergence is meaningfully defined and thus provides a valid objective that enables δ \delta -VAEs to properly learn distributions on unknown manifolds. More specifically, δ \delta -VAEs use a variational posterior density q ϕ Z | X σ q^{Z|X_{\sigma}}_{\phi} and are trained through

 

 
 | 
 max θ , ϕ 𝔼 X σ ∼ p ∗ X σ [ 𝔼 Z ∼ q Z | X σ ϕ ( ⋅ | X σ ) [ log 𝒩 ( X σ ; g θ ( Z ) , σ 2 I D ) ] − 𝕂 𝕃 ( q ϕ Z | X σ ( ⋅ | X σ ) ∥ p Z ) ] , \max_{\theta,\phi}\mathbb{E}_{X_{\sigma}\sim p^{X_{\sigma}}_{\ast}}\left[\mathbb{E}_{Z\sim q^{Z|X_{\sigma}}_{\phi}(\cdot|X_{\sigma})}[\log\mathcal{N}(X_{\sigma};g_{\theta}(Z),\sigma^{2}I_{D})]-\mathbb{KL}\left(q^{Z|X_{\sigma}}_{\phi}(\cdot|X_{\sigma})\,\Big\|\,p_{Z}\right)\right], | 
 | 
 (67) | 
 

 where p ∗ X σ ≔ p ∗ X ⊛ 𝒩 ⁡ ( ⋅ , 0 , σ 2 ​ I D ) p^{X_{\sigma}}_{\ast}\coloneqq{p}^{X}_{\ast}\circledast\mathcal{N}(\ \cdot\ ;0,\sigma^{2}I_{D}) . Note that this objective is equivalent to that of VAEs from Equation 24 , except Gaussian noise is added to the data, and the covariance of the decoder is not learnable, but rather made to match the covariance of the noise that was added to the data.

 
 
 A point about δ \delta -VAEs warrants discussion. A common “cheat” when sampling from a trained Gaussian VAE model is to sample Z ∼ p Z Z\sim{p}^{Z} , and then output the decoder mean g θ ∗ ​ ( Z ) g_{\theta^{\ast}}(Z) , rather than outputting a sample from a Gaussian centered at this point (i.e. the noise from the stochastic decoder is ignored). The use of the spread KL divergence elegantly justifies this common practice, since the noise (corresponding to the 𝒩 ⁡ ( X σ , g θ ​ ( Z ) , σ 2 ​ I D ) \mathcal{N}(X_{\sigma};g_{\theta}(Z),\sigma^{2}I_{D}) term in Equation 67 ) here is added due to the spread KL divergence, rather than being part of the model itself. Zhang et al. (2020a) did not make this observation when introducing δ \delta -VAEs, and to the best of our knowledge, we are the first to point this out.

 
 
 Finally, Zhang et al. (2023) aim to extend the use of the spread KL divergence beyond VAEs, and propose a procedure to use it within the context of normalizing flows ( Section 4.1.3 ). Since the spread KL divergence remains intractable, they introduce another bound as the training objective. Unlike other variational bounds such as the ELBO ( Equation 24 ) or Equation 67 , the bound of Zhang et al. (2023) does not become tight at optimality, so it lacks the theoretical guarantees of δ \delta -VAEs which ensure manifold-awareness.

 
 
 
 

### Section 5.2 Manifold-Awareness through Support-Agnostic Optimization Objectives

 
 In Section 5.1 we showed various DGMs which learn manifolds by first adding noise to p ∗ X {p}^{X}_{\ast} , and then using a full-dimensional objective like score matching or KL minimization (i.e. maximum-likelihood). An alternative to these approaches is using an objective that is agnostic to the supports of the underlying distributions being compared. In particular, divergences between probability distributions which metrize weak convergence, such as Wasserstein distances ( Section 3.5 ) or maximum mean discrepancy ( Section 3.6 ), provide such support-agnostic objectives. We now cover DGMs which are trained through these objectives.

 
 

#### Section 5.2.1 Wasserstein Generative Adversarial Networks

 
 Arjovsky et al. (2017) proposed to minimize Wasserstein distance ( Section 3.5 ) between p ∗ X {p}^{X}_{\ast} and p θ X {p}^{X}_{\theta} as a way to address the issues arising from attempting to train generative adversarial networks using various f f -divergences outlined in Section 4.2 , resulting in strong empirical performance ( Karras et al., 2018 ; Karras et al., 2019 ; Karras et al., 2020 ) . In the GAN context considered by Arjovsky et al. (2017) , where p θ X {p}^{X}_{\theta} is again given by the distribution of X = g θ ​ ( Z ) X=g_{\theta}(Z) , where Z ∼ p Z Z\sim{p}^{Z} (once more, formally, p θ X {p}^{X}_{\theta} is given by the pushforward of p Z {p}^{Z} through g θ g_{\theta} ) and p Z {p}^{Z} is fixed (e.g. standard Gaussian), this means changing the GAN objective from Equation 41 to

 

 
 | 
 min θ ⁡ max ϕ ​ 𝔼 X ∼ p ∗ X ​ [ h ϕ ​ ( X ) ] − 𝔼 Z ∼ p Z ⁡ [ h ϕ ​ ( g θ ​ ( Z ) ) ] , \min_{\theta}\max_{\phi}\Exp_{X\sim{p}^{X}_{\ast}}\left[h_{\phi}(X)\right]-\Exp_{Z\sim{p}^{Z}}\left[h_{\phi}(g_{\theta}(Z))\right], | 
 | 
 (68) | 
 

 where the neural network h ϕ : 𝒳 → ℝ h_{\phi}:\mathcal{X}\rightarrow\mathbb{R} is no longer a classifier and is now constrained to be Lipschitz. If, as with standard GANs, the network h ϕ h_{\phi} is assumed to be arbitrarily flexible, the objective reduces to minimizing the 𝕎 1 \mathbb{W}_{1} distance between the true distribution and the model ( Equation 17 ),

 

 
 | 
 min θ ⁡ 𝕎 1 ​ ( p ∗ X , p θ X ) , \min_{\theta}\mathbb{W}_{1}({p}^{X}_{\ast},{p}^{X}_{\theta}), | 
 | 
 (69) | 
 

 which as mentioned in Section 3.5 , provides a support-agnostic optimization objective for training p θ X {p}^{X}_{\theta} .
 Arjovsky et al. (2017) enforce the Lipschitz constraint on h ϕ h_{\phi} through weight clipping, which was later improved upon by Gulrajani et al. (2017a) and by Miyato et al. (2018) . The former use a regularizer encouraging ‖ ∇ x h ϕ ​ ( x ) ‖ 2 \|\nabla_{x}h_{\phi}(x)\|_{2} to be close to 1 1 , while the latter regularizes the spectral norm of the weight parameters of h ϕ h_{\phi} ; both obtain much better empirical performance than weight clipping.

 
 
 

#### Section 5.2.2 Wasserstein Autoencoders

 
 As described in Section 5.2.1 for Wasserstein generative adversarial networks, Arjovsky et al. (2017) leveraged the dual formulation of 𝕎 1 \mathbb{W}_{1} ( Equation 17 ). Tolstikhin et al. (2018) proposed Wasserstein autoencoders (WAEs), which instead leverage the definition of 𝕎 c \mathbb{W}^{c} ( Equation 16 ) – which remains a support-agnostic objective for training DGMs. In particular, they proved that when p θ X {p}^{X}_{\theta} is given by the distribution of X = g θ ​ ( Z ) X=g_{\theta}(Z) where Z ∼ p Z Z\sim{p}^{Z} (once again, p θ X {p}^{X}_{\theta} formally corresponds to the pushforward of p Z {p}^{Z} through g θ g_{\theta} ), the optimal transport cost can be written as

 

 
 | 
 𝕎 c ( p ∗ X , p θ X ) = inf q Z | X ∈ 𝒬 ⁡ ( p ∗ X , p Z ) 𝔼 X ∼ p ∗ X [ 𝔼 Z ∼ q Z | X ( ⋅ | X ) [ c ( X , g θ ( Z ) ) ] ] , \mathbb{W}^{c}({p}^{X}_{\ast},{p}^{X}_{\theta})=\inf_{q^{Z|X}\in\mathcal{Q}({p}^{X}_{\ast},{p}^{Z})}\Exp_{X\sim{p}^{X}_{\ast}}\left[\Exp_{Z\sim q^{Z|X}(\cdot|X)}\left[c(X,g_{\theta}(Z))\right]\right], | 
 | 
 (70) | 
 

 where 𝒬 ⁡ ( p ∗ X , p Z ) \mathcal{Q}({p}^{X}_{\ast},{p}^{Z}) is the set of conditional (on X X ) densities on 𝒵 \mathcal{Z} whose marginal matches p Z {p}^{Z} , i.e. 𝒬 ( p ∗ X , p Z ) ≔ { q Z | X : 𝒳 → Δ ( 𝒵 ) ∣ q Z = p Z } \mathcal{Q}({p}^{X}_{\ast},{p}^{Z})\coloneqq\{q^{Z|X}:\mathcal{X}\rightarrow\Delta(\mathcal{Z})\mid q^{Z}={p}^{Z}\} , where q Z ≔ 𝔼 X ∼ p ∗ X [ q Z | X ( ⋅ | X ) ] q^{Z}\coloneqq\Exp_{X\sim{p}^{X}_{\ast}}[q^{Z|X}(\cdot|X)] . Tolstikhin et al. (2018) propose two losses to train WAEs, both inspired by this property: analoguously to variational autoencoders, a conditional distribution q ϕ Z | X q^{Z|X}_{\phi} is parameterized (e.g. Equation 26 ), and the first loss is given by

 

 
 | 
 min θ , ϕ max ϕ ′ 𝔼 X ∼ p ∗ X [ 𝔼 Z ∼ q Z | X ϕ ( ⋅ | X ) [ c ( X , g θ ( Z ) ) ] ] + β ( 𝔼 Z ∼ q ϕ Z [ log h ϕ ′ ( Z ) ] + 𝔼 Z ∼ p Z [ log ( 1 − h ϕ ′ ( Z ) ) ] ) , \min_{\theta,\phi}\max_{\phi^{\prime}}\mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[\mathbb{E}_{Z\sim q^{Z|X}_{\phi}(\cdot|X)}[c(X,g_{\theta}(Z))]\right]+\beta\left(\mathbb{E}_{Z\sim q^{Z}_{\phi}}\left[\log h_{\phi^{\prime}}(Z)\right]+\mathbb{E}_{Z\sim{p}^{Z}}\left[\log\left(1-h_{\phi^{\prime}}(Z)\right)\right]\right), | 
 | 
 (71) | 
 

 where h ϕ ′ : 𝒵 → ( 0 , 1 ) h_{\phi^{\prime}}:\mathcal{Z}\rightarrow(0,1) , and β 0 \beta 0 is a hyperparameter. The first term in the above objective can be understood as minimizing the cost in Equation 70 with no regard for the constraint that q ϕ Z | X ∈ 𝒬 ⁡ ( p ∗ X , p Z ) q^{Z|X}_{\phi}\in\mathcal{Q}({p}^{X}_{\ast},{p}^{Z}) , whereas the second term encourages this constraint to be satisfied through an adversarial loss ( Equation 41 ) that aims to minimize 𝕁 𝕊 ( q ϕ Z ∥ p Z ) \mathbb{JS}(q^{Z}_{\phi}\,\|\,{p}^{Z}) ; thus, Equation 71 does indeed aim to minimize the optimal transport cost in Equation 70 . Note that even though q ϕ Z q^{Z}_{\phi} cannot be evaluated, it is trivial to obtain a sample Z Z from it by first sampling X ∼ p ∗ X X\sim{p}^{X}_{\ast} , and then sampling Z | X ∼ q ϕ Z | X ( ⋅ | X ) Z|X\sim q^{Z|X}_{\phi}(\cdot|X) , so that optimizing Equation 71 is tractable.

 
 
 We find it relevant to highlight some differences between WAEs and other DGMs. ( i ) (i) Unlike GANs ( Section 4.2 ) and Wasserstein GANs, WAEs use an adversarial loss on the latent space 𝒵 \mathcal{Z} rather than on ambient space 𝒳 \mathcal{X} . This helps stabilize WAEs, as adversarial losses on ambient space can be notoriously unstable. Nonetheless, properly trained Wasserstein GANs tend to empirically outperform WAEs. ( i ​ i ) (ii) WAEs are also very similar to variational autoencoders ( Section 4.1.2 ) – e.g. if c ⁡ ( x , y ) = ‖ x − y ‖ 2 2 c(x,y)=\|x-y\|_{2}^{2} , the WAE and Gaussian VAE losses become extremely alike – but the KL term in the VAE loss ( Equation 24 ) encourages q ϕ Z | X ( ⋅ | X ) q^{Z|X}_{\phi}(\cdot|X) to match p Z {p}^{Z} for every X X sampled from p ∗ X {p}^{X}_{\ast} , whereas WAEs only encourage this to happen on average, i.e. 𝔼 X ∼ p ∗ X [ q ϕ Z | X ( ⋅ | X ) ] = q ϕ Z = p Z \mathbb{E}_{X\sim{p}^{X}_{\ast}}[q_{\phi}^{Z|X}(\cdot|X)]=\ q^{Z}_{\phi}={p}^{Z} .

 
 
 The second WAE loss is completely analogous, except it uses maximum mean discrepancy ( Section 3.6 ) instead of an adversarial loss to encourage satisfying the constraint that q ϕ Z = p Z q^{Z}_{\phi}={p}^{Z} ,

 

 
 | 
 min θ , ϕ 𝔼 X ∼ p ∗ X [ 𝔼 Z ∼ q Z | X ϕ ( ⋅ | X ) [ c ( X , g θ ( Z ) ) ] ] + β 𝕄 𝕄 𝔻 k 2 ( q ϕ Z , p Z ) \min_{\theta,\phi}\mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[\mathbb{E}_{Z\sim q^{Z|X}_{\phi}(\cdot|X)}[c(X,g_{\theta}(Z))]\right]+\beta\mathbb{MMD}^{2}_{k}\left(q^{Z}_{\phi},{p}^{Z}\right) | 
 | 
 (72) | 
 

 for some kernel k : 𝒵 × 𝒵 → ℝ k:\mathcal{Z}\times\mathcal{Z}\rightarrow\mathbb{R} , which results in an objective with no adversarial component. The choice of which objective to use on latent space to enforce q ϕ Z = p Z q^{Z}_{\phi}={p}^{Z} has also been expanded in follow-up work, for example Kolouri et al. (2018) use the sliced Wasserstein distance, and Patrini et al. (2020) use relaxed (Sinkhorn) optimal transport.

 
 
 Finally, we point out that Tolstikhin et al. (2018) found that using arbitrarily distributions q ϕ Z | X q^{Z|X}_{\phi} was not key for good empirical performance, and they thus restrict 𝒬 ⁡ ( p ∗ X , p θ X ) \mathcal{Q}({p}^{X}_{\ast},{p}^{X}_{\theta}) to only contain point masses, i.e. q ϕ Z | X ( ⋅ | x ) q_{\phi}^{Z|X}(\cdot|x) is given by a point mass at f ϕ ​ ( x ) f_{\phi}(x) . This choice, which amounts to using deterministic rather than stochastic encoders, reduces the first term in Equation 71 and Equation 72 to a reconstruction error (as measured by c c ), 𝔼 X ∼ p ∗ X ​ [ c ⁡ ( X , g θ ​ ( f ϕ ​ ( X ) ) ) ] \mathbb{E}_{X\sim{p}^{X}_{\ast}}[c(X,g_{\theta}(f_{\phi}(X)))] .

 
 
 
 Here we simply highlight that since the distributions involved need not necessarily admit Lebesgue densities, Equation 70 must be formalized by using measures instead of densities: 
 

 
 
 𝕎 c ( ℙ ∗ X , ℙ θ X ) = inf ℚ Z | X ∈ 𝒬 ⁡ ( ℙ ∗ X , ℙ Z ) 𝔼 X ∼ ℙ ∗ X [ 𝔼 Z ∼ ℚ Z | X ( ⋅ | X ) [ c ( X , g θ ( Z ) ) ] ] , \mathbb{W}^{c}(\mathbb{P}^{X}_{*},\mathbb{P}^{X}_{\theta})=\inf_{\mathbb{Q}^{Z|X}\in\mathcal{Q}(\mathbb{P}^{X}_{*},\mathbb{P}^{Z})}\Exp_{X\sim\mathbb{P}^{X}_{*}}\left[\Exp_{Z\sim\mathbb{Q}^{Z|X}(\cdot|X)}\left[c(X,g_{\theta}(Z))\right]\right], 
 
 (73) 
 
 where 𝒬 ( ℙ ∗ X , ℙ Z ) ≔ { ℚ Z | X : 𝒳 → Δ ( 𝒵 ) ∣ ℚ Z = ℙ Z } \mathcal{Q}(\mathbb{P}^{X}_{*},\mathbb{P}^{Z})\coloneqq\{\mathbb{Q}^{Z|X}:\mathcal{X}\rightarrow\Delta(\mathcal{Z})\mid\mathbb{Q}^{Z}=\mathbb{P}^{Z}\} with ℚ Z ≔ 𝔼 X ∼ ℙ ∗ X [ ℚ Z | X ( ⋅ | X ) ] \mathbb{Q}^{Z}\coloneqq\Exp_{X\sim\mathbb{P}^{X}_{*}}[\mathbb{Q}^{Z|X}(\cdot|X)] . 
 
 
 
 

#### Section 5.2.3 Generative Networks Based on Maximum Mean Discrepancy

 
 Dziugaite et al. (2015) and Li et al. (2015) propose another way to train the model p θ X {p}^{X}_{\theta} corresponding to X = g θ ​ ( Z ) X=g_{\theta}(Z) and Z ∼ p Z Z\sim{p}^{Z} (again, we point out that formally, this model corresponds to the pushforward of p Z {p}^{Z} through g θ g_{\theta} ): by minimizing maximum mean discrepancy ( Section 3.6 ). Although their motivation was to avoid the adversarial training involved in generative adversarial networks ( Section 4.2 ) rather than to model manifold-supported data, the resulting objective provides a mathematically principled way of training DGMs under the manifold setting. The training objective of generative moment matching networks is simply

 

 
 | 
 min θ ⁡ 𝕄 ​ 𝕄 ​ 𝔻 k 2 ​ ( p ∗ X , p θ X ) \min_{\theta}\mathbb{MMD}_{k}^{2}\left({p}^{X}_{\ast},{p}^{X}_{\theta}\right) | 
 | 
 (74) | 
 

 for a pre-specified kernel k k . Although minimizing MMD is straightforward, generative moment matching networks are not known for achieving good empirical performance: this highlights that, even though manifold-awareness should be considered a necessary condition for strong empirical results, it is not sufficient.

 
 
 To improve the empirical performance of generative moment matching networks, Li et al. (2017) propose maximum mean discrepancy generative adversarial networks (MMD GANs), where the main idea is to reintroduce adversarial training to learn the kernel. First, for a fixed auxiliary latent space 𝒵 ′ = ℝ d ′ \mathcal{Z}^{\prime}=\mathbb{R}^{d^{\prime}} , a given a kernel k : 𝒵 ′ × 𝒵 ′ → ℝ k:\mathcal{Z}^{\prime}\times\mathcal{Z}^{\prime}\rightarrow\mathbb{R} , and a neural network h ϕ : 𝒳 → 𝒵 ′ h_{\phi}:\mathcal{X}\rightarrow\mathcal{Z}^{\prime} , Li et al. (2017) defined the kernel k ϕ : 𝒳 × 𝒳 → ℝ k_{\phi}:\mathcal{X}\times\mathcal{X}\rightarrow\mathbb{R} as k ϕ ​ ( x , y ) ≔ k ⁡ ( h ϕ ​ ( x ) , h ϕ ​ ( y ) ) k_{\phi}(x,y)\coloneqq k(h_{\phi}(x),h_{\phi}(y)) . They then showed that, if some regularity conditions hold and h ϕ h_{\phi} is injective, then max ϕ ⁡ 𝕄 ​ 𝕄 ​ 𝔻 k ϕ 2 \max_{\phi}\mathbb{MMD}_{k_{\phi}}^{2} metrizes weak convergence, thus making it a mathematically sensible objective to train DGMs. In order to enforce injectivity of h ϕ h_{\phi} , Li et al. (2017) leverage the fact that a function h : 𝒳 → 𝒵 ′ h:\mathcal{X}\rightarrow\mathcal{Z}^{\prime} is injective on 𝒳 \mathcal{X} if and only if it admits a left inverse h † : 𝒵 ′ → 𝒳 h^{\dagger}:\mathcal{Z}^{\prime}\rightarrow\mathcal{X} , i.e. h † ​ ( h ​ ( x ) ) = x h^{\dagger}(h(x))=x for all x ∈ 𝒳 x\in\mathcal{X} . They thus introduce an auxiliary network h ϕ † h^{\dagger}_{\phi} (which need not share parameters with h ϕ h_{\phi} ), and train MMD GANs through

 

 
 | 
 min θ ⁡ max ϕ ​ 𝕄 ​ 𝕄 ​ 𝔻 k ϕ 2 ​ ( p ∗ X , p θ X ) − β ​ 𝔼 X ∼ 1 2 ​ p ∗ X + 1 2 ​ p θ X ​ [ ‖ X − h ϕ † ​ ( h ϕ ​ ( X ) ) ‖ 2 2 ] , \min_{\theta}\max_{\phi}\mathbb{MMD}_{k_{\phi}}^{2}\left({p}^{X}_{\ast},{p}^{X}_{\theta}\right)-\beta\mathbb{E}_{X\sim\frac{1}{2}{p}^{X}_{\ast}+\frac{1}{2}{p}^{X}_{\theta}}\left[\|X-h^{\dagger}_{\phi}\left(h_{\phi}(X)\right)\|_{2}^{2}\right], | 
 | 
 (75) | 
 

 where β 0 \beta 0 is a hyperparameter, and the second term encourages h ϕ h_{\phi} to admit h ϕ † h_{\phi}^{\dagger} as a left inverse on the supports of p ∗ X {p}^{X}_{\ast} and p θ X {p}^{X}_{\theta} .
Similarly to Wasserstein GANs ( Section 5.2.1 ), the empirical performance of MMD GANs benefits from gradient regularization during training ( Bińkowski et al., 2018 ; Arbel et al., 2018 ) .

 
 
 

#### Section 5.2.4 Generalized Energy-Based Models

 
 Generalized energy-based models ( Arbel et al., 2021 , GEBMs;) combine generative adversarial networks ( Section 4.2 ) with energy-based models ( Section 4.1.4 ). GEBMs consist of a fixed prior p Z {p}^{Z} supported on 𝒵 \mathcal{Z} , a generator g θ 1 : 𝒵 → 𝒳 g_{\theta_{1}}:\mathcal{Z}\rightarrow\mathcal{X} , and an energy function E θ 2 : 𝒳 → ℝ E_{\theta_{2}}:\mathcal{X}\rightarrow\mathbb{R} , where we explicitly distinguish between the parameters of these components as θ = ( θ 1 , θ 2 ) \theta=(\theta_{1},\theta_{2}) . As in GANs, the prior along with the generator implicitly define the density p θ 1 g p^{g}_{\theta_{1}} of X = g θ 1 ​ ( Z ) X=g_{\theta_{1}}(Z) , where Z ∼ p Z Z\sim{p}^{Z} . 20 20 
 20 
 
 
 
 Note that in Section 4.2 we denoted p θ 1 g p^{g}_{\theta_{1}} as p θ X {p}^{X}_{\theta} , but we use different notation here as GEBMs further modify p θ 1 g p^{g}_{\theta_{1}} . This is not a full-dimensional density, but rather it is supported on the model manifold ℳ θ 1 ≔ g θ 1 ​ ( 𝒵 ) \mathcal{M}_{\theta_{1}}\coloneqq g_{\theta_{1}}(\mathcal{Z}) . 21 21 
 21 
 
 
 
 Formally ℳ θ 1 \mathcal{M}_{\theta_{1}} need not be a manifold, even if g θ 1 g_{\theta_{1}} is smooth, as it might have points of self-intersection. The measure-theoretic formulation of GEBMs in the grey box below remains nonetheless valid. GEBMs define an EBM on ℳ θ 1 \mathcal{M}_{\theta_{1}} by re-weighting p θ 1 g p_{\theta_{1}}^{g} through the use of E θ 2 E_{\theta_{2}} as an energy function:

 

 
 | 
 p θ X ​ ( x ) ∝ p θ 1 g ​ ( x ) ​ e − E θ 2 ​ ( x ) . {p}^{X}_{\theta}(x)\propto p_{\theta_{1}}^{g}(x)e^{-E_{\theta_{2}}(x)}. | 
 | 
 (76) | 
 

 Despite E θ 2 E_{\theta_{2}} being defined over 𝒳 \mathcal{X} , the above density is only defined on ℳ θ 1 \mathcal{M}_{\theta_{1}} and is thus also not a full-dimensional density. 22 22 
 22 
 
 
 
 More formally, the ∝ \propto symbol in Equation 76 should be understood as proportional within ℳ θ 1 \mathcal{M}_{\theta_{1}} , only integrating over the model manifold, i.e. p θ X ​ ( x ) = p θ 1 g ​ ( x ) ​ e − E θ 2 ​ ( x ) / ∫ ℳ θ 1 p θ 1 g ​ e − E θ 2 ​ d vol ℳ θ 1 {p}^{X}_{\theta}(x)=p_{\theta_{1}}^{g}(x)e^{-E_{\theta_{2}}(x)}/\int_{\mathcal{M}_{\theta_{1}}}p_{\theta_{1}}^{g}e^{-E_{\theta_{2}}}{\textnormal{d}}\textrm{vol}_{\mathcal{M}_{\theta_{1}}} . The intuition behind GEBMs is that the generator can easily learn to map to ℳ \mathcal{M} , whereas the energy function helps correct the distribution within the learned manifold.

 
 
 In order to train GEBMs, Arbel et al. (2021) define the quantity

 

 
 | 
 𝕂 𝔸 𝕃 𝔼 ( p ∗ X ∥ p θ 1 g ) ≔ sup E ∈ ℰ , ϕ ∈ ℝ 1 − ϕ − 𝔼 X ∼ p ∗ X [ E ( X ) ] − 𝔼 Z ∼ p Z [ e − E ​ ( g θ 1 ​ ( Z ) ) − ϕ ] , \mathbb{KALE}\left({p}^{X}_{\ast}\,\|\,p_{\theta_{1}}^{g}\right)\coloneqq\sup_{E\in\mathcal{E},\phi\in\mathbb{R}}1-\phi-\mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[E(X)\right]-\mathbb{E}_{Z\sim{p}^{Z}}\left[e^{-E(g_{\theta_{1}}(Z))-\phi}\right], | 
 | 
 (77) | 
 

 where ℰ \mathcal{E} is a set of Lipschitz energy functions satisfying certain regularity conditions (which are satisfied by feed-forward neural networks). They then show that 𝕂 𝔸 𝕃 𝔼 ( p ∗ X ∥ p θ 1 g ) \mathbb{KALE}({p}^{X}_{\ast}\,\|\,p_{\theta_{1}}^{g}) is a meaningfully defined divergence between p ∗ X {p}^{X}_{\ast} and p θ 1 g p_{\theta_{1}}^{g} , even when their supports do not perfectly overlap (i.e. it metrizes weak convergence, more details are provided in the grey box below), so that it provides a sensible objective to train the generator. As a divergence, 𝕂 ​ 𝔸 ​ 𝕃 ​ 𝔼 \mathbb{KALE} is intimately related to the KL divergence: Arbel et al. (2021) also show that if E θ 2 E_{\theta_{2}} achieves the supremum in Equation 77 , then 𝕂 𝕃 ( p ∗ X ∥ p θ X ) ≤ 𝕂 𝕃 ( p ∗ X ∥ p θ 1 g ) \mathbb{KL}({p}^{X}_{\ast}\,\|\,{p}^{X}_{\theta})\leq\mathbb{KL}({p}^{X}_{\ast}\,\|\,p_{\theta_{1}}^{g}) . 23 23 
 23 
 
 
 
 Note that none of the involved densities – namely p ∗ X {p}^{X}_{\ast} , p θ X {p}^{X}_{\theta} , and p θ 1 g p_{\theta_{1}}^{g} – are full-dimensional densities, so the KL divergence between them could be infinite ( Section 3.4 ). The inequality is thus trivially true when the support of p θ X {p}^{X}_{\theta} and p θ 1 g p_{\theta_{1}}^{g} , i.e. ℳ θ 1 \mathcal{M}_{\theta_{1}} , does not match ℳ \mathcal{M} . Nonetheless, when the generator perfectly recovers ℳ \mathcal{M} , the inequality does justify the use of the energy function in GEBMs. Therefore, the re-weighting done in Equation 76 to p θ 1 g p_{\theta_{1}}^{g} indeed improves upon simply using p θ 1 g p_{\theta_{1}}^{g} . Putting these properties together, Arbel et al. (2021) train GEBMs through

 

 
 | 
 min θ 1 ⁡ max θ 2 , ϕ ​ 1 − ϕ − 𝔼 X ∼ p ∗ X ​ [ E θ 2 ​ ( X ) ] − 𝔼 Z ∼ p Z ​ [ e − E θ 2 ​ ( g θ 1 ​ ( Z ) ) − ϕ ] , \min_{\theta_{1}}\max_{\theta_{2},\phi}1-\phi-\mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[E_{\theta_{2}}(X)\right]-\mathbb{E}_{Z\sim{p}^{Z}}\left[e^{-E_{\theta_{2}}(g_{\theta_{1}}(Z))-\phi}\right], | 
 | 
 (78) | 
 

 where the Lipschitz constraint on E θ 2 E_{\theta_{2}} is enforced as in Wasserstein GANs ( Section 5.2.1 ) and ϕ ∈ ℝ \phi\in\mathbb{R} is a free auxiliary parameter.

 
 
 Arbel et al. (2021) also show that in order to sample from a GEBM as defined through Equation 76 , one can first sample Z Z from the EBM p θ Z {p}^{Z}_{\theta} on 𝒵 \mathcal{Z} given by

 

 
 | 
 p θ Z ​ ( z ) ∝ p Z ​ ( z ) ​ e − E θ 2 ​ ( g θ 1 ​ ( z ) ) , {p}^{Z}_{\theta}(z)\propto{p}^{Z}(z)e^{-E_{\theta_{2}}(g_{\theta_{1}}(z))}, | 
 | 
 (79) | 
 

 and then setting X = g θ 1 ​ ( Z ) X=g_{\theta_{1}}(Z) will produce a sample from p θ X {p}^{X}_{\theta} . Note that p θ Z {p}^{Z}_{\theta} is now a full-dimensional density in 𝒵 \mathcal{Z} , and it can thus be sampled through Markov chain Monte Carlo as standard in EBMs. We point out that GEBMs are intimately linked to two-step models, which we discuss in Section 5.3 .

 
 
 Finally, we highlight some related works. Che et al. (2020) proposed a similar model to GEBMs, but their model is trained as a standard GAN ( Equation 41 ), and the EBM is defined post-hoc by using the discriminator h ϕ ∗ h_{\phi^{\ast}} ; despite the similarity with GEBMs, this procedure does not endow manifold-unaware GANs with manifold-awareness. Birrell et al. (2022) constructed a class of divergences, which 𝕂 ​ 𝔸 ​ 𝕃 ​ 𝔼 \mathbb{KALE} belongs to, by extending its relationship with the KL divergence to general f f -divergences; and Gu et al. (2024) leveraged these divergences, along with optimal transport ( Section 3.5 ), to obtain a manifold-aware adversarial training objective for continuous normalizing flows ( Section 4.1.3 ).

 
 
 
 We now formalize the presentation of GEBMs. Here we denote p θ 1 g p_{\theta_{1}}^{g} as a probability measure, g θ 1 ​ # ​ ℙ Z g_{\theta_{1}\#}\mathbb{P}^{Z} , instead of as a density. The model distribution ℙ θ X \mathbb{P}^{X}_{\theta} is then defined through its Radon-Nikodym derivative with respect to g θ 1 ​ # ​ ℙ Z g_{\theta_{1}\#}\mathbb{P}^{Z} , 
 

 
 
 p θ X ​ ( x ) = d ​ ℙ θ X d ​ g θ 1 ​ # ​ ℙ Z ​ ( x ) ≔ e − E θ 2 ​ ( x ) 𝔼 X ∼ g θ 1 ​ # ​ ℙ Z ​ [ e − E θ 2 ​ ( X ) ] , {p}^{X}_{\theta}(x)=\dfrac{{\textnormal{d}}\mathbb{P}^{X}_{\theta}}{{\textnormal{d}}g_{\theta_{1}\#}\mathbb{P}^{Z}}(x)\coloneqq\frac{e^{-E_{\theta_{2}}(x)}}{\mathbb{E}_{X\sim g_{\theta_{1}\#}\mathbb{P}^{Z}}[e^{-E_{\theta_{2}}(X)}]}, 
 
 (80) 
 
 where the expectation is assumed to be finite. Arbel et al. (2021) proved under mild conditions that: ( i ) (i) 𝕂 𝔸 𝕃 𝔼 ( ℙ ∗ X ∥ g θ 1 ​ # ℙ Z ) ≥ 0 \mathbb{KALE}(\mathbb{P}^{X}_{*}\,\|\,g_{\theta_{1}\#}\mathbb{P}^{Z})\geq 0 with equality if and only if ℙ ∗ X = g θ 1 ​ # ​ ℙ Z \mathbb{P}^{X}_{*}=g_{\theta_{1}\#}\mathbb{P}^{Z} ; and that ( i ​ i ) (ii) for a sequence of generators ( g θ 1 , t ) t = 1 ∞ (g_{\theta_{1,t}})_{t=1}^{\infty} , 𝕂 𝔸 𝕃 𝔼 ( ℙ ∗ X ∥ g θ 1 , t ​ # ℙ Z ) → 0 \mathbb{KALE}(\mathbb{P}^{X}_{*}\,\|\,g_{\theta_{1,t}\#}\mathbb{P}^{Z})\rightarrow 0 as t → ∞ t\rightarrow\infty if and only if g θ 1 , t ​ # ​ ℙ Z → 𝜔 ℙ ∗ X g_{\theta_{1,t}\#}\mathbb{P}^{Z}\xrightarrow{\omega}\mathbb{P}^{X}_{*} as t → ∞ t\rightarrow\infty . 
 
 
 
 

#### Section 5.2.5 Principal Component Flows

 
 Principal component flows ( Cunningham et al., 2022 , PCFs;) are a variant on standard normalizing flows ( Section 4.1.3 ) that seek to uncover manifold structure in a manner analogous to principal component analysis (PCA) and related to disentanglement ( Bengio et al., 2013 ) . If g θ : 𝒵 → 𝒳 g_{\theta}:\mathcal{Z}\to\mathcal{X} is a normalizing flow, the eigenvectors { ν 1 ​ ( x ) , … , ν D ​ ( x ) } \{\nu_{1}(x),\ldots,\nu_{D}(x)\} of ∇ z g θ ​ ( z ) ​ ∇ z g θ ​ ( z ) ⊤ \nabla_{z}g_{\theta}(z)\nabla_{z}g_{\theta}(z)^{\top} are said to be the principal components of g θ g_{\theta} at x = g θ ​ ( z ) x=g_{\theta}(z) and can be ordered using the corresponding eigenvalues as in standard PCA. The principal components at x x represent the principal axes of variation in the flow’s density p θ X {p}^{X}_{\theta} . In a manifold-learning context, principal axes with small eigenvalues represent off-manifold directions, while those with the highest eigenvalues represent primary directions of variation along the manifold, as illustrated in Figure 7 : we highlight that here p ∗ X {p}^{X}_{\ast} is assumed full-dimensional and to concentrate around ℳ \mathcal{M} , rather than being strictly manifold-supported.

 
 
 
 
 
 
 
 
 Figure 7: The motivation behind PCFs. (a) The principal components, ν 1 ​ ( x ) \nu_{1}(x) and ν 2 ​ ( x ) \nu_{2}(x) , of a trained NF with density p θ ∗ X {p}^{X}_{\theta^{\ast}} , taken at a point x ∈ 𝒳 x\in\mathcal{X} . These principal components are scaled according to their eigenvalues: here, ν 1 ​ ( x ) \nu_{1}(x) is the primary direction of variation along the manifold ℳ \mathcal{M} .
 (b) A comparison between the contours of two NFs: p θ NF ∗ X {p}^{X}_{\theta^{\ast}_{\text{NF}}} and p θ PCF ∗ X {p}^{X}_{\theta^{\ast}_{\text{PCF}}} , trained as a standard NF and as a PCF, respectively. Contours are visualized by the way each NF maps the grid lines of 𝒵 \mathcal{Z} . Since the contours mapped by g θ PCF ∗ g_{\theta^{\ast}_{\text{PCF}}} correspond to the principal components of p θ PCF ∗ X {p}^{X}_{\theta^{\ast}_{\text{PCF}}} , the NF model p θ PCF ∗ X {p}^{X}_{\theta^{\ast}_{\text{PCF}}} is formally a principal component flow. 
 
 
 We now discuss a special case of PCFs as an introduction; for a presentation of the method in more generality, see the work of Cunningham et al. (2022) . PCFs involve the notion of contour log-likelihoods , which here can be interpreted as the likelihood along the i i th coordinate curve of the flow,

 

 
 | 
 log ⁡ p Z i ​ ( z i ) − log ⁡ ( ∇ z g θ ​ ( x ) i ​ i ) \log p^{Z_{i}}(z_{i})-\log\left(\nabla_{z}g_{\theta}(x)_{ii}\right) | 
 | 
 (81) | 
 

 for i ∈ { 1 , 2 , … , D } i\in\{1,2,\dots,D\} , where p Z ​ ( z ) = ∏ i = 1 D p Z i ​ ( z i ) {p}^{Z}(z)=\prod_{i=1}^{D}p^{Z_{i}}(z_{i}) is a coordinatewise factorization of the latent prior (which is typically Gaussian), and ∇ x g θ ​ ( x ) i ​ i \nabla_{x}g_{\theta}(x)_{ii} is the i i th entry on the diagonal of ∇ x g θ ​ ( x ) \nabla_{x}g_{\theta}(x) .

 
 
 The goal of training a PCF is to align the principal components of g θ g_{\theta} with its latent coordinates on the manifold ( Figure 7 ). One of the key insights of Cunningham et al. (2022) is that the difference between the model’s log density log ⁡ p θ X ​ ( x ) \log{p}^{X}_{\theta}(x) and the sum of its contour log-likelihoods measures the diagonality of ∇ z g θ ​ ( z ) ​ ∇ z g θ ​ ( z ) ⊤ \nabla_{z}g_{\theta}(z)\nabla_{z}g_{\theta}(z)^{\top} and hence how well the model’s latent coordinates are aligned with its principal components. From this, a regularizer can be derived,

 

 
 | 
 ℐ ⁡ ( θ ) ≔ 𝔼 X ∼ p ∗ X ⁡ [ log ⁡ p θ X ​ ( X ) − ∑ i = 1 D log ⁡ p Z i ​ ( f θ ​ ( X ) i ) + log ⁡ ( ∇ x f θ ​ ( X ) ii ) ] , \mathcal{I}(\theta)\coloneqq\Exp_{X\sim{p}^{X}_{\ast}}\left[\log{p}^{X}_{\theta}(X)-\sum_{i=1}^{D}\log p^{Z_{i}}\left(f_{\theta}(X)_{i}\right)+\log\left(\nabla_{x}f_{\theta}(X)_{ii}\right)\right], | 
 | 
 (82) | 
 

 where f θ ​ ( X ) i f_{\theta}(X)_{i} denotes the i i th coordinate of f θ ​ ( X ) f_{\theta}(X) . 24 24 
 24 
 
 
 
 Note that the meaning of f θ ​ ( X ) 1 f_{\theta}(X)_{1} is different here than in Section 5.1.4 , where it refers to the first d d coordinates of f θ ​ ( X ) f_{\theta}(X) . 
 ℐ ⁡ ( θ ) \mathcal{I}(\theta) is a non-positive quantity such that ℐ ⁡ ( θ ) = 0 \mathcal{I}(\theta)=0 if and only if the latent coordinates are perfectly aligned with the flow’s principal components at each point x ∈ 𝒳 x\in\mathcal{X} . Maximizing ℐ ⁡ ( θ ) \mathcal{I}(\theta) as a regularizer while maximizing the likelihood aligns the flow’s latent coordinates with the principal manifolds of the data. We highlight that while this objective was derived to provide a useful inductive bias when manifolds are involved, it remains nonetheless based on full-dimensional likelihoods, and is thus subject to the corresponding pathologies ( Section 4.1 and Section 4.1.1 ).

 
 
 Canonical manifold flows ( Flouris Konukoglu, 2023 , CMFs;) take a related approach which directly penalizes off-diagonal elements of ∇ z g θ ​ ( f θ ​ ( X ) ) ​ ∇ z g θ ​ ( f θ ​ ( X ) ) ⊤ \nabla_{z}g_{\theta}(f_{\theta}(X))\nabla_{z}g_{\theta}(f_{\theta}(X))^{\top} . This results in a potentially looser regularizer which is designed to achieve the same goal at optimality as PCFs. Both PCFs and CMFs have been shown to naturally represent ℳ \mathcal{M} using a subset of the flow’s latent coordinates, meaning they automatically discover an estimate d d of the dimension d ∗ d^{\ast} of ℳ \mathcal{M} without the practitioner having to set it as a hyperparameter.

 
 
 
 

### Section 5.3 Manifold-Awareness through Two-Step Models

 
 Except for generative adversarial networks, all the DGMs described in Section 4 are manifold-unaware as a direct consequence of misspecified dimensionality: the data is d ∗ d^{\ast} -dimensional, whereas the model is D D -dimensional. The methods described in Section 5.1 address this misspecification by adding noise to the data, making it D D -dimensional, whereas those from Section 5.2 use manifold-appropriate losses. Another approach to enable manifold-awareness, which we cover here, is to instead reduce the ambient dimension of the data to match its intrinsic dimension before learning its distribution. Two-step approaches do this by separating the overall generative modelling process into two distinct steps.
Manifold learning (step 1 1 ) typically involves some form of encoding-decoding, which uncovers a lower-dimensional representation space 𝒵 \mathcal{Z} whose dimension d d ideally matches the intrinsic dimension d ∗ d^{\ast} of the given data.
Distribution learning (step 2 2 ) is then carried out on the obtained manifold, which often takes the form of generative modelling of the d d -dimensional representations learned in the previous step.
Importantly, this two-step procedure aims to remove the dimensionality mismatch between the data and the model, and thus circumvent any woes caused by it, such as manifold overfitting ( Section 4.1 ). When discussing two-step models, it will be useful to distinguish between the generative parameters of each step, and we thus write θ = ( θ 1 , θ 2 ) \theta=(\theta_{1},\theta_{2}) , where θ 1 \theta_{1} are the generative parameters required for the first step, and θ 2 \theta_{2} those for the second one. We now outline these two steps in more detail:

 
 
 
 • 
 
 Manifold learning  Most two-step models use autoencoder-based methods for manifold learning ( Section 3.2 ) as in Equation 6 ,

 

 
 | 
 min θ 1 , ϕ ⁡ 𝔼 X ∼ p ∗ X ​ [ ‖ X − g θ 1 ​ ( f ϕ ​ ( X ) ) ‖ 2 2 ] , \min_{\theta_{1},\phi}\mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[\|X-g_{\theta_{1}}\left(f_{\phi}(X)\right)\|_{2}^{2}\right], | 
 | 
 (83) | 
 

 or any variant such as variational autoencoders ( Section 4.1.2 ), although we will see in Section 5.4.1 that non-autoencoder-based choices are also possible. Importantly, the goal in this step is to perform manifold learning rather than generative modelling, so e.g. even if a VAE is used, it is interpreted as a regularized autoencoder rather than a generative model.

 

 • 
 
 Distribution learning  This step consists of learning the distribution on the manifold obtained in the previous step. In the standard setup where manifold learning is performed with an autoencoder-based model, a pre-trained encoder f ϕ ∗ f_{\phi^{\ast}} and decoder g θ 1 ∗ g_{\theta_{1}^{\ast}} pair is available.
The encoder defines a distribution q ϕ ∗ Z q^{Z}_{\phi^{\ast}} of encoded data f ϕ ∗ ​ ( X ) f_{\phi^{\ast}}(X) 
where X ∼ p ∗ X X\sim{p}^{X}_{\ast} (formally, q ϕ ∗ Z q_{\phi^{\ast}}^{Z} is the pushforward density of p ∗ X {p}^{X}_{\ast} through f ϕ ∗ f_{\phi^{\ast}} ), which can be learned
by instantiating a DGM p θ 2 Z {p}^{Z}_{\theta_{2}} on 𝒵 \mathcal{Z} , and training it with any of the methods covered in this survey (but changing the target distribution from the D D -dimensional p ∗ X {p}^{X}_{\ast} to the d d -dimensional q ϕ ∗ Z q^{Z}_{\phi^{\ast}} ), while keeping the encoder f ϕ ∗ f_{\phi^{\ast}} and decoder g θ 1 ∗ g_{\theta_{1}^{\ast}} frozen. Then, once this low-dimensional DGM is trained, resulting in p θ 2 ∗ Z {p}^{Z}_{\theta_{2}^{\ast}} , the distribution of the two-step model on the learned manifold can be sampled through X = g θ 1 ∗ ​ ( Z ) X=g_{\theta^{\ast}_{1}}(Z) , where Z ∼ p θ 2 ∗ Z Z\sim{p}^{Z}_{\theta_{2}^{\ast}} (i.e. p θ ∗ X {p}^{X}_{\theta^{\ast}} is formally given by the pushforward of p θ 2 ∗ Z {p}^{Z}_{\theta_{2}^{\ast}} through g θ 1 ∗ g_{\theta_{1}^{\ast}} ).

 

 
 
 
 Importantly, by solving the generative modelling task in 𝒵 \mathcal{Z} instead of 𝒳 \mathcal{X} , the support of the target distribution q ϕ ∗ Z q^{Z}_{\phi^{\ast}} is f ϕ ∗ ​ ( ℳ ) ⊆ 𝒵 f_{\phi^{\ast}}(\mathcal{M})\subseteq\mathcal{Z} , whose dimension should intuitively be given by min ⁡ ( d , d ∗ ) \min(d,d^{\ast}) . If the latent dimension d d is chosen properly (i.e. d = d ∗ d=d^{\ast} ), we should then expect q ϕ ∗ Z q^{Z}_{\phi^{\ast}} to be full-dimensional within 𝒵 \mathcal{Z} . This full-dimensionality stands in contrast to the case discussed in Section 4.1 where the target distribution p ∗ X {p}^{X}_{\ast} is d ∗ d^{\ast} -dimensional but its corresponding ambient space 𝒳 \mathcal{X} is D D -dimensional. Furthermore, even if d d is overspecified as d ∗ d D d^{\ast} d D , the “dimensionality gap” – i.e. the ambient dimension of the model minus that of the true manifold – of the second-step model is d − d ∗ d-d^{\ast} , which is smaller than D − d ∗ D-d^{\ast} . 25 25 
 25 
 
 
 
 The case where d d is underspecified, i.e. d d ∗ d d^{\ast} , is less interesting, as in this case ℳ \mathcal{M} , and thus p ∗ X {p}^{X}_{\ast} , cannot be recovered. Thus, we should intuitively expect any manifold-related woes arising from dimensionality mismatch to be milder than the corresponding full-dimensional issues.

 
 
 Many two-step models have been proposed in the literature, sometimes with the manifold setting in mind, and some other times simply for tractability, as training DGMs on a low-dimensional latent space is cheaper than doing so in high-dimensional ambient space. Loaiza-Ganem et al. (2022a) provided a theoretical justification for all of these models, proving that under mild regularity conditions, when d = d ∗ d=d^{\ast} and 𝔼 X ∼ p ∗ X ​ [ ‖ X − g θ 1 ∗ ​ ( f ϕ ∗ ​ ( X ) ) ‖ 2 2 ] = 0 \mathbb{E}_{X\sim{p}^{X}_{\ast}}[\|X-g_{\theta_{1}^{\ast}}(f_{\phi^{\ast}}(X))\|_{2}^{2}]=0 (i.e. perfect reconstructions), then: ( i ) (i) q ϕ ∗ Z q^{Z}_{\phi^{\ast}} is indeed full-dimensional, and thus p θ 2 Z {p}^{Z}_{\theta_{2}} can be any DGM, even if manifold-unaware (e.g. trained through maximum-likelihood), and still learn q ϕ ∗ Z q^{Z}_{\phi^{\ast}} ; and ( i ​ i ) (ii) transforming samples from p θ 2 ∗ Z {p}^{Z}_{\theta_{2}^{\ast}} through g θ 1 ∗ g_{\theta_{1}^{\ast}} is equivalent to sampling from p ∗ X {p}^{X}_{\ast} , i.e. two-step models recover p ∗ X {p}^{X}_{\ast} . This result, which we discuss further in the next grey box, implies that two-step models learn ℳ \mathcal{M} , which is an interesting observation since autoencoders by themselves need not – see Figure 2 and the discussion in Section 3.2 . In particular, when p θ 2 ∗ Z {p}^{Z}_{\theta_{2}^{\ast}} is perfectly trained it must be supported on f ϕ ∗ ​ ( ℳ ) f_{\phi^{\ast}}(\mathcal{M}) , and since g θ 1 ∗ ​ ( f ϕ ∗ ​ ( ℳ ) ) = ℳ g_{\theta_{1}^{\ast}}(f_{\phi^{\ast}}(\mathcal{M}))=\mathcal{M} , it follows that together, the support of p θ 2 ∗ Z {p}^{Z}_{\theta_{2}^{\ast}} and the decoder g θ 1 ∗ g_{\theta_{1}^{\ast}} jointly characterize ℳ \mathcal{M} .

 
 
 Here we briefly summarize two-step models which are straightforward combinations of an autoencoder-based model with any other generative model on latent space; two-step models warranting additional discussion are covered in Section 5.3.2 and Section 5.3.3 . Dai Wipf (2019) use a VAE for manifold learning and another VAE for distribution learning on latent space. Xiao et al. (2019) use an autoencoder and a normalizing flow ( Section 4.1.3 ), as do Boehm Seljak (2022) . Ghosh et al. (2020) use an autoencoder with an added regularization term, along with a Gaussian mixture model (which, while not deep, remains a generative model and thus fits the two-step framework). Li et al. (2015) use an autoencoder and then train a generative moment matching network ( Section 5.2.3 ) on the recovered latents, and Dao et al. (2023) use a regularized VAE along with conditional flow matching ( Section 5.1.3 ). The improved generative performance reported in many of these works compared to their full-dimensional counterparts may be attributed to the reduction of dimension mismatch.

 
 
 We also point out that generalized energy-based models ( Section 5.2.4 ) are very similar to autoencoder-based two-step models, since GEBMs instantiate an energy-based model ( Section 4.1.4 ) on a low-dimensional latent space, whose samples are then mapped through a decoder. GEBMs are nonetheless not autoencoder-based two-step models, the differences being that: GEBMs do not require an encoder, the distribution used by GEBMs on 𝒵 \mathcal{Z} ( Equation 79 ) shares parameters with the decoder, GEBMs are trained end-to-end, and they are limited to using EBMs on the latent space.

 
 
 End-to-end training 

 
 Although the two-step models described above are manifold-aware, the objective in the first step does not necessarily promote representations that are conducive to distribution learning in the second step.
Therefore, one might expect that training these models in an end-to-end manner could further improve their performance. Albeit not always inspired by this motivation, several works have proposed end-to-end objectives, some in the context of variational autoencoders, and some in the context of injective normalizing flows; we discuss the former here and the latter in Section 5.3.3 . VAEs as described in Section 4.1.2 assume a fixed prior p Z {p}^{Z} on the latent space 𝒵 \mathcal{Z} . However, the prior p θ 2 Z {p}^{Z}_{\theta_{2}} can be made trainable, in which case maximizing the ELBO ( Equation 24 ), i.e.

 

 
 | 
 max θ , ϕ 𝔼 X ∼ p ∗ X [ 𝔼 Z ∼ q Z | X ϕ ( ⋅ | X ) [ log p θ 1 X | Z ( X | Z ) ] − 𝕂 𝕃 ( q ϕ Z | X ( ⋅ | X ) ∥ p θ 2 Z ) ] , \max_{\theta,\phi}\mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[\mathbb{E}_{Z\sim q^{Z|X}_{\phi}(\cdot|X)}[\log p^{X|Z}_{\theta_{1}}(X|Z)]-\mathbb{KL}\left(q^{Z|X}_{\phi}(\cdot|X)\,\Big\|\,{p}^{Z}_{\theta_{2}}\right)\right], | 
 | 
 (84) | 
 

 remains a valid objective (assuming p ∗ X {p}^{X}_{\ast} is full-dimensional) for end-to-end training of p θ 1 X | Z p_{\theta_{1}}^{X|Z} and p θ 2 Z {p}^{Z}_{\theta_{2}} . Depending on the choice of p θ 2 Z {p}^{Z}_{\theta_{2}} additional computational tricks might be required to efficiently optimize the ELBO. Tomczak Welling (2018) instantiate p θ 2 Z {p}^{Z}_{\theta_{2}} as a Gaussian mixture model; Sønderby et al. (2016) , Vahdat Kautz (2020) , and Child (2021) use learnable hierarchical priors; Chen et al. (2017) use normalizing flows; Pang et al. (2020) use energy-based models; and Vahdat et al. (2021) use diffusion models ( Section 5.1.2 ). When p θ 1 X | Z p_{\theta_{1}}^{X|Z} is a flexible enough full-dimensional density as in Equation 22 , all these models remain susceptible to the manifold overfitting issues discussed in Section 4.1 and Section 4.1.2 , despite directly encouraging the encoder to learn representations whose distribution can be easily recovered by p θ 2 Z {p}^{Z}_{\theta_{2}} .

 
 
 Even though designing an end-to-end objective for training in a manifold-aware fashion is intuitively desirable, doing so is not always straightforward.
To see why, consider a two-step model whose first-step loss is given by Equation 83 , and whose second-step model is trained by minimizing 𝔻 ⁡ ( q ϕ ∗ Z , p θ 2 Z ) \mathbb{D}(q_{\phi^{\ast}}^{Z},p_{\theta_{2}}^{Z}) over θ 2 \theta_{2} for some divergence 𝔻 \mathbb{D} between probability distributions. Let us further assume that 𝔻 ⁡ ( q ϕ ∗ Z , p θ 2 Z ) \mathbb{D}(q_{\phi^{\ast}}^{Z},p_{\theta_{2}}^{Z}) cannot be computed without evaluating q ϕ ∗ Z q_{\phi^{\ast}}^{Z} , but that 𝔻 ⁡ ( q ϕ ∗ Z , p θ 2 Z ) = ℒ ⁡ ( p θ 2 Z , q ϕ ∗ Z ) + c ⁡ ( q ϕ ∗ Z ) \mathbb{D}(q_{\phi^{\ast}}^{Z},p_{\theta_{2}}^{Z})=\mathcal{L}(p_{\theta_{2}}^{Z};q_{\phi^{\ast}}^{Z})+c(q_{\phi^{\ast}}^{Z}) , where ℒ ⁡ ( p θ 2 Z , q ϕ ∗ Z ) \mathcal{L}(p_{\theta_{2}}^{Z};q_{\phi^{\ast}}^{Z}) can be computed without evaluating q ϕ ∗ Z q_{\phi^{\ast}}^{Z} , and where c ⁡ ( q ϕ ∗ Z ) c(q_{\phi^{\ast}}^{Z}) does not depend on p θ 2 Z p_{\theta_{2}}^{Z} . Divergences with these properties are prevalent; the KL divergence ( Section 3.4 ) is an instance, where ℒ ⁡ ( p θ 2 Z , q ϕ ∗ Z ) = − 𝔼 Z ∼ q ϕ ∗ Z ​ [ log ⁡ p θ 2 Z ​ ( Z ) ] \mathcal{L}(p_{\theta_{2}}^{Z};q_{\phi^{\ast}}^{Z})=-\mathbb{E}_{Z\sim q_{\phi^{\ast}}^{Z}}[\log p_{\theta_{2}}^{Z}(Z)] and c ⁡ ( q ϕ ∗ Z ) = 𝔼 Z ∼ q ϕ ∗ Z ​ [ log ⁡ q ϕ ∗ Z ​ ( Z ) ] c(q_{\phi^{\ast}}^{Z})=\mathbb{E}_{Z\sim q_{\phi^{\ast}}^{Z}}[\log q_{\phi^{\ast}}^{Z}(Z)] , as well as the Fisher divergence ( Section 4.3 ) which underpins score matching and thus diffusion models. In this case, the second-step model is trained by using the loss ℒ ⁡ ( p θ 2 Z , q ϕ ∗ Z ) \mathcal{L}(p_{\theta_{2}}^{Z};q_{\phi^{\ast}}^{Z}) , which is equivalent to minimizing 𝔻 ⁡ ( q ϕ ∗ Z , p θ 2 Z ) \mathbb{D}(q_{\phi^{\ast}}^{Z},p_{\theta_{2}}^{Z}) . Naïvely combining the losses of this two-step model into a single loss for end-to-end training would result in the objective

 

 
 | 
 min θ , ϕ ⁡ 𝔼 X ∼ p ∗ X ​ [ ‖ X − g θ 1 ​ ( f ϕ ​ ( X ) ) ‖ 2 2 ] + β ​ ℒ ​ ( p θ 2 Z , q ϕ Z ) \min_{\theta,\phi}\mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[\|X-g_{\theta_{1}}(f_{\phi}(X))\|_{2}^{2}\right]+\beta\mathcal{L}\left(p_{\theta_{2}}^{Z};q_{\phi}^{Z}\right) | 
 | 
 (85) | 
 

 for some β 0 \beta 0 . This naïve objective ignores c ⁡ ( q ϕ Z ) c(q_{\phi}^{Z}) , so that it is not equivalent to

 

 
 | 
 min θ , ϕ ⁡ 𝔼 X ∼ p ∗ X ​ [ ‖ X − g θ 1 ​ ( f ϕ ​ ( X ) ) ‖ 2 2 ] + β ​ 𝔻 ​ ( q ϕ Z , p θ 2 Z ) . \min_{\theta,\phi}\mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[\|X-g_{\theta_{1}}(f_{\phi}(X))\|_{2}^{2}\right]+\beta\mathbb{D}\left(q_{\phi}^{Z},p_{\theta_{2}}^{Z}\right). | 
 | 
 (86) | 
 

 In short, Equation 85 does not provide a valid objective for manifold-aware end-to-end training since c ⁡ ( q ϕ Z ) c(q_{\phi}^{Z}) cannot be ignored when ϕ \phi is not fixed after the first step of training. Unfortunately, although Equation 86 specifies a principled objective to address the issue, it remains intractable when c ⁡ ( q ϕ Z ) c(q_{\phi}^{Z}) cannot be computed.

 
 
 
 More formally, a trained autoencoder-based two-step model is given by a distribution ℙ θ 2 ∗ Z \mathbb{P}^{Z}_{\theta_{2}^{\ast}} on 𝒵 \mathcal{Z} along with a decoder g θ 1 ∗ : 𝒵 → 𝒳 g_{\theta_{1}^{\ast}}:\mathcal{Z}\rightarrow\mathcal{X} , and the model distribution is given by ℙ θ ∗ X = g θ 1 ∗ ​ # ​ ℙ θ 2 ∗ Z \mathbb{P}^{X}_{\theta^{\ast}}=g_{\theta_{1}^{\ast}\#}\mathbb{P}^{Z}_{\theta_{2}^{\ast}} . The result of Loaiza-Ganem et al. (2022a) justifying two-step models states that, under mild regularity conditions, if d = d ∗ d=d^{\ast} and 𝔼 X ∼ ℙ ∗ X ​ [ ‖ X − g θ 1 ∗ ​ ( f ϕ ∗ ​ ( X ) ) ‖ 2 2 ] = 0 \mathbb{E}_{X\sim\mathbb{P}^{X}_{*}}[\|X-g_{\theta_{1}^{\ast}}(f_{\phi^{\ast}}(X))\|_{2}^{2}]=0 , then: 
 
 • 
 
 ℚ ϕ ∗ Z ≪ λ d \mathbb{Q}^{Z}_{\phi^{\ast}}\ll\lambda_{d} , where ℚ ϕ ∗ Z = f ϕ ∗ ​ # ​ ℙ ∗ X \mathbb{Q}^{Z}_{\phi^{\ast}}=f_{\phi^{\ast}\#}\mathbb{P}^{X}_{*} is the distribution of encoded data. 
 
 • 
 
 g θ 1 ∗ ​ # ​ ℚ ϕ ∗ Z = ℙ ∗ X g_{\theta_{1}^{\ast}\#}\mathbb{Q}^{Z}_{\phi^{\ast}}=\mathbb{P}^{X}_{*} . 
 
 
 The first point ensures ℚ ϕ ∗ Z \mathbb{Q}^{Z}_{\phi^{\ast}} admits a density with respect to λ d \lambda_{d} , so that it is full-dimensional. The second point ensures that if the target distribution ℚ ϕ ∗ Z \mathbb{Q}^{Z}_{\phi^{\ast}} is properly learned during the second step then two-step models recover the true data-generating distribution, i.e. if ℙ θ 2 ∗ Z = ℚ ϕ ∗ Z \mathbb{P}^{Z}_{\theta_{2}^{\ast}}=\mathbb{Q}^{Z}_{\phi^{\ast}} then ℙ θ ∗ X = ℙ ∗ X \mathbb{P}^{X}_{\theta^{\ast}}=\mathbb{P}^{X}_{*} . 
 
 
 
 

#### Section 5.3.1 Two-Step Models Minimize Wasserstein Distance

 
 Before continuing our review of existing two-step models ( Section 5.3 ), we highlight that these models can be interpreted through an optimal transport lens ( Section 3.5 ). To the best of our knowledge, this observation has not been made in the literature, and constitutes a novel contribution of our work. Here we still consider the model p θ X {p}^{X}_{\theta} given by the two learnable components g θ 1 g_{\theta_{1}} and p θ 2 Z {p}^{Z}_{\theta_{2}} , and will use the notation introduced for Wasserstein autoencoders ( Section 5.2.2 ). Key to our insight is Equation 70 , which, for the model p θ X {p}^{X}_{\theta} considered here, can be rewritten as

 

 
 | 
 𝕎 c ( p ∗ X , p θ X ) = inf q Z | X ∈ 𝒬 ⁡ ( p ∗ X , p θ 2 Z ) 𝔼 X ∼ p ∗ X [ 𝔼 Z ∼ q Z | X ( ⋅ | X ) [ c ( X , g θ 1 ( Z ) ) ] ] . \mathbb{W}^{c}({p}^{X}_{\ast},{p}^{X}_{\theta})=\inf_{q^{Z|X}\in\mathcal{Q}({p}^{X}_{\ast},{p}^{Z}_{\theta_{2}})}\Exp_{X\sim{p}^{X}_{\ast}}\left[\Exp_{Z\sim q^{Z|X}(\cdot|X)}\left[c(X,g_{\theta_{1}}(Z))\right]\right]. | 
 | 
 (87) | 
 

 This equality then implies that

 

 
 | 
 𝕎 c ​ ( p ∗ X , p θ X ) ≤ inf f ∈ ℱ ⁡ ( p ∗ X , p θ 2 Z ) 𝔼 X ∼ p ∗ X ​ [ c ⁡ ( X , g θ 1 ​ ( f ⁡ ( X ) ) ) ] , \mathbb{W}^{c}({p}^{X}_{\ast},{p}^{X}_{\theta})\leq\inf_{f\in\mathcal{F}({p}^{X}_{\ast},{p}^{Z}_{\theta_{2}})}\mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[c\left(X,g_{\theta_{1}}(f(X))\right)\right], | 
 | 
 (88) | 
 

 where ℱ ⁡ ( p ∗ X , p θ 2 Z ) \mathcal{F}({p}^{X}_{\ast},{p}^{Z}_{\theta_{2}}) is the set of functions f : 𝒳 → 𝒵 f:\mathcal{X}\rightarrow\mathcal{Z} such that if X ∼ p ∗ X X\sim{p}^{X}_{\ast} , then f ⁡ ( X ) ∼ p θ 2 Z f(X)\sim{p}^{Z}_{\theta_{2}} . To see that Equation 87 indeed implies Equation 88 , simply note that if f ∈ ℱ ⁡ ( p ∗ X , p θ 2 Z ) f\in\mathcal{F}({p}^{X}_{\ast},{p}^{Z}_{\theta_{2}}) , then the conditional (on X = x X=x ) distribution on 𝒵 \mathcal{Z} given by the point mass at f ⁡ ( x ) f(x) is in 𝒬 ⁡ ( p ∗ X , p θ 2 Z ) \mathcal{Q}({p}^{X}_{\ast},{p}^{Z}_{\theta_{2}}) . Equation 88 is used to justify the use of deterministic encoders within WAEs: doing so minimizes an upper bound of 𝕎 c ​ ( p ∗ X , p θ X ) \mathbb{W}^{c}({p}^{X}_{\ast},{p}^{X}_{\theta}) . We also point out that, assuming c ⁡ ( x , y ) c(x,y) is minimal if and only if x = y x=y , the bound becomes tight at optimality as long as perfect reconstructions are achievable with a deterministic autoencoder (i.e. X = g θ 1 ∗ ​ ( f ϕ ∗ ​ ( X ) ) X=g_{\theta_{1}^{\ast}}(f_{\phi^{\ast}}(X)) , see Section 3.2 and Section 5.4 for discussions of when this is possible with continuous autoencoders).

 
 
 For an encoder f ϕ f_{\phi} , we let q ϕ Z q^{Z}_{\phi} be the distribution of f ϕ ​ ( X ) f_{\phi}(X) where X ∼ p ∗ X X\sim{p}^{X}_{\ast} (formally, q ϕ Z q^{Z}_{\phi} is the pushforward density of p ∗ X {p}^{X}_{\ast} through f ϕ f_{\phi} ), and note that f ϕ ∈ ℱ ⁡ ( p ∗ X , p θ 2 Z ) f_{\phi}\in\mathcal{F}({p}^{X}_{\ast},{p}^{Z}_{\theta_{2}}) is equivalent to q ϕ Z = p θ 2 Z q^{Z}_{\phi}={p}^{Z}_{\theta_{2}} . In turn, Equation 88 justifies training the model p θ X {p}^{X}_{\theta} by minimizing an upper bound of its optimal transport cost through

 

 
 | 
 min θ , ϕ ⁡ 𝔼 X ∼ p ∗ X ​ [ c ⁡ ( X , g θ 1 ​ ( f ϕ ​ ( X ) ) ) ] subject to  ​ q ϕ Z = p θ 2 Z . \displaystyle\begin{split} \min_{\theta,\phi}\mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[c\left(X,g_{\theta_{1}}\left(f_{\phi}(X)\right)\right)\right]\\
 \text{subject to }q^{Z}_{\phi}={p}^{Z}_{\theta_{2}}.\end{split} | 
 | 
 (89) | 
 

 The key difference between this objective and that of WAEs is that the distribution on latent space is now learnable instead of being fixed. While this distinction with WAEs might seem conceptually trivial, it enables minimizing optimal transport cost through a two step process: in the first step, θ 1 \theta_{1} and ϕ \phi are trained to minimize the reconstruction error, 𝔼 X ∼ p ∗ X ​ [ c ⁡ ( X , g θ 1 ​ ( f ϕ ​ ( X ) ) ) ] \mathbb{E}_{X\sim{p}^{X}_{\ast}}[c(X,g_{\theta_{1}}(f_{\phi}(X)))] , with no regard for the constraint. This step results in a now fixed distribution q ϕ ∗ Z q^{Z}_{\phi^{\ast}} on 𝒵 \mathcal{Z} , which of course need not match p θ 2 Z {p}^{Z}_{\theta_{2}} . Thanks to p θ 2 Z {p}^{Z}_{\theta_{2}} being learnable and θ 2 \theta_{2} not appearing in the first-step objective, this mismatch can be addressed in the second step, where any objective over θ 2 \theta_{2} to match p θ 2 Z {p}^{Z}_{\theta_{2}} and q ϕ ∗ Z q^{Z}_{\phi^{\ast}} can be used. For example, if the second-step model were trained through maximum-likelihood, its objective would be 26 26 
 26 
 
 
 
 Note that the second-step objective could itself require additional auxiliary parameters but we omit this possibility for notational simplicity. 

 

 
 | 
 min θ 2 𝕂 𝕃 ( q ϕ ∗ Z ∥ p θ 2 Z ) , \min_{\theta_{2}}\mathbb{KL}\left(q^{Z}_{\phi^{\ast}}\,\|\,{p}^{Z}_{\theta_{2}}\right), | 
 | 
 (90) | 
 

 which indeed satisfies the constraint in Equation 89 at optimality. In other words, two-step models solve Equation 89 by optimizing an unconstrained version of the objective during the first step, and then ensuring the constraint is actually satisfied during the second step. Crucially, the second step does not affect the optimality of the first step because 𝔼 X ∼ p ∗ X ​ [ c ⁡ ( X , g θ 1 ​ ( f ϕ ​ ( X ) ) ) ] \mathbb{E}_{X\sim{p}^{X}_{\ast}}[c(X,g_{\theta_{1}}(f_{\phi}(X)))] does not depend on θ 2 \theta_{2} , so that two-step models indeed provide a valid way of solving Equation 89 .

 
 
 We now make some additional observations. ( i ) (i) When c c is given by the squared Euclidean distance, the corresponding loss for the first-step model is exactly that of a standard autoencoder ( Equation 6 and Equation 83 ). When the first-step model is trained through a different autoencoder-based objective, e.g. Equation 24 , we can simply interpret the model as a regularized autoencoder as long as it encourages perfect reconstructions at optimality. ( i ​ i ) (ii) Although two-step models are often trained using deterministic encoders, using arbitrarily flexible stochastic encoders would imply that 𝕎 c ​ ( p ∗ X , p θ X ) \mathbb{W}^{c}({p}^{X}_{\ast},{p}^{X}_{\theta}) is being minimized rather than an upper bound.

 
 
 In summary, we have justified the manifold-awareness of autoencoder-based two-step models through optimal transport. We believe that this result is not only interesting on its own, but also hope that by establishing a connection between seemingly unrelated manifold-aware model classes – namely two-step models and those which are trained through support-agnostic optimization objectives ( Section 5.2 ), such as WAEs – it will enable future improvements to both.

 
 
 
 Equation 88 is formalized as follows: 
 

 
 
 𝕎 c ​ ( ℙ ∗ X , ℙ θ X ) ≤ inf f ∈ ℱ ⁡ ( ℙ ∗ X , ℙ Z ) 𝔼 X ∼ p ∗ X ​ [ c ⁡ ( X , g θ ​ ( f ⁡ ( X ) ) ) ] , \mathbb{W}^{c}(\mathbb{P}^{X}_{*},\mathbb{P}^{X}_{\theta})\leq\inf_{f\in\mathcal{F}(\mathbb{P}^{X}_{*},\mathbb{P}^{Z})}\mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[c\left(X,g_{\theta}(f(X))\right)\right], 
 
 (91) 
 
 where ℱ ( ℙ ∗ X , ℙ Z ) ≔ { f : 𝒳 → 𝒵 ∣ f  is measurable, and  f # ℙ ∗ X = ℙ Z } \mathcal{F}(\mathbb{P}^{X}_{*},\mathbb{P}^{Z})\coloneqq\{f:\mathcal{X}\rightarrow\mathcal{Z}\mid f\text{ is measurable, and }f_{\#}\mathbb{P}^{X}_{*}=\mathbb{P}^{Z}\} . As a technical point, note that the unconstrained version of the right hand side of this equation – upon which the interpretation of two-step models as Wasserstein distance minimizers is based – i.e. 
 

 
 
 inf f ∈ ℱ 𝔼 X ∼ ℙ ∗ X ​ [ c ⁡ ( X , g θ 1 ​ ( f ⁡ ( X ) ) ) ] , \inf_{f\in\mathcal{F}}\mathbb{E}_{X\sim\mathbb{P}^{X}_{*}}\left[c\left(X,g_{\theta_{1}}\left(f(X)\right)\right)\right], 
 
 (92) 
 
 where ℱ ≔ { f : 𝒳 → 𝒵 ∣ f  is measurable } \mathcal{F}\coloneqq\{f:\mathcal{X}\rightarrow\mathcal{Z}\mid f\text{ is measurable}\} , involves an infimum over measurable functions (which is then minimized over θ 1 \theta_{1} during the first step). In the nonparametric regime ( Section 2.2 ) we assume that neural networks are flexible enough to approximate any continuous function arbitrarily well, but this property need not a priori extend to measurable functions. Intuitively this should however not be a problem thanks to Lusin’s theorem – which, informally, states that in certain settings any measurable function can be approximated by a continuous one. It is nonetheless pertinent to show that the infimum in Equation 92 can actually be replaced by a corresponding infimum over continuous functions, which we do below.
 
 
 
 Proposition 1 . 
 
 Let ℙ ∗ X \mathbb{P}^{X}_{*} be a probability measure on 𝒳 \mathcal{X} , g θ 1 : 𝒵 → 𝒳 g_{\theta_{1}}:\mathcal{Z}\rightarrow\mathcal{X} be measurable, and c : 𝒳 × 𝒳 → ℝ c:\mathcal{X}\times\mathcal{X}\rightarrow\mathbb{R} be measurable and such that there exists C 0 C 0 such that 
 

 
 
 sup ( x , y ) ∈ 𝒳 × 𝒳 | c ⁡ ( x , y ) | C . \sup_{(x,y)\in\mathcal{X}\times\mathcal{X}}|c(x,y)| C. 
 
 (93) 
 
 Then, 
 

 
 
 inf f ∈ ℱ 𝔼 X ∼ ℙ ∗ X ​ [ c ⁡ ( X , g θ 1 ​ ( f ⁡ ( X ) ) ) ] = inf f ∈ 𝒞 𝔼 X ∼ ℙ ∗ X ​ [ c ⁡ ( X , g θ 1 ​ ( f ⁡ ( X ) ) ) ] , \inf_{f\in\mathcal{F}}\mathbb{E}_{X\sim\mathbb{P}^{X}_{*}}\left[c\left(X,g_{\theta_{1}}\left(f(X)\right)\right)\right]=\inf_{f\in\mathcal{C}}\mathbb{E}_{X\sim\mathbb{P}^{X}_{*}}\left[c\left(X,g_{\theta_{1}}\left(f(X)\right)\right)\right], 
 
 (94) 
 
 where ℱ ≔ { f : 𝒳 → 𝒵 ∣ f  is measurable } \mathcal{F}\coloneqq\{f:\mathcal{X}\rightarrow\mathcal{Z}\mid f\text{ is measurable}\} and 𝒞 ≔ { f : 𝒳 → 𝒵 ∣ f  is continuous } \mathcal{C}\coloneqq\{f:\mathcal{X}\rightarrow\mathcal{Z}\mid f\text{ is continuous}\} . 
 
 
 
 Proof. 
 
 See Appendix B.2 . ∎ 
 
 
 
 This result allows us to formally interpret two-step models as minimizers of an upper bound of the Wasserstein distance in the nonparametric regime when the assumption in Equation 93 holds. We point out that this is a mild regularity condition, as it is always satisfied in the common case where 𝒳 \mathcal{X} is compact and c c is continuous (since continuous functions always achieve their supremums over compact sets).

 
 Finally, we point out that Patrini et al. (2020) claimed that, as long as ℙ ∗ X \mathbb{P}^{X}_{*} is non-atomic, then the inequality in Equation 91 is actually an equality. This would allow us to interpret two-step models as minimizing Wasserstein distance – not an upper bound – even when using deterministic encoders. However, Lee et al. (2024) found an error in the proof of Patrini et al. (2020) , so that only the upper bound interpretation remains valid. 
 
 
 
 

#### Section 5.3.2 Latent Diffusion Models

 
 Latent diffusion models ( Rombach et al., 2022 ; Peebles Xie, 2023 ; Zhang et al., 2024 ) are another class of two-step models ( Section 5.3 ). They first train a regularized autoencoder, which combines a Gaussian variational autoencoder objective ( Section 4.1.2 ) with various potential regularizers ( Larsen et al., 2016 ; Higgins et al., 2017 ; van den Oord et al., 2017 ) .
Once this autoencoder-based model is trained and the corresponding low-dimensional representations obtained, a diffusion model ( Section 5.1.2 ) s θ 2 Z : 𝒵 × ( 0 , T ] → 𝒵 s_{\theta_{2}}^{Z}:\mathcal{Z}\times(0,T]\rightarrow\mathcal{Z} is trained on them as the second-step model.

 
 
 As previously discussed, diffusion models can learn manifolds, but their score function must diverge to infinity at the end of the backward process ( Equation 52 ) as a consequence of the mismatch between the intrinsic and ambient dimensions of the data. To test that latent diffusion models are not as sensitive to this numerical pathology, we trained a diffusion model and a latent diffusion model on the CIFAR-10 dataset ( Krizhevsky Hinton, 2009 ) , with all experimental details provided in Appendix C . We plot the average squared Euclidean norm of the score functions, normalized by their dimension, 27 27 
 27 
 
 
 
 Note that normalizing squared Euclidean norm by dimension is the most natural way of enabling comparisons across dimensions. To see this consider a constant vector with all entries equal to 1 1 , which has a squared ℓ 2 \ell_{2} norm equal to its dimension; or consider a standard Gaussian vector, whose expected squared Euclidean norm also matches its dimension. along generated paths in Figure 8 : it is evident that the score function of diffusion models on latent space exhibits much better numerical behaviour than when these models are trained on ambient space. This result is suggested by theory, and to the best of our knowledge, we are the first to empirically confirm it.

 
 
 Figure 8: Average squared norm of the learned score function on CIFAR-10 over 100 100 generated paths from Equation 52 , normalized by dimension (i.e. dim = D = 3072 \texttt{dim}=D=3072 for diffusion models, and dim = d = 256 \texttt{dim}=d=256 for latent diffusion models); the shaded area corresponds to one standard deviation. The paths are stopped at time T − ε T-\varepsilon , with ε = 0.001 \varepsilon=0.001 . 
 
 
 Currently, latent diffusion models are amongst the best performing DGMs empirically, and the manifold lens provides a convincing explanation for this: ( i ) (i) they can learn manifolds; ( i ​ i ) (ii) they alleviate the numerical issues of diffusion models; and ( i ​ i ​ i ) (iii) they are robust to misspecification of the dimension of the latent space, in the sense that even if d ∗ d d^{\ast} d (i.e. the latent dimension is specified as larger than the true intrinsic dimension), they still learn their target distribution – albeit with an exploding score function. To see this, simply note that setting d ∗ d d^{\ast} d results in a second-step model whose target distribution q ϕ ∗ Z q^{Z}_{\phi^{\ast}} is still supported on a manifold f ϕ ∗ ​ ( ℳ ) f_{\phi^{\ast}}(\mathcal{M}) of lower-than-ambient dimension, which the diffusion model from the second step can still learn. Although in this case the score function would also diverge to infinity, the fact that the difference between the ambient and intrinsic dimensions for the latent model ( d − d ∗ d-d^{\ast} ) remains much smaller than for a model on ambient space ( D − d ∗ D-d^{\ast} ) intuitively suggests that the numerical issues should nonetheless be diminished for latent diffusion models. Indeed, despite the latent diffusion model shown in Figure 8 using d = 256 d=256 , which is likely larger than the true intrinsic dimension d ∗ d^{\ast} of CIFAR-10 ( Pope et al., 2021 ) , it has a numerically much better behaved score function than the diffusion model on ambient space.

 
 
 

#### Section 5.3.3 Injective Normalizing Flows

 
 Ordinary normalizing flows ( Section 4.1.3 ) are full-dimensional density models with D D -dimensional latent spaces, conflicting with the d ∗ d^{\ast} -dimensional nature of p ∗ X {p}^{X}_{\ast} . To correct this mismatch, a line of research started by Kumar et al. (2020) has proposed to shrink the NF’s latent space dimensionality to d D d D , allowing the model to represent densities on a low-dimensional submanifold of 𝒳 \mathcal{X} . Whereas standard NF architectures must be bijective, g θ 1 g_{\theta_{1}} now cannot be bijective because it maps from d d to D D dimensions. The most one can ask is that g θ 1 g_{\theta_{1}} be injective , resulting in the injective normalizing flow (INF).

 
 
 Brehmer Cranmer (2020) enforce injectivity architecturally, by constructing g θ 1 g_{\theta_{1}} as a zero-padding operation followed by a D D -dimensional NF, in which case the left inverse f θ 1 f_{\theta_{1}} is given by inverting this NF and applying a projection operation; this is the same construction as the one used by denoising NFs ( Section 5.1.4 ). 28 28 
 28 
 
 
 
 Like in standard NFs, here the encoder f θ 1 f_{\theta_{1}} is determined by the decoder g θ 1 g_{\theta_{1}} , and it is thus parameterized by the same parameters (i.e. θ 1 \theta_{1} ), so no auxiliary parameters ϕ \phi are needed to parameterize it. Injectivity enables density evaluation through the injective change-of-variables formula ( Equation 9 ),

 

 
 | 
 p θ X ​ ( x ) = p θ 2 Z ​ ( z ) ​ | det ( ∇ z g θ 1 ​ ( z ) ⊤ ​ ∇ z g θ 1 ​ ( z ) ) | − 1 2 , {p}^{X}_{\theta}(x)={p}^{Z}_{\theta_{2}}(z)\left|\det\left(\nabla_{z}g_{\theta_{1}}(z)^{\top}\nabla_{z}g_{\theta_{1}}(z)\right)\right|^{-\frac{1}{2}}, | 
 | 
 (95) | 
 

 where z = f θ 1 ​ ( x ) z=f_{\theta_{1}}(x) .
Unlike standard NFs, INFs cannot be naïvely trained through maximum-likelihood for two main reasons: ( i ) (i) the involved determinant is much more computationally challenging to compute and optimize than in standard NFs, and more importantly ( i ​ i ) (ii) Brehmer Cranmer (2020) showed that doing so would result in pathological solutions.
To understand why, recall that Equation 95 is only valid when x ∈ g θ 1 ​ ( 𝒵 ) x\in g_{\theta_{1}}(\mathcal{Z}) . When x x lies outside g θ 1 ​ ( 𝒵 ) g_{\theta_{1}}(\mathcal{Z}) , the right hand side of Equation 95 evaluates to p θ X ​ ( x ^ θ 1 ​ ( x ) ) {p}^{X}_{\theta}(\hat{x}_{\theta_{1}}(x)) , where x ^ θ 1 ​ ( x ) ≔ g θ 1 ​ ( f θ 1 ​ ( x ) ) \hat{x}_{\theta_{1}}(x)\coloneqq g_{\theta_{1}}(f_{\theta_{1}}(x)) can be thought of as a projection of x x onto g θ 1 ​ ( 𝒵 ) g_{\theta_{1}}(\mathcal{Z}) . Since g θ 1 ​ ( 𝒵 ) g_{\theta_{1}}(\mathcal{Z}) need not perfectly match the data manifold ℳ \mathcal{M} , attempting to maximize 𝔼 X ∼ p ∗ X ​ [ log ⁡ p θ 2 Z ​ ( f θ 1 ​ ( X ) ) − 1 2 ​ log ⁡ | det ( ∇ z g θ 1 ​ ( f θ 1 ​ ( X ) ) ⊤ ​ ∇ z g θ 1 ​ ( f θ 1 ​ ( X ) ) ) | ] \mathbb{E}_{X\sim{p}^{X}_{\ast}}[\log{p}^{Z}_{\theta_{2}}(f_{\theta_{1}}(X))-\tfrac{1}{2}\log|\det(\nabla_{z}g_{\theta_{1}}(f_{\theta_{1}}(X))^{\top}\nabla_{z}g_{\theta_{1}}(f_{\theta_{1}}(X)))|] over θ \theta would thus result in maximizing the likelihood of projected data. As illustrated in Figure 9 , this objective can admit pathological solutions where the projections collapse onto a single point whose likelihood is sent to infinity.

 
 
 Figure 9: Pathology of naïvely maximizing Equation 95 . In this case, the true density is a standard Gaussian along ℳ = { ( 0 , x 2 ) ∈ ℝ 2 ∣ x 2 ∈ ℝ } \mathcal{M}=\{(0,x_{2})\in\mathbb{R}^{2}\mid x_{2}\in\mathbb{R}\} , here shown in blue with darker values indicating higher density. The model can maximize likelihood by aligning g θ 1 ​ ( 𝒵 ) g_{\theta_{1}}(\mathcal{Z}) perpendicular to ℳ \mathcal{M} , and then learning a density along g θ 1 ​ ( 𝒵 ) g_{\theta_{1}}(\mathcal{Z}) that becomes infinitely peaked at the projection x ^ θ 1 ​ ( x ) \hat{x}_{\theta_{1}}(x) onto ℳ \mathcal{M} ; in this example the projection will always lie at the origin for any x ∈ ℝ 2 x\in\mathbb{R}^{2} . In the figure, we show g θ 1 ​ ( 𝒵 ) = { ( x 1 , 0 ) ∈ ℝ 2 ∣ x 1 ∈ ℝ } g_{\theta_{1}}(\mathcal{Z})=\{(x_{1},0)\in\mathbb{R}^{2}\mid x_{1}\in\mathbb{R}\} with the black line, the projection with the red dot, and increasing peakedness of p θ X {p}^{X}_{\theta} with increasing opacity. 
 
 
 To circumvent this issue, Brehmer Cranmer (2020) propose to train INFs as two-step models ( Section 5.3 ), where g θ 1 g_{\theta_{1}} and f θ 1 f_{\theta_{1}} are trained to minimize an ℓ 2 \ell_{2} reconstruction error as in Equation 83 ; this training procedure also obviates the need to optimize through the determinant in Equation 95 . Brehmer Cranmer (2020) then instantiate p θ 2 Z {p}^{Z}_{\theta_{2}} as a d d -dimensional NF, which is trained on encoded data f θ 1 ∗ ​ ( X ) f_{\theta_{1}^{\ast}}(X) , where X ∼ p ∗ X X\sim{p}^{X}_{\ast} . Kothari et al. (2021) follow up on this work by proposing a more efficient architecture for g θ 1 g_{\theta_{1}} . Kumar et al. (2020) originally proposed relaxed INFs, for which the encoder f ϕ f_{\phi} is parameterized separately (and encouraged to invert the decoder on ℳ \mathcal{M} through a reconstruction error), and where injectivity is instead encouraged by regularizing the singular values of ∇ z g θ 1 ​ ( f ϕ ​ ( X ) ) \nabla_{z}g_{\theta_{1}}(f_{\phi}(X)) for X ∼ p ∗ X X\sim{p}^{X}_{\ast} . They then take the second-step density p θ 2 Z {p}^{Z}_{\theta_{2}} as a Gaussian mixture model. By virtue of being two-step models, all these DGMs are manifold-aware.

 
 
 End-to-end training 

 
 As mentioned in Section 5.3 , it is intuitively desirable to find an end-to-end objective to train two-step models, and INFs provide particularly interesting opportunities. To circumvent the pathological behaviour of naïve maximum-likelihood training of INFs outlined above, several works encourage perfect reconstructions with the goal of ensuring that the model manifold matches the true data manifold:

 

 
 | 
 max θ ⁡ 𝔼 X ∼ p ∗ X ​ [ log ⁡ p θ 2 Z ​ ( f θ 1 ​ ( X ) ) − 1 2 ​ log ​ | det ( ∇ z g θ 1 ​ ( f θ 1 ​ ( X ) ) ⊤ ​ ∇ z g θ 1 ​ ( f θ 1 ​ ( X ) ) ) | ] subject to  ​ 𝔼 X ∼ p ∗ X ​ [ ‖ X − g θ 1 ​ ( f θ 1 ​ ( X ) ) ‖ 2 2 ] = 0 . \displaystyle\begin{split} \max_{\theta}\mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[\log{p}^{Z}_{\theta_{2}}(f_{\theta_{1}}(X))-\tfrac{1}{2}\log\left|\det\left(\nabla_{z}g_{\theta_{1}}(f_{\theta_{1}}(X))^{\top}\nabla_{z}g_{\theta_{1}}(f_{\theta_{1}}(X))\right)\right|\right]\\
 \text{subject to }\mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[\|X-g_{\theta_{1}}(f_{\theta_{1}}(X))\|_{2}^{2}\right]=0.\end{split} | 
 | 
 (96) | 
 

 In practice, the constraint can be encouraged by adding an ℓ 2 \ell_{2} regularization term to the likelihood,

 

 
 | 
 max θ ⁡ 𝔼 X ∼ p ∗ X ​ [ log ⁡ p θ 2 Z ​ ( f θ 1 ​ ( X ) ) − 1 2 ​ log ​ | det ( ∇ z g θ ​ ( f θ 1 ​ ( X ) ) ⊤ ​ ∇ z g θ 1 ​ ( f θ 1 ​ ( X ) ) ) | − β ​ ‖ X − g θ 1 ​ ( f θ 1 ​ ( X ) ) ‖ 2 2 ] , \max_{\theta}\mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[\log{p}^{Z}_{\theta_{2}}(f_{\theta_{1}}(X))-\tfrac{1}{2}\log\left|\det\left(\nabla_{z}g_{\theta}(f_{\theta_{1}}(X))^{\top}\nabla_{z}g_{\theta_{1}}(f_{\theta_{1}}(X))\right)\right|-\beta\|X-g_{\theta_{1}}(f_{\theta_{1}}(X))\|_{2}^{2}\right], | 
 | 
 (97) | 
 

 where β 0 \beta 0 is a hyperparameter. Much like how normalizing flows focus on tractability of the log-det-Jacobian term, a central theme of end-to-end injective flows is tractability of the second term in Equation 97 , which we will refer to as the “log-det- J ⊤ ​ J J^{\top}J ” term.
One approach is to impose structural constraints on g θ 1 g_{\theta_{1}} to make the log-det- J ⊤ ​ J J^{\top}J term easily computable, albeit at the cost of expressiveness, for example using conformal embeddings ( Ross Cresswell, 2021 ) .
A concurrent approach, pursued by Caterini et al. (2021b) , is to approximate the gradient of the log-det- J ⊤ ​ J J^{\top}J term with respect to θ 1 \theta_{1} through a combination of Hutchinson’s estimator ( Hutchinson, 1989 ) and various tricks from linear algebra and automatic differentiation ( Baydin et al., 2018 ) .
Despite these approximations, tractability remains an issue with this technique and it struggles to scale to datasets of higher ambient dimensionality than CIFAR-10 ( Krizhevsky Hinton, 2009 ) .

 
 
 Denoising NFs ( Section 5.1.4 ) perform single-step training of a flow with an injective component for generation, but with a full-dimensional model p θ X σ p^{X_{\sigma}}_{\theta} of a noised-out data density p ∗ X σ ≔ p ∗ X ⊛ 𝒩 ⁡ ( ⋅ , 0 , σ 2 ​ I D ) p_{\ast}^{X_{\sigma}}\coloneqq{p}^{X}_{\ast}\circledast\mathcal{N}(\,\cdot\,;0,\sigma^{2}I_{D}) .
The objective for denoising NFs ( Equation 63 ) ends up quite similar to Equation 97 , although with two important differences: ( i ) (i) the expectation is over p ∗ X σ p_{\ast}^{X_{\sigma}} rather than p ∗ X {p}^{X}_{\ast} , and ( i ​ i ) (ii) the formulation of p θ X σ p^{X_{\sigma}}_{\theta} as a model for p ∗ X σ p_{\ast}^{X_{\sigma}} eliminates the need to optimize over the costly log-det- J ⊤ ​ J J^{\top}J term.
However, the computational benefit comes at the cost of introducing a disconnect between the injective generator and the full-dimensional density model. In particular, the numerical instabilities described in Section 4.1.1 do not apply to INFs trained through Equation 97 since no full-dimensional densities are involved, whereas they do apply denoising NFs.

 
 
 Cunningham et al. (2022) and Flouris Konukoglu (2023) both propose injective variants of the flow models described in Section 5.2.5 using similar objectives to Caterini et al. (2021b) . Cunningham et al. (2022) in particular use a regularizer that, with a certain hyperparameter setting, cancels out the log-det- J ⊤ ​ J J^{\top}J term in the likelihood, making likelihood-based optimization of injective flows more efficient.

 
 
 Meanwhile, several works parameterize the encoder f ϕ f_{\phi} separately, as Kumar et al. (2020) did.
 Zhang et al. (2020b) proposed a similar objective to Equation 97 in the context of variational autoencoders ( Section 4.1.2 ). Sorrenson et al. (2024b) argue that the objective in Equation 97 is subject to similar pathologies to those outlined by Brehmer Cranmer (2020) for naïve maximum-likelihood training if the encoder and decoder are flexible enough.
More specifically, they contend that Equation 97 can be made pathologically large by learning a manifold with arbitrarily large curvature rather than ℳ \mathcal{M} .
This is interesting as it highlights that despite being directly motivated to account for the low-dimensional structure of the data, INFs trained through Equation 97 can nonetheless still fail to learn manifolds. Sorrenson et al. (2024b) propose an intuitively well-motivated but theoretically ad-hoc modification to the objective, along with an improved gradient estimator over that of Caterini et al. (2021b) , for end-to-end training of INFs. Their model no longer falls under the umbrella of two-step models, and hence its ability to learn the manifold is unclear, but it does inherit computational benefits as it circumvents calculation of the challenging log-det- J ⊤ ​ J J^{\top}J term ( Brehmer Cranmer, 2020 ) .
Finally, Nazari et al. (2023) proposed an autoencoder for manifold learning (not generative modelling) which penalizes the variance of the log-det- J ⊤ ​ J J^{\top}J term during training, and which results in a smoother and more interpretable latent space as compared to standard autoencoders.

 
 
 
 
 

### Section 5.4 Overcoming Topological Obstacles to Manifold Learning

 
 As mentioned in Section 5.3 , one might want to set the latent dimension d d of an autoencoder to d ∗ d^{\ast} .
Yet, as discussed in Section 3.2 , when d = d ∗ d=d^{\ast} , perfect manifold learning is not always achievable through bottleneck methods for topological reasons. We illustrate this problem, which can cause downstream issues with density estimation, in Figure 10 . In particular, if ℳ \mathcal{M} has any non-trivial topological properties such as holes or disconnected components, any attempt to model ℳ \mathcal{M} as the image of a decoder g θ g_{\theta} will cause numerical instability in g θ g_{\theta} ( Cornish et al., 2020 ; Salmona et al., 2022 ) .
This problem was originally identified in the context of normalizing flows ( Section 4.1.3 ), with a line of work that first appends additional dimensions to 𝒳 \mathcal{X} , and then trains a normalizing flow on the augmented space ( Dupont et al., 2019 ; Chen et al., 2020 ; Huang et al., 2020 ) . While these techniques indeed increase the stability and expressiveness of the flow itself, they still produce full-support densities that can never truly model non-trivial topological structures in data.
Other works have leveraged tools from the field of topological data analysis ( Chazal Michel, 2021 ; Barannikov et al., 2022 ) to explicitly regularize autoencoders with the goal of encouraging f ϕ ​ ( ℳ ) f_{\phi}(\mathcal{M}) to share topological properties with ℳ \mathcal{M} ( Moor et al., 2020 ; Trofimov et al., 2023 ) . Empirically, these methods have been shown to decrease topological mismatch; yet, as argued in the grey box in Section 3.2 , completely eliminating this mismatch is theoretically impossible when the root cause of the problem is the non-existence of a topological embedding of ℳ \mathcal{M} into 𝒵 \mathcal{Z} , rather than the loss used to train the autoencoder.
Alternatively, to more faithfully tackle topological issues, some works have proposed to restructure the manifold-learning step to better reflect the ways manifolds are defined in theory; we cover these approaches in detail below.

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 Figure 10: Models for ℳ = { ( x 1 , x 2 ) ∣ x 1 2 + x 2 2 = 1 } \mathcal{M}=\{(x_{1},x_{2})\mid x_{1}^{2}+x_{2}^{2}=1\} , the unit circle in 𝒳 = ℝ 2 \mathcal{X}=\mathbb{R}^{2} . (a) Using a single encoder-decoder pair to model the circle. The pair approaches numerical non-invertibility since the encoder f ϕ f_{\phi} must map two nearby points in ℳ \mathcal{M} (in black) to two distant latent points in 𝒵 \mathcal{Z} (also in black).
 (b) Illustration of implicit manifolds.
Here, ℳ \mathcal{M} is characterized as the 0 0 -level set of F : ℝ 2 → ℝ F:\mathbb{R}^{2}\rightarrow\mathbb{R} . Specifically, F ⁡ ( x 1 , x 2 ) = 1 − ( x 1 2 + x 2 2 ) F(x_{1},x_{2})=1-(x_{1}^{2}+x_{2}^{2}) is shown in red, and the grey plane corresponds to { x ∈ ℝ 3 ∣ x 3 = 0 } \{x\in\mathbb{R}^{3}\mid x_{3}=0\} . Neural implicit manifolds use no autoencoders, and instead attempt to learn F θ F_{\theta} so that its 0 0 -level set, F θ − 1 ​ ( { 0 } ) F_{\theta}^{-1}(\{0\}) , matches ℳ \mathcal{M} .
 (c) The problem depicted in (a) can also be circumvented by employing multiple encoder-decoder pairs, each one using its own latent space and covering a different part of ℳ \mathcal{M} .
 
 
 

#### Section 5.4.1 Neural Implicit Manifolds

 
 Under some conditions, manifolds with non-trivial topologies can be defined using level sets of functions. In particular, a level set of a smooth function F : ℝ D → ℝ D − d ∗ F:\mathbb{R}^{D}\to\mathbb{R}^{D-d^{\ast}} represents a d ∗ d^{\ast} -dimensional manifold if its Jacobian has full rank on that level set ( Lee, 2012 ) .
We illustrate such an implicitly defined manifold – which cannot be characterized with a single encoder-decoder pair – in Figure 10(b) .

 
 
 Neural implicit manifold learning ( Ross et al., 2023 ) operationalizes this fact by modelling ℳ \mathcal{M} as the zero set of a neural network F θ 1 : 𝒳 → ℝ D − d F_{\theta_{1}}:\mathcal{X}\to\mathbb{R}^{D-d} . The network F θ 1 F_{\theta_{1}} is trained to align its zero set F θ 1 − 1 ​ ( { 0 } ) F_{\theta_{1}}^{-1}(\{0\}) with ℳ \mathcal{M} while being regularized to have full rank on ℳ \mathcal{M} . The following loss is used,

 

 
 | 
 min θ 1 ⁡ 𝔼 X ∼ p ∗ X , Y ∼ q θ 1 ′ X , V ∼ 𝒰 ⁡ ( ⋅ , 𝒮 D − d − 1 ) ​ [ ‖ F θ 1 ​ ( X ) ‖ 2 − α ​ ‖ F θ 1 ​ ( Y ) ‖ 2 + β ​ ( η − ‖ V ⊤ ​ ∇ x F θ 1 ​ ( X ) ‖ 2 ) + 2 ] , \min_{\theta_{1}}\mathbb{E}_{X\sim{p}^{X}_{\ast},Y\sim q_{\theta_{1}^{\prime}}^{X},V\sim\mathcal{U}(\,\cdot\,;\mathcal{S}^{D-d-1})}\Big[\|F_{\theta_{1}}(X)\|_{2}-\alpha\|F_{\theta_{1}}(Y)\|_{2}+\beta\!\left(\eta-\|V^{\top}\!\nabla_{x}F_{\theta_{1}}(X)\|_{2}\right)_{+}^{2}\Big], | 
 | 
 (98) | 
 

 where: Y Y is independent of X X ; q θ 1 ′ X ​ ( y ) ∝ e − ‖ F θ 1 ′ ​ ( y ) ‖ 2 2 q^{X}_{\theta_{1}^{\prime}}(y)\propto e^{-\|F_{\theta_{1}^{\prime}}(y)\|_{2}^{2}} with θ 1 ′ = 𝚜𝚝𝚘𝚙𝚐𝚛𝚊𝚍 ⁡ ( θ 1 ) \theta_{1}^{\prime}=\mathtt{stopgrad}(\theta_{1}) (as in energy-based models, see Section 4.1.4 );
 𝒰 ⁡ ( ⋅ , 𝒮 D − d − 1 ) \mathcal{U}(\,\cdot\,;\mathcal{S}^{D-d-1}) is the uniform distribution over 𝒮 D − d − 1 ≔ { y ∈ ℝ D − d ∣ ‖ y ‖ 2 = 1 } \mathcal{S}^{D-d-1}\coloneqq\{y\in\mathbb{R}^{D-d}\mid\|y\|_{2}=1\} , the ( D − d − 1 ) (D{-}d{-}1) -sphere in ℝ D − d \mathbb{R}^{D-d} ; α 0 \alpha 0 , β 0 \beta 0 , and η 0 \eta 0 are hyperparameters; and ( ⋅ ) + ≔ max ⁡ ( ⋅ , 0 ) (\cdot)_{+}\coloneqq\max(\,\cdot\,,0) . In this loss, the first term ensures that F θ 1 − 1 ​ ( { 0 } ) F_{\theta_{1}}^{-1}(\{0\}) contains ℳ \mathcal{M} , the second term prevents F θ 1 − 1 ​ ( { 0 } ) F_{\theta_{1}}^{-1}(\{0\}) from containing off-manifold samples, and the third term regularizes the Jacobian of F θ 1 F_{\theta_{1}} to have full rank on ℳ \mathcal{M} .

 
 
 Importantly, F θ 1 F_{\theta_{1}} here is not an encoder; it is best interpreted as defining D − d D-d non-linear constraints on the data, thus leaving d d degrees of freedom for the data manifold. The full-rank requirement can then be interpreted as ensuring none of these constraints is redundant, thereby ensuring the learned manifold has the correct dimensionality. This constraint-based learning procedure for ℳ \mathcal{M} makes implicit manifold learning unusual among two-step models ( Section 5.3 ) in that it is not autoencoder-based. The lack of encoder in this method means there is no latent space, which presents a challenge for learning the distribution on the model manifold, F θ 1 ∗ − 1 ​ ( { 0 } ) F_{\theta_{1}^{*}}^{-1}(\{0\}) .

 
 
 Ross et al. (2023) propose to learn this distribution with the constrained energy-based model, which represents an EBM constrained to the learned manifold, E : F θ 1 ∗ − 1 ​ ( { 0 } ) → ℝ E:F_{\theta_{1}^{*}}^{-1}(\{0\})\to\mathbb{R} . This is parameterized in practice by a neural network E θ 2 : 𝒳 → ℝ E_{\theta_{2}}:\mathcal{X}\to\mathbb{R} , for which values are ignored outside of the learned manifold F θ 1 ∗ − 1 ​ ( { 0 } ) F_{\theta_{1}^{*}}^{-1}(\{0\}) .
To sample from constrained EBMs, constrained Langevin dynamics ( Brubaker et al., 2012 ) is used to generate samples from E θ 2 E_{\theta_{2}} constrained to the manifold. This allows for likelihood maximization via Equation 38 , as with ordinary EBMs – except constrained EBMs are manifold-supported.

 
 
 

#### Section 5.4.2 Multi-Chart Manifolds

 
 Typical techniques model the manifold globally using a single encoder-decoder pair. In general, however, manifolds can consist of a patchwork of many charts : mathematical objects that each serve roughly the same function as a single encoder-decoder pair (for a formal definition, please see Lee (2012) ). Some manifolds may thus require many encoder-decoder pairs ( f ϕ ( i ) , g θ ( i ) ) i = 1 n (f_{\phi}^{(i)},g_{\theta}^{(i)})_{i=1}^{n} – each using its latent space 𝒵 i \mathcal{Z}_{i} to locally describe some subset of the manifold – rather than a single one: we illustrate this fact in Figure 10(c) .

 
 
 Several works have taken this route to learn the data manifold. Schonsheck et al. (2019) first proposed a multi-chart latent space using multiple encoder-decoder pairs in a non-generative modelling context. For each incoming datapoint, the correct chart is selected dynamically using a prediction head for the encoder-decoder pair with the smallest reconstruction error. Kalatzis et al. (2021) propose multi-chart flows, in which likelihoods are defined using a mixture of injective normalizing flows ( Section 5.3.3 ), with mixture weights again computed using a prediction head. On the other hand, Sidheekh et al. (2022) propose a mixture of INFs in which chart membership for an incoming datapoint is computed using discrete latent assignments.

 
 
 Modelling a manifold with multiple charts imposes drawbacks. For one, each encoder-decoder pair has its own latent space, so for a given datapoint x ∈ ℳ x\in\mathcal{M} , choosing the correct encoder can be a challenge. This makes it unclear how to perform tasks involving the manipulation of latent representations, such as interpolation. In many cases, there is no single correct encoder, as the images of various decoders need to overlap to correctly define topologically complex manifolds (such as in Figure 10(c) ). The ambiguity of choosing the correct encoder is underlined by how differently each of the aforementioned methods attempt to do so.

 
 
 

#### Section 5.4.3 Disconnected Manifolds

 
 Another common source of topological complexity in the data manifold is when it consists of more than one connected component. This situation occurs, for example, in datasets with multiple disjoint classes. In this context, theoretical analyses have shown that any decoder-based model will suffer from training instability ( Salmona et al., 2022 ) and poor sample quality ( Luzi et al., 2020 ) . 29 29 
 29 
 
 
 
 Note that the theoretical analysis of Salmona et al. (2022) shows numerical instability when p ∗ X {p}^{X}_{\ast} is multimodal, in which case its support ℳ \mathcal{M} can be considered as numerically disconnected. 
One technique to improve sample quality is to avoid sampling from latent regions where the network g θ ∗ g_{\theta^{*}} is unstable. Tanielian et al. (2020) propose, in the context of generative adversarial networks ( Section 4.2 and Section 5.2.1 ), to reject samples X = g θ ∗ ​ ( Z ) X=g_{\theta^{\ast}}(Z) , where Z ∼ p Z Z\sim{p}^{Z} , for which ∇ z g θ ∗ ​ ( Z ) \nabla_{z}g_{\theta^{\ast}}(Z) has a high Frobenius norm, which they show indicates an off-manifold sample. Other work, discussed below, seeks to avoid instability entirely during training.

 
 
 A few general techniques have been proposed for modelling disconnected manifolds. One way is to use disconnected (or near-disconnected) latent distributions, which aims to match the topology of the support of p θ Z {p}^{Z}_{\theta} with that of ℳ \mathcal{M} . This is typically done with a Gaussian mixture model for p θ Z {p}^{Z}_{\theta} and has been proposed for multiple classes of generative model ( Nalisnick et al., 2016 ; Dilokthanakul et al., 2016 ; Jiang et al., 2017 ; Ben-Yosef Weinshall, 2018 ; Izmailov et al., 2020 ) .

 
 
 A related approach is to use multiple decoder networks in a similar manner to the aforementioned multi-chart methods from Section 5.4.2 . The model then becomes a mixture p θ X ​ ( x ) = ∑ i = 1 n π i ​ p θ , i X ​ ( x ) {p}^{X}_{\theta}(x)=\sum_{i=1}^{n}\pi_{i}p_{\theta,i}^{X}(x) of generative submodels p θ , i X p_{\theta,i}^{X} , where π 1 , … , π n \pi_{1},\ldots,\pi_{n} are the mixture weights (sometimes trainable). For example, Arora et al. (2017) propose to directly train a mixture of generative adversarial networks to stabilize training, wherein the entire mixture is learned with backpropagation.
 Cornish et al. (2020) use a hierarchical continuously-indexed mixture of normalizing flows ( Section 4.1.3 ). Other work uses techniques based on expectation-maximization ( Dempster et al., 1977 ) to train the mixture ( Banijamali et al., 2017 ; Locatello et al., 2018 ) . A key challenge in this line of work is to encourage different submodels to model distinct parts of the distributions. This can be done by partitioning the data beforehand, by class ( Luzi et al., 2020 ) or through unsupervised clustering ( Brown et al., 2023 ) , and training a model on each partition. A more flexible approach is to backpropagate through an ancillary classification model to encourage each submodel to generate data from distinct manifolds ( Hoang et al., 2018 ; Khayatkhoei et al., 2018 ; Ghosh et al., 2018 ) .

 
 
 While all the models mentioned above can properly account for some topological features of ℳ \mathcal{M} such as disconnectedness, we highlight that most are nonetheless manifold-unaware. For example, full-dimensional models trained through maximum-likelihood remain exposed to the corresponding pathologies ( Section 4.1 ), even when they are mixture models.

 
 
 
 
 

## Section 6 Discrete Deep Generative Models

 
 Since this survey’s focus is on the manifold hypothesis, all of the models presented thus far are for continuous distributions. Nonetheless, many DGMs assume that the ambient space 𝒳 \mathcal{X} is discrete. For example, images can be modelled as having pixels which take only finitely many different values, rather than a continuum of them. In this case, p ∗ X {p}^{X}_{\ast} and p θ X {p}^{X}_{\theta} are both probability mass functions over 𝒳 \mathcal{X} , and ℳ ⊂ 𝒳 \mathcal{M}\subset\mathcal{X} denotes the support of p ∗ X {p}^{X}_{\ast} . Formally, in this setting 𝒳 \mathcal{X} is a 0 0 -dimensional manifold, so that even when ℳ \mathcal{M} is a strict subset of 𝒳 \mathcal{X} , it remains a 0 0 -dimensional submanifold. In other words, there can be no dimensionality mismatch for discrete data since D = d ∗ = 0 D=d^{\ast}=0 . In turn, this implies that mathematically, discrete likelihood-based DGMs are not exposed to problems such as manifold overfitting ( Section 4.1 ) which arise from dimensionality mismatch. This view of discrete DGMs through the manifold lens is useful, since it suggests that whenever a manifold-unaware DGM admits a straightforward discrete analogue, the latter should be preferred as it will be unaffected by manifold-related woes. Indeed, as mentioned in Section 4.1.2 , discrete variational autoencoders ( Gulrajani et al., 2017b ; Vahdat Kautz, 2020 ; Vahdat et al., 2021 ) empirically outperform their continuous variants. Similarly, discrete incarnations of likelihood-based autoregressive DGMs ( Germain et al., 2015 ; van den Oord et al., 2016 ; Salimans et al., 2017 ; Parmar et al., 2018 ) outperform continuous ones ( Uria et al., 2013 ) . In contrast, discrete versions of diffusion models ( Austin et al., 2021 ; Campbell et al., 2022 ; Meng et al., 2022 ) do not outperform their manifold-aware continuous counterparts ( Section 5.1.2 ) when modelling images.

 
 
 Discrete and continuous DGMs nonetheless have similarities, despite the differences outlined above. As discussed in Section 1 , a key motivation behind the manifold hypothesis is to capture the intuition that ℳ \mathcal{M} , the support of p ∗ X {p}^{X}_{\ast} , is somehow sparse within 𝒳 \mathcal{X} . This intuition often remains true in the discrete case: using images as an example once again, there are 256 D 256^{D} possible discrete images (assuming each pixel entry takes one of 256 256 possible values), yet the subset of natural images is vanishingly small in comparison and contains orders of magnitude fewer elements. The main idea of continuous two-step models ( Section 5.3 ), namely to first approximate the support of p ∗ X {p}^{X}_{\ast} and then learn the distribution within, remains equally sensible in the discrete case. van den Oord et al. (2017) proposed an autoencoder which recovers discrete representations over which they train a discrete DGM; this idea that has been further developed, with strong empirical results ( Razavi et al., 2019 ; Esser et al., 2021 ; Ramesh et al., 2021 ; Chang et al., 2022 ) . We finish by pointing out that the discussion in Section 5.3.1 applies to all these discrete two-step models, so that they can be interpreted as minimizing a potentially regularized upper bound of the Wasserstein distance between p ∗ X {p}^{X}_{\ast} and p θ X {p}^{X}_{\theta} which becomes tight at optimality, because in the discrete case, perfect reconstructions are always achievable given enough capacity of the encoder and decoder.

 
 
 

## Section 7 Conclusions and Future Outlook

 
 Conclusions 

 
 In this survey we have carried out a review of deep generative models through the lens of the manifold hypothesis. This viewpoint presents a mathematically elegant perspective of DGMs, and suggests that manifold-awareness is an important necessary condition for strong empirical performance. We thus encourage researchers who are developing new DGMs to consider manifold-awareness as a desideratum, and ask themselves: Can my deep generative model learn distributions supported on unknown low-dimensional manifolds? When the answer is yes, demonstrating this fact will strengthen the work’s motivation; and when the answer is no, this suggests that the DGM can be improved by endowing it with manifold-awareness – either through a model-specific fix, or at least by training it on latent space as a two-step model ( Section 5.3 ). We also showed that numerical instabilities of likelihood-evaluation are unavoidable in the manifold setting ( Section 4.1.1 ) and that two-step models can be interpreted as minimizing a (potentially regularized) upper bound of the Wasserstein distance objective ( Section 5.3.1 ).

 
 
 
 Future outlook 

 
 Finally, we outline a non-exhaustive list of research directions involving deep generative models and their interplay with the manifold hypothesis. We believe these lines of inquiry are interesting, and mostly unexplored at the time of writing:

 
 • 
 
 Further understanding dimensionality mismatch  The effects of using a full-dimensional model when the ground truth distribution is manifold-supported are well understood for likelihood-based models ( Section 4.1 ) and diffusion models ( Section 5.1.2 ), yet our grasp of the interplay between DGMs and the manifold hypothesis remains incomplete. For example, a theoretical understanding of score matching ( Section 4.3 ) and conditional flow matching ( Section 5.1.3 ) under misspecified dimension is lacking, as is the effect of using lower-bounded energy functions in energy-based models ( Section 4.1.4 ).

 

 • 
 
 Improved training of DGMs with two-step architectures  Two-step models as presented in Section 5.3 are manifold-aware. Yet, as also discussed in Section 5.3 , two-step training does not encourage the encoder from the first step to represent the data in a way conducive to distribution learning in the second step. Intuitively, this means there is room for improvement in how these models are trained, and since the end-to-end approaches described at the end of Section 5.3 are in general manifold-unaware, several avenues remain open. For example, despite the existence of regularizers for training autoencoders ( Larsen et al., 2016 ; Higgins et al., 2017 ; Nazari et al., 2023 ) , there is very little work explicitly designing autoencoders for two-step training. The only work we are aware of in this direction is by Hu et al. (2023) , who propose to split the first step into two sub-steps: in the first sub-step the encoder is trained along with a low-capacity decoder, and in the second sub-step the encoder is frozen and a more flexible decoder is trained. Another avenue is finding an end-to-end objective to train this type of model in a manifold-aware fashion. Current end-to-end methods are manifold-unaware, despite providing a desirable inductive bias – an exception being generalized energy-based models ( Section 5.2.4 ) which cannot be readily extended beyond using energy-based models ( Section 4.1.4 ) as the latent distribution. We thus hypothesize that any end-to-end, or improved two-step, manifold-aware procedure which can train diffusion models in latent space while scaling to massive datasets ( Schuhmann et al., 2022 ) is likely to improve upon current commercial versions of latent diffusion models ( Section 5.3.2 ).

 

 • 
 
 Extracting and leveraging manifold information  Any manifold-aware DGM which succeeds at learning its target distribution p ∗ X {p}^{X}_{\ast} must have learned its support ℳ \mathcal{M} as well, albeit perhaps implicitly. Extracting information about ℳ \mathcal{M} from a trained DGM is thus a natural problem, as is leveraging this information for any practical use.
Various works have shown that trained DGMs induce Riemannian metrics over the learned manifolds ( Shao et al., 2018 ; Arvanitidis et al., 2018 ; Chadebec Allassonnière, 2022 ; Sorrenson et al., 2024a ) , which can in turn be leveraged for interpolating between datapoints and for improved sampling procedures.
Several works have also shown that DGMs can be used to estimate the intrinsic dimension of ℳ \mathcal{M} ( Tempczyk et al., 2022 ; Zheng et al., 2022 ; Horvat Pfister, 2024 ; Kamkari et al., 2024b ; Stanczuk et al., 2024 ) , and these quantities have already proven useful for unsupervised out-of-distribution detection and to identify memorized samples ( Kamkari et al., 2024a ; Ross et al., 2024 ; Humayun et al., 2024 ) .
The field of topological data analysis ( Chazal Michel, 2021 ) aims to extract topological and geometric features of ℳ \mathcal{M} – such as intrinsic dimension – from an observed dataset, conventionally without the use of DGMs. Another fruitful direction for future research will involve further exploiting DGMs for topological data analysis.

 

 • 
 
 Finite-sample convergence rates  All the analyses presented here assumed the nonparametric regime ( Section 2.2 ). As we have seen throughout our survey, this simplifying assumption enables a useful and practical understanding of DGMs through the lens of the manifold hypothesis. Yet, this assumption remains unrealistic since in practice expectations with respect to p ∗ X {p}^{X}_{\ast} cannot be computed; p ∗ X {p}^{X}_{\ast} must thus be approximated via its empirical distribution – i.e. a mixture of (equally weighted) point masses at the (finitely many) observed datapoints. Formally, the empirical distribution is supported on a 0 0 -dimensional submanifold of 𝒳 \mathcal{X} – namely, the observed dataset – so that any flexible enough and sufficiently well optimized manifold-aware DGM should simply memorize its entire training dataset. Evidently manifold-awareness remains a desirable property – statistical consistency under the manifold hypothesis is impossible without it – but the fact that state-of-the-art DGMs do not suffer from total memorization cannot be explained while assuming the nonparametric regime. Thus, understanding what drives DGMs to generalize rather than memorize remains a relevant problem. Kadkhodaie et al. (2024) study these questions for diffusion models through the lens of the inductive biases provided through the architecture of the score network. More formal explanations of generalization are provided by statistical learning theory in the form of finite-sample convergence rates.
Although these results often do not assume the manifold hypothesis, a recent line of work has, obtaining in turn much faster convergence rates which depend on intrinsic rather than ambient dimension ( Schreuder et al., 2021 ; Huang et al., 2022 ; Dahal et al., 2022 ; Tang Yang, 2023 ; Chae et al., 2023 ; Chen et al., 2023 ; Oko et al., 2023 ; Chakraborty Bartlett, 2024b ; Chakraborty Bartlett, 2024a ; Hu et al., 2024 ; Vardanyan et al., 2024 ; Tang Yang, 2024 ) . We believe that this research direction provides a challenging but highly promising avenue for a full theoretical understanding of DGMs.

 

 
 
 
 
 

## References

 
 
 Ackley et al. (1985) 
 
David H Ackley, Geoffrey E Hinton, and Terrence J Sejnowski.

 
 A learning algorithm for Boltzmann machines.

 
 Cognitive Science , 9(1):147–169, 1985.

 

 
 Albergo Vanden-Eijnden (2023) 
 
Michael S Albergo and Eric Vanden-Eijnden.

 
 Building normalizing flows with stochastic interpolants.

 
 In International Conference on Learning Representations , 2023.

 

 
 Alemi et al. (2017) 
 
Alexander A Alemi, Ian Fischer, Joshua V Dillon, and Kevin Murphy.

 
 Deep variational information bottleneck.

 
 In International Conference on Learning Representations , 2017.

 

 
 Anderson (1982) 
 
Brian DO Anderson.

 
 Reverse-time diffusion equation models.

 
 Stochastic Processes and their Applications , 12(3):313–326, 1982.

 

 
 Arbel et al. (2018) 
 
Michael Arbel, Danica J Sutherland, Mikołaj Bińkowski, and Arthur
Gretton.

 
 On gradient regularizers for MMD GANs.

 
 In Advances in Neural Information Processing Systems , 2018.

 

 
 Arbel et al. (2021) 
 
Michael Arbel, Liang Zhou, and Arthur Gretton.

 
 Generalized energy based models.

 
 In International Conference on Learning Representations , 2021.

 

 
 Arjovsky et al. (2017) 
 
Martin Arjovsky, Soumith Chintala, and Léon Bottou.

 
 Wasserstein generative adversarial networks.

 
 In International Conference on Machine Learning , 2017.

 

 
 Arora et al. (2017) 
 
Sanjeev Arora, Rong Ge, Yingyu Liang, Tengyu Ma, and Yi Zhang.

 
 Generalization and equilibrium in generative adversarial nets
(GANs).

 
 In International Conference on Machine Learning , 2017.

 

 
 Arvanitidis et al. (2018) 
 
Georgios Arvanitidis, Lars Kai Hansen, and Søren Hauberg.

 
 Latent space oddity: On the curvature of deep generative models.

 
 In International Conference on Learning Representations , 2018.

 

 
 Austin et al. (2021) 
 
Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van
Den Berg.

 
 Structured denoising diffusion models in discrete state-spaces.

 
 In Advances in Neural Information Processing Systems , 2021.

 

 
 Bac et al. (2021) 
 
Jonathan Bac, Evgeny M Mirkes, Alexander N Gorban, Ivan Tyukin, and Andrei
Zinovyev.

 
 Scikit-dimension: A python package for intrinsic dimension
estimation.

 
 Entropy , 23(10), 2021.

 

 
 Banijamali et al. (2017) 
 
Ershad Banijamali, Ali Ghodsi, and Pascal Poupart.

 
 Generative mixture of networks.

 
 In International Joint Conference on Neural Networks (IJCNN) ,
2017.

 

 
 Barannikov et al. (2022) 
 
Serguei Barannikov, Ilya Trofimov, Nikita Balabin, and Evgeny Burnaev.

 
 Representation topology divergence: A method for comparing neural
network representations.

 
 In International Conference on Machine Learning , 2022.

 

 
 Baydin et al. (2018) 
 
Atılım Günes Baydin, Barak A Pearlmutter, Alexey Andreyevich Radul,
and Jeffrey Mark Siskind.

 
 Automatic differentiation in machine learning: a survey.

 
 Journal of Machine Learning Research , 18:1–43,
2018.

 

 
 Beals et al. (1968) 
 
Richard Beals, David H Krantz, and Amos Tversky.

 
 Foundations of multidimensional scaling.

 
 Psychological Review , 75(2):127, 1968.

 

 
 Behrmann et al. (2021) 
 
Jens Behrmann, Paul Vicol, Kuan-Chieh Wang, Roger Grosse, and Jörn-Henrik
Jacobsen.

 
 Understanding and mitigating exploding inverses in invertible neural
networks.

 
 In International Conference on Artificial Intelligence and
Statistics , 2021.

 

 
 Ben-Hamu et al. (2022) 
 
Heli Ben-Hamu, Samuel Cohen, Joey Bose, Brandon Amos, Maximillian Nickel,
Aditya Grover, Ricky TQ Chen, and Yaron Lipman.

 
 Matching normalizing flows and probability paths on manifolds.

 
 In International Conference on Machine Learning , 2022.

 

 
 Ben-Yosef Weinshall (2018) 
 
Matan Ben-Yosef and Daphna Weinshall.

 
 Gaussian mixture generative adversarial networks for diverse
datasets, and the unsupervised clustering of images.

 
 arXiv:1808.10356 , 2018.

 

 
 Bengio et al. (2013) 
 
Yoshua Bengio, Aaron Courville, and Pascal Vincent.

 
 Representation learning: A review and new perspectives.

 
 IEEE Transactions on Pattern Analysis and Machine
Intelligence , 35(8):1798–1828, 2013.

 

 
 Berenfeld Hoffmann (2021) 
 
Clément Berenfeld and Marc Hoffmann.

 
 Density estimation on an unknown submanifold.

 
 Electronic Journal of Statistics , 15(1):2179 – 2223, 2021.

 

 
 Berenfeld et al. (2024) 
 
Clément Berenfeld, Paul Rosa, and Judith Rousseau.

 
 Estimating a density near an unknown manifold: A Bayesian
nonparametric approach.

 
 The Annals of Statistics , 2024.

 
 To Appear.

 

 
 Billingsley (2012) 
 
Patrick Billingsley.

 
 Probability and Measure .

 
 Wiley, 2012.

 

 
 Bińkowski et al. (2018) 
 
Mikołaj Bińkowski, Danica J Sutherland, Michael Arbel, and Arthur
Gretton.

 
 Demystifying MMD GANs.

 
 In International Conference on Learning Representations , 2018.

 

 
 Birrell et al. (2022) 
 
Jeremiah Birrell, Paul Dupuis, Markos A Katsoulakis, Yannis Pantazis, and Luc
Rey-Bellet.

 
 ( f , γ ) (f,\gamma) -Divergences: Interpolating between f f -divergences
and integral probability metrics.

 
 The Journal of Machine Learning Research , 23:1816–1885, 2022.

 

 
 Boehm Seljak (2022) 
 
Vanessa M Boehm and Uros Seljak.

 
 Probabilistic autoencoder.

 
 Transactions of Machine Learning Research , 2022.

 

 
 Bogachev (2007) 
 
Vladimir Igorevich Bogachev.

 
 Measure Theory , volume 2.

 
 Springer, 2007.

 

 
 Bond-Taylor et al. (2022) 
 
Sam Bond-Taylor, Adam Leach, Yang Long, and Chris G Willcocks.

 
 Deep generative modelling: A comparative review of VAEs, GANs,
normalizing flows, energy-based and autoregressive models.

 
 IEEE Transactions on Pattern Analysis and Machine
Intelligence , 44(11):7327–7347, 2022.

 

 
 Bonet et al. (2024) 
 
Clément Bonet, Lucas Drumetz, and Nicolas Courty.

 
 Sliced-Wasserstein distances and flows on Cartan-Hadamard
manifolds.

 
 arXiv:2403.06560 , 2024.

 

 
 Borji (2019) 
 
Ali Borji.

 
 Pros and cons of GAN evaluation measures.

 
 Computer Vision and Image Understanding , 179:41–65,
2019.

 

 
 Bose et al. (2020) 
 
Joey Bose, Ariella Smofsky, Renjie Liao, Prakash Panangaden, and Will Hamilton.

 
 Latent variable modelling with hyperbolic normalizing flows.

 
 In International Conference on Machine Learning , pp. 1045–1055, 2020.

 

 
 Boyda et al. (2021) 
 
Denis Boyda, Gurtej Kanwar, Sébastien Racanière, Danilo Jimenez
Rezende, Michael S Albergo, Kyle Cranmer, Daniel C Hackett, and Phiala E
Shanahan.

 
 Sampling using SU(N) gauge equivariant flows.

 
 Physical Review D , 103(7):074504, 2021.

 

 
 Brehmer Cranmer (2020) 
 
Johann Brehmer and Kyle Cranmer.

 
 Flows for simultaneous manifold learning and density estimation.

 
 In Advances in Neural Information Processing Systems , 2020.

 

 
 Brown et al. (2023) 
 
Bradley CA Brown, Anthony L Caterini, Brendan Leigh Ross, Jesse C Cresswell,
and Gabriel Loaiza-Ganem.

 
 Verifying the union of manifolds hypothesis for image data.

 
 In International Conference on Learning Representations , 2023.

 

 
 Brubaker et al. (2012) 
 
Marcus Brubaker, Mathieu Salzmann, and Raquel Urtasun.

 
 A family of MCMC methods on implicitly defined manifolds.

 
 In International Conference on Artificial Intelligence and
Statistics , 2012.

 

 
 Campbell et al. (2022) 
 
Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George
Deligiannidis, and Arnaud Doucet.

 
 A continuous time framework for discrete denoising models.

 
 In Advances in Neural Information Processing Systems , 2022.

 

 
 Caterini et al. (2021a) 
 
Anthony L Caterini, Rob Cornish, Dino Sejdinovic, and Arnaud Doucet.

 
 Variational inference with continuously-indexed normalizing flows.

 
 In Uncertainty in Artificial Intelligence , 2021a.

 

 
 Caterini et al. (2021b) 
 
Anthony L Caterini, Gabriel Loaiza-Ganem, Geoff Pleiss, and John P Cunningham.

 
 Rectangular flows for manifold learning.

 
 In Advances in Neural Information Processing Systems ,
2021b.

 

 
 Chadebec Allassonnière (2022) 
 
Clément Chadebec and Stéphanie Allassonnière.

 
 A geometric perspective on variational autoencoders.

 
 In Advances in Neural Information Processing Systems , 2022.

 

 
 Chae et al. (2023) 
 
Minwoo Chae, Dongha Kim, Yongdai Kim, and Lizhen Lin.

 
 A likelihood approach to nonparametric estimation of a singular
distribution using deep generative models.

 
 Journal of Machine Learning Research , 24(77):1–42, 2023.

 

 
 Chakraborty Bartlett (2024a) 
 
Saptarshi Chakraborty and Peter Bartlett.

 
 A statistical analysis of Wasserstein autoencoders for
intrinsically low-dimensional data.

 
 In International Conference on Learning Representations ,
2024a.

 

 
 Chakraborty Bartlett (2024b) 
 
Saptarshi Chakraborty and Peter L Bartlett.

 
 On the statistical properties of generative adversarial models for
low intrinsic data dimension.

 
 arXiv:2401.15801 , 2024b.

 

 
 Chang et al. (2022) 
 
Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman.

 
 MaskGIT: Masked generative image transformer.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition , 2022.

 

 
 Chazal Michel (2021) 
 
Frédéric Chazal and Bertrand Michel.

 
 An introduction to topological data analysis: Fundamental and
practical aspects for data scientists.

 
 Frontiers in Artificial Intelligence , 4:667963,
2021.

 

 
 Che et al. (2020) 
 
Tong Che, Ruixiang Zhang, Jascha Sohl-Dickstein, Hugo Larochelle, Liam Paull,
Yuan Cao, and Yoshua Bengio.

 
 Your GAN is secretly an energy-based model and you should use
discriminator driven latent sampling.

 
 In Advances in Neural Information Processing Systems , 2020.

 

 
 Chen et al. (2020) 
 
Jianfei Chen, Cheng Lu, Biqi Chenli, Jun Zhu, and Tian Tian.

 
 VFlow: More expressive generative flows with variational data
augmentation.

 
 In International Conference on Machine Learning , 2020.

 

 
 Chen et al. (2023) 
 
Minshuo Chen, Kaixuan Huang, Tuo Zhao, and Mengdi Wang.

 
 Score approximation, estimation and distribution recovery of
diffusion models on low-dimensional data.

 
 In International Conference on Machine Learning , 2023.

 

 
 Chen Lipman (2024) 
 
Ricky TQ Chen and Yaron Lipman.

 
 Flow matching on general geometries.

 
 In International Conference on Learning Representations , 2024.

 

 
 Chen et al. (2018) 
 
Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud.

 
 Neural ordinary differential equations.

 
 In Advances in Neural Information Processing Systems , 2018.

 

 
 Chen et al. (2017) 
 
Xi Chen, Diederik P Kingma, Tim Salimans, Yan Duan, Prafulla Dhariwal, John
Schulman, Ilya Sutskever, and Pieter Abbeel.

 
 Variational lossy autoencoder.

 
 In International Conference on Learning Representations , 2017.

 

 
 Child (2021) 
 
Rewon Child.

 
 Very deep VAEs generalize autoregressive models and can outperform
them on images.

 
 In International Conference on Learning Representations , 2021.

 

 
 Cornish et al. (2020) 
 
Rob Cornish, Anthony L Caterini, George Deligiannidis, and Arnaud Doucet.

 
 Relaxing bijectivity constraints with continuously indexed
normalising flows.

 
 In International Conference on Machine Learning , 2020.

 

 
 Cresswell et al. (2022) 
 
Jesse C Cresswell, Brendan Leigh Ross, Gabriel Loaiza-Ganem, Humberto
Reyes-Gonzalez, Marco Letizia, and Anthony L Caterini.

 
 CaloMan: Fast generation of calorimeter showers with density
estimation on learned manifolds.

 
 In NeurIPS Workshop on Machine Learning and the Physical
Sciences , 2022.

 

 
 Cunningham et al. (2022) 
 
Edmond Cunningham, Adam D Cobb, and Susmit Jha.

 
 Principal component flows.

 
 In International Conference on Machine Learning , 2022.

 

 
 Dahal et al. (2022) 
 
Biraj Dahal, Alexander Havrilla, Minshuo Chen, Tuo Zhao, and Wenjing Liao.

 
 On deep generative models for approximation and estimation of
distributions on manifolds.

 
 In Advances in Neural Information Processing Systems , 2022.

 

 
 Dai Wipf (2019) 
 
Bin Dai and David Wipf.

 
 Diagnosing and enhancing VAE models.

 
 In International Conference on Learning Representations , 2019.

 

 
 Dao et al. (2023) 
 
Quan Dao, Hao Phung, Binh Nguyen, and Anh Tran.

 
 Flow matching in latent space.

 
 arXiv:2307.08698 , 2023.

 

 
 DasGupta (2008) 
 
Anirban DasGupta.

 
 Asymptotic Theory of Statistics and Probability .

 
 Springer, 2008.

 

 
 De Bortoli (2022) 
 
Valentin De Bortoli.

 
 Convergence of denoising diffusion models under the manifold
hypothesis.

 
 Transactions on Machine Learning Research , 2022.

 

 
 De Bortoli et al. (2022) 
 
Valentin De Bortoli, Emile Mathieu, Michael Hutchinson, James Thornton,
Yee Whye Teh, and Arnaud Doucet.

 
 Riemannian score-based generative modeling.

 
 In Advances in Neural Information Processing Systems , 2022.

 

 
 Dempster et al. (1977) 
 
Arthur P Dempster, Nan M Laird, and Donald B Rubin.

 
 Maximum likelihood from incomplete data via the EM algorithm.

 
 Journal of the Royal Statistical Society: Series B
(Methodological) , 39(1):1–22, 1977.

 

 
 Dieudonné (1973) 
 
Jean Dieudonné.

 
 Treatise on Analysis , volume 3.

 
 Academic Press, 1973.

 

 
 Dilokthanakul et al. (2016) 
 
Nat Dilokthanakul, Pedro AM Mediano, Marta Garnelo, Matthew CH Lee, Hugh
Salimbeni, Kai Arulkumaran, and Murray Shanahan.

 
 Deep unsupervised clustering with Gaussian mixture variational
autoencoders.

 
 arXiv:1611.02648 , 2016.

 

 
 Dinh et al. (2015) 
 
Laurent Dinh, David Krueger, and Yoshua Bengio.

 
 NICE: Non-linear independent components estimation.

 
 In ICLR Workshop Track , 2015.

 

 
 Dinh et al. (2017) 
 
Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio.

 
 Density estimation using Real NVP.

 
 In International Conference on Learning Representations , 2017.

 

 
 Divol (2022) 
 
Vincent Divol.

 
 Measure estimation on manifolds: An optimal transport approach.

 
 Probability Theory and Related Fields , 183(1):581–647, 2022.

 

 
 Donahue et al. (2017) 
 
Jeff Donahue, Philipp Krähenbühl, and Trevor Darrell.

 
 Adversarial feature learning.

 
 In International Conference on Learning Representations , 2017.

 

 
 Draxler et al. (2024) 
 
Felix Draxler, Peter Sorrenson, Lea Zimmermann, Armand Rousselot, and Ullrich
Köthe.

 
 Free-form flows: Make any architecture a normalizing flow.

 
 In International Conference on Artificial Intelligence and
Statistics , 2024.

 

 
 Du Mordatch (2019) 
 
Yilun Du and Igor Mordatch.

 
 Implicit generation and modeling with energy based models.

 
 In Advances in Neural Information Processing Systems , 2019.

 

 
 Dupont et al. (2019) 
 
Emilien Dupont, Arnaud Doucet, and Yee Whye Teh.

 
 Augmented neural ODEs.

 
 In Advances in Neural Information Processing Systems , 2019.

 

 
 Durkan et al. (2019) 
 
Conor Durkan, Artur Bekasov, Iain Murray, and George Papamakarios.

 
 Neural spline flows.

 
 In Advances in Neural Information Processing Systems , 2019.

 

 
 Dziugaite et al. (2015) 
 
Gintare Karolina Dziugaite, Daniel M Roy, and Zoubin Ghahramani.

 
 Training generative neural networks via maximum mean discrepancy
optimization.

 
 In Uncertainty in Artificial Intelligence , 2015.

 

 
 Esser et al. (2021) 
 
Patrick Esser, Robin Rombach, and Bjorn Ommer.

 
 Taming transformers for high-resolution image synthesis.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition , 2021.

 

 
 Facco et al. (2017) 
 
Elena Facco, Maria d’Errico, Alex Rodriguez, and Alessandro Laio.

 
 Estimating the intrinsic dimension of datasets by a minimal
neighborhood information.

 
 Scientific Reports , 7(1):12140, 2017.

 

 
 Flouris Konukoglu (2023) 
 
Kyriakos Flouris and Ender Konukoglu.

 
 Canonical normalizing flows for manifold learning.

 
 In Advances in Neural Information Processing Systems , 2023.

 

 
 Gallot et al. (2004) 
 
Sylvestre Gallot, Dominique Hulin, and Jacques Lafontaine.

 
 Riemannian Geometry .

 
 Springer, 3rd edition, 2004.

 

 
 Gao Zhu (2024) 
 
Xuefeng Gao and Lingjiong Zhu.

 
 Convergence analysis for general probability flow ODEs of diffusion
models in Wasserstein distances.

 
 arXiv:2401.17958 , 2024.

 

 
 Gemici et al. (2016) 
 
Mevlana C Gemici, Danilo Jimenez Rezende, and Shakir Mohamed.

 
 Normalizing flows on Riemannian manifolds.

 
 In NeurIPS Workshop on Bayesian Deep Learning , 2016.

 

 
 Germain et al. (2015) 
 
Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle.

 
 MADE: Masked autoencoder for distribution estimation.

 
 In International Conference on Machine Learning , 2015.

 

 
 Ghojogh et al. (2023) 
 
Benyamin Ghojogh, Mark Crowley, Fakhri Karray, and Ali Ghodsi.

 
 Elements of Dimensionality Reduction and Manifold Learning .

 
 Springer Nature, 2023.

 

 
 Ghosh et al. (2018) 
 
Arnab Ghosh, Viveka Kulharia, Vinay P Namboodiri, Philip HS Torr, and Puneet K
Dokania.

 
 Multi-agent diverse generative adversarial networks.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition , 2018.

 

 
 Ghosh et al. (2020) 
 
Partha Ghosh, Mehdi SM Sajjadi, Antonio Vergari, Michael Black, and Bernhard
Schölkopf.

 
 From variational to deterministic autoencoders.

 
 In International Conference on Learning Representations , 2020.

 

 
 Goodfellow et al. (2014) 
 
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley,
Sherjil Ozair, Aaron Courville, and Yoshua Bengio.

 
 Generative adversarial nets.

 
 In Advances in Neural Information Processing Systems , 2014.

 

 
 Grathwohl et al. (2019) 
 
Will Grathwohl, Ricky TQ Chen, Jesse Bettencourt, Ilya Sutskever, and David
Duvenaud.

 
 FFJORD: Free-form continuous dynamics for scalable reversible
generative models.

 
 In International Conference on Learning Representations , 2019.

 

 
 Grathwohl et al. (2020) 
 
Will Grathwohl, Kuan-Chieh Wang, Joern-Henrik Jacobsen, David Duvenaud,
Mohammad Norouzi, and Kevin Swersky.

 
 Your classifier is secretly an energy based model and you should
treat it like one.

 
 In International Conference on Learning Representations , 2020.

 

 
 Gray (1974) 
 
Alfred Gray.

 
 The volume of a small geodesic ball of a Riemannian manifold.

 
 Michigan Mathematical Journal , 20(4):329–344, 1974.

 

 
 Gretton et al. (2006) 
 
Arthur Gretton, Karsten Borgwardt, Malte Rasch, Bernhard Schölkopf, and
Alex Smola.

 
 A kernel method for the two-sample-problem.

 
 In Advances in Neural Information Processing Systems , 2006.

 

 
 Gu et al. (2024) 
 
Hyemin Gu, Markos A Katouslakis, Luc Rey-Bellet, and Benjamin J Zhang.

 
 Combining Wasserstein-1 and Wasserstein-2 proximals: Robust
manifold learning via well-posed generative flows.

 
 arXiv:2407.11901 , 2024.

 

 
 Gulrajani et al. (2017a) 
 
Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron C
Courville.

 
 Improved training of Wasserstein GANs.

 
 In Advances in Neural Information Processing Systems ,
2017a.

 

 
 Gulrajani et al. (2017b) 
 
Ishaan Gulrajani, Kundan Kumar, Faruk Ahmed, Adrien Ali Taiga, Francesco Visin,
David Vazquez, and Aaron Courville.

 
 PixelVAE: A latent variable model for natural images.

 
 In International Conference on Learning Representations ,
2017b.

 

 
 Haussmann Pardoux (1986) 
 
UG Haussmann and E Pardoux.

 
 Time reversal of diffusions.

 
 The Annals of Probability , 14(4):1188–1205, 1986.

 

 
 He et al. (2016) 
 
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.

 
 Deep residual learning for image recognition.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition , 2016.

 

 
 Heusel et al. (2017) 
 
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp
Hochreiter.

 
 GANs trained by a two time-scale update rule converge to a local
Nash equilibrium.

 
 In Advances in Neural Information Processing Systems , 2017.

 

 
 Higgins et al. (2017) 
 
Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot,
Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner.

 
 beta-VAE: Learning basic visual concepts with a constrained
variational framework.

 
 In International Conference on Learning Representations , 2017.

 

 
 Ho et al. (2019) 
 
Jonathan Ho, Xi Chen, Aravind Srinivas, Yan Duan, and Pieter Abbeel.

 
 Flow++: Improving flow-based generative models with variational
dequantization and architecture design.

 
 In International Conference on Machine Learning , 2019.

 

 
 Ho et al. (2020) 
 
Jonathan Ho, Ajay Jain, and Pieter Abbeel.

 
 Denoising diffusion probabilistic models.

 
 In Advances in Neural Information Processing Systems , 2020.

 

 
 Hoang et al. (2018) 
 
Quan Hoang, Tu Dinh Nguyen, Trung Le, and Dinh Phung.

 
 MGAN: Training generative adversarial nets with multiple
generators.

 
 In International Conference on Learning Representations , 2018.

 

 
 Hornik (1991) 
 
Kurt Hornik.

 
 Approximation capabilities of multilayer feedforward networks.

 
 Neural Networks , 4(2):251–257, 1991.

 

 
 Horvat Pfister (2021) 
 
Christian Horvat and Jean-Pascal Pfister.

 
 Denoising normalizing flow.

 
 In Advances in Neural Information Processing Systems , 2021.

 

 
 Horvat Pfister (2023) 
 
Christian Horvat and Jean-Pascal Pfister.

 
 Density estimation on low-dimensional manifolds: An
inflation-deflation approach.

 
 Journal of Machine Learning Research , 24(61):1–37, 2023.

 

 
 Horvat Pfister (2024) 
 
Christian Horvat and Jean-Pascal Pfister.

 
 On gauge freedom, conservativity and intrinsic dimensionality
estimation in diffusion models.

 
 In International Conference on Learning Representations , 2024.

 

 
 Hu et al. (2024) 
 
Jerry Yao-Chieh Hu, Weimin Wu, Zhuoru Li, Zhao Song, and Han Liu.

 
 On statistical rates and provably efficient criteria of latent
diffusion transformers (DiTs).

 
 arXiv:2407.01079 , 2024.

 

 
 Hu et al. (2023) 
 
Tianyang Hu, Fei Chen, Haonan Wang, Jiawei Li, Wenjia Wang, Jiacheng Sun, and
Zhenguo Li.

 
 Complexity matters: Rethinking the latent space for generative
modeling.

 
 In Advances in Neural Information Processing Systems , 2023.

 

 
 Huang et al. (2020) 
 
Chin-Wei Huang, Laurent Dinh, and Aaron Courville.

 
 Augmented normalizing flows: Bridging the gap between generative
flows and latent variable models.

 
 arXiv:2002.07101 , 2020.

 

 
 Huang et al. (2022) 
 
Jian Huang, Yuling Jiao, Zhen Li, Shiao Liu, Yang Wang, and Yunfei Yang.

 
 An error analysis of generative adversarial networks for learning
distributions.

 
 Journal of Machine Learning Research , 23(1):5047–5089, 2022.

 

 
 Humayun et al. (2024) 
 
Ahmed Imtiaz Humayun, Ibtihel Amara, Candice Schumann, Golnoosh Farnadi, Negar
Rostamzadeh, and Mohammad Havaei.

 
 On the local geometry of deep generative manifolds.

 
 In ICML Workshop on Geometry-Grounded Representation Learning
and Generative Modeling , 2024.

 

 
 Hurewicz Wallman (1948) 
 
Witold Hurewicz and Henry Wallman.

 
 Dimension Theory (PMS-4) .

 
 Princeton University Press, 1948.

 

 
 Hutchinson (1989) 
 
M F Hutchinson.

 
 A stochastic estimator of the trace of the influence matrix for
Laplacian smoothing splines.

 
 Communications in Statistics - Simulation and Computation ,
18(3):1059–1076, 1989.

 

 
 Hyvärinen (2005) 
 
Aapo Hyvärinen.

 
 Estimation of non-normalized statistical models by score matching.

 
 Journal of Machine Learning Research , 6(24):695–709, 2005.

 

 
 Izmailov et al. (2020) 
 
Pavel Izmailov, Polina Kirichenko, Marc Finzi, and Andrew Gordon Wilson.

 
 Semi-supervised learning with normalizing flows.

 
 In International Conference on Machine Learning , pp. 4615–4630, 2020.

 

 
 Jayasiri Wijerathne (2020) 
 
Varuna Jayasiri and Nipun Wijerathne.

 
 Annotated paper implementations, 2020.

 
 URL https://nn.labml.ai/ .

 

 
 Jiang et al. (2017) 
 
Zhuxi Jiang, Yin Zheng, Huachun Tan, Bangsheng Tang, and Hanning Zhou.

 
 Variational deep embedding: An unsupervised and generative approach
to clustering.

 
 In Proceedings of the Twenty-Sixth International Joint
Conference on Artificial Intelligence (IJCAI) , 2017.

 

 
 Johnsson et al. (2014) 
 
Kerstin Johnsson, Charlotte Soneson, and Magnus Fontes.

 
 Low bias local intrinsic dimension estimation from expected simplex
skewness.

 
 IEEE Transactions on Pattern Analysis and Machine
Intelligence , 37(1):196–202, 2014.

 

 
 Kadkhodaie Simoncelli (2021) 
 
Zahra Kadkhodaie and Eero P Simoncelli.

 
 Stochastic solutions for linear inverse problems using the prior
implicit in a denoiser.

 
 In Advances in Neural Information Processing Systems , 2021.

 

 
 Kadkhodaie et al. (2024) 
 
Zahra Kadkhodaie, Florentin Guth, Eero P Simoncelli, and Stéphane Mallat.

 
 Generalization in diffusion models arises from geometry-adaptive
harmonic representations.

 
 In International Conference on Learning Representations , 2024.

 

 
 Kalatzis et al. (2021) 
 
Dimitris Kalatzis, Johan Ziruo Ye, Alison Pouplin, Jesper Wohlert, and Søren
Hauberg.

 
 Density estimation on smooth manifolds with normalizing flows.

 
 arXiv:2106.03500 , 2021.

 

 
 Kamkari et al. (2024a) 
 
Hamidreza Kamkari, Brendan Leigh Ross, Jesse C Cresswell, Anthony L Caterini,
Rahul G Krishnan, and Gabriel Loaiza-Ganem.

 
 A geometric explanation of the likelihood OOD detection paradox.

 
 In International Conference on Machine Learning ,
2024a.

 

 
 Kamkari et al. (2024b) 
 
Hamidreza Kamkari, Brendan Leigh Ross, Rasa Hosseinzadeh, Jesse C Cresswell,
and Gabriel Loaiza-Ganem.

 
 A geometric view of data complexity: Efficient local intrinsic
dimension estimation with diffusion models.

 
 In ICML Workshop on Structured Probabilistic Inference and
Generative Modeling , 2024b.

 

 
 Kanwar et al. (2020) 
 
Gurtej Kanwar, Michael S Albergo, Denis Boyda, Kyle Cranmer, Daniel C Hackett,
Sébastien Racaniere, Danilo Jimenez Rezende, and Phiala E Shanahan.

 
 Equivariant flow-based sampling for lattice gauge theory.

 
 Physical Review Letters , 125(12):121601,
2020.

 

 
 Kapusniak et al. (2024) 
 
Kacper Kapusniak, Peter Potaptchik, Teodora Reu, Leo Zhang, Alexander Tong,
Michael Bronstein, Avishek Joey Bose, and Francesco Di Giovanni.

 
 Metric flow matching for smooth interpolations on the data manifold.

 
 arXiv:2405.14780 , 2024.

 

 
 Karras et al. (2018) 
 
Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen.

 
 Progressive growing of GANs for improved quality, stability, and
variation.

 
 In International Conference on Learning Representations , 2018.

 

 
 Karras et al. (2019) 
 
Tero Karras, Samuli Laine, and Timo Aila.

 
 A style-based generator architecture for generative adversarial
networks.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition , 2019.

 

 
 Karras et al. (2020) 
 
Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and
Timo Aila.

 
 Analyzing and improving the image quality of StyleGAN.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition , 2020.

 

 
 Katsman et al. (2021) 
 
Isay Katsman, Aaron Lou, Derek Lim, Qingxuan Jiang, Ser Nam Lim, and
Christopher M De Sa.

 
 Equivariant manifold flows.

 
 In Advances in Neural Information Processing Systems , 2021.

 

 
 Khalil (2002) 
 
Hassan K Khalil.

 
 Nonlinear Systems .

 
 Prentice Hall, 2002.

 

 
 Khayatkhoei et al. (2018) 
 
Mahyar Khayatkhoei, Maneesh K Singh, and Ahmed Elgammal.

 
 Disconnected manifold learning for generative adversarial networks.

 
 In Advances in Neural Information Processing Systems , 2018.

 

 
 Kim et al. (2022) 
 
Dongjun Kim, Seungjae Shin, Kyungwoo Song, Wanmo Kang, and Il-Chul Moon.

 
 Soft truncation: A universal training technique of score-based
diffusion model for high precision score estimation.

 
 In International Conference on Machine Learning , 2022.

 

 
 Kim et al. (2020) 
 
Hyeongju Kim, Hyeonseung Lee, Woo Hyun Kang, Joun Yeop Lee, and Nam Soo Kim.

 
 Softflow: Probabilistic framework for normalizing flow on manifolds.

 
 In Advances in Neural Information Processing Systems , 2020.

 

 
 Kingma Ba (2015) 
 
Diederik P Kingma and Jimmy Ba.

 
 Adam: A method for stochastic optimization.

 
 In International Conference on Learning Representations , 2015.

 

 
 Kingma Dhariwal (2018) 
 
Diederik P Kingma and Prafulla Dhariwal.

 
 Glow: Generative flow with invertible 1x1 convolutions.

 
 In Advances in Neural Information Processing Systems , 2018.

 

 
 Kingma Gao (2023) 
 
Diederik P Kingma and Ruiqi Gao.

 
 Understanding diffusion objectives as the ELBO with simple data
augmentation.

 
 In Advances in Neural Information Processing Systems , 2023.

 

 
 Kingma Welling (2014) 
 
Diederik P Kingma and Max Welling.

 
 Auto-encoding variational Bayes.

 
 In International Conference on Learning Representations , 2014.

 

 
 Kingma et al. (2016) 
 
Diederik P Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, and
Max Welling.

 
 Improved variational inference with inverse autoregressive flow.

 
 In Advances in Neural Information Processing Systems , 2016.

 

 
 Kobyzev et al. (2020) 
 
Ivan Kobyzev, Simon JD Prince, and Marcus A Brubaker.

 
 Normalizing flows: An introduction and review of current methods.

 
 IEEE Transactions on Pattern Analysis and Machine
Intelligence , 43(11):3964–3979, 2020.

 

 
 Koehler et al. (2021) 
 
Frederic Koehler, Viraj Mehta, and Andrej Risteski.

 
 Representational aspects of depth and conditioning in normalizing
flows.

 
 In International Conference on Machine Learning , 2021.

 

 
 Koehler et al. (2022) 
 
Frederic Koehler, Viraj Mehta, Chenghui Zhou, and Andrej Risteski.

 
 Variational autoencoders in the presence of low-dimensional data:
Landscape and implicit bias.

 
 In International Conference on Learning Representations , 2022.

 

 
 Kolouri et al. (2018) 
 
Soheil Kolouri, Phillip E Pope, Charles E Martin, and Gustavo K Rohde.

 
 Sliced Wasserstein auto-encoders.

 
 In International Conference on Learning Representations , 2018.

 

 
 Kothari et al. (2021) 
 
Konik Kothari, AmirEhsan Khorashadizadeh, Maarten de Hoop, and Ivan Dokmanić.

 
 Trumpets: Injective flows for inference and inverse problems.

 
 In Uncertainty in Artificial Intelligence , 2021.

 

 
 Köthe (2023) 
 
Ullrich Köthe.

 
 A review of change of variable formulas for generative modeling.

 
 arXiv:2308.02652 , 2023.

 

 
 Kramer (1991) 
 
Mark A Kramer.

 
 Nonlinear principal component analysis using autoassociative neural
networks.

 
 AIChE Journal , 37(2):233–243, 1991.

 

 
 Krizhevsky Hinton (2009) 
 
Alex Krizhevsky and Geoffrey Hinton.

 
 Learning multiple layers of features from tiny images.

 
 Technical report, University of Toronto, 2009.

 

 
 Kruskal (1964) 
 
Joseph B Kruskal.

 
 Multidimensional scaling by optimizing goodness of fit to a nonmetric
hypothesis.

 
 Psychometrika , 29(1):1–27, 1964.

 

 
 Kumar et al. (2020) 
 
Abhishek Kumar, Ben Poole, and Kevin Murphy.

 
 Regularized autoencoders via relaxed injective probability flow.

 
 In International Conference on Artificial Intelligence and
Statistics , 2020.

 

 
 Kwon et al. (2022) 
 
Dohyun Kwon, Ying Fan, and Kangwook Lee.

 
 Score-based generative modeling secretly minimizes the Wasserstein
distance.

 
 In Advances in Neural Information Processing Systems , 2022.

 

 
 Larsen et al. (2016) 
 
Anders Boesen Lindbo Larsen, Søren Kaae Sønderby, Hugo Larochelle, and
Ole Winther.

 
 Autoencoding beyond pixels using a learned similarity metric.

 
 In International Conference on Machine Learning , 2016.

 

 
 Lee et al. (2024) 
 
Hyunjong Lee, Yedarm Seong, Sungdong Lee, and Joong-Ho Won.

 
 StrWAEs to invariant representations.

 
 In International Conference on Machine Learning , 2024.

 

 
 Lee (2012) 
 
John M Lee.

 
 Introduction to Smooth Manifolds .

 
 Springer, 2nd edition, 2012.

 

 
 Lee (2018) 
 
John M Lee.

 
 Introduction to Riemannian Manifolds .

 
 Springer, 2nd edition, 2018.

 

 
 Lehmann Casella (2006) 
 
Erich L Lehmann and George Casella.

 
 Theory of Point Estimation .

 
 Springer Science Business Media, 2006.

 

 
 Levina Bickel (2004) 
 
Elizaveta Levina and Peter Bickel.

 
 Maximum likelihood estimation of intrinsic dimension.

 
 In Advances in Neural Information Processing Systems , 2004.

 

 
 Li et al. (2017) 
 
Chun-Liang Li, Wei-Cheng Chang, Yu Cheng, Yiming Yang, and Barnabás
Póczos.

 
 MMD GAN: Towards deeper understanding of moment matching network.

 
 In Advances in Neural Information Processing Systems , 2017.

 

 
 Li et al. (2015) 
 
Yujia Li, Kevin Swersky, and Richard Zemel.

 
 Generative moment matching networks.

 
 In International Conference on Machine Learning , 2015.

 

 
 Lin et al. (2019) 
 
Shuyu Lin, Stephen Roberts, Niki Trigoni, and Ronald Clark.

 
 Balancing reconstruction quality and regularisation in evidence lower
bound for variational autoencoders.

 
 arXiv:1909.03765 , 2019.

 

 
 Lipman et al. (2023) 
 
Yaron Lipman, Ricky T Q Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le.

 
 Flow matching for generative modeling.

 
 In International Conference on Learning Representations , 2023.

 

 
 Liu et al. (2023) 
 
Xingchao Liu, Chengyue Gong, and Qiang Liu.

 
 Flow straight and fast: Learning to generate and transfer data with
rectified flow.

 
 In International Conference on Learning Representations , 2023.

 

 
 Loaiza-Ganem et al. (2022a) 
 
Gabriel Loaiza-Ganem, Brendan Leigh Ross, Jesse C Cresswell, and Anthony L
Caterini.

 
 Diagnosing and fixing manifold overfitting in deep generative models.

 
 Transactions on Machine Learning Research , 2022a.

 

 
 Loaiza-Ganem et al. (2022b) 
 
Gabriel Loaiza-Ganem, Brendan Leigh Ross, Luhuan Wu, John Patrick Cunningham,
Jesse C Cresswell, and Anthony L Caterini.

 
 Denoising deep generative models.

 
 In Proceedings on "I Can’t Believe It’s Not Better! -
Understanding Deep Learning Through Empirical Falsification" at NeurIPS 2022
Workshops , 2022b.

 

 
 Locatello et al. (2018) 
 
Francesco Locatello, Damien Vincent, Ilya Tolstikhin, Gunnar Rätsch,
Sylvain Gelly, and Bernhard Schölkopf.

 
 Competitive training of mixtures of independent deep generative
models.

 
 arXiv:1804.11130 , 2018.

 

 
 Lou et al. (2023) 
 
Aaron Lou, Minkai Xu, and Stefano Ermon.

 
 Scaling Riemannian diffusion models.

 
 In Advances in Neural Information Processing Systems , 2023.

 

 
 Lu et al. (2023) 
 
Yubin Lu, Zhongjian Wang, and Guillaume Bal.

 
 Mathematical analysis of singularities in the diffusion model under
the submanifold assumption.

 
 arXiv:2301.07882 , 2023.

 

 
 Luzi et al. (2020) 
 
Lorenzo Luzi, Randall Balestriero, and Richard G Baraniuk.

 
 Ensembles of generative adversarial networks for disconnected data.

 
 arXiv:2006.14600 , 2020.

 

 
 MacKay Ghahramani (2005) 
 
David JC MacKay and Zoubin Ghahramani.

 
 Comments on “Maximum likelihood estimation of intrinsic dimension’
by E. Levina and P. Bickel (2004).

 
 The Inference Group Website, Cavendish Laboratory, Cambridge
University , 2005.

 

 
 Mardia et al. (2016) 
 
Kanti V Mardia, John T Kent, and Arnab K Laha.

 
 Score matching estimators for directional distributions.

 
 arXiv:1604.08470 , 2016.

 

 
 Mathieu Nickel (2020) 
 
Emile Mathieu and Maximilian Nickel.

 
 Riemannian continuous normalizing flows.

 
 In Advances in Neural Information Processing Systems , 2020.

 

 
 McInnes et al. (2018) 
 
Leland McInnes, John Healy, Nathaniel Saul, and Lukas Großberger.

 
 UMAP: Uniform manifold approximation and projection.

 
 Journal of Open Source Software , 3(29):861, 2018.

 

 
 Meng et al. (2022) 
 
Chenlin Meng, Kristy Choi, Jiaming Song, and Stefano Ermon.

 
 Concrete score matching: Generalized score matching for discrete
data.

 
 In Advances in Neural Information Processing Systems , 2022.

 

 
 Miyato et al. (2018) 
 
Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida.

 
 Spectral normalization for generative adversarial networks.

 
 In International Conference on Learning Representations , 2018.

 

 
 Moor et al. (2020) 
 
Michael Moor, Max Horn, Bastian Rieck, and Karsten Borgwardt.

 
 Topological autoencoders.

 
 In International Conference on Machine Learning , 2020.

 

 
 Munkres (2014) 
 
James R Munkres.

 
 Topology .

 
 Pearson Education, 2014.

 

 
 Murphy (2012) 
 
Kevin P Murphy.

 
 Machine Learning: A Probabilistic Perspective .

 
 MIT Press, 2012.

 

 
 Nalisnick et al. (2016) 
 
Eric Nalisnick, Lars Hertel, and Padhraic Smyth.

 
 Approximate inference for deep latent Gaussian mixtures.

 
 In NeurIPS Workshop on Bayesian Deep Learning , 2016.

 

 
 Narayanan Mitter (2010) 
 
Hariharan Narayanan and Sanjoy Mitter.

 
 Sample complexity of testing the manifold hypothesis.

 
 In Advances in Neural Information Processing Systems , 2010.

 

 
 Narayanan Niyogi (2009) 
 
Hariharan Narayanan and Partha Niyogi.

 
 On the sample complexity of learning smooth cuts on a manifold.

 
 In Conference on Learning Theory , 2009.

 

 
 Nazari et al. (2023) 
 
Philipp Nazari, Sebastian Damrich, and Fred A Hamprecht.

 
 Geometric autoencoders - what you see is what you decode.

 
 In International Conference on Machine Learning , 2023.

 

 
 Nichol et al. (2022) 
 
Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela
Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen.

 
 GLIDE: Towards photorealistic image generation and editing with
text-guided diffusion models.

 
 In International Conference on Machine Learning , 2022.

 

 
 Nowozin et al. (2016) 
 
Sebastian Nowozin, Botond Cseke, and Ryota Tomioka.

 
 f-GAN: Training generative neural samplers using variational
divergence minimization.

 
 In Advances in Neural Information Processing Systems , 2016.

 

 
 Oko et al. (2023) 
 
Kazusato Oko, Shunta Akiyama, and Taiji Suzuki.

 
 Diffusion models are minimax optimal distribution estimators.

 
 In International Conference on Machine Learning , 2023.

 

 
 Øksendal (2003) 
 
Bernt Øksendal.

 
 Stochastic Differential Equations , pp. 65–84.

 
 Springer Science Business Media, 2003.

 

 
 Ozakin Gray (2009) 
 
Arkadas Ozakin and Alexander Gray.

 
 Submanifold density estimation.

 
 In Advances in Neural Information Processing Systems , 2009.

 

 
 Pang et al. (2020) 
 
Bo Pang, Tian Han, Erik Nijkamp, Song-Chun Zhu, and Ying Nian Wu.

 
 Learning latent space energy-based prior model.

 
 In Advances in Neural Information Processing Systems , 2020.

 

 
 Papamakarios et al. (2017) 
 
George Papamakarios, Theo Pavlakou, and Iain Murray.

 
 Masked autoregressive flow for density estimation.

 
 In Advances in Neural Information Processing Systems , 2017.

 

 
 Papamakarios et al. (2021) 
 
George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed,
and Balaji Lakshminarayanan.

 
 Normalizing flows for probabilistic modeling and inference.

 
 Journal of Machine Learning Research , 22(57):1–64, 2021.

 

 
 Parmar et al. (2018) 
 
Niki Parmar, Ashish Vaswani, Jakob Uszkoreit, Lukasz Kaiser, Noam Shazeer,
Alexander Ku, and Dustin Tran.

 
 Image transformer.

 
 In International Conference on Machine Learning , 2018.

 

 
 Paszke et al. (2019) 
 
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory
Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban
Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan
Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith
Chintala.

 
 PyTorch: An imperative style, high-performance deep learning
library.

 
 In Advances in Neural Information Processing Systems , 2019.

 

 
 Patrini et al. (2020) 
 
Giorgio Patrini, Rianne van den Berg, Patrick Forre, Marcello Carioni, Samarth
Bhargav, Max Welling, Tim Genewein, and Frank Nielsen.

 
 Sinkhorn autoencoders.

 
 In Uncertainty in Artificial Intelligence , 2020.

 

 
 Pearson (1901) 
 
Karl Pearson.

 
 LIII. On lines and planes of closest fit to systems of points in
space.

 
 The London, Edinburgh, and Dublin Philosophical Magazine and
Journal of Science , 2(11):559–572, 1901.

 

 
 Peebles Xie (2023) 
 
William Peebles and Saining Xie.

 
 Scalable diffusion models with transformers.

 
 In Proceedings of the IEEE/CVF International Conference on
Computer Vision , 2023.

 

 
 Pennec (2006) 
 
Xavier Pennec.

 
 Intrinsic statistics on Riemannian manifolds: Basic tools for
geometric measurements.

 
 Journal of Mathematical Imaging and Vision , 25(1):127–154, 2006.

 

 
 Peyré Cuturi (2019) 
 
Gabriel Peyré and Marco Cuturi.

 
 Computational optimal transport.

 
 Foundations and Trends in Machine Learning , 11(5-6):355–607, 2019.

 

 
 Pidstrigach (2022) 
 
Jakiw Pidstrigach.

 
 Score-based generative models detect manifolds.

 
 In Advances in Neural Information Processing Systems , 2022.

 

 
 Polyanskiy Wu (2022) 
 
Yury Polyanskiy and Yihong Wu.

 
 Information Theory: From Coding to Learning .

 
 Cambridge University Press, 2022.

 

 
 Pope et al. (2021) 
 
Phillip Pope, Chen Zhu, Ahmed Abdelkader, Micah Goldblum, and Tom Goldstein.

 
 The intrinsic dimension of images and its impact on learning.

 
 In International Conference on Learning Representations , 2021.

 

 
 Postels et al. (2022) 
 
Janis Postels, Martin Danelljan, Luc Van Gool, and Federico Tombari.

 
 Maniflow: Implicitly representing manifolds with normalizing flows.

 
 In International Conference on 3D Vision , 2022.

 

 
 Puthawala et al. (2022) 
 
Michael Puthawala, Matti Lassas, Ivan Dokmanic, and Maarten De Hoop.

 
 Universal joint approximation of manifolds and densities by simple
injective flows.

 
 In International Conference on Machine Learning , 2022.

 

 
 Radford et al. (2015) 
 
Alec Radford, Luke Metz, and Soumith Chintala.

 
 Unsupervised representation learning with deep convolutional
generative adversarial networks.

 
 In International Conference on Learning Representations , 2015.

 

 
 Ramesh et al. (2021) 
 
Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec
Radford, Mark Chen, and Ilya Sutskever.

 
 Zero-shot text-to-image generation.

 
 In International Conference on Machine Learning , 2021.

 

 
 Ramesh et al. (2022) 
 
Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen.

 
 Hierarchical text-conditional image generation with CLIP latents.

 
 arXiv:2204.06125 , 2022.

 

 
 Razavi et al. (2019) 
 
Ali Razavi, Aäron van den Oord, and Oriol Vinyals.

 
 Generating diverse high-fidelity images with VQ-VAE-2.

 
 In Advances in Neural Information Processing Systems , 2019.

 

 
 Rezende Mohamed (2015) 
 
Danilo Jimenez Rezende and Shakir Mohamed.

 
 Variational inference with normalizing flows.

 
 In International Conference on Machine Learning , 2015.

 

 
 Rezende et al. (2014) 
 
Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra.

 
 Stochastic backpropagation and approximate inference in deep
generative models.

 
 In International Conference on Machine Learning , 2014.

 

 
 Rezende et al. (2020) 
 
Danilo Jimenez Rezende, George Papamakarios, Sébastien Racaniere, Michael
Albergo, Gurtej Kanwar, Phiala Shanahan, and Kyle Cranmer.

 
 Normalizing flows on tori and spheres.

 
 In International Conference on Machine Learning , 2020.

 

 
 Robbins (1956) 
 
Herbert Robbins.

 
 An empirical Bayes approach to statistics.

 
 In Third Berkeley Symposium on Mathematical Statistics and
Probability , 1956.

 

 
 Robbins Monro (1951) 
 
Herbert Robbins and Sutton Monro.

 
 A stochastic approximation method.

 
 The Annals of Mathematical Statistics , 22(3):400–407, 1951.

 

 
 Rombach et al. (2022) 
 
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn
Ommer.

 
 High-resolution image synthesis with latent diffusion models.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition , 2022.

 

 
 Ronneberger et al. (2015) 
 
Olaf Ronneberger, Philipp Fischer, and Thomas Brox.

 
 U-Net: Convolutional networks for biomedical image segmentation.

 
 In Medical Image Computing and Computer-Assisted Intervention
(MICCAI) , 2015.

 

 
 Ross Cresswell (2021) 
 
Brendan Leigh Ross and Jesse C Cresswell.

 
 Tractable density estimation on learned manifolds with conformal
embedding flows.

 
 In Advances in Neural Information Processing Systems , 2021.

 

 
 Ross et al. (2023) 
 
Brendan Leigh Ross, Gabriel Loaiza-Ganem, Anthony L Caterini, and Jesse C
Cresswell.

 
 Neural implicit manifold learning for topology-aware generative
modelling.

 
 Transactions on Machine Learning Research , 2023.

 

 
 Ross et al. (2024) 
 
Brendan Leigh Ross, Hamidreza Kamkari, Zhaoyan Liu, Tongzi Wu, George Stein,
Gabriel Loaiza-Ganem, and Jesse C Cresswell.

 
 A geometric framework for understanding memorization in generative
models.

 
 In ICML Workshop on Geometry-Grounded Representation Learning
and Generative Modeling , 2024.

 

 
 Roweis Saul (2000) 
 
Sam T Roweis and Lawrence K Saul.

 
 Nonlinear dimensionality reduction by locally linear embedding.

 
 Science , 290(5500):2323–2326, 2000.

 

 
 Rozen et al. (2021) 
 
Noam Rozen, Aditya Grover, Maximilian Nickel, and Yaron Lipman.

 
 Moser flow: Divergence-based generative modeling on manifolds.

 
 In Advances in Neural Information Processing Systems , 2021.

 

 
 Ruan et al. (2021) 
 
Yangjun Ruan, Karen Ullrich, Daniel S Severo, James Townsend, Ashish Khisti,
Arnaud Doucet, Alireza Makhzani, and Chris Maddison.

 
 Improving lossless compression rates via Monte Carlo bits-back
coding.

 
 In International Conference on Machine Learning , 2021.

 

 
 Rudin (1987) 
 
Walter Rudin.

 
 Real and Complex Analysis .

 
 McGraw-Hill, Inc., 3rd edition, 1987.

 

 
 Rumelhart et al. (1988) 
 
David E Rumelhart, Geofrrey E Hinton, and Ronald J Williams.

 
 Learning Internal Representations by Error Propagation , pp. 673–695.

 
 MIT Press, 1988.

 

 
 Rybkin et al. (2021) 
 
Oleh Rybkin, Kostas Daniilidis, and Sergey Levine.

 
 Simple and effective VAE training with calibrated decoders.

 
 In International Conference on Machine Learning , 2021.

 

 
 Saharia et al. (2022) 
 
Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L
Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim
Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi.

 
 Photorealistic text-to-image diffusion models with deep language
understanding.

 
 In Advances in Neural Information Processing Systems , 2022.

 

 
 Salimans et al. (2017) 
 
Tim Salimans, Andrej Karpathy, Xi Chen, and Diederik P Kingma.

 
 PixelCNN++: Improving the PixelCNN with discretized logistic
mixture likelihood and other modifications.

 
 In International Conference on Learning Representations , 2017.

 

 
 Salman et al. (2018) 
 
Hadi Salman, Payman Yadollahpour, Tom Fletcher, and Kayhan Batmanghelich.

 
 Deep diffeomorphic normalizing flows.

 
 arXiv:1810.03256 , 2018.

 

 
 Salmona et al. (2022) 
 
Antoine Salmona, Valentin De Bortoli, Julie Delon, and Agnès Desolneux.

 
 Can push-forward generative models fit multimodal distributions?

 
 In Advances in Neural Information Processing Systems , 2022.

 

 
 Saremi Hyvärinen (2019) 
 
Saeed Saremi and Aapo Hyvärinen.

 
 Neural empirical Bayes.

 
 Journal of Machine Learning Research , 20(181):1–23, 2019.

 

 
 Sauer et al. (2023) 
 
Axel Sauer, Tero Karras, Samuli Laine, Andreas Geiger, and Timo Aila.

 
 StyleGAN-t: Unlocking the power of GANs for fast large-scale
text-to-image synthesis.

 
 In International Conference on Machine Learning , 2023.

 

 
 Schilling Kühn (2021) 
 
René L Schilling and Franziska Kühn.

 
 Counterexamples in Measure and Integration .

 
 Cambridge University Press, 2021.

 

 
 Schonsheck et al. (2019) 
 
Stefan Schonsheck, Jie Chen, and Rongjie Lai.

 
 Chart auto-encoders for manifold structured data.

 
 arXiv:1912.10094 , 2019.

 

 
 Schreuder et al. (2021) 
 
Nicolas Schreuder, Victor-Emmanuel Brunel, and Arnak Dalalyan.

 
 Statistical guarantees for generative models without domination.

 
 In Algorithmic Learning Theory , 2021.

 

 
 Schuhmann et al. (2022) 
 
Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade W Gordon, Ross
Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell
Wortsman, Patrick Schramowski, Srivatsa R Kundurthy, Katherine Crowson,
Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev.

 
 LAION-5b: An open large-scale dataset for training next generation
image-text models.

 
 In Advances in Neural Information Processing Systems , 2022.

 

 
 Schölkopf et al. (1998) 
 
Bernhard Schölkopf, Alexander Smola, and Klaus-Robert Müller.

 
 Nonlinear component analysis as a kernel eigenvalue problem.

 
 Neural Computation , 10(5):1299–1319,
1998.

 

 
 Shao et al. (2018) 
 
Hang Shao, Abhishek Kumar, and P Thomas Fletcher.

 
 The Riemannian geometry of deep generative models.

 
 In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition Workshops , 2018.

 

 
 Sidheekh et al. (2022) 
 
Sahil Sidheekh, Chris B Dock, Tushar Jain, Radu Balan, and Maneesh K Singh.

 
 VQ-Flows: Vector quantized local normalizing flows.

 
 In Uncertainty in Artificial Intelligence , 2022.

 

 
 Simon-Gabriel Schölkopf (2018) 
 
Carl-Johann Simon-Gabriel and Bernhard Schölkopf.

 
 Kernel distribution embeddings: Universal kernels, characteristic
kernels and kernel metrics on distributions.

 
 Journal of Machine Learning Research , 19(44):1–29, 2018.

 

 
 Simon-Gabriel et al. (2023) 
 
Carl-Johann Simon-Gabriel, Alessandro Barp, Bernhard Schölkopf, and Lester
Mackey.

 
 Metrizing weak convergence with maximum mean discrepancies.

 
 Journal of Machine Learning Research , 24(184):1–20, 2023.

 

 
 Sohl-Dickstein et al. (2015) 
 
Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli.

 
 Deep unsupervised learning using nonequilibrium thermodynamics.

 
 In International Conference on Machine Learning , 2015.

 

 
 Sønderby et al. (2016) 
 
Casper Kaae Sønderby, Tapani Raiko, Lars Maaløe, Søren Kaae
Sønderby, and Ole Winther.

 
 Ladder variational autoencoders.

 
 In Advances in Neural Information Processing Systems , 2016.

 

 
 Song Ermon (2019) 
 
Yang Song and Stefano Ermon.

 
 Generative modeling by estimating gradients of the data distribution.

 
 In Advances in Neural Information Processing Systems , 2019.

 

 
 Song et al. (2021a) 
 
Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon.

 
 Maximum likelihood training of score-based diffusion models.

 
 In Advances in Neural Information Processing Systems ,
2021a.

 

 
 Song et al. (2021b) 
 
Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano
Ermon, and Ben Poole.

 
 Score-based generative modeling through stochastic differential
equations.

 
 In International Conference on Learning Representations ,
2021b.

 

 
 Sorrenson et al. (2023) 
 
Peter Sorrenson, Felix Draxler, Armand Rousselot, Sander Hummerich, and Ullrich
Köthe.

 
 Learning distributions on manifolds with free-form flows.

 
 arXiv:2312.09852 , 2023.

 

 
 Sorrenson et al. (2024a) 
 
Peter Sorrenson, Daniel Behrend-Uriarte, Christoph Schnörr, and Ullrich
Köthe.

 
 Learning distances from data with normalizing flows and score
matching.

 
 arXiv:2407.09297 , 2024a.

 

 
 Sorrenson et al. (2024b) 
 
Peter Sorrenson, Felix Draxler, Armand Rousselot, Sander Hummerich, Lea
Zimmerman, and Ullrich Köthe.

 
 Lifting architectural constraints of injective flows.

 
 In International Conference on Learning Representations ,
2024b.

 

 
 Stanczuk et al. (2024) 
 
Jan Stanczuk, Georgios Batzolis, Teo Deveney, and Carola-Bibiane Schönlieb.

 
 Diffusion models encode the intrinsic dimension of data manifolds.

 
 In International Conference on Machine Learning , 2024.

 

 
 Stein et al. (2023) 
 
George Stein, Jesse C Cresswell, Rasa Hosseinzadeh, Yi Sui, Brendan Leigh Ross,
Valentin Villecroze, Zhaoyan Liu, Anthony L Caterini, J Eric T Taylor, and
Gabriel Loaiza-Ganem.

 
 Exposing flaws of generative model evaluation metrics and their
unfair treatment of diffusion models.

 
 In Advances in Neural Information Processing Systems , 2023.

 

 
 Tang Yang (2023) 
 
Rong Tang and Yun Yang.

 
 Minimax rate of distribution estimation on unknown submanifolds
under adversarial losses.

 
 The Annals of Statistics , 51(3):1282–1308, 2023.

 

 
 Tang Yang (2024) 
 
Rong Tang and Yun Yang.

 
 Adaptivity of diffusion models to manifold structures.

 
 In International Conference on Artificial Intelligence and
Statistics , 2024.

 

 
 Tanielian et al. (2020) 
 
Ugo Tanielian, Thibaut Issenhuth, Elvis Dohmatob, and Jeremie Mary.

 
 Learning disconnected manifolds: A no GAN’s land.

 
 In International Conference on Machine Learning , 2020.

 

 
 Tempczyk et al. (2022) 
 
Piotr Tempczyk, Rafał Michaluk, Lukasz Garncarek, Przemysław Spurek,
Jacek Tabor, and Adam Golinski.

 
 LIDL: Local intrinsic dimension estimation using approximate
likelihood.

 
 In International Conference on Machine Learning , 2022.

 

 
 Tenenbaum et al. (2000) 
 
Joshua B Tenenbaum, Vin de Silva, and John C Langford.

 
 A global geometric framework for nonlinear dimensionality reduction.

 
 Science , 290(5500):2319–2323, 2000.

 

 
 Theis et al. (2016) 
 
Lucas Theis, Aäron van den Oord, and Matthias Bethge.

 
 A note on the evaluation of generative models.

 
 In International Conference on Learning Representations , 2016.

 

 
 Tipping Bishop (1999) 
 
Michael E Tipping and Christopher M Bishop.

 
 Probabilistic principal component analysis.

 
 Journal of the Royal Statistical Society Series B: Statistical
Methodology , 61(3):611–622, 1999.

 

 
 Tishby et al. (2000) 
 
Naftali Tishby, Fernando C Pereira, and William Bialek.

 
 The information bottleneck method.

 
 arXiv:physics/0004057 , 2000.

 

 
 Tolstikhin et al. (2018) 
 
Ilya Tolstikhin, Olivier Bousquet, Sylvain Gelly, and Bernhard Schölkopf.

 
 Wasserstein auto-encoders.

 
 In International Conference on Learning Representations , 2018.

 

 
 Tomczak Welling (2018) 
 
Jakub Tomczak and Max Welling.

 
 VAE with a VampPrior.

 
 In International Conference on Artificial Intelligence and
Statistics , 2018.

 

 
 Tong et al. (2024) 
 
Alexander Tong, Kilian Fatras, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang,
Jarrid Rector-Brooks, Guy Wolf, and Yoshua Bengio.

 
 Improving and generalizing flow-based generative models with
minibatch optimal transport.

 
 Transactions on Machine Learning Research , 2024.

 

 
 Townsend et al. (2019) 
 
James Townsend, Thomas Bird, and David Barber.

 
 Practical lossless compression with latent variables using bits back
coding.

 
 In International Conference on Learning Representations , 2019.

 

 
 Tran et al. (2023) 
 
Ba-Hien Tran, Giulio Franzese, Pietro Michiardi, and Maurizio Filippone.

 
 One-line-of-code data mollification improves optimization of
likelihood-based generative models.

 
 In Advances in Neural Information Processing Systems , 2023.

 

 
 Trofimov et al. (2023) 
 
Ilya Trofimov, Daniil Cherniavskii, Eduard Tulchinskii, Nikita Balabin, Evgeny
Burnaev, and Serguei Barannikov.

 
 Learning topology-preserving data representations.

 
 In International Conference on Learning Representations , 2023.

 

 
 Uria et al. (2013) 
 
Benigno Uria, Iain Murray, and Hugo Larochelle.

 
 RNADE: The real-valued neural autoregressive density-estimator.

 
 In Advances in Neural Information Processing Systems , 2013.

 

 
 Vahdat Kautz (2020) 
 
Arash Vahdat and Jan Kautz.

 
 NVAE: A deep hierarchical variational autoencoder.

 
 In Advances in Neural Information Processing Systems , 2020.

 

 
 Vahdat et al. (2021) 
 
Arash Vahdat, Karsten Kreis, and Jan Kautz.

 
 Score-based generative modeling in latent space.

 
 In Advances in Neural Information Processing Systems , 2021.

 

 
 van den Berg et al. (2018) 
 
Rianne van den Berg, Leonard Hasenclever, Jakub M Tomczak, and Max Welling.

 
 Sylvester normalizing flows for variational inference.

 
 In Uncertainty in Artificial Intelligence , 2018.

 

 
 van den Oord et al. (2016) 
 
Aäron van den Oord, Nal Kalchbrenner, Lasse Espeholt, Oriol Vinyals, Alex
Graves, and Koray Kavukcuoglu.

 
 Conditional image generation with PixelCNN decoders.

 
 In Advances in Neural Information Processing Systems , 2016.

 

 
 van den Oord et al. (2017) 
 
Aäron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu.

 
 Neural discrete representation learning.

 
 In Advances in Neural Information Processing Systems , 2017.

 

 
 van der Maaten Hinton (2008) 
 
Laurens van der Maaten and Geoffrey Hinton.

 
 Visualizing data using t-SNE.

 
 Journal of Machine Learning Research , 9(86):2579–2605, 2008.

 

 
 Vardanyan et al. (2024) 
 
Elen Vardanyan, Sona Hunanyan, Tigran Galstyan, Arshak Minasyan, and Arnak S
Dalalyan.

 
 Statistically optimal generative modeling with maximum deviation from
the empirical distribution.

 
 In International Conference on Machine Learning , 2024.

 

 
 Vaswani et al. (2017) 
 
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.

 
 Attention is all you need.

 
 In Advances in Neural Information Processing Systems , 2017.

 

 
 Villani (2009) 
 
Cédric Villani.

 
 Optimal Transport: Old and New .

 
 Springer Science Business Media, 2009.

 

 
 Vincent (2011) 
 
Pascal Vincent.

 
 A connection between score matching and denoising autoencoders.

 
 Neural computation , 23(7):1661–1674,
2011.

 

 
 Von Rohrscheidt Rieck (2023) 
 
Julius Von Rohrscheidt and Bastian Rieck.

 
 Topological singularity detection at multiple scales.

 
 In International Conference on Machine Learning , 2023.

 

 
 Wallace (1992) 
 
GK Wallace.

 
 The JPEG still picture compression standard.

 
 IEEE Transactions on Consumer Electronics , 38(1):18–34, 1992.

 

 
 Wand Jones (1994) 
 
Matt P Wand and M Chris Jones.

 
 Kernel Smoothing .

 
 CRC Press, 1994.

 

 
 Wang Wang (2024) 
 
Yi Wang and Zhiren Wang.

 
 CW complex hypothesis for image data.

 
 In International Conference on Machine Learning , 2024.

 

 
 Wang et al. (2021) 
 
Yixin Wang, David Blei, and John P Cunningham.

 
 Posterior collapse and latent variable non-identifiability.

 
 In Advances in Neural Information Processing Systems , 2021.

 

 
 Welling Teh (2011) 
 
Max Welling and Yee W Teh.

 
 Bayesian learning via stochastic gradient langevin dynamics.

 
 In International Conference on Machine Learning , 2011.

 

 
 Xiao et al. (2019) 
 
Zhisheng Xiao, Qing Yan, and Yali Amit.

 
 Generative latent flow.

 
 arXiv:1905.10485 , 2019.

 

 
 Xiao et al. (2022) 
 
Zhisheng Xiao, Karsten Kreis, and Arash Vahdat.

 
 Tackling the generative learning trilemma with denoising diffusion
GANs.

 
 In International Conference on Learning Representations , 2022.

 

 
 Xie et al. (2016) 
 
Jianwen Xie, Yang Lu, Song-Chun Zhu, and Yingnian Wu.

 
 A theory of generative ConvNet.

 
 In International Conference on Machine Learning , 2016.

 

 
 Yang et al. (2023) 
 
Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao
Zhang, Bin Cui, and Ming-Hsuan Yang.

 
 Diffusion models: A comprehensive survey of methods and applications.

 
 ACM Computing Survey , 2023.

 

 
 Yoon et al. (2021) 
 
Sangwoong Yoon, Yung-Kyun Noh, and Frank Park.

 
 Autoencoding under normalization constraints.

 
 In International Conference on Machine Learning , 2021.

 

 
 Yoon et al. (2023) 
 
Sangwoong Yoon, Young-Uk Jin, Yung-Kyun Noh, and Frank Park.

 
 Energy-based models for anomaly detection: A manifold diffusion
recovery approach.

 
 In Advances in Neural Information Processing Systems , 2023.

 

 
 Zhang et al. (2024) 
 
Hengrui Zhang, Jiani Zhang, Balasubramaniam Srinivasan, Zhengyuan Shen, Xiao
Qin, Christos Faloutsos, Huzefa Rangwala, and George Karypis.

 
 Mixed-type tabular data synthesis with score-based diffusion in
latent space.

 
 In International Conference on Learning Representations , 2024.

 

 
 Zhang et al. (2020a) 
 
Mingtian Zhang, Peter Hayes, Thomas Bird, Raza Habib, and David Barber.

 
 Spread divergence.

 
 In International Conference on Machine Learning ,
2020a.

 

 
 Zhang et al. (2023) 
 
Mingtian Zhang, Yitong Sun, Chen Zhang, and Steven Mcdonagh.

 
 Spread flows for manifold modelling.

 
 In International Conference on Artificial Intelligence and
Statistics , 2023.

 

 
 Zhang et al. (2020b) 
 
Zijun Zhang, Ruixiang Zhang, Zongpeng Li, Yoshua Bengio, and Liam Paull.

 
 Perceptual generative autoencoders.

 
 In International Conference on Machine Learning ,
2020b.

 

 
 Zheng et al. (2022) 
 
Yijia Zheng, Tong He, Yixuan Qiu, and David P Wipf.

 
 Learning manifold dimensions with conditional variational
autoencoders.

 
 In Advances in Neural Information Processing Systems , 2022.

 

 
 
 
 
 

## Appendix A Weak Convergence Primer

 
 We now provide a brief summary of weak convergence of probability measures. We do not use a grey box around this section due to its length, despite the content being fairly technical. As in the main text, all the measures we consider here will be defined on 𝒳 \mathcal{X} along with its Borel σ \sigma -algebra. Given a sequence of probability measures ( ℙ θ t X ) t = 1 ∞ (\mathbb{P}^{X}_{\theta_{t}})_{t=1}^{\infty} , we would like to define what it means for the sequence to converge to some probability measure ℙ † X \mathbb{P}^{X}_{\dagger} . As we will see, there are several ways to define convergence of probability measures; we would like to use one which captures the intuition that ℙ θ t X \mathbb{P}^{X}_{\theta_{t}} “learns” ℙ † X \mathbb{P}^{X}_{\dagger} in the sense that ℙ θ t X \mathbb{P}^{X}_{\theta_{t}} converges to ℙ † X \mathbb{P}^{X}_{\dagger} as t → ∞ t\rightarrow\infty if and only if samples from ℙ θ t X \mathbb{P}^{X}_{\theta_{t}} become progressively harder to distinguish from those of ℙ † X \mathbb{P}^{X}_{\dagger} as t t becomes larger, becoming indistinguishable in the limit. When ℙ θ t X \mathbb{P}^{X}_{\theta_{t}} represents a DGM, this is precisely the form of convergence we would hope to observe as we optimize its parameters θ t \theta_{t} .

 
 
 Strong convergence 

 
 The seemingly most natural way to define convergence is to say that ℙ θ t X \mathbb{P}^{X}_{\theta_{t}} converges to ℙ † X \mathbb{P}^{X}_{\dagger} as t → ∞ t\rightarrow\infty if ℙ θ t X ​ ( B ) → ℙ † X ​ ( B ) \mathbb{P}^{X}_{\theta_{t}}(B)\rightarrow\mathbb{P}^{X}_{\dagger}(B) as t → ∞ t\rightarrow\infty for every Borel set B B . This type of convergence is called strong convergence . As the name suggests, this type of convergence is too strong, to the point where it does not properly capture the intended intuition of “ ℙ θ t X \mathbb{P}^{X}_{\theta_{t}} converges to ℙ † X \mathbb{P}^{X}_{\dagger} if and only if ℙ θ t X \mathbb{P}^{X}_{\theta_{t}} learns ℙ † X \mathbb{P}^{X}_{\dagger} ”.

 
 
 Let us illustrate why this is the case with two examples. First, let ℙ θ t X \mathbb{P}^{X}_{\theta_{t}} be Gaussian with mean 0 0 and covariance matrix 1 / t ​ I D 1/t\,I_{D} . Intuitively, this sequence learns a point mass at 0 0 , δ 0 \delta_{0} , yet it does not strongly converge to it: { 0 } \{0\} is a Borel set, and ℙ θ t X ​ ( { 0 } ) = 0 → 0 \mathbb{P}^{X}_{\theta_{t}}(\{0\})=0\rightarrow 0 as t → ∞ t\rightarrow\infty , yet δ 0 ​ ( { 0 } ) = 1 ≠ 0 \delta_{0}(\{0\})=1\neq 0 . As a second example, consider ℙ θ t X = δ x t \mathbb{P}^{X}_{\theta_{t}}=\delta_{x_{t}} , where ( x t ) t = 1 ∞ (x_{t})_{t=1}^{\infty} is a fixed sequence converging to 0 0 with { x t } t = 1 ∞ ∩ { 0 } = ∅ \{x_{t}\}_{t=1}^{\infty}\cap\{0\}=\emptyset . Intuitively this sequence also learns δ 0 \delta_{0} , but similarly to the previous example, ℙ θ t X ​ ( { 0 } ) = 0 \mathbb{P}^{X}_{\theta_{t}}(\{0\})=0 for every t t and thus the sequence does not strongly converge to δ 0 \delta_{0} either.

 
 
 We highlight that these are not overly-contrived examples in the DGM setting. The first example illustrates a common scenario where a sequence of full-dimensional models ℙ θ t X ≪ λ D \mathbb{P}^{X}_{\theta_{t}}\ll\lambda_{D} “learn” a distribution ℙ † X \mathbb{P}^{X}_{\dagger} supported on a low-dimensional embedded submanifold ℳ \mathcal{M} of 𝒳 \mathcal{X} , without strongly converging to ℙ † X \mathbb{P}^{X}_{\dagger} . In this case, ℳ \mathcal{M} is also a Borel set, and ℙ θ t X ​ ( ℳ ) = 0 → 0 \mathbb{P}^{X}_{\theta_{t}}(\mathcal{M})=0\rightarrow 0 as t → ∞ t\rightarrow\infty even though ℙ † X ​ ( ℳ ) = 1 ≠ 0 \mathbb{P}^{X}_{\dagger}(\mathcal{M})=1\neq 0 . The second example illustrates how a sequence of models whose supports do not overlap with that of their target distribution can “learn” it without strongly converging to it. Indeed, we need a laxer definition of convergence of probability measures to properly convey the idea that a sequence of models ℙ θ t X \mathbb{P}^{X}_{\theta_{t}} “learns” ℙ † X \mathbb{P}^{X}_{\dagger} .

 
 
 
 Weak convergence 

 
 Weak – rather than strong – convergence provides a more appropriate notion of convergence to convey “learning”. We say that ℙ θ t X \mathbb{P}^{X}_{\theta_{t}} converges weakly to ℙ † X \mathbb{P}^{X}_{\dagger} if 𝔼 X ∼ ℙ θ t X ​ [ h ⁡ ( X ) ] → 𝔼 X ∼ ℙ † X ​ [ h ⁡ ( X ) ] \mathbb{E}_{X\sim\mathbb{P}^{X}_{\theta_{t}}}[h(X)]\rightarrow\mathbb{E}_{X\sim\mathbb{P}^{X}_{\dagger}}[h(X)] as t → ∞ t\rightarrow\infty for every bounded and continuous function h : 𝒳 → ℝ h:\mathcal{X}\rightarrow\mathbb{R} . As mentioned in Section 2.1 , we write ℙ θ t X → 𝜔 ℙ † X \mathbb{P}^{X}_{\theta_{t}}\xrightarrow{\omega}\mathbb{P}^{X}_{\dagger} as t → ∞ t\rightarrow\infty to denote weak convergence. Intuitively, ℙ θ t X \mathbb{P}^{X}_{\theta_{t}} converges weakly to ℙ † X \mathbb{P}^{X}_{\dagger} if, as t → ∞ t\rightarrow\infty , it becomes arbitrarily difficult to distinguish between samples from ℙ θ t X \mathbb{P}^{X}_{\theta_{t}} and samples from ℙ † X \mathbb{P}^{X}_{\dagger} by using a bounded and continuous function; weak convergence matches the intuition of ℙ θ t X \mathbb{P}^{X}_{\theta_{t}} “learning” ℙ † X \mathbb{P}^{X}_{\dagger} much better than strong convergence.

 
 
 There are many equivalent definitions of weak convergence, with the standard one being the one presented above. The result establishing the equivalence of these definitions is called the Portmanteau Lemma . We present a reduced version of this lemma below – which we will use to prove the Likelihood Instability Theorem in Appendix B.1 – where only one of these equivalences is stated. Before stating the lemma, we define the continuity sets of a probability measure.

 
 
 Definition 1 (Continuity Set) . 
 
 Let ℙ † X \mathbb{P}^{X}_{\dagger} be a probability measure on 𝒳 \mathcal{X} and B ⊂ 𝒳 B\subset\mathcal{X} a Borel set. We say that B B is a continuity set of ℙ † X \mathbb{P}^{X}_{\dagger} if ℙ † X ​ ( ∂ 𝒳 B ) = 0 \mathbb{P}^{X}_{\dagger}(\partial_{\mathcal{X}}B)=0 , where ∂ 𝒳 B \partial_{\mathcal{X}}B denotes the topological boundary of B B on 𝒳 \mathcal{X} . 

 
 
 
 Lemma 1 (Portmanteau) . 
 
 Let ℙ † X \mathbb{P}^{X}_{\dagger} be a probability measure on 𝒳 \mathcal{X} , and let ( ℙ θ t X ) t = 1 ∞ (\mathbb{P}^{X}_{\theta_{t}})_{t=1}^{\infty} be a sequence of probability measures on 𝒳 \mathcal{X} . Then, ℙ θ t X → 𝜔 ℙ † X \mathbb{P}^{X}_{\theta_{t}}\xrightarrow{\omega}\mathbb{P}^{X}_{\dagger} as t → ∞ t\rightarrow\infty if and only if ℙ θ t X ​ ( B ) → ℙ † X ​ ( B ) \mathbb{P}^{X}_{\theta_{t}}(B)\rightarrow\mathbb{P}^{X}_{\dagger}(B) as t → ∞ t\rightarrow\infty for every continuity set B B of ℙ † X \mathbb{P}^{X}_{\dagger} . 

 
 
 
 Let us consider once again the example where ℙ θ t X \mathbb{P}^{X}_{\theta_{t}} is Gaussian with mean 0 0 and covariance matrix 1 / t ​ I D 1/t\,I_{D} . It is not difficult to prove that ℙ θ t X ​ ( B ) → δ 0 ​ ( B ) \mathbb{P}^{X}_{\theta_{t}}(B)\rightarrow\delta_{0}(B) as t → ∞ t\rightarrow\infty for every Borel set B B such that 0 ∉ ∂ 𝒳 B 0\notin\partial_{\mathcal{X}}B (i.e. δ 0 ​ ( ∂ 𝒳 B ) = 0 \delta_{0}(\partial_{\mathcal{X}}B)=0 ), so that as intended, ℙ θ t X → 𝜔 δ 0 \mathbb{P}^{X}_{\theta_{t}}\xrightarrow{\omega}\delta_{0} as t → ∞ t\rightarrow\infty . Similarly, it is not difficult to prove the same in the example where ℙ θ t X = δ x t \mathbb{P}^{X}_{\theta_{t}}=\delta_{x_{t}} . The fact that these sequences converge weakly but not strongly to δ 0 \delta_{0} illustrates that weak convergence does indeed provide the right tool to talk about a sequence of models “learning” their target distribution.

 
 
 
 Metrizing weak convergence 

 
 Finally, we say that a metric 𝔻 : Δ ⁡ ( 𝒳 ) × Δ ⁡ ( 𝒳 ) → ℝ \mathbb{D}:\Delta(\mathcal{X})\times\Delta(\mathcal{X})\rightarrow\mathbb{R} on the space of probability measures on 𝒳 \mathcal{X} metrizes weak convergence if 𝔻 ⁡ ( ℙ θ t X , ℙ † X ) → 0 \mathbb{D}(\mathbb{P}^{X}_{\theta_{t}},\mathbb{P}^{X}_{\dagger})\rightarrow 0 as t → 0 t\rightarrow 0 holds if and only if ℙ θ t X → 𝜔 ℙ † X \mathbb{P}^{X}_{\theta_{t}}\xrightarrow{\omega}\mathbb{P}^{X}_{\dagger} as t → ∞ t\rightarrow\infty . Throughout the main manuscript, we often abuse language and use the term “metrizing weak convergence” even when 𝔻 \mathbb{D} is only a divergence rather than a metric, as this is enough to ensure that minimizing 𝔻 ⁡ ( ℙ θ X , ℙ ∗ X ) \mathbb{D}(\mathbb{P}^{X}_{\theta},\mathbb{P}^{X}_{*}) over θ \theta is a sensible training objective for a DGM ℙ θ X \mathbb{P}^{X}_{\theta} – even if ℙ ∗ X \mathbb{P}^{X}_{*} has low-dimensional support.

 
 
 
 

## Appendix B Proofs

 
 As in Appendix A , we omit the use of a grey box despite the use of technical language.

 
 

### Section B.1 The Likelihood Instability Theorem

 
 We start by restating the Likelihood Instability Theorem for convenience before discussing it.

 
 
 See 1 

 
 
 As mentioned in Section 4.1.1 we begin with an example satisfying the assumptions of the Likelihood Instability Theorem for which it does not hold that p θ t X ​ ( x ) → ∞ {p}^{X}_{\theta_{t}}(x)\rightarrow\infty for x ∈ cl 𝒳 ⁡ ( M ) x\in\cl_{\mathcal{X}}(M) , thus highlighting that the theorem cannot be “trivially strengthened”. Consider 𝒳 = ℝ 2 \mathcal{X}=\mathbb{R}^{2} , M = { ( x 1 , 0 ) ∈ ℝ 2 ∣ 0 x 1 1 } M=\{(x_{1},0)\in\mathbb{R}^{2}\mid 0 x_{1} 1\} , let ℙ † X \mathbb{P}^{X}_{\dagger} be uniform on M M , and ℙ θ t X \mathbb{P}^{X}_{\theta_{t}} be uniform on M t M_{t} , where M t = { ( x 1 , x 2 ) ∈ ℝ 2 ∣ 0 x 1 1 ​  and  ​ 1 / ( t + 1 ) x 2 1 / t } M_{t}=\{(x_{1},x_{2})\in\mathbb{R}^{2}\mid 0 x_{1} 1\text{ and }1/(t+1) x_{2} 1/t\} . Finally, take the corresponding densities as

 

 
 | 
 p θ t X ​ ( x ) = 𝟙 ​ ( x ∈ M t ) λ 2 ​ ( M t ) = t ⁡ ( t + 1 ) ​ 𝟙 ​ ( x ∈ M t ) , {p}^{X}_{\theta_{t}}(x)=\dfrac{\mathds{1}(x\in M_{t})}{\lambda_{2}(M_{t})}=t(t+1)\mathds{1}(x\in M_{t}), | 
 | 
 (99) | 
 

 where 𝟙 ​ ( ⋅ ) \mathds{1}(\cdot) denotes an indicator function. Figure 11 illustrates this example. Here, it holds that ℙ θ t X → 𝜔 ℙ † X \mathbb{P}^{X}_{\theta_{t}}\xrightarrow{\omega}\mathbb{P}^{X}_{\dagger} – so that the assumptions of the Likelihood Instability Theorem are satisfied – yet p θ t X ​ ( x ) → 0 {p}^{X}_{\theta_{t}}(x)\rightarrow 0 as t → ∞ t\rightarrow\infty for every x ∈ 𝒳 x\in\mathcal{X} , and thus in particular for every x ∈ cl 𝒳 ⁡ ( M ) x\in\cl_{\mathcal{X}}(M) as well. Nonetheless, when x ∈ cl 𝒳 ⁡ ( M ) x\in\cl_{\mathcal{X}}(M) , sup x ′ ∈ B ε ​ ( x ) p θ t X ​ ( x ′ ) → ∞ \sup_{x^{\prime}\in B_{\varepsilon}(x)}{p}^{X}_{\theta_{t}}(x^{\prime})\rightarrow\infty as t → ∞ t\rightarrow\infty does hold for every ε 0 \varepsilon 0 , as concluded by the theorem.

 
 
 Before proving the Likelihood Instability Theorem , we state and prove three lemmas, all of which we will rely on. We will heavily use Continuity Sets and will leverage the Portmanteau Lemma ; see Appendix A for a reminder on these topics.

 
 
 Lemma 2 . 
 
 Let ℙ † X \mathbb{P}^{X}_{\dagger} be a probability measure on 𝒳 \mathcal{X} , x ∈ 𝒳 x\in\mathcal{X} , and ε 0 \varepsilon 0 . Then, there exists ε ′ ∈ ( 0 , ε ) \varepsilon^{\prime}\in(0,\varepsilon) such that B ε ′ ​ ( x ) B_{\varepsilon^{\prime}}(x) is a continuity set of ℙ † X \mathbb{P}^{X}_{\dagger} , where B ε ​ ( x ) = { x ′ ∈ 𝒳 ∣ ‖ x ′ − x ‖ 2 ε } B_{\varepsilon}(x)=\{x^{\prime}\in\mathcal{X}\mid\|x^{\prime}-x\|_{2} \varepsilon\} . 

 
 
 
 Proof. 
 
 We proceed by contradiction: let x ∈ 𝒳 x\in\mathcal{X} and ε 0 \varepsilon 0 , and assume that ℙ † X ​ ( ∂ 𝒳 B ε ′ ​ ( x ) ) 0 \mathbb{P}^{X}_{\dagger}(\partial_{\mathcal{X}}B_{\varepsilon^{\prime}}(x)) 0 for every ε ′ ∈ ( 0 , ε ) \varepsilon^{\prime}\in(0,\varepsilon) . Since we can countably partition ( 0 , 1 ] (0,1] as ∪ n = 2 ∞ ( 1 / n , 1 / ( n − 1 ) ] \cup_{n=2}^{\infty}(1/n,1/(n-1)] , it follows that uncountably many elements from { ℙ † X ​ ( ∂ 𝒳 B ε ′ ​ ( x ) ) } ε ′ ∈ ( 0 , ε ) \{\mathbb{P}^{X}_{\dagger}(\partial_{\mathcal{X}}B_{\varepsilon^{\prime}}(x))\}_{\varepsilon^{\prime}\in(0,\varepsilon)} belong to an element of the partition. Thus, in particular there exists an integer n ′ n^{\prime} and distinct numbers ε 1 ′ , ε 2 ′ , … , ε n ′ ′ \varepsilon^{\prime}_{1},\varepsilon^{\prime}_{2},\dots,\varepsilon^{\prime}_{n^{\prime}} in ( 0 , ε ) (0,\varepsilon) such that ℙ † X ​ ( ∂ 𝒳 B ε i ′ ​ ( x ) ) 1 / n ′ \mathbb{P}^{X}_{\dagger}(\partial_{\mathcal{X}}B_{\varepsilon^{\prime}_{i}}(x)) 1/n^{\prime} for every i = 1 , 2 , … , n ′ i=1,2,\dots,n^{\prime} . Then, because ∂ 𝒳 B ε i ′ ​ ( x ) ∩ ∂ 𝒳 B ε j ′ ​ ( x ) = ∅ \partial_{\mathcal{X}}B_{\varepsilon^{\prime}_{i}}(x)\cap\partial_{\mathcal{X}}B_{\varepsilon^{\prime}_{j}}(x)=\emptyset whenever i ≠ j i\neq j , we have that 

 

 
 | 
 ℙ † X ​ ( ⋃ i = 1 n ′ ∂ 𝒳 B ε i ′ ​ ( x ) ) = ∑ i = 1 n ′ ℙ † X ​ ( ∂ 𝒳 B ε i ′ ​ ( x ) ) ∑ i = 1 n ′ 1 n ′ = 1 , \mathbb{P}^{X}_{\dagger}\left(\bigcup_{i=1}^{n^{\prime}}\partial_{\mathcal{X}}B_{\varepsilon^{\prime}_{i}}(x)\right)=\sum_{i=1}^{n^{\prime}}\mathbb{P}^{X}_{\dagger}\left(\partial_{\mathcal{X}}B_{\varepsilon^{\prime}_{i}}(x)\right) \sum_{i=1}^{n^{\prime}}\dfrac{1}{n^{\prime}}=1, | 
 | 
 (100) | 
 

 which is clearly a contradiction since ℙ † X ​ ( 𝒳 ) = 1 \mathbb{P}^{X}_{\dagger}(\mathcal{X})=1 , thus finishing the proof.
∎ 

 
 
 
 Figure 11: Visualization of the sequence of densities from Equation 99 , which converge weakly to a uniform distribution on M M . For x ∈ cl 𝒳 ⁡ ( M ) x\in\cl_{\mathcal{X}}(M) it always holds that p θ t X ​ ( x ) = 0 {p}^{X}_{\theta_{t}}(x)=0 because the support of p θ t X {p}^{X}_{\theta_{t}} does not overlap with cl 𝒳 ⁡ ( M ) \cl_{\mathcal{X}}(M) , so that p θ t X ​ ( x ) → ∞ {p}^{X}_{\theta_{t}}(x)\rightarrow\infty as t → ∞ t\rightarrow\infty does not hold. Nonetheless, for any fixed ε 0 \varepsilon 0 , the support of p θ t X {p}^{X}_{\theta_{t}} always overlaps with B ε ​ ( x ) B_{\varepsilon}(x) for large enough t t , and thus sup x ′ ∈ B ε ​ ( x ) p θ t X ​ ( x ′ ) → ∞ \sup_{x^{\prime}\in B_{\varepsilon}(x)}{p}^{X}_{\theta_{t}}(x^{\prime})\rightarrow\infty as t → ∞ t\rightarrow\infty . 
 
 
 Lemma 3 . 
 
 Let M ⊂ 𝒳 M\subset\mathcal{X} , and let ℙ † X \mathbb{P}^{X}_{\dagger} be a probability measure on 𝒳 \mathcal{X} such that ℙ † X ​ ( M ) = 1 \mathbb{P}^{X}_{\dagger}(M)=1 . Then for every δ 0 \delta 0 , the set M δ ≔ { x ∈ 𝒳 ∣ inf x ′ ∈ M ‖ x ′ − x ‖ 2 δ } M_{\delta}\coloneqq\{x\in\mathcal{X}\mid\inf_{x^{\prime}\in M}\|x^{\prime}-x\|_{2} \delta\} is open in 𝒳 \mathcal{X} , and is a continuity set of ℙ † X \mathbb{P}^{X}_{\dagger} . 

 
 
 
 Proof. 
 
 M δ M_{\delta} is open because it can be written as a union of open sets: 

 

 
 | 
 M δ = ⋃ x ′ ∈ M B δ ​ ( x ′ ) . M_{\delta}=\bigcup_{x^{\prime}\in M}B_{\delta}(x^{\prime}). | 
 | 
 (101) | 
 

 Then, we have: 

 

 
 | 
 ℙ † X ​ ( ∂ 𝒳 M δ ) \displaystyle\mathbb{P}^{X}_{\dagger}\left(\partial_{\mathcal{X}}M_{\delta}\right) | 
 = ℙ † X ​ ( ∂ 𝒳 M δ ∩ M ) = ℙ † X ​ ( ( cl 𝒳 ⁡ ( M δ ) ∖ int 𝒳 ⁡ ( M δ ) ) ∩ M ) = ℙ † X ​ ( ( cl 𝒳 ⁡ ( M δ ) ∖ M δ ) ∩ M ) \displaystyle=\mathbb{P}^{X}_{\dagger}\left(\partial_{\mathcal{X}}M_{\delta}\cap M\right)=\mathbb{P}^{X}_{\dagger}\left((\cl_{\mathcal{X}}(M_{\delta})\setminus\interior_{\mathcal{X}}(M_{\delta}))\cap M\right)=\mathbb{P}^{X}_{\dagger}\left((\cl_{\mathcal{X}}(M_{\delta})\setminus M_{\delta})\cap M\right) | 
 | 
 (102) | 
 
 
 | 
 | 
 = ℙ † X ​ ( ∅ ) = 0 , \displaystyle=\mathbb{P}^{X}_{\dagger}(\emptyset)=0, | 
 | 
 (103) | 
 

 where int 𝒳 ⁡ ( M ) \interior_{\mathcal{X}}(M) denotes the topological interior of M M in 𝒳 \mathcal{X} , and the first equality follows from ℙ † X ​ ( M ) = 1 \mathbb{P}^{X}_{\dagger}(M)=1 , the second one from the definition of boundary, the third one from M δ M_{\delta} being open, and the fourth one from M ⊂ M δ M\subset M_{\delta} . Thus M δ M_{\delta} is indeed a continuity set of ℙ † X \mathbb{P}^{X}_{\dagger} .
∎ 

 
 
 
 Lemma 4 . 
 
 Let M ⊂ 𝒳 M\subset\mathcal{X} , and let ℙ † X \mathbb{P}^{X}_{\dagger} be a probability measure on 𝒳 \mathcal{X} such that supp ⁡ ( ℙ † X ) = cl 𝒳 ⁡ ( M ) \supp(\mathbb{P}^{X}_{\dagger})=\cl_{\mathcal{X}}(M) . Then ℙ † X ​ ( B ε ​ ( x ) ) 0 \mathbb{P}^{X}_{\dagger}(B_{\varepsilon}(x)) 0 for every x ∈ cl 𝒳 ⁡ ( M ) x\in\cl_{\mathcal{X}}(M) and every ε 0 \varepsilon 0 , where B ε ​ ( x ) ≔ { x ′ ∈ 𝒳 ∣ ‖ x ′ − x ‖ 2 ε } B_{\varepsilon}(x)\coloneqq\{x^{\prime}\in\mathcal{X}\mid\|x^{\prime}-x\|_{2} \varepsilon\} . 

 
 
 
 Proof. 
 
 Since ℙ † X \mathbb{P}^{X}_{\dagger} is a Borel measure and 𝒳 \mathcal{X} is separable, ℙ † X ​ ( supp ⁡ ( ℙ † X ) ) = 1 \mathbb{P}^{X}_{\dagger}(\supp(\mathbb{P}^{X}_{\dagger}))=1 ( Bogachev, 2007 , Proposition 7.2.9) . 30 30 
 30 
 
 
 
 Note that in general, ℙ ⁡ ( supp ⁡ ( ℙ ) ) = 1 \mathbb{P}(\supp(\mathbb{P}))=1 need not hold, see for example Schilling Kühn (2021, Examples 6.2 and 6.3) . It follows that cl 𝒳 ⁡ ( M ) = supp ⁡ ( ℙ † X ) \cl_{\mathcal{X}}(M)=\supp(\mathbb{P}^{X}_{\dagger}) is nonempty.
Let x ∈ cl 𝒳 ⁡ ( M ) x\in\cl_{\mathcal{X}}(M) and ε 0 \varepsilon 0 . We proceed by contradiction, and assume that ℙ † X ​ ( B ε ​ ( x ) ) = 0 \mathbb{P}^{X}_{\dagger}(B_{\varepsilon}(x))=0 .
Since ℙ † X ​ ( cl 𝒳 ⁡ ( M ) ) = 1 \mathbb{P}^{X}_{\dagger}(\cl_{\mathcal{X}}(M))=1 , we have that ℙ † X ​ ( cl 𝒳 ⁡ ( M ) ∖ B ε ​ ( x ) ) = 1 \mathbb{P}^{X}_{\dagger}(\cl_{\mathcal{X}}(M)\setminus B_{\varepsilon}(x))=1 . Since B ε ​ ( x ) B_{\varepsilon}(x) is open, cl 𝒳 ⁡ ( M ) ∖ B ε ​ ( x ) \cl_{\mathcal{X}}(M)\setminus B_{\varepsilon}(x) is closed. Then, by definition of support ( Equation 1 ) and because supp ⁡ ( ℙ † X ) = cl 𝒳 ⁡ ( M ) \supp(\mathbb{P}^{X}_{\dagger})=\cl_{\mathcal{X}}(M) , it follows that cl 𝒳 ⁡ ( M ) ∩ B ε ​ ( x ) = ∅ \cl_{\mathcal{X}}(M)\cap B_{\varepsilon}(x)=\emptyset . This is clearly a contradiction since x ∈ cl 𝒳 ⁡ ( M ) ∩ B ε ​ ( x ) x\in\cl_{\mathcal{X}}(M)\cap B_{\varepsilon}(x) .
∎ 

 
 
 
 Proof of Theorem 1 . 
 
 We first prove that lim inf t → ∞ p θ t X ​ ( x ) = 0 \liminf_{t\rightarrow\infty}{p}^{X}_{\theta_{t}}(x)=0 , λ D \lambda_{D} -almost-surely on 𝒳 ∖ cl 𝒳 ⁡ ( M ) \mathcal{X}\setminus\cl_{\mathcal{X}}(M) . Let x ∈ 𝒳 ∖ cl 𝒳 ⁡ ( M ) x\in\mathcal{X}\setminus\cl_{\mathcal{X}}(M) , and let U x U_{x} be an open neighbourhood of x x such that cl 𝒳 ⁡ ( U x ) ∩ cl 𝒳 ⁡ ( M ) = ∅ \cl_{\mathcal{X}}(U_{x})\cap\cl_{\mathcal{X}}(M)=\emptyset , which exists because 𝒳 \mathcal{X} is regular. Clearly ℙ † X ​ ( ∂ 𝒳 U x ) = 0 \mathbb{P}^{X}_{\dagger}(\partial_{\mathcal{X}}U_{x})=0 (i.e. U x U_{x} is a continuity set of ℙ † X \mathbb{P}^{X}_{\dagger} ) and ℙ † X ​ ( U x ) = 0 \mathbb{P}^{X}_{\dagger}(U_{x})=0 since cl 𝒳 ⁡ ( U x ) ∩ cl 𝒳 ⁡ ( M ) = ∅ \cl_{\mathcal{X}}(U_{x})\cap\cl_{\mathcal{X}}(M)=\emptyset and ℙ † X ​ ( cl 𝒳 ⁡ ( M ) ) = 1 \mathbb{P}^{X}_{\dagger}(\cl_{\mathcal{X}}(M))=1 . 

 
 
 By the Portmanteau Lemma , ℙ θ t X ​ ( U x ) → ℙ † X ​ ( U x ) \mathbb{P}^{X}_{\theta_{t}}(U_{x})\rightarrow\mathbb{P}^{X}_{\dagger}(U_{x}) as t → ∞ t\rightarrow\infty , and we have the following implications: 

 

 
 | 
 lim t → ∞ ℙ θ t X ​ ( U x ) = ℙ † X ​ ( U x ) \displaystyle\lim_{t\rightarrow\infty}\mathbb{P}^{X}_{\theta_{t}}(U_{x})=\mathbb{P}^{X}_{\dagger}(U_{x}) | 
 ⇔ lim t → ∞ ∫ U x p θ t X ​ ( x ′ ) ​ d ​ λ D ​ ( x ′ ) = ℙ † X ​ ( U x ) = 0 \displaystyle\iff\lim_{t\rightarrow\infty}\int_{U_{x}}{p}^{X}_{\theta_{t}}(x^{\prime}){\textnormal{d}}\lambda_{D}(x^{\prime})=\mathbb{P}^{X}_{\dagger}(U_{x})=0 | 
 | 
 (104) | 
 
 
 | 
 | 
 ⟹ lim inf t → ∞ ∫ U x p θ t X ​ ( x ′ ) ​ d ​ λ D ​ ( x ′ ) = 0 . \displaystyle\implies\liminf_{t\rightarrow\infty}\int_{U_{x}}{p}^{X}_{\theta_{t}}(x^{\prime}){\textnormal{d}}\lambda_{D}(x^{\prime})=0. | 
 | 
 (105) | 
 

 Then, by Fatou’s lemma, 

 

 
 | 
 ∫ U x lim inf t → ∞ p θ t X ​ ( x ′ ) ​ d ​ λ D ​ ( x ′ ) ≤ 0 . \int_{U_{x}}\liminf_{t\rightarrow\infty}{p}^{X}_{\theta_{t}}(x^{\prime}){\textnormal{d}}\lambda_{D}(x^{\prime})\leq 0. | 
 | 
 (106) | 
 

 Since lim inf t → ∞ p θ t X ​ ( x ) ≥ 0 \liminf_{t\rightarrow\infty}{p}^{X}_{\theta_{t}}(x)\geq 0 , λ D \lambda_{D} -almost-everywhere on 𝒳 \mathcal{X} , it follows that 

 

 
 | 
 ∫ U x lim inf t → ∞ p θ t X ​ ( x ′ ) ​ d ​ λ D ​ ( x ′ ) = 0 , \int_{U_{x}}\liminf_{t\rightarrow\infty}{p}^{X}_{\theta_{t}}(x^{\prime}){\textnormal{d}}\lambda_{D}(x^{\prime})=0, | 
 | 
 (107) | 
 

 and thus lim inf t → ∞ p θ t X ​ ( x ) = 0 \liminf_{t\rightarrow\infty}{p}^{X}_{\theta_{t}}(x)=0 , λ D \lambda_{D} -almost-everywhere on U x U_{x} . 

 
 
 We still need to extend the result from λ D \lambda_{D} -almost-everywhere on U x U_{x} to λ D \lambda_{D} -almost-everywhere on 𝒳 ∖ cl 𝒳 ⁡ ( M ) \mathcal{X}\setminus\cl_{\mathcal{X}}(M) . Clearly { U x } x ∈ 𝒳 ∖ cl 𝒳 ⁡ ( M ) \{U_{x}\}_{x\in\mathcal{X}\setminus\cl_{\mathcal{X}}(M)} is an open cover of 𝒳 ∖ cl 𝒳 ⁡ ( M ) \mathcal{X}\setminus\cl_{\mathcal{X}}(M) . Since 𝒳 \mathcal{X} is second countable and every subspace of a second countable space is second countable, it follows that 𝒳 ∖ cl 𝒳 ⁡ ( M ) \mathcal{X}\setminus\cl_{\mathcal{X}}(M) is second countable. Then, by Lindelöf’s lemma, { U x } x ∈ 𝒳 ∖ cl 𝒳 ⁡ ( M ) \{U_{x}\}_{x\in\mathcal{X}\setminus\cl_{\mathcal{X}}(M)} has a countable subcover { U x i } i = 1 ∞ \{U_{x_{i}}\}_{i=1}^{\infty} of 𝒳 ∖ cl 𝒳 ⁡ ( M ) \mathcal{X}\setminus\cl_{\mathcal{X}}(M) . The result then holds λ D \lambda_{D} -almost-everywhere on U x i U_{x_{i}} for i = 1 , 2 , … i=1,2,\dots , and because any countable union of sets of measure 0 0 has measure 0 0 , it also holds λ D \lambda_{D} -almost-everywhere on 

 

 
 | 
 ⋃ i = 1 ∞ U x i = 𝒳 ∖ cl 𝒳 ⁡ ( M ) , \bigcup_{i=1}^{\infty}U_{x_{i}}=\mathcal{X}\setminus\cl_{\mathcal{X}}(M), | 
 | 
 (108) | 
 

 which finishes this part of the proof. 

 
 
 Now, let x ∈ cl 𝒳 ⁡ ( M ) x\in\cl_{\mathcal{X}}(M) and ε 0 \varepsilon 0 , and we will prove that sup x ′ ∈ B ε ​ ( x ) p θ t X ​ ( x ′ ) → ∞ \sup_{x^{\prime}\in B_{\varepsilon}(x)}{p}^{X}_{\theta_{t}}(x^{\prime})\rightarrow\infty as t → ∞ t\rightarrow\infty . 

 
 
 First, note that sup x ′ ∈ B ε ​ ( x ) p θ t X ​ ( x ′ ) \sup_{x^{\prime}\in B_{\varepsilon}(x)}{p}^{X}_{\theta_{t}}(x^{\prime}) is increasing in ε \varepsilon for every t t . If B ε ​ ( x ) B_{\varepsilon}(x) is not a continuity set of ℙ † X \mathbb{P}^{X}_{\dagger} , by 2 we could always find ε ′ ∈ ( 0 , ε ) \varepsilon^{\prime}\in(0,\varepsilon) such that B ε ′ ​ ( x ) B_{\varepsilon^{\prime}}(x) is a continuity set of ℙ † X \mathbb{P}^{X}_{\dagger} , and if we managed to prove that sup x ′ ∈ B ε ′ ​ ( x ) p θ t X ​ ( x ′ ) → ∞ \sup_{x^{\prime}\in B_{\varepsilon^{\prime}}(x)}{p}^{X}_{\theta_{t}}(x^{\prime})\rightarrow\infty as t → ∞ t\rightarrow\infty , the same result would immediately follow for ε \varepsilon . We can thus assume without loss of generality that ε \varepsilon is such that B ε ​ ( x ) B_{\varepsilon}(x) is a continuity set of ℙ † X \mathbb{P}^{X}_{\dagger} . 

 
 
 Now, let M δ ≔ { x ∈ 𝒳 ∣ inf x ′ ∈ M ‖ x ′ − x ‖ 2 δ } M_{\delta}\coloneqq\{x\in\mathcal{X}\mid\inf_{x^{\prime}\in M}\|x^{\prime}-x\|_{2} \delta\} and let U ε , δ ​ ( x ) ≔ M δ ∩ B ε ​ ( x ) U_{\varepsilon,\delta}(x)\coloneqq M_{\delta}\cap B_{\varepsilon}(x) . From basic topology we have that ∂ 𝒳 ( M δ ∩ B ε ​ ( x ) ) ⊂ ∂ 𝒳 M δ ∪ ∂ 𝒳 B ε ​ ( x ) \partial_{\mathcal{X}}(M_{\delta}\cap B_{\varepsilon}(x))\subset\partial_{\mathcal{X}}M_{\delta}\cup\partial_{\mathcal{X}}B_{\varepsilon}(x) . Since M δ M_{\delta} and B ε ​ ( x ) B_{\varepsilon}(x) are continuity sets of ℙ † X \mathbb{P}^{X}_{\dagger} by 3 and by assumption, respectively, it follows that U ε , δ ​ ( x ) U_{\varepsilon,\delta}(x) is a continuity set of ℙ † X \mathbb{P}^{X}_{\dagger} , since ℙ † X ​ ( ∂ 𝒳 U ε , δ ​ ( x ) ) ≤ ℙ † X ​ ( ∂ 𝒳 M δ ) + ℙ † X ​ ( ∂ 𝒳 B ε ​ ( x ) ) = 0 \mathbb{P}^{X}_{\dagger}(\partial_{\mathcal{X}}U_{\varepsilon,\delta}(x))\leq\mathbb{P}^{X}_{\dagger}(\partial_{\mathcal{X}}M_{\delta})+\mathbb{P}^{X}_{\dagger}(\partial_{\mathcal{X}}B_{\varepsilon}(x))=0 . Similarly, B ε ​ ( x ) ∖ M δ B_{\varepsilon}(x)\setminus M_{\delta} is a continuity set of ℙ † X \mathbb{P}^{X}_{\dagger} because ∂ 𝒳 ( B ε ​ ( x ) ∖ M δ ) ⊂ ∂ 𝒳 B ε ​ ( x ) ∪ ∂ 𝒳 M δ \partial_{\mathcal{X}}(B_{\varepsilon}(x)\setminus M_{\delta})\subset\partial_{\mathcal{X}}B_{\varepsilon}(x)\cup\partial_{\mathcal{X}}M_{\delta} . We then write: 

 

 
 | 
 ℙ θ t X ​ ( B ε ​ ( x ) ) \displaystyle\mathbb{P}^{X}_{\theta_{t}}\left(B_{\varepsilon}(x)\right) | 
 = ℙ θ t X ​ ( U ε , δ ​ ( x ) ) + ℙ θ t X ​ ( B ε ​ ( x ) ∖ M δ ) = ∫ U ε , δ ​ ( x ) p θ t X ​ ( x ′ ) ​ d ​ λ D ​ ( x ′ ) + ℙ θ t X ​ ( B ε ​ ( x ) ∖ M δ ) \displaystyle=\mathbb{P}^{X}_{\theta_{t}}\left(U_{\varepsilon,\delta}(x)\right)+\mathbb{P}^{X}_{\theta_{t}}\left(B_{\varepsilon}(x)\setminus M_{\delta}\right)=\int_{U_{\varepsilon,\delta}(x)}{p}^{X}_{\theta_{t}}(x^{\prime}){\textnormal{d}}\lambda_{D}(x^{\prime})+\mathbb{P}^{X}_{\theta_{t}}\left(B_{\varepsilon}(x)\setminus M_{\delta}\right) | 
 | 
 (109) | 
 
 
 | 
 | 
 ≤ λ D ​ ( U ε , δ ​ ( x ) ) ​ sup x ′ ∈ U ε , δ ​ ( x ) p θ t X ​ ( x ′ ) + ℙ θ t X ​ ( B ε ​ ( x ) ∖ M δ ) \displaystyle\leq\lambda_{D}\left(U_{\varepsilon,\delta}(x)\right)\sup_{x^{\prime}\in U_{\varepsilon,\delta}(x)}{p}^{X}_{\theta_{t}}(x^{\prime})+\mathbb{P}^{X}_{\theta_{t}}\left(B_{\varepsilon}(x)\setminus M_{\delta}\right) | 
 | 
 (110) | 
 
 
 | 
 | 
 ≤ λ D ​ ( U ε , δ ​ ( x ) ) ​ sup x ′ ∈ B ε ​ ( x ) p θ t X ​ ( x ′ ) + ℙ θ t X ​ ( B ε ​ ( x ) ∖ M δ ) . \displaystyle\leq\lambda_{D}\left(U_{\varepsilon,\delta}(x)\right)\sup_{x^{\prime}\in B_{\varepsilon}(x)}{p}^{X}_{\theta_{t}}(x^{\prime})+\mathbb{P}^{X}_{\theta_{t}}\left(B_{\varepsilon}(x)\setminus M_{\delta}\right). | 
 | 
 (111) | 
 

 Since B ε ​ ( x ) B_{\varepsilon}(x) is open, as is M δ M_{\delta} by 3 , then U ε , δ ​ ( x ) U_{\varepsilon,\delta}(x) is open as well. In turn λ D ​ ( U ε , δ ​ ( x ) ) 0 \lambda_{D}(U_{\varepsilon,\delta}(x)) 0 , and it follows that 

 

 
 | 
 ℙ θ t X ​ ( B ε ​ ( x ) ) − ℙ θ t X ​ ( B ε ​ ( x ) ∖ M δ ) λ D ​ ( U ε , δ ​ ( x ) ) ≤ sup x ′ ∈ B ε ​ ( x ) p θ t X ​ ( x ′ ) . \dfrac{\mathbb{P}^{X}_{\theta_{t}}\left(B_{\varepsilon}(x)\right)-\mathbb{P}^{X}_{\theta_{t}}\left(B_{\varepsilon}(x)\setminus M_{\delta}\right)}{\lambda_{D}\left(U_{\varepsilon,\delta}(x)\right)}\leq\sup_{x^{\prime}\in B_{\varepsilon}(x)}{p}^{X}_{\theta_{t}}(x^{\prime}). | 
 | 
 (112) | 
 

 By the Portmanteau Lemma and since ℙ † X ​ ( B ε ​ ( x ) ∖ M δ ) = 0 \mathbb{P}^{X}_{\dagger}(B_{\varepsilon}(x)\setminus M_{\delta})=0 , taking the limit as t → ∞ t\rightarrow\infty of the left hand side of the above equation yields 

 

 
 | 
 lim t → ∞ ℙ θ t X ​ ( B ε ​ ( x ) ) − ℙ θ t X ​ ( B ε ​ ( x ) ∖ M δ ) λ D ​ ( U ε , δ ​ ( x ) ) = ℙ † X ​ ( B ε ​ ( x ) ) λ D ​ ( U ε , δ ​ ( x ) ) . \lim_{t\rightarrow\infty}\dfrac{\mathbb{P}^{X}_{\theta_{t}}\left(B_{\varepsilon}(x)\right)-\mathbb{P}^{X}_{\theta_{t}}\left(B_{\varepsilon}(x)\setminus M_{\delta}\right)}{\lambda_{D}\left(U_{\varepsilon,\delta}(x)\right)}=\dfrac{\mathbb{P}^{X}_{\dagger}\left(B_{\varepsilon}(x)\right)}{\lambda_{D}\left(U_{\varepsilon,\delta}(x)\right)}. | 
 | 
 (113) | 
 

 
 
 Thus, taking lim inf \liminf as t → ∞ t\rightarrow\infty on both sides of Equation 112 implies that 

 

 
 | 
 ℙ † X ​ ( B ε ​ ( x ) ) λ D ​ ( U ε , δ ​ ( x ) ) ≤ lim inf t → ∞ sup x ′ ∈ B ε ​ ( x ) p θ t X ​ ( x ′ ) . \dfrac{\mathbb{P}^{X}_{\dagger}\left(B_{\varepsilon}(x)\right)}{\lambda_{D}\left(U_{\varepsilon,\delta}(x)\right)}\leq\liminf_{t\rightarrow\infty}\sup_{x^{\prime}\in B_{\varepsilon}(x)}{p}^{X}_{\theta_{t}}(x^{\prime}). | 
 | 
 (114) | 
 

 Since U ε , δ ′ ​ ( x ) ⊂ U ε , δ ​ ( x ) U_{\varepsilon,\delta^{\prime}}(x)\subset U_{\varepsilon,\delta}(x) whenever δ ′ δ \delta^{\prime} \delta , we have that 

 

 
 | 
 lim δ → 0 + λ D ​ ( U ε , δ ​ ( x ) ) \displaystyle\lim_{\delta\rightarrow 0^{+}}\lambda_{D}\left(U_{\varepsilon,\delta}(x)\right) | 
 = λ D ​ ( ⋂ δ 0 U ε , δ ​ ( x ) ) = λ D ​ ( ⋂ δ 0 ( M δ ∩ B ε ​ ( x ) ) ) = λ D ​ ( ( ⋂ δ 0 M δ ) ∩ B ε ​ ( x ) ) \displaystyle=\lambda_{D}\left(\bigcap_{\delta 0}U_{\varepsilon,\delta}(x)\right)=\lambda_{D}\left(\bigcap_{\delta 0}\left(M_{\delta}\cap B_{\varepsilon}(x)\right)\right)=\lambda_{D}\left(\left(\bigcap_{\delta 0}M_{\delta}\right)\cap B_{\varepsilon}(x)\right) | 
 | 
 (115) | 
 
 
 | 
 | 
 = λ D ​ ( cl 𝒳 ⁡ ( M ) ∩ B ε ​ ( x ) ) = 0 , \displaystyle=\lambda_{D}\left(\cl_{\mathcal{X}}(M)\cap B_{\varepsilon}(x)\right)=0, | 
 | 
 (116) | 
 

 where we used that ( i ) (i) ∩ δ 0 M δ = cl 𝒳 ( M ) \cap_{\delta 0}M_{\delta}=\cl_{\mathcal{X}}(M) , which holds because cl 𝒳 ⁡ ( M ) \cl_{\mathcal{X}}(M) is the set of points which are arbitrarily close to M M , and that ( i ​ i ) (ii) λ D ​ ( cl 𝒳 ⁡ ( M ) ) = 0 \lambda_{D}(\cl_{\mathcal{X}}(M))=0 by assumption. Finally, by 4 , ℙ † X ​ ( B ε ​ ( x ) ) 0 \mathbb{P}^{X}_{\dagger}(B_{\varepsilon}(x)) 0 , so that taking the limit as δ → 0 + \delta\rightarrow 0^{+} on both sides of Equation 114 yields that lim inf t → ∞ sup x ′ ∈ B ε ​ ( x ) p θ t X ​ ( x ′ ) = ∞ \liminf_{t\rightarrow\infty}\sup_{x^{\prime}\in B_{\varepsilon}(x)}{p}^{X}_{\theta_{t}}(x^{\prime})=\infty , which in turn implies that sup x ′ ∈ B ε ​ ( x ) p θ t X ​ ( x ′ ) → ∞ \sup_{x^{\prime}\in B_{\varepsilon}(x)}{p}^{X}_{\theta_{t}}(x^{\prime})\rightarrow\infty as t → ∞ t\rightarrow\infty , finishing the proof.
∎ 

 
 
 
 

### Section B.2 Formalizing the Link between Two-Step Models and Optimal Transport

 
 For convenience, we restate 1 below.

 
 
 See 1 

 
 
 Proof. 
 
 Since 𝒞 ⊂ ℱ \mathcal{C}\subset\mathcal{F} , it follows that 

 

 
 | 
 inf f ∈ ℱ 𝔼 X ∼ ℙ ∗ X ​ [ c ⁡ ( X , g θ 1 ​ ( f ⁡ ( X ) ) ) ] ≤ inf f ∈ 𝒞 𝔼 X ∼ ℙ ∗ X ​ [ c ⁡ ( X , g θ 1 ​ ( f ⁡ ( X ) ) ) ] . \inf_{f\in\mathcal{F}}\mathbb{E}_{X\sim\mathbb{P}^{X}_{*}}\left[c\left(X,g_{\theta_{1}}\left(f(X)\right)\right)\right]\leq\inf_{f\in\mathcal{C}}\mathbb{E}_{X\sim\mathbb{P}^{X}_{*}}\left[c\left(X,g_{\theta_{1}}\left(f(X)\right)\right)\right]. | 
 | 
 (117) | 
 

 Since both infimums are finite due to the assumption from Equation 93 , it is enough to show that, for every ε 0 \varepsilon 0 , there exists f ε ∈ 𝒞 f_{\varepsilon}\in\mathcal{C} such that 

 

 
 | 
 inf f ∈ ℱ 𝔼 X ∼ ℙ ∗ X ​ [ c ⁡ ( X , g θ 1 ​ ( f ⁡ ( X ) ) ) ] 𝔼 X ∼ ℙ ∗ X ​ [ c ⁡ ( X , g θ 1 ​ ( f ε ​ ( X ) ) ) ] − ε . \inf_{f\in\mathcal{F}}\mathbb{E}_{X\sim\mathbb{P}^{X}_{*}}\left[c\left(X,g_{\theta_{1}}\left(f(X)\right)\right)\right] \mathbb{E}_{X\sim\mathbb{P}^{X}_{*}}\left[c\left(X,g_{\theta_{1}}\left(f_{\varepsilon}(X)\right)\right)\right]-\varepsilon. | 
 | 
 (118) | 
 

 Let ε 0 \varepsilon 0 , and let f ∗ ∈ ℱ f_{\ast}\in\mathcal{F} be such that 

 

 
 | 
 inf f ∈ ℱ 𝔼 X ∼ ℙ ∗ X ​ [ c ⁡ ( X , g θ 1 ​ ( f ⁡ ( X ) ) ) ] 𝔼 X ∼ ℙ ∗ X ​ [ c ⁡ ( X , g θ 1 ​ ( f ∗ ​ ( X ) ) ) ] − ε 2 . \inf_{f\in\mathcal{F}}\mathbb{E}_{X\sim\mathbb{P}^{X}_{*}}\left[c\left(X,g_{\theta_{1}}\left(f(X)\right)\right)\right] \mathbb{E}_{X\sim\mathbb{P}^{X}_{*}}\left[c\left(X,g_{\theta_{1}}\left(f_{\ast}(X)\right)\right)\right]-\dfrac{\varepsilon}{2}. | 
 | 
 (119) | 
 

 It is thus enough to show that there exists f ε ∈ 𝒞 f_{\varepsilon}\in\mathcal{C} such that 

 

 
 | 
 𝔼 X ∼ ℙ ∗ X ​ [ c ⁡ ( X , g θ 1 ​ ( f ∗ ​ ( X ) ) ) ] 𝔼 X ∼ ℙ ∗ X ​ [ c ⁡ ( X , g θ 1 ​ ( f ε ​ ( X ) ) ) ] − ε 2 , \mathbb{E}_{X\sim\mathbb{P}^{X}_{*}}\left[c\left(X,g_{\theta_{1}}\left(f_{\ast}(X)\right)\right)\right] \mathbb{E}_{X\sim\mathbb{P}^{X}_{*}}\left[c\left(X,g_{\theta_{1}}\left(f_{\varepsilon}(X)\right)\right)\right]-\dfrac{\varepsilon}{2}, | 
 | 
 (120) | 
 

 as this would imply Equation 118 holds.
Since ℙ ∗ X \mathbb{P}^{X}_{*} is a Borel measure and 𝒳 \mathcal{X} is Polish, ℙ ∗ X \mathbb{P}^{X}_{*} is a Radon measure. Additionally, ℙ ∗ X ​ ( 𝒳 ) ∞ \mathbb{P}^{X}_{*}(\mathcal{X}) \infty , 𝒳 \mathcal{X} is locally compact, 𝒵 \mathcal{Z} is second countable, and f ∗ f_{\ast} is measurable; so it follows by Lusin’s theorem that there exists a Borel set E ⊂ 𝒳 E\subset\mathcal{X} and a continuous function f ε : 𝒳 → 𝒵 f_{\varepsilon}:\mathcal{X}\rightarrow\mathcal{Z} such that f ε ​ ( x ) = f ∗ ​ ( x ) f_{\varepsilon}(x)=f_{\ast}(x) for every x ∈ E x\in E , and ℙ ∗ X ​ ( 𝒳 ∖ E ) ε / ( 4 ​ C ) \mathbb{P}^{X}_{*}(\mathcal{X}\setminus E) \varepsilon/(4C) . Then, we have: 

 

 
 | 
 | 
 𝔼 X ∼ ℙ ∗ X ​ [ c ⁡ ( X , g θ 1 ​ ( f ε ​ ( X ) ) ) ] − 𝔼 X ∼ ℙ ∗ X ​ [ c ⁡ ( X , g θ 1 ​ ( f ∗ ​ ( X ) ) ) ] \displaystyle\mathbb{E}_{X\sim\mathbb{P}^{X}_{*}}\left[c\left(X,g_{\theta_{1}}\left(f_{\varepsilon}(X)\right)\right)\right]-\mathbb{E}_{X\sim\mathbb{P}^{X}_{*}}\left[c\left(X,g_{\theta_{1}}\left(f_{\ast}(X)\right)\right)\right] | 
 | 
 (121) | 
 
 
 | 
 | 
 = ∫ 𝒳 c ⁡ ( x , g θ 1 ​ ( f ε ​ ( x ) ) ) − c ⁡ ( x , g θ 1 ​ ( f ∗ ​ ( x ) ) ) ​ d ​ ℙ ∗ X ​ ( x ) = ∫ 𝒳 ∖ E c ⁡ ( x , g θ 1 ​ ( f ε ​ ( x ) ) ) − c ⁡ ( x , g θ 1 ​ ( f ∗ ​ ( x ) ) ) ​ d ​ ℙ ∗ X ​ ( x ) \displaystyle=\int_{\mathcal{X}}c\left(x,g_{\theta_{1}}\left(f_{\varepsilon}(x)\right)\right)-c\left(x,g_{\theta_{1}}\left(f_{\ast}(x)\right)\right){\textnormal{d}}\mathbb{P}^{X}_{*}(x)=\int_{\mathcal{X}\setminus E}c\left(x,g_{\theta_{1}}\left(f_{\varepsilon}(x)\right)\right)-c\left(x,g_{\theta_{1}}\left(f_{\ast}(x)\right)\right){\textnormal{d}}\mathbb{P}^{X}_{*}(x) | 
 | 
 (122) | 
 
 
 | 
 | 
 ≤ ∫ 𝒳 ∖ E | c ⁡ ( x , g θ 1 ​ ( f ε ​ ( x ) ) ) − c ⁡ ( x , g θ 1 ​ ( f ∗ ​ ( x ) ) ) | ​ d ​ ℙ ∗ X ​ ( x ) ≤ ∫ 𝒳 ∖ E 2 ​ C ​ d ​ ℙ ∗ X ​ ( x ) = 2 ​ C ​ ℙ ∗ X ​ ( 𝒳 ∖ E ) ε 2 , \displaystyle\leq\int_{\mathcal{X}\setminus E}\left|c\left(x,g_{\theta_{1}}\left(f_{\varepsilon}(x)\right)\right)-c\left(x,g_{\theta_{1}}\left(f_{\ast}(x)\right)\right)\right|{\textnormal{d}}\mathbb{P}^{X}_{*}(x)\leq\int_{\mathcal{X}\setminus E}2C\,{\textnormal{d}}\mathbb{P}^{X}_{*}(x)=2C\,\mathbb{P}^{X}_{*}(\mathcal{X}\setminus E) \dfrac{\varepsilon}{2}, | 
 | 
 (123) | 
 

 which in turn implies Equation 120 holds, thus finishing the proof.
∎ 

 
 
 
 
 

## Appendix C Experimental Details

 
 Here we provide details on the experiments from Section 5.3.2 .
Our code is available at https://github.com/layer6ai-labs/dgm_manifold_survey .

 
 
 First-step objective for latent diffusion models 

 
 Following Rombach et al. (2022) , we trained latent diffusion models by first using a regularized variational autoencoder ( Section 4.1.2 ) loss ( Larsen et al., 2016 ; Higgins et al., 2017 ) :

 

 
 | 
 min θ 1 , ϕ ⁡ max ϕ ′ 𝔼 X ∼ p ∗ X [ 𝔼 Z ∼ q Z | X ϕ ( ⋅ | X ) [ ∥ X − g θ 1 ( Z ) ∥ 2 2 ] ] + β 1 𝕂 𝕃 ( q ϕ Z | X ( ⋅ | X ) ∥ p Z ) + β 2 𝔼 X ∼ p ∗ X [ 𝔼 Z ∼ q Z | X ϕ ( ⋅ | X ) [ log h ϕ ′ ( X ) + log ( 1 − h ϕ ′ ( g θ 1 ( Z ) ) ] ] , \displaystyle\begin{split}\min_{\theta_{1},\phi}\max_{\phi^{\prime}}\ \mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[\mathbb{E}_{Z\sim q^{Z|X}_{\phi}(\cdot|X)}\left[\|X-g_{\theta_{1}}(Z)\|_{2}^{2}\right]\right]+\beta_{1}\mathbb{KL}\left(q_{\phi}^{Z|X}(\cdot|X)\,\|\,{p}^{Z}\right)\\
 +\beta_{2}\mathbb{E}_{X\sim{p}^{X}_{\ast}}\left[\mathbb{E}_{Z\sim q^{Z|X}_{\phi}(\cdot|X)}\left[\log h_{\phi^{\prime}}(X)+\log\left(1-h_{\phi^{\prime}}(g_{\theta_{1}}(Z)\right)\right]\right],\end{split} | 
 | 
 (124) | 
 

 where β 1 0 \beta_{1} 0 and β 2 0 \beta_{2} 0 are hyperparameters; p Z {p}^{Z} is a standard Gaussian; q ϕ Z | X ( ⋅ | x ) = 𝒩 ( ⋅ ; f ϕ ( x ) , Σ ϕ Z | X ( x ) ) q^{Z|X}_{\phi}(\cdot|x)=\mathcal{N}(\ \cdot\ ;f_{\phi}(x),\Sigma^{Z|X}_{\phi}(x)) with Σ ϕ Z | X ​ ( x ) \Sigma^{Z|X}_{\phi}(x) being diagonal for every x ∈ 𝒳 x\in\mathcal{X} ; and h ϕ ′ : 𝒳 → ( 0 , 1 ) h_{\phi^{\prime}}:\mathcal{X}\rightarrow(0,1) is a binary classifier inspired by generative adversarial networks ( Section 4.2 ), whose objective is to distinguish between real samples X ∼ p ∗ X X\sim{p}^{X}_{\ast} and their stochastic reconstructions g θ 1 ​ ( Z ) g_{\theta_{1}}(Z) , where Z ∼ q ϕ Z | X ( ⋅ | X ) Z\sim q_{\phi}^{Z|X}(\cdot|X) . Rather than fixing β 2 \beta_{2} , we dynamically update it throughout training to ensure that the first and third terms in Equation 124 have roughly the same magnitude; when taking a gradient step, this is achieved by computing the ratio of the values of the first to third term in the previous gradient step, and setting β 2 \beta_{2} to the absolute value of this ratio.

 
 
 
 Training objective for diffusion models 

 
 We train all diffusion models, latent or not, exactly as described in Section 5.1.2 , i.e. through Equation 55 . The integral with respect to t t is approximated by sampling t t uniformly at random in [ 0 , T ] [0,T] during training (one such t t is sampled for every element in the batch).

 
 
 
 Hyperparameters 

 
 We use the Adam optimizer ( Kingma Ba, 2015 ) with a batch size of 128 128 throughout, and train all models until there is no improvement on the validation metric for 50 50 epochs; we keep the models with the best validation performance. For the VAE of latent diffusion models we use the squared reconstruction error 𝔼 X ∼ p ∗ X ​ [ ‖ X − g θ 1 ​ ( f ϕ ​ ( X ) ) ‖ 2 2 ] \mathbb{E}_{X\sim{p}^{X}_{\ast}}[\|X-g_{\theta_{1}}(f_{\phi}(X))\|_{2}^{2}] rather than Equation 124 as the validation metric, β 1 = 10 − 6 \beta_{1}=10^{-6} , a learning rate of 10 − 4 10^{-4} with cosine annealing, and take two gradient steps on ϕ ′ \phi^{\prime} for every gradient step on ( θ 1 , ϕ ) (\theta_{1},\phi) . For the diffusion models, both on ambient and latent space, we use Equation 55 as the validation metric, a learning rate of 5 × 10 − 5 5\times 10^{-5} without cosine annealing, T = 1 T=1 , β min = 0.1 \beta_{\text{min}}=0.1 , β max = 20 \beta_{\text{max}}=20 , w ⁡ ( t ) = σ t 2 w(t)=\sigma_{t}^{2} , and an Euler-Maruyama discretization scheme with 1000 1000 steps to generate the paths in Figure 8 .

 
 
 
 Architectures 

 
 For a fair comparison between diffusion models on ambient and latent space, we attempt to instantiate them in such a way that their overall architectures are as similar as possible. The configurations of the architectures we used are given in Table 1 , which we now describe. We parameterize both the ambient and latent score networks as the output of a neural network divided by σ t \sigma_{t} , as mentioned in Section 5.1.2 (this is equivalent to the so-called “ ε \varepsilon parameterization” of Ho et al. (2020) with − ε -\varepsilon being parameterized instead of ε \varepsilon ). For the diffusion model on ambient space, we use a U-Net architecture ( Ronneberger et al., 2015 ) with residual connections ( He et al., 2016 ) and an additional attention mechanism ( Vaswani et al., 2017 ) , as implemented in the labml package ( Jayasiri Wijerathne, 2020 ) , which uses a sinusoidal positional embedding for the scalar input t t . The ambient space U-Net takes 3 × 32 × 32 3\times 32\times 32 images, and progressively downsamples them to a shape of 1024 × 4 × 4 1024\times 4\times 4 before upscaling them back to their original size. For the latent diffusion model, we attempt to copy the aforementioned U-Net as much as possible; the VAE mimics the U-Net up until the 8 × 8 8\times 8 resolution, and adds a convolutional layer to produce outputs of shape 4 × 8 × 8 4\times 8\times 8 , so that d = 256 d=256 . To ensure the VAE obtains low-dimensional representations through a proper bottleneck, we remove the U-Net skip connections between the encoder and decoder, resulting in a purely residual architecture. The stochastic encoder q ϕ Z | X q^{Z|X}_{\phi} of the VAE consists of a single neural network which takes x ∈ 𝒳 x\in\mathcal{X} and produces a 2 ​ d 2d -dimensional output – the first d d dimensions correspond to f ϕ ​ ( x ) f_{\phi}(x) , and the remaining ones to the diagonal of Σ ϕ Z | X ​ ( x ) \Sigma^{Z|X}_{\phi}(x) . The auxiliary network h ϕ ′ h_{\phi^{\prime}} has the same architecture as the encoder f ϕ f_{\phi} , except a final linear layer is added to ensure the output is a scalar. The score network of the diffusion on latent space is given another U-Net which further downsamples to a 4 × 4 4\times 4 resolution before upsampling.

 
 
 
 Data preprocessing 

 
 Recall that raw image data is integer-valued, with possible values ranging from 0 0 to 255 255 . We dequantize the data before training the diffusion model on ambient space and the VAE for the latent diffusion model, i.e. we add independent uniform [ 0 , 1 ] [0,1] noise to every pixel ( Theis et al., 2016 ) , so that the resulting data now has entries in [ 0,256 ] [0,256] . We then linearly scale the data so that every coordinate lies in [ − 1 , 1 ] [-1,1] . For latent diffusion models, once the VAE is trained, we also scale its encodings to lie in [ − 1 , 1 ] d [-1,1]^{d} before training the diffusion model on latent space.

 
 
 Table 1: Configuration used for neural networks. The architecture of the auxiliary network h ϕ ′ h_{\phi^{\prime}} for the VAE mimics that of the encoder, and the decoder g θ 1 g_{\theta_{1}} is given by simply “reversing” the architecture of the encoder. See text for additional details. 
 
 
 
 PARAMETER | 
 AMBIENT SCORE s ^ θ X \hat{s}_{\theta}^{X} | 
 VAE ENCODER q ϕ Z | X q^{Z|X}_{\phi} | 
 LATENT SCORE s ^ θ 2 Z \hat{s}_{\theta_{2}}^{Z} | 

 
 
 
 n_channels | 
 64 64 | 
 64 64 | 
 256 256 | 

 
 ch_mults | 
 ( 1 , 2 , 2 , 4 ) (1,2,2,4) | 
 ( 1 , 2 , 2 ) (1,2,2) | 
 ( 1 , 2 ) (1,2) | 

 
 is_attn | 
 ( False , False , True , True ) (\texttt{False},\texttt{False},\texttt{True},\texttt{True}) | 
 - | 
 ( True , True ) (\texttt{True},\texttt{True}) | 

 
 n_blocks | 
 2 2 | 
 2 2 | 
 2 2 |