Training Data Influence Analysis and Estimation: A Survey 
 
 
 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: CC BY 4.0
 
 
arXiv:2212.04612v3 [cs.LG] 29 Mar 2024 
 
 
 \MHInternalSyntaxOn \MHInternalSyntaxOff 
 

# Training Data Influence Analysis 
 and Estimation: A Survey

 
 
 Zayd Hammoudeh
 
 Note: Correspondence to zayd@cs.uoregon.edu .
Work primarily done while at the University of Oregon.
This paper is published in journal Machine Learning ˜ [ HL24c ] . 
 
 Affiliation: University of Oregon 
 
 Affiliation: Qualtrics AI 
 
    
 Daniel Lowd
 
 Affiliation: University of Oregon 
 

 Abstract 
 
 Good models require good training data.
For overparameterized deep models, the causal relationship between training data and model predictions is increasingly opaque and poorly understood.
Influence analysis partially demystifies training’s underlying interactions by quantifying the amount each training instance alters the final model.
Measuring the training data’s influence exactly can be provably hard in the worst case;
this has led to the development and use of influence estimators, which only approximate the true influence.
This paper provides the first comprehensive survey of training data influence analysis and estimation.
We begin by formalizing the various, and in places orthogonal, definitions of training data influence.
We then organize state-of-the-art influence analysis methods into a taxonomy;
we describe each of these methods in detail and compare their underlying assumptions, asymptotic complexities, and overall strengths and weaknesses.
Finally, we propose future research directions to make influence analysis more useful in practice as well as more theoretically and empirically sound.
A curated, up-to-date list of resources related to influence analysis is available at https://github.com/ZaydH/influence_analysis_papers . 

 
 
 
 Keywords : Influence analysis,
influence estimation,
training data attribution,
data valuation,
influence functions,
TracIn,
Shapley value 

 
 

## 1 Introduction

 
 Machine learning is built on training data [ Red+21a ] .
Without good training data, nothing else works.
How modern models learn from and use training data is increasingly opaque [ KL17a , Zha+21c , Xia22a ] .
Regarding state-of-the-art black-box models, [ Yam20a ] notes, “If all we have is a ‘black box’ it is impossible to understand causes of failure and improve system safety.” 

 
 
 Large modern models require tremendous amounts of training data [ BF21a ] .
Today’s uncurated, internet-derived datasets
commonly contain numerous anomalous instances [ Ple+20a ] .
These anomalies can arise from multiple potential sources.
For example, training data anomalies may have a natural cause such as
distribution shift [ RL87a , Yan+21a ] ,
measurement error,
or
non-representative samples drawn from the tail of the data distribution
 [ Hub81a , Fel20c , Jia+21c ] .
Anomalous training instances also occur due to human or algorithmic labeling errors – even on well-known, highly-curated datasets [ EGH17a ] .
Malicious adversaries can insert anomalous poison instances into the training data with the goal of manipulating specific model predictions [ BNL12a , Che+17a , Sha+18b , HL23a ] .
Regardless of the cause, anomalous training instances degrade a model’s overall generalization performance. 

 
 
 Today’s large datasets
also generally overrepresent established and dominant viewpoints [ Ben+21a ] .
Models trained on these huge public datasets encode and exhibit biases based on protected characteristics, including gender, race, religion, and disability [ BCC19a , Kur+19a , TC19a , Zha+20a , Hut+20a ] .
These training data biases can translate into real-world harm, where, as an example, a recidivism model falsely flagged black defendants as high risk at twice the rate of white defendants [ Ang+16a ] . 

 
 
 Understanding the data and its relationship to trained models is essential for building trustworthy ML systems.
However, it can be very difficult to answer even basic questions about the relationship between training data and model predictions; for example: 

 
 1. 
 
 Is a prediction well-supported by the training data, or was the prediction just random? 

 

 2. 
 
 Which portions of the training data improve a prediction? Which portions make it worse? 

 

 3. 
 
 Which instances in the training set caused the model to make a specific prediction? 

 

 
 
 
 One strategy to address basic questions like those above is to render them moot by exclusively using simple, transparent model classes [ Lip18a ] .
Evidence exists that this “interpretable-only” strategy may be appropriate in some settings [ Kni17a ] .
However, even interpretable model classes can be grossly affected by training data issues [ Hub81a , CHW82a , CW82a ] .
Moreover, as the performance penalty of interpretable models grows, their continued use becomes harder to justify. 

 
 
 With the growing use of black-box models,
we need better methods to analyze and understand black-box model decisions.
Otherwise, society must carry the burden of black-box failures. 

 
 

### 1.1 Relating Models and Their Training Data

 
 All model decisions are rooted in the training data. Training data influence analysis (also known as data valuation [ GZ19a , Jia+19b , KCC23a ] and data attribution [ Par+23a , NSO23a , DG23a ] ) partially demystifies the relationship between training data and model predictions by determining how to apportion credit (and blame) for specific model behavior to the training instances [ SR88a , KL17a , Yeh+18a , Pru+20a ] .
Essentially, influence analysis’s objective is to answer the question: What is each training instance’s effect on a model ?
An instance’s “effect” is with respect to some specific perspective.
For example, an instance’s effect may be quantified as the change in model performance when some instance is deleted from the training data. The effect can also be relative, e.g., whether one training instance changes the model more than another. 

 
 
 Influence analysis emerged alongside the initial study of linear models and regression [ Jae72a , CW82a ] .
This early analysis focused on quantifying how worst-case perturbations to the training data affected the final model parameters.
The insights gained from early influence analysis contributed to the development of numerous methods that improved model robustness and reduced model sensitivity to training outliers [ Hog79a , Rou94a ] . 

 
 
 Since these early days, machine learning models have grown substantially in complexity and opacity [ Dev+19a , KSH12a , Dos+21a ] .
Training datasets have also exploded in size [ BF21a ] .
These factors combine to make training data influence analysis significantly more challenging where, for multilayer parametric models (e.g., neural networks), determining a single training instance’s exact effect can be NP - complete in the worst case [ BR92a ] . 

 
 
 In practice, influence may not need to be measured exactly.
 Influence estimation methods provide an approximation of training instances’ true influence.
Influence estimation is generally much more computationally efficient and is now the approach of choice [ Sch+22a ] .
However, modern influence estimators achieve their efficiency via various assumptions about the model’s architecture and learning environment [ KL17a , Yeh+18a , GZ19a ] .
These varied assumptions result in influence estimators having different advantages and disadvantages as well as in some cases, even orthogonal perspectives on the definition of influence itself [ Pru+20a ] . 

 
 
 

### 1.2 Our Contributions

 
 To the extent of our knowledge, there has not yet been a comprehensive review of these differing perspectives of training data influence, much less of the various methods themselves.
This paper fills in that gap by providing the first comprehensive survey of existing influence analysis techniques.
We describe how these various methods overlap and, more importantly, the consequences – both positive and negative – that arise out of their differences.
We provide this broad and nuanced understanding of influence analysis so that ML researchers and practitioners can better decide which influence analysis method best suits their specific application objectives [ Sch+22a ] . 

 
 
 Although we aim to provide a comprehensive survey of influence analysis, we cannot cover every method in detail.
Instead, we focus on the most impactful methods so as not to distract from the key takeaways.
In particular, we concentrate on influence analysis methods that are general [ CW82a , GZ19a , FZ20a ] or targeted towards parametric models [ KL17a , Yeh+18a , Pru+20a , Che+21a ] with less emphasis on non-parametric methods [ Sha+18c , Jia+19b , BHL23a ] .
Multiple other research areas are based on ranking and subsampling training instances including data pruning [ Yan+23a ] , coreset selection [ BLK17a , Fel20b ] , active learning [ Ren+21a ] , and submodular dataset selection [ WIB15a ] , but these topics are beyond the scope of this work. 1 1 
 1 
 
 
 
 Section 3.3 briefly contrasts how the objectives of these related areas align with influence analysis’s objectives. 

 
 
 In the remainder of this paper, we first standardize the general notation used throughout this work (Sec. 2 ).
Section 3 reviews the various general formulations through which training data influence is viewed.
We also categorize and summarize the properties of the seven most impactful influence analysis methods.
Sections 4 and 5 describe these foundational influence methods in detail.
For each method, we (1) formalize the associated definition of influence and how it is measured, (2) detail the formulation’s strengths and weaknesses, (3) enumerate any related or derivative methods, and (4) explain the method’s time, space, and storage complexities.
Section 6 reviews various learning tasks where influence analysis has been applied.
We provide our perspective on future directions for influence analysis research in Section 7 . 

 
 
 
 

## 2 General Notation

 
 This section details our primary notation.
In cases where a single influence method requires custom nomenclature, we introduce the unique notation alongside discussion of that method. 2 2 
 2 
 
 
 
 Supplemental Table 3 provides a reference for all notation specific to a single influence analysis method. 
Supplemental Section A provides a full nomenclature reference. 

 
 
 Let [ r ] {[r]} denote the set of integers { 1 ; … ; r } \{1\mathchar 59\relax\ldots\mathchar 59\relax r\} .
 A ∼ m B {A\stackrel{{\scriptstyle m}}{{\sim}}B} denotes that the cardinality of set A A is m m and that A A is drawn uniformly at random (u.a.r.) from set B B .
For singleton set A A (i.e., | A | = 1 {\lvert A\rvert=1} ), the sampling notation is simplified to A ∼ B {A\sim B} .
Let 2 A 2^{A} denote the power set of any set A A .
Set subtraction is denoted A ∖ B {A\setminus B} .
For singleton B = { b } {B=\{b\}} , set subtraction is simplified to A ∖ b {A\setminus b} . 

 
 
 The zero vector is denoted 0 → \vec{0} with the vector’s dimension implicit from context.
 𝟙 ​ [ a ] {\mathbbm{1}[a]} is the indicator function , where 𝟙 ​ [ a ] = 1 {{\mathbbm{1}[a]}=1} if predicate a a is true and 0 otherwise. 

 
 
 Let x ∈ 𝒳 ⊆ d {x\in\mathcal{X}\subseteq\real^{d}} denote an arbitrary feature vector , and let y ∈ 𝒴 {y\in\mathcal{Y}} be a dependent value (e.g., label, target).
 Training set , 𝒟 ≔ { z i } i = 1 n {\mathcal{D}\coloneqq\{z_{i}\}_{i=1}^{n}} , consists of n n training instances where each instance is a tuple, z i ≔ ( x i ; y i ) ∈ 𝒵 {z_{i}\coloneqq(x_{i}\mathchar 59\relax y_{i})\in\mathcal{Z}} and 𝒵 ≔ 𝒳 × 𝒴 {\mathcal{Z}\coloneqq\mathcal{X}\times\mathcal{Y}} .
(Arbitrary) test instances are denoted z te ≔ ( x te ; y te ) ∈ 𝒵 {z_{\text{te}}\coloneqq(x_{\text{te}}\mathchar 59\relax y_{\text{te}})\in\mathcal{Z}} .
Note that y te y_{\text{te}} need not be x te x_{\text{te}} ’s true dependent value; y te y_{\text{te}} can be any value in 𝒴 \mathcal{Y} .
Throughout this work, subscripts “ i i ” and “ te ” entail that the corresponding symbol applies to an arbitrary training and test instance, respectively. 

 
 
 Model f : 𝒳 → 𝒴 {f:\mathcal{X}\rightarrow\mathcal{Y}} is parameterized by θ ∈ p {\theta\in\real^{p}} , where p ≔ | θ | {p\coloneqq\lvert\theta\rvert} ; f f is trained on (a subset of) dataset 𝒟 \mathcal{D} .
Most commonly, f f performs either classification or regression, although more advanced model classes (e.g., generative models) are also considered.
Model performance is evaluated using a loss function ℓ : 𝒴 × 𝒴 → {\ell:\mathcal{Y}\times\mathcal{Y}\rightarrow\real} .
Let ℒ ⁡ ( z , θ ) ≔ ℓ ⁡ ( f ⁡ ( x , θ ) , y ) {{\mathcal{L}(z;\theta)}\coloneqq\ell\big({f(x;\theta)}\mathchar 59\relax y\big)} denote the empirical risk of instance z = ( x , y ) {z=(x\mathchar 59\relax y)} w.r.t. parameters θ \theta .
By convention, a smaller risk is better. 

 
 
 This work primarily focuses on overparameterized models with p ≫ d {p\gg d} , where d d is the data dimension.
Such models are almost exclusively trained using first-order optimization algorithms (e.g., gradient descent), which proceed iteratively over T {T} iterations.
Starting from initial parameters θ ( 0 ) {\theta^{(0)}} , the optimizer returns at the end of each iteration, t ∈ [ T ] {t\in{[T]}} , updated model parameters θ ( t ) \theta^{(t)} , where θ ( t ) \theta^{(t)} is generated from previous parameters θ ( t − 1 ) \theta^{(t-1)} , loss function ℓ \ell , batch ℬ ( t ) ⊆ 𝒟 {\mathcal{B}^{(t)}\subseteq\mathcal{D}} , learning rate η ( t ) 0 {\eta^{(t)} 0} , and weight decay ( L 2 L_{2} ) strength λ ≥ 0 {\lambda\geq 0} .
Training gradients are denoted
 ∇ θ ℒ ​ ( z i , θ ( t ) ) {\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(t)})}} .
The training set’s empirical risk Hessian for iteration t t is denoted H θ ( t ) ≔ 1 n ​ ∑ z i ∈ 𝒟 ∇ θ 2 ​ ℒ ​ ( z i , θ ( t ) ) {H_{\theta}^{(t)}\coloneqq\frac{1}{n}\sum_{z_{i}\in\mathcal{D}}\nabla_{\theta}^{2}{\mathcal{L}(z_{i};\theta^{(t)})}} , with the corresponding inverse risk Hessian denoted ( H θ ( t ) ) − 1 {(H_{\theta}^{(t)})^{-1}} .
Throughout this work, superscript “ ( t ) {(t)} ” entails that the corresponding symbol applies to training iteration t t . 

 
 
 Some models may be trained on data other than full training set 𝒟 \mathcal{D} , e.g., subset 𝒟 ∖ z i {\mathcal{D}\setminus z_{i}} .
Let D ⊂ 𝒟 {D\subset\mathcal{D}} denote an alternate training set, and denote model parameters trained on D D as θ D ( t ) \theta^{(t)}_{D} .
For example, θ 𝒟 ∖ z i ( T ) {\theta^{(T)}_{\mathcal{D}^{\setminus z_{i}}}} are the final parameters for a model trained on all of 𝒟 \mathcal{D} except training instance z i ∈ 𝒟 {z_{i}\in\mathcal{D}} .
When training on all of the training data, subscript 𝒟 \mathcal{D} is dropped, i.e., θ ( t ) ≡ θ 𝒟 ( t ) {\theta^{(t)}\equiv\theta^{(t)}_{\mathcal{D}}} . 

 
 
 

## 3 Overview of Influence and Influence Estimation

 
 As Section 1 explains, training data influence’s objective is to quantify the “effect” of one or more training instances on a model.
This effect’s scope can be as localized as an individual model prediction, e.g., f ⁡ ( x te , θ ( T ) ) {f(x_{\text{te}};\theta^{(T)})} ;
the effect’s scope can also be so broad as to encompass the entire test data distribution. 

 
 
 Positive influence entails that the training instance(s) improve some quality measure, e.g., risk ℒ ⁡ ( z te , θ ( T ) ) {\mathcal{L}(z_{\text{te}};\theta^{(T)})} .
Negative influence means that the training instance(s) make the quality measure worse.
Training instances with positive influence are referred to as proponents or excitatory examples .
Training instances with negative influence are called opponents or inhibitory examples [ KL17a , Yeh+18a ] . 

 
 
 Highly expressive, overparameterized models remain functionally black boxes [ KL17a ] .
Understanding why a model behaves in a specific way remains a significant challenge [ BP21a ] ,
and
the inclusion or removal of even a single training instance can drastically change a trained model’s behavior [ Rou94a , BF21a ] .
In the worst case, quantifying one training instance’s influence may require repeating all of training. 

 
 
 Since measuring influence exactly may be intractable or unnecessary, influence estimators – which only approximate the true influence – are commonly used in practice.
As with any approximation, influence estimation requires making trade-offs, and the various influence estimators balance these design choices differently.
This in turn leads influence estimators to make different assumptions and rely on different mathematical formulations. 

 
 
 When determining which influence analysis methods to highlight in this work, we relied on two primary criteria: (1) a method’s overall impact and (2) the method’s degree of novelty in relation to other approaches.
In particular, we concentrate on influence analysis methods that are either model architecture agnostic or that are targeted towards parametric models (e.g., neural networks).
Nonetheless, we briefly discuss non-parametric methods as well. 

 
 
 The remainder of this section considers progressively more general definitions of influence. 

 
 

### 3.1 Pointwise Training Data Influence

 
 Pointwise influence is the simplest and most commonly studied definition of influence.
It quantifies how a single training instance affects a model’s prediction on a single test instance according to some quality measure (e.g., test loss).
Formally, a pointwise influence analysis method is a function ℐ : 𝒵 × 𝒵 → {\mathcal{I}:\mathcal{Z}\times\mathcal{Z}\rightarrow\real} with the pointwise influence of training instance z i z_{i} on test instance z te z_{\text{te}} denoted ℐ ⁡ ( z i , z te ) {\mathcal{I}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)} .
Pointwise influence estimates are denoted ℐ ^ ​ ( z i , z te ) ∈ {{\widehat{\mathcal{I}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\in\real} .
Note that the model architecture, training algorithm, full training set ( 𝒟 \mathcal{D} ), and even the random seed can (significantly) affect an instance’s influence.
To improve clarity and readability, we treat these parameters as fixed and implicit in our nomenclature for ℐ \mathcal{I} and ℐ ^ \widehat{\mathcal{I}} . 

 
 
 Below, we briefly review early pointwise influence analysis contributions and then transition to a discussion of more recent pointwise methods. 

 
 

#### 3.1.1 Early Pointwise Influence Analysis

 
 
 Inlier Outlier Least Squares Inliers Only Least Squares All 1 1 2 2 3 3 4 4 5 5 1 1 2 2 3 3 4 4 x x y y 
 Figure 1: Outlier Pointwise Influence on Least-Squares Regression :
Influence of a single outlier
( 1 )
on a least-squares model where in-distribution data
( 1 )
are generated from
linear distribution y = 2 ​ x {y=2x} .
The single outlier sample
( x = 5 {x=5} y = 1.2 {y=1.2} )
influences the inlier-only least-squares linear model
( 1 )
substantially such that a least-squares model trained on all instances
( 1 )
predicts all training y y values poorly.
Adapted from [ RL87a , Fig. 2(b)] .
 
 
 
 The earliest notions of pointwise influence emerged out of robust statistics – specifically the analysis of training outliers’ effects on linear regression models [ Coo77a ] .
Given training set 𝒟 \mathcal{D} , the least-mean squares linear model parameters are 3 3 
 3 
 
 
 
 For simplicity, the bias term is considered part of θ \theta . 

 

 
 | 
 θ ∗ ≔ arg ​ min θ ⁡ 1 | 𝒟 | ​ ∑ ( x i ​ ; ​ y i ) ∈ 𝒟 ( y i − θ ⊺ ​ x i ) 2 ​ . \theta^{*}\coloneqq\argmin_{\theta}\frac{1}{\lvert\mathcal{D}\rvert}\sum_{(x_{i}\mathord{\mathchar 59\relax}y_{i})\in\mathcal{D}}\left(y_{i}-\theta^{\intercal}x_{i}\right)^{2}\text{.} | 
 | 
 (1) | 
 

 Observe that this least-squares estimator has a breakdown point of 0 [ Rou94a ] .
This means that least-squares regression is completely non-robust where a single training data outlier can shift model parameters θ ∗ \theta^{*} arbitrarily.
For example, Figure 1 visualizes how a single training data outlier ( 1 ) can induce a nearly orthogonal least-squares model.
Put simply, an outlier training instance’s potential pointwise influence on a least-squares model is unbounded.
Unbounded influence on the model parameters equates to unbounded influence on model predictions. 

 
 
 Early influence analysis methods sought to identify the training instance that was most likely to be an outlier [ Sri61a , TMB73a ] .
A training outlier can be defined as the training instance with the largest negative influence on prediction f ⁡ ( x te , θ ∗ ) {f(x_{\text{te}};\theta^{*})} .
Intuitively, each training instance’s pointwise influence can be measured by training n ≔ | 𝒟 | {n\coloneqq\lvert\mathcal{D}\rvert} models, where each model’s training set leaves out a different training instance.
These n n models would then be compared to identify the outlier. 4 4 
 4 
 
 
 
 Section 4.1 formalizes how to measure pointwise influence by repeatedly retraining with a different training instance left out of the training set each time. 
However, such repeated retraining is expensive
and so more efficient pointwise influence analysis methods were studied. 

 
 
 Different assumptions about the training data distribution lead to different definitions of the most likely outlier.
For example, [ Sri61a ] , [ SC68a ] , and [ Ell76a ] all assume that training data outliers arise from mean shifts in normally distributed training data.
Under this constraint, their methods all identify the maximum likelihood outlier as the training instance with the largest absolute residual , | y i − θ ∗ ⊺ ​ x i | \lvert y_{i}-{\theta^{*}}^{\intercal}x_{i}\rvert .
However, [ CW82a ] prove that under different distributional assumptions (e.g., a variance shift instead of a mean shift), the maximum likelihood outlier may not have the largest residual. 

 
 
 These early influence analysis results demonstrating least-squares fragility spurred development of more robust regressors.
For instance, [ Rou94a ] replaces Eq. ( 1 )’s mean operation with median;
this simple change increases the breakdown point of model parameters θ ∗ \theta^{*} to the maximum value, 50%.
In addition, multiple robust loss functions have been proposed that constrain or cap outliers’ pointwise influence [ Hub64a , BT74a , JW78a , Lec89a ] . 

 
 
 As more complex models grew in prevalence, influence analysis methods similarly grew in complexity.
In recent years, numerous influence analysis methods targeting deep models have been proposed.
We briefly review the most impactful, modern pointwise influence analysis methods next. 

 
 
 

#### 3.1.2 Modern Pointwise Influence Analysis

 
 

 {forest} 
 
 

 Figure 2: Influence Analysis Taxonomy :
Categorization of the seven primary pointwise influence analysis methods.
Section 4 details the three primary retraining-based influence methods,
leave-one-out (Sec. 4.1 ),
 Downsampling (Sec. 4.2 ),
and
Shapley value (Sec. 4.3 ).
Section 5 details gradient-based static estimators
influence functions (Sec. 5.1.1 )
and
representer point (Sec. 5.1.2 )
as well as dynamic estimators
TracIn (Sec. 5.2.1 ) and
 HyDRA (Sec. 5.2.2 ).
Closely-related and derivative estimators are shown as a list below their parent method.
See supplemental Table 4 for the formal mathematical definition of all influence methods and estimators.
Due to space, each method’s citation is in supplemental Table 5 .
 
 
 
 Figure 2 provides a taxonomy of the seven most impactful modern pointwise influence analysis methods.
Below each method appears a list of closely related and derivative approaches.
Modern influence analysis methods broadly categorize into two primary classes, namely: 

 
 • 
 
 Retraining-Based Methods : Measure the training data’s influence by repeatedly retraining model f f using different subsets of training set 𝒟 \mathcal{D} . 

 

 • 
 
 Gradient-Based Influence Estimators : Estimate influence via the alignment of training and test instance gradients either throughout or at the end of training. 

 

 
 An in-depth comparison of these influence analysis methods requires detailed analysis so we defer the extensive discussion of these two categories to Sections 4 and 5 , respectively. Table summarizes the key properties of Figure 2 ’s seven methods – including comparing each method’s assumptions (if any), strengths/weaknesses, and asymptotic complexities.
These three criteria are also discussed when detailing each of these methods in the later sections. 

 
 
 
 

### 3.2 Alternative Perspectives on Influence

 
 Note that pointwise effects are only one perspective on how to analyze the training data’s influence.
Below we briefly summarize six alternate, albeit less common, perspectives of training data influence.
While pointwise influence is this work’s primary focus, later sections also contextualize existing influence methods w.r.t. these alternate perspectives where applicable. 

 
 
 (1) Recall that pointwise influence quantifies the effect of a single training instance on a single test prediction.
In reality, multiple related training instances generally influence a prediction as a group [ FZ20a ] , where group members
have a total effect much larger than the sum of their individual effects [ BYF20a , Das+21a , HL22a ] .
 Group influence quantifies a set of training instances’ total, combined influence on a specific test prediction. 

 
 
 We use very similar notation to denote group and pointwise influence.
The only difference is that for group influence, the first parameter of function ℐ \mathcal{I} is a training (sub)set instead of an individual training instance;
the same applies to group influence estimates ℐ ^ \widehat{\mathcal{I}} .
For example, given some test instance z te z_{\text{te}} , the entire training set’s group influence and group influence estimate are denoted ℐ ⁡ ( 𝒟 , z te ) {\mathcal{I}\left(\mathcal{D}\mathchar 59\relax z_{\text{te}}\right)} and ℐ ^ ​ ( 𝒟 , z te ) {\widehat{\mathcal{I}}\left(\mathcal{D}\mathchar 59\relax z_{\text{te}}\right)} , respectively. 

 
 
 In terms of magnitude,
previous work has shown that the group influence of a related set of training instances is generally lowered bounded by the sum of the set’s pointwise influences [ Koh+19a ] .
Put simply, the true influence of a coherent group is more than the sum of its parts , or formally, for coherent D ⊆ 𝒟 {D\subseteq\mathcal{D}} , it often holds that 

 

 
 | 
 | ℐ ⁡ ( D , z te ) | ∑ z i ∈ D | ℐ ⁡ ( z i , z te ) | ​ . \lvert{\mathcal{I}\left(D\mathchar 59\relax z_{\text{te}}\right)}\rvert \sum_{z_{i}\in D}\lvert{\mathcal{I}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\rvert\text{.} | 
 | 
 (2) | 
 

 Existing work studying group influence is limited.
Later sections note examples where any of the seven primary pointwise influence methods have been extended to consider group effects. 

 
 
 (2) Joint influence extends influence to consider multiple test instances collectively [ Jia+22a , Che+22a ] .
These test instances may be a specific subpopulation within the test distribution – for example in targeted data poisoning attacks [ Jag+21a , Wal+21a ] .
The test instances could also be a representative subset of the entire test data distribution – for example in coreset selection [ BMK20a ] or indiscriminate poisoning attacks [ BNL12a , Fow+21a ] . 

 
 
 Most (pointwise) influence analysis methods are additive meaning for target set 𝒟 te ⊆ 𝒵 {\mathcal{D}_{\text{te}}\subseteq\mathcal{Z}} , the joint (pointwise) influence simplifies to 

 

 
 | 
 ℐ ⁡ ( z i , 𝒟 te ) = ∑ z te ∈ 𝒟 te ℐ ⁡ ( z i , z te ) ​ . {\mathcal{I}\left(z_{i}\mathchar 59\relax\mathcal{D}_{\text{te}}\right)}=\sum_{z_{\text{te}}\in\mathcal{D}_{\text{te}}}{\mathcal{I}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\text{.} | 
 | 
 (3) | 
 

 Additivity is not a requirement of influence analysis, and there are provably non-additive influence estimators [ YP21a ] . 

 
 
 (3) Overparameterized models like deep networks are capable of achieving near-zero training loss in most settings [ Bar+20a , Fel20c , DAm+20a ] .
This holds even if the training set is large and randomly labeled [ Zha+17a , Arp+17a ] .
Near-zero training loss occurs because deep models often memorize some training instances. 

 
 
 Both [ Pru+20a ] and [ FZ20a ] separately define a model’s memorization 5 5 
 5 
 
 
 
 [ Pru+20a ] term “memorization” as self-influence . We use [ FZ20a ] ’s [ FZ20a ] terminology here since it is more consistent with other work [ vW21a , KWR22a ] . of training instance z i z_{i} as the pointwise influence of z i z_{i} on itself .
Formally 

 

 
 | 
 Mem ​ ( z i ) ≔ ℐ ⁡ ( z i , z i ) ≈ ℐ ^ ​ ( z i , z i ) ​ . {\textsc{Mem}(z_{i})}\coloneqq{\mathcal{I}\left(z_{i}\mathchar 59\relax z_{i}\right)}\approx{\widehat{\mathcal{I}}\left(z_{i}\mathchar 59\relax z_{i}\right)}\text{.} | 
 | 
 (4) | 
 

 
 
 (4) Cook’s distance measures the effect of training instances on the model parameters themselves [ Coo77a , Eq. (5)] .
Formally, the pointwise Cook’s distance of z i ∈ 𝒟 {z_{i}\in\mathcal{D}} is 

 

 
 | 
 ℐ Cook ​ ( z i ) ≔ θ ( T ) − θ 𝒟 ∖ z i ( T ) ​ . {\mathcal{I}_{\text{Cook}}\mathopen{}\left(z_{i}\right)\mathclose{}}\coloneqq\theta^{(T)}-{\theta^{(T)}_{\mathcal{D}^{\setminus z_{i}}}}\text{.} | 
 | 
 (5) | 
 

 Eq. ( 5 ) trivially extends to groups of training instances where for any D ⊆ 𝒟 {D\subseteq\mathcal{D}} 

 

 
 | 
 ℐ Cook ​ ( D ) ≔ θ ( T ) − θ 𝒟 ∖ D ( T ) ​ . {\mathcal{I}_{\text{Cook}}\mathopen{}\left(D\right)\mathclose{}}\coloneqq\theta^{(T)}-\theta^{(T)}_{\mathcal{D}\setminus D}\text{.} | 
 | 
 (6) | 
 

 Cook’s distance is particularly relevant for interpretable model classes where feature weights are most transparent.
This includes linear regression [ RL87a , Woj+16a ] and decision trees [ BHL23a ] . 

 
 
 (5) All definitions of influence above consider training instances’ effects w.r.t. a single instantiation of a model.
Across repeated stochastic retrainings, a training instance’s influence may vary – potentially substantially [ BPF21a , SD21a , Ras+22a ] .
 Expected influence is the average influence across all possible instantiations within a given model class [ KS21a , WJ23a ] .
Expected influence is particularly useful in domains where the random component of training is unknowable a priori.
For example, with poisoning and backdoor attacks, an adversary crafts malicious training instances to be highly influential in expectation across all random parameter initializations and batch orderings [ Che+17a , Sha+18b , Fow+21a ] . 

 
 
 Expected influence generalizes to consider group effects.
Existing related work focuses on counterfactuals such as, “what is the expected prediction for x te x_{\text{te}} if a model is trained on some arbitrary subset of 𝒟 \mathcal{D} [ Ily+22a , KCC23a ] ?”
Other work seeks to predict model parameters θ ( T ) \theta^{(T)} given an arbitrary training subset [ Zen+23a ] . 

 
 
 (6) Observe that all preceding definitions view influence as a specific numerical value to measure/estimate.
Influence analysis often simplifies to a relative question of whether one training instance is more influential than another.
An influence ranking orders (groups of) training instances from most positively influential to most negatively influential.
These rankings are useful in a wide range of applications [ KZ22a , WJ23a ] , including data cleaning and poisoning attack defenses as discussed in Section 6 . 

 
 
 

### 3.3 Topics Related to Influence Analysis

 
 All influence analysis methods we highlight in Sections 4 and 5 estimate a function that quantifies the impact of specific training examples on a given model or prediction.
For the most part, these methods take an ablation perspective, i.e., measuring how much a model changes when removing specific training examples.
However, there are several other research areas related to analyzing the impact of different subsets of the training data, albeit with somewhat different methods or objectives.
We briefly describe some of these topics below. 6 6 
 6 
 
 
 
 Section 6 (“ 6 Applications of Influence Analysis ”) discusses additional tasks that have leveraged influence analysis to achieve their own meta-objectives (e.g., adversarial robustness, model explainability, etc.). 

 
 
 Data pruning methods such as coresets [ BLK17a ] also consider the impact of removing examples, but they typically consider removing many examples (rather than one or a few) with the primary goal of increasing computational efficiency.
A coreset D CS ⊂ 𝒟 {D_{\textnormal{CS}}\subset\mathcal{D}} is a (weighted) set of points that can stand in for the overall training data when measuring a cost function , cost ​ ( D CS , Q ) \text{cost}(D_{\textnormal{CS}}\mathchar 59\relax Q) where Q ∈ 𝒬 {Q\in\mathcal{Q}} is a solution in solution space 𝒬 \mathcal{Q} .
 D CS D_{\textnormal{CS}} is an ε \varepsilon - coreset if it approximates the cost function within a factor of ϵ 0 {\epsilon 0} : 

 

 
 | 
 | cost ​ ( D CS , Q ) − cost ​ ( 𝒟 , Q ) | ≤ ε ​ cost ​ ( 𝒟 , Q ) ​ . \lvert\text{cost}(D_{\textnormal{CS}}\mathchar 59\relax Q)-\text{cost}(\mathcal{D}\mathchar 59\relax Q)\rvert\leq\varepsilon\,\text{cost}(\mathcal{D}\mathchar 59\relax Q)\text{.} | 
 | 
 (7) | 
 

 Several techniques exist to efficiently find D CS D_{\textnormal{CS}} [ MBL20a , Fel20b , Tuk+23a ] .
These include methods loosely based on weighted importance sampling , where each training instance’s sampling probability is proportional to the instance’s influence [ BLK17a ] . 

 
 
 Coresets let us learn a model over a much smaller set of points, while still yielding a model that is within a factor of 1 + ε {1+\varepsilon} of the optimal loss on the original training set.
Therefore, a coreset represents a sufficient set of points for a given task, while most influence estimation methods identify the most necessary points — the points without which performance would decline, even given many other points from the original dataset.
Another difference is that coresets are often motivated by efficiency concerns, whereas influence estimation is more motivated by the need to understand the data and its impact on a model — specifically, a model trained on the entire training data. 

 
 
 Coreset construction often involves submodular optimization [ Bil22a ] , so that an efficient, greedy approach finds a nearly-optimal set of points.
However, this also means that if there are multiple, equally-important points, submodular optimization will select one and skip the others as redundant.
This “winner-take-all” approach is in stark contrast to most influence estimation methods, which tend to assign similar importance to similar points. 

 
 
 Active learning seeks to maximize a model’s performance while annotating as little training data as possible [ Ren+21a ] .
Like influence estimation, active learning estimates the relative value different data points would have in fitting a model.
However, unlike influence estimation, this is done without knowledge of the labels of these points.
Furthermore, the goal is maximizing performance more than understanding the data, increasing efficiency, or precisely matching the loss on the full training data. 

 
 
 Often, the data points to label are chosen greedily by identifying the training instance whose labeling would most positively influence the model (in expectation).
Quantifying each unlabeled instance’s true influence at each active learning iteration may be prohibitive, so influence estimation techniques are often used to quantify each remaining unlabeled instance’s marginal contribution at a given iteration [ Liu+21a ] . 

 
 
 With this broad perspective on influence analysis and related concepts in mind, we transition to focusing on specific influence analysis methods in the next two sections. 

 
 
 
 

## 4 Retraining-Based Influence Analysis

 
 Training instances can only influence a model if they are used during training.
As Section 3.1.2 describes in the context of linear regression, one method to measure influence just trains a model with and without some instance; influence is then defined as the difference in these two models’ behavior.
This basic intuition is the foundation of retraining-based influence analysis, and
this simple formulation applies to any model class – parametric or non - parametric. 

 
 
 Observe that the retraining-based framework makes no assumptions about the learning environment.
In fact, this simplicity is one of the primary advantages of retraining-based influence.
For comparison, Table shows that all gradient-based influence estimators make strong assumptions – some of which are known not to hold for deep models (e.g., convexity).
However, retraining’s flexibility comes at the expense of high (sometimes prohibitive) computational cost. 

 
 
 Below, we describe three progressively more complex retraining-based influence analysis methods.
Each method mitigates weaknesses of the preceding method – in particular, devising techniques to make retraining-based influence more viable computationally. 

 
 
 Remark 1 : 
 
 This section treats model training as deterministic where, given a fixed training set, training always yields the same output model.
Since the training of modern models is mostly stochastic, retraining-based estimators should be represented as expectations over different random initializations and batch orderings.
Therefore, (re)training should be repeated multiple times for each relevant training (sub)set with a probabilistic average taken over the valuation metric [ Lin+22a ] .
For simplicity of presentation, expectation over randomness is dropped from the influence and influence estimator definitions below. 

 
 
 
 Remark 2 : 
 
 Section 2 defines 𝒟 \mathcal{D} as a supervised training set.
The three primary retraining-based influence analysis methods detailed below also generalize to unsupervised and semi-supervised training. 

 
 
 
 Remark 3 : 
 
 When calculating retraining’s time complexity below, each training iteration’s time complexity is treated as a constant cost.
This makes the time complexity of training a single model 𝒪 ⁡ ( T ) {\mathcal{O}(T)} .
Depending on the model architecture and hyperparameter settings, a training iteration’s complexity may directly depend on training-set size n n or model parameter count p p . 

 
 
 
 Remark 4 : 
 
 It may be possible to avoid full model retraining by using machine unlearning methods capable of certifiably “forgetting” training instances [ Guo+20a , BL21a , Ngu+22a , Eis+22a ] .
The asymptotic complexity of such methods is model-class specific and beyond the scope of this work.
Nonetheless, certified deletion methods can drastically reduce the overhead of retraining-based influence analysis. 

 
 
 

### 4.1 Leave-One-Out Influence

 
 Leave-one-out (LOO) is the simplest influence measure described in this work.
LOO is also the oldest, dating back to [ CW82a ] who term it case deletion diagnostics . 

 
 
 As its name indicates, leave-one-out influence is the change in z te z_{\text{te}} ’s risk due to the removal of a single instance, z i z_{i} , from the training set [ KL17a , BF21a , Jia+21b ] .
Formally, 

 

 
 | 
 ℐ LOO ​ ( z i , z te ) ≔ ℒ ⁡ ( z te , θ 𝒟 ∖ z i ( T ) ) − ℒ ⁡ ( z te , θ ( T ) ) ​ , {\mathcal{I}_{\textsc{LOO}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\coloneqq{\mathcal{L}(z_{\text{te}};{\theta^{(T)}_{\mathcal{D}^{\setminus z_{i}}}})}-{\mathcal{L}(z_{\text{te}};\theta^{(T)})}\text{,} | 
 | 
 (8) | 
 

 where θ 𝒟 ∖ z i ( T ) {\theta^{(T)}_{\mathcal{D}^{\setminus z_{i}}}} are the final model parameters when training on subset 𝒟 ∖ z i {\mathcal{D}\setminus z_{i}} and θ ( T ) \theta^{(T)} are the final model parameters trained on all of 𝒟 \mathcal{D} . 

 
 
 Measuring the entire training set’s LOO influence requires training ( n + 1 ) {(n+1)} models.
Given a deterministic model class and training algorithm (e.g., convex model optimization [ BV04a ] ), LOO is one of the few influence measures that can be computed exactly in polynomial time w.r.t. training-set size n n and iteration count T T . 

 
 

#### 4.1.1 Time, Space, and Storage Complexity

 
 Training a single model has time complexity 𝒪 ⁡ ( T ) {\mathcal{O}(T)} (see Remark 3 ).
By additivity, training ( n + 1 ) {(n+1)} models has total time complexity 𝒪 ⁡ ( n ​ T ) {\mathcal{O}(nT)} .
Since these ( n + 1 ) {(n+1)} models are independent, they can be trained in parallel. 

 
 
 Pointwise influence analysis always has space complexity of at least 𝒪 ⁡ ( n ) {\mathcal{O}(n)} , i.e., the space taken by the n n influence values ∀ i ℐ ⁡ ( z i , z te ) {\forall_{i}\,{\mathcal{I}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}} .
Training a single model has space complexity 𝒪 ⁡ ( p ) {\mathcal{O}(p)} , where p ≔ | θ | {p\coloneqq\lvert\theta\rvert} ; this complexity scales linearly with the number of models trained in parallel.
Table treats the level of training concurrency as a constant factor, which is why LOO’s total space complexity is listed as 𝒪 ⁡ ( n + p ) {\mathcal{O}(n+p)} . 

 
 
 A naive implementation of LOO would train the n n additional models and immediately discard them after measuring z te z_{\text{te}} ’s test loss.
This simple version of LOO has 𝒪 ⁡ ( 1 ) {\mathcal{O}(1)} storage complexity.
If instead the ( n + 1 ) {(n+1)} models are stored, analysis of subsequent test instances requires no additional retraining.
This drastically reduces LOO’s incremental time complexity for subsequent instances to just 𝒪 ⁡ ( n ) {\mathcal{O}(n)} forward passes -- a huge saving. 7 7 
 7 
 
 
 
 LOO’s incremental computational cost can be (significantly) reduced in practice via batching. 
Note that this amortization of the retraining cost induces an 𝒪 ⁡ ( n ​ p ) {\mathcal{O}(np)} storage complexity as listed in Table . 

 
 
 

#### 4.1.2 Strengths and Weaknesses

 
 Leave-one-out influence’s biggest strength is its simplicity.
LOO is human-intelligible – even by laypersons.
For that reason, LOO has been applied to ensure the fairness of algorithmic decisions [ BF21a ] .
Moreover, like all methods in this section, LOO’s simplicity allows it to be combined with any model architecture. 

 
 
 LOO’s theoretical simplicity comes at the price of huge upfront computational cost.
Training some state-of-the-art models from scratch even once is prohibitive for anyone beyond industrial actors [ Dis+21a ] .
For even the biggest players, it is impractical to train ( n + 1 ) {(n+1)} such models given huge modern datasets [ BPF21a ] .
The climate effects of such retraining also cannot be ignored [ SGM20a ] . 

 
 
 LOO’s simple definition in Eq. ( 8 ) is premised on deterministic training.
However, even when training on the same data, modern models may have significant predictive variance for a given test instance [ BF21a , WJ23a ] .
This variance makes it difficult to disentangle the effect of an instance’s deletion from training’s intrinsic variability [ BPF21a ] .
For a single training instance, estimating the LOO influence within a standard deviation of σ \sigma requires training Ω ⁡ ( 1 / σ 2 ) {\Omega(1/\sigma^{2})} models.
Therefore, estimating the entire training set’s LOO influence requires training Ω ⁡ ( n / σ 2 ) {\Omega(n/\sigma^{2})} models – further exacerbating LOO’s computational infeasibility [ FZ20a ] . 

 
 
 These limitations notwithstanding, LOO’s impact on influence analysis research is substantial.
Many pointwise influence analysis methods either directly estimate the leave-one-out influence –
e.g., Downsampling (Sec. 4.2 ),
influence functions (Sec. 5.1.1 ),
 HyDRA (Sec. 5.2.2 )
–
or are very similar to LOO –
e.g., Shapley value (Sec. 4.3 ). 

 
 
 

#### 4.1.3 Related Methods

 
 Efficient Nearest-Neighbor LOO   
Although LOO has poor upfront and storage complexities in general, it can be quite efficient for some model classes – particularly instance-based learners [ AKA91a ] .
For example, [ Jia+21b ] propose the k k NN leave-out-one ( k k NN LOO) estimator, which calculates the LOO influence over a surrogate k k - nearest neighbors classifier instead of over target model f f .
 k k NN LOO relies on a simple two-step process.
First, the features of test instance z te z_{\text{te}} and training set 𝒟 \mathcal{D} are extracted using a pretrained model.
Next, a k k NN classifier’s LOO influence is calculated exactly [ Jia+21b , Lemma 1] using these extracted features.
 [ Jia+21b ] prove that k k NN LOO influence only requires 𝒪 ⁡ ( n ​ log ⁡ n ) {\mathcal{O}(n\log n)} time – significantly faster in practice than vanilla LOO’s 𝒪 ⁡ ( n ​ T ) {\mathcal{O}(nT)} complexity.
 [ Jia+21b ] also demonstrate empirically that k k NN LOO and vanilla LOO generate similar influence rankings across various learning domains and tasks. 

 
 
 Efficient LOO Estimation in Decision Tree Ensembles   
 [ Sha+18c ] propose LeafRefit , an efficient LOO estimator for decision-tree ensembles.
 LeafRefit ’s efficiency derives from the simplifying assumption that instance deletions do not affect the trees’ structure.
In cases where this assumption holds, LeafRefit ’s tree influence estimates are exact.
To the extent of our knowledge, LeafRefit ’s suitability for surrogate influence analysis of deep models has not yet been explored. 

 
 
 Cook’s Distance and Linear Regression   
For least-squares linear regression, [ Woj+16a ] show that each training instance’s LOO influence on the model parameters (i.e., Cook’s distance) can be efficiently estimated by mapping training set 𝒟 \mathcal{D} into a lower-dimensional subspace.
By the Johnson - Lindenstrauss lemma [ JL84a ] , these influence sketches approximately preserve the pairwise distances between the training instances in 𝒟 \mathcal{D} provided the projected dimension is on the order of log ⁡ n {\log n} . 

 
 
 LOO Group Influence   
Leave - one - out can be extended to leave - m m - out for any integer m ≤ n {m\leq n} . 8 8 
 8 
 
 
 
 Leave - m m - out influence analysis is also called multiple case deletion diagnostics [ RL87a ] .
 
Leave - m m - out has time complexity 𝒪 ⁡ ( ( n m ) ) {\mathcal{O}(\binom{n}{m})} , which is exponential in the worst case.
Shapley value influence [ Sha53a ] (Sec. 4.3 ) shares significant similarity with leave - m m - out. 

 
 
 As mentioned above,
LOO influence serves as the reference influence value for multiple influence estimators including Downsampling , which we describe next. 

 
 
 
 

### 4.2 Downsampling 

 
 Proposed by [ FZ20a ] , Downsampling 9 9 
 9 
 
 
 
 [ FZ20a ] do not specify a name for their influence estimator.
Previous work has referred to [ FZ20a ] ’s method as “subsampling” [ BHL23a ] and as “counterfactual influence” [ Zha+21d ] .
We use “ Downsampling ” to differentiate [ FZ20a ] ’s method from the existing, distinct task of dataset subsampling [ TB18a ] while still emphasizing the methods’ reliance on repeated training-set sampling. 
mitigates leave-one-out influence’s two primary weaknesses: (1) computational complexity dependent on n n and (2) instability due to stochastic training variation. 

 
 
 Downsampling relies on an ensemble of K K submodels each trained on a u.a.r. subset of full training set 𝒟 \mathcal{D} .
Let D k ∼ m 𝒟 {D^{k}\,\stackrel{{\scriptstyle m}}{{\sim}}\,\mathcal{D}} be the k k - th submodel’s training set where ∀ k | D k | = m n {\forall_{k}\,\lvert D^{k}\rvert=m n} . 10 10 
 10 
 
 
 
 [ FZ20a ] propose setting m = ⌈ 0.7 ​ n ⌉ {m=\lceil 0.7n\rceil} . 
Define
 K i ≔ ∑ k = 1 K 𝟙 [ z i ∈ D k ] {K_{i}\coloneqq\sum_{k=1}^{K}{\mathbbm{1}[z_{i}\in D^{k}]}} 
as the number of submodels that used instance z i z_{i} during training.
The Downsampling pointwise influence estimator 11 11 
 11 
 
 
 
 [ FZ20a ] define their estimator specifically for classification.
 Downsampling ’s definition in Eq. ( 9 ) uses a more general form to cover additional learning tasks such as regression.
 [ FZ20a ] ’s original formulation would be equivalent to defining the risk as ℒ ( z te ; θ D k ( T ) ) = 𝟙 [ y te ≠ f ( x te ; θ D k ( T ) ) ] {{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D^{k}})}={\mathbbm{1}\big[y_{\text{te}}\neq{f(x_{\text{te}};\theta^{(T)}_{D^{k}})}\big]}} , i.e., the accuracy subtracted from one. 
is then 

 

 
 | 
 ℐ ^ Down ​ ( z i , z te ) ≔ 1 K − K i ​ ∑ k z i ∉ D k ℒ ⁡ ( z te , θ D k ( T ) ) − 1 K i ​ ∑ k ′ z i ∈ D k ′ ℒ ⁡ ( z te , θ D k ′ ( T ) ) ​ . {\widehat{\mathcal{I}}_{\textsc{Down}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\coloneqq\frac{1}{K-K_{i}}\sum_{\begin{subarray}{c}k\\
z_{i}\notin D^{k}\end{subarray}}{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D^{k}})}-\frac{1}{K_{i}}\sum_{\begin{subarray}{c}k^{\prime}\\
z_{i}\in D^{k^{\prime}}\end{subarray}}{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D^{k^{\prime}}})}\text{.} | 
 | 
 (9) | 
 

 Intuitively, Eq. ( 9 ) is the change in z te z_{\text{te}} ’s average risk when z i z_{i} is not used in submodel training.
By holding out multiple instances simultaneously and then averaging, each Downsampling submodel provides insight into the influence of all training instances.
This allows Downsampling to require (far) fewer retrainings than LOO. 

 
 
 Since each of the K K training subsets is i.i.d., then 

 

 
 | 
 lim K → ∞ ℐ ^ Down ​ ( z i , z te ) = 𝔼 D ′ ∼ m 𝒟 ∖ z i ​ [ ℒ ⁡ ( z te , θ D ′ ( T ) ) ] − 𝔼 D ∼ m − 1 𝒟 ∖ z i ​ [ ℒ ⁡ ( z te , θ D ∪ z i ( T ) ) ] ​ . \lim_{K\rightarrow\infty}{\widehat{\mathcal{I}}_{\textsc{Down}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}=\mathbb{E}_{D^{\prime}\,\stackrel{{\scriptstyle m}}{{\sim}}\,\mathcal{D}^{\setminus z_{i}}}[{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D^{\prime}})}]-\mathbb{E}_{D\,\stackrel{{\scriptstyle m-1}}{{\sim}}\,\mathcal{D}^{\setminus z_{i}}}[{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D\cup z_{i}})}]\text{.} | 
 | 
 (10) | 
 

 For sufficiently large m m and n n , the expected behavior of a model trained on an i.i.d. dataset of size m m becomes indistinguishable from one trained on m − 1 {m-1} i.i.d. instances.
Applying this property along with linearity of expectation and Eq. ( 8 ), Eq. ( 10 ) reformulates as 

 

 
 | 
 lim K ​ ; ​ n ​ ; ​ m → ∞ ℐ ^ Down ​ ( z i , z te ) \displaystyle\lim_{K\mathord{\mathchar 59\relax}n\mathord{\mathchar 59\relax}m\rightarrow\infty}{\widehat{\mathcal{I}}_{\textsc{Down}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)} | 
 = 𝔼 D ′ ∼ m − 1 𝒟 ∖ z i ​ [ ℒ ⁡ ( z te , θ D ′ ( T ) ) ] − 𝔼 D ∼ m − 1 𝒟 ∖ z i ​ [ ℒ ⁡ ( z te , θ D ∪ z i ( T ) ) ] \displaystyle=\mathbb{E}_{D^{\prime}\,\stackrel{{\scriptstyle m-1}}{{\sim}}\,\mathcal{D}^{\setminus z_{i}}}[{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D^{\prime}})}]-\mathbb{E}_{D\,\stackrel{{\scriptstyle m-1}}{{\sim}}\,\mathcal{D}^{\setminus z_{i}}}[{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D\cup z_{i}})}] | 
 | 
 (11) | 
 
 
 | 
 | 
 = 𝔼 D ∼ m − 1 𝒟 ∖ z i ​ [ ℒ ⁡ ( z te , θ D ( T ) ) − ℒ ⁡ ( z te , θ D ∪ z i ( T ) ) ] \displaystyle=\mathbb{E}_{D\,\stackrel{{\scriptstyle m-1}}{{\sim}}\,\mathcal{D}^{\setminus z_{i}}}[{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D})}-{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D\cup z_{i}})}] | 
 | 
 (12) | 
 
 
 | 
 | 
 = 𝔼 D ∼ m − 1 𝒟 ∖ z i ​ [ ℐ LOO ​ ( z i , z te ) ] ​ . \displaystyle=\mathbb{E}_{D\,\stackrel{{\scriptstyle m-1}}{{\sim}}\,\mathcal{D}^{\setminus z_{i}}}[{\mathcal{I}_{\textsc{LOO}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\vphantom{\bigg[}]\text{.} | 
 | 
 (13) | 
 

 
 
 Hence, Downsampling is a statistically consistent estimator of the expected LOO influence .
This means that Downsampling does not estimate the influence of training instance z i z_{i} on a single model instantiation.
Rather, Downsampling estimates z i z_{i} ’s influence on the training algorithm and model architecture as a whole .
By considering influence in expectation, Downsampling addresses LOO’s inaccuracy caused by stochastic training’s implicit variance. 

 
 
 In practice, K K , n n , and m m are finite. Nonetheless, [ FZ20a , Lemma 2.1] prove that, with high probability, Downsampling ’s LOO influence estimation error is bounded given K K and m n \frac{m}{n} . 

 
 
 While Downsampling ’s formulation above is w.r.t. a single training instance,
 Downsampling trivially extends to estimate the expected group influence of multiple training instances.
Observe however that the expected fraction of u.a.r. training subsets that either contains all instances in a group or none of a group decays geometrically with m n {\frac{m}{n}} and ( 1 − m n ) {(1-\frac{m}{n})} , respectively.
Therefore, for large group sizes, K K needs to be exponential in n n to cover sufficient group combinations. 

 
 
 Remark 5 : 
 
 Downsampling trains models on data subsets of size m m under the assumption that statements made about those models generalize to models trained on a dataset of size n n (i.e., all of 𝒟 \mathcal{D} ).
This assumption may not hold for small m m .
To increase the likelihood this assumption holds, [ FZ20a ] propose fixing m n = 0.7 {\frac{m}{n}=0.7} .
This choice balances satisfying the aforementioned assumption against the number of submodels since Downsampling requires K K and ratio m n \frac{m}{n} combined dictate K i K_{i} , i.e., the number of submodels that are trained on z i z_{i} . 

 
 
 

#### 4.2.1 Time, Space, and Storage Complexity

 
 Downsampling ’s complexity analysis is identical to that of LOO (Sec. 4.1.1 ) except, instead of the time and storage complexities being dependent on training-set size n n , Downsampling depends on submodel count K K .
For perspective, [ FZ20a ] ’s empirical evaluation used K = 2 ​ ; ​ 000 {K=2\mathord{\mathchar 59\relax}000} for ImageNet [ Den+09a ] ( n 14 ​ M {n 14\text{M}} ) as well as K = 4 ​ ; ​ 000 {K=4\mathord{\mathchar 59\relax}000} for MNIST [ LeC+98a ] ( n = 60 ​ ; ​ 000 {n=60\mathord{\mathchar 59\relax}000} ) and CIFAR10 [ KNH14a ] ( n = 50 ​ ; ​ 000 {n=50\mathord{\mathchar 59\relax}000} ) – a savings of one to four orders of magnitude over vanilla LOO. 

 
 
 Remark 6 : 
 
 Downsampling ’s incremental time complexity is technically 𝒪 ⁡ ( K + n ) ∈ 𝒪 ⁡ ( n ) {{\mathcal{O}(K+n)}\in{\mathcal{O}(n)}} since pointwise influence is calculated w.r.t. each training instance.
The time complexity analysis above focuses on the difference in the number of forward passes required by LOO and Downsampling ; fewer forward passes translate to Downsampling being much faster than LOO in practice. 

 
 
 
 

#### 4.2.2 Strengths and Weaknesses

 
 Although more complicated than LOO, Downsampling is still comparatively simple to understand and implement.
 Downsampling makes only a single assumption that should generally hold in practice (see Remark 5 ).
 Downsampling is also flexible and can be applied to most applications. 

 
 
 Another strength of Downsampling is its low incremental time complexity.
Each test example requires only K K forward passes. These forward passes can use large batch sizes to further reduce the per-instance cost.
This low incremental cost allows Downsampling to be applied at much larger scales than other methods.
For example, [ FZ20a ] measure all pointwise influence estimates for the entire ImageNet dataset [ Den+09a ] ( n 14 ​ M {n 14\text{M}} ).
These large-scale experiments enabled [ FZ20a ] to draw novel conclusions about neural training dynamics – including that training instance memorization ( 4 ) by overparameterized models is not a bug, but a feature, that is currently necessary to achieve state-of-the-art generalization results. 

 
 
 In terms of weaknesses, while Downsampling is less computationally expensive than LOO, Downsampling still has a high upfront computational cost.
Training multiple models may be prohibitively expensive even when K ≪ n {K\ll n} .
Amortization of this upfront training overhead across multiple test instances is beneficial but by no means a panacea. 

 
 
 

#### 4.2.3 Related Methods

 
 Downsampling has two primary related methods. 

 
 
 Generative Downsampling   Training instance memorization also occurs in generative models where the generated outputs are (nearly) identical copies of training instances [ KWR22a ] .
 [ vW21a ] extend Downsampling to deep generative models – specifically, density models (e.g., variational autoencoders [ KW14a , RMW14a ] ) that estimate posterior probability p ⁡ ( x | 𝒫 ; θ ) {p(x|\mathcal{P}\mathchar 59\relax\theta)} , where 𝒫 \mathcal{P} denotes the training data distribution.
Like Downsampling , [ vW21a ] ’s approach relies on training multiple submodels. 12 12 
 12 
 
 
 
 Rather than training submodels using i.i.d. subsets of 𝒟 \mathcal{D} , [ vW21a ] propose training the submodels via repeated d d - fold cross-validation.
While technically different, [ vW21a ] ’s approach is functionally equivalent to [ FZ20a ] ’s [ FZ20a ] u.a.r. sampling procedure. 
The primary difference is that [ vW21a ] consider generative risk 

 

 
 | 
 ℒ ⁡ ( x te , θ ( T ) ) = − log ⁡ p ⁡ ( x te | 𝒫 ; θ ( T ) ) ​ . {\mathcal{L}(x_{\text{te}};\theta^{(T)})}=-\log p(x_{\text{te}}|\mathcal{P}\mathchar 59\relax\theta^{(T)})\text{.} | 
 | 
 (14) | 
 

 Beyond that, [ vW21a ] ’s method is the same as Downsampling as both methods consider the LOO influence ( 9 ). 

 
 
 Consistency Profile and Score    Downsampling ’s second closely related method is [ Jia+21c ] ’s [ Jia+21c ] consistency profile , defined formally as 13 13 
 13 
 
 
 
 [ Jia+21c ] define their estimator specifically for classification. We present their method more generally to apply to other losses/domains (e.g., regression).
As with Downsampling , it is trivial to map between Eq. ( 15 )’s formulation and that of [ Jia+21c ] 

 

 
 | 
 C m ​ ; ​ 𝒟 ​ ( z te ) ≔ − 𝔼 D ∼ m 𝒟 ​ [ ℒ ⁡ ( z te , θ D ( T ) ) ] ​ . {C_{m\mathord{\mathchar 59\relax}\mathcal{D}}(z_{\text{te}})}\coloneqq-\mathbb{E}_{D\,\stackrel{{\scriptstyle m}}{{\sim}}\,\mathcal{D}}[{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D})}]\text{.} | 
 | 
 (15) | 
 

 By negating the risk in Eq. ( 15 ), a higher expected risk corresponds to a lower consistency profile.
Consistency profile differs from Downsampling in two ways. (1) Downsampling implicitly considers a single submodel training-set size m m while
consistency profile disentangles the estimator from m m .
(2) Downsampling estimates z i z_{i} ’s influence on z te z_{\text{te}} while consistency profile considers all of 𝒟 \mathcal{D} as a group and estimates
the expected group influence of a random subset D ⊆ 𝒟 {D\subseteq\mathcal{D}} given m ≔ | D | {m\coloneqq\lvert D\rvert} . 

 
 
 [ Jia+21c ] also propose the consistency score (C - score), defined formally as 

 

 
 | 
 C 𝒟 ​ ( z te ) ≔ 𝔼 m ∼ [ n ] ​ [ C m ​ ; ​ 𝒟 ​ ( z te ) ] ​ , {C_{\mathcal{D}}(z_{\text{te}})}\coloneqq\mathbb{E}_{m\,\sim\,{[n]}}[{C_{m\mathord{\mathchar 59\relax}\mathcal{D}}(z_{\text{te}})}]\text{,} | 
 | 
 (16) | 
 

 where m m is drawn uniformly from set [ n ] {[n]} .
By taking the expectation over all training-set sizes, C - score provides a total ordering over all test instances.
A large C - score entails that z te z_{\text{te}} is harder for the model to confidently predict.
Large C - scores generally correspond to rare/atypical test instances from the tails of the data distribution.
Since Downsampling considers the effect of each training instance individually, Downsampling may be unable to identify these hard - to - predict test instances – in particular if m m is large enough to cover most data distribution modes. 

 
 
 The next section introduces the Shapley value, which in essence merges the ideas of Downsampling and C - score. 

 
 
 
 

### 4.3 Shapley Value

 
 Derived from cooperative game theory,
 Shapley value (SV) quantifies the increase in value when a group of players cooperates to achieve some shared objective [ Sha53a , SR88a ] .
Given n n total players, characteristic function ν : 2 [ n ] → {\nu:2^{{[n]}}\rightarrow\real} defines the value of any player coalition A ⊆ [ n ] {A\subseteq{[n]}} 
 [ Dub75a ] .
By convention, a larger ν ⁡ ( A ) {\nu(A)} is better.
Formally, player i i ’s Shapley value w.r.t. ν \nu is 

 

 
 | 
 𝒱 ⁡ ( i , ν ) ≔ 1 n ​ ∑ A ⊆ [ n ] ∖ i 1 ( n − 1 | A | ) ​ [ ν ⁡ ( A ∪ i ) − ν ⁡ ( A ) ] ​ , {\mathcal{V}(i;\nu)}\coloneqq\frac{1}{n}\sum_{A\subseteq{[n]}\setminus i}\frac{1}{\binom{n-1}{\lvert A\rvert}}[{\nu(A\cup i)}-{\nu(A)}\vphantom{\Big|}]\text{,} | 
 | 
 (17) | 
 

 where ( n − 1 | A | ) \binom{n-1}{\lvert A\rvert} is the binomial coefficient. 

 
 
 [ GZ19a ] adapt SV to model training by treating the n n instances in training set 𝒟 \mathcal{D} as n n cooperative players with the shared objective of training the ‘‘best’’ model. 14 14 
 14 
 
 
 
 “Best” here is w.r.t. some data valuation measure of interest, with different use cases potentially defining “best” differently. 
For any A ⊆ 𝒟 {A\subseteq\mathcal{D}} , let ν ⁡ ( A ) ≔ − ℒ ⁡ ( z te , θ A ( T ) ) {{\nu(A)}\coloneqq-{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{A})}} where the negation is needed because more “valuable” training subsets have lower risk.
Then, z i z_{i} ’s Shapley value pointwise influence on z te z_{\text{te}} is 

 

 
 | 
 ℐ SV ​ ( z i , z te ) ≔ 1 n ​ ∑ D ⊆ 𝒟 ∖ z i 1 ( n − 1 | D | ) ​ [ ℒ ⁡ ( z te , θ D ( T ) ) − ℒ ⁡ ( z te , θ D ∪ z i ( T ) ) ] ​ . {\mathcal{I}_{\text{SV}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\coloneqq\frac{1}{n}\sum_{D\subseteq\mathcal{D}^{\setminus z_{i}}}\frac{1}{\binom{n-1}{\lvert D\rvert}}[{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D})}-{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D\cup z_{i}})}\vphantom{\Big|}]\text{.} | 
 | 
 (18) | 
 

 More intuitively, SV is the weighted change in z te z_{\text{te}} ’s risk when z i z_{i} is added to a random training subset; the weighting ensures all training subset sizes ( | D | \lvert D\rvert ) are prioritized equally.
Eq. ( 18 ) can be viewed as generalizing the leave - one - out influence, where rather than considering only full training set 𝒟 \mathcal{D} , Shapley value averages the LOO influence across all possible subsets of 𝒟 \mathcal{D} . 

 
 
 There exists multiple extensions of Shapley value to the group context [ BL23a , TYR23a , GR99a , SDA20a ] . 15 15 
 15 
 
 
 
 Most of these works were proposed in the context of studying the interaction between groups of features.
Directly adapting these ideas to groups of training instances is straightforward. 
The most well-known method is [ GR99a ] ’s [ GR99a ] Shapley interaction index , which for any subset A ⊆ 𝒟 {A\subseteq\mathcal{D}} , is defined as 

 

 
 | 
 ℐ SV ( A ; z te ) ≔ − ∑ D ⊆ 𝒟 ∖ A ( n − | A | − | D | ) ! ​ | D | ! ( n − | A | + 1 ) ! ∑ D ′ ⊆ A ( − 1 ) | A | − | D ′ | ℒ ( z te ; θ D ∪ D ′ ( T ) ) . {\mathcal{I}_{\text{SV}}\left(A\mathchar 59\relax z_{\text{te}}\right)}\coloneqq-\sum_{D\subseteq\mathcal{D}\setminus A}\frac{(n-\lvert A\rvert-\lvert D\rvert)!\,\,\lvert D\rvert!}{(n-\lvert A\rvert+1)!}\sum_{D^{\prime}\subseteq A}\left(-1\right)^{\lvert A\rvert-\lvert D^{\prime}\rvert}{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D\cup D^{\prime}})}\text{.} | 
 | 
 (19) | 
 

 To intuitively understand Shapley interaction index’s inner summation, consider when A A consists of only two instances (e.g., A = { z i ; z j } {A=\{z_{i}\mathchar 59\relax z_{j}\}} ), then 

 

 
 | 
 ∑ D ′ ⊆ { z i ; z j } ( − 1 ) 2 − | D ′ | ​ ℒ ​ ( z te , θ D ∪ D ′ ( T ) ) = \displaystyle\sum_{D^{\prime}\subseteq\{z_{i}\mathchar 59\relax z_{j}\}}\left(-1\right)^{2-\lvert D^{\prime}\rvert}{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D\cup D^{\prime}})}= | 
 ℒ ⁡ ( z te , θ D ( T ) ) − ℒ ⁡ ( z te , θ D ∪ z i ( T ) ) \displaystyle{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D})}-{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D\cup z_{i}})} | 
 | 
 (20) | 
 
 
 | 
 | 
 − ℒ ⁡ ( z te , θ D ∪ z j ( T ) ) + ℒ ⁡ ( z te , θ D ∪ { z i ; z j } ( T ) ) ​ . \displaystyle-{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D\cup z_{j}})}+{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D\cup\{z_{i}\mathchar 59\relax z_{j}\}})}\text{.} | 
 | 
 

 [ GR99a ] explain that a positive interaction index entails that the combined group influence of the training instances in A ⊆ 𝒟 {A\subseteq\mathcal{D}} exceeds the sum of their marginal influences;
similarly, a negative interaction index means that A A ’s group influence is less than the sum of their marginal influences.
If the interaction index is zero, A A ’s members have no net interaction. [ SDA20a ] propose an alternate Shapley group formulation they term the Shapley-Taylor interaction index , which is defined as 

 

 
 | 
 ℐ ST ( A ; z te ) ≔ − | A | n ∑ D ⊆ 𝒟 ∖ A 1 ( n − 1 | D | ) ∑ D ′ ⊆ A ( − 1 ) | A | − | D ′ | ℒ ( z te ; θ D ∪ D ′ ( T ) ) . {\mathcal{I}_{\text{ST}}\left(A\mathchar 59\relax z_{\text{te}}\right)}\coloneqq-\frac{\lvert A\rvert}{n}\sum_{D\subseteq\mathcal{D}\setminus A}\frac{1}{\binom{n-1}{\lvert D\rvert}}\sum_{D^{\prime}\subseteq A}\left(-1\right)^{\lvert A\rvert-\lvert D^{\prime}\rvert}{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D\cup D^{\prime}})}\text{.} | 
 | 
 (21) | 
 

 The Shapley and Shapley-Taylor interaction indices provide different mathematical guarantees the details of which extend beyond the scope of this work.
We refer the reader to [ SDA20a ] for additional discussion. 

 
 

#### 4.3.1 Time, Space, and Storage Complexity

 
 [ DP94a ] prove that computing Shapley values is #P - complete.
Therefore, in the worst case, SV requires exponential time to determine exactly assuming P ≠ NP {\text{P}\neq\text{NP}} .
There has been significant follow-on work to develop tractable SV estimators, many of which are reviewed in Sec. 4.3.3 . 

 
 
 The analysis of SV’s space, storage, and incremental time complexities follows that of LOO (Sec. 4.1.1 ) with the exception that SV requires up to 2 n 2^{n} models, not just n n models as with LOO. 

 
 
 

#### 4.3.2 Strengths and Weaknesses

 
 Among all influence analysis methods, SV may have the strongest theoretical foundation with the chain of research extending back several decades.
SV’s dynamics and limitations are well understood, providing confidence in the method’s quality and reliability.
In addition, SV makes minimal assumptions about the nature of the cooperative game (i.e., model to be trained), meaning SV is very flexible.
This simplicity and flexibility allow SV to be applied to many domains beyond dataset influence as discussed in the next section. 

 
 
 SV has been shown to satisfy multiple appealing mathematical axioms.
First, SV satisfies the dummy player axiom where 

 

 
 | 
 ∀ D ⊆ 𝒟 ∖ z i ℒ ⁡ ( z te , θ D ∪ z i ( T ) ) = ℒ ⁡ ( z te , θ D ( T ) ) ⟹ ℐ SV ​ ( z i , z te ) = 0 ​ . {\forall}_{D\subseteq\mathcal{D}\setminus z_{i}}\,\,{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D\cup z_{i}})}={\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D})}\implies{\mathcal{I}_{\text{SV}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}=0\text{.} | 
 | 
 (22) | 
 

 Second, SV is symmetrical meaning for any z i ; z j ∈ 𝒟 {z_{i}\mathchar 59\relax z_{j}\in\mathcal{D}} , 

 

 
 | 
 ∀ D ⊆ 𝒟 ∖ { z i ; z j } ℒ ⁡ ( z te , θ D ∪ z i ( T ) ) = ℒ ⁡ ( z te , θ D ∪ z j ( T ) ) ⟹ ℐ SV ​ ( z i , z te ) = ℐ SV ​ ( z j , z te ) ​ . {\forall}_{D\subseteq\mathcal{D}\setminus\{z_{i}\mathchar 59\relax z_{j}\}}\,\,{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D\cup z_{i}})}={\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D\cup z_{j}})}\implies{\mathcal{I}_{\text{SV}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}={\mathcal{I}_{\text{SV}}\left(z_{j}\mathchar 59\relax z_{\text{te}}\right)}\text{.} | 
 | 
 (23) | 
 

 Third, SV is linear [ Sha53a , Dub75a ] , meaning given any two data valuation metrics ν ′ ; ν ′′ {\nu^{\prime}\mathchar 59\relax\nu^{\prime\prime}} and α ′ ; α ′′ ∈ {\alpha^{\prime}\mathchar 59\relax\alpha^{\prime\prime}\in\real} , it holds that 

 

 
 | 
 𝒱 ⁡ ( D , α ′ ​ ν ′ + α ′′ ​ ν ′′ ) = α ′ ​ 𝒱 ​ ( D , ν ′ ) + α ′′ ​ 𝒱 ​ ( D , ν ′′ ) ​ . {\mathcal{V}(D;\alpha^{\prime}\nu^{\prime}+\alpha^{\prime\prime}\nu^{\prime\prime})}=\alpha^{\prime}\,{\mathcal{V}(D;\nu^{\prime})}+\alpha^{\prime\prime}\,{\mathcal{V}(D;\nu^{\prime\prime})}\text{.} | 
 | 
 (24) | 
 

 SV’s linearity axiom makes it possible to estimate both pointwise and joint SV influences without repeating any data collection [ GZ19a ] .
Any data value satisfying the dummy player, symmetry, and linearity axioms is referred to as a semivalue [ DNW81a , KZ22a ] .
Additional semivalues include leave-one-out (Sec. 4.1 ) and Banzhaf value (Sec. 4.3.3 ) [ Ban65a ] . 

 
 
 Furthermore, by evaluating training sets of different sizes, SV can detect subtle influence behavior that is missed by methods like Downsampling and LOO, which evaluate a single training-set size.
 [ Lin+22a ] evidence this phenomenon empirically showing that adversarial training instances (i.e., poison) can sometimes be better detected with small SV training subsets. 

 
 
 Concerning weaknesses, SV’s computational intractability is catastrophic for non-trivial dataset sizes [ KZ22a ] .
For that reason, numerous (heuristic) SV speed-ups have been proposed, with the most prominent ones detailed next. 

 
 
 

#### 4.3.3 Related Methods

 
 Monte Carlo Shapley   
 [ GZ19a ] propose two SV estimators.
First, truncated Monte Carlo Shapley (TMC - Shapley) relies on randomized subset sampling from training set 𝒟 \mathcal{D} . 16 16 
 16 
 
 
 
 The term “truncated” in TMC - Shapley refers to a speed-up heuristic used when estimating the pointwise influences of multiple training instances at the same time. Truncation is not a necessary component of the method and is not described here. 
As a simplified description of the algorithm, TMC - Shapley relies on random permutations of 𝒟 \mathcal{D} ; for simplicity, denote the permutation ordering z 1 ; … ; z n {z_{1}\mathchar 59\relax\ldots\mathchar 59\relax z_{n}} .
For each permutation, n n models are trained where the i i - th model’s training set is instances { z 1 ; … ; z i } \{z_{1}\mathchar 59\relax\ldots\mathchar 59\relax z_{i}\} .
To measure each z i z_{i} ’s marginal contribution for a given permutation, TMC - Shapley compares the performance of the ( i − 1 ) {(i-1)} - th and i i - th models, i.e., the models trained on datasets { z 1 ; … ; z i − 1 } \{z_{1}\mathchar 59\relax\ldots\mathchar 59\relax z_{i-1}\} and { z 1 ; … ; z i } \{z_{1}\mathchar 59\relax\ldots\mathchar 59\relax z_{i}\} , respectively.
TMC - Shapley generates additional training-set permutations and trains new models until the SV estimates converge.
 [ GZ19a ] state that TMC - Shapley convergence usually
requires analyzing on the order of n n training-set permutations.
Given training each model has time complexity 𝒪 ⁡ ( T ) {\mathcal{O}(T)} (Remark 3 ), TMC - Shapley’s full time complexity is in general 𝒪 ⁡ ( n 2 ​ T ) {\mathcal{O}(n^{2}T)} . 

 
 
 Gradient Shapley   
TMC - Shapley may be feasible for simple models that are fast to train.
For more complex systems, 𝒪 ⁡ ( n 2 ) {\mathcal{O}(n^{2})} complete retrainings are impractical.
 [ GZ19a ] also propose Gradient Shapley (G - Shapley), an even faster SV estimator which follows the same basic procedure as TMC - Shapley with one critical change.
Rather than taking T T iterations to train each model, G - Shapley assumes models are trained in just one gradient step.
This means that G - Shapley’s full and incremental time complexity is only 𝒪 ⁡ ( n 2 ) {\mathcal{O}(n^{2})} – a substantial speed-up over TMC - Shapley’s full 𝒪 ⁡ ( n 2 ​ T ) {\mathcal{O}(n^{2}T)} complexity. 17 17 
 17 
 
 
 
 G - Shapley and TMC - Shapley have the same incremental time complexity – 𝒪 ⁡ ( n 2 ) {\mathcal{O}(n^{2})} .
Both estimators’ storage complexity is 𝒪 ⁡ ( n 2 ​ p ) {\mathcal{O}(n^{2}p)} . 
There is no free lunch, and G - Shapley’s speed-up is usually at the expense of lower influence estimation accuracy. 

 
 
 Efficient Nearest-Neighbor Shapley   
Since TMC - Shapley and G - Shapley rely on heuristics and assumptions to achieve tractability, neither method provides approximation guarantees.
In contrast,
 [ Jia+19b ] prove that for k k - nearest neighbors classification, SV pointwise influence can be calculated exactly in 𝒪 ⁡ ( n ​ lg ⁡ n ) {\mathcal{O}(n\lg n)} time.
Formally, SV’s characteristic function for k k NN classification is 

 

 
 | 
 ν k ​ NN ( D ) ≔ − 1 k ∑ y i ∈ Neigh ​ ( x te , D ) 𝟙 [ y i = y te ] , \nu_{k\text{NN}}(D)\coloneqq-\frac{1}{k}\sum_{y_{i}\in{\text{Neigh}(x_{\text{te}};D)}}{\mathbbm{1}[y_{i}=y_{\text{te}}]}\text{,} | 
 | 
 (25) | 
 

 where Neigh ​ ( x te , D ) {\text{Neigh}(x_{\text{te}};D)} is the set of k k neighbors in D D nearest to x te x_{\text{te}} and 𝟙 ​ [ ⋅ ] {\mathbbm{1}[\cdot]} is the indicator function. Each training instance either has no effect on Eq. ( 25 )’s value ( z i ∉ Neigh ​ ( x te , D ) {z_{i}\notin{\text{Neigh}(x_{\text{te}};D)}} or y i ≠ y te {y_{i}\neq y_{\text{te}}} ).
Otherwise, the training instance increases the value by one ( z i ∈ Neigh ​ ( x te , D ) {z_{i}\in{\text{Neigh}(x_{\text{te}};D)}} and y i = y te {y_{i}=y_{\text{te}}} ). 

 
 
 Assuming the training instances are sorted by increasing distance from x te x_{\text{te}} (i.e., x 1 x_{1} is closest to x te x_{\text{te}} and x n x_{n} is furthest), then z i z_{i} ’s pointwise k k NN Shapley influence is 

 

 
 | 
 ℐ k ​ NN-SV ​ ( z i , z te ) ≔ 𝟙 [ y n = y te ] n + ∑ j = i n − 1 𝟙 [ y j = y te ] − 𝟙 [ y j + 1 = y te ] k ​ min ⁡ { k ​ ; ​ j } j ​ . {\mathcal{I}_{k\text{NN-SV}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\coloneqq\frac{{\mathbbm{1}[y_{n}=y_{\text{te}}]}}{n}+\sum_{j=i}^{n-1}\frac{{\mathbbm{1}[y_{j}=y_{\text{te}}]}-{\mathbbm{1}[y_{j+1}=y_{\text{te}}]}}{k}\frac{\min\{k\mathord{\mathchar 59\relax}j\}}{j}\text{.} | 
 | 
 (26) | 
 

 Observe that ℐ k ​ NN-SV \mathcal{I}_{k\text{NN-SV}} ’s closed form is linear in n n and requires no retraining at all.
In fact, k k NN Shapley’s most computationally expensive component is sorting the training instances by distance from x te x_{\text{te}} .
Similar to k k NN LOO above, [ Jia+19b ] propose using k k NN Shapley as a surrogate SV estimator for more complex model classes.
For example, k k NN Shapley could be applied to the feature representations generated by a deep neural network. 

 
 
 Beta Shapley   
Recent work has also questioned the optimality of SV assigning uniform weight to each training subset size (see Eq. ( 17 )).
Counterintuitively, [ KZ22a ] show theoretically and empirically that influence estimates on larger training subsets are more affected by training noise than influence estimates on smaller subsets.
As such, rather than assigning all data subset sizes ( | D | \lvert D\rvert ) uniform weight, [ KZ22a ] argue that smaller training subsets should be prioritized.
Specifically, [ KZ22a ] propose Beta Shapley , which modifies vanilla SV by weighting the training-set sizes according to a positive skew (i.e., left-leaning) beta distribution. 

 
 
 SV has also been applied to study other types of influence beyond training set membership.
For example, Neuron Shapley applies SV to identify the model neurons that are most critical for a given prediction [ GZ20a ] .
 [ LL17a ] ’s [ LL17a ] SHAP is a very well-known tool that applies SV to measure feature importance.
For a comprehensive survey of Shapley value applications beyond training data influence, see the work of [ SN20a ] and a more recent update by [ Roz+22a ] . 

 
 
 Banzhaf Value   
Also a semivalue [ DNW81a ] , Banzhaf value [ Ban65a ] is closely related to Shapley value.
Formally, the Banzhaf value influence of z i ∈ 𝒟 {z_{i}\in\mathcal{D}} on test instance z te z_{\text{te}} is 

 

 
 | 
 ℐ Banzhaf ​ ( z i , z te ) ≔ 1 2 n − 1 ​ ∑ D ⊆ 𝒟 ∖ z i ℒ ⁡ ( z te , θ D ( T ) ) − ℒ ⁡ ( z te , θ D ∪ z i ( T ) ) ​ . {\mathcal{I}_{\text{Banzhaf}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\coloneqq\frac{1}{2^{n-1}}\sum_{D\subseteq\mathcal{D}^{\setminus z_{i}}}{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D})}-{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D\cup z_{i}})}\text{.} | 
 | 
 (27) | 
 

 Intuitively, the primary differences between Eqs. ( 17 ) and ( 27 ) is that Shapley value assigns each subset size ( | D | \lvert D\rvert ) equal weight while Banzhaf value assigns each subset ( D D ) equal weight.
 [ WJ23a ] prove that influence rankings based on Banzhaf value are more robust to training variance than both leave - one - out and Shapley value.
 [ WJ23a ] also empirically demonstrate that Banzhaf value can (significantly) outperform SV in practice. 

 
 
 Like SV, Banzhaf value has exponential time complexity.
Uniform, Monte Carlo sampling from power set 2 𝒟 ∖ z i 2^{\mathcal{D}\setminus z_{i}} provides an unbiased estimate of Eq. ( 27 ).
 [ WJ23a ] provide a more sophisticated Banzhaf value Monte Carlo sampling strategy they term maximum sample reuse (MSR).
MSR improves the estimates’ sample complexity by a factor of 𝒪 ⁡ ( n ) {\mathcal{O}(n)} over uniform Monte Carlo. 

 
 
 
 
 

## 5 Gradient-Based Influence Estimation

 
 For modern models, retraining even a few times to tune hyperparameters is very expensive.
In such cases, it is prohibitive to retrain an entire model just to gain insight into a single training instance’s influence. 

 
 
 For models trained using gradient descent, training instances only influence a model through training gradients.
Intuitively then, training data influence should be measurable when the right training gradients are analyzed.
This basic idea forms the basis of gradient-based influence estimation.
As detailed below, gradient-based influence estimators rely on Taylor-series approximations or risk stationarity.
These estimators also assume some degree of differentiability – either of just the loss function [ Yeh+18a ] or both the model and loss [ KL17a , Pru+20a , Che+21a ] . 

 
 
 The exact analytical framework each gradient-based method employs depends on the set of model parameters considered [ HL22a ] .
 Static, gradient-based methods – discussed first – estimate the effect of retraining by studying gradients w.r.t. final model parameters θ ( T ) \theta^{(T)} .
Obviously, a single set of model parameters provide limited insight into the entire optimization landscape, meaning static methods generally must make stronger assumptions.
In contrast, dynamic, gradient-based influence estimators reconstruct the training data’s influence by studying model parameters throughout training, e.g., θ ( 0 ) ; … ; θ ( T ) {\theta^{(0)}\mathchar 59\relax\ldots\mathchar 59\relax\theta^{(T)}} .
Analyzing these intermediary model parameters makes dynamic methods more computationally expensive in general, but it enables dynamic methods to make fewer assumptions. 

 
 
 This section concludes with a discussion of a critical limitation common to all existing gradient-based influence estimators – both static and dynamic.
This common weakness can cause gradient-based estimators to systematically overlook highly influential (groups of) training instances. 

 
 

### 5.1 Static, Gradient-Based Influence Estimation

 
 As mentioned above, static estimators are so named because they measure influence using only final model parameters θ ( T ) \theta^{(T)} .
Static estimators’ theoretical formulations assume stationarity (i.e., the model parameters have converged to a risk minimizer) and convexity . 

 
 
 Below we focus on two static estimators – influence functions [ KL17a ] and representer point [ Yeh+18a ] .
Each method takes very different approaches to influence estimation with the former being more general and the latter more scalable.
Both estimators’ underlying assumptions are generally violated in deep networks. 

 
 

#### 5.1.1 Influence Functions

 
 Along with Shapley value (Sec. 4.3 ),
 [ KL17a ] ’s [ KL17a ] influence functions is one of the best-known influence estimators.
The estimator derives its name from influence functions (also known as infinitesimal jackknife [ Jae72a ] ) in robust statistics [ Ham74a ] .
These early statistical analyses consider how a model changes if training instance z i z_{i} ’s weight is infinitesimally perturbed by ϵ i {\epsilon_{i}} .
More formally, consider the change in the empirical risk minimizer from 

 

 
 | 
 θ ( T ) = arg ​ min θ ⁡ 1 n ​ ∑ z ∈ 𝒟 ℒ ⁡ ( z , θ ) \theta^{(T)}=\argmin_{\theta}\frac{1}{n}\sum_{z\in\mathcal{D}}{\mathcal{L}(z;\theta)} | 
 | 
 (28) | 
 

 to 

 

 
 | 
 θ + ϵ i ( T ) = arg ​ min θ ⁡ 1 n ​ ∑ z ∈ 𝒟 ℒ ⁡ ( z , θ ) + ϵ i ​ ℒ ​ ( z i , θ ) ​ . \theta^{(T)}_{+\epsilon_{i}}=\argmin_{\theta}\frac{1}{n}\sum_{z\in\mathcal{D}}{\mathcal{L}(z;\theta)}+\epsilon_{i}{\mathcal{L}(z_{i};\theta)}\text{.} | 
 | 
 (29) | 
 

 
 
 Under the assumption that model f f and loss function ℓ \ell are twice-differentiable and strictly convex, [ CW82a ] prove via a first-order Taylor expansion that 

 

 
 | 
 d ​ θ + ϵ i ( T ) d ​ ϵ i | ϵ i = 0 = − ( H θ ( T ) ) − 1 ​ ∇ θ ℒ ​ ( z i , θ ( T ) ) ​ , \frac{d\theta^{(T)}_{+\epsilon_{i}}}{d\epsilon_{i}}\Big|_{\epsilon_{i}=0}=-{(H_{\theta}^{(T)})^{-1}}\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(T)})}\text{,} | 
 | 
 (30) | 
 

 where empirical risk Hessian H θ ( T ) ≔ 1 n ​ ∑ z ∈ 𝒟 ∇ θ 2 ​ ℒ ​ ( z , θ ( T ) ) {H_{\theta}^{(T)}\coloneqq\frac{1}{n}\sum_{z\in\mathcal{D}}\nabla_{\theta}^{2}{\mathcal{L}(z;\theta^{(T)})}} is by assumption positive definite.
 [ KL17a ] extend [ CW82a ] ’s result to consider the effect of this infinitesimal perturbation on z te z_{\text{te}} ’s risk, where 

 

 
 | 
 d ​ ℒ ​ ( z te , θ ( T ) ) d ​ ϵ i | ϵ i = 0 \displaystyle\frac{d{\mathcal{L}(z_{\text{te}};\theta^{(T)})}}{d\epsilon_{i}}\Big|_{\epsilon_{i}=0} | 
 = d ​ ℒ ​ ( z te , θ ( T ) ) d ​ θ + ϵ i ( T ) ⊺ ​ d ​ θ + ϵ i ( T ) d ​ ϵ i | ϵ i = 0 \displaystyle=\frac{d{\mathcal{L}(z_{\text{te}};\theta^{(T)})}}{d\theta^{(T)}_{+\epsilon_{i}}}^{\intercal}\frac{d\theta^{(T)}_{+\epsilon_{i}}}{d\epsilon_{i}}\Big|_{\epsilon_{i}=0} | 
 ⊳ \triangleright Chain rule | 
 | 
 (31) | 
 
 
 | 
 | 
 = − ∇ θ ℒ ​ ( z te , θ ( T ) ) ⊺ ​ ( H θ ( T ) ) − 1 ​ ∇ θ ℒ ​ ( z i , θ ( T ) ) ​ . \displaystyle=-\nabla_{\theta}{\mathcal{L}(z_{\text{te}};\theta^{(T)})}^{\intercal}{(H_{\theta}^{(T)})^{-1}}\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(T)})}\text{.} | 
 | 
 (32) | 
 

 
 
 Removing training instance z i z_{i} from 𝒟 \mathcal{D} is equivalent to ϵ i = − 1 n {\epsilon_{i}=-\frac{1}{n}} making the pointwise influence functions estimator 

 

 
 | 
 ℐ ^ IF ​ ( z i , z te ) ≔ 1 n ​ ∇ θ ℒ ​ ( z te , θ ( T ) ) ⊺ ​ ( H θ ( T ) ) − 1 ​ ∇ θ ℒ ​ ( z i , θ ( T ) ) ​ . {\widehat{\mathcal{I}}_{\textsc{IF}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\coloneqq\frac{1}{n}\nabla_{\theta}{\mathcal{L}(z_{\text{te}};\theta^{(T)})}^{\intercal}{(H_{\theta}^{(T)})^{-1}}\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(T)})}\text{.} | 
 | 
 (33) | 
 

 More intuitively, Eq. ( 33 ) is the influence functions’ estimate of the leave - one - out influence of z i z_{i} on z te z_{\text{te}} . 

 
 
 5.1.1.1 Time, Space, and Storage Complexity 

 
 Calculating inverse Hessian ( H θ ( T ) ) − 1 {(H_{\theta}^{(T)})^{-1}} directly requires 𝒪 ⁡ ( n ​ p 2 + p 3 ) {\mathcal{O}(np^{2}+p^{3})} time and 𝒪 ⁡ ( p 2 ) {\mathcal{O}(p^{2})} space [ KL17a ] .
For large models, this is clearly prohibitive.
Rather than computing ( H θ ( T ) ) − 1 {(H_{\theta}^{(T)})^{-1}} directly, [ KL17a ] instead estimate Hessian-vector product (HVP) 

 

 
 | 
 s test ≔ ( H θ ( T ) ) − 1 ​ ∇ θ ℒ ​ ( z te , θ ( T ) ) ​ , s_{\text{test}}\coloneqq{(H_{\theta}^{(T)})^{-1}}\nabla_{\theta}{\mathcal{L}(z_{\text{te}};\theta^{(T)})}\text{,} | 
 | 
 (34) | 
 

 with each training instance’s pointwise influence then 

 

 
 | 
 ℐ ^ IF ​ ( z i , z te ) = 1 n ​ s test ⊺ ​ ∇ θ ℒ ​ ( z i , θ ( T ) ) ​ . {\widehat{\mathcal{I}}_{\textsc{IF}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}=\frac{1}{n}s_{\text{test}}^{\intercal}\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(T)})}\text{.} | 
 | 
 (35) | 
 

 
 
 [ KL17a ] use the stochastic algorithm of [ Pea94a ] and [ ABH17a ] to estimate s test s_{\text{test}} .
This reduces influence functions’ complexity to just 𝒪 ⁡ ( n ​ p ) {\mathcal{O}(np)} time and 𝒪 ⁡ ( n + p ) {\mathcal{O}(n+p)} space.
Since s test s_{\text{test}} is specific to test instance z te z_{\text{te}} , s test s_{\text{test}} must be estimated for each test instance individually .
This increases influence functions’ computational cost but also means that influence functions require no additional storage. 

 
 
 
 5.1.1.2 Strengths and Weaknesses 

 
 Influence functions’ clear advantage over retraining-based methods is that influence functions eliminate the need to retrain any models.
Recall from Table that
retraining-based methods have a high upfront complexity of retraining but low incremental complexity.
Influence functions invert this computational trade-off.
While computing s test s_{\text{test}} can be slow (several hours for a single test instance [ Yeh+18a , Guo+21a , HL22a ] ), it is still significantly faster than retraining 𝒪 ⁡ ( n ) {\mathcal{O}(n)} models.
However, influence functions require that s test s_{\text{test}} is recalculated for each test instance.
Hence, when a large number of test instances need to be analyzed, retraining-based methods may actually be faster than influence functions through amortization of the retraining cost. 

 
 
 In addition, while influence functions can be very accurate on convex and some shallow models, the assumption that Hessian H θ ( T ) H_{\theta}^{(T)} is positive definite often does not hold for deep models [ BPF21a ] .
To ensure inverse ( H θ ( T ) ) − 1 {(H_{\theta}^{(T)})^{-1}} exists, [ KL17a ] add a small dampening coefficient to the matrix’s diagonal;
this dampener is a user-specified hyperparameter that is problematic to tune, since there may not be a ground-truth reference.
When this dampening hyperparameter is set too small, s test s_{\text{test}} estimation can diverge [ Yeh+18a , HNM19a , HL22a ] . 

 
 
 More generally, [ BPF21a ] show that training hyperparameters also significantly affect influence functions’ performance.
Specifically, [ BPF21a ] empirically demonstrate that model initialization, model width, model depth, weight-decay strength, and even the test instance being analyzed ( z te z_{\text{te}} ) all can negatively affect influence functions’ LOO estimation accuracy.
Their finding is supported by the analysis of [ ZZ22a ] who show that HVP estimation’s accuracy depends heavily on the model’s training regularizer with HVP accuracy “pretty low under weak regularization.” 

 
 
 [ Bae+22a ] also empirically analyze the potential sources of influence functions’ fragility on deep models.
 [ Bae+22a ] identify five common error sources, the first three of which are the most important. 18 18 
 18 
 
 
 
 [ Bae+22a , Table 1] provide a formal mathematical definition of influence functions’ five error sources. 

 
 • 
 
 Warm-start gap :
Influence functions more closely resembles performing fine-tuning close to final model parameters θ ( T ) \theta^{(T)} than retraining from scratch (i.e., starting from initial parameters θ ( 0 ) \theta^{(0)} ).
This difference in starting conditions can have a significant effect on the LOO estimate. 

 

 • 
 
 Proximity gap :
The error introduced by the dampening term included in the HVP ( s test s_{\text{test}} ) estimation algorithm. 

 

 • 
 
 Non-convergence gap :
The error due to final model parameters θ ( T ) \theta^{(T)} not being a stationary point, i.e., ∑ z i ∈ 𝒟 ∇ θ ℒ ​ ( z i , θ ( T ) ) ≠ 0 → {\sum_{z_{i}\in\mathcal{D}}\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(T)})}\neq\vec{0}} . 

 

 • 
 
 Linearization error :
The error induced by considering only a first-order Taylor approximation when deleting z i z_{i} and ignoring the potential effects of curvature on ℐ ^ IF ​ ( z i , z te ) {\widehat{\mathcal{I}}_{\textsc{IF}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)} [ BYF20a ] . 

 

 • 
 
 Solver error :
General error introduced by the specific solver used to estimate s test s_{\text{test}} . 

 

 
 Rather than estimating the LOO influence,
 [ Bae+22a ] argue that influence functions more closely estimate a different measure they term the proximal Bregman response function (PBRF).
 [ Bae+22a ] provide the intuition that PBRF “approximates the effect of removing a data point while trying to keep predictions consistent with the … trained model” [ Bae+22a ] .
Put simply, PBRF mimics a prediction-constrained LOO influence. 

 
 
 [ Bae+22a ] assert that PBRF can be applied in many of the same situations where LOO is useful.
 [ Bae+22a ] further argue that influence functions’ fragility reported by earlier works [ BPF21a , ZZ22a ] is primarily due to those works focusing on the “wrong question” of LOO.
When the “right question” is posed and influence functions are evaluated w.r.t. PBRF, influence functions give accurate answers. 

 
 
 
 5.1.1.3 Related Methods 

 
 Improving influence functions’ computational scalability has been a primary focus of follow-on work.
For instance, applying influence functions only to the model’s final linear (classification) layer has been considered with at best mixed results [ BBD20a , Yeh+22a ] . 19 19 
 19 
 
 
 
 Section 5.1.2.2 discusses some of the pitfalls associated with estimating influence using only a model’s final (linear) layer. 
Moreover, [ Guo+21a ] ’s [ Guo+21a ] fast influence functions ( FastIF ) integrate multiple speed-up heuristics.
First, they leverage the inherent parallelizability of [ Pea94a ] ’s [ Pea94a ] HVP estimation algorithm.
In addition, FastIF includes recommended hyperparameters for [ Pea94a ] ’s HVP algorithm that reduces its execution time by 50% on average. 

 
 
 Arnoldi-Based Influence Functions   
More recently, [ Sch+22a ] show that influence functions can be sped-up by three to four orders of magnitude by using a different algorithm to estimate s test s_{\text{test}} .
Specifically, [ Sch+22a ] use [ Arn51a ] ’s [ Arn51a ] famous algorithm that quickly finds the dominant (in absolute terms) eigenvalues and eigenvectors of H θ ( T ) H_{\theta}^{(T)} .
These dominant eigenvectors serve as the basis when projecting all gradient vectors to a lower-dimensional subspace.
 [ Sch+22a ] evaluate their revised influence functions estimator on large transformer networks (e.g., ViT - L32 with 300M parameters [ Dos+21a ] ), which are orders of magnitude larger than the simple networks [ KL17a ] consider [ Sch+23a ] . 

 
 
 Influence Functions for Decision Trees   
Another approach to speed up influence functions is to specialize the estimator to model architectures with favorable computational properties.
For example, [ Sha+18c ] ’s [ Sha+18c ] LeafInfluence method adapts influence functions to gradient boosted decision tree ensembles.
By assuming a fixed tree structure and then focusing only on the trees’ leaves, LeafInfluence ’s tree-based estimates are significantly faster than influence functions on deep models [ BHL23a ] . 

 
 
 Group Influence Functions   
A major strength of influence functions is that it is one of the few influence analysis methods that has been studied beyond the pointwise domain.
For example, [ Koh+19a ] ’s [ Koh+19a ] follow-on paper analyzes influence functions’ empirical performance estimating group influence.
In particular, [ Koh+19a ] consider coherent training data subpopulations whose removal is expected to have a large, broad effect on the model.
Even under naive assumptions of (pointwise) influence additivity, [ Koh+19a ] observe that simply summing influence functions estimates tends to underestimate the true group influence.
More formally, let D ⊆ 𝒟 {D\subseteq\mathcal{D}} be a coherent training-set subpopulation, then 

 

 
 | 
 ∑ z i ∈ D | ℐ ^ IF ​ ( z i , z te ) | | ℐ ⁡ ( D , z te ) | ​ . \sum_{z_{i}\in D}{\big\lvert{\widehat{\mathcal{I}}_{\textsc{IF}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\big\rvert} {\big\lvert{\mathcal{I}\left(D\mathchar 59\relax z_{\text{te}}\right)}\big\rvert}\text{.} | 
 | 
 (36) | 
 

 Nonetheless, influence functions’ additive group estimates tend to have strong rank correlation w.r.t. subpopulations’ true group influence.
In addition, [ BYF20a ] extend influence functions to directly account for subpopulation group effects by considering higher-order terms in influence functions’ Taylor-series approximation. 

 
 
 
 

#### 5.1.2 Representer Point Methods

 
 Unlike this section’s other gradient-based estimators, [ Yeh+18a ] ’s [ Yeh+18a ] representer point method does not directly rely on a Taylor-based expansion.
Instead, [ Yeh+18a ] build on [ SHS01a ] ’s [ SHS01a ] generalized representer theorem .
The derivation below assumes that model f f is linear.
 [ Yeh+18a ] use a simple “trick” to extend linear representer point methods to multilayer models like neural networks. 

 
 
 Representer-based methods rely on kernels , which are functions 𝒦 : 𝒳 × 𝒳 → {\mathcal{K}:\mathcal{X}\times\mathcal{X}\rightarrow\real} that measure the similarity between two vectors [ HSS08a ] .
 [ SHS01a ] ’s representer theorem proves that the optimal solution to a class of L 2 L_{2} regularized functions can be reformulated as a weighted sum of the training data in kernel form.
Put simply, representer methods decompose the predictions of specific model classes into the individual contributions (i.e., influence) of each training instance.
This makes influence estimation a natural application of the representer theorem. 

 
 
 Consider regularized empirical risk minimization where optimal parameters satisfy 20 20 
 20 
 
 
 
 Eq. ( 37 ) is slightly different than [ Yeh+18a ] ’s [ Yeh+18a ] presentation.
This alternate formulation requires specifying λ \lambda 2 × 2\times larger.
We selected this alternate presentation to provide consistency with later work [ Tsa+23a , Che+21a ] . 

 

 
 | 
 θ ∗ ≔ arg ​ min θ ⁡ 1 n ​ ∑ z i ∈ 𝒟 ℒ ⁡ ( z i , θ ) + λ 2 ​ ∥ θ ∥ 2 ​ , \theta^{*}\coloneqq\argmin_{\theta}\frac{1}{n}\sum_{z_{i}\in\mathcal{D}}{\mathcal{L}(z_{i};\theta)}+\frac{\lambda}{2}\lVert\theta\rVert_{2}\text{,} | 
 | 
 (37) | 
 

 with λ 0 {\lambda 0} the L 2 L_{2} regularization strength.
Note that Eq. ( 37 ) defines minimizer θ ∗ \theta^{*} slightly differently than the last section ( 28 ) since the representer theorem requires regularization. 

 
 
 Empirical risk minimizers are stationary points meaning 

 

 
 | 
 ∇ θ ( 1 n ​ ∑ z i ∈ 𝒟 ℒ ⁡ ( z i , θ ∗ ) + λ ​ ∥ θ ∗ ∥ 2 ) = 0 → ​ , \nabla_{\theta}\bigg(\frac{1}{n}\sum_{z_{i}\in\mathcal{D}}{\mathcal{L}(z_{i};\theta^{*})}+\lambda\lVert\theta^{*}\rVert_{2}\bigg)=\vec{0}\text{,} | 
 | 
 (38) | 
 

 where 0 → \vec{0} is the p p - dimensional, zero vector. The above simplifies to 

 

 
 | 
 1 n ​ ∑ z i ∈ 𝒟 ∂ ℒ ⁡ ( z i , θ ∗ ) ∂ θ + λ ​ θ ∗ \displaystyle\frac{1}{n}\sum_{z_{i}\in\mathcal{D}}\frac{\partial{\mathcal{L}(z_{i};\theta^{*})}}{\partial\theta}+\lambda\theta^{*} | 
 = 0 → \displaystyle=\vec{0} | 
 | 
 (39) | 
 
 
 | 
 θ ∗ \displaystyle\theta^{*} | 
 = − 1 λ ​ n ∑ z i ∈ 𝒟 ∂ ℒ ⁡ ( z i , θ ∗ ) ∂ θ . \displaystyle=-\frac{1}{\lambda n}\sum_{z_{i}\in\mathcal{D}}\frac{\partial{\mathcal{L}(z_{i};\theta^{*})}}{\partial\theta}\text{.} | 
 | 
 (40) | 
 

 
 
 For a linear model where f ⁡ ( x , θ ) = θ ⊺ ​ x ≕ y ^ {{f(x;\theta)}=\theta^{\intercal}x\eqqcolon\widehat{y}} , Eq. ( 40 ) further simplifies via the chain rule to 

 

 
 | 
 θ ∗ \displaystyle\theta^{*} | 
 = − 1 λ ​ n ∑ ( x i ​ ; ​ y i ) ∈ 𝒟 ∂ ℓ ⁡ ( f ⁡ ( x i , θ ∗ ) , y i ) ∂ θ \displaystyle=-\frac{1}{\lambda n}\sum_{(x_{i}\mathord{\mathchar 59\relax}y_{i})\in\mathcal{D}}\frac{\partial{\ell({f(x_{i};\theta^{*})}\mathchar 59\relax y_{i})}}{\partial\theta} | 
 | 
 (41) | 
 
 
 | 
 | 
 = − 1 λ ​ n ∑ ( x i ​ ; ​ y i ) ∈ 𝒟 ∂ ℓ ⁡ ( y ^ i , y i ) ∂ y ^ ∂ y ^ i ∂ θ \displaystyle=-\frac{1}{\lambda n}\sum_{(x_{i}\mathord{\mathchar 59\relax}y_{i})\in\mathcal{D}}\frac{\partial{\ell(\widehat{y}_{i}\mathchar 59\relax y_{i})}}{\partial\widehat{y}}\,\frac{\partial\widehat{y}_{i}}{\partial\theta} | 
 ⊳ \triangleright Chain Rule | 
 | 
 (42) | 
 
 
 | 
 | 
 = − 1 λ ​ n ∑ ( x i ​ ; ​ y i ) ∈ 𝒟 ∂ ℓ ⁡ ( y ^ i , y i ) ∂ y ^ x i \displaystyle=-\frac{1}{\lambda n}\sum_{(x_{i}\mathord{\mathchar 59\relax}y_{i})\in\mathcal{D}}\frac{\partial{\ell(\widehat{y}_{i}\mathchar 59\relax y_{i})}}{\partial\widehat{y}}\,x_{i} | 
 ⊳ \triangleright y ^ = θ ⊺ ​ x \widehat{y}=\theta^{\intercal}x | 
 | 
 (43) | 
 
 
 | 
 | 
 = ∑ ( x i ​ ; ​ y i ) ∈ 𝒟 α i ​ x i ​ , \displaystyle=\sum_{(x_{i}\mathord{\mathchar 59\relax}y_{i})\in\mathcal{D}}\alpha_{i}x_{i}\text{,} | 
 | 
 (44) | 
 

 where ℓ \ell is any once-differentiable loss function, ∂ ℓ ⁡ ( y ^ i , y i ) ∂ y ^ \frac{\partial{\ell(\widehat{y}_{i}\mathchar 59\relax y_{i})}}{\partial\widehat{y}} is the gradient of just the loss function itself w.r.t. model output y ^ \widehat{y} , 21 21 
 21 
 
 
 
 In the case of classification, ∂ ℓ ⁡ ( y ^ i , y i ) ∂ y ^ \frac{\partial{\ell(\widehat{y}_{i}\mathchar 59\relax y_{i})}}{\partial\widehat{y}} has | 𝒴 | \lvert\mathcal{Y}\rvert - dimensions, i.e., its dimension equals the number of classes. 
and α i ≔ − 1 λ ​ n ​ ∂ ℓ ⁡ ( y ^ i , y i ) ∂ y ^ {\alpha_{i}\coloneqq-\frac{1}{\lambda n}\frac{\partial{\ell(\widehat{y}_{i}\mathchar 59\relax y_{i})}}{\partial\widehat{y}}} is the i i - th training instance’s representer value .
 [ Yeh+18a ] provide the intuition that a larger magnitude α i \alpha_{i} indicates that training instance z i z_{i} has larger influence on the final model parameters θ ∗ \theta^{*} . 

 
 
 Following [ SHS01a ] ’s [ SHS01a ] representer theorem kernelized notation, the training set’s group influence on test instance z te z_{\text{te}} for any linear model is 

 

 
 | 
 ℐ RP ​ ( 𝒟 , z te ) = ∑ i = 1 n α i ​ x i ⊺ ​ x te | y te = ∑ i = 1 n 𝒦 ⁡ ( x i , x te , α i ) | y te = ∑ i = 1 n ℐ RP ​ ( z i , z te ) ​ , {\mathcal{I}_{\textsc{RP}}\left(\mathcal{D}\mathchar 59\relax z_{\text{te}}\right)}=\sum_{i=1}^{n}\alpha_{i}x_{i}^{\intercal}x_{\text{te}}\,|_{y_{\text{te}}}=\sum_{i=1}^{n}{\mathcal{K}(x_{i}\mathchar 59\relax x_{\text{te}}\mathchar 59\relax\alpha_{i})\,|_{y_{\text{te}}}}=\sum_{i=1}^{n}{\mathcal{I}_{\textsc{RP}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\text{,} | 
 | 
 (45) | 
 

 where kernel function 𝒦 ⁡ ( x i , x te ​ ; ​ α i ) ≔ α i ​ x i ⊺ ​ x te {{\mathcal{K}(x_{i}\mathchar 59\relax x_{\text{te}}\mathord{\mathchar 59\relax}\alpha_{i})}\coloneqq\alpha_{i}x_{i}^{\intercal}x_{\text{te}}} returns a vector.
 𝒦 ⁡ ( x i , x te , α i ) | y te {{\mathcal{K}(x_{i}\mathchar 59\relax x_{\text{te}}\mathchar 59\relax\alpha_{i})\,|_{y_{\text{te}}}}} denotes the kernel value’s y te y_{\text{te}} - th dimension.
Then, z i z_{i} ’s pointwise linear representer point influence on z te z_{\text{te}} is 

 

 
 | 
 ℐ RP ​ ( z i , z te ) = α i ​ x i ⊺ ​ x te | y te = 𝒦 ⁡ ( x i , x te , α i ) | y te ​ . {\mathcal{I}_{\textsc{RP}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}=\alpha_{i}x_{i}^{\intercal}x_{\text{te}}\,|_{y_{\text{te}}}={\mathcal{K}(x_{i}\mathchar 59\relax x_{\text{te}}\mathchar 59\relax\alpha_{i})\,|_{y_{\text{te}}}}\text{.} | 
 | 
 (46) | 
 

 
 
 Extending Representer Point to Multilayer Models 
Often, linear models are insufficiently expressive, with multilayer models used instead.
In such cases, the representer theorem above does not directly apply.
To workaround this limitation, [ Yeh+18a ] rely on what they (later) term last layer similarity [ Yeh+22a ] . 

 
 
 Formally, [ Yeh+18a ] partition the model parameters θ ( T ) = [ θ ˙ ( T ) ​ θ ¨ ( T ) ] {\theta^{(T)}=[\penalty\ \dot{\theta}^{(T)}\penalty\ \penalty\ \ddot{\theta}^{(T)}\penalty\ ]} 
into two subsets, where θ ¨ ( T ) \ddot{\theta}^{(T)} is the last linear (i.e., classification) layer’s parameters and θ ˙ ( T ) ≔ θ ( T ) ∖ θ ¨ ( T ) {\dot{\theta}^{(T)}\coloneqq\theta^{(T)}\setminus\ddot{\theta}^{(T)}} is all other model parameters.
Since θ ¨ ( T ) \ddot{\theta}^{(T)} is simply a linear function, the representer theorem analysis above still applies to it.
 [ Yeh+18a ] treat the other parameters, θ ˙ ( T ) \dot{\theta}^{(T)} , as a fixed feature extractor and ignore them in their influence analysis. 

 
 
 To use [ Yeh+18a ] ’s [ Yeh+18a ] multilayer trick,
one small change to Eq. ( 46 ) is required.
In multilayer models, the final (linear) layer does not operate over feature vectors x i x_{i} and x te x_{\text{te}} directly.
Instead, the final layer only sees an intermediate feature representation.
For arbitrary feature vector x x , let 𝐟 \mathbf{f} be the feature representation generated by model parameters θ ˙ ( T ) \dot{\theta}^{(T)} , i.e., vector 𝐟 \mathbf{f} is the input to the model’s last linear layer given input x x .
Then the representer point influence estimator for a multilayer model is 

 

 
 | 
 ℐ ^ RP ​ ( z i , z te ) ≔ α i ​ 𝐟 i ⊺ ​ 𝐟 te | y te = 𝒦 ⁡ ( 𝐟 i , 𝐟 te ​ ; ​ α i ) | y te ​ . {\widehat{\mathcal{I}}_{\textsc{RP}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\coloneqq\alpha_{i}\mathbf{f}_{i}^{\intercal}\mathbf{f}_{\text{te}}\,|_{y_{\text{te}}}={\mathcal{K}(\mathbf{f}_{i}\mathchar 59\relax\mathbf{f}_{\text{te}}\mathord{\mathchar 59\relax}\alpha_{i})}\,|_{y_{\text{te}}}\text{.} | 
 | 
 (47) | 
 

 
 
 5.1.2.1 Time, Space, and Storage Complexity 

 
 Treating as constants feature representation dimension | 𝐟 | \lvert\mathbf{f}\rvert and the overhead to calculate ∂ ℓ ⁡ ( y ^ i , y i ) ∂ y ^ \frac{\partial{\ell(\widehat{y}_{i}\mathchar 59\relax y_{i})}}{\partial\widehat{y}} , estimating the entire training set’s representer point influence only requires calculating n n dot products. This only takes 𝒪 ⁡ ( n ) {\mathcal{O}(n)} time and 𝒪 ⁡ ( n + p ) {\mathcal{O}(n+p)} space with no additional storage requirements. 

 
 
 
 5.1.2.2 Strengths and Weaknesses 

 
 Representer point’s primary advantage is its theoretical and computational simplicity.
Eq. ( 47 ) only considers the training and test instances’ final feature representations and loss function gradient ∂ ℓ ⁡ ( y ^ i , y i ) ∂ y ^ \frac{\partial{\ell(\widehat{y}_{i}\mathchar 59\relax y_{i})}}{\partial\widehat{y}} .
Hence, the majority of representer point’s computation is forward-pass only and can be sped up using batching.
This translates to representer point being very fast – several orders of magnitude faster than influence functions and Section 5.2 ’s dynamic estimators [ HL21a ] . 

 
 
 However, representer points’ simplicity comes at a cost.
First, at the end of training, it is uncommon that a model’s final linear layer has converged to a stationary point.
Before applying their method, [ Yeh+18a ] recommend freezing all model layers except the final one (i.e., freezing θ ˙ ( T ) \dot{\theta}^{(T)} ) and then fine-tuning the classification layer ( θ ¨ ( T ) \ddot{\theta}^{(T)} ) until convergence/stationarity.
Without this extra fine-tuning, representer point’s stationarity assumption does not hold, and poor influence estimation accuracy is expected.
Beyond just complicating the training procedure itself, this extra training procedure also complicates comparison with other influence methods since it may require evaluating the approaches on different parameters. 

 
 
 Moreover, by focusing exclusively on the model’s final linear (classification) layer, representer point methods may miss influential behavior that is clearly visible in other layers.
For example, [ HL22a ] demonstrate that while some training-set attacks are clearly visible in a network’s final layer, other attacks are only visible in a model’s first layer – despite both attacks targeting the same model architecture and dataset.
In their later paper, [ Yeh+22a ] acknowledge the disadvantages of considering only the last layer writing, “that choice critically affects the similarity component of data influence and leads to inferior results.”
 [ Yeh+22a ] further state that the feature representations in the final layer – and by extension representer point’s influence estimates – can be “too reductive.” 

 
 
 In short, [ Yeh+18a ] ’s [ Yeh+18a ] representer point method is highly scalable and efficient but is only suitable to detect behaviors that are obvious in the model’s final linear layer. 

 
 
 
 5.1.2.3 Related Methods 

 
 Given the accuracy limitations of relying on last-layer similarity, limited follow-on work has adapted representer-point methods. 

 
 
 Adapting Representer Point to Decision Trees   
 [ BHL23a ] extend representer point methods to decision forests via their Tree-ensemble Representer Point Examples (TREX) estimator. Specifically, they use supervised tree kernels – which provide an encoding of a tree’s learned representation structure [ DG14a , He+14a ] – for similarity comparison. 

 
 
 Making Representer Point More Robust   
In addition, [ SWS21a ] propose representer point selection based on a local Jacobian expansion (RPS - LJE), which can be viewed as a generalizing [ Yeh+18a ] ’s [ Yeh+18a ] base method.
Rather than relying on [ SHS01a ] ’s [ SHS01a ] representer theorem, [ SWS21a ] ’s formulation relies on a first-order Taylor expansion that estimates the difference between the final model parameters and a true stationary point.
RPS - LJE still follows vanilla representer point’s kernelized decomposition where ℐ ^ RP ​ ( z i , z te ) ≔ α i ​ 𝐟 i ⊺ ​ 𝐟 te {{\widehat{\mathcal{I}}_{\textsc{RP}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\coloneqq\alpha_{i}\mathbf{f}_{i}^{\intercal}\mathbf{f}_{\text{te}}} , albeit with α i \alpha_{i} defined differently. 

 
 
 RPS - LJE addresses two weaknesses of [ Yeh+18a ] ’s [ Yeh+18a ] base approach.
First, [ SWS21a ] ’s formulation does not presume that θ ( T ) \theta^{(T)} is a stationary point.
Therefore, RPS - LJE does not require post-training fine-tuning to enforce stationarity.
Second, as Section 5.3 discusses in detail, gradient-based estimators, including vanilla representer point, tend to mark as most influential those training instances with the largest loss values.
This leads to all test instances from a given class having near identical top - k influence rankings.
RPS - LJE’s alternate definition of α i \alpha_{i} is less influenced by a training instance’s loss value, which enables RPS - LJE to generate more semantically meaningful influence rankings. 

 
 
 Extending Representer Point to Other Regularizers   
 [ Yeh+18a ] ’s [ Yeh+18a ] representer point formulation exclusively considers L 2 L_{2} - regularized models.
Intuitively, regularization’s role is to encourage the model parameters to meet certain desired properties, which may necessitate the use of alternate regularizers.
For example, L 1 L_{1} regularization is often used to induce sparse minimizers. 

 
 
 Recently,
 [ Tsa+23a ] propose high-dimensional representers , a novel extension of [ Yeh+18a ] ’s [ Yeh+18a ] representer theorem to additional types of regularization.
Specifically, [ Tsa+23a ] consider decomposable regularization functions [ Neg+12a ] .
Formally, a regularization function r : p → ≥ 0 {r:\real^{p}\rightarrow\real_{{\geq}0}} is decomposable w.r.t. two subspaces 𝒰 ; 𝒱 ⊆ p {\mathcal{U}\mathchar 59\relax\mathcal{V}\subseteq\real^{p}} if ∀ u ∈ 𝒰 {\forall\,u\in\mathcal{U}} and ∀ v ∈ 𝒱 {\forall\,v\in\mathcal{V}} , 

 

 
 | 
 r ⁡ ( u + v ) = r ⁡ ( u ) + r ⁡ ( v ) ​ . {r(u+v)}={r(u)}+{r(v)}\text{.} | 
 | 
 (48) | 
 

 Examples of decomposable regularizers include L 1 L_{1} - norm [ Tib96a ] and the matrix nuclear norm [ Yua+07a , Rec11a , Yan+17a ] . 

 
 
 High-dimensional representers follow Eq. ( 47 )’s kernelized form.
Representer value α i ≔ − 1 λ ​ n ​ ∂ ℓ ⁡ ( y ^ i , y i ) ∂ y ^ {\alpha_{i}\coloneqq-\frac{1}{\lambda n}\frac{\partial{\ell(\widehat{y}_{i}\mathchar 59\relax y_{i})}}{\partial\widehat{y}}} still quantifies the global importance of each training instance.
Moreover, the similarity between training instance x i x_{i} and test instance x te x_{\text{te}} is still measured via a kernel function ( 𝒦 \mathcal{K} ).
The only difference is that the kernels are specialized for these alternate regularizers; specifically the kernels are based on the decomposable regularization function’s sub - differential [ Neg+12a ] . 

 
 
 
 
 

### 5.2 Dynamic, Gradient-Based Influence Estimation

 
 All preceding influence methods – static, gradient-based and retraining-based – define and estimate influence using only final model parameters, θ D ( T ) \theta^{(T)}_{D} , where D ⊆ 𝒟 {D\subseteq\mathcal{D}} .
These final parameters only provide a snapshot into a training instance’s possible effect.
Since neural network training is NP-complete [ BR92a ] , it can be provably difficult to reconstruct how each training instance affected the training process. 

 
 
 As an intuition, an influence estimator that only considers the final model parameters is akin to only reading the ending of a book.
One might be able to draw some big-picture insights, but the finer details of the story are most likely lost.
Applying a dynamic influence estimator is like reading a book from beginning to end.
By comprehending the whole influence “story,” dynamic methods can observe training data relationships – both fine-grained and general – that other estimators miss. 

 
 
 Since test instance z te z_{\text{te}} may not be known before model training, in-situ influence analysis may not be possible.
Instead, as shown in Alg. 1 , intermediate model parameters Θ ⊆ { θ ( 0 ) ; … ; θ ( T − 1 ) } {\Theta\subseteq\{\theta^{(0)}\mathchar 59\relax\ldots\mathchar 59\relax\theta^{(T-1)}\}} are stored during training for post hoc influence analysis. 22 22 
 22 
 
 
 
 In practice, only a subset of { θ ( 0 ) ; … ; θ ( T − 1 ) } \{\theta^{(0)}\mathchar 59\relax\ldots\mathchar 59\relax\theta^{(T-1)}\} is actually stored.
Heuristics are then applied to this subset to achieve acceptable influence estimation error [ Pru+20a , HL22a ] . 

 
 
 Below we examine two divergent approaches to dynamic influence estimation – the first defines a novel definition of influence while the second estimates leave - one - out influence with fewer assumptions than influence functions. 

 
 
 Algorithm 1 Dynamic influence estimation’s training phase 
 
 
 1: 
 
 
 Training set 𝒟 \mathcal{D} ;
iteration count T T ;
learning rates η ( 1 ) ; … ; η ( T ) {\eta^{(1)}\mathchar 59\relax\ldots\mathchar 59\relax\eta^{(T)}} ;
batch sizes b ( 1 ) ; … ; b ( T ) {b^{(1)}\mathchar 59\relax\ldots\mathchar 59\relax b^{(T)}} ;
and
initial parameters θ ( 0 ) \theta^{(0)} 
 
 
 
 2: 
 
 
 Final parameters θ ( T ) \theta^{(T)} and stored parameter set Θ \Theta 
 
 
 
 3: 
 
 
 Θ ← ∅ \Theta\leftarrow\emptyset 
 
 
 
 4: 
 
 
 for t ← 1 ​  to  ​ T t\leftarrow 1\textbf{ to }T do 
 
 
 
 5: 
 
 
     Θ ← Θ ∪ { θ ( t − 1 ) } \Theta\leftarrow\Theta\cup\{\theta^{(t-1)}\} ⊳ \triangleright Store intermediate params.
 
 
 
 6: 
 
 
     ℬ ( t ) ∼ b ( t ) 𝒟 \mathcal{B}^{(t)}\,\stackrel{{\scriptstyle b^{(t)}}}{{\sim}}\,\mathcal{D} 
 
 
 
 7: 
 
 
     θ ( t ) ← Update ​ ( η ( t ) , θ ( t − 1 ) , ℬ ( t ) ) \theta^{(t)}\leftarrow\textsc{Update}(\eta^{(t)}\mathchar 59\relax\theta^{(t-1)}\mathchar 59\relax\mathcal{B}^{(t)}) 
 
 
 
 8: 
 
 
 return θ ( T ) \theta^{(T)} , Θ \Theta 
 
 
 
 
 

#### 5.2.1 TracIn – Tracing Gradient Descent

 
 Fundamentally, all preceding methods define influence w.r.t. changes to the training set.
 [ Pru+20a ] take an orthogonal perspective.
They treat training set 𝒟 \mathcal{D} as fixed, and consider the change in model parameters as a function of time , or more precisely, the training iterations. 

 
 
 Vacuously, the training set’s group influence on test instance z te z_{\text{te}} is 

 

 
 | 
 ℐ ⁡ ( 𝒟 , z te ) = ℒ ⁡ ( z te , θ ( 0 ) ) − ℒ ⁡ ( z te , θ ( T ) ) ​ . {\mathcal{I}\left(\mathcal{D}\mathchar 59\relax z_{\text{te}}\right)}={\mathcal{L}(z_{\text{te}};\theta^{(0)})}-{\mathcal{L}(z_{\text{te}};\theta^{(T)})}\text{.} | 
 | 
 (49) | 
 

 In words, training set 𝒟 \mathcal{D} causes the entire change in test loss between random initial parameters θ ( 0 ) \theta^{(0)} and final parameters θ ( T ) \theta^{(T)} .
Eq. ( 49 ) decomposes by training iteration t t as 

 

 
 | 
 ℐ ⁡ ( 𝒟 , z te ) = ∑ t = 1 T ( ℒ ⁡ ( z te , θ ( t − 1 ) ) − ℒ ⁡ ( z te , θ ( t ) ) ) ​ . {\mathcal{I}\left(\mathcal{D}\mathchar 59\relax z_{\text{te}}\right)}=\sum_{t=1}^{T}\left({\mathcal{L}(z_{\text{te}};\theta^{(t-1)})}-{\mathcal{L}(z_{\text{te}};\theta^{(t)})}\right)\text{.} | 
 | 
 (50) | 
 

 
 
 Consider training a model with vanilla stochastic gradient descent, where each training minibatch ℬ ( t ) \mathcal{B}^{(t)} is a single instance and gradient updates have no momentum [ RHW86a ] .
Here, each iteration t t has no effect on any other iteration beyond the model parameters themselves.
Combining this with singleton batches enables attribution of each parameter change to a single training instance, namely whichever instance was in ℬ ( t ) \mathcal{B}^{(t)} .
Under this regime, [ Pru+20a ] define the ideal TracIn pointwise influence as 

 

 
 | 
 ℐ TracIn ​ ( z i , z te ) ≔ ∑ t z i = ℬ ( t ) ( ℒ ⁡ ( z te , θ ( t − 1 ) ) − ℒ ⁡ ( z te , θ ( t ) ) ) ​ , {\mathcal{I}_{\text{TracIn{}}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\coloneqq\sum_{\begin{subarray}{c}t\\
z_{i}=\mathcal{B}^{(t)}\end{subarray}}\left({\mathcal{L}(z_{\text{te}};\theta^{(t-1)})}-{\mathcal{L}(z_{\text{te}};\theta^{(t)})}\right)\text{,} | 
 | 
 (51) | 
 

 where the name “TracIn” derives from “tracing gradient descent influence.”
Eq. ( 50 ) under vanilla stochastic gradient descent decomposes into the sum of all pointwise influences 

 

 
 | 
 ℐ ⁡ ( 𝒟 , z te ) = ∑ i = 1 n ( ∑ t z i = ℬ ( t ) ℒ ⁡ ( z te , θ ( t − 1 ) ) − ℒ ⁡ ( z te , θ ( t ) ) ) = ∑ i = 1 n ℐ TracIn ​ ( z i , z te ) ​ . {\mathcal{I}\left(\mathcal{D}\mathchar 59\relax z_{\text{te}}\right)}=\sum_{i=1}^{n}\Bigg(\sum_{\begin{subarray}{c}t\\
z_{i}=\mathcal{B}^{(t)}\end{subarray}}{\mathcal{L}(z_{\text{te}};\theta^{(t-1)})}-{\mathcal{L}(z_{\text{te}};\theta^{(t)})}\Bigg)=\sum_{i=1}^{n}{\mathcal{I}_{\text{TracIn{}}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\text{.} | 
 | 
 (52) | 
 

 
 
 While the ideal TracIn influence has a strong theoretical motivation, its assumption of singleton batches and vanilla stochastic gradient descent is unrealistic in practice.
To achieve reasonable training times, modern models train on batches of up to hundreds of thousands or millions of instances.
Training on a single instance at a time would be far too slow [ YGG17a , Goy+17a , Bro+20a ] . 

 
 
 A naive fix to Eq. ( 51 ) to support non-singleton batches assigns the same influence to all instances in the minibatch, or more formally, divide the change in loss ℒ ⁡ ( z te , θ ( t − 1 ) ) − ℒ ⁡ ( z te , θ ( t ) ) {{\mathcal{L}(z_{\text{te}};\theta^{(t-1)})}-{\mathcal{L}(z_{\text{te}};\theta^{(t)})}} by batch size | ℬ ( t ) | \lvert\mathcal{B}^{(t)}\rvert for each z i ∈ ℬ ( t ) {z_{i}\in\mathcal{B}^{(t)}} .
This naive approach does not differentiate those instances in batch ℬ ( t ) \mathcal{B}^{(t)} that had positive influence on the prediction from those that made the prediction worse. 

 
 
 Instead, [ Pru+20a ] estimate the contribution of each training instance within a minibatch via a first-order Taylor approximation. Formally, 

 

 
 | 
 ℒ ⁡ ( z te , θ ( t ) ) ≈ ℒ ⁡ ( z te , θ ( t − 1 ) ) + ∇ θ ℒ ​ ( z te , θ ( t − 1 ) ) ⊺ ​ ( θ ( t ) − θ ( t − 1 ) ) ​ . {\mathcal{L}(z_{\text{te}};\theta^{(t)})}\approx{\mathcal{L}(z_{\text{te}};\theta^{(t-1)})}+{{\nabla_{\theta}{\mathcal{L}(z_{\text{te}};\theta^{(t-1)})}}^{\intercal}\,(\theta^{(t)}-\theta^{(t-1)})}\text{.} | 
 | 
 (53) | 
 

 Under gradient descent without momentum, the change in model parameters is directly determined by the batch instances’ gradients, i.e., 

 

 
 | 
 θ ( t ) − θ ( t − 1 ) = − η ( t ) | ℬ ( t ) | ∑ z i ∈ ℬ ( t ) ∇ θ ℒ ( z i ; θ ( t − 1 ) ) , \theta^{(t)}-\theta^{(t-1)}=-\frac{\eta^{(t)}}{\lvert\mathcal{B}^{(t)}\rvert}\sum_{z_{i}\in\mathcal{B}^{(t)}}{\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(t-1)})}}\text{,} | 
 | 
 (54) | 
 

 where η ( t ) \eta^{(t)} is iteration t t ’s learning rate. 

 
 
 Combining Eqs. ( 52 ) to ( 54 ), the TracIn pointwise influence estimator is 

 

 
 | 
 ℐ ^ TracIn ​ ( z i , z te ) ≔ ∑ t z i ∈ ℬ ( t ) η ( t ) | ℬ ( t ) | ​ ∇ θ ℒ ​ ( z i , θ ( t − 1 ) ) ⊺ ​ ∇ θ ℒ ​ ( z te , θ ( t − 1 ) ) ​ , {\widehat{\mathcal{I}}_{\text{TracIn{}}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\coloneqq\sum_{\begin{subarray}{c}t\\
z_{i}\in\mathcal{B}^{(t)}\end{subarray}}\frac{\eta^{(t)}}{\lvert\mathcal{B}^{(t)}\rvert}\,{{\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(t-1)})}}^{\intercal}\,{\nabla_{\theta}{\mathcal{L}(z_{\text{te}};\theta^{(t-1)})}}}\text{,} | 
 | 
 (55) | 
 

 with the complete TracIn influence estimation procedure shown in Alg. 2 . 

 
 
 A More “Practical” TracIn 
Training’s stochasticity can negatively affect the performance of both ideal TracIn ( 51 ) and the TracIn influence estimator ( 55 ).
As an intuition, consider when the training set contains two identical copies of some instance.
All preceding gradient-based methods assign those two identical instances the same influence score.
However, it is unlikely that those two training instances will always appear together in the same minibatch.
Therefore, ideal TracIn almost certainly assigns these identical training instances different influence scores.
These assigned scores may even be vastly different – by up to several orders of magnitude [ HL22a ] .
This is despite identical training instances always having the same expected TracIn influence. 

 
 
 [ Pru+20a ] recognize randomness’s effect on TracIn and propose the TracIn Checkpoint influence estimator (TracInCP) as a “practical” alternative.
Rather than retrace all of gradient descent, TracInCP considers only a subset of the training iterations (i.e., checkpoints) 𝒯 ⊆ [ T ] {\mathcal{T}\subseteq{[T]}} .
More importantly, at each t ∈ 𝒯 {t\in\mathcal{T}} , all training instances are analyzed – not just those in recent batches.
Eq. ( 56 ) formalizes TracInCP, with its modified influence estimation procedure shown in Alg. 3 . 

 

 
 | 
 ℐ ^ TracInCP ​ ( z i , z te ) ≔ ∑ t ∈ 𝒯 η ( t ) ​ ∇ θ ℒ ​ ( z i , θ ( t − 1 ) ) ⊺ ​ ∇ θ ℒ ​ ( z te , θ ( t − 1 ) ) {\widehat{\mathcal{I}}_{\text{TracIn{}CP}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\coloneqq\sum_{t\in\mathcal{T}}\eta^{(t)}\,{{\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(t-1)})}}^{\intercal}\,{\nabla_{\theta}{\mathcal{L}(z_{\text{te}};\theta^{(t-1)})}}} | 
 | 
 (56) | 
 

 
 
 Observe that, unlike TracIn, TracInCP assigns identical training instances the same influence estimate.
Therefore, TracInCP more closely estimates expected influence than TracIn.
 [ Pru+20a ] use TracInCP over TracIn in much of their empirical evaluation.
Other work has also shown that TracInCP routinely outperforms TracIn on many tasks [ HL22a ] . 

 
 
 
 
 
 Algorithm 2 TracIn influence estimation 
 
 
 1: 
 
 
 Training param. set Θ \Theta ;
iteration count T T ;
batches ℬ ( 1 ) ; … ; ℬ ( T ) {\mathcal{B}^{(1)}\mathchar 59\relax\ldots\mathchar 59\relax\mathcal{B}^{(T)}} ;
learning rates η ( 1 ) ; … ; η ( T ) {\eta^{(1)}\mathchar 59\relax\ldots\mathchar 59\relax\eta^{(T)}} ;
training instance z i z_{i} ;
and
test example z te z_{\text{te}} 
 
 
 
 2: 
 
 
 TracIn influence estimate ℐ ^ TracIn ​ ( z i , z te ) {\widehat{\mathcal{I}}_{\text{TracIn{}}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)} 
 
 
 
 3: 
 
 
 ℐ ^ ← 0 \widehat{\mathcal{I}}\leftarrow 0 
 
 
 
 4: 
 
 
 for t ← 1 ​  to  ​ T t\leftarrow 1\textbf{ to }T do 
 
 
 
 5: 
 
 
     if z i ∈ ℬ ( t ) {z_{i}\in\mathcal{B}^{(t)}} then 
 
 
 
 6: 
 
 
      θ ( t − 1 ) ← Θ ⁡ [ t ] {\theta^{(t-1)}\leftarrow\Theta[t]} 
 
 
 
 7: 
 
 
      ℐ ^ ← ℐ ^ + η ( t ) | ℬ ( t ) | ​ ∇ θ ℒ ​ ( z i , θ ( t − 1 ) ) ⊺ ​ ∇ θ ℒ ​ ( z te , θ ( t − 1 ) ) \widehat{\mathcal{I}}\leftarrow\widehat{\mathcal{I}}+\frac{\eta^{(t)}}{\lvert\mathcal{B}^{(t)}\rvert}{{\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(t-1)})}}^{\intercal}\,{\nabla_{\theta}{\mathcal{L}(z_{\text{te}};\theta^{(t-1)})}}}     
 
 
 8: 
 
 
 return ℐ ^ \widehat{\mathcal{I}} 
 
 
 
 
 Algorithm 3 TracInCP influence estimation 
 
 
 1: 
 
 
 Training param. set Θ \Theta ;
iteration subset 𝒯 \mathcal{T} ;
learning rates η ( 1 ) ; … ; η ( T ) {\eta^{(1)}\mathchar 59\relax\ldots\mathchar 59\relax\eta^{(T)}} ;
training instance z i z_{i} ;
and
test example z te z_{\text{te}} 
 
 
 
 2: 
 
 
 TracInCP influence est. ℐ ^ TracInCP ​ ( z i , z te ) {\widehat{\mathcal{I}}_{\text{TracIn{}CP}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)} 
 
 
 
 3: 
 
 
 ℐ ^ ← 0 \widehat{\mathcal{I}}\leftarrow 0 
 
 
 
 4: 
 
 
 for each t ∈ 𝒯 t\in\mathcal{T} do 
 
 
 
 5: 
 
 
     θ ( t − 1 ) ← Θ ⁡ [ t ] {\theta^{(t-1)}\leftarrow\Theta[t]} 
 
 
 
 6: 
 
 
     ℐ ^ ← ℐ ^ + η ( t ) ​ ∇ θ ℒ ​ ( z i , θ ( t − 1 ) ) ⊺ ​ ∇ θ ℒ ​ ( z te , θ ( t − 1 ) ) \widehat{\mathcal{I}}\leftarrow\widehat{\mathcal{I}}+\eta^{(t)}{{\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(t-1)})}}^{\intercal}\,{\nabla_{\theta}{\mathcal{L}(z_{\text{te}};\theta^{(t-1)})}}} 
 
 
 
 7: 
 
 
 return ℐ ^ \widehat{\mathcal{I}} 
 
 
 
 
 
 
 Remark 7 : 
 
 [ Pru+20a ] ’s empirical evaluation uses | 𝒯 | ≪ T {\lvert\mathcal{T}\rvert\ll T} .
For example, when identifying mislabeled examples using TracInCP, [ Pru+20a ] evaluate every 30th iteration.
Furthermore, [ Pru+20a ] note that prioritizing the small number of checkpoints where z te z_{\text{te}} ’s loss changes significantly generally outperforms evaluating a larger number of evenly-spaced checkpoints. 

 
 
 
 5.2.1.1 Time, Space, and Storage Complexity 

 
 Below we derive the time complexity of both versions of TracIn.
We then discuss their space and storage complexities. 

 
 
 Consider first TracInCP’s time complexity since it is simpler to derive.
From Alg. 3 , each checkpoint in 𝒯 {\mathcal{T}} requires n n , p p - dimensional dot products making TracInCP’s complexity 𝒪 ⁡ ( n ​ p ​ | 𝒯 | ) {\mathcal{O}(np\lvert\mathcal{T}\rvert)} .
For vanilla TracIn, consider Alg. 2 .
For each iteration t ∈ [ T ] {t\in{[T]}} , a p p - dimensional dot product is performed for each instance in ℬ ( t ) \mathcal{B}^{(t)} .
Let b ≔ max t ⁡ | ℬ ( t ) | {b\coloneqq\max_{t}\lvert\mathcal{B}^{(t)}\rvert} denote the maximum batch size, then TracIn’s time complexity is 𝒪 ⁡ ( b ​ p ​ T ) {\mathcal{O}(bpT)} .
In the worst case where ∀ t ℬ ( t ) = 𝒟 {\forall_{t}\,\mathcal{B}^{(t)}=\mathcal{D}} (full-batch gradient descent), TracIn time complexity is 𝒪 ⁡ ( n ​ p ​ T ) {\mathcal{O}(npT)} . 

 
 
 Recall that, by definition, | 𝒯 | ≤ T {\lvert\mathcal{T}\rvert\leq T} meaning TracInCP is asymptotically faster than TracIn.
However, this is misleading.
In practice, TracInCP is generally slower than TracIn as [ Pru+20a ] note. 

 
 
 Since each gradient calculation is independent, TracIn and TracInCP are fully parallelizable.
Table treats the level of concurrency as a constant factor, making the space complexity of both TracIn and TracInCP 𝒪 ⁡ ( n + p ) {\mathcal{O}(n+p)} . 

 
 
 Lastly, as detailed in Alg. 1 , dynamic influence estimators require that intermediate model parameters Θ \Theta be saved during training for post hoc influence estimation.
In the worst case, each training iteration’s parameters are stored resulting in a storage complexity of 𝒪 ⁡ ( p ​ T ) {\mathcal{O}(pT)} .
In practice however, TracIn only considers a small fraction of these T T training parameter vectors, meaning TracIn’s actual storage complexity is generally (much) lower than the worst case. 

 
 
 
 5.2.1.2 Strengths and Weaknesses 

 
 TracIn and TracInCP avoid many of the primary pitfalls associated with static, gradient-based estimators. 

 
 
 First, recall from Section 5.1.1 that Hessian-vector product s test s_{\text{test}} significantly increases the computational overhead and potential inaccuracy of influence functions.
TracIn’s theoretical simplicity avoids the need to compute any Hessian. 

 
 
 Second, representer point’s theoretical formulation necessitated considering only a model’s final linear layer,
at the risk of (significantly) worse performance.
TracIn has the flexibility to use only the final linear layer for scenarios where that provides sufficient accuracy 23 23 
 23 
 
 
 
 Last layer only TracIn is also referred to as TracIn - Last.
 [ Yeh+22a ] evaluate TracIn - Last’s effectiveness. 
as well as the option to use the full model gradient when needed. 

 
 
 Third, by measuring influence during the training process, TracIn requires no assumptions about stationarity or convergence.
In fact, TracIn can be applied to a model that is only partially trained.
TracIn can also be used to study when during training an instance is most influential.
For example, TracIn can identify whether a training instance is most influential early or late in training. 

 
 
 Fourth, due to how gradient-based methods estimate influence, highly influential instances can actually appear uninfluential at the end of training.
Unlike static estimators, dynamic methods like TracIn may still be able to detect these instances.
See Section 5.3 for more details. 

 
 
 In terms of weaknesses,
TracIn’s theoretical motivation assumes stochastic gradient descent without momentum.
However, momentum and adaptive optimization (e.g., Adam [ KB15a ] ) significantly accelerate model convergence [ Qia99a , DHS11a , KB15a ] .
To align more closely with these sophisticated optimizers, Eq. ( 55 ) and Alg. 2 would need to change significantly.
For context, Section 5.2.2 details another dynamic estimator, HyDRA , which incorporates support for just momentum with the resulting increase in estimator complexity substantial. 

 
 
 
 5.2.1.3 Related Methods 

 
 TracIn has been adapted by numerous derivative/heuristic variants.
For example, TracIn - Last is identical to vanilla TracIn except gradient vectors ∇ θ ℒ ​ ( z i , θ ( t − 1 ) ) {\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(t-1)})}} and ∇ θ ℒ ​ ( z te , θ ( t − 1 ) ) {\nabla_{\theta}{\mathcal{L}(z_{\text{te}};\theta^{(t-1)})}} only consider the model’s final linear layer [ Pru+20a ] .
This can make TracIn significantly faster at the risk of (significantly) worse accuracy [ Yeh+22a ] . 

 
 
 TracIn for Language Models   
As a counter to the disadvantages of solely considering a model’s last layer, 24 24 
 24 
 
 
 
 See Section 5.1.2.2 for an extended discussion of last-layer similarity. 
TracIn’s authors subsequently proposed
 TracIn word embeddings (TracInWE), which targets large language models and considers only the gradients in those models’ word embedding layer [ Yeh+22a ] .
Since language-model word embeddings can still be very large (e.g., BERT - Base’s word embedding layer has 23M parameters [ Dev+19a ] ), the authors specifically use the gradients of only those tokens that appear in both training instance z i z_{i} and test instance z te z_{\text{te}} . 

 
 
 Low Dimensional TracIn   
 [ Pru+20a ] also propose TracIn Random Projection (TracInRP) – a low-memory version of TracIn that provides unbiased estimates of ℐ ^ TracIn \widehat{\mathcal{I}}_{\text{TracIn{}}} (i.e., an estimate of an estimate).
Intuitively, TracInRP maps gradient vectors into a d d - dimensional subspace ( d ≪ p {d\ll p} ) via multiplication by a d × p {d\times p} random matrix where each entry is sampled i.i.d. from Gaussian distribution 𝒩 ⁡ ( 0 ​ ; ​ 1 d ) {\mathcal{N}\big(0\mathord{\mathchar 59\relax}\frac{1}{d}\big)} .
These low-memory gradient “sketches” are used in place of the full gradient vectors in Eq. ( 55 ) [ Woo14a ] .
TracInRP is primarily targeted at applications where p p is sufficiently large that storing the full training set’s gradient vectors ( ∀ t ​ ; ​ i ∇ θ ℒ ​ ( z i , θ ( t − 1 ) ) {\forall_{t\mathord{\mathchar 59\relax}i}\penalty\ {\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(t-1)})}}} ) is prohibitive. 

 
 
 TracIn for Generative Models   
TracIn has also been used outside of supervised settings.
For example, [ KC21a ] apply TracIn to unsupervised learning, in particular density estimation;
they
propose variational autoencoder TracIn (VAE - TracIn), which quantifies the TracIn influence in β \beta - VAEs [ Hig+17a ] .
Moreover,
 [ Thi+22a ] ’s [ Thi+22a ] TracIn anomaly detector (TracInAD) functionally estimates the distribution of influence estimates – using either TracInCP or VAE - TracIn.
TracInAD then marks as anomalous any test instance in the tail of this “influence distribution”. 

 
 
 Note also that TracIn can be applied to any iterative, gradient-based model, including those that are non-parametric.
For example, [ BHL23a ] ’s [ BHL23a ] BoostIn adapts TracIn for gradient-boosted decision tree ensembles. 

 
 
 
 

#### 5.2.2 HyDRA – Hypergradient Data Relevance Analysis

 
 Unlike TracIn which uses a novel definition of influence ( 51 ),
 [ Che+21a ] ’s [ Che+21a ] hypergradient data relevance analysis ( HyDRA ) estimates the leave - one - out influence ( 8 ).
 HyDRA leverages the same Taylor series-based analysis as [ KL17a ] ’s [ KL17a ] influence functions.
The key difference is that HyDRA addresses a fundamental mismatch between influence functions’ assumptions and deep models. 

 
 
 Section 5.1.1 explains that influence functions consider infinitesimally perturbing the weight of training sample z i z_{i} by ϵ i {\epsilon_{i}} .
Recall that the change in z te z_{\text{te}} ’s test risk w.r.t. to this infinitesimal perturbation is 

 

 
 | 
 d ​ ℒ ​ ( z te , θ ( T ) ) d ​ ϵ i = ∂ ℒ ⁡ ( z te , θ ( T ) ) ∂ θ ( T ) ⊺ ​ d ​ θ ( T ) d ​ ϵ i = ∂ ℒ ⁡ ( z te , θ ( T ) ) ∂ θ ( T ) ⊺ ​ h ~ i ( T ) \frac{d{\mathcal{L}(z_{\text{te}};\theta^{(T)})}}{d\epsilon_{i}}=\frac{\partial{\mathcal{L}(z_{\text{te}};\theta^{(T)})}}{\partial\theta^{(T)}}^{\intercal}\,\frac{d\theta^{(T)}}{d\epsilon_{i}}=\frac{\partial{\mathcal{L}(z_{\text{te}};\theta^{(T)})}}{\partial\theta^{(T)}}^{\intercal}\,\widetilde{h}^{(T)}_{i} | 
 | 
 (57) | 
 

 where
 h ~ i ( T ) ≔ d ​ θ + ϵ i ( T ) d ​ ϵ i {\widetilde{h}^{(T)}_{i}\coloneqq\frac{d\theta^{(T)}_{+\epsilon_{i}}}{d\epsilon_{i}}} 
denotes the p p - dimensional hypergradient of training instance i i at the end of training. 

 
 
 [ KL17a ] ’s [ KL17a ] assumptions of differentiability and strict convexity mean that Eq. ( 57 ) has a closed form.
However, deep neural models are not convex.
Under non-convex gradient descent without momentum and with L 2 L_{2} regularization, θ ( t ) ≔ θ ( t − 1 ) − η ( t ) ​ g ( t − 1 ) {\theta^{(t)}\coloneqq\theta^{(t-1)}-\eta^{(t)}g^{(t-1)}} where gradient 

 

 
 | 
 g ( t − 1 ) ≔ ∇ θ ℒ ​ ( ℬ ( t ) , θ ( t − 1 ) ) + λ ​ θ ( t − 1 ) ​ . g^{(t-1)}\coloneqq\nabla_{\theta}{\mathcal{L}(\mathcal{B}^{(t)};\theta^{(t-1)})}+\lambda\theta^{(t-1)}\text{.} | 
 | 
 (58) | 
 

 The exact definition of gradient g ( t − 1 ) g^{(t-1)} depends on the specific contents of batch ℬ ( t ) \mathcal{B}^{(t)} so for simplicity, we encapsulate the batch’s contribution to the gradient using catch-all term ∇ θ ℒ ​ ( ℬ ( t ) , θ ( t − 1 ) ) {\nabla_{\theta}{\mathcal{L}(\mathcal{B}^{(t)};\theta^{(t-1)})}} . 

 
 
 Using Eq. ( 58 ),
hypergradient h ~ i ( T ) \widetilde{h}^{(T)}_{i} can be defined recursively as 

 

 
 | 
 h ~ i ( T ) ≔ d ​ θ + ϵ i ( T ) d ​ ϵ i \displaystyle\widetilde{h}^{(T)}_{i}\coloneqq\frac{d\theta^{(T)}_{+\epsilon_{i}}}{d\epsilon_{i}} | 
 = d d ​ ϵ i ​ ( θ + ϵ i ( T − 1 ) − η ( t ) ​ g ( t − 1 ) ) \displaystyle=\frac{d}{d\epsilon_{i}}\left(\theta^{(T-1)}_{+\epsilon_{i}}-\eta^{(t)}g^{(t-1)}\right) | 
 | 
 (59) | 
 
 
 | 
 | 
 = h ~ i ( T − 1 ) − η ( T ) ​ d d ​ ϵ i ​ ( ∇ θ ℒ ​ ( ℬ ( T ) , θ + ϵ i ( T − 1 ) ) + λ ​ θ + ϵ i ( T − 1 ) ) \displaystyle=\widetilde{h}^{(T-1)}_{i}-\eta^{(T)}\frac{d}{d\epsilon_{i}}\left(\nabla_{\theta}{\mathcal{L}(\mathcal{B}^{(T)};\theta^{(T-1)}_{+\epsilon_{i}})}+\lambda\theta^{(T-1)}_{+\epsilon_{i}}\right) | 
 | 
 (60) | 
 
 
 | 
 | 
 = ( 1 − η ( T ) ​ λ ) ​ h ~ i ( T − 1 ) − η ( t ) ​ d d ​ ϵ i ​ ∇ θ ℒ ​ ( ℬ ( T ) , θ + ϵ i ( T − 1 ) ) \displaystyle=(1-\eta^{(T)}\lambda)\widetilde{h}^{(T-1)}_{i}-\eta^{(t)}\,\frac{d}{d\epsilon_{i}}\,\nabla_{\theta}{\mathcal{L}(\mathcal{B}^{(T)};\theta^{(T-1)}_{+\epsilon_{i}})} | 
 | 
 (61) | 
 

 The recursive definition of hypergradient h ~ i ( T ) \widetilde{h}^{(T)}_{i} needs to be unrolled all the way back to initial parameters θ ( 0 ) \theta^{(0)} . 

 
 
 The key takeaway from Eq. ( 61 ) is that training hypergradients affect the model parameters throughout all of training .
By assuming a convex model and loss, [ KL17a ] ’s [ KL17a ] simplified formulation ignores this very real effect.
As [ Che+21a ] observe, hypergradients often cause non-convex models to converge to a vastly different risk minimizer.
By considering the hypergradients’ cumulative effect, HyDRA can provide more accurate LOO estimates than influence functions on non-convex models – albeit via a significantly more complicated and computationally expensive formulation. 

 
 
 Unrolling Gradient Descent Hypergradients 
The exact procedure to unroll HyDRA ’s hypergradient h ~ i ( T ) \widetilde{h}^{(T)}_{i} is non-trivial.
For the interested reader, supplemental Section C provides hypergradient unrolling’s full derivation for vanilla gradient descent without momentum.
Below, we briefly summarize Section C ’s important takeaways, and Section C ’s full derivation can be skipped with minimal loss of understanding. 

 
 
 At each training iteration, hypergradient unrolling requires estimating the risk Hessian of each training instance in ℬ ( t ) \mathcal{B}^{(t)} . 25 25 
 25 
 
 
 
 For clarity, this is not the inverse Hessian ( H θ ( T ) ) − 1 {(H_{\theta}^{(T)})^{-1}} used by influence functions. 
This significantly slows down HyDRA (by a factor of about 1,000 × \times [ Che+21a ] ).
As a workaround, [ Che+21a ] propose treating these risk Hessians as all zeros, proving that, under mild assumptions, the approximation error of this simplified version of HyDRA is bounded.
Alg. 4 shows HyDRA ’s fast approximation algorithm without Hessians for vanilla gradient descent. 26 26 
 26 
 
 
 
 See HyDRA ’s original paper [ Che+21a ] for the fast approximation algorithm with momentum. 
After calculating the final hypergradient, substituting
 h ~ i ( T ) {\widetilde{h}^{(T)}_{i}} 
into Eq. ( 57 ) with ϵ i = − 1 n {\epsilon_{i}=-\frac{1}{n}} 27 27 
 27 
 
 
 
 ϵ i = − 1 n {\epsilon_{i}=-\frac{1}{n}} is equivalent to deleting instance z i z_{i} from the training set.
Influence functions follow the same procedure for ϵ i \epsilon_{i} .
See Eqs. ( 32 ) and ( 33 ). 
yields training instance z i z_{i} ’s HyDRA pointwise influence estimator 

 

 
 | 
 ℐ ^ HyDRA ​ ( z i , z te ) ≔ − 1 n ​ ∇ θ ℒ ​ ( z te , θ ( T ) ) ⊺ ​ h ~ i ( T ) ​ . {\widehat{\mathcal{I}}_{\textsc{HyDRA}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\coloneqq-\frac{1}{n}\nabla_{\theta}{\mathcal{L}(z_{\text{te}};\theta^{(T)})}^{\intercal}\,\widetilde{h}^{(T)}_{i}\text{.} | 
 | 
 (62) | 
 

 
 
 Algorithm 4 Fast HyDRA influence estimation for gradient descent without momentum 
 
 
 1: 
 
 
 Training parameter set Θ \Theta ;
final parameters θ ( T ) \theta^{(T)} ;
training set size n n ;
iteration count T T ;
batches ℬ ( 1 ) ; … ; ℬ ( T ) {\mathcal{B}^{(1)}\mathchar 59\relax\ldots\mathchar 59\relax\mathcal{B}^{(T)}} ;
learning rates η ( 1 ) ; … ; η ( T ) {\eta^{(1)}\mathchar 59\relax\ldots\mathchar 59\relax\eta^{(T)}} ;
weight decay λ \lambda ;
training instance z i z_{i} ;
and
test example z te z_{\text{te}} 
 
 
 
 2: 
 
 
 HyDRA influence estimate ℐ ^ HyDRA ​ ( z i , z te ) {\widehat{\mathcal{I}}_{\textsc{HyDRA}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)} 
 
 
 
 3: 
 
 
 h ~ i ( 0 ) ← 0 → \widetilde{h}^{(0)}_{i}\leftarrow\vec{0} 
 ⊳ \triangleright Initialize to zero vector
 
 
 
 4: 
 
 
 for t ← 1 ​  to  ​ T t\leftarrow 1\textbf{ to }T do 
 
 
 
 5: 
 
 
     if z i ∈ ℬ ( t ) {z_{i}\in\mathcal{B}^{(t)}} then 
 
 
 
 6: 
 
 
      θ ( t − 1 ) ← Θ ⁡ [ t ] {\theta^{(t-1)}\leftarrow\Theta[t]} 
 
 
 
 7: 
 
 
      h ~ i ( t ) ← ( 1 − η ( t ) ​ λ ) ​ h ~ i ( t − 1 ) − η ( t ) ​ n | ℬ ( t ) | ​ ∇ θ ℒ ​ ( z i , θ ( t − 1 ) ) \widetilde{h}^{(t)}_{i}\leftarrow(1-\eta^{(t)}\lambda)\widetilde{h}^{(t-1)}_{i}-\frac{\eta^{(t)}n}{\lvert\mathcal{B}^{(t)}\rvert}\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(t-1)})} 
 
 
 
 8: 
 
 
     else 
 
 
 9: 
 
 
      h ~ i ( t ) ← ( 1 − η ( t ) ​ λ ) ​ h ~ i ( t − 1 ) \widetilde{h}^{(t)}_{i}\leftarrow(1-\eta^{(t)}\lambda)\widetilde{h}^{(t-1)}_{i} 
    
 
 
 10: 
 
 
 ℐ ^ ← − 1 n ​ ∇ θ ℒ ​ ( z te , θ ( T ) ) ⊺ ​ h ~ i ( T ) \widehat{\mathcal{I}}\leftarrow-\frac{1}{n}\nabla_{\theta}{\mathcal{L}(z_{\text{te}};\theta^{(T)})}^{\intercal}\,\widetilde{h}^{(T)}_{i} 
 ⊳ \triangleright Influence estimate
 
 
 
 11: 
 
 
 return ℐ ^ \widehat{\mathcal{I}} 
 
 
 
 
 
 Relating HyDRA and TracIn 
When λ = 0 {\lambda=0} or weight decay’s effects are ignored (as done by TracIn), HyDRA ’s fast approximation for vanilla gradient descent simplifies to 

 

 
 | 
 ℐ ^ HyDRA ​ ( z i , z te ) \displaystyle{\widehat{\mathcal{I}}_{\textsc{HyDRA}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)} | 
 ≈ − 1 n ∇ θ ℒ ( z te ; θ ( T ) ) ⊺ ∑ t z i ∈ ℬ ( t ) − η ( t ) ​ n | ℬ ( t ) | ∇ θ ℒ ( z i ; θ ( t − 1 ) ) \displaystyle\approx-\frac{1}{n}\nabla_{\theta}{\mathcal{L}(z_{\text{te}};\theta^{(T)})}^{\intercal}\sum_{\begin{subarray}{c}t\\
z_{i}\in\mathcal{B}^{(t)}\end{subarray}}-\frac{\eta^{(t)}n}{\lvert\mathcal{B}^{(t)}\rvert}\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(t-1)})} | 
 | 
 (63) | 
 
 
 | 
 | 
 = ∑ t z i ∈ ℬ ( t ) η ( t ) | ℬ ( t ) | ​ ∇ θ ℒ ​ ( z te , θ ( T ) ) ⊺ ​ ∇ θ ℒ ​ ( z i , θ ( t − 1 ) ) \displaystyle=\sum_{\begin{subarray}{c}t\\
z_{i}\in\mathcal{B}^{(t)}\end{subarray}}\frac{\eta^{(t)}}{\lvert\mathcal{B}^{(t)}\rvert}\nabla_{\theta}{\mathcal{L}(z_{\text{te}};\theta^{(T)})}^{\intercal}\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(t-1)})} | 
 | 
 (64) | 
 
 
 | 
 | 
 = ∇ θ ℒ ​ ( z te , θ ( T ) ) ⊺ ​ ∑ t z i ∈ ℬ ( t ) η ( t ) | ℬ ( t ) | ​ ∇ θ ℒ ​ ( z i , θ ( t − 1 ) ) ​ . \displaystyle=\nabla_{\theta}{\mathcal{L}(z_{\text{te}};\theta^{(T)})}^{\intercal}\sum_{\begin{subarray}{c}t\\
z_{i}\in\mathcal{B}^{(t)}\end{subarray}}\frac{\eta^{(t)}}{\lvert\mathcal{B}^{(t)}\rvert}\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(t-1)})}\text{.} | 
 | 
 (65) | 
 

 Eq. ( 64 ) is very similar to TracIn’s definition in Eq. ( 55 ), despite the two methods estimating different definitions of influence (LOO vs. ideal TracIn ( 51 )).
The only difference between ( 64 ) and ( 55 ) is that HyDRA always uses final test gradient ∇ θ ℒ ​ ( z te , θ ( T ) ) {\nabla_{\theta}{\mathcal{L}(z_{\text{te}};\theta^{(T)})}} while TracIn uses each iteration’s test gradient ∇ θ ℒ ​ ( z te , θ ( t − 1 ) ) {\nabla_{\theta}{\mathcal{L}(z_{\text{te}};\theta^{(t-1)})}} 

 
 
 The key takeaway is that while theoretically different, HyDRA and TracIn are in practice very similar where HyDRA can be viewed as trading (incremental) speed for lower precision w.r.t. z te z_{\text{te}} . 

 
 
 5.2.2.1 Time, Space, and Storage Complexity 

 
 When unrolling the T T training iterations,
 HyDRA ’s fast approximation performs a p p - dimensional (hyper)gradient calculation for each of the n n training instances.
If HyDRA ’s full version with Hessian vector products is used, [ ABH17a ] ’s [ ABH17a ] Hessian approximation algorithm estimates each HVP in 𝒪 ⁡ ( p ) {\mathcal{O}(p)} time and space.
Therefore, the fast and standard versions of HyDRA both have full time complexity 𝒪 ⁡ ( n ​ p ​ T ) {\mathcal{O}(npT)} – same as TracIn, albeit with potentially much worse constant factors. 

 
 
 Observe that each hypergradient h ~ i ( T ) \widetilde{h}^{(T)}_{i} only needs to be computed once and can be reused for each test instance.
Therefore, the fast and standard version of HyDRA have incremental time complexity of just n n gradient dot products – 𝒪 ⁡ ( n ​ p ) {\mathcal{O}(np)} complexity total.
This incremental complexity is much faster than TracIn and asymptotically equivalent to influence functions.
In practice though, HyDRA ’s incremental cost is much lower than that of influence functions. 

 
 
 Alg. 4 requires storing vector h ~ i ( t ) \widetilde{h}^{(t)}_{i} throughout HyDRA ’s entire unrolling procedure, where each training instance’s hypergradient takes 𝒪 ⁡ ( p ) {\mathcal{O}(p)} space.
To analyze all training instances simultaneously, HyDRA requires 𝒪 ⁡ ( n ​ p ) {\mathcal{O}(np)} total space.
In contrast, TracIn only requires 𝒪 ⁡ ( n + p ) {\mathcal{O}(n+p)} space to analyze all instances simultaneously.
This difference is substantial for large models and training sets.
In cases where the fully-parallelized space complexity is prohibitive, each training instance’s hypergradient can be analyzed separately resulting in a reduced space complexity of 𝒪 ⁡ ( p ) {\mathcal{O}(p)} for both fast and standard HyDRA . 

 
 
 Like TracIn, HyDRA requires storing model parameters Θ ⊆ { θ ( 0 ) ; … ; θ ( T − 1 ) } {\Theta\subseteq\{\theta^{(0)}\mathchar 59\relax\ldots\mathchar 59\relax\theta^{(T-1)}\}} making its minimum storage complexity 𝒪 ⁡ ( p ​ T ) {\mathcal{O}(pT)} .
Since hypergradients are reused for each test instance, they can be stored to eliminate the need to recalculate them; this introduces an additional storage complexity of 𝒪 ⁡ ( n ​ p ) {\mathcal{O}(np)} .
This makes HyDRA ’s total storage complexity 𝒪 ⁡ ( p ​ T + n ​ p ) {\mathcal{O}(pT+np)} . 

 
 
 Remark 8 : 
 
 Storing both the training checkpoints and hypergradients is unnecessary.
Once all hypergradients have been calculated, serialized training parameters Θ \Theta are no longer needed and can be discarded.
Therefore, a more typical storage complexity is 𝒪 ⁡ ( p ​ T ) {\mathcal{O}(pT)} or 𝒪 ⁡ ( n ​ p ) {\mathcal{O}(np)} – both of which are still substantial. 

 
 
 
 
 5.2.2.2 Strengths and Weaknesses 

 
 HyDRA and TracIn share many of the same strengths.
For example, HyDRA does not require assumptions of convexity or stationarity.
Moreover, as a dynamic method, HyDRA may be able to detect influential examples that are missed by static methods – in particular when those instances have low loss at the end of training (see Section 5.3 for more discussion). 

 
 
 HyDRA also has some advantages over TracIn.
For example,
as shown in Alg. 2 , TracIn requires that each test instance be retraced through the entire training process.
This significantly increases TracIn’s incremental time complexity.
In contrast, HyDRA only unrolls gradient descent for the training instances, i.e., not the test instances.
Hypergradient unrolling is a one-time cost for each training instance; this upfront cost is amortized over all test instances.
Once the hypergradients have been calculated, HyDRA is much faster than TracIn – potentially by orders of magnitude.
In addition,
 HyDRA ’s overall design allows it to natively support momentum with few additional changes.
Integrating momentum into TracIn, while theoretically possible, requires substantial algorithmic changes and makes TracIn substantially more complicated.
This would mitigate a core strength of TracIn – its simplicity. 

 
 
 HyDRA does have two weaknesses in comparison to TracIn.
First, HyDRA ’s standard (i.e., non-fast) algorithm requires calculating many HVPs.
Second, HyDRA ’s 𝒪 ⁡ ( n ​ p ) {\mathcal{O}(np)} space complexity is much larger than the 𝒪 ⁡ ( n + p ) {\mathcal{O}(n+p)} space complexity of other influence analysis methods (see Table ).
For large models, this significantly worse space complexity may be prohibitive. 

 
 
 
 5.2.2.3 Related Methods 

 
 The method most closely related to HyDRA is [ HNM19a ] ’s [ HNM19a ] SGD - influence .
Both approaches estimate the leave - one - out influence by unrolling gradient descent using empirical risk Hessians.
There are, however, a few key differences.
First, unlike HyDRA , [ HNM19a ] assume that the model and loss function are convex.
Next, SGD - influence primarily applies unrolling to quantify the Cook’s distance, θ ( T ) − θ 𝒟 ∖ z i ( T ) {\theta^{(T)}-{\theta^{(T)}_{\mathcal{D}^{\setminus z_{i}}}}} .
To better align their approach with dataset influence, [ HNM19a ] propose a surrogate (linear) influence estimator which they incrementally update throughout unrolling.
This means the full training process must be unrolled for each test instance individually, significantly increasing SGD-influence’s incremental time complexity. 

 
 
 [ Ter+21a ] adapt the ideas of SGD-influence to estimate training data influence in generative adversarial networks (GANs). 

 
 
 Although proposed exclusively in the context of influence functions (Sec. 5.1.1.3 ), [ Sch+22a ] ’s [ Sch+22a ] basic approach to scale up influence functions via faster Hessian calculation could similarly be applied to speed up HyDRA ’s standard (non-fast) algorithm. 

 
 
 
 
 

### 5.3 Trade-off between Gradient Magnitude and Direction

 
 This section details a limitation common to existing gradient-based influence estimators that can cause these estimators to systematically overlook highly influential (groups of) training instances. 

 
 
 Observe that all gradient-based methods in this section rely on some vector dot product.
For a dot product to be large, one of two criteria must be met: 

 
 
 (1) The vector directions align (i.e., have high cosine similarity).
More specifically, for influence analysis, vectors pointing in similar directions are expected to encode similar information.
This is the ideal case. 

 
 
 (2) Either vector has a large magnitude, e.g., ∥ ∇ θ ℒ ​ ( z , θ ) ∥ \lVert\nabla_{\theta}{\mathcal{L}(z;\theta)}\rVert .
Large gradient magnitudes can occur for many reasons, but the most common cause is that the instance is either incorrectly or not confidently predicted. 

 
 
 Across the training set, gradient magnitudes can vary by several orders of magnitude [ SWS21a ] .
To overcome such a magnitude imbalance, training instances that actually influence a specific prediction may need to have orders of magnitude better vector alignment.
In reality, what commonly happens is that incorrectly predicted or abnormal training instances appear highly influential to all test instances [ SWS21a ] .
 [ BBD20a ] describe such training instances as globally influential .
However, globally influential training instances provide very limited insight into individual model predictions.
As [ BBD20a ] note, locally influential training instances are generally much more relevant and insightful when analyzing specific predictions. 

 
 
 Relative Influence   
To yield a more semantically meaningful influence ranking, [ BBD20a ] propose the θ \theta - relative influence functions estimator (RelatIF), which normalizes [ KL17a ] ’s [ KL17a ] influence functions’ estimator by HVP magnitude
 ∥ ( H θ ( T ) ) − 1 ​ ∇ θ ℒ ​ ( z i , θ ( T ) ) ∥ {\big\lVert{(H_{\theta}^{(T)})^{-1}}\,\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(T)})}\big\rVert} . 28 28 
 28 
 
 
 
 Note that this HVP is different than s test ≔ ( H θ ( T ) ) − 1 ​ ∇ θ ℒ ​ ( z te , θ ( T ) ) {s_{\text{test}}\coloneqq{(H_{\theta}^{(T)})^{-1}}\nabla_{\theta}{\mathcal{L}(z_{\text{te}};\theta^{(T)})}} in Eq. ( 34 ). 
Formally, 

 

 
 | 
 ℐ ^ RelatIF ​ ( z i , z te ) ≔ ℐ ^ IF ​ ( z i , z te ) ∥ ( H θ ( T ) ) − 1 ​ ∇ θ ℒ ​ ( z i , θ ( T ) ) ∥ = 1 n ​ ∇ θ ℒ ​ ( z te , θ ( T ) ) ⊺ ​ ( H θ ( T ) ) − 1 ​ ∇ θ ℒ ​ ( z i , θ ( T ) ) ∥ ( H θ ( T ) ) − 1 ​ ∇ θ ℒ ​ ( z i , θ ( T ) ) ∥ ​ . {\widehat{\mathcal{I}}_{\text{RelatIF}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\coloneqq\frac{{\widehat{\mathcal{I}}_{\textsc{IF}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}}{\lVert{(H_{\theta}^{(T)})^{-1}}\,\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(T)})}\rVert}=\frac{1}{n}\,\frac{\nabla_{\theta}{\mathcal{L}(z_{\text{te}};\theta^{(T)})}^{\intercal}{(H_{\theta}^{(T)})^{-1}}\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(T)})}}{\lVert{(H_{\theta}^{(T)})^{-1}}\,\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(T)})}\rVert}\text{.} | 
 | 
 (66) | 
 

 RelatIF’s normalization inhibits training gradient magnitude
 ∥ ∇ θ ℒ ​ ( z i , θ ( T ) ) ∥ {\big\lVert\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(T)})}\big\rVert} 
dominating the influence estimate. 

 
 
 RelatIF’s biggest limitation is the need to estimate an HVP for every training instance .
As discussed in Section 5.1.1.2 , HVP estimation is expensive and often highly inaccurate in deep models.
To work around these issues in their evaluation of RelatIF, [ BBD20a ] use either very small neural models or just consider a large model’s final layer, both of which can be problematic. 

 
 
 Renormalized Influence   
 [ HL22a ] make a similar observation as [ BBD20a ] but motivate it differently.
By the chain rule, gradient vectors decompose as 

 

 
 | 
 ∇ θ ℒ ​ ( z , θ ) ≔ ∂ ℓ ⁡ ( f ⁡ ( x , θ ) , y ) ∂ θ = ∂ ℓ ⁡ ( f ⁡ ( x , θ ) , y ) ∂ f ⁡ ( x , θ ) ​ ∂ f ⁡ ( x , θ ) ∂ θ ​ . \nabla_{\theta}{\mathcal{L}(z;\theta)}\coloneqq\frac{\partial{\ell({f(x;\theta)}\mathchar 59\relax y)}}{\partial\theta}=\frac{\partial{\ell({f(x;\theta)}\mathchar 59\relax y)}}{\partial{f(x;\theta)}}\,\frac{\partial{f(x;\theta)}}{\partial\theta}\text{.} | 
 | 
 (67) | 
 

 [ HL22a ] note that for many common loss functions (e.g., squared, binary cross-entropy), loss value ℓ ⁡ ( f ⁡ ( x , θ ) , y ) {\ell({f(x;\theta)}\mathchar 59\relax y)} 
induces a strict ordering over loss norm
 ∥ ∂ ℓ ⁡ ( f ⁡ ( x , θ ) , y ) ∂ f ⁡ ( x , θ ) ∥ \big\lVert\frac{\partial{\ell({f(x;\theta)}\mathchar 59\relax y)}}{\partial{f(x;\theta)}}\big\rVert .
 [ HL22a ] term this phenomenon a low-loss penalty , where confidently predicted training instances have smaller gradient magnitudes and by consequence consistently appear uninfluential to gradient-based influence estimators. 

 
 
 To account for the low-loss penalty, [ HL22a ] propose renormalized influence which replaces all gradient vectors – both training and test – in an influence estimator with the corresponding unit vector .
Renormalization can be applied to any gradient-based estimator.
For example, [ HL22a ] observe that renormalized TracInCP, which they term gradient aggregated similarity ( GAS ), 

 

 
 | 
 ℐ ^ GAS ​ ( z i , z te ) ≔ ∑ t ∈ 𝒯 η ( t ) ​ ∇ θ ℒ ​ ( z i , θ ( t − 1 ) ) ⊺ ​ ∇ θ ℒ ​ ( z te , θ ( t − 1 ) ) ∥ ∇ θ ℒ ​ ( z i , θ ( t − 1 ) ) ∥ ​ ∥ ∇ θ ℒ ​ ( z te , θ ( t − 1 ) ) ∥ ​ , {\widehat{\mathcal{I}}_{\textsc{GAS}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\coloneqq\sum_{t\in\mathcal{T}}\eta^{(t)}\,\frac{{{\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(t-1)})}}^{\intercal}\,{\nabla_{\theta}{\mathcal{L}(z_{\text{te}};\theta^{(t-1)})}}}}{\lVert{\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(t-1)})}}\rVert\,\lVert{\nabla_{\theta}{\mathcal{L}(z_{\text{te}};\theta^{(t-1)})}}\rVert}\text{,} | 
 | 
 (68) | 
 

 is particularly effective at generating influence rankings.
 [ HL22a ] also provide a renormalized version of influence functions, 

 

 
 | 
 ℐ ^ RenormIF ​ ( z i , z te ) ≔ ℐ ^ IF ​ ( z i , z te ) ∥ ∇ θ ℒ ​ ( z i , θ ( T ) ) ∥ = 1 n ​ ∇ θ ℒ ​ ( z te , θ ( T ) ) ⊺ ​ ( H θ ( T ) ) − 1 ​ ∇ θ ℒ ​ ( z i , θ ( T ) ) ∥ ∇ θ ℒ ​ ( z i , θ ( T ) ) ∥ ​ . {\widehat{\mathcal{I}}_{\text{RenormIF}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\coloneqq\frac{{\widehat{\mathcal{I}}_{\textsc{IF}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}}{\lVert\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(T)})}\rVert}=\frac{1}{n}\,\frac{\nabla_{\theta}{\mathcal{L}(z_{\text{te}};\theta^{(T)})}^{\intercal}{(H_{\theta}^{(T)})^{-1}}\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(T)})}}{\lVert\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(T)})}\rVert}\text{.} | 
 | 
 (69) | 
 

 Since renormalized influence functions do not require estimating additional HVPs, it is considerably faster than RelatIF.
Renormalized influence functions also do not have the additional error associated with estimating RelatIF’s additional HVPs. 29 29 
 29 
 
 
 
 [ HL22a ] also provide renormalized versions of representer point and TracIn, which are omitted here. 

 
 
 This section should not be interpreted to mean that gradient magnitude is unimportant for influence analysis.
On the contrary, gradient magnitude has a significant effect on training.
However, the approximations made by existing influence estimators often overemphasize gradient magnitude leading to influence rankings that are not semantically meaningful. 

 
 
 Remark 9 : 
 
 [ BBD20a ] ’s RelatIF and [ HL22a ] ’s renormalization do not change the corresponding influence estimators’ time and space complexities. 

 
 
 
 
 

## 6 Applications of Influence Analysis

 
 Section 3.3 discusses how a few topics (formally) relate to influence analysis.
This section provides an extended discussion of influence analysis’s applications.
Specifically, this section focuses
on higher-level learning tasks as opposed to the specific application environments where influence analysis has been used including: toxic speech detection [ HT21a ] , social network graph labeling [ Zha+21e ] , user engagement detection [ LLY21a ] , medical imaging annotation [ Bra+22a ] , etc. 

 
 
 First,
 data cleaning aims to improve a machine learning model’s overall performance by removing “bad” training data.
These “bad” instances arise due to disparate non-malicious causes including human/algorithmic labeling error, non-representative instances, noisy features, missing features, etc. [ Kri+16a , LDG18a , KW19a ] .
Intuitively, “bad” training instances are generally anomalous, and their features clash with the feature distribution of typical “clean” data [ Woj+16a ] .
In practice, overparameterized neural networks commonly memorize these “bad” instances to achieve zero training loss [ HNM19a , FZ20a , Pru+20a , Thi+22a ] .
As explained in Section 3.2 , memorization can be viewed as the influence of a training instance on itself.
Therefore, influence analysis can be used to detect these highly memorized training instances.
These memorized “bad” instances are then either removed from the training data or simply relabeled [ KSH22a ] and the model retrained. 

 
 
 Poisoning and backdoor attacks craft malicious training instances that manipulate a model to align with some attacker objective.
For example, a company may attempt to trick a spam filter so all emails sent by a competitor are erroneously classified as spam [ Sha+18b ] .
Obviously, only influential (malicious) training instances affect a model’s prediction.
Some training set attacks rely on influence analysis to craft better (i.e., more influential) poison instances [ FGL20a , Jag+21a , Oh+22a ] .
Since most training set attacks do not assume the adversary knows training’s random seed or even necessarily the target model’s architecture, poison instances are crafted to maximize their expected group influence [ Che+17a ] . 

 
 
 Influence and memorization analysis have also been used to improve membership inference attacks , where the adversary attempts to extract sensitive training data provided only a pretrained (language) model [ Dem+19a , CG22a ] . 

 
 
 Training set attack defenses detect and mitigate poisoning and backdoor attacks [ Li+22a ] .
Since malicious training instances must be influential to achieve the attacker’s objective, defending against adversarial attacks reduces to identifying abnormally influential training instances.
If attackers are constrained in the number of training instances they can insert [ Wal+21a , YHL23a ] , the target of a training set attack can be identified by searching for test instances that have a few exceptionally influential training instances [ HL22a ] .
The training set attack mitigation removes these anomalously influential instances from the training data and then retrains the model [ Wan+19a ] .
In addition, influence estimation has been applied to the related task of evasion attack detection , where the training set is pristine and only test instances are perturbed [ CSG20a ] . 

 
 
 Algorithmic fairness promotes techniques that enable machine learning models to make decisions free of prejudices and biases based on inherited characteristics such as race, religion, and gender [ Meh+21a ] .
A classic example of model unfairness is the COMPAS software tool, which estimated the recidivism risk of incarcerated individuals.
COMPAS was shown to be biased against black defendants, falsely flagging them as future criminals at twice the rate of white defendants [ Ang+16a ] .
Widespread adoption of algorithmic decision making in domains critical to human safety and well-being is predicated on the public’s perception and understanding of the algorithms’ inherent ethical principles and fairness [ Awa+18a ] .
Yet, how to quantify the extent to which an algorithm is “fair” remains an area of active study [ Dwo+12a , GH19a , Sax+19a ] .
 [ BF21a ] propose leave - one - out unfairness as a measure of a prediction’s fairness.
Intuitively, when a model’s decision (e.g., not granting a loan, hiring an employee) is fundamentally changed by the inclusion of a single instance in a large training set, such a decision may be viewed as unfair or even capricious.
Leave-one-out influence is therefore useful to measure and improve a model’s robustness and fairness. 

 
 
 Explainability attempts to make a black-box model’s decisions understandable by humans [ BH21a ] .
Transparent explanations are critical to achieving user trust of and satisfaction with ML systems [ LDA09a , Kiz16a , Zho+19a ] .
 Example-based explanations communicate why a model made a particular prediction via visual examples [ CJH19a , SWS21a ] – e.g., training images – as social science research has shown that humans can understand complex ideas using only examples [ RHS09a , Ren14a ] .
Influence estimation can assist in the selection of canonical training instances that are particularly important for a given class in general or a single test prediction specifically.
Similarly, normative explanations – which collectively establish a “standard” for a given class [ CJH19a ] – can be selected from those training instances with the highest average influence on a held-out validation set.
In cases where a test instance is misclassified, influence analysis can identify those training instances that most influenced the misprediction. 

 
 
 Subsampling reduces the computational requirements of large datasets by training models using only a subset of the training data [ TB18a ] .
Existing work has shown that high-quality training subsets can be created by greedily selecting training instances based on their overall influence [ Kha+19a , Wan+20a ] .
Under mild assumptions, [ Wan+20a ] even show that, in expectation, influence-based subsampling performs at least as well as training on the full training set. 

 
 
 Annotating unlabeled data can be expensive – in particular for domains like medical imaging where the annotators must be domain experts [ Bra+22a ] .
Compared to labeling instances u.a.r., active learning reduces labeling costs by prioritizing annotation of particularly salient unlabeled data.
In practice, active learning often simplifies to maximizing the add-one-in influence where each unlabeled instance’s marginal influence must be estimated.
Obviously, retraining for each possible unlabeled instance combination has exponential complexity and is intractable.
Instead, a greedy strategy can be used where the influence of each unlabeled instance is estimated to identify the next candidate to label [ Liu+21a , Jia+21b , Zha+21e ] . 

 
 
 To enhance the benefit of limited labeled data, influence analysis has been used to create better augmented training data [ Lee+20a , Oh+21a ] .
These influence-guided data augmentation methods outperform traditional random augmentations, albeit with a higher computational cost. 

 
 
 

## 7 Future Directions

 
 The trend of consistently increasing model complexity and opacity will likely continue for the foreseeable future.
Simultaneously, there are increased societal and regulatory demands for algorithmic transparency and explainability.
Influence analysis sits at the nexus of these competing trajectories [ Zho+19a ] , which points to the field growing in importance and relevance.
This section identifies important directions we believe influence analysis research should take going forward. 

 
 
 Emphasizing Group Influence over Pointwise Influence :
Most existing methods target pointwise influence, which apportions credit for a prediction to training instances individually.
However, for overparameterized models trained on large datasets, only the tails of the data distribution are heavily influenced by an individual instance [ Fel20c ] .
Instead, most predictions are moderately influenced by multiple training instances working in concert [ FZ20a , Das+21a , BYF20a ] . 

 
 
 As an additional complicating factor, pointwise influence within data-distribution modes is often approximately supermodular where the marginal effect of a training instance’s deletion increases as more instances from a group are removed [ HL22a ] .
This makes pointwise influence a particularly poor choice for understanding most model behavior.
To date, very limited work has systematically studied group influence [ Koh+19a , BYF20a , HL22a ] .
Better group influence estimators could be immediately applied in various domains such as poisoning attacks, coreset selection, and model explainability. 

 
 
 Certified Influence Estimation :
Certified defenses against poisoning and backdoor attacks guarantee that deleting a fixed number of instances from the training data will not change a model’s prediction [ SKL17a , LF21a , Jia+22a , WLF22a , HL23a ] .
These methods can be viewed as upper bounding the training data’s group influence – albeit very coarsely.
Most certified poisoning defenses achieve their bounds by leveraging “tricks” associated with particular model architectures (e.g., instance-based learners [ Jia+22a ] and ensembles [ LF21a , WLF22a , HL24b , Rez+23a ] ) as opposed to a detailed analysis of a prediction’s stability [ HL23a ] .
With limited exception [ Jia+19b ] , today’s influence estimators do not provide any meaningful guarantee of their accuracy.
Rather, most influence estimates should be viewed as only providing – at best – guidance on an instance’s “possible influence.”
Guaranteed or even probabilistic bounds on an instance’s influence would enable influence estimation to be applied in settings where more than a “heuristic approximation” is required [ HL23a ] . 

 
 
 Improved Scalability :
Influence estimation is slow.
Analyzing each training instance’s influence on a single test instance can take several hours or more [ BBD20a , Kob+20a , Guo+21a , HL22a ] .
For influence estimation to be a practical tool, it must be at least an order of magnitude faster.
Heuristic influence analysis speed-ups could prove very useful [ Guo+21a , Sch+22a ] .
However, the consequences (and limitations) of any empirical shortcuts need to be thoroughly tested, verified, and understood.
Similarly, limited existing work has specialized influence methods to particular model classes [ Jia+21b ] or data modalities [ Yeh+22a ] .
While application-agnostic influence estimators are useful, their flexibility limits their scalability and accuracy.
Both of these performance metrics may significantly improve via increased influence estimator specialization. 

 
 
 Surrogate Influence and Influence Transferability :
An underexplored opportunity to improve influence analysis lies in the use of surrogate models [ Sha+18c , Jia+19c , Jia+21b , BHL23a ] .
For example, linear surrogates have proven quite useful for model explainability [ LL17a ] .
While using only a model’s linear layer as a surrogate may be “too reductive” [ Yeh+22a ] , it remains an open question whether other compact models remain an option.
Any surrogate method must be accompanied by rigorous empirical evaluation to identify any risks and “blind spots” the surrogate may introduce [ Rud19a ] . 

 
 
 Increased Evaluation Diversity :
Influence analysis has the capability to provide salient insights into why models behave as they do [ FZ20a ] .
As an example, [ BF21a ] demonstrate how influence analysis can identify potential unfairness in an algorithmic decision.
However, influence estimation evaluation is too often superficial and focuses on a very small subset of possible applications.
For instance, most influence estimation evaluation focuses primarily on contrived data cleaning and mislabeled training data experiments [ Woj+16a , KL17a , Kha+19a , GZ19a , Yeh+18a , Pru+20a , Che+21a , Ter+21a , KS21a , SWS21a , BHL23a , Yeh+22a , KSH22a , KZ22a ] .
It is unclear how these experiments translate into real-world or adversarial settings, with recent work pointing to generalization fragility [ BPF21a , Bae+22a , Sch+23a ] .
We question whether these data cleaning experiments – where specialized methods already exist [ Kri+16a , KW19a , Wan+19a ] – adequately satisfy influence analysis’s stated promise of providing “understanding [of] black-box predictions” [ KL17a ] . 

 
 
 Objective Over Subjective Evaluation Criteria :
A common trope when evaluating an influence analysis method is to provide a test example and display training instances the estimator identified as most similar or dissimilar.
These “eye test” evaluations are generally applied to vision datasets [ KL17a , Yeh+18a , Jia+19b , Pru+20a , FZ20a ] and to a limited extent other modalities.
Such experiments are unscientific.
They provide limited meaningful insight given the lack of a ground truth by which to judge the results.
Most readers do not have detailed enough knowledge of a dataset to know whether the selected instances are especially representative. Rather, there may exist numerous training instances that are much more similar to the target that the influence estimator overlooked.
Moreover, such visual assessments are known to be susceptible to confirmation and expectancy biases [ Mah77a , NDM13a , KDK13a ] . 

 
 
 Influence analysis evaluation should focus on experiments that are quantifiable and verifiable w.r.t. a ground truth. 

 
 
 

## 8 Conclusions

 
 While influence analysis has received increased attention in recent years, significant progress remains to be made.
Influence estimation is computationally expensive and can be prone to inaccuracy.
Going forward, fast certified influence estimators are needed.
Nonetheless, despite these shortcomings, existing applications already demonstrate influence estimation’s capabilities and promise. 

 
 
 This work reviews numerous methods with different perspectives on – and even definitions of – training data influence.
It would be a mistake to view this diversity of approaches as a negative.
While no single influence analysis method can be applied to all situations, most use cases should have at least one method that fits well.
An obvious consequence then is the need for researchers and practitioners to understand the strengths and limitations of the various methods so as to know which method best fits their individual use case.
This survey is intended to provide that insight from both empirical and theoretical viewpoints. 

 
 
 

## Acknowledgments

 
 This work was supported by a grant from the Air Force Research Laboratory and the Defense Advanced Research Projects Agency (DARPA) — agreement number FA8750 - 16 - C - 0166, subcontract K001892 - 00 - S05, as well as a second grant from DARPA, agreement number HR00112090135. 

 
 
 

## References

 
 
 [ABH17] 
 Naman Agarwal, Brian Bullins and Elad Hazan 
 
 “Second-Order Stochastic Optimization for Machine Learning in Linear Time” 
 
 In Journal of Machine Learning Research 18.1 
 
 JMLR.org, 2017, pp. 4148–4187 
 
 URL: https://arxiv.org/abs/1602.03943 
 

 
 [AKA91] 
 David. Aha, Dennis Kibler and Marc. Albert 
 
 “Instance-Based Learning Algorithms” 
 
 In Machine Learning 6.1 
 
 USA: Kluwer Academic Publishers, 1991, pp. 37–66 
 
 URL: https://link.springer.com/article/10.1007/bf00153759 
 

 
 [Ang+16] 
 Julia Angwin, Jeff Larson, Surya Mattu and Lauren Kirchner 
 
 “Machine Bias” 
 
 In ProPublica , 2016 
 
 URL: https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing 
 

 
 [Arn51] 
 Walter Arnoldi 
 
 “The Principle of Minimized Iterations in the Solution of the Matrix Eigenvalue Problem” 
 
 In Quarterly of Applied Mathematics 9.1 
 
 Brown University, 1951, pp. 17–29 
 

 
 [Arp+17] 
 Devansh Arpit, Stanisław Jastrzebski, Nicolas Ballas, David Krueger, Emmanuel Bengio, Maxinder. Kanwal, Tegan Maharaj, Asja Fischer, Aaron Courville, Yoshua Bengio and Simon Lacoste-Julien 
 
 “A Closer Look at Memorization in Deep Networks” 
 
 In Proceedings of the 34th International Conference on Machine Learning , ICML’17, 2017 
 
 URL: https://arxiv.org/abs/1706.05394 
 

 
 [Awa+18] 
 Edmond Awad, Sohan Dsouza, Richard Kim, Jonathan Schulz, Joseph Henrich, Azim Shariff, Jean-François Bonnefon and Iyad Rahwan 
 
 “The Moral Machine Experiment” 
 
 In Nature 563.7729 , 2018, pp. 59–64 
 

 
 [BLK17] 
 Olivier Bachem, Mario Lucic and Andreas Krause 
 
 “Practical Coreset Constructions for Machine Learning”, 2017 
 
 arXiv: 1703.06476 [stat.ML] 
 

 
 [Bae+22] 
 Juhan Bae, Nathan Ng, Alston Lo, Marzyeh Ghassemi and Roger Grosse 
 
 “If Influence Functions are the Answer, Then What is the Question?” 
 
 In Proceedings of the 36th Conference on Neural Information Processing Systems , NeurIPS’22 
 
 Curran Associates, Inc., 2022 
 
 URL: https://arxiv.org/abs/2209.05364 
 

 
 [Ban65] 
 John. Banzhaf 
 
 “Weighted Voting Doesn’t Work: A Mathematical Analysis” 
 
 In Rutgers Law Review 19.2 , 1965, pp. 317–343 
 

 
 [BBD20] 
 Elnaz Barshan, Marc-Etienne Brunet and Gintare Dziugaite 
 
 “RelatIF: Identifying Explanatory Training Samples via Relative Influence” 
 
 In Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics , AISTATS’20, 2020 
 
 URL: https://arxiv.org/abs/2003.11630 
 

 
 [Bar+20] 
 Peter. Bartlett, Philip. Long, Gábor Lugosi and Alexander Tsigler 
 
 “Benign Overfitting in Linear Regression” 
 
 In Proceedings of the National Academy of Sciences 117.48 , 2020, pp. 30063–30070 
 
 URL: https://arxiv.org/abs/1906.11300 
 

 
 [BCC19] 
 Christine Basta, Marta. Costa-jussà and Noe Casas 
 
 “Evaluating the Underlying Gender Bias in Contextualized Word Embeddings” 
 
 In Proceedings of the First Workshop on Gender Bias in Natural Language Processing 
 
 Florence, Italy: Association for Computational Linguistics, 2019 
 
 URL: https://arxiv.org/abs/1904.08783 
 

 
 [BPF21] 
 Samyadeep Basu, Phil Pope and Soheil Feizi 
 
 “Influence Functions in Deep Learning Are Fragile” 
 
 In Proceedings of the 9th International Conference on Learning Representations , ICLR’21, 2021 
 
 URL: https://arxiv.org/abs/2006.14651 
 

 
 [BYF20] 
 Samyadeep Basu, Xuchen You and Soheil Feizi 
 
 “On Second-Order Group Influence Functions for Black-Box Predictions” 
 
 In Proceedings of the 37th International Conference on Machine Learning , ICML’20 
 
 Virtual Only: PMLR, 2020 
 
 URL: https://arxiv.org/abs/1911.00418 
 

 
 [BT74] 
 Albert. Beaton and John. Tukey 
 
 “The Fitting of Power Series, Meaning Polynomials, Illustrated on Band-Spectroscopic Data” 
 
 In Technometrics 16.2 
 
 Taylor Francis, 1974, pp. 147–185 
 

 
 [BP21] 
 Vaishak Belle and Ioannis Papantonis 
 
 “Principles and Practice of Explainable Machine Learning” 
 
 In Frontiers in Big Data 4 
 
 Frontiers Media S.A., 2021, pp. 688969–688969 
 
 URL: https://arxiv.org/abs/2009.11698 
 

 
 [Ben+21] 
 Emily. Bender, Timnit Gebru, Angelina McMillan-Major and Shmargaret Shmitchell 
 
 “On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?” 
 
 In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency , FAccT’21 
 
 New York, NY, USA: Association for Computing Machinery, 2021, pp. 610–623 
 
 URL: https://dl.acm.org/doi/10.1145/3442188.3445922 
 

 
 [BNL12] 
 Battista Biggio, Blaine Nelson and Pavel Laskov 
 
 “Poisoning Attacks against Support Vector Machines” 
 
 In Proceedings of the 29th International Conference on Machine Learning , ICML’12 
 
 Edinburgh, Great Britain: PMLR, 2012 
 
 URL: https://arxiv.org/abs/1206.6389 
 

 
 [Bil22] 
 Jeffrey Bilmes 
 
 “Submodularity in Machine Learning and Artificial Intelligence”, 2022 
 
 URL: https://arxiv.org/abs/2202.00132 
 

 
 [BF21] 
 Emily Black and Matt Fredrikson 
 
 “Leave-One-Out Unfairness” 
 
 In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency , FAccT’21, 2021 
 
 URL: https://arxiv.org/abs/2107.10171 
 

 
 [BR92] 
 Avrim. Blum and Ronald. Rivest 
 
 “Training a 3-Node Neural Network is NP-Complete” 
 
 In Neural Networks 5.1 , 1992, pp. 117–127 
 

 
 [BL23] 
 Sebastian Bordt and Ulrike von Luxburg 
 
 “From Shapley Values to Generalized Additive Models and back” 
 
 In Proceedings of The 26th International Conference on Artificial Intelligence and Statistics , AISTATS’23, 2023 
 
 URL: https://arxiv.org/abs/2209.04012 
 

 
 [BMK20] 
 Zalán Borsos, Mojmir Mutny and Andreas Krause 
 
 “Coresets via Bilevel Optimization for Continual Learning and Streaming” 
 
 In Proceedings of the 34th Conference on Neural Information Processing Systems , NeurIPS’20, 2020 
 
 URL: https://arxiv.org/abs/2006.03875 
 

 
 [BV04] 
 Stephen Boyd and Lieven Vandenberghe 
 
 “Convex Optimization” 
 
 Cambridge University Press, 2004 
 

 
 [Bra+22] 
 Joschka Braun, Micha Kornreich, JinHyeong Park, Jayashri Pawar, James Browning, Richard Herzog, Benjamin Odry and Li Zhang 
 
 “Influence Based Re-Weighing for Labeling Noise in Medical Imaging” 
 
 In Proceedings of the 19th IEEE International Symposium on Biomedical Imaging , ISBI’22, 2022 
 

 
 [BHL23] 
 Jonathan Brophy, Zayd Hammoudeh and Daniel Lowd 
 
 “Adapting and Evaluating Influence-Estimation Methods for Gradient-Boosted Decision Trees” 
 
 In Journal of Machine Learning Research 24 , 2023, pp. 1–48 
 
 URL: https://arxiv.org/abs/2205.00359 
 

 
 [BL21] 
 Jonathan Brophy and Daniel Lowd 
 
 “Machine Unlearning for Random Forests” 
 
 In Proceedings of the 38th International Conference on Machine Learning , ICML’21, 2021 
 
 URL: https://arxiv.org/abs/2009.05567 
 

 
 [Bro+20] 
 Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever and Dario Amodei 
 
 “Language Models are Few-Shot Learners” 
 
 In Proceedings of the 34th Conference on Neural Information Processing Systems , NeurIPS’20 
 
 Curran Associates, Inc., 2020 
 
 URL: https://arxiv.org/abs/2005.14165 
 

 
 [BH21] 
 Nadia Burkart and Marco. Huber 
 
 “A Survey on the Explainability of Supervised Machine Learning” 
 
 In Journal Artificial Intelligence Research 70 
 
 El Segundo, CA, USA: AI Access Foundation, 2021, pp. 245–317 
 
 URL: https://arxiv.org/abs/2011.07876 
 

 
 [CJH19] 
 Carrie. Cai, Jonas Jongejan and Jess Holbrook 
 
 “The Effects of Example-Based Explanations in a Machine Learning Interface” 
 
 In Proceedings of the 24th International Conference on Intelligent User Interfaces , IUI’19, 2019, pp. 258–262 
 
 URL: https://dl.acm.org/doi/10.1145/3301275.3302289 
 

 
 [Che+22] 
 Ruoxin Chen, Zenan Li, Jie Li, Chentao Wu and Junchi Yan 
 
 “On Collective Robustness of Bagging Against Data Poisoning” 
 
 In Proceedings of the 39th International Conference on Machine Learning , ICML’22 
 
 PMLR, 2022 
 
 URL: https://arxiv.org/abs/2205.13176 
 

 
 [Che+17] 
 Xinyun Chen, Chang Liu, Bo Li, Kimberly Lu and Dawn Song 
 
 “Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning”, 2017 
 
 arXiv: 1712.05526 [cs.CR] 
 

 
 [Che+21] 
 Yuanyuan Chen, Boyang Li, Han Yu, Pengcheng Wu and Chunyan Miao 
 
 “HyDRA: Hypergradient Data Relevance Analysis for Interpreting Deep Neural Networks” 
 
 In Proceedings of the 35th AAAI Conference on Artificial Intelligence , AAAI’21 
 
 Virtual Only: Association for the Advancement of Artificial Intelligence, 2021 
 
 URL: https://arxiv.org/abs/2102.02515 
 

 
 [CG22] 
 Gilad Cohen and Raja Giryes 
 
 “Membership Inference Attack Using Self Influence Functions”, 2022 
 
 arXiv: 2205.13680 [cs.LG] 
 

 
 [CSG20] 
 Gilad Cohen, Guillermo Sapiro and Raja Giryes 
 
 “Detecting Adversarial Samples Using Influence Functions and Nearest Neighbors” 
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , CVPR’20, 2020 
 
 URL: https://arxiv.org/abs/1909.06872 
 

 
 [Coo77] 
 R. Cook 
 
 “Detection of Influential Observation in Linear Regression” 
 
 In Technometrics 19.1 
 
 American Statistical Association, 1977, pp. 15–18 
 

 
 [CHW82] 
 R. Cook, Norton Holschuh and Sanford Weisberg 
 
 “A Note on an Alternative Outlier Model” 
 
 In Journal of the Royal Statistical Society. Series B (Methodological) 44.3 
 
 Wiley, 1982, pp. 370–376 
 

 
 [CW82] 
 R. Cook and Sanford Weisberg 
 
 “Residuals and Influence in Regression” 
 
 New York: ChapmanHall, 1982 
 

 
 [DAm+20] 
 Alexander D’Amour, Katherine. Heller, Dan Moldovan, Ben Adlam, Babak Alipanahi, Alex Beutel, Christina Chen, Jonathan Deaton, Jacob Eisenstein, Matthew. Hoffman, Farhad Hormozdiari, Neil Houlsby, Shaobo Hou, Ghassen Jerfel, Alan Karthikesalingam, Mario Lucic, Yi-An Ma, Cory. McLean, Diana Mincu, Akinori Mitani, Andrea Montanari, Zachary Nado, Vivek Natarajan, Christopher Nielson, Thomas. Osborne, Rajiv Raman, Kim Ramasamy, Rory Sayres, Jessica Schrouff, Martin Seneviratne, Shannon Sequeira, Harini Suresh, Victor Veitch, Max Vladymyrov, Xuezhi Wang, Kellie Webster, Steve Yadlowsky, Taedong Yun, Xiaohua Zhai and D. Sculley 
 
 “Underspecification Presents Challenges for Credibility in Modern Machine Learning”, 2020 
 
 arXiv: 2011.03395 [cs.LG] 
 

 
 [DG23] 
 Zheng Dai and David. Gifford 
 
 “Training Data Attribution for Diffusion Models”, 2023 
 
 arXiv: 2306.02174 [stat.ML] 
 

 
 [Das+21] 
 Soumi Das, Arshdeep Singh, Saptarshi Chatterjee, Suparna Bhattacharya and Sourangshu Bhattacharya 
 
 “Finding High-Value Training Data Subset through Differentiable Convex Programming” 
 
 In Proceedings of the 2021 European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases , ECML PKDD’21, 2021 
 
 URL: https://arxiv.org/abs/2104.13794 
 

 
 [DG14] 
 Alex Davies and Zoubin Ghahramani 
 
 “The Random Forest Kernel and Other Kernels for Big Data from Random Partitions”, 2014 
 
 arXiv: 1402.4293 [cs.LG] 
 

 
 [Dem+19] 
 Ambra Demontis, Marco Melis, Maura Pintor, Matthew Jagielski, Battista Biggio, Alina Oprea, Cristina Nita-Rotaru and Fabio Roli 
 
 “Why Do Adversarial Attacks Transfer? Explaining Transferability of Evasion and Poisoning Attacks” 
 
 In Proceedings of the 28th USENIX Security Symposium , USENIX’19 
 
 Renton, Washington, USA, 2019 
 
 URL: https://arxiv.org/abs/1809.02861 
 

 
 [Den+09] 
 Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li and Li Fei-Fei 
 
 “ImageNet: A Large-Scale Hierarchical Image Database” 
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , CVPR’09, 2009, pp. 248–255 
 

 
 [DP94] 
 Xiaotie Deng and Christos. Papadimitriou 
 
 “On the Complexity of Cooperative Solution Concepts” 
 
 In Mathematics of Operations Research 19.2 
 
 Linthicum, MD, USA: INFORMS, 1994, pp. 257–266 
 

 
 [Dev+19] 
 Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova 
 
 “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding” 
 
 In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics , ACL’19 
 
 Minneapolis, Minnesota: Association for Computational Linguistics, 2019 
 
 URL: https://arxiv.org/abs/1810.04805 
 

 
 [Dis+21] 
 Michael Diskin, Alexey Bukhtiyarov, Max Ryabinin, Lucile Saulnier, Quentin Lhoest, Anton Sinitsin, Dmitry Popov, Dmitriy Pyrkin, Maxim Kashirin, Alexander Borzunov, Albert del Moral, Denis Mazur, Ilia Kobelev, Yacine Jernite, Thomas Wolf and Gennady Pekhimenko 
 
 “Distributed Deep Learning in Open Collaborations” 
 
 In Proceedings of the 35th Conference on Neural Information Processing Systems , NeurIPS’21, 2021 
 
 URL: https://arxiv.org/abs/2106.10207 
 

 
 [Dos+21] 
 Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit and Neil Houlsby 
 
 “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale” 
 
 In Proceedings of the 9th International Conference on Learning Representations , ICLR’21, 2021 
 
 URL: https://arxiv.org/abs/2010.11929 
 

 
 [Dub75] 
 Pradeep Dubey 
 
 “On the Uniqueness of the Shapley Value” 
 
 In International Journal of Game Theory 4.3 
 
 DEU: Physica-Verlag GmbH, 1975, pp. 131–139 
 

 
 [DNW81] 
 Pradeep Dubey, Abraham Neyman and Robert Weber 
 
 “Value Theory without Efficiency” 
 
 In Mathematics of Operations Research 6.1 
 
 INFORMS, 1981, pp. 122–128 
 

 
 [DHS11] 
 John Duchi, Elad Hazan and Yoram Singer 
 
 “Adaptive Subgradient Methods for Online Learning and Stochastic Optimization” 
 
 In Journal of Machine Learning Research 12 
 
 JMLR.org, 2011, pp. 2121–2159 
 
 URL: https://jmlr.org/papers/v12/duchi11a.html 
 

 
 [Dwo+12] 
 Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold and Richard Zemel 
 
 “Fairness through Awareness” 
 
 In Proceedings of the 3rd Innovations in Theoretical Computer Science Conference , ITCS’12, 2012 
 
 URL: https://arxiv.org/abs/1104.3913 
 

 
 [Eis+22] 
 Thorsten Eisenhofer, Doreen Riepel, Varun Chandrasekaran, Esha Ghosh, Olga Ohrimenko and Nicolas Papernot 
 
 “Verifiable and Provably Secure Machine Unlearning”, 2022 
 
 arXiv: 2210.09126 [cs.LG] 
 

 
 [EGH17] 
 Rajmadhan Ekambaram, Dmitry. Goldgof and Lawrence. Hall 
 
 “Finding Label Noise Examples in Large Scale Datasets” 
 
 In Proceedings of the 2017 IEEE International Conference on Systems, Man, and Cybernetics , SMC’17, 2017 
 
 DOI: 10.1109/SMC.2017.8122985 
 

 
 [Ell76] 
 Jonas. Ellenberg 
 
 “Testing for a Single Outlier from a General Linear Regression” 
 
 In Biometrics 32.3 
 
 [Wiley, International Biometric Society], 1976, pp. 637–645 
 

 
 [FGL20] 
 Minghong Fang, Neil Gong and Jia Liu 
 
 “Influence Function based Data Poisoning Attacks to Top-N Recommender Systems” 
 
 In Proceedings of the Web Conference 2020 , WWW’20, 2020 
 
 URL: https://arxiv.org/abs/2002.08025 
 

 
 [Fel20] 
 Dan Feldman 
 
 “Introduction to Core-sets: an Updated Survey”, 2020 
 
 arXiv: 2011.09384 [cs.LG] 
 

 
 [Fel20a] 
 Vitaly Feldman 
 
 “Does Learning Require Memorization? A Short Tale about a Long Tail” 
 
 In Proceedings of the 52nd Annual ACM SIGACT Symposium on Theory of Computing , STOC’20, 2020 
 
 URL: https://arxiv.org/abs/1906.05271 
 

 
 [FZ20] 
 Vitaly Feldman and Chiyuan Zhang 
 
 “What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation” 
 
 In Proceedings of the 34th Conference on Neural Information Processing Systems , NeurIPS’20 
 
 Virtual Only: Curran Associates, Inc., 2020 
 
 URL: https://arxiv.org/abs/2008.03703 
 

 
 [Fow+21] 
 Liam Fowl, Micah Goldblum, Ping-yeh Chiang, Jonas Geiping, Wojtek Czaja and Tom Goldstein 
 
 “Adversarial Examples Make Strong Poisons” 
 
 In Proceedings of the 35th Conference on Neural Information Processing Systems , NeurIPS’21 
 
 Virtual Only: Curran Associates, Inc., 2021 
 
 URL: https://arxiv.org/abs/2106.10807 
 

 
 [GZ19] 
 Amirata Ghorbani and James Zou 
 
 “Data Shapley: Equitable Valuation of Data for Machine Learning” 
 
 In Proceedings of the 36th International Conference on Machine Learning , ICML’19, 2019 
 
 URL: https://proceedings.mlr.press/v97/ghorbani19c.html 
 

 
 [GZ20] 
 Amirata Ghorbani and James. Zou 
 
 “Neuron Shapley: Discovering the Responsible Neurons” 
 
 In Proceedings of the 34th Conference on Neural Information Processing Systems , NeurIPS’20, 2020 
 
 URL: https://arxiv.org/abs/2002.09815 
 

 
 [GH19] 
 Bruce Glymour and Jonathan Herington 
 
 “Measuring the Biases That Matter: The Ethical and Casual Foundations for Measures of Fairness in Algorithms” 
 
 In Proceedings of the 2019 ACM Conference on Fairness, Accountability, and Transparency , FAccT’19, 2019 
 

 
 [Goy+17] 
 Priya Goyal, Piotr Dollár, Ross. Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia and Kaiming He 
 
 “Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour”, 2017 
 
 arXiv: 1706.02677 [cs.CV] 
 

 
 [GR99] 
 Michel Grabisch and Marc Roubens 
 
 “An Axiomatic Approach to the Concept of Interaction Among Players in Cooperative Games” 
 
 In International Journal of Game Theory 28.4 , 1999, pp. 547–565 
 
 DOI: 10.1007/s001820050125 
 

 
 [Guo+20] 
 Chuan Guo, Tom Goldstein, Awni. Hannun and Laurens van Maaten 
 
 “Certified Data Removal from Machine Learning Models” 
 
 In Proceedings of the 37th International Conference on Machine Learning 119 , ICML’20, 2020, pp. 3832–3842 
 
 URL: https://arxiv.org/abs/1911.03030 
 

 
 [Guo+21] 
 Han Guo, Nazneen Rajani, Peter Hase, Mohit Bansal and Caiming Xiong 
 
 “FastIF: Scalable Influence Functions for Efficient Model Interpretation and Debugging” 
 
 In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing , EMNLP’21, 2021 
 
 URL: https://arxiv.org/abs/2012.15781 
 

 
 [HL21] 
 Zayd Hammoudeh and Daniel Lowd 
 
 “Simple, Attack-Agnostic Defense Against Targeted Training Set Attacks Using Cosine Similarity” 
 
 In Proceedings of the 3rd ICML Workshop on Uncertainty and Robustness in Deep Learning , UDL’21, 2021 
 

 
 [HL22] 
 Zayd Hammoudeh and Daniel Lowd 
 
 “Identifying a Training-Set Attack’s Target Using Renormalized Influence Estimation” 
 
 In Proceedings of the 29th ACM SIGSAC Conference on Computer and Communications Security , CCS’22 
 
 Los Angeles, CA: Association for Computing Machinery, 2022 
 
 URL: https://arxiv.org/abs/2201.10055 
 

 
 [HL23] 
 Zayd Hammoudeh and Daniel Lowd 
 
 “Reducing Certified Regression to Certified Classification for General Poisoning Attacks” 
 
 In Proceedings of the 1st IEEE Conference on Secure and Trustworthy Machine Learning , SaTML’23, 2023 
 
 URL: https://arxiv.org/abs/2208.13904 
 

 
 [HL24] 
 Zayd Hammoudeh and Daniel Lowd 
 
 “Provable Robustness Against a Union of ℓ 0 \ell_{0} Attacks” 
 
 In Proceedings of the 38th AAAI Conference on Artificial Intelligence , AAAI’24, 2024 
 
 URL: https://arxiv.org/abs/2302.11628 
 

 
 [HL24a] 
 Zayd Hammoudeh and Daniel Lowd 
 
 “Training Data Influence Analysis and Estimation: A Survey” 
 
 In Machine Learning , 2024 
 
 DOI: 10.1007/s10994-023-06495-7 
 

 
 [Ham74] 
 Frank. Hampel 
 
 “The Influence Curve and its Role in Robust Estimation” 
 
 In Journal of the American Statistical Association 69.346 
 
 Taylor Francis, 1974, pp. 383–393 
 

 
 [HT21] 
 Xiaochuang Han and Yulia Tsvetkov 
 
 “Fortifying Toxic Speech Detectors Against Veiled Toxicity” 
 
 In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing , EMNLP’20, 2021 
 
 URL: https://arxiv.org/abs/2010.03154 
 

 
 [HNM19] 
 Satoshi Hara, Atsushi Nitanda and Takanori Maehara 
 
 “Data Cleansing for Models Trained with SGD” 
 
 In Proceedings of the 33rd Conference on Neural Information Processing Systems , NeurIPS’19 
 
 Vancouver, Canada: Curran Associates, Inc., 2019 
 
 URL: https://arxiv.org/abs/1906.08473 
 

 
 [He+14] 
 Xinran He, Junfeng Pan, Ou Jin, Tianbing Xu, Bo Liu, Tao Xu, Yanxin Shi, Antoine Atallah, Ralf Herbrich, Stuart Bowers and Joaquinñonero Candela 
 
 “Practical Lessons from Predicting Clicks on Ads at Facebook” 
 
 In Proceedings of the Eighth International Workshop on Data Mining for Online Advertising , AdKDD’14 
 
 New York, NY, USA: Association for Computing Machinery, 2014 
 
 URL: https://research.facebook.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/ 
 

 
 [Hig+17] 
 Irina Higgins, Loïc Matthey, Arka Pal, Christopher. Burgess, Xavier Glorot, Matthew. Botvinick, Shakir Mohamed and Alexander Lerchner 
 
 “beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework” 
 
 In Proceedings of the 5th International Conference on Learning Representations , ICLR’17, 2017 
 
 URL: https://openreview.net/forum?id=Sy2fzU9gl 
 

 
 [HSS08] 
 Thomas Hofmann, Bernhard Schölkopf and Alexander. Smola 
 
 “Kernel Methods in Machine Learning” 
 
 In Annals of Statistics 36.3 , 2008, pp. 1171–1220 
 
 URL: https://arxiv.org/abs/math/0701907 
 

 
 [Hog79] 
 Robert. Hogg 
 
 “Statistical Robustness: One View of its Use in Applications Today” 
 
 In The American Statistician 33.3 
 
 Taylor Francis, 1979, pp. 108–115 
 

 
 [Hub81] 
 Peter Huber 
 
 “Robust Statistics” 
 
 John Wiley Sons, 1981 
 

 
 [Hub64] 
 Peter. Huber 
 
 “Robust Estimation of a Location Parameter” 
 
 In Annals of Mathematical Statistics 35.1 , 1964, pp. 73–101 
 

 
 [Hut+20] 
 Ben Hutchinson, Vinodkumar Prabhakaran, Emily Denton, Kellie Webster, Yu Zhong and Stephen Denuyl 
 
 “Social Biases in NLP Models as Barriers for Persons with Disabilities” 
 
 In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics 
 
 Association for Computational Linguistics, 2020 
 
 DOI: 10.18653/v1/2020.acl-main.487 
 

 
 [Ily+22] 
 Andrew Ilyas, Sung Park, Logan Engstrom, Guillaume Leclerc and Aleksander Madry 
 
 “Datamodels: Understanding Predictions with Data and Data with Predictions” 
 
 In Proceedings of the 39th International Conference on Machine Learning , ICML’22 
 
 PMLR, 2022 
 
 URL: https://arxiv.org/abs/2202.00622 
 

 
 [Jae72] 
 Louis. Jaeckel 
 
 “The Infinitesimal Jackknife”, 1972 
 

 
 [Jag+21] 
 Matthew Jagielski, Giorgio Severi, Niklas Pousette and Alina Oprea 
 
 “Subpopulation Data Poisoning Attacks” 
 
 In Proceedings of the 28th ACM SIGSAC Conference on Computer and Communications Security , CCS ’21 
 
 Virtual Only: Association for Computing Machinery, 2021 
 
 URL: https://arxiv.org/abs/2006.14026 
 

 
 [Jia+22] 
 Jinyuan Jia, Yupei Liu, Xiaoyu Cao and Neil Gong 
 
 “Certified Robustness of Nearest Neighbors against Data Poisoning and Backdoor Attacks” 
 
 In Proceedings of the 36th AAAI Conference on Artificial Intelligence , AAAI’22, 2022 
 
 URL: https://arxiv.org/abs/2012.03765 
 

 
 [Jia+19] 
 Ruoxi Jia, David Dao, Boxin Wang, Frances Hubis, Nezihe Gürel, Bo Li, Ce Zhang, Costas. Spanos and Dawn Song 
 
 “Efficient Task-Specific Data Valuation for Nearest Neighbor Algorithms” 
 
 In Proceedings of the VLDB Endowment , PVLDB’19, 2019 
 
 URL: https://arxiv.org/abs/1908.08619 
 

 
 [Jia+19a] 
 Ruoxi Jia, David Dao, Boxin Wang, Frances Hubis, Nick Hynes, Nezihe Gürel, Bo Li, Ce Zhang, Dawn Song and Costas. Spanos 
 
 “Towards Efficient Data Valuation Based on the Shapley Value” 
 
 In Proceedings of the 22nd Conference on Artificial Intelligence and Statistics , AISTATS’19, 2019, pp. 1167–1176 
 
 URL: https://arxiv.org/abs/1902.10275 
 

 
 [Jia+21] 
 Ruoxi Jia, Fan Wu, Xuehui Sun, Jiacen Xu, David Dao, Bhavya Kailkhura, Ce Zhang, Bo Li and Dawn Song 
 
 “Scalability vs. Utility: Do We Have to Sacrifice One for the Other in Data Importance Quantification?” 
 
 In Proceedings of the 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition , CVPR’21, 2021 
 
 URL: https://arxiv.org/abs/1911.07128 
 

 
 [Jia+21a] 
 Ziheng Jiang, Chiyuan Zhang, Kunal Talwar and Michael Mozer 
 
 “Characterizing Structural Regularities of Labeled Data in Overparameterized Models” 
 
 In Proceedings of the 38th International Conference on Machine Learning , ICML’21, 2021, pp. 5034–5044 
 
 URL: https://arxiv.org/abs/2002.03206 
 

 
 [JL84] 
 William. Johnson and Joram Lindenstrauss 
 
 “Extensions of Lipschitz Mappings into a Hilbert Space” 
 
 In Contemporary Mathematics 26 
 
 American Mathematical Society, 1984, pp. 189–206 
 

 
 [JW78] 
 John. Jr. and Roy. Welsch 
 
 “Techniques for nonlinear least squares and robust regression” 
 
 In Communications in Statistics - Simulation and Computation 7.4 
 
 Taylor Francis, 1978, pp. 345–359 
 

 
 [KS21] 
 Karthikeyan K and Anders Søgaard 
 
 “Revisiting Methods for Finding Influential Examples”, 2021 
 
 arXiv: 2111.04683 [cs.LG] 
 

 
 [KWR22] 
 Nikhil Kandpal, Eric Wallace and Colin Raffel 
 
 “Deduplicating Training Data Mitigates Privacy Risks in Language Models” 
 
 In Proceedings of the 39th International Conference on Machine Learning , ICML’22 
 
 PMLR, 2022 
 
 URL: https://arxiv.org/abs/2202.06539 
 

 
 [KDK13] 
 Saul. Kassin, Itiel. Dror and Jeff Kukucka 
 
 “The Forensic Confirmation Bias: Problems, Perspectives, and Proposed Solutions” 
 
 In Journal of Applied Research in Memory and Cognition 2.1 , 2013, pp. 42–52 
 

 
 [Kha+19] 
 Rajiv Khanna, Been Kim, Joydeep Ghosh and Oluwasanmi Koyejo 
 
 “Interpreting Black Box Predictions using Fisher Kernels” 
 
 In Proceedings of the 22nd Conference on Artificial Intelligence and Statistics , AISTATS’19, 2019 
 
 URL: https://arxiv.org/abs/1810.10118 
 

 
 [KCC23] 
 Nohyun Ki, Hoyong Choi and Hye Chung 
 
 “Data Valuation Without Training of a Model” 
 
 In Proceedings of the 11th International Conference on Learning Representations , ICLR’23, 2023 
 
 URL: https://openreview.net/forum?id=XIzO8zr-WbM 
 

 
 [KB15] 
 Diederik. Kingma and Jimmy Ba 
 
 “Adam: A Method for Stochastic Optimization” 
 
 In Proceedings of the 3rd International Conference on Learning Representations , ICLR’15, 2015 
 
 URL: https://arxiv.org/abs/1412.6980 
 

 
 [KW14] 
 Diederik. Kingma and Max Welling 
 
 “Auto-Encoding Variational Bayes” 
 
 In Proceedings of the 2nd International Conference on Learning Representations , ICLR’14, 2014 
 
 URL: https://arxiv.org/abs/1312.6114 
 

 
 [Kiz16] 
 René. Kizilcec 
 
 “How Much Information? Effects of Transparency on Trust in an Algorithmic Interface” 
 
 In Proceedings of the 2016 CHI Conference on Human Factors in Computing Systems , CHI’16 
 
 San Jose, California, USA: Association for Computing Machinery, 2016, pp. 2390–2395 
 

 
 [Kni17] 
 Will Knight 
 
 “The Dark Secret at the Heart of AI” 
 
 In MIT Technology Review , 2017 
 
 URL: https://www.technologyreview.com/2017/04/11/5113/the-dark-secret-at-the-heart-of-ai/ 
 

 
 [Kob+20] 
 Sosuke Kobayashi, Sho Yokoi, Jun Suzuki and Kentaro Inui 
 
 “Efficient Estimation of Influence of a Training Instance” 
 
 In Proceedings of SustaiNLP: Workshop on Simple and Efficient Natural Language Processing 
 
 Online: Association for Computational Linguistics, 2020 
 
 URL: https://arxiv.org/abs/2012.04207 
 

 
 [Koh+19] 
 Pang Koh, Kai-Siang Ang, Hubert.. Teo and Percy Liang 
 
 “On the Accuracy of Influence Functions for Measuring Group Effects” 
 
 In Proceedings of the 33rd International Conference on Neural Information Processing Systems , NeurIPS’19 
 
 Red Hook, NY, USA: Curran Associates Inc., 2019 
 
 URL: https://arxiv.org/abs/1905.13289 
 

 
 [KL17] 
 Pang Koh and Percy Liang 
 
 “Understanding Black-box Predictions via Influence Functions” 
 
 In Proceedings of the 34th International Conference on Machine Learning , ICML’17 
 
 Sydney, Australia: PMLR, 2017 
 
 URL: https://arxiv.org/abs/1703.04730 
 

 
 [KSH22] 
 Shuming Kong, Yanyan Shen and Linpeng Huang 
 
 “Resolving Training Biases via Influence-based Data Relabeling” 
 
 In Proceedings of the 10th International Conference on Learning Representations , ICLR’22, 2022 
 
 URL: https://openreview.net/forum?id=EskfH0bwNVn 
 

 
 [KC21] 
 Zhifeng Kong and Kamalika Chaudhuri 
 
 “Understanding Instance-based Interpretability of Variational Auto-Encoders” 
 
 In Proceedings of the 35th Conference on Neural Information Processing Systems , NeurIPS’21 
 
 Virtual Only: Curran Associates, Inc., 2021 
 
 URL: https://arxiv.org/abs/2105.14203 
 

 
 [Kri+16] 
 Sanjay Krishnan, Jiannan Wang, Eugene Wu, Michael. Franklin and Ken Goldberg 
 
 “ActiveClean: Interactive Data Cleaning for Statistical Modeling” 
 
 In Proceedings of the VLDB Endowment 
 
 VLDB Endowment, 2016 
 
 URL: https://www.vldb.org/pvldb/vol9/p948-krishnan.pdf 
 

 
 [KW19] 
 Sanjay Krishnan and Eugene Wu 
 
 “AlphaClean: Automatic Generation of Data Cleaning Pipelines”, 2019 
 
 arXiv: 1904.11827 [cs.DB] 
 

 
 [KNH14] 
 Alex Krizhevsky, Vinod Nair and Geoffrey Hinton 
 
 “The CIFAR-10 Dataset”, 2014 
 

 
 [KSH12] 
 Alex Krizhevsky, Ilya Sutskever and Geoffrey Hinton 
 
 “ImageNet Classification with Deep Convolutional Neural Networks” 
 
 In Proceedings of the 25th Conference on Neural Information Processing Systems , NeurIPS’12, 2012, pp. 1097–1105 
 

 
 [Kur+19] 
 Keita Kurita, Nidhi Vyas, Ayush Pareek, Alan Black and Yulia Tsvetkov 
 
 “Measuring Bias in Contextualized Word Representations” 
 
 In Proceedings of the First Workshop on Gender Bias in Natural Language Processing 
 
 Association for Computational Linguistics, 2019 
 
 URL: https://arxiv.org/abs/1906.07337 
 

 
 [KZ22] 
 Yongchan Kwon and James Zou 
 
 “Beta Shapley: A Unified and Noise-Reduced Data Valuation Framework for Machine Learning” 
 
 In Proceedings of the 25th Conference on Artificial Intelligence and Statistics , AISTATS’22 
 
 PMLR, 2022 
 
 URL: https://arxiv.org/abs/2110.14049 
 

 
 [Lec89] 
 Yvan. Leclerc 
 
 “Constructing Simple Stable Descriptions for Image Partitioning” 
 
 In International Journal of Computer Vision 3.1 
 
 Springer ScienceBusiness Media LLC, 1989, pp. 73–102 
 

 
 [LeC+98] 
 Yann LeCun, Léon Bottou, Yoshua Bengio and Patrick Haffner 
 
 “Gradient-Based Learning Applied to Document Recognition” 
 
 In Proceedings of the IEEE 86 , 1998, pp. 2278–2324 
 

 
 [Lee+20] 
 Donghoon Lee, Hyunsin Park, Trung Pham and Chang. Yoo 
 
 “Learning Augmentation Network via Influence Functions” 
 
 In Proceedings of the 33rd Conference on Computer Vision and Pattern Recognition , CVPR’20, 2020 
 

 
 [LF21] 
 Alexander Levine and Soheil Feizi 
 
 “Deep Partition Aggregation: Provable Defenses against General Poisoning Attacks” 
 
 In Proceedings of the 9th International Conference on Learning Representations , ICLR’21, 2021 
 
 URL: https://arxiv.org/abs/2006.14768 
 

 
 [Li+22] 
 Yiming Li, Baoyuan Wu, Yong Jiang, Zhifeng Li and Shu-Tao Xia 
 
 “Backdoor Learning: A Survey” 
 
 In IEEE Transactions on Neural Networks and Learning Systems , 2022 
 
 DOI: 10.1109/TNNLS.2022.3182979 
 

 
 [LLY21] 
 Weixin Liang, Kai-Hui Liang and Zhou Yu 
 
 “HERALD: An Annotation Efficient Method to Detect User Disengagement in Social Conversations” 
 
 In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing , ACL-IJCNLP’21 
 
 Association for Computational Linguistics, 2021 
 
 URL: https://arxiv.org/abs/2106.00162 
 

 
 [LDA09] 
 Brian. Lim, Anind. Dey and Daniel Avrahami 
 
 “Why and Why Not Explanations Improve the Intelligibility of Context-Aware Intelligent Systems” 
 
 In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems , CHI’09 
 
 Boston, MA, USA: Association for Computing Machinery, 2009, pp. 2119–2128 
 

 
 [Lin+22] 
 Jinkun Lin, Anqi Zhang, Mathias Lecuyer, Jinyang Li, Aurojit Panda and Siddhartha Sen 
 
 “Measuring the Effect of Training Data on Deep Learning Predictions via Randomized Experiments” 
 
 In Proceedings of the 39th International Conference on Machine Learning , ICML’22, 2022 
 
 URL: https://arxiv.org/abs/2206.10013 
 

 
 [Lip18] 
 Zachary. Lipton 
 
 “The Mythos of Model Interpretability: In Machine Learning, the Concept of Interpretability is Both Important and Slippery.” 
 
 In Queue 16.3 
 
 New York, NY, USA: Association for Computing Machinery, 2018, pp. 31–57 
 
 URL: https://dl.acm.org/doi/10.1145/3236386.3241340 
 

 
 [LDG18] 
 Kang Liu, Brendan Dolan-Gavitt and Siddharth Garg 
 
 “Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks” 
 
 In Proceedings of the International Symposium on Research in Attacks, Intrusions, and Defenses , RAID’18 
 
 Heraklion, Crete, Greece: Springer, 2018, pp. 273–294 
 
 URL: https://arxiv.org/abs/1805.12185 
 

 
 [Liu+21] 
 Zhuoming Liu, Hao Ding, Huaping Zhong, Weijia Li, Jifeng Dai and Conghui He 
 
 “Influence Selection for Active Learning” 
 
 In Proceedings of the 18th International Conference on Computer Vision , ICCV’21, 2021 
 
 URL: https://arxiv.org/abs/2108.09331 
 

 
 [LL17] 
 Scott. Lundberg and Su-In Lee 
 
 “A Unified Approach to Interpreting Model Predictions” 
 
 In Proceedings of the 31st International Conference on Neural Information Processing Systems , NeurIPS’17, 2017 
 
 URL: https://arxiv.org/abs/1705.07874 
 

 
 [Mah77] 
 Michael. Mahoney 
 
 “Publication Prejudices: An Experimental Study of Confirmatory Bias in the Peer Review System” 
 
 In Cognitive Therapy and Research 1.2 , 1977, pp. 161–175 
 

 
 [Meh+21] 
 Ninareh Mehrabi, Fred Morstatter, Nripsuta Saxena, Kristina Lerman and Aram Galstyan 
 
 “A Survey on Bias and Fairness in Machine Learning” 
 
 In ACM Computing Surveys 54.6 
 
 New York, NY, USA: Association for Computing Machinery, 2021 
 
 URL: https://arxiv.org/abs/1908.09635 
 

 
 [MBL20] 
 Baharan Mirzasoleiman, Jeff Bilmes and Jure Leskovec 
 
 “Coresets for Data-Efficient Training of Machine Learning Models” 
 
 In Proceedings of the 37th International Conference on Machine Learning , ICML’20, 2020 
 
 URL: https://arxiv.org/abs/1906.01827 
 

 
 [NDM13] 
 Sherry Nakhaeizadeh, Itiel Dror and Ruth Morgan 
 
 “Cognitive Bias in Forensic Anthropology: Visual Assessment of Skeletal Remains is Susceptible to Confirmation Bias” 
 
 In Science Justice 54.3 , 2013, pp. 208–214 
 

 
 [Neg+12] 
 Sahand. Negahban, Pradeep Ravikumar, Martin. Wainwright and Bin Yu 
 
 “A Unified Framework for High-Dimensional Analysis of M M -Estimators with Decomposable Regularizers” 
 
 In Statistical Science 27.4 
 
 Institute of Mathematical Statistics, 2012, pp. 538–557 
 
 URL: https://arxiv.org/abs/1010.2731 
 

 
 [NSO23] 
 Elisa Nguyen, Minjoon Seo and Seong Oh 
 
 “A Bayesian Perspective On Training Data Attribution”, 2023 
 
 arXiv: 2305.19765 [cs.LG] 
 

 
 [Ngu+22] 
 Thanh Nguyen, Thanh Huynh, Phi Nguyen, Alan-Chung Liew, Hongzhi Yin and Quoc Nguyen 
 
 “A Survey of Machine Unlearning” 
 
 In arXiv preprint arXiv:2209.02299 , 2022 
 
 arXiv: 2209.02299 [cs.LG] 
 

 
 [Oh+21] 
 Sejoon Oh, Sungchul Kim, Ryan. Rossi and Srijan Kumar 
 
 “Influence-guided Data Augmentation for Neural Tensor Completion” 
 
 In Proceedings of the 30th ACM International Conference on Information and Knowledge Management , CIKM’21 
 
 ACM, 2021 
 
 URL: https://arxiv.org/abs/2108.10248 
 

 
 [Oh+22] 
 Sejoon Oh, Berk Ustun, Julian McAuley and Srijan Kumar 
 
 “Rank List Sensitivity of Recommender Systems to Interaction Perturbations” 
 
 In Proceedings of the 31st ACM International Conference on Information and Knowledge Management , CIKM’22, 2022 
 
 ACM 
 
 URL: https://arxiv.org/abs/2201.12686 
 

 
 [Par+23] 
 Sung Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc and Aleksander Madry 
 
 “TRAK: Attributing Model Behavior at Scale” 
 
 In Proceedings of the 40th International Conference on Machine Learning , ICML’23, 2023 
 
 URL: https://arxiv.org/abs/2303.14186 
 

 
 [Pea94] 
 Barak. Pearlmutter 
 
 “Fast Exact Multiplication by the Hessian” 
 
 In Neural Computation 6 , 1994, pp. 147–160 
 

 
 [Ple+20] 
 Geoff Pleiss, Tianyi Zhang, Ethan Elenberg and Kilian. Weinberger 
 
 “Identifying Mislabeled Data Using the Area Under the Margin Ranking” 
 
 In Proceedings of the 34th International Conference on Neural Information Processing Systems , NeurIPS’20 
 
 Red Hook, NY, USA: Curran Associates Inc., 2020 
 
 URL: https://arxiv.org/abs/2001.10528 
 

 
 [Pru+20] 
 Garima Pruthi, Frederick Liu, Satyen Kale and Mukund Sundararajan 
 
 “Estimating Training Data Influence by Tracing Gradient Descent” 
 
 In Proceedings of the 34th Conference on Neural Information Processing Systems , NeurIPS’20 
 
 Virtual Only: Curran Associates, Inc., 2020 
 
 URL: https://arxiv.org/abs/2002.08484 
 

 
 [Qia99] 
 Ning Qian 
 
 “On the Momentum Term in Gradient Descent Learning Algorithms” 
 
 In Neural Networks 12.1 , 1999, pp. 145–151 
 

 
 [Ras+22] 
 Soham Raste, Rahul Singh, Joel Vaughan and Vijayan. Nair 
 
 “Quantifying Inherent Randomness in Machine Learning Algorithms”, 2022 
 
 arXiv: 2206.12353 [stat.ML] 
 

 
 [Rec11] 
 Benjamin Recht 
 
 “A Simpler Approach to Matrix Completion” 
 
 In Journal of Machine Learning Research 12 , 2011, pp. 3413–3430 
 
 URL: https://arxiv.org/abs/0910.0651 
 

 
 [Red+21] 
 Vijay Reddi, Greg Diamos, Pete Warden, Peter Mattson and David Kanter 
 
 “Data Engineering for Everyone”, 2021 
 
 arXiv: 2102.11447 [cs.LG] 
 

 
 [Ren+21] 
 Pengzhen Ren, Yun Xiao, Xiaojun Chang, Po-Yao Huang, Zhihui Li, Brij. Gupta, Xiaojiang Chen and Xin Wang 
 
 “A Survey of Deep Active Learning” 
 
 In ACM Computing Surveys 54.9 
 
 New York, NY, USA: Association for Computing Machinery, 2021 
 
 URL: https://arxiv.org/abs/2009.00236 
 

 
 [Ren14] 
 Alexander Renkl 
 
 “Toward an Instructionally Oriented Theory of Example-Based Learning” 
 
 In Cognitive Science 38.1 , 2014, pp. 1–37 
 
 URL: https://onlinelibrary.wiley.com/doi/full/10.1111/cogs.12086 
 

 
 [RHS09] 
 Alexander Renkl, Tatjana Hilbert and Silke Schworm 
 
 “Example-Based Learning in Heuristic Domains: A Cognitive Load Theory Account” 
 
 In Educational Psychology Review 21.1 , 2009, pp. 67–78 
 

 
 [Rez+23] 
 Keivan Rezaei, Kiarash Banihashem, Atoosa Chegini and Soheil Feizi 
 
 “Run-Off Election: Improved Provable Defense against Data Poisoning Attacks” 
 
 In Proceedings of the 40th International Conference on Machine Learning , ICML’23, 2023 
 
 URL: https://arxiv.org/abs/2302.02300 
 

 
 [RMW14] 
 Danilo Rezende, Shakir Mohamed and Daan Wierstra 
 
 “Stochastic Backpropagation and Approximate Inference in Deep Generative Models” 
 
 In Proceedings of the 31st International Conference on International Conference on Machine Learning , ICML’14, 2014 
 
 URL: https://arxiv.org/abs/1401.4082 
 

 
 [Rou94] 
 Peter Rousseeuw 
 
 “Least Median of Squares Regression” 
 
 In Journal of the American Statistical Association 79.388 
 
 Taylor Francis, 1994 
 

 
 [RL87] 
 Peter. Rousseeuw and Annick.. Leroy 
 
 “Robust Regression and Outlier Detection” 
 
 USA: John Wiley Sons, Inc., 1987 
 

 
 [Roz+22] 
 Benedek Rozemberczki, Lauren Watson, Péter Bayer, Hao-Tsung Yang, Olivér Kiss, Sebastian Nilsson and Rik Sarkar 
 
 “The Shapley Value in Machine Learning”, 2022 
 
 arXiv: 2202.05594 [cs.LG] 
 

 
 [Rud19] 
 Cynthia Rudin 
 
 “Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead” 
 
 In Nature Machine Intelligence 1.5 , 2019, pp. 206–215 
 
 URL: https://arxiv.org/abs/1811.10154 
 

 
 [RHW86] 
 David. Rumelhart, Geoffrey. Hinton and Ronald. Williams 
 
 “Learning Representations by Back-Propagating Errors” 
 
 In Nature 323.6088 , 1986, pp. 533–536 
 

 
 [Sax+19] 
 Nripsuta Saxena, Karen Huang, Evan DeFilippis, Goran Radanovic, David. Parkes and Yang Liu 
 
 “How Do Fairness Definitions Fare? Examining Public Attitudes Towards Algorithmic Definitions of Fairness” 
 
 In Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society , AIES’19, 2019 
 
 URL: https://arxiv.org/abs/1811.03654 
 

 
 [Sch+23] 
 Andrea Schioppa, Katja Filippova, Ivan Titov and Polina Zablotskaia 
 
 “Theoretical and Practical Perspectives on what Influence Functions Do”, 2023 
 
 arXiv: https://arxiv.org/abs/2305.16971 
 

 
 [Sch+22] 
 Andrea Schioppa, Polina Zablotskaia, David Torres and Artem Sokolov 
 
 “Scaling Up Influence Functions” 
 
 In Proceedings of the 36th AAAI Conference on Artificial Intelligence , AAAI’22, 2022 
 
 URL: https://arxiv.org/abs/2112.03052 
 

 
 [SHS01] 
 Bernhard Schölkopf, Ralf Herbrich and Alex. Smola 
 
 “A Generalized Representer Theorem” 
 
 In Proceedings of the 14th Annual Conference on Computational Learning Theory and 5th European Conference on Computational Learning Theory , COLT’01/EuroCOLT’01 
 
 Berlin, Heidelberg: Springer-Verlag, 2001, pp. 416–426 
 

 
 [Sha+18] 
 Ali Shafahi, W. Huang, Mahyar Najibi, Octavian Suciu, Christoph Studer, Tudor Dumitras and Tom Goldstein 
 
 “Poison Frogs! Targeted Clean-Label Poisoning Attacks on Neural Networks” 
 
 In Proceedings of the 32nd Conference on Neural Information Processing Systems , NeurIPS’18 
 
 Montreal, Canada: Curran Associates, Inc., 2018 
 
 URL: https://arxiv.org/abs/1804.00792 
 

 
 [Sha53] 
 Lloyd. Shapley 
 
 “A Value for n-Person Games” 
 
 In Contributions to the Theory of Games II 
 
 Princeton, NJ USA: Princeton University Press, 1953, pp. 307–317 
 

 
 [SR88] 
 Lloyd. Shapley and Alvin. Roth 
 
 “The Shapley Value: Essays in Honor of Lloyd S. Shapley” 
 
 Cambridge University Press, 1988 
 

 
 [Sha+18a] 
 Boris Sharchilev, Yury Ustinovskiy, Pavel Serdyukov and Maarten de Rijke 
 
 “Finding Influential Training Samples for Gradient Boosted Decision Trees” 
 
 In Proceedings of the 35th International Conference on Machine Learning , ICML’18 
 
 PMLR, 2018, pp. 4577–4585 
 
 URL: https://arxiv.org/abs/1802.06640 
 

 
 [SC68] 
 William. Snedecor and George. Cochran 
 
 “Statistical Methods” 
 
 Iowa State University Press, 1968 
 

 
 [Sri61] 
 K.. Srikantan 
 
 “Testing for the Single Outlier in a Regression Model” 
 
 In Indian Journal of Statistics 23.3 
 
 Springer, 1961, pp. 251–260 
 

 
 [SKL17] 
 Jacob Steinhardt, Pang Koh and Percy Liang 
 
 “Certified Defenses for Data Poisoning Attacks” 
 
 In Proceedings of the 31st Conference on Neural Information Processing Systems , NeurIPS’17 
 
 Long Beach, California, USA: Curran Associates, Inc., 2017 
 
 URL: https://arxiv.org/abs/1706.03691 
 

 
 [SGM20] 
 Emma Strubell, Ananya Ganesh and Andrew McCallum 
 
 “Energy and Policy Considerations for Modern Deep Learning Research” 
 
 In Proceedings of the 34th AAAI Conference on Artificial Intelligence , AAAI’20, 2020 
 
 URL: https://arxiv.org/abs/1906.02243 
 

 
 [SWS21] 
 Yi Sui, Ga Wu and Scott Sanner 
 
 “Representer Point Selection via Local Jacobian Expansion for Post-hoc Classifier Explanation of Deep Neural Networks and Ensemble Models” 
 
 In Proceedings of the 35th Conference on Neural Information Processing Systems , NeurIPS’21 
 
 Virtual Only: Curran Associates, Inc., 2021 
 
 URL: https://openreview.net/forum?id=Wl32WBZnSP4 
 

 
 [SD21] 
 Cecilia Summers and Michael. Dinneen 
 
 “Nondeterminism and Instability in Neural Network Optimization” 
 
 In Proceedings of the 38th International Conference on Machine Learning , ICML’21, 2021 
 
 URL: https://arxiv.org/abs/2103.04514 
 

 
 [SDA20] 
 Mukund Sundararajan, Kedar Dhamdhere and Ashish Agarwal 
 
 “The Shapley Taylor Interaction Index” 
 
 In Proceedings of the 37th International Conference on Machine Learning , ICML’20, 2020 
 
 URL: http://proceedings.mlr.press/v119/sundararajan20a 
 

 
 [SN20] 
 Mukund Sundararajan and Amir Najmi 
 
 “The Many Shapley Values for Model Explanation” 
 
 In Proceedings of the 37th International Conference on Machine Learning , ICML’20, 2020, pp. 9269–9278 
 
 URL: https://arxiv.org/abs/1908.08474 
 

 
 [TC19] 
 Yi Tan and L. Celis 
 
 “Assessing Social and Intersectional Biases in Contextualized Word Representations” 
 
 In Proceedings of the 33rd Conference on Neural Information Processing Systems , NeurIPS’19 
 
 Vancouver, Canada: Curran Associates, Inc., 2019 
 
 URL: https://arxiv.org/abs/1911.01485 
 

 
 [Ter+21] 
 Naoyuki Terashita, Hiroki Ohashi, Yuichi Nonaka and Takashi Kanemaru 
 
 “Influence Estimation for Generative Adversarial Networks” 
 
 In Proceedings of the 9th International Conference on Learning Representations , ICLR’21, 2021 
 
 URL: https://arxiv.org/abs/2101.08367 
 

 
 [Thi+22] 
 Hugo Thimonier, Fabrice Popineau, Arpad Rimmel, Bich-Liên Doan and Fabrice Daniel 
 
 “TracInAD: Measuring Influence for Anomaly Detection” 
 
 In Proceedings of the 2022 International Joint Conference on Neural Networks , IJCNN’22, 2022 
 
 URL: https://arxiv.org/abs/2205.01362 
 

 
 [Tib96] 
 Robert Tibshirani 
 
 “Regression Shrinkage and Selection via the Lasso” 
 
 In Journal of the Royal Statistical Society (Series B) 58 , 1996, pp. 267–288 
 

 
 [TMB73] 
 G.. Tietjen, R.. Moore and R.. Beckman 
 
 “Testing for a Single Outlier in Simple Linear Regression” 
 
 In Technometrics 15.4 
 
 Taylor Francis, 1973, pp. 717–721 
 

 
 [TB18] 
 Daniel Ting and Eric Brochu 
 
 “Optimal Subsampling with Influence Functions” 
 
 In Proceedings of the 32nd Conference on Neural Information Processing Systems , NeurIPS’18 
 
 Curran Associates, Inc., 2018 
 
 URL: https://arxiv.org/abs/1709.01716 
 

 
 [TYR23] 
 Che-Ping Tsai, Chih-Kuan Yeh and Pradeep Ravikumar 
 
 “Faith-Shap: The Faithful Shapley Interaction Index” 
 
 In Journal of Machine Learning Research 24.94 , 2023, pp. 1–42 
 
 URL: https://arxiv.org/abs/2203.00870 
 

 
 [Tsa+23] 
 Che-Ping Tsai, Jiong Zhang, Eli Chien, Hsiang-Fu Yu, Cho-Jui Hsieh and Pradeep Ravikumar 
 
 “Representer Point Selection for Explaining Regularized High-dimensional Models” 
 
 In Proceedings of the 40th International Conference on Machine Learning , ICML’23, 2023 
 
 URL: https://arxiv.org/abs/2305.20002 
 

 
 [Tuk+23] 
 Murad Tukan, Samson Zhou, Alaa Maalouf, Daniela Rus, Vladimir Braverman and Dan Feldman 
 
 “Provable Data Subset Selection for Efficient Neural Networks Training” 
 
 In Proceedings of the 40th International Conference on Machine Learning , ICML’23, 2023 
 
 URL: https://arxiv.org/abs/2303.05151 
 

 
 [vW21] 
 Gerrit.. van den Burg and Christopher.. Williams 
 
 “On Memorization in Probabilistic Deep Generative Models” 
 
 In Proceedings of the 35th Conference on Neural Information Processing Systems , NeurIPS’21 
 
 Curran Associates, Inc., 2021 
 
 URL: https://arxiv.org/abs/2106.03216 
 

 
 [Wal+21] 
 Eric Wallace, Tony. Zhao, Shi Feng and Sameer Singh 
 
 “Concealed Data Poisoning Attacks on NLP Models” 
 
 In Proceedings of the North American Chapter of the Association for Computational Linguistics , NAACL’21, 2021 
 
 URL: https://arxiv.org/abs/2010.12563 
 

 
 [Wan+19] 
 Bolun Wang, Yuanshun Yao, Shawn Shan, Huiying Li, Bimal Viswanath, Haitao Zheng and Ben. Zhao 
 
 “Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks” 
 
 In Proceedings of the 40th IEEE Symposium on Security and Privacy , SP’19, 2019 
 
 URL: https://ieeexplore.ieee.org/document/8835365 
 

 
 [WJ23] 
 Jiachen. Wang and Ruoxi Jia 
 
 “Data Banzhaf: A Robust Data Valuation Framework for Machine Learning” 
 
 In Proceedings of the 26th International Conference on Artificial Intelligence and Statistics , AISTATS’23, 2023 
 
 URL: https://arxiv.org/abs/2205.15466 
 

 
 [WLF22] 
 Wenxiao Wang, Alexander Levine and Soheil Feizi 
 
 “Improved Certified Defenses against Data Poisoning with (Deterministic) Finite Aggregation” 
 
 In Proceedings of the 39th International Conference on Machine Learning , ICML’22, 2022 
 
 URL: https://arxiv.org/abs/2202.02628 
 

 
 [Wan+20] 
 Zifeng Wang, Hong Zhu, Zhenhua Dong, Xiuqiang He and Shao-Lun Huang 
 
 “Less Is Better: Unweighted Data Subsampling via Influence Function” 
 
 In Proceedings of the 34th AAAI Conference on Artificial Intelligence , AAAI’20 
 
 AAAI Press, 2020, pp. 6340–6347 
 
 URL: https://arxiv.org/abs/1912.01321 
 

 
 [WIB15] 
 Kai Wei, Rishabh Iyer and Jeff Bilmes 
 
 “Submodularity in Data Subset Selection and Active Learning” 
 
 In Proceedings of the 32nd International Conference on Machine Learning , ICML’15 
 
 Lille, France: PMLR, 2015 
 
 URL: https://proceedings.mlr.press/v37/wei15.html 
 

 
 [Woj+16] 
 Mike Wojnowicz, Ben Cruz, Xuan Zhao, Brian Wallace, Matt Wolff, Jay Luan and Caleb Crable 
 
 “‘Influence Sketching’: Finding Influential Samples in Large-Scale Regressions” 
 
 In Proceedings of the 2016 IEEE International Conference on Big Data , BigData’16 
 
 IEEE, 2016 
 
 URL: https://arxiv.org/abs/1611.05923 
 

 
 [Woo14] 
 David. Woodruff 
 
 “Sketching as a Tool for Numerical Linear Algebra” 
 
 In Foundations and Trends in Theoretical Computer Science 10.1–2 
 
 Hanover, MA, USA: Now Publishers Inc., 2014, pp. 1–157 
 
 URL: https://arxiv.org/abs/1411.4357 
 

 
 [Xia22] 
 Chloe Xiang 
 
 “Scientists Increasingly Can’t Explain How AI Works” 
 
 In Vice , 2022 
 
 URL: https://www.vice.com/en/article/y3pezm/scientists-increasingly-cant-explain-how-ai-works 
 

 
 [Yam20] 
 Roman. Yampolskiy 
 
 “Unexplainability and Incomprehensibility of AI” 
 
 In Journal of Artificial Intelligence and Consciousness 7.2 , 2020, pp. 277–291 
 
 URL: https://arxiv.org/abs/1907.03869 
 

 
 [YP21] 
 Tom Yan and Ariel. Procaccia 
 
 “If You Like Shapley Then You’ll Love the Core” 
 
 In Proceedings of the 35th AAAI Conference on Artificial Intelligence , AAAI’21 
 
 Virtual Only: Association for the Advancement of Artificial Intelligence, 2021 
 
 URL: https://ojs.aaai.org/index.php/AAAI/article/view/16721 
 

 
 [Yan+17] 
 Jian Yang, Lei Luo, Jianjun Qian, Ying Tai, Fanlong Zhang and Yong Xu 
 
 “Nuclear Norm Based Matrix Regression with Applications to Face Recognition with Occlusion and Illumination Changes” 
 
 In IEEE Transactions on Pattern Analysis and Machine Intelligence 39.1 , 2017, pp. 156–171 
 
 DOI: 10.1109/TPAMI.2016.2535218 
 

 
 [Yan+21] 
 Jingkang Yang, Kaiyang Zhou, Yixuan Li and Ziwei Liu 
 
 “Generalized Out-of-Distribution Detection: A Survey”, 2021 
 
 arXiv: 2110.11334 [cs.CV] 
 

 
 [Yan+23] 
 Shuo Yang, Zeke Xie, Hanyu Peng, Min Xu, Mingming Sun and Ping Li 
 
 “Dataset Pruning: Reducing Training Data by Examining Generalization Influence” 
 
 In Proceedings of the 11th International Conference on Learning Representations , ICLR’23, 2023 
 
 URL: https://arxiv.org/abs/2205.09329 
 

 
 [Yeh+22] 
 Chih-Kuan Yeh, Ankur Taly, Mukund Sundararajan, Frederick Liu and Pradeep Ravikumar 
 
 “First is Better Than Last for Language Data Influence” 
 
 In Proceedings of the 36th Conference on Neural Information Processing Systems , NeurIPS’22 
 
 Curran Associates, Inc., 2022 
 
 URL: https://arxiv.org/abs/2202.11844 
 

 
 [Yeh+18] 
 Chih“=/Kuan Yeh, Joon Kim, Ian.H. Yen and Pradeep Ravikumar 
 
 “Representer Point Selection for Explaining Deep Neural Networks” 
 
 In Proceedings of the 32nd Conference on Neural Information Processing Systems , NeurIPS’18 
 
 Montreal, Canada: Curran Associates, Inc., 2018 
 
 URL: https://arxiv.org/abs/1811.09720 
 

 
 [YHL23] 
 Wencong You, Zayd Hammoudeh and Daniel Lowd 
 
 “Large Language Models Are Better Adversaries: Exploring Generative Clean-Label Backdoor Attacks Against Text Classifiers” 
 
 In Findings of the Association for Computational Linguistics , EMNLP’23, 2023 
 

 
 [YGG17] 
 Yang You, Igor Gitman and Boris Ginsburg 
 
 “Large Batch Training of Convolutional Networks”, 2017 
 
 arXiv: 1708.03888 [cs.CV] 
 

 
 [Yua+07] 
 Ming Yuan, Ali Ekici, Zhaosong Lu and Renato Monteiro 
 
 “Dimension Reduction and Coefficient Estimation in Multivariate Linear Regression” 
 
 In Journal of the Royal Statistical Society: Series B (Statistical Methodology) 69.3 , 2007, pp. 329–346 
 

 
 [Zen+23] 
 Yingyan Zeng, Jiachen. Wang, Si Chen, Hoang Just, Ran Jin and Ruoxi Jia 
 
 “ModelPred: A Framework for Predicting Trained Model from Training Data” 
 
 In Proceedings of the 1st IEEE Conference on Secure and Trustworthy Machine Learning , SaTML’23, 2023 
 
 URL: https://arxiv.org/abs/2111.12545 
 

 
 [Zha+17] 
 Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht and Oriol Vinyals 
 
 “Understanding Deep Learning Requires Rethinking Generalization” 
 
 In Proceedings of the 5th International Conference on Learning Representations , ICLR’17, 2017 
 
 URL: https://arxiv.org/abs/1611.03530 
 

 
 [Zha+21] 
 Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht and Oriol Vinyals 
 
 “Understanding Deep Learning (Still) Requires Rethinking Generalization” 
 
 In Communications of the ACM 64.3 
 
 New York, NY, USA: Association for Computing Machinery, 2021, pp. 107–115 
 
 URL: https://dl.acm.org/doi/10.1145/3446776 
 

 
 [Zha+21a] 
 Chiyuan Zhang, Daphne Ippolito, Katherine Lee, Matthew Jagielski, Florian Tramèr and Nicholas Carlini 
 
 “Counterfactual Memorization in Neural Language Models”, 2021 
 
 arXiv: 2112.12938 [cs.CL] 
 

 
 [Zha+20] 
 Haoran Zhang, Amy. Lu, Mohamed Abdalla, Matthew McDermott and Marzyeh Ghassemi 
 
 “Hurtful Words: Quantifying Biases in Clinical Contextual Word Embeddings” 
 
 In Proceedings of the ACM Conference on Health, Inference, and Learning , CHIL’20 
 
 New York, NY, USA: Association for Computing Machinery, 2020 
 
 DOI: 10.1145/3368555.3384448 
 

 
 [ZZ22] 
 Rui Zhang and Shihua Zhang 
 
 “Rethinking Influence Functions of Neural Networks in the Over-Parameterized Regime” 
 
 In Proceedings of the 36th AAAI Conference on Artificial Intelligence , AAAI’22 
 
 Vancouver, Canada: Association for the Advancement of Artificial Intelligence, 2022 
 
 URL: https://arxiv.org/abs/2112.08297 
 

 
 [Zha+21b] 
 Wentao Zhang, Yexin Wang, Zhenbang You, Meng Cao, Ping Huang, Jiulong Shan, Zhi Yang and Bin Cui 
 
 “RIM: Reliable Influence-based Active Learning on Graphs” 
 
 In Proceedings of the 35th Conference on Neural Information Processing Systems , NeurIPS’21 
 
 Virtual Only: Curran Associates, Inc., 2021 
 
 URL: https://arxiv.org/abs/2110.14854 
 

 
 [Zho+19] 
 Jianlong Zhou, Zhidong Li, Huaiwen Hu, Kun Yu, Fang Chen, Zelin Li and Yang Wang 
 
 “Effects of Influence on User Trust in Predictive Decision Making” 
 
 In Extended Abstracts of the 2019 Conference on Human Factors in Computing Systems , CHI’19 
 
 New York, NY, USA: Association for Computing Machinery, 2019 
 
 DOI: 10.1145/3290607.3312962 
 

 
 
 
 

## References

 
 
 [Arn51a] 
 Walter Arnoldi 
 
 “The Principle of Minimized Iterations in the Solution of the Matrix Eigenvalue Problem” 
 
 In Quarterly of Applied Mathematics 9.1 
 
 Brown University, 1951, pp. 17–29 
 

 
 [Sha53a] 
 Lloyd. Shapley 
 
 “A Value for n-Person Games” 
 
 In Contributions to the Theory of Games II 
 
 Princeton, NJ USA: Princeton University Press, 1953, pp. 307–317 
 

 
 [Sri61a] 
 K.. Srikantan 
 
 “Testing for the Single Outlier in a Regression Model” 
 
 In Indian Journal of Statistics 23.3 
 
 Springer, 1961, pp. 251–260 
 

 
 [Hub64a] 
 Peter. Huber 
 
 “Robust Estimation of a Location Parameter” 
 
 In Annals of Mathematical Statistics 35.1 , 1964, pp. 73–101 
 

 
 [Ban65a] 
 John. Banzhaf 
 
 “Weighted Voting Doesn’t Work: A Mathematical Analysis” 
 
 In Rutgers Law Review 19.2 , 1965, pp. 317–343 
 

 
 [SC68a] 
 William. Snedecor and George. Cochran 
 
 “Statistical Methods” 
 
 Iowa State University Press, 1968 
 

 
 [Jae72a] 
 Louis. Jaeckel 
 
 “The Infinitesimal Jackknife”, 1972 
 

 
 [TMB73a] 
 G.. Tietjen, R.. Moore and R.. Beckman 
 
 “Testing for a Single Outlier in Simple Linear Regression” 
 
 In Technometrics 15.4 
 
 Taylor Francis, 1973, pp. 717–721 
 

 
 [BT74a] 
 Albert. Beaton and John. Tukey 
 
 “The Fitting of Power Series, Meaning Polynomials, Illustrated on Band-Spectroscopic Data” 
 
 In Technometrics 16.2 
 
 Taylor Francis, 1974, pp. 147–185 
 

 
 [Ham74a] 
 Frank. Hampel 
 
 “The Influence Curve and its Role in Robust Estimation” 
 
 In Journal of the American Statistical Association 69.346 
 
 Taylor Francis, 1974, pp. 383–393 
 

 
 [Dub75a] 
 Pradeep Dubey 
 
 “On the Uniqueness of the Shapley Value” 
 
 In International Journal of Game Theory 4.3 
 
 DEU: Physica-Verlag GmbH, 1975, pp. 131–139 
 

 
 [Ell76a] 
 Jonas. Ellenberg 
 
 “Testing for a Single Outlier from a General Linear Regression” 
 
 In Biometrics 32.3 
 
 [Wiley, International Biometric Society], 1976, pp. 637–645 
 

 
 [Coo77a] 
 R. Cook 
 
 “Detection of Influential Observation in Linear Regression” 
 
 In Technometrics 19.1 
 
 American Statistical Association, 1977, pp. 15–18 
 

 
 [Mah77a] 
 Michael. Mahoney 
 
 “Publication Prejudices: An Experimental Study of Confirmatory Bias in the Peer Review System” 
 
 In Cognitive Therapy and Research 1.2 , 1977, pp. 161–175 
 

 
 [JW78a] 
 John. Jr. and Roy. Welsch 
 
 “Techniques for nonlinear least squares and robust regression” 
 
 In Communications in Statistics - Simulation and Computation 7.4 
 
 Taylor Francis, 1978, pp. 345–359 
 

 
 [Hog79a] 
 Robert. Hogg 
 
 “Statistical Robustness: One View of its Use in Applications Today” 
 
 In The American Statistician 33.3 
 
 Taylor Francis, 1979, pp. 108–115 
 

 
 [DNW81a] 
 Pradeep Dubey, Abraham Neyman and Robert Weber 
 
 “Value Theory without Efficiency” 
 
 In Mathematics of Operations Research 6.1 
 
 INFORMS, 1981, pp. 122–128 
 

 
 [Hub81a] 
 Peter Huber 
 
 “Robust Statistics” 
 
 John Wiley Sons, 1981 
 

 
 [CHW82a] 
 R. Cook, Norton Holschuh and Sanford Weisberg 
 
 “A Note on an Alternative Outlier Model” 
 
 In Journal of the Royal Statistical Society. Series B (Methodological) 44.3 
 
 Wiley, 1982, pp. 370–376 
 

 
 [CW82a] 
 R. Cook and Sanford Weisberg 
 
 “Residuals and Influence in Regression” 
 
 New York: ChapmanHall, 1982 
 

 
 [JL84a] 
 William. Johnson and Joram Lindenstrauss 
 
 “Extensions of Lipschitz Mappings into a Hilbert Space” 
 
 In Contemporary Mathematics 26 
 
 American Mathematical Society, 1984, pp. 189–206 
 

 
 [RHW86a] 
 David. Rumelhart, Geoffrey. Hinton and Ronald. Williams 
 
 “Learning Representations by Back-Propagating Errors” 
 
 In Nature 323.6088 , 1986, pp. 533–536 
 

 
 [RL87a] 
 Peter. Rousseeuw and Annick.. Leroy 
 
 “Robust Regression and Outlier Detection” 
 
 USA: John Wiley Sons, Inc., 1987 
 

 
 [SR88a] 
 Lloyd. Shapley and Alvin. Roth 
 
 “The Shapley Value: Essays in Honor of Lloyd S. Shapley” 
 
 Cambridge University Press, 1988 
 

 
 [Lec89a] 
 Yvan. Leclerc 
 
 “Constructing Simple Stable Descriptions for Image Partitioning” 
 
 In International Journal of Computer Vision 3.1 
 
 Springer ScienceBusiness Media LLC, 1989, pp. 73–102 
 

 
 [AKA91a] 
 David. Aha, Dennis Kibler and Marc. Albert 
 
 “Instance-Based Learning Algorithms” 
 
 In Machine Learning 6.1 
 
 USA: Kluwer Academic Publishers, 1991, pp. 37–66 
 
 URL: https://link.springer.com/article/10.1007/bf00153759 
 

 
 [BR92a] 
 Avrim. Blum and Ronald. Rivest 
 
 “Training a 3-Node Neural Network is NP-Complete” 
 
 In Neural Networks 5.1 , 1992, pp. 117–127 
 

 
 [DP94a] 
 Xiaotie Deng and Christos. Papadimitriou 
 
 “On the Complexity of Cooperative Solution Concepts” 
 
 In Mathematics of Operations Research 19.2 
 
 Linthicum, MD, USA: INFORMS, 1994, pp. 257–266 
 

 
 [Pea94a] 
 Barak. Pearlmutter 
 
 “Fast Exact Multiplication by the Hessian” 
 
 In Neural Computation 6 , 1994, pp. 147–160 
 

 
 [Rou94a] 
 Peter Rousseeuw 
 
 “Least Median of Squares Regression” 
 
 In Journal of the American Statistical Association 79.388 
 
 Taylor Francis, 1994 
 

 
 [Tib96a] 
 Robert Tibshirani 
 
 “Regression Shrinkage and Selection via the Lasso” 
 
 In Journal of the Royal Statistical Society (Series B) 58 , 1996, pp. 267–288 
 

 
 [LeC+98a] 
 Yann LeCun, Léon Bottou, Yoshua Bengio and Patrick Haffner 
 
 “Gradient-Based Learning Applied to Document Recognition” 
 
 In Proceedings of the IEEE 86 , 1998, pp. 2278–2324 
 

 
 [GR99a] 
 Michel Grabisch and Marc Roubens 
 
 “An Axiomatic Approach to the Concept of Interaction Among Players in Cooperative Games” 
 
 In International Journal of Game Theory 28.4 , 1999, pp. 547–565 
 
 DOI: 10.1007/s001820050125 
 

 
 [Qia99a] 
 Ning Qian 
 
 “On the Momentum Term in Gradient Descent Learning Algorithms” 
 
 In Neural Networks 12.1 , 1999, pp. 145–151 
 

 
 [SHS01a] 
 Bernhard Schölkopf, Ralf Herbrich and Alex. Smola 
 
 “A Generalized Representer Theorem” 
 
 In Proceedings of the 14th Annual Conference on Computational Learning Theory and 5th European Conference on Computational Learning Theory , COLT’01/EuroCOLT’01 
 
 Berlin, Heidelberg: Springer-Verlag, 2001, pp. 416–426 
 

 
 [BV04a] 
 Stephen Boyd and Lieven Vandenberghe 
 
 “Convex Optimization” 
 
 Cambridge University Press, 2004 
 

 
 [Yua+07a] 
 Ming Yuan, Ali Ekici, Zhaosong Lu and Renato Monteiro 
 
 “Dimension Reduction and Coefficient Estimation in Multivariate Linear Regression” 
 
 In Journal of the Royal Statistical Society: Series B (Statistical Methodology) 69.3 , 2007, pp. 329–346 
 

 
 [HSS08a] 
 Thomas Hofmann, Bernhard Schölkopf and Alexander. Smola 
 
 “Kernel Methods in Machine Learning” 
 
 In Annals of Statistics 36.3 , 2008, pp. 1171–1220 
 
 URL: https://arxiv.org/abs/math/0701907 
 

 
 [Den+09a] 
 Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li and Li Fei-Fei 
 
 “ImageNet: A Large-Scale Hierarchical Image Database” 
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , CVPR’09, 2009, pp. 248–255 
 

 
 [LDA09a] 
 Brian. Lim, Anind. Dey and Daniel Avrahami 
 
 “Why and Why Not Explanations Improve the Intelligibility of Context-Aware Intelligent Systems” 
 
 In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems , CHI’09 
 
 Boston, MA, USA: Association for Computing Machinery, 2009, pp. 2119–2128 
 

 
 [RHS09a] 
 Alexander Renkl, Tatjana Hilbert and Silke Schworm 
 
 “Example-Based Learning in Heuristic Domains: A Cognitive Load Theory Account” 
 
 In Educational Psychology Review 21.1 , 2009, pp. 67–78 
 

 
 [DHS11a] 
 John Duchi, Elad Hazan and Yoram Singer 
 
 “Adaptive Subgradient Methods for Online Learning and Stochastic Optimization” 
 
 In Journal of Machine Learning Research 12 
 
 JMLR.org, 2011, pp. 2121–2159 
 
 URL: https://jmlr.org/papers/v12/duchi11a.html 
 

 
 [Rec11a] 
 Benjamin Recht 
 
 “A Simpler Approach to Matrix Completion” 
 
 In Journal of Machine Learning Research 12 , 2011, pp. 3413–3430 
 
 URL: https://arxiv.org/abs/0910.0651 
 

 
 [BNL12a] 
 Battista Biggio, Blaine Nelson and Pavel Laskov 
 
 “Poisoning Attacks against Support Vector Machines” 
 
 In Proceedings of the 29th International Conference on Machine Learning , ICML’12 
 
 Edinburgh, Great Britain: PMLR, 2012 
 
 URL: https://arxiv.org/abs/1206.6389 
 

 
 [Dwo+12a] 
 Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold and Richard Zemel 
 
 “Fairness through Awareness” 
 
 In Proceedings of the 3rd Innovations in Theoretical Computer Science Conference , ITCS’12, 2012 
 
 URL: https://arxiv.org/abs/1104.3913 
 

 
 [KSH12a] 
 Alex Krizhevsky, Ilya Sutskever and Geoffrey Hinton 
 
 “ImageNet Classification with Deep Convolutional Neural Networks” 
 
 In Proceedings of the 25th Conference on Neural Information Processing Systems , NeurIPS’12, 2012, pp. 1097–1105 
 

 
 [Neg+12a] 
 Sahand. Negahban, Pradeep Ravikumar, Martin. Wainwright and Bin Yu 
 
 “A Unified Framework for High-Dimensional Analysis of M M -Estimators with Decomposable Regularizers” 
 
 In Statistical Science 27.4 
 
 Institute of Mathematical Statistics, 2012, pp. 538–557 
 
 URL: https://arxiv.org/abs/1010.2731 
 

 
 [KDK13a] 
 Saul. Kassin, Itiel. Dror and Jeff Kukucka 
 
 “The Forensic Confirmation Bias: Problems, Perspectives, and Proposed Solutions” 
 
 In Journal of Applied Research in Memory and Cognition 2.1 , 2013, pp. 42–52 
 

 
 [NDM13a] 
 Sherry Nakhaeizadeh, Itiel Dror and Ruth Morgan 
 
 “Cognitive Bias in Forensic Anthropology: Visual Assessment of Skeletal Remains is Susceptible to Confirmation Bias” 
 
 In Science Justice 54.3 , 2013, pp. 208–214 
 

 
 [DG14a] 
 Alex Davies and Zoubin Ghahramani 
 
 “The Random Forest Kernel and Other Kernels for Big Data from Random Partitions”, 2014 
 
 arXiv: 1402.4293 [cs.LG] 
 

 
 [He+14a] 
 Xinran He, Junfeng Pan, Ou Jin, Tianbing Xu, Bo Liu, Tao Xu, Yanxin Shi, Antoine Atallah, Ralf Herbrich, Stuart Bowers and Joaquinñonero Candela 
 
 “Practical Lessons from Predicting Clicks on Ads at Facebook” 
 
 In Proceedings of the Eighth International Workshop on Data Mining for Online Advertising , AdKDD’14 
 
 New York, NY, USA: Association for Computing Machinery, 2014 
 
 URL: https://research.facebook.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/ 
 

 
 [KW14a] 
 Diederik. Kingma and Max Welling 
 
 “Auto-Encoding Variational Bayes” 
 
 In Proceedings of the 2nd International Conference on Learning Representations , ICLR’14, 2014 
 
 URL: https://arxiv.org/abs/1312.6114 
 

 
 [KNH14a] 
 Alex Krizhevsky, Vinod Nair and Geoffrey Hinton 
 
 “The CIFAR-10 Dataset”, 2014 
 

 
 [Ren14a] 
 Alexander Renkl 
 
 “Toward an Instructionally Oriented Theory of Example-Based Learning” 
 
 In Cognitive Science 38.1 , 2014, pp. 1–37 
 
 URL: https://onlinelibrary.wiley.com/doi/full/10.1111/cogs.12086 
 

 
 [RMW14a] 
 Danilo Rezende, Shakir Mohamed and Daan Wierstra 
 
 “Stochastic Backpropagation and Approximate Inference in Deep Generative Models” 
 
 In Proceedings of the 31st International Conference on International Conference on Machine Learning , ICML’14, 2014 
 
 URL: https://arxiv.org/abs/1401.4082 
 

 
 [Woo14a] 
 David. Woodruff 
 
 “Sketching as a Tool for Numerical Linear Algebra” 
 
 In Foundations and Trends in Theoretical Computer Science 10.1–2 
 
 Hanover, MA, USA: Now Publishers Inc., 2014, pp. 1–157 
 
 URL: https://arxiv.org/abs/1411.4357 
 

 
 [KB15a] 
 Diederik. Kingma and Jimmy Ba 
 
 “Adam: A Method for Stochastic Optimization” 
 
 In Proceedings of the 3rd International Conference on Learning Representations , ICLR’15, 2015 
 
 URL: https://arxiv.org/abs/1412.6980 
 

 
 [WIB15a] 
 Kai Wei, Rishabh Iyer and Jeff Bilmes 
 
 “Submodularity in Data Subset Selection and Active Learning” 
 
 In Proceedings of the 32nd International Conference on Machine Learning , ICML’15 
 
 Lille, France: PMLR, 2015 
 
 URL: https://proceedings.mlr.press/v37/wei15.html 
 

 
 [Ang+16a] 
 Julia Angwin, Jeff Larson, Surya Mattu and Lauren Kirchner 
 
 “Machine Bias” 
 
 In ProPublica , 2016 
 
 URL: https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing 
 

 
 [Kiz16a] 
 René. Kizilcec 
 
 “How Much Information? Effects of Transparency on Trust in an Algorithmic Interface” 
 
 In Proceedings of the 2016 CHI Conference on Human Factors in Computing Systems , CHI’16 
 
 San Jose, California, USA: Association for Computing Machinery, 2016, pp. 2390–2395 
 

 
 [Kri+16a] 
 Sanjay Krishnan, Jiannan Wang, Eugene Wu, Michael. Franklin and Ken Goldberg 
 
 “ActiveClean: Interactive Data Cleaning for Statistical Modeling” 
 
 In Proceedings of the VLDB Endowment 
 
 VLDB Endowment, 2016 
 
 URL: https://www.vldb.org/pvldb/vol9/p948-krishnan.pdf 
 

 
 [Woj+16a] 
 Mike Wojnowicz, Ben Cruz, Xuan Zhao, Brian Wallace, Matt Wolff, Jay Luan and Caleb Crable 
 
 “‘Influence Sketching’: Finding Influential Samples in Large-Scale Regressions” 
 
 In Proceedings of the 2016 IEEE International Conference on Big Data , BigData’16 
 
 IEEE, 2016 
 
 URL: https://arxiv.org/abs/1611.05923 
 

 
 [ABH17a] 
 Naman Agarwal, Brian Bullins and Elad Hazan 
 
 “Second-Order Stochastic Optimization for Machine Learning in Linear Time” 
 
 In Journal of Machine Learning Research 18.1 
 
 JMLR.org, 2017, pp. 4148–4187 
 
 URL: https://arxiv.org/abs/1602.03943 
 

 
 [Arp+17a] 
 Devansh Arpit, Stanisław Jastrzebski, Nicolas Ballas, David Krueger, Emmanuel Bengio, Maxinder. Kanwal, Tegan Maharaj, Asja Fischer, Aaron Courville, Yoshua Bengio and Simon Lacoste-Julien 
 
 “A Closer Look at Memorization in Deep Networks” 
 
 In Proceedings of the 34th International Conference on Machine Learning , ICML’17, 2017 
 
 URL: https://arxiv.org/abs/1706.05394 
 

 
 [BLK17a] 
 Olivier Bachem, Mario Lucic and Andreas Krause 
 
 “Practical Coreset Constructions for Machine Learning”, 2017 
 
 arXiv: 1703.06476 [stat.ML] 
 

 
 [Che+17a] 
 Xinyun Chen, Chang Liu, Bo Li, Kimberly Lu and Dawn Song 
 
 “Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning”, 2017 
 
 arXiv: 1712.05526 [cs.CR] 
 

 
 [EGH17a] 
 Rajmadhan Ekambaram, Dmitry. Goldgof and Lawrence. Hall 
 
 “Finding Label Noise Examples in Large Scale Datasets” 
 
 In Proceedings of the 2017 IEEE International Conference on Systems, Man, and Cybernetics , SMC’17, 2017 
 
 DOI: 10.1109/SMC.2017.8122985 
 

 
 [Goy+17a] 
 Priya Goyal, Piotr Dollár, Ross. Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia and Kaiming He 
 
 “Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour”, 2017 
 
 arXiv: 1706.02677 [cs.CV] 
 

 
 [Hig+17a] 
 Irina Higgins, Loïc Matthey, Arka Pal, Christopher. Burgess, Xavier Glorot, Matthew. Botvinick, Shakir Mohamed and Alexander Lerchner 
 
 “beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework” 
 
 In Proceedings of the 5th International Conference on Learning Representations , ICLR’17, 2017 
 
 URL: https://openreview.net/forum?id=Sy2fzU9gl 
 

 
 [Kni17a] 
 Will Knight 
 
 “The Dark Secret at the Heart of AI” 
 
 In MIT Technology Review , 2017 
 
 URL: https://www.technologyreview.com/2017/04/11/5113/the-dark-secret-at-the-heart-of-ai/ 
 

 
 [KL17a] 
 Pang Koh and Percy Liang 
 
 “Understanding Black-box Predictions via Influence Functions” 
 
 In Proceedings of the 34th International Conference on Machine Learning , ICML’17 
 
 Sydney, Australia: PMLR, 2017 
 
 URL: https://arxiv.org/abs/1703.04730 
 

 
 [LL17a] 
 Scott. Lundberg and Su-In Lee 
 
 “A Unified Approach to Interpreting Model Predictions” 
 
 In Proceedings of the 31st International Conference on Neural Information Processing Systems , NeurIPS’17, 2017 
 
 URL: https://arxiv.org/abs/1705.07874 
 

 
 [SKL17a] 
 Jacob Steinhardt, Pang Koh and Percy Liang 
 
 “Certified Defenses for Data Poisoning Attacks” 
 
 In Proceedings of the 31st Conference on Neural Information Processing Systems , NeurIPS’17 
 
 Long Beach, California, USA: Curran Associates, Inc., 2017 
 
 URL: https://arxiv.org/abs/1706.03691 
 

 
 [Yan+17a] 
 Jian Yang, Lei Luo, Jianjun Qian, Ying Tai, Fanlong Zhang and Yong Xu 
 
 “Nuclear Norm Based Matrix Regression with Applications to Face Recognition with Occlusion and Illumination Changes” 
 
 In IEEE Transactions on Pattern Analysis and Machine Intelligence 39.1 , 2017, pp. 156–171 
 
 DOI: 10.1109/TPAMI.2016.2535218 
 

 
 [YGG17a] 
 Yang You, Igor Gitman and Boris Ginsburg 
 
 “Large Batch Training of Convolutional Networks”, 2017 
 
 arXiv: 1708.03888 [cs.CV] 
 

 
 [Zha+17a] 
 Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht and Oriol Vinyals 
 
 “Understanding Deep Learning Requires Rethinking Generalization” 
 
 In Proceedings of the 5th International Conference on Learning Representations , ICLR’17, 2017 
 
 URL: https://arxiv.org/abs/1611.03530 
 

 
 [Awa+18a] 
 Edmond Awad, Sohan Dsouza, Richard Kim, Jonathan Schulz, Joseph Henrich, Azim Shariff, Jean-François Bonnefon and Iyad Rahwan 
 
 “The Moral Machine Experiment” 
 
 In Nature 563.7729 , 2018, pp. 59–64 
 

 
 [Lip18a] 
 Zachary. Lipton 
 
 “The Mythos of Model Interpretability: In Machine Learning, the Concept of Interpretability is Both Important and Slippery.” 
 
 In Queue 16.3 
 
 New York, NY, USA: Association for Computing Machinery, 2018, pp. 31–57 
 
 URL: https://dl.acm.org/doi/10.1145/3236386.3241340 
 

 
 [LDG18a] 
 Kang Liu, Brendan Dolan-Gavitt and Siddharth Garg 
 
 “Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks” 
 
 In Proceedings of the International Symposium on Research in Attacks, Intrusions, and Defenses , RAID’18 
 
 Heraklion, Crete, Greece: Springer, 2018, pp. 273–294 
 
 URL: https://arxiv.org/abs/1805.12185 
 

 
 [Sha+18b] 
 Ali Shafahi, W. Huang, Mahyar Najibi, Octavian Suciu, Christoph Studer, Tudor Dumitras and Tom Goldstein 
 
 “Poison Frogs! Targeted Clean-Label Poisoning Attacks on Neural Networks” 
 
 In Proceedings of the 32nd Conference on Neural Information Processing Systems , NeurIPS’18 
 
 Montreal, Canada: Curran Associates, Inc., 2018 
 
 URL: https://arxiv.org/abs/1804.00792 
 

 
 [Sha+18c] 
 Boris Sharchilev, Yury Ustinovskiy, Pavel Serdyukov and Maarten de Rijke 
 
 “Finding Influential Training Samples for Gradient Boosted Decision Trees” 
 
 In Proceedings of the 35th International Conference on Machine Learning , ICML’18 
 
 PMLR, 2018, pp. 4577–4585 
 
 URL: https://arxiv.org/abs/1802.06640 
 

 
 [TB18a] 
 Daniel Ting and Eric Brochu 
 
 “Optimal Subsampling with Influence Functions” 
 
 In Proceedings of the 32nd Conference on Neural Information Processing Systems , NeurIPS’18 
 
 Curran Associates, Inc., 2018 
 
 URL: https://arxiv.org/abs/1709.01716 
 

 
 [Yeh+18a] 
 Chih“=/Kuan Yeh, Joon Kim, Ian.H. Yen and Pradeep Ravikumar 
 
 “Representer Point Selection for Explaining Deep Neural Networks” 
 
 In Proceedings of the 32nd Conference on Neural Information Processing Systems , NeurIPS’18 
 
 Montreal, Canada: Curran Associates, Inc., 2018 
 
 URL: https://arxiv.org/abs/1811.09720 
 

 
 [BCC19a] 
 Christine Basta, Marta. Costa-jussà and Noe Casas 
 
 “Evaluating the Underlying Gender Bias in Contextualized Word Embeddings” 
 
 In Proceedings of the First Workshop on Gender Bias in Natural Language Processing 
 
 Florence, Italy: Association for Computational Linguistics, 2019 
 
 URL: https://arxiv.org/abs/1904.08783 
 

 
 [CJH19a] 
 Carrie. Cai, Jonas Jongejan and Jess Holbrook 
 
 “The Effects of Example-Based Explanations in a Machine Learning Interface” 
 
 In Proceedings of the 24th International Conference on Intelligent User Interfaces , IUI’19, 2019, pp. 258–262 
 
 URL: https://dl.acm.org/doi/10.1145/3301275.3302289 
 

 
 [Dem+19a] 
 Ambra Demontis, Marco Melis, Maura Pintor, Matthew Jagielski, Battista Biggio, Alina Oprea, Cristina Nita-Rotaru and Fabio Roli 
 
 “Why Do Adversarial Attacks Transfer? Explaining Transferability of Evasion and Poisoning Attacks” 
 
 In Proceedings of the 28th USENIX Security Symposium , USENIX’19 
 
 Renton, Washington, USA, 2019 
 
 URL: https://arxiv.org/abs/1809.02861 
 

 
 [Dev+19a] 
 Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova 
 
 “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding” 
 
 In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics , ACL’19 
 
 Minneapolis, Minnesota: Association for Computational Linguistics, 2019 
 
 URL: https://arxiv.org/abs/1810.04805 
 

 
 [GZ19a] 
 Amirata Ghorbani and James Zou 
 
 “Data Shapley: Equitable Valuation of Data for Machine Learning” 
 
 In Proceedings of the 36th International Conference on Machine Learning , ICML’19, 2019 
 
 URL: https://proceedings.mlr.press/v97/ghorbani19c.html 
 

 
 [GH19a] 
 Bruce Glymour and Jonathan Herington 
 
 “Measuring the Biases That Matter: The Ethical and Casual Foundations for Measures of Fairness in Algorithms” 
 
 In Proceedings of the 2019 ACM Conference on Fairness, Accountability, and Transparency , FAccT’19, 2019 
 

 
 [HNM19a] 
 Satoshi Hara, Atsushi Nitanda and Takanori Maehara 
 
 “Data Cleansing for Models Trained with SGD” 
 
 In Proceedings of the 33rd Conference on Neural Information Processing Systems , NeurIPS’19 
 
 Vancouver, Canada: Curran Associates, Inc., 2019 
 
 URL: https://arxiv.org/abs/1906.08473 
 

 
 [Jia+19b] 
 Ruoxi Jia, David Dao, Boxin Wang, Frances Hubis, Nezihe Gürel, Bo Li, Ce Zhang, Costas. Spanos and Dawn Song 
 
 “Efficient Task-Specific Data Valuation for Nearest Neighbor Algorithms” 
 
 In Proceedings of the VLDB Endowment , PVLDB’19, 2019 
 
 URL: https://arxiv.org/abs/1908.08619 
 

 
 [Jia+19c] 
 Ruoxi Jia, David Dao, Boxin Wang, Frances Hubis, Nick Hynes, Nezihe Gürel, Bo Li, Ce Zhang, Dawn Song and Costas. Spanos 
 
 “Towards Efficient Data Valuation Based on the Shapley Value” 
 
 In Proceedings of the 22nd Conference on Artificial Intelligence and Statistics , AISTATS’19, 2019, pp. 1167–1176 
 
 URL: https://arxiv.org/abs/1902.10275 
 

 
 [Kha+19a] 
 Rajiv Khanna, Been Kim, Joydeep Ghosh and Oluwasanmi Koyejo 
 
 “Interpreting Black Box Predictions using Fisher Kernels” 
 
 In Proceedings of the 22nd Conference on Artificial Intelligence and Statistics , AISTATS’19, 2019 
 
 URL: https://arxiv.org/abs/1810.10118 
 

 
 [Koh+19a] 
 Pang Koh, Kai-Siang Ang, Hubert.. Teo and Percy Liang 
 
 “On the Accuracy of Influence Functions for Measuring Group Effects” 
 
 In Proceedings of the 33rd International Conference on Neural Information Processing Systems , NeurIPS’19 
 
 Red Hook, NY, USA: Curran Associates Inc., 2019 
 
 URL: https://arxiv.org/abs/1905.13289 
 

 
 [KW19a] 
 Sanjay Krishnan and Eugene Wu 
 
 “AlphaClean: Automatic Generation of Data Cleaning Pipelines”, 2019 
 
 arXiv: 1904.11827 [cs.DB] 
 

 
 [Kur+19a] 
 Keita Kurita, Nidhi Vyas, Ayush Pareek, Alan Black and Yulia Tsvetkov 
 
 “Measuring Bias in Contextualized Word Representations” 
 
 In Proceedings of the First Workshop on Gender Bias in Natural Language Processing 
 
 Association for Computational Linguistics, 2019 
 
 URL: https://arxiv.org/abs/1906.07337 
 

 
 [Rud19a] 
 Cynthia Rudin 
 
 “Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead” 
 
 In Nature Machine Intelligence 1.5 , 2019, pp. 206–215 
 
 URL: https://arxiv.org/abs/1811.10154 
 

 
 [Sax+19a] 
 Nripsuta Saxena, Karen Huang, Evan DeFilippis, Goran Radanovic, David. Parkes and Yang Liu 
 
 “How Do Fairness Definitions Fare? Examining Public Attitudes Towards Algorithmic Definitions of Fairness” 
 
 In Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society , AIES’19, 2019 
 
 URL: https://arxiv.org/abs/1811.03654 
 

 
 [TC19a] 
 Yi Tan and L. Celis 
 
 “Assessing Social and Intersectional Biases in Contextualized Word Representations” 
 
 In Proceedings of the 33rd Conference on Neural Information Processing Systems , NeurIPS’19 
 
 Vancouver, Canada: Curran Associates, Inc., 2019 
 
 URL: https://arxiv.org/abs/1911.01485 
 

 
 [Wan+19a] 
 Bolun Wang, Yuanshun Yao, Shawn Shan, Huiying Li, Bimal Viswanath, Haitao Zheng and Ben. Zhao 
 
 “Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks” 
 
 In Proceedings of the 40th IEEE Symposium on Security and Privacy , SP’19, 2019 
 
 URL: https://ieeexplore.ieee.org/document/8835365 
 

 
 [Zho+19a] 
 Jianlong Zhou, Zhidong Li, Huaiwen Hu, Kun Yu, Fang Chen, Zelin Li and Yang Wang 
 
 “Effects of Influence on User Trust in Predictive Decision Making” 
 
 In Extended Abstracts of the 2019 Conference on Human Factors in Computing Systems , CHI’19 
 
 New York, NY, USA: Association for Computing Machinery, 2019 
 
 DOI: 10.1145/3290607.3312962 
 

 
 [BBD20a] 
 Elnaz Barshan, Marc-Etienne Brunet and Gintare Dziugaite 
 
 “RelatIF: Identifying Explanatory Training Samples via Relative Influence” 
 
 In Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics , AISTATS’20, 2020 
 
 URL: https://arxiv.org/abs/2003.11630 
 

 
 [Bar+20a] 
 Peter. Bartlett, Philip. Long, Gábor Lugosi and Alexander Tsigler 
 
 “Benign Overfitting in Linear Regression” 
 
 In Proceedings of the National Academy of Sciences 117.48 , 2020, pp. 30063–30070 
 
 URL: https://arxiv.org/abs/1906.11300 
 

 
 [BYF20a] 
 Samyadeep Basu, Xuchen You and Soheil Feizi 
 
 “On Second-Order Group Influence Functions for Black-Box Predictions” 
 
 In Proceedings of the 37th International Conference on Machine Learning , ICML’20 
 
 Virtual Only: PMLR, 2020 
 
 URL: https://arxiv.org/abs/1911.00418 
 

 
 [BMK20a] 
 Zalán Borsos, Mojmir Mutny and Andreas Krause 
 
 “Coresets via Bilevel Optimization for Continual Learning and Streaming” 
 
 In Proceedings of the 34th Conference on Neural Information Processing Systems , NeurIPS’20, 2020 
 
 URL: https://arxiv.org/abs/2006.03875 
 

 
 [Bro+20a] 
 Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever and Dario Amodei 
 
 “Language Models are Few-Shot Learners” 
 
 In Proceedings of the 34th Conference on Neural Information Processing Systems , NeurIPS’20 
 
 Curran Associates, Inc., 2020 
 
 URL: https://arxiv.org/abs/2005.14165 
 

 
 [CSG20a] 
 Gilad Cohen, Guillermo Sapiro and Raja Giryes 
 
 “Detecting Adversarial Samples Using Influence Functions and Nearest Neighbors” 
 
 In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , CVPR’20, 2020 
 
 URL: https://arxiv.org/abs/1909.06872 
 

 
 [DAm+20a] 
 Alexander D’Amour, Katherine. Heller, Dan Moldovan, Ben Adlam, Babak Alipanahi, Alex Beutel, Christina Chen, Jonathan Deaton, Jacob Eisenstein, Matthew. Hoffman, Farhad Hormozdiari, Neil Houlsby, Shaobo Hou, Ghassen Jerfel, Alan Karthikesalingam, Mario Lucic, Yi-An Ma, Cory. McLean, Diana Mincu, Akinori Mitani, Andrea Montanari, Zachary Nado, Vivek Natarajan, Christopher Nielson, Thomas. Osborne, Rajiv Raman, Kim Ramasamy, Rory Sayres, Jessica Schrouff, Martin Seneviratne, Shannon Sequeira, Harini Suresh, Victor Veitch, Max Vladymyrov, Xuezhi Wang, Kellie Webster, Steve Yadlowsky, Taedong Yun, Xiaohua Zhai and D. Sculley 
 
 “Underspecification Presents Challenges for Credibility in Modern Machine Learning”, 2020 
 
 arXiv: 2011.03395 [cs.LG] 
 

 
 [FGL20a] 
 Minghong Fang, Neil Gong and Jia Liu 
 
 “Influence Function based Data Poisoning Attacks to Top-N Recommender Systems” 
 
 In Proceedings of the Web Conference 2020 , WWW’20, 2020 
 
 URL: https://arxiv.org/abs/2002.08025 
 

 
 [Fel20b] 
 Dan Feldman 
 
 “Introduction to Core-sets: an Updated Survey”, 2020 
 
 arXiv: 2011.09384 [cs.LG] 
 

 
 [Fel20c] 
 Vitaly Feldman 
 
 “Does Learning Require Memorization? A Short Tale about a Long Tail” 
 
 In Proceedings of the 52nd Annual ACM SIGACT Symposium on Theory of Computing , STOC’20, 2020 
 
 URL: https://arxiv.org/abs/1906.05271 
 

 
 [FZ20a] 
 Vitaly Feldman and Chiyuan Zhang 
 
 “What Neural Networks Memorize and Why: Discovering the Long Tail via Influence Estimation” 
 
 In Proceedings of the 34th Conference on Neural Information Processing Systems , NeurIPS’20 
 
 Virtual Only: Curran Associates, Inc., 2020 
 
 URL: https://arxiv.org/abs/2008.03703 
 

 
 [GZ20a] 
 Amirata Ghorbani and James. Zou 
 
 “Neuron Shapley: Discovering the Responsible Neurons” 
 
 In Proceedings of the 34th Conference on Neural Information Processing Systems , NeurIPS’20, 2020 
 
 URL: https://arxiv.org/abs/2002.09815 
 

 
 [Guo+20a] 
 Chuan Guo, Tom Goldstein, Awni. Hannun and Laurens van Maaten 
 
 “Certified Data Removal from Machine Learning Models” 
 
 In Proceedings of the 37th International Conference on Machine Learning 119 , ICML’20, 2020, pp. 3832–3842 
 
 URL: https://arxiv.org/abs/1911.03030 
 

 
 [Hut+20a] 
 Ben Hutchinson, Vinodkumar Prabhakaran, Emily Denton, Kellie Webster, Yu Zhong and Stephen Denuyl 
 
 “Social Biases in NLP Models as Barriers for Persons with Disabilities” 
 
 In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics 
 
 Association for Computational Linguistics, 2020 
 
 DOI: 10.18653/v1/2020.acl-main.487 
 

 
 [Kob+20a] 
 Sosuke Kobayashi, Sho Yokoi, Jun Suzuki and Kentaro Inui 
 
 “Efficient Estimation of Influence of a Training Instance” 
 
 In Proceedings of SustaiNLP: Workshop on Simple and Efficient Natural Language Processing 
 
 Online: Association for Computational Linguistics, 2020 
 
 URL: https://arxiv.org/abs/2012.04207 
 

 
 [Lee+20a] 
 Donghoon Lee, Hyunsin Park, Trung Pham and Chang. Yoo 
 
 “Learning Augmentation Network via Influence Functions” 
 
 In Proceedings of the 33rd Conference on Computer Vision and Pattern Recognition , CVPR’20, 2020 
 

 
 [MBL20a] 
 Baharan Mirzasoleiman, Jeff Bilmes and Jure Leskovec 
 
 “Coresets for Data-Efficient Training of Machine Learning Models” 
 
 In Proceedings of the 37th International Conference on Machine Learning , ICML’20, 2020 
 
 URL: https://arxiv.org/abs/1906.01827 
 

 
 [Ple+20a] 
 Geoff Pleiss, Tianyi Zhang, Ethan Elenberg and Kilian. Weinberger 
 
 “Identifying Mislabeled Data Using the Area Under the Margin Ranking” 
 
 In Proceedings of the 34th International Conference on Neural Information Processing Systems , NeurIPS’20 
 
 Red Hook, NY, USA: Curran Associates Inc., 2020 
 
 URL: https://arxiv.org/abs/2001.10528 
 

 
 [Pru+20a] 
 Garima Pruthi, Frederick Liu, Satyen Kale and Mukund Sundararajan 
 
 “Estimating Training Data Influence by Tracing Gradient Descent” 
 
 In Proceedings of the 34th Conference on Neural Information Processing Systems , NeurIPS’20 
 
 Virtual Only: Curran Associates, Inc., 2020 
 
 URL: https://arxiv.org/abs/2002.08484 
 

 
 [SGM20a] 
 Emma Strubell, Ananya Ganesh and Andrew McCallum 
 
 “Energy and Policy Considerations for Modern Deep Learning Research” 
 
 In Proceedings of the 34th AAAI Conference on Artificial Intelligence , AAAI’20, 2020 
 
 URL: https://arxiv.org/abs/1906.02243 
 

 
 [SDA20a] 
 Mukund Sundararajan, Kedar Dhamdhere and Ashish Agarwal 
 
 “The Shapley Taylor Interaction Index” 
 
 In Proceedings of the 37th International Conference on Machine Learning , ICML’20, 2020 
 
 URL: http://proceedings.mlr.press/v119/sundararajan20a 
 

 
 [SN20a] 
 Mukund Sundararajan and Amir Najmi 
 
 “The Many Shapley Values for Model Explanation” 
 
 In Proceedings of the 37th International Conference on Machine Learning , ICML’20, 2020, pp. 9269–9278 
 
 URL: https://arxiv.org/abs/1908.08474 
 

 
 [Wan+20a] 
 Zifeng Wang, Hong Zhu, Zhenhua Dong, Xiuqiang He and Shao-Lun Huang 
 
 “Less Is Better: Unweighted Data Subsampling via Influence Function” 
 
 In Proceedings of the 34th AAAI Conference on Artificial Intelligence , AAAI’20 
 
 AAAI Press, 2020, pp. 6340–6347 
 
 URL: https://arxiv.org/abs/1912.01321 
 

 
 [Yam20a] 
 Roman. Yampolskiy 
 
 “Unexplainability and Incomprehensibility of AI” 
 
 In Journal of Artificial Intelligence and Consciousness 7.2 , 2020, pp. 277–291 
 
 URL: https://arxiv.org/abs/1907.03869 
 

 
 [Zha+20a] 
 Haoran Zhang, Amy. Lu, Mohamed Abdalla, Matthew McDermott and Marzyeh Ghassemi 
 
 “Hurtful Words: Quantifying Biases in Clinical Contextual Word Embeddings” 
 
 In Proceedings of the ACM Conference on Health, Inference, and Learning , CHIL’20 
 
 New York, NY, USA: Association for Computing Machinery, 2020 
 
 DOI: 10.1145/3368555.3384448 
 

 
 [BPF21a] 
 Samyadeep Basu, Phil Pope and Soheil Feizi 
 
 “Influence Functions in Deep Learning Are Fragile” 
 
 In Proceedings of the 9th International Conference on Learning Representations , ICLR’21, 2021 
 
 URL: https://arxiv.org/abs/2006.14651 
 

 
 [BP21a] 
 Vaishak Belle and Ioannis Papantonis 
 
 “Principles and Practice of Explainable Machine Learning” 
 
 In Frontiers in Big Data 4 
 
 Frontiers Media S.A., 2021, pp. 688969–688969 
 
 URL: https://arxiv.org/abs/2009.11698 
 

 
 [Ben+21a] 
 Emily. Bender, Timnit Gebru, Angelina McMillan-Major and Shmargaret Shmitchell 
 
 “On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?” 
 
 In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency , FAccT’21 
 
 New York, NY, USA: Association for Computing Machinery, 2021, pp. 610–623 
 
 URL: https://dl.acm.org/doi/10.1145/3442188.3445922 
 

 
 [BF21a] 
 Emily Black and Matt Fredrikson 
 
 “Leave-One-Out Unfairness” 
 
 In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency , FAccT’21, 2021 
 
 URL: https://arxiv.org/abs/2107.10171 
 

 
 [BL21a] 
 Jonathan Brophy and Daniel Lowd 
 
 “Machine Unlearning for Random Forests” 
 
 In Proceedings of the 38th International Conference on Machine Learning , ICML’21, 2021 
 
 URL: https://arxiv.org/abs/2009.05567 
 

 
 [BH21a] 
 Nadia Burkart and Marco. Huber 
 
 “A Survey on the Explainability of Supervised Machine Learning” 
 
 In Journal Artificial Intelligence Research 70 
 
 El Segundo, CA, USA: AI Access Foundation, 2021, pp. 245–317 
 
 URL: https://arxiv.org/abs/2011.07876 
 

 
 [Che+21a] 
 Yuanyuan Chen, Boyang Li, Han Yu, Pengcheng Wu and Chunyan Miao 
 
 “HyDRA: Hypergradient Data Relevance Analysis for Interpreting Deep Neural Networks” 
 
 In Proceedings of the 35th AAAI Conference on Artificial Intelligence , AAAI’21 
 
 Virtual Only: Association for the Advancement of Artificial Intelligence, 2021 
 
 URL: https://arxiv.org/abs/2102.02515 
 

 
 [Das+21a] 
 Soumi Das, Arshdeep Singh, Saptarshi Chatterjee, Suparna Bhattacharya and Sourangshu Bhattacharya 
 
 “Finding High-Value Training Data Subset through Differentiable Convex Programming” 
 
 In Proceedings of the 2021 European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases , ECML PKDD’21, 2021 
 
 URL: https://arxiv.org/abs/2104.13794 
 

 
 [Dis+21a] 
 Michael Diskin, Alexey Bukhtiyarov, Max Ryabinin, Lucile Saulnier, Quentin Lhoest, Anton Sinitsin, Dmitry Popov, Dmitriy Pyrkin, Maxim Kashirin, Alexander Borzunov, Albert del Moral, Denis Mazur, Ilia Kobelev, Yacine Jernite, Thomas Wolf and Gennady Pekhimenko 
 
 “Distributed Deep Learning in Open Collaborations” 
 
 In Proceedings of the 35th Conference on Neural Information Processing Systems , NeurIPS’21, 2021 
 
 URL: https://arxiv.org/abs/2106.10207 
 

 
 [Dos+21a] 
 Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit and Neil Houlsby 
 
 “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale” 
 
 In Proceedings of the 9th International Conference on Learning Representations , ICLR’21, 2021 
 
 URL: https://arxiv.org/abs/2010.11929 
 

 
 [Fow+21a] 
 Liam Fowl, Micah Goldblum, Ping-yeh Chiang, Jonas Geiping, Wojtek Czaja and Tom Goldstein 
 
 “Adversarial Examples Make Strong Poisons” 
 
 In Proceedings of the 35th Conference on Neural Information Processing Systems , NeurIPS’21 
 
 Virtual Only: Curran Associates, Inc., 2021 
 
 URL: https://arxiv.org/abs/2106.10807 
 

 
 [Guo+21a] 
 Han Guo, Nazneen Rajani, Peter Hase, Mohit Bansal and Caiming Xiong 
 
 “FastIF: Scalable Influence Functions for Efficient Model Interpretation and Debugging” 
 
 In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing , EMNLP’21, 2021 
 
 URL: https://arxiv.org/abs/2012.15781 
 

 
 [HL21a] 
 Zayd Hammoudeh and Daniel Lowd 
 
 “Simple, Attack-Agnostic Defense Against Targeted Training Set Attacks Using Cosine Similarity” 
 
 In Proceedings of the 3rd ICML Workshop on Uncertainty and Robustness in Deep Learning , UDL’21, 2021 
 

 
 [HT21a] 
 Xiaochuang Han and Yulia Tsvetkov 
 
 “Fortifying Toxic Speech Detectors Against Veiled Toxicity” 
 
 In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing , EMNLP’20, 2021 
 
 URL: https://arxiv.org/abs/2010.03154 
 

 
 [Jag+21a] 
 Matthew Jagielski, Giorgio Severi, Niklas Pousette and Alina Oprea 
 
 “Subpopulation Data Poisoning Attacks” 
 
 In Proceedings of the 28th ACM SIGSAC Conference on Computer and Communications Security , CCS ’21 
 
 Virtual Only: Association for Computing Machinery, 2021 
 
 URL: https://arxiv.org/abs/2006.14026 
 

 
 [Jia+21b] 
 Ruoxi Jia, Fan Wu, Xuehui Sun, Jiacen Xu, David Dao, Bhavya Kailkhura, Ce Zhang, Bo Li and Dawn Song 
 
 “Scalability vs. Utility: Do We Have to Sacrifice One for the Other in Data Importance Quantification?” 
 
 In Proceedings of the 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition , CVPR’21, 2021 
 
 URL: https://arxiv.org/abs/1911.07128 
 

 
 [Jia+21c] 
 Ziheng Jiang, Chiyuan Zhang, Kunal Talwar and Michael Mozer 
 
 “Characterizing Structural Regularities of Labeled Data in Overparameterized Models” 
 
 In Proceedings of the 38th International Conference on Machine Learning , ICML’21, 2021, pp. 5034–5044 
 
 URL: https://arxiv.org/abs/2002.03206 
 

 
 [KS21a] 
 Karthikeyan K and Anders Søgaard 
 
 “Revisiting Methods for Finding Influential Examples”, 2021 
 
 arXiv: 2111.04683 [cs.LG] 
 

 
 [KC21a] 
 Zhifeng Kong and Kamalika Chaudhuri 
 
 “Understanding Instance-based Interpretability of Variational Auto-Encoders” 
 
 In Proceedings of the 35th Conference on Neural Information Processing Systems , NeurIPS’21 
 
 Virtual Only: Curran Associates, Inc., 2021 
 
 URL: https://arxiv.org/abs/2105.14203 
 

 
 [LF21a] 
 Alexander Levine and Soheil Feizi 
 
 “Deep Partition Aggregation: Provable Defenses against General Poisoning Attacks” 
 
 In Proceedings of the 9th International Conference on Learning Representations , ICLR’21, 2021 
 
 URL: https://arxiv.org/abs/2006.14768 
 

 
 [LLY21a] 
 Weixin Liang, Kai-Hui Liang and Zhou Yu 
 
 “HERALD: An Annotation Efficient Method to Detect User Disengagement in Social Conversations” 
 
 In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing , ACL-IJCNLP’21 
 
 Association for Computational Linguistics, 2021 
 
 URL: https://arxiv.org/abs/2106.00162 
 

 
 [Liu+21a] 
 Zhuoming Liu, Hao Ding, Huaping Zhong, Weijia Li, Jifeng Dai and Conghui He 
 
 “Influence Selection for Active Learning” 
 
 In Proceedings of the 18th International Conference on Computer Vision , ICCV’21, 2021 
 
 URL: https://arxiv.org/abs/2108.09331 
 

 
 [Meh+21a] 
 Ninareh Mehrabi, Fred Morstatter, Nripsuta Saxena, Kristina Lerman and Aram Galstyan 
 
 “A Survey on Bias and Fairness in Machine Learning” 
 
 In ACM Computing Surveys 54.6 
 
 New York, NY, USA: Association for Computing Machinery, 2021 
 
 URL: https://arxiv.org/abs/1908.09635 
 

 
 [Oh+21a] 
 Sejoon Oh, Sungchul Kim, Ryan. Rossi and Srijan Kumar 
 
 “Influence-guided Data Augmentation for Neural Tensor Completion” 
 
 In Proceedings of the 30th ACM International Conference on Information and Knowledge Management , CIKM’21 
 
 ACM, 2021 
 
 URL: https://arxiv.org/abs/2108.10248 
 

 
 [Red+21a] 
 Vijay Reddi, Greg Diamos, Pete Warden, Peter Mattson and David Kanter 
 
 “Data Engineering for Everyone”, 2021 
 
 arXiv: 2102.11447 [cs.LG] 
 

 
 [Ren+21a] 
 Pengzhen Ren, Yun Xiao, Xiaojun Chang, Po-Yao Huang, Zhihui Li, Brij. Gupta, Xiaojiang Chen and Xin Wang 
 
 “A Survey of Deep Active Learning” 
 
 In ACM Computing Surveys 54.9 
 
 New York, NY, USA: Association for Computing Machinery, 2021 
 
 URL: https://arxiv.org/abs/2009.00236 
 

 
 [SWS21a] 
 Yi Sui, Ga Wu and Scott Sanner 
 
 “Representer Point Selection via Local Jacobian Expansion for Post-hoc Classifier Explanation of Deep Neural Networks and Ensemble Models” 
 
 In Proceedings of the 35th Conference on Neural Information Processing Systems , NeurIPS’21 
 
 Virtual Only: Curran Associates, Inc., 2021 
 
 URL: https://openreview.net/forum?id=Wl32WBZnSP4 
 

 
 [SD21a] 
 Cecilia Summers and Michael. Dinneen 
 
 “Nondeterminism and Instability in Neural Network Optimization” 
 
 In Proceedings of the 38th International Conference on Machine Learning , ICML’21, 2021 
 
 URL: https://arxiv.org/abs/2103.04514 
 

 
 [Ter+21a] 
 Naoyuki Terashita, Hiroki Ohashi, Yuichi Nonaka and Takashi Kanemaru 
 
 “Influence Estimation for Generative Adversarial Networks” 
 
 In Proceedings of the 9th International Conference on Learning Representations , ICLR’21, 2021 
 
 URL: https://arxiv.org/abs/2101.08367 
 

 
 [vW21a] 
 Gerrit.. van den Burg and Christopher.. Williams 
 
 “On Memorization in Probabilistic Deep Generative Models” 
 
 In Proceedings of the 35th Conference on Neural Information Processing Systems , NeurIPS’21 
 
 Curran Associates, Inc., 2021 
 
 URL: https://arxiv.org/abs/2106.03216 
 

 
 [Wal+21a] 
 Eric Wallace, Tony. Zhao, Shi Feng and Sameer Singh 
 
 “Concealed Data Poisoning Attacks on NLP Models” 
 
 In Proceedings of the North American Chapter of the Association for Computational Linguistics , NAACL’21, 2021 
 
 URL: https://arxiv.org/abs/2010.12563 
 

 
 [YP21a] 
 Tom Yan and Ariel. Procaccia 
 
 “If You Like Shapley Then You’ll Love the Core” 
 
 In Proceedings of the 35th AAAI Conference on Artificial Intelligence , AAAI’21 
 
 Virtual Only: Association for the Advancement of Artificial Intelligence, 2021 
 
 URL: https://ojs.aaai.org/index.php/AAAI/article/view/16721 
 

 
 [Yan+21a] 
 Jingkang Yang, Kaiyang Zhou, Yixuan Li and Ziwei Liu 
 
 “Generalized Out-of-Distribution Detection: A Survey”, 2021 
 
 arXiv: 2110.11334 [cs.CV] 
 

 
 [Zha+21c] 
 Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht and Oriol Vinyals 
 
 “Understanding Deep Learning (Still) Requires Rethinking Generalization” 
 
 In Communications of the ACM 64.3 
 
 New York, NY, USA: Association for Computing Machinery, 2021, pp. 107–115 
 
 URL: https://dl.acm.org/doi/10.1145/3446776 
 

 
 [Zha+21d] 
 Chiyuan Zhang, Daphne Ippolito, Katherine Lee, Matthew Jagielski, Florian Tramèr and Nicholas Carlini 
 
 “Counterfactual Memorization in Neural Language Models”, 2021 
 
 arXiv: 2112.12938 [cs.CL] 
 

 
 [Zha+21e] 
 Wentao Zhang, Yexin Wang, Zhenbang You, Meng Cao, Ping Huang, Jiulong Shan, Zhi Yang and Bin Cui 
 
 “RIM: Reliable Influence-based Active Learning on Graphs” 
 
 In Proceedings of the 35th Conference on Neural Information Processing Systems , NeurIPS’21 
 
 Virtual Only: Curran Associates, Inc., 2021 
 
 URL: https://arxiv.org/abs/2110.14854 
 

 
 [Bae+22a] 
 Juhan Bae, Nathan Ng, Alston Lo, Marzyeh Ghassemi and Roger Grosse 
 
 “If Influence Functions are the Answer, Then What is the Question?” 
 
 In Proceedings of the 36th Conference on Neural Information Processing Systems , NeurIPS’22 
 
 Curran Associates, Inc., 2022 
 
 URL: https://arxiv.org/abs/2209.05364 
 

 
 [Bil22a] 
 Jeffrey Bilmes 
 
 “Submodularity in Machine Learning and Artificial Intelligence”, 2022 
 
 URL: https://arxiv.org/abs/2202.00132 
 

 
 [Bra+22a] 
 Joschka Braun, Micha Kornreich, JinHyeong Park, Jayashri Pawar, James Browning, Richard Herzog, Benjamin Odry and Li Zhang 
 
 “Influence Based Re-Weighing for Labeling Noise in Medical Imaging” 
 
 In Proceedings of the 19th IEEE International Symposium on Biomedical Imaging , ISBI’22, 2022 
 

 
 [Che+22a] 
 Ruoxin Chen, Zenan Li, Jie Li, Chentao Wu and Junchi Yan 
 
 “On Collective Robustness of Bagging Against Data Poisoning” 
 
 In Proceedings of the 39th International Conference on Machine Learning , ICML’22 
 
 PMLR, 2022 
 
 URL: https://arxiv.org/abs/2205.13176 
 

 
 [CG22a] 
 Gilad Cohen and Raja Giryes 
 
 “Membership Inference Attack Using Self Influence Functions”, 2022 
 
 arXiv: 2205.13680 [cs.LG] 
 

 
 [Eis+22a] 
 Thorsten Eisenhofer, Doreen Riepel, Varun Chandrasekaran, Esha Ghosh, Olga Ohrimenko and Nicolas Papernot 
 
 “Verifiable and Provably Secure Machine Unlearning”, 2022 
 
 arXiv: 2210.09126 [cs.LG] 
 

 
 [HL22a] 
 Zayd Hammoudeh and Daniel Lowd 
 
 “Identifying a Training-Set Attack’s Target Using Renormalized Influence Estimation” 
 
 In Proceedings of the 29th ACM SIGSAC Conference on Computer and Communications Security , CCS’22 
 
 Los Angeles, CA: Association for Computing Machinery, 2022 
 
 URL: https://arxiv.org/abs/2201.10055 
 

 
 [Ily+22a] 
 Andrew Ilyas, Sung Park, Logan Engstrom, Guillaume Leclerc and Aleksander Madry 
 
 “Datamodels: Understanding Predictions with Data and Data with Predictions” 
 
 In Proceedings of the 39th International Conference on Machine Learning , ICML’22 
 
 PMLR, 2022 
 
 URL: https://arxiv.org/abs/2202.00622 
 

 
 [Jia+22a] 
 Jinyuan Jia, Yupei Liu, Xiaoyu Cao and Neil Gong 
 
 “Certified Robustness of Nearest Neighbors against Data Poisoning and Backdoor Attacks” 
 
 In Proceedings of the 36th AAAI Conference on Artificial Intelligence , AAAI’22, 2022 
 
 URL: https://arxiv.org/abs/2012.03765 
 

 
 [KWR22a] 
 Nikhil Kandpal, Eric Wallace and Colin Raffel 
 
 “Deduplicating Training Data Mitigates Privacy Risks in Language Models” 
 
 In Proceedings of the 39th International Conference on Machine Learning , ICML’22 
 
 PMLR, 2022 
 
 URL: https://arxiv.org/abs/2202.06539 
 

 
 [KSH22a] 
 Shuming Kong, Yanyan Shen and Linpeng Huang 
 
 “Resolving Training Biases via Influence-based Data Relabeling” 
 
 In Proceedings of the 10th International Conference on Learning Representations , ICLR’22, 2022 
 
 URL: https://openreview.net/forum?id=EskfH0bwNVn 
 

 
 [KZ22a] 
 Yongchan Kwon and James Zou 
 
 “Beta Shapley: A Unified and Noise-Reduced Data Valuation Framework for Machine Learning” 
 
 In Proceedings of the 25th Conference on Artificial Intelligence and Statistics , AISTATS’22 
 
 PMLR, 2022 
 
 URL: https://arxiv.org/abs/2110.14049 
 

 
 [Li+22a] 
 Yiming Li, Baoyuan Wu, Yong Jiang, Zhifeng Li and Shu-Tao Xia 
 
 “Backdoor Learning: A Survey” 
 
 In IEEE Transactions on Neural Networks and Learning Systems , 2022 
 
 DOI: 10.1109/TNNLS.2022.3182979 
 

 
 [Lin+22a] 
 Jinkun Lin, Anqi Zhang, Mathias Lecuyer, Jinyang Li, Aurojit Panda and Siddhartha Sen 
 
 “Measuring the Effect of Training Data on Deep Learning Predictions via Randomized Experiments” 
 
 In Proceedings of the 39th International Conference on Machine Learning , ICML’22, 2022 
 
 URL: https://arxiv.org/abs/2206.10013 
 

 
 [Ngu+22a] 
 Thanh Nguyen, Thanh Huynh, Phi Nguyen, Alan-Chung Liew, Hongzhi Yin and Quoc Nguyen 
 
 “A Survey of Machine Unlearning” 
 
 In arXiv preprint arXiv:2209.02299 , 2022 
 
 arXiv: 2209.02299 [cs.LG] 
 

 
 [Oh+22a] 
 Sejoon Oh, Berk Ustun, Julian McAuley and Srijan Kumar 
 
 “Rank List Sensitivity of Recommender Systems to Interaction Perturbations” 
 
 In Proceedings of the 31st ACM International Conference on Information and Knowledge Management , CIKM’22, 2022 
 
 ACM 
 
 URL: https://arxiv.org/abs/2201.12686 
 

 
 [Ras+22a] 
 Soham Raste, Rahul Singh, Joel Vaughan and Vijayan. Nair 
 
 “Quantifying Inherent Randomness in Machine Learning Algorithms”, 2022 
 
 arXiv: 2206.12353 [stat.ML] 
 

 
 [Roz+22a] 
 Benedek Rozemberczki, Lauren Watson, Péter Bayer, Hao-Tsung Yang, Olivér Kiss, Sebastian Nilsson and Rik Sarkar 
 
 “The Shapley Value in Machine Learning”, 2022 
 
 arXiv: 2202.05594 [cs.LG] 
 

 
 [Sch+22a] 
 Andrea Schioppa, Polina Zablotskaia, David Torres and Artem Sokolov 
 
 “Scaling Up Influence Functions” 
 
 In Proceedings of the 36th AAAI Conference on Artificial Intelligence , AAAI’22, 2022 
 
 URL: https://arxiv.org/abs/2112.03052 
 

 
 [Thi+22a] 
 Hugo Thimonier, Fabrice Popineau, Arpad Rimmel, Bich-Liên Doan and Fabrice Daniel 
 
 “TracInAD: Measuring Influence for Anomaly Detection” 
 
 In Proceedings of the 2022 International Joint Conference on Neural Networks , IJCNN’22, 2022 
 
 URL: https://arxiv.org/abs/2205.01362 
 

 
 [WLF22a] 
 Wenxiao Wang, Alexander Levine and Soheil Feizi 
 
 “Improved Certified Defenses against Data Poisoning with (Deterministic) Finite Aggregation” 
 
 In Proceedings of the 39th International Conference on Machine Learning , ICML’22, 2022 
 
 URL: https://arxiv.org/abs/2202.02628 
 

 
 [Xia22a] 
 Chloe Xiang 
 
 “Scientists Increasingly Can’t Explain How AI Works” 
 
 In Vice , 2022 
 
 URL: https://www.vice.com/en/article/y3pezm/scientists-increasingly-cant-explain-how-ai-works 
 

 
 [Yeh+22a] 
 Chih-Kuan Yeh, Ankur Taly, Mukund Sundararajan, Frederick Liu and Pradeep Ravikumar 
 
 “First is Better Than Last for Language Data Influence” 
 
 In Proceedings of the 36th Conference on Neural Information Processing Systems , NeurIPS’22 
 
 Curran Associates, Inc., 2022 
 
 URL: https://arxiv.org/abs/2202.11844 
 

 
 [ZZ22a] 
 Rui Zhang and Shihua Zhang 
 
 “Rethinking Influence Functions of Neural Networks in the Over-Parameterized Regime” 
 
 In Proceedings of the 36th AAAI Conference on Artificial Intelligence , AAAI’22 
 
 Vancouver, Canada: Association for the Advancement of Artificial Intelligence, 2022 
 
 URL: https://arxiv.org/abs/2112.08297 
 

 
 [BL23a] 
 Sebastian Bordt and Ulrike von Luxburg 
 
 “From Shapley Values to Generalized Additive Models and back” 
 
 In Proceedings of The 26th International Conference on Artificial Intelligence and Statistics , AISTATS’23, 2023 
 
 URL: https://arxiv.org/abs/2209.04012 
 

 
 [BHL23a] 
 Jonathan Brophy, Zayd Hammoudeh and Daniel Lowd 
 
 “Adapting and Evaluating Influence-Estimation Methods for Gradient-Boosted Decision Trees” 
 
 In Journal of Machine Learning Research 24 , 2023, pp. 1–48 
 
 URL: https://arxiv.org/abs/2205.00359 
 

 
 [DG23a] 
 Zheng Dai and David. Gifford 
 
 “Training Data Attribution for Diffusion Models”, 2023 
 
 arXiv: 2306.02174 [stat.ML] 
 

 
 [HL23a] 
 Zayd Hammoudeh and Daniel Lowd 
 
 “Reducing Certified Regression to Certified Classification for General Poisoning Attacks” 
 
 In Proceedings of the 1st IEEE Conference on Secure and Trustworthy Machine Learning , SaTML’23, 2023 
 
 URL: https://arxiv.org/abs/2208.13904 
 

 
 [KCC23a] 
 Nohyun Ki, Hoyong Choi and Hye Chung 
 
 “Data Valuation Without Training of a Model” 
 
 In Proceedings of the 11th International Conference on Learning Representations , ICLR’23, 2023 
 
 URL: https://openreview.net/forum?id=XIzO8zr-WbM 
 

 
 [NSO23a] 
 Elisa Nguyen, Minjoon Seo and Seong Oh 
 
 “A Bayesian Perspective On Training Data Attribution”, 2023 
 
 arXiv: 2305.19765 [cs.LG] 
 

 
 [Par+23a] 
 Sung Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc and Aleksander Madry 
 
 “TRAK: Attributing Model Behavior at Scale” 
 
 In Proceedings of the 40th International Conference on Machine Learning , ICML’23, 2023 
 
 URL: https://arxiv.org/abs/2303.14186 
 

 
 [Rez+23a] 
 Keivan Rezaei, Kiarash Banihashem, Atoosa Chegini and Soheil Feizi 
 
 “Run-Off Election: Improved Provable Defense against Data Poisoning Attacks” 
 
 In Proceedings of the 40th International Conference on Machine Learning , ICML’23, 2023 
 
 URL: https://arxiv.org/abs/2302.02300 
 

 
 [Sch+23a] 
 Andrea Schioppa, Katja Filippova, Ivan Titov and Polina Zablotskaia 
 
 “Theoretical and Practical Perspectives on what Influence Functions Do”, 2023 
 
 arXiv: https://arxiv.org/abs/2305.16971 
 

 
 [TYR23a] 
 Che-Ping Tsai, Chih-Kuan Yeh and Pradeep Ravikumar 
 
 “Faith-Shap: The Faithful Shapley Interaction Index” 
 
 In Journal of Machine Learning Research 24.94 , 2023, pp. 1–42 
 
 URL: https://arxiv.org/abs/2203.00870 
 

 
 [Tsa+23a] 
 Che-Ping Tsai, Jiong Zhang, Eli Chien, Hsiang-Fu Yu, Cho-Jui Hsieh and Pradeep Ravikumar 
 
 “Representer Point Selection for Explaining Regularized High-dimensional Models” 
 
 In Proceedings of the 40th International Conference on Machine Learning , ICML’23, 2023 
 
 URL: https://arxiv.org/abs/2305.20002 
 

 
 [Tuk+23a] 
 Murad Tukan, Samson Zhou, Alaa Maalouf, Daniela Rus, Vladimir Braverman and Dan Feldman 
 
 “Provable Data Subset Selection for Efficient Neural Networks Training” 
 
 In Proceedings of the 40th International Conference on Machine Learning , ICML’23, 2023 
 
 URL: https://arxiv.org/abs/2303.05151 
 

 
 [WJ23a] 
 Jiachen. Wang and Ruoxi Jia 
 
 “Data Banzhaf: A Robust Data Valuation Framework for Machine Learning” 
 
 In Proceedings of the 26th International Conference on Artificial Intelligence and Statistics , AISTATS’23, 2023 
 
 URL: https://arxiv.org/abs/2205.15466 
 

 
 [Yan+23a] 
 Shuo Yang, Zeke Xie, Hanyu Peng, Min Xu, Mingming Sun and Ping Li 
 
 “Dataset Pruning: Reducing Training Data by Examining Generalization Influence” 
 
 In Proceedings of the 11th International Conference on Learning Representations , ICLR’23, 2023 
 
 URL: https://arxiv.org/abs/2205.09329 
 

 
 [YHL23a] 
 Wencong You, Zayd Hammoudeh and Daniel Lowd 
 
 “Large Language Models Are Better Adversaries: Exploring Generative Clean-Label Backdoor Attacks Against Text Classifiers” 
 
 In Findings of the Association for Computational Linguistics , EMNLP’23, 2023 
 

 
 [Zen+23a] 
 Yingyan Zeng, Jiachen. Wang, Si Chen, Hoang Just, Ran Jin and Ruoxi Jia 
 
 “ModelPred: A Framework for Predicting Trained Model from Training Data” 
 
 In Proceedings of the 1st IEEE Conference on Secure and Trustworthy Machine Learning , SaTML’23, 2023 
 
 URL: https://arxiv.org/abs/2111.12545 
 

 
 [HL24b] 
 Zayd Hammoudeh and Daniel Lowd 
 
 “Provable Robustness Against a Union of ℓ 0 \ell_{0} Attacks” 
 
 In Proceedings of the 38th AAAI Conference on Artificial Intelligence , AAAI’24, 2024 
 
 URL: https://arxiv.org/abs/2302.11628 
 

 
 [HL24c] 
 Zayd Hammoudeh and Daniel Lowd 
 
 “Training Data Influence Analysis and Estimation: A Survey” 
 
 In Machine Learning , 2024 
 
 DOI: 10.1007/s10994-023-06495-7 
 

 
 
 
 
 
   
 
 
 Training Data Influence Analysis 
 and Estimation: A Survey 

 Supplemental Materials 
   
 
 
 
 
 Organization of the Appendix 

 
 
 
 

## Appendix A Nomenclature

 
 Table 1 provides a general nomenclature reference that applies throughout this document, including for all influence analysis methods.
Table 2 summarizes the nomenclature related to model training.
Table 3 details nomenclature symbols that are specific to an individual influence analysis method. 

 
 
 Table 1: General nomenclature reference 
 
 
 [ r ] {[r]} | 
 
 
 Set { 1 ; … ; r } \{1\mathchar 59\relax\ldots\mathchar 59\relax r\} for arbitrary positive integer r r 
 | 

 
 A ∼ m B {A\stackrel{{\scriptstyle m}}{{\sim}}B} | 
 
 
 Set A A is a u.a.r. subset of size m m from set B B 
 | 

 
 2 A 2^{A} | 
 
 
 Power set of A A 
 | 

 
 𝟙 ​ [ a ] {\mathbbm{1}[a]} | 
 
 
 Indicator function where 𝟙 ​ [ a ] = 1 {{\mathbbm{1}[a]}=1} if predicate a a is true and 0 otherwise 
 | 

 
 0 → \vec{0} | 
 
 
 Zero vector 
 | 

 
 x x | 
 
 
 Feature vector 
 | 

 
 𝒳 \mathcal{X} | 
 
 
 Feature domain where 𝒳 ⊆ d {\mathcal{X}\subseteq\real^{d}} and ∀ x x ∈ 𝒳 {\forall_{x}\,x\in\mathcal{X}} 
 | 

 
 d d | 
 
 
 Feature dimension where d ≔ | x | {d\coloneqq\lvert x\rvert} 
 | 

 
 y y | 
 
 
 Dependent/target value, e.g., label 
 | 

 
 𝒴 \mathcal{Y} | 
 
 
 Dependent value domain, i.e., ∀ y y ∈ 𝒴 {\forall_{y}\,y\in\mathcal{Y}} . Generally 𝒴 ⊆ {\mathcal{Y}\subseteq\real} 
 | 

 
 z z | 
 
 
 Feature vector-dependent value tuple where z ≔ ( x ​ ; ​ y ) {z\coloneqq(x\mathord{\mathchar 59\relax}y)} 
 | 

 
 𝒵 \mathcal{Z} | 
 
 
 Instance domain where 𝒵 ≔ 𝒳 × 𝒴 {\mathcal{Z}\coloneqq\mathcal{X}\times\mathcal{Y}} and ∀ z z ∈ 𝒵 {\forall_{z}\,z\in\mathcal{Z}} 
 | 

 
 𝒫 \mathcal{P} | 
 
 
 Instance data distribution where 𝒫 : 𝒵 → ≥ 0 {\mathcal{P}:\mathcal{Z}\rightarrow\real_{{\geq}0}} 
 | 

 
 𝒟 \mathcal{D} | 
 
 
 Training set where 𝒟 ≔ { z i } i = 1 n {\mathcal{D}\coloneqq\{z_{i}\}_{i=1}^{n}} 
 | 

 
 n n | 
 
 
 Size of the training set where n ≔ | 𝒟 | {n\coloneqq\lvert\mathcal{D}\rvert} 
 | 

 
 D D | 
 
 
 Arbitrary training subset where D ⊆ 𝒟 {D\subseteq\mathcal{D}} 
 | 

 
 i i | 
 
 
 Arbitrary training example index where i ∈ [ n ] {i\in{[n]}} 
 | 

 
 z te z_{\text{te}} | 
 
 
 Arbitrary test instance where z te ≔ ( x te ; y te ) {z_{\text{te}}\coloneqq(x_{\text{te}}\mathchar 59\relax y_{\text{te}})} 
 | 

 
 f f | 
 
 
 Model where f : 𝒳 → 𝒴 {f:\mathcal{X}\rightarrow\mathcal{Y}} 
 | 

 
 θ \theta | 
 
 
 Model parameters where θ ∈ p {\theta\in\real^{p}} 
 | 

 
 p p | 
 
 
 Parameter dimension where p ≔ | θ | {p\coloneqq\lvert\theta\rvert} 
 | 

 
 ℓ \ell | 
 
 
 Loss function where ℓ : 𝒴 × 𝒴 → {\ell:\mathcal{Y}\times\mathcal{Y}\rightarrow\real} 
 | 

 
 ℒ ⁡ ( z , θ ) {\mathcal{L}(z;\theta)} | 
 
 
 Empirical risk of example z = ( x ​ ; ​ y ) {z=(x\mathord{\mathchar 59\relax}y)} w.r.t. θ \theta , where ℒ ⁡ ( z , θ ) ≔ ℓ ⁡ ( f ⁡ ( x , θ ) , y ) {{\mathcal{L}(z;\theta)}\coloneqq{\ell({f(x;\theta)}\mathchar 59\relax y)}} 
 | 

 
 ℐ ⁡ ( z i , z te ) {\mathcal{I}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)} | 
 
 
 Exact pointwise influence of training instance z i z_{i} on test instance z te z_{\text{te}} 
 | 

 
 ℐ ^ ​ ( z i , z te ) {\widehat{\mathcal{I}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)} | 
 
 
 Estimate of training instance z i z_{i} ’s pointwise influence on test instance z te z_{\text{te}} 
 | 

 
 ℐ ⁡ ( D , z te ) {\mathcal{I}\left(D\mathchar 59\relax z_{\text{te}}\right)} | 
 
 
 Group influence of training subset D ⊆ 𝒟 {D\subseteq\mathcal{D}} on test instance z te z_{\text{te}} 
 | 

 
 ℐ ^ ​ ( D , z te ) {\widehat{\mathcal{I}}\left(D\mathchar 59\relax z_{\text{te}}\right)} | 
 
 
 Estimate of the group influence of training subset D ⊆ 𝒟 {D\subseteq\mathcal{D}} on test instance z te z_{\text{te}} 
 | 

 
 D CS D_{\textnormal{CS}} | 
 
 
 A coreset (Sec. 3.3 ) 
 | 

 
 
 Table 2: Training related nomenclature reference. 
 
 
 T T | 
 
 
 Number of training iterations 
 | 

 
 t t | 
 
 
 Training iteration number where t ∈ { 0 ; 1 ; … ; T } {t\in\{0\mathchar 59\relax 1\mathchar 59\relax\ldots\mathchar 59\relax T\}} . t = 0 {t=0} denotes initial conditions. 
 | 

 
 θ ( t ) \theta^{(t)} | 
 
 
 Model parameters at the end of iteration t t . 
 | 

 
 θ ( 0 ) \theta^{(0)} | 
 
 
 Model parameters at the start of training 
 | 

 
 θ ( T ) \theta^{(T)} | 
 
 
 Model parameters at the end of training 
 | 

 
 θ ∗ \theta^{*} | 
 
 
 Optimal model parameters 
 | 

 
 θ D ( T ) \theta^{(T)}_{D} | 
 
 
 Final model parameters trained on training data subset D ⊆ 𝒟 {D\subseteq\mathcal{D}} 
 | 

 
 λ \lambda | 
 
 
 L 2 L_{2} regularization (i.e., weight decay) hyperparameter 
 | 

 
 ℬ ( t ) \mathcal{B}^{(t)} | 
 
 
 (Mini)batch used during training iteration t t where ℬ ( t ) ⊆ 𝒟 {\mathcal{B}^{(t)}\subseteq\mathcal{D}} 
 | 

 
 b ( t ) b^{(t)} | 
 
 
 Batch size for iteration t t , where b ( t ) ≔ | ℬ ( t ) | {b^{(t)}\coloneqq\lvert\mathcal{B}^{(t)}\rvert} 
 | 

 
 η ( t ) \eta^{(t)} | 
 
 
 Learning rate at training iteration t t 
 | 

 
 Θ \Theta | 
 
 
 Serialized training parameters where Θ ⊆ { θ ( 0 ) ; … ; θ ( T − 1 ) } {\Theta\subseteq\{\theta^{(0)}\mathchar 59\relax\ldots\mathchar 59\relax\theta^{(T-1)}\}} 
 | 

 
 ∇ θ ℒ ​ ( z i , θ ( t ) ) \nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(t)})} | 
 
 
 Training instance z i z_{i} ’s risk gradient for iteration t t 
 | 

 
 ∇ θ 2 ​ ℒ ​ ( z i , θ ( t ) ) \nabla_{\theta}^{2}{\mathcal{L}(z_{i};\theta^{(t)})} | 
 
 
 Training instance z i z_{i} ’s risk Hessian for iteration t t 
 | 

 
 H θ ( t ) H_{\theta}^{(t)} | 
 
 
 Empirical risk Hessian the entire training set where H θ ( t ) ≔ 1 n ​ ∑ i = 1 n ∇ θ 2 ​ ℒ ​ ( z i , θ ( t ) ) {H_{\theta}^{(t)}\coloneqq\frac{1}{n}\sum_{i=1}^{n}\nabla_{\theta}^{2}{\mathcal{L}(z_{i};\theta^{(t)})}} 
 | 

 
 ( H θ ( t ) ) − 1 {(H_{\theta}^{(t)})^{-1}} | 
 
 
 Inverse of the empirical risk Hessian 
 | 

 
 
 Table 3: Influence (estimator) specific hyperparameters and nomenclature 
 
 
 𝒟 ∖ z i \mathcal{D}^{\setminus z_{i}} | 
 
 
 Leave-one-out training set where instance z i z_{i} is held out 
 | 

 
 k k | 
 
 
 k k - nearest neighbors neighborhood size 
 | 

 
 Neigh ​ ( x te , D ) {\text{Neigh}(x_{\text{te}};D)} | 
 
 
 k k NN neighborhood for test feature vector x te x_{\text{te}} from training set D ⊆ 𝒟 {D\subseteq\mathcal{D}} 
 | 

 
 K K | 
 
 
 Number of submodels trained by the Downsampling estimator 
 | 

 
 D k D^{k} | 
 
 
 Training set used by the k k - th Downsampling submodel 
 | 

 
 θ D k ( T ) \theta^{(T)}_{D^{k}} | 
 
 
 Final model parameters used by the k k - th Downsampling submodel 
 | 

 
 K i K_{i} | 
 
 
 Number of Downsampling submodels trained using z i z_{i} , where K i ≔ ∑ k = 1 K 𝟙 [ z i ∈ D k ] {K_{i}\coloneqq\sum_{k=1}^{K}{\mathbbm{1}[z_{i}\in D^{k}]}} 
 | 

 
 m m | 
 
 
 Downsampling submodel training-set size where ∀ k | D k | = m n {\forall_{k}\,\lvert D^{k}\rvert=m n} 
 | 

 
 ν \nu | 
 
 
 Shapley value characteristic function ν : 2 A → {\nu:2^{A}\rightarrow\real} for arbitrary set A A . 
 | 

 
 ϵ i \epsilon_{i} | 
 
 
 Training instance i i weight perturbation 
 | 

 
 θ + ϵ i ( t ) \theta^{(t)}_{+\epsilon_{i}} | 
 
 
 Model parameters trained on a training set perturbed by ϵ i \epsilon_{i} 
 | 

 
 α i \alpha_{i} | 
 
 
 Training instance z i z_{i} ’s representer value, where α i ≔ − 1 λ ​ n ​ ∂ ℓ ⁡ ( y ^ i , y i ) ∂ y ^ {\alpha_{i}\coloneqq-\frac{1}{\lambda n}\frac{\partial{\ell(\widehat{y}_{i}\mathchar 59\relax y_{i})}}{\partial\widehat{y}}} 
 | 

 
 θ ¨ ( T ) \ddot{\theta}^{(T)} | 
 
 
 Model f f ’s final linear layer parameters 
 | 

 
 θ ˙ ( T ) \dot{\theta}^{(T)} | 
 
 
 All model f f ’s parameters except the final layer, where θ ˙ ( T ) := θ ( T ) ∖ θ ¨ ( T ) {\dot{\theta}^{(T)}:=\theta^{(T)}\setminus\ddot{\theta}^{(T)}} 
 | 

 
 𝐟 i \mathbf{f}_{i} | 
 
 
 Training instance z i z_{i} ’s feature representation input into model f f ’s final linear layer 
 | 

 
 𝐟 te \mathbf{f}_{\text{te}} | 
 
 
 Test instance z te z_{\text{te}} ’s feature representation input into model f f ’s final linear layer 
 | 

 
 𝒦 \mathcal{K} | 
 
 
 Kernel (similarity) function between two (feature) vectors 
 | 

 
 r ⁡ ( θ ) {r(\theta)} | 
 
 
 Regularizer function where r : p → ≥ 0 {r:\real^{p}\rightarrow\real_{{\geq}0}} 
 | 

 
 𝒯 \mathcal{T} | 
 
 
 Subset of the training iterations considered by TracInCP, where 𝒯 ⊂ [ T ] {\mathcal{T}\subset{[T]}} . 
 | 

 
 h ~ i ( t ) \widetilde{h}^{(t)}_{i} | 
 
 
 Training hypergradient where h ~ ( t ) i ≔ d ​ θ + ϵ i ( t ) d ​ ϵ i ∈ p {\widetilde{h}^{(t)}_{i}\coloneqq\frac{d\theta^{(t)}_{+\epsilon_{i}}}{d\epsilon_{i}}\in\real^{p}} 
 | 

 
 
 
 
 

## Appendix B Influence Analysis Method Definition Reference

 
 Table 4: Influence analysis method formal definitions including equation numbers and citations.
 
 
 
 
 
 Memorization [ FZ20a , Pru+20a ] 
 | 
 
 
 
 
 
 
 
 
 

 
 
 Mem ​ ( z i ) ≔ ℐ ⁡ ( z i , z i ) {\textsc{Mem}(z_{i})}\coloneqq{\mathcal{I}\left(z_{i}\mathchar 59\relax z_{i}\right)} 
 
 ( 4 ) 
 
 
 
 
 | 

 
 
 
 Cook’s Distance [ Coo77a , Eq. (5)] 
 | 
 
 
 
 
 
 
 
 
 

 
 
 ℐ Cook ​ ( z i ) ≔ θ ( T ) − θ 𝒟 ∖ z i ( T ) {\mathcal{I}_{\text{Cook}}\mathopen{}\left(z_{i}\right)\mathclose{}}\coloneqq\theta^{(T)}-{\theta^{(T)}_{\mathcal{D}^{\setminus z_{i}}}} 
 
 ( 5 ) 
 
 
 
 
 | 

 
 
 
 Leave-One-Out Influence [ CW82a ] 
 | 
 
 
 
 
 
 
 
 
 

 
 
 ℐ LOO ​ ( z i , z te ) ≔ ℒ ⁡ ( z te , θ 𝒟 ∖ z i ( T ) ) − ℒ ⁡ ( z te , θ ( T ) ) {\mathcal{I}_{\textsc{LOO}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\coloneqq{\mathcal{L}(z_{\text{te}};{\theta^{(T)}_{\mathcal{D}^{\setminus z_{i}}}})}-{\mathcal{L}(z_{\text{te}};\theta^{(T)})} 
 
 ( 8 ) 
 
 
 
 
 | 

 
 
 
 Downsampling Influence Estimator [ FZ20a ] 
 | 
 
 
 
 
 
 
 
 
 

 
 
 ℐ ^ Down ​ ( z i , z te ) ≔ 1 K − K i ​ ∑ k z i ∉ D k ℒ ⁡ ( z te , θ D k ( T ) ) − 1 K i ​ ∑ k ′ z i ∈ D k ′ ℒ ⁡ ( z te , θ D k ′ ( T ) ) {\widehat{\mathcal{I}}_{\textsc{Down}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\coloneqq\frac{1}{K-K_{i}}\sum_{\begin{subarray}{c}k\\
z_{i}\notin D^{k}\end{subarray}}{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D^{k}})}-\frac{1}{K_{i}}\sum_{\begin{subarray}{c}k^{\prime}\\
z_{i}\in D^{k^{\prime}}\end{subarray}}{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D^{k^{\prime}}})} 
 
 ( 9 ) 
 
 
 
 
 | 

 
 
 
 Consistency Score [ Jia+21c ] 
 | 
 
 
 
 
 
 
 
 
 

 
 
 C 𝒟 ​ ( z te ) ≔ 𝔼 m ∼ [ n ] ​ [ − 𝔼 D ∼ m 𝒟 ​ [ ℒ ⁡ ( z te , θ D ( T ) ) ] ] {C_{\mathcal{D}}(z_{\text{te}})}\coloneqq\mathbb{E}_{m\,\sim\,{[n]}}[-\mathbb{E}_{D\,\stackrel{{\scriptstyle m}}{{\sim}}\,\mathcal{D}}[{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D})}]] 
 
 
 
 
 
 | 

 
 
 
 Shapley Value Pointwise Influence [ Sha53a , GZ19a ] 
 | 
 
 
 
 
 
 
 
 
 

 
 
 ℐ SV ​ ( z i , z te ) ≔ 1 n ​ ∑ D ⊆ 𝒟 ∖ z i 1 ( n − 1 | D | ) ​ [ ℒ ⁡ ( z te , θ D ( T ) ) − ℒ ⁡ ( z te , θ D ∪ z i ( T ) ) ] {\mathcal{I}_{\text{SV}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\coloneqq\frac{1}{n}\sum_{D\subseteq\mathcal{D}^{\setminus z_{i}}}\frac{1}{\binom{n-1}{\lvert D\rvert}}[{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D})}-{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D\cup z_{i}})}\vphantom{\Big|}] 
 
 ( 18 ) 
 
 
 
 
 | 

 
 
 
 Shapley Interaction Index [ GR99a ] 
 | 
 
 
 
 
 
 
 
 
 

 
 
 ℐ SV ( A ; z te ) ≔ − ∑ D ⊆ 𝒟 ∖ A ( n − | A | − | D | ) ! ​ | D | ! ( n − | A | + 1 ) ! ∑ D ′ ⊆ A ( − 1 ) | A | − | D ′ | ℒ ( z te ; θ D ∪ D ′ ( T ) ) {\mathcal{I}_{\text{SV}}\left(A\mathchar 59\relax z_{\text{te}}\right)}\coloneqq-\sum_{D\subseteq\mathcal{D}\setminus A}\frac{(n-\lvert A\rvert-\lvert D\rvert)!\,\,\lvert D\rvert!}{(n-\lvert A\rvert+1)!}\sum_{D^{\prime}\subseteq A}\left(-1\right)^{\lvert A\rvert-\lvert D^{\prime}\rvert}{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D\cup D^{\prime}})} 
 
 ( 19 ) 
 
 
 
 
 | 

 
 
 
 Shapley-Taylor Interaction Index [ SDA20a ] 
 | 
 
 
 
 
 
 
 
 
 

 
 
 ℐ ST ( A ; z te ) ≔ − | A | n ∑ D ⊆ 𝒟 ∖ A 1 ( n − 1 | D | ) ∑ D ′ ⊆ A ( − 1 ) | A | − | D ′ | ℒ ( z te ; θ D ∪ D ′ ( T ) ) {\mathcal{I}_{\text{ST}}\left(A\mathchar 59\relax z_{\text{te}}\right)}\coloneqq-\frac{\lvert A\rvert}{n}\sum_{D\subseteq\mathcal{D}\setminus A}\frac{1}{\binom{n-1}{\lvert D\rvert}}\sum_{D^{\prime}\subseteq A}\left(-1\right)^{\lvert A\rvert-\lvert D^{\prime}\rvert}{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D\cup D^{\prime}})} 
 
 ( 21 ) 
 
 
 
 
 | 

 
 
 
 k k - Nearest Neighbors Shapley Influence [ Jia+19b ] 
 | 
 
 
 
 
 
 
 
 
 

 
 
 ℐ k ​ NN-SV ​ ( z i , z te ) ≔ 𝟙 [ y n = y te ] n + ∑ j = i n − 1 𝟙 [ y j = y te ] − 𝟙 [ y j + 1 = y te ] k ​ min ⁡ { k ​ ; ​ j } j {\mathcal{I}_{k\text{NN-SV}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\coloneqq\frac{{\mathbbm{1}[y_{n}=y_{\text{te}}]}}{n}+\sum_{j=i}^{n-1}\frac{{\mathbbm{1}[y_{j}=y_{\text{te}}]}-{\mathbbm{1}[y_{j+1}=y_{\text{te}}]}}{k}\frac{\min\{k\mathord{\mathchar 59\relax}j\}}{j} 
 
 ( 26 ) 
 
 
 
 
 | 

 
 
 
 Banzhaf Value [ Ban65a ] 
 | 
 
 
 
 
 
 
 
 
 

 
 
 ℐ Banzhaf ​ ( z i , z te ) ≔ 1 2 n − 1 ​ ∑ D ⊆ 𝒟 ∖ z i ℒ ⁡ ( z te , θ D ( T ) ) − ℒ ⁡ ( z te , θ D ∪ z i ( T ) ) {\mathcal{I}_{\text{Banzhaf}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\coloneqq\frac{1}{2^{n-1}}\sum_{D\subseteq\mathcal{D}^{\setminus z_{i}}}{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D})}-{\mathcal{L}(z_{\text{te}};\theta^{(T)}_{D\cup z_{i}})} 
 
 ( 27 ) 
 
 
 
 
 | 

 
 
 
 Influence Functions Estimator [ KL17a ] 
 | 
 
 
 
 
 
 
 
 
 

 
 
 ℐ ^ IF ​ ( z i , z te ) ≔ 1 n ​ ∇ θ ℒ ​ ( z te , θ ( T ) ) ⊺ ​ ( H θ ( T ) ) − 1 ​ ∇ θ ℒ ​ ( z i , θ ( T ) ) {\widehat{\mathcal{I}}_{\textsc{IF}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\coloneqq\frac{1}{n}\nabla_{\theta}{\mathcal{L}(z_{\text{te}};\theta^{(T)})}^{\intercal}{(H_{\theta}^{(T)})^{-1}}\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(T)})} 
 
 ( 33 ) 
 
 
 
 
 | 

 
 
 
 Linear Model Representer Point Influence [ Yeh+18a ] 
 | 
 
 
 
 
 
 
 
 
 

 
 
 ℐ RP ​ ( z i , z te ) = α i ​ x i ⊺ ​ x te | y te {\mathcal{I}_{\textsc{RP}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}=\alpha_{i}x_{i}^{\intercal}x_{\text{te}}\,|_{y_{\text{te}}} 
 
 ( 46 ) 
 
 
 
 
 | 

 
 
 
 Representer Point Influence Estimator [ Yeh+18a ] 
 | 
 
 
 
 
 
 
 
 
 

 
 
 ℐ ^ RP ​ ( z i , z te ) ≔ α i ​ 𝐟 i ⊺ ​ 𝐟 te | y te {\widehat{\mathcal{I}}_{\textsc{RP}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\coloneqq\alpha_{i}\mathbf{f}_{i}^{\intercal}\mathbf{f}_{\text{te}}\,|_{y_{\text{te}}} 
 
 ( 47 ) 
 
 
 
 
 | 

 
 
 
 TracIn Ideal Pointwise Influence [ Pru+20a ] 
 | 
 
 
 
 
 
 
 
 
 

 
 
 ℐ TracIn ​ ( z i , z te ) ≔ ∑ t z i = ℬ ( t ) ( ℒ ⁡ ( z te , θ ( t − 1 ) ) − ℒ ⁡ ( z te , θ ( t ) ) ) {\mathcal{I}_{\text{TracIn{}}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\coloneqq\sum_{\begin{subarray}{c}t\\
z_{i}=\mathcal{B}^{(t)}\end{subarray}}\left({\mathcal{L}(z_{\text{te}};\theta^{(t-1)})}-{\mathcal{L}(z_{\text{te}};\theta^{(t)})}\right) 
 
 ( 51 ) 
 
 
 
 
 | 

 
 
 
 TracIn Influence Estimator [ Pru+20a ] 
 | 
 
 
 
 
 
 
 
 
 

 
 
 ℐ ^ TracIn ​ ( z i , z te ) ≔ ∑ t z i ∈ ℬ ( t ) η ( t ) | ℬ ( t ) | ​ ∇ θ ℒ ​ ( z i , θ ( t − 1 ) ) ⊺ ​ ∇ θ ℒ ​ ( z te , θ ( t − 1 ) ) {\widehat{\mathcal{I}}_{\text{TracIn{}}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\coloneqq\sum_{\begin{subarray}{c}t\\
z_{i}\in\mathcal{B}^{(t)}\end{subarray}}\frac{\eta^{(t)}}{\lvert\mathcal{B}^{(t)}\rvert}\,{{\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(t-1)})}}^{\intercal}\,{\nabla_{\theta}{\mathcal{L}(z_{\text{te}};\theta^{(t-1)})}}} 
 
 ( 55 ) 
 
 
 
 
 | 

 
 
 
 TracInCP Influence Estimator [ Pru+20a ] 
 | 
 
 
 
 
 
 
 
 
 

 
 
 ℐ ^ TracInCP ​ ( z i , z te ) ≔ ∑ t ∈ 𝒯 η ( t ) ​ ∇ θ ℒ ​ ( z i , θ ( t − 1 ) ) ⊺ ​ ∇ θ ℒ ​ ( z te , θ ( t − 1 ) ) {\widehat{\mathcal{I}}_{\text{TracIn{}CP}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\coloneqq\sum_{t\in\mathcal{T}}\eta^{(t)}\,{{\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(t-1)})}}^{\intercal}\,{\nabla_{\theta}{\mathcal{L}(z_{\text{te}};\theta^{(t-1)})}}} 
 
 ( 56 ) 
 
 
 
 
 | 

 
 
 
 HyDRA Influence Estimator [ Che+21a ] 
 | 
 
 
 
 
 
 
 
 
 

 
 
 ℐ ^ HyDRA ​ ( z i , z te ) ≔ − 1 n ​ ∇ θ ℒ ​ ( z te , θ ( T ) ) ⊺ ​ h ~ i ( T ) {\widehat{\mathcal{I}}_{\textsc{HyDRA}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\coloneqq-\frac{1}{n}\nabla_{\theta}{\mathcal{L}(z_{\text{te}};\theta^{(T)})}^{\intercal}\,\widetilde{h}^{(T)}_{i} 
 
 ( 62 ) 
 
 
 
 
 | 

 
 
 
 θ \theta - Relative Influence Estimator [ BBD20a ] 
 | 
 
 
 
 
 
 
 
 
 

 
 
 ℐ ^ RelatIF ​ ( z i , z te ) ≔ ℐ ^ IF ​ ( z i , z te ) ∥ ( H θ ( T ) ) − 1 ​ ∇ θ ℒ ​ ( z i , θ ( T ) ) ∥ {\widehat{\mathcal{I}}_{\text{RelatIF}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\coloneqq\frac{{\widehat{\mathcal{I}}_{\textsc{IF}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}}{\lVert{(H_{\theta}^{(T)})^{-1}}\,\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(T)})}\rVert} 
 
 ( 66 ) 
 
 
 
 
 | 

 
 
 
 GAS Renormalized Influence Estimator [ HL22a ] 
 | 
 
 
 
 
 
 
 
 
 

 
 
 ℐ ^ GAS ​ ( z i , z te ) ≔ ∑ t ∈ 𝒯 η ( t ) ​ ∇ θ ℒ ​ ( z i , θ ( t − 1 ) ) ⊺ ​ ∇ θ ℒ ​ ( z te , θ ( t − 1 ) ) ∥ ∇ θ ℒ ​ ( z i , θ ( t − 1 ) ) ∥ ​ ∥ ∇ θ ℒ ​ ( z te , θ ( t − 1 ) ) ∥ {\widehat{\mathcal{I}}_{\textsc{GAS}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\coloneqq\sum_{t\in\mathcal{T}}\eta^{(t)}\,\frac{{{\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(t-1)})}}^{\intercal}\,{\nabla_{\theta}{\mathcal{L}(z_{\text{te}};\theta^{(t-1)})}}}}{\lVert{\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(t-1)})}}\rVert\,\lVert{\nabla_{\theta}{\mathcal{L}(z_{\text{te}};\theta^{(t-1)})}}\rVert} 
 
 ( 68 ) 
 
 
 
 
 | 

 
 
 
 Renormalized Influence Functions Estimator [ HL22a ] 
 | 
 
 
 
 
 
 
 
 
 

 
 
 ℐ ^ RenormIF ​ ( z i , z te ) ≔ ℐ ^ IF ​ ( z i , z te ) ∥ ∇ θ ℒ ​ ( z i , θ ( T ) ) ∥ {\widehat{\mathcal{I}}_{\text{RenormIF}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}\coloneqq\frac{{\widehat{\mathcal{I}}_{\textsc{IF}}\left(z_{i}\mathchar 59\relax z_{\text{te}}\right)}}{\lVert\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(T)})}\rVert} 
 
 ( 69 ) 
 
 
 
 
 | 

 Table 4: Influence analysis method formal definitions including equation numbers citations (continued).
 
 
 
 Table 5: Influence Analysis Method Abbreviations : Related methods are grouped together as in Figure 2 . Each method includes its corresponding source reference. 
 
 
 
 
 LOO 
 | 
 
 
 Leave-One-Out Influence [ CW82a , KL17a ] 
 | 

 
 
 
 k k NN LOO 
 | 
 
 
 k k - Nearest-Neighbors Leave-One-Out [ Jia+21b ] 
 | 

 
 
 
 LeafRefit 
 | 
 
 
 Decision Forest Leaf Refitting [ Sha+18c ] 
 | 

 
 
 
 Influence Sketching 
 | 
 
 
 Least Squares Influence Sketching [ Woj+16a ] 
 | 

 
 
 
 Downsampling 
 | 
 
 
 Downsampled Leave-One-Out [ FZ20a ] 
 | 

 
 
 
 C - score 
 | 
 
 
 Consistency Score [ Jia+21c ] 
 | 

 
 
 
 Generative Downsampling 
 | 
 
 
 Downsampling for Generative Density Models [ vW21a ] 
 | 

 
 
 
 SV 
 | 
 
 
 Shapley Value [ Sha53a ] 
 | 

 
 
 
 Interaction Index 
 | 
 
 
 Shapley Interaction Index [ GR99a ] 
 | 

 
 
 
 Shapley-Taylor 
 | 
 
 
 Shapley-Taylor Interaction Index [ SDA20a ] 
 | 

 
 
 
 TMC - Shapley 
 | 
 
 
 Truncated Monte Carlo Shapley [ GZ19a ] 
 | 

 
 
 
 G - Shapley 
 | 
 
 
 Gradient Shapley [ GZ19a ] 
 | 

 
 
 
 k k NN Shapley 
 | 
 
 
 k k - Nearest-Neighbors Shapley [ Jia+19b ] 
 | 

 
 
 
 Beta Shapley 
 | 
 
 
 Beta Distribution-Weighted Shapley Value [ KZ22a ] 
 | 

 
 
 
 Banzhaf Value 
 | 
 
 
 Banzhaf Value [ Ban65a , WJ23a ] 
 | 

 
 
 
 AME 
 | 
 
 
 Average Marginal Effect [ Lin+22a ] 
 | 

 
 
 
 SHAP 
 | 
 
 
 Sh apley A dditive Ex p lanations [ LL17a ] 
 | 

 
 
 
 Neuron Shapley 
 | 
 
 
 Shapley Value-Based Neural Explanations [ GZ20a ] 
 | 

 
 
 
 IF 
 | 
 
 
 Influence Functions [ KL17a ] 
 | 

 
 
 
 FastIF 
 | 
 
 
 Fast Influence Functions [ Guo+21a ] 
 | 

 
 
 
 Arnoldi IF 
 | 
 
 
 Arnoldi-Based Influence Functions [ Sch+22a ] 
 | 

 
 
 
 LeafInfluence 
 | 
 
 
 Decision Forest Leaf Influence [ Sha+18c ] 
 | 

 
 
 
 Group IF 
 | 
 
 
 Group Influence Functions [ Koh+19a ] 
 | 

 
 
 
 Second - Order IF 
 | 
 
 
 Second-Order Group Influence Functions [ BYF20a ] 
 | 

 
 
 
 RelatIF 
 | 
 
 
 Relative Influence (Functions) [ BBD20a ] 
 | 

 
 
 
 Renorm. IF 
 | 
 
 
 Renormalized Influence Functions [ HL22a ] 
 | 

 
 
 
 RP 
 | 
 
 
 Representer Point [ Yeh+18a ] 
 | 

 
 
 
 High Dim. Rep. 
 | 
 
 
 High-Dimensional Representers [ Tsa+23a ] 
 | 

 
 
 
 RPS - LJE 
 | 
 
 
 Representer Point Based on Local Jacobian Expansion [ SWS21a ] 
 | 

 
 
 
 TREX 
 | 
 
 
 T ree-Ensemble Re presenter-Point E x planations [ BHL23a ] 
 | 

 
 
 
 TracIn 
 | 
 
 
 Traced Gradient Descent Influence [ Pru+20a ] 
 | 

 
 
 
 TracInCP 
 | 
 
 
 TracIn Checkpoint [ Pru+20a ] 
 | 

 
 
 
 TracInRP 
 | 
 
 
 TracIn Random Projection [ Pru+20a ] 
 | 

 
 
 
 TracIn - Last 
 | 
 
 
 TracIn Last Layer Only [ Pru+20a , Yeh+22a ] 
 | 

 
 
 
 VAE - TracIn 
 | 
 
 
 Variational Autoencoder TracIn [ KC21a ] 
 | 

 
 
 
 TracInAD 
 | 
 
 
 TracIn Anomaly Detection [ Thi+22a ] 
 | 

 
 
 
 TracInWE 
 | 
 
 
 TracIn Word Embeddings [ Yeh+22a ] 
 | 

 
 
 
 BoostIn 
 | 
 
 
 Boosted (Tree) Influence [ BHL23a ] 
 | 

 
 
 
 GAS 
 | 
 
 
 Gradient Aggregated Similarity [ HL21a , HL22a ] 
 | 

 
 
 
 HyDRA 
 | 
 
 
 Hypergradient Data Relevance Analysis [ Che+21a ] 
 | 

 
 
 
 SGD - Influence 
 | 
 
 
 Stochastic Gradient Descent Influence [ HNM19a ] 
 | 

 
 
 
 
 

## Appendix C Unrolling Gradient Descent Hypergradients

 
 This section provides a formal derivation of how to unroll HyDRA ’s training hypergradients.
We provide this reference for the interested reader to understand unrolling’s complexity. 30 30 
 30 
 
 
 
 Similar complexity would be required if TracIn [ Pru+20a ] were extended to support momentum or adaptive optimization. 
Readers do not need to understand this section’s details to understand how HyDRA relates to other influence analysis methods. 

 
 
 Recall that Eq. ( 61 ) does not describe how the exact contents of batch ℬ ( T ) \mathcal{B}^{(T)} affect unrolling.
Eq. ( 29 ) defines the effect that infinitesimally perturbing the weight of training instance z i z_{i} has on a model’s empirical risk minimizer.
Formally, the perturbed empirical risk is 

 

 
 | 
 ℒ ⁡ ( 𝒟 , θ ) ≔ 1 n ​ ∑ z ∈ 𝒟 ℒ ⁡ ( z , θ ) + ϵ i ​ ℒ ​ ( z i , θ ) ​ , {\mathcal{L}(\mathcal{D};\theta)}\coloneqq\frac{1}{n}\sum_{z\in\mathcal{D}}{\mathcal{L}(z;\theta)}+\epsilon_{i}{\mathcal{L}(z_{i};\theta)}\text{,} | 
 | 
 (70) | 
 

 Observe that ϵ i = − 1 n {\epsilon_{i}=-\frac{1}{n}} is the same as removing instance z i z_{i} from the training set. 

 
 
 We now extend this idea to the effect of ϵ i \epsilon_{i} on a single minibatch.
For any iteration t ∈ [ T ] {t\in{[T]}} ,
Formally, batch ℬ ( t ) \mathcal{B}^{(t)} ’s risk under a training set perturbation by ϵ i \epsilon_{i} is 

 

 
 | 
 ℒ ( ℬ ( t ) ; θ ( t − 1 ) ) ≔ 1 | ℬ ( t ) | ∑ z ∈ ℬ ( t ) ℒ ( z ; θ ( t − 1 ) ) + 𝟙 [ z i ∈ ℬ ( t ) ] ( n ​ ϵ i | ℬ ( t ) | ℒ ( z i ; θ ( t − 1 ) ) ) , {\mathcal{L}(\mathcal{B}^{(t)};\theta^{(t-1)})}\coloneqq\frac{1}{\lvert\mathcal{B}^{(t)}\rvert}\sum_{z\in\mathcal{B}^{(t)}}{\mathcal{L}(z;\theta^{(t-1)})}+{\mathbbm{1}[z_{i}\in\mathcal{B}^{(t)}]}\left(\frac{n\epsilon_{i}}{\lvert\mathcal{B}^{(t)}\rvert}\,{\mathcal{L}(z_{i};\theta^{(t-1)})}\right)\text{,} | 
 | 
 (71) | 
 

 where indicator function 𝟙 [ z i ∈ ℬ ( t ) ] {\mathbbm{1}[z_{i}\in\mathcal{B}^{(t)}]} checks whether instance z i z_{i} is in batch ℬ ( t ) \mathcal{B}^{(t)} .
Observe that
 ℒ ⁡ ( z i , θ ( t − 1 ) ) {\mathcal{L}(z_{i};\theta^{(t-1)})} 
is scaled by n n .
Without this multiplicative factor, then when ϵ i = − 1 n {\epsilon_{i}=-\frac{1}{n}} , z i z_{i} ’s effect is not completely removed from the batch, i.e., 

 

 
 | 
 1 | ℬ ( t ) | ​ ℒ ​ ( z i , θ ( t − 1 ) ) − 1 n ​ | ℬ ( t ) | ​ ℒ ​ ( z i , θ ( t − 1 ) ) ≠ 0 ​ . \frac{1}{\lvert\mathcal{B}^{(t)}\rvert}\,{\mathcal{L}(z_{i};\theta^{(t-1)})}-\frac{1}{n\lvert\mathcal{B}^{(t)}\rvert}\,{\mathcal{L}(z_{i};\theta^{(t-1)})}\neq 0\text{.} | 
 | 
 (72) | 
 

 
 
 Eq. ( 71 )’s gradient w.r.t. θ \theta is 

 

 
 | 
 ∇ θ ℒ ( ℬ ( t ) ; θ ( t − 1 ) ) = 1 | ℬ ( t ) | ∑ z ∈ ℬ ( t ) ∇ θ ℒ ( z ; θ ( t − 1 ) ) + 𝟙 [ z i ∈ ℬ ( t ) ] ( n ​ ϵ i | ℬ ( t ) | ∇ θ ℒ ( z i ; θ ( t − 1 ) ) ) . \nabla_{\theta}{\mathcal{L}(\mathcal{B}^{(t)};\theta^{(t-1)})}=\frac{1}{\lvert\mathcal{B}^{(t)}\rvert}\sum_{z\in\mathcal{B}^{(t)}}\nabla_{\theta}{\mathcal{L}(z;\theta^{(t-1)})}+{\mathbbm{1}[z_{i}\in\mathcal{B}^{(t)}]}\left(\frac{n\epsilon_{i}}{\lvert\mathcal{B}^{(t)}\rvert}\,\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(t-1)})}\right)\text{.} | 
 | 
 (73) | 
 

 HyDRA ’s hypergradients specify that the derivative is taken w.r.t. ϵ i \epsilon_{i} .
Consider the simpler case first where z i ∉ ℬ ( t ) {z_{i}\notin\mathcal{B}^{(t)}} , then 

 

 
 | 
 d d ​ ϵ i ​ [ ∇ θ ℒ ​ ( ℬ ( t ) , θ + ϵ i ( t − 1 ) ) ] \displaystyle\frac{d}{d\epsilon_{i}}[\nabla_{\theta}{\mathcal{L}(\mathcal{B}^{(t)};\theta^{(t-1)}_{+\epsilon_{i}})}] | 
 = 1 | ℬ ( t ) | ​ ∑ z ∈ ℬ ( t ) d d ​ ϵ i ​ ∂ ∂ θ ​ ℒ ​ ( z , θ ( t − 1 ) ) \displaystyle=\frac{1}{\lvert\mathcal{B}^{(t)}\rvert}\sum_{z\in\mathcal{B}^{(t)}}\frac{d}{d\epsilon_{i}}\,\frac{\partial}{\partial\theta}{\mathcal{L}(z;\theta^{(t-1)})} | 
 | 
 (74) | 
 
 
 | 
 | 
 = 1 | ℬ ( t ) | ​ ∑ z ∈ ℬ ( t ) ∂ ∂ θ 2 ​ ℒ ​ ( z , θ ( t − 1 ) ) ​ d ​ θ ( t ) d ​ ϵ i \displaystyle=\frac{1}{\lvert\mathcal{B}^{(t)}\rvert}\sum_{z\in\mathcal{B}^{(t)}}\frac{\partial}{\partial\theta^{2}}{\mathcal{L}(z;\theta^{(t-1)})}\frac{d\theta^{(t)}}{d\epsilon_{i}} | 
 ⊳ \triangleright Chain rule | 
 | 
 (75) | 
 
 
 | 
 | 
 = 1 | ℬ ( t ) | ​ ∑ z ∈ ℬ ( t ) ∇ θ 2 ​ ℒ ​ ( z , θ ( t − 1 ) ) ​ h ~ i ( t − 1 ) ​ . \displaystyle=\frac{1}{\lvert\mathcal{B}^{(t)}\rvert}\sum_{z\in\mathcal{B}^{(t)}}\nabla_{\theta}^{2}{\mathcal{L}(z;\theta^{(t-1)})}\,\widetilde{h}^{(t-1)}_{i}\text{.} | 
 | 
 (76) | 
 

 Note that
 ∇ θ 2 ​ ℒ ​ ( z , θ ( t − 1 ) ) {\nabla_{\theta}^{2}{\mathcal{L}(z;\theta^{(t-1)})}} 
is training instance z z ’s risk Hessian w.r.t. parameters θ ( t − 1 ) \theta^{(t-1)} . 

 
 
 When instance z i z_{i} is in ℬ ( t ) \mathcal{B}^{(t)} ,
unrolling hypergradient h ~ i ( t ) \widetilde{h}^{(t)}_{i} 
requires an additional term.
We derive that term below with similar analysis as when z i ∉ ℬ ( t ) {z_{i}\notin\mathcal{B}^{(t)}} with the addition of using the product rule.
Observe that Eq. ( 78 ) below considers both z i z_{i} ’s risk gradient and Hessian. 

 

 
 | 
 d d ​ ϵ i ​ [ ϵ i ​ ∇ θ ℒ ​ ( z i , θ ( t − 1 ) ) ] \displaystyle\frac{d}{d\epsilon_{i}}[\epsilon_{i}\,\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(t-1)})}] | 
 = ∇ θ ℒ ​ ( z i , θ ( t − 1 ) ) + ϵ i ​ d d ​ ϵ i ​ ∇ θ ℒ ​ ( z i , θ ( t − 1 ) ) \displaystyle=\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(t-1)})}+\epsilon_{i}\frac{d}{d\epsilon_{i}}\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(t-1)})} | 
 ⊳ \triangleright Product rule | 
 | 
 (77) | 
 
 
 | 
 | 
 = ∇ θ ℒ ​ ( z i , θ ( t − 1 ) ) + ϵ i ​ ∇ θ 2 ℒ ​ ( z i , θ ( t − 1 ) ) ​ h ~ i ( t − 1 ) \displaystyle=\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(t-1)})}+\epsilon_{i}\nabla_{\theta}^{2}{\mathcal{L}(z_{i};\theta^{(t-1)})}\widetilde{h}^{(t-1)}_{i} | 
 ⊳ \triangleright Chain rule | 
 | 
 (78) | 
 

 
 
 Combining Eqs. ( 61 ), ( 76 ), and ( 78 ), the hypergradient update rule for vanilla gradient descent without momentum is 

 

 
 | 
 h ~ i ( t ) = \displaystyle\widetilde{h}^{(t)}_{i}=\penalty\ | 
 ( 1 − η ( t ) ​ λ ) ​ h ~ i ( t − 1 ) \displaystyle(1-\eta^{(t)}\lambda)\,\widetilde{h}^{(t-1)}_{i} | 
 | 
 
 
 | 
 | 
 − η ( t ) | ℬ ( t ) | ∑ z ∈ ℬ ( t ) ∇ θ 2 ℒ ( z ; θ ( t − 1 ) ) h ~ i ( t − 1 ) \displaystyle-\frac{\eta^{(t)}}{\lvert\mathcal{B}^{(t)}\rvert}\sum_{z\in\mathcal{B}^{(t)}}\nabla_{\theta}^{2}{\mathcal{L}(z;\theta^{(t-1)})}\,\widetilde{h}^{(t-1)}_{i} | 
 | 
 
 
 | 
 | 
 − η ( t ) ​ n | ℬ ( t ) | 𝟙 [ z i ∈ ℬ ( t ) ] ( ∇ θ ℒ ( z i ; θ ( t − 1 ) ) + ϵ i ∇ θ 2 ℒ ( z i ; θ ( t − 1 ) ) h ~ i ( t − 1 ) ) . \displaystyle-\frac{\eta^{(t)}n}{\lvert\mathcal{B}^{(t)}\rvert}{\mathbbm{1}[z_{i}\in\mathcal{B}^{(t)}]}\left(\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(t-1)})}+\epsilon_{i}\nabla_{\theta}^{2}{\mathcal{L}(z_{i};\theta^{(t-1)})}\,\widetilde{h}^{(t-1)}_{i}\right)\text{.} | 
 | 
 (79) | 
 

 In [ Che+21a ] ’s [ Che+21a ] fast approximation of HyDRA , all Hessians (e.g., ∇ θ 2 ​ ℒ ​ ( z , θ ( t − 1 ) ) {\nabla_{\theta}^{2}{\mathcal{L}(z;\theta^{(t-1)})}} ) in Eq. ( 79 ) are treated as zeros and the associated terms dropped.
The resulting simplified equation, 

 

 
 | 
 h ~ i ( t ) = ( 1 − η ( t ) λ ) h ~ i ( t − 1 ) − η ( t ) ​ n | ℬ ( t ) | 𝟙 [ z i ∈ ℬ ( t ) ] ∇ θ ℒ ( z i ; θ ( t − 1 ) ) , \widetilde{h}^{(t)}_{i}=(1-\eta^{(t)}\lambda)\,\widetilde{h}^{(t-1)}_{i}-\frac{\eta^{(t)}n}{\lvert\mathcal{B}^{(t)}\rvert}{\mathbbm{1}[z_{i}\in\mathcal{B}^{(t)}]}\nabla_{\theta}{\mathcal{L}(z_{i};\theta^{(t-1)})}\text{,} | 
 | 
 (80) | 
 

 is the basis of the fast-approximation update rule in Alg. 4 .