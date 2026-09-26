Instrumental Variables in Causal Inference and Machine Learning: A Survey 
 
 
 

 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: arXiv.org perpetual non-exclusive license
 
 
arXiv:2212.05778v1 [cs.LG] 12 Dec 2022 
 
 

# Instrumental Variables in Causal Inference and Machine Learning: A Survey

 
 
 Anpeng Wu
 
    
 Kun Kuang
 
    
 Ruoxuan Xiong
 
    
 Fei Wu
 † † thanks: 
A. Wu, K. Kuang and F. Wu are with the College of Computer Science and Technology, Zhejiang University, China.
(E-mail: anpwu@zju.edu.cn; kunkuang@zju.edu.cn; wufei@cs.zju.edu.cn).
R. Xiong is with the Department of Quantitative Theory Methods, Emory University, USA.
(E-mail: ruoxuan.xiong@emory.edu).
K. Kuang is the corresponding author.
 

 Abstract 
 
 Causal inference is the process of using assumptions, study designs, and estimation strategies to draw conclusions about the causal relationships between variables based on data. This allows researchers to better understand the underlying mechanisms at work in complex systems and make more informed decisions.
In many settings, we may not fully observe all the confounders that affect both the treatment and outcome variables, complicating the estimation of causal effects. To address this problem, a growing literature in both causal inference and machine learning proposes to use Instrumental Variables (IV).
This paper serves as the first effort to systematically and comprehensively introduce and discuss the IV methods and their applications in both causal inference and machine learning. First, we provide the formal definition of IVs and discuss the identification problem of IV regression methods under different assumptions. Second, we categorize the existing work on IV methods into three streams according to the focus on the proposed methods, including two-stage least squares with IVs, control function with IVs, and evaluation of IVs. For each stream, we present both the classical causal inference methods, and recent developments in the machine learning literature.
Then, we introduce a variety of applications of IV methods in real-world scenarios and provide a summary of the available datasets and algorithms. Finally, we summarize the literature, discuss the open problems and suggest promising future research directions for IV methods and their applications. We also develop a toolkit of IVs methods reviewed in this survey at https://github.com/causal-machine-learning-lab/mliv .

 
 
 
 Index Terms:  Causal Inference, Instrument Variable, Identification.

 
 

## I Introduction 

 
 Nowadays, traditional machine learning and statistical modeling explore correlation patterns among observational variables for data mining and explanatory analysis, and have made amazing achievements in many domains over the past year [ 1 , 2 , 3 ] , especially in speech recognition, image recognition, natural language processing and recommender systems.
As correlation-based algorithms, machine learning techniques gain striking performance from the over-fitting in training distributions under the IID hypothesis that training and testing data are independently sampled from the identical distribution.
However, these models will degrade performance when the test distribution undergoes uncontrolled and unknown distribution shifts [ 4 , 5 ] , i.e., Out of Distribution (OOD) setting.
Essentially, the accuracy drop of current models is mainly caused by the spurious correlation between the features and labels [ 6 , 7 ] , refered as confounding bias 1 1 
 1 
 
 
 
 As introduced in Chapter 3.3 in Causality [ 8 ] , the confounding bias between the feature input and target output can be defined as the bias of causal effect estimation when imbalanced confounders exist. Confounders are common causes of feature and ouput of interest. .
For example, if we do not consider the peak season, we may mistakenly conclude that higher airline ticket prices will lead to higher sales, as the peak season will lead to changes in both prices and demand for airline tickets.
Lack of interpretability, actionability and stability from causality, correlation-based models has poor generalization performance on OOD data [ 5 , 9 ] .

 
 
 To address these issues, machine learning community has tried to develop causality-inspired models by incorporating causal inference paradigms.
The substantive content of these paradigms is to exploit the invariant causal relationships in the data to build models and establish stable and interpretable predictions. Scholkopf and Bengio (2022) [ 5 ] collectively refer to these approaches as structural causal models to answer counterfactual questions and make the model imaginative, that is, models can give a correct prediction in unseen scenarios.
To identify the stable causal effects rather than unstable correlation patterns, the gold standard approach is to perform Randomized Controlled Trials (RCTs), where different treatments are randomly assigned to units. Nevertheless, RCTs are unrealistic in some settings due to ethical and cost issues.
Hence, various methods are developed to draw inference of causal effects from observational datasets, commonly under the unconfoundedness assumption, e.g., propensity score [ 10 , 11 ] , covariate balance [ 12 , 13 , 14 ] , back-door criteria [ 15 , 8 ] and representation learning [ 16 ] .
However, in practice, regardless of the approach that one adopts to control confounding in observational studies, there always exists the possibility of bias, when unmeasured confounders exist.

 
 
 To control for unmeasured confounding, we introduce a third variable, named instrumental variable (IV), which is a cause of input features, has no direct effect on the outcome and does not share common causes with outcome. Using an instrumental variable to identify the hidden (unmeasured) correlation allows one to see the true correlation between the explanatory variable and response variable. For instance, the cost of fuel was used as an instrument in [ 17 ] to estimate the impact of ticket prices on sales. Thus, changes in the cost of fuel create movement in ticket prices that is independent of unmeasured confounders, and this movement is equivalent to randomization for the purposes of causal inference [ 17 ] . See Fig. 1 for a graphical illustration of this example and of the general class of causal graphs that we consider.

 
 
 Fig. 1 : The airline demand example. 
 
 
 Two commonly used estimators for using an instrumental variable to estimate treatment effects are the two stage least squares estimator (2SLS) and the control function estimator (CFN) [ 18 , 19 ] :
(1) 2SLS identifies the probability distribution over the treatment conditioned on the IVs in the treatment regression stage, and regresses the outcome based on the conditional distribution of the treatment (obtained from the treatment stage) in the outcome regression stage.
Based on the adopted model, we divide 2SLS and its variants into three categories as vanilla 2SLS estimator for linear models, sieve estimator and machine learning estimator for non-linear models.
(2) CFN constructs the residual variables (called control functions) in the estimation of treatment from the treatment regression stage, and estimates the outcome from the observed treatment and the residual in the outcome regression stage.
Based on the structural assumption, we divide CFN and its variants into two categories as linear estimator and non-linear estimator.

 
 
 In linearity setting, [ 19 ] show that CFN estimator is a 2SLS estimator with an augmented set (i.e., control function for unmeasured confounders) from instrumental variables. If these augmented variables are valid, then the control function estimator, while less robust than two stage least squares, might be much more precise because it keeps the treatment variables in the second stage [ 18 ] . However, if the augmented variables are not valid, then CFN estimator may be inconsistent, which is common in non-linear models.
Fortunately, with more flexible kernel methods and neural network functions, machine learning methods have developed conditional density estimators, mutual information estimators, representational equilibrium models, etc. that can learn automatically from the data, for IV regression. This relaxes the linearity assumption and allows us to explore more complex causal systems and data.

 
 
 One limitation is that these standard methods and variants of instrumental variable (IV) analysis require a pre-defined strong valid IV. These methods are reliable only when the pre-defined IV only affects the outcome through its strong association with the cause variable of interest, in practice, which is hardly satisfied due to the untestable exclusion association with outcome.
Therefore, in addition to lagged values and prior knowledge [ 20 , 21 , 22 ] ,
researchers usually implement Randomized Controlled Trials (RCTs) to sample a random variable as IV to intervene the received treatments, called intention-to-treat variable, such as Oregon health insurance experiment [ 23 ] and effects of military service on lifetime earnings [ 24 ] , which are too expensive to be universally available.
To save the human effort selecting pre-defined IVs, a growing number of machine learning methods have been proposed to summary existing IV candidates to generate a valid IV representations [ 25 , 26 , 27 , 28 , 29 ] .

 
 
 In this paper, we provide a comprehensive review of the instrumental variable methods under the potential outcome framework. We first introduce the background of the potential outcome framework and instrumental variable, including the basic definitions, corresponding assumptions, and the fundamental problems with their general solutions. To identify the causal effect from instrumental variable, then, we further summarize most of the identification conditions of instrumental variables. Combined with machine learning, subsequently, we introduce the two-stage least-squares
method (2SLS) and the traditional control function method (CFN) to estimate the average treatment effects. To void human effort selecting pre-defined IVs, we also discuss how to use machine learning algorithms to synthesize a summary-IV to plug into IV-based methods.
Then, we provide the related experimental information, including the available datasets that are
commonly adopted in the experiments, and the open-source codes of the above methods.
We also develop a toolkit of IVs methods reviewed in this survey at
 https://github.com/causal-machine-learning-lab/mliv .

 
 
 Machine learning methods provide more flexible network models and conditional moment constraint models, which promote the development of causal inference.
Meanwhile, causal inference also contributes to the development of machine learning methods.
Recently, advent works [ 5 , 7 , 30 ] have revealed the existence and pervasiveness of variant and invariant(stable) features in data-driven algorithms and pointed out that the unstable features can provoke unexpected estimation bias for predictions.
Lacking a causal perspective, machine learning algorithms are prone to exploit subtle statistical correlations present in the training distribution for predictions, which is effective when testing data and training data are independently sampled from identical distribution, i.e., IID hypothesis.
In practice, however, unbalanced samples and attribute-wise imbalance are common across different scenarios [ 30 ] , unlike high-quality experimental data. That means estimators tend to regard high-frequency features from the training as predictive features and view low-frequency stable features as noise, which is unstable in other distributions and even bring additional bias. Due to low-quality observational data and some key unmeasured factors, there still exists a lack of common consensus on underlying invariant features in data-driven algorithms, albeit comprehensive endeavors [ 5 ] .
Therefore, researchers proposed instrumental variable regression to develop causality-inspired models, and the real-world applications that the discussed methods have great potential to benefit are discussed, including the social networks, recommendation system, computer vision, genome project, and domain adaptation as the representative examples.

 
 
 To the best of our knowledge, this is the first paper that provides a comprehensive survey for instrumental variable methods under the potential outcome framework. There also exist several surveys that discuss the causal effect estimation methods under the unconfoundedness assumption, [ 31 , 32 ] introduce.
To summarize, our contributions of this survey are as follows:

 
 • 
 
 Comprehensive review . We provide a comprehensive survey for instrumental variable methods under the potential outcome framework, including identification conditions, two-stage regression methods and control function algorithms.

 

 • 
 
 General setting . When we cannot access a valid instrumental variable directly, we survey a line of IV testing methods and IV synthesis methods.

 

 • 
 
 Abundant resources . In this survey, we list the state-of-art methods, the benchmark data sets, open-source codes, and representative applications.

 

 • 
 
 Reproducible . We integrate the existing resources and codes, and provide a unified interface and parameters to facilitate reproduction.

 

 
 
 
 The rest of the paper is organized as follows. In section 2, we introduce the background of the instrumental variable, including the basic definitions, the assumptions, and the fundamental problems with their general solutions. In section 3, we elaborate the structural assumption for identification of causal effect in IV regression. For estimation, the 2SLS-based methods and CFN-based methods are presented in Section 4 5. In Section 6, we list a series of literature about IV selection and IV synthesis. Afterward, we provide experimental guidelines in section 7, and the typical applications of causal in Section 8. Final, in Section 9, we conclude several IV-based open problems and future directions.

 
 
 Fig. 2 : Outline of the Survey. 
 
 
 

## II Basic of Instrumental Variable 

 
 Although machine learning techniques have provided breakthroughs in statistics, econometrics, epidemiology and related disciplines, they usually suffer from low generalizability, instability, and inexplicability, due to the spurious relationship in which two or more events or variables are associated but not causally related [ 31 , 32 ] . For example, in airplane sales (Example II.1 ), holidays and conferences may confound the causal relationship between prices and sales and introduce additional bias; in hospital (Example II.2 ), comorbidities and physical fitness would distort the causal relationships between the treatments and outcomes. Spurious relationship, deriving from confounders that are common causes of treatments and outcomes, is a common phenomenon in real-world scenarios.
Hence, it is incredibly imperative and highly demanding to eliminate bias from conofunders and develop stable approaches to infer causal effect in observational studies, known as causal inference.

 
 
 Example II.1 . 
 
 In the relationship (Fig. 1 ) between airline ticket prices and sales, prices and demand rise and fall through the seasons, being affected by other events, such as holidays and conferences [ 17 , 33 ] . These events are called confounders that are the common causes of prices (cause variable, often referred to as the ’treatment’) and sales (target outcome). 

 
 
 
 Example II.2 . 
 
 In a hospital for infectious diseases, we study the effect of injection different from taking medicine (treatments) on patients’ cure time (outcomes) from historical data. The patients’ severity level of comorbidities and physical fitness are common causes of the treatments and outcomes, which we define as confounders. We may observe that patients with severe comorbidity have an injection, but the cure time is longer than those with mild comorbidity taking medicine, distorting the causal relationships between the treatments and outcomes. 

 
 
 
 In causal inference, the causal effects of treatment variable on target outcome, often referred to as the treatment effect, can be estimated using control experiments, regression models, matching estimators, re-weighting techniques, and instrumental variable (IV) [ 32 , 31 ] . Among these approaches, the gold standard for treatment effect estimation is to perform Randomized Controlled Trials (RCTs), in which one of two or more treatments (cause variables) are randomly assigned to samples. With enough participants, RCTS would achieve sufficient control over confounding factors and deliver a useful comparison of the treatments studied. Considering the cost and ethical issues [ 34 , 35 ] , fully RCTs are not always feasible in practical. Thus, in observational studies, there are a substantial number of regression models [ 36 , 37 ] , matching estimators [ 38 , 39 ] , and re-weighting techniques are developed to control or adjust the confounders to reduce the confounding bias under unconfoundedness assumption, i.e., all common causes of treatments and outcomes have been observed in data.

 
 
 Nevertheless, in real-world scenarios, it is common that unmeasured confounders exist, violating the unconfounderness assumption and posing a big challenge in estimating treatment effects from observational data.
Regardless of the approach that one adopts to control confounding in observational studies, there always exists the possibility of bias due to unmeasured confounders [ 40 ] , e.g., it is hard to obtain all conferences information in airline demand example (Fig. 1 ). To overcome unmeasured confounder problems, researchers introduced an instrumental variable (fuel costs), an exogenous variable that induces changes in the treatments (prices) but has no independent effect on the outcomes (sales) [ 41 , 17 ] , allowing researchers to uncover the causal effect of the treatment on the outcome under a series of identification assumptions developed by [ 42 , 43 ] .

 
 
 In 1928, the economist Philip Wright (Sewall’s father) introduced IV, IV-estimator, and the equivalent two step least squares estimator, possibly in co-authorship with Sewall Wright, in the context of simultaneous equations in his book The Tariff on Animal and Vegetable Oils 2 2 
 2 
 
 
 
 Based on [ 44 , 45 ] . [ 41 ] .
Later, Haavelmo [ 46 ] and Reiersøl [ 47 ] also applied the similar approach in the context of errors-in-variables models and contributed to the development of IVs unaware of the contributions of the Wrights. In linearity cases, IV estimators implement a two-stage least squares (2SLS) regression analysis for treatment effect estimation: stage 1 performs linear regression from the IVs to the treatments; and stage 2 performs linear regression from the conditional expectation of the treatments (obtained from stage 1) to the outcomes and the corresponding coefficient is used as a measure of treatment effect. To relax linearity assumption, [ 48 , 49 , 42 ] customized a series of identification assumptions for various scenarios, which would be elaborated in Section III .

 
 
 The framework used by IV is essentially similar to potential outcome framework outlined by Rubin [ 50 , 51 ] .
Next, we introduce the notations used in the IV estimator [ 8 ] , and present the main challenges for causal effect estimation as well as general solutions for treatment effect.

 
 

### II-A Definition and Notations 

 
 The Rubin causal model [ 50 , 51 , 52 ] , also known as the potential outcome framework, is a standard approach for IV analysis and treatment effect estimation, named after Donald Rubin.
Similar to [ 32 ] , we define the notations under the potential outcome framework (Fig 3 ).

 
 
 Fig. 3 : The causal framework. 
 
 
 Note, in this paper, we use capital letters for random variables ( X ) (X) , small letters for their values ( x ) (x) , bold letters for vectors/sets of variables ( 𝐗 ) (\mathbf{X}) and their values ( 𝐱 ) (\mathbf{x}) , and calligraphic letters for the spaces where they are defined ( 𝒳 ) (\mathcal{X}) if not explicitly stated. In addition, we use the subscript i i to represent the a variable X i X_{i} belongs to the i i -th unit, and view x i x_{i} as specific value of X i X_{i} .
To simplify notation, we consistently use the shorthand p ⁡ ( x ) p({x}) to represent probabilities or densities p ⁡ ( X = x ) p({X}={x}) . For three random variables X , Y , Z {X},{Y},{Z} , the conditional independence statement ” X {X} is conditionally independent of Y {Y} given Z = z {Z}={z} ” is written as X ⟂ Y | Z {X}\perp{Y}\mid{Z} .

 
 
 Definition II.3 . 
 
 Unit/Sample i i . A unit/sample denotes a single item or a collection of items from a larger whole or group. In the observational data, we can get a subset of n n samples from whole population, and we use the lowercase letter i i to mark each unit, i = 1 , 2 , ⋯ , n i=1,2,\cdots,n .

 
 
 
 Definition II.4 . 
 
 Treatment T T . Treatment refers to an intervention that applies (exposes, or subjects) to a unit. Based on the properties of treatments, we flesh out two cases: (1) in binary treatment cases, different treatment arms T ∈ { 0 , 1 } T\in\{0,1\} denote different intervention (receive treatment or not) and researchers spilt all samples as the treated group ( T = 1 T=1 ) and the control group ( T = 0 T=0 ); (2) in multi-valued or continuous treatment cases, practitioners generalize the binary treatment effects framework and discrete or continuous interventions are used, called dose or dosage, i.e., T ∈ 𝒯 , 𝒯 ⊂ ℝ T\in\mathcal{T},\mathcal{T}\subset\mathbb{R} .

 
 
 
 Definition II.5 . 
 
 Potential outcome Y ⁡ ( T ) Y(T) . Potential outcome is a core element of potential outcome framework, which defines causal effect as a comparison between two states of the world, i.e., “factual” state and “counterfactual” state of the world. In the factual state, the factual outcome Y ⁡ ( T = t ) Y(T=t) is the observed outcome of the treatment T = t T=t that is actually applied; in the counterfactual state, one would question ”what would have happened if another treatment is applied” and imagine that same man takes another treatment T = t ′ T=t^{\prime} and get the the counterfactual outcome { Y ⁡ ( T = t ′ ) } t ′ ≠ t , t ′ ∈ 𝒯 \{Y(T=t^{\prime})\}_{t^{\prime}\not=t,t^{\prime}\in\mathcal{T}} . The above outcomes are called potential outcome Y ⁡ ( T ) Y(T) , which means a proposition stating what would have happened had a potential treatment T T been applied.

 
 
 
 In the observational data, besides the treatment of interest and observed outcome, practitioners would collect other information for each units, which can be separated as pre-treatment variables and the post-treatment variables.

 
 
 Definition II.6 . 
 
 Pre-treatment variables 𝐕 = { Z , X , U , ⋯ } \mathbf{V}=\{Z,X,U,\cdots\} are background variables that occur before the treatment T T is applied and will not be affected by the treatment T T .
Instead, a portion of the pre-treatment variables may be the causes of the treatment, and then researchers will assign treatment based on these variables for obtaining the desired outcome.
 Post-treatment variables 𝐖 = { Y , ⋯ } \mathbf{W}=\{Y,\cdots\} are variables that are affected by the
treatment T T , and these events will occur after the treatment is accepted.
In practice, based on the sequence of events and treatments occurring, Pre-treatment variables and Post-treatment variables are easily distinguished.
In the following sections, we focus on the the pre-treatment variable 𝐕 \mathbf{V} for causal inference, and we refer the terminology variable to the pre-treatment variable unless otherwise specified.

 
 
 
 Both in decision-making applications and in the scientific literature, one tends to choose the level of the treatment to most efficiently pursue their objectives given the constraints they face [ 53 ] . That means that the above pre-treatment variables may affect practitioners’ treatment assignment, leading to unbalanced data distributions across different levels of the treatment. Recently, several works [ 8 , 54 ] shows such a unbalanced data would produce a spurious association, called confounding, because it tends to confound our judgment and to bias our estimate of the causal effect studied. For example, high-frequency but unrelated daily products are likely to be considered to exhibit correlation.
Thus, [ 8 ] claims that if a third variable X X that influences both T T and Y Y , the real and stable causal relationship may be confounded and spurious association would introduce additional bias for stable prediction. Such a variable is then called a confounder.

 
 
 Definition II.7 . 
 
 Confounders 𝐗 \mathbf{X} 𝐔 \mathbf{U} .
In the causal relationship graph (Fig 3 ), confounders are some special pre-treatment variables, which simultaneously affect the treatment assignment and the outcome being studied ( T , Y T,Y ) so that the effect estimation may not reflect the actual relationship ( T → Y T\rightarrow Y ) between the variables under study. In this paper, we denote 𝐗 \mathbf{X} by observable confounders in observational data. For missing key variables in the record that may confound the relationship between the variables being studied ( T , Y T,Y ), we refer to them as unmeasured confounders 𝐔 \mathbf{U} . Confounders 𝐗 \mathbf{X} 𝐔 \mathbf{U} are both pre-treatment variables 𝐕 \mathbf{V} .

 
 
 
 Causal Inference . After introducing the key terminologies and definition for causal inference, the causal effect can be quantitatively defined using the above definitions.
In observational dataset, the treatment can be either binary, multi-valued or continuous.
For notational simplicity, we uniformly use Y ⁡ ( T = t ) Y(T=t) to represent the potential outcome with treatment T = t T=t . Then, the definition of the treatment effect is the difference Y ⁡ ( T = t ) − Y ⁡ ( T = 0 ) Y(T=t)-Y(T=0) , which can be measured at the population, subgroup, and individual levels.

 
 
 Definition II.8 . 
 
 Average Treatment Effect (ATE) .

 

 
 | 
 ATE ​ ( t ) = 𝔼 ⁡ [ Y ⁡ ( T = t ) − Y ⁡ ( T = 0 ) ] , \displaystyle\textbf{ATE}(t)=\mathbb{E}[Y(T=t)-Y(T=0)], | 
 | 
 (1) | 
 

 
 
 
 Definition II.9 . 
 
 Conditional Average Treatment Effect (CATE) .

 

 
 | 
 CATE ​ ( t , 𝐱 ) = 𝔼 ⁡ [ Y ⁡ ( T = t ) − Y ⁡ ( T = 0 ) ∣ 𝐗 = 𝐱 ] , \displaystyle\textbf{CATE}(t,\mathbf{x})=\mathbb{E}[Y(T=t)-Y(T=0)\mid\mathbf{X}=\mathbf{x}], | 
 | 
 (2) | 
 

 which has an another name: 
 Individual Treatment Effect (ITE) .

 

 
 | 
 ITE i ​ ( t ) = Y i ​ ( T = t ) − Y i ​ ( T = 0 ) . \displaystyle\textbf{ITE}_{i}(t)=Y_{i}(T=t)-Y_{i}(T=0). | 
 | 
 (3) | 
 

 
 
 
 

### II-B Instruments and Main Challenges 

 
 In many circumstances, running Randomized Controlled Trials (RCTs) are not possible due to ethical or cost concerns.
In the presence of unmeasured confounders 𝐔 \mathbf{U} , estimating treatment effect from observational data is challenging due to following reasons:

 
 • 
 
 Counterfactual . We only realize the outcome y i ​ ( T = t i ) y_{i}(T=t_{i}) with a specific treatment value t i t_{i} applied to individual i i , but cannot obtain the counterfactual outcomes y i ​ ( T ≠ t i ) y_{i}(T\not=t_{i}) that would potentially happened if a different treatment option was assigned.

 

 • 
 
 Imbalanced observed Covariates . The treatments are typically not assigned at random and the covariate distributions can be quite different between different treatment arms. Some high-frequency but unrelated variables 𝐗 \mathbf{X} would confound the causal effect of treatment on outcome of interest.

 

 • 
 
 Imbalanced Unmeasured Covariates . Even if we control all observed variables and adjust confounding differences from observational covariates, unmeasured key variables and differences 𝐔 \mathbf{U} may distort the causal relationships in the observational data.

 

 
 
 
 Hence, to overcome unmeasured confounder problems in observational data where the treatments are non-random assigned, researchers introduced an instrumental variable, an exogenous variable that induces changes in the treatments but has no direct effect on the outcomes, to estimate treatment effect. The instrumental variable is defined as follows:

 
 
 Definition II.10 . 
 
 Instrument Variable Z Z is an exogenous variable that affects the treatment T T , but does not directly affect the outcome Y Y , as shown in Fig 3 . Besides, an valid instrument variable satisfies the following three restrictions: 
 Relevance: Z Z is a cause of T T , i.e., ℙ ⁡ ( T ∣ Z ) ≠ ℙ ⁡ ( T ) \mathbb{P}(T\mid Z)\neq\mathbb{P}(T) . 
 Exclusion: Z Z does not directly affect the outcome Y Y , i.e., Z ⟂ Y | T , 𝐗 , 𝐔 Z\perp Y\mid T,\mathbf{X},\mathbf{U} . 
 Independent: Z Z is independent of all confounders, including 𝐗 \mathbf{X} and 𝐔 \mathbf{U} , i.e., Z ⟂ 𝐗 , 𝐔 Z\perp\mathbf{X},\mathbf{U} 

 
 
 
 Nevertheless, IV methods are reliable when the pre-defined IV is a valid IV that only affects the outcome through its strong association with treatment options, called exclusion assumption. Besides, they also need some strong structural assumptions, e.g., linear models. To sum up, IV regression has the following main challenges:

 
 • 
 
 Strict Structural Assumption . Even if the instrument Z Z satisfies three restrictions in the definition, at least one structural assumption is required to identify the treatment effect of T T on Y Y [ 49 , 42 , 43 ] . The most common structural assumption is the linearity assumption, which requires that the causal relationships between all variables are linear.

 

 • 
 
 Untestable Exclusion and Independent .
We do not have access to the unmeasured confounders in observational data, and therefore we cannot test for independence between instrumental and unmeasured variables. In addition, we cannot test whether instrumental variables have additional causality on the outcome variable.

 

 • 
 
 Invalid and Weak IV . In instrumental variables regression, the instruments are called weak IV if their correlation with the endogenous regressors is close to zero, or invalid IV if there is a direct effect or a hidden common cause between the instrument and the outcome. Due to untestable exclusion and independent restrictions, the predefined hand-made IVs could be weak or erroneous by violating the conditions of valid IVs.

 

 
 Although IV has been used in tons of empirical papers, these thorny facts hinder the further application of the IV-based methods for treatment effect estimation. Recently, several works devote to relax or resolving these restrictions.

 
 
 For structural assumptions, a substantial number of IV works have been developed to relax the unconfoundedness assumption and the identification assumption for various scenarios [ 55 , 56 , 49 , 43 , 57 , 58 , 59 ] , which would be elaborated in Section III .
Although Exclusion and Independent are not testable, thanks to machine learning algorithms, researchers have developed Summary IV methods to automatically synthesize valid strong instrumental variables from a candidate set of instrumental variables [ 26 , 60 , 29 ] . Unless otherwise stated, in the following, we assume that the instrumental variables obtained from the observational data are valid strong IVs.

 
 
 Remark II.11 . 
 
 Angrist, Imbens and Rubin [ 49 , 42 ] abandoned the effort to draw inference for the overall average effect, and focused on sub-populations for which the average effect could be identified, the so-called compliers. In binary cases, where instrument variables ( Z Z ) are different intervention assignments and treatment variables are individuals’ respond to assignments ( T ⁡ ( Z ) T(Z) ), four different compliance types defined by the pair of values ( T ⁡ ( Z = 0 ) , T ⁡ ( Z = 1 ) T(Z=0),T(Z=1) ) [ 53 ] :

 

 
 | 
 i ∈ { OPEN n ​  (never  −  taker  )  if  ​ T i ​ ( 0 ) = T i ​ ( 1 ) = 0 OPEN c ​  (complier  )  if  ​ T i ​ ( 0 ) = 0 , T i ​ ( 1 ) = 1 d ⁡ (  defier  )  if  ​ T i ​ ( 0 ) = 1 , T i ​ ( 1 ) = 0 OPEN a ​  (always  −  taker  )  if  ​ T i ​ ( 0 ) = T i ​ ( 1 ) = 1 \displaystyle i\in\begin{cases}n\text{ (never }-\text{ taker }) \text{ if }T_{i}(0)=T_{i}(1)=0\\
c\text{ (complier }) \text{ if }T_{i}(0)=0,T_{i}(1)=1\\
d(\text{ defier }) \text{ if }T_{i}(0)=1,T_{i}(1)=0\\
a\text{ (always }-\text{ taker }) \text{ if }T_{i}(0)=T_{i}(1)=1\end{cases} | 
 | 
 (4) | 
 

 The local average treatment effect or complier average causal effect is identified:

 
 
 Definition II.12 . 
 
 Local Average Treatment Effect (LATE) .

 

 
 | 
 LATE = 𝔼 ⁡ [ Y i ​ ( T = 1 ) − Y i ​ ( T = 0 ) | i ∈ c ​ o ​ m ​ p ​ l ​ i ​ e ​ r ] \displaystyle\textbf{LATE}=\mathbb{E}[Y_{i}(T=1)-Y_{i}(T=0)|i\in complier] | 
 | 
 (5) | 
 

 
 
 
 
 Under the monotonicity assumption III.5 3 3 
 3 
 
 
 
 Monotonicity Assumption would be elaborated in Section III . , the proportion of compliers can be obtained from the remainder:

 

 
 | 
 P ⁡ ( i ∈ c ) = 1 − P ⁡ ( T = 1 | Z = 0 ) − P ⁡ ( T = 0 | Z = 1 ) , \displaystyle P(i\in c)=1-P(T=1|Z=0)-P(T=0|Z=1), | 
 | 
 (6) | 
 

 Thus, monotonicity assumption is a sufficient identification assumption for LATE estimation. In this paper, we focus on reviewing more general identification assumption for ATE/CATE estimation (and thus also for LATE).

 
 
 

### II-C General Solutions for Treatment Effect 

 
 In IV Regression, there are two main frameworks for causal inference, i.e., Two-stage Least Squares (2SLS) [ 17 , 61 , 62 , 63 ] and Control Function Method (CFN) [ 64 , 65 , 66 , 67 ] .
The former uses the conditional expectation of the treatment variable to estimate the causal effect, while the latter recovers the unmeasured confounders to estimate the causal effect.
In linearity assumption, we let Z = [ z 1 , z 2 , ⋯ , z n ] ′ Z=[z_{1},z_{2},\cdots,z_{n}]^{\prime} and T = [ t 1 , t 2 , ⋯ , t n ] ′ T=[t_{1},t_{2},\cdots,t_{n}]^{\prime} and assume that the observational data is generated by:

 

 
 | 
 T = Z ​ α + ϵ , Y = T ​ β + ϵ , \displaystyle{T}={Z}\alpha+\epsilon,{Y}={T}\beta+\epsilon, | 
 | 
 (7) | 
 

 where ϵ ∼ 𝒩 ⁡ ( 0 , 1 ) \epsilon\sim\mathcal{N}(0,1) , and { α , β } \{\alpha,\beta\} are the coefficients in the linear equation. Besides, our target is to predict the causal parameter β \beta as treatment effect estimation.
Details of the implementation of 2SLS and CFN are as follows.

 
 

#### II-C 1 Two-stage Least Squares (2SLS)

 
 2SLS identifies the probability distribution over the treatment conditioned on the IVs in the treatment regression stage, and regresses the outcome based on the conditional distribution of the treatment (obtained from the treatment stage) in the outcome regression stage. In linearity models, the predicted values from 2SLS are obtained: 
 Stage 1: Regress treatments T {T} on instruments Z {Z} :

 

 
 | 
 α ^ = arg ⁡ min ⁡ ∑ i = 1 n α ⁡ ( t i − α ​ z i ) 2 = ( Z ′ ​ Z ) − 1 ​ Z ′ ​ T \displaystyle\hat{\alpha}=\arg\min_{\alpha}\sum_{i=1}^{n}(t_{i}-\alpha z_{i})^{2}=\left(Z^{\prime}Z\right)^{-1}Z^{\prime}T | 
 | 
 (8) | 
 

 Let P Z = Z ​ ( Z ′ ​ Z ) − 1 ​ Z ′ P_{Z}=Z\left(Z^{\prime}Z\right)^{-1}Z^{\prime} , and the predicted treatment is:

 

 
 | 
 T ^ = Z ​ α ^ = Z ​ ( Z ′ ​ Z ) − 1 ​ Z ′ ​ T = P Z ​ T \displaystyle\widehat{T}=Z\hat{\alpha}=Z\left(Z^{\prime}Z\right)^{-1}Z^{\prime}T=P_{Z}T | 
 | 
 (9) | 
 

 Stage 2: Regress Y {Y} on the predicted values T ^ \widehat{T} from stage 1:

 

 
 | 
 β ^ = arg ⁡ min ⁡ ∑ i = 1 n β ⁡ ( y i − β ​ t ^ i ) 2 \displaystyle\hat{\beta}=\arg\min_{\beta}\sum_{i=1}^{n}(y_{i}-\beta\hat{t}_{i})^{2} | 
 | 
 (10) | 
 

 which gives:

 

 
 | 
 β 2SLS = ( X ′ ​ P Z T ​ P Z ​ T ) − 1 ​ T ′ ​ P Z ​ Y \displaystyle\beta_{\text{2SLS}}=\left(X^{\prime}P_{Z}^{T}P_{Z}T\right)^{-1}T^{\prime}P_{Z}Y | 
 | 
 (11) | 
 

 This method requires a strong linear relationship between the instrumental variables and the treatment variables, which will be not applicable if the unmeasured noise ϵ \epsilon is large. More nonlinear variants of 2SLS are detailed in Section IV .

 
 
 

#### II-C 2 Control Function Method (CFN)

 
 CFN constructs the residual variables (called control functions) in the estimation of treatment from the treatment regression stage, and estimates the outcome from the true treatment and the residual in the outcome regression stage. In linearity models, the predicted values from CFN are obtained: 
 Stage 1: Regress treatments T {T} on instruments Z {Z} :

 

 
 | 
 α ^ = arg ⁡ min ⁡ ∑ i = 1 n α ⁡ ( t i − α ​ z i ) 2 = ( Z ′ ​ Z ) − 1 ​ Z ′ ​ T \displaystyle\hat{\alpha}=\arg\min_{\alpha}\sum_{i=1}^{n}(t_{i}-\alpha z_{i})^{2}=\left(Z^{\prime}Z\right)^{-1}Z^{\prime}T | 
 | 
 (12) | 
 

 and the predicted residuals is:

 

 
 | 
 ϵ ^ = T − Z ​ α ^ = T − P Z ​ T \displaystyle\hat{\epsilon}=T-Z\hat{\alpha}=T-P_{Z}T | 
 | 
 (13) | 
 

 Stage 2: Regress Y {Y} on the predicted residuals from stage 1:

 

 
 | 
 β ^ , β ^ ϵ = arg ⁡ min ⁡ ∑ i = 1 n β , β ϵ ⁡ ( y i − β ​ t ^ i − β ϵ ​ ϵ ^ i ) 2 \displaystyle\hat{\beta},\hat{\beta}_{\epsilon}=\arg\min_{\beta,\beta_{\epsilon}}\sum_{i=1}^{n}(y_{i}-\beta\hat{t}_{i}-\beta_{\epsilon}\hat{\epsilon}_{i})^{2} | 
 | 
 (14) | 
 

 which gives:

 

 
 | 
 ( β CFN , β ϵ ) = ( ( T , ϵ ) T ​ ( T , ϵ ) ) − 1 ​ ( T , ϵ ) T ​ Y \displaystyle(\beta_{\text{CFN}},\beta_{\epsilon})=((T,\epsilon)^{T}(T,\epsilon))^{-1}(T,\epsilon)^{T}Y | 
 | 
 (15) | 
 

 where ( A , B ) (A,B) means a concatenate of vectors/matrices A A and B B . This method is valid for large unmeasured confounding bias. More nonlinear variants of CFN are detailed in the Section V .

 
 
 
 
 

## III Identification 

 
 In this section, we will discuss the structural assumptions or restrictions on data and model for precise inference to be possible, i.e., identification. After obtaining an infinite number of observations from population, if it is theoretically possible to learn the true values of a model’s underlying parameters, then the model is identifiable. Otherwise, it is not non-identifiable.
Even if the instrument satisfies IV’s three constraints, we might not be able to identify causal effects unless there are additional structural assumptions [ 68 , 55 , 56 , 49 ] .

 
 
 Example III.1 . 
 
 Non-identifiability Without loss of generality, we take continuous treatment cases as an example and assume an unmeasured confound U U is a random variable from a standard normal distribution 𝒩 ⁡ ( 0 , 1 ) \mathcal{N}(0,1) : 

 

 
 | 
 T = Z ​ U , Y = T ​ U , U ∼ 𝒩 ⁡ ( 0 , 1 ) . \displaystyle T=ZU,Y=TU,U\sim\mathcal{N}(0,1). | 
 | 
 (16) | 
 

 Due to the unmeasured confounders, the relationships between Z Z T T , Z Z Y Y and T T Y Y can no longer be accurately regressed by any parametric or non-parametric models. 

 
 
 
 In econometric program evaluation, to address the non-identifiability problem, a standard method is to build structural equation model with linearity assumptions for treatment effect identification [ 69 , 70 ] .

 
 
 Assumption III.2 . 
 
 Linearity Assumptions [ 70 , 42 ] . 

For experimental or observational data, let Y Y be the observed outcome of interest, let T T be the observed treatment, and let Z Z be the observed instrument variable. For continuous treatment cases, a standard structural assumption for the identification of treatment effect would have the form: 

 

 
 | 
 T = α 0 + α 1 ​ Z + ϵ T , \displaystyle T=\alpha_{0}+\alpha_{1}Z+\epsilon_{T}, | 
 | 
 (17) | 
 
 
 | 
 Y = β 0 + β 1 ​ T + ϵ Y , \displaystyle Y=\beta_{0}+\beta_{1}T+\epsilon_{Y}, | 
 | 
 (18) | 
 

 where { α 0 , α 1 , β 0 , β 1 } \{\alpha_{0},\alpha_{1},\beta_{0},\beta_{1}\} are corresponding scalar coefficients, as well as ϵ T \epsilon_{T} and ϵ Y \epsilon_{Y} are additive confounding effect from unmeasured confounders, which influence both the treatment and the outcome ( ϵ T ⟂̸ ϵ Y \epsilon_{T}\not\perp\epsilon_{Y} ). In the model β 1 \beta_{1} represents the causal effect of T T on Y Y . 
 For binary treatment, the structural Eq. ( 17 ) could be reformulated as: 

 

 
 | 
 T = 1 { α 0 + α 1 Z + ϵ T ≥ 0 } , \displaystyle T=1\{\alpha_{0}+\alpha_{1}Z+\epsilon_{T}\geq 0\}, | 
 | 
 (19) | 
 

 where 1 ​ { ⋅ } 1\{\cdot\} is a indicator function. 

 
 
 
 Following IV’s independent restriction, we have Z ⟂ ϵ T , ϵ T Z\perp\epsilon_{T},\epsilon_{T} . Then, the absence of Z Z in Eq. 18 denotes that any effect of Z Z on Y Y must be through an effect of Z Z on T T in Eq. 17 / 19 . Thus, Z Z can be considered as a strong and valid IV for treatment effect estimation (i.e., β 1 \beta_{1} ). The IV estimator is defined as
the ratio of sample covariance [ 71 , 42 ] . For binary instrument and treatment cases,

 

 
 | 
 β ^ 1 \displaystyle\hat{\beta}_{1} | 
 = \displaystyle= | 
 cov ⁡ ( Y , Z ) / cov ⁡ ( T , Z ) \displaystyle\operatorname{cov}\left(Y,Z\right)/\operatorname{cov}\left(T,Z\right) | 
 | 
 (20) | 
 
 
 | 
 | 
 = \displaystyle= | 
 𝔼 ⁡ ( Y ​ Z ) / 𝔼 ⁡ ( Z ) − 𝔼 ⁡ ( Y ⁡ ( 1 − Z ) ) / 𝔼 ⁡ ( 1 − Z ) 𝔼 ⁡ ( T ​ Z ) / 𝔼 ⁡ ( Z ) − 𝔼 ⁡ ( T ⁡ ( 1 − Z ) ) / 𝔼 ⁡ ( 1 − Z ) \displaystyle{\frac{\mathbb{E}(YZ)/\mathbb{E}(Z)-\mathbb{E}(Y\left(1-Z\right))/\mathbb{E}\left(1-Z\right)}{\mathbb{E}(TZ)/\mathbb{E}(Z)-\mathbb{E}(T\left(1-Z\right))/\mathbb{E}\left(1-Z\right)}} | 
 | 
 (21) | 
 

 For continuous instrument and treatment cases,

 

 
 | 
 β ^ 1 \displaystyle\hat{\beta}_{1} | 
 = \displaystyle= | 
 cov ⁡ ( Y , Z ) / cov ⁡ ( T , Z ) \displaystyle\operatorname{cov}\left(Y,Z\right)/\operatorname{cov}\left(T,Z\right) | 
 | 
 (22) | 
 
 
 | 
 | 
 = \displaystyle= | 
 𝔼 ⁡ ( Y − ( 𝔼 ⁡ ( Y ) ) ​ ( Z − 𝔼 ⁡ ( Z ) ) ) 𝔼 ⁡ ( T − ( 𝔼 ⁡ ( T ) ) ​ ( Z − 𝔼 ⁡ ( Z ) ) ) \displaystyle\frac{\mathbb{E}(Y-(\mathbb{E}(Y))(Z-\mathbb{E}(Z)))}{\mathbb{E}(T-(\mathbb{E}(T))(Z-\mathbb{E}(Z)))} | 
 | 
 (23) | 
 

 
 
 Without the linearity assumption in real-world scenarios, even if the instrument satisfies IV’s three constraints, we might not be able to identify causal effects.
However, the linearity assumption (Eq. 17 , 18 19 ) have
not found widespread use in real-world scenarios, and exists only in theoretical studies. Besides, the apparently unreproducible experimental results also prevent the application of instrumental variable parameter models under the linear assumption [ 72 ] .
To relax the linear assumption and avoid parametric evaluation models, researchers had devoted to establishing conditions that guarantee nonparametric identification of treatment effects in observational studies, i.e. identification without relying on functional form restrictions or distributional assumptions [ 49 , 42 , 43 , 73 , 59 ] .

 
 
 Identifiability . Briefly speaking, even if the IVs are valid, further assumptions are required for the identification of treatment effect. Following criticism of
parametric evaluation models [ 72 ] , instead of sticking to the average treatment effects in a population of interest, researchers use some weaker assumptions to identify the average effect for the compliers sub-population, i.e., Local Average Treatment Effect (LATE). Sufficient assumptions for this include: Constant/Additive Treatment Effect [ 68 ] , Zero Probability on Some IV Value [ 55 , 56 ] and Monotonicity [ 49 , 42 , 73 ] .
However, under these assumptions, we can identify the average treatment effect for the group of compliers but not for the specific members.

 
 
 It was not until 2003, when Newey and Powell (2003) gave the identification and estimation results for nonparametric conditional moment restrictions, that practitioners started to focus on the identifiability of the structure function for outcome, i.e., Conditional Average Treatment Effect (CATE) or ATE [ 43 , 57 , 17 ] . Subsequently, some more general homogeneity assumptions are developed one after another [ 74 , 75 , 76 , 58 , 59 ] for ATE(CATE).
In the econometrics literature, Homogeneity Assumption is a more general version than Monotonicity Assumption and Additive Noise Assumption [ 59 , 77 ] .

 
 
 Based on assumption for LATE or CATE, we are going to elaborate on these assumptions as LATE Identification Assumptions, CATE Identification Assumptions, and More General Assumptions.

 
 

### III-A LATE Identification 

 
 As illustrated in Remark  II.11 , Angrist, Imbens and Rubin [ 49 , 42 ] abandoned the effort to draw inference for the overall average effect, and focused on sub-populations for which the average effect could be identified, the so-called compliers. Four different compliance types are defined in Eq.  4 . The compliers means that the groups of people who can be induced to change treatments by assigning different instruments. In this section, we will list some sufficient assumptions or conditions for identifying treatment effect of
the compliers, as follows.

 
 
 Assumption III.3 . 
 
 Constant Treatment Effect [ 68 ] . 

To prevent above problem, one condition is the treatment effect is constant: α = Y ⁡ ( 1 ) − Y ⁡ ( 0 ) \alpha=Y(1)-Y(0) for any unit i i . Then 𝔼 ⁡ [ Y ∣ Z = z ] − 𝔼 ⁡ [ Y ∣ Z = w ] \mathbb{E}[Y\mid Z=z]-\mathbb{E}[Y\mid Z=w] is equal to: 

 

 
 | 
 | 
 | 
 𝔼 ⁡ [ Y ∣ Z = z ] − 𝔼 ⁡ [ Y ∣ Z = w ] \displaystyle\mathbb{E}[Y\mid Z=z]-\mathbb{E}[Y\mid Z=w] | 
 | 
 (24) | 

 
 | 
 | 
 = \displaystyle= | 
 𝔼 ⁡ [ T ⁡ ( z ) ​ Y ​ ( 1 ) + ( 1 − T ⁡ ( z ) ) ⋅ Y ⁡ ( 0 ) ∣ Z = z ] \displaystyle\mathbb{E}[T(z)Y(1)+(1-T(z))\cdot Y(0)\mid Z=z] | 
 | 

 
 | 
 | 
 − \displaystyle- | 
 𝔼 ⁡ [ T ⁡ ( w ) ​ Y ​ ( 1 ) + ( 1 − T ⁡ ( w ) ) ⋅ Y ⁡ ( 0 ) ∣ Z = w ] \displaystyle\mathbb{E}[T(w)Y(1)+(1-T(w))\cdot Y(0)\mid Z=w] | 
 | 

 
 | 
 | 
 = \displaystyle= | 
 Y ⁡ ( 1 ) ​ P ​ [ T = 1 ∣ Z = z ] + Y ⁡ ( 0 ) ​ P ​ [ T = 0 ∣ Z = z ] \displaystyle Y(1)P[T=1\mid Z=z]+Y(0)P[T=0\mid Z=z] | 
 | 

 
 | 
 | 
 − \displaystyle- | 
 Y ⁡ ( 1 ) ​ P ​ [ T = 1 ∣ Z = w ] + Y ⁡ ( 0 ) ​ P ​ [ T = 0 ∣ Z = w ] \displaystyle Y(1)P[T=1\mid Z=w]+Y(0)P[T=0\mid Z=w] | 
 | 

 
 | 
 | 
 = \displaystyle= | 
 [ Y ⁡ ( 1 ) ​ P ​ ( z ) + Y ⁡ ( 0 ) ​ ( 1 − P ⁡ ( z ) ) ] \displaystyle[Y(1)P(z)+Y(0)(1-P(z))] | 
 | 

 
 | 
 | 
 − \displaystyle- | 
 [ Y ⁡ ( 1 ) ​ P ​ ( w ) + Y ⁡ ( 0 ) ​ ( 1 − P ⁡ ( w ) ) ] \displaystyle[Y(1)P(w)+Y(0)(1-P(w))] | 
 | 

 
 | 
 | 
 = \displaystyle= | 
 ( Y ⁡ ( 1 ) − Y ⁡ ( 0 ) ) ​ ( P ⁡ ( z ) − P ⁡ ( w ) ) \displaystyle(Y(1)-Y(0))(P(z)-P(w)) | 
 | 

 
 | 
 | 
 = \displaystyle= | 
 α ⁡ ( P ⁡ ( z ) − P ⁡ ( w ) ) \displaystyle\alpha(P(z)-P(w)) | 
 | 
 

 
 
 
 Assumption III.4 . 
 
 Zero Probability [ 55 , 56 ] . 

A second approach is to assume the existence of some value of the instrument, w ∈ 𝒲 w\in\mathcal{W} , such that the probability of participation conditional on that value is equal to zero, i.e., P ⁡ ( w ) = 0 P(w)=0 . Then P ⁡ ( T ⁡ ( z ) − T ⁡ ( w ) = − 1 ) = 0 P(T(z)-T(w)=-1)=0 : 

 

 
 | 
 | 
 | 
 𝔼 ⁡ [ Y ∣ Z = z ] − 𝔼 ⁡ [ Y ∣ Z = w ] \displaystyle\mathbb{E}[Y\mid Z=z]-\mathbb{E}[Y\mid Z=w] | 
 | 
 (25) | 

 
 | 
 | 
 = \displaystyle= | 
 
 
 [ Y ⁡ ( 1 ) ​ P ​ ( z ) + Y ⁡ ( 0 ) ​ ( 1 − P ⁡ ( z ) ) ] − [ Y ⁡ ( 1 ) ​ P ​ ( w ) + Y ⁡ ( 0 ) ​ ( 1 − P ⁡ ( w ) ) ] [Y(1)P(z)+Y(0)(1-P(z))]-[Y(1)P(w)+Y(0)(1-P(w))] 

 | 
 | 

 
 | 
 | 
 = \displaystyle= | 
 
 
 Y ⁡ ( 1 ) ​ P ​ ( z ) + Y ⁡ ( 0 ) ​ ( 1 − P ⁡ ( z ) ) − Y ⁡ ( 0 ) Y(1)P(z)+Y(0)(1-P(z))-Y(0) 

 | 
 | 

 
 | 
 | 
 = \displaystyle= | 
 
 
 Y ⁡ ( 1 ) ​ P ​ ( z ) − Y ⁡ ( 0 ) ​ P ​ ( z ) Y(1)P(z)-Y(0)P(z) 

 | 
 | 

 
 | 
 | 
 = \displaystyle= | 
 
 
 P ⁡ ( z ) ​ 𝔼 ​ [ Y ⁡ ( 1 ) − Y ⁡ ( 0 ) ∣ T ⁡ ( z ) = 1 ] P(z)\mathbb{E}[Y(1)-Y(0)\mid T(z)=1] 

 | 
 | 
 

 To identify the causal effect, we need to know at least one value w w of 𝒲 \mathcal{W} . 

 
 
 
 Let A A be an indicator for the event Z ∉ 𝒲 Z\not\in\mathcal{W} , i.e., A = 𝟙 { Z ∉ 𝒲 } A=\mathbbm{1}\{Z\not\in\mathcal{W}\} . Then:

 

 
 | 
 | 
 | 
 𝔼 ⁡ [ Y ∣ A = 0 ] \displaystyle\mathbb{E}[Y\mid A=0] | 
 | 
 (26) | 

 
 | 
 | 
 = \displaystyle= | 
 𝔼 ⁡ [ Y ∣ T ⁡ ( w ) = 1 ] ⋅ P ⁡ ( w ) \displaystyle\mathbb{E}[Y\mid T(w)=1]\cdot P(w) | 
 | 

 
 | 
 | 
 + \displaystyle+ | 
 𝔼 ⁡ [ Y ∣ T ⁡ ( w ) = 0 ] ⋅ ( 1 − P ⁡ ( w ) ) \displaystyle\mathbb{E}[Y\mid T(w)=0]\cdot(1-P(w)) | 
 | 

 
 | 
 | 
 = \displaystyle= | 
 𝔼 ⁡ [ Y ⁡ ( 0 ) ] , w ∈ 𝒲 . \displaystyle\mathbb{E}[Y(0)],w\in\mathcal{W}. | 
 | 
 

 and

 

 
 | 
 | 
 | 
 𝔼 ⁡ [ Y ∣ A = 1 ] \displaystyle\mathbb{E}[Y\mid A=1] | 
 | 
 (27) | 

 
 | 
 | 
 = \displaystyle= | 
 𝔼 ⁡ [ Y 0 ∣ A = 1 ] + P ⁡ ( z ) ​ 𝔼 ​ [ Y 1 − Y 0 ∣ T ⁡ ( z ) = 1 ] \displaystyle\mathbb{E}[Y_{0}\mid A=1]+P(z)\mathbb{E}[Y_{1}-Y_{0}\mid T(z)=1] | 
 | 

 
 | 
 | 
 = \displaystyle= | 
 𝔼 ⁡ [ Y 0 ] + P ⁡ ( z ) ​ 𝔼 ​ [ Y 1 − Y 0 ∣ T ⁡ ( z ) = 1 ] , z ∉ 𝒲 . \displaystyle\mathbb{E}[Y_{0}]+P(z)\mathbb{E}[Y_{1}-Y_{0}\mid T(z)=1],z\not\in\mathcal{W}. | 
 | 
 

 Since we can estimate P ⁡ ( z ) P(z) , 𝔼 ⁡ [ Y ∣ A = 0 ] \mathbb{E}[Y\mid A=0] and 𝔼 ⁡ [ Y ∣ A = 1 ] \mathbb{E}[Y\mid A=1] and we know z ∉ 𝒲 z\not\in\mathcal{W} , then we can identify ATE:

 

 
 | 
 𝔼 ⁡ [ Y 1 − Y 0 ∣ T ⁡ ( z ) = 1 ] = 𝔼 ⁡ [ Y ∣ A = 1 ] − 𝔼 ⁡ [ Y ∣ A = 0 ] P ⁡ ( z ) . \displaystyle\mathbb{E}[Y_{1}-Y_{0}\mid T(z)=1]=\frac{\mathbb{E}[Y\mid A=1]-\mathbb{E}[Y\mid A=0]}{P(z)}. | 
 | 
 (28) | 
 

 
 
 Assumption III.5 . 
 
 Monotonicity [ 49 , 42 , 73 ] . 
For all possible value of instrument, z z and w w , either T ⁡ ( z ) ≥ T ⁡ ( w ) T(z)\geq T(w) for any unit i i , or T ⁡ ( z ) ≤ T ⁡ ( w ) T(z)\leq T(w) for any unit i i . Without loss of generality, the assumption is satisfied with T ⁡ ( z ) ≥ T ⁡ ( w ) T(z)\geq T(w) : 

 

 
 | 
 | 
 | 
 𝔼 ⁡ [ Y ∣ Z = z ] − 𝔼 ⁡ [ Y ∣ Z = w ] \displaystyle\mathbb{E}[Y\mid Z=z]-\mathbb{E}[Y\mid Z=w] | 
 | 

 
 | 
 | 
 = \displaystyle= | 
 
 
 ( P ⁡ ( z ) − P ⁡ ( w ) ) ⋅ 𝔼 ⁡ [ Y ⁡ ( 1 ) − Y ⁡ ( 0 ) ∣ T ⁡ ( z ) − T ⁡ ( w ) = 1 ] (P(z)-P(w))\cdot\mathbb{E}[Y(1)-Y(0)\mid T(z)-T(w)=1] 

 | 
 | 
 

 
 
 
 

### III-B ATE/CATE Identification 

 
 For LATE models, assumption III.3 , III.4 or III.5 are sufficient for identification [ 68 , 55 , 56 , 49 ] . Besides, in a linear outcome process, where the outcome process is a sum of the causal effect and zero-mean noise, zero covariance between the instruments Z Z and unmeasured disturbances (confounders) 𝐔 \mathbf{U} , suffices for identify the causal effect [ 43 , 57 ] .
In a nonparametric IV (NPIV) model for CATE, the moment restrictions that unmeasured disturbances has conditional mean zero given instruments is a necessary restriction for identification, i.e., 𝔼 ⁡ [ 𝐔 ∣ Z ] = 0 \mathbb{E}[\mathbf{U}\mid Z]=0 .

 
 
 Assumption III.6 . 
 
 Additive Noise Assumption / Separability Assumption [ 43 , 17 ] . In the parametric/nonparametric model (Eq. ( 30 )), the identification/uniqueness of g ^ ​ ( 𝐗 , T ) \hat{g}(\mathbf{X},T) is equivalent to the nonexistence of any function δ ⁡ ( 𝐗 , T ) := g ⁡ ( 𝐗 , T ) − g ^ ​ ( 𝐗 , T ) ≠ 0 \delta(\mathbf{X},T):=g(\mathbf{X},T)-\hat{g}(\mathbf{X},T)\not=0 such that 𝔼 ⁡ [ δ ⁡ ( 𝐗 , T ) ∣ Z ] = 0 \mathbb{E}[\delta(\mathbf{X},T)\mid Z]=0 . 

 
 
 
 In the nonparametric setting, the relationship between the outcome process and reduced form belongs a 1st Fredholm integral equation [ 78 ] and leads an ill-posed inverse problem [ 43 ] . Considering the identification of a general nonparametric model:

 

 
 | 
 Y = g ⁡ ( 𝐗 , T ) + 𝐔 , 𝔼 ⁡ [ 𝐔 ∣ Z ] = 𝔼 ⁡ [ 𝐔 ] = 0 . \displaystyle Y=g(\mathbf{X},T)+\mathbf{U},\mathbb{E}[\mathbf{U}\mid Z]=\mathbb{E}[\mathbf{U}]=0. | 
 | 
 (30) | 
 

 where g ⁡ ( ⋅ ) g(\cdot) denotes a true, unknown structural function of interest. For a consistency estimation, [ 43 , 57 , 17 ] identified the causal effect as the solution of an integral equation:

 

 
 | 
 𝔼 [ Y ∣ Z , 𝐗 ] \displaystyle\mathbb{E}[Y\mid Z,\mathbf{X}] | 
 = \displaystyle= | 
 𝔼 [ g ( 𝐗 , T ) ∣ Z , 𝐗 ] + 𝔼 [ 𝐔 ∣ 𝐗 ] \displaystyle\mathbb{E}[g(\mathbf{X},T)\mid Z,\mathbf{X}]+\mathbb{E}[\mathbf{U}\mid\mathbf{X}] | 
 | 
 (31) | 

 
 | 
 | 
 = \displaystyle= | 
 ∫ [ g ⁡ ( 𝐗 , T ) + 𝔼 ⁡ [ 𝐔 ∣ 𝐗 ] ] ​ 𝑑 F ​ ( T ∣ Z , 𝐗 ) \displaystyle\int\left[g(\mathbf{X},T)+\mathbb{E}[\mathbf{U}\mid\mathbf{X}]\right]dF(T\mid Z,\mathbf{X}) | 
 | 

 
 | 
 | 
 = \displaystyle= | 
 ∫ g ^ ​ ( 𝐗 , T ) ​ 𝑑 F ​ ( T ∣ Z , 𝐗 ) \displaystyle\int\hat{g}(\mathbf{X},T)dF(T\mid Z,\mathbf{X}) | 
 | 
 

 where F F denotes the conditional cumulative distribution function of T T given { Z , 𝐗 } \{Z,\mathbf{X}\} . Given two observable functions 𝔼 [ Y ∣ Z , 𝐗 ] \mathbb{E}[Y\mid Z,\mathbf{X}] and F ⁡ ( T ∣ Z , 𝐗 ) F(T\mid Z,\mathbf{X}) , g ^ ​ ( 𝐗 , T ) \hat{g}(\mathbf{X},T) is the solution of the inverse problem. Then, we can identify ATE:

 

 
 | 
 ATE = g ^ ​ ( 𝐗 , T ) − g ^ ​ ( 𝐗 , 0 ) = g ⁡ ( 𝐗 , T ) − g ⁡ ( 𝐗 , 0 ) . \displaystyle\text{ATE}=\hat{g}(\mathbf{X},T)-\hat{g}(\mathbf{X},0)=g(\mathbf{X},T)-g(\mathbf{X},0). | 
 | 
 (32) | 
 

 Therefore, [ 43 , 17 ] characterized identification of structural functions as completeness of certain conditional distributions 𝔼 ⁡ [ 𝐔 ∣ Z ] = 0 \mathbb{E}[\mathbf{U}\mid Z]=0 .

 
 
 

### III-C More General Assumptions 

 
 In the econometrics literature [ 59 , 77 ] , Homogeneity Assumption is a more general version than Monotonicity Assumption and Additive Noise Assumption. Next, we describe two general Homogeneity Assumptions and the No Effect Modification Assumption. Note that the previous assumptions (except the Monotonicity Assumption) can be viewed as a special case of the Homogeneity Assumptions.

 
 
 Assumption III.7 . 
 
 Homogeneous Instrument-Treatment Association [ 75 , 76 , 59 ] : The association between the IV and the treatment is homogeneous in the different level of unmeasured confounders, i.e., 𝔼 [ T | Z = a , 𝐔 ] − 𝔼 [ T | Z = b , 𝐔 ] = 𝔼 [ T | Z = a ] − 𝔼 [ T | Z = b ] \mathbb{E}[T|Z=a,\mathbf{U}]-\mathbb{E}[T|Z=b,\mathbf{U}]=\mathbb{E}[T|Z=a]-\mathbb{E}[T|Z=b] . 

 
 
 
 Assumption III.8 . 
 
 Homogeneous Treatment-Outcome Association [ 74 , 58 , 59 ] : The association between the treatment and the outcome is homogeneous in the different level of unmeasured confounders, i.e., 𝔼 [ Y | T = a , 𝐔 ] − 𝔼 [ Y | T = b , 𝐔 ] = 𝔼 [ Y | T = a ] − 𝔼 [ Y | T = b ] \mathbb{E}[Y|T=a,\mathbf{U}]-\mathbb{E}[Y|T=b,\mathbf{U}]=\mathbb{E}[Y|T=a]-\mathbb{E}[Y|T=b] . 

 
 
 
 Meanwhile, No effect modification of the treatment effect (NEM) is weaker than Homogeneity Assumptions, but may not be plausible in many instances [ 58 , 59 ] .

 
 
 Assumption III.9 . 
 
 No Effect Modification [ 58 , 76 , 59 ] : The unmeasured confounders 𝐔 \mathbf{U} would not modify the causal effect of T T on Y Y . 

 
 
 
 
 

## IV Two-Stage Least Squares 

 
 Fig. 4 : Key milestones in the development of instrumental variable. 
 
 
 As discussed above, we know that when there are unmeasured confounder in the data, the causality obtained by direct regression (Ordinary Least Squares, OLS) will be distorted. In this section, we will explain the inconsistency of ordinary least squares for causal effect and introduce some typical IV-based methods for consistency estimation, i.e., two-stage least squares and its variants with machine learning.
As a classical statistical method for causal effect estimation, the two-stage least squares performs linear regression from the instruments Z Z to the treatments T T in stage 1, and fit the counterfactual outcome function to predict the outcomes Y Y from the conditional expectation of the treatments 𝔼 ⁡ [ T ∣ Z ] \mathbb{E}[{T\mid Z}] (obtained from stage 1) in stage 2.

 
 
 Under the nonparametric identification of ATE/CATE in observational studies (See Section III-B ), based on traditional linear methods and advanced non-linear variants, as shown in Fig. 5 , we divide two-stage least squares and its variants into three categories: (1) Vanilla 2SLS and Wald Estimator for linear models (OLS is not applicable to causal effects); (2) Sieve estimator for non-linear models [ 43 , 79 ] ; (3) Machine Learning for further estimation. There are four main research lines from machine learning estimator, incliuding: Kernel-based estimator [ 61 , 62 ] , Deep-based estimator [ 17 , 80 , 81 ] , Moment conditions estimator [ 63 , 82 ] and confounder balance estimator.Finally we will summarize the limitations of these approaches and future works.

 
 
 Fig. 5 : Categorization of 2SLS and variants. 
 
 
 Fig. 6 : The causal diagram in different cases. 
 
 

### IV-A 2SLS Estimator 

 
 Followed the linear Gaussian assumption in traditional 2SLS, without intercept for notational convenience, we assume that the observational data is generated by:

 

 
 | 
 T \displaystyle T | 
 = \displaystyle= | 
 Z ​ α + f ⁡ ( 𝐔 ) = Z ​ α + ϵ T , \displaystyle Z\alpha+f(\mathbf{U})=Z\alpha+\epsilon_{T}, | 
 | 
 (33) | 
 
 
 | 
 Y \displaystyle Y | 
 = \displaystyle= | 
 T ​ β + g ⁡ ( 𝐔 ) = T ​ β + ϵ U , \displaystyle T\beta+g(\mathbf{U})=T\beta+\epsilon_{U}, | 
 | 
 (34) | 
 

 where { α , β } \{\alpha,\beta\} are the coefficients in the linear equation. Without interactions between unmeasured confounders and treatment, we can represent the effect of infinitely many unmeasured causes { f ⁡ ( 𝐔 ) , g ⁡ ( 𝐔 ) } \{f(\mathbf{U}),g(\mathbf{U})\} as an additive noise { ϵ T , ϵ Y } \{\epsilon_{T},\epsilon_{Y}\} regardless of how they interact among themselves, where f ⁡ ( ⋅ ) f(\cdot) and g ⁡ ( ⋅ ) g(\mathbf{\cdot}) can be any continuous functions. Besides, the instrumental variable Z Z is correlated with the independent variable T T and uncorrelated with the unmeasured confounder U U .
We call Eqs. ( 33 ) and ( 34 ) the structural equations or primary equations, especially, Eq. ( 33 ) is treatment-assignment function and Eq. ( 34 ) is counterfactual function in continuous setting. The corresponding causal diagram is shown in Fig. 6 (c).

 
 

#### IV-A 1 Inconsistency of Ordinary Least Squares

 
 In the presence of unmeasured confounder in the observational data, the causality obtained by direct regression (Ordinary Least Squares, OLS) will be distorted.
In causal inference, the goal of regression analysis is to estimate the conditional expectation function 𝔼 ⁡ [ Y ∣ d ​ o ​ ( T ) ] \mathbb{E}[Y\mid do(T)] , i.e., to recover coefficient β \beta from the scalar regression model 4 4 
 4 
 
 
 
 d ​ o ​ ( ⋅ ) do(\cdot) denotes do-operation which manipulates the value of treatments as T in experimental study. . Recall that ordinary least squares (OLS) solves for β ^ \hat{\beta} by minimize the sum of squared errors:

 

 
 | 
 min β ( Y − T ​ β ) ′ ​ ( Y − T ​ β ) . \displaystyle\min_{\beta}\quad(Y-T\beta)^{\prime}(Y-T\beta). | 
 | 
 (35) | 
 

 The first-order condition is T ′ ​ ( Y − T ​ β ^ ) = T ′ ​ ϵ ^ Y = 0 T^{\prime}(Y-T\hat{\beta})=T^{\prime}\hat{\epsilon}_{Y}=0 . The regression results are reliable only when T T and 𝐔 \mathbf{U} are independent ℙ ⁡ ( T ∣ 𝐔 ) = ℙ ⁡ ( T ) \mathbb{P}(T\mid\mathbf{U})=\mathbb{P}(T) , i.e., ϵ T = f ⁡ ( 𝐔 ) ≡ 0 {\epsilon}_{T}=f(\mathbf{U})\equiv 0 in dose-assignment function Eq. ( 33 ), as shown in Fig. 6 (a). Then the treatment variable T T affects the outcome variable Y Y only through T ​ β {T}\beta , and there is no association between T T and 𝐔 \mathbf{U} .

 
 
 But in real-world scenarios, there may exist some unmeasured confounders 𝐔 \mathbf{U} that are the common causes of the treatment T T and the outcome Y Y , i.e., ϵ T = f ⁡ ( 𝐔 ) ≢ 0 {\epsilon}_{T}=f(\mathbf{U})\not\equiv 0 in dose-assignment function Eq. ( 33 ), as shown in Fig. 6 (b). Now there is an association between T T and 𝐔 \mathbf{U} , i.e., ϵ T {\epsilon}_{T} . Then, the true model is believed to have T ′ ​ 𝐔 ≠ 0 T^{\prime}{\mathbf{U}}\not=0 in the presence of unmeasured confounders 𝐔 \mathbf{U} .

 
 
 Based on non-zero 𝐔 \mathbf{U} - T T association ( f ⁡ ( 𝐔 ) ≢ 0 f(\mathbf{U})\not\equiv 0 ), from Eq. ( 34 ) there is a direct effect ( T ​ β T\beta ) and an indirect effect via 𝐔 \mathbf{U} affecting T T which in turn generates an additional false correlation term between T T and Y Y . If we directly perform OLS regression (Eq. ( 35 )) to estimate the causal effect, OLS will combine these two effects to give a bias result, i.e., β ^ OLS ≠ β \hat{\beta}_{\text{OLS}}\not=\beta . In this case, the coefficient on the treatment T T is given:

 

 
 | 
 β ^ OLS \displaystyle\hat{\beta}_{\text{OLS}} | 
 = \displaystyle= | 
 ( T ′ ​ T ) − 1 ​ T ′ ​ Y = ( T ′ ​ T ) − 1 ​ T ′ ​ ( T ​ β + ϵ Y ) \displaystyle\left(T^{\prime}T\right)^{-1}T^{\prime}Y=\left(T^{\prime}T\right)^{-1}T^{\prime}(T\beta+{\epsilon}_{Y}) | 
 | 
 (36) | 

 
 | 
 | 
 = \displaystyle= | 
 β + ( T ′ ​ T ) − 1 ​ T ′ ​ ϵ Y \displaystyle\beta+\left(T^{\prime}T\right)^{-1}T^{\prime}{\epsilon}_{Y} | 
 | 
 
 
 | 
 β ^ OLS \displaystyle\hat{\beta}_{\text{OLS}} | 
 = \displaystyle= | 
 d ​ Y d ​ T = β + d ​ ϵ Y d ​ T \displaystyle\frac{dY}{dT}=\beta+\frac{d{\epsilon}_{Y}}{dT} | 
 | 
 (37) | 
 

 
 
 Therefore, the OLS estimates the bias effect β + d ​ ϵ Y / d ​ T \beta+{d{\epsilon}_{Y}}/{dT} rather than the true effect β \beta . In a conclusion, the OLS estimator is biased and inconsistent for causal inference in the presence of unmeasured confoudners. Therefore, the researchers proposed a two-stage regression method to eliminate confounding bias d ​ ϵ Y / d ​ T {d{\epsilon}_{Y}}/{dT} [ 41 ] .

 
 
 

#### IV-A 2 Two-Stage Least Squares

 
 In the linear Gaussian model (Eqs. ( 33 ) and ( 34 )) discussed above, assumption III.5 is automatically satisfied. Thus, we can identify the causal effect via 2SLS using IVs Z Z , which is not related to 𝐔 \mathbf{U} , i.e., Z ′ ​ ϵ T = Z ′ ​ ϵ Y = 0 Z^{\prime}{\epsilon}_{T}=Z^{\prime}{\epsilon}_{Y}=0 .

 
 
 The Treatment Regression Stage: in stage 1 of 2SLS, estimator regresses treatment T T from IVs Z Z :

 

 
 | 
 α ^ = ( Z ′ ​ Z ) − 1 ​ Z ′ ​ T = ( Z ′ ​ Z ) − 1 ​ Z ′ ​ ( Z ​ α + ϵ T ) = α \displaystyle\hat{\alpha}=\left(Z^{\prime}Z\right)^{-1}Z^{\prime}T=\left(Z^{\prime}Z\right)^{-1}Z^{\prime}(Z\alpha+{\epsilon}_{T})=\alpha | 
 | 
 (38) | 
 
 
 | 
 T ^ = 𝔼 ⁡ [ T ∣ Z ] = Z ​ α \displaystyle\hat{T}=\mathbb{E}[T\mid Z]=Z\alpha | 
 | 
 (39) | 
 

 According to the IVs’ unconfounded assumption, there is no association between T ^ \hat{T} and 𝐔 \mathbf{U} . Hence, as shown in Fig. 6 (d) the first-order condition T ^ ′ ​ ( Y − T ​ β ) = T ^ ′ ​ ϵ Y = 0 \hat{T}^{\prime}(Y-T{\beta})=\hat{T}^{\prime}{\epsilon}_{Y}=0 is satisfied.

 
 
 The Outcome Regression Stage: in stage 2 of 2SLS, estimator regresses the outcome Y Y based on the conditional expectation of the treatment T ^ \hat{T} (obtained from stage 1):

 

 
 | 
 β ^ 2SLS = ( T ^ ′ ​ T ^ ) − 1 ​ T ^ ′ ​ Y = ( T ^ ′ ​ T ^ ) − 1 ​ T ^ ′ ​ ( T ^ ​ β + ϵ Y ) = β \displaystyle\hskip-8.0pt\hat{\beta}_{\text{2SLS}}=\left(\hat{T}^{\prime}\hat{T}\right)^{-1}\hat{T}^{\prime}Y=\left(\hat{T}^{\prime}\hat{T}\right)^{-1}\hat{T}^{\prime}(\hat{T}\beta+{\epsilon}_{Y})=\beta | 
 | 
 (40) | 
 
 
 | 
 Y ^ = 𝔼 ⁡ [ Y ∣ T ^ ] = T ^ ​ β \displaystyle\hat{Y}=\mathbb{E}[Y\mid\hat{T}]=\hat{T}\beta | 
 | 
 (41) | 
 

 Then, we can get the counterfactual function by replacing T ^ \hat{T} with T T :

 

 
 | 
 Y ^ = 𝔼 ⁡ [ Y ∣ d ​ o ​ ( T ) ] = T ​ β . \displaystyle\hat{Y}=\mathbb{E}[Y\mid do(T)]=T\beta. | 
 | 
 (42) | 
 

 
 
 

#### IV-A 3 Wald Estimator

 
 In 1940s, the economist Wald proposed the wald estimator for a non-continuous IV case where the instruments Z Z is a binary instrument [ 83 ] .
Denote the sub-sample averages of Y Y and T T by Y ¯ 1 \bar{Y}_{1} and T ¯ 1 \bar{T}_{1} when Z = 1 Z=1 and by Y ¯ 0 \bar{Y}_{0} and T ¯ 0 \bar{T}_{0} when Z = 0 Z=0 . Then, we can get the derivatives:

 

 
 | 
 d ​ Y d ​ Z = Y ¯ 1 − Y ¯ 0 \displaystyle\frac{dY}{dZ}=\bar{Y}_{1}-\bar{Y}_{0} | 
 | 
 (43) | 
 
 
 | 
 d ​ T d ​ Z = T ¯ 1 − T ¯ 0 \displaystyle\frac{dT}{dZ}=\bar{T}_{1}-\bar{T}_{0} | 
 | 
 (44) | 
 

 therefore, the causal effect is:

 

 
 | 
 β ^ Wald = Y ¯ 1 − Y ¯ 0 T ¯ 1 − T ¯ 0 . \displaystyle\hat{\beta}_{\text{Wald}}=\frac{\bar{Y}_{1}-\bar{Y}_{0}}{\bar{T}_{1}-\bar{T}_{0}}. | 
 | 
 (45) | 
 

 
 
 Wald Estimator is a binary IV version of 2SLS, under linearity assumption.

 
 
 
 

### IV-B Sieve Estimator 

 
 Fig. 7 : The causal diagram in more general cases. 
 
 
 To satisfy the identification conditions, 2SLS simplifies the IV estimation problem by assuming linear models. To generalize 2SLS to the nonlinear setting, motivated by the works [ 84 , 85 ] on sieve estimation, [ 43 , 57 ] propose a non-parametric two-stage basis expansion approach, called Sieve NPIV, with uniform convergence rates [ 79 ] . Under a more general case ((Fig.  7 )), Sieve IV defines an appropriate finite dictionary of basis functions (Hermite polynomial or a set of indicator functions) for the treatments regression T T and the outcomes regression Y Y with instruments Z Z and observed covariates 𝐗 \mathbf{X} , and specifies the number of basis expansion functions [ 86 , 87 ] . Specifically, under homogeneity assumption III.8 , Sieve IV focus on identification of the models:

 

 
 | 
 T = f ⁡ ( Z , 𝐗 ) + ϵ T , \displaystyle T=f(Z,\mathbf{X})+\epsilon_{T}, | 
 | 
 (46) | 
 
 
 | 
 Y = g ⁡ ( T , 𝐗 ) + ϵ Y , \displaystyle Y=g(T,\mathbf{X})+\epsilon_{Y}, | 
 | 
 (47) | 
 

 where g ⁡ ( ⋅ ) g(\cdot) denotes the true, unknown structural function of interest, ϵ T \epsilon_{T} and ϵ T \epsilon_{T} are joint errors from unobserved variables 𝐔 \mathbf{U} , and the unmeasured confounders ϵ T \epsilon_{T} and ϵ T \epsilon_{T} are additive noise, that is independent with the instruments Z Z , i.e., 𝔼 ⁡ [ ϵ T ∣ Z ] = 𝔼 ⁡ [ ϵ Y ∣ Z ] = 0 \mathbb{E}[\epsilon_{T}\mid Z]=\mathbb{E}[\epsilon_{Y}\mid Z]=0 .

 
 
 Based on these sieve bases, Sieve IV implement a two-stage regression to estimate causal effect.

 
 
 Sieve IV . 
 Formally, Sieve IV estimates the structure function using an appropriate finite dictionary of basis functions and we can reformulate the structure function as:

 

 
 | 
 T = ∑ i = 1 d Z ∑ j = 1 d X α i , j ​ ϕ i ​ ( Z ) ​ ξ j ​ ( 𝐗 ) + ϵ T , \displaystyle T=\sum_{i=1}^{d^{Z}}\sum_{j=1}^{d^{X}}\alpha_{i,j}\phi_{i}(Z)\xi_{j}(\mathbf{X})+\epsilon_{T}, | 
 | 
 (48) | 
 
 
 | 
 Y = ∑ k = 1 d T ∑ j = 1 d X β k , j ​ ψ k ​ ( T ) ​ ξ j ​ ( 𝐗 ) + ϵ Y , \displaystyle Y=\sum_{k=1}^{d^{T}}\sum_{j=1}^{d^{X}}\beta_{k,j}\psi_{k}(T)\xi_{j}(\mathbf{X})+\epsilon_{Y}, | 
 | 
 (49) | 
 

 where { ϕ i } i = 1 d Z \{\phi_{i}\}_{i=1}^{d^{Z}} is the sieve basis for IVs Z Z with degree d Z d^{Z} , { ξ i } i = 1 d X \{\xi_{i}\}_{i=1}^{d^{X}} is the sieve basis for confounders 𝐗 \mathbf{X} with degree d X d^{X} , { ϕ k } k = 1 d T \{\phi_{k}\}_{k=1}^{d^{T}} is the sieve basis for treatments T T with degree d T d^{T} , and { α i , j , β i , j } \{\alpha_{i,j},\beta_{i,j}\} are the corresponding coefficients. Each of the ϕ i \phi_{i} is a function from 𝒵 \mathcal{Z} into ℝ \mathbb{R} , each of the ξ j \xi_{j} is a function from 𝒳 \mathcal{X} into ℝ \mathbb{R} , and each of the ψ k \psi_{k} is a function from 𝒯 \mathcal{T} into ℝ \mathbb{R} .

 
 
 Then the goal of Sieve IV is to estimate:

 

 
 | 
 CATE ​ ( x , t ) = ∑ k = 1 d T ∑ j = 1 d X β k , j ​ ξ j ​ ( x ) ​ [ ψ k ​ ( T = t ) − ψ k ​ ( T = 0 ) ] . \displaystyle\text{CATE}(x,t)=\sum_{k=1}^{d^{T}}\sum_{j=1}^{d^{X}}\beta_{k,j}\xi_{j}(x)[\psi_{k}(T=t)-\psi_{k}(T=0)]. | 
 | 
 (50) | 
 

 
 
 In the treatment regression stage , different than 2SLS, Sieve IV regresses each of the treatment basis functions ( 𝔼 ⁡ [ ψ k ​ ( T ) ∣ ϕ i ​ ( Z ) ​ ξ j ​ ( 𝐗 ) ] \mathbb{E}[\psi_{k}(T)\mid\phi_{i}(Z)\xi_{j}(\mathbf{X})] ) on the basis features { ϕ i ​ ( Z ) ​ ξ j ​ ( 𝐗 ) } \{\phi_{i}(Z)\xi_{j}(\mathbf{X})\} rather than the conditional expectation treatment distribution ( 𝔼 [ T ∣ Z , 𝐗 ] \mathbb{E}[T\mid Z,\mathbf{X}] ).

 

 
 | 
 α ^ = argmin α ​ MSE ​ ( ψ ⁡ ( ∑ i = 1 d Z ∑ j = 1 d X [ α i , j ​ ϕ i ​ ( Z ) ​ ξ j ​ ( 𝐗 ) ] ) , ψ ⁡ ( T ) ) . \displaystyle\hat{\alpha}=\text{argmin}_{\alpha}\text{MSE}(\psi(\sum_{i=1}^{d^{Z}}\sum_{j=1}^{d^{X}}[\alpha_{i,j}\phi_{i}(Z)\xi_{j}(\mathbf{X})]),\psi(T)). | 
 | 
 (51) | 
 

 
 
 In the outcome regression stage , Sieve IV estimates the expectation outcome onto these estimated functions 𝔼 ⁡ [ ψ k ​ ( T ) ∣ ϕ i ​ ( Z ) ​ ξ j ​ ( 𝐗 ) ] \mathbb{E}[\psi_{k}(T)\mid\phi_{i}(Z)\xi_{j}(\mathbf{X})] (obtained by the stage 1) and bases ξ j ​ ( 𝐗 ) \xi_{j}(\mathbf{X}) to identify the coefficients β k , j \beta_{k,j} .

 

 
 | 
 β ^ = argmin β ​ MSE ​ ( ∑ k = 1 d T ∑ j = 1 d X [ β k , j ​ ψ k ​ ( T ) ​ ξ j ​ ( 𝐗 ) ] , Y ) . \displaystyle\hat{\beta}=\text{argmin}_{\beta}\text{MSE}(\sum_{k=1}^{d^{T}}\sum_{j=1}^{d^{X}}[\beta_{k,j}\psi_{k}(T)\xi_{j}(\mathbf{X})],Y). | 
 | 
 (52) | 
 

 
 
 In the two-stage regression of Sieve IV, the challenge is how to define an appropriate series basis functions [ 79 ] . Thus, recent works [ 61 , 62 ] introduce machine learning algorithm to obtain the basis functions and estimate causal effect.

 
 
 

### IV-C Machine Learning Estimator 

 
 To implement further estimation, as shown in Fig. 5 , there are four main research lines from machine learning estimator (Fig.  5 ), incliuding: Kernel-based Estimator [ 79 , 61 , 62 ] , Deep-based methods [ 17 , 80 , 81 ] , Moment conditions methods [ 63 , 82 ] and Confounder Balanced Estimator [ 54 ] .

 
 

#### IV-C 1 Kernel-based Estimator

 
 Motivated by Sieve NPIV [ 79 ] and predictive state representation models (PSRs) [ 88 ] and [ 89 ] , [ 61 ] proposes kernel instrumental variable regression (KernelIV) to model relations among Z Z , 𝐗 \mathbf{X} , T T , and Y Y as nonlinear functions in reproducing kernel Hilbert spaces (RKHSs) [ 90 ] , and prove the consistency of KernelIV.

 
 
 Fig. 8 : The Structural Function of KernelIV. 
 
 
 Kernel IV . 
 As shown in Fig. 8 , KernelIV defines two measurable positive definite kernels k 𝒯 : 𝒯 × 𝒯 → ℝ k_{\mathcal{T}}:\mathcal{T}\times\mathcal{T}\rightarrow\mathbb{R} and k 𝒵 : 𝒵 × 𝒵 → ℝ k_{\mathcal{Z}}:\mathcal{Z}\times\mathcal{Z}\rightarrow\mathbb{R} corresponding to scalar-valued RKHSs ℋ 𝒯 \mathcal{H}_{\mathcal{T}} and ℋ 𝒵 \mathcal{H}_{\mathcal{Z}} :

 

 
 | 
 ψ : 𝒯 → ℋ 𝒯 , t ↦ k 𝒯 ​ ( t , ⋅ ) , ϕ : 𝒵 → ℋ 𝒵 , z ↦ k 𝒵 ​ ( z , ⋅ ) \displaystyle\psi:\mathcal{T}\rightarrow\mathcal{H}_{\mathcal{T}},t\mapsto k_{\mathcal{T}}(t,\cdot),\quad\phi:\mathcal{Z}\rightarrow\mathcal{H}_{\mathcal{Z}},z\mapsto k_{\mathcal{Z}}(z,\cdot) | 
 | 
 (53) | 
 

 where ψ \psi and ϕ \phi are the basis functions of 𝒵 \mathcal{Z} and 𝒯 \mathcal{T} .
In this section, 𝒵 \mathcal{Z} means the the horizontal concatenation of IVs 𝒵 \mathcal{Z} and confounders 𝒳 \mathcal{X} , and 𝒯 \mathcal{T} means the the horizontal concatenation of treatments 𝒯 \mathcal{T} and confounders 𝒳 \mathcal{X} , i.e., 𝒵 = 𝒵 ⊕ 𝒳 \mathcal{Z}=\mathcal{Z}\oplus\mathcal{X} and 𝒯 = 𝒯 ⊕ 𝒳 \mathcal{T}=\mathcal{T}\oplus\mathcal{X} . Then, KernelIV reformulates the problem as:

 

 
 | 
 e ∈ E , E : ℋ 𝒵 → ℋ 𝒯 , \displaystyle e\in E,E:\mathcal{H}_{\mathcal{Z}}\rightarrow\mathcal{H}_{\mathcal{T}}, | 
 | 
 (54) | 
 
 
 | 
 h ∈ H , H : ℋ 𝒯 → ℋ 𝒴 . \displaystyle h\in H,H:\mathcal{H}_{\mathcal{T}}\rightarrow\mathcal{H}_{\mathcal{Y}}. | 
 | 
 (55) | 
 

 
 
 In stage 1, KernelIV learns a conditional mean embedding to model the relations between 𝒵 \mathcal{Z} and 𝒯 \mathcal{T} by two kernel functions ψ \psi and ϕ \phi and a conditional expectation operator E E :

 

 
 | 
 ϕ ^ ​ ( T ) = μ ⁡ ( Z ) = e ⁡ ( ψ ⁡ ( Z ) ) = 𝔼 ⁡ [ ϕ ⁡ ( T ) ∣ Z ] , \displaystyle\hat{\phi}(T)=\mu({Z})=e(\psi({Z}))=\mathbb{E}[\phi(T)\mid Z], | 
 | 
 (56) | 
 
 
 | 
 ψ ⁡ ( Z ) ∈ ℋ 𝒵 , ϕ ^ ​ ( T ) , ϕ ⁡ ( T ) ∈ ℋ 𝒯 , \displaystyle\psi(Z)\in{\mathcal{H}}_{\mathcal{Z}},\quad\hat{\phi}(T),\phi(T)\in{\mathcal{H}}_{\mathcal{T}}, | 
 | 
 

 KernelIV constructs a objective for optimizing e ∈ E e\in E by kernel ridge regression:

 

 
 | 
 e λ ∗ = argmin e ∈ E 𝔼 ∥ [ e ψ ] ( Z ) − ϕ ( T ) ∥ 2 + λ ∥ [ e ∥ 2 , \displaystyle e_{\lambda}^{*}=\text{argmin}_{e\in E}\mathbb{E}\|[e\psi](Z)-\phi(T)\|^{2}+\lambda\|[e\|^{2}, | 
 | 
 (57) | 
 
 
 | 
 e λ ∗ = argmin e ∈ E 𝔼 ∥ μ ( Z ) − ϕ ( T ) ∥ 2 + λ ∥ [ e ∥ 2 , \displaystyle e_{\lambda}^{*}=\text{argmin}_{e\in E}\mathbb{E}\|\mu(Z)-\phi(T)\|^{2}+\lambda\|[e\|^{2}, | 
 | 
 (58) | 
 

 where λ \lambda is a hyper-parameter and ∥ [ e ∥ 2 \|[e\|^{2} is a penalty term for function e e . Indeed, T ^ = μ ⁡ ( Z ) = [ e ∗ ​ ψ ] ​ ( Z ) \hat{T}=\mu({Z})=[e^{*}\psi](Z) . Analogously, in 2SLS T ^ = 𝔼 ⁡ [ T ∣ Z ] = α ^ ​ Z \hat{T}=\mathbb{E}[T\mid Z]=\hat{\alpha}Z for stage 1 linear regression parameter α ^ \hat{\alpha} .

 
 
 In stage 2, to estimate the structural function g ⁡ ( ⋅ ) g(\cdot) (Eq. ( 47 )), KernelIV predicts the potential outcome function onto the conditional mean embedding ϕ ^ ​ ( T ) ∈ ℋ 𝒯 \hat{\phi}(T)\in{\mathcal{H}}_{\mathcal{T}} :

 

 
 | 
 Y ^ = g ⁡ ( T ) = h ⁡ ( ϕ ⁡ ( T ) ) = [ h ​ μ ] ​ ( Z ) = 𝔼 ⁡ [ Y ∣ ϕ ^ ​ ( T ) ] , \displaystyle\hat{{Y}}=g({T})=h({\phi}(T))=[h\mu]({Z})=\mathbb{E}[{{Y}}\mid\hat{\phi}(T)], | 
 | 
 (59) | 
 

 KernelIV constructs a objective for optimizing h ∈ H h\in H by kernel ridge regression:

 

 
 | 
 h λ ∗ = argmin h ∈ H 𝔼 ∥ h ( ϕ ( T ) ) − Y ∥ 2 + λ ∥ [ h ∥ 2 , \displaystyle h_{\lambda}^{*}=\text{argmin}_{h\in H}\mathbb{E}\|h({\phi}(T))-Y\|^{2}+\lambda\|[h\|^{2}, | 
 | 
 (60) | 
 
 
 | 
 h λ ∗ = argmin h ∈ H 𝔼 ∥ h ( μ ( Z ) ) − Y ∥ 2 + λ ∥ [ h ∥ 2 , \displaystyle h_{\lambda}^{*}=\text{argmin}_{h\in H}\mathbb{E}\|h({\mu}(Z))-Y\|^{2}+\lambda\|[h\|^{2}, | 
 | 
 (61) | 
 

 where ∥ [ h ∥ 2 \|[h\|^{2} is a penalty term for function h h .
Indeed, Y ^ = g ⁡ ( T ) = [ h ​ ϕ ] ​ ( T ) = [ h ​ μ ] ​ ( Z ) \hat{Y}=g({T})=[h\phi]({T})=[h\mu]({Z}) . Analogously, in 2SLS Y ^ = 𝔼 ⁡ [ Y ∣ T ] = β ^ ​ T \hat{Y}=\mathbb{E}[Y\mid T]=\hat{\beta}T for stage 2 linear regression parameter β ^ \hat{\beta} .

 
 
 Dual IV . 
 Inspired by stochastic programming [ 91 , 92 ] , DualIV [ 62 ] shows that two-stage IV-based regression can be reformulated as a convex-concave saddle-point problem. Then, [ 62 ] develops a simple kernel-based algorithm and simplifies traditional two-stage methods via a dual formulation.

 
 
 Based on the outcome structural function, the expectation of Eq. ( 47 ) w.r.t. Y Y conditioned on { Z , 𝐗 } \{Z,\mathbf{X}\} yields [ 43 ] :

 

 
 | 
 𝔼 [ Y ∣ Z , 𝐗 ] \displaystyle\mathbb{E}[Y\mid Z,\mathbf{X}] | 
 = \displaystyle= | 
 𝔼 [ g ( T , 𝐗 ) ∣ Z , 𝐗 ] + 𝔼 [ ϵ Y ∣ 𝐗 ] \displaystyle\mathbb{E}[g(T,\mathbf{X})\mid Z,\mathbf{X}]+\mathbb{E}[{\epsilon}_{Y}\mid\mathbf{X}] | 
 | 
 (62) | 

 
 | 
 | 
 = \displaystyle= | 
 ∫ g ⁡ ( T , 𝐗 ) ​ 𝑑 F ​ ( T ∣ Z , 𝐗 ) , \displaystyle\int g(T,\mathbf{X})dF(T\mid Z,\mathbf{X}), | 
 | 
 

 where, d ​ F ​ ( T ∣ Z , 𝐗 ) dF(T\mid Z,\mathbf{X}) is the conditional treatment distribution obtained from the treatment regression. [ 62 ] reformulate the equation as an empirical risk minimization problem:

 

 
 | 
 min g ∈ 𝒢 ⁡ R ⁡ ( g ) = 𝔼 Y ​ Z ​ [ ℓ ⁡ ( Y , 𝔼 T | Z , 𝐗 ​ [ g ⁡ ( T , 𝐗 ) ] ) ] \displaystyle\min_{g\in\mathcal{G}}R(g)=\mathbb{E}_{YZ}\left[\ell\left(Y,\mathbb{E}_{T\mid Z,\mathbf{X}}[g(T,\mathbf{X})]\right)\right] | 
 | 
 (63) | 
 

 where ℓ ⁡ ( y , y ′ ) = ( y − y ′ ) 2 \ell\left(y,y^{\prime}\right)=\left(y-y^{\prime}\right)^{2} denotes the mean squared error.

 
 
 Applying the interchangeability and Fenchel duality [ 92 , 91 ] to Eq. ( 63 ):

 

 
 | 
 R ⁡ ( g ) = 𝔼 Y ​ Z ​ 𝐗 ​ [ max u ∈ ℝ ⁡ { 𝔼 T | Z , 𝐗 ​ [ g ⁡ ( T , 𝐗 ) ] ​ u − ℓ ⋆ ​ ( Y , u ) } ] \displaystyle R(g)=\mathbb{E}_{YZ\mathbf{X}}\left[\max_{u\in\mathbb{R}}\left\{\mathbb{E}_{T\mid Z,\mathbf{X}}[g(T,\mathbf{X})]u-\ell^{\star}(Y,u)\right\}\right] | 
 | 
 
 
 | 
 = max u ∈ 𝒰 ⁡ 𝔼 Z ​ 𝐗 ​ Y ​ [ 𝔼 T | Z , 𝐗 ​ [ g ⁡ ( T , 𝐗 ) ] ​ u ​ ( Y , Z , 𝐗 ) − ℓ ⋆ ​ ( Y , u ⁡ ( Y , Z , 𝐗 ) ) ] \displaystyle=\max_{u\in\mathcal{U}}\mathbb{E}_{Z\mathbf{X}Y}\left[\mathbb{E}_{T\mid Z,\mathbf{X}}[g(T,\mathbf{X})]u(Y,Z,\mathbf{X})-\ell^{\star}(Y,u(Y,Z,\mathbf{X}))\right] | 
 | 
 
 
 | 
 = max u ∈ 𝒰 ⁡ 𝔼 Z ​ 𝐗 ​ T ​ Y ​ [ g ⁡ ( T , 𝐗 ) ​ u ​ ( Y , Z , 𝐗 ) ] − 𝔼 Z ​ 𝐗 ​ Y ​ [ ℓ ⋆ ​ ( Y , u ⁡ ( Y , Z , 𝐗 ) ) ] \displaystyle=\max_{u\in\mathcal{U}}\mathbb{E}_{Z\mathbf{X}TY}[g(T,\mathbf{X})u(Y,Z,\mathbf{X})]-\mathbb{E}_{Z\mathbf{X}Y}\left[\ell^{\star}(Y,u(Y,Z,\mathbf{X}))\right] | 
 | 
 

 where 𝒰 ⁡ ( Ω ) = { u ⁡ ( ⋅ ) : Ω → ℝ } \mathcal{U}(\Omega)=\{u(\cdot):\Omega\rightarrow\mathbb{R}\} is the entire space of functions defined on the support Ω \Omega , and Ω \Omega is the corresponding space of random variables 𝒴 ⊕ 𝒵 ⊕ 𝒳 \mathcal{Y}\oplus\mathcal{Z}\oplus\mathcal{X} . ℓ : ℝ × ℝ → ℝ + \ell:\mathbb{R}\times\mathbb{R}\rightarrow\mathbb{R}_{+} is a proper, convex, and lower semi-continuous loss function for any value in its first argument and ℓ y ⋆ = ℓ ⋆ ​ ( y , ⋅ ) \ell_{y}^{\star}=\ell^{\star}(y,\cdot) is a convex conjugate of ℓ y = ℓ ⁡ ( y , ⋅ ) \ell_{y}=\ell(y,\cdot) .

 
 
 To simplify notation, in this section, we denotes by W = Y ⊕ Z ⊕ 𝐗 W=Y\oplus Z\oplus\mathbf{X} and T = T ⊕ 𝐗 T=T\oplus\mathbf{X} . Then, the saddle-point problem is:

 

 
 | 
 min g ∈ 𝒢 ⁡ max u ∈ 𝒰 ​ 𝔼 T ​ W ​ [ g ⁡ ( T ) ​ u ​ ( W ) ] − 𝔼 W ​ [ ℓ ⋆ ​ ( Y , u ⁡ ( W ) ) ] \displaystyle\min_{g\in\mathcal{G}}\max_{u\in\mathcal{U}}\mathbb{E}_{TW}[g(T)u(W)]-\mathbb{E}_{W}\left[\ell^{\star}(Y,u(W))\right] | 
 | 
 (64) | 
 

 With ℓ ⋆ ​ ( y , u ) = u ​ y + 1 2 ​ u 2 \ell^{\star}(y,u)=uy+\frac{1}{2}u^{2} , DualIV reduce the traditional two-stage methods as:

 

 
 | 
 min g ∈ 𝒢 ⁡ max u ∈ 𝒰 ⁡ Ψ ⁡ ( g , u ) , \displaystyle\min_{g\in\mathcal{G}}\max_{u\in\mathcal{U}}\Psi(g,u), | 
 | 
 (65) | 
 
 
 | 
 Ψ ⁡ ( g , u ) = 𝔼 T ​ W ​ { [ g ⁡ ( T ) − Y ] ​ u ​ ( W ) } − 1 2 ​ 𝔼 W ​ [ u ​ ( W ) 2 ] . \displaystyle\hskip-18.0pt\scalebox{1.0}{$\Psi(g,u)=\mathbb{E}_{TW}\{[g(T)-Y]u(W)\}-\frac{1}{2}\mathbb{E}_{W}[u(W)^{2}]$}. | 
 | 
 (66) | 
 

 
 
 Motivated by the reproducing kernel Hilbert spaces (RKHSs) [ 90 ] , DualIV introduces positive definite kernels k : 𝒯 × 𝒯 → ℝ k:\mathcal{T}\times\mathcal{T}\rightarrow\mathbb{R} and l : 𝒲 × 𝒲 → ℝ l:\mathcal{W}\times\mathcal{W}\rightarrow\mathbb{R} for 𝒢 \mathcal{G} and 𝒰 \mathcal{U} , respectively. [ 93 ] introduces the canonical feature maps:

 

 
 | 
 ϕ : t ↦ k ⁡ ( t , ⋅ ) , φ : w ↦ l ⁡ ( w , ⋅ ) . \displaystyle\phi:t\mapsto k(t,\cdot),\varphi:w\mapsto l(w,\cdot). | 
 | 
 (67) | 
 

 
 
 The objective can be rewritten as:

 

 
 | 
 Ψ ⁡ ( f , u ) \displaystyle\Psi(f,u) | 
 = \displaystyle= | 
 𝔼 T ​ W ​ [ f ​ ( T ) ​ u ​ ( W ) ] \displaystyle\mathbb{E}_{TW}[f(T)u(W)] | 
 | 
 (68) | 

 
 | 
 | 
 − \displaystyle- | 
 𝔼 Y ​ Z ​ [ Y ​ u ​ ( Y , Z ) ] − 1 2 ​ 𝔼 W ​ [ u ​ ( W ) 2 ] \displaystyle\mathbb{E}_{YZ}[Yu(Y,Z)]-\frac{1}{2}\mathbb{E}_{W}\left[u(W)^{2}\right] | 
 | 

 
 | 
 | 
 = \displaystyle= | 
 ⟨ 𝒞 W ​ T ​ f − 𝐛 , u ⟩ 𝒰 − 1 2 ​ ⟨ u , 𝒞 W ​ u ⟩ 𝒰 . \displaystyle\left\langle\mathcal{C}_{WT}f-\mathbf{b},u\right\rangle_{\mathcal{U}}-\frac{1}{2}\left\langle u,\mathcal{C}_{W}u\right\rangle_{\mathcal{U}}. | 
 | 
 

 where 𝐛 := 𝔼 Y ​ Z ​ [ Y ​ φ ​ ( Y , Z ) ] ∈ 𝒰 , 𝒞 W := 𝔼 W ​ [ φ ⁡ ( W ) ⊗ φ ⁡ ( W ) ] ∈ 𝒰 ⊗ 𝒰 \mathbf{b}:=\mathbb{E}_{YZ}[Y\varphi(Y,Z)]\in\mathcal{U},\mathcal{C}_{W}:=\mathbb{E}_{W}[\varphi(W)\otimes\varphi(W)]\in\mathcal{U}\otimes\mathcal{U} is a covariance operator, and 𝒞 W ​ T := 𝔼 W ​ T ​ [ φ ⁡ ( W ) ⊗ ϕ ⁡ ( T ) ] ∈ 𝒰 ⊗ ℱ \mathcal{C}_{WT}:=\mathbb{E}_{WT}[\varphi(W)\otimes\phi(T)]\in\mathcal{U}\otimes\mathcal{F} is a cross-covariance operator.
The generalized least squares solution in RKHS is:

 

 
 | 
 f ∗ \displaystyle f^{*} | 
 = \displaystyle= | 
 arg ⁡ min f ∈ ℱ ​ 1 2 ​ ⟨ 𝒞 W ​ T ​ f − 𝐛 , 𝒞 W − 1 ​ ( 𝒞 W ​ T ​ f − 𝐛 ) ⟩ 𝒰 \displaystyle\arg\min_{f\in\mathcal{F}}\frac{1}{2}\left\langle\mathcal{C}_{WT}f-\mathbf{b},\mathcal{C}_{W}^{-1}\left(\mathcal{C}_{WT}f-\mathbf{b}\right)\right\rangle_{\mathcal{U}} | 
 | 
 (69) | 

 
 | 
 | 
 = \displaystyle= | 
 ( 𝒞 T ​ W ​ 𝒞 W − 1 ​ 𝒞 W ​ T ) − 1 ​ 𝒞 T ​ W ​ 𝒞 W − 1 ​ 𝐛 \displaystyle\left(\mathcal{C}_{TW}\mathcal{C}_{W}^{-1}\mathcal{C}_{WT}\right)^{-1}\mathcal{C}_{TW}\mathcal{C}_{W}^{-1}\mathbf{b} | 
 | 
 

 Eq. ( 69 ) gives a solution for IV-based regression in closed form.

 
 
 

#### IV-C 2 Deep-based Estimator

 
 Originally, 2SLS performs linear regressions in both stages under linearity assumption. Recent machine learning methods extend it to non-linear settings with infinite dictionaries of basis functions from reproducing kernel Hibert spaces (RKHS), such as KernelIV [ 61 ] and DualIV [ 62 ] .
Although these methods enjoy desirable theoretical properties, the flexibility of the model is limited, since the basis functions are pre-specified by human-hand or feature engineering [ 17 , 81 ] .

 
 
 DeepIV .
 DeepIV builds upon deep-based methods, i.e., deep neural networks [ 17 ] .
Although there is little theory to justify when learning with neural networks can identify a true model, deep methods make substantially weaker assumptions about the data generating process and automatically learn flexible feature mappings for high-dimension and non-linear data, which saves the human effort selecting pre-defined basis functions and improves the accuracy of causal effect estimation.
Under additive noise assumption or linearity assumption, [ 17 ] provide an unique solution for the inverse problem with the learned representation, as follows.

 
 
 Taking the expectation of both sides of Eq. ( 47 ) conditioned on { Z , 𝐗 } \{Z,\mathbf{X}\} and applying assumptions formulates the relationship [ 43 ] :

 

 
 | 
 𝔼 [ Y ∣ Z , 𝐗 ] \displaystyle\mathbb{E}[Y\mid Z,\mathbf{X}] | 
 = \displaystyle= | 
 𝔼 [ g ( T , 𝐗 ) ∣ Z , 𝐗 ] + 𝔼 [ ϵ Y ∣ 𝐗 ] \displaystyle\mathbb{E}[g(T,\mathbf{X})\mid Z,\mathbf{X}]+\mathbb{E}[{\epsilon}_{Y}\mid\mathbf{X}] | 
 | 
 (70) | 

 
 | 
 | 
 = \displaystyle= | 
 ∫ g ⁡ ( T , 𝐗 ) ​ 𝑑 F ​ ( T ∣ Z , 𝐗 ) , \displaystyle\int g(T,\mathbf{X})dF(T\mid Z,\mathbf{X}), | 
 | 
 

 where, again, d ​ F ​ ( T ∣ Z , 𝐗 ) dF(T\mid Z,\mathbf{X}) is the conditional treatment distribution obtained from the treatment regression. The relationship defines an inverse problem in structural function identification. Given observational data { z i , 𝐱 i , t i , y i } \{z_{i},\mathbf{x}_{i},t_{i},y_{i}\} , the counterfactual functions are recovered by minimizing the objective:

 

 
 | 
 g ^ = argmin g ∈ 𝒢 ​ ∑ i = 1 n ( y i − ∫ t g ⁡ ( t , x i ) ​ 𝑑 F ​ ( t ∣ z i , 𝐱 i ) ) 2 . \displaystyle\hat{g}=\text{argmin}_{g\in\mathcal{G}}\sum_{i=1}^{n}\left(y_{i}-\int_{t}g(t,x_{i})dF(t\mid z_{i},\mathbf{x}_{i})\right)^{2}. | 
 | 
 (71) | 
 

 
 
 Furthermore, in estimation, DeepIV develops a two-stages procedure. To obtain the the conditional probability estimation d ​ F ​ ( t ∣ z i , 𝐱 i ) dF(t\mid z_{i},\mathbf{x}_{i}) of treatments, deep methods use conditional density estimation model as treatment regression module in stage 1 [ 94 , 17 ] . Then, they perform a joint mapping from re-sampled treatments T ^ \hat{T} and confounders 𝐗 \mathbf{X} to the counterfactual outcomes Y Y in stage 2.

 
 
 Treatment Regression Stage . Specifically, we use a deep neural network π ϕ ​ ( Z , 𝐗 ) \pi_{\phi}(Z,\mathbf{X}) with parameters ϕ \phi to model the conditional density function of treatment F ⁡ ( T ∣ Z , 𝐗 ) F(T\mid Z,\mathbf{X}) . The objective can be written as:

 

 
 | 
 min  ​ ℒ 1 = l ⁡ ( T , π ϕ ​ ( Z , 𝐗 ) ) , \displaystyle\text{min }\mathcal{L}_{1}=l(T,\pi_{\phi}(Z,\mathbf{X})), | 
 | 
 (72) | 
 

 where l ​ ( T , π ϕ ​ ( Z , 𝐗 ) ) l(T,\pi_{\phi}(Z,\mathbf{X})) would be an l 2 l_{2} -loss for continuous outcomes or a log-loss for binary outcomes.
For discrete treatments T T , we model π ϕ ​ ( Z , 𝐗 ) \pi_{\phi}(Z,\mathbf{X}) with P ⁡ ( T = k ) = π ϕ , k ​ ( Z , 𝐗 ) P(T=k)=\pi_{\phi,k}(Z,\mathbf{X}) for each treatment arm T = k T=k and where π ϕ , k ​ ( Z , 𝐗 ) \pi_{\phi,k}(Z,\mathbf{X}) is given by the k k -th element of softmax output in a DNN. For continuous treatments T T , we model a mixture of Gaussian distributions with component π ϕ , k ​ ( Z , 𝐗 ) \pi_{\phi,k}(Z,\mathbf{X}) and sub-networks OPEN [ μ ϕ , k ​ ( Z , 𝐗 ) ] , σ ϕ , k ​ ( Z , 𝐗 ) ] [\mu_{\phi,k}(Z,\mathbf{X})],\sigma_{\phi,k}(Z,\mathbf{X})] for Gaussian distribution parameters G ⁡ ( μ , σ ) G(\mu,\sigma) . With enough mixture
components, the network π ϕ ​ ( Z , 𝐗 ) \pi_{\phi}(Z,\mathbf{X}) can approximate arbitrary smooth densities.

 
 
 Outcome Regression Stage . We model a counterfactual prediction network h θ h_{\theta} with parameters θ \theta , to approximate the potential outcome. The objective can be written as:

 

 
 | 
 min  ​ ℒ 2 = 1 n ​ ∑ i = 1 n ( y i − ∫ t h θ ​ ( t , x i ) ​ d ​ F ^ ϕ ​ ( t ∣ z i , x i ) ) 2 , \displaystyle\text{min }\mathcal{L}_{2}=\frac{1}{n}\sum_{i=1}^{n}\left(y_{i}-\int_{t}h_{\theta}(t,x_{i})d\hat{F}_{\phi}(t\mid z_{i},x_{i})\right)^{2}, | 
 | 
 (73) | 
 

 where F ^ ϕ ​ ( T ∣ Z , 𝐗 ) \hat{F}_{\phi}(T\mid Z,\mathbf{X}) is from the stage 1. Then, we can can optimize the F ^ ϕ ​ ( T ∣ Z , 𝐗 ) \hat{F}_{\phi}(T\mid Z,\mathbf{X}) and h θ ​ ( T , 𝐗 ) h_{\theta}(T,\mathbf{X}) by minimizing the loss ℒ 1 ​ ( ϕ ) \mathcal{L}_{1}(\phi) and ℒ 2 ​ ( θ ) \mathcal{L}_{2}(\theta) using gradient descent, respectively.

 
 
 OneSIV .
 However, existing deep-based methods require two stages to separately
estimate the conditional treatment distribution and the potential outcome function, which is not sufficiently effective [ 80 ] . Lin et al. [ 80 ] claims that the information from the outcome regression is one significant component for joint distribution of observations, and we should utilizing this information to improve the conditional treatment distribution estimation.

 
 
 One Stage Regression . Further, they merge the two stages to leverage the outcome regression h θ ​ ( T , 𝐗 ) h_{\theta}(T,\mathbf{X}) to the
treatment distribution estimation F ^ ϕ ​ ( T ∣ Z , 𝐗 ) \hat{F}_{\phi}(T\mid Z,\mathbf{X}) through a cleverly designed deep neural network structure. Then, they present a joint trade-off objective, as follows:

 

 
 | 
 min ϕ , θ ​ w 1 ​ ℒ 1 + w 2 ​ ℒ 2 , \displaystyle\text{min}_{\phi,\theta}w_{1}\mathcal{L}_{1}+w_{2}\mathcal{L}_{2}, | 
 | 
 (74) | 
 

 where w 1 w_{1} and w 2 w_{2} are the hyper-parameters to control the relative importance of treatment regression ℒ 1 \mathcal{L}_{1} (Eq. 72 ) and outcome regression ℒ 2 \mathcal{L}_{2} (Eq. 73 ) obtained from DeepIV. Minimizing this objective, the treatment regression network and the outcome regression network can promote each other’s evolution, i.e., Co-evolution.

 
 
 DFIV . 
 Combining the theoretical advantages of kernel-based methods and the empirical advantages of deep learning methods, DFIV [ 81 ] uses deep neural networks (DNNs) to adaptively learn deep features as kernel basis in the 2SLS approach, which fits structural functions with highly nonlinear flexibility. [ 81 ] develops three DNNs { f ϕ , g ξ , u ψ } \{f_{\phi},g_{\xi},u_{\psi}\} to learn the corresponding feature mappings for { Z , 𝐗 , T } \{Z,\mathbf{X},T\} , respectively. Similar to Eqs. ( 48 )( 49 ), we can reformulate the IV-based regression as:

 

 
 | 
 u ψ , k ​ ( T ) = ∑ i = 1 d Z ∑ j = 1 d X α i , j k ​ f ϕ , i ​ ( Z ) ​ g ξ , j ​ ( 𝐗 ) + ϵ T , \displaystyle u_{\psi,k}(T)=\sum_{i=1}^{d^{Z}}\sum_{j=1}^{d^{X}}\alpha_{i,j}^{k}f_{\phi,i}(Z)g_{\xi,j}(\mathbf{X})+\epsilon_{T}, | 
 | 
 (75) | 
 
 
 | 
 Y = ∑ k = 1 d T ∑ j = 1 d X β k , j ​ u ψ , k ​ ( T ) ​ g ξ , j ​ ( 𝐗 ) + ϵ Y , \displaystyle Y=\sum_{k=1}^{d^{T}}\sum_{j=1}^{d^{X}}\beta_{k,j}u_{\psi,k}(T)g_{\xi,j}(\mathbf{X})+\epsilon_{Y}, | 
 | 
 (76) | 
 

 where f ϕ , i ​ ( Z ) f_{\phi,i}(Z) denotes the i i -th element in the outcome vector of instrument representation network f ϕ ​ ( Z ) f_{\phi}(Z) , g ξ , j ​ ( 𝐗 ) g_{\xi,j}(\mathbf{X}) is the j j -th element in the outcome vector of covariate representation network g ξ ​ ( 𝐗 ) g_{\xi}(\mathbf{X}) , and u ψ , k ​ ( T ) u_{\psi,k}(T) is the k k -th element in the outcome vector of treatment representation network u ψ ​ ( T ) u_{\psi}(T) . { d Z , d X , d T } \{d^{Z},d^{X},d^{T}\} denotes the dimension of the outcome vector f ϕ ​ ( Z ) f_{\phi}(Z) , g ξ ​ ( 𝐗 ) g_{\xi}(\mathbf{X}) , and u ψ ​ ( T ) u_{\psi}(T) . 𝐀 = [ α i , j k ] i , j , k \mathbf{A}=[\alpha_{i,j}^{k}]_{i,j,k} and 𝐁 = [ β i , j ] i , j \mathbf{B}=[\beta_{i,j}]_{i,j} denote the corresponding coefficients in the linear associations between features { f ϕ ​ ( Z ) , g ξ ​ ( 𝐗 ) , u ψ ​ ( T ) , Y } \{f_{\phi}(Z),g_{\xi}(\mathbf{X}),u_{\psi}(T),Y\} .

 
 
 Treatment Regression Stage . Fixing the parameter ψ \psi of the treatment representation network u ψ ​ ( ⋅ ) u_{\psi}(\cdot) and the parameter ξ \xi of the covariate representation network g ξ ​ ( ⋅ ) g_{\xi}(\cdot) during stage 1, DFIV aims to regress the conditional expectation 𝔼 ⁡ [ u ψ ​ ( T ) ∣ f ϕ ​ ( Z ) ⊗ g ξ ​ ( 𝐗 ) ] \mathbb{E}[u_{\psi}(T)\mid f_{\phi}(Z)\otimes g_{\xi}(\mathbf{X})] by learning the network parameter ϕ \phi and the coefficient matrix 𝐀 ∈ ℝ d T × ( d Z ⋅ d X ) \mathbf{A}\in\mathbb{R}^{d^{T}\times(d^{Z}\cdot d^{X})} , where f ϕ ​ ( Z ) ⊗ g ξ ​ ( 𝐗 ) f_{\phi}(Z)\otimes g_{\xi}(\mathbf{X}) denotes the multiplication combination set [ f ϕ , i ​ ( Z ) ​ g ξ , j ​ ( 𝐗 ) ] i , j [f_{\phi,i}(Z)g_{\xi,j}(\mathbf{X})]_{i,j} .

 

 
 | 
 ϕ ∗ \displaystyle\phi^{*} | 
 = \displaystyle= | 
 argmin ϕ ​ ℒ 1 ​ ( ϕ ) , \displaystyle\text{argmin}_{\phi}\mathcal{L}_{1}(\phi), | 
 | 
 (77) | 
 
 
 | 
 
 
 ℒ 1 ​ ( ϕ ) \mathcal{L}_{1}(\phi) 

 | 
 = \displaystyle= | 
 
 
 1 n ​ ∑ i = 1 n [ ‖ u ψ ​ ( t i ) − 𝐀 ​ f ϕ ​ ( z i ) ⊗ g ξ ​ ( x i ) ‖ 2 + λ 1 ​ ‖ 𝐀 ‖ 2 ] \frac{1}{n}\sum_{i=1}^{n}\left[\left\|u_{\psi}(t_{i})-\mathbf{A}f_{\phi}(z_{i})\otimes g_{\xi}(x_{i})\right\|^{2}+\lambda_{1}\|\mathbf{A}\|^{2}\right] 

 | 
 | 
 (78) | 
 
 
 | 
 𝐀 ⁡ ( ϕ ) \displaystyle\mathbf{A}(\phi) | 
 = \displaystyle= | 
 u ψ ​ ( T ) ′ ​ 𝐂 ​ ( 𝐂 ′ ​ 𝐂 + n ​ λ 1 ​ I ) − 1 \displaystyle u_{\psi}(T)^{\prime}\mathbf{C}(\mathbf{C}^{\prime}\mathbf{C}+n\lambda_{1}I)^{-1} | 
 | 
 (79) | 
 

 To simplify notation, in this section, we denotes by 𝐂 = f ϕ ​ ( Z ) ⊗ g ξ ​ ( 𝐗 ) ∈ ℝ n × ( d Z ⋅ d X ) \mathbf{C}=f_{\phi}(Z)\otimes g_{\xi}(\mathbf{X})\in\mathbb{R}^{n\times(d^{Z}\cdot d^{X})} . We can then learn the parameters ϕ \phi of the instrument representation network f ϕ ​ ( ⋅ ) f_{\phi}(\cdot) by minimizing the loss ℒ 1 ​ ( ϕ ) \mathcal{L}_{1}(\phi) using gradient descent.

 
 
 Outcome Regression Stage . Fixing the parameters ϕ \phi of the instrument representation network f ϕ ​ ( ⋅ ) f_{\phi}(\cdot) and the parameter ξ \xi of the covariate representation network g ξ ​ ( ⋅ ) g_{\xi}(\cdot) during stage 2, DFIV predicts the structural function 𝔼 ⁡ [ Y ∣ u ψ ​ ( T ) ⊗ g ξ ​ ( 𝐗 ) ] \mathbb{E}[Y\mid u_{\psi}(T)\otimes g_{\xi}(\mathbf{X})] by learning the network parameter ψ \psi and the coefficient matrix 𝐁 ∈ ℝ 1 × ( d T ⋅ d X ) \mathbf{B}\in\mathbb{R}^{1\times(d^{T}\cdot d^{X})} . To simplify notation, in this section, we use 𝐃 = f ψ ​ ( T ) ⊗ g ξ ​ ( 𝐗 ) \mathbf{D}=f_{\psi}(T)\otimes g_{\xi}(\mathbf{X}) denotes the multiplication combination set [ f ψ , k ​ ( T ) ​ g ξ , j ​ ( 𝐗 ) ] k , j [f_{\psi,k}(T)g_{\xi,j}(\mathbf{X})]_{k,j} .

 

 
 | 
 ψ ∗ \displaystyle\psi^{*} | 
 = \displaystyle= | 
 argmin ψ ​ ℒ 2 ​ ( ψ ) , \displaystyle\text{argmin}_{\psi}\mathcal{L}_{2}(\psi), | 
 | 
 (80) | 
 
 
 | 
 
 
 ℒ 2 ​ ( ψ ) \mathcal{L}_{2}(\psi) 

 | 
 = \displaystyle= | 
 
 
 1 n ​ ∑ i = 1 n [ ‖ y i − 𝐁 ​ f ψ ​ ( t i ) ⊗ g ξ ​ ( x i ) ‖ 2 + λ 2 ​ ‖ 𝐁 ‖ 2 ] \frac{1}{n}\sum_{i=1}^{n}\left[\left\|y_{i}-\mathbf{B}f_{\psi}(t_{i})\otimes g_{\xi}(x_{i})\right\|^{2}+\lambda_{2}\|\mathbf{B}\|^{2}\right] 

 | 
 | 
 (81) | 
 
 
 | 
 𝐁 ⁡ ( ψ ) \displaystyle\mathbf{B}(\psi) | 
 = \displaystyle= | 
 Y ′ ​ 𝐃 ​ ( 𝐃 ′ ​ 𝐃 + n ​ λ 2 ​ I ) − 1 \displaystyle Y^{\prime}\mathbf{D}(\mathbf{D}^{\prime}\mathbf{D}+n\lambda_{2}I)^{-1} | 
 | 
 (82) | 
 

 We can then learn the parameters ψ \psi of the treatment representation network f ψ ​ ( ⋅ ) f_{\psi}(\cdot) by minimizing the loss ℒ 2 ​ ( ψ ) \mathcal{L}_{2}(\psi) using gradient descent.

 
 
 Note that the covariate representation network g ξ ​ ( ⋅ ) g_{\xi}(\cdot) is fixed during stage 1 and stage 2. To update the covariate network g ξ ​ ( ⋅ ) g_{\xi}(\cdot) , fixing the parameters ψ \psi and ϕ \phi , we minimize the loss ℒ 1 ​ ( ξ ) + ℒ 2 ​ ( ξ ) \mathcal{L}_{1}(\xi)+\mathcal{L}_{2}(\xi) using gradient descent. Then, we adopt an alternating training strategy to iteratively optimize the representations for g ψ ​ ( ⋅ ) g_{\psi}(\cdot) , g ϕ ​ ( ⋅ ) g_{\phi}(\cdot) and g ξ ​ ( ⋅ ) g_{\xi}(\cdot) .

 
 
 

#### IV-C 3 GMM-based Estimator

 
 In the presence of heteroskedasticity, although the counterfactual function estimation of the standard IV estimators and some variants is consistent with the true potential outcomes, the standard errors are inconsistent, preventing valid inference [ 95 ] .
Assuming observational data can be formalized in moment conditions, when facing heteroskedasticity of unknown form, we can make use of the conditional moment restrictions to allow for efficient estimation. That is, instrumental variable regression and 2SLS can be seen as special cases of generalized method of moments (GMM), introduced by [ 96 ] , which is a prototypical (non-)parametric estimator [ 97 , 98 , 99 ] .

 
 
 The standard IV estimator is a special case of GMM. Satisfying the IV assumptions, the instruments Z Z is correlated with the endogenous treatments T T and orthogonal to the unmeasured confounders ϵ T {\epsilon}_{T} / ϵ Y \epsilon_{Y} at the same time, i.e., Z ⟂ U Z\perp U . Then, we can design a IV-based GMM estimator to satisfy the orthogonality conditions with the overidentified context. Under the additive noise assumption (Eq. ( 46 )( 47 )), the moment conditions for instruments Z ∈ ℝ n × d Z Z\in\mathbb{R}^{n\times d^{Z}} can be formulated as 𝔼 ⁡ [ Z ​ ϵ T ] = 𝔼 ⁡ [ Z ​ ϵ Y ] = 0 \mathbb{E}[Z\epsilon_{T}]=\mathbb{E}[Z\epsilon_{Y}]=0 . The d Z d^{Z} instruments give a set of d Z d^{Z} moments:

 

 
 | 
 l i ​ ( g ) \displaystyle\hskip-18.0ptl_{i}(g) | 
 = \displaystyle= | 
 z i ′ u i = z i ′ ( y i − g ( t i , x i ) ) , i = 1 , ⋯ , n \displaystyle z_{i}^{\prime}u_{i}=z_{i}^{\prime}(y_{i}-g(t_{i},x_{i})),i=1,\cdots,n | 
 | 
 (83) | 
 
 
 | 
 𝔼 ⁡ [ l ⁡ ( g ) ] \displaystyle\hskip-18.0pt\mathbb{E}[l(g)] | 
 = \displaystyle= | 
 1 n ​ ∑ i = 1 n l i ​ ( g ) = 1 n ​ ∑ i = 1 n z i ′ ​ ( y i − g ⁡ ( t i , x i ) ) = 𝟎 . \displaystyle\frac{1}{n}\sum_{i=1}^{n}l_{i}(g)=\frac{1}{n}\sum_{i=1}^{n}z_{i}^{\prime}(y_{i}-g(t_{i},x_{i}))=\mathbf{0}. | 
 | 
 (84) | 
 

 where 𝔼 ⁡ [ l ⁡ ( g ) ] \mathbb{E}[l(g)] is a d Z d^{Z} vector, and we set L j = 𝔼 ​ [ l ⁡ ( g ) ] j L_{j}=\mathbb{E}[l(g)]_{j} to denote the j j -th element in the expectation error vector 𝔼 ⁡ [ l ⁡ ( g ) ] \mathbb{E}[l(g)] . The intuition of GMM is to choose an estimator for function g g , and set these d Z d^{Z} moments as close to zero as possible.

 
 
 In the estimation of potential outcome function, if the number of unknown parameter is exactly d Z d^{Z} , the estimated equation is exactly identified —— the d Z d^{Z} moment conditions and the d Z d^{Z} parameters in regression function. If we have less unknown parameters than conditional moment restrictions, then the estimated equation is overidentified, and we cannot find a prediction function g g to set all d Z d^{Z} sample moment conditions [ L j = 𝔼 [ l ( g ) ] j ] j = 1 , ⋯ , d Z [L_{j}=\mathbb{E}[l(g)]_{j}]_{j=1,\cdots,d^{Z}} to exactly zero. Thus, GMM estimator replace the theoretical expected value 𝔼 ⁡ [ ⋅ ] \mathbb{E}[\cdot] with its empirical analog—sample average:

 

 
 | 
 𝒥 ( θ ) = ∑ j = 1 d Z L j 2 ( θ ) = ∥ L ( θ ) ∥ 2 = L ( θ ) ′ W L ( θ ) = ∑ j = 1 d Z [ l ( g θ ) ] j ] 2 . \displaystyle\hskip-18.0pt\mathcal{J}(\theta)=\sum_{j=1}^{d^{Z}}L_{j}^{2}(\theta)=\|L(\theta)\|^{2}=L(\theta)^{\prime}WL(\theta)=\sum_{j=1}^{d^{Z}}[l(g_{\theta})]_{j}]^{2}. | 
 | 
 (85) | 
 

 where W = I W=I is an identify matrix, meaning the average effect.
Then we minimize the norm of this expression with respect to function g θ g_{\theta} . The minimizing function of g θ g_{\theta} is our estimate for g g .

 
 
 Although GMM is an incredibly flexible estimator, in practical, there are an infinite number of moment conditions with IV independence assumptions. Imposing all of them is infeasible with finite data. Therefore, recent literature proposes a series of minimax approaches to reformulate the minimax optimization problem.

 
 
 Minimax Approachs . 
 There has also been a recent surge in interest with minimax approaches that reformulate conditional moment conditions as a minimax optimization problem. For example, Lewis Syrgkanis (2018); Zhang et al. (2020) use the reformulation sup h ∈ L 2 ​ ( Z i ) ( 𝔼 ⁡ [ h ⁡ ( Z i ) ​ ( Y i − f ∗ ​ ( T i , 𝐗 i ) ) ] ) 2 \sup_{h\in L_{2}\left(Z_{i}\right)}\left(\mathbb{E}\left[h\left(Z_{i}\right)\left(Y_{i}-f^{*}\left(T_{i},\mathbf{X}_{i}\right)\right)\right]\right)^{2} . Bennett et al. (2019), Bennett Kallus (2020), Muandet et al. (2020), while Dikkala et al. (2020), Chernozhukov et al. (2020), and Liao et al. (2020), employ other reformulations, i.e., sup h ∈ L 2 ​ ( Z i , 𝐗 i ) ( 𝔼 ⁡ [ h ⁡ ( Z i , 𝐗 i ) ​ ( Y i − f ∗ ​ ( T i , 𝐗 i ) ) ] ) 2 \sup_{h\in L_{2}\left(Z_{i},\mathbf{X}_{i}\right)}\left(\mathbb{E}\left[h\left(Z_{i},\mathbf{X}_{i}\right)\left(Y_{i}-f^{*}\left(T_{i},\mathbf{X}_{i}\right)\right)\right]\right)^{2} .

 
 
 With the rapid development of machine learning algorithms, researchers apply adaptive non-parametric learners such as reproducing kernel Hilbert spaces, random forests, and neural networks to reformulate GMM estimation to the minimax optimization problem [ 63 , 82 , 100 ] .
In machine learning and statistics, researchers formulate the target estimand as an objective minimization problem. Then, Lewis et al. [ 63 ] formulate the expectation minimization problem as the maximum moment deviation over the set of potential functions, refered as Adversarial GMM (AGMM):

 

 
 | 
 h ∗ = arginf h ∈ ℋ ​ sup f ∈ ℱ ​ 𝔼 ​ [ ( Y − h ⁡ ( T , 𝐗 ) ) ​ f ​ ( Z , 𝐗 ) ] . \displaystyle h^{*}=\text{arginf}_{h\in\mathcal{H}}\text{sup}_{f\in\mathcal{F}}\mathbb{E}[(Y-h(T,\mathbf{X}))f(Z,\mathbf{X})]. | 
 | 
 (86) | 
 

 Similar to Wasserstein and MMD GANs [ 101 , 102 ] , the formulation proposes a learner network h h to set moments as close to zero as possible, and an adversary network f f to identify moments that are violated for the chosen h h .
 [ 63 ] offers main theorems and applications for several hypothesis spaces of practical interest including reproducing kernel Hilbert spaces (RKHS), functions defined via shape restrictions, random forests, and neural networks.

 
 
 Given observational data { z i , x i , t i , y i } i = 1 , ⋯ , n \{z_{i},x_{i},t_{i},y_{i}\}_{i=1,\cdots,n} , to obtain optimal h ϕ h_{\phi} and f ψ f_{\psi} , AGMM [ 63 ] minimizes the empirical analogue of the minimax objective:

 

 
 | 
 ϕ ∗ \displaystyle\phi^{*} | 
 = \displaystyle= | 
 arginf ϕ ∈ Φ ​ sup ψ ∈ Ψ ​ 𝔼 ​ [ ( Y − h ϕ ​ ( T , 𝐗 ) ) ​ f ψ ​ ( Z , 𝐗 ) ] \displaystyle\text{arginf}_{\phi\in\Phi}\text{sup}_{\psi\in\Psi}\mathbb{E}[(Y-h_{\phi}(T,\mathbf{X}))f_{\psi}(Z,\mathbf{X})] | 
 | 
 (87) | 

 
 | 
 | 
 − \displaystyle- | 
 λ 1 ​ ‖ ψ ‖ 2 − 𝔼 ⁡ [ f ψ ​ ( Z , 𝐗 ) 2 ] + λ 2 ​ ‖ ϕ ‖ 2 . \displaystyle\lambda_{1}\|\psi\|^{2}-\mathbb{E}[f_{\psi}(Z,\mathbf{X})^{2}]+\lambda_{2}\|\phi\|^{2}. | 
 | 
 

 where { λ 1 , λ 2 } \{\lambda_{1},\lambda_{2}\} are the hyper-parameters for penalty items ‖ ϕ ‖ 2 \|\phi\|^{2} and ‖ ψ ‖ 2 \|\psi\|^{2} .

 
 
 DeepGMM . 
 With infinite moment conditions, using identify matrix I I as unweighted vector norm can lead to significant inefficiencies in the minimization of objective Eq. ( 85 ) [ 96 , 103 ] . [ 96 , 103 ] claim that weighting moment conditions by their inverse covariance would yield minimal variance estimates, and it is sufficient to consistently estimate this covariance.
Based on the optimally weighted Generalized Method of Moments (GMM) [ 96 , 82 , 104 ] , DeepGMM [ 82 ] construct an optimal combination of moment conditions via adversarial training, with the objective:

 

 
 | 
 ϕ ∗ \displaystyle\phi^{*} | 
 = \displaystyle= | 
 arginf ϕ ∈ Φ ​ sup ψ ∈ Ψ ​ 𝔼 ​ [ ( Y − h ϕ ​ ( T , 𝐗 ) ) ​ f ψ ​ ( Z , 𝐗 ) ] \displaystyle\text{arginf}_{\phi\in\Phi}\text{sup}_{\psi\in\Psi}\mathbb{E}[(Y-h_{\phi}(T,\mathbf{X}))f_{\psi}(Z,\mathbf{X})] | 
 | 
 (88) | 

 
 | 
 | 
 − \displaystyle- | 
 1 4 ​ 𝔼 ​ [ ( Y − h ϕ ​ ( T , 𝐗 ) ) 2 ​ f ψ 2 ​ ( Z , 𝐗 ) ] . \displaystyle\frac{1}{4}\mathbb{E}[(Y-h_{\phi}(T,\mathbf{X}))^{2}f_{\psi}^{2}(Z,\mathbf{X})]. | 
 | 
 

 Notably, DeepGMM [ 82 ] has a few tuning parameters: the models ℱ \mathcal{F} and ℋ \mathcal{H} (i.e., the neural network architectures) and whatever parameters the optimization method uses.
Besides, other reformulations of minimax problem are developed by [ 105 , 106 ] 

 
 
 

#### IV-C 4 Confounder Balance Estimator

 
 With the development of machine learning, instrumental variables are no longer limited to simple linear models. The recent IV models described above have focused on various complex setting, where interactions between various variables may exist, such as T = Z ​ X + X + U T=ZX+X+U .
At this point, if we do not consider the joint effect of modeling covariates and IVs, the effect of IVs on the treatment variables will be very limited, i.e., weak IV. Therefore, these algorithms combine observed confounders and IVs to predict the conditional distribution of the treatments to eliminate unmeasured confounding bias in stage 1. However, this introduces additional bias due to imbalanced covariates X X on different treatment arms in stage 2 (Fig.  9 ).

 
 
 CBIV . 
 Wu et al. [ 54 ] focus on treatment effect estimation with IV regression under homogeneity assumptions, and they propose a Confounder Balanced IV Regression (CB-IV) algorithm to further remove the confounding bias from observed confounders by balancing in nonlinear scenarios.

 
 
 Based on the Homogeneous Instrument-Treatment Assumption, Wu et al. [ 54 ] model a more general causal relationship by relaxing the additive assumption to multiplicative assumption on response-outcome function as:

 

 
 | 
 T = f 1 ​ ( Z , X ) + f 2 ​ ( X , U ) \displaystyle\scalebox{0.95}{$T$}=\scalebox{0.95}{$f_{1}(Z,X)+f_{2}(X,U)$} | 
 | 
 (89) | 
 
 
 | 
 Y = g 1 ​ ( T , X ) + g 2 ​ ( T ) ​ g 3 ​ ( U ) + g 4 ​ ( X , U ) , Z ⟂ U , X \displaystyle\scalebox{0.95}{$Y$}=\scalebox{0.95}{$g_{1}(T,X)+g_{2}(T)g_{3}(U)+g_{4}(X,U),Z\perp U,X$} | 
 | 
 (90) | 
 

 where f i ​ ( ⋅ ) , g j ​ ( ⋅ ) {f_{i}}(\cdot),{g_{j}}(\cdot) are unknown and potentially non-linear continuous functions. g 2 ​ ( T ) ​ g 3 ​ ( U ) g_{2}(T)g_{3}(U) denotes the multiplicative terms of U U with T T (e.g., U 2 ​ T − U ​ T + U U^{2}T-UT+U ).
The completeness of ℙ ⁡ ( T ∣ Z , X ) \mathbb{P}({T\mid Z,X}) and ℙ ⁡ ( Y ∣ T , X ) \mathbb{P}({Y\mid T,X}) guarantees uniqueness of the solution [ 43 ] .

 
 
 Fig. 9 : Confounding bias from observed confounders. 
 
 
 The CB-IV algorithm contains the following three main components:

 
 
 Treatment Regression in Stage 1: For continuous treatment T T , CBIV regresses treatment T T with IVs Z Z and observed confounders X X .

 

 
 | 
 ℒ T = 1 n ​ ∑ i = 1 n ∑ j = 1 m ( t i − t ^ i j ) 2 , t ^ i j ∼ P ^ ​ ( t i | z i , x i ) , \displaystyle\mathcal{L}_{T}=\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{m}\left(t_{i}-\hat{t}_{i}^{j}\right)^{2},\hat{t}_{i}^{j}\sim\hat{P}(t_{i}|z_{i},x_{i}), | 
 | 
 (91) | 
 

 we sample m m (the larger the better) treatment { t ^ i j } j = 1 , … , m \{\hat{t}_{i}^{j}\}_{j=1,...,m} for each unit { z i , x i } \{z_{i},x_{i}\} to approximate the true treatment t i t_{i} .
Empirically, the above objective (Eq. ( 91 )) is sufficient to accurately estimate causal effects in continuous CB-IV framework.

 
 
 Confounder Balance in Stage 2: 
For continuous treatment T T , we learn a ”balanced” representation (i.e., C C ) of the observed confounders X X as C = f θ ​ ( X ) C=f_{\theta}(X) via mutual information (MI) minimization constraints: firstly, we use variational distribution Q ψ ​ ( T ^ ∣ C ) = 𝒩 ⁡ ( μ ψ ​ ( C ) , σ ψ ​ ( C ) ) Q_{\psi}(\hat{T}\mid C)=\mathcal{N}(\mu_{\psi}(C),\sigma_{\psi}(C)) parameterized by neural networks { μ ψ , σ ψ } \{\mu_{\psi},\sigma_{\psi}\} to approximate the true conditional distribution P ⁡ ( T ^ ∣ C ) P(\hat{T}\mid C) ; then, we minimize the log-likelihood loss function of variational approximation Q ψ ​ ( T ^ ∣ C ) Q_{\psi}(\hat{T}\mid C) with n n samples to estimate MI:

 

 
 | 
 disc ​ ( T ^ , C ) = 1 n 2 ​ ∑ i = 1 n ∑ j = 1 n [ log ⁡ Q ψ ​ ( t ^ i ∣ c i ) − log ⁡ Q ψ ​ ( t ^ j ∣ c i ) ] . \displaystyle{\text{ disc}(\hat{T},C)=\frac{1}{n^{2}}\sum_{i=1}^{n}\sum_{j=1}^{n}\left[\log Q_{\psi}\left(\hat{t}_{i}\mid c_{i}\right)-\log Q_{\psi}\left(\hat{t}_{j}\mid c_{i}\right)\right]}. | 
 | 
 (92) | 
 

 where, C = f θ ​ ( X ) C=f_{\theta}(X) . We adopt an alternating training strategy to iteratively optimize Q ψ ​ ( T ^ ∣ C ) Q_{\psi}(\hat{T}\mid C) and the network C = f θ ⁡ ( X ) C=f_{\theta(X)} to implement balanced representation in the Confounder Balancing.

 
 
 Outcome Regression: Finally, we propose to regress the outcome with the estimated treatment T ^ ∼ P ⁡ ( T | Z , X ) \hat{T}\sim P(T|Z,X) obtained in treatment regression module and the representation of confounders C = f θ ​ ( X ) C=f_{\theta}(X) obtained in confounder balancing module:

 

 
 | 
 ℒ Y = 1 n ​ ∑ i = 1 n ( y i − h ξ ​ ( t ^ i , f θ ​ ( x i ) ) ) 2 \displaystyle\mathcal{L}_{Y}=\scalebox{1.0}{$\frac{1}{n}\sum\limits_{i=1}^{n}\left(y_{i}-h_{\xi}(\hat{t}_{i},f_{\theta}(x_{i}))\right)^{2}$} | 
 | 
 (93) | 
 

 where t ^ i ∼ P ^ ​ ( T | Z , X ) \hat{t}_{i}\sim\hat{P}(T|Z,X) and f θ ​ ( x i ) f_{\theta}(x_{i}) are derived from treatment regression module and confounder balancing module, respectively.

 
 
 Theoretically and empirically, CBIV confirms that eliminating confounding bias in the outcome regression stage will contribute to more accurate treatment effect estimation.

 
 
 
 

### IV-D Limitation and Future Work 

 

#### IV-D 1 Limitation

 
 Invalid IV. The above methods are reliable only if the pre-defined IVs are valid and strongly correlated with the treatment variable. However, such valid IVs are hardly satisfied due to the untestable exclusion association with outcome [ 54 ] . Therefore, we have to rely on expert knowledge to select the instrumental variables, but this often does not guarantee the validity of the instrumental variables: IV does not have a direct effect on the outcome variable, only indirectly through the treatment variable. As an alternative, in instrumental variable literature, researchers usually implement Randomized Controlled Trials (RCTs) to obtain exogenous IVs, such as Oregon health insurance experiment [ 23 ] and effects of military service on lifetime earnings [ 24 ] , which are too expensive to be universally available.

 
 
 Weak IV and Mis-specified Model. In the real world, ones always consider a large number of variables (i.e., pre-treatment variables) that are relevant to the outcome and then choose treatments in the hope of obtaining the optimal results.
The instrumental variables are usually only a few, or even non-existent. Besides, the potential mechanisms of data generation are complex, and there may be interactions between various variables, such as T = Z ​ X + X + U T=ZX+X+U . In other words, IVs may have little causal effect on the treatment variables, which we call weak IV. Therefore, machine learning algorithms tend to combine observed confounders and IVs to predict the conditional distribution of the treatments to eliminate unmeasured confounding bias. Wu et al. [ 54 ] points out that these methods would make the predicted
treatments T ^ \hat{T} correlate with the observed variables X X and imbalanced variables X X will bring additional confounding bias for outcome regression, if the outcome model is misspecified (Fig.  9 ).

 
 
 Limited Sample. Machine learning algorithms are data-driven algorithms, and their performance is highly dependent on the number of samples. When the sample size is infinite, we can obtain unbiased estimates by the above algorithm. However, in finite samples, machine learning algorithms are prone to overfitting, leading to errors in the regression of the intervening variables, which will further lead to failure in the regression of the resulting coefficients. In addition, imbalanced covariates can also induce overfitting and introduce sample selection bias.

 
 
 

#### IV-D 2 Future Work

 
 Causal Discovery. When we have access to a large number of variables, we can try to mine the instrumental variables from the data by using causal discovery algorithms with latent variable, including constraint-based methods, score-based methods and model-based methods, such as SCORE [ 107 ] .

 
 
 Generalized Method of Moments. 
GMM is an incredibly flexible IV estimator that relies on a large number of moment conditions with IV independence conditions.
With the advancement of machine learning algorithms, nonlinear independence detection algorithms have also been developed, which has outperformed first-order moment independent etc.
Therefore, a natural idea is to use independent testing algorithms instead of moment conditions to constrain the instrumental variable regression, such as HSIC-X [ 108 ] 

 
 
 Confounder Balance. 
In the presence of unmeasured confounders and the above IV methods raises a very interesting bias problem in non-linear IV methods. These methods would suffer from the bias from the observed confounders, which are imbalanced in the second stage of IV regression. To address this problem, CBIV [ 54 ] proposes a confounder balanced IV regression algorithm by a novel combination of the confounder balancing and IV regression, where the confounder balancing is designed for removing the bias from the observed confounders and the IV regression is for removing the bias from the unobserved variables. In the provided theoretical analyses and numerical experiments, [ 54 ] demonstrates the effectiveness of the proposed algorithm. In the future, confounder balance is an issue that has to be considered in instrumental variable regression.

 
 
 
 
 

## V Control Function 

 
 Another statistical method to correct for unmeasured confounding bias is control function (CFN), also know as two-stage residual inclusion. The principle of control function can be traced back to some early works 5 5 
 5 
 
 
 
 Based on [ 66 ] . [ 109 , 110 ] , a control function is a variable that renders known cause variables (i.e., Treatments) appropriately exogenous in the outcome regression [ 111 , 110 , 112 ] . In observational data, conditional on control function or confounders 6 6 
 6 
 
 
 
 Under the unconfoundedness assumption, the role of control function in regression is consistent with that of confounding variables , CFN estimator makes the treatment appropriately exogenous in the regression queation. CFN is a two-stage residual inclusion method, which deponds on the parameters estimated by treatments T T and valid IVs Z Z in stage 1 [ 69 ] . And, it is not only useful in linear cases, but also in the non-linear scenarios to elimate bias for endogeneity.

 
 
 In stage 1, based on the variation induced by exogenous IVs in the treatment regression from IVs Z Z to treatments T T , we can obtain a generalized residual that serves as control function. As for stage 2, conditional on control function estimated in stage 1, the treatment become appropriately exogenous in the outcome regression. Next, we show how CFN regression works in cansal inference and machine learning, including linear and non-linear scenarios, as shown in Fig. 10 .

 
 
 Fig. 10 : Categorization of Control Function Estimators. 
 
 

### V-A Linear-based CFN 

 

#### V-A 1 Control Function Estimations

 
 For the most part, the usage of CFN maintains the spirit of the earlier definitions and estimations [ 66 ] . In the presence of unmeasured confounders 𝐔 \mathbf{U} , we assume 𝐕 = f ⁡ ( 𝐔 ) \mathbf{V}=f({\mathbf{U}}) as unmeasured noise for treatments and model structural linearity function in constant coefficients:

 

 
 | 
 T \displaystyle{T} | 
 = \displaystyle= | 
 Z ​ α + f ⁡ ( 𝐔 ) = Z ​ α + 𝐕 , \displaystyle{Z}\alpha+f({\mathbf{U}})={Z}\alpha+\mathbf{V}, | 
 | 
 (94) | 
 
 
 | 
 Y \displaystyle{Y} | 
 = \displaystyle= | 
 T ​ β + 𝐔 = T ​ β + f − 1 ​ ( 𝐕 ) , \displaystyle{T}\beta+{\mathbf{U}}={T}\beta+f^{-1}(\mathbf{V}), | 
 | 
 (95) | 
 

 where instrumental variables are independent of unmearsured confoudners, i.e., 𝔼 ⁡ ( Z ​ 𝐔 ) = 0 \mathbb{E}(Z\mathbf{U})=0 and 𝔼 ⁡ ( 𝐔 ∣ Z ) = 𝔼 ⁡ ( 𝐔 ) \mathbb{E}(\mathbf{U}\mid Z)=\mathbb{E}(\mathbf{U}) . Similarity, the IVs are uncorrelated with f ⁡ ( 𝐔 ) f(\mathbf{U}) . In linearity, we model the 𝐔 \mathbf{U} - T T association (i.e., the residuals) as 𝐕 = f ⁡ ( 𝐔 ) = 𝐔 / ρ \mathbf{V}=f({\mathbf{U}})=\mathbf{U}/\rho , and f − 1 f^{-1} is the inverse function of association f f . Then we can obtain:

 

 
 | 
 f − 1 ​ ( 𝐕 ) = ρ ​ 𝐕 , \displaystyle f^{-1}(\mathbf{V})=\rho\mathbf{V}, | 
 | 
 (96) | 
 

 where ρ \rho is the population regression coefficient. We plug it into the Eq. ( 95 ):

 

 
 | 
 Y = T ​ β + ρ ​ 𝐕 . \displaystyle{Y}={T}\beta+\rho\mathbf{V}. | 
 | 
 (97) | 
 

 
 
 In the observational data 𝒟 = { Z , 𝐔 , T , Y } \mathcal{D}=\{Z,\mathbf{U},T,Y\} , we do not observe 𝐔 \mathbf{U} or the residuals 𝐕 = f ⁡ ( 𝐔 ) \mathbf{V}=f({\mathbf{U}}) . Nevertheless, based on Eq. ( 94 ), we can get 𝐕 = T − Z ​ α \mathbf{V}=T-{Z}\alpha . Because Z Z is uncorrelated with 𝐕 \mathbf{V} in the linear model, we can consistently estimate the coefficient α \alpha by OLS. The two-step control function procedure is as follows:

 
 
 The Residual Learning Stage: in stage 1 of CFN, we perfrom the regression of the treatments T T on exogenous IVs Z Z :

 

 
 | 
 α ^ = ( Z ′ ​ Z ) − 1 ​ Z ′ ​ T = ( Z ′ ​ Z ) − 1 ​ Z ′ ​ ( Z ​ α + 𝐕 ) = α , \displaystyle\hat{\alpha}=\left({Z}^{\prime}{Z}\right)^{-1}{Z}^{\prime}{T}=\left({Z}^{\prime}{Z}\right)^{-1}{Z}^{\prime}({Z}\alpha+\mathbf{\mathbf{V}})=\alpha, | 
 | 
 (98) | 
 
 
 | 
 𝐕 ^ = T − Z ​ α ^ = 𝐕 . \displaystyle\hat{\mathbf{\mathbf{V}}}={T}-{Z}\hat{\alpha}=\mathbf{V}. | 
 | 
 (99) | 
 

 
 
 The Outcome Regression Stage: in stage 2 of CFN, based on the association between residuals 𝐕 \mathbf{V} and unmeasured confounders 𝐔 \mathbf{U} , we can regard residuals 𝐕 \mathbf{V} as a control function for unmeasured confounders. Then we can control the residuals 𝐕 \mathbf{V} to estimate the conditional average causal effect of treatments T T on outcomes Y Y :

 

 
 | 
 CATE = 𝔼 ⁡ [ Y ⁡ ( T = t ) − Y ⁡ ( T = 0 ) ∣ 𝐕 ] . \displaystyle\text{CATE}=\mathbb{E}[Y(T=t)-Y(T=0)\mid\mathbf{V}]. | 
 | 
 (100) | 
 

 or dose-response function (ITE):

 

 
 | 
 ITE = Y ⁡ ( T = t , 𝐕 ) − Y ⁡ ( T = 0 , 𝐕 ) . \displaystyle\text{ITE}=Y(T=t,\mathbf{V})-Y(T=0,\mathbf{V}). | 
 | 
 (101) | 
 

 
 
 The coefficents on Z Z and T T from CFN estimator arenumerically identical to that of 2SLS estimator [ 113 ] . In above linear setting, CFN estimator does not lead to a novel estimator different from 2SLS. In fact, if we perform OLS in the outcome regression, we find it is hard to obtain unbiased causal effect and we need to control the CFN/confounders.

 
 
 

#### V-A 2 Binary/Discrete treatment effects

 
 Binary/Discrete Treatment T = { 0 , 1 } T=\{0,1\} is a special case for CFN. When the treatment is a binary random variable, that is also applicable to discrete variables, a choice is to utilize the binary nature of treatment T T and replace the linear regression with a binary response model. The structural equation is supplemented with the continuous models in Eqs. ( 94 ) and ( 95 ):

 

 
 | 
 T \displaystyle{T} | 
 = \displaystyle= | 
 𝟙 { Z α + 𝐕 0 } , \displaystyle\mathbbm{1}\{{Z}\alpha+\mathbf{V} 0\}, | 
 | 
 (102) | 
 
 
 | 
 Y \displaystyle{Y} | 
 = \displaystyle= | 
 T ​ β + 𝐔 , \displaystyle{T}\beta+\mathbf{U}, | 
 | 
 (103) | 
 

 where 𝟙 ​ { ⋅ } \mathbbm{1}\{\cdot\} is the indicator function, and { 𝐔 , 𝐕 } \{\mathbf{U},\mathbf{V}\} are independent of Z Z . There is a linear causal relationship between 𝐔 \mathbf{U} and 𝐕 \mathbf{V} . Without loss of generality, we assume that the residual satisfies 𝐕 ∼ 𝒩 ⁡ ( 0 , 1 ) \mathbf{V}\sim\mathcal{N}(0,1) . Thus, the treatment assignment can be regarded as a probit model:

 

 
 | 
 P ⁡ ( T = 1 ∣ Z ) = Φ ⁡ ( Z ​ α ) , \displaystyle P(T=1\mid Z)=\Phi(Z\alpha), | 
 | 
 (104) | 
 

 where Φ ⁡ ( ⋅ ) \Phi(\cdot) is the standard normal cumulative distribution function. Then we can derive a CFN for binary treatment cases [ 65 , 98 ] .

 
 
 In stage 1, we estimate the probit model in Eq. ( 104 ) and obtain the generalized residual :

 

 
 | 
 r 𝐕 ^ = T ​ λ ​ ( Z ​ α ) − ( 1 − T ) ​ λ ​ ( − Z ​ α ) \displaystyle\hat{r_{\mathbf{V}}}=T\lambda(Z\alpha)-(1-T)\lambda(-Z\alpha) | 
 | 
 (105) | 
 

 where λ ​ ( ⋅ ) = ϕ Φ ​ ( ⋅ ) \lambda(\cdot)=\frac{\phi}{\Phi}(\cdot) is the well-known inverse Mills ratio [ 114 ] .

 
 
 In stage 2, we control the generalized residual r 𝐕 r_{\mathbf{V}} to estimate the conditional average causal effect of treatments T T on outcomes Y Y :

 

 
 | 
 CATE = 𝔼 ⁡ [ Y ⁡ ( T = t ) − Y ⁡ ( T = 0 ) ∣ r 𝐕 ] . \displaystyle\text{CATE}=\mathbb{E}[Y(T=t)-Y(T=0)\mid r_{\mathbf{V}}]. | 
 | 
 (106) | 
 

 One limitation for CFN in binary/discrete treatment cases is that the results is reliable only when the designed probit model for T T is correct. If the probit model is correctly specified, then the CFN estimator would give an unbias causal effect.

 
 
 

#### V-A 3 Heterogeneous treatment effects

 
 When the coefficients in the structural function is correlated the treatment variable, there are heterogeneous treatment effects in observational data. The random coeffient setting is called a ”correlated random coefficient” (CRC) model [ 115 , 116 ] . Consider the outcome structural function as:

 

 
 | 
 T = Z ​ α + 𝐕 , \displaystyle T=Z\alpha+\mathbf{V}, | 
 | 
 (107) | 
 
 
 | 
 Y = T ​ U 1 + U 2 , \displaystyle Y=TU_{1}+U_{2}, | 
 | 
 (108) | 
 

 where all unobservables are independent of IVs, i.e., Z ⟂ { U 1 , U 2 , 𝐕 } Z\perp\{U_{1},U_{2},\mathbf{V}\} , and the unobservables U 1 U_{1} and U 2 U_{2} are linearly correlated with the residual 𝐕 \mathbf{V} :

 

 
 | 
 𝔼 ⁡ [ U 1 ∣ 𝐕 ] = η ​ 𝐕 + β , 𝔼 ⁡ [ U 2 ∣ 𝐕 ] = ψ ​ 𝐕 + c , \displaystyle\mathbb{E}[U_{1}\mid\mathbf{V}]=\eta\mathbf{V}+\beta,\mathbb{E}[U_{2}\mid\mathbf{V}]=\psi\mathbf{V}+c, | 
 | 
 (109) | 
 

 where { β , c } \{\beta,c\} are constant terms, and { η , ψ } \{\eta,\psi\} are the corresponding regression coefficients.

 
 
 In the heterogeneous treatment effects dataset, there are two sources of unmeasured confounding bias from U 1 U_{1} and U 2 U_{2} . In this cases, we focus on the average treatment effect, i.e., β = 𝔼 ⁡ ( U 1 ) \beta=\mathbb{E}(U_{1}) . Then, we set U 1 = 𝔼 ⁡ ( U 1 ) + R , 𝔼 ⁡ ( R ) = 0 U_{1}=\mathbb{E}(U_{1})+R,\mathbb{E}(R)=0 , and reformulate the outcome structural function as:

 

 
 | 
 Y \displaystyle Y | 
 = \displaystyle= | 
 T ​ β + T ​ R + U 2 , \displaystyle T\beta+TR+U_{2}, | 
 | 
 (110) | 
 

 where 𝔼 ⁡ [ R ] = η ​ 𝐕 \mathbb{E}[R]=\eta\mathbf{V} and the correlation between T T and R R satisfies the assumption: Cov ​ ( T , R ∣ Z ) = Cov ​ ( T , R ) \text{Cov}(T,R\mid Z)=\text{Cov}(T,R) [ 116 ] . Then we formulate the CFN estimator as:

 

 
 | 
 𝔼 [ Y ( T ) ∣ U 1 , U 2 ] \displaystyle\mathbb{E}[Y(T)\mid U_{1},U_{2}] | 
 = \displaystyle= | 
 𝔼 [ Y ( T ) ∣ 𝐕 , T 𝐕 ] \displaystyle\mathbb{E}[Y(T)\mid\mathbf{V},T\mathbf{V}] | 
 | 
 (111) | 

 
 | 
 | 
 = \displaystyle= | 
 T ​ β + η ​ T ​ 𝐕 + ψ ​ 𝐕 + c . \displaystyle T\beta+\eta T\mathbf{V}+\psi\mathbf{V}+c. | 
 | 
 

 
 
 In stage 1, we regress the treatment T T on the exogenous IVs Z Z :

 

 
 | 
 α ^ = ( Z ′ ​ Z ) − 1 ​ Z ′ ​ T = ( Z ′ ​ Z ) − 1 ​ Z ′ ​ ( Z ​ α + 𝐕 ) = α \displaystyle\hat{\alpha}=\left({Z}^{\prime}{Z}\right)^{-1}{Z}^{\prime}{T}=\left({Z}^{\prime}{Z}\right)^{-1}{Z}^{\prime}({Z}\alpha+\mathbf{\mathbf{V}})=\alpha | 
 | 
 (112) | 
 

 Thus, the residual is.

 

 
 | 
 𝐕 ^ = T − Z ​ α ^ = 𝐕 . \displaystyle\hat{\mathbf{\mathbf{V}}}={T}-{Z}\hat{\alpha}=\mathbf{V}. | 
 | 
 (113) | 
 

 
 
 In stage 2, we control the residual 𝐕 \mathbf{V} and the multiplicative interaction T ​ 𝐕 T\mathbf{V} to estimate the conditional average causal effect of treatments T T on outcomes Y Y :

 

 
 | 
 CATE = 𝔼 [ Y ( T = t ) − Y ( T = 0 ) ∣ 𝐕 , T 𝐕 ] . \displaystyle\text{CATE}=\mathbb{E}[Y(T=t)-Y(T=0)\mid\mathbf{V},T\mathbf{V}]. | 
 | 
 (114) | 
 

 Similar CFNs are also applicable to discrete treatment cases.

 
 
 
 

### V-B NonLinear-based CFN 

 
 In the previous section, we have introduced contron function methods employed for linear models, including Probit and Tobit. [ 117 , 118 , 66 ] broaden the scope of the CFN applications. Here, we detail the flexibility of the CFN estimator in the complex non-linear models using machine learning methods.

 
 
 Consider a simple nonlinear model (observed confounders 𝐗 \mathbf{X} includes a multiplicative interaction T ​ 𝐗 T\mathbf{X} ):

 

 
 | 
 T \displaystyle T | 
 = \displaystyle= | 
 Z ​ α 1 + 𝐗 ​ α 2 + 𝐕 , \displaystyle Z\alpha_{1}+\mathbf{X}\alpha_{2}+\mathbf{V}, | 
 | 
 (115) | 
 
 
 | 
 Y \displaystyle Y | 
 = \displaystyle= | 
 𝐗 ​ β 1 + T ​ 𝐗 ​ β 2 + 𝐔 , 𝐔 = 𝐕 ​ ρ . \displaystyle\mathbf{X}\beta_{1}+T\mathbf{X}\beta_{2}+\mathbf{U},\mathbf{U}=\mathbf{V}\rho. | 
 | 
 (116) | 
 

 According to the IV’s three conditions, we have that Z ⟂ { 𝐗 , 𝐔 , 𝐕 } Z\perp\{\mathbf{X},\mathbf{U},\mathbf{V}\} . In this model, the treatment is continuous, then we obtain the residual in the stage 1.

 

 
 | 
 𝐕 ^ = T − 𝔼 [ T ∣ Z , 𝐗 ] = T − ( Z , 𝐗 ) ( α 1 ^ , α 2 ^ ) ′ = 𝐕 \displaystyle\hat{\mathbf{V}}=T-\mathbb{E}[T\mid Z,\mathbf{X}]=T-(Z,\mathbf{X})(\hat{\alpha_{1}},\hat{\alpha_{2}})^{\prime}=\mathbf{V} | 
 | 
 (117) | 
 

 where ( Z , 𝐗 ) (Z,\mathbf{X}) denotes the joint vector of Z Z and 𝐗 \mathbf{X} , and ( α 1 ^ , α 2 ^ ) (\hat{\alpha_{1}},\hat{\alpha_{2}}) is the corresponding coefficients. Sequentially, we can perform the outcome regression on 𝐕 \mathbf{V} , 𝐗 \mathbf{X} , and the interaction T ​ 𝐗 T\mathbf{X} :

 

 
 | 
 CATE = 𝔼 [ Y ( T = t ) − Y ( T = 0 ) ∣ 𝐕 , 𝐗 , T 𝐗 ] . \displaystyle\text{CATE}=\mathbb{E}[Y(T=t)-Y(T=0)\mid\mathbf{V},\mathbf{X},T\mathbf{X}]. | 
 | 
 (118) | 
 

 A similar estimator can be built for a discrete treatment case, in the discrete model Eq. ( 106 ) [ 119 , 65 ] . The limitation is that the results are reliable only when we have modeled the correct model for non-linear relationship with the prior knowledge of interaction T ​ 𝐗 T\mathbf{X} . In the next section, we will give a general solution through probit models.

 
 

#### V-B 1 Non-Parametric BP Estimator

 
 For more general models, there may be some more complex non-linear relationship in the causal structural function. Based on the probit model [ 118 ] , Blundell and Powell (BP) [ 64 ] proposes a non-parametric extension of the Rivers-Vuong approach [ 118 ] , which is applicable in most general setting:

 

 
 | 
 T \displaystyle T | 
 = \displaystyle= | 
 f ⁡ ( Z , 𝐗 ) + 𝐕 , \displaystyle f(Z,\mathbf{X})+\mathbf{V}, | 
 | 
 (119) | 
 
 
 | 
 Y \displaystyle Y | 
 = \displaystyle= | 
 g ⁡ ( 𝐗 , T , 𝐔 ) . \displaystyle g(\mathbf{X},T,\mathbf{U}). | 
 | 
 (120) | 
 

 where f ⁡ ( ⋅ ) f(\cdot) and g ⁡ ( ⋅ ) g(\cdot) are the structural functions. The target of BP approach is to estimate the Average Structural Function (ASF) of outcome, defined as follows:

 

 
 | 
 ASF ( 𝐗 , T ) = 𝔼 [ g ( 𝐗 , T , 𝐔 ) ∣ 𝐗 , T ] . \displaystyle\text{ASF}(\mathbf{X},T)=\mathbb{E}[g(\mathbf{X},T,\mathbf{U})\mid\mathbf{X},T]. | 
 | 
 (121) | 
 

 The notation means that the unmeasured confounders 𝐔 \mathbf{U} are averaged out in the population conditional on the fixed 𝐗 \mathbf{X} and T T , i.e., 𝔼 𝐔 [ g ( 𝐗 , T , 𝐔 ) ] = 𝔼 [ g ( 𝐗 , T , 𝐔 ) ∣ 𝐗 , T ] \mathbb{E}_{\mathbf{U}}[g(\mathbf{X},T,\mathbf{U})]=\mathbb{E}[g(\mathbf{X},T,\mathbf{U})\mid\mathbf{X},T] .

 
 
 BP Model .

 
 
 In the first stage , we can obtain the residual 𝐕 \mathbf{V} from 𝐕 = T − f ⁡ ( Z , 𝐗 ) \mathbf{V}=T-f(Z,\mathbf{X}) , and f ⁡ ( Z , 𝐗 ) f(Z,\mathbf{X}) can be identified by f ( Z , 𝐗 ) = 𝔼 [ T ∣ Z , 𝐗 ] f(Z,\mathbf{X})=\mathbb{E}[T\mid Z,\mathbf{X}] :

 

 
 | 
 𝐕 ^ = T − 𝔼 [ T ∣ Z , 𝐗 ] = T − f ^ ( Z , 𝐗 ) . \displaystyle\hat{\mathbf{V}}=T-\mathbb{E}[T\mid Z,\mathbf{X}]=T-\hat{f}(Z,\mathbf{X}). | 
 | 
 (122) | 
 

 where we can use machine learning methods to estimate the expectation 𝔼 [ T ∣ Z , 𝐗 ] \mathbb{E}[T\mid Z,\mathbf{X}] , such as kernel-based regression and neural networks regression.

 
 
 In the second stage , the conditional distribution of the unmeasured confounders 𝐔 \mathbf{U} is related to { Z , 𝐗 , T } \{Z,\mathbf{X},T\} only through the residual 𝐕 \mathbf{V} [ 120 , 66 ] :

 

 
 | 
 P ⁡ ( 𝐔 ∣ Z , 𝐗 , T ) = P ⁡ ( 𝐔 ∣ Z , 𝐗 , 𝐕 ) = P ⁡ ( 𝐔 ∣ 𝐕 ) . \displaystyle P(\mathbf{U}\mid Z,\mathbf{X},T)=P(\mathbf{U}\mid Z,\mathbf{X},\mathbf{V})=P(\mathbf{U}\mid\mathbf{V}). | 
 | 
 (123) | 
 

 Then, the consistent estimator of the ASF is:

 

 
 | 
 g ^ ′ ( 𝐗 , T , 𝐕 ) = 𝔼 [ Y ∣ 𝐗 , T , 𝐕 ] , \displaystyle\hat{g}^{\prime}(\mathbf{X},T,\mathbf{V})=\mathbb{E}[Y\mid\mathbf{X},T,\mathbf{V}], | 
 | 
 (124) | 
 
 
 | 
 ASF ( 𝐗 , T ) = 𝔼 [ g ^ ′ ( 𝐗 , T , 𝐕 ) ∣ 𝐗 , T ] = 𝔼 𝐕 [ g ^ ′ ( 𝐗 , T , 𝐕 ) ] , \displaystyle\text{ASF}(\mathbf{X},T)=\mathbb{E}[\hat{g}^{\prime}(\mathbf{X},T,\mathbf{V})\mid\mathbf{X},T]=\mathbb{E}_{\mathbf{V}}[\hat{g}^{\prime}(\mathbf{X},T,\mathbf{V})], | 
 | 
 (125) | 
 
 
 | 
 ASF ^ ​ ( 𝐗 , T ) = 1 n ​ ∑ i = 1 n g ^ ′ ​ ( 𝐗 , T , 𝐕 ) . \displaystyle\hat{\text{ASF}}(\mathbf{X},T)=\frac{1}{n}\sum_{i=1}^{n}\hat{g}^{\prime}(\mathbf{X},T,\mathbf{V}). | 
 | 
 (126) | 
 

 where we can use machine learning methods to estimate the expectation g ^ ′ ​ ( 𝐗 , T , 𝐕 ) \hat{g}^{\prime}(\mathbf{X},T,\mathbf{V}) , such as kernel-based regression and neural networks regression.

 
 
 

#### V-B 2 General CFN Estimator

 
 Althrough CFN estimators have been widely used for solving the unmeasured confounders in causal inference, one critical limitation is that CFN usually breakdown under complex non-linear models. Besided, CFN requires that the residual obtained from the treatment outcome regression is linearly related to the unmeasured confounder, i.e., the structural treatment process assumptions, and the results is reliable only when the models are specified correctly.

 
 
 Based on the concept of variational autoencoder (VAE) [ 121 ] , some works study the proxy variable for unmeasured confounders and try to use the proxy to reconstruct the unmeasured confounders [ 122 , 123 , 124 ] . Motivated by this, [ 67 ] develop the general control function method (GCFN) to construct general control functions and estimate effects.

 
 
 With the control function that satisfies the ignorability and positivity assumptions, GCFN does not need the additive separation assumption and simplify the causal effect estimation as outcome regression on the treatment and the control function. The observation data can be sampled from:

 

 
 | 
 T \displaystyle T | 
 = \displaystyle= | 
 f ⁡ ( Z , 𝐗 , 𝐕 ) , \displaystyle f(Z,\mathbf{X},\mathbf{V}), | 
 | 
 (127) | 
 
 
 | 
 Y \displaystyle Y | 
 = \displaystyle= | 
 g ⁡ ( 𝐗 , T , 𝐔 ) . \displaystyle g(\mathbf{X},T,\mathbf{U}). | 
 | 
 (128) | 
 

 Then, the control functions can be characterized:

 
 
 Theorem V.1 . 
 
 Meta-identification. The causal effect is identified by the joint distribution q ⁡ ( Z , 𝐗 , 𝐕 , T ) q(Z,\mathbf{X},\mathbf{V},T) over the control function 𝐕 ^ \hat{\mathbf{V}} and the observables { Z , 𝐗 , T } \{Z,\mathbf{X},T\} : 

 

 
 | 
 𝔼 𝐕 ^ [ Y ∣ T , 𝐕 ^ ] = 𝔼 𝐕 ^ [ Y ∣ do ( T ) , 𝐕 ^ ] = 𝔼 [ Y ∣ do ( T ) ] . \displaystyle\mathbb{E}_{\hat{\mathbf{V}}}[Y\mid T,\hat{\mathbf{V}}]=\mathbb{E}_{\hat{\mathbf{V}}}[Y\mid\text{do}(T),\hat{\mathbf{V}}]=\mathbb{E}[Y\mid\text{do}(T)]. | 
 | 
 (129) | 
 

 With the following assumptions: 

 
 • 
 
 (A1) 𝐕 ^ \hat{\mathbf{V}} satisfies the reconstruction property: the treatment T T can be represented by { Z , 𝐗 , 𝐕 ^ } \{Z,\mathbf{X},\hat{\mathbf{V}}\} ; 

 

 • 
 
 (A2) The IVs Z Z are independent of control functions, confounders and residuals, i.e., Z ⟂ { 𝐗 , 𝐔 , 𝐕 , 𝐕 ^ } Z\perp\{\mathbf{X},\mathbf{U},\mathbf{V},\hat{\mathbf{V}}\} ; 

 

 • 
 
 (A3) Fixing the general control function 𝐕 ^ \hat{\mathbf{V}} , the strong IVs can set treatment to any value. 

 

 
 
 
 
 Then, the control function 𝐕 ^ \hat{\mathbf{V}} satisfies ignorability and positivity:

 

 
 | 
 q ⁡ ( Y ∣ T , 𝐕 ^ ) = q ⁡ ( Y ∣ do ​ ( T ) , 𝐕 ^ ) , \displaystyle q(Y\mid T,\hat{\mathbf{V}})=q(Y\mid\text{do}(T),\hat{\mathbf{V}}), | 
 | 
 (130) | 
 
 
 | 
 q ⁡ ( 𝐕 ^ ) 0 ⇒ q ⁡ ( T ∣ 𝐕 ^ ) 0 . \displaystyle q(\hat{\mathbf{V}}) 0\Rightarrow q(T\mid\hat{\mathbf{V}}) 0. | 
 | 
 (131) | 
 

 
 
 GCFN . 
 Following [ 19 , 122 ] and [ 125 ] , GCFN’s first stage called variational decoupling (VDE) constructs general control functions by using VAE and recovering the residual variation in the treatment given the IV. This yields an evidence lower bound (ELBO) of VAE to reconstruct the latent variables:

 

 
 | 
 | 
 | 
 L ( θ , ϕ , ξ ∣ Z , 𝐗 , T ) \displaystyle L(\theta,\phi,\xi\mid Z,\mathbf{X},T) | 
 | 
 (132) | 

 
 | 
 | 
 = \displaystyle= | 
 ( 1 + λ ) ​ 𝔼 q θ ​ ( 𝐕 ∣ Z , 𝐗 , T ) ​ log ​ p ϕ ​ ( T ∣ Z , 𝐗 , 𝐕 ) \displaystyle(1+\lambda)\mathbb{E}_{q_{\theta}(\mathbf{V}\mid Z,\mathbf{X},T)}\text{log}{p_{\phi}}(T\mid Z,\mathbf{X},\mathbf{V}) | 
 | 

 
 | 
 | 
 − \displaystyle- | 
 λ D K ​ L ( q θ ( 𝐕 ∣ Z , 𝐗 , T ) ∥ p ξ ( 𝐕 ) ) \displaystyle\lambda D_{KL}(q_{\theta}(\mathbf{V}\mid Z,\mathbf{X},T)\|p_{\xi}(\mathbf{V})) | 
 | 
 

 where λ \lambda is the hype-parameter that is used to balance the reconstruction term and the KL term in the beta-VAE. p ξ ​ ( 𝐕 ) p_{\xi}(\mathbf{V}) and p ϕ ​ ( T ∣ Z , 𝐗 , 𝐕 ) {p_{\phi}}(T\mid Z,\mathbf{X},\mathbf{V}) are real (posterior) probability distributions, q θ ​ ( 𝐕 ∣ Z , 𝐗 , T ) q_{\theta}(\mathbf{V}\mid Z,\mathbf{X},T) is the estimated probability distributions by neural networks with parameter θ \theta . D K ​ L ​ ( ⋅ ) D_{KL}(\cdot) denotes the Kullback-Leibler (KL) divergence. By maximizing the above objective function, we can sample the control function 𝐕 ^ \hat{\mathbf{V}} from the observables { Z , 𝐗 , T } \{Z,\mathbf{X},T\} .

 
 
 VDE provides a general control function 𝐕 ^ \hat{\mathbf{V}} and its marginal distribution q θ ​ ( 𝐕 ) q_{\theta}(\mathbf{V}) . Using VDE’s control function, GCFN’s second stage estimates effects via regression. Other confounder adjusting/control methods like matching/balancing methods [ 39 , 126 , 12 ] , doubly robust methods [ 10 ] and representation learning methods [ 36 , 16 , 37 , 127 ] can be used for outcome regression:

 

 
 | 
 CATE = 𝔼 [ Y ( T = t ) − Y ( T = 0 ) ∣ 𝐕 , 𝐗 ] . \displaystyle\text{CATE}=\mathbb{E}[Y(T=t)-Y(T=0)\mid\mathbf{V},\mathbf{X}]. | 
 | 
 (133) | 
 

 
 
 Further, [ 67 ] develop semi-supervised GCFN to construct general control functions using subsets of data that have both IV and confounders observed as supervision; this needs no structural treatment process assumptions.

 
 
 

#### V-B 3 Conditional Variational Autoencoder Estimator

 
 Fig. 11 : Causal graph of Confounded IV. Dashed lines represent unknown causality. 
 
 
 Due to untestable Exclusion and Independent restrictions, finding a valid IV is always a tricky problem. To relax the restriction, Wang et al. [ 128 ] focus on estimating treatment effects with more accessible confounded instruments that violate the unconfounded instruments assumption, i.e., { Z 1 , Z 2 , ⋯ , Z m } ⟂̸ 𝐔 \{Z_{1},Z_{2},\cdots,Z_{m}\}\not\perp\mathbf{U} . Inspired by deep conditional variational autoencoder, they aim to generate a
substitute of unmeasured confounder that obeys strong ignorability, such that Y ⟂ T | 𝐔 , 𝐗 {Y}\perp{T}\mid\mathbf{U},\mathbf{X} . To achieve the ignorability, CVAE-IV [ 128 ] model a substitute 𝐕 ^ \hat{\mathbf{V}} based on the statistical principle Y ⟂ { Z i } i = 1 m | T , 𝐗 , 𝐕 ^ Y\perp\left\{{Z}_{i}\right\}_{i=1}^{m}\mid T,\mathbf{X},\hat{\mathbf{V}} , which states that
the outcome and IV candidates are conditionally independent given the
treatment, observed covariates and the generated 𝐕 ^ \hat{\mathbf{V}} .

 
 
 CVAE-IV . 
 In the first stage , with multiple confounded IVs 𝐙 = { Z i } i = 1 m \mathbf{Z}=\left\{{Z}_{i}\right\}_{i=1}^{m} , as shown in Fig. 11 , CVAE-IV [ 128 ] constructs a conditional variational autoencoder to generate the
confounder substitute 𝐕 ^ \hat{\mathbf{V}} . Specifically, they apply the variational inference to model the conditional distribution P ( Y , 𝐙 ∣ T , 𝐗 ) P(Y,\mathbf{Z}\mid T,\mathbf{X}) as follow:

 

 
 | 
 log P ( Y , 𝐙 ∣ T , 𝐗 ) ≥ 𝔼 [ log P θ ( Y , 𝐙 ∣ T , 𝐗 , 𝐕 ^ ) ] \displaystyle\log P(Y,\mathbf{Z}\mid T,\mathbf{X})\geq\mathbb{E}\left[\log P_{\theta}\left({Y},{\mathbf{Z}}\mid T,\mathbf{X},\hat{\mathbf{V}}\right)\right] | 
 | 
 
 
 | 
 − D K ​ L ( Q ϕ ( 𝐕 ^ ∣ T , Y , 𝐙 , 𝐗 ) ∥ P ( 𝐕 ^ ∣ T , 𝐗 ) ) \displaystyle-D_{KL}\left(Q_{\phi}\left(\hat{\mathbf{V}}\mid T,Y,{\mathbf{Z}},\mathbf{X}\right)\|P\left(\hat{\mathbf{V}}\mid T,\mathbf{X}\right)\right) | 
 | 
 (134) | 
 

 where D K ​ L D_{KL} refers to the KL \mathrm{KL} -divergence between variational posterior and the underlying one, P θ P_{\theta} is the decoder model and Q ϕ Q_{\phi} is the encoder model. By forcing the underlying posterior P ⁡ ( 𝐕 ^ ∣ T , 𝐗 ) P\left(\hat{\mathbf{V}}\mid T,\mathbf{X}\right) to follow the normal distribution.

 
 
 We use networks f Y f_{Y} and f Z f_{Z} to regress the outcome and instruments as well as minimize the evidence lower bound (ELBO) of CVAE as objective to reconstruct the latent variables 𝐕 ^ \hat{\mathbf{V}} :

 

 
 | 
 ℒ \displaystyle\mathcal{L} | 
 = \displaystyle= | 
 ℒ R ​ e ​ c + ℒ C ​ h ​ o ​ l + λ ​ ℒ K ​ L \displaystyle\mathcal{L}_{Rec}+\mathcal{L}_{Chol}+\lambda\mathcal{L}_{KL} | 
 | 
 (135) | 
 
 
 | 
 ℒ R ​ e ​ c \displaystyle\mathcal{L}_{Rec} | 
 = \displaystyle= | 
 ∑ i n [ ( y i − f Y ​ ( t i , 𝐱 i , 𝐯 i ) ) 2 ] / V ​ a ​ r ​ ( Y ) , \displaystyle\sum_{i}^{n}[(y_{i}-f_{Y}(t_{i},\mathbf{x}_{i},\mathbf{v}_{i}))^{2}]/Var(Y), | 
 | 
 
 
 | 
 ℒ C ​ h ​ o ​ l \displaystyle\mathcal{L}_{Chol} | 
 = \displaystyle= | 
 ∑ i n [ ( 𝐳 i − f Z ​ ( t i , 𝐱 i , 𝐯 i ) ) 2 ] , \displaystyle\sum_{i}^{n}[(\mathbf{z}_{i}-f_{Z}(t_{i},\mathbf{x}_{i},\mathbf{v}_{i}))^{2}], | 
 | 
 
 
 | 
 ℒ K ​ L \displaystyle\mathcal{L}_{KL} | 
 = \displaystyle= | 
 D K ​ L ( Q ϕ ( 𝐕 ^ ∣ T , Y , 𝐙 , 𝐗 ) ∥ P ( 𝐕 ^ ∣ T , 𝐗 ) ) , \displaystyle D_{KL}\left(Q_{\phi}\left(\hat{\mathbf{V}}\mid T,Y,{\mathbf{Z}},\mathbf{X}\right)\|P\left(\hat{\mathbf{V}}\mid T,\mathbf{X}\right)\right), | 
 | 
 

 where the λ \lambda controls the variance of the reconstructed output.

 
 
 In the second stage , we fit the observational outcome using two regression functions g ψ 1 g_{\psi_{1}} and g ψ 2 g_{\psi_{2}} , which are parametrized by deep networks with ψ 1 \psi_{1} and ψ 2 \psi_{2} :

 

 
 | 
 ℒ R ​ e ​ g = ∑ i n [ ( y i − g ψ 1 ​ ( t i , 𝐱 i ) − g ψ 2 ​ ( 𝐯 i ) ) 2 ] . \displaystyle\mathcal{L}_{Reg}=\sum_{i}^{n}[(y_{i}-g_{\psi_{1}}(t_{i},\mathbf{x}_{i})-g_{\psi_{2}}(\mathbf{v}_{i}))^{2}]. | 
 | 
 (136) | 
 

 Then, we predict the counterfactual outcome Y ⁡ ( t ) Y(t) and CATE with the trained regression model { ψ 1 , ψ 2 } \{\psi_{1},\psi_{2}\} :

 

 
 | 
 Y ⁡ ( t , 𝐱 , 𝐯 ) = g ψ 1 ​ ( t , 𝐱 ) + g ψ 2 ​ ( 𝐯 ) , \displaystyle Y(t,\mathbf{x},\mathbf{v})=g_{\psi_{1}}(t,\mathbf{x})+g_{\psi_{2}}(\mathbf{v}), | 
 | 
 (137) | 
 
 
 | 
 C ​ A ​ T ​ E = Y ⁡ ( t , 𝐱 , 𝐯 ) − Y ⁡ ( 0 , 𝐱 , 𝐯 ) . \displaystyle CATE=Y(t,\mathbf{x},\mathbf{v})-Y(0,\mathbf{x},\mathbf{v}). | 
 | 
 (138) | 
 

 By constructing the CVAE-IV model to generate a ignorable confounder substitute, we
isolate the influence of the unmeasured confounder from the estimation on
conditional treatment effect.

 
 
 
 

### V-C Limitation and Future Work 

 

#### V-C 1 Limitation

 
 Inverse Relationship . In the structural assumption, CFN implicitly require
a one-to-one mapping (or Inverse Relationship) between the residuals 𝐕 \mathbf{V} from treatment regression and the unmeasured confounders 𝐔 \mathbf{U} . Otherwise, even if we recover the residuals perfectly, we cannot control the unmeasured confounders. For example, if 𝐕 = s ​ i ​ n ​ ( 𝐔 ) \mathbf{V}=sin(\mathbf{U}) , then we control for 𝐕 = 1 \mathbf{V}=1 , but 𝐔 \mathbf{U} still has infinitely many possibilities, which we cannot discuss and analyze.

 
 
 Invalid IV and Weak IV . The performance of these methods relies on the well-predefined IVs that satisfy three instruments restrictions (i.e., IV does not have a direct effect on the outcome variable, only indirectly through the treatment variable), which is untestable and leads to finding a valid IV becomes an art rather than science. Therefore, how to use invalid IV or wark IV to implement CFN is still an open problem.

 
 
 

#### V-C 2 Future Work

 
 Variational Autoencoder . Inverse relationship between the residuals 𝐕 \mathbf{V} and the unmeasured confounders 𝐔 \mathbf{U} means that we can achieve indirect control of 𝐔 \mathbf{U} by controlling the residuals 𝐕 \mathbf{V} . So, naturally, why don’t we just recover 𝐔 \mathbf{U} ? Based on the concept of variational autoencoder (VAE) [ 121 ] , some works study the proxy variable for unmeasured confounders and try to use the proxy to reconstruct the unmeasured confounders [ 122 , 123 , 124 ] . Motivated by this, [ 67 ] develop the general control function method (GCFN) to learn the distribution of unmeasured confoudners and estimate effects.

 
 
 Confounded IV . In reality, the acquisition of valid IV is a tricky project, so Wang et al. [ 128 ] proposes to use confounded IV, having a direct effect on the outcome variable but indirectly through the treatment and confounders, instead of valid IV to recover unmeasured confounders. By considering the conditional independence between confounded instruments and the outcomes, CVAE-IV [ 128 ] generates a substitute of the unmeasured confounder with a conditional variational autoencoder. Therefore, the exploration of invalid IV is a promising research line for the future.

 
 
 
 
 

## VI Evaluating Instrumental Variables 

 
 Fig. 12 : Evaluating of Instrumental Variables. 
 
 
 In Section IV V , we have introduce how to implement two-stage regression with IV for treatment effect estimation. One limitation is that, these methods require a strong and valid IV 7 7 
 7 
 
 
 
 The instrument must be correlated with the endogenous treatment variables. If this correlation is strong, then the instrument is said to have a strong first stage. A weak correlation may provide misleading inferences about parameter estimates and standard errors [ 129 ] . for treatment regression, which is rare in reality. In this Section, we summarize three methods for selecting IV, i.e., Lagged Values, Prior Knowledge of Causal Graph and Randomized Controlled Trials, and provide over-identification test for IV’s exclusion restriction.
Subsequently, we also introduce several machine learning algorithms for strong IV generation, i.e., Summary IVs. The overall skeleton is shown in the Fig. 12 .

 
 

### VI-A IV Selection 

 
 The above IV methods are reliable if and only if the IVs we found only affect the outcomes through its strong association with treatments.
Finding suitable IVs still is a challenge for the IV methods [ 130 ] . Next, we will introduce several methods to find or test IVs.

 
 
 Lagged Values . With panel data, a common strategy of finding IV is to use the lagged values as IVs for the current treatments [ 131 ] .
For example, [ 132 ] estimated the causal
effect of compulsory schooling on earnings by using quarter of
birth as an IV for education. [ 20 ] used characteristics of the respondent’s childhood, husband’s childhood, and parents and husband’s parent as IVs to predict the respondent’s probability to send their children to school in the future, and then used the predicted value from this model as an independent variable in the prediction of contraceptive use.

 
 
 Model Implied Instrumental Variables .
A second strategy draws IVs from among the observed variables is Model Implied Instrumental Variables (MIIVs), taken from [ 133 , 21 , 22 ] . In MIIVs, a prior knowledge of causal graph is used to build the model structure, which tells the researcher which observed variables can serve as IVs and which cannot. Closely related to the MIIV
method is the directed acyclical graph (DAG), [ 134 , 135 ] gave rules to select the variables that can serve as IVs: the correlation of a variable with the residual term of the outcome predict equation is zero [ 22 ] .

 
 
 Randomization Instrumental Variables .
In instrumental variable literature, researchers usually implement Randomized Controlled Trials (RCTs) to sample a random variable as IV to intervene the received treatments, called intention-to-treat variable, such as Oregon health insurance experiment [ 23 ] and effects of military service on lifetime earnings [ 24 ] , which are too expensive to be universally available.
Sometimes, there might be randomization introduced by “nature” [ 136 ] , called natural experiments, such as twin births, gender, and weather events.

 
 
 

### VI-B IV Evaluation 

 
 Regardless of IVs selected by which prior, we must evaluate the IVs’ quality: a valid IV that only affects the outcome through its strong association with treatment options, called exclusion assumption.
If the structure assumptions for IV are dissatisfied and the correlation is weak, then the instrument may provide misleading inferences about parameter estimates and standard errors [ 129 , 137 ] .

 
 
 Over-Identification Test . When the number of IVs is more than the need for just-identification, i.e., there are more IVs than the number of treatments, one can test the exogeneity of IVs. The over-identification tests construct a null hypothesis that all IVs are exogenous variables versus
the alternative hypothesis that at least one IV violates exogeneity (correlates with the residuals from the two-stage IV regression).
In linear setting, [ 138 ] gave a known over-identification tests for IVs:

 

 
 | 
 p = ϵ ′ ​ Z ¯ ​ ( Z ¯ ′ ​ Z ¯ ) − 1 ​ Z ¯ ′ ​ ϵ ϵ ′ ​ ϵ / n ∼ 𝒳 2 , \displaystyle p=\frac{\epsilon^{\prime}\bar{Z}(\bar{Z}^{\prime}\bar{Z})^{-1}\bar{Z}^{\prime}\epsilon}{\epsilon^{\prime}\epsilon/n}\sim\mathcal{X}^{2}, | 
 | 
 (139) | 
 

 where ϵ \epsilon are the residuals from the two-stage IV regression, and Z ¯ \bar{Z} is another instrumental variable (Over Identification) not involved in the regression of causal effects. Asymptotically, the test statistic p p follows a chi square distribution and the degrees of freedom equal to the number of IVs beyond the need for just-identification [ 98 ] .
Besides, [ 139 ] proposed a similar over-identification tests with F-distribution. [ 140 ] developed several variants for homoscedastic disturbances. Considering heteroscedastic-consistent, [ 96 , 99 ] designed a test statistic for GMM-IV models.

 
 
 

### VI-C IV Synthesis 

 
 Strong and valid IVs are hardly satisfied in practice.
Fortunately, with the advent of machine learning, researchers have found some data-driven algorithms to automatically synthesize strong IV from additional data information under some assumptions.
Practitioners combine more commonly available IV candidates—which are not necessarily strong, or even valid, IVs—into a single “summary” that is plugged into causal effect estimators in place of an IV [ 27 ] .

 
 

#### VI-C 1 Allele Scores

 
 In Mendelian randomization (MR) [ 141 ] , a growing number of works have been proposed to synthesize a summary IV by combining widely availabel IV candidates.
 [ 142 ] shows that summary IV can be reproduced using summarized data on genetic assocaitions with the treatment and the outcome, and a representative approach that combines the IV candidates into a summary variable is unweighted/weighted allele scores [ 25 , 26 , 143 ] (UAS/WAS).
UAS/WAS synthesize a summary variable of genetic contribution towards elevating the risk factor, which serve as reliable IVs to infer causal effect among clinical variables, only if genetic variants associated with a risk factor are actually all independent valid IVs [ 144 , 25 ] .

 
 
 UAS . 
 In Mendelian randomization (MR), we can use genetic variants to as IV candidates for IV synthesis. We assume K K genetic variants 𝐆 = { G 1 , G 2 , ⋯ , G K } \mathbf{G}=\{G_{1},G_{2},\cdots,G_{K}\} are actually independent weak IVs, and use them as IV candidates. Then we can obtain UAS:

 

 
 | 
 U ​ A ​ S I ​ V = 1 K ​ ∑ j = 1 K G j , \displaystyle UAS_{IV}=\frac{1}{K}\sum_{j=1}^{K}G_{j}, | 
 | 
 (140) | 
 

 where K K denotes the number of IV candidates, and G j G_{j} denotes the j j -th IV candidate.
Factually, UAS takes the average of IV candidates.

 
 
 WAS . 
 In addition to an unweighted standard allele score where each risk-increasing allele contributed the same value to the allele score, WAS weights each candidate based on the associations with the treatment:

 

 
 | 
 W ​ A ​ S I ​ V = 1 K ​ ∑ j = 1 K W j ​ G j , \displaystyle WAS_{IV}=\frac{1}{K}\sum_{j=1}^{K}W_{j}G_{j}, | 
 | 
 (141) | 
 

 where W j W_{j} denotes the weights that are the same as the coefficients from the treatment regression stage in the 2SLS analysis. In addition, some other weight estimation methods for calculating relevance and importance can be used as an alternative.

 
 
 Ivy . 
 Allele scores require strong assumptions, i.e., all IV candidates are weak IVs for estimation.
To relax these assumptions, [ 27 ] require more than half of the variables in the IV candidates are valid, and then
propose a generalized allele scores to combine valid IV candidates and invalid candidates in a robust manner, with the following steps: (1) Identify Valid IV Candidates and their Dependencies; (2) Estimate Parameters of the Candidate Model; and (3) Synthesize IV and Estimate Causal Effect.

 
 
 

#### VI-C 2 Weak Candidates

 
 Most of Allele Scores follow the assumption that IV candidates are actually all independent weak IVs, which is actually difficult to meet. In this subsection, we review some more weaker assumptions for IV Synthesis.

 
 
 ModeIV . 
 [ 28 ] no longer requires more than half the number of valid instrumental variables in the candidate set, but proposes that each estimate in the tightest cluster of estimation points from each IV candidate is approximately causal effects and these IV candidates are valid.
ModeIV [ 28 ] will iterate over all the elements in the set of instrumental variable candidates 𝐆 = { G 1 , G 2 , ⋅ , G K } \mathbf{G}=\{G_{1},G_{2},\cdot,G_{K}\} and plug G j G_{j} into the instrumental variable regression method to estimate the causal effects τ G j \tau_{G_{j}} . Then, the outcomes { τ G j } j = 1 K \{\tau_{G_{j}}\}_{j=1}^{K} from the valid instrumental variables must all converge to the same value, and IV candidates in the tightest cluster of estimation points just are valid IVs.

 
 
 AutoIV . 
 Furthermore, AutoIV [ 29 ] generate IV representations based on independence conditions and mutual information, with the assumption that all variables in the IV candidates 𝐆 \mathbf{G} are independent of the unmeasured confoudners 𝐔 \mathbf{U} , i.e., 𝐆 ⟂ 𝐔 \mathbf{G}\perp\mathbf{U} .
Given the obaservational data D = { 𝐗 , 𝐆 , T , Y } D=\{\mathbf{X},\mathbf{G},T,Y\} , AutoIV [ 29 ] learn a disentangled representation 𝐙 = ϕ ⁡ ( 𝐆 ) \mathbf{Z}=\phi(\mathbf{G}) based on independence conditions:

 

 
 | 
 | 
 | 
 ϕ ^ = arg ⁡ min ϕ ⁡ ( T − f ⁡ ( ϕ ⁡ ( 𝐆 ) , 𝐗 ) ) 2 , \displaystyle\hat{\phi}=\arg\min_{\phi}(T-f(\phi(\mathbf{G}),\mathbf{X}))^{2}, | 
 | 
 (142) | 

 
 | 
 | 
 s.t. | 
 ϕ ⁡ ( 𝐆 ) ⟂ 𝐗 , \displaystyle\phi(\mathbf{G})\perp\mathbf{X}, | 
 | 

 
 | 
 | 
 | 
 ϕ ⁡ ( 𝐆 ) ⟂ Y | T , 𝐗 , \displaystyle\phi(\mathbf{G})\perp Y\mid T,\mathbf{X}, | 
 | 
 

 where f ⁡ ( ⋅ ) f(\cdot) denotes a regression network of ϕ ⁡ ( 𝐆 ) , 𝐗 \phi(\mathbf{G}),\mathbf{X} to predict treatment variables. According to the independence conditions, AutoIV [ 29 ] obtain valid IVs that does not have a direct effect on the outcome variable, only indirectly through the treatment variable.

 
 
 To learn relevance and exclusion, AutoIV [ 29 ] construct a mutual information estimation network to optimize the network. Take two any random variables X X and Y Y as an example, the log-likelihood loss function of variational approximation Q θ X ​ Y ​ ( Y | ϕ X ​ ( X ) ) Q_{\theta_{XY}}(Y|\phi_{X}(X)) with n n samples is given as:

 

 
 | 
 ℒ X ​ Y L ​ L ​ D = − 1 n ∑ i = 1 n log Q θ X ​ Y ( y i | ϕ X ( x i ) ) . \mathcal{L}_{XY}^{LLD}=-\frac{1}{n}\sum_{i=1}^{n}{\log{Q_{\theta_{XY}}(y_{i}|\phi_{X}({x}_{i}))}}. | 
 | 
 (143) | 
 

 
 
 They minimize Eq. ( 143 ) to get optimal variational approximation Q θ ^ X ​ Y ​ ( Y | ϕ X ​ ( X ) ) Q_{\hat{\theta}_{XY}}(Y|\phi_{X}(X)) with parameters θ ^ X ​ Y \hat{\theta}_{XY} .
To increase the relevance between the IV representations and the treatment, they maximize the mutual information between them:

 
 
 
 | 
 ℒ X ​ Y M ​ I = 1 n 2 ​ ∑ i = 1 n ∑ j = 1 n ( log ⁡ Q θ X ​ Y ​ ( y i | ϕ X ​ ( x i ) ) − CLOSE \displaystyle\mathcal{L}_{XY}^{MI}=\frac{1}{n^{2}}\sum_{i=1}^{n}\sum_{j=1}^{n}(\log{Q_{\theta_{XY}}(y_{i}|\phi_{X}(x_{i}))}- | 
 | 
 (144) | 

 
 | 
 OPEN log ⁡ Q θ X ​ Y ​ ( y j | ϕ X ​ ( x i ) ) ) , \displaystyle\log{Q_{\theta_{XY}}(y_{j}|\phi_{X}(x_{i}))}), | 
 | 
 

 
 
 Besides, they also model the conditional mutual information ℒ X ​ Y | V M ​ I \mathcal{L}_{XY\mid V}^{MI} conditional on random variable Z Z as:

 
 
 
 | 
 ℒ X ​ Y | V M ​ I = 1 n 2 ​ ∑ i = 1 n ∑ j = 1 n ( ω i ​ j ​ ( log ⁡ Q θ X ​ Y ​ ( y i | ϕ X ​ ( x i ) ) − CLOSE CLOSE \displaystyle\mathcal{L}_{XY\mid V}^{MI}=\frac{1}{n^{2}}\sum_{i=1}^{n}\sum_{j=1}^{n}(\omega_{ij}(\log{Q_{\theta_{XY}}(y_{i}|\phi_{X}(x_{i}))}- | 
 | 
 (145) | 

 
 | 
 OPEN OPEN log ⁡ Q θ X ​ Y ​ ( y j | ϕ X ​ ( x i ) ) ) ) , \displaystyle\log{Q_{\theta_{XY}}(y_{j}|\phi_{X}(x_{i}))})), | 
 | 
 

 where, ω i ​ j = softmax ⁡ ( e − ‖ 𝒙 i − 𝒙 j ‖ 2 2 ​ σ 2 ) \omega_{ij}={\rm{softmax}}(e^{-\frac{{\|\boldsymbol{x}_{i}-\boldsymbol{x}_{j}\|}^{2}}{2\sigma^{2}}}) is the conditional weight of each pair of positive and negative samples.

 
 
 Based on the ℒ X ​ Y L ​ L ​ D \mathcal{L}_{XY}^{LLD} and ℒ X ​ Y | V M ​ I \mathcal{L}_{XY\mid V}^{MI} operators, AutoIV [ 29 ] (1) maximize ℒ 𝐆 ​ T M ​ I \mathcal{L}_{\mathbf{G}T}^{MI} to optimize the IV representations ϕ ⁡ ( 𝐆 ) \phi(\mathbf{G}) for relevance condition; (2) minimize ℒ 𝐆 ​ Y | T M ​ I \mathcal{L}_{\mathbf{G}Y\mid T}^{MI} to optimize the IV representations ϕ ⁡ ( 𝐆 ) \phi(\mathbf{G}) for exclusion condition; and (3) minimize ℒ 𝐆𝐗 M ​ I \mathcal{L}_{\mathbf{G}\mathbf{X}}^{MI} to optimize the IV representations ϕ ⁡ ( 𝐆 ) \phi(\mathbf{G}) for observed confounders independence condition.

 
 
 

#### VI-C 3 Without Any Candidates

 
 Limitation .
Although the above IV generation methods no longer require manually selected pre-defined IVs selected, they all require a high-quality IV candidates’ set with at least half valid IVs or unconfounded IV assumption, which is unrealistic in practice due to cost issues and lack of expert knowledge. These methods still cannot get rid of the dependence on predefined candidate sets. Therefore, it is highly demanded to model IVs and implement a data-driven approach to automatically obtain valid IVs directly from the observed variables { 𝐗 , T , Y } \{\mathbf{X},T,Y\} .

 
 
 In 2021, the idea of using clustering methods to generate instrumental variables started to present, such as CluIV [ 145 ] and GIV [ 146 ] .
Under a more practical setting without any candidates, GIV [ 146 ] proposes a novel algorithm (Meta-EM) to model latent GIV and implement a data-driven approach to automatically reconstruct valid Group IVs directly from the observed variables, beyond hand-made IV candidates.

 
 
 Fig. 13 : Overview of Meta-EM Architecture. 
 
 
 GIV . 
 With the advent of the big data era, a variety of observation databases collected from different sources have been established, which may contain the same treatment effect mechanism (from treatment to outcome) but different treatment assignment mechanisms (from covariates to treatment). Here, the omitted source label can serve as a latent multi-valued IV, which only affects the outcome through its strong association with offer decisions.

 
 
 Therefore, as shown in Fig. 13 , Wu et al. [ 146 ] propose a non-linear Meta-EM to (1) map the raw data into a representation space to construct Linear Mixed Models for the assigned treatment variable; (2) estimate the distribution differences and model the GIV for the different treatment assignment mechanisms; and (3) adopt an alternating training strategy to iteratively optimize the representations and the joint distribution to model GIV for IV regression. Empirical results demonstrate the advantages of our Meta-EM compared with state-of-the-art methods.

 
 
 
 
 

## VII Available Datasets and Codes/Packages 

 

### VII-A Datasets 

 
 In real-world applications, it’s thorny to find a strictly valid instrumental variable from observational data, due to the untestable exclusion and unconfounded conditions. In short, the predefined IVs and IV candidates selected by human effort might be invalid IVs that do not strictly satisfy the conditions of the valid IVs, without enough prior knowledge for valid IVs. Besides, in observational dataset, the
ground truth dose-response function (ATE, ATT, CATE or ITE) is not available, due to the lack of the counterfactual outcome. Hence, the datasets used in the IV-based works are often (semi-)synthetic datasets, such as Demand [ 17 ] and Toy Datasets [ 82 , 63 ] . Some datasets combine the prior specific knowledge and the observational control dataset together to create the datasets. We detail the available benchmark datasets, as follows:

 
 
 Low-dimensional Toy [ 82 , 63 ] .
In low-dimensional cases, [ 82 ] generated data via the following process:

 

 
 | 
 Y = g ⁡ ( T ) + U + δ , T = Z + U + γ . \displaystyle Y=g(T)+U+\delta,T=Z+U+\gamma. | 
 | 
 (146) | 
 
 
 | 
 Z ∼ Uniform ( − 3 , 3 ) , U ∼ 𝒩 ( 0 , 1 ) , δ , γ ∼ 𝒩 ( 0 , 0.1 ) . \displaystyle Z\sim\text{Uniform}(-3,3),U\sim\mathcal{N}(0,1),\delta,\gamma\sim\mathcal{N}(0,0.1). | 
 | 
 

 Similarity, [ 63 ] consider the following data generating processes:

 

 
 | 
 Y = g ⁡ ( T ) + U + δ , T = γ ​ Z + ( 1 − γ ) ​ U + γ . \displaystyle Y=g(T)+U+\delta,T=\gamma Z+(1-\gamma)U+\gamma. | 
 | 
 (147) | 
 
 
 | 
 Z ∼ 𝒩 ( 0 , 2 ) , U ∼ 𝒩 ( 0 , 2 ) , δ , γ ∼ 𝒩 ( 0 , 0.1 ) . \displaystyle Z\sim\mathcal{N}(0,2),U\sim\mathcal{N}(0,2),\delta,\gamma\sim\mathcal{N}(0,0.1). | 
 | 
 

 Keeping the data generating process fixed, [ 82 , 63 ] design various true response function g g between the following cases:

 

 
 | 
 sin: | 
 g ⁡ ( T ) = s ​ i ​ n ​ ( x ) , \displaystyle g(T)=sin(x), | 
 step:  g ( T ) = 0 , \displaystyle\textbf{step: }g(T)=0, | 
 | 
 
 
 | 
 abs: | 
 g ⁡ ( T ) = | x | , \displaystyle g(T)=|x|, | 
 linear:  g ( T ) = x . \displaystyle\textbf{linear: }g(T)=x. | 
 | 
 

 
 
 MNIST [ 82 ] .
Similar to [ 17 ] , in high-dimensional cases, [ 82 ] use same data generating process introduced in Low-dimensional Toy, based on the MNIST dataset [ 147 ] , but replace T T and Z Z with MNIST images:

 

 
 | 
 T := RandomImage ​ ( π ⁡ ( T ) ) , Z := RandomImage ​ ( π ⁡ ( Z ) ) . \displaystyle T:=\text{RandomImage}(\pi(T)),Z:=\text{RandomImage}(\pi(Z)). | 
 | 
 

 where π ⁡ ( t ) = round ​ ( min ​ ( max ​ ( 1.5 ​ t + 5 , 0 ) , 9 ) ) \pi(t)=\text{round}(\text{min}(\text{max}(1.5t+5,0),9)) is a transformation function that maps input t t to an integer range from 0 to 9, and the RandomImage( d d ) is a function that samples a image from the digit label d d . The images are 28 × 28 = 784 28\times 28=784 -dimensional digit matrices.

 
 
 Demand [ 17 ] . The demand simulation design is from [ 17 ] , which describes an airline scenario. In this simulation, the airline wants to estimate the effect of prices T T (i.e., treatment) on passenger ticket sales Y Y (i.e., outcome). We assume that the fuel price Z Z , the customer types X 1 X_{1} , the time of year X 2 X_{2} , and the conferences U U are the pre-treatment variables V V , where the instrumental varialble is Z Z , the observable confounders are X = { X 1 , X 2 } X=\{X_{1},X_{2}\} and the unmeasured confounder is U U . The simulation data is generated by:

 

 
 | 
 T \displaystyle T | 
 = \displaystyle= | 
 25 + ( Z + 3 ) ​ ψ ​ ( X 2 ) + U , \displaystyle 25+(Z+3)\psi(X_{2})+U, | 
 | 
 (148) | 
 
 
 | 
 Y \displaystyle Y | 
 = \displaystyle= | 
 100 + ( 10 + T ) ​ X 1 ​ ψ ​ ( X 2 ) − 2 ​ T + ϵ , \displaystyle 100+(10+T)X_{1}\psi(X_{2})-2T+\epsilon, | 
 | 
 (149) | 
 
 
 | 
 ψ ⁡ ( X 2 ) \displaystyle\psi(X_{2}) | 
 = \displaystyle= | 
 2 ​ ( 1 600 ​ ( X 2 − 5 ) 4 + exp ​ [ − 4 ​ ( X 2 − 5 ) 2 ] + X 2 10 − 2 ) , \displaystyle\scalebox{0.95}{$2\left(\frac{1}{600}(X_{2}-5)^{4}+\text{exp}[-4(X_{2}-5)^{2}]+\frac{X_{2}}{10}-2\right)$}, | 
 | 
 
 
 | 
 X 1 \displaystyle X_{1} | 
 ∈ \displaystyle\in | 
 { 1 , ⋯ , 7 } , X 2 ∼ unif ​ ( 0 , 10 ) , \displaystyle\{1,\cdots,7\},\quad X_{2}\sim\text{unif}(0,10), | 
 | 
 
 
 | 
 Z , U \displaystyle Z,U | 
 ∼ \displaystyle\sim | 
 𝒩 ⁡ ( 0 , 1 ) , ϵ ∼ 𝒩 ⁡ ( ρ ​ U , 1 − ρ 2 ) . \displaystyle\mathcal{N}(0,1),\qquad\epsilon\sim\mathcal{N}(\rho U,1-\rho^{2}). | 
 | 
 

 where, the simulation generates the latent errors ϵ \epsilon with a parameter ρ \rho that is used to smoothly vary the unmeasured confounding bias in causal model.

 
 
 The target dose-response function, i.e., counterfactual function is g ⁡ ( T , X ) = ( 10 + T ) ​ X 1 ​ ψ ​ ( X 2 ) − 2 ​ T g(T,X)=(10+T)X_{1}\psi(X_{2})-2T .

 
 
 IHDP 8 8 
 8 
 
 
 
 http://www.fredjo.com [ 36 , 16 ] .
The Infant Health and Development Program (IHDP), from a Randomized Controlled Trial (RCT), assesses whether the future cognitive of of premature infants is affected by specialist home visits. To reduce the randomness and create a observational data, [ 148 ] removed a non-random subset of the treated group to induce selection bias. The dataset comprises 747 units (139 treated, 608 control) with 25 pre-treatment variables related to the children and their mothers. The treatment is the specialist home visits and the outcome is the cognitive test scores in the future. To develop instrument variables, [ 146 ] generate 2-dimension random variables for each unit. Then, [ 146 ] select a subset of pre-treatment variables as the confounders unobserved confounders U U . With known treated and control potential outcome (accessible in IHDP), [ 146 ] designs the treatment assignment policy as:

 

 
 | 
 P ⁡ ( T ∣ Z , X ) = 1 OPEN 1 + exp ⁡ ( − ( ∑ i = 1 2 Z i + ∑ i = 1 m X X i ) + ∑ i = 1 m U U i ) ) , \displaystyle\scalebox{0.9}{$P(T\mid Z,X)=\frac{1}{1+\exp{\left(-(\sum_{i=1}^{2}Z_{i}+\sum_{i=1}^{m_{X}}X_{i})+\sum_{i=1}^{m_{U}}U_{i})\right)}}$}, | 
 | 
 (150) | 
 
 
 | 
 T ∼ B ​ e ​ r ​ n ​ o ​ u ​ l ​ l ​ i ​ ( P ⁡ ( T ∣ Z , X ) ) , Z 1 , Z 2 ∼ 𝒩 ⁡ ( 0 , 1 ) \displaystyle T\sim Bernoulli(P(T\mid Z,X)),Z_{1},Z_{2}\sim\mathcal{N}(0,1) | 
 | 
 (151) | 
 

 where m X {m_{X}} and m U {m_{U}} are the dimensions of X X and U U selected from the IHDP.

 
 
 PISA [ 149 ] .
The PISA survey aims to evaluate the students’ ability to apply their knowledge and skills to real-life situations [ 149 ] , covering three main domains: reading (131 items), mathematics (35 items), and science (53 items).
 [ 150 , 151 ] selected 4951 participants in March 2009, 4041 participants in October 2009 and 3989 participants in April 2010 and there are 3472 students participated in all three rounds. The distance to school was expressed in the number of minutes is an instrument.
Gender and type of school (General comprehensive, Vocational with comprehensive program, and Basic vocational school) are used as covariates.

 
 
 ALSPAC 9 9 
 9 
 
 
 
 http://www.alspac.bris.ac.uk [ 152 , 153 ] .
The Avon Longitudinal Study of Parents and Children (ALSPAC) is a longitudinal, population-based birth cohort study from 14541 pregnant women resident in Avon, UK, with expected dates of delivery range from April 1991 to December 1992 [ 154 ] . Similar to [ 152 ] , through selection, [ 153 ] used four adiposity-associated genetic variants as IVs for estimating the effect of fat mass on kid’s bone density, based on 5509 birth cohorts.

 
 
 MR-base 10 10 
 10 
 
 
 
 https://www.mrbase.org/ [ 155 ] .
 [ 155 ] developed a MR-Base platform that integrates a curated database of complete GWAS results, which used genetic variants as instrumental variables. The database comprises 11 billion single nucleotide polymorphism-trait associations from 1673 GWAS and is under updated.

 
 
 TABLE I : Available Codes of Methods for Instrumental Variables and Causal Inference. 
 
 
 
 
 IV-based Methods | 

 
 Method | 
 Language | 
 Link | 

 
 DeepIV | 
 python | 
 https://github.com/jhartford/DeepIV | 

 
 KernelIV | 
 matlab | 
 https://github.com/r4hu1-5in9h/KIV | 

 
 DualIV | 
 matlab | 
 https://github.com/krikamol/DualIV-NeurIPS2020 | 

 
 DFIV | 
 python | 
 https://github.com/liyuan9988/DeepFeatureIV | 

 
 DeepGMM | 
 python | 
 https://github.com/CausalML/DeepGMM | 

 
 AGMM | 
 python | 
 https://github.com/microsoft/AdversarialGMM | 

 
 CBIV | 
 python | 
 https://github.com/anpwu/CB-IV | 

 
 AutoIV | 
 python | 
 https://github.com/junkunyuan/AutoIV | 

 
 econML | 
 python | 
 https://github.com/microsoft/EconML | 

 
 CausalDCD | 
 python | 
 https://github.com/anpwu/Awesome-Instrumental-Variable | 

 

 
 
 
 

### VII-B Codes/Packages 

 
 In this part, we summarize the available codes for instrumental variables and causal inference, see Table I . Besides, we merge these codes into a tool-box CausalDCD .

 
 
 
 

## VIII Applications 

 
 In practical, unmeasured confounder is a common setting. Therefore, in the presence of unmeasured confounders, IV regression algorithms have a variety of applications in real-world scenarios.

 
 

### VIII-A Mendelian Randomization 

 
 According to the Mendel’s First and Second Laws of Inheritance, when applied to independent heritable units, genotype is independent of unmeasured confounders.
Therefore, Mendelian randomization (MR) analysis (first used by [ 156 ] ), using genetic variants as instrumental variables to estimate causal effects in the presence of unmeasured confounders [ 157 , 158 , 159 ] , is receiving increasing attention from economists, statisticians, epidemiologists and social scientists are focus [ 160 , 161 , 153 ] .
The growing availability in genome-wide association studies (GWAS) facilitated discovery of genetic variants, that only affects the outcomes through its strong association with treatment factors of interest [ 153 , 162 ] .

 
 
 By comparing outcomes in patients with and without human leukocyte antigen (HLA)-compatible siblings, [ 156 ] first proposed ’Mendelian randomization’ method to explore the effect of allogenic sibling bone marrow transplantation on the treatment of acute myeloid leukaemia (AML).
Mendelian randomization provides one method for assessing the causal nature of some treatment exposures [ 158 ] .
 [ 152 ] used two independent genetic markers (FTO and MC4R genes) of obesity as IVs and found a positive effect of fat mass on bone mineral density (BMD), i.e., higher fat mass caused increased accrual of bone mass in childhood.
 [ 153 ] used multiple genetic variants as instrumental variables for increasing statistical precision of IV estimates and for testing underlying IV assumptions.

 
 
 Use of Mendelian randomisation is growing rapidly [ 163 , 153 ] . Recently, MR has been used successfully across a wide range of domains, i.e., drug target validation, drug target repurposing, side effect identification, and interpretation of high-dimensional omics studies [ 164 , 165 ] . [ 164 ] reviewed recent developments in Mendelian randomization Studies and detailed the extensions to the basic MR design:
including two-sample Mendelian randomization [ 141 , 166 , 167 ] ,
bidirectional Mendelian randomization [ 168 ] ,
two-step Mendelian randomization [ 169 ] ,
multivariable Mendelian randomization [ 170 , 171 ] 
and factorial Mendelian randomization [ 172 ] .
In all, MR is a flexible and robust statistical method, which uses genetic variants as IVs to identify the causal relationships from observational studies.

 
 
 

### VIII-B Sociology and Social Sciences 

 
 In sociology and social sciences, the purpose of causal inference is to examine the association between social network and behaviors, also known as peer effects, social contagion or induction [ 173 , 174 , 175 ] .
The peer effect means that the behavior, traits, or characteristics of an individual’s peers (those he is connected to or alters) would affect his behavior [ 175 ] .
Due to contextual confounding, peer selection, simultaneity bias and measurement error, [ 176 ] points that it is very difficult to estimate the peer effects from observational data but instrumental variables (IVs) can help to address these problems.

 
 
 Taking the city-level characteristics serve as instruments, [ 177 ] study the effect of the neighborhood dropout rate on the individual’s chance of finishing high school. To explore whether moving to a lower dropout rate would lower ones’ chance of dropping out, [ 174 ] used characteristics of the local labor market (or city) as instruments and the results suggested that neighborhood conditions do influence an individual’s likelihood of finishing high school.

 
 
 Besides, researchers and data scientists have an increasing interest on the social network services in the Facebook, Twitter, Wechat and etc [ 178 ] , which are collectively called ’social media’. Adopting an instrumental variables approach, [ 178 ] explored the effect of social network services on social capital. [ 178 ] suggested that high intensity users are higher in network social capital than non-users of social network services.

 
 
 

### VIII-C Reinforcement Learning 

 
 In reinforcement learning (RL), an agent would take actions in an environment in order to maximize the cumulative reward [ 179 , 180 ] . Many concepts of reinforcement learning can be found in causal inference: the treatment is the action taken by agents, the environment can be viewed as the confounder and the cumulative reward is the outcome in causal inference [ 181 , 182 ] .

 
 
 For reinforcement learners, the environment information is usually accessible, due to the Markov property [ 183 , 184 ] , which satisfies the unconfoundedness assumption [ 185 ] .
To obtain an unbiased reward estimation, importance sampling weighting and doubly robust policy evaluation [ 186 , 187 ] are common methods adopted in RL. Under unconfoundedness assumption, there are a substantial number of variants can estimate the state-action value (Q-function) [ 188 , 189 , 190 , 191 ] .

 
 
 To relax the unconfoundedness assumption, [ 192 , 193 , 194 , 195 ] introduced instrumental variables to optimize the policy for maximizing the reward.
In the context of offline policy evaluation (OPE), [ 193 ] proposed improved Q-function estimators with different IV techniques and obtain competitive new techniques in recovering previously proposed OPE methods.
Using IVs, [ 194 ] derived a conditional moment restriction (CMR) and propose a IV-aided Value Iteration (IVVI) algorithm based on a primal-dual reformulation of CMR. In addition, [ 195 ] developed a new techniques to apply IV Regression to correct for the bias in RL algorithm in the presence of time-dependency noise.

 
 
 

### VIII-D Recommendation System 

 
 Another application, highly correlated with the treatment effect estimation, is recommendation system [ 196 , 189 , 11 , 197 , 32 ] . Exposing the user to an item can be viewed as a specific treatment and the user’s behaviour (click or activity) is the corresponding outcome.
To elimate the bias form the unmeasured confounders and the self-selection of the users, [ 198 ] proposed an instrumental variable estimate of the click-through rate, where the shock is the instrument, the treatment is exposure to the focal product, and the outcome is click-through to the recommended product.
Jointly considering users’ behaviors in search scenarios and recommendation scenarios, [ 199 ] embedded users’ search
behaviors as instrumental variables (IVs) and implemented a two-stage regression for an unbiased estimate of causal effect.

 
 
 

### VIII-E Computer Vision 

 
 Computer Vision is a typical field of artificial intelligence (AI), suffering from unstable learning and lacking of generalization ability [ 5 ] . To achieve a proactive defense against adversarial examples, [ 200 ] proposed to use the instrumental variable that achieves causal intervention.
Using “retinotopic sampling” as IV [ 201 ] ,
Causal intervention by instrumental Variable (CiiV) [ 200 ] algorithm
implements a spatial data augmentation using different retinotopic sampling masks and learns features linearly responding to spatial interpolations.
In Domian Adaptation, [ 202 ] claimed that the input features of one domain are valid instrumental variables for other domains. Inspired by this finding, we design a simple yet effective framework to learn the Domain-invariant Relationship with Instrumental VariablE (DRIVE) via a two-stage IV method.

 
 
 
 

## IX Conclusion 

 

### IX-A Future Direction 

 
 Instrumental Variable has been an attractive research topic for a long time as it provides an effective way to uncover causal relationships in real-world problems. In this section, we point several lines for further research.

 
 

#### IX-A 1 How to find a valid IV?

 
 The exclusion restriction is the most critical and typically most controversial assumption underlying instrumental variables methods and we don’t have any means to test it. In traditional literature, researchers implement randomized controlled trials (RCTs) to sample a random variable as IV to intervene the received treatments, which are too costly to be universally available.
Therefore, it’s highly demanding to develop a data-driven approach to automatically obtain valid IVs.
Fortunately, machine learning and Bayesian learning provide tools for modeling latent variables. Based on conditional independence test, it is likely to disentangle instrumental variables from these hidden variables.
In addition, causal discovery algorithms are also a promising direction to help us automatically find instrumental variables from observed covariates.

 
 
 

#### IX-A 2 How to relax the IV assumptions? 

 
 An instrument meets the following three assumptions: relevance assumption, exclusion assumption and unconfoundedness assumption. For exclusion assumption, we can use some mediators to block out the direct effect of IV on the outcomes to relax it. For confounded IV, we can also try to recover the unmeasured confounders affecting IV based on conditional independence constraints, and adjust it.

 
 
 

#### IX-A 3 How to combine IV Regression with Confounder Control? 

 
 In traditional instrumental variable regression methods, researchers always ignore the bias caused by the observed confounding variables. Even by CFN, the investigators did not control for confounding of the recovered residuals. Considering confounder balance, a more robust instrumental variable regression method is a promising direction.

 
 
 

#### IX-A 4 How to reduce unmeasured confounding without IV? 

 
 However, in real life, instrumental variables may not always exist, which is the norm. In the past, we have always considered observational datasets or randomized controlled experiments separately. But in fact, even if randomized controlled experiments are expensive, we can still conduct small-scale randomized controlled experiments. Considering small intervention data and a large amount of observational data, i.e., data fusion, it is possible to establish causality
without confounding bias.

 
 
 
 

### IX-B Conclusion 

 
 In this survey, we provide a comprehensive review of the connection between the instrumental variable methods and machine learning models.
Combined with machine learning, we mainly introduce two typical types of methods to estimate the average treatment effect: two-stage least squares (vanilla 2SLS estimator for linear models and machine learning estimator for non-linear models) and the traditional control function method (linear estimator and non-linear estimator).
As IV-based framework relies on one structural assumption and three restrictions for identification of causal effects,
we also review the traditional identifiability assumptions that apply in various scenarios and how to find or generate a valid IV towards these restrictions.
The available benchmark datasets and open-source codes of those methods are also listed.
Finally, some representative real-world applications of causal inference are introduced, such as advertising,
recommendation, medicine, and reinforcement learning.

 
 
 
 

## References

 
 
 [1] 
 
C. Wu, P. Jiang, C. Ding, F. Feng, and T. Chen, “Intelligent fault diagnosis
of rotating machinery based on one-dimensional convolutional neural
network,” Computers in Industry , vol. 108, pp. 53–61, 2019.

 

 
 [2] 
 
S. Wu, F. Sun, W. Zhang, X. Xie, and B. Cui, “Graph neural networks in
recommender systems: a survey,” ACM Computing Surveys (CSUR) , 2020.

 

 
 [3] 
 
S. Lee and D. Kim, “Deep learning based recommender system using cross
convolutional filters,” Information Sciences , vol. 592, pp. 112–122,
2022.

 

 
 [4] 
 
Y. He, Z. Shen, and P. Cui, “Towards non-iid image classification: A dataset
and baselines,” Pattern Recognition , vol. 110, p. 107383, 2021.

 

 
 [5] 
 
B. Schölkopf, F. Locatello, S. Bauer, N. R. Ke, N. Kalchbrenner, A. Goyal,
and Y. Bengio, “Towards causal representation learning,” arXiv
preprint arXiv:2102.11107 , 2021.

 

 
 [6] 
 
Z. Shen, P. Cui, T. Zhang, and K. Kunag, “Stable learning via sample
reweighting,” in Proceedings of the AAAI Conference on Artificial
Intelligence , vol. 34, no. 04, 2020, pp. 5692–5699.

 

 
 [7] 
 
X. Zhang, P. Cui, R. Xu, L. Zhou, Y. He, and Z. Shen, “Deep stable learning
for out-of-distribution generalization,” in Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2021, pp.
5372–5382.

 

 
 [8] 
 
J. Pearl, Causality . Cambridge
university press, 2009.

 

 
 [9] 
 
P. Cui and S. Athey, “Stable learning establishes some common ground between
causal inference and machine learning,” Nature Machine Intelligence ,
vol. 4, no. 2, pp. 110–115, Feb. 2022. [Online]. Available:
 https://doi.org/10.1038/s42256-022-00445-z 

 

 
 [10] 
 
H. Bang and J. M. Robins, “Doubly robust estimation in missing data and causal
inference models,” Biometrics , vol. 61, no. 4, pp. 962–973, 2005.

 

 
 [11] 
 
X. Wang, R. Zhang, Y. Sun, and J. Qi, “Doubly robust joint learning for
recommendation on data missing not at random,” in International
Conference on Machine Learning . PMLR,
2019, pp. 6638–6647.

 

 
 [12] 
 
S. Athey, G. W. Imbens, and S. Wager, “Approximate residual balancing:
debiased inference of average treatment effects in high dimensions,”
 Journal of the Royal Statistical Society: Series B (Statistical
Methodology) , vol. 80, no. 4, pp. 597–623, 2018.

 

 
 [13] 
 
J. R. Zubizarreta, “Stable weights that balance covariates for estimation with
incomplete outcome data,” Journal of the American Statistical
Association , vol. 110, no. 511, pp. 910–922, 2015.

 

 
 [14] 
 
J. Hainmueller, “Entropy balancing for causal effects: A multivariate
reweighting method to produce balanced samples in observational studies,”
 Political analysis , vol. 20, no. 1, pp. 25–46, 2012.

 

 
 [15] 
 
J. Pearl, “Causal diagrams for empirical research,” Biometrika ,
vol. 82, no. 4, pp. 669–688, 1995.

 

 
 [16] 
 
U. Shalit, F. D. Johansson, and D. Sontag, “Estimating individual treatment
effect: generalization bounds and algorithms,” in International
Conference on Machine Learning . PMLR,
2017, pp. 3076–3085.

 

 
 [17] 
 
J. Hartford, G. Lewis, K. Leyton-Brown, and M. Taddy, “Deep iv: A flexible
approach for counterfactual prediction,” in International Conference
on Machine Learning . PMLR, 2017, pp.
1414–1423.

 

 
 [18] 
 
G. Imbens and J. Wooldridge, “Control function and related methods,”
 What’s new in Econometrics , 2007.

 

 
 [19] 
 
Z. Guo and D. S. Small, “Control function instrumental variable estimation of
nonlinear causal effect models,” The Journal of Machine Learning
Research , vol. 17, no. 1, pp. 3448–3482, 2016.

 

 
 [20] 
 
W. G. Axinn and J. S. Barber, “Mass education and fertility transition,”
 American Sociological Review , pp. 481–505, 2001.

 

 
 [21] 
 
K. A. Bollen and D. J. Bauer, “Automating the selection of model-implied
instrumental variables,” Sociological Methods Research , vol. 32,
no. 4, pp. 425–452, 2004.

 

 
 [22] 
 
K. A. Bollen, “Model implied instrumental variables (miivs): An alternative
orientation to structural equation modeling,” Multivariate behavioral
research , vol. 54, no. 1, pp. 31–46, 2019.

 

 
 [23] 
 
A. Finkelstein, S. Taubman, B. Wright, M. Bernstein, J. Gruber, J. P. Newhouse,
H. Allen, K. Baicker, and O. H. S. Group, “The oregon health insurance
experiment: evidence from the first year,” The Quarterly journal of
economics , vol. 127, no. 3, pp. 1057–1106, 2012.

 

 
 [24] 
 
J. D. Angrist, “Lifetime earnings and the vietnam era draft lottery: evidence
from social security administrative records,” The american economic
review , pp. 313–336, 1990.

 

 
 [25] 
 
S. Burgess, D. S. Small, and S. G. Thompson, “A review of instrumental
variable estimators for mendelian randomization,” Statistical methods
in medical research , vol. 26, no. 5, pp. 2333–2355, 2017.

 

 
 [26] 
 
S. Burgess and S. G. Thompson, “Use of allele scores as instrumental variables
for mendelian randomization,” International journal of epidemiology ,
vol. 42, no. 4, pp. 1134–1144, 2013.

 

 
 [27] 
 
Z. Kuang, F. Sala, N. Sohoni, S. Wu, A. Córdova-Palomera, J. Dunnmon,
J. Priest, and C. Ré, “Ivy: Instrumental variable synthesis for causal
inference,” in International Conference on Artificial Intelligence and
Statistics . PMLR, 2020, pp. 398–410.

 

 
 [28] 
 
J. S. Hartford, V. Veitch, D. Sridhar, and K. Leyton-Brown, “Valid causal
inference with (some) invalid instruments,” in International
Conference on Machine Learning . PMLR,
2021, pp. 4096–4106.

 

 
 [29] 
 
J. Yuan, A. Wu, K. Kuang, B. Li, R. Wu, F. Wu, and L. Lin, “Auto iv:
Counterfactual prediction via automatic instrumental variable
decomposition,” ACM Transactions on Knowledge Discovery from Data
(TKDD) , vol. 16, no. 4, pp. 1–20, 2022.

 

 
 [30] 
 
K. Tang, M. Tao, J. Qi, Z. Liu, and H. Zhang, “Invariant feature learning for
generalized long-tailed classification,” arXiv preprint
arXiv:2207.09504 , 2022.

 

 
 [31] 
 
R. Guo, L. Cheng, J. Li, P. R. Hahn, and H. Liu, “A survey of learning
causality with data: Problems and methods,” ACM Computing Surveys
(CSUR) , vol. 53, no. 4, pp. 1–37, 2020.

 

 
 [32] 
 
L. Yao, Z. Chu, S. Li, Y. Li, J. Gao, and A. Zhang, “A survey on causal
inference,” ACM Transactions on Knowledge Discovery from Data (TKDD) ,
vol. 15, no. 5, pp. 1–46, 2021.

 

 
 [33] 
 
M. Kato, H. Kakehi, K. McAlinn, and S. Yasui, “Learning causal relationships
from conditional moment conditions by importance weighting,” arXiv
preprint arXiv:2108.01312 , 2021.

 

 
 [34] 
 
R. Kohavi and R. Longbotham, “Unexpected results in online controlled
experiments,” ACM SIGKDD Explorations Newsletter , vol. 12, no. 2, pp.
31–35, 2011.

 

 
 [35] 
 
L. Bottou, J. Peters, J. Quiñonero-Candela, D. X. Charles, D. M.
Chickering, E. Portugaly, D. Ray, P. Simard, and E. Snelson, “Counterfactual
reasoning and learning systems: The example of computational advertising.”
 Journal of Machine Learning Research , vol. 14, no. 11, 2013.

 

 
 [36] 
 
F. Johansson, U. Shalit, and D. Sontag, “Learning representations for
counterfactual inference,” in International conference on machine
learning . PMLR, 2016, pp. 3020–3029.

 

 
 [37] 
 
N. Hassanpour and R. Greiner, “Learning disentangled representations for
counterfactual regression,” in International Conference on Learning
Representations , 2020.

 

 
 [38] 
 
P. R. Rosenbaum and D. B. Rubin, “The central role of the propensity score in
observational studies for causal effects,” Biometrika , vol. 70,
no. 1, pp. 41–55, 1983.

 

 
 [39] 
 
S. Li, N. Vlassis, J. Kawale, and Y. Fu, “Matching via dimensionality
reduction for estimation of treatment effects in digital marketing
campaigns.” in IJCAI , 2016, pp. 3768–3774.

 

 
 [40] 
 
M. A. Brookhart, T. St”urmer, R. J. Glynn, J. Rassen, and S. Schneeweiss,
“Confounding control in healthcare database research: challenges and
potential approaches,” Medical care , vol. 48, no. 6 0, p. S114, 2010.

 

 
 [41] 
 
P. G. Wright, Tariff on animal and vegetable oils . Macmillan Company, New York, 1928.

 

 
 [42] 
 
J. D. Angrist, G. W. Imbens, and D. B. Rubin, “Identification of causal
effects using instrumental variables,” Journal of the American
statistical Association , vol. 91, no. 434, pp. 444–455, 1996.

 

 
 [43] 
 
W. K. Newey and J. L. Powell, “Instrumental variable estimation of
nonparametric models,” Econometrica , vol. 71, no. 5, pp. 1565–1578,
2003.

 

 
 [44] 
 
J. H. Stock and F. Trebbi, “Retrospectives: Who invented instrumental variable
regression?” Journal of Economic Perspectives , vol. 17, no. 3, pp.
177–194, 2003.

 

 
 [45] 
 
Ø. Hoveid, “Constructing valid instrumental variables in generalized linear
causal models from directed acyclic graphs,” arXiv preprint
arXiv:2102.08056 , 2021.

 

 
 [46] 
 
T. Haavelmo, “The statistical implications of a system of simultaneous
equations,” Econometrica, Journal of the Econometric Society , pp.
1–12, 1943.

 

 
 [47] 
 
O. Reiersøl, “Identifiability of a linear relation between variables which
are subject to error,” Econometrica: Journal of the Econometric
Society , pp. 375–389, 1950.

 

 
 [48] 
 
J. Pearl et al. , “Models, reasoning and inference,” Cambridge,
UK: CambridgeUniversityPress , vol. 19, 2000.

 

 
 [49] 
 
J. Angrist and G. Imbens, “Identification and estimation of local average
treatment effects,” 1995.

 

 
 [50] 
 
D. B. Rubin, “Estimating causal effects of treatments in randomized and
nonrandomized studies.” Journal of educational Psychology , vol. 66,
no. 5, p. 688, 1974.

 

 
 [51] 
 
——, “Bayesian inference for causal effects: The role of randomization,”
 The Annals of statistics , pp. 34–58, 1978.

 

 
 [52] 
 
——, “Comment: Neyman (1923) and causal inference in experiments and
observational studies,” Statistical Science , vol. 5, no. 4, pp.
472–480, 1990.

 

 
 [53] 
 
G. Imbens, “Instrumental variables: an econometrician’s perspective,”
National Bureau of Economic Research, Tech. Rep., 2014.

 

 
 [54] 
 
A. Wu, K. Kuang, B. Li, and F. Wu, “Instrumental variable regression with
confounder balancing,” in International Conference on Machine
Learning . PMLR, 2022, pp.
24 056–24 075.

 

 
 [55] 
 
J. Heckman, “Varieties of selection bias,” The American Economic
Review , vol. 80, no. 2, pp. 313–318, 1990.

 

 
 [56] 
 
J. Angrist and G. Imbens, “Sources of identifying information in evaluation
models,” 1991.

 

 
 [57] 
 
W. K. Newey, “Nonparametric instrumental variables estimation,”
 American Economic Review , vol. 103, no. 3, pp. 550–56, 2013.

 

 
 [58] 
 
M. A. Hernán and J. M. Robins, “Instrumental variable estimation,”
 Causal Inference: What If , pp. 193–206, 2020.

 

 
 [59] 
 
F. P. Hartwig, L. Wang, G. D. Smith, and N. M. Davies, “Average causal effect
estimation via instrumental variables: the no simultaneous heterogeneity
assumption,” arXiv preprint arXiv:2010.10017 , 2020.

 

 
 [60] 
 
L. E. Mokry, O. Ahmad, V. Forgetta, G. Thanassoulis, and J. B. Richards,
“Mendelian randomisation applied to drug development in cardiovascular
disease: a review,” Journal of medical genetics , vol. 52, no. 2, pp.
71–79, 2015.

 

 
 [61] 
 
R. Singh, M. Sahani, and A. Gretton, “Kernel instrumental variable
regression,” in Proceedings of the 33rd International Conference on
Neural Information Processing Systems , 2019, pp. 4593–4605.

 

 
 [62] 
 
K. Muandet, A. Mehrjou, S. Le Kai, and A. Raj, “Dual instrumental variable
regression,” in NeurIPS 2020 , 2020.

 

 
 [63] 
 
N. Dikkala, G. Lewis, L. Mackey, and V. Syrgkanis, “Minimax estimation of
conditional moment models,” in NeurIPS 2020 , 2020.

 

 
 [64] 
 
R. Blundell and J. L. Powell, “Endogeneity in nonparametric and semiparametric
regression models,” Econometric society monographs , vol. 36, pp.
312–357, 2003.

 

 
 [65] 
 
A. Petrin and K. Train, “A control function approach to endogeneity in
consumer choice models,” Journal of marketing research , vol. 47,
no. 1, pp. 3–13, 2010.

 

 
 [66] 
 
J. M. Wooldridge, “Control function methods in applied econometrics,”
 Journal of Human Resources , vol. 50, no. 2, pp. 420–445, 2015.

 

 
 [67] 
 
A. Puli and R. Ranganath, “General control functions for causal effect
estimation from ivs,” Advances in neural information processing
systems , vol. 33, pp. 8440–8451, 2020.

 

 
 [68] 
 
G. Chamberlain, “Asymptotic efficiency in semi-parametric models with
censoring,” journal of Econometrics , vol. 32, no. 2, pp. 189–218,
1986.

 

 
 [69] 
 
J. J. Heckman and R. Robb Jr, “Alternative methods for evaluating the impact
of interventions: An overview,” Journal of econometrics , vol. 30, no.
1-2, pp. 239–267, 1985.

 

 
 [70] 
 
J. J. Heckman and V. J. Hotz, “Choosing among alternative nonexperimental
methods for estimating the impact of social programs: The case of manpower
training,” Journal of the American statistical Association , vol. 84,
no. 408, pp. 862–874, 1989.

 

 
 [71] 
 
J. Durbin, “Errors in variables,” Revue de l’institut International de
Statistique , pp. 23–32, 1954.

 

 
 [72] 
 
R. J. LaLonde, “Evaluating the econometric evaluations of training programs
with experimental data,” The American economic review , pp. 604–620,
1986.

 

 
 [73] 
 
D. Chetverikov and D. Wilhelm, “Nonparametric instrumental variable estimation
under monotonicity,” Econometrica , vol. 85, no. 4, pp. 1303–1320,
2017.

 

 
 [74] 
 
M. A. Hernán and J. M. Robins, “Instruments for causal inference: an
epidemiologist’s dream?” Epidemiology , pp. 360–372, 2006.

 

 
 [75] 
 
M. A. Brookhart and S. Schneeweiss, “Preference-based instrumental variable
methods for the estimation of treatment effects: assessing validity and
interpreting results,” The international journal of biostatistics ,
vol. 3, no. 1, 2007.

 

 
 [76] 
 
L. Wang and E. Tchetgen Tchetgen, “Bounded, efficient and multiply robust
estimation of average treatment effects using instrumental variables,”
 Journal of the Royal Statistical Society: Series B (Statistical
Methodology) , vol. 80, no. 3, pp. 531–550, 2018.

 

 
 [77] 
 
F. P. Hartwig, L. Wang, G. D. Smith, and N. M. Davies, “Homogeneity in the
instrument-treatment association is not sufficient for the wald estimand to
equal the average causal effect for a binary instrument and a continuous
exposure,” arXiv preprint arXiv:2107.01070 , 2021.

 

 
 [78] 
 
R. Kress, V. Maz’ya, and V. Kozlov, Linear integral equations . Springer, 1989, vol. 82.

 

 
 [79] 
 
X. Chen and T. M. Christensen, “Optimal sup-norm rates and uniform inference
on nonlinear functionals of nonparametric iv regression,” Quantitative
Economics , vol. 9, no. 1, pp. 39–84, 2018.

 

 
 [80] 
 
A. Lin, J. Lu, J. Xuan, F. Zhu, and G. Zhang, “One-stage deep instrumental
variable method for causal inference from observational data,” in 2019
IEEE International Conference on Data Mining (ICDM) . IEEE, 2019, pp. 419–428.

 

 
 [81] 
 
L. Xu, Y. Chen, S. Srinivasan, N. de Freitas, A. Doucet, and A. Gretton,
“Learning deep features in instrumental variable regression,” 2021.

 

 
 [82] 
 
A. Bennett, N. Kallus, and T. Schnabel, “Deep generalized method of moments
for instrumental variable analysis,” Advances in neural information
processing systems , vol. 32, 2019.

 

 
 [83] 
 
A. Wald, “The fitting of straight lines if both variables are subject to
error,” The annals of mathematical statistics , vol. 11, no. 3, pp.
284–300, 1940.

 

 
 [84] 
 
A. R. Gallant, “Identification and consistency in seminonparametric
regression,” Advances in Econometrics , vol. 1, pp. 145–170, 1987.

 

 
 [85] 
 
X. Chen and X. Shen, “Sieve extremum estimates for weakly dependent data,”
 Econometrica , pp. 289–314, 1998.

 

 
 [86] 
 
J. L. Horowitz, “Applied nonparametric instrumental variables estimation,”
 Econometrica , vol. 79, no. 2, pp. 347–394, 2011.

 

 
 [87] 
 
X. Chen and D. Pouzo, “Estimation of nonparametric conditional moment models
with possibly nonsmooth generalized residuals,” Econometrica ,
vol. 80, no. 1, pp. 277–321, 2012.

 

 
 [88] 
 
B. Boots, G. Gordon, and A. Gretton, “Hilbert space embeddings of predictive
state representations,” arXiv preprint arXiv:1309.6819 , 2013.

 

 
 [89] 
 
A. Hefny, C. Downey, and G. J. Gordon, “Supervised learning for dynamical
system learning,” Advances in neural information processing systems ,
vol. 28, 2015.

 

 
 [90] 
 
L. Song, J. Huang, A. Smola, and K. Fukumizu, “Hilbert space embeddings of
conditional distributions with applications to dynamical systems,” in
 Proceedings of the 26th Annual International Conference on Machine
Learning , 2009, pp. 961–968.

 

 
 [91] 
 
B. Dai, N. He, Y. Pan, B. Boots, and L. Song, “Learning from conditional
distributions via dual embeddings,” in Artificial Intelligence and
Statistics . PMLR, 2017, pp.
1458–1467.

 

 
 [92] 
 
A. Shapiro, D. Dentcheva, and A. Ruszczynski, “Lectures on stochastic
programming: Modeling and theory,” 2014.

 

 
 [93] 
 
B. Schölkopf, A. J. Smola, F. Bach et al. , Learning with
kernels: support vector machines, regularization, optimization, and
beyond . MIT press, 2002.

 

 
 [94] 
 
S. Darolles, Y. Fan, J.-P. Florens, and E. Renault, “Nonparametric
instrumental regression,” Econometrica , vol. 79, no. 5, pp.
1541–1565, 2011.

 

 
 [95] 
 
C. F. Baum, M. E. Schaffer, and S. Stillman, “Instrumental variables and gmm:
Estimation and testing,” The Stata Journal , vol. 3, no. 1, pp. 1–31,
2003.

 

 
 [96] 
 
L. P. Hansen, “Large sample properties of generalized method of moments
estimators,” Econometrica: Journal of the econometric society , pp.
1029–1054, 1982.

 

 
 [97] 
 
B. E. Hansen, “Testing for structural change in conditional models,”
 Journal of Econometrics , vol. 97, no. 1, pp. 93–115, 2000.

 

 
 [98] 
 
J. M. Wooldridge, Econometric analysis of cross section and panel
data . MIT press, 2010.

 

 
 [99] 
 
F. Hayashi, Econometrics . Princeton University Press, 2011.

 

 
 [100] 
 
R. Zhang, M. Imaizumi, B. Schölkopf, and K. Muandet, “Maximum moment
restriction for instrumental variable regression,” arXiv preprint
arXiv:2010.07684 , 2020.

 

 
 [101] 
 
M. Arjovsky, S. Chintala, and L. Bottou, “Wasserstein generative adversarial
networks,” in International conference on machine learning . PMLR, 2017, pp. 214–223.

 

 
 [102] 
 
C.-L. Li, W.-C. Chang, Y. Cheng, Y. Yang, and B. Póczos, “Mmd gan: Towards
deeper understanding of moment matching network,” Advances in neural
information processing systems , vol. 30, 2017.

 

 
 [103] 
 
L. P. Hansen, J. Heaton, and A. Yaron, “Finite-sample properties of some
alternative gmm estimators,” Journal of Business Economic
Statistics , vol. 14, no. 3, pp. 262–280, 1996.

 

 
 [104] 
 
A. Bennett and N. Kallus, “The variational method of moments,” arXiv
preprint arXiv:2012.09422 , 2020.

 

 
 [105] 
 
L. Liao, Y.-L. Chen, Z. Yang, B. Dai, M. Kolar, and Z. Wang, “Provably
efficient neural estimation of structural equation models: An adversarial
approach,” Advances in Neural Information Processing Systems ,
vol. 33, pp. 8947–8958, 2020.

 

 
 [106] 
 
V. Chernozhukov, W. Newey, R. Singh, and V. Syrgkanis, “Adversarial estimation
of riesz representers,” arXiv preprint arXiv:2101.00009 , 2020.

 

 
 [107] 
 
P. Rolland, V. Cevher, M. Kleindessner, C. Russell, D. Janzing,
B. Schölkopf, and F. Locatello, “Score matching enables causal discovery
of nonlinear additive noise models,” in Proceedings of the 39th
International Conference on Machine Learning , ser. Proceedings of Machine
Learning Research, K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvari, G. Niu,
and S. Sabato, Eds., vol. 162. PMLR,
17–23 Jul 2022, pp. 18 741–18 753. [Online]. Available:
 https://proceedings.mlr.press/v162/rolland22a.html 

 

 
 [108] 
 
S. Saengkyongam, L. Henckel, N. Pfister, and J. Peters, “Exploiting
independent instruments: Identification and distribution generalization,” in
 Proceedings of the 39th International Conference on Machine Learning ,
ser. Proceedings of Machine Learning Research, K. Chaudhuri, S. Jegelka,
L. Song, C. Szepesvari, G. Niu, and S. Sabato, Eds., vol. 162. PMLR, 17–23 Jul 2022, pp. 18 935–18 958.
[Online]. Available:
 https://proceedings.mlr.press/v162/saengkyongam22a.html 

 

 
 [109] 
 
L. G. Telser, “Iterative estimation of a set of linear regression equations,”
 Journal of the American Statistical Association , vol. 59, no. 307, pp.
845–862, 1964.

 

 
 [110] 
 
A. S. Goldberger, “Selection bias in evaluating treatment effects: Some formal
illustrations,” in Modelling and Evaluating Treatment Effects in
Econometrics . Emerald Group
Publishing Limited, 1972, reprinted 2008.

 

 
 [111] 
 
B. Barnow, G. Cain, and A. Goldberg, “Selection on observables,”
 Evaluation Studies , 1981.

 

 
 [112] 
 
A. C. Cameron and P. K. Trivedi, Microeconometrics: methods and
applications . Cambridge university
press, 2005.

 

 
 [113] 
 
J. A. Hausman, “Specification tests in econometrics,” Econometrica:
Journal of the econometric society , pp. 1251–1271, 1978.

 

 
 [114] 
 
W. H. Greene, Econometric analysis . Pearson Education India, 2003.

 

 
 [115] 
 
J. Heckman and E. Vytlacil, “Instrumental variables methods for the correlated
random coefficient model: Estimating the average rate of return to schooling
when the return is correlated with schooling,” Journal of Human
Resources , pp. 974–987, 1998.

 

 
 [116] 
 
D. Card, “Estimating the return to schooling: Progress on some persistent
econometric problems,” Econometrica , vol. 69, no. 5, pp. 1127–1160,
2001.

 

 
 [117] 
 
R. J. Smith and R. W. Blundell, “An exogeneity test for a simultaneous
equation tobit model with an application to labor supply,”
 Econometrica: journal of the Econometric Society , pp. 679–685, 1986.

 

 
 [118] 
 
D. Rivers and Q. H. Vuong, “Limited information estimators and exogeneity
tests for simultaneous probit models,” Journal of econometrics ,
vol. 39, no. 3, pp. 347–366, 1988.

 

 
 [119] 
 
J. V. Terza, A. Basu, and P. J. Rathouz, “Two-stage residual inclusion
estimation: addressing endogeneity in health econometric modeling,”
 Journal of health economics , vol. 27, no. 3, pp. 531–543, 2008.

 

 
 [120] 
 
J. M. Wooldridge, “Unobserved heterogeneity and estimation of average partial
effects,” Identification and inference for econometric models: Essays
in honor of Thomas Rothenberg , pp. 27–55, 2005.

 

 
 [121] 
 
D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” arXiv
preprint arXiv:1312.6114 , 2013.

 

 
 [122] 
 
C. Louizos, U. Shalit, J. M. Mooij, D. Sontag, R. Zemel, and M. Welling,
“Causal effect inference with deep latent-variable models,” Advances
in neural information processing systems , vol. 30, 2017.

 

 
 [123] 
 
W. Zhang, L. Liu, and J. Li, “Treatment effect estimation with disentangled
latent factors,” arXiv preprint arXiv:2001.10652 , 2020.

 

 
 [124] 
 
P. A. Wu and K. Fukumizu, “$\beta$-intact-VAE: Identifying
and estimating causal effects under limited overlap,” in International
Conference on Learning Representations , 2022. [Online]. Available:
 https://openreview.net/forum?id=q7n2RngwOM 

 

 
 [125] 
 
I. Higgins, L. Matthey, A. Pal, C. Burgess, X. Glorot, M. Botvinick,
S. Mohamed, and A. Lerchner, “beta-vae: Learning basic visual concepts with
a constrained variational framework,” 2016.

 

 
 [126] 
 
K. Kuang, L. Li, Z. Geng, L. Xu, K. Zhang, B. Liao, H. Huang, P. Ding, W. Miao,
and Z. Jiang, “Causal inference,” Engineering , vol. 6, no. 3, pp.
253–263, 2020.

 

 
 [127] 
 
A. Wu, J. Yuan, K. Kuang, B. Li, R. Wu, Q. Zhu, Y. T. Zhuang, and F. Wu,
“Learning decomposed representations for treatment effect estimation,”
 IEEE Transactions on Knowledge and Data Engineering , 2022.

 

 
 [128] 
 
H. Wang, W. Yang, L. Yang, A. Wu, L. Xu, J. Ren, F. Wu, and K. Kuang,
“Estimating individualized causal effect with confounded instruments,” in
 Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery
and Data Mining , 2022, pp. 1857–1867.

 

 
 [129] 
 
A. Nichols et al. , “Weak instruments: An overview and new techniques,”
in Stata 5th North American Meeting Presentation , 2006.

 

 
 [130] 
 
K. A. Bollen, “Instrumental variables in sociology and the social sciences,”
 Annual Review of Sociology , vol. 38, pp. 37–72, 2012.

 

 
 [131] 
 
L. Anselin, Spatial econometrics: methods and models . Springer Science Business Media, 1988, vol. 4.

 

 
 [132] 
 
J. D. Angrist and A. B. Keueger, “Does compulsory school attendance affect
schooling and earnings?” The Quarterly Journal of Economics , vol.
106, no. 4, pp. 979–1014, 1991.

 

 
 [133] 
 
K. A. Bollen, “An alternative two stage least squares (2sls) estimator for
latent variable equations,” Psychometrika , vol. 61, no. 1, pp.
109–121, 1996.

 

 
 [134] 
 
C. Brito and J. Pearl, “A graphical criterion for the identification of causal
effects in linear models,” AAAI/IAAI , vol. 2002, pp. 533–539, 2002.

 

 
 [135] 
 
J. Pearl, “The foundations of causal inference,” Sociological
Methodology , vol. 40, no. 1, pp. 75–149, 2010.

 

 
 [136] 
 
M. R. Rosenzweig and K. I. Wolpin, “Natural” natural experiments” in
economics,” Journal of Economic Literature , vol. 38, no. 4, pp.
827–874, 2000.

 

 
 [137] 
 
B. Hansen, Econometrics , 2022.

 

 
 [138] 
 
J. D. Sargan, “The estimation of economic relationships using instrumental
variables,” Econometrica: Journal of the Econometric Society , pp.
393–415, 1958.

 

 
 [139] 
 
R. L. Basmann, “On finite sample distributions of generalized classical linear
identifiability test statistics,” Journal of the American Statistical
Association , vol. 55, no. 292, pp. 650–659, 1960.

 

 
 [140] 
 
J. B. Kirby and K. A. Bollen, “10. using instrumental variable tests to
evaluate model specification in latent variable structural equation models,”
 Sociological Methodology , vol. 39, no. 1, pp. 327–355, 2009.

 

 
 [141] 
 
S. Burgess and S. G. Thompson, Mendelian randomization: methods for using
genetic variants in causal estimation . CRC Press, 2015.

 

 
 [142] 
 
S. Burgess, F. Dudbridge, and S. G. Thompson, “Combining information on
multiple instrumental variables in mendelian randomization: comparison of
allele score and summarized data methods,” Statistics in medicine ,
vol. 35, no. 11, pp. 1880–1906, 2016.

 

 
 [143] 
 
N. M. Davies, S. von Hinke Kessler Scholder, H. Farbmacher, S. Burgess,
F. Windmeijer, and G. D. Smith, “The many weak instruments problem and
mendelian randomization,” Statistics in medicine , vol. 34, no. 3, pp.
454–468, 2015.

 

 
 [144] 
 
P. Sebastiani, N. Solovieff, and J. Sun, “Naïve bayesian classifier and
genetic risk score for genetic risk prediction of a categorical trait: not so
different after all!” Frontiers in genetics , vol. 3, p. 26, 2012.

 

 
 [145] 
 
N. Sokolovska and P.-H. Wuillemin, “The role of instrumental variables in
causal inference based on independence of cause and mechanism,”
 Entropy , vol. 23, no. 8, p. 928, 2021.

 

 
 [146] 
 
A. Wu, K. Kuang, R. Xiong, M. Zhu, Y. Liu, B. Li, F. Liu, Z. Wang, and F. Wu,
“Treatment effect estimation with unmeasured confounders in data fusion,”
 arXiv preprint arXiv:2208.10912 , 2022.

 

 
 [147] 
 
Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning
applied to document recognition,” Proceedings of the IEEE , vol. 86,
no. 11, pp. 2278–2324, 1998.

 

 
 [148] 
 
J. L. Hill, “Bayesian nonparametric modeling for causal inference,”
 Journal of Computational and Graphical Statistics , vol. 20, no. 1, pp.
217–240, 2011.

 

 
 [149] 
 
A. Pokropek, “Introduction to instrumental variables and their application to
large-scale assessment data,” Large-scale Assessments in Education ,
vol. 4, no. 1, pp. 1–20, 2016.

 

 
 [150] 
 
W. Schulz, J. Ainley, and J. Fraillon, “Iccs 2009 technical report,” 2011.

 

 
 [151] 
 
H. Domański, M. Federowicz, A. Pokropek, D. Przybysz, M. Sitek,
M. Smulczyk, and T. Żółtak, “From school to work: Individual and
institutional determinants of educational and occupational career
trajectories of young poles,” ASK: Research Methods , vol. 21,
no. 1, pp. 123–141, 2012.

 

 
 [152] 
 
N. J. Timpson, A. Sayers, G. Davey-Smith, and J. H. Tobias, “How does body fat
influence bone mass in childhood? a mendelian randomization approach,”
 Journal of Bone and Mineral Research , vol. 24, no. 3, pp. 522–533,
2009.

 

 
 [153] 
 
T. M. Palmer, D. A. Lawlor, R. M. Harbord, N. A. Sheehan, J. H. Tobias, N. J.
Timpson, G. D. Smith, and J. A. Sterne, “Using multiple genetic variants as
instrumental variables for modifiable risk factors,” Statistical
methods in medical research , vol. 21, no. 3, pp. 223–242, 2012.

 

 
 [154] 
 
J. Golding, M. Pembrey, R. Jones et al. , “Alspac–the avon longitudinal
study of parents and children. i. study methodology.” Paediatric and
perinatal epidemiology , vol. 15, no. 1, pp. 74–87, 2001.

 

 
 [155] 
 
G. Hemani, J. Zheng, B. Elsworth, K. H. Wade, V. Haberland, D. Baird,
C. Laurin, S. Burgess, J. Bowden, R. Langdon et al. , “The mr-base
platform supports systematic causal inference across the human phenome,”
 elife , vol. 7, p. e34408, 2018.

 

 
 [156] 
 
R. Gray and K. Wheatley, “How to avoid bias when comparing bone marrow
transplantation with chemotherapy.” Bone marrow transplantation ,
vol. 7, pp. 9–12, 1991.

 

 
 [157] 
 
L. Youngman, B. Keavney, A. Palmer, S. Parish, S. Clark, J. Danesh,
M. Delepine, M. Lathrop, R. Peto, and R. Collins, “Plasma fibrinogen and
fibrinogen genotypes in 4685 cases of myocardial infarction and in 6002
controls: Test of causality by” mendelian randomisation”,”
 Circulation , vol. 102, no. 18, 2000.

 

 
 [158] 
 
G. Davey Smith and S. Ebrahim, “‘mendelian randomization’: can genetic
epidemiology contribute to understanding environmental determinants of
disease?” International journal of epidemiology , vol. 32, no. 1, pp.
1–22, 2003.

 

 
 [159] 
 
D. C. Thomas and D. V. Conti, “Commentary: the concept of ‘mendelian
randomization’,” International journal of epidemiology , vol. 33,
no. 1, pp. 21–25, 2004.

 

 
 [160] 
 
G. D. Smith, “Capitalizing on mendelian randomization to assess the effects of
treatments,” Journal of the Royal Society of Medicine , vol. 100,
no. 9, pp. 432–435, 2007.

 

 
 [161] 
 
G. Thanassoulis and C. J. O’Donnell, “Mendelian randomization: nature’s
randomized trial in the post–genome era,” Jama , vol. 301, no. 22,
pp. 2386–2388, 2009.

 

 
 [162] 
 
S. Von Hinke, G. D. Smith, D. A. Lawlor, C. Propper, and F. Windmeijer,
“Genetic markers as instrumental variables,” Journal of Health
Economics , vol. 45, pp. 131–148, 2016.

 

 
 [163] 
 
N. J. Timpson, D. A. Lawlor, R. M. Harbord, T. R. Gaunt, I. N. Day, L. J.
Palmer, A. T. Hattersley, S. Ebrahim, G. D. Lowe, A. Rumley et al. ,
“C-reactive protein and its role in metabolic syndrome: mendelian
randomisation study,” The Lancet , vol. 366, no. 9501, pp. 1954–1959,
2005.

 

 
 [164] 
 
J. Zheng, D. Baird, M.-C. Borges, J. Bowden, G. Hemani, P. Haycock, D. M.
Evans, and G. D. Smith, “Recent developments in mendelian randomization
studies,” Current epidemiology reports , vol. 4, no. 4, pp. 330–345,
2017.

 

 
 [165] 
 
N. M. Davies, M. V. Holmes, and G. D. Smith, “Reading mendelian randomisation
studies: a guide, glossary, and checklist for clinicians,” Bmj , vol.
362, 2018.

 

 
 [166] 
 
S. Burgess, R. A. Scott, N. J. Timpson, G. Davey Smith, and S. G. Thompson,
“Using published data in mendelian randomization: a blueprint for efficient
identification of causal risk factors,” European journal of
epidemiology , vol. 30, no. 7, pp. 543–552, 2015.

 

 
 [167] 
 
F. P. Hartwig, N. M. Davies, G. Hemani, and G. Davey Smith, “Two-sample
mendelian randomization: avoiding the downsides of a powerful, widely
applicable but potentially fallible technique,” pp. 1717–1726, 2016.

 

 
 [168] 
 
N. J. Timpson, B. G. Nordestgaard, R. M. Harbord, J. Zacho, T. M. Frayling,
A. Tybjærg-Hansen, and G. Davey Smith, “C-reactive protein levels and
body mass index: elucidating direction of causation through reciprocal
mendelian randomization,” International journal of obesity , vol. 35,
no. 2, pp. 300–308, 2011.

 

 
 [169] 
 
S. Burgess, R. M. Daniel, A. S. Butterworth, S. G. Thompson, and E.-I.
Consortium, “Network mendelian randomization: using genetic variants as
instrumental variables to investigate mediation in causal pathways,”
 International journal of epidemiology , vol. 44, no. 2, pp. 484–495,
2015.

 

 
 [170] 
 
S. Burgess and S. G. Thompson, “Multivariable mendelian randomization: the use
of pleiotropic genetic variants to estimate causal effects,” American
journal of epidemiology , vol. 181, no. 4, pp. 251–260, 2015.

 

 
 [171] 
 
J. P. Kemp, A. Sayers, G. D. Smith, J. H. Tobias, and D. M. Evans, “Using
mendelian randomization to investigate a possible causal relationship between
adiposity and increased bone mineral density at different skeletal sites in
children,” International journal of epidemiology , vol. 45, no. 5, pp.
1560–1572, 2016.

 

 
 [172] 
 
B. A. Ference, J. J. Kastelein, H. N. Ginsberg, M. J. Chapman, S. J. Nicholls,
K. K. Ray, C. J. Packard, U. Laufs, R. D. Brook, C. Oliver-Williams
 et al. , “Association of genetic variants related to cetp inhibitors
and statins with lipoprotein levels and cardiovascular risk,” Jama ,
vol. 318, no. 10, pp. 947–956, 2017.

 

 
 [173] 
 
C. Jencks and S. E. Mayer, “The social consequences of growing up in a poor
neighborhood,” Inner-city poverty in the United States , vol. 111, p.
186, 1990.

 

 
 [174] 
 
E. M. Foster, “Instrumental variables for logistic regression: an
illustration,” Social Science Research , vol. 26, no. 4, pp. 487–504,
1997.

 

 
 [175] 
 
A. J. O’Malley, F. Elwert, J. N. Rosenquist, A. M. Zaslavsky, and N. A.
Christakis, “Estimating peer effects in longitudinal dyadic data using
instrumental variables,” Biometrics , vol. 70, no. 3, pp. 506–515,
2014.

 

 
 [176] 
 
W. An, “Instrumental variables estimates of peer effects in social networks,”
 Social Science Research , vol. 50, pp. 382–394, 2015.

 

 
 [177] 
 
E. M. Foster and S. McLanahan, “An illustration of the use of instrumental
variables: Do neighborhood conditions affect a young person’s chance of
finishing high school?” Psychological Methods , vol. 1, no. 3, p. 249,
1996.

 

 
 [178] 
 
S. Han and K.-G. Park, “Social network services and their effects on network
social capital: an instrumental variables approach,” International
Journal of Mobile Communications , vol. 18, no. 4, pp. 386–404, 2020.

 

 
 [179] 
 
M. L. Minsky, Theory of neural-analog reinforcement systems and its
application to the brain-model problem . Princeton University, 1954.

 

 
 [180] 
 
C. J. C. H. Watkins, “Learning from delayed rewards,” 1989.

 

 
 [181] 
 
A. Forney, J. Pearl, and E. Bareinboim, “Counterfactual data-fusion for online
reinforcement learners,” in International Conference on Machine
Learning . PMLR, 2017, pp. 1156–1164.

 

 
 [182] 
 
S. J. Gershman, “Reinforcement learning and causal models,” The Oxford
handbook of causal reasoning , vol. 1, p. 295, 2017.

 

 
 [183] 
 
L. P. Kaelbling, M. L. Littman, and A. W. Moore, “Reinforcement learning: A
survey,” Journal of artificial intelligence research , vol. 4, pp.
237–285, 1996.

 

 
 [184] 
 
C. Lei, “Deep reinforcement learning,” in Deep Learning and Practice
with MindSpore . Springer, 2021, pp.
217–243.

 

 
 [185] 
 
N. Kallus and A. Zhou, “Confounding-robust policy evaluation in
infinite-horizon reinforcement learning,” Advances in Neural
Information Processing Systems , vol. 33, pp. 22 293–22 304, 2020.

 

 
 [186] 
 
D. Precup, “Eligibility traces for off-policy policy evaluation,”
 Computer Science Department Faculty Publication Series , p. 80, 2000.

 

 
 [187] 
 
M. Dudík, J. Langford, and L. Li, “Doubly robust policy evaluation and
learning,” arXiv preprint arXiv:1103.4601 , 2011.

 

 
 [188] 
 
A. Swaminathan and T. Joachims, “Counterfactual risk minimization: Learning
from logged bandit feedback,” in International Conference on Machine
Learning . PMLR, 2015, pp. 814–823.

 

 
 [189] 
 
A. Swaminathan, A. Krishnamurthy, A. Agarwal, M. Dudik, J. Langford, D. Jose,
and I. Zitouni, “Off-policy evaluation for slate recommendation,”
 Advances in Neural Information Processing Systems , vol. 30, 2017.

 

 
 [190] 
 
H. Zou, K. Kuang, B. Chen, P. Chen, and P. Cui, “Focused context balancing for
robust offline policy evaluation,” in Proceedings of the 25th ACM
SIGKDD International Conference on Knowledge Discovery Data Mining , 2019,
pp. 696–704.

 

 
 [191] 
 
D. Kumor, J. Zhang, and E. Bareinboim, “Sequential causal imitation learning
with unobserved confounders,” Advances in Neural Information
Processing Systems , vol. 34, 2021.

 

 
 [192] 
 
X. Xu, H.-g. He, and D. Hu, “Efficient reinforcement learning using recursive
least-squares methods,” Journal of Artificial Intelligence Research ,
vol. 16, pp. 259–292, 2002.

 

 
 [193] 
 
Y. Chen, L. Xu, C. Gulcehre, T. L. Paine, A. Gretton, N. de Freitas, and
A. Doucet, “On instrumental variable regression for deep offline policy
evaluation,” arXiv preprint arXiv:2105.10148 , 2021.

 

 
 [194] 
 
L. Liao, Z. Fu, Z. Yang, Y. Wang, M. Kolar, and Z. Wang, “Instrumental
variable value iteration for causal offline reinforcement learning,”
 arXiv preprint arXiv:2102.09907 , 2021.

 

 
 [195] 
 
J. Li, Y. Luo, and X. Zhang, “Causal reinforcement learning: An instrumental
variable approach,” Available at SSRN 3792824 , 2021.

 

 
 [196] 
 
T. Schnabel, A. Swaminathan, A. Singh, N. Chandak, and T. Joachims,
“Recommendations as treatments: Debiasing learning and evaluation,” in
 international conference on machine learning . PMLR, 2016, pp. 1670–1679.

 

 
 [197] 
 
A. Lada, A. Peysakhovich, D. Aparicio, and M. Bailey, “Observational data for
heterogeneous treatment effects with application to recommender systems,” in
 Proceedings of the 2019 ACM Conference on Economics and Computation ,
2019, pp. 199–213.

 

 
 [198] 
 
A. Sharma, J. M. Hofman, and D. J. Watts, “Estimating the causal impact of
recommendation systems from observational data,” in Proceedings of the
Sixteenth ACM Conference on Economics and Computation , 2015, pp. 453–470.

 

 
 [199] 
 
Z. Si, X. Han, X. Zhang, J. Xu, Y. Yin, Y. Song, and J.-R. Wen, “A
model-agnostic causal learning framework for recommendation using search
data,” arXiv preprint arXiv:2202.04514 , 2022.

 

 
 [200] 
 
K. Tang, M. Tao, and H. Zhang, “Adversarial visual robustness by causal
intervention,” arXiv preprint arXiv:2106.09534 , 2021.

 

 
 [201] 
 
M. J. Arcaro, S. A. McMains, B. D. Singer, and S. Kastner, “Retinotopic
organization of human ventral visual cortex,” Journal of
neuroscience , vol. 29, no. 34, pp. 10 638–10 652, 2009.

 

 
 [202] 
 
J. Yuan, X. Ma, K. Kuang, R. Xiong, M. Gong, and L. Lin, “Learning
domain-invariant relationship with instrumental variable for domain
generalization,” arXiv preprint arXiv:2110.01438 , 2021.