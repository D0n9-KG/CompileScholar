Structured learning of rigid-body dynamics:A survey and unified view from a robotics perspective 
 
 
 

 
 
 
 
 
 

 
 
 
 
 

 
 Title: 
 

 Content selection saved. Describe the issue below:

 Description: 
 
 

 
 
 
 
 arXiv is now an independent nonprofit! 
 Learn more 
 
 

 
 
 
 
 License: CC BY-NC-SA 4.0
 
 
arXiv:2012.06250v2 [cs.LG] 16 Apr 2021 
 
 

# Structured learning of rigid-body dynamics:
 A survey and unified view from a robotics perspective

 Publication type:  Article 
 
 
 A. René Geist
 
    
 Sebastian Trimpe
 
 Address:  Intelligent Control Systems Group, Max Planck Institute for Intelligent Systems, \state Stuttgart, Germany
 
 Address:  Institute for Data Science in Mechanical Engineering, RWTH Aachen University, \state Aachen, Germany
 
 Email:  geist@is.mpg.de 
 
 Accepted  Day Month Year 

 Abstract 
 
 Accurate models of mechanical system dynamics are often critical for model-based control and reinforcement learning. Fully data-driven dynamics models promise to ease the process of modeling and analysis, but require considerable amounts of data for training and often do not generalize well to unseen parts of the state space. Combining data-driven modelling with prior analytical knowledge is an attractive alternative as the inclusion of structural knowledge into a regression model improves the model’s data efficiency and physical integrity. In this article, we survey supervised regression models that combine rigid-body mechanics with data-driven modelling techniques. We analyze the different latent functions (such as kinetic energy or dissipative forces) and operators (such as differential operators and projection matrices) underlying common descriptions of rigid-body mechanics. Based on this analysis, we provide a unified view on the combination of data-driven regression models, such as neural networks and Gaussian processes, with analytical model priors. Further, we review and discuss key techniques for designing structured models such as automatic differentiation.

 
 
 keywords Informed Machine Learning, Hybrid modeling, Analytical mechanics
 † † corresponding: A. René Geist, Heisenbergstr. 3, 70569 Stuttgart
Germany, † † 
 
 
 Abbreviations: ML: Machine learning; GP: Gaussian Process; NN: Neural network; EOM: Equations of motion; ODE: Ordinary differential equation; 
 .         ALM: Analytical latent modelling; ARM: Analytical (output) residual modelling; AD: Automatic differentiation; 
 .         APN: Analytical parametric networks; COG: Center of gravity 
 

## 1 Introduction

 
 In recent decades, increasing interest has been put on the control of mobile robots and similar agile mechanical systems. Promising frameworks for the synthesis of control policies are model-predictive control [ 1 ] and model-based reinforcement learning [ 2 ] .
The performance of these algorithms rely on, or at least significantly benefit from an accurate model of the mechanical system dynamics.
A dynamics model f ^ ​ ( x , θ ) \hat{f}(x,\theta) seeks to minimize the error to the real dynamics function f ⁡ ( x ) f(x) , writing

 

 
 | 
 ϵ f ​ ( x ) = f ⁡ ( x ) − f ^ ​ ( x , θ ) , \epsilon_{f}(x)=f(x)-\hat{f}(x,\theta), | 
 | 
 (1) | 
 

 with model parameters θ \theta and model inputs x x .
Many interesting mechanical systems possess complex non-linear dynamics and operate in large regions of their state-space. Therefore, numerous works on dynamics modeling choose the direct identification of the system’s non-linear dynamics instead of local linear approximations. For learning a system’s nonlinear dynamics, several different model types have been suggested (cf. Fig.  1 ), namely:

 
 • 
 
 Analytical models (white-box models) , which consist of functional relationships motivated by first-principles.

 

 • 
 
 Data-driven models (black-box models) , which consist of solely data-driven models.

 

 • 
 
 Analytical structured models (gray-box models) , which denote combinations of analytical with data-driven models.

 

 
 However, the identification of the dynamics of mechanical systems from data is oftentimes challenging because real-world systems are high-dimensional and subject to complex physical phenomena such as friction and contacts. In addition, data collection on physical systems is expensive and time-consuming. These aspects in combination with the curse of dimensionality [ 3 , p. 190] aggravate pure data-driven modeling.

 
 
 For these reasons, recent works aim at combining data-driven modeling techniques with a-priori available knowledge, which we will denote as structured models . A rich source for a-priori structural knowledge in numerous mechanical systems constitutes analytical rigid-body dynamics. While the bodies of a mechanical system are usually slightly elastic, it has been shown in literature that the assumption of rigid-bodies yields a useful model prior for nonlinear dynamics modeling [ 4 , 5 , 6 , 7 ] .

 
 
 In this survey, we give a unified view on the state-of-the-art in structured learning for robotic systems using rigid-body dynamics. Structured models inherit the potential to combine the advantages of both analytical and data-driven modeling while avoiding the shortcoming of these modeling approaches when being used individually. However, a model will be rarely used in practice if its synthesis requires significant effort and its optimization is laborious. Therefore, we emphasize in this survey that recent automatic differentiation libraries such as JAX [ 8 ] and PyTorch [ 9 ] considerably simplify the synthesis of structured models and also enable GPU accelerated estimation of the model’s parameters. In turn, structured modeling has the potential to emerge as one of the defining model classes that is used in the control synthesis and operation of future generations of robotic systems.

 
 Before we present our view on structured modeling of rigid-body mechanics, we first define what we understand as dynamics functions as well as discuss the strengths and shortcomings of analytical and data-driven modeling in Section 1.1 and Section 1.2 as also summarized in Table 1 .

 
 {forest} 
 Figure 1 : Overview of different types of nonlinear dynamics models with a focus on structured modeling. The synthesis of a structured model depends on which part of an analytical model is substituted by a data-driven model. For example, a data-driven model can estimate specific forces or the entries of the inertia matrix inside of an analytical model. 
 
 
 Nonlinear dynamics functions of mechanical systems 

 
 The field of dynamics consists of kinematic equations describing the motion of a system via position, velocity, and acceleration, as well as kinetics, which evolves around the causal effects of generalized forces on the motion of body masses. We assume that we have a thorough understanding of the system’s kinematics, which yields a generalized coordinate description as well as implicit constraint equations. When it comes to structured modeling, we are interested in utilizing knowledge on the causal relationships between the system’s mass, impressed forces, and its motion.
In most modern analytical mechanics textbooks [ 10 , 11 ] , the term dynamics is used as a direct substitute for kinetics.

 
 
 We assume that the configuration of a rigid-body mechanical system is described via the generalized position, velocity, and acceleration, which are denoted by the n n -dimensional vectors q ⁡ ( t ) q(t) , q ˙ ​ ( t ) \dot{q}(t) , and q ¨ ​ ( t ) \ddot{q}(t) , respectively. Although these vectors depend on time t t , we generally omit the time dependence for these and other variables if this is clear from the context. Further, a specific discrete-time instant is denoted by the subscript k k , the subsequent discrete-time step by k + 1 k+1 . In addition, the system is subject to a control input u ⁡ ( t ) u(t) or u k ∈ ℝ w u_{k}\in\mathbb{R}^{w} . The control input induces actuation forces Q u ​ ( q , q ˙ , u ) Q_{\text{u}}(q,\dot{q},u) onto the system, which change the system’s total energy. In the case of robots with actuated joints, Q u Q_{\text{u}} typically represents all forces that are caused by friction and actuation inside the joints. In this case, Q u Q_{\text{u}} is referred to as joint torques (usually being denoted as τ \tau ). Many mechanical systems are control-affine that is Q u = B ⁡ ( q , q ˙ ) ​ u Q_{\text{u}}=B(q,\dot{q})u with B ⁡ ( q , q ˙ ) ∈ ℝ n × w B(q,\dot{q})\in\mathbb{R}^{n\times w} .
For the sake of brevity, we assume that the mechanical system is fully observable . That is, we can directly measure the state variables, or obtain a state estimate.

 
 
 The transition dynamics f T : ℝ 2 ​ n + w → ℝ 2 ​ n f_{\text{T}}:\mathbb{R}^{2n+w}\rightarrow\mathbb{R}^{2n} denotes a function from the current state-control vector to the next state, writing

 

 
 | 
 [ q k + 1 q ˙ k + 1 ] \displaystyle\left[\begin{array}[]{c}q_{k+1}\\
\dot{q}_{k+1}\end{array}\right] | 
 = f T ​ ( q k , q ˙ k , u k ) . \displaystyle=f_{\text{T}}(q_{k},\dot{q}_{k},u_{k}). | 
 | 
 

 While the transition dynamics are readily amenable to data-driven models, analytical rigid-body dynamics are usually described by ordinary differential equations (ODEs).
In turn, the transition dynamics can be split up into a continuous forward dynamics function f q ¨ : ℝ 2 ​ n + w → ℝ n f_{\ddot{q}}:\mathbb{R}^{2n+w}\rightarrow\mathbb{R}^{n} , which maps the state-control vector to the system’s acceleration,

 

 
 | 
 q ¨ \displaystyle\ddot{q} | 
 = f q ¨ ​ ( q , q ˙ , Q u ) , \displaystyle=f_{\ddot{q}}(q,\dot{q},Q_{\text{u}}), | 
 | 
 (4) | 
 

 and a numerical integration method predicting the next state using q ¨ \ddot{q} .
Commonly used integration methods such as Runge-Kutta-45 predict the next state as a function of several state-acceleration vectors at different time steps [ 12 ] .

 
 
 Alternatively, one can rearrange the forward dynamics to obtain the system’s inverse dynamics function f u : ℝ 3 ​ n → ℝ n f_{u}:\mathbb{R}^{3n}\rightarrow\mathbb{R}^{n} , which computes the (feed-forward) generalized actuation force that is required to achieve a state-acceleration vector, writing

 

 
 | 
 Q u \displaystyle Q_{\text{u}} | 
 = f u ​ ( q , q ˙ , q ¨ ) . \displaystyle=f_{u}(q,\dot{q},\ddot{q}). | 
 | 
 (5) | 
 

 The inverse dynamics mapping in ( 5 ) can be non-injective. In this case, the identification of inverse dynamics via supervised learning models requires additional mathematical techniques [ 13 ] , which are not discussed in this survey.

 
 To simplify the discussion, we use the term dynamics interchangeably for functions modeling either the transition, forward, or inverse dynamics functions of a mechanical system. To this end, we consider the dynamics of a mechanical system to be described by the general function f : ℝ n x → ℝ n y f:\mathbb{R}^{n_{x}}\rightarrow\mathbb{R}^{n_{y}} where f ⁡ ( x ) = y f(x)=y with input vector x ∈ ℝ n x x\in\mathbb{R}^{n_{x}} and output vector y ∈ ℝ n y y\in\mathbb{R}^{n_{y}} . For example, if f f denotes the system’s forward dynamics, then x x is a state-control vector and y y the system’s acceleration.

 
 
 
 Actuation dynamics 

 
 In the control of a mechanical system, one usually calculates the desired actuation force Q u,desired Q_{\text{u,desired}} that is sent to computational routines which in return cause the actuators to impress Q u Q_{\text{u}} onto the system.
Yet, Q u Q_{u} likely deviates from Q u,desired Q_{\text{u,desired}} due to friction and other unmodelled physical phenomena inside the actuators, flexibilities in gear-belts or attached shafts, free-play in an attached transmission, and internal dynamics of cascaded feedback control loops.
To account for this difference, one can define the actuation dynamics function f ACT : ℝ n ρ → ℝ n f_{\text{ACT}}:\mathbb{R}^{n_{\rho}}\rightarrow\mathbb{R}^{n} as

 

 
 | 
 Q u ′ = Q u ​ ( ρ , Q u,desired ) − Q u,desired ​ ( ρ ) = f ACT ​ ( ρ , Q u,desired ) , Q_{\text{u}}^{\prime}=Q_{\text{u}}(\rho,Q_{\text{u,desired}})-Q_{\text{u,desired}}(\rho)=f_{\text{ACT}}(\rho,Q_{\text{u,desired}}), | 
 | 
 (6) | 
 

 where ρ ∈ ℝ n ρ \rho\in\mathbb{R}^{n_{\rho}} denotes a suitably chosen vector of observables such as state data { q , q ˙ } \{q,\dot{q}\} from several time steps, the error to a target state, or the current applied to the actuators.
For example, the generation of torque inside a robot arm’s brush-less motors requires
that a Q u,desired Q_{\text{u,desired}} is transformed by low-level control routines into motor currents that cause a motor torque; the motor torque is altered by a potential gear train as well as joint friction; and finally Q u Q_{\text{u}} is measurable in the joints.
While the actuation dynamics can play a significant role in practical robot implementations, rigid-body dynamics typically starts at the level of forces. In this survey, we mostly focus on structured learning of the rigid-body dynamics.

 
 
 
 Supervised regression 

 
 In this survey, we assume that a model f ^ ​ ( x , θ ) = y ^ \hat{f}(x;\theta)=\hat{y} with model parameters θ \theta shall be trained on informative data

 

 
 | 
 𝒟 ​ = ^ ​ { x ~ k , y ~ k } k = 1 N \mathcal{D}\,\widehat{=}\,\{\tilde{x}_{k},\tilde{y}_{k}\}_{k=1}^{N} | 
 | 
 (7) | 
 

 where N N denotes the number of data points, x ~ \tilde{x} denotes noisy measurements of x x , and y ~ \tilde{y} denotes noisy measurements of f f . The term training or learning refers to the estimation of θ \theta from data. The term supervised refers to a model being trained on input-output data pairs of the dynamics function.

 
 
 

### 1.1 Analytical models

 
 Analytical models are commonly used in the identification of the dynamics of mechanical systems ( Sutanto et al., 2020 ; Atkeson et al., 1986 ; Ting et al., 2006 ; Traversaro et al., 2016 ; Wensing et al., 2017 ; Ledezma and Haddadin, 2018 ) .
Analytical models of rigid-body dynamics are based on physical axioms and principles underlying the motion of mechanical systems, e.g. , Newton’s axioms of motion. These axioms and principles spawn dynamics equations such as the Newton-Euler equations and the Euler-Lagrange equations. The equations of analytical mechanics are combinations of numerous physically motivated functions such as coordinate-transformations, forces, and the inertia matrix which depend on physical parameters. Physical parameters are for example the mass of a rigid-body, a friction coefficient, the length of a kinematic link, or the stiffness coefficient of a spring. The numerous functions that form an analytical model can usually not be observed individually which is why we refer to them as being latent . Notably, the latent functions and parameter estimates of analytical models yield a physical interpretation that can guarantee out of sample generalization (i.e., the validity of the model in regions where no data has been observed) and earns them the name white-box models. However, on real systems, physical phenomenons such as friction , damping , and contacts aggravate an accurate identification of an analytical dynamic model. In addition, other physical phenomena such as body elasticities cannot be modeled by analytical rigid-body dynamics in the first place and therefore lead to errors. In practice, some of the parameters of an analytical model are estimated from data using linear regression or gradient-based optimization.
However, the analytical model errors and the limited representative power of analytical force models can lead to physically inconsistent parameter estimates such as negative body masses Ting et al. (2006) . Analytical models whose parameters are estimated via gradient-based optimization could be seen as parametric data-driven models which is why we refer to them as analytical parametric networks (APN). Analytical parametric networks provide profound insights for the synthesis of analytical structured models, which is why we briefly discuss these works in Section 3.2.2 .

 
 
 Table 1 : Pros and cons of analytical rigid-body models compared to data-driven models. 
 
 
 
 Analytical models | 
 Data-driven models | 

 
 
 
 +  Data-efficient | 
 -   Data-hungry | 

 
 +  Human-interpretable | 
 -   Usually black-box model | 

 
 +  Out-of-sample generalization | 
 -   Usually data-point interpolation | 

 
 -   Large prediction errors | 
 +  Less restrictive assumptions | 

 
 -   Full prior knowledge required | 
 +  Less prior knowledge required | 

 
 ∘ \circ   Deterministic | 
 ∘ \circ   Possibly probabilistic | 

 

 
 
 

### 1.2 Data-driven models

 
 When analytical modeling is not feasible (e.g, because the physical process is unknown), or the effort required to obtain a sufficiently small prediction error appears to be too large, the nonlinear dynamics can be directly learned from data. The design of the data-driven model places prior assumptions on the functions it can approximate.
To reduce the prediction error of these models, one needs to estimate their parameters, which is referred to as training. Parameters can be either the hyperparameters of the kernel of a non-parametric method, such as a Gaussian process (GP), or the parameters of a parametric method such as a neural network (NN). As the same optimization methods are commonly used for training parametric models as well as GPs, we refer in what follows to hyper-parameters also as parameters. However, training hyperparameters of a kernel determines an entire population of features while training the parameters of a parametric model usually determines a function approximation.

 
 
 Data-driven models contrast analytical models as most often their parameter estimates do not yield a physically insightful interpretation.
Moreover, the sheer amount of parameters and interlinked latent functions in most data-driven models such as neural networks eludes human comprehension. Thus they are commonly referred to as black-box models.

 
 
 One of the largest concern with the identification of dynamics via data-driven models is sample complexity ; that is, the number of training points a data-driven model requires to successfully approximate a function. On physical systems, data is often sparse because the data collection is subject to life-time and cost constraints. While it has been proposed to identify structural knowledge directly from data Sahoo et al. (2018) ; Baumann et al. (2020) , the identification of structure requires significant amounts of data. In the next section, we detail our understanding of structure, and why it is beneficial to combine analytical structure with data-driven modeling.

 
 
 

### 1.3 Structured models

 
 In this survey, we interpret the term structure as either mathematical properties of functions, or alternatively, causal dependencies between functions. In the field of nonlinear system identification, a gray-box model denotes the combination of analytical with data-driven modeling Nelles (2013) ; Lennart (1999) . In slight contrast, a structured model denotes the combination of data-driven models with some form of structural prior knowledge that reduces the required complexity of the model class (cf. ( Nelles, 2013 , p. 192) ). Such structural knowledge can be an analytical model or alternatively, the way in which different data-driven models are being connected with each other. Therefore, a gray-box model is a structured model, but not all structured models are gray-box models. As suggested by Nelles (2013) , structured models can be further categorized as:

 
 • 
 
 Hybrid structures : The combination of different sub-models to a single structured model. 1 1 
 1 
 
 
 
 The term hybrid structures/models should not be confused with hybrid systems which commonly denote the combination of continuous and discrete-time systems. This model type is further divided into:

 
 – 
 
 Parallel model : The outputs of all sub-models are summed up to yield the output of the hybrid model.

 

 – 
 
 Series model : The sub-models are connected in series, that is the output of one sub-model forms the input to another sub-model and so forth.

 

 – 
 
 Parameter scheduling model : The parameters of a sub-model are scheduled by the output of another sub-model.

 

 
 

 • 
 
 Projection-based structures : The input space is projected into a lower-dimensional latent space.

 

 • 
 
 Additive structures : The input dimensions are grouped into several sub-spaces which act as input to sub-models. The additive model’s output is the sum of the sub-models’ outputs.

 

 • 
 
 Hierarchical structures : A model is composed of parallel, series, and additive structures yielding hierarchical causal relationships between the inputs and outputs of the sub-models.

 

 • 
 
 Input space decompositions : The input space is decomposed into several regions that each constitutes a sub-space to a sub-model. Note that the input space is split up region-wise rather than dimension-wise as in additive structures.

 

 
 Notably, analytical rigid body dynamics can yield many of the above types of structural knowledge, which makes it a versatile toolbox for structured learning.

 
 
 In the field of robotics, the term structured learning usually refers to the combination of analytical mechanics with data-driven modeling Lutter et al. (2019a) ; Gupta et al. (2020) ; Geist and Trimpe (2020) . The term structured mechanics model has also been used in robotics literature Gupta et al. (2020) .
As it is the analytical prior knowledge that structures a data-driven model, we use the term analytical structured model to refer to a structured model that uses analytical rigid-body mechanics as a model prior to data-driven modeling.

 
 
 A unified view on analytical structured models 

 
 The core idea of analytical structured modeling is to complement an analytical model via data-driven modeling to infer its errors from data.
An analytical model f ^ A ​ ( x , θ A ) \hat{f}_{\text{A}}(x;\theta_{\text{A}}) with parameters θ A \theta_{\text{A}} defines a hierarchical structure between several latent functions such as forces and inertias. In principle, one can directly use the analytical model to faithfully estimate θ A \theta_{\text{A}} by minimizing a suitable metric ‖ y ~ − f ^ A ​ ( x ~ , θ A ) ‖ \|\tilde{y}-\hat{f}_{\text{A}}(\tilde{x};\theta_{\text{A}})\| .
However, an analytical model is only an approximation of the dynamics f ⁡ ( x ) f(x) such that

 

 
 | 
 f ⁡ ( x ) = f ^ A ​ ( x , θ A ) + ϵ A ​ ( x , θ A ) , f(x)=\hat{f}_{\text{A}}(x;\theta_{\text{A}})+\epsilon_{\text{A}}(x;\theta_{\text{A}}), | 
 | 
 (8) | 
 

 with the anayltical model errors ϵ A ​ ( x ) \epsilon_{\text{A}}(x) .
To reduce ϵ A \epsilon_{\text{A}} and potentially improve the estimate of θ A \theta_{\text{A}} , one can add a data-driven model ϵ ^ A ​ ( x , θ D ) \hat{\epsilon}_{\text{A}}(x;\theta_{\text{D}}) with parameters θ D \theta_{\text{D}} to the analytical model, writing

 

 
 | 
 f ^ ​ ( x , θ A , θ D ) = f ^ A ​ ( x , θ A ) + ϵ ^ A ​ ( x , θ D ) . \hat{f}(x;\theta_{\text{A}},\theta_{\text{D}})=\hat{f}_{\text{A}}(x;\theta_{\text{A}})+\hat{\epsilon}_{\text{A}}(x;\theta_{\text{D}}). | 
 | 
 (9) | 
 

 We refer to the parallel model structure in ( 9 ) as analytical (output) residual modeling (ARM). Usually, the measurements obtained from the real dynamics are subject to additional measurement noise and bias ϵ y ​ ( x ~ ) \epsilon_{y}(\tilde{x}) , such that a system measurement follows from y ~ = f ^ A ​ ( x ~ , θ A ) + ϵ A ​ ( x ~ ) + ϵ y ​ ( x ~ ) \tilde{y}=\hat{f}_{\text{A}}(\tilde{x};\theta_{\text{A}})+\epsilon_{\text{A}}(\tilde{x})+\epsilon_{y}(\tilde{x}) . Therefore, the data driven model usually models the residual ϵ A ​ ( x ~ ) + ϵ y ​ ( x ~ ) \epsilon_{\text{A}}(\tilde{x})+\epsilon_{y}(\tilde{x}) jointly. ARM forms a natural extension to analytical modeling and is conceptually easy to implement (while there can be of course practical challenges such as noise). However, if f ^ A ​ ( x , θ A ) \hat{f}_{\text{A}}(x;\theta_{\text{A}}) is inaccurate, the data-efficiency of the resulting structured model suffers significantly. In what follows, we omit the arguments in the notation of a function if these are clear from context.

 
 
 Consequently, the question arises on how we can improve the prediction accuracy of f ^ A \hat{f}_{\text{A}} itself by using a data-driven model.
An important insight for structured learning is that the cause of parts of the analytical model error ϵ A \epsilon_{\text{A}} lies hidden in the functions that form the analytical model f ^ A \hat{f}_{\text{A}} .
For example, parts of ϵ A \epsilon_{\text{A}} can be caused by an inaccurate kinematics model or an inaccurate model for the friction forces. Therefore, instead of estimating directly ϵ A \epsilon_{\text{A}} , one can also substitute unknown latent functions or their residuals with a data-driven model inside f ^ A \hat{f}_{\text{A}} (to be made precise in Section 3 ).
We refer to a structured model in which data-driven models substitute/augment parts of f ^ A \hat{f}_{\text{A}} as analytical latent modeling (ALM).

 
 
 
 

### 1.4 Overview and notation

 
 The remainder of this article is organized as follows.
Section 2 introduces the reader to the analytical mechanics of rigid-body systems. This section introduces different types of structural knowledge while emphasizing the importance of linear operators in rigid-body mechanics.
While we do not aim at a
comprehensive overview of rigid-body dynamics herein,
Section 2 
outlines important aspects of analytical mechanics that are useful for the synthesis of structured models and needed for the purpose of this survey.
Section 3 discusses the current state of the art in analytical structured models. Via the discussion of the previous sections, we develop a unified view on ARM as well as ALM.
Section 4 details the key mathematical techniques required to combine the most common data-driven models, namely GPs and NNs, with analytical equations. Further, we emphasize the importance of recent developments on automatic differentiation libraries for a straightforward design and training of analytical structured models.
Finally, we discuss a case study of modeling the dynamics of a robot arm to illustrate the presented concepts.

 
 In this work, we adapt the following notation. A unit matrix is denoted by I I , and a matrix of zeros by 𝟎 \bm{0} . The vertical concatenation of vectors { v 1 , … , v n } \{v_{1},\dots,v_{n}\} is denoted as vec ​ { v 1 , … , v n } ​ = ^ ​ [ v 1 𝖳 ​ … ​ v n 𝖳 ] T \text{vec}\{v_{1},\dots,v_{n}\}\,\widehat{=}\,[v_{1}^{\mathrm{\sf T}}\dots v_{n}^{\mathrm{\sf T}}]^{T} . The null space of a matrix A : ℝ m × n A:\mathbb{R}^{m\times n} is defined as 𝖭 ⁡ ( A ) = { x ∈ ℝ n : A ​ x = 0 } \mathsf{N}(A)=\{x\in\mathbb{R}^{n}:Ax=0\} , its range space as 𝖱 ⁡ ( A ) = { y ∈ ℝ m : ∃ x ∈ ℝ n ​  such that  ​ y = A ​ x } \mathsf{R}(A)=\{y\in\mathbb{R}^{m}:\exists x\in\mathbb{R}^{n}\text{ such that }y=Ax\} , and further we have that ℝ n = 𝖱 ⁡ ( A T ) ⊕ 𝖭 ⁡ ( A ) \mathbb{R}^{n}=\mathsf{R}(A^{T})\oplus\mathsf{N}(A) and ℝ m = 𝖱 ⁡ ( A ) ⊕ 𝖭 ⁡ ( A T ) \mathbb{R}^{m}=\mathsf{R}(A)\oplus\mathsf{N}(A^{T}) ( Beard, 2002 ) . If a matrix A ⁡ ( x ) : ℝ m × n A(x):\mathbb{R}^{m\times n} with m ≤ n m\leq n and rank ​ ( A ) = m \text{rank}(A)=m then 𝖭 ⁡ ( A 𝖳 ) = { 0 } \mathsf{N}(A^{\mathrm{\sf T}})=\{0\} . A + A^{+} denotes the Moore-Penrose pseudo (MP) inverse of A A .
A vector x ∈ ℝ n x\in\mathbb{R}^{n} can be split up in a range space part x 𝖱 ⁡ ( A 𝖳 ) x_{\mathsf{R}(A^{\mathrm{\sf T}})} and a null space part x 𝖭 ⁡ ( A ) x_{\mathsf{N}(A)} , such that x = x 𝖱 ⁡ ( A 𝖳 ) + x 𝖭 ⁡ ( A ) x=x_{\mathsf{R}(A^{\mathrm{\sf T}})}+x_{\mathsf{N}(A)} . The projections of a vector into 𝖭 ⁡ ( A ) \mathsf{N}(A) and 𝖱 ⁡ ( A 𝖳 ) \mathsf{R}(A^{\mathrm{\sf T}}) , (cf. Beard (2002) ), are given by

 

 
 | 
 P 𝖭 ⁡ ( A ) = I − A + ​ A ,  and  ​ P 𝖱 ⁡ ( A 𝖳 ) = A + ​ A , P^{\mathsf{N}(A)}=I-A^{+}A,\ \text{ and }\ P^{\mathsf{R}(A^{\mathrm{\sf T}})}=A^{+}A, | 
 | 
 (10) | 
 

 such that x 𝖱 ⁡ ( A 𝖳 ) = P 𝖱 ⁡ ( A 𝖳 ) ​ x x_{\mathsf{R}(A^{\mathrm{\sf T}})}=P^{\mathsf{R}(A^{\mathrm{\sf T}})}x and x 𝖭 ⁡ ( A ) = P 𝖭 ⁡ ( A ) ​ x x_{\mathsf{N}(A)}=P^{\mathsf{N}(A)}x .
Generalized forces are denoted by Q ( ⋅ ) Q_{(\cdot)} , dimensions by n ( ⋅ ) n_{(\cdot)} , error functions by ϵ ( ⋅ ) \epsilon_{(\cdot)} , dynamics functions as f ( ⋅ ) f_{(\cdot)} , dynamics models as f ^ ( ⋅ ) \hat{f}_{(\cdot)} , and model parameters by θ ( ⋅ ) \theta_{(\cdot)} where the respective indice “ ( ⋅ ) (\cdot) ” specifies a particular type.

 
 
 
 

## 2 A glimpse on analytical rigid-body mechanics

 
 In this section, we give an overview on important aspects of rigid-body dynamics which provides the means to discuss current literature on analytical structured modeling. Here,
we show one possible way on how the system’s equations of motion (EOM) can emerge from the interplay of kinematics, dynamic principles, and constraint equations.

 
 
 We first limit the discussion to the Newton-Euler equations of a system of N b N_{\text{b}} rigid bodies subject to holonomic constraints. By using a set of independent generalized coordinates we eliminate the constraint forces from the EOM. Then, we briefly discuss how in the Euler-Lagrange equations certain forces can be modelled in terms of potential functions. These dynamics formulations are frequently used for the description of robot dynamics Siciliano et al. (2010) and the simulation of rigid-body systems Featherstone (2008) . Afterwards, we discuss how additional implicit constraints are incorporated into the EOM. With the discussion of explicit and implicit constraints, we illustrate how constraint equations provide structural knowledge on the vector spaces in which certain forces are bound to lie.

 
 
 Kinematics 

 
 To derive the EOMs of a multi-body system, the motion variables – position, velocity and acceleration – of its N b N_{b} rigid bodies must be described with respect to an inertial frame. The position of every point of the i i -th rigid body can be described by a position vector r i ​ ( t ) ∈ ℝ 3 r_{i}(t)\in\mathbb{R}^{3} pointing to the origin of a body-fixed frame with respect to an inertial frame and a rotation matrix R i ​ ( t ) ∈ S ​ O ​ ( 3 ) R_{i}(t)\in SO(3) describing the rotation of the body-fixed coordinate frame with respect to the inertial frame. S ​ O ​ ( 3 ) SO(3) denotes the subgroup of orthogonal matrices of size three with determinant + 1 +1 .
Subsequently, the multi-body system has at most 6 ​ N b 6N_{\text{b}} degrees of freedom (DOF). The velocity of the body is described by the the translational velocity r ˙ i ​ ( t ) \dot{r}_{i}(t) and the rotational velocity ω i ​ ( t ) \omega_{i}(t) . The rotational velocity ω i ​ ( t ) = φ ˙ i = [ ω 1 , ω 2 , ω 3 ] 𝖳 \omega_{i}(t)=\dot{\varphi}_{i}=[\omega_{1},\omega_{2},\omega_{3}]^{\mathrm{\sf T}} is obtained in terms of R ⁡ ( t ) R(t) (cf. ( Schiehlen and Eberhard, 2014 , p. 28) ) as

 

 
 | 
 crossp ​ { ω } = R ˙ ​ ( t ) ​ R ​ ( t ) 𝖳 ,  with  crossp ​ { ω } = [ 0 − ω 3 ω 2 ω 3 0 − ω 1 − ω 2 ω 1 0 ] , \text{crossp}\{\omega\}=\dot{R}(t)R(t)^{\mathrm{\sf T}},\text{ with }\text{crossp}\{\omega\}=\begin{bmatrix}0 -\omega_{3} \omega_{2}\\
\omega_{3} 0 -\omega_{1}\\
-\omega_{2} \omega_{1} 0\end{bmatrix}, | 
 | 
 (11) | 
 

 with the infinitesimal instantaneous rotation vector φ i ∈ ℝ 3 \varphi_{i}\in\mathbb{R}^{3} ( Woernle, , p. 67) .

 
 
 Instead of expressing r i ​ ( t ) r_{i}(t) in Cartesian coordinates, it is often more practical to formulate r i ​ ( t ) r_{i}(t) as well as R i ​ ( t ) R_{i}(t) in terms of a generalized coordinate vector q ⁡ ( t ) q(t) .
For example, one could use spherical coordinates to denote a point in space (cf. ( Schiehlen and Eberhard, 2014 , p. 14) ) or Cardano angles to describe rotations (cf. ( Schiehlen and Eberhard, 2014 , p. 24) ). The vector q q is termed minimal if it consists of independent coordinates that equal the system’s DOF.

 
 
 

### 2.1 Newton-Euler equations: Structural knowledge between forces and acceleration

 
 To keep the discussion concise, we make the following assumptions for Section 2.1 and Section 2.2 :

 
 • 
 
 The EOM of a system of N b N_{\text{b}} rigid bodies are expressed in terms of minimal generalized coordinates.

 

 • 
 
 The effect of the constraint forces onto the system are modelled through n E n_{\text{E}} independent holonomic constraint equations.

 

 • 
 
 The origin of the body-fixed frame lies at the body’s centre of gravity (COG).

 

 
 While these assumptions are common for the derivation of robot dynamics, other assumptions such as using an accelerated frame of reference or a reference system with its origin being not placed in the COG are also used. A more far-reaching description of multibody dynamics is given in Schiehlen and Eberhard (2014) .

 
 
 The Newton-Euler EOM of the i-th rigid body with respect to the COG read

 

 
 | 
 M ~ i ​ p ¨ i = F C , i + F e , i + F E , i \tilde{M}_{i}\ddot{p}_{i}=F_{\text{C},i}+F_{\text{e},i}+F_{\text{E},i} | 
 | 
 (12) | 
 

 with p ¨ i = vec ​ { r ¨ i , ω ˙ i } \ddot{p}_{i}=\text{vec}\{\ddot{r}_{i},\dot{\omega}_{i}\} , the block-diagonal matrix as M ~ i = diag ​ { m i ​ I 3 , Θ S , i } \tilde{M}_{i}=\textbf{diag}\{m_{i}I_{3},\Theta_{S,i}\} consisting of the body’s mass m i m_{i} and its inertia matrix Θ S , i \Theta_{S,i} , the bias vector F C , i = vec ​ { 𝟎 , − ω ~ ​ Θ S , i ​ ω i } F_{\text{C},i}=\text{vec}\{\bm{0},-\tilde{\omega}\Theta_{S,i}\omega_{i}\} , the impressed forces and torques F e , i = vec ​ { f e , i , τ e , i } F_{\text{e},i}=\text{vec}\{f_{\text{e},i},\tau_{\text{e},i}\} , and the explicit constraint forces and torques F E , i = vec ​ { f E , i , τ E , i } F_{\text{E},i}=\text{vec}\{f_{\text{E},i},\tau_{\text{E},i}\} ( Schiehlen and Eberhard, 2014 , p. 76) .

 
 
 Principle of virtual work and D’Alembert-Lagrange’s principle 

 
 To eliminate the constraint forces and torques from ( 12 ) it is assumed that the constraint forces are ideal , that is, the constraint forces do zero work under virtual displacements δ ​ r i ∈ ℝ 3 \delta r_{i}\in\mathbb{R}^{3} and the reaction torques do zero work under virtual rotations δ ​ φ i ∈ ℝ 3 \delta\varphi_{i}\in\mathbb{R}^{3} . In turn, with δ ​ p i = vec ​ { δ ​ r i , δ ​ φ i } \delta p_{i}=\text{vec}\{\delta r_{i},\delta\varphi_{i}\} , it is postulated that the ideal constraint forces F E , i F_{\text{E},i} respect the following inner product

 

 
 | 
 ∑ i = 1 N b δ ​ p i 𝖳 ​ F E , i = δ ​ p 𝖳 ​ F E = 0 , \sum_{i=1}^{N_{\text{b}}}\delta p_{i}^{\mathrm{\sf T}}F_{\text{E},i}=\delta p^{\mathrm{\sf T}}F_{\text{E}}=0, | 
 | 
 (13) | 
 

 with the δ ​ p i \delta p_{i} of all bodies being denoted jointly as δ ​ p = vec ​ { δ ​ p 1 , δ ​ p 2 , … ​ δ ​ p N b } \delta p=\text{vec}\{\delta p_{1},\delta p_{2},\dots\delta p_{N_{\text{b}}}\} as well as the explicit constraint forces of all bodies being denoted as F E = vec ​ { F E , 1 , F E , 2 , … , F E , N b } F_{\text{E}}=\text{vec}\{F_{\text{E},1},F_{\text{E},2},\dots,F_{\text{E},N_{\text{b}}}\} . Note that, if the i i -th body is not subject to any constraint forces then F E , i = 0 F_{\text{E},i}=0 .
The vectors of virtual displacements and virtual rotations denote infinitesimal vectors that are compatible with the constraints while by definition not varying the time variable (cf. ( Udwadia and Kalaba, 2007 , p. 133) , ( Schiehlen and Eberhard, 2014 , p. 85) ), see also Section 2.3 for a more in depth discussion. The D’Alembert-Lagrange principle ( d’Alembert, 1743 ; Lagrange, 1787 ) in its to multibody systems extended form ( Schiehlen and Eberhard, 2014 , p. 92) is obtained by multiplication of ( 12 ) from the left by δ ​ p i 𝖳 \delta p_{i}^{\mathrm{\sf T}} and applying ( 13 ), such that

 

 
 | 
 ∑ i = 1 N b δ ​ p i 𝖳 ​ ( M ~ i ​ p ¨ i − F c , i − F e , i ) = 0 . \sum_{i=1}^{N_{\text{b}}}\delta p_{i}^{\mathrm{\sf T}}\left(\tilde{M}_{i}\ddot{p}_{i}-F_{c,i}-F_{e,i}\right)=0. | 
 | 
 (14) | 
 

 Due to the presence of F E F_{\text{E}} , the components of δ ​ p i \delta p_{i} are dependent on each other. In what follows, we outline how to obtain the EOM from ( 14 ) by resorting to explicit constraint equations that are expressed in terms of a minimal set of generalized coordinates.

 
 
 
 
 Explicit holonomic constraints and generalized coordinates 

 
 In multi-body systems, the rigid bodies are usually subject to kinematic mechanisms which apply constraint forces onto the multibody system. The constraint forces reduce the system’s DOF. The effect of the constraint forces on the system’s kinematics can be modelled via constraint equations. In this work, the term constraints refers to algebraic equations that describe the system’s admissible states ( Featherstone, 2008 , p. 44) . Constraints can be either holonomic or nonholonomic. A constrained system is holonomic if its position variables are integrals of the velocity variables; otherwise, the system is nonholonomic ( Featherstone, 2008 , p. 41) . We briefly discuss nonholonomic systems in Section 2.3 .

 
 
 As the system has n q = 6 ​ N b − n E n_{\text{q}}=6N_{\text{b}}-n_{\text{E}} DOF and q ∈ ℝ n q q\in\mathbb{R}^{n_{\text{q}}} , the i i -th bodies position and orientation can be written in terms of explicit constraints as

 

 
 | 
 r i ​ ( t ) \displaystyle r_{i}(t) | 
 = r i ​ ( q , t ) , R i ​ ( t ) = R i ​ ( q , t ) . \displaystyle=r_{i}(q,t),\hskip 28.45274ptR_{i}(t)=R_{i}(q,t). | 
 | 
 (15) | 
 

 Note that a free system (that is no constraint forces act onto the bodies) can be seen as a special case of a holonomic constrained system in which n E = 0 n_{\text{E}}=0 , q q is of dimension 6 ​ N b 6N_{\text{b}} , and ( 15 ) denotes a suitable coordinate transformation ( Schiehlen and Eberhard, 2014 , 51) . Differentiation of ( 15 ) with respect to time yields constraint equations for the systems velocities and accelerations as

 

 
 | 
 r ˙ i ​ ( q , q ˙ , t ) \displaystyle\dot{r}_{i}(q,\dot{q},t) | 
 = J r , i ​ q ˙ + ∂ r i ∂ t , ω i ​ ( q , q ˙ , t ) \displaystyle=J_{r,i}\dot{q}+\frac{\partial r_{i}}{\partial t},\hskip 85.35826pt\omega_{i}(q,\dot{q},t) | 
 | 
 = J R , i ​ q ˙ + ∂ φ i ∂ t , \displaystyle=J_{R,i}\dot{q}+\frac{\partial\varphi_{i}}{\partial t}, | 
 | 
 (16) | 
 
 
 | 
 r ¨ i ​ ( q , q ˙ , q ¨ ) \displaystyle\ddot{r}_{i}(q,\dot{q},\ddot{q}) | 
 = J r , i ​ q ¨ + J ˙ r , i ​ q ˙ + ∂ r ˙ i ∂ t , ω ˙ i ​ ( q , q ˙ , q ¨ ) \displaystyle=J_{r,i}\ddot{q}+\dot{J}_{r,i}\dot{q}+\frac{\partial\dot{r}_{i}}{\partial t},\hskip 56.9055pt\dot{\omega}_{i}(q,\dot{q},\ddot{q}) | 
 | 
 = J R , i ​ q ¨ + J ˙ R , i ​ q ˙ + ∂ φ ˙ i ∂ t , \displaystyle=J_{R,i}\ddot{q}+\dot{J}_{R,i}\dot{q}+\frac{\partial\dot{\varphi}_{i}}{\partial t}, | 
 | 
 (17) | 
 

 with the translational Jacobian matrix J r , i ​ ( q , t ) ∈ ℝ 3 × n q J_{r,i}(q,t)\in\mathbb{R}^{3\times n_{\text{q}}} and the rotational Jacobian matrix J R , i ​ ( q , t ) ∈ ℝ 3 × n q J_{R,i}(q,t)\in\mathbb{R}^{3\times n_{\text{q}}} ( Woernle, , p. 195) . Note while J r , i ​ ( q , t ) = ∂ r i ​ ( q ) ∂ q J_{r,i}(q,t)=\frac{\partial r_{i}(q)}{\partial q} , the matrix J R , i ​ ( q , t ) = ∂ φ i ​ ( q ) ∂ q J_{R,i}(q,t)=\frac{\partial\varphi_{i}(q)}{\partial q} is often obtained by directly expressing ω i \omega_{i} in terms of q ˙ \dot{q} ( Schiehlen and Eberhard, 2014 , p. 33) . One can rewrite ( 17 ) more compactly as

 

 
 | 
 p ¨ i ​ ( q , q ˙ , q ¨ ) = J i ​ q ¨ + J ~ i , \ddot{p}_{i}(q,\dot{q},\ddot{q})=J_{i}\ddot{q}+\tilde{J}_{i}, | 
 | 
 (18) | 
 

 with the Jacobian matrix J i = [ J r , i 𝖳 , J R , i 𝖳 ] 𝖳 J_{i}=[J_{r,i}^{\mathrm{\sf T}},J_{R,i}^{\mathrm{\sf T}}]^{\mathrm{\sf T}} and J ~ i = vec ​ { J ˙ r , i ​ q ˙ + ∂ r ˙ i ∂ t , J ˙ R , i ​ q ˙ + ∂ φ ˙ i ∂ t } \tilde{J}_{i}=\text{vec}\{\dot{J}_{r,i}\dot{q}+\frac{\partial\dot{r}_{i}}{\partial t},\,\dot{J}_{R,i}\dot{q}+\frac{\partial\dot{\varphi}_{i}}{\partial t}\} .
Importantly, the virtual vector δ ​ p i \delta p_{i} can be expressed in terms of the generalized virtual displacement vector δ ​ q ∈ ℝ n q \delta q\in\mathbb{R}^{n_{q}} using ( 16 ) (cf. ( Schiehlen and Eberhard, 2014 , p. 50) ) such that

 

 
 | 
 δ ​ p i = J i ​ δ ​ q . \delta p_{i}=J_{i}\delta q. | 
 | 
 (19) | 
 

 By inserting ( 19 ) into ( 13 ) one obtains

 

 
 | 
 ∑ i = 1 N b δ ​ q 𝖳 ​ J i 𝖳 ​ F E , i = δ ​ q 𝖳 ​ J 𝖳 ​ F E = 0 , \sum_{i=1}^{N_{\text{b}}}\delta q^{\mathrm{\sf T}}J^{\mathrm{\sf T}}_{i}F_{\text{E},i}=\delta q^{\mathrm{\sf T}}J^{\mathrm{\sf T}}F_{\text{E}}=0, | 
 | 
 (20) | 
 

 with J = [ J 1 𝖳 , J 2 𝖳 , … ​ J N b 𝖳 ] 𝖳 J=[J_{1}^{\mathrm{\sf T}},J_{2}^{\mathrm{\sf T}},\dots J_{N_{\text{b}}}^{\mathrm{\sf T}}]^{\mathrm{\sf T}} . As ( 20 ) must hold for an arbitrary δ ​ q \delta q , we also have J 𝖳 ​ F E = 0 J^{\mathrm{\sf T}}F_{\text{E}}=0 such that

 

 
 | 
 F E ∈ 𝖭 ⁡ ( J 𝖳 ) . F_{\text{E}}\in\mathsf{N}(J^{\mathrm{\sf T}}). | 
 | 
 (21) | 
 

 Further, with ( 21 ) and ℝ 6 ​ N b = 𝖱 ⁡ ( J ) ⊕ 𝖭 ⁡ ( J 𝖳 ) \mathbb{R}^{6N_{\text{b}}}=\mathsf{R}(J)\oplus\mathsf{N}(J^{\mathrm{\sf T}}) (cf. Beard (2002) ), from ( 13 ) follows

 

 
 | 
 δ ​ p ∈ 𝖱 ⁡ ( J ) . \delta p\in\mathsf{R}(J). | 
 | 
 (22) | 
 

 
 
 
 Equations of motion with explicit constraints 

 
 By transformation of ( 14 ) into generalized coordinate form using ( 15 ), ( 16 ), ( 17 ), and ( 19 ), one obtains

 

 
 | 
 δ ​ q 𝖳 ​ ∑ i = 1 N b J i 𝖳 ​ ( M ~ i ​ ( J i ​ q ¨ + J ~ i ) − F C , i − F e , i ) = 0 . \delta q^{\mathrm{\sf T}}\sum_{i=1}^{N_{\text{b}}}J^{\mathrm{\sf T}}_{i}\left(\tilde{M}_{i}(J_{i}\ddot{q}+\tilde{J}_{i})-F_{\text{C},i}-F_{\text{e},i}\right)=0. | 
 | 
 (23) | 
 

 As ( 23 ) must hold for an arbitrary δ ​ q \delta q , the local EOM of the i i -th rigid body are obtained as
 M i ​ q ¨ = Q C , i + Q e , i , M_{i}\ddot{q}=Q_{C,i}+Q_{e,i}, 
with M i ​ ( q , t ) = J i 𝖳 ​ M ~ i ​ J i M_{i}(q,t)=J_{i}^{\mathrm{\sf T}}\tilde{M}_{i}J_{i} , Q C , i ​ ( q , q ˙ , t ) = J i 𝖳 ​ ( F C , i ​ ( q , q ˙ , t ) − M ~ i ​ J ~ i ) Q_{\text{C},i}(q,\dot{q},t)=J_{i}^{\mathrm{\sf T}}(F_{\text{C},i}(q,\dot{q},t)-\tilde{M}_{i}\tilde{J}_{i}) , and Q e , i ​ ( q , q ˙ , t ) = J i 𝖳 ​ F e , i Q_{\text{e},i}(q,\dot{q},t)=J_{i}^{\mathrm{\sf T}}F_{\text{e},i} ( Schiehlen and Eberhard, 2014 , p. 100) . In return, one obtains the EOM of the multibody system as

 

 
 | 
 M ​ q ¨ = Q , M\ddot{q}=Q, | 
 | 
 (24) | 
 

 with Q = Q C + Q e Q=Q_{\text{C}}+Q_{\text{e}} , the generalized inertia matrix M = ∑ i = 1 N b M i M=\sum_{i=1}^{N_{\text{b}}}M_{i} , the generalized bias force Q C = ∑ i = 1 N b Q C , i Q_{\text{C}}=\sum_{i=1}^{N_{\text{b}}}Q_{\text{C},i} , and the generalized impressed force Q e = ∑ i = 1 N b Q e , i Q_{\text{e}}=\sum_{i=1}^{N_{\text{b}}}Q_{\text{e},i} ( Schiehlen and Eberhard, 2014 , p. 107) . As pointed out by Featherstone (2008, p, 40) , if the impressed forces Q e = − Q C Q_{\text{e}}=-Q_{\text{C}} then the system’s acceleration amounts to null.
We assume that the generalized coordinates are chosen such that M ⁡ ( q , t ) M(q,t) and its inverse M − 1 ​ ( q , t ) M^{-1}(q,t) are symmetric and positive-definite, writing q ¨ ⊤ ​ M ​ ( q , t ) ​ q ¨ 0 \ddot{q}^{\top}M(q,t)\ddot{q} 0 . Then, M − 1 ​ ( q , t ) M^{-1}(q,t) maps the n q n_{q} -dimensional generalised force space to the n q n_{q} -dimensional generalized acceleration space.
As the impressed forces possess different properties depending on their source of origin, we further split up the force vectors as

 

 
 | 
 Q e = Q G + Q D ​  with  ​ Q D = Q d + Q u . \displaystyle Q_{\text{e}}=Q_{G}+Q_{D}\ \text{ with }\ Q_{\text{D}}=Q_{\text{d}}+Q_{\text{u}}. | 
 | 
 (25) | 
 

 The conservative force Q G ​ ( q , q ˙ ) Q_{\text{G}}(q,\dot{q}) arises from a physical potential V ⁡ ( q , q ˙ ) V(q,\dot{q}) , such as a spring or a gravitational force. The non-conservative force vector Q D ​ ( q , q ˙ , t ) Q_{\text{D}}(q,\dot{q},t) denotes forces that can change the system’s total energy via external forces Q d ​ ( q , q ˙ , t ) Q_{\text{d}}(q,\dot{q},t) and the actuation forces Q u ​ ( q , q ˙ , t ) Q_{\text{u}}(q,\dot{q},t) , respectively. Note that Q d Q_{\text{d}} is often a dissipative force, that is, it can only reduce the system’s total energy. If the system is fully actuated, one obtains the system’s inverse dynamics by rearranging ( 24 ) as

 

 
 | 
 Q u = M ​ q ¨ − Q C − Q G − Q d . Q_{\text{u}}=M\ddot{q}-Q_{\text{C}}-Q_{\text{G}}-Q_{\text{d}}. | 
 | 
 (26) | 
 

 Note that there exist computationally efficient recursive algorithms for the derivation of the EOM of many types of robotic systems. An introduction to recursive rigid-body algorithms is given in Featherstone (2008) .

 
 
 
 

### 2.2 Euler-Lagrange equations: Structural knowledge for energy conservation

 
 In this section, we detail how some of the terms in ( 25 ) can be obtained in terms of potential functions.
The resulting equations can be used to include energy conservation into an analytical structured model as discussed in Section 3.4.1 . A detailed introduction to the Lagrange equations is given in ( Layton, 2012 , p. 68) and more specifically for robot arms in ( Siciliano et al., 2010 , p. 247) .

 
 
 The Lagrangian function ℒ ⁡ ( q , q ˙ ) \mathcal{L}(q,\dot{q}) can be obtained as the difference between the kinetic energy T ⁡ ( q , q ˙ ) T(q,\dot{q}) and the potential energy V ⁡ ( q , q ˙ ) V(q,\dot{q}) , writing

 

 
 | 
 ℒ = T − V = 1 2 ​ q ˙ ⊤ ​ M ​ q ˙ − V . \mathcal{L}=T-V=\frac{1}{2}\dot{q}^{\top}M\dot{q}-V. | 
 | 
 (27) | 
 

 In return, the Euler-Lagrange equation of a rigid-body system’s i i -th dimension can be derived via the calculus of variations as

 

 
 | 
 d d ​ t ​ ∂ ℒ ∂ q ˙ i − ∂ ℒ ∂ q i = 0 . \frac{d}{dt}\frac{\partial\mathcal{L}}{\partial\dot{q}_{i}}-\frac{\partial\mathcal{L}}{\partial q_{i}}=0. | 
 | 
 (28) | 
 

 Note that the Lagrangian differs from the system’s total energy E = T + V E=T+V . The Lagrangian is used to mathematically express that a conservative system must take a path of stationary action via ( 28 ) such that E E remains constant. Rewriting ( 28 ) in vector notation and adding the non-conservative forces Q D Q_{\text{D}} yields

 

 
 | 
 d d ​ t ​ ∇ q ˙ ℒ − ∇ q ℒ = Q D , \frac{d}{dt}\nabla_{\dot{q}}\mathcal{L}-\nabla_{q}\mathcal{L}=Q_{\text{D}}, | 
 | 
 (29) | 
 

 with ( ∇ q ˙ ) i = ∂ ∂ q ˙ i (\nabla_{\dot{q}})_{i}=\frac{\partial}{\partial\dot{q}_{i}} .
One can apply the chain rule to expand the time-derivative of the Lagrangian’s partial derivative as

 

 
 | 
 d d ​ t ​ ∇ q ˙ ℒ = ( ∇ q ˙ ∇ q ˙ ⊤ ​ ℒ ) ​ q ¨ + ( ∇ q ∇ q ˙ ⊤ ​ ℒ ) ​ q ˙ , \frac{d}{dt}\nabla_{\dot{q}}\mathcal{L}=\left(\nabla_{\dot{q}}\nabla_{\dot{q}}^{\top}\mathcal{L}\right)\ddot{q}+\left(\nabla_{q}\nabla_{\dot{q}}^{\top}\mathcal{L}\right)\dot{q}, | 
 | 
 (30) | 
 

 with the n × n n\times n matrix ( ∇ q ∇ q ˙ ⊤ ​ ℒ ) i ​ j = ∂ 2 ℒ ∂ q i ​ ∂ q ˙ j \left(\nabla_{q}\nabla_{\dot{q}}^{\top}\mathcal{L}\right)_{ij}=\frac{\partial^{2}\mathcal{L}}{\partial q_{i}\partial\dot{q}_{j}} . Therefore, one obtains ( 24 ) in terms of the Lagrangian as

 

 
 | 
 q ¨ = ( ∇ q ˙ ∇ q ˙ ⊤ ​ ℒ ) − 1 ​ ( − ( ∇ q ∇ q ˙ ⊤ ​ ℒ ) ​ q ˙ + ∇ q ℒ + Q D ) . \ddot{q}=\left(\nabla_{\dot{q}}\nabla_{\dot{q}}^{\top}\mathcal{L}\right)^{-1}\left(-\left(\nabla_{q}\nabla_{\dot{q}}^{\top}\mathcal{L}\right)\dot{q}+\nabla_{q}\mathcal{L}+Q_{\text{D}}\right). | 
 | 
 (31) | 
 

 In what follows, it is assumed that V ⁡ ( q ) V(q) only depends on q q to keep the expressions concise. The previous equation can also be written in terms of M = ∇ q ˙ ∇ q ˙ ⊤ ​ ℒ M=\nabla_{\dot{q}}\nabla_{\dot{q}}^{\top}\mathcal{L} and Q G = − ∇ q V ​ ( q ) Q_{\text{G}}=-\nabla_{q}V(q) to yield an alternative description of the forward dynamics as

 

 
 | 
 q ¨ = M − 1 ​ ( − ∇ q ( q ˙ ⊤ ​ M ) ​ q ˙ + 1 2 ​ ∇ q ( q ˙ T ​ M ​ q ˙ ) − ∇ q V + Q D ) . \ddot{q}=M^{-1}\left(-\nabla_{q}(\dot{q}^{\top}M)\dot{q}+\frac{1}{2}\nabla_{q}\left(\dot{q}^{T}M\dot{q}\right)-\nabla_{q}V+Q_{\text{D}}\right). | 
 | 
 (32) | 
 

 The above equations yield parametrizations of the fictitious force as

 

 
 | 
 Q C \displaystyle Q_{\text{C}} | 
 = − ( ∇ q ∇ q ˙ ⊤ ​ T ) ​ q ˙ + ∇ q T = − ∇ q ( q ˙ ⊤ ​ M ) ​ q ˙ + 1 2 ​ ∇ q ( q ˙ T ​ M ​ q ˙ ) . \displaystyle=-\left(\nabla_{q}\nabla_{\dot{q}}^{\top}T\right)\dot{q}+\nabla_{q}T=-\nabla_{q}(\dot{q}^{\top}M)\dot{q}+\frac{1}{2}\nabla_{q}\left(\dot{q}^{T}M\dot{q}\right). | 
 | 
 (33) | 
 

 If the system is fully actuated, one obtains expressions for the system’s inverse dynamics by simply rearranging both ( 31 ) and ( 32 ), reading

 

 
 | 
 Q u \displaystyle Q_{\text{u}} | 
 = ( ∇ q ˙ ∇ q ˙ ⊤ ​ ℒ ) ​ q ¨ + ( ∇ q ∇ q ˙ ⊤ ​ ℒ ) ​ q ˙ − ∇ q ℒ − Q d , \displaystyle=\left(\nabla_{\dot{q}}\nabla_{\dot{q}}^{\top}\mathcal{L}\right)\ddot{q}+\left(\nabla_{q}\nabla_{\dot{q}}^{\top}\mathcal{L}\right)\dot{q}-\nabla_{q}\mathcal{L}-Q_{\text{d}}, | 
 | 
 (34) | 
 
 
 | 
 | 
 = M ​ q ¨ + ∇ q ( q ˙ ⊤ ​ M ) ​ q ˙ − 1 2 ​ ∇ q ( q ˙ T ​ M ​ q ˙ ) + ∇ q V − Q d . \displaystyle=M\ddot{q}+\nabla_{q}(\dot{q}^{\top}M)\dot{q}-\frac{1}{2}\nabla_{q}\left(\dot{q}^{T}M\dot{q}\right)+\nabla_{q}V-Q_{\text{d}}. | 
 | 
 (35) | 
 

 
 
 
 
 
 
 
 
 Figure 2 : The feet of a quadruped (Left: Open Dynamic Robot, used with courtesy of Grimminger et al. (2020) ) are subject to contact forces. A pneumatic-actuated leg (Right: RH5 leg, adapted with courtesy of Kumar (2019) ) has several kinematic loops that can be modeled via implicit constraints which induce constraint forces at the cut joints. 
 
 
 

### 2.3 Implicit constraint equations: Structural knowledge on the direction of motion

 
 Section 2.1 outlined how constraint forces F E , i F_{\text{E},i} can be excluded from the EOM by a suitable choice of generalized coordinates q q which yield explicit holonomic constraints.
In this section, we emphasize the utility of implicit constraints as a complement to using explicit constraints and an additional source of structural knowledge.

 
 
 Implicit constraints allow to include constraint forces into the EOM. Implicit constraints can be particularly useful for non-holonomic systems, system’s with kinematic loops, and systems that are subject to inequality constraints. For example, when the foot of a quadruped (Figure 2, left) presses onto a surface, the surface applies a reaction force that reduces the DOF of the system. The force arising from the contact can be modelled using an implicit holonomic constraint. In return, one obtains an analytical expression for the respective constraint force which can be straightforwardly removed from the EOMs if the constraint is inactive.
As another example, a hydraulic-actuated robot leg (Figure 2, right) introduces kinematic loops
that can be modeled via the addition of implicit constraints.
We base the following discussion on the EOM as in ( 24 ). However, we broaden the problem setting compared to Section 2.1 by making the following assumptions:

 
 • 
 
 The system of rigid bodies is subject to constraint forces that now impose n E + n I n_{\text{E}}+n_{\text{I}} independent constraint equations with n I n_{\text{I}} denoting the number of implicit constraints. In return, the system has ( n q − n I ) (n_{q}-n_{\text{I}}) DOF ( Layton, 2012 , p. 43) .

 

 • 
 
 Of these constraint forces, n E n_{\text{E}} are eliminated from the EOM through use of explicit holonomic constraints and a suitable choice of generalized coordinates q ∈ ℝ n q q\in\mathbb{R}^{n_{q}} as discussed in Section 2.1 .

 

 
 These assumptions cover by no means all possible descriptions of the EOM of a rigid-body system. For example, constraint equations can be redundant which in return requires a more extensive treatment on the connection between forces and the vector spaces that constraint-related matrices span. A more general discussion is given in Featherstone (2008) as well as Koganti and Udwadia (2016) . Nevertheless, the following discussion illustrates the interplay between many of the building blocks that are frequently encountered when deriving dynamics equations and which can yield structure to a regression model as detailed in Section 3.4.3 .

 
 
 Implicit constraints can be expressed algebraically as c ⁡ ( q , t ) = 0 c(q,t)=0 if they are holonomic, or more generally as c ⁡ ( q , q ˙ , t ) = 0 c(q,\dot{q},t)=0 if they are non-holonomic. We assume that differentiation with respect to time yields implicit constraint equations on the system’s position, velocity, and acceleration as detailed below in ( 36 ) and ( 37 ).

 

 
 | 
 | 
 position | 
 | 
 velocity | 
 | 
 acceleration | 
 | 
 
 
 | 
 holonomic:    | 
 c ⁡ ( q , t ) = 0 → d / d ​ t \displaystyle c(q,t)=0\hskip 14.22636pt\overset{\text{d}/\text{d}t}{\rightarrow}\hskip 14.22636pt | 
 | 
 A ~ ​ ( q , t ) ​ q ˙ = b ~ ​ ( q , t ) → d / d ​ t \displaystyle\tilde{A}(q,t)\dot{q}=\tilde{b}(q,t)\hskip 14.22636pt\overset{\text{d}/\text{d}t}{\rightarrow}\hskip 14.22636pt | 
 | 
 A ⁡ ( q , t ) ​ q ¨ = b ⁡ ( q , q ˙ , t ) \displaystyle A(q,t)\ddot{q}=b(q,\dot{q},t) | 
 | 
 (36) | 
 
 
 | 
 nonholonomic:    | 
 | 
 | 
 c ⁡ ( q , q ˙ , t ) = 0 → d / d ​ t \displaystyle c(q,\dot{q},t)=0\hskip 28.45274pt\overset{\text{d}/\text{d}t}{\rightarrow} | 
 | 
 A ⁡ ( q , q ˙ , t ) ​ q ¨ = b ⁡ ( q , q ˙ , t ) \displaystyle A(q,\dot{q},t)\ddot{q}=b(q,\dot{q},t) | 
 | 
 (37) | 
 

 The terms such as A ⁡ ( q , t ) = A ~ ​ ( q , t ) = ∂ c ⁡ ( q , t ) ∂ q A(q,t)=\tilde{A}(q,t)=\frac{\partial c(q,t)}{\partial q} and b ⁡ ( q , q ˙ , t ) = ∂ c ⁡ ( q , q ˙ , t ) ∂ t b(q,\dot{q},t)=\frac{\partial c(q,\dot{q},t)}{\partial t} denote partial derivatives with respect to q q or t t , respectively. Comparing ( 36 ) and ( 37 ), one sees that holonomic and non-holonomic constraints can be denoted jointly on the acceleration level as

 

 
 | 
 A ​ q ¨ = b , A\ddot{q}=b, | 
 | 
 (38) | 
 

 where we assume that only n I n_{\text{I}} constraints are expressed in implicit form such that A ∈ ℝ n I × n q A\in\mathbb{R}^{n_{\text{I}}\times n_{q}} and b ∈ ℝ n I b\in\mathbb{R}^{n_{\text{I}}} .

 
 
 Remark 2.1 . 
 
 Minimal state description of non-holonomic systems

 
To obtain a minimal state description, a non-holonomic system requires more position variables than velocity variables. For example, a unicyclist riding on a plane can reach any potential position on the plane. Yet, as an ideally rolling wheel cannot slide sideways, the wheel’s translational Cartesian velocities at the contact point with the plane can be described in terms of a single velocity variable pointing along a position-dependent axis. However, in this work, we limit the discussion on the description of the system’s EOM using as many velocity variables q ˙ \dot{q} as position variables q q . An introduction to the derivation of EOMs of non-holonomic systems with a minimal state representation is given in ( Featherstone, 2008 , p. 41) and ( Schiehlen and Eberhard, 2014 , p. 114) .
 

 
 
 
 Reformulation of virtual displacements 

 
 To be able to eliminate constraint forces using the D’Alembert-Lagrange principle for multibody systems ( 13 ), one must define what constitutes a virtual displacement vector δ ​ q \delta q . Oftentimes, the non-holonomic implicit constraints are assumed Pfaffian taking the form A ~ ​ ( q , t ) ​ q ˙ = b ~ ​ ( q , t ) \tilde{A}(q,t)\dot{q}=\tilde{b}(q,t) with A ~ ​ ( q , t ) \tilde{A}(q,t) and b ~ ​ ( q , t ) \tilde{b}(q,t) not being partial derivatives of a position-level constraint. In this case, for holonomic and Pfaffian non-holonomic constrained systems the virtual displacement is often defined as the infinitesimal vector δ ​ q \delta q that fulfills
 A ~ ​ δ ​ q = 0 \tilde{A}\delta q=0 (cf. ( Udwadia and Kalaba, 2007 , p. 131) and ( Layton, 2012 , p. 50) ). However, this definition is not applicable for non-holonomic constraints of the form in ( 37 ). In this case, many works resort to virtual velocity vectors which denote the infinitesimal change in the velocity that agrees with the constraints while not varying position and time ( Schiehlen and Eberhard, 2014 , p. 55) . In return, one can turn to Jourdain’s principle of virtual power to discuss the effect of non-holonomic constraint forces on the EOM. However, as our previous discussion of explicit constraints was centered around the concept of virtual work, we instead define virtual displacements as proposed by Udwadia et al. (1997) as an arbitrary infinitesimal vector δ ​ q ∈ ℝ n q \delta q\in\mathbb{R}^{n_{q}} fulfilling the equation

 

 
 | 
 A ​ δ ​ q = 0 . A\delta q=0. | 
 | 
 (39) | 
 

 Unlike Section 2.1 , in which δ ​ q \delta q denoted any infinitesimal vector inside ℝ ( 6 ​ N b − n E ) \mathbb{R}^{(6N_{\text{b}}-n_{\text{E}})} ,
( 39 ) implies that in the presence of the additional implicit constraint forces, the virtual displacement denotes any infinitesimal vector δ ​ q ∈ 𝖭 ⁡ ( A ) \delta q\in\mathsf{N}(A) .
Note that the above definition of δ ​ q \delta q is similar to the definition of virtual velocities as found in other textbooks (cf. ( Woernle, , p. 202) ).

 
 
 Remark 2.2 . 
 
 Outline of the derivation of ( 39 )

 
As shown by Udwadia et al. (1997) , ( 39 ) follows from the Taylor expansion about time t t of a displacement q ⁡ ( t + d ​ t ) q(t+dt) as 

 

 
 | 
 q ⁡ ( t + d ​ t ) = q ⁡ ( t ) + ( d ​ t ) ​ q ˙ ​ ( t ) + ( d ​ t ) 2 2 ​ q ¨ ​ ( t ) + 𝒪 ⁡ ( d ​ t 3 ) , q(t+dt)=q(t)+(dt)\dot{q}(t)+\frac{(dt)^{2}}{2}\ddot{q}(t)+\mathcal{O}(dt^{3}), | 
 | 
 (40) | 
 

 with d ​ t dt being an infinitesimal quantity.
With ( 40 ) one defines the virtual displacement at time (t+dt) as the difference between a possible displacement q b ​ ( t + d ​ t ) = vec ​ { q , q ˙ , q ¨ b } q^{b}(t+dt)=\text{vec}\{q,\dot{q},\ddot{q}^{b}\} and the actual displacement q a ​ ( t + d ​ t ) = vec ​ { q , q ˙ , q ¨ a } q^{a}(t+dt)=\text{vec}\{q,\dot{q},\ddot{q}^{a}\} such that 

 

 
 | 
 δ ​ q = q b ​ ( t + d ​ t ) − q a ​ ( t + d ​ t ) = ( d ​ t ) 2 2 ​ [ q ¨ b ​ ( t ) − q ¨ a ​ ( t ) + 𝒪 ⁡ ( d ​ t ) ] . \delta q=q^{b}(t+dt)-q^{a}(t+dt)=\frac{(dt)^{2}}{2}[\ddot{q}^{b}(t)-\ddot{q}^{a}(t)+\mathcal{O}(dt)]. | 
 | 
 (41) | 
 

 As every possible motion must fulfill ( 38 ), we can insert q a ​ ( t + d ​ t ) q^{a}(t+dt) and q b ​ ( t + d ​ t ) q^{b}(t+dt) into ( 38 ) and take the difference between the expressions to obtain 

 

 
 | 
 A ⁡ ( q ⁡ ( t ) , q ˙ ​ ( t ) , t ) ​ [ q ¨ b ​ ( t ) − q ¨ a ​ ( t ) ] = 0 , A(q(t),\dot{q}(t),t)[\ddot{q}^{b}(t)-\ddot{q}^{a}(t)]=0, | 
 | 
 (42) | 
 

 which with ( 40 ) can be shown for d ​ t → 0 dt\rightarrow 0 to yield ( 39 ).
 

 
 
 
 
 D’Alembert’s principle and implicit constraint forces 

 
 The implicit constraints are the consequence of constraint forces and torques F I , i = vec ​ { f I , i , τ I , i } F_{\text{I},i}=\text{vec}\{f_{\text{I},i},\tau_{\text{I},i}\} acting onto each body. In turn, the D’Alembert-Lagrange principle for the multibody system ( 13 ) with the explicit transformation to generalized coordinates ( 19 ) as well as ∑ i = 1 N b J i 𝖳 ​ F E , i = 0 \sum_{i=1}^{N_{\text{b}}}J_{i}^{\mathrm{\sf T}}F_{\text{E},i}=0 due to the specific choice of q q , reads

 

 
 | 
 ∑ i = 1 N b δ ​ p i 𝖳 ​ ( F E , i + F I , i ) = δ ​ q 𝖳 ​ ∑ i = 1 N b J i 𝖳 ​ ( F E , i + F I , i ) = δ ​ q 𝖳 ​ Q I = 0 , \sum_{i=1}^{N_{\text{b}}}\delta p_{i}^{\mathrm{\sf T}}(F_{\text{E},i}+F_{\text{I},i})=\delta q^{\mathrm{\sf T}}\sum_{i=1}^{N_{\text{b}}}J_{i}^{\mathrm{\sf T}}(F_{\text{E},i}+F_{\text{I},i})=\delta q^{\mathrm{\sf T}}Q_{\text{I}}=0, | 
 | 
 (43) | 
 

 with the generalized implicit constraint forces Q I = ∑ i = 1 N b J i 𝖳 ​ F I , i Q_{I}=\sum_{i=1}^{N_{\text{b}}}J_{i}^{\mathrm{\sf T}}F_{\text{I},i} .
As ( 39 ) requires δ ​ q ∈ 𝖭 ⁡ ( A ) \delta q\in\mathsf{N}(A) , ( 43 ) yields that

 

 
 | 
 Q I ∈ 𝖱 ⁡ ( A 𝖳 ) . Q_{\text{I}}\in\mathsf{R}(A^{\mathrm{\sf T}}). | 
 | 
 (44) | 
 

 In turn, ( 44 ) motivates to parametrize the ideal constraint force in terms of a Lagrange multiplier vector λ ⁡ ( q , q ˙ , t ) ∈ ℝ n I \lambda(q,\dot{q},t)\in\mathbb{R}^{n_{\text{I}}} as

 

 
 | 
 Q I = A ⊤ ​ λ . Q_{\text{I}}=A^{\top}\lambda. | 
 | 
 (45) | 
 

 By inserting ( 45 ) to the EOM in ( 24 ) or ( 29 ) one obtains an index-3 system of differential algebraic equations (cf. ( Schiehlen and Eberhard, 2014 , p. 105) ) as

 

 
 | 
 [ M − A 𝖳 A 𝟎 ] ​ [ q ¨ λ ] = [ Q b ] . \begin{bmatrix}M -A^{\mathrm{\sf T}}\\
A \bm{0}\end{bmatrix}\begin{bmatrix}\ddot{q}\\
\lambda\end{bmatrix}=\begin{bmatrix}Q\\
b\end{bmatrix}. | 
 | 
 (46) | 
 

 
 
 
 
 ∈ 
 
 ⁢ 
 δ 
 p 
 
 
 R 
 
 
 
 ( 
 J 
 ) 
 
 
 
 
 
 E 
 
 
 ∈ 
 
 
 F 
 E 
 
 
 N 
 
 
 
 ( 
 
 
 J 
 T 
 
 ) 
 
 
 
 
 
 
 
 ⊕ 
 
 
 Figure 3 : Under the assumption that the explicit and implicit constraints are independent,
the constraint forces and virtual displacements must lie in the null and range spaces of J , J 𝖳 , A , J,J^{\mathrm{\sf T}},A, and A 𝖳 A^{\mathrm{\sf T}} as illustrated above. In the above diagram being inspired by Beard (2002) , vertical lines denote vector spaces. One can move a vector between two of these spaces by multiplication from the left with the constraint-related matrix transformation indicated on the respective arrow. 
 
 
 
 Non-ideal constraint forces 

 
 Oftentimes, constraint forces are not ideal and do virtual work. In such a case, Udwadia and Kalaba (2002) proposed to divide the constraint force into an ideal part Q I Q_{\text{I}} and a non-ideal part Q d,I Q_{\text{d,I}} doing virtual work. The forces Q d,I Q_{\text{d,I}} are dissipative and therefore can be seen as being part of Q d Q_{\text{d}} as long as the respective constraints are active. Oftentimes, Q I Q_{\text{I}} causes Q d,I Q_{\text{d,I}} . For example, the friction force Q d,I Q_{\text{d,I}} between a robot’s foot and a surface is caused by the normal force Q I Q_{\text{I}} that the foot applies onto the surface. As emphasized by Udwadia and Kalaba (2000) , the part of any arbitrary force which is doing virtual work, Q ′ Q^{\prime} , writing δ ​ q 𝖳 ​ Q ′ ≠ 0 \delta q^{\mathrm{\sf T}}Q^{\prime}\neq 0 , must have the same direction as δ ​ q \delta q and hence

 

 
 | 
 Q ′ ∈ 𝖭 ⁡ ( A ) . Q^{\prime}\in\mathsf{N}(A). | 
 | 
 (47) | 
 

 
 
 
 ODE form of the EOM for implicitly constrained systems 

 
 Many different dynamic formulations for Q I Q_{\text{I}} in terms of implicit constraints and the unconstrained dynamics equations have been proposed in literature. For example, Aghili (2005) details several dynamics equations of implicitly holonomic constrained systems. Alternatively, for the case of independent implicit constraints and a positive-definite mass matrix, applying the matrix inversion lemma on ( 46 ) yields the description of the EOM in ODE form

 

 
 | 
 q ¨ \displaystyle\ddot{q} | 
 OPEN = M − 1 ​ ( Q + A T ​ ( A ​ M − 1 ​ A T ) − 1 ​ ( b − A ​ M − 1 ​ Q CLOSE ⏟ Q I ) ) . \displaystyle=M^{-1}\Big(Q+\underbrace{A^{T}(AM^{-1}A^{T})^{-1}(b-AM^{-1}Q}_{\text{\normalsize$Q_{\text{I}}$}})\Big). | 
 | 
 (48) | 
 

 Udwadia and Phohomsiri (2006) showed that ( 48 ) is a special case of the so called Udwadia-Kalaba equation, which yields EOM for systems with positive semi -definite inertia matrix and dependent implicit constraint equations. An introduction to the usage of the Udwadia-Kalaba equation for robot control is given in ( Peters et al., 2008 ) .
Note that Q I = A T ​ ( A ​ M − 1 ​ A T ) − 1 ​ ( b − A ​ M − 1 ​ Q ) Q_{\text{I}}=A^{T}(AM^{-1}A^{T})^{-1}(b-AM^{-1}Q) yields an explicit form for the Lagrange multiplier parametrization in terms of the constraining equation ( 38 ).
Rearranging ( 48 ) yields

 

 
 | 
 q ¨ \displaystyle\ddot{q} | 
 = M − 1 ​ ( P ​ Q + Q b ) , \displaystyle=M^{-1}\left(PQ+Q_{b}\right), | 
 | 
 (49) | 
 

 with the weighted constraint projection P ⁡ ( q , q ˙ , t ) = I n − A T ​ ( A ​ M − 1 ​ A T ) − 1 ​ A ​ M − 1 P(q,\dot{q},t)=I_{n}-A^{T}(AM^{-1}A^{T})^{-1}AM^{-1} and the constraining force Q b = A T ​ ( A ​ M − 1 ​ A T ) − 1 ​ b Q_{b}=A^{T}(AM^{-1}A^{T})^{-1}b .
As further discussed by Udwadia and Kalaba (1992) , Gauss Gauß (1829) observed that the acceleration caused by an ideal constraint force, q ¨ I = M − 1 ​ Q I \ddot{q}_{I}=M^{-1}Q_{\text{I}} , minimizes the quadratic functional

 

 
 | 
 G ⁡ ( q , q ˙ , t ) = q ¨ I ⊤ ​ M ​ q ¨ I . G(q,\dot{q},t)=\ddot{q}_{I}^{\top}M\ddot{q}_{I}. | 
 | 
 (50) | 
 

 The above equation, being referred to as Gauss’ principle of least constraint , uniquely defines the length of the vector q ¨ I \ddot{q}_{\text{I}} in ( 48 ). In turn, the matrix P P is a weighted projection from the n q n_{q} -dimensional force space to 𝖭 ⁡ ( A ) \mathsf{N}(A) such that Gauss’ principle is fulfilled, where 𝖭 ⁡ ( A ) \mathsf{N}(A) is a ( n q − n I ) (n_{q}-n_{I}) -dimensional manifold inside ℝ n q \mathbb{R}^{n_{q}} .

 
 
 
 Vector spaces and constrained dynamics 

 
 The assumption that the explicit and implicit constraint equations are independent requires J 𝖳 J^{\mathrm{\sf T}} and A A to have full row-rank at every non-zero state-vector { q , q ˙ , t } \{q,\dot{q},t\} that respects the constraints. Under these assumptions, the insights on the direction of F E F_{\text{E}} in ( 21 ), δ ​ p \delta p in ( 22 ), δ ​ q \delta q in ( 39 ), and Q I Q_{\text{I}} in ( 44 ) can be summarized in a single diagram as depicted in Figure 3 . Figure 3 emphasizes that at an admissible point in the system’s state-space { q , q ˙ , t } \{q,\dot{q},t\} , the virtual displacement vectors as well as constraint forces are bound to lie in spaces spanned by the constraint-related matrices J J and A A . In addition, the inertia matrix M M defines the mapping from the acceleration space to the force space. Gauss’ principle as in ( 50 ) emphasizes that M M also determines an analytical expression for the implicit constraint force Q I Q_{\text{I}} as in ( 48 ).

 
 
 
 
 

## 3 Analytical structured learning

 
 In the following section, we propose a unified view on analytical structured modeling. For this, we leverage that the dynamics descriptions presented in Section 2 consist of sums of latent vector-valued functions ( e.g. , forces) that are multiplied with matrix-valued functions ( e.g. , the inertia matrix). We then show that this perspective enables us to decompose and discuss the error functions inherent in an analytical model.
As data-driven models are a substantial part of analytical structured modeling, we then proceed by giving a brief introduction to common data-driven dynamics models and analytical models in parametric network form. Finally, we discuss selected literature on ARM in Section 3.3 and ALM in Section 3.4 .

 
 

### 3.1 A unified view on analytical model errors

 
 To understand the pros and cons of different analytical structured models, we require a thorough understanding of the cause of the analytical model errors. As shown in Section 2 , the forward dynamics can be expressed either in terms of force vectors ( 24 ), additional potential functions ( 29 ), or with additional implicit constraints ( 49 ). All of these formulations of the EOM consist of a sum of generalized forces ( Q + Q I ) (Q+Q_{\text{I}}) that are multiplied by M − 1 ​ ( q , θ A ) M^{-1}(q,\theta_{\text{A}}) . In comparison, the inverse dynamics formulations in ( 35 ) or ( 34 ) form sums of latent functions that are transformed by M ⁡ ( q , θ A ) M(q,\theta_{\text{A}}) or simply identity matrices. To unify the discussion, we therefore assume that rigid-body dynamics equations can be written as a sum of latent vector-valued functions f ^ i ​ ( x , θ A ) \hat{f}_{i}(x,\theta_{\text{A}}) that are transformed by latent matrix-valued functions 𝒞 i ​ ( x , θ A ) \mathcal{C}_{i}(x;\theta_{\text{A}}) , such that the analytical model becomes

 

 
 | 
 f ^ A ​ ( x , θ A ) = ∑ i 𝒞 i ​ f ^ i . \hat{f}_{\text{A}}(x;\theta_{\text{A}})=\sum_{i}\mathcal{C}_{i}\hat{f}_{i}. | 
 | 
 (51) | 
 

 As analytical models of a mechanical system dynamics often erroneous, one can substitute ( 51 ) into ( 8 ) to obtain an expression of the system’s dynamics in terms of the analytical model and multiple error functions, writing

 

 
 | 
 f ⁡ ( x ) \displaystyle f(x) | 
 = f ^ A + ϵ A = f ^ A + ϵ E + ϵ R , \displaystyle=\hat{f}_{\text{A}}+\epsilon_{\text{A}}=\hat{f}_{\text{A}}+\epsilon_{\text{E}}+\epsilon_{\text{R}}, | 
 | 
 (52) | 
 
 
 | 
 | 
 = ϵ R + ∑ i ( 𝒞 i + ϵ 𝒞 , i ) ​ ( f ^ i + ϵ f ^ , i ) , \displaystyle=\epsilon_{\text{R}}+\sum_{i}\left(\mathcal{C}_{i}+\epsilon_{\mathcal{C},i}\right)\left(\hat{f}_{i}+\epsilon_{\hat{f},i}\right), | 
 | 
 (53) | 
 

 with the vector-valued error functions ϵ R ​ ( x ) \epsilon_{\text{R}}(x) , ϵ E ​ ( x ) \epsilon_{\text{E}}(x) , and ϵ f ^ , i ​ ( x ) \epsilon_{\hat{f},i}(x) , as well as the matrix-valued error function ϵ 𝒞 , i ​ ( x ) \epsilon_{\mathcal{C},i}(x) .

 
 
 Figure 4 : Illustration of the errors of a rigid-body dynamics model. 
 
 
 Ideal models and latent error functions 

 
 To shed further light on the error functions in ( 52 ) and ( 53 ), we define the ideal rigid-body dynamics model 

 

 
 | 
 f ^ A ∗ ​ ( x , θ A ∗ ) = f ^ A + ϵ E = ∑ i 𝒞 i ∗ ​ f ^ i ∗ , \hat{f}_{\text{A}}^{*}(x;\theta_{\text{A}}^{*})=\hat{f}_{\text{A}}+\epsilon_{\text{E}}=\sum_{i}\mathcal{C}_{i}^{*}\hat{f}_{i}^{*}, | 
 | 
 (54) | 
 

 which corresponds to the analytical model in which the asterisk symbol ( ∗ ) (^{*}) denotes adequate choices of 𝒞 i \mathcal{C}_{i} , f ^ i \hat{f}_{i} , and θ A \theta_{\text{A}} such that ‖ ϵ R ​ ( x ) ‖ \|\epsilon_{\text{R}}(x)\| is minimized. With this definition, ϵ R \epsilon_{\text{R}} denotes all errors that cannot be captured using rigid-body dynamics modelling. For example, ϵ R \epsilon_{\text{R}} can be caused by elasticities in the system’s bodies or disturbances caused by attached cables. In contrast, ϵ f ^ , i \epsilon_{\hat{f},i} and ϵ 𝒞 , i \epsilon_{\mathcal{C},i} cause errors between the rigid-body dynamics model f ^ A \hat{f}_{\text{A}} and the ideal rigid-body dynamics model f ^ A ∗ \hat{f}_{\text{A}}^{*} . Figure 4 , being inspired by Von Luxburg and Schölkopf (2011) , illustrates the errors between the system’s dynamics f f and an analytical approximation f ^ A \hat{f}_{\text{A}} . In turn, the analytical model error reads

 

 
 | 
 ϵ A = ϵ R + ϵ E = ϵ R + ∑ i ( ϵ 𝒞 , i ​ f ^ i + ϵ 𝒞 , i ​ ϵ f ^ , i + 𝒞 i ​ ϵ f ^ , i ) . \epsilon_{\text{A}}=\epsilon_{\text{R}}+\epsilon_{\text{E}}=\epsilon_{\text{R}}+\sum_{i}\left(\epsilon_{\mathcal{C},i}\hat{f}_{i}+\epsilon_{\mathcal{C},i}\epsilon_{\hat{f},i}+\mathcal{C}_{i}\epsilon_{\hat{f},i}\right). | 
 | 
 (55) | 
 

 The foremost goal behind the combination of data-driven models with an analytical model is to reduce ϵ A \epsilon_{\text{A}} . Therefore, as depicted in Figure 5 , analytical structured models can be distinguished by how ϵ A \epsilon_{\text{A}} is reduced using data-driven modeling, namely into:

 
 • 
 
 ARM, in which data-driven models approximates ϵ A \epsilon_{\text{A}} directly,

 

 • 
 
 ALM of latent analytical functions, in which data-driven models approximate f ^ i \hat{f}_{i} and/or 𝒞 i \mathcal{C}_{i} in ( 52 ) if the respective analytical model is inaccurate, or alternatively,

 

 • 
 
 ALM of latent residual functions, in which data-driven models approximate ϵ 𝒞 , i \epsilon_{\mathcal{C},i} and/or ϵ f ^ , i \epsilon_{\hat{f},i} in ( 52 ) hence also using the analytical functions f ^ i \hat{f}_{i} and 𝒞 i \mathcal{C}_{i} .

 

 
 
 
 
 Using the unified view to compare direct and inverse dynamics 

 
 To illustrate the implications underlying ( 52 ), consider the forward dynamics as in ( 48 ). With ( 52 ) we then obtain

 

 
 | 
 q ¨ = ϵ R + ( M − 1 + ϵ M − 1 ) ​ ( Q + Q I + ϵ Q ) , \ddot{q}=\epsilon_{\text{R}}+\left(M^{-1}+\epsilon_{M^{-1}}\right)\left(Q+Q_{\text{I}}+\epsilon_{Q}\right), | 
 | 
 (56) | 
 

 where the error functions in the entries of M − 1 M^{-1} are denoted by ϵ M − 1 ​ ( q ) \epsilon_{M^{-1}}(q) , and error functions in the entries of Q + Q I ​ = ^ ​ ∑ i f ^ i Q+Q_{\text{I}}\,\widehat{=}\,\sum_{i}\hat{f}_{i} are denoted by ϵ Q ​ ( q , q ˙ , t ) \epsilon_{Q}(q,\dot{q},t) .
In comparison, the system’s inverse dynamics read

 

 
 | 
 Q u = ϵ R + ( M + ϵ M ) ​ q ¨ − ( Q C + Q G + Q d + Q I + ϵ CGd ) , Q_{\text{u}}=\epsilon_{\text{R}}+\left(M+\epsilon_{M}\right)\ddot{q}-\left(Q_{\text{C}}+Q_{\text{G}}+Q_{\text{d}}+Q_{\text{I}}+\epsilon_{\text{CGd}}\right), | 
 | 
 (57) | 
 

 with the force errors ϵ CGd ​ ( q , q ˙ , t ) \epsilon_{\text{CGd}}(q,\dot{q},t) .

 
 
 A model deviates from the dynamics either trough model errors as in ( 56 ) and ( 57 ) or, through observation noise.
The model errors occurring in ( 57 ) can be denoted jointly as ϵ R + ϵ M ​ q ¨ + ϵ CGd \epsilon_{\text{R}}+\epsilon_{M}\ddot{q}+\epsilon_{\text{CGd}} . As these errors directly occur in the output of the inverse dynamics they can straightforwardly approximated using ARM. In comparison, forward dynamics formulations as in ( 56 ), nonlinearly transform the error ϵ Q \epsilon_{Q} by ( M − 1 + ϵ M − 1 ) \left(M^{-1}+\epsilon_{M^{-1}}\right) .
As illustrated in Example 3.1 , this can render ARM significantly more challenging for forward dynamics compared to inverse dynamics.

 
 
 Example 3.1 . 
 
 Structured modeling of pendulum dynamics 
 The forward dynamics of an undamped-uncontrolled pendulum in terms of its angle q q reads 

 

 
 | 
 q ¨ ​ ( q ) = M − 1 ​ Q G , \ddot{q}(q)=M^{-1}Q_{\text{G}}, | 
 | 
 (58) | 
 

 with the inverse of the inertia matrix M − 1 = 1 / m ​ L 2 M^{-1}=1/mL^{2} and the gravitational torque Q G = − sin ⁡ ( q ) ​ L ​ m ​ g Q_{\text{G}}=-\sin(q)Lmg consisting of the gravitational acceleration g g , the pendulum’s mass m m , and the length of the pendulum’s rod L L . For the sake of this illustration, assume that the estimate for the gravitational acceleration g g is erroneous such that an a-priori available model reads Q ^ g = − sin ⁡ ( q ) ​ L ​ m ​ ( g + ϵ g ) \hat{Q}_{g}=-\sin(q)Lm(g+\epsilon_{g}) where ϵ g \epsilon_{g} denotes a constant function. Even though ϵ g \epsilon_{g} is constant, if its propagated through the dynamics function the output prediction error becomes a nonlinear function 

 

 
 | 
 ϵ q ¨ ​ ( q ) = − sin ⁡ ( q ) L ​ ϵ g . \epsilon_{\ddot{q}}(q)=\frac{-\sin(q)}{L}\epsilon_{g}. | 
 | 
 (59) | 
 

 Therefore, if the model designer is confident that the inverse inertia matrix M ​ ( q , m , L ) − 1 M(q,m,L)^{-1} and the kinematic dependency sin ⁡ ( q ) ​ L \sin(q)L are suiting parametrizations of the real system’s physics, one can place a data-driven model on ϵ g \epsilon_{g} instead of ϵ q ¨ \epsilon_{\ddot{q}} . In turn, a less complex data-driven model can be used.
Alternatively, one can use the additional prior knowledge that g 0 g 0 , such that Q ^ g = − sin ⁡ ( q ) ​ L ​ m ​ g ^ 2 \hat{Q}_{g}=-\sin(q)Lm\hat{g}^{2} where g ^ ​ ( x , θ D ) \hat{g}(x;\theta_{\text{D}}) denotes a suitable data-driven model.
 

 
 
 
 Another significant quantity to consider is the acceleration noise . Estimates of q ¨ \ddot{q} are usually obtained via numerical time-differentiation of velocity or position measurements. In return, the measurement noise is amplified in the acceleration estimate. As it is often easier for data-driven models to approximate the comparably large noise in the model’s output compared to the model’s input, forward dynamics models can be preferred to inverse dynamics models. This is also a reason that might explain why the majority of data-driven models approximate transition dynamics ( 1 ) as state measurements are usually readily available. However, as forward dynamics formulations are usually used to compute trajectory predictions via numerical integration methods, these trajectory predictions contain an additional integration error.

 
 
 Another important aspect forms the measurement of Q u Q_{\text{u}} . Recall that the actuation dynamics Q u ′ ​ ( x , Q u,desired ) Q_{\text{u}}^{\prime}(x,Q_{\text{u,desired}}) denote the difference between Q u,desired Q_{\text{u,desired}} and Q u Q_{\text{u}} . The identification of Q u ′ Q_{\text{u}}^{\prime} is therefore critical if we want to use a dynamics model for control or simulation. However, the estimation of Q u Q_{\text{u}} via current measurements in electric motors is inaccurate. In return, if we learn an inverse dynamics model, the model’s output error ϵ y \epsilon_{y} will also contain the difference between Q u Q_{\text{u}} and its estimation. Alternatively, we can use the end-effector force to estimate Q u Q_{\text{u}} . However, such estimates depend on mechanical parameters.
In comparison, directly measuring Q u Q_{\text{u}} through force sensors in the actuators yields accurate measurements. Albeit, sensors, such as joint torque sensors, are costly. Compared to learning inverse dynamics with ARM, learning forward dynamics with ALM has the advantage that we do not need to measure Q u Q_{\text{u}} but instead approximate it directly with a data driven-model.

 
 
 
 
 
 
 ϵ 
 x 
 
 
 
 
 
 
 ϵ 
 y 
 
 
 
 
 y 
 
 
 
 x 
 
 R 
 
 
 
 ϵ 
 R 
 
 
 A A 
 
 
 = 
 
 - 
 
 ~ 
 y 
 
 
 
 
 ^ 
 f 
 
 A 
 
 
 
 + 
 
 
 ϵ 
 A 
 
 
 
 ϵ 
 y 
 
 
 
 
 
 
 D 
 
 
 
 
 ~ 
 y 
 
 
 
 
 
 ~ 
 x 
 
 
 A 
 
 
 
 ϵ 
 A 
 
 
 E 
 
 
 
 ϵ 
 E 
 
 
 A 
 
 
 
 
 ^ 
 f 
 
 A 
 
 
 
 
 
 = 
 y 
 
 f 
 
 
 
 ( 
 x 
 ) 
 
 
 
 
 
 A 
 
 
 = 
 
 
 
 ^ 
 f 
 
 A 
 
 
 
 
 ∑ 
 i 
 
 
 ⁢ 
 
 
 C 
 i 
 
 
 
 
 ^ 
 f 
 
 i 
 
 
 
 
 
 
 
 
 
 
 ∑ 
 i 
 
 
 
 
 ( 
 
 + 
 
 ⁢ 
 
 
 ϵ 
 
 
 
 
 
 
 
 C 
 , 
 i 
 
 
 
 
 
 
 ^ 
 f 
 
 i 
 
 
 
 ⁢ 
 
 
 ϵ 
 
 
 
 
 
 
 
 C 
 , 
 i 
 
 
 
 
 
 ϵ 
 
 
 
 
 
 
 
 
 ^ 
 f 
 
 , 
 i 
 
 
 
 
 
 ⁢ 
 
 
 C 
 i 
 
 
 
 ϵ 
 
 
 
 
 
 
 
 
 ^ 
 f 
 
 , 
 i 
 
 
 
 
 
 ) 
 
 
 
 
 
 
 Figure 5 : Schematic of analytical structured modeling. Data-driven models can approximate the output error of the analytical model (ARM) and/or reduce latent errors inside the analytical model (ALM). Robot image with courtesy of Grimminger et al. (2020) . 
 
 
 
 Combining ARM with ALM: An uncharted territory 

 
 While we separate analytical structured modeling in between the lines of ARM and ALM, these two modeling approaches can be combined if the dynamics are affected by phenomenons that are not describable by rigid-body dynamics.

 
 
 As shown in Section ( 3.4.2 ), some of the recent works that apply ALM on forward dynamics equations assume that M − 1 M^{-1} in ( 56 ) accurately depicts the causal map between the forces and acceleration of a system. Accordingly, the error ϵ M − 1 \epsilon_{M^{-1}} is only caused by a wrong estimate of θ A \theta_{\text{A}} .
Placing a data-driven model on the error-prone parts of Q Q (ALM) as well as on the output residual of the structured model (ARM), potentially reduces ϵ Q \epsilon_{Q} and ϵ R \epsilon_{\text{R}} . If the reduction in the former error terms causes θ A \theta_{\text{A}} to be closer to θ A ∗ \theta_{\text{A}}^{*} , ϵ M − 1 \epsilon_{M^{-1}} also reduces. Consequently, the prediction performance significantly improves. However, this is currently solely a hypothesis based on the empirical validations made in the literature discussed in Section 3.4 . To which extend modeling of forward dynamics benefits from a combination of ALM and ARM requires further future analysis.

 
 
 
 

### 3.2 Data-driven dynamics models and analytical parametric networks

 
 In this section, we briefly discuss solely data-driven dynamics models as these models are important for structured modeling. Further, we discuss works that learn the parameters of analytical models using gradient-based optimization. The insights about analytical models are useful for improving future generations of analytical structured models.

 
 

#### 3.2.1 Data-driven dynamics models

 
 Common data-driven models used for the identification of dynamics are (Bayesian) linear regression, neural networks (NNs), and Gaussian processes (GPs). Most works that learn dynamics function using solely data-driven models use transition dynamics or inverse transition dynamics. Nguyen-Tuong and Peters (2011) surveys these approaches for robot control. In the following, we briefly discuss NNs and GPs as examples of data-driven models being used for analytical structured learning.

 
 
 NNs achieved breakthroughs in big-data domains such as computer vision ( Krizhevsky et al., 2012 ) , the game of go ( Silver et al., 2016 ) , and control of computer games Berner et al. (2019) ; Vinyals et al. (2019) . NNs have also been deployed for the inference of physical system dynamic’s ( Funahashi and Nakamura, 1993 ; Kuschewski et al., 1993 ; Jansen, 1994 ; Morton et al., 2018 ) . Notably, NNs have learned successfully transition dynamics and control-policies on contact rich domains ( Chua et al., 2018 ; Nagabandi et al., 2020 ) . Remarkably, Nagabandi et al. (2020) demonstrated that NNs enable the inference of specific control policies for handling objects inside a robotic hand. However, in ( Nagabandi et al., 2020 ) the training of a control policy for a specific task-object combination requires several hours of data.

 
 
 In comparison, GPs are non-parametric and probabilistic models that define a normal distribution over functions. GPs provide a measure of uncertainty of the estimation result in form of their posterior variance. In addition, GPs convert Bayesian inference into numerically efficient linear algebraic equations. An introduction to multivariate GP regression is given in Alvarez et al. (2011) . In what follows, we denote a multivariate GP as f ^ ∼ 𝒢 ​ 𝒫 ​ ( 𝐦 ⁡ ( x ) , 𝐤 ⁡ ( x , x ′ ) ) \hat{f}\sim\mathcal{GP}\left(\mathbf{m}(x),\mathbf{k}(x,x^{\prime})\right) with the vector-valued mean function 𝐦 ⁡ ( x ) \mathbf{m}(x) and matrix-valued kernel 𝐤 ⁡ ( x , x ′ ) \mathbf{k}(x,x^{\prime}) .
 Deisenroth and Rasmussen (2011) learned the system’s transition dynamics with a GP. This GP dynamics model was then used for the nonlinear control of real mechanical systems with to NNs comparably small amounts of data. One key take away of ( Deisenroth and Rasmussen, 2011 ) has been the usage of a probabilistic model which allows propagating uncertainty in the observed state-actions through the system dynamics. In turn, this approach improved significantly the robustness of a control policy trained on the GP dynamics model. Other applications of dynamics modeling with GPs for control includes
 ( Nguyen-Tuong and Peters, 2010 ; Kocijan et al., 2005 ; Frigola et al., 2013 ; Mattos et al., 2016 ; Doerr et al., 2017 ; Eleftheriadis et al., 2017 ; Doerr et al., 2018 ) . The main disadvantage of GPs is their computational complexity, which typically scales cubically with the number of data points. Even though their computational complexity can be reduced using sparse GPs Quiñonero-Candela and Rasmussen (2005) , the computational effort required to work with such models is considerably larger than using NNs.

 
 
 Table 2 : Summary of works on analytical structured models that are detailed in this survey. We use the abbreviation FD for forward dynamics and ID for inverse dynamics. The entries in the rightmost corner link to code repositories that have been submitted alongside the publications. 
 
 
 
 | 
 Publication | 
 Year | 
 Dynamics | 
 
 
 
 Data-driven | 

 
 model type | 

 | 
 
 
 
 Data-driven | 

 
 approximation | 

 | 
 
 
 
 Real system | 

 
 experiments | 

 | 
 Optimization library | 

 
 \multirow 1*APN | 
 Ledezma et al.  Ledezma and Haddadin (2017) | 
 2017 | 
 ID An et al. (1985) | 
 × \times | 
 × \times | 
 ✓ | 
 fmincon (Matlab) | 

 
 | 
 Ledezma et al.  Ledezma and Haddadin (2018) | 
 2018 | 
 ID An et al. (1985) | 
 × \times | 
 × \times | 
 ✓ | 
 fmincon (Matlab) | 

 
 | 
 Sutanto et al. (2020) | 
 2020 | 
 FD Luh et al. (1980) | 
 × \times | 
 × \times | 
 ✓ | 
 PyTorch (Python) | 

 
 \multirow 1*ARM | 
 Nguyen-Tong et al.  Nguyen-Tuong and Peters (2010) | 
 2010 | 
 ID | 
 GP | 
 ϵ A \epsilon_{\text{A}} | 
 ✓ | 
 × \times | 

 
 | 
 De La Cruz et al. (2011) | 
 2011 | 
 ID | 
 LWPR | 
 ϵ A \epsilon_{\text{A}} | 
 × \times | 
 × \times | 

 
 | 
 Um et al. (2014) | 
 2014 | 
 ID | 
 GP | 
 ϵ A \epsilon_{\text{A}} | 
 × \times | 
 × \times | 

 
 | 
 Grandia et al. (2018) | 
 2018 | 
 ID | 
 GP/LWPR | 
 ϵ ~ u \tilde{\epsilon}_{u} | 
 ✓ | 
 GPy (Python) | 

 
 \multirow 1*ALM | 
 Cheng et al. (2015) | 
 2016 | 
 ID | 
 GP | 
 ℒ \mathcal{L} | 
 ✓ | 
 × \times | 

 
 | 
 Geist et al. Geist and Trimpe (2020) | 
 2020 | 
 FD | 
 GP | 
 Q I Q_{\text{I}} | 
 × \times | 
 scikit-learn (Python) | 

 
 | 
 Hwangbo et al. (2019) | 
 2019 | 
 FD Hwangbo et al. (2018) | 
 NN | 
 Q u Q_{\text{u}} | 
 ✓ | 
 × \times | 

 
 | 
 Greydanus et al. (2019) | 
 2019 | 
 FD | 
 NN | 
 ℋ \mathcal{H} | 
 ✓ | 
 PyTorch (Python) | 

 
 | 
 Lutter et al. (2019a) | 
 2019 | 
 FD/ID | 
 NN | 
 M M , Q G Q_{\text{G}} | 
 ✓ | 
 PyTorch (Python) | 

 
 | 
 Lutter et al. (2019b) | 
 2019 | 
 FD/ID | 
 NN | 
 M M , V V | 
 ✓ | 
 PyTorch (Python) | 

 
 | 
 Gupta et al. (2020) | 
 2020 | 
 FD/ID | 
 NN | 
 M M , V V , B B , Q d Q_{\text{d}} | 
 × \times | 
 PyTorch / Flux (Julia) | 

 
 | 
 Lutter et al. (2020) | 
 2020 | 
 FD Kim (2012) | 
 NN | 
 Q u Q_{\text{u}} | 
 ✓ | 
 × \times | 

 
 | 
 Toth et al. (2019) | 
 2020 | 
 FD | 
 NN | 
 ℋ \mathcal{H} | 
 × \times | 
 × \times | 

 
 | 
 Cranmer et al. (2020) | 
 2020 | 
 FD | 
 NN | 
 ℒ \mathcal{L} | 
 × \times | 
 JAX (Python) | 

 

 
 
 

#### 3.2.2 Analytical parametric networks

 
 Recent works propose to formulate analytical models as analytical parametric networks (APN) that are trained with gradient-based optimization methods. While technically these kinds of models are analytical models, the techniques used are closely related to techniques from data-driven modeling which in return blurs the line between analytical and solely data-driven modeling.

 
 
 Ledezma and Haddadin (2017) reformulated the inverse dynamics of a robot arm such that both the kinematic and dynamic parameters can be estimated via gradient-based optimization. In this approach, the inverse dynamics of the robot arm are separated into a kinematic and dynamic network which allows estimating the kinematic parameters before estimating the dynamic parameters. This approach has been extended by Ledezma and Haddadin (2018) towards the identification of inverse dynamics of humanoid robots.

 
 
 Similar to Ledezma and Haddadin (2017) , Sutanto et al. (2020) described a recursive formulation of the Newton-Euler inverse dynamics as a differentiable computational graph. The dynamics parameters are estimated via automatic differentiation. In this work, the authors placed special emphasis on the incorporation of additional structural knowledge contained in the mass parameters as discussed in Traversaro et al. (2016) . Traversaro et al. (2016) show that not every positive-definite matrix constitutes a physically plausible inertia matrix as the rotational inertia matrix must also fulfill triangular inequalities with respect to the principal moments of inertia. These insights led Sutanto et al. (2020) to a parametrization of the rotational inertia matrix of each body respecting triangular inequalities.
It should be noted that the experiments detailed in Sutanto et al. (2020) show similar training results for the models with parametrization of the rotational inertia matrix either solely in terms of positive definiteness or by taking the triangular inequality property into account.

 
 
 
 

### 3.3 Analytical output residual modeling

 
 In this section, we discuss selected literature on ARM. In practice, in ARM a data-driven model simply approximates the analytical residual function ϵ A + ϵ y \epsilon_{\text{A}}+\epsilon_{y} .

 
 
 From linear-regression of analytical models to semi-parametric ARM models 

 
 Historically, ARM originated from the need for accurate dynamic models on robot arms. Here, Siciliano et al. (2010, p. 280) referrers to Atkeson et al. (1986) as the standard approach used for the identification of the dynamics parameters of robot arms. Atkeson et al. (1986) leverage the linearity of the (rigid) robot arm’s inverse dynamics with respect to the dynamical parameters ( Siciliano et al., 2010 , p. 259) , writing

 

 
 | 
 Q u = Φ ⁡ ( q , q ˙ , q ¨ ) ​ θ A . Q_{\text{u}}=\Phi(q,\dot{q},\ddot{q})\theta_{\text{A}}. | 
 | 
 (60) | 
 

 The least squares estimate of θ A \theta_{\text{A}} is obtained as

 

 
 | 
 θ ^ A = ( Φ ⊤ ​ Φ ) − 1 ​ Φ ⊤ ​ Q ~ u , \hat{\theta}_{\text{A}}=(\Phi^{\top}\Phi)^{-1}\Phi^{\top}\tilde{Q}_{u}, | 
 | 
 (61) | 
 

 with measurements of Q u Q_{\text{u}} being denoted as Q ~ u \tilde{Q}_{u} and ( Φ ⊤ ​ Φ ) − 1 ​ Φ ⊤ (\Phi^{\top}\Phi)^{-1}\Phi^{\top} being the left pseudo-inverse matrix of Φ \Phi . Some of these parameters denote linear combinations of dynamical parameters ( Siciliano et al., 2010 , p. 280) . However, the least-squares approach requires a good prior model of the kinematic parameters and dynamic parameters (masses and inertias) acting on the systems. Therefore, Nguyen-Tuong and Peters (2010) proposed to combine ( 61 ) with GP regression resulting in a semi-parametric or fully-parametric structured model. The semi-parametric modeling approach simply approximates the analytical model’s residual via a zero-mean GP, writing ϵ ^ A ∼ 𝒢 ​ 𝒫 ​ ( 0 , 𝐤 ⁡ ( x , x ′ ) ) \hat{\epsilon}_{\text{A}}\sim\mathcal{GP}\left(0,\mathbf{k}(x,x^{\prime})\right) such that

 

 
 | 
 Q ^ u ∼ Φ ⁡ ( x ) ​ θ ^ A + 𝒢 ​ 𝒫 ​ ( 0 , 𝐤 ⁡ ( x , x ′ ) ) = 𝒢 ​ 𝒫 ​ ( Φ ⁡ ( x ) ​ θ ^ A , 𝐤 ⁡ ( x , x ′ ) ) , \hat{Q}_{u}\sim\Phi(x)\hat{\theta}_{\text{A}}+\mathcal{GP}\left(0,\mathbf{k}(x,x^{\prime})\right)=\mathcal{GP}\left(\Phi(x)\hat{\theta}_{\text{A}},\mathbf{k}(x,x^{\prime})\right), | 
 | 
 (62) | 
 

 with 𝐤 ⁡ ( x , x ′ ) \mathbf{k}(x,x^{\prime}) denoting a diagonal matrix-valued kernel function. A Gaussian distributed estimate of the system’s dynamic parameters θ ^ A \hat{\theta}_{\text{A}} can then be inferred via the posterior distribution of ( 62 ), writing Q ^ u | 𝒟 \hat{Q}_{u}|\mathcal{D} ( Williams and Rasmussen, 2006 , p. 27-29) .

 
 
 The second model proposed in Nguyen-Tuong and Peters (2010) uses the kernel trick to obtain an analytical kernel of the inverse dynamics’, writing

 

 
 | 
 𝐤 A ​ ( x , x ′ ) = Φ ⊤ ​ W ​ Φ + Σ y , \mathbf{k}_{\text{A}}(x,x^{\prime})=\Phi^{\top}W\Phi+\Sigma_{y}, | 
 | 
 (63) | 
 

 with Σ y \Sigma_{y} denoting a diagonal matrix of observation noise variances and W W being a diagonal matrix denoting the prior variance on θ p \theta_{p} . An algorithm that is defined solely in terms of inner products in input space is lifted by the kernel trick into feature space ( Williams and Rasmussen, 2006 , p. 12) .
The analytical kernel GP can then be combined with ϵ ^ A ∼ 𝒢 ​ 𝒫 ​ ( 0 , k ⁡ ( x , x ′ ) ) \hat{\epsilon}_{\text{A}}\sim\mathcal{GP}\left(0,k(x,x^{\prime})\right) to yield

 

 
 | 
 Q ^ u = 𝒢 ​ 𝒫 ​ ( 0 , k A ​ ( x , x ′ ) + k ⁡ ( x , x ′ ) ) . \hat{Q}_{u}=\mathcal{GP}\left(0,k_{\text{A}}(x,x^{\prime})+k(x,x^{\prime})\right). | 
 | 
 (64) | 
 

 Nguyen-Tuong and Peters (2010) showed that ( 62 ) and ( 64 ) achieved comparable prediction accuracy on a real robot arm while ( 62 ) was slightly faster in computing predictions.
Similar to Nguyen-Tuong and Peters (2010) , De La Cruz et al. (2011) combined prior analytical knowledge of a robot arm’s inverse dynamics with locally weighted parametric regression (LWPR) such that its receptive field is a first-order approximation of the analytical model. De La Cruz et al. (2011) assumed the analytical parameters as fixed. Other semi-parametric models for learning a robot arms inverse dynamics are found in Camoriano et al. (2016) .

 
 
 
 Combining a recursive Newton-Euler formulation with GP regression 

 
 Um et al. (2014) extended the semi-parametric model proposed in Nguyen-Tuong and Peters (2010) . In this work, the authors emphasize that for general kinematic trees – such as robot arms – one can obtain the system’s inverse dynamics in form of a recursive formulation of the Newton-Euler equations Siciliano et al. (2010) . The model assumes an accurate prior analytical model whose parameters are assumed known. The proposed modeling procedure splits into a forward and backward pass computation scheme. First, via assumed knowledge of the kinematic and inertial parameters, the joint velocities as well Q ^ C \hat{Q}_{\text{C}} and Q ^ g \hat{Q}_{\text{g}} at each joint are computed in a forward pass through the kinematic tree. Secondly, in a backward pass, the joint torques residuals of each joint are computed using the respective joint velocity as well as the generalized force acting on the parent joints. The generalized force acting on the parent joints is itself a prediction by another GP. To pass knowledge recursively from joint to joint as well as using only a subset of relevant states as GP inputs are both important propositions.

 
 
 
 Using kinematics to derive a contact-invariant formulation of errors 

 
 Grandia et al. (2018) learned the residual of a quadruped robot’s inverse dynamics in a formulation that stays invariant under changes in the contact configuration . Here, if the point-feet of the quadruped are pressed onto the surface, constraints are activated which are expressed via ( 38 ). As first step, the authors eliminate Q I Q_{\text{I}} from the quadrupeds dynamics using D’Alemberts principle ( 44 ) such that

 

 
 | 
 ( I − A + ​ A ) ​ ϵ ^ A = ( I − A + ​ A ) ​ ( Q d + Q u + Q C − M ​ q ¨ ) . (I-A^{+}A)\hat{\epsilon}_{\text{A}}=(I-A^{+}A)(Q_{\text{d}}+Q_{\text{u}}+Q_{\text{C}}-M\ddot{q}). | 
 | 
 (65) | 
 

 Secondly, while the constraint is active, every part of a force pointing in 𝖱 ⁡ ( A + ) \mathsf{R}(A^{+}) causes a reaction force Q I Q_{\text{I}} which in return can cause a jump in every component of q ¨ \ddot{q} . As learning the jump in q ¨ \ddot{q} is difficult, Grandia et al. (2018) use a coordinate transformation J u J_{u} to express the force error via

 

 
 | 
 ϵ A = A ⊤ ​ ϵ ~ c + J u ​ ϵ ~ u \epsilon_{\text{A}}=A^{\top}\tilde{\epsilon}_{c}+J_{u}\tilde{\epsilon}_{u} | 
 | 
 (66) | 
 

 with A ⊤ ​ ϵ ~ c ∈ 𝖱 ⁡ ( A + ) A^{\top}\tilde{\epsilon}_{c}\in\mathsf{R}(A^{+}) and J u ​ ϵ ~ u ∈ 𝖭 ⁡ ( A ) J_{u}\tilde{\epsilon}_{u}\in\mathsf{N}(A) . A careful choice of J u J_{u} ensures that the activation of constraints only changes the error ϵ ~ c \tilde{\epsilon}_{c} .
Therefore, with ( 65 ) one obtains ( I − A + ​ A ) ​ ϵ ^ A = ( I − A + ​ A ) ​ J u ​ ϵ ~ u (I-A^{+}A)\hat{\epsilon}_{\text{A}}=(I-A^{+}A)J_{u}\tilde{\epsilon}_{u} and in return a constraint invariant formulation of the force error as

 

 
 | 
 ϵ ~ u = ( ( I − A + ​ A ) ​ J u ) + ​ ( I − A + ​ A ) ​ ( Q d + Q u + Q C − M ​ q ¨ ) . \tilde{\epsilon}_{u}=\left((I-A^{+}A)J_{u}\right)^{+}(I-A^{+}A)(Q_{\text{d}}+Q_{\text{u}}+Q_{\text{C}}-M\ddot{q}). | 
 | 
 (67) | 
 

 The authors approximated ϵ ~ u \tilde{\epsilon}_{u} using LWPR as well as GP regression.

 
 
 
 
 

### 3.4 Analytical latent modeling

 
 In analytical models, it is often known which of its latent functions are prone to modeling errors.
ALM seeks to reduce the error of an analytical model by placing data-driven models on the unknown latent functions of an analytical model. Further, ALM allows to incorporate prior knowledge on the mathematical properties of an analytical latent function into the design of its data-driven approximation. In the following, we detail different works on ALM. These works differ in which part of the analytical model is approximated with a data-driven model, namely: (i) The entries of M M , M − 1 M^{-1} , or the Lagrangian function, (ii) the entries of Q u Q_{\text{u}} , or (iii) the entries of Q Q which are transformed using constraint knowledge.

 
 

#### 3.4.1 Latent modeling using energy conservation

 
 The generalized inertia matrix M M is of great significance for rigid-body dynamics modeling. The fictitious force Q C Q_{\text{C}} can be derived in terms of M M , see ( 33 ). Moreover, if Q G Q_{\text{G}} denotes the gravitational force then this force is also a function of the mass parameters. Furthermore, in forward dynamics, M − 1 M^{-1} is multiplied with Q + Q I Q+Q_{\text{I}} while in inverse dynamics the inertial force M ​ q ¨ M\ddot{q} has a significant impact on the final estimation results. In the works presented in the previous section, it is a common assumption that the inertia matrix and even its parameters are known a-priori. However, this can lead to large errors in ϵ M \epsilon_{M} / ϵ M − 1 \epsilon_{M^{-1}} and the part of ϵ Q \epsilon_{Q} / ϵ C ​ G ​ d \epsilon_{CGd} that is caused by Q C Q_{\text{C}} . Therefore, recent works propose to approximate ϵ M \epsilon_{M} / ϵ M − 1 \epsilon_{M^{-1}} via a data-driven model by either directly modeling the entries of M M or by parametrization of the inertia matrix in terms of a Lagrangian. The works on Langrangian and Hamiltonian NN were inspired by the seminal work of Chen et al. (2018) on neural differential equations.

 
 
 Learning the Lagrangian 

 
 Often analytical parametrizations of M − 1 M^{-1} and Q C Q_{C} are either not available or the effort required obtaining these is considered too large. Instead, one can express the system’s dynamics in terms of the Lagrangian function ℒ \mathcal{L} . If the Lagrangian is not explicitly time-dependent one obtains an expression for the forward-dynamics in ( 31 ) and for the inverse-dynamics in ( 34 ). One of the first works that modeled the Lagrangian function via a GP is ( Cheng et al., 2015 ) . Cheng et al. (2015) placed a GP prior on ℒ \mathcal{L} writing ℒ ^ ∼ 𝒢 ​ 𝒫 ​ ( 0 , k ⁡ ( x , x ′ ) ) \hat{\mathcal{L}}\sim\mathcal{GP}(0,k(x,x^{\prime})) and then transformed the GP prior by the operators in ( 34 ) to obtain a structured model for the inverse dynamics equation of a conservative system. However, it is currently not clear how to insert such an ℒ ^ \hat{\mathcal{L}} into the forward dynamics equation ( 31 ) as efficient multi-output GP regression requires that ℒ ^ \hat{\mathcal{L}} is solely linearly transformed. In comparison, Cranmer et al. (2020) models ℒ \mathcal{L} via a NN. In return, the authors obtain structured models both for ( 31 ) as well as ( 34 ).

 
 
 
 Learning the Hamiltonian 

 
 Unlike the Newtonian and Lagrangian formulations of classical mechanics, Hamiltonian mechanics is rarely used for describing the motion of rigid-body systems. Yet, Hamiltonian dynamics is of utmost importance in other branches of mechanics such as quantum mechanics, celestial mechanics, and thermodynamics (cf. Greydanus et al. (2019) ). Hamiltonian mechanics is a reformulation of classical mechanics using the Legendre transform into 2 ​ n 2n first-order ODEs in terms of position coordinates q ∈ ℝ n q\in\mathbb{R}^{n} and a canonical impulse p ∈ ℝ n p\in\mathbb{R}^{n} , writing

 

 
 | 
 q ˙ i = ∂ ℋ ∂ p i , p ˙ i = − ∂ ℋ ∂ q i , \dot{q}_{i}=\frac{\partial\mathcal{H}}{\partial p_{i}},\hskip 28.45274pt\dot{p}_{i}=-\frac{\partial\mathcal{H}}{\partial q_{i}}, | 
 | 
 (68) | 
 

 with ℋ ⁡ ( q , p , t ) \mathcal{H}(q,p,t) , ℋ : ∈ ℝ 2 ​ n → ℝ \mathcal{H}:\in\mathbb{R}^{2n}\rightarrow\mathbb{R} denoting the Hamiltonian function. Similar to the Lagrangian NN, Greydanus et al. (2019) parameterized the Hamiltonian in the above equation via a NN. This model was extended by Toth et al. (2019) using a generative NN structure which enables the inference of Hamiltonian dynamics from high-dimensional observations such as images.

 
 
 
 Learning the inertia matrix 

 
 As shown in ( 27 ), the kinetic energy of a rigid-body system is described in terms of the generalized inertia matrix M M . Thereby, the forward dynamics ( 32 ) as well as inverse dynamics ( 35 ) can be denoted in terms of M M instead of ℒ \mathcal{L} . However, it is inexpedient to directly model the function entries of M M using a data-driven model. As also discussed by Traversaro et al. (2016) , the inertia matrix M M as derived in Section 2.1 must be positive definite. Therefore, Lutter et al. (2019a) proposed a parametrization of M M in terms of a lower triangular matrix L ⁡ ( q , q ˙ ) L(q,\dot{q}) such that M ^ = L ^ ​ ( q , q ˙ ) ​ L ^ ​ ( q , q ˙ ) T \hat{M}=\hat{L}(q,\dot{q})\hat{L}(q,\dot{q})^{T} ensures that M ^ \hat{M} is symmetric. In addition, the diagonal of L ^ ​ ( q , q ˙ ) \hat{L}(q,\dot{q}) is enforced to be positive such that all eigenvalues of M ^ \hat{M} are positive (cf. Section 3.2.2 ). To do so, the output layer of the NN that forms the diagonal entries of L ^ ​ ( q , q ˙ ) \hat{L}(q,\dot{q}) uses a non-negative activation function such as ReLU or Softplus onto which a small positive number is added to prevent numerical instabilities. Lutter et al. (2019a) modeled the potential forces Q G Q_{\text{G}} as well as non-conservative forces Q D Q_{D} jointly via a NN. The authors named the resulting structured model a Deep Lagrangian Neural Network (DELAN) . Lutter et al. (2019b) showed that DELAN can be used for the energy-based control of a Furuta pendulum in which they used the fact that the conservative force can be written in terms of a NN parametrization of the potential energy function V ^ ​ ( q ) \hat{V}(q) , writing Q G = − ∇ q V ^ ​ ( q ) Q_{\text{G}}=-\nabla_{q}\hat{V}(q) .

 
 
 Gupta et al. (2020) extended DELAN by leveraging that the control force is affine in the control signal u u , writing Q u = B ⁡ ( q ) ​ u Q_{u}=B(q)u .
 Gupta et al. (2020) added NN parametrizations of B ⁡ ( q ) B(q) and V ⁡ ( q ) V(q) , such that the deep Lagrangian network becomes

 

 
 | 
 q ¨ = ( L ^ ​ L ^ T ) − 1 ​ ( − ∇ q ( q ˙ ⊤ ​ L ^ ​ L ^ T ) ​ q ˙ + 1 2 ​ ( ∇ q ( q ˙ T ​ L ^ ​ L ^ T ​ q ˙ ) ) T − ∇ q V ^ ​ ( q ) + B ^ ​ ( q ) ​ u + Q ^ d ) . \ddot{q}=(\hat{L}\hat{L}^{T})^{-1}\left(-\nabla_{q}(\dot{q}^{\top}\hat{L}\hat{L}^{T})\dot{q}+\frac{1}{2}\left(\nabla_{q}\left(\dot{q}^{T}\hat{L}\hat{L}^{T}\dot{q}\right)\right)^{T}-\nabla_{q}\hat{V}(q)+\hat{B}(q)u+\hat{Q}_{d}\right). | 
 | 
 (69) | 
 

 
 
 
 

#### 3.4.2 Latent modeling of joint torques

 
 For many analytical descriptions of the system’s dynamics, one can assume that some of the analytical latent functions form better approximations of the real physics than others. For example, for a robot arm, one can argue that the inertia matrix M M , the fictitious force Q C Q_{\text{C}} , and gravitational force Q G Q_{\text{G}} form good parametrizations of the respective physics. The parameters θ A \theta_{\text{A}} of these analytical functions are most likely unknown but can be estimated alongside the parameters of a data-driven model. In this case, the system would still respect energy conservation with respect to the model’s energy E ^ ​ ( q , q ˙ , θ A ) = 1 2 ​ q ˙ T ​ M ^ ​ ( q , θ A ) ​ q ˙ + V ^ ​ ( q , q ˙ , θ A ) \hat{E}(q,\dot{q};\theta_{\text{A}})=\frac{1}{2}\dot{q}^{T}\hat{M}(q;\theta_{\text{A}})\dot{q}+\hat{V}(q,\dot{q};\theta_{\text{A}}) similarly to the structured Lagrangian models. Under this assumption, the majority of the errors ϵ Q \epsilon_{Q} / ϵ C ​ G ​ d \epsilon_{CGd} in ( 56 ) and ( 57 ) stem from an inaccurate description of the joint torques Q u Q_{\text{u}} . The following works learn the dynamics of robots with motor torques being applied inside the robot’s joints.

 
 
 Direct identification of joint torques and combination with a contact model 

 
 Hwangbo et al. (2019) identified the joint torques Q u Q_{\text{u}} induced by a quadruped’s electric motors via a NN a-priori .
The authors used a simple control scheme to let a quadruped trot and meanwhile measured Q u Q_{\text{u}} using joint-torque sensors as well as position errors and velocities. Then a NN was trained to predict Q u Q_{\text{u}} given a sequence of position errors and velocities. In this manner, the authors learned the complete mapping of Q u Q_{\text{u}} including complex interlaced control routines of the motors (PD torque control, PID current control, field-oriented control) as well as transmission and friction disturbances.

 
 
 Afterward, the trained NN joint torque model is combined with a rigid-body simulation Hwangbo et al. (2018) . The simulator uses a hard contact model that respects Coulomb friction cone constraints. Then, a NN based reinforcement learning algorithm was trained to control the quadruped in simulation. Notably, the kinematic and mass parameters of the analytical model were randomly initialized to increase the robustness of the control policy during training.

 
 
 The fact that the trained control policy achieved impressive results on the real quadruped, indicates that the gap between modeled and real dynamics can be bridged via randomization of physical parameters (body length and mass) of a good analytical model in combination with a prior identification of Q u Q_{\text{u}} . This work was further extended by Lee et al. (2020) who trained an unprecedented robust control policy for a quadruped robot traversing challenging terrain.

 
 
 
 Modeling of joint torques inside forward dynamics 

 
 Lutter et al. (2020) (being the authors of DELAN) combined Newton-Euler dynamics in Lie Algebra form Kim (2012) with a NN parametrization of Q u Q_{\text{u}} . The authors compared several models for Q u Q_{\text{u}} , with f NN f_{\text{NN}} denoting a NN model, namely

 

 
 | 
 Viscous:  | 
 Q ^ u ​ = ^ ​ Q u,desired − θ v ​ q ˙ , \displaystyle\hat{Q}_{\text{u}}\,\hat{=}\,Q_{\text{u,desired}}-\theta_{v}\dot{q}, | 
 | 
 (70) | 
 
 
 | 
 Stribeck:  | 
 Q ^ u ​ = ^ ​ Q u,desired − sign ⁡ ( q ˙ ) ​ ( f s + f d ​ exp ⁡ ( − θ s ​ q ˙ 2 ) ) − θ v ​ q ˙ , \displaystyle\hat{Q}_{\text{u}}\,\hat{=}\,Q_{\text{u,desired}}-\operatorname{sign}(\dot{q})\left(f_{s}+f_{d}\exp\left(-\theta_{s}\dot{q}^{2}\right)\right)-\theta_{v}\dot{q}, | 
 | 
 (71) | 
 
 
 | 
 NN Friction:  | 
 Q ^ u ​ = ^ ​ Q u,desired − sign ⁡ ( q ˙ ) ​ ‖ f NN ​ ( q , q ˙ ) ‖ 1 , \displaystyle\hat{Q}_{\text{u}}\,\hat{=}\,Q_{\text{u,desired}}-\operatorname{sign}(\dot{q})\left\|f_{\mathrm{NN}}\left(q,\dot{q}\right)\right\|_{1}, | 
 | 
 (72) | 
 
 
 | 
 NN Residual:  | 
 Q ^ u ​ = ^ ​ Q u,desired − f NN ​ ( q , q ˙ ) , \displaystyle\hat{Q}_{\text{u}}\,\hat{=}\,Q_{\text{u,desired}}-f_{\text{NN}}(q,\dot{q}), | 
 | 
 (73) | 
 
 
 | 
 FF-NN:  | 
 Q ^ u ​ = ^ ​ f NN ​ ( Q u,desired , q , q ˙ ) . \displaystyle\hat{Q}_{\text{u}}\,\hat{=}\,f_{\text{NN}}(Q_{\text{u,desired}},q,\dot{q}). | 
 | 
 (74) | 
 

 Equations ( 70 ),( 71 ), and ( 72 ) are guaranteed to be solely dissipative , while the more classical NN parametrization in ( 73 ) and ( 74 ) do not respect energy dissipativity. However, the first three models assume that the internal motor control routines do not cause an overshoot such that the real Q u Q_{\text{u}} is actually larger than Q u,desired Q_{\text{u,desired}} .

 
 
 The different structured forward dynamics models were trained on data from simulated and real pendulums. Additionally, these models where compared to NN black-box modeling as well as the linear regression model denoted in ( 60 ) and ( 61 ) (cf. Atkeson et al. (1986) ). The training results show that a random initialization of the link parameters compares similarly to having a good prior knowledge of the link parameters. This indicates that it is possible to learn analytical and NN parameters jointly inside a structured model. Further, the joint torque models which enforce dissipativity of Q ^ u \hat{Q}_{\text{u}} gave significantly better long-term predictions. The long-term predictions were computed by feeding the models’ acceleration predictions to a Runge-Kutta-4 solver.

 
 
 
 

#### 3.4.3 Latent modeling using implicit constraint knowledge

 
 For some mechanical systems, such as a robot arm whose end-effector touches a surface or a quadruped robot walking over terrain, it can be desirable to identify the force acting in an implicit constraint directly from data. As detailed in Section 2.3 , the presence of implicit constraint forces Q I Q_{\text{I}} potentially also causes a non-ideal force Q d,I Q_{\text{d,I}} . While Q d,I Q_{\text{d,I}} can be modeled jointly with Q d Q_{\text{d}} , the implicit constraints as in ( 38 ) allows to place additional prior knowledge onto the direction of the force vectors. 
 
 To the best of our knowledge, Geist and Trimpe (2020) were the first that proposed the usage of constrained knowledge for ALM. Here, the authors assumed that an analytical parametrization of M − 1 M^{-1} , and the terms of the constraining equation A A , and b b is given. The parameters of these analytical functions are estimated alongside the data-driven model’s parameters.
In Geist and Trimpe (2020) , a GP prior is placed onto ( M − 1 ​ Q ) ∼ 𝒢 ​ 𝒫 ​ ( 0 , k ⁡ ( x , x ′ ) ) (M^{-1}Q)\sim\mathcal{GP}(0,k(x,x^{\prime})) inside ( 49 ), that is, the acceleration that would be caused by Q Q if the constraints were absent. However, it is more straightforward to place a prior on Q Q rather than its acceleration M − 1 ​ Q M^{-1}Q . With this small adaptation, the model obtained in Geist and Trimpe (2020) reads

 

 
 | 
 q ¨ ^ ∼ 𝒢 ​ 𝒫 ​ ( M − 1 ​ Q b + ( M − 1 ​ P ) ​ m Q , ( M − 1 ​ P ) ​ k Q ​ ( M − 1 ​ P ) ⊤ ) , \hat{\ddot{q}}\sim\mathcal{GP}(M^{-1}Q_{b}+(M^{-1}P)m_{Q},\ (M^{-1}P)k_{Q}(M^{-1}P)^{\top}), | 
 | 
 (75) | 
 

 with analytical mean function m Q ​ ( x ) m_{Q}(x) and kernel function k Q ​ ( x , x ′ ) k_{Q}(x,x^{\prime}) . Here, one can include analytical prior knowledge of Q G Q_{\text{G}} and Q C Q_{\text{C}} via the analytical mean function, writing m Q = Q G + Q C + Q u m_{Q}=Q_{\text{G}}+Q_{\text{C}}+Q_{\text{u}} . If such a prior mean function is chosen, k Q ​ ( x , x ′ ) k_{Q}(x,x^{\prime}) models Q d Q_{\text{d}} as well as the residual of m Q m_{Q} . While the combination of ( 49 ) with a parametric model is straightforward, using a GP has several advantages. For example, one can denote the joint distribution between Q Q and q ¨ \ddot{q} , writing

 

 
 | 
 [ Q ^ q ¨ ^ ] ∼ 𝒢 ​ 𝒫 ​ ( [ m Q M − 1 ​ Q b + ( M − 1 ​ P ) ​ m Q ] , [ k Q k Q ​ ( M − 1 ​ P ) 𝖳 ( M − 1 ​ P ) ​ k Q ( M − 1 ​ P ) ​ k Q ​ ( M − 1 ​ P ) 𝖳 ] ) . \begin{bmatrix}\hat{Q}\\
\hat{\ddot{q}}\end{bmatrix}\sim\mathcal{GP}\begin{pmatrix}\begin{bmatrix}m_{Q}\\
M^{-1}Q_{b}+(M^{-1}P)m_{Q}\end{bmatrix},\begin{bmatrix}k_{Q} k_{Q}(M^{-1}P)^{\mathrm{\sf T}}\\
(M^{-1}P)k_{Q} (M^{-1}P)k_{Q}(M^{-1}P)^{\mathrm{\sf T}}\end{bmatrix}\end{pmatrix}. | 
 | 
 (76) | 
 

 This allows inferring Q ^ \hat{Q} from observations of q ¨ \ddot{q} . Secondly, in GP 2 one can learn on one constraint configuration { A , b } \{A,b\} and then change to a different constraint configuration { A ′ , b ′ } \{A^{\prime},b^{\prime}\} without the need for retraining if both constraints induce the same dissipative force function Q d,I Q_{\text{d,I}} inside the constraint.

 
 
 Besides directly combining ( 49 ) with a regression model, one could also use projections into the constraint related vector spaces as depicted in Figure 3 without including the inertia matrix. In particular, the projection of a vector in ℝ n q \mathbb{R}^{n_{q}} into either 𝖭 ⁡ ( A ) \mathsf{N}(A) and 𝖱 ⁡ ( A 𝖳 ) \mathsf{R}(A^{\mathrm{\sf T}}) , (cf. ( Beard, 2002 , p. 74) ), are given by P 𝖭 ⁡ ( A ) P^{\mathsf{N}(A)} and P 𝖱 ⁡ ( A 𝖳 ) P^{\mathsf{R}(A^{\mathrm{\sf T}})} .

 
 
 
 
 

## 4 Key techniques for analytical structured modeling

 
 In order to design an analytical structured model, the following steps are required:

 
 1. 
 
 Derive an rigid-body dynamics model f ^ A ​ ( x , θ A ) \hat{f}_{\text{A}}(x;\theta_{\text{A}}) .

 

 2. 
 
 Obtain a structured model f ^ ​ ( x , θ ) \hat{f}(x,\theta) , by the combination of data-driven models with the analytical model.

 

 3. 
 
 Collect informative data and estimate the parameters of the structured model.

 

 
 
 
 If the parameters of an analytical model are assumed to be known, one can simply obtain an ARM model by learning ϵ A + ϵ y \epsilon_{\text{A}}+\epsilon_{y} using a data-driven model. However, ARM becomes significantly more challenging if θ A \theta_{\text{A}} is to be learned jointly with θ D \theta_{\text{D}} . This case is particularly interesting as reducing ϵ A \epsilon_{\text{A}} potentially improves the estimate of θ A \theta_{\text{A}} .

 
 
 In this section, we detail key techniques for the design and training of analytical structured models, in which θ A \theta_{\text{A}} and θ D \theta_{\text{D}} are estimated jointly as θ = { θ A , θ D } \theta=\{\theta_{\text{A}},\theta_{\text{D}}\} . The first key technique is the optimization algorithm itself. Note that, all works in Table 2 use gradient-based optimization. Therefore, in Section 4.1 , we discuss gradient-based optimization and illustrate why automatic differentiation is particularly useful to compute the gradient of structured models. By discussing the optimization method, we see which requirement the optimization poses onto the design of an analytical structured model. Subsequently, we detail in Section 4.2 important aspects that must be considered when combining analytical models with data-driven models.

 
 
 

### 4.1 Gradient-based optimization

 
 Analytical structured models are used if standard analytical or data-driven approaches yield unsatisfactory predictions. Usually, this occurs if the system is high-dimensional and subject to nonlinear physical phenomenons. Therefore, most works utilizing analytical structured models require the training of high-dimensional models on large datasets.
 2 2 
 2 
 
 
 
 A dynamics model of a quadruped robot usually has more than 14 output dimensions ( e.g. , Six DOF for the floating base plus two DOF per leg). In this scenario, a large dataset can denote just several thousands of points. 
Subsequently, recent works on analytical structured learning predominantly use gradient-based optimization algorithms to compute the model’s parameters.

 
 
 Gradient-based optimization forms a cornerstone of solely data-driven modeling Bottou et al. (2018) and constitutes one of the most common nonlinear local optimization techniques ( Nelles, 2013 , p. 90) . Table 2 summarizes the literature presented in this survey that utilizes gradient-based optimization. Gradient-based optimization techniques require the computation of the partial derivative ∇ θ ℓ ​ = ^ ​ ∂ ℓ ∂ θ \nabla_{\theta}\ell\,\widehat{=}\,\frac{\partial\ell}{\partial\theta} of an objective function ℓ ⁡ ( f ^ ​ ( x ~ , θ ) , y ~ ) ​ = ^ ​ ℓ ​ ( θ ) \ell(\hat{f}(\tilde{x};\theta),\tilde{y})\,\widehat{=}\,\ell(\theta) , ℓ ∈ ℝ \ell\in\mathbb{R} .
Often it is useful to include higher-order derivatives of the objective function in the parameter update rule. As gradient-based optimization is a local optimization technique and ℓ ⁡ ( θ ) \ell(\theta) is usually non-linear, it is indispensable to restart the optimization several times with random initialization of θ \theta , keeping only the best optimization result. In addition, as models based on NN have a large number of parameters, the gradient update is computed only for a randomly selected batch of data. This optimization procedure is called stochastic gradient-descent in machine learning literature.

 
 
 A typically used objective function for NN models is the L 2 L_{2} loss function with an L 2 L_{2} regularization term on the network’s weights. For GP models the objective function is commonly chosen to be the logarithmic likelihood function.
Both of these objective functions denote forward functions mapping θ \theta to a cost ℓ ⁡ ( θ ) \ell(\theta) .

 
 
 As detailed by Baydin et al. (2017) , gradients of a forward function can be obtained via:

 
 • 
 
 Analytical derivation and implementation , which is time-consuming and error-prone.

 

 • 
 
 Numerical differentiation , which is inaccurate due to round-off and truncation errors and scales poorly with the size of θ \theta .

 

 • 
 
 Symbolic differentiation yielding functions for the analytical gradient. However, these expressions are also closed-form which hinders GPU acceleration and quickly become cryptic due to an ”expression swell” Corliss (1988) .

 

 • 
 
 Automatic differentiation , which computes accurate gradients, is considerably faster than the aforementioned methods, and thanks to steady improvements in the usability of optimization libraries is straightforward to use with ALM.

 

 
 
 
 Structured learning with automatic differentiation 

 
 At the core of automatic differentiation (AD), which is also called algorithmic differentiation or ”autodiff“, lies the insight that forward functions are compositions of elementary operations whose derivatives are known. In return, the derivative of a function can be constructed using the elementary operators’ gradient expressions with the chain rule of differentiation. AD computes a numerical value of the derivative of a computational graph with branching, recursions, loops, and procedure calls Baydin et al. (2017) .

 
 
 Figure 6 : Graph illustrating backward mode automatic differentiation of the objective function of a forward dynamics model. 
 
 
 The two basic forms of AD are forward-mode AD and reverse-mode AD. In what follows, we denote the objective function as ℓ ⁡ ( θ ) = ℓ ⁡ ( c ⁡ ( b ⁡ ( a ⁡ ( θ ) ) ) ) \ell(\theta)=\ell(c(b(a(\theta)))) where a ⁡ ( θ ) a(\theta) , b ⁡ ( a ) b(a) , and c ⁡ ( b ) c(b) are functions of appropriate size. Then, in forward-mode AD , the gradient of a function is obtained via forward accumulation, e.g. , writing ∇ θ ℓ ​ ( θ ) = ∇ c ℓ ​ ( ∇ b c ​ ( ∇ a b ​ ∇ θ a ) ) \nabla_{\theta}\ell(\theta)=\nabla_{c}\ell(\nabla_{b}c(\nabla_{a}b\nabla_{\theta}a)) where ∇ θ b = ∇ a b ​ ∇ θ a \nabla_{\theta}b=\nabla_{a}b\nabla_{\theta}a denotes a Jacobian matrix.

 
 
 Alternatively, reverse-mode AD being also referred to as ”backprop“ in machine learning literature, computes gradients via reverse accumulation, e.g. , writing ∇ θ ℓ ​ ( θ ) = ( ( ∇ c ℓ ​ ∇ b c ) ​ ∇ a b ) ​ ∇ θ a \nabla_{\theta}\ell(\theta)=((\nabla_{c}\ell\nabla_{b}c)\nabla_{a}b)\nabla_{\theta}a . For functions ℓ : ℝ n 1 → ℝ n 2 \ell:\mathbb{R}^{n_{1}}\rightarrow\mathbb{R}^{n_{2}} with n 1 n 2 n_{1} n_{2} reverse mode AD is preferred as it requires less operation counts for the computation of vector-Jacobian products Baydin et al. (2017) . However, reverse-mode AD can have increased memory requirements compared to forward-mode AD. Many automatic differentiation packages allow to combine forward and reverse accumulation.
In the following example, we illustrate reverse-mode AD for ALM.

 
 
 Example 4.1 . 
 
 Computing the gradient of a structured model with reverse-mode AD 
 
Assume that the dynamics function of a mechanical system is modeled via a structured analytical model as 

 

 
 | 
 q ¨ ^ ​ ( x , θ ) = M − 1 ​ ( x , θ ) ​ Q ^ C ​ ( x , θ ) ⏟ a + M − 1 ​ ( x , θ ) ​ Q ^ G ​ ( x , θ ) ⏟ b , \hat{\ddot{q}}(x;\theta)=\underbrace{M^{-1}(x;\theta)\hat{Q}_{\text{C}}(x;\theta)}_{\text{\normalsize$a$}}+\underbrace{M^{-1}(x;\theta)\hat{Q}_{\text{G}}(x;\theta)}_{\text{\normalsize$b$}}, | 
 | 
 (77) | 
 

 in which Q ^ G ​ ( x , θ ) \hat{Q}_{\text{G}}(x;\theta) and Q ^ C ​ ( x , θ ) \hat{Q}_{\text{C}}(x;\theta) denote either analytical and/or data-driven models, and intermediate function expressions are abbreviated as a = M − 1 ​ Q ^ C a=M^{-1}\hat{Q}_{\text{C}} , b = M − 1 ​ Q ^ G b=M^{-1}\hat{Q}_{\text{G}} . Note that Figure 6 depicts the computational graph formed by ( 77 ). The parameters θ \theta are estimated jointly using an objective function of the form ℓ ⁡ ( f ^ ​ ( x , θ ) , y ~ ) ​ = ^ ​ ℓ ​ ( θ ) \ell(\hat{f}(x;\theta),\tilde{y})\,\hat{=}\,\ell(\theta) .
In reverse mode AD, the gradient of ℓ \ell with respect to θ \theta is decomposed as 

 

 
 | 
 ∇ θ ℓ ​ ( θ ) = ∇ Q ^ C ℓ ​ ∇ θ Q ^ C ⏟ m ¯ 1 + ∇ M − 1 ℓ ​ ∇ θ M − 1 ⏟ m ¯ 2 + ∇ Q ^ G ℓ ​ ∇ θ Q ^ G ⏟ m ¯ 3 , \nabla_{\theta}\ell(\theta)=\underbrace{\nabla_{\hat{Q}_{\text{C}}}\ell\nabla_{\theta}\hat{Q}_{\text{C}}}_{\text{\normalsize$\bar{m}_{1}$}}+\underbrace{\nabla_{M^{-1}}\ell\nabla_{\theta}M^{-1}}_{\text{\normalsize$\bar{m}_{2}$}}+\underbrace{\nabla_{\hat{Q}_{\text{G}}}\ell\nabla_{\theta}\hat{Q}_{\text{G}}}_{\text{\normalsize$\bar{m}_{3}$}}, | 
 | 
 (78) | 
 

 with 

 

 
 | 
 | 
 m ¯ 1 = ∇ a ℓ ​ ∇ Q ^ C ​ a ⏟ m ¯ 4 ​ ∇ θ Q ^ C , m ¯ 2 = ( ∇ a ℓ ​ ∇ M − 1 ​ a ⏟ m ¯ 5 + ∇ b ℓ ​ ∇ M − 1 ​ b ⏟ m ¯ 6 ) ​ ∇ θ M − 1 , m ¯ 3 = ∇ b ℓ ​ ∇ Q ^ G ​ b ⏟ m ¯ 7 ​ ∇ θ Q ^ G , \displaystyle\bar{m}_{1}=\underbrace{\nabla_{a}\ell\nabla_{\hat{Q}_{\text{C}}}a}_{\text{\normalsize$\bar{m}_{4}$}}\nabla_{\theta}\hat{Q}_{\text{C}},\hskip 14.22636pt\bar{m}_{2}=\big(\underbrace{\nabla_{a}\ell\nabla_{M^{-1}}a}_{\text{\normalsize$\bar{m}_{5}$}}+\underbrace{\nabla_{b}\ell\nabla_{M^{-1}}b}_{\text{\normalsize$\bar{m}_{6}$}}\big)\nabla_{\theta}M^{-1},\hskip 14.22636pt\bar{m}_{3}=\underbrace{\nabla_{b}\ell\nabla_{\hat{Q}_{\text{G}}}b}_{\text{\normalsize$\bar{m}_{7}$}}\nabla_{\theta}\hat{Q}_{\text{G}}, | 
 | 
 (79) | 
 
 
 | 
 | 
 m ¯ 4 = ∇ q ¨ ^ ℓ ​ ∇ a q ¨ ^ ⏟ m ¯ 8 ​ ∇ Q ^ C a , m ¯ 5 = ∇ q ¨ ^ ℓ ​ ∇ a q ¨ ^ ⏟ m ¯ 8 ​ ∇ M − 1 a , m ¯ 6 = ∇ q ¨ ^ ℓ ​ ∇ b q ¨ ^ ⏟ m ¯ 9 ​ ∇ M − 1 b , m ¯ 7 = ∇ q ¨ ^ ℓ ​ ∇ b q ¨ ^ ⏟ m ¯ 9 ​ ∇ Q ^ G b . \displaystyle\bar{m}_{4}=\underbrace{\nabla_{\hat{\ddot{q}}}\ell\nabla_{a}\hat{\ddot{q}}}_{\text{\normalsize$\bar{m}_{8}$}}\nabla_{\hat{Q}_{\text{C}}}a,\hskip 14.22636pt\bar{m}_{5}=\underbrace{\nabla_{\hat{\ddot{q}}}\ell\nabla_{a}\hat{\ddot{q}}}_{\text{\normalsize$\bar{m}_{8}$}}\nabla_{M^{-1}}a,\hskip 14.22636pt\bar{m}_{6}=\underbrace{\nabla_{\hat{\ddot{q}}}\ell\nabla_{b}\hat{\ddot{q}}}_{\text{\normalsize$\bar{m}_{9}$}}\nabla_{M^{-1}}b,\hskip 14.22636pt\bar{m}_{7}=\underbrace{\nabla_{\hat{\ddot{q}}}\ell\nabla_{b}\hat{\ddot{q}}}_{\text{\normalsize$\bar{m}_{9}$}}\nabla_{\hat{Q}_{\text{G}}}b. | 
 | 
 (80) | 
 

 Therefore, the gradient of an analytical structured model can be computed with AD. Yet, this requires the computation of the numerical expressions for ∇ θ M − 1 \nabla_{\theta}M^{-1} , ∇ θ Q ^ G \nabla_{\theta}\hat{Q}_{\text{G}} , and ∇ θ Q ^ C \nabla_{\theta}\hat{Q}_{\text{C}} . Ideally, an AD library will compute these terms for us by also decomposing the gradients into product-sums of basic gradient functions.
 

 
 
 
 Example 4.1 illustrates that the computation of ∇ θ ℓ ​ ( θ ) \nabla_{\theta}\ell(\theta) requires the partial differentiation of analytical latent functions. Yet, to do this efficiently, several requirements must be fulfilled by an AD library, namely:

 
 • 
 
 Analytical structured models denote forward functions that consist of numerous basic mathematical operators. Ideally, the AD library should only require that the analytical structured model is expressed in terms of a computer library for basic linear algebraic operations such as Numpy Harris et al. (2020) or Torch Paszke et al. (2019) . In turn, the model designer is only required to convert the derived analytical functions into the programming language of the respective AD library.

 

 • 
 
 As detailed in Section 3.4.1 , models for Q ^ C \hat{Q}_{\text{C}} , Q ^ G \hat{Q}_{\text{G}} , or M ^ \hat{M} can be obtained by transformation of an analytical or data-driven function via the partial derivative operators ∇ q \nabla_{q} or ∇ q ˙ \nabla_{\dot{q}} . Therefore, the AD library must be able to automatically compute the gradients (w.r.t. to θ \theta ) of gradients (w.r.t. q q or q ˙ \dot{q} ) of latent functions.

 

 
 Fortunately, recent developments in AD packages such as ”AutoGrad“ Maclaurin et al. (2015) and ”Torch autograd“ allow to compute ∇ θ ℓ ​ ( θ ) \nabla_{\theta}\ell(\theta) in which ℓ ⁡ ( θ ) \ell(\theta) is expressed in either native python (Numpy) code or python (torch) code. Importantly in these AD packages, ∇ θ ℓ ​ ( θ ) \nabla_{\theta}\ell(\theta) can itself include higher-order partial derivatives without braking the AD routines. These AD packages have been further improved in packages such as JAX Bradbury et al. (2018) and PyTorch Paszke et al. (2019) , which additionally combine AD with compilers for GPU acceleration such as XLA . These packages tremendously simplify the synthesis of an analytical structured model, enables easy debugging of the respective computer code, as well as provide computationally efficient implementations with GPU acceleration.

 
 
 
 
 

### 4.2 Model construction of analytical latent models

 
 In the previous section, we detailed how an analytical structured model is trained using modern libraries for AD. These AD libraries only require that a forward function ℓ ​ ( f ^ ​ ( x ~ , θ ) , y ~ ) \ell(\hat{f}(\tilde{x};\theta),\tilde{y}) is expressed in terms of a specified programming language ( e.g. , Python using solely Numpy expressions). In this section, we detail important aspects that must be considered when constructing an analytical structured model.
An analytical model f ^ A ​ ( x , θ A ) \hat{f}_{\text{A}}(x;\theta_{\text{A}}) can be derived using for example a rigid-body dynamics software package such as Pinocchio Carpentier et al. (2019) .

 
 
 If a data-driven model is used as an approximation for a function inside an analytical mode, this usually places additional requirements on the properties of the data-driven model. ALM requires placing data-driven models on unknown functions inside an analytical model.
As discussed in Section 3 , these unknown quantities are either vector-valued functions f i f_{i} ( e.g. , forces) or matrix-valued functions 𝒞 i \mathcal{C}_{i} ( e.g. , matrices).
Depending on the analytical mechanics formulation, these data-driven functions are usually transformed by operators which in return poses additional mathematical requirements onto the data-driven models.
These transformations of data-driven models can be distinguished into:

 
 Matrix transformations , writing

 

 
 | 
 f ^ i 𝒞 ​ ( x ) = 𝒞 i ​ ( x ) ​ f ^ i ​ ( x ) . \hat{f}_{i}^{\mathcal{C}}(x)=\mathcal{C}_{i}(x)\hat{f}_{i}(x). | 
 | 
 (81) | 
 

 e.g. , in forward dynamics a model can be placed on Q d Q_{\text{d}} which is transformed by M − 1 M^{-1} , writing
 q ¨ ^ d = M − 1 ​ F ^ D . \hat{\ddot{q}}_{\text{d}}=M^{-1}\hat{F}_{D}. 

 
 
 Partial derivative transformations , writing

 

 
 | 
 f ^ i ∇ x ​ ( x ) = ∇ x f i ​ ( x ) | x = z , \hat{f}_{i}^{\nabla_{x}}(x)=\nabla_{x}f_{i}(x)|_{x=z}, | 
 | 
 (82) | 
 

 which we denote by abuse of notation as

 

 
 | 
 f ^ i ∇ x ​ ( x ) = ∇ x f i ​ ( x ) , \hat{f}_{i}^{\nabla_{x}}(x)=\nabla_{x}f_{i}(x), | 
 | 
 (83) | 
 

 e.g. , modeling Q C Q_{\text{C}} in terms of a model for the kinetic energy T ^ \hat{T} (cf. ( 33 )) reads
 Q ^ C ​ ( q , q ˙ ) = − ( ∇ q ∇ q ˙ ⊤ ​ T ^ ) ​ q ˙ + ∇ q T ^ . \hat{Q}_{\text{C}}(q,\dot{q})=-\left(\nabla_{q}\nabla_{\dot{q}}^{\top}\hat{T}\right)\dot{q}+\nabla_{q}\hat{T}. 

 
 
 Substitution of input variables by a nonlinear mapping x = u ⁡ ( z ) x=u(z) such that

 

 
 | 
 f ^ i u ​ ( x ) = f i ​ ( z ) | z = u ⁡ ( x ) , \hat{f}_{i}^{u}(x)=f_{i}(z)|_{z=u(x)}, | 
 | 
 (84) | 
 

 e.g. , instead of using an angle coordinate x ​ = ^ ​ ϕ x\,\widehat{=}\,\phi for describing the pose of a pendulum, the pendulums pose can be expressed as z = [ cos ​ ( ϕ ) , sin ​ ( ϕ ) ] z=[\text{cos}(\phi),\text{sin}(\phi)] . This coordinate transformation has the advantage that the entries of z z are bounded to [ − 1 , 1 ] [-1,1] . Another example is the descriptions of z z in terms of a NN with inputs x x .

 
 Note that all of the above operators are linear operators. We illustrate the additional requirements that the above operators pose onto a data-driven model on the examples of NN and GPs.

 
 
 Linearly transformed neural networks 

 
 Hendriks et al. (2020) details linear transformations of NNs such that an equality constrained is fulfilled. Additional details on ARMs with linear transformed NNs is found in Lutter et al. (2019a) ; Lutter et al. (2019b) ; Gupta et al. (2020) ; Cranmer et al. (2020) ; Greydanus et al. (2019) ; Toth et al. (2019) (cf. Section 3.4 ). If the NN outputs parameterize the entries of M M , the positive definiteness of M M poses additional requirements on the last layer of the deep network. Moreover, transforming a NN with partial differential operators requires a careful selection of the activation functions. For example, the Lagrangian NN developed by Cranmer et al. (2020) contains the Hessian ( ∇ q ∇ q ˙ ⊤ ​ ℒ ^ ) \left(\nabla_{q}\nabla_{\dot{q}}^{\top}\hat{\mathcal{L}}\right) of the NN ℒ ^ \hat{\mathcal{L}} . The second-order derivative of a RELU activation with respect to its inputs is zero. Therefore, the authors used and compared RELU 2 , RELU 3 , tanh, sigmoid, and softplus activation functions. For their Lagrangian NN the authors chose the softplus activation function. The usage of higher-order gradient-based optimization methods additionally affects the choice of the activation functions.

 
 
 
 Linearly transformed Gaussian processes 

 
 Jidling et al. (2017) and Lange-Hegermann (2018) detail linear transformations of GPs such that an equality constrained is fulfilled. Additional details on ARMs with linear transformed GPs is found in Cheng et al. (2015) ; Geist and Trimpe (2020) (cf. Section 3.4 ). Gaussian processes are closed under linear operators, such that with f ^ i ∼ 𝒢 ​ 𝒫 ​ ( 𝐦 ⁡ ( x ) , 𝐤 ⁡ ( x , x ′ ) ) \hat{f}_{i}\sim\mathcal{GP}(\mathbf{m}(x),\mathbf{k}(x,x^{\prime})) , ( 81 ) yields

 

 
 | 
 f ^ i 𝒞 ∼ 𝒢 ​ 𝒫 ​ ( ( 𝒞 i ​ ( x ) ​ 𝐦 ​ ( x ) , 𝒞 i ​ ( x ) ​ 𝐤 ​ ( x , x ′ ) ​ 𝒞 i ⊤ ​ ( x ′ ) ) CLOSE , \hat{f}_{i}^{\mathcal{C}}\sim\mathcal{GP}\left((\mathcal{C}_{i}(x)\mathbf{m}(x),\ \mathcal{C}_{i}(x)\mathbf{k}(x,x^{\prime})\mathcal{C}_{i}^{\top}(x^{\prime})\right), | 
 | 
 (85) | 
 

 with the vector-valued mean function 𝐦 ⁡ ( x ) \mathbf{m}(x) and matrix-valued (diagonal) kernel function 𝐤 ⁡ ( x , x ′ ) \mathbf{k}(x,x^{\prime}) . The resulting GP f ^ i 𝒞 \hat{f}_{i}^{\mathcal{C}} is a multi-output GP Álvarez et al. (2012) . Most works that do ARM with GPs, model all output dimensions via independent GPs as the computation of the inverse covariance matrices of n n one-dimensional GPs scales with 𝒪 ⁡ ( n ​ N 3 ) \mathcal{O}\big(nN^{3}\big) . In comparison, the computation of the covariance matrix of an n n -dimensional multitask GP scales with 𝒪 ⁡ ( ( n ​ N ) 3 ) \mathcal{O}\big((nN)^{3}\big) . However, a big advantage of multi-output GPs obtained from ALM modeling is that the transformation 𝒞 i ​ ( x , θ ) \mathcal{C}_{i}(x;\theta) is often known a-priori from analytical mechanics. For example, in case of the Newton-Euler equation q ¨ = M − 1 ​ Q \ddot{q}=M^{-1}Q , we can place a GP prior on Q Q , writing Q ^ ∼ 𝒢 ​ 𝒫 ​ ( 𝐦 ⁡ ( x ) , 𝐤 ⁡ ( x , x ′ ) ) \hat{Q}\sim\mathcal{GP}(\mathbf{m}(x),\mathbf{k}(x,x^{\prime})) , such that

 

 
 | 
 q ¨ ^ ∼ 𝒢 ​ 𝒫 ​ ( ( M − 1 ​ 𝐦 , M − 1 ​ 𝐤 ​ [ M − 1 ] ⊤ ) CLOSE . \hat{\ddot{q}}\sim\mathcal{GP}\left((M^{-1}\mathbf{m},\ M^{-1}\,\mathbf{k}\,[M^{-1}]^{\top}\right). | 
 | 
 (86) | 
 

 In turn, the prior knowledge of M − 1 M^{-1} specifies how the individual dimensions of Q ^ \hat{Q} are correlated with each other to yield q ¨ ^ \hat{\ddot{q}} .
As detailed in a Jidling et al. (2017) and Lange-Hegermann (2018) , a GP can be transformed by a matrix of partial derivatives to yield another GP. For example, for n = 2 n=2 , the matrix of operators

 

 
 | 
 𝒞 x = ( ∇ q ∇ q ˙ ⊤ ) = [ ∂ 2 ∂ q 1 ​ ∂ q ˙ 1 ∂ 2 ∂ q 1 ​ ∂ q ˙ 2 ∂ 2 ∂ q 2 ​ ∂ q ˙ 1 ∂ 2 ∂ q 2 ​ ∂ q ˙ 2 ] \mathcal{C}_{x}=(\nabla_{q}\nabla_{\dot{q}}^{\top})=\begin{bmatrix}\frac{\partial^{2}}{\partial q_{1}\partial\dot{q}_{1}} \frac{\partial^{2}}{\partial q_{1}\partial\dot{q}_{2}}\\
\frac{\partial^{2}}{\partial q_{2}\partial\dot{q}_{1}} \frac{\partial^{2}}{\partial q_{2}\partial\dot{q}_{2}}\end{bmatrix} | 
 | 
 (87) | 
 

 transforms a prior ℒ ^ ∼ 𝒢 ​ 𝒫 ​ ( 𝐦 ⁡ ( x ) , 𝐤 ⁡ ( x , x ′ ) ) \hat{\mathcal{L}}\sim\mathcal{GP}(\mathbf{m}(x),\mathbf{k}(x,x^{\prime})) (cf. Cheng et al. (2015) ) as

 

 
 | 
 OPEN 𝒞 x ​ ℒ ^ ∼ 𝒢 ​ 𝒫 ​ ( 𝒞 x ​ 𝐦 ​ ( x ) , 𝒞 x ​ 𝐤 ​ ( x , x ′ ) ​ 𝒞 x ⊤ ) ) . \mathcal{C}_{x}\hat{\mathcal{L}}\sim\mathcal{GP}(\mathcal{C}_{x}\mathbf{m}(x),\ \mathcal{C}_{x}\mathbf{k}(x,x^{\prime})\mathcal{C}_{x}^{\top})). | 
 | 
 (88) | 
 

 Therefore, if a GP is transformed by a matrix of partial derivatives, 𝐦 ⁡ ( x ) \mathbf{m}(x) and 𝐤 ⁡ ( x , x ′ ) \mathbf{k}(x,x^{\prime}) must be sufficiently often differentiable with respect to the kernel’s inputs. Further, as the sum of GPs are itself a GP (cf. ( Duvenaud, 2014 , p. 13) ), a model for Q ​ = ^ ​ Q G + Q C Q\,\widehat{=}\,Q_{\text{G}}+Q_{\text{C}} can be obtained as

 

 
 | 
 q ^ ∼ 𝒢 ​ 𝒫 ​ ( 𝐦 C + 𝐦 G , 𝐤 C + 𝐤 G ) , \hat{q}\sim\mathcal{GP}(\mathbf{m}_{\text{C}}+\mathbf{m}_{\text{G}},\ \mathbf{k}_{\text{C}}+\mathbf{k}_{\text{G}}\big), | 
 | 
 (89) | 
 

 with Q ^ C ∼ 𝒢 ​ 𝒫 ​ ( 𝐦 C ​ ( x ) , 𝐤 C ​ ( x , x ′ ) ) \hat{Q}_{\text{C}}\sim\mathcal{GP}(\mathbf{m}_{\text{C}}(x),\mathbf{k}_{\text{C}}(x,x^{\prime})) and Q ^ G ∼ 𝒢 ​ 𝒫 ​ ( 𝐦 G ​ ( x ) , 𝐤 G ​ ( x , x ′ ) ) \hat{Q}_{\text{G}}\sim\mathcal{GP}(\mathbf{m}_{\text{G}}(x),\mathbf{k}_{\text{G}}(x,x^{\prime})) .

 
 
 
 
 
 
 e 
 1 
 
 
 
 
 
 
 e 
 2 
 
 
 
 
 
 
 q 
 1 
 
 
 
 
 
 - 
 
 
 q 
 2 
 
 
 
 G 
 
 
 
 f 
 
 
 
 
 
 
 
 G 
 , 
 2 
 
 
 
 
 G 
 
 
 
 f 
 
 
 
 
 
 
 
 G 
 , 
 1 
 
 
 
 
 u 
 
 
 
 Q 
 
 
 
 
 
 
 
 u 
 , 
 1 
 
 
 
 
 u 
 
 
 
 Q 
 
 
 
 
 
 
 
 u 
 , 
 2 
 
 
 
 
 D 
 
 
 
 f 
 D 
 
 
 I 
 
 
 
 f 
 I 
 
 
 
 
 Figure 7 : Sketch of a two-link robot arm who’s end-effector touches a surface. 
 
 
 
 
 

## 5 Case study: Structured modeling of a robot arm

 
 To illustrate the concepts presented in this survey, let us assume that we want to design an analytical structured model for a robot arm whose end-effector is in contact with a surface. We chose this example, as robot arms are widely used mechanical systems that yield complex dynamics functions. Moreover, a robot arm shares similar mechanical characteristics to a robotic leg which is an important component in robotics. While a 2D-system forms an abstraction of a real robot arm, similar modeling assumptions as in the 3D case are made while keeping the discussion concise. This discussion is based on the robot arm equations derived by Siciliano et al. (2010, p. 265-269) . Figure 7 depicts a schematic of the 2D robot arm. Here, Cartesian forces are denoted as f f . The arm orientation is described via the generalized angle coordinates q 1 q_{1} and q 2 q_{2} .
As described in Section 2 , the arm’s rigid-body dynamics model takes the form

 

 
 | 
 M ​ q ¨ = Q C + Q G ⏟ mass-related quantities + Q u + Q d + Q I ⏟ constraint forces . \underbrace{M\ddot{q}=Q_{\text{C}}+Q_{\text{G}}}_{\text{mass-related quantities}}\ +\ Q_{\text{u}}+\underbrace{Q_{\text{d}}+Q_{\text{I}}}_{\text{constraint forces}}. | 
 | 
 (90) | 
 

 In this example, Q d ​ = ^ ​ Q d,I Q_{\text{d}}\,\widehat{=}\,Q_{\text{d,I}} denotes the friction force that is applied by the surface onto the end-effector while Q u Q_{\text{u}} denotes the joint torques.
While for all of these terms, analytical functions have been proposed, we need to ask ourselves if these models form good descriptions of the real system dynamics. As discussed in Section 3 , how we model the analytical functions heavily depends on the system and which quantities we can observe accurately.

 
 
 Defining the analytical parameters 

 
 It is assumed that the COG of the first body lies on an axis of length a 1 a_{1} between the base joint and second joint.
The COM of the second body lies on the axis of length a 2 a_{2} between the second joint and the end-effector tip. The distance from the respective joints to a body’s COM is denoted by L i L_{i} . The mass of each body is denoted by m i m_{i} and the inertia around axis e 3 e_{3} at the COM of the body by I L i I_{L_{i}} .
The robot arm is actuated by motors located in the joints. It is assumed that the COG of the motors is located at the origin of the respective generalized coordinate frames. The mass of the motor’s rotors is m m i m_{m_{i}} and its inertia amounts to I m i I_{m_{i}} . The gear reduction ratio of each motor is k r ​ i k_{ri} .

 
 
 
 Modeling the mass related quantities 

 
 The inertia matrix of the robot arm follows from an analytical derivation as

 

 
 | 
 M ⁡ ( q ) = [ b 11 ​ ( q 2 ) b 12 ​ ( q 2 ) b 21 ​ ( q 2 ) b 22 ] ​  with  ​ b 11 = I L 1 + m L 1 ​ L 1 2 + k r ​ 1 2 ​ I m 1 + I L 2 + m L 2 ​ ( a 1 2 + L 2 2 + 2 ​ a 1 ​ L 2 ​ cos ​ ( q 2 ) ) + I m 2 + m m 2 ​ a 1 2 , b 12 = b 21 = I L 2 + m L 2 ​ ( L 2 2 + a 1 ​ L 2 ​ cos ​ ( q 2 ) ) + k r ​ 2 ​ I m 2 , b 22 = I L 2 + m L 2 ​ L 2 2 + k r ​ 2 2 ​ I m 2 . M(q)=\left[\begin{array}[]{ll}b_{11}\left(q_{2}\right) b_{12}\left(q_{2}\right)\\
b_{21}\left(q_{2}\right) b_{22}\end{array}\right]\text{ with }\ \begin{aligned} b_{11}= I_{L_{1}}+m_{L_{1}}L_{1}^{2}+k_{r1}^{2}I_{m_{1}}+I_{L_{2}}\\
 +m_{L_{2}}\left(a_{1}^{2}+L_{2}^{2}+2a_{1}L_{2}\text{cos}(q_{2})\right)+I_{m_{2}}+m_{m_{2}}a_{1}^{2},\\
b_{12}= b_{21}=I_{L_{2}}+m_{L_{2}}\left(L_{2}^{2}+a_{1}L_{2}\text{cos}(q_{2})\right)+k_{r2}I_{m_{2}},\\
b_{22}= I_{L_{2}}+m_{L_{2}}L_{2}^{2}+k_{r2}^{2}I_{m_{2}}.\end{aligned} | 
 | 
 (91) | 
 

 The fictitious force follows from ( 33 ) as

 

 
 | 
 Q C = − ∇ q ( q ˙ ⊤ ​ M ) ​ q ˙ + 1 2 ​ ∇ q ( q ˙ T ​ M ​ q ˙ ) = − C ⁡ ( q , q ˙ ) ​ q ˙ ,  with  ​ C ​ ( q , q ˙ ) = ∇ q ( q ˙ ⊤ ​ M ) = [ h ​ q ˙ 2 h ⁡ ( q ˙ 1 + q ˙ 2 ) − h ​ q ˙ 1 0 ] , Q_{\text{C}}=-\nabla_{q}(\dot{q}^{\top}M)\dot{q}+\frac{1}{2}\nabla_{q}\left(\dot{q}^{T}M\dot{q}\right)=-C(q,\dot{q})\dot{q},\text{ with }\ C(q,\dot{q})=\nabla_{q}(\dot{q}^{\top}M)=\left[\begin{array}[]{cc}h\dot{q}_{2} h\left(\dot{q}_{1}+\dot{q}_{2}\right)\\
-h\dot{q}_{1} 0\end{array}\right], | 
 | 
 (92) | 
 

 while the model for the gravitational force reads

 

 
 | 
 Q G = [ ( m ℓ 1 ​ ℓ 1 + m m 2 ​ a 1 + m ℓ 2 ​ a 1 ) ​ g ​ cos ​ ( q 1 ) + m ℓ 2 ​ ℓ 2 ​ g ​ cos ​ ( q 1 + q 2 ) m ℓ 2 ​ ℓ 2 ​ g ​ cos ​ ( q 1 + q 2 ) ] , Q_{\text{G}}=\left[\begin{array}[]{l}\left(m_{\ell_{1}}\ell_{1}+m_{m_{2}}a_{1}+m_{\ell_{2}}a_{1}\right)g\text{cos}(q_{1})+m_{\ell_{2}}\ell_{2}g\text{cos}(q_{1}+q_{2})\\
m_{\ell_{2}}\ell_{2}g\text{cos}(q_{1}+q_{2})\end{array}\right], | 
 | 
 (93) | 
 

 with h = − m L 2 ​ a 1 ​ L 2 ​ sin ⁡ ( q 2 ) h=-m_{L_{2}}a_{1}L_{2}\sin(q_{2}) .
To simplify this derivation, several assumptions were made which are also typically found in other analytical models, namely:

 
 • 
 
 The COM of the bodies as well as of the motor’s rotors lies on a predefined axis.

 

 • 
 
 Elasticities of bodies and compliance inside the joints are not modeled.

 

 
 The latter point cannot be avoided using rigid-body modeling and hence can contribute to ϵ R \epsilon_{\text{R}} (cf. Section 3 ). To reduce ϵ R \epsilon_{\text{R}} we can use ARM. However, learning these errors is difficult as we have only limited sensory information, e.g. , measurements of q q , q ˙ \dot{q} , and Q u Q_{\text{u}} ( q ¨ \ddot{q} is often estimated by filtering measurements of q ˙ \dot{q} ( Siciliano et al., 2010 , p. 281) ). If we do not want to model ϵ R \epsilon_{\text{R}} solely as a noise process, one can either take the history of the dynamics into account to learn on a streak of data points as done in Hwangbo et al. (2019) or add additional sensors that capture information about the elasticities.

 
 
 Errors that stem from a wrong description of the bodies COM positions should ideally be avoided by improving the analytical model itself. It is tempting to directly resort to the data-driven methods presented in Section ( 3.4.1 ) which model M M , Q G Q_{\text{G}} , and Q C Q_{\text{C}} via a NNs or GPs. However, for many mechanical systems, an accurate depiction of M M , Q G Q_{\text{G}} , and Q C Q_{\text{C}} can be obtained from analytical modeling. The question is how much time and expertise are we willing to invest to obtain a sufficient analytical description of M M , Q G Q_{\text{G}} , and Q C Q_{\text{C}} and in return potentially improve the sample-efficiency of an analytical structured model.

 
 
 
 Modeling the joint torques 

 
 The identification of the non-conservative forces constitutes a big challenge in robotic systems. As discussed in Section 3 , how the joint torques Q u Q_{\text{u}} are modeled depends on how they are measured.

 
 
 If the joint torques are directly measured through joint-torque sensors, one can directly learn the function Q u Q_{\text{u}} . However, while theoretically, it should be possible to learn Q u Q_{\text{u}} from data of { q , q ˙ , q ¨ } \{q,\dot{q},\ddot{q}\} , the acceleration q ¨ \ddot{q} is usually not directly observed and is therefore considerably noisy. In return, we suggest learning Q u Q_{\text{u}} using the techniques discussed in Section 3.4.2 . For example, Q u Q_{\text{u}} can be directly learned using a NN that has as input a sequence of position errors and velocities or Q u,desired Q_{\text{u,desired}} .

 
 
 Alternatively, if no accurate measurements of Q u Q_{\text{u}} are available one can design a structured inverse dynamics model as discussed in Section 3.3 . However, this model will learn the mapping from the current state-acceleration observation to the measured Q u Q_{\text{u}} . In return, such a model is not necessarily suited for a simulation of the joint torques. A second approach is to directly learn the forward dynamics via ALM. By doing so, we need to only specify a model for Q u Q_{\text{u}} .

 
 
 
 Modeling the implicit constraint forces 

 
 The identification of Q d Q_{\text{d}} and Q I Q_{\text{I}} is also challenging as Q d Q_{\text{d}} is highly environmental dependent and Q I Q_{\text{I}} is a contact force. We observed two different philosophies on how to model these forces in robotic systems.

 
 
 The first approach, which Lee et al. (2020) used to achieve seminal results for the control a quadruped, is to directly measure and identify Q u Q_{\text{u}} . Then we can insert the model for Q u Q_{\text{u}} in a rigid-body dynamics simulation where we can simulate different Q d Q_{\text{d}} and Q I Q_{\text{I}} . In return, one can design or learn a control strategy that yields robust performance over a large set of different Q d Q_{\text{d}} . However, even in this scenario, it can desirable to identify a specific Q d Q_{\text{d}} as the robustness of a specific control policy to different dissipative forces can come at the cost of losing control performance.

 
 
 The second approach, aims at modeling Q d Q_{\text{d}} inside the inverse or forward dynamics of the system. As shown in Section 2.3 one can include constrained knowledge to eliminate Q I Q_{\text{I}} from the dynamics equations. For example, assuming that the robot arm’s end-effector is pressing on a flat surface such that the endeffector’s Cartesian coordinate in e 1 e_{1} direction denotes r e , 1 = 0 r_{e,1}=0 . Then, this surface is described by an implicit constraint equation as

 

 
 | 
 0 \displaystyle 0 | 
 = r e , 1 = a 1 ​ cos ​ ( q 1 ) + a 2 ​ cos ​ ( q 1 + q 2 ) . \displaystyle=r_{e,1}=a_{1}\text{cos}(q_{1})+a_{2}\text{cos}(q_{1}+q_{2}). | 
 | 
 (94) | 
 

 The the second time-derivative of ( 94 ) yields the constraining equation ( 38 ) as

 

 
 | 
 [ − a 1 ​ sin ​ ( q 1 ) − a 2 ​ sin ​ ( q 1 + q 2 ) , − a 2 ​ sin ​ ( q 1 + q 2 ) ] ​ q ¨ \displaystyle\left[-a_{1}\text{sin}(q_{1})-a_{2}\text{sin}(q_{1}+q_{2}),\ -a_{2}\text{sin}(q_{1}+q_{2})\right]\ddot{q} | 
 = − a 1 ​ cos ​ ( q 1 ) ​ q ˙ 1 2 − a 2 ​ cos ​ ( q 1 + q 2 ) ​ ( q ˙ 1 + q ˙ 2 ) 2 . \displaystyle=-a_{1}\text{cos}(q_{1})\dot{q}_{1}^{2}-a_{2}\text{cos}(q_{1}+q_{2})(\dot{q}_{1}+\dot{q}_{2})^{2}. | 
 | 
 (95) | 
 

 This equation can be used to include constraint knowledge into an analytical structured model as detailed in Section 3.3 and Section 3.4.3 .

 
 
 
 Constructing the structured model and training 

 
 Now we assume that all mechanical functions and error functions of the dynamics model are either modeled analytically or as a data-driven model. As discussed in Section 4.1 , we suggest that these functions are then expressed in the programming language of a suitable automatic differentiation library to yield the analytical structured model. After the specification of the objective function, one can estimate the unidentified parameters of the structured model using collected data.

 
 
 
 

## 6 Conclusion and future outlook

 
 Classical mechanics provides us with an understanding of the structural relationships underlying the motion of mechanical systems. One such insight is that motion is a consequence of the interaction of mass and force. Yet, to predict motion we need to describe the effects of mass and force mathematically. This turns out to be problematic in practice as the distribution of mass, as well as the applied forces, are itself the product of convoluted physical phenomena that are difficult, if not impossible, to describe a-priori. To address this and similar bottlenecks in a mathematical description of the world, we forged tools to learn functions from data. However, using data-driven learning methods comes at the price of requiring significant amounts of data as well as losing insight on how a prediction of such a model came about.

 
 
 In this work, we discussed the different sources of structural knowledge inherent in analytical descriptions of rigid-body mechanics (Sec. 2). Here, we place special emphasis on the different type of forces, potential functions, and explicit and implicit constraint equations.

 
 
 Further, we propose a unified view on structured modeling by emphasizing that output and latent modeling form key building blocks for analytical structured models (Sec. 3). In turn, we distinguish learning the residuals of an analytical model in the model’s output, which is referred to as analytical (output) residual modeling (ARM), and analytical latent modeling (ALM), in which a data-driven model approximates latent residuals or functions inside the analytical model. To this effect, we consolidated the proposed view by illustrating that unknown analytical latent functions are often transformed by a known linear operator, which advocates the usage of ALM.

 
 
 In addition, our review of recent literature on analytical structured modelling (Sec. 3 ), as well as related aspects of a model synthesis (Sec. 4 ), revealed a shift in the optimization techniques used in works on structured modeling. Namely, we observed a shift from analytical models that are optimized via linear regression towards structured latent models that are optimized via automatic differentiation. We believe that this shift is due to recent improvements in automatic differentiation libraries that significantly ease the synthesis of analytical structured models.

 
 
 Recently, an analytical structured model Lee et al. (2020) enabled the synthesis of a quadruped robot’s control policy with so far unseen and ineffable robustness to changes in the environment and carrying-load. We believe that this and the other works presented in this survey form merely the beginning of a broad application of analytical structured models in wide ranges of engineering, and we are excited to see what applications these promising class of models will enable in the near future.

   
 
 As analytical structured modeling is a relatively recent field, many open questions on the design of these models remain. Below we provide a list of what we perceive as potential directions that require further analysis in the field of analytical structured modeling, but that were beyond the scope of this survey.

 
 
 Further types of rigid-body dynamics 

 
 Many of the discussed works use the recursive Newton-Euler algorithm as thoroughly discussed in Featherstone (2008) . However, some works also resorted to alternative dynamics descriptions such as the Newton-Euler equation in Lie algebra formulation Kim (2012) or the Udwadia-Kalaba equation Udwadia and Kalaba (2002) . It seems that the recent works on analytical structured learning only touched the surface of the vast field of rigid-body simulations and it is worth further discussion which descriptions of motion are particularly suited for the combination with data-driven learning techniques.

 
 
 
 Partially observable systems 

 
 There exist numerous works on data-driven modeling of partially observable systems Lee et al. (2007) ; Doerr et al. (2018) ; Curi et al. (2020) . Yet, works on analytical structured learning do typically not assume that the state is not directly observable and only partial state information is given via the observation equation o k = h ⁡ ( q T , q ˙ T , u k , ϵ o ) o_{k}=h(q^{T},\dot{q}^{T},u_{k},\epsilon_{o}) . Here h ⁡ ( q T , q ˙ T , u k , ϵ o ) h(q^{T},\dot{q}^{T},u_{k},\epsilon_{o}) most often denotes a simple algebraic equation and ϵ o \epsilon_{o} an additional noise process. The question remains if the existing insights from data-driven modeling can be extended to analytical structured modeling of partially observable systems.

 
 
 
 Noise on model inputs 

 
 The observations of the input variables x x are often noisy and biased. In dynamics modeling this is often not taken into account during training. The question arises if input noise can and should be addressed. Notably, some works propagate input noise through the trained dynamics model when doing predictions Deisenroth and Rasmussen (2011) ; Parmas et al. (2018) .

 
 
 
 Active learning 

 
 We assumed that informative system data is given. However, the collection of informative data from dynamical systems is itself a subject of ongoing research efforts ( Schön et al., 2011 ; Simchowitz et al., 2018 ; Schoukens and Ljung, 2019 ; Buisson-Fenet et al., 2020 ) . The question arises as to which extend structural knowledge can be used to make data-collection more efficient as well as how to use this data to train analytical structured models.

 
 
 
 Interpretable data-driven models 

 
 While data-driven models are often referred to as black-box models, there are efforts made to provide guaranties for data-driven model predictions ( Umlauft et al., 2017 ; Beckers et al., 2017 ) as well as carefully designing data-driven models to make them more interpretable ( Ahmadi and Khadir, 2020 ) . These works are critical if the application of a data-driven model requires prediction guarantees.

 
 
 
 Combination of forward dynamics models with numerical integration 

 
 As previously mentioned, a disadvantage of analytical structured learning of forward dynamics forms the noise on the acceleration estimates. Here, the forward dynamics are usually modeled to obtain a trajectory prediction using numerical integration techniques. An open question is to which extend analytical structured models can be combined with numerical integration schemes such that training of such a combined model is done solely on trajectory data. Recent works discuss how to approximate ODEs using GPs Heinonen et al. (2018) ; Yildiz et al. (2018) ; Wenk et al. (2020) and NNs Chen et al. (2018) ; Li et al. (2020) as well as solve initial-value problems using GPs Schober et al. (2014) ; Schober et al. (2019) .

 
 
 

### Acknowledgments

 
 We thank Lucas Rath, Steve Heim, Alexander von Rohr, and Christian Fiedler for the valuable feedback and many interesting discussions. This work has been supported in part by the Max Planck Society and in part by the Cyber Valley initiative. The authors thank the International Max Planck Research School for Intelligent Systems (IMPRS-IS) for supporting A. René Geist.

 
 
 
 

## References

 
 
 Allgöwer and Zheng (2012) 
 
Frank Allgöwer and Alex Zheng.

 
 Nonlinear model predictive control , volume 26.

 
 Birkhäuser, 2012.

 

 
 Sutton et al. (1998) 
 
Richard S Sutton, Andrew G Barto, et al.

 
 Introduction to reinforcement learning , volume 135.

 
 MIT press Cambridge, 1998.

 

 
 Nelles (2013) 
 
Oliver Nelles.

 
 Nonlinear system identification: from classical approaches to
neural networks and fuzzy models .

 
 Springer Science Business Media, 2013.

 

 
 Nguyen-Tuong and Peters (2010) 
 
Duy Nguyen-Tuong and Jan Peters.

 
 Using model knowledge for learning inverse dynamics.

 
 In International Conference on Robotics and Automation (ICRA) ,
pages 2677–2682. IEEE, 2010.

 

 
 Sutanto et al. (2020) 
 
Giovanni Sutanto, Austin Wang, Yixin Lin, Mustafa Mukadam, Gaurav Sukhatme,
Akshara Rai, and Franziska Meier.

 
 Encoding physical constraints in differentiable newton-euler
algorithm.

 
 In Proceedings of the 2nd Conference on Learning for Dynamics
and Control , volume 120 of Proceedings of Machine Learning Research ,
pages 804–813, The Cloud, 10–11 Jun 2020. PMLR.

 

 
 Hwangbo et al. (2019) 
 
Jemin Hwangbo, Joonho Lee, Alexey Dosovitskiy, Dario Bellicoso, Vassilios
Tsounis, Vladlen Koltun, and Marco Hutter.

 
 Learning agile and dynamic motor skills for legged robots.

 
 Science Robotics , 4(26), 2019.

 

 
 Ledezma and Haddadin (2017) 
 
Fernando Díaz Ledezma and Sami Haddadin.

 
 First-order-principles-based constructive network topologies: An
application to robot inverse dynamics.

 
 In 2017 IEEE-RAS 17th International Conference on Humanoid
Robotics (Humanoids) , pages 438–445. IEEE, 2017.

 

 
 Bradbury et al. (2018) 
 
James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary,
Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye
Wanderman-Milne, and Qiao Zhang.

 
 JAX: composable transformations of Python+NumPy programs,
2018.

 
 URL http://github.com/google/jax .

 

 
 Paszke et al. (2019) 
 
Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory
Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban
Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan
Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith
Chintala.

 
 Pytorch: An imperative style, high-performance deep learning library.

 
 In Advances in Neural Information Processing Systems 32 , pages
8024–8035. Curran Associates, Inc., 2019.

 
 URL
 http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf .

 

 
 Kane and Levinson (1985) 
 
Thomas R Kane and David A Levinson.

 
 Dynamics, theory and applications .

 
 McGraw Hill, 1985.

 

 
 Featherstone (2008) 
 
Roy Featherstone.

 
 Rigid body dynamics algorithms .

 
 Springer, 2008.

 

 
 Davis and Rabinowitz (2007) 
 
Philip J Davis and Philip Rabinowitz.

 
 Methods of numerical integration .

 
 Courier Corporation, 2007.

 

 
 Jordan (1992) 
 
Michael I Jordan.

 
 Constrained supervised learning.

 
 Journal of Mathematical Psychology , 36(3):396–425, 1992.

 

 
 Atkeson et al. (1986) 
 
Christopher G Atkeson, Chae H An, and John M Hollerbach.

 
 Estimation of inertial parameters of manipulator loads and links.

 
 The International Journal of Robotics Research , 5(3):101–119, 1986.

 

 
 Ting et al. (2006) 
 
Jo-Anne Ting, Michael N Mistry, Jan Peters, Stefan Schaal, and Jun Nakanishi.

 
 A bayesian approach to nonlinear parameter identification for rigid
body dynamics.

 
 In Robotics: Science and Systems , pages 32–39. Philadelphia,
USA, 2006.

 

 
 Traversaro et al. (2016) 
 
Silvio Traversaro, Stanislas Brossette, Adrien Escande, and Francesco Nori.

 
 Identification of fully physical consistent inertial parameters using
optimization on manifolds.

 
 In 2016 IEEE/RSJ International Conference on Intelligent Robots
and Systems (IROS) , pages 5446–5451. IEEE, 2016.

 

 
 Wensing et al. (2017) 
 
Patrick M Wensing, Sangbae Kim, and Jean-Jacques E Slotine.

 
 Linear matrix inequalities for physically consistent inertial
parameter identification: A statistical perspective on the mass distribution.

 
 IEEE Robotics and Automation Letters , 3(1):60–67, 2017.

 

 
 Ledezma and Haddadin (2018) 
 
Fernando Díaz Ledezma and Sami Haddadin.

 
 Fop networks for learning humanoid body schema and dynamics.

 
 In 2018 IEEE-RAS 18th International Conference on Humanoid
Robots (Humanoids) , pages 1–9. IEEE, 2018.

 

 
 Sahoo et al. (2018) 
 
Subham S. Sahoo, Christoph H. Lampert, and Georg Martius.

 
 Learning equations for extrapolation and control.

 
 In Proc. 35th International Conference on Machine Learning,
ICML 2018, Stockholm, Sweden, 2018 , volume 80, pages 4442–4450. PMLR,
2018.

 
 URL http://proceedings.mlr.press/v80/sahoo18a.html .

 

 
 Baumann et al. (2020) 
 
Dominik Baumann, Friedrich Solowjow, Karl H Johansson, and Sebastian Trimpe.

 
 Identifying causal structure in dynamical systems.

 
 arXiv preprint arXiv:2006.03906 , 2020.

 

 
 Lennart (1999) 
 
Ljung Lennart.

 
 System identification: theory for the user.

 
 PTR Prentice Hall, Upper Saddle River, NJ , pages 1–14, 1999.

 

 
 Lutter et al. (2019a) 
 
M. Lutter, C. Ritter, and J. Peters.

 
 Deep lagrangian networks: Using physics as model prior for deep
learning.

 
 May 2019a.

 

 
 Gupta et al. (2020) 
 
Jayesh K. Gupta, Kunal Menda, Zachary Manchester, and Mykel Kochenderfer.

 
 Structured mechanical models for robot learning and control.

 
 In Proceedings of the 2nd Conference on Learning for Dynamics
and Control , volume 120 of Proceedings of Machine Learning Research ,
pages 328–337, The Cloud, 10–11 Jun 2020. PMLR.

 

 
 Geist and Trimpe (2020) 
 
Andreas Geist and Sebastian Trimpe.

 
 Learning constrained dynamics with gauss’ principle adhering
gaussian processes.

 
 In Proceedings of the 2nd Conference on Learning for Dynamics
and Control , volume 120 of Proceedings of Machine Learning Research ,
pages 225–234. PMLR, 10–11 Jun 2020.

 

 
 Beard (2002) 
 
Randal W Beard.

 
 Linear operator equations with applications in control and signal
processing.

 
 IEEE Control Systems Magazine , 22(2):69–79, 2002.

 

 
 Siciliano et al. (2010) 
 
Bruno Siciliano, Lorenzo Sciavicco, Luigi Villani, and Giuseppe Oriolo.

 
 Robotics: modelling, planning and control .

 
 Springer Science Business Media, 2010.

 

 
 Schiehlen and Eberhard (2014) 
 
Werner Schiehlen and Peter Eberhard.

 
 Applied dynamics , volume 57.

 
 Springer, 2014.

 

 
 (28) 
 
C Woernle.

 
 Mehrkörpersysteme: Eine einführung in die kinematik und
dynamik von systemen starrer körper. 2016.

 

 
 Udwadia and Kalaba (2007) 
 
Firdaus E Udwadia and Robert Kalaba.

 
 Analytical dynamics: a new approach .

 
 Cambridge University Press, 2007.

 

 
 d’Alembert (1743) 
 
Jean Le Rond d’Alembert.

 
 Traité de dynamique .

 
 1743.

 

 
 Lagrange (1787) 
 
Joseph Louis Lagrange.

 
 Méchanique analytique .

 
 Mme. De Courcier, Paris, 1787.

 

 
 Layton (2012) 
 
Richard A Layton.

 
 Principles of analytical system dynamics .

 
 Springer Science Business Media, 2012.

 

 
 Grimminger et al. (2020) 
 
F. Grimminger, A. Meduri, M. Khadiv, J. Viereck, M. Wüthrich,
M. Naveau, V. Berenz, S. Heim, F. Widmaier, T. Flayols, J. Fiene,
A. Badri-Spröwitz, and L. Righetti.

 
 An open torque-controlled modular robot architecture for legged
locomotion research.

 
 IEEE Robotics and Automation Letters , 5(2):3650–3657, 2020.

 
 10.1109/LRA.2020.2976639 .

 

 
 Kumar (2019) 
 
Shivesh Kumar.

 
 Modular and Analytical Methods for Solving Kinematics and
Dynamics of Series-Parallel Hybrid Robots .

 
 PhD thesis, Universität Bremen, 2019.

 

 
 Koganti and Udwadia (2016) 
 
Prasanth B Koganti and Firdaus E Udwadia.

 
 Unified approach to modeling and control of rigid multibody systems.

 
 Journal of Guidance, Control, and Dynamics , 39(12):2683–2698, 2016.

 

 
 Udwadia et al. (1997) 
 
Firdaus E Udwadia, Robert E Kalaba, and Hee-Chang Eun.

 
 Equations of motion for constrained mechanical systems and the
extended d’alembert’s principle.

 
 Quarterly of Applied Mathematics , 55(2):321–331, 1997.

 

 
 Udwadia and Kalaba (2002) 
 
Firdaus E Udwadia and Robert E Kalaba.

 
 On the foundations of analytical dynamics.

 
 International Journal of non-linear mechanics , 37(6):1079–1090, 2002.

 

 
 Udwadia and Kalaba (2000) 
 
Firdaus E Udwadia and Robert E Kalaba.

 
 Nonideal constraints and lagrangian dynamics.

 
 Journal of Aerospace Engineering , 13(1):17–22, 2000.

 

 
 Aghili (2005) 
 
Farhad Aghili.

 
 A unified approach for inverse and direct dynamics of constrained
multibody systems based on linear projection operator: applications to
control and simulation.

 
 IEEE Transactions on Robotics , 21(5):834–849, 2005.

 

 
 Udwadia and Phohomsiri (2006) 
 
Firdaus E Udwadia and Phailaung Phohomsiri.

 
 Explicit equations of motion for constrained mechanical systems with
singular mass matrices and applications to multi-body dynamics.

 
 Proceedings of the Royal Society A: Mathematical, Physical and
Engineering Sciences , 462(2071):2097–2117, 2006.

 

 
 Peters et al. (2008) 
 
Jan Peters, Michael Mistry, Firdaus Udwadia, Jun Nakanishi, and Stefan Schaal.

 
 A unifying framework for robot control with redundant dofs.

 
 Autonomous Robots , 24(1):1–12, 2008.

 

 
 Udwadia and Kalaba (1992) 
 
Firdaus E Udwadia and Robert E Kalaba.

 
 A new perspective on constrained motion.

 
 Proceedings of the Royal Society of London. Series A:
Mathematical and Physical Sciences , 439(1906):407–410,
1992.

 

 
 Gauß (1829) 
 
Carl Friedrich Gauß.

 
 Über ein neues allgemeines grundgesetz der mechanik.

 
 Journal für die reine und angewandte Mathematik ,
4:232–235, 1829.

 

 
 Von Luxburg and Schölkopf (2011) 
 
Ulrike Von Luxburg and Bernhard Schölkopf.

 
 Statistical learning theory: Models, concepts, and results.

 
 In Handbook of the History of Logic , volume 10, pages
651–706. Elsevier, 2011.

 

 
 Nguyen-Tuong and Peters (2011) 
 
Duy Nguyen-Tuong and Jan Peters.

 
 Model learning for robot control: a survey.

 
 Cognitive processing , 12(4):319–340,
2011.

 

 
 Krizhevsky et al. (2012) 
 
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton.

 
 Imagenet classification with deep convolutional neural networks.

 
 In Advances in neural information processing systems
(NeurIPS) , pages 1097–1105, 2012.

 

 
 Silver et al. (2016) 
 
David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George
Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda
Panneershelvam, Marc Lanctot, et al.

 
 Mastering the game of go with deep neural networks and tree search.

 
 nature , 529(7587):484–489, 2016.

 

 
 Berner et al. (2019) 
 
Christopher Berner, Greg Brockman, Brooke Chan, Vicki Cheung, Przemysław
Dębiak, Christy Dennison, David Farhi, Quirin Fischer, Shariq Hashme,
Chris Hesse, et al.

 
 Dota 2 with large scale deep reinforcement learning.

 
 arXiv preprint arXiv:1912.06680 , 2019.

 

 
 Vinyals et al. (2019) 
 
Oriol Vinyals, Igor Babuschkin, Junyoung Chung, Michael Mathieu, Max Jaderberg,
Wojciech M Czarnecki, Andrew Dudzik, Aja Huang, Petko Georgiev, Richard
Powell, et al.

 
 Alphastar: Mastering the real-time strategy game starcraft ii.

 
 DeepMind blog , page 2, 2019.

 

 
 Funahashi and Nakamura (1993) 
 
Ken-ichi Funahashi and Yuichi Nakamura.

 
 Approximation of dynamical systems by continuous time recurrent
neural networks.

 
 Neural networks , 6(6):801–806, 1993.

 

 
 Kuschewski et al. (1993) 
 
John G Kuschewski, Stefen Hui, and Stanislaw H Zak.

 
 Application of feedforward neural networks to dynamical system
identification and control.

 
 IEEE Transactions on Control Systems Technology , 1(1):37–49, 1993.

 

 
 Jansen (1994) 
 
M Jansen.

 
 Learning an accurate neural model of the dynamics of a typical
industrial robot.

 
 In Int. Conf. on Artificial Neural Networks , pages 1257–1260,
1994.

 

 
 Morton et al. (2018) 
 
Jeremy Morton, Antony Jameson, Mykel J Kochenderfer, and Freddie Witherden.

 
 Deep dynamical modeling and control of unsteady fluid flows.

 
 In Advances in Neural Information Processing Systems
(NeurIPS) , pages 9258–9268, 2018.

 

 
 Chua et al. (2018) 
 
Kurtland Chua, Roberto Calandra, Rowan McAllister, and Sergey Levine.

 
 Deep reinforcement learning in a handful of trials using
probabilistic dynamics models.

 
 In Advances in Neural Information Processing Systems
(NeurIPS) , pages 4754–4765, 2018.

 

 
 Nagabandi et al. (2020) 
 
Anusha Nagabandi, Kurt Konolige, Sergey Levine, and Vikash Kumar.

 
 Deep dynamics models for learning dexterous manipulation.

 
 In Conference on Robot Learning , pages 1101–1112, 2020.

 

 
 Alvarez et al. (2011) 
 
Mauricio A Alvarez, Lorenzo Rosasco, and Neil D Lawrence.

 
 Kernels for vector-valued functions: A review.

 
 arXiv preprint arXiv:1106.6251 , 2011.

 

 
 Deisenroth and Rasmussen (2011) 
 
Marc Deisenroth and Carl E Rasmussen.

 
 Pilco: A model-based and data-efficient approach to policy search.

 
 In Proceedings of the 28th International Conference on machine
learning (ICML-11) , pages 465–472, 2011.

 

 
 Kocijan et al. (2005) 
 
Juš Kocijan, Agathe Girard, Blaž Banko, and Roderick Murray-Smith.

 
 Dynamic systems identification with Gaussian processes.

 
 Mathematical and Computer Modelling of Dynamical Systems ,
11(4):411–424, 2005.

 

 
 Frigola et al. (2013) 
 
Roger Frigola, Fredrik Lindsten, Thomas B Schön, and Carl Edward Rasmussen.

 
 Bayesian inference and learning in Gaussian process state-space
models with particle MCMC.

 
 In Advances in Neural Information Processing Systems
(NeurIPS) , pages 3156–3164, 2013.

 

 
 Mattos et al. (2016) 
 
César Lincoln C Mattos, Andreas Damianou, Guilherme A Barreto, and Neil D
Lawrence.

 
 Latent autoregressive Gaussian processes models for robust system
identification.

 
 IFAC-PapersOnLine , 49(7):1121–1126, 2016.

 

 
 Doerr et al. (2017) 
 
Andreas Doerr, Christian Daniel, Duy Nguyen-Tuong, Alonso Marco, Stefan Schaal,
Marc Toussaint, and Sebastian Trimpe.

 
 Optimizing long-term predictions for model-based policy search.

 
 In Conference on Robot Learning , pages 227–238, 2017.

 

 
 Eleftheriadis et al. (2017) 
 
Stefanos Eleftheriadis, Tom Nicholson, Marc Deisenroth, and James Hensman.

 
 Identification of Gaussian process state space models.

 
 In Advances in neural information processing systems
(NeurIPS) , pages 5309–5319, 2017.

 

 
 Doerr et al. (2018) 
 
Andreas Doerr, Christian Daniel, Martin Schiegg, Nguyen-Tuong Duy, Stefan
Schaal, Marc Toussaint, and Trimpe Sebastian.

 
 Probabilistic recurrent state-space models.

 
 In Proceedings of the 35th International Conference on Machine
Learning , volume 80 of Proceedings of Machine Learning Research ,
pages 1280–1289, Stockholmsmässan, Stockholm Sweden, 10–15 Jul 2018. PMLR.

 

 
 Quiñonero-Candela and Rasmussen (2005) 
 
Joaquin Quiñonero-Candela and Carl Edward Rasmussen.

 
 A unifying view of sparse approximate gaussian process regression.

 
 Journal of Machine Learning Research , 6(Dec):1939–1959, 2005.

 

 
 An et al. (1985) 
 
Chae H An, Christopher G Atkeson, and John M Hollerbach.

 
 Estimation of inertial parameters of rigid body links of
manipulators.

 
 In 1985 24th IEEE Conference on Decision and Control , pages
990–995. IEEE, 1985.

 

 
 Luh et al. (1980) 
 
John YS Luh, Michael W Walker, and Richard PC Paul.

 
 On-line computational scheme for mechanical manipulators.

 
 1980.

 

 
 De La Cruz et al. (2011) 
 
Joseph Sun De La Cruz, Dana Kulić, and William Owen.

 
 Online incremental learning of inverse dynamics incorporating prior
knowledge.

 
 In International Conference on Autonomous and Intelligent
Systems , pages 167–176. Springer, 2011.

 

 
 Um et al. (2014) 
 
Terry Taewoong Um, Myoung Soo Park, and Jung-Min Park.

 
 Independent joint learning: A novel task-to-task transfer learning
scheme for robot models.

 
 In 2014 IEEE International Conference on Robotics and
Automation (ICRA) , pages 5679–5684. IEEE, 2014.

 

 
 Grandia et al. (2018) 
 
Ruben Grandia, Diego Pardo, and Jonas Buchli.

 
 Contact invariant model learning for legged robot locomotion.

 
 IEEE Robotics and Automation Letters , 3(3):2291–2298, 2018.

 

 
 Cheng et al. (2015) 
 
Ching-An Cheng, Han-Pang Huang, Huan-Kun Hsu, Wei-Zh Lai, and Chih-Chun Cheng.

 
 Learning the inverse dynamics of robotic manipulators in structured
reproducing kernel hilbert space.

 
 IEEE transactions on cybernetics , 46(7):1691–1703, 2015.

 

 
 Hwangbo et al. (2018) 
 
Jemin Hwangbo, Joonho Lee, and Marco Hutter.

 
 Per-contact iteration method for solving contact dynamics.

 
 IEEE Robotics and Automation Letters , 3(2):895–902, 2018.

 

 
 Greydanus et al. (2019) 
 
Samuel Greydanus, Misko Dzamba, and Jason Yosinski.

 
 Hamiltonian neural networks.

 
 In Advances in Neural Information Processing Systems
(NeurIPS) , pages 15379–15389, 2019.

 

 
 Lutter et al. (2019b) 
 
Michael Lutter, Kim Listmann, and Jan Peters.

 
 Deep lagrangian networks for end-to-end learning of energy-based
control for under-actuated systems.

 
 In 2019 IEEE/RSJ International Conference on Intelligent Robots
and Systems (IROS) , pages 7718–7725. IEEE, 2019b.

 

 
 Lutter et al. (2020) 
 
Michael Lutter, Johannes Silberbauer, Joe Watson, and Jan Peters.

 
 A differentiable newton euler algorithm for multi-body model
learning.

 
 arXiv preprint arXiv:2010.09802 , 2020.

 

 
 Kim (2012) 
 
Junggon Kim.

 
 Lie group formulation of articulated rigid body dynamics.

 
 Technical report, Technical Report. Carnegie Mellon University, 2012.

 

 
 Toth et al. (2019) 
 
Peter Toth, Danilo J Rezende, Andrew Jaegle, Sébastien Racanière,
Aleksandar Botev, and Irina Higgins.

 
 Hamiltonian generative networks.

 
 In International Conference on Learning Representations , 2019.

 

 
 Cranmer et al. (2020) 
 
Miles Cranmer, Sam Greydanus, Stephan Hoyer, Peter Battaglia, David Spergel,
and Shirley Ho.

 
 Lagrangian neural networks.

 
 In ICLR 2020 Workshop on Integration of Deep Neural Models and
Differential Equations , 2020.

 

 
 Williams and Rasmussen (2006) 
 
Christopher KI Williams and Carl Edward Rasmussen.

 
 Gaussian processes for machine learning , volume 2.

 
 MIT press Cambridge, MA, 2006.

 

 
 Camoriano et al. (2016) 
 
Raffaello Camoriano, Silvio Traversaro, Lorenzo Rosasco, Giorgio Metta, and
Francesco Nori.

 
 Incremental semiparametric inverse dynamics learning.

 
 In 2016 IEEE International Conference on Robotics and
Automation (ICRA) , pages 544–550. IEEE, 2016.

 

 
 Chen et al. (2018) 
 
Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud.

 
 Neural ordinary differential equations.

 
 In Advances in neural information processing systems
(NeurIPS) , pages 6571–6583, 2018.

 

 
 Lee et al. (2020) 
 
Joonho Lee, Jemin Hwangbo, Lorenz Wellhausen, Vladlen Koltun, and Marco Hutter.

 
 Learning quadrupedal locomotion over challenging terrain.

 
 Science robotics , 5(47), 2020.

 

 
 Bottou et al. (2018) 
 
Léon Bottou, Frank E Curtis, and Jorge Nocedal.

 
 Optimization methods for large-scale machine learning.

 
 Siam Review , 60(2):223–311, 2018.

 

 
 Baydin et al. (2017) 
 
Atılım Günes Baydin, Barak A Pearlmutter, Alexey Andreyevich Radul,
and Jeffrey Mark Siskind.

 
 Automatic differentiation in machine learning: a survey.

 
 The Journal of Machine Learning Research , 18(1):5595–5637, 2017.

 

 
 Corliss (1988) 
 
George F Corliss.

 
 Applications of differentiation arithmetic.

 
 In Reliability in Computing , pages 127–148. Elsevier, 1988.

 

 
 Harris et al. (2020) 
 
Charles R. Harris, K. Jarrod Millman, St’efan J. van der Walt, Ralf
Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor,
Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer,
Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fern’andez
del R’ıo, Mark Wiebe, Pearu Peterson, Pierre G’erard-Marchant, Kevin
Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, and
Travis E. Oliphant.

 
 Array programming with NumPy.

 
 Nature , 585(7825):357–362, September
2020.

 
 10.1038/s41586-020-2649-2 .

 

 
 Maclaurin et al. (2015) 
 
Dougal Maclaurin, David Duvenaud, and Ryan P Adams.

 
 Autograd: Effortless gradients in numpy.

 
 In ICML 2015 AutoML Workshop , volume 238, page 5, 2015.

 

 
 Carpentier et al. (2019) 
 
Justin Carpentier, Guilhem Saurel, Gabriele Buondonno, Joseph Mirabel, Florent
Lamiraux, Olivier Stasse, and Nicolas Mansard.

 
 The pinocchio c++ library – a fast and flexible implementation of
rigid body dynamics algorithms and their analytical derivatives.

 
 In IEEE International Symposium on System Integrations (SII) ,
2019.

 

 
 Hendriks et al. (2020) 
 
Johannes Hendriks, Carl Jidling, Adrian Wills, and Thomas Schön.

 
 Linearly constrained neural networks.

 
 arXiv preprint arXiv:2002.01600 , 2020.

 

 
 Jidling et al. (2017) 
 
Carl Jidling, Niklas Wahlström, Adrian Wills, and Thomas B Schön.

 
 Linearly constrained gaussian processes.

 
 In Advances in Neural Information Processing Systems
(NeurIPS) , pages 1215–1224, 2017.

 

 
 Lange-Hegermann (2018) 
 
Markus Lange-Hegermann.

 
 Algorithmic linearly constrained gaussian processes.

 
 In Advances in Neural Information Processing Systems
(NeurIPS) , pages 2137–2148, 2018.

 

 
 Álvarez et al. (2012) 
 
Mauricio A. Álvarez, Lorenzo Rosasco, and Neil D. Lawrence.

 
 Kernels for vector-valued functions: A review.

 
 Foundations and trends in machine learning , 4(3):195–266, 2012.

 
 ISSN 1935-8237.

 

 
 Duvenaud (2014) 
 
David Duvenaud.

 
 Automatic model construction with Gaussian processes .

 
 PhD thesis, University of Cambridge, 2014.

 

 
 Lee et al. (2007) 
 
Wee Lee, Nan Rong, and David Hsu.

 
 What makes some pomdp problems easy to approximate?

 
 Advances in neural information processing systems (NeurIPS) ,
20:689–696, 2007.

 

 
 Curi et al. (2020) 
 
Sebastian Curi, Silvan Melchior, Felix Berkenkamp, and Andreas Krause.

 
 Structured variational inference in partially observable unstable
gaussian process state space models.

 
 In Proceedings of the 2nd Conference on Learning for Dynamics
and Control , volume 120 of Proceedings of Machine Learning Research ,
pages 147–157, The Cloud, 10–11 Jun 2020. PMLR.

 

 
 Parmas et al. (2018) 
 
Paavo Parmas, Carl Edward Rasmussen, Jan Peters, and Kenji Doya.

 
 Pipps: Flexible model-based policy search robust to the curse of
chaos.

 
 In International Conference on Machine Learning , pages
4065–4074. PMLR, 2018.

 

 
 Schön et al. (2011) 
 
Thomas B Schön, Adrian Wills, and Brett Ninness.

 
 System identification of nonlinear state-space models.

 
 Automatica , 47(1):39–49, 2011.

 

 
 Simchowitz et al. (2018) 
 
Max Simchowitz, Horia Mania, Stephen Tu, Michael I. Jordan, and Benjamin Recht.

 
 Learning without mixing: Towards a sharp analysis of linear system
identification.

 
 In Proceedings of the 31st Conference On Learning Theory ,
volume 75 of Proceedings of Machine Learning Research , pages 439–473.
PMLR, 06–09 Jul 2018.

 

 
 Schoukens and Ljung (2019) 
 
Johan Schoukens and Lennart Ljung.

 
 Nonlinear system identification: A user-oriented road map.

 
 IEEE Control Systems Magazine , 39(6):28–99, 2019.

 

 
 Buisson-Fenet et al. (2020) 
 
Mona Buisson-Fenet, Friedrich Solowjow, and Sebastian Trimpe.

 
 Actively learning gaussian process dynamics.

 
 In Learning for Dynamics and Control , pages 5–15. PMLR, 2020.

 

 
 Umlauft et al. (2017) 
 
Jonas Umlauft, Armin Lederer, and Sandra Hirche.

 
 Learning stable Gaussian process state space models.

 
 In American Control Conference (ACC) , pages 1499–1504. IEEE,
2017.

 

 
 Beckers et al. (2017) 
 
Thomas Beckers, Jonas Umlauft, Dana Kulic, and Sandra Hirche.

 
 Stable Gaussian process based tracking control of Lagrangian
systems.

 
 In Conference on Decision and Control (CDC) , pages 5180–5185.
IEEE, 2017.

 

 
 Ahmadi and Khadir (2020) 
 
Amir Ali Ahmadi and Bachir El Khadir.

 
 Learning dynamical systems with side information.

 
 In Proceedings of the 2nd Conference on Learning for Dynamics
and Control , volume 120 of Proceedings of Machine Learning Research ,
pages 718–727, The Cloud, 10–11 Jun 2020. PMLR.

 

 
 Heinonen et al. (2018) 
 
Markus Heinonen, Cagatay Yildiz, Henrik Mannerström, Jukka Intosalmi, and
Harri Lähdesmäki.

 
 Learning unknown ode models with gaussian processes.

 
 In International Conference on Machine Learning , pages
1959–1968, 2018.

 

 
 Yildiz et al. (2018) 
 
Cagatay Yildiz, Markus Heinonen, Jukka Intosalmi, Henrik Mannerstrom, and Harri
Lahdesmaki.

 
 Learning stochastic differential equations with gaussian processes
without gradient matching.

 
 In 2018 IEEE 28th International Workshop on Machine Learning
for Signal Processing (MLSP) , pages 1–6. IEEE, 2018.

 

 
 Wenk et al. (2020) 
 
P. Wenk, G. Abbati, M. A. Osborne, B. Schölkopf, A. Krause, and S. Bauer.

 
 Odin: Ode-informed regression for parameter and state inference in
time-continuous dynamical systems.

 
 In Proceedings of the 34th Conference on Artificial
Intelligence (AAAI) , volume 34, pages 6364–6371. AAAI Press, February 2020.

 
 AAAI Technical Track: Machine Learning.

 

 
 Li et al. (2020) 
 
Xuechen Li, Ting-Kam Leonard Wong, Ricky T. Q. Chen, and David Duvenaud.

 
 Scalable gradients for stochastic differential equations.

 
 In Proceedings of the Twenty Third International Conference on
Artificial Intelligence and Statistics , volume 108 of Proceedings of
Machine Learning Research , pages 3870–3882, Online, 26–28 Aug 2020. PMLR.

 

 
 Schober et al. (2014) 
 
M. Schober, D. Duvenaud, and P. Hennig.

 
 Probabilistic ODE solvers with runge-kutta means.

 
 In Advances in Neural Information Processing Systems 27 , pages
739–747. Curran Associates, Inc., 2014.

 

 
 Schober et al. (2019) 
 
Michael Schober, Simo Särkkä, and Philipp Hennig.

 
 A probabilistic model for the numerical solution of initial value
problems.

 
 Statistics and Computing , 29(1):99–122,
2019.