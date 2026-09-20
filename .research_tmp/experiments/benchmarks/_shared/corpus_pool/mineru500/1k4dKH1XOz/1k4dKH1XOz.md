# End-to-End Learning Framework for Solving Non-Markovian Optimal Control

Xiaole Zhang $^{1}$ Peiyu Zhang $^{1}$ Xiongye Xiao $^{1}$ Shixuan Li $^{1}$ Vasileios Tzoumas $^{2}$ Vijay Gupta $^{3}$ Paul Bogdan $^{1}$

# Abstract

Integer-order calculus often falls short in capturing the long-range dependencies and memory effects found in many real-world processes. Fractional calculus addresses these gaps via fractional-order integrals and derivatives, but fractional-order dynamical systems pose substantial challenges in system identification and optimal control due to the lack of standard control methodologies. In this paper, we theoretically derive the optimal control via linear quadratic regulator (LQR) for fractional-order linear time-invariant (FOLTI) systems and develop an end-to-end deep learning framework based on this theoretical foundation. Our approach establishes a rigorous mathematical model, derives analytical solutions, and incorporates deep learning to achieve data-driven optimal control of FOLTI systems. Our key contributions include: (i) proposing an innovative system identification method control strategy for FOLTI systems, (ii) developing the first end-to-end data-driven learning framework, Fractional-Order Learning for Optimal Control (FOLOC), that learns control policies from observed trajectories, and (iii) deriving a theoretical analysis of sample complexity to quantify the number of samples required for accurate optimal control in complex real-world problems. Experimental results indicate that our method accurately approximates fractional-order system behaviors without relying on Gaussian noise assumptions, pointing to promising avenues for advanced optimal control.

$^{1}$ Ming Hsieh Department of Electrical and Computer Engineering, University of Southern California, Los Angeles, CA 90089, USA $^{2}$ Department of Aerospace Engineering, University of Michigan, Ann Arbor, MI 48109, USA $^{3}$ Elmore Family School of Electrical and Computer Engineering, Purdue University, West Lafayette, IN 47907, USA. Correspondence to: Xiaole Zhang <xiaolezh@usc.edu>, Paul Bogdan <pbogdan@usc.edu>.

Proceedings of the $42^{nd}$ International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

# 1. Introduction

Many real-world systems, including biological networks, financial markets, and control systems, exhibit long-range dependencies and memory effects that traditional integer-order models struggle to capture (Lundstrom et al., 2008; Ivanov et al., 1999; Ghorbani et al., 2018). Fractional-order linear time-invariant (FOLTI) systems extend classical integer-order models through fractional calculus, providing a more flexible framework for modeling such complex behaviors. Due to their ability to represent non-local dependencies and history-dependent dynamics, FOLTI systems have found broad applications in control systems, bioengineering, neuroscience, physical, and financial modeling. However, the practical deployment of FOLTI systems faces significant challenges in system identification and optimal control. Unlike Markovian systems, where the current state depends solely on the immediate past state, FOLTI systems exhibit non-Markovian properties, meaning their evolution is influenced by a broad range of past states. This memory effect introduces additional complexities, making it difficult for traditional methods to generalize effectively to FOLTI systems.

Despite their advantages, FOLTI systems present significant challenges in both modeling and control. Identifying system parameters for fractional-order dynamical systems remains an open research problem. One major difficulty in system identification arises from the non-Markovian nature of fractional-order systems, where the state evolution is not solely determined by recent states but depends on an entire history of past observations. Additionally, the fractional derivative operators, such as the Grünwald–Letnikov fractional derivative (Hilfer, 2000; Monje et al., 2010; Oldham & Spanier, 1974), introduce combinatorial complexity and nonlinearities, making standard parameter estimation techniques ineffective. Existing methods often rely on strong assumptions, such as noiseless data or constrained data generation processes, limiting their applicability in practical scenarios.

Beyond system identification, optimal control of FOLTI systems remains a major challenge. Unlike integer-order systems, where well-established control frameworks exist,

fractional-order systems exhibit long-range dependencies and memory effects, making classical control strategies difficult to apply directly. In particular, the lack of analytical solutions for optimal control in FOLTI systems necessitates new approaches that account for their unique dynamics. Addressing these challenges is crucial for enabling real-world applications of FOLTI systems in control systems.

A well-established framework for optimal control in integer-order systems is the Linear Quadratic Regulator (LQR), introduced by Rudolf Kalman in 1960 (Kalman et al., 1960). LQR aims to determine an optimal control policy that minimizes a quadratic cost function defined over the system state and control input, subject to weighing matrices, for systems governed by linear dynamics. The traditional integer-order LQR problem has been extensively studied over the decades (Dean et al., 2018; 2020; Tu & Recht, 2019). For linear time-invariant (LTI) systems with known parameters, the LQR problem has a closed-form solution on the infinite time horizon and can be solved efficiently using dynamic programming on finite horizons (Anderson & Moore, 2007). Furthermore, numerous researchers have developed end-to-end guarantees for LQR in the context of LTI systems. These guarantees often involve a two-step process: (I) estimating the unknown system parameters, and (ii) designing robust controllers to account for model uncertainties.

While the LQR problem for integer-order LTI systems is well understood (see related work in Appendix B), far less attention has been devoted to the case of fractional-order dynamical systems (Monje et al., 2010). These systems, which originate from fractional calculus (Hilfer, 2000; Ionescu et al., 2017), differ significantly from their integer-order counterparts. Unlike Markovian systems, where the current state depends solely on the immediate past state, fractional-order systems exhibit a non-Markovian behavior, with the current state influenced by a memory effect spanning a broad range of past states. This memory effect makes the analysis and control of fractional-order dynamical systems particularly challenging. Estimation of unknown parameters in fractional-order dynamical systems remains an open problem (Yaghooti & Sinopoli, 2023; Chatterjee & Pequito, 2022; Zhang et al., 2025), even under the assumption of linear fractional operators. For finite horizons, optimal control problems have been addressed under specific assumptions (Li & Chen, 2008), but no comprehensive results exist for infinite horizons. Moreover, achieving end-to-end guarantees for fractional-order systems is particularly challenging, as both system identification (Yaghooti & Sinopoli, 2023; Zhang et al., 2025) and robust control present significant technical difficulties.

To address these challenges, this work establishes a theoretical framework for fractional-order LQR, deriving a principled end-to-end learning approach for FOLTI optimal control. Specifically, we formulate the mathematical structure of FOLTI systems, enabling a rigorous extension of LQR to the fractional-order setting. Given the diagonal elements of the system matrix, we propose a method to estimate the unknown system parameters from multiple observed trajectories. Using this estimated model, we derive the optimal control policy using LQR, leveraging least squares optimization and Lagrange multipliers to obtain analytically tractable solutions despite the inherent complexities introduced by fractional-order dynamics.

Beyond theoretical derivations, we develop an end-to-end data-driven learning framework, fractional-order learning for optimal control (FOLOC), that jointly performs system identification and optimal control. Unlike classical control methods that require pre-specified system dynamics, our approach learns both the system parameters and control policies directly from observed trajectories. We further analyze the sample complexity of this learning process, quantifying the number of samples required for the estimated LQR loss to converge to the true LQR loss under unknown input conditions. Unlike classical approaches that assume structured noise distributions, our framework is designed to operate in realistic, noisy environments. Specifically, we incorporate deep learning-based modeling techniques to ensure that the control policy remains robust under non-Gaussian noise, distributional shifts, and real-world uncertainties.

In summary, this work makes the following key contributions: (i) We provide theoretical foundations for optimal control of fractional-order linear time-invariant (FOLTI) systems. (ii) We develop a principled end-to-end data-driven learning framework, FOLOC, that jointly optimizes system identification loss and control policies, ensuring robustness to non-Gaussian noise and real-world uncertainties. (iii) We analyze the theoretical sample complexity, providing convergence guarantees for learned control policies under unknown system dynamics. (iv) We conduct extensive experiments, demonstrating the FOLOC's robustness across noise distributions, scalability to different system complexities, and computational efficiency in real-time control tasks.

# 2. Preliminaries and Problem Formulation

In this section, we introduce key theoretical foundations from fractional calculus, focusing on the Grünwald–Letnikov fractional-order derivative and its role in modeling FOLTI systems. By extending LQR to FOLTI systems, we then formulate the end-to-end learning problem, where the objective is to identify system parameters and optimize control policies directly from data. This formulation integrates system identification and control optimization, addressing challenges arising from memory effects and non-Markovian dynamics in FOLTI systems.

# 2.1. Fractional-Order Derivative

Definition 2.1 (Grünwald–Letnikov fractional-order derivative). The left-side and right-side Grünwald–Letnikov fractional-order derivatives of order $\alpha$ on the finite interval $I = [a, b]$ are defined as follows:

$$
\left(^ {G L} D _ {a +} ^ {\alpha} f\right) (x) = \lim _ {h \rightarrow 0 ^ {+}} \frac {1}{h ^ {\alpha}} \sum_ {k = 0} ^ {\left\lfloor \frac {x - a}{h} \right\rfloor} (- 1) ^ {k} \binom {\alpha} {k} f (x - k h), \tag {1}
$$

$$
\left(^ {G L} D _ {b -} ^ {\alpha} f\right) (x) = \lim _ {h \rightarrow 0 ^ {-}} \frac {1}{h ^ {\alpha}} \sum_ {k = 0} ^ {\lfloor \frac {b - x}{h} \rfloor} (- 1) ^ {k} \binom {\alpha} {k} f (x + k h), \tag {2}
$$

where $x \in R$ , 0 < h < b - a, and $\alpha > 0$ . Here, $\binom{\alpha}{k}$ denotes the generalized binomial coefficient, defined as $\binom{\alpha}{k} = \frac{\Gamma(\alpha+1)}{\Gamma(k+1)\Gamma(\alpha-k+1)}$ , where $\Gamma(\cdot)$ denotes the gamma function, which is defined as $\Gamma(z) = \int_{0}^{\infty} t^{z-1} e^{-t} dt$ for z > 0.

# 2.2. Grünwald–Letnikov Difference Operator

The fractional-order Grünwald–Letnikov difference operator allows to discretize the fractional-order derivative and represent it as a finite difference as follows:

$$
\Delta^ {\alpha} x _ {k} := \sum_ {j = 0} ^ {k} D (\alpha , j) x _ {k - j}, \tag {3}
$$

where $x_{k} \in R^{n}, \alpha = [\alpha_{1}, \alpha_{2}, \ldots, \alpha_{n}]^{\top} \in R^{n}$ represents the order of the difference operator, and $D(\alpha, j) \in \mathbb{R}^{n \times n}$ is an $n \times n$ diagonal matrix defined as

$$
D (\alpha , j) := \mathrm{diag} \left(\psi (\alpha_ {1}, j), \psi (\alpha_ {2}, j), \dots , \psi (\alpha_ {n}, j)\right), \tag {4}
$$

with

$$
\psi (\alpha_ {i}, j) := \frac {\Gamma (j - \alpha_ {i})}{\Gamma (- \alpha_ {i}) \Gamma (j + 1)}, \quad i = 1, 2, \dots , n. \tag {5}
$$

# 2.3. Fractional-Order Linear Time Invariant System

The state-space representation of the discrete-time fractional-order linear time invariant system reads:

$$
\Delta^ {\alpha} x _ {k + 1} = A x _ {k} + B u _ {k}, \tag {6}
$$

where $x_{k} \in R^{n}$ is the state vector, $u_{k} \in R^{m}$ is the system input, matrices A and B are constant matrices with size $n \times n$ and $n \times m$ , respectively.

Using the Grünwald–Letnikov difference operator, we can write the system as follows:

$$
x _ {k + 1} = A x _ {k} + B u _ {k} - \sum_ {j = 1} ^ {k + 1} D (\alpha , j) x _ {k + 1 - j}. \tag {7}
$$

The last term in Eq. 7 represents the memory-dependent nature of fractional-order dynamical systems, making them suitable for modeling non-Markovian and long-range dependent processes.

# 2.4. Linear Quadratic Regulator for FOLTI systems

The linear quadratic regulator (LQR) problem is that of optimal control of a dynamical system given known and fixed quadratic costs. Formally, the goal of LQR is to find the optimal control $u_{t}$ that minimizes the quadratic loss fitting in the FOLTI systems, which in turn solves the following optimization problem.

Definition 2.2 (LQR for FOLTI systems). The LQR problem for FOLTI systems is defined as follows:

$$
\min _ {\left\{u _ {i} \right\} _ {i = 0} ^ {T - 1}} J _ {T} (u) := \sum_ {k = 0} ^ {T - 1} \left(x _ {k} ^ {\top} Q x _ {k} + u _ {k} ^ {\top} R u _ {k}\right) + x _ {T} ^ {\top} Q _ {f} x _ {T} \tag {8}
$$

$$
\text { s.   t. } \Delta^ {\alpha} x _ {k + 1} = A x _ {k} + B u _ {k}, \tag {9}
$$

where $Q \in \mathbb{R}^{n \times n}$ and $Q_{f} \in \mathbb{R}^{n \times n}$ are positive semi-definite matrices representing the state cost and terminal state cost, respectively, where $n$ is the dimension of the state vector $x_{k}$ . Similarly, $R \in \mathbb{R}^{m \times m}$ is a positive semi-definite matrix representing the control cost, where $m$ is the dimension of the control input vector $u_{k}$ .

# 2.5. Problem Formulation

Consider a fractional-order linear time-invariant system defined using the Grünwald–Letnikov difference operator. The goal is to find an optimal control sequence via solving LQR problem based on system identification from observed data.

Specifically, we are given N trajectories, each trajectory consisting of T time steps, sampled from a joint distribution $\Pi$ over initial states, process noise, system parameters and cost matrices, i.e., $\{x_{0},\{w_{k}\}_{k=0}^{T-1},\{u_{k}\}_{k=0}^{T-1},\{A,B,\alpha\},\{Q,R\}\} \sim \Pi$ . Thus, we can represent the dataset as $D = \{\{x_{k}^{i}\}_{k=0}^{T},\{u_{k}^{i}\}_{k=0}^{T-1},Q^{i},R^{i},\{u_{k}^{i}{}^{o}(x_{0}^{i},A,B,\alpha)\}_{k=0}^{T-1}\}_{i=1}^{N}$ , where $u_{k}^{i}{}^{o}$ denotes the optimal control input. We aim to solve the following two-step problem:

(i) System Identification. The fractional-order system dynamics are parameterized by $\Theta := \{A, B, \alpha\}$ , where $A \in R^{n \times n}$ and $B \in R^{n \times m}$ are system matrices, and $\alpha \in (0,1)^{n}$ is the fractional order. The system identification problem is posed as:

$$
\hat {\Theta} = \underset {\Theta} {\arg \min} \sum_ {i = 1} ^ {N} \sum_ {k = 1} ^ {T} \left\| \hat {x} _ {k} ^ {i} (\Theta) - x _ {k} ^ {i} \right\| _ {2} ^ {2}, \tag {10}
$$

where $\hat{x}_{k}^{i}(\Theta)$ represents the predicted state at time k for the i-th trajectory, obtained using the identified parameters $\Theta$ .

(ii) Optimal Control. Once $\hat{\Theta} = \{\hat{A}, \hat{B}, \hat{\alpha}\}$ is estimated, the optimal control problem is formulated as:

$$
\min _ {\left\{u _ {k} \right\} _ {k = 0} ^ {T - 1}} \hat {J} _ {T} (u) := \sum_ {k = 0} ^ {T - 1} \left(x _ {k} ^ {\top} Q x _ {k} + u _ {k} ^ {\top} R u _ {k}\right) + x _ {T} ^ {\top} Q _ {f} x _ {T}, \tag {11}
$$

subject to the identified system dynamics:

$$
\Delta^ {\hat {\alpha}} x _ {k + 1} = \hat {A} x _ {k} + \hat {B} u _ {k}. \tag {12}
$$

Deep Learning-Based Reformulation. The optimal control problem can be framed in a deep learning context as learning an operator $L: D \to U$ , where U represents the set of optimal control inputs. Formally, the operator $L: D \to U$ computes the optimal control policy $\{u_{t}^{o}\}_{t=0}^{T-1}$ directly from the observed trajectory, while simultaneously uncovering the system parameters $\Theta = \{A, B, \alpha\}$ that encapsulate the dynamics of the fractional-order system.

# 3. Methodology

# 3.1. Mathematical Foundations for Fractional-Order Learning and Control

Lemma 3.1 (Discrete-time FOLTI system solution). The solution to the discrete-time FOLTI system is given by (Guer-mah et al., 2012):

$$
x _ {k} = G _ {k} x _ {0} + \sum_ {j = 0} ^ {k - 1} G _ {k - 1 - j} B u _ {j}, \tag {13}
$$

where the matrices $G_{k}$ are defined recursively as:

$$
G _ {k} = \left\{ \begin{array}{l l} I & \text { for   } k = 0, \\ \sum_ {j = 0} ^ {k - 1} A _ {j} G _ {k - 1 - j} & \text { for   } k \geq 1, \end{array} \right. \tag {14}
$$

and the matrices $A_{j}$ are given by:

$$
A _ {j} = \left\{ \begin{array}{l l} A - \text { diag } (\alpha_ {1}, \dots , \alpha_ {n}) & \text { if   } j = 0, \\ - D (\alpha , j + 1) & \text { if   } j \geq 1. \end{array} \right. \tag {15}
$$

The proof is provided in Appendix D.2. The first component of the FOLTI system solution represents the unforced response of the system. The term $G_{k}$ exhibits the particularity of being time-varying, attributed to the fractional-order $\alpha$ which inherently accounts for all the past states. The second component takes the role of the convolution sum corresponding to the forced response.

Theorem 3.2 (LQR solution for FOLTI systems). The least-squares optimal control solution for the LQR problem of a FOLTI system is given by:

$$
U = - \left(G ^ {\top} \bar {Q} G + \bar {R}\right) ^ {- 1} G ^ {\top} \bar {Q} ^ {\top} H x _ {0}, \tag {16}
$$

whereas the Lagrange multiplier optimal control solution is:

$$
U = - R ^ {- 1} B ^ {\top} \otimes (I - G _ {\lambda}) ^ {- 1} H _ {\lambda} x _ {0}. \tag {17}
$$

Here, U, G, $G_{\lambda}$ , H, $H_{\lambda}$ , $\bar{R}$ , and $\bar{Q}$ are defined in Appendix D.3, and $\otimes$ denotes the Kronecker product. Notably, $U = \left[u_{0}^{\top} \quad u_{1}^{\top} \quad \cdots \quad u_{T-1}^{\top}\right]^{\top}$ .

Using the optimal control solution via LQR, a natural question arises: given a system identification algorithm and a threshold error $\epsilon$ , how many samples are required for the system identification process to ensure that the error between the estimated LQR loss $\hat{J}$ and the true LQR loss J (computed with the true system parameters) remains within the threshold $d(\hat{J}-J)\leq\epsilon$ , where $d(\cdot)$ denotes the metric. We derive the following sample complexity results.

Theorem 3.3 (Sample Complexity for FOLTI systems). Consider a FOLTI system with a known matrix A, fractional order $\alpha$ , and an unknown matrix B. The following sample complexity bounds hold using the system identification in Appendix D.1: Let

$$
K _ {B} = \frac {1}{N} (\phi^ {\top} \phi) ^ {- 1} \phi^ {\top} K _ {w} \phi (\phi^ {\top} \phi) ^ {- 1},
$$

(a) Least-squares solution.

$$
\mathbb {E} \left[ | \hat {J} - J | \right] \leq \| z \| _ {2} ^ {2} \| R ^ {- 1} \| _ {2} \left(1 + \| B \| _ {2} ^ {2} \| R ^ {- 1} \| _ {2} \| S \| _ {2}\right)
$$

$$
\cdot \big (\mathrm{Tr} (K _ {B}) + 2 \| B \| _ {2} \sqrt {\mathrm{Tr} (K _ {B})} \big), (1 8)
$$

(b) Lagrange multiplier solution. Assume $L_{QG}^{-1}(I - L_u) \succeq 0$ ,

$$
\mathbb {E} \left[ | \hat {J} - J | \right] \leq \| z \| _ {2} \| H _ {\lambda} x _ {0} \| _ {2} \| R ^ {- 1} \| _ {2} \| \mathbb {L} \| _ {2} (1 + \| B \| _ {2} ^ {2}
$$

$$
\cdot \left. \| R ^ {- 1} \| _ {2} \| \mathbb {L} \| _ {2} \| L _ {Q G} \| _ {2}\right) \big (\operatorname{Tr} (K _ {B}) +
$$

$$
2 \| B \| _ {2} \sqrt {\operatorname{Tr} (K _ {B})}), \tag {19}
$$

where $z = G_d^\top \bar{Q}^\top Hx_0$ , $S = G_d^\top \bar{Q}G_d$ , $\| \mathbb{L}\| _2 = \| (L_{QG})^{-1}\| _2\| L_{QG}\| _2$ , and $G_d, \bar{Q}$ , and $H$ are system matrices as defined in Appendix D.4.

Corollary 3.4 (Simplified Sample Complexity for FOLTI systems). Let $K_{w} = \sigma_{w}^{2}I$ and $u_{0}^{i} \sim \mathcal{N}(0, I)$ . For p > m + 1, then:

$$
\operatorname{Tr} \left(K _ {B}\right) = \frac {n m \sigma_ {w} ^ {2}}{N (p - m - 1)} \tag {20}
$$

where p and N denote the number of samples, and m is the dimension of control inputs (see Appendix D.5).

The sample complexity results demonstrate that the overall convergence rates of the least-squares solution and the Lagrange multiplier solution are both $\mathcal{O}\left(\frac{1}{\sqrt{N}}\right)$ , differing only in their constant factors.

# 3.2. Theoretical Approach for Fractional-Order System Identification and Optimal Control

Our end-to-end theoretical learning method follows a similar strategy to those used in Markovian processes, specifically for LTI systems. The learning procedure can be summarized in two main steps. Given N noisy trajectories, each of T time steps, and assuming the diagonal elements of the system matrix A are known, we first formulate a least-squares problem to identify the system parameters by using the linearity of the Grünwald–Letnikov difference operator. The state evolution with noise can be represented as

$$
\Delta^ {\alpha} x _ {k + 1} = A x _ {k} + B u _ {k} + w _ {k}, \tag {21}
$$

where $\{w_{k}\}_{k=0}^{T-1}$ is an independent and identically distributed (i.i.d) Gaussian noise process. Using the linearity of the fractional difference operator, the state evolution for each trajectory can be expressed as:

$$
x _ {1} = A x _ {0} + B u _ {0} + C _ {\alpha} x _ {0} + w _ {0} \tag {22}
$$

Using this formulation, we define the following optimization problem across $N$ trajectories

$$
\hat {\theta} = \arg \min _ {\theta} \| X - \xi \theta \| _ {2} ^ {2}, \tag {23}
$$

with the closed-form solution

$$
\hat {\theta} = \left(\xi^ {\top} \xi\right) ^ {- 1} \xi^ {\top} X, \tag {24}
$$

where system parameters $C_{\alpha}, \theta$ and feature matrices $X, \xi$ are detailed in Appendix D.1.

In the second step, the derived system parameters are used to compute the optimal control solution as described in Theorem 3.2. For a given identified FOLTI system and a time horizon T, the optimal control sequence can be derived for any cost matrices Q and R, along with the initial condition $x_{0}$ .

# 3.3. Fractional-Order Learning for Optimal Control Framework

Our end-to-end deep learning framework FOLOC is guided by the mathematical derivation of the optimal control policy for FOLTI systems (see Methodology 3.1). Generally, FOLOC framework can be structured into two main components.

System Identification Module: Inspired by traditional LTI and our proposed FOLTI system identification, we use deep learning based time-series models to estimate system parameters and approximate intermediate variables. The objective is to learn the underlying function that maps trajectories to the variables required for solving the LQR problem.

Theoretical system identification for both LTI and FOLTI systems predominantly relies on least-squares methods, which can be viewed as operators learning a function that maps from the trajectory and control input space X to the system parameter space S. Consider the temporal nature of the trajectory, we choose a recurrent neural network to estimate the system parameters. The mapping is defined as $\mathcal{R}_{\theta}: \mathcal{X} \in \mathbb{R}^{T \times (n+m)} \to \mathcal{S} \in \mathbb{R}^{n^{2}+nm+n}$ such that $\text{vectorize}(\hat{A}_{\theta}, \hat{B}_{\theta}, \hat{\alpha}_{\theta}) = \mathcal{R}_{\theta}(\{x_{i}, u_{i}\}_{i=0}^{T-1})$ . From Lemma 3.1, the function mapping between the system parameters to the time-varying matrices $A_{k}$ and then to the matrices $G_{k}$ can be represented as follows respectively

$$
A _ {k} = g _ {k} (A, \alpha ; \Gamma),
$$

$$
G _ {k} = f _ {k} (A _ {0}, \dots , A _ {k - 1}), \tag {25}
$$

where $g_{k}(\cdot;\Gamma)$ is obtained as a function of the Gamma function, which governs the fractional-order dynamics. To model the first mapping, we use MLPs with injecting time embeddings to approximate the function $R_{g}:S\to H\in R^{T\times h}$ such that $\{\hat{A}_{i_{g}}\}_{i=0}^{T-1}=R_{g}(\hat{A}_{\theta},\hat{\alpha}_{\theta},\{e_{i}\}_{i=0}^{T-1})$ , where h can be chosen based on the dynamics complexity and $\{e_{i}\}_{i=0}^{T-1}$ represents an embedding of a specific timestamp i. The second mapping is approximated by an Encoder-only Transformer $R_{f}:H\to G\in R^{T\times n}$ such that $\{\hat{G}_{i_{f}}\}_{i=0}^{T-1}=R_{f}(\{\hat{A}_{i_{g}}\}_{i=0}^{T-1})$ , where $\hat{G}_{i_{f}}$ represents the learned approximation of the intermediate parameters $G_{k}$ .

Optimal Control Module: Building upon the mathematical derivation of the LQR problem using Lagrange multipliers, we use a neural operator to learn the mapping between the input space (variable required for solving the LQR problem) and the output space $\{u_{i}^{o}\}_{i=0}^{T-1}$ (optimal control sequence).

The core of optimal control module mirrors the process of solving the linear system derived from the LQR problem through Lagrange multipliers. The mathematical formulation of the linear system is expressed as:

$$
(I - G _ {\lambda}) \lambda = 2 H _ {\lambda} x _ {0}, \tag {26}
$$

where $\lambda$ is defined in Appendix D.3 and $I - G_{\lambda} \in \mathbb{R}^{Tn \times Tn}$ is a block Toeplitz matrix (see Definition D.2). Using the solution of the Lagrange multipliers, the optimal control sequence $\{u_i^o\}_{i=0}^{T-1}$ can be given by:

$$
U = - R ^ {- 1} B ^ {\top} \otimes (I - G _ {\lambda}) ^ {- 1} H _ {\lambda} x _ {0}. \tag {27}
$$

Luo et al. (2024) has demonstrated that solving linear systems via Krylov iterations can be significantly accelerated by using the Fourier Neural Operator (FNO). Specifically, FNO learns the mapping from the linear system to its corresponding invariant subspace, using the predicted subspace to guide iterative solvers. Inspired by this approach, we use a neural operator to directly learn the mapping between two Hilbert spaces, denoted as $T: A \to U \in R^{T \times n}$ . Due to the block Toeplitz structure of the linear system, the input to FNO is the reduced relevant function variables $a \in R^{T \times d_{a}}$ .

![](images/b17d56550c8c6efe181349ef83f8355e532c79145fa614e81572c214b6533d8d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input X"] --> B["RNN"]
    B --> C["Identification Loss"]
    C --> D["Stack MLP"]
    D --> E["Optimal Control Loss"]
    E --> F["Total Loss"]
    G["SEQ Model"] --> H["Tokens A_i"]
    H --> I["SEQ Encoder"]
    I --> J["Tokens G_i"]
    J --> K["Stack MLP"]
    K --> L["Optimal Control Loss"]
    M["Time Encoding"] --> N["Identification Loss"]
    N --> O["Stack MLP"]
    O --> P["Optimal Control Loss"]
    Q["Q"] --> O
    R["R"] --> O
    S["A"] --> B
    T["Â"] --> B
    U["α"] --> B
    V["â"] --> B
    W["B"] --> B
    X["Â̂"] --> B
    Y["Â"] --> B
    Z["α̂"] --> B
    AA["B̂"] --> B
    AB["β̂"] --> B
    AC["Input X"] --> B
    AD["Input X"] --> B
    AE["Input X"] --> B
    AF["Input X"] --> B
    AG["Input X"] --> B
    AH["Input X"] --> B
    AI["Input X"] --> B
    AJ["Input X"] --> B
    AK["Input X"] --> B
    AL["Input X"] --> B
    AM["Input X"] --> B
    AN["Input X"] --> B
    AO["Input X"] --> B
    AP["Input X"] --> B
    AQ["Input X"] --> B
    AR["Input X"] --> B
    AS["Input X"] --> B
    AT["Input X"] --> B
    AU["Input X"] --> B
    AV["Input X"] --> B
    AW["Input X"] --> B
    AX["Input X"] --> B
    AY["Time Encoding"] --> AZ["Identification Loss"]
    BA["Time Encoding"] --> BB["Identification Loss"]
    BC["Time Encoding"] --> BD["Identification Loss"]
    BE["Time Encoding"] --> BF["Identification Loss"]
    BG["Time Encoding"] --> BH["Identification Loss"]
    BI["Time Encoding"] --> BJ["Identification Loss"]
    BK["Time Encoding"] --> BL["Identification Loss"]
    BM["Time Encoding"] --> BN["Identification Loss"]
    BO["Time Encoding"] --> BP["Identification Loss"]
    BP --> BR["Identification Loss"]
    BS["Input X"] --> BT["Input X"]
    BU["X"] --> BV["X"]
    BW["X"] --> BX["X"]
    BY["X"] --> BZ["X"]
    CA["X"] --> BYX["X"]
    CB["X"] --> BYXX["X"]
    CC["X"] --> BYXXX["X"]
    DD["X"] --> DB["X"]
    DB["X"] --> DC["X"]
    DV["X"] --> DCX["X"]
    DW["X"] --> DCXX["X"]
    DX["X"] --> DBX["X"]
    DBX["X"] --> DCXX["X"]
    DG["X"] --> DVX["X"]
    DVX["X"] --> DCXX["X"]
```
</details>

Figure 1. Overview of the proposed model architecture. The pipeline first infers fractional-order system parameters $(A, \alpha, B)$ from input X using an RNN+MLP based system identification module, These parameters are then encoded as embeddings of sequential tokens $A_{i}$ for time-dependent modeling. An attention-based Sequence Encoder processes these embeddings to obtain latent representations, which along with cost matrices Q, R, estimated system matrix B are fed to the Stack MLPs with residual connection input X for Fourier Neural Operator to predict optimal control signals. Finally, a Composite Loss function unifies system identification and control prediction, enabling end-to-end training of both system parameter estimation and control law synthesis.

The transformation operator L first lifts the input a to a higher dimensional channel space $v_{0} \in R^{d_{v} \times c}$ , where c is the number of channels. Then iteratively applying several Fourier layer to update the representation $v_{i} \to v_{i+1}$ , we have the final $v_{T} \in R^{d_{v} \times c}$ , which has the same dimensionality as $v_{0}$ . The FNO outputs optimal control sequence $U = \mathcal{Q}(v_{T})$ through the projection of $v_{T}$ by the transformation operator $Q : R^{d_{v} \times c} \to R^{T \times n}$ . This design enables efficient computation of the optimal control sequence by leveraging the special properties of the block Toeplitz linear system and the expressive capabilities of the FNO.

The loss function used in the training process is defined as follows:

$$
l (\theta) = \lambda_ {w} l _ {s} (\theta) + (1 - \lambda_ {w}) l _ {o} (\theta), \tag {28}
$$

where $l_{s}$ quantifies the loss due to system identification errors, and $l_{o}$ quantifies the loss due to optimal control errors. Specifically,

$$
l _ {s} (\theta) = \| A - \hat {A} _ {\theta} \| _ {2} ^ {2} + \| B - \hat {B} _ {\theta} \| _ {2} ^ {2} + \| \alpha - \hat {\alpha} _ {\theta} \| _ {2} ^ {2},
$$

$$
l _ {o} (\theta) = \left\| U - \hat {U} _ {\theta} \right\| _ {2} ^ {2}. \tag {29}
$$

Here, $\lambda$ is a user-defined weighting parameter that determines the balance between system identification and optimal control errors during the learning process.

Remark. The fractional-order dynamical system parameters A, B and $\alpha$ can be extracted from an intermediate layer in the network because of the underlying loss function. Though our end-to-end deep learning framework is guided by the theoretical optimal control solution derived by using Lagrange multipliers, certain learned parameters do not directly correspond one-to-one with their theoretical counterparts (see Appendix E.1 for details).

# 4. Experiments

In this section, we empirically evaluate the proposed system identification algorithm and FOLOC framework using both synthetic and real-world system dynamics. The synthetic data is categorized into two primary types: (i) data generated from a single FOLTI system with fixed system parameters A, B, and $\alpha$ ; and (ii) data generated from multiple FOLTI systems, where A, B are fixed, and $\alpha$ vary across systems. The real-world data is simulated from a cart pole and a quadrotor dynamical systems.

# 4.1. Performance Evaluation on Synthetic Data

We evaluate our theoretical end-to-end learning approach and FOLOC framework using synthetic data. Due to the strict assumptions required for the theoretical learning process, we test it using synthetic data generated from a single underlying FOLTI system, with data corrupted by Gaussian noise. Our empirical analysis examines the performance of FOLOC framework from different perspectives.

The underlying fractional-order dynamics used to generate

Table 1. Experimental results on synthetic FOLTI system under varying input conditions. All reported values are in units of $10^{-3}$ . Bolded values represent the minimum in each row. 

<table><tr><td>Metrics</td><td>MSE/MAE</td><td>MSE/MAE</td><td>MSE/MAE</td><td>MSE/MAE</td><td>MSE/MAE</td></tr><tr><td rowspan="2">Fractional Order</td><td>alpha = 0.1</td><td>alpha = 0.3</td><td>alpha = 0.5</td><td>alpha = 0.7</td><td>alpha = 0.9</td></tr><tr><td> $3.39 \pm 0.52 / 8.04 \pm 0.42$ </td><td> $5.04 \pm 1.82 / 12.5 \pm 1.11$ </td><td> $5.61 \pm 0.84 / 16.5 \pm 0.61$ </td><td> $5.18 \pm 0.06 / 9.22 \pm 0.08$ </td><td> $4.94 \pm 0.09 / 10.5 \pm 0.09$ </td></tr><tr><td rowspan="2">State Dimension</td><td>n = 1</td><td>n = 2</td><td>n = 4</td><td>n = 6</td><td>n = 8</td></tr><tr><td> $1.98 \pm 0.08 / 6.60 \pm 0.13$ </td><td> $5.61 \pm 0.84 / 16.5 \pm 0.61$ </td><td> $5.84 \pm 0.71 / 13.4 \pm 0.55$ </td><td> $4.88 \pm 0.34 / 13.4 \pm 0.26$ </td><td> $4.26 \pm 0.22 / 12.3 \pm 0.14$ </td></tr><tr><td rowspan="2">Time Horizon</td><td>T = 8</td><td>T = 16</td><td>T = 32</td><td>T = 64</td><td>T = 128</td></tr><tr><td> $101 \pm 18.4 / 111 \pm 4.40$ </td><td> $43.2 \pm 8.20 / 59.1 \pm 1.12$ </td><td> $14.2 \pm 2.96 / 31.5 \pm 1.70$ </td><td> $5.61 \pm 0.84 / 16.5 \pm 0.61$ </td><td> $2.66 \pm 0.64 / 8.62 \pm 0.70$ </td></tr><tr><td rowspan="2">Noise Type</td><td>Cauchy</td><td>Gamma</td><td>Sinc-squared</td><td>Uniform</td><td>Poisson</td></tr><tr><td> $6.98 \pm 1.89 / 17.0 \pm 0.71$ </td><td> $5.84 \pm 1.15 / 16.7 \pm 1.16$ </td><td> $6.87 \pm 1.46 / 17.6 \pm 0.75$ </td><td> $6.46 \pm 1.29 / 17.0 \pm 1.30$ </td><td> $6.59 \pm 1.59 / 16.5 \pm 0.90$ </td></tr><tr><td rowspan="2">Noise Scale (Gaussian)</td><td> $\sigma = 0.0001$ </td><td> $\sigma = 0.001$ </td><td> $\sigma = 0.01$ </td><td> $\sigma = 0.1$ </td><td> $\sigma = 1$ </td></tr><tr><td> $5.85 \pm 0.65 / 16.9 \pm 0.65$ </td><td> $6.09 \pm 0.74 / 17.4 \pm 0.87$ </td><td> $5.74 \pm 0.83 / 16.7 \pm 0.86$ </td><td> $5.65 \pm 0.70 / 16.0 \pm 0.39$ </td><td> $10.2 \pm 8.11 / 20.5 \pm 7.37$ </td></tr><tr><td rowspan="2">Training Size</td><td> $N_{T} = 1000$ </td><td> $N_{T} = 2000$ </td><td> $N_{T} = 4000$ </td><td> $N_{T} = 6000$ </td><td> $N_{T} = 8000$ </td></tr><tr><td> $5.18 \pm 1.13 / 12.9 \pm 0.80$ </td><td> $4.44 \pm 1.30 / 11.4 \pm 0.38$ </td><td> $3.98 \pm 0.89 / 10.9 \pm 1.01$ </td><td> $4.79 \pm 2.11 / 10.4 \pm 0.58$ </td><td> $5.22 \pm 2.31 / 10.1 \pm 0.78$ </td></tr></table>

the data for our model are defined as follows:

$$
x _ {k + 1} = A x _ {k} + B u _ {k} - \sum_ {j = 1} ^ {k + 1} D (\alpha , j) x _ {k + 1 - j} + w _ {k}, \tag {30}
$$

where $w_{k}$ represents the process noise.

Theoretical approach validation. By varying the number of trajectories used in the system identification process, we empirically validate the sample complexity results presented in Theorem 3.3 as shown in Fig. 2. We also compare our method with the traditional end-to-end learning framework for LTI systems (see Appendix I). The mean MSE of FOLTI system learning framework decreases by 81.36% compared with LTI system learning framework.

![](images/4c80a2fb4085be86ee52a2318d9c7c1c8851eb846507576dfcdf2e48b96a3a46.jpg)

<details>
<summary>line</summary>

| 1/√p | Mean L1 Loss (LQR Loss) | Mean L1 Loss (LQR Loss) |
|------|--------------------------|--------------------------|
| 0.01 | 0.015                    | 0.004                    |
| 0.02 | 0.025                    | 0.006                    |
| 0.03 | 0.045                    | 0.010                    |
</details>

Figure 2. Sample complexity bound simulation.

![](images/c1e572507ca6ef27a2733977d5a39b65692b06aa174a8767a9a0ec087f9f3fe2.jpg)

<details>
<summary>line</summary>

| t  | X_LTI | X_FOLTTLI | X_true |
|----|-------|-----------|--------|
| 0  | 2.0   | 0.5       | -2.0   |
| 5  | 4.0   | 1.0       | 4.0    |
| 10 | 1.0   | 0.5       | -1.0   |
| 15 | 0.5   | 0.2       | -0.5   |
| 20 | 0.2   | 0.1       | -0.2   |
| 25 | 0.1   | 0.05      | -0.1   |
| 30 | 0.05  | 0.02      | -0.05  |
</details>

Figure 3. Numerical simulation.

FOLOC framework evaluation. We evaluate our proposed data-driven approach by varying different variables. Our main findings are as follows: (i) FOLOC framework effectively captures the underlying system dynamics with high accuracy and remains robust to variations in fractional orders, system dimensions, noise types, noise scales, and training samples. (ii) The model's ability to learn the system dynamics improves as the time horizon increases. (iii) FOLOC framework can be trained using a small number of samples.

Varying fractional orders. By varying the fractional order $\alpha$ , we observe that FOLOC framework effectively captures long-term dependencies. The results of our experiments on commensurate FOLTI systems (see Appendix C) with different fractional orders $\alpha = 0.1, 0.3, 0.5, 0.7, 0.9$ are shown in Table 1. As the fractional order $\alpha$ increases, we observe that the MSE and MAE remain approximately the same, with an overall mean MSE of 0.40% (standard deviation: 0.21%) and an overall mean MAE of 0.95% (standard deviation: 0.55%).

Varying time horizons. We show the results of learning fractional-order dynamics with different time horizons T = 8, 16, 32, 64, 128 in Table 1. As the time horizon increases, FOLOC framework demonstrates improved ability to capture the fractional-order dynamics. Compared with at T = 8, the MSE and MAE at T = 128 of FOLOC framework decrease by 97.37% and 92.23% respectively.

Varying state dimensions. We vary the system state dimension n = 1, 2, 4, 6, 8 to evaluate its impact on the learning process (assume n = m). Increasing the state dimension significantly raises the learning complexity due to the larger number of variables and the interactions or couplings between the states. Despite this, FOLOC frame-

work demonstrates consistent performance, with an overall mean MSE of 0.45% (standard deviation: 0.15%) and an overall mean MAE of 1.24% (standard deviation: 0.36%). Compared with at n = 2, the MSE (MAE) of FOLOC framework decreases by 13.01% (18.79%) at n = 6 and 24.06% (25.45%) at n = 8. This experiment demonstrates the scalability of our model, and verifies its ability to capture the interactions within the fractional dynamics of the state.

![](images/bfb7c0963ab294e03fdad58e74bb2b9d06bc5c5363ebe5a335b5991a49f46e14.jpg)

<details>
<summary>line</summary>

| Time Steps | MSE   | MAE   |
| ---------- | ----- | ----- |
| 2^3        | 0.10  | 0.11  |
| 2^4        | 0.045 | 0.06  |
| 2^5        | 0.015 | 0.035 |
| 2^6        | 0.005 | 0.02  |
| 2^7        | 0.00  | 0.00  |
</details>

Figure 4. Vary time horizons.

Varying training samples. We show the model performance with different number of training samples N = 1000, 2000, 4000, 6000, 8000. Compared with at N = 8000, the MSE only decreases by 1.65% at N = 1000. This demonstrates that FOLOC framework is sample-efficient and capable of learning the underlying fractional-order dynamics with a limited number of samples.

Varying noise distributions. By varying the distribution of process noise, we demonstrate the robustness and effectiveness of FOLOC framework. We evaluate the model's performance by simulating the noise from the following probability distribution functions: Gaussian, Poisson, Uniform, Gamma, Sinc-squared (Adigun & Kosko, 2023; Zhang et al., 2024), and Cauchy. Compared with using Gaussian noise, the maximal increases of MSE/MAE by $24.42\% / 6.67\%$ (Cauchy/Sinc-squared). The empirical results (Table 1) show that FOLOC framework can not only predict an approximately accurate optimal control sequence but also effectively estimate the system parameters.

Varying noise scales. FOLOC framework shows the ability to learning the fractional-order dynamics with data corrupted by different Gaussian noise scales. As data is corrupted by different scales of Gaussian noise, FOLOC framework demonstrates consistent performance, with an overall mean MSE of 0.67% (standard deviation: 0.19%) and an overall mean MAE of 1.75% (standard deviation: 0.18%).

The additional results also demonstrate that the FOLOC framework is computationally efficient (see Appendix J.3) and capable of learning from multiple FOLTI systems (see Appendix J.2). We also evaluate the performance by varying the weight parameter $\lambda_{w}$ in the loss function (28) (see Appendix H.2).

# 4.2. Performance Evaluation on Real-World Systems

To demonstrate that our deep learning model can effectively capture complex system dynamics, we evaluate it on two practical scenarios: cart-pole dynamics and quadrotor dynamics (Zhou & Tzoumas, 2024). The detail of system dynamics is given in Appendix E.3.

Cart Pole. The empirical experiments show that FOLOC framework can correctly capture the underlying system dynamics even when we add some large-scale Gaussian noise. The overall mean MSE is less than $3.99 \times 10^{-5}$ and the overall mean MAE is less than $3.39 \times 10^{-3}$ .

Table 2. Model performance in the cart pole experiment. 

<table><tr><td rowspan="2">Metrics</td><td colspan="4">Noise Scale (σ)</td><td rowspan="2">No noise</td></tr><tr><td>1</td><td> $10^{-1}$ </td><td> $10^{-2}$ </td><td> $10^{-3}$ </td></tr><tr><td>MSE ( $10^{-5}$ )</td><td>4.0</td><td>3.9</td><td>4.1</td><td>3.9</td><td>3.9</td></tr><tr><td>MAE ( $10^{-3}$ )</td><td>3.4</td><td>3.4</td><td>3.4</td><td>3.3</td><td>3.3</td></tr></table>

Quadrotor. The empirical experiments show that FOLOC framework is robust to different Gaussian noise scales when learning the Quadrotor dynamics. The overall mean MSE is 0.13 (standard deviation: 0.052), and the overall mean MAE is 0.21 (standard deviation: 0.086).

Table 3. Model performance in the quadrotor experiment. 

<table><tr><td rowspan="2">Metrics</td><td colspan="4">Noise Scale ( $\sigma$ )</td><td rowspan="2">Nonoise</td></tr><tr><td>1</td><td> $10^{-1}$ </td><td> $10^{-2}$ </td><td> $10^{-3}$ </td></tr><tr><td>MSE ( $10^{-1}$ )</td><td>1.27</td><td>1.27</td><td>1.25</td><td>1.32</td><td>1.23</td></tr><tr><td>MAE ( $10^{-1}$ )</td><td>2.10</td><td>2.14</td><td>2.08</td><td>2.13</td><td>2.06</td></tr></table>

# 5. Conclusion

This paper introduces a theoretical learning framework for optimal control of FOLTI systems, addressing key challenges in system identification, control design, and sample complexity analysis. By extending LQR to the fractional-order setting, we derive analytical solutions and provide sample complexity guarantees. Based on this theoretical foundation, we propose FOLOC, an end-to-end framework that jointly estimates system parameters and learns optimal control policies from noisy trajectories. Unlike classical methods, FOLOC framework is designed to operate effectively in real-world, noisy environments, ensuring robustness against non-Gaussian noise and limited training samples. Extensive simulations and real-world applications demonstrate the scalability, robustness, and computational efficiency of our approach.

Limitation: While our framework is effective for FOLTI systems, its LTI-based design requires further validation in non-LTI settings. Future work will extend the framework to non-LTI dynamics, enhancing its applicability to more complex systems.

# Impact Statement

Fractional-order dynamical systems can model complex dynamical behaviors characterized by long-range dependencies and memory effects, yet their practical implementation remains challenging due to the lack of system identification and control strategies. This work advances the field by formulating both theoretical and deep learning frameworks for solving fractional-order optimal control via the LQR. For theoretical learning, we provide a new fractional-order system identification and a new sample complexity analysis that gives guarantees on the number of samples required for achieving reliable control performance. Beyond theoretical contributions, we develop a novel end-to-end data-driven learning framework that jointly estimates system parameters and optimizes control policies directly from observed trajectories. Unlike traditional methods that rely on strong assumptions (e.g., noiseless data or structured disturbances), our approach remains robust under realistic, noisy environments with non-Gaussian uncertainties.

By bridging theoretical foundations with deep learning techniques, our work enables scalable and efficient optimal control for fractional-order dynamical systems in domains such as biomedical engineering, neuroscience, and financial modeling. Experimental results validate the effectiveness of our approach, demonstrating accurate system identification and control policy learning across various noise distributions and system complexities.

This research lays the groundwork for future advancements in learning-based control of fractional-order dynamical systems, paving the way for real-world deployment of fractional-order optimal controllers in autonomous systems, robotics, and large-scale networks.

# Acknowledgement

The authors acknowledge the support by the U.S. Army Research Office (ARO) under Grant No. W911NF-23-1-0111, the National Science Foundation (NSF) under the Career Award CPS-1453860, MCB-1936775, CNS-1932620 and the NSF award No. 2243104 under the Center for Complex Particle Systems (COMPASS), the Defense Advanced Research Projects Agency (DARPA) Young Faculty Award and DARPA Director Fellowship Award under Grant Number N66001-17-1-4044, Intel faculty awards and a Northrop Grumman grant. P.B. is also grateful to National Institute of Health (NIH) for the grants R01 AG 079957 “Interpretable machine learning to synergize brain age estimation and neuroimaging genetics” and RF1 AG 082201 “Neurovascular calcification and ADRD in two nonindustrial Native American populations”. It was a wonderful experience designing and writing the grant application entitled “Neurovascular calcification and ADRD in two nonindustrial Native Ameri can populations" and awarded under RF1 AG 082201. The views, opinions, and/or findings in this article are those of the authors and should not be interpreted as official views or policies of the Department of Defense, the National Institute of Health or the National Science Foundation.

# References

Adigun, O. and Kosko, B. Hidden priors for bayesian bidirectional backpropagation. In 2023 IEEE International Conference on Systems, Man, and Cybernetics (SMC), pp. 445–451. IEEE, 2023.   
Anderson, B. D. and Moore, J. B. Optimal control: linear quadratic methods. Courier Corporation, 2007.   
Aronszajn, N. Theory of reproducing kernels. Transactions of the American mathematical society, 68(3):337–404, 1950.   
Chan, R. H.-F. and Jin, X.-Q. An introduction to iterative Toeplitz solvers. SIAM, 2007.   
Chandrasekaran, S. and Sayed, A. H. A fast stable solver for nonsymmetric toeplitz and quasi-toeplitz systems of linear equations. SIAM Journal on Matrix Analysis and Applications, 19(1):107–139, 1998.   
Chatterjee, S. and Pequito, S. On learning discrete-time fractional-order dynamical systems. In 2022 American Control Conference (ACC), pp. 4335–4340. IEEE, 2022.   
Chen, S., Billings, S. A., and Grant, P. Non-linear system identification using neural networks. International journal of control, 51(6):1191–1214, 1990.   
Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., and Bengio, Y. Learning phrase representations using rnn encoder-decoder for statistical machine translation, 2014. URL https://arxiv.org/abs/1406.1078.   
Cho, Y. and Saul, L. Kernel methods for deep learning. Advances in neural information processing systems, 22, 2009.   
Dean, S., Mania, H., Matni, N., Recht, B., and Tu, S. Regret bounds for robust adaptive control of the linear quadratic regulator, 2018. URL https://arxiv.org/abs/1805.09388.   
Dean, S., Mania, H., Matni, N., Recht, B., and Tu, S. On the sample complexity of the linear quadratic regulator. Foundations of Computational Mathematics, 20(4):633-679, 2020.   
Dinuzzo, F. and Schölkopf, B. The representer theorem for hilbert spaces: a necessary and sufficient condition.

Advances in neural information processing systems, 25, 2012.   
Faradonbeh, M. K. S., Tewari, A., and Michailidis, G. Finite time identification in unstable linear systems, 2018. URL https://arxiv.org/abs/1710.01852.   
Gedon, D., Wahlström, N., Schön, T. B., and Ljung, L. Deep state space models for nonlinear system identification. IFAC-PapersOnLine, 54(7):481–486, 2021.   
Ghorbani, M., Jonckheere, E. A., and Bogdan, P. Gene expression is not random: scaling, long-range cross-dependence, and fractal characteristics of gene regulatory networks. Frontiers in physiology, 9:1446, 2018.   
Guermah, S., Djennoune, S., and Bettayeb, M. Discrete-time fractional-order systems: Modeling and stability issues. Advances in Discrete Time Systems, pp. 183–212, 2012.   
Gupta, G., Xiao, X., and Bogdan, P. Multiwavelet-based operator learning for differential equations. Advances in neural information processing systems, 34:24048–24062, 2021.   
Gupta, G., Xiao, X., Balan, R., and Bogdan, P. Non-linear operator approximations for initial value problems. In International Conference on Learning Representations (ICLR), 2022.   
Hendriks, J. N., Gustafsson, F. K., Ribeiro, A. H., Wills, A. G., and Schön, T. B. Deep energy-based narx models, 2020. URL https://arxiv.org/abs/2012.04136.   
Hilfer, R. Applications of fractional calculus in physics. World scientific, 2000.   
Hochreiter, S. Long short-term memory. Neural Computation MIT-Press, 1997.   
Ionescu, C., Lopes, A., Copot, D., Machado, J. T., and Bates, J. H. The role of fractional calculus in modeling biological phenomena: A review. Communications in Nonlinear Science and Numerical Simulation, 51:141–159, 2017.   
Ivanov, P. C., Amaral, L. A. N., Goldberger, A. L., Havlin, S., Rosenblum, M. G., Struzik, Z. R., and Stanley, H. E. Multifractality in human heartbeat dynamics. Nature, 399(6735):461–465, 1999.   
Jelich, C., Karimi, M., Kessissoglou, N., and Marburg, S. Efficient solution of block toeplitz systems with multiple right-hand sides arising from a periodic boundary element formulation. Engineering Analysis with Boundary Elements, 130:135–144, 2021.

Kalman, R. E. et al. Contributions to the theory of optimal control. Bol. soc. mat. mexicana, 5(2):102–119, 1960.   
Kalouptsidis, N., Carayannis, G., and Manolakis, D. Fast algorithms for block toeplitz matrices with toeplitz entries. Signal Processing, 6(1):77–81, 1984.   
Langley, P. Crafting papers on machine learning. In Langley, P. (ed.), Proceedings of the 17th International Conference on Machine Learning (ICML 2000), pp. 1207–1216, Stanford, CA, 2000. Morgan Kaufmann.   
Li, Y. and Chen, Y. Fractional order linear quadratic regulator. In 2008 IEEE/ASME International Conference on Mechtronic and Embedded Systems and Applications, pp. 363–368. IEEE, 2008.   
Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., and Anandkumar, A. Fourier neural operator for parametric partial differential equations, 2021. URL https://arxiv.org/abs/2010.08895.   
Lundstrom, B. N., Higgs, M. H., Spain, W. J., and Fairhall, A. L. Fractional differentiation by neocortical pyramidal neurons. Nature neuroscience, 11(11):1335–1342, 2008.   
Luo, J., Wang, J., Wang, H., huanshuo dong, Geng, Z., Chen, H., and Kuang, Y. Neural krylov iteration for accelerating linear system solving. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=cqfE9eYMdP.   
Lusch, B., Kutz, J. N., and Brunton, S. L. Deep learning for universal linear embeddings of nonlinear dynamics. Nature communications, 9(1):4950, 2018.   
Masti, D. and Bemporad, A. Learning nonlinear state–space models using autoencoders. Automatica, 129:109666, 2021.   
Monje, C. A., Chen, Y., Vinagre, B. M., Xue, D., and Feliu-Batlle, V. Fractional-order systems and controls: fundamentals and applications. Springer Science & Business Media, 2010.   
Narendra, K. and Parthasarathy, K. Identification and control of dynamical systems using neural networks. IEEE Transactions on Neural Networks, 1(1):4–27, 1990. doi:10.1109/72.80202.   
Oldham, K. and Spanier, J. The fractional calculus theory and applications of differentiation and integration to arbitrary order. Elsevier, 1974.   
Saad, Y. and Schultz, M. H. Gmres: A generalized minimal residual algorithm for solving nonsymmetric linear

systems. SIAM Journal on scientific and statistical computing, 7(3):856–869, 1986.   
Sarkar, T., Rakhlin, A., and Dahleh, M. A. Finite time lti system identification. Journal of Machine Learning Research, 22(26):1–61, 2021. URL http://jmlr.org/papers/v22/19-725.html.   
Simchowitz, M., Mania, H., Tu, S., Jordan, M. I., and Recht, B. Learning without mixing: Towards a sharp analysis of linear system identification, 2018. URL https://arxiv.org/abs/1802.08334.   
Strang, G. A proposal for toeplitz matrix calculations. Studies in Applied Mathematics, 74(2):171–176, 1986.   
Trench, W. F. Solution of systems with toeplitz matrices generated by rational functions. Linear algebra and its applications, 74:191–211, 1986.   
Tu, S. and Recht, B. The gap between model-based and model-free methods on the linear quadratic regulator: An asymptotic viewpoint. In Beygelzimer, A. and Hsu, D. (eds.), Proceedings of the Thirty-Second Conference on Learning Theory, volume 99 of Proceedings of Machine Learning Research, pp. 3036–3083. PMLR, 25–28 Jun 2019. URL https://proceedings.mlr.press/v99/tu19a.html.   
Wishart, J. The generalised product moment distribution in samples from a normal multivariate population. Biometrika, pp. 32–52, 1928.   
Xiao, X., Cao, D., Yang, R., Gupta, G., Liu, G., Yin, C., Balan, R., and Bogdan, P. Coupled multiwavelet operator learning for coupled differential equations. In The Eleventh International Conference on Learning Representations, 2022.   
Yaghooti, B. and Sinopoli, B. Inferring dynamics of discrete-time, fractional-order control-affine nonlinear systems. In 2023 American Control Conference (ACC), pp. 935–940. IEEE, 2023.   
Zancato, L. and Chiuso, A. A novel deep neural network architecture for non-linear system identification, 2021. URL https://arxiv.org/abs/2106.03078.   
Zhang, X., Zuo, J., and Shao, X. Efficient neural decoder: Mixture-regularized bidirectional gru with attention. In 2024 International Joint Conference on Neural Networks (IJCNN), pp. 1–7, 2024. doi: 10.1109/IJCNN60899.2024.10651174.   
Zhang, X., Gupta, V., and Bogdan, P. A sampling complexity-aware framework for discrete-time fractional-order dynamical system identification, 2025. URL https://arxiv.org/abs/2501.17499.

Zhou, H. and Tzoumas, V. Simultaneous system identification and model predictive control with no dynamic regret, 2024. URL https://arxiv.org/abs/2407.04143.

# A. Code Availability

The source code is available at https://anonymous.4open.science/r/Fractional-Order-Learning-for-Control-Framework-69A0.

# B. Related Work

While research on system identification and optimal control for FOLTI systems remains limited, significant progress has been made in LTI systems. Estimating unknown parameters of linear dynamical systems is a well-established subfield of system identification in control theory (Sarkar et al., 2021; Faradonbeh et al., 2018; Simchowitz et al., 2018), where the task is to estimate parameters from input-output time series generated by the underlying system. With the increasing interest in machine learning and deep learning, many researchers use different deep learning based modeling techniques to solve general nonlinear system identification. Chen et al. (1990) shows a single hidden layer neural network can identify discrete time nonlinear systems and derives new parameter estimation algorithms based on a prediction error formulation. Narendra & Parthasarathy (1990) demonstrates that neural networks can be used effectively for both identification and control of nonlinear dynamical systems. The hierarchical structures of multilayer feedforward neural networks can include dynamic systems features (Zancato & Chiuso, 2021) and bring extra flexibility for probabilistic approaches (Hendriks et al., 2020). Kernel-based methods also have been studied in system identification. Linear system identification can be seen as an application of learning the impulse response function, which can be approximated by kernels (Aronszajn, 1950; Dinuzzo & Schölkopf, 2012; Cho & Saul, 2009). Deep state-space models (Gedon et al., 2021) like recurrent neural networks (Hochreiter, 1997; Cho et al., 2014) and autoencoders (Masti & Bemporad, 2021; Lusch et al., 2018) are also gaining in popularity for system identification. For Another critical aspect of optimal control via LQR is the design of control inputs, which are typically formulated as linear combinations of disturbance processes (Dean et al., 2020).

Solving the optimal control problem via LQR involves addressing a potentially large linear system with a block Toeplitz matrix of size $R^{Tn\times Tn}$ . While solutions for Toeplitz and block Toeplitz systems have been extensively studied (Trench, 1986; Kalouptsidis et al., 1984; Chandrasekaran & Sayed, 1998), they still present significant computational challenges. Direct solution methods for Toeplitz systems typically exhibit a computational complexity of $\mathcal{O}(N^{2})$ , where N represents the degrees of freedom. However, the high memory requirements of these methods often limit their applicability to large-scale problems. Iterative solvers (Chan & Jin, 2007; Strang, 1986) are more memory-efficient and can achieve a complexity of $\mathcal{O}(N\log^{2}(N))$ for a single linear block Toeplitz system. Advanced approaches, such as global and block variants of the generalized minimal residual (GMRES) method (Saad & Schultz, 1986), can efficiently address sequences of block Toeplitz systems with multiple right-hand sides (Jelich et al., 2021). Machine learning-based methods, such as neural operators (Li et al., 2021; Gupta et al., 2021; 2022; Xiao et al., 2022), offer a promising alternative for accelerating the solution of linear systems. Neural operator-assisted Krylov iterations (Luo et al., 2024), for example, have demonstrated significant computational advantages, achieving up to a $5.5\times$ speedup in computation time and a $16.1\times$ reduction in the number of iterations. These advancements underscore the potential of machine learning techniques to address computational bottlenecks in optimal control for FOLTI systems.

# C. Why Fractional Order?

According to the well-known definition, the first-order derivative of the function $f(t)$ , denoted by $\mathcal{D}^{1}f(t)$ , is defined by

$$
\mathcal {D} ^ {1} f (t) = \frac {d f (t)}{d t} = \lim _ {h \rightarrow 0} \frac {f (t) - f (t - h)}{h}, \tag {31}
$$

that is, as the limit of a backward difference. Similarly,

$$
\mathcal {D} ^ {2} f (t) = \frac {d ^ {2} f (t)}{d t ^ {2}} = \lim _ {h \rightarrow 0} \frac {1}{h ^ {2}} [ f (t) - 2 f (t - h) + f (t - 2 h) ] \tag {32}
$$

and

$$
\mathcal {D} ^ {3} f (t) = \frac {d ^ {3} f (t)}{d t ^ {3}} = \lim _ {h \rightarrow 0} \frac {1}{h ^ {3}} [ f (t) - 3 f (t - h) + 3 f (t - 2 h) - f (t - 3 h) ]. \tag {33}
$$

Iterating n-times, we can obtain

$$
\mathcal {D} ^ {n} f (t) = \frac {d ^ {n} f (t)}{d t ^ {n}} = \lim _ {h \rightarrow 0} \frac {1}{h ^ {n}} \sum_ {k = 0} ^ {n} (- 1) ^ {n} \binom {n} {k} f (t - k h), \tag {34}
$$

where

$$
\binom {n} {k} = \frac {n (n - 1) (n - 2) \cdots (n - k + 1)}{k !}. \tag {35}
$$

Naturally we can extend the common binomial coefficients for any $n \in \mathbb{R}^+$ by letting $(n - 1)! = \Gamma(n)$ . So the Riemann-Liouville fractional-order integral can be defined as a consequence of Cauchy's formula for repeated integrals.

Definition C.1 (Riemann-Liouville fractional-order integral). Given a positive real number $\alpha$ , the Riemann-Liouville fractional-order integral is defined as

$$
\mathcal {I} _ {c} ^ {\alpha} f (t) := \frac {1}{\Gamma (\alpha)} \int_ {c} ^ {t} (t - \tau) ^ {\alpha - 1} f (\tau) \mathrm{d} \tau , \quad t > c, \alpha \in \mathbb {R} ^ {+}. \tag {36}
$$

Consider an integer-order derivative operator $D^{n}$ ( $n \in N$ ). The derivative operator $D^{n}$ is only a left-inverse of the integral operator $I^{n}$ such that $D^{n}I^{n} = I$ and $I^{n}D^{n} \neq I$ . In order to generalize integer-order derivatives, the fractional-order derivative operator $D^{\alpha}$ should follow the same left-inverse rule. We show the definition of the Riemann-Liouville fractional-order derivative as follows.

Definition C.2 (Riemann–Liouville fractional-order derivative). Let $f : R \to R$ be a function and let $I = [a, b]$ be a finite interval on the real axis R. The left-side and right-side fractional-order derivatives $^{RL}D_{a+}^{\alpha}f$ and $^{RL}D_{b-}^{\alpha}f$ of order $\alpha \in R$ are defined by

$$
\left(^ {R L} \mathcal {D} _ {a +} ^ {\alpha} f\right) (x) = \frac {1}{\Gamma (n - \alpha)} \left(\frac {d}{d x}\right) ^ {n} \int_ {a} ^ {x} \frac {f (t) d t}{(x - t) ^ {\alpha - n + 1}}, \tag {37}
$$

$$
\left(^ {R L} \mathcal {D} _ {b -} ^ {\alpha} f\right) (x) = \frac {1}{\Gamma (n - \alpha)} \left(- \frac {d}{d x}\right) ^ {n} \int_ {x} ^ {b} \frac {f (t) d t}{(t - x) ^ {\alpha - n + 1}}, \tag {38}
$$

where $n - 1 \leq \alpha \leq n, x > a$ in (37), and $x < b$ in (38).

The Grünwald–Letnikov fractional-order derivative is defined as the limit of finite differences and is mathematically equivalent to the Riemann–Liouville fractional-order derivative. The Grünwald–Letnikov fractional-order derivative plays an important role establishing fractional-order dynamical systems. The equations for a continuous-time fractional-order dynamical system can be written as follows:

$$
H \left(\mathcal {D} ^ {\alpha_ {0} \alpha_ {1} \alpha_ {2} \dots \alpha_ {n}}\right) (y _ {1}, y _ {2}, \dots , y _ {l}) = G \left(\mathcal {D} ^ {\beta_ {0} \beta_ {1} \beta_ {2} \dots \beta_ {m}}\right) (u _ {1}, u _ {2}, \dots , u _ {k}), \tag {39}
$$

where $y_{i}, u_{i}$ are functions of time and $H(\cdot), G(\cdot)$ are the combination laws of the fractional-order derivative operator.

A single-input single-output LTI fractional-order system can be described by a fractional differential equation of the form

$$
\sum_ {k = 0} ^ {n} a _ {k} \mathcal {D} ^ {\alpha_ {k}} y (t) = \sum_ {k = 0} ^ {m} b _ {k} \mathcal {D} ^ {\beta_ {k}} u (t), \tag {40}
$$

with a corresponding transfer function of the form:

$$
G (s) = \frac {Y (s)}{U (s)} = \frac {\sum_ {k = 0} ^ {m} b _ {k} s ^ {\beta_ {k}}}{\sum_ {k = 0} ^ {n} a _ {k} s ^ {\alpha_ {k}}}, \tag {41}
$$

where $a_{k}, b_{k} \in \mathbb{R}$ . The system is said to be commensurate if $\alpha_{k} = \beta_{k} = k\alpha$ , where $\alpha \in \mathbb{R}^{+}$ , otherwise the system is non-commensurate.

We can use the Grünwald–Letnikov fractional-order derivative to rewrite the fractional-order differential equation as follows in the discrete-time case:

$$
\sum_ {k = 0} ^ {n} a _ {k} \Delta_ {h} ^ {\alpha_ {k}} y (t) = \sum_ {k = 0} ^ {m} b _ {k} \Delta_ {h} ^ {\beta_ {k}} u (t), \tag {42}
$$

with a corresponding transfer function of the following form:

$$
G (z) = \frac {\sum_ {k = 0} ^ {m} b _ {k} \left(\omega (z ^ {- 1})\right) ^ {\beta_ {k}}}{\sum_ {k = 0} ^ {n} a _ {k} \left(\omega (z ^ {- 1})\right) ^ {\alpha_ {k}}}, \tag {43}
$$

where $\omega (z^{-1})$ is the $Z$ transform of the operator $\Delta_h^1$ .

# D. Theorems and Proofs

# D.1. System Identification

We aim to estimate the parameters $\Theta = \{A, B, \alpha\}$ of the following FOLTI system:

$$
x _ {k + 1} = A x _ {k} + B u _ {k} - \sum_ {j = 1} ^ {k + 1} \psi (\alpha , j) x _ {k + 1 - j}. \tag {44}
$$

To note, we can not identify $\Theta$ jointly so we make some assumptions during the system identification.

# D.1.1. DATA GENERATION

In this step, we generate data following the procedure outlined below:

$$
\begin{array}{l} x _ {1} ^ {1} = A x _ {0} ^ {1} + B u _ {0} ^ {1} - \psi (\alpha , 1) x _ {0} ^ {1} + w _ {0} ^ {1} \\ = A x _ {0} ^ {1} + B u _ {0} ^ {1} + C _ {\alpha} x _ {0} ^ {1} + w _ {0} ^ {1} \tag {45} \\ \end{array}
$$

$$
\begin{array}{l} x _ {1} ^ {2} = A x _ {0} ^ {2} + B u _ {0} ^ {2} - \psi (\alpha , 1) x _ {0} ^ {2} + w _ {0} ^ {2} \\ = A x _ {0} ^ {2} + B u _ {0} ^ {2} + C _ {\alpha} x _ {0} ^ {2} + w _ {0} ^ {2} \tag {46} \\ \end{array}
$$

...

$$
x _ {1} ^ {p} = A x _ {0} ^ {p} + B u _ {0} ^ {p} - \psi (\alpha , 1) x _ {0} ^ {p} + w _ {0} ^ {p}
$$

$$
= A x _ {0} ^ {p} + B u _ {0} ^ {p} + C _ {\alpha} x _ {0} ^ {p} + w _ {0} ^ {p} \tag {47}
$$

where $x_{0}^{i}$ are the i-th initial conditions, $x_{1}^{i}$ represents the corresponding state at the second time step generated from the initial condition, and $w_{0}^{i}$ are independent and identically distributed (i.i.d.) white Gaussian noise for $i = 1, \ldots, p$ , and $C_{\alpha} = \text{diag}(\alpha_{1}, \alpha_{2}, \ldots, \alpha_{n})$ .

# D.1.2. ESTIMATION

We can write (86) as follows

$$
\begin{array}{l} x _ {1} ^ {i} = A x _ {0} ^ {i} + B u _ {0} ^ {i} + C _ {\alpha} x _ {0} ^ {i} + w _ {0} ^ {i} \\ = \underbrace {\left[ \begin{array}{c c c c} \gamma_ {1} ^ {1} + \alpha_ {1} & \gamma_ {2} ^ {1} & \cdots & \gamma_ {n} ^ {1} \\ \gamma_ {1} ^ {2} & \gamma_ {2} ^ {2} + \alpha_ {2} & \cdots & \gamma_ {n} ^ {2} \\ \vdots & \vdots & \cdots & \vdots \\ \gamma_ {1} ^ {n} & \gamma_ {2} ^ {n} & \cdots & \gamma_ {n} ^ {n} + \alpha_ {n} \end{array} \right]} _ {A _ {\alpha}} x _ {0} ^ {i} + \underbrace {\left[ \begin{array}{c c c c} \beta_ {1} ^ {1} & \beta_ {2} ^ {1} & \cdots & \beta_ {m} ^ {1} \\ \beta_ {1} ^ {2} & \beta_ {2} ^ {2} & \cdots & \beta_ {m} ^ {2} \\ \vdots & \vdots & \cdots & \vdots \\ \beta_ {1} ^ {n} & \beta_ {2} ^ {n} & \cdots & \beta_ {m} ^ {n} \end{array} \right]} _ {B} u _ {0} ^ {i} + w _ {0} ^ {i} \\ = \underbrace {\left[ \begin{array}{c c c c} x _ {0} ^ {i ^ {\top}} & 0 & \cdots & \cdots \\ 0 & x _ {0} ^ {i ^ {\top}} & 0 & \cdots \\ \vdots & \vdots & \ddots & \vdots \\ 0 & \cdots & \cdots & x _ {0} ^ {i ^ {\top}} \end{array} \right]} _ {\pi_ {i}} \left[ \begin{array}{c} \gamma_ {1} \\ \vdots \\ \gamma_ {n} \end{array} \right] + \underbrace {\left[ \begin{array}{c c c c} u _ {0} ^ {i ^ {\top}} & 0 & \cdots & \cdots \\ 0 & u _ {0} ^ {i ^ {\top}} & 0 & \cdots \\ \vdots & \vdots & \ddots & \vdots \\ 0 & \cdots & \cdots & u _ {0} ^ {i ^ {\top}} \end{array} \right]} _ {\phi_ {i}} \left[ \begin{array}{c} \beta_ {1} \\ \vdots \\ \beta_ {n} \end{array} \right] + w _ {0} ^ {i} \tag {48} \\ \end{array}
$$

Thus, given the generated data, we solve the following least squares problem:

$$
\underbrace {\left[ \begin{array}{c} x _ {1} ^ {1} \\ \vdots \\ x _ {1} ^ {p} \end{array} \right]} _ {X} = \underbrace {\left[ \begin{array}{c} \pi_ {1} \\ \vdots \\ \pi_ {p} \end{array} \right]} _ {\pi} \underbrace {\left[ \begin{array}{c} \gamma_ {1} \\ \vdots \\ \gamma_ {n} \end{array} \right]} _ {\gamma} + \underbrace {\left[ \begin{array}{c} \phi_ {1} \\ \vdots \\ \phi_ {p} \end{array} \right]} _ {\phi} \underbrace {\left[ \begin{array}{c} \beta_ {1} \\ \vdots \\ \beta_ {n} \end{array} \right]} _ {\beta} + \underbrace {\left[ \begin{array}{c} w _ {0} ^ {1} \\ \vdots \\ w _ {0} ^ {p} \end{array} \right]} _ {\omega}, \tag {49}
$$

$$
X = \pi \gamma + \phi \beta + w = \left[ \begin{array}{l l} \pi & \phi \end{array} \right] \left[ \begin{array}{l} \gamma \\ \beta \end{array} \right] + w. \tag {50}
$$

Then we have the following optimization problem

$$
\hat {\theta} = \arg \min _ {\theta} \| X - \xi \theta \| _ {2} ^ {2}, \tag {51}
$$

with the solution

$$
\hat {\theta} = \left(\xi^ {\top} \xi\right) ^ {- 1} \xi^ {\top} X, \tag {52}
$$

where $\theta = [\gamma^{\top},\beta^{\top}]^{\top}$ , and $\xi = [\pi ,\phi ]$

Follow (Zhang et al., 2025), we can revise the above strategy and formulate the sample complexity problem as follows:

$$
\theta_ {t + 1} = \theta_ {t},
$$

$$
X _ {t} = \xi \theta_ {t} + w _ {t}, \tag {53}
$$

where the state vector $\theta_{t}=[\gamma^{\top},\beta^{\top}]^{\top}$ , and $w_{k}$ is a white Gaussian noise vector $w_{t}\sim\mathcal{N}(0,K_{w})$ , where $K_{w}$ is a diagonal covariance matrix, and $t=1,2,\ldots,N$ . Thus, the solution $\hat{\theta}$ has the following Gaussian distribution:

$$
\hat {\theta} \sim N \left(\theta , \frac {1}{N} (\xi \top \xi) ^ {- 1} \xi \top K _ {w} \xi (\xi \top \xi) ^ {- 1}\right). \tag {54}
$$

Notice that the matrix A and the fractional order $\alpha$ are coupled. Therefore, if the diagonal elements of A are known, the fractional order $\alpha$ can be identified by subtracting these constant values from the corresponding positions in the estimated parameter vector $\hat{\theta}$ .

# D.2. Proof of Lemma 3.1

Proof. By mathematical induction as follows:

(1) When $t = 0, x[0] = G_0x[0] = x[0]$ . When $t = 1, x[1] = G_1x[0] + Bu[0]$   
(2) Assume we have $x[k] = G_kx[0] + \sum_{j=0}^{k-1} G_{k-1-j}Bu[j]$ , when $t = k$ .

When $t = k + 1$ ,

$$
\begin{array}{l} x [ k + 1 ] = \sum_ {j = 0} ^ {k} A _ {j} x [ k - j ] + B u [ k ] \\ = \sum_ {j = 0} ^ {k} (A _ {j} (G _ {k - j} x [ 0 ] + \sum_ {i = 0} ^ {k - j - 1} G _ {k - 1 - i - j} B u [ i ])) + B u [ k ] \\ = \sum_ {j = 0} ^ {k} A _ {j} G _ {k - j} x [ 0 ] + \sum_ {j = 0} ^ {k} A _ {j} (\sum_ {i = 0} ^ {k - j - 1} G _ {k - 1 - i - j} B u [ i ]) + B u [ k ] \\ = G _ {k + 1} x [ 0 ] + \sum_ {j = 0} ^ {k} \sum_ {i = 0} ^ {k - j - 1} A _ {j} (G _ {k - 1 - i - j} B u [ i ]) + B u [ k ] \tag {55} \\ \end{array}
$$

By expanding the second term, thus

$$
x [ k + 1 ] = G _ {k + 1} x [ 0 ] + \sum_ {j = 0} ^ {k} G _ {k - j} B u [ j ]. \tag {56}
$$

# D.3. Proof of Theorem 3.2

Proof of least-squares solution. Rewrite as follows:

$$
\left[ \begin{array}{c} x _ {0} \\ x _ {1} \\ x _ {2} \\ x _ {3} \\ \vdots \\ x _ {T} \end{array} \right] = \underbrace {\left[ \begin{array}{c c c c c c} 0 & 0 & 0 & 0 & \cdots & 0 \\ G _ {0} B & 0 & 0 & 0 & \cdots & 0 \\ G _ {1} B & G _ {0} B & 0 & 0 & \cdots & 0 \\ G _ {2} B & G _ {1} B & G _ {0} B & 0 & \cdots & 0 \\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\ G _ {T - 1} B & G _ {T - 2} B & G _ {T - 3} B & G _ {T - 4} B & \cdots & G _ {0} B \end{array} \right]} _ {G} \left[ \begin{array}{c} u _ {0} \\ u _ {1} \\ u _ {2} \\ u _ {3} \\ \vdots \\ u _ {T - 1} \end{array} \right] + \underbrace {\left[ \begin{array}{c} I \\ G _ {1} \\ G _ {2} \\ G _ {3} \\ \vdots \\ G _ {T} \end{array} \right]} _ {H} x _ {0}. \tag {57}
$$

$$
X = G U + H x _ {0} \tag {58}
$$

Rewrite as follows:

$$
J _ {T} (U) = \min _ {U} \left\{X ^ {\top} \bar {Q} X + U ^ {\top} \bar {R} U \right\} \tag {59}
$$

where

$$
\bar {Q} = \left[ \begin{array}{c c c c c c} Q & 0 & 0 & 0 & \dots & 0 \\ 0 & Q & 0 & 0 & \dots & 0 \\ 0 & 0 & Q & 0 & \dots & 0 \\ 0 & 0 & 0 & Q & \dots & 0 \\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & 0 & \dots & Q _ {f} \end{array} \right]
$$

$$
\bar {R} = \left[ \begin{array}{c c c c c c} R & 0 & 0 & 0 & \dots & 0 \\ 0 & R & 0 & 0 & \dots & 0 \\ 0 & 0 & R & 0 & \dots & 0 \\ 0 & 0 & 0 & R & \dots & 0 \\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & 0 & \dots & R \end{array} \right] \tag {60}
$$

Substitute (59) using (58):

$$
\min _ {U} J _ {T} (U) = \min _ {U} \left\{\left(G U + H x _ {0}\right) ^ {\top} \bar {Q} \left(G U + H x _ {0}\right) + U ^ {\top} \bar {R} U \right\} \tag {61}
$$

The solution for the least squares problem is

$$
(G ^ {\top} \bar {Q} G + \bar {R}) U = - G ^ {\top} \bar {Q} ^ {\top} H x _ {0}
$$

$$
U = - (G ^ {\top} \bar {Q} G + \bar {R}) ^ {- 1} G ^ {\top} \bar {Q} ^ {\top} H x _ {0} \tag {62}
$$

Proof of Lagrange multiplier solution.

Lemma D.1 (Lagrange multiplier condition). The Lagrangian function $\mathcal{L}$ is given by:

$$
\mathcal {L} (x, u, \lambda) = \sum_ {k = 0} ^ {T - 1} \left(x _ {k} ^ {\top} Q x _ {k} + u _ {k} ^ {\top} R u _ {k} + \lambda_ {k + 1} ^ {\top} (A x _ {k} + B u _ {k} - \Delta^ {\alpha} x _ {k + 1})\right) + x _ {T} ^ {\top} Q _ {f} x _ {T}. \tag {63}
$$

and the solution is given by:

$$
u _ {k} = - \frac {1}{2} R ^ {- 1} B ^ {\top} \lambda_ {k + 1},
$$

$$
\lambda_ {k} = 2 Q x _ {k} + A ^ {\top} \lambda_ {k + 1} - \sum_ {i = k + 1} ^ {T} D (\alpha , i - k) \lambda_ {i} \tag {64}
$$

with $x_0 = x_0$ and $\lambda_T = 2Q_f x_T$ .

Proof of Lemma 1. Taking the partial derivative with respect to $u_{k}, x_{k}, \lambda_{k}$ , we have

$$
\frac {\partial \mathcal {L}}{\partial u _ {k}} = 2 R u _ {k} + B ^ {\top} \lambda_ {k + 1} = 0 \implies u _ {k} = - \frac {1}{2} R ^ {- 1} B ^ {\top} \lambda_ {k + 1}, \tag {65}
$$

$$
\frac {\partial \mathcal {L}}{\partial x _ {k}} = 2 Q x _ {k} + A ^ {\top} \lambda_ {k + 1} - \sum_ {i = k} ^ {T} D (\alpha , i - k) \lambda_ {i} = 0, \tag {66}
$$

$$
\Longrightarrow \lambda_ {k} = 2 Q x _ {k} + A ^ {\top} \lambda_ {k + 1} - \sum_ {i = k + 1} ^ {T} D (\alpha , i - k) \lambda_ {i} \tag {67}
$$

$$
\frac {\partial \mathcal {L}}{\partial \lambda_ {k}} = A x _ {k} + B u _ {k} - \Delta^ {\alpha} x _ {k + 1} = 0. \tag {68}
$$

Explanation for Eq. (66): From Eq. (3), we have

$$
\Delta^ {\alpha} x _ {k} = \sum_ {j = 0} ^ {k} D (\alpha , j) x [ k - j ] \tag {69}
$$

Then, we have

$$
\frac {\partial}{\partial x _ {k}} \left(\lambda_ {k} ^ {\top} \Delta^ {\alpha} x _ {k}\right) = D (\alpha , 0) \lambda_ {k}
$$

$$
\frac {\partial}{\partial x _ {k}} \left(\lambda_ {k + 1} ^ {\top} \Delta^ {\alpha} x _ {k + 1}\right) = D (\alpha , 1) \lambda_ {k + 1}
$$

•
•
•

$$
\frac {\partial}{\partial x _ {k}} \left(\lambda_ {T} ^ {\top} \Delta^ {\alpha} x _ {T}\right) = D (\alpha , T - k) \lambda_ {T} \tag {70}
$$

Solve the $\lambda$ jointly and then substitute in (65) to get $u_{k}$ .

Note we have the following equations by (56), (65) and (66):

$$
u _ {k} = - \frac {1}{2} R ^ {- 1} B ^ {\top} \lambda_ {k + 1},
$$

$$
\lambda_ {k} = 2 Q x _ {k} + A ^ {\top} \lambda_ {k + 1} - \sum_ {i = k + 1} ^ {T} D (\alpha , i - k) \lambda_ {i}
$$

$$
x [ k ] = G _ {k} x [ 0 ] + \sum_ {j = 0} ^ {k - 1} G _ {k - 1 - j} B u [ j ] \tag {71}
$$

Obeserve that

$$
\lambda_ {k} = \left[ \begin{array}{c c c c c c c} 2 Q G _ {k} & - Q G _ {k - 1} B R ^ {- 1} B ^ {\top} & \dots & - Q G _ {0} B R ^ {- 1} B ^ {\top} & A ^ {\top} - D (\alpha , 1) & \dots & - D (\alpha , T - k) \end{array} \right] \left[ \begin{array}{c} x _ {0} \\ \lambda_ {1} \\ \vdots \\ \lambda_ {k} \\ \lambda_ {k + 1} \\ \vdots \\ \lambda_ {T} \end{array} \right] \tag {72}
$$

We have the following matrix form, where $G_{\lambda}$ and $I - G_{\lambda}$ are block Toeplitz matrices (see Definition D.2).

$$
\begin{array}{l} \left[ \begin{array}{c} \lambda_ {1} \\ \lambda_ {2} \\ \lambda_ {3} \\ \vdots \\ \lambda_ {T} \end{array} \right] = \underbrace {\left[ \begin{array}{c c c c c} - Q G _ {0} B R ^ {- 1} B ^ {\top} & A ^ {\top} - D (\alpha , 1) & - D (\alpha , 2) & \cdots & - D (\alpha , T - 1) \\ - Q G _ {1} B R ^ {- 1} B ^ {\top} & - Q G _ {0} B R ^ {- 1} B ^ {\top} & A ^ {\top} - D (\alpha , 1) & \cdots & - D (\alpha , T - 2) \\ - Q G _ {2} B R ^ {- 1} B ^ {\top} & - Q G _ {1} B R ^ {- 1} B ^ {\top} & - Q G _ {0} B R ^ {- 1} B ^ {\top} & \cdots & - D (\alpha , T - 3) \\ \vdots & \vdots & \vdots & \vdots & \ddots \\ - Q G _ {T - 1} B R ^ {- 1} B ^ {\top} & - Q G _ {T - 2} B R ^ {- 1} B ^ {\top} & \cdots & \cdots & - Q G _ {0} B R ^ {- 1} B ^ {\top} \end{array} \right]} _ {G _ {\lambda}} \left[ \begin{array}{c} \lambda_ {1} \\ \lambda_ {2} \\ \lambda_ {3} \\ \vdots \\ \lambda_ {T} \end{array} \right] \\ + 2 \underbrace {\left[ \begin{array}{c} Q G _ {1} \\ Q G _ {2} \\ Q G _ {3} \\ \vdots \\ Q G _ {T} \end{array} \right]} _ {H _ {\lambda}} x _ {0}. \tag {73} \\ \end{array}
$$

$$
\lambda = G _ {\lambda} \lambda + 2 H _ {\lambda} x _ {0}
$$

$$
(I - G _ {\lambda}) \lambda = 2 H _ {\lambda} x _ {0}
$$

$$
\lambda = 2 \left(I - G _ {\lambda}\right) ^ {- 1} H _ {\lambda} x _ {0} \tag {74}
$$

Definition D.2 (Block Toeplitz matrix). A block Toeplitz matrix is a block matrix where the block structure follows a Toeplitz pattern. Specifically, let $T \in R^{mn \times mn}$ be partitioned into $m \times m$ blocks, each of size $n \times n$ . The matrix T is defined as:

$$
\mathcal {T} = \left[ \begin{array}{c c c c} \mathcal {A} _ {0} & \mathcal {A} _ {- 1} & \dots & \mathcal {A} _ {- (m - 1)} \\ \mathcal {A} _ {1} & \mathcal {A} _ {0} & \dots & \mathcal {A} _ {- (m - 2)} \\ \vdots & \vdots & \ddots & \vdots \\ \mathcal {A} _ {m - 1} & \mathcal {A} _ {m - 2} & \dots & \mathcal {A} _ {0} \end{array} \right],
$$

where each $\mathcal{A}_k\in \mathbb{R}^{n\times n}$ represents a block matrix.

Using (65), we have

$$
U = - R ^ {- 1} B ^ {\top} \otimes (I - G _ {\lambda}) ^ {- 1} H _ {\lambda} x _ {0} \tag {75}
$$

# D.4. Proof of Theorem 3.3

Proof of least-squares sample complexity result. If we know A and $\alpha$ , then $\theta$ will reduce to $\text{vectorize}(B)$ and the distribution reduces to the following:

$$
\hat {\theta} = \text { vectorize } (\hat {B}) \sim N \left(\text { vectorize } (B), \frac {1}{N} (\phi \top \phi) ^ {- 1} \phi \top K _ {w} \phi (\phi \top \phi) ^ {- 1}\right). \tag {76}
$$

Then

$$
\begin{array}{l} \mathbb {E} \left[ | \hat {J} - J | \right] = \mathbb {E} \left[ \| \hat {J} - J \| _ {2} \right] = \mathbb {E} \left[ \| \hat {a} - a \| _ {2} \right] \\ = \mathbb {E} \left[ \| z ^ {\top} \left(\underbrace {\hat {P} (\hat {P} ^ {\top} S \hat {P} + \bar {R}) ^ {- 1} \hat {P} ^ {\top}} _ {F (\hat {P})} - \underbrace {P (P ^ {\top} S P + \bar {R}) ^ {- 1} P ^ {\top}} _ {F (P)}\right) z \| _ {2} \right] \\ \leq \mathbb {E} \left[ \| F (\hat {P}) - F (P) \| _ {2} \right] \| z \| _ {2} ^ {2} \tag {77} \\ \end{array}
$$

We rewrite $F(\hat{P}) - F(P)$ as follows:

$$
\begin{array}{l} F (\hat {P}) - F (P) = \hat {P} (\hat {P} ^ {\top} S \hat {P} + \bar {R}) ^ {- 1} (\hat {P} ^ {\top} - P ^ {\top}) \\ \left. + \left(\hat {P} (\hat {P} ^ {\top} S \hat {P} + \bar {R}) ^ {- 1} - P (P ^ {\top} S P + \bar {R}) ^ {- 1}\right) P ^ {\top} \right. \\ = \hat {P} (\hat {P} ^ {\top} S \hat {P} + \bar {R}) ^ {- 1} (\hat {P} ^ {\top} - P ^ {\top}) \\ \left. + \left((\hat {P} - P) (\hat {P} ^ {\top} S \hat {P} + \bar {R}) ^ {- 1} + P \left(\left(\hat {P} ^ {\top} S \hat {P} + \bar {R}\right) ^ {- 1} - \left(P ^ {\top} S P + \bar {R}\right) ^ {- 1}\right)\right) P ^ {\top} \right. \\ = \hat {P} (\hat {P} ^ {\top} S \hat {P} + \bar {R}) ^ {- 1} (\hat {P} ^ {\top} - P ^ {\top}) \\ + (\hat {P} - P) \left(\hat {P} ^ {\top} S \hat {P} + \bar {R}\right) ^ {- 1} P ^ {\top} + P \left(\left(\hat {P} ^ {\top} S \hat {P} + \bar {R}\right) ^ {- 1} - \left(P ^ {\top} S P + \bar {R}\right) ^ {- 1}\right) P ^ {\top} \tag {78} \\ \end{array}
$$

Then we have

$$
\begin{array}{l} \mathbb {E} \left[ | \hat {J} - J | \right] \leq \mathbb {E} \left[ \| F (\hat {P}) - F (P) \| _ {2} \right] \| z \| _ {2} ^ {2} \\ \leq \underbrace {\mathbb {E} \left[ \left\| \hat {P} (\hat {P} ^ {\top} S \hat {P} + \bar {R}) ^ {- 1} (\hat {P} ^ {\top} - P ^ {\top}) \right\| _ {2} \right]} _ {(i)} \| z \| _ {2} ^ {2} \\ + \underbrace {\mathbb {E} \left[ \left\| (\hat {P} - P) (\hat {P} ^ {\top} S \hat {P} + \bar {R}) ^ {- 1} P ^ {\top} \right\| _ {2} \right]} _ {(i i)} \| z \| _ {2} ^ {2} \\ + \underbrace {\mathbb {E} \left[ \left\| P \left(\left(\hat {P} ^ {\top} S \hat {P} + \bar {R}\right) ^ {- 1} - \left(P ^ {\top} S P + \bar {R}\right) ^ {- 1}\right) P ^ {\top} \right\| _ {2} \right]} _ {(i i i)} \| z \| _ {2} ^ {2} \tag {79} \\ \end{array}
$$

We bound term (i) as follows:

$$
\begin{array}{l} (i) \quad \mathbb {E} \left[ \left\| \hat {P} (\hat {P} ^ {\top} S \hat {P} + \bar {R}) ^ {- 1} (\hat {P} ^ {\top} - P ^ {\top}) \right\| _ {2} \right] \\ \leq \mathbb {E} \left[ \left\| \hat {P} \right\| _ {2} \left\| \left(\hat {P} ^ {\top} S \hat {P} + \bar {R}\right) ^ {- 1} \right\| _ {2} \left\| \hat {P} - P \right\| _ {2} \right] \\ \leq \left\| \bar {R} ^ {- 1} \right\| _ {2} \mathbb {E} \left[ \left\| \hat {P} \right\| _ {2} \left\| \hat {P} - P \right\| _ {2} \right]. \tag {80} \\ \end{array}
$$

We now use the following:

$$
\left\{ \begin{array}{l} \left\| \hat {P} \right\| _ {2} = \left\| \hat {P} - P + P \right\| _ {2} \leq \left\| \hat {P} - P \right\| _ {2} + \| P \| _ {2}, \\ \left\| \hat {P} \right\| _ {2} \left\| \hat {P} - P \right\| _ {2} \leq (\left\| \hat {P} - P \right\| _ {2} + \| P \| _ {2}) \left\| \hat {P} - P \right\| _ {2} = \left\| \hat {P} - P \right\| _ {2} ^ {2} + \| P \| _ {2} \left\| \hat {P} - P \right\| _ {2}. \end{array} \right.
$$

$$
\begin{array}{l} \leq \left\| \bar {R} ^ {- 1} \right\| _ {2} \mathbb {E} \left[ \left\| \hat {P} - P \right\| _ {2} ^ {2} + \| P \| _ {2} \left\| \hat {P} - P \right\| _ {2} \right] \\ = \left\| \bar {R} ^ {- 1} \right\| _ {2} \mathbb {E} \left[ \left\| \hat {B} - B \right\| _ {2} ^ {2} + \| P \| _ {2} \left\| \hat {B} - B \right\| _ {2} \right] \\ \leq \left\| \bar {R} ^ {- 1} \right\| _ {2} \mathbb {E} \left[ \left\| \hat {B} - B \right\| _ {F} ^ {2} + \| P \| _ {2} \left\| \hat {B} - B \right\| _ {F} \right] \\ \leq \left\| \bar {R} ^ {- 1} \right\| _ {2} \left(\operatorname{Tr} \left(K _ {B}\right) + \| P \| _ {2} \sqrt {\operatorname{Tr} \left(K _ {B}\right)}\right) \\ = \left\| \bar {R} ^ {- 1} \right\| _ {2} \left(\mathrm{Tr} (K _ {B}) + \| B \| _ {2} \sqrt {\mathrm{Tr} (K _ {B})}\right). \tag {81} \\ \end{array}
$$

We bound (ii) as follows:

$$
\begin{array}{l} (i i) \quad \mathbb {E} \left[ \left\| (\hat {P} - P) (\hat {P} ^ {\top} S \hat {P} + \bar {R}) ^ {- 1} P ^ {\top} \right\| _ {2} \right] \\ \leq \left\| \bar {R} ^ {- 1} \right\| _ {2} \mathbb {E} \left[ \left\| \hat {P} - P \right\| _ {2} \right] \| B \| _ {2} \\ = \left\| \bar {R} ^ {- 1} \right\| _ {2} \mathbb {E} \left[ \left\| \hat {B} - B \right\| _ {2} \right] \| B \| _ {2} \\ \leq \left\| \bar {R} ^ {- 1} \right\| _ {2} \mathbb {E} \left[ \left\| \hat {B} - B \right\| _ {F} \right] \| B \| _ {2} \\ \leq \left\| \bar {R} ^ {- 1} \right\| _ {2} \sqrt {\operatorname{Tr} (K _ {B})} \| B \| _ {2}. \tag {82} \\ \end{array}
$$

We bound (iii) as follows:

$$
\begin{array}{l} \mathbb {E} \left[ \left\| P \left(\left(\hat {P} ^ {\top} S \hat {P} + \bar {R}\right) ^ {- 1} - \left(P ^ {\top} S P + \bar {R}\right) ^ {- 1}\right) P ^ {\top} \right\| _ {2} \right] (3) \\ \leq \| P \| _ {2} ^ {2} \mathbb {E} \left[ \left\| \left(\hat {P} ^ {\top} S \hat {P} + \bar {R}\right) ^ {- 1} - \left(P ^ {\top} S P + \bar {R}\right) ^ {- 1} \right\| _ {2} \right] \\ \leq \| P \| _ {2} ^ {2} \mathbb {E} \left[ \left\| - \left(P ^ {\top} S P + \bar {R}\right) ^ {- 1} \left(\hat {P} ^ {\top} S \hat {P} - P ^ {\top} S P\right) \left(\hat {P} ^ {\top} S \hat {P} + \bar {R}\right) ^ {- 1} \right\| _ {2} \right] \\ \leq \| P \| _ {2} ^ {2} \left\| \bar {R} ^ {- 1} \right\| _ {2} ^ {2} \mathbb {E} \left[ \left\| \hat {P} ^ {\top} S \hat {P} - P ^ {\top} S P \right\| _ {2} \right] \\ = \| P \| _ {2} ^ {2} \left\| \bar {R} ^ {- 1} \right\| _ {2} ^ {2} \mathbb {E} \left[ \left\| \hat {P} ^ {\top} S (\hat {P} - P) + \left(\hat {P} ^ {\top} - P ^ {\top}\right) S P) \right\| _ {2} \right] \\ = \| P \| _ {2} ^ {2} \left\| \bar {R} ^ {- 1} \right\| _ {2} ^ {2} \mathbb {E} \left[ \left\| \left(\hat {P} ^ {\top} - P ^ {\top}\right) S (\hat {P} - P) + P ^ {\top} S (\hat {P} - P) + \left(\hat {P} ^ {\top} - P ^ {\top}\right) S P \right\| _ {2} \right] \\ \leq \| P \| _ {2} ^ {2} \left\| \bar {R} ^ {- 1} \right\| _ {2} ^ {2} \mathbb {E} \left[ \left\| \left(\hat {P} ^ {\top} - P ^ {\top}\right) S (\hat {P} - P) \right\| _ {2} + \left\| P ^ {\top} S (\hat {P} - P) \right\| _ {2} + \left\| \left(\hat {P} ^ {\top} - P ^ {\top}\right) S P \right\| _ {2} \right] \\ \leq \| P \| _ {2} ^ {2} \left\| \bar {R} ^ {- 1} \right\| _ {2} ^ {2} \left(\| S \| _ {2} \operatorname{Tr} \left(K _ {B}\right) + \| B \| _ {2} \| S \| _ {2} \sqrt {\operatorname{Tr} \left(K _ {B}\right)} + \| S \| _ {2} \| B \| _ {2} \sqrt {\operatorname{Tr} \left(K _ {B}\right)}\right) \\ = \| B \| _ {2} ^ {2} \left\| \bar {R} ^ {- 1} \right\| _ {2} ^ {2} \left(\| S \| _ {2} \operatorname{Tr} (K _ {B}) + 2 \| B \| _ {2} \| S \| _ {2} \sqrt {\operatorname{Tr} (K _ {B})}\right). (83) \\ \end{array}
$$

Put everything together, we have the following bound:

$$
\begin{array}{l} \mathbb {E} \left[ \left| \hat {J} - J \right| \right] \leq \| z \| _ {2} ^ {2} \left[ \left\| \bar {R} ^ {- 1} \right\| _ {2} \left(\operatorname{Tr} \left(K _ {B}\right) + \| B \| _ {2} \sqrt {\operatorname{Tr} \left(K _ {B}\right)}\right) \right. \\ + \left\| \bar {R} ^ {- 1} \right\| _ {2} \sqrt {\operatorname{Tr} (K _ {B})} \| B \| _ {2} \\ \left. + \| B \| _ {2} ^ {2} \left\| \bar {R} ^ {- 1} \right\| _ {2} ^ {2} \left(\| S \| _ {2} \operatorname{Tr} (K _ {B}) + 2 \| B \| _ {2} \| S \| _ {2} \sqrt {\operatorname{Tr} (K _ {B})}\right) \right] \\ = \| z \| _ {2} ^ {2} \left[ \left(\left\| \bar {R} ^ {- 1} \right\| _ {2} + \| B \| _ {2} ^ {2} \left\| \bar {R} ^ {- 1} \right\| _ {2} ^ {2} \| S \| _ {2}\right) \operatorname{Tr} \left(K _ {B}\right) \right. \\ \left. + \left(2 \left\| \bar {R} ^ {- 1} \right\| _ {2} \| B \| _ {2} + 2 \| B \| _ {2} ^ {3} \left\| \bar {R} ^ {- 1} \right\| _ {2} ^ {2} \| S \| _ {2}\right) \sqrt {\operatorname{Tr} (K _ {B})} \right] \\ = \| z \| _ {2} ^ {2} \| R ^ {- 1} \| _ {2} \left(1 + \| B \| _ {2} ^ {2} \| R ^ {- 1} \| _ {2} \| S \| _ {2}\right) \left(\operatorname{Tr} \left(K _ {B}\right) + 2 \| B \| _ {2} \sqrt {\operatorname{Tr} \left(K _ {B}\right)}\right). \tag {84} \\ \end{array}
$$

![](images/4eb1bd36064d32ea3b3882b8d9830d2fe50e95b4d125856cc2f1dd2bb00b606f.jpg)

Proof of Lagrange multiplier sample complexity result.

$$
\begin{array}{l} U = - R ^ {- 1} B ^ {\top} \otimes (I - G _ {\lambda}) ^ {- 1} H _ {\lambda} x _ {0} \\ = - \underbrace {\left[ \begin{array}{c c c c} R ^ {- 1} B ^ {\top} & & & \\ & \ddots & & \\ & & \ddots & \\ & & & R ^ {- 1} B ^ {\top} \end{array} \right]} _ {D _ {R B}} (I - G _ {\lambda}) ^ {- 1} H _ {\lambda} x _ {0} \\ = - D _ {R B} \left(I - G _ {\lambda}\right) ^ {- 1} H _ {\lambda} x _ {0} \\ = - \underbrace {\left[ \begin{array}{c c c c} R ^ {- 1} & & & \\ & \ddots & & \\ & & \ddots & \\ & & & R ^ {- 1} \end{array} \right]} _ {\bar {R} ^ {- 1}} \underbrace {\left[ \begin{array}{c c c c} B ^ {\top} & & & \\ & \ddots & & \\ & & \ddots & \\ & & & B ^ {\top} \end{array} \right]} _ {P ^ {\top}} (I - G _ {\lambda}) ^ {- 1} H _ {\lambda} x _ {0} \\ = - \bar {R} ^ {- 1} P ^ {\top} (I - G _ {\lambda}) ^ {- 1} H _ {\lambda} x _ {0} \tag {85} \\ \end{array}
$$

$$
\begin{array}{l} J = x _ {0} ^ {\top} H ^ {\top} \bar {Q} G U + x _ {0} ^ {\top} H ^ {\top} \bar {Q} H x _ {0} \\ = - x _ {0} ^ {\top} H ^ {\top} \bar {Q} G D _ {R B} (I - G _ {\lambda}) ^ {- 1} H _ {\lambda} x _ {0} + x _ {0} ^ {\top} H ^ {\top} \bar {Q} H x _ {0} \\ = - \underbrace {x _ {0} ^ {\top} H ^ {\top} \bar {Q} G _ {d} P \bar {R} ^ {- 1} P ^ {\top} (I - G _ {\lambda}) ^ {- 1} H _ {\lambda} x _ {0}} _ {a} + \underbrace {x _ {0} ^ {\top} H ^ {\top} \bar {Q} H x _ {0}} _ {b} \tag {86} \\ \end{array}
$$

$$
\mathbb {E} \left[ | \hat {J} - J | \right] = \mathbb {E} \left[ \| \hat {J} - J \| _ {2} \right] = \mathbb {E} \left[ \| \hat {a} - a \| _ {2} \right] \tag {87}
$$

Then

$$
\begin{array}{l} \mathbb {E} \left[ \| \hat {a} - a \| _ {2} \right] = \mathbb {E} \left[ \| x _ {0} ^ {\top} H ^ {\top} \bar {Q} G _ {d} \hat {P} \bar {R} ^ {- 1} \hat {P} ^ {\top} (I - \hat {G} _ {\lambda}) ^ {- 1} H _ {\lambda} x _ {0} - x _ {0} ^ {\top} H ^ {\top} \bar {Q} G _ {d} P \bar {R} ^ {- 1} P ^ {\top} (I - G _ {\lambda}) ^ {- 1} H _ {\lambda} x _ {0} \| _ {2} \right] \\ = \mathbb {E} \left[ \| x _ {0} ^ {\top} H ^ {\top} \bar {Q} G _ {d} \left(\hat {P} \bar {R} ^ {- 1} \hat {P} ^ {\top} (I - \hat {G} _ {\lambda}) ^ {- 1} - P \bar {R} ^ {- 1} P ^ {\top} (I - G _ {\lambda}) ^ {- 1}\right) H _ {\lambda} x _ {0} \| _ {2} \right] \\ \leq \| x _ {0} ^ {\top} H ^ {\top} \bar {Q} G _ {d} \| _ {2} \| H _ {\lambda} x _ {0} \| _ {2} \mathbb {E} \left[ \| \hat {P} \bar {R} ^ {- 1} \hat {P} ^ {\top} (I - \hat {G} _ {\lambda}) ^ {- 1} - P \bar {R} ^ {- 1} P ^ {\top} (I - G _ {\lambda}) ^ {- 1} \| _ {2} \right]. \tag {88} \\ \end{array}
$$

Then

$$
\begin{array}{l} \mathbb {E} \left[ \| \hat {P} \bar {R} ^ {- 1} \hat {P} ^ {\top} (I - \hat {G} _ {\lambda}) ^ {- 1} - P \bar {R} ^ {- 1} P ^ {\top} (I - G _ {\lambda}) ^ {- 1} \| _ {2} \right] \\ = \mathbb {E} \left[ \| (\hat {P} - P) \bar {R} ^ {- 1} \hat {P} ^ {\top} (I - \hat {G} _ {\lambda}) ^ {- 1} + P \bar {R} ^ {- 1} \left(\hat {P} ^ {\top} (I - \hat {G} _ {\lambda}) ^ {- 1} - P ^ {\top} (I - G _ {\lambda}) ^ {- 1}\right) \| _ {2} \right] \\ = \mathbb {E} \left[ \| (\hat {P} - P) \bar {R} ^ {- 1} \hat {P} ^ {\top} (I - \hat {G} _ {\lambda}) ^ {- 1} + P \bar {R} ^ {- 1} \left((\hat {P} ^ {\top} - P ^ {\top}) (I - \hat {G} _ {\lambda}) ^ {- 1} + P ^ {\top} (I - \hat {G} _ {\lambda}) ^ {- 1} - P ^ {\top} (I - G _ {\lambda}) ^ {- 1}\right) \| _ {2} \right] \\ = \mathbb {E} \left[ \| (\hat {P} - P) \bar {R} ^ {- 1} \hat {P} ^ {\top} (I - \hat {G} _ {\lambda}) ^ {- 1} + P \bar {R} ^ {- 1} (\hat {P} ^ {\top} - P ^ {\top}) (I - \hat {G} _ {\lambda}) ^ {- 1} + P \bar {R} ^ {- 1} P ^ {\top} \left((I - \hat {G} _ {\lambda}) ^ {- 1} - (I - G _ {\lambda}) ^ {- 1}\right) \| _ {2} \right] \\ \leq \underbrace {\mathbb {E} \left[ \| (\hat {P} - P) \bar {R} ^ {- 1} \hat {P} ^ {\top} (I - \hat {G} _ {\lambda}) ^ {- 1} \| _ {2} \right]} _ {(i)} + \underbrace {\mathbb {E} \left[ \| P \bar {R} ^ {- 1} (\hat {P} ^ {\top} - P ^ {\top}) (I - \hat {G} _ {\lambda}) ^ {- 1} \| _ {2} \right]} _ {(i i)} \\ + \underbrace {\mathbb {E} \left[ \| P \bar {R} ^ {- 1} P ^ {\top} \left((I - \hat {G} _ {\lambda}) ^ {- 1} - (I - G _ {\lambda}) ^ {- 1}\right) \| _ {2} \right]} _ {(i i i)} \tag {89} \\ \end{array}
$$

Let us rewrite $G_{\lambda}$ as follows:

$$
\begin{array}{l} G _ {\lambda} = L _ {d} + L _ {u} \\ = - \underbrace {\left[ \begin{array}{c c c c} Q G _ {0} & 0 & \ldots & 0 \\ Q G _ {1} & Q G _ {0} & \ldots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ Q G _ {T - 1} & Q G _ {T - 2} & \ldots & Q G _ {0} \end{array} \right]} _ {L _ {Q G}} \underbrace {\left[ \begin{array}{c c c c} B R ^ {- 1} B ^ {\top} & & & \\ & \ddots & & \\ & & \ddots & \\ & & & B R ^ {- 1} B ^ {\top} \end{array} \right]} _ {D _ {B R B}} + L _ {u} \\ = - \underbrace {\left[ \begin{array}{c c c c} Q & & & \\ & \ddots & & \\ & & \ddots & \\ & & & Q \end{array} \right]} _ {D _ {Q}} \underbrace {\left[ \begin{array}{c c c c} G _ {0} & 0 & \ldots & 0 \\ G _ {1} & G _ {0} & \ldots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ G _ {T - 1} & G _ {T - 2} & \ldots & G _ {0} \end{array} \right]} _ {L _ {G}} D _ {B R B} + L _ {u} \\ = - D _ {Q} L _ {G} D _ {B R B} + L _ {u}, \tag {90} \\ \end{array}
$$

where

$$
L _ {u} = \left[ \begin{array}{c c c c c} 0 & A ^ {\top} - D (\alpha , 1) & - D (\alpha , 2) & \dots & - D (\alpha , T - 1) \\ 0 & 0 & A ^ {\top} - D (\alpha , 1) & \dots & - D (\alpha , T - 2) \\ 0 & 0 & 0 & \dots & - D (\alpha , T - 3) \\ \vdots & \vdots & \vdots & \vdots & \ddots \\ 0 & 0 & \dots & \dots & 0 \end{array} \right] \tag {91}
$$

Then using the psd assumption $L_{QG}^{-1}(I - L_{u}) \succeq 0$ ,

$$
\begin{array}{l} \| (I - G _ {\lambda}) ^ {- 1} \| _ {2} = \| (I - L _ {u} + D _ {Q} L _ {G} D _ {B R B}) ^ {- 1} \| _ {2} \\ = \left\| \left[ D _ {Q} L _ {G} \left(\left(D _ {Q} L _ {G}\right) ^ {- 1} (I - L _ {u}) + D _ {B R B}\right) \right] ^ {- 1} \right\| _ {2} \\ \leq \left\| \left(D _ {Q} L _ {G}\right) ^ {- 1} \right\| _ {2} \| \left(\left(D _ {Q} L _ {G}\right) ^ {- 1} \left(I - L _ {u}\right) + D _ {B R B}\right) ^ {- 1} \| _ {2} \\ \leq \left\| (D _ {Q} L _ {G}) ^ {- 1} \right\| _ {2} \left\| \left((D _ {Q} L _ {G}) ^ {- 1} (I - L _ {u})\right) ^ {- 1} \right\| _ {2} \\ \leq \| (L _ {Q G}) ^ {- 1} \| _ {2} \| L _ {Q G} \| _ {2} \| (I - L _ {u}) ^ {- 1} \| _ {2}. \tag {92} \\ \end{array}
$$

We bound (i) as follows:

$$
\begin{array}{l} \mathbb {E} \left[ \| (\hat {P} - P) \bar {R} ^ {- 1} \hat {P} ^ {\top} (I - \hat {G} _ {\lambda}) ^ {- 1} \| _ {2} \right] \\ \leq \mathbb {E} \left[ \| \hat {P} - P \| _ {2} \| \bar {R} ^ {- 1} \| _ {2} \| \hat {P} ^ {\top} \| _ {2} \| (I - \hat {G} _ {\lambda}) ^ {- 1} \| _ {2} \right] \\ \leq \| \bar {R} ^ {- 1} \| _ {2} \mathbb {E} \left[ \| \hat {P} - P \| _ {2} \left(\| \hat {P} - P \| _ {2} + \| P \| _ {2}\right) \| (I - \hat {G} _ {\lambda}) ^ {- 1} \| _ {2} \right] \\ \leq \| \bar {R} ^ {- 1} \| _ {2} \mathbb {E} \left[ \| \hat {P} - P \| _ {2} \left(\| \hat {P} - P \| _ {2} + \| P \| _ {2}\right) \right] \| (L _ {Q G}) ^ {- 1} \| _ {2} \| L _ {Q G} \| _ {2} \| (I - L _ {u}) ^ {- 1} \| _ {2} \\ \leq \| \bar {R} ^ {- 1} \| _ {2} \| (L _ {Q G}) ^ {- 1} \| _ {2} \| L _ {Q G} \| _ {2} \| (I - L _ {u}) ^ {- 1} \| _ {2} \left(\mathbb {E} \left[ \| \hat {B} - B \| _ {2} ^ {2} \right] + \mathbb {E} \left[ \| \hat {B} - B \| _ {2} \right] \| B \| _ {2}\right) \\ \leq \| \bar {R} ^ {- 1} \| _ {2} \| (L _ {Q G}) ^ {- 1} \| _ {2} \| L _ {Q G} \| _ {2} \| (I - L _ {u}) ^ {- 1} \| _ {2} \left(\mathrm{Tr} (K _ {B}) + \sqrt {\mathrm{Tr} (\bar {K} _ {B})} \| B \| _ {2}\right). \tag {93} \\ \end{array}
$$

We bound (ii) as follows:

$$
\begin{array}{l} \mathbb {E} \left[ \| P \bar {R} ^ {- 1} (\hat {P} ^ {\top} - P ^ {\top}) (I - \hat {G} _ {\lambda}) ^ {- 1} \| _ {2} \right] \\ \leq \| B \| _ {2} \| \bar {R} ^ {- 1} \| _ {2} \| (L _ {Q G}) ^ {- 1} \| _ {2} \| L _ {Q G} \| _ {2} \| (I - L _ {u}) ^ {- 1} \| _ {2} \sqrt {\operatorname{Tr} (K _ {B})}. \tag {94} \\ \end{array}
$$

We Bound (iii) as follows:

$$
\begin{array}{l} \text {(iii)} \quad \mathbb {E} \left[ \| P \bar {R} ^ {- 1} P ^ {\top} \left((I - \hat {G} _ {\lambda}) ^ {- 1} - (I - G _ {\lambda}) ^ {- 1}\right) \| _ {2} \right] \\ \leq \| P \| _ {2} ^ {2} \| \bar {R} ^ {- 1} \| _ {2} \mathbb {E} \left[ \| (I - \hat {G} _ {\lambda}) ^ {- 1} - (I - G _ {\lambda}) ^ {- 1} \| _ {2} \right] \\ = \| P \| _ {2} ^ {2} \| \bar {R} ^ {- 1} \| _ {2} \mathbb {E} \left[ \| (I - \hat {G} _ {\lambda}) ^ {- 1} (\hat {G} _ {\lambda} - G _ {\lambda}) (I - G _ {\lambda}) ^ {- 1} \| _ {2} \right] \\ \leq \| P \| _ {2} ^ {2} \| \bar {R} ^ {- 1} \| _ {2} \| (L _ {Q G}) ^ {- 1} \| _ {2} ^ {2} \| L _ {Q G} \| _ {2} ^ {2} \| (I - L _ {u}) ^ {- 1} \| _ {2} ^ {2} \mathbb {E} \left[ \| \hat {G} _ {\lambda} - G _ {\lambda} \| _ {2} \right] \\ = \| P \| _ {2} ^ {2} \| \bar {R} ^ {- 1} \| _ {2} \| (L _ {Q G}) ^ {- 1} \| _ {2} ^ {2} \| L _ {Q G} \| _ {2} ^ {2} \| (I - L _ {u}) ^ {- 1} \| _ {2} ^ {2} \mathbb {E} \left[ \| L _ {Q G} (\hat {D} _ {B R B} - D _ {B R B}) \| _ {2} \right] \\ \leq \| P \| _ {2} ^ {2} \| \bar {R} ^ {- 1} \| _ {2} \| (L _ {Q G}) ^ {- 1} \| _ {2} ^ {2} \| L _ {Q G} \| _ {2} ^ {3} \| (I - L _ {u}) ^ {- 1} \| _ {2} ^ {2} \mathbb {E} \left[ \| \hat {B} R ^ {- 1} \hat {B} ^ {\top} - B R ^ {- 1} B ^ {\top} \| _ {2} \right] \\ = \| P \| _ {2} ^ {2} \| \bar {R} ^ {- 1} \| _ {2} \| (L _ {Q G}) ^ {- 1} \| _ {2} ^ {2} \| L _ {Q G} \| _ {2} ^ {3} \| (I - L _ {u}) ^ {- 1} \| _ {2} ^ {2} \mathbb {E} \left[ \| (\hat {B} - B) R ^ {- 1} (\hat {B} - B) ^ {\top} \right. \\ \left. + (\hat {B} - B) R ^ {- 1} B ^ {\top} + B R ^ {- 1} (\hat {B} - B) ^ {\top} \| _ {2} \right] \\ \leq \| B \| _ {2} ^ {2} \| \bar {R} ^ {- 1} \| _ {2} ^ {2} \| (L _ {Q G}) ^ {- 1} \| _ {2} ^ {2} \| L _ {Q G} \| _ {2} ^ {3} \| (I - L _ {u}) ^ {- 1} \| _ {2} ^ {2} \left(\mathrm{Tr} (K _ {B}) + 2 \| B \| _ {2} \sqrt {\mathrm{Tr} (K _ {B})}\right). \tag {95} \\ \end{array}
$$

Using $\|(I-L_{u})^{-1}\|_{2}=1$ and putting everything together we have

$$
\begin{array}{l} \mathbb {E} \left[ | \hat {J} - J | \right] \\ \leq \| z \| _ {2} \| H _ {\lambda} x _ {0} \| _ {2} \| R ^ {- 1} \| _ {2} \| \mathbb {L} \| _ {2} \left(1 + \| B \| _ {2} ^ {2} \| R ^ {- 1} \| _ {2} \| \mathbb {L} \| _ {2} \| L _ {Q G} \| _ {2}\right) \left[ \operatorname{Tr} (K _ {B}) + 2 \| B \| _ {2} \sqrt {\operatorname{Tr} (K _ {B})} \right], \\ \end{array}
$$

$$
\text { where } \| \mathbb {L} \| _ {2} = \| (L _ {Q G}) ^ {- 1} \| _ {2} \| L _ {Q G} \| _ {2}. \tag {96}
$$

# D.5. Proof of Corollary 3.4

Proof. We first expand $K_B$ as follows

$$
\begin{array}{l} K _ {B} = \frac {1}{N} \left(\phi^ {\top} \phi\right) ^ {- 1} \phi^ {\top} \sigma_ {w} ^ {2} I \phi \left(\phi^ {\top} \phi\right) ^ {- 1} \\ = \frac {\sigma_ {w} ^ {2}}{N} \left(\phi^ {\top} \phi\right) ^ {- 1} \\ = \frac {\sigma_ {w} ^ {2}}{N} \left[ \sum_ {i = 1} ^ {p} \phi_ {i} ^ {\top} \phi_ {i} \right] ^ {- 1}, \tag {97} \\ \end{array}
$$

where

$$
\sum_ {i = 1} ^ {p} \phi_ {i} ^ {\top} \phi_ {i} = \left[ \begin{array}{c c c} \sum_ {i = 1} ^ {p} u _ {0} ^ {i} u _ {0} ^ {i ^ {\top}} & \dots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \dots & \sum_ {i = 1} ^ {p} u _ {0} ^ {i} u _ {0} ^ {i ^ {\top}} \end{array} \right]. \tag {98}
$$

Then we have

$$
\begin{array}{l} \operatorname{Trace} \left(K _ {B}\right) = \frac {\sigma_ {w} ^ {2}}{N} \mathbb {E} \left[ \operatorname{Trace} \left(\left[ \sum_ {i = 1} ^ {p} \phi_ {i} ^ {\top} \phi_ {i} \right] ^ {- 1}\right) \right] \\ = \frac {n \sigma_ {w} ^ {2}}{N} \mathbb {E} \left[ \operatorname{Trace} \left(\left(\sum_ {i = 1} ^ {p} u _ {0} ^ {i} u _ {0} ^ {\top}\right) ^ {- 1}\right) \right] \\ = \frac {n \sigma_ {w} ^ {2}}{N} \operatorname{Trace} \left(\mathbb {E} \left[ \left(\sum_ {i = 1} ^ {p} u _ {0} ^ {i} u _ {0} ^ {\top}\right) ^ {- 1} \right]\right). \tag {99} \\ \end{array}
$$

$$
\left\{ \begin{array}{l} u _ {0} ^ {i} \sim \mathcal {N} (0, I) \implies \sum_ {i = 1} ^ {p} u _ {0} ^ {i} u _ {0} ^ {i ^ {\top}} \sim \mathcal {W} (I, p), \quad \text { Wishart   Distribution   (Wishart,   1928) }, \\ \implies \left(\sum_ {i = 1} ^ {p} u _ {0} ^ {i} u _ {0} ^ {i ^ {\top}}\right) ^ {- 1} \sim \mathcal {W} ^ {- 1} (I, p), \quad \text { Inverse   Wishart   Distribution }, \\ \implies \mathbb {E} \left[ \left(\sum_ {i = 1} ^ {p} u _ {0} ^ {i} u _ {0} ^ {i ^ {\top}}\right) ^ {- 1} \right] = \frac {I}{p - m - 1}, \quad \text { for   } p > m + 1. \end{array} \right. \tag {1}
$$

$$
\begin{array}{l} \operatorname{Trace} (K _ {B}) \stackrel {(1)} {=} \frac {n \sigma_ {w} ^ {2}}{N} \operatorname{Trace} \left(\frac {I}{p - m - 1}\right) \\ = \frac {n m \sigma_ {w} ^ {2}}{N (p - m - 1)}, \quad \text { for   } p > m + 1. \tag {100} \\ \end{array}
$$

![](images/282e09ec7ba90325dbe09d66b7538cb5b753e8d51f9906abaa58c0c67ffb969f.jpg)

# E. Experiment Details

# E.1. Model Architecture

Our proposed Fractional-Order Learning for Optimal Control Framework (FOLOC) is a deep learning framework designed for fractional-order optimal control tasks, combining the strengths of neural operater, sequential modeling, and parameter regression. The architecture is structured into four key components, enabling end-to-end learning of complex control dynamics from system states and parameters. Fig. 1 illustrates the overall pipeline.

# E.1.1. SYSTEM IDENTIFICATION

The system identification module is designed to infer fractional-order time-invariant system parameters, including the fractional-order $\alpha$ of size n and the constant matrices A and B of sizes $n \times n$ and $n \times m$ respectively, as introduced in Eq. 6. These parameters are directly inferred from the observed control input space X, which is formed by concatenating $x_{k}$ and $u_{k}$ , where $x_{k} \in R^{n}$ is the state vector and $u_{k} \in R^{m}$ is the system input. This component leverages a hybrid architecture to accommodate both static and temporally correlated systems. Specifically, we use an MLP with residual skip connections and a Sequence Model (e.g., Recurrent Neural Networks, Gated Recurrent Unit Networks, or Long Short-Term Memory Networks) as the backbone of system identification module.

For non-Markovian systems with long-range temporal temporal dependencies, sequential encoders—such as bidirectional LSTMs, GRUs, or RNNs—are adopted. These models process the input sequence autoregressively, capturing latent temporal patterns through hidden state evolution. Bidirectional processing further enables context aggregation from both past and future observations within the horizon T. The regressor outputs a flattened tensor, which is decomposed into A, B and $\alpha$ via learned linear projections, ensuring dimensional consistency with the underlying physical system. We formalize this module as follows.

For non-Markovian systems, bidirectional LSTMs model temporal dependencies:

$$
h _ {t}, c _ {t} = \text { LSTM } ([ x _ {t}; u _ {t} ], (h _ {t - 1}, c _ {t - 1})), \tag {101}
$$

where $h_{t} \in R^{2d}$ (bidirectional hidden states). The final state $h_{T}$ aggregates sequence information.

Then we apply a stack of residual MLP blocks processes final state $h_{T}$ :

$$
\mathrm{h} ^ {(0)} = h _ {T}, \mathrm{h} ^ {(l)} = \text { BatchNorm } \left(\sigma \left(W ^ {(l)} \mathrm{h} ^ {(l - 1)} + b ^ {(l)}\right) + \mathrm{h} ^ {(l - 1)}\right), \tag {102}
$$

where $\sigma$ is an activation function (e.g., ReLU) and 1, ..., L indexes the residual blocks.

The regressor outputs a flattened tensor $\theta \in \mathbb{R}^{n^2 + nm + n}$ , which can be decomposed into $A, B$ and $\alpha$

$$
\theta = W _ {o u t} \mathrm{h} ^ {(\mathrm{L})}, \tag {103}
$$

$$
A = \text { Reshape } (\theta_ {1: n ^ {2}}), B = \text { Reshape } (\theta_ {n ^ {2} + 1: n ^ {2} + n m}), \alpha = \theta_ {n ^ {2} + n m + 1: n ^ {2} + n m + n} \tag {104}
$$

The module minimizes a multi-task regression loss:

$$
\mathcal {L} _ {\mathrm{s}} = \| A - A ^ {*} \| _ {2} ^ {2} + \| B - B ^ {*} \| _ {2} ^ {2} + \| \alpha - \alpha^ {*} \| _ {2} ^ {2}. \tag {105}
$$

This module is trained under a multi-task objective, jointly optimizing the reconstruction of system parameters while preserving their interpretable structure (e.g., enforcing A as a state transition matrix). By unifying static and sequential modeling paradigms, the regressor adapts to diverse dynamical regimes, laying a foundation for downstream control synthesis.

# E.1.2. TEMPORAL EMBEDDING FOR A MATRIX EVOLUTION

In order to obtain the temporal embedding of matrix A, we propose an embedding estimation module that transforms the estimated fractional-order system parameters into time-aware representations. These representations capture both the spectral properties of the dynamics and their temporal evolution. By addressing the challenge of modeling parameter drift in partially observed systems, this approach maintains physical consistency throughout.

Our architecture implements a novel parameterization of the system matrix A, explicitly encoding hierarchical damping effects. This allows for both stability guarantees and adaptive temporal evolution. Building on Eq. 15, the module constructs time-dependent embeddings $A_{t}^{emb} \in R^{d}$ through a dual-branch architecture.

Let $A \in R^{n \times n}$ and $\alpha \in R^n$ denote the estimated constant matrix and fractional-order from the system identification module. To compute the embedding of the first token $A_0$ , we decompose the system matrix $A_\alpha = A - \text{diag}(a)$ through spectral analysis:

$$
A _ {\alpha} = V \Lambda V ^ {- 1}, \quad \Lambda = \operatorname{diag} \left(\lambda_ {1}, \dots , \lambda_ {n}\right), \tag {106}
$$

$$
h _ {s p e c} = \sigma \big (W _ {\gamma} \big [ \mathrm{Re} (\Lambda), \mathrm{Im} (\Lambda) \big ] \big), \tag {107}
$$

where $\lambda_{i} \in \mathbb{C}$ are eigenvalues encoding system stability. A residual MLP processes the concatenated real/imaginary components $[\mathrm{Re}(\Lambda), \mathrm{Im}(\Lambda)]$ to produce base embeddings $A_0$ .

In parallel, time embeddings $e_{t}$ are generated via an embedding layer that encodes sequential time indices. These are concatenated with embeddings derived from $\alpha$ , enabling the model to capture both temporal and parametric variations:

$$
h _ {\text { state }} = \phi_ {x} (e _ {t}) \oplus \phi_ {\alpha} (\alpha), \quad \phi_ {x}, \phi_ {\alpha}: \text { Linear   Projections }. \tag {108}
$$

Next, the adaptive fused embeddings pass through a series of stacked residual multi-layer perceptron (MLP) blocks with normalization and an activation function (e.g., ReLU or GELU). This produces the final embeddings that serve as inputs to the temporal processor. This residual architecture enhances the model's capacity to learn complex representations while preserving information from earlier layers:

$$
A ^ {e m b} = \text { ResMLP } \left([ h _ {s p e c} \oplus h _ {\text { state }} ]; W _ {l}, b _ {l l = 1} ^ {L}\right), A ^ {e m b} \in \mathbb {R} ^ {T \times \text { hidden\_size }} \tag {109}
$$

This embedding mechanism enables the model to generate time-dependent parameter representations without directly relying on the input state vector x. By encoding dynamic properties through eigenvalues and temporal embeddings, the model effectively captures system behavior over time, making it suitable for tasks such as time-series forecasting and control parameter prediction.

# E.1.3. SEQUENCE ENCODER PROCESSOR

The Sequence Processor translates spectral-temporal embeddings $A_{t}^{emb}$ into latent states $G_{t}^{enc}$ that encode the system's temporal evolution, enabling adaptive control under time-varying dynamics. The sequence processor computes latent states $G_{t}^{enc}$ through a recursive hierarchical structure, enabling efficient modeling of temporal dependencies while preserving stability, rooted in the governing Eq. 14.

we implements this recurrence through a deep encoder architecture, which projects the raw recursive computation into a latent space. For systems requiring global temporal context, we use Transformer as backbone:

$$
G _ {k} ^ {\text { enc }} = \text { TransformerEncoder } (A _ {k} ^ {\text { emb }} + P k), \tag {110}
$$

where $P_{k}$ is a positional encoding encoding the step k. Multi-head self-attention layers implicitly approximate the summation in Eq. 14 by learning attention weights that mirror the hierarchy $A_{j}$ .

# E.1.4. STACK MLPs FOR FOURIER NEURAL OPERATOR

The final stage of our architecture synthesizes hierarchical system representations into spatiotemporal control signals through a two-step process: feature unification via the Stack MLPs and spectral control synthesis via the Fourier Neural Operator (FNO). This combination enables efficient learning of distributed control laws while preserving physical consistency. Given temporal states $G_{k}^{enc}$ , system parameters B, LQR cost matrices Q,R, and residual skip connection input $[x;u]$ , the transformation constructs an enriched input tensor $X \in R^{T \times d}$ for the FNO:

$$
\chi_ {t} = \phi_ {G} (G _ {t}) \oplus \phi_ {B} (B) \oplus \phi_ {Q} (Q) \oplus \phi_ {R} (R) \oplus x _ {t} \oplus u _ {t} \tag {111}
$$

where $\phi$ are a series of stacked residual multi-layer perceptron (MLP) blocks and $\oplus$ denotes channel-wise concatenation. Next, The FNO processes X through spectral convolutions and residual connections as we introduced in Section 3.3:

$$
\mathcal {F} (\chi) (k) = \mathbf {W} (k) \cdot \chi (k) + \mathbf {b} (k), \quad k \in \mathbb {Z} ^ {d} \tag {112}
$$

where $\chi(k)=\mathcal{F}(\chi)$ is the Fourier transform of the input, and W, b are learnable parameters in frequency space. Only low-frequency modes $\|k\|\leq k_{max}$ are retained, enforcing spectral sparsity.

Then, we apply the inverse transformation to predict the optimal control U:

$$
U = \mathcal {F} ^ {- 1} \left(\prod_ {l = 1} ^ {L} \left((W _ {l} \mathcal {F} + K _ {l}) \mathcal {F} (\chi)\right)\right) + \mathrm{MLP} (\chi), \tag {113}
$$

where $K_{l}$ are local kernel integrations in physical space.

The module minimizes the deviations between predicted $U_{t}$ and optimal control trajectories $U_{t}^{*}$ :

$$
\mathcal {L} _ {u} = \frac {1}{T m} \sum_ {t = 1} ^ {T} \left\| U _ {t} - U _ {t} ^ {*} \right\| _ {2} ^ {2}. \tag {114}
$$

# E.1.5. COMPOSITE LOSS FORMULATION

The training objective combines fractional-order system parameter regression and control prediction errors through a multi-task loss function, enabling joint optimization of system identification and control synthesis:

$$
\mathcal {L} _ {\text { total }} = \underbrace {\mathcal {L} _ {\mathrm{s}}} _ {\text { System   ID }} + \underbrace {\mathcal {L} _ {u}} _ {\text { Control   Prediction }}. \tag {115}
$$

This unified framework successfully marries physics-guided feature engineering with data-driven spectral learning, enabling real-time optimal control in high-dimensional dynamical systems and the unified loss formulation enables end-to-end learning of control policies while maintaining physically consistent parameter estimates—a critical requirement for deployment in safety-critical dynamical systems.

remark: Specifically, the system parameters A, B and $\alpha$ can be extracted from an intermediate layer in the network because of the underlying loss function. The intermediate variable $A_{k}$ is approximated using a shallow neural network P such that $A_{k} = \mathcal{P}(A, \alpha, t)$ and the variable $G_{k}$ is reduced to $R^{n}$ instead of $R^{n \times n}$ . Similarly, the cost matrices Q and R are transformed into reduced representations via local transformation operators $N_{Q}: R^{n \times n} \to R^{d_{Q}}$ and $N_{R}: R^{n \times m} \to R^{d_{R}}$ . The block Toeplitz structure of $G_{\lambda}$ indicates that it can be fully represented using the parameters Q, R, A, B, $\alpha$ , $x_{0}$ and $\{G_{k}\}_{k=0}^{T}$ . Using this property, the input dimension of the FNO is determined as $d_{a} = T \times (2n + m + d_{Q} + d_{R} + d_{B})$ , where the extra terms n and m correspond to the trajectory and control input dimensions, respectively. These additional features empirically enhance the performance of the FNO by providing more comprehensive representations of the underlying dynamics.

# E.2. Synthetic Data

The synthetic data is generated as follows:

State Transition Matrix (A): Rondomly sampled from a uniform distribution and normalized to ensure stability by scaling with its spectral radius.

Control Input Matrix (B): Randomly sampled from a uniform distribution.

Initial State $(x_0)$ : Randomly sampled from a standard normal distribution.

Control Inputs (u): Sequence of uniformly random inputs.

Cost Matrices $(Q, R)$ : Symmetric positive semi-definite matrices constructed via $M^{\top}M$ for some uniformly random M.

# E.3. Real-World System Dynamics

In our experiments, we simulate the system dynamics to generate trajectories and optimal control sequences by randomly choosing an initial state and control inputs and then using model predictive control (MPC) to find the optimal control sequence given the quadratic cost matrices $Q$ and $R$ .

# E.3.1. CART POLE SYSTEM DYNAMICS

The dynamics of the cart pole is as follows:

$$
\ddot {x} = \frac {m _ {p} l (\dot {\theta} ^ {2} \sin \theta - \ddot {\theta} \cos \theta) + F}{m _ {c} + m _ {p}},
$$

$$
\ddot {\theta} = \frac {g \sin \theta + \cos \theta \left(\frac {- m _ {p} l \dot {\theta} ^ {2} \sin \theta - F}{m _ {c} + m _ {p}}\right)}{l \left(\frac {4}{3} - \frac {m _ {p} \cos^ {2} \theta}{m _ {c} + m _ {p}}\right)}, \tag {116}
$$

where $\ddot{x}$ is the horizontal acceleration of the cart, $\ddot{\theta}$ is the acceleration of the pole, $m_{c}$ is the mass of the cart, $m_{p}$ is the mass of the pole, and g is the acceleration due to gravity. The control input is the external force F applied to the center of mass of the cart.

# E.3.2. QUADROTOR SYSTEM DYNAMICS

The quadrotor dynamics is as follows:

$$
\dot {p} = v,
$$

$$
\dot {q} = \frac {1}{2} q \otimes \left[ \begin{array}{c} 0 \\ \omega \end{array} \right],
$$

$$
m \dot {v} = m g + f + f _ {a},
$$

$$
\mathcal {J} \dot {\omega} = - \omega \times \mathcal {J} \omega + \tau . \tag {117}
$$

Here, $p \in R^{3}$ and $v \in R^{3}$ represent the position and velocity in the inertial frame, $\omega \in R^{3}$ is the angular velocity, and the quaternion q describes the orientation with $\otimes$ denoting quaternion multiplication. The mass of the quadrotor is given by m, and J denotes the inertia matrix. Gravity is denoted by g, the aerodynamic force is denoted by $f_{a} \in R^{3}$ , the total thrust and body torques are denoted by $f \in R^{3}$ and $\tau \in R^{3}$ .

# F. Training Parameters

The experiments utilizing GPU acceleration on an NVIDIA A100 80GB PCIe GPU. The experiments are conducted on a machine running Ubuntu 22.04.5 LTS with an Intel(R) Xeon(R) Platinum 8358 CPU (2.60 GHz), featuring 128 cores and support for 256 concurrent threads.

The training process is configured to optimize model performance using the Adam optimizer with a learning rate of $10^{-3}$ . The model is trained for 300 epochs with a batch size of 128, Learning rate scheduling is managed by a ReduceLROnPlateau scheduler, which monitors validation loss and reduces the learning rate by a factor of 0.1 after 5 epochs without improvement. The scheduler uses a relative threshold of 0.0001 to detect improvement. The minimum learning rate is set to zero, and early learning stabilization is facilitated by an epsilon value of $1 \times 10^{-8}$ . The loss function based on the $L_{p}$ -norm.

To ensure robust training, data normalization is applied, with both the input data and control signals $(U)$ included in the modeling process. The architecture supports both encoder-only and sequence-to-sequence configurations; however, in this setup, the sequence-to-sequence mode is disabled.

# G. Metrics

Mean Squared Error (MSE). The Mean Squared Error is given by

$$
\mathcal {L} _ {\mathrm{MSE}} (\hat {y}, y) = \frac {1}{n} \sum_ {i = 1} ^ {n} (y _ {i} - \hat {y} _ {i}) ^ {2}, \tag {118}
$$

where $n$ denotes the number of samples, $y_{i} \in \mathbb{R}$ is the ground truth value, and $\hat{y}_i \in \mathbb{R}$ is the predicted value for the $i$ -th sample.

Mean Absolute Error (MAE). The Mean Absolute Error is given by

$$
\mathcal {L} _ {\mathrm{MAE}} (\hat {y}, y) = \frac {1}{n} \sum_ {i = 1} ^ {n} \left| y _ {i} - \hat {y} _ {i} \right|, \tag {119}
$$

where $n, y_{i}$ , and $\hat{y}_{i}$ are defined as in the MSE formulation.

# H. Ablation Study

In this section, we provide further analyses to uncover how the choice of encoder models and weight of identification loss affect the performance of the FOLOC framework.

# H.1. Varying Sequence Encoder Module

To investigate the performance of different encoder layers on fractional-order system, we choose four sequence encoder variants—Transformer, RNN, LSTM, and GRU. Our analysis of sequence encoder architectures reveals statistically significant differences in performance and stability across models. The results are given in Table 4. The Transformer encoder demonstrates superior performance, achieving the lowest mean squared error (MSE = 0.0049 ± 0.00009) with minimal variance, underscoring its robustness to initialization. These results highlight the ability of our model's self-attention mechanisms to handle tasks involving global temporal dependencies. Based on our experiments, the Transformer architecture mitigates the risk of gradient vanishing or explosion by distributing information through attention, effectively stabilizing the optimization process and demonstrating strong suitability for safety-critical control tasks.

Table 4. Ablation study of different sequence encoders. 

<table><tr><td>Model</td><td>MSE ( $10^{-3}$ )</td><td>MAE ( $10^{-2}$ )</td><td>LpLoss ( $10^{-1}$ )</td></tr><tr><td>Transformer</td><td> $4.95 \pm 0.10$ </td><td> $1.05 \pm 0.10$ </td><td> $2.21 \pm 0.02$ </td></tr><tr><td>RNN</td><td> $6.00 \pm 1.46$ </td><td> $1.15 \pm 0.17$ </td><td> $2.50 \pm 0.48$ </td></tr><tr><td>LSTM</td><td> $5.54 \pm 1.49$ </td><td> $1.12 \pm 0.18$ </td><td> $2.42 \pm 0.49$ </td></tr><tr><td>GRU</td><td> $6.78 \pm 2.36$ </td><td> $1.29 \pm 0.34$ </td><td> $2.84 \pm 0.88$ </td></tr></table>

Table 5. Varying identification weight. All reported values are in units of ${10}^{-3}$ . 

<table><tr><td>Metrics</td><td>MSE/MAE</td><td>MSE/MAE</td><td>MSE/MAE</td><td>MSE/MAE</td><td>MSE/MAE</td></tr><tr><td>Identification</td><td> $\lambda_w = 0.1$ </td><td> $\lambda_w = 0.2$ </td><td> $\lambda_w = 0.3$ </td><td> $\lambda_w = 0.4$ </td><td> $\lambda_w = 0.5$ </td></tr><tr><td>Weight</td><td> $4.95±0.10/10.5±0.10$ </td><td> $4.89±0.14/10.3±0.07$ </td><td> $5.19±0.65/11.0±0.83$ </td><td> $5.84±1.36/11.4±1.69$ </td><td> $5.08±0.29/11.1±0.54$ </td></tr></table>

# H.2. Varying Identification Weight

We evaluate the performance of the FOLOC framework by varying the system identification weight $\lambda_w$ over the values $\{0.1, 0.2, 0.3, 0.4, 0.5\}$ . Experimental results show that FOLOC's performance remains approximately the same, with a mean MSE of $0.52\%$ (standard deviation: $0.04\%$ ) and a mean MAE of $1.09\%$ (standard deviation: $1.11\%$ ). The results are given in Table 5.

# I. Baseline

To the best of our knowledge, no existing end-to-end deep learning framework addresses the optimal control problem under fractional-order dynamical systems, nor is there a theoretical framework that integrates end-to-end learning with FOLTI system identification and optimal control laws. Due to the principle that fractional-order calculus generalizes integer-order calculus, we did some experiments to compare our proposed methods with the traditional theoretical end-to-end approach for optimal control in LTI systems. The end-to-end learning framework for LTI systems involves system identification using ordinary least squares, followed by solving the optimal control problem through dynamic programming (Anderson & Moore, 2007).

# J. Additional Results

# J.1. Test Loss of Different Time Horizons

We present the MSE test loss across different time horizons as shown in Fig. 5, reveals critical insights into its temporal generalization capabilities. With performance steadily improving as the horizon extends, it reflects the FOLOC framework capacity to suppress high-frequency errors through spectral filtering while gradually resolving low-frequency dynamics. T = 128 not only shows a smooth reduction in MSE but also achieves the lowest MSE loss among all tested horizons, indicating that the model better captures long-range temporal dependencies when provided with more context. The finding that extending the prediction window enhances both model robustness and accuracy confirms that, when given sufficient historical information, our approach is better equipped to handle complex dynamics of fractional-order systems.

# J.2. Results of Different Fractional-Order Systems

To evaluate the model's robustness across diverse fractional-order systems, we tested FOLOC framework on a dataset constructed from 10,000 unique systems with varying parameters. Our approach demonstrates consistent performance on this heterogeneous evaluation set, achieving a mean squared error (MSE) of $(8.0025 \pm 1.1420) \times 10^{-3}$ and mean absolute error (MAE) of $(9.4546 \pm 0.7698) \times 10^{-3}$ , indicating stable generalization despite substantial system diversity. These results reflect the architecture's capacity to adapt to different fractional-order systems without task-specific fine-tuning. Such findings align with theoretical expectations, as the spectral-temporal embeddings and the FOLOC framework inherently decouple system-specific features from control synthesis, enabling robust out-of-distribution operation.

![](images/6712c5ae8c4bcdcceb9a6779b53740525c4847727b7b48c154d95d6dccfa573a.jpg)

<details>
<summary>line</summary>

| Epoch | T=8     | T=16    | T=32    | T=64    | T=128   |
|-------|---------|---------|---------|---------|---------|
| 0     | ~0.3    | ~0.1    | ~0.05   | ~0.03   | ~0.01   |
| 25    | ~0.1    | ~0.05   | ~0.02   | ~0.01   | ~0.005  |
| 50    | ~0.1    | ~0.05   | ~0.02   | ~0.01   | ~0.005  |
| 75    | ~0.1    | ~0.05   | ~0.02   | ~0.01   | ~0.005  |
| 100   | ~0.1    | ~0.05   | ~0.02   | ~0.01   | ~0.005  |
| 125   | ~0.1    | ~0.05   | ~0.02   | ~0.01   | ~0.005  |
| 150   | ~0.1    | ~0.05   | ~0.02   | ~0.01   | ~0.005  |
| 175   | ~0.1    | ~0.05   | ~0.02   | ~0.01   | ~0.005  |
| 200   | ~0.1    | ~0.05   | ~0.02   | ~0.01   | ~0.005  |
</details>

Figure 5. Test MSE loss under each epochs.

# J.3. Comparison of Inference Time

The experimental results demonstrate that FOLOC framework achieves superior computational efficiency compares to LTI model. Specifically, FOLOC framework requires only 1.215 ms per sample for inference, which represents a significant $44.6\%$ reduction in processing time compared to LTI model's 2.1935 ms per sample.

Table 6. Model inference time comparison. 

<table><tr><td>Model</td><td>Inference time (ms/sample)</td></tr><tr><td>LTI Model</td><td>2.1935</td></tr><tr><td>FOLOC</td><td>1.215</td></tr></table>