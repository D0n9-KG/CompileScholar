# Interpolating Neural Network-Tensor Decomposition (INN-TD): a scalable and interpretable approach for large-scale physics-based problems

Jiachen Guo $^{1}$ Xiaoyu Xie $^{2}$ Chanwook Park $^{2}$ Hantao Zhang $^{1}$ Matthew J. Politis $^{1}$ Gino Domel $^{2}$ Thomas J.R. Hughes $^{3}$ Wing Kam Liu $^{24}$

# Abstract

Deep learning has been extensively employed as a powerful function approximator for modeling physics-based problems described by partial differential equations (PDEs). Despite its popularity, standard deep learning models often demand prohibitively large computational resources and yield limited accuracy when scaling to large-scale, high-dimensional physical problems. Their black-box nature further hinders the application in industrial problems where interpretability and high precision are critical. To overcome these challenges, this paper introduces Interpolating Neural Network-Tensor Decomposition (INN-TD), a scalable and interpretable framework that has the merits of both machine learning and finite element methods for modeling large-scale physical systems. By integrating locally supported interpolation functions from the finite element into the network architecture, INN-TD achieves a sparse learning structure with enhanced accuracy, faster training/solving speed, and reduced memory footprint. This makes it particularly effective for tackling large-scale high-dimensional parametric PDEs in training, solving, and inverse optimization tasks in physical problems where high precision is required.

# 1. Introduction

Deep learning methods have become a popular approach to model physical problems in recent years. Once trained, these models can make predictions much faster than stan-$^{1}$ Theoretical and applied mechanics program, Northwestern University, 2145 Sheridan Rd, Evanston, Illinois, USA $^{2}$ Department of mechanical engineering, Northwestern University, 2145 Sheridan Rd, Evanston, Illinois, USA $^{3}$ Institute for Computational Engineering and Sciences, The University of Texas at Austin, Austin, Texas, USA $^{4}$ HIDENN-AI, Evanston, Illinois, USA. Correspondence to: Wing Kam Liu <w-liu@northwestern.edu>.

Copyright 2025 by the author(s).

dard numerical simulations. However, deep learning faces challenges in modeling physical problems such as lower accuracy, instability, and the need for a huge amount of high-fidelity training data. As a result, its applications remain limited for many engineering fields that require high precision, such as aerospace, semiconductor manufacturing, and earthquake engineering. In this paper, we propose Interpolating Neural Network-Tensor Decomposition (INN-TD), which leverages the merits of both machine learning and numerical analysis to tackle large-scale, high-dimensional physical problems governed by partial differential equations (PDEs).

# 2. Related work

Deep learning theory has been applied to solve physical problems that are challenging for standard numerical algorithms. Based on the objective of the task, the current works can be broadly classified into 3 major categories: data-driven trainers, data-free solvers, and inverse problems.

# 2.1. Data-driven trainers

In the data-driven regime, we aim to leverage the universal approximation capabilities of neural networks to approximate PDE solutions. Multilayer perceptrons (MLP) have been used due to their universal approximation capability (Cybenko, 1989). Depending on the shape of the domain, convolutional neural networks and graph neural networks can be used as additional inductive biases for data on regular grids and irregular geometries, respectively (Gao et al., 2021; Pfaff et al., 2020). In recent years, many operator learning methods have been proposed to learn functional mappings instead of functions (Li et al., 2020; Lu et al., 2021; Huang et al., 2023). However, due to the black-box nature, designing optimal model architectures and optimization hyperparameters is still largely heuristic and requires trial and error (Rathore et al., 2024; Colbrook et al., 2022). For problems where physics is fully known, it becomes cumbersome to first generate offline data using standard numerical solvers and then train surrogate models from the offline data. Data generation and storage for large-scale physical problems can be extremely expensive. For these

problems, a better approach is to directly solve PDEs using deep learning models.

# 2.2. Data-free solvers

For cases where no data is available, deep learning can be used as function approximators to approximate PDE solutions. For example, MLP has been vastly used in physics-informed neural networks (PINNs) and its variations to approximate solutions to PDEs (Raissi et al., 2019; Zhang et al., 2022). Despite their vast usage, however, unlike classical numerical solvers which have a solid theoretical foundation, most of the current deep learning models are based on heuristics and lack theoretical analysis (Wang et al., 2022). It has been shown that PINN results have often been compared to weak baselines, potentially overestimating their capabilities (McGreivy & Hakim, 2024). Most data-free deep learning solvers also suffer from the curse of dimensionality (Krishnapriyan et al., 2021) and are relatively slow compared to standard numerical solvers such as finite difference/volume/element methods (Grossmann et al., 2024).

# 2.3. Finite element interpolation

Finite element method (FEM) is the computational workhorse to numerically solve differential equations arising in physics-based simulation (Liu et al., 2022). FEM is based on finite element (FE) interpolation theory. FEM first decomposes the original computational domain into a finite number of elements and then leverages locally-supported FE interpolation functions to approximate the field variables within each local domain. FEM can solve a wide range of problems, including structural mechanics, heat transfer, fluid flow, weather forecasting, and, in general, every conceivable problem that can be described by PDEs. Recent advancements in finite element (FE) methods focus on utilizing machine learning to enhance traditional FE interpolation theory. By employing Convolutional-Hierarchical Deep Neural Network (C-HiDeNN), a generalized FE interpolation framework has been proposed, offering improved flexibility and greater accuracy (Lu et al., 2023; Li et al., 2023; Park et al., 2023). Despite their robustness and widespread application, finite element (FE)-type methods have been primarily applied to address space/space-time problems, exhibiting limitations in handling parametric partial differential equations (PDEs). Furthermore, FE methods are particularly susceptible to the curse of dimensionality, which poses significant computational challenges and escalates costs when applied to large-scale, high-dimensional problems (Bungartz & Griebel, 2004).

# 2.4. Tensor decomposition

Tensor decomposition (TD) has been widely used in data compression tasks (Sidiropoulos et al., 2017). Among different tensor decomposition schemes, the Canonical Polyadic (CP) decomposition has been very popular since it decomposes higher-order tensors into a tensor product of 1D vectors (Kolda & Bader, 2009). CP decomposition has also been used vastly to solve PDEs (Beylkin & Mohlenkamp, 2005; Bachmayr, 2023; Dolgov et al., 2012; Cohen et al., 2010). From now on, we refer to CP decomposition as TD unless stated otherwise. TD has also been used in regression and classification tasks as an efficient model due to a lesser number of parameters (Ahmed et al., 2020; Park et al., 2024). Recently, it has been proved that if MLP is used to approximate each mode in TD, then it's also a universal approximation (Vemuri et al., 2025).

# 3. INN-TD formulation

Interpolating Neural Network-Tensor Decomposition (INN-TD) achieves its remarkable efficiency through two key components: the utilization of C-HiDeNN, which is a general locally supported interpolation theory based on finite element, and the integration of tensor decomposition techniques. These elements collectively enhance the framework's capability to model complex physical systems with improved accuracy and computational efficiency.

# 3.1. C-HiDeNN interpolation

Assume we want to learn a univariate function $f(x)$ given data $(x_{i}, y_{i})$ . In INN-TD, the target function $f(x)$ is first broken into small elements, and then data within each element is approximated locally. This concept has been widely used in numerical methods such as finite element (FE). Here, we use a generalized version of FEM shape functions called Convolutional-Hierarchical Deep Neural Network (C-HiDeNN) (Lu et al., 2023) as the basis for the local interpolation function. For example, to approximate a simple sinusoidal function over multiple periods as shown in Fig. 1, we start by dividing the original domain into several elements, which we refer to as a mesh. The boundaries of each element are referred to as nodes, which are highlighted as green dots in Fig. 1. For each segment of the mesh, we employ local interpolation functions to perform the approximation locally. The final approximation of the entire function is achieved by linking these locally approximated segments together.

Assuming approximating the original function using n elements, the univariate function $f(x)$ is approximated using C-HiDeNN interpolation functions in INN-TD, and can be written as:

![](images/fd4ae1a93cecb5564c8a6985f2c0978d64d623f5ce71826d7831b0b69923a0d5.jpg)

<details>
<summary>line</summary>

| x    | Original function | INN-TD approximation |
| ---- | ----------------- | -------------------- |
| 0.0  | 0.0               | 0.0                  |
| 0.1  | 1.0               | 1.0                  |
| 0.2  | 0.0               | 0.0                  |
| 0.3  | -1.0              | -1.0                 |
| 0.4  | 0.0               | 0.0                  |
| 0.5  | 1.0               | 1.0                  |
| 0.6  | 0.0               | 0.0                  |
| 0.7  | -1.0              | -1.0                 |
| 0.8  | 0.0               | 0.0                  |
| 0.9  | 1.0               | 1.0                  |
| 1.0  | 0.0               | 0.0                  |
</details>

Figure 1. C-HiDeNN approximation based on locally supported basis functions

$$
f (x) \approx \widetilde {\boldsymbol {N}} (x; s, a, p) \boldsymbol {u} _ {x} \tag {1}
$$

where $u_{x} \in R^{n+1}$ is defined as the nodal values (grid point values); $\widetilde{\mathbf{N}}(x; s, a, p)$ is the C-HiDeNN interpolation basis functions for different n elements as shown in Eq. 2. The approximation is governed by three hyperparameters, which allow for the approximation of polynomial orders of any degree: patch size s, reproducing order p, and dilation parameter a. Details regarding the role of each hyperparameter are provided in the Appendix. A.

$$
\widetilde {N} (x; s, a, p) = [ \widetilde {N} _ {0} (x; s, a, p),..., \widetilde {N} _ {n} (x; s, a, p) ] \tag {2}
$$

C-HiDeNN interpolation function maintains all the essential finite element approximation properties such as Kronecker delta (Hughes, 2003), partition of unity (Melenk & Babuška, 1996), error estimations and convergence theorems (Lu et al., 2023). For example, $\widetilde{N}(x)$ satisfies the following Kronecker delta property at nodal position $x_{l}$ :

$$
\widetilde {N} _ {k} (x _ {l}; s, a, p) = \delta_ {k l} \tag {3}
$$

where the Kronecker delta is defined as:

$$
\delta_ {k l} = \left\{ \begin{array}{l l} 0 & \text { if } k \neq l, \\ 1 & \text { if } k = l. \end{array} \right. \tag {4}
$$

At the Dirichlet boundary node $(x_{l})$ , we have:

$$
\sum_ {k} \widetilde {N} _ {k} (x _ {l}; a, s, p) u _ {k} = u _ {l} \tag {5}
$$

Thus, unlike many data-free solvers where an additional penalty term has to be added to softly enforce the Dirichlet boundary condition, C-HiDeNN can easily enforce the Dirichlet boundary condition.

In summary, C-HiDeNN basis function has the following advantages over MLP, especially when applied to physical problems: 1. The Dirichlet boundary condition is automatically satisfied. 2. C-HiDeNN is interpretable as it uses locally supported basis functions and the number of elements in C-HiDeNN is closely related to the resolution. 3. C-HiDeNN facilitates numerical integration through the use of Gaussian quadrature (Hughes, 2003), making it straightforward and efficient for many classical integral methods in solving PDEs. For simplicity of the notation, hyperparameters s, a, p are dropped from now on.

# 3.2. INN-TD approximation

A general multivariate function $u(x_{1}, x_{2}, \ldots, x_{D})$ can be approximated using functional tensor decomposition (TD) as $^{1}$

$$
u (x _ {1}, x _ {2},..., x _ {D}) \approx \mathcal {J} u (\boldsymbol {x}) = \sum_ {m = 1} ^ {M} \Pi_ {d = 1} ^ {D} f _ {d} ^ {(m)} (x _ {d}) \tag {6}
$$

where M is defined as the total number of modes in TD; D is the total number of input dimensions; $\boldsymbol{x} = (x_{1}, x_{2}, \ldots, x_{D})$ ; $f_{d}^{(m)}(x_{d})$ is the univariate function for d-th dimension and m-th mode. It has been proved that when the mode number M is sufficiently large and each univariate function $f_{d}^{(m)}(x_{d})$ serves as a universal approximation, Eq. 6 becomes a universal approximation for any multivariate function (Vemuri et al., 2025).

In INN-TD, we utilize the C-HiDeNN interpolation function for each univariate function $f_{d}^{(m)}(x_{d})$ . As a result, Eq. 6 can be written as:

$$
\mathcal {J} u (\boldsymbol {x}) = \sum_ {m = 1} ^ {M} \widetilde {\boldsymbol {N}} _ {1} (x _ {1}) \boldsymbol {u} _ {x _ {1}} ^ {(m)} \cdot \widetilde {\boldsymbol {N}} _ {2} (x _ {2}) \boldsymbol {u} _ {x _ {2}} ^ {(m)} \dots \widetilde {\boldsymbol {N}} _ {D} (x _ {D}) \boldsymbol {u} _ {x _ {D}} ^ {(m)} \tag {7}
$$

The model parameters for each mode in the INN-TD framework are $\boldsymbol{U}^{(m)} = [\boldsymbol{u}_{1}^{(m)}, \boldsymbol{u}_{2}^{(m)}, ..., \boldsymbol{u}_{D}^{(m)}]$ . Since we have M modes in the INN-TD interpolation, the complete model parameters are represented as $\boldsymbol{U} = [\boldsymbol{U}^{(1)}, \boldsymbol{U}^{(2)}, ..., \boldsymbol{U}^{(M)}]$ . The overall structure of INN-TD is shown in Fig. 9 in the Appendix.

# 3.3. Interpretability of INN-TD

As stated in (Doshi-Velez & Kim, 2017), model interpretability refers to the ability to explain a model's behavior in a way that is understandable to humans. INN-TD is an interpretable machine learning algorithm designed for sci-

entific and engineering problems, offering transparency in its structure, clear convergence guarantees, and explainable insight into how data or parameter changes affect outcomes. In addressing physics-based problems described using PDEs, we harness several inductive biases inherent in both physics and classical numerical methods. Many physical problems tend to be low-dimensional in nature. For instance, in structural engineering, it is often observed that only the initial few frequencies significantly influence a structure's dynamic response. As such, this response can be effectively approximated using a limited number of modes. More broadly, proper orthogonal decomposition (POD) or Karhunen-Loève expansions are widely employed to simplify the complexity inherent in original physical models and have been widely used in reduced-order modeling of structural analysis and fluid dynamics problems (Rathinam & Petzold, 2003). Consequently, we utilize INN-TD as an efficient tool to uncover the naturally low-dimensional characteristics of physical problems.

Additionally, INN-TD incorporates the concept of locally supported interpolation functions from the finite element method as approximators. Specifically, we employ the C-HiDeNN interpolation function, which balances the robust capabilities of generalized finite element methods with the adaptability of machine learning approaches. As a result, INN-TD adeptly accounts for the Kronecker Delta and partition of unity properties, providing flexibility, locality, stability, and control over the interpolation process. Detailed definitions of these desired properties are explained in Appendix A. As a result, INN-TD is very efficient and accurate in terms of capturing complex physics for both data-driven training and data-free solving tasks.

Finally, unlike many black-box deep learning methods, INN-TD offers a clear interpretation. Specifically, the number of elements $n_d$ in dimension $d$ determines the resolution of the interpolation, with larger $n_d$ corresponding to a higher-resolution interpolation in dimension $d$ . For cases where $D = 2$ , the total number of modes $M$ is closely linked to the number of singular values in singular value decomposition (SVD) (Kolda & Bader, 2009). The benefits of such interpretability when solving PDEs are demonstrated in detail through a numerical example in Fig. 2. In this example, INN-TD is leveraged to solve the 2D Poisson's equation with the local source term. By controlling mesh density and hyperparameters $s$ and $p$ , INN-TD can accurately predict the solution field. Details of this example can be found in Appendix B.

Moreover, due to its interpretable nature, INN-TD exhibits convergence properties as a data-free solver. Specifically, increasing the number of model parameters tends to improve accuracy. This is corroborated using multiple numerical examples in Section 4.2. The error bound is also theoretically shown in (Guo et al., 2025).

![](images/d9551e7b37d44ee920a5bacc8e9cc29cb3ecf43e5994980fd49ab91edf62e56a.jpg)

<details>
<summary>scatter</summary>

| Mesh Type     | x    | y    |
| ------------- | ---- | ---- |
| coarse mesh   | 60   | 75   |
| fine mesh     | 20   | 25   |
| Gaussian      | -    | -    |
</details>

(a)   
![](images/d76f3aecb7d20ade97cb3116994a350b065dda55737961560b0db198cb9e3004.jpg)

<details>
<summary>heatmap</summary>

| x1  | x2  | Absolute error |
| --- | --- | -------------- |
| 20  | 0   | 8.1e-6         |
| 20  | 20  | 7.2e-6         |
| 20  | 40  | 6.3e-6         |
| 20  | 60  | 5.4e-6         |
| 20  | 80  | 4.5e-6         |
| 20  | 100 | 3.6e-6         |
| 40  | 0   | 8.1e-6         |
| 40  | 20  | 7.2e-6         |
| 40  | 40  | 6.3e-6         |
| 40  | 60  | 5.4e-6         |
| 40  | 80  | 4.5e-6         |
| 40  | 100 | 3.6e-6         |
| 60  | 0   | 8.1e-6         |
| 60  | 20  | 7.2e-6         |
| 60  | 40  | 6.3e-6         |
| 60  | 60  | 5.4e-6         |
| 60  | 80  | 4.5e-6         |
| 60  | 100 | 3.6e-6         |
| 80  | 0   | 8.1e-6         |
| 80  | 20  | 7.2e-6         |
| 80  | 40  | 6.3e-6         |
| 80  | 60  | 5.4e-6         |
| 80  | 80  | 4.5e-6         |
| 80  | 100 | 3.6e-6         |
| 100 | 0   | 8.1e-6         |
| 100 | 20  | 7.2e-6         |
| 100 | 40  | 6.3e-6         |
| 100 | 60  | 5.4e-6         |
| 100 | 80  | 4.5e-6         |
| 100 | 100 | 3.6e-6         |
</details>

(b)   
Figure 2. Interpretable design of INN-TD: (a) the 1D meshes used in INN-TD are only refined at the location where nonlinearity happens with larger s and p used to achieve higher-order smoothness (b) point-wise absolute error between INN-TD and the exact solution

# 3.4. Data-driven training

For training tasks, our goal is to estimate the function $y \approx \mathcal{J}u(\boldsymbol{x})$ given the data pairs $(\boldsymbol{x}_{i}, y_{i})$ . In this section, we focus solely on regression tasks, as our primary interest lies in learning function mapping to approximate physical relationships $^{2}$ . The model parameters U can be obtained via a standard data-driven framework. We utilize the mean squared error (MSE) as the loss function to measure the difference between predicted values and true labels:

$$
\mathcal {L} = \frac {1}{K} \sum_ {k} (\mathcal {J} u (\boldsymbol {x} _ {k} ^ {*}; \boldsymbol {U}) - y _ {k} ^ {*}) ^ {2}. \tag {8}
$$

where k is the index of training data; K is the total number of training data; $(x^{*}, y^{*})$ is the training data pairs.

We propose two training schemes for determining model

parameters. The first scheme utilizes a boosting algorithm, treating each mode as a weak learner. We iteratively introduce new weak learner $\mathcal{J}^{(M)}u(\boldsymbol{x}_{\boldsymbol{k}}^{*};\boldsymbol{U})$ until the loss meets the predefined criteria:

$$
\begin{array}{l} \underset {\boldsymbol {U} ^ {(M)}} {\operatorname{argmin}} \mathcal {L} = \underset {\boldsymbol {U} ^ {(M)}} {\operatorname{argmin}} \frac {1}{K} \sum_ {k} (\Sigma_ {m = 1} ^ {M - 1} \mathcal {J} ^ {(m)} u (\boldsymbol {x} _ {k} ^ {*}; \boldsymbol {U} ^ {(m)}) + \\ \mathcal {J} ^ {(M)} u (\boldsymbol {x} _ {k} ^ {*}; \boldsymbol {U} ^ {(M)}) - y _ {k} ^ {*}) ^ {2}. \tag {9} \\ \end{array}
$$

In the second scheme, known as all-at-once optimization, model parameters for all modes are determined simultaneously.

$$
\underset {\boldsymbol {U}} {\operatorname{argmin}} \mathcal {L} = \underset {\boldsymbol {U}} {\operatorname{argmin}} \frac {1}{K} \sum_ {k} \left(\mathcal {J} u \left(\boldsymbol {x} _ {k} ^ {*}; \boldsymbol {U}\right) - y _ {k} ^ {*}\right) ^ {2}. \tag {10}
$$

Details about the boosting and all-at-once training algorithms can be found in Appendix C.

# 3.5. Data-free solving

Contrary to data-driven training tasks, no data is available to train the model parameters for data-free solving. However, we know the underlying physics is governed by the corresponding PDE. This knowledge can be leveraged to solve the model parameters in an a priori sense so that the PDE and its initial and boundary conditions are satisfied. As a result, data-free solving tasks are significantly more challenging than data-driven training tasks. $^{3}$

In the solving task, we categorize the independent variables x into three distinct types based on their physical significance. These are: spatial variables $x_{s}$ , which describe geometric features; the temporal variable $x_{t}$ ; and parametric variables $x_{p}$ , which may include PDE coefficients, forcing terms, as well as initial and boundary conditions. Thus, a general PDE for physical problems can be represented as a space-parameter-time (S-P-T) problem.

$$
\mathcal {L} (\mathcal {J} u (\boldsymbol {x} _ {s}, \boldsymbol {x} _ {p}, x _ {t})) = 0, \tag {11}
$$

where $\mathcal{L}(\cdot)$ denotes a general (nonlinear) differential operator; $\mathcal{J}u(\boldsymbol{x}_{s},\boldsymbol{x}_{p},x_{t})$ is the approximated solution, commonly referred to as the trial function in numerical analysis.

Methods for solving partial differential equations (PDEs) generally fall into two categories. The first involves directly minimizing the residual of the PDE in its differential form. The second projects the PDE loss onto a subspace, evaluating the loss via a weighted-sum residual in the integral form. Similar to FEM, INN-TD employs the integral form, which benefits from the efficiency of Gaussian quadrature-based numerical integration:

$$
\int \delta \mathcal {J} u (\boldsymbol {x} _ {s}, \boldsymbol {x} _ {p}, x _ {t}) \mathcal {L} (\mathcal {J} u (\boldsymbol {x} _ {s}, \boldsymbol {x} _ {p}, x _ {t})) d \boldsymbol {x} _ {s} d \boldsymbol {x} _ {p} d x _ {t} = 0 \tag {12}
$$

where $\delta\mathcal{J}u(\boldsymbol{x}_{s},\boldsymbol{x}_{p},x_{t})$ is called as the test function (Hughes, 2003). Different test functions can be adopted depending on the formulation. In this paper, we use the Galerkin formulation where the same function space is used for trial and test functions (Hughes, 2003). Additionally, we leverage integration by parts in Eq. 12 to relax the requirement for function continuity in INN-TD approximation (Guo et al., 2024).

In solving high-dimensional S-P-T partial differential equations (PDEs), a major challenge is the exponential increase in computational cost as the number of dimensions rises. This phenomenon, known as the curse of dimensionality, has been one of the most critical bottlenecks for both standard numerical and deep learning-based solvers. To address this challenge, only one-dimensional integration is performed in Eq. 12 due to the tensor decomposition structure of the INN-TD approximation. This approach ensures that the number of unknowns in the resulting system of equations depends only on the grid points across each dimension. Consequently, INN-TD effectively circumvents the curse of dimensionality, allowing it to handle high-dimensional problems using very fine grids and achieving high resolution. This capability is especially advantageous for modeling physical phenomena that demand an extremely fine mesh to accurately capture and resolve localized features.

In this paper, Eq. 12 is solved using a boosting algorithm. Moreover, instead of relying on stochastic optimization schemes, we use subspace iteration to linearize Eq. 12, taking advantage of the sparsity in the resulting system of equations, and solve it using sparse direct solvers. Details about the data-free solver algorithm can be found in Appendix D.

# 3.6. Inverse optimization

In many engineering applications, determining the optimal coefficients or design parameters from experimental data or underlying physics is essential. In such cases, INN-TD serves as an efficient parametric mapping tool from the input space to the output space, thanks to its sparse structure.

There are, in general, two different ways to tackle this problem. In the first case, the spatial, temporal, and parametric variables are all treated as inputs to the machine learning model. Once the model is trained/solved, one can easily leverage automatic differentiation to infer the parametric

inputs (Takamoto et al., 2022). This approach requires the model to accurately learn the parametric mapping. In the second case, the physical parameters become trainable neural network parameters. While they don't appear directly in the network, they influence the loss function through an additional residual (Zhang et al., 2022). The performance of this approach can be sensitive to the contribution of each loss term and may require adaptive scaling of the loss function components (Berardi et al., 2025).

In this paper, we adopt the first approach for INN-TD to solve inverse problems, as INN-TD can efficiently and accurately handle general parametric function mapping. The optimal parameters can be obtained by minimizing the following loss function.

$$
\underset {\boldsymbol {x} _ {p}} {\operatorname{argmin}} \mathcal {L} = \underset {\boldsymbol {x} _ {p}} {\operatorname{argmin}} \| \mathcal {J} u (\boldsymbol {x} _ {s}, x _ {t}; \boldsymbol {x} _ {p}) - u ^ {*} (\boldsymbol {x} _ {s}, x _ {t}) \| _ {\ell_ {2}} \tag {13}
$$

where $\|\cdot\|_{\ell_{2}}$ is the L2 norm and is defined as $\|x\|_{\ell_{2}} = \sqrt{\sum_{i=1}^{n}|x_{i}|^{2}}$ ; $u^{*}(x_{s}, x_{t})$ is the target spatial-temporal field.

# 4. Results

In this section, we present the performance of INN-TD in three key areas: data-driven training, data-free solving, and inverse optimization. We compare its performance with other deep learning models including vanilla MLP, SIREN (Sitzmann et al., 2020), Kolmogorov-Arnold networks (KAN) (Liu et al., 2024), and tensor-decomposition physics-informed neural networks (CP-PINN) (Vemuri et al., 2025). All of the cases were run on a single NVIDIA RTX A6000 GPU.

# 4.1. Data-driven training task

# 4.1.1. TRAINING ON PARAMETRIC PDE SOLUTIONS

In this example, we aim to learn the solution to a time-dependent parametric PDE that models heat transfer inside the domain. The input parameters are spatial location $(x,y)$ , time t, heat conductivity k, and source power P. The output is the temperature u. As a result, the function mapping of the parametric PDE can be written as $(x,y,k,P,t)\rightarrow u$ .

The governing parametric PDE can be written as:

$$
\left\{ \begin{array}{l l} \dot {u} (\boldsymbol {x}, t) + k \Delta u (\boldsymbol {x}, t) = b (\boldsymbol {x}, P) & \text { in } \quad \Omega_ {\boldsymbol {x}} \otimes \Omega_ {t}, \\ u (\boldsymbol {x}, t) | _ {\partial \Omega} = 0, \\ u (\boldsymbol {x}, 0) = 0, \end{array} \right. \tag {14}
$$

defined in a spatial domain $\Omega_{x} = [0, 1]^{2}$ , a temporal domain $\Omega_{t} = (0, 0.04]$ , and parametric domains $\Omega_{k} = [1, 4]$ and $\Omega_{P} = [100, 200]$ .

We use the Gaussian source function to model the right-hand side (RHS) source term $b(\pmb{x}, P)$ :

$$
b (\boldsymbol {x}, P) = \Sigma_ {i = 1} ^ {n _ {s}} P \exp \left(- \frac {2 ((x - x _ {i}) ^ {2} + (y - y _ {i}) ^ {2})}{r _ {0} ^ {2}}\right) \tag {15}
$$

where $r_{0}=0.05$ is the standard deviation that characterizes the width of the source function and $(x_{i},y_{i})$ is the i-th source center. We used $n_{s}=16$ in this example as shown in Fig. 3.

![](images/40a16eb0f790b17dfb434e96d104a000643704a5dcc9c517936371d9bb30a1d2.jpg)

<details>
<summary>heatmap</summary>

| Row | Column | Temperature |
|-----|--------|-----------|
| 1   | 1      | 2.2e+00   |
| 1   | 2      | 1.5       |
| 1   | 3      | 1.0       |
| 1   | 4      | 0.5       |
| 1   | 5      | 0.0e+00   |
</details>

Figure 3. Problem statement

The simulation data are generated by running finite element analysis (FEA) with different parameters: k and P. We compare 4 different methods: MLP, SIREN, KAN, INN-TD. The results are summarized in Table. 1 where the root mean squared error (RMSE) is reported. Details on the model setup are provided in Appendix E. We can see that INN-TD outperforms in the training and test sets. Note that for KAN, the 30% and 100% training data cases use a batch size of 256 to prevent out-of-memory issues, while the 10% training data case employs full-batch optimization, resulting in a significantly lower error than the other two cases. Nevertheless, for all cases listed here, INN has higher accuracy compared to the other 3 different models.

Table 1. Root Mean Square Error (RMSE) Comparison for Training and Test Sets 

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td colspan="2">Training ( $\times 10^{-3}$ )</td><td colspan="2">Test ( $\times 10^{-3}$ )</td></tr><tr><td>Mean</td><td>Std</td><td>Mean</td><td>Std</td></tr><tr><td rowspan="4">Training 10%</td><td>MLP</td><td>4.180</td><td>0.873</td><td>3.660</td><td>0.498</td></tr><tr><td>SIREN</td><td>2.330</td><td>0.188</td><td>4.550</td><td>0.297</td></tr><tr><td>KAN</td><td>4.600</td><td>0.243</td><td>4.840</td><td>0.323</td></tr><tr><td>INN-TD</td><td>0.213</td><td>0.009</td><td>0.748</td><td>0.911</td></tr><tr><td rowspan="4">Training 30%</td><td>MLP</td><td>0.699</td><td>0.143</td><td>0.912</td><td>0.155</td></tr><tr><td>SIREN</td><td>1.240</td><td>0.161</td><td>2.020</td><td>0.197</td></tr><tr><td>KAN</td><td>4.710</td><td>1.120</td><td>4.490</td><td>0.996</td></tr><tr><td>INN-TD</td><td>0.154</td><td>0.003</td><td>0.154</td><td>0.006</td></tr><tr><td rowspan="4">Training 100%</td><td>MLP</td><td>0.385</td><td>0.037</td><td>0.534</td><td>0.041</td></tr><tr><td>SIREN</td><td>1.170</td><td>0.121</td><td>1.660</td><td>0.200</td></tr><tr><td>KAN</td><td>4.590</td><td>0.231</td><td>4.260</td><td>0.087</td></tr><tr><td>INN-TD</td><td>0.126</td><td>0.002</td><td>0.129</td><td>0.005</td></tr></table>

![](images/e6fa4b6545dfc3d2d818d1ee8390f994c2d6893025e64c0eea220fae9dfa0d77.jpg)

<details>
<summary>surface_3d</summary>

| x1    | x2    | u     |
|-------|-------|-------|
| 0.0   | 0.0   | 0.0   |
| 0.2   | 0.2   | 1.0   |
| 0.4   | 0.4   | 2.0   |
| 0.6   | 0.6   | 3.0   |
| 0.8   | 0.8   | 4.0   |
| 1.0   | 1.0   | 5.0   |
</details>

(a) Domain size $\pmb{x} \in [0,1]^2$

![](images/ec8b98cb75b10599159eea7e74565aef58a20d9b5a1797f94f4521652016bb1b.jpg)

<details>
<summary>surface_3d</summary>

| x1 \ u | 0    | 2    | 4    | 6    | 8    | 10   | 12   |
|--------|------|------|------|------|------|------|------|
| 0      | -4   | -2   | 0    | 2    | 4    | 6    | 8    |
| 2      | -4   | -2   | 0    | 2    | 4    | 6    | 8    |
| 4      | -4   | -2   | 0    | 2    | 4    | 6    | 8    |
| 6      | -4   | -2   | 0    | 2    | 4    | 6    | 8    |
| 8      | -4   | -2   | 0    | 2    | 4    | 6    | 8    |
| 10     | -4   | -2   | 0    | 2    | 4    | 6    | 8    |
| 12     | -4   | -2   | 0    | 2    | 4    | 6    | 8    |
| 14     | -4   | -2   | 0    | 2    | 4    | 6    | 8    |
| 16     | -4   | -2   | 0    | 2    | 4    | 6    | 8    |
| 18     | -4   | -2   | 0    | 2    | 4    | 6    | 8    |
| 20     | -4   | -2   | 0    | 2    | 4    | 6    | 8    |
| 22     | -4   | -2   | 0    | 2    | 4    | 6    | 8    |
| 24     | -4   | -2   | 0    | 2    | 4    | 6    | 8    |
| 26     | -4   | -2   | 0    | 2    | 4    | 6    | 8    |
| 28     | -4   | -2   | 0    | 2    | 4    | 6    | 8    |
| 30     | -4   | -2   | 0    | 2    | 4    | 6    | 8    |
| 32     | -4   | -2   | 0    | 2    | 4    | 6    | 8    |
| 34     | -4   | -2   | 0    | 2    | 4    | 6    | 8    |
| 36     | -4   | -2   | 0    | 2    | 4    | 6    | 8    |
| 38     | -4   | -2   | 0    | 2    | 4    | 6    | 8    |
| 40     | -4   | -2   | 0    | 2    | 4    | 6    | 8    |
| 42     | -4   | -2   | 0    | 2    | 4    | 6    | 8    |
| 44     | -4   | -2   | 0    | 2    | 4    | 6    | 8    |
| 46     | -4   | -2   | 0    | 2    | 4    | 6    | 8    |
| 48     | -4   | -2   | 0    | 2    | 4    | 6    | 8    |
| 50     | -4   | -2   | 0    | 2    | 4    | 6    | 8    |
| Note: The y-axis label is 'u' and the x-axis label is 'x1', but the y-axis label is 'x' and the x-axis label is 'x' and the y-axis label is 'u'. The color scale is 'blue' to 'red'. The label 'u' appears to be a variable in the chart.
</details>

(b) Domain size $\pmb{x} \in [0, 12]^2$   
Figure 4. Analytical solution for 2D cases.

# 4.2. Data-free Solving task

# 4.2.1. SOLVING HIGH-DIMENSIONAL PDE

In this example, we use INN-TD as a data-free solver to directly solve high-dimensional Poisson's equation with different source terms and compare its performance in terms of accuracy and speed against CP-PINN and KAN. Detailed model setups can be found in Appendix E.

The multi-dimensional Poisson's equation is written as:

$$
\Delta u (x _ {1}, x _ {2},..., x _ {D}) = f (x _ {1}, x _ {2},..., x _ {D}) \tag {16}
$$

where $\Delta(\cdot)$ is the Laplace operator which is defined as: $\Delta(\cdot):=\frac{\partial^{2}(\cdot)}{\partial x_{1}^{2}}+\frac{\partial^{2}(\cdot)}{\partial x_{2}^{2}}+,\ldots,\frac{\partial^{2}(\cdot)}{\partial x_{D}^{2}};D$ denotes the total number of dimensions.

Point-wise relative L2 norm error is used to measure the accuracy since INN-TD is also compared with collocatio-based machine learning method:

$$
\epsilon = \left[ \sum_ {k = 1} ^ {K} [ \mathcal {J} u (\boldsymbol {x} _ {k}) - u ^ {\text { exact }} (\boldsymbol {x} _ {k}) ] ^ {2} \right] ^ {\frac {1}{2}} / \left[ \sum_ {k = 1} ^ {K} u ^ {\text { exact }} (\boldsymbol {x} _ {k}) ^ {2} \right] ^ {\frac {1}{2}} \tag {17}
$$

where K is the total number of test data; $u^{exact}(\boldsymbol{x}_{k})$ is the analytical solution. We let $f(\boldsymbol{x}) = -\frac{\pi^{2}}{4}\sum_{i=1}^{D}\sin\left(\frac{\pi}{2}x_{i}\right)$ and the corresponding analytical solution is:

$$
u ^ {e x a c t} (\boldsymbol {x}) = \sum_ {i = 1} ^ {n} \left(\sin \left(\frac {\pi}{2} x _ {i}\right)\right) \tag {18}
$$

Two cases are examined here. In Case 1, the domain size is $[0,1]^D$ , while Case 2 explores a larger domain size $[0,12]^D$ . The analytical solution is shown in Fig. 4 for both cases when $D = 2$ . As evident from the figure, Case 2 is more challenging to solve due to the presence of higher-frequency signals in its analytical solution. It's known that vanilla MLP has spectral bias and thus poses challenges to learn Case 2 accurately and efficiently (Rahaman et al., 2019).

Table. 2 lists the detailed performance of each different method in Case 1. INN-TD stands out due to its better performance in terms of accuracy and computational efficiency. For instance, in the 2D setting, INN-TD achieves a low relative L2 norm error of $1.754 \times 10^{-8}$ in just 0.81 seconds, significantly outperforming other methods in both accuracy and speed. As the dimensionality increases to 5 and 10, INN-TD maintains its high accuracy with errors of $1.659 \times 10^{-8}$ and $1.238 \times 10^{-8}$ , respectively, while still being computationally efficient. CP-PINN and KAN are able to tackle high-dimensional problems. However, since the total number of collocation points will increase exponentially with the increase of dimension, they still suffer from expensive computational costs for high-dimensional problems. This demonstrates the robustness and scalability of the INN-TD method, making it a promising approach for solving high-dimensional PDEs.

Table 2. Comparison of different data-free solvers with domain size $[0,1]^{D}$ : INN-TD outperforms other models in both accuracy and speed. The mean values for error and wall time are reported. 

<table><tr><td>Model</td><td># dim</td><td># collocation points in each dim</td><td># model parameters</td><td>Relative L2 norm error</td><td>GPU wall time (sec)</td></tr><tr><td rowspan="3">INN-TD</td><td>2</td><td>-</td><td>256</td><td> $1.754 \times 10^{-8}$ </td><td>0.81</td></tr><tr><td>5</td><td>-</td><td>1,600</td><td> $1.659 \times 10^{-8}$ </td><td>6.88</td></tr><tr><td>10</td><td>-</td><td>6,400</td><td> $1.238 \times 10^{-8}$ </td><td>57.43</td></tr><tr><td rowspan="3">CP-PINN</td><td>2</td><td>32</td><td>96</td><td> $4.63 \times 10^{-3}$ </td><td>60.68</td></tr><tr><td>2</td><td>128</td><td>96</td><td> $2.64 \times 10^{-3}$ </td><td>169.2</td></tr><tr><td>5</td><td>32</td><td>1,200</td><td> $5.90 \times 10^{-3}$ </td><td>2,130</td></tr><tr><td rowspan="3">KAN</td><td>2</td><td>32</td><td>348</td><td> $9.44 \times 10^{-5}$ </td><td>56.16</td></tr><tr><td>2</td><td>128</td><td>348</td><td> $4.84 \times 10^{-5}$ </td><td>58.88</td></tr><tr><td>5</td><td>12</td><td>624</td><td> $1.68 \times 10^{-4}$ </td><td>842.1</td></tr></table>

Table. 3 provides a performance comparison of different models in Case 2. It is evident that the accuracy of both CP-PINN and KAN significantly decrease compared to Case 1, due to the increase in domain size. Despite utilizing more collocation points, the relative L2 norm error for these two methods does not converge to a smaller value, indicating possible limitations in applying them to industrial-level problems requiring high accuracy and solver convergence. In contrast, INN-TD demonstrates significantly better accuracy, outperforming other methods by orders of magnitude while requiring much less computational time.

Furthermore, we evaluate the convergence of the INN-TD data-free solver for Case 1. As illustrated in Fig.5 (a), INN-TD demonstrates favorable convergence properties, with reduced error achieved as the model complexity increases. This convergence property enhances confidence in the model's accuracy, enabling the design of the network to align with the anticipated precision.

In addition to this example, three more examples are included in Appendix E.2 to further demonstrate the efficiency and accuracy of INN-TD as a data-free solver for tackling

Table 3. Comparison of different data-free solvers with domain size $[0, 12]^{D}$ : INN-TD outperforms other models in both accuracy and speed. The mean values for error and wall time are reported. 

<table><tr><td>Model</td><td>Dim</td><td># collocation points in each dim</td><td># model parameters</td><td>Relative L2 norm error</td><td>GPU wall time (sec)</td></tr><tr><td rowspan="3">INN-TD</td><td>2</td><td>-</td><td>256</td><td> $4.77 \times 10^{-4}$ </td><td>0.22</td></tr><tr><td>5</td><td>-</td><td>1,600</td><td> $3.97 \times 10^{-4}$ </td><td>7.04</td></tr><tr><td>10</td><td>-</td><td>6,400</td><td> $3.71 \times 10^{-4}$ </td><td>59.75</td></tr><tr><td rowspan="3">CP-PINN</td><td>2</td><td>32</td><td>216</td><td> $2.5 \times 10^{-2}$ </td><td>87.6</td></tr><tr><td>2</td><td>1,024</td><td>216</td><td> $2.1 \times 10^{-2}$ </td><td>261.6</td></tr><tr><td>2</td><td>2,048</td><td>296</td><td> $3.2 \times 10^{-2}$ </td><td>292</td></tr><tr><td rowspan="3">KAN</td><td>2</td><td>32</td><td>348</td><td> $7.8 \times 10^{-1}$ </td><td>60.68</td></tr><tr><td>2</td><td>128</td><td>348</td><td> $8.0 \times 10^{-1}$ </td><td>72.6</td></tr><tr><td>2</td><td>128</td><td>3,188</td><td> $2.8 \times 10^{-1}$ </td><td>119.7</td></tr></table>

large-scale high-dimensional PDEs. In particular, we show INN-TD doesn't exhibit the failure modes of PINN when solving the Helmholtz equation (Wang et al., 2021).

# 4.2.2. SOLVING GENERAL SPACE-PARAMETER-TIME (S-P-T) PDE

A distinct advantage of machine learning-based data-free solvers is their ability to naturally handle parametric PDEs by treating parametric variables as inputs. In contrast, standard numerical solvers like FEM must rerun simulations for each different set of parametric inputs. In this example, we demonstrate that INN-TD effectively handles parametric PDEs by utilizing a space-parameter-time (S-P-T) interpolation. We consider solving the time-dependent parametric PDE described in Eq. 14, treating it as a 3D spatial problem $\boldsymbol{x}_{s} = (x, y, z)$ with 2D parametric inputs $\boldsymbol{x}_{p} = (k, P)$ and 1D temporal input $x_{t} = t$ .

The new source function can be written as:

$$
b (\boldsymbol {x}, P) = \Sigma_ {i = 1} ^ {n _ {s}} P \exp \left(- \frac {2 ((x - x _ {i}) ^ {2} + (y - y _ {i}) ^ {2})}{r _ {0} ^ {2}}\right) \mathbb {1} _ {z \geq d _ {0}} \tag {19}
$$

where $d_{0}$ is the depth of the source and $d_{0}=0.5$ in the current example; $1_{z\geq d_{0}}$ is the indicator function where $1_{z\geq d_{0}}$ equals 1 if $z\geq d_{0}$ or 0 if $z<d_{0}$ . Therefore, this PDE can be interpreted as modeling heat transfer with multiple fixed heat sources, each characterized by a radius $r_{0}$ and a penetration depth $d_{0}$ , as shown in Fig. 6.

This problem is solved using INN-TD using the algorithm listed in Algorithm 3 by discretizing each input dimension with 101 grid points and utilizing 100 modes. The total solving time takes 155 s on a single NVIDIA RTX A6000 GPU. Fig. 7 compares the INN-TD solution with FEM when k = 1.03, P = 101, t = 0.95.

To quantitatively assess the accuracy of the solution obtained from INN-TD, we measured the accuracy of INN-TD against the high-fidelity simulation by running implicit FEM with different realizations of parametric inputs $(k, P)$ . The relative L2 norm error for INN-TD is $3.38 \times 10^{-3}$ . Details about the error computation can be found in Appendix E.2.4.

![](images/98c8ac8b16083d18770f2beca4aa1ad54bf0836ed3fed62fe19323ef05153a91.jpg)

<details>
<summary>line</summary>

| Number of model parameters | s=1, p=1 | s=4, p=1 | s=2, p=2 | s=3, p=1 |
| --------------------------- | -------- | -------- | -------- | -------- |
| 2 × 10³                     | ~10⁻⁴    | ~10⁻⁶    | ~10⁻⁶    | ~10⁻⁵    |
| 3 × 10³                     | ~10⁻⁵    | ~10⁻⁷    | ~10⁻⁷    | ~10⁻⁶    |
| 4 × 10³                     | ~10⁻⁶    | ~10⁻⁸    | ~10⁻⁸    | ~10⁻⁷    |
| 6 × 10³                     | ~10⁻⁷    | ~10⁻⁹    | ~10⁻⁹    | ~10⁻⁸    |
</details>

(a)   
![](images/acac78419fcde65389f86bc78c77b09379a96a4e01dfc7151e8e70a788ae1153.jpg)

<details>
<summary>line</summary>

| Number of model parameters | s=1, p=1 | s=4, p=1 | s=2, p=2 | s=3, p=1 |
| --------------------------- | -------- | -------- | -------- | -------- |
| 2000                        | 0.70     | 0.75     | 0.72     | 0.73     |
| 3000                        | 0.75     | 0.80     | 0.76     | 0.78     |
| 4000                        | 0.80     | 0.85     | 0.80     | 0.82     |
| 6000                        | 0.90     | 0.95     | 0.92     | 0.93     |
</details>

(b)

![](images/96186526b5d9959fbab6d6ed872b6fa279564f8ca82fd938fc59a433348f95be.jpg)

<details>
<summary>line</summary>

| Number of model parameters | s=1, p=1 | s=4, p=1 | s=2, p=2 | s=3, p=1 |
| -------------------------- | -------- | -------- | -------- | -------- |
| 1                          | 7.65     | -        | 7.70     | -        |
| 2 × 10³                    | 7.70     | -        | 7.90     | -        |
| 3 × 10³                    | 7.90     | -        | 7.90     | -        |
| 4 × 10³                    | 8.00     | 8.10     | 8.25     | 8.25     |
| 6 × 10³                    | 8.25     | -        | 8.85     | -        |
</details>

(c)   
Figure 5. Solving the 2D Poisson's Eq. using INN-TD with different number of model parameters and hyperparameters s and p. a=20 is used for all cases. The statistics are obtained by repeating each case for 10 times and the shaded area represents 1 standard deviation (a) convergence of INN-TD using different s and p (b) statistics of total computational time (c) statistics of GPU VRAM usage

Multiple heat sources   
![](images/f481a5caf6f5807d2324f28e579b096fabe3ff3f884572536e425409b6471775.jpg)

<details>
<summary>text_image</summary>

1
0.1
0.5
1
</details>

Figure 6. S-P-T data-free solver problem

![](images/98def15d5076e3fc63e02b3e770a145e2d3d65fb3cb63ed7e175366e3a4b0b55.jpg)

<details>
<summary>heatmap</summary>

| Material | Value Range |
| -------- | ----------- |
| INN-TD   | 0.0e+00–2.1e+02 |
| FEM      | 0.0e+00–2.1e+02 |
</details>

Figure 7. Comparison of INN-TD and FEM solution for $k = 1.03$ , $P = 101$ , $t = 0.95$

More data-free solving problems can be found in Appendix E.2.

# 4.3. Inverse optimization task

In the previous section, we have shown INN-TD can effectively capture complex S-P-T function mapping for both data-driven training and data-free solving tasks. Consequently, for the inverse problem, we first train (solve) the function mapping from S-P-T continuum to output using the data-driven trainer (data-free solver). Then we recover the unknown parametric coefficients by minimizing the difference between prediction and measurements using automatic differentiation.

As an example, we use the same PDE as in section 4.2.2 to illustrate the performance of INN-TD for the inverse problem. In this task, we aim to find the optimal parametric inputs $(k, P)$ to obtain the target spatial-temporal field $u^{*}(\boldsymbol{x}_{s}, x_{t})$ .

We adopt the following 2 metrics to analyze the accuracy. The error of identified parametric input is defined as:

$$
\epsilon_ {x _ {p}} = \frac {\left| x _ {p} - x _ {p} ^ {*} \right|}{\left| x _ {p} ^ {*} \right|} \tag {20}
$$

The error of the prediction is defined as the relative L2 norm error between estimated output and target output.

$$
\epsilon_ {L 2} = \frac {\| \mathcal {J} u (\boldsymbol {x} _ {s} , x _ {t} ; \boldsymbol {x} _ {p}) - u ^ {*} (\boldsymbol {x} _ {s} , x _ {t}) \| _ {2}}{\| u ^ {*} (\boldsymbol {x} _ {s} , x _ {t}) \| _ {2}} \tag {21}
$$

The error metrics of this example are summarized in Table. 4. As shown in the table, INN-TD effectively recovers the optimal parametric inputs with high accuracy. Details on the model setups can be found in Appendix E.2.5.

Table 4. Error for inverse optimization task. 

<table><tr><td>Error Type</td><td>Mean</td><td>Standard Deviation</td></tr><tr><td> $\epsilon_{L_2}$ </td><td> $2.18 \times 10^{-3}$ </td><td> $5.64 \times 10^{-5}$ </td></tr><tr><td> $\epsilon_k$ </td><td> $2.76 \times 10^{-3}$ </td><td> $4.64 \times 10^{-4}$ </td></tr><tr><td> $\epsilon_P$ </td><td> $2.62 \times 10^{-3}$ </td><td> $3.11 \times 10^{-4}$ </td></tr></table>

# 5. Conclusion

In this paper, we introduced INN-TD as an efficient and accurate function approximation for large-scale, high-dimensional problems governed by PDEs. By combining the strengths of machine learning techniques and classical numerical algorithms, INN-TD demonstrated high accuracy in data-driven training, data-free solving, and inverse optimization tasks for problems where the underlying physics are described by PDEs. Particularly in the solving task, INN-TD achieves orders of magnitude better accuracy compared to other machine learning models. Convergence properties have been observed where more model parameters result in higher accuracy. Furthermore, INN-TD efficiently scales to higher-dimensional problems with extremely fast solving speeds. Lastly, due to its interpretable nature, we can tailor the structure of INN-TD according to the required resolution in physical problems. In conclusion, by unifying training, solving, and inverse optimization within a single framework, INN-TD offers a seamless approach to addressing challenging, large-scale physical problems that demand high precision and rapid solution speeds.

# Impact Statement

INN-TD lies at the intersection of artificial intelligence (AI) and traditional simulation approaches. By embedding inductive biases inspired by numerical analysis directly into the network architecture, INN-TD effectively tackles the challenges of low accuracy and high computational cost associated with current deep learning models when applied to high-dimensional, large-scale physical problems governed by PDEs.

# References

Ahmed, T., Raja, H., and Bajwa, W. U. Tensor regression using low-rank and sparse tucker decompositions. SIAM

Journal on Mathematics of Data Science, 2(4):944–966, 2020.   
Bachmayr, M. Low-rank tensor methods for partial differential equations. Acta Numerica, 32:1–121, 2023.   
Berardi, M., Difonzo, F. V., and Icardi, M. Inverse physics-informed neural networks for transport models in porous materials. Computer Methods in Applied Mechanics and Engineering, 435:117628, 2025.   
Beylkin, G. and Mohlenkamp, M. J. Algorithms for numerical analysis in high dimensions. SIAM Journal on Scientific Computing, 26(6):2133–2159, 2005.   
Bungartz, H.-J. and Griebel, M. Sparse grids. Acta numerica, 13:147–269, 2004.   
Cohen, A., DeVore, R., and Schwab, C. Convergence rates of best n-term galerkin approximations for a class of elliptic spdes. Foundations of Computational Mathematics, 10(6):615–646, 2010.   
Colbrook, M. J., Antun, V., and Hansen, A. C. The difficulty of computing stable and accurate neural networks: On the barriers of deep learning and smale's 18th problem. Proceedings of the National Academy of Sciences, 119(12):e2107151119, 2022.   
Cybenko, G. Approximation by superpositions of a sigmoidal function. Mathematics of control, signals and systems, 2(4):303-314, 1989.   
Dolgov, S. V., Khoromskij, B. N., and Oseledets, I. V. Fast solution of parabolic problems in the tensor train/quantized tensor train format with initial application to the fokker–planck equation. SIAM Journal on Scientific Computing, 34(6):A3016–A3038, 2012.   
Doshi-Velez, F. and Kim, B. Towards a rigorous science of interpretable machine learning. arXiv preprint arXiv:1702.08608, 2017.   
Gao, H., Sun, L., and Wang, J.-X. Phygeonet: Physics-informed geometry-adaptive convolutional neural networks for solving parameterized steady-state pdes on irregular domain. Journal of Computational Physics, 428:110079, 2021.   
Grossmann, T. G., Komorowska, U. J., Latz, J., and Schönlieb, C.-B. Can physics-informed neural networks beat the finite element method? IMA Journal of Applied Mathematics, pp. hxae011, 2024.   
Guo, J., Park, C., Xie, X., Sang, Z., Wagner, G. J., and Liu, W. K. Convolutional hierarchical deep learning neural networks-tensor decomposition (c-hidenn-td): a scalable surrogate modeling approach for large-scale physical systems. arXiv preprint arXiv:2409.00329, 2024.

Guo, J., Domel, G., Park, C., Zhang, H., Gumus, O. C., Lu, Y., Wagner, G. J., Qian, D., Cao, J., Hughes, T. J., et al. Tensor-decomposition-based a priori surrogate (taps) modeling for ultra large-scale simulations. arXiv preprint arXiv:2503.13933, 2025.   
Huang, O., Saha, S., Guo, J., and Liu, W. K. An introduction to kernel and operator learning methods for homogenization by self-consistent clustering analysis. Computational Mechanics, 72(1):195–219, 2023.   
Hughes, T. J. The finite element method: linear static and dynamic finite element analysis. Courier Corporation, 2003.   
Kharazmi, E., Zhang, Z., and Karniadakis, G. E. hp-vpinns: Variational physics-informed neural networks with domain decomposition. Computer Methods in Applied Mechanics and Engineering, 374:113547, 2021.   
Kolda, T. G. and Bader, B. W. Tensor decompositions and applications. SIAM review, 51(3):455–500, 2009.   
Krishnapriyan, A., Gholami, A., Zhe, S., Kirby, R., and Mahoney, M. W. Characterizing possible failure modes in physics-informed neural networks. Advances in neural information processing systems, 34:26548–26560, 2021.   
Li, H., Knapik, S., Li, Y., Park, C., Guo, J., Mojumder, S., Lu, Y., Chen, W., Apley, D. W., and Liu, W. K. Convolution hierarchical deep-learning neural network tensor decomposition (c-hidenn-td) for high-resolution topology optimization. Computational Mechanics, 72(2):363–382, 2023.   
Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., and Anandkumar, A. Fourier neural operator for parametric partial differential equations. arXiv preprint arXiv:2010.08895, 2020.   
Liu, W. K., Li, S., and Park, H. S. Eighty years of the finite element method: Birth, evolution, and future. Archives of Computational Methods in Engineering, 29(6):4431–4453, 2022.   
Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T. Y., and Tegmark, M. Kan: Kolmogorov-arnold networks. arXiv preprint arXiv:2404.19756, 2024.   
Lu, L., Jin, P., Pang, G., Zhang, Z., and Karniadakis, G. E. Learning nonlinear operators via deeponet based on the universal approximation theorem of operators. Nature machine intelligence, 3(3):218–229, 2021.   
Lu, Y., Li, H., Zhang, L., Park, C., Mojumder, S., Knapik, S., Sang, Z., Tang, S., Apley, D. W., Wagner, G. J., et al. Convolution hierarchical deep-learning neural networks

(c-hidenn): finite elements, isogeometric analysis, tensor decomposition, and beyond. Computational Mechanics, 72(2):333-362, 2023.   
McGreivy, N. and Hakim, A. Weak baselines and reporting biases lead to overoptimism in machine learning for fluid-related partial differential equations. Nature Machine Intelligence, pp. 1–14, 2024.   
Melenk, J. M. and Babuška, I. The partition of unity finite element method: basic theory and applications. Computer methods in applied mechanics and engineering, 139(1-4):289–314, 1996.   
Park, C., Lu, Y., Saha, S., Xue, T., Guo, J., Mojumder, S., Apley, D. W., Wagner, G. J., and Liu, W. K. Convolution hierarchical deep-learning neural network (c-hidenn) with graphics processing unit (gpu) acceleration. Computational Mechanics, 72(2):383–409, 2023.   
Park, C., Saha, S., Guo, J., Xie, X., Mojumder, S., Bessa, M. A., Qian, D., Chen, W., Wagner, G. J., Cao, J., et al. Engineering software 2.0 by interpolating neural networks: unifying training, solving, and calibration. arXiv preprint arXiv:2404.10296, 2024.   
Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., and Battaglia, P. W. Learning mesh-based simulation with graph networks. arXiv preprint arXiv:2010.03409, 2020.   
Rahaman, N., Baratin, A., Arpit, D., Draxler, F., Lin, M., Hamprecht, F., Bengio, Y., and Courville, A. On the spectral bias of neural networks. In International conference on machine learning, pp. 5301–5310. PMLR, 2019.   
Raissi, M., Perdikaris, P., and Karniadakis, G. E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational physics, 378:686–707, 2019.   
Rathinam, M. and Petzold, L. R. A new look at proper orthogonal decomposition. SIAM Journal on Numerical Analysis, 41(5):1893–1925, 2003.   
Rathore, P., Lei, W., Frangella, Z., Lu, L., and Udell, M. Challenges in training pinns: a loss landscape perspective. In Proceedings of the 41st International Conference on Machine Learning, ICML'24. JMLR.org, 2024.   
Sidiropoulos, N. D., De Lathauwer, L., Fu, X., Huang, K., Papalexakis, E. E., and Faloutsos, C. Tensor decomposition for signal processing and machine learning. IEEE Transactions on signal processing, 65(13):3551–3582, 2017.

Sitzmann, V., Martel, J., Bergman, A., Lindell, D., and Wetzstein, G. Implicit neural representations with periodic activation functions. Advances in neural information processing systems, 33:7462–7473, 2020.   
Takamoto, M., Praditia, T., Leiteritz, R., MacKinlay, D., Alesiani, F., Pflüger, D., and Niepert, M. Pdebench: An extensive benchmark for scientific machine learning. Advances in Neural Information Processing Systems, 35:1596–1611, 2022.   
Vemuri, S. K., Büchner, T., Niebling, J., and Denzler, J. Functional tensor decompositions for physics-informed neural networks. In International Conference on Pattern Recognition, pp. 32–46. Springer, 2025.   
Wang, S., Teng, Y., and Perdikaris, P. Understanding and mitigating gradient flow pathologies in physics-informed neural networks. SIAM Journal on Scientific Computing, 43(5):A3055–A3081, 2021.   
Wang, S., Yu, X., and Perdikaris, P. When and why pinns fail to train: A neural tangent kernel perspective. Journal of Computational Physics, 449:110768, 2022.   
Zhang, E., Dao, M., Karniadakis, G. E., and Suresh, S. Analyses of internal structures and defects in materials using physics-informed neural networks. Science advances, 8(7):eabk0644, 2022.

# A. C-HiDeNN interpolation theory

Convolutional-Hierarchical Deep Neural Network (C-HiDeNN) interpolation theory synergizes the advantages of finite element interpolation, mesh-free interpolation, and machine learning optimization. (Lu et al., 2023; Park et al., 2023). A 1D C-HiDeNN interpolation can be written in the algebraic form as

$$
\mathcal {J} u (x) = \sum_ {i \in A ^ {e}} N _ {i} (x) \sum_ {j \in A _ {s} ^ {i}} \mathcal {W} _ {i} ^ {(j)} (x; s, a, p) u _ {j} = \sum_ {k \in A _ {s} ^ {e}} \widetilde {N} _ {k} (x; s, a, p) u _ {k} = \widetilde {\boldsymbol {N}} (x; s, a, p) \boldsymbol {u} \tag {22}
$$

where $N_{i}(x)$ is the standard finite element basis function at node i; $W_{i}^{(j)}(x;s,a,p)$ is the convolution patch function at node j defined on the support domain $A_{s}^{i}$ . The double summation can be combined to a single summation over elemental nodal patches defined as $A_{s}^{e}=\bigcup_{i\in A^{e}}A_{s}^{i}$ . We can also interpret Eq. 22 as a partially connected MLP, as shown in Fig. 8.

![](images/9cb259f1d53138e61f8ab4f6284dc581b6c3ad3d89e864cbe92c72a6830a59fc.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["x_{i-3}"] --> B["x_{i-2}"]
    B --> C["x_{i-1}"]
    C --> D["x_i"]
    D --> E["ξ = -1"]
    E --> F["ξ = 1"]
    F --> G["s: patch size, integer"]
    F --> H["a: dilation parameter"]
    F --> I["p: reproducing polynomial order, integer"]
    J["A^e"] --> K["A^i = x_i"]
    L["A^s"] --> M["A^e_i = x_{i+1}"]
    N["s = 2, p = 2"] --> O["mapping"]
    style A fill:#f9f,stroke:#333
    style B fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style D fill:#f9f,stroke:#333
    style E fill:#f9f,stroke:#333
    style F fill:#f9f,stroke:#333
    style G fill:#ccf,stroke:#333
    style H fill:#ccf,stroke:#333
    style I fill:#ccf,stroke:#333
    style J fill:#dfd,stroke:#333
    style K fill:#dfd,stroke:#333
    style L fill:#dfd,stroke:#333
    style M fill:#dfd,stroke:#333
    style N fill:#dfd,stroke:#333
```
</details>

![](images/7973e7654467d4ccb5fde47c10db1e0b749f43d45265aeb243f59f9e28c344c5.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph (b)
        A["ξ"] --> B["x"]
        B --> C["convolution patch function"]
        C --> D["λ_i^A_i^s"]
        C --> E["λ_y=|x_i|^a-1"]
        C --> F["λ_y=R_1(x)"]
        C --> G["..."]
        C --> H["λ_y=x^i"]
        C --> I["..."]
        C --> J["λ_y=x^p"]
        C --> K["G_{1,1}^-T(x^{A_s}^i)"]
        C --> L["W_1(ξ)"]
        C --> M["W_2(ξ)"]
        C --> N["..."]
        C --> O["W_n(ξ)"]
        C --> P["G_{n,(n+m)}^{-T}(x^{A_s}^i)"]
        P --> Q["convolution patch functions"]
        Q --> R["λ_i^A_i^s"]
        Q --> S["λ_y=|x_i|^a-1"]
        Q --> T["λ_y=R_1(x)"]
        Q --> U["..."]
        Q --> V["..."]
        Q --> W["λ_y=x^i"]
        Q --> X["..."]
        Q --> Y["..."]
        Q --> Z["G_{n,(n+m)}^{-T}(x^{A_s}^i)"]
        Z --> AA["convolution patch functions"]
        AA --> AB["u^{h,e}(ξ) = Σ_{i∈A^e} N_i(ξ) Σ_{j∈A_i^e} W_{a,p,j}^{x_i}(x^{h,e}(ξ))u_j"]
        AA --> AC["u^{h,e}(ξ) = Σ_{i∈A^e} N_i(ξ) Σ_{j∈A_i^e} W_{a,p,j}^{x_i}(x^{h,e}(ξ))u_j"]
        AA --> AD["u^{h,e}(ξ) = Σ_{i∈A^e} N_i(ξ) Σ_{j∈A_i^e} W_{a,p,j}^{x_i}(x^{h,e}(ξ))u_j"]
    end
    subgraph (c)
        AE["ξ"] --> AF["λ_i^A_i^s"]
        AE --> AG["λ_y=|x_i|^a-1"]
        AE --> AH["λ_y=R_1(x)"]
        AE --> AI["..."]
        AE --> AJ["..."]
        AE --> AK["λ_y=x^i"]
        AE --> AL["..."]
        AE --> AM["..."]
        AE --> AN["G_{n,(n+m)}^{-T}(x^{A_s}^i)"]
        AN --> AO["u^{h,e}(ξ) = Σ_{i∈A^e} N_i(ξ) Σ_{j∈A_i^e} W_{a,p,j}^{x_i}(x^{h,e}(ξ))u_j"]
        AN --> AP["u^{h,e}(ξ) = Σ_{i∈A^e} N_i(ξ) Σ_{j∈A_i^e} W_{a,p,j}^{x_i}(x^{h,e}(ξ))u_j"]
    end
    subgraph (c)
        AQ["u^{h,e}(ξ) = Σ_{i∈A^e} N_i(ξ) Σ_{j∈A_i^e} W_{a,p,j}^{x_i}(x^{h,e}(ξ))u_j"]
        AQ --> AR["u^{h,e}(ξ) = Σ_{i∈A^e} N_i(ξ) Σ_{j∈A_i^e} W_{a,p,j}^{x_i}(x^{h,e}(ξ))u_j"]
    end
    subgraph (c)
        AS["u^{h,e}(ξ) = Σ_{i∈A^e} N_i(ξ) Σ_{j∈A_i^e} W_{a,p,j}^{x_i}(x^{h,e}(ξ))u_j"]
        AS --> AT["u^{h,e}(ξ) = Σ_{i∈A^e} N_i(ξ) Σ_{j∈A_i^e} W_{a,p,j}^{x_i}(x^{h,e}(ξ))u_j"]
    end
    subgraph (c)
        AU["u^{h,e}(ξ) = Σ_{i∈A^e} N_i(ξ) Σ_{j∈A_i^e} W_{a,p,j}^{x_i}(x^{h,e}(ξ))u_j"]
        AU --> AV["u^{h,e}(ξ) = Σ_{i∈A^e} N_i(ξ) Σ_{j∈A_i^e} W_{a,p,j}^{x_i}(x^{h,e}(ξ))u_j"]
    end
    subgraph (c)
        AW["u^{h,e}(ξ) = Σ_{i∈A^e} N_i(ξ) Σ_{j∈A_i^e} W_{a,p,j}^{x_i}(x^{h,e}(ξ))u_j"]
        AW --> AX["u^{h,e}(ξ) = Σ_{i∈A^e} N_i(ξ) Σ_{j∈A_i^e} W_{a,p,j}^{x_i}(x^{h,e}(ξ))u_j"]
    end
    subgraph (c)
        AY["u^{h,e}(ξ) = Σ_{i∈A^e} N_i(ξ) Σ_{j∈A_i^e} W_{a,p,j}^{x_i}(x^{h,e}(ξ))u_j"]
        AY --> AZ["u^{h,e}(ξ) = Σ_{i∈A^e} N_i(ξ) Σ_{j∈A_i^e} W_{a,p,j}^{x_i}(x^{h,e}(ξ))u_j"]
    end
    subgraph (c)
        BA["u^{h,e}(ξ) = Σ_{i∈A^e} N_i(ξ) Σ_{j∈A_i^e} W_{a,p,j}^{x_i}(x^{h,e}(ξ))u_j"]
        BA --> BB["u^{h,e}(ξ) = Σ_{i∈A^e} N_i(ξ) Σ_{j∈A_i^e} W_{a,p,j}^{x_i}(x^{h,e}(ξ))u_j"]
    end
    subgraph (c)
        BC["u^{h,e}(ξ) = Σ_{i∈A^e} N_i(ξ) Σ_{j∈A_i^e} W_{a,p,j}^{x_i}(x^{h,e}(ξ))u_j"]
        BC --> BD["u^{h,e}(ξ) = Σ_{i∈A^e} N_i(ξ) Σ_{j∈A_i^e} W_{a,p,j}^{x_i}(x^{h,e}(ξ))u_j"]
    end
    subgraph (c)
        BE["u^{h,e}(ξ) = Σ_{i∈A^e} N_i(ξ) Σ_{j∈A_i^e} W_{a,p,j}^{x_i}(x^{h,e}(ξ))u_j"]
        BE --> BF["u^{h,e}(ξ) = Σ_{i∈A^e} N_i(ξ) Σ_{j∈A_i^e} W_{a,p,j}^{x_i}(x^{h,e}(ξ))u_j"]
    end
    subgraph (c)
        BG["u^{h,e}(ξ) = Σ_{i∈A^e} N_i(ξ) Σ_{j∈A_i^e} W_{a,p,j}^{x_i}(x^{h,e}(ξ))u_j"]
        BG --> BH["u^{h,e}(ξ) = Σ_{i∈A^e} N_i(ξ) Σ_{j∈A_i^e} W_{a,p,j}^{x_i}(x^{h,e}(ξ))u_j"]
    end
    subgraph (c)
        BI["u^{h,e}(ξ) = Σ_{i∈A^e} N_i(ξ) Σ_{j∈A_i^e} W_{a,p,j}^{x_i}(x^{h,e}(ξ))u_j"]
        BI --> BJ["u^{h,e}(ξ) = Σ_{i∈A^e} N_i(ξ) Σ_{j∈A_i^e} W_{a,p,j}^{x_i}(x^{h,e}(ξ))u_j"]
    end
    subgraph (c)
        BK["u^{h,e}(ξ) = Σ_{i∈A^e} N_i(ξ) Σ_{j∈A_i^e} W_{a,p,j}^{x_i}(x^{h,e}(ξ))u_j"]
        BK --> BL["u^{h,e}(ξ) = Σ_{i∈A^e} N_i(ξ) Σ_{j∈A_i^e} W_{a,p,j}^{x_i}(x^{h,e}(ξ))u_j"]
    end
```
</details>

Figure 8. (a) Convolution patch in 1D C-HiDeNN shape function (b) Construction of convolution patch function (c) C-HiDeNN shape function as MLP with 3 hidden layers. This plot is adopted from (Park et al., 2024).

In Fig. 8, the convolution patch function $W_{i}^{(j)}(x; s, a, p)$ is controlled by three hyperparameters: patch size s that controls nodal connectivity, dilation parameter a that normalizes distances between patch nodes, and reproducing order p that defines types/orders of activation functions to be reproduced by the patch functions. C-HiDeNN can adapt to these hyperparameters node by node using optimization, rendering an adaptable functional space without altering the number of nodes or layers.

Unlike the traditional black-box MLP, C-HiDeNN can be interpreted as a tailored MLP structure that preserves several desired properties (inductive biases) that are particularly important in numerical analysis: locality, Kronecker delta property (Hughes, 2003), partition of unity (Melenk & Babuška, 1996), and easy for integration (Hughes, 2003). These attributes enable C-HiDeNN to efficiently model complex physical problems with high accuracy.

Table 5. Comparison of MLP and C-HiDeNN 

<table><tr><td>Properties</td><td>MLP</td><td>C-HiDeNN</td></tr><tr><td>Boundary/initial condition</td><td>Penalty term in the loss function (Raissi et al., 2019)</td><td>Automatic satisfaction due to Kronecker delta (Lu et al., 2023)</td></tr><tr><td>Convergence and stability</td><td>Stochastic and not guaranteed (Colbrook et al., 2022)</td><td>Shown for different PDEs (Guo et al., 2024)</td></tr><tr><td>Numerical integration</td><td>Quasi-Monte Carlo integration (Kharazmi et al., 2021)</td><td>Gaussian quadrature (Hughes, 2003)</td></tr><tr><td>Interpretability</td><td>Black-box model</td><td>Fully interpretable</td></tr></table>

# B. Interpretability of INN-TD

Unlike most black-box machine learning models, INN-TD has a fully interpretable architecture. As shown in Fig. 9, INN-TD can be interpreted as a specially pruned MLP where the first 2 hidden layers represent locally supported linear finite element basis functions. The 3rd hidden layer reconstructs the C-HiDeNN interpolation function with higher-order smoothness which is controlled by hyperparameter patch size $s$ and reproducing polynomial order $p$ . The 4th hidden layer leverages tensor decomposition to counter the curse of dimensionality. As a result, INN-TD's learning parameters are interpretable nodal values of TD components, in contrast to the opaque weights and biases of neural networks.

To demonstrate the benefits of INN's interpretability, we solve the following 2D Poisson's equation in $\Omega = [0,100]^2$ with a local source function.

$$
\left(\frac {\partial^ {2}}{\partial x _ {1} ^ {2}} + \frac {\partial^ {2}}{\partial x _ {2} ^ {2}}\right) u = f (x, y) \tag {23}
$$

The local Gaussian source function $f(x, y)$ is defined as:

$$
\begin{array}{l} f (x, y) = 0. 0 6 4 (x - 4 0) ^ {2} e ^ {- 0. 0 4 (x - 4 4) ^ {2}} e ^ {- 0. 0 4 (y - 7 5) ^ {2}} + 0. 0 6 4 (x - 2 0) ^ {2} e ^ {- 0. 0 4 (x - 2 0) ^ {2}} e ^ {- 0. 0 4 (y - 2 5) ^ {2}} \\ + 0. 0 6 4 (y - 1 5) ^ {2} e ^ {- 0. 0 4 (x - 4 6) ^ {2}} e ^ {- 0. 0 4 (y - 5 5) ^ {2}} + 0. 0 6 4 (y - 2 5) ^ {2} e ^ {- 0. 0 4 (x - 2 6) ^ {2}} e ^ {- 0. 0 4 (y - 2 5) ^ {2}} \tag {24} \\ - 1. 6 e ^ {- 0. 0 4 (x - 2 0) ^ {2}} e ^ {- 0. 0 4 (y - 2 5) ^ {2}} - 1. 6 e ^ {- 0. 0 4 (x - 4 0) ^ {2}} e ^ {- 0. 0 4 (y - 5 5) ^ {2}} \\ \end{array}
$$

The exact solution to this problem is:

$$
u ^ {e x} (x, y) = 1 0 e ^ {- \frac {(x - 2 0) ^ {2}}{2 5}} e ^ {- \frac {(y - 2 5) ^ {2}}{2 5}} + 1 0 e ^ {- \frac {(x - 6 0) ^ {2}}{2 5}} e ^ {- \frac {(y - 7 5) ^ {2}}{2 5}} \tag {25}
$$

The Dirichlet boundary condition is $u|_{\partial \Omega} = u^{ex}(x,y)|_{\partial \Omega}$ .

To efficiently solve this problem, we leverage the interpretable locally supported basis function in INN-TD. As can be seen from Fig. 10 (a), a nonuniform mesh is used to accommodate the local source function: a fine mesh is used for the source function region, whereas a coarse mesh is used for other regions. Moreover, since the nonlinearity of the solution is expected to be localized near the source function region, larger s and p can be used for these local regions to improve the accuracy. The absolute pointwise error with respect to the exact solution is plotted in Fig. 10 (d) where the order is around $10^{-6}$ .

# C. INN-TD trainer

Two different training schemes can be used for INN-TD trainer. The first method is based on the boosting algorithm. The core idea behind boosting is to convert a set of weak learners into a strong learner. A weak learner is a model that performs slightly better than random guessing. The boosting process involves sequentially training multiple weak learners, each trying to correct the errors made by its predecessor.

We define the weak learner for mode m as $\mathcal{J}^{(m)}\boldsymbol{y}(\boldsymbol{x};\boldsymbol{U}^{(m)})$ , where $\boldsymbol{U}^{(m)} = [\boldsymbol{u}_{1}^{(m)}, \boldsymbol{u}_{2}^{(m)}, ..., \boldsymbol{u}_{D}^{(m)}]$ is the model parameter for mode m. The strong learner $\mathcal{J}\boldsymbol{y}(\boldsymbol{x};\boldsymbol{U})$ can be obtained by adding weak learners for each mode, and we define $\boldsymbol{U} = [\boldsymbol{U}^{(1)}, \boldsymbol{U}^{(2)}, ..., \boldsymbol{U}^{(M)}]$ .

$$
\mathcal {J} \boldsymbol {y} (\boldsymbol {x}; \boldsymbol {U}) = \Sigma_ {m = 1} ^ {M} \mathcal {J} ^ {(m)} \boldsymbol {y} (\boldsymbol {x}; \boldsymbol {U} ^ {(m)}) \tag {26}
$$

![](images/f99bfcfa0cc289bd214aad267f10983116c64639d67b235bc30328c5fd33a161.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Input Variables
        A["Input variables x1, x2, ..., xd"]
    end

    subgraph C-HiDeNN
        B["x1"] --> C["Neural Network"]
        C --> D["Neural Network"]
        D --> E["Neural Network"]
        E --> F["Neural Network"]
        F --> G["Neural Network"]
        G --> H["Neural Network"]
        H --> I["Neural Network"]
    end

    subgraph HiDeNN
        J["x1"] --> K["ReLU"]
        K --> L["ReLU"]
        L --> M["ReLU"]
        M --> N["ReLU"]
        N --> O["ReLU"]
        O --> P["ReLU"]

    end

    subgraph Mode M
        Q["x1"] --> R["Element 1"]
        R --> S["Element 1"]
        S --> T["Element 1"]
        T --> U["Element 1"]
        U --> V["Element 1"]
        V --> W["Element 1"]
        W --> X["Element 1"]
        X --> Y["Element 1"]
        Y --> Z["Element 1"]
        Z --> AA["Element 1"]
        AA --> AB["Element 1"]
        AB --> AC["Element 1"]
        AC --> AD["Element 1"]
        AD --> AE["Element 1"]
        AE --> AF["Element 1"]
        AF --> AG["Element 1"]
        AG --> AH["Element 1"]
        AH --> AI["Element 1"]
        AI --> AJ["Element 1"]
        AJ --> AK["Element 1"]
        AK --> AL["Element 1"]
        AL --> AM["Element 1"]
        AM --> AN["Element 1"]
        AN --> AO["Element 1"]
        AO --> AP["Element 1"]
        AP --> AQ["Element 1"]
        AQ --> AR["Element 1"]
        AR --> AS["Element 1"]
        AS --> AT["Element 1"]
        AT --> AU["Element 1"]
        AU --> AV["Element 1"]
        AV --> AW["Element 1"]
    end

    subgraph Mode 1
        AX["x2"] --> AY["Element 1"]
        AZ["x3"] --> BA["Element 1"]
        BB["x4"] --> BC["Element 1"]
        BD["x5"] --> BE["Element 1"]
        BF["x6"] --> BG["Element 1"]
        BH["x7"] --> BI["Element 1"]
        BJ["x8"] --> BK["Element 1"]
        BL["x9"] --> BM["Element 1"]
        BN["x10"] --> BO["Element 1"]
        BP["x11"] --> BQ["Element 1"]
        BR["x12"] --> BS["Element 1"]
        BT["x13"] --> BU["Element 1"]
        BV["x14"] --> BW["Element 1"]
        BX["x15"] --> BY["Element 1"]
        BZ["x16"] --> CA["Element 1"]
        CB["x17"] --> CC["Element 1"]
        DD["x18"] --> EE["Element 1"]
        FF["x19"] --> DG["Element 1"]
        DH["x20"] --> DI["Element 1"]
        DJ["x21"] --> DK["Element 1"]
        DL["x22"] --> DM["Element 1"]
        DN["x23"] --> DE["Element 1"]
        DF["x24"] --> DG
        DG --> DG
    end

    subgraph Mode 2
        EJ["xj"] --> KJ["Aj, Ae, n, trn, cn, cnn, cnn, cnn, cnn, cnn, cnn, cnn, cnn, cnn, cnn, cnn, cnn, cnn, cnn, cnn, cnn, cnn, cnn, cnn, cnn, cnn, cnn, cnn, cnn, cnn, cnn, cnn, cnn, cnn, cnn, cnn, cnn, cnn, cnn<br>    end<br><br>    subgraph Mode M<br>        OS[xo"] --> R
        OS --> S
        OS --> TW
        OS --> X
        OS --> Y
        OS --> Z
    end

    style Input Variables fill:#f9f9f9,stroke:#333
    style Mode 2 fill:#f9f9f9,stroke:#333
```
</details>

Figure 9. Interpretable structure of INN-TD

The loss of model prediction and training data $(x^{*}, y^{*})$ is defined using the mean squared error:

$$
\mathcal {L} = \frac {1}{K} \sum_ {k} (\mathcal {J} \boldsymbol {y} (x _ {\boldsymbol {k}} ^ {*}; \boldsymbol {U}) - \boldsymbol {y} _ {\boldsymbol {k}} ^ {*}) ^ {2}. \tag {27}
$$

Assuming the previous M-1 weak learners have been obtained, and we aim to learn the new weak learner $\mathcal{J}^{(M)}\boldsymbol{y}(\boldsymbol{x};\boldsymbol{U}^{(M)})$ , Eq. 27 can be written as:

$$
\mathcal {L} = \frac {1}{K} \sum_ {k} (\Sigma_ {m = 1} ^ {M - 1} \mathcal {J} ^ {(m)} \boldsymbol {y} (\boldsymbol {x} _ {\boldsymbol {k}} ^ {*}; \boldsymbol {U} ^ {(m)}) + \mathcal {J} ^ {(M)} \boldsymbol {y} (\boldsymbol {x} _ {\boldsymbol {k}} ^ {*}; \boldsymbol {U} ^ {(M)}) - \boldsymbol {y} _ {\boldsymbol {k}} ^ {*}) ^ {2}. \tag {28}
$$

Consequently, the goal is to learn to model parameters $\pmb{U}^{(M)}$ for new weak learner $\mathcal{J}^{(M)}\pmb{y}(\pmb{x};\pmb{U}^{(M)})$ using optimization schemes.

$$
\underset {\boldsymbol {U} ^ {(M)}} {\operatorname{argmin}} \mathcal {L} = \frac {1}{K} \sum_ {k} \left(\Sigma_ {m = 1} ^ {M - 1} \mathcal {J} ^ {(m)} \boldsymbol {y} \left(\boldsymbol {x} _ {\boldsymbol {k}} ^ {*}; \boldsymbol {U} ^ {(m)}\right) + \mathcal {J} ^ {(M)} \boldsymbol {y} \left(\boldsymbol {x} _ {\boldsymbol {k}} ^ {*}; \boldsymbol {U} ^ {(M)}\right) - \boldsymbol {y} _ {\boldsymbol {k}} ^ {*}\right) ^ {2}. \tag {29}
$$

The complete boosting algorithm for INN-TD trainer is shown in algorithm 1.

Instead of gradually enhancing the accuracy of the model by adding weak learners, one can also predefine the total number of modes M required in the model and optimize the unknown parameters in the model in an all-at-once fashion as shown in Eq. 26. Consequently, we treat the total number of modes as one additional hyperparameter. The all-at-once training

![](images/69404b2ca0a13dab1bd4ed08e8ada790b08d0e2fd707ee402b1a5cb6200cf03f.jpg)

<details>
<summary>text_image</summary>

INN-TD design
coarse mesh
fine mesh
r = 5
(60,75)
Gaussian
r = 5
(20,25)
x²
s = p = 1
s = p = 3
100
x
100
</details>

(a)

![](images/0ad0857d42fd647ec4c5815fb00d3c552d2f19c108ec46c8b005f66d03a0451f.jpg)

<details>
<summary>heatmap</summary>

| x1 \ x2 | 0    | 20   | 40   | 60   | 80   | 100  |
|---------|------|------|------|------|------|------|
| 20      | 2.5  | 2.5  | 2.5  | 2.5  | 2.5  | 2.5  |
| 60      | 7.5  | 7.5  | 7.5  | 7.5  | 7.5  | 7.5  |
</details>

(b)

![](images/daf157dd832b85b1ca4008f40a7f613588b954f9064636eb2fdcfa0782ce1848.jpg)

<details>
<summary>heatmap</summary>

| x1  | y2  | Value |
| --- | --- | ----- |
| 20  | 25  | 8.8   |
| 60  | 75  | 9.9   |
</details>

(c)

![](images/0b410679f1e2c9c4f20156cba815db9c57f5ec2d570e7d76b150f8d62254049c.jpg)

<details>
<summary>heatmap</summary>

| x1 \ x2 | 0    | 20   | 40   | 60   | 80   | 100  |
|---------|------|------|------|------|------|------|
| 0       | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| 20      | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| 40      | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| 60      | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| 80      | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| 100     | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
</details>

(d)   
Figure 10. Interpretable design of INN-TD: (a) 1D meshes used in INN-TD are only refined at the location where nonlinearity happens with larger s and p used to achieve higher-order smoothness (b) INN-TD solution (c) Exact solution (d) point-wise absolute error

Algorithm 1 INN-TD trainer: boosting   
Define maximum number of weak learners (modes) M, maximum epoch number $epoch_{max}$ , grid for each dimension $x_{d}$ Initialize model $\mathcal{J}\boldsymbol{y}(\boldsymbol{x}) = 0$ for (epoch loop) epoch = 1 to $epoch_{max}$ do

    for (boosting loop) m = 1 to $M_{max}$ do

    Initialize solution vector $U^{(m)}$ for the m-th weak learner.

    Optimize Eq. 29 using optimization schemes such as Adam

    Update learner: $\mathcal{J}\boldsymbol{y}(\boldsymbol{x}) = \mathcal{J}\boldsymbol{y}(\boldsymbol{x}) + \mathcal{J}^{(m)}\boldsymbol{y}(\boldsymbol{x};\boldsymbol{U}^{(m)})$ Check loss

    end for

end for

strategy has been shown to generally lead to better accuracy compared to the boosting strategy when the total number of modes M is the same.

The complete all-at-once algorithm for INN-TD trainer is shown in algorithm 2. We use this algorithm for all of the training experiments presented in the paper.

Algorithm 2 INN-TD trainer: all-at-once   
Define total number of modes $M$ , maximum epoch number $\mathrm{epoch}_{max}$ , grid for each dimension $\pmb{x}_d$ . Initialize model parameters $\pmb{U} = [\pmb{U}^{(1)}, \pmb{U}^{(2)}, ..., \pmb{U}^{(M)}]$ .  
for (epoch loop) epoch = 1 to $\mathrm{epoch}_{max}$ do  
    Optimize Eq. 26 using optimization schemes such as Adam.  
    Check loss  
end for

# D. INN-TD solver

Boosting algorithm is adopted for INN-TD solver. Such method is known as proper generalized decomposition (PGD) in the field of computational mechanics (Li et al., 2023). A similar all-at-once approach named a priori tensor decomposition is also available (Guo et al., 2024). Different from the training approach where the loss is computed in a point-wise manner, we compute the PDE loss in a weighted-sum residual format. Without loss of generality, we focus on the discussion of the scalar solution field. Vector and higher-order tensorial solution fields can be treated in a similar way. Assume PDE can be written as:

$$
\mathcal {L} u (\boldsymbol {x}) = 0 \tag {30}
$$

where L is a general (nonlinear) differential operator, $u(\boldsymbol{x})$ is the D-dimensional PDE solution depending on independent variables $\boldsymbol{x} = (x_{1}, x_{2}, ..., x_{D})$ . The INN-TD approximation to the solution can be written as:

$$
\mathcal {J} u (\boldsymbol {x}) = \sum_ {m = 1} ^ {M} \widetilde {\boldsymbol {N}} _ {1} (x _ {1}) \boldsymbol {u} _ {x _ {1}} ^ {(m)} \cdot \widetilde {\boldsymbol {N}} _ {2} (x _ {2}) \boldsymbol {u} _ {x _ {2}} ^ {(m)} \cdot \dots \cdot \widetilde {\boldsymbol {N}} _ {D} (x _ {D}) \boldsymbol {u} _ {x _ {D}} ^ {(m)} \tag {31}
$$

In the context of solutions to time-dependent parametric partial differential equations (PDEs), the independent variables $(x_{1}, x_{2}, \ldots, x_{D})$ can be divided into three distinct categories: spatial variables $x_{s}$ , parametric variables $x_{p}$ , and temporal variables $x_{t}$ . Spatial variables $x_{s}$ define the spatial coordinates relevant to the problem. Parametric variables $x_{p}$ serve as additional coordinates and can represent PDE coefficients, initial conditions, boundary conditions, and geometry descriptors. Temporal variable $x_{t}$ represents time.

The weighted-sum residual form is adopted to solve PDEs.

$$
\int \delta (\mathcal {J} u (\boldsymbol {x})) \mathcal {L} (\mathcal {J} u (\boldsymbol {x})) d \boldsymbol {x} = 0 \tag {32}
$$

where $\delta(\mathcal{J}u(\boldsymbol{x}))$ is the so-called weighting function. Depending on the form of the PDE, different choices of $\delta(\mathcal{J}u(\boldsymbol{x}))$ can be adopted. For instance, if the Dirac Delta function $\delta(\boldsymbol{x}-\boldsymbol{x}_{i})$ ( $\delta$ here refers to the Dirac Delta function) is used for the weighting function where $x_{i}$ is the collocation point, Eq. 32 becomes the differential (strong) form: $\mathcal{L}u(\boldsymbol{x})=0$ . In this paper, we adopt the Galerkin formulation, where the function space of the weighting function is the same as our approximation. Moreover, integration by parts can be used to alleviate the differentiability requirement on the approximated solution (Hughes, 2003).

In the boosting solution scheme, the solution is obtained incrementally in a mode by mode fashion:

$$
\mathcal {J} u (\boldsymbol {x}) = \sum_ {m = 1} ^ {M - 1} \widetilde {\boldsymbol {N}} _ {1} (x _ {1}) \boldsymbol {u} _ {x _ {1}} ^ {(m)} \cdot \widetilde {\boldsymbol {N}} _ {2} (x _ {2}) \boldsymbol {u} _ {x _ {2}} ^ {(m)} \cdot \dots \cdot \widetilde {\boldsymbol {N}} _ {D} (x _ {D}) \boldsymbol {u} _ {x _ {D}} ^ {(m)} + \widetilde {\boldsymbol {N}} _ {1} (x _ {1}) \boldsymbol {u} _ {x _ {1}} \cdot \widetilde {\boldsymbol {N}} _ {2} (x _ {2}) \boldsymbol {u} _ {x _ {2}} \cdot \dots \cdot \widetilde {\boldsymbol {N}} _ {D} (x _ {D}) \boldsymbol {u} _ {x _ {D}} \tag {33}
$$

where we neglect the superscript $M$ for the last mode $M$ that we are solving. As a result, the corresponding weighting function can be written as:

$$
\delta \mathcal {J} u (\boldsymbol {x}) = \widetilde {\boldsymbol {N}} _ {1} (x _ {1}) \delta \boldsymbol {u} _ {x _ {1}} \cdot \widetilde {\boldsymbol {N}} _ {2} (x _ {2}) \boldsymbol {u} _ {x _ {2}} \cdot \dots \cdot \widetilde {\boldsymbol {N}} _ {D} (x _ {D}) \boldsymbol {u} _ {x _ {D}} +
$$

$$
\widetilde {\widetilde {N}} _ {1} \left(x _ {1}\right) \boldsymbol {u} _ {x _ {1}} \cdot \widetilde {\widetilde {N}} _ {2} \left(x _ {2}\right) \delta \boldsymbol {u} _ {x _ {2}} \dots \cdot \widetilde {\widetilde {N}} _ {D} \left(x _ {D}\right) \boldsymbol {u} _ {x _ {D}} + \tag {34}
$$

$$
\widetilde {\boldsymbol {N}} _ {1} (x _ {1}) \boldsymbol {u} _ {x _ {1}} \cdot \widetilde {\boldsymbol {N}} _ {2} (x _ {2}) \boldsymbol {u} _ {x _ {2}} \cdot \dots \cdot \widetilde {\boldsymbol {N}} _ {D} (x _ {D}) \boldsymbol {u} _ {x _ {D}} +
$$

$$
\widetilde {\pmb {N}} _ {1} (x _ {1}) \pmb {u} _ {x _ {1}} \cdot \widetilde {\pmb {N}} _ {2} (x _ {2}) \pmb {u} _ {x _ {2}} \cdot \ldots \cdot \widetilde {\pmb {N}} _ {D} (x _ {D}) \delta \pmb {u} _ {x _ {D}}
$$

Based on Eq. 34, we further use subspace iteration to linearize the problem. In subspace iteration, we sequentially alternate 1 unknown variable while treating others as known constant values. In this case, the variation for other unknowns will be 0. Taking subspace iteration in $x_{1}$ as an example, Eq. 34 becomes:

$$
\delta_ {x _ {1}} \mathcal {J} u (\boldsymbol {x}) = \widetilde {\boldsymbol {N}} _ {1} (x _ {1}) \delta \boldsymbol {u} _ {x _ {1}} \cdot \widetilde {\boldsymbol {N}} _ {2} (x _ {2}) \boldsymbol {u} _ {x _ {2}} \cdot \dots \cdot \widetilde {\boldsymbol {N}} _ {D} (x _ {D}) \boldsymbol {u} _ {x _ {D}} \tag {35}
$$

Plugging Eq. 33, 35 into Eq. 32 and integrating along each 1D dimension, the matrix equations for $x_{1}$ dimension can be derived:

$$
\boldsymbol {A} _ {x _ {1}} \boldsymbol {u} _ {x _ {1}} = \boldsymbol {Q} _ {x _ {1}} \tag {36}
$$

where $u_{x_{1}}$ is the solution vector for the current dimension at the current mode; $A_{x_{1}}$ is a banded sparse matrix as in the standard finite element method thanks to locally supported basis functions; $Q_{x_{1}}$ is a 1D vector.

Similarly, we can sequentially alternate other dimensions and obtain the corresponding linear system of equations:

$$
\boldsymbol {A} _ {x _ {d}} \boldsymbol {u} _ {x _ {d}} = \boldsymbol {Q} _ {x _ {d}} \tag {37}
$$

where $d = 1, \ldots, D$ ; $u_{x_{d}}$ is the solution vector for the current dimension at the current mode; $A_{x_{d}}$ is a banded sparse matrix; $Q_{x_{d}}$ is a 1D vector. As a result, Eq. 37 can be efficiently solved using sparse matrix solvers. We can keep iterating Eq. 37 until the variation of solution vector for each dimension is within tolerance. In practice, 3-5 iterations will yield good accuracy.

After the current mode is solved, we can introduce additional modes to further enhance the accuracy of the INN-TD model. This iterative process will be continued until the results meet the predefined accuracy criteria.

The complete boosting algorithm for INN-TD solver is shown in algorithm 3.

Algorithm 3 INN-TD solver: boosting   
Define total number of modes M; maximum number of iteration $iter_{max}$ ; tolerance; grid for each dimension $x_{d}$ Initialize solution $\mathcal{J}u(\boldsymbol{x}) = 0$ Initialize solution vector $\boldsymbol{u}_{x_{d}}^{(m)}$ .

for (boosting loop) m = 1 to M do

    for (subspace iteration loop) iter = 1 to $iter_{max}$ do

    for (dimension loop) d = 1 to D do

    Update $A_{x_{d}}$ and $Q_{x_{d}}$ Solve $A_{x_{d}}\boldsymbol{u}_{x_{d}}^{(m)} = \boldsymbol{Q}_{x_{d}}$ using sparse matrix solver

    end for

    Check convergence

end for $\mathcal{J}u(\boldsymbol{x}) = \mathcal{J}u(\boldsymbol{x}) + \widetilde{\boldsymbol{N}}_{1}(x_{1})\boldsymbol{u}_{x_{1}}^{(m)} \cdot \widetilde{\boldsymbol{N}}_{2}(x_{2})\boldsymbol{u}_{x_{2}}^{(m)} \cdot \ldots \cdot \widetilde{\boldsymbol{N}}_{D}(x_{D})\boldsymbol{u}_{x_{D}}^{(m)}$ Check Error

if Error ≤ Tolerance then

    Break

end if

end for

# E. Detailed model setups

# E.1. Data-driven training examples

# E.1.1. MODEL SETUPS OF TRAINING ON PARAMETRIC PDE SOLUTIONS

This section presents the hyperparameters for training models on the PDE dataset. For the MLP model, we set the initial learning rate to 0.001, the batch size to 128, and use the Adam optimizer. Training runs for a maximum of 500 epochs with early stopping based on the validation error. If the validation error increases for 10 consecutive epochs, the model training will be stopped. We also apply an adaptive learning rate scheduler, reducing the learning rate by 10% after each epoch. The MLP model has 5 hidden layers, each having 50 neurons.

For the SIREN model, we use almost the same hyperparameters as the MLP model. But we reduce the initial learning rate to 0.0001 to avoid loss explosion. We also increase the model complexity of the SIREN model. The model consists of 4 hidden layers, each containing 100 neurons.

For the KAN model, we set the layer widths to $[5, 5, 5, 1]$ for the 10% training data case, given that the input and output dimensions are $5(x, y, t, k, p)$ and $1(u)$ , respectively. We use 10 grid intervals, third-order of piecewise polynomials, and the LBFGS optimizer with 2,000 steps and full-batch training. For the 30% and 100% training data cases, full-batch training is infeasible due to the GPU memory limitation, so we set the batch size to 256. With a smaller batch size, we increase the layer widths to $[5, 10, 10, 1]$ and extend the number of training steps to 3,000. Notably, training loss can easily explode to NaN and increasing batch size does not change the final error too much.

The INN-TD model uses 60 modes with 40 elements while the hyperparameters are: patch size s = 4, dilation parameter a = 20, and polynomial order p = 1. Thus, there are a total of $5 \times 60 \times 41 = 12$ , 300 trainable parameters. We use the batch size of 128 for 100 epochs with the same early stopping criteria adopted in MLP. The Adam optimizer is adopted with a learning rate of 0.0001.

# E.2. Data-free solver examples

# E.2.1. HIGH-DIMENSIONAL PDE

This subsection has detailed information discussed in section 4.2.1. Four different models, namely, INN-TD, PINN, CP-PINN, and KAN, were used as data-free solvers. The following high-dimensional PDE is solved.

$$
\Delta u (x _ {1}, x _ {2},..., x _ {D}) = f (x _ {1}, x _ {2},..., x _ {D}) \tag {38}
$$

In case 1, we adopt the following analytical solution:

$$
u ^ {e x a c t} (\boldsymbol {x}) = \sum_ {i = 1} ^ {n} \left(\sin \left(\frac {\pi}{2} x _ {i}\right)\right) \tag {39}
$$

The corresponding right-hand side (RHS) is:

$$
f (\boldsymbol {x}) = - \frac {\pi^ {2}}{4} \sum_ {i = 1} ^ {D} \sin \left(\frac {\pi}{2} x _ {i}\right) \tag {40}
$$

Detailed performance comparison of different models is listed in the tables below.

Table 6. Case 1: domain size $x \in [0,1]^D$ (each case is repeated 10 times to obtain the statistics) 

<table><tr><td>Model</td><td># Dim. (D)</td><td># colloc. points</td><td># grid pts in each dim</td><td># modes</td><td># model params.</td><td>Mean rel. L2 norm error w/ std.</td><td>Mean GPU wall time w/ std. (s)</td><td>Mean GPU VRAM usage w/ std. (MB)</td></tr><tr><td rowspan="3">INN-TD(4 iterations)</td><td>2</td><td>-</td><td>32</td><td>4</td><td>256</td><td> $1.754 \times 10^{-8} \pm 1 \times 10^{-12}$ </td><td> $0.81 \pm 0.10$ </td><td> $760 \pm 0$ </td></tr><tr><td>5</td><td>-</td><td>32</td><td>10</td><td>1,600</td><td> $1.659 \times 10^{-8} \pm 1.2 \times 10^{-13}$ </td><td> $6.88 \pm 0.03$ </td><td> $760 \pm 0$ </td></tr><tr><td>10</td><td>-</td><td>32</td><td>10</td><td>6,400</td><td> $1.238 \times 10^{-8} \pm 1.6 \times 10^{-12}$ </td><td> $57.43 \pm 0.41$ </td><td> $760 \pm 0$ </td></tr><tr><td rowspan="3">CP-PINN(80 K epochs)</td><td>2</td><td> $32^2$ </td><td>-</td><td>4</td><td>96</td><td> $4.63 \times 10^{-3} \pm 7.14 \times 10^{-4}$ </td><td> $60.68 \pm 0.53$ </td><td> $880 \pm 0$ </td></tr><tr><td>2</td><td> $128^2$ </td><td>-</td><td>4</td><td>96</td><td> $2.64 \times 10^{-3} \pm 3.54 \times 10^{-4}$ </td><td> $169.2 \pm 4.83$ </td><td> $882 \pm 0$ </td></tr><tr><td>5</td><td> $32^5$ </td><td>-</td><td>10</td><td>1,200</td><td> $5.90 \times 10^{-3} \pm 6.91 \times 10^{-4}$ </td><td> $2,130 \pm 18.6$ </td><td> $3018 \pm 0$ </td></tr><tr><td rowspan="3">KAN(50 steps)</td><td>2</td><td> $32^2$ </td><td>5</td><td>-</td><td>348</td><td> $9.44 \times 10^{-5} \pm 8.37 \times 10^{-6}$ </td><td> $56.16 \pm 0.58$ </td><td> $866 \pm 0$ </td></tr><tr><td>2</td><td> $128^2$ </td><td>5</td><td>-</td><td>348</td><td> $4.84 \times 10^{-5} \pm 8.95 \times 10^{-6}$ </td><td> $58.88 \pm 0.58$ </td><td> $1,160 \pm 0$ </td></tr><tr><td>5</td><td> $12^5$ </td><td>5</td><td>-</td><td>624</td><td> $1.68 \times 10^{-4} \pm 2.45 \times 10^{-5}$ </td><td> $842.1 \pm 3.94$ </td><td> $9,988 \pm 0$ </td></tr></table>

Table 7. Case 1: domain size $x \in [0,12]^D$ (each case is repeated 10 times to obtain the statistics) 

<table><tr><td>Model</td><td># Dim. (D)</td><td># colloc. points</td><td># grid pts in each dim</td><td># modes</td><td># model params.</td><td>Mean rel. L2 norm error w/ std.</td><td>Mean GPU wall time w/ std. (s)</td><td>Mean GPU VRAM usage w/ std. (MB)</td></tr><tr><td rowspan="3">INN-TD(4 iterations)</td><td>2</td><td>-</td><td>32</td><td>4</td><td>256</td><td> $4.77 \times 10^{-4} \pm 3.27 \times 10^{-8}$ </td><td> $0.22 \pm 0.006$ </td><td> $760 \pm 0$ </td></tr><tr><td>5</td><td>-</td><td>32</td><td>10</td><td>1,600</td><td> $3.97 \times 10^{-4} \pm 4.62 \times 10^{-8}$ </td><td> $7.04 \pm 0.04$ </td><td> $760 \pm 0$ </td></tr><tr><td>10</td><td>-</td><td>32</td><td>10</td><td>6,400</td><td> $3.71 \times 10^{-4} \pm 2.69 \times 10^{-9}$ </td><td> $59.75 \pm 0.39$ </td><td> $760 \pm 0$ </td></tr><tr><td rowspan="3">CP-PINN(80 K epochs)</td><td>2</td><td> $32^2$ </td><td>-</td><td>4</td><td>216</td><td> $0.025 \pm 0.004$ </td><td> $87.6 \pm 6.53$ </td><td> $882 \pm 0$ </td></tr><tr><td>2</td><td> $1,024^2$ </td><td>-</td><td>4</td><td>216</td><td> $0.021 \pm 0.012$ </td><td> $261.6 \pm 7.79$ </td><td> $1,014 \pm 0$ </td></tr><tr><td>2</td><td> $2,048^2$ </td><td>-</td><td>4</td><td>296</td><td> $0.032 \pm 0.007$ </td><td> $292 \pm 10.3$ </td><td> $1,014 \pm 0$ </td></tr><tr><td rowspan="3">KAN(50 steps)</td><td>2</td><td> $32^2$ </td><td>5</td><td>-</td><td>348</td><td> $0.78 \pm 0.40$ </td><td> $60.68 \pm 0.53$ </td><td> $866 \pm 0$ </td></tr><tr><td>2</td><td> $1,024^2$ </td><td>5</td><td>-</td><td>348</td><td> $0.80 \pm 0.27$ </td><td> $72.6 \pm 1.19$ </td><td> $19,948 \pm 0$ </td></tr><tr><td>2</td><td> $128^2$ </td><td>10</td><td>-</td><td>3,188</td><td> $0.28 \pm 0.08$ </td><td> $119.7 \pm 2.05$ </td><td> $2,486 \pm 0$ </td></tr></table>

We also investigated the performance of four different models for the following analytical solution as in Case 2.

$$
u (\boldsymbol {x}) = \prod_ {d = 1} ^ {D} \left(\sin (\pi x _ {d})\right) \tag {41}
$$

For 2D cases, the analytical solutions are shown in Fig. 11.

![](images/7464ef4209e63c816d998c8741f4007bb941a4b2ec69d9c1c346d10930402a5e.jpg)

<details>
<summary>surface_3d</summary>

| x1    | x2    | u     |
|-------|-------|-------|
| 0.0   | 0.0   | 0.0   |
| 0.2   | 0.2   | 0.2   |
| 0.4   | 0.4   | 0.4   |
| 0.6   | 0.6   | 0.6   |
| 0.8   | 0.8   | 0.8   |
| 1.0   | 1.0   | 1.0   |
</details>

(a) Domain size $x \in [0, 1]^{2}$

![](images/90337206b4d92f66168b3c038b2741cc11744f4aa77a9b0d36e5375166580977.jpg)

<details>
<summary>area_stacked</summary>

| x1 | x2 | u    |
|----|----|------|
| 0  | 0  | -1.00|
| 2  | 2  | -0.75|
| 4  | 4  | -0.50|
| 6  | 6  | -0.25|
| 8  | 8  | 0.00 |
| 10 | 10 | 0.25 |
| 12 | 12 | 0.50 |
| 0  | 0  | 1.00 |
| 2  | 2  | 0.75 |
| 4  | 4  | 0.50 |
| 6  | 6  | 0.25 |
| 8  | 8  | -0.00|
| 10 | 10 | -0.25|
| 12 | 12 | -0.50|
</details>

(b) Domain size $\pmb{x} \in [0, 12]^2$   
Figure 11. Analytical solution for 2D cases.

The corresponding RHS can be written as:

$$
f (\boldsymbol {x}) = - D \pi^ {2} \prod_ {d = 1} ^ {D} \left(\sin \left(\pi x _ {d}\right)\right) \tag {42}
$$

Similarly, we studied 2 subcases with different domain sizes. The detailed comparison is shown in the table below.

Table 8. Case 2: domain size $x \in [0,1]^D$ (each case is repeated 10 times to obtain the statistics) 

<table><tr><td>Model</td><td># Dim. (D)</td><td># colloc. points</td><td># grid pts in each dim</td><td># modes</td><td># model params.</td><td>Mean rel. L2 norm error w/ std.</td><td>Mean GPU wall time w/ std. (s)</td><td>Mean GPU VRAM usage w/ std. (MB)</td></tr><tr><td rowspan="3">INN-TD(4 iterations)</td><td>2</td><td>-</td><td>32</td><td>1</td><td>64</td><td> $5.03 \times 10^{-7} \pm 1.31 \times 10^{-14}$ </td><td> $0.81 \pm 0.10$ </td><td> $760 \pm 0$ </td></tr><tr><td>5</td><td>-</td><td>32</td><td>1</td><td>160</td><td> $1.23 \times 10^{-6} \pm 8.65 \times 10^{-14}$ </td><td> $1.01 \pm 0.02$ </td><td> $760 \pm 0$ </td></tr><tr><td>10</td><td>-</td><td>32</td><td>1</td><td>320</td><td> $2.42 \times 10^{-6} \pm 8.1 \times 10^{-14}$ </td><td> $3.62 \pm 0.06$ </td><td> $764 \pm 0$ </td></tr><tr><td rowspan="3">CP-PINN(80 K epochs)</td><td>2</td><td> $32^2$ </td><td>-</td><td>4</td><td>96</td><td> $4.13 \times 10^{-3} \pm 1.00 \times 10^{-3}$ </td><td> $171.68 \pm 15.3$ </td><td> $880 \pm 0$ </td></tr><tr><td>2</td><td> $128^2$ </td><td>-</td><td>4</td><td>96</td><td> $5.17 \times 10^{-3} \pm 1.38 \times 10^{-4}$ </td><td> $169.2 \pm 14.83$ </td><td> $882 \pm 0$ </td></tr><tr><td>5</td><td> $32^5$ </td><td>-</td><td>10</td><td>1,200</td><td> $8.40 \times 10^{-3} \pm 1.45 \times 10^{-3}$ </td><td> $2,102 \pm 73.8$ </td><td> $3,018 \pm 0$ </td></tr><tr><td rowspan="3">KAN(50 steps)</td><td>2</td><td> $32^2$ </td><td>5</td><td>-</td><td>348</td><td> $8.03 \times 10^{-4} \pm 6.90 \times 10^{-5}$ </td><td> $57.5 \pm 0.48$ </td><td> $866 \pm 0$ </td></tr><tr><td>2</td><td> $128^2$ </td><td>5</td><td>-</td><td>348</td><td> $9.13 \times 10^{-4} \pm 1.90 \times 10^{-4}$ </td><td> $61.0 \pm 0.35$ </td><td> $1,160 \pm 0$ </td></tr><tr><td>5</td><td> $12^5$ </td><td>5</td><td>-</td><td>624</td><td> $0.59 \pm 0.0054$ </td><td> $872.8 \pm 3.71$ </td><td> $9,988 \pm 0$ </td></tr></table>

Here we investigate the convergence of INN-TD for the 2D case with domain size $\Omega \in [0,12]^2$ . The details are shown in Fig. 12. INN-TD shows a convergence property where a larger number of model parameters leads to better accuracy. Moreover, the convergence rate can be controlled by adopting different combinations of hyperparameters $s$ and $p$ .

# E.2.2. SOLVING THE HELMHOLTZ EQUATION

In this example, INN-TD is used to solve the Helmholtz equation as shown below. In (Wang et al., 2021), it is shown that the vanilla PINN exhibits failure modes when solving this equation. Here we use INN-TD to solve the exact same problem.

$$
\left(\frac {\partial^ {2}}{\partial x _ {1} ^ {2}} + \frac {\partial^ {2}}{\partial x _ {2} ^ {2}}\right) u + u (x, y) = q (x, y) \tag {43}
$$

where the forcing function is defined as:

$$
q (x, y) = - \left(a _ {1} \pi\right) ^ {2} \sin \left(a _ {1} \pi x\right) \sin \left(a _ {2} \pi y\right) - \left(a _ {2} \pi\right) ^ {2} \sin \left(a _ {1} \pi x\right) \sin \left(a _ {2} \pi y\right) + k ^ {2} \sin \left(a _ {1} \pi x\right) \sin \left(a _ {2} \pi y\right) \tag {44}
$$

Table 9. Case 2: domain size $x \in [0,12]^D$ (each case is repeated 10 times to obtain the statistics) 

<table><tr><td>Model</td><td># Dim. (D)</td><td># colloc. points</td><td># grid pts in each dim</td><td># modes</td><td># model params.</td><td>Mean rel. L2 norm error w/ std.</td><td>Mean GPU wall time w/ std. (s)</td><td>Mean GPU VRAM usage w/ std. (MB)</td></tr><tr><td rowspan="4">INN-TD(4 iterations)</td><td>2</td><td>-</td><td>32</td><td>1</td><td>64</td><td> $1.56 \times 10^{-2} \pm 6.17 \times 10^{-14}$ </td><td> $0.77 \pm 0.30$ </td><td> $760 \pm 0$ </td></tr><tr><td>2</td><td>-</td><td>256</td><td>1</td><td>512</td><td> $2.45 \times 10^{-6} \pm 1.37 \times 10^{-14}$ </td><td> $0.82 \pm 0.04$ </td><td> $760 \pm 0$ </td></tr><tr><td>5</td><td>-</td><td>32</td><td>1</td><td>160</td><td> $6.33 \times 10^{-6} \pm 1.36 \times 10^{-13}$ </td><td> $1.56 \pm 0.01$ </td><td> $760 \pm 0$ </td></tr><tr><td>10</td><td>-</td><td>32</td><td>1</td><td>320</td><td> $1.33 \times 10^{-7} \pm 3.65 \times 10^{-14}$ </td><td> $3.82 \pm 0.02$ </td><td> $764 \pm 0$ </td></tr><tr><td rowspan="3">CP-PINN(80 K epochs)</td><td>2</td><td> $32^2$ </td><td>-</td><td>4</td><td>216</td><td> $1.45 \pm 0.21$ </td><td> $102.1 \pm 12.3$ </td><td> $882 \pm 0$ </td></tr><tr><td>2</td><td> $1,024^2$ </td><td>-</td><td>4</td><td>216</td><td> $0.63 \pm 0.13$ </td><td> $315.5 \pm 10.7$ </td><td> $1,014 \pm 0$ </td></tr><tr><td>2</td><td> $2,048^2$ </td><td>-</td><td>4</td><td>296</td><td> $0.60 \pm 0.26$ </td><td> $1,378 \pm 30.7$ </td><td> $1,014 \pm 0$ </td></tr><tr><td rowspan="3">KAN(50 steps)</td><td>2</td><td> $32^2$ </td><td>5</td><td>-</td><td>348</td><td> $1.00 \pm 1.96 \times 10^{-5}$ </td><td> $38.9 \pm 3.05$ </td><td> $866 \pm 0$ </td></tr><tr><td>2</td><td> $128^2$ </td><td>10</td><td>-</td><td>5,498</td><td> $0.89 \pm 0.10$ </td><td> $191 \pm 4.81$ </td><td> $3,368 \pm 0$ </td></tr><tr><td>2</td><td> $128^2$ </td><td>20</td><td>-</td><td>28,578</td><td> $0.53 \pm 0.03$ </td><td> $369.6 \pm 2.06$ </td><td> $11,668 \pm 0$ </td></tr></table>

![](images/251965ab466bb6e1c839278a53893e8771d3bf46f2f5ddb7293fc6e5f2e2881d.jpg)

<details>
<summary>line</summary>

| Number of model parameters | s=1, p=1 | s=4, p=1 | s=2, p=2 | s=3, p=1 |
| --------------------------- | -------- | -------- | -------- | -------- |
| 6 × 10²                     | 10⁻³     | 10⁻⁵     | 10⁻⁵     | 10⁻⁴     |
| 10³                         | 10⁻⁴     | 10⁻⁶     | 10⁻⁶     | 10⁻⁵     |
| 2 × 10³                     | 10⁻⁵     | 10⁻⁷     | 10⁻⁷     | 10⁻⁶     |
</details>

![](images/9ffa57e6c47f91292a530bec6fba9d37b5f0008208b18612cd4146a5f23abc41.jpg)

<details>
<summary>line</summary>

| Number of model parameters | s=1, p=1 | s=4, p=1 | s=2, p=2 | s=3, p=1 |
| -------------------------- | -------- | -------- | -------- | -------- |
| 600                        | 0.25     | 0.28     | 0.27     | 0.26     |
| 1000                       | 0.28     | 0.32     | 0.30     | 0.29     |
| 2000                       | 0.32     | 0.42     | 0.35     | 0.38     |
</details>

![](images/8dc6075f77d478f6cc6a72c9f00d8ffacba16b1ae07a892c6921e6f6b001e1a1.jpg)

<details>
<summary>line</summary>

| Number of model parameters | s=1, p=1 | s=4, p=1 | s=2, p=2 | s=3, p=1 |
| --------------------------- | -------- | -------- | -------- | -------- |
| 6 × 10²                     | 7.65     | 7.70     | 7.75     | 7.75     |
| 10³                         | 7.90     | 8.05     | 7.90     | 7.90     |
| 2 × 10³                     | 8.25     | 8.30     | 8.90     | 8.90     |
</details>

Figure 12. Solving the 2D Poisson Eq. using INN-TD with different number of model parameters and hyperparameters s and p. a=20 is used for all cases. The statistics are obtained by repeating each case for 10 times and the shaded area represents 1 standard deviation (a) convergence of INN-TD using different s and p (b) statistics of total computational time (c) statistics of GPU VRAM usage

The solution domain is defined as: $\Omega\in[-1,1]\times[-1,1]$ . The analytical solution to this problem is:

$$
u \left(x _ {1}, x _ {2}\right) = \sin \left(a _ {1} \pi x\right) \sin \left(a _ {2} \pi y\right) \tag {45}
$$

The comparison of INN-TD solution and the analytical solution is shown in Fig. 13. It can be seen that INN-TD achieves high accuracy without any failure modes. Furthermore, we study the convergence of INN-TD with different hyperparameters. The result is shown in Fig. 14. As expected, INN-TD shows the convergence property where a larger number of model parameters leads to higher accuracy. Higher s and p can also decrease the error.

# E.2.3. SPACE-TIME PDE EXAMPLE

In this example, we show the convergence property of INN-TD data-free solver for a space-time problem. The governing PDE is shown below:

$$
\dot {u} (\boldsymbol {x}, t) + k \Delta u (\boldsymbol {x}, t) = b (\boldsymbol {x}, t) \quad \text { in } \quad \Omega_ {\boldsymbol {x}} \otimes \Omega_ {t} \tag {46}
$$

where $k$ is heat conductivity. $b(\pmb {x},t)$ is defined as:

$$
b (\boldsymbol {x}, t) = \exp \left(- \frac {2 \left((x - x _ {0} (t)) ^ {2} + (y - y _ {0} (t)) ^ {2}\right)}{r ^ {2}}\right) \tag {47}
$$

where r is the standard deviation that characterizes the width of the heat source; $[x_{0}(t), y_{0}(t)]$ is the heat source center. This equation models the heat transfer with a moving heat source. The 4D space-time continuum is adopted as: $\Omega_{x} = [-10, 10]^{3}$ ; $\Omega_{t} = (0, 0.1]$ .

![](images/f9bae1c09067f4b9cf5ab79dcc4062dbc5bcdd30400ad97d278c3e4519c8e335.jpg)

<details>
<summary>heatmap</summary>

| X1    | X2    | Value  |
|-------|-------|--------|
| -0.5  | 0.75  | 0.88   |
| -0.5  | 0.50  | 0.66   |
| -0.5  | 0.25  | 0.44   |
| -0.5  | 0.00  | 0.22   |
| -0.5  | -0.25 | 0.00   |
| -0.5  | -0.50 | -0.22  |
| -0.5  | -0.75 | -0.44  |
| 0.0   | 0.75  | 0.88   |
| 0.0   | 0.50  | 0.66   |
| 0.0   | 0.25  | 0.44   |
| 0.0   | 0.00  | 0.22   |
| 0.0   | -0.25 | 0.00   |
| 0.0   | -0.50 | -0.22  |
| 0.0   | -0.75 | -0.44  |
| 0.5   | 0.75  | 0.88   |
| 0.5   | 0.50  | 0.66   |
| 0.5   | 0.25  | 0.44   |
| 0.5   | 0.00  | 0.22   |
| 0.5   | -0.25 | 0.00   |
| 0.5   | -0.50 | -0.22  |
| 0.5   | -0.75 | -0.44  |
</details>

(a)

![](images/9032c41849ba6e931b0ffec1b88234adb1def7801e16f25acb938a6aef7754a8.jpg)

<details>
<summary>heatmap</summary>

| X1    | X2    | Value  |
|-------|-------|--------|
| -0.5  | 0.75  | 0.88   |
| -0.5  | 0.50  | 0.66   |
| -0.5  | 0.25  | 0.44   |
| -0.5  | 0.00  | 0.22   |
| -0.5  | -0.25 | 0.00   |
| -0.5  | -0.50 | -0.22  |
| -0.5  | -0.75 | -0.44  |
| 0.0   | 0.75  | 0.88   |
| 0.0   | 0.50  | 0.66   |
| 0.0   | 0.25  | 0.44   |
| 0.0   | 0.00  | 0.22   |
| 0.0   | -0.25 | 0.00   |
| 0.0   | -0.50 | -0.22  |
| 0.0   | -0.75 | -0.44  |
| 0.5   | 0.75  | 0.88   |
| 0.5   | 0.50  | 0.66   |
| 0.5   | 0.25  | 0.44   |
| 0.5   | 0.00  | 0.22   |
| 0.5   | -0.25 | 0.00   |
| 0.5   | -0.50 | -0.22  |
| 0.5   | -0.75 | -0.44  |
</details>

(b)

![](images/c0a42d831af6000fae220d70e165548851f975e5d7b9d266f9f6bd4bc40671e6.jpg)

<details>
<summary>heatmap</summary>

| X1    | X2    | Absolute error |
|-------|-------|----------------|
| -0.5  | 0.75  | 2.88           |
| -0.5  | 0.50  | 2.56           |
| -0.5  | 0.25  | 2.24           |
| -0.5  | 0.00  | 1.92           |
| -0.5  | -0.25 | 1.60           |
| -0.5  | -0.50 | 1.28           |
| -0.5  | -0.75 | 0.96           |
| 0.0   | 0.75  | 2.88           |
| 0.0   | 0.50  | 2.56           |
| 0.0   | 0.25  | 2.24           |
| 0.0   | 0.00  | 1.92           |
| 0.0   | -0.25 | 1.60           |
| 0.0   | -0.50 | 1.28           |
| 0.0   | -0.75 | 0.96           |
| 0.5   | 0.75  | 2.88           |
| 0.5   | 0.50  | 2.56           |
| 0.5   | 0.25  | 2.24           |
| 0.5   | 0.00  | 1.92           |
| 0.5   | -0.25 | 1.60           |
| 0.5   | -0.50 | 1.28           |
| 0.5   | -0.75 | 0.96           |
</details>

(c)

Figure 13. Solving the 2D Helmholtz Eq. using INN-TD (s = p = 1, a = 20) with 2 modes and 250 elements along each direction. The total solution time is 0.48 sec (a) INN-TD solution (b) exact solution (c) point-wise absolute error   
![](images/1d5937a734baa8a76e9f4c1cb714703f7ac6b63cdee42fb44979928d8dac5c00.jpg)  
(a)

![](images/f6d1d8b567c884395a3bfe28d59e66c525a2fea757fb8e23fecacd4eb639fbe9.jpg)

<details>
<summary>line</summary>

| Number of model parameters | s=1, p=1 | s=4, p=1 | s=2, p=2 | s=3, p=3 |
| -------------------------- | -------- | -------- | -------- | -------- |
| 4 × 10²                    | 0.52     | 0.54     | 0.54     | 0.54     |
| 6 × 10²                    | 0.58     | 0.58     | 0.58     | 0.59     |
| 10³                        | 0.60     | 0.60     | 0.60     | 0.61     |
</details>

(b)

![](images/77a05ba5f64648cd6eb8297ada8ddd405ee803853cc3db02b9b5855d20eece4e.jpg)

<details>
<summary>line</summary>

| Number of model parameters | s=1, p=1 | s=4, p=1 | s=2, p=2 | s=3, p=3 |
| -------------------------- | -------- | -------- | -------- | -------- |
| 4 × 10²                    | 815      | 825      | 828      | 828      |
| 6 × 10²                    | 835      | 830      | 860      | 860      |
| 10³                        | 920      | 930      | 935      | 935      |
</details>

(c)   
Figure 14. Solving the 2D Helmholtz Eq. using INN-TD with different number of model parameters and hyperparameters s and p. a = 20 is used for all cases. The statistics are obtained by repeating each case for 10 times and the shaded area represents 1 standard deviation (a) convergence of INN-TD using different s and p (b) statistics of total computational time (c) statistics of GPU VRAM usage

To investigate the convergence of INN-TD for space-time problems, we use the following forcing function:

$$
\begin{array}{l} b (\pmb {x}, t) = 2 (1 - 2 y ^ {2}) (1 - e ^ {- 1 5 t}) e ^ {- y ^ {2} - (1 0 0 t - x - 5) ^ {2}} \\ + 2 (1 - 2 (1 0 0 t - x - 5) ^ {2}) \left(1 - e ^ {- 1 5 t}\right) e ^ {- y ^ {2} - (1 0 0 t - x - 5) ^ {2}} + (1 \tag {48} \\ - e ^ {- 1 5 t}) (2 0 0 x + 1 0 0 0 - 2 0 0 0 0 t) e ^ {- y ^ {2} - (1 0 0 t - x - 5) ^ {2}} \\ - 1 5 e ^ {- 1 5 t} e ^ {- y ^ {2} - (1 0 0 t - x - 5) ^ {2}} \\ \end{array}
$$

As a result, the exact solution can be written as:

$$
u ^ {\text { exact }} (\boldsymbol {x}, t) = (1 - e ^ {- 1 5 t}) e ^ {- y ^ {2} - (x - 1 0 0 t - 5) ^ {2}} \tag {49}
$$

A sample snapshot of the solution is shown in Fig. 15 (a).

Since it's straightforward for numerical integration of C-HiDeNN interpolation function, here we define the relative L2

![](images/d88a4bfb9e94a23c7f7eab8fc80383c91fa64c3ff31f88bb9d73c1748c246dd3.jpg)

<details>
<summary>heatmap</summary>

| Solution | Value     |
| -------- | --------- |
| 7.7e-01  | 0.7       |
| 0.65     | 0.6       |
| 0.6      | 0.55      |
| 0.55     | 0.5       |
| 0.5      | 0.45      |
| 0.45     | 0.4       |
| 0.4      | 0.35      |
| 0.35     | 0.3       |
| 0.3      | 0.25      |
| 0.25     | 0.2       |
| 0.2      | 0.15      |
| 0.15     | 0.1       |
| 0.1      | 0.0e+00   |
</details>

(a)

![](images/0586f88346b9ec40db9b435429159fe19662555c950cdbea54d53ca61d696f88.jpg)

<details>
<summary>line</summary>

| # model parameters | s=p=1 | s=p=2 | s=p=3 |
| ------------------ | ----- | ----- | ----- |
| 10^3               | 0.5   | 0.45  | 0.4   |
| 10^3.5             | 0.2   | 0.15  | 0.1   |
| 10^4               | 0.05  | 0.03  | 0.02  |
| 10^4.5             | 0.01  | 0.008 | 0.005 |
| 10^5               | 0.003 | 0.002 | 0.001 |
| 10^5.5             | 0.001 | 0.0008| 0.0005|
| 10^6               | 0.0005| 0.0003| 0.0002|
| 10^6.5             | 0.0002| 0.0001| 0.0001|
</details>

(b)   
Figure 15. Space-time PDE: (a) snapshot of the space-time solution (b) convergence for the 4D space-time problem: relationship between number of model parameters and relative L2 norm error

norm error in the integral form:

$$
\frac {\left(\int \left(u ^ {e x a c t} (\boldsymbol {x} , t) - \mathcal {J} u (\boldsymbol {x} , t)\right) ^ {2} \mathrm{d} \boldsymbol {x} d t\right) ^ {\frac {1}{2}}}{\left(\int \left(u ^ {e x a c t} (\boldsymbol {x})\right) ^ {2} \mathrm{d} \boldsymbol {x} d t\right) ^ {\frac {1}{2}}} \tag {50}
$$

We investigated the convergence properties of INN-TD using different hyperparameters s and p in C-HiDeNN interpolation. As shown in Fig. 15 (b), the error of INN-TD decreases as more model parameters are utilized. This guaranteed convergence distinguishes INN-TD from other black-box deep learning models, where an increase in model parameters does not necessarily result in reduced error. Additionally, the convergence rate can be controlled by adjusting s and p, providing flexibility in designing the INN-TD according to the accuracy requirement.

# E.2.4. SPACE-PARAMETER-TIME (S-P-T) PDE EXAMPLE

To solve the 6D S-P-T problem, we used 101 grid points to discretize each input dimension of $(x,y,z,k,P,t)$ . 100 modes were used for INN-TD approximation. Therefore, the INN-TD interpolation function can be written as:

$$
\mathcal {J} u (\boldsymbol {x}) = \sum_ {m = 1} ^ {1 0 0} \widetilde {\boldsymbol {N}} _ {x} (x) \boldsymbol {u} _ {x} ^ {(m)} \cdot \widetilde {\boldsymbol {N}} _ {y} (y) \boldsymbol {u} _ {y} ^ {(m)} \cdot \widetilde {\boldsymbol {N}} _ {z} (z) \boldsymbol {u} _ {z} ^ {(m)} \cdot \widetilde {\boldsymbol {N}} _ {k} (k) \boldsymbol {u} _ {k} ^ {(m)} \cdot \widetilde {\boldsymbol {N}} _ {P} (P) \boldsymbol {u} _ {P} ^ {(m)} \cdot \widetilde {\boldsymbol {N}} _ {t} (t) \boldsymbol {u} _ {t} ^ {(m)} \tag {51}
$$

Algorithm 3 was used to solve the S-P-T problem, and it took in total 155 s to run the solver on a single NVIDIA RTX A6000 GPU.

Since the analytical solution is not available for this problem, we used the finite element method (FEM) to generate the simulation data and treated it as the ground truth. For ease of validation, the same spatial mesh size $(100 \times 100 \times 100)$ and 100 time steps were adopted in FEM simulation. We repeated running FEM simulations with 10 randomly selected parametric input pairs $(k, P)$ . The relative L2 norm error is computed as follows:

$$
\epsilon = \left[ \sum_ {i, j, a, b, c, d} \left[ \mathcal {J} u (x _ {i}, y _ {j}, z _ {a}, k _ {b}, P _ {c}, t _ {d}) - u ^ {F E M} (x _ {i}, y _ {j}, z _ {a}, t _ {d}; k _ {b}, P _ {c}) ^ {2} \right] ^ {\frac {1}{2}} / \left[ \sum_ {i, j, a, b, c, d} u ^ {F E M} (x _ {i}, y _ {j}, z _ {a}, t _ {d}; k _ {b}, P _ {c}) ^ {2} \right] ^ {\frac {1}{2}} \right. \tag {52}
$$

where $i, j, a, b, c, d$ refer to the data index of each input dimension.

# E.2.5. INVERSE OPTIMIZATION PROBLEM

In this problem, we solved the S-P-T problem using INN-TD data-free solver. As a result, we obtained the S-P-T interpolation function as shown in Eq. 51.

To test the effectiveness of INN-TD for the inverse optimization problem, we randomly sampled 100 different parametric input pairs $(k_{i}, P_{i}), i = 1,..,100$ , and generated the corresponding ground truth through high-fidelity FEM simulation. Then we used Adam optimizer to optimize the loss function as shown in Eq. 13. The learning rate was chosen as 0.1. To ensure the parameters $(k_{i}, P_{i})$ are within the predefined range, we added a box constraint on the parameter such that:

$$
k _ {m i n} \leq k _ {i} <   k _ {m a x}
$$

$$
P _ {m i n} \leq P _ {i} <   P _ {m a x}
$$

For each optimization case, we used uniform distributions for the initial guess. The mean and standard deviation for the relative L2 norm error and parameter error were then computed across all cases.

# E.2.6. VECTOR SOLUTION FIELD: ELASTICITY PROBLEM

In this example, INN-TD is used to solve the 3D elasticity problem in solid mechanics where the solution field is a vector field. The solution $\boldsymbol{u} = [u_1(x,y,z), u_2(x,y,z), u_3(x,y,z)]$ represents the displacement along the $x, y$ and $z$ directions. The governing equation is shown below:

$$
\mu u _ {i, j j} + (\mu + \lambda) u _ {j, j i} + b _ {i} = 0 \tag {53}
$$

where Einstein summation is used in the index notation, i = 1, 2, 3 and j = 1, 2, 3; $\lambda, \mu$ are the Lamé constants. The solution domain is defined as $\Omega = [0, 1] \times [0, 1] \times [0, 2]$ and homogeneous boundary condition is assumed. The forcing function $b_{i}$ is defined as:

$$
\left\{ \begin{array}{l l} b _ {1} & = 9 2 2 7 6 5 1 9 6. 7 0 6 3 8 2 \cdot n _ {2} ^ {2} x (x - 1) \sin \left(n _ {1} \pi y\right) \sin \left(\frac {n _ {1} \pi z}{2}\right) - 5 8 7 4 5 0 5 3. 1 0 9 7 4 6 \cdot n _ {2} \cos \left(n _ {2} \pi x\right) (2 y - 1) \sin \left(\frac {n _ {2} \pi z}{2}\right) \\ & - 5 8 7 4 5 0 5 6. 1 0 9 7 4 6 \cdot n _ {3} \cos \left(n _ {3} \pi y\right) \sin \left(n _ {3} \pi y\right) (z - 1) - 5 2 3 5 7 5 7 0 1. 2 6 9 7 8 6 \cdot \sin \left(n _ {1} \pi y\right) \sin \left(\frac {n _ {1} \pi z}{2}\right) \\ b _ {2} & = 9 2 2 7 6 5 1 8 6. 7 0 6 3 8 2 \cdot n _ {2} ^ {2} \sin \left(n _ {2} \pi x\right) y (y - 1) \sin \left(\frac {n _ {2} \pi z}{2}\right) - 5 8 7 4 5 0 5 6 3. 1 0 9 7 4 6 \cdot n _ {1} (2 x - 1) \cos \left(n _ {1} \pi y\right) \sin \left(\frac {n _ {1} \pi z}{2}\right) \\ & - 5 8 7 4 5 0 5 6 3. 1 0 9 7 4 6 \cdot n _ {3} \sin \left(n _ {3} \pi x\right) \cos \left(n _ {3} \pi y\right) (z - 1) - 5 2 3 5 7 5 7 0 1. 2 6 9 7 8 8 \cdot \sin \left(n _ {2} \pi x\right) \sin \left(\frac {n _ {2} \pi z}{2}\right) \\ b _ {3} & = 7 3 8 2 1 2 1 4 9. 3 6 5 1 0 \cdot n _ {3} ^ {2} \sin \left(n _ {3} \pi x\right) \sin \left(n _ {3} \pi y\right) z (z - 2) - 2 9 3 7 2 5 2 8 1. 5 5 4 8 7 3 \cdot n _ {2} \sin \left(n _ {2} \pi x\right) (2 y - 1) \cos \left(\frac {n _ {2} \pi z}{2}\right) \\ & - 2 9 3 7 2 5 2 1 8 1 5 5 4 8 7 3 \cdot n _ {1} (2 x - 1) \sin \left(n _ {1} \pi y\right) \cos \left(\frac {n _ {1} \pi z}{2}\right) - 2 6 1 7 8 7 8 5 0. 6 3 4 9 9 4 \cdot \sin \left(n _ {3} \pi x\right) \sin \left(n _ {3} \pi y\right) \end{array} \right. \tag {54}
$$

The analytical solution to the problem is:

$$
\left\{ \begin{array}{l} u _ {1} (x, y, z) = 0. 0 0 0 9 7 2 3 5 4 8 7 3 7 8 6 7 4 9 \cdot \left(x ^ {2} - x\right) \sin \left(n _ {1} \pi y\right) \sin \left(\frac {n _ {1} \pi z}{2}\right) \\ u _ {2} (x, y, z) = 0. 0 0 0 9 7 2 3 5 4 8 7 3 7 8 6 7 4 9 \cdot \left(y ^ {2} - y\right) \sin \left(n _ {2} \pi x\right) \sin \left(\frac {n _ {2} \pi z}{2}\right) \\ u _ {3} (x, y, z) = 0. 0 0 0 9 7 2 3 5 4 8 7 3 7 8 6 7 4 9 \cdot \sin \left(n _ {3} \pi x\right) \sin \left(n _ {3} \pi y\right) \left(\frac {z ^ {2}}{2} - z\right) \end{array} \right. \tag {55}
$$

The convergence of INN-TD is shown in Fig. 16.

# E.2.7. OPERATOR SOLVING

INN-TD can also be extended to approximate PDE operators. As an example, we show that INN-TD can be leveraged to approximate the PDE operator for the following equation:

$$
\frac {\partial u}{\partial t} - \frac {\partial}{\partial x} k (x) \frac {\partial u}{\partial x} = f (x) \tag {56}
$$

with homogeneous initial and boundary conditions. Here we aim to approximate the PDE operator from the conductivity field $k(x)$ to the space-time PDE solution $u(x,t)$ .

![](images/515d6d91ed75e1fa9e424cf2466f6e0a7782beebaaa45570e9ed830b445dfb3e.jpg)

<details>
<summary>line</summary>

| Mesh Size (h) | FEM Energy Norm Error | INN-TD s1, p1 | INN-TD s2, p1 | INN-TD s4, p1 | INN-TD s6, p1 |
| ------------- | --------------------- | ------------- | ------------- | ------------- | ------------- |
| 1.E-06        | ~1.E-05               | -             | -             | -             | -             |
| 1.E-05        | ~1.E-05               | ~1.E-04       | -             | -             | -             |
| 1.E-04        | ~1.E-04               | ~1.E-03       | ~1.E-03       | -             | -             |
| 1.E-03        | ~1.E-03               | ~1.E-02       | ~1.E-02       | ~1.E-03       | ~1.E-04       |
| 1.E-02        | ~1.E-02               | ~1.E-01       | ~1.E-01       | ~1.E-02       | ~1.E-03       |
| 1.E-01        | ~1.E-01               | ~1.E-00       | ~1.E-00       | ~1.E-01       | ~1.E-02       |
</details>

Figure 16. Convergence for the elasticity problem: relationship between mesh size and energy norm error

To account for the arbitrariness of $k(x)$ , we discretize $x$ and assume the nodal values of $k(x)$ are subjected to the following covariance function:

$$
C (x _ {i}, x _ {j}) = \sigma^ {2} \exp \left(- \frac {(x _ {i} - x _ {j}) ^ {2}}{2 l ^ {2}}\right) \tag {57}
$$

where hyperparameters $\sigma$ and l refer to standard deviation and length scale, respectively. The covariance matrix can be further decomposed using eigen-decomposition and arbitrary $k(x)$ is approximated using the following equation:

$$
k (x, \zeta) = k _ {\mu} + \sum_ {I = 1} ^ {n _ {x}} \widetilde {N} _ {I} (x) \sum_ {J = 1} ^ {n _ {e}} \sqrt {\lambda_ {J}} \phi_ {I J} \zeta_ {J} \tag {58}
$$

where $k_{\mu}$ is the mean value; $\lambda_{J}$ is the J-th eigenvalue; $\phi_{IJ}$ refers to the I-th component of J-th eigenvector; $\zeta_{J}$ is the J-th uncorrelated variable; $\widetilde{N}_{I}(x)$ is the I-th C-HiDeNN basis function. As a result, the INN-TD approximation of the PDE operator can be written as:

$$
\mathcal {J} u (x, \zeta , t) = \sum_ {m = 1} ^ {M} u _ {x} ^ {(m)} (x) \psi_ {1} ^ {(m)} (\zeta_ {1}) \psi_ {2} ^ {(m)} (\zeta_ {2}) \cdot \dots \cdot \psi_ {n _ {e}} ^ {(m)} (\zeta_ {n _ {e}}) u _ {t} ^ {(m)} (t) \tag {59}
$$

where each univariate function is approximated using C-HiDeNN basis functions. The details of the model discretizations are shown in Table 10. INN-TD requires $96.74 \pm 4.65$ sec for the offline computation. Fig. 17 shows that INN-TD gives accurate predictions as compared to the corresponding finite difference solutions for different realizations of the conductivity field.

Table 10. Summary of the INN-TD model for approximating PDE operator 

<table><tr><td>Discretization variable</td><td>Space (x)</td><td>Parameter for k (ζ)</td><td>Time (t)</td></tr><tr><td>Domain</td><td>[0, 1]</td><td>[-5, 5]</td><td>[0, 0.01]</td></tr><tr><td>Number of independent variables</td><td>1</td><td>71</td><td>1</td></tr><tr><td>Number of elements</td><td>71</td><td>101</td><td>151</td></tr><tr><td>Number of eigenvalues</td><td>-</td><td>71</td><td>-</td></tr></table>

(a)   
![](images/3675bd9387cfc414cccd9f5ae0be80d4d0123c5c169d4cb0a3d6569b6cc86db4.jpg)

<details>
<summary>line</summary>

| x    | k(x) |
| ---- | ---- |
| 0.0  | 11.0 |
| 0.25 | 9.0  |
| 0.5  | 11.0 |
| 0.75 | 9.0  |
| 1.0  | 11.0 |
</details>

![](images/21802d9fbb7bf7a16f44dff4186630ad0200f87a66f5f413131df7106da51abc.jpg)

<details>
<summary>heatmap</summary>

| x    | t=0.000 | t=0.002 | t=0.004 | t=0.006 | t=0.008 |
|------|---------|---------|---------|---------|---------|
| 0.0  | 0.000   | 0.001   | 0.002   | 0.003   | 0.004   |
| 0.5  | 0.002   | 0.003   | 0.004   | 0.005   | 0.006   |
| 1.0  | 0.004   | 0.005   | 0.006   | 0.007   | 0.008   |
</details>

![](images/630fed96e8964ed642c51961442ef16b5ec0d4841dd33d6d62a2e42fcc7b2a3e.jpg)

<details>
<summary>heatmap</summary>

| x    | 0.000 | 0.002 | 0.004 | 0.006 | 0.008 |
|------|-------|-------|-------|-------|-------|
| 0.0  | 0.000 | 0.002 | 0.004 | 0.006 | 0.008 |
| 1.0  | 0.000 | 0.002 | 0.004 | 0.006 | 0.008 |
</details>

![](images/667bfd9f221c87b2fdaba0153427ce4657fbc0440251041bcc942aff2e833c3a.jpg)

(b)   
![](images/4a095dbf730b95e14d3ffad4077525b1f3a9cd3c88eee3b5537ad8c4e3703b0e.jpg)

<details>
<summary>line</summary>

| x    | k(x) |
| ---- | ---- |
| 0.0  | 13.0 |
| 0.1  | 10.0 |
| 0.2  | 14.0 |
| 0.3  | 12.0 |
| 0.4  | 13.0 |
| 0.5  | 14.0 |
| 0.6  | 12.0 |
| 0.7  | 13.0 |
| 0.8  | 14.0 |
| 0.9  | 15.0 |
| 1.0  | 12.0 |
</details>

![](images/80eebd33198185c66e4c6d8f4e3d27e6cf5a8a689f4792c125873101864ec07b.jpg)

<details>
<summary>contour</summary>

| x    | t      | value  |
|------|--------|--------|
| 0.0  | 0.000  | 0.000  |
| 0.5  | 0.002  | 0.003  |
| 1.0  | 0.004  | 0.004  |
| 1.0  | 0.006  | 0.005  |
| 1.0  | 0.008  | 0.007  |
| 1.0  | 0.010  | 0.008  |
| 1.0  | 0.012  | 0.007  |
| 1.0  | 0.014  | 0.006  |
| 1.0  | 0.016  | 0.005  |
| 1.0  | 0.018  | 0.004  |
| 1.0  | 0.020  | 0.003  |
| 1.0  | 0.022  | 0.002  |
| 1.0  | 0.024  | 0.001  |
| 1.0  | 0.026  | 0.000  |
</details>

![](images/4f8c3e0a04b9bb79bb4f2e05a944c36c967e03ba948104ab4280b1b000d20a6d.jpg)

<details>
<summary>heatmap</summary>

| x    | 0.000 | 0.002 | 0.004 | 0.006 | 0.008 |
|------|-------|-------|-------|-------|-------|
| 0.0  | 0.000 | 0.002 | 0.004 | 0.006 | 0.008 |
| 1.0  | 0.000 | 0.002 | 0.004 | 0.006 | 0.008 |
</details>

![](images/bd561c9666e30d4ec5f91543a2c36361f59c87b72a701fd82c3ad84c501de87d.jpg)

<details>
<summary>heatmap</summary>

| x    | t=0.000 | t=0.002 | t=0.004 | t=0.006 | t=0.008 |
|------|---------|---------|---------|---------|---------|
| 0.0  | 0.0     | 0.0     | 0.0     | 0.0     | 0.0     |
| 0.5  | 0.0     | 0.0     | 0.0     | 0.0     | 0.0     |
| 1.0  | 0.0     | 0.0     | 0.0     | 0.0     | 0.0     |
</details>

Figure 17. Results for operator solving task when conductivity field $k(x)$ is a (a) sinusoidal function (b) random function.