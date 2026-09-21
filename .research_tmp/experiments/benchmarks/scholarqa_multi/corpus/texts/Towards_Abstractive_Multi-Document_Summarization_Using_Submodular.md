# SHOCK PROFILES FOR HYDRODYNAMIC MODELS FOR FLUID-PARTICLES FLOWS IN THE FLOWING REGIME

THIERRY GOUDON

Université Côte d'Azur, Inria, CNRS, LJAD, Parc Valrose, F-06108 Nice, France

PAULINE LAFITTE

CentraleSupélec, Fédération de Mathématiques FR CNRS 3487 & Labo. MICS, F-91192 Gif-sur-Yvette, France

CORRADO MASCIA

Dipartimento di Matematica Guido Castelnuovo,
Sapienza, University of Rome, Italy

ABSTRACT. Starting from coupled fluid-kinetic equations for the modeling of laden flows, we derive relevant viscous corrections to be added to asymptotic hydrodynamic systems, by means of Chapman-Enskog expansions and analyse the shock profile structure for such limiting systems. Our main findings can be summarized as follows. Firstly, we consider simplified models, which are intended to reproduce the main difficulties and features of more intricate systems. However, while they are more easily accessible to analysis, such toy-models should be considered with caution since they might lose many important structural properties of the more realistic systems. Secondly, shock profiles can be identified also in such a case, which can be proven to be stable at least in the regime of small amplitude shocks. Last, but not least, regarding at the temperature of the mixture flow as a parameter of the problem, we show that the zero-temperature model admits viscous shock profiles. Numerical results indicate that a similar conclusion should apply in the regime of small positive temperatures.

Keywords. Fluid-particles interactions; two-phase flow; hydrodynamic limit; shock profiles
Math. Subject Classification. 35C07 35L65 35Q35 76L05

# CONTENTS

1. Introduction 2   
2. General properties of conservation laws 7   
2.1. Shock wave solutions 8   
2.2. Stability concepts 8   
2.3. Entropy in the general setting 9   
2.4. Energy estimates and viscous dissipation 10

3. Flowing regime for the Burgers fluid-particle system 12   
3.1. Derivation and hyperbolicity 12   
3.2. Shock solutions 14   
3.3. Entropy for the inviscid Burgers fluid-particle system 16   
3.4. Viscous corrections leading to (vB) 18   
3.5. A few remarks on the stability estimate 22   
4. Flowing regime for the Euler fluid-particle system 23   
4.1. Derivation and hyperbolicity 23   
4.2. Shock solutions 24   
4.3. Entropy for the inviscid Euler fluid-particle system 26   
4.4. Viscous corrections leading to (vE) 27   
4.5. The temperature-less case 29   
4.6. Small-amplitude shock profiles analysis 30   
5. Large amplitude profiles for viscous Euler fluid-particle system 32   
5.1. Analysis of the temperature-less case 33   
5.2. Further scrutiny for positive temperature 40   
References 43

# 1. INTRODUCTION

A particle-laden flow is a class of two-phase fluid flow composed of a carrier phase, the surrounding continuous medium, and a disperse phase, constituted of small, immiscible and dilute particles. Such flows occur in many natural phenomena and industrial processes: snow and rock avalanches $[5, 31]$ , desert sandstorms, dispersions of pollutants, pollen and allergens in air $[35]$ , aerosols in respiratory flows $[2, 6]$ , fluidised beds $[19]$ , fuel injector, chemical reactors, internal combustion engines $[1, 24, 44]$ , just to name a few.

The broad variety of applications, and the wide range of scales involved in these situations, make it difficult to develop a unified framework. Two main viewpoints have been adopted to model such flows. The so-called Eulerian approach considers all phases as a continuum so that one is led to hydrodynamic systems for the densities and velocities (at least) of the disperse phase and the carrier phase $[3, 13, 25]$ . In contrast, the Lagrangian approach describes the particles by means of their distribution function in phase space, the evolution of which is coupled to a hydrodynamic model, based on either Euler or Navier-Stokes equations, for the carrier fluid. This defines a fluid-kinetic framework for describing the laden flow under consideration $[36, 37]$ . In both cases, the coupling is mainly achieved through the drag forces exerted by a phase on the other, which induces momentum exchanges between the two phases. A valuable approach consists in bringing out connections between these different settings, following the derivation of fluid equations from the kinetic equations of gas dynamics $[41]$ : several asymptotic regimes have been identified and investigated, both on theoretical and numerical grounds $[11, 15, 16, 17, 20, 26, 27]$ . The present work is a contribution in this direction.

As stated above, an alternative to the continuum approach describes the disperse phase by means of a Fokker-Planck equation for the dimensionless particle distribution function $f_{\epsilon}:(t,x,v)\mapsto f_{\epsilon}(t,x,v)$ , that is

$$
\frac {1}{T _ {\mathrm{ref}}} \partial_ {t} f _ {\epsilon} + \frac {V _ {\mathrm{ref}} v}{L _ {\mathrm{ref}}} \partial_ {x} f _ {\epsilon} = \frac {1}{T _ {S} V _ {\mathrm{ref}}} \partial_ {v} \left\{V _ {\mathrm{ref}} (v - u _ {\epsilon}) f _ {\epsilon} + \frac {V _ {\mathrm{th}} ^ {2}}{V _ {\mathrm{ref}}} \partial_ {v} f _ {\epsilon} \right\},
$$

where

- $t > 0$ , $x \in \mathbb{R}$ and $v \in \mathbb{R}$ are the dimensionless time, position and velocity variables, respectively;   
- $T_{\text{ref}}$ , $L_{\text{ref}}$ and $V_{\text{ref}} := L_{\text{ref}} / T_{\text{ref}}$ are the time, position and velocity dimensions, respectively;   
- $u_{\epsilon}$ is the velocity of the surrounding medium (dimensionless with respect to $V_{\mathrm{ref}}$ );   
- the Stokes settling time $T_{S}$ and the thermal speed $V_{\mathrm{th}}$ are defined by

$$
T _ {S} := \frac {m}{6 \pi \mu a} \qquad \mathrm{and} \qquad V _ {\mathrm{th}} := \sqrt {\frac {\kappa_ {B} \Theta}{m}},
$$

where a and m are the radius and mass of the particles, $\mu$ and $\Theta$ are the dynamic viscosity and temperature of the surrounding fluid, and $\kappa_{B}$ is the Boltzmann constant.

In the following, we concentrate on the flowing regime where $T_{S} = \epsilon T_{ref}$ and $V_{ref} = V_{th} \sqrt{\theta}$ . The parameters $\epsilon$ and $\theta$ are the reminders of the process of making the equation dimensionless. Moreover, we focus on the regime $\epsilon$ small, viz. $0 < \epsilon \ll 1$ . We refer the reader to [7, 8] for further details on this scaling.

The resulting equation for the unknown $f_{\epsilon}$ , describing the particle distribution in the phase space, is

$$
\partial_ {t} f _ {\epsilon} + v \partial_ {x} f _ {\epsilon} = \frac {1}{\epsilon} L _ {u _ {\epsilon}} (f _ {\epsilon})  , \tag {1.1}
$$

with the Fokker-Planck operator $L_{u}$ defined by

$$
L _ {u} (f) := \partial_ {v} \bigl \{(v - u) f + \theta \partial_ {v} f \bigr \} \tag {1.2}
$$

Since $u_{\epsilon}$ represents the velocity of the surrounding medium, the term $\partial_{v}\left\{(v-u_{\epsilon})f_{\epsilon}\right\}$ describes the drag force exerted on the particles by the fluid, assumed to be proportional to the relative velocity between the two species. Taking the zero-th and first order moments over the velocity variable gives the apparent mass density of particles $\rho_{\epsilon}$ and momentum of the disperse phase $J_{\epsilon}$ , where

$$
\rho_ {\epsilon} (t, x) := \int f _ {\epsilon} (t, x, v) \mathrm{d} v, \qquad J _ {\epsilon} (t, x) := \int v f _ {\epsilon} (t, x, v) \mathrm{d} v.
$$

Equation (1.1) is coupled to a balance law for the momentum of the carrier phase

$$
\partial_ {t} (n _ {\epsilon} u _ {\epsilon}) + \partial_ {x} \left\{n _ {\epsilon} u _ {\epsilon} ^ {2} + p (n _ {\epsilon}) \right\} = \frac {1}{\epsilon} (J _ {\epsilon} - \rho_ {\epsilon} u _ {\epsilon}), \tag {1.3}
$$

where $n_{\epsilon}$ and $u_{\epsilon}$ are, respectively, the mass density and the velocity field of the fluid. We assume $n_{\epsilon}$ is already dimensionless with respect to a reference density $n_{ref}$ and also make

$\rho_{\epsilon}$ dimensionless with respect to $n_{ref}$ . In the same way, p, that describes the pressure of the carrier phase, is already supposed dimensionless being defined by

$$
p (n) := \frac {\tilde {p} (n _ {\mathrm{ref}} n)}{n _ {\mathrm{ref}} V _ {\mathrm{ref}} ^ {2}},
$$

where $\tilde{p}$ is the dimensionalized pressure. The right-hand side in (1.3) accounts for the back-friction force exerted by the particles on the fluid.

Some hypotheses are required on the function $n \mapsto p(n)$ . Precisely, we assume that $p \in C^2$ is a strictly increasing, convex and coercive function, i.e.

$$
p ^ {\prime}, p ^ {\prime \prime} > 0 \quad \text { in } (0, \infty) \quad \text { and } \quad \lim _ {n \to + \infty} \frac {p (n)}{n} = + \infty . \tag {1.4}
$$

Since the pressure is determined up to an additive constant, we assume the additional condition $p(0) = 0$ . Moreover, we focus on the case $p'(0) = 0$ , a relevant case being the standard pure power form, usually referred to as $\gamma-law$ ,

$$
p (n) := C n ^ {\gamma} \quad \text { with } \quad C > 0, \quad \gamma > 1. \tag {1.5}
$$

As $\epsilon \to 0$ in (1.1), we guess that

$$
f _ {\epsilon} (t, x, v) \simeq \rho_ {\epsilon} (t, x) M _ {u _ {\epsilon} (t, x)} (v), \tag {1.6}
$$

where $M_u$ is the standard Maxwellian distribution, defined by

$$
M _ {u} (v) := \frac {1}{\sqrt {2 \pi \theta}} \exp \left(- \frac {| v - u | ^ {2}}{2 \theta}\right). \tag {1.7}
$$

Since $\theta \partial_v M_u = -(v - u)M_u$ , the Fokker-Planck operator $L_u$ can be rewritten as

$$
L _ {u} (f) = \theta \partial_ {v} \left\{M _ {u}   \partial_ {v} (M _ {u} ^ {- 1} f) \right\}, \tag {1.8}
$$

showing, in particular, that $L_{u}$ vanishes when computed at $v \mapsto f(v) = \rho M_{u}(v)$ . As a consequence, we expect that the dynamics can be described by means of macroscopic quantities in such a regime. Indeed, integrating (1.1) with respect to velocity yields

$$
\partial_ {t} \rho_ {\epsilon} + \partial_ {x} J _ {\epsilon} = 0.
$$

Next, we add the equation for the first order moment to (1.3) in order to get rid of the singular term by using the identity

$$
\int v \partial_ {v} L _ {u _ {\epsilon}} (f _ {\epsilon}) \mathrm{d} v = - \int \left\{(v - u _ {\epsilon}) f _ {\epsilon} + \theta \partial_ {v} f _ {\epsilon} \right\} \mathrm{d} v = - J _ {\epsilon} + \rho_ {\epsilon} u _ {\epsilon}.
$$

Hence, we end up with

$$
\partial_ {t} (J _ {\epsilon} + n _ {\epsilon} u _ {\epsilon}) + \partial_ {x} \biggl \{\int v ^ {2} f \mathrm{d} v + n _ {\epsilon} u _ {\epsilon} ^ {2} + p (n _ {\epsilon}) \biggr \} = 0.
$$

Going back to the ansatz (1.6), we infer

$$
J _ {\epsilon} \simeq \rho_ {\epsilon} u _ {\epsilon}, \quad \int v ^ {2} f _ {\epsilon}   \mathrm{d} v \simeq \rho_ {\epsilon} u _ {\epsilon} ^ {2} + \theta \rho_ {\epsilon}, \tag {1.9}
$$

and, dropping the dependence with respect to $\epsilon$ , we get the first order system

$$
\left\{ \begin{array}{l} \partial_ {t} \rho + \partial_ {x} (\rho u) = 0, \\ \partial_ {t} (r u) + \partial_ {x} \left\{r u ^ {2} + p (n) + \theta \rho \right\} = 0. \end{array} \right. \tag {1.10}
$$

where $r := \rho + n$ is called hybrid density, being the sum of the densities of the disperse and the carrier phases, denoted by $\rho$ and n, respectively.

From the modeling viewpoint, in some circumstances, it might be questionable to consider the diffusion with respect to the velocity variable as a stiff term in equation (1.1). Thus, it is equally relevant to consider the situation where $\theta = 0$ , which means that the Brownian velocity fluctuations are negligible. This situation is much more difficult for the analysis, since the formal ansatz becomes singular. Namely, as $\epsilon \to 0$ , denoting by $\delta_{v=u}$ the Dirac delta centered at u, we formally infer

$$
f _ {\epsilon} (t, x, v) \simeq \rho_ {\epsilon} (t, x) \delta_ {v = u _ {\epsilon} (t, x)}
$$

which leads to (1.9) with $\theta = 0$ . This approximation is often used in the modeling of laden flows, but depending on the considered coupling or asymptotic regime, this pressureless regime might lead to difficulties, both for the analysis [20, 26, 27] and for numerics, and possibly to physically irrelevant results [18]. Nevertheless, in this paper, we also consider the system (1.10) with $\theta = 0$ , regarded as a (formal) limiting regime.

Still inspired by the kinetic theory of gases, our objectives are the following. First, we formally derive diffusive corrections to system (1.10) coupled with (1.3), in the same spirit as the Chapman-Enskog procedure leads to the Navier-Stokes equations, keeping track of the $\mathcal{O}(\epsilon)$ -viscosity terms. Second, we investigate the structure of viscous shock profiles for the obtained systems. Namely, following the pioneering work [14], we wish to identify solutions of the diffusive equations with the form

$$
(\rho , n, u) (t, x) = \mathrm{W} (y) \quad \mathrm{where} \quad y := x - c t,
$$

for some given profile W with prescribed far-end states, that correspond to “admissible” discontinuous solutions of the diffusion-less system.

As a warm-up, we start with the case where (1.3) reduces to the mere Burgers equation: namely in (1.3), we (brutally) set $n_{\epsilon}=1$ . Hence, we firstly approach system (1.1)-(1.3) with the inviscid Burgers fluid-particle system, given by

$$
\partial_ {t} \binom{\rho}{r u} + \partial_ {x} \binom{\rho u}{r u ^ {2} + \theta \rho} = 0, \tag {iB}
$$

recalling that $r = 1 + \rho$ , and its corresponding viscous correction, referred to as the viscous Burgers fluid-particle system, whose explicit form is

$$
\partial_ {t} \binom{\rho}{r u} + \partial_ {x} \binom{\rho u}{r u ^ {2} + \theta \rho} = \epsilon \partial_ {x} \left(\mathbf {D} (\rho , r u) \partial_ {x} \binom{\rho}{r u}\right), \tag {vB}
$$

where

$$
\mathbf {D} (\rho , r u) = \frac {\rho u}{r ^ {3}} \left( \begin{array}{c c} u & - 1 \\ 0 & 0 \end{array} \right) + \frac {\theta}{r} \left( \begin{array}{c c} 1 / r & 0 \\ - \rho u & \rho \end{array} \right) \tag {1.11}
$$

(the formal derivation of the correction terms of order $\epsilon$ will be detailed later on). Even if both (iB) and (vB) possess an entropy $\zeta$ , defined by

$$
\zeta (\rho , r u) := \frac {1}{2} r u ^ {2} + \theta \rho \ln \rho ,
$$

such toy models are not fully physically meaningful, the main criticism being that they are not invariant under Galilean transformations. Nevertheless, they are considered here because they are amenable to detailed computations, which we consider illuminating.

Next, we move to the coupling with the Euler equations, where the density of the carrier fluid is driven by the additional conservation law

$$
\partial_ {t} n _ {\epsilon} + \partial_ {x} (n _ {\epsilon} u _ {\epsilon}) = 0.
$$

The corresponding inviscid Euler fluid-particle system is

$$
\partial_ {t} \left( \begin{array}{c} r \\ \rho \\ r u \end{array} \right) + \partial_ {x} \left( \begin{array}{c} r u \\ \rho u \\ r u ^ {2} + p (n) + \theta \rho \end{array} \right) = 0, \tag {iE}
$$

and the higher-order correction, named viscous Euler fluid-particle system is

$$
\partial_ {t} \left( \begin{array}{c} r \\ \rho \\ r u \end{array} \right) + \partial_ {x} \left( \begin{array}{c} r u \\ \rho u \\ r u ^ {2} + p (n) + \theta \rho \end{array} \right) = \epsilon \partial_ {x} \left(\mathbf {D} (r, \rho , r u)   \partial_ {x} \binom{r}{\rho}\right), \tag {vE}
$$

where

$$
\mathbf {D} (r, \rho , r u) = \frac {\rho n p ^ {\prime} (n)}{r ^ {2}} \left( \begin{array}{c c c} 0 & 0 & 0 \\ - 1 & 1 & 0 \\ 0 & 0 & 0 \end{array} \right) + \theta \left( \begin{array}{c c c} 0 & 0 & 0 \\ 0 & n ^ {2} / r ^ {2} & 0 \\ - \rho u / r & 0 & \rho / r \end{array} \right) \tag {1.12}
$$

(again, the formal derivation will be detailed later on). Differently from the previous case, systems (iE) and (vE) are invariant under Galilean transformations. In addition, (vE) also possesses an entropy, defined by

$$
\zeta (r, \rho , r u) := \frac {1}{2} r u ^ {2} + \Pi (n) + \theta \rho \ln \rho
$$

where

$$
\Pi (n) := \int_ {0} ^ {n} \int_ {0} ^ {s} \frac {1}{\varsigma} \frac {\mathrm{d} p}{\mathrm{d} \varsigma} (\varsigma) \mathrm{d} \varsigma \mathrm{d} s.
$$

In general, for both (vB) and (vE), the existence of an entropy $\zeta$ plays a pivotal role; specifically, it will be crucial to establish existence (and stability) of viscous shock profiles.

The paper is organized as follows. Section 2 collects some useful notions and basic facts on general hyperbolic-parabolic systems. It can be safely skipped by the reader familiar with these topics. In Section 3, we consider the model (vB), establishing the existence of viscous profile for weak shocks with positive temperatures. Subsequently, in Section 4, we turn to analyze system (vE) where the diffusion correction term is degenerate. Nevertheless, we are still able to provide a rigorous proof for the existence of weak shock profiles, whose stability can be established by appealing to general results for small-amplitude profiles of hyperbolic-parabolic systems. We also investigate the case where $\theta = 0$ , which induces new degeneracies; in particular, the entropy of the system is not strictly convex. Finally, Section 5 is devoted to further studying the model (vE) starting from the basic observation that a more complete result can be obtained for the temperature-less system, proceeding by direct inspection of the corresponding ODE. Expressing the ODE in reduced variables allows us to show that there are in fact two parameters of interest. This leads to showing the existence of a shock profile, which is illustrated numerically. In the temperature case, the differential system is also expressed in these reduced variables and

solved numerically for small temperatures. Finally, the numerical profile is compared to its temperature-less counterpart.

# 2. GENERAL PROPERTIES OF CONSERVATION LAWS

Let us collect here a series of definitions and basic statements that will be used throughout the paper. For further details, we refer the reader to the classical textbooks [9, 43]. Let $\mathcal{M}_m(\mathbb{R})$ be the space of $m\times m$ matrices with real entries. Then, given functions $F:\mathbb{R}^m\to \mathbb{R}^m$ and $\mathbf{D}:\mathbb{R}^m\to \mathcal{M}_m$ , we consider the system of conservation laws for the unknown function $\mathcal{W}:[0,\infty)\times \mathbb{R}\rightarrow \mathbb{R}^m$

$$
\partial_ {t} \mathscr {W} + \partial_ {x} F (\mathscr {W}) = \epsilon \partial_ {x} \left\{\mathbf {D} (\mathscr {W}) \partial_ {x} \mathscr {W} \right\} \quad t \geqslant 0, \quad x \in \mathbb {R}, \tag {2.1}
$$

for some $\epsilon > 0$ under the assumption that the formal limiting system $\epsilon \to 0^{+}$

$$
\partial_ {t} \mathscr {W} + \partial_ {x} F (\mathscr {W}) = 0 \qquad t \geqslant 0, \quad x \in \mathbb {R}, \tag {2.2}
$$

is strictly hyperbolic, i.e. the Jacobian dF has real distinct eigenvalues for any W under consideration.

Definition 2.1. Let $\mathbf{A},\mathbf{B}\in \mathcal{M}_n$ two matrices with $\mathbf{B}$ invertible. A (column) vector $\mathbf{r}\neq 0$ is said to be a right eigenvector of $\mathbf{A}$ with respect to $\mathbf{B}$ relative to the eigenvalue $\lambda$ if there holds $(\mathbf{A} - \lambda \mathbf{B})\mathbf{r} = 0$ . A left (row) eigenvector $\ell \neq 0$ of $\mathbf{A}$ with respect to $\mathbf{B}$ relative to the eigenvalue $\lambda$ is defined as $\ell (\mathbf{A} - \lambda \mathbf{B}) = 0$ .

For shortness, we use the shortened names right/left eigenvector of A with respect to B whenever the eigenvalue $\lambda$ is clear from the context.

To start with, we state and prove a straightforward Lemma showing that the directional derivatives of the eigenvalues of dF with respect to the corresponding right eigenvectors are invariant under diffeomorphisms.

Lemma 2.2. Let $F$ , $G$ , $H: \mathbb{R}^m \to \mathbb{R}^m$ be three differentiable functions such that $\mathrm{d}G$ is invertible and $F = H \circ G^{-1}$ . Let $\lambda$ be an eigenvalue of $\mathrm{d}F(\mathcal{W})$ , or, equivalently, an eigenvalue of $\mathrm{d}H(\mathcal{U})$ with respect to $\mathrm{d}G(\mathcal{U})$ , where $\mathcal{W} = G(\mathcal{U})$ . Let $\mathbf{r}$ be a right eigenvector of $\mathrm{d}F$ with respect to $\mathbf{I}$ . Then $\mathbf{s} = \mathrm{d}G(\mathcal{U})^{-1}\mathbf{r}$ is a right eigenvector of $\mathrm{d}H$ with respect to $\mathrm{d}G$ , also for the eigenvalue $\lambda$ . Moreover, the scalar products $\nabla_{\mathcal{W}}\lambda \cdot \mathbf{r}$ and $\nabla_{\mathcal{U}}\mu \cdot \mathbf{s}$ , where $\mu(\mathcal{U}) = \lambda(G(\mathcal{U}))$ , coincide.

Proof. Let $H(\mathcal{U}) := F \circ G(\mathcal{U}) = F(\mathcal{W})$ . The statement is a consequence of the chain rule which leads to the identities

$$
\mathrm{d} F (\mathcal {W}) = \mathrm{d} H (G ^ {- 1} (\mathcal {W})) \mathrm{d} (G ^ {- 1}) (\mathcal {W}), \qquad \mathrm{d} (G ^ {- 1}) (\mathcal {W}) = \left(\mathrm{d} G (\mathcal {U})\right) ^ {- 1},
$$

with the former recast simply as $\mathrm{d}F(\mathcal{W})=\mathrm{d}H(\mathcal{U})\mathrm{d}G(\mathcal{U})^{-1}$ . For $(\lambda,\mathbf{r})$ a left eigenpair of the matrix dF, we obtain

$$
0 = \big (\mathrm{d} F (\mathcal {W}) - \lambda \mathbf {I} \big) \mathbf {r} = \big (\mathrm{d} H (\mathcal {U}) \mathrm{d} G (\mathcal {U}) ^ {- 1} - \lambda \mathbf {I} \big) \mathbf {r} = \big (\mathrm{d} H (\mathcal {U}) - \lambda \mathrm{d} G (\mathcal {U}) \big) \mathbf {s}
$$

with $\mathbf{s} := \mathrm{d}G(\mathcal{U})^{-1}\mathbf{r}$ . Similarly, if $\boldsymbol{\ell}$ is a left eigenvector of $\mathrm{d}F(\mathcal{W})$ , we get

$$
0 = \ell \big (\mathrm{d} F (\mathcal {W}) - \lambda \mathbf {I} \big) = \ell \big (\mathrm{d} H (\mathcal {U}) - \lambda \mathrm{d} G (\mathcal {U}) \big) \mathrm{d} G (\mathcal {U}) ^ {- 1}.
$$

Thus, we infer that $\ell$ is a left eigenvector of $dH$ with respect to $dG$ . Next, we compute the gradient of the eigenvalue $\lambda(\mathcal{W}) = \lambda(G(\mathcal{U})) = \mu(\mathcal{U})$ with respect to the variables $\mathcal{W}$ (conservative) and $\mathcal{U}$ (non conservative) obtaining

$$
\nabla_ {\mathcal {U}} \mu (\mathcal {U}) = \mathrm{d} G (\mathcal {U}) ^ {\intercal} \nabla_ {\mathcal {W}} \lambda (G (\mathcal {U})).
$$

Hence, there holds

$$
\nabla_ {\mathcal {W}} \lambda (\mathcal {W}) \cdot \mathbf {r} = \nabla_ {\mathcal {U}} \mu (\mathcal {U}) \cdot \mathrm{d} G (\mathcal {U}) ^ {- 1} \mathbf {r} = \nabla_ {\mathcal {U}} \mu (\mathcal {U}) \cdot \mathbf {s},
$$

which concludes the proof.

![](images/acef7416b34dd9e54d7d12785af04b9af407d7505b97bae9458e5b2e7021b4a8.jpg)

The condition $\nabla_{\mathscr{W}}\lambda \cdot r \neq 0$ characterizes genuinely nonlinear fields. It plays the same role as strict convexity for scalar conservation laws, see [43, Section 17-B]. Oppositely, linearly degenerate fields, defined as the ones for which $\nabla_{\mathscr{W}}\lambda \cdot r = 0$ holds, correspond to linear transport equations with a pure motion of the initial datum without gain and loss of regularity. In particular, asymptotically stable shock solutions cannot be expected to appear into play.

2.1. Shock wave solutions. In the limiting regime $\epsilon = 0$ , we are specifically interested in discontinuous solutions that reach a specific state $\mathcal{W}_{*}$ , which are required to satisfy the classical Rankine-Hugoniot conditions [21, 22, 40]

$$
c \llbracket \mathcal {W} \rrbracket = \llbracket F (\mathcal {W}) \rrbracket , \tag {2.3}
$$

where c is the jump speed and $\left[\left[W\right]\right]=W-W_{*}$ . Such solutions can be parameterized by the scalar quantity $s\geqslant0$ and they are described by curves $s\mapsto\mathcal{W}(s)$ with speed function $s\mapsto c(s)$ associated to the eigenpairs of dF such that

$$
\left\{ \begin{array}{l} \mathcal {W} (0) = \mathcal {W} _ {*} \\ \dot {\mathcal {W}} (0) = \mathbf {r} (\mathcal {W} _ {*}) \end{array} \right. \text {and} \left\{ \begin{array}{l} c (0) = \lambda (\mathcal {W} _ {*}) \\ \dot {c} (0) = \frac {1}{2} \lambda (\mathcal {W} _ {*}) \mathbf {r} (\mathcal {W} _ {*}) \end{array} \right.
$$

(see e.g. [9, Section 8.2] or [43, Section 17-B]).

These pure jump solutions are said to satisfy Liu's entropy criterion when

$$
c (s) \leqslant c (\sigma) \text {   holds   for   any   } 0 \leqslant \sigma \leqslant s. \tag {2.4}
$$

The above criterion is crucial because it can be used to select relevant solutions among all weak discontinuous solutions of the equation. We refer the reader to $[9]$ for motivations and technical details about the conditions, which date back to $[29]$ .

2.2. Stability concepts. Next, let us switch on the diffusive term in system (2.1) by considering the case $\epsilon > 0$ . As a starting point, we consider the initial value problem for the linearized system at the state $W_{*}$ , namely

$$
\partial_ {t} \mathscr {W} _ {\epsilon} + \mathbf {A} \partial_ {x} \mathscr {W} _ {\epsilon} = \epsilon \mathbf {D} \partial_ {x} ^ {2} \mathscr {W} _ {\epsilon}, \quad \mathscr {W} _ {\epsilon} (0, \cdot) = \mathscr {W} _ {\epsilon , 0} (\cdot), \tag {2.5}
$$

where $\mathbf{A} := \mathrm{d}F(\mathcal{W}_{*})$ and $\mathbf{D} := \mathbf{D}(\mathcal{W}_{*})$ .

System (2.5) has constant coefficients and, consequently, it can be scrutinized by means of standard Fourier analysis, analysing the corresponding symbol $P_{*}^{\epsilon}(\xi):=i\xi\mathbf{A}+\epsilon\xi^{2}\mathbf{D}$ . As it is well-known, the Fourier transform $\hat{W}_{\epsilon}$ of $W_{\epsilon}$ solves $\partial_{t}\hat{W}_{\epsilon}=-P_{*}^{\epsilon}(\xi)\hat{W}_{\epsilon}$ with initial condition $\hat{W}_{\epsilon}(0)=\hat{W}_{\epsilon,0}$ , whose solution $\hat{W}_{\epsilon}=\hat{W}_{\epsilon}(t;\xi)$ is formally given by the operator $t\mapsto\exp\{-tP_{*}^{\epsilon}(\xi)\}\hat{W}_{\epsilon,0}$ .

In [30, 38] different stability notions have been introduced, which turn out to be crucial for the existence of shock profiles.

Definition 2.3. The linear system (2.5) is uniformly stable at $\mathcal{W}_{*}$ with respect to $\epsilon$ , or simply stable at $\mathcal{W}_{*}$ , if for any $T > 0$ there exists $C_T > 0$ , independent of $\epsilon$ , such that

$$
\sup \left\{\frac {\| \mathcal {W} _ {\epsilon} (t , \cdot) \| _ {L ^ {2}}}{\| \mathcal {W} _ {\epsilon , 0} \| _ {L ^ {2}}}: 0 <   \epsilon <   1, t \in [ 0, T ] \right\} \leqslant C _ {T}.
$$

for some initial datum $W_{\epsilon,0}$ with non-zero $L^{2}$ -norm. The set of stable linear systems (2.5) is denoted by S. The interior of such set is composed by strictly stable systems.

Stability of $(2.5)$ can be rephrased by means of a property on the matrices A and D. Namely, according to [38], one has to check that the matrix D is uniformly stable with respect to A, meaning that for each T > 0 there exists a constant $C_{T}$ such that

$$
\sup \left\{\| \exp \{- t P _ {*} ^ {\epsilon} (\xi) \} \| _ {\mathcal {M} _ {m}}: 0 <   \epsilon <   1,   t \in [ 0, T ],   \xi \in \mathbb {R} \right\} \leqslant C _ {T}  , \tag {2.6}
$$

where $\|\cdot\|_{\mathcal{L}(L^{2})}$ denotes the operator norm from $L^{2}$ to $L^{2}$ . The latter is also equivalent to the existence of a universal constant C > 0 such that

$$
\sup _ {t \geqslant 0,   \zeta \in \mathbb {R}} \left\| \exp \{- t P _ {*} ^ {1} (\zeta) \} \right\| _ {\mathcal {M} _ {m}} \leqslant C  .
$$

In [30, Theorem 2.1] a list of properties equivalent to strict stability is given. Among them, we recall the following one for readers' convenience.

Theorem 2.4. The linear system (2.5) is strictly stable if and only if there exists $\delta > 0$ such that the eigenvalues $\lambda_j(\xi)$ of the symbol $P_*^\epsilon(\xi)$ satisfy the condition

$$
\operatorname{Re} \lambda_ {j} (\xi) \leqslant - \delta | \xi | ^ {2} \quad f o r a n y \quad \xi \in \mathbb {R}.
$$

The above result induces a necessary and sufficient condition for strict stability which is more manageable with respect to the original (and more abstract) definition.

2.3. Entropy in the general setting. A pivotal role is played by the notion of entropy, which provides very strong structural consequences on the underlying PDE system.

Definition 2.5. Let $\mathcal{U} \subset \mathbb{R}^m$ be a neighborhood of some reference point $\mathcal{W}_*$ . The $C^2$ functions $\zeta: \mathcal{U} \to \mathbb{R}$ and $q: \mathcal{U} \to \mathbb{R}$ with $\nabla q^\intercal = \nabla \zeta^\intercal$ dF form an entropy/entropy flux pair for system (2.1) if, for any $\mathcal{W} \in \mathcal{U}$ ,

i. (entropy convexity) $\mathrm{d}^2\zeta$ is positive definite;   
ii. (dissipativity) $\mathrm{d}^2\zeta \mathbf{D}$ has a positive definite symmetric part.

Incidentally, let us note that a necessary condition for the existence of a function q such that $\nabla q^{\intercal} = \nabla \zeta^{\intercal} \, dF$ is that the derivative of $\nabla \zeta^{\intercal} \, dF$ is symmetric. In coordinates, this amounts to require

$$
(\mathrm{d} ^ {2} q) _ {i j} = \partial_ {j} \Bigl (\sum_ {k} \partial_ {k} \zeta_ {k} \partial_ {i} F _ {k} \Bigr) = \sum_ {k} \partial_ {k} \zeta_ {k} \partial_ {j i} ^ {2} F _ {k} + \sum_ {k} \partial_ {j k} ^ {2} \zeta_ {k} \partial_ {i} F _ {k}.
$$

Hence, $d^{2}F_{k}$ being symmetric, this is equivalent to requesting that $d^{2}\zeta dF$ is symmetric.

Proposition 2.6. Assume system (2.1) admits a strictly convex entropy $\zeta$ . Then, the entropy variable $\mathcal{U} := \nabla \zeta(\mathcal{W})$ satisfies

$$
\partial_ {t} G (\mathcal {U}) + \partial_ {x} H (\mathcal {U}) = \epsilon \partial_ {x} \left\{\mathbf {B} (\mathcal {U}) \partial_ {x} \mathcal {U} \right\} \tag {2.7}
$$

where $\mathcal{W} = G(\mathcal{U})$ , $\mathrm{d}G$ is symmetric positive definite, $\mathrm{d}H$ is symmetric, $\mathbf{B}$ is symmetric.

Proof. The change of coordinates $\mathcal{W} \to \mathcal{U} = \nabla \zeta(\mathcal{W})$ is globally invertible, since its Jacobian $\mathrm{d}^2\zeta$ is symmetric and positive definite. In turn, system (2.1) can be cast under the form (2.7), with $\mathrm{d}G(\mathcal{U}) = (d^2\zeta(\mathcal{W}))^{-1}$ symmetric positive definite, since the entropy is strictly convex, where $H(\mathcal{U}) = (F \circ G)(\mathcal{U})$ and $\mathbf{B}(\mathcal{U}) = (\mathbf{D} \circ G)(\mathcal{U})\mathrm{d}G(\mathcal{U}) = \mathbf{D}(\mathcal{W})(\mathrm{d}^2\zeta(\mathcal{W}))^{-1}$ . The symmetry of $\mathrm{d}H = \mathrm{d}F(\mathrm{d}^2\zeta)^{-1}$ , and $\mathbf{B}$ follow from the symmetry of $\mathrm{d}^2\zeta\mathrm{d}H$ and $\mathrm{d}^2\zeta\mathbf{D}$ .

In addition, following [30, Corollary 2.2], it can be proved that a sufficient condition for strict stability at $\mathcal{W}_*$ is the existence of a positive definite symmetric matrix $\mathbf{X}$ so that $\mathbf{XA}$ is symmetric and $\mathbf{XD}$ is positive definite (not necessarily symmetric). Later on, the matrix $\mathbf{X}$ will be chosen equal to the hessian $\mathrm{d}^2\zeta$ of the entropy $\zeta$ , i.e. $\mathbf{X} = \mathrm{d}^2\zeta$ .

2.4. Energy estimates and viscous dissipation. The existence of an entropy is crucial to develop some basic energy estimates holding for $(2.1)$ . For the sake of simplicity, let us explain the role of entropy by considering the linearized equations in $(2.5)$ .

Preliminarily, let us recall a standard property. Decomposing a (constant) matrix $\mathbf{A}$ as the sum of its symmetric and skew-symmetric parts $\mathbf{A} = \mathbf{A}_{\mathrm{sym}} + \mathbf{A}_{\mathrm{skew}}$ where $\mathbf{A}_{\mathrm{sym}} := \frac{1}{2} (\mathbf{A} + \mathbf{A}^{\intercal})$ and $\mathbf{A}_{\mathrm{skew}} := \frac{1}{2} (\mathbf{A} - \mathbf{A}^{\intercal})$ , there holds

$$
\int_ {\mathbb {R}} \mathcal {W} \cdot (\mathbf {A} \partial_ {x} \mathcal {W}) \mathrm{d} x = \int_ {\mathbb {R}} \mathcal {W} \cdot (\mathbf {A} _ {\text {skew}} \partial_ {x} \mathcal {W}) \mathrm{d} x. \tag {2.8}
$$

for any real-valued smooth function W such that $\mathcal{W}(\pm\infty)=0$ , Indeed for symmetric matrices S, there holds

$$
\int_ {\mathbb {R}} \boldsymbol {\mathcal {W}} \cdot (\mathbf {S} \partial_ {x} \boldsymbol {\mathcal {W}}) \mathrm{d} x = \int_ {\mathbb {R}} (\mathbf {S} ^ {\intercal} \boldsymbol {\mathcal {W}}) \cdot \partial_ {x} \boldsymbol {\mathcal {W}} \mathrm{d} x = \int_ {\mathbb {R}} (\mathbf {S} \boldsymbol {\mathcal {W}}) \cdot \partial_ {x} \boldsymbol {\mathcal {W}} \mathrm{d} x = - \int_ {\mathbb {R}} (\mathbf {S} \partial_ {x} \boldsymbol {\mathcal {W}}) \cdot \boldsymbol {\mathcal {W}} \mathrm{d} x
$$

so that (2.8) is zero for $\mathbf{A}$ symmetric, i.e. if $\mathbf{A} = \mathbf{A}_{\mathrm{sym}}$ .

Such property suggests the following preliminary definition.

Definition 2.7. System (2.1) is said to be parabolic at $\mathcal{W}_{*}$ if the (real) eigenvalues of the symmetric matrix $\mathbf{D}_{\mathrm{sym}} := \frac{1}{2} (\mathbf{D} + \mathbf{D}^{\intercal})$ lie in $(0, +\infty)$ .

In such a case, assuming appropriate boundary conditions at $\infty$ on $W_{\epsilon}$ , it is possible to deduce an energy estimate for (2.5). Precisely, multiplying by $W_{\epsilon}$ and integrating with respect to the space variable x, we end up with (after an additional integration by parts)

$$
\frac {\mathrm{d}}{\mathrm{d} t} \left(\frac {1}{2} \| \mathcal {W} _ {\epsilon} (t, \cdot) \| _ {L ^ {2}} ^ {2}\right) + \epsilon \int_ {\mathbb {R}} \partial_ {x} \mathcal {W} _ {\epsilon} \cdot \mathbf {D} \partial_ {x} \mathcal {W} _ {\epsilon} \mathrm{d} x = - \int_ {\mathbb {R}} \mathcal {W} _ {\epsilon} \cdot (\mathbf {A} \partial_ {x} \mathcal {W} _ {\epsilon}) \mathrm{d} x
$$

which, taking into account (2.8), reduces to

$$
\frac {\mathrm{d}}{\mathrm{d} t} \left(\frac {1}{2} \| \mathcal {W} _ {\epsilon} (t, \cdot) \| _ {L ^ {2}} ^ {2}\right) + \epsilon \int_ {\mathbb {R}} \partial_ {x} \mathcal {W} _ {\epsilon} \cdot \mathbf {D} _ {\mathrm{sym}} \partial_ {x} \mathcal {W} _ {\epsilon} \mathrm{d} x = - \int_ {\mathbb {R}} \mathcal {W} _ {\epsilon} \cdot \mathbf {A} _ {\mathrm{skew}} \partial_ {x} \mathcal {W} _ {\epsilon} \mathrm{d} x.
$$

For any M > 0, the above equality provides the estimate

$$
\begin{array}{l} \frac {\mathrm{d}}{\mathrm{d} t} \left(\frac {1}{2} \| \mathcal {W} _ {\epsilon} (t, \cdot) \| _ {L ^ {2}} ^ {2}\right) + \epsilon \int_ {\mathbb {R}} \partial_ {x} \mathcal {W} _ {\epsilon} \cdot \mathbf {D} _ {\text {sym}} \partial_ {x} \mathcal {W} _ {\epsilon} \mathrm{d} x \leqslant C _ {\mathbf {A}} \| \mathcal {W} _ {\epsilon} (t, \cdot) \| _ {L ^ {2}} \| \partial_ {x} \mathcal {W} _ {\epsilon} (t, \cdot) \| _ {L ^ {2}} \\ \leqslant \frac {1}{2} C _ {\mathbf {A}} M ^ {2} \| \mathscr {W} _ {\epsilon} (t, \cdot) \| _ {L ^ {2}} ^ {2} + \frac {1}{2} C _ {\mathbf {A}} M ^ {- 2} \| \partial_ {x} \mathscr {W} _ {\epsilon} (t, \cdot) \| _ {L ^ {2}} ^ {2}, \\ \end{array}
$$

with $C_{\mathbf{A}}$ depending only on $\mathbf{A}_{\mathrm{skew}}$ . In particular, if $\mathbf{A}$ is symmetric, then $C_{\mathbf{A}} = 0$ and parabolicity implies uniform stability.

In the general case, if system (2.1) is parabolic, denoting by $\lambda_{1} > 0$ the minimal eigenvalue of $D_{sym}$ , we have $\partial_{x}W_{\epsilon} \cdot D_{sym} \partial_{x}W_{\epsilon} \geqslant \lambda_{1}\|\partial_{x}W_{\epsilon}\|^{2}$ , such that

$$
\frac {\mathrm{d}}{\mathrm{d} t} \| \mathcal {W} _ {\epsilon} (t, \cdot) \| _ {L ^ {2}} ^ {2} + 2 \left(\epsilon \lambda_ {1} - \frac {1}{2} C _ {\mathbf {A}} M ^ {- 2}\right) \| \partial_ {x} \mathcal {W} _ {\epsilon} (t, \cdot) \| _ {L ^ {2}} ^ {2} \leqslant C _ {\mathbf {A}} M ^ {2} \| \mathcal {W} _ {\epsilon} (t, \cdot) \| _ {L ^ {2}} ^ {2}.
$$

Then, choosing $M^{2} = C_{\mathbf{A}}/(2\epsilon\lambda_{1})$ , we infer the estimate

$$
\frac {\mathrm{d}}{\mathrm{d} t} \| \mathcal {W} _ {\epsilon} (t, \cdot) \| _ {L ^ {2}} ^ {2} \leqslant \frac {C _ {\mathbf {A}} ^ {2}}{2 \epsilon \lambda_ {1}} \| \mathcal {W} _ {\epsilon} (t, \cdot) \| _ {L ^ {2}} ^ {2}.
$$

Hence, by a straightforward application of Grönwall's Lemma, we infer the bound

$$
\left\| \mathcal {W} _ {\epsilon} (t, \cdot) \right\| _ {L ^ {2}} \leqslant C _ {\epsilon , T} \left\| \mathcal {W} _ {\epsilon} (0, \cdot) \right\| _ {L ^ {2}}
$$

where $C_{\epsilon,T} = \exp \left\{ C_{\mathbf{A}}^{2} T / (4 \epsilon \lambda_{1}) \right\}$ tends to $+\infty$ as $\epsilon \to 0^{+}$ if $C_{A} > 0$ . Hence, it is transparent that such bounds do not provide any information relative to the (eventual) uniform stability of system (2.5). In fact, some choices of (non-symmetric) A lead to the non-uniform stability of (2.5).

Differently, let us explore the case in which there exists a symmetric positive definite matrix X such that X A is symmetric and $(\mathbf{X}\mathbf{D})_{\mathrm{sym}}$ is positive definite. Then, multiplying the linear system in (2.5) by X, we obtain the modified system

$$
\mathbf {X} \partial_ {t} \mathcal {W} _ {\epsilon} + \mathbf {X A} \partial_ {x} \mathcal {W} _ {\epsilon} = \epsilon \mathbf {X D} \partial_ {x} ^ {2} \mathcal {W} _ {\epsilon}. \tag {2.9}
$$

Next, let us proceed as before: multiplying by $\mathcal{W}_{\epsilon}$ and integrating over $\mathbb{R}$ ,

$$
\frac {\mathrm{d}}{\mathrm{d} t} \| \mathbf {X} ^ {1 / 2} \mathcal {W} _ {\epsilon} (t, \cdot) \| _ {L ^ {2}} ^ {2} + 2 \epsilon \int_ {\mathbb {R}} \partial_ {x} \mathcal {W} _ {\epsilon} \cdot (\mathbf {X D}) _ {\text {sym}} \partial_ {x} \mathcal {W} _ {\epsilon} \mathrm{d} x \leqslant 0
$$

having used the identity (2.8) to the symmetric matrix X A which provides a corresponding starting energy estimates for $\|X^{1/2}W_{\epsilon}\|_{L^{2}}$ which is also uniform with respect to $\epsilon$ . Uniform stability is thus guaranteed under the assumption of the existence of a symmetrizer X with the properties described above.

When the system of conservation laws (2.1) possesses an entropy $\zeta$ , it can be proved that $d^2\zeta$ is indeed a symmetrizer for (2.1) and, thus, plays the role of $\mathbf{X}$ previously used to deduce an energy estimate uniform in $\epsilon$ . Entropy and its compatibility with the diffusion matrix thus allows us to derive stability estimates that are stronger than the ones obtained by using the parabolicity of the matrix $\mathbf{D}$ . This issue will be further illustrated later on.

If the matrix $(\mathbf{X}\mathbf{D})_{\mathrm{sym}}$ is only positive semi-definite, additional assumptions are required. Among others, a well-established approach posits that the celebrated Kawashima-Shizuta condition holds, consisting in the request that the linear equation in (2.5) is such that no eigenvectors of A are in the kernel of D (see [28, 42]). Difficulties relative to the case in which the above condition is not satisfied are explored in details in [4, 32].

# 3. FLOWING REGIME FOR THE BURGERS FLUID-PARTICLE SYSTEM

Let us assume that the carrier fluid is incompressible in the sense that $n_{\epsilon} \equiv 1$ in $(0, \infty) \times \mathbb{R}$ , so that the dimensionless hybrid density of the mixture becomes $r = 1 + \rho$ . Incidentally, let us observe that this is not the standard incompressibility assumption required in fluid-dynamics giving rise to Euler and Navier-Stokes equations for incompressible media. Indeed, assuming that the carrier fluid keeps a constant homogeneous density is a quite crude assumption. Even if controversial in principle, it makes some computations easier and more explicit, allowing to bring out interesting structural properties of the model. It is worth pointing out the analysis of traveling wave solutions and their stability has been already performed in [10] for a variant of this toy-model with temperature $\theta = 0$ and non-zero fluid viscosity.

# 3.1. Derivation and hyperbolicity. Given $\theta, \epsilon > 0$ , let us consider the coupled fluid-kinetic system

$$
\left\{ \begin{array}{l} \partial_ {t} f _ {\epsilon} + v \partial_ {x} f _ {\epsilon} = \epsilon^ {- 1} \partial_ {v} \left\{(v - u _ {\epsilon}) f _ {\epsilon} + \theta \partial_ {v} f _ {\epsilon} \right\}, \\ \partial_ {t} u _ {\epsilon} + \partial_ {x} u _ {\epsilon} ^ {2} = \epsilon^ {- 1} (J _ {\epsilon} - \rho_ {\epsilon} u _ {\epsilon}), \end{array} \right. \tag {3.1}
$$

where

$$
\rho_ {\epsilon} (t, x) = \int f _ {\epsilon} (t, x, v) \mathrm{d} v \quad \mathrm{and} \quad J _ {\epsilon} (t, x) = \int v f _ {\epsilon} (t, x, v) \mathrm{d} v,
$$

As explained in the Introduction, the expected limit as $\epsilon \rightarrow 0$ is system (iB).

Remark 3.1. As stated before, system (iB) is not invariant under Galilean transformations. Indeed, let us consider the change of variables $(s,y)=(t,x-u_{0}t)$ , with $u_{0}\in R$ a constant velocity, corresponding to $(\partial_{t},\partial_{x})=(\partial_{s}-u_{0}\partial_{y},\partial_{y})$ and set $v:=u-u_{0}$ . Applying the transformation to the first equation in (iB), we infer

$$
\partial_ {t} \rho + \partial_ {x} (\rho u) = \partial_ {s} \rho - u _ {0} \partial_ {y} \rho + \partial_ {y} \bigl \{\rho (v + u _ {0}) \bigr \} = \partial_ {s} \rho + \partial_ {y} (\rho v).
$$

Concerning the second equation, we deduce upon computation

$$
\partial_ {t} (r u) + \partial_ {x} (r u ^ {2} + \theta \rho) = \partial_ {s} (r v) + \partial_ {y} (r v ^ {2} + \theta \rho) + u _ {0} \partial_ {y} v.
$$

In particular, in the new reference frame $(s,y)$ , system (iB) becomes

$$
\left\{ \begin{array}{l} \partial_ {s} \rho + \partial_ {y} (\rho v) = 0, \\ \partial_ {s} (r v) + \partial_ {y} (r v ^ {2} + \theta \rho) + u _ {0} \partial_ {y} v = 0, \end{array} \right.
$$

with $v := u - u_{0}$ , coinciding with the previous system if and only if $u_{0} = 0$ .

Differently, system (iB) is invariant under space reversal: indeed, applying the transformation $(s,y)=(t,-x)$ and v=-u, we obtain

$$
\left\{ \begin{array}{l} \partial_ {t} \rho + \partial_ {x} (\rho u) = \partial_ {s} \rho - \partial_ {y} (- \rho v) = \partial_ {s} \rho + \partial_ {y} (\rho v) = 0, \\ \partial_ {t} (r u) + \partial_ {x} (r u ^ {2} + \theta \rho) = - \partial_ {s} (r v) - \partial_ {y} (r v ^ {2} + \theta \rho) = 0. \end{array} \right.
$$

System (iB) can be cast in a conservative vector form (2.2) where

$$
\mathscr {W} = (\rho , w) ^ {\intercal} \qquad \text { and } \qquad F (\mathscr {W}) = (\rho w / r, w ^ {2} / r + \theta \rho) ^ {\intercal}, \tag {3.2}
$$

where w = ru. Examining hyperbolicity amounts to focus on the linearization

$$
\partial_ {t} \mathcal {W} + \mathrm{d} F (\mathcal {W} _ {*}) \partial_ {x} \mathcal {W} = 0,
$$

where

$$
\mathrm{d} F (\mathcal {W}) := \left( \begin{array}{c c} w / r ^ {2} & \rho / r \\ - w ^ {2} / r ^ {2} + \theta & 2 w / r \end{array} \right) = \left( \begin{array}{c c} u / r & \rho / r \\ - u ^ {2} + \theta & 2 u \end{array} \right).
$$

In the following computations, let us drop the subscript \* for the sake of shortness. By definition, system (2.2) is strictly hyperbolic at W if and only if the polynomial

$$
p (\lambda) := \det \left(\mathrm{d} F (\mathscr {W}) - \lambda \mathbf {I}\right) = 0
$$

has distinct real roots. Upon substitution, we obtain

$$
\lambda^ {2} - 2 \left(1 + \frac {1}{2 r}\right) u \lambda + \frac {2 + \rho}{r} u ^ {2} - \frac {\theta \rho}{r} = 0
$$

whose solutions are

$$
\lambda_ {\pm} (\mathscr {W}) := u \pm \frac {\sqrt {u ^ {2} + \theta \delta^ {2}} \pm u}{2 r} \quad \text {with} \quad \delta (\rho) := 2 \sqrt {\rho r}. \tag {3.3}
$$

Given $\theta > 0$ , the function $\rho \mapsto \delta(\rho)$ is invertible for $\rho \in [0, +\infty)$ . Indeed, the relation defining $\chi := \delta^{2} = 4\rho r = 4\rho(1 + \rho)$ can be rewritten as a second order polynomial in $\rho$ , viz. $4\rho^{2} + 4\rho - \chi = 0$ . Taking the positive root in the standard formula for the roots of second order polynomials, we infer

$$
\rho = \varphi (\chi) := \frac {\sqrt {1 + \chi} - 1}{2} = \frac {1}{2} \frac {\chi}{\sqrt {1 + \chi} + 1}.
$$

If $\rho$ is strictly positive, so are $\delta$ and $\chi$ , thus the system is strictly hyperbolic for $\theta > 0$ .

To classify the type of hyperbolic system we are dealing with, we analyse the scalar product $\nabla_{\mathcal{W}}\lambda_{\pm}\cdot r_{\pm}$ where $r_{\pm}$ are right eigenvectors of the matrix dF - $\lambda I$ relative to $\lambda_{\pm}$ .

Proposition 3.2. For $\theta > 0$ , system (iB) is strictly hyperbolic with two genuinely nonlinear fields for $(\rho, ru) \in (0, \infty) \times \mathbb{R}$ .

Proof. System (2.2) can be also written as a system in $\mathcal{U} := (\rho, u)$ :

$$
\partial_ {t} G (\mathcal {U}) + \partial_ {x} H (\mathcal {U}) = 0 \tag {3.4}
$$

where the functions $G(\mathcal{U}) = (\rho, ru)^{\intercal}$ and $H(\mathcal{U}) = (\rho u, ru^2 + \theta \rho)^{\intercal}$ are such that

$$
\mathrm{d} G (\mathcal {U}) := \left( \begin{array}{c c} 1 & 0 \\ u & r \end{array} \right), \qquad \mathrm{d} H (\mathcal {U}) := \left( \begin{array}{c c} u & \rho \\ u ^ {2} + \theta & 2 r u \end{array} \right).
$$

Let us set $\mu_{\pm}(\mathcal{U}) = \lambda_{\pm}(G(\mathcal{U}))$ . In particular, $\mu_{\pm}\big|_{u=0} = \pm \sqrt{\theta\rho/r}$ . By Lemma 2.2, it is equivalent to compute $\nabla_{\mathcal{U}}\mu_{\pm}\cdot\mathbf{s}_{\pm}$ where $(\mathrm{d}H-\mu_{\pm}\mathrm{d}G)\mathbf{s}_{\pm}=0$ . In turn, this reduces to finding $\mathbf{s}_{\pm}$ such that $(u-\mu_{\pm},\rho)\cdot\mathbf{s}_{\pm}=0$ . Let us choose $\mathbf{s}_{\pm}=(\rho,\mu_{\pm}-u)^{\intercal}$ , so that the functions $\mathcal{U}\mapsto\mathbf{s}_{\pm}(\mathcal{U})$ are smooth on $(0+\infty)\times\mathbb{R}$ .

The auxiliary function $\sigma : \mathbb{R} \to (-1, 1)$ , defined by $\sigma(x) := x / \sqrt{1 + x^2}$ , see Fig. 1, is continuous, odd and such that

$$
0 \leqslant | \sigma (x) | \leqslant \min \left\{1, | x | \right\}, \quad \sigma^ {\prime} (x) = (1 + x ^ {2}) ^ {- 3 / 2}. \tag {3.5}
$$

![](images/7699f366cee174d9a748b3d10821e7c3b524507725197e02776e695fd74e6e86.jpg)

<details>
<summary>line</summary>

| x   | Solid Line | Dotted Line |
| --- | ---------- | ----------- |
| -3  | 1.0        | 1.0         |
| -2  | 0.9        | 1.0         |
| -1  | 0.7        | 1.0         |
| 0   | 0.0        | 1.0         |
| 1   | 0.7        | 1.0         |
| 2   | 0.9        | 1.0         |
| 3   | 1.0        | 1.0         |
</details>

FIGURE 1. Graph of the function $x \mapsto |\sigma(x)|$ (continuous line) compared to the one of $x \mapsto \min \{1, |x|\}$ (dotted line) for $x \in \mathbb{R}$ .

Moreover, $\sigma$ is invertible with inverse $\psi : (-1, 1) \to \mathbb{R}$ given by $x = \psi(y) := y / \sqrt{1 - y^2}$ . In term of $\sigma$ , the eigenvalues $\mu_{\pm}$ can be represented as

$$
\mu_ {\pm} (\mathcal {U}) = u \pm \frac {1}{2 r} (1 \pm \sigma) \sqrt {u ^ {2} + \theta \delta^ {2}}
$$

with $\sigma$ computed at $u / \sqrt{\theta\delta^2}$ .

Since the gradient $\nabla_{\mathcal{U}}\mu_{\pm} = (\partial_{\rho}\mu_{\pm},\partial_{u}\mu_{\pm})$ is given by

$$
\partial_ {\rho} \mu_ {\pm} (\mathcal {U}) = - \frac {(1 \pm \sigma) u}{2 r ^ {2}} \pm \frac {\theta}{r \sqrt {u ^ {2} + \theta \delta^ {2}}}, \qquad \partial_ {u} \mu_ {\pm} (\mathcal {U}) = 1 + \frac {1 \pm \sigma}{2 r},
$$

there holds

$$
\begin{array}{l} \nabla_ {\mathcal {U}} \mu_ {\pm} (\mathcal {U}) \cdot \mathbf {s} _ {\pm} = \left(- \frac {(1 \pm \sigma) u}{2 r ^ {2}} \pm \frac {\theta}{r \sqrt {u ^ {2} + \theta \delta^ {2}}}, 1 + \frac {1 \pm \sigma}{2 r}\right) \cdot \left(\rho , \pm \frac {1 \pm \sigma}{2 r} \sqrt {u ^ {2} + \theta \delta^ {2}}\right) \\ = - \frac {(1 \pm \sigma) \rho u}{2 r ^ {2}} \pm \frac {\theta \rho}{r \sqrt {u ^ {2} + \theta \delta^ {2}}} \pm \frac {1 \pm \sigma}{2 r} \sqrt {u ^ {2} + \theta \delta^ {2}} \pm \frac {(1 \pm \sigma) ^ {2}}{4 r ^ {2}} \sqrt {u ^ {2} + \theta \delta^ {2}} \\ = \pm \frac {1 \pm \sigma}{2 r} \left\{\sqrt {u ^ {2} + \theta \delta^ {2}} + \frac {(1 \pm \sigma) \sqrt {u ^ {2} + \theta \delta^ {2}}}{2 r} \mp \frac {\rho u}{r} \right\} \pm \frac {\theta \rho}{r \sqrt {u ^ {2} + \theta \delta^ {2}}}. \\ \end{array}
$$

Since $r = 1 + \rho$ and $\sigma = u/\sqrt{u^{2} + \theta\delta^{2}}$ , the three terms in braces can be recast as

$$
\begin{array}{l} \sqrt {u ^ {2} + \theta \delta^ {2}} + \frac {(1 \pm \sigma) \sqrt {u ^ {2} + \theta \delta^ {2}}}{2 r} \mp \frac {\rho u}{r} = \left(1 + \frac {1 \pm \sigma}{2 r} \mp \frac {\rho \sigma}{r}\right) \sqrt {u ^ {2} + \theta \delta^ {2}} \\ = \left\{1 + \frac {1 \pm \sigma}{2} + \rho (1 \mp \sigma) \right\} \frac {\sqrt {u ^ {2} + \theta \delta^ {2}}}{r} \geqslant \frac {\sqrt {u ^ {2} + \theta \delta^ {2}}}{r} \geqslant 0, \\ \end{array}
$$

with the equality holding only for $\mathcal{U} = \mathbf{0}$ in the case $\theta > 0$ . Hence, for $\rho > 0$ , there hold

$$
\nabla_ {\mathcal {W}} \lambda_ {-} \cdot \mathbf {r} _ {-} = \nabla_ {\mathcal {U}} \mu_ {-} \cdot \mathbf {s} _ {-} <   0 <   \nabla_ {\mathcal {U}} \mu_ {+} \cdot \mathbf {s} _ {+} = \nabla_ {\mathcal {W}} \lambda_ {+} \cdot \mathbf {r} _ {+},
$$

where we make use of Lemma 2.2.

3.2. Shock solutions. Shock waves of system (2.2) are special solutions $\mathcal{W}(x,t)=\mathrm{W}(y)$ depending only on the variable y:=x-ct with the form of a pure jump

$$
\mathcal {W} (x, t) = \mathrm{W} (y) := \left\{ \begin{array}{l l} \mathcal {W} _ {*} & \quad \text { if } \quad y <   0, \\ \mathcal {W} & \quad \text { if } \quad y \geqslant 0. \end{array} \right.
$$

where $\mathcal{W}_{*} := (\rho_{*}, r_{*}u_{*})$ and $\mathcal{W} := (\rho, ru)$ . In presence of Galilean invariance, we could focus without loss of generality on stationary solutions W, i.e. c = 0 and y = x. Unfortunately, as observed in Remark 3.1, system (iB) does not possess such a symmetry and the corresponding reduction cannot be considered.

In order to be weak solutions, such functions are forced to satisfy the Rankine–Hugoniot conditions (2.3). For system (iB), they take the specific form

$$
\left\{ \begin{array}{l} - c [   [ \rho ]   ] + [   [ \rho u ]   ] = 0, \\ - c [   [ r u ]   ] + [   [ r u ^ {2} + \theta \rho ]   ] = 0, \end{array} \right. \tag {3.6}
$$

where $\llbracket g\rrbracket = g - g_*$ .

Given $\rho_{*}$ and $u_{*}$ , let us show that these relations lead to $u$ being a function of $\rho$ . If $[[\rho] = 0$ , then from the first equation in (3.6), we infer $\rho_{*}[[u]] = 0$ . Hence, assuming $\rho_{*} > 0$ , we are forced to have $[[u]] = 0$ , so that the solution is actually a constant state. Being interested in non constant profiles, we assume $[[\rho] \neq 0$ . Then, the propagation speed can be expressed as

$$
c = \frac {\llbracket \rho u \rrbracket}{\llbracket \rho \rrbracket}. \tag {3.7}
$$

Next, we are going to use the two following relations, valid for any functions f and g,

$$
\llbracket f g \rrbracket = \llbracket f \rrbracket g _ {*} + f \llbracket g \rrbracket \quad \text {and} \quad \llbracket f g ^ {2} \rrbracket = \llbracket f \rrbracket g _ {*} ^ {2} + 2 f g _ {*} \llbracket g \rrbracket + f \llbracket g \rrbracket^ {2}. \tag {3.8}
$$

Substituting (3.7) in the identity (3.6), we obtain

$$
[ [ \rho u ] ] [ [ u ] ] + [ [ \rho u ] ] ^ {2} = [ [ \rho u ] ] [ [ r u ] ] = [ [ \rho ] ] [ [ r u ^ {2} ] ] + \theta [ [ \rho ] ] ^ {2},
$$

and, taking advantage of (3.8), we infer

$$
\rho_ {*} r [ [ u ] ] ^ {2} - [ [ \rho ] ] u _ {*} [ [ u ] ] - \theta [ [ \rho ] ] ^ {2} = 0.
$$

Considering the form (3.4) of the original system (2.2), the set of admissible shocks $H_{W_{*}}$ of a given state $\mathcal{W}_{*} = (\rho_{*}, r_{*}u_{*})$ , usually called Hugoniot locus, is given by the union of two distinct branches, here denoted by $H_{W_{*},+}$ and $H_{W_{*},-}$ (see Figure 2)

$$
\mathcal {H} _ {\mathscr {W} _ {*}, \pm} = \left\{(\rho , r u _ {\pm}): \rho > 0, u _ {\pm} (\rho) = u _ {*} \pm \frac {[ [ \rho ] ]}{\rho_ {*}} \cdot \frac {\sqrt {u _ {*} ^ {2} + \theta \Delta^ {2}} \pm u _ {*}}{2 r} \right\}, \tag {3.9}
$$

with $\Delta(\rho,\rho_{*}) := 2\sqrt{\rho_{*}r}$ . Accordingly, along each branch, the shock speed is given by (3.7), that becomes, using again (3.8),

$$
c _ {\pm} (\rho ; \mathscr {W} _ {*}) = u _ {*} + \rho \frac {[ [ u ] ]}{[ [ \rho ] ]} = u _ {*} \pm \frac {\rho}{\rho_ {*}} \cdot \frac {\sqrt {u _ {*} ^ {2} + \theta \Delta^ {2}} \pm u _ {*}}{2 r}. \tag {3.10}
$$

Note that we can equally write

$$
c _ {\pm} (\rho ; \mathcal {W} _ {*}) = u + \rho_ {*} \frac {[ [ u ] ]}{[ [ \rho ] ]} = u \pm \frac {\sqrt {u _ {*} ^ {2} + \theta \Delta^ {2}} \pm u _ {*}}{2 r}.
$$

With the sign (+), respectively $(-)$ , $c_{\pm}$ is larger, resp. smaller, than both the left velocity $u_{*}$ and the right velocity u.

Specifically, we regard at the curves defined by (3.9) and (3.10) as parametrizations of the states $\mathcal{W}$ that can be connected to $\mathcal{W}_*$ by a pure discontinuity providing a weak

![](images/ec4f14be33655a1436c2f278393daddeb6022e8ccf5a636111a8a9b19663aadf.jpg)

<details>
<summary>line</summary>

| ρ   | Blue Solid | Red Dashed | Red Dotted |
| --- | ---------- | ---------- | ---------- |
| 0   | 3.5        | -0.5       | -2.0       |
| 1   | 3.0        | 1.0        | -1.5       |
| 2   | 2.5        | 2.0        | -1.0       |
| 3   | 2.0        | 2.5        | -0.5       |
| 4   | 1.5        | 3.0        | 0.0        |
| 5   | 1.0        | 3.5        | 0.5        |
| 6   | 0.5        | 4.0        | 1.0        |
| 7   | 0.0        | 4.5        | 1.5        |
| 8   | -0.5       | 5.0        | 2.0        |
| 9   | -1.0       | 5.5        | 2.5        |
| 10  | -1.5       | 6.0        | 3.0        |
</details>

FIGURE 2. Hugoniot locus for several states $\mathcal{U}_{*} = (\rho_{*}, u_{*})$ . Plots are given of several curves $\rho \mapsto u = u(\rho)$ defined by (3.9), $U_{*}$ being the intersection point of the two curves drawn with the same line-specification (dotted, dashed or dot-dashed).

solution to system (2.2) with corresponding parameter given by the density $\rho\in(0,+\infty)$ . As a matter of fact, we observe that (3.10) satisfies

$$
c _ {\pm} (\rho_ {*}; \mathcal {W} _ {*}) := \lim _ {\rho \to \rho_ {*}} c _ {\pm} (\rho ; \mathcal {W} _ {*}) = \lambda_ {\pm} (\mathcal {W} _ {*}) \quad \text {for} \rho_ {*} > 0 \qquad \text {and} \qquad c _ {\pm} (0; \mathcal {W} _ {*}) = u _ {*}.
$$

Moreover, since

$$
\partial_ {\rho} c _ {\pm} (\rho ; \mathcal {W} _ {*}) = \pm \frac {1}{2 \rho_ {*} r} \left\{\frac {\sqrt {u _ {*} ^ {2} + \theta \Delta^ {2}} \pm u _ {*}}{r} + \frac {2 \theta \rho_ {*} \rho}{\sqrt {u _ {*} ^ {2} + \theta \Delta^ {2}}} \right\}.
$$

there hold, for $\rho_{*} > 0$ ,

$$
\partial_ {\rho} c _ {-} (\rho ; \mathcal {W} _ {*}) <   0 <   \partial_ {\rho} c _ {+} (\rho ; \mathcal {W} _ {*}).
$$

As a consequence, we infer the equivalences valid for any $\rho$ between $\rho_{*}$ and $\bar{\rho}$

$$
c _ {+} (\bar {\rho}; \mathcal {W} _ {*}) - c _ {+} (\rho ; \mathcal {W} _ {*}) = \int_ {\rho} ^ {\bar {\rho}} \partial_ {\rho} c _ {+} (\xi ; \mathcal {W} _ {*}) \mathrm{d} \xi <   0 \quad \Longleftrightarrow \quad 0 \leqslant \bar {\rho} <   \rho , \tag {3.11}
$$

$$
c _ {-} (\bar {\rho}; \mathcal {W} _ {*}) - c _ {-} (\rho ; \mathcal {W} _ {*}) = \int_ {\rho} ^ {\bar {\rho}} \partial_ {\rho} c _ {-} (\xi ; \mathcal {W} _ {*}) \mathrm{d} \xi <   0 \quad \Longleftrightarrow \quad 0 \leqslant \rho <   \bar {\rho}.
$$

In particular, the (strict) Liu's entropy criterion is satisfied for $\bar{\rho} < \rho_*$ in the case of sign $+$ and for $\rho_* < \bar{\rho}$ in the case of sign - (see [29]). Since the system is genuinely nonlinear, this is equivalent to Lax's entropy condition for weak shocks [9, Theorem 8.4.2].

3.3. Entropy for the inviscid Burgers fluid-particle system. The quantity

$$
\mathscr {H} (f, u) := \underbrace {\frac {1}{2} u ^ {2}} _ {f l u i d} + \underbrace {\int H (f , v) \mathrm{d} v} _ {p a r t i c l e s} \quad \text { where } \quad H (f, v) := f \left(\frac {1}{2} v ^ {2} + \theta \ln f\right) \tag {3.12}
$$

defines an entropy for the fluid-kinetic model (3.1). Indeed, as previously observed, the kinetic equation in (3.1) can be rephrased as

$$
\partial_ {t} f _ {\epsilon} + v \partial_ {x} f _ {\epsilon} = \frac {\theta}{\epsilon} \partial_ {v} \left\{M _ {u _ {\epsilon}} \partial_ {v} (M _ {u _ {\epsilon}} ^ {- 1} f _ {\epsilon}) \right\},
$$

which involves the maxwellian $M_{u}$ defined in (1.7), to be considered coupled with

$$
\partial_ {t} u _ {\epsilon} + \partial_ {x} u _ {\epsilon} ^ {2} = \frac {\theta}{\epsilon} \int \left\{M _ {u _ {\epsilon}} \partial_ {v} (M _ {u _ {\epsilon}} ^ {- 1} f _ {\epsilon}) - \partial_ {v} f _ {\epsilon} \right\} \mathrm{d} v = \frac {\theta}{\epsilon} \int M _ {u _ {\epsilon}} \partial_ {v} (M _ {u _ {\epsilon}} ^ {- 1} f _ {\epsilon}) \mathrm{d} v.
$$

since $f_{\epsilon}$ is assumed to vanish at $\infty$ . Next, setting

$$
\mathcal {G} (f, u) := \frac {2}{3} u ^ {3} + \int v f \left(\frac {1}{2} v ^ {2} + \theta \ln f\right) \mathrm{d} v,
$$

we infer, integrating by parts,

$$
\begin{array}{l} \partial_ {t} \mathcal {H} = \int \left\{\frac {1}{2} v ^ {2} + \theta (\ln f _ {\epsilon} + 1) \right\} \partial_ {t} f _ {\epsilon} \mathrm{d} v + u _ {\epsilon} \partial_ {t} u _ {\epsilon} \\ = - \partial_ {x} \mathcal {G} + \frac {\theta}{\epsilon} \int u _ {\epsilon} M _ {u _ {\epsilon}} \partial_ {v} (M _ {u _ {\epsilon}} ^ {- 1} f _ {\epsilon}) \mathrm{d} v \\ + \frac {\theta}{\epsilon} \int \left\{\frac {1}{2} v ^ {2} + \theta (\ln f _ {\epsilon} + 1) \right\} \partial_ {v} \left[ M _ {u _ {\epsilon}} \partial_ {v} (M _ {u _ {\epsilon}} ^ {- 1} f _ {\epsilon}) \right] \mathrm{d} v \\ = - \partial_ {x} \mathcal {G} + \frac {\theta}{\epsilon} \int \partial_ {v} (M _ {u _ {\epsilon}} ^ {- 1} f _ {\epsilon}) \left\{M _ {u _ {\epsilon}} (u _ {\epsilon} - v) - \theta M _ {u _ {\epsilon}} f _ {\epsilon} ^ {- 1} \partial_ {v} f _ {\epsilon} \right\} d v \\ = - \partial_ {x} \mathcal {G} - \frac {\theta^ {2}}{\epsilon} \int f _ {\epsilon} ^ {- 1} \partial_ {v} (M _ {u _ {\epsilon}} ^ {- 1} f _ {\epsilon}) \left\{M _ {u _ {\epsilon}} (\partial_ {v} f _ {\epsilon}) - (\partial_ {v} M _ {u _ {\epsilon}}) f _ {\epsilon} \right\} \mathrm{d} v, \\ \end{array}
$$

since, as previously seen, $M_u(u - v) = \theta \partial_v M_u$ . Therefore, we deduce the estimate

$$
\partial_ {t} \mathcal {H} + \partial_ {x} \mathcal {G} = - \frac {\theta^ {2}}{\epsilon} \int M _ {u _ {\epsilon}} ^ {2} f _ {\epsilon} ^ {- 1} \left\{\partial_ {v} (M _ {u _ {\epsilon}} ^ {- 1} f _ {\epsilon}) \right\} ^ {2} \mathrm{d} v \leqslant 0.
$$

Next, let us focus on the regime $\epsilon \to 0^{+}$ for which the formal ansatz (1.6) is assumed to hold. As a consequence, inspired by the kinetic representation of conservation laws [39], we guess an entropy for the limit system by evaluating the functional $\mathcal{H}$ at the equilibrium $\rho_{\epsilon}M_{u_{\epsilon}}$ .

Preliminarly, let us observe that, knowing that

$$
\int M _ {u}   \mathrm{d} v = 1, \qquad \int (v - u) M _ {u}   \mathrm{d} v = 0, \qquad \int | v - u | ^ {2} M _ {u}   \mathrm{d} v = \theta  , \tag {3.13}
$$

there holds

$$
\int v ^ {2} M _ {u} \mathrm{d} v = \int \left\{u ^ {2} + 2 u (v - u) + | v - u | ^ {2} \right\} M _ {u} \mathrm{d} v = u ^ {2} + \theta
$$

Hence, inspired by the kinetic representation of conservation laws [39], the formal identity

$$
\begin{array}{l} \mathcal {H} (f _ {\epsilon}, u _ {\epsilon}) \simeq \mathcal {H} (\rho_ {\epsilon} M _ {u _ {\epsilon}}, u _ {\epsilon}) = \frac {1}{2} u _ {\epsilon} ^ {2} + \int \rho_ {\epsilon} M _ {u _ {\epsilon}} \left\{\frac {1}{2} v ^ {2} + \theta \ln (\rho_ {\epsilon} M _ {u _ {\epsilon}}) \right\} \mathrm{d} v \\ = \frac {1}{2} r _ {\epsilon} u _ {\epsilon} ^ {2} + \frac {1}{2} \theta \rho_ {\epsilon} + \rho_ {\epsilon} \int M _ {u _ {\epsilon}} \left\{\theta \ln \rho_ {\epsilon} - \frac {1}{2} \theta \ln (2 \pi \theta) - \frac {1}{2} (v - u) ^ {2} \right\} \mathrm{d} v \\ = \frac {1}{2} r _ {\epsilon} u _ {\epsilon} ^ {2} + \theta \rho \left\{\ln \rho_ {\epsilon} - \frac {1}{2} \ln (2 \pi \theta) \right\} \\ \end{array}
$$

suggests the (simplified) choice $\eta(\mathcal{U}) = \frac{1}{2} ru^2 + \theta\rho \ln \rho$ , obtained by disregarding the linear term in $\rho$ (since we already know that $\rho$ satisfies a convection equation), with

corresponding entropy flux given by $q(\mathcal{U}) = \left( \frac{2}{3} + \frac{1}{2} \rho \right) u^{3} + \theta \rho (\ln \rho + 1) u$ . The pair $(\eta, q)$ is an entropy/entropy flux pair for (3.4). Indeed, let us set

$$
Q := \partial_ {t} (\rho M _ {u}) + v \partial_ {x} (\rho M _ {u}).
$$

Using again (3.13), we infer for any (smooth) solution $(\rho, u)$ of (3.4)

$$
\int \binom{1}{v} Q   \mathrm{d} v = - \binom{0}{\partial_ {t} u + \partial_ {x} u ^ {2}}
$$

since integration with respect to $v$ yields the system of conservation laws. It follows that

$$
\partial_ {t} \eta + \partial_ {x} q = \partial_ {t} \left(\frac {1}{2} u ^ {2}\right) + \partial_ {x} \left(\frac {2}{3} u ^ {3}\right) + \int Q \left\{\frac {1}{2} v ^ {2} + \theta \ln (\rho M _ {u}) - \frac {1}{2} \theta \ln (2 \pi \theta) + 1 \right\} \mathrm{d} v
$$

$$
= \partial_ {t} \left(\frac {1}{2} u ^ {2}\right) + \partial_ {x} \left(\frac {2}{3} u ^ {3}\right) + \frac {1}{2} \int Q \left(v ^ {2} - | v - u | ^ {2}\right) \mathrm{d} v
$$

$$
= \partial_ {t} \left(\frac {1}{2} u ^ {2}\right) + \partial_ {x} \left(\frac {2}{3} u ^ {3}\right) + u \int v Q \mathrm{d} v = 0.
$$

In terms of the variables $\mathcal{W} = (\rho, w)$ , the entropy $\zeta$ is given by

$$
\zeta (\mathscr {W}) = \frac {w ^ {2}}{2 r} + \theta \rho \ln \rho . \tag {3.14}
$$

Upon differentiation, denoting by the same symbols $\nabla_{W}\zeta$ and $d_{W}^{2}\zeta$ the corresponding vector/matrix computed both at W, we obtain the following expressions that will be useful later on

$$
\nabla_ {\mathcal {W}} \zeta (\mathcal {W}) ^ {\intercal} = \left(- w ^ {2} / (2 r ^ {2}) + \theta (1 + \ln \rho), w / r\right) = \left(- u ^ {2} / 2 + \theta (1 + \ln \rho), u\right),
$$

$$
\mathrm{d} _ {\mathscr {W}} ^ {2} \zeta (\mathscr {W}) = \left( \begin{array}{c c} w ^ {2} / r ^ {3} + \theta / \rho & - w / r ^ {2} \\ - w / r ^ {2} & 1 / r \end{array} \right) = \left( \begin{array}{c c} u ^ {2} / r + \theta / \rho & - u / r \\ - u / r & 1 / r \end{array} \right). \tag {3.15}
$$

In addition, $\zeta$ is a convex function, since the hessian $d_{W}^{2}\zeta$ is positive definite.

The function $\zeta$ defined in (3.14) furnishes an entropy for system (2.2). Hence, the matrix $X := d_{W}^{2}\zeta$ is a symmetrizer for the flux F as can be directly checked (in fact, such property holds true for general hyperbolic systems, see [9, 34]).

3.4. Viscous corrections leading to (vB). We now use the Chapman-Enskog expansion to get the diffusive correction associated to system (iB). Specifically, we search for a hydrodynamic model with an appropriate modification, namely $(\rho_{\epsilon}, u_{\epsilon})$ (where the dependence on $\epsilon$ is explicitly stated) satisfies

$$
\partial_ {t} \binom{\rho_ {\epsilon}}{r _ {\epsilon} u _ {\epsilon}} + \partial_ {x} \binom{\rho_ {\epsilon} u _ {\epsilon}}{r _ {\epsilon} u _ {\epsilon} ^ {2} + \theta \rho_ {\epsilon}} = \mathcal {O} (\epsilon).
$$

In order to define the correction term, we expand the solution of the kinetic equation as

$$
f _ {\epsilon} = \rho_ {\epsilon} M _ {u _ {\epsilon}} + \epsilon g _ {\epsilon}, \qquad \int f _ {\epsilon} \mathrm{d} v = \rho_ {\epsilon}, \qquad \int g _ {\epsilon} \mathrm{d} v = 0,
$$

where $M_{u}$ is the Maxwellian distribution defined in (1.7). Recalling the identity (1.8), the system can be rewritten as

$$
(\partial_ {t} + v \partial_ {x}) (\rho_ {\epsilon} M _ {u _ {\epsilon}} + \epsilon g _ {\epsilon}) = L _ {u _ {\epsilon}} (g _ {\epsilon}),
$$

coupled with the equation for $u_{\epsilon}$

$$
\partial_ {t} u _ {\epsilon} + \partial_ {x} u _ {\epsilon} ^ {2} = \int (v - u _ {\epsilon}) g _ {\epsilon} \mathrm{d} v.
$$

Note that the integration of the kinetic equation yields

$$
\partial_ {t} \rho_ {\epsilon} + \partial_ {x} (\rho_ {\epsilon} u _ {\epsilon}) + \epsilon \partial_ {x} \int (v - u _ {\epsilon}) g _ {\epsilon}   \mathrm{d} v = 0, \tag {3.16}
$$

and

$$
\begin{array}{l} \partial_ {t} \left(\rho_ {\epsilon} u _ {\epsilon} + \epsilon \int v g _ {\epsilon} \mathrm{d} v\right) + \partial_ {x} \left(\rho_ {\epsilon} u _ {\epsilon} ^ {2} + \theta \rho_ {\epsilon} + \epsilon \int v ^ {2} g \mathrm{d} v\right) = - \int (v - u _ {\epsilon}) g _ {\epsilon} \mathrm{d} v. \tag {3.17} \\ = - \partial_ {t} u _ {\epsilon} - \partial_ {x} u _ {\epsilon} ^ {2}. \\ \end{array}
$$

We compute

$$
\begin{array}{l} \left(\partial_ {t} + v \partial_ {x}\right) \left(\rho_ {\epsilon} M _ {u _ {\epsilon}}\right) = M _ {u _ {\epsilon}} \left\{\partial_ {t} \rho_ {\epsilon} + \partial_ {x} \left(\rho_ {\epsilon} u _ {\epsilon}\right) \right\} + (v - u _ {\epsilon}) M _ {u _ {\epsilon}} \partial_ {x} \rho_ {\epsilon} - M _ {u _ {\epsilon}} \rho_ {\epsilon} \partial_ {x} u _ {\epsilon} \\ + \frac {(v - u _ {\epsilon})}{\theta} \rho_ {\epsilon} M _ {u _ {\epsilon}} (\partial_ {t} u _ {\epsilon} + u _ {\epsilon} \partial_ {x} u _ {\epsilon}) + \rho_ {\epsilon} M _ {u _ {\epsilon}} \frac {| v - u _ {\epsilon} | ^ {2}}{\theta} \partial_ {x} u _ {\epsilon} \\ = \frac {(v - u _ {\epsilon})}{\theta} \rho_ {\epsilon} M _ {u _ {\epsilon}} \left\{\int (v - u _ {\epsilon}) g _ {\epsilon} \mathrm{d} v + \theta \frac {1}{\rho_ {\epsilon}} \partial_ {x} \rho_ {\epsilon} - u _ {\epsilon} \partial_ {x} u _ {\epsilon} \right\} \\ + \rho_ {\epsilon} M _ {u _ {\epsilon}} \left(\frac {| v - u _ {\epsilon} | ^ {2}}{\theta} - 1\right) \partial_ {x} u _ {\epsilon} \\ = L _ {u _ {\epsilon}} (g _ {\epsilon}) - \epsilon (\partial_ {t} + v \partial_ {x}) g _ {\epsilon}. \\ \end{array}
$$

From now on, we neglect the last $\mathcal{O}(\epsilon)$ terms and thus obtain a relation that defines $g_{\epsilon}$ by inverting $L_{u_{\epsilon}}$ as we are going to detail now. Multiplying and integrating over v, we find

$$
\int v g _ {\epsilon} \mathrm{d} v = \int (v - u _ {\epsilon}) g _ {\epsilon} \mathrm{d} v = \frac {1}{r _ {\epsilon}} \left(\rho_ {\epsilon} u _ {\epsilon} \partial_ {x} u _ {\epsilon} - \theta \partial_ {x} \rho_ {\epsilon}\right). \tag {3.18}
$$

Hence, we are led to

$$
L _ {u _ {\epsilon}} (g _ {\epsilon}) = \frac {\theta (v - u _ {\epsilon}) M _ {u _ {\epsilon}}}{r _ {\epsilon}} \left(\theta \partial_ {x} \rho_ {\epsilon} - \rho_ {\epsilon} u _ {\epsilon} \partial_ {x} u _ {\epsilon}\right) + \rho_ {\epsilon} M _ {u _ {\epsilon}} \left(d f r a c | v - u _ {\epsilon} | ^ {2} \theta - 1\right) \partial_ {x} u _ {\epsilon}.
$$

Observe that the integral with respect to $v$ of all terms in the right-hand side vanishes. Bearing in mind that

$$
L _ {0} (v M _ {0}) = - v M _ {0} \qquad \mathrm{and} \qquad L _ {0} \left(\left(\frac {v ^ {2}}{\theta} - 1\right) M _ {0}\right) = - 2 \left(\frac {v ^ {2}}{\theta} - 1\right) M _ {0},
$$

we obtain

$$
g _ {\epsilon} = - \frac {1}{2} \left(\frac {| v - u _ {\epsilon} | ^ {2}}{\theta} - 1\right) \rho_ {\epsilon} M _ {u _ {\epsilon}} \partial_ {x} u _ {\epsilon} - \frac {1}{\theta} \frac {(v - u _ {\epsilon}) M _ {u _ {\epsilon}}}{r _ {\epsilon}} \left(\theta \partial_ {x} \rho_ {\epsilon} - \rho_ {\epsilon} u _ {\epsilon} \partial_ {x} u _ {\epsilon}\right).
$$

For further purposes, observe that

$$
\begin{array}{l} \int v ^ {2} g _ {\epsilon} \mathrm{d} v = \int (v - u _ {\epsilon}) ^ {2} g _ {\epsilon} \mathrm{d} v + 2 u _ {\epsilon} \int v g _ {\epsilon} \mathrm{d} v \tag {3.10} \\ = \frac {2 u _ {\epsilon}}{r _ {\epsilon}} \left(\rho_ {\epsilon} u _ {\epsilon} \partial_ {x} u _ {\epsilon} - \theta \partial_ {x} \rho_ {\epsilon} +\right) - \theta \rho_ {\epsilon} \partial_ {x} u _ {\epsilon}. \\ \end{array}
$$

We are now going back to the hydrodynamic system (3.16)-(3.17), where we similarly get rid of terms of order higher than $\mathcal{O}(\epsilon)$ . To this end, we introduce a convenient change of variables by setting

$$
w _ {\epsilon} := r _ {\epsilon} u _ {\epsilon} + \epsilon \int v g _ {\epsilon} \mathrm{d} v.
$$

Moreover, we shall replace the quantities arising in the previous expression by their first order approximations:

$$
\partial_ {x} u _ {\epsilon} \quad \rightsquigarrow - \frac {w _ {\epsilon}}{r _ {\epsilon} ^ {2}} \partial_ {x} \rho_ {\epsilon} + \frac {1}{r _ {\epsilon}} \partial_ {x} w _ {\epsilon},
$$

$$
u _ {\epsilon} \partial_ {x} u _ {\epsilon} \rightsquigarrow \frac {w _ {\epsilon}}{r _ {\epsilon} ^ {2}} \partial_ {x} w _ {\epsilon} - \frac {w _ {\epsilon} ^ {2}}{r _ {\epsilon} ^ {3}} \partial_ {x} \rho_ {\epsilon},
$$

and

$$
\epsilon \int v g _ {\epsilon} \mathrm{d} v \quad \rightsquigarrow \quad I _ {1, \epsilon} = \frac {\epsilon}{r _ {\epsilon}} \left(\frac {\rho_ {\epsilon} w _ {\epsilon}}{r _ {\epsilon} ^ {2}} \partial_ {x} w _ {\epsilon} - \frac {\rho_ {\epsilon} w _ {\epsilon} ^ {2}}{r _ {\epsilon} ^ {3}} \partial_ {x} \rho_ {\epsilon} - \theta \partial_ {x} \rho_ {\epsilon}\right),
$$

$$
\epsilon \int v ^ {2} g _ {\epsilon} \mathrm{d} v \rightsquigarrow I _ {2, \epsilon} = - \epsilon \theta \rho_ {\epsilon} \left(\frac {1}{r _ {\epsilon}} \partial_ {x} w _ {\epsilon} - \frac {w _ {\epsilon}}{r _ {\epsilon} ^ {2}} \partial_ {x} \rho_ {\epsilon}\right) + \frac {2 w _ {\epsilon}}{r _ {\epsilon}} I _ {1, \epsilon},
$$

where the last two expressions should be compared to $(3.18)$ and $(3.19)$ , respectively. Therefore, based on these approximations, equality $(3.16)$ leads to

$$
\begin{array}{l} \partial_ {t} \rho_ {\epsilon} + \partial_ {x} \left(\frac {\rho_ {\epsilon} w _ {\epsilon}}{r _ {\epsilon}}\right) = \epsilon \partial_ {x} \left\{\left(\frac {\rho_ {\epsilon}}{r _ {\epsilon}} - 1\right) I _ {1, \epsilon} \right\} \tag {3.20} \\ = \epsilon \partial_ {x} \left\{\left(\frac {\rho_ {\epsilon} w _ {\epsilon} ^ {2}}{r _ {\epsilon} ^ {5}} + \frac {\theta}{r _ {\epsilon} ^ {2}}\right) \partial_ {x} \rho_ {\epsilon} - \frac {\rho_ {\epsilon} w _ {\epsilon}}{r _ {\epsilon} ^ {4}} \partial_ {x} w _ {\epsilon} \right\}. \\ \end{array}
$$

Next, for relation (3.17), approximating $u_{\epsilon}^{2}$ by $\frac{w_{\epsilon}^{2}}{r_{\epsilon}^{2}} - \frac{2\epsilon w_{\epsilon}^{2}}{r_{\epsilon}^{2}} I_{1,\epsilon}$ , we get

$$
\begin{array}{l} \partial_ {t} w _ {\epsilon} + \partial_ {x} \left(\frac {w _ {\epsilon} ^ {2}}{r _ {\epsilon}} + \theta \rho_ {\epsilon}\right) = - \epsilon \left(I _ {2, \epsilon} - \frac {2 w _ {\epsilon}}{r _ {\epsilon}} I _ {1, \epsilon}\right) \tag {3.21} \\ = \epsilon \partial_ {x} \left(- \frac {\theta \rho_ {\epsilon} w _ {\epsilon}}{r _ {\epsilon} ^ {2}} \partial_ {x} \rho_ {\epsilon} + \frac {\theta \rho_ {\epsilon}}{r _ {\epsilon}} \partial_ {x} w _ {\epsilon}\right). \\ \end{array}
$$

Dropping for shortness the dependence with respect to $\epsilon$ , we end up with the second-order system in the variable $\mathcal{W} = (\rho, w)$ which is

$$
\partial_ {t} \mathcal {W} + \partial_ {x} F (\mathcal {W}) = \epsilon \partial_ {x} \bigl \{\mathbf {D} (\mathcal {W}) \partial_ {x} \mathcal {W} \bigr \} \tag {3.22}
$$

with the flux $F$ given in (3.2) and the diffusion matrix $\mathbf{D}$ defined as

$$
\mathbf {D} (\mathscr {W}) := \mathbf {D} _ {0} (\mathscr {W}) + \theta   \mathbf {D} _ {1} (\mathscr {W}), \tag {3.23}
$$

where

$$
\mathbf {D} _ {0} (\mathcal {W}) := \frac {\rho w}{r ^ {5}} \left( \begin{array}{c c} w & - r \\ 0 & 0 \end{array} \right) = \frac {\rho u}{r ^ {3}} \left( \begin{array}{c c} u & - 1 \\ 0 & 0 \end{array} \right)
$$

and

$$
\mathbf {D} _ {1} (\mathcal {W}) := \frac {1}{r ^ {2}} \left( \begin{array}{c c} 1 & 0 \\ - \rho w & \rho   r \end{array} \right) = \frac {1}{r} \left( \begin{array}{c c} 1 / r & 0 \\ - \rho u & \rho \end{array} \right)
$$

Remark 3.3. Since system (3.4) is not invariant under Galilean transformations, the same curse occurs for the extended model (3.22). Moreover, it can be easily checked that invariance with respect to space reversal also holds for such a higher order system.

Once more, recalling [30, Corollary 2.2] and having already verified that $d^{2}\eta\,dF$ is symmetric, it is enough to show that, choosing X := $d^{2}\eta$ , the modified diffusion term XD is positive definite. Indeed, using the shorthand notation $\chi = 1 + \rho r$ , we compute the composition

$$
\mathbf {X D} = \frac {1}{\rho r ^ {4}} \left( \begin{array}{c c} \rho^ {2} u ^ {4} + \theta (1 + \chi) \rho r u ^ {2} + \theta^ {2} r ^ {2} & - \rho u (\rho u ^ {2} + \theta \chi r) \\ - \rho u (\rho u ^ {2} + \theta \chi r) & \rho^ {2} (u ^ {2} + \theta r ^ {2}) \end{array} \right)
$$

which we observe to be symmetric too. Moreover, the trace $\mathrm{tr}(\mathbf{X}\mathbf{D})$ is clearly strictly-positive for $\rho > 0$ and $\theta > 0$ . By the Binet Theorem for determinants, there holds

$$
\det (\mathbf {X D}) = \det \mathbf {X} \cdot \det \mathbf {D} = \frac {\theta}{\rho r} \left(\frac {\theta \rho^ {2} u ^ {2}}{r ^ {3}} + \frac {\theta^ {2} \rho}{r ^ {2}} - \frac {\theta \rho^ {2} u ^ {2}}{r ^ {3}}\right) = \frac {\theta^ {3}}{r ^ {3}},
$$

having used the explicit form of $\mathbf{D}$ given in (3.23). Therefore, we infer that $\mathbf{XD}$ is symmetric and positive-definite for $\theta > 0$ .

We summarize our findings in a concise statement.

Proposition 3.4. For any $\theta > 0$ , then system (3.22) with the flux F given in (3.2) and the diffusion matrix D as in (3.23) is strictly stable in the sense of Definition 2.3.

Next, having already verified the validity of Liu's entropy conditions, we directly apply [30, Corollary 2] to establish the existence of weak viscous shocks, i.e. a solution to the two-dimensional ODE system

$$
\epsilon \mathbf {D} (\mathrm{W}) \frac {\mathrm{dW}}{\mathrm{dy}} = F (\mathrm{W}) - F (\mathcal {W} _ {*}) - c (\mathrm{W} - \mathcal {W} _ {*}), \tag {3.24}
$$

satisfying the asymptotic conditions

$$
\lim _ {y \to - \infty} \mathrm{W} (y) = \mathscr {W} _ {*}, \quad \lim _ {y \to + \infty} \mathrm{W} (y) = \mathscr {W} _ {\times} \tag {3.25}
$$

with the propagation speed $c$ given by the Rankine-Hugoniot conditions (3.6) and $\mathcal{W}_{\times}$ sufficiently close to $\mathcal{W}_{*}$ .

We summarize our result in a synthetic statement whose proof follows from the results taken from [30] together with the discussion relative to the validity of Liu's entropy condition (3.11).

Theorem 3.5. Let the triple $(\mathcal{W}_*, \mathcal{W}_x, c)$ be such that the Rankine-Hugoniot conditions (3.6) is satisfied. The strictly stable system (vE) supports weak shock profiles - i.e. there exists $\delta > 0$ such that if $|\mathcal{W}_x - \mathcal{W}_*| \leqslant \delta$ there exists a function $y \mapsto W^\epsilon(y)$ with $\sup_{y \in \mathbb{R}} |\mathrm{W}^\epsilon(y) - \mathcal{W}_*| \leqslant \delta$ , solution to (3.24) with asymptotics (3.25)- if and only Liu's criterion (3.11) is satisfied, that is, $\bar{\rho} < \rho_*$ for the sign + in the choice of $c$ , and $\rho_* < \bar{\rho}$ for the sign -.

Using the appropriate unknowns (specifically, the entropy variables) is crucial to obtain the existence result stated in Theorem 3.5. Different coordinates could support incorrect conclusions. Among others, a detailed discussion on stability properties of weak propagation fronts proved in Theorem 3.5 can be found in [45].

3.5. A few remarks on the stability estimate. Let us go back to the discussion in subsection 2.4 to further illustrate some relevant implications in the case of the viscous Burgers fluid-particle problem (vB). At first sight, even if tempting, requiring D in (1.11) to be parabolic in the sense of Definition 2.7 involves (unphysical) limitations on the temperature, as shown in the following claim.

Proposition 3.6. Let D be defined in (3.23) and set $\Lambda := \sqrt{\rho ru^{2}}$ . The symmetric part $D_{sym}$ of D is strictly positive definite if and only if

$$
\theta \in \left\{ \begin{array}{l l} (\theta_ {1}, + \infty) & \quad i f \quad 0 \leqslant \Lambda \leqslant 2, \\ (\theta_ {1}, \theta_ {2}) & \quad i f \quad \Lambda > 2, \end{array} \right.
$$

with $\theta_{1} = \theta_{1}(\mathcal{U})\coloneqq r^{-2}\Lambda /\bigl (\Lambda +2\bigr)$ and $\theta_{2} = \theta_{2}(\mathcal{U})\coloneqq r^{-2}\Lambda /\bigl (\Lambda -2\bigr)$ .

![](images/eda462858981bc4b664fb6154ff919fd5c68a4530252dad1086f67ba082fc016.jpg)

<details>
<summary>area</summary>

| λ   | θ (lower curve) | θ (upper curve) |
| --- | --------------- | --------------- |
| 0   | 0.0             | 2.5             |
| 2   | 0.2             | 1.8             |
| 4   | 0.3             | 1.2             |
| 6   | 0.35            | 0.8             |
| 8   | 0.4             | 0.6             |
| 10  | 0.45            | 0.5             |
| 12  | 0.48            | 0.45            |
| 14  | 0.5             | 0.4             |
</details>

FIGURE 3. Admissible region in the $(\Lambda, \theta)$ -plane where $\mathbf{D}_{\mathrm{sym}}$ is strictly positive definite for the choice $\rho = 1$ .

This has to be compared to Proposition 3.4, concluding that the notion of parabolicity provided in Definition 2.7 is not the appropriate notion to investigate the stability of viscous perturbations of hyperbolic problems. On the one hand, as explained in Section 2.4 it is not enough to obtain stability estimates which are uniform with respect to $\epsilon$ . On the other hand, it might involve irrelevant restrictions on the parameters of the problem.

Proof. It is readily seen that the trace of the matrix $D_{sym}$ , that is

$$
\operatorname{tr} \left(\mathbf {D} _ {\mathrm{sym}}\right) = \operatorname{tr} \left(\mathbf {D}\right) = r ^ {- 3} \left\{\rho u ^ {2} + \theta r (1 + \rho r) \right\},
$$

is positive for any $\rho\geqslant0$ and $\theta>0$ . The determinant of the symmetric part $D_{sym}$ can be regarded as a second-order polynomial with respect to $\theta$ :

$$
P (\theta) = \frac {1}{4} \rho r ^ {- 3} Q (\theta) \qquad \mathrm{where} \quad Q (\theta) := (4 - \Lambda^ {2}) \theta^ {2} + 2 r ^ {- 2} \Lambda^ {2} \theta - r ^ {- 4} \Lambda^ {2}.
$$

Since the reduced discriminant of $Q$ is $\Delta / 4 = 4\Lambda^2 / r^4$ , we infer the factorization

$$
Q (\theta) = \left\{(2 - \Lambda) \theta + \Lambda / r ^ {2} \right\} \left\{(2 + \Lambda) \theta - \Lambda / r ^ {2} \right\}.
$$

In particular, the symmetric matrix $D_{sym}$ is strictly definite positive if and only if $Q(\theta) > 0$ providing the above restrictions on the parameter $\theta$ . ☐

# 4. FLOWING REGIME FOR THE EULER FLUID-PARTICLE SYSTEM

A more realistic model couples the evolution of the particles, with the Euler equation for the carrier fluid. Namely, we consider

$$
\partial_ {t} f _ {\epsilon} + v \partial_ {x} f _ {\epsilon} = \frac {1}{\epsilon} L _ {u _ {\epsilon}} (f _ {\epsilon})  , \tag {4.1}
$$

coupled to

$$
\left\{ \begin{array}{l} \partial_ {t} n _ {\epsilon} + \partial_ {x} (n _ {\epsilon} u _ {\epsilon}) = 0, \\ \partial_ {t} (n _ {\epsilon} u _ {\epsilon}) + \partial_ {x} \left\{n _ {\epsilon} u _ {\epsilon} ^ {2} + p (n _ {\epsilon}) \right\} = - \frac {1}{\epsilon} \int v L _ {u _ {\epsilon}} (f _ {\epsilon}) \mathrm{d} v = \frac {1}{\epsilon} (J _ {\epsilon} - \rho_ {\epsilon} u _ {\epsilon}), \end{array} \right. \tag {4.2}
$$

still with the notation

$$
\rho_ {\epsilon} = \int f _ {\epsilon} \mathrm{d} v \quad \mathrm{and} \quad J _ {\epsilon} = \int v f _ {\epsilon} \mathrm{d} v.
$$

Here the unknown $n_{\epsilon}$ stands for the density of the carrier fluid, and $u_{\epsilon}$ for its velocity field. The pressure function $p = p(n)$ obeys the standard principles of thermodynamics: it is increasing and strictly convex, a typical example being the $\gamma$ -law given in (1.5).

4.1. Derivation and hyperbolicity. Again, as $\epsilon$ goes to 0, we infer heuristically that

$$
f _ {\epsilon} (t, x) \simeq \rho_ {\epsilon} M _ {u _ {\epsilon} (t, x)} (v),
$$

where $M_{u}$ is the Maxwellian distribution introduced in (1.7). Hence, setting $r := n + \rho$ and w := ru, the limiting quantity $\mathcal{W} = (r, \rho, w)$ satisfies at leading order the extended nonlinear system (iE), which has the form (2.2) where the flux F is given by

$$
F (\mathscr {W}) := \left(w, \rho w / r, w ^ {2} / r + p (n) + \theta \rho\right). \tag {4.3}
$$

We refer the reader to [7] for the introduction of this model; further numerical investigation can be found in [8].

Following again the standard approach, we verify that the extended system (iE) is hyperbolic, i.e. the Jacobian $\mathrm{d}F = \mathrm{d}F(\mathcal{W})$ , explicitly given by

$$
\mathrm{d} F = \left( \begin{array}{c c c} 0 & 0 & 1 \\ - \rho w / r ^ {2} & w / r & \rho / r \\ - w ^ {2} / r ^ {2} + p ^ {\prime} & - p ^ {\prime} + \theta & 2 w / r \end{array} \right) = \left( \begin{array}{c c c} 0 & 0 & 1 \\ - \rho u / r & u & \rho / r \\ - u ^ {2} + p ^ {\prime} & - p ^ {\prime} + \theta & 2 u \end{array} \right)
$$

is such that

$$
\det (\mathrm{d} F - \lambda \mathbf {I}) = - (\lambda - u) \left\{(\lambda - u) ^ {2} - (n p ^ {\prime} + \theta \rho) / r \right\}
$$

so that the eigenvalues are real, being explicitly given by

$$
\lambda = \lambda_ {0} = u \quad \text { and } \quad \lambda_ {\pm} = u \pm \sqrt {(n p ^ {\prime} + \theta \rho) / r}. \tag {4.4}
$$

Remark 4.1. Set $(y, s) = (x - u_{0}t, t)$ , corresponding to $(\partial_{x}, \partial_{t}) = (\partial_{y}, \partial_{s} - u_{0}\partial_{y})$ and set $v := u - u_{0}$ . The first two equations in (iE) are invariant with respect to Galilean transformations. Indeed, there holds

$$
\partial_ {t} r + \partial_ {x} (r u) = \partial_ {s} r - u _ {0} \partial_ {y} r + \partial_ {y} \bigl \{r (v + u _ {0}) \bigr \} = \partial_ {s} r + \partial_ {y} (r v),
$$

with an analogous computations for the unknown $\rho$ . Concerning the third equation, introducing the total pressure $P := p + \theta\rho$ , there holds

$$
\begin{array}{l} \partial_ {t} (r u) + \partial_ {x} (r u ^ {2} + P) = \partial_ {s} \bigl \{r (v + u _ {0}) \bigr \} - u _ {0} \partial_ {y} \bigl \{r (v + u _ {0}) \bigr \} + \partial_ {y} \bigl \{r (v + u _ {0}) ^ {2} + P \bigr \} \\ = \partial_ {s} (r v) + u _ {0} \partial_ {s} r - u _ {0} \partial_ {y} (r v) - u _ {0} ^ {2} \partial_ {y} r + \partial_ {y} \bigl \{r (v ^ {2} + 2 u _ {0} v + u _ {0} ^ {2}) + P \bigr \} \\ = \partial_ {s} (r v) + \partial_ {y} (r v ^ {2} + P) - 2 u _ {0} \partial_ {y} (r v) - u _ {0} ^ {2} \partial_ {y} r + 2 u _ {0} \partial_ {y} (r v) + u _ {0} ^ {2} \partial_ {y} r \\ = \partial_ {s} (r v) + \partial_ {y} (r v ^ {2} + P), \\ \end{array}
$$

showing that the hyperbolic system (iE) is invariant with respect to Galilean transformations. In addition, it can also be shown that the above system is invariant under space reversal, the proof being very similar to the one for the reduced system (iB).

In parallel with Proposition 3.2, we are now interested in a more precise classification of the characteristic fields for the conservation law system (iE).

Proposition 4.2. Let assumption (1.4) be satisfied. Then, for any $\theta \geqslant 0$ , system (iE) is strictly hyperbolic with one linearly degenerate field and two genuinely nonlinear fields whenever $n > 0$ and $\rho, \theta \geqslant 0$ or $n = 0$ and $\rho, \theta > 0$ .

Proof. To start with, let us compute $\nabla_{\mathscr{W}}\lambda$ for $\lambda\in\{\lambda_{0},\lambda_{\pm}\}$ . Upon computations, we infer

$$
\nabla_ {\mathcal {W}} \lambda_ {0} = \left(- \frac {w}{r ^ {2}}, 0, \frac {1}{r}\right) \quad \mathrm{and} \quad \nabla_ {\mathcal {W}} \lambda_ {\pm} = \left(- \frac {w}{r ^ {2}} \pm \frac {n p ^ {\prime \prime} r + \rho (p ^ {\prime} - \theta)}{2 d r ^ {2}}, \mp \frac {p ^ {\prime} + n p ^ {\prime \prime} - \theta}{2 d r}, \frac {1}{r}\right),
$$

where $d := \sqrt{(np' + \theta\rho)/r}$ . Relying on the Galilean invariance, we may reduce to the case u = 0 (corresponding to w = 0), hence upon computations, we infer $\lambda_{0} = 0$ and $\lambda_{\pm} = \pm d$ together with

$$
\nabla_ {\mathcal {W}} \lambda_ {0} = \left(0, 0, \frac {1}{r}\right) \quad \mathrm{and} \quad \nabla_ {\mathcal {W}} \lambda_ {\pm} = \left(\pm \frac {n p ^ {\prime \prime} r + \rho (p ^ {\prime} - \theta)}{2 d r ^ {2}}, \mp \frac {p ^ {\prime} + n p ^ {\prime \prime} - \theta}{2 d r}, \frac {1}{r}\right).
$$

Right eigenvectors relative to $\lambda_{0}$ are proportional to the vector $\mathbf{r}_{0} := (p' - \theta, p', 0)^{\intercal}$ . Since

$$
\nabla_ {\mathcal {W}} \lambda_ {0} \cdot \mathbf {r} _ {0} = 0 \cdot (p ^ {\prime} - \theta) + 0 \cdot p ^ {\prime} + \frac {1}{r} \cdot 0 = 0,
$$

the field $\lambda_0$ is linearly degenerate.

Right eigenvectors relative to eigenvalues $\lambda_{\pm}$ are proportional to $\mathbf{r}_{\pm} := (1, \rho/r, \pm d)^{\intercal}$ . Therefore, there holds

$$
\begin{array}{l} \nabla_ {\mathcal {W}} \lambda_ {\pm} \cdot \mathbf {r} _ {\pm} = \pm \frac {n p ^ {\prime \prime} r + \rho (p ^ {\prime} - \theta)}{2 d r ^ {2}} \cdot 1 \mp \frac {p ^ {\prime} + n p ^ {\prime \prime} - \theta}{2 d r} \cdot \frac {\rho}{r} \pm \frac {1}{r} \cdot d \\ = \pm \left\{\frac {n p ^ {\prime \prime} r + \rho (p ^ {\prime} - \theta) - (n p ^ {\prime \prime} + p ^ {\prime} - \theta) \rho}{2 d r ^ {2}} + \frac {d}{r} \right\} = \pm \frac {n ^ {2} p ^ {\prime \prime} + 2 n p ^ {\prime} + 2 \theta \rho}{2 d r ^ {2}} \neq 0, \\ \end{array}
$$

for any $n > 0$ and $\rho, \theta \geqslant 0$ or $n = 0$ and $\rho, \theta > 0$ . In particular, the characteristic fields $\lambda_{\pm}$ are genuinely nonlinear in such a regime.

4.2. Shock solutions. To investigate discontinuous solutions, we again take advantage of relations (3.8). Having fixed a state $(\rho, n, u) \neq (\rho_{*}, n_{*}, u_{*})$ , the Rankine-Hugoniot conditions associated to system (iE) read

$$
c [   [ \rho ]   ] = [   [ \rho u ]   ], \quad c [   [ n ]   ] = [   [ n u ]   ], \quad c [   [ r u ]   ] = [   [ r u ^ {2} + \theta \rho + p (n) ]   ]. \tag {4.5}
$$

Lemma 4.3. The following implications hold true.

i. If one among the quantities $[\rho]$ , $[n]$ , $[r]$ and $c - u_{*}$ is zero then $[u] = 0$ .   
ii. If $[u]=0$ and $(\llbracket\rho\rrbracket,\llbracket n\rrbracket,\llbracket r\rrbracket)\neq(0,0,0)$ , then $c=c_{0}:=u_{*}$ .

Proof. i. There holds $c[\rho] = [\rho u] = [\rho]u_* + \rho[u]$ , hence

$$
[ [ \rho ] ] (c - u _ {*}) = \rho [ [ u ] ],
$$

and the conclusion follows. A similar proof holds for $n$ and $r = \rho + n$ , observing that, summing up the equations for $\rho$ and $n$ , there holds $\partial_t r + \partial_x(ru) = 0$ and $c[[r]] = [[ru]]$ .

ii. Since $c[\rho] = [\rho u] = [\rho]u_{*}$ , the conclusion is trivial if $[\rho] \neq 0$ . A similar argument can be invoked if $[n] \neq 0$ and $[r] \neq 0$ using the analogous relation for n and r. □

If $\llbracket \rho \rrbracket \neq 0$ and $\llbracket n \rrbracket \neq 0$ , then, equations (4.5) are equivalent to

$$
c = \frac {\llbracket \rho u \rrbracket}{\llbracket \rho \rrbracket} = \frac {\llbracket n u \rrbracket}{\llbracket n \rrbracket} = \frac {\llbracket r u ^ {2} + \theta \rho + p (n) \rrbracket}{\llbracket r u \rrbracket}. \tag {4.6}
$$

As proved in the following result, such shock solutions enjoy Liu's entropy condition under appropriate standard assumptions on the pressure $p$ .

Proposition 4.4. If $[u] \neq 0$ then the speed $c$ , given in the equalities (4.6), satisfies Liu's entropy condition.

Proof. From the second equality in (4.6), we infer $\llbracket nu\rrbracket\llbracket\rho\rrbracket = \llbracket n\rrbracket\llbracket\rho u\rrbracket$ , which, after a straightforward computation, gives $nu\rho_{*} + n_{*}u_{*}\rho = n\rho_{*}u_{*} + u_{*}\rho u$ . In turn, the latter reduces to $(n\rho_{*} - n_{*}\rho)\llbracket u\rrbracket = 0$ so that $n\rho_{*} = n_{*}\rho$ . Therefore, we obtain

$$
\rho = n \rho_ {*} / n _ {*} \quad \text { and } \quad [   [ \rho ]   ] = [   [ n ]   ] \rho_ {*} / n _ {*}. \tag {4.7}
$$

Recalling the identity $r = n + \rho$ , from (4.6) it also follows

$$
[ [ r u ^ {2} + \theta \rho + p ] ] [ [ \rho ] ] = [ [ \rho u ] ] [ [ r u ] ]
$$

with a similar relation holding for n in place of $\rho$ , so that, summing up,

$$
\llbracket r u ^ {2} + \theta \rho + p \rrbracket \llbracket r \rrbracket = \llbracket r u \rrbracket^ {2}. \tag {4.8}
$$

The first term on the lefthand side of $(4.8)$ can be rewritten as

$$
[ [ r u ^ {2} + \theta \rho + p ] ] = r [ [ u ] ] ^ {2} + 2 r u _ {*} [ [ u ] ] + [ [ r ] ] u _ {*} ^ {2} + \theta [ [ \rho ] ] + [ [ p ] ].
$$

Similarly, there holds $\llbracket ru\rrbracket^2 = (r[\llbracket u\rrbracket + \llbracket r\rrbracket u_*)^2$ . Hence, plugging into (4.8), we infer

$$
\begin{array}{l} r [   [ r ]   ] [   [ u ]   ] ^ {2} + 2 r [   [ r ]   ] u _ {*} [   [ u ]   ] + [   [ r ]   ] ^ {2} u _ {*} ^ {2} + [   [ r ]   ] \left\{\theta [   [ \rho ]   ] + [   [ p ]   ] \right\} \\ = r ^ {2} [   [ u ]   ] ^ {2} + 2 r [   [ r ]   ] u _ {*} [   [ u ]   ] + [   [ r ]   ] ^ {2} u _ {*} ^ {2}, \\ \end{array}
$$

that is,

$$
\llbracket u \rrbracket^ {2} = \frac {\llbracket r \rrbracket}{r _ {*} r} \left(\theta \llbracket \rho \rrbracket + \llbracket p \rrbracket\right).
$$

Taking advantage of $(4.7)$ , we infer

$$
\llbracket u \rrbracket^ {2} = \frac {\llbracket n \rrbracket^ {2}}{r _ {*} n} \left(\frac {\theta \rho_ {*}}{n _ {*}} + \frac {\llbracket p \rrbracket}{\llbracket n \rrbracket}\right). \tag {4.9}
$$

The right-hand side is non-negative provided p is a non-decreasing function, which makes this relation consistent. In particular, there holds

$$
\left| \frac {[ [ u ] ]}{[ [ n ] ]} \right| = \left\{\frac {1}{r _ {*} n} \left(\frac {\theta \rho_ {*}}{n _ {*}} + \frac {[ [ p ] ]}{[ [ n ] ]}\right) \right\} ^ {1 / 2}
$$

and, as a consequence,

$$
c = u _ {*} + n \frac {[ [ u ] ]}{[ [ n ] ]} = u _ {*} \pm \left\{\frac {n}{r _ {*}} \left(\frac {\theta \rho_ {*}}{n _ {*}} + \frac {[ [ p ] ]}{[ [ n ] ]}\right) \right\} ^ {1 / 2} =: c _ {\pm} (n).
$$

Differentiating $c_{+}$ with respect to n, we deduce

$$
\partial_ {n} c _ {+} = \frac {1}{2} \left\{r _ {*} n \left(\frac {\theta \rho_ {*}}{n _ {*}} + \frac {[ [ p ] ]}{[ [ n ] ]}\right) \right\} ^ {- 1 / 2} \left\{\frac {\theta \rho_ {*}}{n _ {*}} + \frac {[ [ p ] ]}{[ [ n ] ]} + n \frac {\mathrm{d}}{\mathrm{d} n} \frac {[ [ p ] ]}{[ [ n ] ]} \right\}.
$$

Since p is strictly convex, there holds

$$
\frac {\mathrm{d}}{\mathrm{d} n} \frac {[ [ p ] ]}{[ [ n ] ]} = \frac {\mathrm{d}}{\mathrm{d} n} \left\{\frac {p (n) - p (n _ {*})}{n - n _ {*}} \right\} = \frac {p (n _ {*}) - p (n) - p ^ {\prime} (n) (n _ {*} - n)}{(n _ {*} - n) ^ {2}} > 0.
$$

In particular, Liu's condition is satisfied for $c_{+}$ since $p'' > 0$ .

A similar computation can be used to prove the same property for $c_{-}$ .

![](images/df8e7e348dbd0c4ce72efb1fee7155b31554fc582d5e72dc3611842334c3dab9.jpg)

# 4.3. Entropy for the inviscid Euler fluid-particle system. Similarly to the (vB) case, the kinetic-fluid formulation suggests the functional

$$
\zeta (\mathcal {W}) = \frac {w ^ {2}}{2 r} + \Pi (n) + \theta \rho \ln \rho \quad \mathrm{with} \quad \Pi (n) := \int_ {0} ^ {n} \int_ {0} ^ {s} \frac {1}{\varsigma} \frac {\mathrm{d} p}{\mathrm{d} \varsigma} (\varsigma) \mathrm{d} \varsigma \mathrm{d} s
$$

as an entropy for system (iE), see [7]. For later use, we stress the identity $\Pi'' = p'/n$ .

In the special case of isentropic flows with pressure p given by the standard $\gamma$ -law, i.e. $p(n) = Cn^{\gamma}$ with $\gamma > 1$ , there holds

$$
\Pi (n) = C \gamma \int_ {0} ^ {n} \int_ {0} ^ {s} \varsigma^ {\gamma - 2} \mathrm{d} \varsigma \mathrm{d} s = \frac {C \gamma}{\gamma - 1} \int_ {0} ^ {n} s ^ {\gamma - 1} \mathrm{d} \varsigma \mathrm{d} s = \frac {C n ^ {\gamma}}{\gamma - 1}.
$$

The gradient $\nabla_{W}\zeta$ of the entropy $\zeta$ is explicitly given by

$$
\nabla_ {\mathcal {W}} \zeta (\mathcal {W}) ^ {\intercal} = \left(- w ^ {2} / 2 r ^ {2} + \Pi^ {\prime}, - \Pi^ {\prime} + \theta (1 + \ln \rho), w / r\right)
$$

$$
= \left(- u ^ {2} / 2 + \Pi^ {\prime}, - \Pi^ {\prime} + \theta (1 + \ln \rho), u\right),
$$

while the hessian $\mathrm{d}_\mathcal{W}^2\zeta$ is

$$
\mathrm{d} _ {\mathcal {W}} ^ {2} \zeta (\mathcal {W}) = \left( \begin{array}{c c c} w ^ {2} / r ^ {3} + \Pi^ {\prime \prime} & - \Pi^ {\prime \prime} & - w / r ^ {2} \\ - \Pi^ {\prime \prime} & \Pi^ {\prime \prime} + \theta / \rho & 0 \\ - w / r ^ {2} & 0 & 1 / r \end{array} \right) = \left( \begin{array}{c c c} u ^ {2} / r + p ^ {\prime} / n & - p ^ {\prime} / n & - u / r \\ - p ^ {\prime} / n & p ^ {\prime} / n + \theta / \rho & 0 \\ - u / r & 0 & 1 / r \end{array} \right).
$$

As before, tedious computations confirm that $X := d_{\psi}^{2}\zeta$ symmetrizes the Jacobian dF of the flux dF of the hyperbolic system of conservation laws (iE).

4.4. Viscous corrections leading to (vE). Again, we derive the second-order corrections associated to (iE) by using the Chapman-Enskog expansion. Namely, the function

$$
g _ {\epsilon} := \frac {1}{\epsilon} \left(f _ {\epsilon} - \rho_ {\epsilon} M _ {u _ {\epsilon}}\right)
$$

satisfies

$$
\begin{array}{l} L _ {u _ {\epsilon}} \left(g _ {\epsilon}\right) = \epsilon \left\{\partial_ {t} + v \partial_ {x} \right\} g _ {\epsilon} + M _ {u _ {\epsilon}} \left\{\partial_ {t} \rho_ {\epsilon} + \partial_ {x} \left(\rho_ {\epsilon} u _ {\epsilon}\right) \right\} + \left(\frac {| v - u _ {\epsilon} | ^ {2}}{\theta} - 1\right) \rho_ {\epsilon} M _ {u _ {\epsilon}} \partial_ {x} u _ {\epsilon} \tag {4.10} \\ + \frac {v - u _ {\epsilon}}{\theta} \rho_ {\epsilon} M _ {u _ {\epsilon}} \bigl \{\rho_ {\epsilon} (\partial_ {t} u _ {\epsilon} + u _ {\epsilon} \partial_ {x} u _ {\epsilon}) + \theta \partial_ {x} \rho_ {\epsilon} \bigr \}. \\ \end{array}
$$

Integrating the kinetic equation yields

$$
\partial_ {t} \rho_ {\epsilon} + \partial_ {x} (\rho_ {\epsilon} u _ {\epsilon}) + \epsilon \partial_ {x} \int v g _ {\epsilon}   \mathrm{d} v = 0. \tag {4.11}
$$

Hence the first two terms in the right-hand side of (4.10) contributes only to the $\mathcal{O}(\epsilon)$ correction. Next, by using system (4.2), we get

$$
\rho_ {\epsilon} (\partial_ {t} u _ {\epsilon} + u _ {\epsilon} \partial_ {x} u _ {\epsilon}) = \frac {\rho_ {\epsilon}}{n _ {\epsilon}} \left\{\partial_ {t} (n _ {\epsilon} u _ {\epsilon}) + \partial_ {x} (n _ {\epsilon} u _ {\epsilon} ^ {2}) \right\} = \frac {\rho_ {\epsilon}}{n _ {\epsilon}} \left\{- \partial_ {x} p + \int (v - u _ {\epsilon}) g _ {\epsilon} \mathrm{d} v \right\}.
$$

Therefore, we arrive at

$$
\begin{array}{l} L _ {u _ {\epsilon}} (g _ {\epsilon}) = \left(\frac {| v - u _ {\epsilon} | ^ {2}}{\theta} - 1\right) \rho_ {\epsilon} M _ {u _ {\epsilon}} \partial_ {x} u _ {\epsilon} \\ + \frac {v - u _ {\epsilon}}{\theta} \rho_ {\epsilon} M _ {u _ {\epsilon}} \left(- \frac {\rho_ {\epsilon}}{n _ {\epsilon}} \partial_ {x} p + \frac {\rho_ {\epsilon}}{n _ {\epsilon}} \int v g _ {\epsilon} \mathrm{d} v + \theta \partial_ {x} \rho_ {\epsilon}\right) + \mathcal {O} (\epsilon). \\ \end{array}
$$

Again, let us set $r_{\epsilon} := \rho_{\epsilon} + n_{\epsilon}$ . Next, we multiply by v and integrate in order to obtain a simple relation for $\int vg_{\epsilon} dv$ , deducing

$$
\int v g _ {\epsilon} \mathrm{d} v = \frac {n _ {\epsilon}}{r _ {\epsilon}} \left(\frac {\rho_ {\epsilon}}{n _ {\epsilon}} \partial_ {x} p - \theta \partial_ {x} \rho_ {\epsilon}\right) + \mathcal {O} (\epsilon)
$$

and, consequently,

$$
g _ {\epsilon} = - \frac {1}{2 \theta} \left(| v - u _ {\epsilon} | ^ {2} - \theta\right) \rho_ {\epsilon} M _ {u _ {\epsilon}} \partial_ {x} u _ {\epsilon} - \frac {v - u _ {\epsilon}}{\theta r _ {\epsilon}} \rho_ {\epsilon} M _ {u _ {\epsilon}} \left\{\theta n _ {\epsilon} \partial_ {x} \rho_ {\epsilon} - \rho_ {\epsilon} \partial_ {x} p \right\} + \mathcal {O} (\epsilon). \tag {4.12}
$$

As a matter of fact, we have

$$
\int v ^ {2} g _ {\epsilon}   \mathrm{d} v = - \theta \rho_ {\epsilon} \partial_ {x} u _ {\epsilon} + 2 u _ {\epsilon} \int v g _ {\epsilon}   \mathrm{d} v + \mathcal {O} (\epsilon). \tag {4.13}
$$

Finally, we express the conservation of the total momentum

$$
\partial_ {t} w _ {\epsilon} + \partial_ {x} \left\{r _ {\epsilon} u _ {\epsilon} ^ {2} + p + \theta \rho_ {\epsilon} + \epsilon \int v ^ {2} g _ {\epsilon}   \mathrm{d} v \right\} = 0  , \tag {4.14}
$$

where

$$
w _ {\epsilon} := r _ {\epsilon} u _ {\epsilon} + \epsilon \int v g _ {\epsilon} \mathrm{d} v.
$$

We are now going to write the hydrodynamic system, which arises by getting rid of the terms of order higher than $\mathcal{O}(\epsilon)$ . Thus, in the previous expressions we make use of the

following approximations:

$$
u _ {\epsilon} \quad \rightsquigarrow \quad \frac {w _ {\epsilon}}{r _ {\epsilon}} - \frac {\epsilon n _ {\epsilon}}{r _ {\epsilon} ^ {2}} \left(\frac {\rho_ {\epsilon}}{n _ {\epsilon}} \partial_ {x} p - \theta \partial_ {x} \rho_ {\epsilon}\right),
$$

$$
\partial_ {x} u _ {\epsilon} \quad \rightsquigarrow \quad \partial_ {x} \left(\frac {w _ {\epsilon}}{r _ {\epsilon}}\right) = \frac {1}{r _ {\epsilon}} \partial_ {x} w _ {\epsilon} - \frac {w _ {\epsilon}}{r _ {\epsilon} ^ {2}} \partial_ {x} r _ {\epsilon},
$$

and

$$
u _ {\epsilon} ^ {2} \rightsquigarrow \frac {w _ {\epsilon} ^ {2}}{r _ {\epsilon} ^ {2}} - \frac {2 \epsilon n _ {\epsilon} w _ {\epsilon}}{r _ {\epsilon} ^ {2}} \left(\frac {\rho_ {\epsilon}}{n _ {\epsilon}} \partial_ {x} p - \theta \partial_ {x} \rho_ {\epsilon}\right),
$$

$$
\int v ^ {2} g _ {\epsilon} \mathrm{d} v \quad \rightsquigarrow \quad - \theta \rho_ {\epsilon} \partial_ {x} \left(\frac {w _ {\epsilon}}{r _ {\epsilon}}\right) - \frac {2 n _ {\epsilon} w _ {\epsilon}}{r _ {\epsilon} ^ {2}} \left(\frac {\rho_ {\epsilon}}{n _ {\epsilon}} \partial_ {x} p - \theta \partial_ {x} \rho_ {\epsilon}\right).
$$

Based on these approximations, we obtain a second-order system for $\mathcal{W}_{\epsilon} = (r_{\epsilon}, \rho_{\epsilon}, w_{\epsilon})$ in the form (2.1), that is

$$
\partial_ {t} \mathscr {W} _ {\epsilon} + \partial_ {x} F (\mathscr {W} _ {\epsilon}) = \epsilon \partial_ {x} \bigl \{\mathbf {D} (\mathscr {W} _ {\epsilon}) \partial_ {x} \mathscr {W} _ {\epsilon} \bigr \}, \tag {4.15}
$$

where the flux $F$ is given in (4.3) and the diffusion matrix $\mathbf{D}$ is given by (1.12), which can also be decomposed as

$$
\mathbf {D} (\mathscr {W}) = \mathbf {D} _ {0} (\mathscr {W}) + \theta   \mathbf {D} _ {1} (\mathscr {W})  , \tag {4.16}
$$

where, setting $\nu := n / r \in (0,1)$ , there holds

$$
\mathbf {D} _ {0} := \nu (1 - \nu) p ^ {\prime} \left( \begin{array}{c c c} 0 & 0 & 0 \\ - 1 & 1 & 0 \\ 0 & 0 & 0 \end{array} \right) \quad \text {and} \quad \mathbf {D} _ {1} := \left( \begin{array}{c c c} 0 & 0 & 0 \\ 0 & \nu^ {2} & 0 \\ - (1 - \nu) u & 0 & 1 - \nu \end{array} \right). \tag {4.17}
$$

The eigenvalues $\{\beta_{0},\beta_{1},\beta_{2}\}$ of the (triangular) diffusion matrix D are the element of its principal diagonal, viz.

$$
\beta_ {0} := 0, \quad \beta_ {1} := \nu (1 - \nu) p ^ {\prime} + \theta \nu^ {2} \quad \text {and} \quad \beta_ {2} := \theta (1 - \nu).
$$

In particular, they are non-negative and, differently from system (vB), do not depend explicitly on the velocity $u$ .

We can check the invariance with respect to the Galilean change of coordinates of system (4.15). Reformulating with respect to the variable $\mathcal{U} = (r, \rho, u)$ , we end up with

$$
\partial_ {t} G (\mathcal {U}) + \partial_ {x} H (\mathcal {U}) = \epsilon \partial_ {x} \left\{\mathbf {E} (\mathcal {U}) \partial_ {x} \mathcal {U} \right\} \tag {4.18}
$$

with $G(\mathcal{U}) = (r,\rho ,ru)$ , $H(\mathcal{U}) = (ru,\rho u,ru^2 +p + \theta \rho)$ and

$$
\mathbf {E} (\mathcal {U}) := \nu \left( \begin{array}{c c c} 0 & 0 & 0 \\ - 1 & 1 & 0 \\ 0 & 0 & 0 \end{array} \right) + \theta (1 - \nu) \left( \begin{array}{c c c} 0 & 0 & 0 \\ 0 & 1 + \nu & 0 \\ 0 & 0 & r \end{array} \right).
$$

Introducing the variables $(y, s)$ and u as in Remark 4.1, where we proved that the left-hand side is invariant with respect to Galilean transformations, we can also show that the whole system (4.18) preserve the same property, as a consequence of the independence of E with respect to the velocity variable u.

Since one of the eigenvalue of $\mathbf{D}$ is zero, the induced dissipation is partial and some additional stability is required. In the present setting, the Kawashima-Shizuta condition -stating that there is no right eigenvector of $\mathrm{d}F$ in the kernel of $\mathbf{D}-$ holds (see [28, 42]).

Indeed, focusing without loss of generality on the case $u = 0$ , the eigenvectors are proportional to $\mathbf{r} = (1, \rho / r, \lambda)^{\intercal}$ where $\lambda$ is a non-zero eigenvalue of $\mathrm{d}F$ or to $\mathbf{r} = (1, 1 - \theta / p', 0)^{\intercal}$ when $\lambda = 0$ . Computing $\mathbf{D}\mathbf{r}$ for $\lambda \neq 0$ gives $(\mathbf{D}\mathbf{r})_3 = \theta \lambda (1 - \nu) \neq 0$ for $\theta > 0$ and $\nu < 1$ . Similarly, for $\lambda = 0$ , there holds $(\mathbf{D}\mathbf{r})_2 = -\theta^2 \nu^2 / p' \neq 0$ for $\theta > 0$ and $\nu > 0$ . Summarizing, (vE) satisfies the Kawashima-Shizuta stability condition for strictly positive temperature $\theta$ and $\nu$ in the open interval (0, 1) corresponding to $\rho$ and $n$ strictly positive.

For later use, let us also explore in more details the temperature-less regime $\theta=0$ . In the case $\lambda\neq0$ , the third component $(\mathbf{D}\mathbf{r})_{3}$ is null. Nevertheless, the second component $(\mathbf{D}\mathbf{r})_{2}$ is equal to $-\nu^{2}(1-\nu)p^{\prime}$ which is strictly negative if $\nu\in(0,1)$ . Hence the Kawashima-Shizuta condition holds for $\lambda\neq0$ . Differently, for $\lambda=0$ , there holds $\mathbf{D}\mathbf{r}_{0}=(0,-\nu(1-\nu)p^{\prime}+\nu(1-\nu)p^{\prime},0)^{\intercal}=\mathbf{0}$ and the condition is not satisfied.

Going further, we aim to show that the matrix $d_{W}^{2}\zeta D$ is symmetric. With this target, we rewrite the hessian $d_{W}^{2}\zeta$ of the entropy $\zeta$ (again with u=0, thanks to the Galilean invariance) in terms of the scalar quantity $\nu=n/r$ , obtaining $d_{W}^{2}\zeta=X_{0}+\theta X_{1}$ where

$$
\mathbf {X} _ {0} := \frac {1}{n} \left( \begin{array}{c c c} p ^ {\prime} & - p ^ {\prime} & 0 \\ - p ^ {\prime} & p ^ {\prime} & 0 \\ 0 & 0 & \nu (1 - \nu) \end{array} \right) \qquad \text {and} \qquad \mathbf {X} _ {1} := \frac {\theta}{\rho} \left( \begin{array}{c c c} 0 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{array} \right).
$$

Then, we compute the matrix product

$$
\mathbf {X} \mathbf {D} = (\mathbf {X} _ {0} + \theta \mathbf {X} _ {1}) (\mathbf {D} _ {0} + \theta \mathbf {D} _ {1}) = \mathbf {X} _ {0} \mathbf {D} _ {0} + \theta (\mathbf {X} _ {0} \mathbf {D} _ {1} + \mathbf {X} _ {1} \mathbf {D} _ {0}) + \theta^ {2} \mathbf {X} _ {1} \mathbf {D} _ {1}.
$$

Tedious computations bring the following final formulas

$$
\mathbf {X} _ {0} \mathbf {D} _ {0} = \frac {\nu (1 - \nu) (p ^ {\prime}) ^ {2}}{n} \left( \begin{array}{c c c} + 1 & - 1 & 0 \\ - 1 & + 1 & 0 \\ 0 & 0 & 0 \end{array} \right) \quad \mathrm{and} \quad \mathbf {X} _ {1} \mathbf {D} _ {1} = \frac {\nu^ {2}}{\rho} \left( \begin{array}{c c c} 0 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{array} \right),
$$

together with

$$
\mathbf {X} _ {1} \mathbf {D} _ {0} + \mathbf {X} _ {0} \mathbf {D} _ {1} = \frac {1}{r} \left( \begin{array}{c c c} 0 & - \nu p ^ {\prime} & 0 \\ - \nu p ^ {\prime} & 2 \nu p ^ {\prime} & 0 \\ 0 & 0 & 1 - \nu \end{array} \right),
$$

showing the symmetry of the matrix D.

4.5. The temperature-less case. As stated in the Introduction, the case where the Brownian velocity fluctuations are neglected is relevant in many applications. Therefore, let us briefly discuss how the discussion adapts to handle the case $\theta = 0$ : we consider system (4.1)-(4.2) where the Fokker-Planck operator in the right-hand side of (4.1) is replaced by $\partial_{v}\{(v - u_{\epsilon})f_{\epsilon}\}$ . This does not modify the coupling term in (4.2) which is still given by $J_{\epsilon} - \rho_{\epsilon}u_{\epsilon}$ . The “equilibrium state” that makes the stiff terms vanish is now a Dirac mass with respect to the velocity variable

$$
f _ {\epsilon} (t, x, v) \simeq \rho_ {\epsilon} (t, x) \delta_ {v = u _ {\epsilon} (t, x)}.
$$

This modifies the limit equation: since $\int v^{2}f_{\epsilon}\,dv\simeq\rho_{\epsilon}u_{\epsilon}^{2}$ , there is no pressure term induced by the kinetic part of the equation and the limit equation becomes

$$
\left\{ \begin{array}{l} \partial_ {t} n + \partial_ {x} (n u) = 0, \\ \partial_ {t} \rho + \partial_ {x} (\rho u) = 0, \\ \partial_ {t} (r u) + \partial_ {x} \left\{r u ^ {2} + p (n) \right\} = 0, \end{array} \right. \tag {4.19}
$$

instead of (iE). Therefore, we can simply use the formula for the flux F and the Jacobian matrix dF by setting $\theta = 0$ . In particular, the eigenvalues of dF become

$$
\lambda = \lambda_ {0} = u, \quad \lambda_ {\pm} = u \pm \sqrt {n p ^ {\prime} / r}. \tag {4.20}
$$

Accordingly we can set $\theta = 0$ in the expressions of subsection 4.2.

We shall see that the conclusion is essentially the same for the viscous correction, but the computation should be performed with some caution. The rationale consists in using the fact that $M_{u}$ , defined in (1.7), converges to a Dirac mass $\delta_{v=u}$ as $\theta \rightarrow 0^{+}$ in the sense of distributions. Accordingly, we also have

$$
\lim _ {\theta \to 0 ^ {+}} \partial_ {v} M _ {u} = - \lim _ {\theta \to 0 ^ {+}} \frac {1}{\theta} (v - u) M _ {u} = \delta_ {v = u} ^ {\prime},
$$

$$
\lim _ {\theta \to 0 ^ {+}} \partial_ {v v} ^ {2} M _ {u} = \lim _ {\theta \to 0 ^ {+}} \frac {1}{\theta^ {2}} \left(| v - u | ^ {2} - \theta\right) M _ {u} = \delta_ {v = u} ^ {\prime \prime},
$$

both being weak limits. Thus, as $\theta \rightarrow 0^{+}$ in the right-hand side of (4.10) and in the remainder in (4.12), we infer that $g_{\epsilon} := -\frac{\rho_{\epsilon}}{r_{\epsilon}} \partial_{x} p(n_{\epsilon}) \delta'_{v=u_{\epsilon}}$ is such that

$$
\partial_ {v} \left\{(v - u _ {\epsilon}) g _ {\epsilon} \right\} = \frac {\rho_ {\epsilon}}{r _ {\epsilon}} \partial_ {x} p (n _ {\epsilon}) \delta_ {v = u _ {\epsilon}} ^ {\prime}, \qquad \int g _ {\epsilon} \mathrm{d} v = 0, \qquad \int v g _ {\epsilon} \mathrm{d} v = \frac {\rho_ {\epsilon}}{r _ {\epsilon}} \partial_ {x} p (n _ {\epsilon}).
$$

Furthermore, the second order moment becomes

$$
\int v ^ {2} g _ {\epsilon} \mathrm{d} v = \frac {2 \rho_ {\epsilon} u _ {\epsilon}}{r _ {\epsilon}} \partial_ {x} p (n _ {\epsilon}).
$$

By using the above formula, we obtain the closed equation (4.15) with the diffusion matrix (4.16) where we simply set $\theta = 0$ , i.e. $D = D_{0}$ .

Let us stress that when $\theta = 0$ the entropy $\zeta$ is convex but not strictly convex, since we can easily check that $X = X_{0}$ is a singular matrix. In particular, this precludes the possibility of applying the symmetrization method presented in Proposition 2.6.

4.6. Small-amplitude shock profiles analysis. As in the previous computations, we may consider, without loss of generality, a co-moving frame such that $u_{*}=0$ . To apply the result [38, Theorem 4.1], we focus on a genuinely nonlinear field $\lambda$ for system (vE), hence excluding the field $\lambda_{0}$ (see Proposition 4.2). For definiteness, let us concentrate on the case $\lambda=\lambda_{+}$ , the case $\lambda=\lambda_{-}$ being similar. For later convenience, let us recall the identity

$$
\lambda_ {+} ^ {2} = \nu p ^ {\prime} + \theta (1 - \nu) \quad \text { with } \quad \nu = n / r \in (0, 1)  . \tag {4.21}
$$

We are going to use the following result, stated and proved in [38], reported here for reader's convenience in a variation fitting the present context (see [12] for an alternative formulation).

Theorem 4.5 (Theorem 4.1, [38]). Let $\ell_{+}$ and $\mathbf{r}_{+}$ denote left and right eigenvectors of the matrix $\mathbf{A}$ relative to the eigenvalue $\lambda_{+}$ , respectively. In addition, let us assume

i. $\mathbf{D}(\mathcal{W})$ has constant rank in a neighborhood of $\mathcal{W}_{*}$ ;

ii. there holds $\ell_{+}\mathbf{D}\mathbf{r}_{+}(\mathcal{W}_{*})\neq 0;$

iii. the operator $\mathbf{B}(\xi) := i\xi(\mathbf{A} - \lambda_{+}\mathbf{I}) - \mathbf{D}$ is one-to-one on $\mathbb{C}Z$ for all $\xi \in \mathbb{R}$ , i.e. $\operatorname{Ker} \mathbf{B}(\xi)|_{\mathbb{C}Z} = \{0\}$ , where

$$
Z := \left\{\mathbf {v} \in \mathbb {R} ^ {3}: (\mathbf {A} - \lambda_ {+} \mathbf {I}) \mathbf {v} \in \operatorname{Ran} \mathbf {D} \right\}; \tag {4.22}
$$

Then, the following are equivalent

I. there holds $\ell_{+}\mathbf{D}\mathbf{r}_{+}(\mathcal{W}_{*}) > 0$ ;

II. there exists $\delta > 0$ so that if $\mathcal{W}_{*}$ and $\mathcal{W}_{\times}$ are such that $|\mathcal{W}_{*} - \mathcal{W}_{\times}| < \delta$ and the Rankine-Hugoniot condition holds for some speed $c$ , there exists a shock profile connecting $\mathcal{W}_{*}$ to $\mathcal{W}_{\times}$ if and only if Liu's entropy criterion (2.4) is satisfied.

Verification of the above assumptions leads to the proof of existence of shock profiles in the small amplitude regime.

Theorem 4.6. Let $\theta \geqslant 0$ and let $\mathcal{W}_{*}$ and $\mathcal{W}_{\times}$ are such that the Rankine-Hugoniot condition is satisfied for some speed $c$ . Then there exists $\delta > 0$ so that there exists a shock profile solution to (4.15) connecting $\mathcal{W}_{*}$ to $\mathcal{W}_{\times}$ with $|\mathcal{W}_{*} - \mathcal{W}_{\times}| < \delta$ .

Proof. The result is proved if the assumption of Theorem 4.5 are satisfied. Without loss of generality, we consider the case u = 0 by using once more the invariance with respect to Galilean transformations.

Case $\theta = 0$ . For zero temperature, the matrix D reduces to $D_{0}$ defined in (4.17). Also, a triple of right/left eigenvectors of A relative to the eigenvalue $\lambda_{k}$ is given by $\mathbf{r}_{k} = (1, 1 - \nu, \lambda_{k})$ and $\ell_{k} = (p', -p' + \theta, \lambda_{k})$ where $k \in \{0, \pm\}$ . Condition i. in Theorem 4.5 is clearly satisfied since $\operatorname{Ran}\mathbf{D}(\mathcal{W})$ coincides with $\operatorname{Span}\{\mathbf{e}_{2}\}$ for any W where $\{e_{1}, e_{2}, e_{3}\}$ is the canonical basis of $R^{3}$ . As a consequence, $\operatorname{Ran}\mathbf{D}(\mathcal{W})$ has rank one.

Next, we state that $Z$ coincides with $\mathrm{Span}\{\mathbf{r}_+\}$ . Indeed, let us consider the vector $\mathbf{v} = (x,y,z)\in \mathbb{R}^3$ such that $(\mathbf{A} - \lambda_{+}\mathbf{I})\mathbf{v}\in \mathrm{Ran}\mathbf{D}$ . Then there holds

$$
(\mathbf {A} - \lambda_ {+} \mathbf {I}) \mathbf {v} = \left( \begin{array}{c c c} - \lambda_ {+} & 0 & 1 \\ 0 & - \lambda_ {+} & \rho / r \\ p ^ {\prime} & - p ^ {\prime} & - \lambda_ {+} \end{array} \right) \left( \begin{array}{c} x \\ y \\ z \end{array} \right) = \left( \begin{array}{c} - d x + z \\ - d y + \rho z / r \\ p ^ {\prime} x - p ^ {\prime} y - d z \end{array} \right) = \alpha \mathbf {e} _ {2},
$$

for some $\alpha \in \mathbb{R}$ . Plugging the relation $z = dx$ , into the third component, we deduce the identity $y = \rho x / r$ . Finally, we insert both equations for $z$ and $y$ , into the second component getting

$$
- d y + \frac {1}{r} \rho z = - \frac {1}{r} d \rho x + \frac {1}{r} d \rho x = \alpha
$$

which implies $\alpha = 0$ . In particular, the set Z coincides with the one-dimensional eigenspace of the eigenvalue $\lambda_{+}$ , that is, $Z = \operatorname{Ker}(\mathbf{A} - \lambda_{+}\mathbf{I})$ . Thus, we are required to analyze the kernel of the operator $\mathbf{B}(\xi)$ restricted to Z, that is, we look for vectors $v = \alpha r_{+}$ for some $\alpha \in C$ such that $\mathbf{B}(\xi)\mathbf{v} = -\alpha \mathbf{D}\mathbf{r}_{+} = \mathbf{0}$ . Since the Kawashima–Shizuta condition is satisfied also for $\theta = 0$ , $Dr_{+} \neq 0$ and, therefore, $\alpha = 0$ . As a consequence, hypothesis iii. is satisfied.

Finally, let us show that conditions iii./I. are also verified. Indeed, there holds

$$
\boldsymbol {\ell} _ {+} \mathbf {D r} _ {+} = \nu (1 - \nu) p ^ {\prime} \left( \begin{array}{c c c} p ^ {\prime} & - p ^ {\prime} & \lambda_ {+} \end{array} \right) \left( \begin{array}{c c c} 0 & 0 & 0 \\ - 1 & 1 & 0 \\ 0 & 0 & 0 \end{array} \right) \binom{1}{1 - \nu} = \nu^ {2} (1 - \nu) (p ^ {\prime}) ^ {2} > 0.
$$

Case $\theta > 0$ . For strictly positive temperatures, it is readily verified that $\operatorname{Ran} \mathbf{D}(\mathcal{W}) = \operatorname{Span}\{\mathbf{e}_2, \mathbf{e}_3\}$ for any $\mathcal{W}$ . Hence, hypothesis $\mathbf{i}$ . holds.

A vector $\mathbf{v} = (x,y,z)$ lies in $Z$ if and only if $z = \lambda_{+}x$ . Therefore the action of the linear operator $\mathbf{B}(\xi)$ is described by

$$
\mathbf {B} (\xi) \mathbf {v} = i \xi \left( \begin{array}{c} 0 \\ (1 - \nu) \lambda_ {+} x - \lambda_ {+} y \\ (p ^ {\prime} - \lambda_ {+} ^ {2}) x + (- p ^ {\prime} + \theta) y \end{array} \right) + \left( \begin{array}{c} 0 \\ - \nu (1 - \nu) p ^ {\prime} x + \nu \{(1 - \nu) p ^ {\prime} + \theta \nu \} y \\ \theta \lambda_ {+} (1 - \nu) x \end{array} \right)
$$

which can be rewritten as a reduced two dimensional system with coefficient matrix

$$
\begin{array}{l} \mathbf {M} := \left( \begin{array}{c c} i \xi (1 - \nu) \lambda_ {+} - \nu (1 - \nu) p ^ {\prime} & - i \xi \lambda_ {+} + \nu (1 - \nu) p ^ {\prime} + \theta \nu^ {2} \\ i \xi (p ^ {\prime} - \lambda_ {+} ^ {2}) + \theta \lambda_ {+} (1 - \nu) & - i \xi (p ^ {\prime} - \theta) \end{array} \right) \\ = \left( \begin{array}{c c} i \xi (1 - \nu) \lambda_ {+} - \nu (1 - \nu) p ^ {\prime} & - i \xi \lambda_ {+} + \nu (1 - \nu) p ^ {\prime} + \theta \nu^ {2} \\ i \xi (1 - \nu) (p ^ {\prime} - \theta) + \theta \lambda_ {+} (1 - \nu) & - i \xi (p ^ {\prime} - \theta) \end{array} \right) \\ \end{array}
$$

The real part of the determinant of $\mathbf{M}$ is

$$
\mathrm{Re} (\det \mathbf {M}) = - \theta \lambda_ {+} \{\nu (1 - \nu) p ^ {\prime} + \theta \nu^ {2} \} (1 - \nu) <   - \theta \lambda_ {+} \nu (1 - \nu) ^ {2} p ^ {\prime}
$$

which is strictly negative for any $\theta > 0$ and $\nu \in (0,1)$ . Hence, the linear transformation M is a one-to-one correspondence, exhibiting the validity of iii.

Finally, we compute explicitly the value of $\ell_{+}Dr_{+}>0$ . Since

$$
\mathbf {D r} _ {+} = \left( \begin{array}{c c c} 0 & 0 & 0 \\ - \nu (1 - \nu) p ^ {\prime} & \nu (1 - \nu) p ^ {\prime} + \theta \nu^ {2} & 0 \\ 0 & 0 & \theta (1 - \nu) \end{array} \right) \left( \begin{array}{c} 1 \\ 1 - \nu \\ \lambda_ {+} \end{array} \right) = (1 - \nu) \left( \begin{array}{c} 0 \\ - \nu^ {2} (p ^ {\prime} - \theta) \\ \theta \lambda_ {+} \end{array} \right),
$$

there holds

$$
\boldsymbol {\ell} _ {+} \mathbf {D r} _ {+} = (1 - \nu) \nu^ {2} (p ^ {\prime} - \theta) ^ {2} + \theta^ {2} \lambda_ {+} \geqslant \theta^ {2} \lambda_ {+} > 0.
$$

Thus, since Liu's entropy condition is satisfied (see Proposition 4.4), we deduce the existence of small amplitude shock profiles as a consequence of Theorem 4.1 in [38]. $\square$

Furthermore, conditions described in $[23]$ and $[33]$ are satisfied, so that the small amplitude shock profiles are also asymptotically stable in some appropriate Sobolev space.

# 5. LARGE AMPLITUDE PROFILES FOR VISCOUS EULER FLUID-PARTICLE SYSTEM

In this final Section, we continue the analysis relative to the existence of shock profiles for (vE) in the large amplitude regime. Such a choice is dictated by the fact that the model has the additional feature of being invariant with respect to Galilean transformations. As a consequence, we can assume, without loss of generality, that the chosen reference frame is comoving with the wave, i.e. the speed c is equal to zero. Hence, after the straightforward rescaling $x \mapsto y := x/\epsilon$ , we search for a solution $W = (r, \rho, w)$ of

$$
\mathbf {D} (\mathrm{W}) \frac {\mathrm{dW}}{\mathrm{dy}} = F (\mathrm{W}) - F (\mathscr {W} _ {*}), \tag {5.1}
$$

where the flux F has been introduced in (4.3) and the diffusion matrix $D = D_{0} + \theta D_{1}$ with $D_{0}$ and $D_{1}$ defined in (4.17). Moreover, we assume that the solution W is subjected to far-end states, denoted by $W_{*}$ and $W_{x}$ , which are related by the Rankine–Hugoniot conditions (4.5). Whether the far-end state of the asymptotic values $W_{*}$ and $W_{x}$ is reached at $-\infty$ or at $+\infty$ will be made precise further on.

Since the first row of $\mathbf{D}$ vanishes, the first equation in (5.1) imposes that $w$ is constant:

$$
w = w _ {*} := r _ {*} u _ {*}.
$$

We are thus led to a $2 \times 2$ differential system for the pair $(r, \rho)$ given by

$$
\left\{ \begin{array}{l} - \frac {n p ^ {\prime} (n) \rho}{r ^ {2}} \frac {\mathrm{d} r}{\mathrm{dy}} + \left\{\frac {n p ^ {\prime} (n) \rho}{r ^ {2}} + \frac {\theta n ^ {2}}{r ^ {2}} \right\} \frac {d \rho}{\mathrm{dy}} = \left(\frac {\rho}{r} - \frac {\rho_ {*}}{r _ {*}}\right) w _ {*}, \\ - \theta \frac {\rho w _ {*}}{r ^ {2}} \frac {\mathrm{d} r}{\mathrm{dy}} = \frac {w _ {*} ^ {2}}{r} + p (n) + \theta \rho - \frac {w _ {*} ^ {2}}{r _ {*}} - p (n _ {*}) - \theta \rho_ {*}, \end{array} \right.
$$

which, on its turn, is equivalent to a system for the pair $(r,n)$ that is

$$
\left\{ \begin{array}{l} - \frac {\theta n ^ {2}}{r ^ {2}} \frac {\mathrm{d} r}{\mathrm{dy}} + \left(\frac {\theta n ^ {2}}{r ^ {2}} + \frac {\rho   n p ^ {\prime} (n)}{r ^ {2}}\right) \frac {\mathrm{d} n}{\mathrm{dy}} = w _ {*} \left(\frac {n}{r} - \frac {n _ {*}}{r _ {*}}\right), \\ - \frac {\theta \rho w _ {*}}{r ^ {2}} \frac {\mathrm{d} r}{\mathrm{dy}} = \frac {w _ {*} ^ {2}}{r} + p (n) + \theta \rho - \frac {w _ {*} ^ {2}}{r _ {*}} - p (n _ {*}) - \theta \rho_ {*} \end{array} \right. \tag {5.2}
$$

Any solution to the dynamical system (5.2) asymptotically converging to $W_{*}$ and $W_{x}$ corresponds to a (smooth) shock profile for (4.15).

5.1. Analysis of the temperature-less case. When $\theta = 0$ and $u_{*} \neq 0$ , system (5.2) degenerates to the scalar differential equation for $n$

$$
\frac {(r - n) p ^ {\prime} (n)}{r ^ {2}} \frac {\mathrm{d} n}{\mathrm{dy}} = u _ {*} \left(\frac {r _ {*}}{r} - \frac {n _ {*}}{n}\right), \tag {5.3}
$$

coupled with the identity

$$
\frac {r}{r _ {*}} = \frac {r _ {*} u _ {*} ^ {2}}{r _ {*} u _ {*} ^ {2} + p _ {*} - p (n)}. \tag {5.4}
$$

where $p_{*} := p(n_{*})$ . We bear in mind that the function $r = n + \rho$ has the meaning of a hybrid density, being the sum of the densities of the carrier and the disperse phases, denoted by n and $\rho$ , respectively. The system degenerating to a single equation, we replaced $\rho$ by r - n in (5.3). Accordingly, r is required to satisfy the admissibility constraint r > n for any $n \in (0, \infty)$ , since $\rho = r - n > 0$ . Under this constraint, one sees at once that the equilibrium states of (5.3) satisfy $r_{*}/r = n_{*}/n$ .

To make our computations on system (5.3)-(5.4) easier to follow, we will introduce rescaled variables. However, we will formulate our main theorem in the natural variables. Let

$$
\mathrm{n} := \frac {n}{n _ {*}} \quad \text { and } \quad \mathrm{r} := \frac {r}{r _ {*}}, \tag {5.5}
$$

together with the auxiliary parameters

$$
\tau := \frac {n _ {*}}{r _ {*}} \in (0, 1)  , \qquad \kappa := \frac {r _ {*} u _ {*} ^ {2}}{p _ {*}} \in (0, \infty)  , \qquad \kappa_ {*} := \frac {n _ {*} p _ {*} ^ {\prime}}{p _ {*}} \in (0, \infty)  , \tag {5.6}
$$

where $p_{*}^{\prime}=p^{\prime}(n_{*})$ . The parameter $\tau$ describes the ratio between the density of the disperse phase and the corresponding total density. In particular, in term of the rescaled variables, the discussion about the sign of $\rho$ will then concern the one of r- $\tau$ n. The dimensionless number $\kappa$ is reminiscent of the Eckert number in fluid mechanics and it compares the kinetic energy of the mixture to the pressure of the carrier phase. The value $\kappa_{*}$ is a given threshold separating different behaviors for the solution of problem (5.3)–(5.4). Note that, once $n_{*}$ is fixed, $\kappa_{*}$ is completely determined. Moreover, if $\tau$ is fixed, $r_{*}$ is also given. Finally, if additionally $\kappa$ is fixed, the absolute value of $u_{*}$ is determined by the formula

$$
| u _ {*} | = \sqrt {p _ {*} \tau \kappa / n _ {*}}. \tag {5.7}
$$

Finally, let us introduce the rescaled pressure

$$
\mathrm{p(n)} := \frac {p (n _ {*} \mathrm{n})}{p _ {*}}. \tag {5.8}
$$

Note that the function p shares the same monotonicity and convexity of p and that

$$
\mathrm{p} (0) = 0, \quad \mathrm{p} (1) = 1, \quad \mathrm{p} ^ {\prime} (1) = \kappa_ {*}. \tag {5.9}
$$

Taking advantage of the previous definitions, the differential equation (5.3) with constraint (5.4) rewrites as

$$
\left\{ \begin{array}{l} \frac {u _ {*}}{\kappa} \frac {\left(\mathrm{r} - \tau \mathrm{n}\right) \mathrm{p} ^ {\prime} (\mathrm{n})}{\mathrm{r} ^ {2}} \frac {\mathrm{dn}}{\mathrm{dy}} = \mathcal {T} (\mathrm{n}, \mathrm{r}) := \frac {1}{\mathrm{r}} - \frac {1}{\mathrm{n}}, \\ \mathrm{r} = \mathrm{r} _ {\kappa} (\mathrm{n}) := \frac {\kappa}{1 + \kappa - \mathrm{p} (\mathrm{n})}. \end{array} \right. \tag {5.10}
$$

where the function $\mathrm{n} \mapsto \mathrm{r}_{\kappa}(\mathrm{n})$ is defined for $\mathrm{n} \in (0, \bar{\mathrm{n}}(\kappa))$ with $\bar{\mathrm{n}}(\kappa) := \mathrm{p}^{-1}(1 + \kappa)$ .

Lemma 5.1. For any $\kappa > 0$ with $\kappa \neq \kappa_*$ there exists a unique $n(\kappa) \neq 1$ solution to $g_{\kappa}(n) := \mathcal{T}(n, r_{\kappa}(n)) = 0$ . Moreover, the function $\kappa \mapsto n(\kappa)$ is one-to-one from $(0, \infty) \backslash \{\kappa_*\}$ to $(0, \infty) \backslash \{1\}$ with $n(\kappa) < 1$ if and only if $\kappa < \kappa_*$ .

Proof. For $\kappa \neq \kappa_{*}$ , the function $\mathrm{g}_{\kappa}$ is such that

$$
\lim _ {\mathrm{n} \to 0 ^ {+}} \mathrm{g} _ {\kappa} (\mathrm{n}) = - \infty , \qquad \mathrm{g} _ {\kappa} (1) = 0, \qquad \mathrm{g} _ {\kappa} ^ {\prime} (1) = 1 - \frac {\kappa_ {*}}{\kappa} \neq 0, \qquad \mathrm{g} _ {\kappa} (\bar {\mathrm{n}}) = - \frac {1}{\bar {\mathrm{n}}} <   0.
$$

Moreover, the derivative

$$
\mathrm{g} _ {\kappa} ^ {\prime} (\mathrm{n}) = \frac {1}{\mathrm{n} ^ {2}} - \frac {\mathrm{p} ^ {\prime} (\mathrm{n})}{\kappa} \tag {5.11}
$$

is decreasing in n, hence the function $g_{\kappa}$ is concave. (Graphs of the function $g_{\kappa}$ for several values of $\kappa$ are depicted in Fig. 4, in the case of the pressure law (1.5) with $\gamma = 2$ .) In particular, for $\kappa < \kappa_{*}$ , respectively $\kappa > \kappa_{*}$ , there exists a unique value $n \in (0,1)$ , respectively $n \in (1,\bar{n})$ , such that $g_{\kappa}(n) = 0$ .

Conversely, given $n \in (0, +\infty) \setminus \{1\}$ , let $\kappa(n)$ be such that $g_{\kappa}(n) = 0$ . The latter identity can be equivalently written as $n = r_{\kappa}(n) = \kappa / \{1 + \kappa - p(n)\}$ . As a consequence, we infer

$$
\kappa (\mathrm{n}) = \frac {\mathrm{n} \left\{\mathrm{p} (\mathrm{n}) - 1 \right\}}{\mathrm{n} - 1} _ {3 4} \quad \text {for} \quad \mathrm{n} \neq 1. \tag {5.12}
$$

and thus, since (1.4) holds,

$$
\lim _ {n \to 0 ^ {+}} \kappa (n) = 0, \quad \lim _ {n \to 1} \kappa (n) = - 1, \quad \lim _ {n \to + \infty} \kappa (n) = + \infty .
$$

In addition, $n \mapsto \kappa(n)$ is differentiable with respect to n for $n \neq 1$ with derivative

$$
\kappa^ {\prime} (\mathrm{n}) = \frac {\mathrm{n} (\mathrm{n} - 1) \mathrm{p} ^ {\prime} (\mathrm{n}) + 1 - \mathrm{p} (\mathrm{n})}{(\mathrm{n} - 1) ^ {2}}.
$$

Then, applying de l'Hôpital rule, we infer

$$
\lim _ {n \rightarrow 1} \kappa^ {\prime} (n) = \lim _ {n \rightarrow 1} \frac {2 p ^ {\prime} (n) + n p ^ {\prime \prime} (n)}{2} = \kappa_ {*} + \frac {1}{2} p ^ {\prime \prime} (1) > 0,
$$

showing that $\kappa \in C^1(0, +\infty)$ . Moreover, the numerator in the expression for the derivative $\kappa'(n)$ is positive, since it vanishes at $n = 1$ and a further differentiation gives

$$
\frac {\mathrm{d}}{\mathrm{dn}} \left\{\mathrm{n(n-1)p} ^ {\prime} (\mathrm{n}) + 1 - \mathrm{p(n)} \right\} = (\mathrm{n-1}) \left\{2 \mathrm{p} ^ {\prime} (\mathrm{n}) + \mathrm{np} ^ {\prime \prime} (\mathrm{n}) \right\}
$$

which is of the same sign as $n - 1$ and so $n \mapsto \kappa(n)$ is increasing.

Finally, thanks to the strict positivity of $\mathfrak{p}'$ , $\mathrm{p(n)} - 1$ is of the same sign as $n - 1$ and, since $\kappa(n)$ can be rewritten as

$$
\kappa (\mathrm{n}) = \mathrm{p(n)} - 1 + \frac {\mathrm{p(n)} - 1}{\mathrm{n} - 1},
$$

we deduce that $\kappa (\mathrm{n}) > \mathrm{p}(\mathrm{n}) - 1$ . Therefore, $\mathrm{n} < \bar{\mathrm{n}} (\kappa (\mathrm{n}))$ and we conclude that there exists a unique $\kappa$ such that $\mathrm{g}_{\kappa}(\mathrm{n}) = 0$ .

![](images/32ea87b1d5044fd122ad0dd26f656f911e0cc7c94a6eb940ae64b427f48f1db2.jpg)

<details>
<summary>line</summary>

| n    | κ = 0.5 | κ = 0.75 | κ = 1   | κ = 2   | κ = 3   | κ = 4   | κ = 6   |
|------|---------|----------|---------|---------|---------|---------|---------|
| 0.0  | -0.5    | -0.5     | -0.5    | -0.5    | -0.5    | -0.5    | -0.5    |
| 0.5  | 0.25    | 0.25     | 0.25    | 0.25    | 0.25    | 0.25    | 0.25    |
| 1.0  | 0.5     | 0.5      | 0.5     | 0.5     | 0.5     | 0.5     | 0.5     |
| 1.5  | 0.25    | 0.25     | 0.25    | 0.25    | 0.25    | 0.25    | 0.25    |
| 2.0  | 0.0     | 0.0      | 0.0     | 0.0     | 0.0     | 0.0     | 0.0     |
</details>

FIGURE 4. The graph of the function $g_{\kappa}$ in (5.18) for the $\gamma$ -law (1.5) with exponent $\gamma = 2$ . The markers are the same as in Figures 6 and 7.

Let the function h be defined by

$$
\mathrm{h} (\mathrm{n}) := \{\mathrm{np} (\mathrm{n}) \} ^ {\prime} = \mathrm{p} (\mathrm{n}) + \mathrm{np} ^ {\prime} (\mathrm{n})  . \tag {5.13}
$$

In particular, because $h' = 2p' + p'' > 0$ , the function h is strictly increasing together with its inverse $h^{-1}$ . Then, the following function is well-defined for any $\kappa > 0$

$$
\mathrm{n} _ {\#} (\kappa) := \mathrm{h} ^ {- 1} (1 + \kappa) \in (1, \infty). \tag {5.14}
$$

Being h(1) = p(1) + p'(1) = 1 + $\kappa_{*}$ , there holds n $_{\#}$ ( $\kappa_{*}$ ) = 1.

Lemma 5.2. Given $\kappa > 0$ , let $n_{\#} = n_{\#}(\kappa)$ be defined as in (5.14). The function

$$
\tau_ {\#} (\kappa) := \frac {\kappa}{\mathrm{n} _ {\#} ^ {2} \mathrm{p} ^ {\prime} (\mathrm{n} _ {\#})} \tag {5.15}
$$

is such that $0 < \tau_{\#}(\kappa) \leqslant 1$ for all $\kappa > 0$ and $\tau_{\#}(\kappa) = 1$ if and only if $\kappa = \kappa_{*}$ .

Moreover, $\tau_{\#} = \tau_{\#}(\kappa)$ tends to 0 as $\kappa \to 0^{+}$ and as $\kappa \to +\infty$ .

Proof. To begin, let us observe that $n_{\#}(\kappa_{*}) = 1$ and $p'(1) = \kappa_{*}$ so that $\tau_{\#}(\kappa_{*}) = 1$ . The positivity of $\tau_{\#}$ being obvious, let us show that $\tau_{\#} \leqslant 1$ for $\kappa > 0$ , with the equality holding only if $\kappa = \kappa_{*}$ . Indeed, the above inequality is equivalent to

$$
\mathrm{f} (\kappa) := \mathrm{n} _ {\#} ^ {2} \mathrm{p} ^ {\prime} (\mathrm{n} _ {\#}) - \kappa \geqslant 0 \quad \forall \kappa > 0. \tag {5.16}
$$

Note that $f(\kappa_{*}) = p'(1) - \kappa_{*} = 0$ . Differentiating with respect to $\kappa$ , we infer

$$
\mathrm{f} ^ {\prime} (\kappa) = \left\{2 \mathrm{p} ^ {\prime} (\mathrm{n} _ {\#}) + \mathrm{n} _ {\#} \mathrm{p} ^ {\prime \prime} (\mathrm{n} _ {\#}) \right\} \mathrm{n} _ {\#} \mathrm{n} _ {\#} ^ {\prime} - 1 = \frac {\left\{2 \mathrm{p} ^ {\prime} (\mathrm{n} _ {\#}) + \mathrm{n} _ {\#} \mathrm{p} ^ {\prime \prime} (\mathrm{n} _ {\#}) \right\} \mathrm{n} _ {\#}}{\mathrm{h} ^ {\prime} (\mathrm{h} ^ {- 1} (1 + \kappa))} - 1 = \mathrm{n} _ {\#} - 1.
$$

Differentiating again, since $n_{\#}^{\prime}=1/h^{\prime}(n_{\#})>0$ , we conclude that f is strictly convex, its unique minimum being 0 at $\kappa=\kappa_{*}$ . As a consequence, inequality (5.16) holds.

Next, let us observe that $\mathrm{n}_{\#}(0) = \mathrm{h}^{-1}(1) > \mathrm{h}^{-1}(0) = 0$ since $\mathrm{h}(0) = \mathrm{p}(0) = 0$ and $h^{-1}$ is strictly increasing. Hence, $\frac{\tau_{\#}(\kappa)}{\kappa}$ tends to a strictly positive number as $\kappa \to 0^{+}$ and the limit of $\tau_{\#}$ at $\kappa = 0$ is identified.

Concerning the behavior at $+\infty$ , since $\mathrm{h}(+\infty) = +\infty$ , there holds $n_{\#}(+\infty) = +\infty$ , Then, applying de l'Hôpital rule, we obtain

$$
\lim _ {\kappa \to + \infty} \tau_ {\#} (\kappa) = \lim _ {\kappa \to + \infty} \frac {1}{\{\mathrm{n} _ {\#} ^ {2} \mathrm{p} ^ {\prime} (\mathrm{n} _ {\#}) \} ^ {\prime}} = \lim _ {\kappa \to + \infty} \frac {1}{n _ {\#}} \frac {\mathrm{h} ^ {\prime}}{2 \mathrm{p} ^ {\prime} + \mathrm{n} _ {\#} \mathrm{p} ^ {\prime \prime}} = \lim _ {\kappa \to + \infty} \frac {1}{\mathrm{n} _ {\#}} = 0,
$$

completing the proof.

Theorem 5.3. Given $\kappa > 0$ with $\kappa \neq \kappa_{*}$ , let $n_{\times} = n_{*}n_{\times}(\kappa) \neq n_{*}$ be the equilibrium value defined thanks to $n_{\times}(\kappa)$ the solution given by Lemma 5.1. Then, if $\tau < \tau_{\#}(\kappa)$ , problem (5.3)-(5.4) admits monotone solutions $y \mapsto n(y)$ connecting asymptotically $n_{\times}$ to $n_{*}$ with monotonicity related to the sign of $u_{*}$ .

Remark 5.4. The definition of the parameters has practical consequences, for instance for numerical purposes. Choosing $u_{*}$ , $\tau$ and $\kappa$ leads to inverting $n \mapsto \frac{n}{p(n)}$ in order to retrieve $n_{*}$ , which might require additional assumptions on the pressure law, hopefully satisfied by the $\gamma$ -law.

Graph of $\tau_{\#}$   
![](images/29f0a539f4204995821ed124c7a377f1e272745a5d521323176bb94dcd61b9eb.jpg)

<details>
<summary>area</summary>

| x     | y    |
|-------|------|
| 0     | 0    |
| κ*    | 1    |
| k     | ~0.8 |
| >k*   | ~0.6 |
</details>

FIGURE 5. The graph of the function $\tau_{\#}$ in the case of the $\gamma$ -law (1.5) with $\gamma = 2$ . Small shocks are concentrated in a neighborhood of $\kappa = \kappa_{*} = 2$ .

Proof. For $\mathrm{r} - \tau \mathrm{n} \neq 0$ and introducing the new variable $\mathrm{z}$ such that

$$
\frac {\mathrm{d}}{\mathrm{dz}} = \frac {u _ {*} (\mathrm{r} - \tau \mathrm{n})   \mathrm{p} ^ {\prime} (\mathrm{n})}{\kappa r ^ {2}} \frac {\mathrm{d}}{\mathrm{dy}}, \tag {5.17}
$$

problem (5.10) becomes

$$
\frac {\mathrm{d} \mathbf {n}}{\mathrm{d} z} = g _ {\kappa} (\mathbf {n}). \tag {5.18}
$$

where $g_{\kappa}$ is defined as in Lemma 5.1. A straightforward argument, based on the analysis of the sign of function $g_{\kappa}$ , shows the existence of the heteroclinic connection between 1 and $n_{\times}$ for (5.18) for $\kappa \neq \kappa_{*}$ , whenever $r - \tau n > 0$ .

The threshold level $\tau_{\#}$ appears as a consequence of the constraint $r > \tau n$ , indicating that the curve $(n, r_{\kappa})$ lies above $(n, \tau n)$ . Differentiating $r_{\kappa}$ with respect to n, we infer

$$
\frac {\mathrm{dr} _ {\kappa}}{\mathrm{dn}} = \frac {\kappa \mathrm{p} ^ {\prime} (\mathrm{n})}{\left\{1 + \kappa - \mathrm{p} (\mathrm{n}) \right\} ^ {2}},
$$

which is positive and increasing for the properites of p. In particular, $r_{\kappa}$ is convex in $(0,\bar{\mathrm{n}}(\kappa))$ where $\bar{\mathrm{n}}(\kappa)$ has been introduced right after (5.10).

Next let us look for the pair $(n_{\#}, \tau_{\#})$ such that the tangent to the graph of the functions $r_{\kappa}$ is given by the straight line $r = \tau_{\#} n$ . This amounts to searching the solutions of

$$
\mathrm{r} _ {\kappa} (\mathrm{n} _ {\#}) = \frac {\kappa}{1 + \kappa - \mathrm{p} (\mathrm{n} _ {\#})} = \tau_ {\#} \mathrm{n} _ {\#} \quad \mathrm{and} \quad \mathrm{r} _ {\kappa} ^ {\prime} (\mathrm{n} _ {\#}) = \frac {\kappa \mathrm{p} ^ {\prime} (\mathrm{n} _ {\#})}{\{1 + \kappa - \mathrm{p} (\mathrm{n} _ {\#}) \} ^ {2}} = \tau_ {\#}.
$$

Replacing the first identity into the second and simplifying, we get $\mathrm{p}(\mathrm{n}_{\#}) + \mathrm{n}_{\#}\mathrm{p}'(\mathrm{n}_{\#}) = 1 + \kappa$ . Then, we immediately recognize that $\mathrm{n}_{\#} = \mathrm{h}^{-1}(1 + \kappa)$ and $\tau_{\#} = \mathrm{r}_{\kappa}(\mathrm{n}_{\#}) / \mathrm{n}_{\#} = \mathrm{r}_{\kappa}'(\mathrm{n}_{\#})$ which corresponds to the value defined in (5.15). Note also that $\mathrm{g}_{\kappa}(\mathrm{n}_{\#}) = (1 - \tau_{\#}) / \mathrm{n}_{\#} \geqslant 0$ . Summarizing, for $\tau \in (0, \tau_{\#})$ the constraint $r_{\kappa} > \tau n$ is always satisfied and the change of variables is legit.

Conversely, for $\tau \in (\tau_{\#}, 1)$ there exist two values $n_{\ell}, n_{r} \in (0, \bar{n})$ with $n_{\ell} < n_{r}$ and $r_{\kappa}(n_{\ell,r}) = \tau n_{\ell,r}$ , such that the condition $r_{\kappa} > \tau n$ holds if and only if $n \in (0, n_{\ell})$ or

Profiles of $n / n_*$ (plain) and of $r(n) / r_*$ (dotted)   
![](images/d5c42a7a8ade4fcb2d784b3ec238fcf49c610fc4f5ba1a5df17cc8806da3e215.jpg)  
FIGURE 6. Graphs of $n/n_{*}$ (plain) and $r/r_{*} = \mathrm{r}_{\kappa}(n/n_{*})$ (dotted) where n solves problem (5.3)-(5.4) for several values of $\kappa$ such that $\tau < \tau_{\#}$ . The pressure law p is the $\gamma$ -law (1.5) with exponent $\gamma = 2$ .

$n \in (n_{r}, \bar{n})$ . In addition, for $\tau > \tau_{\#}$ , we have

$$
\mathrm{g} _ {\kappa} (\mathrm{n} _ {\ell , r}) = \frac {1}{\mathrm{r} _ {\kappa} (\mathrm{n} _ {\ell , r})} - \frac {1}{\mathrm{n} _ {\ell , r}} = \frac {1 + \kappa - \mathrm{p} (\mathrm{n} _ {\ell , r})}{\kappa} - \frac {1}{\mathrm{n} _ {\ell , r}} = \frac {1 - \tau}{\tau \mathrm{n} _ {\ell , r}} > 0,
$$

so that for $\kappa < \kappa_{*}$ , there holds $0 < n_{\times} < n_{\ell} < n_{r} < 1 < \bar{n}$ . In particular, for $\tau > \tau_{\#}$ and $\kappa < \kappa_{*}$ , the function $\varphi(n) := r_{\kappa}(n) - \tau n$ is negative in the interval $(n_{\ell}, n_{r}) \subset (n_{\times}, 1)$ . Similarly, for $\kappa > \kappa_{*}$ , $\varphi$ is negative in $(n_{\ell}, n_{r}) \subset (1, n_{\times})$ . In both cases, the change of variables (5.17) is not applicable and existence of the connection is precluded since the physical requirement $\rho > 0$ is violated.

Remark 5.5. Figure 6 shows the profiles n (respectively, r) connecting 1 to $n_{\times}$ (resp. 1 to $\mathrm{r}_{\kappa}(\mathrm{n}_{\times})$ ) associated to several values of $\kappa$ for the choice $\mathrm{p}(\mathrm{n}) = \mathrm{n}^{2}$ , illustrating the increasing character of the equilibrium map $\mathrm{n}_{\times} = \mathrm{n}_{\times}(\kappa)$ . This point is emphasized in Figure 7 in the phase portrait corresponding to the same values of $\kappa$ , showing that the orbits are convex. Also, note that $n_{\times}$ and $\mathrm{r}_{\kappa}(\mathrm{n}_{\times})$ do not depend on $\tau$ , but the profiles n and r do, through $r_{*} = n_{*}/\tau$ . The condition $r - n = r_{*}(\mathrm{r} - \tau\mathrm{n}) > 0$ , with $\tau < \tau_{\#}$ , shows as $n \mapsto \tau_{\#}n$ is tangent to the orbit at the point $(\mathrm{n}_{\#}, \mathrm{r}_{\kappa}(\mathrm{n}_{\#}))$ .

Let us also observe that, since $p(1) = 1$ , there holds $r_{\kappa}(1) - \tau = 1 - \tau > 0$ . Hence, small shocks are always admissible also in the case of zero-temperature.

Example 5.6. For the sake of concreteness, let us again consider the pressure given by the $\gamma$ -law (1.5). Incidentally, let us note that $\kappa_{*} = \gamma$ for any positive constant C. Then,

Heteroclinic orbits   
![](images/911d3ec42459bd75b903fdf2b7b6d5c58f56a4c354728a293c096e8ad191e6e7.jpg)

<details>
<summary>line</summary>

| n/n* | κ = 0.5 | κ = 0.75 | κ = 1   |
|------|---------|----------|---------|
| 0.4  | 0.37    | -        | 0.26    |
| 0.5  | -       | 0.42     | -       |
| 0.6  | -       | -        | 0.62    |
| 0.7  | 0.50    | -        | -       |
| 0.8  | -       | 0.65     | 0.75    |
| 1.0  | -       | 0.85     | 1.00    |
</details>

![](images/6162a1a7ff62adc1f394718953908b6630fda10a30b975783a68c770c194977b.jpg)

<details>
<summary>line</summary>

| n/n* | κ = 3 | κ = 4 | κ = 6 |
|------|-------|-------|-------|
| 1.00 | 1.00  | 1.00  | 1.00  |
| 1.25 | 1.25  | 1.25  | 1.25  |
| 1.50 | 1.50  | 1.50  | 1.50  |
| 2.00 | 2.00  | 1.75  | 2.00  |
</details>

FIGURE 7. Orbits connecting $(\mathrm{n}_{\times}, \mathrm{r}_{\kappa}(\mathrm{n}_{\times}))$ to (1,1) for several values of $\kappa$ for the $\gamma$ -law (1.5) with exponent $\gamma = 2$ . The straight line $n \mapsto \tau_{\#}(\kappa)n$ is plotted for each value of $\kappa$ , and the tangent point with the corresponding orbit is indicated. The markers are the same as in Figure 6.

most auxiliary functions can be determined giving the explicit expressions

$$
\mathrm{p} (\mathrm{n}) = \mathrm{n} ^ {\gamma}, \qquad \mathrm{h} (\mathrm{n}) = (1 + \gamma) \mathrm{n} ^ {\gamma}, \qquad \mathrm{h} ^ {- 1} (\mathrm{r}) = \left(\frac {\mathrm{r}}{1 + \gamma}\right) ^ {1 / \gamma}.
$$

Moreover, there holds

$$
\mathrm{n} _ {\#} (\kappa) = \left(\frac {1 + \kappa}{1 + \gamma}\right) ^ {1 / \gamma} \quad \text {and} \quad \tau_ {\#} (\kappa) = \frac {(1 + \gamma) ^ {1 + 1 / \gamma}}{\gamma} \frac {\kappa}{(1 + \kappa) ^ {1 + 1 / \gamma}}
$$

In the special case $\gamma = 2$ , the function $g_{\kappa}$ is a rational function whose factorization is

$$
\mathrm{g} _ {\kappa} (\mathrm{n}) = \frac {1 + \kappa - \mathrm{n} ^ {2}}{\kappa} - \frac {1}{\mathrm{n}} = - \frac {\mathrm{n} ^ {3} - (1 + \kappa) \mathrm{n} + \kappa}{\kappa \mathrm{n}} = - \frac {1}{\mathrm{n}} (\mathrm{n} + \mathrm{n} _ {-}) (\mathrm{n} - 1) (\mathrm{n} - \mathrm{n} _ {\times}),
$$

where $n_{-}$ and $n_{\times}$ are given by

$$
\mathrm{n} _ {-} := \frac {1}{2} \left\{(1 + 4 \kappa) ^ {1 / 2} + 1 \right\}, \qquad \mathrm{n} _ {\times} := \frac {1}{2} \left\{(1 + 4 \kappa) ^ {1 / 2} - 1 \right\}.
$$

Corresponding graphical representations of the function $\varphi$ (defined at the very end of proof of Theorem 5.3) for different choices of $\tau$ are given in Figure 8. Here, the limiting value $\tau_{\#}$ is equal to 1 at $\kappa = \gamma = 2$ and is explicitly represented to show tangency of the graph with the horizontal axis. Above this $\kappa$ -dependent threshold value, the still existing heteroclinic connection from $n_{\times}$ and 1 (corresponding to the connection from $n_{\times}$ and $n_{*}$ ) is not physically admissible since the carrier phase $\rho$ is negative in a neighborhood of both asymptotic states.

![](images/2086b77919427ba6827f24f973c0d4c5e21428d008f7e20c2081b95c3d0e0b20.jpg)

<details>
<summary>line</summary>

| n    | τ = 0.7 | τ = 0.9 | τ = 1   | τ = 1.2 | g     | r(n)/n | r'    |
|------|---------|---------|---------|---------|-------|--------|-------|
| 0.0  | 0.6     | 0.6     | 0.6     | 0.6     | 0.6   | 0.6    | 0.6   |
| 0.5  | 0.3     | 0.4     | -0.8    | -0.2    | 0.4   | -0.8   | 0.4   |
| 1.0  | 0.0     | 0.0     | 0.0     | 0.0     | 1.0   | 1.0    | 1.0   |
| 1.5  | -0.3    | -0.2    | -0.4    | -0.6    | 1.4   | -0.4   | 1.4   |
| 2.0  | -0.6    | -0.4    | -0.6    | -0.8    | 1.6   | -0.6   | 1.6   |
</details>

FIGURE 8. Graphs of functions $\varphi (\mathrm{n}) = \mathrm{r}_{\kappa}(\mathrm{n}) - \tau \mathrm{n}$ in the case of $\gamma$ -law (1.5) with $\gamma = \kappa = 2$ and various choices of $\tau$ . The graph of $\mathrm{g}_{\kappa}$ (with opposite convexity) has been superposed for comparison, as well as the maps $\mathrm{n}\mapsto \mathrm{r}_{\kappa}(\mathrm{n}) / \mathrm{n}$ and $\mathrm{n}\mapsto \mathrm{r}'(\mathrm{n})$ that intersect at $(\mathrm{n}_{\#},\tau_{\#})$ .

5.2. Further scrutiny for positive temperature. Next, we focus on the case $\theta > 0$ with the intention of grasping information from the singular limit behavior $\theta = 0$ . System (5.2) can be equivalently written as

$$
\left\{ \begin{array}{l} \frac {r ^ {2}}{\rho p ^ {\prime} + \theta n} \frac {\mathrm{d} n}{\mathrm{dy}} = \frac {w _ {*}}{n} \left(\frac {n}{r} - \frac {n _ {*}}{r _ {*}}\right) - \frac {w _ {*} n}{\rho} \left\{\frac {1}{r} - \frac {1}{r _ {*}} + \frac {p - p _ {*} + \theta (\rho - \rho_ {*})}{w _ {*} ^ {2}} \right\}, \\ \theta \frac {\mathrm{d} r}{\mathrm{dy}} = - \frac {w _ {*} r ^ {2}}{\rho} \left\{\frac {1}{r} - \frac {1}{r _ {*}} + \frac {p - p _ {*} + \theta (\rho - \rho_ {*})}{w _ {*} ^ {2}} \right\}. \end{array} \right. \tag {5.19}
$$

where $w_{*} = r_{*}u_{*}$ . Next, with same notation as before for n, r, $\tau$ , $\kappa$ , p (see (5.5)-(5.6)) and observing that $\rho = r_{*}(r - \tau n)$ , we set T as in (5.10) and $B_{\epsilon} := B^{0} + \epsilon B^{1}$ where

$$
\mathcal {B} ^ {0} (\mathrm{n}, \mathrm{r}) := \frac {1 + \kappa - \mathrm{p(n)}}{\kappa} - \frac {1}{\mathrm{r}} = \mathrm{g} _ {\kappa} (\mathrm{n}) - \mathcal {T} (\mathrm{n}, \mathrm{r}), \qquad \mathcal {B} ^ {1} (\mathrm{n}, \mathrm{r}) := \frac {1 - \tau - (\mathrm{r} - \tau \mathrm{n})}{\kappa}.
$$

and $\epsilon := \kappa \theta / u_{*}^{2} = r_{*}\theta / p_{*}$ . Then, the renormalized version of system (5.19) is

$$
\left\{ \begin{array}{l l} u _ {*} \frac {\mathrm{dn}}{\mathrm{dy}} = \frac {\kappa \mathrm{r} ^ {2}}{(\mathrm{r} - \tau \mathrm{n}) \mathrm{p} ^ {\prime} (\mathrm{n}) + \varepsilon \tau^ {2} \mathrm{n}} \left\{\mathcal {T} (\mathrm{n}, \mathrm{r}) + \frac {\tau \mathrm{n}}{\mathrm{r} - \tau \mathrm{n}} \mathcal {B} _ {\epsilon} (\mathrm{n}, \mathrm{r}) \right\}, \\ \epsilon u _ {*} \frac {\mathrm{dr}}{\mathrm{dy}} = \frac {\kappa \mathrm{r} ^ {2}}{\mathrm{r} - \tau \mathrm{n}} \mathcal {B} _ {\epsilon} (\mathrm{n}, \mathrm{r}), \end{array} \right.
$$

Note that $B_{\epsilon}$ , varying linearly with respect to $\varepsilon$ , depends also on p (through $B_{0}$ ), on $\tau$ (through $B_{1}$ ) and on $\kappa$ (through both $B^{0}$ and $B^{1}$ ). In addition, we remark that the parameter $\epsilon$ is small also in cases where $\theta$ is of order 1 and $\kappa/u_{*}^{2}$ is small.

Introducing the new variable z such that

$$
\frac {\mathrm{d}}{\mathrm{dz}} = \frac {u _ {*} (\mathrm{r} - \tau \mathrm{n}) \mathrm{p} ^ {\prime} (\mathrm{n}) + \epsilon \tau^ {2} \mathrm{n}}{\kappa \mathrm{r} ^ {2}} \frac {\mathrm{d}}{\mathrm{dy}}, \tag {5.20}
$$

we arrive at the final expression

$$
\left\{ \begin{array}{l} \frac {\mathrm{dn}}{\mathrm{dz}} = \mathcal {T} (\mathrm{n}, \mathrm{r}) + \frac {\tau \mathrm{n}}{\mathrm{r} - \tau \mathrm{n}} \mathcal {B} _ {\epsilon} (\mathrm{n}, \mathrm{r}), \\ \epsilon \frac {\mathrm{dr}}{\mathrm{dz}} = \left(\mathrm{p} ^ {\prime} (\mathrm{n}) + \frac {\epsilon \tau^ {2} \mathrm{n}}{\mathrm{r} - \tau \mathrm{n}}\right) \mathcal {B} _ {\epsilon} (\mathrm{n}, \mathrm{r}). \end{array} \right. \tag {5.21}
$$

Preliminarily, observe that, since $p(1) = 1$ and thus $\mathcal{B}_{\epsilon}(1,1) = 0$ , the pair $(n,r) = (1,1)$ defines an equilibrium solution for (5.21) for any $\varepsilon$ , $\tau$ and $\kappa$ .

Proposition 5.7. Let $\kappa > 0$ and $\tau \in (0,1)$ . Then, for any $\epsilon > 0$ there exists a unique equilibrium point $(n_{\times}^{\epsilon}, r_{\times}^{\epsilon}) \neq (n_{*}, r_{*})$ of system (5.19). Denoting $n_{\times}^{\epsilon} = n_{\times}^{\epsilon} / n_{*}$ and $r_{\times}^{\epsilon} = r_{\times}^{\epsilon} / r_{*}$ , we have $\mathcal{T}(n_{\times}^{\epsilon}, r_{\times}^{\epsilon}) = \mathcal{B}_{\epsilon}(n_{\times}^{\epsilon}, r_{\times}^{\epsilon}) = 0$ . Moreover, the two coefficients $n_{\times,0}$ and $n_{\times,1}$ of the first order Taylor expansion of $n_{\times}^{\epsilon}$ at $\epsilon = 0$ , viz. $n_{\times}^{\epsilon} = n_{\times,0} + \epsilon n_{\times,1} + \mathcal{O}(\epsilon^{2})$ , are

$$
\mathrm{n} _ {\times , 0} = \mathrm{n} _ {\times} \qquad a n d \qquad \mathrm{n} _ {\times , 1} = \frac {1 - \tau}{\kappa} \frac {\mathrm{n} _ {\times} - 1}{\mathrm{g} _ {\kappa} ^ {\prime} (\mathrm{n} _ {\times})} <   0, \tag {5.22}
$$

where $n_{\times}$ is the equilibrium of system (5.10) (as described in Lemma 5.1).

Proof. The pair $(\mathrm{n}_{\times}^{\epsilon},\mathrm{r}_{\times}^{\epsilon})$ solves $\mathcal{T}(\mathrm{n}_{\times}^{\epsilon},\mathrm{r}_{\times}^{\epsilon}) = \mathcal{B}_{\epsilon}(\mathrm{n}_{\times}^{\epsilon},\mathrm{r}_{\times}^{\epsilon}) = 0$ which is equivalent to

$$
\left\{ \begin{array}{c} \mathrm{r} _ {\times} ^ {\epsilon} = \mathrm{n} _ {\times} ^ {\epsilon}, \\ \mathrm{g} _ {\kappa} (\mathrm{n} _ {\times} ^ {\epsilon}) = \frac {\epsilon (1 - \tau)}{\kappa} (\mathrm{n} _ {\times} ^ {\epsilon} - 1), \end{array} \right. \tag {5.23}
$$

referring to the notation in Lemma 5.1. Since $g_{\kappa}(n_{\times}) = 0$ , the zero-th order $n_{\times,0}$ in the expansion with respect to $\epsilon$ of the solution $n_{\times}^{\epsilon}$ coincides with $n_{\times}$ . Moreover, the first order coefficient $n_{\times,1}$ can be obtained from (5.23) by substitution of the expansion and cancellation of the common coefficient $\epsilon$ , that is

$$
\mathrm{n} _ {\times , 1} \mathrm{g} _ {\kappa} ^ {\prime} (\mathrm{n} _ {\times}) = \frac {1 - \tau}{\kappa} (\mathrm{n} _ {\times} - 1)  . \tag {5.24}
$$

which gives the desired equality. Note that $g_{\kappa}^{\prime}(n_{\times})$ cannot vanish simultaneously with $g_{\kappa}(n_{\times})$ since $g_{\kappa}^{\prime}$ is strictly decreasing –see (5.11)– and $g_{\kappa}(1)=g_{\kappa}(n_{\times})=0$ .

Finally, $g_{\kappa}^{\prime}$ being decreasing yields

$$
\frac {\mathrm{g} _ {\kappa} ^ {\prime} (\mathrm{n} _ {\times})}{\mathrm{n} _ {\times} - 1} = \frac {\mathrm{g} _ {\kappa} ^ {\prime} (\mathrm{n} _ {\times}) - \mathrm{g} _ {\kappa} ^ {\prime} (1)}{\mathrm{n} _ {\times} - 1} <   0,
$$

and thus $n_{\times,1}$ is negative.

Remark 5.8. The formation of viscous profiles joining monotonically the equilibrium values is shown in Figure 9, while Figure 10 represents the corresponding heteroclinic orbits in the $(n,r)$ plane. The fact that $n_{x}^{\epsilon}<n_{x}$ for small values of $\epsilon$ is showing well.

These numerical results are given with a purpose reduced to an illustration of the previous discussion, showing a computational evidence for the existence of viscous shock profiles. However, the apparent convexity of the orbits is worth investigating, as is the fact that the sign of $n_{1}$ seems to imply that, if $\kappa = 3$ , $\tau$ might be chosen closer to $\tau_{\#}$ .

Profiles of $n / n_*$ (plain) and $r / r_*$ (dotted)   
![](images/692045105f9d7d6f903a0811550e4e1647bb8255281670aeee90d1e23022382e.jpg)

<details>
<summary>line</summary>

| z    | ε = 0  | ε = 0.05 | ε = 0.1 | ε = 0.1582 |
| ---- | ------ | -------- | ------- | ---------- |
| 0.0  | 0.82   | 0.81     | 0.80    | 0.79       |
| 0.1  | 0.88   | 0.87     | 0.86    | 0.85       |
| 0.2  | 0.94   | 0.93     | 0.92    | 0.91       |
| 0.3  | 0.97   | 0.96     | 0.95    | 0.94       |
| 0.4  | 0.99   | 0.98     | 0.97    | 0.96       |
| 0.5  | 1.00   | 1.00     | 1.00    | 1.00       |
</details>

![](images/65c183b333a826d262211238d5d365c3c3b52c3808b01cc6b7a974beea652853.jpg)

<details>
<summary>line</summary>

| z    | ε = 0  | ε = 0.05 | ε = 0.1  | ε = 0.1582 |
| ---- | ------ | -------- | -------- | ---------- |
| 0    | 1.0000 | 1.0000   | 1.0000   | 1.0000     |
| 0.1  | 1.0200 | 1.0150   | 1.0100   | 1.0050     |
| 0.2  | 1.0500 | 1.0450   | 1.0400   | 1.0350     |
| 0.3  | 1.0800 | 1.0750   | 1.0700   | 1.0650     |
| 0.4  | 1.1200 | 1.1150   | 1.1100   | 1.1050     |
| 0.5  | 1.1600 | 1.1550   | 1.1500   | 1.1450     |
| 0.6  | 1.2000 | 1.1950   | 1.1900   | 1.1850     |
| 0.7  | 1.2400 | 1.2350   | 1.2300   | 1.2250     |
| 0.8  | 1.2800 | 1.2750   | 1.2700   | 1.2650     |
| 0.9  | 1.3000 | 1.2950   | 1.2900   | 1.2850     |
| 1.0  | 1.3000 | 1.2950   | 1.2900   | 1.2850     |
</details>

FIGURE 9. Graphs of n (plain) and r(n) (dotted) where n solves problem (5.21) for several values of $\epsilon$ , $n_{*}$ being fixed and $\tau = n_{*}/r_{*}$ being chosen strictly less than $\tau_{\#}(\kappa)$ (here, $\tau = 0.3 \tau_{\#}$ ). The pressure law is the $\gamma$ -law (1.5) with exponent $\gamma = 2$ .

Heteroclinic orbits   
![](images/de94a5d93a1c3ddfc408fc018d6c15c00906f3c8e49c041f06b0b1488fd93e8d.jpg)

<details>
<summary>line</summary>

| n/n* | ε = 0 | ε = 0.05 | ε = 0.1 | ε = 0.1582 |
|------|-------|----------|---------|------------|
| 0.8  | 0.8   | 0.8      | 0.8     | 0.79       |
| 0.85 | 0.83  | 0.84     | 0.84    | 0.84       |
| 0.9  | 0.87  | 0.88     | 0.88    | 0.88       |
| 0.95 | 0.93  | 0.94     | 0.94    | 0.94       |
| 1.0  | 1.0   | 1.0      | 1.0     | 1.0        |
</details>

![](images/83f2e041699a4e2380e7d0024c29c9b388c37bec3eee3a071f3ed6d2904a5408.jpg)

<details>
<summary>line</summary>

| n/n*  | ε = 0   | ε = 0.05 | ε = 0.1  | ε = 0.1582 |
|-------|---------|----------|----------|------------|
| 1.00  | 1.00    | 1.00     | 1.00     | 1.00       |
| 1.05  | 1.05    | 1.05     | 1.05     | 1.05       |
| 1.10  | 1.10    | 1.10     | 1.10     | 1.10       |
| 1.15  | 1.15    | 1.15     | 1.15     | 1.15       |
| 1.20  | 1.20    | 1.20     | 1.20     | 1.20       |
| 1.25  | 1.25    | 1.25     | 1.25     | 1.25       |
| 1.30  | 1.30    | 1.30     | 1.30     | 1.30       |
</details>

FIGURE 10. Orbits connecting $(\mathrm{n}_{\times},\mathrm{r}(\mathrm{n}_{\times}))$ to (1,1) for several values of $\epsilon$ for the $\gamma$ -law (1.5) with exponent $\gamma=2$ . The straight line $n\mapsto\tau_{\#}(\kappa)n$ is plotted for reference, and the tangent point with the corresponding orbit is indicated (markers as in Figure 9).

Capturing viscous profiles is very sensitive because it requires the determination of the equilibrium value with high accuracy. Again, the resolution of the differential system should be performed with a high-order method in order to capture the profile. A thorough numerical investigation will be presented elsewhere, addressing in further details the computational difficulties and the role of the parameters of the model.

# REFERENCES

[1] A. A. AMSDEN, J. D. RAMSHAW, P. J. O'ROURKE, AND J. K. DUKOWICZ, KIVA: A computer program for two- and three-dimensional fluid flows with chemical reactions and fuel sprays, tech. rep., Los Alamos National Laboratory, 1985. Technical Report LA-10245-MS.   
[2] C. BARANGER, L. BOUDIN, P.-E. JABIN, AND S. MANCINI, A modeling of biospray for the upper airways, ESAIM:Proc, 14 (2005), pp. 41–47.   
[3] G. K. BATCHELOR, A new theory of the instability of a uniform fluidized bed, J. Fluid Mech., 193 (1988), pp. 75–110.   
[4] K. BEAUCHARD AND E. ZUAZUA, Large time asymptotics for partially dissipative hyperbolic systems, Arch. Ration. Mech. Anal., 199 (2011), pp. 177–227.   
[5] F. BOUCHUT, E. FERNÁNDEZ-NIETO, A. MANGENEY, AND G. NARBONA-REINA, A two-phase shallow debris flow model with energy balance, ESAIM: M2AN, 49 (2015), pp. 101–140.   
[6] L. BOUDIN, C. GRANDMONT, A. LORZ, AND A. MOUSSA, Modelling and numerics for respiratory aerosols, Comm. Comput. Phys., 18 (2015), pp. 723–756.   
[7] J. A. CARRILLO AND T. GOUDON, Stability and asymptotic analysis of a fluid-particle interaction model, Comm. Partial Differential Equations, 31 (2006), pp. 1349–1379.   
[8] J.-A. CARRILLO, T. GOUDON, AND P. LAFITTE, Simulation of fluid and particles flows: asymptotic preserving schemes for bubbling and flowing regimes, J. Comput. Phys, 227 (2008), pp. 7929–7951.   
[9] C. M. DAFERMOS, Hyperbolic conservation laws in continuum physics, vol. 325 of Grundlehren der Mathematischen Wissenschaften [Fundamental Principles of Mathematical Sciences], Springer-Verlag, Berlin, third ed., 2010.   
[10] K. DOMELEVO AND J.-M. ROQUEJOFFRE, Existence and stability of travelling wave solutions in a kinetic model of two-phase flows, Commun. PDE, 24 (1999), pp. 61–108.   
[11] K. DOMELEVO AND P. VILLEDIEU, A hierarchy of models for turbulent dispersed two-phase flows derived from a kinetic equation for the joint particle-gas pdf, Commun. Math. Sci., 5 (2007), pp. 331-353.   
[12] H. FREISTÜHLER, C. FRIES, AND C. ROHDE, Existence, bifurcation and stability of profiles for classical and non-classical shock waves, tech. rep., Max Planck Institute für Math. in den Naturwissenschaften Leipzig, 2000.   
[13] D. GIDASPOW, Hydrodynamics of fluidization and heat transfer: supercomputer modeling, Appl. Mech. Rev., 39 (1986), pp. 1-22.   
[14] D. GILBARG, The existence and limit behavior of the one-dimensional shock layer, Amer. J. Math., 73 (1951), pp. 256-274.   
[15] T. GOUDON, P.-E. JABIN, AND A. VASSEUR, Hydrodynamic limit for the Vlasov-Navier-Stokes equations. I. Light particles regime, Indiana Univ. Math. J., 53 (2004), pp. 1495–1515.   
[16] \_\_\_\_, Hydrodynamic limit for the Vlasov-Navier-Stokes equations. II. Fine particles regime, Indiana Univ. Math. J., 53 (2004), pp. 1517–1536.   
[17] K. HAMDACHE, Global existence and large time behaviour of solutions for the Vlasov-Stokes equations, Japan J. Indust. Appl. Math., 15 (1998), pp. 51–74.   
[18] S. HANK, R. SAUREL, AND O. LE METAYER, A hyperbolic Eulerian model for dilute two-phase suspensions, J. Modern Physics, 2 (2011), pp. 997–1011.   
[19] S. E. HARRIS AND D. G. CRIGHTON, Solitons, solitary waves, and voidage disturbances in gas-fluidized beds, J. Fluid. Mech., 266 (1994), pp. 243–276.   
[20] R. M. HÖFER, The inertialess limit of particle sedimentation modeled by the Vlasov-Stokes equations, SIAM J. Math. Anal., 50 (2018), pp. 5446-5476.   
[21] H. HUGONIOT, Sur la propagation du mouvement dans les corps et spécialement dans les gaz parfaits, I, J. Ecole Polytechnique, 57 (1887), pp. 3–97.   
[22] \_\_\_\_, Sur la propagation du mouvement dans les corps et spécialement dans les gaz parfaits, II, J. Ecole Polytechnique, 58 (1889), pp. 1–125.

[23] J. HUMPHERYS AND K. ZUMBRUN, Spectral stability of small-amplitude shock profiles for dissipative symmetric hyperbolic-parabolic systems, Z. Angew. Math. Phys., 53 (2002), pp. 20-34.   
[24] J. HYLKEMA, Modélisation cinétique et simulation numérique d'un brouillard dense de gouttelettes. Application aux propulseurs à poudre, PhD thesis, École Nationale Supérieure de l'Aéronautique et de l'Espace (Toulouse), 1999.   
[25] M. ISHII, One-dimensional drift-flux model and constitutive equations for relative motion between phases in various two-phase flow regimes, tech. rep., Argonne National Lab., 1977. ANL-77-47.   
[26] P.-E. JABIN, Large time concentrations for solutions to kinetic equations with energy dissipation, Comm. Partial Differential Equations, 25 (2000), pp. 541–557.   
[27] ——, Macroscopic limit of Vlasov type equations with friction, Ann. Inst. H. Poincaré Anal. Non Linéaire, 17 (2000), pp. 651–672.   
[28] S. KAWASHIMA, Systems of a hyperbolic-parabolic composite type, with applications to the equations of magnetohydrodynamics, PhD thesis, Kyoto University, 1983.   
[29] T.-P. LIU, The entropy condition and the admissibility of shocks, J. Math. Anal. Appl., 53 (1976), pp. 78–88.   
[30] A. MAJDA AND R. L. PEGO, Stable viscosity matrices for systems of conservation laws, J. Differential Equations, 56 (1985), pp. 229–262.   
[31] A. MANGENEY, P. HEINRICH, AND R. ROCHE, Analytical and numerical solution of the dam-break problem for application to water floods, debris and dense snow avalanches, Pure Appl. Geophys., 157 (2000), pp. 1081–1096.   
[32] C. MASCIA AND R. NATALINI, On relaxation hyperbolic systems violating the shizuta-kawashima condition, Arch. Ration. Mech. Anal., 195 (2010), pp. 729–762.   
[33] C. MASCIA AND K. ZUMBRUN, Stability of small-amplitude shock profiles of symmetric hyperbolic-parabolic systems, Comm. Pure Appl. Math., 57 (2004), pp. 841-876.   
[34] M. S. Mock, A topological degree for orbits connecting critical points of autonomous systems, J. Differential Equations, 38 (1980), pp. 176–191.   
[35] L. MORAWSKA, Environmental aerosol physics, tech. rep., International Laboratory for Air Quality and Health, 2004.   
[36] P. J. O'ROURKE, Collective drop effects on vaporizing liquid sprays, PhD thesis, Princeton University, NJ, 1981. Available as Technical Report #87545 Los Alamos National Laboratory.   
[37] N. A. PATANKAR AND D. D. JOSEPH, Modeling and numerical simulation of particulate flows by the Eulerian–Lagrangian approach, Int. J. Multiphase Flow, 27 (2001), pp. 1659–1684.   
[38] R. L. PEGO, Stable viscosities and shock profiles for systems of conservation laws, Trans. Amer. Math. Soc., 282 (1984), pp. 749–763.   
[39] B. PERTHAME, Kinetic formulation of conservation laws, vol. 21 of Oxford Lecture Series in Math. and its Appl., Oxford Univ. Press, 2002.   
[40] W. J. M. RANKINE, On the thermodynamic theory of waves of finite longitudinal disturbance, Phil. Trans. Royal Soc. London, 160 (1870), pp. 277–288.   
[41] L. SAINT RAYMOND, Hydrodynamic limits of the Boltzmann equation, vol. 1971 of Lect. Notes in Math., Springer, 2009.   
[42] Y. SHIZUTA AND S. KAWASHIMA, Systems of equations of hyperbolic-parabolic type with applications to the discrete Boltzmann equation, Hokkaido Math. J., 14 (1984), pp. 435-457.   
[43] J. SMOLLER, Shock waves and reaction-diffusion equations, vol. 208 of Grundlehren der mathematischen Wissenschaften, Springer, 1994. 2nd. ed.   
[44] F. A. WILLIAMS, Combustion theory, Benjamin Cummings Publ., 1985. Second edition.   
[45] K. ZUMBRUN AND P. HOWARD, Pointwise semigroup methods and stability of viscous shock waves, Indiana Univ. Math. J., 47 (1998), pp. 741-871.