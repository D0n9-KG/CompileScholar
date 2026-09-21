# Effective one-body approach to general relativistic two-body dynamics

A. Buonanno
Institut des Hautes Etudes Scientifiques, 91440 Bures-sur-Yvette, France

T. Damour
Institut des Hautes Etudes Scientifiques, 91440 Bures-sur-Yvette, France and DARC, CNRS-Observatoire de Paris, 92195 Meudon, France (Received 30 November 1998; published 8 March 1999)

We map the general relativistic two-body problem onto that of a test particle moving in an effective external metric. This effective-one-body approach defines, in a non-perturbative manner, the late dynamical evolution of a coalescing binary system of compact objects. The transition from the adiabatic inspiral, driven by gravitational radiation damping, to an unstable plunge, induced by strong spacetime curvature, is predicted to occur for orbits more tightly bound than the innermost stable circular orbit in a Schwarzschild metric of mass $M = m_{1} + m_{2}$ . The binding energy, angular momentum and orbital frequency of the innermost stable circular orbit for the time-symmetric two-body problem are determined as a function of the mass ratio. [S0556-2821(99)04806-7]

PACS number(s): 04.30.Db, 04.25.Nx, 97.80.Fk

# I. INTRODUCTION

Binary systems made of compact objects (neutron stars or black holes), and driven toward coalescence by gravitational radiation damping, are among the most promising candidate sources for interferometric gravitational-wave detectors such as the Laser Interferometric Gravitational Wave Observatory (LIGO) and VIRGO. It is therefore important to study the late dynamical evolution of a coalescing binary system of compact objects and, in particular, to estimate when the transition occurs from an adiabatic inspiral, driven by gravitational radiation damping, to an unstable plunge, induced by strong spacetime curvature. The global structure of the gravitational wave signal emitted by a coalescing binary depends sensitively on the location of the transition from inspiral to plunge. For instance, in the case of a system of two equal-mass neutron stars, if this transition occurs for relatively loosely bound orbits, the inspiral phase will evolve into a plunge phase before tidal disruption takes place. On the other hand, if the transition occurs for tightly bound orbits, tidal effects will dominate the late dynamical evolution.

In this paper we introduce a novel approach to the general relativistic two-body problem. The basic idea is to map (by a canonical transformation) the two-body problem onto an effective one-body problem, i.e. the motion of a test particle in some effective external metric. When turning off radiation damping, the effective metric will be a static and spherically symmetric deformation of the Schwarzschild metric. [The deformation parameter is the symmetric mass ratio $\nu \equiv m_{1}m_{2}/(m_{1}+m_{2})^{2}$ .] Solving exactly the effective problem of a test particle in this deformed Schwarzschild metric amounts to introducing a particular non-perturbative method for re-summing the post-Newtonian expansion of the equations of motion.

Our effective one-body approach is inspired by (though different from) an approach to electromagnetically interacting quantum two-body problems developed in the works of Brézin, Itzykson and Zinn-Justin $[1]$ (see also $[2]$ ) and of Todorov and co-workers [3,4]. Reference [1] has shown that an approximate summation (corresponding to the eikonal approximation) of the “crossed-ladder” Feynman diagrams for the quantum scattering of two charged particles led to a “relativistic Balmer formula” for the squared mass of bound states which correctly included recoil effects [i.e. effects linked to the finite symmetric mass ratio $\nu=m_{1}m_{2}/(m_{1}+m_{2})^{2}$ ]. However, the eikonal approximation does not capture some of the centrifugal barrier shifts which have to be added by hand through a shift $n\to n-\epsilon_{j}$ of the principal quantum number [1,2]. The approach of Ref. [3] is more systematic, being based on a (Lippmann-Schwinger-type) quasi-potential equation whose solution is fitted to the Feynman expansion of the (on-shell) scattering amplitudes $\langle p_{1}^{\prime}p_{2}^{\prime}|S|p_{1}p_{2}\rangle$ . However, several arbitrary choices have to be made to define the (off-shell) quasi-potential equation and the nice form of the relativistic Balmer formula proposed in Ref. [1] is recovered only at the end, after two seemingly accidental simplifications: (i) the ratio of some complicated energy-dependent quantities simplifies [5], and (ii) the second-order contribution to the quasi-potential contributes only to third order. We note also that the extension of Todorov’s quasi-potential approach (initially developed for quantum two-body electrodynamics) to the gravitational two-body problem [4] leads to much more complicated expressions than the approach developed here.

Before entering into the technical details of the effective one-body approach, let us outline the main features of our work. We use as input the explicit, post-Newtonian (PN) expanded classical equations of motion of a gravitationally interacting system of two compact objects. In harmonic coordinates (which are convenient to start with because they are standardly used for computing the generation of gravitational radiation), these equations of motion are explicitly known up to the 2.5PN level $[(v/c)^{5}$ accuracy] [6,7]. They have the form $(a,b=1,2)$

$$
\boldsymbol {a} _ {a} = \boldsymbol {\mathcal {A}} _ {a} ^ {2 \mathrm{PN}} (\boldsymbol {z} _ {b}, \boldsymbol {v} _ {b}) + \boldsymbol {A} _ {a} ^ {\text { reac }} (\boldsymbol {z} _ {b}, \boldsymbol {v} _ {b}) + \mathcal {O} (c ^ {- 6}), \tag {1.1}
$$

where $A^{2PN}=A_{0}+c^{-2}A_{2}+c^{-4}A_{4}$ denotes the time-symmetric part of the equations of motion, and $A^{reac}=c^{-5}A_{5}$ their time-antisymmetric part. Here, $z_{a}$ , $v_{a}$ , $a_{a}$ , denote the positions, velocities and accelerations, in harmonic coordinates, of the two bodies. (In this work we consider only non-spinning objects.) Throughout this paper, we shall use the following notation for the quantities related to the masses $m_{1}$ and $m_{2}$ of the two bodies:

$$
M \equiv m _ {1} + m _ {2}, \quad \mu \equiv \frac {m _ {1} m _ {2}}{M}, \quad \nu \equiv \frac {\mu}{M} \equiv \frac {m _ {1} m _ {2}}{(m _ {1} + m _ {2}) ^ {2}}. \tag {1.2}
$$

Note that the “symmetric mass ratio” $\nu$ varies between 0 (test mass limit) and $\frac{1}{4}$ (equal mass case).

We first focus on the time-symmetric, 2PN dynamics defined by $\mathcal{A}_{a}^{2\mathrm{PN}}(z_{b},\boldsymbol{v}_{b})$ . After going to the center of mass frame (uniquely defined by the Poincaré symmetries of the 2PN dynamics), and after a suitable coordinate transformation [from harmonic coordinates to Arnowitt-Deser-Misner (ADM) coordinates $z_{a}\rightarrow q_{a}$ ], the dynamics of the relative coordinates $q\equiv q_{1}-q_{2}$ is defined by a 2PN Hamiltonian $H(\boldsymbol{q},\boldsymbol{p})$ . Starting from $H(\boldsymbol{q},\boldsymbol{p})$ , we shall uniquely introduce a 2PN-accurate static and spherically symmetric “effective metric”

$$
\begin{array}{l} d s _ {\mathrm{eff}} ^ {2} = - A (R _ {\mathrm{eff}}) c ^ {2} d t _ {\mathrm{eff}} ^ {2} \\ + \frac {D (R _ {\text { eff }})}{A (R _ {\text { eff }})} d R _ {\text { eff }} ^ {2} + R _ {\text { eff }} ^ {2} (d \theta_ {\text { eff }} ^ {2} + \sin^ {2} \theta_ {\text { eff }} d \varphi_ {\text { eff }} ^ {2}), \tag {1.3} \\ \end{array}
$$

where

$$
A (R) = 1 + \frac {a _ {1}}{c ^ {2} R} + \frac {a _ {2}}{c ^ {4} R ^ {2}} + \frac {a _ {3}}{c ^ {6} R ^ {3}}, \tag {1.4}
$$

$$
D (R) = 1 + \frac {d _ {1}}{c ^ {2} R} + \frac {d _ {2}}{c ^ {4} R ^ {2}},
$$

such that the “linearized” effective metric (defined by $a_{1}$ and $d_{1}$ ) is the linearized Schwarzschild metric defined by the total mass $M=m_{1}+m_{2}$ , and such that the effective Hamiltonian $H_{\mathrm{eff}}(\boldsymbol{q}_{\mathrm{eff}},\boldsymbol{p}_{\mathrm{eff}})$ defined by the geodesic action $-\int\mu cd s_{eff}$ , where $\mu=m_{1}m_{2}/M$ is the reduced mass, can be mapped onto the relative-motion 2PN Hamiltonian $H(\boldsymbol{q},\boldsymbol{p})$ by the combination of a canonical transformation $(\boldsymbol{q}_{\mathrm{eff}},\boldsymbol{p}_{\mathrm{eff}})\rightarrow(\boldsymbol{q},\boldsymbol{p})$ and of an energy transformation $H=f(H_{\mathrm{eff}})$ , corresponding to an energy-dependent “canonical” rescaling of the time coordinate $dt_{\mathrm{eff}}=dt(dH/dH_{\mathrm{eff}})$ .

The effective metric so constructed is a deformation of the Schwarzschild metric, with the deformation parameter being the symmetric mass ratio $\nu=\mu/M$ . Considering this deformed Schwarzschild metric as an exact external metric then defines (in the effective coordinates) a $\nu$ -deformed Schwarzschild-like dynamics, which can be mapped back onto the original coordinates $q_{a}$ or $z_{a}$ . Our construction can be seen as a non-perturbative way of re-summing the post-Newtonian expansion in the relativistic regime where $GM/(c^{2}|q_{1}-q_{2}|)$ becomes of order unity. In particular, our construction defines a specific $\nu$ -deformed innermost stable circular orbit (ISCO). Superposing the gravitational reaction force $A^{\text{reac}}$ onto the “exact” deformed-Schwarzschild dynamics (defined by mapping back the effective problem onto the real one) finally defines, in a non-perturbative manner, a dynamical system which is a good candidate for describing the late stages of evolution of a coalescing compact binary.

# II. SECOND POST-NEWTONIAN DYNAMICS OF THE RELATIVE MOTION OF A TWO-BODY SYSTEM

Let us recall some of the basic properties of the dynamics defined by neglecting the time-odd reaction force in the Damour-Deruelle equations of motion (1.1). The 2PN [i.e. $(v / c)^4$ -accurate] truncation of these equations of motion defines a time-symmetric dynamics which is derivable from a generalized Lagrangian $L(z_{1},z_{2},\pmb{v}_{1},\pmb{v}_{2},\pmb{a}_{1},\pmb{a}_{2})$ [8,7] (a function of the harmonic positions, $z_{1},z_{2}$ , velocities $\pmb{v}_1,\pmb{v}_2$ and accelerations $\pmb{a}_1,\pmb{a}_2$ ). The generalized Lagrangian $L(z_{1},z_{2},\pmb{v}_{1},\pmb{v}_{2},\pmb{a}_{1},\pmb{a}_{2})$ is (approximately) invariant under the Poincaré group [9]. This invariance leads (via Noether's theorem) to an explicit construction of the usual ten relativistic conserved quantities for a dynamical system: energy $\mathcal{E}$ , linear momentum $\mathcal{P}$ , angular momentum $\mathcal{J}$ , and center-of-mass constant $\mathcal{K} = \mathcal{G} - \mathcal{P}t$ . Because of the freedom to perform a Poincaré transformation (in harmonic coordinates), we can go to the (2PN) center-of-mass frame, defined such as

$$
\mathcal {P} = \mathcal {K} = \mathcal {G} = 0. \tag {2.1}
$$

References [10,11] explicitly constructed the coordinate transformation between the harmonic (or de Donder) coordinates, say $z^{\mu}$ , used in the Damour-Deruelle equations of motion, and the coordinates, say $q^{\mu}$ , introduced by Arnowitt, Deser and Misner [12] in the framework of their canonical approach to the dynamics of the gravitational field. The Lagrangian giving the 2PN motion in ADM coordinates has the advantage of being an ordinary Lagrangian $L(\boldsymbol{q}_1,\boldsymbol{q}_2,\dot{\boldsymbol{q}}_1,\dot{\boldsymbol{q}}_2)$ (depending only on positions and velocities), which is equivalent to an ordinary Hamiltonian $H(\boldsymbol{q}_1,\boldsymbol{q}_2,\boldsymbol{p}_1,\boldsymbol{p}_2)$ [13,14]. The explicit expression of the 2PN Hamiltonian in ADM coordinates, $H(\boldsymbol{q}_1,\boldsymbol{q}_2,\boldsymbol{p}_1,\boldsymbol{p}_2)$ , has been derived in Ref. [11] by applying a contact transformation

$$
\boldsymbol {q} _ {a} (t) = \boldsymbol {z} _ {a} (t) - \delta^ {*} \boldsymbol {z} _ {a} (z, v) \tag {2.2}
$$

to the generalized Lagrangian $L(z_a, \boldsymbol{v}_a, \boldsymbol{a}_a)$ . The shift $\delta^* z_a$ is of order $\mathcal{O}(c^{-4})$ and is defined in Eq. (35) of [10] or Eqs. (2.4) of [11]. The contact transformation (2.2) removes the acceleration dependence of the harmonic-coordinate Lagrangian $L^{\mathrm{harm}}(z, v, a)$ and transforms it into the ADM-coordinate ordinary Lagrangian $L^{\mathrm{ADM}}(q, \dot{q})$ . A further Legendre transform turns $L^{\mathrm{ADM}}(\boldsymbol{q}_1, \boldsymbol{q}_2, \dot{\boldsymbol{q}}_1, \dot{\boldsymbol{q}}_2)$ into the needed 2PN Hamiltonian $H(\boldsymbol{q}_1, \boldsymbol{q}_2, \boldsymbol{p}_1, \boldsymbol{p}_2)$ in ADM coordinates. The explicit expression of this Hamiltonian is given in Eq. (2.5) of Ref. [11]. It has also been shown in Ref. [10] that the Hamiltonian $H(\boldsymbol{q}_1, \boldsymbol{q}_2, \boldsymbol{p}_1, \boldsymbol{p}_2)$ can be directly derived in ADM coordinates from the (not fully explicit) $N$ -body re-

sults of Ref. [13] by computing a certain integral entering the two-body interaction potential. (For further references on the general relativistic problem of motion, see the review in [15]; for recent work on the gravitational Hamiltonian see [16,17,18,19].)

The ADM expression of the total Noether linear momentum P associated to the translational invariance of $L(z,v,a)$ is simply $P=p_{1}+p_{2}$ . Therefore it is easily checked that, in the center-of-mass frame (2.1), the relative motion is obtained by substituting in the two-body Hamiltonian $H(q_{1},q_{2},p_{1},p_{2})$ ,

$$
\boldsymbol {p} _ {1} \rightarrow \boldsymbol {P}, \quad \boldsymbol {p} _ {2} \rightarrow - \boldsymbol {P}, \tag {2.3}
$$

where $P=\partial S/\partial Q$ is the canonical momentum associated with the relative ADM position vector $Q\equiv q_{1}-q_{2}$ . (For clarity, we modify the notation of Ref. [11] by using $q_{1}$ , $q_{2}$ , Q and q for the ADM position coordinates which are denoted $r_{1}$ , $r_{2}$ , R and r, respectively, in Ref. [11].)

Our technical starting point in this work will be the reduced center-of-mass 2PN Hamiltonian (in reduced ADM coordinates). We introduce the following reduced variables (all defined in ADM coordinates, and in the center-of-mass frame):

$$
\boldsymbol {q} \equiv \frac {\boldsymbol {Q}}{G M} \equiv \frac {\boldsymbol {q} _ {1} - \boldsymbol {q} _ {2}}{G M}, \quad \boldsymbol {p} \equiv \frac {\boldsymbol {P}}{\mu},
$$

$$
\hat {t} \equiv \frac {t}{G M}, \quad \hat {H} \equiv \frac {H ^ {\mathrm{NR}}}{\mu} \equiv \frac {H ^ {\mathrm{R}} - M c ^ {2}}{\mu}. \tag {2.4}
$$

In the last equation, the superscript “NR” means “non-relativistic” (i.e. after subtraction of the appropriate rest-mass contribution), while “R” means “relativistic” (i.e. including the appropriate rest-mass contribution). From Eq. (3.1) of [11] the reduced 2PN relative-motion Hamiltonian (without the rest-mass contribution) reads

$$
\hat {H} (\boldsymbol {q}, \boldsymbol {p}) = \hat {H} _ {0} (\boldsymbol {q}, \boldsymbol {p}) + \frac {1}{c ^ {2}} \hat {H} _ {2} (\boldsymbol {q}, \boldsymbol {p}) + \frac {1}{c ^ {4}} \hat {H} _ {4} (\boldsymbol {q}, \boldsymbol {p}), \tag {2.5}
$$

where

$$
\hat {H} _ {0} (\boldsymbol {q}, \boldsymbol {p}) = \frac {1}{2} \boldsymbol {p} ^ {2} - \frac {1}{q}, \tag {2.6a}
$$

$$
\begin{array}{l} \hat {H} _ {2} (\boldsymbol {q}, \boldsymbol {p}) = - \frac {1}{8} (1 - 3 \nu) \boldsymbol {p} ^ {4} - \frac {1}{2 q} [ (3 + \nu) \boldsymbol {p} ^ {2} + \nu (\boldsymbol {n} \cdot \boldsymbol {p}) ^ {2} ] \\ + \frac {1}{2 q ^ {2}}, \tag {2.6b} \\ \end{array}
$$

$$
\begin{array}{l} \hat {H} _ {4} (\boldsymbol {q}, \boldsymbol {p}) = \frac {1}{1 6} (1 - 5 \nu + 5 \nu^ {2}) \boldsymbol {p} ^ {6} \\ + \frac {1}{8 q} [ (5 - 2 0 \nu - 3 \nu^ {2}) p ^ {4} - 2 \nu^ {2} p ^ {2} (\boldsymbol {n} \cdot \boldsymbol {p}) ^ {2} \\ \left. - 3 \nu^ {2} (\pmb {n} \cdot \pmb {p}) ^ {4} \right] \\ + \frac {1}{2 q ^ {2}} [ (5 + 8 \nu) \boldsymbol {p} ^ {2} + 3 \nu (\boldsymbol {n} \cdot \boldsymbol {p}) ^ {2} ] - \frac {1}{4 q ^ {3}} (1 + 3 \nu), \tag {2.6c} \\ \end{array}
$$

in which $q \equiv |q| \equiv (q^{2})^{1/2}$ and $n \equiv q/q$ . When convenient, we shall also use the notation r for the reduced radial separation q (and R for the unreduced one Q) [as in Eqs. (2.8)–(2.12) below].

The relative-motion Hamiltonian (2.5) is invariant under time translations and space rotations. The associated conserved quantities are the reduced center-of-mass (c.m.) energy and angular momentum of the binary system:

$$
\hat {H} (\boldsymbol {q}, \boldsymbol {p}) = \hat {\mathcal {E}} ^ {\mathrm{NR}} \equiv \frac {\mathcal {E} _ {\mathrm{c.m.}} ^ {\mathrm{NR}}}{\mu}, \quad \boldsymbol {q} \times \boldsymbol {p} = \boldsymbol {j} \equiv \frac {\mathcal {J} _ {\mathrm{c.m.}}}{\mu G M}. \tag {2.7}
$$

A convenient way of solving the 2PN relative-motion dynamics is to use the Hamilton-Jacobi approach. The motion in the plane of the relative trajectory is obtained, in polar coordinates

$$
q ^ {x} = r \cos \varphi , \quad q ^ {y} = r \sin \varphi , \quad q ^ {z} = 0, \tag {2.8}
$$

by separating the time and angular coordinates in the (planar) reduced action

$$
\hat {S} \equiv \frac {S}{\mu G M} = - \hat {\mathcal {E}} ^ {\mathrm{NR}} \hat {t} + j \varphi + \hat {S} _ {r} (r, \hat {\mathcal {E}} ^ {\mathrm{NR}}, j). \tag {2.9}
$$

The time-independent Hamilton-Jacobi equation $\hat{H}^{\mathrm{NR}}(\pmb {q},\pmb {p}) = \hat{\mathcal{E}}^{\mathrm{NR}}$ with $\pmb {p} = \partial \hat{S} /\partial \pmb{q}$ can be (iteratively) solved with respect to $(d\hat{S}_r /dr)^2$ with a result of the form

$$
\hat {S} _ {r} (r, \hat {\mathcal {E}} ^ {\mathrm{NR}}, j) = \int d r \sqrt {\mathcal {R} (r , \hat {\mathcal {E}} ^ {\mathrm{NR}} , j)}. \tag {2.10}
$$

The radial “effective potential” $\mathcal{R}(r,\hat{\mathcal{E}}^{\mathrm{NR}},j)$ is a fifth-order polynomial in $1/r\equiv1/q$ which is explicitly written down in Eqs. (3.4) of [11]. In this section, we shall only need the corresponding (integrated) radial action variable

$$
i _ {r} \equiv \frac {I _ {R}}{\mu G M} \equiv \frac {2}{2 \pi} \int_ {r _ {\min}} ^ {r _ {\max}} d r \sqrt {\mathcal {R} (r , \hat {\mathcal {E}} ^ {\mathrm{NR}} , j)}. \tag {2.11}
$$

The function $i_{r}(\hat{\mathcal{E}}^{\mathrm{NR}}, j)$ has been computed, at 2PN accuracy, in Ref. [11] [see Eq. (3.10) there]. To clarify some issues connected with the fact that the natural scalings in the “effective one-body problem” (to be considered below) differ from those in the present, real two-body problem, let us quote the expression of the unscaled radial action variable,

$$
I _ {R} = \alpha i _ {r} = \frac {2}{2 \pi} \int_ {R _ {\min}} ^ {R _ {\max}} d R \frac {d S _ {R} (R , \mathcal {E} ^ {\mathrm{NR}} , \mathcal {J})}{d R}, \tag {2.12}
$$

in terms of the unscaled variables $\mathcal{E}^{\mathrm{NR}} = \mu \hat{\mathcal{E}}^{\mathrm{NR}}$ and $\mathcal{J} = \alpha j$ . Here $R = Q = GMr = GMq$ , and we introduced the short-hand notation

$$
\alpha \equiv \mu G M = G m _ {1} m _ {2} \tag {2.13}
$$

for the gravitational two-body coupling constant. We have

$$
\begin{array}{l} I _ {R} \left(\mathcal {E} ^ {\mathrm{NR}}, \mathcal {J}\right) = \frac {\alpha \mu^ {1 / 2}}{\sqrt {- 2 \mathcal {E} ^ {\mathrm{NR}}}} \left[ 1 + \left(\frac {1 5}{4} - \frac {\nu}{4}\right) \frac {\mathcal {E} ^ {\mathrm{NR}}}{\mu c ^ {2}} + \left(\frac {3 5}{3 2} + \frac {1 5}{1 6} \nu \right. \right. \\ \left. + \frac {3}{3 2} \nu^ {2}\right) \left(\frac {\mathcal {E} ^ {\mathrm{NR}}}{\mu c ^ {2}}\right) ^ {2} \biggr ] \\ - \mathcal {J} + \frac {\alpha^ {2}}{c ^ {2} \mathcal {J}} \left[ 3 + \left(\frac {1 5}{2} - 3 \nu\right) \frac {\mathcal {E} ^ {\mathrm{NR}}}{\mu c ^ {2}} \right] \\ + \left(\frac {3 5}{4} - \frac {5}{2} \nu\right) \frac {\alpha^ {4}}{c ^ {4} \mathcal {J} ^ {3}}. \tag {2.14} \\ \end{array}
$$

Equation (2.14) can also be solved with respect to $\mathcal{E}^{\mathrm{NR}} \equiv \mathcal{E}^{\mathrm{R}} - Mc^{2}$ with the (2PN-accurate) result [see Eq. (3.13) of Ref. [11]]

$$
\begin{array}{l} \mathcal {E} ^ {\mathrm{R}} (\mathcal {N}, \mathcal {J}) = M c ^ {2} - \frac {1}{2} \frac {\mu \alpha^ {2}}{\mathcal {N} ^ {2}} \left[ 1 + \frac {\alpha^ {2}}{c ^ {2}} \left(\frac {6}{\mathcal {N J}} - \frac {1}{4} \frac {1 5 - \nu}{\mathcal {N} ^ {2}}\right) \right. \\ + \frac {\alpha^ {4}}{c ^ {4}} \left(\frac {5}{2} \frac {7 - 2 \nu}{\mathcal {N J} ^ {3}} + \frac {2 7}{\mathcal {N} ^ {2} \mathcal {J} ^ {2}} - \frac {3}{2} \frac {3 5 - 4 \nu}{\mathcal {N} ^ {3} \mathcal {J}} \right. \\ \left. \left. + \frac {1}{8} \frac {1 4 5 - 1 5 \nu + \nu^ {2}}{\mathcal {N} ^ {4}}\right) \right], \tag {2.15} \\ \end{array}
$$

where N denotes the Delaunay action variable $N \equiv I_{R} + J$ . The notation is chosen so as to evoke the one often used in the quantum Coulomb problem. Indeed, the classical action variables $I_{R}$ and J, or their combinations $N = I_{R} + J$ and J, are adiabatic invariants which, according to the Bohr-Sommerfeld rules, become (approximately) quantized in units of $\hbar$ for the corresponding quantum bound states. More precisely $N/\hbar$ becomes the “principal quantum number” and $J/\hbar$ the total angular momentum quantum number. The fact that the Newtonian-level non-relativistic energy $E^{NR} = -\frac{1}{2}\mu\alpha^{2}/N^{2} + O(c^{-2})$ depends only on the combination $N = I_{R} + J$ is the famous special degeneracy of the Coulomb problem. Note that 1PN (and 2PN) effects lift this degeneracy by bringing an extra dependence on J. There remains, however, the degeneracy associated with the spherical symmetry of the problem, which implies that the energy does not depend on the “magnetic quantum number,” i.e. on $M = J_{z}$ , but only on the magnitude of the angular momentum vector $J = \sqrt{J^{2}}$ . Though we shall only be interested in the classical gravitational two-body problem, it is conceptually useful to think in terms of the associated quantum problem. From this point of view, the formula (2.15) describes, when $N/\hbar$ and $J/\hbar$ take (non-zero) integer values, all the quantum energy levels as a function of the parameters $M=m_{1}+m_{2}$ , $\mu=m_{1}m_{2}/(m_{1}+m_{2})$ , $\alpha=Gm_{1}m_{2}$ and $\nu=\mu/M$ . It is to be noted that the function $\mathcal{E}^{\mathrm{R}}(\mathcal{N},\mathcal{J})$ describing the energy levels is a coordinate-invariant object.

# III. SECOND POST-NEWTONIAN ENERGY LEVELS OF THE EFFECTIVE ONE-BODY PROBLEM

The “energy levels” (2.15) summarize, at the 2PN accuracy, the dynamics obtained by eliminating the field variables $g_{\mu\nu}(x)$ in the total action of a gravitationally interacting binary system:

$$
\begin{array}{l} S _ {\mathrm{tot}} \left[ z _ {1} ^ {\mu}, z _ {2} ^ {\mu}, g _ {\mu \nu} \right] = - \int m _ {1} c d s _ {1} - \int m _ {2} c d s _ {2} \\ + S _ {\text { field }} [ g _ {\mu \nu} (x) ], \tag {3.1} \\ \end{array}
$$

where $ds_{1}=\sqrt{-g_{\mu\nu}(z_{1}^{\lambda})dz_{1}^{\mu}dz_{1}^{\nu}}$ and where $S_{\mathrm{field}}[g_{\mu\nu}(x)]$ is the (gauge-fixed) Einstein-Hilbert action for the gravitational field. Let $S_{\mathrm{real}}[z_{1}^{\mu},z_{2}^{\mu}]$ be the Fokker-type action obtained by (formally) integrating out $g_{\mu\nu}(x)$ in Eq. (3.1). (See, e.g., [10] for more details on Fokker-type actions. As we work here only at the 2PN level, and take advantage of the explicit results of Refs. [8,7], we do not need to enter the subtleties of the elimination of the field degrees of freedom, which are probably best treated within the ADM approach. See [20,14].)

The basic idea of the present work is to, somehow, associate to the “real” two-body dynamics $S_{\mathrm{real}}[z_{1}^{\mu}, z_{2}^{\mu}]$ some “effective” one-body dynamics in an external spacetime, as described by the action

$$
S _ {\mathrm{eff}} [ z _ {0} ^ {\mu} ] = - \int m _ {0} c d s _ {0}, \tag {3.2}
$$

where $ds_{0}=\sqrt{-g_{\mu\nu}^{\mathrm{eff}}(z_{0}^{\lambda})dz_{0}^{\mu}dz_{0}^{\nu}}$ , with some spherically symmetric static effective metric

$$
\begin{array}{l} d s _ {\text { eff }} ^ {2} = g _ {\mu \nu} ^ {\text { eff }} (x _ {\text { eff }} ^ {\lambda}) d x _ {\text { eff }} ^ {\mu} d x _ {\text { eff }} ^ {\nu} = - A (R _ {\text { eff }}) c ^ {2} d t _ {\text { eff }} ^ {2} + B (R _ {\text { eff }}) d R _ {\text { eff }} ^ {2} \\ + C (R _ {\text { eff }}) R _ {\text { eff }} ^ {2} (d \theta_ {\text { eff }} ^ {2} + \sin^ {2} \theta_ {\text { eff }} d \varphi_ {\text { eff }} ^ {2}). \tag {3.3} \\ \end{array}
$$

To simplify the notation we shall, henceforth in this section, drop the subscript “eff” on the coordinates used in the effective problem. (Later in this paper we shall explicitly relate the coordinates $z_{0}^{\mu}$ of the effective particle to the coordinates $z_{1}^{\mu}$ , $z_{2}^{\nu}$ of the two real particles.) The metric functions $A(R)$ , $B(R)$ , $C(R)$ will be constructed in the form of an expansion in 1/R:

$$
A (R) = 1 + \frac {a _ {1}}{c ^ {2} R} + \frac {a _ {2}}{c ^ {4} R ^ {2}} + \frac {a _ {3}}{c ^ {6} R ^ {3}} + \dots ,
$$

$$
B (R) = 1 + \frac {b _ {1}}{c ^ {2} R} + \frac {b _ {2}}{c ^ {4} R ^ {2}} + \dots . \tag {3.4}
$$

Beware that the variable R in Eqs. (3.4) denotes (in this section) the effective radial coordinate, which differs from the real ADM separation $Q = R_{ADM} = GMr$ used in the pre-

vious section (e.g. in the definition of $I_{R}$ ). We indicate in Eq. (3.4) the terms that we shall need at the 2PN level. The third function $C(R)$ entering the effective metric will be either fixed to $C_{S}(R) \equiv 1$ (in “Schwarzschild” coordinates) or to satisfy $C_{I}(R) \equiv B(R)$ (in “isotropic” coordinates).

There are two mass parameters entering the effective problem: (i) the mass $m_{0}$ of the effective particle and (ii) some mass parameter $M_{0}$ used to scale the coefficients $a_{i}$ , $b_{i}$ entering the effective metric. For instance, we can define $M_{0}$ by conventionally setting

$$
a _ {1} \equiv - 2 G M _ {0}. \tag {3.5}
$$

By analogy to Eq. (2.15), we can summarize, in a coordinate-invariant manner, the dynamics of the effective one-body problem (3.2)-(3.4) by considering the “energy levels” of the bound states of the particle $m_0$ in the metric $g_{\mu\nu}^{\mathrm{eff}}$ :

$$
\mathcal {E} _ {0} ^ {\mathrm{R}} = m _ {0} c ^ {2} + \mathcal {E} _ {0} ^ {\mathrm{NR}} = \mathcal {F} (\mathcal {N} _ {0}, \mathcal {J} _ {0}; m _ {0}, a _ {i}, b _ {i}). \tag {3.6}
$$

Here, the relativistic effective energy $E_{0}^{R}$ and the effective action variables $N_{0}, J_{0}$ are unambiguously defined by the action (3.2). Namely, we can separate the effective Hamilton-Jacobi equation

$$
g _ {\mathrm{eff}} ^ {\mu \nu} \frac {\partial S _ {\mathrm{eff}}}{\partial x ^ {\mu}} \frac {\partial S _ {\mathrm{eff}}}{\partial x ^ {\nu}} + m _ {0} ^ {2} c ^ {2} = 0, \tag {3.7}
$$

by writing (considering, for simplicity, only motions in the equatorial plane $\theta=\pi/2$ )

$$
S _ {\text { eff }} = - \mathcal {E} _ {0} t + \mathcal {J} _ {0} \varphi + S _ {R} ^ {0} (R, \mathcal {E} _ {0}, \mathcal {J} _ {0}). \tag {3.8}
$$

To abbreviate the notation we suppress the superscript “R” on the relativistic effective energy $E_{0}$ . Inserting Eq. (3.8) into Eq. (3.7) yields

$$
- \frac {1}{A (R)} \frac {\mathcal {E} _ {0} ^ {2}}{c ^ {2}} + \frac {1}{B (R)} \left(\frac {d S _ {R} ^ {0}}{d R}\right) ^ {2} + \frac {\mathcal {J} _ {0} ^ {2}}{C (R) R ^ {2}} + m _ {0} ^ {2} c ^ {2} = 0, \tag {3.9}
$$

and therefore

$$
S _ {R} ^ {0} (R, \mathcal {E} _ {0}, \mathcal {J} _ {0}) = \int d R \sqrt {\mathcal {R} _ {0} (R , \mathcal {E} _ {0} , \mathcal {J} _ {0})}, \tag {3.10}
$$

where

$$
\mathcal {R} _ {0} (R, \mathcal {E} _ {0}, \mathcal {J} _ {0}) \equiv \frac {B (R)}{A (R)} \frac {\mathcal {E} _ {0} ^ {2}}{c ^ {2}} - B (R) \left(m _ {0} ^ {2} c ^ {2} + \frac {\mathcal {J} _ {0} ^ {2}}{C (R) R ^ {2}}\right). \tag {3.11}
$$

The effective radial action variable $I_{R}^{0}$ is then defined as

$$
I _ {R} ^ {0} (\mathcal {E} _ {0}, \mathcal {J} _ {0}) \equiv \frac {2}{2 \pi} \int_ {R _ {\min}} ^ {R _ {\max}} d R \sqrt {\mathcal {R} _ {0} (R , \mathcal {E} _ {0} , \mathcal {J} _ {0})}, \tag {3.12}
$$

while the effective “principal” action variable $N_{0}$ is defined as the combination $N_{0} \equiv I_{R}^{0} + J_{0}$ .

To obtain the effective “energy levels” $\mathcal{E}_{0}=\mathcal{F}(\mathcal{N}_{0},\mathcal{J}_{0})$ one needs to compute the definite radial integral (3.12). Reference [11] (extending some classic work of Sommerfeld, used in the old quantum theory) has shown how to compute the PN expansion of the radial integral (3.12) to any order in the 1/R expansions (3.4). At the present 2PN order, Ref. [11] gave a general formula [their Eq. (3.9)] which can be straightforwardly applied to our case.

As we said above, the function describing the “energy levels,” $\mathcal{E}_{0}=\mathcal{F}(\mathcal{N}_{0},\mathcal{J}_{0})$ , is a coordinate-invariant construct. As a check on our calculations, we have computed it [or rather, we have computed the radial action $I_{R}^{0}(\mathcal{E}_{0},\mathcal{J}_{0})$ ] in the two preferred coordinate gauges for a spherically symmetric metric: the “Schwarzschild gauge” and the “isotropic” one. If $(a_{i},b_{i})$ denote the expansion coefficients (3.4) in the Schwarzschild gauge $[C_{S}(R)\equiv1]$ , we find (at the 2PN accuracy)

$$
\begin{array}{l} I _ {R} ^ {0} \left(\mathcal {E} _ {0}, \mathcal {J} _ {0}\right) = \frac {m _ {0} ^ {3 / 2}}{\sqrt {- 2 \mathcal {E} _ {0} ^ {\mathrm{NR}}}} \left[ A + B \frac {\mathcal {E} _ {0} ^ {\mathrm{NR}}}{m _ {0} c ^ {2}} + C \left(\frac {\mathcal {E} _ {0} ^ {\mathrm{NR}}}{m _ {0} c ^ {2}}\right) ^ {2} \right] - \mathcal {J} _ {0} \\ + \frac {m _ {0} ^ {2}}{c ^ {2} \mathcal {J} _ {0}} \left[ D + E \frac {\mathcal {E} _ {0} ^ {\mathrm{NR}}}{m _ {0} c ^ {2}} \right] + \frac {m _ {0} ^ {4}}{c ^ {4} \mathcal {J} _ {0} ^ {3}} F, \tag {3.13} \\ \end{array}
$$

where $\mathcal{E}_0^{\mathrm{NR}}\equiv \mathcal{E}_0 - m_0c^2$ , and where

$$
A = - \frac {1}{2} a _ {1}, \quad B = b _ {1} - \frac {7}{8} a _ {1}, \quad C = \frac {b _ {1}}{4} - \frac {1 9}{6 4} a _ {1},
$$

$$
D = \frac {a _ {1} ^ {2}}{2} - \frac {a _ {2}}{2} - \frac {a _ {1} b _ {1}}{4}, \quad E = a _ {1} ^ {2} - a _ {2} - \frac {a _ {1} b _ {1}}{2} - \frac {b _ {1} ^ {2}}{8} + \frac {b _ {2}}{2},
$$

$$
\begin{array}{l} F = \frac {1}{6 4} \left[ 2 4 a _ {1} ^ {4} - 4 8 a _ {1} ^ {2} a _ {2} + 8 a _ {2} ^ {2} + 1 6 a _ {1} a _ {3} - 8 a _ {1} ^ {3} b _ {1} + 8 a _ {1} a _ {2} b _ {1} \right. \\ \left. - a _ {1} ^ {2} b _ {1} ^ {2} + 4 a _ {1} ^ {2} b _ {2} \right]. \tag {3.14} \\ \end{array}
$$

Denoting by $(\widetilde{a}_{i},\widetilde{b}_{i})$ the expansion coefficients (3.4) in the isotropic gauge $[C_{I}(R)\equiv B_{I}(R)]$ , we find, by calculating $I_{R}^{0}$ directly in the isotropic gauge, that the coefficients $A,B,\ldots,F$ entering Eq. (3.13) have the following (slightly simpler) expressions in terms of $\widetilde{a}_{i}$ and $\widetilde{b}_{i}$ :

$$
A = - \frac {1}{2} \widetilde {a} _ {1}, \quad B = \widetilde {b} _ {1} - \frac {7}{8} \widetilde {a} _ {1}, \quad C = \frac {\widetilde {b} _ {1}}{4} - \frac {1 9}{6 4} \widetilde {a} _ {1},
$$

$$
D = \frac {\tilde {a} _ {1} ^ {2}}{2} - \frac {\tilde {a} _ {2}}{2} - \frac {\tilde {a} _ {1} \tilde {b} _ {1}}{2}, \quad E = \tilde {a} _ {1} ^ {2} - \tilde {a} _ {2} - \tilde {a} _ {1} \tilde {b} _ {1} + \tilde {b} _ {2},
$$

$$
F = \frac {1}{8} \left[ 3 \widetilde {a} _ {1} ^ {4} - 6 \widetilde {a} _ {1} ^ {2} \widetilde {a} _ {2} + \widetilde {a} _ {2} ^ {2} + 2 \widetilde {a} _ {1} \widetilde {a} _ {3} - 4 \widetilde {a} _ {1} ^ {3} \widetilde {b} _ {1} + 4 \widetilde {a} _ {1} \widetilde {a} _ {2} \widetilde {b} _ {1} \right.
$$

$$
\left. + \tilde {a} _ {1} ^ {2} \tilde {b} _ {1} ^ {2} + 2 \tilde {a} _ {1} ^ {2} \tilde {b} _ {2} \right]. \tag {3.15}
$$

The numerical values of the coefficients $A, B, \ldots, F$ are checked to be coordinate invariant by using the following relation between the $(a_i, b_i)$ and the $(\widetilde{a}_i, \widetilde{b}_i)$ [which is easily

derived either by integrating $dR_{I}/R_{I}=\sqrt{B_{S}(R_{S})}dR_{S}/R_{S}$ or by using the algebraic link $R_{S}=R_{I}\sqrt{B_{I}(R_{I})}$ :

$$
\tilde {a} _ {1} = a _ {1}, \quad \tilde {b} _ {1} = b _ {1}, \tag {3.16}
$$

$$
\widetilde {a} _ {2} = a _ {2} - \frac {1}{2} a _ {1} b _ {1}, \quad \widetilde {b} _ {2} = \frac {1}{2} b _ {2} - \frac {1}{8} b _ {1} ^ {2},
$$

$$
\tilde {a} _ {3} = a _ {3} - a _ {2} b _ {1} + \frac {7}{1 6} a _ {1} b _ {1} ^ {2} - \frac {1}{4} a _ {1} b _ {2}.
$$

Finally, solving iteratively Eq. (3.13) with respect to $E_{0}^{NR}$ , we find the analogue of Eq. (2.15), i.e. the explicit formula giving the effective “energy levels.” It is convenient to write it in terms of $N_{0} \equiv I_{R}^{0} + J_{0}$ , of the coupling constant

$$
\alpha_ {0} \equiv G M _ {0} m _ {0}, \tag {3.17}
$$

where $M_{0}$ is defined by Eq. (3.5), and of the $(GM_{0})$ -rescaled, dimensionless expansion coefficients $\hat{a}_{i}$ and $\hat{b}_{i}$ , of the Schwarzschild gauge:

$$
\hat {a} _ {i} \equiv a _ {i} / (G M _ {0}) ^ {i}, \quad \hat {b} _ {i} \equiv b _ {i} / (G M _ {0}) ^ {i}, \tag {3.18}
$$

with $\hat{a}_1\equiv -2$

We find

$$
\begin{array}{l} \mathcal {E} _ {0} \left(\mathcal {N} _ {0}, \mathcal {J} _ {0}\right) = m _ {0} c ^ {2} - \frac {1}{2} \frac {m _ {0} \alpha_ {0} ^ {2}}{\mathcal {N} _ {0} ^ {2}} \left[ 1 + \frac {\alpha_ {0} ^ {2}}{c ^ {2}} \left(\frac {C _ {3 , 1}}{\mathcal {N} _ {0} \mathcal {J} _ {0}} + \frac {C _ {4 , 0}}{\mathcal {N} _ {0} ^ {2}}\right) \right. \\ \left. + \frac {\alpha_ {0} ^ {4}}{c ^ {4}} \left(\frac {C _ {3 , 3}}{\mathcal {N} _ {0} \mathcal {J} _ {0} ^ {3}} + \frac {C _ {4 , 2}}{\mathcal {N} _ {0} ^ {2} \mathcal {J} _ {0} ^ {2}} + \frac {C _ {5 , 1}}{\mathcal {N} _ {0} ^ {3} \mathcal {J} _ {0}} + \frac {C _ {6 , 0}}{\mathcal {N} _ {0} ^ {4}}\right) \right], \tag {3.19} \\ \end{array}
$$

where the coefficients $C_{p,q}$ [which parametrize the contributions $\propto-\frac{1}{2}(\alpha_{0}/c)^{p+q}\mathcal{N}_{0}^{-p}\mathcal{J}_{0}^{-q}$ to $E_{0}/m_{0}c^{2}$ ] are given by

$$
C _ {3, 1} = 2 \hat {D}, \quad C _ {4, 0} = - \hat {B},
$$

$$
C _ {3, 3} = 2 \hat {F}, \quad C _ {4, 2} = 3 \hat {D} ^ {2},
$$

$$
C _ {5, 1} = - (4 \hat {B} \hat {D} + \hat {E}), \quad C _ {6, 0} = \frac {1}{4} (5 \hat {B} ^ {2} + 2 \hat {C}). \tag {3.20}
$$

Here, the dimensionless quantities $\hat{B},\hat{C},\hat{D},\hat{E},\hat{F}$ are the $GM_0$ -rescaled versions of the coefficients of Eq. (3.13), given by replacing the $a_i$ 's by $\hat{a}_i$ in Eqs. (3.14). For instance, $\hat{B} = \hat{b}_1 - 7 / 8\hat{a}_1 = \hat{b}_1 + 7 / 4$ , etc.

# IV. RELATING THE “REAL” AND THE “EFFECTIVE” ENERGY LEVELS, AND DETERMINING THE EFFECTIVE METRIC

We still have to define the precise rules by which we wish to relate the real two-body problem to the effective one-body one. If we think in quantum terms, there is a natural correspondence between N and $N_{0}$ , and J and $J_{0}$ , which are quantized in units of $\hbar$ . It is therefore very natural to require the identification

$$
\mathcal {N} = \mathcal {N} _ {0}, \quad \mathcal {J} = \mathcal {J} _ {0}, \tag {4.1}
$$

between the real action variables and the effective ones, and we will do so in the following. What is a priori less clear is the relation between the real masses and energies, $m_{1}$ , $m_{2}$ , $\mathcal{E}_{\mathrm{real}}^{R}=(m_{1}+m_{2})c^{2}+\mathcal{E}_{\mathrm{real}}^{\mathrm{NR}}$ , and the effective ones, $m_{0}$ , $M_{0}$ , $\mathcal{E}_{0}=m_{0}c^{2}+\mathcal{E}_{0}^{\mathrm{NR}}$ . The usual non-relativistic definition of an effective dynamics associated with the relative motion of a (Galileo-invariant) two-body system introduces an effective particle whose position $q_{0}$ is the relative position, $q_{0}=q_{1}-q_{2}$ , whose inertial mass $m_{0}^{NR}$ is the “reduced” mass $\mu\equiv m_{1}m_{2}/(m_{1}+m_{2})$ , and whose potential energy is the potential energy of the system, $V_{\mathrm{eff}}(q_{0})=V_{\mathrm{real}}(q_{1}-q_{2})$ . In the present case of a gravitationally interacting two-body system, with $V_{real}^{NR}=-Gm_{1}m_{2}/|q_{1}-q_{2}|$ , this would determine

$$
m _ {0} ^ {\mathrm{NR}} = \mu , \quad \text { and } \quad M _ {0} ^ {\mathrm{NR}} = m _ {1} + m _ {2} \equiv M, \tag {4.2}
$$

such that $\alpha_{real}=Gm_{1}m_{2}=\alpha_{0}=GM_{0}^{NR}m_{0}^{NR}$ . The non-relativistic identifications (4.2) are, however, paradoxical within a relativistic framework, even if they are modified by “relativistic corrections,” so that, say, $m_{0}=\mu+\mathcal{O}(c^{-2})$ , $M_{0}=M+\mathcal{O}(c^{-2})$ , because the reference level (and accumulation point for $N,J\to\infty$ ) of the real relativistic levels (2.15) will be the total rest-mass-energy $Mc^{2}$ , and will therefore be completely different from the reference level $m_{0}c^{2}\simeq\mu c^{2}$ of the effective relativistic energy levels. This difference in the relativistic reference energy level shows that, while it is very natural to require the straightforward identifications (4.1) of the action variables, the mapping between $E_{real}$ and $E_{0}$ must be more subtle.

One might a priori think that the most natural relativistic generalization of the usual non-relativistic rules for defining an effective one-body problem consists in requiring that

$$
\mathcal {E} _ {0} (\mathcal {N} _ {0}, \mathcal {J} _ {0}) = \mathcal {E} _ {\text { real }} (\mathcal {N}, \mathcal {J}) - c _ {0}, \tag {4.3}
$$

with a properly chosen constant $c_{0}=Mc^{2}-m_{0}c^{2}$ taking care of the shift in reference level. The rule (4.3) is equivalent to requiring the identification of the “non-relativistic” Hamiltonians (with subtraction of the rest-mass contribution)

$$
H _ {0} ^ {\mathrm{NR}} (\boldsymbol {q} ^ {\prime}, \boldsymbol {p} ^ {\prime}) = H _ {\text { real }} ^ {\mathrm{NR}} (\boldsymbol {q}, \boldsymbol {p}), \tag {4.4}
$$

where the canonical coordinates in each problem must be mapped [because of the identification (4.1)] by a canonical transformation,

$$
\sum_ {i} p _ {i} d q ^ {i} = \sum_ {i} p _ {i} ^ {\prime} d q ^ {\prime i} + d g (q, q ^ {\prime}), \tag {4.5}
$$

with some “generating function” $g(q, q')$ .

We have explored the naive identification (4.3), or (4.4), and found that it was unsatisfactory. Indeed, one finds that it is impossible to require simultaneously that (i) the energy levels coincide modulo an overall shift (4.3), (ii) the effective mass $m_{0}$ coincides with the usual reduced mass $\mu$

$= m_{1}m_{2} / (m_{1} + m_{2})$ , and (iii) the effective metric (3.3) depends only on $m_{1}$ and $m_{2}$ . [This impossibility comes from the fact that the requirement (4.4) is a very strong constraint which imposes more equations than unknowns.] If one insists on imposing the naive identification (4.3), there is a price to pay: one must drop at least one of the requirements (ii) or (iii). Various possibilities are discussed in the Appendixes of this paper. One possibility is to drop the requirement that $m_{0} = \mu$ . As discussed in Appendix A, we find that there is a unique choice of masses in the effective problem, namely,

$$
m _ {0} = \mu \xi^ {- 2}, \quad G M _ {0} = G M \xi^ {3}, \tag {4.6}
$$

with

$$
\xi^ {2} = \frac {1}{5} [ 2 \sqrt {1 0 0 + 3 0 \nu + 4 \nu^ {2}} - 1 5 + \nu ], \tag {4.7}
$$

which is compatible with the requirements (i) and (iii) above. However, we feel that it is quite unnatural to introduce an effective mass $m_{0}$ which differs from $\mu$ even in the non-relativistic limit $c\to+\infty$ . We feel also that this possibility is so constrained that it is only available at the 2PN level and will not be generalizable to higher post-Newtonian orders.

A second (formal) possibility is to introduce some energy dependence, either in $m_0$ , say

$$
m _ {0} = \mu \left[ 1 + \beta_ {1} \frac {\mathcal {E} _ {0} ^ {\mathrm{NR}}}{\mu c ^ {2}} + \beta_ {2} \left(\frac {\mathcal {E} _ {0} ^ {\mathrm{NR}}}{\mu c ^ {2}}\right) ^ {2} + \dots \right], \tag {4.8}
$$

or in the effective metric (3.3). Namely, the various coefficients $a_{1}, b_{1}, a_{2}, b_{2}, a_{3}, \ldots$ in Eq. (3.4) can be expanded as

$$
a _ {1} (\mathcal {E} _ {0}) = a _ {1} ^ {(0)} + a _ {1} ^ {(2)} \frac {\mathcal {E} _ {0} ^ {\mathrm{NR}}}{m _ {0} c ^ {2}} + a _ {1} ^ {(4)} \left(\frac {\mathcal {E} _ {0} ^ {\mathrm{NR}}}{m _ {0} c ^ {2}}\right) ^ {2} + \dots , \tag {4.9}
$$

etc. These possibilities are discussed, for completeness, in Appendix B.

Though the trick of introducing an energy dependence in (both) $m_{0}$ and the effective potential has been advocated, and used, in the quasi-potential approach of Todorov and coworkers [3,4], we feel that it is unsatisfactory. Conceptually, it obscures very much the nature of the mapping between the two problems, and, technically, it renders very difficult the generalization (we are interested in) to the case where radiation damping is taken into account (and where the energy is no longer conserved). We find much more satisfactory to drop the naive requirement (4.3), and to replace it by the more general requirement that there exist a certain one-to-one mapping between the real energy levels and the effective ones, say

$$
\mathcal {E} _ {0} (\mathcal {N} _ {0}, \mathcal {J} _ {0}) = f [ \mathcal {E} _ {\text { real }} (\mathcal {N}, \mathcal {J}) ]. \tag {4.10}
$$

In explicit, expanded form, the requirement (4.10) yields a deformed version of Eq. (4.3):

$$
\frac {\mathcal {E} _ {0} ^ {\mathrm{NR}}}{m _ {0} c ^ {2}} = \frac {\mathcal {E} _ {\text { real }} ^ {\mathrm{NR}}}{\mu c ^ {2}} \left[ 1 + \alpha_ {1} \frac {\mathcal {E} _ {\text { real }} ^ {\mathrm{NR}}}{\mu c ^ {2}} + \alpha_ {2} \left(\frac {\mathcal {E} _ {\text { real }} ^ {\mathrm{NR}}}{\mu c ^ {2}}\right) ^ {2} + \dots \right]. \tag {4.11}
$$

Here, we assume that the standard identification (4.3) holds [together with $m_0 = \mu +\mathcal{O}(c^{-2})]$ in the non-relativistic limit $c\rightarrow \infty$

We are going to show that the a priori arbitrary function f, i.e. the parameters $\alpha_{1},\alpha_{2},\ldots$ , can be uniquely selected (at the 2PN level) by imposing the following physically natural requirements: (a) the mass of the effective test particle coincides with the usual reduced mass,

$$
m _ {0} = \mu , \tag {4.12}
$$

and (b) the linearized (“one-graviton-exchange”) effective metric coincides with the linearized Schwarzschild metric with mass $M \equiv m_{1} + m_{2}$ , i.e.

$$
a _ {1} = - 2 G M, \quad b _ {1} = 2 G M. \tag {4.13}
$$

Note that the requirement (4.12) is actually imposed by dimensional analysis as soon as one requires $m_{0}=\mu + \mathcal{O}(c^{-2})$ . Indeed, as we bar any dependence on the energy, it is impossible to write any correction terms $\mathcal{O}(c^{-2})$ in the link between $m_{0}$ and $\mu$ . The requirement (4.13) is very natural when one thinks that the role of the effective metric is to reproduce, at all orders in the coupling constant G, the interaction generated by exchanging gravitons between two masses $m_{1}$ and $m_{2}$ . The “one-graviton-exchange” interaction (linear in $Gm_{1}m_{2}$ ) depends only on the (Lorentz-invariant) relative velocity and corresponds to a linearized Schwarzschild effective metric in the test-mass limit $\nu\to0$ . As the coefficient $-\frac{1}{2}a_{1}$ is fixed (by dimensional analysis, as above) to its non-relativistic value $-\frac{1}{2}a_{1}m_{0}=GM_{0}m_{0}=Gm_{1}m_{2}$ , it is very natural not to deform the coefficient $b_{1}$ by $\nu$ -dependent corrections.

Let us now prove the consistency of the requirements (4.12), (4.13) and determine the energy mapping f. We can start from the result (3.13), in which one replaces $E_{0}^{NR}$ by the expansion (4.11). This leads again to an expression of the form (3.13), with $E_{0}^{NR}$ replaced by $E_{real}^{NR}$ . One can simplify this expression by working with scaled variables:

$$
\hat {I} _ {R} ^ {0} \equiv \frac {I _ {R} ^ {0}}{\alpha_ {0}}, \quad \hat {I} _ {R} ^ {\text {real}} \equiv \frac {I _ {R} ^ {\text {real}}}{\alpha} \equiv i _ {r}, \quad E _ {0} \equiv \frac {\mathcal {E} _ {0} ^ {\text {NR}}}{m _ {0}}, \quad E _ {\text {real}} \equiv \frac {\mathcal {E} _ {\text {real}} ^ {\text {NR}}}{\mu},
$$

$$
j _ {0} \equiv \frac {\mathcal {J} _ {0}}{\alpha_ {0}}, \quad j \equiv \frac {\mathcal {J}}{\alpha}. \tag {4.14}
$$

Here $\alpha_0\equiv GM_0m_0$ and $\alpha \equiv GM\mu \equiv Gm_1m_2$ as above. We use also the scaled metric coefficients $\hat{a}_i$ and $\hat{b}_i$ of Eq. (3.18). Let us note, in passing, that, very generally, the dimensionless quantity $\hat{\mathcal{E}}_0 / c^2\equiv \mathcal{E}_0 / (m_0c^2) = 1 + c^{-2}E_0$ is expressible entirely in terms of the dimensionless scaled action variables $\hat{I}_a^0 /c = I_a^0 /\left(\alpha_0c\right)$ and of the dimensionless scaled metric coefficients $\hat{a}_i,\hat{b}_i$ . [This scaling behavior can be proved very easily by scaling from the start the effective action $S_0 = -\int m_0cds_0^{\mathrm{eff}} = -\alpha_0c\int ds_0^{\mathrm{eff}}$ with $d\hat{s}_0^2\equiv (GM_0)^{-2}ds_0^2$ , and by using scaled coordinates: $\hat{R} = R / GM_0$ , $\hat{t} = t / GM_0$ .]

Let us now make use of the assumptions $m_{0}=\mu$ and $GM_{0}\equiv-\frac{1}{2}a_{1}=GM$ (so that $\alpha_{0}=GM_{0}m_{0}=GM\mu=\alpha$ ). But

let us not yet assume the second equation (4.13); i.e., let us assume $\hat{a}_{1}\equiv-2$ , but let us not yet assume any value for $\hat{b}_{1}\equiv b_{1}/GM_{0}\equiv b_{1}/GM$ . Within these assumptions, the scaled version of the result (3.13), with $E_{0}^{NR}$ replaced by Eq. (4.11), reads

$$
\begin{array}{l} \hat {I} _ {R} ^ {0} (E _ {0} (E _ {\mathrm{real}}), j _ {0}) = - j _ {0} + \frac {1}{\sqrt {- 2 E _ {\mathrm{real}}}} \left[ \hat {A} + \hat {B} ^ {\prime} \frac {E _ {\mathrm{real}}}{c ^ {2}} \right. \\ \left. + \hat {C} ^ {\prime} \left(\frac {E _ {\mathrm{real}}}{c ^ {2}}\right) ^ {2} \right] \\ + \frac {1}{c ^ {2} j _ {0}} \left[ \hat {D} + \hat {E} \frac {E _ {\text { real }}}{c ^ {2}} \right] + \frac {1}{c ^ {4} j _ {0} ^ {3}} \hat {F}, \tag {4.15} \\ \end{array}
$$

where

$$
\hat {A} = - \frac {1}{2} \hat {a} _ {1} = 1, \quad \hat {B} ^ {\prime} = \frac {7}{4} + \hat {b} _ {1} - \frac {\alpha_ {1}}{2},
$$

$$
\hat {C} ^ {\prime} = \frac {1 9}{3 2} + \frac {\hat {b} _ {1}}{4} + \frac {\alpha_ {1}}{2} \left(\hat {b} _ {1} + \frac {7}{4}\right) + \frac {3}{8} \alpha_ {1} ^ {2} - \frac {\alpha_ {2}}{2}, \tag {4.16}
$$

and where $\hat{D}$ , $\hat{E}$ and $\hat{F}$ are obtained from the expressions (3.14) by the replacements $a_i \to \hat{a}_i$ , $b_i \to \hat{b}_i$ (with $\hat{a}_1 = -2$ ). Finally, identifying $[I_R^0(\mathcal{E}_0, \mathcal{J}_0)]_{\mathcal{J}_0 = \mathcal{J}_{\text {real}}}^{\mathcal{E}_0 = f(\mathcal{E}_{\text {real}})}$ with $I_R(\mathcal{E}_{\text {real}}, \mathcal{J}_{\text {real}})$ or, equivalently, $\hat{I}_R^0(E_0(E_{\text {real}}), j_0)$ with $\hat{I}_R(E_{\text {real}}, j_0)$ yields five equations to be satisfied, namely the equations stating that $\hat{B}'$ , $\hat{C}'$ , $\hat{D}$ , $\hat{E}$ and $\hat{F}$ coincide with the corresponding coefficients in Eq. (2.14). The explicit form of these equations is

$$
\frac {7}{4} + \hat {b} _ {1} - \frac {\alpha_ {1}}{2} = \frac {1 5}{4} - \frac {\nu}{4}, \tag {4.17}
$$

$$
\frac {1 9}{3 2} + \frac {\hat {b} _ {1}}{4} + \frac {\alpha_ {1}}{2} \left(\hat {b} _ {1} + \frac {7}{4}\right) + \frac {3}{8} \alpha_ {1} ^ {2} - \frac {\alpha_ {2}}{2} = \frac {3 5}{3 2} + \frac {1 5}{1 6} \nu + \frac {3}{3 2} \nu^ {2}, \tag {4.18}
$$

$$
2 - \frac {\hat {a} _ {2}}{2} + \frac {\hat {b} _ {1}}{2} = 3, \tag {4.19}
$$

$$
4 - \hat {a} _ {2} + \hat {b} _ {1} - \frac {\hat {b} _ {1} ^ {2}}{8} + \frac {\hat {b} _ {2}}{2} = \frac {1 5}{2} - 3 \nu , \tag {4.20}
$$

$$
6 - 3 \hat {a} _ {2} + \frac {\hat {a} _ {2} ^ {2}}{8} - \frac {\hat {a} _ {3}}{2} + \hat {b} _ {1} - \frac {1}{4} \hat {a} _ {2} \hat {b} _ {1} - \frac {\hat {b} _ {1} ^ {2}}{1 6} + \frac {\hat {b} _ {2}}{4} = \frac {3 5}{4} - \frac {5}{2} \nu . \tag {4.21}
$$

Note that the subsystem made of the two equations (4.17), (4.18) (corresponding to $\hat{B}'$ and $\hat{C}'$ ) contains the three unknowns $\hat{b}_1, \alpha_1, \alpha_2$ , while the three equations (4.19)-(4.21) (corresponding to $\hat{D}$ , $\hat{E}$ and $\hat{F}$ ) contain the unknowns $\hat{b}_1$ ,

$\hat{b}_{2}, \hat{a}_{2}, \hat{a}_{3}$ . In this section we shall consider only the first (“BC”) subsystem, leaving the “DEF” system to the next section.

It is easily seen that the BC subsystem would admit no solution in $\hat{b}_{1}$ if we were to impose $\alpha_{1}=\alpha_{2}=0$ . This proves the assertion made above that one needs a non-trivial energy mapping $\mathcal{E}_{0}=f(\mathcal{E}_{\mathrm{real}})$ . On the other hand, if we introduce the two free parameters $\alpha_{1},\alpha_{2}$ , the BC subsystem becomes an indeterminate system of two equations for three unknowns. As argued above, it is physically very natural to impose that the linearized effective metric coincide with the linearized Schwarzschild metric, i.e. that

$$
\hat {b} _ {1} = 2. \tag {4.22}
$$

Then the BC system (4.17),(4.18) admits the unique solution

$$
\alpha_ {1} = \frac {\nu}{2}, \quad \alpha_ {2} = 0. \tag {4.23}
$$

This solution corresponds to the link

$$
\frac {\mathcal {E} _ {0} ^ {\mathrm{NR}}}{m _ {0} c ^ {2}} = \frac {\mathcal {E} _ {\text { real }} ^ {\mathrm{NR}}}{\mu c ^ {2}} \left(1 + \frac {\nu}{2} \frac {\mathcal {E} _ {\text { real }} ^ {\mathrm{NR}}}{\mu c ^ {2}}\right), \tag {4.24}
$$

which is equivalent to

$$
\frac {\mathcal {E} _ {0}}{m _ {0} c ^ {2}} \equiv \frac {\mathcal {E} _ {\text { real }} ^ {2} - m _ {1} ^ {2} c ^ {4} - m _ {2} ^ {2} c ^ {4}}{2 m _ {1} m _ {2} c ^ {4}}. \tag {4.25}
$$

Remarkably, the map (4.25) between the real total relativistic energy $E_{real}=Mc^{2}+E_{real}^{NR}$ , and the effective relativistic energy $E_{0}=m_{0}c^{2}+E_{0}^{NR}$ coincides with the one introduced by Brézin, Itzykson and Zinn-Justin [1], which maps very simply the one-body relativistic Balmer formula onto the two-body one (in quantum electrodynamics). The same map was also recently used by Damour, Iyer and Sathyaprakash [21]. There it was emphasized that the function $\varphi(s)$ of the Mandelstam invariant $s=E_{real}^{2}$ appearing on the right-hand side (RHS) of Eq. (4.25) is the most natural symmetric function of the asymptotic $^{1}$ 4-momenta $p_{1}^{\mu}, p_{2}^{\mu}$ of a two-particle system which reduces, in the test-mass limit $m_{2}\ll m_{1}$ , to the energy of $m_{2}$ in the rest-frame of $m_{1}$ . Indeed (setting here c=1 for simplicity),

$$
\varphi (s) \equiv \frac {s - m _ {1} ^ {2} - m _ {2} ^ {2}}{2 m _ {1} m _ {2}} = \frac {- \left(p _ {1} + p _ {2}\right) ^ {2} - m _ {1} ^ {2} - m _ {2} ^ {2}}{2 m _ {1} m _ {2}} = - \frac {p _ {1} \cdot p _ {2}}{m _ {1} m _ {2}}. \tag {4.26}
$$

Finally, we have two a priori independent motivations for using the function $\varphi(s)$ , i.e. the link (4.25), to map the real two-body energy onto the effective one-body one: (i) the simplicity, and the symmetry, of the expression (4.26) which generalizes the test-mass conserved energy $E_{0}/m_{0}=$

$-K_{\mu}p_{0}^{\mu}/m_{0}$ (where $K_{\mu}$ is the Killing vector defined by the time direction of the background field) (see [21]), and (ii) the fact that it corresponds to a linearized effective metric coinciding with the linearized Schwarzschild metric. Actually, these two facts are not really independent, because (as discussed in [1] and [2]) they correspond heuristically to saying that the “effective interaction” is the interaction felt by any of the two particles in the rest frame of the other particle.

Summarizing, the rules we shall assume for relating the real two-body problem to the effective one-body one are Eqs. (4.1) [or equivalently the condition (4.5) that the phase-space coordinates be canonically mapped] and Eq. (4.25).

# V. EFFECTIVE ONE-BODY METRIC AND THE DYNAMICS IT DEFINES

Having specified the rules linking the real two-body problem to the effective one-body one, we can now proceed to the determination of the effective metric (at the 2PN level). We shall work in Schwarzschild coordinates:

$$
d s _ {\text { eff }} ^ {2} = - A (R) c ^ {2} d t ^ {2} + B (R) d R ^ {2} + R ^ {2} \left(d \theta^ {2} + \sin^ {2} \theta d \varphi^ {2}\right), \tag {5.1}
$$

with $A(R)$ and $B(R)$ constructed as expansions of the form (3.4). It will be useful to rewrite also the effective metric in the form

$$
d s _ {\text { eff }} ^ {2} = - A (R) c ^ {2} d t ^ {2} + \frac {D (R)}{A (R)} d R ^ {2} + R ^ {2} (d \theta^ {2} + \sin^ {2} \theta d \varphi^ {2}), \tag {5.2}
$$

in which we factorize, in the manner of Schwarzschild, $g_{00}^{-1}$ in front of the $dR^{2}$ term, and consider that, besides $A(R)$ , the second function constructed as an expansion in 1/R is

$$
D (R) = A (R) B (R) = 1 + \frac {d _ {1}}{c ^ {2} R} + \frac {d _ {2}}{c ^ {4} R ^ {2}} + \dots , \tag {5.3}
$$

where

$$
d _ {1} = a _ {1} + b _ {1}, \quad d _ {2} = a _ {2} + a _ {1} b _ {1} + b _ {2}. \tag {5.4}
$$

To determine the effective metric, i.e. the coefficients $\hat{a}_{i}$ and $\hat{b}_{i}$ or, equivalently, $\hat{a}_{i}$ and $\hat{d}_{i}\equiv d_{i}/(GM)^{i}$ , we insert the known values of $\hat{b}_{1}$ , $\alpha_{1}$ and $\alpha_{2}$ (namely $\hat{b}_{1}=2$ , $\alpha_{1}=\nu/2$ , $\alpha_{2}=0$ ) into the remaining equations (4.19)–(4.21) (“DEF system”). This yields three equations for the three unknowns $\hat{a}_{2}$ , $\hat{b}_{2}$ and $\hat{a}_{3}$ . The unique solution of this DEF system reads

$$
\hat {a} _ {2} = 0, \quad \hat {a} _ {3} = 2 \nu , \quad \hat {b} _ {2} = 4 - 6 \nu . \tag {5.5}
$$

In other words, our natural assumptions (4.12),(4.13) have led us uniquely to the simple energy map (4.25) and to an effective one-body metric given by

$$
A (R) = 1 - \frac {2 G M}{c ^ {2} R} + 2 \nu \left(\frac {G M}{c ^ {2} R}\right) ^ {3} + \dots , \tag {5.6}
$$

$$
B (R) = 1 + \frac {2 G M}{c ^ {2} R} + (4 - 6 \nu) \left(\frac {G M}{c ^ {2} R}\right) ^ {2} + \dots , \tag {5.7}
$$

$$
D (R) = 1 - 6 \nu \left(\frac {G M}{c ^ {2} R}\right) ^ {2} + \dots . \tag {5.8}
$$

The simplicity of the final results (5.6)–(5.8) is striking. The effective metric (5.2) is a simple deformation of the Schwarzschild metric $[A_{s}(R)=1-2GM/c^{2}R, D_{s}(R)=1]$ with deformation parameter $\nu$ . Note also that there are no $\nu$ -dependent corrections to $A(R)$ at the 1PN level, i.e. no $\nu(GM/c^{2}R)^{2}$ contribution to $A(R)$ . The first $\nu$ -dependent corrections enter at the 2PN level. Remembering that the (2PN) effective metric fully encodes the information contained in the complicated 2PN expressions (2.14) or (2.15), it is remarkable that the metric coefficients (5.6)–(5.8) are so simple. The previous approach of Ref. [4] led to much more complicated expressions at the 1PN level (to which it was limited).

In this paper, we propose to trust the physical consequences of the effective metric (5.2), with $A(R)$ given by Eq. (5.6) and $D(R)$ given by Eq. (5.8), even in the region where R is of order of a few times $GM/c^{2}$ . Note that even in the extreme case where $\nu=1/4$ and $R\simeq2GM/c^{2}$ the $\nu$ -dependent additional terms entering the effective metric remain relatively small: indeed, in this case, $\delta_{\nu}A(R)=2\nu(GM/c^{2}R)^{3}=1/16$ and $-\delta_{\nu}D(R)=6\nu(GM/c^{2}R)^{2}=3/8$ . We expect, therefore, that it should be a fortiori possible to trust the predictions of the effective metric (5.2) near the innermost stable circular orbit, i.e. around $R\simeq6GM/c^{2}$ [where $\delta_{\nu}A(R)\simeq2\times10^{-3}$ and $-\delta_{\nu}D(R)\simeq4\times10^{-2}$ ]. Note that this nice feature of having only a small deformation of the Schwarzschild metric, even when $\nu=1/4$ , is not shared by the “hybrid” approach of Kidder, Will and Wiseman [22]. Indeed, as emphasized in Ref. [21], the $\nu$ deformations considered in the hybrid approach are, for some coefficients, larger than unity when $\nu=1/4$ . This is related to the fact pointed out by Wex and Schäfer [23,24] that, by applying the hybrid approach of [22] to the Hamiltonian, instead of the equations of motion, one gets significantly different predictions.

Let us note also that, if we decide to write the effective metric in the form (5.2), the existence of a simple zero in the function $A(R)$ , say $A(R_H) = 0$ , implies [if $D(R_H) \neq 0$ , and $D(R) > 0$ for $R > R_H$ ] that the hypersurface $R = R_H$ is (like in the undeformed Schwarzschild case) a regular (Killing) horizon. As usual, one can define Kruskal-like coordinates to see explicitly the regular nature of the horizon $R = R_H$ (made of two intersecting null hypersurfaces). In our case, one checks easily that the function $A_{2\mathrm{PN}}(R)$ defined by the first three terms on the RHS of Eq. (5.6) admits a simple zero $^2$ at some $R_H(\nu)$ , when $0 \leqslant \nu \leqslant \frac{1}{4}$ . The position $R_H(\nu)$ of this

“effective horizon” smoothly, and monotonically, evolves with the deformation parameter $\nu$ between $R_{H}(0)=2GM/c^{2}$ and

$$
R _ {H} (1 / 4) \simeq 0. 9 2 7 7 (2 G M / c ^ {2}). \tag {5.9}
$$

This relatively small change of the horizon toward a smaller value, i.e. a smaller horizon area (to quote an invariant measure of the location of the horizon), suggests that the dynamics of trajectories in the effective metric will also be only a small deformation of the standard Schwarzschild case.

One of the main aims of the present work is indeed to study the dynamics (and the energetics) in the effective metric $(5.2)$ . In particular, as gravitational radiation damping is known to circularize binary orbits, we are especially interested in studying the stable circular orbits in the effective metric. A convenient tool for doing this is to introduce an effective potential $[28,29]$ . Note that the Hamilton-Jacobi equation $(3.9)$ yields

$$
\left(\frac {\mathcal {E} _ {0}}{m _ {0} c ^ {2}}\right) ^ {2} = W _ {\mathcal {J} _ {0}} (R) + \frac {A (R)}{B (R)} \left(\frac {P _ {R}}{m _ {0} c}\right) ^ {2} \geqslant W _ {\mathcal {J} _ {0}} (R), \tag {5.10}
$$

where $P_{R} \equiv \partial S_{eff}/\partial R$ is the effective radial momentum, and where the “effective radial potential” $W_{\mathcal{J}_{0}}(R)$ is defined as

$$
W _ {\mathcal {J} _ {0}} (R) \equiv A (R) \left[ 1 + \frac {\left(\mathcal {J} _ {0} / m _ {0} c\right) ^ {2}}{C (R) R ^ {2}} \right]. \tag {5.11}
$$

We read also from Eq. (5.10) the relativistic effective Hamiltonian

$$
\begin{array}{l} H _ {0} ^ {R} (R, P _ {R}, P _ {\varphi}) \\ = m _ {0} c ^ {2} \sqrt {A (R) \left[ 1 + \frac {P _ {R} ^ {2}}{m _ {0} ^ {2} c ^ {2} B (R)} + \frac {P _ {\varphi} ^ {2}}{m _ {0} ^ {2} c ^ {2} C (R) R ^ {2}} \right]}, \\ \equiv m _ {0} c ^ {2} \sqrt {W _ {P _ {\varphi}} (R) + \frac {A (R)}{B (R)} \left(\frac {P _ {R}}{m _ {0} c}\right) ^ {2}}. \tag {5.12} \\ \end{array}
$$

The coordinate angular frequency along circular orbits is obtained by differentiating the Hamiltonian, that is

$$
\omega_ {0} \equiv \left(\frac {d \varphi}{d t}\right) _ {\text { circ }} = \left(\frac {\partial H _ {0} ^ {R} (R , P _ {R} , P _ {\varphi})}{\partial P _ {\varphi}}\right) _ {P _ {R} = 0}, \tag {5.13}
$$

which gives, explicitly (using $P_{\varphi} = J_{0}$ ),

$$
\omega_ {0} = \frac {\mathcal {J} _ {0}}{m _ {0} C (R) R ^ {2}} \frac {\sqrt {A (R)}}{\sqrt {1 + \frac {\mathcal {J} _ {0} ^ {2}}{m _ {0} ^ {2} c ^ {2} C (R) R ^ {2}}}}. \tag {5.14}
$$

Equations (5.11) and (5.14) are valid in an arbitrary radial coordinate gauge, but we shall use them in the Schwarzschild gauge where the metric coefficient $C(R) \equiv 1$ . Note that $W(R)$ and $\omega_{0}$ then depend only on the metric coefficient $A(R)$ . In dimensionless scaled variables $\hat{R} \equiv c^{2} R / (GM)$ , $j_{0}$

![](images/493097f90a580dfd4f4633ff5c3cfe8c7a6e601abd7f41fb0906165b465c27ec.jpg)

<details>
<summary>line</summary>

| c²R/GM | w_j (j₁) | w_j (j₂) | w_j (j_*) |
| ------ | -------- | -------- | --------- |
| 2.0    | 1.0      | 1.0      | 0.8       |
| 6.0    | 0.9      | 0.9      | 0.85      |
| 10.0   | 0.9      | 0.9      | 0.9       |
| 14.0   | 0.9      | 0.9      | 0.92      |
| 20.0   | 0.9      | 0.9      | 0.95      |
| 26.0   | 0.9      | 0.9      | 0.97      |
| 30.0   | 0.9      | 0.9      | 0.98      |
</details>

FIG. 1. The effective radial potential $W_{j}(R)$ (at the 2PN level and for $\nu = 1/4$ ) versus the dimensionless radial variable $c^2 R/(GM)$ for three different values of the dimensionless angular momentum $j = c\mathcal{J}_{\mathrm{real}} / (GM\mu)$ . Note that the effective radial potential tends to 1 for $R \to \infty$ . The stable circular orbits are located at the minima of the effective potential and are indicated by the solid circles. The innermost stable circular orbit corresponds to the critical value $j_*$ . In the case of the $j_1$ curve the orbit of a particle with energy $E_0^R = \hat{\mathcal{E}}_0$ is an elliptical rosette.

$\equiv c\mathcal{J}_{0}/(GM\mu)$ , $\hat{\omega}_{0}\equiv GM\omega_{0}/c^{3}$ (in our case $M_{0}=M$ and $m_{0}=\mu$ ), the effective potential and the orbital frequency (along circular orbits) are quite simple:

$$
W _ {j _ {0}} (\hat {R}) = A (\hat {R}) \left[ 1 + \frac {j _ {0} ^ {2}}{\hat {R} ^ {2}} \right],
$$

$$
\hat {\omega} _ {0} = \frac {j _ {0}}{\hat {R} ^ {2}} \frac {\sqrt {A (\hat {R})}}{\sqrt {1 + \frac {j _ {0} ^ {2}}{\hat {R} ^ {2}}}}. \tag {5.15}
$$

If we define the 2PN-accurate $A(R)$ by the straightforward truncation of Eq. (5.6), namely

$$
A _ {2 \mathrm{PN}} (\hat {R}) = 1 - \frac {2}{\hat {R}} + \frac {2 \nu}{\hat {R} ^ {3}}; \tag {5.16}
$$

$W_{j_{0}}$ is a fifth-order polynomial in $u \equiv 1/\hat{R} \equiv GM/(c^{2}R)$ . As the analytical study of the extrema of $W_{j_{0}}$ is rather complicated, we have used a numerical approach. When $\nu$ varies between 0 and 1/4, $W_{j_{0}}$ evolves into a smoothly deformed version of the standard Schwarzschild effective potential. To illustrate this fact, we plot, in Fig. 1, $W_{j_{0}}(\hat{R})$ for $\nu = \frac{1}{4}$ and for various values of the dimensionless angular momentum $j_{0}$ . Note that the latter quantity coincides (in view of our rules) with the corresponding real two-body dimensionless angular momentum $j$ :

$$
j _ {0} \equiv \frac {c \mathcal {J} _ {0}}{G M _ {0} m _ {0}} = \frac {c \mathcal {J} _ {\mathrm{real}}}{G M \mu} \equiv j. \tag {5.17}
$$

(Note that our definition of the $j$ 's differs by a factor of $c$ from the one used in the previous section.)

As usual, because of the inequality (5.10), when j and $\hat{\mathcal{E}}_{0}\equiv\mathcal{E}_{0}/(m_{0}c^{2})$ are fixed, the trajectory of a particle following a geodesic in the effective metric (5.2) can be qualitatively read in Fig. 1. For instance, in the case illustrated for the $j_{1}$ curve ( $E_{0}^{R}\equiv\hat{\mathcal{E}}_{0}$ line), the orbit will be an elliptical rosette, with the radial variable oscillating between a minimum and a maximum (solid line in Fig. 1). The stable circular orbits are located at the minima of the effective potential (the maxima being unstable circular orbits). The ISCO corresponds to the critical value $j_{*}$ of the angular momentum where the maximum and the minimum of the effective potential fuse together to form an horizontal inflection point:

$$
\frac {\partial W _ {j _ {*}}}{\partial \hat {R} _ {*}} = 0 = \frac {\partial^ {2} W _ {j _ {*}}}{\partial \hat {R} _ {*} ^ {2}}. \tag {5.18}
$$

Let us, for comparison with our deformed case, recall the standard results for circular orbits in a Schwarzschild spacetime [28,29]. With the notation $u \equiv GM_0 / c^2 R$ (for a Schwarzschild metric of mass $M_0$ ), the location, orbital frequency $^3$ and energy of circular orbits are given, when $j$ varies, by

$$
u = \frac {1}{6} \left[ 1 - \sqrt {1 - \frac {1 2}{j ^ {2}}} \right], \tag {5.19}
$$

$$
\hat {\omega} _ {S} \equiv \frac {G M _ {0}}{c ^ {3}} \omega = u ^ {3 / 2}, \tag {5.20}
$$

$$
\hat {\mathcal {E}} _ {S} \equiv \left(\frac {\mathcal {E} _ {0}}{m _ {0} c ^ {2}}\right) ^ {S} = j (1 - 2 u) u ^ {1 / 2}. \tag {5.21}
$$

The ISCO corresponds to the critical values

$$
j _ {*} ^ {S} = \sqrt {1 2}, \quad u _ {*} ^ {S} = \frac {1}{6}, \quad \hat {\omega} _ {*} ^ {S} = \frac {1}{6 \sqrt {6}}, \quad \hat {\mathcal {E}} _ {*} ^ {S} = \sqrt {\frac {8}{9}}. \tag {5.22}
$$

In the deformed Schwarzschild case defined by Eq. (5.16), the ISCO for the extreme case $\nu=\frac{1}{4}$ is numerically found to correspond to the values

$$
j _ {*} ^ {2 \mathrm{PN}} \equiv \left(\frac {c \mathcal {J} _ {\text { real }}}{G M \mu}\right) _ {\text { ISCO }} = 3. 4 0 4 = 0. 9 8 3 j _ {*} ^ {S}, \tag {5.23}
$$

$$
u _ {0 *} ^ {2 \mathrm{PN}} \equiv \left(\frac {G M}{c ^ {2} R}\right) _ {\text { ISCO }} = 0. 1 7 4 9 = 1. 0 4 9 u _ {*} ^ {S}, \tag {5.24}
$$

$$
\hat {\omega} _ {0 *} ^ {2 \mathrm{PN}} \equiv \left(\frac {G M \omega_ {0}}{c ^ {3}}\right) _ {\text { ISCO }} = 0. 0 7 2 3 0 = 1. 0 6 3 \hat {\omega} _ {*} ^ {S}, \tag {5.25}
$$

$$
\hat {\mathcal {E}} _ {0 *} ^ {2 \mathrm{PN}} \equiv \left(\frac {\mathcal {E} _ {0}}{\mu c ^ {2}}\right) _ {\text { ISCO }} = 0. 9 4 0 4 0 = 0. 9 9 7 4 4 \hat {\mathcal {E}} _ {*} ^ {S}. \tag {5.26}
$$

Note that the Schwarzschild-coordinate radius of the effective ISCO is (when $\nu=1/4$ ) $R^{ISCO}=5.718GM/c^{2}$ , i.e. lower than the standard Schwarzschild value $6GM/c^{2}$ corresponding to the total mass $M=m_{1}+m_{2}$ . This is consistent with the fact that the effective horizon was drawn in below $2GM/c^{2}$ when $\nu$ was turned on. Note, however, that the three quantities $u_{0}^{2PN}$ , $\omega_{0}^{2PN}$ and $E_{0}^{2PN}$ entering equations (5.24)–(5.26) are mathematical quantities defined in the effective problem, and not physical quantities defined in the real problem (hence the subscript 0 added as a warning). [By contrast, $j^{2PN}$ , Eq. (5.23) is directly related to the real, two-body angular momentum.] For physical (and astrophysical) purposes, we need to transform the information contained in Eqs. (5.24)–(5.26) into numbers concerning physical quantities defined in the real, two-body problem. For the energy, this is achieved (by definition) by using Eq. (4.25) to compute the real, two-body total energy $E_{real}$ . Explicitly, the solution of Eq. (4.25) is (see also [21])

$$
\mathcal {E} _ {\text { real }} = M c ^ {2} \sqrt {1 + 2 \nu \left(\frac {\mathcal {E} _ {0} - m _ {0} c ^ {2}}{m _ {0} c ^ {2}}\right)}. \tag {5.27}
$$

We need also to transform the effective orbital frequency $\omega_{0}$ . This is easily done as follows. We know that the Hamiltonians of the real and effective problems are related by a mapping

$$
H _ {\text { real }} (I _ {a} ^ {\text { real }}) = h (H _ {0} (I _ {a} ^ {0})), \tag {5.28}
$$

where $a = R, \theta, \varphi$ (for the 3-dimensional problem), and where the function $h$ [the inverse of the function $f$ of Eq. (4.10)] is, in our case, explicitly defined by Eq. (5.27). On the other hand, we know that the action variables are identically mapped onto each other: $I_a^0 = I_a^{\text{real}}$ (canonical transformation). The frequency of the motion of any separated degree of freedom is given by the general formulas $\omega_a^0 = \partial H_0(I^0) / \partial I_a^0$ , $\omega_a^{\text{real}} = \partial H_{\text{real}}(I^{\text{real}}) / \partial I_a^{\text{real}}$ , where the Hamiltonians are considered as functions of the canonically conjugate action-angle variables $(I_a, \theta_a)$ (remembering that for such integrable systems the Hamiltonian does not depend on the $\theta$ 's). Therefore the frequencies of the real problem are all obtained from the frequencies of the effective one by a common, energy-dependent factor

$$
\frac {\omega_ {a} ^ {\text { real }}}{\omega_ {a} ^ {0}} = \frac {d t _ {0}}{d t ^ {\text { real }}} = \frac {d H _ {\text { real }}}{d H _ {0}} = \frac {\partial h (H _ {0})}{\partial H _ {0}}. \tag {5.29}
$$

In our case this “blueshift” $^{4}$ factor reads

![](images/7d8eb28652e3e2bbf847b3220e9fb34a948d394200a062c2c85911e83581283c.jpg)

![](images/52264249a050df3f48506323008ccae9b5a0f1f04debe0886a2941b9acb366c1.jpg)

<details>
<summary>line</summary>

| 4 v | j/js  |
| --- | ----- |
| 0.0 | 1.000 |
| 1.0 | 0.983 |
</details>

FIG. 2. Variation with $\nu$ (at the 2PN level) of the ISCO values of the real non-relativistic energy $E_{\mathrm{real}} \equiv \hat{\mathcal{E}}_{\mathrm{real}}^{\mathrm{NR}} \equiv (\mathcal{E}_{\mathrm{real}} - Mc^{2}) / \mu c^{2}$ (on the left) and of the real angular momentum $j \equiv c\mathcal{J}_{\mathrm{real}} / GM\mu$ (on the right), divided by the corresponding Schwarzschild values $|E_{\mathrm{S}}| \equiv |\hat{\mathcal{E}}_{\mathrm{S}}^{\mathrm{NR}}| = 1 - \sqrt{8/9} \simeq 0.05719$ and $j_{\mathrm{S}} = \sqrt{12}$ , respectively.

$$
\frac {\omega_ {a} ^ {\text { real }}}{\omega_ {a} ^ {0}} = \frac {d t _ {0}}{d t ^ {\text { real }}} = \frac {1}{\sqrt {1 + 2 \nu (\mathcal {E} _ {0} - m _ {0} c ^ {2}) / m _ {0} c ^ {2}}}. \tag {5.30}
$$

As indicated in Eqs. (5.29) and (5.30) the same energy-dependent “blueshift” factor maps the effective and the real times (along corresponding orbits). Note that we have here a simple generalization of the spatial canonical transformation $dp \wedge dq = dp_{0} \wedge dq_{0}$ to the time domain $dH \wedge dt = dH_{0} \wedge dt_{0}$ .

Applying the transformations (5.27) and (5.29), we obtain the physical quantities $^{5}$ predicted by our effective 2PN metric, still in the extreme case $\nu=1/4$ ,

$$
\hat {\omega} _ {\text { real } *} ^ {2 \mathrm{PN}} = \left(\frac {G M}{c ^ {3}} \omega_ {\text { real }}\right) _ {\text { ISCO }} = 1. 0 7 9 \hat {\omega} _ {*} ^ {S} = 0. 0 7 3 4 0, \tag {5.31}
$$

$$
\left(\frac {\mathcal {E} _ {\text {real}} ^ {2 \mathrm{PN}} - M c ^ {2}}{\mu c ^ {2}}\right) _ {\text {ISCO}} = 1. 0 5 0 (\hat {\mathcal {E}} _ {*} ^ {S} - 1) = - 0. 0 6 0 0 5. \tag {5.32}
$$

We represent in Figs. 2 and 3 the variation with $\nu$ of the ISCO values of the real non-relativistic energy, $E_{\mathrm{real}} \equiv \hat{\mathcal{E}}_{\mathrm{real}}^{\mathrm{NR}} \equiv (\mathcal{E}_{\mathrm{real}} - Mc^2) / \mu c^2$ , the real angular momentum, $j \equiv c\mathcal{J}_{\mathrm{real}} / GM\mu$ , and of the quantity

$$
z \equiv \left(\frac {G M}{c ^ {3}} \omega_ {\text { real }}\right) ^ {- 2 / 3}, \tag {5.33}
$$

which is an invariant measure of the radial position of the orbit, and which coincides with the scaled Schwarzschild

radius $\hat{R}=c^{2}R/(GM)$ in the test-mass limit $\nu\to0$ . One checks that our ISCO values respect the “black hole limit” $J_{real}<GE_{real}^{2}/c^{5}$ , so that the system does not need to radiate a lot of gravitational waves in the final coalescence before being able to settle down as a black hole.

Let us now briefly compare our predictions with previous ones in the literature. The first attempt to address the question of the ISCO for binary systems of comparable masses was made by Clark and Eardley [30]. They worked only at the 1PN level, and predicted that the ISCO should be significantly more tightly bound than in the Schwarzschild case (with $M_0 = M = m_1 + m_2$ ): $\mathcal{E}_{\mathrm{CE}}^{\mathrm{NR}} / \mu c^2 \simeq -0.1$ when $\nu = 1/4$ , compared to $\mathcal{E}_{\mathrm{Schwarz}}^{\mathrm{NR}} / m_0c^2 = \sqrt{8/9} - 1 \simeq -0.0572$ . Blackburn and Detweiler [31] used an initial value formalism (which is only a rough approximation, even in the test-mass limit) to predict an extremely tight ISCO when $\nu = 1/4$ : $\mathcal{E}_{\mathrm{BD}}^{\mathrm{NR}} / \mu c^2 \simeq -0.7$ . Kidder, Will and Wiseman (KWW) [22] were the first to try to use the full 2PN information contained in the Damour-Deruelle equations of motion (1.1) to estimate analytically the change of the ISCO brought by turning on a finite mass ratio $\nu$ . They introduced a “hybrid” approach in which one re-sums exactly the “Schwarzschild” ( $\nu$ -

![](images/ee5d1dae78c593b2d091c7ec94ad6b13fd9267323cdb4307edf5e81b7fc08fff.jpg)

<details>
<summary>line</summary>

| 4 v | z / z_s |
| --- | ------- |
| 0.0 | 1.000   |
| 1.0 | 0.950   |
</details>

FIG. 3. ISCO values (at the 2PN level) of the quantity $z = (GM\omega_{\mathrm{real}} / c^3)^{-2 / 3}$ , divided by the Schwarzschild value $z_S = 6$ , versus $\nu$ .

![](images/41f546d9424e5b74779e20565a0976e40e6148b781556ee6b2b3f9fccbc6d1b4.jpg)  
FIG. 4. ISCO values (for $\nu = 1/4$ ) of the real non-relativistic energy $E \equiv \hat{\mathcal{E}}_{\mathrm{real}}^{\mathrm{NR}}$ , divided by the corresponding Schwarzschild value $E_S \equiv \hat{\mathcal{E}}_{\mathrm{S}}^{\mathrm{NR}}$ , versus $z/z_S$ . On the left we have compared our predictions at the 1PN level ( $\blacksquare$ ) and 2PN level ( $\blacklozenge$ ) with the results obtained in [21] ( $\blacktriangleright$ ) and [22] ( $\blacktriangleleft$ ). The (\*) indicates the Schwarzschild predictions. The right panel is a magnification of the part of the left one in which we analyze the robustness of our method by exhibiting the points ( $\bullet$ ) obtained by introducing in the effective metric reasonable 3PN and 4PN contributions: $(a_4', a_5') = (\pm 4, -4)$ , $(\pm 4, 0)$ and $(\pm 4, +4)$ in the notation of Eq. (5.34).

independent) terms in the equations of motion, and treats the $\nu$ -dependent terms as additional corrections. In contrast with our present 2PN-effective approach (and also with the less reliable previous studies [30,31]), they predict $^{6}$ that, when $\nu$ increases, the ISCO becomes markedly less tightly bound: e.g. $\mathcal{E}_{\mathrm{KWW}}^{\mathrm{NR}} / \mu c^{2} \simeq -0.0377$ when $\nu = 1/4$ . If their trend were real, this would imply that, except for the very stiff equations of state of nuclear matter (leading to large neutron star radii), the final plunge triggered when the ISCO is reached by an inspirating ( $1.4M_{\odot} + 1.4M_{\odot}$ ) neutron star binary would probably take place before tidal disruption. However, both the robustness and the consistency of the hybrid approach of [22] have been questioned. Wex and Schäfer [23] showed that the predictions of the hybrid approach were not “robust” in that they could be significantly modified by applying this approach to the Hamiltonian, rather than to the equations of motion. Schäfer and Wex [24] further showed that the predictions of the hybrid approach were not robust under a change of coordinate system. Moreover, Ref. [21] has questioned the consistency of the hybrid approach by pointing out that the formal “ $\nu$ corrections” represent, in several cases, a very large (larger than 100%) modification of the corresponding $\nu$ -independent terms. This unreliability of the hybrid approach casts a doubt on the ISCO estimates of Ref. [25] which are based on hybrid orbital terms, and which use only 1PN accuracy in most terms.

Damour, Iyer and Sathyaprakash (DIS) [21] have introduced (at the 2PN level) another analytical approach to the determination of the ISCO, based on the Padé approximants of some invariant energy function [closely related to the energy transformation (4.25)]. Their trend is consistent with the one found in the present paper, namely a more tightly bound ISCO: for $\nu = 1/4$ , the Padé approximant approach predicts $\mathcal{E}_{\mathrm{DIS}}^{\mathrm{NR}} / \mu c^2 \simeq -0.0653$ .

Numerical methods have recently been used to try to locate the ISCO for binary neutron stars $[26,27]$ . However, we do not think that the truncation of Einstein's field equations (to a conformally flat spatial metric) used in these works is a good approximation for close orbits. Indeed, at the 2PN approximation, some numerically significant terms in the interaction potential come from the transverse-traceless part of the metric $[13,7,10]$ . Moreover, the (unrealistic) assumption used in these works that the stars are corotating has probably also a significant effect on the location of the ISCO by adding both spin-orbit and spin-spin interaction terms.

This large scatter in the predictions for the location of the ISCO for comparable masses poses the question of the “robustness” of our new, effective-action approach. The main problem can be formulated as follows. Assuming that the effective-action approach (for the time-symmetric part of the dynamics) makes sense at higher post-Newtonian levels, the “exact” effective function $A(R)$ will read

$$
\begin{array}{l} A (R) = 1 - 2 \left(\frac {G M}{c ^ {2} R}\right) + 2 \nu \left(\frac {G M}{c ^ {2} R}\right) ^ {3} + \nu a _ {4} ^ {\prime} \left(\frac {G M}{c ^ {2} R}\right) ^ {4} \\ + \nu a _ {5} ^ {\prime} \left(\frac {G M}{c ^ {2} R}\right) ^ {5} + \dots . \tag {5.34} \\ \end{array}
$$

The question is then to know how sensitive is the location of the ISCO to the values of the (still unknown) coefficients $a_4', a_5', \ldots$ . One should have some a priori idea of the reasonable range of values of $a_4', a_5', \ldots$ . A rationale for deciding upon the reasonable values of $a_4'$ is the following. At the 2PN level, it is formally equivalent to use (with $u \equiv GM / c^2 R$ ) $A_{2\mathrm{PN}} = 1 - 2u + 2\nu u^3$ or the factorized form $A_{2\mathrm{PN}}' = (1 - 2u)(1 + 2\nu u^3)$ . However, $A_{2\mathrm{PN}}' = A_{2\mathrm{PN}} - 4\nu u^4$ which corresponds to $a_4' = -4$ . This suggests that $-4 \leqslant a_4' \leqslant +4$ is a reasonable range. We shall also consider $-4 \leqslant a_5' \leqslant +4$ as a plausible range. Note that both choices correspond to having coefficients of $u^n$ which vary between $-1$ and $+1$ when $\nu = 1/4$ . The robustness of our effective-action

TABLE I. Summary of the ISCO values used in Fig. 4 ( $\nu=1/4$ ). Note that we give here $E_{real}^{NR}/Mc^{2}$ , that is the ratio between the energy that can be radiated in gravitational waves before the final plunge and the total mass-energy initially available. The first row refers to the naive estimate defined by a test particle of mass $\mu$ in a Schwarzschild spacetime of mass M. We show also in the last column the solar-mass-scaled orbital frequency $f_{\odot}$ defined by $f_{\mathrm{real}}=\omega_{\mathrm{real}}/(2\pi)\equiv f_{\odot}(M_{\odot}/M)$ . 

<table><tr><td>Method</td><td> $\mathcal{E}_{\text{real}}^{\text{NR}}/Mc^{2}$ </td><td>z</td><td> $\hat{\omega}_{\text{real}}$ </td><td> $f_{\odot}$ (kHz)</td></tr><tr><td>“Schwarzschild”</td><td>-0.01430</td><td>6</td><td>0.06804</td><td>2.199</td></tr><tr><td>Eff. action 1PN</td><td>-0.01440</td><td>5.942</td><td>0.06904</td><td>2.231</td></tr><tr><td>Eff. action 2PN</td><td>-0.01501</td><td>5.704</td><td>0.07340</td><td>2.372</td></tr><tr><td>Eff. action  $(a_{4}^{\prime}, a_{5}^{\prime}) = (-4, -4)$ </td><td>-0.01462</td><td>5.891</td><td>0.06994</td><td>2.260</td></tr><tr><td>Eff. action  $(a_{4}^{\prime}, a_{5}^{\prime}) = (-4, 0)$ </td><td>-0.01469</td><td>5.854</td><td>0.07061</td><td>2.267</td></tr><tr><td>Eff. action  $(a_{4}^{\prime}, a_{5}^{\prime}) = (-4, +4)$ </td><td>-0.01476</td><td>5.815</td><td>0.07131</td><td>2.304</td></tr><tr><td>Eff. action  $(a_{4}^{\prime}, a_{5}^{\prime}) = (+4, -4)$ </td><td>-0.01530</td><td>5.583</td><td>0.07582</td><td>2.450</td></tr><tr><td>Eff. action  $(a_{4}^{\prime}, a_{5}^{\prime}) = (+4, 0)$ </td><td>-0.01540</td><td>5.531</td><td>0.07688</td><td>2.484</td></tr><tr><td>Eff. action  $(a_{4}^{\prime}, a_{5}^{\prime}) = (+4, +4)$ </td><td>-0.01551</td><td>5.475</td><td>0.07806</td><td>2.522</td></tr><tr><td>DIS [21]</td><td>-0.01633</td><td>5.036</td><td>0.08850</td><td>2.860</td></tr><tr><td>KWW [22]</td><td>-0.00943</td><td>6.49</td><td>0.0605</td><td>1.96</td></tr></table>

predictions against the introduction of $a_{4}^{\prime}$ and $a_{5}^{\prime}$ is illustrated in Fig. 4. The numerical values used in Fig. 4 are exhibited in Table I.

Figure 4 plots the ratio $E / |E_S|$ where $E \equiv \mathcal{E}_{\mathrm{real}}^{\mathrm{NR}} / \mu c^2 \equiv (\mathcal{E}_{\mathrm{real}} - Mc^2) / \mu c^2$ at the ISCO (for $\nu = 1/4$ ) and $E_S = \sqrt{(8/9)} - 1 \simeq -0.05719$ is the corresponding “Schwarzschild” value, versus $z/z_S$ where $z$ is defined in Eq. (5.33), and where $z_S = 6$ . This figure compares the predictions of Ref. [22], of Ref. [21] and of our new, effective-action prediction (at the 2PN level). We have also added what would be the prediction of the effective-action approach at the 1PN level. Note that, at the 1PN level, the function $A(R)$ , Eq. (5.6), exactly coincides with the Schwarzschild one, but that the energy mapping (4.24) introduces a slight deviation from the test-mass limit. Figure 4 exhibits also the points obtained when considering $(a_4', a_5') = (\pm 4, -4)$ , $(\pm 4, 0)$ and $(\pm 4, +4)$ . We see in this figure that the main prediction of the present approach [a prediction already clear from the fact that the 2PN contribution to $A(R)$ is fractionally small], namely that the ISCO is only slightly more bound than in the test-mass limit, is robust under the addition of higher PN contributions. The sensitivity to $a_4'$ of the binding energy is only at the $\sim 3\%$ level (for $a_4' = \pm 4$ ), while its sensitivity to the 4PN coefficient $a_5'$ is further reduced to the $\sim 0.6\%$ level (for $a_5' = \pm 4$ ). Still, it would be important to determine the 3PN coefficient $a_4'$ to refine the determination of the ISCO quantities.

# VI. EXPLICIT MAPPING BETWEEN THE REAL PROBLEM AND THE EFFECTIVE ONE

The basic idea of the effective one-body approach is to map the complicated and badly convergent PN expansion of the dynamics of a two-body system onto a simpler auxiliary one-body problem. We have shown in the previous sections that by imposing some simple, coordinate-invariant requirements, we could uniquely determine that the one-body dynamics was defined (at the 2PN level) by geodesic motion in a certain deformed Schwarzschild spacetime. The latter dynamics can be solved exactly by means of quadratures [e.g. by using the Hamilton-Jacobi method: see Eqs. (3.7)-(3.12)]. Note that this exact solution defines a particular resummation of the original 2PN-expanded dynamics. The hope (which we tried to substantiate in Sec. V) is that this re-summation captures, with sufficient approximation, the crucial non-perturbative aspects of the two-body dynamics, such as the existence of an ISCO.

As all the current work about the equations of motion and/or the gravitational-wave radiation of binary systems is done in some specific coordinate systems (harmonic or ADM), we need to complete the (coordinate-invariant) work done in the previous sections by explicitly constructing the transformation which maps the variables entering the effective problem onto those of the real one. We have already mentioned that the transformation between harmonic and ADM coordinates has been explicitly worked out in Refs. [10] and [11]. Here, we shall explicitly relate the ADM phase-space variables $Q=q_{1}-q_{2}$ and $P=\partial S/\partial Q$ of the relative motion (as defined in Sec. II above) to the coordinate and momenta of the effective problem. More precisely, we shall construct the map

$$
q ^ {\prime i} = \mathcal {Q} ^ {i} (q ^ {j}, p _ {j}), \quad p _ {i} ^ {\prime} = \mathcal {P} _ {i} (q ^ {j}, p _ {j}), \tag {6.1}
$$

transforming the reduced ADM relative position and momenta $(q^{i}, p_{i})$ , defined in Eq. (2.4), into the corresponding reduced Cartesian-like position and momenta $(q^{'i}, p_{i}^{'})$ canonically defined by the (Schwarzschild-gauge) effective action (3.2). In other words,

$$
q ^ {\prime i} = \frac {Q ^ {\prime i}}{G M}, \quad p _ {i} ^ {\prime} = \frac {P _ {i} ^ {\prime}}{\mu}, \tag {6.2}
$$

with $Q^{\prime1}=R\sin\theta\cos\varphi$ , $Q^{\prime2}=R\sin\theta\sin\varphi$ , $Q^{\prime3}=R\cos\theta$ , and $P_{i}^{\prime}=\partial S_{eff}/\partial Q^{\prime i}$ . Here, the “effective” coordinates $R,\theta,\varphi$ are those of Eq. (5.1) (in Schwarzschild gauge) and

$S_{\mathrm{eff}} = -\int \mu cd s_{\mathrm{eff}}$ . The corresponding effective Hamiltonian (with respect to the coordinate time $t$ of the effective problem) is easily found by solving $g_{\mathrm{eff}}^{\mu \nu}(Q') P_{\mu}^{\prime} P_{\nu}^{\prime} + m_{0}^{2} c^{2} = 0$ in terms of the energy $\mathcal{E}_{0} = -P_{0}'$ . Transforming the usual polar-coordinate result [equivalent to Eq. (5.10)] into Cartesian coordinates leads to

$$
\begin{array}{l} H _ {\mathrm{eff}} (Q ^ {\prime}, P ^ {\prime}) \\ = \mu c ^ {2} \sqrt {A (Q ^ {\prime}) \left[ 1 + \frac {(n ^ {\prime} \cdot P ^ {\prime}) ^ {2}}{\mu^ {2} c ^ {2} B (Q ^ {\prime})} + \frac {(n ^ {\prime} \times P ^ {\prime}) ^ {2}}{\mu^ {2} c ^ {2}} \right]}, \tag {6.3} \\ \end{array}
$$

where $Q' \equiv \sqrt{\delta_{ij} Q'^i Q'^j} = R$ , where $n'^i = Q'^i / Q'$ is the unit vector in the radial direction, and where the scalar and vector products are performed as in Euclidean space. When scaling the effective coordinates as in Eq. (6.2), we need to scale correspondingly the time variable, the Hamiltonian and the action of the effective problem:

$$
\hat {t} \equiv \frac {t}{G M}, \quad \hat {H} _ {\mathrm{eff}} \equiv \frac {H _ {\mathrm{eff}}}{\mu}, \quad \hat {S} _ {\mathrm{eff}} \equiv \frac {S _ {\mathrm{eff}}}{\mu G M}. \tag {6.4}
$$

Note that the effective Hamiltonian (6.3) contains the rest-mass contribution. The scaled version of Eq. (6.3) simplifies to

$$
\begin{array}{l} \hat {H} _ {\mathrm{eff}} (\pmb {q} ^ {\prime}, \pmb {p} ^ {\prime}) \\ = c ^ {2} \sqrt {A (q ^ {\prime}) \left[ 1 + \frac {\boldsymbol {p} ^ {\prime 2}}{c ^ {2}} + \frac {(\boldsymbol {n} ^ {\prime} \cdot \boldsymbol {p} ^ {\prime}) ^ {2}}{c ^ {2}} \left(\frac {1}{B (q ^ {\prime})} - 1\right) \right]}, \tag {6.5} \\ \end{array}
$$

where $q' \equiv \sqrt{\delta_{ij} q'^i q'^j} = R / GM$ and $n'^i \equiv q'^i / q'$ . As was mentioned above the identification of the action variables in the real and effective problems guarantees that the two problems are mapped by a canonical transformation, i.e. a transformation such that Eq. (4.5) is satisfied. It will be more convenient to replace the generating function $g(q, q')$ of Eq. (4.5) by the new generating function $\tilde{G}(q, p') = g(q, q') + p'_i q'^i$ such that

$$
p _ {i} d q ^ {i} + q ^ {\prime i} d p _ {i} ^ {\prime} = d \widetilde {G} (q, p ^ {\prime}). \tag {6.6}
$$

We can further separate $\widetilde{G}(q,p^{\prime})$ into $\widetilde{G}_{\mathrm{id}}(q,p^{\prime})\equiv q^{i}p_{i}^{\prime}$ , which generates the identity transformation, and an additional (perturbative) contribution $G(q,p^{\prime})$ :

$$
\widetilde {G} (q, p ^ {\prime}) = q ^ {i} p _ {i} ^ {\prime} + G (q, p ^ {\prime}),
$$

$$
G (q, p ^ {\prime}) = \frac {1}{c ^ {2}} G _ {1 \mathrm{PN}} (q, p ^ {\prime}) + \frac {1}{c ^ {4}} G _ {2 \mathrm{PN}} (q, p ^ {\prime}). \tag {6.7}
$$

Equations (6.6),(6.7) yield the link

$$
q ^ {\prime i} = q ^ {i} + \frac {\partial G (q , p ^ {\prime})}{\partial p _ {i} ^ {\prime}}, \quad p _ {i} ^ {\prime} = p _ {i} - \frac {\partial G (q , p ^ {\prime})}{\partial q ^ {i}}. \tag {6.8}
$$

Note that Eqs. (6.8) are exact and determine $q'$ and p in function of q and $p'$ . We have, however, written them in a form appropriate for determining, by successive iteration, $q'$ and $p'$ in function of q and p. If needed (e.g. for applications of the present work to the direct numerical calculation of the effective dynamics in the original q, p coordinates), it is numerically fast to iterate Eqs. (6.8) to get Eqs. (6.1). For our present purpose we need an explicit analytical approximation of Eqs. (6.1) at the 2PN level. Remembering that G starts at order $1/c^{2}$ , one easily finds that

$$
q ^ {\prime i} = q ^ {i} + \frac {\partial G (q , p)}{\partial p _ {i}} - \frac {\partial G (q , p)}{\partial q ^ {j}} \frac {\partial^ {2} G (q , p)}{\partial p _ {j} \partial p _ {i}} + \mathcal {O} \left(\frac {1}{c ^ {6}}\right),
$$

$$
p _ {i} ^ {\prime} = p _ {i} - \frac {\partial G (q , p)}{\partial q ^ {i}} + \frac {\partial G (q , p)}{\partial q ^ {j}} \frac {\partial^ {2} G (q , p)}{\partial p _ {j} \partial q ^ {i}} + \mathcal {O} \left(\frac {1}{c ^ {6}}\right). \tag {6.9}
$$

In the terms linear in $G(q,p)$ one needs to use the full (1PN+2PN) expression of $G(q,p)$ , while in the quadratic terms it is enough to use $G_{1\mathrm{PN}} / c^2$ .

To determine the generating function $G(q,p)$ we need to write the equation stating that, under the canonical transformation (6.8), the effective Hamiltonian $H_{\mathrm{eff}}(q',p')$ is mapped into a function of q and p which is linked to the real (relativistic) Hamiltonian $H_{\mathrm{real}}^{R}(q,p)$ by our rule (4.25). If we write this link in terms of the reduced effective Hamiltonian (6.5), and of the reduced, non-relativistic real Hamiltonian $\hat{H}_{\mathrm{real}}^{\mathrm{NR}} \equiv (H_{\mathrm{real}}^{R} - Mc^{2}) / \mu$ [the same as $\hat{H}$ appearing in Eqs. (2.5),(2.6) above], it reads

$$
\begin{array}{l} 1 + \frac {\hat {H} _ {\mathrm{real}} ^ {\mathrm{NR}} (q , p)}{c ^ {2}} \left(1 + \frac {\nu}{2} \frac {\hat {H} _ {\mathrm{real}} ^ {\mathrm{NR}} (q , p)}{c ^ {2}}\right) \\ = \frac {1}{c ^ {2}} \hat {H} _ {\text { eff }} [ q ^ {\prime} (q, p), p ^ {\prime} (q, p) ]. \tag {6.10} \\ \end{array}
$$

Actually, we found it more convenient to work with the square of Eq. (6.10), so as to get rid of the square root in $\hat{H}_{\mathrm{eff}}$ , Eq. (6.5). Hence, writing (half) the square of Eq. (6.10), and Taylor-expanding $\hat{H}_{\mathrm{eff}}[q'(q,p),p'(q,p)]$ using Eqs. (6.7)-(6.9), we get, at order $1 / c^4$ , the following partial differential equation for $G_{1\mathrm{PN}}(q,p)$ :

$$
\begin{array}{l} \frac {\partial \hat {H} _ {\text { Newt }}}{\partial q ^ {i}} \frac {\partial G _ {1 \mathrm{PN}}}{\partial p _ {i}} - \frac {\partial \hat {H} _ {\text { Newt }}}{\partial p _ {i}} \frac {\partial G _ {1 \mathrm{PN}}}{\partial q ^ {i}} = \frac {\nu}{2} \boldsymbol {p} ^ {4} - (1 + \nu) \frac {\boldsymbol {p} ^ {2}}{q} \\ + \left(1 - \frac {\nu}{2}\right) \frac {(\boldsymbol {n} \cdot \boldsymbol {p}) ^ {2}}{q} + \left(1 + \frac {\nu}{2}\right) \frac {1}{q ^ {2}}, \tag {6.11} \\ \end{array}
$$

where we have denoted the Newtonian Hamiltonian as $\hat{H}_{Newt} \equiv \hat{H}_{0} = p^{2}/2 - 1/q$ [see Eq. (2.6a)]. At order $1/c^{6}$ , a more complex calculation gives the partial differential equation for $G_{2PN}(q, p)$ ,

$$
\begin{array}{l} \frac {\partial \hat {H} _ {\mathrm{Newt}}}{\partial q ^ {i}} \frac {\partial G _ {2 \mathrm{PN}}}{\partial p _ {i}} - \frac {\partial \hat {H} _ {\mathrm{Newt}}}{\partial p _ {i}} \frac {\partial G _ {2 \mathrm{PN}}}{\partial q ^ {i}} \\ = \frac {\nu}{2} \hat {H} _ {0} ^ {3} + (1 + \nu) \hat {H} _ {0} \hat {H} _ {2} + \hat {H} _ {4} \\ - (2 + 3 \nu) \frac {(\boldsymbol {n} \cdot \boldsymbol {p}) ^ {2}}{q ^ {2}} - \frac {\nu}{q ^ {3}} + \frac {\partial \mathcal {R}}{\partial q ^ {i}} \frac {\partial G _ {\mathrm{1PN}}}{\partial p _ {i}} - \frac {\partial \mathcal {R}}{\partial p _ {i}} \frac {\partial G _ {\mathrm{1PN}}}{\partial q ^ {i}} \\ + \frac {\partial G _ {1 \mathrm{PN}}}{\partial q ^ {j}} \frac {\partial^ {2} G _ {1 \mathrm{PN}}}{\partial p _ {j} \partial p _ {i}} \frac {\partial \hat {H} _ {\mathrm{Newt}}}{\partial q ^ {i}} - \frac {\partial G _ {1 \mathrm{PN}}}{\partial q ^ {j}} \frac {\partial^ {2} G _ {1 \mathrm{PN}}}{\partial p _ {j} \partial q ^ {i}} \frac {\partial \hat {H} _ {\mathrm{Newt}}}{\partial p _ {i}} \\ - \frac {1}{2} \frac {\partial G _ {\mathrm{1PN}}}{\partial p _ {i}} \frac {\partial G _ {\mathrm{1PN}}}{\partial p _ {j}} \frac {\partial^ {2} \hat {H} _ {\mathrm{Newt}}}{\partial q ^ {i} \partial q ^ {j}} \\ - \frac {1}{2} \frac {\partial G _ {1 \mathrm{PN}}}{\partial q ^ {i}} \frac {\partial G _ {1 \mathrm{PN}}}{\partial q ^ {j}} \frac {\partial^ {2} \hat {H} _ {\text { Newt }}}{\partial p _ {i} \partial p _ {j}}, \tag {6.12} \\ \end{array}
$$

where $\hat{H}_{2}$ and $\hat{H}_{4}$ are given by Eqs. (2.6b),(2.6c), while

$$
\mathcal {R} = \frac {1}{q} [ (\boldsymbol {n} \cdot \boldsymbol {p}) ^ {2} + \boldsymbol {p} ^ {2} ]. \tag {6.13}
$$

The partial differential equations (6.11) and (6.12) have the general form

$$
\frac {\partial \hat {H} _ {\text { Newt }}}{\partial q ^ {i}} \frac {\partial G _ {n}}{\partial p _ {i}} - \frac {\partial \hat {H} _ {\text { Newt }}}{\partial p _ {i}} \frac {\partial G _ {n}}{\partial q ^ {i}} = \frac {q ^ {i}}{q ^ {3}} \frac {\partial G _ {n}}{\partial p _ {i}} - p _ {i} \frac {\partial G _ {n}}{\partial q _ {i}} = K _ {n} (q, p), \tag {6.14}
$$

where, at each PN order n=1PN or 2PN, the RHS is a known source term $K_{n}(q,p)$ . Note that the LHS of Eq. (6.14) is the Poisson brackets $\{\hat{H}_{\mathrm{Newt}},G_{n}\}$ or, equivalently, minus the time derivative of $G_{n}$ along the Newtonian motion. It is easily checked that the solution of Eq. (6.14) is unique modulo the addition of terms generating a constant time shift or a spatial rotation. [Indeed, the homogeneous scalar solutions of Eq. (6.14) must correspond to the scalar constants of motion of the Keplerian motion: $\hat{H}_{\mathrm{Newt}}(\boldsymbol{q},\boldsymbol{p})$ and $(\boldsymbol{q}\times\boldsymbol{p})^{2}$ .] If we require (as we can) that $G(q,p)$ change sign when q or (separately) p change sign, the generating function is uniquely fixed. In particular, at 1PN level, by looking at the structure of the source terms, i.e. the RHS of Eq. (6.11), we can prove in advance that $G_{1PN}$ must be of the form

$$
G _ {1 \mathrm{PN}} (\boldsymbol {q}, \boldsymbol {p}) = (\boldsymbol {q} \cdot \boldsymbol {p}) \left[ \alpha_ {1} \boldsymbol {p} ^ {2} + \frac {\beta_ {1}}{q} \right]. \tag {6.15}
$$

Inserting Eq. (6.15) into the equation to be satisfied, Eq. (6.11) gives a system of four equations for the two unknown coefficients $\alpha_{1}$ and $\beta_{1}$ . Two of these equations give directly the values $\alpha_{1}$ and $\beta_{1}$ ,

$$
\alpha_ {1} = - \frac {\nu}{2}, \quad \beta_ {1} = 1 + \frac {\nu}{2}, \tag {6.16}
$$

while the two redundant equations

$$
\alpha_ {1} - \beta_ {1} = - 1 - \nu , \quad 2 \alpha_ {1} + \beta_ {1} = 1 - \frac {\nu}{2} \tag {6.17}
$$

are identically satisfied by the solution (6.16).

Using these 1PN results we can go further and evaluate the 2PN-source term $K_{2}(q,p)$ in Eq. (6.14):

$$
\begin{array}{l} K _ {2} (q, p) = - \frac {\nu}{8} (1 + 3 \nu) \boldsymbol {p} ^ {6} + \frac {\nu}{8} (- 1 + 8 \nu) \frac {\boldsymbol {p} ^ {4}}{q} \\ - \frac {\nu}{4} (9 + \nu) \frac {(\boldsymbol {n} \cdot \boldsymbol {p}) ^ {2} \boldsymbol {p} ^ {2}}{q} + \frac {3}{8} \nu (8 + 3 \nu) \frac {(\boldsymbol {n} \cdot \boldsymbol {p}) ^ {4}}{q} \\ + \frac {1}{8} (- 2 + 1 6 \nu - 7 \nu^ {2}) \frac {\pmb {p} ^ {2}}{q ^ {2}} \\ + \frac {1}{8} (4 + 3 \nu^ {2}) \frac {(\boldsymbol {n} \cdot \boldsymbol {p}) ^ {2}}{q ^ {2}} + \frac {1}{4} (1 - 7 \nu + \nu^ {2}) \frac {1}{q ^ {3}}. \tag {6.18} \\ \end{array}
$$

By looking at the structures in Eq. (6.18) we deduce that the most general form of $G_{2PN}$ is

$$
G _ {2 \mathrm{PN}} (\boldsymbol {q}, \boldsymbol {p}) = (\boldsymbol {q} \cdot \boldsymbol {p}) \left[ \alpha_ {2} \boldsymbol {p} ^ {4} + \frac {1}{q} \left(\beta_ {2} \boldsymbol {p} ^ {2} + \gamma_ {2} (\boldsymbol {n} \cdot \boldsymbol {p}) ^ {2}\right) + \frac {\delta_ {2}}{q ^ {2}} \right]. \tag {6.19}
$$

Inserting the ansatz (6.19) and the 1PN results in Eq. (6.12), we get again more equations than unknowns:

$$
- \alpha_ {2} + \frac {\nu}{8} + \frac {3}{8} \nu^ {2} = 0, \quad \alpha_ {2} - \beta_ {2} + \frac {\nu}{8} - \nu^ {2} = 0,
$$

$$
4 \alpha_ {2} + \beta_ {2} - 3 \gamma_ {2} + \frac {9}{4} \nu + \frac {\nu^ {2}}{4} = 0, \quad 3 \gamma_ {2} - 3 \nu - \frac {9}{8} \nu^ {2} = 0,
$$

$$
\frac {1}{4} + \beta_ {2} - \delta_ {2} - 2 \nu + \frac {7}{8} \nu^ {2} = 0,
$$

$$
- \frac {1}{2} + 2 \beta_ {2} + 2 \delta_ {2} + 3 \gamma_ {2} - \frac {3}{8} \nu^ {2} = 0,
$$

$$
- \frac {1}{4} + \delta_ {2} + \frac {7}{4} \nu - \frac {\nu^ {2}}{4} = 0. \tag {6.20}
$$

As it should (in view of the work of the previous sections) one finds that all the redundant equations can be satisfied. The final, unique solutions for the coefficients $\alpha_{2}$ , $\beta_{2}$ , $\gamma_{2}$ and $\delta_{2}$ are

$$
\alpha_ {2} = \frac {\nu + 3 \nu^ {2}}{8}, \quad \beta_ {2} = \frac {2 \nu - 5 \nu^ {2}}{8},
$$

$$
\gamma_ {2} = \frac {8 \nu + 3 \nu^ {2}}{8}, \quad \delta_ {2} = \frac {1 - 7 \nu + \nu^ {2}}{4}. \tag {6.21}
$$

Finally, we give the explicit form of the canonical transformation between the coordinates $(q,p)$ and $(q',p')$ at the 2PN level [see Eq. (6.9)]:

$$
\begin{array}{l} q ^ {\prime i} - q ^ {i} = \frac {1}{c ^ {2}} \left[ \left(1 + \frac {\nu}{2}\right) \frac {q ^ {i}}{q} - \frac {\nu}{2} q ^ {i} \pmb {p} ^ {2} - \nu p ^ {i} (\pmb {q} \cdot \pmb {p}) \right] \\ + \frac {1}{c ^ {4}} \left[ \nu \left(1 + \frac {\nu}{8}\right) \frac {q ^ {i} (\boldsymbol {q} \cdot \boldsymbol {p}) ^ {2}}{q ^ {3}} + \frac {\nu}{4} \left(5 - \frac {\nu}{2}\right) \frac {q ^ {i} \boldsymbol {p} ^ {2}}{q} + \frac {3}{2} \nu \left(1 - \frac {\nu}{2}\right) \frac {p ^ {i} (\boldsymbol {q} \cdot \boldsymbol {p})}{q} \right. \\ \left. + \frac {1}{4} (1 - 7 \nu + \nu^ {2}) \frac {q ^ {i}}{q ^ {2}} + \frac {\nu}{8} (1 - \nu) q ^ {i} \boldsymbol {p} ^ {4} + \frac {\nu}{2} (1 + \nu) p ^ {i} \boldsymbol {p} ^ {2} (\boldsymbol {q} \cdot \boldsymbol {p}) \right], (6.22) \\ p _ {i} ^ {\prime} - p _ {i} = \frac {1}{c ^ {2}} \left[ - \left(1 + \frac {\nu}{2}\right) \frac {p _ {i}}{q} + \frac {\nu}{2} p _ {i} \boldsymbol {p} ^ {2} + \left(1 + \frac {\nu}{2}\right) \frac {q _ {i} (\boldsymbol {q} \cdot \boldsymbol {p})}{q ^ {3}} \right] \\ + \frac {1}{c ^ {4}} \left[ \frac {\nu}{8} (- 1 + 3 \nu) p _ {i} \boldsymbol {p} ^ {4} + \frac {1}{4} (3 + 1 1 \nu) \frac {p _ {i}}{q ^ {2}} - \frac {3}{4} \nu \left(3 + \frac {\nu}{2}\right) \frac {p _ {i} \boldsymbol {p} ^ {2}}{q} \right. \\ + \frac {1}{4} (- 2 - 1 8 \nu + \nu^ {2}) \frac {q _ {i} (\boldsymbol {q} \cdot \boldsymbol {p})}{q ^ {4}} + \frac {\nu}{8} (1 0 - \nu) \frac {q _ {i} (\boldsymbol {q} \cdot \boldsymbol {p}) p ^ {2}}{q ^ {3}} \\ \left. - \frac {\nu}{8} (1 6 + 5 \nu) \frac {p _ {i} (\boldsymbol {q} \cdot \boldsymbol {p}) ^ {2}}{q ^ {3}} + \frac {3}{8} \nu (8 + 3 \nu) \frac {q _ {i} (\boldsymbol {q} \cdot \boldsymbol {p}) ^ {3}}{q ^ {5}} \right]. (6.23) \\ \end{array}
$$

Note that the $\nu\to0$ limit of Eq. (6.22) gives $q^{\prime i}=[1+1/(2c^{2}q)]^{2}q^{i}$ which is (as it should) the relation between “Schwarzschild” ( $q^{\prime}$ ) and “isotropic” (q) quasi-Cartesian coordinates in a Schwarzschild spacetime. (In this case, ADM=isotropic.) As a check on Eqs. (6.22),(6.23) we have verified that (at the 2PN level) $q^{\prime}\times p^{\prime}$ coincides with $q\times p$ . [They should coincide exactly, when solving exactly Eqs. (6.8) with any (spherically symmetric) generating function $G(q,p)$ .] Let us quote, for completeness, the partial derivatives of the generating function $G=c^{-2}G_{1PN}+c^{-4}G_{2PN}$ , which must be used to solve by successive iterations the exact equations (6.8) and determine $q^{\prime}$ and $p^{\prime}$ in terms of q and p:

$$
\frac {\partial G _ {1 \mathrm{PN}} (q , p)}{\partial q ^ {i}} = - \frac {\nu}{2} p _ {i} \boldsymbol {p} ^ {2} + \left(1 + \frac {\nu}{2}\right) \frac {p _ {i}}{q} - \left(1 + \frac {\nu}{2}\right) \frac {q _ {i} (\boldsymbol {q} \cdot \boldsymbol {p})}{q ^ {3}}, \tag {6.24}
$$

$$
\frac {\partial G _ {1 \mathrm{PN}} (q , p)}{\partial p _ {i}} = - \frac {\nu}{2} q ^ {i} \boldsymbol {p} ^ {2} + \left(1 + \frac {\nu}{2}\right) \frac {q ^ {i}}{q} - \nu p ^ {i} (\boldsymbol {q} \cdot \boldsymbol {p}), \tag {6.25}
$$

$$
\begin{array}{l} \frac {\partial G _ {2 \mathrm{PN}} (q , p)}{\partial q ^ {i}} = \frac {1}{8} \nu (1 + 3 \nu) p _ {i} \boldsymbol {p} ^ {4} + \frac {\nu}{8} (2 - 5 \nu) \frac {p _ {i} \boldsymbol {p} ^ {2}}{q} + \frac {3}{8} \nu (8 + 3 \nu) \frac {p _ {i} (\boldsymbol {q} \cdot \boldsymbol {p}) ^ {2}}{q ^ {3}} \\ - \frac {3}{8} \nu (8 + 3 \nu) \frac {q _ {i} (\boldsymbol {q} \cdot \boldsymbol {p}) ^ {3}}{q ^ {5}} + \frac {1}{4} (1 - 7 \nu + \nu^ {2}) \frac {p _ {i}}{q ^ {2}} - \frac {\nu}{8} (2 - 5 \nu) \frac {q _ {i} (\boldsymbol {q} \cdot \boldsymbol {p}) \boldsymbol {p} ^ {2}}{q ^ {3}} \\ - \frac {1}{2} (1 - 7 \nu + \nu^ {2}) \frac {q _ {i} (\boldsymbol {q} \cdot \boldsymbol {p})}{q ^ {4}}, \tag {6.26} \\ \end{array}
$$

$$
\frac {\partial G _ {2 \mathrm{PN}} (q , p)}{\partial p _ {i}} = \frac {1}{8} \nu (1 + 3 \nu) q ^ {i} \boldsymbol {p} ^ {4} + \frac {\nu}{8} (2 - 5 \nu) \frac {q ^ {i} \boldsymbol {p} ^ {2}}{q} + \frac {3}{8} \nu (8 + 3 \nu) \frac {q ^ {i} (\boldsymbol {q} \cdot \boldsymbol {p}) ^ {2}}{q ^ {3}}
$$

$$
+ \frac {1}{4} (1 - 7 \nu + \nu^ {2}) \frac {q ^ {i}}{q ^ {2}} + \frac {\nu}{2} (1 + 3 \nu) p ^ {i} \boldsymbol {p} ^ {2} (\boldsymbol {q} \cdot \boldsymbol {p}) + \frac {\nu}{4} (2 - 5 \nu) \frac {p ^ {i} (\boldsymbol {q} \cdot \boldsymbol {p})}{q}. \tag {6.27}
$$

# VII. INCLUSION OF RADIATION REACTION EFFECTS AND TRANSITION BETWEEN INSPIRAL AND PLUNGE

In the preceding sections we have limited our attention to the conservative (time-symmetric) part of the dynamics of a two-body system, i.e. the one defined, at the 2PN level, by neglecting $A_{a}^{reac}$ in Eq. (1.1). We expect that the separation of the dynamics in a conservative part plus a reactive part makes sense also at higher PN orders (though it probably gets blurred at some high PN level). However, there exists, at present, no algorithm defining precisely this separation. Anyway we shall content ourselves here to working at the 2.5PN level where this separation is well defined, as shown in Eq. (1.1). When dealing with the relative motion we find it convenient to continue using an Hamiltonian formalism. Schäfer [20,14,18] has shown how to treat radiation reaction effects within the ADM canonical formalism. His result (at the 2.5PN level) is that it is enough to use as Hamiltonian for the dynamics of two masses a time-dependent Hamiltonian obtained by adding to the conservative 2PN Hamiltonian $H_{2\mathrm{PN}}(\boldsymbol{q}_{1},\boldsymbol{q}_{2},\boldsymbol{p}_{1},\boldsymbol{p}_{2})$ the following “reactive” Hamiltonian:

$$
\begin{array}{l} H _ {\text { r   e   a   c }} \left(\boldsymbol {q} _ {1}, \boldsymbol {q} _ {2}, \boldsymbol {p} _ {1}, \boldsymbol {p} _ {2}; t\right) = - h _ {i j} ^ {T T \text { r   e   a   c }} (t) \left[ \frac {p _ {1} ^ {i} p _ {1} ^ {j}}{2 m _ {1}} + \frac {p _ {2} ^ {i} p _ {2} ^ {j}}{2 m _ {2}} \right. \\ \left. - \frac {1}{2} G m _ {1} m _ {2} \frac {\left(q _ {1} ^ {i} - q _ {2} ^ {i}\right) \left(q _ {1} ^ {j} - q _ {2} ^ {j}\right)}{\left| \boldsymbol {q} _ {1} - \boldsymbol {q} _ {2} \right| ^ {3}} \right], \tag {7.1} \\ \end{array}
$$

where

$$
h _ {i j} ^ {T T \text {   reac }} (t) = - \frac {4}{5} \frac {G}{c ^ {5}} \frac {d ^ {3} Q _ {i j} (t)}{d t ^ {3}}, \tag {7.2}
$$

$Q_{ij}$ denoting the quadrupole moment of the two-body system,

$$
Q _ {i j} (t) = \sum_ {a = 1, 2} m _ {a} \left(q _ {a} ^ {i} q _ {a} ^ {j} - \frac {1}{3} \boldsymbol {q} _ {a} ^ {2} \delta^ {i j}\right). \tag {7.3}
$$

Note that $h_{ij}^{TTreac}$ in Eq. (7.1) should be treated as a given, time-dependent external field, considered as being independent of the canonical variables $q_{a}, p_{a}$ . In other words, when writing the canonical equations of motion $\dot{q} = \partial H_{tot} / \partial p$ , $\dot{p} = -\partial H_{tot} / \partial q$ , one should consider only the explicit q-p dependence appearing in the square brackets on the RHS of Eq. (7.1). After differentiation with respect to q and p one can insert the explicit phase-space expression of the third time derivative of $Q_{ij}(t)$ [obtained, with sufficient precision, by using the Newtonian-level dynamics, i.e. by computing a repeated Poisson bracket of $Q_{ij}(q, p)$ with $H_{\text{Newton}}(q, p)$ ].

Finally, we propose to graft radiation-reaction effects onto the non-perturbatively re-summed conservative dynamics defined by our effective-action approach in the following way. The total Hamiltonian for the relative motion Q, P in ADM coordinates is

$$
H _ {\text { tot }} (Q, P; t) = H _ {\text { real }} ^ {\text { improved }} (Q, P) + H ^ {\text { reac }} (Q, P; t), \tag {7.4}
$$

where the “improved 2PN” Hamiltonian is that defined by solving Eq. (4.25) for $E_{real}=H_{real}^{R}$ , i.e.

$$
\begin{array}{l} \frac {H _ {\mathrm{real}} ^ {\mathrm{improved}} (Q , P)}{M c ^ {2}} \\ = \sqrt {1 + 2 \nu \left(\frac {H _ {\mathrm{eff}} (Q ^ {\prime} (Q , P) , P ^ {\prime} (Q , P))}{\mu c ^ {2}} - 1\right)}, \tag {7.5} \\ \end{array}
$$

on the RHS of which one must transform, by the canonical transformation discussed in Sec. VI, the (exact) effective Hamiltonian defined by Eq. (6.3). In the latter, we propose to use our current best estimates of the effective metric coefficients $A(Q')$ , $B(Q')$ , namely

$$
A (Q ^ {\prime}) \equiv 1 - \frac {2 G M}{c ^ {2} Q ^ {\prime}} + 2 \nu \left(\frac {G M}{c ^ {2} Q ^ {\prime}}\right) ^ {3},
$$

$$
B (Q ^ {\prime}) \equiv A ^ {- 1} (Q ^ {\prime}) \left[ 1 - 6 \nu \left(\frac {G M}{c ^ {2} Q ^ {\prime}}\right) ^ {2} \right]. \tag {7.6}
$$

On the other hand the “reactive” contribution to the total Hamiltonian (7.4) is the center of mass reduction $(\boldsymbol{p}_{1}=-\boldsymbol{p}_{2}=\boldsymbol{P},\boldsymbol{Q}=\boldsymbol{q}_{1}-\boldsymbol{q}_{2})$ of Eq. (7.1).

In terms of reduced variables $(q=Q/GM, p=P/\mu)$ and of the non-relativistic reduced Hamiltonian, $\hat{H}_{\mathrm{real}}^{\mathrm{NR}}\equiv(H_{\mathrm{real}}^{R}-Mc^{2})/\mu$ , our proposal reads

$$
\hat {H} _ {\mathrm{tot}} ^ {\mathrm{NR}} (q, p; t) = \hat {H} _ {\mathrm{real}} ^ {\mathrm{NR} \text { improved }} (q, p) + \hat {H} ^ {\mathrm{reac}} (q, p; t), \tag {7.7}
$$

with

$$
\begin{array}{l} \hat {H} _ {\text { real }} ^ {\text { NR   improved }} (q, p) \\ \equiv \frac {c ^ {2}}{\nu} \left[ \sqrt {1 + 2 \nu \left(\frac {1}{c ^ {2}} \hat {H} _ {\text { eff }} (q ^ {\prime} (q , p) , p ^ {\prime} (q , p)) - 1\right)} - 1 \right], \tag {7.8} \\ \end{array}
$$

where $\hat{H}_{\mathrm{eff}}(q', p')$ is defined by inserting Eq. (7.6) into Eq. (6.5), and with

$$
\hat {H} ^ {\text { reac }} (q, p; t) = - h _ {i j} ^ {T T \text { reac }} (t) \left[ \frac {1}{2} p ^ {i} p ^ {j} - \frac {1}{2} \frac {q ^ {i} q ^ {j}}{q ^ {3}} \right], \tag {7.9}
$$

$$
\begin{array}{l} h _ {i j} ^ {T T \text {   reac }} (t) = - \frac {4}{5 c ^ {5}} \frac {\nu}{q ^ {2}} \left[ - 4 (p ^ {i} n ^ {j} + p ^ {j} n ^ {i}) + 6 n ^ {i} n ^ {j} (\boldsymbol {n} \cdot \boldsymbol {p}) \right. \\ \left. + \frac {2}{3} (\boldsymbol {n} \cdot \boldsymbol {p}) \delta^ {i j} \right], \tag {7.10} \\ \end{array}
$$

where $n^{i} \equiv q^{i}/q$ . As explained above, the quantity $h_{ij}^{TT\text{reac}}(t)$ should not be differentiated with respect to q and p when writing the equations of motion

![](images/8f618292ec61a00b1045af65ca9f879d144b7fa7a64cc5195149eacfe3a619ac.jpg)

<details>
<summary>contour</summary>

| q'_x | q'_y | Method   |
|------|------|----------|
| -20.0| 20.0 | horizon  |
| -10.0| 10.0 | horizon  |
| 0.0  | 0.0  | horizon  |
| 10.0 | -10.0| horizon  |
| 20.0 | -20.0| horizon  |
| -20.0| 15.0 | ISCO     |
| -10.0| 8.0  | ISCO     |
| 0.0  | 4.0  | ISCO     |
| 10.0 | 2.0  | ISCO     |
| 20.0 | 0.0  | ISCO     |
</details>

![](images/1a69b9f42ec2ffb422c95008a173eae9f07ca69030977f93d2ace86113df50c4.jpg)

<details>
<summary>contour</summary>

| q'_x | q'_y | Method   |
|------|------|----------|
| -20.0| 20.0 | horizon  |
| -10.0| 10.0 | horizon  |
| 0.0  | 0.0  | horizon  |
| 10.0 | -10.0| horizon  |
| 20.0 | -20.0| horizon  |
| -20.0| 15.0 | ISCO     |
| -10.0| 12.0 | ISCO     |
| 0.0  | 8.0  | ISCO     |
| 10.0 | 4.0  | ISCO     |
| 20.0 | 0.0  | ISCO     |
</details>

FIG. 5. Inspirating circular orbits in $(q', p')$ coordinates including radiation reaction effects for $\nu=0.1$ (left panel) and $\nu=1/4$ (right panel). The location of the ISCO and of the horizon are indicated.

$$
\dot {q} ^ {i} = \frac {\partial \hat {H} _ {\text { real }} ^ {\text { NR   improved }} (q , p)}{\partial p _ {i}} + \frac {\partial \hat {H} ^ {\text { reac }} (q , p ; h _ {i j} ^ {T T \text { reac }} (t))}{\partial p _ {i}},
$$

$$
\dot {p} _ {i} = - \frac {\partial \hat {H} _ {\text {real}} ^ {\mathrm{NR} \text {improved}} (q , p)}{\partial q ^ {i}} - \frac {\partial \hat {H} ^ {\text {reac}} (q , p ; h _ {i j} ^ {T T \text {reac}} (t))}{\partial q ^ {i}}. \tag {7.11}
$$

When inserting, after differentiation, Eq. (7.10), the equations of motion (7.11) become an explicit, autonomous (time-independent) evolution equation in phase space: $\dot{\boldsymbol{x}} = f(\boldsymbol{x})$ where $\boldsymbol{x} = (q^{i}, p_{i})$ . From the study in Sec. V above of the circular orbits defined by the exact, non-perturbative Hamiltonian $H_{eff}$ , we expect that the combined dynamics (7.11) will exhibit a transition from inspiral to plunge when $q = |q|$ (which decreases under radiation damping) reaches the image in the q-p phase space of the ISCO, studied above in $q', p'$ coordinates. We have in mind here quasi-circular, inspiraling orbits (circularized by radiation reaction), though, evidently, our approach can be used to study all possible orbits. We further expect that, when $\nu \ll 1$ , the inspiral will be very slow [the reaction Hamiltonian being proportional to $\nu$ ; see Eq. (7.10)] and therefore the transition to plunge will be quite sharp and well located at the ISCO. When $\nu = 1/4$ the radiation reaction effects are numerically smallish, but not parametrically small at the ISCO, and the transition to plunge cannot be expected to be very sharp. These expected behaviors are illustrated in Fig. 5.

For simplicity, we have computed the orbits exhibited in these figures in $q^{\prime}$ space, neglecting the (formally 3.5PN) effect of the $(q,p)\rightarrow(q^{\prime},p^{\prime})$ transformation on the reactive part of the equations of motion. [Thanks to the canonical invariance of the Hamilton equations of motion, the crucial conservative part of the evolution in $q^{\prime},p^{\prime}$ space is simply obtained from the Hamiltonian $\hat{H}_{\mathrm{real}}^{\mathrm{NR}}\mathrm{improved}(q^{\prime},p^{\prime})$ defined by keeping the variables $q^{\prime}$ and $p^{\prime}$ on the RHS of Eq. (7.8).]

Let us finally mention another possibility for incorporating radiation reaction effects directly in the effective one-body dynamics. In the q-p coordinates the (2.5PN) reaction Hamiltonian (7.1) can be simply seen as due to perturbing the Euclidean metric $g_{ij}^{0}=\delta_{ij}$ appearing in the lowest order Newtonian Hamiltonian ( $q_{ab}^{i}\equiv q_{a}^{i}-q_{b}^{i}$ )

$$
H _ {\text { Newtonian }} \left(q _ {a}, p _ {a}\right) = \sum_ {a} \frac {g _ {0} ^ {i j} p _ {a i} p _ {a j}}{2 m _ {a}} - \sum_ {a <   b} \frac {G m _ {a} m _ {b}}{\left(g _ {i j} ^ {0} q _ {a b} ^ {i} q _ {a b} ^ {j}\right) ^ {1 / 2}}, \tag {7.12}
$$

by taking into account the near zone radiative field:

$$
g _ {i j} \simeq g _ {i j} ^ {0} + h _ {i j} ^ {T T \text {   reac }} (t), \quad g ^ {i j} \simeq g _ {0} ^ {i j} - h _ {\text {   reac }} ^ {i j T T} (t). \tag {7.13}
$$

By mapping back [through our $(qp)\leftrightarrow(q^{\prime}p^{\prime})$ link] the metric perturbation $h_{ij}^{TTreac}$ onto the effective problem, one might try to incorporate reaction effects by defining a suitable “reactive” perturbation of our effective metric:

$$
g _ {\mu \nu} (q ^ {\prime}) = g _ {\mu \nu} ^ {\mathrm{eff}} (q ^ {\prime}) + \delta^ {\mathrm{reac}} g _ {\mu \nu} ^ {\mathrm{eff}} (q ^ {\prime}). \tag {7.14}
$$

This approach might be useful for trying to go beyond the 2.5PN level discussed here and to define a “re-summed” version of reaction effects. Alternatively, if one has at one’s disposal a more complete PN-expanded reactive force expressed in the original q coordinates [32], one can, following the strategy proposed in Eq. (7.4), graft this improved (perturbative) reactive force onto the non-perturbatively improved conservative force defined by mapping back our effective dynamics onto the q coordinates.

# VIII. CONCLUSIONS

We have introduced a novel approach to studying the late dynamical evolution of a coalescing binary system of compact objects. This approach is based on mapping (by a canonical transformation) the dynamics of the relative motion of a two-body system, with comparable masses $m_{1}, m_{2}$ , onto the dynamics of one particle of mass $\mu = m_{1}m_{2}/(m_{1} + m_{2})$ moving in some effective metric $ds_{eff}$ . When neglecting radiation reaction, the mapping rules between the two problems are best interpreted in quantum terms (mapping between the discrete energy spectrum of bound states). They involve a physically natural transformation of the energy

axis between the two problems, stating essentially that the effective energy of the effective particle is the energy of particle 1 in the rest frame of particle 2 (or reciprocally); see Eq. (4.26). The usefulness of this energy mapping was previously emphasized both in quantum two-body problems [1] and in classical ones [21].

Starting from the currently most accurate knowledge of two-body dynamics [6,7], we have shown that, when neglecting radiation reaction, our rules uniquely determine the effective metric $g_{\mu\nu}^{\mathrm{eff}}(q')$ in which the effective particle moves. This metric is a simple deformation of a Schwarzschild metric of mass $M=m_{1}+m_{2}$ , with deformation parameter $\nu=\mu/M$ . Our suggestion is then to define (as is done in quantum two-body problems [1,3]) a particular nonperturbative re-summation of the usual, badly convergent, post-Newtonian-expanded dynamics by considering the dynamics defined by the effective metric as exact. This definition leads, in particular, to specific predictions for the characteristics of the innermost stable circular orbit for comparable-mass systems. In agreement with some previous predictions (notably one based on Padé approximants [21]), but in disagreement with the predictions of the “hybrid” approach of Ref. [22], we predict an ISCO which is more tightly bound than the usual test-mass-in-Schwarzschild one. The invariant physical characteristics of our predicted ISCO are given in Eqs. (5.31) and (5.32); see also Table I. Note in particular that the binding energy at the ISCO is robustly predicted to be $E_{real}^{NR}\simeq-1.5\%Mc^{2}$ (for equal-mass systems, $\nu=1/4$ ), while the orbital frequency at the ISCO is numerically predicted to be (again for $\nu=1/4$ )

$$
f ^ {\mathrm{ISCO}} = 2 3 7 2 \mathrm{Hz} \left(\frac {M _ {\odot}}{M}\right). \tag {8.1}
$$

Note that this corresponds to $\sim847$ Hz for $(1.4M_{\odot},1.4M_{\odot})$ neutron star systems.

We have argued, by studying the effects of higher (time-symmetric) post-Newtonian contributions, that our predictions for the characteristics of the ISCO are rather robust (especially when compared to the scatter of previous predictions). See Fig. 4 and Table I. We note, however, that knowledge of the 3PN dynamics (currently in progress [19,33]) would significantly reduce the present (2PN-based) uncertainty on the knowledge of the effective metric.

The coordinate separation, in effective Schwarzschild coordinates, corresponding to the ISCO is $Q^{\prime}=R\simeq5.72GM/c^{2}$ , i.e. $\sim23.6km$ for a $(1.4M_{\odot},1.4M_{\odot})$ neutron star system [from our canonical transformation (6.8), this corresponds to an ADM-coordinate relative separation of $Q\simeq4.79GM/c^{2}$ ]. This value is near the sum of the nominal radii of (isolated) neutron stars for most nuclear equations of state [34]. This suggests that the inspiral phase of coalescing neutron star systems might terminate into tidal disruption (or at least tidally dominated dynamics) without going through a well-defined plunge phase. Fully relativistic 3D numerical simulations are needed to investigate this question. We note that a positive aspect of having (as predicted here) a rather low ISCO is that the end of the inspiral phase might well be very sensitive to the nuclear equation of state, so that LIGO and VIRGO observations might teach us something new about dense nuclear matter.

Finally, we have proposed two ways of adding radiation reaction effects to our effective one-body dynamics. The most straightforward one consists in directly combining radiation effects determined in the real two-body problem with the non-perturbative conservative dynamics (which, in particular, features a dynamical instability at our ISCO) obtained by mapping the effective dynamics onto some standard (ADM or harmonic) two-body coordinate system: see Eq. (7.7). A more subtle approach, which needs to be further developed, would consist in adding radiation reaction effects at the level of the effective metric itself; see Eq. (7.14). We have illustrated in Fig. 5 the transition from inspiral to plunge implied by (an approximation to) Eq. (7.7). In principle, this transition, and in particular the frequency at the ISCO, will be observable in gravitational wave observations of systems containing black holes.

We hope that the approach presented here will also be of value for supplementing numerical relativity investigations. Indeed, our main (hopeful) claim is that the effective one-body dynamics is a “good” non-perturbative re-summation of the standard post-Newtonian-expanded results. Therefore, it gives a simple way of boosting up the accuracy of many PN-expanded results. (We leave to future work a more systematic analysis of the extension of our approach to higher post-Newtonian orders.) Effectively, this extends the validity of the post-Newtonian expansions in a new way (e.g. different from Padé approximants $^{7}$ ). In particular, our results could be used to define initial conditions for two-body systems very near, or even at, the ISCO, thereby cutting down significantly the numerical work needed to evolve fully relativistic 3D binary-system simulations.

As a final remark, let us note that many extensions of the approach presented here are possible. In particular, the addition of the (classical) spin degrees of freedom to the effective one-body problem (in the effective metric and/or in the effective particle) suggests itself as an interesting issue (with possibly important physical consequences).

# ACKNOWLEDGMENTS

We thank Gerhard Schäfer for useful comments.

# APPENDIX A

In this appendix we determine, at the 2PN level and in the Schwarzschild gauge, the effective metric

$$
d s _ {\text { eff }} ^ {2} = - A (R) c ^ {2} d t ^ {2} + B (R) d R ^ {2} + R ^ {2} \left(d \theta^ {2} + \sin^ {2} \theta d \varphi^ {2}\right), \tag {A1}
$$

$$
A (R) = 1 + \frac {a _ {1}}{c ^ {2} R} + \frac {a _ {2}}{c ^ {4} R ^ {2}} + \frac {a _ {3}}{c ^ {6} R ^ {3}}, \quad B (R) = 1 + \frac {b _ {1}}{c ^ {2} R} + \frac {b _ {2}}{c ^ {4} R ^ {2}}, \tag {A2}
$$

when requiring simultaneously that (a) the energy levels of the “effective” and “real” problems coincide modulo an overall shift, i.e. $\mathcal{E}_{0}(\mathcal{N}_{0},\mathcal{J}_{0})=\mathcal{E}_{\mathrm{real}}(\mathcal{N},\mathcal{J})-c_{0}$ , with $c_{0}=Mc^{2}-m_{0}c^{2}$ , $J_{0}=J$ and $N_{0}=N$ and (b) the effective metric depend only on $m_{1}$ and $m_{2}$ . In this case, as anticipated in Sec. IV, we will see that it not possible to satisfy the condition $m_{0}=\mu$ .

The radial action $I_{R}^{0}(\mathcal{E}_{0},\mathcal{J}_{0})$ of the “effective” description is

$$
I _ {R} ^ {0} \left(\mathcal {E} _ {0}, \mathcal {J} _ {0}\right) = \frac {\alpha_ {0} m _ {0} ^ {1 / 2}}{\sqrt {- 2 \mathcal {E} _ {0} ^ {\mathrm{NR}}}} \left[ \hat {A} + \hat {B} \frac {\mathcal {E} _ {0} ^ {\mathrm{NR}}}{m _ {0} c ^ {2}} + \hat {C} \left(\frac {\mathcal {E} _ {0} ^ {\mathrm{NR}}}{m _ {0} c ^ {2}}\right) ^ {2} \right] - \mathcal {J} _ {0}
$$

$$
+ \frac {\alpha_ {0} ^ {2}}{c ^ {2} \mathcal {J} _ {0}} \left[ \hat {D} + \hat {E} \frac {\mathcal {E} _ {0} ^ {\mathrm{NR}}}{m _ {0} c ^ {2}} \right] + \frac {\alpha_ {0} ^ {4}}{c ^ {4} \mathcal {J} _ {0} ^ {3}} \hat {F}, \tag {A3}
$$

where $\mathcal{E}_0^{\mathrm{NR}}\equiv \mathcal{E}_0 - m_0c^2$ $\alpha_0\equiv GM_0m_0$

$$
\hat {A} = - \frac {1}{2} \hat {a} _ {1}, \quad \hat {B} = \hat {b} _ {1} - \frac {7}{8} \hat {a} _ {1}, \quad \hat {C} = \frac {\hat {b} _ {1}}{4} - \frac {1 9}{6 4} \hat {a} _ {1}, \tag {A4}
$$

$$
\hat {D} = \frac {\hat {a} _ {1} ^ {2}}{2} - \frac {\hat {a} _ {2}}{2} - \frac {\hat {a} _ {1} \hat {b} _ {1}}{4}, \quad \hat {E} = \hat {a} _ {1} ^ {2} - \hat {a} _ {2} - \frac {\hat {a} _ {1} \hat {b} _ {1}}{2} - \frac {\hat {b} _ {1} ^ {2}}{8} + \frac {\hat {b} _ {2}}{2},
$$

$$
\hat {F} = \frac {1}{6 4} \left[ 2 4 \hat {a} _ {1} ^ {4} - 4 8 \hat {a} _ {1} ^ {2} \hat {a} _ {2} + 8 \hat {a} _ {2} ^ {2} + 1 6 \hat {a} _ {1} \hat {a} _ {3} - 8 \hat {a} _ {1} ^ {3} \hat {b} _ {1} + 8 \hat {a} _ {1} \hat {a} _ {2} \hat {b} _ {1} \right.
$$

$$
\left. - \hat {a} _ {1} ^ {2} \hat {b} _ {1} ^ {2} + 4 \hat {a} _ {1} ^ {2} \hat {b} _ {2} \right],
$$

and we have introduced the dimensionless coefficients

$$
\hat {a} _ {i} = \frac {a _ {i}}{(G M _ {0}) ^ {i}}, \quad \hat {b} _ {i} = \frac {b _ {i}}{(G M _ {0}) ^ {i}}. \tag {A5}
$$

We define the mass $M_{0}$ used to scale the coefficients $a_{i}$ and $b_{i}$ by requiring $\hat{a}_{1} \equiv -2$ (i.e. $a_{1} \equiv -2GM_{0}$ ). Identifying Eq. (A3) with the radial action $I_{R}^{0}(\mathcal{E}^{\mathrm{NR}}, \mathcal{J})$ of the “real” problem, i.e.

$$
I _ {R} \left(\mathcal {E} ^ {\mathrm{NR}}, \mathcal {J}\right) = \frac {\alpha \mu^ {1 / 2}}{\sqrt {- 2 \mathcal {E} ^ {\mathrm{NR}}}} \left[ 1 + \left(\frac {1 5}{4} - \frac {\nu}{4}\right) \frac {\mathcal {E} ^ {\mathrm{NR}}}{\mu c ^ {2}} + \left(\frac {3 5}{3 2} + \frac {1 5}{1 6} \nu \right. \right.
$$

$$
\left. + \frac {3}{3 2} \nu^ {2}\right) \left(\frac {\mathcal {E} ^ {\mathrm{NR}}}{\mu c ^ {2}}\right) ^ {2} \biggr ]
$$

$$
- \mathcal {J} + \frac {\alpha^ {2}}{c ^ {2} \mathcal {J}} \left[ 3 + \left(\frac {1 5}{2} - 3 \nu\right) \frac {\mathcal {E} ^ {\mathrm{NR}}}{\mu c ^ {2}} \right]
$$

$$
+ \left(\frac {3 5}{4} - \frac {5}{2} \nu\right) \frac {\alpha^ {4}}{c ^ {4} \mathcal {J} ^ {3}}, \tag {A6}
$$

where $\alpha\equiv GM\mu$ and $E^{NR}\equiv E_{real}-Mc^{2}$ , yields six equations to be satisfied. The requirement (a) above implies the simple identification of the variables entering Eqs. (A3) and (A6):

$\mathcal{E}_{0}^{\mathrm{NR}}=\mathcal{E}^{\mathrm{NR}},\;\mathcal{J}_{0}=\mathcal{J},\;I_{R}^{0}=I_{R}$ . The explicit form of the equations stating that $\hat{A}m_{0}^{1/2}\alpha_{0}$ (0PN level), $\hat{B}m_{0}^{-1/2}\alpha_{0}$ , $\hat{D}\alpha_{0}^{2}$ (1PN level) and $\hat{C}m_{0}^{-3/2}\alpha_{0}$ , $\hat{E}\alpha_{0}^{2}/m_{0}$ and $\hat{F}\alpha_{0}^{4}$ (2PN level) in Eq. (A3) coincide with the analogous coefficients in Eq. (A6) yields

$$
m _ {0} ^ {1 / 2} \alpha_ {0} = \mu^ {1 / 2} \alpha , \tag {A7}
$$

$$
\left(\hat {b} _ {1} + \frac {7}{4}\right) m _ {0} ^ {- 1 / 2} \alpha_ {0} = \frac {1}{4} (1 5 - \nu) \mu^ {- 1 / 2} \alpha , \tag {A8}
$$

$$
(4 - \hat {a} _ {2} + \hat {b} _ {1}) \alpha_ {0} ^ {2} = 6 \alpha^ {2}, \tag {A9}
$$

$$
\left(\frac {1 9}{3 2} + \frac {\hat {b} _ {1}}{4}\right) m _ {0} ^ {- 3 / 2} \alpha_ {0} = \left(\frac {3 5}{3 2} + \frac {1 5}{1 6} \nu + \frac {3}{3 2} \nu^ {2}\right) \mu^ {- 3 / 2} \alpha , \tag {A10}
$$

$$
\left(4 - \hat {a} _ {2} + \hat {b} _ {1} - \frac {\hat {b} _ {1} ^ {2}}{8} + \frac {\hat {b} _ {2}}{2}\right) \frac {\alpha_ {0} ^ {2}}{m _ {0}} = \left(\frac {1 5}{2} - 3 \nu\right) \frac {\alpha^ {2}}{\mu}, \tag {A11}
$$

$$
\hat {F} \alpha_ {0} ^ {4} = \left(\frac {3 5}{4} - \frac {5}{2} \nu\right) \alpha^ {4}. \tag {A12}
$$

It is to be noted that if we impose $m_{0}=\mu$ and $GM_{0}=GM$ (so that $\alpha_{0}=\alpha$ ), we get an incompatibility at the 2PN level. Indeed, Eq. (A7) is satisfied and we can solve Eqs. (A8),(A9) in terms of the 1PN coefficients $\hat{b}_{1}$ and $\hat{a}_{2}$ , but then the 2PN equation (A10), which contains only $\hat{b}_{1}$ , is not satisfied. (This problem is due to the fact that we have more equations than unknowns.) Hence, we are obliged to relax the constraint $m_{0}=\mu$ . Let us introduce the parameter $\xi$ , defined by $m_{0}\equiv\mu\xi^{-2}$ . Equation (A7) then gives $GM_{0}=GM\xi^{3}$ . Note that we are crucially using here the fact that the Newton-order energy levels $\mathcal{E}^{\mathrm{NR}}=-m_{0}\alpha_{0}^{2}/(2\mathcal{N}_{0})+\mathcal{O}(c^{-2})$ do not depend separately on $m_{0}$ and $\alpha_{0}=GM_{0}m_{0}$ , but only on the combination $m_{0}\alpha_{0}^{2}=G^{2}M_{0}^{2}m_{0}^{3}$ . Solving the 1PN-level equations (A8),(A9) we then get

$$
\hat {b} _ {1} = \frac {1}{4 \xi^ {2}} (1 5 - 7 \xi^ {2} - \nu), \quad \hat {a} _ {2} = \frac {1}{4 \xi^ {2}} (- 9 + 9 \xi^ {2} - \nu), \tag {A13}
$$

while the 2PN-level equation (A10) gives a quadratic equation in $\xi^2$ which fixes uniquely its value (as well as that of the positive parameter $\xi$ ), namely

$$
\xi^ {2} = \frac {\mu}{m _ {0}} = \frac {1}{5} [ - 1 5 + \nu + 2 \sqrt {2} \sqrt {5 0 + 1 5 \nu + 2 \nu^ {2}} ]. \tag {A14}
$$

Finally, the remaining 2PN equations (A11) and (A12) determine the coefficients of the effective metric at the 2PN level:

$$
\hat {b} _ {2} = \frac {1}{6 4 \xi^ {2}} (1 1 8 5 - 9 7 8 \xi^ {2} + 4 9 \xi^ {4} - 4 1 4 \nu + 1 4 \xi^ {2} \nu + \nu^ {2}), \tag {A15}
$$

$$
\hat {a} _ {3} = \frac {1}{6 4 \xi^ {4}} (- 2 8 9 + 4 0 2 \xi^ {2} - 1 1 3 \xi^ {4} + 1 5 8 \nu + 5 0 \xi^ {2} \nu - \nu^ {2}). \tag {A16}
$$

The complexity of the results (A13)-(A16), compared to the simplicity of our preferred solution (5.6)-(5.8), convinced us that the requirement (a) above should be relaxed. Also, it seems suspicious to have an effective mass $m_0$ which differs from $\mu$ even in the non-relativistic limit $c \to \infty$ . Finally, it is not evident that this method can be generalized to higher post-Newtonian orders (where more redundant equations will have to be satisfied).

# APPENDIX B

In this appendix we describe an alternative, more formal method to map the “effective” one-body problem onto the “real” two-body one. We work in the Schwarzschild gauge. Here we require simultaneously that (a) the energy levels of the “effective” and “real” descriptions coincide modulo an overall shift, i.e. $\mathcal{E}_{0}(\mathcal{N}_{0},\mathcal{J}_{0})=\mathcal{E}_{\mathrm{real}}(\mathcal{N},\mathcal{J})-c_{0}$ , with $c_{0}=Mc^{2}-m_{0}c^{2}$ , $J_{0}=J$ and $N_{0}=N$ and (b) the effective mass $m_{0}$ be equal to the reduced mass $\mu=m_{1}m_{2}/(m_{1}+m_{2})$ . Introducing the dimensionless quantities

$$
\hat {I} _ {R} ^ {0} \equiv \frac {I _ {R} ^ {0}}{\alpha_ {0}}, \quad \hat {I} _ {R} ^ {\text { real }} \equiv \frac {I _ {R} ^ {\text { real }}}{\alpha}, \quad E _ {0} \equiv \frac {\mathcal {E} _ {0} ^ {\mathrm{NR}}}{m _ {0}}, \quad E _ {\text { real }} \equiv \frac {\mathcal {E} _ {\text { real }} ^ {\mathrm{NR}}}{\mu}, \tag {B1}
$$

$$
j _ {0} \equiv \frac {\mathcal {J} _ {0}}{\alpha_ {0}}, \quad j \equiv \frac {\mathcal {J}}{\alpha},
$$

where $\alpha_{0}\equiv GM_{0}m_{0}$ and $\alpha\equiv GM\mu\equiv Gm_{1}m_{2}$ , we can rewrite the radial action for the “effective” problem, Eq. (3.13), in the form

$$
\begin{array}{l} \hat {I} _ {R} ^ {0} \left(E _ {0}, j _ {0}\right) = \frac {1}{\sqrt {- 2 E _ {0}}} \left[ \hat {A} + \hat {B} \frac {E _ {0}}{c ^ {2}} + \hat {C} \left(\frac {E _ {0}}{c ^ {2}}\right) ^ {2} \right] - j _ {0} \\ + \frac {1}{c ^ {2} j _ {0}} \left[ \hat {D} + \hat {E} \frac {E _ {0}}{c ^ {2}} \right] + \frac {1}{c ^ {4} j _ {0} ^ {3}} \hat {F}, \tag {B2} \\ \end{array}
$$

where

$$
\hat {A} = - \frac {1}{2} \hat {a} _ {1}, \quad \hat {B} = \hat {b} _ {1} - \frac {7}{8} \hat {a} _ {1}, \quad \hat {C} = \frac {\hat {b} _ {1}}{4} - \frac {1 9}{6 4} \hat {a} _ {1},
$$

$$
\hat {D} = \frac {\hat {a} _ {1} ^ {2}}{2} - \frac {\hat {a} _ {2}}{2} - \frac {\hat {a} _ {1} \hat {b} _ {1}}{4}, \quad \hat {E} = \hat {a} _ {1} ^ {2} - \hat {a} _ {2} - \frac {\hat {a} _ {1} \hat {b} _ {1}}{2} - \frac {\hat {b} _ {1} ^ {2}}{8} + \frac {\hat {b} _ {2}}{2},
$$

$$
\begin{array}{l} \hat {F} = \frac {1}{6 4} \left[ 2 4 \hat {a} _ {1} ^ {4} - 4 8 \hat {a} _ {1} ^ {2} \hat {a} _ {2} + 8 \hat {a} _ {2} ^ {2} + 1 6 \hat {a} _ {1} \hat {a} _ {3} - 8 \hat {a} _ {1} ^ {3} \hat {b} _ {1} \right. \\ \left. + 8 \hat {a} _ {1} \hat {a} _ {2} \hat {b} _ {1} - \hat {a} _ {1} ^ {2} \hat {b} _ {1} ^ {2} + 4 \hat {a} _ {1} ^ {2} \hat {b} _ {2} \right], \tag {B3} \\ \end{array}
$$

and where we have used, as above, the scaled metric coefficients

$$
\hat {a} _ {i} = \frac {a _ {i}}{(G M _ {0}) ^ {i}}, \quad \hat {b} _ {i} = \frac {b _ {i}}{(G M _ {0}) ^ {i}}. \tag {B4}
$$

Identifying $\hat{I}_{R}^{0}(E_{0},j_{0})$ with the analogous expression for the “real” problem,

$$
\begin{array}{l} \hat {I} _ {R} \left(E _ {\text {real}}, j\right) = \frac {1}{\sqrt {- 2 E _ {\text {real}}}} \left[ 1 + \left(\frac {1 5}{4} - \frac {\nu}{4}\right) \frac {E _ {\text {real}}}{c ^ {2}} + \left(\frac {3 5}{3 2} + \frac {1 5}{1 6} \nu \right. \right. \\ \left. + \frac {3}{3 2} \nu^ {2}\right) \left(\frac {E _ {\text { real }}}{c ^ {2}}\right) ^ {2} \bigg ] \\ - j + \frac {1}{c ^ {2} j} \left[ 3 + \left(\frac {1 5}{2} - 3 \nu\right) \frac {E _ {\text { real }}}{c ^ {2}} \right] \\ + \left(\frac {3 5}{4} - \frac {5}{2} \nu\right) \frac {1}{c ^ {4} j ^ {3}}, \tag {B5} \\ \end{array}
$$

and imposing $E_{0}=E_{real}$ , $m_{0}=\mu$ , $\alpha_{0}=\alpha$ , we get more equations to be satisfied than unknowns,

$$
- \frac {1}{2} \hat {a} _ {1} = 1, \tag {B6}
$$

$$
\hat {b} _ {1} - \frac {7}{8} \hat {a} _ {1} = \frac {1}{4} (1 5 - \nu), \tag {B7}
$$

$$
\hat {a} _ {1} ^ {2} - \hat {a} _ {2} - \frac {\hat {a} _ {1} \hat {b} _ {1}}{2} = 6, \tag {B8}
$$

$$
- \frac {1 9}{6 4} \hat {a} _ {1} + \frac {\hat {b} _ {1}}{4} = \frac {3 5}{3 2} + \frac {1 5}{1 6} \nu + \frac {3}{3 2} \nu^ {2}, \tag {B9}
$$

$$
\hat {a} _ {1} ^ {2} - \hat {a} _ {2} - \frac {\hat {a} _ {1} \hat {b} _ {1}}{2} - \frac {\hat {b} _ {1} ^ {2}}{8} + \frac {\hat {b} _ {2}}{2} = \frac {1 5}{2} - 3 \nu , \tag {B10}
$$

$$
\hat {F} = \frac {3 5}{4} - \frac {5}{2} \nu . \tag {B11}
$$

Note that Eqs. (B7) and (B9) depend only on $\hat{a}_{1}$ and $\hat{b}_{1}$ , and cannot both be satisfied. To solve this incompatibility we consider here the possibility that the various coefficients that appear in the effective metric depend on the energy. Namely, at the 2PN level we consider the following expansions:

$$
\hat {a} _ {1} (E _ {0}) = \hat {a} _ {1} ^ {(0)} + \hat {a} _ {1} ^ {(2)} \left(\frac {E _ {0}}{c ^ {2}}\right) + \hat {a} _ {1} ^ {(4)} \left(\frac {E _ {0}}{c ^ {2}}\right) ^ {2}, \tag {B12}
$$

$$
\hat {a} _ {2} (E _ {0}) = \hat {a} _ {2} ^ {(0)} + \hat {a} _ {2} ^ {(2)} \left(\frac {E _ {0}}{c ^ {2}}\right), \tag {B13}
$$

$$
\hat {a} _ {3} (E _ {0}) = \hat {a} _ {3} ^ {(0)}, \tag {B14}
$$

and

$$
\hat {b} _ {1} (E _ {0}) = \hat {b} _ {1} ^ {(0)} + \hat {b} _ {1} ^ {(2)} \left(\frac {E _ {0}}{c ^ {2}}\right), \quad \hat {b} _ {2} (E _ {0}) = \hat {b} _ {2} ^ {(0)}. \tag {B15}
$$

The introduction of an energy dependence in the coefficients $\hat{a}_{i},\hat{b}_{i}$ reshuffles the $c^{-2}$ expansion of Eq. (B2) and modifies Eqs. (B6)–(B11) which are to be satisfied. It is easy to see that the flexibility introduced by the new coefficients $\hat{a}_{i}^{(2n)}$ , $\hat{b}_{i}^{(2n)}$ allows one to solve in many ways the constraints to be satisfied. The simplest solution is obtained by requiring that the energy dependence enter only in $\hat{a}_{1}(E_{0})$ and only at the 2PN level,

$$
\hat {a} _ {1} ^ {(2)} = 0, \quad \hat {a} _ {2} ^ {(2)} = 0, \quad \hat {b} _ {1} ^ {(2)} = 0, \tag {B16}
$$

because in this case only Eq. (B9) gets modified. Indeed, it is straightforward to derive the new equation replacing (B9):

$$
- \frac {1 9}{6 4} \hat {a} _ {1} ^ {(0)} + \frac {\hat {b} _ {1} ^ {(0)}}{4} - \frac {\hat {a} _ {1} ^ {(4)}}{2} = \frac {3 5}{3 2} + \frac {1 5}{1 6} \nu + \frac {3}{3 2} \nu^ {2}. \tag {B17}
$$

Hence, from Eqs. (B6)-(B8) we obtain the effective metric coefficients at the 1PN level:

$$
\hat {a} _ {1} ^ {(0)} = - 2, \quad \hat {a} _ {2} ^ {(0)} = - \frac {\nu}{4}, \quad \hat {b} _ {1} ^ {(0)} = \frac {1}{4} (8 - \nu), \tag {B18}
$$

while the 2PN equations (B17) and (B11),(B12) give

$$
\hat {a} _ {1} ^ {(4)} = - \frac {\nu}{1 6} (3 2 + 3 \nu), \quad \hat {a} _ {3} ^ {(0)} = \frac {\nu}{6 4} (2 0 8 - \nu),
$$

$$
\hat {b} _ {2} ^ {(0)} = \frac {1}{6 4} (2 5 6 - 4 0 0 \nu + \nu^ {2}). \tag {B19}
$$

Again this solution is more complex than our preferred solution (5.6)-(5.8). Moreover, we think that the assumption of an energy dependence in the effective metric introduces a conceptual obscurity in the entire approach: Indeed, one should introduce two separate (effective) energies: the energy parameter $E_0^{(0)}$ appearing explicitly in $g_{\mu \nu}^{\mathrm{eff}}$ and the conserved energy $E_0^{(1)}$ of some individual geodesic motion in $g_{\mu \nu}^{\mathrm{eff}}(E_0^{(0)})$ . They can only be identified, a posteriori, for each specified geodesic motion. This makes it also quite difficult to incorporate radiation reaction effects.

Finally, one can require that the effective metric does not depend on the energy, but that the effective mass $m_{0}$ depends on $E_{0}$ . One then finds the solution

$$
m _ {0} (E _ {0}) = \mu \left[ 1 + \frac {\nu}{4 8} (3 2 + 3 \nu) \left(\frac {E _ {0}}{c ^ {2}}\right) ^ {2} \right], \tag {B20}
$$

with a corresponding effective metric defined by the energy-independent part $\hat{a}_{i}^{(0)}$ , $\hat{b}_{i}^{(0)}$ of the solution above. The objections of complexity and conceptual obscurity raised above also apply to this energy-dependent effective-mass solution.

[1] E. Brézin, C. Itzykson and J. Zinn-Justin, Phys. Rev. D 1, 2349 (1970).   
[2] C. Itzykson and J. B. Zuber, Quantum Field Theory (McGraw-Hill, 1980), p. 83.   
[3] I. T. Todorov, Phys. Rev. D 3, 2351 (1971); V. A. Rizov, I. T. Todorov and B. L. Aneva, Nucl. Phys. B98, 447 (1975); I. T. Todorov, in Properties of Fundamental Interactions, edited by A. Zichichi (Editrice Compositori, Bologna, 1973), Vol. 9C, pp. 951–979.   
[4] A. Maheswari, E. R. Nissimov and I. T. Todorov, Lett. Math. Phys. 5, 359 (1981).   
[5] While both the “relativistic reduced mass” $m_{w} \equiv m_{1} m_{2}/w$ and the “energy of the effective particle” $E_{w} \equiv \sqrt{m_{w}^{2} + b^{2}(w^{2})}$ , introduced in Ref. [3], are rather complicated functions of the total energy $w = \sqrt{s}$ , their ratio $E_{w}/m_{w}$ simplifies to the function $\epsilon \equiv (s - m_{1}^{2} - m_{2}^{2})/(2 m_{1} m_{2})$ , implicit in Ref. [1], which we use in our approach.   
[6] T. Damour and N. Deruelle, Phys. Lett. 87A, 81 (1981).   
[7] T. Damour, C. R. Seances Acad. Sci., Ser. 2 294, 1355 (1982).   
[8] T. Damour and N. Deruelle, C. R. Seances Acad. Sci., Ser. 2293, 537 (1981).   
[9] T. Damour and N. Deruelle, C. R. Seances Acad. Sci., Ser. 2293, 877 (1981).   
[10] T. Damour and G. Schäfer, Gen. Relativ. Gravit. 17, 879 (1985).   
[11] T. Damour and G. Schäfer, Nuovo Cimento 10, 123 (1988).   
[12] R. Arnowitt, S. Deser and C. W. Misner, Phys. Rev. 120, 313 (1960).

[13] T. Ohta, H. Okamura, T. Kimura and K. Hiida, Prog. Theor. Phys. 51, 1220 (1974).   
[14] G. Schäfer, Gen. Relativ. Gravit. 18, 255 (1986).   
[15] T. Damour, in 300 Years of Gravitation, edited by S. W. Hawking and W. Israel (Cambridge University Press, Cambridge, England, 1987), pp. 128–198.   
[16] T. Ohta and T. Kimura, Prog. Theor. Phys. 81, 679 (1989); 81, 662 (1989).   
[17] G. Schäfer and N. Wex, Phys. Lett. A 174, 196 (1993); 177, 461(E) (1993).   
[18] G. Schäfer, in Symposia Gaussiana, Proceedings of the 2nd Gauss Symposium, Conference A: Mathematics and Theoretical Physics, edited by M. Behara, R. Fritsch and R. Lintz (Walter de Gruyter, Berlin, 1995), p. 667.   
[19] P. Jaranowski and G. Schäfer, Phys. Rev. D 57, 5948 (1998); 57, 7274 (1998).   
[20] G. Schäfer, Ann. Phys. (N.Y.) 161, 81 (1985).   
[21] T. Damour, B. R. Iyer and B. S. Sathyaprakash, Phys. Rev. D 57, 885 (1998).   
[22] L. E. Kidder, C. M. Will and A. G. Wiseman, Class. Quantum Grav. 9, L127 (1992); Phys. Rev. D 47, 3281 (1993).   
[23] N. Wex and G. Schäfer, Class. Quantum Grav. 10, 2729 (1993).   
[24] G. Schäfer and N. Wex, in XIIIth Moriond Workshop: Perspectives in Neutrinos, Atomic Physics and Gravitation, edited by J. Trân Thanh Vân, T. Damour, E. Hinds and J. Wilkerson (Editions Frontières, Gif-sur-Yvette, 1993), p. 513.

[25] J. C. Lombardi, F. A. Rasio and S. L. Shapiro, Phys. Rev. D 56, 3416 (1997).   
[26] J. R. Wilson and G. J. Mathews, Phys. Rev. Lett. 75, 4161 (1995); J. R. Wilson, G. J. Mathews and P. Marronetti, Phys. Rev. D 54, 1317 (1996).   
[27] T. W. Baumgarte, G. B. Cook, M. A. Scheel, S. L. Shapiro and S. A. Teukolsky, Phys. Rev. D 57, 7299 (1998).   
[28] C. W. Misner, K. Thorne and J. A. Wheeler, Gravitation (Freeman, New York, 1973).   
[29] L. D. Landau and E. M. Lifshitz, The Classical Theory of Fields (Pergamon, Oxford, 1962).   
[30] J. P. A. Clark and D. M. Eardley, Astrophys. J. 215, 311 (1977).

[31] J. K. Blackburn and S. Detweiler, Phys. Rev. D 46, 2318 (1992).   
[32] B. R. Iyer and C. Will, Phys. Rev. Lett. 70, 113 (1993); Phys. Rev. D 52, 6882 (1995); L. Blanchet, T. Damour, B. R. Iyer, C. M. Will and A. G. Wiseman, Phys. Rev. Lett. 74, 3515 (1995); L. Blanchet, Phys. Rev. D 54, 1417 (1996); 55, 714 (1997); P. Jaranowski and G. Schäfer, ibid. 55, 4712 (1997); A. Gopakumar, B. R. Iyer and S. Iyer, ibid. 55, 6030 (1997); 57, 6562(E) (1998); A. Gopakumar and B. R. Iyer, ibid. 56, 7708 (1997).   
[33] L. Blanchet et al. (in preparation).   
[34] D. W. Arnett and R. L. Bowers, Astrophys. J., Suppl. 33, 415 (1977).