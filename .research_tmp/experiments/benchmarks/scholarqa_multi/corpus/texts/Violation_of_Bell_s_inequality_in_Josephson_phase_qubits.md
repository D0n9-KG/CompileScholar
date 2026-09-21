# Analysis of Bell inequality violation in superconducting phase qubits

Abraham G. Kofman and Alexander N. Korotkov

Department of Electrical Engineering, University of California, Riverside, California 92521

(Dated: November 1, 2018)

We analyze conditions for violation of the Bell inequality in the Clauser-Horne-Shimony-Holt form, focusing on the Josephson phase qubits. We start the analysis with maximum violation in the ideal case, and then take into account the effects of the local measurement errors and decoherence. A special attention is paid to configurations of the qubit measurement directions in the pseudospin space lying within either horizontal or vertical planes; these configurations are optimal in certain cases. Besides local measurement errors and decoherence, we also discuss the effect of measurement crosstalk, which affects both the classical inequality and the quantum result. In particular, we propose a version of the Bell inequality which is insensitive to the crosstalk.

PACS numbers: 03.65.Ud, 85.25.Cp, 03.67.Lx

# I. INTRODUCTION

In 1935 Einstein, Podolsky, and Rosen (EPR) have shown in the classical paper $^{1}$ that quantum mechanics contradicts the natural assumption (the “local realism”) that a measurement of one of two spatially separated objects does not affect the other one. This “spooky action at a distance” – known as an entanglement – is now recognized as a major resource in the field of quantum information and quantum computing. $^{2}$ The paradox has led EPR to conclude that quantum mechanics is an incomplete description of physical reality, thus implying that some local hidden variables are needed.

The EPR paradox remained at the level of semi-philosophical discussions until 1964, when John Bell contributed an inequality for results of a spin-correlation experiment, $^{3}$ which should hold for any theory involving local hidden variables, but is violated by quantum mechanics. Inspired by Bell's idea, in 1969, Clauser, Horne, Shimony, and Holt $^{4}$ (CHSH) proposed a version of the Bell inequality (BI, the generic name for a family of inequalities), which made the experimental testing of local hidden-variable theories possible. The main advantage of the CHSH inequality in comparison with the original BI is that it does not rely on an experimentally unrealistic assumption of a perfect anticorrelation between the measurement results when two spin-1/2 particles (in the spin-0 state) are measured along the same direction. Many interesting experiments $^{5,6,7,8,9,10}$ have been done since then. The results of these experiments clearly show a violation of the Bell inequalities, in accordance with quantum mechanical predictions.

The BI violation has been mostly demonstrated in the experiments with photons; $^{5,6,7}$ it has been also shown in the experiments with ions in traps $^{8}$ and with an atom-photon system; $^{9}$ a Bell-like inequality violation has been also demonstrated in an experiment with single neutrons. $^{10}$ Experimental violation of the BI in solid-state qubits would be an important step towards practical quantum information processing by solid-state devices. $^{11,12,13,14,15}$ Experiments on observation of the BI violation in Josephson phase qubits are currently underway. $^{16,17}$ Theoretical study related to the BI violation in solid-state systems has also attracted significant attention in recent years. $^{18,19,20,21,22,23,24,25}$

In this paper we discuss the Bell inequality (in the CHSH form) for solid-state systems, focusing on experiments with superconducting phase qubits. We study effects of various factors detrimental for observation of the BI violation, including local measurement errors, decoherence, and interaction between qubits (crosstalk), and analyze optimal conditions in presence of these nonidealities.

The paper is organized as follows. In Sec. II we review the CHSH type of the BI, and also discuss tomography-type measurements using qubit rotations. In Sec. III we consider the ideal case and describe all situations for which the BI is violated maximally. Sections IV-VI are devoted to the effects of various nonidealities on the observation of the BI violation. In Sec. IV we discuss the effect of local measurement errors, using a more general error model than in the previous approaches. $^{26,27,28,29,30}$ Analytical results for maximally entangled states and numerical results for general two-qubit states are presented. In Sec. V we consider the effect of local decoherence of the qubits. In Sec. VI we discuss measurement crosstalk, which affects both the BI (since crosstalk is a classical mechanism of communication between qubits) and the quantum result. We also propose a version of the BI, which is not affected by the crosstalk. Section VII provides concluding remarks.

# II. PRELIMINARIES

# A. CHSH inequality

We begin with a brief review of the CHSH inequality, $^{4,31,32}$ a type of the Bell inequality usually used in experiments. Let us consider a pair of two-level systems (qubits) a and b. Assuming that a realistic (classical) theory based on local observables $^{3}$ holds and there is no communication between the qubits (i.e., no crosstalk), the two-qubit measurement results should

satisfy the CHSH inequality $^{4,33}$

$$
- 2 \leq S \leq 2, \tag {1}
$$

where

$$
S = E (\vec {a}, \vec {b}) - E (\vec {a}, \vec {b} ^ {\prime}) + E (\vec {a} ^ {\prime}, \vec {b}) + E (\vec {a} ^ {\prime}, \vec {b} ^ {\prime}). (2)
$$

Here $\vec{a}$ and $\vec{a}'$ ( $\vec{b}$ and $\vec{b}'$ ) are the unit radius-vectors on the Bloch sphere along the measurement axes for qubit a (b) and $E(\vec{a},\vec{b})$ is the correlator of the measurement results:

$$
E (\vec {a}, \vec {b}) = p _ {+ +} (\vec {a}, \vec {b}) + p _ {- - } (\vec {a}, \vec {b}) - p _ {+ -} (\vec {a}, \vec {b}) - p _ {- +} (\vec {a}, \vec {b}), \tag {3}
$$

where $p_{ij}(\vec{a},\vec{b})$ ( $i,j = \pm$ ) is the joint probability of measurement results $i$ and $j$ for qubits $a$ and $b$ , respectively.

The sum of the probabilities in Eq. (3) equals one. This can be used to recast Eq. (2) in the form

$$
S = 4 T + 2, \tag {4}
$$

where

$$
T = p (\vec {a}, \vec {b}) - p (\vec {a}, \vec {b} ^ {\prime}) + p (\vec {a} ^ {\prime}, \vec {b}) + p (\vec {a} ^ {\prime}, \vec {b} ^ {\prime}) - p _ {a} (\vec {a} ^ {\prime}) - p _ {b} (\vec {b}). \tag {5}
$$

Here $p(\vec{a},\vec{b}) = p_{++}(\vec{a},\vec{b})$ , whereas $p_{a}(\vec{a}^{\prime}) = p_{++}(\vec{a}^{\prime},\vec{b}) + p_{+-}(\vec{a}^{\prime},\vec{b})$ [or $p_{b}(\vec{b}) = p_{++}(\vec{a},\vec{b}) + p_{-+}(\vec{a},\vec{b})$ ] is the probability of the measurement result “+” for qubit a (or b) irrespective of the measurement result for the other qubit [in the classical theory in absence of communication between qubits, $p_{a}(\vec{a}^{\prime})$ is obviously independent of the direction $\vec{b}$ and even independent of the very fact of the qubit b measurement; similarly for $p_{b}(\vec{b})$ ]. Thus, instead of the probabilities $p_{ij}$ , one can equivalently use the probabilities p, $p_{a}$ , and $p_{b}$ , and the inequality (1) can be recast in an equivalent form $^{4}$

$$
- 1 \leq T \leq 0. \tag {6}
$$

Notice that both inequalities (1) and (6) can have somewhat different meanings in different physical situations. In particular, the results “+” and “−” may correspond to the presence or absence of the detector “click” (so called “one-channel measurement” $^{4,34}$ ); in this case a low-efficiency detector significantly increases the chance of the result “−”. Another possibility is the so-called “two-channel measurement”, in which the qubit states“+” and “−” are supposed to produce clicks in different detectors; $^{33}$ in this case inefficient detection leads to three possible results: “+”, “−”, and “no result”. Significant inefficiency of the optical detectors leads to the so-called “detector loophole,” $^{35,36}$ which arises because effectively not the whole ensemble of the qubit pairs is being measured. This problem is often discussed in terms of contrasting the Clauser-Horne $^{34}$ (CH) and CHSH interpretations of the inequalities, which differ by considering either the whole ensemble or a subensemble of the qubit pairs. It is important to mention that in the case of the Josephson phase qubits (which formally belongs to the class of one-channel measurements) the whole ensemble of qubit pairs is being measured, and therefore there is no detector loophole (if one avoids $^{17}$ corrections for measurement errors), as well as there is no difference between CHSH and CH interpretations.

# B. Tomographic measurements

In some cases, as for Josephson phase qubits, the measurement (detector) axis cannot be physically rotated. However, instead of the detector rotation, one can rotate the qubit state. $^{37,38}$ Let us show the equivalence of the two methods explicitly, using the example of the phase qubit and assuming ideal (orthodox) measurement.

The Hamiltonian of the phase qubit in a microwave field in the subspace of the two lowest states in the qubit potential well $|\psi_{0}\rangle$ and $|\psi_{1}\rangle$ is

$$
\begin{array}{l} H _ {q} = (\hbar \omega_ {q} / 2) (| \psi_ {1} \rangle \langle \psi_ {1} | - | \psi_ {0} \rangle \langle \psi_ {0} |) \\ + \hbar \Omega (t) \sin (\omega t + \phi) (| \psi_ {0} \rangle \langle \psi_ {1} | + | \psi_ {1} \rangle \langle \psi_ {0} |), (7) \\ \end{array}
$$

where $\hbar$ is the Planck constant, $\omega_{q}$ is the qubit resonance frequency, $\omega$ is the microwave frequency, $\Omega(t)=d_{10}E(t)/\hbar$ is the time-dependent Rabi frequency, $d_{10}$ is the dipole-moment matrix element, and $E(t)$ is the electric-strength amplitude of the microwave field. Transforming to the qubit basis (the rotating frame) $|0\rangle=|\psi_{0}\rangle$ , $|1\rangle=e^{i\omega t}|\psi_{1}\rangle$ , neglecting fast oscillating terms in the Hamiltonian (the rotating-wave approximation) and assuming the resonance condition $\omega=\omega_{q}$ , we arrive at the Hamiltonian

$$
H (t) = \frac {\hbar \Omega (t)}{2} \vec {n} \cdot \vec {\sigma}, \tag {8}
$$

where the unit vector $\vec{n} = (\sin \phi, -\cos \phi, 0)$ lies in the xy plane making the angle $\phi - \pi/2$ with the x axis, whereas $\vec{\sigma} = (\sigma_x, \sigma_y, \sigma_z)$ is the vector of the Pauli matrices. $^{39}$ Here and below we associate state $|1\rangle(|0\rangle)$ with the measurement result “+” (“−”) and with the eigenvalue 1 (-1) of $\sigma_z$ , so that $\sigma_z = |1\rangle\langle1| - |0\rangle\langle0|$ .

As follows from Eq. (8), the microwave pulse rotates the qubit state such that the initial density matrix $\rho_{q}$ becomes $\tilde{\rho}_{q}=U_{R}\rho_{q}U_{R}^{\dagger}$ , where

$$
\begin{array}{l} U _ {R} = e ^ {- i \theta \vec {n} \cdot \vec {\sigma} / 2} = \cos (\theta / 2) - i \vec {n} \cdot \vec {\sigma} \sin (\theta / 2) \\ = \left( \begin{array}{c c} \cos (\theta / 2) & e ^ {- i \phi} \sin (\theta / 2) \\ - e ^ {i \phi} \sin (\theta / 2) & \cos (\theta / 2) \end{array} \right). \tag {9} \\ \end{array}
$$

Here $\theta = \int_{t_1}^{t_2} dt \Omega(t)$ , where $t_1$ and $t_2$ are the pulse starting and ending time moments.

The probability of the qubit to be found in state $|i\rangle$ is

$$
p _ {i} = \mathrm{Tr} (| i \rangle \langle i | \tilde {\rho} _ {q}) = \mathrm{Tr} (P _ {i} \rho_ {q}) (i = 0, 1). \tag {10}
$$

Here $P_{i}$ is the projection operator $P_{i} = U_{R}^{\dagger}|i\rangle \langle i|U_{R}$ , i.e.,

$$
P _ {1} (\vec {a}) = \frac {1}{2} \left( \begin{array}{c c} 1 + \cos \theta & e ^ {- i \phi} \sin \theta \\ e ^ {i \phi} \sin \theta & 1 - \cos \theta \end{array} \right) = \frac {1}{2} (I + \vec {a} \cdot \vec {\sigma}),
$$

$$
P _ {0} (\vec {a}) = I - P _ {1} (\vec {a}) = \frac {1}{2} (I - \vec {a} \cdot \vec {\sigma}), \tag {11}
$$

where $I$ is the identity matrix and $\vec{a}$ is the unit vector

$$
\vec {a} = (\cos \phi \sin \theta , \sin \phi \sin \theta , \cos \theta) \tag {12}
$$

defining the measurement axis.

Equations (10) and (11) show explicitly the equivalence between the qubit and detector rotations. Namely, one can interpret $p_{1}$ ( $p_{0}$ ) as the probability of the qubit to be found in the state with the pseudospin parallel (antiparallel) to the measurement axis $\vec{a}$ .

Notice that the microwave phase $\phi$ is naturally defined modulo $2\pi$ , while the Rabi rotation angle $\theta$ can be always reduced to a $2\pi$ range (we will assume $-\pi < \theta \leq \pi$ ). Nevertheless, with this restriction there are still two sets of angles $(\theta, \phi)$ corresponding to the same measurement direction $\vec{a}$ , because $\vec{a}$ is invariant under the change

$$
(\theta , \phi) \leftrightarrow (- \theta , \phi + \pi). \tag {13}
$$

It is easy to make a one-to-one correspondence between the measurement direction $\vec{a}$ and angles $(\theta, \phi)$ by limiting either $\theta$ or $\phi$ to a $\pi$ -range (instead of $2\pi$ ). However, we prefer not to do that because it is convenient and natural physically to have a $2\pi$ range for one angle when the other angle is fixed. So, as follows from Eqs. (12) and (13), the polar (zenith) and azimuth spherical coordinates of $\vec{a}$ are equal to, respectively, $\theta$ and $\phi$ when $\theta \geq 0$ , and $-\theta$ and $\phi + \pi$ when $\theta < 0$ .

The joint probability of the two-qubit measurement can be written as

$$
p _ {i j} (\vec {a}, \vec {b}) = \mathrm{Tr} [ P _ {i} ^ {a} (\vec {a}) P _ {j} ^ {b} (\vec {b}) \rho ], \tag {14}
$$

where $\rho$ is the two-qubit density matrix, $P_{i}^{a} = P_{i} \otimes I$ , and $P_{i}^{b} = I \otimes P_{i}$ .

# III. MAXIMUM BI VIOLATION: IDEAL CASE

The purpose of this paper is to analyze conditions needed to observe the BI violation in experiment. Since it is usually easier to observe an effect when it is maximal, we start the analysis with the situations where violation of the BI is maximal.

# A. Bell operator and Cirel'son's bounds

Equations (3) and (14) yield $E(\vec{a},\vec{b}) = \mathrm{Tr}(AB\rho)$ , where

$$
A = P _ {1} ^ {a} (\vec {a}) - P _ {0} ^ {a} (\vec {a}) = \vec {a} \cdot \vec {\sigma} _ {a} \tag {15}
$$

and similarly $B = \vec{b} \cdot \vec{\sigma}_{b}$ . Here $\vec{\sigma}_{a} = \vec{\sigma} \otimes I$ and $\vec{\sigma}_{b} = I \otimes \vec{\sigma}$ , the eigenvalues of A and B being $\pm 1$ . Correspondingly, as follows from Eq. (2),

$$
S = \operatorname{Tr} (\mathcal {B} \rho), \tag {16}
$$

where the Bell operator $^{40}$ B is

$$
\mathcal {B} = A B - A B ^ {\prime} + A ^ {\prime} B + A ^ {\prime} B ^ {\prime} \tag {17}
$$

(here $A' = \vec{a}' \cdot \vec{\sigma}_a$ and $B' = \vec{b}' \cdot \vec{\sigma}_b$ ).

The maximum and minimum values of $S$ (so-called Cirel'son's bounds $^{41}$ )

$$
S _ {\pm} = \pm 2 \sqrt {2} \tag {18}
$$

can be obtained, for example, in the following way. $^{42}$ Since the Bell operator B is Hermitian, $S_{\pm}$ are equal to the maximum and minimum eigenvalues of B. These eigenvalues can be found analyzing the eigenvalues of $B^{2}$ :

$$
\mathcal {B} ^ {2} = 4 + [ A, A ^ {\prime} ] [ B, B ^ {\prime} ] = 4 - 4 (\vec {a} \times \vec {a} ^ {\prime} \cdot \vec {\sigma} _ {a}) (\vec {b} \times \vec {b} ^ {\prime} \cdot \vec {\sigma} _ {b}) (1 9)
$$

(here the vector product is taken before the scalar product). Since the eigenvalues of the Pauli matrices are equal to $\pm1$ , the largest eigenvalue of $B^{2}$ is 8, achieved when $\vec{a} \perp \vec{a}'$ and $\vec{b} \perp \vec{b}'$ . In this case the maximum and minimum eigenvalues of B are $\pm\sqrt{8}$ , thus leading to Eq. (18). (Both values $\pm\sqrt{8}$ are realized because in this case the other eigenvalue of $B^{2}$ is 0, and the sum of all four eigenvalues of B should be equal to 0 since TrB = 0.)

Consider some useful properties of S. As follows from Eqs. (16) and (17), the value of S is invariant under arbitrary local unitary transformations $U_{a}$ and $U_{b}$ ,

$$
\rho \rightarrow (U _ {a} \otimes U _ {b}) \rho (U _ {a} ^ {\dagger} \otimes U _ {b} ^ {\dagger}), \tag {20a}
$$

if simultaneously $A \to U_aAU_a^\dagger$ , $A' \to U_aA'U_a^\dagger$ , $B \to U_bBU_b^\dagger$ , $B' \to U_bB'U_b^\dagger$ or, equivalently,

$$
\vec {a} \rightarrow R _ {a} \vec {a}, \quad \vec {a} ^ {\prime} \rightarrow R _ {a} \vec {a} ^ {\prime}, \quad \vec {b} \rightarrow R _ {b} \vec {b}, \quad \vec {b} ^ {\prime} \rightarrow R _ {b} \vec {b} ^ {\prime}, \tag {20b}
$$

where $R_{a}$ ( $R_{b}$ ) is the rotation matrix corresponding to $U_{a}$ ( $U_{b}$ ), so that, e.g., $U_{a}(\vec{a} \cdot \vec{\sigma})U_{a}^{\dagger} = (R_{a}\vec{a}) \cdot \vec{\sigma}$ . This invariance is an obvious consequence of the equivalence between the qubit and detector rotations, discussed in the previous section. As a result of the invariance, if some state is known to violate the BI for a given configuration of the detectors, one can obtain many other states and the corresponding detector configurations providing the same BI violation, by using Eqs. (20) with all possible local rotations.

Note also that $\mathcal{B}$ inverts the sign if the pair of vectors $\vec{a},\vec{a}^{\prime}$ (or $\vec{b},\vec{b}^{\prime}$ ) inverts the sign. Correspondingly, for a given state

$$
S \rightarrow - S \text {   if   } \vec {a} \rightarrow - \vec {a}, \vec {a} ^ {\prime} \rightarrow - \vec {a} ^ {\prime} (\text { or   } \vec {b} \rightarrow - \vec {b}, \vec {b} ^ {\prime} \rightarrow - \vec {b} ^ {\prime}). \tag {21}
$$

As follows from Eq. (21), there is a one-to-one correspondence between the classes of detector configurations maximizing and minimizing S for a given state.

# B. Optimal detector configurations for maximally entangled states

For any given detector configuration satisfying the condition

$$
\vec {a} \perp \vec {a} ^ {\prime} \quad \text { and } \quad \vec {b} \perp \vec {b} ^ {\prime}, \tag {22}
$$

each of the Cirel'son's bounds (18) is achieved for a unique maximally entangled state. $^{43,44}$ In contrast, for a given maximally entangled state there can be many optimal detector configurations giving the maximal BI violation. To the best of our knowledge, only the configurations with the detector axes lying in one plane have been usually considered in the literature, though generally the detector axes for different qubits may lie in two different planes. Moreover, the BI violation has been studied mainly for one of the Bell states, $^{2}$

$$
\left| \Psi_ {\pm} \right\rangle = \left(\left| 1 0 \right\rangle \pm \left| 0 1 \right\rangle\right) / \sqrt {2}, \tag {23}
$$

$$
\left| \Phi_ {\pm} \right\rangle = \left(\left| 0 0 \right\rangle \pm \left| 1 1 \right\rangle\right) / \sqrt {2}, \tag {24}
$$

while an arbitrary maximally entangled state can be written as

$$
| \Psi_ {\mathrm{me}} \rangle = (| \chi_ {1} ^ {a} \chi_ {1} ^ {b} \rangle + | \chi_ {2} ^ {a} \chi_ {2} ^ {b} \rangle) / \sqrt {2}, \tag {25}
$$

where $\{|\chi_{1}^{k}\rangle, |\chi_{2}^{k}\rangle\}$ is an orthonormal basis for qubit k. Our purpose here is to determine all optimal detector configurations for any maximally entangled state.

# 1. Singlet state

Let us start, assuming that the qubits are in the singlet state $|\Psi_{-}\rangle$ . For this state $E(\vec{a},\vec{b})=\langle\Psi_{-}|(\vec{a}\cdot\vec{\sigma})\otimes(\vec{b}\cdot\vec{\sigma})|\Psi_{-}\rangle=-\vec{a}\cdot\vec{b}$ , so that [see Eq. (2)]

$$
S = \vec {a} \cdot (\vec {b} ^ {\prime} - \vec {b}) - \vec {a} ^ {\prime} \cdot (\vec {b} + \vec {b} ^ {\prime}). \tag {26}
$$

Maximizing this formula over $\vec{a}$ and $\vec{a}'$ , we should choose $\vec{a}$ to be parallel to $\vec{b}' - \vec{b}$ , while $\vec{a}'$ should be antiparallel to $\vec{b}' + \vec{b}$ . Therefore $^{45}$ $\vec{a} \perp \vec{a}'$ , since $\vec{b}' - \vec{b}$ and $\vec{b} + \vec{b}'$ are mutually orthogonal. Similarly, rewriting S as $S = \vec{b}' \cdot (\vec{a} + \vec{a}') - \vec{b} \cdot (\vec{a} - \vec{a}')$ , we can show that maximization of S requires $\vec{b} \perp \vec{b}'$ . In this way we easily show that the necessary and sufficient condition for reaching the upper bound $S_{+} = 2\sqrt{\frac{1}{2}}$ for the singlet state is

$$
\vec {a} \perp \vec {a} ^ {\prime}, \vec {b} = - (\vec {a} + \vec {a} ^ {\prime}) / \sqrt {2}, \vec {b} ^ {\prime} = (\vec {a} ^ {\prime} - \vec {a}) / \sqrt {2}. \tag {27}
$$

Because of the symmetry (21), the necessary and sufficient condition for reaching the lower bound $S_{-} = -2\sqrt{2}$ can be obtained by the inversion of the detector directions for one of the qubits:

$$
\vec {a} \perp \vec {a} ^ {\prime}, \quad \vec {b} = (\vec {a} + \vec {a} ^ {\prime}) / \sqrt {2}, \quad \vec {b} ^ {\prime} = (\vec {a} - \vec {a} ^ {\prime}) / \sqrt {2}. \tag {28}
$$

Equations (27) and (28) show that for the singlet state the maximum BI violation requires that the detector axes for both qubits lie in the same plane. However, the orientation of this plane is arbitrary, since Eqs. (27) and (28) determine only the angles between the detector axes.

All configurations maximizing (or minimizing) S for the singlet state can be obtained from one maximizing (minimizing) configuration by all possible rotations of the plane containing the detector axes. As the initial maximizing case we can choose the most standard configuration $^{4,32}$ when all detector axes are within xz plane,

$$
\phi_ {a} = \phi_ {a} ^ {\prime} = \phi_ {b} = \phi_ {b} ^ {\prime} = 0, \tag {29}
$$

and the polar (zenith) angles of the detector directions $\vec{a},\vec{a}^{\prime},\vec{b},\vec{b}^{\prime}$ are

$$
\theta_ {a} = 0, \theta_ {a} ^ {\prime} = \pi / 2, \theta_ {b} = - 3 \pi / 4, \theta_ {b} ^ {\prime} = - \pi / 4. (3 0)
$$

Then all detector configurations with $S = 2\sqrt{2}$ can be parametrized by three Euler angles $^{46}$ $\kappa_{1}$ , $\kappa_{2}$ , and $\kappa_{3}$ ( $0 \leq \kappa_{1,3} \leq 2\pi$ , $0 \leq \kappa_{2} \leq \pi$ ), which describe an arbitrary rotation of the configuration (29)–(30).

Similarly, all minimizing configurations $(S = -2\sqrt{2})$ can be obtained from the standard $xz$ case [Eq. (29)] with

$$
\theta_ {a} = 0, \theta_ {a} ^ {\prime} = \pi / 2, \theta_ {b} = \pi / 4, \theta_ {b} ^ {\prime} = 3 \pi / 4 \tag {31}
$$

by arbitrary rotations of this configuration, characterized by three Euler angles $\kappa_{1,2,3}$ .

# 2. General maximally entangled state

Any maximally entangled two-qubit state can be obtained from the singlet state by a unitary transformation of the basis of one of the qubits $^{47}$ (i.e. a one-qubit rotation). Therefore, an arbitrary case corresponding to the bounds $S_{\pm} = \pm 2\sqrt{2}$ can be reduced to the singlet state considered above by a unitary transformation $U_{b}$ of the qubit b basis and simultaneous corresponding rotation of the detector axes $\vec{b}$ and $\vec{b}'$ for the second qubit. Since the transformation $U_{b}$ can also be characterized by three Euler angles $\kappa_{1}^{b}, \kappa_{2}^{b}$ , and $\kappa_{3}^{b}$ , an arbitrary situation with $S = 2\sqrt{2}$ can be characterized by six independent parameters $(\kappa_{1}, \kappa_{2}, \kappa_{3}, \kappa_{1}^{b}, \kappa_{2}^{b}, \kappa_{3}^{b})$ , using the standard configuration (29)–(30) as a starting point. Similarly, any situation with $S = -2\sqrt{2}$ is characterized by the same six parameters, starting with the xz configuration (31).

Since these six parameters can describe arbitrary directions of four measurement axes $(\vec{a}, \vec{a}', \vec{b}, \vec{b}')$ still satisfying the conditions $\vec{a} \perp \vec{a}'$ and $\vec{b} \perp \vec{b}'$ , it is obvious that any such four-axes configuration produces $S = 2\sqrt{2}$ for exactly one entangled state and also produces $S = -2\sqrt{2}$ for another entangled state. Notice that the sign of S can obviously be flipped by a $\pi$ -rotation of qubit a (or b) around the axis $\vec{a} \times \vec{a}'$ ( $\vec{b} \times \vec{b}'$ ) instead of the $\pi$ -rotation (21) of its detector axes. Also notice that six independent parameters for an optimal configuration can be alternatively chosen as any parameters characterizing the four measurement axes, which are pair-wise orthogonal: $\vec{a} \perp \vec{a}'$ and $\vec{b} \perp \vec{b}'$ .

# 3. Odd states

An important special case is the class of “odd” maximally entangled states

$$
| \Psi \rangle = (| 1 0 \rangle + e ^ {i \alpha} | 0 1 \rangle) / \sqrt {2} (0 \leq \alpha <   2 \pi), \tag {32}
$$

which is of relevance for experiments with Josephson phase qubits. $^{12}$ Such states can be obtained (with an accuracy up to an overall phase factor) from the singlet state $|\Psi_{-}\rangle$ by unequal rotations of the two qubits around the z axis. Indeed, since $U_{z}(\varphi)=e^{-i\varphi\sigma_{z}/2}$ rotates a spin 1/2 around the z axis by angle $\varphi$ , we obtain $[U_{z}(\alpha_{0})\otimes U_{z}(\alpha_{0}+\pi-\alpha)]|\Psi_{-}\rangle=-ie^{-i\alpha/2}|\Psi\rangle$ , where $\alpha_{0}$ is arbitrary. Thus, in view of Eq. (20), the optimal detector configurations for the odd state (32) can be obtained from those for $|\Psi_{-}\rangle$ by rotating the detectors for the qubit b around the z axis by the angle $\pi-\alpha$ [notice that the state (32) reduces to the singlet for $\alpha=\pi$ ]. In terms of the parameters $\theta$ and $\phi$ , this is equivalent to the change

$$
\phi_ {b} \rightarrow \phi_ {b} + \pi - \alpha , \quad \phi_ {b} ^ {\prime} \rightarrow \phi_ {b} ^ {\prime} + \pi - \alpha . \tag {33}
$$

Thus, for the odd states the class of optimal configurations maximizing S (as well as the class minimizing S) is characterized by four parameters: $\kappa_{1}$ , $\kappa_{2}$ , $\kappa_{3}$ , and $\alpha$ .

Now let us focus on the optimal configurations with the detector axes lying either in a “vertical” plane for each qubit (i.e., a plane containing the z axis) or the “horizontal” (xy) plane; such configurations will be important in the study of effects of errors (Sec. IV) and decoherence (Sec. V).

To obtain all “vertical” cases with the maximum BI violation $S = 2\sqrt{2}$ , we start with the standard configuration (29)–(30) for the singlet, then apply a rotation in the xz plane by an arbitrary angle C (we can also apply the mirror reflection), then rotate the resulting configuration around the z axis by an arbitrary angle $\phi_{0}$ , $^{48}$ and finally apply the $\alpha$ -rotation (33) determined by the phase of the odd state (32). As a result, the optimal measurement directions for the qubits a and b generally lie in different vertical planes,

$$
\phi_ {a} = \phi_ {a} ^ {\prime} = \phi_ {0}, \quad \phi_ {b} = \phi_ {b} ^ {\prime} = \phi_ {0} + \pi - \alpha , \tag {34}
$$

while the polar angles corresponding to $S = 2\sqrt{2}$ are

$$
\left(\theta_ {a}, \theta_ {a} ^ {\prime}, \theta_ {b}, \theta_ {b} ^ {\prime}\right) = \pm (0, \pi / 2, - 3 \pi / 4, - \pi / 4) + C, \tag {35}
$$

where $\phi_{0}$ and C are arbitrary angles, while $\alpha$ is determined by the state (32). Notice that the signs $\pm$ correspond to the possibility of the mirror reflection, which we did not have to consider in the previous subsections because it can be reproduced using 3D rotations, while it is a necessary extra transformation in the 2D case.

Similarly, the minimum $S = -2\sqrt{2}$ for the odd state (32) is achieved for the vertical configurations within the planes given by Eq. (34) for the polar angles

$$
\left(\theta_ {a}, \theta_ {a} ^ {\prime}, \theta_ {b}, \theta_ {b} ^ {\prime}\right) = \pm (0, \pi / 2, \pi / 4, 3 \pi / 4) + C. \tag {36}
$$

Recall that we define both $\theta$ and $\phi$ modulo $2\pi$ , and therefore each measurement direction corresponds to two sets of $(\theta,\phi)$ [see Eq. (13)]. Consequently, the optimal configurations described by Eqs. (34)–(36) can be also described in several equivalent forms by applying the transformation (13) to some of the four measurement directions.

As follows from Eq. (34), the only odd states for which the optimal vertical configurations lie in the same plane are the Bell states $|\Psi_{-}\rangle$ (corresponding to $\alpha = \pi$ ) and $|\Psi_{+}\rangle$ (corresponding to $\alpha = 0$ ). The optimal vertical configurations for the singlet state $|\Psi_{-}\rangle$ are given by

$$
\phi_ {a} = \phi_ {a} ^ {\prime} = \phi_ {b} = \phi_ {b} ^ {\prime} = \phi_ {0} \tag {37}
$$

and Eqs. (35)-(36). To describe the optimal vertical configurations for the state $|\Psi_{+}\rangle$ , it is natural to apply the equivalence (13) to the qubit $b$ measurement directions, so that the angles $\phi$ are still all equal as in Eq. (37), while the angles $\theta$ are given by Eqs. (35)-(36) with flipped signs for the qubit $b$ , i.e. $\theta_{b} \to -\theta_{b}$ and $\theta_{b}' \to -\theta_{b}'$ .

Now let us consider the optimal detector configurations in the horizontal (xy) plane:

$$
\theta_ {a} = \theta_ {a} ^ {\prime} = \theta_ {b} = \theta_ {b} ^ {\prime} = \pi / 2. \tag {38}
$$

All configurations for $S = 2\sqrt{2}$ can be obtained from the standard configuration (29)–(30) by rotating it into the xy plane (so that the angles $\theta$ are essentially replaced by the angles $\phi$ ), then applying an arbitrary rotation within xy plane and possibly the mirror reflection, and finally applying the transformation (33) with the state-dependent parameter $\alpha$ , so that

$$
\left(\phi_ {a}, \phi_ {a} ^ {\prime}, \phi_ {b} + \alpha , \phi_ {b} ^ {\prime} + \alpha\right) = \pm (0, \pi / 2, \pi / 4, 3 \pi / 4) + C \tag {39}
$$

with arbitrary $C$ (the signs $\pm$ correspond again to the possibility of the mirror reflection).

Similarly, all horizontal configurations corresponding to $S = -2\sqrt{2}$ for the odd states are described by the angles

$$
\left(\phi_ {a}, \phi_ {a} ^ {\prime}, \phi_ {b} + \alpha , \phi_ {b} ^ {\prime} + \alpha\right) = \pm (0, \pi / 2, - 3 \pi / 4, - \pi / 4) + C. \tag {40}
$$

Notice that the application of the equivalence (13) to all four measurement directions changes $\pi/2$ into $-\pi/2$ in Eq. (38), while Eqs. (39) and (40) do not change, since the corresponding $\pi$ -shift of angles $\phi$ can be absorbed by the arbitrary parameter C.

# IV. LOCAL MEASUREMENT ERRORS

In this section we consider the effects of local (independent) measurement errors on the BI violation.

# A. Error model

The probabilities of the measurement results for a single qubit can be written in the form

$$
p _ {i} ^ {M} = \sum_ {m = 0} ^ {1} F _ {i m} p _ {m} = \mathrm{Tr} (Q _ {i} \rho_ {q}), \tag {41}
$$

where $p_{m}$ are the probabilities which would be obtained by ideal measurements, $F_{im}$ is the probability to find the qubit in the state $|i\rangle$ when it is actually in the state $|m\rangle$ , and operator $Q_{i}=F_{i0}P_{0}+F_{i1}P_{1}$ contains the projector operators $P_{0,1}$ [see Eq. (10)]. The operators $Q_{i}$ satisfy the same condition as the POVM measurement operators, $^{2}$ namely, $Q_{i}$ are positive and $Q_{0}+Q_{1}=1$ . The condition $p_{0}^{M}+p_{1}^{M}=1$ implies that $F_{0m}+F_{1m}=1$ . Hence, the matrix F has two independent parameters, which can be chosen as the measurement fidelities $F_{0}\equiv F_{00}$ and $F_{1}\equiv F_{11}$ for the states $|0\rangle$ and $|1\rangle$ , so that

$$
p _ {0} ^ {M} = F _ {0} \tilde {\rho} _ {0 0} + (1 - F _ {1}) \tilde {\rho} _ {1 1}, p _ {1} ^ {M} = (1 - F _ {0}) \tilde {\rho} _ {0 0} + F _ {1} \tilde {\rho} _ {1 1} \tag {42}
$$

(here $\tilde{\rho}_{ij}$ are the components of the one-qubit density matrix after the tomographic rotation and $0 \leq F_{0,1} \leq 1$ ). It can be always assumed that $F_{0} + F_{1} \geq 1$ , since in the opposite case the measurement results can be simply renamed: $0 \leftrightarrow 1$ ; as a consequence, $\max(F_{0}, F_{1}) \geq 1/2$ .

Using the assumption that measurement errors for each qubit can be considered independently of the errors for the other qubit, the measured probabilities for a qubit pair can be written in the form $^{31}$

$$
p _ {i j} ^ {M} = \sum_ {m, n = 0} ^ {1} F _ {i m} ^ {a} F _ {j n} ^ {b} p _ {m n} = \mathrm{Tr} (Q _ {i} ^ {a} Q _ {j} ^ {b} \rho), \tag {43}
$$

where $F_{im}^{k}$ is the matrix $F_{im}$ for qubit $k$ and $Q_{i}^{k} = F_{i0}^{k}P_{0}^{k} + F_{i1}^{k}P_{1}^{k}$ [see Eq. (14)].

In this section we will discuss the condition for the BI violation as a function of measurement fidelities $F_{0}^{k}$ and $F_{1}^{k}$ . Sometimes we will limit the analysis to the case of equal measurement fidelities for both qubits,

$$
F _ {i} ^ {a} = F _ {i} ^ {b} = F _ {i}, \tag {44}
$$

however, the case of different measurement fidelities for the two qubits is also of interest. (Different fidelities are especially of interest when the qubits have different physical implementations. For instance, in the case of an atom-photon qubit pair $^{9}$ the detection efficiency for the atom is nearly 100%, whereas the photon-detector efficiency is significantly less than 100%.) Notice that for the Josephson phase qubits the trade-off between the fidelities $F_{0}^{k}$ and $F_{1}^{k}$ can be controlled in the experiment $^{12,49}$ for each qubit individually by changing the measurement pulse strength.

Several special cases of our error model have been previously discussed in the literature, starting with the CHSH paper. $^{4}$ For example, in the problem of the detector loophole $^{35,36}$ the CH inequality with $F_{0}^{a}=F_{0}^{b}=1$ is often considered; then $F_{1}^{k}$ is called the detector efficiency; both the cases $F_{1}^{a} = F_{1}^{b}$ (Refs. 26,29) and $F_{1}^{b} \neq F_{1}^{a}$ (Ref. 30) have been considered. Let us also mention the effect of nonidealities on the BI violation considered for the experiments on two-photon interference. $^{27,28}$ The situations of Refs. 27,28 formally correspond to the special case of our model with

$$
F _ {0} ^ {a} = F _ {1} ^ {a} = F _ {a}, F _ {0} ^ {b} = F _ {1} ^ {b} = F _ {b}. \tag {45}
$$

Then the product $(2F_{a}-1)(2F_{b}-1)$ equals either the visibility $^{27}$ or the product of the visibility and the square of the signal acceptance probability. $^{28}$

# B. General relations for $S$

The Bell operator (17) can be generalized to the case of measurement errors. Inserting Eq. (43) into Eq. (3) yields $E(\vec{a}, \vec{b}) = \mathrm{Tr}(\tilde{A}\tilde{B}\rho)$ , where

$$
\tilde {A} = Q _ {1} ^ {a} - Q _ {0} ^ {a} = F _ {1} ^ {a} - F _ {0} ^ {a} + (F _ {0} ^ {a} + F _ {1} ^ {a} - 1) \vec {a} \cdot \vec {\sigma} _ {a}, (4 6 \mathrm{a})
$$

$$
\tilde {B} = F _ {1} ^ {b} - F _ {0} ^ {b} + \left(F _ {0} ^ {b} + F _ {1} ^ {b} - 1\right) \vec {b} \cdot \vec {\sigma} _ {b}. \tag {46b}
$$

Therefore $S$ can be expressed as

$$
S = \operatorname{Tr} (\tilde {\mathcal {B}} \rho) \tag {47}
$$

via the modified Bell operator

$$
\tilde {\mathcal {B}} = \tilde {A} \tilde {B} - \tilde {A} \tilde {B} ^ {\prime} + \tilde {A} ^ {\prime} \tilde {B} + \tilde {A} ^ {\prime} \tilde {B} ^ {\prime}, \tag {48}
$$

where $\tilde{A}'$ and $\tilde{B}'$ are obtained from $\tilde{A}$ and $\tilde{B}$ by replacing $\vec{a}$ and $\vec{b}$ with $\vec{a}'$ and $\vec{b}'$ , respectively. Notice that $\tilde{B}$ is a Hermitian operator and therefore in some cases it is useful to think about the measurement of S as a measurement of a physical quantity corresponding to the operator $\tilde{B}$ (even though this analogy works only for averages).

It is rather trivial to show $^{50}$ that in the presence of local measurement errors the Cirel'son's inequality $|S| \leq 2\sqrt{2}$ remains valid (this fact can be proven $^{51}$ for any POVM-type measurement). Moreover, a stricter inequality for $|S|$ [see Eq. (50) below] can be obtained, using the method similar to that of Ref. 51. We will prove this inequality for all pure two-qubit states, $\rho = |\psi\rangle\langle\psi|$ , which automatically means that it is also valid for any mixed state $\rho$ . Using notation $\langle O\rangle = \mathrm{Tr}(O\rho) = \langle\psi|O|\psi\rangle$ for any operator O, we start with the obvious relation $|S| = |\langle\tilde{A}(\tilde{B} - \tilde{B}')\rangle + \langle\tilde{A}'(\tilde{B} + \tilde{B}')\rangle| \leq |\langle\tilde{A}(\tilde{B} - \tilde{B}')\rangle| + |\langle\tilde{A}'(\tilde{B} + \tilde{B}')\rangle|$ . The next step is to apply the general inequality $|\langle O_1O_2\rangle|^2 \leq \langle O_1O_1^\dagger\rangle\langle O_2^\dagger O_2\rangle$ to both terms in the sum (this inequality is the direct consequence of the Cauchy-Schwartz inequality $|\langle\psi_1|\psi_2\rangle|^2 \leq \langle\psi_1|\psi_1\rangle\langle\psi_2|\psi_2\rangle$ for the vectors $|\psi_1\rangle = O_1^\dagger|\psi\rangle$ and $|\psi_2\rangle = O_2|\psi\rangle$ ). In this way we obtain

$$
| S | \leq \sqrt {\langle \tilde {A} ^ {2} \rangle \langle (\tilde {B} - \tilde {B} ^ {\prime}) ^ {2} \rangle} + \sqrt {\langle \tilde {A} ^ {\prime 2} \rangle \langle (\tilde {B} + \tilde {B} ^ {\prime}) ^ {2} \rangle} (4 9)
$$

(notice that operators $\tilde{A}$ , $\tilde{A}'$ , $\tilde{B}$ , and $\tilde{B}'$ are Hermitian). As the next step, we notice that the eigenvalues of $\tilde{A}$ (as well as eigenvalues of $\tilde{A}'$ ) are $2F_1^a - 1$ and $1 - 2F_0^a$ , which follows from Eq. (46) and the fact that the eigenvalues of $\vec{a} \cdot \vec{\sigma}_a$ are $\pm 1$ . Therefore, $\langle \tilde{A}^2 \rangle \leq (2F_{\max}^a - 1)^2$ and $\langle \tilde{A}'^2 \rangle \leq (2F_{\max}^a - 1)^2$ , where $F_{\max}^k = \max(F_0^k, F_1^k)$ ; and so from Eq. (49) we obtain $|S| \leq (2F_{\max}^a - 1)\left[\sqrt{\langle (\tilde{B} - \tilde{B}')^2 \rangle} + \sqrt{\langle (\tilde{B} + \tilde{B}')^2 \rangle}\right]$ . Next, since $\sqrt{x_1} + \sqrt{x_2} \leq \sqrt{2(x_1 + x_2)}$ for any positive numbers $x_1$ and $x_2$ , and using the relation $\langle (\tilde{B} - \tilde{B}')^2 \rangle + \langle (\tilde{B} + \tilde{B}')^2 \rangle = 2\langle \tilde{B}^2 + \tilde{B}'^2 \rangle$ , we obtain the inequality $|S| \leq 2(2F_{\max}^a - 1)\sqrt{\langle \tilde{B}^2 + \tilde{B}'^2 \rangle}$ . Finally, using the relations $\langle \tilde{B}^2 \rangle \leq (2F_{\max}^b - 1)^2$ and $\langle \tilde{B}'^2 \rangle \leq (2F_{\max}^b - 1)^2$ , derived in a similar way as above, we obtain the upper bound

$$
| S | \leq 2 \sqrt {2} (2 F _ {\mathrm{max}} ^ {a} - 1) (2 F _ {\mathrm{max}} ^ {b} - 1). \tag {50}
$$

This upper bound is generally not exact and can be reached only in the case when the errors are symmetric in both qubits [Eq. (45)], leading to Eq. (55) below. While the bound (50) depends only on the largest measurement fidelity for each qubit, our numerical results show that the exact bounds $S_{\pm}$ shrink monotonously with the decrease of all fidelities, if the errors are small enough to allow the BI violation (see below).

A useful expression for S can be obtained from Eqs. (46)–(47) by separating the terms for the ideal case:

$$
S = 2 \xi_ {-} ^ {a} \xi_ {-} ^ {b} + 2 \xi_ {+} ^ {a} \xi_ {-} ^ {b} \vec {a} ^ {\prime} \cdot \vec {s} _ {a} + 2 \xi_ {-} ^ {a} \xi_ {+} ^ {b} \vec {b} \cdot \vec {s} _ {b} + \xi_ {+} ^ {a} \xi_ {+} ^ {b} S _ {0}, (5 1)
$$

where $\xi_{+}^{k}=F_{0}^{k}+F_{1}^{k}-1$ , $\xi_{-}^{k}=F_{1}^{k}-F_{0}^{k}$ , $S_{0}$ is the value of S in the absence of errors [Eq. (16)], and $\vec{s}_{k}$ is the Bloch vector characterizing the reduced density matrix for the qubit k, i.e. $\rho_{k}=\mathrm{Tr}_{k^{\prime}\neq k}\rho=(I+\vec{s}_{k}\cdot\vec{\sigma})/2$ . Notice that $\vec{s}_{a}=\vec{s}_{b}=0$ for a maximally entangled state and therefore the second and third terms in Eq. (51) may increase $|S|$ for nonmaximally entangled optimal states. That is why in the presence of errors the states maximizing and minimizing S are usually nonmaximally entangled $^{26}$ (see below).

Notice that in the presence of errors, S still preserves the invariance with respect to the local transformations of qubits and simultaneous rotation of measurement directions described by Eqs. (20). The symmetry described by Eq. (21) (sign flip of S for the reversal of one-qubit measurement directions) is no longer valid; however, it can be easily modified by adding simultaneous interchange $F_{0} \leftrightarrow F_{1}$ of one-qubit fidelities:

$$
S \rightarrow - S \text {if} \vec {a} \rightarrow - \vec {a}, \vec {a} ^ {\prime} \rightarrow - \vec {a} ^ {\prime}, F _ {0} ^ {a} \leftrightarrow F _ {1} ^ {a}; (5 2 a)
$$

$$
S \rightarrow - S \mathrm{if} \vec {b} \rightarrow - \vec {b}, \vec {b} ^ {\prime} \rightarrow - \vec {b} ^ {\prime}, F _ {0} ^ {b} \leftrightarrow F _ {1} ^ {b}. (5 2 \mathrm{b})
$$

Obviously, S does not change if both transformations (52) are made simultaneously. As a consequence, the maximum $S_{+}$ and minimum $S_{-}$ (optimized over the qubit states and over measurement directions) are invariant with respect to simultaneous interchange of measurement fidelities

$$
F _ {0} ^ {a} \leftrightarrow F _ {1} ^ {a}, \quad F _ {0} ^ {b} \leftrightarrow F _ {1} ^ {b}, \tag {53}
$$

while the extremum values change as $S_{+} \to -S_{-}$ , $S_{-} \to -S_{+}$ if only one-qubit fidelity interchange ( $F_0^a \leftrightarrow F_1^a$ or $F_0^b \leftrightarrow F_1^b$ ) is made. (The corresponding optimal states obviously do not change.)

In the presence of measurement errors the magnitudes of the maximum and minimum of S generally differ, $S_{+} \neq |S_{-}|$ . However, as follows from the latter symmetry, $S_{+} = |S_{-}|$ (as in the ideal case) if the two measurement fidelities are symmetric (equal) at least for one qubit:

$$
F _ {0} ^ {a} = F _ {1} ^ {a} \text { or } F _ {0} ^ {b} = F _ {1} ^ {b}. \tag {54}
$$

If the fidelities are symmetric for both qubits [the situation described by Eq. (45)], then the expression for S given by Eq. (51) becomes simple: $S = (2F_{a} - 1)(2F_{b} - 1)S_{0}$ and directly related to the value $S_{0}$ without measurement errors. Then the extremum values

$$
S _ {\pm} = \pm 2 \sqrt {2} (2 F _ {a} - 1) (2 F _ {b} - 1) \tag {55}
$$

are obviously achieved for any maximally entangled state under the same conditions as in Sec. III. Correspondingly, the requirement for the fidelities for a violation of the BI is $^{27,28}$

$$
(2 F _ {a} - 1) (2 F _ {b} - 1) > 2 ^ {- 1 / 2} \approx 0. 7 0 7. \tag {56}
$$

Notice that when the measurement fidelities for the both qubits are the same, $F_{a} = F_{b} = F$ , Eq. (56) reduces to the threshold fidelity

$$
F > 0. 5 + 2 ^ {- 5 / 4} \approx 0. 9 2 0, \tag {57}
$$

while if the measurement for one of the qubits is ideal (for example, $F_{a} = 1$ ), then the BI violation requires

$$
F _ {b} > 0. 5 + 2 ^ {- 3 / 2} \approx 0. 8 5 4. \tag {58}
$$

# C. Analytical results for maximally entangled states

Let us first analyze the extremum values of S for the class of maximally entangled states (25). Since in this case $\vec{s}_{a} = \vec{s}_{b} = 0$ , we obtain from Eq. (51) that for maximally entangled states

$$
S = 2 \left(F _ {1} ^ {a} - F _ {0} ^ {a}\right) \left(F _ {1} ^ {b} - F _ {0} ^ {b}\right) + \left(F _ {1} ^ {a} + F _ {0} ^ {a} - 1\right) \left(F _ {1} ^ {b} + F _ {0} ^ {b} - 1\right) S _ {0} \tag {59}
$$

is directly related to the corresponding quantity $S_{0}$ in the absence of errors. Therefore the extremum values of S for maximally entangles states are

$$
S _ {\pm} = 2 (F _ {1} ^ {a} - F _ {0} ^ {a}) (F _ {1} ^ {b} - F _ {0} ^ {b}) \pm 2 \sqrt {2} (F _ {1} ^ {a} + F _ {0} ^ {a} - 1) (F _ {1} ^ {b} + F _ {0} ^ {b} - 1), \tag {60}
$$

and they are achieved under the same conditions as discussed in Sec. III.

When the asymmetry of measurement fidelities is similar for both qubits ( $F_1^a > F_0^a$ and $F_1^b > F_0^b$ or both inequalities with “<” sign), the first term in Eq. (60) is positive, and therefore the BI $|S| \leq 2$ can be stronger violated for positive $S$ than for negative $S$ . Similarly, if the asymmetries are opposite (for example, $F_1^a > F_0^a$ and $F_1^b < F_0^b$ ), then it is easier to violate the BI for negative $S$ . If fidelities are symmetric at least for one qubit [Eq. (54)], then the first term in Eq. (60) vanishes, and therefore $S_{-} = -S_{+}$ , as discussed in the previous subsection.

If the fidelities are the same for both qubits, $F_0^a = F_0^b = F_0$ and $F_1^a = F_1^b = F_1$ , then the positive $S$ is preferable and

$$
S _ {+} = 2 (F _ {1} - F _ {0}) ^ {2} + 2 \sqrt {2} (F _ {1} + F _ {0} - 1) ^ {2} \tag {61}
$$

(see the dashed lines in Fig. 1). This value of $S_{+}$ reaches the maximum $2\sqrt{2}$ when both errors vanish ( $F_{0} = F_{1} = 1$ ) and decreases with the decrease of each fidelity in the interesting region $S_{+} > 2$ (more accurately, as long as $S_{+} > 4 - 2\sqrt{2} \approx 1.17$ ). The condition for the BI violation in this case is

$$
(F _ {1} - F _ {0}) ^ {2} + \sqrt {2} (F _ {1} + F _ {0} - 1) ^ {2} > 1. \tag {62}
$$

This threshold of the BI violation on the $F_{0}-F_{1}$ plane is shown by the lowest dashed line in Fig. 1. It is an arc of the ellipse (corresponding to $S_{+}=2$ ), which is symmetric with respect to the line $F_{0}=F_{1}$ and is centered at $F_{0}=F_{1}=0.5$ . However, as seen from Fig. 1, this threshold looks quite close to a straight line on the $F_{0}-F_{1}$ plane. Notice that in the case $F_{0}=1$ the threshold (62) reduces to the most well-known condition $^{4,36}$

$$
F _ {1} > 2 \sqrt {2} - 2 \approx 0. 8 2 8, \tag {63}
$$

while in the case of symmetric error, $F_0 = F_1 = F$ , we recover Eq. (57).

# D. Numerical results

To optimize the CHSH inequality violation in presence of the measurement errors over all two-qubit states, including non-maximally entangled states, we have used numerical calculations. The analysis has been performed in two different ways (with coinciding results). First, we have searched for the maximum violation by finding extrema of the eigenvalues of the modified Bell operator $\tilde{B}$ defined by Eq. (48). Since $\tilde{B}$ is a Hermitian operator and $S = \mathrm{Tr}(\tilde{\mathcal{B}}\rho)$ , for fixed measurement directions $(\vec{a}, \vec{a}', \vec{b}, \vec{b}')$ the maximum and minimum eigenvalues of $\tilde{B}$ are equal to the maximum and minimum values of S, optimized over the two-qubit states. Therefore, optimization of the eigenvalues over the measurement directions gives the extrema of S. Similar method has been previously used $^{26}$ for the case of identical local errors (44) with $F_{0} = 1$ , while we apply this method to our more general error model.

![](images/3843bbaddad87291cd8a8f236c68b90c2709a3363a7953fb915c1b7ee59d1038.jpg)

<details>
<summary>line</summary>

| F0    | F1    | S+   |
|-------|-------|------|
| 0.65  | 1.00  | 2.0  |
| 0.70  | 0.99  | 2.2  |
| 0.75  | 0.98  | 2.4  |
| 0.80  | 0.96  | 2.6  |
| 0.85  | 0.93  | 2.8  |
| 0.90  | 0.88  |      |
| 0.95  | 0.82  |      |
| 1.00  | 0.65  |      |
</details>

FIG. 1: Solid lines: contour plot of $S_{+}$ (the maximum quantum value of S) versus the measurement fidelities $F_{0}$ and $F_{1}$ (assumed equal for both qubits), optimized over all two-qubit states. The dashed lines show the result of $S_{+}$ maximization over the maximally-entangled states only [Eq. (61)]. The Bell (CHSH) inequality can be violated when $S_{+} > 2$ .

The numerical maximization (minimization) of the largest (smallest) eigenvalue of the Bell operator $\tilde{B}$ has been performed using the software package Mathematica. The full optimization should be over all four measurement directions $(\vec{a},\vec{a}^{\prime},\vec{b},\vec{b}^{\prime})$ , described by 8 angles total. However, because of the invariance of S under local transformations, it is sufficient to optimize $\tilde{B}$ over only two angles: the angle between $\vec{a}$ and $\vec{a}^{\prime}$ and the angle between $\vec{b}$ and $\vec{b}^{\prime}$ , while the other angles are kept fixed.

Our numerical results show that in general the optimal values of these two angles are different from each other; however, for equal fidelity matrices [Eq. (44)] these angles are equal, so that $\vec{a} \cdot \vec{a}' = \vec{b} \cdot \vec{b}'$ . This result has been obtained previously $^{26}$ for the special case $F_{0} = 1$ . Our numerical results also show that $S_{+} > |S_{-}|$ for positive values of the product $(F_{1}^{a} - F_{0}^{a})(F_{1}^{b} - F_{0}^{b})$ and $S_{+} < |S_{-}|$ when this product is negative, similar to the result for the maximally entangled states [see discussion after Eq. (60)].

We have checked that the numerical results for $S_{+}$ and $|S_{-}|$ obtained via optimization of the eigenvalues of the Bell operator $\tilde{B}$ coincide with the results (see Fig. 1) obtained by our second numerical method based on the direct optimization of S. The second method happened to be more efficient numerically; as another advantage, it provides the optimal measurement directions together with optimal values $S_{+}$ and $S_{-}$ , while the Bell-operator method gives only $S_{+}$ and $S_{-}$ .

In principle, direct optimization of $S$ (for fixed measurement fidelities) implies optimization over the two-

qubit density matrix and over 8 measurement directions. However, there is a simplification. It is obviously sufficient to consider only pure states, since probabilistic mixtures of pure states cannot extend the range of S. Moreover, it is sufficient to consider only the states of the form

$$
| \Psi \rangle = \cos (\beta / 2) | 1 0 \rangle + \sin (\beta / 2) | 0 1 \rangle , \tag {64}
$$

since any pure two-qubit states can be reduced to this form by local rotations of the qubits (which are equivalent to rotations of the measurement directions); this fact is a direct consequence of the Schmidt decomposition theorem. $^{2}$ The angle $\beta$ can be limited within the range $0 \leq \beta \leq \pi$ because the coefficients of the Schmidt decomposition are non-negative. This range can be further reduced to $0 \leq \beta \leq \pi/2$ since $\pi$ -rotation of both qubits about x-axis (or any horizontal axis) exchanges states $|10\rangle \leftrightarrow |01\rangle$ and therefore corresponds to the transformation $\beta \rightarrow \pi - \beta$ .

Our numerical optimization of S within the class of two-qubit states (64) has shown that for non-zero measurement errors (we considered $2/3 \leq F_{i}^{k} \leq 1$ ) the optimal measurement directions $(\vec{a}, \vec{a}', \vec{b}, \vec{b}')$ always lie in the same vertical plane [such configuration is described by Eq. (37)]. This vertical plane can be rotated by an arbitrary angle about z-axis [such rotation is equivalent to an overall phase factor in Eq. (64)]; therefore we can assume $\phi_{0} = 0$ in Eq. (37). Notice that for the state (64) the vectors $\vec{s}_{a}$ and $\vec{s}_{b}$ in Eq. (51) are along the z-axis, $\vec{s}_{a} = -\vec{s}_{b} = \vec{z}\cos\beta$ . These vectors are zero for the maximally entangled state ( $\beta = \pi/2$ ), then the vertical configuration is no longer preferential; however, the maximally entangled state is optimal only when there are no measurement errors.

For the state (64) and vertical configuration (37) of the detector axes the expression for $S$ has the form

$$
\begin{array}{l} S = 2 \xi_ {-} ^ {a} \xi_ {-} ^ {b} - \xi_ {+} ^ {a} \xi_ {+} ^ {b} (g - h \sin \beta) \\ + 2 \cos \beta (\xi_ {+} ^ {a} \xi_ {-} ^ {b} \cos \theta_ {a} ^ {\prime} - \xi_ {-} ^ {a} \xi_ {+} ^ {b} \cos \theta_ {b}), \tag {65} \\ \end{array}
$$

where

$$
\begin{array}{l} g = \cos \theta_ {a} \cos \theta_ {b} - \cos \theta_ {a} \cos \theta_ {b} ^ {\prime} + \cos \theta_ {a} ^ {\prime} \cos \theta_ {b} \\ + \cos \theta_ {a} ^ {\prime} \cos \theta_ {b} ^ {\prime}, \\ h = \sin \theta_ {a} \sin \theta_ {b} - \sin \theta_ {a} \sin \theta_ {b} ^ {\prime} + \sin \theta_ {a} ^ {\prime} \sin \theta_ {b} \\ + \sin \theta_ {a} ^ {\prime} \sin \theta_ {b} ^ {\prime}. \tag {66} \\ \end{array}
$$

The numerical maximization and minimization of S in this case involves optimization over 5 parameters: $\beta$ , $\theta_{a}$ , $\theta_{a}^{\prime}$ , $\theta_{b}$ , and $\theta_{b}^{\prime}$ . Nevertheless, in our calculations this procedure happened to be faster than optimization over only two parameters in the method based on the Bell operator eigenvalues (we used Mathematica in both methods).

The solid lines in Fig. 1 show the contour plot of maximum value $S_{+}$ on the plane $F_0 - F_1$ for the case when the measurement fidelities for two qubits are equal [Eq. (44); in this case $S_{+} \geq |S_{-}|$ ]. Notice that the line for

$S_{+} = 2$ ends at the points $^{26}$ $F_{0} = 1$ , $F_{1} = 2/3$ and $F_{0} = 2/3$ , $F_{1} = 1$ (strictly speaking, this line corresponds to $S_{+} = 2 + 0$ since S = 2 can be easily realized without entanglement). The dashed lines, which correspond to the optimization over the maximally entangled states only [Eq. (61)], coincide with the solid lines at the points $F_{0} = F_{1}$ , because in this case the optimum is achieved at the maximally entangled states, as follows from the discussion after Eq. (54). When $F_{0} \neq F_{1}$ , the use of non-maximally entangled states gives a wider range of measurement fidelities allowing the BI violation. However, as seen from Fig. 1, the difference between the solid and dashed lines significantly shrinks with the increase of $S_{+}$ , so that there is practically no benefit of using non-maximally entangled states for the BI violation stronger than $S_{+} > 2.4$ . Notice that the solid and dashed lines in Fig. 1 are symmetric about the line $F_{0} = F_{1}$ since the interchange $F_{0} \leftrightarrow F_{1}$ does not change $S_{+}$ , as was discussed after Eq. (52).

Numerical calculations show that in the case of equal fidelity matrices, Eq. (44), the optimal detector configurations for a nonmaximally entangled state (64) have a “tilted-X” shape: $\vec{a} = -\vec{b}'$ and $\vec{a}' = -\vec{b}$ . In this case the number of parameters to be optimized in (65) reduced from 5 to 3, significantly speeding up the numerical procedure.

Notice that each optimal configuration within the class of states (64) corresponds to a 6-dimensional manifold of optimal configurations, obtained by simultaneous local rotations of the measurement axes and the two-qubit state (see discussion in Sec. III B 2).

# V. DECOHERENCE

The detailed analysis of the effects of decoherence will be presented elsewhere. $^{52}$ In this sections we discuss only some results of this analysis, and also discuss the combined effect of local measurement errors and decoherence.

To study effects of decoherence we assume for simplicity that the qubit rotations are infinitely fast. Thus, we assume that after a fast preparation of a two-qubit state $\rho$ there is a decoherence during time t resulting in the state $\rho'$ , which is followed by fast measurement of $\rho'$ (including tomographic rotations). Now S is given by Eq. (16) (in the absence of errors) or (47) and (51) (in the presence of errors) where $\rho$ should be substituted by $\rho'$ . To obtain $\rho'$ we assume independent (local) decoherence of each qubit due to zero-temperature environment, described by the parameters $\gamma_{k} = \exp(-t/T_{1}^{k})$ and $\lambda_{k} = \exp(-t/T_{2}^{k})$ (here k = a, b) where $T_{1}^{k}$ and $T_{2}^{k}$ are the usual relaxation times for the qubit k ( $T_{2}^{k} \leq 2T_{1}^{k}$ ).

As the initial state we still assume the state of the form (64) (even though in presence of decoherence this state actually does not always provide $^{52}$ the extrema of S). It can be shown analytically $^{52}$ that in the absence of measurement errors the maximum violation of the BI for the state (64) can be achieved when the detector

axes lie in either a horizontal [Eq. (38)] or vertical [Eq. (37)] plane (any other detector configuration cannot give stronger violation). In the case of only population relaxation $T_{2}^{k}=2T_{1}^{k}$ the horizontal configuration is better, while in the case of only $T_{2}$ -effect $T_{1}^{k}=\infty$ the vertical configuration is better.

When local measurement errors are considered together with decoherence, the optimal detector configurations may be neither vertical nor horizontal. To elucidate this fact, note that in Eq. (51) with $\rho$ replaced by $\rho'$ the vectors $\vec{s}_{a}$ and $\vec{s}_{b}$ remain vertical in the presence of decoherence. As a result, when in the absence of errors the optimal configuration is horizontal, measurement errors may make the optimal detector axes to go out of the horizontal plane. Note, however, that for some parameter ranges the vertical and horizontal configurations are still optimal.

In numerical calculations we should optimize S over 8 parameters: $\beta$ and 7 detector angles (one of the angles $\phi$ can be fixed because of the invariance of S under identical rotations of the qubits around the z axis). We have performed such optimization for a few hundred parameter points, choosing the measurement fidelities $F_{i}^{k}$ and decoherence parameters $\gamma_{k}$ and $\lambda_{k}$ randomly from the range (0.7, 1). For many (more than half of) parameter points the optimal configuration was still found to be either vertical or (in much smaller number of cases) horizontal. Even when the optimal configuration was neither vertical nor horizontal, we found that restricting optimization to only the vertical and horizontal configurations gives a very good approximation of the extrema $S_{\pm}$ (within 0.01 for all calculated parameter points). Such restriction significantly speeds up the calculations, since we need to optimize over only 5 parameters instead of 8 parameters.

Assuming initial state (64) and replacing $\rho$ by $\rho'$ in Eq. (47) we obtain

$$
\begin{array}{l} S = 2 \xi_ {-} ^ {a} \xi_ {-} ^ {b} + \xi_ {+} ^ {a} \xi_ {+} ^ {b} \{[ 1 - \gamma_ {a} - \gamma_ {b} - (\gamma_ {a} - \gamma_ {b}) \cos \beta ] g \\ \left. + \lambda_ {a} \lambda_ {b} h \sin \beta \right\} + 2 \xi_ {+} ^ {a} \xi_ {-} ^ {b} \left(\gamma_ {a} + \gamma_ {a} \cos \beta - 1\right) \cos \theta_ {a} ^ {\prime} \\ + 2 \xi_ {-} ^ {a} \xi_ {+} ^ {b} (\gamma_ {b} - \gamma_ {b} \cos \beta - 1) \cos \theta_ {b}, \tag {67} \\ \end{array}
$$

when the detector axes are in a vertical plane $[g$ and $h$ are defined in Eq. (66)], and

$$
\begin{array}{l} S = 2 \xi_ {-} ^ {a} \xi_ {-} ^ {b} + \xi_ {+} ^ {a} \xi_ {+} ^ {b} \lambda_ {a} \lambda_ {b} \sin \beta [ \cos (\phi_ {a} - \phi_ {b}) \\ \left. - \cos (\phi_ {a} - \phi_ {b} ^ {\prime}) + \cos (\phi_ {a} ^ {\prime} - \phi_ {b}) + \cos (\phi_ {a} ^ {\prime} - \phi_ {b} ^ {\prime}) \right] (6 8) \\ \end{array}
$$

for a horizontal detector configuration.

To find the extrema of S within the class of vertical configurations, Eq. (67) should be numerically optimized over the parameter $\beta$ and four angles $\theta$ . The optimization of S within the class of horizontal configurations is much simpler, because the term in the square brackets in Eq. (68) can be optimized independently of $\beta$ . This optimization is exactly the same as in the ideal case [see Eqs. (39) and (40) with $\alpha = 0$ ], therefore the term in the square brackets has extrema $\pm2\sqrt{2}$ , and therefore the extrema of Eq. (68) are reached at $\beta = \pi / 2$ (maximally entangled state), thus yielding a rather simple formula

$$
S _ {\pm} = 2 \xi_ {-} ^ {a} \xi_ {-} ^ {b} \pm 2 \sqrt {2} \xi_ {+} ^ {a} \xi_ {+} ^ {b} \lambda_ {a} \lambda_ {b}, \tag {69}
$$

which depends only on the $T_{2}$ -relaxation and measurement fidelities.

![](images/2900a9dc5a54629187eed7b49e5598eeb46e9fcea289305d4988c7b3ed1c9da5.jpg)

<details>
<summary>line</summary>

| F0   | F1   |
|------|------|
| 0.85 | 1.00 |
| 0.90 | 0.95 |
| 0.95 | 0.90 |
| 1.00 | 0.85 |
</details>

FIG. 2: Contour plot of the maximum value $S_{+}$ versus the measurement fidelities $F_{0}$ and $F_{1}$ for the initial state (64) in presence of decoherence with $\gamma_{a} = \gamma_{b} = 0.96$ and $\lambda_{a} = \lambda_{b} = 0.94$ , in absence (solid lines) or presence (dashed lines) of the symmetric crosstalk with $p_{c} = 0.1$ . Solid and dashed lines coincide for $S_{+} = 2$ .

For the numerical example shown by solid lines in Fig. 2 we assume that decoherence is identical for the two qubits and choose $\gamma_{a} = \gamma_{b} = 0.96$ and $\lambda_{a} = \lambda_{b} = 0.94$ , that corresponds to realistically good values for phase qubits: $T_{1} \simeq 450$ ns, $T_{2} \simeq 300$ ns, and $t \simeq 20$ ns. We also assume identical errors for both qubits, Eq. (44), which implies $S_{+} \geq |S_{-}|$ (as in the absence of decoherence), so in Fig. 2 we show the contour plot only for $S_{+}$ . We have found that for these decoherence parameters the vertical detector configuration is better than any other configuration [assuming initial state (64)] for any measurement fidelities in the analyzed range ( $0.8 \leq F_{0,1} \leq 1$ ). Notice that for the assumed decoherence parameters $S_{+} = 2.50$ in the absence of measurement errors, and the BI violation requires F > 0.947 (for $F_{0} = F_{1} = F$ ), that should be compared to the threshold F > 0.920 in absence of decoherence.

Let us mention that the error model previously discussed in relation to the BI violation in the two-photon interference $^{53}$ can be shown to be formally equivalent to the special case of our model with pure dephasing $(T_{1}^{k}=\infty)$ and identical errors with $F_{0}=1$ . Then our quantities $F_{1}$ and $\lambda_{a}\lambda_{b}$ correspond, respectively, to the detector efficiency and visibility in Ref. 53. The case of pure dephasing in the absence of errors has been also con-

sidered in connection with the BI violation in mesoscopic conductors. $^{21}$

# VI. MEASUREMENT CROSSTALK

The nonidealities discussed above are common for many types of qubits. Now let us discuss a more specific type of error: the measurement crosstalk for Josephson phase qubits. $^{12,54}$ The crosstalk error originates from the fixed (capacitive) coupling between the qubits, which is still on in the process of measurement. The mechanism of the crosstalk is the following. $^{12}$ If the measurement outcome for the phase qubit a is “1”, then this qubit is physically switched to a highly excited state (outside of the qubit Hilbert space), and its dissipative oscillating evolution after the switching affects the qubit b. As a result, the extra excitation of the qubit b may lead to its erroneous switching in the process of measurement, so that instead of the measurement outcome “1,0” we may get “1,1” with some probability $p_{c}^{a}$ . Similarly, because of the crosstalk from the qubit b to the qubit a, we may obtain the measurement result “1,1” instead of “0,1” with some probability $p_{c}^{b}$ . The values of $p_{c}^{a}$ and $p_{c}^{b}$ significantly depend on the timing of the measurement pulses applied to the qubits. $^{12}$ If the qubit a is measured few nanoseconds earlier than the qubit b, then $p_{c}^{a} \gg p_{c}^{b}$ ; if the qubit b is measured first, then $p_{c}^{a} \ll p_{c}^{b}$ . In the case when the measurement pulses are practically simultaneous, the crosstalk probability becomes significantly lower and $p_{c}^{a} \approx p_{c}^{b}$ .

Let us model the crosstalk in the following simple way. Even though physically the crosstalk develops at the same time as the measurement process and its description is quite nontrivial, $^{54}$ we will assume (for simplicity) that the crosstalk effect happens after the “actual” measurement (characterized by measurement fidelities as in Sections IV and V), so that the only effect of the crosstalk is the change of the outcome “1,0” into “1,1” with probability $p_{c}^{a}$ and the change of the outcome “0,1” into “1,1” with probability $p_{c}^{b}$ . Moreover, we assume that the probabilities $p_{c}^{a}$ and $p_{c}^{b}$ do not depend on the measurement axes ( $\vec{a}$ , $\vec{b}$ , etc.).

Notice that the measurement crosstalk obviously violates the fundamental assumption of locality, on which the BI is based (so, strictly speaking the BI approach is not applicable in such situation). In this section we discuss the modification of the classical bound for S, taking the crosstalk into account (this bound is now model-dependent, in contrast to the usual BI), and we also analyze the effect of the crosstalk on the quantum result for $S_{\pm}$ . Besides that, we discuss a simple modification of the experimental procedure, which eliminates the effect of crosstalk by using only the “negative result” outcomes.

# A. Modified Bell (CHSH) inequality

First, let us briefly review the derivation of the CHSH inequality, presented in Ref. 32. In a local realistic theory

$$
S = \int s (\Lambda) F (\Lambda) d \Lambda , \tag {70}
$$

where $F(\Lambda)$ is a distribution of the hidden variable $\Lambda$ and

$$
\begin{array}{l} s (\Lambda) = A (\Lambda , \vec {a}) B (\Lambda , \vec {b}) - A (\Lambda , \vec {a}) B (\Lambda , \vec {b} ^ {\prime}) \\ + A (\Lambda , \vec {a} ^ {\prime}) B (\Lambda , \vec {b}) + A (\Lambda , \vec {a} ^ {\prime}) B (\Lambda , \vec {b} ^ {\prime}). \tag {71} \\ \end{array}
$$

Here the measurement outcomes $A(\Lambda, \vec{a})$ and $B(\Lambda, \vec{b})$ can take only the values $\pm1$ , depending on the hidden variable $\Lambda$ and the detector orientations for the qubits a and b. (Notice that in Sections III and IV the notation A and B has been used for operators; now they are classical quantities. Also notice that the outcome value -1 is associated with the result “0”.) It is easy to check that $s(\Lambda) = \pm2$ under the locality assumption: the result $A(\Lambda, \vec{a})$ does not depend on the orientations $\vec{b}$ of the qubit b and vice versa; similarly, $F(\Lambda)$ does not depend on $\vec{a}$ and $\vec{b}$ . After integration (70), this leads to the CHSH inequality (1).

In our model the measurement crosstalk cannot change the positive product of outcomes AB = 1; however, it changes AB = -1 into AB = 1 with the probability $p_{c}^{a}(\Lambda)$ if A = -B = 1 or with probability $p_{c}^{b}(\Lambda)$ if A = -B = -1. (The locality assumption is obviously violated, since the value of A now depends not only on $\Lambda$ and $\vec{a}$ but also on B and thus implicitly on $\vec{b}$ ; similarly for B.) Notice that we assume that the crosstalk probabilities $p_{c}^{a}$ and $p_{c}^{b}$ may in principle depend on $\Lambda$ (this is a slight generalization of the more natural $\Lambda$ -independent model for $p_{c}^{a}$ and $p_{c}^{b}$ ).

The random change from AB = -1 to AB = 1 due to the crosstalk leads to the modification of the CHSH inequality (1). Here we mention only the main points of the derivation of the modified CHSH inequality; the details are in the Appendix. We start with fixing $\Lambda$ and considering all possible changes of the quantity $s(\Lambda)$ in Eq. (71) due to the crosstalk, thus obtaining the new values of s together with their probabilities for each of 16 realizations of the vector $\mathcal{C} = (A, A', B, B')$ . It is easy to see that due to the crosstalk $s(\Lambda)$ can get the values $\pm4$ , which are outside of the limits $\pm2$ . Averaging the value $s(\Lambda)$ over the crosstalk scenarios and then maximizing and minimizing the result over 16 realizations of the vector C, we get

$$
- 2 + 4 \min \left\{p _ {c} ^ {a} (\Lambda), p _ {c} ^ {b} (\Lambda) \right\} \leq \langle s (\Lambda) \rangle \leq 2 + 2 \left| p _ {c} ^ {a} (\Lambda) - p _ {c} ^ {b} (\Lambda) \right|. \tag {72}
$$

Finally averaging this result over $\Lambda$ , we obtain the modified CHSH inequality:

$$
- 2 + 4 \min \{p _ {c} ^ {a}, p _ {c} ^ {b} \} \leq S \leq 2 + 2 | p _ {c} ^ {a} - p _ {c} ^ {b} |, \tag {73}
$$

which is the main result of this subsection.

Let us consider two special cases. For a symmetric crosstalk, $p_c^a = p_c^b = p_c$ , the inequality (73) becomes

$$
- 2 + 4 p _ {c} \leq S \leq 2, \tag {74}
$$

while for a fully asymmetric crosstalk, $p_c^b = 0$ , it becomes

$$
- 2 \leq S \leq 2 + 2 p _ {c} ^ {a}. \tag {75}
$$

(similarly for $p_c^a = 0$ ).

It is interesting to notice that the inequality in the symmetric case is more restrictive than the BI (1) (“easier” negative bound and no change of the positive bound). This is actually quite expected because in the limiting case $p_{c}=1$ the crosstalk makes all AB products equal 1, so that S=2 always, as also follows from Eq. (74). In contrast, for the fully asymmetric crosstalk the inequality (75) is less restrictive than (1) (“harder to violate” positive bound and no change of the negative bound). In the case of a finite crosstalk asymmetry, both the positive and negative bounds change [Eq. (73)].

Let us emphasize that in contrast to the derivation of the original CHSH inequality, $s(\Lambda)$ may be significantly outside of the range $(-2,2)$ ; it can get the values $s(\Lambda)=\pm4$ for any (symmetric or asymmetric) crosstalk. So the fact that the lower bound for S never decreases and the upper bound increases only slightly for small crosstalk, is due to the statistical averaging of the random increase and decrease of $s(\Lambda)$ due to the crosstalk.

# B. Quantum calculation of $S$

In the quantum case the crosstalk changes the measurement probabilities $p_{ij} \rightarrow p_{ij}^{C}$ as

$$
p _ {0 0} ^ {C} = p _ {0 0}, p _ {1 0} ^ {C} = (1 - p _ {c} ^ {a}) p _ {1 0}, p _ {0 1} ^ {C} = (1 - p _ {c} ^ {b}) p _ {0 1},
$$

$$
p _ {1 1} ^ {C} = p _ {1 1} + p _ {c} ^ {a} p _ {1 0} + p _ {c} ^ {b} p _ {0 1}. \tag {76}
$$

Then using the definitions (2) and (3) we obtain that the measured value of $S$ becomes

$$
S ^ {C} = 2 \tilde {p} _ {c} + (1 - \tilde {p} _ {c}) S + 2 (p _ {c} ^ {a} - p _ {c} ^ {b}) [ p _ {b} (\vec {b}) - p _ {a} (\vec {a} ^ {\prime}) ], (7 7)
$$

where $\tilde{p}_c = (p_c^a + p_c^b) / 2$ , while $S$ , $p_a(\vec{a}')$ , and $p_b(\vec{b})$ are the quantities obtained in the absence of crosstalk $[p_a(\vec{a}')$ and $p_b(\vec{b})$ are defined after Eq. (5)].

In a general case the maxima and minima $S_{\pm}^{C}$ of Eq. (77) can be found numerically. To estimate the effect of the crosstalk, let us calculate Eq. (77) for a maximally entangled state in the absence of errors and decoherence. Then $p_{a}(\vec{a}') = p_{b}(\vec{b}) = 1/2$ , and using $S_{\pm} = \pm 2\sqrt{2}$ we obtain

$$
S _ {+} ^ {C} = 2 \sqrt {2} - (2 \sqrt {2} - 2) \tilde {p} _ {c}, S _ {-} ^ {C} = - 2 \sqrt {2} + (2 \sqrt {2} + 2) \tilde {p} _ {c}. \tag {78}
$$

As we see, both extrema are affected by the crosstalk, making the range narrower from both sides; however, the lower boundary is affected much stronger than the upper boundary. Comparing Eq. (78) with the modified CHSH inequality (73), we see that the lower bound shifts up for the quantum result always faster than for the classical bound; therefore the gap between the quantum and classical bounds always shrinks due to the crosstalk. The classical-quantum gap at positive S shrinks from both sides due to the crosstalk.

In the case of symmetric crosstalk, $p_{c}^{a} = p_{c}^{b} = p_{c}$ , we can easily consider non-maximally entangled states, measurement errors and decoherence, since Eq. (77) in this case reduces to $S^{C} = 2p_{c} + (1 - p_{c})S$ . Therefore, $S_{\pm}^{C}$ are simply related to the values $S_{\pm}$ without crosstalk (but with measurement errors and decoherence):

$$
S _ {\pm} ^ {C} = 2 p _ {c} + (1 - p _ {c}) S _ {\pm} \tag {79}
$$

[similar dependence was used in Eq. (78)]. A violation of the upper bound of the modified CHSH inequality (74) can be observed when $S_{+}^{C}=2p_{c}+(1-p_{c})S_{+}>2$ , which yields $S_{+}>2$ , while a violation of the lower limit in Eq. (74) requires $S_{-}^{C}=2p_{c}+(1-p_{c})S_{-}<-2+4p_{c}$ , which yields $S_{-}<-2$ . Quite surprisingly, the symmetric crosstalk does not change the conditions for the BI violation. (Of course, the violation of the increased lower bound is not as convincing psychologically as the violation on the increased upper bound.)

Let us discuss the combined effect of local errors, decoherence, and symmetric crosstalk for the numerical example considered in Sec. V, assuming symmetric crosstalk with $p_{c}=0.1$ . Now for the state (64) in the absence of local measurement errors ( $F_{0}=F_{1}=1$ ) we get $S_{+}^{C}=2.45$ , which is slightly less than the value $S_{+}=2.50$ obtained in the absence of the crosstalk. The dependence of $S_{+}^{C}$ on the measurement fidelities $F_{0}$ and $F_{1}$ in this case is illustrated by the dashed lines in Fig. 2. In accordance with the above discussion, the solid line for $S_{+}=2$ and the dashed line for $S_{+}^{C}=2$ coincide (the BI violation boundary is not affected), while the comparison of the solid and dashed lines for $S_{+}=2.2$ and 2.4 shows that the crosstalk makes an observation of a given BI-violating value of $S_{+}$ more difficult.

Figure 3 shows a similar contour plot on the $F_{0}-F_{1}$ plane for the lowest quantum value $S_{-}^{C}$ assuming the same parameters as in Fig. 2. Comparing Figs. 2 and 3 we see that it is more difficult to violate the lower classical bound (even though it is now only -1.6) than the upper classical bound of 2. This is because we assumed the same measurement fidelities for both qubits, that generally shifts the quantum result up [as in Eq. (60)]. Notice that in Fig. 3 the reference bound of -2 can still be violated, though only for almost perfect fidelities.

# C. Elimination of the crosstalk effect

Slight modification of the CHSH inequality (1) can make it insensitive to the crosstalk. The main idea is to use only experimental outcomes with the result “0”, when a qubit does not switch, and therefore the crosstalk

![](images/9f7cb1fbea8143ba0af2c6091c73983453a54b769734142eb50caa7f0b22f9f2.jpg)

<details>
<summary>line</summary>

| F₀   | F₁   |
|------|------|
| 0.8  | 1.0  |
| 0.9  | 0.95 |
| 1.0  | 0.85 |
</details>

FIG. 3: Contour plot of the quantum minimum $S_{-}^{C}$ versus the measurement fidelities $F_{0}$ and $F_{1}$ (same for both qubits) optimized over the states (64). We assume symmetric crosstalk with $p_{c} = 0.1$ and decoherence parameters $\gamma_{a} = \gamma_{b} = 0.96$ and $\lambda_{a} = \lambda_{b} = 0.94$ (same as for Fig. 2). The classical bound is shifted from -2 to -1.6 by the crosstalk.

does not occur. Such “negative-result” (“null-result”) experiments with the Josephson phase qubits are very interesting from the quantum point of view; $^{37,55}$ however, here we are interested only in the classical consequence (or rather absence of it) for a measurement with a null result.

Instead of the inequality (1) let us use the equivalent inequality (6), and let us change the definition of T in Eq. (5) by interchanging the measurement outcomes “1” and “0”. Then by symmetry the inequality (6) is still valid, so we get the classical bounds

$$
- 1 \leq \tilde {T} \leq 0 \tag {80}
$$

for

$$
\begin{array}{l} \tilde {T} = p _ {0 0} (\vec {a}, \vec {b}) - p _ {0 0} (\vec {a}, \vec {b ^ {\prime}}) + p _ {0 0} (\vec {a ^ {\prime}}, \vec {b}) + p _ {0 0} (\vec {a ^ {\prime}}, \vec {b ^ {\prime}}) \\ - p _ {0} (\vec {a ^ {\prime}}) - p _ {0} (\vec {b}), \tag {81} \\ \end{array}
$$

where the probability $p_{0}(\vec{a'})$ is for measuring the qubit a only (without measuring the qubit b), while $p_{0}(\vec{b})$ is for measuring only the qubit b.

With this simple modification, the CHSH inequality becomes insensitive to the mechanism of the measurement crosstalk $^{12,54}$ considered in this section. Notice, however, that in performing experiment in this way it is still important to check the absence of a direct crosstalk (due to the measurement pulse itself). This can be done by applying a measurement pulse to the well-detuned qubit b (so that it cannot switch) and checking that this does not affect switching probabilities for the qubit a (and similarly for a interchanged with b).

# VII. CONCLUSION

In this paper we have considered the conditions for the violation of the Bell inequality in the CHSH form $^{4}$ for the entangled pairs of solid-state qubits, when instead of the rotation of optical polarizers (detectors) we have to rotate the states of two qubits before the measurement, which itself is always performed in the logical z-basis $(|0\rangle,|1\rangle)$ for each qubit. While most of our results are applicable to many types of qubits, we have focused on the experiments with the Josephson phase qubits. $^{49}$ We have analyzed the BI violation for the ideal case as well as in presence of various nonidealities, including local measurement errors, local decoherence, and measurement crosstalk.

In the ideal case the maximum violation of the BI $(S_{\pm} = \pm2\sqrt{2}, \text{while the classical bound is } |S| \leq 2)$ can be realized for any maximally entangled state. The optimal configuration of the measurement directions in this case can be realized with three degrees of freedom for each maximally-entangled state (the measurement direction in our terminology actually refers to the qubit rotation before the measurement). However, in presence of non-idealities there is typically less freedom in choosing the optimal configuration. For the “odd” two-qubit states involving superpositions (64) of the states $|01\rangle$ and $|10\rangle$ , we have focused on the “vertical” measurement configurations, for which all measurement directions $(\vec{a}, \vec{a}', \vec{b}, \vec{b}')$ are within the same vertical plane of the Bloch sphere, and the “horizontal” configuration, for which the four measurement axes are within the x-y plane.

The qubit measurement with finite local errors (characterized by the fidelities $F_{0}$ and $F_{1}$ for each qubit) shrinks the quantum range for S. We have found that for a maximally entangled state the BI violation is still possible when the classical bounds $\pm2$ are exceeded by the extrema of the quantum result given by Eq. (60). In particular, when two qubits have the same fidelities, the violation condition is given by Eq. (62) and shown by the lowest dashed line in Fig. 1; it can be crudely approximated by the condition $(F_{1} + F_{0})/2 > 0.92$ . A significantly softer violation condition can be obtained when allowing the two-qubit state to be non-maximally entangled; $^{26}$ this condition is shown by the lowest solid line in Fig. 1. However, the trick of using a non-maximally-entangled state does not help much when we need a BI violation with a significant margin (not just barely); this can be seen by comparing the solid and dashed lines in Fig. 1. For non-maximally entangled odd two-qubit states (64) in presence of local measurement errors, the vertical measurement configuration is found to be preferable in comparison with other configurations.

Analyzing the effect of local decoherence of the qubits for the odd two-qubit states (64), we have found that either vertical or horizontal configuration of the measurement directions is optimal, depending on the parameters. In particular, in the case of population (energy) relaxation in z-basis, the horizontal configuration is optimal, while for pure dephasing the vertical configuration

is optimal. In presence of both the decoherence and local measurement errors, the optimal configuration can be neither horizontal nor vertical; however, restricting optimization to only these two classes of configurations gives a very good approximation of the extrema $S_{\pm}$ . Obviously, both the decoherence and the measurement errors make the observation of the BI violation more difficult.

We have also analyzed the effect of the measurement crosstalk $^{12}$ which plays an important role in measurement of the capacitively-coupled phase qubits. Since the crosstalk is a mechanism of classical communication between the qubits, strictly speaking the BI is inapplicable. However, for a particular model of the crosstalk it is possible to derive a modified CHSH inequality [see Eq. (73)]. In particular, we have found that the symmetric crosstalk does not change the upper classical bound but increases the lower classical bound. The crosstalk also affects the quantum bounds, which are given by Eq. (78) for the maximally entangled state in the otherwise ideal case with arbitrary crosstalk and by Eq. (79) for an arbitrary case but assuming a symmetric crosstalk. Quite unexpectedly, the symmetric crosstalk does not change the threshold condition for the observation of the BI violation. However, the crosstalk always reduces the gap between the classical and quantum bounds and makes an observation of the BI violation with a finite margin more difficult. It is important to mention that the detrimental effect of the crosstalk can be eliminated by a slight change of the CHSH inequality (by using only negative-result outcomes), which makes it insensitive to the crosstalk [see Eqs. (80) and (81)].

We have performed the numerical simulations with the parameters similar to the experimental values for the best present-day experiments with the Josephson phase qubits. $^{17,38}$ Our results (see Fig. 2) show the possibility of the CHSH inequality violation with a significant margin even without further improvement of the phase qubit technology.

# Acknowledgments

We thank Qin Zhang, John Martinis, Nadav Katz, and Markus Ansmann for useful discussions. The work was supported by NSA and DTO under ARO grant W911NF-04-1-0204.

# APPENDIX: DERIVATION OF THE INEQUALITY (73)

To derive the inequality (72), we fix $\Lambda$ and use the abbreviated notation $s = s(\Lambda)$ , $p_{c}^{a} = p_{c}^{a}(\Lambda)$ , $p_{c}^{b} = p_{c}^{b}(\Lambda)$ , $A = A(\Lambda, \vec{a})$ , $A' = A(\Lambda, \vec{a}')$ and similarly for B and $B'$ . Let us introduce the vectors $\mathcal{C} = (A, A', B, B')$ and $\mathcal{A} = (AB', AB, A'B, A'B')$ , so that $s = \mathcal{A} \cdot (-1, 1, 1, 1)$ . The vector C can assume 16 values, whereas without the crosstalk A can take 8 values, since the number of pluses or minuses in A is even. Each pair of C and -C yields one value of A. Generally, the crosstalk effect differs for C and -C, except for the symmetric case, $p_{c}^{a} = p_{c}^{b} = p_{c}$ , when crosstalk depends only on A. We start the analysis with the symmetric case and then consider the general asymmetric case.

# 1. Symmetric crosstalk

It is easy to check that without crosstalk s = 2 or -2. To obtain the upper bound of the modified BI, we consider four values of A corresponding to s = 2. The crosstalk cannot change $\mathcal{A} = (1, 1, 1, 1)$ , hence we discuss three other values: $(-1, -1, 1, 1)$ , $(-1, 1, -1, 1)$ , and $(-1, 1, 1, -1)$ . For any of these vectors, the crosstalk makes s to take the values 0, 2, and 4 with the probabilities $p_{c}q_{c}$ , $q_{c}^{2} + p_{c}^{2}$ , and $p_{c}q_{c}$ , respectively, where $q_{c} = 1 - p_{c}$ . Indeed, the change of any -1 to 1 in A occurs with the probability $p_{c}q_{c}$ , yielding s = 0 for the change of the first -1 in A and s = 4 for the change of any other -1, whereas $q_{c}^{2} + p_{c}^{2}$ is the probability of no change or change to $\mathcal{A} = (1, 1, 1, 1)$ , both cases yielding s = 2. Thus, though now s can achieve the maximal mathematically possible value 4, it is easy to see that the maximal value for the average of s over crosstalk is still $\langle s \rangle_{\max} = 2$ .

To obtain the lower limit of the inequality, we consider four values of A corresponding to s = -2 in the absence of the crosstalk. Consider first $\mathcal{A} = (-1, -1, -1, -1)$ . This vector can be changed by the crosstalk to any of 16 possible combinations of four pluses and minuses, yielding the values s = -4, -2, 0, 2, 4 with the probabilities $p_{c}q_{c}^{3}$ , $q_{c}^{4} + 3p_{c}^{2}q_{c}^{2}$ , $3p_{c}q_{c}^{3} + 3p_{c}^{3}q_{c}$ , $3p_{c}^{2}q_{c}^{2} + p_{c}^{4}$ , $p_{c}^{3}q_{c}$ , respectively (note that the sum of the above probabilities equals 1). The above probabilities can be easily obtained if one takes into account that s = -4 results from the change of only the first -1 in A, s = -2 occurs when either A have not changed or the first and one of the last 3 components have changed, s = 0 occurs when the crosstalk results in A with the first and two other components equal to 1 or -1, s = 2 occurs when two of the last three components or all the components of A change the sign, and s = 4 results from the changes of all the last three components in A. As a result, in the case when only $\mathcal{A} = (-1, -1, -1, -1)$ is realized, we obtain

$$
\langle s \rangle = - 2 + 4 p _ {c}. \tag {A.1}
$$

Finally, let us consider the values $\mathcal{A} = (1, -1, -1, 1)$ , $(1, 1, -1, -1)$ , and $(1, -1, 1, -1)$ . For any of these vectors the crosstalk makes $s$ to take the values $-2$ , $0$ , $2$ with the probabilities $q_c^2$ , $2p_c q_c$ , and $p_c^2$ , respectively. These probabilities follow from the fact that $s = -2$ when $\mathcal{A}$ is not changed, $s = 0$ results from the change of only one component $-1$ , and $s = 2$ results from the change of the both negative components. It is easy to check that for any of the three above vectors, we again obtain Eq. (A.1).

Combining the results for the upper and lower bounds, we obtain the inequality $-2 + 4p_{c} \leq \langle s \rangle \leq 2$ , the average of which over $\Lambda$ yields the modified BI for the symmetric crosstalk given by Eq. (74).

# 2. Asymmetric crosstalk

As mentioned above, when $p_{c}^{a} \neq p_{c}^{b}$ , the crosstalk yields different results for C and -C. However, recalling that the change $AB = -1 \rightarrow 1$ occurs with the probability $p_{c}^{a}$ if A = 1 and B = -1 or with the probability $p_{c}^{b}$ if A = -1 and B = 1, we can obtain a symmetry relation for the probability of a change of A with a given C to some $A'$ as a function of $p_{c}^{a}$ and $p_{c}^{b}$ :

$$
P _ {\mathcal {A} \to \mathcal {A} ^ {\prime}} (p _ {c} ^ {a}, p _ {c} ^ {b}, \mathcal {C}) = P _ {\mathcal {A} \to \mathcal {A} ^ {\prime}} (p _ {c} ^ {b}, p _ {c} ^ {a}, - \mathcal {C}). \tag {A.2}
$$

To extend the results of Sec. 1 to the general case of asymmetric crosstalk, we should reconsider all possible changes of A discussed in Sec. 1 with the account of the two vectors C and -C corresponding to each A. As a result, the probability of each value of s obtained above is replaced by two probabilities, which differ from each other by the change $p_{c}^{a} \leftrightarrow p_{c}^{b}$ [cf. Eq. (A.2)].

It happens that for most combinations $\mathcal{A}$ the probability obtained in Sec. 1 is replaced by two probabilities just by replacing $p_c$ with $p_c^a$ or with $p_c^b$ . One of two exceptions is the case $\mathcal{A} = (-1,1,-1,1)$ . Then for $\mathcal{C} = (-1,1,-1,1)$ the quantity $s = 2$ changes to $s = 0,2,4$ with the probabilities $q_c^a p_c^b$ , $q_c^a q_c^b + p_c^a p_c^b$ , and $p_c^a q_c^b$ , respectively, where $q_c^k = 1 - p_c^k$ . This yields $\langle s\rangle = 2 + 2(p_c^a - p_c^b)$ . The other value of $\langle s\rangle$ [corresponding to $\mathcal{C} = (1,-1,1,-1)$ ] is obtained from this formula by substitution $p_c^a \leftrightarrow p_c^b$ , so that $\langle s\rangle = 2 + 2(p_c^b - p_c^a)$ . The other exception is $\mathcal{A} = (1,-1,1,-1)$ . In this case we obtain $\langle s\rangle = -2 + 2(p_c^a + p_c^b)$ (for both possible values of $\mathcal{C}$ ). Finally, by choosing the worst cases for the lower and upper bounds for $\langle s\rangle$ and averaging the result over $\Lambda$ , we obtain Eq. (73).

$^{1}$ A. Einstein, B. Podolsky, and N. Rosen, Phys. Rev. 47, 777 (1935).   
$^{2}$ M. A. Nielsen and I. L. Chuang, Quantum Computation and Quantum Information (Cambridge University Press, Cambridge, 2000).   
$^{3}$ J. S. Bell, Physics 1, 195 (1964).   
$^{4}$ J. F. Clauser, M. A. Horne, A. Shimony, and R. A. Holt, Phys. Rev. Lett. 23, 880 (1969).   
$^{5}$ S. J. Freedman and J. F. Clauser, Phys. Rev. Lett. 28, 938 (1972); E. S. Fry and R. C. Thompson, ibid. 37, 465 (1976).   
$^{6}$ A. Aspect, P. Grangier, and G. Roger, Phys. Rev. Lett. 47, 460 (1981); ibid. 49, 91 (1982); A. Aspect, J. Dalibard, and G. Roger, ibid. 49, 1804 (1982).   
$^{7}$ Z. Y. Ou and L. Mandel, Phys. Rev. Lett. 61, 50 (1988); Y. H. Shih and C. O. Alley, ibid. 61, 2921 (1988); P. R. Tapster, J. G. Rarity, and P. C. M. Owens, ibid. 73, 1923 (1994); P. G. Kwiat, K. Mattle, H. Weinfurter, A. Zeilinger, A. V. Sergienko, and Y. Shih, ibid. 75, 4337 (1995); W. Tittel, J. Brendel, H. Zbinden, and N. Gisin, ibid. 81, 3563 (1998); G. Weihs, T. Jennewein, C. Simon, H. Weinfurter, and A. Zeilinger, ibid. 81, 5039 (1998).   
$^{8}$ M. A. Rowe, D. Kielpinski, V. Meyer, C. A. Sackett, W. M. Itano, C. Monroe, and D. J. Wineland, Nature 409, 791 (2001).   
$^{9}$ D. L. Moehring, M. J. Madsen, B. B. Blinov, and C. Monroe, Phys. Rev. Lett. 93, 090410 (2004).   
$^{10}$ Y. Hasegawa, R. Loidl, G. Badurek, M. Baron, and H. Rauch, Nature (London) 425, 45 (2003).   
$^{11}$ Yu. A. Pashkin, T. Yamamoto, O. Astafiev, Y. Nakamura, D. V. Averin, and J. S. Tsai, Nature 421, 823 (2003).   
$^{12}$ R. McDermott, R. W. Simmonds, M. Steffen, K. B. Cooper, K. Cicak, K. D. Osborn, S. Oh, D. P. Pappas, and J. M. Martinis, Science 307, 1299 (2005).   
$^{13}$ J. H. Plantenberg, P. C. de Groot, G. J. P. M. Harmans, and J. E. Mooij, Nature 447, 836 (2007).   
$^{14}$ A. Wallraff, D. I. Schuster, A. Blais, L. Frunzio, J. Majer,

M. H. Devoret, S. M. Girvin, and R. J. Schoelkopf, Phys. Rev. Lett. 95, 060501 (2005).   
$^{15}$ J. R. Petta, A. C. Johnson, J. M. Taylor, E. A. Laird, A. Yacoby, M. D. Lukin, C. M. Marcus, M. P. Hanson, and A. C. Gossard, Science 309, 2180 (2005).   
$^{16}$ M. Ansmann, R. Bialczak, N. Katz, E. Lucero, R. McDermott, M. Neeley, M. Steffen, E. M. Weig, A. N. Cleland, and J. M. Martinis, Bulletin of APS 51, Abstract P40.00003 (2006).   
$^{17}$ M. Ansmann, R. Bialczak, N. Katz, E. Lucero, R. McDermott, M. Neeley, A. D. O'Connell, M. Steffen, E. Weig, A. Cleland, and J. M. Martinis, Bulletin of APS 52, Abstract L33.00005 (2007).   
$^{18}$ L. F. Wei, Yu-xi Liu, and F. Nori, Phys. Rev. B 72, 104516 (2005).   
$^{19}$ L. F. Wei, Yu-xi Liu, M. J. Storcz, and F. Nori, Phys. Rev. A 73, 052307 (2006).   
$^{20}$ R. Ionicioiu, P. Zanardi, and F. Rossi, Phys. Rev. A 63, 050101(R) (2001).   
$^{21}$ P. Samuelsson, E. V. Sukhorukov, and M. Buttiker, Phys. Rev. Lett. 91, 157002 (2003).   
$^{22}$ C. W. J. Beenakker, C. Emary, M. Kindermann, J. L. van Velsen, Phys. Rev. Lett. 91, 147901 (2003).   
$^{23}$ A. V. Lebedev, G. Blatter, C. W. J. Beenakker, and G. B. Lesovik, Phys. Rev. B 69, 235312 (2004).   
$^{24}$ P. Samuelsson and M. Buttiker, Phys. Rev. B 71, 245317 (2005).   
$^{25}$ B. Trauzettel, A. N. Jordan, C. W. J. Beenakker, and M. Buttiker, Phys. Rev. B 73, 235331 (2006).   
$^{26}$ P. H. Eberhard, Phys. Rev. A 47, R747 (1993).   
$^{27}$ P. G. Kwiat, A. M. Steinberg, and R. Y. Chiao, Phys. Rev. A 47, R2472 (1993).   
$^{28}$ M. Zukowski, A. Zeilinger, M. A. Horne, and A. K. Ekert, Phys. Rev. Lett. 71, 4287 (1993).   
$^{29}$ J.-A. Larsson and J. Semitecolos, Phys. Rev. A 63, 022117 (2001).   
$^{30}$ A. Cabello and J.-A. Larsson, Phys. Rev. Lett. 98, 220402

(2007); N. Brunner, N. Gisin, V. Scarani, and C. Simon, ibid. 98, 220403 (2007).   
$^{31}$ J. F. Clauser and A. Shimony, Rep. Prog. Phys. 41, 1881 (1978).   
$^{32}$ A. Aspect, in Quantum [Un]speakables – From Bell to Quantum Information, ed. by R. A. Bertlmann and A. Zeilinger (Springer, 2002).   
$^{33}$ J. S. Bell, in Foundations of Quantum Mechanics, ed. B. d'Espagnat (Academic, NY, 1971), p. 171.   
$^{34}$ J. F. Clauser and M. A. Horne, Phys. Rev. D 10, 526 (1974).   
$^{35}$ P. M. Pearle, Phys. Rev. D 2, 1418 (1970).   
$^{36}$ A. Garg and N. D. Mermin, Phys. Rev. D 35, 3831 (1987).   
$^{37}$ N. Katz, M. Ansmann, R. C. Bialczak, E. Lucero, R. McDermott, M. Neeley, M. Steffen, E. M. Weig, A. N. Cleland, J. M. Martinis, and A.N. Korotkov, Science 312, 1498 (2006).   
$^{38}$ M. Steffen, M. Ansmann, R. C. Bialczak, N. Katz, E. Lucero, R. McDermott, M. Neeley, E. M. Weig, A. N. Cleland, and J. M. Martinis, Science 313, 1423 (2006).   
$^{39}$ L. D. Landau and E. M. Lifshitz, Quantum Mechanics: Non-Relativistic Theory (Pergamon Press, Oxford, 1977).   
$^{40}$ S. L. Braunstein, A. Mann, and M. Revzen, Phys. Rev. Lett. 68, 3259 (1992).   
$^{41}$ B. S. Cirel'son, Lett. Math. Phys. 4, 93 (1980).   
$^{42}$ L. J. Landau, Phys. Lett. A 120, 54 (1987).   
$^{43}$ S. Popescu and D. Rohrlich, Phys. Lett. A 169, 411 (1992).   
$^{44}$ G. Kar, Phys. Lett. A 204, 99 (1995).   
$^{45}$ S. Popescu and D. Rohrlich, Phys. Lett. A 166, 293 (1992).   
$^{46}$ This parametrization is performed as follows. In view of Eq. (27), any maximizing detector configuration is determined by $\vec{a}$ and $\vec{a}'$ , and hence can be equivalently described by a right-handed system of axes $(\vec{X},\vec{Y},\vec{Z})$ such that $\vec{Z}=\vec{a}$ , $\vec{X}=\vec{a}'$ , $\vec{Y}=\vec{a}\times\vec{a}'$ . This configuration can be obtained from the standard configuration (29)–(30) (with

the corresponding frame $\vec{X}_0 = \vec{x}$ , $\vec{Y}_0 = \vec{y}$ , and $\vec{Z}_0 = \vec{z}$ by the axes rotation described by the Euler angles $\kappa_1, \kappa_2$ , and $\kappa_3$ . This rotation is performed in three stages: (i) the rotation around $\vec{Z}_0$ by $\kappa_1$ , (ii) the rotation around the new position of $\vec{Y}_0$ by $\kappa_2$ , (iii) the rotation around $\vec{Z}$ by $\kappa_3$ .   
$^{47}$ K. G. H. Vollbrecht and R. F. Werner, J. Math. Phys. 41, 6772 (2000).   
$^{48}$ These transformations are equivalent to the rotations described by the Euler angles $(\kappa_{1},\kappa_{2},\kappa_{3})=(\phi_{0},C,0)$ and $(\phi_{0}+\pi,-C,0)$ where $\phi_{0}$ and C are arbitrary angles.   
$^{49}$ K. B. Cooper, M. Steffen, R. McDermott, R. W. Simmonds, S. Oh, D. A. Hite, D. P. Pappas, and J. M. Martinis, Phys. Rev. Lett. 93, 180401 (2004).   
$^{50}$ Our measurement error model for the qubit k can be interpreted as the exchange of the measurement results $0 \leftrightarrow 1$ with probability $1 - \max(F_{0}^{k}, F_{1}^{k})$ and the predetermined result with probability $|F_{0}^{k} - F_{1}^{k}|$ (result 0 if $F_{0}^{k} > F_{1}^{k}$ or 1 if $F_{0}^{k} > F_{1}^{k}$ ), while the “correct” operation is realized with the remaining probability $\min(F_{0}^{k}, F_{1}^{k})$ . Since the bound $|S| \leq 2\sqrt{2}$ obviously remains valid in all the cases when one or both qubits operate in the mode of result exchange or in the mode of predetermined result, and since the erroneous operation modes are independent of the correct results, the averaging over these modes (for each qubit) cannot invalidate the bound $|S| \leq 2\sqrt{2}$ .   
$^{51}$ D. Dieks, Phys. Rev. A 66, 062104 (2002).   
$^{52}$ A. G. Kofman and A. N. Korotkov, in preparation.   
$^{53}$ M. Pavičić, Opt. Commun. 142, 308 (1997).   
$^{54}$ A. G. Kofman, Q. Zhang, J. M. Martinis, and A. N. Korotkov, Phys. Rev. B 75, 014524 (2007).   
$^{55}$ R. Ruskov, A. Mizel, and A. N. Korotkov, Phys. Rev. B 75, 220501(R) (2007); L. P. Pryadko and A. N. Korotkov, Phys. Rev. B 76, 100503(R) (2007).