# Pole and zero edge state invariant for 1D non-Hermitian sublattice symmetry

Janet Zhong, $^{1}$ Heming Wang, $^{2}$ and Shanhui Fan $^{1,2,*}$

$^{1}$ Department of Applied Physics, Stanford University, Stanford, California 94305, USA

$^{2}$ Department of Electrical Engineering, Ginzton Laboratory,

Stanford University, Stanford, California 94305, USA

(Dated: December 19, 2024)

There have been several criteria for the existence of topological edge states in 1D non-Hermitian two-band sublattice-symmetric tight-binding Hamiltonians. The generalized Brillouin zone (GBZ) approach uses the integration of the Berry connection over the GBZ contour in the complex wavevector space. An alternate 'pole-zero' approach uses algebraic properties of the off-diagonal matrix elements of the sublattice-symmetric Hamiltonian in off-diagonal form. Both correctly predict the presence or absence of edge states, but there has not been an explicit proof of their equivalence.

Here we provide such an explicit proof and moreover we extend the pole-zero approach so that it also applies for sublattice-symmetric models when the Hamiltonian is not in off-diagonal form. We give numerical examples for these invariants.

# I. INTRODUCTION

There has been substantial recent interest on the topology of band structures [1, 2]. For Hermitian systems, the notion of bulk-edge correspondence states that the presence in a bulk system of a nonzero topological invariant, involving the integration of the Berry connection over the Brillouin zone, indicates the existences of topological edge states. This notion has been generalized to non-Hermitian systems [3-6], where the integration needs to be performed over the generalized Brillouin zone (GBZ) [7, 8]. In this paper, we refer to this approach as the GBZ approach.

As an alternative criterion for the existence of topological edge state, Lee and Thomale [9] considered a specific class of one-dimensional two-band model having sublattice symmetry. For this class of model, they consider the complex functions that describe the $z$ -dependency of the off-diagonal matrix elements of the Bloch Hamiltonian, where $z = e^{ik}$ with $k$ being the wavevector. They show that the number of topological edge states is related to the poles and zeros of these complex functions. In this paper, we refer to this approach as the pole-zero approach.

For non-Hermitian systems, the computation of the GBZ in general can be quite involved. Thus, the pole-zero approach of Ref. [9] is interesting because no explicit computation of the GBZ is necessary. This is in contrast with several related works [10, 11] which also relate the topological edge states to the behaviors of certain poles and zeros, but still require the knowledge of the GBZ in order to establish the criterion for the topological edge states. Both the GBZ approach and the pole-zero approach correctly describe the number of topological edge states and therefore must be equivalent to each other. However, there has not been a direct mathematical proof that establishes the equivalence of these two approaches.

In this paper, we provide an explicit proof of the equivalence of the GBZ approach and the pole-zero approach for the description of the topological edge states. We use a Riemann-sphere interpretation of the eigenstates which provides some physical motivation for the pole-zero approach. We also generalize the pole-zero approach to a broader set of Hamiltonians with sublattice symmetry, such as SSH-Creutz models [12-14], by a modification of the formalism. Our results clarify certain theoretical aspects related to bulk-edge correspondence of edge states in non-Hermitian systems.

Besides the GBZ and pole zero approach, some other approaches [4] use the delocalization transition of the biorthogonal polarization [15], doubled Green's functions [16] and real space wave-functions [17]. Non-Hermitian topological edge states have since been experimentally demonstrated in photonics [18, 19], electric circuits [20-22], metamaterials [23, 24], quantum optics [25] and more.

Thus, as the 1D non-Hermitian sublattice symmetric model is arguably the simplest case exhibiting non-Hermitian topological zero-energy edge states, a deeper understanding of the topological invariants they correspond to may lead to new avenues in the interplay of edge states with other exotic phenomena [26-48] or more complex experimental systems.

The paper is organized as follows. In Sec. II, we summarize the main theoretical formulas and then introduce eigenvalue and eigenvector topology on a Riemann surface. We show the equivalence of the GBZ edge-state invariant and pole-zero invariant for one-dimensional, two-band, tight-binding models with sublattice symmetry where the Bloch matrix is in the off-diagonal form, i.e. the diagonal matrix elements are all zero.

We then extend the formalism of the pole-zero approach so that the invariant applies for sublattice symmetric two-band models where the Bloch matrix is not in the off-diagonal form. In Sec. III, we numerically demonstrate our pole-zero invariant and the GBZ invariant for a Hermitian SSH model, a generalized non-Hermitian SSH model with $t_3$ hoppings, a longer-range SSH with hopping range across three unit cells and a Hermitian and non-Hermitian SSH-

arXiv:2410.11257v2 [cond-mat.mes-hall] 18 Dec 2024

\* shanhui@stanford.edu

Creutz model. We conclude in Sec. IV.

# II. THEORY

# A. Theoretical setup and sublattice symmetry

In this paper we consider two-band Hamiltonians, which can be written in the wavevector space as:

$$
H (z) = \sum_ {i = o, x, y, z} d _ {i} (z) \sigma_ {i} \tag {1}
$$

where $\sigma_{i}$ are Pauli matrices and the $d_{i}$ 's are in general complex. The variable $z = e^{ik}$ is the phase factor with $k$ being the wavevector which is generally complex. Note that phase factor $z$ here differs from the subscript $z$ in $\sigma_{z}$ and $d_{z}$ . The right eigenvalue equation for this Hamiltonian then reads:

$$
\left[ \begin{array}{l l} d _ {o} (z) + d _ {z} (z) & d _ {x} (z) - i d _ {y} (z) \\ d _ {x} (z) + i d _ {y} (z) & d _ {o} (z) - d _ {z} (z) \end{array} \right] \left[ \begin{array}{l} v _ {1} ^ {R} \\ v _ {2} ^ {R} \end{array} \right] = E (z) \left[ \begin{array}{l} v _ {1} ^ {R} \\ v _ {2} ^ {R} \end{array} \right] \tag {2}
$$

where $E(z)$ is the energy eigenvalue which has dispersion relation of

$$
E _ {\pm} (z) = d _ {o} (z) \pm \sqrt {d _ {x} ^ {2} (z) + d _ {y} ^ {2} (z) + d _ {z} ^ {2} (z)} \qquad (3)
$$

as can be obtained from the characteristic polynomial of $H(z)$ , i.e. $\operatorname{det}(H - E\mathbb{I}) = 0$ .

We now consider a sublattice-symmetric Hamiltonian [7, 9, 49]. A Hamiltonian has a sublattice symmetry when it satisfies [50, 51]

$$
\Gamma H (z) \Gamma^ {- 1} = \Gamma H (z) \Gamma^ {\dagger} = - H (z), \tag {4}
$$

where $\Gamma$ is a unitary and Hermitian matrix [51, 52]. Without loss of generality, we choose $\Gamma = \sigma_z$ . From Eq. (4), this implies that $d_{o}(z) = -d_{o}(z)$ and $d_{z}(z) = -d_{z}(z)$ , which then implies $d_{o}(z) = 0$ and $d_{z}(z) = 0$ . Thus, $H(z)$ can be written in the off-diagonal form:

$$
H (z) = \left[ \begin{array}{c c} 0 & H _ {a b} (z) \\ H _ {b a} (z) & 0 \end{array} \right] \tag {5}
$$

The Hamiltonian of the form of Eq. (5) is widely used for the study of bulk-edge correspondence in non-Hermitian [7-9, 15] and Hermitian [2] systems.

We note that the sublattice symmetry we consider in this paper differs from the non-Hermitian chiral symmetry as defined by [51]:

$$
\Gamma H ^ {\dagger} (z) \Gamma^ {- 1} = \Gamma H ^ {\dagger} (z) \Gamma^ {\dagger} = - H (z). \qquad (6)
$$

To contrast the two symmetries, again choose $\Gamma = \sigma_z$ . Then from Eq. (6), we get the constraints $d_{o}(z) = -d_{o}^{*}(z)$ and $d_{z}(z) = -d_{z}^{*}(z)$ which implies $d_{o}(z)$ and $d_{z}(z)$ are imaginary but not necessarily zero. Thus, in non-Hermitian systems, chiral symmetry does not imply th
at the Hamiltonian takes the off-diagonal form. As a caveat,

the notation about various symmetries has not been standardized in the literature. A number of papers refer to the sublattice symmetry as defined by Eq. (4) as chiral symmetry [7, 15, 35, 53]. Here we follow the notation of Ref. [51].

For subsequent use, we define the eigenvector ratio $M(z) = v_{1}^{R}(z) / v_{2}^{R}(z)$ . From Eq. (2), we have:

$$
M (z) = \frac {d _ {x} (z) - i d _ {y} (z)}{E (z) - d _ {o} (z) - d _ {z} (z)} = \frac {E (z) - d _ {o} (z) + d _ {z} (z)}{d _ {x} (z) + i d _ {y} (z)}. \tag {7}
$$

Using $M(z)$ allows us to study the eigenvector topology using a single complex number $M(z)$ rather than the two-component Bloch vector.

# B. Summary of main theoretical results

When a real-space lattice as described by the Hamiltonian of Eq. (1) is truncated, it may support topological edge states. Our paper concerns the theoretical criterion for the existence of edge states as deduced from topological properties derived from the bulk eigenstates. For non-Hermitian systems, the bulk eigenstates we use are the open-boundary solutions when the number of the unit cells $N\rightarrow \infty$ . These eigenstates are characterized by $z$ -values that form the generalized Brillouin zone (GBZ) [7, 8].

The GBZ can be obtained by analyzing the characteristic equation $\operatorname *{det}(H(z) - E\mathbb{I}) = 0$ and defines a contour on the complex $z$ -plane, $\mathcal{C}_{\mathrm{gbz}}$ (see Sec. IIC for more details). The GBZ invariant for the bulk-edge correspondence of a two-band Hamiltonian in the off-diagonal form is [7, 8, 11]:

$$
W _ {\mathrm {g b z}} \equiv \oint_ {\mathcal {C} _ {\mathrm {g b z}}} \frac {1}{2 \pi} \frac {d}{d z} \arg (M ^ {2} (z)) d z \tag {8}
$$

where $W_{\mathrm{gbz}}$ is a winding number. Though written in a different way by using $M(z)$ , Eq. (8) is equivalent to twice the non-Bloch winding number [7, 8] (see Sec. II C). It was shown in Ref. [7, 8, 10, 11] that $|W_{\mathrm{gbz}}|$ is equal to the number of topological edge states.

Instead of Eq. (8), Ref. [9] proposed an alternative way to obtain the number of topological edge states using algebraic properties involving the roots of the matrix elements of $H_{ab}(z)$ and $H_{ba}(z)$ . We restate the theorem from Ref. [9] in a different way, but we prove that our statement implies the statement from Ref. [9] in Appendix. VB. Note that $H_{ab}(z)$ and $H_{ba}(z)$ are Laurent polynomials in $z$ . Let $m > 0$ be the highest negative power of $z$ in $H_{ab}(z)$ , let $n > 0$ be the highest negative power of $z$ in $H_{ba}(z)$ and let $\mu = \max \{m,n\}$ . $\mu$ is then the range of unit cells (not sublattice sites) the longest hopping term reaches across. Then the number of edge states is equal to the absolute value of the following pole-zero invariant:

$$
W _ {\mathrm {p z}} = \# M _ {\text {z e r o s}} ^ {2} - \# M _ {\text {p o l e s}} ^ {2} \tag {9}
$$

2

for the first $2\mu$ zeros and poles of $M^2 (z)$ where the zeros and poles are sorted in increasing order of the $|z|$ magnitude at which they occur. Note that we count each zero and pole according to its multiplicity.

The benefit of using poles and zeros of $M^2(z)$ in Eq. (9) is that it can be easier to calculate than the GBZ. The main result of this paper that we will prove

$$
W _ {\mathrm {p z}} = W _ {\mathrm {g b z}}. \tag {10}
$$

We do not try to verify or prove Eq. (9) from first principles as this is done in Ref. [9]. Instead, the goal of this paper is to show the equivalence of Ref. [9] to Ref. [7, 8] (Sec. II C). By bridging these two previously independently defined topological edge state invariants, we hope analytical insights from each invariant can now be connected, resulting in a deeper overall understanding of topological edge states in 1D non-Hermitian sublattice symmetry. Our work goes beyond Ref. [7-9] by extending the pole-zero invariant and an interpretation of the GBZ invariant on the Riemann sphere for sublattice symmetric two-band models not in off-diagonal form (Sec. II E).

# C. Proof of equivalence of GBZ and pole-zero invariant

Consider an arbitrary one-dimensional tight-binding model with $d$ sublattice sites in a unit cell. The characteristic polynomial is:

$$
\det (H (z) - E (z) \mathbb {I}) = 0. \tag {11}
$$

Eq. (11) is a polynomial of degree $d$ in $E$ and a Laurent polynomial in $z$ with exponents $z^l$ where $-p \leq l \leq q$ . Here $p$ and $q$ are given by the hopping range to the left and right directions, and the choices of coupling parameters [50, 54]. By definition, $p$ is also the magnitude of the largest negative exponent of $z$ in the characteristics polynomial as defined above. There are $p + q$ solutions for $z$ at every $E$ , and we number them in increasing order of magnitude $|z_1| \leq |z_2| \leq \ldots |z_{p+q}|$ . Then the GBZ are the $z_p$ and $z_{p+1}$ solutions for Eq. (11) when $|z_p| = |z_{p+1}|$ . For more details and more general cases, see Ref. [50].

Now let us restrict to a two-band model so that $d = 2$ . For the rest of this section, we further assume that the Hamiltonian has sublattice symmetry and is already in the off-diagonal form. Then the characteristic polynomial in Eq. (11) gives

$$
E ^ {2} (z) = d _ {x} ^ {2} (z) + d _ {y} ^ {2} (z). \tag {12}
$$

Note that $d_x^2 (z) + d_y^2 (z)$ is again a Laurent polynomial in $z$ . We also define a single-band Hamiltonian $Q(z)$ given by [55]:

$$
Q (z) = d _ {x} ^ {2} (z) + d _ {y} ^ {2} (z) \tag {13}
$$

for which the GBZ is still given by $|z_{p}| = |z_{p + 1}|$ . Thus, the two-band model in Eq. (12) is related to the single-band model via $E(z) = \pm \sqrt{Q(z)}$ and the GBZ of the

two-band sublattice-symmetric model is two copies of the GBZ of the single-band model [50]. Note that here we used $d_{o}(z) = 0$ which arises from $H(z)$ being in the off-diagonal form as a result of sublattice symmetry. For general two-band models with $d_{o}(z) \neq 0$ , one can no longer construct in a simple way a one-band model that has the same GBZ, as we have done here [55].

Let us give a bit more background on where Eq. (8) comes from. The non-Bloch winding number $W$ [7, 8] is given by the winding of the off-diagonal matrix elements:

$$
W _ {1} = \frac {1}{2 \pi} \oint_ {\mathcal {C} _ {\mathrm {g b z}}} \arg H _ {a b} (z) d z
$$

$$
W _ {2} = \frac {1}{2 \pi} \oint_ {\mathcal {C} _ {\mathrm {g b z}}} \arg H _ {b a} (z) d z \tag {14}
$$

$$
W = W _ {1} - W _ {2}.
$$

where $H_{ab}(z) = d_x(z) - id_y(z), H_{ba}(z) = d_x(z) + id_y(z)$ . Note that for a two-band model with sublattice symmetry, there are two copies of the GBZ laying on top of each other on the $z$ -plane [50]. Let us call each of the copies a subGBZ loop. Here, the integration contour $C_{\mathrm{gbz}}$ in Eq. (8), $W_1$ and in $W_2$ is an integral over just one of the two subGBZ loops, which is similar to taking the integral over the BZ of the Berry connection for a single band only for a two-band Hermitian Hamiltonian. Eq.

(8) is single-valued and analytic in $\arg(M^2(z))$ along the GBZ contour so long as the GBZ contour does not intersect zeros or poles of $M^2(z)$ on the $z$ -plane. Thus, taking the integration along one of the two copies is well-defined regardless of the underlying GBZ energy band braiding topology [50, 56-60]. $|W|$ is equal to the total number of edge states [7, 8, 10, 61] (note that some references include a factor of $1/2$ in Eq. (14) [8, 10, 61]). Eq. (8) is just the Hermitian $Q$ -matrix invariant [2, 52] when the model is Hermitian. Eq.

(8) is not analytic if the GBZ contour intersects with zeros or poles of $M^2(z)$ on the $z$ -plane. Such cases would correspond to gap closing points [62] where the OBC bands touch at $E = 0$ (see Appendix. V A).

When $H(z)$ is in the off-diagonal form, we can write

$$
M ^ {2} (z) = \frac {H _ {a b} (z)}{H _ {b a} (z)}. \tag {15}
$$

Then since

$$
\arg \left(M ^ {2} (z)\right) = \arg \left(\frac {H _
{a b} (z)}{H _ {b a} (z)}\right) = \arg H _ {a b} (z) - \arg H _ {b a} (z). \tag {16}
$$

we have

$$
W = \frac {1}{2 \pi} \oint_ {\mathcal {C} _ {g b z}} \arg M ^ {2} (z) d z = W _ {\mathrm {g b z}} \tag {17}
$$

As $M^2(z)$ is the ratio of two Laurent polynomials, it is a meromorphic function. From the argument principle, Eq. (17) states that $W$ is equal to the number of zeros minus the number of poles of $M^2(z)$ within the GBZ on

3

the $z$ plane. While this statement already resembles Eq. (9), the important difference here is that $W_{\mathrm{gbz}}$ counts the poles and zeros within the GBZ, while $W_{\mathrm{pz}}$ counts the poles and zeros within a circle centered on $z = 0$ that encloses the first $2\mu$ points.

To prove Eq. (10), we note that the single-band Hamiltonian $Q(z)$ can be written as

$$
Q (z) = \frac {P _ {p + q} (z)}{z ^ {p}} \tag {18}
$$

where $P_{p + q}(z)$ is a polynomial in $z$ of degree $p + q$ . Refs. [54, 63] states that the GBZ encloses the first $p$ zeros of $P_{p + q}(z)$ . The proof is quite extensive and can be found in the Supplementary of Ref. [54].

The zeros of $P_{p+q}(z)$ are also either poles or zeros of $M^2(z)$ . This is a direct consequence of

$$
\frac {P _ {p + q} (z)}{z ^ {p}} = H _ {a b} (z) H _ {b a} (z) \tag {19}
$$

As such, the zeros of $P_{p + q}(z)$ must also be zeros of either $H_{ab}(z) = 0$ or $H_{ba}(z) = 0$ which is a zero or pole of $M^2 (z)$ respectively. The zeros of $P_{p + q}(z)$ account for all poles and zeros of $M^2 (z)$ at finite and non-zero $z$ , however $z = 0$ and $z = \infty$ may also be poles and zeros of $M^2 (z)$ . We are mainly interested if $z = 0$ is a pole or zero as we are interested in poles and zeros within the GBZ. Recall that $m > 0$ is the highest negative power of $z$ in $H_{ab}(z)$ , $n > 0$ is the highest negative power of $z$ in $H_{ba}(z)$ and that $\mu = \max \{m,n\}$ . At $z = 0$ , the Laurent polynomials $H_{ab}(z)$ and $H_{ba}(z)$ are dominated by the $z^{-m}$ and $z^{-n}$ terms respectively and we have

$$
M ^ {2} (z) = \frac {H _ {a b} (z)}{H _ {b a} (z)} \approx C z ^ {m - n} \tag {20}
$$

where $C$ is a constant. If $m = n$ , then $M^2(z)$ is finite and $z = 0$ is neither a pole nor a zero of $M^2(z)$ . If $m > n$ , then $z = 0$ is a zero of $M^2(z)$ of order $m - n$ and if $n > m$ , then $z = 0$ is a pole of $M^2(z)$ of order $n - m$ .

We therefore have that the total number of poles and zeros within the GBZ is $p + |m - n|$ . Note that $p = m + n$ . If $m > n$ , the GBZ encloses the first $2m$ poles and zeros and if $n > m$ , the GBZ encloses the first $2n$ poles and zeros. Hence, the GBZ encloses the first $2\mu$ poles and zeros of $M^2 (z)$ . Eq. (10) therefore follows by the argument principle.

# D. Edge-state invariant on the $M$ -Riemann sphere

Both the $W_{\mathrm{gbz}}$ of Eq. (8) and the $W_{\mathrm{pz}}$ of Eq. (9) are related to the poles and zeros of $M^2(z)$ , which are closely related to the "poles and zeros" of $M(z)$ (more rigorously, the points where $M(z) \to \infty$ and $M(z) \to 0$ , respectively). The poles and zeros of $M(z)$ have a simple Riemann-sphere interpretation. The right-eigenvector

ratio $M$ is a complex number. We convert $M$ to a Riemann-sphere via the stereographic projection:

$$
X (M) = \frac {2 \operatorname {R e} (M)}{1 + | M | ^ {2}} \tag {21}
$$

$$
Y (M) = \frac {2 \operatorname {I m} (M)}{1 + | M | ^ {2}} \tag {22}
$$

$$
Z (M) = \frac {\left| M \right| ^ {2} - 1}{\left| M \right| ^ {2} + 1} \tag {23}
$$

where $(X,Y,Z)$ are Cartesian coordinates of a point on the Riemann sphere. Below we refer to the Riemann sphere as defined by Eqs. (21) to (23) as the $M$ -Riemann sphere. A point on the $M$ -Riemann sphere can be alternatively described in spherical coordinates: i.e. $(X,Y,Z) = (\sin \theta \cos \varphi, \sin \theta \sin \varphi, \cos \theta)$ . We can then map the $M$ -Riemann sphere to $M$ by

$$
M (\theta , \varphi) = e ^ {i \varphi} \cot \left(\frac {\theta}{2}\right). \tag {24}
$$

In the maps of Eqs. (21)-(24), the poles and zeros of $M(z)$ correspond to the north and south pole on the $M$ -Riemann sphere respectively.

Eq. (8) describes the winding of the image of the GBZ about the origin on the complex $M$ -plane, which corresponds to the winding number of the image of the GBZ on the $M$ -Riemann sphere about the north and south poles. The $M$ -Riemann sphere reduces to the typical Bloch sphere for two-level Hermitian models [52], and is equivalent to the non-Hermitian Bloch sphere in Ref. [64] when we take the GBZ as the integration contour.

For simple cases with two-band sublattice symmetry, a subGBZ loop winds around the origin of the $z$ -plane once [50]. For a model in off-diagonal form, the image of such a subGBZ loop that winds twice around the origin on the $M^2(z)$ -plane corresponds to a subGBZ loop that winds once around the origin on the $M(z)$ -plane. Using the $M^2(z)$ plane is easier for interpreting the pole-zero invariant, as there are no branch cuts, but the winding on $M(z)$ plane (and hence the $M$ -Riemann sphere) is more similar to typical Bloch sphere interpretations. A winding number of 2 in Eq. (8) corresponds to a subGBZ loop winding around the north-south axis once on the $M$ -Riemann sphere which corresponds to two topological edge states in a finite open-boundary case.

# E. General sublattice-symmetric edge-state invariant

The results of Eq. (8) and Eq. (9) require the Bloch matrix to be in an off-diagonal form. Here, we show that the results can be generalized to a two-band non-Hermitian Hamiltonian with sublattice symmetry that may not be represented in this form. A two-band Hamiltonian with sublattice symmetry in general is described

4

by Eq. (4) above, where $\Gamma$ may not be equal to $\sigma_z$ . However, one can always find a unitarily equivalent model in off-diagonal form for a sublattice symmetric model [50]. Since $\Gamma$ is both unitary and Hermitian, it is in the form of $\Gamma = \mathbf{d}_{\Gamma} \cdot \sigma$ , where $\mathbf{d}_{\Gamma}$ is a unit real vector in three dimensions. As such, there exists a real, orthogonal matrix $R$ that rotates $\mathbf{d}_{\Gamma}$ to the $Z$ -axis in the three-dimensional space.

Using spherical coordinates, we set $\mathbf{d}_{\Gamma} = (\sin \theta_{\Gamma} \cos \varphi_{\Gamma}, \sin \theta_{\Gamma} \sin \varphi_{\Gamma}, \cos \theta_{\Gamma})^T$ with $0 \leq \varphi_{\Gamma} < 2\pi$ and $0 \leq \theta_{\Gamma} \leq \pi$ . Then $R$ can be mapped to a unitary matrix $U$ in the two-dimensional Hilbert space of the two-band model where

$$
U = \left[ \begin{array}{c c} \cos \theta_ {\Gamma} / 2 & - e ^ {- i \varphi_ {\Gamma}} \sin \theta_ {\Gamma} / 2 \\ e ^ {i \varphi_ {\Gamma}} \sin \theta_ {\Gamma} / 2 & \cos \theta_ {\Gamma} / 2 \end{array} \right] \tag {25}
$$

with $\Gamma = U^{-1}\sigma_zU$ . As such, the transformed Hamiltonian $UH(z)U^{-1}$ satisfies $\sigma_z(UH(z)U^{-1})\sigma_z = -UH(z)U^{-1}$ and has the off-diagonal form.

We discuss the generalization of the GBZ invariant in Eq. (8) first. For $H(z)$ in the off-diagonal form, the GBZ invariant is the winding of the image of the GBZ on the $M$ -Riemann sphere about the $Z$ -axis. The $Z$ -axis is physically significant in this case because the intersection of the $M$ -Riemann sphere with the $Z$ -axis corresponds to the $M(z)$ poles and zeros which are the gap closing points [64]. A unitary transform leaves the eigenvalues of a model unchanged but affects the eigenvectors and therefore $M(z)$ .

It can be shown that such a transform leads to a rotation on the $M$ -Riemann sphere that rotates the $Z$ -axis to the $\mathbf{d}_{\Gamma}$ -axis (see Supplementary VC). Thus, the winding number on the $M$ -Riemann sphere for a general sublattice symmetric model must be around the $\mathbf{d}_{\Gamma}$ -axis instead of the $Z$ -axis. On the other hand, the gap closing points for any two-band non-Hermitian model are the $z$ -plane branch points gi
ven by [50, 62]

$$
Q (z) = d _ {x} ^ {2} (z) + d _ {y} ^ {2} (z) + d _ {z} ^ {2} (z) = 0. \tag {26}
$$

More information on why these give the branch points is given in Supplementary V A. The $z$ solutions to the above equation map to one of the two antipodal points on the $M$ -Riemann sphere, which we denote as $M_{\mathrm{N}}$ and $M_{\mathrm{S}}$ . These two points satisfy $M_{\mathrm{N}} M_{\mathrm{S}}^{*} = -1$ and are precisely the intersections of the $\mathbf{d}_{\Gamma}$ -axis with the $M$ -Riemann sphere. The roots of $Q(z)$ provide an alternative method of deriving the unitary transform required between off-diagonal forms and more general forms. To use the GBZ form of the general sublattice symmetry invariant, one can find the $M$ value for any one of the branch points on the $M$ -Riemann sphere by solving $Q(z) = 0$ , and plot the GBZ trajectory on the $M$ -Riemann sphere.

We now discuss the generalization of the pole-zero invariant in Eq. (9). For $H(z)$ in the off-diagonal form, the construction of $M^2 (z)$ as a single-valued function allows direct identification of its zeros and poles. This construction is no longer suitable for a general sublattice symmetric model as the branch points of $M(z)$ , located at $M_N$ or $M_S$ , are generally not coincident with $M = 0$ and

$M = \infty$ . As such, we will convert the invariant into a form that involves the image of the $z$ -plane branch points in $M$ , which then generalizes to the cases with general sublattice symmetry.

Consider the pole-zero invariant using $M(z)$ in Eq. (9) instead of $M^2(z)$ for a model in off-diagonal form first. For simplicity, let us assume all poles and zeros in $M^2(z)$ have order $\pm 1$ . The pole-zero invariant using $M(z)$ then becomes

$$
\# (M = 0) - \# (M = \infty) \tag {27}
$$

for the first $2\mu$ $M(z)$ values where $z$ are the solutions to $z^{2\mu}Q(z) = 0$ sorted by the $|z|$ magnitude. As the higher-order poles or zeros correspond to degenerate $z$ solutions, this formula also applies to higher-order poles or zeros in $M^2$ if we count each $z$ solution exactly once.

Using the fact that a unitary transform leads to a rotation in the $M$ -Riemann sphere, the above $M(z)$ pole-zero invariant remains the same for a sublattice-symmetric model not in off-diagonal form, except that we replace $M = 0$ and $M = \infty$ with $M = M_N$ or $M = M_S$ . Depending on which of the $M_N$ and $M_S$ replaces 0 and which replaces the other, the $W_{\mathrm{pz}}$ defined will differ by a sign, but the edge state count $|W_{\mathrm{pz}}|$ remains the same.

As such, the pole-zero invariant for general sublattice symmetry using $M(z)$ is

$$
\# (M = M _ {N}) - \# (M = M _ {S}) \tag {28}
$$

for the first $2\mu$ $M(z)$ values where $z$ are the solutions to $z^{2\mu}Q(z) = 0$ sorted by the $|z|$ magnitude. To calculate the invariant in this form, the first $2\mu$ solutions to $z^{2\mu}Q(z) = 0$ is listed, possibly including $z = 0$ . The corresponding $M(z)$ values can be found by substituting $z$ into the definition of $M$ (Eq. (7)). Since the $M(z)$ obtained are from $z$ branch points, they can only take two distinct values ( $M_N$ and $M_S$ ). The invariant then becomes the difference between the number of appearances of $M_N$ and $M_S$ .

# III. NUMERICAL RESULTS

In this section, we illustrate the connections between the GBZ invariant in Eq. (8) and pole-zero invariant Eq. (9) through numerical examples. The models that we consider include the Hermitian SSH model [65], a generalized non-Hermitian SSH model with $t_3$ hoppings where $\mu = 1$ [7, 8, 15, 66], a longer-range non-Hermitian model with hopping range across three unit cells where $\mu = 3$ , and an example with winding number in Eq. (8) greater than 2 where $\mu = 2$ . These are all in off-diagonal form and Eq. (8) and Eq. (9) apply. We then also provide numerical results on the SSH-Creutz model [4, 12-14, 55] which has sublattice symmetry but is not in the off-diagonal form, where Eq. (28) applies.

For each of these examples, we demonstrate the GBZ invariant in Eq. (8) by plotting the image of the GBZ on

5

the $M$ -Riemann sphere in subpanel (b) of Figs. 1 to 11. The GBZ invariant is interpreted as the winding around the north-south pole axis for the models in off-diagonal form or the $M_{\mathrm{N,S}}$ axis (i.e the $\mathbf{d}_{\Gamma}$ -axis) for SSH-Creutz models.

For the models in off-diagonal form, we also provide plots on the $z$ -plane colored by $\arg (M^2 (z)) = \frac{H_{ab}(z)}{H_{ba}(z)}$ which gives a way to visualize the GBZ invariant in Eq. (17) in subpanel (c) of Figs. 1 to 7. The color scheme follows domain coloring [67] which maps argument values of $\arg (M^2 (z))$ from $-\pi$ to $\pi$ as a hue gradient from red through the spectrum back to red. The argument principle then corresponds to the number of hue cycles crossed along a contour on the $z$ -plane. For instance, $W = 2$ means two hue cycles are crossed along the GBZ path.

Poles and zeros are visible as singularities in the color scheme, where the hue cycle wraps in opposite directions around poles when compared to zeros (and vice versa). Using domain colouring, we can also tell what the order of the pole or zeros are, as they correspond to the number of hue cycles emanating from that point. We mark poles and zeros with $\times$ and $\circ$ markers respectively as well as the GBZ path in Figs. 1 to 7, allowing us to visually count the poles and zeros in the GBZ and verify Eq. (9).

For the SSH-Creutz models which have sublattice symmetry but is not in off-diagonal form, we plot the GBZ values on the $z$ -plane as well as N and S markers for the $z$ values that correspond to $M_N$ and $M_S$ points in subpanel (c) for Figs. 8 to 11, which allows us to verify Eq. (28). We do not color the $z$ -plane by $\arg(M^2(z))$ or $\arg(M(z))$ in these plots as these quantities do not have a simple form for general sublattice symmetry form and have branch cuts, making it harder to visually interpret the plots. As the relevant axis is no longer along the north-south pole, the hue cycles crossed along a contour are no longer relevant.

For Figs. 1 to 9, in subpanel (a) we show the OBC eigenvalues for $N = 100$ unit cells, which leads to 200 OBC eigenvalues as there are two sublattice sites per unit cell. The OBC plot shows whether or not we have topological zero energy edge states. If there are edge states, we also include in subpanel (d) a list plot of the real part of the OBC eigenvalues $\mathrm{Re}(E)$ plotted against their sorted $\mathrm{Re}(E)$ order. This way, the zero energy edge states must appear in the middle of the sorted indices. We include the middle 40 OBC eigenvalues which allows us to visually count the number of degenerate zero energy edge states.

# A. Hermitian SSH

We begin with the Hermitian SSH model, which is very well-studied in the literature [52, 65]. It is given by the Hamiltonian

$$
H _ {\mathrm {S S H}} (z) = \left[ \begin{array}{c c} 0 & t _ {1} + t _ {2} / z \\ t _ {1} + t _ {2} z & 0 \end{array} \right]. \tag {29}
$$

![](dt=2026-06-10/ht=20/421aac75a3dbb17a3f9233201a5a0f331b6681b5674201ca38f851a340aed2b7.jpg)

![](dt=2026-06-10/ht=20/c31243b6da69a4bceed7351e30ca404ef4ca548ef4257d52c40ab12f93d91802.jpg)

![](dt=2026-06-10/ht=20/4ba80d877555138096d8a3849b15df6e1cbf8c6e8b1798a9df319233b57a992a.jpg)

![](dt=2026-06-10/ht=20/699aa2b2ebbdd937dee89b95e0779d51e89aa028e92715790da553c9179ec16d.jpg)

where $t_1$ are the intracell couplings and $t_2$ are the intercell couplings between the two sublattice sites in a unit cell. In Fig. 1, we use the parameters $t_1 = 0.5, t_2 = 1$ for which th
e SSH model supports topological edge states. In Fig. 1(a), we plot the eigenenergy of this Hamiltonian for a finite lattice with $N = 100$ unit cell sites. We note the presence of edge states at zero energy. For Hermitian models, the GBZ is the same as BZ with $|z| = 1$ . In Fig. 1(b), we plot the $M_{+}(z)$ and $M_{-}(z)$ contours as $z$ varies on the BZ.

Both contours are located at the equator of the $M$ -Riemann sphere and winds around the north and the south poles. The results here provide a validation of the connection between the winding on the $M$ -Riemann sphere and the existence of topological edge states. In Fig. 1(c), the first $2\mu = 2$ poles and zeros of $M^2(z)$ within the GBZ are both poles, which leads to $W = 2$ which corresponds to 2 nontrivial topological edge states, as verified in Fig. 1(d).

In Fig. 2 we use the parameters $t_1 = 1.5, t_2 = 1$ . In Fig. 2(a), there are no edge states. In Fig. 2(b) the image of the GBZ on the $M$ -Riemann sphere has a trivial winding. In Fig. 2(c) the first $2\mu = 2$ poles and zeros of $M^2(z)$ within the GBZ consist of a pole and a zero, thus $W = 0$ and there are no topological zero-energy edge states.

# B. Non-Hermitian longer-range SSH models

We now consider a generalized non-Hermitian version of the SSH model [7, 8, 15, 66] which is in off-diagonal form and has $H_{ab}(z) = \frac{\gamma_1}{2} + t_1 - \left(\frac{\gamma_2}{2} - t_2\right) / z + \left(\frac{\gamma_3}{2} + t_3\right)z$ .

6

![](dt=2026-06-10/ht=20/76e13b132fdb96a40e1139727d6057a8891dd0fea5a77a073477a94a0bc598e4.jpg)

![](dt=2026-06-10/ht=20/d46bf73bb7a14de4ffa4f5de5156d89513bc5cb0119e5d02c3ee4d3ec470c497.jpg)

![](dt=2026-06-10/ht=20/3cf695305bdb78cfc8454d0e9844cd84df9baa6baf373cfc3d55beb7094c56a4.jpg)

![](dt=2026-06-10/ht=20/09cbb14a95fbc752c2c534c3c9267a0d1fc65090d355133d9b32dc36e46da2d3.jpg)

![](dt=2026-06-10/ht=20/cbb823d0636c9c4e902db5bf1cd20d7bcaedabed6b80dbfacbaa319ea8fc5ae3.jpg)

![](dt=2026-06-10/ht=20/3e8ae078179424194a902d15b208e77c5e2b140bc9ecbba71de2f9dc1cf34a45.jpg)

![](dt=2026-06-10/ht=20/31e1115b2cc59c1aabc707608c8b4b75a5522e6fe01ecf715596e650bdeee1c1.jpg)

![](dt=2026-06-10/ht=20/217fcb9521dd69b4cdf06cd55a6479a2ccd16b0858d10e62ba88470f14924b95.jpg)

![](dt=2026-06-10/ht=20/393ceae871db56612e61eea0ecbeaa03404561744dc41f7eaaa3e8a30ea71a25.jpg)

![](dt=2026-06-10/ht=20/4bb2342da858cce921e74c1d33338b1312adc6aec3b381242b8c2d749fbea3d9.jpg)

![](dt=2026-06-10/ht=20/ef403313425ccc096cc522aa4f6ed7baa09ffac61b049801f5d26137969a4377.jpg)

![](dt=2026-06-10/ht=20/f013a495254e8d1d6b5be1de8dc0c76d2d22e114d1b5c8bfcc721ba060651d43.jpg)

![](dt=2026-06-10/ht=20/c220ad520ae6707c948689f391f4b1e52c6140ffd584d72664bdbe7df8666c09.jpg)

![](dt=2026-06-10/ht=20/e58860fa50ad102cc264ed23da69afa7dd5862294aa252438bb48479cf367420.jpg)

and $H_{ba}(z) = -\frac{\gamma_1}{2} + t_1 + \left(\frac{\gamma_2}{2} + t_2\right)z - \left(\frac{\gamma_3}{2} - t_3\right) / z$ . Here, $\mu = 1$ as well.

In Fig. 3, we use the parameters $t_1 = 1, t_2 = 1, t_3 = 3, \gamma_1 = 2, \gamma_2 = 1, \gamma_3 = 1$ for which the system has topological edge states near $E = 0$ as seen in Fig. 3(a). We see that in Fig. 3(b), even though $d_o(z) = d_z(z) = 0$ , for non-Hermitian models the image of the GBZ on the M-Riemann sphere may not be confined to the equator like in Hermitian models. In this case, both subGBZ winds around the north-south pole axis. In Fig. 3(c), we show that the first $2M^2(z)$ poles and zeros are both zeros, $W = 2$ corresponding to two topological edge states, as

verified in Fig. 3(d).

In Fig. 4, we use the parameters $t_1 = 8, t_2 = 1, t_3 = 3, \gamma_1 = 2, \gamma_2 = 1, \gamma_3 = 1$ for which the system has no topological edge states, as seen in Fig. 4(a). We see that in Fig. 4(b), the image of both subGBZ loops do not wind around the north-south pole axis. In Fig. 4(c), we show that the first 2 $M^2(z)$ poles and zeros are a zero and a pole, thus $W = 0$ indicating no topological edge states.

We now consider an even longer range extension which is also in off-diagonal form and has $H_{ab}(z) = t_1 + t_1' / z + v / z^2 + uz^3$ , $H_{ba}(z) = t_2 + t_2' z + v z^2 + u / z^3$ . This time, $\mu = 3$ and we are interested in the first 6 $M(z)$ pole and

7

![](dt=2026-06-10/ht=20/e442a8371610d6c0f369f9497225803388c37c81beb7f4e17a0a774dd3bf5efa.jpg)

![](dt=2026-06-10/ht=20/c51dfa94b5c1e72d7a4c40bb903618c956fcaa4e3cc0b44782d69fee614197ba.jpg)

![](dt=2026-06-10/ht=20/63165b3842ef1bf21e993b69cacbfa9626d415f27b2511e3d20f574b323c0539.jpg)

# Zeros.

In Fig. 5, we use the parameters $t_1 = 0.5 + 0.5i$ , $t_1' = 2$ , $t_2 = 0.5 - 0.2i$ , $t_2' = 2$ , $v = 0.5$ , $u = 0.3$ for which the system has topological edge states near $E = 0$ , as seen in Fig. 5(a). We see that in Fig. 5(b) the image of both subGBZ loops wind around the north-south pole axis. In Fig. 5(c), we show that the first 6 $M(z)$ poles and zeros consists of four poles and two zeros, $W = 2$ indicating two topological edge states, as verified in Fig. 5(d).

In Fig. 6, we use the parameters $t_1 = 2 + 0.5i$ , $t_1' = 2$ , $t_2 = 2 - 0.2i$ , $t_2' = 2$ , $v = 0.5$ , $u = 0.3$ for which the system has no topological edge states, as seen in Fig. 6(a). We see that in Fig. 6(b), both subGBZ loops do not wind around the north-south pole axis. In Fig. 6(c), we show that the first $6M(z)$ poles and zeros consists of three poles and three zeros, thus $W = 0$ indicating no topological edge states.

# C. Example with winding number larger than 2

We now consider an example that winding numbers in Eq. (8) larger than 2 which corresponds to more than two edge states. Consider a non-Hermitian generalization of a model in Ref. [68] with $H_{ab}(z) = t_0 + \frac{t_1 + g_1}{z} + \frac{t_2 + g_2}{z^2}$ and $H_{ba}(z) = t_0 + (t_1 - g_1)z + (t_2 - g_2)z^2$ with $t_0 = 5, t_1 = 10, t_2 = 15, g_1 = 1, g_2 = 1$ .

In Fig. 7(a), we see that there are topological edge states. In Fig. 7(b), we see that both subGBZ loops wind around the north-south pole axis twice. In Fig. 7(c), noting that the pole at the origin has order two (as there are two hue cycles emanating from that point) and that the first $2\mu = 4$ poles and zeros of $M^2(z)$ consist of four poles, we have $W = 4$ corresponding to four edge states, as verified in Fig. 7(d).

![](dt=2026-06-10/ht=20/ece0c2bd1520c3a29b5529c39e465a50e610301e157b20ebbb1309ddb4518136.jpg)

![](image)
17fd7e74be447cf704f73dab2b843e6600da1dc644127ee0cfa.jpg)

![](dt=2026-06-10/ht=20/6f646aeaf3a04fd172366cd0ff11dcd811c0197797f42f1e1099b22ca7763b38.jpg)

![](dt=2026-06-10/ht=20/3589509f812d12aadb40e8026639cdf3662b430b5d3716e262ae160613d00e73.jpg)

# D. SSH-Creutz model

We now consider a sublattice symmetric model not in the off-diagonal form. To do this, we start with the Creutz model [14, 55, 69] given by:

$$
H _ {\text {C r e u t z}} (z) = \left[ \begin{array}{c c} i t _ {2} / (2 z) - i t _ {2} z / 2 & t _ {1} + t _ {2} / (2 z) + t _ {2} z / 2 \\ t _ {1} + t _ {2} / (2 z) + t _ {2} z / 2 & - i t _ {2} / (2 z) + i t _ {2} z / 2 \end{array} \right] \tag {30}
$$

This model has the same eigenvalues as the SSH model and is equivalent under a unitary transform [4]. We will consider the following SSH-Creutz model described by

$$
H (z) = a H _ {\mathrm {S S H}} (z) + b H _ {\mathrm {C r e u t z}} (z) \tag {31}
$$

where $a$ and $b$ can be considered as weightings of the SSH and Creutz model respectively. Here, $\mu = 1$ and we are interested in the first $2M_{\mathrm{N}}$ and $M_{\mathrm{S}}$ points as discussed in Sec. II E.

In Fig. 8, we use the parameters $a = 0.3$ , $b = 0.7$ , $t_1 = 0.5$ , $t_2 = 1$ for which the system has topological edge states, as seen in Fig. 8(a). In Fig. 8(b), we also plot the image of the $z$ -plane branch points on the M-Riemann sphere as a red dots. They map to the antipodal points $M(z) = -0.66i$ or $1.52i$ , which in M-Riemann sphere coordinates is $(X(M), Y(M), Z(M)) = (0, 0.92, 0.39)$ , $(0, -0.92, -0.39)$ . Call these two points N and S respectively. We see that both subGBZ wind around the axis formed by these two antipodal points, indicating a topological phase. In Fig. 8(c), the $M_N$ and $M_S$ are marked as N and S on the $z$ -plane. In this case, the first $2\mu = 2$ $M_N$ or $M_S$ points inside the GBZ are

8

![](dt=2026-06-10/ht=20/5705b30ae3fc82feddef994c17ae4343ccf1c73bdae7b871982ba028adcf5c22.jpg)

![](dt=2026-06-10/ht=20/4c10590a99bf7b8a7eaf59f9868a0a30920efc384f5c0cd3598bdb714d55cf3e.jpg)

![](dt=2026-06-10/ht=20/8ccb931a8d1a55881d38e707a6e34571dc84a828078cf504b2793ede2e89ef3e.jpg)

![](dt=2026-06-10/ht=20/c512cb30d099e83892254dabe97b272c4a29798e65248a5b600f7fcc4fb658ef.jpg)

![](dt=2026-06-10/ht=20/570d307a3e8af0cc3e12a62a4b9305da619f097b04ef5f63fbb2a232acacc74a.jpg)

![](dt=2026-06-10/ht=20/ab3f116a6b4e6c35777ee8afbd6c98b8e46e4001f65aa6690779882d181ab23c.jpg)

![](dt=2026-06-10/ht=20/34cbb01e30851dc697f365d64232e72512d654a7db7865d110afecb426db631c.jpg)

![](dt=2026-06-10/ht=20/d20d0524ec0eeb500594e88c988ef6321d348a6ba91459b36733314c351d8ec8.jpg)

![](dt=2026-06-10/ht=20/db5d36966a7a9952610fee195e1fe3f8c356ec3c8c843b9e29c062d77b81e8f2.jpg)

![](dt=2026-06-10/ht=20/e8ac0a03a554b0645ca4a74b925c445866fc9d9fefc12954c618bb2413e68c71.jpg)

![](dt=2026-06-10/ht=20/bf3b4639b5e5d296bebb286fe7f49871773625a4049e8b66ccb9630b8b7b763d.jpg)

$M(z) = 1.52i$ and $1.52i$ , (or $(X(M),Y(M),Z(M)) = (0,0.92,0.39),(0,0.92,0.39)$ ). Both of these $M_{\mathrm{N}}$ or $M_{\mathrm{S}}$ points inside the GBZ are S poles, which indicates a topological phase.

In Fig. 9, we use the parameters $a = 0.3$ , $b = 0.7$ , $t_1 = 1.5$ , $t_2 = 1$ which has no edge states, as seen in Fig. 9(a). In Fig. 9(b), we again see that both subGBZ do not wind around the axis formed by these two antipodal points, indicating a topological phase. In Fig. 8(c), the first $2\mu = 2$ $M_{\mathrm{N,S}}$ points inside the GBZ are

$M(z) = -0.66i$ and $1.52i$ , (or $(X(M), Y(M), Z(M)) = (0, 0.92, 0.39)$ , $(0, 0.92, -0.39)$ ). These $M_{\mathrm{N}}$ or $M_{\mathrm{S}}$ points consist of a S and an N pole, which indicates that the system is in a trivial phase.

# E. Non-Hermitian SSH-Creutz model

Let us consider the non-Hermitian Creutz model [4, 14]:

$$
H _ {\mathrm {n h C r e u t z}} (z) = \left[ \begin{array}{l l} \frac {- i (t _ {2} + \gamma)}{2 z} + \frac {i t _ {2} z}{2} & t _ {1} + \frac {t _ {2}}{2 z} + \frac {t _ {2} z}{2} \\ t _ {1} + \frac {t _ {2}}{2 z} + \frac {t _ {2} z}{2} & \frac {i (t _ {2} + \gamma)}{2 z} - \frac {i t _ {2} z}{2} \end{array} \right] \tag {32}
$$

as well as the generalized non-Hermitian SSH described above (and call it $H_{\mathrm{nhSSH}}(z)$ ) but where $\gamma_1 = \gamma = 0.7, \gamma_2 = \gamma_3 = t_3 = 0$ . Now consider the following non-Hermitian SSH-Creutz model:

$$
H (z) = a H _ {\mathrm {n h S S H}} (z) + b H _ {\mathrm {n h C r e u t z}} (z) \tag {33}
$$

which is non-Hermitian, has sublattice symmetry and is not in off-diagonal form.

In Fig. 10, we use the parameters $a = 0.3$ , $b = 0.7$ , $t_1 = 0.5$ , $t_2 = 1$ which has topological edge states. We make the same conclusions as in Fig. 8 except that in Fig. 10 we have a non-Hermitian example. The main difference is that the GBZ is not confined to $|z| = 1$ , and the GBZ trajectory on the $M$ -Riemann sphere does not have to be on the equatorial plane perpendicular to the $M_{\mathrm{N,S}}$ axis due to non-Hermiticity.

In Fig. 11, we use the parameters $a = 0.3$ , $b = 0.7$ , $t_1 = 1.5$ , $t_2 = 1$ which has no topological edge states. We make

9

![](dt=2026-06-10/ht=20/0eb093f9c764bcca6338fe265c237526ea0704e137818a2dd8f0723f24aff19d.jpg)

![](dt=2026-06-10/ht=20/240dbfa642e6a1819f229652d8eefb4bd4a177bbc8a77de2c0ea08150b689513.jpg)

![](dt=2026-06-10/ht=20/505aa8301b2ef78f53568c590f641bd20f6fc66df255aed4b3eac3f3860f11f8.jpg)

the same conclusions as Fig. 9 but for a non-Hermitian example. Thus, we see that our generalized sublattice symmetry GBZ and pole-zero invariant also applies in the case of non-Hermiticity.

# IV. CONCLUSION

In conclusion, we have demonstrated that for non-Hermitian, two-band, sublattice-symmetric tight-binding models in off-diagonal form, the GBZ invariant in Eq. (8) is equivalent to the pole-zero invariant in Eq. (9). By introducing an $M$ -Riemann sphere interpretation of the GBZ invariant based on the eigenvector ratio $M$ , we derived a more generalized version of the pole-zero invariant applicable to sublattice-symmetric models not necessarily in off-diagonal form. This $M$ -Riemann sphere interpretation of the GBZ invariant also extends to models beyond the off-diagonal form.

We numerically illustrated these invariants and cases using non-Hermitian SSH models, their variants, and non-Hermitian SSH-Creutz models. In many cases for sublattice symmetric models, the calculation of the pole-zero invariant can be numerically or analytically easier than the GBZ invariant as it is a lower order polynomial equation. By establishing the equivalence of these approaches, we hope that the tools from both methods can be used more interchangeably or
provide more insights for either method.

We also hope that the $M$ -Riemann sphere formalism introduced in this paper may yield further insights in non-Hermitian systems beyond this symmetry class or that connections to concepts in this work may be found in non-tight-binding models [46]. A deeper understanding of the topological invariants for 1D non-Hermitian sublattice symmetric edge states [7, 8, 10, 14, 15, 55, 70, 71] may lead to new

insights in the interplay of edge states with other phenomena (such as lasing [26, 27], squeezed states [28-31], subsymmetry protected systems [32], the critical non-Hermitian skin effect [33], braiding of edge states [34] and more) or models beyond this work (such as multiband [35, 36], driven [37-39], higher dimensional [40-43], nonlinear [44], stochastics [45], continuous models [46-48] and more).

# ACKNOWLEDGMENTS

J.Z acknowledges discussions on related concepts for an earlier version of this work with C. Wojcik and members/ alumni of the Z. Wang group. She also thanks C. Lee and D. Felbacq for email correspondence. This work is funded by a Simons Investigator in Physics grant from the Simons Foundation (Grant No. 827065), and by a MURI project from the U.S. Air Force of Office of Scientific Research (Grant No FA9550-21-1-0244). J.Z was supported by a Fulbright Future Scholarship and the Quad Fellowship.

# V. APPENDIX

# A. Calculating $z$ -plane branch points

The zeros of $P_{p + q}(z)$ is related to the $z$ -plane branch points of Eq. (11). To determine these branch points, we rewrite the characteristic polynomial as a polynomial in $E$ and $z$ :

$$
E ^ {2} z ^ {p} - P _ {p + q} (z) = 0 \tag {34}
$$

The nonzero $z$ -plane branch points occur when the discriminant of the above equation with respect to $E$ is zero [50, 72], which corresponds to the roots of $P_{p+q}(z)$ . This can alternatively be seen by noting that one can write $E(z) = \sqrt{Q(z)} = \sqrt{P_{p+q}(z)/z^p}$ . Branch points are given when the term in the radical is zero, i.e. when $P_{p+q}(z) = 0$ . Thus, the $z$ -plane branch points occur at $E = 0$ for sublattice symmetric models, which are the gap closing points [62].

In addition the $z$ -plane branch points as determined by $P_{p + q}(z) = 0$ , the points $z = 0, \infty$ may also be branch points [50, 72]. To check whether $z = 0$ is a branch point, we consider

$$
Q (z) = \sum_ {l = - p} ^ {q} c _ {l} z ^ {l}. \tag {35}
$$

At $z = 0$ , $Q(z)$ is dominated by the $z^{-p}$ term. Here, $c_{l}$ are constants. Then $E(z)$ around $z = 0$ is

$$
E \approx \sqrt {c _ {- p} z ^ {- p}} = \sqrt {c _ {- p}} \cdot z ^ {- p / 2}. \tag {36}
$$

As such, $z = 0$ is a branch point if and only if $p$ is odd [73]. Similarly, $z = \infty$ is a branch point if and only if $q$ is odd.

10

Note that when there is a branch point at $z = 0$ or $z = \infty$ for a model in off-diagonal form, it necessarily corresponds to a pole or zero of $M(z)$ . However, the converse is not true, and one can have a pole and zero of $M(z)$ without being a branch point. To see this, note that near $z = 0$ ,

$$
E (z) \approx \sqrt {c _ {- p}} z ^ {- p / 2} \tag {37}
$$

$$
H _ {a b} (z) \approx a _ {- m} z ^ {- m} \tag {38}
$$

$$
H _ {b a} (z) \approx b _ {- n} z ^ {- n} \tag {39}
$$

where $c_{-p}, a_{-m}, b_{-n}$ are constants. Since $p = m + n$ , if $p$ is even, and $m \neq n$ , then $z = 0$ is a pole or zero of $M(z)$ but it is not a $z$ -plane branch point.

# B. Eq. (9) implies the statement in Ref. [9]

We first restate the theorem from Ref. [9] for the existence of topological edge modes using the notation from our paper. The statement is that an isolated topological zero mode exists when the $m' + n'$ members from the roots of $H_{ab}H_{ba}$ having the largest magnitude do not contain $m'$ members from the roots of $H_{ab}$ and $n'$ members from the roots of $H_{ba}$ . Here, $m'(n')$ is the highest positive power of $z$ in $H_{ab}(z)$ ( $H_{ba}(z)$ ).

We first note that Ref. [9] uses the largest roots. By taking the complement of the set of roots, we get an equivalent formulation of this statement: edge modes exist when the $m + n$ members from the roots of $H_{ab}H_{ba}$ having the smallest magnitude do not contain $m$ members from the roots of $H_{ab}$ and $n$ members from the roots of $H_{ba}$ . The two statements have a similar form because reversing the entire tight-binding chain is equivalent to $z \rightarrow 1/z$ , which takes the largest roots to the smallest roots while preserving the spectrum.

In Ref. [9], $z = 0$ and $z = \infty$ are sometimes included as roots, but it is not specified when this is needed. To compare with our statement, it becomes necessary to include these roots when $m \neq n$ . If $m > n$ , then we add $(m - n)$ counts of $z = 0$ roots to $H_{ba}(z)$ , and vice versa. The extra $z = 0$ roots are guaranteed to rank first, and the statement becomes that the first $2\mu$ roots are not $\mu$ roots from $H_{ab}(z)$ and $\mu$ roots from $H_{ba}(z)$ .

Finally, note that the set of roots, including $z = 0$ and counted with multiplicity, can be succinctly written as the roots from the polynomial equation $z^{|m - n|}(z^{m}H_{ab})(z^{n}H_{ba}) = 0$ . This simplifies to $z^{2\mu}Q(z) = 0$ , and lists all of the poles and zeros of $M^2 (z)$ in the off-diagonal basis. As the roots from $H_{ab}(z)$ and

the roots from $H_{ba}(z)$ are the poles and zeros of $M^2 (z)$ respectively, the statement becomes that there must be an unequal number of poles and zeros of $M^2 (z)$ corresponding to the first $2\mu$ roots for edge mode existence, which is a special case of Eq. (9) with $W_{\mathrm{pz}}\neq 0$ .

# C. Unitary transformation in $H$ and rotation of the M-Riemann sphere

Let $U$ be a unitary matrix and $H$ be a $2 \times 2$ Bloch matrix. Assume $H|\psi\rangle = E|\psi\rangle$ . A unitary transformation on $H$ gives $H' = UHU^{-1}$ where $H'$ has the same eigenvalues $E$ as $H$ but transformed eigenvectors $U|\psi\rangle$ :

$$
\begin{array}{l} \left(U H U ^ {- 1}\right) (U | \psi \rangle) = U H | \psi \rangle (40) \\ = U E | \psi \rangle = E (U | \psi \rangle). (41) \\ \end{array}
$$

An arbitrary unitary matrix can be written in the form:

$$
U = \left[ \begin{array}{c c} e ^ {i \alpha} \cos \theta & e ^ {i \beta} \sin \theta \\ - e ^ {- i \beta} \sin \theta & e ^ {- i \alpha} \cos \theta \end{array} \right] \tag {42}
$$

where $\alpha, \beta, \theta \in \mathbb{R}$ . Consider

$$
\begin{array}{l} U \left[ \begin{array}{l} v _ {1} ^ {R} \\ v _ {2} ^ {R} \end{array} \right] = \left[ \begin{array}{c c} e ^ {i \alpha} \cos \theta & e ^ {i \beta} \sin \theta \\ - e ^ {- i \beta} \sin \theta & e ^ {- i \alpha} \cos \theta \end{array} \right] \left[ \begin{array}{l} v _ {1} ^ {R} \\ v _ {2} ^ {R} \end{array} \right] (43) \\ = \left[ \begin{array}{c} e ^ {i \alpha} \cos \theta v _ {1} ^ {R} + e ^ {i \beta} \sin \theta v _ {2} ^ {R} \\ - e ^ {- i \beta} \sin \theta v _ {1} ^ {R} + e ^ {- i \alpha} \cos \theta v _ {2} ^ {R} \end{array} \right] (44) \\ \end{array}
$$

If the original eigenvector ratio was $M = \frac{v_1^R(z)}{v_2^R(z)}$ , the new eigenvector ratio $M'$ becomes

$$
M ^ {\prime} = \frac {e ^ {i \alpha} \cos \theta M + e ^ {i \beta} \sin \theta}{- e ^ {- i \beta} \sin \theta M + e ^ {- i \alpha} \cos \theta}. \tag {45}
$$

This is a special case of the Mobius transformation $M' = \frac{aM + b}{cM + d}$ , where $a, b, c, d$ are complex constants.

Mobius transformations that are rotations on the Riemann sphere can be written in the form

$$
M ^ {\prime} = \frac {a M + b}{- b ^ {*} M + a ^ {*}} \tag {46}
$$

where $aa^{*} + bb^{*} = 1$ . We see that our Mobius transformation is of this form so unitary transforms of $H$ are also rotations in the $M$ -Riemann sphere.

Finally, by comparing with Eq. (25), we set $\alpha = 0$ , $\beta = \pi - \varphi_{\Gamma}$ and $\theta = \theta_{\Gamma} / 2$ . In this case the $M = \infty$ point is rotated into $M' = e^{i\varphi_{\Gamma}}\cot \theta_{\Gam
ma} / 2$ . Applying Eq. (24) then confirms that $M'$ is the endpoint of $\mathbf{d}_{\Gamma}$ . Similarly, the $M = 0$ point is rotated into the endpoint of $-\mathbf{d}_{\Gamma}$ .

11

12

13