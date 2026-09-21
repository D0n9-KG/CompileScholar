# Spawning rings of exceptional points out of Dirac cones

Bo Zhen $^{1*}$ , Chia Wei Hsu $^{1,2*}$ , Yuichi Igarashi $^{1,3*}$ , Ling Lu $^{1}$ , Ido Kaminer $^{1}$ , Adi Pick $^{1,4}$ , Song-Liang Chua $^{5}$ , John D. Joannopoulos $^{1}$ & Marin Soljačić $^{1}$

April 6, 2015

arXiv:1504.00734v1 [physics optics] 3 Apr 2015

1

The Dirac cone underlies many unique electronic properties of graphene<sup>1</sup> and topological insulators<sup>2</sup>, and its band structure—two conical bands touching at a single point—has also been realized for photons in waveguide arrays<sup>3</sup>, atoms in optical lattices<sup>4</sup>, and through accidental degeneracy<sup>5,6</sup>. Deformations of the Dirac cone often reveal intriguing properties; an example is the quantum Hall effect, where a constant magnetic field breaks the Dirac cone into isolated Landau levels<sup>7</sup>.

A seemingly unrelated phenomenon is the exceptional point<sup>8-11</sup>, also known as the parity-time symmetry breaking point<sup>12-15</sup>, where two resonances coincide in both their positions and widths. Exceptional points lead to counter-intuitive phenomena such as loss-induced transparency<sup>16</sup>, unidirectional transmission or reflection<sup>17-23</sup>, and lasers with reversed pump dependence<sup>24-26</sup> or single-mode operation<sup>27,28</sup>.

These two fields of research are in fact connected: here we discover the ability of a Dirac cone to evolve into a ring of exceptional points, which we call an “exceptional ring.” We experimentally demonstrate this concept in a photonic crystal slab. Angle-resolved reflection measurements of the photonic crystal slab reveal that the peaks of reflectivity follow the conical band structure of a Dirac cone from accidental degeneracy, whereas the complex eigenvalues of the system are deformed into a two-dimensional flat band enclosed by an exceptional ring.

This deformation arises from the dissimilar radiation rates of dipole and quadrupole resonances, which play a role analogous to the loss and gain in parity-time symmetric systems. Our results indicate that the radiation that exists in any open system can fundamentally alter its physical properties in ways previously expected only in the presence of material loss and gain.

Closed and lossless physical systems are described by Hermitian operators, which guarantee realness of the eigenvalues and a complete set of eigenfunctions that are orthogonal to each

2

other. On the other hand, systems with open boundaries $^{10,29}$ or with material loss and gain $^{12-14,16-28}$ are non-Hermitian $^{8}$ and have non-orthogonal eigenfunctions with complex eigenvalues where the imaginary part corresponds to decay or growth. The most drastic difference between Hermitian and non-Hermitian systems is that the latter exhibit exceptional points (EPs) where both the real and the imaginary parts of the eigenvalues coalesce.

At an EP, two (or more) eigenfunctions collapse into one so the eigenspace no longer forms a complete basis, and this eigenfunction becomes orthogonal to itself under the unconjugated inner product $^{8-11}$ . To date, most studies of EP and its intriguing consequences concern parity-time symmetric systems that rely on material loss and gain $^{12-14,16-28}$ , but EP is a general property that requires only non-Hermiticity. Here, we show the existence of EPs in a photonic crystal slab with negligible absorption loss and no artificial gain.

When a Dirac-cone system has dissimilar radiation rates, the band structure is altered abruptly to show branching features with a ring of EPs. We provide a complete picture from analytic model and numerical simulation to experimental observation; together, they illustrate the role of radiation-induced non-Hermiticity that bridges the study of EPs and the study of Dirac cones.

We start by showing that non-Hermiticity from radiation can deform an accidental Dirac point into a ring of EPs. First, consider a 2D photonic crystal $(\mathrm{PhC})^{30}$ (inset of Fig. 1a), where a square lattice (periodicity $a$ ) of circular air holes (radius $r$ ) is introduced in a dielectric material. This is a Hermitian system, as there is no material gain or loss and no open boundary for radiation. By tuning a system parameter (for example, $r$ ), one can achieve accidental degeneracy between a quadrupole mode and two degenerate dipole modes at the $\Gamma$ point (center of the Brillouin zone), leading to a linear Dirac dispersion due to the anti-crossing between two bands with

3

the same symmetry $^{5,31}$ . The accidental Dirac dispersion from the effective Hamiltonian model (see equation (1) below with $\gamma_0 = 0$ ) is shown as solid lines in Fig. 1a, agreeing with numerical simulation results (symbols in Fig. 1a). In the effective Hamiltonian we do not consider the dispersionless third band (gray line) due to symmetry arguments (section I in Supplementary Information), although this third band cannot be neglected in certain calculations, including Berry phase and effective medium property $^{32,33}$ .

Next, we consider a similar, but open, system: a PhC slab (inset of Fig. 1b) with finite thickness $h$ . With the open boundary, modes within the radiation continuum become resonances because they radiate by coupling to extended plane waves in the surrounding medium. Non-Hermitian perturbations need to be included in the Hamiltonian to account for the radiation loss.

To the leading order, radiation of the dipole mode can be described by adding an imaginary part $-i\gamma_{\mathrm{d}}$ to the Hamiltonian, while the quadrupole mode does not radiate due to its symmetry mismatch with the plane waves<sup>34</sup>.

Specifically, at the $\Gamma$ point the system has $C_2$ rotational symmetry (invariant under $180^\circ$ rotation around the $z$ axis), and the quadrupole mode does not couple to the radiating plane wave because the former is even $[\mathbf{E}(\mathbf{r}) = \hat{O}_{C_2}\mathbf{E}(\mathbf{r})]$ whereas the latter is odd $[\mathbf{E}(\mathbf{r}) = -\hat{O}_{C_2}\mathbf{E}(\mathbf{r})]$ under $C_2$ rotation<sup>30</sup>. The effective Hamiltonian is

$$
H _ {\mathrm {e f f}} = \left( \begin{array}{c c} \omega_ {0} & v _ {g} k \\ v _ {g} k & \omega_ {0} - i \gamma_ {\mathrm {d}} \end{array} \right), \tag {1}
$$

with complex eigenvalues

$$
\omega_ {\pm} = \omega_ {0} - i \frac {\gamma_ {\mathrm {d}}}{2} \pm v _ {g} \sqrt {k ^ {2} - k _ {c} ^ {2}}, \tag {2}
$$

where $\omega_0$ is the frequency at accidental degeneracy, $v_{g}$ is the group velocity of the linear Dirac dispersion in the absence of radiation, $k$ is the magnitude of the in-plane wavevector $(k_x,k_y)$ , and

4

$k_{c} \equiv \gamma_{\mathrm{d}} / 2v_{g}$ . Here, one of the three bands is decoupled from the other two and is not included in equation (1) (see section II of Supplementary Information). In equation (2), a ring defined by $k = k_{c}$ separates the $k$ space into two regions: inside the ring ( $k < k_{c}$ ), $\operatorname{Re}(\omega_{\pm})$ are dispersionless and degenerate; outside the ring ( $k > k_{c}$ ), $\operatorname{Im}(\omega_{\pm})$ are dispersionless and degenerate.

In the vicinity of $k_{c}$ , $\operatorname{Im}(\omega_{\pm})$ and $\operatorname{Re}(\omega_{\pm})$ exhibit square-root dispersion (also known as branching behavior) inside and outside the ring, respectively. Exactly on the ring ( $k = k_{c}$ ), the two eigenvalues $\omega_{\pm}$ are degenerate in both real and imaginary parts; meanwhile, the matrix $H_{\mathrm{eff}}$ becomes defective with an incomplete eigenspace spanned by only one eigenvector $(1, -i)^T$ that is orthogonal to itself under the unconjugated inner product.

This self-orthogonality is the definition of EPs; hence, here we have not just one EP, but a continuous ring of EPs. We call it an exceptional ring.

Fig. 1b,c show the complex eigenvalues of the PhC slab structure calculated numerically (symbols), which closely follow
the analytic model of equation (2) shown as solid lines in the figure. When the radius $r$ of the holes is tuned away from accidental degeneracy, the exceptional ring and the associated branching behavior disappear, as shown in Fig. S1. Several properties of the PhC slab contribute to the existence of this exceptional ring. Due to periodicity, one can probe the dispersion from two degrees of freedom, $k_{x}$ and $k_{y}$ , in just one structure. The open boundary provides radiation loss, and the $C_2$ rotational symmetry differentiates the radiation loss of the dipole mode and of the quadrupole mode.

We can rigorously show that the exceptional ring exists in realistic PhC slabs, not just in the effective Hamiltonian model. Our proof is based on the unique topological property of EPs: when the system parameters evolve adiabatically along a loop encircling an EP, the two eigenvalues

5

switch their positions when the system returns to its initial parameters $^{10,11,29,35}$ , in contrast to the typical case where the two eigenvalues return to themselves. Using this property, we numerically show, in Fig. S2 and section III of Supplementary Information, that the complex eigenvalues always switch their positions along every direction in the $k$ space, and therefore prove the existence of this exceptional ring. As opposed to the simplified effective Hamiltonian model, in a real PhC slab, the EP may exist at a slightly different magnitude of $k$ and for a slightly different hole radius $r$ along different directions in the momentum space, but this variation is small and negligible in practice (section IV of Supplementary Information).

To demonstrate the existence of the exceptional ring in such a system, we fabricate large-area periodic patterns in a $\mathrm{Si}_3\mathrm{N}_4$ slab ( $n = 2.02$ , thickness $180~\mathrm{nm}$ ) on top of $6\mu \mathrm{m}$ of silica ( $n = 1.46$ ) using interference photolithography<sup>34</sup>. Scanning electron microscope (SEM) images of the sample are shown in Fig. 2a, featuring a square lattice (periodicity $a = 336~\mathrm{nm}$ ) of air cylindrical holes with radius of $109~\mathrm{nm}$ .

We immerse the structure into an optical liquid and tune the refractive index of the liquid; accidental degeneracy in the Hermitian part is achieved when the liquid index is selected to be $n = 1.48$ . We perform angle-resolved reflectivity measurements (setup shown in Fig. 2b) between 0 and 2 degrees along the $\Gamma$ to X direction and the $\Gamma$ to M direction, for both $s$ and $p$ polarizations. The measured reflectivity for the relevant polarization is plotted in the upper panel of Fig.

2c, showing good agreement with numerical simulation results (lower panel), with differences coming from scattering of disorder, inhomogeneous broadening, and the uncertainty in the measurements of system parameters. The complete experimental result for both polarizations is shown in Fig. S3; the third and dispersionless band shows up in the other polarization, decoupled

6

from the two bands of interest.

The peaks of reflectivity (dark red color in Fig. 2c) follow the linear Dirac dispersion; this feature disappears for structures with different radii that do not reach accidental degeneracy (experimental results in Fig. S4). To understand the reflection peaks, we consider a generic two-by-two Hamiltonian $H$ with no assumption made about its matrix elements. We separate $H$ into a Hermitian part A and an anti-Hermitian part -iB (so that A and B are both Hermitian), and choose the basis in which A is diagonal:

$$
U H U ^ {\mathrm {T}} = \underbrace {\left( \begin{array}{l l} \Omega_ {1} & 0 \\ 0 & \Omega_ {2} \end{array} \right)} _ {\mathrm {A}} - \underbrace {i \left( \begin{array}{l l} \gamma_ {1} & \gamma_ {1 2} \\ \gamma_ {1 2} ^ {*} & \gamma_ {2} \end{array} \right)} _ {i \mathrm {B}} \xrightarrow {\text {e i g e n v a l u e s}} \left( \begin{array}{l l} \omega_ {+} & 0 \\ 0 & \omega_ {-} \end{array} \right). \tag {3}
$$

As before, we use $\omega_{\pm}$ to denote the complex eigenvalues of the Hamiltonian A - iB. The reflectivity in our system can be modeled using temporal coupled-mode theory (TCMT, with details in section V of Supplementary Information), where we show that the reflection peaks generally occur near the eigenvalues $\Omega_{1,2}$ of the Hermitian part A and are independent of the non-Hermitian part $-iB$ (Fig. S5 with details in section VI in Supplementary Information). Therefore, the peak locations in Fig.

2c (dark red) reveal information about only the Hermitian part of the Hamiltonian; the fact that they show linear Dirac dispersion indicates that we have successfully achieved accidental degeneracy in the eigenvalues of the Hermitian part, consistent with the simplified model in equation (1). In Fig. S6, we plot the $\Omega_{1,2}$ extracted from the reflectivity data through a more rigorous data analysis using TCMT (described below); the linear dispersion is indeed observed.

The eigenvalues of the Hamiltonian, $\omega_{\pm}$ , behave very differently from the reflectivity peaks. Simulation results (white lines in the lower panel of Fig. 2c) show $\mathrm{Re}(\omega_{\pm})$ are dispersionless at

7

small angles with a branch-point singularity around $0.31^{\circ}$ —consistent with the feature predicted by the simplified Hamiltonian in equation 2. In Fig. 2d, we compare the reflectivity spectra from simulations (with peaks indicated in red arrows) with the corresponding complex eigenvalues at three representative angles ( $0.8^{\circ}$ in blue, $0.31^{\circ}$ in green, and $0.1^{\circ}$ in magenta).

At $0.31^{\circ}$ , the two complex eigenvalues are degenerate, indicating an EP; however, the two reflection peaks do not coincide since they represent the eigenvalues of only the Hermitian part of the Hamiltonian, which does not have degeneracy here. The dip in reflectivity between the two peaks (marked as black arrows in Figs. 2 and 3) is the coupled-resonator-induced transparency (CRIT) that arises from the interference between radiation of the two resonances $^{36,37}$ , similar to electromagnetically induced transparency (EIT) $^{38}$ .

To extract the underlying Hamiltonian matrix and its eigenvalues from the measured reflectivity spectrum, we use TCMT to model the direct and the resonant reflection processes; the expression for reflectivity is given in equation (S.15) with the full derivation given in section V of the Supplementary Information. Fitting the reflectivity curves with the TCMT expression gives us the matrix elements of the Hamiltonian (as shown in equation (3)) that we use to calculate its eigenvalues; this procedure is the same as our approach in Ref.

39 except that here we handle multiple resonances simultaneously, accounting for their non-orthogonality and radiative coupling $^{40}$ . Fig. 3a compares the fitted and the measured reflectivity curves at three representative angles (with more comparison in Fig. S6a); the excellent agreement shows the validity of the TCMT equations. Underneath the reflectivity curves, we show the complex eigenvalues.

Repeating the fitting procedure for reflectivity spectrum measured at different angles, we

8

obtain the dispersion curves for all complex eigenvalues, which are plotted in Fig. 3b. Along both directions in $k$ space ( $\Gamma \to \mathrm{X}$ and $\Gamma \to \mathbf{M}$ ), the two bands of interest (shown in blue and red) exhibit the EP behavior predicted in equation (2): for $k < k_{c}$ the real parts are degenerate and dispersionless; for $k > k_{c}$ the imaginary parts are degenerate and dispersionless; for $k$ in the vicinity of $k_{c}$ branching features are observed in the real or imaginary part. In Fig.

3c, we plot the eigenvalues on the complex plane for both the $\Gamma \to \mathrm{X}$ and $\Gamma \to \mathbf{M}$ directions. We can see that in both directions, the two eigenvalues approach each other and become very close at certain $k$
point, which is a clear signature of the system being very near EP.

We have shown that non-Hermiticity arising from radiation can significantly alter fundamental properties of the system including the band structures and density of states; this effect becomes most prominent near EPs. The PhC slab described here provides a simple-to-realize platform for studying the influence of EPs on light-matter interaction, such as for single particle detection $^{41}$ and modulation of quantum noise $^{42}$ . The two-dimensional flat band also provides high density of states and therefore high Purcell factors.

The strong dispersion of loss in the vicinity of the $\Gamma$ point can improve the performance of large-area single-mode PhC lasers $^{43}$ . The deformation into exceptional ring can also occur for non-accidental Dirac points $^{44}$ . Further studies can advance the understanding of the connection between the topological property of Dirac points $^{2,45}$ and that of EPs $^{35}$ in general non-Hermitian wave systems, and this method of our study goes beyond photonics to phonons, electrons, and atoms.

# METHODS SUMMARY

Sample fabrication. The $\mathrm{Si}_3\mathrm{N}_4$ layer was grown with the low-pressure chemical vapor deposition

9

method on a $6\mu \mathrm{m}$ -thick cladding of $\mathrm{SiO}_2$ on the backbone of a silicon wafer (LioniX). Before exposure, the wafer was coated with a layer of polymer as anti-reflection coating, a thin layer of $\mathrm{SiO}_2$ as an intermediate layer for etching, and a layer of negative photoresist for exposure. The square lattice pattern was created with Mach-Zehnder interference lithography using a 325-nm He/Cd laser. The angle between the two arms of the laser beam was chosen for a periodicity of $336~\mathrm{nm}$ . After exposures, the pattern in the photoresist was transferred to $\mathrm{Si}_3\mathrm{N}_4$ by reactive-ion etching.

Experimental details. The source was a supercontinuum laser from NKT Photonics (SuperK-Compact). A polarizer selected $s$ - or $p$ -polarized light. The sample was immersed in a colorless liquid with tunable refractive indices (Cargille Labs). The sample was mounted on two perpendicular motorized rotation stages (Newport): one to orient the PhC to the $\Gamma$ -X or $\Gamma$ -M direction, and the other to determine the incident angle $\theta$ . The reflectivity spectra were measured with a spectrometer with spectral resolution of $0.02\mathrm{nm}$ (HR4000; Ocean Optics).

10

11

12

13

14

Supplementary Information is available in the online version of the paper.

Acknowledgments The authors thank Dr. Tim Savas for fabrication of the samples. Also, the authors thank Fan Wang, Yi Yang, Nick Rivera, Scott Skirlo, Dr. Owen Miller, and Prof. Steven G. Johnson for

15

helpful discussions. This work was partly supported by the Army Research Office through the Institute for Soldier Nanotechnologies under contract no. W911NF-07-D0004 and no. W911NF-13-D-0001. B.Z., L.L., and M.S. were partly supported by S3TEC, an Energy Frontier Research Center funded by the US Department of Energy under grant no. de-sc0001299. L.L. was supported in part by the Materials Research Science and Engineering Center of the National Science Foundation (award no. DMR-1419807). I.K. was supported in part by Marie Curie grant no. 328853-MC-BSiCS.

Author Contributions All authors discussed the results and made critical contributions to the work.

Author Information Reprints and permissions information is available at www.nature.com/reprints. The authors declare no competing financial interests. Correspondence and requests for materials should be addressed to B.Z. (email: bozhen@mit.edu).

Competing financial interests The authors declare no competing financial interests.

16

![](dt=2026-06-08/ht=11/1285e870a1841075add5eda94b638a1e33e48a597f7bd9fa157521468b955687.jpg)

![](dt=2026-06-08/ht=11/e006973d1828a1449ae5ca453fcbaf2fc864a4fe195139e89c875edcd2a7fd7f.jpg)

![](dt=2026-06-08/ht=11/cad7d3d0a95ff795ec7a3bd73e906a8590047dadc68f4e9c28b8d77b01260819.jpg)

![](dt=2026-06-08/ht=11/4495241565085e78826b1c56222e533980354926a197517466f708065d315f18.jpg)

![](dt=2026-06-08/ht=11/932db7e7fa9e5f9f14d6d6fe311fd735767a15e8b8a62672bee9deee7c1bf33a.jpg)

![](dt=2026-06-08/ht=11/94794c7fd79936a202b256e5579c47348b119805c9f9005fcbc83d86d56fc79d.jpg)

a, Band structure of a 2D PhC consisting of a square lattice of circular air holes. Tuning the radius $r$ leads to accidental degeneracy between a non-degenerate quadrapole band and two doubly degenerate dipole bands, resulting in two bands with linear Dirac dispersion (red and blue) and a flat band (gray). b,c, The real and imaginary parts of the eigenvalues of an open, and therefore non-Hermitian, system: a PhC slab with finite thickness $h$ . By tuning the radius, accidental degeneracy in the real part can be achieved, but the Dirac dispersion is deformed due to the non-Hermiticity.

The analytic model predicts that the real (imaginary) part of the eigenvalue stays as a constant within (outside) a ring in the wavevector space, indicating two flat bands in dispersion, with a ring of exceptional points (EPs) where both the real and the imaginary parts are degenerate.

In the upper panels, solid lines are from the analytic model and symbols are from numerical simulations: red squares represent the band connecting to the quadrapole mode at the center; blue circles represent the band connecting to the dipole mode at the center; and gray crosses represent the third band that is decoupled from the previous two due to symmetry. The 3D plots in the lower panels are from simulations.

17

![](dt=2026-06-08/ht=11/930b9a3528dc8c0eb3eb87c5a12758bc5b511ff1ca90c60938aa9c0a7ba4431a.jpg)

![](dt=2026-06-08/ht=11/e980c26db78d1c93e76fea55136c5dab5a252a1801957269d762d04db25ecfc3.jpg)

![](dt=2026-06-08/ht=11/6498c2a1f11c7d250da953296e33a650a77efa340240885a08d0994baa074f30.jpg)

![](dt=2026-06-08/ht=11/dc46ca0f89a506365a3766516cc4ac5788ec570aba51a904c63016877a10b352.jpg)

![](dt=2026-06-08/ht=11/6f46f5a2f03132a10aa9cd2c30b89cda59cf31cc3c5d62a06cdbca19ae58c561.jpg)

![](dt=2026-06-08/ht=11/908ce537e895eab4ff48b018c008314224d753599df28745657614a5764fda96.jpg)

![](dt=2026-06-08/ht=11/2c6e73d9b8a7164bfacacfc8669e44b20c98d6718f880ffb9c8dc8df4c45f739.jpg)

of the PhC samples: side view (upper panel) and top view (lower panel). b, Schematic drawing of the measurement setup. Light from a super-continuum source reflects off the PhC slab and is collected using a spectrometer. The incident angle is controlled using a precision rotationary stage. (BS: beam splitter; SP: spectrometer) c, Reflectivity spectrum of the sample measured experimentally (upper panel) and calculated numerically (lower panel) along the $\Gamma$ to X and the $\Gamma$ to M directions.

The peak location of reflectivity reveals the Hermitian part of the system, which forms Dirac dispersio
n due to accidental degeneracy. White lines in the lower panel indicate real part of the eigenvalues. d, Three line cuts of reflectivity from simulation results. Also shown are the complex eigenvalues (hollow circles) calculated numerically. At large angles $(0.8^{\circ})$ , the two resonances are far apart, so the reflectivity peaks (red arrows) are close to the actual positions of the complex eigenvalues.

However, at small angles $(0.3^{\circ}, 0.1^{\circ})$ , the coupling between resonances cause the resonance peaks (red arrows) to have much greater separations in frequencies compared to the complex eigenvalues. The black arrows mark the dips in reflectivity that correspond to the coupled-resonator induced transparency (CRIT, see text for details).

18

![](dt=2026-06-08/ht=11/d1e3d68c790bf06ecb73e08c427268abb3d6457666b9c640dcc46c6147072dd7.jpg)

![](dt=2026-06-08/ht=11/d80659a858e605069162cf7c21215cdf42c46eb3c6cd2201d8edfac531b0f04a.jpg)

![](dt=2026-06-08/ht=11/d9c200c8d4f42f4933c2deb40a9f8c6d32fbb27ae0df31bf1858b753eb6f8149.jpg)

![](dt=2026-06-08/ht=11/387687d80ff075dfacf64e0e9ad30c07b7d8ce2142f6f17040b891f8e6cfb9db.jpg)

![](dt=2026-06-08/ht=11/1339ab11351a5ae2c8f16bd72bfc3d00018a68dbdcc37f62d38ca025fddcc4cc.jpg)

![](dt=2026-06-08/ht=11/3e36a118e39a85f0b3cb436f96ada18f8f818fb72bad46a1681a8f52851ddec8.jpg)

![](dt=2026-06-08/ht=11/7418dad62b79fc19c17d2b3ddfde5e7c905c1a49f37f45fbc49469a44b182f86.jpg)

19

# Supplementary information

# Section I. Effective Hamiltonian of accidental Dirac points in Hermitian systems

To the leading order of approximation, the effective Hamiltonian for accidental Dirac cones in Hermitian systems (2D PhC) is written as a $3 \times 3$ matrix due to the involvement of three bands:

$$
H _ {\text {e f f}} ^ {\mathrm {2 D}} = \left( \begin{array}{c c c} \omega_ {0} & v _ {g} k _ {x} & v _ {g} k _ {y} \\ v _ {g} k _ {x} & \omega_ {0} & 0 \\ v _ {g} k _ {y} & 0 & \omega_ {0} \end{array} \right) \tag {S.1}
$$

that can be transformed into:

$$
U H _ {\text {e f f}} ^ {2 \mathrm {D}} U ^ {\mathrm {T}} = \left( \begin{array}{c c c} \omega_ {0} & v _ {g} k & 0 \\ v _ {g} k & \omega_ {0} & 0 \\ 0 & 0 & \omega_ {0} \end{array} \right) \tag {S.2}
$$

with the orthogonal transformation matrix

$$
U = \left( \begin{array}{c c c} 1 & 0 & 0 \\ 0 & \cos \theta & \sin \theta \\ 0 & - \sin \theta & \cos \theta \end{array} \right) \tag {S.3}
$$

Here, $\cos \theta = k_x / k$ , $\sin \theta = k_y / k$ , $k_{x,y}$ are in-plane wavevectors. After transformation, the $3 \times 3$ matrix becomes two isolated blocks: the upper $2 \times 2$ block gives the conical Dirac dispersion $(\omega = \omega_0 \pm v_g k)$ , while the lower block is the intersecting flat band $(\omega = \omega_0)$ .

# Section II. Effective non-Hermitian Hamiltonian of the exceptional ring

For a 3D PhC slab that has finite thickness, the two dipole modes become resonances with finite lifetime due to their coupling to radiation; therefore, their eigenvalues become complex $(\omega_0 - i\gamma_{\mathrm{d}})$ .

20

With $C_4$ rotational symmetry, these two dipole modes are identical to each other under an $90^\circ$ rotation and therefore share the same complex eigenvalue. Meanwhile, the quadrupole mode does not couple to radiation at the $\Gamma$ point due to symmetry mismatch, and to leading order its eigenvalue remains at $\omega_0$ . The effective non-Hermitian Hamiltonian of the 3D PhC slab becomes

$$
H _ {\mathrm {e f f}} ^ {\mathrm {3 D}} = \left( \begin{array}{c c c} \omega_ {0} & v _ {g} k _ {x} & v _ {g} k _ {y} \\ v _ {g} k _ {x} & \omega_ {0} - i \gamma_ {\mathrm {d}} & 0 \\ v _ {g} k _ {y} & 0 & \omega_ {0} - i \gamma_ {\mathrm {d}} \end{array} \right), \tag {S.4}
$$

which transforms to

$$
U H _ {\mathrm {e f f}} ^ {\mathrm {3 D}} U ^ {\mathrm {T}} = \left( \begin{array}{c c c} \omega_ {0} & v _ {g} k & 0 \\ v _ {g} k & \omega_ {0} - i \gamma_ {\mathrm {d}} & 0 \\ 0 & 0 & \omega_ {0} - i \gamma_ {\mathrm {d}} \end{array} \right) \tag {S.5}
$$

with the same matrix $U$ as in equation S.3. The upper $2 \times 2$ block is the $H_{\mathrm{eff}}$ we refer to in equation (1) that gives rise to an exceptional ring, while the lower block is the intersecting flat band.

# Section III, Existence of exceptional points along every direction in momentum space

In this section, we demonstrate that EPs exist in all directions in the $k$ space, not only for a simplified Hamiltonian (equation 1), but also for realistic structures. To prove their existence, we use the unique topological property of EPs: when the system evolves adiabatically in the parameter space around an EP, the eigenvalues will switch their positions at the end of the loop $^{10,35}$ . In our system, the parameter space in which we choose to evolve the eigenfunctions is three-dimensional, consisting of the two in-plane wavevectors $(k_x, k_y)$ and the radius of the air holes $r$ , as shown in Fig. S2a. Here, $r$ can also be other parameters, like the refractive index of the PhC slab $(n)$ , the

21

periodicity of the square lattice $(a)$ , or the thickness of the slab $(h)$ . For simplicity of this demonstration, we choose $r$ as the varying parameter while keeping all other parameters $(n, a, \text{and } h)$ fixed throughout.

First, we compare the evolution of the eigenvalues when the system parameters follow (1) a loop that does not enclose an EP, and (2) one that encloses an EP. Following the loop $A \rightarrow B \rightarrow C \rightarrow D \rightarrow A$ in Fig. S2a,b that does not enclose an EP (point $\mathrm{E_P}$ ), we see that the complex eigenvalues come back to themselves at the end of the loop (Fig. S2c where the red dot and the blue dot return to their initial positions at the end of the loop). However, following the loop $A' \rightarrow B' \rightarrow C' \rightarrow D' \rightarrow A'$ in Fig.

S2d, which encloses an EP (point $\mathrm{E_P}$ ), we see that the complex eigenvalues switch their positions in the complex plane (Fig. S2f where the red dot and the blue dot switch their positions). This switching of the eigenvalues shows the existence of an EP along the $\Gamma$ to $\mathbf{X}$ direction, at some particular value of $k_x$ and some particular value of radius $r$ . This shows the existence of the EP without having to locate the exact parameters of $k_x$ and $r$ at which the EP occurs.

Similarly, we can evolve the parameters along any direction $\theta = \tan (k_y / k_x)$ in the $k$ space and check if an EP exists along this direction or not.

As two examples, we show the evolution of the complex eigenvalues when we evolve the parameters along the $\theta = \pi /8$ direction following the loop $A^{\prime \prime}\rightarrow B^{\prime \prime}\rightarrow C^{\prime \prime}\rightarrow D^{\prime \prime}\rightarrow A^{\prime \prime}$ and along the $\theta = \pi /4$ direction following the loop $A^{\prime \prime \prime}\to B^{\prime \prime \prime}\to C^{\prime \prime \prime}\to D^{\prime \prime \prime}\to A^{\prime \prime \prime}$ in Fig. S2g,h.

In both cases, we observe the switching of the eigenvalues, showing the existence of an EP along these two directions. The same should hold for every direction in $k$ space.

22

The above calculations show that for every direction $\theta$ we examined in the $k$ space, there is always a particular combination of $k_{c}$ and $r_c$ , which supports an EP. However, we note that i
n general, different directions can have different $k_{c}$ and different $r_c$ , so the exceptional ring for the realistic PhC slab structure is parameterized by $k_{c}(\theta)$ and $r_c(\theta)$ . This angular variation of $k_{c}(\theta)$ and $r_c(\theta)$ can be described by introducing higher order corrections in the effective Hamiltonian, which we examine in the next section.

# Section IV, Generalization of the effective Hamiltonian

Here, we generalize the effective Hamiltonian in equation (1) and (S.5). First, the radiation of the quadrapole mode is zero only at the $\Gamma$ point; away from the $\Gamma$ point, the quadrapole mode has a $\vec{k}$ -dependent radiation that is small but non-zero, which we denote with $\gamma_{\mathrm{q}}$ . Second, we consider possible deviation from accidental degeneracy in the Hermitian part, with a frequency walk-off $\delta$ . With these two additional ingredients, the effective Hamiltonian becomes

$$
\left( \begin{array}{c c} \omega_ {0} + \delta & v _ {g} k \\ v _ {g} k & \omega_ {0} \end{array} \right) - i \left( \begin{array}{c c} \gamma_ {\mathrm {q}} & \sqrt {\gamma_ {\mathrm {q}} \gamma_ {\mathrm {d}}} \\ \sqrt {\gamma_ {\mathrm {q}} \gamma_ {\mathrm {d}}} & \gamma_ {\mathrm {d}} \end{array} \right), \tag {S.6}
$$

with complex eigenvalues of

$$
\omega_ {\pm} = \omega_ {0} + \frac {\delta}{2} - i \frac {\gamma_ {\mathrm {q}} + \gamma_ {\mathrm {d}}}{2} \pm \sqrt {\left(v _ {g} k - i \sqrt {\gamma_ {\mathrm {q}} \gamma_ {\mathrm {d}}}\right) ^ {2} - \left(\frac {\gamma_ {\mathrm {d}} - \gamma_ {\mathrm {q}}}{2} - i \frac {\delta}{2}\right) ^ {2}}, \tag {S.7}
$$

which generalizes equations (1) and (2). We note that the off-diagonal term $\sqrt{\gamma_{\mathrm{q}}\gamma_{\mathrm{d}}}$ in equation (S.6) is required by energy conservation and time-reversal symmetry<sup>40,46</sup>, as we will discuss more in the next section. Equation (S.7) shows that EP occurs when the two conditions

$$
\left\{ \begin{array}{l} k = (\gamma_ {\mathrm {d}} - \gamma_ {\mathrm {q}}) / (2 v _ {g}) \approx \gamma_ {\mathrm {d}} / 2 v _ {g}, \\ \delta = 2 \sqrt {\gamma_ {\mathrm {d}} \gamma_ {\mathrm {q}}}, \end{array} \right. \tag {S.8}
$$

23

are satisfied. In the region of momentum space of interest, $\gamma_{\mathrm{q}}$ is much smaller than $\gamma_0$ (this can be seen, for example, from the imaginary parts of Fig. S1a,c), so the first condition becomes $k_{\mathrm{c}}\approx \gamma_{\mathrm{d}} / 2v_{g}$ , same as in the simplified model. For a given direction $\theta$ in the $k$ space (as discussed in the previous section), we can vary the magnitude $k$ and the radius $r$ to find the $k_{c}(\theta)$ and $r_c(\theta)$ where these two conditions are met simultaneously.

We can now analyze the angular dependence of $k_{c}(\theta)$ and $r_{c}(\theta)$ without having to find their exact values. The first condition of equation (S.8) says that the angular dependence of $k_{c}(\theta)$ comes from $\gamma_{\mathrm{d}}$ and $v_{g}$ ; in the PhC slab structure here, we find that $\gamma_{\mathrm{d}}$ varies by about 20% as the angle $\theta = \tan(k_{y}/k_{x})$ is varied; while $v_{g}$ remains almost the same; therefore, $k_{c}(\theta)$ potentially varies by around 10% along the exceptional ring. For the second condition of equation (S.

8), we have $\gamma_{\mathrm{d}} \approx 5 \times 10^{-3}\omega_{0}$ and $\gamma_{\mathrm{q}} \approx 5 \times 10^{-5}\omega_{0}$ for our PhC slab structure, so $\delta_{c} = 2\sqrt{\gamma_{\mathrm{d}}\gamma_{\mathrm{q}}} \approx 1 \times 10^{-3}\omega_{0}$ . Again, $\gamma_{\mathrm{d}}$ and $\gamma_{\mathrm{q}}$ vary by about 20% as the angle $\theta$ is changed, so $\delta_{c}$ can change by around $2 \times 10^{-4}\omega_{0}$ .

Empirically, we find that a change of $\delta$ by $2 \times 10^{-4}\omega_{0}$ corresponds to a change in the radius $r$ of around 0.06 nm, which is the estimated range of variation for $r_{c}(\theta)$ of all $\theta \in [0,2\pi)$ . This angular variation is much smaller than our structure can resolve in practice, since the radii of different holes within one fabricated PhC slab will already differ by more than 0.06 nm. So, in practice a given fabricated structure can be close to EP along all different directions $\theta$ , but is unlikely to be an exact EP for any direction.

# Section V, Temporal Coupled Mode Theory (TCMT)

To connect the Hamiltonian of the resonances to the experimentally measured reflectivity, we resort to temporal coupled-mode theory (TCMT) $^{30,47}$ . Here, we consider a very general setup with an

24

arbitrary number of resonances in the PhC slab. The time evolution of these $n$ resonances, whose complex amplitudes are denoted by an $n \times 1$ column vector $A$ , is described by the Hamiltonian $H$ and a driving term,

$$
\frac {d A}{d t} = - i H A + K ^ {\mathrm {T}} s _ {+}, \tag {S.9}
$$

where the Hamiltonian is an $n\times n$ non-Hermitian matrix

$$
H = \Omega - i \Gamma - i \gamma_ {\mathrm {n r}}, \tag {S.10}
$$

with $\Omega$ denoting its Hermitian part, $-i\Gamma$ denoting its anti-Hermitian part from radiation loss, and $-i\gamma_{\mathrm{nr}}$ its anti-Hermitian part from non-radiative decays including absorption and surface roughness. For simplicity, we consider the same non-radiative loss for all resonances, so $\gamma_{\mathrm{nr}}$ is a real number instead of a matrix.

Reflectivity measurements couple the $n$ resonances to the incoming and outgoing planewaves whose complex amplitudes we denote by two $2 \times 1$ column vectors, $s_{+}$ and $s_{-}$ . The direct reflection and transmission of the planewaves through the slab (in the absence of resonances) are described by a $2 \times 2$ complex symmetric matrix $C$ , and

$$
s _ {-} = C s _ {+} + D A, \tag {S.11}
$$

where $D$ and $K$ in equation (S.9) are $2 \times n$ complex matrices denoting coupling between the resonances and the planewaves. We approximate the direct scattering matrix $C$ by that of a homogeneous slab whose permittivity is equal to the spatial average of the $\mathrm{PhC~slab}^{39,48,49}$ . Lastly, outgoing planewaves into the silica substrate are reflected at the silica-silicon interface, so

$$
s _ {2 +} = e ^ {2 i \beta h _ {s}} r _ {2 3} s _ {2 -}, \tag {S.12}
$$

25

where $h_s$ is the thickness of the silica substrate with refractive index $n_{\mathrm{s}} = 1.46$ , $\beta = \sqrt{n_{\mathrm{s}}^2 \omega^2 / c^2 - |\mathbf{k}_{\parallel}|^2}$ is the propagation constant in silica, and $r_{23}$ is the Fresnel reflection coefficient between silica and the underlying silicon. The formalism described above is the same as Ref. 39 except that here we describe the $n$ resonances in a more general setting that accounts for their coupling (off-diagonal terms of $H$ ) and therefore their non-orthogonality.

For steady state with $e^{-i\omega t}$ time dependence, we solve for vector $A$ from equation (S.9) to get the scattering matrix of the whole system that includes both direct and resonant processes,

$$
s _ {-} = (C + C _ {\mathrm {r e s}}) s _ {+}, \tag {S.13}
$$

where the effect of the $n$ resonances is captured in a $2 \times 2$ matrix

$$
C _ {\mathrm {r e s}} = i D (\omega - H) ^ {- 1} K ^ {\mathrm {T}}. \tag {S.14}
$$

We can solve equation (S.12) and equation (S.13) to obtain the reflectivity

$$
R _ {\mathrm {T C M T}} = \left| \frac {s _ {1 -}}{s _ {1 +}} \right| ^ {2}. \tag {S.15}
$$

In this expression, the only unknown is $C_{\mathrm{res}}$ . Therefore, by comparing the experimentally measured reflectivity spectrum $R(\omega)$ and the one given by TCMT in equation (S.15), we can extract the unknown parameters in the resonant scattering matrix $C_{\mathrm{res}}$ and obtain the eigenvalues of the Hamiltonian $H$ .

The remaining task is to write $C_{\mathrm{res}}$ using as few unknowns as possible so that the eigenvalues of $H$ can be extracted unambiguously. In equation (S.14), there are a large number of unknowns in the matrix elements of $H
, D,$ and $K$ , but there is much redundancy because the matrix elements are not independent variables and because $C_{\mathrm{res}}$ is independent of the basis choice. Below, we show

26

that we can express $C_{\mathrm{res}}$ with only $2n + 1$ unknown real numbers, and these $2n + 1$ real numbers are enough to determine the $n$ complex eigenvalues of $H$ .

First, we normalize the amplitudes of $A$ and $s_{\pm}$ such that their magnitude squared are the energy of the resonances per unit cell and the power of the incoming/outgoing planewaves per unit cell, respectively. Then, energy conservation, time-reversal symmetry, and $C_2$ rotational symmetry of the PhC slab<sup>40,46</sup> require the direct scattering matrix to satisfy $C^\dagger = C^* = C^{-1}$ and the coupling matrices to satisfy $D^\dagger D = 2\Gamma$ , $K = D$ , and $CD^* = -D$ . It follows that the matrix $\Gamma$ is real and symmetric. Next, using the Woodbury matrix identity and these constrains, we can rewrite equation (S.14) as

$$
C _ {\text {r e s}} = - 2 W (2 + W) ^ {- 1} C, \tag {S.16}
$$

where $W \equiv iD(\omega - \Omega + i\gamma_0)^{-1}D^\dagger$ is a 2-by-2 matrix. We note that the matrix $W$ , and therefore the matrix $C_{\mathrm{res}}$ , is invariant under a change of basis for the resonances through any orthogonal matrix $U$ (where $\Omega$ is transformed to $U\Omega U^{-1}$ , and $D$ is transformed to $DU^{-1}$ ). Therefore, we are free to choose any basis. Given the expression for $W$ , we choose the basis where $\Omega$ is diagonal, so $\Omega_{ij} = \Omega_j\delta_{ij}$ , with $\{\Omega_j\}_{j=1}^n$ being the eigenvalues of $\Omega$ .

To proceed further, we note that the PhC slab sits on a silica substrate with $n_{\mathrm{s}} = 1.46$ and is immersed in a liquid with $n = 1.48$ , so the structure is nearly symmetric in $z$ direction. The mirror symmetry requires the coupling to the two sides to be symmetric or anti-symmetric<sup>30</sup>,

$$
\frac {D _ {1 j}}{D _ {2 j}} \equiv \sigma_ {j} = \pm 1, \quad j = 1, \dots , n, \tag {S.17}
$$

where $\sigma_{j} = 1$ for TE-like resonances and $\sigma_{j} = -1$ for TM-like resonances, in the convention where $(E_x,E_y)$ determines the phase of $A_{j}$ and $s_\pm$ . Then, the diagonal elements of $\Gamma$ are related

27

to $D$ by $\Gamma_{jj} \equiv \gamma_j = |D_{1j}|^2$ , and in this basis we have

$$
W = \sum_ {j = 1} ^ {n} \frac {i \gamma_ {j}}{\omega - \Omega_ {j} + i \gamma_ {\mathrm {n r}}} \left( \begin{array}{c c} 1 & \sigma_ {j} \\ \sigma_ {j} & 1 \end{array} \right). \tag {S.18}
$$

This completes our derivation. Equations (S.16) and (S.18) provide an expression for $C_{\mathrm{res}}$ that depends only on $2n + 1$ unknown non-negative real numbers: the $n$ eigenvalues $\{\Omega_j\}_{j=1}^n$ of the Hermitian matrix $\Omega$ , the $n$ diagonal elements $\{\gamma_j\}_{j=1}^n$ of the real-symmetric radiation matrix $\Gamma$ in the basis where $\Omega$ is diagonal, and the non-radiative decay rate $\gamma_{\mathrm{nr}}$ .

At each angle and each polarization, we fit the experimentally measured reflectivity spectrum $R(\omega)$ to the TCMT expression equation (S.15) to determine these $2n + 1$ unknown parameters. Fig. 3a and Fig. S6a show the comparison between the experimental reflectivity spectrum and the fitted TCMT reflectivity spectrum at some representative angles. The near-perfect agreement between the two demonstrates the validity of the TCMT model.

To obtain the eigenvalues of the Hamiltonian $H$ , we also need to know the off-diagonal elements of $\Gamma$ . From $D^{\dagger}D = 2\Gamma$ and $D_{1j} / D_{2j} = \sigma_j$ , we see that $\Gamma_{ij} = 0$ when resonance $i$ and resonance $j$ have different symmetries in $z$ (i.e. when $\sigma_i\sigma_j \neq 1$ ), and that $\Gamma_{ij} = \pm \sqrt{\gamma_i\gamma_j}$ when $\sigma_i\sigma_j = 1$ . In the latter case, the sign of $\Gamma_{ij}$ depends on the choice of basis; the eigenvalues of $H$ are independent of the basis choice, so to calculate the eigenvalues of $H$ , we can simply take the positive root for all of the non-zero off-diagonal elements of $\Gamma$ .

We note that the model Hamiltonians introduced previously, such as equation (1) in the main text and equations (S.5) and (S.6) above, are all special cases of the general Hamiltonian in equation (S.10) that we consider in the TCMT formalism in this section. Those model Hamiltonians fix the number of resonances, assume simple forms of their parameters, and choose a specific basis

28

in order to convey the physical picture. Meanwhile, the TCMT formalism in this section does not make such assumptions (aside from basic principles such as energy conservation and time-reversal symmetry) so that it can be used as an unbiased method for analyzing the experimental data.

We also note that the TCMT equations, from (S.9) to (S.17), are all written in the general matrix notation where one is free to choose any basis for the Hamiltonian $H$ ; we only make the specific basis choice (the basis where $\Omega$ is diagonal) in equation (S.18) in order to simplify the expression for matrix $W$ , and in equation (3) of the main text in order to emphasize the eigenvalues $\Omega$ . Meanwhile, physical observables, such as $C_{\mathrm{res}}$ in equation (S.14), $R_{\mathrm{TCMT}}$ in equation (S.15), and the eigenvalues of $H$ , are all independent of the basis choice.

# Section VI, Reflection peaks and CRIT

In this section, we use a simplified scenario (a special case of the previous section) to illustrate that the peaks of the reflectivity generally follow the eigenvalues of $\Omega$ and to show the coupled-resonator-induced-transparency (CRIT).

Consider a simplified scenario with two resonances of the same symmetry in $z$ and without non-radiative loss (i.e. $n = 2$ , $\sigma_{1} = \sigma_{2}$ , $\gamma_{\mathrm{nr}} = 0$ ), and ignore the direct Fresnel reflection between the dielectric layers (so that the direct scattering matrix $C$ has no reflection, and that $s_{2+} = 0$ in equation (S.12)). In such case, equations (S.15) (S.16) (S.18) give

$$
R _ {\mathrm {T C M T}} (\omega) = \frac {1}{1 + f ^ {2} (\omega)}, \quad \frac {1}{f (\omega)} = \frac {\gamma_ {1}}{\omega - \Omega_ {1}} + \frac {\gamma_ {2}}{\omega - \Omega_ {2}}. \tag {S.19}
$$

We immediately see that the reflectivity reaches its maximal value of 1 when $\omega = \Omega_{1}$ or $\omega = \Omega_{2}$ , namely at the eigenvalues of the matrix $\Omega$ . Another feature we can observe is that the reflectivity is 0 when $\omega = (\gamma_{1}\Omega_{2} + \gamma_{2}\Omega_{1}) / (\gamma_{1} + \gamma_{2})$ , which is a phenomenon called coupled-resonator-induced-

29

transparency (CRIT) $^{36,37}$ .

We emphasize that the reflectivity peaks are different from the complex eigenvalues of the Hamiltonian $H$ . Consider a simple example with $\Omega_{1,2} = \omega_0 \pm b$ , and $\gamma_1 = \gamma_2 = b$ . The reflection peaks at $\Omega_{1,2} = \omega_0 \pm b$ , while the two complex eigenvalues are degenerate at $\omega_+ = \omega_- = \omega_0 - ib$ , whose real part is in the middle of the two reflection peaks. This explains the reflectivity from the PhC slabs at $0.3^\circ$ shown in Fig. 2d and Fig. 3a, where the degenerate complex eigenvalues of the system are in between the two reflection peaks.

In Fig. S5, we use some examples to illustrate the difference between the reflectivity peaks and the complex eigenvalues. Fig. S5a shows the case when there is only one resonance (removing one of the two terms in equation (S.19)) with complex eigenvalue $\omega_0 - i\gamma$ ; in this case, the reflection peak (red arrow) is at the same position as the real part of the complex eigenvalue. In contrast, Fig. S5b shows the case when there are two resonances (equation (S.19)) with $\Omega_{1,2}$ fixed at $\omega_0 \pm b$ ; as we vary $\gamma_{1,2}$ , the complex eigenvalues (circles) vary accordingly, whereas the reflectivity peaks (red arrows) always sh
ow up at $\Omega_{1,2}$ .

For the realistic PhC slab structure in our experiment, the reflectivity is described by the more general expression, equations (S.15), but equation (S.19) serves as a qualitative approximation near the frequency range of interest, because the far-away resonances do not contribute much, the nonradiative loss is small, and the Fresnel reflection between the dielectric layers (liquid, $\mathrm{Si}_3\mathrm{N}_4$ , silica, and silicon) is small. So, we can still see the general trend that the reflectivity peaks follow the eigenvalues of $\Omega$ (as evident by comparing Fig. S3 and Fig. S6b), and we can still see reflectivity dips for CRIT (such as in Fig. 3a).

30

![](dt=2026-06-08/ht=11/e1a3fe50180dd8b05ab24cecb0c20f788a26b11915958febc46da01fcb433d9a.jpg)

![](dt=2026-06-08/ht=11/018e8d9b65fcae129ed2af7b095ba3b93cd65036c345bfdcd0205261d8018c91.jpg)

![](dt=2026-06-08/ht=11/bd7c719b3c9f00b0301cdefb8148981de33757422e41866fd9451b9cf8271d68.jpg)

31

![](dt=2026-06-08/ht=11/7598157bd106edd17f2dfe0ab1e1fcc0404cf361c9b038b327c5bb648eb4e0b2.jpg)

![](dt=2026-06-08/ht=11/9183c7d3487bdbda231705fcdae59ec732836e04bda17092830f63a9de118868.jpg)

![](dt=2026-06-08/ht=11/fee9cba9f03d3be8a6eadd80dbbd556576aea4d0c8c573eac2bc81ec89a49f6a.jpg)

![](dt=2026-06-08/ht=11/ffc338fbd1953f31324fa9667abb66f2405f87fd85f4d7841c85a717b6628b4d.jpg)

![](dt=2026-06-08/ht=11/0da14e7075c94df3896b76b380cdd4759474ac8ad3df8cc926d630a2562b8630.jpg)

![](dt=2026-06-08/ht=11/403f39d4fc846363f0fbadc45357fd831ed02e475411b8d5246a1b08ad8dbb61.jpg)

![](dt=2026-06-08/ht=11/d26bd142efd837477787800f3f53b08bb1d79b2975d4e23a84cc7a45673aab96.jpg)

![](dt=2026-06-08/ht=11/46e4216928e53ea2896578de3f4206f750ac373d36e117528cd42e1b44a688dd.jpg)

32

FIG. S2. Existence of EP along every direction in the momentum space for the realistic PhC slab structure. a, A loop is created in the parameter space of the structure ( $A \to B \to C \to D \to A$ ), which does not enclose the EP of the system (point $\mathrm{E_p}$ ). Here, $r$ is the radius of the air holes, and $k_{x,y}$ are the in-plane wavevectors. b, For each point along the loop, we numerically calculate the eigenvalues of the PhC slab with the corresponding hole radius at the corresponding in-plane wavevector.

c, The complex eigenvalues return to their initial positions at the end of the loop (namely, the blue dot and the red dot come back to themselves) when the system parameters come back to point A. d,e,f, Another loop is created ( $A' \to B' \to C' \to D' \to A'$ ), which encloses an EP of the system (the same point $\mathrm{E_p}$ as in a.). Following this new loop, the two eigenvalues switch their positions at the end of the loop (namely, the blue dot and the red dot switch their positions) when the system parameters come back to point $A'$ .

g,h, The two complex eigenvalues always switch their positions when we choose the right loops along other directions in the momentum space ( $\Gamma$ to N in g and $\Gamma$ to M in h).

33

![](dt=2026-06-08/ht=11/756a3d77ce4deffb42c64661aaa717620206666aac6486d0a4ec4a4209358703.jpg)

![](dt=2026-06-08/ht=11/db85901d8175fb88d6ad9820aaf138c237d4a972800c615dd6b429540162fba5.jpg)

34

![](dt=2026-06-08/ht=11/9f6aa55d522228f77ab52c62e085ed931f0abbb3c044839a98a28ae66802d53b.jpg)

![](dt=2026-06-08/ht=11/b3275956ff47d6b27b71bf761e55d57279b7d4c8d2c535b0e80d207c46e9dc85.jpg)

![](dt=2026-06-08/ht=11/28673d9e392bf3aaeebc3a2c156895a11452a4d469e0fdc84ab7e526301394ce.jpg)

35

![](dt=2026-06-08/ht=11/5521c8c72f4efb2ccfd87b05b155bd17bf7266508e6e26c835df6cc145bab4a0.jpg)

![](dt=2026-06-08/ht=11/d41614576f537dde3a8ac28e7c60f580aef4b38908b56e3991927d0372fd1085.jpg)

![](dt=2026-06-08/ht=11/b0edbccfceb5a14c330c740cbab0e38af80035bad51c859280811074a1963610.jpg)

![](dt=2026-06-08/ht=11/2eb05e3c16d869cf2692435c29cee03461a8482dfaed5f5ad740560441e521f3.jpg)

36

![](dt=2026-06-08/ht=11/bca029d8226e838a506b3fc17c9f5c0fe86a8a0a9a5038e9eeac064e54984522.jpg)

![](dt=2026-06-08/ht=11/e0f761d9e7dc691cf58740996be4f3e6adb556c9cbe309b3200caa960813ccad.jpg)

37