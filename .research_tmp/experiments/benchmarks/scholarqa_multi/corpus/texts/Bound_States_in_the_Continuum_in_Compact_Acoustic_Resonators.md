# Bound States in the Continuum in Compact Acoustic Resonators

Ilya Deriy $^{ID}$ , $^{*}$ Ivan Toftul, Mihail Petrov, $^{\ddagger}$ and Andrey Bogdanov $^{ID}$ School of Physics and Engineering, ITMO University, 191002 St. Petersburg, Russia

![](images/215dd91bb7b5fdbc706217c377ba76a9da0d4392dfe09e065d8b4c1764a70845.jpg)

(Received 7 May 2021; accepted 27 January 2022; published 23 February 2022)

We reveal that finite-size solid acoustic resonators can support genuine bound states in the continuum (BICs) completely localized inside the resonator. The developed theory provides the multipole classification of such BICs in the resonators of various shapes. It is shown how breaking of the resonator's symmetry turns BICs into quasi-BICs manifesting themselves in the scattering spectra as high-Q Fano resonances. We believe that the revealed novel states will push the performance limits of acoustic devices and will serve as high-Q building blocks for acoustic sensors, antennas, and topological acoustic structures.

DOI: 10.1103/PhysRevLett.128.084301

Bound states in the continuum (BICs) are the non-radiating states of an open system with a spectrum embedded in the continuum of the radiating modes of the surrounding space $[1,2]$ . BICs were first predicted in quantum mechanics by von Neumann and Wigner in 1929 $[3]$ but shortly after they were extended to the wave equations in general as their specific solutions. As a result, BICs were found in various fields of physics such as atomic physics, hydrodynamics, acoustics, and mechanics $[4–8]$ . The zero radiative losses lead to a diverging radiative quality factor (Q factor) making BICs extremely prospective for the energy localization and enhancement of the incident fields. Recently, these unique properties of BICs have been actively utilized in photonics, where they have already proven themselves as an effective platform for lasing, polaritonic, sensing, and optical harmonic generation applications $[9–14]$ .

Decoupling the resonance from all open scattering channels one can obtain a genuine BIC, which becomes possible only if the number of the adjusting parameters is more than the number of the scattering channels. A typical example of a system with a finite number of scattering channels is a resonator coupled to one or several waveguide modes or infinite periodic structure $[15–19]$ . For finite size structures the number of scattering channels is infinite, and the existence of BICs in such systems is prohibited by the nonexistence theorem $[1]$ . The only exception is the structures surrounded by a completely opaque shell providing decoupling of the internal resonances from the outside radiation continuum, which in quantum mechanics corresponds to infinite high potential barriers, in acoustics to hard-wall boundaries, and in optics to perfect conducting walls or epsilon-near-zero barriers $[20–23]$ . Finding a genuine BIC in compact systems is a challenging fundamental problem, and its solution would make possible the implementation of subwavelength high-Q resonators having broad range of potential applications.

In this Letter, a back door in the “nonexistence theorem” is revealed. While BICs are prohibited in finite photonic, acoustic, and quantum mechanical systems, we show that acoustics is relieved of these constraints for some special cases. We propose a genuine acoustic BIC in compact solid resonators placed in nonviscous fluids (gas or liquid). The origin of the unexplored BICs is illustrated in Fig. 1. In acoustics, in contrast to photonics and quantum mechanics, there are two types of waves: (i) pressure waves with longitudinal polarization ( $u\parallel k$ ) and (ii) shear waves with transversal polarization ( $u\perp k$ ) [24]. Here, u is the displacement vector and k is the wave vector. While in solids, both waves coexist, nonviscous fluids host only pressure waves which are longitudinal. In the general case of an arbitrary shaped solid resonator, all the eigenmodes are hybrid, containing longitudinal and transverse components, and thus, are coupled to the radiation continuum. However, one can find specific shapes of the resonators allowing purely torsion modes for which displacement is tangential at each point of the resonator boundary. The torsion

![](images/229ed9b836979bf09aeb1072929f5248587f903c385c358c189b39f5698c4e39.jpg)

<details>
<summary>text_image</summary>

Longitudinal modes
(leaky resonance)
Transverse modes
(BIC)
Hybrid modes
(leaky resonance)
κ ≠ 0
κ = 0
κ ≠ 0
Radiation Continuum u || k
</details>

FIG. 1. Polarization of eigenmodes in a solid acoustic resonator. The transversal modes ( $u \perp k$ ) do not couple to the radiation continuum forming a bound state in the continuum. Here, $\kappa$ is the coupling coefficient, u is the displacement vector, and k is the wave vector.

oscillations do not produce pressure on the surrounding fluid being completely decoupled from the radiation continuum. Their energy remains perfectly confined inside the resonator forming genuine acoustic BICs in contrast to Refs. [25] and [26] where only a quasi-BIC has been discussed.

We may designate the proposed nonradiating states as polarization-protected acoustic BICs as they are possible exclusively due to the fact that nonviscous fluids support longitudinal waves only, while solids host both longitudinal and transversal waves. Because of similar reasons, the shear modes in a solid slab appear to be nonradiating $[27,28]$ . To be frank, BICs in compact resonators can exist even in optics, for example, in the form of the radial plasmonic oscillations in a metal spherical particle at the plasma frequency. In this case, the longitudinal plasmonic oscillations are not coupled to the far-field electromagnetic radiation which is purely transverse. However, the observation of these modes is hindered due to high losses in plasmonic nanostructures.

We start with considering a problem of eigenmodes in a rigid sphere of radius $a$ in order to show rigorously that the polarization-protected BICs exist in compact resonators. We assume that the sphere is made of a solid isotropic material surrounded by a gas or fluid environment. The displacement field $\mathbf{u}(\mathbf{r})$ inside and outside the sphere obeys the Helmholtz equation [24]

$$
\Delta \mathbf {u} _ {i} ^ {j} (\mathbf {r}) + (k _ {i} ^ {j}) ^ {2} \mathbf {u} _ {i} ^ {j} (\mathbf {r}) = 0. \tag {1}
$$

Here, the lower index i = s, p encodes the displacements of shear or pressure waves, respectively; the upper index $j = \{in, out\}$ corresponds to the fields inside and outside of the resonator, respectively; $k_{i}^{j} = \omega / c_{i}^{j}$ is the wave vector; and $c_{i}^{j}$ is the velocity of displacement waves. The boundary conditions at the surface of a solid sphere placed in gas or fluid can be written in spherical coordinates $(r, \theta, \varphi)$ as follows [29]:

$$
\sigma_ {r r} ^ {\text { in }} = - p ^ {\text { out }}, \quad \sigma_ {r \theta} ^ {\text { in }} = 0, \quad \sigma_ {r \varphi} ^ {\text { in }} = 0, \quad u _ {r} ^ {\text { in }} = u _ {r} ^ {\text { out }}, \tag {2}
$$

where $\sigma_{ij} = \lambda \delta_{ij}\mathrm{Tr}\hat{\varepsilon} +\mu \varepsilon_{ij}$ is the Cauchy stress tensor, $2\hat{\varepsilon} = [\nabla \mathbf{u} + (\nabla \mathbf{u})^{\mathrm{T}}]$ is the strain tensor, and $p^{\mathrm{out}}$ is the gas pressure, which is connected to the displacement field as $\rho^{\mathrm{out}}\omega^{2}\mathbf{u}^{\mathrm{out}} = \nabla p^{\mathrm{out}}$ (see Supplemental Material [30]). We also assume that outside the sphere the solution has a form of the outgoing waves. Based on that, the solutions of the vector Helmholtz equation (1) can be written in terms of the vector spherical harmonics [31]

$$
\begin{array}{l} \mathbf {u} ^ {j} (\mathbf {r}) = \sum_ {\ell m} a _ {\ell m} ^ {j} \mathbf {M} _ {\ell m} (\mathbf {r}, k _ {s} ^ {j}) \\ + b _ {\ell m} ^ {j} \mathbf {N} _ {\ell m} (\mathbf {r}, k _ {s} ^ {j}) + c _ {\ell m} ^ {j} \mathbf {L} _ {\ell m} (\mathbf {r}, k _ {p} ^ {j}). \tag {3} \\ \end{array}
$$

Here, $\ell = 0, 1, 2, \ldots$ is the total angular momentum quantum number and $m = 0, \pm 1, \ldots, \pm \ell$ is the projection of the total angular momentum on the z axis (magnetic quantum number) [see Fig. 2(b)]. While outside the cavity one should put $a_{\ell m}^{out} = b_{\ell m}^{out} = 0$ for all $\ell$ and m as fluid (gas) supports only longitudinal waves, inside the cavity one should account for all the vector harmonics. As a result, the homogeneous system of equations on the expansion coefficients can be obtained,

$$
\hat {\mathbf {D}} _ {\ell m} \mathbf {f} _ {\ell m} = 0, \tag {4}
$$

where vector $f_{\ell m} = [a_{\ell m}^{in}, b_{\ell m}^{in}, c_{\ell m}^{in}, c_{\ell m}^{out}]^{T}$ is the vector of coefficients, and $\hat{D}_{\ell m} = diag[\hat{D}_{\ell m}^{1\times1}, \hat{D}_{\ell m}^{3\times3}]$ is a block-diagonal matrix with the explicit form provided in the Supplemental Material [30]. By virtue of the block-diagonal form of $\hat{D}_{\ell m}$ , the equation on eigenfrequencies is factorized:

$$
\underbrace {\det \hat {\mathbf {D}} _ {\ell m} ^ {1 \times 1}} _ {\text { BIC }} \cdot \underbrace {\det \hat {\mathbf {D}} _ {\ell m} ^ {3 \times 3}} _ {\text { Rad.   modes }} = 0. \tag {5}
$$

Generally, the eigenfrequency can be rescaled by the factor of $c_{s}/a$ . However, to have an illustrative physical example we will use the secondary frequency axis corresponding to

![](images/8a034e6fc18148684fb8bae36188b04b9368cae6284edeba4224fbaaf90c1a07.jpg)

<details>
<summary>bar</summary>

| k_s a | Q-factor | ω (kHz) |
|-------|----------|---------|
| 2.0   | ~10^1    | 30      |
| 2.5   | ~10^2    | 40      |
| 3.0   | ~10^2    | 50      |
| 3.5   | ~10^2    | 60      |
| 4.0   | ~10^2    | 70      |
| 4.5   | ~10^2    | 80      |
| 5.0   | ~10^2    | 90      |
| 5.5   | ~10^2    | 100     |
| 6.0   | ~10^3    | 110     |
</details>

(b)   
![](images/47ff93654e9cb278d1b53c7bc348681abb81b66ad118e4dbf18867ac3e799e20.jpg)  
Dipole

![](images/2e5c70487628b6bb5999d3f03f0eb15bb1139891afcc40c1d1544599d5817afa.jpg)  
Quadrupole

![](images/fdb56223e8567173be0a6eea97c136eabff9fd5644c8a18b237b81d7fad57c82.jpg)  
Octupole

![](images/454c8a4adf7616822d73e9bf23909b76b34bd6dff2ea22d2026652f37653663f.jpg)

<details>
<summary>text_image</summary>

min uφ max
Hexadecapole
</details>

FIG. 2. Polarization-protected BICs in a solid spherical resonator. (a) Spectrum and radiative Q factor of the resonator. Parameters of the resonator are mentioned in the text. (b) Distributions of the azimuthal component of the displacement $u_{\varphi}$ for the dipole, quadrupole, octupole, and hexadecapole BICs with m = 0.

the sphere of radius a = 5 cm placed in air with density $\rho_{0} = 1.23 \, kg/m^{3}$ . The density of the material of the sphere equals to $\rho = 10\rho_{0}$ , while the velocity of shear and pressure waves are taken equal to $c_{s} = 2c_{0}$ , $c_{p} = 3c_{0}$ , respectively, where $c_{0} = 343 \, m/s$ is the velocity of sound in the air.

The spectrum of the resonator (eigenfrequencies and Q factors) obtained from the numerical solution of Eq. (5) is shown in Fig. 2(a). It consists of radiative modes and BICs with infinitely high radiative Q factors. One can show that purely real eigenfrequencies corresponding to BICs satisfy the equation $\det\hat{D}_{\ell m}^{1\times1}=0$ , which can be written explicitly as follows (see Supplemental Material [30]):

$$
(1 - \ell) j _ {\ell} (x) + x j _ {\ell + 1} (x) = 0. \tag {6}
$$

Here $j_{\ell}(x)$ it the spherical Bessel function, and $x = k_{s}a$ . Because of the spherical symmetry of the problem, the obtained equation does not depend on m, therefore, its solution is $(2\ell + 1)$ -fold degenerate. The table containing the roots of Eq. (6) is provided in the Supplemental Material [30].

It also follows from Eqs. (4) and (5) that $b_{\ell m}^{in} = c_{\ell m}^{in} = c_{\ell m}^{out} = 0$ for BICs and, consequently, the displacement field u contains only vector harmonics $M_{\ell m}$ and they are completely localized inside the resonator. This is in complete accordance with a general theorem [32] which establishes that for nonradiating resonators of finite size, the corresponding radiated fields must vanish in the vacuum region outside the resonator. Vector harmonics $M_{\ell m}$ correspond to the torsion oscillations which totally lack radial components in sharp contrast to $L_{\ell m}$ and $N_{\ell m}$ and, thus, cannot excite pressure waves in the surrounding fluids. The distribution of the displacement field for BICs with m = 0 and $\ell = 1, 2, 3, 4$ is shown in Fig. 2(b). Equation (6) can be also obtained from Eq. (1) applying stress-free boundary conditions [33,34].

A curious fact deserving special attention is that the fundamental acoustic BIC in a solid sphere is a quadrupole mode $(\ell=2)$ rather than a dipole. This fact can be understood intuitively: the time-average angular momentum of the resonator should be zero, i.e., the resonator should not rotate as a whole. For the dipole modes, the external and internal layers oscillate in antiphase compensating the rotation [see Fig. 2(b)], and it results in quite a large radial wave number $k_{s}a\approx5.8$ . For the quadrupole mode $k_{s}a\approx2.5$ and the oscillation phase does not change along the radial direction. Thus, the time-average angular momentum is compensated by antiphase oscillations of the upper and lower hemispheres [see Fig. 2(b)].

A BIC, according to its definition, is completely decoupled from all propagating waves of the surrounding space, and, thus, it cannot be excited from the far-field by pressure waves. However, the excitation is possible by near-field sources or due to nonlinear effects $[35–37]$ . Another efficient method, that is the most used in practice, is based on the introduction of small coupling between the BIC and radiative modes. Therefore, a genuine BIC turns into a quasi-BIC (QBIC)—a high-Q state that manifests itself in the scattering spectrum as a narrow Fano resonance [38]. Recently, the QBICs were suggested as very promising candidates for sensing, lasing, and nonlinear optics applications [14,39–44].

![](images/b18cfde9b74a1de9e1851d2cfc7b39f6b909b2c04d94e236ceee55880a7228bc.jpg)

<details>
<summary>line</summary>

| Asymmetry parameter, α (x10⁻²) | k_s a (Genuine BIC) | k_s a (Quasi-BIC) | Re(ω), (kHz) |
| ------------------------------ | ------------------- | ----------------- | ------------ |
| 0                              | 2.5                 | 2.5               | 34           |
| 2.5                            | 2.6                 | 2.55              | 35           |
| 5.0                            | 2.7                 | 2.6               | 36           |
| 7.5                            | 2.8                 | 2.65              | 37           |
| 10.0                           | 2.9                 | 2.7               | 38           |
| 12.5                           | 3.0                 | 2.75              | 39           |
| 15.0                           | 3.1                 | 2.8               | 40           |
</details>

![](images/ebc9144f50072d42f37f3efef50165fdded7bef37fd4d5873cb8dbb26134e232.jpg)

<details>
<summary>line</summary>

| Asymmetry parameter, α | quadrupole (ℓ = 2) | octupole (ℓ = 3) | hexadecapole (ℓ = 4) |
| ---------------------- | ------------------ | ---------------- | -------------------- |
| 10⁻⁶                   | ~10¹⁴              | ~10¹⁴            | ~10¹⁴                |
| 10⁻⁵                   | ~10¹²              | ~10¹²            | ~10¹²                |
| 10⁻⁴                   | ~10¹⁰              | ~10¹⁰            | ~10¹⁰                |
| 10⁻³                   | ~10⁸               | ~10⁸             | ~10⁸                 |
| 10⁻²                   | ~10⁶               | ~10⁶             | ~10⁶                 |
</details>

FIG. 3. Splitting of a quadrupole multiplet into BIC and quasi-BIC in a solid spheroid. (a) Frequency shift and Q factors versus asymmetry parameter $\alpha = 1 - b/a$ . (b) Dependence of the Q factor on the asymmetry parameter $\alpha$ for quadrupole, octupole, and hexadecapole quasi-BICs with $m = \pm2$ .

In order to show how a genuine acoustic BIC turns into a QBIC, we slightly deform the spherical resonator of radius $a$ into an oblate spheroid with the semiaxes $a$ and $b$ [see inset in Fig. 3(a)]. The dependence of the resonant frequency and $Q$ factor for the quadrupole BIC ( $\ell = 2$ ) on the asymmetry parameter $\alpha = 1 - a / b$ is shown in Fig. 3(a). As it was mentioned above, the BICs in spherical resonator are $(2\ell + 1)$ -fold degenerate multiplets. In the spheroid, this multiplet splits into one singlet state

TABLE I. Splitting of a quadrupole multiplet BIC ( $\ell = 2$ ) in a spheroid. Multipole content of the modes and selection rules for their excitation by a plane wave. Here index p is a non-negative integer. 

<table><tr><td>Irrep</td><td>Mode</td><td>Multipoles</td><td>Excitation along x or y</td><td>Excitation along z</td></tr><tr><td> $A_{2u}$ </td><td>BIC</td><td> $\mathbf{M}_{2,0}, ..., \mathbf{M}_{2p,0}$  $\mathbf{L}_{1,\pm 1}, ..., \mathbf{L}_{2p+1,\pm 1}$ </td><td>No</td><td>No</td></tr><tr><td> $E_{1u}$ </td><td>QBIC</td><td> $\mathbf{N}_{1,\pm 1}, ..., \mathbf{N}_{2p+1,\pm 1}$  $\mathbf{M}_{2,\pm 1}, ..., \mathbf{M}_{2p,\pm 1}$ </td><td>Yes</td><td>No</td></tr><tr><td> $E_{2u}$ </td><td>QBIC</td><td> $\mathbf{N}_{3,\pm 2}, ..., \mathbf{N}_{2p+1,\pm 2}$  $\mathbf{M}_{2,\pm 2}, ..., \mathbf{M}_{2p,\pm 2}$ </td><td>No</td><td>No</td></tr></table>

corresponding to the BIC and $2\ell$ of two-fold degenerate doublet QBICs (see Supplemental Material [30]) [Fig. 3(a)]. The Q factor of the QBICs drops quadratically with the asymmetry parameter $\alpha$ [see Fig. 3(b)] that completely agrees with the general theory of QBICs [38].

In terms of the group theory, the quadrupole BIC in a spheroid ( $D_{\infty h}$ symmetry) corresponds to one-dimensional irreducible representation (irrep) $A_{2u}$ , and the doublets of QBICs correspond to two-dimensional irreps $E_{1u}$ and $E_{2u}$ . The degeneracy of the QBICs remains due to the rotational symmetry of the spheroid, thus, the doublet can be associated with clockwise- and counterclockwise rotational modes. Following Ref. [45] we can write the multipole series of the BICs and QBICs, and selection rules for their excitation by a plane pressure wave incident from different directions (see Table I). The survivor BICs in a spheroid are contributed only by the vector harmonics $M_{\ell m}$ with even $\ell$ and m = 0. As a spheroid has a symmetry plane (xy), there is a second series of genuine BICs contributed by the vector harmonics $M_{\ell m}$ with odd $\ell$ and m = 0 (see Supplemental Material [30]).

One may see from Table I that QBICs from $E_{1u}$ can be excited by a plane pressure wave propagating along the x or y axis. The QBICs from $E_{2u}$ can be excited only at oblique incidence. Indeed, the excitation from the x, y directions is forbidden due to inconsistency between the parity of the mode and the incident wave, and the excitation from the z direction is forbidden due to the fact that the incident wave contains only the harmonics with m = 0.

Figure 4(a) shows the scattering efficiency $\sigma_{\mathrm{sct}}$ of a solid spheroid excited by a plane pressure wave propagating along the $x$ axis. The scheme of excitation is illustrated in Fig. 4(c). The spectra for different eccentricities $e$ were calculated numerically using COMSOL Multiphysics. One can see that the QBIC corresponding to irrep $E_{1u}$ appears as a high- $Q$ Fano resonance that collapses when $e$ tends to zero manifesting the formation of a genuine BIC. Figure 4(b) shows the contribution of the resonant and nonresonant scattering channels $\sigma_{\ell m}$ to the total scattering efficiency $\sigma_{tot}$ . Figures 4(d) and 4(e) show the directivity patterns for the total scattered field and scattered field accounted for only the resonant harmonics of the QBIC (see Table I). The diagrams are plotted for the spheroid with e=0.1 at the resonant frequency of the QBIC ( $k_{s}a=2.51$ ). One can see that the QBIC behaves as a dipole in the far field but the nonresonant scattering drastically changes the directivity pattern.

(a)   
![](images/61c3f2da98c29362adff3b3e3f9f484f022618eaa718b6200aff66fa2866544b.jpg)

<details>
<summary>line</summary>

| k_s a | σ_sct (e=0) | σ_sct (e=0.2) | σ_sct (e=0.3) | σ_sct (e=0.4) | σ_sct (e=0.5) |
|-------|-------------|---------------|---------------|---------------|---------------|
| 2.52  | ~1.0        | ~1.0          | ~1.0          | ~1.0          | ~1.0          |
| 2.56  | ~0.8        | ~0.8          | ~0.8          | ~0.8          | ~0.8          |
| 2.60  | ~0.7        | ~0.7          | ~0.7          | ~0.7          | ~0.7          |
| 2.64  | ~1.2        | ~1.1          | ~1.0          | ~0.9          | ~0.8          |
</details>

(c)

![](images/d90d66e24f43c45ea9c3cdad4ef8fdb839215b49aea8d72d20e53854a5cfa6c4.jpg)

(d)   
![](images/6416e8119c5f411d311e3d35def45b40bdbb3f91b77ebf68cc8c3dc934b6048e.jpg)

<details>
<summary>radar</summary>

| Angle | xz-pl | xy-pl |
|-------|-------|-------|
| 0°    | 0.1   | 0.1   |
| 90°   | 0.05  | 0.05  |
| 270°  | 0.02  | 0.02  |
| 280°  | 0.01  | 0.01  |
| 290°  | 0.01  | 0.01  |
| 300°  | 0.01  | 0.01  |
| 310°  | 0.01  | 0.01  |
| 320°  | 0.01  | 0.01  |
| 330°  | 0.01  | 0.01  |
| 340°  | 0.01  | 0.01  |
| 350°  | 0.01  | 0.01  |
| 360°  | 0.01  | 0.01  |
</details>

(e)   
![](images/442b2e571a7689a5d79249c921c154bb02f1ebc3ece26279ce9b8bbc47be4b1b.jpg)

<details>
<summary>radar</summary>

| Angle | Value |
|-------|-------|
| 0°    | 0.1   |
| 90°   | 0.3   |
| 270°  | 0.1   |
</details>

FIG. 4. (a) Spectrum of scattering efficiency of a solid spheroid calculated for various eccentricities e. (b) Spectrum of the total and partial scattering efficiencies calculated near a QBIC $E_{1u}$ for e = 0.5. (c) The excitation scheme. (d) Directivity patterns of the total scattered field and (e) scattered field accounted for the resonant contribution of a QBIC only. The diagrams are plotted for e = 0.1 at the resonant frequency of a QBIC $k_{s}a = 2.51$ .

A reasonable question is, Can we find all possible shapes of resonators capable of supporting such BICs? From group theory it can be shown that symmetry breaking results in mixing the multipoles (see Table I). In the general case of an arbitrarily shaped resonator, all modes are radiative. However, multipole mixing occurs according to the selection rules defined by the symmetry group of the resonator. Thus, BICs survive in the resonators when the multipoles $M_{\ell m}$ do not couple with $N_{\ell m}$ and $L_{\ell m}$ . Such a requirement is fulfilled for resonators with an infinite-fold rotation axis-like cylinder, cone, dimer, etc. ( $D_{\infty h}$ and $D_{\infty v}$ symmetry groups).

In this prospective, special attention should be paid to the polarization-protected acoustic BICs in a solid cylinder of radius a and height h. Though the eigenvalue problem for an open cylindrical resonator cannot be solved analytically since the variables cannot be separated, it becomes possible for the BIC as the system behaves as a closed one. Therefore, the eigenfrequencies can be calculated analytically using the stress-free boundary conditions:

$(\omega / c_s)^2 = (\alpha_n / a)^2 + (\pi q / h)^2$ . Here $\alpha_{n}$ is the $n$ th root of the Bessel function $J_2(x)$ , and $q$ is an integer.

The total $Q$ factor $(Q_{\mathrm{tot}})$ of acoustic BICs is limited by the absorption $Q$ factor $(Q_{\mathrm{abs}})$ that is defined by the attenuation of shear waves in real materials. For example, the longitudinal loss factor $\eta = E'' / E'$ of steel is reported to be on the order of $10^{-4} - 10^{-5}$ [46,47], where $E = E' + iE''$ is the complex Young's modulus. A loss factor of the same order is reported for different ceramics and glasses [46], while for silica the loss factor is on the order of $10^{-6}$ . Since the longitudinal and shear loss factors are of the same order [48], one can expect the total $Q$ factor of BICs will be around $Q = 1 / \eta \sim 10^4$ to $10^6$ . Analysis of the influence of other $Q$ -factor reduction mechanisms, such as viscous or thermal losses, is presented in the Supplemental Material, section (S9) [30].

In conclusion, we have revealed that genuine acoustic bound states in the continuum may exist in compact solid resonators with a rotational symmetry placed in a gas or nonviscous fluid environment. The predicted states are possible due to polarization mismatch between the shear waves in the solid resonator and pressure waves in the surrounding media. Usually, BICs have topological properties as they are robust in configuration space to variation of some parameters of the system. However, the topological properties for the revealed acoustic BIC are a matter of further research. We believe that our findings are an important step in the development of high-Q resonant acoustics, and the revealed novel BICs in compact structures will serve as building blocks for acoustic antennas, high-sensitive acoustic sensors, and topological acoustic structures.

The authors thank K. Frizyuk, K. Koshelev, D. Maksimov, A. Sadreev, and Yu. Kivshar for useful discussions. The analytical simulation was supported by the Russian Science Foundation Project No. 20-72-10141. The numerical results were obtained within the Russian Science Foundation Project No. 22-42-04420. A. B. and M. P. acknowledge the “BASIS” Foundation. This research was supported by Priority 2030 Federal Academic Leadership Program.

\*ilya.deriy@metalab.itmo.ru

$^{\dagger}$ a.bogdanov@metalab.itmo.ru

$^{\ddagger}$ m.petrov@metalab.ifmo.ru

[1] C. W. Hsu, B. Zhen, A. D. Stone, J. D. Joannopoulos, and M. Soljačić, Bound states in the continuum, Nat. Rev. Mater. 1, 16048 (2016).   
[2] A. F. Sadreev, Interference traps waves in open system: Bound states in the continuum, Rep. Prog. Phys. 84, 055901 (2021).   
[3] J. von Neuman and E. Wigner, Über merkwürdige diskrete Eigenwerte, Phys. Z. 30, 467 (1929).

[4] L. Fonda, Bound states embedded in the continuum and the formal theory of scattering, Ann. Phys. (N.Y.) 22, 123 (1963).   
[5] F. Ursell, Trapping modes in the theory of surface waves, Proc. Cambridge Philos. Soc. 47, 347 (1951).   
[6] R. Parker, Resonance effects in wake shedding from parallel plates: Some experimental observations, J. Sound Vib. 4, 62 (1966).   
[7] H. Tong, S. Liu, M. Zhao, and K. Fang, Observation of phonon trapping in the continuum with topological charges, Nat. Commun. 11, 5216 (2020).   
[8] Y. Yu, X. Xi, and X. Sun, Observation of bound states in the continuum in a micromechanical resonator, arXiv:2109.09498.   
[9] S. Azzam and A. Kildishev, Photonic bound states in the continuum: From basics to applications, Adv. Opt. Mater. 9, 2001469 (2021).   
[10] K. Koshelev, G. Favraud, A. Bogdanov, Y. Kivshar, and A. Fratalocchi, Nonradiating photonics with resonant dielectric nanostructures, Proc. Soc. Photo-Opt. Instrum. Eng. 8, 725 (2019).   
[11] A. Kodigala, T. Lepetit, Q. Gu, B. Bahari, Y. Fainman, and B. Kanté, Lasing action from photonic bound states in continuum, Nature (London) 541, 196 (2017).   
[12] K. Koshelev, S. Kruk, E. Melik-Gaykazyan, J.-H. Choi, A. Bogdanov, H.-G. Park, and Y. Kivshar, Subwavelength dielectric resonators for nonlinear nanophotonics, Science 367, 288 (2020).   
[13] V. Kravtsov, E. Khestanova, F. A. Benimetskiy, T. Ivanova, A. K. Samusev, I. S. Sinev, D. Pidgayko, A. M. Mozharov, I. S. Mukhin, M. S. Lozhkin et al., Nonlinear polaritons in a monolayer semiconductor coupled to optical bound states in the continuum, Light Sci. Appl. 9, 56 (2020).   
[14] A. Tittl, A. Leitis, M. Liu, F. Yesilkoy, D.-Y. Choi, D.N. Neshev, Y.S. Kivshar, and H. Altug, Imaging-based molecular barcoding with pixelated dielectric metasurfaces, Science 360, 1105 (2018).   
[15] T. Lepetit and B. Kanté, Controlling multipolar radiation with symmetries for electromagnetic bound states in the continuum, Phys. Rev. B 90, 241103(R) (2014).   
[16] A. Pilipchuk, A. Pilipchuk, and A. Sadreev, Bound states in the continuum in open spherical resonator, Phys. Scr. 95, 085002 (2020).   
[17] A. Lyapina, D. Maksimov, A. Pilipchuk, and A. Sadreev, Bound states in the continuum in open acoustic resonators, J. Fluid. Mech. 780, 370 (2015).   
[18] C. W. Hsu, B. Zhen, J. Lee, S.-L. Chua, S. G. Johnson, J. D. Joannopoulos, and M. Soljačić, Observation of trapped light within the radiation continuum, Nature (London) 499, 188 (2013).   
[19] Z. F. Sadrieva, M. A. Belyakov, M. A. Balezin, P. V. Kapitanova, E. A. Nenasheva, A. F. Sadreev, and A. A. Bogdanov, Experimental observation of a symmetry-protected bound state in the continuum in a chain of dielectric disks, Phys. Rev. A 99, 053804 (2019).   
[20] M. G. Silveirinha, Trapping light in open plasmonic nanostructures, Phys. Rev. A 89, 023813 (2014).   
[21] S. Lannebère and M. G. Silveirinha, Optical meta-atom for localization of light with quantized energy, Nat. Commun. 6, 8766 (2015).

[22] F. Monticone and A. Alu, Embedded Photonic Eigenvalues in 3d Nanostructures, Phys. Rev. Lett. 112, 213903 (2014).   
[23] I. Liberal and N. Engheta, Nonradiating and radiating modes excited by quantum emitters in open epsilon-near-zero cavities, Sci. Adv. 2, e1600987 (2016).   
[24] L. E. Kinsler, A. R. Frey, A. B. Coppens, and J. V. Sanders, Fundamentals of Acoustics, 4th ed. (John Wiley & Sons, New York, 1999).   
[25] L. Huang, Y. K. Chiang, S. Huang, C. Shen, F. Deng, Y. Cheng, B. Jia, Y. Li, D. A. Powell, and A. E. Miroshnichenko, Sound trapping in an open resonator, Nat. Commun. 12, 4819 (2021).   
[26] A. S. Pilipchuk, A. A. Pilipchuk, and A. F. Sadreev, Bound states in the continuum in open spherical resonator, Phys. Scr. 95, 085002 (2020).   
[27] I. Quotane, E. H. El Boudouti, and B. Djafari-Rouhani, Trapped-mode-induced fano resonance and acoustical transparency in a one-dimensional solid-fluid phononic crystal, Phys. Rev. B 97, 024304 (2018).   
[28] S. Mizuno, Fano resonances and bound states in the continuum in a simple phononic system, Appl. Phys. Express 12, 035504 (2019).   
[29] M. A. Isakovich, General Acoustics (Nauka, Moscow, 1973).   
[30] See Supplemental Material at http://link.aps.org/supplemental/10.1103/PhysRevLett.128.084301 for the details of the rigorous derivation of the Helmholtz equation for the displacement field in isotropic media from the equation of motion for the elastic media; for the connection between pressure and displacement field in the non-viscous fluid; for information about vector harmonics, their main properties, and explicit view in spherical and cylindrical systems of coordinates; for the explicit derivation of the eigenfrequency equation for the BICs in sphere and cylinder; for the mode profiles for BICs with $\ell = 1, 2, 3, 4$ and m = 0, 1, 2, 3, 4; for the table with the classification of the eigenmodes in solids with $D_{\infty h}$ symmetry; for the information about the influence of different non-radiative losses on BICs, and, for the details of the numerical simulations.   
[31] C. F. Bohren and D. R. Huffman, Absorption and Scattering of Light by Small Particles (John Wiley & Sons, New York, 2008).   
[32] D. L. Colton, R. Kress, and R. Kress, Inverse acoustic and electromagnetic scattering theory (Springer, Berlin, 1998), Vol. 93, p. 165.   
[33] A. Tamura, K. Higeta, and T. Ichinokawa, Lattice vibrations and specific heat of a small particle, J. Phys. C 15, 4975 (1982).

[34] S. Tamim and J. Bostwick, The elastic rayleigh drop, Soft Matter 15, 9244 (2019).   
[35] E. Bulgakov, K. Pichugin, and A. Sadreev, All-optical light storage in bound states in the continuum and release by demand, Opt. Express 23, 22520 (2015).   
[36] A. Chukhrov, S. Krasikov, A. Yulin, and A. Bogdanov, Excitation of a bound state in the continuum via spontaneous symmetry breaking, Phys. Rev. B 103, 214312 (2021).   
[37] L. Yuan and Y.Y. Lu, Excitation of bound states in the continuum via second harmonic generations, SIAM J. Appl. Math. 80, 864 (2020).   
[38] K. Koshelev, S. Lepeshov, M. Liu, A. Bogdanov, and Y. Kivshar, Asymmetric Metasurfaces with High-q Resonances Governed by Bound States in the Continuum, Phys. Rev. Lett. 121, 193903 (2018).   
[39] K. Koshelev, A. Bogdanov, and Y. Kivshar, Meta-optics and bound states in the continuum, Sci. Bull. 64, 836 (2019).   
[40] K. Koshelev, Y. Tang, K. Li, D.-Y. Choi, G. Li, and Y. Kivshar, Nonlinear metasurfaces governed by bound states in the continuum, ACS Photonics 6, 1639 (2019).   
[41] Z. Liu, Y. Xu, Y. Lin, J. Xiang, T. Feng, Q. Cao, J. Li, S. Lan, and J. Liu, High-q Quasibound States in the Continuum for Nonlinear Metasurfaces, Phys. Rev. Lett. 123, 253901 (2019).   
[42] A. Leitis, A. Tittl, M. Liu, B. H. Lee, M. B. Gu, Y. Kivshar, and H. Altug, Angle-multiplexed all-dielectric metasurfaces for broadband molecular fingerprint retrieval, Sci. Adv. 5, eaaw2871 (2019).   
[43] A. Vaskin, R. Kolkowski, A. F. Koenderink, and I. Staude, Light-emitting metasurfaces, Nanophotonics 8, 1151 (2019).   
[44] M. Liu and D.-Y. Choi, Extreme huygens' metasurfaces based on quasi-bound states in the continuum, Nano Lett. 18, 8062 (2018).   
[45] S. Gladyshev, K. Frizyuk, and A. Bogdanov, Symmetry analysis and multipole classification of eigenmodes in electromagnetic resonators for engineering their optical properties, Phys. Rev. B 102, 075103 (2020).   
[46] J. Zhang, R. Perez, and E. Lavernia, Documentation of damping capacity of metallic, ceramic and metal-matrix composite materials, J. Mater. Sci. 28, 2395 (1993).   
[47] T. Irvine, Damping properties of materials, Magnesium 5000, 10 (2004).   
[48] T. Pritz, Relation of bulk to shear loss factor of solid viscoelastic materials, J. Sound Vib. 324, 514 (2009).