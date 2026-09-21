# Enhancing the Sensitivity of Frequency and Energy Splitting Detection by Using Exceptional Points: Application to Microcavity Sensors for Single-Particle Detection

Jan Wiersig
Institut für Theoretische Physik, Universität Magdeburg, Postfach 4120, D-39016 Magdeburg, Germany
(Received 30 January 2014; published 20 May 2014)

Several types of sensors used in physics are based on the detection of splittings of resonant frequencies or energy levels. We propose here to operate such sensors at so-called exceptional points, which are degeneracies in open wave and quantum systems where at least two resonant frequencies or energy levels and the corresponding eigenstates coalesce. We argue that this has great potential for enhanced sensitivity provided that one is able to measure both the frequency splitting as well as the linewidth splitting. We apply this concept to a microcavity sensor for single-particle detection. An analytical theory and numerical simulations prove a more than threefold enhanced sensitivity. We discuss the possibility to resolve individual linewidths using active optical microcavities.

DOI: 10.1103/PhysRevLett.112.203901

PACS numbers: 42.60.Da, 42.55.Sa, 42.79.-e, 78.67.Bf

A degeneracy of resonant frequencies (energy levels) can serve as a basic element of a sensor because a small perturbation can lift the degeneracy and can therefore lead to a detectable splitting of these frequencies. This simple principle is used in modern sensor devices, such as microcavity sensors for single or few particle detection $[1-3]$ , optical gyroscopes $[4,5]$ , weak magnetic field sensors $[6]$ , nanomechanical mass sensors $[7]$ , and bending curvature sensors $[8]$ . In all of these examples, the perturbation acts on a conventional twofold degeneracy known from the quantum mechanics of conservative systems. A point in parameter space at which such a degeneracy is located is called the diabolic point (DP) $[9]$ . Two properties are characteristic of a DP. First, only the energy levels degenerate but not the associated eigenstates. The eigenstates can always be chosen to be orthogonal to each other. Second, a perturbation of strength $\varepsilon$ leads to energy shifts and splitting proportional to $\varepsilon$ .

Most systems in physics are, however, open systems. For example, the optical modes (the optical analogue of energy eigenstates) in microcavities are subjected to losses due to absorption and radiation. This leads to decaying modes and finite spectral linewidths at resonant frequencies. These features can often be well described by a non-Hermitian effective Hamiltonian $[10]$ . In such open systems there is, besides the DPs, another type of degeneracy called the exceptional point (EP). The behavior at an EP is more drastic than at a DP as at this point in parameter space not only the eigenvalues (at least two) but also the corresponding eigenstates coalesce $[11–14]$ . The physical existence of EPs has been demonstrated by experiments on a number of systems, e.g., microwave cavities $[15–17]$ , optical microcavities $[18,19]$ , and coupled atom-cavity systems $[20]$ . Moreover, theoretical studies suggest that EPs also play a role in other physical systems, such as hydrogen atoms in crossed magnetic and electric fields [21], photonic lattices [22], and nonuniformly pumped lasers [23].

If an EP for two coalescing levels is subjected to a perturbation of strength $\varepsilon$ then the resulting energy splitting is typically proportional to $\sqrt{\varepsilon}$ [11,24]. In other words, for sufficiently small $\varepsilon$ the splitting is enhanced if compared to a DP even though exactly the same perturbation is applied. It is precisely this basic characteristic of EPs that we here exploit for sensor applications. As an example, we discuss a microcavity sensor for label-free, single-particle detection. For such sensors one often uses whispering-gallery mode (WGM) optical microcavities like microdisks [25–27], microspheres [28–32], and microtoroids [1,19,33]. When a target particle comes close to the boundary of the microcavity then the evanescent coupling leads to light backscattering of clockwise (CW) and counterclockwise (CCW) propagating modes. This coupling of modes results in frequency shifts and splitting of WGM pairs. Measuring the frequency splitting rather than a frequency shift is more robust against noise from the probe laser and environmental temperature fluctuations [1,34].

The setup that we study here is motivated by a recent experiment on a microtoroid with two nanofiber tips placed in the evanescent field of the modes $[19]$ . These two tips were used to remove an unwanted frequency splitting which originated from fabrication imperfections. Based on spectral data only, it was claimed that the degeneracy created in this way is an EP. This was confirmed by a study of the spatial and spectral properties of modes in a similar system: two scatterers close to a microdisk $[35]$ . However, none of these two papers addressed the utility of EPs for sensor applications. The aim of this Letter is to present a theoretical study which reveals that the sensitivity of the microcavity operating at an EP can be significantly enhanced provided that frequency and linewidth splittings are both experimentally accessible.

![](images/78c5822b19d6b48a064f11ff0899809a4a8d5cb6d92e34e0c98e005630c058fc.jpg)

<details>
<summary>text_image</summary>

y
d₂
β₂
d₁
x
β₃
R
d₃
</details>

FIG. 1. Microdisk of refractive index n and radius R with three nanoparticles of refractive index $n_{j}$ at distance $d_{j}$ and at azimuthal position $\beta_{j}$ ( $\beta_{1}=0$ ). The two gray dots mark the particles that implement the exceptional point and the small black one marks the target nanoparticle.

Our microcavity sensor is the two-particle-microdisk system introduced in Ref. [35]. Together with the target nanoparticle, there are in total three particles of radii $r_j$ ( $j = 1,2,3$ ) in the periphery of the microdisk of radius $R$ as illustrated in Fig. 1. Note that the precise shape is not important in the regime of Rayleigh scattering where the wavelength $\lambda$ is much larger than the size of the scatterers. The position of each particle's center is $(x_j, y_j) = (R + r_j + d_j)(\cos \beta_j, \sin \beta_j)$ .

We solve Maxwell's equations in two dimensions using the effective index approximation. The solutions with time dependence $e^{-i\omega t}$ are the optical modes. Sommerfeld outgoing wave conditions are applied at infinity, leading to quasibound states with complex frequencies $\omega$ in the lower half-plane. For convenience, we use a dimensionless frequency $\Omega = (\omega / c)R$ where $c$ is the speed of light in vacuum. The real part is the conventional frequency whereas the imaginary part determines the linewidth (decay rate) $\gamma = -2\mathrm{Im}\Omega$ and the quality factor $Q = -\mathrm{Re}\Omega / [2\mathrm{Im}\Omega]$ of the given mode.

In Ref. [35] an effective Hamiltonian has been introduced that describes this nanoparticles-microdisk system within a two-mode approximation and slowly varying envelope approximation in the time domain [36]. With the azimuthal mode number $m \in \mathbb{N}$ and the frequency $\Omega^{(0)}$ of the unperturbed WGMs, the total effective Hamiltonian for the microdisk with $N$ nanoparticles in the traveling-wave basis (CCW,CW) is given by the $2 \times 2$ matrix

$$
H ^ {(N)} = \left( \begin{array}{c c} \Omega^ {(N)} & A ^ {(N)} \\ B ^ {(N)} & \Omega^ {(N)} \end{array} \right) \tag {1}
$$

with

$$
\Omega^ {(N)} = \Omega^ {(0)} + \sum_ {j = 1} ^ {N} (V _ {j} + U _ {j}), \tag {2}
$$

$$
A ^ {(N)} = \sum_ {j = 1} ^ {N} (V _ {j} - U _ {j}) e ^ {- i 2 m \beta_ {j}}, \tag {3}
$$

$$
B ^ {(N)} = \sum_ {j = 1} ^ {N} (V _ {j} - U _ {j}) e ^ {i 2 m \beta_ {j}}. \tag {4}
$$

The quantities $2V_{j}$ and $2U_{j}$ are given by the complex frequency shifts for positive- and negative-parity modes introduced by particle $j$ alone. These quantities can be calculated for the single-particle-microdisk system either fully numerically, using, e.g., the boundary element method (BEM) [37], the finite-difference time-domain method [38], the finite-difference frequency-domain method [39], or approximately using the Green's function approach for point scatterers [40] ( $U_{j} = 0$ ). Note that the matrix (1) is not Hermitian since $V_{j}$ , $U_{j}$ , and $\Omega^{(0)}$ are complex numbers reflecting the presence of losses.

An important result of Ref. [35] is that in general $|A^{(N)}| \neq |B^{(N)}|$ ; i.e., the backscattering between CW and CCW traveling waves is asymmetric; see also Refs. [41,42]. This asymmetric backscattering has interesting consequences [43,44], such as the appearance of pairs of optical modes, where in each pair (i) the two modes are significantly nonorthogonal, (ii) each of the two modes has a finite orbital angular momentum, and (iii) both modes mainly copropagate. In the case of full asymmetry, $A^{(N)} = 0$ or $B^{(N)} = 0$ , the modes coalesce at an EP.

In our setup, the first two particles implement the EP and the third is disturbing it; i.e., the microcavity sensor together with the target particle are described by

$$
H = H _ {0} + H _ {1} \tag {5}
$$

with $H = H^{(3)}$ and $H_0 = H^{(2)}$ from Eq. (1), and

$$
H _ {1} = \left( \begin{array}{c c} V + U & (V - U) e ^ {- i 2 m \beta} \\ (V - U) e ^ {i 2 m \beta} & V + U \end{array} \right) \tag {6}
$$

using the short-hand notation $V = V_{3}$ , $U = U_{3}$ , $\beta = \beta_{3}$ . We require that the unperturbed sensor, described by $H_{0}$ , has zero frequency splitting. This can be achieved either by using a DP or an EP. A DP is here given by $B^{(2)} = 0$ and $A^{(2)} = 0$ , i.e., without any backscattering between CW and CCW traveling waves. An EP results for $B^{(2)} = 0$ or $A^{(2)} = 0$ . There is no backscattering into the CW (or CCW) direction even though there is backscattering into the opposite direction. The EP realized for the microtoroid with two nanofiber tips [19] is of such a kind.

A straightforward calculation shows that the disturbance due to the target particle, $H_{1}$ , induces for the DP the complex frequency splitting

$$
\Delta \Omega_ {\mathrm{DP}} = 2 (V - U). \tag {7}
$$

The real part of $\Delta\Omega_{DP}$ is the conventional frequency splitting observed for instance in Ref. [1]. The imaginary part corresponds to a small linewidth splitting. In the case of the EP [with $B^{(2)}=0$ and $A^{(2)}\neq0$ from Eq. (3)], however, one gets

$$
\Delta \Omega_ {\mathrm{EP}} = \Delta \Omega_ {\mathrm{DP}} \sqrt {1 + \frac {A ^ {(2)} e ^ {i 2 m \beta}}{V - U}}. \tag {8}
$$

For the opposite case $A^{(2)} = 0$ , the quantities $A^{(2)}$ and $\beta$ have to be replaced in Eq. (8) by $B^{(2)}$ and $-\beta$ , respectively. If the square root in Eq. (8) is larger than unity then the frequency splitting at the EP is larger than the one at the DP even though in both cases the perturbation $H_{1}$ is exactly the same. The intuitive explanation is that the intrinsic (and fully asymmetric) backscattering of strength $|A^{(2)}|$ does not lead to a splitting as long as there is no target particle, but in the presence of a target particle it is able to give a significant contribution to the splitting.

If the intrinsic backscattering is much larger than the backscattering at the target particle, $|A^{(2)}| \gg |V - U|$ , Eq. (8) simplifies to

$$
\Delta \Omega_ {\mathrm{EP}} = \Delta \Omega_ {\mathrm{DP}} e ^ {i m \beta} \sqrt {\frac {A ^ {(2)}}{V - U}}. \tag {9}
$$

Here, the absolute value of the complex frequency splitting $|\Delta\Omega_{EP}|\gg|\Delta\Omega_{DP}|$ is independent from the azimuthal position of the target particle, $\beta$ . But the corresponding real and imaginary parts do depend on $\beta$ . It is therefore necessary to measure both the frequency splitting $\mathrm{Re}(\Delta\Omega)$ and the linewidth splitting $-2\mathrm{Im}(\Delta\Omega)$ . This can be done as recent experiments show [1,45].

The proposed scheme works also for more than one target particle provided that the total perturbation remains small compared to the strength of the intrinsic backscattering. More precisely,

$$
\left| A ^ {(2)} \right| \gg \left| \sum_ {j = 3} ^ {N} (V _ {j} - U _ {j}) e ^ {- i 2 m \beta_ {j}} \right|. \tag {10}
$$

In such a case $\left|\Delta\Omega_{EP}\right|$ does depend on the relative azimuthal positions of the target particles, but this is also true for the conventional splitting $\left|\Delta\Omega_{DP}\right|$ ; see, e.g., Refs. [19,35].

In the following, we compare the analytical results in Eqs. (8) and (9) to full numerical simulations for transverse magnetic (TM) polarization using the BEM. For the microcavity sensor consisting of the disk and the first two particles we use $n = n_{1} = n_{2} = 2$ (material losses are ignored), $d_{1}/R = 0.01$ , $d_{2}/R = 0.02$ , $r_{1}/R = 0.043$ , $r_{2}/R = 0.048378$ , $\beta_{1} = 0$ , and $\beta_{2} = 1.08468132$ in radian. The radius R of the disk does not have to be specified if the dimensionless frequency $\Omega = (\omega/c)R = 2\pi(R/\lambda)$ is used. For example, with $\Omega \approx 10$ and $\lambda \approx 1\mu m$ we have $R \approx 1.6\mu m$ , $d_{1} \approx 16\ nm$ , $d_{2} \approx 32\ nm$ , $r_{1} \approx 68\ nm$ , and $r_{2} \approx 77\ nm$ . Note that this set of parameters serves as an example. There is no principle limitation in microdisk size, particle size, wavelength, or refractive indices.

Figure 2 shows the pair of modes corresponding to unperturbed modes with azimuthal mode number m = 16 and the lowest radial mode number. This pair is close to an EP with $\Omega \approx 9.8780 - i0.00243$ . This EP is the degeneracy that we use for our detection scheme. It can be seen that the modes look very similar. This is consistent with the coalescence of the modes exactly at the EP. Moreover, no clear nodal line structure in the azimuthal direction is visible in Fig. 2, reflecting the fact that the modes near such an EP are not standing waves but mainly traveling waves [35].

For the target particle with azimuthal position $\beta = \beta_{3}$ we fix $d_{3} / R = 0.015$ , $r_3 / R = 0.01$ , and $n_3 = 1.5$ . Figure 3 demonstrates good agreement between the full numerical results and analytical results according to Eqs. (8) and (9) ( $V_j$ , $U_j$ are determined beforehand using the BEM) for the frequency splitting $\Delta \Omega = \Omega_{+} - \Omega_{-}$ . For convenience, the splitting is normalized by the conventional splitting $|\Delta \Omega_{\mathrm{DP}}|$ occurring at a DP. It can be observed in Fig. 3(a) that $|\Delta \Omega|$ is (i) approximately independent of the particle's position $\beta$ and (ii) that it is enhanced by a factor of 3 to 3.5 if compared to $|\Delta \Omega_{\mathrm{DP}}|$ . Note that, according to Eq. (9), there is no principle limitation of the enhancement factor. In agreement with Eq. (9), Figs. 3(b) and 3(c) show that the real and the imaginary part of the frequency splitting undergo roughly a harmonic oscillation with period $\Delta \beta = 2\pi / m \approx 0.4$ . The small discrepancies between the

![](images/ae028a1ca4bd6ffa07cf04d31a4e69ef28bf749cfd044e84cb350315f8193ddc.jpg)

<details>
<summary>natural_image</summary>

Circular thermal or emission pattern with gradient from red to blue, no text or symbols present
</details>

![](images/51b4f33a308e4f41234747ea9cf3e49da7cbb79fa9b59129f78989a590e8e544.jpg)

<details>
<summary>natural_image</summary>

Circular thermal or emission pattern with concentric rings and color gradient (no text or symbols)
</details>

FIG. 2 (color online). Computed near-field intensity patterns of a mode pair in the microcavity sensor (without target particle) close to an exceptional point. (a) Dimensionless frequency $\Omega_{+} = 9.878070 - i0.002427$ and quality factor $Q_{+} \approx 2035$ . (b) $\Omega_{-} = 9.878033 - i0.002428$ and $Q_{-} \approx 2034$ .

![](images/5bdbc31fd56f2b7a04844206546da3bef2b60770e99fbd437893d7f684ed49c6.jpg)  
FIG. 3 (color online). Frequency splitting $\Delta\Omega$ normalized by the conventional splitting $|\Delta\Omega_{DP}|$ vs target particle position $\beta$ in radian. The black solid curve is the full numerical solution, the blue dash-dotted curve is the result of Eq. (8) and the red dashed line is obtained from Eq. (9). (a) Absolute value of $\Delta\Omega$ . A more than threefold enhancement of the sensitivity is observed. Inset: near-field intensity patterns of the mode pair for the target particle (arrow) at $\beta = 4$ . (b) Real and (c) imaginary part of $\Delta\Omega$ .

full numerical results and the analytical results in Fig. 3 are traced back to the two-mode approximation [35].

The mode pair in the microcavity sensor with a target particle [inset of Fig. 3(a)] exhibits a visible nodal line structure in contrast to the case without the target particle in Fig. 2. This is because the target particle drives the system slightly away from the EP (nonzero frequency splitting) and thereby introduces a standing-wave component into the mode structure.

It should be mentioned that there are two technical difficulties with our proposal. The first one is related to the particular geometry chosen in our example. Because of the two nanoparticles which implement the EP, the individual linewidths $\gamma_{\pm} = -2Im\Omega_{\pm}$ are increased to $\approx7|\Delta\Omega|$ , making it difficult to resolve the frequency splitting in experiments. Fortunately, the problem of resolving overlapping peaks has been solved in recent experiments by (i) interferometric detection of frequency splittings [46], (ii) the application of the harmonic inversion technique [47], and (iii) linewidth reduction by optical gain in an active microcavity [2,48]. We confirm the later approach by simulating optical gain in the microdisk using a negative imaginary part of the refractive index n. Figure 4 shows that the absolute value of the frequency splitting stays nearly constant whereas the individual linewidths of the two involved modes decay linearly with -Imn. For -Im n > 0.00043, the linewidths are smaller than the frequency splitting and therefore the latter can be resolved. Hence, using a microlaser is a promising approach for our purpose.

![](images/ab341ae3b6e0e1f268bfcf4b0467955e9dd7f9899317d6d7c58fa9fcab4cfb6c.jpg)  
FIG. 4 (color online). Full numerical solution of the dimensionless frequency splitting $|\Delta\Omega|$ (black solid curve) and individual linewidths $\gamma_{\pm} = -2Im\Omega_{\pm}$ (blue dash-dotted line and red dashed line) of the mode pair vs imaginary part of the refractive index; $\beta = 3.9$ .

The second technical difficulty is that there are certain values of $\beta$ at which the splitting $\mathrm{Re}(\Delta\Omega)$ vanishes, for example, at $\beta\approx4$ as shown in Fig. 3(b). Near these values of $\beta$ it is difficult to resolve the two peaks using a conventional Lorentzian curve fit, even though the linewidth splitting $|2\mathrm{Im}(\Delta\Omega)|$ is maximal. However, the harmonic inversion technique has proven to work reliably also in such extreme cases [47]. Hence, these technical difficulties can be overcome.

The discussed mechanism for microcavity sensors is not restricted to the geometry, size, wavelength, and refractive indices studied in this Letter. For instance, the EP could also be implemented by introducing holes or defects (see, e.g., Refs. [40,49]) into the disk. This would be advantageous as target particles can dock uniformly along the disk boundary. Another possibility would be to use a deformed microdisk cavity at an EP [18]. Moreover, other whispering-gallery cavities such as microrings, microtoroids, and microspheres can be adapted to our scheme.

To further stress the broad applicability of our idea we briefly address the topical field of nanomechanical mass sensing. One approach here is to measure the change of mechanical frequencies of a micro- or nanocantilever of mass M due to the attachment of a target mass $\Delta M \ll M$ [7]. Sensitivity can be enhanced by using two identical coupled cantilevers [50], but in both cases the sensitivity is proportional to $\varepsilon = \Delta M / M$ . Avoided frequency crossings based on coupling of nanomechanical modes have been experimentally demonstrated [51]. Such an avoided crossing can be converted to an EP by adjusting losses. For the coupled cantilever problem, approximated by two coupled harmonic oscillators [50], the appearance of EPs has been explicitly shown for nonidentical damped oscillators [14]. As for identical oscillators the perturbation by the target mass is proportional to $\varepsilon$ . At the EP this induces a frequency splitting of order $\sqrt{\varepsilon}$ [11] which can be exploited for ultrasensitive mass sensing.

In summary, we proposed to use exceptional points to improve the performance of sensors based on the detection

of frequency or energy splittings. It is demonstrated that the sensitivity of a microcavity sensor for single-particle detection can be enhanced provided that both the frequency splitting and the linewidth splitting can be measured. We believe that our approach not only helps to improve the sensitivity of detection schemes but also fosters the fascinating research on the physics of exceptional points.

I would like to thank F. Vollmer for helpful questions and comments. Financial support by the DFG (Project No. WI1986/6-1) is acknowledged.

[1] J. Zhu, Ş. K. Özdemir, Y.-F. Xiao, L. Li, L. He, D.-R. Chen, and L. Yang, Nat. Photonics 4, 46 (2010).   
[2] L. He, S. K. Özdemir, J. Zhu, W. Kim, and L. Yang, Nat. Nanotechnol. 6, 428 (2011).   
[3] F. Vollmer and L. Yang, Nanophotonics 1, 267 (2012).   
[4] W. W. Chow, J. Gea-Banacloche, L. M. Pedrotti, V. E. Sanders, W. Schleich, and M. O. Scully, Rev. Mod. Phys. 57, 61 (1985).   
[5] S. Sunada and T. Harayama, Opt. Express 15, 16245 (2007).   
[6] L. Rondin, J.-P. Tetienne, T. Hingant, J.-F. Roch, P. Maletinsky, and V. Jacques, arXiv:1311.5214v2.   
[7] E. Gil-Santos, D. Ramos, J. Martínez, M. Fernández-Regúlez, R. García, A. San Paulo, M. Calleja, and J. Tamayo, Nat. Nanotechnol. 5, 641 (2010).   
[8] Y. Liu, L. Zhang, J. A. R. Williams, and I. Bennio, IEEE Photonics Technol. Lett. 12, 531 (2000).   
[9] M. V. Berry and M. Wilkinson, Proc. R. Soc. A 392, 15 (1984).   
[10] I. Rotter, J. Phys. A 42, 153001 (2009).   
[11] T. Kato, Perturbation Theory for Linear Operators (Springer, New York, 1966).   
[12] W. D. Heiss, Phys. Rev. E 61, 929 (2000).   
[13] M. V. Berry, Czech. J. Phys. 54, 1039 (2004).   
[14] W. D. Heiss, J. Phys. A 37, 2455 (2004).   
[15] C. Dembowski, H.-D. Gräf, H. L. Harney, A. Heine, W. D. Heiss, H. Rehfeld, and A. Richter, Phys. Rev. Lett. 86, 787 (2001).   
[16] C. Dembowski, B. Dietz, H.-D. Gräf, H. L. Harney, A. Heine, W. D. Heiss, and A. Richter, Phys. Rev. E 69, 056216 (2004).   
[17] B. Dietz, T. Friedrich, J. Metz, M. Miski-Oglu, A. Richter, F. Schäfer, and C. A. Stafford, Phys. Rev. E 75, 027201 (2007).   
[18] S.-B. Lee, J. Yang, S. Moon, S.-Y. Lee, J.-B. Shim, S. W. Kim, J.-H. Lee, and K. An, Phys. Rev. Lett. 103, 134101 (2009).   
[19] J. Zhu, Ş. K. Özdemir, L. He, and L. Yang, Opt. Express 18, 23535 (2010).   
[20] Y. Choi, S. Kang, S. Lim, W. Kim, J.-R. Kim, J.-H. Lee, and K. An, Phys. Rev. Lett. 104, 153601 (2010).   
[21] H. Cartarius, J. Main, and G. Wunner, Phys. Rev. Lett. 99, 173003 (2007).   
[22] B. Alfassi, O. Peleg, N. Moiseyev, and M. Segev, Phys. Rev. Lett. 106, 073901 (2011).

[23] M. Liertzer, L. Ge, A. Cerjan, A. D. Stone, H. E. Türeci, and S. Rotter, Phys. Rev. Lett. 108, 173901 (2012).   
[24] A. P. Seyranian, O. N. Kirillov, and A. A. Mailybaev, J. Phys. A 38, 1723 (2005).   
[25] L. Chantada, N. I. Nikolaev, A. L. Ivanov, P. Borri, and W. Langbein, J. Opt. Soc. Am. B 25, 1312 (2008).   
[26] Q. H. Song and Y. L. Kim, J. Lightwave Technol. 28, 2818 (2010).   
[27] L. Deych, M. Ostroswski, and Y. Yi, Opt. Lett. 36, 3154 (2011).   
[28] D. S. Weiss, V. Sandoghdar, J. Hare, V. Lefèvre-Seguin, J.-M. Raimond, and S. Haroche, Opt. Lett. 20, 1835 (1995).   
[29] M. L. Gorodetsky, A. D. Pryamikov, and V. S. Ilchenko, J. Opt. Soc. Am. B 17, 1051 (2000).   
[30] T. J. Kippenberg, S. M. Spillane, and K. J. Vahala, Opt. Lett. 27, 1669 (2002).   
[31] F. Vollmer, S. Arnold, and D. Keng, Proc. Natl. Acad. Sci. U.S.A. 105, 20701 (2008).   
[32] J. T. Rubin and L. Deych, Phys. Rev. A 81, 053827 (2010).   
[33] T. J. Kippenberg, A. L. Tchebotareva, J. Kalkman, A. Polman, and K. J. Vahala, Phys. Rev. Lett. 103, 027406 (2009).   
[34] L. Shao, X.-F. Jiang, X.-C. Yu, B.-B. Li, W. R. Clements, F. Vollmer, W. Wang, Y.-F. Xiao, and Q. Gong, Adv. Mater. 25, 5616 (2013).   
[35] J. Wiersig, Phys. Rev. A 84, 063828 (2011).   
[36] A. E. Siegman, Lasers (University Science Books, Sausalito, CA, 1986).   
[37] J. Wiersig, J. Opt. A 5, 53 (2003).   
[38] A. Taflove and S.C. Hagness, Computational Electrodynamics the Finite-Difference Time-Domain Method (Artech House, London, 2000).   
[39] J. Shainline, S. Elston, Z. Liu, G. Fernandes, R. Zia, and J. Xu, Opt. Express 17, 23 323 (2009).   
[40] C. P. Dettmann, G. V. Morozov, M. Sieber, and H. Waalkens, Phys. Rev. A 80, 063813 (2009).   
[41] X. Yi, Y.-F. Xiao, Y.-C. Liu, B.-B. Li, Y.-L. Chen, Y. Li, and Q. Gong, Phys. Rev. A 83, 023803 (2011).   
[42] J. Wiersig, Phys. Rev. A 89, 012119 (2014).   
[43] J. Wiersig, S. W. Kim, and M. Hentschel, Phys. Rev. A 78, 053809 (2008).   
[44] J. Wiersig, A. Eberspächer, J.-B. Shim, J.-W. Ryu, S. Shinohara, M. Hentschel, and H. Schomerus, Phys. Rev. A 84, 023845 (2011).   
[45] A. Mazzei, S. Götzinger, L. de S. Menezes, G. Zumofen, O. Benson, and V. Sandoghdar, Phys. Rev. Lett. 99, 173603 (2007).   
[46] J. Knittel, T. G. McRae, K. H. Lee, and W. P. Bowen, Appl. Phys. Lett. 97, 123704 (2010).   
[47] U. Kuhl, R. Höhmann, J. Main, and H.-J. Stöckmann, Phys. Rev. Lett. 100, 254101 (2008).   
[48] L. He, S. K. Özdemir, J. Zhu, and L. Yang, Phys. Rev. A 82, 053810 (2010).   
[49] S. Preu, S. I. Schmid, F. Sedlmeir, J. Evers, and H. G. L. Schwefel, Opt. Express 21, 16370 (2013).   
[50] M. Spletzer, A. Raman, A. Q. Wu, and X. Xu, Appl. Phys. Lett. 88, 254102 (2006).   
[51] T. Faust, J. Rieger, M. J. Seitner, P. Krenn, J. P. Kotthaus, and E. M. Weig, Phys. Rev. Lett. 109, 037205 (2012).