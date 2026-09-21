# Single virus and nanoparticle size spectrometry by whispering-gallery-mode microcavities

Jiangang Zhu, $^{1,3}$ Sahin Kaya Özdemir, $^{1,4}$ Lina He, $^{1}$ Da-Ren Chen, $^{2}$ and Lan Yang $^{1,*}$

$^{1}$ Department of Electrical and Systems Engineering, Washington University, St. Louis, Missouri 63130, USA

$^{2}$ Department of Energy, Environmental and Chemical Engineering, Washington University, St.

Louis, Missouri 63130, USA

$^{3}jzhu@$ seas.wustl.edu

4 ozdemir@ese.wustl.edu

* yang@seas.wustl.edu

Abstract: Detecting and characterizing single nanoparticles and airborne viruses are of paramount importance for disease control and diagnosis, for environmental monitoring, and for understanding size dependent properties of nanoparticles for developing innovative products. Although single particle and virus detection have been demonstrated in various platforms, single-shot size measurement of each detected particle has remained a significant challenge.

Here, we present a nanoparticle size spectrometry scheme for label-free, real-time and continuous detection and sizing of single Influenza A virions, polystyrene and gold nanoparticles using split whispering-gallery-modes (WGMs) in an ultra-high-Q resonator. We show that the size of each particle and virion can be measured as they continuously bind to the resonator one-by-one, eliminating the need for ensemble measurements, stochastic analysis or imaging techniques employed in previous works. Moreover, we show that our scheme has the ability to identify the components of particle mixtures.

$\odot$ 2011 Optical Society of America

OCIS codes: (140.3945) Microcavities; (140.4780) Optical resonator; (130.6010) Sensors; (280.4788) Optical sensing and sensors.

# References and links

#148606 - $15.00 USD

Received 2 Jun 2011; revised 18 Jul 2011; accepted 19 Jul 2011; published 9 Aug 2011

(C) 2011 OSA

15 August 2011 / Vol. 19, No. 17 / OPTICS EXPRESS 16195

# 1. Introduction

With the increasing presence of nanoparticles in daily lives, there is a growing interest in assessing their benefits and risks. Meanwhile, there is also a strong need to detect and characterize biological nanoparticles such as viruses, which are responsible for the outbreak of many infectious diseases. A critical step in this assessment is to establish label-free, reliable and cost-effective techniques for real-time and on-site detection and quantification of individual viruses and nanoparticles. This will facilitate studies of physical and biological properties of single

148606 - $15.00 USD Received 2 Jun 2011; revised 18 Jul 2011; accepted 19 Jul 2011; published 9 Aug 2011

(C) 2011 OSA

15 August 2011 / Vol. 19, No. 17 / OPTICS EXPRESS 16196

viruses, and of size and material dependent properties of nanoparticles at single particle level.

Among various techniques [1-13], micro/nano-sized photonic [3-8] and electromechanical [9, 10] resonators are emerging as forerunners for label-free detection of single nanoparticles and molecules due to their immense susceptibility to perturbations in their environments which enhances sensitivity and resolution, and due to the increasing demand for shrinking device dimensions to achieve massive parallelism and integration with the existing micro/nanosystems.

The sensing mechanism used in the existing photonic and electromechanical resonators relies on the detection of the spectral shift of a resonance mode upon the landing of a nanoparticle within the mode volume. The amount of the spectral shift depends on both the position of the particle and its properties such as mass, size, refractive index or polarizability. Thus, although each arriving particle or virion can be detected, accurate quantitative measurement of the properties of each arriving particle and virion cannot be done.

Instead, statistical analysis, such as building histograms of event probability versus spectral shift for ensembles of sequentially arriving particles of similar properties, is used to extract the required information (e.g., size, mass or polarizability) [4, 10]. This prevents single-shot measurement of each particle and real-time sizing capabilities. Moreover, an equally important issue affecting these schemes is the absence of a reference. The shift induced by a nanoparticle is very small, and it is sensitive to instrumental noise and environmental disturbances.

Thus, discriminating between the interactions of interest and the interfering perturbations becomes difficult.

Recently, we have introduced position-independent size measurement for a single nanoparticle by using scattering-induced mode-splitting in an ultra-high-quality $(Q)$ WGM microtoroid resonator [14]. Although this prototype scheme allowed to overcome some difficulties (i.e., position-dependence and lack of reference) associated with measurement of nanoparticles, there remained critical issues to be solved. First, the sizing method is applicable only for WGM without observable splitting. One has to find a splitting-free WGM for measurement of the particle. Since intrinsic mode splitting (e.g.

, not intentionally induced or caused by the target scatterers) [15, 16] takes place in almost all practical realizations of ultra-high-Q resonators due to contaminations or structural inhomogeneities, it is difficult to find such an initially splitting-free WGM. Second, position independent size estimation was valid only for the first nanoparticle as the previous model did not explain how the split modes are affected when the particles sequentially enter the mode volume after the first one.

In this study, we demonstrate a technique which overcomes the above mentioned limitations of existing resonator-based schemes thus allowing the detection, counting and size measurement of consecutively adsorbed nanoparticles and virions one-by-one at single particle resolution. This new mode splitting based approach requires neither simultaneous excitation, frequency-locking and tracking of multiple modes of a resonator nor histogram preparations or stochastic analysis.

We achieved label-free detection and accurate size-measurement of InfA virions, gold (Au) and polystyrene (PS) particles down to $R = 30\mathrm{nm}$ . The ability to detect and measure single virions/nanoparticles allows determining their polarizability and size distributions. The demonstrated techniques offer the possibility of an ultra-compact single nanoparticle/biomolecule size spectrometry system for in-lab or in-field use.

# 2. Experiments

Single Virus/Nanoparticle Size Spectrometry Figure 1(a) schematically depicts our experimental system. Microresonators used in our experiments are silica microtoroids prepared by photolithography followed by $\mathrm{CO}_{2}$ laser re-flow [17]. The resonators have quality factors above $10^{7}$ and diameters below $40~\mu \mathrm{m}$ . A tunable $670~\mathrm{nm}$ band laser is used in experiments with viri

148606 - $15.00 USD Received 2 Jun 2011; revised 18 Jul 2011; accepted 19 Jul 2011; published 9 Aug 2011

(C) 2011 OSA

15 August 2011 / Vol. 19, No. 17 / OPTICS EXPRESS 16197

![](dt=2026-06-07/ht=17/6675656d71d6bc97610248162e78b0561d212deca7df9b983d9ff0f95ab81b80.jpg)

![](dt=2026-06-07/ht=17/817259501f4ff7e1a1123c78f2060bad6fcad74e0d819fd3cf4e47f1aa0962f1.jpg)

ons and nanoparticles smaller than $\mathrm{R} = 70\mathrm{nm}$ , and a $1550~\mathrm{nm}$ band laser is used in experiments with bigger particles. A fiber taper prepared from a standard single mode fiber by heat-and-pull technique is used to couple light into and out of the microresonator. Transmitted light is detected by a photodetector which is connected to an oscilloscope for monitoring the transmission spectra (Fig. 1(b)). The spectra are acquired to the computer at rate of 10 frames per second, and processed by doub
le Lorenzian curve fitting to find the frequencies and linewidths of the resonances.

Mode splitting can be induced either by the scattering centers due to structural inhomogeneities (i.e., intrinsic mode splitting) [15, 16] or by the scatterers intentionally introduced into the resonator mode volume [14, 15, 18]. The underlying physics of the interaction between a resonant mode and a single sub-wavelength scatterer in the mode volume was studied using the dipole approximation [18]. The resonator-scatterer interaction lifts the frequency-degeneracy of the two counter-propagating WGMs of the resonator and splits the single resonance into two, leading to two standing wave modes (SWMs). Consequently, the two SWMs are identified in the transmission spectrum as a doublet with two spectrally shifted resonance modes of different linewidths (Fig. 1(b)).

For the deposition of nanoparticles and viruses, a set-up consisting of a differential mobility analyzer (DMA) and a nozzle with an inner tip diameter of $80\mu \mathrm{m}$ was used. First, particles are carried out by compressed air using a collision atomizer. The solvent in droplets is then evaporated in a dryer with the silica gel desiccants. Solid particles are further neutralized by a radioactive source such that they have a well-defined charge distribution. Then they are sent to the DMA where they are classified according to their electrical mobility. Particles within a narrow range of mobility can exit through the output slit. The nozzle was placed at about 300

148606 - $15.00 USD Received 2 Jun 2011; revised 18 Jul 2011; accepted 19 Jul 2011; published 9 Aug 2011

(C) 2011 OSA

15 August 2011 / Vol. 19, No. 17 / OPTICS EXPRESS 16198

$\mu \mathrm{m}$ away from the microtoroid, and particles are blown directly towards the microtoroid. Purified and inactivated Influenza virus X-31 A/AICHI/68 was purchased in 4-(2-hydroxyethyl)-1-piperazineethanesulfonic acid (Hepes) buffer from Charles River Laboratories. The virus sample was passed through a $0.2\mu \mathrm{m}$ nylon membrane filter to remove aggregates. Gold nanoparticles (50 and $100\mathrm{nm}$ ) were purchased from British Biocell International Limited. PS nanoparticles (50–135 nm) are from Thermo Scientific (Duke Standards 3K series).

Figure 2 shows the mode splitting spectra induced by InfA virions entering the resonator mode volume one-by-one. With the arrival of the first virion, the single resonance splits into two. The subsequent single virion adsorptions lead to redistribution of the existing SWMs and abrupt jumps in the splitting spectra (Fig. 2(a)). The sudden changes in the frequencies and linewidths of the resonance modes signal the detection of particle adsorption events, and the amount of change depends on the virion size and its position in the mode volume.

In the next section, we will explain how to take advantage of the split modes for constructing a position-independent measurement scheme for individual scatterers. The mode splitting resonance frequencies and linewidths extracted from the spectra in Fig. 2(a) show step changes corresponding to each individual virion adsorption event (Figs. 2(b) and 2(c)). Processing this data allows to extract the polarizability and hence the size of each virion (Fig. 2(g)), as will be shown in the next section.

Note that particles deposited outside the mode volume do not affect the WGM, so they have no effect on the resonance spectrum.

# 3. Theory

Single Particle Model Let's denote the two SWMs formed after the adsorption of a single nanoparticle/virion as lower and higher frequency modes with the corresponding resonance frequencies and linewidths denoted as $(\omega_{1}^{-},\gamma_{1}^{-})$ and $(\omega_{1}^{+},\gamma_{1}^{+})$ .

The dipole approximation predicts that the spectral distance of these two modes is given by the coupling coefficient between the counter-propagating WGMs as $2g = -\alpha f^2 (\mathbf{r})\omega /V$ , and the linewidth difference due to coupling of the WGMs to the environment via scattering is given as $2\Gamma = \alpha^{2}f^{2}(\mathbf{r})\omega^{4} / 3\pi \nu^{3}V$ .

Here $\omega$ is the angular resonant frequency, $V$ is the microcavity mode volume, $\nu = c / \sqrt{\varepsilon_m}$ with $c$ representing the speed of light, and $f(\mathbf{r})$ is a scalar quantity and designates the normalized mode intensity distribution. The polarizability $\alpha$ of the scatterer is $\alpha = 4\pi R^3 (\varepsilon_p - \varepsilon_m) / (\varepsilon_p + 2\varepsilon_m)$ for a spherical particle of radius $R$ and electric permittivity $\varepsilon_{p}$ in a surrounding medium of electric permittivity $\varepsilon_{m}$ (e.g., $\varepsilon_{m} = 1$ for air).

Subsequently, the frequency shift $(\Delta \omega_{1}^{-},\Delta \omega_{1}^{+})$ and linewidth change $(\Delta \gamma_1^{-},\Delta \gamma_1^{+})$ of the split modes (doublet) with respect to the resonance frequency $\omega_0$ and the linewidth $\gamma_0$ of the initial (pre-scatterer) WGM are

$$
\Delta \omega_ {1} ^ {-} = \omega_ {1} ^ {-} - \omega_ {0} = 2 g _ {1}, \quad \Delta \omega_ {1} ^ {+} = \omega_ {1} ^ {+} - \omega_ {0} = 0 \tag {1}
$$

$$
\Delta \gamma_ {1} ^ {-} = \gamma_ {1} ^ {-} - \gamma_ {0} = 2 \Gamma_ {1}, \quad \Delta \gamma_ {1} ^ {+} = \gamma_ {1} ^ {+} - \gamma_ {0} = 0. \tag {2}
$$

Thus, for $\varepsilon_{m} = 1$ , the polarizability of the scatterer becomes

$$
\alpha_ {1} = - \frac {\Gamma_ {1}}{g _ {1}} \frac {3 \lambda^ {3}}{8 \pi^ {2}} = - \frac {\Delta \gamma_ {1} ^ {-}}{\Delta \omega_ {1} ^ {-}} \frac {3 \lambda^ {3}}{8 \pi^ {2}} \tag {3}
$$

which is independent of the position $\mathbf{r}$ of the scatterer in the mode volume. In a typical experiment, $2g$ and $2\Gamma$ , i.e. the frequency separation and linewidth difference of the two split modes, are measured from the transmission spectrum, and subsequently, the polarizability of the scatterer is derived using Eq. (3), assuming particles are spherical.

Multi-Particle Model In the case of multiple scatterers, with each new scatterer entering the resonator mode volume, the resonator-scatterer interaction changes, leading to redistribution of SWMs. Consequently, the locations of nodes and anti-nodes of the SWMs with respect to the individual scatterers are modified (Media 1).

148606 - $15.00 USD Received 2 Jun 2011; revised 18 Jul 2011; accepted 19 Jul 2011; published 9 Aug 2011

(C) 2011 OSA

15 August 2011 / Vol. 19, No. 17 / OPTICS EXPRESS 16199

![](dt=2026-06-07/ht=17/6bdb251a97aaf9be50e315267c85fb96da4a575333ae3d7349c95ca2f8899add.jpg)

![](dt=2026-06-07/ht=17/be5822746e16863fe6bbcbab1b61e8067012f078d4c4b893a9766edb0edb6c59.jpg)

![](dt=2026-06-07/ht=17/a31767db60feb3a1ee6a139c30e212f9014e70da3ff33e66eb77d761480e555b.jpg)

![](dt=2026-06-07/ht=17/3003bf348692585cd01040643fe057a15f68d7252c4421b7df068a627b0e9674.jpg)

![](dt=2026-06-07/ht=17/d14bab39f27eb4c0e9755e43061a1a8f4316f1a03b94117582b05c53ebad775e.jpg)

![](dt=2026-06-07/ht=17/28c362887685d7663e5bd42b608e08c0d72af165dab55b273d09e749244dc0d1.jpg)

![](dt=2026-06-07/ht=17/fa5ab735cc89d261ce68808f645365921719259c98464c3a53d7f0af2a99f2ba.jpg)

#148606 - $15.00 USD

Received 2 Jun 2011; revised 18 Jul 2011; accepted 19 Jul 2011; published 9 Aug 2011

(C) 2011 OSA

15 August 2011 / Vol. 19, No. 17 / OPTICS EXPRESS 16200

The locations of each scatterers with respect to each other in the resonator mode volume determines
the distribution of SWMs. Assuming $N$ -scatterers in the mode volume, we define $\phi_N$ as the spatial phase distance between the antinode of the $\omega_N^-$ mode and the 1st scatterer, and $\beta_i$ as the spatial phase distance between the $i$ -th and the 1st scatterer. We can write the frequency shift and the linewidth broadening experienced by the split modes as

$$
\Delta \omega_ {N} ^ {-} = \sum_ {i = 1} ^ {N} 2 g _ {i} \cos^ {2} \left(\psi_ {N i}\right), \quad \Delta \omega_ {N} ^ {+} = \sum_ {i = 1} ^ {N} 2 g _ {i} \sin^ {2} \left(\psi_ {N i}\right) \tag {4}
$$

$$
\Delta \gamma_ {N} ^ {-} = \sum_ {i = 1} ^ {N} 2 \Gamma_ {i} \cos^ {2} \left(\psi_ {N i}\right), \quad \Delta \gamma_ {N} ^ {+} = \sum_ {i = 1} ^ {N} 2 \Gamma_ {i} \sin^ {2} \left(\psi_ {N i}\right) \tag {5}
$$

where $\psi_{Ni} = \phi_N - \beta_i$ is the spatial phase distance between the antinode of the $\omega_N^-$ mode and $i$ -th scatterer. $2g_{i}$ and $2\Gamma_{i}$ only depend on the $\alpha_{i}$ with the relation defined in the single mode case. They respectively characterize the splitting and linewidth difference of the split modes if the $i$ -th scatterer is the only scatterer in the mode volume, i.e. i.e. a single scatterer locates at the anti-node of one SWM.

When multiple particles bind to the microcavity at arbitrary locations, the anti-node does not necessarily correspond to the location of a particle. The deviation of the $i$ -th scatterer from the anti-node can be characterized by the spatial phase distance, $\psi_{Ni}$ , which can be used to scale the particle-field interactions. Since the SWMs exhibit sinusoidal spatial patterns, the $\cos^2 (\dots)$ and $\sin^2 (\dots)$ terms in Eqs. (4) and (5) correct the interaction strength between a scatter with each of the two SWMs.

Imposing the condition that the SWMs distribute themselves to maximize mode splitting [19, 20], we find that $\phi_N$ should be adjusted to satisfy

$$
\tan \left(2 \phi_ {N}\right) = \frac {\sum_ {i = 1} ^ {N} g _ {i} \sin \left(2 \beta_ {i}\right)}{\sum_ {i = 1} ^ {N} g _ {i} \cos \left(2 \beta_ {i}\right)}. \tag {6}
$$

Equivalently, one SWM is maximally shifted and the other SWM is minimally shifted, and the two SWMs are orthogonal to each other. The resonance wavelength of a SWM is proportional to, by a factor of azimuthal wavenumber, the round-trip optical path length. Therefore maximizing or minimizing the frequency (wavelength) shift equals to maximizing or minimizing the optical round trip path length. The underlying physical mechanism can be intuitively understood from the Fermat's principal, which states that rays of light traverse the path of stationary (could be maximal or minimal) time [20, 31].

Next, we define $\delta_N^{-} = \Delta \omega_N^{+} - \Delta \omega_N^{-} = \omega_N^{+} - \omega_N^{-}$ as the mode splitting between the two modes and $\delta_N^+ = \Delta \omega_N^+ +\Delta \omega_N^- = \omega_N^+ +\omega_N^- - 2\omega_0$ as the total frequency shift. Similarly, $\rho_{N}^{-} = \Delta \gamma_{N}^{+} - \Delta \gamma_{N}^{-} = \gamma_{N}^{+} - \gamma_{N}^{-}$ corresponds to the linewidth difference between the two split modes and $\rho_N^+ = \Delta \gamma_N^+ +\Delta \gamma_N^- = \gamma_N^+ +\gamma_N^- - 2\gamma_0$ corresponds to the sum of the linewidth change. Using the definitions of $\delta_N^{\pm}$ and $\rho_N^{\pm}$ and Eqs. (4) and (5), we find

$$
\delta_ {N} ^ {-} = 2 \sum_ {i = 1} ^ {N} g _ {i} \cos (2 \psi_ {N i}), \quad \delta_ {N} ^ {+} = 2 \sum_ {i = 1} ^ {N} g _ {i} \tag {7}
$$

$$
\rho_ {N} ^ {-} = 2 \sum_ {i = 1} ^ {N} \Gamma_ {i} \cos \left(2 \psi_ {N i}\right), \quad \rho_ {N} ^ {+} = 2 \sum_ {i = 1} ^ {N} \Gamma_ {i} \tag {8}
$$

In practical realizations, it is not possible to know the exact values of $\psi_{Ni}$ , hence $\delta_N^{-}$ and $\rho_N^{-}$ , to extract useful information of the deposited scatterers. However, one can use $\delta_N^+$ and $\rho_N^+$ because they only depend on $g_{i}$ and $\Gamma_{i}$ which are directly related to the polarizability of the $i$ -th scatterer.

148606 - $15.00 USD Received 2 Jun 2011; revised 18 Jul 2011; accepted 19 Jul 2011; published 9 Aug 2011

(C) 2011 OSA

15 August 2011 / Vol. 19, No. 17 / OPTICS EXPRESS 16201

Consequently, we can write the polarizability of the $N$ -th particle $\alpha_{N}$ as

$$
\begin{array}{l} \alpha_ {N} = - \frac {\Gamma_ {N}}{g _ {N}} \frac {3 \lambda^ {3}}{8 \pi^ {2}} = - \frac {3 \lambda^ {3}}{8 \pi^ {2}} \frac {\rho_ {N} ^ {+} - \rho_ {N - 1} ^ {+}}{\delta_ {N} ^ {+} - \delta_ {N - 1} ^ {+}} \\ = - \frac {3 \lambda^ {3}}{8 \pi^ {2}} \frac {\left(\gamma_ {N} ^ {+} + \gamma_ {N} ^ {-}\right) - \left(\gamma_ {N - 1} ^ {+} + \gamma_ {N - 1} ^ {-}\right)}{\left(\omega_ {N} ^ {+} + \omega_ {N} ^ {-}\right) - \left(\omega_ {N - 1} ^ {+} + \omega_ {N - 1} ^ {-}\right)} \tag {9} \\ \end{array}
$$

which states that the polarizability of the N-th scatterer can be calculated just by comparing the total frequencies and linewidths of the split modes right before and after its deposition. Then the radius $R_N$ can be calculated as

$$
R _ {N} = \left[ \frac {\alpha_ {N}}{4 \pi} \frac {\varepsilon_ {p} + 2}{\varepsilon_ {p} - 1} \right] ^ {1 / 3}. \tag {10}
$$

# 4. Discussion

Mode Splitting Size Spectrometry of Single InfA Virions The spectrogram shown in Fig. 2(a) presents examples of transmission spectra obtained in experiments with InfA virions. With each consecutive individual nanoparticle adsorption, the frequency and the linewidth of the split modes change abruptly. The heights of discrete jumps depend on the positions of the virions relative to the SWMs according to Eqs. (4)-(6).

Extracted frequencies and linewidths of the split resonances from the experimental data shown in Fig. 2(a) are depicted in Figs. 2(b,c). This information is subsequently used to calculate $\delta_N^+$ and $\rho_N^+$ (Fig. 2(d) and 2(e)). Single virion adsorption events are clearly visible as discrete jumps in Figs. 2(d) and 2(e). Although the height of each discrete jump depends on the position of each virion within the resonator mode volume, we can accurately measure the size regardless of the virion position. Using Eqs.

(9) and (10), we estimated the polarizability from which the size of the adsorbed virions was derived and presented in Fig. 2(g). Assuming a refractive index of 1.48 [11] for virions, we calculated the radii of the adsorbed virions to be in the range $46 - 55\mathrm{nm}$ , for the data in Fig. 2. As seen in Fig. 2(e), the change in total linewidth $\rho_N^+$ for the fourth virion adsorption event is within the noise level of our system.

Although the estimated size for this virion differs from the expected nominal size, this does not prevent detecting this virion thanks to the distinct change in total frequency $\delta_N^+$ (Fig. 2(d)).

We obtained the polarizability and size distributions of InfA virions by performing many experiments using different resonators. The results are depicted in Figs. 3(a) and 2(b). Measured radius $R = 53.2 \pm 5.5 \mathrm{~nm}$ for InfA virions agrees very well with the values reported in the literatures [4, 11, 22].

Having identified that the developed model and the scheme allow measuring the polarizability and size of individual InfA virions, we set out to measure polarizability and size distributions of Au and PS nanoparticles. For comparison, we show in Figs. 3(a) and 3(c) the experimentally obtained polarizability distributions of Au nanoparticles with $R = 50 \mathrm{~nm}$ and $R = 100 \mathrm{~nm}$ . Figure 3(d) depicts the distribution of estimated sizes for PS particles of $R = 100 \mathrm{~nm}$ and $R = 135 \mathrm{~nm}$ . The measured distributions of the tested nanoparticles are significantly different correlating with their sizes and material properties.

Size Resolution and Detection Limit For PS particles with nominal radius of $R = 100 \pm 1.7$ nm and $R = 135 \pm 2.1$ nm, our size estimation yielded $R = 101.2 \pm 9.05$ nm and
$R = 135.9 \pm 9.96$ nm, respectively. The standard deviations of measured polarizability distributions (Fig. 3(a) and 3(c)) for Au particles are $32\%$ and $31\%$ , respectively for $R = 50$ nm and $R = 100$ nm. These are slightly larger than the $24\%$ polarizability deviation estimated from the $8\%$ size deviation claimed by the manufacturer.

148606 - $15.00 USD Received 2 Jun 2011; revised 18 Jul 2011; accepted 19 Jul 2011; published 9 Aug 2011

(C) 2011 OSA

15 August 2011 / Vol. 19, No. 17 / OPTICS EXPRESS 16202

![](dt=2026-06-07/ht=17/31a73f54b9e07c92da083073f309c7ca8c9845b78ffce0f9c24433432be1dbdd.jpg)

![](dt=2026-06-07/ht=17/846b665fcb11df54bb3bbcdcc6d3d5f843de1ebf68e2fbd99f7ef57c7556d139.jpg)

![](dt=2026-06-07/ht=17/16905d35503f467ca24334cb8059d9611d96aabc9a9ad2d461c78223727b26a9.jpg)

![](dt=2026-06-07/ht=17/3d1432ea06999a6ea2365efe5a8108983548d13e68d7999c2ca10b1c941857c1.jpg)

The standard deviation of the estimated particle sizes and polarizabilities using our technique have four main contributions: (i) Standard deviation of the particles, (ii) detection noise and the laser frequency fluctuations, (iii) curve fitting noise in extracting the resonance frequencies and linewidths of the split modes, and (iv) fluctuations in the taper-resonator gap. We performed all experiments in normal laboratory environment with no active control of the conditions. Thus, we believe that the reported results can be improved by proper conditioning and control of laser phase and intensity noise as well as taper-resonator gap fluctuations.

Theoretical detection limit of our scheme is mainly dependent on $Q / V$ of the resonator and the wavelength of the resonance. For a dielectric nanoparticle of refractive index 1.5, detection limit is around $R = 10 \mathrm{~nm}$ with an ultra-high-Q resonance in the $670 \mathrm{~nm}$ wavelength band. In our experiments using microtoroids with $Q \geq 10^8$ , the smallest detected PS particles were of radii $R = 20 \mathrm{~nm}$ (nominal value provided by the manufacturer), and the smallest PS particles detected and accurately measured were of radii $R = 30 \mathrm{~nm}$ . These are the smallest dielectric nanoparticles ever detected and measured using optical resonators.

We performed accurate size measurement of up to 50 nanoparticles consecutively deposited on a single WGM resonator with high- $Q$ , without cleaning of the resonator. On the other hand, hundreds of virions or nanoparticles can be detected with the same resonator. This discrepancy in the detection and measurement limits can be explained as follows. In order to detect a single virion/nanoparticle binding event, it is sufficient to detect any change in either the resonance frequencies or the linewidths.

However, accurate size measurement requires that the changes in both the frequencies and the linewidths are accurately measured. Thus, size measurement imposes a much stricter condition. For example, as the number of particles in the resonator mode volume increases, the increasing scattering loss leads to broadening of the resonance linewidths. Eventually, the change induced in the linewidths by a single virion/nanoparticle falls within the noise level (i.e., similar to the fourth virion event in Fig. 2).

In such a case, linewidth information is partially or completely lost, and size information cannot be extracted

#148606 - $15.00 USD Received 2 Jun 2011; revised 18 Jul 2011; accepted 19 Jul 2011; published 9 Aug 2011

(C) 2011 OSA

15 August 2011 / Vol. 19, No. 17 / OPTICS EXPRESS 16203

correctly. However, there may still be a discernible change in frequency which would allow detection of particle binding. Indeed, accurate detection of resonance frequency changes has a higher saturation limit and is less prone to noise than the linewidth measurement.

Measurement of Nanoparticle Mixtures We challenged our system with a mixture of PS and gold nanoparticles with radii $R = 50$ nm. The measured polarizability distributions are shown in Fig. 4. The two maxima are easily seen and the two distributions have small overlap suggesting that our method can be reliably used to detect multiple components of a homogenously mixed ensemble of particles and to decide whether the given composition of particle ensemble is mono or poly-modal. This is expected as our scheme measures nanoparticles one-by-one. No apriori information is needed to differentiate particles of different polarizabilities.

![](dt=2026-06-07/ht=17/eac0914eccdb06eafd33fe544ccd6dcfad2688b8c6a924141828b78023b1551e.jpg)

Resolvability of Mode Splitting In order to resolve mode splitting in the transmission spectra after the deposition of the $N$ -th particle, $\delta_N^- > \rho_N^+ / 2 + \gamma_0$ should be satisfied [23,24]. Consequently, mode splitting quality $Q_{\mathrm{sp}} = 2\delta_N^- / (\rho_N^+ + 2\gamma_0)$ should be larger than one, $Q_{\mathrm{sp}} > 1$ [23]. The change in $Q_{\mathrm{sp}}$ as the InfA virions bind to the resonator is shown in Fig. 2(f) where we see that $Q_{\mathrm{sp}} > 1$ is satisfied during the measurements.

As the particle binding continues, each additional scatterer increases the linewidths, at some point $Q_{\mathrm{sp}}$ may become less than one and mode splitting can no longer be resolved. However, even in such cases, one can extract some useful information if we assume that mode splitting is much smaller than the individual linewidths $\omega_N^+ - \omega_N^-\ll \gamma_N^-, \gamma_N^+$ , and the two resonances have similar linewidths $\gamma_N^-\simeq \gamma_N^+$ .

In such a case, the transmission spectrum will show a single lorentzian peak with a linewidth of $\gamma_N = \sqrt{\gamma_N^- \gamma_N^+} \simeq (\gamma_N^- + \gamma_N^+)/2$ and a resonance at $\omega_N \simeq (\omega_N^- + \omega_N^+)/2$ . This expressions then can be used in Eqs. (9) and (10) to calculate the polarizability and the size of the N-th particle, provided that $\gamma_N - \gamma_{N-1}$ and $\omega_N - \omega_{N-1}$ are resolvable [25]. It should be noted that working in the split mode regime allows a lower measurable particle size for single particle analysis.

Measurement of Ensembles of InfA Virions Discussions and experimental results presented in the previous sections clearly demonstrate that our scheme is very effective and efficient in estimating the polarizability of individual nanoparticles entering the resonator mode volume one-by-one. Indeed, during continuous deposition of nanoparticles, mode splitting spectra at any instant is related to the effective polarizability of already deposited nanoparticles. If we assume that $N$ -particles are deposited to the mode volume of the resonator, we can assign an effective polarizability $\alpha_{\mathrm{eff}}$ sensed by the resonator using

$$
\alpha_ {\text {e f f}} = - \frac {3 \lambda^ {3}}{8 \pi^ {2}} \frac {\rho_ {N} ^ {-}}{\delta_ {N} ^ {-}} = - \frac {3 \lambda^ {3}}{8 \pi^ {2}} \frac {\left(\gamma_ {N} ^ {+} - \gamma_ {N} ^ {-}\right)}{\left(\omega_ {N} ^ {+} - \omega_ {N} ^ {-}\right)} \tag {11}
$$

148606 - $15.00 USD Received 2 Jun 2011; revised 18 Jul 2011; accepted 19 Jul 2011; published 9 Aug 2011

(C) 2011 OSA

15 August 2011 / Vol. 19, No. 17 / OPTICS EXPRESS 16204

If all the deposited particles are the same, Eqs. (7) and (8) reveal that $\alpha_{\mathrm{eff}}$ of Eq. (11) corresponds to the polarizability of a single particle. For verification, we used the data presented in Fig. 2 for InfA virions. The calculated size using Eq. (11) is given in Fig. 5. The result coincides well with the sizes of single viri
ons acquired in Fig. 2(g). Notice that the noise level decreases as the splitting quality $Q_{\mathrm{sp}}$ increases (Fig. 2(f)).

This method of size estimation requires that all the particles on the resonator have very similar sizes and materials, and the mode splitting has decent quality. For example, the effective polarizability of a virus ensemble coated onto a resonator pre-treated with specific antibody receptors can be measured using this approach. Selectivity of virus-antibody binding will then determine the accuracy of this scheme.

![](dt=2026-06-07/ht=17/683b7becae56b364afcec8ef8cb2f2cec07534c00e322b170d7056d4a76fb23e.jpg)

# 5. Conclusion

We have shown that adsorption of individual viruses and nanoparticles leads to discrete changes in the mode splitting spectra of a WGM microcavity. We developed an accurate and efficient method to detect and measure individual nanospecies one-by-one as they are adsorbed in the mode volume of a microresonator and experimentally verified it using InfA virions, PS and Au nanoparticles of various sizes. We achieved this by developing a new theoretical model and measurement strategy which take into account the effect of multiple scatterers deposited on an optical WGM resonator.

Contrary to the existing schemes, this new approach works equally well regardless of whether there is intrinsic mode splitting or whether a particle is deposited in the resonator mode volume before the actual measurement starts. The particles are characterized accurately regardless of their positions in the mode volume without the need for complicated processes such as stochastic analysis or excitation and tracking of multiple resonant modes. Moreover, our method is capable of identifying the modality of mixtures of nanoparticle ensembles.

Thus, the proposed single nanoparticle size spectrometry technique provides a suitable platform for in-situ, real-time and highly sensitive detection and sizing of individual nano-sized particles and viruses.

Since nanoparticle induced mode splitting has been demonstrated in water [26], the techniques developed here could be effectively extended to aqueous environment and incorporated into microfluidic or lab-on-chip devices which will pave the way for detecting and sorting of single bio-molecules/particles based on their polarizability or size. Although in this work, we considered spherical particles and isotropic polarizability, our method could be applied to distinguish between spherical and nonspherical particles by probing the particles with light fields of orthogonal polarizations [27, 28]. Moreover, the demonstrated techniques are not limited to microtoroidal resonators and in principle can be used with any WGM resonator (e.g., micro

148606 - $15.00 USD Received 2 Jun 2011; revised 18 Jul 2011; accepted 19 Jul 2011; published 9 Aug 2011

(C) 2011 OSA

15 August 2011 / Vol. 19, No. 17 / OPTICS EXPRESS 16205

sphere, microdisk or microring), independent of the mode number and polarization. We should note that for spherical particles, the estimated polarizability using our technique is the same for all polarizations; however, for non-spherical particles different polarizations will give different polarizabilities. This can be used to estimate the shape and size of the particles. [25,30,31] Further improvements in detection and size measurement limits could be made by improving the system stability, using noise reduction methods (e.g. [8, 29]), as well as employing gain-media doped microresonators [32].

We believe that the abilities provided by this single nanoparticle size spectrometry scheme will find applications in in bio/chemical sensing, environmental monitoring, pharmaceutical diagnosis and biomedical researches and nanotechnologies where size dependent properties of individual particles and their interactions play significant roles.

# Acknowledgments

The authors gratefully acknowledge the support from NSF under Grant No. 0954941. This work was performed in part at the NRF-NNIN (NSF Grant No. ECS-0335765) of Washington University in St. Louis. We also thank W. Kim, L. Li, F. Monifi for discussions.

148606 - $15.00 USD Received 2 Jun 2011; revised 18 Jul 2011; accepted 19 Jul 2011; published 9 Aug 2011 (C) 2011 OSA 15 August 2011 / Vol. 19, No. 17 / OPTICS EXPRESS 16206