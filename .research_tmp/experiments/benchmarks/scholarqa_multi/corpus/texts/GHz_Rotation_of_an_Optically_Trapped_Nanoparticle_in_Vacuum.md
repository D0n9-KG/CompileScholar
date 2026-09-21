# GHz Rotation of an Optically Trapped Nanoparticle in Vacuum

René Reimann, $^{1,*}$ Michael Doderer, $^{1}$ Erik Hebestreit, $^{1}$ Rozenn Diehl, $^{1}$ Martin Frimmer, $^{1}$ Dominik Windey, $^{1}$ Felix Tebbenjohanns, $^{1}$ and Lukas Novotny $^{1}$ Photonics Laboratory, ETH Zürich, 8093 Zürich, Switzerland

We report on rotating an optically trapped silica nanoparticle in vacuum by transferring spin angular momentum of light to the particle's mechanical angular momentum. At sufficiently low damping, realized at pressures below $10^{-5}$ mbar, we observe rotation frequencies of single 100 nm particles exceeding 1 GHz. We find that the steady-state rotation frequency scales linearly with the optical trapping power and inversely with pressure, consistent with theoretical considerations based on conservation of angular momentum. Rapidly changing the polarization of the trapping light allows us to extract the pressure-dependent response time of the particle's rotational degree of freedom.

Introduction. Optomechanics is the science of measuring and controlling mechanical motion using light $[1]$ . One particularly interesting optomechanical system is a dielectric nanoparticle levitated in a strongly focused laser beam using the forces of light $[2–7]$ . The trapping laser confines the particle to the focal region and scatters off the particle, providing a measurement of its center-of-mass motion. Using active feedback mechanisms and autonomous cavity-assisted cooling schemes, the center-of-mass motion of an optically levitated nanoparticle has been controlled to a remarkable degree, putting the quantum regime of mechanical motion within reach $[8–11]$ . Recently, researchers have started to turn their attention to rotational degrees of freedom of levitated objects $[12]$ . Taking inspiration from optically induced rotation of particles trapped in liquid media $[13]$ , the torsional and rotational motion of optically levitated objects featuring shape asymmetries or anisotropic optical properties have been investigated $[14–18]$ . Gaining control over the rotation of a levitated object is interesting from two perspectives. First, in high vacuum, optically levitated particles offer the potential to reach extremely high rotation speeds $[19–21]$ , necessary to investigate unexplored types of rotation-induced fluctuating forces and vacuum-friction effects $[22, 23]$ . Second, adding rotational control to the toolbox of levitated optomechanics is appealing when considering the regime of low excitation numbers currently investigated for the translational degrees of freedom. Each center-of-mass degree of freedom of a trapped particle embodies a quantum mechanical harmonic oscillator with equidistantly spaced energy levels and finite ground-state energy $[1]$ . In contrast, the rotational degrees of freedom offer a nonlinear energy spectrum with vanishing ground-state energy $[12]$ . Accessing this new regime of rich mesoscopic physics requires optical control of both the rotational and the center-of-mass motion in high vacuum. Importantly, while the center-of-mass motion of silica nanoparticles can be cooled close to the quantum regime $[10]$ , controlling the rotational motion of such a particle in high vacuum has remained elusive to date.

In this Letter, we measure the rotational motion of an optically trapped silica nanoparticle with a diameter of 100 nm. We drive the particle's rotation using circularly polarized light. At pressures below $10^{-5}$ mbar, we reach rotation frequencies exceeding 1 GHz. To our knowledge, these are the highest rotation frequencies of a mechanical object that have been realized to date.

Experimental setup. Our experimental setup is depicted in Fig. 1. A laser beam (wavelength $\lambda = 1565\mathrm{nm}$ , linearly polarized along the $x$ axis) propagates along the $z$ direction. Before entering a vacuum chamber, the polarization of the light can be set from linear over elliptical to circular by means of a quarter-wave plate. An aspheric lens (0.77 NA) inside the chamber focuses the beam to a diffraction-limited spot that forms an optical tweezer trap for a single silica nanoparticle with a nominal diameter of $100\mathrm{nm}$ . An identical lens collects and collimates the light for detection. A nonpolarizing beam splitter (BS) behind the vacuum chamber splits the beam such that half of the optical power is used for detecting the particle's center-of-mass motion, as de

![](images/aecb594ddc3339537d0e8a1746d9e2e1d41152416187bda71f7f04424f157148.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["1565 nm"] --> B["particle"]
    B --> C["vacuum chamber"]
    C --> D["BS"]
    D --> E["to c.m. detection"]
    E --> F["PBS"]
    F --> G["Output"]
```
</details>

FIG. 1. Simplified experimental setup. The polarization state of an initially linearly polarized laser beam can be set by a quarter-wave plate before entering a vacuum chamber. Inside the chamber, an optical trap for a nanoparticle is formed by focusing the beam with an aspheric lens (0.77 NA). The light is collected by an identical lens and equally split by a beam splitter (BS). One half of the power is utilized for center-of-mass (c.m.) detection of the particle. The other half is sent onto a half-wave plate followed by a polarizing beam splitter (PBS), which enables balanced detection of the particle rotation via a spectrum analyzer.

scribed in Ref. [6], and the other half for measuring its rotation. The center-of-mass motion of the particle shows three distinct oscillation frequencies, corresponding to the motion of the particle along the $x$ , $y$ , and $z$ direction. The transverse center-of-mass oscillation frequencies $\Omega_{\mathrm{c.m.}}^{(x)}$ and $\Omega_{\mathrm{c.m.}}^{(y)}$ along the $x$ and $y$ direction are around $2\pi \times 100\mathrm{kHz}$ and sensitively depend on the polarization of the trapping light. This dependence can be utilized to cross-check the polarization of the trapping light, as $\Omega_{\mathrm{c.m.}}^{(x)}$ and $\Omega_{\mathrm{c.m.}}^{(y)}$ are maximally distinct for linear polarization and become degenerate for circular polarization. In order to measure the particle's rotation frequency $\Omega_{\mathrm{rot}}$ , the second half of the detection beam passes a half-wave plate before it is split at a polarizing beam splitter (PBS) and sent onto a fast balanced photodetector (bandwidth $1.6\mathrm{GHz}$ ), which is connected to a spectrum analyzer (bandwidth $20\mathrm{GHz}$ ) [15, 16]. Independent of the rotation detection mechanism—which can arise from residual birefringence [15, 24], from asymmetric particle shape [18], or the angular Doppler shift [25]—we expect the signal at the spectrum analyzer to oscillate at $2\Omega_{\mathrm{rot}}$ .

Results and discussion. For our first set of measurements, we adjust the polarization of the trapping laser close to circular. In Fig. 2(a), we show a typical rotation spectrum, recorded at a pressure of $10^{-5}$ mbar and a trapping laser power $P = 226(5)$ mW. The signal at 1.31 GHz corresponds to a particle rotation frequency $\Omega_{\mathrm{rot}}/(2\pi) = 655$ MHz. In Fig. 2(b), we plot the observed rotation frequency $\Omega_{rot}$ as a function of pressure in the vacuum chamber. As the pressure is decreased from $10^{-1}$ mbar to below $10^{-5}$ mbar, $\Omega_{\mathrm{rot}}/(2\pi)$ increases from a few hundred kHz to above 1 GHz, where our measurement is currently limited by the finite bandwidth of our photodetector.

To understand the pressure dependence of $\Omega_{\mathrm{rot}}$ observed in Fig. 2(b), we consider the equation of motion of the particle's angular momentum $L$ , whose time rate of change equals the sum of all applied torques

$$
\frac {d}{d t} L = I \frac {d}{d t} \Omega_ {\mathrm{rot}} = \tau_ {\mathrm{opt}} + \tau_ {\mathrm{drag}}. \tag {1}
$$

We approximate the particle as a sphere with a moment of inertia $I = 0.4mR^2$ , mass $m \approx 1\mathrm{fg}$ , and radius $R = 50\mathrm{nm}$ . The optical torque $\tau_{\mathrm{opt}}$ arises from the particle's interaction with the laser field. The drag $\tau_{\mathrm{drag}}$ is due to the interaction of the particle with the residual gas in the vacuum chamber. We consider three possible contributions to the optical torque $\tau_{\mathrm{opt}} = \tau_{\mathrm{abs}} + \tau_{\mathrm{brf}} + \tau_{\mathrm{shape}}$ . The first component of the optical torque $\tau_{\mathrm{abs}} = \sigma_{\mathrm{abs}}\Delta s_{\mathrm{abs}}\mathcal{I}\lambda / (2\pi c)$ originates from absorbed photons that transfer their spin angular momentum to the particle [14]. Here, $\sigma_{\mathrm{abs}}$ is the particle's absorption cross section, $\mathcal{I}$ is the intensity at the particle's position, and $c$ is the speed of light. The degree of circular polarization $\Delta s_{\mathrm{abs}} \in [-1,1]$ becomes -1 for left-circularly polarized (LCP) trapping light and 1 for a right-circularly polarized (RCP) field. The second optical torque component $\tau_{\mathrm{brf}} \propto \mathcal{I}$ arises from photons changing their polarization state when scattering off the particle. This torque exists only for a particle exhibiting a finite birefringence and depends on the difference between the ordinary and the extraordinary refractive indices of the particle material [13]. The third optical torque component $\tau_{\mathrm{shape}} \propto \mathcal{I}$ describes the force arising due to a possible shape asymmetry of the particle [18]. On the other hand, the torque $\tau_{\mathrm{drag}}$ in Eq. (1) damps the particle's rotation due to viscous interaction with gas molecules in the vacuum chamber. Our experiment operates in a regime where the mean free path of gas molecules is much bigger than the particle diameter. In this regime, the viscous torque is given by $\tau_{\mathrm{drag}} = -I\Omega_{\mathrm{rot}} / t_{\mathrm{damp}}$ [26]. The damping time $t_{\mathrm{damp}} = \beta m\overline{v} / (p_{\mathrm{gas}}R^2)$ depends on the pressure $p_{\mathrm{gas}}$ , on the mean molecular velocity $\overline{v}$ , and on an accommodation factor $\beta$ that takes into account the efficiency with which molecules transfer angular momentum via collisions to the particle.

![](images/1fb75a0c32f719b5ae9b2321825a8bd299b52d0680dfc73332097b642af02020.jpg)

<details>
<summary>line</summary>

| frequency Ω/(2π) (GHz) | PSD (μV²/Hz) |
| --------------------- | ------------ |
| 1.31                  | 4.5          |
</details>

![](images/53170eb622e72c260c960c517767351fb695d8a23e3a75402b1a394b93272b92.jpg)

<details>
<summary>line</summary>

| pressure p_gas (mbar) | rotation frequency Ω_rot/(2π) (Hz) |
| --------------------- | ---------------------------------- |
| 1e-5                  | ~1e9                               |
| 1e-4                  | ~1e8                               |
| 1e-3                  | ~1e7                               |
| 1e-2                  | ~1e6                               |
| 1e-1                  | ~1e5                               |
</details>

FIG. 2. Rotation frequency as a function of pressure at a focal trapping power of $226(5)$ mW and for nearly left-circularly polarized (LCP) trapping light. (a) Measured power spectral density (PSD) of the rotation signal at a pressure of $1.1 \times 10^{-5}$ mbar showing a signal at 1.31 GHz, which corresponds to a rotation frequency of $\Omega_{\mathrm{rot}}/(2\pi) = 655$ MHz. (b) Measured rotation frequency for varying gas pressure with a maximum rotation frequency of $\Omega_{\mathrm{rot}}/(2\pi) = 1.029(1)$ GHz at a pressure of $7.2 \times 10^{-6}$ mbar. We attribute the deviation between theory and experiment below $10^{-3}$ mbar to an underestimation of pressure by the used ion gauge.

In the regime of almost circularly polarized light, any torque contribution trying to align the particle relative to the polarization ellipse of the trapping field can be neglected. Therefore, the particle rotates continuously at

![](images/58711f6e4113c3e20465e239eda941b7de762c2518707e4609c8181ab263edf1.jpg)

<details>
<summary>line</summary>

| focal power P (mW) | rotation frequency Ωrot/(2π) (MHz) |
| ------------------ | --------------------------------- |
| 0                  | 0                                 |
| 50                 | 15                                |
| 100                | 30                                |
| 150                | 45                                |
| 160                | 50                                |
</details>

FIG. 3. (a) Power dependence of the rotation frequency at a pressure $p_{\mathrm{gas}} = 1.0 \times 10^{-4} \, \mathrm{mbar}$ . The polarization of the trapping light equals the one in Fig. 2 and corresponds to a quarter-wave plate angle of $-29^{\circ}$ , see arrow in (b). (b) Particle rotation frequency as a function of quarter-wave plate angle at a focal power of $226(5) \, \mathrm{mW}$ . The wave plate angle determines the degree of linear vs circular polarization of the trapping light. A finite degree of circular polarization is necessary to induce rotation of the particle. The orange dashed line is a fit to $\Omega_{\mathrm{rot}} = \mathrm{Re}[a\sqrt{(1 - \cos b)^2}\sin^2 (2\phi -\phi_0) - \sin^2 (b)\cos^2 (2\phi -\phi_0)]$ [13]. The data in (a) and (b) have been recorded with different particles.

a frequency $\Omega_{rot} \propto \tau_{opt}/p_{gas}$ , which can be determined by solving for the steady state of Eq. (1). The inverse scaling of the rotation frequency with pressure is displayed as the dashed line in Fig. 2(b). We interpret the deviation of the measured data from the model as an artifact known as gauge pumping [27, 28]. This effect arises for ion gauges (as used in our experiment at pressures below $10^{-3}$ mbar) and leads to an underestimation of the pressure at the particle position.

Having investigated the pressure dependence of the rotation frequency, we turn to its dependence on the power of the trapping laser. In the limit of nearly circularly polarized trapping light, the optical torque scales linearly with optical power $\tau_{opt} \propto P$ , such that we find for the particle's rotation frequency $\Omega_{rot} \propto P$ . Keeping the polarization of the trapping light nearly circularly polarized at constant pressure $p_{gas} = 1 \times 10^{-4}$ mbar, we measure the rotation frequency $\Omega_{rot}$ as a function of focal power P. The result is displayed in Fig. 3(a) and shows very good agreement with the theoretically expected linear scaling.

Thus far, our experiments have been carried out with nearly circularly polarized trapping light. In Fig. 3(b), we investigate the dependence of the particle's rotation frequency on the polarization state of the light field by varying the angle $\phi$ between the fast axis of the quarter-wave plate in front of the vacuum chamber and the polarization vector of the initially linearly polarized light (see setup in Fig. 1). We observe a vanishing rotation fre quency for angles $\phi$ between $-14^{\circ}$ and $0^{\circ}$ and an increasing rotation frequency with increasing (absolute) value of $\phi$ outside that range. This experimental result can be explained by remembering that both $\tau_{\mathrm{brf}}$ and $\tau_{\mathrm{shape}}$ have two contributions [13, 18]. The first contribution leads to a restoring torque which tends to align the particle's symmetry axes (given by its shape or birefringence) to the main axes of the polarization ellipse. This contribution vanishes for perfectly circularly polarized light and is maximized for linearly polarized light. The second contribution drives the particle rotation. It vanishes for a linearly polarized field and increases with the degree of circular polarization. We conclude that for angles between $-14^{\circ}$ and $0^{\circ}$ the trapping light is predominantly linearly polarized, which leads to a dominant restoring torque pinning the particle's orientation to the polarization ellipse. As the quarter-wave plate is rotated beyond that range, the torque contribution leading to rotation overcomes that pinning the particle's orientation and the particle starts to rotate. Finally, we turn to the observation that the data in Fig. 3(b) are not symmetric around $\phi = 0$ , as expected. We attribute the horizontal shift of the data by roughly $-6^{\circ}$ , as well as the slight deviation from the fit, to the birefringence of the vacuum window and the aspheric trapping lens. We also note that for extremely pure LCP or RCP light, the particle quickly escapes from the trap. We speculate that this instability might be due to a spin-orbit coupling of the particle's rotation and its center-of-mass motion, similar to the effects observed in Ref. [15].

In a final experiment, we study the timescale of equilibration of the rotational dynamics of the levitated particle. To this end, we apply a steplike increase of the optical torque applied to the particle. Experimentally, we rapidly change the angle $\phi$ of the quarter-wave plate from $-19^{\circ}$ to $-29^{\circ}$ and record $\Omega_{rot}$ as a function of time t at fixed pressure. In Fig. 4(a), we show the time dependence of the observed rotation frequency after switching the polarization state of the trapping light. We observe a nonlinear acceleration of the rotation frequency. We explain our observation by considering the time dependent solution of Eq. (1), yielding $\Omega_{\mathrm{rot}} = c_{1} + c_{2} \exp(-t/t_{\mathrm{damp}})$ , with constants $c_{1}$ and $c_{2}$ that depend on the initial parameters. Indeed, the data in Fig. 4(a) fit well to the expected exponential behavior (dashed line). We repeat this experiment at different pressures, extract $t_{damp}$ from the fits, and plot the damping times in Fig. 4(b). As expected, the damping time scales inversely with pressure according to $t_{damp} \propto 1/p_{gas}$ .

We note that the data displayed in the figures of this manuscript have been recorded with different particles. The reason is that particles occasionally escape from the trap or possibly disintegrate due to high rotation frequencies and the associated high centrifugal forces acting on the particle $[29]$ . The properties (including exact diameter, shape, birefringence, absorption, surface roughness)

![](images/cadc8e85e91de36d03c4f0f86631042ee13c60e9384838d4a5819414c5d8f221.jpg)

<details>
<summary>line</summary>

| time t (s) | rotation frequency Ωrot/(2π) (MHz) |
| ---------- | --------------------------------- |
| 0          | 28.0                              |
| 5          | 34.0                              |
| 10         | 38.0                              |
| 15         | 41.0                              |
| 20         | 43.0                              |
| 25         | 44.0                              |
</details>

![](images/6d66841f664a14ceeba32fcd5cffe26bc6e5f0842ff711a0a554a5abd89ecc4c.jpg)

<details>
<summary>line</summary>

| pressure p_gas (mbar) | damping time t_damp (s) |
| --------------------- | ------------------------ |
| 1e-05                 | 30                       |
| 2e-05                 | 28                       |
| 3e-05                 | 26                       |
| 4e-05                 | 24                       |
| 5e-05                 | 22                       |
| 6e-05                 | 20                       |
| 7e-05                 | 18                       |
| 8e-05                 | 16                       |
| 9e-05                 | 14                       |
| 1e-04                 | 12                       |
| 2e-04                 | 10                       |
| 3e-04                 | 8                        |
| 4e-04                 | 6                        |
| 5e-04                 | 4                        |
| 6e-04                 | 2                        |
| 7e-04                 | 1                        |
| 8e-04                 | 0.5                      |
| 9e-04                 | 0.3                      |
| 1e-03                 | 0.2                      |
</details>

FIG. 4. Measurement of system dynamics at a focal power $P = 230(5) \mathrm{mW}$ . (a) After a steplike change of the optical torque by a fast rotation of the quarter-wave plate [see Fig. 1 and Fig. 3(b)] we measure the rotation frequency as a function of time at a fixed pressure. A fit (dashed line) of $\Omega_{\mathrm{rot}} = c_1 + c_2 \exp(-t / t_{\mathrm{damp}})$ , with $t_{\mathrm{damp}}$ , $c_1$ , and $c_2$ as free parameters, reveals the characteristic response time $t_{\mathrm{damp}}$ of our system. (b) Repeating the procedure described in (a) for different pressures results in a damping time which scales as $t_{\mathrm{damp}} \propto 1 / p_{\mathrm{gas}}$ (dashed line).

of each particle may therefore vary between figures. Nevertheless, all rotation states reported in this paper have been observed to be stable on a minimal timescale of minutes.

Conclusion. We have demonstrated the stable rotation of an optically trapped dielectric particle of 100 nm diameter at rotation frequencies exceeding 1 GHz. With a simple model, we were able to describe our experimental observations. However, while our model yields the correct scaling of the parameters involved (e.g. laser power and pressure) it does not identify the dominating torque transfer mechanism. Ongoing work is aimed at elucidating this mechanism.

Our results have important implications for quantum optomechanics, cosmology and material tests at the nanoscale: Together with the fact that the center-of-mass motion of optically levitated nanoparticles can be cooled to the sub-mK regime $[10]$ , our control over the rotational degree of freedom could be utilized for studies of friction at a fundamental level $[22, 23]$ . Another exciting prospect is to explore the interaction of center-of-mass and rotational degrees of freedom, which may allow studies of spin-orbit coupling in optically levitated systems. Such a coupling, which has been reported in Ref. $[15]$ , is not yet observed in our experiment, for reasons to be investigated. Interestingly, the ability to rotate particles at GHz frequencies might provide a platform to study questions arising in the context of cosmology. Rapidly spinning charged dust particles in the interstellar medium have been proposed to be responsible for GHz radiation in measurements of the cosmic background radiation $[30]$ . Together with controlling the particle charge [31], our method of rotating a nanoparticle at GHz frequencies could enable a test bed for this hypothesis. In the direction of more applied research [29], rapidly rotating nanoparticles with circumferential speeds exceeding 300 m/s and radial accelerations on the surface of more than $10^{12}$ m/s $^{2}$ (corresponding to the gravitational acceleration on the surface of a neutron star [32]) can be utilized to test material limits under centrifugal stress on the nanoscale. For our glass particles with density $\rho = 2000 \, kg/m^{3}$ , the realized maximal tensile strength $\sigma_{tens} \approx \rho \Omega_{rot}^{2} R^{2}$ [29] is about 0.2 GPa. Accordingly, our experiments operate in the interesting regime close to the ultimate tensile strength on the order of 10 GPa [33] at which defect-free glass would disintegrate.

Acknowledgments. We thank P. Kurpiers and A. Wallraff for lending us a high bandwidth spectrum analyzer. This work has been supported by ERC-QMES (No. 338763). R.R acknowledges funding from the European Union's Horizon 2020 research and innovation program under the Marie Skłodowska-Curie Grant Agreement No. 702172.

R.R. and M.D. contributed equally to this work.

Note added. We have recently become aware of related work on optically rotating micron-sized spheres in high vacuum [34] and on GHz rotation of nanodumbbells [35].

\* rreimann@ethz.ch

[1] M. Aspelmeyer, T. J. Kippenberg, and F. Marquardt, Rev. Mod. Phys. 86, 1391–1452 (2014).   
[2] A. Ashkin, Optical Trapping and Manipulation of Neutral Particles Using Lasers (World Scientific Publishing, Singapore, 2007).   
[3] D. E. Chang, C. A. Regal, S. B. Papp, D. J. Wilson, J. Ye, O. Painter, H. J. Kimble, and P. Zoller, Proc. Natl. Acad. Sci. USA 107, 1005–1010 (2010).   
[4] O. Romero-Isart, A. C. Pflanzer, M. L. Juan, R. Quidant, N. Kiesel, M. Aspelmeyer, and J. I. Cirac, Phys. Rev. A 83, 013803 (2011).   
[5] T. Li, S. Kheifets, and M. G. Raizen, Nat. Phys. 7, 527-530 (2011).   
[6] J. Gieseler, B. Deutsch, R. Quidant, and L. Novotny, Phys. Rev. Lett. 109, 103603 (2012).   
[7] Z. Yin, A. A. Geraci, and T. Li, Int. J. Mod. Phys. B 27, 1330018 (2013).   
[8] N. Kiesel, F. Blaser, U. Delić, D. Grass, R. Kaltenbaek, and M. Aspelmeyer, Proc. Natl. Acad. Sci. USA 110, 14180–14185 (2013).   
[9] J. Millen, P. Z. G. Fonseca, T. Mavrogordatos, T. S. Monteiro, and P. F. Barker, Phys. Rev. Lett. 114, 123602 (2015).   
[10] V. Jain, J. Gieseler, C. Moritz, C. Dellago, R. Quidant, and L. Novotny, Phys. Rev. Lett. 116, 243601 (2016).   
[11] J. Vovrosh, M. Rashid, D. Hempston, J. Bateman, M. Paternostro, and H. Ulbricht, J. Opt. Soc. Am. B 34, 1421-1428 (2017).

[12] H. Shi and M. Bhattacharya, J. Phys. B 49, 153001 (2016).   
[13] M. E. J. Friese, T. A. Nieminen, N. R. Heckenberg, and H. Rubinsztein-Dunlop, Nature (London) 394, 348–350 (1998).   
[14] Y. Arita, A. W. McKinley, M. Mazilu, H. Rubinsztein-Dunlop, and K. Dholakia, Anal. Chem. 83, 8855–8858 (2011).   
[15] Y. Arita, M. Mazilu, and K. Dholakia, Nat. Commun. 4, 2374 (2013).   
[16] T. M. Hoang, Y. Ma, J. Ahn, J. Bang, F. Robicheaux, Z.-Q. Yin, and T. Li, Phys. Rev. Lett. 117, 123604 (2016).   
[17] S. Kuhn, B. A. Stickler, A. Kosloff, F. Patolsky, K. Hornberger, M. Arndt, and J. Millen, Nat. Commun. 8, 1670–(2017).   
[18] S. Kuhn, A. Kosloff, B. A. Stickler, F. Patolsky, K. Hornberger, M. Arndt, and J. Millen, Optica 4, 356 (2017).   
[19] B. E. Kane, Phys. Rev. B 82, 115441 (2010).   
[20] P. Nagornykh, J. E. Coppock, J. P. J. Murphy, and B. E. Kane, Phys. Rev. B 96, 035402 (2017).   
[21] S. Kuhn, P. Asenbaum, A. Kosloff, M. Sclafani, B. a. Stickler, S. Nimmrichter, K. Hornberger, O. Cheshnovsky, F. Patolsky, and M. Arndt, Nano Lett. 15, 5604–5608 (2015).   
[22] R. Zhao, A. Manjavacas, F. J. García de Abajo, and J. B. Pendry, Phys. Rev. Lett. 109, 123604 (2012).   
[23] A. Manjavacas, F. J. Rodríguez-Fortuño, F. J. García de Abajo, and A. V. Zayats, Phys. Rev. Lett. 118, 133605

(2017).   
[24] B. A. Garetz and S. Arnold, Opt. Commun. 31, 1–3 (1979).   
[25] B. A. Garetz, J. Opt. Soc. Am. 71, 609 (1981).   
[26] J. Fremerey, Vacuum 32, 685–690 (1982).   
[27] Stanford Research Systems, Bayard-Alpert Ionization Gauges, Application Note.   
[28] H. F. Winters, D. R. Denison, and D. G. Bills, Rev. Sci. Instr. 33, 520-523 (1962).   
[29] M. Schuck, D. Steinert, T. Nussbaumer, and J. W. Kolar, Sci. Adv. 4, e1701519 (2018).   
[30] B. T. Draine and A. Lazarian, Astrophys. J. 508, 157-179 (1998).   
[31] M. Frimmer, K. Luszcz, S. Ferreiro, V. Jain, E. Hebestreit, and L. Novotny, Phys. Rev. A 95, 061801 (2017).   
[32] S. F. Green and M. H. Jones, An Introduction to the Sun and Stars (Cambridge University Press, Cambridge, England, p. 322, 2004).   
[33] E. Le Bourhis, Glass: Mechanics and Technology, New York (Wiley-VCH, chapter 7, 2007).   
[34] F. Monteiro, S. Ghosh, E. C. van Assendelft, and D. C. Moore, Physical Review A 97, 051802 (2018).   
[35] J. Ahn, Z. Xu, J. Bang, Y.-H. Deng, T. M. Hoang, Q. Han, R.-M. Ma, and T. Li, Physical Review Letters 121, 033603 (2018).