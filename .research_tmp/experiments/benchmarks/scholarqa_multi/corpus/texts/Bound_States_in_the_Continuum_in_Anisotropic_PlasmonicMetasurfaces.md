# Supporting Information

# Bound states in the continuum in anisotropic plasmonic metasurfaces

Yao Liang $^{1}$ , Kirill Koshelev $^{2,3}$ , Fengchun Zhang $^{4}$ , Han Lin $^{1}$ , Shirong Lin $^{1}$ , Jiayang Wu $^{5}$ , Baohua Jia $^{1*}$ , and Yuri Kivshar $^{2,3*}$

These authors contributed equally

*Corresponding authors: Baohua Jia bjia@swin.edu.au and Yuri Kivshar ysk@internode.on.net

1/11

# 1. FDTD Simulations

![](dt=2026-06-03/ht=11/136c77b187ab1738cf6861f12548d83192744acababe372123e377067911afb6.jpg)

![](dt=2026-06-03/ht=11/2a518f558ae8c7bada8ed0616bdaf5545e6d4a642f5a7a27c2e89f875bfa9b5a.jpg)

The reflection characteristics of the proposed structures are numerically simulated by using the finite-different time-domain (FDTD) method (FDTD Solutions package from Lumerical Inc.). We apply the perfect matching layer (PML) conditions along the z-direction to avoid boundary reflections. For normal incidence plane wave, the periodic boundary conditions in the x- and y-directions are applied to replicate an infinite array (Figure S1a).

For oblique incidence, we use Broadband Fixed Angle Source Technique (BFAST) in the plane wave source to simulate periodic structures illuminated with a broadband source at an angle, and the periodic boundary conditions are overridden by BFAST, as shown in Figure S1b. A uniform mesh size of $20\mathrm{nm}$ ( $\mathbf{x}, \mathbf{y}$ , and $\mathbf{z}$ directions) was used. The permittivity $(\varepsilon)$ of the materials (Au and $\mathrm{SiO}_2$ ) we used are extracted from the data of Palik[1].

2/11

# 2. Near-field excitation of dark and bright modes

![](dt=2026-06-03/ht=11/c91f029c0b569b8223c317ed830678a1dcdac3bbfce7e46867ccf004c32ac30d.jpg)

![](dt=2026-06-03/ht=11/134c12795a05144f2f50557ca3adfebbacdbe3d8f58212e7fe431d803777bf2c.jpg)

![](dt=2026-06-03/ht=11/c9bd0bde93dcbe3d6978668cdf542dd0d75a78bcacd9af25ee6329aa298a0a9b.jpg)

![](dt=2026-06-03/ht=11/c0cf90d697820fb050094a9c240d15a98ac1c747686a2803f0ada642b28fb358.jpg)

![](dt=2026-06-03/ht=11/dee00c07f52e2815efe3e360c149e1846b0205f40ef3543e66b8cd1d8db471ea.jpg)

![](dt=2026-06-03/ht=11/de2ab738571c969f2418eb671ed802f1d55343925e97e316761bbf6211449e81.jpg)

![](dt=2026-06-03/ht=11/c118392b1159256c2c62b3d8cf4142394362b709ffb494f2efb0d6759916766a.jpg)

![](dt=2026-06-03/ht=11/99fc06feb2b6b8825e20f6b60d0ebf32bc5d73e20bb93c38a4078b9ba5fd17fb.jpg)

![](dt=2026-06-03/ht=11/1b587a0774af4dbb249fb112683a040d522f2b025dc0f904a3d96e8cb2febaa1.jpg)

![](dt=2026-06-03/ht=11/d8492e5f9472d6c33258e81417c5539048c4d03861d6ec8a5104851f90b505fd.jpg)

![](dt=2026-06-03/ht=11/42c9f372093601a8a12551b1dc4a8e39426d3246a8a26b83477b6a9b6db2e7b9.jpg)

![](dt=2026-06-03/ht=11/ae9f4159ccb73bb8f100779751128929b7a9a329dc0018f3130bc487f58c4a23.jpg)

(ii)

![](dt=2026-06-03/ht=11/d17f3f6235e7888d0f08e341121921f09d8534eee2465b566bffe87bffdec301.jpg)

In order to effectively excite the dark and bright modes supported in the plasmonic structure, we use two kinds of electric dipole (ED) near-field sources in FDTD simulations—an in-plane ED along y-direction and an out-of-plane ED oscillating in the z-direction (Figure S2a). In simulations, periodic boundary conditions are used in x- and y-directions while PML conditions are applied in the z-direction.

For in-plane ED excitation, only the bright mode is excited, and the near-field electric intensity $(\propto |\mathbf{E}|)$ decays much faster than that dark mode excited by an out-of-plane ED (Figure S2b). This explains the sub-radiative dark mode can store light energy in the near field and offer longer lifetimes than its bright counterpart[2]. This point is also manifested as the difference in the resonant spectra (Figure S2c), where the linewidth of dark modes is much narrower

3/11

than that of the bright mode. The electric (E) and magnetic (H) field distribution of the dark and bright modes are illustrated in Figure S2d. Clearly, the bright mode has odd charge distribution with in-plane dipolar moment while the dark mode has even charge distribution and out-of-plane dipolar moment. The schematics of charges distribution and oscillations (grey arrows) for these two modes are illustrated in Figure S2e.

# 3. Far-field excitation of the bright mode

![](dt=2026-06-03/ht=11/c11585b8c52db574b93ff268571ae2f1515a557d078e61bd67a6097a7e3e6760.jpg)

![](dt=2026-06-03/ht=11/87c37e0e0d65d03b1f08cf9db679b00a564275ac4dd20fd6fc7690787198676e.jpg)

![](dt=2026-06-03/ht=11/cc3119ae68cfbd619d34cad8c34e7ed0d2b416242e87dc62b4ec7af3cf634199.jpg)

![](dt=2026-06-03/ht=11/0f5d8ee8c0c039aee24dc393b3f2b939ff1efae8639361663586dd5daad63362.jpg)

The bright mode has odd charge distribution feature and non-vanishing far-field radiation. Thus, we are able to excite it by using a linear plane wave. We use TE- (along $\Gamma -X$ ) and TM-polarized (along $\Gamma -X^{\prime}$ ) plane wave as the excitation source and the reflection band-diagram is shown in Figure S3a. Figure S3b shows the reflection spectrum for TM polarization at the normal incidence, where only the bright mode is excited, and the right panel shows the field distributions at the resonance wavelength, which are similar to those excited by the in-plane ED (Figure S2d-i). In

4/11

contrast to dark modes, the Q factors of the bright mode remain constant (Q~19) for various oblique incidences angles in both $\Gamma - X$ and $\Gamma - X'$ directions (Figure S3c). Interestingly, the resonance wavelengths for oblique incidences in $\Gamma - X$ direction remain hardly unchanged, but a red-shift is observed in the $\Gamma - X'$ direction (Figure S3d).

5/11

![](dt=2026-06-03/ht=11/f8f61c56f646d87e321df1dccf4047568735bae3a1fb25acd99c2378ac61454a.jpg)

![](dt=2026-06-03/ht=11/b81ae114f763ed412ba7c90ca33e1ca67cef94f76524c0f5996584853d18f9e5.jpg)

The plasmonic BIC is robust against the change of the lattice spacing. As shown in Figure S4a, we change the period (P) of the unit cell from 2.8 to $3.2\mu \mathrm{m}$ (the other parameters are fixed). For the quasi-BIC state (TM, $\theta 1 = 2^{\circ}$ ), the Q-factor is robust against the period change (Figure S4b).

# 5. Fabrication

The vertical plasmonic split-ring resonator array $(20\times 25$ units) was first fabricated by femtosecond laser lithography (Libra, $800\mathr
m{nm}$ , 100 fs, $10\mathrm{kHz}$ ) with an oil immersion objective (Olympus, 1.4 N.A.).[5] A piezoelectric nanotranslation stage is used to trace out the microstructures in the photoresist (Figure 3a). We use a silicon-zirconium hybrid photoresist (negative tone) for laser lithography due to its excellent resistance to shrinkage[6].

Then, $100\mathrm{nm}$ thick gold film is coated on the sample by gold sputtering (Emitech K975X) at the deposition rate of $12.5\mathrm{nm / min}$ (Figure 3b). To make the coating uniform, we first do sputtering in the vertical direction, and then we put the sample on a $45^{\circ}$ degree holder and do it again for 4 times for different sample orientations.

4. Robustness of quasi-BICs against the change of lattice spacing

6/11

a

![](dt=2026-06-03/ht=11/df0c8dd905673944d49c7825b20eac970d117e42d494d6e81a8fe2a7d1ef23f0.jpg)

![](dt=2026-06-03/ht=11/34fd5f952e6c3a0af2178559d22d6af60496213b63e3f29386e97312153a18e1.jpg)

![](dt=2026-06-03/ht=11/3c58f176927266d14de5e9438e2b7572f97a418cbdc8461860b5a028cb305ccf.jpg)

![](dt=2026-06-03/ht=11/33831f9060970cc4e44b3aa2506a7342fa226f6f7858eabdab43d791fb5bcc84.jpg)

![](dt=2026-06-03/ht=11/b457c9cca2971ef5275ee569a28e9b330c04dc203fee7e3dffa1865017acedb4.jpg)

![](dt=2026-06-03/ht=11/4fc3778e9fefc39e40b4c521f860535c156b1f3d659793f97065ae45b50c8ff2.jpg)

![](dt=2026-06-03/ht=11/3bfaea2ee1326020fd9d9c9038b424b3cd4f6002178d71884e9d3bde9f3ecb1d.jpg)

In our theoretical design, the excitation of dark modes requires oblique light incidence in a certain incidence plane, being xz- or yz-plane. For a specific linear polarization, light coming from different incidence plane has a dramatically different spectral response (Figure 2a). In the experiment, however, the reflective objective we use for the excitation has both light components in xz-plane and yz-plane. To reduce the influence of the undesired light component, we used a piece of tape to partially block the objective in the x- or y-direction.

6. Optical characterization

7/11

In a typical focusing application of a reflective microscope objective, collimated light passes through the aperture hole in the primary mirror to the secondary mirror (Figure S5a). The secondary mirror reflects and diverges the beam to fill the primary mirror. The primary mirror focuses the beam to a small spot. This dual mirror configuration is known as a reverse Cassegrain[7].

To analyze the influence of the tape, we first introduce how the reflective objective works by using ray optics. As shown in Figure S5a, at an arbitrary point $(\mathbf{x},\mathbf{y})$ , the light beam can be expressed as $(\mathrm{I}(\mathbf{x},\mathbf{y}),\theta)$ , where $\mathrm{I}(\mathbf{x},\mathbf{y})$ and $\theta$ represent the intensity and the incident angle with respect to the z-axis respectively.

The light beam $(\mathrm{I}(\mathbf{x},\mathbf{y}),\theta)$ can be decomposed into two components of $(\mathrm{I}_{\mathrm{xz}}(\mathbf{x},\mathbf{y}),\theta 1)$ and $(\mathrm{I}_{\mathrm{yz}}(\mathbf{x},\mathbf{y}),\theta 2)$ in the xz-plane and yz-plane respectively. the $\mathrm{I_{xz}(x,y)}$ and $\mathrm{I_{yz}(x,y)}$ represent the intensity while $\theta 1$ and $\theta 2$ are the angles with respect to z-axis in each projection plane.

The intensity of those light components have quantitative realationship as $\mathrm{I} = \mathrm{I}_{\mathrm{xz}} + \mathrm{I}_{\mathrm{yz}}$ , and $\mathrm{I_{xz}(x,y)} = \mathrm{I}*\cos^2 (\alpha (\mathrm{x},\mathrm{y}))$ , $\mathrm{I_{yz}(x,y)} = \mathrm{I}*\sin^2 (\alpha (\mathrm{x},\mathrm{y}))$ , where $\alpha (\mathrm{x},\mathrm{y})$ is the angle down from x-axis at point (x,y). Correspondingly, the incidence angles for the two components in orthogonal incidence planes are given by

$$
\left\{ \begin{array}{l} \theta 1 (x, y) = \operatorname {a t a n} (\frac {\rho (x , y) * c o s (\alpha (x , y))}{W D}) \\ \theta 2 (x, y) = \operatorname {a t a n} (\frac {\rho (x , y) * \sin (\alpha (x , y))}{W D}) \end{array} \right. \tag {1}
$$

where $\rho(x, y) = \sqrt{x^2 + y^2}$ , and $WD$ is the working distance of the objective (Figure S5a). Our calculation assumes that each point on the output plane of the reflective objective has equal contribution to the focus. The light intensity and incidence angle distributions of an unblocked objective are shown in Figure S5b.

Since $\mathrm{I}_{\mathrm{xz}}$ and $\mathrm{I}_{\mathrm{yz}}$ components have dramatically different spatial distributions, it is able to block most of $\mathrm{I}_{\mathrm{yz}}$ component and increase the purity of light coming from the xz-plane ( $\mathrm{I}_{\mathrm{xz}}$ ) by using a piece of tape in the middle of the objective in y-direction (Figure S5c). To quantify the influence of various incidence angle and intensities in the two orthogonal planes, we calculate the percentage density of each incidence angle for both xz- and yz-plane components by using

$$
\left\{ \begin{array}{l} P _ {x z} (\theta 1) = \frac {\iint_ {S} \mathrm {I} _ {\mathrm {x z}} (\theta 1) d S}{\iint_ {S} \mathrm {I} d S} \\ P _ {y z} (\theta 2) = \frac {\iint_ {S} \mathrm {I} _ {\mathrm {y z}} (\theta 2) d S}{\iint_ {S} \mathrm {I} d S} \end{array} \right. \tag {2}
$$

where $S$ is the area where light passing through on the output plane of the reflective objective. The sum of each incidence angle in xz-plane and yz-plane result in $\int P_{xz}(\theta 1)\mathrm{d}\theta 1 + \int P_{yz}(\theta 2)\mathrm{d}\theta 2 = 1$ . The calculation results are shown in Figure S5d and Figure S5e. For the unblocked objective, Ixz and Iyz components account for $50\%$ respectively (Figure S5d). However, after applying a piece of tape in the y-direction, the percentage of Ixz increases to $\sim 72\%$ while the one of Iyz drops to $\sim 28\%$ (Figure S5e). Therefore, the tape helps to increase the purity of light in a certain incidence plane.

The percentage density calculation gives the weight of various angles in different incidence planes. By combining the band diagram (angle-dependent spectra) in Figures 2a,b in the main text, we estimate the overall spectrum $(\overline{s})$ of the proposed structure by a weighted average of simulation spectra,

8/11

$$
\bar {s} = \int_ {0 ^ {\circ}} ^ {3 0 ^ {\circ}} P _ {x z} (\theta 1) * s _ {x z} (\theta 1) d \theta 1 + \int_ {0 ^ {\circ}} ^ {3 0 ^ {\circ}} P _ {y z} (\theta 2) * s _ {y z} (\theta 2) d \theta 2 \qquad (3)
$$

here $s_{xz}(\theta 1)$ represents the simulation reflection spectrum of the structure in the xz-plane at the incidence angle of $\theta 1$ while $s_{yz}(\theta 2)$ is the one in the yz-plane at the incidence angle of $\theta 2$ (spectra data from Figure 2a). The simulation weighted average and experimental results for TE and TM polarizations and for different objective configurations are depicted in Figure S5f and Figure S5g.

9/11

![](dt=2026-06-03/ht=11/6c9ab5af54d9ec7128536e6af5d51b917011677af43275df37a224e0586d5a06.jpg)

![](dt=2026-06-03/ht=11/968f13b010232d4a759a506e4a1b0ea33f9797b30583ab4b2d723b2313c49244.jpg)

![](dt=2026-06-03/ht=11/3a06e00042bd7239e2fb35c94ba9881794a64d1542e0ab1d479ba2066c00005b.jpg)

![](image)
oduce.db/mineru_full_text/v0/result=success/type=image/dt=2026-06-03/ht=11//cdc1a1490105abd08baaf3ace303267d0bad774268d2ca8f382a2ccf5b83c694.jpg)

![](dt=2026-06-03/ht=11/6e11753132822871d7777fefdad8984e57ffe7963358dc050525ec247f666314.jpg)

The simulation results are, in general, in agreement with the experimental results. However, the reflection in the experiment is usually weaker than the simulation prediction, especially for the dark modes. One important reason is the difference in light collection. In the simulation, all the backward scattering light is collected by the monitor (Figure S6a). However, in the experiment, the reflective objective only collects backward scattering light at some specific angles.

In particular, at large oblique angle incidence, most backward scattering energy cannot be collected by the objectives for wavelengths at (near) the resonance due to large-angle scattering. To illustrate this point, in Figure S6c, we plot the far-field distribution of the resonance wavelength of TM-polarized light at $\theta 1 = 29^{\circ}$ in the xz-plane (Figure S6b). The far-field energy is concentrated in the vicinity of the point $(\psi ,\varphi) = (180^{\circ},25^{\circ})$ .

For the unblocked reflective objective, it only collects backward scattering light at angles where $12.5^{\circ} < \psi < 29^{\circ}$ . Therefore, the uncollected light is responsible for the deviation between the simulation weight results and the experimental results.

10/11

# References

11/11