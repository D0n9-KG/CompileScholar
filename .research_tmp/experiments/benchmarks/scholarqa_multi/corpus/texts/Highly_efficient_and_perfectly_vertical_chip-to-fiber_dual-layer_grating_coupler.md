# High-efficiency dual-layer grating coupler for vertical fiber-chip coupling in two polarizations

KE LI, $^{1}$ JINGPING ZHU, $^{1,*}$ QIHANG DUAN, $^{2,3}$ AND XUN HOU $^{1,2}$

*jpzhu@xjtu.edu.cn

Received 15 February 2023; revised 25 March 2023; accepted 31 March 2023; posted 31 March 2023; published 2 May 2023

Efficient coupling between optical fibers and high-index-contrast silicon waveguides is essential for the development of integrated nanophotonics. Herein, a high-efficiency dual-layer grating coupler is demonstrated for vertical polarization-diversity fiber-chip coupling. The two waveguide layers are orthogonally distributed and designed for $y$ - and $x$ -polarized $\mathrm{LP}_{01}$ fiber modes, respectively. Each layer consists of two 1D stacked gratings, allowing for both perfectly vertical coupling and high coupling directionality.

The gratings are optimized using the particle swarm algorithm with a preset varying trend of parameters to thin out the optimization variables. The interlayer thickness is determined to ensure efficient coupling of both polarizations. The optimized results exhibit record highs of $92\%$ (-0.38 dB) and $85\%$ (-0.72 dB) 3D finite-difference time-domain simulation efficiencies for $y$ and $x$ polarizations, respectively. The polarization-dependent loss (PDL) is below 2 dB in a 160 nm spectral bandwidth with cross talk between the two polarizations less than -24 dB.

Fabrication imperfections are also investigated. Dimensional offsets of $\pm 10$ nm in etching width and $\pm 8$ nm in lateral shift are tolerated for a 1 dB loss penalty. The proposed structure offers an ultimate solution for polarization diversity coupling schemes in silicon photonics with high directionality, low PDL, and a possibility to vertically couple. © 2023 Optica Publishing Group

https://doi.org/10.1364/JOSAA.487739

# 1. INTRODUCTION

As the link capacity has increased dramatically, advanced silicon photonic integrated circuits (PICs) have attracted extensive attention in the past few decades [1,2]. Silicon on insulator has emerged as a promising platform for future dense integrated photonic devices, owing to silicon's low cost, complementary metal-oxide semiconductor (CMOS) compatibility and high index contrast [3,4]. The typical thickness of a silicon slab waveguide is a submicrometer, which is much smaller than the size of a conventional single-mode fiber (SMF) (core diameter $\sim 9\mu \mathrm{m}$ ). High-performance fiber-to-chip coupling is an important problem in silicon photonic systems.

In general, edge couplers [5,6] and grating couplers (GCs) [7,8] are two primary interfaces used between optical fibers and PICs. GCs afford various distinctive advantages, including out-of-plane coupling, wafer-scale testing, avoidance of facet polishing, and alleviated alignment tolerance. These are attractive for mass production and packaging of commercial silicon photonic circuits. Considering 1D GCs, they can be chirped to couple with near unity efficiency but are normally demonstrated for only one polarization state [9-12]. However, in practical applications, such as sensing and coherent communications, an efficient coupling of both polarizations is needed. Although

some specially designed 1D GCs can couple two polarizations, they yield a significant shift in the spectral response between two polarizations because of the birefringence effect [13-15].

To achieve polarization diversity, 2D GCs are a promising solution to integrate a polarization splitter and a coupler. However, most 2D GCs are designed to couple the fiber at an angle (tilted fiber [16-23] or tilted grating [24,25]) to suppress backreflections. The tilting coupling scheme not only adds extra mounting complexity, but also leads to polarization-dependent loss (PDL).

Although various configurations have been adopted to alleviate the PDL, including five-cylinder grating cells [19], slanted arrays [20], elliptical etching patterns [21], and diamondlike grating lattices [22], these 2D GCs represent a relative low efficiency level ( $\sim -3$ dB). Furthermore, some attempts have been made to achieve polarization diversity and vertical coupling simultaneously. Zou et al. introduced four $45^{\circ}$ reflectors and two multimode interference power combiners in a four-port symmetric configuration [26]. Watanabe et al.

demonstrated a 2D GC consisting of subwavelength reflectors and blazed structures for vertical fiber coupling in two polarizations [27]. However, there was a significant degradation in the coupling efficiency (CE) or PDL.

Check for updates

1022 Vol. 40, No. 6 / June 2023 / Journal of the Optical Society of America A

Research Article

Journal of the Optical Society of America

OPTICS, IMAGE SCIENCE, AND VISION

1084-7529/23/061022-07 Journal © 2023 Optica Publishing Group

Multilayer configurations are considered as a viable alternative to balance the CE and PDL especially for 3D photonic integration. Mak et al. proposed a polarization-independent GC based on the supermodes in a three-layer SiN-on-Si photonic platform [23]. The peak CE was simulated as $-2.1\mathrm{dB}$ with a 1 dB PDL bandwidth of $69~\mathrm{nm}$ . Yu and Yamada designed a dual-layer GC with a top long-period grating functioning as a beam splitter [28]. Zhang et al. optimized the chirped grating period, fiber position, and cladding thickness of a dual-layer vertical GC [29]. However, the optimized peak efficiencies were all below $60\%$ .

In this paper, we present a dual-layer polarization-diversity GC for perfectly vertical fiber-to-chip coupling with a very high CE and low PDL. The incident light is split into two orthogonal polarization states, i.e., $y$ - and $x$ -polarized $\mathrm{LP}_{01}$ fiber modes, and then coupled to two layers, respectively. By properly designing the gap between the two layers, most of the $x$ -polarized power can be transferred to the bottom layer.

The CE of $y$ - or $x$ -polarized light is defined as the portion coupled to the first or second waveguide layer when the corresponding polarization state is incident from the vertical fiber. Each layer consists of two 1D stacked gratings, allowing for both perfectly vertical coupling and high coupling directionality. The stacked gratings are optimized using the particle swarm algorithm.

To thin out the optimization variables, the varying trends of the parameters are preset on the basis of theoretic analysis, achieving in 3D finite-difference time domain (FDTD) simulation $92\%$ (-0.38 dB) and $85\%$ (-0.72 dB) efficiencies for $y$ - and $x$ -polarizations, respectively. The PDL is below 2 dB over the spectral bandwidth of $160~\mathrm{nm}$ . To the best of our knowledge, this is the highest CE among polarization-diversity GCs to date. The fabrication tolerances of etching width, lateral shift of the stacked grating, and the fiber misalignment are also investigated.

Such an efficient polarization-diversity GC is promising for polarization splitting, fiber arrays coupling, as well as high 3D photonic integration.

# 2. DEVICE CONFIGURATION AND PRINCIPLE

Figure 1(a) shows a schematic of the proposed dual-layer polarization-diversity GC. The two Si waveguide layers are separated by a $\mathrm{SiO}_2$ gap. The $y-$ and $x$ -polarized $\mathrm{LP}_{01}$ fiber modes are launched from a vertically aligned SMF (core diameter $9\mu \mathrm{m}$ and index contrast $0.36\%$ ) and then split into the first top and second bottom waveguide layers, respectively. In this process, both polarizations are converted to TE mode of the silicon waveguide. As shown in the front view of Fig.

1(b), the top waveguide layer (the first grating) for $y$ polarization, consists of two stacked gratings on top of one another, i.e., upper grating and lower grating. The two stacked gratings with a lateral shift $s$ function as a blazed structure, providing an antireflection effect to break the scattering symmetry. The pitch period is donated by
$\Lambda$ , and duty factor $D$ describes the etched fraction of a period. The bottom waveguide layer (the second grating) is rotated $90^{\circ}$ around the fiber centerline relative to the top layer as depicted in the side view Fig. 1(c).

The $x$ -polarized light passes through the top waveguide layer and the gap layer and is thereafter coupled to the bottom layer. The outgoing waves of two polarizations can be channeled into one layer by inserting a vertical interlayer coupling structure between the two layers. The device is

![](dt=2026-03-11/ht=12/35dd2bddaf2aea4e2659c15261b4add10ec29d63217973d8ff7330230d19f2a1.jpg)

established on a Si substrate, and $\mathrm{SiO}_2$ is used as the cladding and buried oxide (BOX) layers.

The phase-matching condition between incoming and outgoing waves can be expressed as

$$
\frac {2 \pi n _ {\mathrm {c}}}{\lambda} \sin \theta_ {\text {i n}} + m \frac {2 \pi}{\Lambda} = \frac {2 \pi n _ {\mathrm {c}}}{\lambda} \sin \theta_ {\text {o u t}}, \tag {1}
$$

where $m$ is the diffraction order (equal to 1 here), $\lambda$ is the free-space wavelength, $n_c$ is the top cladding index, $n_e$ is the effective index of an outbound wave, and $\theta_{\mathrm{in}}$ and $\theta_{\mathrm{out}}$ are the input and output angles relative to the grating surface normal. For vertical fiber-to-chip coupling $(\theta_{\mathrm{in}} = 0)$ , when $\Lambda = \lambda / n_e$ , we have $\theta_{\mathrm{out}} = \pi / 2$ for $m = 1$ and $\theta_{\mathrm{out}} = -\pi / 2$ for $m = -1$ .

This indicates that part of the optical power is diffracted along two in-plane opposite directions because of the symmetry configuration, resulting in a low directionality. Moreover, $\theta_{\mathrm{out}} = 0$ or $\pi$ if $m = 0$ , meaning that downward emission and backward refraction will also occur. To avoid light scattering in all directions, stacked gratings have recently received attention, which act as a set of tilted mirrors with inherent high directionality.

The working principle of the stacked gratings is shown as Fig. 2(a). When each grating shows an optical thickness of $\lambda /4$ , a wave scattered from a lower groove back to the fiber accumulates a phase shift of $\pi$ relative to the wave from its adjacent upper groove. Therefore, the light interferes destructively in the backward direction [red dashed arrow on the right in Fig. 2(a)], leading to low backreflections for both polarizations. For a wavelength of $1550~\mathrm{nm}$ , each grating is $110~\mathrm{nm}$ thick resulting in a total layer thickness of $220~\mathrm{nm}$ .

On the other hand, the lateral shift between the upper and the lower gratings (labeled $s$ in Fig. 2) is designed with an optical length of $\lambda /4$ to accrue an additional $\pi /2$ phase. In this case, the light interferes constructively in the left direction (red solid arrow) but destructively in the right (red dashed arrow), resulting in an extremely high directionality. Figures 2(b) and 2(c) depict diagrams of gratings with "small" ( $s > \Lambda D$ ) and "large" ( $s \leq \Lambda D$ ) duty factors, respectively. The lateral shift is solved as

Research Article

Vol. 40, No. 6 / June 2023 / Journal of the Optical Society of America A

1023

![](dt=2026-03-11/ht=12/400af81f7b5aa0233b2b7d2582256a8a7519f5bd90a686df57a5743e5dd613de.jpg)

![](dt=2026-03-11/ht=12/9a7ca99d1053417c36de4ac68f03cbb2ceef1a1de805bf877d5ea4ff86f6ffb1.jpg)

![](dt=2026-03-11/ht=12/d78bfbe332eebaeaba06f0b7cedd365c9b32472692e250e1f4509f2c7f444843.jpg)

$$
s = \left\{ \begin{array}{l l} \frac {\lambda}{4 n _ {e 1}} + \Lambda D \left(1 - \frac {n _ {e 2}}{n _ {e 1}}\right), & s > \Lambda D, \\ \frac {\lambda}{4 n _ {e 2}}, & s \leq \Lambda D, \end{array} \right. \tag {2}
$$

where $n_{\mathrm{c1}}, n_{\mathrm{c2}}$ , and $n_{\mathrm{e3}}$ are the effective indices of different grating sections as shown in Fig. 2. The pitch period $\Lambda$ can be, therefore, derived from the grating equation as

$$
\Lambda = \left\{ \begin{array}{l} \frac {m \lambda}{(1 - 2 D) n _ {\mathrm {e} 1} + D (n _ {\mathrm {e} 2} + n _ {\mathrm {e} 3})}, \quad s > \Lambda D, \\ \frac {m \lambda - \frac {\lambda}{4 n _ {\mathrm {e} 2}} (n _ {\mathrm {e} 2} + n _ {\mathrm {e} 3} - n _ {\mathrm {e} 1} - n _ {\mathrm {c}})}{(1 - D) n _ {\mathrm {e} 1} + D n _ {\mathrm {c}}}, \quad s \leq \Lambda D. \end{array} \right. \tag {3}
$$

The critical duty factor $D_{\mathrm{cv}}$ is obtained by substituting $s$ and $\Lambda$ into $s = \Lambda D$ , expressed as

$$
D _ {\mathrm {c v}} = \frac {n _ {\mathrm {e l}}}{2 n _ {\mathrm {e l}} + (4 m - 1) n _ {\mathrm {e} 2} - n _ {\mathrm {e} 3}}. \tag {4}
$$

# 3. DESIGN AND OPTIMIZATION

# A. Optimization of the First Grating

The periodic grating produces an exponentially decaying intensity profile along the grating, which does not match the approximate Gaussian distribution of the fiber mode. Therefore, in the realistic design, the grating has to be chirped. The first grating was chirped and optimized with $y$ -polarized light launched using the built-in particle swarm algorithm of a commercial FDTD solver (by Lumerical) in a 2D manner for maximum coupled power at a $1550~\mathrm{nm}$ wavelength.

The grating is surrounded by $\mathrm{SiO}_2$ , and the refractive indices of Si and $\mathrm{SiO}_2$ are 3.476 and 1.444, respectively. The thicknesses of the cladding, waveguide, gap, and BOX layers were initially set to $3\mu \mathrm{m}$ , $220\mathrm{nm}$ , $2\mu \mathrm{m}$ , and $2\mu \mathrm{m}$ , respectively. The grating width was $15\mu \mathrm{m}$ . The parameters of the lower $(\Lambda_L,D_L)$ and upper $(\Lambda_U,D_U)$ gratings were adjusted independently. The grating has 25 periods, and each period has four structural parameters.

Considering the lateral shift $s$ of the first period and the fiber displacement from the grating edge (labeled $p$ in Fig. 1), 102 $(25\times 4 + 2)$ parameters must be optimized in total.

To solve such an optimization problem with many variables, we preset the varying trends of the parameters on the basis of

theoretic analysis. The diffracted strength is approximately proportional to the duty factor of the grating. Based on this intuition, we assumed the duty factors increased linearly with the grating number $j$ . Therefore, $\Lambda_{L}$ grew approximately linearly based on Eq. (3), i.e.,

$$
D _ {\mathrm {L}} (j) = k _ {\mathrm {L}} j + b _ {\mathrm {L}},
$$

$$
D _ {\mathrm {U}} (j) = k _ {\mathrm {U}} j + b _ {\mathrm {U}},
$$

$$
\Lambda_ {\mathrm {L}} (j) = k _ {\Lambda} j + b _ {\Lambda}. \tag {5}
$$

$k_{L}, b_{L}, k_{U}, b_{U}, k_{\Lambda}, b_{\Lambda}$ are undetermined parameters. The lateral shift $s$ presented less variations and was fixed as $s_{sm}$ and $s_{la}$ for small and large duty factors, respectively. $\Lambda_{U}$ can be thereafter calculated by

$$
\Lambda_ {\mathrm {U}} (j) = \Lambda_ {\mathrm {L}} (j) + s (j + 1) - s (j). \tag {6}
$$

The duty factors of the first few periods would approach 0. This means that the structure contains features that are as small as a few nanometers wide and, hence, extremely difficult to be fabricated. Therefore, we constrained the optimization process by setting a lower bound on the etching width $w$ , and the parameters $(\Lambda_L, w_L, w_U,$ and $s)$ of the first three teeth are optimized individually. With the given trends and constraints, the variables are now reduced from 102 to 21.

Based on this method, we identified an optimized design for the mature $65\mathrm{nm}$ resolution lithography, i.e., a minimum feature size of $65~\mathrm{nm}$ . The lower bound on the etching width was set to $65~\mathrm{nm}$
. The optimization took roughly 200 iterations, after which the efficiency converged to $93\%$ $(-0.30\mathrm{dB})$ as shown in Fig. 3(a). The optimized parameters are displayed in Fig. 3(b) with $p = 5.4\mu \mathrm{m}$ .

Furthermore, better than $-0.66\mathrm{dB}$ is achievable for a minimum feature size of $130~\mathrm{nm}$ that has already been used in the commercial lithography. As the minimum feature size is further increased, the efficiency begins to drop rapidly.

# B. Determination of Gap Thickness

The second grating shares the same configuration as the first grating, but the strips are distributed orthogonally. By properly designing the gap between the two waveguide layers, the $x$ -polarized light can be efficiently coupled by the second grating and converted to the TE mode of to the second bottom layer. First, most $x$ -polarized power should be transmitted through the first grating and the interlayer. Therefore, the gap layer thickness $h_g$ here is chosen to generate destructive interference between upward reflections at the upper and lower interfaces of the interlayer.

Although this will penalize the CE of $y$ -polarized light for which minimum coupling loss can be achieved as the two reflection waves interfere constructively. Fortunately, due to the inherent high directionality of the stacked grating, the penalization is relatively insignificant. As shown in Fig.

4(a) for $h_g$ varied from $0.5 \mu \mathrm{m}$ to $2.5 \mu \mathrm{m}$ , the simulated transmittance (TR) of the first grating of $x$ -polarized light at $\lambda = 1550 \mathrm{~nm}$ yields a maximum of $99\%$ (-0.06 dB) with a fluctuation in a range of $45\%$ , whereas the fluctuation of the CE of $y$ -polarized light is below $2\%$ . Moreover, a high TR of $>95\%$ of $x$ -polarized light can still be maintained for $h_g$ deviations of $\pm 36 \mathrm{~nm}$ , which

1024

Vol. 40, No. 6 / June 2023 / Journal of the Optical Society of America A

Research Article

![](dt=2026-03-11/ht=12/86530bda47033bed473b0de7a74e5834f0db9c15c8e96a16094974624ceaaaa3.jpg)

![](dt=2026-03-11/ht=12/6a326950a3b6eddec98834e1659ed4ab829e984f794d3df75dc599c809ad1fe4.jpg)

![](dt=2026-03-11/ht=12/a969f8806ffa02f5b1e414503819de963ae822134b672c255a42dd3031080db2.jpg)

![](dt=2026-03-11/ht=12/e93b2cf068c856a565ae37f0ee6fba25017642f934de560756c0c1e666fe6b60.jpg)

shows a high tolerance of deposition thickness in practice. The fluctuation period is approximately $0.54\mu \mathrm{m}$ , which is equal to $\lambda /2n_{\mathrm{SiO2}}$ . Figure 4(b) shows the 3D simulated transmission

![](dt=2026-03-11/ht=12/af185790a3d31eb66e94afc8d1a50a34e37e17d9c244a47dd61f6dbe2f639999.jpg)

spectra of $x$ -polarized light to the second bottom layer with various $h_{g}$ 's and wavelengths from $1450 \mathrm{~nm}$ to $1650 \mathrm{~nm}$ . In one period, the central wavelength of the transmission spectrum is redshifted with the increase in $h_{g}$ .

However, considering the mode field changes and the diffraction of the second grating, $h_{g}$ is finally determined according to the coupling spectra of $x$ -polarized light. Figure 5 shows the 3D simulated coupling spectra detected in the output waveguide of the second grating with $x$ -polarized light launched from the fiber. The peak CE of $\sim 85\%$ can be achieved for $\lambda = 1557 \mathrm{~nm}$ at $h_{\mathrm{g}} = 0.925 \mu \mathrm{m}$ . Thus, here $h_{g}$ is tuned to $0.925 \mu \mathrm{m}$ to maximize the CE of $x$ polarization. Cladding thickness dependence is also simulated, which only makes the coupling efficiencies of both polarizations fluctuate within $0.2\%$ as it varies from $1 \mu \mathrm{m}$ to $3 \mu \mathrm{m}$ . This effect is negligible here.

# 4. SIMULATION AND DISCUSSION

# A. Performance of the Optimized GC

Figure 6 shows the simulated electric-field profile of the optimized GC. As desired, the $y$ -polarized light is coupled vertically to the first top waveguide layer, whereas the $x$ -polarized light is transmitted through the first waveguide layer to the second bottom layer. The 3D-FDTD simulation results are shown in Fig. 7. Record highs of $92\%$ $(-0.38\mathrm{dB})$ and $85\%$ $(-0.72\mathrm{dB})$ coupling efficiencies are achieved at wavelengths of $1549\mathrm{nm}$ and $1557\mathrm{nm}$ for $y$ and $x$ polarizations, respectively, as can be seen from the solid black and blue lines in Fig.

7(a). The relatively wide $3\mathrm{dB}$ bandwidth of $56\mathrm{nm}$ and $58\mathrm{nm}$ is obtained for $y$ and $x$ polarizations, respectively. The dashed black and blue lines in Fig. 7(a) show the CT induced by a specific polarization. The CT is predicted to be less than $-24\mathrm{dB}$ , which is much better than $-16\mathrm{dB}$ of a conventional 2D GC [27]. Furthermore, we also simulated the BR of this perfectly vertical GC as shown in Fig. 7(a) (red lines).

The BRs at the peak wavelengths for $y$ and $x$ polarizations are $\sim 2\%$ $(-16\mathrm{dB})$ and $\sim 7\%$ $(-12\mathrm{dB})$ respectively. Since the chirped GC was optimized at the peak wavelength compared with conventional periodic GC, the BRs are relatively large other than at the peak wavelength. For both polarizations, BRs are $< 47\%$ in the simulated wavelength range from $1450\mathrm{nm}$ to $1650\mathrm{nm}$ . Using asymmetric grating trenches, further improvements to the BR characteristics are expected [30]. Figure 7(b) shows the simulated PDL.

The separate optimization of two polarizations avoids huge peak-wavelength shifts between coupling spectra, hence, leading to a

Research Article

Vol. 40, No. 6 / June 2023 / Journal of the Optical Society of America A

1025

![](dt=2026-03-11/ht=12/1bcf3921319673baa5bb0a27d0d7b032132626ebd3af1392cf74a6204b6aee0a.jpg)

![](dt=2026-03-11/ht=12/977aa8d5d36feb72bd228fc03bd356457ccd6558b1dced62f8c38e655c5522f6.jpg)

![](dt=2026-03-11/ht=12/260fbae355f0cc25c6017cd0079c00f1fc6671e32fcc6607686f6399321757a0.jpg)

![](dt=2026-03-11/ht=12/5cebac99c5dccc6a010a5b238861dd8ddb3347a002a0d8fcb0c2d79fe0002b41.jpg)

![](dt=2026-03-11/ht=12/dddbe4ab3693351b75c18d324e5531648c06d830cb0a25fc448767259ed44c65.jpg)

![](dt=2026-03-11/ht=12/16e4b2c95b7ee0a99c5d77efd76b14fe52894cd2aa249b604f7afcbab7395acb.jpg)

reduced PDL of below 2 dB over spectral range from $1480\mathrm{nm}$ to $1640\mathrm{nm}$ .

The CE of $x$ polarization is slightly lower than that of $y$ polarization. This is primarily arising from the loss and a slight mode field change in the $x$ -polarized light as passing through the top waveguide layer. As shown in Fig. 8, the electric field of the $x$ -polarized light is attenuated, shifted, and deformed after passing through the first grating. The CE of $x$ -polarized

![](dt=2026-03-11/ht=12/7cd7d0112460976d1e711e908f2359568417820c018db340f8931d44e17fa73d.jpg)

![](dt=2026-03-11/ht=12/1e5ebdc8236ceda4cd39679eacb9a776d189d13fd849f6b7242df2745f587032.jpg)

light can be further improved by optim
izing the second grating to match the changed mode field distribution. However, the iterative process of 3D optimization is bound to be very time consuming.

# B. Fabrication Tolerances

The dual-layer GC can be fabricated by four times of inductively coupled plasma etching of Si and plasma-enhanced chemical vapor deposition of $\mathrm{SiO}_2$ after each etching. Three times of Si layer growth are also required. Fabrication tolerances of various parameters are also studied, such as the etching width, lateral shift, and fiber alignment. Figures 9(a)–9(c) show the calculated results of the CE against etching width errors $\Delta w$ , lateral shift errors $\Delta s$ , and fiber horizontal alignment deviations $\Delta p$ , respectively.

The simulation is only for the single layer and the polarization of interest is a $y$ polarization. It can be seen that as the etching width increases, the center passband is blueshifted. This is because an increase in the etching width results in a decrease in the effective refractive index of the grating, which corresponds to a blueshift of the peak wavelength according to the grating equation. The effect of the lateral shift on the CE and peak position is opposite to that of the etching width. Moreover,

![](dt=2026-03-11/ht=12/d1cb143e209e69060cb89701c631223abd5037361e5afdd25157a8d1d686bb12.jpg)

![](dt=2026-03-11/ht=12/18c9f065f63c9669bf3bb29b2912b1b337289b8cb498163a442fb8ea8d1e892b.jpg)

![](dt=2026-03-11/ht=12/81ec81f6888151dfd1ee428df0e0e6949460a6aaf4856765f34cda55e37e805a.jpg)

![](dt=2026-03-11/ht=12/4f2aa399d75f6a56a5888cbf0489ec65594c8f0df34f9c308de207d8f04d78c9.jpg)

![](dt=2026-03-11/ht=12/9fd28c93a0847edc3e88f31bfc62a3e39eb10614b2b0ba7f2bb2df071abaf849.jpg)

1026

Vol. 40, No. 6 / June 2023 / Journal of the Optical Society of America A

Research Article

Table 1. Comparison of Figures of Merits of Polarization-Diversity GCs in Recent Years ${}^{a}$

![](dt=2026-03-11/ht=12/4e7ac78e045906b03d43c9b37f968658559a4e5b5a02776754d4f2046d61bdfb.jpg)

<table><tr><td rowspan="2">Ref.</td><td rowspan="2">θin(°)</td><td rowspan="2">λc(nm)</td><td colspan="2">Coupling Loss (dB)</td><td rowspan="2">PDL and Wavelength Range</td><td rowspan="2">Feature Shape and Size (nm)</td><td rowspan="2">Reflector</td><td rowspan="2">Year</td></tr><tr><td>Sim.</td><td>Exp.</td></tr><tr><td rowspan="2">[31]</td><td rowspan="2">0</td><td rowspan="2">1550</td><td>-5</td><td>-7</td><td rowspan="2">/</td><td rowspan="2">Circle, /</td><td>No</td><td rowspan="2">2014</td></tr><tr><td>-4</td><td>-5.5</td><td>Side distributed Bragg reflectors (DBRs)</td></tr><tr><td rowspan="2">[16]</td><td rowspan="2">10</td><td rowspan="2">1550</td><td>-1.9</td><td rowspan="2">/</td><td rowspan="2">0.3 dB at λc</td><td>Circle, 167</td><td>No</td><td rowspan="2">2014</td></tr><tr><td>-0.95</td><td>Circle, 209</td><td>Bottom metal</td></tr><tr><td>[17]</td><td>10</td><td>1550</td><td>-5.8</td><td>-6</td><td>1 dB in 40 nm</td><td>Circle, 75</td><td>No</td><td>2015</td></tr><tr><td>[19]</td><td>14</td><td>1548</td><td>/</td><td>-5</td><td>0.25 dB in 40 nm</td><td>Cross, 110</td><td>No</td><td>2016</td></tr><tr><td rowspan="2">[18]</td><td rowspan="2">12</td><td rowspan="2">1550</td><td>-3.3</td><td>-4</td><td>/</td><td rowspan="2">Circle, 173</td><td>No</td><td rowspan="2">2018</td></tr><tr><td>-1.37</td><td>-1.8</td><td>1 dB at λc</td><td>Bottom metal</td></tr><tr><td>[20]</td><td>10</td><td>1285</td><td>/</td><td>-3.1</td><td>0.3 dB in 60 nm</td><td>Cross, 75</td><td>No</td><td>2018</td></tr><tr><td>[23]</td><td>34</td><td>1310</td><td>-2.1</td><td>-4.8</td><td>1 dB in 69 nm</td><td>Threelayer, 338</td><td>No</td><td>2018</td></tr><tr><td>[21]</td><td>12</td><td>1550</td><td>-3.4</td><td>-4.2</td><td>0.2 dB in 35 nm</td><td>Oval, 140</td><td>No</td><td>2019</td></tr><tr><td>[27]</td><td>0</td><td>1550</td><td>-2.4</td><td>-2.6</td><td>3 dB at λc</td><td>L shape, 40</td><td>No</td><td>2019</td></tr><tr><td>[29]</td><td>0</td><td>1550</td><td>-2.8</td><td>/</td><td>0.5 dB in 22 nm</td><td>Two layer, /</td><td>No</td><td>2019</td></tr><tr><td>[22]</td><td>3.3</td><td>1310</td><td>-1.73</td><td>-2.37</td><td>0.2 dB in 78 nm</td><td>Cross, 98.8</td><td>Bottom metal</td><td>2020</td></tr><tr><td>This paper</td><td>0</td><td>1550</td><td>-0.38</td><td>/</td><td>2 dB in 160 nm</td><td>1D strip, 65</td><td>No</td><td>2022</td></tr></table>

<sup>a</sup>Ref.: References, Sim.: Simulated, Exp.: Experimental, Feature size: Minimum circle radius or side length in an etched pattern.

a CE of $>74\%$ (a 1 dB loss penalty) can still be maintained at the wavelength of $1550~\mathrm{nm}$ for $\pm 10\mathrm{nm}$ etching width error or $\pm 8\mathrm{nm}$ lateral shift deviation. In Fig. 9(c), we observe an obvious decrease of the peak CE for fiber misalignment in $x$ direction. However, the peak CE can still reach $90\%$ for $\Delta p = \pm 1\mu \mathrm{m}$ and drops no more than $1\mathrm{dB}$ for $\Delta p = \pm 2.5\mu \mathrm{m}$ . The central wavelength is hardly affected. For $x$ -polarized light, similar effects occur when there are dislocations between the first and the second waveguide layers in $y$ direction.

Figures 9(d) and 9(e) show the tolerance towards fiber inclination along the $x(\theta_{x})$ and $y$ axes $(\theta_y)$ , respectively. The 3D simulations are performed here for both $y$ (solid lines) and $x$ polarizations (dotted lines). As shown in the figures, the first layer for $y$ polarization is more sensitive to the fiber inclination along the $x$ axis, whereas the second layer for $x$ polarization is more sensitive to the fiber inclination along the $y$ axis. When the fiber is tilted along the input polarization axis, the spectral variation is negligible.

Moreover, the dual-layer GC is more tolerant to the fiber inclination in negative directions (orientations of straight waveguides) than that in positive directions. Particularly, as $\theta_y = -2^\circ$ , the coupling spectrum of $x$ -polarized light is redshifted, but the peak CE increases compared to normal incidence. This mainly arises from the slightly changed mode field of $x$ -polarized light as shown in Fig. 8. Furthermore, the vertical CE of $x$ -polarized light can be further improved to $>90\%$ by detailed optimization for the second grating.

Table 1 summarizes the performances of reported polarization-diversity GCs in recent years. Although vertical incidence can be achieved in Refs. [27,29,31], the CE degrades significantly below $52\%$ in Ref. [29] and $40\%$ in Ref. [31], and a PDL of approximately 3 dB around the peak wavelength is observed in Ref. [27]. Our GC shows a significant improvement on the achievable CE and PDL simultaneously for perfectly vertical coupling without antireflection structures, such as bottom metal mirrors or DBRs. Even compared with angled coupling schemes, it outperforms others in subdecibel performance. Moreover, most etched cells is circular or cross

holes, and, thus, need precise manufacturing. The proposed pattern, however, is a simple 1D strip, which is compatible with industrial-scale manufacturing and can be fabricated using mature $65\mathrm{nm}$ COMS platform.

# 5. CONCLUSION

In conclusion, a highly efficient dual-layer polarization-diversity GC was demonstrated for perfectly vertical fiber-to-chip coupling. The two waveguide layers were orthogonally distributed and designed for $y$ - and $x$ -polarization states, respectively. Each layer consisted of two 1D stacked
gratings, allowing for both perfectly vertical coupling and high coupling directionality. The stacked gratings were optimized using the particle swarm algorithm.

The given varying trend of the parameters was employed on the basis of the theoretic analysis to considerably simplify the optimization process. The 3D-FDTD simulation $92\%$ $(-0.38\mathrm{dB})$ and $85\%$ $(-0.72\mathrm{dB})$ efficiencies were achieved for $y$ and $x$ polarizations, respectively, with the PDL below $2\mathrm{dB}$ in a $160~\mathrm{nm}$ spectral bandwidth. The cross talk between two polarizations was less than $-24\mathrm{dB}$ . To the best of our knowledge, this was the highest CE demonstrated with polarization diversity and vertical coupling hitherto.

The proposed pattern can be fabricated using mature $65~\mathrm{nm}$ resolution lithography. Fabrication imperfections were also investigated for realistic fabrication guidance with limited degradation in performance for an etching width uncertainty of $\pm 10\mathrm{nm}$ or a lateral shift deviation of $\pm 8\mathrm{nm}$ . We believe such a high-efficiency dual-layer GC will play a key role in polarization-dependent application and significantly enhance the integration density in silicon photonics.

Funding. National Natural Science Foundation of China (61890961, 62127813); Natural Science Basic Research Program of Shaanxi Province (2018JM6008).

Disclosures. The authors declare no conflicts of interest.

Research Article

Vol. 40, No. 6 / June 2023 / Journal of the Optical Society of America A

1027

Data availability. Data underlying the results presented in this paper are not publicly available at this time but may be obtained from the authors upon reasonable request.

# REFERENCES

1028

Vol. 40, No. 6 / June 2023 / Journal of the Optical Society of America A

Research Article