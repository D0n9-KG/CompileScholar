ARTICLE

Open Access

# Observation of mechanical bound states in the continuum in an optomechanical microresonator

Yue Yu $1$ , Xiang Xi $1$ and Xiankai Sun $1$

# Abstract

Bound states in the continuum (BICs) are a type of waves that are perfectly confined in the continuous spectrum of radiating waves without interaction with them. Here, we fabricated, with CMOS-compatible processes on a silicon chip, a wheel-shaped optomechanical microresonator, in which we experimentally observed the BIC in the micromechanical domain. The BIC results from destructive interference between two dissipative mechanical modes of the microresonator under broken azimuthal symmetry.

Such BICs can be obtained from devices with large and robust supporting structures with variable sizes, which substantially reduces fabrication difficulty and allows for versatile application environments. Our results open a new way of phonon trapping in micromechanical structures with dissipation channels, and produce long phonon lifetimes that are desired in many mechanical applications such as mechanical oscillators, sensors, and quantum information processors.

# Introduction

Micro- and nanomechanical resonators, which possess a very small mass and can be strongly coupled to light and matter, have been explored for precision metrology applications like mass and force sensing $^{1}$ and employed for investigating macroscopic quantum physics $^{2,3}$ . Reducing mechanical dissipation is crucial to these applications since it allows enhanced mechanical fields with long coherence time and thus leads to improved performance.

The conventional wisdom of reducing the dissipation loss relies on separating their eigenmodes from the continuum of lossy modes by constructing deep energy potentials with different materials or periodic structures $^{4,5}$ . For another type of nonperiodic individual resonators, where the bandgap shielding strategy cannot be applied, reducing the dissipation loss relies on minimizing their supporting structures $^{6,7}$ , which increases device fabrication difficulty and sets restrictions on their application areas.

For example, devices based on such delicate mechanical structures cannot be used repeatedly for fluid-based

applications, because they would likely fail when the ambient environment changes from a liquid to a gas.

Bound states in the continuum (BICs) refer to a type of eigenstates with infinite lifetime yet spectrally overlapping with lossy states in the continuum. Originally introduced to quantum mechanics, the concept of BICs has been extended to optical $^{9-14}$ , acoustic $^{15-18}$ , and mechanical $^{19,20}$ domains, and enabled many unprecedented applications such as low-threshold lasing $^{21-23}$ , ultrasensitive sensing $^{24}$ , and vortex beam generation $^{25}$ .

To date, most experimental demonstrations of BICs in optics and mechanics are based on periodic structures with certain symmetry $^{26-30}$ . These devices usually have a large footprint with a large modal volume or effective mass, which sets limitations to their application scenarios. In contrast to devices based on periodic structures, nonperiodic individual optical and mechanical resonators can more easily have confined fields with strong modal intensity at the micro/nanoscale, leading to a series of applications in precision metrology as well as studies of macroscopic quantum physics.

BICs in individual optical $^{31,32}$ and acoustic $^{33-36}$ resonators have been demonstrated. However, experimental demonstration of BICs in an individual mechanical resonator remains elusive.

Yu et al. Light: Science & Applications (2022)11:328

https://doi.org/10.1038/s41377-022-00971-w

Official journal of the CIOMP 2047-7538

www.nature.com/Isa

Correspondence: Xiankai Sun (xksun@cuhk.edu.hk)

$^{1}$ Department of Electronic Engineering, The Chinese University of Hong Kong,

Shatin, New Territories, Hong Kong SAR, China

© The Author(s) 2022

CC BY

Open Access This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction

in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons license, and indicate if

changes were made. The images or other third party material in this article are included in the article's Creative Commons license, unless indicated otherwise in a credit line to the material. If material is not included in the article's Creative Commons license and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.

Here, we experimentally demonstrated mechanical BICs in an optomechanical microresonator. By breaking the azimuthal symmetry, we introduced coupling between a radial-contour mode and a wine-glass mode of a wheel-shaped structure to obtain destructive interference between energy dissipation of the two modes, which produces a mechanical BIC under the Friedrich-Wintgen condition[37]. The mechanical BIC was experimentally confirmed by optomechanical measurement of the devices in vacuum.

In contrast to conventional BICs requiring certain symmetry, the demonstrated mechanical BICs represent a new paradigm for constructing high- $Q$ micromechanical resonators through symmetry breaking[38]. In addition, the low-loss mechanical BIC has high tolerance on the supporting rods' width from hundreds of nanometers to several micrometers.

# Results

To construct BICs in an individual mechanical resonator, suppose we have a resonator supporting two dissipative modes coupled with each other, as shown in Fig. 1a. Such a system can be described by a Hamiltonian

$$
H = \left( \begin{array}{c c} \omega_ {1} - j \gamma_ {1} & \kappa - j \sqrt {\gamma_ {1} \gamma_ {2}} \\ \kappa - j \sqrt {\gamma_ {1} \gamma_ {2}} & \omega_ {2} - j \gamma_ {2} \end{array} \right) \tag {1}
$$

where $\omega_{1}(\gamma_{1})$ and $\omega_{2}(\gamma_{2})$ are the resonant frequencies (dissipation rates) of the two modes. The two modes are coupled with each other with a coupling coefficient $\kappa$ , which results in an anticrossing of these two modes. At this anticrossing point, when the Friedrich-Wintgen condition

$$
\kappa \left(\gamma_ {1} - \gamma_ {2}\right) = \sqrt {\gamma_ {1} \gamma_ {2}} \left(\omega_ {1} - \omega_ {2}\right) \tag {2}
$$

is satisfied $^{13,39}$ , the complex resonant frequencies become (see Supplementary Information, Section S1)

$$
\Omega_ {1} = \frac {\omega_ {1} + \omega_ {2}}{2} + \frac {\kappa \left(\gamma_ {1} + \gamma_ {2}\right)}{2 \sqrt {\gamma_ {1} \gamma_ {2}}} - j \left(\gamma_ {1} + \gamma_ {2}\right) \tag {3}
$$

$$
\Omega_ {2} = \frac {\omega_ {1} + \omega_ {2}}{2} - \frac {\kappa \left(\gamma_ {1} + \gamma_ {2}\right)}{2 \sqrt {\gamma_ {1} \gamma_ {2}}} \tag {4}
$$

As shown in Eq. (4), $\Omega_{2}$ has a vanishing imaginary part, which means that this hybrid mode experiences zero dissipation loss and thus can be a BIC. In this system, when one of the two hybrid modes becomes lossless, the Friedrich-Wintgen condition is also satisfied (See Supplementary Information, Section S2). Therefore, one can verify a Friedrich-Wintgen BIC by measuring the dissipation loss of the two hybrid modes of the system.

First, we consider a ring-shaped thin-plate micromechanical resonator as shown in Fig. 1b. It is made in 220-nm-thick silicon and has an inner radius $r$ and an outer

radius $R$ ( $r, R \gg 220 \mathrm{~nm}$ ). Such resonators support two types of in-plane mechanical modes: radial-contour modes and wine-glass modes. Since these two types of modes have different dependence on $r$ , fixing $R = 26.1 \mu \mathrm{m}$ and varying $r$ lead to a crossing of resonant frequencies of the fundamental radial-conto
ur mode (mode A in Fig. 1b) and the 4th-order wine-glass mode (mode B in Fig. 1b). In a ring-shaped resonator with perfect azimuthal symmetry, mode A and mode B are orthogonal to each other without modal coupling.

Therefore, the Friedrich-Wintgen condition in Eq. (2) for BICs cannot be satisfied. To introduce modal coupling for satisfying the Friedrich-Wintgen condition, we break the azimuthal symmetry of the ring-shaped resonator by modifying its inner boundary to an ellipse, with semi-major and semi-minor axes being respectively $r_x$ and $r_y$ , as shown in Fig. 1c.

Figure 1c also plots the simulated modal frequencies of the modified structure as a function of $r_x$ with fixed $r_y = 18.7 \mu \mathrm{m}$ and $R = 26.1 \mu \mathrm{m}$ , where an anticrossing occurs near $r_x = 20.6 \mu \mathrm{m}$ indicating the coupling between the two mechanical modes. At the anticrossing point, the energy exchange between the original mode A and mode B leads to two hybrid modes, namely mode A' and mode B', as shown in Fig. 1d. Note that although the structure in Fig.

1c can have the required coupling between different modes which can support a BIC, a realistic device must also include supporting structures that are connected to the substrate, which act as the dissipation channel for both mechanical modes. Compared with the original modes A and B, the hybrid modes A' and B' have larger regions where the modal displacement is near zero (Fig. 1d). Therefore, by attaching the supporting structures to these regions, it is possible to reduce energy dissipation of the ring-shaped mechanical resonator to the substrate.

Next, we investigate a realistic structure in which two supporting rods are added to the azimuthal-symmetry-broken ring-shaped resonator making a wheel-shaped resonator as shown in Fig. 2a, b. We need to engineer this structure and analyze the modal coupling to satisfy the Friedrich-Wintgen condition [Eq. (2)] for constructing a mechanical BIC. Figure 2a is a three-dimensional view of the entire device structure where the wheel-shaped resonator is seated on a silicon oxide $(\mathrm{SiO}_2)$ pedestal on the substrate.

Figure 2b shows the top and side views of the entire device, where the wheel-shaped silicon micromechanical resonator, the $\mathrm{SiO}_2$ pedestal, and the substrate are marked in blue, black, and gray, respectively. The additional two parameters for the wheel-shaped resonator $d$ and $r_s$ are the supporting rods' width and the center disk radius, respectively.

To investigate the influence of the supporting rods on the modal coupling, we simulated the frequencies and mechanical $Q$ factors of mode A' and mode B' as a function of the semi-major axis $r_x$ for different rod widths $d$ , with the results shown in Fig. 2c, d.

Yu et al. Light: Science & Applications (2022)11:328

Page 2 of 8

![](dt=2026-03-18/ht=16/a97ea9394f6fe7b8743fa1d5cff0357d9f5d324261b5d1e31d55b21d314593ba.jpg)

![](dt=2026-03-18/ht=16/f139f8b7a20a2c4268feb86faa7bba2579816a3d5570b7c4a6e219bc363b2042.jpg)

![](dt=2026-03-18/ht=16/f11cd3141422d076e986ac3f9adf44c8b81232307a296d79c8cb4ac63a4fb610.jpg)

![](dt=2026-03-18/ht=16/87a30f0dd1026581a6d52577f3cab2ff488154b40d3b458fa6eda8e2c1085d02.jpg)

![](dt=2026-03-18/ht=16/42e394b80881e89223143c63fe14017b9039095a34ca50a29657e7a021c06ce8.jpg)

![](dt=2026-03-18/ht=16/a81cf9be76fac7cb712148e324b9be755f96165d246fb576b5e8be3666bb687b.jpg)

![](dt=2026-03-18/ht=16/67503da456b65a8613f6461964aa6fb15b91a151598e726e2d28272c36755c2a.jpg)

![](dt=2026-03-18/ht=16/25c4587aa2f63d5e4dc9569e95b3b784160cd728967d90587b03d979dbdac9f9.jpg)

![](dt=2026-03-18/ht=16/5811df00fba4bddcfb975d9547a030cc30077780f8d54fec17a4ff0295b4a655.jpg)

![](dt=2026-03-18/ht=16/d5f1966006229ba6642105d38a65efe80ca564ce6ddc63b2eb3ece1dfc17d1f3.jpg)

![](dt=2026-03-18/ht=16/473ae91f57188a45370e89cd83f0c559e47fd4cc71219aebcb721c8fe7951b7e.jpg)

![](dt=2026-03-18/ht=16/97073521086435595ead03ea4972f8e50831dd4759a92150e6952a1a2ed445df.jpg)

![](dt=2026-03-18/ht=16/5036beb0ba64d313bc5e64dc3fd057d11a726b898a88354749674ab9e2669705.jpg)

![](dt=2026-03-18/ht=16/78414b8f51d2e90a91571bfea6ec4d858ff5607721a934e0017cccd830af77f2.jpg)

![](dt=2026-03-18/ht=16/0d2a64bee9af82fce2daf7cdee813f1e6b2ffd206dcae8c1fa8dd620fee8342b.jpg)

The other geometric parameters are fixed at $r_y = 18.7 \, \mu \mathrm{m}$ , $R = 26.1 \, \mu \mathrm{m}$ , and $r_s = 14.7 \, \mu \mathrm{m}$ . The insets in Fig. 2c show the displacement profiles of the corresponding mechanical modes. It can be found that an anticrossing in the modal frequencies (Fig. 2c) and a drastic variation in the mechanical $Q$ factor of mode A' (Fig. 2d) occur simultaneously near $r_x = 20.8 \, \mu \mathrm{m}$ , despite a large variation of $d$ from 0.5 to $5 \, \mu \mathrm{m}$ . These behaviors indicate that mode A' becomes a Friedrich-Wintgen quasi-BIC $^{37}$ .

Note that the high- $Q$ Friedrich-Wintgen BIC can be obtained from the wheel-shaped resonator with $d$ as large as several micrometers. One reason is that the hybrid mode A' has a larger region of near-zero displacement than the original uncoupled wine-glass mode (mode B). Actually, we simulated a series of structures with $d$ varying from 0.5 to $8 \, \mu \mathrm{m}$ and collected the $r_x$ value and mechanical $Q$ factor when the BIC is achieved (marked by the red circles in Fig. 2d), with the results plotted in Fig. 2e, f, respectively.

Figure 2f shows that the simulated mechanical $Q$ factor of

the BIC can maintain above $10^{8}$ in such a wide range of $d$ from 0.5 to $8\mu \mathrm{m}$ (See Supplementary Information, Section S5), demonstrating excellent robustness against variations of the width of the dissipation channel. Compared with conventional mechanical systems which rely on minimized supporting rods[6,7] or surrounding phononic bandgap structures[5] for reducing the clamping loss and achieving high mechanical $Q$ factors, the Friedrich-Wintgen BIC can exist in mechanical resonators with simply designed sturdy supporting structures, which substantially alleviate device fabrication difficulty, facilitate thermalization and heat dissipation, and enable device applications in versatile environments.

To measure the mechanical BIC, we fabricated the wheel-shaped optomechanical microresonators on a silicon-on-insulator wafer and used optomechanical transduction for detecting their mechanical $Q$ factors. Under the guidance of theoretical analysis and numerical simulation, we varied the parameter $r_x$ for different

Yu et al. Light: Science & Applications (2022)11:328

Page 3 of 8

![](image)
d.jpg)

![](dt=2026-03-18/ht=16/0281bd11ae34732b37aaf80643af3c16f92f42d49ea898abc2aa2c7fcdacc681.jpg)

![](dt=2026-03-18/ht=16/01ac5d72ac3f06a4812f9c129fc32cffb4fbe24cce89a33a01ffc08ebe541555.jpg)

![](dt=2026-03-18/ht=16/63f937f8f375ff6a4616adc0ab9b4689036ebcaa0ee879d48da0ea032322193e.jpg)

![](dt=2026-03-18/ht=16/9cbc7d3a67d45a18a19cd6f36014e076b07e2f9b221d39bcd6b87e18ab2ea09c.jpg)

![](dt=2026-03-18/ht=16/4c6bb466228a45fe7859c10e0f1972eaa55f8c2895b7725dc542e1a00511cc3f.jpg)

![](dt=2026-03-18/ht=16/c66b7c77f2c9fd9aba4824a83291ac1df9efa16fce8a2dd33cb802e289f83728.jpg)

![](dt=2026-03-18/ht=16/287327381f0bcbf4dca0dada3480ee51b5f07a03897b0a122f739fb092ea5787.jpg)

![](dt=2026-03-18/ht=16/771d44542f58d35c0a49cb26a414227f219ec2240838624fe7cde676c1d46c8d.jpg)

![](dt=2026-03-18/ht=16/c981e086d557c2226064f1e6050cd04d07260070a8a7724a724250ac78a90732.jpg)

resonator devices while keeping the following structural parameters fixed: $d = 5 \mu \mathrm{m}$ , $r_s = 14.7 \mu \mathrm{m}$ , $r_y = 18.7 \mu \mathrm{m}$ , and $R = 26.1 \mu \mathrm{m}$ . Figure 3a shows scanning electron microscope images of a fabricated device. Note that the wheel-shaped optomechanical microresonator also supports optical whispering-gallery modes circulating around its outer periphery, which were employed to detect the thermomechanical vibration of the resonator via optomechanical transduction. We also fabricated a bus waveguide in close proximity of the resonator for coupling light into and out of the resonator. The inset of Fig. 3a is a

close-up showing the details in the coupling region of the resonator and bus waveguide. Figure 3b shows the experimental setup for device characterization. Figure 3c plots a measured optical transmission spectrum of the resonator, where the dips correspond to the optical whispering-gallery modes in different orders. Figure 3d is a close-up of a dip at $\sim 1558.4\mathrm{nm}$ , which shows that the loaded optical $Q$ factor is $2.3\times 10^{5}$ .

To experimentally verify the mechanical BIC, we measured the wheel-shaped optomechanical microresonators in a vacuum chamber, which could provide an ambient

Yu et al. Light: Science & Applications (2022)11:328

Page 4 of 8

![](dt=2026-03-18/ht=16/41e27e3458b3467bec8fefd1d6a0ee128a878e9c660d4d04683e885470b010ff.jpg)

![](dt=2026-03-18/ht=16/de69d69287dc63d3e20b4ba6ecf2f5bf34e0a8875d2bdef5894afa090506d519.jpg)

![](dt=2026-03-18/ht=16/6a19b2adc81c33f6e40d211913bdb1ce2ad787e23530df70d7b3ebba9bada932.jpg)

pressure from $1.0 \times 10^{5}$ to $6.0 \times 10^{-3} \mathrm{~Pa}$ for the devices. Figure 4a plots the simulated and measured frequencies of modes A' and B' (modal profiles in Fig. 4a insets) for devices with different $r_x$ . The simulated results are extracted directly from the rightmost plot of Fig. 2c. The measured results agree well with the simulated results, which confirms the existence of the two modes. Figure 4b plots the mechanical Q factors of modes A' and B' as a function of $r_x$ measured at the ambient pressure of $6.0 \times 10^{-3} \mathrm{~Pa}$ .

Mode A' achieves its maximal mechanical Q factor of 9453 at $r_x = 20.8 \mu \mathrm{m}$ , which agrees with the simulated results in Fig. 2d. Therefore, we confirm attainment of the mechanical BIC in our fabricated optomechanical microresonators. Figure 4c shows the measured displacement noise power spectral density of modes A' and B' at $r_x = 20.8 \mu \mathrm{m}$ where the BIC is achieved. The Lorentzian fitting of these spectra provides the mechanical Q factor of 9453 for mode A' and 882 for mode B'.

The measured mechanical Q factor of mode A' at the BIC point is lower than the simulated value in Fig. 2d.

It should be noted that the strategy of engineering a Friedrich-Wintgen BIC can only be used for eliminating the clamping loss. The other loss mechanisms such as air damping loss and material loss cannot be reduced effectively by the structural engineering and modal control $^{40,41}$ . Next, we investigated the residual loss in our BIC device ( $r_x = 20.8 \mu \mathrm{m}$ ) where the clamping loss has been completely eliminated. To this end, we measured its mechanical $Q$ factor under different ambient pressures in the vacuum chamber, with the results shown in Fig. 4d.

The mechanical $Q$ factors for devices with other $r_x$ values at different ambient pressures can be found in Supplementary Information, Section S6. Figure 4d shows that the mechanical $Q$ factor decreases as the ambient pressure increases and this effect becomes more pronounced when the ambient pressure is above $1 \mathrm{~Pa}$ , which indicates that air damping loss was the main loss mechanism. Since material loss usually depends on temperature and maintains constant at a given temperature, e.g., room temperature in our experiment, we can express the

Yu et al. Light: Science & Applications (2022)11:328

Page 5 of 8

![](dt=2026-03-18/ht=16/d3b28258b5289546bac8ca6af4f128515da0590ecfe308fa780cb8cc3d148d9f.jpg)

![](dt=2026-03-18/ht=16/0ceca3a86cfebf23540fe5156f197f38b9f26419c746815c129a588dd57afe0b.jpg)

![](dt=2026-03-18/ht=16/2a74785f1ae5643571d7dfa140c32de6442787d49ef6c556d5f014ff08fbb143.jpg)

![](dt=2026-03-18/ht=16/025c6db09ed82e0c444180031c03f3881e0b20e7a901743a855ff27e83353cee.jpg)

mechanical $Q$ factor as

$$
Q ^ {- 1} = Q _ {0} ^ {- 1} + Q _ {\text {e x t}} ^ {- 1} = Q _ {0} ^ {- 1} + C ^ {- 1} P \tag {5}
$$

where $Q_0$ , $Q_{\mathrm{ext}}$ , and $P$ are the intrinsic $Q$ factor, extrinsic $Q$ factor, and the ambient pressure, respectively. $C$ is a proportionality constant defined as $\rho hf\sqrt{\pi^3RT / 8M}$ where $\rho$ , $h$ , $f$ , $R$ , $T$ , and $M$ are the material mass density, resonator thickness, mechanical frequency, ideal gas constant, temperature, and molar mass of air, respectively[42].

With $\rho = 2329\mathrm{kgm}^{-3}$ , $h = 220\mathrm{nm}$ , $f = 57\mathrm{MHz}$ , $R = 8.31\mathrm{JK}^{-1}\mathrm{mol}^{-1}$ , $T = 300\mathrm{K}$ , and $M = 28.97\mathrm{g mol}^{-1}$ , $C$ has the theoretically calculated value of $1.69\times 10^{7}\mathrm{Pa}$ . Note that the above expression for the mechanical $Q$ factor in Eq. (5) applies only to a relatively low ambient pressure where the free-molecular-flow approximation is

valid. Under a high pressure, the air-damping-dominated $Q_{\mathrm{ext}}$ follows a $P^{-1/2}$ dependence<sup>42</sup>. Therefore, we fitted the experimental results at the ambient pressure below $10^{4}$ Pa based on Eq. (5), obtaining the orange curve shown in Fig. 4d with the fitted $C$ being $1.49 \times 10^{7}$ Pa, which agrees well with the theoretically calculated value.

# Discussion

In summary, we experimentally realized a
BIC in an individual optomechanical microresonator, which provides a new strategy of phonon trapping in micromechanical structures with dissipation channels. By breaking the azimuthal symmetry, we introduced coupling between two dissipative mechanical modes of a wheel-shaped microresonator for making destructive interference between the dissipation channels. As a result,

Yu et al. Light: Science & Applications (2022)11:328

Page 6 of 8

we obtained a Friedrich-Wintgen BIC with zero clamping loss, and achieved a mechanical $Q$ factor of $\sim 10^4$ in the very high frequency band at room temperature from a wheel-shaped optomechanical microresonator with its supporting rods' width as large as $5\mu \mathrm{m}$ . To obtain high- $Q$ resonances in individual micromechanical resonators, the conventional wisdom relies on minimizing the size of the supporting structures which renders the fabricated mechanical device fragile.

Defying the conventional wisdom, our strategy applies to robust mechanical structures, which not only substantially reduces device fabrication difficulty but also enables device operation in versatile environments for broader application areas. Our experimental results open a new way of obtaining high- $Q$ micro- and nanomechanical resonators, which will inspire plenty of applications in interdisciplinary research areas such as electromechanics, optomechanics, and quantum physics.

# Materials and methods

# Simulation

A finite-element method was adopted to simulate the mechanical modes in commercial software COMSOL. The following parameters were used in the simulation model: silicon's Young's modulus $E = 150 \, \mathrm{GPa}$ , Poisson's ratio $\nu = 0.28$ , and mass density $\rho = 2329 \, \mathrm{kg} \, \mathrm{m}^{-3}$ ; silicon oxide's Young's modulus $E = 70 \, \mathrm{GPa}$ , Poisson's ratio $\nu = 0.17$ , and mass density $\rho = 2200 \, \mathrm{kg} \, \mathrm{m}^{-3}$ . A 2-μm-thick perfectly matched layer was placed at the bottom of the substrate for analysis of the mechanical loss. More details of mechanical simulation can be found in Supplementary Information, Section S4.

# Fabrication

The devices were fabricated with CMOS-compatible processes on a standard silicon-on-insulator wafer, where the thicknesses of the top silicon device layer and the buried silicon oxide layer are $220\mathrm{nm}$ and $3\mu \mathrm{m}$ , respectively. The patterns of the optomechanical microresonator, bus waveguide, and grating couplers (not shown in Fig. 3) were defined by high-resolution electronbeam lithography in an electron-beam resist (ZEP520A).

Then, the patterns in the electron-beam resist were transferred to the top silicon device layer by plasma dry etching with $\mathrm{SF}_6 / \mathrm{C}_4\mathrm{F}_8$ chemistry. Next, a step of photolithography was performed to define the areas to be exposed for wet etching. After that, the optomechanical microresonators were released from the substrate by wet etching in a buffered oxide etchant. Finally, the devices were dried in a critical point dryer to prevent stiction.

# Measurement

The fabricated devices were placed inside a vacuum chamber, in which the pressure could be varied from $1.01 \times 10^{5}$ to $6.0 \times 10^{-3}$ Pa. A laser beam from a tunable

semiconductor laser (TSL) was sent through a fiber polarization controller (FPC) and a variable optical attenuator (VOA) before it was coupled into the bus waveguide of the device under test (DUT) via an input grating coupler. The laser beam was further coupled into the optomechanical microresonator to detect its thermomechanical vibration via optomechanical transduction (see details in Supplementary Information, Section S3).

To obtain the intrinsic mechanical $Q$ factor, the laser beam was attenuated by using the VOA such that the dynamic backaction from optomechanical interaction was negligible in the resonator (See Supplementary Information, Section S6). The light coupled out of the DUT was first amplified by a low-noise erbium-doped fiber amplifier (EDFA) and then collected by a photodetector (PD). Then, the optical signal carrying the mechanical modal information of the optomechanical microresonator was converted into the electrical domain.

The converted electrical signal was received by a signal analyzer for producing the power spectral density. The mechanical $Q$ factors were obtained by fitting the mechanical resonant peaks in the measured power spectral density with a Lorentzian line shape (see details in Supplementary Information, Section S3).

# Acknowledgements

This work was supported by the Research Grants Council of Hong Kong (Project Nos. 14208717 and 14208421).

# Author contributions

X.S. conceived the idea. Y.Y. performed the theoretical modeling, device fabrication, device characterization, and data analysis. X.X. assisted in the device fabrication. Y.Y., X.X., and X.S. wrote the manuscript. X.S. supervised the project.

# Data availability

The data that support the findings of this study are available from the corresponding author upon reasonable request.

# Conflict of interest

The authors declare no competing interests.

Supplementary information The online version contains supplementary material available at https://doi.org/10.1038/s41377-022-00971-w.

Received: 8 April 2022 Revised: 26 August 2022 Accepted: 28 August 2022

Published online: 18 November 2022

# References

Yu et al. Light: Science & Applications (2022)11:328

Page 7 of 8

Yu et al. Light: Science & Applications (2022)11:328

Page 8 of 8