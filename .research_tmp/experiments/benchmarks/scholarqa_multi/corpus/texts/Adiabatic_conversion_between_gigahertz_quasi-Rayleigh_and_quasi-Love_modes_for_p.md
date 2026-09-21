# Adiabatic conversion between gigahertz quasi-Rayleigh and quasi-Love modes for phononic integrated circuits

Bao-Zhen Wang, $^{1}$ Xin-Biao Xu, $^{2,4,*}$ Yan-Lei Zhang, $^{2,4}$ Weiting

Wang, $^{3}$ Luyan Sun, $^{3}$ Guang-Can Guo, $^{2,4}$ and Chang-Ling Zou $^{2,4,5,\dagger}$

$^{1}$ School of Civil Engineering, Hefei University of Technology, Hefei, Anhui 230009, P.R. China.

$^{2}$ CAS Key Laboratory of Quantum Information, University of Science and Technology of China, Hefei, Anhui 230026, China.

$^{3}$ Center for Quantum Information, Institute for Interdisciplinary

Information Sciences, Tsinghua University, Beijing 100084, China

$^{4}$ CAS Center For Excellence in Quantum Information and Quantum Physics,

University of Science and Technology of China, Hefei, Anhui 230026, China

$^{5}$ National Laboratory of Solid State Microstructures, Nanjing University, Nanjing 210093, China.

Unsuspended phononic integrated circuits have been proposed for on-chip acoustic information processing. Limited by the operation mechanism of a conventional interdigital transducer, the excitation of the quasi-Love mode in GaN-on-Sapphire is inefficient and thus a high-efficiency Rayleigh-to-Love mode converter is of great significance for future integrated phononic devices. Here, we propose a high-efficiency and robust phononic mode converter based on an adiabatic conversion mechanism. Utilizing the anisotropic elastic property of the substrate, the adiabatic mode converter is realized by a simple tapered phononic waveguide. A conversion efficiency exceeds 98% with a 3 dB bandwidth of 1.7 GHz can be realized for phononic waveguides working at GHz frequency band, and excellent tolerance to the fabrication errors is also numerically validated. The device that we proposed can be useful in both classical and quantum phononic information processing, and the adiabatic mechanism could be generalized to other phononic device designs.

# I. INTRODUCTION

In the past decades, phononic devices have played an important role in RF signal processing, and found applications in various fields include radars $[1]$ , communications $[2]$ , and sensing $[3, 4]$ . Recently, phononic microstructures have also been used for the studies of quantum mechanics $[5]$ , for the controlling of photons, electrons $[6, 7]$ , quantum dots $[8, 9]$ , NV centers $[10, 11]$ , and superconducting qubits $[12–15]$ . In contrast to the conventional acoustic devices that utilize surface acoustic waves or bulk acoustic waves without lateral confinement of phonon, an unsuspended phononic integrated circuit (PnIC) platform has been proposed to guide and manipulate phonons with a strong lateral confinement $[16, 17]$ . In such a PnIC, phonons are confined by the high-acoustic-index-contrast between the waveguide and the substrate material, and various compact integrated phononic devices have been developed $[16–19]$ . Compared with the conventional acoustic devices, PnICs have the advantages of good extensibility, stability, and field restriction capability. Besides, by integrating phononic and superconducting devices together, a powerful hybrid integrated chip can be constructed for both classical and quantum information processing.

In a PnIC, the efficient excitation and manipulation of the phononic waveguide modes are essential. However, due to the anisotropy of the piezoelectric material, some phononic modes are piezoelectrically inactive. For instance, due to the limitation of the piezoelectric activity of the material and the fabrication capability, only the guiding mode with out-of-plane displacement (quasi-Rayleigh mode) could be efficiently excited by an interdigital transducer (IDT) fabricated on the GaN-on-Sapphire (GNOS) chip $[16]$ . Hence, the quasi-Rayleigh mode has received more research attention and has been widely used in phononic devices. In fact, the quasi-Love mode that is dominated by in-plane displacements has different properties that are complementary to the quasi-Rayleigh mode, and thus has huge potential in future phononic applications. For example, the in-plane displacement of the quasi-Love mode makes the phonons have minimal damping in the air and liquid environment, which is favorable for high-quality integrated phononic devices. However, this mode is piezoelectrically inactive and is generally difficult to be directly excited by an IDT. Therefore, a robust and high-efficiency phononic mode converter between the quasi-Rayleigh and the quasi-Love modes is critical for a multi-functional PnIC.

Here, a high-efficiency mode conversion between the quasi-Rayleigh and the quasi-Love modes is investigated theoretically. The mode analysis of the GNOS waveguide structure is conducted to study the anisotropic-substrate-induced mode coupling of phononic modes. The acoustic wave propagation in the waveguide is analogous to the temporal state evolution in a quantum system, and a nearly perfect mode conversion can be realized by adiabatically changing the wave number along the waveguide, i.e. in a tapered phononic waveguide by adiabatically changing the waveguide width. As confirmed by 3D simulations, the proposed adiabatic converter has a large working bandwidth and is robust against geometry parameter uncertainties. We expect the devices that take advantage of the adiabatic evolution mechanisms would

![](images/189fd575cafd2179c36ff039ed022ee3439f604fd661beb151cccea2db96cabe.jpg)

<details>
<summary>text_image</summary>

(a)
Acoustic Wave
GaN
w
h
Sapphire
(b)
Quasi-Rayleigh Mode
(c)
Z[0001]
y[11̅00]
x[11̅20]
</details>

FIG. 1. (a) Schematic of the strip GaN waveguide on a c-plane sapphire substrate along the [1120] direction. w and h represent the width and thickness of the waveguide, respectively. (b)-(c) The displacement fields of the quasi-Rayleigh and the quasi-Love eigenmodes at the cross-section of the phononic waveguide, respectively.

play a significant role in future studies of PnICs.

# II. ANISOTROPIC-SUBSTRATE-INDUCED MODE COUPLING

In this work, we consider a PnIC platform based on the GNOS [16]. The schematic diagram of the model, which consists of a tapered GaN waveguide on c-plane sapphire substrate, is shown in Fig. 1(a). The x, y, and z axes are consistent with the three orthogonal crystal directions of the sapphire substrate: $[11\bar{2}0]$ , $[1\bar{1}00]$ , and $[0001]$ , respectively. In the following analyses, the thickness of the GaN waveguide is set as $h = 0.7\mu \mathrm{m}$ . Figures 1(b) and (c) show the typical mode field distributions of the fundamental quasi-Rayleigh $(R_0)$ and quasi-Love $(L_0)$ modes of the phononic waveguide that are simulated by a finite-element method (COMSOL Multiphysics ver5.2). The quasi-Rayleigh and the quasi-Love modes are dominated by the out-plane and in-plane deformations, respectively. The exponential decay of the evanescent field in the substrate is similar to that of an optical waveguide [20], and the evanescent acoustic field enables the directional couplers for the coupling between phononic waveguides and resonators [17]. The material parameters of GaN and sapphire are given in Ref. [21].

The propagation properties of the acoustic wave in the phononic waveguide are closely related not only to the dimension and the elastic properties of the waveguide, but also to the elastic properties of the substrate. Because of its hexagonal crystal structure, GaN exhibits isotropic mechanical properties in the c-plane and has five independent elastic constants. However, the sapphire with a trigonal crystal structure has six independent elastic constants and shows anisotropic elastic properties. To see the influence of the anisotropy sapphire substrate on the phononic modes in the GaN waveguide, we perform numerical simulations and analyze the geometry-dependent mode wave vector, as shown in Fig. 2. At a given frequency of 1.2 GHz, the confined phononic modes in the waveguide are studied with the waveguide width ranging from $0.5 \mu m$ to $6 \mu m$ .

![](images/e1ffb54caa74ee3f7af4aeb83baee8cb1820471e115ba4dabf952b0f0ddc55d6.jpg)  
FIG. 2. The calculated wave numbers as a function of w. (a) the waveguide along the $[11\bar{2}0]$ direction with f = 1.2 GHz. The insets show the deformation mode shapes at the specific widths. (b) the waveguide along the $[1\bar{1}00]$ direction with f = 1.2 GHz. (c) the waveguide along the $[11\bar{2}0]$ direction with f = 5 GHz. (d) the waveguide on an isotropic substrate with f = 1.2 GHz.

Figure 2(a) investigates the dependence of the eigenmode wave vector on the waveguide width for the waveguide along the $[11\bar{2}0]$ direction. The inset plots the mode profiles of $R_{0}$ and $L_{0}$ modes. The black and red curves do not cross each other and form an avoid-crossing line-shape in the figure. However, for the modes in the avoid-crossing regions, the eigenmodes show hybridization of horizontal and vertical vibrations and correspond to the coupling between $R_{0}$ and $L_{0}$ modes. For a uniform waveguide, considering that $R_{0}$ mode is excited in the avoid-crossing region, it can be decomposed as a superposition of two hybrid modes. Because of the difference in the propagation speeds of the two hybrid modes, the output acoustic field can be either $R_{0}$ or $L_{0}$ depending on the length of the waveguide, which means the mode coupling between $R_{0}$ and $L_{0}$ is realized. Unlike the situation along the $[11\bar{2}0]$ direction, the eigenmodes show distinct properties against w for the waveguide in the $[1\bar{1}00]$ direction, as shown in Fig. 2(b). In this direction, the black and red curves for $R_{0}$ and $L_{0}$ modes could cross each other, which means that the coupling between the two modes is negligible.

To further demonstrate that the mode coupling is mainly caused by the anisotropic substrate, the phononic mode with a higher frequency of 5 GHz along the $[11\bar{2}0]$ direction is also calculated in Fig. 2(c). Because the higher frequency the acoustic wave has the shorter wave-

(a)   
![](images/18b2c6253f97a825d02f7347a4042532262fb41f99772c22f3e17df0a898c933.jpg)

![](images/70b3f36d41b78f7b3de726ee545b3d71f27a2547a755ce0a70f25993d5ea456c.jpg)

<details>
<summary>natural_image</summary>

3D rendered blue rectangular block with a white arrow pointing to its side, labeled [1120] (no other text or symbols)
</details>

(b)   
![](images/7a3adca7d7e07a79e0c3c31219c186da95a6d81aa2c1774cb8564d0b7f0859fc.jpg)

<details>
<summary>text_image</summary>

u_y
u_z
[11̅20]
</details>

(d)   
![](images/453fe4ee50d0d4209ad09d02a9d675ff33d78b186386770d6015775d0944f1b9.jpg)

<details>
<summary>text_image</summary>

d)
Mode R₀
uᵧ
-1
u_z
-1
uᵧ
-1
u_z
-1
uᵧ
0
Mode L₀
uᵧ
-1
u_z
-1
</details>

(c)   
![](images/8e3f785b64a6268dea5df17a1d8db89cedc9e04c6f33f86970c3353fa89f17cd.jpg)  
The simulated 3D displacement field distributions of the acoustic wave propagating in the taper waveguide along the section and the $y$ and $z$ components of the output displacement fields when the input field is mode $R_0$ . (a) $l_{t} = 0\mu \mathrm{m}$ . $\mu \mathrm{m}$ . (c) $l_{t} = 150\mu \mathrm{m}$ . (d) The eigenmodes of the tapered waveguide at the output port along the [1120] direction.

length, the acoustic energy of the phononic mode is mainly concentrated in the GaN layer so the influence of the substrate anisotropy on the phonon propagation becomes weak. As depicted in the curves, the gap between the black and red curves is reduced compared to that in Fig. 2(a), indicating the coupling strength between $R_{0}$ and $L_{0}$ modes in the avoid-crossing region is smaller. Furthermore, by changing the substrate as an isotropic material, the phononic modes feature no coupling, as illustrated in Fig. 2(d). These results imply possible approaches to suppress the anisotropic effect of the substrate by either changing the waveguide geometry to avoid the mode hybridization or suppressing the evanescent field in the substrate.

# III. ADIABATIC MODE CONVERSION

Compared to the mode coupling with a constant coupling strength, where a proper waveguide length should be selected to realize the highest mode conversion efficiency because of the periodic form of energy distribution along the propagation direction, adiabatic mode conversion is another widely used theory in quantum system and integrated photonic circuits. As a quantum control technique, the quantum adiabatic evolution has been extensively studied in coherent control of quantum states and quantum computation $[22–24]$ . In an integrated optical waveguide, inspired by the similarity between the evolution of the state in quantum mechanics and electromagnetic wave propagation in optics, robust and broadband mode conversion with high-efficiency has been realized by using the adiabatic theory on an integrated photonic chip [25-28]. The optical mode evolution in waveguide is determined by the parameters of the waveguide. By changing the parameters adiabatically in some situations, the adiabatic mode conversion can be realized. For a tapered waveguide with a varying waveguide width rate $dw/dx$ , according to the Landau-Zener tunneling theory [25], the achievable mode conversion efficiency is

$$
\eta = 1 - e ^ {- 2 \pi g ^ {2} / \left| \kappa \frac {d w}{d x} \right|}, \tag {1}
$$

where g is the mode coupling coefficient in the avoid-crossing regions and $\kappa = d(\beta_{r} - \beta_{l})/dw$ with $\beta_{r}$ and $\beta_{l}$ being the wave numbers of the quasi-Rayleigh and the quasi-Love modes, respectively. Therefore, when the waveguide width changes adiabatically, i.e. dw/dx is small enough, a robust, broadband, and high-efficiency mode converter can be realized.

Following the pink dash curve with arrows in Fig. 2(a), the adiabatic mode conversion is numerically investigated by simulating the propagation of the acoustic wave in a tapered phononic waveguide along the [1120] direction, as shown in Fig. 3. By fixing the input port waveguide width $w_{i} = 5\mu \mathrm{m}$ and the output port waveguide width $w_{o} = 2\mu \mathrm{m}$ , and varying the length of the tapered waveguide $l_{t}$ , the evolution of the displacement field and the corresponding $y$ and $z$ components of the output field profile for $l_{t} = 0$ , 50 and $150\mu \mathrm{m}$ are illustrated in Figs. 3(a)-(c), respectively. Here, the frequency of the excited $R_0$ mode at the input port is $1.2\mathrm{GHz}$ . Fig. 3(d) shows the $y$ and $z$ components of the displacement field of two eigenmodes ( $R_0$ and $L_0$ ) of the output port along the [1120] direction. It can be found that the output fields, when $l_{t} = 150\mu \mathrm{m}$ as shown in Fig. 3(c), are

![](images/60234be2d84fde6653b26cb5b9b3d0cd5c5c2880649fa2342af875107b679c25.jpg)

<details>
<summary>text_image</summary>

(a)
uₓ
-1
u₂
0
[1̅100]
(a)
</details>

![](images/493c3a8953355af4d802fea4531b0a90165e6cdf92c124fddf2ff56df56a09cb.jpg)

<details>
<summary>text_image</summary>

(b)
ux
-1
uZ
0
[11̄00]
</details>

![](images/872c863f7c09ee2d7371b8c645b9063a31f6fd87f4bca0b7bbfbafd570cb9cf2.jpg)

<details>
<summary>text_image</summary>

(d)
Mode R₀
uₓ 1
-1 u₂ 1
0
Mode L₀
uₓ 1
0 u₂ 1
-1
</details>

![](images/4648b858bc55ad35ba0c8e54e41a1dee6624850c74086ff1af876c0b5c340017.jpg)

<details>
<summary>text_image</summary>

(c)
ux
uz
[1̅700]
</details>

FIG. 4. The simulated 3D displacement field distributions of the acoustic wave propagating in the taper waveguides along the [1100] direction and the $x$ and $z$ components of the output displacement fields when the input field is mode $R_0$ . (a) $l_t = 0\mu \mathrm{m}$ . (b) $l_t = 50\mu \mathrm{m}$ . (c) $l_t = 150\mu \mathrm{m}$ . (d) The eigenmodes of the tapered waveguide at the output port along the [1100] direction.

very similar to those of the quasi-Love eigenmode of the output port, which indicates a nearly-perfect conversion from the quasi-Rayleigh mode to the quasi-Love mode. As expected, the converter becomes less adiabatic since $dw/dx \propto 1/l_{t}$ increases with the reduced taper length so that the conversion efficiency decreases when $l_{t} = 0 \mu m$ and $50 \mu m$ .

To verify the effect of the anisotropic substrate, we also simulate the mode evolution in the tapered waveguide along the $[1\bar{1}00]$ direction, and the results are summarized in Fig. 4. Compared with the results in Fig. 3, all the results for the $[1\bar{1}00]$ direction show that the output mode is the quasi-Rayleigh mode with z component dominated displacement field, which indicates the mode conversion almost does not occur. This behavior of the tapered waveguides is consistent with our theoretical prediction, since the substrate-induced coupling $g \approx 0$ for the waveguide along the $[1\bar{1}00]$ direction, and $e^{-2\pi g^{2}/|\kappa\frac{dw}{dx}|} \approx 1$ regardless of how adiabatic the waveguide width varies. Comparing Figs. 4(a), (b) and (c), there are still minor differences in the displacement field profile at the output port. We attribute these $l_{t}$ -dependent effects to the varying waveguide geometry-induced coupling between the guided $R_{0}$ mode to other high-order Rayleigh modes and diffused Rayleigh modes of the substrate.

Compared with $L_{0}$ mode, the $R_{0}$ mode could be more efficiently excited by the IDT in GNOS, so it is of great interest to achieve the efficient conversion from mode $R_{0}$ to mode $L_{0}$ . To quantify the performance of the converter, conversion efficiency $\eta$ is calculated numerically. Since GaN is a piezoelectric material with a relatively low piezo-coefficient and the velocity of the acoustic wave is much smaller than the speed of light, the quasi-static approximation is applicable for our model. Therefore, only the elastic field is considered to calculate the power flow P in the phononic waveguide as

$$
P = \frac {1}{2} \mathrm{Re} \int (- \boldsymbol {V} ^ {*} \cdot \tilde {\boldsymbol {T}}) \cdot d \vec {\boldsymbol {s}}, \tag {2}
$$

where V is the vector of velocity, $\tilde{T}$ is the stress field, the symbol “\*” means complex conjugate, and s is the cross-section of the phononic waveguide. Using the power flow, the mode conversion efficiency can be calculated according to the overlap integral of the displacement field between the target eigenmode at the output port and the output field. Here, the conversion efficiency from the quasi-Rayleigh mode to the quasi-Love mode can be calculated as follows

$$
\eta = \frac {\left[ \operatorname{Re} \int (\boldsymbol {V} _ {L o v e} ^ {*} \cdot \tilde {\boldsymbol {T}} _ {T a p e r} + \boldsymbol {V} _ {T a p e r} ^ {*} \cdot \tilde {\boldsymbol {T}} _ {L o v e} \right] / 2 \cdot d \vec {\boldsymbol {s}}) ^ {2}}{\operatorname{Re} \int \boldsymbol {V} _ {T a p e r} ^ {*} \cdot \tilde {\boldsymbol {T}} _ {T a p e r} \cdot d \vec {\boldsymbol {s}} \cdot \operatorname{Re} \int \boldsymbol {V} _ {L o v e} ^ {*} \cdot \tilde {\boldsymbol {T}} _ {L o v e} \cdot d \vec {\boldsymbol {s}}},
$$

where the stress fields and the velocity fields are all obtained numerically, the subscript “Taper” denotes the field distribution at the output port by simulating the evolution of the acoustic waves with a $R_{0}$ mode input, and the subscript “Love” denotes the distribution for the $L_{0}$ mode of the output port.

Figure 5 plots the results of conversion efficiency for the tapered structure versus the geometry parameters. As shown in Fig. 5(a), for a fixed excitation frequency at 1.2 GHz, the mode conversion efficiency increases with the taper length $l_{t}$ , and saturates to unity when

![](images/7ed06c874213692c23fc4840e16d17cf27156779b3013b8d027857af9df81996.jpg)  
FIG. 5. (a) The conversion efficiency $\eta$ from mode $R_{0}$ to mode $L_{0}$ versus taper length at frequency 1.2 GHz (b) $\eta$ versus the frequency of phonons f with the taper length $l_{t} = 220 \mu m$ . (c)-(d) The tolerance of the device to the variation of the thickness h and the width w of the waveguide, respectively.

$l_{t} > 100 \mu m$ . Such a saturation effect confirms the adiabatic conversion mechanism as that the $\eta \approx 1$ when the dw/dz is small enough. The robustness of the adiabatic conversion mechanism makes the structure insensitive to the working frequency and also to the variations of the geometry parameters. With a fixed taper length of $l_{t} = 220 \mu m$ , it is observed that $\eta \approx 1$ in a wide range of frequencies in Fig. 5(b). The results indicate a broad working bandwidth of about 1.7 GHz for $\eta > 50\%$ in the frequency range of $0.9 \sim 2.6 GHz$ . Note that the lowest working frequency at 0.9 GHz is limited by the cut-off frequency of the waveguide at the output port, i.e. the $L_{0}$ mode does not exist when frequency is below 0.9 GHz. Besides, the robustness of the adiabatic mode converter is further investigated by adding a uniform change of thickness or width to the tapered waveguide at each position. The results are summarized in Figs. 5(c)-(d). As the waveguide thickness varies within $\pm 50 nm$ and the width of the waveguide varies within $\pm 1 \mu m$ , $\eta$ is always larger than 98% and shows great tolerance against the geometry parameter uncertainty in practical fabrications. The presented fluctuation of the conversion is mainly attributed to the numerical calculation accuracy.

# IV. CONCLUSION

In conclusion, we theoretically propose and investigate the phononic adiabatic mode converter for PnICs. In a practical PnIC platform, the conversion between the quasi-Rayleigh and the quasi-Love modes is enabled by utilizing the anisotropy of the sapphire substrate. Benefiting from the adiabatic mode evolution mechanism, a compact tapered phononic waveguide allows efficient and robust conversion. The mode conversion efficiency exceeding 98% is obtained with a waveguide length around 100 $\mu$ m. The device has a broad working bandwidth of 1.7 GHz and is insensitive to the practical fabrication uncertainties of the structure. Our work reveals the important role of the anisotropy of the crystal compared to the isotropic materials, and appeals careful design of integrated phononic devices by considering the orientation of the device. Furthermore, our work represents one example that shows the analogy between the acoustic wave propagation dynamics and the evolution of the quantum system, and thus opens a new avenue for designing useful phononic devices.

This work was supported by National Natural Science Foundation of China (Grant No. 12061131011, 11922411, 11874342, 12104441, 92165209, and 11925404), the Natural Science Foundation of Anhui Provincial (Grant No. 2108085MA17 and 2108085MA22), China Postdoctoral Science Foundation (BX2021167), and Key-Area Research and Development Program of Guangdong Province (Grant No. 2020B0303030001). This work was partially carried out at the USTC Center for Micro and Nanoscale Research and Fabrication. The numerical calculations in this paper have been done on the supercomputing system in the Supercomputing Center of University of Science and Technology of China.

\* xbxuphys@ustc.edu.cn
† clzou321@ustc.edu.cn

[1] L. Reindl, C. Ruppel, S. Berek, U. Knauer, M. Vossiek, P. Heide, and L. Oreans, “Design, fabrication, and application of precise SAW delay lines used in an FMCW radar system,” IEEE Trans. Microw. Theory Tech. 49, 787 (2001).   
[2] C. Campbell, Surface Acoustic Wave Devices for Mobile and Wireless Communications, Four-Volume Set (Academic press, 1998).   
[3] K. Länge, B. E. Rapp, and M. Rapp, “Surface acoustic wave biosensors: a review,” Anal. Bioanal. Chem. 391, 1509 (2008).   
[4] B. Liu, X. Chen, H. Cai, M. Mohammad Ali, X. Tian, L. Tao, Y. Yang, and T. Ren, “Surface acoustic wave devices for sensor applications,” J. Semicond. 37, 021001 (2016).   
[5] M. J. A. Schuetz, E. M. Kessler, G. Giedke, L. M. K. Vandersypen, M. D. Lukin, and J. I. Cirac, “Universal Quantum Transducers Based on Surface Acoustic Waves,” Phys. Rev. X. 5, 031031 (2015).   
[6] C. H. W. Barnes, J. M. Shilton, and A. M. Robinson, "Quantum computation using electrons trapped by surface acoustic waves," Phys. Rev. B. 62, 8410 (2000).   
[7] S. Hermelin, S. Takada, M. Yamamoto, S. Tarucha, A. D. Wieck, L. Saminadayar, C. Bäuerle, and T. Meunier, “Electrons surfing on a sound wave as

a platform for quantum optics with flying electrons," Nature 477, 435 (2011).   
[8] W. J. M. Naber, T. Fujisawa, H. W. Liu, and W. G. Van Der Wiel, “Surface-Acoustic-Wave-Induced Transport in a Double Quantum Dot,” Phys. Rev. Lett. 96, 136807 (2006).   
[9] J. R. Gell, M. B. Ward, R. J. Young, R. M. Stevenson, P. Atkinson, D. Anderson, G. A. C. Jones, D. A. Ritchie, and A. J. Shields, “Modulation of single quantum dot energy levels by a surface-acoustic-wave,” Appl. Phys. Lett. 93, 081115 (2008).   
[10] D. A. Golter, T. Oo, M. Amezcua, K. A. Stewart, and H. Wang, “Optomechanical Quantum Control of a Nitrogen-Vacancy Center in Diamond,” Phys. Rev. Lett. 116, 143602 (2016).   
[11] D. A. Golter, T. Oo, M. Amezcua, I. Lekavicius, K. A. Stewart, and H. Wang, “Coupling a Surface Acoustic Wave to an Electron Spin in Diamond via a Dark State,” Phys. Rev. X. 6, 041060 (2016).   
[12] M. V. Gustafsson, T. Aref, A. F. Kockum, M. K. Ekström, G. Johansson, and P. Delsing, “Propagating phonons coupled to an artificial atom,” Science 346, 207 (2014).   
[13] R. Manenti, A. F. Kockum, A. Patterson, T. Behrle, J. Rahamim, G. Tancredi, F. Nori, and P. J. Leek, "Circuit quantum acoustodynamics with surface acoustic waves," Nat. Commun. 8, 975 (2017).   
[14] Y. Chu, P. Kharel, W. H. Renninger, L. D. Burkhart, L. Frunzio, P. T. Rakich, and R. J. Schoelkopf, "Quantum acoustics with superconducting qubits," Science 358, 199 (2017).   
[15] K. J. Satzinger, Y. P. Zhong, H.-S. Chang, G. A. Peairs, A. Bienfait, M.-H. Chou, A. Y. Cleland, C. R. Conner, É. Dumur, J. Grebel, I. Gutierrez, B. H. November, R. G. Povey, S. J. Whiteley, D. D. Awschalom, D. I. Schuster, and A. N. Cleland, “Quantum control of surface acoustic-wave phonons,” Nature 563, 661 (2018).   
[16] W. Fu, Z. Shen, Y. Xu, C.-L. Zou, R. Cheng, X. Han, and H. X. Tang, “Phononic integrated circuitry and spin-orbit interaction of phonons,” Nat. Commun. 10, 2743 (2019).   
[17] W. Wang, M. Shen, C.-L. Zou, W. Fu, Z. Shen, and H. X. Tang, “High-acoustic-index-

contrast phononic circuits: Numerical modeling," J. Appl. Phys. 128, 184503 (2020).   
[18] F. M. Mayor, W. Jiang, C. J. Sarabalis, T. P. McKenna, J. D. Witmer, and A. H. Safavi-Naeini, “Gigahertz Phononic Integrated Circuits on Thin-Film Lithium Niobate on Sapphire,” Phys. Rev. Appl. 15, 014039 (2021).   
[19] L. Shao, D. Zhu, M. Colangelo, D. H. Lee, N. Sinclair, Y. Hu, P. T. Rakich, K. Lai, K. K. Berggren, and M. Loncar, “Electrical Control of Surface Acoustic Waves,” arXiv:2101.01626 (2021).   
[20] B. E. A. Saleh and M. C. Teich, Fundamentals of Photonics, Wiley Series in Pure and Applied Optics (John Wiley & Sons, Inc., New York, USA, 1991).   
[21] J. Pedrós, F. Calle, J. Grajal, R. J. Jiménez Riobóo, Y. Takagaki, K. H. Ploog, and Z. Bougrioua, “Anisotropy-induced polarization mixture of surface acoustic waves in GaNc-sapphire heterostructures,” Phys. Rev. B. 72, 075306 (2005).   
[22] E. Farhi, J. Goldstone, S. Gutmann, and M. Sipser, "Quantum Computation by Adiabatic Evolution," arXiv:quant-ph/0001106 (2000).   
[23] A. M. Childs, E. Farhi, and J. Preskill, “Robustness of adiabatic quantum computation,” Phys. Rev. A. 65, 012322 (2001).   
[24] B. B. Zhou, A. Baksic, H. Ribeiro, C. G. Yale, F. J. Heremans, P. C. Jerger, A. Auer, G. Burkard, A. A. Clerk, and D. D. Awschalom, “Accelerated quantum control using superadiabatic dynamics in a solid-state lambda system,” Nat. Phys. 13, 330 (2017).   
[25] X.-B. Xu, L. Shi, G.-C. Guo, C.-H. Dong, and C.-L. Zou, “‘Möbius’ microring resonator,” Appl. Phys. Lett. 114, 101106 (2019).   
[26] X.-B. Xu, X. Guo, W. Chen, H. X. Tang, C.-H. Dong, G.-C. Guo, and C.-L. Zou, “Flat-top optical filter via the adiabatic evolution of light in an asymmetric coupler,” Phys. Rev. A. 100, 023809 (2019).   
[27] S.-H. Kim, R. Takei, Y. Shoji, and T. Mizumoto, "Single-trench waveguide TE-TM mode converter," Opt. Express. 17, 11267 (2009).   
[28] D. Dai, Y. Tang, and J. E. Bowers, “Mode conversion in tapered submicron silicon ridge optical waveguides,” Opt. Express. 20, 13425 (2012).