# SCIENTIFIC REPORTS

OPEN

Received: 19 October 2017

Accepted: 13 November 2017

Published online: 30 November 2017

# High-efficiency grating-couplers: demonstration of a new design strategy

Riccardo Marchetti $^{1}$ , Cosimo Lacava $^{2}$ , Ali Khokhar $^{2}$ , Xia Chen $^{2}$ , Ilaria Cristiani $^{1}$ , David J. Richardson $^{2}$ , Graham T. Reed $^{2}$ , Periklis Petropoulos $^{2}$ & Paolo Minzioni $^{1}$

We present a simple and practical strategy that allows to design high-efficiency grating couplers. The technique is based on the simultaneous apodization of two structural parameters: the grating period and the fill-factor, along with the optimization of the grating coupler etching depth. Considering a 260 nm Si-thick Silicon-on-insulator platform, we numerically demonstrated a coupling efficiency of -0.8 dB (83%), well matching the experimental value of -0.9 dB (81%). Thanks to the optimized design, these results represent the best performance ever reported in the literature for SOI structures without the use of any back-reflector.

In recent years silicon photonics has established itself as a mature technology for the development of integrated and low cost optical devices for telecom applications $^{1}$ . Various optical components such as modulators $^{2}$ , optical filters $^{3,4}$ , photodetectors $^{5}$ , arrayed waveguide gratings (AWGs) $^{6}$ , and all-optical wavelength converters $^{7}$ have successfully been demonstrated, showing that multiple functions can be effectively integrated in a single integrated photonic chip $^{8}$ . Considering Silicon-on-Insulator (SOI) technology, the strong refractive index difference between Si and $SiO_{2}$ allows the implementation of very small, single mode waveguide structures $^{9}$ . On one hand this enables dense on-chip device integration, but on the other side it makes efficient fiber-to-waveguide coupling particularly challenging because of the vast difference in the mode areas ( $\approx80\ \mu m^{2}$ for fibers and $<0.2\ \mu m^{2}$ for SOI waveguides). Among the various proposed solutions, two different approaches have attracted most attention: edge-coupling and grating-coupling. By exploiting edge-coupling, insertion losses lower than 0.5 dB can be achieved over a broad bandwidth ( $>100\ nm$ ) $^{10}$ . Nevertheless, this technique requires complicated post-fabrication processes, such as high quality facet polishing, and high-resolution optical alignment, making it not suitable for wafer-level testing and high-volume manufacturing. Grating couplers (GC), on the other hand, represent a very attractive solution, since they can be placed anywhere on the chip, allowing simple wafer-scale automated testing $^{1}$ and providing much wider alignment tolerance than that allowed by edge-coupling. The main drawbacks of standard uniform GC are their narrow bandwidth (usually between 30–40 nm, less than half of edge-coupling solutions) and the relatively low coupling efficiency (CE), which is usually lower than 61% $^{11}$ . The relatively low CE can be attributed to two main factors: grating directionality and mode mismatch. Considering a beam coupling from the SOI waveguide to a fiber, a low grating directionality ( $\approx65\%$ ) means that a large percentage of the optical power incident on the GC is diffracted towards the substrate ( $\approx35\%$ ) instead of being sent towards the fiber. The mode mismatch limitation is due to the fact that the optical intensity profile radiated by a uniform grating shows an almost exponential-decay shape with a reduced overlap integral with the mode profile (almost Gaussian) of standard optical fibers $^{11}$ . Different solutions have been proposed in the literature to increase the directionality by reducing the power leakage to the substrate. They make use of different approaches, such as the use of either poly-Silicon over-layers $^{12,13}$ or back-reflectors embedded in the substrate, e.g. distributed Bragg reflectors (DBRs) $^{11,14}$ and metallic mirrors $^{15-17}$ . All of these techniques enhance the GC directionality, but require additional fabrication steps, that sometimes even involve the use of non-CMOS-compatible materials. CE can also be increased by tailoring the amount of optical power scattered by each element of the grating, so as to reduce the mode mismatch between the radiated field profile and the optical mode in a single-mode fiber (SMF). This is generally achieved by varying either the etching depth $^{18}$ or the fill-factor of each grating period $^{11,12,16,17,19-22}$ . Alternatively, the hole-size in fully-etched photonic crystals (PhC) may be varied along the grating, however this often requires more complex fabrication processes $^{17,23}$ . Apodized GC are generally designed by applying

$^{1}$ University of Pavia, Department of Electrical Engineering and Computer Science, Pavia, IT27100, Italy. $^{2}$ University of Southampton, Optoelectronics Research Centre, Southampton, SO17 1BJ, United Kingdom. Correspondence and requests for materials should be addressed to C.L. (email: C.Lacava@soton.ac.uk)

![](images/8e300a16c74d18f302cbd1218494900a9d46c2b2dcfa3e817c1d1d0ecb3309e2.jpg)

<details>
<summary>text_image</summary>

x
z
z = 0
Fiber offset
d/θair
L_E L_O
SiO2 TOX
h
Waveguide power monitor
Λ
Si
SiO2 BOX
Si substrate
</details>

![](images/eec15072ee91175050af649926901e03b28529462ac9bb5893cd7be0048971b6.jpg)

<details>
<summary>text_image</summary>

z = 0
Λ_i ≠ Λ_{i+1}
Λ_i \cdot (n_{eff,i} - n_c \cdot \sin\theta_c) = cost
Λ_i
Λ = cost
Λ
</details>

Figure 1. Left: Cross-sectional schematic and simulation layout of a non-uniform GC in a SOI wafer, based on a linear apodization of the grating F. Right: Cross-sectional schematic of the proposed linearly apodized GC (top) compared to that of a linearly apodized grating with fixed period $\Lambda$ (bottom).

numerical techniques, such as genetic algorithms (GA), to optimize a simple starting-structure. These techniques, in addition to being computationally intensive and time-consuming, do not allow getting a real physical insight on why a particular apodization curve yields to a CE increase. In this paper, we start by discussing the physical principle underlying the design of apodized GC with optimal CE in Section 2 and then describe the fabrication procedure and experimental characterization results in Section 3. Counter-intuitively, the obtained GC achieves higher CE than GA-optimized designs reported in the literature that do not use back-reflectors and the reason for this will be clear by the end of the paper.

# Grating design and simulation

In standard SOI uniform GC, trenches having a length $L_{E}$ and depth e are etched in the silicon layer with a periodicity $\Lambda$ . If we define the grating fill-factor F as the ratio between the length of the un-etched section $L_{O}$ to the total length $\Lambda$ of the radiative unit (see Fig. 1(left)), we can express the effective index of the grating $n_{eff}$ as

$$
n _ {e f f} = F \cdot n _ {O} + (1 - F) \cdot n _ {E} \tag {1}
$$

where $n_{O}$ and $n_{E}$ are the effective indices of the original silicon slab and the etched areas, respectively. The periodic refractive index change between the trenches and the un-etched teeth, allows the optical mode propagating in the silicon waveguide to be diffracted to free space. According to the first order Bragg condition, the grating periodicity $\Lambda$ is given by

$$
\Lambda = \frac {\lambda_ {c}}{\left(n _ {e f f} - \sin \theta_ {a i r}\right)} \tag {2}
$$

where $\lambda_{c}$ is the coupling-wavelength, $\theta_{air}$ is the diffraction angle, and $n_{eff}$ is the effective refractive index of the radiative unit, as shown in Fig. 1(left). Starting from the basic structure of a uniform GC, we decided to introduce a linear apodization on the grating to improve the CE. It is well known that by linearly varying the F value along the grating, two positive effects can be achieved: the amount of optical power radiated by the first elements of the grating is reduced, and the optical impedance matching between the waveguide and the grating section is improved $^{20}$ .

The equation used to apodize the grating can then be expressed as

$$
F = F _ {0} - R \cdot z \tag {3}
$$

where $F_{0}$ is the initial fill-factor of the first radiative unit, R is the linear apodization factor and z is the distance of each radiative unit from the starting point of the grating. A cross-sectional schematic of the grating is shown in Fig. 1(left). It is important to highlight that Eq. 3 describes the evolution of F along the grating, but does not impose any limitation on $\Lambda$ , which still remains a free parameter. In the majority of the works reported in the literature that discuss linearly-apodized GC, $\Lambda$ is considered to be constant along the whole grating length $^{20-22}$ . This assumption prevents from simultaneously satisfying the Bragg condition by all of the grating elements: as F is varied along the grating, the effective index $n_{eff}$ of each radiative unit changes, and thus the Bragg condition is satisfied only at a specific point of the grating but not along the whole structure. To alleviate this, we decided to design a GC with a linear apodization of F and a value of $\Lambda$ that was varied along the structure, so as to satisfy the Bragg condition along the whole grating. To calculate the effective index $n_{eff}$ of each radiative unit, according to the chosen level of etch depth e, Eq. 1 was used. The values of $n_{O}$ and $n_{E}$ were extracted from mode simulations of slab waveguides having height respectively equal to h and h - e, these two quantities representing the thickness of the Si layer of the wafer, for the non-etched (teeth) and etched (trenches) areas respectively. Consequently, while $n_{O}$ is a constant value, $n_{E}$ depends on the etch-depth e, and consequently also $n_{eff}$ is a function of e. Using the obtained values of $n_{eff}$ and taking advantage of Eq. 2 we calculated the length of the teeth ( $L_{O}$ ) and trenches ( $L_{E}$ ) of each radiative unit to optimize the CE for a beam with $\lambda = 1550$ nm and an angle of incidence $\theta_{air}$ of 14.5°. A graphical representation of the difference between our linear apodization scheme and the standard approach based on fixed $\Lambda$ , is given in Fig. 1(right). It is worth noting that using the proposed approach, and considering the standard situation where the thickness of the Si layer (h) and BOX layers are imposed by the available SOI wafers, only two design parameters need to be optimized: the etch depth e and the apodization factor R; this is rather different from the case of GA-optimized techniques, which specify the lengths of the etched and un-etched

![](images/500789116ca1ea86bd7d1ea432aa3cb5f6cb2c4aabe3c844f7d8029ec8bb4caf.jpg)

<details>
<summary>heatmap</summary>

| R [μm⁻¹] | Etch Depth [nm] | Coupling Efficiency % at 1550 nm |
| -------- | --------------- | --------------------------------- |
| 0.01     | 100             | 45                                |
| 0.01     | 110             | 50                                |
| 0.01     | 120             | 55                                |
| 0.01     | 130             | 60                                |
| 0.01     | 140             | 65                                |
| 0.01     | 150             | 70                                |
| 0.01     | 160             | 75                                |
| 0.01     | 170             | 80                                |
| 0.01     | 180             | 75                                |
| 0.01     | 190             | 70                                |
| 0.01     | 200             | 65                                |
| 0.01     | 210             | 60                                |
| 0.01     | 220             | 55                                |
| 0.01     | 230             | 50                                |
| 0.02     | 100             | 45                                |
| 0.02     | 110             | 50                                |
| 0.02     | 120             | 55                                |
| 0.02     | 130             | 60                                |
| 0.02     | 140             | 65                                |
| 0.02     | 150             | 70                                |
| 0.02     | 160             | 75                                |
| 0.02     | 170             | 80                                |
| 0.02     | 180             | 75                                |
| 0.02     | 190             | 70                                |
| 0.02     | 200             | 65                                |
| 0.02     | 210             | 60                                |
| 0.02     | 220             | 55                                |
| 0.02     | 230             | 50                                |
| 0.03     | 100             | 45                                |
| 0.03     | 110             | 50                                |
| 0.03     | 120             | 55                                |
| 0.03     | 130             | 60                                |
| 0.03     | 140             | 65                                |
| 0.03     | 150             | 70                                |
| 0.03     | 160             | 75                                |
| 0.03     | 170             | 80                                |
| 0.03     | 180             | 75                                |
| 0.03     | 190             | 70                                |
| 0.03     | 200             | 65                                |
| 0.03     | 210             | 60                                |
| 0.03     | 220             | 55                                |
| 0.03     | 230             | 50                                |
| 0.04     | 100             | 45                                |
| 0.04     | 110             | 50                                |
| 0.04     | 120             | 55                                |
| 0.04     | 130             | 60                                |
| 0.04     | 140             | 65                                |
| 0.04     | 150             | 70                                |
| 0.04     | 160             | 75                                |
| 0.04     | 170             | 80                                |
| 0.04     | 180             | 75                                |
| 0.04     | 190             | 70                                |
| 0.04     | 200             | 65                                |
| 0.04     | 210             | 60                                |
| 0.04     | 220             | 55                                |
| 0.04     | 230             | 50                                |
| 0.05     | 100             | 45                                |
| 0.05     | 110             | 50                                |
| 0.05     | 120             | 55                                |
| 0.05     | 130             | 60                                |
| 0.05     | 140             | 65                                |
| 0.05     | 150             | 70                                |
| 0.05     | 160             | 75                                |
| 0.05     | 170             | 80                                |
| 0.05     | 180             | 75                                |
| 0.05     | 190             | 70                                |
| 0.05     | 200             | 65                                |
| 0.05     | 210             | 60                                |
| 0.05     | 220             | 55                                |
| 0.05     | 230             | 50                                |
</details>

![](images/212ab3e400e52b27e1f2c44f0af2ab7dbe7c99d27cc421a664a6866dc2d4668d.jpg)

<details>
<summary>line</summary>

| Etch Depth [nm] | Si 260 nm | Si 220 nm |
| --------------- | --------- | --------- |
| 60              | -         | 58        |
| 80              | -         | 67        |
| 100             | 73        | 69        |
| 120             | 79        | 70        |
| 140             | 82        | 68        |
| 160             | 83        | 64        |
| 180             | 83        | 55        |
| 200             | 81        | -         |
| 220             | 77        | -         |
| 240             | 68        | -         |
</details>

Figure 2. Left: Contour plot of the peak CE at $\lambda=1550$ nm of the linearly apodized GC realized in 260 nm SOI platform, as a function of etch depth and of the linear apodization factor R. Right: maximum CE at the central wavelength of 1550 nm for the linearly apodized GC based on 220 nm SOI (red plot) and 260 nm (blue plot).

portions of each radiative unit. Concerning the choice of the initial fill-factor $F_{0}$ in our design, it has been shown that reducing $F_{0}$ results in an increase in the grating CE $^{22}$ . Unfortunately, the value producing the highest CE ( $F_{0}=0.95^{22}$ ) implies a minimum feature size of about 60 nm, which is not achievable with fabrication processes available to us. We therefore set $F_{0}$ to 0.9, which corresponded to a value of $L_{E}=60$ nm for the first trench. Full vectorial 2D-FDTD simulations using FDTD Solutions $^{TM}$ (from Lumerical Inc.) were carried out in order to find the optimum values of the e and R parameters for the structure shown in Fig. 1(right). The details of the numerical simulations are reported in the Methods section. We took into consideration two different types of SOI wafer, both having a 2- $\mu$ m-thick buried SiO $_{2}$ layer (BOX, $n_{ox}=1.44$ at 1550 nm) and having a Si-layer ( $n_{Si}=3.48$ at 1550 nm) of height h=220 nm and h=260 nm respectively. A top SiO $_{2}$ layer (TOX, $n_{ox}=1.44$ ) was included in the layout, having a height of 720 nm for the 220 nm SOI wafer, and 680 nm for the 260 nm SOI wafer. Trenches were assumed to be completely filled by the TOX and no index-matching liquid was considered between the TOX and the fiber.

We evaluated the CE by considering the grating as an in-coupling device, i.e. coupling light from a SMF into the SOI waveguide by means of the GC. The fiber, which was assumed to have an outer diameter of $125 \mu m$ and a mode field diameter (MFD) of $10.4 \mu m$ , was tilted by $\theta_{air} = 14.5^{\circ}$ with respect to the vertical direction (corresponding to an incidence angle on the grating of $\theta_{c} = 10^{\circ}$ in the TOX), causing the Gaussian beam output from the fiber to impinge on the TOX with a MFD of about $10.8 \mu m$ . The selected angle of incidence helped in reducing back-reflections into the launch-fibre and, at the same time, provided directionality to the light coupled into the waveguide taper. The electric field of the Gaussian beam was polarized along the $\hat{x}$ direction with reference to Fig. 1(left), so that the incoming light could be coupled to the fundamental TE mode of the integrated waveguide. A frequency-domain power monitor, positioned along the Si waveguide at a $10 \mu m$ distance from the grating, was used to measure the amount of optical power coupled to the waveguide. The CE of different grating configurations was calculated while simultaneously sweeping both the etch depth e, by 10 nm steps, and the apodization factor R, by $0.0025 \mu m^{-1}$ steps. For every grating configuration, the offset of the fiber mode from the starting point of the grating was also optimized. To illustrate an example of the simulation results, we plot in Fig. 2(left) a contour diagram showing the peak CE at 1550 nm for a 260-nm thick Si-layer as a function of the parameters e and R. It can be noticed that, starting from an etch depth of 100 nm, the maximum CE is found to be 73% (-1.4 dB) when the apodization factor R is equal to $0.0425 \mu m^{-1}$ .

Increasing the etch depth while reducing the factor R, yields a large CE improvement, reaching a maximum of 83% ( $-0.8\,dB$ ) for e=160 nm and $R=0.025\,\mu m^{-1}$ . If the etch depth is further increased, the CE starts to decrease, with a value of 68% ( $-1.7\,dB$ ) when e=230 nm and $R=0.025\,\mu m^{-1}$ . It is also interesting to note that the simulated CE is higher than 80% for a wide range of $(e,R)$ combinations, thus showing a good tolerance to small variations in the fabrication process. The same procedure used for the design of the apodized GC based on the 260 nm SOI platform, was also applied to the 220 nm SOI grating, achieving a maximum CE of 70% ( $-1.6\,dB$ ) when e=110 nm and $R=0.0275\,\mu m^{-1}$ . The maximum CE achieved at the central wavelength of 1550 nm, as a function of the etch-depth, for each of the two SOI platforms is reported in Fig. 2(right). It can be seen that by using a SOI platform with a thicker Si layer it is possible to obtain a higher CE, as already reported in other works $^{20}$ , and that deep etching levels (110 nm and 160 nm for the 220 nm and 260 nm SOI platforms, respectively) are required to obtain the maximum CE.

To complete the GC analysis we also simulated the grating as an out-coupling device. We set a fundamental-mode waveguide source in the Si waveguide, a power-monitor above the GC to assess its directionality and an additional power-monitor in the Si-layer to assess grating reflectivity by measuring the optical power reflected back in the waveguide, as shown in Fig. 3.

The analysis was carried out for the 260 nm Si-thick SOI platform, comparing the performances of our apodized design at $\lambda=1550$ nm to that of uniform gratings (F=0.5) realized in the same platform and having the same TOX height. Directionality results are shown, as a function of etch depth e, in Fig. 4(left). It can be noted

![](images/df23df802f65e2d6e0ecf9ea84f9cf1a2c2494160492a3115779ccc31b5d85d0.jpg)

<details>
<summary>text_image</summary>

z = 0
Directionality Monitor
SiO₂ TOX
Si
Reflectivity
Monitor
Waveguide Source
SiO₂ BOX
Si substrate
</details>

Figure 3. Cross-sectional schematic and simulation layout used to assess the directionality and reflectivity of uniforms and apodized GC.

![](images/3b5a17c6fe03495fa3e2720afa30ff1e5e4358372a6b79028bfa17a0311a0763.jpg)

<details>
<summary>line</summary>

| Etch Depth [nm] | 260 nm SOI apodized GC | 260 nm SOI uniform GC |
| --------------- | ---------------------- | ---------------------- |
| 100             | 75                     | 74                     |
| 120             | 80                     | 75                     |
| 140             | 84                     | 74                     |
| 160             | 85                     | 73                     |
| 180             | 85                     | 72                     |
| 200             | 83                     | 70                     |
| 220             | 78                     | 65                     |
| 240             | 70                     | 53                     |
</details>

![](images/004f0ba696ce84ec8eef2a85aa7723f703a48d21a46dc9d6f0521bdf7624e01d.jpg)

<details>
<summary>line</summary>

| Etch Depth [nm] | 260 nm SOI apodized GC | 260 nm SOI uniform GC |
| --------------- | ---------------------- | --------------------- |
| 100             | 0.8                    | 1.0                   |
| 110             | 0.9                    | 1.7                   |
| 120             | 1.0                    | 2.8                   |
| 130             | 1.1                    | 3.9                   |
| 140             | 1.3                    | 5.1                   |
| 150             | 1.5                    | 6.4                   |
| 160             | 1.8                    | 6.9                   |
| 170             | 2.2                    | 5.8                   |
| 180             | 2.4                    | 5.1                   |
| 190             | 2.6                    | 4.9                   |
| 200             | 3.0                    | 5.4                   |
| 210             | 3.3                    | 6.5                   |
| 220             | 3.5                    | 7.7                   |
| 230             | 3.5                    | 8.6                   |
</details>

Figure 4. GC directionality (left) and reflectivity (right) at $\lambda = 1550\mathrm{nm}$ as a function of $e$ for the proposed linearly-apodized GC (green curve) and for a uniform GC (purple curve). The arrows in the left panel indicate the maximum directionality achievable in each structure.

<table><tr><td>Si [nm]</td><td>Description</td><td> $\text{CE}_{\text{T}}$  [dB]</td><td> $\text{CE}_{\text{E}}$  [dB]</td><td>Ref.</td><td>Si [nm]</td><td>Description</td><td> $\text{CE}_{\text{T}}$  [dB]</td><td> $\text{CE}_{\text{E}}$  [dB]</td><td>Ref.</td></tr><tr><td>220</td><td>GA</td><td>-2.15</td><td>—</td><td>11</td><td>250</td><td>fully-etched PhC</td><td>-1.8</td><td>-1.74</td><td>23</td></tr><tr><td>220</td><td>poly-Si overlay</td><td>-1.08</td><td>—</td><td>12</td><td>250</td><td>lag effect in etching</td><td>-1.31</td><td>-1.9</td><td>18</td></tr><tr><td>220</td><td>poly-Si overlay</td><td>—</td><td>-1.6</td><td>13</td><td>250</td><td>linear apodization</td><td>-2.3</td><td>-2.7</td><td>22</td></tr><tr><td>220</td><td>linear apodization</td><td>-2.6</td><td>-2.7</td><td>21</td><td>250</td><td>BR: Aluminum</td><td>-0.26</td><td>-0.62</td><td>16</td></tr><tr><td>220</td><td>GA</td><td>-1.9</td><td>—</td><td>20</td><td>250</td><td>BR: Aluminum</td><td>-0.43</td><td>-0.58</td><td>17</td></tr><tr><td>220</td><td>BR: Gold</td><td>-1.43</td><td>-1.61</td><td>15</td><td>260</td><td>GA</td><td>-1.0</td><td>—</td><td>20</td></tr><tr><td>220</td><td>BR: DBR</td><td>-0.36</td><td>—</td><td>11</td><td>260</td><td>linear apodization</td><td>-0.8</td><td>-0.9</td><td>*</td></tr><tr><td>220</td><td>BR: DBR</td><td>-0.86</td><td>-1.58</td><td>14</td><td>340</td><td>200-nm deep etching</td><td>-0.8</td><td>-1.2</td><td>19</td></tr><tr><td>220</td><td>linear apodization</td><td>-1.6</td><td>—</td><td>*</td><td>340</td><td>GA</td><td>-0.5</td><td>—</td><td>20</td></tr></table>

Table 1. Summary of the theoretical (CE $_T$ ) and experimental (CE $_E$ ) coupling efficiencies for different GC reported in the literature. Results are ordered with respect to the thickness of the Silicon layer. Devices that required the use of a back reflector (BR) are reported in italics, while those relating to this work are shown in bold. All the values are reported with the number of significant figures provided by the authors.

from the figure that the maximum directionality achievable by the apodized GC is equal to 85%, almost 10% better than the maximum directionality achievable with the uniform design. Since TOX and BOX values were kept constant, this shows that directionality is significantly improved by the adoption of the proposed apodization technique. It is also worth noting, that the optimum value of the etching depth is different when a uniform GC is considered rather than the apodized one (see Fig. 4(left)). This demonstrates that, in order to design the optimum GC, both of the apodization profile and the etching depth must be simultaneously optimized. This aspect was not considered in previous works such as in $^{19}$ , leading to a reduced directionality and CE.

We report the reflectivity for both uniform and apodized GC as a function of the etch-depth in Fig. 4(right). The proposed design strategy allows a significant reduction in the GC reflectivity with respect to the uniform structure. Moreover, it also suggests that the big difference in the identification of the optimal etch-depth between uniform and apodized GC could be related to the large reflectivity increase observed in the uniform GC around $160\mathrm{nm}$ .

A comparison of our result to those obtained in previous works (see Table 1) shows that our design strategy allowed for the first time a CE better than -1 dB (considering a Si thickness <340 nm and without using a back-reflector). A direct comparison with $^{21}$ and $^{22}$ , in which constant- $\Lambda$ linear-apodization gratings were

![](images/bbf369466eb85b7d118691be9a9b16220935640386e3fc532f3451cb533b272c.jpg)

<details>
<summary>natural_image</summary>

Microscopic view of a rectangular device with a 12 μm scale indicator, showing internal layered structure (no text or symbols beyond the scale)
</details>

![](images/892fdf507355592e8e898c9e0726d0d502551fb551d79a110bcc347ece9e3292.jpg)

<details>
<summary>line</summary>

| Wavelength [μm] | 2D-FDTD Coupling Efficiency [dB] | Experimental Data Coupling Efficiency [dB] |
| --------------- | -------------------------------- | ------------------------------------------ |
| 1.50            | -10.0                            | -6.0                                       |
| 1.52            | -8.0                             | -5.0                                       |
| 1.54            | -6.0                             | -4.0                                       |
| 1.56            | -4.0                             | -3.0                                       |
| 1.58            | -2.0                             | -2.0                                       |
| 1.60            | -10.0                            | -6.0                                       |
</details>

Figure 5. Left: Optical micrograph of one of the apodized GC. Right: CE of the best-performing grating (red trace) together with the simulated CE.

employed, shows that our design approach achieves a CE improvement of at least 1 dB. Furthermore, our results also show a non-negligible CE improvement with respect to results reported in $^{20}$ , where a constant- $\Lambda$ linear-apodization design was used as the starting configuration for a GA based refinement, thus highlighting the importance of allowing $\Lambda$ to vary.

We also applied our design procedure to the case of a SOI wafer with 340 nm-thickness Si-layer (as in $^{19}$ ), obtaining a CE of 85% (-0.7 dB), for e = 210 nm and R = 0.0425. The obtained CE is slightly higher than that reported in $^{19}$ where a fill-factor (F) apodization suitable to produce a Gaussian-shaped beam profile (CE = 83%; -0.8 dB) was considered. The improvement achieved using the proposed design strategy can be explained by considering two aspects: i) only minor differences are present between the F apodization curve reported in $^{19}$ and a linear apodization and ii) identifying the optimal etch depth and the single-element scattering coefficient in uniform GC does not guarantee obtaining the optimal CE, as previously shown (see Fig. 4). In comparison to results obtained by application of GA, it is important to stress that GA generally require a significant computational effort. For this reason different parameters (such as etch-depth or the grating period) are generally considered as fixed, thus allowing to reduce the computation time but, simultaneously, limiting the explored solutions-range. Using the proposed design approach, we limited the explored design-set to linearly apodized GC, focusing our optimization efforts on three parameters, i.e. period (thanks to Eq. 2), etch depth and fill-factor apodization, which significantly affect the GC-CE, as discussed above. This eventually led to the optimal linearly apodized GC design of this work.

# Results

We then fabricated and characterized (see Methods section) the grating-coupler structure identified as optimal by the previous analysis. One of the measured CE curves as a function of wavelength is shown in Fig. 5(right), together with the 2D-FDTD simulation result.

We believe that the parasitic amplitude ripple affecting the grating transmission spectrum is caused by the transition between the taper and the single-mode waveguide. The period of the ripple is indeed compatible with a cavity having the same length as the taper section. We carried out specific simulations in order to ensure that the interaction between the non-ideal taper and the grating would not lead to an unexpected spectral redistribution of the transmission efficiency. It is in fact important to clarify if the CE achievable with a proper taper would correspond to the maximum or the average values of the observed ripple. To investigate this aspect, we performed 2D-FDTD simulations of the apodized GC, inserting a $100\mathrm{nm}$ wide trench in the Si waveguide, at a distance of $500~{\mu\mathrm{m}}$ from the end of the grating section. The transmitted optical power was assessed by a frequency-domain power monitor, positioned along the Si waveguide $10~{\mu\mathrm{m}}$ after the trench. By varying the trench depth we simulated different amounts of back-reflection at the taper-waveguide interface, and obtained the results reported in Fig. 6: the blue, yellow and purple curves show the transmission function obtained by considering an etch-depth of the trench equal to $0\%$ , $40\%$ and $60\%$ of the Si-layer, respectively. As expected, increasing the etch-depth leads to an increasing amplitude of the ripple and to a reduction of the maximum CE. It is interesting to notice that the peaks of the ripple never exceed the threshold set by the CE of the unperturbed grating ( $0\%$ etch, blue line), meaning that the maximum CE of the analyzed sample can be conveniently taken to correspond to one of the maxima of the parasitic oscillation.

We performed measurements on several different structures, from four different chips, obtaining an average CE of -1.1 dB, and a maximum CE of -0.9 dB. A summary of the measured coupling efficiencies for all of the analyzed devices is given in Table 2. These results show a very good agreement with the theoretical CE values (-0.8 dB, i.e. 83%), and indicate a good consistency in the fabrication process. The discrepancy between the simulated -1 dB bandwidth of 32.8 nm and the experimentally derived one, with an average value of 38.8 nm and a maximum value of 39.8, can be explained by small fabrication imperfections and variations over the wafer, such as variations in the actual etch depth or in the uniformity of the deposited TOX layer.

It is worth highlighting that, as reported in Table 1, the experimentally measured CE is better than that reported in $^{21}$ and $^{22}$ (-2.7 dB), where a linear grating apodization was used, and even better than the value (-1.2 dB) obtained by using a fill-factor apodization in a 340 nm Si-thick SOI wafer $^{19}$ . The relatively large

![](images/2b92c8b6466dd3341cdf257289da2c0e8ba5d4366c17c20f3a1309bb468210fa.jpg)

<details>
<summary>text_image</summary>

100 nm
trench
10 µm
CE monitor
500 µm long WG
Grating section
</details>

![](images/e049db3aafe3cb7803c7810d8c2c2b470f82049744223e6c506d02e73de801fb.jpg)

<details>
<summary>line</summary>

| Wavelength [µm] | 0%     | 40%    | 60%    |
| --------------- | ------ | ------ | ------ |
| 1.52            | -4.0   | -4.0   | -4.0   |
| 1.53            | -2.5   | -2.7   | -2.9   |
| 1.54            | -1.5   | -1.7   | -1.9   |
| 1.55            | -0.8   | -1.0   | -1.2   |
| 1.56            | -1.5   | -1.7   | -1.9   |
| 1.57            | -2.5   | -2.7   | -2.9   |
| 1.58            | -4.0   | -4.0   | -4.0   |
</details>

Figure 6. Left: Cross-sectional schematic and simulation layout (not in scale) used to evaluate the impact of a back-reflecting element (the 100-nm wide trench) on the measured CE. Right: 2D-FDTD simulation of the optimum grating transmission, when a 100 nm wide hole is inserted in the Si waveguide at a distance of 500 $\mu$ m from the end of the grating section, at different etching percentage. The blue curve represents the unperturbed case, while the yellow and purple curves correspond to hole etching percentages of respectively 40% and 60%.

<table><tr><td rowspan="2">Struct.</td><td colspan="2">Chip 1</td><td colspan="2">Chip 2</td><td colspan="2">Chip 3</td><td colspan="2">Chip 4</td></tr><tr><td>C.E. [dB]</td><td>BW. [nm]</td><td>C.E. [dB]</td><td>BW. [nm]</td><td>C.E. [dB]</td><td>BW. [nm]</td><td>C.E. [dB]</td><td>BW. [nm]</td></tr><tr><td>1</td><td>1.0</td><td>38.4</td><td>1.0</td><td>39.8</td><td>1.4</td><td>37.8</td><td>1.2</td><td>39.4</td></tr><tr><td>2</td><td>1.0</td><td>38.8</td><td>0.9</td><td>37.4</td><td>1.4</td><td>39.8</td><td>1.3</td><td>39.3</td></tr></table>

Table 2. Summary of the fabricated gratings coupling efficiency and -1 dB bandwidth, and waveguide propagation loss, for all of the measured chips.

deviation between theoretical and experimental data in $^{19}$ can probably be associated to the high aspect ratio of the required trenches (>4.5 compared to 2.7 in the proposed structure), whose proper fabrication can be quite challenging.

Our result is also quite close the record value of -0.62 dB reported in $^{16}$ , which was achieved using an Al back-reflector in a 250 nm Si-thick SOI wafer, thus proving that that a very high CE, below the -1 dB threshold, can be obtained even without the use of any back-reflector.

# Conclusion

In this paper, we described a new design strategy for apodized diffractive gratings yielding highly efficient optical coupling between SOI waveguides and SMF. Thanks to a simultaneous apodization of both the grating fill-factor and the period along the optimization of the GC etching depth, we demonstrated a theoretical CE up to 70% ( $-1.6\,dB$ ) when using a 220 nm Si-thick SOI platform, and up to 83% ( $-0.8\,dB$ ) when a using a 260 nm Si-thick SOI platform. Experimental characterization of different samples, fabricated on a 260-nm Si-thick SOI wafer by E-beam lithography and a single-etch process, showed that thanks to the optimized design, an average CE of -1.1 dB and average -1 dB bandwidth of 38.8 nm were achieved. The highest CE experimentally measured was -0.9 dB, which currently represents the best result ever obtained without the use of embedded back-reflectors.

# Methods

Numerical Simulations. Full vectorial 2D-FDTD simulations were carried out using FDTD Solutions $^{TM}$ (by Lumerical Inc.). The 2D computational area was set to be 33.4 $\mu$ m wide and 5.8 $\mu$ m high, while the refractive index of both Silicon and SiO $_{2}$ were calculated following the data reported by Palik: as a consequence the refractive indices at the design wavelength of 1550 nm are $n_{Si}=3.48$ and $n_{SiO_{2}}=1.44$ . The simulation grid was defined using the conformal mesh method embedded in the software and setting the mesh accuracy to 8 (the highest possible value), which corresponds to a mesh grid minimum feature of 12 nm inside the material. Each simulation required, on a Intel(R) Core(TM) i7-3930K CPU (@3.20 GHz) computer, about 216 MB of RAM (initialization and mesh: 51 MB, simulation running: 72 MB, data collection: 93 MB) and a computation time of about 50 seconds. The diffracted power was calculated using the frequency-domain power monitors, set with 500 frequency points from a minimum wavelength of 1.425 $\mu$ m to a maximum wavelength of 1.675 $\mu$ m.

Sample Fabrication. Based on simulation results, we fabricated the apodized GC design yielding the highest CE in a 260 nm Si-thick SOI platform (e = 160 nm, $F_{0} = 0.9$ , $R = 0.025 \mu m^{-1}$ ). The grating was 12 $\mu m$ wide (along the $\hat{x}$ direction of Fig. 1(left)), to properly accommodate the Gaussian mode of the fiber, and 14.847 $\mu m$ long. The length of each tooth ( $L_{O}$ ) and trench ( $L_{E}$ ) of the grating is given in Table 3. To assess the grating CEs' we designed structures composed of two GCs', connected by a straight single-mode waveguide, having width equal to 500 nm, and length equal to 5.2 mm. An additional set of 5 spiral waveguides (with lengths ranging from 6.2 mm to 26.2 mm, at 5 mm steps) was designed, so as to allow separating the contributions of grating coupling and waveguide propagation losses. The waveguides were connected to the gratings by using 500 $\mu m$ long linear tapers.

<table><tr><td>Period</td><td> $L_{E}$  (nm)</td><td> $L_{O}$  (nm)</td><td>Period</td><td> $L_{E}$  (nm)</td><td> $L_{O}$  (nm)</td><td>Period</td><td> $L_{E}$  (nm)</td><td> $L_{O}$  (nm)</td></tr><tr><td>1</td><td>60</td><td>540</td><td>9</td><td>138</td><td>484</td><td>17</td><td>225</td><td>422</td></tr><tr><td>2</td><td>69</td><td>533</td><td>10</td><td>148</td><td>477</td><td>18</td><td>237</td><td>413</td></tr><tr><td>3</td><td>79</td><td>526</td><td>11</td><td>159</td><td>469</td><td>19</td><td>249</td><td>405</td></tr><tr><td>4</td><td>88</td><td>520</td><td>12</td><td>170</td><td>461</td><td>20</td><td>261</td><td>396</td></tr><tr><td>5</td><td>98</td><td>513</td><td>13</td><td>180</td><td>454</td><td>21</td><td>273</td><td>387</td></tr><tr><td>6</td><td>108</td><td>506</td><td>14</td><td>191</td><td>446</td><td>22</td><td>286</td><td>378</td></tr><tr><td>7</td><td>118</td><td>498</td><td>15</td><td>203</td><td>438</td><td>23</td><td>298</td><td>369</td></tr><tr><td>8</td><td>128</td><td>491</td><td>16</td><td>214</td><td>430</td><td>24</td><td>311</td><td>//</td></tr></table>

Table 3. Optimal trench and tooth width obtained from the optimization of the apodized GC in $260\mathrm{nm}$ SOI platform, having $F_{0} = 0.9$ and $R = 0.025\mu \mathrm{m}^{-1}$ .

Electron Beam Lithography (E-beam) was used to define the sample pattern and Inductively-coupled plasma (ICP) dry etching defined the final structures.

A protective TOX layer was finally deposited by means of plasma enhanced chemical vapor deposition (PECVD). An optical micrograph of the fabricated apodized GC is shown in Fig. 5(left).

Experimental Characterization. The experimental characterization was carried out by means of a vertical coupling scheme, using polarization maintaining (PM) single mode fibers. The optical source was a PM external cavity laser (ECL), tunable from 1523 to 1600 nm at 5 pm steps, and a power meter was used to measure the output optical power collected from the test structures. The grating CE was obtained by subtracting the waveguide propagation loss (measured on the spiral structures) from the measured fiber-to-fiber transmission and dividing by two (thus assuming that the input and output coupling losses were equal). Propagation losses were independently measured across the four samples and were assessed to be $\alpha_{S1} = 3.1 \pm 0.12$ dB/cm, $\alpha_{S2} = 3.3 \pm 0.12$ dB/cm, $\alpha_{S3} = 3 \pm 0.12$ dB/cm and $\alpha_{S4} = 4.5 \pm 0.12$ dB/cm.

# References

1. Jalali, B. & Fathpour, S. Silicon Photonics. J. Light. Technol. 24(12), 4600–4615 (2006).   
2. Reed, G. T. et al. Recent breakthroughs in carrier depletion based silicon optical modulators. Nanophotonics 3(4–5), 229–245 (2014).   
3. Bogaerts, W. et al. Silicon microring resonators. Laser Photon. Rev. 6(1), 47-73 (2012).   
4. Marchetti, R. et al. Low-Loss Micro-resonator Filters Fabricated in Silicon by CMOS-Compatible Lithographic Techniques: Design and Characterization. MDPI Appl. Sciences 7(2), 174–185 (2017).   
5. Martinez, N. J. D. et al. High performance waveguide-coupled Ge-on-Si linear mode avalanche photodiodes. Optics Express 24(17), 19072–19081 (2016).   
6. Fu, X. & Dai, D. Ultra-small Si-nanowire-based 400 GHz-spacing 15 X 15 arrayed-waveguide grating router with microbends. Electron. Lett. 47(4), 266–268 (2011).   
7. Lacava, C., Ettabib, M. A. & Petropoulos, P. Nonlinear Silicon Photonic Signal Processing Devices for Future Optical Networks. MDPI Appl. Sciences 7(1), 103–120 (2017).   
8. Carroll, L. et al. Photonic Packaging: Transforming Silicon Photonic Integrated Circuits into Photonic Devices. MDPI Appl. Sciences 6(12), 426–446 (2016).   
9. Marchetti, R. et al. Group-velocity dispersion in SOI-based channel waveguides with reduced-height. Optics Express 25(9), 9761–9767 (2017).   
10. Papes, M. et al. Fiber-chip edge coupler with large mode size for silicon photonic wire waveguides. Optics Express 24(5), 5026–5038 (2016).   
11. Taillaert, D., Bienstman, P. & Baets, R. Compact efficient broadband grating coupler for silicon-on-insulator waveguides. Optics Letters 29(23), 2749-2751 (2004).   
12. Roelkens, G., Van Thourhout, D. & Baets, R. High efficiency Silicon-on-Insulator grating coupler based on a poly-Silicon overlay. Optics Express 14(24), 11622–11630 (2006).   
13. Vermeulen, D. et al. High-efficiency fiber-to-chip grating couplers realized using an advanced CMOS-compatible Silicon-On-Insulator platform. Optics Express 18(17), 18278–18283 (2010).   
14. S. K. Selvaraja et al. "Highly efficient grating coupler between optical fiber and silicon photonic circuit", presented at the Conference on Lasers and Electro-Optics, Baltimore, Maryland, CTuC6 (2009).   
15. Van Laere, F. et al. Compact and Highly Efficient Grating Couplers Between Optical Fiber and Nanophotonic Waveguides. J. Lightwave Technol. 25(1), 151–156 (2007).   
16. Zaoui, W. S. et al. Bridging the gap between optical fibers and silicon photonic integrated circuits. Optics Express 22(2), 1277–1286 (2014).   
17. Ding, Y., Peucheret, C., Ou, H. & Yvind, K. Fully etched apodized grating coupler on the SOI platform with -0.58 dB CE. Optics Letters 39(18), 5348–5350 (2014).   
18. Tang, Y., Wang, Z., Wosinski, L., Westergren, U. & He, S. Highly efficient nonuniform grating couplers for silicon-on-insulator nanophotonic circuits. Optics Letters 35(8), 1290–1292 (2010).   
19. Chen, X., Li, C., Fung, C. K. Y., Lo, S. M. G. & Tsang, H. K. Apodized Waveguide Grating Couplers for Efficient Coupling to Optical Fibers. IEEE Photonics Technol. Lett. 22(15), 1156–1158 (2010).   
20. Bozzola, A., Carrol, L., Gerace, D., Cristiani, I. & Andreani, L. C. Optimising apodized grating couplers in a pure SOI platform to -0.5 dB coupling efficiency. Optics Express 23(12), 16289–16304 (2015).   
21. He, L. et al. A High-Efficiency Nonuniform Grating Coupler Realized With 248-nm Optical Lithography. IEEE Photonics Technol. Lett. 25(14), 1358–1361 (2013).   
22. Lee, M. H., Jo, J. Y., Kim, D. W., Kim, Y. & Kim, K. H. Comparative Study of Uniform and Nonuniform Grating Couplers for Optimized Fiber Coupling to Silicon Waveguides. Journal of the Optical Society of Korea 20(2), 291–299 (2016).   
23. Ding, Y., Ou, H. & Peucheret, C. Ultrahigh-efficiency apodized grating coupler using fully etched photonic crystals. Optics Letters 38(15), 2732-2734 (2013).

# Acknowledgements

Graham T. Reed is a Royal Society Wolfson Research Merit Award holder. He is grateful to the Wolfson Foundation and the Royal Society for funding of the award. This work has been partially supported by EPSRC, UK through the SPFS Programme Grant http://doi.org/10.5258/SOTON/403875.

# Author Contributions

R.M. conceived the idea, performed the numerical simulations and performed the experimental measurements. C.L. fabricated the samples, substantially contributed to the numerical simulations and samples characterization and coordinated the work; A.K. and X.C. provided technical support during the fabrication steps; G.T.R. supervised the cleanroom-related activities, D.J.R. P.P. and I.C. provided steerage on the research direction; P.M. provided the overall supervision and technical leadership; R.M., C.L. and P.M. wrote the initial draft of the paper, all the authors read the document and contribute to the final text.

# Additional Information

Competing Interests: The authors declare that they have no competing interests.

Publisher's note: Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

![](images/57b930c9c65138c11c62645cbdeb154ce089db38f6993f4363db117114d4843b.jpg)

Open Access This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons license, and indicate if changes were made. The images or other third party material in this article are included in the article's Creative Commons license, unless indicated otherwise in a credit line to the material. If material is not included in the article's Creative Commons license and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.

© The Author(s) 2017