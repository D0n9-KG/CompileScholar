# OPEN

Received: 7 November 2018

Accepted: 14 August 2019

Published online: 10 September 2019

# A high efficiency silicon nitride waveguide grating coupler with a multilayer bottom reflector

Jianxun Hong $^{1,3}$ , Andrew M. Spring $^{2}$ , Feng Qiu $^{1}$ & Shiyoshi Yokoyama $^{1,2}$

We propose a high efficiency apodized grating coupler with a bottom reflector for silicon nitride photonic integrated circuits. The reflector consists of a stack of alternate silicon nitride and silicon dioxide quarter-wave films. The design, fabrication and optical characterization of the couplers has been presented. The measured fiber to detector insertion loss was -3.5 dB which corresponds to a peak coupling efficiency of -1.75 dB. A 3 dB wavelength bandwidth of 76.34 nm was demonstrated for the grating coupler with a 20-layer reflector. The fabrication process is CMOS-compatible and requires only a single etching step.

Silicon nitride (SiN $_x$ ) has recently attracted increasing attention and has emerged as an alternative material for photonic integrated circuits (PICs) $^1$ . Although it has a refractive index of around 1.7–2.3 smaller than that of silicon, it is sufficient for compact integration and high optical power confinement. The moderately small index contrast can reduce the scattering loss due to the sidewall roughness, and allows a high fabrication tolerance $^{2,3}$ . The SiN $_x$ is transparent through both visible and infrared spectra $^4$ , and can be readily prepared by conventional deposition technologies $^5$ .

In order to obtain wafer-scale testing flexibility and facile coupling operation, waveguide grating couplers have been used in Si and $SiN_{x}$ PICs $^{6-12}$ . Compared to a silicon grating coupler, $SiN_{x}$ grating couplers have a lower coupling efficiency due to a lower index contrast $^{2,9,13}$ . In order to improve the coupling efficiency, two problems must be addressed. One of the problems is that a majority of the optical power is reflected downward into the substrate and dissipates. Bottom reflectors such as a metal mirror $^{12,14}$ and distributed Bragg reflector $^{3,9,15}$ have been utilized to improve the coupling efficiency of silicon and silicon nitride grating couplers. Linear silicon grating reflectors have also been adopted to achieve a higher directionality $^{2,16}$ . However, this involves several pattern steps and multiple waveguide layers. Another problem is that the field distribution of the coupled out beam from a uniform $SiN_{x}$ grating coupler does not match the field distribution of a single mode fiber (SMF) $^{2,11}$ . This mismatch limits the coupling efficiency. Apodizing of the grating can facilitate a matching of the field distributions.

In this study, we have introduced a bottom multilayer reflector and period apodization to improve the coupling efficiency of the $SiN_{x}$ grating couplers. The reflector consists of a stack of alternate $SiN_{x}$ and $SiO_{2}$ quarter-wave layers, which can be deposited by using the liquid source chemical vapor deposition (LSCVD) technique. The main advantages of the proposed waveguide grating coupler are a high coupling efficiency, versatile design, simple fabrication and complementary metal-oxide semiconductor (CMOS) compatibility.

# Results and Discussion

Figure 1 shows the waveguide grating coupler with a bottom multilayer reflector on a silicon nitride platform. A stack of the multilayer is deposited on the Si substrate and cladded by a buried oxide (BOX) buffer layer. It consists of a $\mathrm{SiN_x}$ layer at the buffer layer side, followed by alternate $\mathrm{SiO_2}$ and $\mathrm{SiN_x}$ layers. Refractive indices of $\mathrm{SiO_2}$ , $\mathrm{SiN_x}$ and Si are 1.45, 1.80 and 3.45, respectively. The stack of $\mathrm{SiO_2}$ and $\mathrm{SiN_x}$ layers can be fabricated by manipulating the recipe of the LSVCD machine at the same temperature. As a consequence of this deposition technique, the multilayer can be easily fabricated. Furthermore, the low index contrast between $\mathrm{SiO_2}$ and $\mathrm{SiN_x}$ provides a good fabrication tolerance for film thickness control. The thickness of the top silicon nitride waveguide layer is $325\mathrm{nm}$ .

$^{1}$ Institute for Materials Chemistry and Engineering, Kyushu University, 6-1 Kasuga-koen Kasuga-city, Fukuoka, 816-8580, Japan. $^{2}$ Department of Molecular and Material Sciences, Kyushu University, 6-1 Kasuga-koen Kasuga-city, Fukuoka, 816-8580, Japan. $^{3}$ School of Information Engineering, Wuhan University of Technology, Wuhan, 430070, China. Correspondence and requests for materials should be addressed to S.Y. (email: s\_yokoyama@cm.kyushu-u.ac.jp)

![](images/94ec6aec97da62db872248d8743e2f818c2ad40370a2a749b7f36f0291ea5948.jpg)

<details>
<summary>text_image</summary>

Period
Silicon nitride
Grating
325 nm
Silicon dioxide
2.8 µm
Stack
Silicon substrate
</details>

Figure 1. Schematic illustration of the grating coupler with a bottom reflector. The multilayer reflector consists of alternate $\mathrm{SiO}_2$ and $\mathrm{SiN_x}$ layers.

Multilayer reflector design. The bottom reflector consists of a stack of a multilayer film with alternate $SiO_{2}$ and $SiN_{x}$ layers. This type of multilayer film can be modeled by the transfer matrix method $^{17}$ . For the theoretical discussion, we denote the wavelength and center wavelength by $\lambda$ and $\lambda_{0}$ , respectively. While all layers in the stack are one-quarter of a wavelength ( $\lambda_{0}$ ) thick, the multilayer system performs as a highly reflective optical film. The reflectance of the quarter-wave stack at $\lambda_{0}$ can be simply expressed as follows.

$$
R = \left[ \frac {n _ {\mathrm{clad}} - \left(\frac {n _ {\mathrm{SiN}}}{n _ {\mathrm{SiO} _ {2}}}\right) ^ {2 p} \left(\frac {n _ {\mathrm{SiN}} ^ {2}}{n _ {\mathrm{sub}}}\right)}{n _ {\mathrm{clad}} + \left(\frac {n _ {\mathrm{SiN}}}{n _ {\mathrm{SiO} _ {2}}}\right) ^ {2 p} \left(\frac {n _ {\mathrm{SiN}} ^ {2}}{n _ {\mathrm{sub}}}\right)} \right] ^ {2} \quad \text { for } N = 2 p + 1, \tag {1}
$$

$$
R = \left[ \frac {n _ {\mathrm{clad}} - \left(\frac {n _ {\mathrm{SiN}}}{n _ {\mathrm{SiO} _ {2}}}\right) ^ {2 p} n _ {\mathrm{sub}}}{n _ {\mathrm{clad}} + \left(\frac {n _ {\mathrm{SiN}}}{n _ {\mathrm{SiO} _ {2}}}\right) ^ {2 p} n _ {\mathrm{sub}}} \right] ^ {2} \quad \text { for } N = 2 p, \tag {2}
$$

where N is the number of layers, p is an integer, $n_{SiO2}$ , $n_{SiN}$ , $n_{clad}$ and $n_{sub}$ are the refractive indices of silicon dioxide, silicon nitride, the cladding and substrate, respectively. Figure 2 shows the calculated reflection spectra of quarter-wave stacks by using the transfer matrix method. The plateau centered at $\lambda_{0}$ offers a high-reflectance zone within which the reflectance at each wavelength increases monotonically with increasing N, and approaching 1.0 as N tends to infinity. The width of the high-reflectance zone is near $0.2\lambda_{0}$ . The multilayer effectively functions as a broadband reflector within this zone. The insert of Fig. 2 shows the reflectance of quarter-wave stacks with different layers at $\lambda_{0}$ calculated by using Eqs (1) and (2). It can be observed that the films with an even number of layers have higher reflectance than the films with an adjacent odd number of layers. So, we selected N to be 18 or 20 for the fabrication.

Grating optimization and apodization design. Firstly, the period and the BOX thickness of the uniform silicon nitride grating coupler were optimized $^{8}$ . Subsequently, the apodization was designed near the optimized values. Here, we analyze and discuss the coupling efficiency from grating to single mode fiber by assuming the coupling efficiency from fiber to grating is the same. The thicknesses of the alternate $SiO_{2}$ and $SiN_{x}$ layers are set to be 258.6 nm and 208.3 nm, which are exactly a quarter-wave thickness at $\lambda_{0}=1.55\mu m$ . For the first order Bragg diffraction, the grating period can be simplified as $^{3}$

$$
\Lambda = \frac {\lambda_ {0}}{n _ {\text {eff}} - n _ {\mathrm{c}} \sin \theta} \tag {3}
$$

where, $n_{c}$ is the refractive index of the cladding material, $\theta$ is the coupling angle, $n_{eff}$ is the effective refractive index and can be estimated by $^{18}$

![](images/496886d48ba43bb69c26ee79993aeb60bd136a826e116fa165a680d3688510f7.jpg)

<details>
<summary>line</summary>

| λ₀/λ | Reflectance (a. u.) |
|------|---------------------|
| 1.0  | 0.9                 |
| 1.0  | 0.7                 |
| 1.0  | 0.5                 |
| 1.0  | 0.3                 |
| 1.0  | 0.1                 |
| 1.0  | -0.1                |
| 1.0  | -0.2                |
| 1.2  | 0.3                 |
| 1.2  | 0.5                 |
| 1.2  | 0.7                 |
| 1.2  | 0.9                 |
| 1.2  | 1.1                 |
| 1.4  | 0.3                 |
| 1.4  | 0.5                 |
| 1.4  | 0.7                 |
| 1.4  | 0.9                 |
| 1.4  | 1.1                 |
| 1.6  | 0.3                 |
| 1.6  | 0.5                 |
| 1.6  | 0.7                 |
| 1.6  | 0.9                 |
| 1.6  | 1.1                 |
| 1.8  | 0.3                 |
| 1.8  | 0.5                 |
| 1.8  | 0.7                 |
| 1.8  | 0.9                 |
| 1.8  | 1.1                 |
| 2.0  | 0.3                 |
| 2.0  | 0.5                 |
| 2.0  | 0.7                 |
| 2.0  | 0.9                 |
| 2.0  | 1.1                 |
</details>

Figure 2. Reflection spectra of quarter-wave multilayer stacks. The number of layers is shown as a parameter on the curves. The insert is the reflectance at the center wavelength for the multilayer stacks with different layers.

![](images/f4c12d6d012117e0482952da7a7f1a2be3e430cd84d9265133d721993162ae95.jpg)

<details>
<summary>line</summary>

| Period (μm) | 20 Layers | 20 Layers | 18 Layers | Without Reflector |
|-------------|-----------|-----------|-----------|-------------------|
| 0.6         | -18       | -19       | -21       | -25               |
| 0.8         | -10       | -13       | -15       | -20               |
| 1.0         | -5        | -4        | -10       | -15               |
| 1.2         | -2        | -3        | -5        | -7                |
| 1.4         | -5        | -8        | -10       | -13               |
| 1.6         | -10       | -15       | -17       | -18               |
</details>

![](images/d403f2ce4e9ddc208553f3d5e87d23285553dd5edda66887d9932b386af0e6d0.jpg)

<details>
<summary>line</summary>

| BOX Thickness (μm) | 20 Layers | 18 Layers | Without Reflector |
| ------------------ | --------- | --------- | ----------------- |
| 2.0                | -12.0     | -14.0     | -8.0              |
| 2.5                | -16.0     | -18.0     | -10.0             |
| 3.0                | -2.0      | -2.0      | -4.0              |
| 3.5                | -6.0      | -4.0      | -8.0              |
| 4.0                | -12.0     | -12.0     | -10.0             |
| 4.5                | -16.0     | -6.0      | -4.0              |
| 5.0                | -2.0      | -2.0      | -8.0              |
| 5.5                | -6.0      | -4.0      | -10.0             |
| 6.0                | -12.0     | -14.0     | -8.0              |
| 6.5                | -6.0      | -4.0      | -10.0             |
| 7.0                | -16.0     | -18.0     | -8.0              |
</details>

Figure 3. Coupling efficiencies of the grating couplers. (a) Calculated (dashed line) and measured (solid lines) coupling efficiencies as a function of the grating period. (b) Calculated coupling efficiencies as a function of the thickness of the BOX layer.

$$
n _ {\text { eff }} = f f n _ {\text { eff1 }} + (1 - f f) n _ {\text { eff2 }} \tag {4}
$$

where $ff$ is the fill factor defined as the ratio of the grating teeth to the period, $n_{\mathrm{eff1}}$ and $n_{\mathrm{eff2}}$ are the effective refractive index of the grating teeth and the grating slots, respectively.

In order to compare the calculation with experiment results for period optimization, we also fabricated grating couplers with periods from 0.6 $\mu$ m to 1.5 $\mu$ m. The coupling efficiencies of couplers with different periods are shown in Fig. 3(a). It can be observed that the coupling efficiency is highly dependent on the grating period. The maximum coupling efficiencies were obtained at the period of 1.2 $\mu$ m in both calculation and experiment for a standard uniform grating coupler.

The thickness of the BOX layer also has an important impaction on the coupling efficiency. Figure 3(b) shows the dependence of the coupling efficiency on the BOX thickness for grating couplers with different layers in the reflector. It can be observed that the patterns for different values of N are similar. BOX thicknesses of the peaks for different N are also similar. Therefore, we selected the BOX thickness as $2.8 \mu m$ . Because of the oscillation in the pattern, the thickness must be controlled precisely.

The spectra of grating couplers with different periods and fill factors were calculated and are shown in Fig. 4. It can be observed that the spectrum shifts to a longer wavelength with the period and fill factor increasing. According to Eqs (3) and (4), the effective refractive index increases with the fill factor, which results in a spectrum shifting. The peak coupling efficiency also changes with the period and fill factor. Moreover, the period has a much larger impact on the spectrum and coupling efficiency than the fill factor. In order to maximize the

![](images/0c0d4b3fddd288e484e5b63d9b2d0fe30920e87d9f55ed105666d6652ff68592.jpg)

<details>
<summary>line</summary>

| Wavelength (nm) | Period 1.3 | Period 1.25 | Period 1.2 | Period 1.15 | Period 1.1 |
| --------------- | ---------- | ----------- | ---------- | ----------- | ---------- |
| 1500            | -7.0       | -5.5        | -4.0       | -3.5        | -2.5       |
| 1520            | -5.0       | -4.0        | -3.0       | -2.5        | -2.0       |
| 1540            | -3.0       | -2.5        | -2.0       | -1.5        | -1.5       |
| 1560            | -2.0       | -2.0        | -1.5       | -1.0        | -1.0       |
| 1580            | -3.0       | -3.0        | -2.5       | -2.0        | -2.5       |
| 1600            | -5.0       | -5.5        | -6.0       | -7.0        | -7.5       |
</details>

![](images/6a406a71ce0c260d97ff1cf26302ddb21bdcd62616e11730703eaa2f901a03f8.jpg)

<details>
<summary>line</summary>

| Wavelength (nm) | Fill Factor 0.6 | Fill Factor 0.55 | Fill Factor 0.5 | Fill Factor 0.45 | Fill Factor 0.4 |
| --------------- | --------------- | ---------------- | --------------- | ---------------- | --------------- |
| 1500            | -4.2            | -4.3             | -4.4            | -4.5             | -4.6            |
| 1520            | -3.5            | -3.6             | -3.7            | -3.8             | -3.9            |
| 1540            | -2.8            | -2.9             | -3.0            | -3.1             | -3.2            |
| 1560            | -2.2            | -2.3             | -2.4            | -2.5             | -2.6            |
| 1580            | -3.0            | -3.1             | -3.2            | -3.3             | -3.4            |
| 1600            | -6.8            | -6.9             | -7.0            | -7.1             | -7.2            |
</details>

Figure 4. Calculated spectra of grating couplers with N=20. (a) Spectra of grating couplers with different periods. The fill factor of the grating is set as 0.5. (b) Spectra of grating couplers with different fill factors. The period of the grating is set as 1.2 $\mu$ m.

![](images/5c643e0e19e926b880ec66243937030175247d625e52f8b0c62bc291ba9f46a1.jpg)

<details>
<summary>line</summary>

| Waveguide Thickness (nm) | Coupling Efficiency (dB) |
| ------------------------ | ------------------------ |
| 200                      | -5.5                     |
| 250                      | -3.0                     |
| 300                      | -1.5                     |
| 350                      | -1.5                     |
| 400                      | -2.0                     |
| 450                      | -2.5                     |
| 500                      | -3.0                     |
</details>

Figure 5. Calculated coupling efficiencies of grating couplers with a different waveguide thickness. The period, fill factor and N are $1.2 \mu m$ , 0.5 and 20, respectively.

coupling efficiency at a specific wavelength, both the period and fill factor must be optimized. Here, the period and fill factor were selected as $1.2\mu \mathrm{m}$ and 0.5, respectively. In this case, the 3 dB and 1 dB bandwidths are $82\mathrm{nm}$ and $54\mathrm{nm}$ , respectively.

The coupling efficiency dependence on the thickness of the top silicon nitride layer was calculated and is shown in Fig. 5. It can be observed that the thickness of $325\mathrm{nm}$ is near the center of the maximum region. Therefore, the thickness of the top silicon nitride was selected as $325\mathrm{nm}$ in this investigation. The silicon nitride waveguide with a width of $1.0\mu \mathrm{m}$ and a thickness of $325\mathrm{nm}$ functions in a single mode condition. The insert of Fig. 5 shows the field distribution of the fundamental mode.

The coupling efficiency can also be improved further by apodizing the period and the fill factor of the grating $^{11}$ . The optical power guided in the uniform grating structure can be described as $^{19,20}$

$$
P = P _ {0} e ^ {- 2 \alpha z} \tag {5}
$$

where, $\alpha$ is the field leakage factor, $P_{0}$ is power of the guided wave at the beginning of the grating. dP is the power due to the harmonic fields radiate away from the waveguide. Therefore, the radiated out power can be expressed as $^{19,20}$

$$
P _ {\text { rad }} = - \frac {d P}{d z} = 2 \alpha P _ {0} e ^ {- 2 \alpha z} \tag {6}
$$

Note, $P_{\mathrm{rad}}$ is the power per unit length along $z$ direction. The leakage factor $\alpha$ can be extracted from the radiation power density data by using Eq. (6).

![](images/da3f63f2dc34c19c3e0d4bc094f1550d348d9f898d1b3038bf110e2a50f1a6cb.jpg)

<details>
<summary>line</summary>

| Period (μm) | Power Leakage Factor (μm⁻¹) | Fill Factor (%) |
|-------------|-----------------------------|-----------------|
| 1.13        | 0.0                         | 0.05            |
| 1.14        | 0.1                         | 0.10            |
| 1.15        | 0.2                         | 0.15            |
| 1.16        | 0.3                         | 0.20            |
| 1.17        | 0.4                         | 0.25            |
| 1.18        | 0.5                         | 0.30            |
| 1.19        | 0.6                         | 0.35            |
| 1.20        | 0.7                         | 0.40            |
| 1.21        | 0.8                         | 0.45            |
</details>

![](images/580ab10d096b9c14b1bd3ef4712b369a9385d4c62c087273e0126558db65d892.jpg)

<details>
<summary>line</summary>

| z (μm) | SMF Power Profile | Apodized Grating Coupler |
| ------ | ----------------- | ------------------------ |
| -10    | 0.0               | 0.0                      |
| -5     | 0.0               | 0.0                      |
| 0      | 0.0               | 0.0                      |
| 5      | 0.8               | 0.9                      |
| 10     | 0.2               | 0.3                      |
| 15     | 0.0               | 0.0                      |
| 20     | 0.0               | 0.0                      |
| 25     | 0.0               | 0.0                      |
</details>

Figure 6. The apodization results of the grating couplers. (a) Calculated power leakage factor and the required fill factor of a grating cell as a function of grating period. (b) The coupling out field profile of the apodized grating and the Gaussian fiber field profile with an MFD of $10.4 \mu m$ .

In order to obtain a Gaussian output electric field profile $G(z)$ , the leakage factor $\alpha$ is varied along the z direction to make $P_{\mathrm{rad}} = G^{2}(z)^{20-22}$ . The dependence of the leakage factor $2\alpha$ on the grating length can be expressed as

$$
2 \alpha = \frac {G ^ {2} (z)}{P _ {0} - \int_ {0} ^ {z} G ^ {2} (z) d z} \tag {7}
$$

where, $G(z)$ is used to describe the Gaussian field profile of a standard single mode fiber with a mode field diameter (MFD) of $10.4\mu \mathrm{m}$ . Neglecting the radiation into the substrate, $\int_0^{z_1}G^2 (z)dz = \eta P_0$ . Here, $\eta$ is the coupling efficiency, the beginning and the end of the grating are assumed at 0 and $z_{1}$ , respectively.

The leakage factor $\alpha$ for gratings with different periods were calculated by using 2D simulation and Eq. (6). The result is shown in Fig. 6(a). During simulation, the fill factor was determined by carefully checking the scattering direction for every period. In this case, the Bloch condition is satisfied, which can ensure the phase match between adjacent grating cells in the apodized grating coupler.

Then, the required period and fill factor distribution for the apodized grating coupler were determined according to Eq. (7) and Fig. 6(a). The periods and fill factors (in brackets) from the front to the back end are $1.135\mu \mathrm{m}$ $(0.086\mu \mathrm{m})$ , $1.14\mu \mathrm{m}$ $(0.12\mu \mathrm{m})$ , $1.147\mu \mathrm{m}$ $(0.16\mu \mathrm{m})$ , $1.153\mu \mathrm{m}$ $(0.212\mu \mathrm{m})$ , $1.161\mu \mathrm{m}$ $(0.26\mu \mathrm{m})$ , $1.168\mu \mathrm{m}$ $(0.316\mu \mathrm{m})$ , $1.175\mu \mathrm{m}$ $(0.365\mu \mathrm{m})$ , $1.182\mu \mathrm{m}$ $(0.413\mu \mathrm{m})$ , $1.191\mu \mathrm{m}$ $(0.47\mu \mathrm{m})$ , $1.2\mu \mathrm{m}$ $(0.54\mu \mathrm{m})$ , $1.191\mu \mathrm{m}$ $(0.47\mu \mathrm{m})$ and $1.182\mu \mathrm{m}$ $(0.413\mu \mathrm{m})$ , followed by a 9-period uniform grating of $1.2\mu \mathrm{m}$ $(0.54\mu \mathrm{m})$ . The output field distribution profile of the designed apodized grating coupler is shown in Fig. 6(b). It overlaps well with the SMF Gaussian profile which ensures a high coupling efficiency.

After the grating optimization and bottom reflector design, coupling efficiencies of grating couplers with different N were calculated as shown in Fig. 7. As can be observed, couplers with an even number of bottom layers showed a higher coupling efficiency than the couplers with an adjacent odd number of bottom layers. The tendency agreed well with that shown in Fig. 2. The coupling efficiency increased up to -1.0 dB with increasing N, and reached a plateau for N > 12.

Fabrication and coupling efficiency measurement. The scanning electron microscopy (SEM) images of the fabricated waveguide are shown in Fig. 8. A 20-period grating with a short period of $0.52\mu \mathrm{m}$ was arranged at the end of the coupler to reflect back the escaping power. The periods and fill factor of the grating fit the designed pattern well. The input and output grating couplers were linked by a standard $3.0\mathrm{mm}$ long strip waveguide with a width of $1.0\mu \mathrm{m}$ . The coupler had a $13\mu \mathrm{m}$ -wide and $30\mu \mathrm{m}$ -long structure. The mode in the strip was transformed adiabatically to the grating coupler waveguides via a $500\mu \mathrm{m}$ -long taper.

We characterized the coupling efficiencies of grating couplers by measuring the laser to detector insertion losses. The grating couplers diffract light into/from cleaved single mode fibers tilted at an angle of $8^{\circ}$ with respect to the grating surface normal. Usually, the transmission loss of the silicon nitride strip should be subtracted from the insertion loss $^{2,22-24}$ . The coupling efficiency is half of the insertion loss after subtraction. The propagating loss was measured to be 2.4 dB/cm by comparing straight waveguides with different lengths. The measured coupling efficiencies at 1550 nm of apodized and unapodized couplers with different bottom reflectors are summarized in Table 1. The coupling efficiency increases with N as expected from the calculation. The coupling efficiency of the grating couplers with 18 and 20 layers are very close which agrees well with the simulation. By comparing with the standard grating coupler, the coupling efficiency was improved by 3.65 dB. It can be observed that the bottom reflector can improve the coupling efficiency by approximately 3.2 dB and the apodization can improve the coupling efficiency by approximately 0.7 dB.

Figure 9(a) shows the transmission spectra of the apodized grating couplers (N=0, 18 and 20). The light from a tunable laser (SANTEC TSL550) was coupled into the grating coupler. The wavelength was swept over a range of 100 nm and the spectrum recorded by an optical power meter (SANTEC MPM200) with 1 pm wavelength resolution. The measured bandwidth of grating couplers with reflectors is almost identical to that of the standard

![](images/df67c77434c646ab809606e0552ae2c11148e9d3526f4c71cfaa1679a053313d.jpg)

<details>
<summary>line</summary>

| Number of Layers (a. u.) | Even Layers | Odd Layers |
| ------------------------ | ----------- | ---------- |
| 0                        | -4.0        | -8.5       |
| 2                        | -3.0        | -6.0       |
| 4                        | -2.5        | -5.0       |
| 6                        | -2.0        | -4.0       |
| 8                        | -1.8        | -3.0       |
| 10                       | -1.6        | -2.5       |
| 12                       | -1.5        | -2.2       |
| 14                       | -1.4        | -2.0       |
| 16                       | -1.3        | -1.8       |
| 18                       | -1.2        | -1.7       |
| 20                       | -1.1        | -1.6       |
| 22                       | -1.0        | -1.5       |
| 24                       | -0.9        | -1.4       |
</details>

Figure 7. Calculated coupling efficiencies of grating couplers with different bottom reflector layers. Grating apodization and optimized BOX thickness were adopted in this calculation.

![](images/29663fb92011e2d064cc1e2bd0ab2a96ce5486a111d69ed37afb78264ce4f0d4.jpg)

<details>
<summary>natural_image</summary>

Microscopic view of a striped pattern with 10 μm scale bar, labeled (a), showing no text or symbols beyond the scale indicator.
</details>

![](images/233fd7257ea36366a517e7bb1a67339a2c1598035b060f53afd98bfeab138a68.jpg)

<details>
<summary>natural_image</summary>

Microscopic grayscale image showing layered material structure with no visible text or symbols
</details>

Figure 8. SEM pictures of the device. (a) SEM top view of the grating coupler. (b) SEM cross-section of the stack of 18 layers. The insert shows the zoom-in image including 10 layers. The bright and dark strips are $SiN_{x}$ and $SiO_{2}$ layers, respectively.

<table><tr><td rowspan="2">Number of layers (N)</td><td colspan="2">Coupling efficiency (dB)</td></tr><tr><td>Apodized</td><td>Unapodized</td></tr><tr><td>0</td><td>-4.9</td><td>-5.6</td></tr><tr><td>18</td><td>-2.07</td><td>-2.6</td></tr><tr><td>20</td><td>-1.95</td><td>-2.4</td></tr></table>

Table 1. Measured coupling efficiencies at a wavelength of 1550 nm.

grating coupler without a reflector. The 3 dB bandwidth of the grating coupler with N=20 is 76.34 nm, which covers most of operation frequencies of C-band and L-band. The corresponding 1 dB bandwidth is 52.5 nm, which is smaller than the reported 80 nm of the $Si_{3}N_{4}$ -on-SOI dual-level grating coupler $^{16}$ . We attributed the smaller 1 dB bandwidth to the non-flat high-reflectance zone of the bottom multilayer reflector as shown in Fig. 2. Even though the width of the high-reflectance zone is as large as several hundred nanometers, the reflectance characteristic is not totally flat. For comparison, the transmission spectra of the unapodized grating couplers are also shown in Fig. 9(b). It can be observed that the apodization can slightly broaden the bandwidth of the grating couplers because more periods were used in apodized grating couplers $^{18}$ .

![](images/e526ee8ae5ccaa6a9cfabce07c808b6c79be0a37be6ae2bceef49781d4c78b1b.jpg)

<details>
<summary>line</summary>

| Wavelength (nm) | 20 Layers | 18 Layers | without reflector |
| --------------- | --------- | --------- | ----------------- |
| 1500            | -10.0     | -10.5     | -12.0             |
| 1520            | -4.0      | -4.5      | -6.0              |
| 1540            | -2.0      | -2.5      | -5.5              |
| 1560            | -2.5      | -3.0      | -6.0              |
| 1580            | -3.5      | -4.0      | -7.0              |
| 1600            | -5.0      | -6.0      | -9.0              |
</details>

![](images/6ebb109f498217ddb6bbc6db2c978622c5390e13733440676c06ae0c309a08df.jpg)

<details>
<summary>line</summary>

| Wavelength (nm) | 20 Layers | 18 layers | without reflector |
| --------------- | --------- | --------- | ----------------- |
| 1500            | -14.0     | -14.0     | -17.0             |
| 1520            | -4.0      | -4.0      | -8.0              |
| 1540            | -2.0      | -2.0      | -6.0              |
| 1560            | -3.0      | -3.0      | -6.0              |
| 1580            | -4.0      | -4.0      | -8.0              |
| 1600            | -10.0     | -10.0     | -14.0             |
</details>

Figure 9. Measured transmission spectra. (a) Spectra of the apodized grating couplers with reflectors; (b) Spectra of the unapodized grating couplers with reflectors.

It should be noted that no index matching fluid was applied between the gratings and fibers during the measurements for practical convenience and repeated use. However, an estimate of the excess loss due to the use of air coupling would be beneficial in understanding the performance of the device. It was evaluated experimentally that the coupling efficiency can be improved by about 0.35 dB while the index matching fluid was used as the waveguide top cladding and filled the gap between the gratings and fibers $^{25}$ . In our experimental samples, $SiO_{2}$ films were deposited as a top cladding. In order to evaluate the improvement effect of the index matching fluid, we dropped the index matching fluid on the surface to fill the gaps between the gratings and fibers and measured the coupling efficiency. Results show that the coupling efficiency was improved by about 0.20 dB. Taking into account this improvement, the peak coupling efficiency of the proposed grating coupler is -1.75 dB.

# Conclusions

An efficient apodized $SiN_{x}$ waveguide grating coupler with a bottom multilayer reflector was theoretically and experimentally demonstrated. Apodization and a multilayer reflector were adopted to improve the coupling efficiency. The reflector was designed by using alternate quarter-wave $SiN_{x}$ and $SiO_{2}$ layers. The waveguide was fabricated by one EBL and ICP etching step. Results show that the bottom reflector and the grating apodization were able to enhance the coupling efficiency by approximately 3.2 dB and 0.7 dB, respectively. A high coupling efficiency of -1.75 dB with 3 dB bandwidth of 76.34 nm has been demonstrated. This approach facilitates the design and fabrication process of the grating couplers with a bottom reflector.

# Methods

This multilayer bottom reflector was modeled by the general transfer matrix method. The reflectance and spectra were calculated by using C-language programming. For design and optimization of the grating coupler with a bottom multilayer reflector, we used the two-dimensional (2D) finite difference time domain method (FDTD) by using the commercial software Rsoft $^{8,11,12,26}$ .

In the fabrication, a silicon wafer was used as the substrate. The $SiN_{x}$ and $SiO_{2}$ films were deposited using the LSCVD (SAMCO) at $150^{\circ}C$ . LSCVD can deposit silicon nitride at low temperature by using $N_{2}$ instead of $NH_{3}$ as the deposition precursor, which decreases the concentration of the dangling hydrogen bonds and the tensile stress in the film $^{27,28}$ . There is no obvious absorption peak near 1520 nm which exists in silicon nitride films deposited by plasma enhanced chemical vapor deposition (PECVD) $^{29}$ . Moreover, no cracks were observed over the entire sample. We prepared bottom reflectors with N=0, 18 and 20 for comparison. The waveguide and grating structures were patterned by electronic beam lithography (EBL) and ion coupling plasma (ICP) with $CHF_{3}$ gas.

# References

1. Spencer, D. T., Bauters, J. F., Heck, M. J. & Bowers, J. E. Integrated Waveguide Coupled $Si_{3}N_{4}$ Resonators in the Ultrahigh-Q Regime. Optica 1(3), 153–157 (2014).   
2. Zou, J. et al. Ultra Efficient Silicon Nitride Grating Coupler with Bottom Grating Reflector. Opt. Express 23(20), 26305–26312 (2015).   
3. Zhang, H. et al. Efficient Silicon Nitride Grating Coupler with Distributed Bragg Reflectors. Opt. Express 22(18), 21800–21805 (2014).   
4. Subramanian, A. Z. et al. Near-Infrared Grating Couplers for Silicon Nitride Photonic Wires. IEEE Photon. Technol. Lett. 24(19), 1700–1703 (2012).   
5. Daldosso, N. et al. Comparison Among Various $Si_{3}N_{4}$ Waveguide Geometries Grown within a CMOS Fabrication Pilot Line. J. Lightwave Technol. 22(7), 1734–1740 (2004).   
6. Liang, Z. et al. A Novel Highly Efficient Grating Coupler with Large Filling Factor Used for Optoelectronic Integration. Chinese physics B 19(12), 124214 (2010).   
7. Mekis, A. et al. A Grating-Coupler-Enabled CMOS Photonics Platform. IEEE J. Sel. Top. Quantum Electron. 17(3), 597–608 (2011).   
8. Chen, X., Li, C. & Tsang, H. K. Fabrication-Tolerant Waveguide Chirped Grating Coupler for Coupling to a Perfectly Vertical Optical Fiber. IEEE Photon. Technol. Lett. 20(23), 1914–1916 (2008).   
9. Doerr, C. R., Chen, L., Chen, Y. & Buhl, L. L. Wide Bandwidth Silicon Nitride Grating Coupler. IEEE Photon. Technol. Lett. 22(19), 1461–1463 (2010).

10. Shim, E., Chen, Y., Masmanidis, S. & Li, M. Multisite Silicon Neural Probes with Integrated Silicon Nitride Waveguides and Gratings for Optogenetic Applications. Scientific Reports 6, 22693 (2016).   
11. Marchetti, R. et al. High-Efficiency Grating-Couplers: Demonstration of a New Design Strategy. Scientific Reports 7, 16670 (2017).   
12. Asaduzzaman, M., Bakaul, M., Skafidas, E. & Khandokar, M. R. H. A Compact Silicon Grating Coupler Based On Hollow Tapered Spot-Size Converter. Scientific Reports 8, 2540 (2018).   
13. Zhang, H. et al. High Efficiency Silicon Nitride Grating Coupler. Applied Physics A 115(1), 79–82 (2014).   
14. Van Laere, F. et al. Compact and Highly Efficient Grating Couplers Between Optical Fiber and Nanophotonic Waveguides. J. Lightwave Technol. 25(1), 151–156 (2007).   
15. Taillaert, D. et al. Grating Couplers for Coupling Between Optical Fibers and Nanophotonic Waveguides. Jpn. J. Appl. Phys. 45(8A), 6071–6077 (2006).   
16. Sacher, W. D. et al. Wide Bandwidth and High Coupling Efficiency $Si_{3}N_{4}$ -on-SOI Dual-Level Grating Coupler. Opt. Express 22(9), 10938–10947 (2014).   
17. MacLeod, H. A. Thin-Film Optical Filters. 216–230 (CRC Press, USA, 2010).   
18. Hong, J., Qiu, F., Spring, A. M. & Yokoyama, S. Silicon Waveguide Grating Coupler Based On a Segmented Grating Structure. Applied Optics 57(12), 3301–3305 (2018).   
19. Tamir, T. & Peng, S. Analysis and Design of Grating Couplers. Applied physics 14(3), 235-254 (1977).   
20. Waldhäusl, R. et al. Efficient Coupling into Polymer Waveguides by Gratings. Applied optics 36(36), 9383-9390 (1997).   
21. Chen, X., Li, C., Fung, C. K., Lo, S. M. & Tsang, H. K. Apodized Waveguide Grating Couplers for Efficient Coupling to Optical Fibers. IEEE Photon. Technol. Lett 22(15), 1156–1158 (2010).   
22. Ding, Y., Ou, H. & Peucheret, C. Ultrahigh-Efficiency Apodized Grating Coupler Using Fully Etched Photonic. Crystals. Opt. Letters 38(15), 2732–2734 (2013).   
23. Zhao, X. et al. Compact Grating Coupler for 700-Nm Silicon Nitride Strip Waveguides. J. Lightwave Technol. 34(4), 1322–1327 (2016).   
24. Ding, Y., Peucheret, C., Ou, H. & Yvind, K. Fully Etched Apodized Grating Coupler On the SOI Platform with -0.58 dB Coupling Efficiency. Opt. Letters 39(18), 5348–5350 (2014).   
25. Wirth, J. C. Silicon Grating Couplers for Low Loss Coupling Between Optical Fiber and Silicon Nanowires. (Purdue Univ., West Lafayette, USA, 2011).   
26. Piggott, A. Y. et al. Inverse Design and Implementation of a Wavelength Demultiplexing Grating Coupler. Scientific Reports 4, 7210 (2014).   
27. Shao, Z. et al. Ultra-Low Temperature Silicon Nitride Photonic Integration Platform. Opt. Express 24(3), 1865-1872 (2016).   
28. Mao, S. C. et al. Low Propagation Loss SiN Optical Waveguide Prepared by Optimal Low-Hydrogen Module. Opt. Express 16(25), 20809–20816 (2008).   
29. Worhoff, K., Lambeck, P. V. & Driessen, A. Design, Tolerance Analysis, and Fabrication of Silicon Oxynitride Based Planar Optical Waveguides for Communication Devices. J. Lightwave Technol. 17(8), 1401–1407 (1999).

# Acknowledgements

This work is supported by the Cooperative Research Programs of “Network Joint Research Center for Material and Device” and “Dynamic Alliance for Open Innovation Bridging Human, Environment, and materials” of MEXT, KAKENHI (266220712, 19H00770) of JSPS, CREST (16815359) and SICORP (JPMJSC1807) of JST.

# Author Contributions

Jianxun Hong and Shiyoshi Yokoyama proposed the idea, performed the calculations and wrote the paper. Andrew M. Spring provided helpful advices on structure design and refined the paper. Feng Qiu provided helpful advices on device fabrication. All authors reviewed the manuscript.

# Additional Information

Competing Interests: The authors declare no competing interests.

Publisher's note: Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

![](images/5ddc4cbf629ee667800a8c4d3eb65cb53c28dc08c7f5dac4fa5472d5b8b89730.jpg)

Open Access This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or

format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons license, and indicate if changes were made. The images or other third party material in this article are included in the article's Creative Commons license, unless indicated otherwise in a credit line to the material. If material is not included in the article's Creative Commons license and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.

© The Author(s) 2019