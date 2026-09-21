ARTICLE

https://doi.org/10.1038/s41467-022-30965-6

OPEN

![](images/e5eb732501a6ce4d76eae17c49f52493efc7105d06f155b541dd1a5976bde6a6.jpg)

Check for updates

# Optomechanical crystal with bound states in the continuum

Shengyan Liu $^{1,2,3}$ , Hao Tong $^{1,2,3}$ & Kejie Fang $^{1,2}$

Chipscale micro- and nano-optomechanical systems, hinging on the intangible radiation-pressure force, have shown their unique strength in sensing, signal transduction, and exploration of quantum physics with mechanical resonators. Optomechanical crystals, as one of the leading device platforms, enable simultaneous molding of the band structure of optical photons and microwave phonons with strong optomechanical coupling. Here, we demonstrate a new breed of optomechanical crystals in two-dimensional slab-on-substrate structures empowered by mechanical bound states in the continuum (BICs) at 8 GHz. We show symmetry-induced BIC emergence with optomechanical couplings up to $g/2\pi \approx 2.5$ MHz per unit cell, on par with low-dimensional optomechanical crystals. Our work paves the way towards exploration of photon-phonon interaction beyond suspended microcavities, which might lead to new applications of optomechanics from phonon sensing to quantum transduction.

Cavity-optomechanics has attracted extensive studies in recent years because of the rich physics associated with the nonlinear optomechanical interaction and a broad range of prospective applications from signal transduction to sensing $^{1}$ . One of the leading optomechanical device architectures is optomechanical crystals $^{2}$ , where micro- and nano-scale structures give rise to strong radiation-pressure force coupling between wavelength-similar optical photons and microwave phonons. By band-structure engineering of suspended optomechanical crystals, both one-dimensional and quasi-two-dimensional defect cavities $^{2-4}$ have been created with long-lived optical and mechanical resonances $^{5}$ . Such optomechanical crystal microcavities have enabled groundbreaking quantum experiments including ground-state cooling of mechanical resonators $^{6}$ , testing Bell-type inequalities $^{7}$ , and a mechanical quantum memory $^{8}$ .

Despite the success of optomechanical microcavities, it is highly desirable to explore two-dimensional optomechanical crystals. On one hand, two-dimensional optomechanical crystals offer more degrees of freedom for manipulation of photon-phonon interaction to induce collective phenomena $^{9-12}$ . On the other hand, extended optomechanical crystals, especially in unsuspended structures $^{13-15}$ , might alleviate the optical-absorption-induced heating that plagues released microcavities $^{16}$ . Ideally, such slab-on-substrate optomechanical crystals should facilitate the dissipation of heat phonons while sustaining long-lived mechanical resonances in the device layer.

Recently, mechanical bound states in the continuum (BICs) are observed in two-dimensional slab-on-substrate phononic crystals $^{17}$ . Despite having a zero Bloch wavevector and thus spectrally immersing in the sound cone of the substrate, these mechanical BICs are confined in the slab because of the symmetry-induced decoupling from the acoustic radiation field. There are also proposals and demonstrations of mechanical BICs in microcavities due to, for example, accidental radiation amplitude cancellation $^{18,19}$ . A significant step forward thus would be coupling mechanical BICs with optical resonances in an optomechanical crystal which will bring the effective radiation-pressure force control and associated functionalities.

In this work, we realize two-dimensional silicon-on-insulator optomechanical crystals with mechanical BICs coupled with optical guided resonances. In such periodic optomechanical crystals, the radiation-pressure coupling between the mechanical BIC and optical modes strongly depends on the mode symmetry $^{20}$ , which in many cases dictates an adversely null optomechanical coupling. Here, taking into account both symmetry of the optomechanical crystal and the silicon crystal lattice, we are able to achieve optomechanical coupling up to $g/2\pi \approx 2.5$ MHz per unit cell between a mechanical BIC at 8 GHz and an optical band-edge mode at 193 THz, which is comparable to one-dimensional suspended optomechanical crystals $^{21}$ . With optically-transduced mechanical spectroscopy at room temperature, we demonstrate control of mechanical BICs and optomechanical coupling via the interplay of symmetry of the optomechanical crystal and crystalline material. Our work paves the way for the study of photon-phonon interaction in BIC optomechanical crystals at low temperatures, when the benefit of slab-on-substrate device architecture is expected to arise, and Floquet topological physics beyond the tight-binding model $^{22}$ .

# Results

Design of BIC optomechanical crystals. The designed optomechanical crystal in the silicon-on-oxide material system has a hexagonal “snowflake” unit cell (Fig. 1a). The reason behind this design is that sixfold symmetric structures tend to yield mechanical BICs coupled with optical guided resonances $^{20}$ and, also, the spike feature of the “snowflake” could lead to sizable vibrations for large optomechanical couplings. We here consider optical modes at the M point below the light cone and mechanical modes at the $\Gamma$ point. The latter is necessary because mechanical vibrations generally induce a linear perturbation of the optical mode energy, i.e., $\delta U \sim \int \delta x(\mathbf{R})\varepsilon|E(\mathbf{R})|^2 d\mathbf{R}$ , and this integral is nonzero in a periodic structure only when the mechanical displacement $\delta x$ has a zero Bloch wavevector (Supplementary Information (SI)). Since the refractive index of crystalline silicon is isotropic, the symmetry of optical modes at the M point is governed by the $C_{2\nu}$ group. We select the fundamental transverse-electric(TE)-like mode which is odd with respect to the xz-plane. The stiffness tensor of silicon, on the other hand, is anisotropic in the crystal plane with a $C_{4\nu}$ group symmetry, which is incommensurable with the $C_{6\nu}$ group symmetry of the hexagonal optomechanical crystal. As a result, the symmetry of mechanical modes at the $\Gamma$ point will depend on the orientation angle $\theta$ between the optomechanical crystal and the silicon crystal lattice. When $\theta = 0^\circ$ , $15^\circ$ , $30^\circ$ , and $45^\circ$ , the symmetry group of the mechanical mode is $C_{2\nu}$ , while for other orientations it will be $C_2$ . Only $C_{2\nu}$ group supports mechanical BICs which decouple from both transverse and longitudinal radiation waves $^{17}$ , whose displacement field is perpendicular and parallel to the wavevector, respectively.

Figure 1b shows the optical band structure near the M point and mechanical band structure near the $\Gamma$ point for $\theta=0^{\circ}$ and $(r,w,a,t,h)=(167,34,389,220,3000)$ nm, where t and h are the thickness of the silicon and oxide layers, respectively, a is the lattice constant, and $(r,w)$ are “snowflake” dimensions. The fundamental TE-like optical mode has a frequency of 193 THz and the relevant mechanical modes have frequencies of about 7.5 and 8.2 GHz, respectively. Their mode profiles are shown in Fig. 1c. Simulation shows that the radiation quality factor $Q_{r}$ , i.e., the ratio between the frequency and radiation loss rate, of mechanical mode 2 (8.2 GHz) diverges and that of mode 1 is finite but remains relatively high compared to other lossy modes. As a result, for $\theta=0^{\circ}$ , modes 2 and 1 are mechanical BIC and quasi-BIC, respectively. The fact that mode 1 is a quasi-BIC is because the two mechanical bands are degenerate at the $\Gamma$ point when the stiffness tensor is isotropic and both modes are BICs belonging to the $E_{2}$ representation of the $C_{6\nu}$ group; the actual anisotropic stiffness tensor of silicon splits the degeneracy, reducing the $E_{2}$ representation to $A_{2}$ (BIC) and $A_{1}$ (quasi-BIC) representations of the $C_{2\nu}$ group. Mode 1 turns out to have the $A_{1}$ representation, leading to coupling with the longitudinal radiation wave. When $\theta$ changes, BIC and quasi-BIC alternate between modes 1 and 2 (while the frequencies of the two modes remain almost unchanged) as shown in Fig. 1d. This is further illustrated in Fig. 1e using the mode symmetry. Below, the mode symmetry under certain symmetry operations is defined with regard to the vector parity of the electric field of the optical mode or the displacement field of the mechanical mode. Taking $\theta=0^{\circ}$ as an example again, both mode 1 and 2 are even under the $180^{\circ}$ rotation, which leads to decoupling from the transverse radiation wave. In addition, mode 2(1) is odd(even) with respect to the symmetry axes (dashed lines), resulting in decoupling from (coupling with) the longitudinal radiation wave and thus a rigorous(quasi) BIC. Same arguments apply to the other three $\theta$ s while noticing the mirror symmetry axes rotate with $\theta$ . The mechanical BIC and quasi-BIC are also associated with transverse topological charges, defined as the winding number of far-field transverse polarization around the $\Gamma$ point $^{17}$ . The polarization fields rotate together with $\theta$ , which is unique for the anisotropic mechanical system. For orientations other than the four specific

a   
![](images/32c94de4a59d1d78c79635886d569e2f0dd7e1e6b895f7a33b0e12b1544bda1c.jpg)

<details>
<summary>text_image</summary>

Si
SiO₂
</details>

![](images/f1b332bb6d3106acdaa13015d2aa68a2d6de9637ca34eb5efbccc38c8dce5bb2.jpg)

![](images/d3de48e3deed6936eb0abffef58a784c58fc30165111648af1fb1904d32374b6.jpg)

<details>
<summary>text_image</summary>

silicon crystal
optomechanical
crystal
</details>

d

![](images/1fbd1ff8316129aa9bc9a66a506cd1b33ca0677e5352d596a13fffff075f790b.jpg)

<details>
<summary>line</summary>

| θ (°) | σ̃ (blue line) | σ̃ (orange line) |
|-------|----------------|------------------|
| 0     | 10^4           | 10^12            |
| 15    | 10^8           | 10^4             |
| 30    | 10^4           | 10^12            |
| 45    | 10^8           | 10^4             |
</details>

f   
![](images/47ce6948a449e85a1994f1de094907d4b6ae7d84f00e679d8a1e67c92198fe43.jpg)

<details>
<summary>line</summary>

| θ (°) | g/2π (MHz) - Blue Line | g/2π (MHz) - Orange Line |
|-------|------------------------|--------------------------|
| 0     | 2.4                    | 0.0                      |
| 15    | 2.0                    | 1.0                      |
| 30    | 1.2                    | 1.8                      |
| 45    | 0.0                    | 2.0                      |
</details>

b   
![](images/dd18ac908de55e837510c6464cd63fc6293f26cb4387bf70be514793b9741f17.jpg)

<details>
<summary>line</summary>

| Δk (π/a) | Frequency (THz) |
| -------- | --------------- |
| -0.1     | 260             |
| 0        | 240             |
| 0.1      | 240             |
</details>

![](images/2cec5df24d67165e608fe4468f21cb15c53697adff4135b50dd5c0b80a8ec830.jpg)

<details>
<summary>line</summary>

| k (n/a) | Mode 1 Frequency (GHz) | Mode 2 Frequency (GHz) |
| ------- | ---------------------- | ---------------------- |
| -0.1    | 7.5                    | 8.2                    |
| 0       | 7.5                    | 8.2                    |
| 0.1     | 7.5                    | 8.2                    |
</details>

c   
![](images/50af5364651eeb5acd2f6763284fab9d84f4a8ea9213c91182ecc6a678b7db8d.jpg)

<details>
<summary>text_image</summary>

θ = 0°
Mode 1
Mode 2
0
1
Electrical field
Displacement
10⁻⁴
10⁻²
10⁰
</details>

e   
![](images/0e2480f50e1ffd8ced19eb5a8aa899c08d414a5ffaaaed809432ccb901a90223.jpg)

<details>
<summary>text_image</summary>

Mode 1 Symmetry
0°
15°
30°
45°
Polarization Topology
①
②
③
④
Mode 2 Symmetry
①
②
③
④
</details>

Fig. 1 Two-dimensional optomechanical crystal with mechanical BICs. a A schematic diagram of the two-dimensional silicon-on-insulator optomechanical crystal and its unit-cell structure. The optomechanical crystal could be rotated by an angle $\theta$ relative to the silicon crystal lattice. b Optical band structure near the $M$ point (top) and mechanical band structure near the $\Gamma$ point (bottom) for $\theta = 0$ . The relevant optical and mechanical bands are highlighted in color. c Simulated fundamental TE-like optical mode at the $M$ point (normalized total electric field) and mechanical modes 1 and 2 at the $\Gamma$ point (normalized total displacement field). d Simulated radiative quality factor of the mechanical modes with respect to $\theta$ . e Mechanical mode symmetry is illustrated using the normalized $z$ -direction displacement, momentum-space transverse polarization distribution, and the winding number. f Unit-cell optomechanical coupling of mechanical modes 1 and 2 with the fundamental TE-like optical mode.

Table 1 Summary of mechanical mode representation and optomechanical coupling. 

<table><tr><td></td><td>0°</td><td>15°</td><td>30°</td><td>45°</td></tr><tr><td>Mode 1</td><td> $A_{1}$ , quasi-BIC</td><td> $A_{2}$ , BIC</td><td> $A_{1}$ , quasi-BIC</td><td> $A_{2}$ , BIC</td></tr><tr><td>g</td><td>≠0</td><td>≠0</td><td>≠0</td><td>0</td></tr><tr><td>Mode 2</td><td> $A_{2}$ , BIC</td><td> $A_{1}$ , quasi-BIC</td><td> $A_{2}$ , BIC</td><td> $A_{1}$ , quasi-BIC</td></tr><tr><td>g</td><td>0</td><td>≠0</td><td>≠0</td><td>≠0</td></tr></table>

angles, the two modes belong to the A representation of the $C_{2}$ group and thus are quasi-BICs which only couple to the longitudinal acoustic waves. As a result, their quality factor over $10^{4}$ is significantly higher than unconfined modes.

The interaction between the optical and mechanical modes can be analyzed using mode symmetry (SI). Roughly, because the M-point optical mode energy density is even with respect to the xz-plane, mechanical modes that are odd with respect to the xz-plane, including the BICs for $\theta=0^{\circ}$ and $45^{\circ}$ , will not interact with the optical mode. For other cases, the optomechanical coupling could be nonzero (see Table 1 for a summary), thanks to the incommensurable symmetry of the optomechanical crystal and silicon lattice crystal. The bare optomechanical coupling of a unit cell, g, including both moving boundary and photoelastic effects, is calculated and plotted in Fig. 1f. For example, the BIC at $\theta=15^{\circ}$ has $g/2\pi=1.96$ MHz, with a contribution from the moving boundary and photoelastic effect of 0.61 and 1.35 MHz, respectively. The coupling g of the “snowflake” optomechanical crystal will increase with smaller air gaps. Here, with a practical air gap $w\approx30$ nm, the BIC optomechanical crystal achieves a

coupling rate (per unit cell) on par with one-dimensional suspended optomechanical crystals $^{21}$ .

Photonic crystal band-edge multimodes. The optomechanical crystal device is fabricated from a silicon-on-insulator wafer with a 220 nm thick silicon device layer and 3 $\mu$ m buried oxide. The air gap of the “snowflake” as small as 30 nm is achieved. The device consists of several functional components (Fig. 2). The hexagonal BIC optomechanical crystal has N unit cells along each edge and is surrounded by photonic crystal mirrors with a triangular lattice of cylindrical holes on five edges. The photonic crystal mirror, with the same lattice constant as the snowflake optomechanical crystal, is designed to have a complete TE-like bandgap with the center wavelength around 1550 nm to suppress the lateral radiation of the optical band-edge mode. The photonic crystal mirror is also slightly displaced to minimize the out-of-plane radiation due to the boundary effect on the finite-size optical mode (Fig. 2d). One side of the hexagonal snowflake crystal is connected to a waveguide terminated with an apodized grating coupler for coupling light from a single-mode optical fiber. This configuration makes the optomechanical crystal effectively a one-port device with (mode-dependent) external, i.e., to-waveguide, and intrinsic loss rate of $\kappa_{e}$ and $\kappa_{i}$ , respectively. The former can be controlled by the number of photonic crystal mirror layers between the waveguide and optomechanical crystal (Fig. 2c) to achieve different coupling conditions including the critical coupling, i.e., $\kappa_{e} \approx \kappa_{i}$ , which is desirable for sideband-resolved mechanical spectroscopy.

The device is measured using the setup shown in Fig. 3a. An angle-polished optical fiber is used to guide light via the on-chip apodized grating coupler $^{23,24}$ to the optomechanical crystal and

![](images/dbdc355cca7acf6f98d2628b95c3f0a0ede2b42cbd0fb92fe48a032c9a0a0042.jpg)

<details>
<summary>natural_image</summary>

Microscopic image of a nanoscale structure with scale bar (10 μm) and labeled components, no readable text or symbols beyond annotations
</details>

![](images/913ca7d6e306f1fe051a623f7c4112e0da823017fb38e89ed9cca731a36adfc5.jpg)

<details>
<summary>natural_image</summary>

Microscopic view of a periodic array of parallel grooves with 2 μm scale bar (no text or symbols)
</details>

![](images/2e1f835be8a86c7b7b528f2805ec68e3f1e70932c82a0733184a2f72e4dc5e28.jpg)

<details>
<summary>natural_image</summary>

Microscopic image showing a grid of circular features with a 1 μm scale bar and a highlighted region (no text or symbols beyond scale)
</details>

![](images/a84da13a31304e62d3c29a2872f2bd4041f8856d36e0de91cfe8817d87954d36.jpg)

<details>
<summary>text_image</summary>

d
a
d
0.5 µm
</details>

![](images/4226a6b2d5131f9d7ebab5098804d86b13648070338c3c6a32c4a8fab6705350.jpg)

<details>
<summary>natural_image</summary>

Microscopic image showing nanoscale structures with 200 nm scale bar (no text or symbols beyond label)
</details>

Fig. 2 Scanning electron microscopy of the optomechanical crystal device. a Full device. b Apodized grating coupler. c Junction between the optomechanical crystal and waveguide with tapered photonic crystal mirrors. d Boundary between the optomechanical crystal and photonic crystal mirror. The photonic crystal mirror is displaced by $d - a$ . e Zoom-in view of the optomechanical crystal.

![](images/296981f20969d129a10ad540500345c7a7af0178d6d474b8218e6b762cc33962.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Angle-polished Fiber"] --> B["FPC"]
    B --> C["Circulator"]
    C --> D["VOA"]
    D --> E["λ~1550 nm"]
    E --> F["SA"]
    E --> G["PD"]
    E --> H["EDFA"]
    H --> I["PD"]
```
</details>

![](images/e6a94498913bb67204652e915ff9a6a499e87ce2abc32f5c8299ae9fc961d6a0.jpg)

<details>
<summary>line</summary>

| Wavelength (nm) | Reflection |
| --------------- | ---------- |
| 1520            | 0.05       |
| 1530            | 0.15       |
| 1540            | 0.25       |
| 1550            | 0.10       |
| 1560            | 0.05       |
| 1570            | 0.02       |
</details>

![](images/f81eb71caf94e5d5664b48592b457b9ccc13038e266734c22eb1d7bb6deb1a22.jpg)

<details>
<summary>text_image</summary>

C
(1,1)
(1,3)
(2,1)
P
(3,1)
(3,3)
(4,1)
-1 0 1
φ (R)
</details>

![](images/acb6e41b114bb3ac7bb531893606b8aea42fb6f1cabdae2858f226890a322cc4.jpg)

<details>
<summary>scatter</summary>

| Number of mirror layers | κₑ/κᵢ |
| ------------------------ | ----- |
| 2                        | 0.1   |
| 2                        | 0.5   |
| 2                        | 0.8   |
| 3                        | 0.05  |
| 3                        | 0.3   |
| 4                        | 0.02  |
| 4                        | 0.1   |
| 5                        | 0.01  |
| 5                        | 0.05  |
| 6                        | 0.005 |
| 6                        | 0.1   |
| 7                        | 0.002 |
| 7                        | 0.05  |
</details>

![](images/ba8125958fdc0c6d9f5e8dc2cd2dce137d89cee56fc722f5dffadecc84282f99.jpg)

<details>
<summary>scatter</summary>

| Mirror displacement (nm) | Q_t     |
| ------------------------ | ------- |
| -80                      | 1000    |
| -75                      | 2000    |
| -70                      | 5000    |
| -65                      | 10000   |
| -60                      | 20000   |
| -55                      | 10000   |
| -50                      | 5000    |
</details>

Fig. 3 Multimode optical spectroscopy. a Schematic diagram of the measurement setup. The pump light is guided into the on-chip device using an angle-polished fiber. The reflected light is divided by a directional coupler for optical (10%) and mechanical (90%) spectroscopy. The optical signal is used to lock the laser-cavity detuning. VOA variable optical attenuator, FPC fiber polarization controller, EDFA erbium-doped fiber amplifier, PD photodetector, SA spectrum analyzer. b Optical spectrum of a device with multiple standing-wave resonances, one of which has a total quality factor of 21,000 (right). c Mode envelope $\phi (\mathbf{R})$ of standing-wave resonances with given orders corresponding to the dotted resonances in b. The arrow indicates the waveguide input. d $\kappa_{\mathrm{e}} / \kappa_{\mathrm{i}}$ for different numbers of nominal mirror layers at the junction between the optomechanical crystal and waveguide. For each number of mirror layers, multiple devices with different photonic crystal mirror displacements are measured. e Total optical quality factor versus the displacement of the photonic crystal mirror, i.e., $d - a$ . e is plotted using the same set of data as d.

![](images/310fa906e3abd56093c58e33c379f9271d8bf87167640473061e141618fdf9ac.jpg)

<details>
<summary>line</summary>

| Frequency (GHz) | 0°    | 15°   | 30°   | 45°   |
| --------------- | ----- | ----- | ----- | ----- |
| 7.4             | 1.02  | 1.02  | 1.02  | 1.02  |
| 7.6             | 1.02  | 1.02  | 1.02  | 1.02  |
| 7.8             | 1.02  | 1.02  | 1.02  | 1.02  |
| 8.0             | 1.02  | 1.02  | 1.02  | 1.02  |
| 8.2             | 1.02  | 1.02  | 1.02  | 1.02  |
| 8.4             | 1.02  | 1.02  | 1.02  | 1.02  |
</details>

![](images/2b57fd1161638728af9069c1eacb831aa071d07135bc4fff2429da68228bc679.jpg)

<details>
<summary>scatter</summary>

| θ (°) | Frequency (GHz) |
| ----- | --------------- |
| 0     | 7.7             |
| 15    | 7.6             |
| 30    | 7.9             |
| 45    | 8.5             |
</details>

![](images/067ec140ee9c8936374a5f9d25042ccaa9addccffc667bee61b0985276b903d9.jpg)

<details>
<summary>line</summary>

| θ (°) | g/2π (MHz) - Blue Line | g/2π (MHz) - Orange Line |
|-------|------------------------|--------------------------|
| 0     | 2.5                    | 0                        |
| 15    | 2.0                    | 1.0                      |
| 30    | 1.5                    | 1.5                      |
| 45    | 0                      | 2.0                      |
</details>

![](images/71b563cfff4d672601d397bf16bbd78e65b1eb12d60396f0c824b81cb948e0c5.jpg)

<details>
<summary>scatter</summary>

| θ (°) | Q^ε (blue) | Q^ε (orange) |
|-------|------------|--------------|
| 0     | ~200       | -            |
| 15    | ~500       | ~300         |
| 30    | ~300       | ~250         |
| 45    | -          | ~150         |
</details>

Fig. 4 Room temperature mechanical spectroscopy. a Measured mechanical noise power spectrum for one device each for $\theta = 0^{\circ}$ , $15^{\circ}$ , $30^{\circ}$ , and $45^{\circ}$ . b Mechanical frequency distribution of a group of five devices for each $\theta$ . Error bars indicate one standard deviation (error bars smaller than the symbol are not shown). Dashed lines are the simulated mechanical frequencies. c Unit-cell optomechanical coupling. The dashed lines correspond to simulated coupling for $w = 28$ nm (top) and $40$ nm (bottom). d Total quality factor of the mechanical modes. The dashed lines are the estimated bound and mean from the simulation of a $1 \times 4$ super-cell with disorders.

collect the reflected light. The angle of the fiber ( $\approx35.5^{\circ}$ ) and the apodized grating coupler are co-designed to realize an optimized coupling efficiency of 52% for 1550 nm light and 3-dB bandwidth of 40 nm (SI). Because of the reflection at the boundary of finite-size photonic/phononic crystals, band-edge standing-wave resonances will be formed $^{17,25-28}$ . The mode envelope of the standing-wave resonances can be approximated by the $(p,q)$ -th order eigenfunctions of a flat-top potential well within the boundary of the photonic/phononic crystal $^{29}$ . Figure 3b shows the optical reflection spectrum of a hexagonal optomechanical crystal with N=25, where a series of band-edge standing-wave resonances are observed. The major order p of a standing-wave resonance is identified from the group of resonances it belongs to and the minor order q is determined by the position of the resonance in a group. Because of the hyperbolic paraboloid topology of the band structure near the M point, resonances with smaller p and larger q will have shorter wavelengths. In addition, only q-odd resonances are observed because the excited waveguide mode is even with respect to the center of the waveguide. Order q=1 modes are expected to have the deepest resonance dip given they have the largest $\kappa_{e}$ close to the critical coupling. The mode envelope $\phi(\mathbf{R})$ of some standing-wave resonances are shown in Fig. 3c.

The number of photonic crystal mirror layers at the waveguide-optomechanical crystal junction is varied to tune $\kappa_{e}/\kappa_{i}$ , which is extracted by fitting the optical resonance spectrum using the normalized reflection coefficient of the one-port waveguide-coupled resonator, $R[\omega]=\left|1-\frac{\kappa_{e}}{i(\omega-\omega_{o})+(\kappa_{e}+\kappa_{i})/2}\right|^{2}$ . Figure 3d shows $\kappa_{e}/\kappa_{i}$ of the (1,1) standing-wave resonance of a group of devices, where a decreasing trend over the number of junction mirror layers is observed as a result of the reduced coupling between the waveguide mode and standing-wave resonances. The displacement of the photonic crystal mirror around the optomechanical crystal is also varied to optimize the total quality factor of the standing-wave resonance. As shown in Fig. 3e, a displacement of -60 nm from the nominal lattice constant is optimal, which is consistent with the simulation (SI). We note the variation of photonic crystal mirrors changes not only $\kappa_{e}$ but also $\kappa_{i}$ via the perturbation of the evanescent field, leading to fluctuations of $\kappa_{e}/\kappa_{i}$ . An optimized (2,1) optical resonance is shown in Fig. 3b with a total quality factor of 21,000 while being close to the critical coupling. The quality factor of the optical resonance and its variation is largely due to the disorder-induced scattering to the modes above the light cone $^{26}$ .

Room temperature mechanical spectroscopy. We perform the mechanical spectroscopy at room temperature using a blue-detuned laser with a frequency $\omega_{l} = \omega_{o} + \omega_{m}$ , where $\omega_{o}$ and $\omega_{m}$ are the frequencies of the optical and mechanical band-edge modes, respectively. We stabilize the laser-cavity-detuning by locking the device-reflected power with feedback control of the laser frequency (Fig. 3a). For each device, an optical standing-wave resonance with $Q_{t}$ about $2 \times 10^{4}$ and $\kappa_{e}/\kappa_{i}$ close to 1 is chosen, which means the device is operated near the sideband-resolved regime, i.e., $\omega_{m} \approx \omega_{o}/Q_{t}$ . The reflected pump light with sidebands due to the modulation of mechanical modes is sent to a high-speed photodetector. The beating between the pump and sidebands thus yields the mechanical spectrum which is observed by a spectrum analyzer.

Devices with different orientations between the optomechanical crystal and the silicon crystal lattice are fabricated to reveal the mechanical (quasi-)BICs and the impact of symmetry on the optomechanical interaction. Figure 4a shows the mechanical spectroscopy of a group of devices with $\theta = 0^{\circ}$ , $15^{\circ}$ , $30^{\circ}$ , and $45^{\circ}$ . The spectrum is normalized with respect to the total background noise. At $\theta = 0^{\circ}$ and $45^{\circ}$ , only the quasi-BIC mode is observed while the BIC mode is invisible because of the symmetry-inhibited optomechanical coupling. At $\theta = 15^{\circ}$ and $30^{\circ}$ , both BIC and quasi-BIC modes are detected. These observed mechanical modes are the (1,1) standing-wave resonance, while higher-order resonances are obscure because of both lower quality factors and optomechanical coupling. Figure 4b shows the frequency distribution of a group of devices. The occurrence of the mechanical BIC and quasi-BIC modes and their frequencies are consistent with the simulation.

Under a blue-detuned pump, the optomechanical interaction between the mechanical and optical resonances is described by a linearized two-mode squeezing Hamiltonian, $\hat{H} = G(\hat{a}^{\dagger}\hat{b}^{\dagger} + \hat{a}\hat{b})$ , where $\hat{a}^{\dagger}(\hat{b}^{\dagger})$ and $\hat{a}(\hat{b})$ are the creation and annihilation operators of the optical(mechanical) resonance. The parametrically-enhanced optomechanical mode coupling G of standing-wave resonances is given by

$$
G = \zeta g \sqrt {n _ {c}}, \tag {1}
$$

where $n_c$ is the average number of photons in one unit cell and $\zeta$ is a resonance-dependent $O(1)$ parameter due to the finite-mode-size correction (SI). Given the nature of the weak radiation-pressure force, the important coupling in generic optomechanical systems is the parametric mode coupling, which is related to the

cooperativity given by $C = 4G^{2}/\kappa\gamma$ . It is seen from Eq. 1 that the parametric mode coupling of a finite optomechanical crystal is primarily determined by the unit-cell coupling and the per unit cell photon number, and is independent of the size of the crystal. In other words, despite that the bare mode coupling of optomechanical crystals roughly scales as 1/N, it can be compensated by the large number of photons available which scales with $N^{2}$ . The per unit cell photon number is largely constrained by the thermo-optic effect and the heat capacity of the unit cell, which the slab-on-substrate structure can ameliorate, especially when the thermal conductivity of the substrate is comparable to the slab. In our experiments, the device is typically operated under a pump power P of a few mW and $n_{c} = \kappa_{e}P/((\omega_{m})^{2} + (\kappa/2)^{2})/\frac{3\sqrt{3}}{2}N^{2}$ is on the order of 10-20. The unit-cell optomechanical coupling g is extracted from the noise power spectrum (see Methods and SI) and plotted in Fig. 4c. Large optomechanical coupling $g/2\pi \approx 2.5$ MHz is observed for the mechanical BIC. The deviation from the simulated value could be due to the variation of the actual snowflake gap size and disorders in the optomechanical crystal. Despite being unreleased and two-dimensional, the optomechanical interaction (per unit cell) in the BIC optomechanical crystal is on par with the suspended low-dimensional optomechanical crystal devices, such as nanobeams.

The quality factor of the observed mechanical modes is shown in Fig. 4d. There are two main factors limiting the mechanical quality factor of the current device. First, the size of the optomechanical crystal is N=25, corresponding to a finite wavevector of $k \approx 2\pi/(3Na)$ for the fundamental standing-wave resonance. The radiation quality factor of mechanical standing-wave resonances with finite wavevectors decreases drastically as they deviate from the BIC at the $\Gamma$ point $^{17}$ . Second, the fabricated optomechanical crystal has critical dimensions as small as 30 nm, which cause random variations among unit cells. These disorders induce scattering between different orders of band-edge resonances, which effectively introduces more radiation channels to a given resonance and degrades its quality factor $^{26,30,31}$ . The lateral radiation for this size of the crystal is not a dominant loss according to the simulation and can be optimized with a phononic crystal mirror. In addition, at room temperature, the quality factor of GHz-frequency mechanical modes is ultimately limited by material absorption to around 1000–2000 $^{2}$ . Because of these, there is no significant statistical difference between the quality factor of BIC and quasi-BIC modes. The observed quality factor is verified with numerical simulations of finite super-cells with realistic disorders (SI). However, the mechanical radiation quality factor will increase with the size of optomechanical crystals $^{17}$ , which will be critical to low-temperature measurements when the material absorption is suppressed.

# Discussion

In summary, we have realized the first two-dimensional slab-on-substrate optomechanical crystals with mechanical BICs. Guided by a symmetry-based design approach, this architecture offers optomechanical interaction (per unit cell) on par with suspended optomechanical crystal devices. The two-dimensional optomechanical crystal with tunable symmetry provides an arena for the exploration of rich multimode physics $^{32,33}$ . In addition, the cavity-less optomechanical crystal might realize Floquet topological physics beyond the tight-binding model $^{22}$ . The benefit of the slab-on-substrate device architecture in terms of heat dissipation is obscured at room temperature, especially for optomechanical crystals with a large area-to-perimeter ratio, because of the low thermal conductivity of the oxide substrate two orders smaller than silicon. However, such benefit will become evident at low temperatures (<1 K), relevant for quantum experiments involving GHz mechanical modes, or in a different material system, when/where the thermal conductivity of the substrate and slab becomes comparable $^{34,35}$ . Besides the suppressed material-absorption loss at low temperatures, the mechanical quality factor could be enhanced using the merging BIC mechanism $^{26}$ and implementing appropriate phononic crystal mirrors. As a consequence, slab-on-substrate BIC optomechanical crystals with improved optical and mechanical losses could be unique at low temperatures for exploration of modalities including phonon sensing and macroscopic mechanical oscillators in the quantum regime $^{36,37}$ .

# Methods

Device fabrication. Devices are fabricated in silicon-on-insulator microchips (220 nm silicon device layer and 3 $\mu$ m buried oxide layer) using electron beam lithography with ZEP520A mask, followed by inductively coupled plasma reactive ion etch of silicon using $SF_{6}$ and $CHF_{3}$ .

Mechanical noise power spectrum. The total noise power spectral density measured by the photodetector is given by refs. $^{16,38}$

$$
S [ \omega ] = S _ {e} + \frac {G _ {e} ^ {2}}{R _ {I}} \left(S _ {\mathrm{EDFA}} + G _ {\mathrm{EDFA}} ^ {2} S _ {\mathrm{SN}} ^ {2} (1 + \eta \frac {\kappa_ {e}}{\kappa} \frac {8 G ^ {2}}{\kappa} S _ {m} [ \omega ])\right) \tag {2}
$$

where $S_{e}$ is the electronic noise of the detector, $S_{EDFA}$ is the noise of EDFA,

$S_{\mathrm{SN}} = \sqrt{2P_{\mathrm{out}}\hbar\omega_{I}}$ is the optical shot noise, $S_{m}[\omega]$ is the mechanical noise spectrum, $\kappa = \kappa_{i} + \kappa_{e}$ is the total loss rate of the optical resonance, $G$ is the parametrically-enhanced optomechanical coupling, $\eta$ is the total detection efficiency, $G_{\mathrm{EDFA}}$ is the EDFA gain, $G_{e}$ is the detector gain factor from optical power to voltage, and $R_{I}$ is the input impedance of the spectrum analyzer. The optically-transduced mechanical noise spectrum is given by

$$
S _ {m} [ \omega ] = \frac {1}{2} \left(\frac {\gamma (\bar {n} _ {m} + 1)}{\left(\omega_ {m} - \omega\right) ^ {2} + (\gamma / 2) ^ {2}} + \frac {\gamma \bar {n} _ {m}}{\left(\omega_ {m} + \omega\right) ^ {2} + (\gamma / 2) ^ {2}}\right) \tag {3}
$$

with $\bar{n}_{m}=\frac{k_{\mathrm{e}}T}{\hbar\omega_{m}}$ the thermal occupation of the mechanical mode. The total detection efficiency is $\eta=\eta_{cpl}\eta_{t}\eta_{det}$ , where $\eta_{cpl}$ is the coupling efficiency of the grating coupler at the pump wavelength, $\eta_{t}$ is the total transmission efficiency in the optical fiber path from the chip to the detector, and $\eta_{det}$ is the quantum efficiency of the photodetector. We measured $\eta_{t}=0.80$ and $\eta_{cpl}\approx0.5$ depending on the pump wavelength, while $\eta_{det}=0.68$ is given by the photodetector, which results in $\eta\approx0.27$ . $S_{e}$ is determined by blocking the light and $\frac{G_{e}^{2}}{R_{i}}G_{EDFA}^{2}S_{SN}^{2}$ is determined by removing EDFA while keeping the optical power incident onto the photodetector the same. Then the measured noise power spectrum is fitted using Eqs. 2 and 3, with $\omega_{m},\gamma$ , and G the only fitting parameters. The unit-cell optomechanical coupling is calculated from $g=G/(\zeta\sqrt{n_{c}})$ , using $n_{c}=\kappa_{e}P/((\omega_{m})^{2}+(\kappa/2)^{2})/\frac{3\sqrt{3}}{2}N^{2}$ and $\zeta$ depending on the optical standing-wave resonance that is used for the mechanical spectroscopy (SI).

# Data availability

Data supporting the findings of this study are available within the article and its Supplementary Information, or from the corresponding author upon reasonable request.

Received: 22 January 2022; Accepted: 25 May 2022;

Published online: 08 June 2022

# References

1. Aspelmeyer, M., Kippenberg, T. J. & Marquardt, F. Cavity optomechanics. Rev. Mod. Phys. 86, 1391 (2014).   
2. Eichenfield, M., Chan, J., Camacho, R. M., Vahala, K. J. & Painter, O. Optomechanical crystals. Nature 462, 78–82 (2009).   
3. Safavi-Naeini, A. H. et al. Two-dimensional phononic-photonic band gap optomechanical crystal cavity. Phys. Rev. Lett. 112, 153603 (2014).   
4. Ren, H. et al. Two-dimensional optomechanical crystal cavity with high quantum cooperativity. Nat. Commun. 11, 1–10 (2020).   
5. MacCabe, G. S. et al. Nano-acoustic resonator with ultralong phonon lifetime. Science 370, 840–843 (2020).   
6. Chan, J. et al. Laser cooling of a nanomechanical oscillator into its quantum ground state. Nature 478, 89–92 (2011).

7. Marinković, I. et al. Optomechanical bell test. Phys. Rev. Lett. 121, 220404 (2018).   
8. Wallucks, A., Marinković, I., Hensen, B., Stockill, R. & Gröblacher, S. A quantum memory at telecom wavelengths. Nat. Phys. 16, 772–777 (2020).   
9. Ludwig, M. & Marquardt, F. Quantum many-body dynamics in optomechanical arrays. Phys. Rev. Lett. 111, 073603 (2013).   
10. Brendel, C., Peano, V., Painter, O. J. & Marquardt, F. Pseudomagnetic fields for sound at the nanoscale. Proc. Natl Acad. Sci. USA 114, E3390–E3395 (2017).   
11. Brendel, C., Peano, V., Painter, O. & Marquardt, F. Snowflake phononic topological insulator at the nanoscale. Phys. Rev. B 97, 020102 (2018).   
12. Ren, H. et al Topological phonon transport in an optomechanical system. Preprint at arXiv:2009.06174 (2020).   
13. Sarabalis, C. J., Dahmani, Y. D., Patel, R. N., Hill, J. T. & Safavi-Naeini, A. H. Release-free silicon-on-insulator cavity optomechanics. Optica 4, 1147–1150 (2017).   
14. Qi, R. et al. Nonsuspended optomechanical crystal cavities using as 2 s 3 chalcogenide glass. Photonics Res. 9, 893–898 (2021).   
15. Zhang, J. et al. Silicon-on-insulator optomechanical microresonator with tight photon and phonon confinement. Preprint at arXiv:2103.08465 (2021).   
16. Meenehan, S. M. et al. Silicon optomechanical crystal resonator at millikelvin temperatures. Phys. Rev. A 90, 011803 (2014).   
17. Tong, H., Liu, S., Zhao, M. & Fang, K. Observation of phonon trapping in the continuum with topological charges. Nat. Commun. 11, 1–7 (2020).   
18. Chen, Y. et al. Mechanical bound state in the continuum for optomechanical microresonators. N. J. Phys. 18, 063031 (2016).   
19. Yu, Y., Xi, X. & Sun, X. Observation of bound states in the continuum in a micromechanical resonator. Preprint at arXiv:2109.09498 (2021).   
20. Zhao, M. & Fang, K. Mechanical bound states in the continuum for macroscopic optomechanics. Opt. Express 27, 10138-10151 (2019).   
21. Chan, J., Safavi-Naeini, A. H., Hill, J. T., Meenehan, S. & Painter, O. Optimized optomechanical crystal cavity with acoustic radiation shield. Appl. Phys. Lett. 101, 081115 (2012).   
22. Fang, K. & Wang, Y. Anomalous quantum hall effect of light in bloch-wave modulated photonic crystals. Phys. Rev. Lett. 122, 233904 (2019).   
23. Snyder, B. & O'Brien, P. Packaging process for grating-coupled silicon photonic waveguides using angle-polished fibers. IEEE Trans. Compon., Packaging Manuf. Technol. 3, 954–959 (2013).   
24. Li, C. et al. Silicon photonics packaging with lateral fiber coupling to apodized grating coupler embedded circuit. Opt. Express 22, 24235–24240 (2014).   
25. Hood, J. D. et al. Atom–atom interactions around the band edge of a photonic crystal waveguide. Proc. Natl Acad. Sci. USA 113, 10507–10512 (2016).   
26. Jin, J. et al. Topologically enabled ultrahigh-Q guided resonances robust to out-of-plane scattering. Nature 574, 501–504 (2019).   
27. Chua, S.-L., Lu, L., Bravo-Abad, J., Joannopoulos, J. D. & Soljačić, M. Larger-area single-mode photonic crystal surface-emitting lasers enabled by an accidental dirac point. Opt. Lett. 39, 2072–2075 (2014).   
28. Chen, Z. et al. Analytical theory of finite-size photonic crystal slabs near the band edge. Opt. Express 30, 14033-14047 (2022).   
29. Xu, T., Yang, S., Nair, S. V. & Ruda, H. Confined modes in finite-size photonic crystals. Phys. Rev. B 72, 045126 (2005).   
30. Regan, E. C. et al. Direct imaging of isofrequency contours in photonic structures. Sci. Adv. 2, e1601591 (2016).   
31. Ni, L., Jin, J., Peng, C. & Li, Z. Analytical and statistical investigation on structural fluctuations induced radiation in photonic crystal slabs. Opt. Express 25, 5580–5593 (2017).   
32. Nielsen, W. H. P., Tsaturyan, Y., Møller, C. B., Polzik, E. S. & Schliesser, A. Multimode optomechanical system in the quantum regime. Proc. Natl Acad. Sci. USA 114, 62–66 (2017).

33. Renninger, W., Kharel, P., Behunin, R. & Rakich, P. Bulk crystalline optomechanics. Nat. Phys. 14, 601–607 (2018).   
34. Zeller, R. & Pohl, R. Thermal conductivity and specific heat of noncrystalline solids. Phys. Rev. B 4, 2029 (1971).   
35. Thompson, J. & Younglove, B. Thermal conductivity of silicon at low temperatures. J. Phys. Chem. Solids 20, 146–149 (1961).   
36. Chu, Y. et al. Quantum acoustics with superconducting qubits. Science 358, 199–202 (2017).   
37. Kotler, S. et al. Direct observation of deterministic macroscopic entanglement. Science 372, 622–625 (2021).   
38. Safavi-Naeini, A. H. et al. Laser noise in cavity-optomechanical cooling and thermometry. N. J. Phys. 15, 035007 (2013).

# Acknowledgements

This work is supported by the U.S. National Science Foundation (Grant No. 1944728 and 2137642) and the Office of Naval Research (Grant No. N00014-21-1-2136).

# Author contributions

S.L. and H.T. designed the device. H.T. manufactured the device. S.L. and H.T. performed the experiment. S.L., H.T., and K.F. analyzed the data. All authors contributed to the writing of the manuscript.

# Competing interests

The authors declare no competing interests.

# Additional information

Supplementary information The online version contains supplementary material available at https://doi.org/10.1038/s41467-022-30965-6.

Correspondence and requests for materials should be addressed to Kejie Fang.

Peer review information Nature Communications thanks Jie Luo and the other anonymous reviewer(s) for their contribution to the peer review of this work.

Reprints and permission information is available at http://www.nature.com/reprints

Publisher's note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

![](images/70a947a426f0c6e57d57330d517eb41007a526a4fcc6dfab4a9e72fdc8e32a20.jpg)

Open Access This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing,

adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons license, and indicate if changes were made. The images or other third party material in this article are included in the article's Creative Commons license, unless indicated otherwise in a credit line to the material. If material is not included in the article's Creative Commons license and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.

© The Author(s) 2022, corrected publication 2022