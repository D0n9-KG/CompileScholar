ARTICLE

https://doi.org/10.1038/s41467-019-10852-3

OPEN

# Phononic integrated circuitry and spin-orbit interaction of phonons

Wei Fu $^{1,2}$ , Zhen Shen $^{1,2}$ , Yuntao Xu $^{1}$ , Chang-Ling Zou $^{1}$ , Risheng Cheng $^{1}$ , Xu Han $^{1}$ & Hong X. Tang $^{1}$

High-index-contrast optical waveguides are crucial for the development of photonic integrated circuits with complex functionalities. Despite many similarities between optical and acoustic waves, high-acoustic-index-contrast phononic waveguides remain elusive, preventing intricate manipulation of phonons on par with its photonic counterpart. Here, we present the realization of such phononic waveguides and the formation of phononic integrated circuits through exploiting a gallium-nitride-on-sapphire platform, which provides strong confinement and control of phonons. By demonstrating key building blocks analogous to photonic circuit components, we establish the functionality and scalability of the phononic circuits. Moreover, the unidirectional excitation of propagating phononic modes allows the exploration of unconventional spin-orbit interaction of phonons in this circuit platform, which opens up the possibility of novel applications such as acoustic gyroscopic and non-reciprocal devices. Such phononic integrated circuits could provide an invaluable resource for both classical and quantum information processing.

Phononics—the study of vibrating structure—has become an emerging research field, owing to recent progresses in micro/nano-fabrication and investigation of phononic wave dynamics, phononic crystals, and acoustic metamaterials $^{1-11}$ . The engineered phononic structures not only provide stronger phonon interaction with light and matter by reducing the device footprint, but also allow phonon reservoir engineering $^{12,13}$ to increase phonon lifetime, boosting the performance of phononic devices to an unprecedented level $^{14}$ . Recent experiments of quantum acoustics have pushed the study of matter-phonon and photon–phonon interactions to the single-quantum level, making possible phononics-based hybrid quantum systems and quantum memories $^{15-22}$ . Compared to its photonic and electronic counterparts, phononic devices have the desired property that the GHz frequency range corresponds to wavelengths on the order of m, which uniquely bridges frequency and wavelength gaps between optical and electrical circuits $^{23-25}$ .

Despite the efforts to develop novel phononic structures in micro/nano-electromechanics and optomechanics $^{26-37}$ , scalable phononic integrated circuits (PnIC) remain largely unexplored compared to their electric and photonic counterparts. A critical step towards PnICs with complex functionalities is the implementation and integration of key building blocks—phononic waveguides, multiport phononic structures, evanescently coupled phononic resonators, and scalable input/output couplers—on a single chip. Recent experiments on hybrid system employing phononic wires—the core of a PnIC—are encouraging $^{4-11}$ , however, challenges remain for each of these systems to form a more complex circuit.

In this article, we present an experimental demonstration of a PnIC, as an analog to the photonic integrated circuit (PIC). A notional schematic of the PnIC chip is shown in Fig. 1a. An incoming radio-frequency (RF) signal from a transmission line is sent to an interdigital transducer (IDT), which converts RF photons to phonons and vice versa. The generated phonons are then confined, guided, and routed on the top layer of the chip by the phononic waveguides and go through processing circuits such as an array of ring resonators. Finally, the transmitted phonons are collected and converted back to RF photons by the output IDTs. The system described above can be summarized to three critical components: (1) scalable input and output ports; (2) phononic circuitry to confine, guide, and route phonons; (3) ring resonators to store, enhance, and modulate phonons. Interestingly, with such a PnIC, the highly confined, unidirectionally excited whispering-gallery modes carry both orbital angular moment and spin. We demonstrate, for the first time, the spin-orbit interaction (SOI) of propagating phonons. The phononic SOI brings new concept in designing of phononic system at sub-wavelength scale, opens possibilities of chiral phonon-matter interaction and non-reciprocal phononic devices $^{38}$ .

# Results

Phononic strip waveguide. At the core of the proposed circuits is low loss, high-acoustic-index-contrast, single-mode phononic waveguides for routing phonons between localized phononic components such as phononic resonators. For short-distance acoustic wave propagation, it is possible to create such waveguides from suspended structures using surface micromachining techniques, such as phononic crystal waveguides supported in a suspended membrane $^{4-8}$ and suspended wires supported by localized tethers $^{9-11}$ . However, due to the three-dimensional nature of these structures, it is a major challenge to freely lay out waveguide patterns with a complexity similar to that of photonic or electrical wires on a planar chip. Here, we present an alternative phononic architecture with phononic waveguides that harness acoustic velocity mismatch $^{39}$ , as shown in Fig. 1b, in a way similar to high-index contrast photonic waveguides in PIC.

We survey a number of material platforms for building acoustic waveguide structures considering the following criteria: (1) the top layer has an acoustic velocity lower than that of the substrate for a strong confinement of acoustic waves; (2) the top layer is piezoelectric to allow efficient coupling to RF input/output fields; (3) the material selected is compatible with standard wafer-scale semiconductor fabrication processing. Table 1 summarizes commonly utilized piezoelectric materials and their substrates. Gallium nitride (GaN)-on-sapphire (GNOS) $^{40}$ stands out as a unique material platform that meets all these requirements. The ever-growing demand for light-emitting devices and high-power electronics has led GaN to become a mature semiconductor material. Moreover, the GNOS platform is compatible with standard semiconductor fabrication procedures (see Methods and Supplementary Note 2 for details about simulations and fabrications).

The simulation results by the finite-element method in Fig. 1c shows cross-sectional elastic energy distribution, where the

![](images/de0042e9f20a0092c112bb7be180ff86ec37093e218af53e99dc79ad02dcf7a7.jpg)

<details>
<summary>text_image</summary>

a
RF input
Interdigital transducer
RF output
b
GaN
h
w
Sapphire
c
0 dB
-30 dB
Energy density
</details>

Fig. 1 Phononic integrated circuit. a A notional schematic of a phononic circuit. Input radio-frequency (RF) photons are first converted into phonons by an interdigital transducer (IDT) through the piezoelectric effect of the top gallium nitride (GaN) epi-layer. The phonons are confined, routed through phononic circuit and received by output IDTs. b A schematic cross-section of GaN-on-sapphire strip waveguide with thickness h and width w. c Simulated cross-sectional acoustic energy distribution of a Rayleigh-like mode shown in logarithmic scale, indicating confinement of acoustic wave in the waveguide. The white dashed line marks where the energy density drops to $10^{-3}$ , compared with the highest energy density

phonon field is mostly confined in the GaN waveguide and decays exponentially with increasing depth into the substrate. Without loss of generality, we have taken Rayleigh-like mode (to be introduced in the following section) as an example. As indicated by the white dashed line, elastic energy decays to $10^{-3}$ compared with the highest energy density within a distance of only one wavelength.

In the experiment, we first consider a phononic waveguide device (Fig. 2c) optimized to support acoustic waves with wavelength around 50 $\mu$ m (frequency around 100 MHz). The waveguide is 5 $\mu$ m tall and 50 $\mu$ m wide, which provides strong confinement while suppressing higher-order modes. As illustrated by the schematic in Fig. 2a, the phonon mode is actuated by an IDT at the end of the phononic waveguide. The IDT is composed of two comb electrodes deposited on the surface of GNOS, with parallel fingers interdigitated to provide a periodically distributed electric field. The acoustic wave is actuated by the electric field through piezoelectric effect, and the maximum coupling efficiency is achieved when the acoustic wavelength matches the period of the IDT fingers. In our experiments, the efficiency of a single IDT is around -35 dB.

To experimentally characterize the phononic devices, we employ a home-built vibrometer to map out both the amplitude

Table 1 Survey of acoustic waveguide and substrate materials 

<table><tr><td>Material</td><td>Longitudinal wave speed (m s-1)</td><td>Transverse wave speed (m s-1)</td><td>Piezoelectricity</td></tr><tr><td> $GaN^{54}$ </td><td>7350</td><td>4578</td><td>Yes</td></tr><tr><td> $AlN^{54}$ </td><td>10,169</td><td>6369</td><td>Yes</td></tr><tr><td> $Quartz^{55}$ </td><td>5700</td><td>3158</td><td>Yes</td></tr><tr><td> $Sapphire^{56}$ </td><td>10,658</td><td>5796</td><td>No</td></tr><tr><td> $Silicon^{57}$ </td><td>8433</td><td>5843</td><td>No</td></tr><tr><td> $Silica^{55}$ </td><td>5800</td><td>3700</td><td>No</td></tr><tr><td> $Diamond^{58}$ </td><td>18,000</td><td>12000</td><td>No</td></tr></table>

and phase of the phonon modes' out-of-plane displacement $u$ (see Methods and Supplementary Note 2 for details of the vibrometer). As indicated by the black box in the measurement schematic (Fig. 2a), we scan a chip area of $400\mu \mathrm{m}\times 400\mu \mathrm{m}$ with input RF excitation frequency at $102\mathrm{MHz}$ for the maximum IDT efficiency. The reconstructed out-of-plane displacement of the acoustic waveguide mode is shown in Fig. 2d. Good agreement between the measured (purple dots) and simulated (red line) amplitude cross the waveguide in Fig. 2b confirms that the acoustic wave is indeed confined. The amplitude along the center of the waveguide is plotted in Fig. 2e; no obvious decay is observed within the scanning area. The slight periodic oscillation of the amplitude is due to the reflection from the output IDT, which can be fitted by taking into account a reflection factor of $10\%$ (red line in Fig. 2e). Figure 2f shows the phase distribution, which linearly increases along the propagation direction, revealing the itinerant nature of phonons in the waveguide.

Phononic ring resonator. Ring resonators are essential for PnIC and can enable narrow band signal filtrations, phonon bufferings and memories, as well as enhancement of phonon–matter/photon interaction $^{41-43}$ . For unidirectional actuation of the phonon modes, directional coupler structures are designed, simulated (see Supplementary Note 1), and here experimentally demonstrated in the form of wrap-around coupler. Figure 3a presents an SEM image of phononic ring resonators coupled with a wrap-around bus phononic waveguide, which is fabricated from a 5- $\mu$ m-thick GaN film to support phonon modes at a wavelength of around 25 $\mu$ m. By adjusting the vibrometer laser focal point on the ring and simultaneously sweeping the RF input frequency, we acquire an intracavity displacement spectrum as shown in Fig. 3b. Similar to photonic rings that support both transverse electric (TE) and transverse magnetic (TM) modes $^{44}$ , phononic rings support phonon modes with two orthogonal polarizations: Rayleigh-like modes (labeled by magenta arrows) and Love-like modes (labeled by blue arrows). Both families of modes show a free spectral range

![](images/56e3b61134602a9ffa52860a37a691d000f9a16ceeb5b39d5233ad44f07822fb.jpg)

<details>
<summary>scatter</summary>

| Panel | Description                     | X (μm) Range       | Y (μm) Range       | Intensity Level |
|-------|---------------------------------|--------------------|--------------------|-----------------|
| a     | Input RF signal                 | -200 to 0          | 0 to 200           | Low             |
| b     | Amplitude (dip)                | 0 to 200           | 0 to 200           | Medium          |
| c     | GaN / Sapphire                  | 50 μm              | 300–400            | High            |
| d     | Image area: Imaging area        | -200 to 200        | 0 to 400            | Medium          |
| e     | Phase angle: -180 to 180        | 100–200            | 100–350            | High            |
| f     | Phase angle: -180 to 180        | -180 to 180        | 180–350            | Medium          |
</details>

Fig. 2 Characterization of phononic waveguide and identification of a traveling Rayleigh-like mode. a The measurement scheme. c An scanning electron microscope (SEM) image of a phononic waveguide. b, d-f, The out-of-plane (z-direction) displacement of a traveling Rayleigh-like mode. The data are measured by sending RF signal at 102 MHz into the IDT and scanning the vibrometer focal point in an area of $400 \mu m \times 400 \mu m$ as illustrated in a. The reconstructed image of instant displacement is shown in d. The amplitude or phase at cross-sections of the two-dimensional scan are plotted in b, e, and f and taken at correspondingly colored lines in d. The magenta line in b represents the simulated displacement amplitude cross the waveguide, and the magenta lines in e and f are fitted to traveling acoustic wave with weak reflection (10% reflection)

![](images/4db1ff21d015cc9d9f1182f8ac3a447a4c64485b5603e4cc8ce50b43e25f6bf5.jpg)  
Fig. 3 Phononic ring resonator and mode identification. a An SEM image of arrays of phononic ring resonators. b A typical spectrum of the phononic ring resonator measured by the vibrometer with the focal point adjusted to the ring. Two sets of modes are observed, corresponding to Rayleigh-like and Love-like mode, marked with magenta and blue arrows, respectively. c The vibrometer image of a Rayleigh-like mode in a waveguide coupled ring resonator (only the displacement on the ring is presented). d, e The vibrometer images of the Rayleigh-like and Love-like modes. f, g Simulated z-direction displacement profile of Rayleigh-like and Love-like modes, in excellent agreement with the experimental results. h, i Simulated 3-d mode profiles of Rayleigh-like and Love-like mode, respectively

(FSR) of around 1.31 MHz, corresponding to a group velocity ( $v_{g}$ ) of around 4180 $\mu$ m s $^{-1}$ .

To further characterize the modes and determine their polarization, we actuate the ring at different resonant frequencies and image the resonance mode profiles using the vibrometer. Figure 3c shows a displacement profile of the ring excited at 193.085 MHz, where a Rayleigh-like resonance can be clearly identified with an azimuthal number m=115. Zoomed-in views of the Rayleigh-like mode and a Love-like mode excited at 192.575 MHz are shown in Fig. 3d, e, respectively, in exceptional agreement with the simulation results (Fig. 3f, g). The three-dimensional simulated mode profiles of both modes are presented in Fig. 3h, i: the Rayleigh-like mode is dominated by the out-of-plane displacement, while the Love-like mode is dominated by the in-plane displacement. Moreover, we study the intrinsic and the external decay rates of the phononic rings with different radii and coupling gaps. The highest Q of $2.5 \times 10^{4}$ is obtained at room temperature, which is significantly improved to $3.2 \times 10^{5}$ at 50 mK measured in a dilution refrigerator. We can further estimate the phonon loss rate per unit length by the Qs: the loss rate is $\alpha = \frac{\omega}{Qv_{g}} = 0.5$ dB per cm at room temperature and 0.03 dB per cm at 50 mK (see more details of decay rate study in the Supplementary Note 2).

Spin-orbit interactions of phonons. Similar to optical whispering-gallery modes $^{45}$ , phonon modes traveling in a ring resonator carry orbital angular momentum (OAM) L, as illustrated in Fig. 4b. Depending on the propagation directions of the mode (clockwise (CW) or counter-clockwise (CCW)), L can point along negative or positive z-axis, respectively. At the same time, traveling Love-like modes also carry spin: the sidewall surface and the bending-induced asymmetry cause hybridization between transverse and longitudinal displacements, resulting in a circular-like trajectory, as illustrated in the enlarged simulation plot in Fig. 4c. We define the left-hand (CCW) and the right-hand (CW) circular motions as two 'eigenstates' of the spin, and introduce chirality $\chi$ to characterize the collective spin motion of all particles in the ring (see Supplementary Note 1 for detailed definition of OAM and chirality of phonon modes). Moreover, the spin and the OAM of phonons are coupled due to the sub-wavelength scale confinement. Figure 4c showcases a point (indicated by purple dot) possess right-hand (left-hand) spin when phonons propagate in CW (CCW) direction. As a result of this spin-orbit interaction (SOI), phonons circulating in a ring resonator possess chiralities that are in the same direction with the OAM, also known as spin-orbit locking $^{46}$ .

In order to experimentally demonstrate the SOI of phonons, we harness the gyroscopic effect to induce a spin-dependent velocity change of the acoustic wave. The spin-dependent gyroscopic effect of phonon originates from the Coriolis forces on particles in the presence of rotation $F = -2 \, m\Omega \times v$ , where $\Omega$ is the rotation with respect to the inertial frame of reference and v is the velocity of the particles. Particles spinning in parallel or anti-parallel direction of the rotation experience centrifugal or centripetal Coriolis force, respectively. For Love-like modes, the Coriolis force causes resonant frequency shift depending on the chirality of the mode $\omega = \omega_{0} - \chi \cdot \Omega^{47}$ (see Supplementary Note 1 for details). Therefore, by observing the OAM-dependent frequency shift (Fig. 4e) under a rotation along $\chi$ direction, we are able to confirm that the direction of chirality is the same with that of the OAM, manifesting the spin–orbit locking and the SOI of phonons.

In our experiment, we use a phononic ring device shown in Fig. 4a to test the OAM-dependent response of phononic modes.

a   
![](images/e81c63c3b94d36e355371a8ce234d98af3c328288710c012858d4c9342c84622.jpg)

<details>
<summary>text_image</summary>

Shield
1 mm
</details>

b   
![](images/a30951d06bdca477995fd745f24504240039a592113316effef3eb432387a63b.jpg)

<details>
<summary>text_image</summary>

OAM
</details>

C   
![](images/d42333a7a5e1579791dadc3a7433d7ec83e5392e049a85e5b478e08935d7ea26.jpg)

<details>
<summary>text_image</summary>

Spin and spin-OAM locking
A
</details>

d   
![](images/c7bcc3f281565a1972c48c1e2316efd19a55eda3341f1c4f303b9a0936248d05.jpg)

<details>
<summary>line</summary>

| Frequency (Hz) | Rotation Rate (Hz) |
| -------------- | ------------------ |
| 0              | 0                  |
| 1              | 0.5                |
| 2              | 0                  |
| 3              | -0.5               |
| 4              | 0                  |
| 5              | 0.5                |
| 6              | 0                  |
| 7              | -0.5               |
| 8              | 0                  |
| 9              | 0.5                |
| 10             | 0                  |
| 11             | -0.5               |
| 12             | 0                  |
| 13             | 0.5                |
| 14             | 0                  |
| 15             | -0.5               |
| 16             | 0                  |
| 17             | 0.5                |
| 18             | 0                  |
| 19             | -0.5               |
| 20             | 0                  |
| 21             | 0.5                |
| 22             | 0                  |
| 23             | -0.5               |
| 24             | 0                  |
| 25             | 0.5                |
| 26             | 0                  |
| 27             | -0.5               |
| 28             | 0                  |
| 29             | 0.5                |
| 30             | 0                  |
| 31             | -0.5               |
| 32             | 0                  |
| 33             | 0.5                |
| 34             | 0                  |
| 35             | -0.5               |
| 36             | 0                  |
| 37             | 0.5                |
| 38             | 0                  |
| 39             | -0.5               |
| 40             | 0                  |
| 41             | 0.5                |
| 42             | 0                  |
| 43             | -0.5               |
| 44             | 0                  |
| 45             | 0.5                |
| 46             | 0                  |
| 47             | -0.5               |
| 48             | 0                  |
| 49             | 0.5                |
| 50             | 0                  |
| 51             | -0.5               |
| 52             | 0                  |
| 53             | 0.5                |
| 54             | 0                  |
| 55             | -0.5               |
| 56             | 0                  |
| 57             | 0.5                |
| 58             | 0                  |
| 59             | -0.5               |
| 60             | 0                  |
| 61             | 0.5                |
| 62             | 0                  |
| 63             | -0.5               |
| 64             | 0                  |
| 65             | 0.5                |
| 66             | 0                  |
| 67             | -0.5               |
| 68             | 0                  |
| 69             | 0.5                |
| 70             | 0                  |
| 71             | -0.5               |
| 72             | 0                  |
| 73             | 0.5                |
| 74             | 0                  |
| 75             | -0.5               |
| 76             | 0                  |
| 77             | 0.5                |
| 78             | 0                  |
| 79             | -0.5               |
| 80             | 0                  |
| 81             | 0.5                |
| 82             | 0                  |
| 83             | -0.5               |
| 84             | 0                  |
| 85             | 0.5                |
| 86             | 0                  |
| 87             | -0.5               |
| 88             | 0                  |
| 89             | 0.5                |
| 90             | 0                  |
| 91             | -0.5               |
| 92             | 0                  |
| 93             | 0.5                |
| 94             | 0                  |
| 95             | -0.5               |
| 96             | 0                  |
| 97             | 0.5                |
| 98             | 0                  |
| 99             | -0.5               |
| Note: The actual values may vary due to the random nature of the data generation. The provided values are just an example.
</details>

f   
![](images/4153ebb4af08830ae0d4269c3de26f209a858c327ff9ee248a82df7a1600d0c8.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["ω_cw, ω_ccw"] --> B["Non-reciprocal gyroscopic effect"]
    C["ω_cw"] --> B
    D["ω_ccw"] --> B
    style A fill:#f9f,stroke:#333
    style B fill:#bbf,stroke:#333
    note1["Apply Ω_x or w/o Ω"] --> A
    note2["Apply + Ω_z"] --> C
```
</details>

g   
![](images/4329b9efc7a02511ecde3f9ce4de4bc0fc7e5bbcce3a9235a39040364e274cdd.jpg)

<details>
<summary>line</summary>

| Peak rotation rate (Hz) | Δφ (mdeg.) |
| ----------------------- | ---------- |
| 0.00                    | 0.0        |
| 0.10                    | 0.1        |
| 0.20                    | 0.2        |
| 0.30                    | 0.3        |
| 0.40                    | 0.4        |
| 0.50                    | 0.5        |
| 0.60                    | 0.6        |
| 0.70                    | 0.7        |
| 0.75                    | 0.8        |
</details>

e   
![](images/bb62117bfeb0e7e9f37edf7f29172919eadc97878e1b12c37ec4e6697dedbf00.jpg)

<details>
<summary>line</summary>

| Time (s) | Apply Ωₓ Δφ (mdeg.) | Apply Ω₂ Δφ (mdeg.) |
| -------- | ------------------- | ------------------- |
| 0        | ~0                  | ~0                  |
| 100      | ~0                  | ~0                  |
| 200      | ~0                  | ~0                  |
| 300      | ~0                  | ~0                  |
| 400      | ~0                  | ~0                  |
</details>

Fig. 4 Phononic spin-orbit interaction in a ring resonator. a Device under test. The pair of IDTs function as the input and output of the circuit and address the clockwise (CW) and counter-clockwise (CCW) love-like mode simultaneously. Focused IDTs and shield structure are used to improve signal to noise ratio. b Orbital angular momentum (OAM) carried by Love-like mode. c Left: Zoomed in mode profile of Love-like mode, with displacements indicated by the black arrows. We observe circular trajectory of displacement. Right: Spin-momentum locking—CW and CCW propagating phonons possess spins of opposite direction. The direction of spin of point A (the purple dot in the left panel) is indicated by either cross (right-hand) or dot (left-hand). Dash line indicates the trajectory of displacement (solid dot). d Phase response difference between CW and CCW modes, when the device is subject to in-plane and out-of-plane rotation. The comparison between $\Delta\phi$ measured with rotation of different direction is a clear signature of SOI. e Applying an out-of-plane CCW rotation to the system, CW and CCW Modes have resonant frequency shift towards the opposite direction, due to the opposite chiralities that they possess. f Phase response measured at various peak rotation rate

The device incorporates a pair of carefully engineered IDTs to unidirectionally actuate and detect the CW and the CCW modes simultaneously. The device is mounted on a rotating stage and a sinusoidally varying rotation is applied to the system with a peak rate at 0.667 Hz (the upper panel of Fig. 4d). To monitor the resonant frequency shift under rotation, we send to the device CW and CCW RF input signals at the original resonant frequency and record the phase change of the output signals. The difference in phase change between CW and CCW signals $\Delta\phi=\phi_{\mathrm{cw}}(\Omega(t))-\phi_{\mathrm{ccw}}(\Omega(t))$ is plotted in the lower panel of Fig. 4d. Both in-plane $(\Omega_{\mathrm{x}})$ and out-of-plane $(\Omega_{\mathrm{z}})$ rotations are measured, with the system drifts calibrated out (see Supplementary Note 2 for details). When the device is subject to $\Omega_{x}$ rotation, we do not observe significant phase response, confirming $|\chi_{x}|=0$ (due to the symmetry of the ring). When the device is subject to $\Omega_{z}$ rotation, however, we observe a phase response possesses the same period with the modulation of the input rotation, which is a clear signature of the OAM-dependent frequency shift. We further measure the phase response at varied peak rotation rates. By fitting the sinusoidal output signal, we acquire phase responses at rotation rate varying from 0.083 Hz to 0.667 Hz (Fig. 4f). The linearly increasing phase response to the rotation is in good agreement with theory. With the slope of the phase response signal calibrated against the phase slope of the resonance, we are able to calculate the chirality of the Love-like mode, $|\chi_{z}|=0.09$ , which is in a reasonable agreement with the simulation result of $|\chi_{z}|=0.12$ . The discrepancy might be attributed to the inaccurate

Table 2 Comparison between the PnIC and the PIC 

<table><tr><td></td><td>PnIC</td><td>PIC</td></tr><tr><td>Input/Output</td><td>Interdigital transducer</td><td>Grating coupler</td></tr><tr><td>External ports</td><td>RF cable</td><td>Fiber</td></tr><tr><td>Interconnection</td><td>Phononic waveguide</td><td>Photonic waveguide</td></tr><tr><td>Velocity</td><td>4000 m s $^{-1}$ </td><td>1.3 × 10 $^{8}$  m s $^{-1}$  (in GaN)</td></tr><tr><td>Modes</td><td>Rayleigh/Love-like</td><td>Quasi-TE/TM</td></tr></table>

material parameters used in the simulation and the mode perturbation caused by the anisotropic substrate and the coupling bus waveguide. Therefore, we confirm the spin-orbit locking and the SOI of phonons. It is worth noting that the SOI and non-reciprocity of optical photons have been observed and studied $^{46,48-52}$ , and this non-reciprocal frequency shift of phonon mode is in particular analogous to Fizeau drag of photons, which has been recently demonstrated in a spinning whispering-gallery resonator $^{52}$ . Owing to the slow sound velocity compared to that of the light, the SOI effect of phonons can be observed at relatively slow rotating rate (<1 Hz), whereas the Fizeau drag of light was measured at several kHz ultrafast spinning speed.

# Discussion

In conclusion, we have established a PnIC architecture based on GaN-on-sapphire semiconductor substrates. Low loss

single-mode waveguides, chip-to-cable connectors, evanescent directional couplers, and waveguide coupled high-Q acoustic ring resonators have been presented to demonstrate basic functionalities of phononic circuits as an analog to PICs. A comparison between PnICs and PICs is shown in Table 2. The sensitivity and versatility of conventional acoustic-based sensing devices can be greatly enhanced by confining phonons in a small volume with long lifetime and routing propagating phonons in more complex circuits in the PnIC architecture. Improved performance of the circuit in a cryogenic environment paves the way for future interfacing with superconducting quantum devices. Additionally, our GNOS platform is compatible with PICs $^{40}$ , allowing the coupling of phonons and photons through optoacoustic interaction $^{53}$ on a single chip. Therefore, quantum links between superconducting qubit and propagating optical field can be built based on the PnIC. Moreover, the ring geometry gives rise to the coupling between OAM and spin of phonons. This new phenomenon opens doors for applications such as acoustic gyroscopic and non-reciprocal devices.

# Methods

Fabrication. A detailed description of fabrication procedures is provided in the Supplementary Note 2. In brief, our PnIC devices are patterned by electron-beam lithography and dry etching processes on GaN-on-sapphire wafers with the GaN layer thickness of 5 $\mu$ m or 10 $\mu$ m. The IDTs are patterned by electron-beam lithography using polymethyl methacrylate (PMMA) resist, followed by the deposition of 10-nm-thick chromium and 50-nm-thick gold and a subsequent lift-off process in acetone.

Measurement methods. A detailed description of measurement methods is provided in the Supplementary Note 2. In brief, the phononic circuit devices are characterized by two methods. The first is an optical measurement of the mechanical vibration of the sample surface by a home-built vibrometer with vibration amplitude sensitivity of around 70 fm Hz $^{-1/2}$ . The principle of the vibrometer is based on a quadrature measurement of vibration-modulated light signals by a heterodyne interferometer. The capability of simultaneous detection of amplitude and phase allows us to fully characterize the traveling acoustic modes in waveguides and ring resonators. In the frequency domain, it offers wide-range frequency spectrum. The second method is an electrical transmission measurement, by which the frequency spectrum $S_{21}$ of the phononic device can be obtained conveniently using a network analyzer. This method is particularly suitable for the measurements in a cryogenic environment.

# Data availability

The data that support the findings of this study are available from the corresponding author (H.T.) upon reasonable request.

Received: 10 May 2019 Accepted: 29 May 2019

Published online: 21 June 2019

# References

1. Vasseur, J. et al. Experimental and theoretical evidence for the existence of absolute acoustic band gaps in two-dimensional solid phononic crystals. Phys. Rev. Lett. 86, 3012 (2001).   
2. Eichenfield, M., Chan, J., Camacho, R. M., Vahala, K. J. & Painter, O. Optomechanical crystals. Nature 462, 78 (2009).   
3. Zhang, S., Yin, L. & Fang, N. Focusing ultrasound with an acoustic metamaterial network. Phys. Rev. Lett. 102, 194301 (2009).   
4. Otsuka, P. H. et al. Broadband evolution of phononic-crystal-waveguide eigenstates in real-and k-spaces. Sci. Rep. 3, 3351 (2013).   
5. Fang, K., Matheny, M. H., Luan, X. & Painter, O. Optical transduction and routing of microwave phonons in cavity-optomechanical circuits. Nat. Photon. 10, 489–496 (2016).   
6. Mohammadi, S. & Adibi, A. On chip complex signal processing devices using coupled phononic crystal slab resonators and waveguides. AIP Adv. 1, 041903 (2011).   
7. Vainsencher, A., Satzinger, K., Peairs, G. & Cleland, A. Bi-directional conversion between microwave and optical frequencies in a piezoelectric optomechanical device. Appl. Phys. Lett. 109, 033107 (2016).

8. Balram, K. C., Davanço, M. I., Song, J. D. & Srinivasan, K. Coherent coupling between radiofrequency, optical and acoustic waves in piezo-optomechanical circuits. Nat. photon. 10, 346 (2016).   
9. Hatanaka, D., Mahboob, I., Onomitsu, K. & Yamaguchi, H. Phonon waveguides for electromechanical circuits. Nat. Nanotech. 9, 520–524 (2014).   
10. Patel, R. N. et al. Single-mode phononic wire. Phys. Rev. Lett. 121, 040501 (2018).   
11. Fan, L. et al. Integrated optomechanical single-photon frequency shifter. Nat. Photon. 10, 766 (2016).   
12. Metelmann, A. & Clerk, A. A. Nonreciprocal photon transmission and amplification via reservoir engineering. Phys. Rev. X 5, 021025 (2015).   
13. Chen, Y. et al. Mechanical bound state in the continuum for optomechanical microresonators. New J. Phys. 18, 063031 (2016).   
14. Ghadimi, A. et al. Elastic strain engineering for ultralow mechanical dissipation. Science 360, 764–768 (2018).   
15. Gustafsson, M. V. et al. Propagating phonons coupled to an artificial atom. Science 346, 207–211 (2014).   
16. Arrangoiz-Arriola, P. et al. Coupling a superconducting quantum circuit to a phononic crystal defect cavity. Phys. Rev. X 8, 031007 (2018).   
17. Chu, Y. et al. Quantum acoustics with superconducting qubits. Science 358, 199–202 (2017).   
18. Noguchi, A., Yamazaki, R., Tabuchi, Y. & Nakamura, Y. Qubit-assisted transduction for a detection of surface acoustic waves near the quantum limit. Phys. Rev. Lett. 119, 180505 (2017).   
19. Riedinger, R. et al. Remote quantum entanglement between two micromechanical oscillators. Nature 556, 473 (2018).   
20. Golter, D. A. et al. Coupling a surface acoustic wave to an electron spin in diamond via a dark state. Phys. Rev. X 6, 041060 (2016).   
21. Kim, P., Hauer, B., Doolin, C., Souris, F. & Davis, J. Approaching the standard quantum limit of mechanical torque sensing. Nat. Commun. 7, 13165 (2016).   
22. O'Connell, A. D. et al. Quantum ground state and single-phonon control of a mechanical resonator. Nature 464, 697 (2010).   
23. Kittlaus, E. A., Shin, H. & Rakich, P. T. Large brillouin amplification in silicon. Nat. Photon. 10, 463 (2016).   
24. Merklein, M. et al. Enhancing and inhibiting stimulated brillouin scattering in photonic integrated circuits. Nat. Commun. 6, 6396 (2015).   
25. Van Laer, R., Kuyken, B., Van Thourhout, D. & Baets, R. Interaction between light and highly confined hypersound in a silicon photonic nanowire. Nat. Photon. 9, 199 (2015).   
26. Aspelmeyer, M., Kippenberg, T. J. & Marquardt, F. Cavity optomechanics. Rev. Mod. Phys. 86, 1391 (2014).   
27. Fon, W. et al. Complex dynamical networks constructed with fully controllable nonlinear nanomechanical oscillators. Nano Lett. 17, 5977–5983 (2017).   
28. Bochmann, J., Vainsencher, A., Awschalom, D. D. & Cleland, A. N. Nanomechanical coupling between microwave and optical photons. Nat. Phys. 9, 712 (2013).   
29. Mahboob, I., Nishiguchi, K., Fujiwara, A. & Yamaguchi, H. Phonon lasing in an electromechanical resonator. Phys. Rev. Lett. 110, 127202 (2013).   
30. Regal, C., Teufel, J. & Lehnert, K. Measuring nanomechanical motion with a microwave cavity interferometer. Nat. Phys. 4, 555 (2008).   
31. Chan, J. et al. Laser cooling of a nanomechanical oscillator into its quantum ground state. Nature 478, 89 (2011).   
32. Lee, H. et al. Chemically etched ultrahigh-q wedge-resonator on a silicon chip. Nat. Photon. 6, 369 (2012).   
33. Wilson, D. et al. Measurement-based control of a mechanical oscillator at its thermal decoherence rate. Nature 524, 325 (2015).   
34. Favero, I. & Karrai, K. Optomechanics of deformable optical cavities. Nat. Photon. 3, 201 (2009).   
35. Han, X., Zou, C. -L. & Tang, H. X. Multimode strong coupling in superconducting cavity piezoelectromechanics. Phys. Rev. Lett. 117, 123603 (2016).   
36. Khanaliloo, B. et al. Single-crystal diamond nanobeam waveguide optomechanics. Phys. Rev. X 5, 041051 (2015).   
37. Renninger, W., Kharel, P., Behunin, R. & Rakich, P. Bulk crystalline optomechanics. Nat. Phys. 14, 601–607 (2018).   
38. Fleury, R., Sounas, D. L., Sieck, C. F., Haberman, M. R. & Alù, A. Sound isolation and giant linear nonreciprocity in a compact acoustic circulator. Science 343, 516–519 (2014).   
39. Poulton, C. G., Pant, R. & Eggleton, B. J. Acoustic confinement and stimulated brillouin scattering in integrated optical waveguides. J. Opt. Soc. Am. B 30, 2657–2664 (2013).   
40. Bruch, A. W. et al. Broadband nanophotonic waveguides and resonators based on epitaxial gan thin films. Appl. Phys. Lett. 107, 141113 (2015).   
41. Dong, C. -H. et al. Brillouin-scattering-induced transparency and non-reciprocal light storage. Nat. Commun. 6, 6193 (2015).   
42. Fiore, V. et al. Storing optical information as a mechanical excitation in a silica optomechanical resonator. Phys. Rev. Lett. 107, 133601 (2011).

43. Merklein, M., Stiller, B., Vu, K., Madden, S. J. & Eggleton, B. J. A chip-integrated coherent photonic-phononic memory. Nat. Commun. 8, 574 (2017).   
44. Coldren, L. A. & Corzine, S. W. Diode Lasers and Photonic Integrated Circuits (Wiley, New York, 1995).   
45. Matsko, A. B., Savchenkov, A. A., Strekalov, D. & Maleki, L. Whispering gallery resonators for studying orbital angular momentum of a photon. Phys. Rev. Lett. 95, 143904 (2005).   
46. Bliokh, K. Y., Smirnova, D. & Nori, F. Quantum spin hall effect of light. Science 348, 1448–1451 (2015).   
47. Lao, B. Y. Gyroscopic effect in surface acoustic waves. In IEEE Ultrasonics Symp., 687–691 (IEEE, Boston, 1980).   
48. Bliokh, K. Y., Rodriguez-Fortuño, F. J., Nori, F. & Zayats, A. V. Spin-orbit interactions of light. Nat. Photon. 9, 796 (2015).   
49. Peng, B. et al. Chiral modes and directional lasing at exceptional points. Proc. Natl Acad. Sci. U.S.A. 113, 6845–6850 (2016).   
50. Lodahl, P. et al. Chiral quantum optics. Nature 541, 473 (2017).   
51. Sohn, D. B., Kim, S. & Bahl, G. Time-reversal symmetry breaking with acoustic pumping of nanophotonic circuits. Nat. Photon. 12, 91 (2018).   
52. Maayani, S. et al. Flying couplers above spinning resonators generate irreversible refraction. Nature 558, 569 (2018).   
53. Tadesse, S. A. & Li, M. Sub-optical wavelength acoustic wave modulation of integrated photonic resonators at microwave frequencies. Nat. Commun. 5, 5402 (2014).   
54. Levinshtein, M. E., Rumyantsev, S. L. & Shur, M. S. Properties of Advanced Semiconductor Materials: GaN, AIN, InN, BN, SiC, SiGe (Wiley, New York, 2001).   
55. Pohl, R. O., Liu, X. & Thompson, E. Low-temperature thermal conductivity and acoustic attenuation in amorphous solids. Rev. Mod. Phys. 74, 991 (2002).   
56. Auld, B. A. Acoustic Fields and Waves in Solids. (Wiley, New York, 1973).   
57. Hopcroft, M. A., Nix, W. D. & Kenny, T. W. What is the young's modulus of silicon? J. Micro. Syst. 19, 229–238 (2010).   
58. Flannery, C. M., Whitfield, M. D. & Jackman, R. B. Acoustic wave properties of cvd diamond. Semicond. Sci. Technol. 18, S86 (2003).

# Acknowledgements

This work is supported by Air Force Office of Scientific Research (AFOSR) MURI grant (FA9550-15-1-0029) and DARPA/MTO's PRIGM:AIMS program through a grant from SPAWAR (N66001-16-1-4026). H.X.T. acknowledges support from a Packard Fellowship in Science and Engineering. We thank Dr. Michael Rooks, Michael Power, James Agresta,

and Christopher Tillinghast for assistance in device fabrication. We acknowledges Joshua B. Surya, Linran Fan, Wance Wang, Mohan Shen and Liang Jiang for helpful discussions.

# Author contributions

W.F. and Z.S. contribute equally to this work. C.-L.Z. and H.X.T. conceived the experiments, W.F. and R.C. prepared the samples, Z.S., W.F., and Y.T.X. built the optical vibrometer set-up and carried out measurements. W.F. and Z.S. performed the numerical simulation and analyzed the data, C.-L.Z. provided theoretical supports. W.F. wrote the manuscript with input from all co-authors. H.X.T. supervised the project. All authors contributed extensively to the work presented in this paper.

# Additional information

Supplementary Information accompanies this paper at https://doi.org/10.1038/s41467-019-10852-3.

Competing interests: The authors declare no competing interests.

Reprints and permission information is available online at http://npg.nature.com/reprintsandpermissions/

Peer review information: Nature Communications thanks the anonymous reviewer(s) for their contribution to the peer review of this work.

Publisher's note: Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

![](images/71c75daf6e333e2918b6ac39bdae1c8eb15d6989b56fad5eee3f335046e28736.jpg)

Open Access This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons license, and indicate if changes were made. The images or other third party material in this article are included in the article's Creative Commons license, unless indicated otherwise in a credit line to the material. If material is not included in the article's Creative Commons license and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.

© The Author(s) 2019