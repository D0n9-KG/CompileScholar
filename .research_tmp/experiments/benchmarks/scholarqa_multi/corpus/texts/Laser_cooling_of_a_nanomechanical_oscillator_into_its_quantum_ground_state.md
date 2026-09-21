# Laser cooling of a nanomechanical oscillator into its quantum ground state

Jasper Chan, $^{1}$ T. P. Mayer Alegre, $^{1}$ Amir H. Safavi-Naeini, $^{1}$ Jeff T. Hill, $^{1}$ Alex Krause, $^{1}$ Simon Gröblacher, $^{1,2}$ Markus Aspelmeyer, $^{2}$ and Oskar Painter $^{1,*}$

$^{1}$ Thomas J. Watson, Sr., Laboratory of Applied Physics,

California Institute of Technology, Pasadena, CA 91125

$^{2}$ Vienna Center for Quantum Science and Technology (VCQ), Faculty of Physics,

University of Vienna, Boltzmanngasse 5, A-1090 Vienna, Austria

(Dated: November 26, 2024)

A patterned Si nanobeam is formed which supports co-localized acoustic and optical resonances that are coupled via radiation pressure. Starting from a bath temperature of $T_{b} \approx 20$ K, the 3.68 GHz nanomechanical mode is cooled into its quantum mechanical ground state utilizing optical radiation pressure. The mechanical mode displacement fluctuations, imprinted on the transmitted cooling laser beam, indicate that a final phonon mode occupancy of $\bar{n} = 0.85 \pm 0.04$ is obtained.

The simple mechanical oscillator, canonically consisting of a coupled mass-spring system, is used in a wide variety of sensitive measurements, including the detection of weak forces $[1]$ and small masses $[2]$ . A classical oscillator can take on a well-defined amplitude of sinusoidal motion. A quantum oscillator, however, has a lowest energy state, or ground state, with a finite amplitude uncertainty corresponding to the zero-point motion. In our everyday experience mechanical oscillators are filled with many energy quanta due to interactions with their highly fluctuating thermal environment, and the oscillator's quantum nature is all but hidden. Recently, in experiments performed at temperatures of a few hundredths of a Kelvin, engineered nanomechanical resonators coupled to electrical circuits have been measured to be oscillating quietly in their quantum ground state $[3, 4]$ . These experiments, in addition to providing a glimpse into the underlying quantum behavior of mesoscopic systems consisting of billions of atoms, represent the initial steps towards the use of mechanical elements as tools for quantum metrology $[5, 6]$ or as a means to couple hybrid quantum systems $[7–9]$ . In this work we have created a coupled, nanoscale optical and mechanical resonator $[10]$ formed in a silicon microchip, in which radiation pressure from a laser is used to cool the mechanical motion down to the quantum ground state (average phonon occupancy, $\bar{n} = 0.85 \pm 0.04$ ). Critically, this cooling is realized at an environmental temperature some thousand times larger than in previous experiments ( $T_{b} \approx 20~K$ ), and paves the way for optical control of mesoscale mechanical oscillators in the quantum regime.

It has been known for some time $[11]$ that atoms and ions nearly resonant with an applied laser beam (or series of beams) may be mechanically manipulated, even trapped and cooled down to the quantum ground state of their center-of-mass motion $[12]$ . Equally well known $[1]$ has been the fact that radiation pressure can be exerted on regular dielectric (i.e., non-resonant) objects to damp and cool their mechanical motion. In so-called cavity-assisted schemes, the radiation pressure force is enhanced by coupling the motion of a mechanical object to the light field in an optical cavity. Pumping of the optical cavity by a single-frequency electromagnetic source produces a coupling between the mechanical motion and the intensity of the electromagnetic field built-up in the resonator. As the radiation pressure force exerted on the mechanical object is proportional to the field intensity in the resonator, a form of dynamical back-action results $[1, 13]$ . For a lower-frequency (red) detuning of the laser from the cavity, this leads to damping and cooling of the mechanical motion.

Recent experiments involving micro- and nanomechanical resonators, coupled to electromagnetic fields at optical and microwave frequencies, have demonstrated significant radiation pressure dynamic back-action $[13]$ . These structures have included Fabry-Pérot cavities with mechanically-compliant miniature end-mirrors $[14–16]$ or internal nanomembranes $[17]$ , whispering-gallery glass resonators $[18]$ , nanowires capacitively coupled to co-planar microwave transmission-line cavities $[6, 19]$ , and lumped circuit microwave resonators with deformable, nanoscale, vacuum-gap capacitors $[20]$ . The first measurement of an engineered mesoscopic mechanical resonator predominantly in its quantum ground state, however, has been performed not using back-action cooling, but rather, using conventional cryogenic cooling (bath temperature $T_{b} \approx 25$ mK) of a high frequency, and thus lower thermal occupancy, oscillator $[3]$ . Read-out and control of mechanical motion at the single quanta level was performed by strongly coupling the GHz-frequency piezoelectric mechanical resonator to a resonant superconducting quantum circuit. Only recently have microwave systems, also operating at bath temperatures of $T_{b} \approx 25$ mK, utilized radiation pressure back-action to cool a high-Q, MHz-frequency mechanical oscillator to the ground state $[4, 19]$ .

Optically coupled mechanical devices, while allowing for control of the mechanical system through well-established quantum optical techniques $[21]$ , have thus far not reached the quantum regime due to a myriad of technical difficulties $[18]$ . A particular challenge has been maintaining efficient optical coupling and low-loss optics and mechanics in a cryogenic, sub-Kelvin environment. The optomechanical system studied in this work enables large optical coupling to a high-Q GHz-frequency mechanical

![](images/81858736eb78bb6c79e525db5ebcc6d6dd85f0ab217ae525dbf1c72d16e186b8.jpg)

<details>
<summary>natural_image</summary>

Microscopic view of a square microstructure with a central linear feature and 5 μm scale bar (no text or symbols beyond scale indicator)
</details>

![](images/2800dcfbb7bdb0a5a3857b97d07b10988087205746c73e56c139653f0c7a814e.jpg)

<details>
<summary>natural_image</summary>

Microscopic images of a microfluidic device with circular features, showing color-coded flow patterns (no text or symbols)
</details>

![](images/7ffc6e67b42189037cffba172dd7c8fa167d9e30c4e2945e9c7cd5c028ec1752.jpg)

<details>
<summary>natural_image</summary>

Microscopic and thermal imaging of a hexagonal lattice structure with scale bar (1 μm) and color-coded heat map overlay (no text or symbols)
</details>

FIG. 1: Optomechanical resonator with phononic shield. a, Scanning electron microscope (SEM) image of the patterned Si nanobeam with external phononic bandgap shield. b, Enlarged SEM image of the central cavity region of the nanobeam. c, Top, finite element method (FEM) simulated normalized electric field of the localized optical resonance of the nanobeam cavity. Bottom, FEM simulated normalized displacement field of the acoustic resonance (breathing mode) which is coupled via radiation pressure to the co-localized optical resonance. The displacement field is indicated by the exaggerated deformation of the structure, with the relative magnitude of the local displacement (strain) indicated by the color. d, SEM image of the interface between the nanobeam and the phononic bandgap shield. e, FEM simulation of the normalized squared displacement field amplitude of the localized acoustic resonance at the nanobeam-shield interface, indicating the strong suppression of acoustic radiation provided by the phononic bandgap shield. The color scale represents $\log(x^{2}/\max\{x^{2}\})$ where x is the displacement.

oscillator, allowing for both efficient back-action cooling and significantly higher operating temperatures. As shown in Fig. 1a, the system consists of an integrated optical and nanomechanical resonator formed in the surface layer of a silicon-on-insulator microchip. The periodic patterning of the nanobeam is designed to result in Bragg scattering of both optical and acoustic guided waves. A perturbation in the periodicity at the center of the beam results in co-localized optical and mechanical resonances (Fig. 1b-c), which are coupled via radiation pressure [10]. The fundamental optical resonance of the structure occurs at a frequency $\omega_{o} / 2\pi = 195$ THz ( $\lambda = 1537$ nm), while, due to the much slower speed of sound, the mechanical resonance occurs at $\omega_{m} / 2\pi = 3.68$ GHz. In order to minimize mechanical damping in the structure, an external acoustic radiation shield is added in the periphery of the nanobeam (Fig. 1d-e). This acoustic shield consists of a two-dimensional “cross” pattern, which has been shown both theoretically and experimentally to yield a substantial phononic bandgap in the GHz frequency band [22].

A fiber taper nanoprobe, formed from standard single-mode optical fiber, is used to optically couple to the silicon nanoscale resonators. As shown in Fig. 2, a tunable laser (New Focus Velocity swept laser; $200\mathrm{kHz}$ linewidth) is used to optically cool and transduce the mechanical motion of the nanomechanical oscillator. Placing the optomechanical devices into a continuous-flow helium cryostat provides a modicum of pre-cooling down to $T_{b} \approx 20\mathrm{K}$ , reducing the bath occupancy of the $3.68\mathrm{GHz}$ mechanical mode to $n_{b} \approx 100$ . At this temperature the mechanical $Q$ -factor increases up to a measured value of $Q_{m} \approx 10^{5}$ , corresponding to an intrinsic mechanical damping rate of $\gamma_{i}/2\pi = 35\mathrm{kHz}$ . The optical $Q$ -factor is measured to be $Q_{o} = 4 \times 10^{5}$ , corresponding to an optical linewidth of $\kappa/2\pi = 500\mathrm{MHz}$ , slightly reduced from its room temperature value.

In the resolved sideband limit in which $\omega_{m} / \kappa > 1$ , driving the system with a laser (frequency $\omega_{l}$ ) tuned to the red side of the optical cavity (detuning $\Delta \equiv \omega_{o} - \omega_{l} = \omega_{m}$ ), creates an optically-induced damping, $\gamma_{\mathrm{OM}}$ , of the mechanical resonance [23, 24]. In the weak-coupling regime ( $\gamma_{\mathrm{OM}} \ll \kappa$ ) and for a detuning $\Delta = \omega_{m}$ , the optical back-action damping is given by $\gamma_{\mathrm{OM}} = 4g^{2}n_{c} / \kappa$ , where $n_{c}$ is the average number of drive photons stored in the cavity and $g$ is the optomechanical coupling rate between the mechanical and optical modes. This coupling rate, $g$ , is quantified as the shift in the optical resonance for an amplitude of motion equal to the zero-point amplitude ( $x_{\mathrm{zpf}} = (\hbar / 2m\omega_{m})^{1/2}$ ; $m$ the motional mass of the localized acoustic mode, $\hbar$ Planck's constant divided by $2\pi$ ). The optomechanical damping, a result of the preferential scattering of drive laser photons into the upper-frequency sideband, also cools the mechanical mode. For a quantum-limited drive laser, the phonon occupancy of the mechanical oscillator can be reduced from $n_{b} = k_{B}T_{b}/\hbar\omega_{m} \gg 1$ , to a value $\bar{n} = n_{b}/(1+C) + n_{\min}$ , where $C \equiv \gamma_{\mathrm{OM}}/\gamma_{i}$ is the cooperativity. The residual scattering of drive laser photons into the lower-frequency sideband limits the cooled phonon occupancy to $n_{\min} = (\kappa/4\omega_{m})^{2}$ , determined by the level of sideband resolution [23, 24].

The drive laser, in addition to providing mechanical damping and cooling, can be used to measure the mechanical and optical properties of the system through a series of calibrated measurements. In a first set of measurements the noise power spectral density (PSD) of the drive laser transmitted through the optomechanical cavity is used to perform spectroscopy of the mechanical

![](images/d645f87bf15cbb9bad1f3848fdaae0c846fec9006bc992af89a6e315b4c687d6.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["RF S.G."] --> B["lock-in"]
    B --> C["D1"]
    C --> D["RF"]
    D --> E["EOM"]
    E --> F["VOA"]
    F --> G["3"]
    G --> H["2"]
    H --> I["FPC"]
    I --> J["DC"]
    J --> K["FM"]
    K --> L["Laser"]
    L --> M["WM"]
    M --> N["RF"]
    N --> O["LF"]
    O --> P["Lock-in"]
    P --> Q["RF S.G."]
    R["RSA"] --> S["D2"]
    S --> T["EDFA"]
    T --> U["Cryostat Taper device"]
    U --> V["FPC"]
    V --> W["FC"]
    W --> X["DC"]
    X --> Y["FM"]
    Y --> Z["LF"]
    Z --> AA["Lock-in"]
```
</details>

FIG. 2: Experimental setup. A single tunable 1550 nm diode laser is used as the cooling and mechanical transduction beam sent into the nanobeam optomechanical resonator cavity held in a continuous flow Helium cryostat. A wavemeter (WM) is used to track and lock the laser frequency, while a variable optical attenuator (VOA) is used to set the laser power. The transmitted signal is amplified by an erbium doped fiber amplifier (EDFA), and detected on a high-speed photodetector (D2) connected to a real-time spectrum analyzer (RSA), where the mechanical noise power spectrum is measured. A slowly modulated probe signal used for optical spectroscopy and calibration is generated from the cooling laser beam via an amplitude electro-optic modulator (EOM) driven by a microwave source (RFSG). The reflected component of this signal is separated from the input via an optical circulator, sent to a photodetector (D1), and then demodulated on a lock-in amplifier. Paddle-wheel fiber polarization controllers (FPCs) are used to set the laser polarization at the input to the EOM and the input to the optomechanical cavity. For more detail see Appendix D.

mode. As shown in Appendix A, the noise power spectral density of the photocurrent generated by the transmitted field of the drive laser with red-sideband detuning $(\Delta = \omega_{m})$ yields a Lorentzian component of the single-sided PSD proportional to $S_{b}(\omega) = \bar{n}\gamma/((\omega - \omega_{m})^{2} + (\gamma/2)^{2})$ , where $\gamma = \gamma_{i} + \gamma_{OM} = \gamma_{i}(1 + C)$ is the total mechanical damping rate. For a blue laser detuning of $\Delta = -\omega_{m}$ , the optically-induced damping is negative ( $\gamma_{OM} = -4g^{2}n_{c}/\kappa$ ) and the photocurrent noise PSD is proportional to $S_{b^{\dagger}}(\omega) = (\bar{n} + 1)\gamma/((\omega - \omega_{m})^{2} + (\gamma/2)^{2})$ . Typical measured noise power spectra under low power laser drive ( $n_{c} = 1.4$ , C = 0.27), for both red and blue detuning, are shown in Fig. 3a. Even at these small drive powers the effects of back-action are clearly evident on the measured spectra, with the red-detuned drive broadening the mechanical line and the blue-detuned drive narrowing the line. The noise floor in Fig. 3a (shaded in gray) corresponds to the noise generated by the erbium doped fiber amplifier (EDFA) used to pre-amplify the transmitted drive laser signal prior to photodetection, and is many orders of magnitude above the electronic noise of the photoreceiver and real-time spectrum analyzer.

Calibration of the EDFA gain, along with the photoreceiver and real-time spectrum analyzer photodetection gain, allows one to convert the measured area under the photocurrent noise PSD into a mechanical mode phonon occupancy. As described in detail in the Appendices, these calibrations, along with measurements of low drive power $(C \ll 1)$ rf-spectra of both $\Delta = \pm\omega_{m}$ detunings, are performed to provide an accurate, local thermometry of the optomechanical cavity. An example of this form of calibrated mode thermometry is shown in Fig. 3b, where we plot the optically measured mechanical mode bath temperature $(T_{b})$ versus the cryostat sample mount temperature $(T_{c};$ independently measured using a Si diode thermometer attached to the copper sample mount). As one can see from this plot, the optical mode thermometry accurately predicts the absolute temperature of the sample for $T_{c} > 50$ K; below this value the mode temperature deviates from $T_{c}$ and saturates to a value of $T_{b} = 17.6 \pm 0.8$ K due to black-body heating of the device through the imaging aperture in the radiation shield of our cryostat.

In a second set of measurements, the mechanical damping, $\gamma$ , and the cavity-laser detuning, $\Delta$ , can be measured by optical spectroscopy of the driven cavity. By sweeping a second probe beam of frequency $\omega_{s}$ over the cavity, with the cooling beam tuned to $\Delta = \omega_{m}$ , spectra exhibiting electromagnetically-induced transparency (EIT) [20, 25, 26] are measured, as shown in Fig. 3c. Due to the high single-photon cooperativity in the system, an intracavity population of only $n_c \approx 5$ switches the system from reflecting to transmitting for the probe beam. The corresponding dip at the center of the optical cavity resonance occurs at a two-photon detuning $\Delta_{sl} \equiv \omega_s - \omega_l = \omega_m$ and has a bandwidth equal to the mechanical damping rate, $\gamma_i(1 + C)$ . Figure 4a shows a plot of the measured mechanical linewidth versus intracavity photon number, displaying good correspondence between

![](images/83ba95f77997199079e6ad5faaa9b270e396b7bef82d5338c45505bcf7ab1f73.jpg)  
FIG. 3: Mechanical and optical response. a, Typical measured mechanical noise spectra around the resonance frequency of the breathing mode for low laser drive power ( $n_{c} = 1.4$ ). The blue (red) curve corresponds to the measured spectrum with laser drive blue (red) detuned by a mechanical frequency from the optical cavity resonance. The black trace corresponds to the measured noise floor (dominated by EDFA noise) with the drive laser detuned far from cavity resonance. b, Plot of the measured (☐) mechanical mode bath temperature ( $T_{b}$ ) versus cryostat sample mount temperature ( $T_{c}$ ). The dashed line indicates the curve corresponding to perfect following of the mode temperature with the cryostat temperature ( $T_{b} = T_{c}$ ). c, Typical reflection spectrum of the cavity while driven by the cooling laser at $\Delta = \omega_{m}$ as measured by a weaker probe beam at two-photon detuning $\Delta_{sl}$ . The signature reflection dip on-resonance with the bare cavity mode, highlighted in the inset, is indicative of electromagnetically-induced transparency (EIT) caused by the coupling of the optical and mechanical degrees of freedom by the cooling laser beam.

both mechanical and optical spectroscopy techniques. From a fit to the measured mechanical damping rate versus $n_{c}$ (dashed red line in Fig. 4a), the zero-point-motion optomechanical coupling rate is estimated to be $g/2\pi = 910$ kHz, placing the system well within the weak-coupling regime for all measured drive powers.

In Fig. 4b we plot the calibrated Lorentzian noise PSD area, in units of phonon occupancy, versus red-detuned ( $\Delta = \omega_{m}$ ) drive laser power. Due to the low effective temperature of the laser drive, the mechanical mode is not only damped but also cooled substantially. The minimum measured mode occupancy for the highest drive power of $n_{c} \approx 2000$ is $\bar{n} = 0.85 \pm 0.04$ , putting the mechanical oscillator in a thermal state with ground state occupancy probability greater than 50%. The dashed blue line in Fig. 4b represents the ideal back-action cooled phonon occupancy estimated using both the measured mechanical damping rate in Fig. 4a and the low drive power intrinsic mechanical damping rate. Deviation of the measured phonon occupancy from the ideal cooling model is seen to occur at the highest drive powers, and as detailed in Appendices I-K, is due to both an increase in the bath temperature due to optical absorption (Fig. 4c) and an increase in the intrinsic mechanical damping rate (Fig. 4d) induced by the generation of free-carriers through optical absorption. In order to evaluate the efficiency of the optical transduction of the mechanical motion, we also plot in Fig. 4e the measured background noise PSD (or imprecision level) alongside that for an ideal cavity transducer with shot-noise limited detection. The minimum measurement imprecision corresponds to $n_{imp} \approx 20$ in units of phonon quanta, 40% of which is due to light lost inside the optical cavity (and thus not detected). The remaining 60% of the imprecision stems from optical loss in the fiber taper (approximately 2 dB) and added noise due to the EDFA pre-amplification.

Looking ahead, the combination of strong optical back-action cooling and efficient mechanical motion transduction realized in the chip-scale optomechanical cavities of this work, represents a first step towards optical quantum control of nanomechanical objects [27]. For example, optomechanical entanglement between light and mechanics [28] or quantum state transfer between single optical photons and mechanical phonons [9, 29] may be envisioned, enabling mechanical systems to function as either quantum transducers [8] or quantum memory elements [30]. The efficacy of such quantum protocols and devices, and the ability to measure quantum dynamics, relies on the thermal decoherence time of the mechanical system, given by $\tau \equiv k_{B}T_{b} / \hbar Q_{m}$ . For the measured devices in this work at $T_{b} \approx 20\mathrm{K}$ , the decoherence time corresponds to $N_{\mathrm{osc}} \equiv \tau \omega_{m} / 2\pi \approx 200$ periods of coherent

![](images/1c0390df57acd4dfd232618b5350e6d262513dc5b663f9645f269f78a5b41edf.jpg)

<details>
<summary>line</summary>

| n_c | γ/2π (kHz) |
| --- | --- |
| 1 | ~50 |
| 10 | ~200 |
| 100 | ~1000 |
| 1000 | ~10000 |
</details>

![](images/0ecfd34af1d98662b2cf6fd97d16b441a0c56e0ab472be473297314f8c19faa0.jpg)

<details>
<summary>line</summary>

| n_c | n̄     |
|-----|-------|
| 1   | 100   |
| 10  | 30    |
| 100 | 10    |
| 1000| 1     |
</details>

![](images/d1e691de03439a20135b3dfc44cc93ebd1221c45d50360f3197ff94359dc334c.jpg)

<details>
<summary>line</summary>

| nc    | Tb (K) | Δγ/2π (kHz) | Sensitivity (quanta) |
|-------|--------|-------------|----------------------|
| 1     | 18.0   | 0           | 1000                 |
| 10    | 17.5   | 0           | 500                  |
| 100   | 18.5   | 5           | 200                  |
| 1000  | 30.0   | 40          | 10                   |
</details>

FIG. 4: Optical cooling results. a, Measured mechanical mode linewidth ( $\square$ ), EIT transparency bandwidth ( $\circ$ ), and predicted optomechanical damping rate estimated using the zero-point optomechanical coupling rate $g/2\pi = 910$ kHz (red dashed line). The inset shows the measured EIT transparency window at the highest cooling drive power. b, Measured ( $\circ$ ) average phonon number, $\bar{n}$ , in the breathing mechanical mode at $\omega_{m}/2\pi = 3.68$ GHz versus cooling laser drive power (in units of intracavity photons, $n_{c}$ ), as deduced from the calibrated area under the Lorentzian lineshape of the mechanical noise power spectrum, $S_{b}$ . The left and right inset spectra correspond to the measured noise power spectrum in units of m $^{2}$ /Hz at low and high laser cooling power, respectively. The dashed blue line indicates the estimated mode phonon number from the measured optical damping alone. Error bars indicate computed standard deviations as outlined in the SI. c, Estimated bath temperature, $T_{b}$ , versus cooling laser intracavity photon number, $n_{c}$ . d, Measured change in the intrinsic mechanical damping rate versus $n_{c}$ ( $\circ$ ). A polynomial fit to the mechanical damping dependence on $n_{c}$ is shown as a dashed line. For more detail see the SI. e, The measured ( $\square$ ) background noise PSD versus laser drive power ( $n_{c}$ ), in units of phonon quanta. Shown as a black dashed line is the theoretical shot-noise limited noise PSD for an ideal single-sided cavity, a unit quantum efficiency detector, and no optical loss in the transmitted optical field.

oscillation of the mechanical resonator. Back-action cooling[24] from a given bath temperature to the quantum ground state requires $\hbar \kappa \gtrsim k_{B}T_{b} / Q_{m}$ . Given the properties of the devices in this work, ground state cooling from room temperature with measurable quantum dynamics ( $N_{\mathrm{osc}} \approx 10$ ) seems feasible. This opens up the possibility for future experimentation with, and utilization of, quantum nanomechanical objects in a room temperature environment.

# Acknowledgements

This work was supported by the DARPA/MTO ORCHID program through a grant from AFOSR, the European Commission (MINOS, QUESSENCE), European Research Council (ERC QOM), the Austrian Science Fund (CoQuS, FOQUS, START), and the Kavli Nanoscience Institute at Caltech. JC and ASN gratefully acknowledge support from NSERC.

[1] Braginsky, V. & Manukin, A. Measurement of weak forces in Physics experiments (Univ. of Chicago Press, 1977).   
[2] Jensen, K., Kim, K. & Zettl, A. An atomic-resolution nanomechanical mass sensor. Nature Nanotech. 3, 533–537 (2008).   
[3] O'Connell, A. D. et al. Quantum ground state and single-phonon control of a mechanical resonator. Nature 464, 697–703 (2010).   
[4] Teufel, J. D. et al. Sideband cooling micromechanical motion to the quantum ground state. arXiv:1103.2144 (2011).   
[5] Caves, C., Thorne, K., Drever, R., Sandberg, V. D. & Zimmermann, M. On the measurement of a weak classical force coupled to a quantum-mechanical oscillator. Reviews of Modern Physics 52, 341–392 (1980).   
[6] Regal, C. A., Teufel, J. D. & Lehnert, K. W. Measuring nanomechanical motion with a microwave cavity interferometer. Nature Phys. 4, 555–560 (2008).   
[7] Wallquist, M., Hammerer, K., Rabl, P., Lukin, M. & Zoller, P. Hybrid quantum devices and quantum engineering. Physica Scripta 2009, 014001 (2009).

[8] Stannigel, K., Rabl, P., Sørensen, A. S., Zoller, P. & Lukin, M. D. Optomechanical Transducers for Long-Distance Quantum Communication. Phys. Rev. Lett. 105, 220501 (2010).   
[9] Safavi-Naeini, A. H. & Painter, O. Proposal for an optomechanical traveling wave phonon-photon translator. New J. Phys. 13, 013017 (2011).   
[10] Eichenfield, M., Chan, J., Camacho, R. M., Vahala, K. J. & Painter, O. Optomechanical crystals. Nature 462, 78–82 (2009).   
[11] Cohen-Tannoudji, C. N. & Phillips, W. D. New Mechanisms for Laser Cooling. Phys. Today 43, 33–40 (1990).   
[12] Monroe, C. et al. Resolved-Sideband Raman Cooling of a Bound Atom to the 3D Zero-Point Energy. Phys. Rev. Lett. 75, 4011–4014 (1995).   
[13] Kippenberg, T. J. & Vahala, K. J. Cavity Optomechanics: Back-Action at the Mesoscale. Science 321, 1172–1176 (2008).   
[14] Gigan, S. et al. Self-cooling of a micromirror by radiation pressure. Nature 444, 67–70 (2006).   
[15] Arcizet, O., Cohadon, P.-F., Briant, T., Pinard, M. & Heidmann, A. Radiation-pressure cooling and optomechanical instability of a micromirror. Nature 444, 71–74 (2006).   
[16] Gröblacher, S. et al. Demonstration of an ultracold micro-optomechanical oscillator in a cryogenic cavity. Nature Phys. 5, 485–488 (2009).   
[17] Thompson, J. D. et al. Strong dispersive coupling of a high-finesse cavity to a micromechanical membrane. Nature 452, 72–75 (2008).   
[18] Rivière, R. et al. Optomechanical sideband cooling of a micromechanical oscillator close to the quantum ground state. arXiv:1011.0290 (2010).   
[19] Rocheleau, T. et al. Preparation and detection of a mechanical resonator near the ground state of motion. Nature 463, 72–75 (2010).   
[20] Teufel, J. D. et al. Circuit cavity electromechanics in the strong-coupling regime. Nature 471, 204-208 (2011).   
[21] Wiseman, H. M. & Milburn, G. J. Quantum Measurement and Control (Cambridge University Press, 2010).   
[22] Alegre, T. P. M., Safavi-Naeini, A., Winger, M. & Painter, O. Quasi-two-dimensional optomechanical crystals with a complete phononic bandgap. Opt. Express 19, 5658–5669 (2011).   
[23] Wilson-Rae, I., Nooshi, N., Zwerger, W. & Kippenberg, T. J. Theory of ground state cooling of a mechanical oscillator using dynamical back-action. Phys. Rev. Lett. 99, 093901 (2007).   
[24] Marquardt, F., Chen, J. P., Clerk, A. A. & Girvin, S. M. Quantum theory of cavity-assisted sideband cooling of mechanical motion. Phys. Rev. Lett. 99, 093902 (2007).   
[25] Weis, S. et al. Optomechanically Induced Transparency. Science 330, 1520–1523 (2010).   
[26] Safavi-Naeini, A. H. et al. Electromagnetically induced transparency and slow light with optomechanics. Nature 472, 69–73 (2011).   
[27] Aspelmeyer, M., Gröblacher, S., Hammerer, K. & Kiesel, N. Quantum optomechanics – throwing a glance. J. Opt. Soc. Am. B 27, A189–A197 (2010).   
[28] Vitali, D. et al. Optomechanical Entanglement between a Movable Mirror and a Cavity Field. Phys. Rev. Lett. 98, 030405 (2007).   
[29] Akram, U., Kiesel, N., Aspelmeyer, M. & Milburn, G. J. Single-photon optomechanics in the strong coupling regime. New J. Phys. 12, 083030 (2010).   
[30] Chang, D., Safavi-Naeini, A. H., Hafezi, M. & Painter, O. Slowing and stopping light using an optomechanical crystal array. New J. Phys. 13, 023003 (2011).   
[31] Borselli, M., Johnson, T. J. & Painter, O. Measuring the role of surface chemistry in silicon microphotonics. App. Phys. Lett. 88, 131114 (2006).   
[32] Frey, B. J., Leviton, D. B. & Madison, T. J. Temperature-dependent refractive index of silicon and germanium. In Proc. SPIE, vol. 6273, 62732J (2006).   
[33] Harrington, R. F. Time-Harmonic Electromagnetic Fields (IEEE Press, 1961).   
[34] Barclay, P., Srinivasan, K. & Painter, O. Nonlinear response of silicon photonic crystal microresonators excited via an integrated waveguide and fiber taper. Opt. Express 13, 801–820 (2005).   
[35] Desurvire, E., Bayart, D., Desthieux, B. & Bigo, S. Erbium-Doped Fiber Amplifiers, Device and System Developments (Wiley-Interscience, 2002).

# Appendix A: Classical Derivation of Transduced Signal

We begin by modeling the optomechanical system with the Hamiltonian

$$
\hat {H} = \hbar \Delta \hat {a} ^ {\dagger} \hat {a} + \hbar \omega_ {m} \hat {b} ^ {\dagger} \hat {b} + \hbar g (\hat {b} ^ {\dagger} + \hat {b}) \hat {a} ^ {\dagger} \hat {a} + i \hbar \sqrt {\frac {\kappa_ {e}}{2}} \alpha_ {\text { in }, 0} (\hat {a} - \hat {a} ^ {\dagger}), \tag {A1}
$$

where $\Delta = \omega_{o} - \omega_{l}$ , with laser frequency $\omega_{l}$ , optical mode frequency $\omega_{o}$ and mechanical mode frequency $\omega_{m}$ . Here $\hat{a} (\hat{a}^{\dagger})$ and $\hat{b} (\hat{b}^{\dagger})$ are respectively the annihilation (creation) operators of photon and phonon resonator quanta, g is the optomechanical coupling rate corresponding physically to the shift in the optical mode frequency due to the zero-point fluctuations ( $x_{zpf} = \sqrt{\hbar/2m\omega_{m}}$ , m motional mass) of the phonon mode. By making the substitutions

$$
\hat {a} \rightarrow \alpha = \sum_ {q} \alpha_ {q} e ^ {- i q \omega_ {m} t}, \quad \hat {b} \rightarrow \beta_ {0} e ^ {- i \omega_ {m} t} \tag {A2}
$$

we can treat the system classically by representing the photon amplitudes as a Fourier decomposition of sidebands. Notice that the infinite summation over each sideband order $q$ , can be relaxed to a few orders in the sideband resolved regime ( $\kappa \ll \omega_m$ ). The

phonon amplitude, $\beta_{0}$ , is the classical mechanical excitation amplitude. For an oscillator undergoing thermal Brownian motion, $\beta_{0}$ , is a stochastic process. We assert the stochastic nature of the variable, at the end of the derivation where the power spectral density is calculated. The equation of motion for the slowly varying component is then

$$
- i \omega_ {m} \sum_ {q} q \alpha_ {q} e ^ {- i q \omega_ {m} t} = - \left(i \Delta + \frac {\kappa}{2}\right) \sum_ {q} \alpha_ {q} e ^ {- i q \omega_ {m} t} - i g \beta_ {0} \sum_ {q} \alpha_ {q} \left(e ^ {- i (q + 1) \omega_ {m} t} + e ^ {- i (q - 1) \omega_ {m} t}\right) - \sqrt {\frac {\kappa_ {e}}{2}} \alpha_ {\text {in,0}}, \tag {A3}
$$

where we introduce the cavity (optical) energy loss rate, $\kappa$ , and the cavity coupling rate, $\kappa_{e}$ . This can be written as a system of equations $M \cdot \vec{\alpha} = a_{in}$ where

$$
M _ {p q} = \left(i (\Delta - p \omega_ {m}) + \frac {\kappa}{2}\right) \delta_ {p q} + i g \beta_ {0} (\delta_ {p, q + 1} + \delta_ {p, q - 1}), \tag {A4}
$$

$$
\mathrm{a} _ {\text { in }, p} = - \sqrt {\frac {\kappa_ {e}}{2}} \alpha_ {\text { in }, 0} \delta_ {p 0}. \tag {A5}
$$

By truncating and inverting the coupling matrix M one can determine each one of the sidebands amplitude as $\alpha_{q} = (M^{-1})_{qp}$ $a_{in,p}$ and therefore determine the steady state power leaving the cavity to be

$$
\alpha_ {\text {out}} = \alpha_ {\text {in}, 0} + \sqrt {\frac {\kappa_ {e}}{2}} \alpha \tag {A6}
$$

where we assumed that the input pump is not depleted by the cavity, which is the frame of interest of this work. In this case the total power measured at the photodetector will be proportional to

$$
\left| \alpha_ {\text { out }} \right| ^ {2} = \left| \alpha_ {\text { in }, 0} + \sqrt {\frac {\kappa_ {e}}{2}} \alpha \right| ^ {2} \tag {A7}
$$

$$
= \left| \alpha_ {\text {in}, 0} \right| ^ {2} + \frac {\kappa_ {e}}{2} \sum_ {q} \sum_ {p} \alpha_ {q} \alpha_ {p} ^ {*} e ^ {- i (q - p) \omega_ {m} t} + 2 \operatorname{Re} \left\{\alpha_ {\text {in}, 0} \sqrt {\frac {\kappa_ {e}}{2}} \sum_ {q} \alpha_ {q} e ^ {i q \omega_ {m} t} \right\} \tag {A8}
$$

# 1. Resolved Sideband Limit

The equations presented in the previous section are exact and therefore can be solved for any case. However our interests lie in the so-called resolved sideband limit, $\omega_{m} > \kappa/2$ , where further simplification can be done. Specifically for our system, $\omega_{m}/2\pi = 3.68$ GHz, $\kappa/2\pi = 500$ MHz putting us well within this limiting case.

Additionally, in the cavities studied, the optomechanical phase-modulation factor (proportional to $gx_{zpf}/\omega_{m}$ ), is much less than $10^{-3}$ . As such, only the $\omega = \omega_{l} \pm \omega_{m}$ sidebands ( $q = \pm 1$ ) are significant and $\alpha_{0} \gg \alpha_{\pm}$ . Truncating the matrix equations appropriately, we find

$$
\alpha_ {0} = \frac {- \sqrt {\kappa_ {e} / 2} \alpha_ {\mathrm{in,0}}}{i \Delta + \kappa / 2} \tag {A9}
$$

$$
\alpha_ {\pm} = \frac {- i g \beta_ {0} \alpha_ {0}}{i (\Delta \mp \omega_ {m}) + \kappa / 2} \tag {A10}
$$

where $\alpha_{\mathrm{in},0} = \sqrt{N_{\mathrm{in}}}$ , $N_{\mathrm{in}} = P_{\mathrm{in}} / \hbar \omega_0$ and $P_{\mathrm{in}}$ the input power at the cavity.

In the sideband resolved limit, we have $|\alpha_{\mathrm{in},0}| > |\sqrt{\kappa_e / 2}\alpha_0| > |\sqrt{\kappa_e / 2}\alpha_\pm|$ . Therefore the photodetector signal is predominantly composed of the mixing between sidebands with the input pump beam and terms containing $|\alpha_0|^2$ , and can be written

as [10]:

$$
\begin{array}{l} \left| \alpha_ {\text { out }} \right| ^ {2} = \left| \alpha_ {\text { in }, 0} \right| ^ {2} + \sqrt {\frac {\kappa_ {e}}{2}} \alpha_ {\text { in }, 0} \left(\alpha_ {0} + \alpha_ {0} ^ {*}\right) + \frac {\kappa_ {e}}{2} \left| \alpha_ {0} \right| ^ {2} + \dots \\ \sqrt {\frac {\kappa_ {e}}{2}} \alpha_ {\mathrm{in,0}} (\alpha_ {-} e ^ {- i \omega_ {m} t} + \alpha_ {-} ^ {*} e ^ {i \omega_ {m} t}) + \dots \\ \sqrt {\frac {\kappa_ {e}}{2}} \alpha_ {\text {in,0}} (\alpha_ {+} e ^ {i \omega_ {m} t} + \alpha_ {+} ^ {*} e ^ {- i \omega_ {m} t}) + \mathcal {O} (| \alpha_ {0} | | \alpha_ {\pm} |) (A11) \\ \approx | \alpha_ {\mathrm{in}, 0} | ^ {2} \left| 1 - \frac {\kappa_ {e} / 2}{i \Delta + \kappa / 2} \right| ^ {2} + \dots \\ \cos (\omega_ {m} t) \left[ | A _ {+} | \cos (\varphi_ {+}) + | A _ {-} | \cos (\varphi_ {-}) \right] + \dots \\ \sin (\omega_ {m} t) \left[ | A _ {+} | \sin (\varphi_ {+}) - | A _ {-} | \sin (\varphi_ {-}) \right] (A12) \\ \end{array}
$$

where $A_{\pm} \equiv 2\sqrt{\kappa_e / 2}\alpha_{\mathrm{in},0}\alpha_{\pm} = |A_{\pm}|\exp (-i\varphi_{\pm})$ . We can easily recognize the first term in Equation (A12) as the DC cavity transmission spectra. The remaining two terms compose the total power at the mechanical frequency $P_{\mathrm{SB}}(\omega_m) = \hbar \omega_0\sqrt{A_\cos^2 + A_\sin^2}$ , where $A_{\cos} = |A_{+}|\cos (\varphi_{+}) + |A_{-}|\cos (\varphi_{-})$ and $A_{\sin} = |A_{+}|\sin (\varphi_{+}) - |A_{-}|\sin (\varphi_{-})$ .

Given a mechanical system which is oscillating coherently at a frequency $\omega_{m}$ ( $\beta_{0}$ is simply a complex number) the single sided spectral density of the power at the detector, as a function of the laser detuning $\Delta$ , and frequency $\omega$ , will be given by

$$
S _ {\mathrm{PP}} (\omega , \Delta) = \hbar^ {2} \omega_ {0} ^ {2} \kappa_ {e} ^ {2} \left| \alpha_ {\text {in}, 0} \right| ^ {4} \times \left| \frac {i g \beta_ {0}}{(i \Delta + \kappa / 2) (i (\Delta - \omega_ {m}) + \kappa / 2)} \right| ^ {2} \times \delta (\omega - \omega_ {m}) \tag {A13}
$$

$$
= \hbar^ {2} \omega_ {0} ^ {2} \frac {g ^ {2} | \beta_ {0} | ^ {2} \kappa_ {e} ^ {2} | \alpha_ {\mathrm{in,0}} | ^ {4}}{(\Delta^ {2} + (\kappa / 2) ^ {2}) ((\Delta - \omega_ {m}) ^ {2} + (\kappa / 2) ^ {2})} \times \delta (\omega - \omega_ {m}).
$$

For mechanical systems undergoing random oscillations, the important quantity is the power spectral density of the detected signal. Since this is calculated from the autocorrelation functions, it will contain products of the form $\beta_{0}^{*}(t)\beta_{0}(t^{\prime})$ . Classically, these averages may be calculated from the Boltzmann distribution, and can be replaced with $\bar{n}_{T}=k_{B}T_{b}/\hbar\omega_{m}$ , with $k_{B}$ the Boltzmann constant, and $T_{b}$ the bath temperature. Additionally, since the measured sideband is blue of the pump frequency $(\omega_{1}+\omega_{m})$ , from the quantum theory [23, 24], and the derivation below, the proper ordering to be used is the normal one $(b^{\dagger}b)$ , and thus the expectation values may be replaced with $\bar{n}$ , the number of phonons occupying the mechanical mode. As such, we can effectively use the derivations shown above, in both the classical and quantum cases, substituting $\beta_{0}$ by $\sqrt{\bar{n}}$ , and replacing the delta functions $\delta(\omega-\omega_{m})$ with unit-area Lorentzian functions. A fully quantum mechanical derivation of this result is shown below.

# 2. Quantum Mechanical Derivation of Observed Spectra

In this section we use the following conventions for Fourier transforms and spectral densities. Given an operator A, we take

$$
\hat {A} (t) = \frac {1}{\sqrt {2 \pi}} \int_ {- \infty} ^ {\infty} d \omega e ^ {- i \omega t} \hat {A} (\omega),
$$

$$
\hat {A} (\omega) = \frac {1}{\sqrt {2 \pi}} \int_ {- \infty} ^ {\infty} d t e ^ {i \omega t} \hat {A} (t),
$$

$$
S _ {A A} (\omega) = \int_ {- \infty} ^ {\infty} d \tau e ^ {i \omega \tau} \langle \hat {A} ^ {\dagger} (t + \tau) \hat {A} (t) \rangle .
$$

Additionally we define the symmetrized spectral density as $\bar{S}_{AA}(\omega)=\frac{1}{2}(S_{AA}(\omega)+S_{AA}(-\omega))$ , and one-sided spectral densities which are those measured by the spectrum analyzer as $\bar{S}_{A}(\omega)=2\bar{S}_{AA}(\omega)$ . Starting from the quantum-optical Langevin equations for the mechanical ( $\hat{b}$ ) and optical ( $\hat{a}$ ) annihilation operators,

$$
\dot {\hat {b}} (t) = - \left(i \omega_ {m} + \frac {\gamma_ {i}}{2}\right) \hat {b} - i g \hat {a} ^ {\dagger} \hat {a} - \sqrt {\gamma_ {i}} \hat {b} _ {\text {in}} \quad \text {and} \tag {A14}
$$

$$
\dot {\hat {a}} (t) = - \left(i \Delta + \frac {\kappa}{2}\right) \hat {a} - i g \hat {a} (\hat {b} ^ {\dagger} + \hat {b}) - \sqrt {\kappa_ {e} / 2} \hat {a} _ {\text { in }} (t) - \sqrt {\kappa^ {\prime}} \hat {a} _ {\text { in }, i} (t), \tag {A15}
$$

we linearize the equations about a large optical field intensity by displacing $\hat{a} \to \alpha_0 + \hat{a}$ . Then in Fourier domain the fluctuations are then given by

$$
\hat {b} (\omega) = \frac {- \sqrt {\gamma_ {i}} \hat {b} _ {\text {in}} (\omega)}{i (\omega_ {m} - \omega) + \gamma_ {i} / 2} - \frac {i G (\hat {a} (\omega) + \hat {a} ^ {\dagger} (\omega))}{i (\omega_ {m} - \omega) + \gamma_ {i} / 2}, \tag {A16}
$$

$$
\hat {a} (\omega) = \frac {- \sqrt {\kappa_ {e} / 2} \hat {a} _ {\text { in }} (\omega) - \sqrt {\kappa^ {\prime}} \hat {a} _ {\text { in } , i} - i G (\hat {b} (\omega) + \hat {b} ^ {\dagger} (\omega))}{i (\Delta - \omega) + \kappa / 2}, \tag {A17}
$$

where $\kappa^{\prime}=\kappa-\kappa_{e}/2$ denotes all the optical loss channels which go undetected (i.e. information is lost) and $G=g\alpha_{0}$ . For an ideal measurement, $\kappa_{e}/2=\kappa$ , and $\kappa^{\prime}=0$ , so the intrinsic vacuum fluctuations ( $\hat{a}_{in,i}$ ) never enter the optical cavity. For a double sided coupling scheme, such as the one with a fiber taper, $\kappa=\kappa_{i}+\kappa_{e}$ , and so $\kappa^{\prime}=\kappa_{e}/2$ at best, due to the back reflection from the cavity, which contains information about the mechanics which is lost.

We account for all the fluctuations (vacuum and thermal) incident on the photodetector, and calculating the spectra of each term, we find the heterodyne detected signal. Using Equations (A16) and (A17) we arrive at the operator for the mechanical fluctuations,

$$
\begin{array}{l} \hat {b} (\omega) = \frac {- \sqrt {\gamma_ {i}} \hat {b} _ {\mathrm{in}} (\omega)}{i (\omega_ {m} - \omega) + \gamma / 2} \\ + \frac {i G}{i (\Delta - \omega) + \kappa / 2} \frac {\sqrt {\kappa_ {e} / 2} \hat {a} _ {\text { in }} (\omega) + \sqrt {\kappa^ {\prime}} \hat {a} _ {\text { in } , 1} (\omega)}{i (\omega_ {m} - \omega) + \gamma / 2} \\ + \frac {i G}{- i (\Delta + \omega) + \kappa / 2} \frac {\sqrt {\kappa_ {e} / 2} \hat {a} _ {\text {in}} ^ {\dagger} (\omega) + \sqrt {\kappa^ {\prime}} \hat {a} _ {\text {in} , i} ^ {\dagger} (\omega)}{i (\omega_ {m} - \omega) + \gamma / 2} \tag {A18} \\ \end{array}
$$

where $\omega_{m}$ is now the optical-spring shifted mechanical frequency, and $\gamma=\gamma_{i}+\gamma_{OM}$ , the optically damped mechanical loss-rate.

a. Simplified Result under RWA ( $\kappa^{2}/16\omega_{m}^{2}\ll1$ ) and Weak-Coupling ( $G\ll\kappa$ )

Assuming that $\Delta = \omega_{m}$ and that we care mainly about the system response around $\omega_{m}$ (where $\hat{b} (\omega)$ is peaked), the relation (A18) can be simplified,

$$
\hat {b} (\omega) = \frac {- \sqrt {\gamma_ {i}} \hat {b} _ {\text {in}} (\omega)}{i (\omega_ {m} - \omega) + \gamma / 2} + \frac {2 i G}{\kappa} \frac {\sqrt {\kappa_ {e} / 2} \hat {a} _ {\text {in}} + \sqrt {\kappa^ {\prime}} \hat {a} _ {\text {in} , i} (\omega)}{i (\omega_ {m} - \omega) + \gamma / 2} + \mathcal {O} \left(\frac {G}{2 \omega_ {m}}\right) \tag {A19}
$$

and we drop the term $\propto \frac{1}{2\omega_m}$ (RWA).

We find using the input-output boundary condition

$$
\begin{array}{l} \hat {a} _ {\text { out }} (\omega) = \hat {a} _ {\text { in }} (\omega) + \sqrt {\kappa_ {e} / 2} \hat {a} (\omega) + E _ {\mathrm{LO}} \delta (\omega) (A20) \\ = \hat {a} _ {\mathrm{in}} (\boldsymbol {\omega}) \left(1 - \frac {\kappa_ {e}}{\kappa} + \frac {4 | G | ^ {2}}{\kappa} \frac {\kappa_ {e}}{2 \kappa} \frac {1}{i (\boldsymbol {\omega} _ {m} - \boldsymbol {\omega}) + \gamma / 2}\right) \\ + \hat {a} _ {\mathrm{in}, i} (\boldsymbol {\omega}) \left(- \sqrt {\frac {2 \kappa^ {\prime} \kappa_ {e}}{\kappa^ {2}}} + \frac {4 | G | ^ {2}}{\kappa} \sqrt {\frac {\kappa^ {\prime} \kappa_ {e}}{2 \kappa^ {2}}} \frac {1}{i (\boldsymbol {\omega} _ {m} - \boldsymbol {\omega}) + \gamma / 2}\right) \\ + \hat {b} _ {\text { in }} (\omega) \left(i G \sqrt {\frac {2 \gamma_ {i} \kappa_ {e}}{\kappa^ {2}}} \frac {1}{i (\omega_ {m} - \omega) + \gamma / 2}\right) + E _ {\mathrm{LO}} \delta (\omega) \\ = s _ {1 1} (\omega) \hat {a} _ {\text { in }} (\omega) + n _ {\text { opt }} (\omega) \hat {a} _ {\text { in }, i} (\omega) + s _ {1 2} (\omega) \hat {b} _ {\text { in }} (\omega) + E _ {\mathrm{LO}} \delta (\omega) (A21) \\ \end{array}
$$

with the scattering matrix elements above defined as in Ref. [9]. For the case where the mechanical bath is at zero temperature, the spectral density will be given simply by $|s_{11}(\omega)|^{2} + |n_{\mathrm{opt}}(\omega)|^{2} + |s_{12}(\omega)|^{2} = 1$ , as a result of all input fluctuations being uncorrelated, and therefore no feature is present at the mechanical frequency.

For the case of the $n_{b}>0$ , we find the autocorrelation of the detected normalized photocurrent $\hat{I}(t)=\hat{a}_{\mathrm{out}}(t)+\hat{a}_{\mathrm{out}}^{\dagger}(t)$ to be

$$
\begin{array}{l} S _ {I I} (\omega) = | s _ {1 1} (\omega) | ^ {2} + | n _ {\text { opt }} (\omega) | ^ {2} + | s _ {1 2} (\omega) | ^ {2} (n _ {b} + 1) + | s _ {1 2} (- \omega) | ^ {2} n _ {b} \\ = 1 + n _ {b} \left(\left| s _ {1 2} (\omega) \right| ^ {2} + \left| s _ {1 2} (- \omega) \right| ^ {2}\right) \\ = 1 + \frac {\kappa_ {e}}{2 \kappa} \frac {4 | G | ^ {2}}{\kappa} \left(\frac {\gamma_ {i} n _ {b} / \gamma}{\left(\omega_ {m} - \omega\right) ^ {2} + (\gamma / 2) ^ {2}} + \frac {\gamma_ {i} n _ {b} / \gamma}{\left(\omega_ {m} + \omega\right) ^ {2} + (\gamma / 2) ^ {2}}\right) \\ = 1 + \frac {\kappa_ {e}}{2 \kappa} \frac {8 | G | ^ {2}}{\kappa} \bar {S} _ {b b} (\omega) \tag {A22} \\ \end{array}
$$

where for the last step we've used the fact that in the highly sideband-resolved regime, $\bar{n} = \gamma_{i}n_{b} / \gamma$ .

# 3. Optomechanical Damping

For the case where $\Delta = \omega_{m}$ (pumping on the red side of the cavity), the steady-state phonon amplitude can be written for a sideband resolved system far from strong coupling as [23, 24]

$$
\bar {n} = \frac {\gamma_ {i}}{\gamma_ {\mathrm{OM}} + \gamma_ {i}} n _ {b}, \tag {A23}
$$

with $n_{b}$ the equilibrium mechanical mode occupation number determined by the mechanical bath temperature, $\gamma_{i}$ the mechanical coupling rate to the bath and

$$
\gamma_ {\mathrm{OM}} = \frac {4 g ^ {2} | \alpha_ {0} | ^ {2}}{\kappa}, \tag {A24}
$$

the resonant optomechanical damping rate.

# Appendix B: Fabrication

The nano-beam cavities were fabricated using a Silicon-On-Insulator wafer from SOITEC ( $\rho = 4-20 \Omega\cdot cm$ , device layer thickness t = 220 nm, buried-oxide layer thickness 2 $\mu m$ ). The cavity geometry is defined by electron beam lithography followed by inductively-coupled-plasma reactive ion etching (ICP-RIE) to transfer the pattern through the 220 nm silicon device layer. The cavities were then undercut using a HF:H $_{2}$ O solution to remove the buried oxide layer, and cleaned using a piranha/HF cycle [31]. The dimensions and design of the nanobeam will be discussed in detail elsewhere.

# Appendix C: Device Parameters

Under vacuum and cryogenic parameters, the optical resonance was found to have $Q_{o} = 4 \times 10^{5}$ (corresponding to $\kappa = 500 MHz$ ), $\omega_{o}/2\pi = 195 THz$ (corresponding to $\lambda_{o} = 1537 nm$ ), and a resonant transmission contrast of $\Delta T = 25\%$ . The mechanical mode was found to have $Q_{m} = 1.06 \times 10^{5}$ (corresponding to $\gamma_{i} = 35 kHz$ ) and $\omega_{m}/2\pi = 3.68 GHz$ . The optomechanical coupling rate was found to be $g/2\pi = 910 kHz$ .

# Appendix D: Experimental Setup

The detailed experimental setup used to measure the cooling spectra and the electromagnetically-induced transparency (EIT) window of the optomechanical crystal is shown in Figure (5). The setup is designed to measure both the EIT-like reflected signal and the transmission signal of the laser used to cool the mechanical system (though not simultaneously).

As a light source we use a fiber-coupled, tunable, near-infrared laser, (New Focus Velocity, model TLB-6328) spanning approximately 60 nm centered around 1550 nm, which has its intensity controlled by a variable optical attenuator (VOA). A small percentage (10%) of the laser intensity is sent to wavemeter (WM, High Finesse, WS/6 High Precision) for passive frequency stabilization of the laser. To minimize polarization dependent losses on the electro-optical-modulator (EOM), a fiber polarization controller (FPC) is placed before it.

The EOM is driven by the microwave source (RF S.G., Agilent, model E8257D-520). The RF signal is composed of an amplitude modulated RF-signal carrier swept between $\Delta = 1 - 8$ GHz modulated at the lock-in detection frequency, $\omega_{LI}$ . As

![](images/aa00ab6a91675126216114c0b73f975cfc1d4eaaf427e2c1970237f9ded55600.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    Laser -->|WM| RF
    RF -->|RF S.G.| lock-in
    lock-in -->|LF| EOM
    EOM -->|DC| VOA
    VOA -->|1 2 3| SW1
    SW1 --> EDFA
    EDFA -->|SW2| SW3
    SW3 --> D2
    D2 --> RSA
    SW3 --> D3
    D3 --> PM
    FPC -->|FPC| lock-in
    lock-in -->|Taper device| Cryostat
```
</details>

FIG. 5: Expanded experimental setup to include optical switches SW1, SW2 and SW3. The blue lines indicate the optical path for the cooling measurement (the '0' position of each of the switches), while the dashed black lines indicate the alternative switched paths (the '1' position of each of the switches). A single tunable laser is used as the cooling laser and mechanical transduction laser. A wavemeter (WM) is used to track and lock the laser's frequency to an absolute and relative value better than 100 MHz and 5 MHz, respectively. A calibrated (to better than 0.01 dB) variable optical attenuator (VOA) is used to set the cooling laser power. The transmitted component of the cooling laser beam that is sent into the optomechanical cavity is directed to an erbium doped fiber amplifier (EDFA), where the optical signal is pre-amplified before being detected on a high-speed photodetector (D2). The measured photocurrent from D2 is sent to a real-time spectrum analyzer, where the mechanical noise power spectrum is measured. A slowly modulated optical signal, on-resonance with the optical nanobeam cavity, is generated from the cooling laser beam via an amplitude electro-optic modulator (EOM) driven by a microwave source (RFSG). The reflected component of the on-resonance laser signal injected into the cavity is separated from the input via an optical circulator, sent to a photodetector (D1), and then demodulated on a lock-in amplifier. Paddle-wheel fiber polarization controllers (FPCs) are used to set the laser polarization at the input to the EOM and the input to the optomechanical cavity.

a result, the EOM modulation produces two probe sidebands at $\pm\Delta$ , each with a small amplitude modulation at the lock-in frequency.

A small portion of the signal from the EOM output (10%) is used as a DC control signal to keep the EOM level locked, compensating for any low frequency power drift during the experiment. The remaining laser light is passed through a circulator, a switch (SW1), and a fiber polarization controller (FPC). It is then coupled to a tapered and dimpled optical fiber (Taper) which has its position controlled with nanometer-scale precision.

Switches 2 and 3 (SW2 and SW3) determine the path that the light transmitted through the taper follows. In the normal configuration, the transmitted light is optically amplified by an erbium doped fiber amplifier (EDFA) and then detected by a high-speed photoreceiver (D2, New Focus, model 1554-B) which is connected to a real-time spectrum Analyzer (RSA, Tektronix RSA3408B). Detector 3 (D3) is used to measured the DC transmission response of the cavity. All the other configurations are used to calibrate the system as discussed in the calibration section.

Any reflected signal coming from the taper/device is detected by a high-gain photodetector (D1, New Focus, model 1811) and its signal is sent to a lock-in amplifier (L.I., SRS-830). The output from the in-phase and quadrature signals from the L.I. are recorded, producing the reflection scan shown in Figure (3c) of the main text.

# Appendix E: Calibration for Mechanical Mode Thermometry

To perform accurate mode thermometry, several optical switches were incorporated into the setup as shown in Figure (5). A $2 \times 2$ switch (SW1) was positioned on the input/output ports of the fiber taper to control the direction of light through the taper, allowing the characterization of taper insertion loss asymmetry. Another $2 \times 2$ switch (SW2) was placed at the input/output ports of the EDFA, allowing the characterization of the optical gain, $G_{EDFA}$ . Lastly, a $1 \times 2$ switch (SW3) was inserted before the RSA to switch the optical path between the RSA and a power meter (D3) that reads the total power, $P_{RSA}$ , reaching the RSA, which allow us to monitor total insertion loss and provide a calibration for the electronic gain, $G_{e}$ .

At the beginning of a measurement, the power into the taper from both SW1 paths is measured ( $P_{0}$ and $P_{1}$ for the '0' and '1' position of the switch respectively). The insertion loss of the taper $L_{taper}$ is also measured. Finally, a pickoff power is recorded for each of the SW1 paths providing a correspondence between the measured taper input powers and the RSA powers, $P_{RSA,0}$ and $P_{RSA,1}$ . From these calibration values, the insertion loss before ( $L_{0}$ ) and after ( $L_{1}$ ) the device (when the fiber taper is coupled) can be computed by assuming the bistability shift of the optical mode when sweeping the laser from blue to red is proportional to the dropped power. If we let primed values represent measurements made with the taper coupled to the device, and let $\Delta\lambda$ be

the bistability shift, we have

$$
\frac {P _ {\text { in } , 0}}{P _ {\text { in } , 1}} = \frac {P _ {0} L _ {0}}{P _ {1} L _ {1}} = \frac {\Delta \lambda_ {0}}{\Delta \lambda_ {1}} \tag {E1}
$$

$$
L _ {0} L _ {1} = L _ {\text { taper }} \frac {P _ {\mathrm{RSA} , 0} ^ {\prime}}{P _ {\mathrm{RSA} , 0}} = L _ {\text { taper }} \frac {P _ {\mathrm{RSA} , 1} ^ {\prime}}{P _ {\mathrm{RSA} , 1}} \tag {E2}
$$

from which $L_{0}$ and $L_{1}$ can be extracted. As a consequence, the intracavity power, $P_{in}$ , can be accurately determined.

We directly measure $G_{e}$ , by setting the attenuator to 0 dB attenuation to measure $P_{RSA}$ and the corresponding DC bias voltage $V_{DC}$ , and defining $G_{e} = V_{DC}/P_{RSA}$ . The purpose of this is a technical one; the dynamic range of an optical power meter is much larger than that of a voltmeter. This allows us to accurately determine the DC bias voltage of the detector for any amplitude of optical signal through

$$
V _ {\mathrm{DC}} ^ {\prime} = V _ {\mathrm{DC}} \frac {P _ {\mathrm{RSA}} ^ {\prime}}{P _ {\mathrm{RSA}}} = G _ {e} P _ {\mathrm{RSA}} ^ {\prime}. \tag {E3}
$$

This value is critical because for photodetector signals of the form $V_{PD}(t) = V_{\mathrm{DC}}'(1 + \beta \sin \Omega t)$ where $\beta$ is the modulation depth and $\Omega$ is the modulation frequency, we have simply

$$
\beta = \frac {\sqrt {2 P _ {\Omega} R _ {L}}}{V _ {\mathrm{DC}} ^ {\prime}}, \tag {E4}
$$

where $P_{\Omega}$ is the integrated spectral power at $\Omega$ , and we have assumed a detector load of $R_{L}$ and an RSA that reports $V_{RMS}$ .

During the measurement, two calibration values are measured for each point in the power-dependent cooling run: $G_{EDFA}^{\prime}$ and $P_{RSA}^{\prime}$ . The EDFA gain is measured by utilizing SW2 to insert and remove the EDFA from the optical train, while measuring a fixed tone at the mechanical frequency $\omega_{m}$ generated by the Electro-Optic Amplitude Modulator (EOM). The ratio of the integrated spectral power of the tones gives $G_{EDFA}^{\prime2}$ . We also measure $P_{RSA}^{\prime}$ without the EDFA in line so that when the EDFA is included, we have instead of Equation (E4),

$$
\beta = \frac {\sqrt {2 P _ {\Omega} R _ {L}}}{G _ {\mathrm{EDFA}} ^ {\prime} G _ {e} P _ {\mathrm{RSA}} ^ {\prime}} \tag {E5}
$$

to account for the additional optical gain.

Finally, by integrating the Lorentzian component of the power spectral density from Equation (A13), we relate the detected signal on the spectrum analyzer to the calibrated values shown in Equation (E5). This allows us to make accurate determinations of the phonon number and mode temperature.

# Appendix F: EIT Measurements

Here we will show how the amplitude modulation of the signal sideband $\Delta$ is used to measure the reflection ( $|r(\omega)|^{2}$ ) of the signal reflected from the cavity. The output of the EOM can be written as:

$$
a _ {\text { out }} (t) = a _ {\text { in }} \left[ 1 + \beta \left(1 + m _ {\mathrm{LI}} \cos (\omega_ {\mathrm{LI}} t)\right) \cos (\Delta t) \right], \tag {F1}
$$

where the input field amplitude $a_{\mathrm{in}}(t) = a_{o} \cos(\omega_{\mathrm{l}} t)$ , $a_{o} = \sqrt{P_{\mathrm{in}} / \hbar \omega_{\mathrm{l}}}$ , $\beta$ is the EOM-modulation index, $\omega_{LI}$ and $m_{LI}$ are, respectively, the frequency and amplitude modulation index on the RF signal at $\Delta$ . For the measurements shown in the main text $m_{LI} = 1$ . In this case one can write the field of the EOM output (cavity input) in the time domain as:

$$
\begin{array}{l} a _ {\text { out }} (t) = a _ {o} \left[ \cos (\omega_ {1} t) + \frac {\beta}{2} \left[ \cos ((\omega_ {1} + \Delta) t) + \cos ((\omega_ {1} - \Delta) t) \right] \right. \\ + \frac {\beta}{4} \left[ \cos \left(\left(\omega_ {1} + \Delta + \omega_ {L I}\right) t\right) + \cos \left(\left(\omega_ {1} + \Delta - \omega_ {L I}\right) t\right) \right. \\ \left. \left. + \cos ((\omega_ {\mathrm{l}} - \Delta + \omega_ {\mathrm{LI}}) t) + \cos ((\omega_ {\mathrm{l}} - \Delta - \omega_ {\mathrm{LI}}) t) \right] \right]. \tag {F2} \\ \end{array}
$$

The reflected signal is filtered by the cavity dispersion and considering the case where the pump is on the red-side of the cavity $(\omega_{1} < \omega_{o})$ the reflected field is:

$$
a _ {R} (t) = r \left(\omega_ {s}\right) \frac {a _ {o} \beta}{4} \left[ \cos \left(\left(\omega_ {\mathrm{l}} + \Delta\right) t\right) + \cos \left(\left(\omega_ {\mathrm{l}} + \Delta\right) t + \left(\omega_ {\mathrm{LI}} t - \varphi\right)\right) + \cos \left(\left(\omega_ {\mathrm{l}} + \Delta\right) t - \left(\omega_ {\mathrm{LI}} t - \varphi\right)\right) \right] \tag {F3}
$$

First we assume that $r(\omega_{s})$ is roughly constant over a range of $\omega_{LI}$ which is true for $\omega_{\mathrm{LI}} < (\gamma_{i} + \gamma_{\mathrm{OM}})/2$ . This implies that the smallest transparency window we could measure is on the order of the lock-in detection frequency. This limit on the transparency window size is reflected on Fig. 4c, where only transparency windows larger 200 kHz are reported.

We can now write the time average detected power spectral density on the photoreceiver (D1 in Figure (5)) by taking the absolute square value of the reflected field and keeping only the terms with frequency smaller than the detector bandwidth. In this case:

$$
P | _ {\omega_ {s}} = \frac {a _ {o} ^ {2} \beta^ {2} R _ {\mathrm{PD}} G _ {\mathrm{PD}}}{8 R _ {L}} | r (\omega_ {s}) | ^ {2} \left[ 3 + 4 \cos (\omega_ {\mathrm{LI}} t - \varphi) + \frac {1}{2} \cos (2 \omega_ {\mathrm{LI}} t - 2 \varphi) + \mathcal {O} (2 \omega_ {\mathrm{l}}) \right].
$$

where $R_{PD} = 1 \, A/W$ is the detector responsivity, $G_{PD} = 40,000 \, V/A$ is the detector gain and $R_{L} = 50 \, \Omega$ is the load resistance.

This signal is then sent to the lock-in which can measure independently the in-phase (X) and quadrature (Y) power spectral densities at $\omega_{LI}$ :

$$
X | _ {\omega_ {\mathrm{LI}}} = \frac {a _ {o} ^ {2} \beta^ {2} R _ {\mathrm{PD}} G _ {\mathrm{PD}}}{4 R _ {L}} | r (\omega_ {s}) | ^ {2} \cos (\varphi)
$$

$$
Y | _ {\omega_ {\mathrm{LI}}} = \frac {a _ {o} ^ {2} \beta^ {2} R _ {\mathrm{PD}} G _ {\mathrm{PD}}}{4 R _ {L}} | r (\omega_ {s}) | ^ {2} \sin (\varphi) \tag {F4}
$$

It is then easy to see the reflection amplitude and phase are given by

$$
\left| r \left(\omega_ {s}\right) \right| ^ {2} = \frac {4 R _ {L}}{a _ {o} ^ {2} \beta^ {2} R _ {\mathrm{PD}} G _ {\mathrm{PD}}} \sqrt {X \left| _ {\omega_ {\mathrm{LI}}} ^ {2} + Y \right| _ {\omega_ {\mathrm{LI}}} ^ {2}} \quad \text { and } \quad \tan (\varphi) = \frac {Y | _ {\omega_ {\mathrm{LI}}}}{X | _ {\omega_ {\mathrm{LI}}}}.
$$

From the imparted change in the phase the signal delay is then calculated as:

$$
\tau^ {\mathrm{(R)}} = \frac {\varphi}{\omega_ {\mathrm{LI}}}
$$

where $\tau^{(R)} > 0$ ( $\tau^{(R)} < 0$ ) represent a delay (advance) on the signal.

Here we have neglected the gain provided by the lock-in, which is important to determine the absolute value of $r(\omega_{s})$ . To account for that we calibrate the X channel by a normalized transmission curve taken with low input power. Our assumption is that the cavity-taper coupling is not affected by the input power. A analogous result can be found for the case where the control laser is on the blue side of the cavity ( $\omega_{l} > \omega_{o}$ ). A more detailed description of the EIT experiment can be found in Ref. [26].

# Appendix G: Analyzing the Mechanical Mode Spectra

To determine the total spectral power at $\omega_{m}$ for a given measured spectra, we first subtract a background taken with the cooling laser far-detuned from the cavity (in the same calibration conditions). We then perform a least squares fit to a Lorentzian function of the form

$$
L (\omega) = \frac {A}{\left(\frac {\omega - \omega_ {m}}{2 \gamma}\right) ^ {2} + 1} \tag {G1}
$$

with fit parameters A, $\omega_{m}$ and $\gamma$ . The spectral power is then given simply by

$$
P _ {\omega_ {m}} = \frac {A \gamma}{4}. \tag {G2}
$$

To extract the intrinsic linewidth $\gamma_{i}$ we first fix the input power $P_{\mathrm{in}}$ . We then lock the pump on the red side of the cavity (at $\Delta = +\omega_{m}$ ) and measure the total linewidth, $\gamma_{\mathrm{red}} = \gamma_{i} + \gamma_{\mathrm{OM}}^{\mathrm{(red)}}$ . We repeat the measurement on the blue side (at $\Delta = -\omega_{m}$ ), where $\gamma_{\mathrm{blue}} = \gamma_{i} - \gamma_{\mathrm{OM}}^{\mathrm{(blue)}}$ . Using low input powers where $\gamma_{\mathrm{OM}}^{\mathrm{(blue)}} \ll \gamma_{i}$ to avoid amplification of the mechanical oscillations, we have $\gamma_{\mathrm{OM}}^{\mathrm{(red)}} = \gamma_{\mathrm{OM}}^{\mathrm{(blue)}}$ , which leads to

$$
\gamma_ {i} = \frac {\gamma_ {\mathrm{red}} + \gamma_ {\mathrm{blue}}}{2}. \tag {G3}
$$

Equation (A13) shows an explicit form for the sideband power amplitude seen by a photodetector for a red detuned pump laser. More specifically we can find a relation between the number of phonons inside the cavity and the power spectrum for our experimental setup. As shown before, the RF-spectra are detected via a RSA which displays the power spectral density of the voltage coming from the photodetector (with gain $G_{e}$ and $G_{EDFA}$ ). The single sided power spectral density at the detector is given by Equation (A13), and denoted $S_{\mathrm{PP}}(\omega)$ . Since the power is related to voltage by an electronic gain $G_{e}$ , then $S_{VV}(\omega) = G_{e}^{2} S_{\mathrm{PP}}(\omega)$ . When the EDFA is used, there is an additional gain factor, and $S_{VV}(\omega) = G_{\mathrm{EDFA}}^{2} G_{e}^{2} S_{\mathrm{PP}}(\omega)$ . Finally, the RSA reports power as opposed to squared voltages, and so the final spectral density measured is $S(\omega) = S_{VV}(\omega)/2 R_{L}$ , where $R_{L} = 50 \Omega$ is the input impedance of the RSA and the factor of two in the denominator comes from the conversion of peak-to-peak voltage to RMS voltage. Then, in terms of integrated power, the power detected in the sideband on the RSA, $P_{\omega_{m}}$ is related to the heterodyne detected integrated spectral density by the relation

$$
P _ {\omega_ {m}} = \frac {(G _ {e} G _ {\mathrm{EDFA}}) ^ {2}}{2 R _ {L}} P _ {\mathrm{SB}}. \tag {H1}
$$

We would like to write this equation as a function of all the independent variables measured for the system, and from that estimate the error on the measured number of phonons.

From the DC transmission spectra, the optical components $\kappa$ , $\kappa_{e}$ and $\omega_{0}$ can be determined. Both $G_{EDFA}$ and $G_{e}$ are measured and latter compensates for any discrepancy in the value of $R_{L}$ . From the RF-spectra one can determine the total mechanical linewidth $\gamma = \gamma_{i} + \gamma_{OM}$ , the mechanical frequency $\omega_{m}$ and the total RF-power $P_{RSA}$ . The EIT spectra give the true detuning $\Delta$ between the pump laser and the cavity and, as shown before, can be used to determine the power at the cavity $P_{in}$ .

Using Equations (A24) and (A9) we can rewrite the integrated form of Equation (A13) as:

$$
P _ {\mathrm{SB}} = \left(\hbar \omega_ {0}\right) ^ {2} \frac {\left(\kappa_ {e} / 2\right) N _ {\text {in}}}{\left(\Delta - \omega_ {m}\right) ^ {2} + (\kappa / 2) ^ {2}} \kappa \left(\gamma - \gamma_ {i}\right) \bar {n} \tag {H2}
$$

From Equation (H2) we can write the expression that relates the number of phonons, $\bar{n}$ , and all the system parameters as:

$$
\bar {n} = \left(\frac {2 R _ {L}}{G _ {e} ^ {2} G _ {\mathrm{EDFA}} ^ {2}} \frac {P _ {\omega_ {m}}}{\hbar \omega_ {0}}\right) \left(\frac {1}{\kappa (\gamma - \gamma_ {i})}\right) \left(\frac {(\Delta - \omega_ {m}) ^ {2} + (\kappa / 2) ^ {2}}{(\kappa_ {e} / 2) P _ {\mathrm{in}}}\right) \tag {H3}
$$

We can calculate the cumulative error for the number of measured phonons on the cavity using this equation and based upon on the measurable variables which gives:

$$
\begin{array}{l} \frac {\Delta \bar {n}}{\bar {n}} = \left[ \frac {\delta \omega_ {0} ^ {2}}{\omega_ {0} ^ {2}} + \frac {\delta \kappa_ {e} ^ {2}}{\kappa_ {e} ^ {2}} + \frac {\delta P _ {\text {in}} ^ {2}}{P _ {\text {in}} ^ {2}} + \frac {\delta P _ {\mathrm{RSA}} ^ {2}}{P _ {\mathrm{RSA}} ^ {2}} + \frac {\delta \gamma_ {i} ^ {2}}{(\gamma - \gamma_ {i}) ^ {2}} + \frac {\delta \gamma^ {2}}{\gamma - \gamma_ {i}) ^ {2}} + \dots \right. \tag {H4} \\ \left. + \left(\frac {\kappa / 2}{(\kappa / 2) ^ {2} + (\Delta - \omega_ {m}) ^ {2}} - \frac {1}{\kappa}\right) ^ {2} \delta \kappa^ {2} + \left(\frac {2 (\Delta - \omega_ {m})}{(\kappa / 2) ^ {2} + (\Delta - \omega_ {m}) ^ {2}}\right) ^ {2} \delta \Delta^ {2} + \left(\frac {2 (\Delta - \omega_ {m})}{(\kappa / 2) ^ {2} + (\Delta - \omega_ {m}) ^ {2}}\right) ^ {2} \delta \omega_ {m} ^ {2} \right] ^ {1 / 2} \\ \end{array}
$$

Here we neglected the error on $G_{e}$ and $G_{EDFA}$ which are much smaller than any other error quantity. To determine the variation for $\kappa$ , $\kappa_{e}$ and $\omega_{0}$ , we measured the DC optical spectrum for every single data point in Figure (4a) of the main text and determined $\delta\kappa$ , $\delta\kappa_{e}$ and $\delta\omega_{0}$ from the normalized standard deviations of each of the values. The measurement uncertainty of these values are below 0.7%. The mechanical properties $\delta\gamma$ , $\delta P_{RSA}$ and $\delta\omega_{m}$ , were determined from the deviation on the spectra fits using a 95% confidence interval, which produces percent errors below 0.6%. The pump laser detuning from the cavity is controlled by the EIT reflection spectra. To find the variation of the detuning $\delta\Delta$ we once again computed the standard deviation of all the measured detunings, which results in a deviation of less than 0.3%.

Finally, the two main sources of error in our data are the determination of the intrinsic mechanical quality factor (reflected in $\gamma_{i}$ ) and the input power, $P_{in}$ . The uncertainty in the mechanical linewidth, $\delta\gamma_{i}$ , is found by repeatedly measuring it at a single power level and computing its standard deviation (found to be $\sim1.6\%$ ). Using the calibration procedure discussed above for $P_{in}$ , the error lies in the determination of losses $L_{0}$ and $L_{1}$ . In the worst case the calibration would be off by the ratio between the input loss, $L_{0}$ in the present experiment, and the square root of total loss $\sqrt{L_{0}L_{1}}$ producing a percentage error of $\sim4.0\%$ to the input power.

Taking all of these factors into account produces an overall uncertainty of $\sim4.5\%$ in the measured absolute phonon number.

Absorption in the dielectric cavity causes the temperature of the dielectric cavity to increase locally. This effect is expressed through shifts in the refractive index of the structure, and the thermo-optic coefficient of Silicon [32]. As such, we can estimate the temperature of the cavity by looking at the shift in the cavity frequency, starting from a known temperature.

The starting point of the analysis is the cavity-perturbation formula for dielectric cavities [33],

$$
\frac {\omega - \omega_ {0}}{\omega_ {0}} \approx - \frac {1}{2} \frac {\int \delta \varepsilon (\mathbf {r}) | \mathbf {E} (\mathbf {r}) | ^ {2} \mathrm{d} \mathbf {r}}{\int \varepsilon (\mathbf {r}) | \mathbf {E} (\mathbf {r}) | ^ {2} \mathrm{d} \mathbf {r}}. \tag {I1}
$$

From the relation $\varepsilon / \varepsilon_0 = n^2$ , we find $\delta \varepsilon = 2n\delta n\varepsilon_0$ . By assuming that the cavity as a whole is heated to a temperature $T_0$ , the integral in Equation (I1) can be written as

$$
\omega - \omega_ {0} \approx - n (T _ {0}) \omega_ {0} \frac {\int_ {\mathrm{Si}} | \mathbf {E} (\mathbf {r}) | ^ {2} \mathrm{d} \mathbf {r}}{\int (n (T _ {0})) ^ {2} | \mathbf {E} (\mathbf {r}) | ^ {2} \mathrm{d} \mathbf {r}} \times (n (T) - n (T _ {0})). \tag {I2}
$$

Using the values of $n(T)$ found in literature [32], and a value of

$$
\frac {\int_ {\mathrm{Si}} | \mathbf {E} (\mathbf {r}) | ^ {2} \mathrm{d} \mathbf {r}}{\int (n (T _ {0})) ^ {2} | \mathbf {E} (\mathbf {r}) | ^ {2} \mathrm{d} \mathbf {r}} \approx 7. 5 0 6 6 \times 1 0 ^ {- 2},
$$

calculated from the finite element simulations (FEM) of the mode profiles, we plot the wavelength shift from 17.6 K up to 300 K in Figure (6a). The total shift of 12.5 nm agrees with the experimentally observed change in resonance wavelength.

![](images/4fbf170d70b9a409503f5f0837618eb30bed9e4038e60fd3d24229cc81dfbf03.jpg)

<details>
<summary>line</summary>

| Cavity Temperature (K) | Δλ (nm) |
| ---------------------- | ------- |
| 0                      | 0.0     |
| 10                     | 0.0     |
| 20                     | 0.0     |
| 30                     | 0.0     |
| 40                     | 0.0     |
| 50                     | 0.0     |
| 60                     | 0.0     |
| 70                     | 0.0     |
| 80                     | 0.5     |
| 90                     | 1.0     |
| 100                    | 2.0     |
| 150                    | 4.0     |
| 200                    | 6.0     |
| 250                    | 8.0     |
| 300                    | 12.0    |
</details>

![](images/ed3d5190ab36696cffaf43080e6a6bbae911a20129d09b4cfba1b8588624ec66.jpg)

<details>
<summary>line</summary>

| Intracavity Photons (N_cav) | upper bound | lower bound |
| --------------------------- | ----------- | ----------- |
| 0                           | 0           | 0           |
| 400                         | ~1          | ~-3         |
| 800                         | ~3          | ~-6         |
| 1200                        | ~5          | ~-9         |
| 1600                        | ~7          | ~-12        |
| 2000                        | ~10         | ~-15        |
</details>

FIG. 6: a, the measured wavelength shift compared to the theoretical shift predicted by Equation (I2) for a range of cavity temperatures, using 17.6 K as the reference point. b, the measured power-dependent wavelength shift of the cavity with fitted individual contributions due to free carrier dispersion (blue) and refractive index change (red), as well as their sum (black), for the two bounds discussed in the text.

This analysis can be applied to the wavelength shift data for various input powers at low temperature where the initial cavity temperature is measured by thermometry methods discussed above. We note an initial blue-shift of the cavity, which is attributed to free-carrier dispersion effects [34] and can be modeled by a power law dependence on intracavity photon number, $A(n_{c})^{B}$ . The temperature-dependent data for the refractive index of Silicon in Ref. [32] is valid only for $T > 30\mathrm{K}$ so the power-dependent cavity heating for a starting temperature of $17.6\mathrm{K}$ , for the largest intracavity photon number, can only be bounded. For the upper bound, we assume $dn / dT = 0$ for $T < 30\mathrm{K}$ , resulting in a $\Delta T_{\mathrm{max}}$ of $16.8\mathrm{K}$ . For the lower bound, we assume $dn / dT = dn / dT|_{T = 30\mathrm{K}}$ for $T < 30\mathrm{K}$ , resulting in a $\Delta T_{\mathrm{min}}$ of $7.8\mathrm{K}$ . These bounds and their respective fits are shown in Figure (6b).

# Appendix J: Temperature-Dependent modifications to the intrinsic mechanical damping

Independent measurements of the mechanical quality factor, $Q_{m}$ , at varying bath temperatures indicate that the $Q_{m}$ changes with temperature (Figure (7a)). These measurements are taken at low intracavity photon number, rendering free-carrier effects negligible. As such we can model the mechanical loss rate as $\gamma_{i}(T)=\gamma_{i,T}(T)+\gamma_{i}^{(0)}$ where $\gamma_{i}^{(0)}$ is the measured loss rate at the reference temperature (17.6 K). The extracted form of $\gamma_{i,T}(T)$ is shown in Figure (7b).

![](images/dda2919acb36a759a612322df2843f4e54c4884227aa22fee286f096eb5bd6d9.jpg)

<details>
<summary>scatter</summary>

| Cavity Temperature (K) | Q^m     |
| ---------------------- | ------- |
| 20                     | 10500   |
| 20                     | 10400   |
| 20                     | 10300   |
| 20                     | 9500    |
| 25                     | 8200    |
| 35                     | 7400    |
| 45                     | 6500    |
| 50                     | 6000    |
| 55                     | 5200    |
| 65                     | 4500    |
| 75                     | 3700    |
| 90                     | 2300    |
| 250                    | 10      |
</details>

![](images/c201e3ae8e7f7c42c3f3613efbaa7846583ece6ba944326b7c19108c951728f8.jpg)

<details>
<summary>scatter</summary>

| Cavity Temperature (K) | γ_i,t / 2π (MHz) |
| ---------------------- | ---------------- |
| 20                     | 10^-3            |
| 25                     | 10^-2            |
| 30                     | 10^-2            |
| 40                     | 10^-1            |
| 50                     | 10^-1            |
| 60                     | 10^-1            |
| 70                     | 10^-1            |
| 80                     | 10^-1            |
| 90                     | 10^-1            |
| 100                    | 10^-1            |
| 250                    | 10^0             |
</details>

FIG. 7: a, the measured intrinsic mechanical quality factor for various cavity temperatures. b, the inferred intrinsic mechanical loss rate due to temperature, $\gamma_{i,T}$ , modeled by a polynomial fit.

# Appendix K: Photon-number dependent modifications to the intrinsic mechanical damping

The deviation of the expected cooled phonon number from the measured value is a result of two factors: bath heating and an increase in the intrinsic mechanical loss rate $\left(\gamma_{i}\right)$ due to heating and free carriers. Since the integrated spectral power of the mechanical mode depends only on the product $\gamma_{i}T_{b}$ (for large intracavity photon numbers $n_{c}$ ), naively ignoring the latter effect results in an estimated change of $\Delta T > 50$ K in the bath temperature for 2000 intracavity photons. This is unrealistic as such a temperature change would tune the optical mode red by >300 pm (from Equation (I2)), while the actual measured shift is closer to 10 – 20 pm (from Figure (6b)). In fact through independent measurements (where dynamic back-action was minimized), we found that the mechanical linewidth is a function of the number of photons in the cavity. We attribute this to a nonlinear process in the cavity involving the generation of free carriers, which will be explored in depth elsewhere, and introduce an additional loss channel $\gamma_{i,FC}$ in the mechanical loss rate so that we have

$$
\gamma_ {i} \rightarrow \gamma_ {i} \equiv \gamma_ {i} ^ {(0)} + \gamma_ {i, T} (T (n _ {c})) + \gamma_ {i, \mathrm{FC}} (n _ {c}). \tag {K1}
$$

From the relations shown on previous sections, we have $\gamma_{\mathrm{cooled}}^{(0)} = \gamma_{i}^{(0)} + \gamma_{\mathrm{OM}}$ . Incorporating Equation (K1), we have experimentally, $\gamma_{cooled} = \gamma_{i} + \gamma_{OM}$ , with their difference yielding the magnitude of the additional loss rates. However, for $\Delta = \omega_{m}$ and $n_{c} > 10$ , $\gamma_{OM}$ tends to be large compared to $\gamma_{i}$ , making this subtraction quite error prone. Thus, to get accurate data for high intracavity photon numbers we use a range of larger detunings, noting that $\gamma_{OM} \propto \Delta^{-2}$ for $\Delta \gg \omega_{m}$ and fixed $n_{c}$ (approximately). This loss is then modeled using a power law dependence on $n_{c}$ (Figure (8a)).

Using the models of mechanical loss from Figure (8a) and Figure (7b) with the thermometry technique outlined earlier (making the replacement to $\gamma_{i}$ ) allows a more accurate determination of the temperature rise in the cavity, as well as the characterization of the individual contributions of $\gamma_{i,T}$ and $\gamma_{i,FC}$ as a function of $n_{c}$ . The result is an estimated increase of 13.2 K in $T_{b}$ at the highest input power, well within the previously fitted temperature bounds.

The addition of a free carrier related loss channel is further corroborated by pumping the Si sample above the band gap with a 532 nm solid state green laser, directly stimulating the production of free carriers. The degradation in $Q_{m}$ can be only partially explained by heating due to absorption since the maximum 19 K temperature rise estimated from the cavity red-shift results in an expected $Q_{m}$ of approximately 70,000 at the highest power (Figure (7a)), whereas a far lower value is measured. The remaining excess loss is attributed to the presence of free-carriers. These results are shown in Figure (9).

# Appendix L: Shot Noise Considerations

We consider here the impact of using a non-ideal amplifier (EDFA) on the measured signal, and the deviation from quantum limits. For a coherent optical beam with frequency $\omega_{l}$ and power P incident on a photo detector, the single sided power spectral density of the shot noise is simply

$$
S _ {\text { shot }} (\omega) = \sqrt {2 \hbar \omega_ {\mathrm{l}} P}, \tag {L1}
$$

![](images/15c35c65927a574cb44cfcc5112f4fb63f405232085ac39eeda3d20633a74ce2.jpg)

<details>
<summary>scatter</summary>

| Intracavity Photons | (γi,T + γi,FC)/2π (kHz) |
| ------------------- | ------------------------ |
| 1                   | -8                       |
| 2                   | -3                       |
| 3                   | -1                       |
| 4                   | 0                        |
| 5                   | 1                        |
| 6                   | 2                        |
| 7                   | 3                        |
| 8                   | 4                        |
| 9                   | 5                        |
| 10                  | 6                        |
| 15                  | 7                        |
| 20                  | 8                        |
| 25                  | 9                        |
| 30                  | 10                       |
| 40                  | 11                       |
| 50                  | 12                       |
| 60                  | 13                       |
| 70                  | 14                       |
| 80                  | 15                       |
| 90                  | 16                       |
| 100                 | 17                       |
| 150                 | 18                       |
| 200                 | 19                       |
| 250                 | 20                       |
| 300                 | 21                       |
| 400                 | 22                       |
| 500                 | 23                       |
| 600                 | 24                       |
| 700                 | 25                       |
| 800                 | 26                       |
| 900                 | 27                       |
| 1000                | 28                       |
</details>

![](images/c4f6844b0a1c134dcb085b3e57e897b24dbf18187b954ea73d5725bc38a28ec9.jpg)

<details>
<summary>line</summary>

| Intracavity Photons | γ/2π (kHz) |
| ------------------- | ---------- |
| 1                   | 35         |
| 10                  | 35         |
| 100                 | 38         |
| 1000                | 85         |
</details>

FIG. 8: a, excess loss as a function of $n_{c}$ , inferred from far-red-detuned ( $\Delta > 5.5$ GHz) measurements of $\gamma_{cooled}$ . b, breakdown showing the individual contributions of $\gamma_{i}^{(0)}$ (gray), $\gamma_{i,T}$ (red), and $\gamma_{i,FC}$ (blue) to the total $\gamma_{i}$ ( $\circ$ ).   
![](images/a215224c854276e350b9a3949051af9a8424d1460f80013a43459a6609aa5027.jpg)

<details>
<summary>scatter</summary>

| HeNe Power (nW) | Q_m (×10⁴) | Δλ (pm) |
| --------------- | ---------- | ------- |
| 10              | 10.5       | 0       |
| 100             | 10.3       | 0       |
| 1000            | 9.5        | 0       |
| 10000           | 7.0        | 5       |
| 100000          | 4.5        | 15      |
</details>

FIG. 9: The $Q_{m}$ degradation as a function of 532 nm laser power. The purple line shows the expected $Q_{m}$ for the bath temperature rise inferred from the wavelength shift data. The deviation of the data from this prediction suggests an additional loss channel related to the presence of free carriers.

independent of frequency. As part of the calibration procedure, $G_{EDFA}$ is measured to characterize the gain provided by the EDFA and $G_{e}$ is measured to characterize the transimpedance gain and quantum efficiency of the photodetector. These values can be used in conjunction with Equation (L1) to predict the expected spectral background assuming only the presence of shot noise and noise-free gain. To wit,

$$
S _ {\text { shot }} ^ {(\text { amplified })} = \sqrt {2 \hbar \omega_ {o} G _ {\mathrm{EDFA}} ^ {2} G _ {e} ^ {2} P _ {\mathrm{in}} ^ {\prime}}, \tag {L2}
$$

where the prime indicates the value has been adjusted to account for insertion loss from the cavity to the photodetector. The difference between this predicted level and the measured background level, $S_{background}$ , gives the non-ideality of the EDFA and is attributed to an excess noise which also includes the amplified spontaneous emission (ASE) noise [35]. This deviation is shown in Figure (10a). Defining

$$
S _ {\text { excess }} ^ {2} = S _ {\text { background }} ^ {2} - \left(S _ {\text { shot }} ^ {(\text { amplified })}\right) ^ {2} \tag {L3}
$$

this additional noise reduces the measured signal-to-noise ratio (SNR). We can predict the shot noise limited SNR, representing the largest measurable SNR for the current experimental setup set assuming perfectly efficient detection from the measured

![](images/09836fec4bcf0a36d18a88fa536163661bc9d10acd4ab4ed5e6fbacf9c990500.jpg)

<details>
<summary>line</summary>

| Intracavity Photons | measured (dBm/Hz) | noise-free gain (dBm/Hz) |
| ------------------- | ----------------- | ------------------------ |
| 1                   | -120              | -135                     |
| 10                  | -120              | -125                     |
| 100                 | -125              | -130                     |
| 1000                | -130              | -140                     |
| 2000                | -135              | -140                     |
</details>

![](images/cee97caad862117f3d0b60eb08a529e885b8c20dcdc3dacb68a924767567482b.jpg)

<details>
<summary>line</summary>

| Intracavity Photons | measured | noise-free gain | predicted |
| ------------------- | -------- | --------------- | --------- |
| 1                   | -10      | 5               | 15        |
| 10                  | -2       | 6               | 16        |
| 100                 | -5       | 0               | 8         |
| 1000                | -18      | -7              | 3         |
</details>

FIG. 10: a, a comparison of the measured background spectrum against the shot noise level amplified by an ideal, noise-free amplifier. b, a comparison of the measured signal-to-noise ratio (SNR) to the maximum achievable SNR for the experimental setup assuming an ideal, noise-free amplifier and perfect quantum detection; the predicted SNR uses only measured device and calibration parameters, along with $S_{excess}^{2}$ determined from (a); the purple dashed line shows SNR assuming an overcoupled system, where $\kappa_{e} = \kappa$ , with noise-free gain and the black line shows the same system without taper/insertion loss (representing the most ideal of ideal cases).

device and calibration parameters using Equation (H2) and (L1). This ratio is given by

$$
\mathrm{SNR} _ {\text { shot }} = \frac {4 P _ {\mathrm{SB}} ^ {\prime} / (\gamma_ {i} + \gamma_ {\mathrm{OM}})}{2 \hbar \omega_ {o} P _ {\mathrm{in}} ^ {\prime}}. \tag {L4}
$$

Similarly, the expected measurement SNR, using only device and calibration parameters, is given by

$$
\mathrm{SNR} _ {\text { predicted }} = \frac {4 G _ {\mathrm{EDFA}} ^ {2} G _ {e} ^ {2} P _ {\mathrm{SB}} ^ {\prime} / (\gamma_ {i} + \gamma_ {\mathrm{OM}})}{2 \hbar \omega_ {o} G _ {\mathrm{EDFA}} ^ {2} G _ {e} ^ {2} P _ {\mathrm{in}} ^ {\prime} + S _ {\text { excess }} ^ {2}}, \tag {L5}
$$

which closely corresponds to the measured values, as seen in Figure (10b).