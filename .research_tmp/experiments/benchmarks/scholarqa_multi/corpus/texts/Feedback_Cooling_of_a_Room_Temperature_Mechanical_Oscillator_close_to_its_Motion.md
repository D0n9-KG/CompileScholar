# Feedback Cooling of a Room Temperature Mechanical Oscillator close to its Motional Ground State

Jingkun Guo, $^{1}$ Richard Norte, $^{1,2,*}$ and Simon Gröblacher $^{1,\dagger}$

$^{1}$ Kavli Institute of Nanoscience, Department of Quantum Nanoscience, Delft University of Technology, 2628CJ Delft, The Netherlands

$^{2}$ Department of Precision and Microsystems Engineering, Delft University of Technology, Mekelweg 2, 2628CD Delft, The Netherlands

![](images/0b0192437c1ae86f310b992b99c4d5b32afcd1b596c45654c9a2c4f9d2c1b3d0.jpg)

(Received 18 September 2019; published 27 November 2019)

Preparing mechanical systems in their lowest possible entropy state, the quantum ground state, starting from a room temperature environment is a key challenge in quantum optomechanics. This would not only enable creating quantum states of truly macroscopic systems, but at the same time also lay the groundwork for a new generation of quantum-limited mechanical sensors in ambient environments. Laser cooling of optomechanical devices using the radiation pressure force combined with cryogenic precooling has been successful at demonstrating ground state preparation of various devices, while a similar demonstration starting from a room temperature environment remains an outstanding goal. Here, we combine integrated nanophotonics with phononic band gap engineering to simultaneously overcome prior limitations in the isolation from the surrounding environment and the achievable mechanical frequencies, as well as limited optomechanical coupling strength, demonstrating a single-photon cooperativity of 200. This new microchip technology allows us to feedback cool a mechanical resonator to around 1 mK, near its motional ground state, from room temperature. Our experiment marks a major step toward accessible, widespread quantum technologies with mechanical resonators.

DOI: 10.1103/PhysRevLett.123.223602

The last decade has seen immense progress on observing quantum effects with microfabricated mechanical oscillators $[1-6]$ . This is not only of significant interest for understanding the fundamental aspects of quantum physics in macroscopic objects, but also for the potential of using mechanical systems for quantum information processing tasks and as novel quantum sensors $[7]$ . Excess classical (i.e., thermal) noise typically obscures the quantum features of these devices, thus limiting their usefulness and practical adoption for quantum-limited sensing. Ground state cooling can alleviate this problem, but so far has only been possible by precooling the devices using cryogenic methods $[8-12]$ . The main limitations preventing reach of this regime from room temperature include insufficient isolation from the surrounding environment and too low mechanical frequencies, which can be formulated into the condition of the product of the mechanical frequency and its quality factor $f_{m}Q_{m} \geq 6 \times 10^{12}$ $[13]$ . In addition, the optomechanical coupling rate $g_{0}$ also plays a dominant role in the ability to efficiently laser cool the motion of a resonator. There are several approaches focusing on overcoming these limitations. In particular, experiments featuring optically levitated nanospheres have come to within a few thermal phonons of the mechanical ground state recently $[14-17]$ . While the absence of any physical attachment to the environment allows trapped nanospheres to exhibit extremely large quality factors, they require UHV systems and complex stabilization mechanisms for their optical traps, making them impractical as sensors and for other applications. Chip-based mechanical oscillators have recently also been shown to feature competitively large mechanical quality factors at room temperature with $Q_{m} \gtrsim 10^{8}$ , most prominently in high-stress silicon nitride membranes [18–20]. Here, similar limitations as with levitated nanospheres, such as mirror noise [21,22], exist, as well as the requirement to use bulky setups for optical readout.

In this Letter, we develop a new type of fully integrated optomechanical structure that allows us to significantly increase the mechanical quality factor of a high-frequency in-plane mode, while also allowing us to realize a coupled optomechanical cavity used for on-chip optical readout of the motion. We measure a $f_{m}Q_{m} \approx 2.6 \times 10^{13}$ , approaching the performance of the best out-of-plane mechanical resonators [12,23], combined with an optomechanical coupling of $G_{om}/2\pi = 21.6 \pm 0.2$ GHz/nm, enabling us to cool the mechanical mode from room temperature to 1.2 mK. This corresponds to a thermal mode occupation of less than 27 phonons, a reduction by more than 5 orders of magnitude in the effective temperature. Our novel design applies previous discoveries on the dominant role of bending losses [23,24] and results in a device that resembles a fishbonelike photonic and phononic structure [cf. Fig. 1(a)].

![](images/217746168d60d5525e37544f4457d06176ba1d26550f3afe16d58f87d98b304b.jpg)

<details>
<summary>text_image</summary>

(a)
0.5 mm
(b)
5µm
phanonic
</details>

![](images/11bc3493f53afd0045b5e4b0d0a0dabfa667c84a865bdcda2bc1a4254c265848.jpg)

<details>
<summary>text_image</summary>

(c)
coupling
waveguide
phononic structure
coupling
waveguide
photonic
structure
~5μm
silicon substrate
</details>

![](images/51dbeb70d8910d583e3c3c94f7ade9a40488eee9662fdfc11d2760d58eed806d.jpg)

<details>
<summary>line</summary>

| k_x/(π/a) | Frequency (MHz) |
| --------- | --------------- |
| 0.0       | 1.50            |
| 0.2       | 1.35            |
| 0.4       | 1.25            |
| 0.6       | 1.15            |
| 0.8       | 1.05            |
| 1.0       | 1.00            |
</details>

![](images/7bf2a4189599faee14c0d72357fc960d4f777a50cee9e97f1a955bfbc0c6062f.jpg)

<details>
<summary>line</summary>

| Position from center (μm) | Fishbone (Normalized bending ×10³μm⁻²) | Uniform (Normalized bending ×10³μm⁻²) |
| ------------------------- | -------------------------------------- | ------------------------------------- |
| 0                         | ~0.5                                   | ~0.5                                  |
| 5                         | ~0.5                                   | ~0.4                                  |
| 10                        | ~0.5                                   | ~0.3                                  |
| 15                        | ~0.5                                   | ~0.2                                  |
| 17                        | ~0.1                                   | ~1.0                                  |
| 20                        | ~0.1                                   | ~0.1                                  |
</details>

FIG. 1. (a) Shown is a stitched microscope image of the fabricated structure (top) and the corresponding mechanical simulation of the long beam (bottom). The enlargement shows a fishbone structure designed to reduce bending losses. (b) Mechanical (top) and photonic (bottom) simulation of the center part containing the photonic crystal. A short second beam forming the other half of the photonic crystal cavity is fixed close to the mechanical beam. The short structure does not feature any mechanical motion around the defect mode. The two structures form an optical cavity with the light strongly confined in the gap between the two beams. The mechanical motion changes the gap size, shifting the optical resonance frequency and hence giving rise to the optomechanical interaction. (c) Sketch of the central photonic crystal and coupling waveguide. Clearly visible is the short second beam in the center. A cross section is shown in the box on the bottom. (d) Band diagram for the in-plane mode of the phononic fishbone structure. We apply a periodic boundary condition in the x direction, with $k_{x}$ being the wave vector. The blocks in the long beam form a band gap between 0.6 and 1.1 MHz. (e) Bending ( $\partial^{2}v/\partial x^{2}$ ) normalized to the displacement v within the photonic fishbone structure. In the center, the magnitude of the bending alternates between the thin (large bending, white) and thick (small bending, gray) parts of the structure. As the mechanical losses are proportional to the cube of the width, the fishbone devices exhibit significantly higher mechanical $Q_{m}$ than a uniform beam of equal width (orange), typically used for photonic crystal zipper cavities [25].

Significant progress has been made over the last years in understanding and mitigating the losses in integrated (opto) mechanical structures, resulting in experimental demonstrations of ultrahigh- $Q_{m}$ devices. In particular, bending loss has been shown to be one of the dominant limiting mechanisms for mechanical quality factors in 1D high-stress silicon nitride structures [24]. Various approaches in strain [18,19] and mode shape engineering [23] have recently succeeded in achieving ultrahigh- $Q_{m}$ mechanical resonators. By using adiabatically chirped phononic crystals, for example [23], the mechanical mode is localized in the center of the beam and the bending can significantly be reduced, leading to increases in $Q_{m}$ . While this concept works very well for the out-of-plane motion, it is much more challenging for an in-plane mode [26]. This is due to the loss $\Delta U$ being proportional to the cube of the thickness in the motional direction [23], which for the in-plane mode is equivalent to the width w of the structure $\Delta U \propto w^{3}(\partial^{2}v/\partial x^{2})$ , with v being the displacement. In practice, there are several parts of an optomechanical structure that require a certain minimum width, such as the phononic crystal itself, which is partly composed of wide blocks of material. The bending of these very wide blocks results in large mechanical dissipations. Furthermore, in order to form a good optical cavity, the photonic crystal at the center of the structure also requires a minimum width, which is comparable to the optical wavelength [25]. Both factors largely reduce the attainable mechanical quality factor. With our new fishbone design, we minimize w in the parts with maximal bending, allowing us to significantly reduce $\Delta U$ and hence significantly increase the mechanical quality factor of the mechanical in-plane modes.

Our structure is fabricated from a 350 nm thick high-stress (1.3 GPa) silicon nitride layer deposited on a silicon handle wafer. As shown in Fig. 1(a), it is based on the differential motion of two strings, where one of them is significantly longer (2.6 mm) than the other (115 $\mu$ m). The longer string of this zipper structure is connected to the chip through a phononic crystal, with a band gap for the in-plane mode between 610 kHz and 1.10 MHz [see Fig. 1(d)]. This

design forms a defect in the center, introducing confined modes with frequencies within the band gap, significantly reducing the losses of these modes. As the amplitude of the modes of interest is largest in the center of the structure, we reduce its bending by introducing an adiabatic transition of the unit cells of the phononic crystals. This results in a weaker confinement and smaller bending close to the center [23]. As mentioned above, this does, however, not immediately result in a high-quality factor of the in-plane modes, as the width of the structure close to the bending areas becomes important. We therefore design the overall device as a string with a width of only $165\mathrm{nm}$ , limited by our fabrication process. When adding the phononic crystal, we avoid wide and rigid regions in the design, segmenting the blocks closest to the center into a fishbonelike structure.

A similar approach is also taken for the central photonic crystal used to readout the mechanical motion. Instead of a traditional photonic structure with holes in a waveguide $[25]$ , we achieve an alternating index contrast through a fishbone design. The wider parts are roughly $1 \mu m$ in width, while the narrow ones between the teeth have a width of only 165 nm. This geometry localizes the bending to the narrow parts [cf. Fig. 1(e)], significantly reducing the overall bending losses. For comparison, we observe a typical enhancement of $f_{m}Q_{m}$ by more than a factor of 5 between devices with and without the photonic fishbone structure.

The optical cavity is formed between the long and short strings and the optical field is confined within the gap formed by the fishbones and designed to operate at a wavelength of around $\lambda = 1550$ nm. Because of the strong confinement, the resonance frequency of the cavity is very sensitive to the gap size. In the simulation, shown in Fig. 1(b), we obtain an optomechanical coupling strength $G_{\mathrm{om}}/2\pi = (\partial\omega_{c}/\partial x) = 23.0$ GHz/nm, where $\omega_{c}$ is the cavity frequency for a typical 200 nm gap. Combining $G_{om}$ with the localized mechanical mode of interest, which has small effective mass $m_{eff} = 7.36 \times 10^{-14}$ kg, we obtain a single-photon coupling rate $g_{0}/2\pi = 252$ kHz. The resulting optimized structure features a 16.5 $\mu$ m long photonic crystal cavity, while the overall structure has a length of 2.6 mm.

We design the mechanical defect mode at a frequency $f_{m} = \omega_{m}/2\pi = 950$ kHz. A ringdown measurement of this mode in $5 \times 10^{-6}$ mbar vacuum shows a quality factor of $2.73 \times 10^{7}$ (see Supplemental Material for more details [27]), yielding $f_{m}Q_{m} = 2.59 \times 10^{13}$ . The total optical resonance's ( $\lambda = 1549.9$ nm) linewidth is measured to be $\kappa/2\pi = 33.0$ GHz, and the coupling rate to an adjacent optical waveguide is $\kappa_{e}/2\pi = 31.4$ GHz. The strongly overcoupled cavity ensures that most of the light in the cavity is reflected back into the waveguide, which is necessary to achieve high detection efficiency. To further characterize the device, we measure the optical spring effect (see Supplemental Material for details [27]), allowing us to experimentally determine a single-photon coupling rate of $g_{0}/2\pi = 237 \pm 2$ kHz, corresponding to an optomechanical coupling of $G_{om}/2\pi = 21.6 \pm 0.2$ GHz/nm, in good agreement with simulations. We determine the single-photon cooperativity of our device $C_{0} = (4g_{0}^{2}/\kappa\Gamma_{m}) \approx 200$ , which represents the relative strength of the single-photon interactions against any loss channels, a key characteristic of the system [25,31].

In this unresolved sideband regime $[32]$ , an active feedback cooling scheme can be used to reduce the thermal energy of the mechanical oscillator $[12,33]$ . In this scheme, unlike in the traditional cavity cooling approach $[8]$ , the extremely large bandwidth of the optical cavity allows us to retrieve information on the motion of the mechanical resonator at a high rate. In our experiment, we measure the position of the mechanics and process it in real time, using the resulting signal to actively control the optical input power into the optomechanical cavity. The modulation of the intensity changes the radiation-pressure force, hence allowing us to control and actively cool the mechanics itself. Figure 3(a) shows a sketch and description of our setup used to demonstrate such feedback cooling of our mechanical resonator. The measured signal containing the information on the position of the mechanical oscillator is sent to a field-programmable gate array (FPGA) controller (RedPitaya 125-14), with its output directly connected to an electro-optical intensity modulator just before the device. The FPGA control allows us to implement an almost arbitrarily complicated feedback filter. We apply a derivative filter with a second-order underdamped low-pass filter to cool the mechanical defect mode. The feedback phase at the resonance frequency is tuned to be $-\pi/2$ . Because of a small delay in the system, applying this signal directly would heat other nearby mechanical modes and make the system unstable. We therefore cascade a series of notch filters to tune the phase response locally, which provides a weak cooling over the surrounding modes (cf. Fig. 2). The total delay of the feedback loop is measured to be 0.49 $\mu$ s.

Figure 3(b) shows the calibrated displacement power spectrum ( $S_{yy}$ ) of the mechanical oscillator with different levels of cooling from a bath at room temperature. We keep the cavity photon number fixed at $n_{c} = 120$ , while increasing the gain of the feedback filter to increase the amount of feedback. The mechanical peak amplitude reduces and broadens, corresponding to a cooling of the mode of interest. The curves are then fitted and we extract the displacement spectrum $S_{xx}$ [12,25]. This allows us to calculate the average phonon number $\bar{n}$ , which is shown as a function of electronic gain in Fig. 3(c). The lowest occupation we obtain is $\bar{n} = 26.6 \pm 0.7$ , reduced from $6.5 \times 10^{6}$ at room temperature. We note that our measurements are not quantum noise limited in this experiment, resulting in a slightly increased phonon occupancy compared to the theoretically expected value. This additional noise floor results from the optical fiber touching the waveguide and introducing broadband mechanical modes.

![](images/aed100990c71de45eb3d4ab38c3854f983658b6455ba6d15df81c507d5a21c99.jpg)

<details>
<summary>line</summary>

| Frequency (MHz) | Power (a.u.) | Phase / π | Gain |
| --------------- | ------------ | --------- | ---- |
| 0.00            | ~10⁻⁵        | -0.5      | ~10⁻² |
| 0.25            | ~10⁻⁴        | 0.0       | ~10⁻¹ |
| 0.50            | ~10⁻⁵        | 0.5       | ~10⁻¹ |
| 0.75            | ~10⁻⁴        | 1.0       | ~10⁻¹ |
| 1.00            | ~10⁻³        | 1.0       | ~10⁰  |
| 1.25            | ~10⁻⁴        | 0.5       | ~10⁻¹ |
| 1.50            | ~10⁻⁵        | 1.0       | ~10⁻¹ |
| 1.75            | ~10⁻⁴        | 1.0       | ~10⁻¹ |
| 2.00            | ~10⁻⁵        | 0.5       | ~10⁻¹ |
</details>

FIG. 2. (a) Mechanical spectrum. The red dashed line marks the high- $Q_{m}$ defect mode. Gray lines indicate additional mechanical modes that strongly couple to the optical cavity, while most other spurious peaks arise from the mixing of these modes in the detection itself. (b) Phase response of the feedback control. The circuit has a phase of $-0.5\pi$ at the resonance of the defect mode, with the gray area indicating the unstable region due to heating. (c) Gain of the feedback control. We implement several filter functions with large bandwidth, allowing us to suppress and partly cool other modes that are excited in order to stabilize the system.

These mechanical waveguide modes shift the resonance frequency of the cavity weakly, resulting in an increase of the detection noise. At high gain, this noise is fed into the mechanics and limits the cooling efficiency. Unlike in the ideal case of a quantum-noise-limited measurement, increasing the input optical power does not reduce the classical noise and hence it does not lead to more efficient cooling. Using different types of coupling methods or redesigning the waveguide will allow us to reduce the classical noise further, allowing us to, in principle, cool to an occupation of $\bar{n}_{min} \approx 14$ , with everything else left unchanged.

In order to get even closer to the ground state in the continuous feedback cooling scheme, the measurement rate $\left(\Gamma_{\mathrm{meas}}=x_{\mathrm{zpf}}^{2}/S_{xx}^{\mathrm{imp}}=4\eta n_{c}g_{0}^{2}/\kappa\right.$ , where $x_{zpf}$ is the zero-point fluctuation of the oscillator) has to be comparable or larger than the decoherence rates in the system, i.e., the thermal decoherence $\left(\Gamma_{\mathrm{th}}\approx\Gamma_{m}n_{\mathrm{th}}\right)$ and the backaction rate $\left(\Gamma_{\mathrm{ba}}=\Gamma_{m}n_{\mathrm{ba}}\right.$ , where $n_{ba}=n_{c}C_{0}$ [31]. Here, $\eta$ is the overall detection efficiency, which for our experiment is $\eta=0.50$ , while $n_{ba}=2.4\times10^{4}\ll n_{th}$ , hence making the thermal component the dominant decoherence channel. Excluding classical noise, we find $\Gamma_{\mathrm{meas}}/(\Gamma_{\mathrm{th}}/8)=0.015\ll1$ [31], which is orders of magnitude larger than in previous similar experiments [25].

Several approaches to increasing this ratio exist. For example, by redesigning the coupling waveguide to obtain a quantum-noise-limited measurement, the intracavity photon number can be raised further and is eventually only bound by absorption heating. Increasing the optomechanical coupling rate can be achieved by improving the fabrication and reducing the gap size between the strings forming the optical cavity. A reduction of the gap to 100 nm yields $G_{om} = 2\pi \times 45$ GHz/nm, which is more than twice the current value. Another way is to further reduce the thermal decoherence rate, through device improvements. Our current design is not optimized to maximize the stress [18], which would lead to more stored energy, increasing $Q_{m}$ . At the moment, the maximum simulated stress in the structure is 1.5 GPa, which is still far below the yield strength of SiN ( $\sim6$ GPa). Higher stress can also be achieved through an overall longer

![](images/4e2e1ba44801a3283cc600ec095802641e0539ed1319389a52240fd4a04b9947.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["5um"] --> B["VOA"]
    B --> C["EOM"]
    C --> D["Controller"]
    D --> E["vacuum chamber"]
    F["VOA"] --> G["φ"]
    G --> H["φ"]
    H --> I["VOA"]
    I --> J["VOA"]
    J --> K["φ"]
    K --> L["VOA"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#cff,stroke:#333
    style F fill:#ffc,stroke:#333
    style G fill:#cfc,stroke:#333
    style H fill:#cfc,stroke:#333
    style I fill:#cfc,stroke:#333
    style J fill:#cfc,stroke:#333
    style K fill:#cfc,stroke:#333
    style L fill:#cfc,stroke:#333
```
</details>

FIG. 3. (a) Sketch of the feedback cooling setup. A laser is first tuned on cavity resonance and phase modulated to generate a calibration tone. It is then split into two arms, with both intensities being controlled through variable optical attenuators (VOAs). The bottom path is the local oscillator for the homodyne detection scheme, where the phase can be controlled using a fiber stretcher ( $\phi$ ). The light in the upper (signal) arm is intensity modulated in an electro-optical modulator (EOM) and sent to the waveguide [34], where it is then evanescently coupled into the optical cavity. At the end of the waveguide, we pattern a photonic crystal mirror, which allows the light from the cavity to be reflected back into the fiber with a collection efficiency of $91\%$ . This light is then mixed with the local oscillator on a beam splitter and measured in a home-built low-noise balanced photodetector with a quantum efficiency of $70\%$ in order to perform the phase-sensitive measurement. The detected signal is electronically processed in a FPGA-based controller, which directly modulates the light in real time through the EOM, and hence allows us to cool the mechanical resonator. (b) Cooled mechanical spectra $S_{yy}$ , with increased feedback gain from orange to blue and constant intracavity photon number $n_c$ . The dashed lines are fits to the spectra, while the gray dotted line indicates the quantum-limited noise floor. (c) Average phonon number extracted from the spectra in (b), with corresponding color coding. The gray dashed line represents the theoretically predicted quantum-noise-limited phonon number, and the dark blue dotted line is the expected phonon number when taking the noise into account.

beam, while at the same time allowing for more adiabatic chirping in the geometry, which would further reduce mechanical losses. Combining these approaches should allow us to reach the quantum ground state. In particular, moderately increasing the mechanical quality factor to $Q_{m} \approx 1 \times 10^{8}$ , would lead to an increase of $\Gamma_{\mathrm{meas}} / (\Gamma_{\mathrm{th}} / 8)$ to 0.06 and the phonon number could be reduced to six. Together with a reasonable reduction of the gap to 100 nm and a small increase of the cavity photon number to 200, a phonon occupation around three will be achievable. Further improvements in $Q_{m}$ to $\gtrsim 7 \times 10^{8}$ [20,23] will finally enable phonon numbers below unity starting from room temperature.

In summary, we have designed and fabricated a novel, fully integrated optomechanical system, featuring a fish-bonelike photonic and phononic structure, with a $Q_{m} = 2.73 \times 10^{7}$ of an in-plane mechanical mode combined with a large optomechanical coupling rate of $g_{0} = 237$ kHz. We use this device to demonstrate active-feedback cooling close to the quantum ground state of motion, starting from room temperature. By tuning the FPGA-based feedback filter, we stabilize spurious modes that strongly couple to the optics, allowing us to reach an effective mode temperature of 1.2 mK, corresponding to less than 27 phonons. Further improvements in the noise performance of our setup, together with enhancements of $Q_{m}$ and optomechanical coupling, should allow for these structures to be cooled fully into their ground state, which will enable mechanical quantum experiments at ambient temperatures. In addition, the simplicity in fabrication of our devices, consisting of a single-SiN layer on chip only, combined with their fully integrated on-chip character, makes them ideal candidates for quantum sensing applications [7,25].

We would like to thank Bas Hensen, Alex Krause, Igor Marinković, Rob Stockill, and Andreas Wallucks for valuable discussions and also acknowledge assistance from the Kavli Nanolab Delft. This work is further supported by the Foundation for Fundamental Research on Matter (FOM) Projectruimte Grants No. 15PR3210 and No. 16PR1054, the European Research Council (ERC StG Strong-Q, 676842), the EMPIR Programme cofinanced by the Participating States and from the European Union's Horizon 2020 Research and Innovation Programme, and by the Netherlands Organization for Scientific Research (NWO/OCW), as part of the Frontiers of Nanoscience program, as well as through a Vidi Grant (No. 680-47-541/994). J.G. gratefully acknowledges support through a Casimir Ph.D. fellowship.

\* r.a.norte@tudelft.nl   
$^{\dagger}$ s.groeblacher@tudelft.nl   
[1] A. D. O'Connell, M. Hofheinz, M. Ansmann, R. C. Bialczak, M. Lenander, E. Lucero, M. Neeley, D. Sank,

H. Wang, M. Weides, J. Wenner, J. M. Martinis, and A. N. Cleland, Nature (London) 464, 697 (2010).   
[2] T. Palomaki, J. Teufel, R. Simmonds, and K. Lehnert, Science 342, 710 (2013).   
[3] S. Hong, R. Riedinger, I. Marinković, A. Wallucks, S. G. Hofer, R. A. Norte, M. Aspelmeyer, and S. Gröblacher, Science 358, 203 (2017).   
[4] R. Riedinger, A. Wallucks, I. Marinković, C. Löschnauer, M. Aspelmeyer, S. Hong, and S. Gröblacher, Nature (London) 556, 473 (2018).   
[5] C. F. Ockeloen-Korppi, E. Damskägg, J.-M. Pirkkalainen, M. Asjad, A. A. Clerk, F. Massel, M. J. Woolley, and M. A. Sillanpää, Nature (London) 556, 478 (2018).   
[6] I. Marinković, A. Wallucks, R. Riedinger, S. Hong, M. Aspelmeyer, and S. Gröblacher, Phys. Rev. Lett. 121, 220404 (2018).   
[7] R. A. Norte, M. Forsch, A. Wallucks, I. Marinković, and S. Gröblacher, Phys. Rev. Lett. 121, 030405 (2018).   
[8] J. Chan, T. P. M. Alegre, A. H. Safavi-Naeini, J. T. Hill, A. Krause, S. Gröblacher, M. Aspelmeyer, and O. Painter, Nature (London) 478, 89 (2011).   
[9] J. D. Teufel, T. Donner, D. Li, J. W. Harlow, M. S. Allman, K. Cicak, A. J. Sirois, J. D. Whittaker, K. W. Lehnert, and R. W. Simmonds, Nature (London) 475, 359 (2011).   
[10] M. Underwood, D. Mason, D. Lee, H. Xu, L. Jiang, A. B. Shkarin, K. Børkje, S. M. Girvin, and J. G. E. Harris, Phys. Rev. A 92, 061801(R) (2015).   
[11] R. W. Peterson, T. P. Purdy, N. S. Kampel, R. W. Andrews, P.-L. Yu, K. W. Lehnert, and C. A. Regal, Phys. Rev. Lett. 116, 063601 (2016).   
[12] M. Rossi, D. Mason, J. Chen, Y. Tsaturyan, and A. Schliesser, Nature (London) 563, 53 (2018).   
[13] F. Marquardt, J. P. Chen, A. A. Clerk, and S. M. Girvin, Phys. Rev. Lett. 99, 093902 (2007).   
[14] U. Delić, M. Reisenbauer, D. Grass, N. Kiesel, V. Vuletić, and M. Aspelmeyer, Phys. Rev. Lett. 122, 123602 (2019).   
[15] F. Tebbenjohanns, M. Frimmer, A. Militaru, V. Jain, and L. Novotny, Phys. Rev. Lett. 122, 223601 (2019).   
[16] G. P. Conangla, F. Ricci, M. T. Cuairan, A. W. Schell, N. Meyer, and R. Quidant, Phys. Rev. Lett. 122, 223602 (2019).   
[17] F. Tebbenjohanns, M. Frimmer, V. Jain, D. Windey, and L. Novotny, arXiv:1908.05079.   
[18] R. A. Norte, J. P. Moura, and S. Gröblacher, Phys. Rev. Lett. 116, 147202 (2016).   
[19] C. Reinhardt, T. Müller, A. Bourassa, and J. C. Sankey, Phys. Rev. X 6, 021001 (2016).   
[20] Y. Tsaturyan, A. Barg, E. S. Polzik, and A. Schliesser, Nat. Nanotechnol. 12, 776 (2017).   
[21] Y. T. Liu and K. S. Thorne, Phys. Rev. D 62, 122002 (2000).   
[22] G. M. Harry, H. Armandula, E. Black, D. R. M. Crooks, G. Cagnoli, J. Hough, P. Murray, S. Reid, S. Rowan, P. Sneddon, M. M. Fejer, R. Route, and S. D. Penn, Appl. Opt. 45, 1569 (2006).   
[23] A. H. Ghadimi, S. A. Fedorov, N. J. Engelsen, M. J. Bereyhi, R. Schilling, D. J. Wilson, and T. J. Kippenberg, Science 360, 764 (2018).   
[24] Q. P. Unterreithmeier, T. Faust, and J. P. Kotthaus, Phys. Rev. Lett. 105, 027205 (2010).   
[25] A. G. Krause, T. D. Blasius, and O. Painter, arXiv:1506.01249.

[26] A. H. Ghadimi, Ph.D. thesis, École polytechnique fédérale de Lausanne, 2018.   
[27] See Supplemental Material at http://link.aps.org/supplemental/10.1103/PhysRevLett.123.223602 for additional details on the design and fabrication processes, which includes Refs. [28–30].   
[28] J. Chan, Ph.D. thesis, California Institute of Technology, 2012.   
[29] S. G. Johnson, M. Ibanescu, M. A. Skorobogatiy, O. Weisberg, J. D. Joannopoulos, and Y. Fink, Phys. Rev. E 65, 066611 (2002).

[30] A. H. Ghadimi, D. J. Wilson, and T. J. Kippenberg, Nano Lett. 17, 3501 (2017).   
[31] D. J. Wilson, V. Sudhir, N. Piro, R. Schilling, A. Ghadimi, and T. J. Kippenberg, Nature (London) 524, 325 (2015).   
[32] M. Aspelmeyer, T. J. Kippenberg, and F. Marquardt, Rev. Mod. Phys. 86, 1391 (2014).   
[33] D. Kleckner and D. Bouwmeester, Nature (London) 444, 75 (2006).   
[34] S. Gröblacher, J. T. Hill, A. H. Safavi-Naeini, J. Chan, and O. Painter, Appl. Phys. Lett. 103, 181104 (2013).