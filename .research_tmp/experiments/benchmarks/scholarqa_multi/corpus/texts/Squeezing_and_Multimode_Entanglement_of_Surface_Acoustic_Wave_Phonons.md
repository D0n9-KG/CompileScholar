# Squeezing and Multimode Entanglement of Surface Acoustic Wave Phonons

Gustav Andersson $^{1,*,†,‡}$ Shan W. Jolin $^{2,†,§}$ Marco Scigliuzzo $^{1}$ , Riccardo Borgani $^{2}$ , Mats O. Tholén $^{2,3}$ J.C. Rivera Hernández $^{2}$ , Vitaly Shumeiko $^{1}$ , David B. Haviland $^{2}$ , and Per Delsing $^{1}$ $^{1}$ Department of Microtechnology and Nanoscience MC2, Chalmers University of Technology, Göteborg SE-41296, Sweden $^{2}$ Nanostructure Physics, KTH Royal Institute of Technology, Stockholm SE-10691, Sweden $^{3}$ Intermodulation Products AB, Segersta SE-82393, Sweden

(Received 17 July 2020; revised 26 August 2021; accepted 8 December 2021; published 20 January 2022)

Exploiting multiple modes in a quantum acoustic device could enable applications in quantum information in a hardware-efficient setup, including quantum simulation in a synthetic dimension and continuous-variable quantum computing with cluster states. We develop a multimode surface acoustic wave (SAW) resonator with a superconducting quantum interference device (SQUID) integrated in one of the Bragg reflectors. The interaction with the SQUID-shunted mirror gives rise to coupling between the more than 20 accessible resonator modes. We exploit this coupling to demonstrate two-mode squeezing of SAW phonons, as well as four-mode multipartite entanglement. Our results open avenues for continuous-variable quantum computing in a compact hybrid quantum system.

DOI: 10.1103/PRXQuantum.3.010312

# I. INTRODUCTION

Quantum computation and simulation show potential for tackling difficult computational problems by leveraging superposition and entanglement in engineered quantum devices. Although individual quantum systems can be controlled with excellent precision, scaling the hardware to the complexity required while maintaining sufficient control remains a challenging problem [1]. Most architectures proposed for quantum simulation and computation [2-6] use one circuit component for each node in the processor, leading to demanding hardware requirements for practical applications. It is therefore attractive to explore alternative approaches to quantum computing that provide for compact encoding and processing of quantum information.

In principle, the use of continuous variables (CVs) allows for realizations of measurement-based quantum computing with frequency combs, requiring only a small number of coupled quantum systems [7,8]. This paradigm

of quantum computing relies on entangling a large number of modes rather than qubits, and does not face fundamental restrictions preventing universality and fault tolerance [9]. Much experimental progress in CV encoding of quantum information has been achieved in the domain of quantum optics [10-14], where optical parametric oscillators can be used to generate large cluster states, a type of multipartite entangled state providing the resource for CV quantum computation.

With superconducting circuits, CV encoding has been pursued mainly for error-correction schemes on logical qubits encoded in many-photon superconducting cavity states [15,16]. While parametric devices are important for low-noise amplification [17], multimode measurement-based schemes for quantum computing at microwave frequencies have received relatively little attention [18-20]. The dominant approach to quantum computation with superconducting quantum circuits has been the gate-based quantum processor, with most effort expended on scaling up the number of physical qubits [21].

A limiting factor for realizing CV encoding in circuit quantum electrodynamic (QED) systems is the typically large electromagnetic mode spacing, making devices with a large number of accessible modes very long or difficult to design. On the other hand, microwave frequencies are amenable to digital signal processing and thereby a greater degree of programmable control than is currently possible in optical systems. The prospect of integrating superconducting qubits as a means of providing non-Gaussian operations necessary for quantum advantage in computation [22] is an additional strength of microwave circuits.

PRX QUANTUM 3, 010312 (2022)

\* gandersson@uchicago.edu

These authors contributed equally.

$^{\ddagger}$ Current address: Pritzker School of Molecular Engineering, University of Chicago, Chicago IL 60637, USA.

$^{\S}$ Current address: IQM Finland Oy, FI-021 50 Espoo, Finland.

Published by the American Physical Society under the terms of the Creative Commons Attribution 4.0 International license. Further distribution of this work must maintain attribution to the author(s) and the published article's title, journal citation, and DOI.

2691-3399/22/3(1)/010312(16)

010312-1

Published by the American Physical Society

We demonstrate an approach towards realizing CV quantum computation based on cluster state generation and control in a multimode hybrid superconducting quantum acoustic device. The interaction between mechanical oscillators and superconducting circuits has been used to show quantum effects [23], including entanglement [24]. Surface acoustic wave (SAW) resonators support dense mode spectra with high- $Q$ factors ( $>10^5$ ) and have been used in multimode experiments in the quantum regime [25-27]. Substantial progress has also been made in recent years in the controlled generation of nonclassical phononic states [28-30], and applications as quantum random-access memories have been proposed [31].

Here, we develop a multimode quantum acoustic device by integrating a superconducting quantum interference device (SQUID) into one of the Bragg reflectors of a SAW resonator. The SQUID inductance modulates the reflectivity of a unit cell in the mirror and hence the effective length of the resonator. Due to the narrow free spectral range, the SQUID reflector gives rise to coupling of more than 20 modes. We exploit this coupling to generate

two-mode squeezed states between phonons in different SAW modes with a parametric drive. Extending the pump scheme to four tones, we demonstrate multipartite entanglement between four acoustic modes. Our results suggest this quantum acoustic platform can be used to create highly entangled multimode states for CV quantum computing.

# II. DEVICE DESIGN AND SETUP

The SAW resonator, shown in Fig. 1, is defined by two reflectors with the leading edges separated by a distance of $600\mu \mathrm{m}$ . The reflector on the left-hand side has 1200 fingers all shorted together. On the right-hand side, the reflector has an interdigitated structure, where fingers are connected to either the top or bottom electrode in an alternating pattern. The top and bottom electrodes each have $N_{p} = 500$ fingers with an overlap of $W = 100\mu \mathrm{m}$ and are connected via a SQUID. An interdigitated transducer (IDT) centered between the reflectors provides a single port to the resonator. The port IDT has 75 periods and a double-finger structure to suppress mechanical

![](dt=2026-06-06/ht=05/450cdc1d53a9a6d19feb49342ad59939a219cfc5a5eae672a1631596dc906933.jpg)

GUSTAV ANDERSSON et al.

PRX QUANTUM 3,010312 (2022)

010312-2

reflections [32]. An on-chip fluxline is used to apply a rf flux through the SQUID. The IDT and reflectors are fabricated from aluminium on a gallium arsenide substrate. Due to the piezoelectric coupling, the SAW field inside the resonator induces an electric potential difference between the top and bottom electrodes, generating currents through the SQUID.

Configurations where an interdigitated reflector is shunted by a variable load impedance have been used for SAW-based sensors [33]. Here, the SQUID impedance provides a means of flux tuning the SAW resonator, as well as a cross-Kerr interaction between the modes. Integrating the SQUID makes the device a kind of acoustic analog to the superconducting cavity-based Josephson parametric amplifier [17], where the short waveleng
th of sound allows for a much denser mode spacing than in the purely electromagnetic case. The SQUID reflector is equivalent to a dispersively coupled nonlinear resonator, as the interdigitated fingers give rise to a large capacitance connected in parallel with the SQUID inductance. We use this model to explain the effect of parametric modulation in this system.

The mode structure of the resonator is shown in Fig. 2. While the IDT is centered with respect to the leading edge of each reflector, the broken symmetry due to the SQUID allows coupling to both odd and even modes with a free spectral range (FSR) $= 2.3\mathrm{MHz}$ . The alternating pattern of even and odd modes is apparent in the external and internal quality factors $Q_{c}, Q_{i}$ extracted from fits to reflection

![](dt=2026-06-06/ht=05/a36b5da9174e056c30e60484c204b771c308c9eb94384b37231ab21742ffb4b9.jpg)

measurements. The IDT couples more effectively to the even modes, resulting in a lower $Q_{c}$ . The frequency dependence of the IDT and mirrors provide a bandwidth of around 40 MHz around the IDT center frequency where SAW modes are overcoupled.

The narrow free spectral range of the resonator allows for simultaneous measurement of the response at multiple resonances all multiplexed in a single channel. For this measurement we use a digital microwave measurement platform [34] to directly digitally synthesize and measure signals at multiple frequencies simultaneously without analog mixers for frequency conversion.

# III. COUPLED PARAMETRIC RESONATOR INTERACTION

The electromagnetic mode of the mirror has a frequency $\omega_{LC}$ , which is parametrically modulated in time. The coupling to the SAW modes gives rise to the effective Hamiltonian (Appendix D)

$$
H _ {\text {e f f}} = \sum_ {j} \hbar \tilde {\omega} _ {j} b _ {j} ^ {\dagger} b _ {j} + s (t) \sum_ {j, k} \tilde {g} _ {j} \tilde {g} _ {k} \left(b _ {j} - b _ {j} ^ {\dagger}\right) \left(b _ {k} - b _ {k} ^ {\dagger}\right). \tag {1}
$$

Here $b_{j}$ and $b_{j}^{\dagger}$ are the ladder operators for the SAW modes, while $s(t)$ is a time-dependent factor determined by the flux pump amplitude and frequency. Assuming a uniform vacuum coupling strength $g$ , the effective coupling rates $\tilde{g}_{j}$ depend on the mirror and SAW mode resonance frequencies as

$$
\tilde {g} _ {j} = \frac {2 g \omega_ {j}}{\omega_ {j} ^ {2} - \omega_ {L C} ^ {2}}. \tag {2}
$$

The bare SAW frequencies $\omega_{j}$ are renormalized due to the interaction to

$$
\tilde {\omega} _ {j} = \omega_ {j} - g \tilde {g} _ {j} \frac {\omega_ {L C}}{\omega_ {j}}. \tag {3}
$$

The second term in Eq. (1) contains both beam-splitter and two-mode squeezing interactions. Depending on the modulation frequency of $s(t)$ , either interaction can be selected. A beam-splitting interaction may be implemented by a parametric drive close to the difference frequency of the SAW modes. Modulating near the sum frequency will induce two-mode squeezing between pairs of modes. In our experiment we modulate the magnetic flux through the SQUID without dc flux bias at the frequency of a SAW mode.

The parabolic dependence of the frequency on flux implies the electromagnetic mirror mode, and hence the effective coupling rate, are modulated at twice the pump frequency $s(t)\tilde{g}_j\tilde{g}_k\sim (\cos 2\omega_p t + 1)\tilde{g}_j\tilde{g}_k$ . This gives rise to two-mode squeezing as two photons from the pump tone are converted to one phonon each in modes symmetric around the pump.

SQUEEZING AND MULTIMODE ENTANGLEMENT...

PRX QUANTUM 3,010312 (2022)

010312-3

# IV. TWO-MODE SQUEEZING

To observe two-mode squeezing of the SAW field, we apply a parametric pump via the on-chip fluxline at the frequency of a SAW mode $f_{p}$ . We measure the output field from the IDT at SAW mode frequencies symmetrically around the pump $f_{i,\pm}$ , such that $2f_{p} - f_{i, - } - f_{i, + } = 0$ . The

![](dt=2026-06-06/ht=05/520cb5cb477a77c323c56e269429a9a5dc921b39eb9535a3ad9883068878245e.jpg)

![](dt=2026-06-06/ht=05/020fa1db0377da61cedbe2fc01d7f3438947b0820f148815e469f23e6ba317f9.jpg)

![](dt=2026-06-06/ht=05/87e4f66dbf20475fdb10f79caa12c4c6b2a2c138597cdbfcad9c7699637a07dc.jpg)

frequency configuration of the measurement is illustrated in Fig. 3(a). To characterize the correlations, we obtain reference histograms of the $I$ and $Q$ quadratures with the pump turned off. To minimize the effect of slow drift in the experimental setup, the pump output is switched on and off at a rate of $2\mathrm{Hz}$ . The output signal is amplified using a traveling-wave parametric amplifier [35]. Data are collected over approximately $7\mathrm{h}$ (3.5 h each with the pump on and off). A quadrature rotation is applied to the measured data to compensate for slow phase drift in the experiment.

In Fig. 3(b) we show subtracted quadrature histograms measured simultaneously in four pairs of SAW modes. Histograms are generated from $N = 1.25 \times 10^{6}$ points measured in each mode with the pump on (off). The unsqueezed histograms obtained with the pump off are then subtracted from those produced with the pump on. We observe squeezing below the pump-off level in all four mode pairs, extending the two-mode squeezing effect to SAW phonon fields. The ellipticity, defined as the ratio of the squeezed and antisqueezed axes

$$
R _ {e} = \frac {\sigma_ {\operatorname* {m a x}}}{\sigma_ {\operatorname* {m i n}}} = \frac {\sqrt {\left\langle \left(I _ {+} + I _ {-}\right) ^ {2} \right\rangle}}{\sqrt {\left\langle \left(I _ {+} - I _ {-}\right) ^ {2} \right\rangle}} \tag {4}
$$

is well above unity for all four mode pairs and shows a diminishing trend with increased detuning. The ratio of the standard deviation in the squeezed quadrature to the pump-off case, given by

$$
R _ {p} = \frac {\sigma_ {\operatorname* {m i n}}}{\sigma_ {\text {o f f}}} = \frac {\sqrt {\left\langle \left(I _ {+} - I _ {-}\right) ^ {2} \right\rangle}}{\sqrt {\left\langle \left(I _ {\text {o f f}}\right) ^ {2} \right\rangle}}, \tag {5}
$$

has a value $R_{p} < 1$ across the four mode pairs. The correlation ratios are plotted as a function of pump-probe detuning in Fig. 3(c). As expected, no correlations are observed outside the two-mode squeezed pairs. From our analysis of mode correlations (see Appendix F 1) we estimate that the squeezing is below the vacuum level, if the modes are cooled below an effective temperature of $80~\mathrm{mK}$ . Even if our device is not perfectly thermalized to the $10\mathrm{-mK}$ cryostat temperature, it is unlikely the effective phonon temperature should exceed this bound, leading to a strong indication that we observe squeezing below the vacuum in the phonon field.

The ability to generate two-mode squeezing with a single pump tone is an important step towards multimode entanglement, as this can be achieved using multimode squeezing [36]. In the next section we present such an experiment with a multitone pump and calibrated measurement chain.

GUSTAV ANDERSSON et al.

PRX QUANTUM 3,010312 (2022)

010312-4

# V. MULTIMODE ENTANGLEMENT

Following the two-mode squeezing measurement, we develop the experiment further to observe multipartite entanglement involving more SAW modes. For this purpose we use another device with reduced cross talk between the pump line and IDT described in more detail in Appendix B. With a multitone modulation, the Hamiltonian of Eq. (1) can provide coupling between any pair of modes in the resonator. Instead of single pump tone, we now apply a regularly spaced comb of pump frequencies containing up to four tones.

Due to the slight deviation from equidistance in th
e SAW modes the resulting entanglement is restricted to a set of modes in the vicinity of the pumps. The sharp mode structure also implies the amplitude and phase of correlations are sensitive to the pump comb settings. This is apparent in the scattering response shown in Appendix H.

For the two-mode squeezing measurement data we subtract the noise measured with the pump off. In order to establish multipartite entanglement we instead perform a calibration of the gain (Appendix F 2) and added noise of the measurement amplification chain. The calibration is based on Planck spectroscopy [37] and provides an estimate of the power level corresponding to vacuum fluctuations in the SAW modes, allowing us to test the measurement data for continuous-variable entanglement.

The Gaussian state of the probe modes is characterized by the quadrature covariance matrix. Drift and noise in the measurement setup can diminish mode correlations and render averaged covariance matrices unphysical. To mitigate this problem, we divide the 2.5-min measurement into 2-s intervals and perform a reconstruction [38] to ensure a physical state and test for entanglement on each interval separately.

The pump and measurement configuration for four modes using four pump tones is shown in Fig. 4(a). Figure 4(b) shows the quadrature covariance matrix obtained in this measurement. Using the calibration reference, the measured amplitudes are scaled with the single photon energy and measurement bandwidth $\Delta_{\mathrm{BW}}$ as

$$
V _ {i j} = \frac {\left\langle A _ {i} A _ {j} \right\rangle}{\frac {1}{2} Z _ {0} \hbar \sqrt {\omega_ {i} \omega_ {j}} \Delta_ {\mathrm {B W}}}, \tag {6}
$$

where $A \in \{I, Q\}$ . With this scaling the vacuum state is given by the identity matrix. The measured mode set extends beyond the four modes analyzed here, but we are not able to recover physical covariance matrices for all modes from our calibration of the amplifier gain and added noise.

The histograms of the output quadratures for the $f_{k}$ modes are all measured in parallel. In this case the correlations are not restricted to pairwise two-mode squeezing, but all modes are mutually correlated. As shown in

![](dt=2026-06-06/ht=05/458a275ab223d71afc76dfb03b52c869a3a069b63d9d1b262fff536ae306dbdc.jpg)

![](dt=2026-06-06/ht=05/e58a685993d4a07a4d0c9a6efd330ed89dd5ee9d5e3b82ca2171b1f0d45815ab.jpg)

![](dt=2026-06-06/ht=05/2f057982d9b035c8a5417f62126df32b980f405b2747dcce3d8b99f791a4ccf7.jpg)

Fig. 4(c), the correlation features are qualitatively captured by our theoretical model presented in Appendix E.

The multimode correlated state can be analyzed for entanglement. We evaluate the entanglement using a variant of negativity of partial transpositions [39] developed in Ref. [40]. This test relies on violating the inequality

$$
\begin{array}{l} \mathcal {E} = \operatorname {T r} \left[ V ^ {I I} (\mathbf {h} \otimes \mathbf {h}) \right] + \operatorname {T r} \left[ V ^ {Q Q} (\mathbf {g} \otimes \mathbf {g}) \right] \\ - 2 | \langle h _ {\mathcal {I}}, g _ {\mathcal {I}} \rangle | - 2 | \langle h _ {\mathcal {J}}, g _ {\mathcal {J}} \rangle | \geq 0, \tag {7} \\ \end{array}
$$

which holds for separable states.

SQUEEZING AND MULTIMODE ENTANGLEMENT...

PRX QUANTUM 3,010312 (2022)

010312-5

In computing the quantity $\mathcal{E}$ , the covariance matrix is rotated to eliminate correlations between $I$ and $Q$ quadratures and $V^{II}(V^{QQ})$ denotes the submatrix containing the $I - I(Q - Q)$ correlations. The vectors $\mathbf{h}$ and $\mathbf{g}$ are real valued with lengths equal to the number of modes. The subscripted terms indicate elements (and their corresponding modes) of $\mathbf{h}$ and $\mathbf{g}$ belonging to the bipartition subsets $\mathcal{I}$ and $\mathcal{J}$ .

One should consider $\mathbf{h}$ and $\mathbf{g}$ as the coefficients of a general test operator acting as our entanglement witness [41]. We are free to optimize $\mathbf{h}$ and $\mathbf{g}$ to maximize any violation of the inequality Eq. (7) for a given bipartition $\mathcal{I}$ and $\mathcal{J}$ . The entanglement measure is then obtained as a weighted mean over all 75 intervals within the full integration time. We estimate the standard deviation in the measurement and express entanglement in terms of the significance $\Sigma_w$ , given by

$$
\Sigma_ {w} = \frac {\mathcal {E}}{\sigma}. \tag {8}
$$

The uncertainty $\sigma$ is calculated as

$$
\sigma = \sqrt {\sum_ {i j} \left(\sigma_ {i j} ^ {2} h _ {i} ^ {2} h _ {j} ^ {2} + \sigma_ {i j} ^ {2} g _ {i} ^ {2} g _ {j} ^ {2}\right)}, \tag {9}
$$

where the matrix elements $\sigma_{ij}$ are obtained by error propagation accounting for the uncertainty in the calibration gain and noise parameters as well as noise in the measurement (Appendix F 2).

The entanglement significance is computed for all bipartitions of the four-mode set. As shown in Table I, all bipartitions yield entanglement by at least 2.4 standard deviations. Negativity of the entanglement test for all bipartitions is a signature of full multipartite entanglement [7]. We observe that the entanglement significance is substantially higher for bipartitions where modes 3 and 4 appear in separate sets. This is due to the imperfect alignment of the equidistant pump comb with the mode structure. In the measured covariance matrix in Fig. 4(b), this leads to stronger correlations involving modes 3 and 4. More uniform correlations and enhanced entanglement

TABLE I. Significance of detected entanglement for all bipartitions using the estimate of Eqs. (7)-(8). The highest significance is obtained for partitions separating modes 3 and 4.

![](dt=2026-06-06/ht=05/ad3ca6cec0e05e5a697b5f7ee7672b9343e1ddc8ede2febbd6e2b3c4e58fd485.jpg)

<table><tr><td>Bipartition</td><td>Σw</td></tr><tr><td>{1} : {2,3,4}</td><td>-13.7</td></tr><tr><td>{2} : {1,3,4}</td><td>-2.4</td></tr><tr><td>{1,2} : {3,4}</td><td>-2.9</td></tr><tr><td>{3} : {1,2,4}</td><td>-70.6</td></tr><tr><td>{1,3} : {2,4}</td><td>-74.1</td></tr><tr><td>{2,3} : {1,4}</td><td>-69.3</td></tr><tr><td>{1,2,3} : {4}</td><td>-75.7</td></tr></table>

significance can be obtained by optimizing the pump settings.

Tailoring the digitally synthesized microwave frequency-pump spectrum is also a way to obtain different entanglement structures in this setup. The square lattice is one example of a cluster state that can be used to implement universal quantum computation and can be generated from the Hamiltonian of Eq. (1) [36]. The digital control of the amplitude and phase of each pump tone also enables extending the measurement scheme to observe multipartite entangled states with sizes approaching the number of modes in the SAW resonator. Beyond the straight-forward generation of particular target cluster states, technical (and theoretical) challenges remain to overcome errors due to the finite squeezing and achieve fault tolerance [42].

# VI. CONCLUSIONS

We have demonstrated two-mode squeezing in a surface acoustic wave resonator, likely below the phononic vacuum level. Extending this scheme to a multitone pump spectrum and calibrated measurement chain, we observed fully inseparable multipartite entanglement of four resonator modes. The dense mode structure of the resonator enables multiplexing all modes to one measurement channel without analog frequency conversion. The small on-chip footprint of our device $(< 0.2\mathrm{mm}^2)$ further contributes to scalability.

For the measurements presented here, the pump strengths are similar to the loss rates. This limits the amount of squeezing and correlations that can be induced. To enable stronger pumping and
enhance the entanglement generation, a three-wave mixing scheme could be adopted where the pumping occurs at around twice the mode frequencies. For the same drive amplitude, this yields stronger pumping as well as less effect of saturation in the parametric amplifier.

A prospect for further development is using superconducting qubits to implement non-Gaussian operations on the resonator state such as the addition or subtraction of single phonons. Non-Gaussianity is important to many applications [43] and qubit-controlled operations on the resonator state are more readily implemented in a circuit QED setting than optical experiments where nonlinearities are typically weaker. As a step in this direction, the device used to measure multimode correlations has an integrated transmon qubit, although it was not used for this experiment.

Another promising application for this device is in quantum simulation using the resonator modes as lattice sites in a synthetic dimension. Modulating the reflector SQUID at a frequency corresponding to the free spectral range will induce nearest-neighbor hopping of phonons, giving rise to an effective lattice Hamiltonian in a hardware-efficient way.

GUSTAV ANDERSSON et al.

PRX QUANTUM 3,010312 (2022)

010312-6

# ACKNOWLEDGMENTS

We acknowledge IARPA and Lincoln Labs for providing the TWPA used in this experiment. We are grateful to G. Ferrini, I. Strandberg, and F. Quijandria for fruitful discussions. R.B., M.O.T., and D.B.H. are part owners of the company Intermodulation Products AB, which produces the digital multifrequency lock-in amplifier used in this experiment. This work was supported by the Knut and Alice Wallenberg foundation through the Wallenberg Center for Quantum Technology (WACQT), and the Swedish Research Council, VR.

# APPENDIX A: COUPLING-STRENGTH ESTIMATE

The vacuum coupling strength between the $LC$ mode of the SQUID mirror and the SAW modes is given by the overlap of the zero-point voltage fluctuations of the SAW mode with the charge fluctuations on the mirror fingers [25]. As any acoustic mode that is confined in the resonator is necessarily efficiently reflected by the mirror, we make the simplifying assumption that the SAW wavelength matches the mirror period for all modes. The amplitude of the voltage zero-point fluctuations is given by

$$
\phi_ {0} = \frac {e _ {1 4}}{\epsilon} \sqrt {\frac {\hbar}{2 \rho v _ {\mathrm {S A W}} A}}. \tag {A1}
$$

The piezoelectric coefficient $e_{14}$ and the dielectric constant $\epsilon$ , as well as the substrate density $\rho$ and SAW velocity $v_{\mathrm{SAW}}$ are material parameters, while $A$ denotes the effective mode area. The charge fluctuations across the mirror fingers are

$$
Q _ {0} = 2 e \beta \left(\frac {E _ {L}}{3 2 E _ {C}}\right) ^ {1 / 4}, \tag {A2}
$$

where $E_{L} = (\Phi_{0} / 2\pi)^{2} / L_{J}$ is the characteristic inductive energy, and $E_{C} = e^{2} / (2C)$ sets the charging energy scale. The capacitance ratio $\beta$ indicates the ratio of the mirror capacitance seen by the SAW modes to the total mirror capacitance. Because the SAW field decays exponentially into the mirror with a penetration depth $L_{p}$ , this ratio is approximately given by $\beta = L_{p} / L_{m}$ , where $L_{m}$ is the total length of the mirror. This yields an approximate coupling strength

$$
\hbar g = \phi_ {0} Q _ {0} = e \frac {e _ {1 4}}{\epsilon} \frac {L _ {p}}{L _ {m}} \left(\frac {E _ {L}}{8 E _ {C}}\right) ^ {1 / 4} \sqrt {\frac {\hbar}{\rho v _ {\mathrm {S A W}} A}}. \tag {A3}
$$

With literature values for the material parameters [44,45] and the penetration depth $L_{P}$ estimated from the measured free spectral range, we obtain $g \approx 2\pi \times 1.6$ MHz.

This weak vacuum coupling strength implies the dispersive interaction between SAW modes and the mirror are small compared to the linewidth, and parametric excitation thus relies on strong pumping.

# APPENDIX B: DEVICE $B$

The device used for the multimode entanglement experiments of Sec. V has slight design variations. The mirror edge separation is $L_{e} = 560 \mu \mathrm{m}$ . To enhance the coupling of the electromagnetic mirror mode to SAW, the number of finger pairs in the mirror has been reduced to $N_{p} = 275$ . The estimated vacuum coupling [Eq. (A3)] is $g = 2\pi \times 1.6 \mathrm{MHz}$

The total capacitance is $C = 3.3$ pF and the SQUID shunting the two electrodes has a total critical current of $I = 190$ nA ( $L_{J} = 1.7$ nH). The associated $LC$ mode has a frequency of $\omega_{LC} = 2\pi \times 2.1$ GHz. To enable further operations on the resonator state, a transmon qubit has been integrated between the port IDT and left-hand mirror. While not used in the present experiment, qubit operations could be relevant to creating non-Gaussian SAW states. A microscope image of the device is shown in Fig. 5.

# APPENDIX C: EXPERIMENTAL DETAILS

A schematic illustrating the measurement setup is shown in Fig. 6. The digital microwave measurement platform has eight channels of high speed digital-to-analog (DAC) and analog-to-digital converters (ADC) serviced by a large field programmable gate array (FPGA), all synchronized to one stable clock [34]. With this setup we are able to digitally synthesize drive signals and digitize response signals in the band $2 - 4\mathrm{GHz}$ , without analog $IQ$ mixers.

We set the sampling clock at four GSamples/s for the ADCs, and at five GSamples/s for the DACs, resulting in Nyquist frequencies of 2 and $2.5\mathrm{GHz}$ , respectively. The SAW cavity modes are designed to fall in the band $3.8 - 3.9\mathrm{GHz}$ , within the second Nyquist zone of the converters. Tones outside the second Nyquist zone are removed with external bandpass filters.

![](dt=2026-06-06/ht=05/1a8b31cea0ffd2968f4d916b200df98877c0f0834a2fa1d98442276db8a850d3.jpg)

SQUEEZING AND MULTIMODE ENTANGLEMENT...

PRX QUANTUM 3,010312 (2022)

010312-7

![](dt=2026-06-06/ht=05/f9bd180e732d8a2d0bf96cf2ba938a12762ecf7b191ed84444b157462dbe5746.jpg)

# APPENDIX D: PARAMETRIC COUPLING MODEL

In this model we consider the coupling induced between linear SAW modes by the common interaction with the $LC$ resonance of the mirror under parametric modulation. If we indicate the $LC$ mode and SAW modes by the $a$ and $b_{j}$ operators, respectively, the total Hamiltonian is a sum of three terms, $H = H_{0} + V + D(t)$ given by

$$
H _ {0} / \hbar = \omega_ {L C} a ^ {\dagger} a + \sum_ {j} \omega_ {j} b _ {j} ^ {\dagger} b _ {j}, \tag {D1}
$$

$$
V / \hbar = i \sum_ {j} g (a - a ^ {\dagger}) (b _ {j} + b _ {j} ^ {\dagger}), \tag {D2}
$$

$$
D (t) / \hbar = - \frac {1}{2} \omega_ {L C} \left(1 - \cos \frac {\pi \Phi (t)}{\Phi_ {0}}\right) (a + a ^ {\dagger}) ^ {2}, \quad (\mathrm {D 3})
$$

where the driving term $D(t)$ is the time-dependent part of the Josephson energy due to the flux pump $\Phi (t)$ .

If $|\Phi(t)| \ll \Phi_0$ , we can expand the driving term $D(t)$ to second order in $\Phi(t)$

$$
D (t) \approx - \frac {1}{4} \hbar \omega_ {L C} \left(\frac {\pi \Phi (t)}{\Phi_ {0}}\right) ^ {2} (a + a ^ {\dagger}) ^ {2}. \tag {D4}
$$

In the case for a single flux pump, the time dependence is described by a sinusoidal function $\Phi (t) = \Phi_{AC}\cos (\omega_{p}t + \theta)$ . Inserting this into Eq. (D4), we arrive at the expression of the drive term for a single flux pump at $\omega_{p}$

$$
\begin{array}{l} D (t) = - 2 d \cos^ {2} (\omega_ {p} t + \theta) (a + a ^ {\dagger}) ^ {2} \\ = - d \left[ \cos (2 \omega_ {p} t + 2 \theta) + 1 \right] (a + a ^ {\dagger}) ^ {2}, \tag {D5} \\ \end{array}
$$

where $d = \hbar \omega_{LC}(\pi \Phi_{\mathrm{AC}} / 2\Phi_0)^2 /2$ is the effective pump amplitude. For multiple flux pumps at different frequencies, $\Phi (t)$ is instead a superposition of sinusoidal functions.

In the limit o
f weak interaction between the electromagnetic mirror mode and SAW, $g \ll |\omega_i - \omega_{LC}|$ , we perturbatively expand the Hamiltonian by applying the Schrieffer-Wolff transformation [46]

$$
H ^ {\prime} = e ^ {- S} H e ^ {S} \approx H _ {0} + \frac {1}{2} [ S, V ] + e ^ {- S} D (t) e ^ {S} \tag {D6}
$$

where

$$
\begin{array}{l} S = \sum_ {j} \left[ \frac {i g}{\omega_ {L C} - \omega_ {j}} \left(a ^ {\dagger} b _ {j} + a b _ {j} ^ {\dagger}\right) \right. \\ \left. + \frac {i g}{\omega_ {L C} + \omega_ {j}} \left(a b _ {j} + a ^ {\dagger} b _ {j} ^ {\dagger}\right) \right] \tag {D7} \\ \end{array}
$$

and we use the form of the drive term $D(t)$ given by Eq. (D5). The commutator $[S, V]$ is therefore

$$
\begin{array}{l} [ S, V ] = - g \hbar \left(\sum_ {j} \tilde {g} _ {j} (a - a ^ {\dagger}) ^ {2} \right. \\ \left. - \sum_ {j, k} \bar {g} _ {j} \left(b _ {j} + b _ {j} ^ {\dagger}\right) \left(b _ {k} + b _ {k} ^ {\dagger}\right)\right), \tag {D8} \\ \end{array}
$$

where we introduce the effective coupling rates $\tilde{g}_j$ and $\bar{g}_j$ :

$$
\tilde {g} _ {j} = \frac {2 g \omega_ {j}}{\omega_ {j} ^ {2} - \omega_ {L C} ^ {2}}, \tag {D9}
$$

$$
\bar {g} _ {j} = \frac {- 2 g \omega_ {L C}}{\omega_ {j} ^ {2} - \omega_ {L C} ^ {2}}. \tag {D10}
$$

GUSTAV ANDERSSON et al.

PRX QUANTUM 3,010312 (2022)

010312-8

We also treat the pump-dependent term $e^{-S}D(t)e^{S}$ perturbatively, by expanding up to second order in $S$ according to

$$
e ^ {- S} D (t) e ^ {S} \approx D (t) - [ S, D (t) ] + \frac {1}{2} \{S, [ S, D (t) ] \}. \tag {D11}
$$

The commutators are calculated to be

$$
[ S, D (t) ] = - 2 d \left[ \cos \left(2 \omega_ {p} t + 2 \theta\right) + 1 \right] \sum_ {j} i \tilde {g} _ {j} \left(a + a ^ {\dagger}\right) \left(b _ {j} - b _ {j} ^ {\dagger}\right), \tag {D12}
$$

$$
\{S, [ S, D (t) ] \} = - 2 d \left[ \cos \left(2 \omega_ {p} t + 2 \theta\right) + 1 \right] \left(- \sum_ {j, k} \tilde {g} _ {j} \tilde {g} _ {k} \left(b _ {j} - b _ {j} ^ {\dagger}\right) \left(b _ {k} - b _ {k} ^ {\dagger}\right) + \sum_ {j} \bar {g} _ {j} \tilde {g} _ {j} \left(a + a ^ {\dagger}\right) ^ {2}\right), \tag {D13}
$$

where we neglect an unimportant constant term.

To summarize, we write down the Hamiltonian $H^{\prime}$

$$
\begin{array}{l} H ^ {\prime} = H _ {0} + D (t) - \frac {g \hbar}{2} \left(\sum_ {j} \tilde {g} _ {j} (a - a ^ {\dagger}) ^ {2} - \sum_ {j, k} \bar {g} _ {j} (b _ {j} + b _ {j} ^ {\dagger}) (b _ {k} + b _ {k} ^ {\dagger})\right) \\ + 2 d \left[ \cos \left(2 \omega_ {p} t + 2 \theta\right) + 1 \right] \sum_ {j} i \tilde {g} _ {j} (a + a ^ {\dagger}) \left(b _ {j} - b _ {j} ^ {\dagger}\right) \\ + d \left[ \cos \left(2 \omega_ {p} t + 2 \theta\right) + 1 \right] \left(\sum_ {j, k} \tilde {g} _ {j} \tilde {g} _ {k} \left(b _ {j} - b _ {j} ^ {\dagger}\right) \left(b _ {k} - b _ {k} ^ {\dagger}\right) - \sum_ {j} \bar {g} _ {j} \tilde {g} _ {j} \left(a + a ^ {\dagger}\right) ^ {2}\right), \tag {D14} \\ \end{array}
$$

where the last term produces squeezing and beam-splitter interactions. Finally, we perform a resonance approximation by dropping all rapidly oscillating terms. This leaves us with the Hamiltonian $\tilde{H}$

$$
\begin{array}{l} \tilde {H} = \hbar \tilde {\omega} _ {L C} a ^ {\dagger} a + \hbar \sum_ {j} \left(\tilde {\omega} _ {j} - 2 d \tilde {g} ^ {2}\right) b _ {j} ^ {\dagger} b _ {j} \\ + \sum_ {j} \sum_ {k = 2 p - j} \frac {d \tilde {g} _ {j} \tilde {g} _ {k}}{2} \left(e ^ {2 i (\omega_ {p} t + \theta)} b _ {j} b _ {k} + e ^ {- 2 i (\omega_ {p} t + \theta)} b _ {j} ^ {\dagger} b _ {k} ^ {\dagger}\right), \tag {D15} \\ \end{array}
$$

where the final sum is only over SAW modes $k$ satisfying the four-wave mixing criterion $\omega_{k} = 2\omega_{p} - \omega_{j}$ . For multiple pumps, this would result in several four-wave mixing criteria (one for each pump) and thus couple each SAW mode to more modes. We also define the renormalized frequencies $\tilde{\omega}_{LC}$ and $\tilde{\omega}_{j}$ as

$$
\tilde {\omega} _ {L C} = \omega_ {L C} + g \sum_ {j} \tilde {g} _ {j} - \frac {2 d}{\hbar} \left(1 + \sum_ {j} \bar {g} _ {j} \tilde {g} _ {j}\right), \tag {D16}
$$

$$
\tilde {\omega} _ {j} = \omega_ {j} + g \bar {g} _ {j}. \tag {D17}
$$

# APPENDIX E: CALCULATING THE THEORETICAL COVARIANCE MATRIX

Here we outline how to arrive at a covariance matrix from a system of Langevin equations. Calculating the Heisenberg equations of motion for the SAW modes $b$ using Hamiltonian $\tilde{H}$ Eq. (D15), we arrive at a system of equations describing multiple parametrically coupled modes. Assuming identical external couplings $\gamma$ and no internal losses, we arrive at [47]

$$
\dot {b} _ {j} + i \left(\tilde {\omega} _ {j} - 2 d \tilde {g} _ {j} ^ {2} / \hbar\right) b _ {j} + \frac {\gamma}{2} b _ {j} + i \sum_ {k} \epsilon_ {j k} b _ {k} ^ {\dagger} = \sqrt {\gamma} b ^ {\mathrm {i n}}, \tag {E1}
$$

where we define the complex parametric coupling $\epsilon_{jk}$ to be

$$
\epsilon_ {j k} = \frac {d \tilde {g} _ {j} \tilde {g} _ {k}}{2 \hbar} e ^ {- 2 i (\omega_ {p} t + \theta)}. \tag {E2}
$$

The sum is made over all modes $k$ satisfying all four-wave mixing criteria.

Since the FSR is small compared to $\tilde{\omega}_j$ and the detuning between the SAW modes and the mirror electromagnetic mode, we make the approximation $\tilde{g}_j\approx \tilde{g}_k$ , which corresponds to the idealized case with identical parametric

SQUEEZING AND MULTIMODE ENTANGLEMENT...

PRX QUANTUM 3,010312 (2022)

010312-9

![](dt=2026-06-06/ht=05/bd94d398dbb188947ef44aaf04e3c9a1335a449dd57065917fb67e973024a0f7.jpg)

couplings. Equation (E1) is then instead

$$
\dot {b} _ {j} + i \left(\tilde {\omega} _ {j} - 4 | \epsilon |\right) b _ {j} + \frac {\gamma}{2} b _ {j} + i \epsilon \sum_ {k} b _ {k} ^ {\dagger} = \sqrt {\gamma} b ^ {\mathrm {i n}}. \tag {E3}
$$

Working in the frequency domain is more convenient. The Fourier transform of Eq. (E3) is

$$
- i \Delta_ {j} b _ {j} [ \Omega_ {j} ] + i \epsilon \sum_ {k} b _ {j} ^ {\dagger} [ \Omega_ {j} ] = \sqrt {\gamma} b ^ {\mathrm {i n}} [ \Omega_ {j} ], \tag {E4}
$$

where $\Delta_j = \Omega_j - \tilde{\omega}_n + 4|\epsilon| + i\gamma/2$ . If the measurement frequency $\Omega_j = \tilde{\omega}_j - 4|\epsilon|$ , the expression simplifies to $\Delta = i\gamma/2$ .

For many pumps and modes, the system of equations can be conveniently summarized by a complex weighted directed graph [48]. We consider the case for the multipartite entanglement result of Fig. 4 and draw the corresponding graph in Fig. 7.

The graph illustrates the mode couplings and can also be associated with a mode-coupling matrix $M$ . In the basis of $\bar{b} = (b_{1},\ldots ,b_{4},b_{1}^{\dagger},\ldots ,b_{4}^{\dagger})$ , the matrix $M$ is

$$
M = \left( \begin{array}{c c c c c c c c} \Delta_ {1} & 0 & 0 & 0 & 0 & - \epsilon & 0 & - \epsilon \\ 0 & \Delta_ {2} & 0 & 0 & - \epsilon & 0 & - \epsilon & 0 \\ 0 & 0 & \Delta_ {3} & 0 & 0 & - \epsilon & 0 & - \epsilon \\ 0 & 0 & 0 & \Delta_ {4} & - \epsilon & 0 & - \epsilon & 0 \\ 0 & \epsilon^ {*} & 0 & \epsilon^ {*} & - \Delta_ {1} ^ {*} & 0 & 0 & 0 \\ \epsilon^ {*} & 0 & \epsilon^ {*} & 0 & 0 & - \Delta_ {2} ^ {*} & 0 & 0 \\ 0 & \epsilon^ {*} & 0 & \epsilon^ {*} & 0 & 0 & - \Delta_ {3} ^ {*} & 0 \\ \epsilon^ {*} & 0 & \epsilon^ {*} & 0 & 0 & 0 & 0 & - \Delta_ {4} ^ {*} \end{array} \right), \tag {E5}
$$

which provides a complete description of the frequency-domain expression by $-iM\bar{b} = \sqrt{\gamma}\bar{b}^{\mathrm{in}}$

The covariance matrix $V$ can be obtained from $M$ via the scattering matrix $S$ , given by

$$
S = i K M ^ {- 1} K - I, \tag {E6}
$$

$$
K = \sqrt {\gamma} I. \tag {E7}
$$

The scattering matrix relates the incoming modes to the outgoing modes $\bar{b}^{\mathrm{out}} = S\bar{b}^{\mathrm{in}}$ according to input-output theory [47-49]. The covariance matrix of the incoming
noise modes $V^{\mathrm{in}}$ then transforms as [7]

$$
V ^ {\mathrm {o u t}} = S _ {I Q} V ^ {\mathrm {i n}} S _ {I Q} ^ {T}, \tag {E8}
$$

where the subscript $IQ$ indicates the scattering matrix has been (linearly) transformed into the quadrature basis defined by $I = b + b^{\dagger}$ , $Q = -i(b - b^{\dagger})$ . Any Gaussian state is fully characterized by $V^{\mathrm{out}}$ .

In the case of internal losses $\gamma^{\mathrm{int}}$ , we need to make some minor adjustments to the preceding method. The definition of $\Delta_j$ is adjusted to be $\Delta_j = \Omega_j - \tilde{\omega}_j + 4|\epsilon| + i\gamma^{\mathrm{tot}}/2$ with $\gamma^{\mathrm{tot}} = \gamma + \gamma^{\mathrm{int}}$ . In addition, we introduce the diagonal matrix $K^{\mathrm{int}} = \sqrt{\gamma^{\mathrm{int}}}I$ , which is used to define a scattering matrix for the loss channel as

$$
S ^ {\text {l o s s}} = i K M ^ {- 1} K ^ {\text {i n t}}. \tag {E9}
$$

In the presence of internal losses, Eq. (E8) is instead replaced by

$$
V ^ {\text {o u t}} = S _ {I Q} V ^ {\text {i n}} S _ {I Q} ^ {T} + S _ {I Q} ^ {\text {l o s s}} V ^ {\text {d o s s}} \left(S _ {I Q} ^ {\text {l o s s}}\right) ^ {T}. \tag {E10}
$$

The noise coming from the internal loss port is characterized by $V^{\mathrm{loss}}$ . Typically, it is assumed to be identical to the incoming noise port $V^{\mathrm{in}} = V^{\mathrm{loss}}$ .

So far in this discussion, the pump modes have been ignored. These modes form a set of correlated modes, which is separate from the probe modes and can therefore be ignored in the analysis. We also note that due to the restricted probe mode set, the highest-frequency pump tone used in the measurement does not contribute to the measured correlations.

A covariance matrix calculated theoretically from Eqs. (E5)-(E10) is shown in Fig. 4(c). We use a uniform pump strength of $|\epsilon| = 30 \, \mathrm{kHz}$ and equal external and internal loss rates of $\gamma = \gamma_{\mathrm{ext}} = 20 \, \mathrm{kHz}$ . Although simplified, this configuration corresponds approximately to that of the multimode entanglement experiment and the theoretical covariance matrix qualitatively reproduces the features of the measured data shown in Fig. 4(b).

GUSTAV ANDERSSON et al.

PRX QUANTUM 3,010312 (2022)

010312-10

# APPENDIX F: AMPLIFIER GAIN AND ADDED NOISE

We model the effect of amplification on the covariance matrix according to [7]

$$
\tilde {V} = T V T + N, \tag {F1}
$$

where $T = \sqrt{G} I$ and $N = (G - 1)(2n + 1)I$ . Our amplifier chain is characterized by an effective amplitude gain $\sqrt{G}$ and effective added mean photon number $n$ . The covariance matrix measured after amplification is given by $\tilde{V}$ , while $V$ represents the quantum state. Thus a good estimate of $\sqrt{G}$ and $n$ would allow us to reconstruct the quantum statistics.

# 1. Two-mode squeezed state

There are different methods to calibrate gain and added noise, which typically require some form of calibrated noise source [20] or a temperature sweep [37,50]. A method to roughly estimate the gain of the amplification chain using only our device is by measuring cross-correlations in a two-mode squeezed state. Assuming the added noise is thermal, the cross-correlations corresponding to squeezing are independent of the added noise but not the amplifier gain. We quantify these correlations by adding the relevant elements of the covariance matrix as

$$
C = \sqrt {\tilde {V} _ {1 3} ^ {2} + \tilde {V} _ {1 4} ^ {2} + \tilde {V} _ {2 3} ^ {2} + \tilde {V} _ {2 4} ^ {2}}. \tag {F2}
$$

We measure $C$ by applying a flux pump at 3.8732 GHz, while measuring the noise in a pair of neighboring modes. As illustrated in Fig. 8, the pump frequency is placed directly on a SAW mode and probe frequencies are swept across neighboring SAW modes, always keeping them strictly symmetric with respect to the pump to satisfy the four-wave mixing criterion. In Fig. 8, $C$ is plotted as a function of the detuning between the pump and the probe frequencies. The probe-frequency sweep results in a Lorentzian-like shape of $C$ as a function of detuning, where the peak occurs when both probes are located within their respective SAW modes.

The $C$ lineshape is calculated by deriving the covariance matrix $\tilde{V}$ for two coupled SAW modes, according to the method outlined in Appendix E. More specifically, the mode-coupling matrix $M$ for two modes is graphically represented in Fig. 9. The outgoing noise is then fully characterized by the $4\times 4$ matrix $V^{\mathrm{out}}$ , which we can find by following Eq. (E6)-(E10). Amplification is taken into account by substituting $V\rightarrow V^{\mathrm{out}}$ in Eq. (F1). The resulting covariance matrix $\tilde{V}$ is used to calculate $C$ .

Given the resonance frequencies of the SAW modes along with their linewidths and assuming $\tilde{g}_1 = \tilde{g}_2$ , we are left with three unknown parameters: the gain $G$ , parametric coupling $\epsilon$ , and the effective phonon temperature $T_{\mathrm{eff}}$ .

![](dt=2026-06-06/ht=05/b3371d638c69c6a89a8ef4190453e8b3acba2a937f94cd874041b0dffe7a9e9f.jpg)

![](dt=2026-06-06/ht=05/e80066a0a8650497009c8b0176b8d75a1ab2c2dc78288d0a4ecaa6e4b635b511.jpg)

If we fix the phonon temperature, a fit to $C$ will give us the gain $G$ and the parametric coupling $\epsilon$ . An example fit is shown as a solid line in Fig. 8, with an assumed phonon temperature of $T_{\mathrm{eff}} = 30 \mathrm{mK}$ . This yields an estimate of the gain to be $G \approx 80 \mathrm{dB}$ , which lies within the range of our expectations.

This method is not a substitute for proper gain and noise calibration procedures. However, we use this method to make an estimate on the maximal phonon temperature possible for the two-mode squeezing in Fig. 3 to be a signature of entanglement. The procedure consists of essentially three steps: extract $G$ by fitting to $C$ , estimate the added noise $N$ , and finally reconstruct the original quantum statistics from data in Fig. 3 according to Eq. (F1).

After the gain $G$ is extracted from fitting to $C$ , the amplifier noise is estimated by solving for $n$ in Eq. (F1), by replacing $\tilde{V}$ by the pump off statistics $V_{\mathrm{off}}$ while $V$

![](dt=2026-06-06/ht=05/e7318ac6bf4edae636b51d7ff81fc58b2e9faefe9e721dcc93890ac3bab53b5c.jpg)

SQUEEZING AND MULTIMODE ENTANGLEMENT...

PRX QUANTUM 3,010312 (2022)

010312-11

is substituted by a thermal state with the corresponding temperature $T_{\mathrm{eff}}$ . Finally, solving for $n$ with these assumptions yields $n \approx 0.08$ . This added noise value is a very low estimate. Using a higher added noise value during reconstruction of $V$ , however, will result in more squeezing.

With the gain $G$ and noise $n$ , one can attempt reconstructing the preamplified covariance matrix from the two-mode squeezing data in Fig. 3. To determine whether the reconstructed state is entangled, we apply the partial positive transpose (PPT) criterion [7,39]. If the reconstructed two-mode squeezed state is labeled $V_{\mathrm{TMS}}$ , the PPT criteria states that if we do a partial transposition:

$$
\Lambda = \operatorname {d i a g} (1, 1, 1, - 1), \tag {F3}
$$

$$
\bar {V} _ {\mathrm {T M S}} = \Lambda V _ {\mathrm {T M S}} \Lambda , \tag {F4}
$$

then a necessary and sufficient condition for separability for bipartite Gaussian states is that the matrix $H = \bar{V}_{\mathrm{TMS}} + i\Omega$ is positive semidefinite. $\Omega$ is the symplectic matrix, defined as

$$
\Omega = \left( \begin{array}{c c c c} 0 & 1 & 0 & 0 \\ - 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & - 1 & 0 \end{array} \right). \tag {F5}
$$

Thus we test for entanglement by calculating the eigenvalues
$\lambda$ of $H$ and checking whether the smallest eigenvalue $\lambda_{\mathrm{min}}$ is negative. Note that for $\Omega$ and $\Lambda$ we are assuming the covariance matrix is in the basis $(I_1, Q_1, I_2, Q_2)$ .

However, the eigenvalue $\lambda_{\mathrm{min}}$ depends on the phonon temperature, since it affects our estimate of $G$ and $n$ . We take this into account by calculating the value of $\lambda_{\mathrm{min}}$ at various phonon temperatures $T_{\mathrm{eff}}$ , presented in Fig. 10. Accordingly, we observe that entanglement persists for phonon temperatures up to $63~\mathrm{mK}$ . This should be compared to the mixing chamber temperature of roughly $10~\mathrm{mK}$ and previous experiments estimating the effective SAW phonon temperature to $37~\mathrm{mK}$ [51]. Together, these observations suggest that the measured two-mode squeezing is a signature of entangled SAW modes.

# 2. Calibration and multimode state reconstruction

For the multimode entanglement experiment we perform a calibration of the gain and added noise in the amplification chain. We substitute a resistor at the mixing chamber for the device and measure the noise power as a function of temperature. Fits to the expression

$$
P = G h f \left[ \frac {1}{2} \coth \frac {h f}{2 k _ {B} T _ {\mathrm {m x c}}} + \frac {1}{2} (2 n + 1) \right] \tag {F6}
$$

give the gain and noise parameters. The calibration is performed without the resonator connected at each frequency

![](dt=2026-06-06/ht=05/d8e09cdb236333c8b18ecafd5a95204dee583604c1bdaacac3578b1ab94b3a86.jpg)

used in the measurement and the heating raises the temperature of the entire mixing chamber stage of the cryostat. Figure 11 shows the noise power at the frequency of mode $f_{2}$ (cf. Fig. 4). As the amplification chain makes use of a parametric amplifier, care must be taken to account for the idler noise in the analysis [52,53].

To accurately account for the variation in gain with frequency, the gain and noise terms appearing in Eq. (F1) are extended to $T = \bigoplus_{i=1}^{n} \sqrt{G_i} I$ and $N = \bigoplus_{i=1}^{n} (G_i - 1)(2n + 1)I + (G_{I,i} - 1)(2n_{I,i} + 1)I$ . The index $i$ denotes the frequency modes and the subscript $I$ denotes the idler contribution. To obtain reasonable fit parameters, we restrict the signal-idler gain to $G_{I,i} = G_i$ and assume an idler noise to originate from a thermal state at temperature $T_I = 30 \mathrm{mK}$ . From our analysis we obtain a lower than expected added noise temperature of $T_n \leq 300 \mathrm{mK}$ .

![](dt=2026-06-06/ht=05/ab7191a0cc6292ecbd2a681cf312140e6f4a1f5b7bbe819ae383303626216c42.jpg)

GUSTAV ANDERSSON et al.

PRX QUANTUM 3,010312 (2022)

010312-12

In case our calibration underestimates the real added noise in the amplification chain, that should imply more entanglement in the reconstructed multimode state as the added noise obscures the correlations.

The fits yield uncertainties for the estimated gain and noise, which influence the entanglement significance by error propagation. The error accounting for the calibration as well as measurement error can be written as

$$
\sigma_ {i j} ^ {2} = \sigma_ {i j, A} ^ {2} + \sigma_ {i j, B} ^ {2} + \sigma_ {i j, C} ^ {2}. \tag {F7}
$$

The contribution related to uncertainty in the signal gain is given by

$$
\begin{array}{l} \sigma_ {i j, A} ^ {2} = \left[ \left(\frac {\tilde {V} _ {i j}}{2 \sqrt {G _ {i} ^ {3} G _ {j}}} \sigma_ {G _ {i}}\right) ^ {2} + \left(\frac {\tilde {V} _ {i j}}{2 \sqrt {G _ {j} ^ {3} G _ {i}}} \sigma_ {G _ {j}}\right) ^ {2} \right] (1 + \delta_ {i j}) \\ + 2 \delta_ {i j} \left(\frac {2 n _ {i} + 1}{G _ {i}} \sigma_ {G _ {i}}\right) ^ {2}. \tag {F8} \\ \end{array}
$$

Additional contributions arise from uncertainty in the added noise $\sigma_{ii,B} = 2\sigma_{n_i}$ as well as the measured fluctuations in the covariance matrix

$$
\sigma_ {i j, C} = \frac {\sigma_ {\tilde {V} _ {i j}}}{\sqrt {G _ {i} G _ {j}}}. \tag {F9}
$$

Here, $\sigma_{\tilde{V}_{ij}}$ is given by the standard error of the mean of each covariance matrix element as calculated from the measured data. Furthermore, we cannot assume that errors in gain and noise are uncorrelated, which leads to the diagonal error term

$$
\sigma_ {i j, \text {c o r r}} ^ {2} = 4 \frac {V _ {i i} - (2 n _ {i} + 1)}{G _ {i}} \operatorname {c o v} \left(G _ {i}, n _ {i}\right) \delta_ {i j}. \tag {F10}
$$

The covariance matrix for a physical state must satisfy the Heisenberg uncertainty relations, which may be expressed as

$$
V \geq 0, \tag {F11}
$$

$$
V - i \Omega \geq 0. \tag {F12}
$$

where $\Omega$ is the symplectic matrix [cf. Eq. (F5)]. Due to measurement noise and drift, this is not guaranteed to hold for the covariance matrix $V$ obtained by inverting Eq. (F1). To ensure a physical state before applying entanglement tests, we apply a reconstruction to find the most probable physical state $V$ given a noisy measured state $V'$ [38]. This $V$ is obtained by solving the optimization problem

$$
\min  _ {V} \left(\max  _ {\alpha \beta} \frac {\left| V _ {\alpha \beta} ^ {\prime} - V _ {\alpha \beta} \right|}{\sigma_ {\alpha \beta}}\right). \tag {F13}
$$

The pump configuration used in our measurement would suggest including more probe modes at higher frequency

![](dt=2026-06-06/ht=05/ffc5b55d00e65bdea57fc711b9388b03aa5cb42bfe060826deb2a1c3831220d2.jpg)

![](dt=2026-06-06/ht=05/60c9f67ec68e6bdc1c430e4bada6982cd2080f5a6f35ad74bc1669407c6d2630.jpg)

![](dt=2026-06-06/ht=05/c1023f7d2ebe7ba16ce325b545f34420b896a1b5263388d156d0916d24e4d9be.jpg)

![](dt=2026-06-06/ht=05/772b6b5e6db491a0bb2092eaf704acacdea20239acc7078fa6ea6f4966d218e4.jpg)

![](dt=2026-06-06/ht=05/90fb52106abb3d891d5a3bf4c83a8fe72a6662ee208da29f94c9edee55f141d2.jpg)

in the analysis. Including these modes renders the covariance matrix obtained unphysical by multiple standard deviations, presumably due to error in our calibration.

# APPENDIX G: TWO-MODE QUADRATURE HISTOGRAMS

Figure 3(b) shows the two-mode quadrature histograms in the $I_{+} - I_{-}$ plane. Here, the squeezing axis corresponds to the diagonal. In Fig. 12 we plot all two-mode quadrature histograms for the mode pair closest to the pump. The squeezing is manifest also in the $Q_{+} - Q_{-}$ histogram, while the other quadratures show amplified noise.

The $I_{+} - I_{-}$ histogram is shown without subtraction in Fig. 13(a). In Fig. 13(b) we plot the reference histogram obtained with the pump off that is subtracted to generate the data shown in Fig. 12.

![](dt=2026-06-06/ht=05/f378f8bf903272b3a95e33aad246c6e32ddd6bed5611257fae59743aa25e8917.jpg)

![](dt=2026-06-06/ht=05/d6db44762902db9a50d43ae2bdc90353b708a239aeebd9d2d601a8230a4a92da.jpg)

SQUEEZING AND MULTIMODE ENTANGLEMENT...

PRX QUANTUM 3,010312 (2022)

010312-13

![](dt=2026-06-06/ht=05/befe58c5396e74d6c604b0eeed1d21e36ca733767b65fce00b3f9dd77bd1bbac.jpg)

# APPENDIX H: SCATTERING MEASUREMENTS

To verify the mode couplings induced by parametric pumping we perform scattering measurements in sample $B$ . A signal is injected into one mode via the IDT and the scattering into other modes is measured. The scattering matrix for a single pump tone (f
our tones) is shown in Fig. 14 (Fig. 15). The scattering measurements verify that the parametric couplings relied on to generate entanglement are present. For an evenly spaced pump comb the couplings are not all to all due to the deviation from uniform SAW mode spacing.

The phase and amplitude of the scattered signal is sensitive to the pump configuration. Figure 16 shows the $I$ and $Q$ quadratures of a single scattering matrix element as a function of the pump spacing.

![](dt=2026-06-06/ht=05/35e00eb1fa2db626cff7da59842d69120f12db832046ee80f2f784fb6803faa5.jpg)

![](dt=2026-06-06/ht=05/2420b8734c092a4b6a658eac69546b564ff625052ce5712de03c2c27e950aaf1.jpg)

GUSTAV ANDERSSON et al.

PRX QUANTUM 3,010312 (2022)

010312-14

SQUEEZING AND MULTIMODE ENTANGLEMENT...

PRX QUANTUM 3, 010312 (2022)

010312-15

GUSTAV ANDERSSON et al.

PRX QUANTUM 3,010312 (2022)

010312-16