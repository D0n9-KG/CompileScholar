# RESEARCH ARTICLE

# QUANTUM PHYSICS

# Splitting phonons: Building a platform for linear mechanical quantum computing

H. Qiao $^{1}$ , É. Dumur $^{1,2}$ , G. Andersson $^{1}$ , H. Yan $^{1}$ , M.-H. Chou $^{1,3}$ , J. Grebel $^{1}$ , C. R. Conner $^{1}$ , Y. J. Joshi $^{1}$ , J. M. Miller $^{1,3}$ , R. G. Povey $^{1,3}$ , X. Wu $^{1}$ , A. N. Cleland $^{1,2*}$

Linear optical quantum computing provides a desirable approach to quantum computing, with only a short list of required computational elements. The similarity between photons and phonons points to the interesting potential for linear mechanical quantum computing using phonons in place of photons. Although single-phonon sources and detectors have been demonstrated, a phononic beam splitter element remains an outstanding requirement. Here we demonstrate such an element, using two superconducting qubits to fully characterize a beam splitter with single phonons.

We further use the beam splitter to demonstrate two-phonon interference, a requirement for two-qubit gates in linear computing. This advances a new solid-state system for implementing linear quantum computing, further providing straightforward conversion between itinerant phonons and superconducting qubits.

linear optical quantum computing (LOQC) presents a scalable approach to quantum computing that relies only on relatively simple optical elements such as beam splitters, phase shifters, and single-photon sources and detectors (1). The similarity between photons and phonons poses the question as to whether linear mechanical quantum computing (LMQC) might be achieved using phonons.

Prior experiments with phonons in solid systems have included the quantum control of mechanical motion (2-4), entanglement between macroscopic mechanical objects (5-8), coupling between surface acoustic waves and qubits (9-13), the deterministic emission and detection of individual surface acoustic wave (SAW) phonons (14, 15), and the transmission of quantum information (14-18), among other demonstrations (19, 20).

Here, we explore the potential for linear quantum computing by demonstrating a phonon beam splitter for SAW phonons, first showing that the beam splitter deterministically converts a single incident phonon to a superposition output state, with one phonon in either of the two output channels. This is a phase-coherent process, which we further exploit to demonstrate a single-phonon interferometer, using qubits to control the phonon phase. We further explore two-phonon interference through the Hong-Ou-Mandel (HOM) effect (21), central to a controlled-phase gate in LOQC (1, 22), using two SAW phonons whose simul

taneous arrival suppresses the output of coincident phonons in the two output channels, in favor of a superposed two-phonon-per-channel output, with a suppression visibility of $0.910 \pm 0.013$ .

These results demonstrate the basic toolset to begin exploring linear mechanical quantum computing, which is perhaps unexpected given that in our system, a single phonon represents the collective motion of a large number $(\sim 10^{15})$ of atoms.

# Device description

Our device comprises two superconducting Xmon qubits, $\mathrm{Q}_1$ and $\mathrm{Q}_2$ (23, 24), coupled via two tunable couplers $\mathrm{G}_1$ and $\mathrm{G}_2$ (25) to two unidirectional interdigitated transducers, $\mathrm{UDT}_1$ and $\mathrm{UDT}_2$ (15). These are linked by a 2-mm-long SAW phonon channel, interrupted by a phonon beam splitter BS (Fig. 1, A to C). The beam splitter, comprising a set of 16 parallel metal fingers, is designed to reflect approximately half of the acoustic signal, while transmitting the remainder. Details for the qubit, UDT, and BS designs appear in (26).

The beam splitter is intentionally positioned $300\mathrm{nm}$ closer to $\mathrm{UDT}_1$ than $\mathrm{UDT}_2$ , resulting in 214- and 290-ns travel times from each transducer to the beam splitter. The variable coupler between each qubit and its associated UDT allows us to shape the phonon emission, with typical emission times ranging from 14 ns (maximum coupling) to more than $10~\mu \mathrm{s}$ (minimum, "off" coupling), which we characterize from the couplers' time-dependent emission rates $\kappa_{1,2}(t)$ (fig. S2) (14).

The qubits, variable couplers, and their associated control and readout lines are fabricated on a sapphire substrate, whereas the acoustic elements (UDTs and BS) are fabricated on a separate lithium niobate substrate. After fabrication, the two dies are aligned and attached to one another using a flip-chip

assembly (27). The device is operated in a tion refrigerator with a base temperature about $10\mathrm{mK}$

# Single-phonon beam splitter

We first characterize the system by measuring the response to single phonons with both qubits set to the operating frequency of $3.925\mathrm{GHz}$ . One qubit (either $\mathbf{Q}_1$ or $\mathbf{Q}_2$ ) is excited to its $|e\rangle$ state and a phonon is emitted, where we shape the emission to have a hyperbolic secant waveform, $\phi_{1,2}(t) \propto \mathrm{sech}\left(t / 2\sigma_{1,2}\right)$ , through the calibrated time-dependent modulation of the qubit's variable coupling rate $\kappa_{1,2}(t)(14)$ ; the characteristic wave packet width is $\sigma_{1,2} = 17.9$ ns.

The phonon released from $\mathbf{Q}_1(\mathbf{Q}_2)$ interacts with the beam splitter, ideally resulting in the output state $(i|10\rangle + |01\rangle) / \sqrt{2}((|10\rangle + i|01\rangle) / \sqrt{2})$ ; see also fig. S1, B and C. We use the notation $|\mathrm{ph}_1\mathrm{ph}_2\rangle$ , where $\mathrm{ph}_{1,2}$ denotes the phonon number in the output channel directed toward $\mathbf{Q}_{1,2}$ , respectively. The beam splitter output is then captured by both qubits, through a calibrated time-dependent variation of each qubit's coupling rate. In Fig.

1, D and E, we display the excited-state probability $P_{\mathrm{Q1,2}}(t)$ for each qubit and their joint excitation probability $P_{ee}(t)$ as a function of measurement time $t$ . The joint excitation probability $P_{ee}(t)$ remains very small, consistent with the expectation of no joint qubit excitation in this measurement, as each experiment involves only one phonon at a time.

From these data, we extract a beam splitter reflectivity $\eta = 0.61$ and effective itinerant phonon lifetime $\tau_{\mathrm{ph}} = 1.3~\mu \mathrm{s}$ , in agreement with a separate characterization of the BS design (fig. S1D) and previous measurements of phonon propagation loss in similar systems (14, 15). The effective lifetime includes loss from phonon scattering during transits between the UDTs and the BS, and from the UDT and BS elements themselves, as well as any losses during qubit release or capture.

We perform two-qubit state tomography for the final joint qubit state generated in Fig. 1D, displaying the absolute value of the density matrix $\rho$ in Fig. 1F. We find a Bell state fidelity $\mathcal{F} = \sqrt{\mathrm{Tr}(\rho_{\mathrm{Bell}}\cdot|\rho|)} = 0.816\pm 0.004$ to the ideal Bell state $\rho_{\mathrm{Bell}}$ , indicating that the phonon beam splitter maintains quantum coherence; the uncertainties here and elsewhere represent one standard deviation. Decay in the principal density matrix elements is consistent with phonon propagation and scattering loss, included via the effective phonon lifetime in the simulations that generate the dashed frames.

# Single phonon interferometry

We further demonstrate the coherence of the BS element by performing a Mach-Zehnderlike interference experiment using a single itinerant phonon. In Fig. 2A, we display the

RESEARCH

Check for updates

$^{1}$ Pritzker School of Molecular Engineering, University of Chicago, Chicago, IL 60637, USA. $^{2}$ Center for Molecular Engineering and Material Science Division, Argonne National Laboratory, Lemont, IL 60439, USA. $^{3}$ Department of Physics, University of Chicago, Chicago, IL 60637, USA. *Corresponding author. Email: anc@uchicago.
edu  
†Present address: Univ. Grenoble Alpes, CEA, Grenoble INP, IRIG, PHELIQS, 38000 Grenoble, France.

Qiao et al., Science 380, 1030-1033 (2023) 9 June 2023

1 of 4

![](dt=2026-02-28/ht=00/e11e8e5fe7f32024dc4e9dd41364989cde8e0b534e73d0a0aa78c7e928c0f95d.jpg)

![](dt=2026-02-28/ht=00/1a17ba9a83dd76dad980f95999ab6d021aea4e636eb7b67d8f3186406a848c1e.jpg)

![](dt=2026-02-28/ht=00/ae92c19f1a746e4e59b48c1d244867e79724019282621b34c5babb0a47e83b05.jpg)

![](dt=2026-02-28/ht=00/9a8b76008e11f9a3e6d15847550561aa7712653ed89551e40c3cb7780274d1d8.jpg)

![](dt=2026-02-28/ht=00/59bab5d836548d44b680dfcb5fb12d391e69945a1c33201f0703258caa5689cc.jpg)

![](dt=2026-02-28/ht=00/e9b43331c872b61dc38cbcf6ccee5a9aa4c5574330896850b2b4ed4b2dde1194.jpg)

![](dt=2026-02-28/ht=00/c82ab632ffc9ee4e5cf034a2f6baf9657f64b4633cb5fe60f7279f5a4c38bc95.jpg)

![](dt=2026-02-28/ht=00/90b97c7aad2cd05f9a795a3f27c673071a9528822fb1f2e80ec8b3af02e39b2b.jpg)

![](dt=2026-02-28/ht=00/5e0782d966e9963a6fa117753e0ecbb7a520a9bb3867cb72edea0976360d3e26.jpg)

![](dt=2026-02-28/ht=00/2698b692bb06384b4aec92d43b26142feed9fc7d508f316c0705d4f1a731f4e1.jpg)

![](dt=2026-02-28/ht=00/ce3b3993fdd8b653792e56c54304e1b67b17760f173fc7948f086f7ba8d02fd2.jpg)

control pulse sequence, in Fig. 2B a schematic representation of the experiment, and in Fig. 2, C to E, the results from this experiment. A single phonon is emitted from $\mathbf{Q}_1$ with the same waveform as in Fig. 1D, and the beam-split phonon is captured by $\mathbf{Q}_1$ and $\mathbf{Q}_2$ . The phase of $\mathbf{Q}_1$ relative to $\mathbf{Q}_2$ is then changed by $\Delta \varphi$ , and the excitations are reemitted through a timed release from each qubit, resulting in a zero-delay

interference at the BS. The phase-dependent interference allows control of which channel receives the output phonon, shown by plotting the excited-state probabilities $P_{\mathrm{Q1,2}}$ , as well as the joint excitation probability $P_{ee}$ , for two choices of phase $\Delta \varphi = 1.46\pi$ and $0.54\pi$ in Fig. 2, C and D; these choices of phase result in routing the maximal output phonon population to $\mathbf{Q}_1$ or $\mathbf{Q}_2$ , respectively. In Fig. 2E, we dis

play the resulting high-visibility interference fringes for the final excitation probabilities $P_{\mathrm{Q1,2}}(t_f)$ as a function of $\Delta \varphi$ . The interference fringe visibility for $Q_1$ is $\mathcal{V}_{\mathrm{Q1}} = 0.806 \pm 0.004$ and for $Q_2$ , $\mathcal{V}_{\mathrm{Q2}} = 0.910 \pm 0.005$ , where the visibilities are defined as $\mathcal{V}_{\mathrm{Q1,2}} = (P_{\mathrm{Q1,2,max}} - P_{\mathrm{Q1,2,min}}) / (P_{\mathrm{Q1,2,max}} + P_{\mathrm{Q1,2,min}})$ . There is a slight misalignment in the interference pattern for $P_{\mathrm{Q1}}$ compared to $P_{\mathrm{Q2}}$ , so that, e.g., the phases

RESEARCH RESEARCH ARTICLE

Qiao et al., Science 380, 1030-1033 (2023) 9 June 2023

2 of 4

for Fig. 2, C and D, do not differ by exactly $\pi$ . This is possibly because the phase difference between reflected and transmitted phonons is not exactly $\pi/2$ [see fig. S1C and discussion in (28)].

# Hong-Ou-Mandel effect

We next study two-phonon interference, shown in Fig. 3, with the schematic process shown in the inset to Fig. 3A. One phonon is emitted by each qubit, timed so that the phonons arrive at the beam splitter with a relative delay $\tau$ ; the beam splitter output is then captured by the two qubits. The probability $P_{11}$ of having co

incident single phonons in the outputs of a lossless beam splitter with reflectivity $\eta$ is (26)

$$
P _ {1 1} (\tau) = 1 - 2 \eta + 2 \eta^ {2} +
$$

$$
\left(2 \eta^ {2} - 2 \eta\right) \left[ \int \phi_ {1} (t - \tau) \phi_ {2} (t) d t \right] ^ {2} \tag {1}
$$

For large delays, the integral is zero, so $P_{\Pi}(\tau \gg \sigma_{1,2})\rightarrow 1 - 2\eta +2\eta^2 = 0.524,$ using the measured beam splitter reflectivity $\eta$ .For zero relative delay and identical time-matched waveforms, the integral is unity, giving $P_{\mathrm{II}}(0) = (1-$ $2\eta)^{2} = 0.048$ . The coincident probability $P_{\mathrm{II}}$ is thus strongly suppressed for zero delay com

![](dt=2026-02-28/ht=00/124135a4b8ee042ad772a6579c1d942f69e91eee1c0b296dbef808515e61f49a.jpg)

![](dt=2026-02-28/ht=00/6d37030289b5182ec77854ab19cc169fa5b6f9c6e3f20b7b404185897d82e236.jpg)

![](dt=2026-02-28/ht=00/2a977b8972528065fe77c806ca3123d032b756dfe9088430b3c7bcb9ed0012b0.jpg)

![](dt=2026-02-28/ht=00/893b18ec09cb2b9719c60b0afaeb098514622e58f0bd847ea46adcdcade76f99.jpg)

pared to large delays, with a theoretical visibility $\mathcal{V}_{\mathrm{th}}\equiv \left[P_{\mathrm{II}}(\tau \gg \sigma_{1,2}) - P_{\mathrm{II}}(0)\right] / P_{\mathrm{II}}(\tau \gg \sigma_{1,2}) = 0.908.$

We cannot directly measure the coincident phonon probability $P_{11}$ , but the qubit joint excitation probability $P_{ee}$ serves as a proxy: The probability for both qubits to be excited is closely related to the probability of having a phonon in each output channel. The two probabilities are not in general simply related, owing in part to the nonzero probability of the two-phonon state in each output channel, as well as the different phonon loss rates and imperfect phonon capture by the qubits in the two channels.

However, we find experimentally that $P_{ee}$ and $P_{11}$ are closely proportional, with $P_{ee} \approx \alpha P_{11}$ with an empirical scale factor $\alpha = 0.265$ . The experimental $P_{ee}$ is also in good agreement with simulations for large and for zero delay (Fig. 3, A to C).

We show the experimental pulse sequence in the inset to Fig. 3B: The qubits are calibrated to emit phonons at $3.925\mathrm{GHz}$ , and the variable couplers are tuned to release phonons with a hyperbolic secant shape, with fit wave packet widths $\sigma_{1,2} = 8.4$ and $8.3~\mathrm{ns}$ respectively, timed so the phonons arrive at the beam splitter with relative time delay $\tau$ . After emission, the qubits are in their ground states $|g\rangle$ . The phonons output from the beam splitter are captured by the two qubits, using the reverse process to emission (14, 15), and the two qubits are measured simultaneously. The capture process is timed so that each qubit catches its own beam-split phonon, with an efficiency very similar to the single-phonon experiment in Fig. 1, D and E.

In Fig. 3, A to C, we show the excited-state probabilities for the two qubits for three different relative delays, $\tau = -150$ ns, $\tau = 0$ ns, and $\tau = 150$ ns, including data taken during the phonon emission and capture processes. In Fig. 3D, we show the corresponding joint excitation probability $P_{ee}$ as a function of delay time $\tau$ , directly extracted from the joint two-qubit state measurements.

When the dela
y $\tau \gg \sigma_{1,2}$ , phonons pass independently through the beam splitter, yielding a joint qubit excitation probability $P_{ee,\max} = P_{Q1} \times P_{Q2} = 0.139 \pm 0.003$ . However, when the two phonons arrive at the beam splitter with zero delay $\tau$ , two-phonon interference suppresses the coincident phonon output state $|\Pi_{\mathrm{ph}}\rangle$ , and similarly suppresses $P_{ee}$ , with a minimum at zero delay of $P_{ee,\min} = 0.0125 \pm 0.0018$ .

The visibility of the dip in $P_{ee}$ , $\mathcal{V}_{\mathrm{exp}} \equiv (P_{ee,\max} - P_{ee,\min}) / P_{ee,\max} = 0.910 \pm 0.013$ , agrees well with the visibility calculated for $P_{11}$ . In the "catch" portion of this experiment, the qubits do not capture the entire phonon signal; this can be seen in Fig. 3A, where the transmitted portion of the phonon from $Q_1$ briefly excites $Q_2$ at $t \sim 670$ ns. As $Q_2$ 's coupler is left on, this excitation is subsequently reemitted. As in this process, the two qubits are detecting the same phonon emitted by $Q_1$ ;

RESEARCH | RESEARCH ARTICLE

Qiao et al., Science 380, 1030-1033 (2023) 9 June 2023

3 of 4

![](dt=2026-02-28/ht=00/b19be2e9f4dcef3ebe42b198552f3ff3306f8848484d7802327fa4fb7d8f3f15.jpg)

![](dt=2026-02-28/ht=00/4583c27be9ed43d2867d31935bc71698b2922e91db8d30e6595399f27ea66c52.jpg)

only one qubit can be excited at a time, so $P_{\mathrm{ee}}$ remains small in this portion of the measurement. There is a similar behavior at $t \sim 750$ ns in Fig. 3C, involving the phonon emitted by $\mathrm{Q}_2$ . For zero delay, the two-phonon signal is only partially captured by the receiving qubits, as each qubit can only catch a single phonon; this is similar to the response of non-number-resolving photon detectors (see fig. S3, showing a variation of this measurement process).

We also study the indistinguishability of the two phonons by varying their relative center frequencies and wave packet widths. While fixing the phonon wave packet widths $\sigma_{1,2} = 16.0$ ns and $16.7\mathrm{ns}$ setting the delay $\tau$ to zero, and fixing $\mathbf{Q}_1$ 's frequency to $f_{1} = 3.925\mathrm{GHz}$ we systematically vary $\mathbf{Q}_2$ 's frequency $f_{2}$ using the qubit flux control. In Fig. 4A, we display the resulting measured joint excitation probability $P_{ee}$ as a function of relative frequency detuning $\Delta f = f_{1} - f_{2}$ .

We observe a clear dip in $P_{ee}$ with frequency $\Delta f$ with visibility $\nu = 0.899\pm 0.013$ . The full width at half maximum of the dip is $9.5\mathrm{MHz}$ , close to the bandwidth of the phonon wave packets. A similar experiment has been done with optical photons (29), with similar results. Next, to study the waveform width dependence, we vary the wave packet width $\sigma_2$ of $\mathbf{Q}_2$ while fixing that of $\mathbf{Q}_1$ to $\sigma_{1} = 8.8$ ns. With a pulse sequence similar to that shown in Fig.

3B, we perform a two-phonon interference experiment, with results displayed in Fig. 4B. The observed variation in visibility is in excellent agreement with the

prediction of Eq. 1. This experiment can also be performed while arbitrarily varying the functional dependence of the phonon waveform; we show an example of this in fig. S5.

# Conclusion and outlook

Together, these experiments demonstrate a quantum-coherent beam splitter, operated with single-phonon sources and detectors, and the ability to control phonon phase through a qubit. These provide all the elements needed to explore the implementation of linear mechanical quantum computing in this system. An outstanding question is whether phonon loss can be made sufficiently small to enable scaling to useful computational systems.

Certainly, gigahertz-frequency bulk acoustic systems, as well as mechanically suspended optomechanical resonators, have much longer phonon lifetimes than we demonstrate here (3, 30), providing some optimism for better performance. It is unlikely that this acoustic approach to linear quantum computing will compete with optical approaches, in which recent implementations have somewhat smaller size elements operating at much higher speeds (31, 32).

However, the straightforward integration of phononic circuits with superconducting qubits might provide important opportunities for hybrid computing systems and will further support the development of phononic communication networks (33-38), possibly integrating computational capabilities.

# REFERENCES AND NOTES

# ACKNOWLEDGMENTS

We thank A. Bienfait and P. Duda for helpful discussions. Funding: Devices and experiments were supported by the Air Force Office of Scientific Research and the Army Research Laboratory. Results are in part based on work supported by the US Department of Energy Office of Science National Quantum Information Science Research Centers. E.D. was supported by LDRD funds from Argonne National Laboratory. This work was partially supported by UChicago's MRSEC (NSF award DMR-2011854) and by the NSF QLCI for HQAN (NSF Award 2016136). We made use of the Pritzker Nanofabrication Facility, which receives support from SHyNE, a node of the National Science Foundation's National Nanotechnology Coordinated Infrastructure (NSF grant no. NNCI ECCS-2025633).

Author contributions: H.Q. designed and fabricated the devices, performed the measurements, and analyzed the data. E.D. contributed to UDT design and measurement techniques. G.A., H.Y., M.H.C., and J.G. provided suggestions with measurement and data analysis. A.N.C. advised on all efforts. All authors contributed to the discussions and production of the manuscript. Competing interests: The authors declare no competing interests. Data and materials availability: All data are available, in both the manuscript and the supplementary materials.

License information: Copyright © 2023 the authors, some rights reserved; exclusive licensee American Association for the Advancement of Science. No claim to original US government works. https://www.sciencemag.org/about/science-licenses-journal-article-reuse

# SUPPLEMENTARY MATERIALS

science.org/doi/10.1126/science.adg8715

Materials and Methods

Supplementary Text

Figs. S1 to S5

Table S1

References (39-43)

Submitted 26 January 2023; accepted 28 April 2023

10.1126/science.adg8715

RESEARCH | RESEARCH ARTICLE

Qiao et al., Science 380, 1030-1033 (2023) 9 June 2023

4 of 4

# Splitting phonons: Building a platform for linear mechanical quantum computing

H. Qiao, . Dumur, G. Andersson, H. Yan, M.-H. Chou, J. Grebel, C. R. Conner, Y. J. Joshi, J. M. Miller, R. G. Povey, X. Wu, and A. N. Cleland

Science, 380 (6649), .

DOI: 10.1126/science.adg8715

# Editor's summary

Phonons are the fundamental quantum vibrations within materials, with individual phonons representing the collective motion of many trillions of atoms. Efforts are underway to determine whether these mechanical vibrations can be developed into a quantum-computing architecture just like their optical cousin, photons. Qiao et al. demonstrate a beam splitter for single phonons and controlled two-phonon interference. Adding to the ability to launch and detect single phonons, a beam splitter now provides the final piece in the toolbox to develop a mechanically based platform for quantum computing. —Ian S. Osborne

# View the article online

https://www.science.org/doi/10.1126/science.adg8715

# Permissions

https://www.science.org/help/reprints-and-permissions

Use of this article is subject to the Terms of service

Science AAAS

Science (ISSN) is published by the American Association for the Advancement of Science. 1200 New York Avenue NW, Washington, DC 20005. The title Science is a registered trademark of AAAS.

Copyright © 2023 The Authors, some rights reserved; exclusive licensee American Association for the Advancement of Science. No
claim to original U.S. Government Works