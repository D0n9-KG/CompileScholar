# Quantum control of surface acoustic-wave phonons

K. J. Satzinger $^{1,2}$ , Y. P. Zhong $^{2}$ , H.-S. Chang $^{2}$ , G. A. Peairs $^{1,2}$ , A. Bienfait $^{2}$ , Ming-Han Chou $^{2,3}$ , A. Y. Cleland $^{2}$ , C. R. Conner $^{2}$ , E. Dumur $^{2,4}$ , J. Grebel $^{2}$ , I. Gutierrez $^{2}$ , B. H. November $^{2}$ , R. G. Povey $^{2,3}$ , S. J. Whiteley $^{2,3}$ , D. D. Awschalom $^{2,4}$ , D. I. Schuster $^{3}$ & A. N. Cleland $^{2,4*}$

One of the hallmarks of quantum physics is the generation of non-classical quantum states and superpositions, which has been demonstrated in several quantum systems, including ions, solid-state qubits and photons. However, only indirect demonstrations of non-classical states have been achieved in mechanical systems, despite the scientific appeal and technical utility of such a capability $^{1,2}$ , including in quantum sensing, computation and communication applications.

This is due in part to the highly linear response of most mechanical systems, which makes quantum operations difficult, as well as their characteristically low frequencies, which hinder access to the quantum ground state $^{3-7}$ . Here we demonstrate full quantum control of the mechanical state of a macroscale mechanical resonator. We strongly couple a surface acoustic-wave $^{8}$ resonator to a superconducting qubit, using the qubit to control and measure quantum states in the mechanical resonator.

We generate a non-classical superposition of the zero-and one-phonon Fock states and map this and other states using Wigner tomography $^{9-14}$ . Such precise, programmable quantum control is essential to a range of applications of surface acoustic waves in the quantum limit, including the coupling of disparate quantum systems $^{15,16}$ .

Linear resonant systems are traditionally challenging to control at the level of single quanta because they are always in the correspondence limit $^{17}$ , where quantum behaviour is indistinguishable from classical motion. The recent advent of engineered quantum devices in the form of qubits has enabled full quantum control over some linear systems, in particular electromagnetic resonators $^{13,14}$ .

A number of experiments have demonstrated that qubits may provide similar control over mechanical degrees of freedom, including qubits coupled to bulk acoustic modes $^{3,7,18}$ , surface acoustic waves (SAWs) $^{19-21}$ and flexural modes in suspended beams $^{22-25}$ . In addition, several experiments have studied entanglement between remote mechanical modes generated via heralding measurements $^{18,26}$ and reservoir engineering $^{27}$ .

Of particular note are experiments in which a superconducting qubit is coupled via a piezoelectric material to a microwave-frequency bulk acoustic mode $^{28}$ , where the ground state can be achieved at moderate cryogenic temperatures; such experiments include controlled vacuum Rabi swaps between the qubit and the mechanical mode $^{3,7}$ . However, the level of quantum control and measurement has been limited by the difficulty in engineering a single mechanical mode with sufficient coupling and quantum state lifetime.

More advanced operations, such as synthesizing arbitrary acoustic quantum states and measuring those states using Wigner tomography, remain a challenge. Here we report an important advance in the level of quantum control of a mechanical device, where we couple a superconducting qubit to a microwave-frequency SAW resonance, demonstrating ground-state operation, vacuum Rabi swaps between the qubit and the acoustic mode, and the synthesis of mechanical Fock states as well as a Fock state superposition.

We map out the Wigner function for these mechanical states using qubit-based Wigner tomography. We note that a similar achievement has been recently reported in an experiment coupling a superconducting qubit to a bulk acoustic mode $^{29}$ .

The device that we use for this experiment is shown in Fig. 1. The superconducting qubit is a frequency-tunable planar transmon $^{30,31}$ , connected to the SAW device through a tunable inductor network that provides electronic control $^{32}$ of the coupling strength $g_{0}$ (see Supplementary Information). Qubit rotations about the $X$ and $Y$ axes in the Bloch sphere representation are performed using pulses on the microwave (XY) line, and $Z$ -axis rotations are achieved by application of a flux bias current on the frequency-control $(Z)$ line.

We measure the qubit state using a dispensively coupled readout resonator (see Supplementary Information). The superconducting qubit is fabricated on a sapphire substrate with standard techniques (see Supplementary Information). The SAW resonator is fabricated separately on a lithium niobate substrate, a strong piezoelectric material commonly used for SAW devices $^{8}$ . The SAW resonator comprises an interdigital transducer placed between two Bragg mirrors, designed to support a single SAW resonance in the mirror stop band $^{8}$ (see Supplementary Information).

The SAW wavelength $\lambda$ is set by the period of the metal lines that constitute the resonator; here, $\lambda = 1\mu \mathrm{m}$ , which corresponds to a frequency of $4.0~\mathrm{GHz}$ . At the experiment temperature, about $10\mathrm{mK}$ , both the SAWs and the qubit should be in their quantum ground states.

The electromechanical properties of the SAW resonator are modelled using an equivalent electrical circuit with a complex, frequency-dependent acoustic admittance $^{8}$ $Y_{\mathrm{a}}(\omega)$ connected in parallel with an interdigital capacitance $C_{\mathrm{t}} = 0.75\mathrm{pF}$ . The admittance includes the complete response of the SAW transducer and the interaction of the SAW with the mirrors. The strong electromechanical coupling coefficient of lithium niobate makes it feasible to strongly couple the SAW resonance to a standard transmon-style qubit (see Supplementary Information).

The separate qubit and SAW-resonator chips are connected together in a flip-chip assembly, in which the lithium niobate chip is inverted, aligned and affixed to the sapphire chip, and are separated vertically by about $7\mu \mathrm{m}$ (see Supplementary Information). Coupling between the two chips is achieved using two overlaid planar inductors, one on each chip. The coupling strength is controlled using a radio-frequency superconducting quantum interference device (SQUID) tunable coupler $^{32}$ , where an externally controlled flux bias $\varPhi$ controls the path of the qubit current.

We note that the flip-chip technique used here enables a wide range of future hybrid combinations of different substrate types with superconducting or other types of qubits; as shown below, the coherence of the qubit in this experiment was not affected by this approach.

We use qubit measurements to evaluate the SAW resonator. The qubit itself has a lifetime of $T_{1} \approx 20~\mu \mathrm{s}$ and a Ramsey lifetime of $T_{2,\mathrm{Ramsey}} \approx 2\mu \mathrm{s}$ over the frequency range $3.5 - 4.5\mathrm{GHz}$ , measured with the coupling $g_{0}$ set to zero (see Supplementary Information). Adjusting $g_{0}$ away from zero shortens the qubit lifetime and makes it strongly frequency-dependent, as the transducer converts electromagnetic energy from the qubit into acoustic waves. In Fig.

2, we demonstrate this with $|g_0| / 2\pi$ set to $2.3 \pm 0.1 \mathrm{MHz}$ (all uncertainties are one standard deviation), where acoustic loss is the dominant decay channel for the qubit. We measure the qubit lifetime $T_{1}$ as a function of qubit frequency, $\omega_{ge} / 2\pi$ , and use it to obtain the quality factor $Q = \omega_{ge}T_{1}$ and

LETTER

https://doi.org/10.1038/s41586-018-0719-5

$^{1}$ Department of Physics, University of California, Santa Barbara, CA, USA. $^{2}$ Institute for Molecular Engineering, University of Chicago, Chicago, IL, USA. $^{3}$ Department of Physics, University of Chicago, Chicago, IL, USA. $^{4}$ Inst
itute for Molecular Engineering and Materials Science Division, Argonne National Laboratory, Argonne, Lemont, IL, USA. *e-mail: anc@uchicago.edu

© 2018 Springer Nature Limited. All rights reserved

NATURE|www.nature.com/nature

![](dt=2026-03-01/ht=18/b82ce5ea100d57d469ff57f5da87c9b05906bbc0fc89cc8cbdd67d9dc99a7bdb.jpg)

![](dt=2026-03-01/ht=18/748f44f11de23edc194ed4c1252ada070b249bb32baa3e455d0c3b3cfac683dc.jpg)

![](dt=2026-03-01/ht=18/1898d40fad8cc6593ae902ce45940dc6b3db51686ab316559329950f6d1ee580.jpg)

![](dt=2026-03-01/ht=18/477fb9a2fb7347cb0a3f14c705ad115dfb9386698c2db42d18e6ce3a31c088c0.jpg)

the corresponding loss $1 / Q$ . We compare our measurements to the results of a numerical model based on the SAW resonator design with parameters fine-tuned to reproduce the frequency response observed in the qubit loss (see Supplementary Information). The SAW transducer itself can efficiently emit phonons over a wide range of frequencies, roughly from 3.8 GHz to 4.1 GHz, owing to its small number of finger pairs (20 pairs). The SAW mirror reflects acoustic waves efficiently in the mirror stop band from 3.96 GHz to 4.04 GHz.

The resultant interference frustrates the transducer emission except when a resonance condition is met, in this case at the single SAW resonance frequency of $\omega_{\mathrm{r}} / 2\pi = 3.985$ GHz. The resonator admittance near that resonance can be approximated by an equivalent resonant electrical circuit, which constitutes the Butterworth-van Dyke model. Outside the mirror stop band, the mirror reflection decreases rapidly, and the transducer is free to emit travelling phonons.

The qubit sees this as increased loss, especially from 3.85 GHz to 3.90 GHz, where the transducer is most efficient. The ripples in the out-of-band mirror reflection arise from the finite extent of each mirror (500 lines). These features are clearly

![](dt=2026-03-01/ht=18/ad8813d1ab63d6f4265fdf8dd7e419fee06e570fdbf792fe2a1fdbc67d0e24ce.jpg)

![](dt=2026-03-01/ht=18/2435683d1efd32c0cfa2d86b339b7cfb5f4ff325fc504d1e3c22ad3842159be0.jpg)

![](dt=2026-03-01/ht=18/5ca7a6c85bf16780c3cd4b45c423a264c74acbc57c6a2adff47f5272131c3592.jpg)

displayed in the measured qubit loss. The qubit also weakly couples to unidentified resonances near $3.8\mathrm{GHz}$ . The SAW resonance at $3.985\mathrm{GHz}$ can resonantly and rapidly exchange energy with the qubit. In subsequent experiments, we avoid unwanted qubit loss by usually keeping the coupling small and only increasing it when deliberately interacting with the SAW resonance.

We now focus on the interaction between the single SAW resonance and the qubit. In Fig. 3a, we illustrate the full range of qubit coupling to the resonance, determined using spectroscopic measurements of the qubit. We observe a maximum coupling of $|g_0| / (2\pi) = 7.3 \pm 0.1 \mathrm{MHz}$ , which is equal to half of the avoided-crossing splitting. The ratio of the maximum to the minimum coupling strength is measured to be at least 300 (see Supplementary Information).

Figure 3c shows time-domain Rabi swapping of a single excitation between the qubit and the mechanical mode, which represents a photon-phonon exchange in each half-oscillation. A resonant swap operation is executed by setting the qubit frequency to $\omega_{\mathrm{r}}$ and turning on the coupling for approximately 37 ns. The number and amplitude of the swaps is primarily limited by the resonator lifetime, $T_{\mathrm{lr}}$ .

We show the characterization results for the single-phonon properties of the resonator in Fig. 3c. We prepare a quantum state in the qubit, swap it into the resonator, wait for a delay time $t$ , swap the state back into the qubit, and measure the qubit. The decay of the phonon is consistent with an energy lifetime of $T_{1\mathrm{r}} = 148 \pm 1$ ns and a dephasing time of $T_{2\mathrm{r}} = 293 \pm 1$ ns, where the ratio $T_{2\mathrm{r}} / T_{1\mathrm{r}} \approx 2$ is consistent with little to no additional phase decoherence, as expected for a harmonic oscillator.

The $T_{2\mathrm{r}}$ experiment involves generating a quantum superposition of the resonator phonon Fock states $|0\rangle$ and $|1\rangle$ by performing a Rabi swap from a qubit in the state $(|g\rangle - i|e\rangle) / \sqrt{2}$ , where $|g\rangle$ and $|e\rangle$ are the qubit ground and excited states, respectively. The probabilities oscillate at the idle detuning frequency $\Delta / 2\pi = 53$ MHz, exhibiting interference between the resonator state and the qubit tomography pulses.

RESEARCH LETTER

NATURE|www.nature.com/nature

© 2018 Springer Nature Limited. All rights reserved.

![](dt=2026-03-01/ht=18/25da23629264a037d014b2594ce66e7dae2da5ef1d367715a5fc7b0adc35a19a.jpg)

![](dt=2026-03-01/ht=18/3df44efb8df008377a344323c2f446dad0df00cf51603d640f55a2469fe11606.jpg)

![](dt=2026-03-01/ht=18/b3961aa8d2a8aec040986f6d67070a46bf6146702bcb53e4d56f8068b2434589.jpg)

We attempt to create the higher Fock state $|2\rangle$ in the SAW resonator by exciting the qubit and swapping its excitation into the resonator two times. We show the result in Fig. 4. The experiment is limited by the resonator lifetime $T_{\mathrm{1p}}$ which is comparable to the duration of the pulse sequence used to generate $|2\rangle$ , about $100\mathrm{ns}$ . We do observe higher-frequency oscillations in the initial interaction, as expected.

The experimental result is in excellent agreement with a numerical master-equation model, which is fitted to the experiment by adjusting the initial qubit and resonator states. The resonator state is closest to $|2\rangle$ after an interaction time of $\tau = 26$ ns. At that time, the resonator state calculated by the model is a statistical mixture of $47.3\%$ $|2\rangle$ , $38.2\%$ $|1\rangle$ and $14.5\%$ $|0\rangle$ , with the unwanted lower states appearing owing to decay during state preparation.

We now characterize the quantum state of the resonator in greater detail. Verifying that the resonator is indeed in its ground state is an

![](dt=2026-03-01/ht=18/6c1b046986d09d7a6b6f85421990cbf3ad0f2cd128456bf4032b6861c401e07d.jpg)

important step in evaluating its quantum behaviour. We examine the residual thermal populations in the qubit and resonator excited states, $|e\rangle$ and $|1\rangle$ , respectively, using a Rabi population measurement technique[7,33] (see Supplementary Information). Driven transitions between $|e\rangle$ and the second excited qubit state, $|f\rangle$ , are used to quantify the $|e\rangle$ population by measuring the amplitudes of Rabi-like oscillations. The experimental results are shown in Fig. 5a, where we vary the amplitude of a microwave pulse that drives $e - f$ transitions.

In the left panel, we show the result of probing the ground-state population of the qubit; the large-amplitude oscillations show near-unity initial ground-state population. In the right panel, we probe the excited-state population, which is much smaller. We calculate the excited-state population from the amplitudes of these oscillations (see Supplementary Information). When performing the
experiment on the qubit alone, we observe an excited-state population of $0.0169 \pm 0.0002$ .

To assess the thermal population of the resonator, we first execute a resonant-swap operation, and then we conduct the experiment again. The swap exchanges the small excited-state populations in the resonator and the qubit. In this case, we observe an excited-state population of $0.0049 \pm 0.0002$ , which we interpret as an upper bound on the excited-state population of the resonator[7].

The level of control achievable in this experiment allows us to controllably generate the resonator states $|0\rangle, |1\rangle$ , $(|0\rangle + |1\rangle) / \sqrt{2}$ and, to a lesser extent, $|2\rangle$ . We prepare these resonator states deterministically, by exciting the qubit and transferring energy into the resonator with resonant swaps. We use Wigner tomography to determine the fidelities of these quantum states<sup>13</sup> (see Supplementary Information), examining the three lowest-energy states in detail.

Following state preparation, we measure the Wigner function $W(\alpha)$ of the resonator by using the qubit to measure the parity of the resonator states at different complex displacements $\alpha$ in the resonator phase space (see Supplementary Information). The required displacements $\alpha$ are created by driving the resonator with a resonant Gaussian microwave pulse applied to a control line (see Fig. 1c). During the pulse, the coupling is turned off, and the qubit is detuned above the resonator by $\Delta / 2\pi = 400\mathrm{MHz}$ .

With the qubit initially in its ground state $|g\rangle$ , we allow the qubit and resonator to resonantly interact for a time $\tau$ , and then we measure the qubit. An example is shown in Fig. 5b. The plot of the qubit state as a function of delay $\tau$ contains information about the displaced-resonator state. We fit the experimental results with a numerical master-equation model to deduce the phonon number $(n)$ distribution of the displaced resonator, $P_{n}$ . We then calculate the Wigner function, which is proportional to the phonon number parity[13] (see Supplementary Information).

We repeat the experiment for many values of $\alpha$ to map out the Wigner function. The results are displayed in Fig. 5d, along with the prediction of the numerical model using the same pulse sequence. The value of each pixel is determined independently. We then convert each experimental $W(\alpha)$ into a density matrix $\rho$ (see Supplementary Information).

LETTER RESEARCH

NATURE|www.nature.com/nature

© 2018 Springer Nature Limited. All rights reserved.

![](dt=2026-03-01/ht=18/2591c11de79c5725c7399a60b42989f029509b3fa5a4b97e33abd1294adfa834.jpg)

![](dt=2026-03-01/ht=18/e5f168dd9e4787fad5a21fdf70a9d8d81ccdb8c942b364cb5e5c9cb8a116672d.jpg)

![](dt=2026-03-01/ht=18/34be47c1b74e0f39cc0302f77724f3c41de79559d6b1d1f6d1d287caeed7acf9.jpg)

![](dt=2026-03-01/ht=18/f9667a6c20c0430fe9df2931f7912e6ecc334c32cfed1c180d200e5c297a3f0e.jpg)

![](dt=2026-03-01/ht=18/2fdcf819e7086d8ac595baf53c3f987dbeaab2d5d81717b3a773debd4eb42e31.jpg)

From the density matrices, we calculate the quantum state fidelities to the ideal states $|\psi \rangle$ , $F = \sqrt{\langle\psi|\rho|\psi\rangle}$ . We obtain $F = 0.985 \pm 0.005$ for $|0\rangle$ , $F = 0.858 \pm 0.007$ for $|0\rangle$ and $F = 0.945 \pm 0.006$ for $(|0\rangle + |1\rangle) / \sqrt{2}$ . The numerical model predicts similar fidelities: $F = 0.998$ , 0.879 and 0.962, respectively. These experiments would benefit from a longer phonon lifetime $T_{\mathrm{lr}}$ and larger coupling strength.

In conclusion, we demonstrate high-fidelity, on-demand synthesis of quantum states in a macroscale mechanical resonator and characterize them with Wigner tomography. The primary limitation in these experiments is the phonon lifetime in combination with the maximum coupling strength. These could be improved substantially in future work, for example, with design and material changes in the mechanical

resonator and adjustments to the coupling circuit. Our demonstration involves a hybrid architecture incorporating a high-performance qubit with strong tunable coupling to SAWs. This scalable platform holds promise for future quantum acoustics experiments coupling stationary qubits to 'flying' qubits based on phonons. The technologies demonstrated here may also enable a wide range of experiments coupling superconducting circuits to diverse quantum systems, such as semiconductor spin systems.

# Data availability

The datasets supporting this work are available from the corresponding author on request.

Received: 19 April 2018; Accepted: 10 September 2018; Published online: 21 November 2018

RESEARCH LETTER

NATURE|www.nature.com/nature

© 2018 Springer Nature Limited. All rights reserved.

33. Geerlings, K. et al. Demonstrating a driven reset protocol for a superconducting qubit. Phys. Rev. Lett. 110, 120501 (2013).

Acknowledgements We thank P. J. Duda, A. Dunsworth and D. Sank for discussions. Devices and experiments were supported by the Air Force Office of Scientific Research, the Army Research Laboratory and the Department of Energy (DOE). K.J.S. and S.J.W. were supported by the US National Science Foundation (NSF) GRFP (NSF DGE-1144085); E.D. was supported by LDRD funds from Argonne National Laboratory; A.N.C. and D.D.A. were supported by the DOE, Office of Basic Energy Sciences; and D.I.S. acknowledges support from the David and Lucile Packard Foundation.

This work was partially supported by the UChicago MRSEC (NSF DMR-1420709) and made use of the Pritzker Nanofabrication Facility, which receives support from SHyNE, a node of the NSF's National Nanotechnology Coordinated Infrastructure (NSF NNCI-1542205).

Reviewer information Nature thanks S. Deleglise and the other anonymous reviewer(s) for their contribution to the peer review of this work.

Author contributions K.J.S. designed and fabricated the devices. K.J.S., H.-S.C., J.G., A.Y.C. and S.J.W. developed the fabrication processes. G.A.P., E.D. and A.N.C. contributed to device design. K.J.S. performed the experiments and analysed the data with assistance from Y.P.Z., A.B. and E.D. Assistance was provided by I.G. and B.H.N., A.N.C., D.I.S. and D.D.A. advised on all efforts. All authors contributed to discussions and the production of the manuscript.

Competing interests The authors declare no competing interests.

# Additional information

Supplementary information is available for this paper at https://doi.org/10.1038/s41586-018-0719-5.

Reprints and permissions information is available at http://www.nature.com/reprints.

Correspondence and requests for materials should be addressed to A.N.C.  
Publisher's note: Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

LETTER RESEARCH

© 2018 Springer Nature Limited. All rights reserved.

NATURE|www.nature.com/nature