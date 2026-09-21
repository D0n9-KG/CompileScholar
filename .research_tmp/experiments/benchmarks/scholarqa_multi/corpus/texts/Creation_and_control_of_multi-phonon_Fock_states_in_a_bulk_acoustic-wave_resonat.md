# Creation and control of multi-phonon Fock states in a bulk acoustic wave resonator

Yiwen Chu $^{1*}$ , Prashanta Kharel $^{1*}$ , Taekwan Yoon $^{1}$ , Luigi Frunzio $^{1}$ , Peter T. Rakich $^{1}$ , & Robert J. Schoelkopf $^{1}$

$^{1}$ Department of Applied Physics, Yale University, New Haven, Connecticut 06511, USA and Yale Quantum Institute, Yale University, New Haven, Connecticut 06520, USA

Quantum states of mechanical motion can be important resources for quantum information, metrology, and studies of fundamental physics. Recent demonstrations of superconducting qubits coupled to acoustic resonators have opened up the possibility of performing quantum operations on macroscopic motional modes $^{1-3}$ , which can act as long-lived quantum memories or transducers. In addition, they can potentially be used to test for novel decoherence mechanisms in macroscopic objects and other modifications to standard quantum theory $^{4,5}$ . Many of these applications call for the ability to create and characterize complex quantum states, putting demanding requirements on the speed of quantum operations and the coherence of the mechanical mode. In this work, we demonstrate the controlled generation of multi-phonon Fock states in a macroscopic bulk-acoustic wave resonator. We also perform Wigner tomography and state reconstruction to highlight the quantum nature of the prepared states $^{6}$ . These demonstrations are made possible by the long coherence times of our acoustic resonator and our ability to selectively couple to individual phonon modes. Our work shows that circuit quantum acousto-dynamics (circuit QAD) $^{7}$ enables sophisticated

# quantum control of macroscopic mechanical objects and opens the door to using acoustic modes as novel quantum resources.

Light and sound are two familiar examples of wave phenomena in the classical world. By now, the field of quantum optics has extensively demonstrated the particle nature of light in quantum mechanics through the study of single photons and other non-Gaussian electromagnetic states. The concept of particles of sound, or phonons, is used widely in solid state physics. However, the ability to create states of individual phonons has only been demonstrated in a few instances $^{1,3,8}$ , while complete quantum tomography of such states has only been achieved in a single trapped ion $^{6}$ . This disparity between electromagnetic and acoustic degrees of freedom is largely because sound propagates inside the complex and potentially lossy environment of a massive material rather than vacuum. As a result, an open question remains: Is it feasible to control and measure complex quantum states in the motion of a macroscopic solid state object, or what we usually think of as sound, analogously to what has been done with light?

The relatively new field of quantum acoustics is attempting to answer this question using a variety of optomechanical and electromechanical systems $^{1,2,7,9-13}$ , and one particularly promising approach within quantum acoustics is circuit QAD $^{1,2,7,13}$ . In analogy to circuit quantum electrodynamics (circuit QED), circuit QAD uses superconducting quantum circuits that operate at microwave frequencies to manipulate and measure mechanical resonators. Circuit QAD takes advantage of the strong interactions between mechanics and electromagnetism enabled by, for example, piezoelectricity. It also incorporates the non-linearity provided by the Josephson junction, which is

a crucial ingredient for creating non-Gaussian states of motion. In turn, the ability to create these states make mechanical resonators useful as resources in quantum circuits, offering capabilities beyond those of electromagnetic resonators. For example, mechanical transduction is a promising method for transferring quantum information between microwave circuits and other systems such as optical light or spin qubits $^{14,15}$ . Due to the difference between the speeds of sound and light, an acoustic resonator is much more compact and well isolated than an electromagnetic one at the same frequency and provides many more independent modes that are individually addressable by a superconducting qubit. Such an architecture is desirable for simulating many-body quantum systems $^{2,16}$ and provides a highly hardware efficient way of storing, protecting, and manipulating quantum information using bosonic encodings $^{17,18}$ . These examples show that, by repurposing the toolbox of circuit QED through the similarities between light and sound, circuit QAD allows us to make use of the important differences between these quantum degrees of freedom. However, in order to access this toolbox, we first need to demonstrate that a circuit QAD system can be engineered to have the necessary mode structure, strong enough interactions, and sufficient quantum coherence to create and characterize quantum states of motion.

In this work, we experimentally prepare and perform full quantum tomography on Fock states of phonons and their superpositions inside a high-overtone bulk acoustic wave resonator (HBAR). These demonstrations are enabled by a robust new flip-chip device geometry that couples a superconducting transmon qubit to the HBAR. This geometry allows us to separately optimize the design of the acoustic resonator and qubit to extend phonon coherence while enhancing the selectivity of the coupling to a single mode. The combination of these improvements leads to a device

that is deeper in the strong coupling regime of circuit QAD, which is necessary for the generation and manipulation of more complex quantum states. We note that a similar demonstration using a superconducting qubit and surface acoustic waves was recently reported $^{19}$ .

We now motivate and describe the design of our circuit QAD system in more detail. Figure 1a shows a schematic of our device, which we call the $\hbar$ BAR from now on. The first important difference from our previous device is the flip-chip geometry, where the qubit and acoustic resonator are now on separate sapphire chips $^{19}$ . This simplifies the fabrication procedure and increases the yield of successful devices (see Supplementary Information), while allowing for qubits and acoustic resonators to be individually tested before assembly. Second, the $\hbar$ BAR now incorporates a plano-convex acoustic resonator that is fabricated using a simple, robust recipe and supports stable, transversely confined acoustic modes (see Supplementary information). Since the measured acoustic lifetime in the previous unstable resonator geometry $^{3}$ was consistent with being limited by diffraction loss, this modification to our device could significantly improve the phonon coherence. Another important requirement is the ability to selectively couple the qubit to a single acoustic mode. This is partly achieved by the plano-convex resonator design, which allows us to control the frequency spacing between transverse modes. To further increase mode selectivity, the third improvement is the addition of an optimized transduction electrode to the qubit. The electrode was designed to match the strain profile of the fundamental Gaussian transverse mode of the acoustic resonator (see Supplementary information). We point out that even though the acoustic resonator is not in physical contact with the electrode, the electric field of the qubit extends across the gap between the two chips and through the AlN film, thus allowing for piezeoelectric transduction.

We now experimentally show that the new design does indeed lead to improvements in the electro-mechanical coupling, acoustic mode spectrum, and coherence of our device. As in our previous work, the $\hbar$ BAR is measured using a standard circuit QED setup that allows for flux tuning of the qubit frequency. Figure 1b shows qubit spectroscopy near the $l = l_{1}$ and $m$ , $n = 0$ mode of the $\hbar$ BAR, which reveals a single distinct anticrossing feature (Figure 1b). Here $l$ is the longitudinal mode number, and $m$ , $n$ are the mode numbers of the Hermite-Gaussian-like transverse modes. $l_{1} \sim 466$ corresponds to the highest frequency longitudinal mode fully within the tunable range of the qubit, as indicated in Figure 2, where we investigate the mode structure of the $\hbar$ BAR over several longitudinal free-spectral ranges. Figure 2a shows time dynamics of the qubit-phonon interaction, which reveals vacuum Rabi oscillations every $\nu_{\mathrm{FSR}} = 13.5$ MHz as we tune the qubit frequency, each corresponding to an anticrossing feature similar to the one shown in Figure 1b. The Fourier transform of the data in Figure 2a is shown in Figure 2b and gives a qubit-phonon coupling rate of $g_{0} = 2\pi \times (350 \pm 3)$ kHz. In addition to the dominant set of oscillations corresponding to the $m$ , $n = 0$ Gaussian modes, there are clear signatures of other acoustic modes visible in Figures 2a and b, which simulations indicate correspond to higher order transverse modes (see Supplementary Information). However, the closest observable higher order mode is $\sim 1$ MHz away from the $m$ , $n = 0$ mode and about ten times less strongly coupled to the qubit, while all others are at least five times less strongly coupled. From now on, we use only the longitudinal mode number to represent the $m$ , $n = 0$ modes. These results indicate that the $\hbar$ BAR is a good approximation of a system in which the qubit can be tuned to interact with a single acoustic mode at a time.

We demonstrate improvements in the coherence of our system by performing quantum oper-

ations on the phonon mode using the qubit. Using techniques described in our previous work $^{3}$ , we find that the phonon mode has a $T_{1}$ of $(64 \pm 2)$ $\mu$ s, a Ramsey $T_{2}$ of $(38 \pm 2)$ $\mu$ s, and an echo $T_{2}$ of $(45 \pm 2)$ $\mu$ s. On other devices, we measured that the phonon $T_{1}$ can be as long as $(113 \pm 4)$ $\mu$ s. These coherence times are now comparable to that of state-of-the-art superconducting qubits and suggest that the plano-convex resonator design does indeed support much longer lived phonons. The qubit in this device has a $T_{1}$ of $(7 \pm 1)$ $\mu$ s, which is similar to our previous device. As will be discussed later, we believe these device parameters can be further improved through modifications of the materials, fabrication procedure, and device geometry.

The improvements presented above allow us to perform quantum operations on the phonon mode with a new level of sophistication, which we now illustrate by creating and measuring multi-phonon Fock states. We use a procedure for Fock state preparation that has previously only been demonstrated in electromagnetic systems $^{20}$ (Figure 3a). The experiment begins with the qubit set to a frequency $\nu_0$ that is $\delta = -5$ MHz detuned from the target $l_1$ phonon mode at frequency $\nu_1$ . The qubit ideally starts out in the ground state $|g\rangle$ , but in reality has a thermal population of $4 - 8\%$ in the excited state $|e\rangle$ . The phonon modes, on the other hand, were shown to be colder $^3$ . Therefore we first perform a swap operation between the qubit and the $l_2$ mode with frequency $\nu_2$ . This procedure effectively uses an additional acoustic mode to cool the qubit to an excited state population of $\sim 2\%$ . The qubit is then excited with a $\pi$ pulse and brought into resonance with the $l_1$ mode to swap the energy into the acoustic resonator. This is repeated $N$ times to climb up the Fock state ladder, ideally resulting in a state of $N$ phonons, which is then probed by bringing the qubit and phonon on resonance for a variable time $t$ and measuring the final qubit state. We

note that this measurement procedure gives the total population in the qubit excited state subspace of the joint system and traces over the resonator state. The resulting time dynamics of $p_{e,N}(t)$ for up to N = 7 are shown in Figure 3b. In Figure 3c, we plot the Fourier transform of the data in Figure 3b. As expected, we observe oscillations with a dominant frequency of $2g_{N} = 2\sqrt{N}g_{0}$ , corresponding to the rate of energy exchange between the $|g, N\rangle$ and $|e, N - 1\rangle$ states.

In order to more quantitatively characterize the states we have created, we extract the population in each phonon Fock state n after performing a N phonon preparation. We do this by first simulating the expected time traces $p_{e,n}(t)$ if the phonon mode is prepared in an ideal Fock state ranging from n = 1 to $n_{max} = 14$ . The independently measured value of $g_{0}$ , along with the qubit and phonon decay and dephasing rates, are used in the simulations. Then, the experimental data for each N (Figure 3b) are fitted to a weighted sum of the form

$$
p _ {e, N} (t) = \sum_ {n = 1} ^ {n _ {\max}} p _ {n, N} p _ {e, n} (t), \tag {1}
$$

where $p_{n,N}$ is then the population in $|g,n\rangle$ after performing a N phonon preparation. The fit for each N is subject to the constraints $p_{n,N} \leq 1 \forall n$ and $\sum_{n=1}^{n_{\max}} p_{n,N} \leq 1$ . Finally, the population in the zero phonon state is calculated as $p_{0,N} = 1 - \sum_{n=1}^{n_{\max}} p_{n,N}$ . Ideally, $p_{n,N} = \delta_{n,N}$ . As shown in Figure 3d, we observe that the resulting distribution of populations for each experiment is indeed peaked at n = N. However, the population in the nominally prepared state decreases with increasing N. We find that $p_{1,1} = 0.86$ , which is consistent with a simple estimate taking into account the energy decay from the one excitation manifold during a swap operation, which is dominated by the qubit decay rate, and the imperfect preparation of the qubit in $|g\rangle$ . For larger N's, the state preparation may be affected by additional effects such as off-resonant driving of the phonon mode during the

qubit $\pi$ pulses, which could lead to excess population in the n > N states. We also found that the largest source of potential error in extracting $p_{n,N}$ comes from uncertainty in the system parameters that are used in simulating $p_{e,n}(t)$ . In particular, slight drifts of the qubit frequency can result in a mismatch between the value of $2g_{0}$ used in the simulations and the actual oscillation frequency of the vacuum Rabi data. An estimate of the effect of such miscalibrations are given by the errorbars in Figure 3d.

We now build upon our ability to extract the phonon number distribution to perform full Wigner tomography and explore the quantum nature of the prepared mechanical state. As in previous experiments in circuit QED and trapped ions $^{6,21}$ , we make use of the definition $^{22}$

$$
P (\alpha) = \mathrm{Tr} [ \hat {D} (- \alpha) \rho \hat {D} (\alpha) \hat {P} ] = \frac {\pi}{2} W (\alpha). (2)
$$

Here $P(\alpha)$ and $W(\alpha)$ are the values of the displaced parity and Wigner functions at a phase space amplitude $\alpha$ , respectively, $\rho$ is the prepared state, and $\hat{P}$ is the parity operator. From now on we will plot the values of $P(\alpha)$ for clarity, but use the terms displaced parity and Wigner function interchangeably. The resonator displacement $\hat{D}(\alpha)$ is implemented by a microwave pulse at the phonon frequency while the qubit is detuned at $\nu_0$ . Under these conditions, the phonon mode is still coupled to the microwave drive port, in part due to its hybridization with the qubit. To verify this and calibrate our displacement amplitudes, we first apply a Gaussian phonon drive pulse of varying amplitude $\alpha$ with a 1 $\mu$ s RMS width and truncated to 4 $\mu$ s total length. We then measure the subsequent Fock state populations $p_{n,|0\rangle}(\alpha)$ and check that they agree well with the expected Poisson distributions up to an overall scaling between the amplitudes of the applied drive and the actual displacement (see Supplementary Information). We can then calculate the displaced

parity for the vacuum state $|0\rangle$ using $P_{|0\rangle}(\alpha) = \sum_{n}(-1)^{n}p_{n,|0\rangle}(\alpha)$ . Similarly, we can measure the displaced parity $P_{\rho}(\alpha)$ for an arbitrary state $\rho$ by adding a phonon drive pulse between state preparation and measurement.

In Figure 4, we present the results of Wigner state tomography on the nominally prepared states $|1\rangle$ , $(|0\rangle + |1\rangle) / \sqrt{2}$ , and $|2\rangle$ . The $(|0\rangle + |1\rangle) / \sqrt{2}$ state was prepared by performing a $\pi/2$ pulse on the qubit and followed by a swap operation with the phonon mode. From the measured data shown in Figures 4a, b, and c, we can reconstruct the measured state using a maximum likelihood method $^{18}$ (see Supplementary Information). The Wigner functions of the reconstructed states are presented in Figures 4d, e, and f. Figures 4g, h, and i show that the reconstructed parities agree well with the raw data. The negativity of the Wigner functions clearly demonstrate the quantum nature of the states. From the reconstructed density matrices, we find that the fidelities of the prepared states to the target states are $F_{|1\rangle} = 0.87 \pm 0.01$ , $F_{(|0\rangle + |1\rangle) / \sqrt{2}} = 0.94 \pm 0.01$ , and $F_{|2\rangle} = 0.78 \pm 0.02$ . The infidelity for all three states are dominated by excess population in the lower number Fock states, which is an expected consequence of energy decay during state preparation and measurement (see Supplementary Information).

These results show that the quantum state of motion in a macroscopic mechanical resonator can be prepared, controlled, and fully characterized in a circuit QAD device. The demonstration of even more complex quantum states should be possible with further improvements of the device performance. Currently, the dominant source of loss is the qubit, and we found that its $T_{1}$ is higher when the resonator chip is either not present or rotated by $180^{\circ}$ relative to the qubit chip. This

indicates that the qubit lifetime may be limited by loss due to the AlN, which could be mitigated by using a different piezoelectric material or optimizing the device geometry to minimize the electric field in the AlN that does not contribute to transduction. The current limitations on the phonon coherence also require further investigation. The energy loss is likely to be dominated by surface roughness or imperfections in the fabricated geometry, while additional dephasing could result from thermal excitations and frequency fluctuations of the detuned qubit $^{23}$ . In addition, we can more carefully characterize the final flip-chip geometry, such as the spacing and alignment between the chips. The assembly process can then be modified accordingly, potentially leading to further improvements in the coupling and mode selectivity.

The next generation of devices could give us access to even more sophisticated methods for quantum control of the acoustic resonator. Our current device is close to being able to reach the strong dispersive regime where circuit QED systems currently operate, which would allow for quantum non-demolition measurements of phonon numbers $^{24}$ and more sophisticated techniques for generating arbitrary quantum states of harmonic resonators $^{17,25}$ . Furthermore, our technique for cooling the qubit already takes advantage of the multimode nature of the acoustic resonator. Future experiments would, for example, demonstrate qubit-mediated interactions between multiple modes and the creation of multipartite entangled states of mechanical motion $^{11,12}$ . Recent efforts in improving the efficiency of electromechanical and optomechanical transduction with mechanical resonators could enable conversion of quantum information between the microwave and optical domains $^{14,26}$ . Beyond the use of acoustic resonators as resources for quantum information, the creation of increasingly complex quantum states in highly coherent mechanical resonators can

provide insight into the question of whether quantum superpositions of massive objects are suppressed due to mechanisms other than environmental decoherence $^{4,27}$ . In addition, the ability to perform quantum control on our large effective mass, high frequency, and low thermal occupation mechanical system may put new bounds on modifications to quantum mechanics at small length scales $^{28,29}$ . These examples suggest that the wide range of quantum acoustics demonstrations that may soon be possible with $\hbar BAR$ will give rise to new quantum technologies while furthering our understanding of fundamental physics.

1. O'Connell, A. D. et al. Quantum ground state and single-phonon control of a mechanical resonator. Nature 464, 697–703 (2010).   
2. Moores, B. A., Sletten, L. R., Viennot, J. J. & Lehnert, K. W. Cavity quantum acoustic device in the multimode strong coupling regime. Phys. Rev. Lett. 120, 227701 (2018).   
3. Chu, Y. et al. Quantum acoustics with superconducting qubits. Science 358, 199–202 (2017).   
4. Arndt, M. & Hornberger, K. Testing the limits of quantum mechanical superpositions. Nature Physics 10, 271–277 (2014).   
5. Marshall, W., Simon, C., Penrose, R. & Bouwmeester, D. Towards quantum superpositions of a mirror. Phys. Rev. Lett. 91, 130401 (2003).   
6. Leibfried, D. et al. Experimental determination of the motional quantum state of a trapped atom. Phys. Rev. Lett. 77, 4281-4285 (1996).

7. Manenti, R. et al. Circuit quantum acoustodynamics with surface acoustic waves. Nature Communications 8, 975 (2017).   
8. Riedinger, R. et al. Non-classical correlations between single photons and phonons from a mechanical oscillator. Nature 530, 313–316 (2016).   
9. Lee, K. C. et al. Entangling macroscopic diamonds at room temperature. Science 334, 1253-1256 (2011).   
10. Safavi-Naeini, A. H. et al. Squeezed light from a silicon micromechanical resonator. Nature 500, 185–189 (2013).   
11. Riedinger, R. et al. Remote quantum entanglement between two micromechanical oscillators.
Nature 556, 473–477 (2018).   
12. Ockeloen-Korppi, C. F. et al. Stabilized entanglement of massive mechanical oscillators. Nature 556, 478–482 (2018).   
13. Gustafsson, M. V. et al. Propagating phonons coupled to an artificial atom. Science 346, 207–211 (2014).   
14. Andrews, R. W. et al. Bidirectional and efficient conversion between microwave and optical light. Nature Physics 10, 321–326 (2013).   
15. Schuetz, M. J. A. et al. Universal quantum transducers based on surface acoustic waves. Phys. Rev. X 5, 031031 (2015).

16. Naik, R. K. et al. Random access quantum information processors using multimode circuit quantum electrodynamics. Nature Communications 8, 1904 (2017).   
17. Leghtas, Z. et al. Hardware-efficient autonomous quantum memory protection. Phys. Rev. Lett. 111, 120501 (2013).   
18. Chou, K. et al. Deterministic teleportation of a quantum gate between two logical qubits. arXiv:1801.05283 (2018).   
19. Satzinger, K. J. et al. Quantum control of surface acoustic wave phonons. arXiv:1804.07308 (2018).   
20. Hofheinz, M. et al. Generation of Fock states in a superconducting quantum circuit. Nature 454, 310–314 (2008).   
21. Hofheinz, M. et al. Synthesizing arbitrary quantum states in a superconducting resonator. Nature 459, 546–549 (2009).   
22. Royer, A. Wigner function as the expectation value of a parity operator. Phys. Rev. A 15, 449–450 (1977).   
23. Gambetta, J. et al. Qubit-photon interactions in a cavity: Measurement-induced dephasing and number splitting. Phys. Rev. A 74, 042318 (2006).   
24. Schuster, D. I. et al. Resolving photon number states in a superconducting circuit. Nature 445, 515–518 (2007).

25. Heeres, R. W. et al. Implementing a universal gate set on a logical qubit encoded in an oscillator. Nature Communications 8, 1–7 (2017).   
26. Kharel, P. et al. Ultra-high-Q phononic resonators on-chip at cryogenic temperatures. APL Photonics 3, 066101 (2018).   
27. Penrose, R. On gravity's role in quantum state reduction. General Relativity and Gravitation 28, 581–600 (1996).   
28. Pikovski, I., Vanner, M. R., Aspelmeyer, M., Kim, M. S. & Brukner, v. Probing planck-scale physics with quantum optics. Nature Physics 8, 393–397 (2012).   
29. Marin, F. et al. Gravitational bar detectors set limits to Planck-scale physics on macroscopic variables. Nature Physics 9, 71–73 (2013).

Acknowledgements We thank Michel Devoret, Steve Girvin, Yaxing Zhang, Kevin Chou, and Vijay Jain for helpful discussions. We thank Katrina Silwa for providing the Josephson parametric converter amplifier. This research was supported by the US Army Research Office (W911NF-14-1-0011), ONR YIP (N00014-17-1-2514), NSF MRSEC (DMR-1119826), and the Packard Fellowship for Science and Engineering. Facilities use was supported by the Yale SEAS cleanroom, the Yale West Campus Cleanroom, and the Yale Institute for Nanoscience and Quantum Engineering (YINQE).

Author contributions Y.C. performed the experiment and analyzed the data under the supervision of P.T.R. and R.F.S. Y.C., P.K., and L.F. designed and fabricated the device. P.K. and T.Y. provided experimental suggestions and theory support. Y.C., P.K., P.T.R, and R.J.S. wrote the manuscript with contributions from all authors.

Author information : Reprints and permissions information is available at www.nature.com/reprints.R.J.S., and L.F. are founders and equity shareholders of Quantum Circuits, Inc. Correspondence and requests for materials should be addressed to Y. Chu (email: yiwen.chu@yale.edu) or R. J. Schoelkopf (robert.schoelkopf@yale.edu).

![](images/e74b5fac9acfb9ff8d893b6d57a80f8f4d83a107bf8e952c199df8f42d49271f.jpg)

<details>
<summary>text_image</summary>

GE varnish (adhesive)
d_e=50 µm
d_c=255 µm
h_s~500 µm
AlN
Al
Sapphire
d_m~40 µm
Phonon mode
Al
h_g~650 nm
h_AIN=960 nm
</details>

![](images/516105dbedfaff53713a9de266e5aa0b3fea26349c8d5f9a4b7e657045f48125.jpg)

<details>
<summary>heatmap</summary>

| Current (mA) | Frequency (GHz) | Population |
| ------------ | --------------- | ---------- |
| 0.94         | 6.289           | 0.0        |
| 0.96         | 6.288           | 0.1        |
| 0.98         | 6.287           | 0.2        |
| 1.00         | 6.286           | 0.3        |
| 1.02         | 6.285           | 0.4        |
| 1.04         | 6.285           | 0.5        |
| 1.04         | 6.285           | 0.6        |
| 1.04         | 6.285           | 0.7        |
| 1.04         | 6.285           | 0.8        |
| 1.04         | 6.285           | 0.9        |
| 1.04         | 6.285           | 1.0        |
</details>

Figure 1: The $\hbar$ BAR and strong qubit-phonon coupling. a, Top and side view schematic of the $\hbar$ BAR (not to scale). The chip containing the acoustic resonator also has six nominally identical resonators on the edges of the chip that have Al spacers deposited on them. The measured thickness of the Al spacer between the qubit and acoustic resonator ( $h_g$ ) is shown, but the actual spacing may be larger due to imperfections in the flip-chip assembly. Other dimensions indicated are the diameters of the transducer electrode ( $d_e$ ), curved resonator surface ( $d_c$ ), and acoustic mode waist ( $d_m$ ), along with the thicknesses of the AlN ( $h_{\text{AlN}}$ ) and sapphire substrate ( $h_s$ ). b, Spectroscopy of the transmon qubit near the $l_1$ acoustic mode while varying the current in an external coil used to flux tune the qubit frequency.

![](images/5cc5909c897a9a72de071593368873ea31b21c68d851f278b66c2e106ba0656e.jpg)  
Figure 2: Mode structure of the $\hbar$ BAR. a, Result of exciting the qubit and measuring its excited state population after a variable delay as the qubit is flux tuned. b, Logarithm of the Fourier transform of the data in a. The qubit frequencies shown on the horizontal axis are determined from spectroscopy data taken at each applied flux. The three highest frequency fundamental transverse modes ( $m, n = 0$ ) that are fully accessible by the qubit are shown and labeled with their longitudinal mode numbers. The white dashed lines indicate the hyperbolic dependence of the effective vacuum Rabi frequencies for the three most dominant modes in one FSR. The black dashed line indicates the value of $2g_0$ for the fundamental mode, which is at least a factor of five larger than the coupling rates to the other modes.

![](images/5e81cf99ca5b5215a9d43c6723c8addbe1656716e87a6109ddad085341f13189.jpg)

<details>
<summary>line</summary>

| Time Segment         | Qubit drive (v1) | Qubit frequency (v2) |
|----------------------|------------------|----------------------|
| Qubit cooling       | π                | Ts                   |
| State preparation   | Ts               | Ts/√2                |
| Measurement          | √N               | Ts/√3                |
</details>

![](images/b901b7ea9c55180f080b4cb22aeb4313dd8f63c4a56ab44adc7cda3e4eb6b27d.jpg)

<details>
<summary>line</summary>

| t (μs) | N=0     | N=1     | N=2     | N=3     | N=4     | N=5     | N=6     | N=7     |
|--------|---------|---------|---------|---------|---------|---------|---------|---------|
| 0      | ~0      | ~0      | ~0      | ~0      | ~0      | ~0      | ~0      | ~0      |
| 2      | ~0      | ~0      | ~0      | ~0      | ~0      | ~0      | ~0      | ~0      |
| 4      | ~0      | ~0      | ~0      | ~0      | ~0      | ~0      | ~0      | ~0      |
| 6      | ~0      | ~0      | ~0      | ~0      | ~0      | ~0      | ~0      | ~0      |
| 8      | ~0      | ~0      | ~0      | ~0      | ~0      | ~0      | ~0      | ~0      |
| 10     | ~0      | ~0      | ~0      | ~0      | ~0      | ~0      | ~0      | ~0      |
| 12     | ~0      | ~0      | ~0      | ~0      | ~0      | ~0      | ~0      | ~0      |
</details>

![](images/a186e3f730edf63ae26c10c150bd670266c06a845d3ddad88538b7db8069b440.jpg)

<details>
<summary>heatmap</summary>

| Frequency (kHz) | Amplitude (a.u.) |
| --------------- | ---------------- |
| 0               | 0                |
| 500             | 5                |
| 1000            | 15               |
| 1500            | 25               |
| 2000            | 40               |
</details>

![](images/780860c52bc4cfc453fe3434b2fd8e58bd0c8877397966bd05f8dc203e1432c2.jpg)

<details>
<summary>bar</summary>

| n  | Population p_n,N |
|----|------------------|
| 0  | 1.00             |
| 2  | 0.86             |
| 4  | 0.74             |
| 6  | 0.62             |
| 8  | 0.55             |
| 10 | 0.45             |
| 12 | 0.36             |
| 14 | 0.26             |
</details>

Figure 3: Climbing the phonon Fock state ladder. a, Pulse sequence for the generation and measurement of phonon Fock states. $T_{s} = \pi / 2g_{0}$ is the duration of a swap operation in the one excitation manifold. In the state preparation step, the duration of the $k$ th swap is scaled to account for the coupling rate of $g_{k} = \sqrt{k}g_{0}$ between the $|g, k\rangle$ and $|e, k - 1\rangle$ states. Pulses intended to excite the qubit from $|g\rangle$ to $|e\rangle$ are labeled with $\pi$ . Qubit frequencies $\nu_{0}, \nu_{1}$ , and $\nu_{2}$ are described in the text. b, Qubit excited state population after interacting with the phonon for a time $t$ following a $N$ phonon preparation procedure. Black lines are fits used to extract the Fock state populations shown in d. c, Fourier transform of the data in b, obtained by subtracting the mean of each dataset in b and padding with the resultant final value to effectively smooth the Fourier transform. Black line is $2g_{N}$ . d, Populations in Fock state $n$ extracted from b. Numbers show the populations in $n = N$ . Error bars indicate the result of changing the value of $g_{0}$ used in the simulations of $p_{e,n}(t)$ by $\pm 5\mathrm{kHz}$ .

![](images/5d2d5e598e9c9ce94f776e22bd241befc0f4ad92932fd3e6b0ddd59ec3de5766.jpg)

<details>
<summary>heatmap</summary>

| Re(α) \ Im(α) | 2    | 1    | -1   | -2   |
|---------------|------|------|------|------|
| -2            |      |      |      |      |
| -1            |      |      |      |      |
| 0             |      |      |      |      |
| 1             |      |      |      |      |
| 2             |      |      |      |      |
</details>

![](images/1a446229cddf2d6cd787a2eb15931ecb99b1983f56b504a560ada62fb753720a.jpg)

![](images/1eeff97a3555d66c668315a3ba6c23457c1ed3b35ed7c93f3cfbb168e7db6f96.jpg)

<details>
<summary>heatmap</summary>

| Re(α) \ Parity | -2    | -1    | 0     | 1     | 2     |
| -------------- | ----- | ----- | ----- | ----- | ----- |
| -2             | -1.0  | 0.8   | 0.6   | 0.4   | 0.2   |
| -1             | -0.8  | 0.6   | 0.4   | 0.2   | 0.0   |
| 0              | -0.6  | 0.4   | 0.2   | 0.0   | -0.2  |
| 1              | -0.4  | 0.2   | 0.0   | -0.2  | -0.4  |
| 2              | -0.2  | 0.0   | -0.2  | -0.4  | -0.6  |
</details>

![](images/e7e61e104c09cc0191fd15cb54381f3843281a5444f138c40982a964b0213c54.jpg)

<details>
<summary>heatmap</summary>

| Re(α) | Im(α) | Value |
|-------|-------|-------|
| -2    | -2    | -1.0  |
| -2    | -1    | -0.8  |
| -2    | 0     | -0.6  |
| -2    | 1     | -0.4  |
| -2    | 2     | -0.2  |
| 2     | -2    | 0.0   |
| 2     | -1    | 0.2   |
| 2     | 0     | 0.4   |
| 2     | 1     | 0.6   |
| 2     | 2     | 0.8   |
| 1     | -2    | 1.0   |
| 1     | -1    | 0.8   |
| 1     | 0     | 0.6   |
| 1     | 1     | 0.4   |
| 1     | 2     | 0.2   |
| -1    | -2    | 0.0   |
| -1    | -1    | -0.2  |
| -1    | 0     | -0.4  |
| -1    | 1     | -0.6  |
| -1    | 2     | -0.8  |
| 0     | -2    | -1.0  |
| 0     | -1    | -0.8  |
| 0     | 0     | -0.6  |
| 0     | 1     | -0.4  |
| 0     | 2     | -0.2  |
| 1     | -2    | 0.0   |
| 1     | -1    | 0.2   |
| 1     | 0     | 0.4   |
| 1     | 1     | 0.6   |
| 1     | 2     | 0.8   |
| 2     | -2    | 1.0   |
| 2     | -1    | 0.8   |
| 2     | 0     | 0.6   |
| 2     | 1     | 0.4   |
| 2     | 2     | 0.2   |
</details>

![](images/57f355fdeb8b499b32f9b069b5c3ade61b18c092ea58063a988fffcc097210ca.jpg)

<details>
<summary>heatmap</summary>

| Re(α) | Im(α) | Parity |
|-------|-------|--------|
| -1.6  | 1.6   | -1.6   |
| -1.6  | 0.8   | -0.8   |
| -1.6  | 0.0   | -0.8   |
| -1.6  | -0.8  | -0.8   |
| -1.6  | -1.6  | -1.6   |
| 1.6   | 1.6   | -1.6   |
| 1.6   | 0.8   | -0.8   |
| 1.6   | 0.0   | -0.8   |
| 1.6   | -0.8  | -0.8   |
| 1.6   | -1.6  | -1.6   |
</details>

![](images/5de47681205b2ba5284acbc9163c8d99c51618da9733aea1ea8326f38b53de8c.jpg)

<details>
<summary>heatmap</summary>

| Re(α) | Im(α) | Parity |
|-------|-------|--------|
| -2    | 2     | -1.0   |
| -1    | 1     | -0.8   |
| 0     | 0     | 0.0    |
| 1     | -1    | 0.8    |
| 2     | -2    | 1.0    |
</details>

![](images/54e55cd9ff5404c0841aee271101d293aaad06fb72e7c6fb90fef40d067c5c8c.jpg)

<details>
<summary>line</summary>

| Re(α) | Parity |
|-------|--------|
| -2.0  | 0.0    |
| -1.5  | 0.3    |
| -1.0  | 0.4    |
| -0.5  | 0.3    |
| 0.0   | -1.0   |
| 0.5   | -0.5   |
| 1.0   | 0.4    |
| 1.5   | 0.2    |
| 2.0   | 0.0    |
</details>

![](images/858b743b4ecb578ecba694e343a3bd78e55c0f53a4b2153a5f18df6fbb485bbd.jpg)

<details>
<summary>line</summary>

| Re(α) | Parity (Black Line) | Parity (Teal Line) |
|-------|---------------------|--------------------|
| -1.5  | 0.0                 | 0.0                |
| -1.0  | 0.0                 | 0.0                |
| -0.5  | -0.2                | -0.1               |
| 0.0   | -0.4                | -0.3               |
| 0.5   | 0.9                 | 0.8                |
| 1.0   | 0.6                 | 0.5                |
| 1.5   | 0.0                 | 0.0                |
</details>

![](images/abc3d470eee421cabeceb9472daf6e5720f571df0dd8af6135c6bdb05d0ee66a.jpg)

<details>
<summary>line</summary>

| Re(α) | Parity (Black Line) | Parity (Teal Line) |
|-------|---------------------|--------------------|
| -2.0  | 0.0                 | 0.0                |
| -1.5  | 0.3                 | 0.2                |
| -1.0  | 0.1                 | 0.1                |
| -0.5  | -0.4                | -0.2               |
| 0.0   | 1.0                 | 0.5                |
| 0.5   | -0.4                | -0.2               |
| 1.0   | 0.3                 | 0.2                |
| 1.5   | 0.1                 | 0.1                |
| 2.0   | 0.0                 | 0.0                |
</details>

Figure 4: Wigner tomography of non-classical states of motion. a, b, c, Measured Wigner functions of the prepared states $|1\rangle$ , $(|0\rangle + |1\rangle) / \sqrt{2}$ , and $|2\rangle$ . Each grid point is a separate experiment with displacement by a phase space amplitude $\alpha$ . d, e, f, Wigner functions of the density matrices reconstructed from a, b, and c. g, h, i, Cuts of the ideal Wigner function (black line), data (brown points), and reconstructed Wigner function (green line) along the $\mathrm{Im}(\alpha) = 0$ axis. Negative values of the Wigner function are indicators of a non-classical state of motion. Error bars on the data are extracted in the same way as in Figure 3d.

# Data availability

The data that support the findings of this study are available from the corresponding authors upon reasonable request.

# Supplementary information for:

# Climbing the phonon Fock state ladder

# 1 Fabrication procedures

The transmon qubit used in our device is fabricated on sapphire using a standard e-beam lithography and Dolan bridge process $^{1}$ . The fabrication procedures for the acoustic resonator chip is shown in Figure S1a. We begin with a commercially purchased double side polished 2” sapphire wafer with a 1 $\mu$ m thick film of c-axis oriented AlN grown on one side (Kyma Technologies, part number H.AT.U.050.1000). The convex surfaces of the device resonator and the spacer resonators at the edges of the chip are then fabricated on the side with AlN using a procedure that has previously been demonstrated for making micro-lenses and high-Q high-overtone bulk acoustic wave resonators (HBAR) in other materials $^{2}$ . The procedure uses photoresist that has been reflown into hemispheres as a mask for reactive ion etching (RIE), which transfers the hemispherical geometry into the substrate due to the finite etch selectivity.

We first pattern disks of photoresist (AZP 4620) on the wafer, which is then attached to a heated chuck at $55^{\circ}$ C and inverted above a beaker of solvent (AZ-EBR) on a hotplate at $60^{\circ}$ C. After 2-3 hours, the disks of resist will have reabsorbed the solvent and reflowed into hemispheres due to surface tension. The mode structure of the resonator depends on the radius of curvature of the final convex surface, which in turn depends on the radius of curvature and geometry of the reflowed resist hemisphere. The radii of each hemisphere is approximately that of the original disk,

while the height and shape depend on the disk radius, the thickness of the resist, and the reflow time. Therefore, we use identical radii for the device and spacer resonators within each chip to ensure that they result in nominally identical heights of the final surface.

After reflowing the resist, we bake the wafer starting at 90 °C, gradually increasing to 145 °C over the course of 15 minutes to remove the solvent and harden the resist. The wafer is then etched in an Oxford 100 RIE/ICP etcher (Cl₂/BCl₃/Ar at 4/26/15 sccm, 8 mTorr, 70 W RF power, 350 W ICP power). The resist mask has an etch rate of ∼200 nm/min, while AlN and sapphire have etch rates of ∼60 nm/min and ∼20 nm/min, respectively. We stop the etch when the maximum thickness of the AlN is ∼ λ/2, where λ is the acoustic wavelength at the qubit frequency, to maximize the qubit-phonon coupling strength. The different etch rates of the AlN and sapphire results in a two-tiered geometry of the final surface, as shown in Figure S1b. However, the simulated mode radius is only ∼20 μm (see Section 2), which means that the mode does not extend beyond the AlN part of the curved surface.

After making the resonators, we use photolithography, e-beam evaporation, and liftoff to add Al spacers on top of the spacer resonators. The wafer is then diced into individual chips, which are then combined with the qubit chips using a home-built alignment and flip-chip bonding setup. We first pick up the resonator chip with a small drop of PDMS at the end of a tungsten tip, which is attached to a three-axis translation stage. We then use the stage to align the Al spacers on the resonator chip to corresponding features patterned on the qubit chip, which results in the device resonator being aligned with the transduction electrode on the qubit. It is also possible to directly

align the device resonator using interference rings that are visible once the two chips are brought into contact. Once alignment is achieved, we add small drops of GE varnish on the edges to hold the chips together.

![](images/4efb6d0e87b4b4f60e82e36e8fc6fa6dfde5fc4a9360a2ac5386cb4196be57e4.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["c-axis AlN on sapphire"] --> B["Photolithography AZP 4620"]
    B --> C["Reflow in AZ-EBR vapor"]
    C --> D["Photolithography AZ nLoF 2035"]
    D --> E["ICP RIE Cl₂/BCl₃/Ar"]
    E --> F["Liftoff"]
    subgraph A
        G["Al"]
        H["Photoresist"]
        I["Aluminum Nitride"]
        J["Sapphire"]
    end
```
</details>

![](images/8bc7adde59370c4675017447d66bfbf4cdab557773342547db16833b91d8142b.jpg)

<details>
<summary>surface_3d</summary>

| x (mm) | y (mm) | z (mm) |
| ------ | ------ | ------ |
| -0.2   | -0.2   | 0      |
| -0.2   | 0.0    | 1.5    |
| -0.2   | 0.2    | 2      |
| 0.0    | -0.2   | 0      |
| 0.0    | 0.0    | 1.5    |
| 0.0    | 0.2    | 2      |
| 0.2    | -0.2   | 0      |
| 0.2    | 0.0    | 1.5    |
| 0.2    | 0.2    | 2      |
</details>

Figure S1: Fabrication of acoustic resonator chip. a. Fabrication procedure. Drawings show a cross section through a part of the chip that includes the actual acoustic resonator device and one of the spacer resonators at the edges of the chip. b. Convex surface of acoustic resonator measured using optical profilometry.

# 2 Acoustic mode simulations

Detailed descriptions of the design and simulation of high-Q HBAR resonators are given in $^{2-4}$ . The simulation propagates the acoustic wave over many roundtrips through the resonator and calculates the interferometric sum of the fields. At frequencies where a stable mode exists, the interference is constructive, and we obtain a well-defined mode profile with a large total intensity. We note that the simulated mode spectrum should not depend on the initial excitation, but certain modes can be missed if the initial excitation has particular symmetries. Therefore, we use an initial

acoustic excitation that was found by first simulating the qubit electric field pattern with HFSS, then calculating the strain profile that is generated by this field through the piezoelectric transducer.

We now show that the simulated mode structure and coupling strengths are in qualitative agreement with what we observe. Figure S2a shows one free spectral range (FSR) of the measured data from Figure 2a of the main text, where we can clearly see a collection of modes that couple to the qubit. We compare this to the simulated acoustic mode spectrum shown in Figure S2b, where we plot the total mode intensity versus the frequency of the acoustic excitation. The free parameters to match the simulated mode frequencies to the experimental data are the longitudinal and transverse sound velocities of sapphire, which are not well known at low temperatures. We find them to be $v_{l} = 11100$ m/s and $v_{t} = 8540$ m/s, which is similar to typical measured values and the values used in our previous work $^{4}$ . All other geometrical parameters, including the shape of the curved surface, are known from independent measurements such as the one shown in Figure S1b. The simulated spectrum agrees reasonably well with the experimental data, especially for the first few transverse modes in each FSR. There appear to be additional modes in the measured data, which may be higher order transverse modes from the neighboring longitudinal mode number. Another possibility is the existence of acoustic modes with other polarizations in the resonator, which are not included in the simulations. The qubit may couple to these modes through the non-zero transverse components of the electric field and shear components of the AlN piezoelectric tensor.

The relative heights of the peaks in Figure S2b give some sense of how well each mode is

coupled to the qubit. However, to obtain the coupling rates more accurately, we find the strain profiles at only the simulated mode frequencies after $2 \times 10^{5}$ round trips. We then calculate the coupling rate, which includes the overlap of the simulated qubit electric field, the shape of the piezoelectric transducer, and the simulated acoustic mode profiles, as described in the Supplementary Materials of $^{4}$ . Figure S2c shows the resulting predicted coupling rate g for each mode. We see that the relative coupling strengths qualitatively agree with what we experimentally observe, with the fundamental Gaussian mode being at least a factor of five more strongly coupled than any higher order mode. The overall strength of the coupling is higher in the simulations than what is observed in our experiment, which could be due to a variety of reasons. One is a lower piezoelectric constant for the actual material than the value used in the simulation, which we have not independently measured. Misalignment and a larger gap between the two chips than expected would also lead to different overall and relative coupling strengths. To resolve these discrepancies, we hope to more systematically characterize the flip-chip device geometry in the future.

# 3 Coherent displacement calibration

In this section, we show that our procedure for doing a direct displacement on the phonon mode does indeed result in a coherent state. Figure S3 shows the extracted Fock state populations after performing displacements with six different amplitudes within the range used for the measurements of Wigner functions. As in the main text, the displacement drive is a Gaussian pulse at the phonon frequency with a 1 $\mu$ s RMS width and 4 $\mu$ s total width. We find that they agree well with the expected Poisson distributions for coherent states whose amplitudes $\alpha$ are given by a single scale

factor multiplied by the actual drive amplitudes used in the experiments. This is in practice how we calibrate the experimental drive amplitudes to the actual displacement amplitudes, and the same scale factor is used in all Wigner tomography experiments. We find that for shorter pulses and larger drive amplitudes, the population distributions deviate from that of coherent states. These effects are reproduced in simulations, and are potentially due to off-resonant excitations of the qubit. In the case of significant initial excited state population of the qubit, our procedure for measuring the phonon populations is no longer valid. On the other hand, a longer pulse with smaller amplitudes results in more decoherence during the displacement pulse, thus limiting our ability to perform Wigner tomography on a prepared state. The pulse length used here and in the data shown in Figure 4 of the main text is a compromise between these two considerations. Future optimization of the qubit detuning during the displacement and the pulse shape could potentially improve the robustness and range of amplitudes of the displacement drive.

# 4 State reconstruction

State reconstruction was done using the formalism described in $^{5}$ . We use a maximum likelihood method that finds the most probable density matrix $\rho$ that results in the Wigner functions we measured, subject to the constraints that $\rho$ is physical (positive semi-definite, Hermitian, and unit trace). We used Fock states up to n = 14 for extracting the populations $p_{n,\rho}(\alpha)$ of both the Fock states shown in Figure 3 of the main text and for calculating the Wigner functions shown in Figure 4 of the main text. We truncate the Hilbert space at Fock state n = 9 for the reconstruction. Figure S4 shows the reconstructed density matrices that were used to plot the Wigner functions in Figures

4d, e, and f of the main text.

1. Wang, C. et al. Surface participation and dielectric loss in superconducting qubits. Applied Physics Letters 107 (2015).   
2. Kharel, P. et al. Ultra-high-Q phononic resonators on-chip at cryogenic temperatures. arXiv:1803.10077 (2018).   
3. Renninger, W. H., Kharel, P., Behunin, R. O. & Rakich, P. T. Bulk crystalline optomechanics. Nature Physics (2018).   
4. Chu, Y. et al. Quantum acoustics with superconducting qubits. Science 358, 199–202 (2017).   
5. Chou, K. et al. Deterministic teleportation of a quantum gate between two logical qubits. arXiv:1801.05283 (2018).

![](images/add30f85aa6b9d617c2440ae656fddeb369a2cdbfed1ee29a47705466b668b04.jpg)  
Figure S2: Acoustic resonator simulations. a. Zoomed in view of one acoustic FSR from Figure 2a of the main text. b. Simulated acoustic mode spectrum. c. Simulated coupling strengths for a 1 $\mu$ m spacing between the qubit and resonator chips. Insets show simulated strain profiles for three modes in a 300 $\mu$ m × 300 $\mu$ m area. Thick dashed lines indicate the locations of the fundamental Gaussian modes, while thin dashed lines indicate the locations of higher order Hermite-Gaussian modes.

![](images/d17d2b83d5f7a898898427be629827cff21f44d70d9a591f01b3e7f56d6ca89f.jpg)

<details>
<summary>bar</summary>

| n  | α = 2.08 | α = 1.66 | α = 1.25 | α = 0.83 | α = 0.42 | α = 0.00 |
|----|----------|----------|----------|----------|----------|----------|
| 0  | ~0.1     | ~0.1     | ~0.1     | ~0.1     | ~0.1     | ~0.1     |
| 2  | ~0.1     | ~0.1     | ~0.1     | ~0.1     | ~0.1     | ~0.1     |
| 4  | ~0.1     | ~0.1     | ~0.1     | ~0.1     | ~0.1     | ~0.1     |
| 6  | ~0.1     | ~0.1     | ~0.1     | ~0.1     | ~0.1     | ~0.1     |
| 8  | ~0.1     | ~0.1     | ~0.1     | ~0.1     | ~0.1     | ~0.1     |
| 10 | ~0.1     | ~0.1     | ~0.1     | ~0.1     | ~0.1     | ~0.1     |
| 12 | ~0.1     | ~0.1     | ~0.1     | ~0.1     | ~0.1     | ~0.1     |
| 14 | ~0.1     | ~0.1     | ~0.1     | ~0.1     | ~0.1     | ~0.1     |
</details>

Figure S3: Coherent displacements of the phonon mode. Bars indicate the measured populations. Black dots are the ideal Poisson distributions for each $\alpha$ .

a   
![](images/10358c17803cfd4fe1a19023c898b5c09a3e16dbdcaa92ae974e790316516d0a.jpg)

<details>
<summary>heatmap</summary>

| n \ p | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    |
|-------|------|------|------|------|------|------|------|------|
| 0     | -0.8 | 0.8  | 0.6  | 0.4  | 0.2  | 0.0  | -0.2 | -0.4 |
| 1     | -0.8 | 1.0  | 0.6  | 0.4  | 0.2  | 0.0  | -0.2 | -0.4 |
| 2     | -0.8 | 0.6  | 0.4  | 0.2  | 0.0  | -0.2 | -0.4 | -0.6 |
| 3     | -0.8 | 0.4  | 0.2  | 0.0  | -0.2 | -0.4 | -0.6 | -0.8 |
| 4     | -0.8 | 0.2  | 0.0  | -0.2 | -0.4 | -0.6 | -0.8 | -1.0 |
| 5     | -0.8 | 0.0  | -0.2 | -0.4 | -0.6 | -0.8 | -1.0 | -1.0 |
| 6     | -0.8 | -0.2 | -0.4 | -0.6 | -0.8 | -1.0 | -1.0 | -1.0 |
| 7     | -0.8 | -0.4 | -0.6 | -0.8 | -1.0 | -1.0 | -1.0 | -1.0 |
| 8     | -0.8 | -0.6 | -0.8 | -1.0 | -1.0 | -1.0 | -1.0 | -1.0 |
The heatmap uses a color scale from red to blue to indicate values ranging from -1.0 to 1.0, with darker shades representing higher values and lighter shades indicating lower values. The x-axis labels are 'n' and 'p', and the y-axis also labels 'n'. There is no explicit title or legend provided in the image.
</details>

|b   
![](images/32373f86044346e471dc6221ee9ea191fbdb543dd83c00e741cc9996759b332a.jpg)

<details>
<summary>heatmap</summary>

| n \ n | 0    | 2    | 4    | 6    | 8    |
|-------|------|------|------|------|------|
| 0     | 0.0  | 1.5  | -    | -    | -    |
| 2     | -    | -    | -    | -    | -    |
| 4     | -    | -    | -    | -    | -    |
| 6     | -    | -    | -    | -    | -    |
| 8     | -    | -    | -    | -    | -    |
</details>

lc   
![](images/2baccf11a4e6b24d393c5ba4480b93ba2facbdbedff9a472043b73cbd04ca413.jpg)

<details>
<summary>heatmap</summary>

| n \ n | 0    | 2    | 4    | 6    | 8    |
|--------|------|------|------|------|------|
| 0      | -0.2 | 0.1  | 0.3  | 0.5  | 0.7  |
| 2      | 0.1  | 0.2  | 0.4  | 0.6  | 0.8  |
| 4      | -0.3 | 0.4  | 0.6  | 0.8  | 1.0  |
| 6      | -0.5 | 0.6  | 0.8  | 1.0  | 1.2  |
| 8      | -0.7 | 0.8  | 1.0  | 1.2  | 1.4  |
</details>

![](images/5896eeec72fa5b392222871f5be1c49f1c066e815dbe5fd68fca661f6dd3c513.jpg)

<details>
<summary>heatmap</summary>

| n \ ε | 0    | 2    | 4    | 6    | 8    |
|-------|------|------|------|------|------|
| 0     | -0.045 | -0.030 | -0.015 | 0.000 | 0.015 |
| 2     | -0.015 | 0.015 | 0.030 | 0.045 | 0.030 |
| 4     | 0.000 | 0.015 | 0.030 | 0.045 | 0.030 |
| 6     | 0.015 | 0.030 | 0.045 | 0.030 | 0.015 |
| 8     | 0.030 | 0.045 | 0.030 | 0.015 | 0.000 |
</details>

![](images/e39364c252e0d9ad7b7ae1ae6a9685eda6eb0b83d6ad4b01608b7cbfb49e4f36.jpg)

<details>
<summary>heatmap</summary>

| n \ e | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | -0.045 | -0.030 | -0.015 | 0.000 | 0.015 | 0.030 | 0.045 | 0.030 | 0.015 |
| 1 | -0.030 | -0.015 | 0.000 | 0.015 | 0.030 | 0.045 | 0.030 | 0.015 | 0.000 |
| 2 | -0.015 | 0.000 | 0.015 | 0.030 | 0.045 | 0.030 | 0.015 | 0.000 | -0.015 |
| 3 | 0.000 | 0.015 | 0.030 | 0.045 | 0.030 | 0.015 | 0.000 | -0.015 | -0.030 |
| 4 | 0.015 | 0.030 | 0.045 | 0.030 | 0.015 | 0.000 | -0.015 | -0.030 | -0.045 |
| 5 | 0.030 | 0.045 | 0.030 | 0.015 | 0.000 | -0.015 | -0.030 | -0.045 | -0.045 |
| 6 | 0.045 | 0.030 | 0.015 | 0.000 | -0.015 | -0.030 | -0.045 | -0.045 | -0.045 |
| 7 | 0.030 | 0.015 | 0.000 | -0.015 | -0.030 | -0.045 | -0.045 | -0.045 | -0.045 |
| 8 | 0.015 | 0.015 | 0.15 | 1.5 | 2.5 | 3.5 | 4.5 | 5.5 | 6.5 |
| 9 | 1.5 | 2.5 | 3.5 | 4.5 | 5.5 | 6.5 | 7.5 | 8.5 | 9 |
| 10 | 2.5 | 3.5 | 4.5 | 5.5 | 6.5 | 7.5 | 8.5 | 9.5 | 12 |
| 11+ | -2.5 | -3.5 | -4.5 | -5.5 | -6.5 | -7.5 | -8.5 | -9.5 | -12 |
The values in the table represent the magnitude of the y-axis variable 'e'. The color scale indicates the value range from -12 to +12.
</details>

![](images/fb5368381ebf03aa838584f2892747b61809d76e2d80201834214e85d7bdc1b2.jpg)

<details>
<summary>heatmap</summary>

| n \ c | 0 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| 0 | -0.015 | -0.030 | -0.015 | -0.030 | -0.045 |
| 2 | -0.045 | -0.015 | -0.030 | -0.015 | -0.015 |
| 4 | -0.015 | -0.015 | -0.015 | -0.015 | -0.015 |
| 6 | -0.015 | -0.015 | -0.015 | -0.015 | -0.015 |
| 8 | -0.015 | -0.015 | -0.015 | -0.015 | -0.015 |
The image displays a color-coded grid where darker red indicates higher values and lighter blue indicates lower values, suggesting a gradient of magnitude across the grid. The x-axis is labeled 'n', and the y-axis is labeled 'c'. There are no labels or additional data series in this image.
</details>

Figure S4: Density matrices. Real (top row) and imaginary (bottom row) parts of the density matrices for the states a. $|1\rangle$ , b. $(|0\rangle + |1\rangle)/\sqrt{2}$ , and c. $|2\rangle$ .