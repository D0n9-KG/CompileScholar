# Superconducting qubit to optical photon transduction

https://doi.org/10.1038/s41586-020-3038-6

Received: 10 April 2020

Accepted: 2 October 2020

Published online: 23 December 2020

Check for updates

Mohammad Mirhosseini $^{1,2,3,5}$ , Alp Sipahigil $^{1,2,3,5}$ , Mahmoud Kalaee $^{1,2,3,4,5}$ & Oskar Painter $^{1,2,3,4}$

Conversion of electrical and optical signals lies at the foundation of the global internet. Such converters are used to extend the reach of long-haul fibre-optic communication systems and within data centres for high-speed optical networking of computers. Likewise, coherent microwave-to-optical conversion of single photons would enable the exchange of quantum states between remotely connected superconducting quantum processors<sup>1</sup>.

Despite the prospects of quantum networking<sup>2</sup>, maintaining the fragile quantum state in such a conversion process with superconducting qubits has not yet been achieved. Here we demonstrate the conversion of a microwave-frequency excitation of a transmon-a type of superconducting qubit--into an optical photon.

We achieve this by using an intermediary nanomechanical resonator that converts the electrical excitation of the qubit into a single phonon by means of a piezoelectric interaction<sup>3</sup> and subsequently converts the phonon to an optical photon by means of radiation pressure<sup>4</sup>. We demonstrate optical photon generation from the qubit by recording quantum Rabi oscillations of the qubit through single-photon detection of the emitted light over an optical fibre.

With proposed improvements in the device and external measurement set-up, such quantum transducers might be used to realize new hybrid quantum networks<sup>2,5</sup> and, ultimately, distributed quantum computers<sup>6,7</sup>.

Recent developments with superconducting qubits have demonstrated fast, high-fidelity single- and two-qubit logic gates, making them a promising system for realizing quantum computers<sup>1</sup>. The low-loss environment of a superconductor and the strong single-photon nonlinearity from the Josephson effect provide an ideal combination for processing quantum information in the microwave domain<sup>8</sup>, but optical photons are a natural choice for quantum networking tasks<sup>9</sup> where they provide low propagation loss in room-temperature environments<sup>10</sup>.

A coherent microwave-to-optical interface can thus lead to hybrid architectures for quantum repeaters<sup>2,5</sup> by connecting superconducting qubits and ultrahigh- $Q$ (quality factor) microwave cavities<sup>11</sup>—serving as logic and memory registers—to 'flying' optical qubits as a means of long-distance information transfer.

Although the process of frequency conversion can be understood simply as a noise-free and lossless linear operation, an optical interface for superconducting qubits has not been realized because of the technical challenges inherent in the vast frequency difference between microwave (-5 GHz) and telecommunication-band optical (-200 THz) photons.

Microwave-to-optical frequency conversion can be achieved by bulk optical nonlinearities $^{12}$ . Alternatively, effective nonlinearities can be realized by intermediary degrees of freedom such as rare earth ions, magnons or phonons $^{13-15}$ that can simultaneously couple to microwave and optical fields. Using engineered nanomechanical resonators as such intermediary channels has been a particularly promising direction, where pioneering work in the past decade has demonstrated electrical

and optical preparation, control and readout of mechanical modes near their quantum ground state $^{3,16,17}$ . These demonstrations, together with rapid developments in superconducting quantum circuits $^{8}$ , have motivated recent experimental efforts to combine electromechanical and optomechanical devices to build a microwave-to-optical quantum transducer $^{18-22}$ . Although this approach has led to impressive conversion efficiencies $^{23}$ , all demonstrations so far have been limited to classical signals owing to a combination of challenges associated with optically induced or thermal noise, small transduction bandwidths and device integration complexities.

Here, we demonstrate the transduction of the microwave-frequency quantum excitations of a superconducting qubit into light at optical telecommunication frequencies, and use an optical fibre link and single-photon detection to register the quantum Rabi oscillations of the qubit. This is achieved using a chip-scale platform that integrates a transmon qubit with a piezo-optomechanical transducer.

We use a pulsed scheme to coherently transfer the quantum state of the qubit into a nanomechanical mode by a piezoelectric swap operation, and subsequently convert it to the optical domain by using a pulsed laser drive. This approach separates electrical and optical parts of the transduction sequence, avoiding the effects of light-induced noise on the superconducting circuitry. We find an overall added noise photon level for the transduction process to be $0.57 \pm 0.2$ , approaching the threshold required for remote entanglement generation of qubits[24].

<sup>1</sup>Kavli Nanoscience Institute, Laboratory of Applied Physics, California Institute of Technology, Pasadena, CA, USA. <sup>2</sup>Thomas J. Watson, Sr., Laboratory of Applied Physics, California Institute of Technology, Pasadena, CA, USA. <sup>3</sup>Institute for Quantum Information and Matter, California Institute of Technology, Pasadena, CA, USA. <sup>4</sup>AWS Center for Quantum Computing, Pasadena, CA, USA. <sup>5</sup>These authors contributed equally: Mohammad Mirhosseini, Alp Sipahigil, Mahmoud Kalaee. <sup>e</sup>-e-mail: opainter@caltech.edu

Article

Nature | Vol 588 | 24/31 December 2020

599

![](dt=2026-03-17/ht=12/c94eac8cea0d693ac451bd32bd4b745286a59c0387c605b280c51c5b1dfa0b13.jpg)

Figure 1a shows a schematic of the transduction process used in our experiment, where an intermediary mechanical mode is coupled to a qubit via a resonant piezoelectric interaction and to an optical mode via a parametric optomechanical interaction. The Hamiltonian for this system can be written as $\hat{H} = \hat{H}_0 + \hat{H}_{\mathrm{pe}} + \hat{H}_{\mathrm{om}}$ .

Here, $\hat{H}_0 / \hbar = \omega_{\mathrm{c}}\hat{a}_{\mathrm{o}}^{\dagger}\hat{a}_{\mathrm{o}} + \omega_{\mathrm{m}}\hat{b}_{\mathrm{m}}^{\dagger}\hat{b}_{\mathrm{m}} + \omega_{\mathrm{q}}(t)\hat{\sigma}_{\mathrm{ee}}$ describes the evolution of non-interacting subsystems, with $\hat{H}_{\mathrm{pe}} / \hbar = g_{\mathrm{pe}}(\hat{\sigma}_{\mathrm{eg}}\hat{b}_{\mathrm{m}} + \hat{\sigma}_{\mathrm{ge}}\hat{b}_{\mathrm{m}}^{\dagger})$ and $\hat{H}_{\mathrm{om}} / \hbar = G_{\mathrm{om}}(t)(\hat{a}_{\mathrm{o}}^{\dagger}\hat{b}_{\mathrm{m}} +

\hat{a}_{\mathrm{o}}\hat{b}_{\mathrm{m}}^{\dagger})$ describing the piezoelectric and optomechanical interactions, respectively.

The quantum modes are represented by: the creation (annihilation) operator of the optical mode, $a_{\mathrm{o}}^{\dagger}(\hat{a}_{\mathrm{o}})$ ; the creation (annihilation) operator of the mechanical mode, $\hat{b}_{\mathrm{m}}^{\dagger}(\hat{b}_{\mathrm{m}})$ ; the qubit excited-state projection operator, $\hat{\sigma}_{\mathrm{ee}}$ ; and the raising (lowering) operator of the qubit from ground to excited state, $\hat{\sigma}_{\mathrm{eg}}(\hat{\sigma}_{\mathrm{ge}})$ .

The centre frequency of the optical and mechanical modes are given by $\omega_{\mathrm{c}}$ and $\omega_{\mathrm{m}}$ , respectively, the transition frequency between ground and excited state of the qubit is given by $\omega_{\mathrm{q}}$ and $g_{\mathrm{pe}}$ is the single-phonon piezoelectric coupling rate between the mechanical mode and the qubit.

Here, we use the 'beam-splitter' form of the optomechanical interaction which is specific to the case where the
optical drive is red-detuned from the optical cavity resonance ( $\Delta = \omega_{\mathrm{c}} - \omega_{\mathrm{drive}} = \omega_{\mathrm{m}}$ ) and assumes operation in the resolved-sideband limit, where $\omega_{\mathrm{m}} \gg \kappa_{\mathrm{o}}$ ( $\kappa_{\mathrm{o}}$ is the linewidth of the optical resonance)[25].

In this case, the optomechanical coupling $G_{\mathrm{om}}(t) = \sqrt{n_{\mathrm{c}}(t)} g_{\mathrm{om}}$ is parametrically enhanced from the single-photon rate, $g_{\mathrm{om}}$ by the intra-cavity photon number $n_{\mathrm{c}}(t)$ from the pump laser at optical frequency $\omega_{\mathrm{drive}}$ .

Physical realization of the above Hamiltonian requires a materials platform that can support both optomechanical and piezoelectric components. High-fidelity qubit-mechanics swap operations can be realized based on the piezoelectric effect, where the electric field

from a qubit can be transformed to displacement in a mechanical resonator $^{26,27}$ . Quantum coherent optomechanical readout of mechanical modes can also be realized in optomechanical crystal (OMC) cavities $^{4,28}$ , where co-localization of mechanical and optical fields results in a large parametric coupling. Here we use a high-resistivity silicon-on-insulator wafer to integrate OMC cavities with large optomechanical coupling rates and low mechanical loss together with transmon qubits $^{29}$ . By sputter depositing and selectively patterning a thin film of aluminium nitride (AlN) on the silicon substrate, we are also able to achieve a localized piezoelectric response (see Methods).

Figure 1b shows a model of the transducer region of our device, which consists of a hybridized acoustic cavity formed from a wavelength-scale piezoacoustic resonator connected by a phonon (acoustic) waveguide to an OMC optomechanical cavity. To achieve simultaneously strong electrical and optical coupling to the hybridized cavity modes requires careful design of the individual (that is, detached) piezoelectric and optomechanical $^{30}$ resonators. The piezoacoustic cavity section (A.S., M.M., M.K., S. Messala & O.P.

, manuscript in preparation) is designed as a wavelength-scale Lamb wave resonator—made from a slab of AlN on top of the silicon device layer—that is released from the underlying buried oxide layer and connected laterally to the peripheral substrate by patterned silicon tethers that act as acoustic mirrors. A pair of aluminium electrodes in the form of an interdigital transducer (IDT) connect the transmon's capacitive leads to the piezoacoustic resonator as shown in the schematic of Fig. 1d. The submicrometre scale of the design in Fig.

1b results in a smaller piezoelectric coupling rate compared with earlier work $^{3}$ , but the small mechanical mode density limits the number of parasitic modes that can lead to both qubit decoherence and a reduction in optomechanical coupling. Hybridization of the acoustic

Article

600

Nature | Vol 588 | 24/31 December 2020

![](dt=2026-03-17/ht=12/12cdea2dafa20476fa56df6087d473c32980f20abd83741d51aced7a8c3394ab.jpg)

![](dt=2026-03-17/ht=12/93ba2c9827c745483a6240983a4d57010ac0ec501e159b4c3da4a62701329173.jpg)

modes of the piezoacoustic and OMC cavities is achieved by deforming the OMC 'mirror' cells in the intermediate section between the cavities to make a transmissive phonon waveguide while maintaining a high reflectivity for the optical field $^{31}$ . The resulting extended acoustic cavity supports several hybridized acoustic modes with different levels of energy concentrations in the AlN and silicon sections, one of which is shown in Fig. 1b. This results in different levels of optomechanical and piezoelectric interaction rates, as shown in Fig. 1c.

The strong hybridization via the phonon waveguide gives rise to a robust design, with substantial optomechanical and piezoelectric coupling rates for a range of hybridized modes even in the presence of fabrication-induced disorder and detuning between the bare piezoacoustic and OMC cavity modes (verified in numerical modelling; see Methods).

The characterization of the qubit and transducer device is performed in a dilution refrigerator, with the sample mounted to the mixing chamber plate of the fridge (base temperature $T_{\mathrm{f}} \approx 15 \mathrm{mK}$ ). Independent sets of optical and microwave spectroscopy measurements are initially performed to determine the piezoelectric and optomechanical coupling rates of the hybridized acoustic modes of the transducer.

We perform optical characterization by coupling the light from a tunable external cavity diode laser into a silicon waveguide on the chip via a lensed optical fibre (see Fig. 1e). By measuring the intensity of the reflected light from the on-chip waveguide as a function of laser detuning, the optical modes of the OMC cavity can be determined. For the device under test in this work, we find an optical resonance at a

![](dt=2026-03-17/ht=12/54b1b1e5bed0d2276dd997b65ebbd7ee2509906acb7532d1378d3b9083d9dfc9.jpg)

![](dt=2026-03-17/ht=12/36ba00c394895a2097d7933b7da5a11cb11f9ade3382958d22088f95c96fedcb.jpg)

wavelength of $\lambda = 1,541.7\mathrm{nm}$ with an intrinsic (extrinsic) cavity mode coupling rate of $\kappa_{\mathrm{i,o}} / 2\pi = 0.80\mathrm{GHz}$ ( $\kappa_{\mathrm{e,o}} / 2\pi = 0.81\mathrm{GHz}$ ). As shown in Fig. 2a, the thermal Brownian motion of the mechanical modes of the OMC cavity (primarily caused by optical absorption heating) can also be observed in the noise power spectral density of the reflected optical signal when measured on a photodiode.

The noise power spectral density exhibits a Lorentzian profile for each mechanical mode with a central frequency of $\omega_{\mathrm{m}}$ and a linewidth of $\kappa_{\mathrm{m}} = \kappa_{\mathrm{i,m}} + \gamma_{\mathrm{om}}$ , where $\kappa_{\mathrm{i,m}}$ is the intrinsic linewidth of the mechanical mode and $\gamma_{\mathrm{om}} \approx 4n_{\mathrm{c}}g_{\mathrm{om}}^{2} / \kappa_{\mathrm{o}}$ is the back-action damping when pumping on the red sideband in the resolved-sideband limit[25].

We fit the increase in the mechanical linewidth of each mode as a function of optical power to find the optomechanical coupling for each mode (see Methods), the results of which are in good agreement with our design (Fig. 1c).

We perform microwave spectroscopy of the device using the on-chip coplanar waveguide (CPW) as shown in Fig. 1d, e. This waveguide has been designed to have direct coupling to the qubit (see Methods), allowing for microwave measurements of the transmon qubit response. Figure 2b shows the corresponding measured microwave spectrum,

Nature | Vol 588 | 24/31 December 2020

601

![](dt=2026-03-17/ht=12/168841401e6723b8983f4ffb44f0305b121fb4adbff72509fff91ff7430ba305.jpg)

![](dt=2026-03-17/ht=12/bda9b8e9a617b834d0fe9e8743aae223b2a380dd2e1964a4cf9407a1c4662d31.jpg)

![](dt=2026-03-17/ht=12/8dcddc99df894814f6c73825f49489e2d7534c9368744659330bd83d9bba5f98.jpg)

where the qubit transition frequency is flux-tuned by a current-biased external coil. The electrically coupled acoustic modes are identifiable as avoided mode-crossings in the spectrum, and the measured acoustic mode frequencies are in agreement with the optomechanical spectroscopy measurements, verifying the presence of four hybridized acoustic modes. In particular, the observed acoustic modes at 5.159 GHz and 5.23 GHz show spectrally
resolved mode splitting, indicative of strong coupling with the qubit. Owing to the larger value of $g_{\mathrm{om}}$ for the acoustic resonance at $\omega_{\mathrm{m}} / 2\pi = 5.159 \mathrm{GHz}$ , we have chosen this mode (hereafter referred to as the mechanical mode) for the transduction experiments presented below.

The initial stage in the microwave-to-optics transduction process is the qubit-mechanics swap operation. To perform the swap, we first characterize the qubit in the time domain using pulsed excitation through the CPW and dispersive state readout via an integrated microwave readout resonator (see Fig. 1d, e and Methods). Figure 3a shows the measured Rabi oscillation of the qubit.

Using the Rabi curve to calibrated drive pulses, we measure the qubit's free decay profile and Ramsey interference fringes to find the lifetime $(T_{1,\mathrm{q}} = 522\pm 9$ ns) and coherence time $(T_{2,\mathrm{q}}^{*} = 678\pm 45$ ns) at a bias point $(\omega_{\mathrm{q}} / 2\pi = 5.1\mathrm{GHz})$ near the mechanical mode. To perform a coherent qubit-mechanics swap operation, we control the effective interaction time by exciting the qubit and then rapidly tuning it into, and then out of, resonance with the mechanical mode.

Realizing a high-fidelity swap operation requires a qubit frequency shift that is several times larger than the qubit-mechanics bare interaction rate $g_{\mathrm{pe}}$ , with a frequency transition time that is much shorter than $1 / g_{\mathrm{pe}}$ . We realize this rapid qubit frequency shift by applying a sharp pulse (15 ns rise/fall time) on the CPW line which is detuned with respect to the qubit by $50\mathrm{MHz}$ . This tunes the qubit frequency by $10\mathrm{MHz}$ owing to the a.c.-Stark shift, making it resonant with the mechanical mode for the duration of the pulse.

Figure 3b shows a measurement of the qubit's excited-state population after exciting the qubit and applying the Stark-shift pulse for a variable amount of time. From the observed vacuum Rabi oscillations between qubit and mechanics, we find an optimal Stark-shift pulse duration of $T_{\mathrm{swap}} = 104$ ns for realizing a qubit-mechanics swap (in agreement with the value of $g_{\mathrm{pe}}$ from spectroscopy), with a single-phonon initialization probability of $\eta_{\mathrm{swap}} = (75\pm 3)\%$ .

We also use a pair of swaps with a variable time delay between them to measure the phonon lifetime of the mechanical mode, finding a value of $T_{\mathrm{I,m}} = 357\pm 25$ ns corresponding to an energy decay rate of $\kappa_{\mathrm{m,T}} / 2\pi = 446\mathrm{kHz}$ (the optomechanically measured intrinsic linewidth is $\kappa_{\mathrm{i,m}} / 2\pi = 1\mathrm{MHz}$ , indicating spectral diffusion of the mechanical mode).

We now proceed to mechanics-to-optics state transfer as the last stage in the transduction process. We realize this by selectively turning on the 'beam-splitter' optomechanical interaction with a red-detuned laser pulse at $\omega_{\mathrm{drive}} = \omega_{\mathrm{c}} - \omega_{\mathrm{m}}$ . This results in an up-conversion process, in which the phonons in the mechanical mode are mapped to anti-Stokes scattered photons at the centre frequency $(\omega_{\mathrm{c}})$ of the optical cavity.

We use this method to directly measure the phonon occupancy of the mechanical mode by passing the reflected pulsed light from the device through a series of narrowband tunable filters aligned to the optical cavity (to eliminate the unscattered portion of the pump laser at $\omega_{\mathrm{drive}}$ ), and detecting the scattered photons with a single-photon detector[32].

Figure 4a, b shows the full transduction sequence which consists of an initial qubit drive by a resonant microwave $\pi$ -pulse, followed by a qubit-mechanics swap, and finally the parametric optomechanical conversion of phonons in the mechanical mode to photons. Repeating the experiment with and without qubit drive, we find the scattered photon counts in both cases, which correspond to the excited and ground states of the qubit (see Fig. 4c). The data shows a statistically significant separation between the two cases, indicating detection of transduced optical photons from the qubit.

We find the overall efficiency and added noise (referred to the qubit) of the transduction scheme to be $n_{\mathrm{t}} = P_{\pi} - P_{0} = (0.88 \pm 0.16) \times 10^{-5}$ and $n_{\mathrm{add}} = P_{0} / n_{\mathrm{t}} = 0.57 \pm 0.2$ respectively. These values are calibrated assuming the qubit acts as a single (microwave) photon source, and are consistent with the independently calibrated qubit-mechanics swap operation and optomechanical readout (see Methods).

As a further verification, we repeat the transduction process while varying the duration of the resonant qubit drive, and observe the previously measured Rabi oscillations of the qubit population in the detected optical photon counts (see Fig. 4d).

Performing optomechanical thermometry without the qubit drive, we have verified a non-zero thermal phonon occupancy as the source of added noise $(n_{\mathrm{add}})$ in the transduction process (see Methods). The measured residual phonon occupancy can be reduced by using a shorter optical readout pulse, which points to optical absorption heating as the likely origin. This creates a trade-off in our system, where the noise can be reduced further at the cost of a lower phonon-to-photon conversion efficiency.

The measured total readout efficiency $(\eta_{\mathrm{ro}} = \eta_{\mathrm{t}} / \eta_{\mathrm{swap}} = (1.18 \pm 0.26) \times 10^{-5})$ is a product of this intrinsic efficiency $(\eta_{\mathrm{i,ro}} \approx 10^{-3})$ and the external optical collection efficiency of our measurement set-up $(\eta_{\mathrm{e,ro}} \approx 10^{-2})$ . We note that the measured intrinsic efficiency is significantly lower than for similar previous optomechanical experiments, due in large part to a faster intrinsic damping rate of the mechanical modes.

The source of this excess mechanical mode damping is currently

Article

602

Nature | Vol 588 | 24/31 December 2020

under investigation. In addition, the repetition rate of the current measurements, $R = 100\mathrm{Hz}$ , is currently limited by the recovery time of the superconducting circuitry following the application of the readout laser pulse. We have verified optically induced quasiparticle generation and quasiparticle decay time as the main mechanisms limiting the repetition rate (see Methods).

Although these results provide direct evidence for transduction of a qubit's excitation into an optical photon, the low photon flux in our experiment does not allow for a direct verification of quantum statistics in the emitted light field. Looking ahead, we identify several avenues for improvements in device parameters, which would enable this task and further pave the way for system-level demonstrations of remote entanglement of superconducting qubits.

The transduction repetition rate may be improved by using qubit electrodes formed from niobium with much shorter quasiparticle lifetimes than the current aluminium electrodes $^{33}$ , and elimination of the insertion loss in the optical filtering of the current set-up would increase the external efficiency of converted photons by nearly two orders of magnitude. Further, using a material such as lithium niobate with a larger piezoelectric coupling rate would allow for devices with a larger energy participation in the optomechanical cavity, translating to a larger intrinsic readout efficiency.

Finally, a lower added noise is achievable by reducing optical absorption using surface passivation $^{34}$ and by realizing better thermal contact between the localized acoustic mode and the cold environment $^{35,36}$ . Adoption of these techniques is expected to improve transduction efficiency (to $\eta_{\mathrm{t}} \approx 0.1$ ) while maintaining a low added noise value ( $n_{\mathrm{add}} \lesssim 0.1$ ).

With these improveme
nts, observation of anti-bunching from transduced photons and optically mediated entanglement generation between remote superconducting qubits should be possible, ultimately leading to new applications for superconducting processors in optical quantum networks.

# Online content

Any methods, additional references, Nature Research reporting summaries, source data, extended data, supplementary information, acknowledgements, peer review information; details of author contributions and competing interests; and statements of data and code availability are available at https://doi.org/10.1038/s41586-020-3038-6.

Publisher's note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

© The Author(s), under exclusive licence to Springer Nature Limited 2020

Nature | Vol 588 | 24/31 December 2020

603

# Article

# Methods

# Fabrication

The fabrication process for piezoelectric resonator, OMC cavity, and superconducting circuit elements are illustrated in Extended Data Fig. 1. We start with a 4-inch silicon-on-insulator (SOI) wafer with the following specifications: silicon device layer, float zone grown, $220\mathrm{nm}$ thick, resistivity $\rho \geq 5\mathrm{k}\Omega$ cm; buried oxide layer, $3\mu \mathrm{m}$ thick, silicon dioxide; silicon handle, Czochralski grown, $750\mu \mathrm{m}$ thick, $\rho \geq 5\mathrm{k}\Omega$ cm.

We then perform the following fabrication steps: (i) sputter deposit $300\mathrm{-nm}$ -thick $c$ -axis AlN piezoelectric film (grown by OEM group; stress $T = +55\mathrm{MPa}$ ; (002) X-ray diffraction peak of full-width at half-maximum $1.79^{\circ}$ ) and dice the wafer into $1\mathrm{cm} \times 1\mathrm{cm}$ chips. The following steps are each carried out using e-beam lithography unless noted otherwise.

(ii) Niobium marker deposition (liftoff) and AlN trench etch (Oxford Plasmalab 100 etcher; Ar:Cl₂ = 40:80 standard cubic centimetres per minute (sccm); RF power = 120 W; inductively coupled plasma (ICP) power = 600 W; d.c. bias = 220 V). In this step, we etch only a small area (width ~100 nm) at the perimeter of the AlN region of the transducer. This dry etch step ensures that the boundaries of the AlN transducers are precisely defined via dry etching, while still making sure that the silicon device layer is undamaged for the rest of the chip.

This is important for achieving optical, mechanical and microwave resonances with high quality factors. (iii) Conformal deposition of a hard mask of $\mathrm{SiO}_x$ via plasma-enhanced chemical vapour deposition (PECVD). (iv) Patterning of the $\mathrm{SiO}_x$ mask via dry etching (Oxford Plasmalab 100 etcher; $\mathrm{C_4F_8:SF_6} = 80:40$ sccm; RF power = 14 W; ICP power = 1,300 W, time = 6 min). (v) Removal of the remaining AlN on the chip with $\mathrm{H}_3\mathrm{PO}_4$ (heated to $80^{\circ}\mathrm{C}, 85\%$ by weight).

(vi) $\mathrm{SiO}_x$ mask removal by 10:1 buffered oxide etchant (buffered hydrofluoric acid, BHF). At the end of step (vi), we have produced a local AlN piezoelectric box (typical dimension $2\mu \mathrm{m} \times 0.5\mu \mathrm{m}$ ) on the silicon device layer while still maintaining a silicon surface that is smooth outside the AlN box. The following steps (vii-x) follow previously published results for fabricating superconducting qubits on SOI substrates[29].

For the devices used in the final experiment in the cryogenic set-up, the procedure includes an additional step for realizing end-fire fibre coupling as follows. (xi) We use photo-resist to define a 'trench' region of the chip to be etched for fibre access to the devices' optical waveguides. We use a plasma etch to first remove the buried oxide layer in the trench region, and subsequently etch the handle silicon in this area to a depth exceeding $100\mu \mathrm{m}$ . The chips are then cleaned to remove the photoresist and are released in a vapour-HF etch step.

# Qubit and microwave readout design

The superconducting qubits in the experiment are designed to operate in the transmon limit at the resonance frequency of $f_{\mathrm{q}} = 5.7$ GHz with capacitive and tunnelling junction energies equal to $E_{\mathrm{c}} / (\hbar 2\pi) = 292$ MHz and $E_{\mathrm{j}} / (\hbar 2\pi) = 15.5$ GHz, respectively. This is achieved using a total qubit capacitance of $C_{\mathrm{q}} = 66.2$ fF, which includes the IDT contribution along with the junctions' capacitance, estimated at $C_{\mathrm{j}} = 4.6$ fF.

The junction energy corresponds to a pair of (identical) Josephson junctions in the SQUID geometry with each junction's electrode dimensions designed to be $A_{\mathrm{j}} = 290$ nm × 240 nm in area. The readout resonator is designed as a $\lambda / 4$ lumped-element resonator with a resonance frequency of $f_{\mathrm{RO}} = 7.56$ GHz and a (capacitive) coupling rate of $g_{\mathrm{RO}} / 2\pi = 85$ MHz with the qubit.

The CPW geometry is designed to achieve an external coupling rate of $\kappa_{\mathrm{e,q}} / 2\pi = 120\mathrm{kHz}$ and $\kappa_{\mathrm{Ro}} / 2\pi = 5.5\mathrm{MHz}$ to the qubit and the readout resonator, respectively. In addition to direct capacitive coupling to the CPW, the qubit can also radiate energy out the CPW port through the non-resonant readout resonator. The estimated Purcell decay rate for such a process is only $\kappa_{\mathrm{P,q}} / 2\pi = 15\mathrm{kHz}$ .

The qubit and transducers devices were laid out in pairs, with a single CPW designed to provide access to a pair of qubit/readout resonator

devices placed in mirror symmetry at the end of the waveguide (see Fig. 1e). To eliminate hybridization between neighbouring readout resonators due to near-field coupling (estimated to be $2.5\mathrm{MHz}$ from simulations), the pair of readout resonators are intentionally designed to be detuned by $100\mathrm{MHz}$ with respect to each other (7.56 GHz and 7.66 GHz). The fabrication disorder in the qubits' resonance frequency ( $\geq 50\mathrm{MHz}$ from previous experiments and set by variations in junction area) is expected to prohibit hybridization of the neighbouring qubits.

# Optomechanical and piezoelectric simulations

The optical, mechanical and piezoelectric properties of the device are simulated using the finite-element method (COMSOL Multiphysics software package) before fabrication. We start by independently designing a pair of nearly resonant piezoacoustic and OMC cavities (see Extended Data Fig. 2). The piezoelectric coupling rate is extracted by numerical calculation of the overlap integral between the electric field induced by the mechanical mode and the electric field from the qubit applied to the IDT with proper normalization to single-quantum level.

To model the piezoelectric response of the AlN thin film, we have used a charge constant that is a factor of 3 smaller than the typically observed bulk value ( $d_{33}^{\mathrm{bulk}} = 4.96 \, \mathrm{pmV}^{-1}$ ), which is found from fitting room-temperature calibration measurements of thin-film AlN-on-SOI piezoelectric transducers fabricated using our process. The optomechanical coupling rate is calculated by numerical evaluation of surface and bulk contributions in a similar fashion to previous work<sup>30</sup>.

In the final design step, we simulate a structure made by attaching the OMC and piezoacoustic cavity parts and modifying the intermediate section to form a phonon waveguide. The final in-plane dimensions from the simulation are scaled by a factor of 0.96 before device fabrication. This scaling factor is found from calibration measurements for achieving good agreement between the simulated and measured values of optical resonance frequencies of the OMC cavity.

# Phonon waveguide design

Mechanical hybridization between the AlN-on-SOI piezoacoustic resonator and the silicon OMC cavity is achieved by modifying the OMC structure to make a mechanically transparent section between the two components. Extended Data Fig. 3 shows a unit cell of
the nanobeam in this intermediate section in the original OMC design and after modifying it. To make this section transparent to mechanical waves, we have simulated the optical and mechanical band structure of the unit cell while sweeping the ellipticity of the central 'hole' in the nanobeam.

Heuristically, this modification results in a unit cell with a similar dielectric to vacuum ratio (due to the nearly constant area of the holes), which maintains the qualitative form of the optical band structure, whereas the acoustic band structure is modified because of the hard boundary between silicon and vacuum for acoustic waves (acoustic waves cannot travel in vacuum as electromagnetic fields do).

# Disorder and acoustic mode hybridization

The phonon waveguide design presented in the previous section allows for achieving hybridization between the OMC and piezoacoustic cavity modes when the two modes are in resonance. In practice, however, the exact resonance frequency of each component cavity is dependent on variations in the geometry that inevitably happen during the fabrication process. This variation can be reduced by careful calibration of the fabrication process to values $\geq 50\mathrm{MHz}$ . Alternatively the devices can be trimmed post fabrication to achieve the resonance conditions.

In our experiment, we eliminate the need for such measures by realizing a large mechanical inter-mode coupling between the OMC and the piezoelectric resonator via the phonon waveguide, which makes the hybridization a weak function of the detuning between the two. Extended Data Fig. 4 shows the simulated optomechanical and piezoelectric coupling rates of the hybridized modes for different values of the IDT period in the piezoacoustic cavity. Assuming a linear dispersion

for the mechanical modes confined in the piezoacoustic cavity, the $50~\mathrm{nm}$ change of the period in Extended Data Fig. 4 corresponds to approximately a $280\mathrm{MHz}$ change in the detuning between the OMC and piezoacoustic cavity modes. As evident from simulations, while the details of the spectral composition and coupling rates is subject to change, the mechanical mode hybridization is robust to non-zero detunings resulting in qualitatively similar mode spacing and coupling rates for various IDT periods. For the fabricated devices, we have swept the IDT period by $20\mathrm{nm}$ (-110 MHz frequency shift) around the central periodicity of $885\mathrm{nm}$ .

# Measurementset-up

Extended Data Fig. 5 shows a schematic of the measurement set-up. The arrangement of the main optical components is adopted and modified from previous work on phonon counting $^{32,37}$ . A digital delay generator is used to synchronize the generation of the microwave drive, the optical readout pulses generated by optical modulation, and the placement of the detection window in time. Optomechanical readout and characterization is performed with an external cavity diode laser source that is frequency-stabilized using a wavenumber in a computer feedback loop.

The laser light is pre-filtered to avoid laser phase-noise at the mechanical resonance frequency. A phase modulator is used to create optical sidebands for the purpose of coherent driving of the mechanical modes (during characterization) or for creating an optical reference for locking the tunable filter array in the detection path (as part of the transduction sequence).

Readout optical pulses are shaped using an acousto-optic modulator (rise and fall time of $20~\mathrm{ns}$ ) and a pair of mechanical switches (rise/fall time of $100~\mathrm{ns} / 30~\mu \mathrm{s}$ ) cascaded together to achieve high-contrast pulses (extinction $>86$ dB). The reflected optical signal from the device is routed via an optical circulator to a mechanical switch, where it is directed towards two separates paths for performing continuous-wave and pulsed measurements.

For characterization of the acoustic modes of the OMC cavity, the optical reflected signal is routed to the path with an erbium-doped fibre amplifier (EDFA) and a high-speed photodetector (PD). The photo-current from the detector is registered with a analogue-to-digital converter to measure the optical resonance while sweeping the laser frequency. Measurement of the thermal Brownian motion of the mechanical modes is performed using a spectrum analyser (SA) for registering the photo-current when the laser is locked at $\Delta \approx \omega_{\mathrm{m}}$ .

The coherent mechanical response of the device is measured with two-tone spectroscopy, where the pump laser light is modulated with a phase-modulator driven by the output of a vector network analyser (VNA), and the RF component of the photo-current is measured at the input port of the VNA. For the transduction experiment, the second detection path is used for photon counting.

Here, the light is passed through three cascaded high-finesse tunable fibre Fabry-Perot filters (Micron Optics FFP-TF2) placed inside a thermally insulating housing, and is then routed to a single-photon detector (SPD) mounted on the still plate inside the dilution refrigerator. The SPD used in the experiment is a WSi-based superconducting nanowire detector.

The RF output of the SPD is amplified with a cryogenic amplifier at the $50\mathrm{K}$ stage of the dilution fridge and a room-temperature amplifier before detection with a triggered time-correlated single photon counting (TCSPC) detection module.

The fabricated device is cooled to $T_{\mathrm{f}} \approx 15$ mK in a dilution refrigerator, and is shielded from the magnetic environment using a mu-metal shield inserted into the external vacuum can of the cryostat and a cryoperm shield located at the mixing chamber plate. Optical alignment and coupling of laser light to the tapered silicon optical waveguide of a given transducer device is performed using a stack of cryogenic piezo steppers. Microwave signals are routed to the device via a coaxial cable, with thermal grounding to the 4 K and mixing chamber plates of the fridge, and then onto the CPW of a printed circuit board, which is wire-bonded to an on-chip CPW. Reflected microwave signals from

the device are collected using an RF circulator and amplified through an amplifier chain consisting of a high-electron-mobility transistor amplifier (HEMT) at the $4\mathrm{K}$ stage of the fridge and a low noise amplifier (LNA) at room-temperature. Qubit frequency tuning is achieved by flux tuning of the SQUID loop using an external hand-wound coil made from Nb-Ti superconducting wire which is placed several millimetres above the chip. The flux bias current is applied to the coil using a low-noise d.c. source.

# Optomechanical scattering rate and mechanical mode occupancy calibration

We measure the spectral response of the mechanical modes by performing a two-tone spectroscopy technique. In this approach, the laser is locked to the red sideband of the optical cavity $(\Delta = \omega_{\mathrm{m}})$ , and phase modulation is used to generate a pair of optical sidebands, where one of the sidebands is swept across the optical cavity.

For the case where the pump-sideband frequency separation coincides with the mechanical resonance frequency, the optical susceptibility of the cavity is strongly suppressed owing to cancellation of coherent scattering components from the mechanical and optical resonances, in a similar fashion to electromagnetically induced transparency<sup>4</sup>.

We calibrate the optomechanical coupling rate of the mechanical resonance at $\omega_{\mathrm{m}} / 2\pi = 5.159$ GHz by fitting the linewidth from the two-tone spectroscopy as a function of the pump photon number in the optical cavity, finding $g_{\mathrm{om}} / 2\pi = 420$ kHz. The optomechanical scattering rate for other mechanical resonances are found by integrating the area under the thermal Brownian spectrum at each resonance frequency (measured using a spectrum analyser) and using the optomechanical coupl
ing rate of the mode at $\omega_{\mathrm{m}} / 2\pi = 5.159$ GHz as a reference.

The optomechanical coupling rate is independently verified by measuring the photon scattering rates with the pump locked to the red and blue sideband of the optical cavity $(\Delta = \pm \omega_{\mathrm{m}})$ . In this setting, the photon detection rate can be written as,

$$
\Gamma_ {\mathrm {R} / \mathrm {B}} (t) = \Gamma_ {\text {d a r k}} + \eta_ {\kappa} \eta_ {\text {s y s}} \frac {4 g _ {\mathrm {o m}} ^ {2} n _ {\mathrm {c}} (t)}{\kappa_ {\mathrm {o}}} \left(n _ {\mathrm {m}} + \frac {1}{2} (1 \mp 1)\right), \tag {1}
$$

where $\Gamma_{\mathrm{dark}} = 10$ cps is the rate of dark counts, $n_{\mathrm{m}}$ is the occupancy of the mechanical mode, and $\eta_{\kappa} = \kappa_{\mathrm{e,o}} / (\kappa_{\mathrm{i,o}} + \kappa_{\mathrm{e,o}}) \approx 0.5$ is the cavity-to-waveguide coupling efficiency. The external optical efficiency of the system is independently measured to be $\eta_{\mathrm{sys}} = 1.5\%$ , which includes the efficiency of coupling from the lensed fibre to the device ( $\eta_{\mathrm{cplr}} = 65\%$ , one-way), transmission efficiency through the measurement system including the optical filter bank ( $\eta_{\mathrm{tran}} = 3\%$ ), and the quantum efficiency of the SPD ( $\eta_{\mathrm{spd}} = 85\%$ ).

So long as the optomechanical back-action rate is negligible compared with the intrinsic mechanical mode damping rate (that is, $n_{\mathrm{m}}$ is not influenced by the back-action), the difference in red- and blue-sideband scattering rates provides a calibration-free means of determining the per-phonon count rate by using the vacuum contribution (that is, spontaneous Stokes scattering) as a reference.

From the independent measurements of $g_{\mathrm{om}}$ and $\kappa_{\mathrm{o}}$ we estimate that for the optical drive power used in these experiments (2 μW at the chip; $n_{\mathrm{c}} = 44$ ) that the optomechanical back-action is $\gamma_{\mathrm{om}} / 2\pi = 19 \mathrm{kHz}$ , far less than the measured $\kappa_{\mathrm{m},\Pi} / 2\pi = 446 \mathrm{kHz}$ . We can use this to find an independent estimate of the optomechanical readout efficiency of the transducer. The optomechanical readout efficiency as a function of readout time, $\tau_{\mathrm{ro}}$ , is given by

$$
\eta_ {\mathrm {r o}} \left(\tau_ {\mathrm {r o}}\right) = \left(\frac {\gamma_ {\mathrm {o m}}}{\gamma_ {\mathrm {o m}} + \kappa_ {\mathrm {m} , T _ {1}}}\right) \left(1 - \exp \left[ - \left(\gamma_ {\mathrm {o m}} + \kappa_ {\mathrm {m}, T _ {1}}\right) \tau_ {\mathrm {r o}} \right]\right), \tag {2}
$$

which in the long-time limit is just the fraction of 'good' damping, $\gamma_{\mathrm{om}} / (\gamma_{\mathrm{om}} + \kappa_{\mathrm{m},T1})$ . In the small-time limit $(\tau_{\mathrm{ro}} \ll 1 / (\gamma_{\mathrm{om}} + \kappa_{\mathrm{m},T1}))$ , appropriate to the measurements reported here, the optomechanical readout

# Article

efficiency can be simply related to the difference in the blue and red sideband scattering rates, $\eta_{\mathrm{ro}}(\tau_{\mathrm{ro}})\eta_{\kappa}\eta_{\mathrm{sys}}\approx p_{\mathrm{d}}\equiv \int_{0}^{\tau_{\mathrm{ro}}}(\Gamma_{\mathrm{B}}(t) - \Gamma_{\mathrm{R}}(t))\mathrm{d}t,$ where the integration accounts for the fact that the scattering rates may be time-dependent due to the turn-on of the optical pulse, and $p_{\mathrm{d}}$ is the probability of detecting a single photon converted from a single phonon excitation in the mechanical mode.

Similarly, we find the time-averaged (over $\tau_{\mathrm{ro}}$ ) residual thermal noise of the heated mechanical mode occupancy by integrating the red-sideband rate normalized to the per-phonon rate: $\langle n_{\mathrm{m}}\rangle_{\tau} = \int_{0}^{\tau_{\mathrm{ro}}}\Gamma_{\mathrm{R}}\mathrm{d}t / \int_{0}^{\tau_{\mathrm{ro}}}(\Gamma_{\mathrm{B}} - \Gamma_{\mathrm{R}})\mathrm{d}t.$

Extended Data Fig. 6 shows the calculated noise and efficiency from the optomechanical scattering measurements as a function of the pulse integration time $(\tau_{\mathrm{ro}})$ with the qubit decoupled from the transducer. As is evident, the time-averaged phonon noise in the mechanical resonator starts at a small value and grows with the optical pulse measurement time, in agreement with previous observations of optical absorption heating in silicon OMC devices. The readout efficiency also increases with integration time, leading to a trade-off between noise and efficiency.

For an optical readout integration time of $\tau_{\mathrm{ro}} = 38$ ns, equivalent to that used in the qubit transduction data of Fig. 4, the extracted average mechanical mode occupancy and optomechanical readout efficiency are $\langle n_{\mathrm{m}}\rangle_{\tau_{\mathrm{ro}}} = 0.64 \pm 0.15$ and $\eta_{\mathrm{ro}}(\tau_{\mathrm{ro}}) = (0.88 \pm 0.13) \times 10^{-5}$ , respectively.

These 'vacuum-calibrated' values are consistent with the corresponding 'qubit-calibrated' values of $\eta_{\mathrm{ro}}^{\mathrm{q}} = (1.18 \pm 0.26) \times 10^{-5}$ and $\langle n_{\mathrm{m}}\rangle^{\mathrm{q}} = n_{\mathrm{add}}\eta_{\mathrm{swap}} = 0.43 \pm 0.17$ .

# Quasiparticle trapping

Absorption of pump laser light in our experiment leads to generation of excited electrons (that is, quasiparticles, QP) from broken Cooper pairs due to the large energy of the infrared photons with respect to the energy gap of superconducting aluminium, $\hbar \omega_{\mathrm{c}} \gg 2\Delta_{\mathrm{gap}}$ .

The generated non-equilibrium QP population in the transmon's capacitive leads gives rise to a dissipative component in the admittance of Josephson junctions, which reduces the lifetime of the qubit, and in turn compromises the ability to perform coherent gate operations in the microwave domain. This excess QP population eventually decays to the steady-state value via electron-phonon-mediated pair recombination with a characteristic lifetime set by the material properties of the superconducting film and the substrate<sup>38</sup>. Extended Data Fig.

7 shows the QP relaxation measured by sending optical pulses into the transducer device, followed by a delayed measurement of the qubit Rabi curve contrast performed via the microwave readout resonator. As is evident, for the pulse duration and optical power used in the transduction measurements, the transmon qubit requires a delay of approximately 8 ms between pulses to allow for full recovery (hence our limited repetition rate in the transduction measurements of $R = 100\mathrm{Hz}$ ).

To verify QP generation as the source of light-induced decoherence, we have performed electrical QP relaxation measurements, in which

we inject QPs into the qubit by creating a large off-resonant a.c. voltage across the qubit junctions $^{38}$ via a pulsed drive applied on the CPW. Using the measured decay rate of the Rabi oscillations of the qubit as an indicator of the qubit's coherence time after the QP injection with a variable delay, we extract a QP lifetime of 1.5 ms for our device. The measured value is in qualitative agreement with previously reported values for thin-film aluminium and is qualitatively consistent with results of the optics measurement.

In the next step, we try to reduce QP relaxation lifetime via vortex trapping $^{38}$ . To do this, the device is warmed to above the superconducting transition temperature, a magnetic field (normal to substrate) is applied via the flux-tuning coil and the device is subsequently cooled down to base temperature. This results in creation of vortices in electrodes with the number of vortices approximately equal to the magnetic flux threading a square of electrode width divided by flux quantum.

At the base temperature, we turn off the magnetic field slowly, which is expected to pin the vortexes to defects in the thin film aluminium. We then repeat the electrical measurement of QP relaxation, finding a lifetime of $320~\mu \mathrm{s}$ for a cooling magnetic field of 15 Gauss (co
rresponding to $\sim 75$ vortices per square in the transmon electrodes). As a final verification, we repeat the light-induced QP generation experiment in this setting, finding a recovery time of approximately 2 ms for the transmon qubit.

The electrical and optical measurements consistently point to a factor of $-5$ enhancement in the measurement repetition rate via vortex trapping of QPs.

# Data availability

The data that support the findings of this study are available from the corresponding author (O.P.) upon reasonable request.

Acknowledgements We thank M. Shaw, J. Banker, H. Ren, E. Kim and X. Zhang for their various contributions to this work. This work was supported by the ARO/LPS Cross Quantum Technology Systems programme (grant W911NF-18-1-0103), the Institute for Quantum Information and Matter, an NSF Physics Frontiers Center (grant PHY-1125565) with support of the Gordon and Betty Moore Foundation, and the Kavli Nanoscience Institute at Caltech. M.M. (A.S.) acknowledges support from a KNI (IQIM) Postdoctoral Fellowship.

Author contributions All authors contributed to the concept and planning of the experiment, the device design and fabrication, the measurements and analysis of data, and the writing of the manuscript.

Competing interests The authors declare no competing interests.

# Additional information

Correspondence and requests for materials should be addressed to O.P.

Peer review information Nature thanks Konrad Lehnert and the other, anonymous, reviewer(s) for their contribution to the peer review of this work. Peer review reports are available.

Reprints and permissions information is available at http://www.nature.com/reprints.

![](dt=2026-03-17/ht=12/29a689ab45fb71aa39a889c872fea8079f1c97b96c630e9af9bd4fd7679ff9de.jpg)

a $\omega_{\mathrm{m}} / 2\pi = 5.03\mathrm{GHz}$

${g}_{\mathrm{{pe}}}/{2\pi } = {5.8}\mathrm{{MHz}}$

![](dt=2026-03-17/ht=12/669c04f7b8530745d635bf9546c680b983ae66f6f523048077342a46d97e0461.jpg)

![](dt=2026-03-17/ht=12/42fb0461f24f0ee889a7939e448c4ac6facfcb2625293c1694bf7f7a4d6f70b5.jpg)

![](dt=2026-03-17/ht=12/c68af4429f9848dc013351c72f8eb460ac8c2ad08ecf9ddf1aa86a219dd1f311.jpg)

![](dt=2026-03-17/ht=12/c18f99e7ff1f3c2aabb33c4e86aeb53aa302f30e4159221bda4ced0c44b36da2.jpg)

# Extended Data Fig. 2 | Optomechanical and piezoelectric design.

a, Simulated mechanical mode shape (deformation) and electric voltage (colour) of the piezoacoustic cavity mode of interest with an IDT period of 930 nm and a beam width of 600 nm. b, Simulated optical (top) and mechanical (bottom) mode profile of an OMC designed with the mechanical mode near 5 GHz. c, Simulated optical (top) and mechanical (bottom) mode profile of the full piezo-optomechanical transducer device formed by attaching the piezoacoustic cavity of (a) and the OMC cavity of (b) through a phonon waveguide section in which the mirror holes in the OMC cavity are modified

nearest the piezoacoustic cavity. The optomechanical and piezoelectric coupling rates listed are calculated for the hybridized mode with the largest optomechanical coupling. d, Radii of the patterned holes along the nanobeam OMC cavity and phonon waveguide section. $d_{1}(d_{2})$ designates the hole diameter normal (parallel) to the nanobeam's long axis. The optical cavity region (shaded 13 central holes) is located between a phonon/photon mirror (left seven holes) and a photon mirror/phonon waveguide section (right seven holes).

Article

![](dt=2026-03-17/ht=12/8a316fd1c8d3ccd166b271d7fe97a68d456710f63c4741ffb2bd00b9bdd22276.jpg)

![](dt=2026-03-17/ht=12/67844862df8505079d8f53a6c214c701406e83f6be7adf5a5149f2cf778bf165.jpg)

![](dt=2026-03-17/ht=12/03ebe0ffe9b037fb05dacf32b93c77f87c7a654373a8322071ba44f00951337b.jpg)

![](dt=2026-03-17/ht=12/9d6c9e021bea34bf244b9de6841da5727d29ee2d6635e8bf613221e6cc6fa107.jpg)

![](dt=2026-03-17/ht=12/2bedf151717b5eced2a0abcd41e01e3a9bcc89b6ec3e689efa43d6dd47949d32.jpg)

![](dt=2026-03-17/ht=12/e4de71daec16dc66fa864c825a2dbd4ba9f0bd008506203a6663984348f2d819.jpg)

# Extended Data Fig. 3| Design of the phonon waveguide unit cell.

a, Schematics of the original (top) and modified (bottom) unit cell of the OMC cavity mirror section adjacent to the phonon waveguide. The dimensional parameters are equal to $d_{1} = 366 \mathrm{~nm}$ ( $d_{1} = 295 \mathrm{~nm}$ ) and $d_{2} = 205 \mathrm{~nm}$ ( $d_{2} = 320 \mathrm{~nm}$ ) for the original (modified) holes. The nanobeam parameters ( $h = 220 \mathrm{~nm}$ , $a = 436 \mathrm{~nm}$ , $b = 529 \mathrm{~nm}$ ) are identical for both cases. b, Simulated mechanical band structures. The dashed line marks a nominal mechanical resonance frequency of $f_{\mathrm{m}} = 5.3 \mathrm{GHz}$ for the decoupled OMC and piezoacoustic cavity modes. The intersection of the dashed line and the energy band in the bottom

acoustic band-structure plot for the modified hole structure allows for guiding of acoustic waves between the piezoacoustic and OMC cavity. c, Optical band structure for TE-like modes of the OMC cavity, again with the top plot being for the original OMC cavity and the bottom plot for the OMC cavity with modified holes. The solid black line marks the light line, and the dashed line refers to a nominal optical resonance frequency of the fundamental mode of the OMC cavity at a frequency $f_{0} = 193$ THz. Unlike the acoustic mode case, the hole modifications in the OMC cavity actually increase the optical bandgap, further suppressing optical radiation into the phonon waveguide.

![](dt=2026-03-17/ht=12/e65c9bc4fc5d4ca6e496c9dba3f6daeaa9f0752f4ad079a4fbf1ec750bf313ae.jpg)

![](dt=2026-03-17/ht=12/6246d2146bb80ce36b95b6bb52df8eb0d3a78688bb4fe5a529b84d813b66e9b8.jpg)

![](dt=2026-03-17/ht=12/dd1825d72013644693d1df19779c22e52ff844f3ec4c928f293cacff02175e70.jpg)

![](dt=2026-03-17/ht=12/36ea7b5329a0a4862865d07573787fbc17d1ed69f760378e33f38e67624d4fbe.jpg)

![](dt=2026-03-17/ht=12/4fb00de52807a179072acfec2cc95e109086f94c2f0ae806bc0b3c030c9770cb.jpg)

![](dt=2026-03-17/ht=12/7f980e70ad6e60ad6bd1063ee0bcf08111dd212349d10a76519fdb2493841239.jpg)

Article

![](dt=2026-03-17/ht=12/6b809a8d9321968275fbd7715a26fc71cf6109ab04237520edd1fc3ea7e9c5d4.jpg)

![](dt=2026-03-17/ht=12/0314c052b74dbc08a36e87d70f95b1ae742664323d511022816d92e02171af51.jpg)

Article

![](image)
0d7222ec969d1a6b5b5c9.jpg)