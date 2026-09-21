# Optically heralded microwave photon addition

Received: 28 September 2022

Accepted: 6 June 2023

Published online: 20 July 2023

![](dt=2026-03-15/ht=11/2b5c4d77e73846f4c3caa76eb1600519ab2138c897fb7578ade2cb0eae4a9fad.jpg)

Check for updates

Wentao Jiang $①, 3$ , Felix M. Mayor $①, 3$ , Sultan Malik<sup>1</sup>, Raphaël Van Laer<sup>1,2</sup>, Timothy P. McKenna<sup>1</sup>, Rishi N. Patel<sup>1</sup>, Jeremy D. Witmer<sup>1</sup> & Amir H. Safavi-Naeini

Photons with optical frequencies of a few hundred terahertz are perhaps the only way to distribute quantum information over long distances. Superconducting qubits, which are one of the most promising approaches for realizing large-scale quantum machines, operate on microwave photons at frequencies that are $\sim 40,000$ times lower. To network these quantum machines across appreciable distances, we must bridge this frequency gap. Here we implement and demonstrate a transducer that can generate correlated optical and microwave photons.

We use it to show that by detecting an optical photon we generate an added microwave photon with an efficiency of $\sim 35\%$ . Our device uses a gigahertz nanomechanical resonance as an intermediary, which efficiently couples to optical and microwave channels through strong optomechanical and piezoelectric interactions. We show continuous operation of the transducer with $5\%$ frequency conversion efficiency, input-referred added noise of $\sim 100$ , and pulsed microwave photon generation at a heralding rate of $15\mathrm{Hz}$ .

Optical absorption in the device generates thermal noise of less than two microwave photons. Improvements of the system efficiencies and device performance are necessary to realize a high rate of entanglement generation between distant microwave-frequency quantum nodes, but these enhancements are within reach.

Manipulating and transmitting quantum states with higher fidelity and at larger scales will enable technologies that promise breakthroughs in sensing, communication and computation<sup>1-3</sup>. Over the past two decades, our ability to manipulate the states of photons in superconducting circuits has advanced rapidly and led to demonstrations of quantum advantage for certain computational tasks<sup>4-6</sup>. Separately, the first quantum networks have been realized based on the transmission of quantum states and the distribution of entanglement over a small number of nodes using optically coupled qubits<sup>7,8</sup>. Optical interconnects between superconducting quantum machines with advanced computational and error-correction capabilities<sup>9,10</sup> would substantially

accelerate the development and deployment of quantum networks. Moreover, microwave quantum processors would be beneficiaries of the quantum networks that would support distributed quantum computing $^{11}$ , sensing $^{12,13}$ and secured communications $^{14}$ . Unfortunately, in contrast to ions, atoms and semiconductor defect centres, superconducting circuits lack a natural optical transition that would generate entanglement between propagating optical photons and their internal states. To compensate for this deficiency, a highly efficient and low-noise quantum transducer needs to be developed.

Quantum transducers that connect microwave and optical photons have used a variety of physical processes, including optomechanical,

<sup>1</sup>Department of Applied Physics and Ginzton Laboratory, Stanford University, Stanford, CA, USA. <sup>2</sup>Present address: Department of Microtechnology and Nanoscience, Chalmers University of Technology, Gothenburg, Sweden. <sup>3</sup>These authors contributed equally: Wentao Jiang, Felix M. Mayor.

e-mail: safavi@stanford.edu

nature physics

Article

https://doi.org/10.1038/s41567-023-02129-w

Nature Physics

electro-optical, magneto-optical interactions and atomic degrees of freedom $^{15}$ . In optomechanical transducers, the interaction between light and motion via radiation pressure and electrostriction offers the nonlinearity necessary for frequency conversion $^{16}$ . Mechanical vibration is then converted to a microwave signal with either parametric electromechanical coupling $^{17,18}$ or the piezoelectric effect $^{19,20}$ .

Steady progress in the field has led to constant improvements in transduction efficiencies and added noise $^{21-23}$ , culminating in remarkable recent demonstrations of superconducting qubit state readout $^{24,25}$ . Although direct quantum frequency transduction is approaching the $50\%$ efficiency required for non-zero quantum capacity $^{26}$ , the pump powers required for such efficiencies add considerable noise during the conversion process.

On the other hand, these transducers are equally capable of generating correlated optical-microwave photon pairs, allowing the trade off between pair generation rate and added thermal noise.

In this Article we split optical photons into correlated pairs of optical and microwave photons in a quantum transducer and herald microwave photons by detecting optical photons. Although we do not observe non-classical correlations between light and microwave, our demonstration is an important step towards realizing heralded entanglement generation between distant superconducting quantum machines that may form nodes in a quantum network[27]. We send a laser pulse with the sum frequency of the optical and microwave frequencies to the transducer.

Through spontaneous downconversion, the input optical photon is converted to a pair of optical and microwave photons. The phonon half of this pair efficiently radiates from the device into an output line as a microwave-frequency electromagnetic signal, generated by a strong engineered piezoelectric coupling. The light is sent onto a single photon detector and, upon detection, we herald the microwave photon. Finally, we characterize the state of the microwave field using linear detection tomography and verify the presence of the additional photon.

The two physical processes in the transducer are the nonlinear optomechanical interaction between the optical mode and the mechanical mode, and the linear piezoelectric interaction between the mechanical mode and the microwave mode (Fig. 1a). The nonlinearity that facilitates the frequency conversion process is governed by the optomechanical interaction Hamiltonian $\hat{H}_{\mathrm{om}} = \hbar g_{\mathrm{o}}\hat{a}^{\dagger}\hat{a} (\hat{b} +\hat{b}^{\dagger})$ where $\hat{a} (\hat{b})$ is the annihilation operator for the optical (mechanical) resonance.

The optomechanical coupling rate $g_{\mathrm{o}}$ represents the cavity frequency uncertainty from the zero-point motion in the mechanical mode. As shown in Fig.

1b, when the system is pumped with a red-detuned laser, the interaction can be described by the beamsplitter Hamiltonian $\hat{H}_{\mathrm{bs}} = \hbar G_{\mathrm{o}}(\hat{a}\hat{b}^{\dagger} + \hat{a}^{\dagger}\hat{b})$ while a blue-detuned laser implements a two-mode squeezing Hamiltonian $\hat{H}_{\mathrm{rms}} = \hbar G_{\mathrm{o}}(\hat{a}^{\dagger}\hat{b}^{\dagger} + \hat{a}\hat{b})$ We define the linearized coupling rate $G_{\mathrm{o}} = \sqrt{n_{\mathrm{a}}g_{\mathrm{o}}}$ where $n_{\mathrm{a}}$ is the intracavity pump photon number.

The operator $\hat{Q}$ now represents the sideband component of the optical mode that is resonant with the cavity. The piezoelectric interaction between the mechanical and microwave resonance is described by $\hat{H}_{\mathrm{pe}} = \hbar g_{\mu}(\hat{b}\hat{c}^{\dagger} + \hat{b}^{\dagger}\hat{c})$ ,where $g_{\mu}$ is the coupling rate, and $\hat{c}$ is the lowering operator of the microwave mode.

We operate our transducer in a fast-cavity limit, where the optical (microwave) linewidth $\kappa_{\mathrm{o}}(\kappa_{\mu})$ is much larger than the corresponding coupling rate, $\kappa_{\mathrm{o}} \gg G_{\mathrm{o}}(\kappa_{\mu} \gg
g_{\mu})$ . In this limit, both the optical and microwave subsystems act approximately as a broad Markovian bath for the mechanical mode. This means that the resonant phonons decay at a rate of $\gamma_{\mu} = 4g_{\mu}^{2} / \kappa_{\mu}$ into the microwave transmission line.

Similarly, for a red-side pump, the phonons decay directly into the photon loss channels at a rate of $\gamma_{\mathrm{om}} = 4G_{\mathrm{o}}^{2} / \kappa_{\mathrm{o}}$ . The peak on-chip conversion efficiency is given by the ratio between external coupling rates and the total loss rates:

$$
\eta = \eta_ {\mathrm {o}} \eta_ {\mu} \frac {4 \gamma_ {\mathrm {o m}} \gamma_ {\mu}}{\left(\gamma_ {\mathrm {i}} + \gamma_ {\mathrm {o m}} + \gamma_ {\mu}\right) ^ {2}} \tag {1}
$$

where $\gamma_{\mathrm{i}}$ is the intrinsic loss rate of the mechanical mode, and $\eta_{\mathrm{o}}\equiv \kappa_{\mathrm{o,e}} / \kappa_{\mathrm{o}}$ $(\eta_{\mu}\equiv \kappa_{\mu ,\mathrm{e}} / \kappa_{\mu})$ is the external coupling efficiency of the optical (microwave) mode (Fig. 1a). For a blue-side pump, the interaction induces a non-degenerate parametric amplification rate of $\gamma_{\mathrm{om}}$ for the mechanical resonator, with correlated photons emitted at optical frequency.

To achieve higher conversion efficiency and bandwidth, larger $\gamma_{\mathrm{om}}$ and $\gamma_{\mu}$ are required. This translates to higher optomechanical and piezoelectric coupling coefficients. We found that no single material system is optimal with respect to all the needs of the converter. As such, we pursued a heterogeneous integration approach that combines materials with good optomechanical and piezoelectric properties.

In addition to the materials integration challenge, this raises a challenge in design, as we must co-design the optomechanical and piezoelectric constituents of the transducer to maximize the modal overlaps while maintaining small mode volumes for high interaction rates. We implement the transducer by combining highly piezoelectric thin-film lithium niobate (LN)[28] with thin-film silicon (Si), which has been shown to have strong optomechanical coupling and low loss[29].

We use a silicon optomechanical crystal (OMC) to co-localize the optical and mechanical resonances in a wavelength-scale volume[30], and engineer the mechanical mode to be partially extended and strongly hybridized with a Si-LN hybrid piezoelectric mode[24]. The orientation between the piezoelectric resonator and the Si OMC is chosen to maximize the mechanical hybridization. The full transducer structure is released and supported by one-dimensional (1D) silicon phononic shields to minimize unwanted mechanical loss (Fig. 1c,d).

To couple the phonons to microwaves, we pattern aluminium electrodes on the LN and run these over the phononic shields that suspend the transducer. Because of the vastly different dimensions of the piezo-optomechanical element $(\sim 10\mu \mathrm{m})$ and the microwave circuit $(\sim 10\mathrm{mm})$ , we fabricate them on separate chips that we then combine by wirebonding (Fig. 1f). The microwave resonance is formed by a standing wave in a high-impedance (high- $Z$ ) microwave coplanar waveguide (Fig. 1b,e) arising due to impedance mismatch with the output line.

A niobium titanium nitride (NbTiN) thin film on high-resistivity silicon substrate is patterned into nanowires to support travelling waves with a characteristic impedance of $Z = 1,000\Omega$ . The large kinetic inductance from the nanowires enables high impedance and magnetic frequency tunability[31]. Finally, by using NbTiN, with its short quasi-particle lifetime, for the microwave resonator, and by fabricating the microwave subsystem on a separate chip, we mitigate some of the effects associated with absorption of stray optical radiation[24].

We characterize the transducer at the mixing chamber plate in a dilution refrigerator where the temperature is $T \lesssim 10 \, \mathrm{mK}$ – the same environment in which superconducting circuits and qubits are operated. We use a lensed fibre to focus light into an on-chip photonic waveguide, to which the optical cavity is evanescently coupled. We first measure the linear scattering parameters of the transducer with a red-detuned continuous-wave pump. We use $4.4\text{-}\mu \mathrm{W}$ on-chip pump power, corresponding to an intracavity photon number $n_{\mathrm{a}} = 230$ .

We sweep a weak probe tone across the optical resonance, generated by electro-optic modulation of the pump light by the microwave signal from a vector network analyser (VNA). The reflected light is amplified and detected by a high-speed photodetector and subsequently the VNA. We obtain the optical sideband response and extract the phase (Fig. 2a). We observe electromagnetically induced transparency (EIT) $^{16}$ and fit the response using input-output theory, which gives a single-photon optomechanical coupling rate of $g_{\mathrm{o}} / 2\pi = 410 \, \mathrm{kHz}$ .

The mechanical mode has frequency $\omega_{\mathrm{m}} / 2\pi = 3.596 \, \mathrm{GHz}$ . We find that the optical cavity is nearly critically coupled with a total linewidth of $\kappa_{\mathrm{o}} / 2\pi = 1.12 \, \mathrm{GHz}$ . Subsequently, we characterize the piezoelectric interaction in the transducer by a microwave reflection measurement. We obtain the scattering parameter $S_{\mu \mu}$ and plot its magnitude and phase in Fig. 2b.

We find a piezoelectric coupling rate $g_{\mu} / 2\pi = 420 \, \mathrm{kHz}$ and extract an intrinsic mechanical linewidth of $\gamma_{\mathrm{i}} / 2\pi = 1.1 \, \mathrm{MHz}$ . During

Article

https://doi.org/10.1038/s41567-023-02129-w

Nature Physics

![](dt=2026-03-15/ht=11/24f4a9242586d57a09e43b62af7c81a6dbc08afa4b06869fe23b4d735e7ee9d4.jpg)

![](dt=2026-03-15/ht=11/085d8a387b33e634345762beafe7a91f291ea2f3224879fdd375f68480d461b4.jpg)

![](dt=2026-03-15/ht=11/0a160b1aa4b4708a6ecffbc2416b4bbb0cd002ea90ba740ceeeb31673f0df2fe.jpg)

![](dt=2026-03-15/ht=11/55405104cebebd58895f3697b0f3b9aa0a5e9c1e11c1f696df9ced82e1fc2154.jpg)

![](dt=2026-03-15/ht=11/effb77fa4292e621748501d9a4100751e79d0bf743f5948719a97f283b1bb419.jpg)

![](dt=2026-03-15/ht=11/657bde6140785e0b24de1c7048aef7c54fad7c595330ba8f02d579d1b8ea3a32.jpg)

![](dt=2026-03-15/ht=11/b3c5c7f9b654790537111f4331891a87bb85ddcba993652d70eb1ddd69c75d35.jpg)

![](dt=2026-03-15/ht=11/f75c6b271879d4100f5d943ab771e6a4fda753c51e02d4a5c16b231da4e1c7bb.jpg)

![](dt=2026-03-15/ht=11/01665e374297f6c06f96cb428ad9e4f19708c9ba5bc8779c41cb842f5b2326af.jpg)

![](dt=2026-03-15/ht=11/09e544b2f6c4b9d88b8fd8a6e8c1f7e8e91c10fb00d4d41364b69470e74ae9d6.jpg)

![](dt=2026-03-15/ht=11/eeaff955995c83575a3baa4f7e5cb0038ed1fe91ea8d8c8f70d90935a4e418b0.jpg)

![](dt=2026-03-15/ht=11/57c1dc3f4c235d5decaf6ec073a48127e565928e9ef091877e03f0d83eb5ce40.jpg)

![](dt=2026-03-15/ht=11/e9b886b2f8581cf61bca733412416c2e0b9835a9196eb098d246e66923f022b6.jpg)

![](image)
e2/hive-ha/produce.db/mineru_full_text/v0/result=success/type=image/dt=2026-03-15/ht=11//5479d659a7f33a417f87a7fab73f2002be741153b27ae5f0e997e12344343bd4.jpg)

Article

https://doi.org/10.1038/s41567-023-02129-w

Nature Physics

![](dt=2026-03-15/ht=11/af96a266baa5c2bbe34f836ba76da1bd71f969017973dcfa701889ed24fee2a4.jpg)

![](dt=2026-03-15/ht=11/b30c660dd196b32f3076100513b21e12860f91dea661674bc962be921d4c95d6.jpg)

![](dt=2026-03-15/ht=11/7fc945e6933d3b6b7afc6400762b9148c4a1271d977cf674509b05351df69a94.jpg)

this measurement, the pump is sent to the device with $10.1\mathrm{-}\mu \mathrm{W}$ on-chip power $(\gamma_{\mathrm{om}} / 2\pi = 324\mathrm{kHz})$ . The device parameters are summarized in Extended Data Table 1. Next, we characterize the microwave-to-optical $(S_{\mathrm{op}})$ and optical-to-microwave $(S_{\mathrm{mo}})$ frequency conversion using the VNA. The set-up is the same as in the reflection measurements, with an on-chip pump power of $10.2\mu \mathrm{W}$ , corresponding to $n_{\mathrm{a}} = 540$ and $\gamma_{\mathrm{om}} / 2\pi = 327\mathrm{kHz}$ .

The measured scattering parameters are shown in Fig. 2c. Using a method similar to refs. 18,32, we determine the peak on-chip conversion efficiency to be $\eta = 4.9\pm 0.5\%$ (Methods) with a 3-dB bandwidth of $1.5\mathrm{MHz}$ . The added noise referred to the input for optical-to-microwave conversion is measured to be $n_{\mathrm{added},\mu \mathrm{o}} = 99\pm 10$ . Like state-of-the-art demonstrations (see ref. 15 for a recent review), our achieved efficiency and added noise are currently insufficient for direct conversion of quantum states.

However, the ability to post-select quantum optical states by single-photon detection (SPD) obviates the need for high efficiency by leveraging the nonlinearity afforded by strong measurement.

For heralded microwave photon generation, a blue-detuned laser pulse first adds a phonon in the mechanical mode, which subsequently leaks out to the microwave channel. The phonon added thermal state and single-phonon Fock state have been generated with similar techniques in an optomechanical system, where the mechanical state is probed with optical readout $^{33-36}$ . Via piezoelectric coupling, the mechanical state in our transducer is converted to a propagating microwave state, and subsequently measured with microwave tomography $^{37}$ .

This allows us to estimate the noise added during the laser pulse from parasitic optical absorption $^{38}$ in both the mechanical state and the propagating microwave state. Lower added noise is desired for high-fidelity single-photon generation, which can be achieved with lower pump power and lower experiment repetition rate. A more important goal for our device is to achieve a high rate of microwave photon addition so that heralded protocols become practicable.

A blue-detuned pump laser pulse with duration $\tau \approx 20$ ns is sent to the transducer. The pulse duration is chosen so that $\tau^{-1}$ is much smaller than the optical decay rate and laser detuning, and so the intracavity photon number follows the pulse amplitude closely. However, the pulse is much faster than the response time of the mechanical and microwave system. As shown in Fig. 3a, upon heralding, a phonon is effectively added to the mechanical resonator, which then leaks out as microwave radiation at a rate on the order of $\gamma_{\mu}$ . We integrate the optomechanical scattering rate over the optical pulse duration

to find the photon-phonon-pair generation probability, $P \approx \gamma_{\mathrm{om}}\tau$ (ref. 39). The sideband photon leaks out of the optical cavity together with reflected pump photons, most of which are filtered out by two Fabry-Pérot cavities, enabling SPD of the sideband photon. The generated microwave photon is amplified by a near quantum-limited travelling wave parametric amplifier (TWPA) $^{40}$ followed by a cryogenic low-noise amplifier. At the end of a room-temperature amplification chain, the microwave signal is demodulated to obtain a quadrature sample. The samples are labelled according to whether the microwave detection was accompanied by a photon click.

We use an on-chip peak pump power of $\sim 5\mu \mathrm{W}$ and a pulse duration of $\tau \approx 20$ ns, giving a scattering probability of $P \approx 3.6\%$ . We choose a repetition rate of $170\mathrm{kHz}$ , which leads to a thermal occupation of the mechanical mode prior to the arrival of the pump pulse of $n_{\mathrm{th}} = 0.68 \pm 0.08$ (Methods). A lower repetition rate reduces this source of noise, but makes the experiment slower. We use a pair of filter cavities to realize $>90$ -dB suppression of the pump photons with respect to the optomechanically generated sideband photons.

The probability of detecting a photon that starts inside the optical mode is $\eta_{\mathrm{sys}} = 1\%$ . This low system efficiency is caused by the optical-mode external coupling efficiency $\eta_{\mathrm{o}} = 50\%$ , an insertion loss of $25.4\%$ from the on-chip photonic waveguide to the fridge optical port, transmission through the filter cavities ( $15\%$ ) and the quantum efficiency of the SPD ( $65\%$ ).

Finally, we use linear detectors to characterize the microwave field emitted from the device. Linear phase-insensitive amplification of microwave fields effectively measures both quadratures, necessarily adding half a quantum of noise. The resulting amplifier output corresponds to the Husimi $Q$ function $Q(\alpha)$ of the microwave state up to a scaling $^{36,37}$ where $\alpha$ is the quadrature-phase amplitude of the microwave mode. In practice, noise in excess of the quantum limit is added due to the device inefficiency, amplifier noise and demodulation inefficiency.

We denote the measured probability distribution by $\mathcal{D}(\alpha)$ for the thermal state without the SPD event, and $\mathcal{D}_s(\alpha)$ for the photon-added state after post-selection with the SPD event. As a result of the excess noise, the measured $\mathcal{D}(\alpha)$ and $\mathcal{D}_s(\alpha)$ look virtually identical to the eyes, but their numerical difference is clearly resolvable $^{37}$ .

We accumulated $\sim 1.4\times 10^{6}$ post-selected samples over the course of $\sim 35\mathrm{h}$ , and binned them into a 2D histogram with $41\times 41$ entries to obtain the probabilistic distribution $\mathcal{D}_s(\alpha)$ . During the same experiment, $4.3\times 10^{7}$ samples of the thermal state were collected and binned in the same way for $\mathcal{D}(\alpha)$ .

Article

https://doi.org/10.1038/s41567-023-02129-w

Nature Physics

The difference in the probability distributions $\mathcal{D}_{\mathrm{s}}(\alpha) - \mathcal{D}(\alpha)$ is shown in Fig. 3b. As the states have no well-defined phase, we bin the samples radially and plot the results as blue points in Fig. 3c. The shaded blue region is the theoretical distribution of the phonon added state with excess noise, calculated from independently measured device parameters. As a control, we randomly sampled a subset from the thermal dataset, and obtained the probability difference shown in black in Fig. 3c.

In both the 2D histogram and the radially binned data, the axes are calibrated assuming an excess noise $n_{\mathrm{ex}} = 39 \pm 6$ , obtained from the change in variance between thermal and post-selected samples (Methods). An independent quantum calibration of the gain and excess noise in the detection chain uses sideband asymmetry to determine the phonon temperature, and leads to values of gain and excess noise that are within roughly a factor of two.

Other calibration approaches, especially when they require changing optical power, are unreliable due to the sensitive dependen
ce of system parameters such as the mechanics-microwave output efficiency on optical pump power (Methods). Multiple noise sources contribute to the excess noise, including heating after the optical pulse, an added noise of $n_{\mathrm{m}} = 2.4 \pm 0.4$ from the TWPA, and the total measurement efficiency of $-8\%$ .

Heating after the optical pulse adds extra thermal noise in the output microwave photon from the transducer, which is important for evaluating the quality of the microwave photon for potential applications. We estimate an added thermal noise of $n_{\mathrm{n}} = 1.6 \pm 0.5$ in the propagating microwave photon with two different methods. First, we use the excess noise from the heralding experiment and microwave readout efficiencies to infer the added thermal noise.

Alternatively, we calculate the overlap between the independently measured time-domain heating after the optical pulse and the temporal mode of the microwave field to obtain the added noise (Methods).

We have demonstrated the direct measurement of optically heralded microwave added photon states. The heralding rate in our experiment is $\sim 15\mathrm{Hz}$ , limited by the total optical readout system efficiency of $\eta_{\mathrm{sys}} = 1\%$ . Although the heralding rate is comparable to other quantum systems for entanglement generation[41-45], we can drastically improve it by increasing the fibre-to-chip coupling efficiency[46] and reducing optical filter insertion loss.

These substantially improved heralding rates (approximately kilohertz) would be comparable to the state-of-the-art coherence time of microwave[47,48] and acoustic[49] resonators. Another challenge is reducing the effects of induced thermal noise. Recent designs have emerged that demonstrate much better thermalization[50], allowing lower initial thermal occupation and added noise $n_{\mathrm{th}}, n_{\mathrm{n}} \lesssim 0.1$ . Adopting these 2D structures would also increase the achievable rates.

The piezoelectric coupling rate of our transducer is also limited by the multimode nature of our microwave resonator. Moving to a single-mode microwave system[24] would increase our phonon-to-microwave photon output efficiency from an $\eta_{\mu \mathrm{m}}$ of $\sim 35\%$ to close to $100\%$ . To entangle two distant microwave systems, in addition to reducing the heating, we will need to implement two copies of our transducer to produce indistinguishable optical photons inside two separate fridges.

Overcoming frequency variations in different devices by frequency-shifting in the optical domain[44,51] or entanglement-swapping with entangled optical photons at different frequencies from optical spontaneous parametric downconversion[39] would relax the device frequency-matching requirements. Our results show that, with these improvements, piezo-optomechanical quantum frequency transducers that entangle distant quantum microwave systems are within reach.

# Online content

Any methods, additional references, Nature Portfolio reporting summaries, source data, extended data, supplementary information, acknowledgements, peer review information; details of author contributions and competing interests; and statements of data and code availability are available at https://doi.org/10.1038/s41567-023-02129-w.

# References

Article

https://doi.org/10.1038/s41567-023-02129-w

Nature Physics

Publisher's note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

Springer Nature or its licensor (e.g. a society or other partner) holds exclusive rights to this article under a publishing agreement with the author(s) or other rightsholder(s); author self-archiving of the accepted manuscript version of this article is solely governed by the terms of such publishing agreement and applicable law.

© The Author(s), under exclusive licence to Springer Nature Limited 2023

Article

https://doi.org/10.1038/s41567-023-02129-w

Nature Physics

# Methods

# Device parameters

Extended Data Table 1 shows the device parameters for the optical, mechanical and microwave modes and the coupling rates. The measurement method used to obtain each parameter is also listed. We observed that the mechanical frequency and intrinsic loss rate are different under different optical powers. When the optical pump is turned off, $\gamma_{\mathrm{i}}$ is reduced by a factor of $-3$ and the mechanical frequency redshifts by $-700\mathrm{kHz}$ .

Variation of the mechanical frequency and intrinsic loss rate are probably due to a combination of optically induced heating and saturation of two-level systems. Reducing the frequency fluctuation and improving the intrinsic mechanical quality factor will be the subject of future efforts. Device and measurement set-up efficiencies are summarized in Extended Data Table 2.

# Tunable high-impedance waveguide

The high-impedance (high- $Z$ ) waveguide was realized with a hybrid aluminium-NbTiN coplanar waveguide (CPW). The ground plane of the CPW is made of electron-beam-evaporated aluminium. The centre conductor of the CPW is composed of thin and narrow NbTiN nanowires to achieve high kinetic inductance. The thickness of the NbTiN layer is $10\mathrm{nm}$ , deposited by StarCryo on a high-resistivity Si $(\rho > 10\mathrm{k}\Omega \mathrm{cm}^{-1})$ substrate from WaferPro. The width of the nanowire is $500\mathrm{nm}$ .

Square-shaped loops of $50 - \mu \mathrm{m}$ width, formed by the nanowires along the centre conductor of the CPW, allow wireless tuning of the kinetic inductance through an external magnetic field<sup>31</sup>. The distance between the edges of the ground plane was chosen to be $150\mu \mathrm{m}$ to reduce the capacitance. A home-made tuning coil with $50\mathrm{-mm}$ diameter and $\sim2,000$ turns of NbTi wire (Supercon Inc., SC-T48B-M-0.10mm) generates $\sim0.01\mathrm{mT}\mathrm{mA}^{-1}$ at the chip.

We observed negligible heating from the coil with up to $200\mathrm{-mA}$ continuous current thanks to zero heat dissipation in the superconducting NbTi wire. The total length of the high- $Z$ CPW is $\sim65\mathrm{mm}$ , giving rise to standing-wave resonances with a free spectral range (FSR) of $\sim110\mathrm{MHz}$ (Extended Data Fig. 1). The waveguide resonances can be frequency-tuned by more than $150\mathrm{MHz}$ , larger than the FSR of the resonances, allowing us to match a waveguide resonance to a mechanical mode over a broad frequency range.

We show the measured microwave-to-optical conversion $S$ parameter in Extended Data Fig. 2 as we vary the coil current. Mechanical resonances of the transducer are not affected by the coil current, while the conversion is enhanced when the microwave modes are resonant with the mechanical modes. The microwave modes reach their maximal frequencies at non-zero current due to non-zero trapped external flux in the tuning loops from a non-zero background magnetic field during the cooldown.

# Thermal occupation measurement and time-domain heating from the optical pulse

Thermal occupation of the mechanical mode $n_{\mathrm{th}}$ is measured with sideband asymmetry by comparing the SPD rates of the optomechanically scattered optical sidebands. Two tunable external cavity diode lasers (Pure Photonics PPCL300) are locked to detunings $\Delta_{\pm} \equiv \omega_{\pm} - \omega_{\mathrm{o}} = \pm \omega_{\mathrm{m}}$ with respect to the optical mode $\omega_{\mathrm{o}}$ as blue- and red-detuned pumps.

Due to imperfect laser detunings and the narrow linewidths of the optical filter cavities, the count rates are recorded as the filter cavities are tuned by $-20\mathrm{MHz}$ between the blue-detuned and red-detuned sideband count rate measurement to optimize the sideband count rate. The insertion loss between counts at the two frequencies could be different.

To calibrate the varying insertion loss, we drive a coherent phonon occupation $n_{\mathrm{coh}}$ with a microwave pulse and measure its si
deband count rate in addition to sideband count rate from the thermal phonons $n_{\mathrm{th}}$ in an interleaved fashion. The ratio of the sideband count rates between $n_{\mathrm{th}}$ and $n_{\mathrm{coh}}$ is independent of the filter insertion loss:

$$
R _ {\mathrm {r}} = \frac {n _ {\mathrm {c o h}}}{n _ {\mathrm {t h}}}, R _ {\mathrm {b}} = \frac {n _ {\mathrm {c o h}} + 1}{n _ {\mathrm {t h}} + 1} \tag {2}
$$

where the subscript denotes the laser detuning. By further comparing these ratios between blue- and red-detuned pump, we find

$$
\frac {R _ {\mathrm {b}} - 1}{R _ {\mathrm {r}} - 1} = \frac {n _ {\mathrm {t h}}}{n _ {\mathrm {t h}} + 1} \equiv A \tag {3}
$$

giving us the sideband asymmetry ratio $A$ and thermal phonon number $n_{\mathrm{th}} = 1 / (1 / A - 1)$ . Extended Data Fig. 3a shows the measured thermal occupation versus different repetition rates of the optical pulse. We find that our repetition rate and thermal occupation are comparable to similar optomechanical systems[52].

The microwave port of the transducer allows us to monitor the microwave noise from the transducer, which is dominated by converted thermal mechanical noise when the transducer is under a pulsed optical pump. As shown in Extended Data Fig. 3b, we measured the temporal heating of the mechanical mode by using a series of consecutive demodulation windows, each of duration 48 ns. Changes in the variance of the demodulated data correspond to varying thermal noise in the microwave signal, which can be calibrated to a varying thermal phonon occupation by the initial thermal occupation measured with optical sideband asymmetry.

# Conversion efficiency and added noise measurement

The on-chip conversion efficiency of the transducer is defined as

$$
\eta_ {\mu \mathrm {o}} \equiv \frac {\dot {N} _ {\text {o u t ,} \mu}}{\dot {N} _ {\text {i n ,} \mathrm {o}}}, \eta_ {\mathrm {o} \mu} \equiv \frac {\dot {N} _ {\text {o u t ,} \mathrm {o}}}{\dot {N} _ {\text {i n ,} \mu}} \tag {4}
$$

where $\dot{N}_{\mathrm{in(out),\mu (o)}}$ is the input (output) microwave (optical) photon flux.

The input and output optical photon flux are measured with the sideband filter and the SPD. For the output photon flux from microwave-to-optical conversion, it can be directly measured with the optical set-up shown in Extended Data Fig. 4. The system detection efficiency of the optical detection set-up, including the insertion loss of the optical switches, the isolator, the two sideband filters and the SPD quantum efficiency, is measured independently to be $\eta_{\mathrm{SPD}} = 9.9\%$ .

The optical insertion loss within the dilution refrigerator, and the one-way coupling efficiency between the lensed fibre and the on-chip waveguide, is measured to be $\eta_{\mathrm{in-fridge}} = 25.4\%$ in total. We measure the insertion loss of the output circulator to be $\eta_{\mathrm{circ}} = 77\%$ . Together we calculate an overall optical set-up efficiency from on-chip photonic waveguide to the SPD to be $\eta_{\mathrm{setup}} = 2\%$ .

Note that the optical mode external coupling efficiency $\eta_{\mathrm{o}} = 50\%$ is not included in the set-up efficiency, whereas it is included in the on-chip conversion efficiency $\eta$ and system efficiency $\eta_{\mathrm{sys}}$ . The measured count rate at the SPD and all the output insertion losses are used to calculate the output photon flux at the device.

To calibrate the input photon flux that reaches the device during the optical-to-microwave conversion, the dilution refrigerator and the circulator are bypassed in the optical circuit while not changing anything else in the set-up. As a result, the same output optical circuit is used to measure the sideband photon flux at the fridge input port. The in-fridge insertion loss $\eta_{\mathrm{in -fridge}}$ is then used to calculate the input photon flux at the device.

The microwave photon flux at the device is not directly measurable, and there is also no independent way of measuring the microwave input attenuation and the output amplification $G_{\mathrm{m}}$ separately in our experiment. However, $G_{\mathrm{m}}$ can be calculated using the measured optical photon flux and external microwave power assuming the conversion efficiencies are equal between the two directions. More specifically

$$
\dot {N} _ {\text {o u t}, \mu} = \frac {P _ {\mu , \mu \mathrm {o}}}{\hbar \omega_ {\mu}} \frac {1}{G _ {\mathrm {m}}}, \dot {N} _ {\text {i n}, \mu} = \frac {P _ {\mu , \mathrm {o} \mu}}{\hbar \omega_ {\mu}} \frac {1}{G _ {\mathrm {m}} | S _ {\mu \mu} | ^ {2}} \tag {5}
$$

Article

https://doi.org/10.1038/s41567-023-02129-w

Nature Physics

where $P_{\mu, \mu \sigma (\mathrm{O})}$ denotes the output microwave power measured at the real-time spectrum analyser (RSA) for the optical-to-microwave (microwave-to-optical) conversion. $G_{\mathrm{m}}$ converts between the microwave photon flux at the RSA and the flux leaving the transducer $\dot{N}_{\mathrm{out},\mu}$ . For the microwave input flux, the output flux is converted to input using microwave reflection $S_{\mu \mu}$ independently measured with the VNA. Substituting equation (5) into equation (4), we find that $\eta_{\mu \sigma} = \eta_{\sigma \mu}$ leads to

$$
\begin{array}{l} G _ {\mathrm {m}} = \left(\frac {P _ {\mu , \mu \mathrm {o}} P _ {\mu , \mathrm {o} \mu} / \left(\hbar \omega_ {\mu}\right) ^ {2}}{\dot {N} _ {\mathrm {i n} , \mathrm {o}} \dot {N} _ {\mathrm {o u t} , \mathrm {o}} | S _ {\mu \mu} | ^ {2}}\right) ^ {1 / 2} (6) \\ \eta \equiv \eta_ {\mu \mathrm {o}} = \eta_ {\mathrm {o} \mu} = \left(\frac {\dot {N} _ {\text {o u t , o}}}{\dot {N} _ {\text {i n , o}}} \frac {P _ {\mu , \mu \mathrm {o}} \left| S _ {\mu \mathrm {u}} \right| ^ {2}}{P _ {\mu , \mathrm {o} \mu}}\right) ^ {1 / 2} (7) \\ \end{array}
$$

We obtain an on-chip conversion efficiency $\eta = 4.9\pm 0.5\%$ with output amplification $G_{\mathrm{m}} = 101\mathrm{dB}$ . $G_{\mathrm{m}}$ can be used to calculate the added noise in the microwave output during the optical-to-microwave conversion, which is directly measured by the RSA. We find it to be 4.9 at the microwave output of the device, and the corresponding added noise referred to the input is $n_{\mathrm{added},\mu \mathrm{o}} = 99\pm 10$ . Based on the relationships between theoretical added noises and the thermal occupation of the mechanical mode[15], we further estimate the thermal occupation of the mechanical mode to be $n_{\mathrm{th}}\approx 22$ , and the added noise for microwave-to-optical conversion to be $n_{\mathrm{added,0i}}\approx 181$ .

# Microwave readout and added noise

The microwave readout uses a phase-insensitive amplifier with $G_{\mathrm{m}} \gg 1$ , necessarily adding noise, which we represent with $\hat{h}$ :

$$
\hat {S} = \sqrt {G _ {\mathrm {m}}} \hat {c} _ {\text {o u t}} + \sqrt {G _ {\mathrm {m}} - 1 \hat {h} ^ {\dagger}} \approx \sqrt {G _ {\mathrm {m}}} (\hat {c} _ {\text {o u t}} + \hat {h} ^ {\dagger}) \tag {8}
$$

where $\hat{c}_{\mathrm{out}}$ is the microwave output operator. The output operator is related to the mechanical mode operator by the external coupling efficiency $\eta_{\mu \mathrm{m}}$ . Note that $\eta_{\mu \mathrm{m}}$ is different from the external coupling efficiency of the microwave resonator $\eta_{\mu}$ . Assuming perfect demodulation, the temporal microwave mode detected from the device has a ladder operator given by

$$
\hat {c} _ {\text {o u t}} = \sqrt {\eta_ {\mu \mathrm {m}}} \hat {b} + \sqrt {1 - \eta_ {\mu \mathrm {m}}} \hat {b} _ {\mathrm {n}} \tag {9}
$$

where $\hat{b}_{\mathrm{n}}$ describes noise added from other degrees of freedom because of the device inefficiency, and follows the relations $\langle \hat{b}_{\mathrm{n}}\hat{b}_{\mathrm{n}}^{\dagger}\rangle = n_{\mathrm{n}} + 1$ and $\langle \hat{b}_{\mathrm{n}}^{\dagger}\hat{b}_{\mathrm{n}}\rangle = n_{\mathrm{n}}$ .

Noise is further added by the microwave readout chain. More specifically, we characterize th
e microwave readout by the gain $G_{\mathrm{m}}$ , measurement noise $n_{\mathrm{m}}$ referred to the TWPA input, and the demodulation efficiency $\eta_{\mathrm{d}}$ :

$$
\begin{array}{l} \hat {I} = \hat {S} + \hat {S} ^ {\dagger} \\ = \sqrt {\eta_ {\mathrm {d}} \eta_ {\mu \mathrm {m}} G _ {\mathrm {m}}} \hat {X} \tag {10} \\ + \sqrt {\eta_ {d} (1 - \eta_ {\mu m}) G _ {m}} \hat {X} _ {n} + \sqrt {G _ {m}} \hat {X} _ {m} \\ \end{array}
$$

where $\hat{X}_{\mathrm{(n)}} = \hat{b}_{\mathrm{(n)}} + \hat{b}_{\mathrm{(n)}}^{\dagger}$ . The measurement added noise is assumed to be broadband and is independent of the demodulation. $X_{\mathrm{n(m)}}$ follows the noise statistics $\langle X_{\mathrm{n(m)}}^2\rangle = 2n_{\mathrm{n(m)}} + 1$ .

When the mechanical mode is in a thermal state with mean phonon number $n_{\mathrm{th}}$ , the variance of the demodulated IQ data is

$$
\begin{array}{l} \left\langle I ^ {2} \right\rangle = \eta_ {\mathrm {d}} \eta_ {\mu \mathrm {m}} G _ {\mathrm {m}} (2 n _ {\mathrm {t h}} + 1) + \eta_ {\mathrm {d}} (1 - \eta_ {\mu \mathrm {m}}) G _ {\mathrm {m}} (2 n _ {\mathrm {n}} + 1) \\ + G _ {\mathrm {m}} \left(2 n _ {\mathrm {m}} + 1\right) \tag {11} \\ = \eta_ {\mathrm {d}} \eta_ {\mu \mathrm {m}} G _ {\mathrm {m}} (2 \left(n _ {\mathrm {t h}} + n _ {\mathrm {e x}}\right) + 2) \\ \end{array}
$$

where $n_{\mathrm{n}}$ is thermal noise from other degrees of freedom, including heating from the optical pump pulse, emitted into the microwave channel, and $n_{\mathrm{m}}$ is excess noise from the microwave amplifiers. We have lumped all excess noise into $n_{\mathrm{ex}}$ :

$$
\begin{array}{l} n _ {\mathrm {e x}} = \frac {1 - \eta_ {\mu \mathrm {m}}}{\eta_ {\mu \mathrm {m}}} n _ {\mathrm {n}} + \frac {1}{\eta_ {\mathrm {d}} \eta_ {\mu \mathrm {m}}} n _ {\mathrm {m}} \tag {12} \\ + \frac {1}{2} \left(\frac {1 - \eta_ {\mu \mathrm {m}}}{\eta_ {\mu \mathrm {m}}} + \frac {1}{\eta_ {\mathrm {d}} \eta_ {\mu \mathrm {m}}} - 1\right) \\ \end{array}
$$

For ideal quadrature-phase measurement, $\eta_{\mathrm{um}} = \eta_{\mathrm{d}} = 1$ and $n_{\mathrm{m}} = 0$ , and a minimal noise of $1/2$ is added, giving us the phase space distribution as the Husimi $Q$ representation of the state. Note that this minimal added noise is not included in our definition of excess noise $n_{\mathrm{ex}}$ , and $n_{\mathrm{ex}} = 0$ for ideal quadrature-phase measurement.

When a phonon is added to the state via post-selection with SPD, the mean phonon number is $n_{\mathrm{th,PS}} = 2n_{\mathrm{th}} + 1$ . As a result

$$
\left\langle I ^ {2} \right\rangle | _ {\mathrm {P S}} = \left\langle I ^ {2} \right\rangle + \eta_ {\mathrm {d}} \eta_ {\mu \mathrm {m}} G _ {\mathrm {m}} (2 n _ {\mathrm {t h}} + 2) \tag {13}
$$

$$
\frac {\left\langle I ^ {2} \right\rangle | _ {\mathrm {P S}}}{\left\langle P ^ {2} \right\rangle} = 1 + \frac {n _ {\mathrm {t h}} + 1}{n _ {\mathrm {t h}} + n _ {\mathrm {e x}} + 1} \tag {14}
$$

In the actual experiment, there is a non-zero dark count rate of $70 \pm 10\mathrm{Hz}$ on the SPD, resulting in a heralding efficiency of $\eta_{\mathrm{herald}} \approx 85\%$ , defined as the fraction of the total counts that are actually from the sideband photons. The normalized post-selected variance is then given by

$$
\frac {\left\langle P ^ {2} \right\rangle | _ {\mathrm {P S}}}{\langle P ^ {2} \rangle} = 1 + \eta_ {\text {h e r a l d}} \frac {n _ {\mathrm {t h}} + 1}{n _ {\mathrm {t h}} + n _ {\mathrm {e x}} + 1} \tag {15}
$$

We note that the dark count rate appears to be higher than the heralding rate of $\sim 15\mathrm{Hz}$ . The heralding rate is an average rate combining the sideband photon count rate within the sideband pulse time window, duration of the sideband pulse and repetition rate of the experiment. The average sideband photon count rate within its duration is $\sim 600\mathrm{Hz}$ , much higher than the dark count rate, which is approximately constant over time.

To minimize added noise $n_{\mathrm{n}}$ in the microwave tomography measurement, matched filtering is desired and can be realized during the digital demodulation[36,37]. The optimal filter shape is given by the time-domain waveform of the outgoing photon, which has the same time-domain waveform as the classical solution of the propagating microwave field $A(t)$ . We solve the coupled-mode theory (CMT) to calculate the propagating microwave field with one initial phonon in the mechanical mode, as shown in Extended Data Fig. 5a. The intracavity photon numbers of the mechanical and microwave modes are also shown for comparison.

We have attempted demodulation with different filtering including the matched filtering with the numerical waveform $f_{\mathrm{demod}}(t) = A(t)$ calculated from CMT, and exponential waveforms $f_{\mathrm{d}}(t) = \sqrt{\kappa_{\mathrm{d}}} \exp(-\kappa_{\mathrm{d}} t / 2) \theta(t)$ with a decay rate of $\kappa_{\mathrm{d}} / 2\pi = 1 \mathrm{MHz}$ and $5 \mathrm{MHz}$ . $\theta(t)$ is the Heaviside function. The matched filter is plotted as a dashed black line to show the theoretical maximal efficiency limited by the device parameters. Extended Data Fig.

5b shows the theoretical measurement efficiency $\eta_{\mathrm{m}}(t) \equiv |\int \mathrm{d}\tau A^*(\tau) f_{\mathrm{d}}(\tau - t)|^2$ , where $t$ is the demodulation delay. The matched waveform gives the highest possible measurement efficiency equal to $\eta_{\mu \mathrm{m}}$ , which is the device mechanical-microwave efficiency, and is limited by device internal loss. Extended Data Fig. 5c shows the measured relative change of the $IQ$ variance before and after post-selection as a figure of merit for the state tomography measurement<sup>36</sup>.

A higher initial thermal phonon occupation both increases the sideband count rate, and makes the post-selected

Article

https://doi.org/10.1038/s41567-023-02129-w

Nature Physics

state more distinguishable with the same excess noise from the measurement, because the mean phonon number roughly doubles after the phonon addition event $^{35,36}$ . Therefore, we apply a heating pulse to the transducer 140 ns before the heralding pulse to increase the initial thermal phonon occupation to $\sim 10$ . We find that the two exponential filters give a more visible change in variance, and thus lower excess noise. We suspect this is due to two-level system-induced fluctuation of the mechanical frequency, as shown in Extended Data Fig.

5e, with microwave reflection measurement under no optical pump. As a result, the demodulation in the heralding experiment is conducted with a 5-MHz exponential filter, and the theoretical total measurement efficiency is $\eta_{\mathrm{d}}\eta_{\mu \mathrm{m}}\approx 8\%$ , where the demod efficiency $\eta_{\mathrm{d}}\approx 24\%$ and the device mechanics-microwave efficiency $\eta_{\mu \mathrm{m}}\approx 35\%$ . This is larger than the magnitude of the frequency fluctuations and therefore masks their effect at the cost of lowered measurement efficiency.

Characterization and improvement of the frequency fluctuation will be the subject of future study.

To calibrate the gain $G_{\mathrm{m}}$ and excess noise $n_{\mathrm{m}}$ , two different states of the microwave field are required. We carry out two types of calibration. First, we use the laser heating to generate two thermal states with different thermal occupations. For the thermal state with higher $n_{\mathrm{th}}$ , a heating pulse is applied 140 ns before the probe pulse, where the sideband asymmetry from the probe pulse is used to obtain the thermal occupation of the mechanical mode $n_{\mathrm{th,high}} = 5.9 \pm 2$ .

For the thermal state with lower $n_{\mathrm{th}}$ , we use a repetition of $10~\mu \mathrm{s}$ and no heating pulse to obtain $n_{\mathrm{th,low}} = 0.56 \pm 0.2$ . The demodulation is aligned to the probe pulse, and we measure the IQ variance from these two different thermal states. From equation (11), we calculate $n_{\mathrm{ex}} = 18 \pm
8$ . However, we find that the device parameters are varying under different optical powers, which results in different $n_{\mu \mathrm{m}}$ . A $10\%$ change in $n_{\mu \mathrm{m}}$ could lead to a factor of 2 difference in $n_{\mathrm{ex}}$ .

Alternatively, $n_{\mathrm{ex}}$ can be calculated using the thermal and photon-added state as shown in equation (15). For the data shown in Fig. 3, we find $n_{\mathrm{ex}} = 39 \pm 6$ .

To better understand the potential of the microwave field from the transducer, it is important to estimate the added noise $n_{\mathrm{n}}$ in it. Noise in the microwave field is not directly measurable with imperfect detection. Nevertheless, we carry out the estimation with two different methods. In addition to the pre- and post-selected datasets, we further take a control measurement every 500 experimental runs, where we execute the same demodulation with the same timing except that no optical pump pulse is sent to the device. The system can be approximated to be in the initial thermal state with $n_{\mathrm{th}} = 0.68$ , independently measured by optical sideband asymmetry, and no heating or microwave photon is generated. As a result, the variance becomes

$$
\left\langle I ^ {2} \right\rangle | _ {\mathrm {t h}} = \eta_ {\mathrm {d}} G _ {\mathrm {m}} \left(2 n _ {\mathrm {t h}} + 1\right) + G _ {\mathrm {m}} \left(2 n _ {\mathrm {m}} + 1\right) \tag {16}
$$

Using $\eta_{\mathrm{d}}$ and $\eta_{\mu \mathrm{m}}$ calculated from theory, the measured $\langle I^2\rangle ,\langle I^2\rangle |_{\mathrm{PS}}$ and $\langle I^2\rangle |_{\mathrm{th}}$ allow us to calculate $n_{\mathrm{n}} = 1.9\pm 0.4$ and $n_\mathrm{m} = 2.4\pm 0.4$

Alternatively, we could estimate the added noise in the microwave field from the overlap between the measured temporal heating and the temporal mode of the microwave photon. We fit the measured temporal heating assuming the mechanical mode is coupled to a thermal bath with an exponentially decaying population excited by the pump pulse. Noise in the output microwave signal is calculated with semiclassical Monte Carlo simulation of an ensemble of 3,000 instances. Extended Data Fig. 5c shows the temporal heating together with the single-photon temporal mode. Overlap between the heating and single-photon temporal mode gives $n_{\mathrm{n}} = 1.3 \pm 0.2$ . The uncertainty mostly comes from the sideband asymmetry measurement of the initial $n_{\mathrm{th}}$ .

# Measurement set-up

Extended Data Fig. 4 shows the optical set-up used in this work. Two tunable external cavity diode lasers (PurePhotonics PPCL300) are

first intensity-stabilized with electro-optic modulators, and then frequency-stabilized using temperature-stabilized fibre Fabry-Pérot filters (F1 and F2). The filters also suppress the laser phase noise at the converter frequencies. A fast wavelength-scanning laser (Freedom Photonics FP4209) is used for fibre-to-chip coupling optimization. Two acousto-optic modulators (AOMs) are simultaneously pulsed to generate the optical pump pulse with high on-off ratio ( $>90$ dB). The duration of the pulse is limited by the rise-fall time of the 200-MHz AOM to be $\geq 20$ ns.

Multiple MEMS optical switches are implemented to route the input light to either the transducer or filter cavities FA and FB for coarse tuning. The filter cavities have 15-MHz bandwidth and 14-GHz free spectral range, and provide 90-dB pump suppression, while dispersion at the filter resonance delays the sideband photons by an extra $\sim 40$ ns compared to the feed-through transmitted pump photons. This allows us to further separate the pump and sideband photon counts in the time domain.

The reflected light from the device can be routed to one of the photodetectors for the EIT measurement or microwave-to-optical conversion measurement, or to the filter cavities for SPD on the sideband photons.

We show the microwave set-up in Extended Data Fig. 6. An intermediate frequency (IF) of $125\mathrm{MHz}$ is used from the quantum machine (QM OPX01) and upconverted for the microwave input. Proper attenuations on the microwave input line guarantee the input microwave thermal noise to be less than 0.01. The microwave signal from the transducer is first amplified by a TWPA with the dispersive feature around $6.42\mathrm{GHz}$ and pumped at $5.027\mathrm{GHz}$ (not shown) to maximize the gain at the transducer frequency.

Two broadband isolators $(3 - 12\mathrm{GHz})$ are installed before and after the TWPA to minimize reflection within its gain bandwidth. The signal is then further amplified by a high electron mobility transistor (HEMT) and two room-temperature low-noise amplifiers, and measured by either the RSA or the VNA, or downconverted and digitized on the QM. Temporal delay in the optical set-up is longer than in the microwave set-up, and the microwave signal from the transducer arrives at the QM before the voltage pulse from the SPD.

As a result, the demodulation is always executed first, and then stored differently in real time, conditioned on the SPD event.

# Data availability

The data for Figs. 2 and 3 and Extended Data Figs. 1, 2, 3 and 5 are available on Zenodo at https://doi.org/10.5281/zenodo.7903643. Additional data that support the findings of this study are available from the corresponding author upon reasonable request. Source data are provided with this paper.

# References

52. Fiaschi, N. et al. Optomechanical quantum teleportation. Nat. Photon. 15, 817-821 (2021).

# Acknowledgements

W.J. and F.M.M. thank C. J. Sarabalis and H. Xiong for helpful discussions. A.-H.S.N. acknowledges useful discussions with O. Painter, C. Regal, K. Lehnert, M. Fejer and S. Groeblacher. The authors thank K. K. S. Multani, A. Y. Cleland, O. A. Hitchcock, C. Langrock, B. Kuyken, T. Vandekerckhove and M. P. Maksymowych for fabrication assistance, K. A. Villegas Rosales and N. Drucker at Quantum Machines and Y. Guo for technical support, and K. Serniak and W. D. Oliver at MIT Lincoln Laboratory for providing the TWPA.

This work was primarily supported by the US Army Research Office (ARO) Cross-Quantum Systems Science & Technology (CQTS) programme (grant no. W911NF-18-1-0103), the National Science Foundation CAREER award no. ECCS-1941826, the Airforce Office of Scientific Research (AFOSR) (MURI no. FA9550-17-1-0002 led by CUNY) and the David and Lucille Packard Fellowship. Device fabrication was performed at the Stanford Nano Shared Facilities (SNSF) and the Stanford Nanofabrication Facility (SNF), supported by NSF award ECCS-2026822. A.H.S.-N.

Article

https://doi.org/10.1038/s41567-023-02129-w

Nature Physics

acknowledges support via a Sloan Fellowship. We also thank NTT Research and Amazon Web Services Inc. for their financial support. Some of this work was funded by the US Department of Energy through grant no. DE-AC02-76SF00515 and via the Q-NEXT Center.

# Author contributions

W.J. designed the device with assistance from F.M.M. and S.M. W.J. and F.M.M. fabricated the device assisted by S.M. W.J., F.M.M. and R.V.L. developed the fabrication process. W.J. and F.M.M. measured the device with assistance from S.M. R.N.P., T.P.M., J.D.W. and A.H.S.-N. provided assistance with the measurement set-up. W.J., F.M.M. and A.H.S.-N. wrote the manuscript with input from all authors. A.H.S.-N. supervised the project.

# Competing interests

A.H.S.-N. is an Amazon Scholar. The other authors declare no competing interests.

# Additional information

Extended data is available for this paper at https://doi.org/10.1038/s41567-023-02129-w.

Supplementary information The online version contains supplementary material available at https://doi.org/10.1038/s41567-023-02129-w.

Correspondence and requests for materials should be addressed to Amir H. Safavi-Naeini.

Peer review information Nature Physics thanks Christophe Galland and the other, anonymous, reviewer(s) for their contribution to the peer review of this work.

Reprints and permissio
ns information is available at

www.nature.com/reprints.

Article

https://doi.org/10.1038/s41567-023-02129-w

Nature Physics

![](dt=2026-03-15/ht=11/f32e652df22d05eaa36936cf8918d854e6fc47a9cbd8f06cd03e852c4a144916.jpg)

![](dt=2026-03-15/ht=11/a559488fddfa11b1a5755491649208e27085f5b8ce83b06a09aa7046c5d49229.jpg)

![](dt=2026-03-15/ht=11/822bb5081bac9ff75d65b7257bec83fcb26da9494caf0753254133facaee34a1.jpg)

Article

https://doi.org/10.1038/s41567-023-02129-w

Nature Physics

![](dt=2026-03-15/ht=11/2a6ff5abc10bdb303de3bf3a5ccd607fea03e0709e9efc3c55ca267d273841fc.jpg)

Article

https://doi.org/10.1038/s41567-023-02129-w

Nature Physics

![](dt=2026-03-15/ht=11/df782ce9ae5eb7fd8b3119ce50531f5fe11c95f83faee86dab62fa3b42990e02.jpg)

![](dt=2026-03-15/ht=11/0ae7dcb7fceae41801751b624cd1a86811bb7f0f589bbb705509b12d39a709e3.jpg)

Article

https://doi.org/10.1038/s41567-023-02129-w

Nature Physics

![](dt=2026-03-15/ht=11/8019834c813d1cffc765677adf1e9247b5d3a63021b8ac02d5eb573711eb6b0a.jpg)

![](dt=2026-03-15/ht=11/3d70e5549d9f6e32d47ae70308a06d829fbd72dcf12f8864fad644f3a77da038.jpg)

Article

https://doi.org/10.1038/s41567-023-02129-w

Nature Physics

![](dt=2026-03-15/ht=11/2f9780eb8626e2e7606ad03afe97e95354acab22bcae69e0fdf74ad5552d08dc.jpg)

![](dt=2026-03-15/ht=11/1ca6526db70c2e57f84efb4b15a59ae4ee07c72d3edbab3d5459ab53dd9777ff.jpg)

![](dt=2026-03-15/ht=11/05eae339106e621757c80d8e0e826c5f395625c799078f8821af53fe0a4ccc3a.jpg)

![](dt=2026-03-15/ht=11/5660ddb5fd268e6afc01c3022b859c42bc398afe3806835e12bf4da8e89c2749.jpg)

![](dt=2026-03-15/ht=11/22492d4d719b81ab34e399dcfd21a2fd2eb895440bdc60feb69b45ccb61b0fa2.jpg)

Article

https://doi.org/10.1038/s41567-023-02129-w

Nature Physics

![](dt=2026-03-15/ht=11/d56f3241b83f4d2d2253bf137eb16f5d0869f34ba575b87850da010c8710d8cb.jpg)

Article

https://doi.org/10.1038/s41567-023-02129-w

Nature Physics

Extended Data Table 1 | Device and system parameters

![](dt=2026-03-15/ht=11/529270e4477f6b2f90443a0a3ebcb09b3cd38ecf0bda5c956ef9e064bdb900b7.jpg)

<table><tr><td>Parameter</td><td>Value</td><td>Method</td></tr><tr><td>ωo/2π</td><td>193.53 THz</td><td>Laser wavelength sweep</td></tr><tr><td>κo/2π</td><td>1.122 GHz</td><td>EIT</td></tr><tr><td>κo,e/2π</td><td>0.561 GHz</td><td>EIT</td></tr><tr><td>g0/2π</td><td>413 kHz</td><td>EIT</td></tr><tr><td>ωm/2π</td><td>~ 3.596 GHz</td><td>Microwave Sμμ</td></tr><tr><td>γi/2π (pump off)</td><td>0.36 MHz</td><td>Microwave Sμμ</td></tr><tr><td>γi/2π (pump on)</td><td>1.07 MHz</td><td>Microwave Sμμ</td></tr><tr><td>ωμ/2π</td><td>3.5958 GHz</td><td>Microwave Sμμ</td></tr><tr><td>κμ/2π</td><td>3.06 MHz</td><td>Microwave Sμμ</td></tr><tr><td>κμ,e/2π</td><td>3.04 MHz</td><td>Microwave Sμμ</td></tr><tr><td>gμ/2π</td><td>424 kHz</td><td>Microwave Sμμ</td></tr></table>

Article

https://doi.org/10.1038/s41567-023-02129-w

Nature Physics

Extended Data Table 2 | Device and system efficiencies

![](dt=2026-03-15/ht=11/b348773b63643495ed2008be6d3bc8c3fcc7bb17aec442923ddbc5f7ee3203f3.jpg)

<table><tr><td>Parameter</td><td>Symbol</td><td>Value</td></tr><tr><td>Optical mode external coupling efficiency</td><td>ηo</td><td>50 %</td></tr><tr><td>Microwave mode external coupling efficiency</td><td>ημ</td><td>99 %</td></tr><tr><td>On-chip peak conversion efficiency</td><td>η</td><td>5 %</td></tr><tr><td>Demodulation efficiency</td><td>ηd</td><td>24 %</td></tr><tr><td>Mechanics-microwave external coupling efficiencya</td><td>ημm</td><td>35 %</td></tr><tr><td>Total optical readout system efficiency</td><td>ηsys</td><td>1 %</td></tr><tr><td>Heralding efficiency</td><td>ηherald</td><td>85 %</td></tr></table>

a This is also the conversion efficiency from an intracavity phonon to a propagating microwave photon.

Article

https://doi.org/10.1038/s41567-023-02129-w

Nature Physics