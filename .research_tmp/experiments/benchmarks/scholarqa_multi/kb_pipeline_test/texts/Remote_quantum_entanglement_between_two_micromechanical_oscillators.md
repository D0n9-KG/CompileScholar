# Remote quantum entanglement between two micromechanical oscillators

Ralf Riedinger $^{1,3}$ , Andreas Wallucks $^{2,3}$ , Igor Marinković $^{2,3}$ , Clemens Löschnauer $^{1}$ , Markus Aspelmeyer $^{1}$ , Sungkun Hong $^{1*}$ & Simon Gröblacher $^{2*}$

Entanglement, an essential feature of quantum theory that allows for inseparable quantum correlations to be shared between distant parties, is a crucial resource for quantum networks<sup>1</sup>. Of particular importance is the ability to distribute entanglement between remote objects that can also serve as quantum memories. This has been previously realized using systems such as $\mathrm{warm}^{2,3}$ and cold atomic vapours<sup>4,5</sup>, individual atoms<sup>6</sup> and ions<sup>7,8</sup>, and defects in solid-state systems<sup>9-11</sup>.

Practical communication applications require a combination of several advantageous features, such as a particular operating wavelength, high bandwidth and long memory lifetimes. Here we introduce a purely micromachined solid-state platform in the form of chip-based optomechanical resonators made of nanostructured silicon beams. We create and demonstrate entanglement between two micromechanical oscillators across two chips that are separated by 20 centimetres. The entangled quantum state is distributed by an optical field at a designed wavelength near 1,550 nanometres.

Therefore, our system can be directly incorporated in a realistic fibre-optic quantum network operating in the conventional optical telecommunication band. Our results are an important step towards the development of large-area quantum networks based on silicon photonics.

In recent years, nanofabricated mechanical oscillators have emerged as a promising platform for quantum information processing. The field of opto- and electromechanics has seen great progress, including ground-state cooling $^{12,13}$ , quantum interfaces to optical or microwave modes $^{14,15}$ , mechanical squeezing $^{16}$ and single-phonon manipulation $^{17-20}$ . Demonstrations of distributed mechanical entanglement, however, have so far been limited to intrinsic material resonances $^{21}$ and the motion of trapped ions $^{8}$ .

Entanglement of engineered (opto-)mechanical resonances, on the other hand, would provide a route towards scalable quantum networks. The freedom of designing and choosing optical resonances would allow operation in the entire frequency range of the technologically important C-, S- and L-bands of fibre-optic telecommunications. Together with dense wavelength-division multiplexing (on the ITU-T grid; ITU-T, International Telecommunication Union Standardization Sector), this could enable quantum nodes separated by long distances (about $100\mathrm{km}$ ) that can communicate at large bandwidths.

State-of-the-art engineered mechanical elements have energy lifetimes that typically range between micro- $^{15}$ and milliseconds $^{22}$ , which would allow entanglement distribution on a regional level $^{23}$ . In addition, these entangled mechanical systems could be interfaced with microwaves $^{24}$ , opening up the possibility of integrating superconducting quantum processors in the local nodes of the network.

Here we report on the observation of distributed entanglement between two nanomechanical resonators, mediated by telecommunication-wavelength photons. We use the DLCZ protocol[25], which was experimentally pioneered with ensembles of cold atoms[4]. The entanglement is generated probabilistically through the conditional preparation of a single phonon, heralded by the detection of a signal photon that could

originate from either of two identical optomechanical oscillators. Fabrication imperfections have previously limited the use of artificial structures, requiring external tuning mechanisms to render such systems indistinguishable. Here we demonstrate not only that obtaining sufficiently identical devices is in fact possible through nanofabrication, but also that our method could in principle be applied to more than two systems.

The mechanical oscillators that we use in our experiment are nanostructured silicon beams with co-localized mechanical and optical resonances. Radiation pressure forces and the photoelastic effect couple the optical and mechanical modes with a rate $g_{0}$ , causing the optical frequency to shift under the displacement of the mechanical oscillator[26]. This effect can be used to selectively address Stokes and anti-Stokes transitions by driving the optical resonance with detuned laser beams, resulting in a linear optomechanical interaction. As was recently shown, this technique can be used to create non-classical mechanical and optomechanical states at the single-quantum level for individual devices by using photon counting and post-selection[15,19].

To apply the DLCZ scheme to the entanglement of two separate optomechanical crystals, a critical requirement is that the photons emitted from the optomechanical cavities must be indistinguishable. This can be achieved by creating a pair of nanobeams with identical optical and mechanical resonances. Until now, however, fabrication variations have inhibited the deterministic generation of identical devices and the design of current oscillators does not include any tuning capabilities.

Considering the optical mode alone, typical fabrication runs result in a spread of the resonance frequency of about $2\mathrm{nm}$ around the centre wavelength. Therefore, finding a pair of matching optical resonances on two chips close to a target frequency currently relies on fabricating a large enough set, in which the probability of obtaining an identical pair is sufficiently high. In fact, this is achievable with a few hundred devices per chip (see Supplementary Information for details).

In addition, a small mismatch in the mechanical frequencies, which is typically around $1\%$ , can readily be compensated by appropriate manipulation of the optical pulse frequencies in the experiment.

For the experiments presented here, we chose a pair of devices with optical resonances at wavelength $\lambda = 1,553.8\mathrm{nm}$ (optical quality factor $Q = 2.2\times 10^{5}$ and $g_0 / (2\pi) = 550\mathrm{kHz}$ and $790\mathrm{kHz}$ for devices A and B, respectively; see Fig. 1). For these structures, the mechanical resonance frequencies are centred around $\Omega_{\mathrm{m}} / (2\pi)\approx 5.1\mathrm{GHz}$ and have a difference of $\Delta \Omega_{\mathrm{m}} / (2\pi) = 45\mathrm{MHz}$ . The two chips are mounted $20\mathrm{cm}$ apart in a dilution refrigerator.

Although we use a single cryostat, there is in principle no fundamental or technical reason for keeping the devices in a common cold environment. For our setup, if the telecommunication fibres linking the two devices were to be unwrapped, our setup would already allow us to bridge a separation of about $70\mathrm{m}$ between the two chips without further modification.

The protocol<sup>25</sup> for the creation and verification of the remote mechanical entanglement consists of three steps (for a schematic,

LETTER

https://doi.org/10.1038/s41586-018-0036-z

<sup>1</sup>Vienna Center for Quantum Science and Technology, Faculty of Physics, University of Vienna, Vienna, Austria. <sup>2</sup>Kavli Institute of Nanoscience, Delft University of Technology, Delft, The Netherlands. <sup>3</sup>These authors contributed equally: Ralf Riedinger, Andreas Wallucks, Igor Marinkovic. *e-mail: sungkun.hong@univie.ac.at; s.groeblacher@tudelft.nl

26 APRIL 2018 | VOL 556 | NATURE | 473

© 2018 Macmillan Publishers Limited, part of Springer Nature. All rights reserved.

![](dt=2026-03-01/ht=18/397a5eafa6040505a362a8a163fd75a6d28195a813a331de52332075b597ffb5.jpg)

![](image)
3b87cebe.jpg)

![](dt=2026-03-01/ht=18/e7efcfcd69b9cf0bde62900a899f6737b7281a0533992f0d94b457ea6f5f65d2.jpg)

Fig. 1 | Devices and experimental setup. a, Optical resonances of device A (grey) and device B (magenta). The Lorentzian fit result (red line) yields a quality factor of $Q \approx 2.2 \times 10^{5}$ for each cavity. b, Mechanical resonances of device A (grey) and device B (magenta). The normalized mechanical resonances are measured through the optomechanical sideband scattering rates. The linewidth is limited by the bandwidth of the optical pulses and filters.

The frequencies of the devices differ by $\Delta \Omega_{\mathrm{m}} / (2\pi) = 45 \mathrm{MHz}$ , which could result in distinguishable photons, potentially reducing the entanglement in the system. We compensate for this shift by tuning the optical pump fields accordingly through serrodyning, erasing any information that could lead to a separable state. c, Experimental setup. We create optical pulses using two lasers, which are detuned to the Stokes (pump) and anti-Stokes (read) transition of the optomechanical cavities.

see Fig. 2). First, the two mechanical resonators are cryogenically cooled, and thus initialized close to their quantum ground states[15,19,22] (see Supplementary Information). Second, a weak 'pump' pulse tuned to the upper mechanical sideband (at frequency $\omega_{\mathrm{pump}} = 2\pi c / \lambda +\Omega_{\mathrm{m}}$ where $c$ is the speed of light), is sent into a phase-stabilized interferometer (with a fixed phase difference $\phi_0$ , see Fig. 1 and Supplementary Information) with one device in each arm.

This drives the Stokes process—that is, the scattering of a pump photon into the cavity resonance while simultaneously creating a phonon[15]. The presence of a single phonon is heralded by the detection of a scattered Stokes photon in one of our superconducting nanowire single-photon detectors. The two optical paths of the interferometer are overlapped on a beam splitter, and a variable optical attenuator is set on one of the arms so that a scattered photon from either device is equally likely to reach either detector.

The heralding detection event therefore contains no information about which device the scattering took place in and thus where the phonon was created. The energy of the pulse is tuned to ensure that the scattering probability $p_{\mathrm{pump}}\approx 0.7\%$ is low, making the likelihood of simultaneously creating phonons in both devices negligible.

The heralding measurement therefore projects the mechanical state into a superposition of a single-excitation state in device A $(|A\rangle = |1\rangle_{\mathrm{A}}|0\rangle_{\mathrm{B}})$ or device B $(|B\rangle = |0\rangle_{\mathrm{A}}|1\rangle_{\mathrm{B}})$ , with the other device remaining in the ground state. The joint state of the two mechanical systems

$$
| \Psi \rangle = \frac {1}{\sqrt {2}} (| 1 \rangle_ {\mathrm {A}} | 0 \rangle_ {\mathrm {B}} \pm \mathrm {e} ^ {i \theta_ {\mathrm {m}} (0)} | 0 \rangle_ {\mathrm {A}} | 1 \rangle_ {\mathrm {B}}) \tag {1}
$$

is therefore entangled, where $\theta_{\mathrm{m}}(0) = \phi_0$ is the phase with which the mechanical state is initialized at delay $\tau = 0$ . This phase is determined from the relative phase difference that the pump beam acquires

in the two interferometer arms<sup>4</sup>, which we can choose using our interferometer lock. However, because the two mechanical frequencies differ by $\Delta \Omega_{\mathrm{m}}$ , the phase of the entangled state will continue to evolve as $\theta_{\mathrm{m}}(\tau) = \phi_0 + \Delta \Omega_{\mathrm{m}}\tau$ . The sign in equation (1) reflects which detector is used for heralding, with $+(-)$ corresponding to the positive (negative) detector, as defined by the sign convention of the interferometer phase $\phi_0$ .

In the third step of our protocol, we experimentally verify the entanglement between the two mechanical oscillators. To achieve this, we map the mechanical state onto an optical field using a 'read' pulse after a variable delay $\tau$ . This relatively strong pulse is tuned to the lower mechanical sideband of the optical resonance $(\omega_{\mathrm{read}} = 2\pi c / \lambda - \Omega_{\mathrm{m}})$ . At this detuning, the field drives the anti-Stokes transition—that is, a pump photon is scattered onto the cavity resonance while annihilating a phonon<sup>15</sup>. Ideally, this state transfer will convert $|\varPsi\rangle$ into

$$
\left| \Phi \right\rangle = \frac {1}{\sqrt {2}} \left(\left| 1 \right\rangle_ {\mathrm {r} _ {\mathrm {A}}} \left| 0 \right\rangle_ {\mathrm {r} _ {\mathrm {B}}} \pm \mathrm {e} ^ {i \left(\theta_ {\mathrm {r}} + \theta_ {\mathrm {m}} (\tau)\right)} \left| 0 \right\rangle_ {\mathrm {r} _ {\mathrm {A}}} \left| 1 \right\rangle_ {\mathrm {r} _ {\mathrm {B}}}\right) \tag {2}
$$

where $\mathrm{r_A}$ and $\mathrm{r_B}$ are the optical modes in the two interferometer arms. The state of the optical field now contains the mechanical phase as well as the phase difference $\theta_{\mathrm{r}}$ acquired by the read pulse. We can add an additional phase offset $\Delta \phi$ to the read pulse in one of the interferometer arms so that $\theta_{\mathrm{r}} = \phi_0 + \Delta \phi$ by using an electro-optic phase modulator, as shown in Fig. 1.

Sweeping $\Delta \phi$ allows us to probe the relative phase $\theta_{\mathrm{m}}(\tau)$ between the superpositions $|A\rangle$ and $|B\rangle$ of the mechanical state for fixed delays $\tau$ . To avoid substantial absorption heating creating thermal excitations in the oscillators, we limit the energy of the read pulse to a state-swap fidelity of about $3.4\%$ , reducing the number of added incoherent phonons to about 0.07 at a delay of $\tau = 123$ ns (see Supplementary Information).

RESEARCH LETTER

474 | NATURE | VOL 556 | 26 APRIL 2018

© 2018 Macmillan Publishers Limited, part of Springer Nature. All rights reserved.

![](dt=2026-03-01/ht=18/b279a188d728cd7c1eede8fb9e5e14d3bf0ecb508b35732841efa78bf3e092f3.jpg)

So far we have neglected the consequence of slightly differing mechanical resonance frequencies for our heralding scheme. To compensate for the resulting frequency offset in the scattered (anti-) Stokes photons and to erase any available 'which device' information, we shift the frequency of the laser pulses by means of serrodyning (see Supplementary Information). Specifically, we use the electro-optic phase modulator, which controls the phase offset $\Delta \phi$ , to also shift the frequency of the pump (read) pulses to device A by $+\Delta \Omega_{\mathrm{m}}(-\Delta \Omega_{\mathrm{m}})$ .

The frequency differences of the pulses in the two opposing paths cancel out their mechanical frequency differences exactly, ensuring that the scattered photons at the output of the interferometer are indistinguishable.

To confirm that the measured state is indeed entangled, we need to distinguish it from all possible separable states, that is, the set of all states for which systems A and B can be described independently. A specifically tailored measure that can be used to verify this nonseparability of the state is called an 'entanglement witness'. Here we use a witness that is designed for optomechanical systems[27].

In contrast to other path-entanglement witnesses based on partial state tomography, such as concurrence, this approach replaces measurements of third-order coherences, $g^{(3)}$ , by expressing them as second-order coherences, $g^{(2)}$ , assuming linear interactions between Gaussian states. This greatly simplifies the requirements and reduces the measurement times for our experiments. Because the coherences refer to the unconditional states, the nonlinear detection and state projection do not contradict these assumptions.

The above assumptions are sati
sfied for our system because the initial mechanical states of our devices are in fact thermal states close to the corresponding quantum ground states (step 1 of our protocol; see Supplementary Information) and we use linearized optomechanical interactions (described in steps 2 and 3)[28]. The upper bound for this witness of mechanical entanglement is given by[27] (see Supplementary Information).

$$
R _ {\mathrm {m}} (\theta , j) = 4 \frac {g _ {r _ {1} , p _ {j}} ^ {(2)} (\theta) + g _ {r _ {2} , p _ {j}} ^ {(2)} (\theta) - 1}{\left(g _ {r _ {1} , p _ {j}} ^ {(2)} (\theta) - g _ {r _ {2} , p _ {j}} ^ {(2)} (\theta)\right) ^ {2}} \tag {3}
$$

in a symmetric setup.

In equation (3), $\theta = \theta_{\mathrm{r}} + \theta_{\mathrm{m}}, j = 1,2$ denotes the heralding detectors and $g_{r_i,p_j}^{(2)} = \langle \hat{r}_i^\dagger \hat{p}_j^\dagger \hat{r}_i\hat{p}_j\rangle /(\langle \hat{r}_i^\dagger \hat{r}_i\rangle \langle \hat{p}_j^\dagger \hat{p}_j\rangle)$ is the second-order coherence between the photons scattered by the pump pulse (with $\hat{p}_j^\dagger$ and $\hat{p}_j$ the creation and annihilation operators, respectively, of the mode going to detector $j$ ) and the converted phonons from the read pulse (with $\hat{r}_j^\dagger$ and $\hat{r}_j$ the

creation and annihilation operators, respectively, of the mode going to detector $j$ ).

For all separable states of the mechanical oscillators A and B, the witness yields $R_{\mathrm{m}}(\theta ,j)\geq 1$ for any $\theta$ and $j$ . Hence, if there exists a $\theta$ and $j$ for which $R_{\mathrm{m}}(\theta ,j) < 1$ , the mechanical systems must be entangled.

Although entanglement witnesses are designed to be efficient classifiers, they typically depend on the individual characteristics of the experimental setup. If, for example, the second beam splitter (see Fig. 1) were to malfunction and act as a perfect mirror—that is, if all photons from device A (B) were transmitted to detector 1 (2)—then $R_{\mathrm{m}}(\theta ,j)$ could still be less than 1 for separable states.

This is because the witness in equation (3) estimates the visibility of the interference between $|A\rangle$ and $|B\rangle$ from a single measurement, without requiring a full phase scan of the interference fringe. To ensure the applicability of the witness, we therefore verify experimentally that our system fulfils its assumptions. We first check whether our setup is balanced by adjusting the energy of the pump pulses in each arm, as described above. This guarantees that the scattered photon fluxes impinging on the beam splitter from both arms are equal (see Supplementary Information).

To make the detection symmetric, we use heralding detection events from both superconducting nanowire single-photon detectors—that is, we obtain the actual bound on the entanglement witness $R_{\mathrm{m,sym}}(\theta)$ from averaging measurements of $R_{\mathrm{m}}(\theta ,1)$ and $R_{\mathrm{m}}(\theta ,2)$ (see Supplementary Information).

By choosing a phase $\theta$ such that the correlations between different detectors exceed the correlations at the same detector, $g_{r_i p_j,i\neq j}^{(2)} > g_{r_i p_j}^{(2)}$ with $i,j\in \{1,2\}$ , we avoid our measurements' susceptibility to unequal splitting ratios applied by the beam splitter.

In Fig. 3, we show a series of measurements of the second-order coherence $g^{(2)}$ , performed by sweeping $\Delta \phi$ with a readout delay of $\tau = 123 \mathrm{~ns}$ which verify the coherence between $|A\rangle$ and $|B\rangle$ . Using these data, we chose an optimal phase setting $\theta = \theta_{\mathrm{opt}}$ with $\Delta \phi = 0.2\pi$ for the main experiment. We obtain $R_{\mathrm{m,sym}}(\theta_{\mathrm{opt}}) = 0.74_{-0.06}^{+0.12}$ which is well below the separability bound of 1.

By including measurements at the non-optimal adjacent phases $\Delta \phi = 0$ and $0.25\pi$ , the statistical uncertainty improves, and we obtain $R_{\mathrm{m,sym}}([\theta_{\mathrm{opt}} - 0.2\pi ,\theta_{\mathrm{opt}} + 0.05\pi ]) = 0.74_{-0.05}^{+0.08}$ . Hence, we experimentally observe entanglement between the two remote mechanical oscillators with a confidence level above $99.8\%$ .

The coherence properties of the generated state can be characterized through the decay of the visibility

$$
V = \frac {\operatorname* {m a x} \left(g _ {r _ {i} , p _ {j}} ^ {(2)}\right) - \operatorname* {m i n} \left(g _ {r _ {i} , p _ {j}} ^ {(2)}\right)}{\operatorname* {m a x} \left(g _ {r _ {i} , p _ {j}} ^ {(2)}\right) + \operatorname* {m i n} \left(g _ {r _ {i} , p _ {j}} ^ {(2)}\right)} \tag {4}
$$

We therefore sweep the delay time $\tau$ between the pump pulse and the read pulse. The mechanical frequency difference $\Delta \Omega_{\mathrm{m}}$ allows us to sweep a full interference fringe by changing the delay $\tau$ by 22 ns. Owing to the technically limited hold time of our cryostat, this sweep had to be performed at a higher bath temperature of about $80 - 90\mathrm{mK}$ (see Fig. 1), yielding a slightly lower, thermally limited visibility at short delays when compared to the data in Fig. 3.

By varying the delay further, we observe interference between $|A\rangle$ and $|B\rangle$ ( $V > 0$ ) up to $\tau \approx 3\mu s$ (see Fig. 4). The loss of coherence can be explained by absorption heating and mechanical decay (see Supplementary Information) and appears to be limited at long delays $\tau$ by the lifetime $1 / \Gamma_{\mathrm{A}} \approx 4\mu s$ of device A, which has the shorter lifetime of the two devices.

LETTER RESEARCH

26 APRIL 2018 | VOL 556 | NATURE

475

© 2018 Macmillan Publishers Limited, part of Springer Nature. All rights reserved.

![](dt=2026-03-01/ht=18/a655c7f2dd838c30f7c2c44684946768ef2b57f5e943689de3abb6b4cc19d362.jpg)

We have experimentally demonstrated entanglement between two engineered mechanical oscillators separated spatially by $20\mathrm{cm}$ and optically by $70\mathrm{m}$ . Imperfections in the fabrication process and the resulting small deviations of optical and mechanical frequencies for nominally identical devices are overcome through the statistical selection of devices and optical frequency shifting using a serrodyne approach. The mechanical systems do not interact directly at any point, but are interfaced remotely through optical photons in the telecommunication-wavelength band.

The coherence time of the entangled state is several microseconds and appears to be limited by the mechanical lifetime of the devices and by absorption heating. Both of these limitations can be considerably mitigated. On the one hand, optical absorption can be substantially suppressed by using intrinsic, desiccated silicon[29]. Mechanical lifetimes, on the other hand, can be greatly increased by adding a phononic bandgap shield[22].

Although our devices are engineered to have short mechanical lifetimes[19,30], earlier designs including such a phononic shield have reached[22] $1 / \Gamma \approx 0.5\mathrm{ms}$ and could still be further improved. Combined with reduced optical absorption, which would allow efficient laser cooling, such lifetimes can potentially put our devices on par with other state-of-the-art quantum systems[31].

Our experiment demonstrates a protocol for realistic, fibre telecommunication-compatible entanglement distribution using engineered mechanical quantum systems. With the current parameters of our system, a device separation of $75\mathrm{km}$ using commercially available telecommunication fibres would result in a drop of less than $5\%$ in the interference visibility (see discussion in Supplementary Information for more details). The system presented here is directly scalable to include more devices (see Supplementary Information) and could be integrated into a real quantum network.

Combining our results with those of optomechanical devices capable of transferring quantum information from the optic
al to the microwave domain, which is a highly active field of research[24,32,33], could provide a backbone for a future quantum internet based on superconducting quantum computers.

![](dt=2026-03-01/ht=18/cbbeeaab4a930713efda497d7769244e1d829bd74ea4317e2d788ef7fb5f4b26.jpg)

# Data availability

All relevant data generated and analysed during this study are included in this paper (and its Supplementary Information).

Received: 24 October 2017; Accepted: 2 March 2018; Published online 25 April 2018.

RESEARCH LETTER

476

| NATURE | VOL 556 | 26 APRIL 2018

© 2018 Macmillan Publishers Limited, part of Springer Nature. All rights reserved.

Additional information

Supplementary information is available for this paper at https://doi.org/10.1038/s41586-018-0036-z.

Reprints and permissions information is available at http://www.nature.com/reprints.

Correspondence and requests for materials should be addressed to S.H. or S.G.  
Publisher's note: Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

LETTER RESEARCH

26 APRIL 2018 | VOL 556 | NATURE

477

© 2018 Macmillan Publishers Limited, part of Springer Nature. All rights reserved.