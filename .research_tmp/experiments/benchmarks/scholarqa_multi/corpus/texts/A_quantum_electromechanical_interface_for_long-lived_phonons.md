# A quantum electromechanical interface for long-lived phonons

Alkim Bozkurt, $^{1,2}$ Han Zhao, $^{1,2}$ Chaitali Joshi, $^{1,2}$ Henry G. LeDuc, $^{3}$ Peter K. Day, $^{3}$ and Mohammad Mirhosseini $^{1,2,*}$

<sup>1</sup> The Gordon and Betty Moore Laboratory of Engineering,

California Institute of Technology, Pasadena, California 91125

$^{2}$ Institute for Quantum Information and Matter,

California Institute of Technology, Pasadena, California 91125

$^{3}$ Jet Propulsion Laboratory, California Institute of Technology, Pasadena, California 91109

(Dated: July 25, 2022)

Controlling long-lived mechanical oscillators in the quantum regime holds promises for quantum information processing. Here, we present an electromechanical system capable of operating in the GHz-frequency band in a silicon-on-insulator platform. Relying on a novel driving scheme based on an electrostatic field and high-impedance microwave cavities based on TiN superconductors, we are able to demonstrate a parametrically-enhanced electromechanical coupling of $g / 2\pi = 1.1$ MHz, sufficient to enter the strong-coupling regime with a cooperativity of $\mathcal{C} = 1200$ .

The absence of piezoelectric materials in our platform leads to long mechanical lifetimes, finding intrinsic values up to $\tau_{\mathrm{d}} = 265$ $\mu \mathrm{s}$ $Q = 8.4 \times 10^{6}$ at $\omega_{\mathrm{m}} / 2\pi = 5$ GHz) measured at low-phonon numbers and millikelvin temperatures. Despite the strong parametric drives, we find the cavity-mechanics system in the quantum ground state by performing sideband thermometry measurements.

Simultaneously achieving ground-state operation, long mechanical lifetimes, and strong coupling sets the stage for employing silicon electromechanical resonators as memory elements and transducers in hybrid quantum systems, and as a tool for probing the origins of acoustic loss in the quantum regime.

Phonons, the quanta of energy stored in vibrations in solids, promise unique opportunities for storing and communicating quantum information. The intrinsic mechanisms for phonon dissipation get suppressed at low temperatures [1], leading to extremely low acoustic loss in single crystalline materials [2, 3]. Additionally, the inability of sound waves to propagate in vacuum makes it possible to trap phonons in wavelength-scale dimensions via geometric structuring, leading to near-complete suppression of environment-induced decay [4].

Finally, phonons interact with solid-state qubits and the electromagnetic waves across a broad spectrum, making them near-universal intermediaries for cross-platform information transfer [5]. Motivated by these properties, pioneering work in the past two decades has enabled sensitive measurement and control of mechanical oscillators in the quantum regime via optical and electrical interfaces, making them viable candidates for quantum sensors, memories, and transducers [6, 7].

While optomechanical experiments have been successful in measuring phonons with millisecond-to-second lifetimes [4, 8], accessing long-lived mechanical resonances with electrical circuits has been more challenging. In the gigahertz frequency range, where the spectral proximity to superconducting qubits holds the most promise for quantum technologies, piezoelectricity is the predominant mechanism for converting microwave photons to phonons. Piezoelectric devices have been used with remarkable success in coupling mechanical modes to superconducting qubits [9-11]. However, their need for hybrid material integration, sophisticated fabrication process, and reliance on lossy poly-crystalline materi

als [12] has limited the state-of-the-art experiments to sub-microseconds mechanical lifetimes in devices with compact geometries [11, 13]. This evidently large gap between the mechanical lifetimes accessible to optical and electrical interfaces motivates pursuing less invasive forms of electromechanical interaction. Creating better electrical interfaces for long-lived phonons holds the potential for revolutionizing our current quantum toolbox by pairing the superior coherence of acoustics with the massive nonlinearity of Josephson junction circuits [14].

Here, we realize electromechanical coupling between microwave photons in a superconducting circuit and long-lived phonons in a 5-GHz crystalline silicon oscillator. To achieve this, we rely on electrostatic transduction, where we use a static electric field, as opposed to conventionally used radio-frequency drives, to realize a parametrically-enhanced interaction in a microwave cavity with a motion-dependent capacitor.

The absence of alternating currents from the driving field in this scheme eliminates conductive loss, allowing us to achieve large parametrically-enhanced coupling rates without causing heating in the system. To further enhance electromechanical interactions, we rely on frequency-tunable high-impedance microwave resonators made from TiN superinductors. Relying on these innovations, we are able to demonstrate electromechanical interaction in the strong coupling regime, enabling the coherent exchange of microwave photons and phonon at a cooperativity of $\mathcal{C} \approx 1200$ .

We measure mechanical lifetimes in the few-phonon regime, demonstrating quality factors in excess of 8 million (at $5\mathrm{GHz}$ ) in our best devices. To the best of our knowledge, this is the highest value measured via electrical interfaces in this frequency band. Crucially, we observe no parasitic heating for a large range of electrostatic biasing fields in our system, allowing us

* mohmir@caltech.edu; http://qubit.caltech.edu

arXiv:2207.10972v1 [quant-ph] 22 Jul 2022

![](dt=2026-06-05/ht=21/b3f19697b422681072d1eb4e4fca0dd65e4f71ef408b3aecede4230ede1d2126.jpg)

![](dt=2026-06-05/ht=21/16811df3c71a2ca73bccadd57750871d93f9433f4004127a71636959f0401d8d.jpg)

![](dt=2026-06-05/ht=21/978075c3f5bde7d85459fedb4523f0f7a42acdbdaa89449c2e0275e95d0bfcac.jpg)

![](dt=2026-06-05/ht=21/dbb9479cdad1edee9f79f1669ca22a6c154c275a9399281cd83424b847870f62.jpg)

![](dt=2026-06-05/ht=21/9b466a6cd2043fd16fd11a1ae9b8de7918411fdd4bf128c0e112b6cb50216c1b.jpg)

![](dt=2026-06-05/ht=21/663e3968474fcc11ea32454531571237bb1775407c71ac36ac34fd45558cd20a.jpg)

to operate in the quantum ground state as verified by calibrated sideband thermometry measurements in a dilution refrigerator. The combination of long lifetimes, strong interaction, and the compact geometry of our platform promises future experiments in employing mechanical modes as microwave-frequency quantum memories, improved microwave-optical transducers, and new measurement capabilities for exploring the origins of acoustic loss crystalline materials.

# A PHONONIC CRYSTAL ELECTROSTATIC TRANSDUCER

The operating principles of our experiment can be understood by considering a capacitor with mechanically moving electrodes connected to an external DC voltage source. In this setting, the mechanical vibrations of the capacitor electrodes create a time-dependent dipole oscillating at mechanical resonance frequency. Connecting this charged moving capacitor (i.e. the transducer) to an

electromagnetic cavity (see fig. 1a) leads to an interaction between the voltage operator of the photons in the cavity $(\hat{V} / V_{\mathrm{zpf}} = i(\hat{a} - \hat{a}^{\dagger}))$ and the quantized mechanical displacement operator $(\hat{x} / x_{\mathrm{zpf}} = (\hat{b} + \hat{b}^{\dagger}))$ . This interaction can be described by the Hamiltonia
n (see appendix B)

$$
\hat {H} _ {\mathrm {i n t}} = i \left(x _ {\mathrm {z p f}} \partial_ {x} C\right) V _ {\mathrm {z p f}} V _ {\mathrm {D C}} \left(\hat {a} \hat {b} ^ {\dagger} - \hat {a} ^ {\dagger} \hat {b}\right) = i \hbar g _ {\mathrm {e m}} \left(\hat {a} \hat {b} ^ {\dagger} - \hat {a} ^ {\dagger} \hat {b}\right) \tag {1}
$$

Here, $x_{\mathrm{zpf}}$ and $V_{\mathrm{zpf}}$ represent the zero-point motion and voltage of the phonon and photon fields, respectively. The coupling rate is a function of the geometry (through $\partial_x C$ ) and the applied bias voltage $V_{\mathrm{DC}}$ , and arises as a result of the change in the stored electrostatic energy as a function of mechanical motion. The DC voltage in this process can be understood as a 'pump' in a parametric process [15].

Unlike the conventional parametric electromechanics, however, the pump is solely comprised of electric fields at zero frequency and is not accompanied by alternating currents. As we will see, this distinction is crucial in our experiment because it increases the net coupling rate at large voltages without being limited by the dissipation in the superconducting cavity [16, 17].

2

Despite its conceptual simplicity, electrostatic transduction is challenging to realize at GHz frequencies [18, 19]. Getting substantial coupling requires increasing the motion-dependent capacitance and the zero-point displacement. This combination has been previously achieved in low-mass, narrow-gap suspended capacitors, which support MHz-frequency mechanical resonances [20]. However, the frequency scaling of acoustic loss in metals (speculated to be caused at grain boundaries [12, 21, 22]) makes these structures unsuitable for GHz frequencies. Additionally, the short wavelength of GHz-frequency phonons leads to increased acoustic radiative loss to the surrounding environment, making it challenging to localize high- $Q$ resonances.

Our solution is to utilize planar nano-structured devices made from crystalline silicon membranes. Adding a thin layer of metal on top of the membrane allows us to form a capacitor in this platform while relying on phononic crystals to engineer localized resonances. Our transducer consists of a phononic crystal resonator made of a periodic array of multiple unit cells (see fig. 1b), patterned on the inner electrode of a vacuum-gap capacitor. The in-plane movement of the 'breathing' mechanical mode in this structure leads to the modulation of the capacitance.

We have maximized the rate of change of this motion-dependent capacitance by fabricating capacitors with narrow gaps in the range of $65 - 70\mathrm{nm}$ . Additionally, we have maximized the capacitance by increasing the number of phononic crystal unit cells to the limit set by the onset of disorder effects, which lead to mode breakup as observed in finite-element simulations (see appendix D). A key benefit of this planar geometry is the possibility of creating phononic crystals with a wide band gap for all phonon polarizations [4].

We use these 'phononic shields' for clamping the transducer to its surrounding membrane (see fig. 1c, d).

The size mismatch is a central challenge in coupling the presented phononic crystal transducer to a microwave circuit. Set by the small wavelength of phonons ( $\sim 1\mu \mathrm{m}$ at the target frequency of $5\mathrm{GHz}$ ), the small size of the transducer translates to a motion-dependent capacitance that is much smaller than the typical capacitance of a microwave cavity. This mismatch leads to a poor electric energy density overlap (see appendix B), which dilutes the electromechanical interaction.

Formally, this effect is captured by a linear dependence of the electromechanical coupling to the zero-point voltage of microwave photons ( $V_{\mathrm{zpf}}$ , see eq. (1)). Recent progress in developing circuits for error-protected superconducting qubits has led to established techniques [23-25] for magnifying zero-point voltage of microwave photons in high-impedance resonators. In our experiment, we achieve this by galvanically connecting the transducer to a microwave resonator (fig.

1d) consisting of a $110\mathrm{-nm}$ wide nanowire formed from thin-film titanium nitride (thickness $\sim 15$ nm). The inertia of charge carriers in this disordered superconductor leads to a large inductance, which is enhanced by patterning structures with small cross sections

![](dt=2026-06-05/ht=21/a1ed88957c418bdbff18b1c73cf69e070e9b1c382af8cf941d3a008883ab8544.jpg)

![](dt=2026-06-05/ht=21/2f031b22d7f6eb91a1aa0aa60e0155b09d8bb263b379fb6d5fe9170dcdb5e5f7.jpg)

![](dt=2026-06-05/ht=21/a82baee2219a673284664957578d382d04c426e0fb0309ef679882442a228e13.jpg)

![](dt=2026-06-05/ht=21/8f98f2634a78b9715c89de06e0305db53b11b24ae1aea51c4f24990c527fafa7.jpg)

[23, 26]. Beyond magnifying the electromechanical interactions, kinetic inductance is tunable via external magnetic fields, which allows us to control the resonance frequency of the microwave cavity in-situ [27].

# CHARACTERIZATION AND MEASUREMENTS

We have characterized fabricated electromechanical resonators in a dilution refrigerator with a base temperature of $20\mathrm{mK}$ (see fabrication details at appendix A). A coplanar waveguide is connected to the device for simultaneously applying DC voltages to our mechanical capacitor and probing our microwave resonator in reflection via its coupling to the waveguide. In the absence of electromechanical coupling, we can measure the bare microwave cavity response using a vector network analyzer (VNA, see fig. 2a). To locate the mechanical resonance, we apply a DC voltage and continuously tune the frequency of the microwave resonator via an external magnetic field (see appendix C). The electromechanical interaction leads to a large reflection at the point

3

where the microwave and mechanical frequencies cross, in a phenomenon known as the electromechanically-induced transparency (EIT) (fig. 2b) [20]. We extract the electromechanical coupling rate using a fit to the theory expressions for the EIT response (see supplementary appendix B, fig. 2c), and plot it as a function of the applied voltage in fig. 2d. As evident, the coupling rate is found to be a linear function of the voltage bias with a slope $(g_{0,\mathrm{B}} / 2\pi = 45.4\pm 1.1\mathrm{kHz} / \mathrm{V})$ closely matching the results from numerical modeling (see appendix D).

In addition, we present measurement results from a second device with an identical geometry, but with a small coupling $g_{0,\mathrm{A}} / 2\pi = 22.0\pm 1.4\mathrm{kHz} / \mathrm{V}$ , which is likely impacted by mode-breakup due to fabrication disorder (see appendix D). An interesting feature in the data is the small non-zero coupling at the zero-voltage bias, where zero coupling is achieved at a negative offset voltage.

This feature is found to be persistent in a shorted capacitor geometry and in the absence of a voltage source, which may indicate the presence of trapped charges in the transducer [28].

# COHERENCE PROPERTIES

We next investigate the coherence properties of our devices by exciting the mechanical resonator with a pulse and registering its free decay via electromechanical readout (see appendix F). In this measurement, the electromechanical readout rate is a function of the detuning between the mechanical and microwave resonances $\Gamma_{\mathrm{em}} = g^2\kappa /(\Delta^2 +(\kappa /2)^2)$ . At any given detuning, the total decay rate of the mechanics is given by $\Gamma = \Gamma_{i} + \Gamma_{\mathrm{em}}$ where $\Gamma_{i}$ is the intrinsic decay rate.

In order to precisely measure $\Gamma_{i}$ , we perform mu
ltiple measurements where we gradually increase $\Delta$ , leading to a gradual reduction in $\Gamma_{\mathrm{em}}$ until the total decay becomes dominated by the intrinsic part. Fitting the total decay rate expression as a function of $\Delta$ , we extract an intrinsic $\tau_{\mathrm{d}}$ lifetime of $265\pm 25~\mu s$ (fig. 3a), corresponding to a Q factor of $8\times 10^{6}$ .

At the largest detuning we achieve, we can reach the regime where the intrinsic decay dominates the dynamics and obtain a total energy decay lifetime $\tau_{\mathrm{d}}$ of $220\pm 6~\mu s$ (fig. 3b), which strongly supports the large intrinsic lifetime inferred from the fits. We note that the measured mechanical lifetime is remarkably large for a device with compact geometry, corresponding to a quality factor that is more than two orders of magnitude large than the state-of-the-art piezoelectric devices [11].

Apart from the energy relaxation lifetime, the coherence time of our mechanics bears significance for future quantum applications. We find the coherence time as the reciprocal of the linewidth extracted from fitting the EIT response in the large detuning regime $(\Delta >> \Gamma_i)$ . Using this method, we find $\tau_{\mathrm{c}} \approx 5 \mu s$ (fig. 3b), a value that is substantially shorter than the lifetime. This discrepancy between the decay and coherence times hints at the presence of frequency jitter and has been previ

ously observed in optomechanical experiments with silicon phononic crystal resonators. While the exact nature of this dephasing remains unknown, it has been hypothesized to be caused by interaction with the two-level system (TLS) defects [4, 8]. To better understand possible dephasing and decay sources, we carry out ringdown measurements for different numbers of phonons in the mechanical resonator.

We change the intra-cavity phonon number in accordance with the readout efficiency $\Gamma_{\mathrm{em}} / \Gamma$ , keeping the output detection powers nearly constant at the lowest levels detectable by our amplifiers to ensure we can reach the low-phonon regime while maintaining feasible measurement times. Interestingly, we observe exponential decay with no sign of power dependence down to the single-phonon level (see fig. 3c). Similarly, we find the coherence time extracted from EIT measurements to be insensitive to phonon numbers in the range of 4-500 (see appendix F).

Although we do not find signatures of saturable loss (a common signature of a TLS bath) in this measurement setting, we find that the coherence and decay times change as a function of the applied DC voltage, providing direct evidence for the presence of TLS defects. This behavior is explored in detail further in the manuscript.

# PROBING THE LIMITS OF PARAMETRIC-ENHANCEMENT

A naturally emerging question is about the maximum achievable rate of the electromechanical interactions in our platform. This rate is set by the magnitude of the DC voltage that can be applied before the onset of any spurious heating or instabilities in the system.

We do not expect to see any significant leakage current passing through the devices because of the freezing of the charge carriers at the low measurement temperatures. However, applying large voltages to the narrow-gap capacitors in our devices leads to strong electric fields which may lead to ionization and dielectric breakdown [29]. We measure the leakage current through the transducer structures as a function of applied voltage. As evident in fig.

4a, we see a leakage current (with a characteristic resistance of $\sim 500\mathrm{G}\Omega$ ) that is found to be dominantly caused by the cables in our measurement, bounding the actual leakage through the device to below the measurement sensitivity of our measurement. When increasing the voltage beyond $30\mathrm{V}$ , we observe an abrupt spike in the leakage current that was initially attributed to dielectric breakdown.

However, imaging of our devices post warm-up at the room-temperature indicates that the sudden leakage is most likely due to the onset of pull-in instabilities, resulting in the shut-down of the capacitor gap (see appendix E). Repeating this experiment on four identical test devices, we find the onset of this instability in a consistent range (29-31 V).

To probe the signatures of any potential pump-induced heating (previously observed in similar devices under

4

![](dt=2026-06-05/ht=21/102e714e6f14fe859366c89c735a8d9f55f0bc9ac8512e56de563f8ddb252846.jpg)

![](dt=2026-06-05/ht=21/835bd389e8d4db5fb47cd795c472116966c881172738b0594b136d69840c1653.jpg)

![](dt=2026-06-05/ht=21/d4a516f69cdbd62bfbea8f5c4adce927fac60344e6ea42e51e12c5bc4f037702.jpg)

radio-frequency drives [17]), we perform side-band thermometry [15]. The thermometry measurement process for the mechanical resonator is visualized in fig. 4c. We drive the system with a weak tone and measure the incoherent emission from the inelastic scattering of the drive to locate the mechanical resonance (see appendix E) [30]. A subsequent measurement with no drives shows a negligibly small emission, which is calibrated in experiments with long averages to extract the resonator's occupation.

We have found the system of mechanics-microwave to be in the quantum ground state for the entire range of applied voltages in our device $(0 - 25\mathrm{V})$ . Despite observing no heating, we note that the presence of a strong electromechanical back-action cooling (for the mechanical mode) and radiative cooling through the on-chip waveguide (for the microwave cavity) may mask a weak heating process.

To find a more sensitive trace of any potential heating, we perform thermometry in a regime where the microwave cavity is detuned far away from the mechanical resonator to reduce the effect of electromechanical back-action cooling and deduce the temperature of the phenomenological intrinsic baths that the mechanical and microwave resonators interact with (see appendix E for details). Figure 4b shows the measured microwave $(n_{\mathrm{b,r}})$ and mechanical bath occupancy $(n_{\mathrm{b,m}})$ as a function of the applied DC bias.

As evident, the baths remain in the ground state for a large voltage range, with the exception of the mechanical bath at $25\mathrm{V}$ , where the occupancy rises to $0.86\pm 0.08$ phonons (attempts to do measurements at higher voltages were unsuccessful due to the onset of the pull-in instability). The slightly higher occupation of the microwave bath across all voltages (including $0\mathrm{V}$ ) is likely caused by the absence of IR-shielding in our measurement setup, which leads to

the generation of quasiparticles, and can raise the microwave bath temperature in devices with a large kinetic inductance [31, 32]. Regardless, despite the observation of these subtle features testifying to the accuracy of our measurement technique, we do not observe any indication of significant pump-induced heating in the mechanical resonator or the microwave cavity for a wide range of voltages, which is a promising feature of our electrostatic driving scheme.

# THE STRONG-COUPLING REGIME

Having discovered the safe range of voltages we can apply to our devices, we investigate the maximum electromechanical coupling rate within reach. For this purpose we use the device B, which has larger $g_{0}$ . Setting the external voltage to $25\mathrm{V}$ , we sweep our microwave frequency by changing the external magnetic field, which gives rise to an avoided crossing feature between the microwave and mechanics as seen in fig. 5a. Fitting the frequencies of the two hybridized modes, we extract a parametrically-enhanced electromechanical coupling rate of $g = 1.08\mathrm{MHz}$ .

This value corresponds t
o $g_{0} = 42.7\mathrm{kHz / V}$ which matches the values we have calculated at low voltages, indicating that the parametric enhancement scales linearly with voltage in the entire measurement range. Achieving the strong-coupling regime is manifested clearly in the measurements of the reflection spectrum at resonance, where we see a pair of hybridized modes with linewidths of $(\kappa +\gamma) / 2 = 692\mathrm{kHz}$ , satisfying $2g > \kappa +\gamma$ (fig. 5b).

Achieving the strong-coupling regime in our system is significant as it allows the coherent exchange of phonons and microwave photons, a

5

![](dt=2026-06-05/ht=21/760002bf4bcce492d94cc3057b82b952c56435c614cd3b7760a74b69f6f15e3b.jpg)

![](dt=2026-06-05/ht=21/d5c9bff43004f8bbd4f9fe2ba127fb2f2ff1fcff004a5ffc38b8da175cff6811.jpg)

![](dt=2026-06-05/ht=21/1bf780e714bf6a2d7b3f710b602a14e8640b3f2890c60a45492be57851469c75.jpg)

pre-requisite for utilizing electromechanical systems in a range of quantum applications. Another key figure of merit in coupling mechanical modes to qubits and microwave resonators is cooperativity, defined as the ratio of the electromechanical readout rate to the intrinsic mechanical decay rate $\mathcal{C} = 4g^2 /(\kappa_i\Gamma_i)$ . We use the measured electromechanical coupling rates and the mechanical intrinsic decay rates (from the ringdown measurements) to find the cooperativity as a function of bias voltage (see fig. 5b).

For the maximum voltage value of 25 volts, we find $\Gamma_{i} / 2\pi = 4.8~\mathrm{kHz}$ ( $\tau_{\mathrm{d}} = 33~\mu s$ ), corresponding to a cooperativity of 1270 (see appendix E). As a guide, we also mark the values of cooperativity assuming the maximum coupling rates and lifetimes measured across different devices on the same plot, finding estimates that exceed $10^{4}$ at $25\mathrm{V}$ .

# INTERACTION WITH TWO-LEVEL SYSTEMS

As noted earlier, we suspect that the mechanical dephasing in our measurements can be attributed to coupling to TLS, which is previously shown to be the dom

![](dt=2026-06-05/ht=21/805be89d828ba1048292a3e2388e255bd646b73a8ab9b62e9d14e3389cd25e0d.jpg)

![](dt=2026-06-05/ht=21/8f80347ac5059346e88b59ca94b3c49ba60360720c5a55828c26e53ef152ce8e.jpg)

![](dt=2026-06-05/ht=21/bcd171a9f3707f22e81b659624fc14c8f96f02a661521c951fe9581ef95d9aca.jpg)

inant loss mechanism for acoustic resonators with substantial surface participation at millikelvin temperatures [4, 12]. Modeled phenomenologically as two nearly degenerate energy configurations of electrons in amorphous materials, a TLS manifests as a resonant defect with both electrical and acoustic susceptibilities [33, 34].

Using previous theory work as a guide [35] and extracting the spatial distribution of the strain field from FEM numerical modeling, we estimate the spectral density (3/ GHz) and the coupling rate (13 MHz) of individual TLS defects to the nanomechanical resonators in our system. The large coupling rate and the small density suggest a departure from the continuum TLS-bath picture observed in the past work [36, 37], and offers the possibility of observing mechanics-TLS interactions at the individual defect level.

To make this observation, we take advantage of the TLS frequency tuning via the Stark shift [38] from the electrostatic bias in our system (estimated to create TLS frequency shifts at a rate of $20\mathrm{GHz} / \mathrm{V}$ , see appendix F). Figure 6a shows the measured decay and coherence times as a function of voltage, manifesting strong modifications indicative of interaction with TLS defects. Further, we make a measurement of the mechanical spectrum as a function of voltage, observing abrupt frequency shifts (in form of avoided crossings) commensurate with the changes in the coherence and decay times. Finally, by comparing measurement results at two probe powers, we see a saturation behavior in the vicinity of the voltage

6

![](dt=2026-06-05/ht=21/9db1ad58970881dfea6cbd08623cf69c7bba3b4511d43dec2930ad96ebf15ce0.jpg)

![](dt=2026-06-05/ht=21/0be2ebedb834efcb27c93957bc1235305ce77a763d695c95c68d1e0b3447b9b2.jpg)

![](dt=2026-06-05/ht=21/6bbc9e210d111e0fd5e76b0172dcfeae3191577821ffb7359ff770a6be580054.jpg)

values where the mechanical mode is heavily affected by TLS (see fig. 6b,c). A more detailed measurement of mechanical linewidth as a function of phonon number at these points provides a good fit to the widely used TLS model (see appendix F). We note that in our device we cannot fully saturate the signatures of the TLS before the onset of acoustic Kerr nonlinearities at large phonon numbers [39].

# CONCLUSIONS AND OUTLOOK

In conclusion, we present an integrated cavity electromechanical system capable of achieving $\mathrm{MHz}$ -level coupling rates at a mechanical frequency of several GHz. Using this system, we demonstrate achieving the strong coupling regime, with a cooperativity exceeding 1200. Relying on an electrostatic driving field, we are able to obtain a large parametric enhancement of the interaction with negligible parasitic heating, leading to operation in the quantum ground state.

Device fabrication is performed using a TiN-on-SOI material system, which is compatible with superconducting qubits and optomechanical crystals. Additionally, by relying on thin films and single-crystalline silicon, we are able to show mechanical quality factors above 8 million, corresponding to two orders of magnitude improvement over piezoelectric devices in similar geometries [11, 13]. Finally, we note the material-agnostic nature of the underlying process in our experiment, which holds the potential for adoption in platforms hosting spin qubits [40].

Looking ahead, we envision several avenues for further improvement of the presented devices. The electromechanical coupling rates can be readily increased by multiple folds upon integration of electrostatic transducers with microwave cavities with ultra-high impedance [25, 41], reaching full parity with piezo-electric platforms.

Additionally, while we observe record-long lifetimes in devices with electrical connectivity, our measurements remain much shorter than the second-long results from optomechanical experiments in silicon structures with no metallic components [4, 8]. This observation motivates a systematic study of the sources of residual acoustic loss, including the role of metallic components, fabrication disorder in the acoustic shields, and two-levelsystem defects.

A better understanding of the loss mechanisms along with the implementation of proper mitigation techniques is expected to lead to longer mechanical lifetimes. With moderate improvements, achieving the millisecond regime is expected to be within reach in near future, with the potential to deliver transformative impacts on mechanics-based microwave-optical interconnects [7], error-protected bosonic qubits [42], and quantum memories [43, 44].

# ACKNOWLEDGMENTS

We thank Oskar Painter and Mahmud Kalaee for fruitful discussions, which led to the conception of this work. This work was supported by the startup funds from the EAS division at Caltech, National Research Foundation (grant No. 2137776), and a KNI-Wheatley scholarship. C.J. gratefully acknowledges support the IQIM/AWS Postdoctoral Fellowship. M.M
gratefully acknowledges support from the Q-NEXT.

7

# 1. Fabrication

A 4-inch silicon on insulator (SOI) wafer is utilized at the beginning of the fabrication process, and is covered with $\sim 15$ nm TiN films via sputtering [45]. Following sputtering, the wafer is diced into $1\times 1\mathrm{cm}^2$ dies. For the following steps the patterning of the structures is achieved by electron beam lithography. (i) Deposition of niobium markers followed by lift-off. (ii) Inductively coupled plasma - reactive ion etching (ICP-RIE) of TiN and Si with $\mathrm{SF}_6 / \mathrm{Ar}$ and $\mathrm{SF}_6 / \mathrm{C}_4\mathrm{F}_8$ chemistry.

These etching steps are used to define the capacitor vacuum gap, phononic crystal nanobeam, phononic shields and the release holes throughout the metalized and bare sections of the Si membrane. The devices are released with Hydrogen Fluoride (HF) post-fabrication.

# 2. Measurement Setup

The chip is wirebonded to a PCB, which is placed into a copper box and then mounted to the mixing stage of the dilution refrigerator at $\sim 15\mathrm{mK}$ . The box has a coil on top for magnetic field tuning the microwave resonators. The coil is obtained by hand winding a superconducting wire around a cylindrical extrusion.

The device is measured in reflection with the aid of a cryogenic circulator. A bias tee is placed between the chip and the circulator to enable DC biasing of the transducer and readout of the microwave cavity via the same CPW. The RF input line consists of multiple cascaded attenuators with a total attenuation of 74 dB. A tunable attenuator is further added to the input line to control the input power in a programmable manner. The DC input has no attenuation and is directly attached to the bias tee (low frequency transmission band up to 500 MHz). At the output, we have an amplifier chain that consists of a HEMT amplifier thermalized to the 4K stage and a room temperature amplifier, with a total gain of $\sim 65$ dB.

The external DC voltage for the electromechanical interaction and the current for the tuning coil are applied via a multi-channel programmable low-noise DC source. Since, significant amount of current is required to tune the microwave resonators, the normal metal parts in the coil wiring leads to spurious heating of the mixing stage. The coherent response of the mechanics-cavity system is probed via a vector network analyzer (VNA) in reflection. For thermometry and investigations of the full driven response of the system, the microwave emission is detected by a spectrum analyzer.

For time-resolved measurements, we use Quantum Machines OPX+ module. This tool enables the generation of pulses with an arbitrary waveform generator (AWG), heterodyne detection, demodulation of signals via a digitizer, and the processing of detected signals with an FPGA.

As part of our experiment we have fabricated and extensively characterized two devices. The parameters obtained from the measurements are described in table I. Based on these parameters, we select different devices to exhibit specific features of our platform. Due to its larger electromechanical coupling strength, device B can enter the strong coupling regime and attain large cooperativities above 1000.

On the other hand, device A has the largest energy decay lifetime we have observed in our platform and is well suited to demonstrate the substantial lifetimes that can be achieved with our devices. Furthermore, as it requires less frequency tuning (due to a smaller mechanics-cavity detuning at zero magnetic field), we are able to measure it at a lower temperature $(20\mathrm{mK})$ where the heating due to the current on the coil is minimized, which makes it more suitable to perform sensitive sideband thermometry measurements and investigate TLS physics without thermal saturation of TLS.

# Appendix B: Electrostatic Interaction

# 1. Hamiltonian Derivation

The interaction term in the Hamiltonian for a capacitor with mechanically moving electrodes $(C_{\mathrm{m}})$ is given as

$$
H _ {\mathrm {i n t}} = \frac {\hat {q} ^ {2}}{2 C _ {\mathrm {m}}} \left(\frac {1}{C _ {\mathrm {m}}} \frac {\partial C _ {\mathrm {m}}}{\partial x} \hat {x}\right). \qquad \mathrm {(B 1)}
$$

The displacement operator can be written as $\hat{x} = x_{\mathrm{zpf}}(\hat{b} + \hat{b}^{\dagger})$ . We have both electrostatic charge due to external voltage source and RF charge associated with the microwave resonance on top of the capacitor. This leads to a charge operator which can be written as $\hat{q} = iQ_{\mathrm{zpf}}(\hat{a} - \hat{a}^{\dagger}) + Q_{\mathrm{DC}}$ . Inserting these operators into the Hamiltonian, we obtain

$$
H _ {\mathrm {i n t}} = \frac {1}{2} \frac {\partial C _ {\mathrm {m}}}{\partial x} x _ {\mathrm {z p f}} \left[ i \frac {Q _ {\mathrm {z p f}}}{C _ {\mathrm {m}}} (\hat {a} - \hat {a} ^ {\dagger}) + \frac {Q _ {\mathrm {D C}}}{C _ {\mathrm {m}}} \right] ^ {2} (\hat {b} + \hat {b} ^ {\dagger}). \tag {B2}
$$

Noting that $Q_{\mathrm{zpf}} / C_{\mathrm{m}} = V_{\mathrm{zpf}}$ , $Q_{\mathrm{DC}} / C_{\mathrm{m}} = V_{\mathrm{DC}}$ and expanding the terms

$$
\begin{array}{l} H _ {\mathrm {i n t}} = \frac {1}{2} \frac {\partial C _ {\mathrm {m}}}{\partial x} x _ {\mathrm {z p f}} (\hat {b} + \hat {b} ^ {\dagger}) \times \\ \left[ - V _ {\mathrm {z p f}} ^ {2} (\hat {a} - \hat {a} ^ {\dagger}) ^ {2} + V _ {\mathrm {D C}} ^ {2} + 2 i V _ {\mathrm {z p f}} V _ {\mathrm {D C}} (\hat {a} - \hat {a} ^ {\dagger}) \right]. \quad \mathrm {(B 3)} \\ \end{array}
$$

Keeping only the interaction terms between the DC voltage and the RF fields and carrying out the rotating wave approximation for our case of mechanics in resonance with microwave gives us

$$
H _ {\mathrm {i n t}} = i \frac {\partial C _ {\mathrm {m}}}{\partial x} x _ {\mathrm {z p f}} V _ {\mathrm {z p f}} V _ {\mathrm {D C}} (\hat {a} \hat {b} ^ {\dagger} - \hat {a} ^ {\dagger} \hat {b}) = i \hbar g _ {\mathrm {e m}} (\hat {a} \hat {b} ^ {\dagger} - \hat {a} ^ {\dagger} \hat {b}). \tag {B4}
$$

8

Appendix A: Methods

3. Device Parameters

TABLE I. Summary of device parameters. The referenced maximum decay lifetime and the coherence time are at the few-phonon level.

![](dt=2026-06-05/ht=21/a43c3319a525c2c02e044adac7df75946d368dcec50b7d2e2443113f627c2feb.jpg)

<table><tr><td>Device</td><td>ωm/2π (GHz)</td><td>ωrmax/2π (GHz)</td><td>κi/2π (kHz)</td><td>κe/2π (kHz)</td><td>g0/2π (kHz/V)</td><td>τdmax(μs)</td><td>τcmax(μs)</td><td>TMAXC (mK)</td></tr><tr><td>A</td><td>5.087</td><td>5.096</td><td>520</td><td>800</td><td>22.0</td><td>265</td><td>8</td><td>20</td></tr><tr><td>B</td><td>5.296</td><td>5.483</td><td>775</td><td>490</td><td>45.4</td><td>77</td><td>2</td><td>90</td></tr></table>

This Hamiltonian has the form of an artificial piezoelectric response with an interaction strength

$$
\hbar g _ {\mathrm {e m}} = \frac {\partial C _ {\mathrm {m}}}{\partial x} x _ {\mathrm {z p f}} V _ {\mathrm {z p f}} V _ {\mathrm {D C}}. \tag {B5}
$$

This constitutes a parametric interaction where the interaction strength scales linearly with the applied external voltage.

Upon obtaining the interaction term, we can write the full Hamiltonian of our system as

$$
H / \hbar = \omega_ {r} \hat {a} ^ {\dagger} \hat {a} + \omega_ {m} \hat {b} ^ {\dagger} \hat {b} + i g (\hat {a} \hat {b} ^ {\dagger} - \hat {a} ^ {\dagger} \hat {b}), \qquad \mathrm {(B 6)}
$$

where we have dropped the subscript from $g_{\mathrm{em}}$ for brevity. In probing our cavity-mechanics system, we use a microwave tone at frequency $\omega_{d}$ . We can write the Langevin equations for eq. (B6) as

$$
\dot {\hat {a}} = - \left(i \omega_ {r} + \kappa / 2\right) \hat {a} - g \hat {b} - \sqrt {\kappa_ {e}} \hat {a} _ {\mathrm {i n}} \qquad \mathrm {(B 7)}
$$

$$
\dot {\hat {b}} = - \left(i \omega_ {m} + \gamma / 2\right) \hat {b} + g \hat {a}. \tag {B8}
$$

where we have
neglected the thermal fluctuations entering the microwave and mechanics, as we are focused on the coherent response of our system.

Taking the Fourier transform of the Langevin equations, we obtain

$$
\left(i \Delta + \kappa / 2\right) \hat {a} (\omega) = - g \hat {b} (\omega) - \sqrt {\kappa_ {e}} \hat {a} _ {\mathrm {i n}} (\omega) \tag {B9}
$$

$$
\left(i \delta + \gamma / 2\right) \hat {b} (\omega) = g \hat {a} (\omega), \tag {B10}
$$

where $\Delta = \omega_r - \omega$ and $\delta = \omega_m - \omega$ . Substituting eq. (B10) into eq. (B9), we can express the microwave cavity operator as

$$
\hat {a} (\omega) = - \frac {\sqrt {\kappa_ {e}} \hat {a} _ {\mathrm {i n}} (\omega)}{i \Delta + \kappa / 2 + \frac {g ^ {2}}{i \delta + \gamma / 2}}. \qquad \mathrm {(B 1 1)}
$$

Using input-output theory, we can define the output operator as $\hat{a}_{\mathrm{in}}(\omega) = \hat{a}_{\mathrm{out}}(\omega) + \sqrt{\kappa_e}\hat{a} (\omega)$ . Using eq. (B11), we obtain the familiar electromechanically induced transparency (EIT) expression

$$
\frac {\hat {a} _ {\mathrm {o u t}} (\omega)}{\hat {a} _ {\mathrm {i n}} (\omega)} = 1 - \frac {\kappa_ {e}}{i \Delta + \kappa / 2 + \frac {g ^ {2}}{i \delta + \gamma / 2}}. \tag {B12}
$$

We utilize this EIT expression to fit the reflection traces we obtain of the cavity-mechanics system.

# 2. Coupling Perturbation Theory

Within the framework of cavity electromechanics, we can consider our interaction to be caused by radiation pressure. More precisely, the stored electrical energy in our system changes with the mechanical displacement via the modulation of the capacitance. Looking at the term in the Hamiltonian that leads to electrostatic interaction in eq. (B3), we can see that the change in the cross electrical energy is the origin of this interaction and thus can be used to capture the change in the capacitance.

It is possible to express this change in the energy via a perturbative integral, similar to the moving boundary integrals for electromechanical systems [46]. In this perturbative approach, we assume that the displacement of the material boundaries does not change the electric field but alters the local permittivity due to leading to a electromechanical coupling rate

$$
\begin{array}{l} \hbar g _ {e m} = x _ {\mathrm {z p f}} \oint d A \left(\mathbf {Q} (\mathbf {r}) \cdot \hat {n}\right) (\Delta \epsilon \mathbf {E} ^ {\parallel} (\mathbf {r}) _ {\mathrm {D C}} \mathbf {E} ^ {\parallel} (\mathbf {r}) _ {\mathrm {R F}} \\ \left. - \Delta \epsilon^ {- 1} \mathbf {D} ^ {\perp} (\mathbf {r}) _ {\mathrm {D C}} \mathbf {D} ^ {\perp} (\mathbf {r}) _ {\mathrm {R F}}\right). \quad \mathrm {(B 1 3)} \\ \end{array}
$$

Here, $x_{\mathrm{zpf}} = \sqrt{\hbar / 2m_{\mathrm{eff}}\omega_m}$ is the zero point fluctuations of displacement and $m_{\mathrm{eff}}$ is the effective mass of the acoustic resonator. $\mathbf{Q}(\mathbf{r})$ is the normalized displacement where $\max [\mathbf{Q}(\mathbf{r})] = 1$ . $\Delta \epsilon = \epsilon_{1} - \epsilon_{2}$ and $\Delta \epsilon^{-1} = 1 / \epsilon_1 - 1 / \epsilon_2$ are the electrical permittivity contrast between the two materials that are on the boundary covered by the surface integral.

$\mathbf{E}^{\parallel}(\mathbf{r})_{\mathrm{DC}}(\mathbf{E}^{\parallel}(\mathbf{r})_{\mathrm{RF}})$ is the parallel electric field component obtained from electrostatic simulations of the capacitor with $V_{\mathrm{DC}}$ ( $V_{\mathrm{zpf}}$ ) applied to the capacitors. Likewise, $\mathbf{D}^{\perp}(\mathbf{r})_{\mathrm{DC}}(\mathbf{D}^{\perp}(\mathbf{r})_{\mathrm{RF}})$ is the perpendicular displacement field obtained from the same simulation. In this expression the voltage dependence of the coupling is directly embedded in the capacitor voltages used in the simulations.

One can alternatively solve for a given voltage (such as 1V) and then scale the fields appropriately based on $V_{\mathrm{DC}}$ and $V_{\mathrm{zpf}}$ . For instance, $g_{0}$ can be simply obtained by setting $V_{\mathrm{DC}} = 1\mathrm{V}$ .

It has been previously observed that the electrical response of the capacitor at DC and RF frequencies may differ from one another [19]. Generally, the charge carriers freeze off at cryogenic temperatures, giving rise to massive resistivity values for silicon [29]. However, in capacitor structures under DC voltages, band bending leads to the formation of a narrow space charge region which effectively screens out the field at the bulk of the

9

silicon [47]. This leads to a DC response which can approximately be modelled by modelling silicon as a perfect conductor. On the other hand, our microwave field which oscillates at a frequency above the RC cutoff cannot be screened and silicon behaves like a perfect insulator for these fields.

Taking this into account, only fields perpendicular to the boundaries will exist for the DC field and the integral for the electromechanical coupling can be simplified to.

$$
\hbar g _ {e m} \approx x _ {\mathrm {z p f}} \oint d A \left(\mathbf {Q} (\mathbf {r}) \cdot \hat {n}\right) \left(\epsilon_ {0} ^ {- 1} \mathbf {D} ^ {\perp} (\mathbf {r}) _ {\mathrm {D C}} \mathbf {D} ^ {\perp} (\mathbf {r}) _ {\mathrm {R F}}\right). \tag {B14}
$$

For our devices with $g_0 / 2\pi = 45.4 \mathrm{kHz} / \mathrm{V}$ this approach gives us a very accurate simulation result of $46 \mathrm{kHz} / \mathrm{V}$ . If we were to assume that silicon was an insulator at all frequencies and the field distributions were identical, we would underestimate our coupling strength and obtain $32 \mathrm{kHz} / \mathrm{V}$ .

# 3. Equivalent Circuit Model

![](dt=2026-06-05/ht=21/9d1e4b3203ea1eef936457645e00eb9c4e666c1e5d69d594fe9d3d53b4dd3f07.jpg)

The calculation of the electromechanical coupling strength via finite element method simulations is sufficient to completely describe the behavior of our system which consists of two coupled resonators. However, obtaining an electrical equivalent circuit for the mechanical resonator is crucial for making the analysis of complex electromechanical circuits more tractable. For this purpose, we model our mechanical resonator by a parallel LC resonator $(C_{\mathrm{k}},L_{\mathrm{k}})$ in series with a capacitor $C_\mathrm{m}$ . This model can be seen in fig. 7, where the coupling to the lumped element microwave resonator $(C_{\mathrm{r}},L_{\mathrm{r}})$ is capacitive via $C_\mathrm{m}$ .

The equivalent mechanical capacitance can be expressed as

$$
C _ {\mathrm {k}} = \frac {2 C _ {\mathrm {m}} ^ {2} \omega_ {m}}{\hbar (\partial_ {x} C _ {\mathrm {m}} x _ {\mathrm {z p f}} V _ {\mathrm {D C}}) ^ {2}} - C _ {\mathrm {m}}. \qquad \mathrm {(B 1 5)}
$$

The equivalent mechanical inductance can be obtained following the calculation of $C_{\mathrm{k}}$ by noting that $\omega_{m} =$

$[L_{\mathrm{k}}(C_{\mathrm{k}} + C_{\mathrm{m}})]^{-1 / 2}$ . This circuit model gives us the correct expression for the electromechanical coupling strength, which is attained by capacitive coupling between the two circuit modes where

$$
g _ {\mathrm {e m}} = \frac {C _ {\mathrm {m}}}{\sqrt {\left(C _ {\mathrm {k}} + C _ {\mathrm {m}}\right) \left(C _ {\mathrm {r}} + C _ {\mathrm {m}}\right)}} \sqrt {\omega_ {r} \omega_ {m}}. \tag {B16}
$$

In this capacitive coupling picture, the value of $C_{\mathrm{r}}$ primarily sets the zero point fluctuations of voltage for the microwave resonator since we have a small electromechanical participation ratio $(\eta = C_{\mathrm{m}} / (C_{\mathrm{m}} + C_{\mathrm{r}}) \ll 1)$ . The small participation ratio of our capacitor is caused by the small physical dimensions of our electromechanical capacitor which is commensurate with $1\mu \mathrm{m}$ transverse acoustic wavelength at $5\mathrm{GHz}$ . This leads to the electrical energy on top of the electromechanical capacitor to be substantially diluted compared to the total electrical energy
stored on the microwave resonator.

We express the circuit parameters for device B in Table II. We note that the equivalent circuit model parameters for the mechanical resonator are dependent on the external voltage bias $V_{\mathrm{DC}}$ . This scaling is noted on the table, where $L_{\mathrm{k}}$ and $C_{\mathrm{k}}$ are provided for 1V.

TABLE II. Equivalent circuit parameters for device B. The mechanical parameters $C_{\mathrm{k}}$ and $L_{\mathrm{k}}$ are dependent on the applied external voltage. The given values are for $1\mathrm{V}$ and the dependence on $V_{\mathrm{DC}}$ is specified.

![](dt=2026-06-05/ht=21/2f28b162a9b3069dac70c10cc4c32a73c23d4f1b81d2eb57abe9ca24d1595b6b.jpg)

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Cm</td><td>0.2 fF</td></tr><tr><td>Cr</td><td>12.1 fF</td></tr><tr><td>Lr</td><td>75.8 nH</td></tr><tr><td>η</td><td>1.5 %</td></tr><tr><td>Ck</td><td>43.5 nH (1V/VDC)2</td></tr><tr><td>Lk</td><td>21.1 fF (VDC/1V)2</td></tr><tr><td>fr,m</td><td>5.26 GHz</td></tr><tr><td>g0/2π</td><td>45.4 kHz/V</td></tr></table>

In the circuit model, one can see that the mechanical resonator is capacitively coupled to the CPW which has an impedance $Z_{0} = 50\Omega$ . This leads to some direct external decay to the CPW. However, even at the maximum voltage we can apply of $25\mathrm{V}$ , due to our massive equivalent capacitance this readout is at the level of $5\mathrm{Hz}$ which is negligible and further emphasizes the need for a microwave cavity in order to enhance electromechanical readout of mechanics.

# Appendix C: Microwave Devices

# 1. Microwave Resonator

The $\lambda /4$ tunable microwave resonators in our work are formed by ICP-RIE etching of TiN films ( $t\approx 15~\mathrm{nm}$ )

10

![](dt=2026-06-05/ht=21/afee6305c5f5238fd9480f732652f3053c53bb261aa088055e10f03ab4842a2b.jpg)

with a sheet kinetic inductance of $40\mathrm{pH} / \square$ , which are sputtered on high resistivity ( $>3\mathrm{k}\Omega$ ) SOI substrates (device layer thickness $220\mathrm{nm}$ ). The high kinetic inductance TiN films are chosen for two main reasons: (i) obtaining a large impedance ( $Z = \sqrt{L / C}$ ) resonator in order to enhance the electromechanical interaction ( $g_{\mathrm{em}} \propto \sqrt{Z}$ ) and (ii) attaining a high degree of tunability via an external magnetic field to bring the microwave into resonance with mechanics. These goals are satisfied by forming a $\lambda / 4$ resonator, with the inductive component realized by a nanowire. The total kinetic in this structure is a function of film properties and geometry

$$
L _ {\mathrm {k}} = L _ {\square} \frac {8}{\pi^ {2}} \frac {l}{w}, \tag {C1}
$$

where $L_{\square}$ is the sheet inductance, $l$ is the nanowire length and $w$ is the nanowire width. In order to maximize the impedance, we etch our nanowires to be as narrow as possible, which in our case corresponds to a width of approximately $110~\mathrm{nm}$ . Attempts to reduce wire width below this number led to reduced repeatability and a large disorder in resonator frequency.

A current passing through a TiN nanowire modifies the kinetic inductance in a nonlinear fashion

$$
L _ {\mathrm {k}} (I) = L _ {\mathrm {k}} (0) \left[ 1 + \left(\frac {I}{I _ {*}}\right) ^ {2} \right]. \tag {C2}
$$

where $I_{*}$ is the critical current of the nanowire. Patterning of the nanowires to form closed loops and application of an external perpendicular magnetic field, provides a 'wireless' means of modifying the kinetic inductance via the screening current induced through the loops [27]. This is the mechanism we us to tune the resonator's frequency.

With these principles in mind, we utilize a ladder-like topology for our microwave resonator as seen in fig. 1. Finite-element method (FEM) simulations indicate that our kinetic inductance excellently matches eq. (C1) and

we have negligible geometric inductance, leading to a kinetic inductance participation ratio of approximately unity $(>98\%)$ . Furthermore, we calculate the impedance of our resonator to be $2.5\mathrm{k}\Omega$ . The calculated lumped element equivalent circuit parameters for the microwave resonator of device B can be seen in Table II. Despite having a substantial impedance, due to our mechanical capacitance being very small, we have a low electrical energy participation ratio for the mechanical capacitor $(\eta = C_{\mathrm{m}} / (C_{\mathrm{m}} + C_{\mathrm{r}}))$ of $1.5\%$ . This small participation ratio permits us to galvanically connect multiple electromechanical capacitors to our microwave resonator.

Following the fabrication of our devices, we find the measured resonance frequencies in good agreement with the device modeling, to within a random offset of approximately $300\mathrm{MHz}$ , which is attributed to fabrication disorder. We test the tunability of our devices by applying external magnetic fields via currents passing through a superconducting coil mounted on top of the sample box. For small tuning compared to the frequency, we expect the frequency shift to be given by

$$
\Delta_ {f} = - k B ^ {2}, \tag {C3}
$$

where $k$ is a device dependent proportionality constant and $B$ is the external perpendicular magnetic field amplitude [27]. This parabolic dependence is verified for device B on fig. 8. We can tune our device by about $6\%$ of their resonance frequency without causing any substantial degradation of the microwave intrinsic quality factor beyond $\kappa_{i} / 2\pi \sim 500~\mathrm{kHz}$ .

# 2. Purcell Filter

In our design, we prioritize redundancy in terms of the number of mechanical resonators by attaching four electromechanical capacitors to a single microwave resonator. This is done to mitigate the impact of frequency shifts caused by fabrication disorder of the microwave resonators, where the different mechanical resonators spans roughly $500\mathrm{MHz}$ (the disorder in the mechanical frequencies is much smaller than the microwave disorder). However, this approach leads to an increased capacitance between our microwave resonator and the CPW, which gives rise to undesirably large external microwave decay. In order to increase the cooperativity of our system and reach the strong-coupling regime, this decay channel has to be suppressed.

We utilize an off-chip Purcell filter for this purpose as depicted schematically in fig. 9. We place the filter in parallel to our device, where the notch filter reduces the external coupling strength of the microwave resonator. For this purpose, we make use of home-built microwave filters, realized as open transmission lines with a multipole spectrum. To get approximate frequency matching between our devices and the filter, we use a mechanical switch that enables us to choose among multiple filters with different free spectral ranges (in a span of $200\mathrm{MHz}$

11

![](dt=2026-06-05/ht=21/88f2dceb12f5af8a88d3b3308f59f15a80ebb6d0c6c22a6fe52fae528eb9c58e.jpg)

to $500\mathrm{MHz}$ ). In cases where we get perfect frequency matching between a device and a filter resonance, we can reduce the external decay rate $\kappa_{e} / 2\pi$ to sub- $200\mathrm{kHz}$ levels. Furthermore, we have a line in our switch which is not connected to a transmission line, permitting us to effectively turn our filter off.

# Appendix D: Mechanics Design

The mechanical behavior of our structures is investigated via FEM simulations on COMSOL. The inner electrode of the electrostatic transducer is patterned to form a phononic crystal cavity consisting of multiple identical unit cells, with the unit cell depicted in fig. 10a. These unit cells are completely metallized with a $15\mathrm{-nm}$ layer of TiN on top of a $220\mathrm{-nm}$ Si membrane (the anisotropic elasticity
tensor is used for silicon as in [48]). We are primarily interested in the $\Gamma$ point of the breathing mode shown by the red band on the unit cell band diagram on fig. 10b.

The breathing mode is selected because it has a displacement profile that significantly modulates the electromechanical capacitance. The $\Gamma$ mode is further necessary to ensure phase matching of the electromechanical interaction. Once multiple unit cells are attached to form a 1-D chain and are clamped by phononic shields that have a full bandgap, they hybridize to form a supermode that spans all the unit cells, which is our mechanical mode of interest.

Ideally, we would like to have as many unit cells as possible in forming our central electrode, since the electromechanical interaction strength scales as $\propto \sqrt{N_{\mathrm{cells}}}$ . In practice, however, fabrication disorder leads to mode break-up, limiting the number of cells that can be uti

ized. We investigate this phenomenon via disorder simulations, which constitutes a crucial tool in investigating the disorder limited properties of periodically patterned acoustic and optical structures [49-51]. The disorder can be modeled as random changes in the center positions and the width of the negative patterns, which are etched to form the actual structure. These random changes are represented as realizations of independent Gaussian random variables with zero mean and standard deviation $\sigma$ .

The precise algorithm is as follows: (i) the center of the etched filleted rectangles are varied by $\sigma$ in each direction (ii) the height and width of the rectangles are varied by $2\sigma$ . Since each simulation represents a given disorder realization, to accurately extract the statistics we simulate multiple instances for a given number of unit cells. We investigate the impact of disorder on the coupling strength as shown in fig. 10c.

In this situation, the changes in the mode profile and potential disorder induced mode breakup is captured in variations in the coupling strength, which is the experimentally relevant parameter to us. We can see that up until 9 unit cells, $g_{0}$ grows with the number of cells. However, beyond this number the growth rate diminishes and the disorder-induced variations increase substantially. At 9 unit cells and a $70\mathrm{-nm}$ gap, we have a coupling strength of $46_{-7}^{+4}\mathrm{kHz / V}$ which is in excellent agreement with our experimental results.

Finally, we investigate the dependence of the coupling strength on the vacuum gap as shown in fig. 10d. The fit indicates that the coupling strength is a very strong function of the gap and the scaling is more rapid than that of a parallel plate capacitor.

Since acoustic waves cannot propagate in vacuum, the only radiative leakage pathway is through the silicon substrate. We minimize this loss by clamping our acoustic resonator with a phononic shield that has a complete bandgap. We investigate the effectiveness of the phononic shields in suppressing radiation loss via finite-element simulations, where the continuum of modes in the substrate is modeled by a perfectly matched layer. The simulations provide us with the radiation limited mechanical $Q$ in the absence of fabrication disorder. As depicted in fig. 11a the mechanical quality factor roughly nearly exponentially with the number of phononic shields. We have used 5 shield periods in our final design, for which we plot the acoustic energy distribution in fig. 11b.

# Appendix E: Thermometry

# 1. Theory

We perform sideband thermometry by measuring the emission from the microwave and mechanical resonators with a spectrum analyzer. Our Hamiltonian is identical to that of a red-detuned optomechanical system with a large sideband resolution $(\omega_{m} / \kappa \gg 1)$ . Hence, in order to

12

![](dt=2026-06-05/ht=21/d65af0958b83008bf65b9227dedf05eaf4d871090720cf13354febfae97402d3.jpg)

![](dt=2026-06-05/ht=21/06a8819ff42085500aa9e1df77fff65c2ae58a7aff81b5ab841ae40b9eb76524.jpg)

![](dt=2026-06-05/ht=21/34bc85987c315bd6f8e312acc97ddc217b626bd7cb3ee4c08725bcf7db7dbd78.jpg)

![](dt=2026-06-05/ht=21/cc56a6ecfb3333c77682de4eddd971264ed532a648ffc1275a9954e990dbdd58.jpg)

![](dt=2026-06-05/ht=21/d94f473c10df99cb080d4ffb085eb8c6f286657c1005c4aada11fc7feb0c5833.jpg)

![](dt=2026-06-05/ht=21/088174f85c419ff48e0864de208a6531ebc3d36a3fbb6baa3925d5cba51a7ad7.jpg)

extract the thermal occupancy values for the microwave and mechanical thermal baths, we use the expression for the noise power spectral density following previous electromechanics work [15, 52]. The power spectral density following the amplifier chain is

$$
S (\omega) = \hbar \omega 1 0 ^ {\mathcal {G} / 1 0} \left(n _ {\mathrm {a d d}} + \frac {1}{2} + \right.
$$

$$
\begin{array}{l} n _ {\mathrm {w g}} \left| \left(1 - \frac {\kappa_ {e} \chi_ {r}}{1 + g ^ {2} \chi_ {m} \chi_ {r}}\right) \right| ^ {2} + n _ {\mathrm {b , r}} \frac {\kappa_ {e} \kappa_ {i} | \chi_ {r} | ^ {2}}{| 1 + g ^ {2} \chi_ {m} \chi_ {r} | ^ {2}} \\ \left. + n _ {\mathrm {b , m}} \frac {\kappa_ {e} \gamma_ {i} g ^ {2} | \chi_ {r} | ^ {2} | \chi_ {m} | ^ {2}}{| 1 + g ^ {2} \chi_ {m} \chi_ {r} | ^ {2}}\right). \tag {E1} \\ \end{array}
$$

Here, $\mathcal{G}$ is the amplifier gain in dB and $n_{\mathrm{add}}$ is the noise added by our amplifiers, which is dominated by the HEMT in our case. In this picture, the microwave resonator interacts with an intrinsic bath of occupancy $n_{\mathrm{b,r}}$ with rate $\kappa_{i}$ and the waveguide having an occupancy of $n_{\mathrm{wg}}$ with rate $\kappa_{e}$ . Similarly, the mechanical resonator is coupled to an intrinsic bath having an occupancy of $n_{\mathrm{b,m}}$ with rate $\gamma_{i}$ . The electromechanical interaction with strength $g$ will further lead to Purcell decay into the mi

crowave for the mechanics, giving rise to electromechanical back-action. The PSD expression also includes the bare electrical and mechanical susceptibilities which are given as

$$
\chi_ {r} ^ {- 1} (\omega) = \kappa / 2 - i \left(\omega - \omega_ {r}\right) \tag {E2}
$$

$$
\chi_ {m} ^ {- 1} (\omega) = \gamma_ {i} / 2 - i (\omega - \omega_ {m}), \tag {E3}
$$

where $\omega_{r}\left(\omega_{m}\right)$ is the microwave (mechanics) resonance frequency. In the weak coupling regime, we can express the mechanics and microwave resonator thermal occupancies as

$$
n _ {r} = \frac {\kappa_ {i} n _ {\mathrm {b , r}} + \kappa_ {e} n _ {\mathrm {w g}}}{\kappa_ {e} + \kappa_ {i}}, \tag {E4}
$$

$$
n _ {m} = \frac {n _ {\mathrm {b , m}} + C n _ {r}}{1 + C}. \tag {E5}
$$

Here, $C = \frac{g^2\kappa\gamma_i^{-1}}{\Delta^2 + (\kappa / 2)^2}$ is the effective cooperativity when the mechanical and microwave resonators are detuned by $\Delta$ . In the absence of any detuning and at large bias voltages, the cooperativity becomes large, leading to substantial electromechanical cooling, which makes it challenging to unambiguously find the mechanical bath occupancy $(n_{\mathrm{b,m}})$ . Therefore, we operate in a low-cooperativity regime (large $\Delta$ ), which permits precise extraction of thermal bath occupancies.

Extraction of the mechanical intrinsic bath occupancy is particularly important for quantum memory applications of our mechanical resonators, as there won't be permanent electromechanical back-action cooling in this scenario and the thermal decoherence rate depends on this bath occupancy.

To facilitate the analysis, we simplify the emission due to
the mechanics intrinsic bath in eq. (E1) as

$$
S _ {m} (\delta) \sim 4 \tilde {n} _ {\mathrm {m}} \frac {\kappa_ {e}}{\kappa} \frac {\Gamma_ {\mathrm {e m}}}{\gamma} \frac {(\gamma / 2) ^ {2}}{\delta^ {2} + (\gamma / 2) ^ {2}}. \tag {E6}
$$

Here, $\delta = \omega -\tilde{\omega}_m$ is the detuning between the emission frequency and mechanical resonance frequency, with

13

$\tilde{\omega}_{m}$ being the mechanical frequency shifted by the optical spring effect. $\gamma = \gamma_{i} + \Gamma_{\mathrm{em}}$ is the total mechanical linewidth and $\tilde{n}_{\mathrm{m}} = n_{\mathrm{b,m}}\gamma_{i} / \gamma$ is the thermal occupancy of the mechanical resonator due to fluctuations of the intrinsic mechanical bath.

To accurately utilize the expressions for the noise power spectral density, it is crucial to take into account the distinction between broadening due to frequency jitter and radiative coupling to an intrinsic bath for the mechanical resonator. To this end, we note that the area underneath $S_{m}(\delta)$ is proportional to the total Purcell enhanced emission from mechanics, which is given as $\Gamma_{\mathrm{em}}\tilde{n}_{\mathrm{m}}$ . Hence, we can see that eq. (E6) represents the emission for a mechanical resonator that has a total linewidth $\gamma$ with arbitrary frequency jitter.

To take the distinction between jitter and decay into account, once we have extracted $\tilde{n}_{\mathrm{m}}$ , we should use only the mechanical intrinsic decay rate $\Gamma_{i}$ to calculate the bath thermal occupancy as

$$
n _ {\mathrm {b , m}} = \tilde {n} _ {\mathrm {m}} \frac {\Gamma_ {\mathrm {e m}} + \Gamma_ {i}}{\Gamma_ {i}}. \tag {E7}
$$

We note that due to the small non-zero microwave bath occupancy, we have some emission from the mechanical resonator due to microwave thermal fluctuations entering the mechanics by electromechanical back-action. This emission interferes with that of the microwave resonator and leads to noise squashing as can be seen in the term in eq. (E1) proportional to $n_{\mathrm{b,r}}$ [53]. This noise squashing is also taken into account in our analysis in order to correctly calculate our mechanical bath occupancy.

# 2. Driven Response

We generally do not see significant emission from the mechanical resonator during thermometry, due to the mechanics being deep in its motional ground state. Due to the noise introduced by our HEMT, the spectrum analyzer traces are noisy to an extent that precludes numerically fitting the mechanical emission. Thus, in order to accurately analyze this noisy data and extract the mechanical thermal occupancy, it is crucial to find the mechanical frequency, linewidth and decay rate at a given voltage.

We achieve this by investigating the driven response of our resonators. Following the application of a coherent tone in resonance with our mechanics, the drive tone is elastically scattered, leading to a delta-like emission from the mechanics and the generation of a coherent phonon population defined as $n_{\mathrm{coh}} = |\langle \hat{a}\rangle |^2$ . This coherent population dynamics is the origin of the EIT response which can be detected by a VNA. However, the frequency jitter of our system also leads to inelastic scattering of the drive tone. For frequency noise which has a correlation time smaller than the decay rate, we get an emission with the cavity lineshape as the absorbed incoherent phonons

lose memory of the drive frequency due to frequency jitter [30]. This inelastic scattering is due to the generation of a number of incoherent phonons in the cavity, whose population is $n_{\mathrm{inc}} = \langle \hat{a}^{\dagger}\hat{a}\rangle -|\langle \hat{a}\rangle |^{2}$ . The incoherent emission enables us to extract the cavity linewidth and frequency via fitting the Lorentzian response. This routine for parameter extraction is repeated prior to mechanics thermometry for each voltage to facilitate accurate calculations with eq. (E6) and center the spectrum analyzer detection window.

Apart from extracting the lineshape of the mechanics, we can also use the driven response to make non time-resolved measurement of the intrinsic decay rate $\Gamma_{i}$ . The total coherent and incoherent phonon populations are related to the total decay rate $\Gamma_{\mathrm{d}} = \Gamma_{i} + \Gamma_{\mathrm{em}}$ and the total linewidth $\gamma$ as follows [54]

$$
\frac {n _ {\text {i n c}}}{n _ {\text {c o h}}} = \frac {\gamma}{\Gamma_ {\mathrm {d}}} - 1. \tag {E8}
$$

These phonon populations can be further related to the areas detected via the spectrum analyzer, where $S_{\delta}$ is the area underneath the coherent emission, $S_{\mathrm{bb}}$ ( $S_{\mathrm{nb}}$ ) is the area underneath the incoherent emission due to broadband (narrow-band) frequency noise. Therefore, the relation between the decay rates can be expressed as

$$
\frac {\gamma}{\Gamma_ {\mathrm {d}}} = 1 + \frac {S _ {\mathrm {b b}}}{S _ {\delta}} \left(1 - \frac {S _ {\mathrm {n b}}}{S _ {\delta}}\right), \tag {E9}
$$

where the broadband frequency noise can be arbitrarily strong and the narrow-band noise is weak [30]. Subtracting the electromechanical readout rate from the total decay rate, we can calculate $\Gamma_{i}$ . The $\Gamma_{i}$ obtained in this manner is used to extract the mechanical bath occupancy via eq. (E7).

# 3. Line Calibration

We calibrate the total gain of the output line using thermometry of a $50\Omega$ cryogenic terminator that is thermalized to the mixing (MXC) stage of the cryostat. The output line consists of a HEMT amplifier (LNF-LNC48C) thermalized to the $4\mathrm{K}$ stage and a room temperature amplifier. The MXC stage temperature is raised by reducing cooling power by turning off the turbo to reduce ${}^{3}\mathrm{He}/{}^{4}\mathrm{He}$ mixture flow, and applying heat using the MXC stage heater.

With no external input power, we measure the output power from the amplifier chain with a spectrum analyzer at different mixing stage temperatures $T_{\mathrm{MXC}}$ . The measured output power has contributions from the thermal noise of the resistor thermalized to the MXC stage, and the HEMT noise characterized by a fixed noise temperature $T_{\mathrm{HEMT}}$ . The total power measured in an IF bandwidth $\Delta \nu_{\mathrm{IF}}$ on the spectrum analyzer is equal to the sum of the Johnson-Nyquist noise from the two sources and is given by,

$$
P _ {\text {O U T}} = \Delta \nu_ {\mathrm {I F}} k _ {\mathrm {B}} G _ {\mathrm {A}} \left(T _ {\mathrm {M X C}} + T _ {\mathrm {H E M T}}\right) \tag {E10}
$$

14

Here, $G_{A}$ is the absolute (net) gain factor of the output line, and is a combination of the total gain due to the amplifier chain and losses due to coaxial cables. At $T_{\mathrm{MXC}} = 10 \mathrm{mK}$ , the measured output power is dominated by HEMT noise. The output gain $G_{A}$ can be calculated by subtracting the contribution of the HEMT noise from the total output power measured at various $T_{\mathrm{MXC}}$ . We perform this measurement at multiple different MXC temperatures between 730 mK and 1.05 K. The mean and standard deviation of these measurements are used to obtain $G_{A}$ .

We use a factor $\eta = (h\nu / kT) / (\exp(h\nu / kT) - 1)$ to account for corrections to eq. (E10) due to the Bose-Einstein distribution in the regime $h\nu \sim k_B T$ [55]. Using this calibration method, we obtain a net gain of 65.6 ± 0.4 dB for the output line.

# 4. Pull-in Instability

Post measurement imaging of the devices which were subject to breakdown indicates that the breakdown behavior is caused by pull-in of the inner electrode to an outer electrode, as seen in fig. 12. Once the capacitor gap becomes shut, a short resistive leakage path appears, leading to substantial current flow as observed in our leakage current measurements. This 'pull-in' phenomenon is commonly observed in electrostatic actuators. Increasing the bias voltage, the strong electrostatic forces can no longer be offset by the mechanic
al spring force following a certain gap shrinkage, leading to unstable mechanical dynamics [56]. Furthermore, stiction can render this phenomenon irreversible as we have observed, where removal of the external voltage does not lead to recovery of the device.

![](dt=2026-06-05/ht=21/5c6b910c3673f7f2fb1ca82be8db85bfb44f44ba9c95e163fe794d706c5e2544.jpg)

The observed pull-in behavior in our devices is not fully explained with common behavior of larger electrostatic actuators. For these MEMS devices, the capacitor gap start to shrink gradually and the onset of instability oc

curs once the gap has shrunk by about a third of its initial value for parallel plate geometries [57]. However, we observe excellent linearity of $g$ vs $V_{\mathrm{DC}}$ , which indicates the absence of any significant continuous shrinkage of the gap. At the moment, the precise mechanism of the observed pull-in instability is not clear to us and will be the subject of further studies.

# Appendix F: Mechanical Coherence

# 1. Ringdown Measurements

In performing the ringdown measurements, we use a constant DC drive at a selected voltage level and control the external readout rate via setting the detuning between the microwave resonator and the mechanics. We populate the mechanical cavity via sending microwave pulses resonant with the mechanical mode. The pulses are synthesized with an arbitrary waveform generator (AWG), and their length is chosen to be sufficiently long to ensure the mechanical population can reach the steady state.

Following the drive pulse, we detect the emitted power in a given interval by processing the downconverted signal from a digitizer. A power measurement (as opposed to field-quadrature) is obtained by summing the square of the demodulated I and Q quadrature values in a detection window in an FPGA. This detection window length is set to be much shorter than the reciprocal of mechanical linewidth to ensure we detect all the phonons emitted from the resonator. The multiple consecutive detection windows during a single measurement gives us a ringdown curve.

We then proceed to average many instances of the experiment to improve our signal-to-noise (SNR) ratio and obtain our final data. The AWG, digitizer, and FPGA functionalities are realized using a Quantum Machines OPX+ module. We calibrate the digitizer output using the calibrated gain of our output lines and refer the detected voltage levels to the number of phonons in the cavity.

In finding the optimal lifetime in a voltage range, we carry out multiple ringdowns while sweeping the voltage. These ringdowns are performed at a large number of phonons in order to improve the SNR and make the measurements more tractable. Apart from showing signatures of spectral collisions with TLS that is manifested as deteriorating lifetimes, these measurements enable us to extract statistics about our lifetimes. We can see that for the two devices we have measured, we can reliably obtain lifetimes around $30~\mu \mathrm{s}$ by slightly optimizing our voltage level.

Such a typical ringdown is visualized in fig. 13a. We can further see that the points where we have improved coherence properties do not exhibit extreme sensitivity to the applied voltage. For example, for the voltage value where we have obtained our best lifetime on device A, we can obtain similar lifetimes in a $250~\mathrm{mV}$ range as shown in fig. 13b. We can see similar broad regions where the lifetime is enhanced for device

15

![](dt=2026-06-05/ht=21/795af2277bdf6edd583d13dacddf28e2fe469f0e21a751511affc7f6218fc8d1.jpg)

![](dt=2026-06-05/ht=21/b4cb04c06d5d92df3fba213588749eb7c6b48658b8f88db8d955117eb1397810.jpg)

![](dt=2026-06-05/ht=21/b1addcc0e357a1fd3b304c046bae0b9c3a0ffd6d0b55e5e188e14b77ab36fedb.jpg)

B too, as depicted in fig. 13c.

# 2. Linewidth Characterization

The reflection spectrum measurements enable us to investigate the mechanical linewidth at different power levels in order to investigate their power dependence. For device A, these results corroborate our previous observations concerning the absence of saturable TLS dependent losses at our long lifetime point. As shown in fig. 14a, the linewidth does not show any saturation behavior when the phonon number varies between 4-500.

During our ringdown measurements, we keep track of our mechanical resonance frequency in long timescales and observe telegraphic frequency jumps as shown in fig. 14b. A potential model for this behavior could be that of a mechanical resonator directly coupled to a low frequency TLS that has non-negligible thermal population, which is sometimes referred to as a thermal fluctuator (TF) [58]. The TF jumping to its excited state leads to a dispersive shift for our device, giving rise to a central mechanical frequencies which are bunched at a higher frequency. Using this data, once can infer the frequency of the TF, its switching rate and the coupling strength of it to the mechanics, which is consistent with our first principle calculations and previous literature [58-60].

In contrast to the previously discussed non-saturable behavior, when we go to a operation point in device B with signatures of TLS spectral collision, we can recover the familiar saturable behavior of the linewidth for as seen in fig. 14c. We can fit this data to a TLS continuum model, where the total linewidth $\gamma$ can be expressed as [61]

$$
\gamma = \frac {F \gamma_ {\mathrm {T L S}}}{\tanh \left(\frac {\hbar \omega}{2 k _ {B} T}\right)} \sqrt {1 + \left(\frac {n}{n _ {c}}\right) ^ {\beta}} + \gamma_ {0}, \tag {F1}
$$

where $n$ is the number of phonons inside the cavity, $F$ is the TLS participation ratio, $\gamma_{\mathrm{TLS}}$ is the TLS decay rate,

$n_{c}$ is the critical phonon number, $\beta$ is a fit parameter and $\gamma_0$ is the decay rate from power independent broadening mechanisms. Due to ambiguity in the phonon number inside the cavity for VNA measurements, we use the coherent phonon number $n_{\mathrm{coh}}$ in our fit (see appendix E for this distinction). We cannot see the complete saturation of the linewidth at large phonon numbers due to our mechanical nonlinearity which leads to narrowing down of the linewidth and deviation from Lorentzian lineshape. The fit gives us $\gamma_0 / 2\pi \approx 30~\mathrm{kHz}$

# 3. TLS Model

In analyzing the interactions of TLS with acoustic and electrical fields, we use the standard tunnelling model [34]. The TLS is modeled as two potential wells having an asymmetry energy $\epsilon$ and a tunnelling energy $\Delta$ , which leads to a TLS energy of $E = \sqrt{\Delta^2 + \epsilon^2}$ . The interaction of TLS with external fields is via modification of its asymmetry energy

$$
\epsilon = 2 \gamma \cdot \mathbf {S} + 2 \mathbf {p} \cdot \mathbf {E} + \epsilon_ {0}, \tag {F2}
$$

where $\gamma$ is the mechanical deformation potential around $1.5\mathrm{eV}$ in magnitude, $\mathbf{S}$ is the external strain field, $\mathbf{p}$ is the TLS dipole moment of roughly 1 Debye, $\mathbf{E}$ is the external electric field and $\epsilon_0$ is the residual asymmetry from the environment.

The dependence of the asymmetry energy on the electric field enables us to Stark shift the frequency of TLS via the voltage applied on our electromechanical capacitor. The tuning rate can be calculated as

$$
\delta E = 2 \frac {\epsilon}{E} \mathbf {p} \cdot \mathbf {E} \tag {F3}
$$

We use $\epsilon / E$ of 0.5 in our analysis. The narrow vacuum gap capacitors gives rise to large electric fields approaching $5 \times 10^{6} \mathrm{~V/m}$ , leading to a steep tuning rate of $\delta E / h \approx 25 \mathrm{GHz} / \mathrm{V}$ .

16

!
[](s3://lakehouse2/hive-ha/produce.db/mineru_full_text/v0/result=success/type=image/dt=2026-06-05/ht=21//673937a16dcf292ced84869eaad14066ae7c8ad1e5041f340c7cede70007cb4a.jpg)

![](dt=2026-06-05/ht=21/9eda91007c2751ca337fc53b2f9ace83c2b06209b8df8190a23b8775cc10895b.jpg)

![](dt=2026-06-05/ht=21/af22b6ec7ec01909b7cfe6b2fc4aa41daa81c70956e0ed1db112e4b1087baf8c.jpg)

Apart from Stark shifts, the electrical dipole of the TLS also leads to $\mathbf{p} \cdot \mathbf{E}$ coupling to the microwave fields. The zero-point fluctuations of voltage in our microwave resonator is approximately $10\mu \mathrm{V}$ , with maximum electric field values of roughly $50\mathrm{V / m}$ on the interfaces, leading to a TLS-microwave coupling of $250\mathrm{kHz}$ .

Due to substantial acoustic susceptibility of TLS, strain coupling constitutes an important mechanism for mechanics-TLS interactions. The coupling strength can be written as

$$
\hbar \lambda = \gamma \frac {\epsilon}{E} S _ {\mathrm {z p f}} \tag {F4}
$$

where $S_{\mathrm{zpf}}$ is the zero-point fluctuations of strain associ-

ated with our mechanical mode [35]. This quantity is related to the strain mode volume of our mechanical mode

$$
S _ {\mathrm {z p f}} = \left(\frac {\hbar \omega_ {m}}{2 \mathcal {E} V _ {m}}\right) ^ {1 / 2} \tag {F5}
$$

where $\mathcal{E}$ is the Young's modulus and $V_{m}$ is the strain mode volume. Due to the small physical dimensions of our mechanical resonator, we have an extremely small strain mode volume of $6\times 10^{-3}\mu \mathrm{m}^3$ and a corresponding $S_{\mathrm{zpf}}$ of $4\times 10^{-8}\mathrm{m / m}$ . This causes a strain coupling strength of $\lambda /2\pi = 13\mathrm{MHz}$ , which clearly dominates other coupling mechanisms to TLS for the mechanical resonator.

17

18

19