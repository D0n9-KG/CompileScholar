# QUANTUM SYSTEMS

# Direct observation of deterministic macroscopic entanglement

Shlomi Kotler $^{1,2*}$ , Gabriel A. Peterson $^{1,2}$ , Ezad Shojaee $^{1,2}$ , Florent Lecocq $^{1,2}$ , Katarina Cicak $^{1}$ , Alex Kwiatkowski $^{1,2}$ , Shawn Geller $^{1,2}$ , Scott Glancy $^{1}$ , Emanuel Knill $^{1,3}$ , Raymond W. Simmonds $^{1}$ , José Aumentado $^{1}$ , John D. Teufel $^{1}$

Quantum entanglement of mechanical systems emerges when distinct objects move with such a high degree of correlation that they can no longer be described separately. Although quantum mechanics presumably applies to objects of all sizes, directly observing entanglement becomes challenging as masses increase, requiring measurement and control with a vanishingly small error. Here, using pulsed electromechanics, we deterministically entangle two mechanical drumheads with masses of 70 picograms.

Through nearly quantum-limited measurements of the position and momentum quadratures of both drums, we perform quantum state tomography and thereby directly observe entanglement. Such entangled macroscopic systems are poised to serve in fundamental tests of quantum mechanics, enable sensing beyond the standard quantum limit, and function as long-lived nodes of future quantum networks.

The idea that motion has a nonclassical nature dates back to the early days of quantum mechanics. One of the first triumphs of the theory was explaining the emission and absorption spectra of atoms by quantizing the motion of their electrons. Quantum mechanics is not limited to the atomic scale; in principle, it extends to all objects of all sizes. We expect that quantum behavior of macroscopic systems will enhance our ability to build more powerful sensing, communication, processing, and storage devices (I).

Many future applications of quantum technology rely heavily on entanglement; that is, on the ability to generate strong quantum correlations between separate objects. For entanglement to be useful, it must be prepared efficiently, followed by measurement and control with precision that is inversely proportional to the square root of the masses of the objects involved. The task becomes more difficult in the presence of noise, especially given that larger objects tend to interact more strongly with noisy environments and that the measurement process also introduces noise (Fig. 1A). The communication or processing protocol in which the entanglement might be used limits the amount of noise allowed before the entanglement is rendered useless.

Entanglement of mechanical motion was first demonstrated with two trapped atomic ions (2). It was generated deterministically and measured directly with high fidelity, and was therefore available as a resource that could be used for further processing. Taking the same level of quantum control and measurement

from the atomic scale to macroscopic engineered objects then remained an outstanding challenge. Important experimental milestones toward this goal have been reached using optical photons in a probabilistic scheme (3) or microwave radiation with indirect inference (4).

Here we strongly and deterministically entangle two massive mechanical oscillators and directly observe their state. Our technology allows for on-demand reproducible entanglement generation. For direct observation of the entangled state, we implement a near quantum-limited measurement of the position and momentum quadratures of both mechanical oscillators in every realization of the experiment. By repeating these measurements, we completely characterize their joint covariance matrix. This tomography demonstrates clear evidence of continuous-variable (CV) entanglement (5) in the measurement signals, without noise subtraction.

The entangled state measured here manifests strong correlations between seemingly disparate systems. The two-oscillator system can be characterized by the first and second moments of the dimensionless quadratures of motion $X_{j}$ and $P_{j}$ $(j = 1,2)$ , which satisfy the canonical commutation relations $[X_j,P_j] = i$ and are related to the underlying mechanical positions and momenta (6). After entanglement generation, the $X$ -quadrature of each harmonic oscillator is drawn from a Gaussian probability distribution with a variance that is large compared to its zero-point fluctuations.

However, when compared against each other, $X_{1}$ and $X_{2}$ are highly correlated, and similarly, $P_{1}$ and $P_{2}$ are anticorrelated. These features, however notable, could be consistent with classical correlations. To verify that the correlations originate from entanglement, we use the Simon-Duan criterion (7-9), calculated from the covariance matrix $C$ of $\vec{S} = (X_1,$

$P_{1},X_{2},P_{2})$ .Element $j,k$ of the $4\times 4$ covariance matrix is $C_{jk} = \frac{1}{2}\langle (S_k - \langle S_k\rangle)(S_j - \langle S_j\rangle) +$ $(S_{j} - \langle S_{j}\rangle)(S_{k} - \langle S_{k}\rangle)\rangle$ where $\langle \dots \rangle$ denotes expectation value. The smallest symplectic eigenvalue $\nu$ of the partially transposed covariance matrix quantifies the entanglement of the system (9, 10) [see (II) for explicit expressions]. The two-oscillator state is entangled if $\nu <  \frac{1}{2}$ ,where the zero-point fluctuations have variance $\frac{1}{2}$

To extract the full covariance matrix with minimal assumptions, we measure all four quadratures of motion describing the two oscillators in each single experiment. Our method is analogous to a heterodyne measurement of electromagnetic radiation and allows more accurate covariance matrix estimation than its homodyne counterpart (12). Crucially, this concurrent measurement improves substantially if it is efficient.

The inefficiency of our measurement apparatus can be modeled as an effective beam splitter (13-15), where each variable describing the motion $S_{j}\in \{X_{1}, P_{1},X_{2},P_{2}\}$ becomes mixed with vacuum noise such that $s_j = \sqrt{\eta_j} S_j + \sqrt{1 - \eta_j}\xi_j$ , where $\eta_{j}$ is the efficiency of the measurement of oscillator $j$ , $\xi_{j}$ is a Gaussian random variable with zero mean and vacuum variance $\langle \xi_j^2\rangle = \frac{1}{2}$ , and $\xi_{j}$ and $\xi_{k}$ are independently distributed for $j\neq k$ .

Therefore, following calibration of the measurement chain, we have direct access to the measured variables $s_j$ and their minimal symplectic eigenvalue $\nu_{\mathrm{meas}}$ . For the states measured here, $\nu_{\mathrm{meas}} < \frac{1}{2}$ implies that $\nu < \frac{1}{2}$ and vice versa (11). This mutual relation takes an even simpler form if in addition the state is symmetric with respect to exchanging the roles of the two oscillators and undergoes symmetric loss: $\nu_{\mathrm{meas}} = \eta \nu +(1 - \eta)\frac{1}{2}$ .

In both cases, if efficiencies are low, $\nu_{\mathrm{meas}}$ will approach $\frac{1}{2}$ , and it will be more difficult to certify that $\nu < \frac{1}{2}$ with high confidence. Moreover, quantum information protocols, such as teleportation and entanglement swapping, require a high measurement efficiency, as demonstrated for light fields (16, 17).

Our two mechanical oscillators are made of lithographically patterned thin-film aluminum that forms drum-like membranes (18), each with a mass of $\approx 70\mathrm{pg}$ suspended above a sapphire substrate (Fig. 1B). We use the $f_{\mathrm{m,1}} = 10.9\mathrm{MHz}$ mode of the left drum and the $f_{\mathrm{m,2}} = 15.9\mathrm{MHz}$ mode of the right drum. We manipulate and measure the motion of the drums using electromechanics (9).

The drums are embedded into a single microwave resonator, known as the "cavity," whose resonance frequency, centered at $f_{\mathrm{c}} = 6.0806\mathrm{GHz}$ , shifts according to the drums' motion (Fig. 1, C and D). A microwave pulse, reflected off the cavity, imparts forces on the drums and encodes the amplitudes of their quadratures of motion into Doppler-shifted sidebands. The carrier
frequency of the incoming microwave pulse determines whether the drums are cooled,

$^{1}$ National Institute of Standards and Technology, Boulder, CO 80305, USA. $^{2}$ Department of Physics, University of Colorado, Boulder, CO 80309, USA. $^{3}$ Center for Theory of Quantum Matter, University of Colorado, Boulder, CO 80309, USA. *Corresponding author. Email: shlomi.kotler@mail.huji.ac.il †Present address: Department of Applied Physics, The Hebrew University of Jerusalem, Jerusalem, 9190401, Israel.

RESEARCH

Kotler et al., Science 372, 622-625 (2021) 7 May 2021

1 of 4

![](dt=2026-02-28/ht=18/f7988c7db3f745eb6130ac89c3d275b01d50742ba5e4bedac895437d0be3a1e9.jpg)

![](dt=2026-02-28/ht=18/2f47aed73837f31f77ba621721a8c7c8b1f99934547330de32bc84a2cb291d5b.jpg)

![](dt=2026-02-28/ht=18/3a918184ca2ae241ac30fefb43afb226cc1bdb7c45468d55655857e962d7e48d.jpg)

![](dt=2026-02-28/ht=18/87edaf49c751edc5bc74f280de6fc325375e6e1ed805182ba95850b71cd5b209.jpg)

![](dt=2026-02-28/ht=18/8ee8423f503c3a5bf411f37e487872ff7b440bd4c3d4c50e30b71721f3397c4d.jpg)

entangled, or measured (Fig. 1E). To entangle the two drums, we irradiate the cavity with two pulses simultaneously. One pulse has a carrier frequency $f_{\mathrm{c}} + f_{\mathrm{m},1}$ . This results in a two-mode squeezing (TMS) interaction that entangles the cavity with drum 1, by generating correlated photon-phonon pairs: photons at a frequency $f_{\mathrm{c}}$ and phonons at $f_{\mathrm{m},1}$ (14). The other pulse has a carrier frequency $f_{\mathrm{c}} - f_{\mathrm{m},2}$ .

This results in a beam-splitter (BS) interaction that swaps cavity photons at a frequency $f_{\mathrm{c}}$ with phonons in drum 2 at a frequency $f_{\mathrm{m},2}$ (13). If the TMS and BS interactions were applied separately, the former would energize drum 1 and the latter would cool drum 2 (11). However, because we apply the TMS and BS pulses simultaneously, energy flows to both drums. Thus, the cavity mediates the interaction between the drums in a manner that is similar to other theoretical proposals (19-27). As a result, strong correlations form between the quadratures of motion.

Measurement is performed by amplifying a reflected microwave readout pulse after it has interacted with the drums. Typical microwave measurement efficiencies are $\sim 0.01$ even

when using the best commercially available low-noise amplifiers. Here we achieve higher effective efficiency by using the TMS interactions native to the device as a preamplifier, on the basis of techniques that were developed for single-drum readout (14, 15, 28, 29). We extend these methods, using frequency multiplexing, and improve our measurement efficiencies by more than an order of magnitude: $\eta_{1} = 0.26(2)$ , and $\eta_{2} = 0.153(3)$ . Ultimately, the Heisenberg uncertainty principle prevents these efficiencies from exceeding $\frac{1}{2}$ because they quantify a concurrent measurement of both quadratures of motion (6).

Figure 2 shows tomography of the two-drum system, as characterized by its covariance matrix, for different protocols. First, we prepare a fiducial cold state by applying a pulse sequence of ground-state cooling followed by readout, rendering a single concurrent measurement of $x_{1}, p_{1}, x_{2}, p_{2}$ . Figure 2A shows the experimental distribution of the measured variables for 10,000 repetitions of the experiment. The two-dimensional (2D) histograms show no correlation between any of the measured variables. This is reaffirmed by the

covariance matrix of the state, which is diagonal to a good approximation (Fig. 2C). The magnitude of the diagonal elements correspond to nearly ground-state variances of $V_{1} = 0.75(1)$ and $V_{2} = 0.63(1)$ for drum 1 and 2, respectively, where $V_{j} = \frac{1}{2}\left(\left\langle (x_{j} - \langle x_{j}\rangle)^{2} + (p_{j} - \langle p_{j}\rangle)^{2}\right\rangle\right)$ for $j = 1,2$ . We now turn to a pulse sequence of ground-state cooling, entanglement, and readout. Figure 2B exhibits all the expected features of a highly correlated state.

First, the $x,p$ histogram for each drum is consistent with a Gaussian distribution of large variance. Drum 1, which undergoes a TMS interaction, has a bigger variance $(V_{1} = 10.9(1))$ than that of drum 2 $(V_{2} = 4.63(5))$ , which undergoes a BS interaction. Second, a clear signature of drum-drum interaction is demonstrated by the correlation of $x_{1},x_{2}$ and the anticorrelation of $p_1,p_2$ . The covariance matrix of the measured variables in Fig. 2D displays a dominant diagonal and four off-diagonal elements $C_{1,3}\approx -C_{2,4}$ and $C_{3,1}\approx -C_{4,2}$ .

Indeed, these clear correlations are directly observable in the measured variables. Delineating them from classical correlations requires an application of the Simon-Duan criteria.

RESEARCH | REPORT

Kotler et al., Science 372, 622-625 (2021) 7 May 2021

2 of 4

![](dt=2026-02-28/ht=18/7224c3571377c29f8b9a645e0c5ead7fd72353a25015c5a7b55c47bfd5e7253d.jpg)

![](dt=2026-02-28/ht=18/027646c6aa0acc13b0b52a4435b6046309b71b4973d549a59dea84866cbf2f45.jpg)

![](dt=2026-02-28/ht=18/738f55c14aa1b7b251a08d9e2ec072098319a75d302eec714d1677e5056042c5.jpg)

![](dt=2026-02-28/ht=18/6ee48c46548eb3a081c26723a227a87ce76e7312fbf93f47b5c4219a9ea830a4.jpg)

(A) Histograms of a sideband-cooled state of two drums. Each experiment records the system variables $\vec{s} = (x_{1},p_{1},x_{2},p_{2})$ concurrently. Variables are scaled to dimensionless units according to the canonical commutation relation $[x_j,p_j] = i$ for $j = 1,2,$ so the vacuum state has variance $\frac{1}{2}$ . Panel with legend $s_j,s_k$ corresponds to a correlation of $s_j$ along the $x$ axis and $s_k$ along the $y$ axis, quantified using a normalized 2D histogram of 10,000 repetitions of the experiment.

The drums' individual variances are $V_{1} = 0.75(1)$ and $V_{2} = 0.63(1)$ . (B) Histograms of an entangled state of two drums. After sideband cooling, a 16.8-μs entangling pulse generates $x_{1},x_{2}$ correlation and $p_1,p_2$ anticorrelation. Because the entanglement pulse pumps energy into the two-drum system, each drum's individual variance grows from its ground-state cooled value to $V_{1} = 10.9(1)$ and $V_{2} = 4.63(5)$ , respectively. (C) Covariance matrix of the data in (A). (D) Covariance matrix of the data in (B).

Correlations and anticorrelations are apparent in the off-diagonal elements.

Evolution of the two-drum state is shown for different entangling pulse durations (Fig. 3), all of which are kept shorter than $100\mu \mathrm{s}$ to avoid the thermal decoherence of both drums (11). First, we focus on the individual variances of each drum (Fig. 3A). At short times $< 1\mu \mathrm{s}$ , the drums show no evidence of interaction. Drum 2 cools while drum 1 becomes energized, the same behavior that would have been expected if the drums did not interact with one another. The dashed lines in Fig. 3A

![](dt=2026-02-28/ht=18/cde2e8e457b4e1f137696ac0a2b8c727375d655d55f53f85d1db62dc86fee021.jpg)

![](image)
a38255eaa424dfe99ca47a254.jpg)

![](dt=2026-02-28/ht=18/deb5e2bbba9c475ddb2d0a9284581c8c834c802d1861ae8449e5001d23c4db11.jpg)

![](dt=2026-02-28/ht=18/cb6c1aee2fed8029be6e882a7b32393e2d595c7b222952e8dab828ce2d6f98d0.jpg)

Entangling is composed of an energizing pulse for drum 1 and a cooling pulse for drum 2, applied simultaneously. The dashed lines show a theoretical prediction of the drums' individual variances if energizing and cooling were employed separately. Blue and red marks are the measured variances $V_{1}$ and $V_{2}$ for drums 1 and 2, respectively. Solid lines in all panels show theory with parameters obtained by fitting to an independent data set and without further adjustment (11).

(B) Angle of the $x_{1}, x_{2}$ correlation that determines the squeezed and anti-squeezed joint quadratures of the bipartite system. (C) Entanglement in the measured variables, after loss, quantified by $\nu_{\text{meas}}$ . Points below $\frac{1}{2}$ indicate entanglement of the two drums. Statistical error bars, quantified by 1-σ bias-corrected bootstrapping confidence intervals, are smaller than the markers for most points. Shaded gray area corresponds to a 1-σ uncertainty region in the location of the black theoretical prediction curve caused by measurement efficiency uncertainties.

Inset shows the systematic uncertainty (±1-σ) of the last measured point in the main graph (circle), indicated by the upper and lower points (triangles). The last point attains $\nu_{\text{meas}} = 0.44_{-0.004}^{+0.004}(\text{stat})_{-0.021}^{+0.022}(\text{sys})$ . (D) Entanglement in the mechanical variables, prior to loss, quantified by $\nu$ . Uncertainties and inset plot are similar to those in (C). The last point attains $\nu = 0.18_{-0.02}^{+0.03}(\text{stat})_{-0.11}^{+0.13}(\text{sys})$ .

extrapolate this noninteracting behavior. Entangling pulse durations of $>1\mu s$ result in both drums deviating from this independent evolution; their variances now grow together in time with similar rates. Second, recall that in Fig. 2C, the histogram plot exhibited a correlation between $x_{1}$ and $x_{2}$ . Theory predicts that the angle that the elliptical $x_{1}, x_{2}$ distribution's major axis makes with the horizontal will evolve as shown by the solid theory line in Fig. 3B. Third, we focus on the measured variable $\nu_{\mathrm{meas}}$ , shown in Fig. 3C.

Because our cooling of the drums is imperfect, their initial state contains some residual thermal motion in addition to the quantum fluctuations. The entangling operation must overcome this classical noise before the drums can be truly entangled. This is why, for pulse durations shorter than $\sim 4\mu s$ , $\nu_{\mathrm{meas}} > \frac{1}{2}$ which strongly suggests that the drums are only classically correlated.

For longer pulse durations, true quantum behavior, inconsistent with classical correlation, is observed and $\nu_{\mathrm{meas}}$ crosses below $\frac{1}{2}$ , indicating entanglement. At $16.8 - \mu s$ interaction time, we observe $\nu_{\mathrm{meas}} = 0.44_{-0.004}^{+0.004}(\mathrm{stat})_{-0.021}^{+0.022}(\mathrm{sys})$ ,

where "sys" indicates systematic uncertainty dominated by uncertainty in the measurement efficiencies, and "stat" indicates statistical uncertainty estimated using bootstrapping (11). This is a direct measurement of entanglement for a bipartite system of macroscopic objects. It quantifies the amount of entanglement left in the system after noise processes have intervened during the measurement, and is therefore important for future quantum information applications. The amount of entanglement prior to the effect of noise can be estimated as well.

To that end, we use a semidefinite program to find the closest (in $l_{2}$ distance) physically realizable covariance matrix that, after an interaction with noise, is consistent with the data. From that we estimate the entanglement criterion $\nu$ , which is shown in Fig. 3D. Indeed, we see more entanglement for the same interaction time, with $\nu = 0.18_{-0.02}^{+0.03}(\mathrm{stat})_{-0.11}^{+0.13}(\mathrm{sys})$ . Such a level of entanglement might be useful for the exploration of mesoscopic Einstein-Podolsky-Rosen nonlocality (30) and fundamental tests of quantum mechanics (27).

RESEARCH | REPORT

Kotler et al., Science 372, 622-625 (2021) 7 May 2021

3 of 4

Our results demonstrate pulsed, time-domain control of three important building blocks for CV quantum information processing and quantum communication: state initialization, entanglement, and measurement. Pulsed control played a key role. It allowed optimization of each piece separately and improved our measurement efficiency by more than an order of magnitude compared to traditional steady-state operation.

As a result, we generated a highly entangled state of two macroscopic mechanical oscillators, surpassing the entanglement threshold by $4.43_{-0.7}^{+0.6}(\mathrm{stat})_{-2.3}^{+4.4}(\mathrm{sys})\mathrm{dB}$ . Most excitingly, we observe entanglement directly in the measured variables. This is relevant to future applications that require decisions based on measurement outcomes. We therefore expect the methods described here to serve as a stepping stone for teleportation and entanglement swapping of states of massive objects.

This would enable hybrid quantum networks, in which mechanics entangles with microwave fields (14) or with spin systems (31), or is used as an intermediary to entangle radiation (32, 33).

# REFERENCES AND NOTES

33. J. Chen, M. Rossi, D. Mason, A. Schliesser, Nat. Commun. 11, 943 (2020).

# ACKNOWLEDGMENTS

We thank B. Katz and D. Ben-Zvi for feedback and insight on data taking and analysis. We thank B. Hauer and A. Sirois for their careful reading of the manuscript. We thank K. Lehnert and R. Delaney for useful discussions on measurement efficiency. We thank N. Kotler for consulting on data and concept visualization.

Funding: At the time this work was performed, S.K., E.S. F.L., A.K., and S.Geller were supported as Associates in the Professional Research Experience Program (PREP) operated jointly by NIST and the University of Colorado Boulder under Award no. 70NANB18H006 from the U.S. Department of Commerce. Author contributions: J.D.T. and S.K. designed the experiment. S.K. and G.A.P. fabricated the device. F.L. and K.C. supervised device fabrication. S.K. performed the measurements. F.L., R.W.S., and J.A. advised on measurement techniques. S.K., E.S., A.K., and S.

Geller developed the theory and wrote, analyzed, and tested the data analysis code. S.K. and E.S. analyzed the results. S.Glancy and E.K. supervised theory work and data analysis. S.K., G.A.P., F.L., K.C., R.W.S., J.A., and J.D.T. developed the experimental infrastructure necessary to conduct the experiment. J.D.T. supervised the work. All authors provided experimental suggestions, discussed the results, and contributed to the writing of the manuscript. Competing interests: The authors declare no competing interests. S.K. is also affiliated with Qedma Quantum Computing Ltd. G.A.P.

is also affiliated with PsiQuantum. Data and materials availability: All data are available in the manuscript or the supplementary materials. This is a contribution of the National Institute of Standards and Technology, not subject to U.S. copyright.

# SUPPLEMENTARY MATERIALS

science.sciencemag.org/content/372/6542/622/suppl/DC1 Supplementary Text

Figs. S1 to S4

Tables S1 to S3

References (34-42)

16 October 2020; accepted 4 March 2021

10.1126/science.abf2998

RESEARCH | REPORT

Kotler et al., Science 372, 622-625 (2021) 7 May 2021

4 of 4

# Direct observation of deterministic macroscopic entanglement

Shlomi Kotler, Gabriel A. Peterson, Ezad Shojaee, Florent Lecocq, Katarina Cicak, Alex Kwiatkowski, Shawn Geller, Scott Glancy, Emanuel Knill
, Raymond W. Simmonds, José Aumentado and John D. Teufel

Science 372 (6542), 622-625. DOI: 10.1126/science.abf2998

# Quantum entanglement goes large

Quantum entanglement occurs when two separate entities become strongly linked in a way that cannot be explained by classical physics; it is a powerful resource in quantum communication protocols and advanced technologies that aim to exploit the enhanced capabilities of quantum systems. To date, entanglement has generally been limited to microscopic quantum units such as pairs or multiples of single ions, atoms, photons, and so on. Kotler et al. and Mercier de Lépinay et al.

demonstrate the ability to extend quantum entanglement to massive macroscopic systems (see the Perspective by Lau and Clerk). Entanglement of two mechanical oscillators on such a large length and mass scale is expected to find widespread use in both applications and fundamental physics to probe the boundary between the classical and quantum worlds.

Science, this issue p. 622, p. 625; see also p. 570

ARTICLE TOOLS http://science.sciencemag.org/content/372/6542/622

SUPPLEMENTARY MATERIALS http://science.sciencemag.org/content/suppl/2021/05/05/372.6542.622.DC1

http://science.sciencemag.org/content/sci/372/6542/570.full  
http://science.sciencemag.org/content/sci/372/6542/625.full  
file:/content

REFERENCES This article cites 41 articles, 2 of which you can access for free: http://science.sciencemag.org/content/372/6542/622#BIBL

PERMISSIONS http://www.sciencemag.org/help/reprints-and-permissions

Use of this article is subject to the Terms of Service

Science (print ISSN 0036-8075; online ISSN 1095-9203) is published by the American Association for the Advancement of Science, 1200 New York Avenue NW, Washington, DC 20005. The title Science is a registered trademark of AAAS.

Copyright © 2021 The Authors, some rights reserved; exclusive licensee American Association for the Advancement of Science. No claim to original U.S. Government Works

Science