# Laser cooling of a micromechanical membrane to the quantum backaction limit

R.W. Peterson, $^{1,2}$ T.P. Purdy, $^{1,2}$ N.S. Kampel, $^{1,2}$ R.W. Andrews, $^{1,2}$

P.-L. Yu, $^{1,2}$ K.W. Lehnert, $^{1,2,3}$ and C.A. Regal $^{1,2}$

$^{1}$ JILA, University of Colorado and NIST, Boulder, Colorado 80309, USA

$^{2}$ Department of Physics, University of Colorado, Boulder, Colorado 80309, USA

$^{3}$ National Institute of Standards and Technology (NIST), Boulder, Colorado, 80305, USA

# Abstract

The radiation pressure of light can act to damp and cool the vibrational motion of a mechanical resonator. In understanding the quantum limits of this cooling, one must consider the effect of shot noise fluctuations on the final thermal occupation. In optomechanical sideband cooling in a cavity, the finite Stokes Raman scattering defined by the cavity linewidth combined with shot noise fluctuations dictates a quantum backaction limit, analogous to the Doppler limit of atomic laser cooling. In our work we sideband cool to the quantum backaction limit by using a micromechanical membrane precooled in a dilution refrigerator. Monitoring the optical sidebands allows us to directly observe the mechanical object come to thermal equilibrium with the optical bath.

Laser cooling revolutionized atomic physics and paved the way to the creation of extreme states of matter with ensembles of atoms. Recent achievements in optomechanics have paralleled the early development of atomic laser cooling $[1, 2]$ , and raise the promise of using micromechanical resonators in a variety of quantum devices. The prospects for sensitive manipulation of macroscopic objects with light were first studied in the context of gravitational wave detection $[3, 4]$ . It was recognized that the radiation pressure force of intense laser light could act as a source of dissipation for a harmonically bound object, an effect termed dynamical backaction. Not until later was the quantum nature of dynamical backaction considered: Shot noise imparts a random backaction force on the resonator, termed radiation pressure shot noise (RPSN) $[5]$ . In the context of optomechanical sideband cooling, the relative rates of Stokes and anti-Stokes Raman scattering define a quantum backaction limit below which the resonator cannot be cooled $[6, 7]$ . This limit is analogous to the Doppler limit in atomic laser cooling, which is the minimum achievable temperature on a given linewidth atomic transition, due to the randomly-oriented momentum kicks from the spontaneously emitted photons $[8]$ .

Recent work in optomechanics has explored the interaction of light and mechanical motion in the quantum regime, from observing the backaction induced by position measurement $[9]$ , to using backaction to generate squeezed states of light $[10, 11]$ . Although sideband cooling has allowed preparation of a mechanical resonator into its quantum ground state $[1, 2]$ , a regime where the quantum nature of backaction is relevant $[12–14]$ , these experiments still operate far from the quantum limit of sideband cooling. In this Letter, by precooling a micromechanical membrane in a dilution refrigerator, we are able to directly observe the mechanical resonator come to thermal equilibrium with an optical bath, which for the sideband resolving power of our cavity reaches a mechanical phonon occupation of $\bar{n} = 0.20 \pm 0.02$ . Even at the low phonon occupation that defines the quantum backaction limit in our device, we observe no evidence of heating of the mechanical material due to optical absorption. This is a crucial realization for future application of cryogenic optomechanical devices such as transduction between microwave and optical photons $[15, 16]$ .

In an optomechanical sideband cooling experiment, the cooling laser is red-detuned from a resonant mode of an optical cavity coupled to mechanical motion. The cavity's susceptibility enhances the near-resonant anti-Stokes scattered light, which removes mechanical quanta from the mechanical resonator when it exits the cavity, and suppresses the far-off-resonant

Stokes scattered light, which adds mechanical quanta (Fig. 1a). This net cooling effect can be strengthened by increasing the cooling laser power, with the mechanical mode's final temperature a balance between its intrinsic thermal bath temperature and the cold optical bath provided by the cooling laser and cavity. In the so-called resolved-sideband regime, near-complete suppression of Stokes scattering [1, 2] can be achieved by setting the cavity linewidth $\kappa$ to be much smaller than the mechanical frequency $\omega_{m}$ . In this limit, optical cooling can only remove energy from the mechanical system. However, for any finite sideband resolution, the cooling due to the imbalance of cavity susceptibility at each sideband is eventually undone by the fundamental asymmetry of Raman scattering. Namely, anti-Stokes scattering is proportional to the mechanical mode's average phonon occupation $\bar{n}$ , while Stokes scattering is proportional to $\bar{n} + 1$ and hence can become an important contribution at low $\bar{n}$ . When both scattering rates are equal, no further cooling is possible, leaving the mechanical mode in thermal equilibrium with the optical bath. For optimal detuning of the cooling laser ( $\Delta_{\mathrm{opt}} = -\omega_m\sqrt{1 + \kappa^2 / 4\omega_m^2}$ ), this temperature limit of the optical bath in units of mechanical quanta is given by $n_{\mathrm{ba}} = (\kappa /4\omega_m)^2$ in the resolved-sideband regime.

As shown in Fig. 1b, the process by which the mechanical motion comes into thermal equilibrium with the optical bath can be observed directly by monitoring the Stokes and anti-Stokes sideband amplitudes. In our experiment, we collect the light that is Raman-scattered by the cavity directly from the red-detuned cooling laser and perform a heterodyne measurement to separate the sideband amplitudes. Our experiments operate in a regime where $\kappa \approx \omega_{m}$ . Here it is possible to be near the mechanical ground state and at the quantum backaction limit simultaneously. Additionally, collecting the Raman-scattered light for thermometry is practical because the Stokes sideband is only partially suppressed by the cavity. The initial asymmetry of the sidebands is given by the ratio of cavity susceptibility set by the cooling laser detuning $\Delta$ . Hence, in Fig. 1b the anti-Stokes light dominates initially. As the cooling laser power is increased, the coherent cooling rate $\Gamma_{\mathrm{opt}}$ between the mechanical mode and the optical bath is increased. As the mode nears the ground state, the difference between the bosonic factors $\bar{n}$ and $\bar{n} + 1$ in the anti-Stokes and Stokes sidebands manifests itself by modifying the sideband asymmetry [12, 17-21]. Here the mode temperature can be determined directly from the scattered light from the cooling laser and hence does not require detailed knowledge of system parameters. The phonon occupation is given by a

rate equation that describes the mechanical mode's response to both its environment (at temperature $n_0$ ) and $n_{\mathrm{ba}}$ [22], which can be written:

$$
\bar {n} (\Gamma_ {\mathrm{opt}}) = \frac {n _ {0} \Gamma_ {0} + n _ {\mathrm{ba}} \Gamma_ {\mathrm{opt}}}{\Gamma_ {0} + \Gamma_ {\mathrm{opt}}} \tag {1}
$$

When $\Gamma_{\mathrm{opt}}$ starts to dominate over the mechanical mode's coupling $\Gamma_0$ to its environment, laser cooling begins while maintaining the initial sideband ratio. Once the product $n_{\mathrm{ba}}\Gamma_{\mathrm{opt}}$ exceeds the mechanical decoherence rate $n_0\Gamma_0$ , cooling ceases at the backaction limit ( $\bar{n} = n_{\mathrm{ba}}$ ) where the Stokes and anti-Stokes rates are equal. Here the competing Stokes and anti-Stokes scattering accomplish nothing but allowing shot noise fluctuations of the cooling laser to set a finite temperature [6, 7].

Our optomechanical cavity consists of the optical mode of a Fabry-Perot cavity with $\kappa = 2\pi \times 2.6$ MHz coupled to the motion of the $\omega_{m} = 2\pi \times 1.48$ MHz mode of a 500 $\mu \mathrm{m}$ square by $40~\mathrm{nm}$ thick $\mathrm{Si}_3\mathrm{N}_4$ drum resonator (Fig. 1c). The cooling laser is injected into a 10-ppm-transmission mirror. The Raman-scattered light preferentially couples out the second, 100-ppm-transmission mirror, where it is collected via heterodyne detection. An auxiliary locking laser is injected in an orthogonal polarization into the $100~\mathrm{ppm}$ mirror. The cavity is anchored to the base of a dilution refrigerator and light is coupled into the cavity via free space through a narrow cryogenic beam path designed to filter $300\mathrm{K}$ blackbody radiation [23]. The optical bath is coupled to the mechanical mode at $\Gamma_{\mathrm{opt}} \leq 2\pi \times 30\mathrm{kHz}$ , proportional to cooling laser power. The mechanical mode is also coupled to its thermal environment (consisting of cryostat temperature, locking laser RPSN, and other effects), which, in units of mechanical quanta, has a temperature $n_0 = k_B T_0 / \hbar \omega_m \sim 10^3$ corresponding to a temperature $T_0$ (to be determined below), where $k_B$ is Boltzmann's constant and $\hbar$ is the reduced Planck's constant. The high quality-factor $\mathrm{Si}_3\mathrm{N}_4$ membrane provides a very low coupling rate to $n_0$ , with $\Gamma_0 = 0.18\mathrm{Hz}$ , measured via ringdown of a mechanical excitation.

Collecting the Raman-scattered light in heterodyne detection allows for thermometry of the mechanical mode without additional probe lasers, since both sidebands are visible in the heterodyne spectrum (Fig. 2 insets at low and high $\Gamma_{opt}$ ). A simultaneous fit to both mechanical sidebands with a common $\omega_{m}$ (separation from the heterodyne beat note) and $\Gamma_{opt}$ gives Stokes and anti-Stokes sideband amplitudes normalized to the off-resonant background set by the shot noise of the cooling laser—let the ratio of these amplitudes be R.

![](images/f9357b15cda8ee91503aa386cb09923507535d32c18889293ae4b329f8f93ad8.jpg)

![](images/3ae45b76f47f40a8bbbc3a238f93af6da7c7b809ec5bac659119a10877580a27.jpg)

<details>
<summary>text_image</summary>

C
10 ppm
100 ppm
Cooling laser
Dilution refrigerator.
Locking laser
LO
</details>

FIG. 1. Optomechanical sideband cooling and Raman-ratio thermometry. a) Stokes and anti-Stokes scattering rates $\Gamma_{+}$ and $\Gamma_{-}$ depend on both the mechanical mode's phonon occupation $\bar{n}$ , as well as the factors proportional to cavity susceptibility at $\Delta \mp \omega_{m}$ (gray curve; full linewidth $\kappa$ ). b) Fractional sideband amplitude for the case of backaction limit $n_{\mathrm{ba}} < 1$ (log-log scale). When negligible optical cooling is applied ( $\Gamma_{\mathrm{opt}} < \Gamma_0$ ) (left), the Stokes (red) and anti-Stokes (blue) sidebands correspond to the ratio of cavity susceptibility, but the cooling laser is too weak to lower the temperature. In the classical cooling regime (center), temperature decreases while the ratio of Stokes and anti-Stokes sidebands remains constant. When $\Gamma_{\mathrm{opt}} \approx n_0\Gamma_0$ , the mechanical decoherence rate, $\bar{n}$ is approaching the ground state. At $\Gamma_{\mathrm{opt}} \approx \frac{n_0}{n_{\mathrm{ba}}}\Gamma_0$ , backaction and thermal motion equally contribute to mechanical motion. Beyond this is the backaction limit regime (right). c) Experimental setup. The cooling laser (orange) is injected into the optical cavity through the 10 part per million (ppm) transmission mirror. Transmitted cooling laser light passes through the 100 ppm mirror and is collected in heterodyne detection with a local oscillator (LO). The cavity is actively stabilized using a locking laser (purple) injected into the orthogonal polarization mode of the cavity.

By extrapolating our data to $\Gamma_{opt}=0$ , we fit the amplitude ratio s due to cavity susceptibility only. Now, we can directly compute $\bar{n}^{-1}=\frac{R}{s}-1$ . For each cooling experiment, $\Delta$ is determined by the extrapolated parameter s, a function only of cavity susceptibility with otherwise independently measured parameters. A typical cooling experiment is shown in Fig. 2. As the cooling laser power is increased, $\Gamma_{opt}$ increases. Monitoring Stokes and

anti-Stokes sideband amplitudes (Fig. 2a and insets) shows the classical regime of constant sideband ratio transitioning to the quantum backaction limit, with equal sideband heights as the mechanical resonator comes into equilibrium with the optical bath. Inferred $\bar{n}$ (Fig. 2b) shows saturation of the temperature at $n_{ba}$ (dashed line). For low $\Gamma_{opt}$ , the situation is analogous to previous optomechanical sideband cooling demonstrations. Classically, one expects an inverse relation between $\bar{n}$ and $\Gamma_{opt}$ . Typically when deviations from this expectation are observed they result from effects such as physical heating due to absorption of light, entering the strong coupling regime, or interference from classical amplitude or phase noise on the cooling laser [22, 24]. However, we find a final temperature in close agreement with $n_{ba} = 0.18$ , determined by independently-measured cavity parameters; the last data point in the cooling curve for Fig. 2 is our lowest measured $\bar{n} = 0.20 \pm 0.02$ for $\Delta = -2\pi \times 1.62$ MHz.

For an arbitrary detuning of the cooling laser from the cavity, the quantum backaction limit is expected to change as the sideband resolution is modified. Specifically, the quantum backaction limit $n_{ba}$ , expressed as an average phonon occupation, takes the form [6]

$$
n _ {\mathrm{ba}} (\Delta) = - \frac {(\omega_ {m} + \Delta) ^ {2} + (\kappa / 2) ^ {2}}{4 \omega_ {m} \Delta} \tag {2}
$$

We have repeated the cooling experiment for a variety of $\Delta$ , and can demonstrate saturation of the $\bar{n} = n_{ba}$ quantum backaction limit at a variety of minimum temperatures (Fig. 3). The divergence of $n_{ba}$ as $\Delta \rightarrow 0$ corresponds to the RPSN condition at $\Delta = 0$ , where there is no sideband cooling or heating but shot noise in the amplitude quadrature of the light drives the mechanical motion [9].

We must also consider the effect of potential classical noise sources on the measurement. Overall, the fact that $\bar{n}$ saturates at the expected $n_{ba}$ is one indicator that classical amplitude and phase noise are not significant systematic errors in the measurement; we have modeled the functional dependences of a variety of forms of classical noise as a function of $\Delta$ and find they will generally make the apparent $\bar{n}$ lie above or below the theoretically predicted $n_{ba}$ [18, 19, 25, 26]. Nonetheless, we complete a number of independent checks. First, we independently measure the level of classical amplitude (phase) noise on the cooling laser, finding it to be 0.2% (2%) of shot noise at 5 $\mu$ W, a representative power at which $\bar{n} = n_{ba}$ . An analysis of the sideband cooling data that explicitly includes effects from cooling laser amplitude and phase noise changes the final measured phonon occupation in Fig. 2b by

![](images/ff168a9c0cdc1d445446e8d5f4bda99396ce1606adc364c035b61a20e9314219.jpg)

<details>
<summary>line</summary>

| Cooling rate Γ_opt (Hz) | Sideband amplitude (SN) - Blue | Sideband amplitude (SN) - Red | Phonon occupation η - Blue | Phonon occupation η - Red |
| ------------------------ | ------------------------------ | ----------------------------- | -------------------------- | ------------------------- |
| 100                      | ~2.5                           | ~0.7                          | ~10                        | ~10                       |
| 300                      | ~1.2                           | ~0.3                          | ~5                         | ~5                        |
| 1000                     | ~0.4                           | ~0.1                          | ~1                         | ~1                        |
| 3000                     | ~0.2                           | ~0.05                         | ~0.5                       | ~0.5                      |
| 10000                    | ~0.1                           | ~0.03                         | ~0.2                       | ~0.2                      |
| 30000                    | ~0.05                          | ~0.02                         | ~0.1                       | ~0.1                      |
</details>

FIG. 2. Reaching the quantum backaction limit of sideband cooling near $\Delta_{\mathrm{opt}}$ . a) Sideband amplitudes. As $\Gamma_{\mathrm{opt}}$ is increased, the ratio of Stokes (red circles, data; red line, fit) and anti-Stokes (blue circles, data; blue line, fit) sideband amplitudes approaches one. Sideband amplitude is normalized to shot noise (SN). The systematically small values of the sideband amplitudes at largest $\Gamma_{\mathrm{opt}}$ and their small effect on the measurement of $\bar{n}$ are discussed in the main text. b) Minimum temperature. Mechanical occupancy $\bar{n}$ (purple squares) saturates at $\bar{n} = 0.20 \pm 0.02$ for the largest $\Gamma_{\mathrm{opt}}$ , in agreement with the quantum backaction limit (dashed black), which is at $n_{\mathrm{ba}} = 0.18$ for $\Delta = -2\pi \times 1.62$ MHz. A fit to the data (solid black) can be used to infer minimum phonon occupation, as well as the bath temperature $T_0$ of the mode by extrapolation to $\Gamma_{\mathrm{opt}} = 0$ . Insets) Overlaid Stokes (red) and anti-Stokes (blue) mechanical spectra, normalized to shot noise. Spectra are third-smallest (left) and third-largest (right) data points.

$\Delta \bar{n}^{\mathrm{laser}} \approx 0.006$ , less than half the size of the statistical error in the measurement. Another systematic error is the presence of off-resonant substrate mechanical modes that rise above the shot noise floor [24, 27, 28]. The normalization of sideband amplitude in our analysis to the off-resonant shot noise level is affected by this noise, causing both sideband amplitudes to lie below the fit in Fig. 2a. However, since $\bar{n}$ is a function of the sideband ratio, not their absolute amplitudes, it is not strongly affected, and again leads to a small $\Delta \bar{n}^{\mathrm{sub}} \approx 0.006$ . Additional confirmation of the substrate noise's small effect is that the mechanical sidebands (Fig. 2 inset) retain a Lorentzian lineshape [25]. Because both laser noise and substrate noise are small, we otherwise do not include the effects of classical noise directly in the data presentation.

For each cooling curve, sideband thermometry and knowledge of $\Gamma_{0}$ allow for extrapolation of $n_{0}$ , and therefore the effective temperature $T_{0}$ of the mechanical mode. We find a constant bath temperature of $T_{0} = 360$ mK. Beyond the cryostat temperature (70 mK), we can partially trace its origin to RPSN from the locking laser (170 mK), with the balance due to effects whose origin is not completely known. As RPSN from the locking laser was the dominant thermal effect, improved detection of weaker powers would allow for operation at a lower $T_{0}$ . Additionally, increasing locking laser power from 0.6 $\mu$ W to 4.8 $\mu$ W adds RPSN that raises $T_{0}$ , but does not affect laser cooling to $\bar{n} = n_{ba}$ . Importantly, the lack of power-dependence in $T_{0}$ suggests that material absorption is not a dominant effect. This demonstration is key to advancing cryogenic compatibility of these optomechanical devices, which especially for sub-kelvin temperatures can suffer from limited thermalization to cryostat base temperature and heating due to material absorption [23, 29].

The visibility of the mechanical motion against the imprecision noise floor in units of shot noise (Fig. 2a) is defined by a low total collection efficiency, independently measured to be $\epsilon = 0.04$ based on detection of squeezed light produced by the device. For this work, this excess noise was inconsequential because we only require the relative amplitude of the Stokes and anti-Stokes sidebands. However, other quantum measurement applications rely on the absolute imprecision of the measurement. The low efficiency in our measurements was due to heterodyne visibility, cavity losses, propagation loss, and detector efficiency. The dominant contribution was heterodyne visibility, which can be improved with better alignment.

The work shown here explores the quantum backaction limit of optomechanical sideband cooling, and demonstrates a mechanical resonator in thermal equilibrium with an optical

![](images/5f6254a6d9447c07ec628b7baa516dae36f4b4934e4119da4f5e2f7001d64721.jpg)

<details>
<summary>line</summary>

| Cooling laser detuning Δ (MHz) | Minimum phonon occupation |
| ------------------------------ | -------------------------- |
| -1.7                           | 0.2                        |
| -1.5                           | 0.2                        |
| -1.0                           | 0.4                        |
| -0.8                           | 0.3                        |
| -0.5                           | 1.0                        |
| -0.2                           | 2.5                        |
| 0.0                            | 6.0                        |
</details>

FIG. 3. Saturation of the quantum backaction limit. Data (blue open circles) are the fits to the lower limit of $\bar{n}$ in cooling curves such as Fig. 2. $n_{ba}$ (dashed line) is given by Eqn. 2. Insets: Ratio of sidebands in the classical cooling regime for $\Delta\simeq-1.5$ MHz (left) and $\Delta\simeq-0.5$ MHz (right). A larger ratio in cavity susceptibility allows a lower final $n_{ba}$ .

bath rather than its thermal environment. The parameters of our device allow for saturation of the quantum backaction limit while the mechanical mode is in its quantum ground state, coupling mechanical and optical degrees of freedom both in the quantum regime. Recently, similar regimes of coupling have been reached in demonstrations of squeezed mechanical motion $[30]$ and backaction-limited coupling between multiple mechanical modes $[31]$ . We note that the quantum backaction limit can be circumvented by introducing additional couplings to alter the dynamics of the cavity optomechanical system, for example dissipative optomechanical coupling $[32, 33]$ , or measurement and active feedback $[34–36]$ .

# ACKNOWLEDGMENTS

This work was supported by DURIP, the DARPA QuASAR program, AFOSR-MURI, AFOSR PECASE, and the National Science Foundation under grant number 1125844. C.R. thanks the Clare Boothe Luce Foundation for support. P.-L.Y. thanks the Taiwan Ministry of Education for support. We thank Antoinne Heidmann and Aurelien Kuhn for design information about free-space optical cryostats.

[1] Jasper Chan, T. P. Mayer Alegre, Amir H. Safavi-Naeini, Jeff T. Hill, Alex Krause, Simon Groblacher, Markus Aspelmeyer, and Oskar Painter, “Laser cooling of a nanomechanical oscillator into its quantum ground state,” Nature 478, 89–92 (2011).   
[2] J. D. Teufel, Dale Li, M. S. Allman, K. Cicak, A. J. Sirois, J. D. Whittaker, and R. W. Simmonds, “Circuit cavity electromechanics in the strong-coupling regime,” Nature 471, 204–208 (2011).   
[3] Vladimir Braginsky and A. B. Manukin, “Ponderomotive effects of electromagnetic radiation,” Journal of Experimental and Theoretical Physics 25, 653 (1967).   
[4] Vladimir Braginsky, A. B. Manukin, and M. Y. Tikhonov, “Investigation of dissipative ponderomotive effects of electromagnetic radiation,” Journal of Experimental and Theoretical Physics 31, 829 (1970).   
[5] Carlton M. Caves, “Quantum-mechanical radiation-pressure fluctuations in an interferometer,” Phys. Rev. Lett. 45, 75–79 (1980).   
[6] Florian Marquardt, Joe P. Chen, A. A. Clerk, and S. M. Girvin, “Quantum theory of cavity-assisted sideband cooling of mechanical motion,” Phys. Rev. Lett. 99, 093902 (2007).   
[7] I. Wilson-Rae, N. Nooshi, W. Zwerger, and T. J. Kippenberg, “Theory of ground state cooling of a mechanical oscillator using dynamical backaction,” Phys. Rev. Lett. 99, 093901 (2007).   
[8] Steven Chu, L. Hollberg, J. E. Bjorkholm, Alex Cable, and A. Ashkin, “Three-dimensional viscous confinement and cooling of atoms by resonance radiation pressure,” Phys. Rev. Lett. 55, 48–51 (1985).   
[9] T. P. Purdy, R. W. Peterson, and C. A. Regal, “Observation of radiation pressure shot noise on a macroscopic object,” Science 339, 801–804 (2013).

[10] Amir H. Safavi-Naeini, Simon Grblacher, Jeff T. Hill, Jasper Chan, Markus Aspelmeyer, and Oskar Painter, “Squeezed light from a silicon micromechanical resonator,” Nature 500, 185–189 (2013).   
[11] T. P. Purdy, P.-L. Yu, R. W. Peterson, N. S. Kampel, and C. A. Regal, “Strong optomechanical squeezing of light,” Phys. Rev. X 3, 031012 (2013).   
[12] Amir H. Safavi-Naeini, Jasper Chan, Jeff T. Hill, Thiago P. Mayer Alegre, Alex Krause, and Oskar Painter, “Observation of quantum motion of a nanomechanical resonator,” Phys. Rev. Lett. 108, 033602 (2012).   
[13] Farid Ya. Khalili, Haixing Miao, Huan Yang, Amir H. Safavi-Naeini, Oskar Painter, and Yanbei Chen, “Quantum back-action in measurements of zero-point mechanical oscillations,” Phys. Rev. A 86, 033840 (2012).   
[14] T. A. Palomaki, J. D. Teufel, R. W. Simmonds, and K. W. Lehnert, “Entangling mechanical motion with microwave fields,” Science 342, 710–713 (2013).   
[15] Joerg Bochmann, Amit Vainsencher, David D. Awschalom, and Andrew N. Cleland, "Nanomechanical coupling between microwave and optical photons," Nature Physics, 712–716 (2013).   
[16] Reed W Andrews, Robert W Peterson, Tom P Purdy, Katarina Cicak, Raymond W Simmonds, Cindy A Regal, and Konrad W Lehnert, “Bidirectional and efficient conversion between microwave and optical light,” Nature Phys. 10, 321–326 (2014).   
[17] A. J. Weinstein, C. U. Lei, E. E. Wollman, J. Suh, A. Metelmann, A. A. Clerk, and K. C. Schwab, “Observation and interpretation of motional sideband asymmetry in a quantum electromechanical device,” Phys. Rev. X 4, 041003 (2014).   
[18] T. P. Purdy, P.-L. Yu, N. S. Kampel, R. W. Peterson, K. Cicak, R. W. Simmonds, and C. A. Regal, “Optomechanical raman-ratio thermometry,” Phys. Rev. A 92, 031802 (2015).   
[19] Donghun Lee, Mitchell Underwood, David Mason, Alexey B. Shkarin, Kjetil Borkje, Steve M. Girvin, and Jack G. E. Harris, “Observation of quantum motion in a nanogram-scale object,” (2014), arXiv:1406.7254.   
[20] Sean M. Meenehan, Justin D. Cohen, Gregory S. MacCabe, Francesco Marsili, Matthew D. Shaw, and Oskar Painter, “Pulsed excitation dynamics of an optomechanical crystal resonator near its quantum ground-state of motion,” (2015), arXiv:1503.05135.   
[21] Justin D. Cohen, Sean M. Meenehan, Gregory S. MacCabe, Simon Groblacher, Amir H.

Safavi-Naeini, Francesco Marsili, Matthew D. Shaw, and Oskar Painter, “Phonon counting and intensity interferometry of a nanomechanical resonator,” Nature 520, 522–525 (2015).   
[22] Markus Aspelmeyer, Tobias J. Kippenberg, and Florian Marquardt, “Cavity optomechanics,” Rev. Mod. Phys. 86, 1391–1452 (2014).   
[23] A. G. Kuhn, J. Teissier, L. Neuhaus, S. Zerkani, E. van Brackel, S. Delglise, T. Briant, P.-F. Cohadon, A. Heidmann, C. Michel, L. Pinard, V. Dolique, R. Flaminio, R. Tabi, C. Chartier, and O. Le Traon, “Free-space cavity optomechanics in a cryogenic environment,” Applied Physics Letters 104, 044102 (2014).   
[24] T. P. Purdy, R. W. Peterson, P.-L. Yu, and C. A. Regal, “Cavity optomechanics with $Si_{3}N_{4}$ membranes at cryogenic temperatures,” New Journal of Physics 14, 115021 (2012).   
[25] A M Jayich, J C Sankey, K Brkje, D Lee, C Yang, M Underwood, L Childress, A Petrenko, S M Girvin, and J G E Harris, “Cryogenic optomechanics with a $Si_{3}N_{4}$ membrane and classical laser noise,” New Journal of Physics 14, 115018 (2012).   
[26] Amir H Safavi-Naeini, Jasper Chan, Jeff T Hill, Simon Grblacher, Haixing Miao, Yanbei Chen, Markus Aspelmeyer, and Oskar Painter, “Laser noise in cavity-optomechanical cooling and thermometry,” New Journal of Physics 15, 035007 (2013).   
[27] Yi Zhao, Dalziel J. Wilson, K.-K. Ni, and H. J. Kimble, “Suppression of extraneous thermal noise in cavity optomechanics,” Opt. Express 20, 3586–3612 (2012).   
[28] P.-L. Yu, K. Cicak, N. S. Kampel, Y. Tsaturyan, T. P. Purdy, R. W. Simmonds, and C. A. Regal, “A phononic bandgap shield for high-q membrane microresonators,” Applied Physics Letters 104, 023510 (2014).   
[29] Seán M. Meenehan, Justin D. Cohen, Simon Gröblacher, Jeff T. Hill, Amir H. Safavi-Naeini, Markus Aspelmeyer, and Oskar Painter, “Silicon optomechanical crystal resonator at millikelvin temperatures,” Phys. Rev. A 90, 011803 (2014).   
[30] E. E. Wollman, C. U. Lei, A. J. Weinstein, J. Suh, A. Kronwald, F. Marquardt, A. A. Clerk, and K. C. Schwab, “Quantum squeezing of motion in a mechanical resonator,” Science 349, 952–955 (2015).   
[31] Nicolas Spethmann, Jonathan Kohler, Sydney Schreppler, Lukas Buchmann, and Dan M. Stamper-Kurn, “Cavity-mediated coupling of mechanical oscillators limited by quantum back-action,” (2015), arXiv:1505.05850.   
[32] Florian Elste, S. M. Girvin, and A. A. Clerk, “Quantum noise interference and backaction

cooling in cavity nanomechanics," Phys. Rev. Lett. 102, 207209 (2009).   
[33] Talitha Weiss and Andreas Nunnenkamp, “Quantum limit of laser cooling in dispersively and dissipatively coupled optomechanical systems,” Phys. Rev. A 88, 023850 (2013).   
[34] Stefano Mancini, David Vitali, and Paolo Tombesi, “Optomechanical cooling of a macroscopic oscillator by homodyne feedback,” Phys. Rev. Lett. 80, 688–691 (1998).   
[35] J.-M. Courty, A. Heidmann, and M. Pinard, “Quantum limits of cold damping with optomechanical coupling,” The European Physical Journal D - Atomic, Molecular, Optical and Plasma Physics 17, 399–408 (2001).   
[36] C. Genes, D. Vitali, P. Tombesi, S. Gigan, and M. Aspelmeyer, “Ground-state cooling of a micromechanical oscillator: Comparing cold damping and cavity-assisted cooling schemes,” Phys. Rev. A 77, 033804 (2008).