# Article

# Yoctonewton force detection based on optically levitated oscillator

Tao Liang<sup>a</sup>, Shaochong Zhu<sup>a</sup>, Peitong He<sup>a</sup>, Zhiming Chen<sup>a</sup>, Yingying Wang<sup>a</sup>, Cuihong Li<sup>a</sup>, Zhenhai Fu<sup>a</sup>, Xiaowen Gao<sup>a</sup>, Xinfan Chen<sup>b</sup>, Nan Li<sup>b</sup>, Qi Zhu<sup>a</sup>, Huizhu Hu<sup>b,a,*</sup>

![](dt=2026-06-01/ht=10/f9ed14141468c62322c7355e41a48e2e5edf1bd127e9f93413751aeea60cea16.jpg)

a Zhejiang Lab, Hangzhou 311121, China

$^{\mathrm{b}}$ State Key Laboratory of Modern Optical Instrumentation, College of Optical Science and Engineering, Zhejiang University, Hangzhou 310027, China

# ARTICLE INFO

Article history:

Received 8 July 2022

Received in revised form 5 September 2022

Accepted 21 September 2022

Available online 14 October 2022

Keywords:

Levitated oscillators

Optical trap

Feedback cooling

Force detection

Allan variance

# ABSTRACT

Optically levitated oscillators in high vacuum have excellent environmental isolation and low mass compared with conventional solid-state sensors, which makes them suitable for ultrasensitive force detection. The force resolution usually scales with the measurement bandwidth, which represents the ultimate detection capability of the system under ideal conditions if sufficient time is provided for measurement. However, considering the stability of a real system, a method based on the Allan variance is more reliable to evaluate the actual force detection performance.

In this study, a levitated optomechanical system with a force detection sensitivity of $6.33 \pm 1.62 \mathrm{zN} / \mathrm{Hz}^{1/2}$ was demonstrated. And for the first time, the Allan variance was introduced to evaluate the system stability due to the force sensitivity fluctuations. The force detection resolution of $166.40 \pm 55.48 \mathrm{yN}$ was reached at the optimal measurement time of $2751 \mathrm{~s}$ . The system demonstrated in this work has the best force detection performance in both sensitivity and resolution that have been reported so far for optically levitated particles.

The reported high-sensitivity force detection system is an excellent candidate for the exploration of new physics such as fifth force searching, high-frequency gravitational waves detection, dark matter research and so on.

# 1. Introduction

Micro- and nanoscale optical levitation oscillator sensors have great potential for the detection of ultra-weak physical quantities [1,2], such as external force [3-8], displacement [9,10], torque [11,12], and acceleration [13-15]. Although the force detection sensitivities of resonant solid-state force sensors, such as dielectric microcantilevers [5] and carbon nanotubes [6], have reached the attonewton and zeptonewton levels, respectively, these devices generally work at low temperatures to reduce the effects of thermal noise.

Compared with mechanically clamped oscillators, micro and nanoparticles levitated in high vacuum (HV) are considered to be low-dissipative optomechanical oscillators because of minimal thermal contact with the environment [16,17]. In addition, levitated oscillators can provide a larger dynamic range than a clamped resonator through closed-loop control, and its oscillation frequency can be modified by adjusting the optical trap.

The low mass and excellent environmental isolation in HV enable such systems to achieve better quality factors and force sensitivity than conventional solid-state sensors, without a cryogenic system [4,7,18]. These advantages allow levitated optomechanical systems to be applicable for the detection of many physical quantities. For example, a charged

particle will respond to electric fields, thus constituting an electric field sensor [4,19]. Similarly, if a particle has magnetic moment, it can be used as a magnetic field sensor [12,20]. In addition, large levitated oscillators are natural sensors of gravity and inertial forces, e.g., accelerometers and gyroscopes [13,14]. Anisotropic particles, such as rods and dumbbells, can rotate and be used for torque sensing [11,12]. Furthermore, polarizable particles can be used to measure photon recoil [21] because they respond to optical forces (radiation pressure).

In addition, levitated optomechanical systems provide an excellent platform for exploring macroscopic quantum mechanics, such as macroscopically separated superposition states [22], matter-wave interferometry [23], fifth force searching [24,25], high-frequency gravitational wave detection [26,27], and the Schrödinger-Newton equation [28].

The force detection sensitivity and resolution of levitated optomechanical systems have notably improved over the past few years. Their force detection sensitivity is usually evaluated based on the thermal noise limit of the system. Rider et al. [29] captured silica microspheres near an oscillating Au-coated cantilever with a sensitivity of 20 $\mathrm{aN / Hz^{1 / 2}}$ . Gieseler et al. [18] used a silica nanoparticle under feedbackcooling to detect a periodic optical force gradient, which is induced by low-frequency modulation of the trapping potential, with a force detection sensitivity of $20~\mathrm{zN / Hz^{1 / 2}}$ . Magrini et al. [30] reported a sensitiv-

Fundamental Research 3 (2023) 57-62

国自科学基

Contents lists available at ScienceDirect

Fundamental Research

journal homepage: http://www.kaipublishing.com/en/journals/fundamental-research/

Fundamental Fundamental Research

* Corresponding author.

E-mail address: huhuizhu2000@zju.edu.cn (H. Hu).

https://doi.org/10.1016/j.fmre.2022.09.021

2667-3258/© 2022 The Authors. Publishing Services by Elsevier B.V. on behalf of KeAi Communications Co. Ltd. This is an open access article under the CC

BY-NC-ND license (http://creativecommons.org/licenses/by-nc-nd/4.0/)

ity of approximately $10\mathrm{zN / Hz}^{1 / 2}$ for near-field coupling to a photonic crystal cavity. Geraci et al. [31] used antinodes of an optical cavity field to trap and cool microparticles and concluded that such a system could reach yN-level force sensitivity.

The minimum detectable force under ideal conditions usually scales down with measurement time, which is also called the force resolution. In 2015, Geraci et al. [3] achieved a force sensitivity and resolution of $217\mathrm{aN / Hz}^{1 / 2}$ and $2\mathrm{aN}$ respectively, on a $10\mathrm{h}$ timescale using a dualbeam optical trap and active feedback cooling.

The following year, they [4] used the spacing of the optical lattice to calibrate the displacement of optically levitated nanoparticles with active cooling, achieving a sensitivity and resolution of $1.63\mathrm{aN / Hz}^{1 / 2}$ and $5.8~\mathrm{zN}$ respectively, on a time scale of $10^{5}\mathrm{s}$ . Hebestreit et al. [7] reported a static force detection method that achieved gravitational and electrostatic force resolutions at the $10\mathrm{aN}$ level by measuring the displacement of particles in free-fall after being released from an optical trap. Hempston et al.

[8] applied an alternating electric field to a charged levitated particle, achieving a sensitivity and resolution of $32\mathrm{zN / Hz}^{1 / 2}$ and $30~\mathrm{zN}$ respectively, with a $10\mathrm{s}$ integration time. In 2020, Dadras et al. [32] reported an injection locking phenomenon using a levitated optomechanical device and achieved a sensitivity of $23~\mathrm{zN / Hz}^{1 / 2}$ and resolution of $1\mathrm{zN}$ with an average time of $300\mathrm{s}$ . The enhancement of force detection resolution due to a longer integration time is evident.

However, these results only hold under ideal measurement conditions.

Based on our study on levitated optomechanical systems [33], we proposed a fast size estimation method for levitated particles [34] and investigated the particle capture region shrinkage and levitation instability properties of optical traps in
vacuum [35]. In addition, we proposed a method to directly detect the force field gradient using a pair of levitated nanospheres [36]. Recently, the electric field force was used for the calibration of the force detection sensitivity, and a sensitivity of $43.9\mathrm{zN / Hz^{1 / 2}}$ was achieved [37].

In this study, to evaluate the actual force detection resolution of levitated optomechanical systems, we propose a new method based on the Allan variance [38]. The Allan variance is used to estimate the measurement stability due to inevitable noise, which must be considered in practical measurements. It can effectively describe the fluctuation level (instability) of the target time sequence error on different time scales. A silica nanosphere was stably levitated in ultrahigh vacuum (UHV) by applying parametric feedback cooling.

The center of mass (CoM) of the y-axis was cooled to $1.98 \pm 0.28 \mathrm{mK}$ at $7.9 \times 10^{-9} \mathrm{mbar}$ , resulting in a sensitivity of $6.33 \pm 1.62 \mathrm{zN} / \mathrm{Hz}^{1/2}$ , and achieving the optimal sensitivity of $4.34 \mathrm{zN} / \mathrm{Hz}^{1/2}$ . The system allows long integration time measurements to reduce thermal noise, and the fluctuations in force sensitivity are obtained via the Allan variance, calculated on different timescales considering of system stability.

The time scale that reaches maximum stability is chosen as the optimal measurement time for resolution calculation. The system in this work achieved a force detection resolution of $166.40 \pm 55.48 \mathrm{yN}$ at an optimal measurement time of $2751 \mathrm{~s}$ .

# 2. Methods and experiment

# 2.1. Methods

# 2.1.1. Harmonic motion of particle with feedback cooling

For a particle in a harmonic potential with a trap stiffness of $k_{0} = m\Omega_{0}^{2}$ and experiencing a damping force (damping rate $\Gamma_0$ ), the equation of motion along the $x$ -axis can be written as

$$
m \ddot {x} (t) + m \Gamma_ {0} \dot {x} (t) + m \Omega_ {0} ^ {2} x (t) = F _ {t h} (t) + F _ {o p t} (t) \tag {1}
$$

where $x(t)$ is the particle position in the $\mathbf{x}$ direction, and $\mathrm{m}$ is the mass of the particle. $F_{th}(t)$ is an external noise according to the thermal stochastic noise that satisfies $\langle F_{th}(t)F_{th}(t^{\prime})\rangle = 2m\Gamma_0k_BT\delta (t - t^{\prime})$ , according to the fluctuation-dissipation theorem, and $F_{opt}$ is the optical force of the trap that causes the $\delta \Gamma$ and $\delta \Omega$ shifts in the damping rate $\Gamma_0$ and oscillation frequency $\Omega_0$ , respectively, when feedback cooling is applied. Similar

equations and considerations are applied in the $y$ and $z$ directions of particle motion.

The activation of parametric feedback leads to the following motion power spectral density (PSD) $S_{x}(\Omega)$ :

$$
\begin{array}{l} S _ {x} (\Omega) = \int_ {- \infty} ^ {\infty} \langle x (t) x \left(t - t ^ {\prime}\right) \rangle e ^ {- i \Omega t ^ {\prime}} d t ^ {\prime} \\ = \frac {k _ {B} T}{\pi m} \frac {\Gamma_ {0}}{\left(\left[ \Omega_ {0} + \delta \Omega \right] ^ {2} - \Omega^ {2}\right) ^ {2} + \Omega^ {2} \left[ \Gamma_ {0} + \delta \Gamma \right] ^ {2}} \tag {2} \\ \end{array}
$$

where $k_{B}$ is Boltzmann's constant and $T$ is the equilibrium temperature in Kelvin, $\delta \Omega$ and $\delta \Gamma$ are the frequency and damping rate shifts induced by feedback cooling, respectively.

The PSD is a useful tool for analyzing the motion of optically levitated particles and obtaining some important parameters, such as the CoM temperature, damping rate, and oscillation frequency. When the pressure of the gas and damping rate are known, according to kinetic theory, the particle radius $r$ can be obtained as [39]

$$
r = 0. 6 1 9 \frac {9 \pi}{\sqrt {2}} \frac {\eta d _ {g a s} ^ {2}}{\rho k _ {b} T _ {0}} \frac {P _ {g a s}}{\Gamma_ {0}} \tag {3}
$$

where $\eta$ is the viscosity of air, $d_{gas}$ is the diameter of a gas particle, $\rho$ is the density of a levitated particle, $T_0$ is the temperature of the environment, and $P_{gas}$ is the pressure of the gas.

# 2.1.2. Force detection sensitivity and resolution

The performance of force detection is primarily limited by the thermal Langevin force, with the PSD given by $S_{F} = 4k_{B}T_{0}m\Omega_{0} / Q$ [40] according to the fluctuation-dissipation theorem. The $S_F^{1 / 2}$ serves as the force detection sensitivity, $k_{B}$ is Boltzmann's constant, and $Q = \Omega_0 / \Gamma_0$ is the quality factor. The minimum detectable force for a harmonic oscillator [4] at temperature $T$ is

$$
F _ {\text {m i n}} = \sqrt {S _ {F} \cdot b} = \sqrt {4 k _ {B} T _ {0} m \Gamma_ {0} b} \tag {4}
$$

where $b$ denotes the measurement bandwidth. For a nanoparticle oscillator with feedback cooling, the $T_{0}$ and $\Gamma_0$ in Eq. 4 become $T_{\mathrm{eff}}$ and $\Gamma_{\mathrm{eff}}$ to include the effect of cooling, where $T_{\mathrm{eff}} = T_0 + \delta T$ and $\Gamma_{\mathrm{eff}} = \Gamma_0 + \delta \Gamma$ , respectively. The minimum detectable force serves as the force detection resolution of the system.

# 2.2. Experimental setup

We demonstrated force sensing using the experimental setup shown in Fig. 1. According to the previously proposed fast size estimation method [34], a fused-silica nanosphere (monodisperse silica nanoparticles with nominal diameter of $150~\mathrm{nm}$ , Nanocym) was trapped in a vacuum chamber using a single $1064~\mathrm{nm}$ laser beam (YFL-SF-1064-10CW, Precilasers). The beam was focused with an objective of numerical aperture $(\mathrm{NA}) = 0.8$ (TU plan ELWD, Nikon) and the trap was initially operated with a total power of $230~\mathrm{mW}$ .

The nanoparticle was loaded into the trap by atomization at 1 atm. Light scattered from the nanoparticle was collimated by an aspheric lens and collected by a self-developed quadrant photodetector (QPD). The motion signal on all three axes can then be obtained using a displacement detection decoupling scheme [41]. The detector consisted of one InGaAs PIN photodiode (G8370-82, Hamamatsu Photonics) and one quadrant-type InGaAs PIN photodiode array (G6849, Hamamatsu Photonics).

We define the "axial" or the z-axis in the direction of the trapping beam, whereas the x-axis and y-axis correspond to the directions parallel and perpendicular to the polarization of the trapping beam in the focal plane. The x and y motion signals were obtained by subtracting the power of every two quadrants, and the z motion signal was evaluated by calculating the total power difference between the two photodiodes. The motion signals of the three directions from the QPD were connected to three lock-in amplifiers (MFLI, Zurich

T. Liang, S. Zhu, P. He et al.

Fundamental Research 3 (2023) 57-62

58

![](dt=2026-06-01/ht=10/571102973c7017726385e4567ff4b91f529a8c473dcd2fc11f16ceef83de6b90.jpg)

Instruments) respectively, and the corresponding feedback signals were generated and added together, before being sent to the acousto-optic modulator (AOM).

To achieve UHV, the particle must be cooled for stable levitation. Here, the parametric feedback scheme [16] was applied to cool the particle's CoM motion. The basic concept of the feedback is to introduce modulation of laser intensity in the time domain at the focus. Such modulation is specifically built using the information of the particle's position $x$ and velocity $\dot{x}$ to create an effective stiffening and softening spring constant with respect to the particle's oscillation.

Intuitively, the optical trap must be stiffened when the particle is moving away from the equilibrium position $x_0$ and its kinetic energy is being converted into potential energy. On the contrary, when the particle is falling back to equilibrium potential and the potential energy is converted into kinetic energy, the trap must be softened. Note that in

th
e frequency domain, this is equivalent to a modulation at twice the frequency of the particle oscillation. In addition, owing to the latency of the feedback loop, phase shifting is necessary to ensure the correct phase of modulation. By modifying the phase of the feedback signal, enhancement and suppression of the particle's oscillation amplitude can be achieved separately, usually referred to as "heating" and "cooling" respectively.

Since the three directions are spectrally separated and have different frequencies, there is no cross-coupling between the three signals. Therefore, adding all the feedback signals to drive the AOM that modulates the power of the trapping laser is possible. In other words, a single beam can be used to effectively cool all three degrees of freedom. The conversion factor from voltage to displacement (unit: V/m) and particle radius $r$ can be extracted by fitting to the PSD under thermal equilibrium at 10 mbar. A proportional-integral-derivative (PID) controller was used to adjust the frequency of the feedback signal, and the cooling effect was achieved by frequency doubling and phase shifting.

# 3. Results and discussion

# 3.1. Feedback cooling of CoM temperature

We performed force measurements in the y direction with a sampling rate of $937.5\mathrm{kHz}$ . Fig. 2a shows the y-axis motion spectrum $S_{xx}^{1/2}$ of a diameter $159.87 \pm 5.44$ nm particle under feedback cooling, at vacuum pressures of $5 \times 10^{-3}$ , $9 \times 10^{-7}$ , and $4.2 \times 10^{-8}$ mbar. The PSD with no feedback cooling at $5 \times 10^{-3}$ mbar is also shown. We observed a resonant frequency of $193.8\mathrm{kHz}$ in the natural state. Additionally, a peak shift appeared when feedback cooling was applied.

This is due to the fact that the phase is not precisely $90^{\circ}$ and is usually called optical spring effect [3]. The CoM temperature in the y direction was cooled to $1.93 \pm 0.15\mathrm{mK}$ with a damping rate of $16.25 \pm 1.12\mathrm{Hz}$ at $4.2 \times 10^{-8}$ mbar, and the corresponding force sensitivity $S_F^{1/2}$ was $6.81 \pm 0.50\mathrm{zN/Hz^{1/2}}$ . Then a higher vacuum of $7.9 \times 10^{-9}$ mbar was reached with another nanoparticle of diameter $142.60 \pm 5.54\mathrm{nm}$ .

Five sensitivity measurements were performed at $7.9 \times 10^{-9}$ mbar and the $S_{xx}^{1/2}$ in y direction is shown in Fig. 2b. The CoM temperature could be $1.98 \pm 0.28\mathrm{mK}$ with a damping rate of $19.60 \pm 6.54\mathrm{Hz}$ and the corresponding $S_F^{1/2}$ was $6.33 \pm 1.62\mathrm{zN/Hz^{1/2}}$ . The optimal sensitivity of the five measurements is $4.34\mathrm{zN/Hz^{1/2}}$ , which is the best force sensitivity reported yet for optically levitated particles. The detailed data of the five measurements are provided in Supplementary Material.

The lowest cooling temperature was limited by environmental noise and trapping

![](dt=2026-06-01/ht=10/1ceda7fe94a8ef348b642ed9d8329c693ffed19a36b0b62c5544db361032a9e7.jpg)

T. Liang, S. Zhu, P. He et al.

Fundamental Research 3 (2023) 57-62

59

Table 1 CoM temperature $T_{\mathrm{eff}}$ damping rate $\Gamma_{\mathrm{eff}}$ and force detection sensitivity $S_F^{1 / 2}$ of levitated optomechanical systems.

![](dt=2026-06-01/ht=10/62eb46eac1ada9dc30dd5bfa6a2750716357885934d53d9e040bfafc7631d119.jpg)

<table><tr><td>Group</td><td>Vacuum (mbar)</td><td>Teff(mK)</td><td>Γeff/2π(Hz)</td><td>SF1/2(N/Hz1/2)</td><td>Year</td></tr><tr><td rowspan="2">UN Reno</td><td>6.67 × 10-6</td><td>(1 ± 0.3) × 104</td><td>454 ± 29</td><td>(2.17 ± 0.48) × 10-16</td><td>2015 [3]</td></tr><tr><td>6.67 × 10-6</td><td>460 ± 60</td><td>460 ± 49</td><td>(1.63 ± 0.37) × 10-18</td><td>2016 [4]</td></tr><tr><td>Southampton</td><td>1.6 × 10-5</td><td>3 × 103</td><td>\</td><td>3.2 × 10-20</td><td>2017 [8]</td></tr><tr><td>Stanford</td><td>~ 10-6</td><td>\</td><td>\</td><td>1 × 10-17</td><td>2020 [42]</td></tr><tr><td>Vienna</td><td>~ 10-6</td><td>(1.22 ± 0.05) × 10-2</td><td>(2.06 ± 0.23) × 104</td><td>1.71 × 10-20</td><td>2020 [2,43]</td></tr><tr><td>Yale</td><td>~ 10-7</td><td>(5 ± 2.2) × 10-2</td><td>\</td><td>(9.5 ± 1.1) × 10-19</td><td>2020 [13]</td></tr><tr><td rowspan="2">ETH Zurich</td><td>1.4 × 10-8</td><td>0.1</td><td>1 × 103</td><td>1.00 × 10-20</td><td>2019 [2,44]</td></tr><tr><td>7.5 × 10-9</td><td>9.6 × 10-3</td><td>4 × 103</td><td>8.00 × 10-21</td><td>2020 [2,45]</td></tr><tr><td rowspan="5">Zhejiang Lab &amp; Zhejiang University</td><td>5 × 10-3</td><td>(22.76 ± 4.79) × 103</td><td>201.51 ± 75.88</td><td>(2.54 ± 0.77) × 10-18</td><td>2022</td></tr><tr><td>9 × 10-7</td><td>4.38 ± 1.34</td><td>109.19 ± 32.51</td><td>(2.59 ± 0.78) × 10-20</td><td rowspan="3">[This work]</td></tr><tr><td>4.2 × 10-8</td><td>1.93 ± 0.15</td><td>16.25 ± 1.12</td><td>(6.81 ± 0.50) × 10-21</td></tr><tr><td>7.9 × 10-9</td><td>1.98 ± 0.28</td><td>19.60 ± 6.54</td><td>(6.33 ± 1.62) × 10-21</td></tr><tr><td>7.9 × 10-9</td><td>1.72</td><td>10.4</td><td>4.34 × 10-21</td><td>Optimal</td></tr></table>

![](dt=2026-06-01/ht=10/06f2a0a8a2858e4d36251a1ca67680247994fee9c9d047db790de10e61aafd6d.jpg)

laser noise. Here we summarized the force detection sensitivities and related parameters of various levitated optomechanical systems [2], which are listed in Table 1 together with this work. A few of force sensitivities were not directly reported, and instead were estimated from the reported effective temperature $T_{\mathrm{eff}}$ and damping rate $\Gamma_{\mathrm{eff}}$ .

# 3.2. Force resolution measurements

In the absence of an external force, the estimated thermal noise of the oscillation averages down as $b^{-1}$ . This behavior is executed in the same way as the Geraci group [3,4] for an integration time exceeding 36,000 s at $7.9 \times 10^{-9}$ mbar, as shown in Fig. 3. This method can obtain the ultimate detection capability of the system under an ideal state without environmental interference and stability fluctuation. The minimum detectable force $F_{\min}$ is represented by the solid orange line, as calculated by Eq. 4 using the measured parameters for $T_{\mathrm{eff}}$ , and $\Gamma_{\mathrm{eff}}$ . $F_{\min,\text{best}} = 40.80 \pm 8.55$ yN was achieved at the maximum time scale; to the best of our knowledge, this is the best force detection resolution ever reported.

Although a better force detection resolution can be obtained by extending the integration time, system stability gradually deteriorates and becomes the primary limiting factor for system performance. Hence, we have proposed a more suitable method for evaluating the force sensitivity based on the Allan variance [38] to obtain the actual force detection

![](dt=2026-06-01/ht=10/0af6bb0f0d70db95cc9b5b62687d4262d7338f42fef0200003689cd601ee021d.jpg)

resolution. The Allan variance can be used to evaluate the stability of the system and has been widely applied in investigating clock frequency stability [46,47], inertial sensing [48,49] and noise quantification [50]. The target data sequence was divided into several segments according to a specific length $T_{seg}$ , and the mean values of the force detection resolution of each segment were calculated separately. The difference between the means of adjacent segments was then calculated, and finally, the mean-square value of these differences was obtained. This method can erase both fast-changing $(< T_{seg})$ and slow-changing $(> 2T_{seg})$ components, yielding the error fluctuation in a narrow time scale range between $T_{seg}$ and $2T_{seg}$ .

If measurement is required on various time scales, the segment length $T_{seg}$ can be traversed from short to long in order to obtain a set val
ue of the Allan variance. Then the Allan deviation vs. $T_{seg}$ curve can be obtained to fully reflect the characteristics of error fluctuation. As shown in Fig. 4, the Allan deviation of the force detection sensitivity was calculated over a series of time scales, reflecting the stability of the system.

Since the thermal noise reduces with the measurement time, the deviation decreases with integration time and reaches a minimum value of $24.49\mathrm{yN / Hz^{1 / 2}}$ at approximately $t_{stable} = 2751\mathrm{s}$ (indicated by the red dashed line). The main factor limiting the stability may be the vibration of the environment. Subsequently, as time increases, the system stability deteriorates, resulting in a larger deviation. This means that the optimal measurement time scale is $t_{stable}$ , where the system performance is the most stable. The corresponding force resolution

T. Liang, S. Zhu, P. He et al.

Fundamental Research 3 (2023) 57-62

60

at time scale $t_{\mathrm{stable}}$ was approximately $F_{\mathrm{min,stable}} = 166.40 \pm 55.48 \mathrm{yN}$ . Compared with $F_{\mathrm{min,best}}$ obtained at the maximum measurement time, $F_{\mathrm{min,stable}}$ is considered to be the actual force detection resolution. Moreover, $F_{\mathrm{min,stable}} = 166.40 \pm 55.48 \mathrm{yN}$ is also the best force detection resolution reported yet for optically levitated particles.

As indicated by the aforementioned results, the thermal noise can be averaged down as $b^{-1}$ ; hence, the force detection resolution improves with a longer integration time. Consistent with other studies [3,4,32], the force resolution $F_{\mathrm{min}}$ of this work decreased linearly with time and attained the best force resolution for optically levitated particles reported to date: $F_{\mathrm{min\_best}} = 40.80 \pm 8.55 \mathrm{yN}$ at the maximum measurement time of 2751 s. However, maintaining stability for such a long time is difficult for the system.

Thus, the Allan variance was applied to evaluate the system stability and obtain the actual force detection performance. The Allan deviation of the force detection sensitivity reached a minimum at the optimal measurement time $t_{\mathrm{stable}}$ , implying that the system stability could not be improved thereafter. Therefore, the time scale $t_{\mathrm{stable}}$ was used to obtain the actual force detection resolution, which was approximately $F_{\mathrm{min\_stable}} = 166.40 \pm 55.48 \mathrm{yN}$ .

# 4. Conclusion

In this work, a yoctonewton force detection resolution was achieved with an optically levitated oscillator under UHV conditions. On the y-axis, a CoM temperature of $1.98 \pm 0.28 \mathrm{mK}$ at $7.9 \times 10^{-9} \mathrm{mbar}$ was achieved by applying parametric feedback cooling, with a sensitivity of $6.33 \pm 1.62 \mathrm{zN} / \mathrm{Hz}^{1/2}$ . Five measurements of force sensitivity were performed, where the optimal result is $4.34 \mathrm{zN} / \mathrm{Hz}^{1/2}$ .

Then, a force detection resolution of $40.80 \pm 8.55 \mathrm{yN}$ was achieved using the reported method [3,4], with an integration time exceeding $36,000 \mathrm{s}$ . Moreover, the Allan deviation was introduced to characterize the system stability at different time scales by evaluating the force sensitivity fluctuations. The results showed that the optimal measurement time was approximately $2751 \mathrm{s}$ , and the corresponding force detection resolution was approximately $166.40 \pm 55.48 \mathrm{yN}$ .

The stability of system may be limited by the environment vibration, but the magnitude and source of the interference need to be further analyzed. Up to date, these are the best force sensitivity and resolution obtained within the thermal noise limit for optically levitated particles. Furthermore, this is the first time that the Allan variance has been employed for force detection in such system; it has been proven to be a suitable method to evaluate the system stability to further obtain the actual force resolution.

In addition, utilizing more efficient feedback cooling schemes and trapping smaller particles can lead to better force detection performance. An optically levitated oscillator system with such force detection performance will find more exploration of new physics.

# Declaration of competing interest

The authors declare that they have no conflicts of interest in this work.

# Acknowledgments

This work was supported by grants from the National Natural Science Foundation of China (62005248, 62075193), Major Project of Natural Science Foundation of Zhejiang Province (LD22F050002), Major Scientific Research Project of Zhejiang Lab (2019MB0AD01, 2021MB0AL02, 2022MB0AL02), the Fundamental Research Funds for the Central Universities, China (2016XZZX00401 and 2018FZA5002), and the National Program for Special Support of Top-Notch Young Professionals (W02070390), China.

# Supplementary materials

Supplementary material associated with this article can be found, in the online version, at doi:10.1016/j.fmre.2022.09.021.

# References

T. Liang, S. Zhu, P. He et al.

Fundamental Research 3 (2023) 57-62

61

[50] F. Czerwinski, A.C. Richardson, L.B. Oddershede, Quantifying noise in optical tweezers by Allan variance, Opt. Express 17 (15) (2009) 13255-13269.

![](dt=2026-06-01/ht=10/4723fa43a33cc99608634553aed335a02b9b9aa1072802d9d1bbc42c99824818.jpg)

Tao Liang is an assistant researcher of Research Center for Quantum Sensing of Zhejiang Lab. He received Ph.D. degree of biomedical engineering in Zhejiang University in 2021. His major research directions are feedback cooling and ultraweak force sensing in levitated optomechanical systems.

![](dt=2026-06-01/ht=10/4b897eeba3e6efbacaae68c7445ae759c04b32d278f68c50a1269f3e71316a55.jpg)

Huizhu Hu is a Qiushi distinguished professor of Zhejiang University, PI of Zhejiang Lab, and director of Fundamental Science on Optical Inertial and Sensing Technology Laboratory. He got his bachelor of science from Xi'an Jiaotong University in 1999, and received his doctor of philosophy from Zhejiang University in 2004. Then, he joined the current University as a faculty and was promoted as a professor in 2012. His current research interests are in optical sensing and precision measurement.

T. Liang, S. Zhu, P. He et al.

Fundamental Research 3 (2023) 57-62

62