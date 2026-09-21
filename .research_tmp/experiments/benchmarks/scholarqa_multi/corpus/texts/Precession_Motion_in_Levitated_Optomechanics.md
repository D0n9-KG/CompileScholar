# Precession Motion in Levitated Optomechanics

Muddassar Rashid, $^{1,*}$ Marko Toroš, $^{1,\dagger}$ Ashley Setter, $^{1}$ and Hendrik Ulbricht $^{1,\ddagger}$

$^{1}$ Department of Physics and Astronomy, University of Southampton, SO17 1BJ, UK

We investigate experimentally the dynamics of a non-spherical levitated nanoparticle in vacuum. In addition to translation and rotation motion, we observe the light torque-induced precession and nutation of the trapped particle. We provide a theoretical model, which we numerically simulate and from which we derive approximate expressions for the motional frequencies. Both, the simulation and approximate expressions, we find in good agreement with experiments. We measure a torque of $1.9 \pm 0.5 \times 10^{-23}$ Nm at $1 \times 10^{-1}$ mbar, with an estimated torque sensitivity of $3.6 \pm 1.1 \times 10^{-31}$ Nm/ $\sqrt{Hz}$ at $1 \times 10^{-7}$ mbar.

Introduction- A typically levitated optomechanical setup comprises of a particle, which is trapped using a tightly focussed laser beam. The trapped particle is best described as a harmonic oscillator and can exhibit a rich variety of dynamics. For example, the dynamics can be under-damped or over-damped depending on the number of collisions controlled by the background gas pressure. The dynamics can be linear or can show strong non-linearities depending on the oscillation amplitude, which is controlled by temporal and spatial modulated external electric, magnetic, and light forces: the dynamics can be driven or cooled. This tunability of the dynamics has enabled various studies of the rich physics of this harmonic oscillator, among them Brownian motion [1], nonlinear dynamics [2, 3], test of fluctuation theorems and non-equilibrium physics [4–6], and thermodynamics in the single particle regime [7–9]. The unique properties of the underlying dynamics and the ability to control levitated optomechanics demonstrate its immense potential for sensing [10–12], as well as, to address fundamental questions in physics [13–15].

The centre-of-mass (c.o.m.) translation motion of the trapped nanoparticle has been studied in quite some detail already. Optical feedback, cavity-assisted schemes and electric forces have been used to control the translation and have been ultimately used to cool the motion to millikelvin temperatures $[16–19]$ and below $[20]$ , already close to the ground state of the harmonic oscillator.

In addition, to exhibiting translation motion, trapped particles can also show rotation $[21–23]$ and libration $[24]$ motion. Different light-matter physics has been used to drive and control the rotation of levitated particles, such as the polarizability-anisotropy in silicon rods coupling to the polarization of light $[25]$ . Rotational frequencies of up to some GHz $[26, 27]$ have been observed, only limited by the centrifugal damage threshold of the rotating particle.

Variation of the linear to circular polarization of light gives a handle to switch between rotation and libration. Libration has been demonstrated experimentally with trapped nanodiamonds $[24]$ , silicon rods $[25]$ and dumbbells $[27]$ . Like translation, libration motion is described by a harmonic oscillator model and is therefore a candidate to apply similar optical techniques for cooling with reasonable promise to reach a quantum ground state, making libration a stark contender in the race towards the quantum regime. First proposals discuss the usefulness of libration to generate macroscopic quantum states such as angular superpositions $[28, 29]$ . Additionally, both rotation and libration motion promise unprecedented high levels of sensitivity $[24, 30, 31]$ for detection of weak forces such as gravity $[28, 32]$ and dispersive forces $[33]$ .

In this letter, we report on the observation of light-induced precession motion of a non-spherical silica particle compound. We give a theoretical description of the system and numerically simulate the model. We identify the mechanical frequencies in the experimental spectrum: in particular, translation, rotation (spin), precession, and nutation motion. We investigate the precession by variation of background gas pressure and of the power of the trapping laser in agreement with the theoretical model. We discuss the possibilities for torque sensing applications.

Theoretical model- We consider an anisotropic polarizable particle, which is optically trapped by an elliptically polarized Gaussian laser beam. Part of the scattered photons are collected and directed, using optical elements, towards a single photodetector. The scattered light is mixed with a local oscillator to obtain the direct homodyne photo-current $I_{\mathrm{exp}}$ , see Fig. 1(a) [34]. This experimental situation has been analyzed theoretically using a quantum model [35], as well as numerically, using an approximate classical model [36]. We now summarize the dynamics referring to the latter, where we assume that we are at relatively high pressure, such that we can neglect photon-recoil heating terms.

The dynamics can be described in a twelve dimensional phase space of the following classical variables: center of mass position $\boldsymbol{r} = (x, y, z)^{\top}$ , center of mass conjugate momentum $\boldsymbol{p} = (p_x, p_y, p_z)^{\top}$ , angle $\boldsymbol{\phi} = (\alpha, \beta, \gamma)^{\top}$ , and angle conjugate momentum $\boldsymbol{\pi} = (\pi_{\alpha}, \pi_{\beta}, \pi_{\gamma})^{\top}$ , where the angles are defined in the Euler $z-y'-z''$ convention (see Fig. 1(b)). In particular, the dynamics is given by (in

(a)   
![](images/474624be8b20a1058256881832709c1151512f1eb7fc149cfd1e9e9025f526f0.jpg)

<details>
<summary>text_image</summary>

Paraboloidal
Mirror
λ/4
λ/2
1550 nm
Fiber Laser
Ediv + Escat fast
photodetector
</details>

(b)   
![](images/671ba8f5ca00a09f53fe87cc6824490d1fc8bc520e18afd6ddd6cbf635068737.jpg)

<details>
<summary>text_image</summary>

ωαprecession
x
y
x''
y''
β
γ
z''
z
α
</details>

![](images/b7db6751eb223c7fafb02455ca1ad2d6c1b7c1eee0140a462bec47190b2c75c9.jpg)

<details>
<summary>natural_image</summary>

Microscopic image of dark spherical particles with a 100 nm scale bar, no text or symbols present
</details>

Figure 1. The Levitated Optomechanical System: (a) A 1550 nm laser beam is focused by the paraboloidal mirror, the particle is trapped in the focus. Once trapped, the scattered light, $E_{scat}$ , from the particle is collected and directed towards the detection system. The interference between the scattered and the diverging electric fields, $E_{div}$ , is used to detect the motion of the particle. (b) Paremetrization: A non-spherical particle that is trapped by a laser field propagating in the z-direction. The laboratory axis are denoted by x, y, z, while the body-frame axis are denoted by the $x''$ , $y''$ , $z''$ . The relation between the two frames is parametrized by the Euler angles $\alpha$ , $\beta$ and $\gamma$ in $z-y'-z''$ convention. $\alpha$ denotes the angle of rotation about the laboratory z-axis (from x towards y). $\beta$ is the angle between the laboratory z-axis and the body $z''$ -axis (rotated about the $y'$ -axis, i.e. the y-axis after it has been rotated by $\alpha$ about the z-axis; from z towards x). $\gamma$ denotes the angle of rotation about the body frame $z''$ -axis (from $x''$ towards $y''$ ). (c) TEM image of compound silica particle solution in water, few weeks after preparation. The original particle solution is made of 50 nm radius silica spheres (Corpuscular Inc.).

Itô form):

$$
d \boldsymbol {r} = - \partial_ {\boldsymbol {p}} H _ {\text { free }} d t \tag {1}
$$

$$
d \boldsymbol {p} = - \partial_ {\boldsymbol {r}} H _ {\text { grad }} d t + d \boldsymbol {p} _ {\text { scatt }} + d \boldsymbol {p} _ {\text { coll }} \tag {2}
$$

$$
d \phi = \partial_ {\pi} (H _ {\text { free }} + H _ {\text { grad }}) d t \tag {3}
$$

$$
d \boldsymbol {\pi} = - \partial_ {\phi} (H _ {\text {free}} + H _ {\text {grad}}) d t + d \boldsymbol {\pi} _ {\text {scatt}} + d \boldsymbol {\pi} _ {\text {coll}}, \tag {4}
$$

where $H_{free}$ and $H_{grad}$ denote the free Hamiltonian and the gradient potential, respectively, $dp_{scatt}$ , $d\pi_{scatt}$ denote the non-conservative terms induced by photon scattering, and $dp_{coll}$ , $d\pi_{coll}$ the non-conservative terms, which arise from gas collisions. Specifically, $dp_{scatt}$ corresponds to the radiation pressure scattering force, which displaces the particle along the positive z direction, and $dp_{coll}$ denotes the terms, which tend to thermalize the center of mass motion to the temperature T of the gas of particles. Similarly, $d\pi_{scatt}$ denotes the terms, which quantify the transfer of angular momentum from the photons to the particle, i.e. the driving terms, and $d\pi_{coll}$ denotes the terms that tend to thermalize rotations to the temperature of the gas, i.e. the friction and diffusive terms (see [37]).

We consider a specific experimental situation, where we illustrate the physical content of Eqs. (1)-(4), and we obtain approximate expression for the dominant mechanical frequencies (further details can be found elsewhere [37]). Specifically, we consider the experimental situation that produces the power spectral density in Fig. 2, where the rotational frequencies are significantly higher or lower than the translational ones. To obtain the dominant mechanical frequencies, we can in first approximation treat translation and rotation as decoupled motion.

We start by looking at translational degrees of freedom. We suppose that $|\frac{\mathbf{r}}{\lambda}| \ll 1$ , where $\lambda$ is the laser wavelength, which limits translations to harmonic oscillations. In particular, the frequencies for the x, y, z motion are given by:

$$
\omega_ {x} ^ {2} = \frac {2 P a _ {1} \chi_ {0}}{c \sigma_ {L} w _ {0} ^ {2} \rho}, \quad \omega_ {y} ^ {2} = \frac {2 P a _ {2} \chi_ {0}}{c \sigma_ {L} w _ {0} ^ {2} \rho}, \quad \omega_ {z} ^ {2} = \frac {2 P \chi_ {0}}{c \sigma_ {L} \rho z _ {R} ^ {2}}, \tag {5}
$$

respectively, where P is the laser power, $\sigma_{L} = \pi w_{0}^{2}$ is the effective laser beam cross section area, $w_{0}$ is the mean beam waist radius, $a_{1}, a_{2}$ quantify the asymmetry of the beam along the x, y directions, respectively, $z_{R}$ is the Rayleigh length, $\rho$ is the particle density, $\chi_{0} = \frac{1}{3} \sum_{i=1}^{3} \chi_{i}$ is an effective susceptibility of the particle, and c is the speed of light. These frequencies are obtained directly from $H_{\mathrm{grad}}$ by expanding to order $\mathcal{O}((\frac{\mathbf{r}}{\lambda})^{2})$ .

The rotational frequencies arise from (i) the transfer of angular momentum during photon scattering, and from (ii) the gradient torque. On the one hand, the scattering torque drives the system into a fast spinning motion, while, on the other hand, the gradient torque would like to align the system with the polarization of the incoming beam in such a way to minimize the electric dipole potential energy, resulting in nutation and precession. Before deriving the rotational frequencies mathematically we now first give an intuitive picture of the two mechanisms.

The mechanism (i) can be understood in terms of the angular momentum carried by the incoming light beam (in a particle picture one can think of an individual photon carrying a small amount of angular momentum, e.g. $\hbar$ for circular polarization). During scattering the angular momentum is transferred to the nanoparticle, where the amount that is transferred depends on the susceptibility anisotropy and orientation of the nanoparticle. As a consequence, the particle starts to spin, until an

asymptotic rotational frequency is reached, which is constrained by friction due to gas collisions. In particular, we consider the experimental situation, where the photon scattering gives rise to high spinning frequencies $\omega_{\alpha}^{(\mathrm{spin})}$ and $\omega_{\gamma}^{(\mathrm{spin})}$ about the z and $z''$ axis, respectively.

Besides the dominant spinning motions there are also two additional secondary motions, which arise as a consequence of the mechanism (ii). In a nutshell, the gradient torque would like to align the nanoparticle in such a way to minimize the electric dipole potential energy, i.e. $\beta_{0} = \frac{\pi}{2}$ , but once the nanoparticle starts to spin, i.e. it acquires a large angular momenta along the z and $z''$ axis, it is unable to fully align, but rather settles around an equilibrium position $\beta_{0} \neq \frac{\pi}{2}$ , which can be readily understood in terms of angular momenta addition. Any small perturbation, e.g. gas collisions, will make the $\beta$ angle oscillate around the $\beta_{0}$ angle, which results in libration (nutation) motion with frequency $\omega_{\beta}^{(\text{mutation})}$ . In addition, the coupling between $\beta$ and $\alpha$ also creates a second frequency for the $\alpha$ motion, which we denote by $\omega_{\alpha}^{(\text{precession})}$ : this motion can be visualized as a slow precession of the $z''$ axis about the z axis (see Fig 1 (b)). We now derive the rotational frequencies.

The spinning frequencies can be obtained from Eq. (4), by setting $d\pi = 0$ , $\beta = \beta_{0}$ , and neglecting conservative and stochastic terms, i.e. we consider only the driving term due to photon scattering and the friction term due to gas collisions. We find the asymptotic angle conjugate momentum $\boldsymbol{\pi}^{(\mathrm{spin})} = -\frac{1}{2\Gamma_{c}}\boldsymbol{N}_{s}$ , where $\Gamma_{c}$ is the collisional damping rate, and $\boldsymbol{N}_{s} = (N_{\alpha}, N_{\beta}, N_{\gamma})^{\top}$ is the photon scattering torque. We then immediately find the spinning frequencies:

$$
(\omega_ {\alpha} ^ {(\mathrm{spin})}, 0, \omega_ {\gamma} ^ {(\mathrm{spin})}) ^ {\top} = \mathbb {E} [ Y ] \boldsymbol {\pi} ^ {(\mathrm{spin})}, \tag {6}
$$

where $E[\cdot]$ denotes the time-average over fast oscillating terms, $Y = Y(I)$ is the matrix that maps $\pi$ to $\dot{\phi}$ , and I is the moment of inertia tensor in the body-frame (see [37]). The explicit expressions for $\omega_{\alpha}^{(\mathrm{spin})}$ and $\omega_{\gamma}^{(\mathrm{spin})}$ are given in Eqs. (B3) and (B5), respectively.

We now consider the oscillations of $\beta$ about the equilibrium point $\beta_{0}$ , which we denote by $\delta\beta_{0}$ . This oscillatory, libration motion is induced by the (ii) conservative terms, as well as by the fast spinning motion of $\alpha$ and $\gamma$ . In particular, after writing the Hamiltonian eqs. of motion for $\delta\beta_{0}$ , using Eqs. (3) and (4), performing the time average $E[\cdot]$ , and keeping only the dominant terms, we eventually obtain:

$$
\omega_ {\beta} ^ {\text { nutation }} = \frac {1}{2} \frac {I _ {1} + I _ {2}}{I _ {1} I _ {2}} \csc^ {2} (\beta_ {0}) \pi_ {\alpha} ^ {(\text { spin })}, \tag {7}
$$

where $I_{1}$ and $I_{2}$ denote the moment of inertia along the $x''$ and $y''$ principal axis. The explicit expressions for $\omega_{\beta}^{\mathrm{(nutation)}}$ and $\beta_{0}$ are given in Eqs. (B5) and (B6), respectively.

The $\delta\beta_0$ oscillations also perturb the $\alpha$ motion: we denote the perturbation to the spinning motion by $\delta\alpha$ , i.e. $\alpha(t) = \omega_{\alpha}^{(\mathrm{spin})}t + \delta\alpha$ . In particular, from Eq. (3), performing time-average $\mathbb{E}[\cdot]$ , and using Eq. (7), we eventually obtain $\dot{\delta\alpha} = 2\omega_{\beta}^{(\mathrm{mutation})}\cot(\beta_0)\delta\beta$ , where $\delta\beta = \mathcal{B}\cos(\omega_{\beta}^{(\mathrm{mutation})}t)$ , and $\mathcal{B}$ denotes the amplitude of $\delta\beta$ oscillations. However, in a typical detection we do not measure directly $\alpha$ , but normally its sine or cosine value. We consider here $\sin(\alpha)$ , which we Taylor expand to order $\mathcal{O}(\delta\alpha)$ , i.e. $\sin(\alpha) = \sin(\omega_{\alpha}^{(\mathrm{spin})}t) + \cos(\omega_{\alpha}^{(\mathrm{spin})}t)\delta\alpha$ . In particular, from the last term, using trigonometric identities, and the expressions for $\delta\alpha$ and $\delta\beta_0$ , we obtain a term proportional to $\sin(\omega_{\alpha}^{(\mathrm{precession})}t)$ , where

$$
\omega_ {\alpha} ^ {(\mathrm{precession})} = \omega_ {\alpha} ^ {(\mathrm{spin})} - \omega_ {\beta} ^ {(\mathrm{nutation})}. (8)
$$

The $\alpha$ degree of freedom has thus two distinct motions: a fast spinning motion with the frequency given in Eq. (6) and a slow precession motion with the frequency given in Eq. (8). This precession motion can be seen as a consequence of the $\beta$ motion, which perturbs back the $\alpha$ motion.

From Eqs. (6) and (7), noting that $\pi^{(\mathrm{spin})}\propto\frac{\mathbf{N}_{s}}{\Gamma_{c}}\propto\frac{P}{p}$ , we find that $\omega_{\alpha}^{(\mathrm{spin})}$ and $\omega_{\beta}^{(\mathrm{nutation})}$ scale linearly with the laser power P, and are inversely proportional to the gas pressure p, where we have assumed that the equilibrium position $\beta_{0}$ does not change significantly near an initially chosen power $P_{0}$ and pressure $p_{0}$ , $N_{s}$ is the torque due to photon scattering, and $\Gamma_{c}$ is the damping rate due to gas collisions. On the other hand, the precession frequency $\omega_{\alpha}^{(\mathrm{precession})}$ given in Eq. (8) can scale differently depending on the values of $\partial_{P}\Delta\omega|_{P_{0}}$ and $\partial_{p}\Delta\omega|_{p_{0}}$ , where $\Delta\omega=\omega_{\alpha}^{(\mathrm{spin})}-\omega_{\beta}^{(\mathrm{nutation})}$ .

We have confirmed the validity of the obtained approximate formulae in Eqs. (5)-(8) by numerically simulating Eqs. (1)-(4). More details of the simulations will be discussed elsewhere [36].

Experiments- The optical trap is shown in Fig. 1(a). In the experiments presented we use initially individual silica nanoparticles dispersed in water. We observe ageing of the solution with the result of clustering of the nanospheres into compounds of two to three nanospheres, a few weeks after preparation, as shown in Fig. 1(c). The aged solution is delivered to the trap using a nebuliser. The optical scattering force, limits the maximum particle mean radius that can be optically trapped to about $150\mathrm{nm}$ to $200\mathrm{nm}$ .

Results- The power spectral density (PSD) shown in Fig. 2(a) is generated from the time trace recorded as the photodetector signal, $I_{\mathrm{exp}}$ , over a time interval of one second. The PSD shows a rich spectrum of frequencies, which respond differently for changing laser power and background gas pressure.

Translation motion is observed at frequencies, $\omega_{x} = 2\pi \times 196$ kHz, $\omega_{y} = 2\pi \times 246$ kHz, $\omega_{z} = 2\pi \times 124$ kHz in

![](images/5fd8bde48dcae8a60004548435c155ec994c342942baee090353dba40e0dac32.jpg)

<details>
<summary>line</summary>

| Frequency ω/2π (Hz) | Experimental PSD | Simulated PSD |
| ------------------- | ---------------- | ------------- |
| 10^3                | ~10^-10          | ~10^-16       |
| 10^4                | ~10^-10          | ~10^-16       |
| 10^5                | ~10^-10          | ~10^-14       |
| 10^6                | ~10^-12          | ~10^-18       |
| 10^7                | ~10^-12          | ~10^-20       |
</details>

![](images/ca412bf771b07d470a528d6cb8e9eee1b0dc2836ea9aa9c7dce0a7170a815491.jpg)

<details>
<summary>line</summary>

| Pressure (mbar) | ωγ(spin) (Hz) | ωz (Hz) | ωα(precession) (Hz) |
| --------------- | ------------- | ------- | -------------------- |
| 0.1             | ~10^7         | ~10^5   | ~10^3                |
| 1               | ~10^6         | ~10^5   | ~10^4                |
</details>

![](images/9ac13cc3f6fb56796ab0da24abef5af4c3f46474b475cd30c2a257a6ea3ade8f.jpg)

<details>
<summary>line</summary>

| Laser Power (mW) | Frequency, ω/2π (kHz) (c) | Frequency, ω/2π (kHz) (d) |
| ---------------- | -------------------------- | -------------------------- |
| 100              | 30                         | 150                        |
| 200              | 70                         | 250                        |
| 300              | 90                         | 400                        |
| 400              | 110                        | 600                        |
| 500              | 130                        | 800                        |
| 600              | 140                        | 1000                       |
</details>

![](images/60f1ef7869e2a73a9fc989f38aa644b1f47ff3ecc39f06ffb90266e48e21ec62.jpg)

<details>
<summary>line</summary>

| Frequency, ω/2π (MHz) | S(ω) V²/Hz |
| --------------------- | ---------- |
| 1.4                   | ~10⁻¹⁸     |
| 1.8                   | ~10⁻¹⁴     |
| 2.2                   | ~10⁻¹⁰     |
| 2.6                   | ~10⁻¹⁶     |
</details>

Figure 2. Measured Spectrum: (a) The full experimental spectrum obtained at $1 \times 10^{-1}$ mbar (shown in purple). It includes contributions from translation, rotation and precession motion. Additionally, a simulated spectrum (shown in blue) fitted to the experimental PSD is shifted for visibility. The simulated spectrum is obtained for $10^{8}$ timesteps. (b) Shows the frequency dependencies with pressure for translation (blue), rotation (magenta) and precession (red). The inverse relation is signature of rotational, while the direct proportionality is signature of precession. (c) Shows the change in translation frequency due to square root of laser power. (d) shows linear dependency of rotation, $\omega_{\gamma}^{(\mathrm{spin})}$ , with laser power whilst precession is independent of laser power. (e) Detailed spectrum of rotation frequency, $\omega_{\gamma}^{(\mathrm{spin})}$ , with translation sidebands and $\omega_{\alpha}^{\prime}$ an additional frequency peak appearing in the motion of $\alpha$ motion.

Fig. 2(a). We do not observe pressure dependency of the translation frequencies (see Fig. 2(b)), but find that they scale proportionately to the square root of laser power, P, in agreement with Eq.(5), see Fig. 2(c). The x and y peaks are separated because we use elliptically polarized light, which after it is reflected from the paraboloidal mirror, generates an asymmetric optical trap. The polarization of light was kept constant during the course of the experiments in this letter.

From the experimental data, we find the fundamental frequencies for the rotational motions, $\omega_{\gamma}^{spin}$ and $\omega_{\alpha}^{spin}$ to be $2\pi \times 1.9$ MHz and $2\pi \times 3.8$ MHz, respectively, which are perfectly reproduced using Eqs. (6), (B3) and (B5). The rotational frequency $\omega_{\gamma}^{(\mathrm{spin})}$ changes with the damping, $\Gamma_{c}$ , which depends linearly on the gas pressure, p, i.e. $\omega_{\gamma}^{(\mathrm{spin})} \propto \frac{1}{\Gamma_{c}} \propto \frac{1}{p}$ . This is a clear signature of rotation motion, as shown in Fig. 2(b). We also observe a dependency of the frequency on the laser power P, as shown in Fig 2(d) in agreement with the dependency on the photon scattering torque, $N_{s} \propto P$ . Zooming to the fundamental frequency $\omega_{\gamma}^{(\mathrm{spin})}$ reveals sidebands, see Fig. 2(e), which are the addition and subtraction of the three translational frequencies. Using the numerical simulation we identify another mode in $\alpha$ rotation, with frequency $\omega_{\alpha}^{'} = 2\pi \times 393$ kHz: this gives rise to the sideband in $\omega_{\gamma}^{(\mathrm{spin})}$ (see [37]). In the presented data set, we only resolve one of the $\omega_{\alpha}^{'}$ sideband peaks. The light-matter interactions introduce couplings between translation and rotation, which explain the observed sidebands and higher harmonics in agreement with numerical simulation, see Eqs. (A2), (A5)-(A7) in [37] for further details. We also observe $\omega_{\alpha}^{(\mathrm{spin})}$ to scale linearly with power and inversely with pressure.

Further to observing translation and rotation peaks, we observe a frequency at $2\pi \times 5.4$ kHz, as shown in Fig. 2(a). This frequency is well-isolated and characterised by its dependency on laser power, P and gas pressure, p. We associate this low frequency with the gradient torque-induced precession, $\omega_{\alpha}^{(\text{precession})}$ . From Eq. (8), Taylor expanding about the initial pressure $p_{0}$ , we find that the dominant term is linear in p (the constant terms cancel), i.e. $\omega_{\alpha}^{(\text{precession})} = \partial_{p}(\Delta\omega)|_{p_{0}}p$ . On the other hand, we find a weak dependence on the laser power P, i.e. $\omega_{\alpha}^{(\text{precession})} = \Delta\omega|_{P_{0}}$ , where $P_{0}$ is the initial laser power. We verify these frequency dependencies on gas pressure and laser power in Figs. 2(b) and

2(d), respectively. We exclude $\omega_{\alpha}^{(\text{precession})}$ to be caused by nonlinear translational motion, as the translation frequencies do not change with pressure. This is evident from Fig. 2(b) where $\omega_{z}$ is constant. Thus, we conclude, that the observed frequency and its behaviour is signature of precession motion as described by Eq. (8).

The precession motion arises due to the fast spinning $\alpha$ degree of freedom, as well as, the nutation motion of $\beta$ . The observation of precession is thus an indirect indication of nutation motion. We associate the time trace of the $\beta$ motion to libration motion, which is linearly dependant on power and inverse proportional to gas pressure, as a result of the numerical simulation. The $\beta$ libration is due to the coupling between $\alpha$ and $\beta$ motion.

Discussion- From the theoretical analysis, the precession motion arises due to an optical-torque acting upon the trapped particle. Torque can also be generated by an external force, which opens the way for sensitive detection of forces by precession. In particular, by combining Eqs. (6), (7), and (8), we get an expression for the $\alpha$ -component of the photon scattering torque:

$$
N _ {\alpha} = \Gamma_ {c} \sin^ {2} (\beta_ {0}) \left[ \frac {2 I _ {1} I _ {2} (I _ {1} + I _ {2})}{I _ {1} ^ {2} + I _ {2} ^ {2}} \right] \omega_ {\alpha} ^ {(\text { precession })}, \tag {9}
$$

where we have kept only the dominant terms, and denote the term in the square brackets by J, and name it the effective moment of inertia. Using the experimentally measured $\omega_{\alpha}^{(\mathrm{precession})}$ , and estimating J, we achieve a measured torque of $N_{\alpha}=1.9\pm0.5\times10^{-23}Nm$ at $1\times10^{-1}mbar$ , in comparison to the measurement of nanoscale torque sensors reported down to $10^{-20}Nm$ [38] and to estimates of $10^{-22}Nm$ [30] for silicon nanorods.

We now consider an additional small external torque acting on $\alpha$ , which we denote by $\delta N_{ext}$ . We suppose that Eq. (9) remains valid, when we formally make the replacement $N_{\alpha} \rightarrow N_{\alpha} + \delta N_{ext}$ , and we denote the corresponding change in the precession frequency by $\delta\omega_{\alpha}$ . Furthermore, assuming that the equilibrium value of $\beta_{0}$ remains largely unaffected, we obtain the torque sensitivity $\delta N_{\mathrm{ext}} = \Gamma_{c}\sin(\beta_{0})^{2}\mathcal{J}\delta\omega_{\alpha}$ . Using experimental parameters we estimate a torque sensitivity of $3.6 \pm 1.1 \times 10^{-31}$ Nm/ $\sqrt{Hz}$ at $1 \times 10^{-7}$ mbar, limited only by photon shot noise (see supplementary D [37]) in comparison to the torque sensitivity using libration motion, which is at $1 \times 10^{-29}$ Nm/ $\sqrt{Hz}$ at $10^{-9}$ mbar [24].

Conclusions- We have observed precession motion in levitated optomechanics and inferred the presence of nutation motion. We present a theory of this motion and show that it arises from the equations of motion for a rotating object experiencing scattering and gradient forces and torques. Additionally, we characterise the rich spectrum detected with translation, rotation and higher harmonics. We further show a measured torque of $10^{-23}\mathrm{Nm}$ at $10^{-1}$ mbar, and predict the ability to reach sensitivities down to $10^{-31}\mathrm{Nm} / \sqrt{\mathrm{Hz}}$ at $10^{-7}$ mbar.

Thus, precession motion is a degree of freedom that could be utilised for torque sensing with the sensitivities to resolve single electron $[39]$ and even nuclear spins $[24]$ at low pressure. This work paves the way for gyroscope applications, as shown in $[40]$ . The precession motion can also be used for dynamical model selection to distinguish between quantum and classical evolution due to the inherent nonlinearities in rotation motion $[41]$ , if sufficient coherence can be prepared. Additionally, this work can be used to reconstruct the shape and effective moment of inertia, J, from the knowledge of the full spectrum of freedom containing all motional degrees of the trapped particle.

Acknowledgments- We thank C. Timberlake and G. Winstone for discussion and M. Rademacher for assistance with the TEM image. We thank for funding the Leverhulme Trust [RPG-2016-046] and the EU Horizon 2020 research and innovation programme under grant agreement No 766900 [TEQ]. A.S. is supported by the UK Engineering and Physical Sciences Research Council (EPSRC) under Centre for Doctoral Training Grant No. EP/L015382/1. All data supporting this study are openly available from the University of Southampton repository at https://doi.org/10.5258/SOTON/D0523"

\* m.rashid@soton.ac.uk   
$^{\dagger}$ m.toros@soton.ac.uk   
$^{\ddagger}$ h.ulbricht@soton.ac.uk   
[1] T. Li, S. Kheifets, D. Medellin, and M. G. Raizen, Science 328, 1673 (2010).   
[2] J. Gieseler, L. Novotny, and R. Quidant, Nat Phys 9, 806 (2013).   
[3] P. Fonseca, E. Aranas, J. Millen, T. Monteiro, and P. Barker, Phys. Rev. Lett. 117, 173602 (2016).   
[4] J. Gieseler, R. Quidant, C. Dellago, and L. Novotny, Nat. Nanotechnol. 9, 358 (2014).   
[5] J. Gieseler, M. Spasenović, L. Novotny, and R. Quidant, Phys. Rev. Lett. 112 (2014).   
[6] T. M. Hoang, R. Pan, J. Ahn, J. Bang, H. T. Quan, and T. Li, Phys. Rev. Lett. 120, 080602 (2018).   
[7] J. Millen, T. Deesuwan, P. Barker, and J. Anders, Nat. Nanotechnol. 9, 425 (2014).   
[8] L. Rondin, J. Gieseler, F. Ricci, R. Quidant, C. Dellago, and L. Novotny, Nat. Nanotechnol. 12, 1130 (2017).   
[9] E. B. Aranas, P. Z. Fonseca, P. F. Barker, and T. S. Monteiro, J. Opt 19, 034003 (2017).   
[10] G. Ranjit, M. Cunningham, K. Casey, and A. A. Geraci, Physical Review A 93, 053801 (2016).   
[11] E. Hebestreit, M. Frimmer, R. Reimann, and L. Novotny, Arxiv preprint (2017), arXiv:1801.01169.   
[12] D. Hempston, J. Vovrosh, M. Toroš, G. Winstone, M. Rashid, and H. Ulbricht, Appl. Phys. Lett. 111, 133111 (2017).   
[13] O. Romero-Isart, A. C. Pflanzer, F. Blaser, R. Kaltenbaek, N. Kiesel, M. Aspelmeyer, and J. I. Cirac, Phys. Rev. Lett. 107, 020405 (2011), 1103.4081.   
[14] A. Bassi, K. Lochan, S. Satin, T. P. Singh, and H. Ul-

bricht, Rev. Mod. Phys. 85, 471 (2013).   
[15] J. Bateman, S. Nimmrichter, K. Hornberger, and H. Ulbricht, Nat. Commun. 5, 4788 (2014).   
[16] T. Li, S. Kheifets, and M. G. Raizen, Nature Physics 7, 18 (2011).   
[17] N. Kiesel, F. Blaser, U. Delić, D. Grass, R. Kaltenbaek, and M. Aspelmeyer, Proc. Natl. Acad. Sci. U.S.A. 110, 14180 (2013).   
[18] J. Vovrosh, M. Rashid, D. Hempston, J. Bateman, M. Paternostro, and H. Ulbricht, J. Opt. Soc. Am. B 34, 1421 (2017).   
[19] A. Setter, M. Toroš, J. F. Ralph, and H. Ulbricht, Phys. Rev. A 97, 033822 (2018).   
[20] V. Jain, J. Gieseler, C. Moritz, C. Dellago, R. Quidant, and L. Novotny, Phys. Rev. Lett. 116, 243601 (2016).   
[21] Y. Arita, M. Mazilu, and K. Dholakia, Nat. Commun. 4, 2374 (2013).   
[22] S. Kuhn, P. Asenbaum, A. Kosloff, M. Sclafani, B. A. Stickler, S. Nimmrichter, K. Hornberger, O. Cheshnovsky, F. Patolsky, and M. Arndt, Nano Lett. 15, 5604 (2015).   
[23] A. T. M. A. Rahman and P. F. Barker, Nat. Photonics 11, 634 (2017).   
[24] T. M. Hoang, Y. Ma, J. Ahn, J. Bang, F. Robicheaux, Z.-Q. Yin, and T. Li, Phys. Rev. Lett. 117, 123604 (2016).   
[25] S. Kuhn, A. Kosloff, B. A. Stickler, F. Patolsky, K. Hornberger, M. Arndt, and J. Millen, Optica 4, 356 (2017).   
[26] R. Reimann, M. Doderer, E. Hebestreit, R. Diehl, M. Frimmer, D. Windey, F. Tebbenjohanns, and L. Novotny, Arxiv preprint (2018), arXiv:1803.11160.   
[27] J. Ahn, Z. Xu, J. Bang, Y.-H. Deng, T. M. Hoang, Q. Han, R.-M. Ma, and T. Li, Arxiv preprint (2018), arXiv:1804.06570.

[28] M. Carlesso, M. Paternostro, H. Ulbricht, and A. Bassi, Arxiv preprint (2017), arXiv:1710.08695.   
[29] Y. Ma, T. M. Hoang, M. Gong, T. Li, and Z.-Q. Yin, Physical Review A 96, 023827 (2017).   
[30] S. Kuhn, B. A. Stickler, A. Kosloff, F. Patolsky, K. Hornberger, M. Arndt, and J. Millen, Nat. Commun. 8, 1670 (2017).   
[31] B. A. Stickler, B. Papendell, S. Kuhn, J. Millen, M. Arndt, and K. Hornberger, Arxiv preprint (2018), arXiv:1803.01778.   
[32] M. Carlesso, M. Paternostro, H. Ulbricht, A. Vinante, and A. Bassi, Arxiv preprint (2017), arXiv:1708.04812.   
[33] A. Manjavacas, F. J. Rodríguez-Fortuño, F. J. García de Abajo, and A. V. Zayats, Phys. Rev. Lett. 118, 133605 (2017).   
[34] M. Rashid, M. Toroš, and H. Ulbricht, Quantum Meas. Quantum Metrol. 4, 17 (2018).   
[35] M. Toroš, M. Rashid, and H. Ulbricht, Arxiv preprint (2018), arXiv:1804.01150.   
[36] M. Toroš, M. Rashid, and H. Ulbricht, (2018), (in preparation).   
[37] (Supplementary Material, URL to be added by editor).   
[38] P. H. Kim, C. Doolin, B. D. Hauer, A. J. MacDonald, M. R. Freeman, P. E. Barclay, and J. P. Davis, Applied Physics Letters 102, 053102 (2013).   
[39] D. Rugar, R. Budakian, H. J. Mamin, and B. W. Chui, Nature 430, 329 (2004).   
[40] P. Nagornykh, J. E. Coppock, J. P. Murphy, and B. Kane, Physical Review B 96, 035402 (2017).   
[41] J. F. Ralph, M. Toroš, S. Maskell, K. Jacobs, M. Rashid, A. J. Setter, and H. Ulbricht, arXiv preprint arXiv:1711.09635 (2017).

# Appendix A: Dynamics

In this supplementary section we list the terms obtained in $[35, 36]$ . We start by specifying the conservative terms. The free Hamiltonian is given by:

$$
\begin{array}{l} H _ {\mathrm{free}} = \frac {p _ {x} ^ {2} + p _ {y} ^ {2} + p _ {z} ^ {2}}{2 M} + \left(\frac {\csc^ {2} (\beta) (\cos (\gamma) (\pi_ {\alpha} - \pi_ {\gamma} \cos (\beta)) - \pi_ {\beta} \sin (\beta) \sin (\gamma)) ^ {2}}{2 I _ {1}}\right) \\ \left. + \frac {\csc^ {2} (\beta) (\sin (\gamma) (\pi_ {\alpha} - \pi_ {\gamma} \cos (\beta)) + \pi_ {\beta} \sin (\beta) \cos (\gamma)) ^ {2}}{2 I _ {2}} + \frac {\pi_ {\gamma} {} ^ {2}}{2 I _ {3}}\right), \tag {A1} \\ \end{array}
$$

where $M$ is the mass of the nanoparticle, and $I = \mathrm{diag}(I_1, I_2, I_3)$ is the moment of inertia tensor in the body frame.

The gradient potential is given by

$$
\begin{array}{l} H _ {\mathrm{grad}} = - \frac {V P}{c \sigma_ {L}} | u (\boldsymbol {r}) | ^ {2} \bigg (a ^ {2} \big (\chi_ {1} (\cos (\alpha) \cos (\beta) \cos (\gamma) - \sin (\alpha) \sin (\gamma)) ^ {2} \big) \\ + \chi_ {2} (\cos (\alpha) \cos (\beta) \sin (\gamma) + \sin (\alpha) \cos (\gamma)) ^ {2} + \chi_ {3} \cos^ {2} (\alpha) \sin^ {2} (\beta)) \\ + b ^ {2} \left(\chi_ {1} (\sin (\alpha) \cos (\beta) \cos (\gamma) + \cos (\alpha) \sin (\gamma)) ^ {2} \right. \\ \left. \left. + \chi_ {2} (\cos (\alpha) \cos (\gamma) - \sin (\alpha) \cos (\beta) \sin (\gamma)) ^ {2} + \chi_ {3} \sin^ {2} (\alpha) \sin^ {2} (\beta)\right)\right). \tag {A2} \\ \end{array}
$$

where $u$ is a modified Gaussian mode function:

$$
u (\boldsymbol {r}) = \frac {w _ {0}}{w (z)} \exp \left(- \frac {a _ {1} x ^ {2} + a _ {2} y ^ {2}}{w (z) ^ {2}}\right) e ^ {- i k z}, \tag {A3}
$$

$w_{0}$ is an effective beam waist, $\sigma_{L} = \pi w_{0}^{2}$ , $a_{1}$ , $a_{2}$ denote the asymmetry along the $x$ , $y$ axis, respectively, $a_{1}a_{2} = 1$ , $w(z) = w_{0}\sqrt{(1 + \frac{z^{2}}{z_{R}^{2}})}$ , $z_{R}$ is the Rayleigh range, $k = \frac{2\pi}{\lambda}$ , $\lambda$ is the laser wavelength, $P$ is the laser power, $c$ is the speed of light, and $\chi = \mathrm{diag}(\chi_1,\chi_2,\chi_3)$ is the susceptibility tensor in the body-frame.

We now specify the non-conservative terms. We first discuss the deterministic terms related to photon scattering, namely $dp_{\mathrm{scatt}} = (0, 0, dp_{z}^{(\mathrm{ds})})^{\top}$ and $d\pi_{\mathrm{scatt}} = (d\pi_{\alpha}^{(\mathrm{ds})}, d\pi_{\beta}^{(\mathrm{ds})}, d\pi_{\gamma}^{(\mathrm{ds})})^{\top}$ . In particular, we have:

$$
d p _ {z} ^ {\mathrm{(ds)}} = \frac {1 6 \pi \hbar \Gamma_ {s}}{3} \frac {2 \pi}{\lambda} | u (\boldsymbol {r}) | ^ {2} d t, \tag {A4}
$$

$$
d \pi_ {\alpha} ^ {(\mathrm{ds})} = \frac {4 \pi b _ {x} b _ {y} \hbar \Gamma_ {s}}{3} | u (\pmb {r}) | ^ {2} \bigg (- 2 \sin^ {2} (\beta) \cos (2 \gamma) (\chi_ {1} - \chi_ {2}) (\chi_ {1} + \chi_ {2} - 2 \chi_ {3})
$$

$$
+ \cos (2 \beta) \left(\chi_ {1} ^ {2} + 2 \chi_ {3} (\chi_ {1} + \chi_ {2}) - 4 \chi_ {1} \chi_ {2} + \chi_ {2} ^ {2} - 2 \chi_ {3} ^ {2}\right)
$$

$$
\left. + 3 \chi_ {1} ^ {2} - 2 \chi_ {3} (\chi_ {1} + \chi_ {2}) - 4 \chi_ {1} \chi_ {2} + 3 \chi_ {2} ^ {2} + 2 \chi_ {3} ^ {2}\right) d t, \tag {A5}
$$

$$
d \pi_ {\beta} ^ {(\mathrm{ds})} = \frac {1 6 \pi b _ {x} b _ {y} \hbar \Gamma_ {s}}{3} | u (\boldsymbol {r}) | ^ {2} \sin (\beta) \sin (\gamma) \cos (\gamma) (\chi_ {1} - \chi_ {2}) (\chi_ {1} + \chi_ {2} - 2 \chi_ {3}) d t, \tag {A6}
$$

$$
d \pi_ {\gamma} ^ {(\mathrm{ds})} = \frac {1 6 \pi b _ {x} b _ {y} \hbar \Gamma_ {s}}{3} | u (\boldsymbol {r}) | ^ {2} \cos (\beta) (\chi_ {1} - \chi_ {2}) ^ {2} d t, \tag {A7}
$$

where $\Gamma_s = \frac{\tilde{\sigma}_R}{\sigma_L} \frac{P}{\hbar \omega_L}$ is the scattering rate, $\omega_L = \frac{2\pi c}{\lambda}$ , $\tilde{\sigma}_R = \frac{\pi^2 V_0^2}{\lambda^4}$ is an effective scattering cross-section, $b_x$ and $b_y$ are the components of the Gaussian beam polarization vector $\epsilon_d = (b_x, ib_y, 0)^\top$ , and $b_x^2 + b_y^2 = 1$ .

We now discuss the gas collision terms denoted by $d\pmb{p}_{\mathrm{coll}} = d\pmb{p}^{(\mathrm{dc})} + d\pmb{p}^{(\mathrm{sc})}$ and $d\pmb{\pi}_{\mathrm{coll}} = d\pmb{\pi}^{(\mathrm{dc})} + d\pmb{\pi}^{(\mathrm{sc})}$ . The terms denoted by the superscripts (ds) and (sc) corresponds to non-conservative deterministic and stochastic terms,

respectively. In particular, we have

$$
d \boldsymbol {p} ^ {\left(\mathrm{dc}\right)} = - 2 \Gamma_ {C} \boldsymbol {p} d t, \tag {A8}
$$

$$
d \boldsymbol {\pi} ^ {(\mathrm{dc})} = - 2 \Gamma_ {C} \boldsymbol {\pi} d t, \tag {A9}
$$

$$
d \boldsymbol {p} _ {k} ^ {\text {(sc)}} = \sqrt {4 m k _ {B} T \Gamma_ {c}} d V _ {k}, \tag {A10}
$$

$$
d \boldsymbol {\pi} _ {k} ^ {(\mathrm{sc})} = \sum_ {\zeta , j = 1} ^ {3} \sqrt {4 k _ {B} m T \tilde {D} _ {\zeta} \Gamma_ {c}} (\partial_ {\phi_ {k}} R) _ {j, \zeta} d Z _ {\zeta , j}, \tag {A11}
$$

where $\Gamma_{c} = \frac{\pi p_{g}r_{g}^{2}}{\sqrt{8m_{g}k_{B}T}}$ is a characteristic collision rate, $p_{g}$ is the gas pressure, $r_g$ and $m_{g}$ are the radius and mass of a gas particle, respectively, $k_{B}$ is Boltzmann's constant, $T$ is the gas temperature, and $\tilde{D}_{\zeta} = \frac{1}{2} (\mathrm{tr}(I) - I_{\zeta})$ . $V_{k}$ and $Z_{\zeta,j}$ are zero mean independent Wiener processes.

# Appendix B: System frequencies

Here we list the formulae, which have been used in the main text, to obtain the dominant frequencies of the system. The photon scattering torque is given by

$$
\boldsymbol {N} _ {s} = \mathbb {E} \left[ \left(\frac {d \pi_ {\alpha} ^ {(\mathrm{ds})}}{d t}, \frac {d \pi_ {\beta} ^ {(\mathrm{ds})}}{d t}, \frac {d \pi_ {\gamma} ^ {(\mathrm{ds})}}{d t}\right) ^ {\top} \right], \tag {B1}
$$

where $d\pi_{\alpha}^{(\mathrm{ds})}, d\pi_{\beta}^{(\mathrm{ds})}$ , and $d\pi_{\gamma}^{(\mathrm{ds})}$ are given in Eqs. (A5), (A6), and (A7), respectively, and E denotes the time average over fast oscillating terms.

The conversion matrix $Y$ from the conjugate angle momenta $\pi$ to the time-derivative of the angle vector $\phi$ is defined as

$$
Y = \left(\left( \begin{array}{c c c} - \sin (\beta) \cos (\gamma) & \sin (\gamma) & 0 \\ \sin (\beta) \sin (\gamma) & \cos (\gamma) & 0 \\ \cos (\beta) & 0 & 1 \end{array} \right) ^ {\top} \left( \begin{array}{c c c} I _ {1} & 0 & 0 \\ 0 & I _ {2} & 0 \\ 0 & 0 & I _ {3} \end{array} \right) \left( \begin{array}{c c c} - \sin (\beta) \cos (\gamma) & \sin (\gamma) & 0 \\ \sin (\beta) \sin (\gamma) & \cos (\gamma) & 0 \\ \cos (\beta) & 0 & 1 \end{array} \right)\right) ^ {- 1}. \tag {B2}
$$

The dominant frequencies for the spin motion are given by

$$
\omega_ {\alpha} ^ {(\mathrm{spin})} = \frac {2 b _ {x} b _ {y} \pi \hbar}{\frac {3}{4} (I _ {1} ^ {2} + 2 I _ {1} I _ {2} + I _ {2} ^ {2}) I _ {3}} \frac {\Gamma_ {s}}{\Gamma_ {c}} \csc^ {2} (\beta_ {0}) \bigg [ \frac {1}{2} (I _ {1} + I _ {2}) I _ {1} (\chi_ {1} - \chi_ {2}) ^ {2} \cos^ {2} (\beta_ {0}) - (\frac {1}{2} I _ {1} I _ {3} + \frac {1}{2} I _ {2} I _ {3}) \bigg ]
$$

$$
\left. \left(3 \chi_ {1} ^ {2} - 4 \chi_ {2} \chi_ {1} + 3 \chi_ {2} ^ {2} + 2 \chi_ {3} ^ {2} - 2 (\chi_ {1} + \chi_ {2}) \chi_ {3} + \left(\chi_ {1} ^ {2} - 4 \chi_ {2} \chi_ {1} + \chi_ {2} ^ {2} - 2 \chi_ {3} ^ {2} + 2 (\chi_ {1} + \chi_ {2}) \chi_ {3}\right) \cos (2 \beta_ {0})\right) \right], \quad \mathrm{(B3)}
$$

and

$$
\omega_ {\gamma} ^ {(\mathrm{spin})} = \frac {2 b _ {x} b _ {y} \pi \hbar}{\frac {3}{4} (I _ {1} ^ {2} + 2 I _ {1} I _ {2} + I _ {2} ^ {2}) I _ {3}} \frac {\Gamma_ {s}}{\Gamma_ {c}} \cot (\beta_ {0}) \csc (\beta_ {0})
$$

$$
\left[ - \frac {1}{2} \left((I _ {1} + I _ {2}) I _ {3} \cos^ {2} (\beta_ {0}) + \left(\frac {1}{2} I _ {1} ^ {2} + I _ {2} I _ {1} + \frac {1}{2} I _ {2} ^ {2}\right) \sin^ {2} (\beta_ {0})\right) (\chi_ {1} - \chi_ {2}) ^ {2} \left(- (- \frac {1}{2} I _ {1} - \frac {1}{2} I _ {2}) I _ {3} \cos^ {2} (\beta_ {0})\right) \right].
$$

$$
\left. \left(3 \chi_ {1} ^ {2} - 4 \chi_ {2} \chi_ {1} + 3 \chi_ {2} ^ {2} + 2 \chi_ {3} ^ {2} - 2 (\chi_ {1} + \chi_ {2}) \chi_ {3} + \left(\chi_ {1} ^ {2} - 4 \chi_ {2} \chi_ {1} + \chi_ {2} ^ {2} - 2 \chi_ {3} ^ {2} + 2 (\chi_ {1} + \chi_ {2}) \chi_ {3}\right) \cos (2 \beta_ {0})\right)\right) \Bigg ], \tag {B4}
$$

while the nutation frequency is given by

$$
\begin{array}{l} \omega_ {\beta} ^ {\mathrm{mutation}} = \frac {1}{2} \frac {I _ {1} + I _ {2}}{I _ {1} I _ {2}} \csc^ {2} (\beta_ {0}) \frac {2 \pi b _ {x} b _ {y} \hbar}{3} \frac {\Gamma_ {s}}{\Gamma_ {c}} \bigg [ \cos (2 \beta_ {0}) \big (\chi_ {1} ^ {2} + 2 \chi_ {3} (\chi_ {1} + \chi_ {2}) - 4 \chi_ {1} \chi_ {2} + \chi_ {2} ^ {2} - 2 \chi_ {3} ^ {2} \big) \\ \left. + 3 \chi_ {1} ^ {2} - 2 \chi_ {3} (\chi_ {1} + \chi_ {2}) - 4 \chi_ {1} \chi_ {2} + 3 \chi_ {2} ^ {2} + 2 \chi_ {3} ^ {2} \right]. \tag {B5} \\ \end{array}
$$

![](images/bf0e8eb2073864c83e3dbf3b41c8c9916007be43e7d22e620a229f09634b2eee.jpg)

<details>
<summary>line</summary>

| Frequency, ω/2π (Hz) | ωα^(precession) | ωγ^(spin) | ωβ^(nutation) | ωα' | ωα' (spin) | ωα | ωz |
|----------------------|------------------|-----------|---------------|-----|------------|----|----|
| 10^3                 | ~-150            | ~-50      | ~-100         | ~-200 | ~-200      | ~-200 | ~-200 |
| 10^4                 | ~-180            | ~-60      | ~-110         | ~-210 | ~-210      | ~-210 | ~-210 |
| 10^5                 | ~-200            | ~-70      | ~-120         | ~-220 | ~-220      | ~-220 | ~-220 |
| 10^6                 | ~-220            | ~-80      | ~-130         | ~-230 | ~-230      | ~-230 | ~-230 |
| 10^7                 | ~-250            | ~-90      | ~-140         | ~-240 | ~-240      | ~-240 | ~-240 |
</details>

Figure 3. Simulated Spectrum: The power spectral density of the six different degrees of motion, where $\omega_{\gamma}^{(\mathrm{spin})}$ (in green), $\omega_{\beta}^{(\mathrm{mutation})}$ (in magenta), $\omega_{\alpha}^{(\mathrm{spin})}$ (in blue) are the rotation motions; $\omega_{x}$ (in grey), $\omega_{y}$ (in brown), and $\omega_{z}$ (in dark green) are the translation motions.

The equilibrium position of the $\beta$ angle is given approximately by:

$$
\beta_ {0} = \sin^ {- 1} \left(\sqrt [ 4 ]{\frac {(I _ {1} + I _ {2}) \pi c w _ {0} ^ {2} \pi_ {\alpha} ^ {2}}{I _ {1} I _ {2} P V (2 \chi_ {3} - \chi_ {1} - \chi_ {2})}}\right). \tag {B6}
$$

The fact that $\beta_0$ depends on $\pi_{\alpha}$ is a consequence of the coupling in Eqs. (A1) and (A2).

To estimate the dominant frequencies using Eqs. (B3) - (B5) we have numerically simulated the system and extracted the fitted parameters (see supplementary material C). In particular, we obtain from the simulation the following values $\omega_{\alpha}^{(\mathrm{spin})}=2\pi\times3.919\mathrm{MHz},\;\omega_{\gamma}^{(\mathrm{spin})}=2\pi\times1.957\mathrm{MHz},\;\omega_{\beta}^{(\mathrm{mutation})}=2\pi\times3.924\mathrm{MHz},\;\mathrm{and}\;\omega_{\alpha}^{(\mathrm{precession})}=2\pi\times5.5\mathrm{kHz}$ in perfect agreement with experimental data. Using the simulation parameters, we also find good agreement within order of magnitude with the approximate expressions in Eqs. (B3) - (B5). Thus, Eqs. (B3) - (B5), can be used for estimating the initial simulation parameters for the fitting algorithm.

# Appendix C: Numerical Simulation

In this section we further discuss the different contributions to the simulated spectrum shown in Fig. 2(a). The solution to the twelve coupled SDEs, give information of $x(t)$ , $y(t)$ , $z(t)$ , $\alpha(t)$ , $\beta(t)$ and $\gamma(t)$ . From these we can extract the frequency spectrum, as shown in Fig. 3. The spectrum for each degree of freedom demonstrates the source of the numerous frequencies observed in Fig. 2(a).

$\alpha$ -rotation (blue line in Fig. 3): Starting from the right to left, the $\alpha$ rotation contains frequencies relating the main rotation peak, $\omega_{\alpha}^{(\mathrm{spin})}$ and its higher harmonics. In addition to this there are sidebands to $\omega_{\alpha}^{(\mathrm{spin})}$ which relate to another mode of motion designated as $\omega_{\alpha}^{\prime}$ . Farther to the left, of the spectrum we observe this additional mode, $\omega_{\alpha}^{\prime}$ and its second harmonic. The far left of the $\alpha$ spectrum show the precession motion.   
$\beta$ -nutation (magenta line in Fig. 3): The spectrum for $\beta$ motion shows the central nutation frequency $\omega_{\beta}^{(\mathrm{nutation})}$ and its higher harmonics. The sidebands refer to $\omega_{\alpha}^{\prime}$ and $2\omega_{\alpha}^{\prime}$ . Both these frequencies also appear in the actual spectrum as well.   
$\gamma$ -rotation (green line in Fig. 3): The spectrum for $\gamma$ rotation shows $\omega_{\alpha}^{(\mathrm{spin})}$ and its higher harmonics. The sidebands relate to $\omega_{\alpha}^{\prime}$ and come about due to the coupling of $\gamma$ -rotation with $\alpha$ -rotation. In addition to the rotation degrees of

<table><tr><td></td><td>gas collisions</td><td>photon scattering</td></tr><tr><td>translations</td><td> $4k_{B}TM\Gamma_{c}$ </td><td> $\Gamma_{s}\hbar^{2}\left(\frac{2\pi}{\lambda}\right)^{2}$ </td></tr><tr><td>rotations</td><td> $\frac{4}{5}k_{B}TMR^{2}\Gamma_{c}$ </td><td> $\Gamma_{s}\hbar^{2}$ </td></tr></table>

Table I. The expressions denote estimates for the variance of momentum (angular momentum) fluctuations per unit time for translations (rotations), induced by gas collisions and photon scattering. Note that this expressions are good estimates for a system that is not highly anisotropic, while for a highly anisotropic objects, the moment of inertia and electric susceptibility tensors have to be taken into account [36]. $M$ and $R$ denote an effective radius and mass of the nanoparticle, $\Gamma_s$ and $\Gamma_c$ denote photon scattering and gas collision rate, respectively, $T$ is the temperature of the gas, and $\lambda$ is the laser wavelength (see supplementary material A).

freedom, Fig. 3 also shows the translation motions, $\omega_{x}$ , $\omega_{y}$ and $\omega_{z}$ .

# Appendix D: Recoil heating

We can estimate the recoil heating rates for translational and rotational degrees of freedom due to gas collisions and photon scattering using the Table. I. These expressions can be derived heuristically by considering the amount of linear and angular momentum carried by a single gas particle or photon. Specifically, the amount of linear and angular momentum carried by a gas particle can be estimated as $\sqrt{2m_{g}k_{b}T}$ and $\sqrt{2m_{g}k_{b}TR^{2}}$ , respectively, where $m_{g}$ is the mass of a gas particle and R is an effective radius of the nanoparticle. To obtain the net effect on the nanoparticle we have to take into account the gas collisions scattering cross section: we formally replace $m_{g}$ with the mass of the nanoparticle M (we do not change the expressions in $\Gamma_{c}$ , which is associated with a single atom of the nanoparticle). We can make a similar argument for photons: the amount of linear and angular momentum carried by a photon particle is $\hbar\frac{2\pi}{\lambda}$ and $\hbar$ , respectively. However, the photon scattering cross section is already included in $\Gamma_{s}$ , and thus we immediately obtain the linear and angular momentum fluctuations per unit time in Table. I. A more rigorous derivation, based on the quantum model [35], will be given elsewhere [36].

To get a rough numerical estimate we have considered the moment of inertia tensor of a sphere and a susceptibility tensor that in the body frame has the diagonal elements close to unity.

Using the formulae in Table. I we compare the strength of the heating mechanisms in Fig. 4. This analysis also reproduces the transition to the photon recoil heating regime for translations is in accordance with Fig. 3 from [20].

![](images/0554e3e1e61da7f34a5792611bf05c96cf2653f8b578de1532b60cf397db1356.jpg)

<details>
<summary>line</summary>

| p / mbar | translations | rotations |
| -------- | ------------ | --------- |
| 1e-9     | 0.01         | 0.01      |
| 1e-8     | 0.1          | 0.1       |
| 1e-7     | 1            | 1         |
| 1e-6     | 10           | 10        |
| 1e-5     | 100          | 100       |
</details>

Figure 4. Comparison of recoil heating from gas collisions and from photon scattering. We plot the ratio of the estimates in each row from Table. I. Values bigger (smaller) than 1 mean that the recoil heating from gas collisions (photon scattering) is dominant.