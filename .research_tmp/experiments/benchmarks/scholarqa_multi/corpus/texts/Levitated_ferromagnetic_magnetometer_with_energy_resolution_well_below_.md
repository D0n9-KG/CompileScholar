# Single-domain Bose condensate magnetometer achieves energy resolution per bandwidth below $\hbar$

Silvana Palacios Alvarez<sup>a</sup>, Pau Gomez<sup>a</sup>, Simon Coop<sup>a</sup>, Roberto Zamora-Zamora<sup>b</sup>, Chiara Mazzinghi<sup>a</sup>, and Morgan W. Mitchell<sup>a,c,1</sup>

$^{a}$ ICFO - Institut de Ciencies Fotoniques, The Barcelona Institute of Science and Technology, 08860 Castelldefels (Barcelona), Spain; $^{b}$ Quantum Computing and Devices (QCD) Labs, Department of Applied Physics, Aalto University and Quantum Technology Finland (QTF) Centre of Excellence, FI-00076 Aalto, Finland; and $^{c}$ ICREA - Institucio Catalana de Recerca i Estudis Avançats, 08010 Barcelona, Spain

Edited by Vanderlei Bagnato, Instituto de Fisica de São Carlos, Universidade de São Paulo, Sao Carlos, Brazil; received August 27, 2021; accepted December 13, 2021

We present a magnetic sensor with energy resolution per bandwidth $E_{R} < \hbar$ . We show how a $^{87}\mathrm{Rb}$ single-domain spinor Bose-Einstein condensate, detected by nondestructive Faraday rotation probing, achieves single-shot low-frequency magnetic sensitivity of 72(8) fT measuring a volume $V = 1,091(30)\mu \mathrm{m}^3$ for 3.5 s, and thus, $E_{R} = 0.075(16)\hbar$ . We measure experimentally the condensate volume, spin coherence time, and readout noise and use phase space methods, backed by three-dimensional mean-field simulations, to compute the spin noise.

Contributions to the spin noise include one-body and three-body losses and shearing of the projection noise distribution, due to competition of ferromagnetic contact interactions and quadratic Zeeman shifts. Nonetheless, the fully coherent nature of the single-domain, ultracold two-body interactions allows the system to escape the coherence vs. density trade-off that imposes an energy resolution limit on traditional spin precession sensors.

We predict that other Bose-condensed alkalis, especially the antiferromagnetic $^{23}\mathrm{Na}$ , can further improve the energy resolution of this method.

quantum sensing | magnetometry | Bose-Einstein condensates

Well-known quantum limits profoundly, but not irreremediately, constrain our knowledge of the physical world. Uncertainty relations forbid precise, simultaneous knowledge of observables such as position and momentum. Parameter estimation limits, e.g., the standard quantum limit and Heisenberg limit, constrain our ability to measure transformations not subject to uncertainty relations, e.g., rotations (1, 2). Both these classes of quantum limits admit trade-offs: uncertainty principles allow an observable to be precisely known if one foregoes knowledge of its conjugate observable, and parameter estimation limits allow better precision in exchange for a greater investment of resources, e.g., particle number.

A qualitatively different sort of quantum limit is found in magnetic field sensing, where well-studied sensor technologies are known to obey a quantum limit on the energy resolution per bandwidth,

$$
E _ {R} \equiv \frac {\left\langle \delta B ^ {2} \right\rangle V T}{2 \mu_ {0}}. \tag {1}
$$

Here $\langle \delta B^2\rangle$ is the mean squared error of the measurement, $V$ is the sensed volume, $T$ is the duration of the measurement, and $\mu_0$ is the vacuum permeability\*.

A limit on $\bar{E}_R$ constrains sensitivity when measuring the field in a given space-time region, without reference to any other physical observable, nor to any resource. In contrast to other quantum sensing limits, this allows nothing to be traded for greater precision; it means that details of the field distribution are simply unmeasurable. Known limits on $E_R$ , derived from quantum statistical modeling, show that direct current (dc) superconducting quantum interference devices (dc SQUIDs)

(3, 5, 6), rubidium vapor magnetometers (7, 8), and immobilized spin precession sensors, e.g., nitrogen-vacancy centers in diamond (NVD) (4, 9), are all limited to $E_{R} \geq \alpha \hbar$ , where $\hbar$ is the reduced Planck constant and $\alpha$ is a number of order unity. These limits, though, are imposed by technology-specific mechanisms, not by a universal constraint on all sensor technologies (10).

A variety of exotic sensing techniques, including noble gas spin precession sensors (11-13), levitated ferromagnets (14, 15), and dissipationless superconducting devices (16-18), have been proposed to achieve $E_{R} < \hbar$ by evading specific relaxation mechanisms (10). If $E_{R} < \hbar$ can be achieved, it will break an impasse that has held since the early 1980s, when $E_{R} \approx \hbar$ was reached in dc SQUID sensors (6, 19).

In addition to resolving the question of whether $E_{R} \geq \hbar$ is universal, achieving $E_{R} < \hbar$ would open horizons in condensed matter physics (20) and neuroscience (21). For example, to enable single-shot discrimination of brain events, a magnetometer would need $\delta B \sim 1$ fT sensitivity to $T \sim 10$ ms events when measuring in $V \sim (3 \mathrm{~mm})^{3}$ volumes (22, 23), or $E_{R} \sim 1\hbar$ .

Here we study an exotic magnetometer technology, the single-domain spinor Bose-Einstein condensate (SDSBEC), that freezes out relaxation pathways due to collisions, dipolar

# Significance

Energy resolution per bandwidth $E_{R}$ is a cross-technology figure of merit that quantifies the combined spatial, temporal, and field resolution of a magnetic sensor. Today's best-developed magnetometer technologies, including superconducting quantum interference devices, spin-exchange relaxation-free Rb vapors, and nitrogen-vacancy centers in diamond, are limited by quantum noise to $E_{R} \gtrsim \hbar$ . Meanwhile, important sensing applications, e.g., noninvasive discrimination of individual brain events, would be enabled by $E_{R} < \hbar$ .

This situation has motivated proposals for sensors operating by new physical principles. Our result, $E_{R} = 0.075(16)\hbar$ , far beyond the best possible performance of established sensor technologies, confirms the potential of this class of proposed sensors. The result opens horizons for condensed matter, neuroscience, and tests of fundamental physics.

Author contributions: S.P.A. built the apparatus and performed the experiments with help from S.C.; P.G. and C.M. analyzed the data and performed the TWA simulations; R.Z.Z. performed Gross-Pitaevskii equation simulations; M.W.M. supervised the work and developed the TWA analysis; and S.P.A and M.W.M. wrote the manuscript with input from all authors.

The authors declare no competing interest.

This article is a PNAS Direct Submission.

This article is distributed under Creative Commons Attribution-NonCommercial-NoDerivatives License 4.0 (CC BY-NC-ND).

To whom correspondence may be addressed. Email: morgan.mitchell@icfo.eu.

Published February 7, 2022.

Check for updates

Downloaded from https://www.pnas.org by 218.70.106.126 on December 24, 2023 from IP address 218.70.106.126.

$^{*}$ A related definition, scaling as $E_{R} \propto A^{3/2}$ , $A \equiv$ active area, applies to planar sensors (3, 4).

PNAS 2022 Vol. 119 No. 6 e2115339119

https://doi.org/10.1073/pnas.2115339119

1 of 6

SDISAHd

![](dt=2026-03-07/ht=00/9195a5d7bea579eba5c4fe6bec1bebcd584efc925e0265082c67b286f035c3ad.jpg)

![](dt=2026-03-07/ht=00/27fbaaddd0164cd15de6119106951f044e6dfad601848e379a016fd3b368290a.jpg)

![](dt=2026-03-07/ht=00/bad55706a4d20fef1691b730f5d85355438d38abc1b7c59d1ffe496f7f29072e.jpg)

interactions, and also spin diffusion (24) and domain formation (25, 26), which occur in unconfined condensates. With a $^{87}\mathrm{Rb}$ SDSBEC, we find $E_{R} = 0.075(16)\hbar$ , far beyond what is possible, even in principle, with established technol
ogies (10, 27). Our results demonstrate the possibility of $E_{R} \ll \hbar$ sensors and motivate the study of other exotic sensor types.

To understand how the SDSBEC evades the $\hbar$ limit, it is instructive to first show why other spin precession sensors, which include NVD and alkali vapors, obey such a limit. The principle of operation of a spin precession sensor is represented in Fig. 1C: An ensemble of $N$ atoms is first initialized with its net spin $\mathbf{F}$ along the magnetic field $\mathbf{B}$ to be measured. The spin is then tipped by a radiofrequency pulse, making $\mathbf{F}$ orthogonal to $\mathbf{B}$ . The spins are allowed to precess for a time $T$ before the resulting precession angle $\theta = \gamma BT$ is detected, where $\gamma$ is the gyromagnetic ratio of the atomic species and $B = |\mathbf{B}|$ is the magnitude of the field. The resulting energy resolution per bandwidth is

$$
E _ {R} = \frac {V \left\langle \delta \theta^ {2} \right\rangle_ {F}}{2 \mu_ {0} \gamma^ {2} T} + \frac {V \left\langle \delta \theta^ {2} \right\rangle_ {\mathrm {R O}}}{2 \mu_ {0} \gamma^ {2} T}, \tag {2}
$$

where $\langle \delta \theta^2\rangle_F$ and $\langle \delta \theta^2\rangle_{\mathrm{RO}}$ are the angular variance due to intrinsic uncertainty of $\mathbf{F}$ and readout noise, respectively.

Readout noise can in principle be arbitrarily reduced using projective measurement, so we focus on the intrinsic spin noise. This scales as $\langle \delta \theta^2\rangle_F\propto N^{-1}$ and is minimized at the optimal readout time $T\approx T_2 / 2$ where $T_{2}$ is the transverse relaxation time. The quantum noise contribution to Eq. 2 thus scales as $1 / (nT_{2})$ ,where $n = N / V$ is the number density of spins. In ordinary spin systems, the relaxation rate $1 / T_{2}$ will grow proportionally to $n$ due to two-body decoherence processes, e.g., spin destruction collisions in vapors (8) or magnetic dipole-dipole coupling in NVD (9, 10). This density-coherence tradeoff ensures that $E_{R}$ has a finite lower bound (Energy Resolution Limit for Markovian Spin Systems).

To circumvent this limit, we implement a spin precession sensor with an SDSBEC. This ultracold sensor differs from the above in three important ways. First, because it is so cold, inelastic two-body interactions, including both short-range hyperfine-changing collisions and long-range dipole-dipole interactions, are energetically forbidden for a sensor operating in the ground hyperfine

state (28). Second, because of quantum degeneracy, the elastic two-body interactions (spin-independent and spin-dependent contact interactions) produce a coherent dynamics that does not raise the entropy of the many-body spin state (29). Third, in the single-domain regime, these coherent dynamics cannot reduce the net polarization through domain formation, as happens in extended SBECs (30). As we will show, $1 / T_{2}$ then contains no contribution $\propto n$ , and we escape the density-coherence trade-off.

To understand the SDSBEC sensitivity†, we compute $\langle \delta \theta^2\rangle_F$ including quantum statistical effects due to collisional interactions, which can importantly modify the spin distribution from its mean-field behavior (33). We employ the truncated Wigner approximation (TWA) (34, 35), previously applied to study spatial coherence in BECs (36). In the single-mode approximation (SMA), the quantum field describing the condensate factorizes into a spatial distribution $\phi_N(\mathbf{r})$ and a spinor field operator $\chi$ describing all atoms in the condensate.

$\chi \equiv (\hat{a}_{+1},\hat{a}_0,\hat{a}_{-1})^T$ where $\hat{a}_m$ are bosonic annihilation operators, such that $N\equiv \chi^{\dagger}\cdot \chi$ is the atomic number operator. $\phi_N(\mathbf{r})$ is the ground-state solution to the spin-independent part of the Hamiltonian in the Thomas-Fermi approximation and with $N$ atoms. We normalize $\phi_N$ such that $I_{2} = 1$ , where $I_{d}\equiv \int d^{3}\vec{r} |\phi_{N}(\vec{r})|^{d}$

The spinor field $\chi$ evolves under the SMA Hamiltonian (37)

$$
H _ {\mathrm {S M A}} = \frac {g}{2} \chi^ {\dagger} \mathbf {f} \chi \cdot \chi^ {\dagger} \mathbf {f} \chi + q \chi^ {\dagger} f _ {z} ^ {2} \chi , \tag {3}
$$

where $g \equiv g_2 I_4 \propto N^{-3/5}$ describes the spin-dependent interaction strength and the $q$ term describes the quadratic Zeeman

Downloaded from https://www.pnas.org by 218.70.106.126 on December 24, 2023 from IP address 218.70.106.126.

$^{\dagger}$ A direct measurement of the sensor's equivalent magnetic noise could in principle be made by placing the magnetometer in a shielded environment with magnetic noise below that of the sensor. To our knowledge, shielding at the required level, $\sim 50$ fT/√Hz at sub-Hz frequencies, has never been implemented in a cold-atom experiment and appears intrinsically challenging. As described below, the single-shot, optimized SDSBEC is sensitive to frequencies below $f = 1 / T \approx 0.29\mathrm{Hz}$ , while multishot measurements would be still slower.

At these low frequencies, magnetic shielding is limited by the innermost shield's thermal magnetization noise, with power spectral density $\propto 1 / f$ and typical values $\langle \delta B^2\rangle T = f^{-1}120\mathrm{fT}^2$ (31). For this reason, we base our sensitivity estimates on a combination of measured readout noise and calculations of the quantum noise dynamics in the SBEC using measured parameters. Due to the very clean nature of the BEC system, such calculations have proven reliable in other contexts (32).

2 of 6

PNAS

https://doi.org/10.1073/pnas.2115339119

Palacios Alvarez et al.

Single-domain Bose condensate magnetometer achieves energy resolution

per bandwidth below $\hbar$

shift, including contributions from the external field and from microwave or optical fields. The combined action of the $q$ and $g$ terms induces a shearing of the condensate's spin noise distribution from its initial coherent-state distribution. Losses occur at rate $dN / dt = -\Gamma_1N - \Gamma_3N^{9 / 5}$ , where $\Gamma_{1}$ describes the rate of collisions with background gas and $\Gamma_{3}$ is proportional to the three-body loss cross section. The evolution of the many-body spin state $\rho$ is described by the master equation $d\rho /dt = [H_{\mathrm{SMA}},\rho ] / (i\hbar) + \mathcal{L}[\rho ]$ , where $\mathcal{L}[\rho ]$ is the Liouvillian

$$
\mathcal {L} [ \rho ] = \sum_ {l} \kappa_ {l} \left(2 \hat {O} _ {l} \rho \hat {O} _ {l} ^ {\dagger} - \rho \hat {O} _ {l} ^ {\dagger} \hat {O} _ {l} - \hat {O} _ {l} ^ {\dagger} \hat {O} _ {l} \rho\right), \tag {4}
$$

and the jump operators $\hat{O}_l$ , with associated rates $\kappa_l$ , describe the various loss processes (Mode Shape, Interaction Strengths, and Jump Operators and Quantum Noise Evolution).

Fig. 2 shows the evolution of the spin noise contribution to $E_{R}$ over time as computed by TWA. For a given trapping potential and finite $\Gamma_1, q$ , and/or $\Gamma_3$ , the energy resolution shows a global minimum with $T$ .

To understand the in-principle limits of this $T$ -optimized noise level, we note the following: 1) $\Gamma_1$ can in principle be arbitrarily reduced through improved vacuum conditions, while $q$ can also be made arbitrarily small by compensating the contribution of the external field with microwave or optical dressing, leaving $\Gamma_3$ as the sole factor to introduce spin noise. 2) The noise effects of $\Gamma_3$ , which are a strong function of density, can also be made arbitrarily small, by increasing $r_{\mathrm{TF}}$ and $N$ to give a large, low-density condensate.

3) The corresponding increase in $V$ is more than offset by the increase in $T_2$ , such that $E_R \propto V / T_2$ tends toward zero. 4) At the same time, the SMA and TWA approximations become more accurate in this limit. We conclude that a low-density SBEC in a loose trapping potential can operate deep in th
e single-mode regime, suffer small three-body losses, and achieve $E_R \ll \hbar$ .

We now show that an SDSBEC magnetometer can in practice operate with $E_{R}$ well below $\hbar$ . The experimental configuration is illustrated in Fig. 1A and described in detail in Palacios et al. (29). In brief, a pure condensate of $^{87}\mathrm{Rb}$ atoms in the $F = 1$ manifold with an initial atom number $N_0 = 6.8(5)\times 10^4$ is produced by

![](dt=2026-03-07/ht=00/fd93ef1cc6a0f13ff126343aabe55554cc19f4a2b8ba29701b19b772510d1efb.jpg)

forced evaporation in a crossed-beam optical dipole trap. The condensate is initialized fully polarized along $\mathbf{B}$ by evaporation in the presence of a magnetic gradient, tipped by a radiofrequency pulse to be orthogonal to $\mathbf{B}$ , then allowed to precess for a time $T$ before readout, as depicted in Fig. 1C. A probe light tuned $258\mathrm{MHz}$ to the red of the $F = 1\rightarrow F^{\prime} = 0$ transition of the $\mathrm{D}_2$ line is used for nondestructive Faraday rotation measurement of the collective spin of the condensate (Fig. 1D).

Atom number is measured by time-of-flight absorption imaging. From atom-number decay we observe $\Gamma_{1} = 8.6(31) \times 10^{-2}\mathrm{s}^{-1}$ and $\Gamma_{3} = 1.0(6) \times 10^{-5}\mathrm{atom}^{-4/5}\mathrm{s}^{-1}$ one-body and three-body collision rates, respectively. The very small three-body loss rate allows us to approximate atomic losses as exponentially decaying with lifetime 7.1(2) s. In this approximation the resulting rate $dN/dt$ never differs by more than $4\%$ from the numerical solution when both $\Gamma_{1}$ and $\Gamma_{3}$ are included. The coherence time is found to be equal to the atomic lifetime in the trap and therefore $T_{2} = 7.1(2)$ s.

The curvature of the trapping potential is determined from the measured SBEC oscillation frequencies. We find $\omega_{1} / 2\pi = 67.2(10)\mathrm{Hz}$ , $\omega_{2} / 2\pi = 89.0(7)\mathrm{Hz}$ , and $\omega_{3} / 2\pi = 97.6(9)\mathrm{Hz}$ where the subscripts index the principal axes of the trap. For our number of atoms $N = 6.8(5)\times 10^{4}$ these correspond to Thomas-Fermi radii $r_{\mathrm{TF}}^{(1,2,3)} = 7.0(1)\mu \mathrm{m}$ , $6.20(9)\mu \mathrm{m}$ , and $6.00(9)\mu \mathrm{m}$ in the Thomas-Fermi approximation (Mode Shape, Interaction Strengths, and Jump Operators). This parabolic geometry defines the volume containing the entire condensate $V\equiv 4\pi r_{\mathrm{TF}}^{(1)}r_{\mathrm{TF}}^{(2)}r_{\mathrm{TF}}^{(3)} / 3 = 1,091(30)\mu \mathrm{m}^3$

As shown in Fig. 1D, measurements of the spin precession can be taken over several precession cycles with little damage to the polarization, allowing the precession angle to be estimated with readout noise $\langle \delta \hat{\theta}^2\rangle_{\mathrm{RO}} = 1.08(24)\times 10^{-4}\mathrm{rad}^2$ at the time of optimal readout $T = T_{2} / 2$ (Readout Noise). We note that $\langle \delta \hat{\theta}^2\rangle_{\mathrm{RO}}$ could be further reduced through improved probe-atom coupling and/or squeezed light (38, 39).

Combining the above we have volume $V = 1,091(30)\mu \mathrm{m}^3$ , readout noise $\langle \delta \hat{\theta}^2\rangle_{\mathrm{RO}} = 1.08(24)\times 10^{-4}\mathrm{rad}^2$ , and spin quantum noise $\langle \delta \theta^2\rangle_F = 1.46(100)\times 10^{-5}\mathrm{rad}^2$ . For an optimum readout time of $T = 3.5\mathrm{s}$ , these give a magnetic sensitivity of 72(8) fT and $E_{R} = 0.075(16)\hbar$ (Duty Cycle). This is a factor of 17 better than any previously reported value (24, 40, 41) and well beyond the level $E_{R}\approx \hbar$ that constrains the most advanced existing technologies.

In applying the TWA, we assumed the validity of the SMA. To check this, we integrate in time the three-dimensional Gross-Pitaevskii equation (Description of the Condensate) on a graphical processing unit, as described in refs. 42, 43. Spatially resolved polarization column densities are shown in Fig. 1 $B$ and $E$ and indicate fractional polarization defects at the $10^{-5}$ level. The defect $N - F_{\perp}$ of the condensate as a whole is of order 1 atom.

By vector addition, the contribution to the variance of the azimuth spin component $F_{\theta}$ is then no larger than the projection noise $\langle \delta F_{\theta}^{2}\rangle_{\mathrm{PN}} = N / 2$ and could be far smaller. These mean-field results, together with coherence measurements reported in (29), give a quantitative justification for the use of the SMA.

We extend the analysis to other $F = 1$ alkali species and find that some could perform still better than the $^{87}\mathrm{Rb}$ system studied here. Two considerations are relevant here. First, we note the conditions for single-mode dynamics: $r_{\mathrm{TF}} / \xi_s \ll 1$ and $r_{\mathrm{TF}} / \lambda \ll 1$ , where $\xi_s$ is the spin-healing length (37) and $\lambda$ is the threshold wavelength for spin wave amplification (44) (SMA Validity Conditions). In Fig.

3 we show $\max(r_{\mathrm{TF}} / \xi_s, r_{\mathrm{TF}} / \lambda)$ versus $V$ and $q$ and note that $^{87}\mathrm{Rb}$ and $^{23}\mathrm{Na}$ remain single-domain for smaller volumes and for stronger fields than do $^{7}\mathrm{Li}$ and $^{41}\mathrm{K}$ . We note also that the dynamical condition $r_{\mathrm{TF}} / \lambda \ll 1$ favors antiferromagnetic interactions, giving $^{23}\mathrm{Na}$ a marked advantage by this criterion. The second consideration concerns

Downloaded from https://www.pnas.org by 218.70.106.126 on December 24, 2023 from IP address 218.70.106.126.

Palacios Alvarez et al.

Single-domain Bose condensate magnetometer achieves energy resolution

per bandwidth below $\hbar$

PNAS

3 of 6

https://doi.org/10.1073/pnas.2115339119

SDISAHd

![](dt=2026-03-07/ht=00/9a0bda335ceb4921d898fe03bf8daf20d03a7f7452872e09147a32c33a4389b4.jpg)

![](dt=2026-03-07/ht=00/6875e3681f63e14f1aaf4983182f05c2a384d66557078546a6311c2f770a4f43.jpg)

![](dt=2026-03-07/ht=00/c4b3c43337a1affff75109eba9098c86d3e34741a4e033d85a4066521760c222.jpg)

![](dt=2026-03-07/ht=00/3e09ca30e61196d3ad2fee53b721597a4d094a1c20f953584922158f06983289.jpg)

the three-body recombination rate (45) $\Gamma_3 \propto \hbar a_0^4 / M$ , where $a_0$ is the s-wave scattering length for the channel of total spin zero. Relative to $^{87}\mathrm{Rb}$ , this rate in $^{7}\mathrm{Li}$ , $^{23}\mathrm{Na}$ , and $^{41}\mathrm{K}$ is a factor 25, 4, and 2 smaller, respectively, suggesting an advantage for these species when limited by three-body losses.

In conclusion, we have shown that an appropriately confined, quantum degenerate Bose gas, i.e., an SDSBEC, has a qualitative advantage over the best existing magnetic sensors as regards temporal, spatial, and field resolution, as summarized in the energy resolution per bandwidth $E_{R}$ .

Whereas the best-developed approaches to superconducting, hot vapor, and color center magnetometers are limited to $E_{R} \gtrsim \hbar$ , the SDSBEC, which retains a strong global response to an external field, while freezing out internal interactions that would otherwise produce depolarization, can operate with $E_{R}$ far below $\hbar$ . With a $^{87}\mathrm{Rb}$ SDSBEC, we have demonstrated $E_{R} = 0.075(16)\hbar$ , a factor of 17 improvement over the best previously reported (24, 40, 41) and well beyond the level that limits todays most advanced magnetic sensors.

$E_{R}$ in the demonstrated $^{87}\mathrm{Rb}$ system could be reduced with better light-atom coupling. Other alkali SBCs could also achieve smaller values for $E_{R}$ . The results show the promise of
a new generation of proposed sensors, including noble gas magnetometers (11-13), levitated ferromagnets (14, 15), and dissipationless superconducting devices (16-18), that operate by similar principles.

# Materials and Methods

Energy Resolution Limit for Markovian Spin Systems. We describe an ensemble of $N$ spin- $F$ atoms by the collective spin operator $\mathbf{F}$ , i.e., the sum of the vector spin operators for the individual atoms. $\mathbf{F}$ is initialized in a fully polarized state orthogonal to the magnetic field $\mathbf{B}$ . The spin angle precesses at a rate $\dot{\theta} = \gamma B$ , where $\gamma$ is the gyromagnetic ratio.

It is convenient to work with spin components in a frame rotating at the nominal Larmor frequency, such that a small change in angle can be expressed as $\delta \theta = \delta F_{\theta} / F_{\perp}$ , where $F_{\theta}$ is the azimuthal component and $F_{\perp}$ is the lever arm or spin component orthogonal to the axis of rotation and thus orthogonal to $\mathbf{B}$ . If a measurement of $F_{\theta}$ is made at time $T$ to infer $\theta$ and thus $B$ , the equivalent magnetic noise is $\langle \delta B^2\rangle = \langle \delta \theta^2\rangle /(\gamma^2 T^2)$ , by propagation of error.

If $F_{\perp}$ experiences Markovian relaxation, then $F_{\perp}$ at the time of measurement is $F_{\perp}(T) = FN\exp [-T / T_2]$ , where $T_{2}$ is the transverse relaxation time and $FN$ is the initial, full polarization. The initial, fully polarized state has azimuthal spin noise $\langle \delta F_{\theta}^{2}\rangle = FN / 2$ , i.e., the standard quantum limit. If $N$ does not decrease during the evolution (as is the case for color center and vapor phase ensembles), this describes a minimum noise for $F_{\theta}$ during the evolution.

We thus find $\langle \delta B^2\rangle T\geq \exp [2T / T_2] / (2\gamma^2 TFN)$ . Choosing $T$ to minimize the

right-hand side of this inequality, we find $T = T_{2} / 2$ and thus $\langle \delta B^{2} \rangle T \geq \exp [1] / (2\gamma^{2}T_{2}FN)$ . Including the sensor volume $V$ , the energy resolution is lower-bounded by $E_{R} \geq \exp [1] / (4\mu_{0}\gamma^{2}FT_{2}n)$ , where $n = N / V$ is the number density.

Writing the relaxation rate as $1 / T_{2} = A_{1}n^{0} + A_{2}n^{1} + \dots$ , $d$ -body interactions contribute to the $A_{d}$ term. When $A_{2}$ is nonzero, $E_{R} \propto A_{1}n^{-1} + A_{2}n^{0} + \dots$ is manifestly lower-bounded. First principle calculations for immobilized spin precession sensors (4) and models including measured spin relaxation rates for optimized Rb vapor magnetometers (8) show that these lower bounds are within a factor of 2 of $E_{R} = \hbar$ .

Description of the Condensate. A $F = 1$ spinor condensate with weak collisional interactions is well described by a three-component field $\psi_{\alpha}(\mathbf{r})$ evolving under the Hamiltonian

$$
H = H _ {\mathrm {S l}} + H _ {\mathrm {S D}}, \tag {5}
$$

where $H_{\mathrm{SI}}$ and $H_{\mathrm{SD}}$ are the spin-independent and spin-dependent parts, respectively. Summing over repeated indices, and omitting position dependence for clarity, these are

$$
H _ {S I} = \int d ^ {3} r \left(\psi_ {\alpha} ^ {\dagger} \left[ - \frac {\hbar^ {2} \nabla^ {2}}{2 M} + U \right] \psi_ {\alpha} + \frac {g _ {1}}{2} \psi_ {\alpha} ^ {\dagger} \psi_ {\beta} ^ {\dagger} \psi_ {\beta} \psi_ {\alpha}\right) \tag {6}
$$

$$
\begin{array}{l} H _ {S D} = \int d ^ {3} r \frac {g _ {2}}{2} \psi_ {\alpha} ^ {\dagger} \left(f _ {\eta}\right) _ {\alpha \beta} \psi_ {\beta} \psi_ {\gamma} ^ {\dagger} \left(f _ {\eta}\right) _ {\gamma \delta} \psi_ {\delta} + p \psi_ {\alpha} ^ {\dagger} \left(f _ {z}\right) _ {\alpha \beta} \psi_ {\beta} \\ + q \psi_ {\alpha} ^ {\dagger} \left(f _ {z} f _ {z}\right) _ {\alpha \beta} \psi_ {\beta}. \tag {7} \\ \end{array}
$$

Here $f_{\eta}$ is the matrix representing the single-atom spin projection operator onto the axis $\eta$ . In $H_{\mathrm{SD}}$ , the terms are ferromagnetic interaction, linear Zeeman, and quadratic Zeeman energies, respectively; $p = \hbar \gamma B$ ; and $q = (\hbar \gamma B)^2 / E_{\mathrm{hf}}$ , where $B$ is the field strength and $E_{\mathrm{hf}}$ is the hyperfine splitting energy.

S-wave scattering contributes the state-independent and state-dependent contact interactions, characterized by $g_1 \equiv 4\pi \hbar^2 (a_0 + 2a_2) / (3M)$ and $g_2 \equiv 4\pi \hbar^2 (a_2 - a_0) / (3M)$ , respectively. Here $M$ is the atomic mass and $a_0, a_2$ are the s-wave scattering lengths for the channels of total spin 0 and 2, respectively (46). We neglect the magnetic dipole-dipole interaction, which in $^{87}\mathrm{Rb}$ is orders of magnitude weaker than the contact interactions and vanishes identically for a single-mode spherical distribution.

Mode Shape, Interaction Strengths, and Jump Operators. In the Thomas-Fermi approximation (47), a pure condensate in a spherical harmonic potential has the mode function

$$
\left| \phi (r) \right| ^ {2} = \frac {1 5}{8 \pi r _ {\mathrm {T F}} ^ {3}} \left(1 - \frac {r ^ {2}}{r _ {\mathrm {T F}} ^ {2}}\right) \tag {8}
$$

for $r \leq r_{\mathrm{TF}}$ and zero otherwise, where $r$ is the radial coordinate, $r_{\mathrm{TF}} = \left[15g_1N / (4\pi M\omega^2)\right]^{1 / 5}$ is the Thomas-Fermi radius, and $\omega$ is the trap angular frequency. Because $r_{\mathrm{TF}} \propto N^{1 / 5}$ , the integrals $I_d$ that determine the effective strength of two- and three-body interactions are $I_4 \propto N^{-3 / 5}$ and $I_6 \propto N^{-6 / 5}$ , respectively. The rate of three-body collisions can then be written $\Gamma_3N^{9 / 5} \propto I_6N^3$ , such that atom losses are described by

$$
\frac {d N}{d t} = - \Gamma_ {1} N - \Gamma_ {3} N ^ {9 / 5}. \tag {9}
$$

We note that in this model, losses are independent of internal state. While this is well established for one-body losses, for three-body losses the state dependence is, to our knowledge, unknown. Two-body losses due to magnetic dipole-dipole scattering and spin-orbit interaction in second order (28) are energetically forbidden in the low-field scenario of interest here.

We use a set of jump operators that reproduces Eq. 9 while also respecting the symmetry of the loss process: one-body losses are described by $\hat{O}_m^{(1b)} = \hat{a}_m$ , $m \in \{-1,0,1\}$ , where $\hat{a}_m$ annihilates an atom in internal state $m$ , with strengths $\kappa_m^{(1b)} = \Gamma_1 / 2$ , while three-body losses are described by $\hat{O}_{mno}^{(3b)} = N^{-3/5}\hat{a}_m\hat{a}_n\hat{a}_o$ , $m,n,o \in \{-1,0,1\}$ , where $N \equiv (\hat{a}_{-1}^{\dagger}\hat{a}_{-1} + \hat{a}_0^\dagger\hat{a}_0 + \hat{a}_{+1}^\dagger\hat{a}_{+1})$ with strengths $\kappa_{mno}^{(3b)} = 5\Gamma_3 / 24$ .

Quantum Noise Evolution. We use the TWA (32, 34, 48) to compute the evolution of the spin distribution arising from the master equation $d\rho /dt = [H_{\mathrm{SMA}},\rho ] / (i\hbar) + \mathcal{L}[\rho ]$ . Our treatment follows that of Opanchuk et al. (49), restricted to a single spatial mode. In the TWA, the Wigner-Moyal equation describing the time evolution of the Wigner distribution is truncated at second order, such that an initially positive Wigner distribution remains positive, and the Wigner-Moyal equation becomes a Fokker-Planck equation\*

Downloaded from https://www.pnas.org by 218.70.106.126 on December 24, 2023 from IP address 218.70.106.126.

The approximation is believed valid for noncritical systems in which each simulated mode contains on average many particles (36, 50). This condition is very well satisfied here.

4 of 6

PNAS

https://doi.org/10.1073/pnas.2115339119

Palacios Alvarez et al.

Single-domain Bose condensate magnetometer achieves energy resolution

per bandwidth below $\hbar$

The Fokker-Planck equation describes the evolving probability distribution of a particle undergoing Brownian motion and as such can be described by a stochastic differential equation that is straightf
orward to integrate numerically.

We identify a complex-valued vector $\mathbf{c} = (c_{+1}, c_0, c_{-1})^T$ with the spinor field $\chi$ , and $c$ -number functions $O_m^{(1b)} = c_m$ , $m \in \{-1, 0, 1\}$ , $O_{mn\sigma}^{(3b)} = |c|^{-6/5} c_m c_n c_o$ , $m, n, o \in \{-1, 0, 1\}$ , with the jump operators $\hat{O}_m^{(1b)}$ and $\hat{O}_{mn\sigma}^{(3b)}$ , respectively. To account for the uncertainty of the initial state, a collection of starting points are chosen with values $\mathbf{c}_i = \mathbf{c}_0 + (z_{-1}, z_0, z_{+1})^T / \sqrt{2}$ , where $\mathbf{c}_0 = (1, \sqrt{2}, 1)^T / 2$ is the initial, fully $F_x$ -polarized state, $z_m = x_m + iy_m$ and $x_m, y_m$ are zero-mean unit variance Gaussian random variables. For the simulations shown in Fig. 2, we used 5,000 starting points.

Each initial point evolves by the (Itô) stochastic differential equation

$$
d c _ {m} = \left[ \frac {1}{i \hbar} \frac {\partial H}{\partial c _ {m} ^ {*}} - \sum_ {I} \kappa_ {I} \frac {\partial O _ {I} ^ {*}}{\partial c _ {m} ^ {*}} O _ {I} \right] d t + \sum_ {I} \sqrt {\kappa_ {I}} \frac {\partial O _ {I} ^ {*}}{\partial c _ {m} ^ {*}} d Z _ {I}, \tag {10}
$$

where $dZ = (dX + idY) / \sqrt{2}$ is a complex Wiener increment, in which $dX$ and $dY$ are independent Wiener increments, i.e., zero-mean normal deviates with variance $dt$ . Using the jump operators $O_{m}^{(1b)}$ , $O_{mn0}^{(3b)}$ defined above and adding their noise contributions in quadrature, we find

$$
d \mathbf {c} = \left[ \frac {2 g}{i \hbar} \sum_ {\alpha} \left(\mathbf {c} ^ {\dagger} f _ {\alpha} \mathbf {c}\right) f _ {\alpha} \mathbf {c} + \frac {q}{i \hbar} f _ {z} ^ {2} \mathbf {c} + A \right] d t + \mathbf {B} ^ {(\mathbf {c})} \cdot d \mathbf {Z}, \tag {11}
$$

where $dZ$ is a vector of three complex Wiener increments as defined above and

$$
A = - \frac {\Gamma_ {1}}{2} \mathbf {c} - \frac {\Gamma_ {3}}{2} | \mathbf {c} | ^ {8 / 5} \mathbf {c}, \tag {12}
$$

$$
\left(\boldsymbol {B} _ {j} ^ {(\mathbf {c})}\right) ^ {2} = \frac {\Gamma_ {1}}{2} + \frac {5 \Gamma_ {3}}{8} | \mathbf {c} | ^ {- 2 / 5} \left(| \mathbf {c} | ^ {2} + \frac {2 3}{2 5} | c _ {j} | ^ {2}\right). \tag {13}
$$

We use fourth-order Runge-Kutta explicit integration (51) to evaluate the trajectories. Statistics, e.g., $\langle F_x\rangle$ or $\operatorname{var}(F_y)$ , are computed as the corresponding population statistic on the set of evolved values, e.g., mean $\{\mathbf{c}_i^* f_x\mathbf{c}_i\}$ or $\operatorname{var}\{\mathbf{c}_i^* f_y\mathbf{c}_i\}$ . Because the calculation is run in a frame rotating at the Larmor frequency, the observed results are scattered about the ideal value $F_y = 0$ , and the atomic contribution to the angular mean squared error is simply $\langle \delta \theta^2\rangle_F = \langle F_y^2\rangle / \langle F_x\rangle^2$ .

Readout Noise. We experimentally prepare SBECs of $^{87}\mathrm{Rb}$ atoms in the $f = 1$ , $m = +1$ ground state under a bias field along direction $z$ and strength $B = 29\mu T$ , which induces Larmor precession at angular frequency $\omega_{L} = 2\pi \times 200\mathrm{kHz}$ . A radiofrequency $\pi /2$ pulse is applied to tip the spins to the xy plane. After a free evolution time $T$ we detect the spin precession by Faraday rotation, sending 60 pulses, each of 200-ns duration and containing $2\times 10^{6}$ photons, to observe rotation angles $\varphi_{i}$ at times $t_i$ , $i = 1,\dots ,60$ . Representative data are shown in Fig. 1D and are well described as a free induction decay signal. We parametrize the signal plus noise as

$$
\varphi_ {i} = G _ {1} \left[ \cos \left(\omega_ {L} \tau_ {i}\right) F _ {y} (T) + \sin \left(\omega_ {L} \tau_ {i}\right) F _ {x} (T) \right] e ^ {- \tau_ {i} / T _ {\text {s c a t}}} + \varphi_ {i} ^ {(R O)}, \tag {14}
$$

where $G_{1}$ is the effective atom-light coupling in radians per spin, $\tau_{i} \equiv t_{i} - T$ is the time since the start of probing, $\mathbf{F}(T)$ is the collective spin at the start of probing, $1 / T_{\mathrm{scat}}$ is the spin relaxation rate due to probe scattering, and $\varphi_{i}^{(\mathrm{RO})}$ is the readout noise. $G_{1} = 2.5(1) \times 10^{-7}$ rad/atoms is found by fully polarizing the atoms along $y$ , such that $F_{y} = N$ , and measuring $\varphi$ by Faraday rotation. $N$ is then measured by absorption imaging. $T_{\mathrm{scat}} = 29.7~\mu \mathrm{s}$ , found by fitting free induction decays as in Fig. 1D.

To determine the atomic precession angle from a free induction decay we define the angle estimator $\hat{\theta}_{e} \equiv \arctan[\hat{F}_{x}(T), \hat{F}_{y}(T)]$ in terms of the parameters $\hat{F}_{x}(T)$ , $\hat{F}_{y}(T)$ that make the best least-squares fit of Eq. 14 to a

given free induction decay $\{\varphi_i\}$ with the previously determined $G_{1}$ and $T_{\mathrm{scat}}$ . By propagation of errors, and due to the fit function's linear dependence on $F_{x}$ and $F_{y}$ , the estimator's mean squared error is

$$
\left\langle \delta \theta^ {2} \right\rangle_ {\mathrm {R O}} = \frac {\mathbf {r} ^ {\mathrm {T}} \cdot \Gamma^ {(\mathrm {R O})} \cdot \mathbf {r}}{N _ {0} ^ {2} \exp [ - 2 T / T _ {2} ]}, \tag {15}
$$

where $\mathbf{r} \equiv (\cos \theta, -\sin \theta)^T$ is a projector on the azimuthal direction and $\Gamma_{ij}^{(RO)}$ is the covariance matrix of the contribution made by $\varphi_i^{(RO)}$ to the fit parameters.

To evaluate Eq. 15, we note that $\Gamma^{(RO)}$ can be directly measured: we collect 40 traces $\{\varphi_i\}$ at time $T$ with no atoms in the trap. We then fit Eq. 14 using the $G_{1}$ and $T_{\mathrm{scat}}$ obtained previously. The result is

$$
\Gamma^ {(\mathrm {R O})} = \left[ \left( \begin{array}{c c} 1 8 4 & - 2 \\ - 2 & 2 2 2 \end{array} \right) \pm \left( \begin{array}{c c} 3 8 & 3 0 \\ 3 0 & 4 6 \end{array} \right) \right] \times 1 0 ^ {3}. \tag {16}
$$

Combining the above, we find the readout noise reaches its minimum value of $\langle \delta \theta^2\rangle_{\mathrm{RO}} = 1.08(24)\times 10^{-4}\mathrm{rad}^2$ when $T = T_{2} / 2$

SMA Validity Conditions. Two criteria for the validity of the SMA are found in the literature for the scenario of interest, in which a $F = 1$ condensate precesses about an orthogonal magnetic field. The first compares the ferromagnetic energy associated with a spatial overlap of the different $m_{F}$ states to the kinetic energy associated with a domain wall, to derive the condition $r_{\mathrm{TF}} \ll \xi_{s} \equiv 2\pi \hbar / \sqrt{2M|g_{2}|n}$ , where $\xi_{s}$ is known as the spin-healing length (52, 53).

The second criterion derives from a consideration of dynamical stability (44): in a plane wave scenario, spin wave perturbations to an initially uniform spin precessing at $\omega_{L} = p / \hbar$ are nonincreasing for wavelengths smaller than $\lambda_{\min} = 2\pi \hbar / \sqrt{2M(|g_{2}|n - g_{2}n + q)}$ . A second condition for the SMA is then $r_{\mathrm{TF}} \ll \lambda_{\min}$ . We note that for ferromagnetic interactions ( $g_{2} < 0$ ), but not for antiferromagnetic ones, this second condition is stricter than the first because $\lambda_{\min} < \xi_{s}$ .

Duty Cycle. While the main result of this work is a single-shot sensitivity, i.e., the noise level when measuring a field over a continuous interval $T$ , it is also interesting to consider averaging multiple sequential sensor readings to obtain a time-averaged estimate for the field. In this multishot scenario, the dead time between measurements must be accounted for in the energy resolution per bandwidth. Including the 30 s required to produce the next SBEC sample, we find a multishot sensitivity of 344(39) fT/√Hz and an energy resolution of $\langle \delta B^2 \rangle VT / (2\mu_0) = 0.48(11)\hbar$ , which is also significantly below $\hbar$ and well below any previously reported value.

Data Availability. Data and data analysis codes are available for download at Zenodo (DOI: 10.5281/zenodo.5751414) (54).

ACKNOWLEDGMENT
S. We thank Luca Tagliacozzo for insightful feedback.

This work was supported by H2020 Future and Emerging Technologies Quantum Technologies Flagship projects MACQSIMAL (Grant Agreement 820393) and ORANGE (Grant Agreement 820405); H2020 Marie Skłodowska-Curie Actions project ITN ZULF-NMR (Grant Agreement 766402); Spanish Ministry of Science "Severo Ochoa" Center of Excellence CEX2019-000910-S and project OCARINA (PGC2018-097056-B-I00 project funded by MCIN/AEI/10.13039/501100011033/FEDER "A way to make Europe"); Generalitat de Catalunya through the CERCA program; Agencia de Gestio d'Ajuts Universitaris i de Recerca Grant 2017-SGR-1354;

Secretaria d'Universitats i Recerca del Departament d'Empresa i Coneixement de la Generalitat de Catalunya, cofunded by the European Union Regional Development Fund within the ERDF Operational Program of Catalunya (project QuantumCat, ref.

001-P-001644); Fundacio Privada Cellex; Fundacio Mir-Puig; 17FUN03 USOQS, which has received funding from the EMPIR programme cofinanced by the Participating States and from the European Union's Horizon 2020 research and innovation programme; and CONACYT 255573 (Mexico) PAPIITIN105217 (UNAM).

Downloaded from https://www.pnas.org by 218.70.106.126 on December 24, 2023 from IP address 218.70.106.126.

Palacios Alvarez et al.

Single-domain Bose condensate magnetometer achieves energy resolution

per bandwidth below $\hbar$

PNAS

5 of 6

https://doi.org/10.1073/pnas.2115339119

SDISAHd

Downloaded from https://www.pnas.org by 218.70.106.126 on December 24, 2023 from IP address 218.70.106.126.

6 of 6

PNAS

https://doi.org/10.1073/pnas.2115339119

Palacios Alvarez et al.

Single-domain Bose condensate magnetometer achieves energy resolution

per bandwidth below $\hbar$