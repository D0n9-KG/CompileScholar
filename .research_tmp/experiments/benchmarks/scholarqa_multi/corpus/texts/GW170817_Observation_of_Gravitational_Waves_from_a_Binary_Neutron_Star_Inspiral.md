# Gravitational waves from neutron star excitations in binary inspirals

Alessandro Parisi $^{1}$ and Riccardo Sturani $^{2*}$

$^{1}$ ICTP-South American Institute for Fundamental Research, Instituto de Física Teórica (UNESP), 01140-070 São Paulo, Brazil

$^{2}$ International Institute of Physics (IIP), Universidade Federal do Rio Grande do Norte (UFRN) CP 1613, 59078-970 Natal-RN, Brazil

May 16, 2017

# ABSTRACT

In the context of binary inspiral of mixed neutron star - black hole systems, we investigate the excitation of the neutron star oscillation modes by the orbital motion. We study generic eccentric orbits and show that tidal interaction can excite the $f$ -mode oscillations of the star by computing the amount of energy and angular momentum deposited into the star by the orbital motion tidal forces via closed form analytic expressions. We study the $f$ -mode oscillations of cold neutron stars using recent microscopic nuclear equations of state, and we compute their imprint into the emitted gravitational waves.

Key words: Neutron stars, gravitational wave sources, relativistic star oscillations

# 1 INTRODUCTION

After the historical detections of gravitational waves by binary black holes Abbott et al. (2016), it is expected that mixed binaries composed of a neutron star (NS) and a black hole (BH) may be the next, qualitatively different type of source to be detected in the gravitational wave (GW) channel. At first approximation mixed NS-BH can be treated in General Relativity (GR) on equal footing as binary BH systems, however the presence of matter in the GW source may lead to new detectable astrophysical effects in the GW signal that are not expected to appear in the binary BH case like e.g. NS tidal deformations leaving an imprint in the GW signal Bildsten & Cutler (1992); Flanagan & Hinderer (2008) and breaking of the NS giving origin to a gamma ray burst or more general electromagnetic counterpart Lattimer & Schramm (1976), to name only the most studied effects.

Beside their direct phenomenological relevance, these effects carry information on the highly uncertain equation of state of the NS, thus making GW detection an invaluable probe of the internal structure of NSs. In this work we focus on a specific effect in GW signals: NS can be tidally deformed by the orbital motion in generic elliptic orbits, hence setting oscillations of the NS normal modes. The orbit being elliptical can induce resonant oscillations at a frequency much higher than the frequency scale set by the inverse of the orbital period, since in general NS oscillations are much higher than orbital frequency of inspiral binary systems.

Quantifying this phenomenon in light of the exciting prospect of a future GW detection has been the subject of extensive investigations in literature in a number of different contexts. The theoretical setup for studied such tidally induced NS oscillations has been provided in Thorne (1969); Press & Teukolsky (1977). In Fabian et al. (1975) it was originally proposed that tidal encounters between a NS and a main-sequence star might lead to the formation of X-ray binaries in globular clusters. In Shibata (1994) the effects of the tidal resonances for a circular orbital motion has been studied, with the result if the companion of a NS is a BH of mass $\geq 6M_{\odot}$ , the g-mode resonance is unimportant, while the f-mode resonance may affect the orbital evolution just before the merging. Rathore et al. (2005) considered the energy absorbed by tidal excitations in eccentric orbit (but not their imprint in the GW-form). Reisenegger & Goldreich (1994) compute the effect on the emitted GW phase of resonant mode excitation by the circular inspiral motion. Rotating NS we considered by Ho & Lai (1999) (including g-modes and r-modes) when the spin axis is aligned or anti-aligned with the orbital angular momentum axis. Carter & Luminet (1983) solved for the tidal deformation dynamics of a NS in an external field of a massive object and recently Chirenti et al. (2017) presented a framework for the discussion of binary NS and mixed NS-BH ones oscillation mode excitation and detection via the GWs observed by future GW detector as Einstein Telescope or Cosmic Explorer. Numerical results on the GW emission of tidally excited NS oscillations in the last stages of a coalescence have been given in Gold et al. (2012), and in Steinhoff et al. (2016) the imprint of resonant tidal on the gravitational waveform has been computed within the effective one body description of the two body orbital motion.

In the present paper we consider non-rotating NS with four different equations of states Akmal et al. (1998); Douchin & Haensel (2001); Walecka (1974); Bethe & Johnson (1974) with the goal of translating resonant excitations of various f-modes for NSs inspirating binary NS-BH systems that move in an elliptical orbit into quantitative prediction for the emitted GW-form.

Numerical simulations show that most of the energy released in gravitational waves is indeed transferred into f-modes, which are characterized by a wave-function free of nodes along the radial direction. We do not study the possibilities of exciting the g-modes because these modes are related to the presence of density discontinuities in the outer envelopes of NSs, see Finn (1987) and

# 2 A. Parisi, R. Sturani

Strohmayer (1993), density discontinuities in the inner core as a consequence of phase transitions at high density, as studied in Sotani et al. (2001), and/or thermal gradients as for a proto-NS, see e.g. Ferrari et al. (2003). In this paper we do not consider the possibility of having discontinuities of the density, moreover we focus on barotropic equations of state where the pressure depends only on the energy density, implying that all g-modes degenerate to zero frequency, hence we focus on the excitations of f-modes. Our study is based on the following simplifying assumptions:

(i) we neglect BH rotation, thus we treat the BH as a point particle with mass $M_{BH}$ ; (ii) the hydrodynamic stability of NS is computed using the Oppenheimer-Volkoff equations, but we use Newtonian equations to calculate the oscillation modes, see Appendix A; (iii) the NS does not rotate and we neglect viscous effects.

By implementing the formalism presented in Thorne (1969); Press & Teukolsky (1977) we find generic analytic expressions for the energy and angular momentum deposited into NS oscillations during the elliptic orbital motion, allowing to compute the mass quadrupole which is sourcing GW emission, and eventually comparing it with the orbital quadrupole.

The outline of this paper is as follows: in Sec. 2 we present the setup of the physical system under consideration, and we provide new analytic expressions for the dynamics of tidally induced NS oscillations, which are the main result of this paper. In Sec. 3 we analyze quantitatively their GW emission. Finally, conclusions for future detectability of NS oscillations in the GW channel are drawn in Sec. 4. We set the speed of light c = 1 throughout this paper.

# 2 COUPLING OF NEUTRON STAR OSCILLATION MODES TO ORBITAL MOTION

In this section we study the tidal excitation of NS oscillation modes in non-rotating stars in an elliptical orbit. Our analysis will be general, but the astrophysical case we have in mind is that of a binary NS-BH system. The idea to compute the energy deposited in stellar oscillations by the tidal gravitational field is first described by Turner (1977) and Press & Teukolsky (1977).

In this paper we use Newtonian linearized equations to calculate the oscillation modes. The use of Newtonian equations is consistent with our Newtonian description of tidal interactions. For the $f$ -mode, general relativistic effects are expected to modify our results of oscillation frequencies by not more than $GM_{*} / (R_{*}c^{2}) \sim 20$ per cent, see Lai (1994), where $M_{*}$ and $R_{*}$ are the mass and radius of the NS. We also neglect the spin $\Omega_s$ of the NS. When $\Omega_s \neq 0$ , the normal modes of the star get more complicated, especially when $\Omega_s$ becomes comparable to the mode frequencies Gaertig & Kokkotas (2008). For $\Omega_s \equiv 0$ the eigenmodes can be adequately approximated by those of a non-rotating spherical star, the basic equations that governing the oscillations of stars are discussed in more detail in Appendix A.

The NS oscillations are excited by tidal forces while the NS is bound in a binary system with black hole in an eccentric orbit whose evolution is driven by gravitational radiation. The distance D between two objects in an elliptic orbit can be parametrized by, see e.g. eq. (4.54) of Maggiore (2008),

$$
\mathcal {D} = \frac {a (1 - e ^ {2})}{1 + e \cos \psi} \tag {1}
$$

being a the semi-major axis and e the eccentricity (with $\psi = 0$ corresponding to the periastron), and the true anomaly $\psi$ is related to the eccentric anomaly u and time t via, see e.g. eqs. (4.57,58) of Maggiore (2008),

$$
\begin{array}{l} \beta \equiv u - e \sin u = \omega_ {0} t, \\ \cos \psi = \frac {\cos u - e}{1 - e \cos u}, \end{array} \tag {2}
$$

being $T$ the orbital period, $\omega_0 \equiv 2\pi / T$ with the following relationships holding among orbital parameters

$$
\dot {\psi} = \frac {\left[ G _ {N} M a (1 - e ^ {2}) \right] ^ {1 / 2}}{\mathcal {D} ^ {2}}, \tag {3}
$$

(where $M$ is the total mass of the binary system and $G_{N}$ the Newton constant) and the standard definition of the relativistic orbital parameter

$$
x \equiv (G _ {N} M \omega_ {0}) ^ {2 / 3} = \frac {G _ {N} M}{a}, \tag {4}
$$

the last equality holding only at Newtonian level.

In order to study quantitatively the effect of the gravitational force inducing oscillations into the NS and following the procedure outlined in Press & Teukolsky (1977), it is useful to expand the Newtonian potential in spherical harmonics, see e.g. eq. (3.70) of Jackson (1998), centered at the star as per

$$
\frac {1}{| \mathcal {D} - r |} = \sum_ {\ell = 0} ^ {\infty} \sum_ {m = - \ell} ^ {\ell} \frac {4 \pi}{2 \ell + 1} \frac {r ^ {\ell}}{\mathcal {D} ^ {\ell + 1}} Y _ {\ell m} ^ {*} (\theta , \phi) Y _ {\ell m} (\pi / 2, \psi), \tag {5}
$$

being $r, \theta, \phi$ coordinates of the mass elements of the NS, $\ell, |m| \leq \ell$ are the spherical harmonic indices and the orbital motion is assumed to be planar (no spin-induced precession). Using eq. (5) for elliptic orbit, it will be useful to expand $e^{im\psi}/D^{\ell+1}$ for generic $\ell$ into a Fourier series of the type

$$
\frac {e ^ {i m \psi}}{\mathcal {D} ^ {\ell + 1}} = \frac {1}{a ^ {\ell + 1}} \sum_ {j = 0} ^ {\infty} \left\{c _ {j} ^ {(\ell , m)} (e) \cos (j \beta) + i s _ {j} ^ {(\ell , m)} (e) \sin (j \beta) \right\}. \tag {6}
$$

The detailed calculation of the Fourier coefficients $c_{j}^{(\ell,m)}(e), s_{j}^{(\ell,m)}(e)$ and their analytic expressions are presented in Appendix B.

In order to perform an analytic quantitative analysis we borrow here the framework of Rathore et al. (2005), where NS oscillations are modeled as a series of damped harmonic oscillator displacements $x_{n}(t)$ driven by external force, that we can take purely monocromatic:

$$
\ddot {x} _ {n} (t) + 2 \frac {\dot {x} _ {n} (t)}{\tau_ {n}} + \omega_ {n} ^ {2} x _ {n} (t) = C _ {j} \cos (\omega_ {j} t) + S _ {j} \sin (\omega_ {j} t), \tag {7}
$$

where $\omega_{n}$ is the stellar mode frequency, $\tau_{n}$ its damping time $^{1}$ , $\omega_{j} \equiv j\omega_{0}$ is the j-th harmonic of the main orbital angular frequency $\omega_{0}$ , and $C_{j}, S_{j}$ the exciting force amplitude. $^{2}$ Eq. (7) admits the exact analytic solution

$$
\begin{array}{l} \left[ \left(\omega_ {j} ^ {2} - \omega_ {n} ^ {2}\right) ^ {2} + 4 \omega_ {j} ^ {2} / \tau_ {n} ^ {2} \right] x _ {n} (t) = \\ = \left(\omega_ {n} ^ {2} - \omega_ {j} ^ {2}\right) \left(C _ {j} \cos (\omega_ {j} t) + S _ {j} \sin (\omega_ {j} t)\right) \tag {8} \\ + \quad 2 \omega_ {j} / \tau_ {n} \left(C _ {j} \sin (\omega_ {j} t) - S _ {j} \cos (\omega_ {j} t)\right), \\ \end{array}
$$

the solution $x_{n}^{(h)}$ to the homogeneous equation being

$$
x _ {n} ^ {(h)} \propto e ^ {- t / \tau_ {n}} \cos \left[ \left(\omega_ {n} ^ {2} - 1 / \tau_ {n} ^ {2}\right) ^ {1 / 2} t + \phi_ {0} \right], \tag {9}
$$

leading to an average absorbed energy per unit of mass $\mathcal{E}$ per unit of time

$$
\dot {\mathcal {E}} = \frac {\left(C _ {j} ^ {2} + S _ {j} ^ {2}\right) \omega_ {j} ^ {2} / \tau_ {n}}{(\omega_ {j} ^ {2} - \omega_ {n} ^ {2}) ^ {2} + 4 \omega_ {j} ^ {2} / \tau_ {n} ^ {2}}. \tag {10}
$$

The NS oscillation vectors $\vec{\zeta}(t,\vec{r})$ satisfy an equation of the type see Kosovichev & Novikov (1992)

$$
\left(\rho \frac {d ^ {2}}{d t ^ {2}} + \mathcal {L}\right) \vec {\zeta} (t, \vec {r}) = - \rho \vec {\nabla} U (\vec {r}), \tag {11}
$$

where L is an operator characterizing the internal restoring force of the star. In order to apply this toy model of a damped harmonic oscillator to the tidally excited NS oscillation, we decompose the oscillation field $\vec{\zeta}(t,\vec{r})$ into normal modes with factorized time and space dependence:

$$
\vec {\zeta} (t, \vec {r}) = \sum_ {n, \ell , m} q _ {n \ell m} (t) \vec {\xi} _ {n \ell m} (\vec {r}), \tag {12}
$$

where we have added the spherical harmonics $\ell, m$ labels and the spatial mode eigenfunctions $\xi_{n\ell m}$ satisfy

$$
\left(\mathcal {L} - \rho \omega_ {n} ^ {2}\right) \vec {\xi} _ {n \ell m} = 0, \tag {13}
$$

allowing the identification of $\omega_{n}$ with the stellar frequency of the eigenmode. The differential equations the oscillation modes fields $\xi$ satisfy are summarized in Appendix A, which are solved for 4 different equations of state and 4 values of the central density of the NS, with the resulting mass, radius, frequency and damping times (the last two depending on $\ell$ ) are reported in Appendix C for $2 \leq \ell \leq 4$ .

It is also useful to expand the eigenmodes into a radial $(r)$ and a poloidal $(h)$ component

$$
\vec {\xi} _ {n \ell m} (\vec {r}) = \left(\xi_ {n \ell} ^ {(r)} (r) \hat {e} _ {r} + r \xi_ {n \ell} ^ {(h)} (r) \vec {\nabla}\right) Y _ {\ell m} (\theta , \phi), \tag {14}
$$

and impose the normalization condition $^{3}$

$$
\begin{array}{l} \int d ^ {3} x \rho (r) \vec {\xi} _ {n \ell m} ^ {*} \cdot \vec {\xi} _ {n ^ {\prime} \ell^ {\prime} m ^ {\prime}} \\ = \int d r r ^ {2} \rho (r) \left(\xi_ {n \ell} ^ {(r)} \xi_ {n ^ {\prime} \ell^ {\prime}} ^ {(r)} + \ell (\ell + 1) \xi_ {n \ell} ^ {(h)} \xi_ {n ^ {\prime} \ell^ {\prime}} ^ {(h)}\right) \delta_ {\ell , \ell^ {\prime}} \delta_ {m, m ^ {\prime}} \tag {15} \\ = \rho_ {0} R _ {*} ^ {5} \delta_ {n, n ^ {\prime}} \delta_ {\ell , \ell^ {\prime}} \delta_ {m, m ^ {\prime}}, \\ \end{array}
$$

where $\rho (r),\rho_0,R_*$ are respectively the density, central density and radius of the NS, and we used

$$
\begin{array}{l} \int d \Omega Y _ {\ell m} (\theta , \phi) Y _ {\ell^ {\prime} m ^ {\prime}} ^ {*} (\theta , \phi) = \delta_ {\ell , \ell^ {\prime}} \delta_ {m, m ^ {\prime}} \\ \int d \Omega r ^ {2} \vec {\nabla} Y _ {\ell m} (\theta , \phi) \cdot \vec {\nabla} Y _ {\ell^ {\prime} m ^ {\prime}} ^ {*} (\theta , \phi) = \ell (\ell + 1) \delta_ {\ell , \ell^ {\prime}} \delta_ {m, m ^ {\prime}}, \\ \int d \Omega \vec {r} \cdot \vec {\nabla} Y _ {\ell m} (\theta , \phi) \vec {r} \cdot \vec {\nabla} Y _ {\ell^ {\prime} m ^ {\prime}} ^ {*} (\theta , \phi) = \delta_ {\ell , \ell^ {\prime}} \delta_ {m, m ^ {\prime}}, \tag {16} \\ \end{array}
$$

and the integral of products of spherical harmonics with unequal number of derivatives vanish for any $\ell, m, \ell', m'$ .

By multiplying both members of eq.(11) by $\rho(r)\xi_{n\ell m}^{*}(\vec{r})$ , substituting the expansion in eq. (5), and integrating over the NS volume the mode $q_{n\ell m}(t)$ is singled out and it satisfies an equation of the type (7):

$$
\begin{array}{l} \ddot {q} _ {n \ell m} (t) + \frac {2}{\tau_ {n \ell}} \dot {q} _ {n \ell m} (t) + \omega_ {n} ^ {2} q _ {n \ell m} (t) = \frac {G _ {N} M _ {B H}}{a ^ {3}} \left(\frac {R _ {*}}{a}\right) ^ {\ell - 2} Q _ {n \ell} W _ {\ell m} \\ \times \sum_ {j} \left(c _ {j} ^ {(\ell + 1, m)} (e) \cos (j \beta) + i s _ {j} ^ {(\ell + 1, m)} (e) \sin (j \beta)\right) \\ \end{array}
$$

(17)

$^{3}$ Note that with the normalization chosen $\xi_{n\ell}^{(r,h)}$ have dimension of length, $q_{n\ell m}$ is dimension-less. However the normalization can be arbitrarily chosen without affecting physical results, our choice has the advantage of making following formulae simpler.

where

$$
\begin{array}{l} W _ {\ell m} \equiv \frac {4 \pi}{2 \ell + 1} Y _ {\ell m} (\pi / 2, 0), \\ Q _ {n \ell} \equiv \frac {1}{\rho_ {0} R _ {*} ^ {\ell + 3}} \int_ {0} ^ {R _ {*}} d r r ^ {2} \rho (r) \ell r ^ {\ell - 1} \left(\xi_ {n \ell} ^ {(r)} + (\ell + 1) \xi_ {n \ell} ^ {(h)}\right), \tag {18} \\ \end{array}
$$

$M_{BH}$ is the black hole mass. Note that the r.h.s of eq.(17) is complex, but given the symmetries of the $c,s$ coefficients: $W_{\ell m} = (-1)^{\ell}W_{\ell -m}$ (and $W_{\ell m} = 0$ if $\ell ,m$ have different parity), $c_j^{\ell ,m} = c_j^{\ell , - m}$ , $s_j^{\ell ,m} = -s_j^{\ell , - m}$ the sum of $\sum_{m}q_{n\ell m}\times Y_{\ell m}$ returns a real quantity. The modes $q_{n\ell m}$ thus satisfy an equation of the type (7) with the coefficients $C_j,S_j$ replaced by

$$
\left(C _ {j}, S _ {j}\right)\rightarrow \frac {G _ {N} M _ {B H}}{a ^ {3}} \left(\frac {R _ {*}}{a}\right) ^ {\ell - 2} Q _ {n \ell} W _ {\ell m} \left(c _ {j} ^ {(\ell , m)} (e), s _ {j} ^ {(\ell , m)} (e)\right). \tag {19}
$$

These expressions will be needed in sec. 3 to compute the time varying quadrupole associated to these oscillations, source of GWs. The rate of energy (per unit of mass, per unit NS radius) absorbed by each oscillation modes can be read from eq. (10) by inserting the above values of $C_{j}, S_{j}$ , summing over n, j > 0, $\ell \geq 2$ and $|m| \leq \ell$ , the rate of absorbed energy via tidal mechanism $\dot{E}_{*}$ being

$$
\begin{array}{l} \dot {E} _ {*} = \sum_ {j} \dot {E} _ {j} = \rho_ {0} R _ {*} \left(\frac {R _ {*}}{a}\right) ^ {4} \left(\frac {G _ {N} M _ {B H}}{a}\right) ^ {2} \quad \sum_ {j, n, \ell , m} \left(c _ {j} ^ {(\ell , m) 2} + s _ {j} ^ {(\ell , m) 2}\right) \left(\frac {R _ {*}}{a}\right) ^ {2 \ell - 4} \\ \times \quad Q _ {n \ell} ^ {2} W _ {\ell m} ^ {2} \frac {\omega_ {j} ^ {2} / \tau_ {n \ell}}{(\omega_ {j} ^ {2} - \omega_ {n \ell} ^ {2}) ^ {2} + 4 \omega_ {j} ^ {2} / \tau_ {n \ell} ^ {2}}. \tag {20} \\ \end{array}
$$

The contribution from individual $j$ modes to the rate of energy absorption is plotted in fig. 1 after being divided by the factor

$$
\begin{array}{l} K \equiv \rho_ {0} R _ {*} \left(\frac {R _ {*}}{a}\right) ^ {4} \left(\frac {G _ {N} M _ {B H}}{a}\right) ^ {2} \frac {(G _ {N} M \omega_ {0}) ^ {2}}{\omega_ {0 2}} \\ \simeq 1. 5 \cdot 1 0 ^ {- 1 4} \frac {M _ {\odot}}{\sec} \left(\frac {x}{0 . 0 1}\right) ^ {9} \left(\frac {\rho_ {0}}{1 0 ^ {1 5} \mathrm{gr} / \mathrm{cm} ^ {3}}\right) \left(\frac {R _ {*}}{1 0 \mathrm{Km}}\right) ^ {5} \left(\frac {M _ {B H}}{4 M _ {\odot}}\right) ^ {2} \left(\frac {M}{6 M _ {\odot}}\right) ^ {- 6}, \tag {21} \\ \end{array}
$$

where $\omega_{02} = \omega_{n\ell}$ for $n = 0$ , $\ell = 2$ . Factorizing the absorbed energy rate by the quantity $K$ has the virtue of making $\dot{E}_j / K$ dimensionless and independent on the relativistic parameter $x$ (as long as the orbital frequency does not hit a resonance with $\omega_j \equiv j\omega_0$ ) and mildly dependent on $\rho_0, a$ .

In fig. 2 we report the absorbed energy rate $\dot{E}_{*}$ normalized by

$$
\dot {E} _ {G W 0} \equiv \frac {3 2}{5 G _ {N}} \eta^ {2} x ^ {5} \tag {22}
$$

(being $\eta\equiv M_{*}M_{BH}/(M_{*}+M_{BH})^{2}$ the reduced mass of the orbital system), which is the expression of the leading order in x of the GW emission rate at zero eccentricity from a binary inspiral, making visually easier the comparison between GW radiated energy $\dot{E}_{GW}$ and $\dot{E}_{*}$ . For $\dot{E}_{GW}$ we use the 3PN formula taken from Arun et al. (2008), see also sec. 10.3 of Blanchet (2014).

The absorbed angular momentum can be computed in a similar way, following Lai (1994), where it is noted that the variation of angular momentum

$$
\dot {L} _ {*} = - \int d ^ {3} x (\rho_ {0} + \delta \rho) (\hat {z} \cdot \vec {r} \times \vec {\nabla} U) \tag {23}
$$

we can derive in our setup

$$
\begin{array}{l} \dot {L} _ {*} = \sum_ {n \ell} q _ {n \ell m} (t) \int d ^ {3} x \vec {\nabla} \cdot (\rho_ {0} \vec {\xi} _ {n \ell m}) \frac {\partial U}{\partial \psi} \\ = \sum_ {n \ell m} q _ {n \ell m} (t) \int d ^ {3} x \vec {\nabla} \cdot (\rho_ {0} \vec {\xi} _ {n \ell m}) \frac {G _ {N} M _ {B H}}{a} \left(\frac {r}{a}\right) ^ {\ell} W _ {\ell m} i m \\ \times Y _ {\ell m} ^ {*} (\theta , \phi) \sum_ {j} \left(c _ {j} ^ {(\ell , m)} (e) \cos (j \beta) + i s _ {j} ^ {(\ell , m)} (e) \sin (j \beta)\right), \tag {24} \\ \end{array}
$$

![](images/2bf648f4fc09ef156b03e7a0d425a78b9b67710e2fb2d41d3f64582709199a90.jpg)

<details>
<summary>line</summary>

| j  | Series 1 | Series 2 | Series 3 | Series 4 | Series 5 | Series 6 | Series 7 | Series 8 |
|----|----------|----------|----------|----------|----------|----------|----------|----------|
| 0  | 10^-5    | 10^-5    | 10^-5    | 10^-5    | 10^-5    | 10^-5    | 10^-5    | 10^-5    |
| 5  | 10^-3    | 10^-3    | 10^-3    | 10^-3    | 10^-3    | 10^-3    | 10^-3    | 10^-3    |
| 10 | 10^-2    | 10^-2    | 10^-2    | 10^-2    | 10^-2    | 10^-2    | 10^-2    | 10^-2    |
| 15 | 10^-1    | 10^-1    | 10^-1    | 10^-1    | 10^-1    | 10^-1    | 10^-1    | 10^-1    |
| 20 | 10^0     | 10^0     | 10^0     | 10^0     | 10^0     | 10^0     | 10^0     | 10^0     |
| 25 | 10^1     | 10^1     | 10^1     | 10^1     | 10^1     | 10^1     | 10^1     | 10^1     |
| 30 | 10^2     | 10^2     | 10^2     | 10^2     | 10^2     | 10^2     | 10^2     | 10^2     |
| 35 | 10^3     | 10^3     | 10^3     | 10^3     | 10^3     | 10^3     | 10^3     | 10^3     |
| 40 | 10^4     | 10^4     | 10^4     | 10^4     | 10^4     | 10^4     | 10^4     | 10^4     |
| 45 | 10^5     | 10^5     | 10^5     | 10^5     | 10^5     | 10^5     | 10^5     | 10^5     |
| 50 | 10^6     | 10^6     | 10^6     | 10^6     | 10^6     | 10^6     | 10^6     | 10^6     |
</details>

Figure 1. Distribution of the energy per unit of mass absorbed by the fundamental NS oscillation mode divided by the quantity $K$ (defined in eq. (21)) as a function of the harmonic of the fundamental mode $j$ in eccentric orbits for 9 equally spaced values of eccentricity (from $e_0 = 0$ in red, through $e_i = i / 10$ until $e_8 = 0.8$ in yellow). For each value of eccentricity two curves are reported, for $x = 0.01$ and $x = 0.07$ ( $x$ defined in eq. (4)). For the largest value of $x$ the resonant absorption peaks are visible for $e = 0.5, 0.6, 0.7, 0.8$ , as $\bar{j}\omega_0 = \bar{j}x^{3/2} / (G_N M) \simeq 18.1\mathrm{kHz}(\bar{j}/34)(6.965\mathrm{M}_{\odot}/\mathrm{M})(\mathrm{x}/0.07)^{3/2}$ where for this plot $M_{BH} = 5M_{\odot}$ and we used the equation of state A (APR) of Akmal et al. (1998) and central density $\rho_0 = 1.5\times 10^{15}\mathrm{gr/cm^3}$ , see tab. C1. In this case the $n = 0$ $f$ -mode has frequency $v_f^{\ell = 2} = 2.888\mathrm{kHz}$ (we have verified that for $x < 0.07$ the NS is safe from tidal braking, whose condition requires $\mathcal{D} \lesssim 0.3R_{NS}(M_{NS} / M_{BH})^{1/3}x^{1/2}(M / M_{BH})^{1/2}$ , see Vallisneri (2000)). Plots for the other equations of states described in App. C are shown in fig. E1 and are qualitatively similar.

![](images/c2c6d518cfe7e995bdec9727f658851a17db64bdc37a65fec8c4ae9b09a69b75.jpg)

<details>
<summary>line</summary>

| eccentricity | E_GW     | x = 0.07 | x = 0.01 |
| ------------ | -------- | -------- | -------- |
| 0.0          | 1.0      | 1.0      | 1e-8     |
| 0.1          | 1.0      | 1.0      | 1e-8     |
| 0.2          | 1.0      | 1.0      | 1e-8     |
| 0.3          | 1.0      | 1.0      | 1e-8     |
| 0.4          | 1.0      | 1.0      | 1e-8     |
| 0.5          | 1.0      | 1.0      | 1e-8     |
| 0.6          | 1.0      | 1.0      | 1e-7     |
| 0.7          | 1.0      | 1.0      | 1e-6     |
| 0.8          | 1.0      | 1.0      | 1e-5     |
| 0.9          | 1.0      | 1.0      | 1e-4     |
</details>

Figure 2. Rate of energy absorbed $\dot{E}_{*}$ as a function of eccentricity, with $M_{BH} = 5M_{\odot}$ , NS with equation of state A (APR) of Akmal et al. (1998) for different values of the central density $\rho_0 = (1.5, 1.2, 0.99) \times 10^{15} \mathrm{gr/cm}^3$ , lines of increasing thickness shows results for increasing $\rho_0$ . For comparison we also plot the GW luminosity for the two values of $x$ , all functions are divided by the Newtonian GW luminosity at zero eccentricity $\dot{E}_{GW0}$ given by eq. (22). Plots for the other equations of states described in App. C are shown in fig. E2 and are qualitatively similar.

![](images/cf82819aff620355596ade107db6ded99f09154ff4660685b54198a6380b6480.jpg)

<details>
<summary>line</summary>

| eccentricity | L_GW       | x=0.07     | x=0.01     |
| ------------ | ---------- | ---------- | ---------- |
| 0.0          | 1.0        | 1e-10      | 1e-14      |
| 0.1          | 1.0        | 1e-10      | 1e-14      |
| 0.2          | 1.0        | 1e-10      | 1e-14      |
| 0.3          | 1.0        | 1e-10      | 1e-14      |
| 0.4          | 1.0        | 1e-9       | 1e-13      |
| 0.5          | 1.0        | 1e-8       | 1e-12      |
| 0.6          | 1.0        | 1e-7       | 1e-11      |
| 0.7          | 1.0        | 1e-6       | 1e-10      |
| 0.8          | 1.0        | 1e-5       | 1e-9       |
| 0.9          | 1.0        | 1e-4       | 1e-8       |
</details>

Figure 3. Rate of angular momentum absorbed as a function of eccentricity, same parameters as in fig. 2. Here $\dot{L}_{GW}$ is the Newtonian angular momentum loss in GWs for small eccentricities $\dot{L}_{GW} = \frac{32}{5}\eta^2 M\frac{x^{1/2}}{(1 - e^2)^2}\left(1 + \frac{7}{8}e^2\right)$ and $\dot{L}_{GW0} = \dot{L}_{GW}|_{e=0}$ .

where in the last passage we have inserted the expansion of eqs. (5,6) and derived by parts inside the integral. In this form the angular momentum absorption rate by NS oscillations can be rewritten as:

$$
\begin{array}{l} \dot {L} _ {*} = \rho_ {0} R _ {*} \left(\frac {G _ {N} M _ {B H}}{a}\right) ^ {2} 2 \sum_ {j, n, \ell , m > 0} m c _ {j} ^ {(\ell , m)} s _ {j} ^ {(\ell , m)} \left(\frac {R _ {*}}{a}\right) ^ {2 \ell} \\ \times Q _ {n \ell} ^ {2} W _ {\ell m} ^ {2} \frac {\omega_ {j} / \tau_ {n \ell}}{\left(\omega_ {j} ^ {2} - \omega_ {n \ell} ^ {2}\right) ^ {2} + 4 \omega_ {j} ^ {2} / \tau_ {n \ell} ^ {2}}. \tag {25} \\ \end{array}
$$

In fig. 3 the absorbed angular momentum rate $\dot{L}_{*}$ normalized by the leading order expression in $x$ of $\dot{L}_{GW0} \equiv 32/5M\eta^{2}x^{7/2}$ is reported for various values of the relativistic parameter $x$ and the eccentricity $e$ . The values of $\dot{L}$ are negligible with respect to $\dot{L}_{GW}$ and given the typical moment of inertia of a NS ( $\sim 10^{45}$ gr cm $^{2}$ , see book of Haensel et al. (2007)), the induced rotation on the NS is also negligibly small.

# 3 GRAVITATIONAL WAVE EMISSION

We have seen in the previous section that the energy absorbed in by the NS is very small compared to the orbital energy at moderate eccentricity values ( $e \lesssim 0.6$ ), hence such absorption will not alter in any significant way the chirping signal. However the energy absorbed will set oscillations in the neutron star that gives rise to a time varying quadrupole, which will in turn generate GWs with a significantly different pattern that the GWs associated to the decaying orbital motion.

The general expression for the GW in the TT gauge is given by, see e.g. eq. (3.275) of Maggiore (2008),

$$
h _ {i j} ^ {T T} (t, r) = \frac {1}{r} G _ {N} \sum_ {\ell = 2} ^ {+ \infty} \sum_ {m = - \ell} ^ {\ell} \left[ u _ {\ell m} \left(T _ {\ell m} ^ {E 2}\right) _ {i j} + v _ {\ell m} \left(T _ {\ell m} ^ {B 2}\right) _ {i j} \right] \tag {26}
$$

where $u_{\ell m}$ ( $v_{\ell m}$ ) is linearly related to the $\ell$ -th time derivative of the mass (momentum) multipole moments. The leading-order contribution to radiation reaction comes from the mass quadrupole term, for which it is (see e.g. sec. 3 of Maggiore (2008))

$$
u _ {2 m} = \frac {1 6}{1 5} \pi \sqrt {3} \ddot {Q} ^ {i j} \mathcal {Y} _ {i j} ^ {2 m ^ {*}}, \tag {27}
$$

being $Y_{i_{1}\ldots i_{\ell}}^{\ell m}$ the tensor spherical harmonics and $Q^{ij} \equiv \int d^{3}x\rho x^{j}x^{j}$ is the standard quadrupole mass moment in Cartesian coordinates. It will be convenient to express the leading order GW amplitude in terms of the spherical components $Q_{m}$ of the quadrupole, related to their Cartesian counterpart via

$$
\begin{array}{r c l} Q _ {2 m} & \equiv & \frac {8 \pi}{1 5} Q _ {i j} \left(\mathcal {Y} _ {i j} ^ {2 m}\right) ^ {*}, \\ Q _ {i j} & = & \sum_ {| m | \leq 2} Q _ {2 m} \mathcal {Y} _ {i j} ^ {2 m}, \end{array} \tag {28}
$$

leading to (explicit expressions of $\ell = 2$ tensor spherical harmonics are reported in app. D)

$$
u _ {2 m} = 2 \sqrt {3} \ddot {Q} _ {m}. \tag {29}
$$

We now have all the ingredients to relate the leading GW source $u_{2m}$ to the NS tidal oscillations via

$$
Q _ {* 2 m} = \frac {8 \pi}{1 5} \int \rho r ^ {2} Y _ {2 m} ^ {*} d ^ {3} x, \tag {30}
$$

that in terms of the displacement vector introduced in eqs. (11,12) can be expressed as, see Ushomirsky et al. (2000), by

$$
\begin{array}{l} \frac {1 5}{8 \pi} Q _ {* 2 m} = \int (\rho_ {0} + \delta \rho) r ^ {2} Y _ {2 m} ^ {*} d ^ {3} x \\ { = } { - \sum _ { n } \int \vec { \nabla } \cdot ( \rho _ { 0 } \vec { \zeta } _ { n 2 m } ) r ^ { 2 } Y _ { 2 m } ^ { * } d ^ { 3 } x } \\ \simeq q _ {0 2 m} (t) \left(2 \int_ {0} ^ {R _ {*}} \rho_ {0} \left\{\xi_ {0 2} ^ {(r)} + 3 \xi_ {0 2} ^ {(h)} \right\} r ^ {3} d r - \rho_ {0} \xi_ {0 2} ^ {(r)} r ^ {4} \Big | _ {0} ^ {R _ {*}}\right), \tag {31} \\ \end{array}
$$

where an integration by parts has been performed in the last step, the explicit expression of $\vec{\zeta}_{n2m}(t,\vec{x})$ has been inserted and only the n=0 contribution has been considered. since we analyzed only the f-mode. Observing that the boundary term is numerically smaller than the integral term, substituting the solution of eq. (17) and considering only the resonant contribution for $\omega_{\bar{j}}\simeq\omega_{02}$ the NS average quadrupole value can be written as

$$
\begin{array}{l} \langle Q _ {* 2 2} ^ {2} \rangle^ {1 / 2} \simeq \frac {2 \sqrt {2} \pi}{1 5} \rho_ {0} R _ {*} ^ {5} Q _ {0 2} ^ {2} W _ {2 2} \frac {G _ {N} M _ {B H}}{a ^ {3}} \frac {\tau_ {0 2}}{\omega_ {\bar {j}}} \left[ \left(c _ {\bar {j}} ^ {(2, 2)}\right) ^ {2} + \left(s _ {\bar {j}} ^ {(2, 2)}\right) ^ {2} \right] ^ {1 / 2} \\ \simeq \frac {4 \sqrt {2} \pi}{1 5} \frac {\left(\rho_ {0} R _ {*} ^ {5} \tau_ {0 2}\right) ^ {1 / 2}}{\omega_ {0 2}} Q _ {0 2} \sqrt {\dot {E} _ {\bar {j}} ^ {(\ell = 2)}} \\ \simeq 1 0 ^ {- 2} M _ {\odot} \mathrm{km} ^ {2} \left(\frac {\rho_ {0}}{1 0 ^ {1 5} \mathrm{gr/cm} ^ {3}}\right) ^ {1 / 2} \left(\frac {\mathrm{R} _ {*}}{1 0 \mathrm{km}}\right) ^ {5 / 2} \\ \times \left(\frac {\dot {E} _ {j} ^ {(l = 2)}}{1 0 ^ {- 8} M _ {\odot} / \sec}\right) ^ {1 / 2} \left(\frac {\tau_ {0 2}}{0 . 1 \sec}\right) ^ {1 / 2} \left(\frac {\omega_ {0 2}}{1 8 \mathrm{kHz}}\right) ^ {- 1}. \tag {32} \\ \end{array}
$$

The quantity directly related to GW emission, $u_{2m}^{(NS)}$ , follows straightforwardly via eq. (29). In fig. 4 we report the contribution to the second time derivative of the quadrupole (divided by the reduced mass of the binary system) and as a comparison the (magnified) second derivative of the quadrupole associated to n = 0, $\ell = 2$ NS oscillations during an ordinary binary inspiral in which the orbit shrinks due to GW back reactions.

For comparison, we also report in fig. 5 the time evolution of the displacement $q_{0\ell \ell} \ell = 2,3,4$ along the inspiral phase.

# 4 CONCLUSIONS

In this paper we have developed and presented a framework able to perform analytic and quantitative study of the excitations of a neutron star in an inspiralling binary system of arbitrary eccentricity. We have computed the energy and the angular momentum deposited into stellar mode oscillations by the tidal field via closed form analytic formulae. The amount of energy absorbed by the neutron star in a given mode depends on the overlap of the tidal force field with the displacement field of the mode, hence it requires solving the equilibrium equations of a neutron star, done here in the Newtonian approximation. We focused our analysis on the fundamental f-mode of a non-relativistic star, finding the rate of energy absorbed and angular momentum as a function of eccentricity and of the period of the inspiral orbital, when f-mode can be in resonance with higher harmonics of the main orbital frequency.

![](images/82a112f60d733501b24bdf392d89afe1c945972378d0fb235b44cf7d49fb3fd2.jpg)

<details>
<summary>line</summary>

| tη/(G_N M) | orbital | 10^3 Q̃_22 |
| ---------- | ------- | --------- |
| -8000      | ~0.2    | ~0.0      |
| -7500      | ~0.2    | ~0.0      |
| -7000      | ~0.2    | ~0.0      |
| -6500      | ~0.2    | ~0.0      |
| -6000      | ~0.2    | ~0.0      |
| -5500      | ~0.2    | ~0.0      |
| -5000      | ~0.2    | ~0.0      |
| -4500      | ~0.2    | ~0.0      |
| -4000      | ~0.2    | ~0.1      |
| -3500      | ~-0.3   | ~-0.3     |
</details>

Figure 4. Second derivative of the quadrupole $\ddot{Q}_{22}$ divided by the reduced mass $\mu \equiv \eta M$ : contribution from orbital dynamics compared with (magnified) contribution from the NS oscillation $Q_{*22}$ for an inspiral with initial conditions $x_{i} = 0.04$ , $e_{i} = 0.4$ , $M_{BH} = 5M_{\odot}$ and parameter for the NS given by equation of state B (SLy4) Douchin & Haensel (2001) with $\rho_0 = 2\times 10^{15}\mathrm{gr / cm^3}$ .

![](images/29f8cf4b298486a115336bd08d82af2e30f79198fe314c494f8698858a85f8eb.jpg)

<details>
<summary>line</summary>

| t(G_N M / η) | l=2       | l=3       | l=4       |
| ------------ | --------- | --------- | --------- |
| -4200        | 0.05      | 0.00      | 180       |
| -4150        | 0.00      | 0.00      | 190       |
| -4100        | 0.05      | 0.00      | 200       |
| -4050        | 0.00      | 0.00      | 210       |
| -4000        | 0.05      | 0.00      | 220       |
| -3950        | 0.00      | 0.00      | 230       |
| -3900        | 0.10      | 0.00      | 240       |
</details>

Figure 5. Given the same parameters of fig. 4, here are displayed the f-mode displacements $q_{0\ell\ell}$ for $\ell = 2, 3, 4$ (magnified by a factor $10^{3}$ ). Also shown are the main gravitational wave frequency $f_{GW} \equiv \omega_{0}/\pi$ and the eccentricity along the inspiral dynamics considered.

As a future development of this work, we intend to extend our

analysis to the General Relativistic equilibrium equations of a rotating neutron star, with the inclusion of r-mode and g-modes, and considering a not barotropic equation of state: such modes have lower frequency values than the f-mode, and can therefore be excited at resonance in an elliptical orbit earlier in the inspiral phase. The phenomenological impact of the computations presented here relies on the signature that neutron star oscillations will imprint onto the gravitational signals of an inspiral binary system. Despite being sub-dominant with respect to the gravitational wave sourced by the orbital motion, the detailed features of the star oscillation bears invaluable information on its equation of state and density, allowing to make a bridge to the nuclear physics ruling its equilibrium. Since it is expected in the near future that third generation gravitational wave detector could observe signals from binary systems involving neutron star at signal-to-noise ratio of order $10^{2}$ or more, see e.g. Punturo et al. (2010), and that such detection will involve the observation of hundred of thousand gravitational wave cycles during the inspiral of a binary system for a time stretch of order of several days, the quantitative prediction of the modification of the inspiral signal, even at very low level, will have an impact on the physics outcome of the detection.

# ACKNOWLEDGMENTS

The authors wish to thank C. Chirenti for useful discussions. The work of AP has been supported by the FAPESP grant 2016/00096-6, RS has been supported by FAPESP grant 2012/14132-3.

# References

Abbott B. P., et al., 2016, Phys. Rev., X6, 041015   
Akmal A., Pandharipande V. R., Ravenhall D. G., 1998, Phys. Rev., C58, 1804   
Arun K. G., Blanchet L., Iyer B. R., Qusailah M. S. S., 2008, Phys. Rev., D77, 064035   
Balbinski E., Schutz B. F., 1982, Mon. Not. R. ast. Soc., 200, 43P   
Bethe H. A., Johnson M. B., 1974, Nucl. Phys., A230, 1   
Bildsten L., Cutler C., 1992, ApJ, 400, 175   
Blanchet L., 2014, Living Rev. Rel., 17, 2   
Carter B., Luminet J.-P., 1983, Astro. Astrophys, 121, 97   
Chirenti C., Gold R., Miller M. C., 2017, Astrophys. J., 837, 67   
Douchin F., Haensel P., 2001, Astron. Astrophys., 380, 151   
Dziembowski W. A., 1971, Acta Astronomica, 21, 289   
Fabian A. C., Pringle J. E., Rees M. J., 1975, Mon. Not. R. Astr. Soc, 172, 15p   
Ferrari V., Miniutti G., Pons J. A., 2003, Mon. Not. Roy. Astron. Soc., 342, 629   
Finn L. S., 1987, Mon.Not.R.Astr.Soc, 227, 265   
Flanagan É. É., Hinderer T., 2008, Phys. Rev., D77, 021502   
Gaertig E., Kokkotas K. D., 2008, Phys. Rev., D78, 064063   
Gold R., Bernuzzi S., Thierfelder M., Brugmann B., Pretorius F., 2012, Phys. Rev., D86, 121501   
Haensel P., Zdunik J. L., 2008, Astron. Astrophys., 480, 459   
Haensel P., Potekhin A. Y., Yakovlev D. G., 2007, Neutron stars 1: Equation of state and structure. Springer   
Ho W. C. G., Lai D., 1999, Mon. Not. Roy. Astron. Soc., 308, 153   
Jackson D., 1998, Classical Electrodynamics. Wiley   
Kosovichev A. G., Novikov I. D., 1992, MNRAS, 258, 715   
Lai D., 1994, "Mon. Not. Roy. Astron. Soc.", 270, 611   
Lattimer J. M., Schramm D. N., 1976, ApJ, 210, 549   
M.Abramowitz Stegun I., 1964, Handbook of mathematical functions. Dover   
Maggiore M., 2008, Gravitational Waves. Oxford University Press   
Press W., Teukolsky S., 1977, Astrophys. J., 213, 183

Punturo M., et al., 2010, Class. Quant. Grav., 27, 194002   
Rathore Y., Blandford R. D., Broderick A. E., 2005, Mon. Not. Roy. Astron. Soc., 357, 834   
Reisenegger A., Goldreich P., 1994, Astrophys. J.   
Shibata M., 1994, Prog. Theor. Phys., 91, 871   
Sotani H., Tominaga K., Maeda K.-I., 2001, Phys. Rev. D, 65, 024010   
Steinhoff J., Hinderer T., Buonanno A., Taracchini A., 2016, Phys. Rev., D94, 104028   
Strohmayer T. E., 1993, The Astrophysical Journal, 417, 273   
Thorne K. S., 1969, Astrophys. J., 158, 997   
Turner M., 1977, ApJ, 216, 914   
Ushomirsky G., Cutler C., Bildsten L., 2000, Mon. Not. Roy. Astron. Soc., 319, 902   
Vallisneri M., 2000, Phys. Rev. Lett., 84, 3519   
Walecka J. D., 1974, Annals Phys., 83, 491

# APPENDIX A: FOUR FIRST-ORDER LINEAR DIFFERENTIAL EQUATIONS OF NON-RADIAL OSCILLATIONS

The normal modes of a spherical star can be labeled by spherical harmonic indices $\ell$ and m, and by a “radial quantum number” n. In spherical coordinates the Lagrangian displacement $\xi$ of a fluid element is given by

$$
\xi_ {n \ell m} = \left[ \xi_ {n \ell} ^ {(r)} (r), \xi_ {n \ell} ^ {(h)} (r) \frac {\partial}{\partial \theta}, \frac {\xi_ {n \ell} ^ {(h)} (r)}{\sin \theta} \frac {\partial}{\partial \phi} \right] Y _ {\ell m} (\theta , \phi) e ^ {i \sigma t} \tag {A1}
$$

where $Y_{\ell m}$ denotes a spherical harmonic; and $\sigma$ denotes the pulsation angular frequency. The oscillation is assumed to be adiabatic, we ignore the thermal evolution of the NS, for simplicity we use the Newtonian description in the Dziembowski (1971) formulation, in this case the equations reduce to a system of four first-order differential equations with four dimensionless variables, given by:

$$
y _ {1} = \frac {\xi_ {n \ell} ^ {(r)}}{r}, \quad y _ {2} = \frac {1}{g r} \left(\frac {p ^ {\prime}}{\rho} + \Phi^ {\prime}\right) = \frac {\sigma^ {2}}{g} \xi_ {n \ell} ^ {(h)}, \tag {A2}
$$

$$
y _ {3} = \frac {\Phi^ {\prime}}{g r}, \quad y _ {4} = \frac {1}{g} \frac {d \Phi^ {\prime}}{d r}, \tag {A3}
$$

Here, the meanings of the symbols are as follows: $p'$ and $\Phi'$ are the radial part of the Eulerian perturbation to the pressure p and the gravitational potential $\Phi$ , respectively; r is the distance from the center of the star, $\rho$ is the density, and $g \equiv Gm(r)/r^{2}$ is the local acceleration due to gravity. The system of differential equations that governs the linear adiabatic oscillations of stars is then given by:

$$
r \frac {d y _ {1}}{d r} = \left(V _ {g} - 1 - \ell\right) y _ {1} + \left[ \frac {\ell (\ell + 1)}{c _ {1} \omega^ {2}} - V _ {g} \right] y _ {2} + V _ {g} y _ {3} \tag {A4}
$$

$$
r \frac {d y _ {2}}{d r} = (c _ {1} \omega^ {2} - A ^ {*}) y _ {1} + (3 - U + A ^ {*} - \ell) y _ {2} - A ^ {*} y _ {3} \tag {A5}
$$

$$
r \frac {d y _ {3}}{d r} = (3 - U - \ell) y _ {3} + y _ {4} \tag {A6}
$$

$$
r \frac {d y _ {4}}{d r} = A ^ {*} U y _ {1} + U V _ {g} y _ {2} + \left[ \ell (\ell + 1) - U V _ {g} \right] y _ {3} - (U + \ell - 2) y _ {4} \tag {A7}
$$

Where

$$
V _ {g} = - \frac {1}{\Gamma_ {1}} \frac {d \ln p}{d \ln r} = \frac {g r}{c _ {s} ^ {2}}, \quad A ^ {*} = \frac {1}{\Gamma_ {1}} \frac {d \ln p}{d \ln r} - \frac {d \ln \rho}{d \ln r}, \quad U \equiv \frac {d \ln m (r)}{d \ln r} = \frac {4 \pi \rho r ^ {3}}{m (r)}, \tag {A8}
$$

$$
c _ {1} \equiv \frac {r ^ {3}}{R _ {*} ^ {3}} \frac {M _ {*}}{m (r)}, \quad \Gamma_ {1} = \left(\frac {\partial \ln p}{\partial \ln \rho}\right) _ {S}, \quad \omega^ {2} = \frac {R _ {*} ^ {3}}{G _ {N} M _ {*}} \sigma^ {2}. \tag {A9}
$$

Here $\Gamma_{1}$ is the first adiabatic exponent, $c_{s}$ is the sound speed, $m(r)$ is the concentric mass, $M_{*}$ and $R_{*}$ are the total mass and radius of the star, respectively, and $G_{N}$ is the gravitational constant. There are four boundary conditions, the inner boundary conditions at $r = 0$ are:

$$
\left\{ \begin{array}{c} c _ {1}   \omega^ {2} y _ {1} - \ell y _ {2} = 0 \\ \ell y _ {3} - y _ {4} = 0 \end{array} \right.,
$$

the outer boundary conditions at $r = R_{*}$ are:

$$
\left\{ \begin{array}{c} y _ {1} - y _ {2} + y _ {3} = 0 \\ (\ell + 1) y _ {3} + y _ {4} = 0 \end{array} \right..
$$

The two central boundary conditions require that the two divergences involved, $\nabla \cdot \xi_{n\ell}^{(r)}, \nabla \cdot \Phi'$ , remain finite. At the surface we require $\delta P/P$ to be finite and $\Phi'$ , the gravitational force per unit mass, to be continuous across the perturbed surfaces. The above equations and boundary conditions constitute an eigenvalue problem for the eigenvalue $\sigma$ .

The expression for the damping time due to emission of gravitational waves in the Newtonian case see Thorne (1969); Balbinski & Schutz (1982) is given by:

$$
\tau_ {n \ell} \equiv \frac {(\ell - 1) [ (2 \ell + 1) ! ! ] ^ {2}}{\ell (\ell + 1) (\ell + 2)} \left(\frac {\sigma}{2 \pi G}\right) \left(\frac {c}{\sigma}\right) ^ {2 \ell + 1} \frac {\int_ {0} ^ {R _ {*}} d r \rho r ^ {2} \left[ \xi_ {n \ell} ^ {(r)} (r) ^ {2} + \ell (\ell + 1) \xi_ {n \ell} ^ {(h)} (r) ^ {2} \right]}{\left\{\int_ {0} ^ {R _ {*}} d r \rho r ^ {\ell + 1} \left[ \xi_ {n \ell} ^ {(r)} (r) + (\ell + 1) \xi_ {n \ell} ^ {(h)} (r) \right] \right\} ^ {2}} \tag {A10}
$$

where $n!! = 1 \cdot \cdot (n - 4)(n - 2)n.$

# APPENDIX B: EXPANSION OF THE FOURIER COEFFICIENTS

Expanding in eq. (6) we have

$$
c _ {j} ^ {(\ell , m)} (e) = \frac {c _ {j}}{\pi (1 - e ^ {2}) ^ {\ell + 1}} \int_ {- \pi} ^ {\pi} \cos (m \psi) (1 + e \cos \psi) ^ {\ell + 1} \cos (j \beta) d \beta , \tag {B1}
$$

for $\ell \geq 0, |m| \leq l$ , where we used that $\psi$ is an odd function of time, hence $\cos \psi (\sin \psi)$ is an even (odd) function of time, $c_j = 1$ for $j \neq 0$ , and $c_0 = 1/2$ .

In order to expand $\cos(m\psi(t))$ into sums of terms of the type $\cos(n\beta)$ it is useful to express it in terms of powers of $\cos(\psi)$ via M.Abramowitz & Stegun (1964)

$$
\cos (m \theta) = T _ {m} (\cos (\theta)), \tag {B2}
$$

where $T_{m}$ is the Chebyshev polynomial of order r and it has the form

$$
T _ {m} (x) = \sum_ {k = 0} ^ {[ m / 2 ]} t _ {r} ^ {(s)} x ^ {m - 2 k}, \tag {B3}
$$

begin $[x]$ the integer part of $x$ . Using the standard relationships between eccentric anomaly $\psi$ , true anomaly $u$ and time $t$ , see sec. 2, one finds

$$
\begin{array}{r c l} 1 + e \cos \psi & = & \frac {1 - e ^ {2}}{1 - e \cos u}, \\ d \beta & = & (1 - e \cos u) d u, \end{array} \tag {B4}
$$

to obtain

$$
c _ {j} ^ {(\ell , m)} = \frac {2 c _ {n}}{\pi} \int_ {0} ^ {\pi} \sum_ {k = 0} ^ {[ m / 2 ]} t _ {m} ^ {(k)} \frac {(\cos u - e) ^ {m - 2 k}}{(1 - e \cos u) ^ {m - 2 k + \ell}} \cos (j u - j e \sin u) d u. \tag {B5}
$$

In order to perform this integral we use the standard Taylor-expansions

$$
\begin{array}{r c l} (1 - x) ^ {n} & = & \sum_ {k = 0} ^ {n} (- 1) ^ {k} \frac {n !}{k ! (n - k) !} x ^ {k}, \\ \frac {1}{(1 - x) ^ {n}} & = & \sum_ {k = 0} ^ {\infty} \frac {(n + k - 1) !}{k ! (n - 1) !} x ^ {k}, \end{array} \tag {B6}
$$

to write

$$
\begin{array}{l} c _ {j} ^ {(\ell , m)} (e) = \frac {2 c _ {j}}{\pi} \int_ {0} ^ {\pi} \sum_ {k = 0} ^ {[ m / 2 ]} t _ {m} ^ {(k)} (- e) ^ {m - 2 k} \sum_ {p = 0} ^ {m - 2 k} \frac {(m - 2 k) !}{p ! (m - 2 k - p) !} (- 1) ^ {p} \left(\frac {\cos u}{e}\right) ^ {p} \sum_ {n = 0} ^ {\infty} \frac {(m - 2 k + \ell + n - 1) !}{n ! (m - 2 k + \ell - 1) !} (e \cos u) ^ {n} \cos (j u - j e \sin u) d u \\ = \frac {2 c _ {j}}{\pi} \int_ {0} ^ {\pi} \sum_ {k = 0} ^ {[ m / 2 ]} \sum_ {p = 0} ^ {m - 2 k} \sum_ {n = 0} ^ {\infty} (- 1) ^ {p + m} t _ {m} ^ {(k)} \frac {(m - 2 k) !}{p ! (m - 2 k - p) !} \frac {(m - 2 k + \ell + n - 1) !}{n ! (m - 2 k + \ell - 1) !} e ^ {m - 2 k + n - p} (\cos u) ^ {p + n} \cos (j u - j e \sin u) d u, \tag {B7} \\ \end{array}
$$

and then we use the DeMoivre formula

$$
\cos^ {n} (u) = \frac {1}{2 ^ {n}} \sum_ {k = 0} ^ {n} \frac {n !}{k ! (n - k) !} \cos (n - 2 k) u, \tag {B8}
$$

to get to

$$
\begin{array}{l} c _ {j} ^ {(\ell , m)} (e) = \frac {c _ {j}}{\pi} \int_ {0} ^ {\pi} \sum_ {k = 0} ^ {[ m / 2 ]} \sum_ {p = 0} ^ {m - 2 k} \sum_ {n = 0} ^ {\infty} \sum_ {q = 0} ^ {p + n} (- 1) ^ {p + m} \frac {t _ {m} ^ {(k)}}{2 ^ {p + n - 1}} \quad \frac {(m - 2 k) !}{p ! (m - 2 k - p) !} \frac {(m - 2 k + \ell + n - 1) !}{n ! (m - 2 k + \ell - 1) !} \frac {(p + n) !}{q ! (p + n - q) !} \tag {B9} \\ \times e ^ {m - 2 k + n - p} \cos [ (p + n - 2 q) u ] \cos (j u - j e \sin u) d u. \\ \end{array}
$$

Finally using the integral representation of the Bessel functions

$$
J _ {n} (z) = \frac {1}{\pi} \int_ {0} ^ {\pi} \cos (n u - z \sin u) d u, \tag {B10}
$$

and the standard trigonometric identity

$$
2 \cos \alpha \cos \beta = \cos (\alpha + \beta) + \cos (\alpha - \beta), \tag {B11}
$$

one gets to

$$
\begin{array}{l} c _ {j} ^ {(\ell , m)} (e) = c _ {j} (- e) ^ {m} \sum_ {k = 0} ^ {[ m / 2 ]} \sum_ {p = 0} ^ {m - 2 k} \sum_ {n = 0} ^ {\infty} \sum_ {q = 0} ^ {p + n} (- 1) ^ {p} e ^ {n - 2 k - p} \frac {t _ {m} ^ {(k)}}{2 ^ {p + n}} \frac {(m - 2 k) !}{p ! (m - 2 k - p) !} \frac {(m - 2 k + \ell + n - 1) !}{n ! (m - 2 k + \ell - 1) !} \quad \frac {(p + n) !}{q ! (p + n - q) !} \tag {B12} \\ \times \left(J _ {n + p + j - 2 q} (j e) + J _ {j - p - n + 2 q} (j e)\right). \\ \end{array}
$$

Analogously for $s_j^{(\ell,m)}(e)$ , one can use the Chebyshev polynomial of the second kind $U_{n}(x)$ satisfying the equation

$$
\sin (m \theta) = U _ {m - 1} (\cos \theta) \sin \theta = \sin \theta \sum_ {k = 0} ^ {[ (m - 1) / 2 ]} u _ {m - 1} ^ {(k)} (\cos \theta) ^ {m - 1 - 2 k}, \tag {B13}
$$

Table C1. Data for the equation of state A (APR) Akmal et al. (1998), and Haensel & Zdunik (2008) for the crust. 

<table><tr><td> $\rho_0(\text{gr/cm}^3)$ </td><td>R(km)</td><td> $M(M_\odot)$ </td><td> $\nu_f^{\ell=2}(\text{kHz})$ </td><td> $\nu_f^{\ell=3}(\text{kHz})$ </td><td> $\nu_f^{\ell=4}(\text{kHz})$ </td><td> $|Q_{02}|$ </td><td> $|Q_{03}|$ </td><td> $|Q_{04}|$ </td></tr><tr><td> $1.5 \times 10^{15}$ </td><td>11.132</td><td>1.965</td><td>2.888</td><td>3.742</td><td>4.420</td><td>2.321</td><td>2.437</td><td>2.613</td></tr><tr><td> $1.2 \times 10^{15}$ </td><td>11.433</td><td>1.704</td><td>2.741</td><td>3.456</td><td>4.033</td><td>2.258</td><td>2.482</td><td>2.501</td></tr><tr><td> $9.9 \times 10^{14}$ </td><td>11.603</td><td>1.408</td><td>2.384</td><td>3.071</td><td>3.602</td><td>2.323</td><td>2.594</td><td>2.653</td></tr></table>

Table C2. Data for the equation of state B (SLy4) Douchin & Haensel (2001) 

<table><tr><td> $\rho_0(\text{gr/cm}^3)$ </td><td>R(km)</td><td> $M(M_\odot)$ </td><td> $v_f^{\ell=2}(\text{kHz})$ </td><td> $v_f^{\ell=3}(\text{kHz})$ </td><td> $v_f^{\ell=4}(\text{kHz})$ </td><td> $|Q_{02}|$ </td><td> $|Q_{03}|$ </td><td> $|Q_{04}|$ </td></tr><tr><td> $2.0 \times 10^{15}$ </td><td>10.615</td><td>1.994</td><td>3.300</td><td>4.143</td><td>4.829</td><td>2.148</td><td>2.372</td><td>2.443</td></tr><tr><td> $1.6 \times 10^{15}$ </td><td>11.017</td><td>1.884</td><td>3.024</td><td>3.808</td><td>4.461</td><td>2.180</td><td>2.439</td><td>2.446</td></tr><tr><td> $1.2 \times 10^{15}$ </td><td>11.435</td><td>1.634</td><td>2.654</td><td>3.372</td><td>3.967</td><td>2.270</td><td>2.989</td><td>2.508</td></tr></table>

Table C3. Data for the equation of state C Walecka (1974) 

<table><tr><td> $\rho_0(\text{gr/cm}^3)$ </td><td>R(km)</td><td> $M(M_\odot)$ </td><td> $v_f^{\ell=2}(\text{kHz})$ </td><td> $v_f^{\ell=3}(\text{kHz})$ </td><td> $v_f^{\ell=4}(\text{kHz})$ </td><td> $|Q_{02}|$ </td><td> $|Q_{03}|$ </td><td> $|Q_{04}|$ </td></tr><tr><td> $1.0 \times 10^{15}$ </td><td>13.639</td><td>2.472</td><td>2.491</td><td>3.139</td><td>3.673</td><td>2.093</td><td>2.365</td><td>2.465</td></tr><tr><td> $6.0 \times 10^{14}$ </td><td>13.902</td><td>1.727</td><td>2.043</td><td>2.602</td><td>3.050</td><td>2.281</td><td>2.300</td><td>2.449</td></tr><tr><td> $5.0 \times 10^{14}$ </td><td>13.677</td><td>1.311</td><td>1.817</td><td>2.355</td><td>2.782</td><td>2.261</td><td>2.351</td><td>2.622</td></tr></table>

to obtain

$$
\begin{array}{l} s _ {j} ^ {(\ell , m)} (e) = \frac {2 (1 - e ^ {2}) ^ {1 / 2}}{\pi} \int_ {0} ^ {\pi} \sin u \sum_ {k = 0} ^ {[ (m - 1) / 2 ]} u _ {m - 1} ^ {(k)} \frac {(\cos u - e) ^ {m - 1 - 2 k}}{(1 - e \cos u) ^ {m - 2 k + \ell}} \sin (j u - j e \sin u) d u \\ = - \frac {2 (1 - e ^ {2}) ^ {1 / 2}}{\pi} \int_ {0} ^ {\pi} \sum_ {k = 0} ^ {[ (m - 1) / 2 ]} \sum_ {p = 0} ^ {m - 1 - 2 k} \sum_ {n = 0} ^ {\infty} \sum_ {q = 0} ^ {p + n} (- 1) ^ {p + m} e ^ {m - 1 - 2 k + n - p} \frac {u _ {m} ^ {(k)}}{2 ^ {p + n}} \tag {B14} \\ \times \frac {(m - 2 k - 1) !}{p ! (m - 2 k - 1 - p) !} \frac {(m - 2 k + \ell + n - 1) !}{n ! (m - 2 k + \ell - 1) !} \frac {(p + n) !}{q ! (p + n - q) !} \sin u \cos [ (p + n - 2 q) u ] \sin (j u - j e \sin u) d u. \\ \end{array}
$$

Now using

$$
\begin{array}{l} 2 \sin \alpha \cos \beta = \sin (\alpha + \beta) + \sin (\alpha - \beta), \\ 2 \sin \alpha \sin \beta = \cos (\alpha - \beta) - \cos (\alpha + \beta), \\ \end{array}
$$

one finally obtains

$$
\begin{array}{l} s _ {j} ^ {(\ell , m)} (e) = (1 - e ^ {2}) ^ {1 / 2} (- e) ^ {m} \sum_ {k = 0} ^ {[ (m - 1) / 2 ]} \sum_ {p = 0} ^ {m - 1 - 2 k} \sum_ {n = 0} ^ {\infty} \sum_ {q = 0} ^ {p + n} (- 1) ^ {p} e ^ {n - 2 k - p} \frac {u _ {m - 1} ^ {(k)}}{2 ^ {p + n + 1}} \\ \times \frac {(m - 2 k - 1) !}{p ! (m - 2 k - 1 - p) !} \frac {\left(m ^ {\prime} - 2 k + \ell^ {\prime} + n - 1\right) !}{j ! (m - 2 k + \ell - 1) !} \frac {(p + n) !}{q ! (p + n - q) !} \tag {B16} \\ \times \left[ J _ {n + p + j + 1 - 2 q} (j e) + J _ {j - p - n + 1 + 2 q} (j e) - J _ {n + p + j - 1 - 2 q} (j e) - J _ {j - p - n - 1 + 2 q} (j e) \right]. \\ \end{array}
$$

# APPENDIX C: NEUTRON STAR EQUATIONS OF STATE

This appendix provides the numerical data for f-mode frequencies of four realistic equations of state. In the first part of the table of each equation of state we list the central density, the radius, the mass of the stellar model, the frequencies of the f-mode for increasing values of $\ell$ . In the second part of each table we list the coefficients $|Q_{0\ell}|$ .

Table C4. Data for the equation of state D Bethe & Johnson (1974) 

<table><tr><td> $\rho_0(\text{gr/cm}^3)$ </td><td>R(km)</td><td> $M(M_\odot)$ </td><td> $\nu_f^{\ell=2}(\text{kHz})$ </td><td> $\nu_f^{\ell=3}(\text{kHz})$ </td><td> $\nu_f^{\ell=4}(\text{kHz})$ </td><td> $|Q_{02}|$ </td><td> $|Q_{03}|$ </td><td> $|Q_{04}|$ </td></tr><tr><td> $1.6 \times 10^{15}$ </td><td>11.131</td><td>1.691</td><td>2.842</td><td>3.593</td><td>4.202</td><td>2.130</td><td>2.609</td><td>2.576</td></tr><tr><td> $1.3 \times 10^{15}$ </td><td>11.476</td><td>1.554</td><td>2.621</td><td>3.308</td><td>3.866</td><td>2.300</td><td>2.481</td><td>2.431</td></tr><tr><td> $1.2 \times 10^{15}$ </td><td>11.620</td><td>1.488</td><td>2.517</td><td>3.151</td><td>3.660</td><td>2.249</td><td>2.426</td><td>2.406</td></tr></table>

# APPENDIX D: TENSOR SPHERICAL HARMONICS.

The explicit expression of the tensor spherical harmonics $\mathcal{Y}_{i_1\dots i_l}^{lm}$ for $\ell = 2$ are

$$
\mathcal {Y} _ {i j} ^ {2 2} = \sqrt {\frac {1 5}{3 2 \pi}} \left( \begin{array}{c c c} 1 & i & 0 \\ i & - 1 & 0 \\ 0 & 0 & 0 \end{array} \right) _ {i j} \quad \mathcal {Y} _ {i j} ^ {2 1} = - \sqrt {\frac {1 5}{3 2 \pi}} \left( \begin{array}{c c c} 0 & 0 & 1 \\ 0 & 0 & i \\ 1 & i & 0 \end{array} \right) _ {i j} \quad \mathcal {Y} _ {i j} ^ {2 0} = \sqrt {\frac {5}{1 6 \pi}} \left( \begin{array}{c c c} - 1 & 0 & 0 \\ 0 & - 1 & 0 \\ 0 & 0 & 2 \end{array} \right) _ {i j} \tag {D1}
$$

and $\mathcal{Y}^{2, - m} = (-1)^{m}\mathcal{Y}^{2,m^{*}}$ .

# APPENDIX E: ENERGY AND ANGULAR MOMENTUM ABSORPTION RATES

In this Appendix we report results for additional equations of state than the one considered in the main text. Figs. E1 show the distribution of energy absorbed $\dot{E}_j$ as a function of the fundamental mode frequency harmonic $j$ for the three equations of state in tabs. C2,C3,C4.

Figs. 2,3, display respectively the energy and angular momentum absorbed by NS oscillations during the inspiral motion for three different central density for each of the three equation of states reported in tabs. C2,C3,C4. For comparison the gravitational luminosity and angular momentum emitted in gravitational wave are also reported.

![](images/a5f6c1522a3dd5723c2148112ea7c6b07e5965d75f3b272797ace2cc833e8e81.jpg)

<details>
<summary>line</summary>

| j  | Series 1 | Series 2 | Series 3 | Series 4 | Series 5 | Series 6 | Series 7 | Series 8 |
|----|----------|----------|----------|----------|----------|----------|----------|----------|
| 0  | 10^-5    | 10^-5    | 10^-5    | 10^-5    | 10^-5    | 10^-5    | 10^-5    | 10^-5    |
| 5  | 10^-3    | 10^-3    | 10^-3    | 10^-3    | 10^-3    | 10^-3    | 10^-3    | 10^-3    |
| 10 | 10^-2    | 10^-2    | 10^-2    | 10^-2    | 10^-2    | 10^-2    | 10^-2    | 10^-2    |
| 15 | 10^-1    | 10^-1    | 10^-1    | 10^-1    | 10^-1    | 10^-1    | 10^-1    | 10^-1    |
| 20 | 1        | 1        | 1        | 1        | 1        | 1        | 1        | 1        |
| 25 | 10       | 10       | 10       | 10       | 10       | 10       | 10       | 10       |
| 30 | 100      | 100      | 100      | 100      | 100      | 100      | 100      | 100      |
| 35 | 1000     | 1000     | 1000     | 1000     | 1000     | 1000     | 1000     | 1000     |
| 40 | 10^6     | 10^6     | 10^6     | 10^6     | 10^6     | 10^6     | 10^6     | 10^6     |
| 45 | 10^4     | 10^4     | 10^4     | 10^4     | 10^4     | 10^4     | 10^4     | 10^4     |
| 50 | 1        | 1        | 1        | 1        | 1        | 1        | 1        | 1        |
</details>

![](images/382246ad9d7f687e47274b5043d2130a90a69a0e9e508e15e843215590101519.jpg)

<details>
<summary>line</summary>

| j  | Series 1 | Series 2 | Series 3 | Series 4 | Series 5 | Series 6 | Series 7 | Series 8 |
|----|----------|----------|----------|----------|----------|----------|----------|----------|
| 0  | 10^-5    | 10^-5    | 10^-5    | 10^-5    | 10^-5    | 10^-5    | 10^-5    | 10^-5    |
| 5  | 10^-3    | 10^-3    | 10^-3    | 10^-3    | 10^-3    | 10^-3    | 10^-3    | 10^-3    |
| 10 | 10^-2    | 10^-2    | 10^-2    | 10^-2    | 10^-2    | 10^-2    | 10^-2    | 10^-2    |
| 15 | 10^-1    | 10^-1    | 10^-1    | 10^-1    | 10^-1    | 10^-1    | 10^-1    | 10^-1    |
| 20 | 1        | 1        | 1        | 1        | 1        | 1        | 1        | 1        |
| 25 | 10       | 10       | 10       | 10       | 10       | 10       | 10       | 10       |
| 30 | 100      | 100      | 100      | 100      | 100      | 100      | 100      | 100      |
| 35 | 1000     | 1000     | 1000     | 1000     | 1000     | 1000     | 1000     | 1000     |
| 40 | 10^7     | 10^7     | 10^7     | 10^7     | 10^7     | 10^7     | 10^7     | 10^7     |
| 45 | 10^6     | 10^6     | 10^6     | 10^6     | 10^6     | 10^6     | 10^6     | 10^6     |
| 50 | 10^5     | 10^5     | 10^5     | 10^5     | 10^5     | 10^5     | 10^5     | 10^5     |
</details>

![](images/e2fd9ec8254607c0a832b8bbbde9a5c4a98662b80d2afcbbb5d01fe056b54843.jpg)

<details>
<summary>line</summary>

| j  | E_j/K (Line 1) | E_j/K (Line 2) | E_j/K (Line 3) | E_j/K (Line 4) | E_j/K (Line 5) | E_j/K (Line 6) | E_j/K (Line 7) | E_j/K (Line 8) | E_j/K (Line 9) |
|----|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| 0  | 10^-5          | 10^-5          | 10^-5          | 10^-5          | 10^-5          | 10^-5          | 10^-5          | 10^-5          | 10^-5          |
| 5  | 10^-3          | 10^-3          | 10^-3          | 10^-3          | 10^-3          | 10^-3          | 10^-3          | 10^-3          | 10^-3          |
| 10 | 10^-2          | 10^-2          | 10^-2          | 10^-2          | 10^-2          | 10^-2          | 10^-2          | 10^-2          | 10^-2          |
| 15 | 10^-1          | 10^-1          | 10^-1          | 10^-1          | 10^-1          | 10^-1          | 10^-1          | 10^-1          | 10^-1          |
| 20 | 10^0           | 10^0           | 10^0           | 10^0           | 10^0           | 10^0           | 10^0           | 10^0           | 10^0           |
| 25 | 10^1           | 10^1           | 10^1           | 10^1           | 10^1           | 10^1           | 10^1           | 10^1           | 10^1           |
| 30 | 10^2           | 10^2           | 10^2           | 10^2           | 10^2           | 10^2           | 10^2           | 10^2           | 10^2           |
| 35 | 10^3           | 10^3           | 10^3           | 10^3           | 10^3           | 10^3           | 10^3           | 10^3           | 10^3           |
| 40 | 10^4           | 10^4           | 10^4           | 10^4           | 10^4           | 10^4           | 10^4           | 10^4           | 10^4           |
| 45 | 10^5           | 10^5           | 10^5           | 10^5           | 10^5           | 10^5           | 10^5           | 10^5           | 10^5           |
| 50 | 10^6           | 10^6           | 10^6           | 10^6           | 10^6           | 10^6           | 10^6           | 10^6           | 10^6           |
</details>

Figure E1. Distribution of the energy per unit of mass absorbed by a single NS oscillation $f$ -mode $\dot{E}_j$ divided by the quantity $K$ defined in eq. (21) as a function of the harmonic mode $j$ of the fundamental orbital frequency in eccentric orbits. Going anti-clockwise from top-left, the results are for the equation of state Douchin & Haensel (2001) in tab. C2 for $\rho_0 = 2.0 \cdot 10^{15} \mathrm{gr/cm^3}$ , Walecka (1974) in tab. C3 for $\rho_0 = 1.0 \cdot 10^{15} \mathrm{gr/cm^3}$ , Bethe & Johnson (1974) in tab. C4 for $\rho_0 = 1.6 \cdot 10^{15} \mathrm{gr/cm^3}$ .

![](images/0e9a14864836a9815fbe711635de88d774336d9d2fc36168001adbfd8256e4b4.jpg)

<details>
<summary>line</summary>

| eccentricity | E_GW     | x = 0.07 | x = 0.01 |
| ------------ | -------- | -------- | -------- |
| 0.0          | 1.0      | 1.0      | 1.0      |
| 0.1          | 1.0      | 1.0      | 1.0      |
| 0.2          | 1.0      | 1.0      | 1.0      |
| 0.3          | 1.0      | 1.0      | 1.0      |
| 0.4          | 1.0      | 1.0      | 1.0      |
| 0.5          | 1.0      | 1.0      | 1.0      |
| 0.6          | 1.0      | 1.0      | 1.0      |
| 0.7          | 1.0      | 1.0      | 1.0      |
| 0.8          | 100.0    | 100.0    | 100.0    |
| 0.9          | 100.0    | 100.0    | 100.0    |
</details>

![](images/aa4120c23e176ac01576be1ee93a97aaebd123ee02df08ec8a49cc1ffb26de70.jpg)

<details>
<summary>line</summary>

| eccentricity | E_GW       | x = 0.07   | x = 0.01   |
| ------------ | ---------- | ---------- | ---------- |
| 0.0          | 1.0        | 1.0        | 1.0        |
| 0.1          | 1.0        | 1.0        | 1.0        |
| 0.2          | 1.0        | 1.0        | 1.0        |
| 0.3          | 1.0        | 1.0        | 1.0        |
| 0.4          | 1.0        | 1.0        | 1.0        |
| 0.5          | 1.0        | 1.0        | 1.0        |
| 0.6          | 1.0        | 1.0        | 1.0        |
| 0.7          | 1.0        | 1.0        | 1.0        |
| 0.8          | 10^2       | 10^2       | 10^2       |
| 0.9          | 10^2       | 10^2       | 10^2       |
</details>

![](images/d0d07af630faa4f2dbb7f03cb34102933c2bc6add41bfd434907f37deca578f5.jpg)

<details>
<summary>line</summary>

| eccentricity | E_GW     | x = 0.07 | x = 0.01 |
| ------------ | -------- | -------- | -------- |
| 0.0          | 1.0      | 1.0      | 1.0      |
| 0.1          | 1.0      | 1.0      | 1.0      |
| 0.2          | 1.0      | 1.0      | 1.0      |
| 0.3          | 1.0      | 1.0      | 1.0      |
| 0.4          | 1.0      | 1.0      | 1.0      |
| 0.5          | 1.0      | 1.0      | 1.0      |
| 0.6          | 1.0      | 1.0      | 1.0      |
| 0.7          | 1.0      | 1.0      | 1.0      |
| 0.8          | 1.0      | 1.0      | 1.0      |
| 0.9          | 1.0      | 1.0      | 1.0      |
</details>

Figure E2. Rate of energy absorbed $\dot{E}_{*}$ as a function of eccentricity, with $M_{BH} = 5M_{\odot}$ , NS with equation of state respectively given, moving anti-clockwise from top-left, by Douchin & Haensel (2001) in tab. C2, Walecka (1974) in tab. C3, and Bethe & Johnson (1974) in tab. C4. For comparison we also plot the GW luminosity for two values of x, all functions are divided by the Newtonian GW luminosity at zero eccentricity $\dot{E}_{GW0}$ given by eq. (22). Note that for large eccentricity e > 0.7 absorption by NS as computed in this approximation is not negligible compared to GW emission. For each equation of states results for the three values of the central density reported in the corresponding tables are reported, increasing line thickness denoting higher central density.

![](images/7f5b1239d343640cec0506397f570901d1a17e257b5948db61c33b786c6e1a70.jpg)

<details>
<summary>line</summary>

| eccentricity | L̂_GW       | x=0.07     | x=0.01     |
| ------------ | ---------- | ---------- | ---------- |
| 0.0          | 1.0        | 1e-10      | 1e-14      |
| 0.1          | 1.0        | 1e-10      | 1e-14      |
| 0.2          | 1.0        | 1e-10      | 1e-14      |
| 0.3          | 1.0        | 1e-10      | 1e-14      |
| 0.4          | 1.0        | 1e-10      | 1e-14      |
| 0.5          | 1.0        | 1e-9       | 1e-13      |
| 0.6          | 1.0        | 1e-8       | 1e-12      |
| 0.7          | 1.0        | 1e-7       | 1e-11      |
| 0.8          | 1.0        | 1e-6       | 1e-10      |
| 0.9          | 1.0        | 1e-5       | 1e-9       |
</details>

![](images/2a153cc9c96dc9d7ac7653639ccf6fc531563329ab73b7735487526ce1c15b19.jpg)

<details>
<summary>line</summary>

| eccentricity | L̂_GW       | x=0.07     | x=0.01     |
| ------------ | ---------- | ---------- | ---------- |
| 0.0          | 1.0        | 1e-10      | 1e-14      |
| 0.1          | 1.0        | 1e-10      | 1e-14      |
| 0.2          | 1.0        | 1e-10      | 1e-14      |
| 0.3          | 1.0        | 1e-10      | 1e-14      |
| 0.4          | 1.0        | 1e-10      | 1e-14      |
| 0.5          | 1.0        | 1e-8       | 1e-13      |
| 0.6          | 1.0        | 1e-8       | 1e-12      |
| 0.7          | 1.0        | 1e-6       | 1e-11      |
| 0.8          | 1.0        | 1e-4       | 1e-10      |
| 0.9          | 1.0        | 1e-3       | 1e-9       |
</details>

![](images/345c772611e8d7ae177b4234a98f6257f8d278a47f83c446d4c587ce02927292.jpg)

<details>
<summary>line</summary>

| eccentricity | L_GW       | x=0.07     | x=0.01     |
| ------------ | ---------- | ---------- | ---------- |
| 0.0          | 1.0        | 1.0e-11    | 1.0e-13    |
| 0.1          | 1.0        | 1.0e-11    | 1.0e-13    |
| 0.2          | 1.0        | 1.0e-11    | 1.0e-13    |
| 0.3          | 1.0        | 1.0e-11    | 1.0e-13    |
| 0.4          | 1.0        | 1.0e-11    | 1.0e-13    |
| 0.5          | 1.0        | 1.0e-11    | 1.0e-13    |
| 0.6          | 1.0        | 1.0e-9     | 1.0e-12    |
| 0.7          | 1.0        | 1.0e-8     | 1.0e-12    |
| 0.8          | 1.0        | 1.0e-7     | 1.0e-12    |
| 0.9          | 1.0        | 1.0e-6     | 1.0e-12    |
</details>

Figure E3. Rate of angular momentum absorbed as a function of eccentricity, same parameters as in fig. 2. Here $\dot{L}_{GW}$ is the Newtonian angular momentum loss in GWs for small eccentricities $\dot{L}_{GW} = \frac{32}{5}\eta^{2}M\frac{x^{7/2}}{(1-e^{2})^{2}}\left(1 + \frac{7}{8}e^{2}\right)$ and $\dot{L}_{GW0} = \dot{L}_{GW}|_{e=0}$ , with $M \equiv M_{*} + M_{BH}$ , $\eta \equiv M_{*}M_{BH}/M^{2}$ .