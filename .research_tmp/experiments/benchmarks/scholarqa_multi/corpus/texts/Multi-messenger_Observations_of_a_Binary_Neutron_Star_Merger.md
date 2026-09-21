# Investigating binary-neutron-star mergers as production sites of high-energy neutrinos

S. Rossoni,\(^{a}\) D. Boncioli,\(^{b,c}\) G. Sigl\(^{a}\)

\(^{a}\)II. Institute for Theoretical Physics, Hamburg University, Luruper Chaussee 149, 22761, Hamburg, Germany \(^{b}\)Università degli Studi dell'Aquila, Dipartimento di Scienze Fisiche e Chimiche, Via Vetoio, 67100, L'Aquila, Italy \(^{c}\)INFN Laboratori Nazionali del Gran Sasso, Assergi (L'Aquila), Italy

E-mail: simone.rossoni@desy.de, denise.boncioli@univaq.it, guenter.sigl@desy.de

Abstract. The end state of binary-neutron-star (BNS) mergers can manifest conditions to produce high-energy neutrinos. Inspired by the event GW170817, detected in gravitational waves and in optical/infrared emission, we investigate a scenario in which cosmic-ray (CR) particles are accelerated, in a population of BNS mergers, in the energy range that might contribute from the knee to the ankle of the CR measured spectrum.

By taking into account the measured thermal and non-thermal energy density of the photon fields in the source environment as a function of the time after the merger, we model the CR interactions and the consequent neutrino production. We propagate the escaped CR and neutrino fluxes through the extragalactic space and compare the expected diffuse fluxes to the experimental data and current limits.

Depending on the CR spectral and composition parameters at acceleration, and on the possible contribution to the sub-ankle CR flux, we discuss the predicted diffuse neutrino flux associated to this class of astrophysical objects, as a function of the details of the photon field characterizing the merger stage, including its evolution in time. We constrain the fraction of accelerated baryons in the source site given the BNS merger rate per volume, taking into account at the same time the constraints from the measured CR and neutrino fluxes.

# Contents

1 Introduction

2 Modeling interactions in the source environment

2.1 Interaction efficiency 5  2.2 Numerical simulations 7

3 Results

3.1 Source escape and interactions 8  3.2 Extragalactic propagation 12  3.3 Study of source parameters 15  3.4 Source temporal evolution 17

4 Discussion and conclusions 20

A Ballistic approximation 24

B UHECR interaction rate 24

C Cosmic-ray spectra 25

D Parameter study with the SFR 26

# 1 Introduction

In the last ten years, the IceCube Neutrino Observatory [1] has reported the observation of a diffuse neutrino flux in the TeV - PeV energy range [2]. In particular, the muon-neutrino flux from the IceCube muon-track data [3] is consistent with a single power-law spectrum of the form \( E_{\nu}^{-\gamma} \), with normalization at 100 TeV of \( \phi_{\nu_{\mu}+\bar{\nu}_{\mu}} \simeq 1.5 \cdot 10^{-18} \, \mathrm{GeV}^{-1} \, \mathrm{cm}^{-2} \, \mathrm{s}^{-1} \, \mathrm{sr}^{-1} \) and spectral index \( \gamma \simeq 2.4 \).

Recently, the flux of neutrinos from cascade events [4] has also been shown to be consistent with a single power law with normalization of \( \phi_{\nu+\bar{\nu}} \simeq 1.7 \cdot 10^{-18} \, \mathrm{GeV}^{-1} \, \mathrm{cm}^{-2} \, \mathrm{s}^{-1} \, \mathrm{sr}^{-1} \) at 100 TeV, and spectral index \( \gamma \simeq 2.5 \). Upper limits have been set for contributions from different classes of sources, such as for instance the one from blazars [5].

Observations of astrophysical sources through different messengers led to the emergence of multi-messenger astronomy [6], such as the successful observation in neutrinos and gamma rays of the blazar TXS 0506+056 [7], and the recent association of 79 neutrino events to the Seyfert galaxy NGC 1068 [8]. It has also been shown that IceCube neutrino events can be associated with the optical counterparts of the emission from sites where the disruption of stars from a supermassive black hole is supposed to happen (tidal disruption events, TDEs) [9–11].

The observation of high-energy neutrinos from astrophysical sites might reveal hadronic or photo-hadronic processes within the source region, involving hadronic particles accelerated in the environment as well as photons and/or matter of the source site. Phenomenological models are developed to explain at the same time the electromagnetic emission and the neutrino production, as for instance in [12] for the blazar TXS 0506+056 or in [13] for NGC 1068, without the need of invoking acceleration mechanisms of cosmic rays (CRs) in jets.

The observation of high-energy neutrinos from astrophysical sites might reveal hadronic or photo-hadronic processes within the source region, involving hadronic particles accelerated in the environment as well as photons and/or matter of the source site. Phenomenological models are developed to explain at the same time the electromagnetic emission and the neutrino production, as for instance in [12] for the blazar TXS 0506+056 or in [13] for NGC 1068, without the need of invoking acceleration mechanisms of cosmic rays (CRs) in jets.

In particular, in the latter case a two-zone model is investigated by modeling interactions of cosmic-ray particles in the corona as well as in the circumnuclear starburst region. Non-jetted sites are therefore nowadays increasingly interesting as possible high-energy neutrino factories, as recently investigated also in [14]. This is also supported by the interpretation of the neutrino emission from the TDE AT2019aalc as modeled in [15], where it is shown that the delay of the neutrino signal with respect to the optical-infrared emission can be due to the confinement of protons in regions not aligned with the jet.

On the other hand, the first joint observation of a gravitational wave signal and the electromagnetic counterpart [16–18] happened in 2017. Although, nowadays, no evidence of correlations between high-energy neutrinos and gravitational waves has been established [19], the source environments responsible for gravitational-wave signals are considered to be of great interest for the study of high-energy interactions, as already shown in [20]. A particular class of gravitational-wave sources are binary systems of coalescing neutron stars (NSs).

The probable end state of binary-neutron-star (BNS) mergers is a black hole (BH) with a relativistic jet, powered by the material in the accretion disk. The formation of the jet gives rise to a short gamma-ray burst (GRB), which represents a promising site for the production of high-energy neutrinos, as also reported in [21]. An alternative scenario for the production of astrophysical neutrinos is described in [22], where a small fraction of the ejected material is considered to fall back to the central compact object produced after the merger.

This fallback outflow encounters the earlier ejected mass shell producing a shock wave where particles can be accelerated. It has been shown in previous studies [23–27] that these environments might be interesting acceleration sites of cosmic rays. In particular, in [27] it is discussed how the characteristic magnetic field might allow cosmic-ray particles to reach the energy of the ankle in the CR energy spectrum. Therefore, these sources could be considered as candidates for contributing to the energy region of the cosmic-ray spectrum beyond the Galactic contribution.

In the present work, we consider the modeling of the BNS merger remnant described in [22] to study the interaction of accelerated ultra-high-energy cosmic rays (UHECRs, i.e. atomic nuclei with energy \(\gtrsim 10^{17}\,\mathrm{eV}\)) with the local photon fields. In particular, we consider the source region to be populated by a thermal field produced by the nuclear decay of synthesized nuclei in the ejecta, and a non-thermal synchrotron component (see [27, 28]).

Interactions of UHECRs with local photons give rise to the production of unstable mesons, and therefore to the production of high-energy neutrinos. In particular, after being produced, the latter ones leave the source undisturbed a
nd travel through the outer space without undergoing any interaction or magnetic deflection. In contrast, the escape condition of UHECRs represents a non-trivial problem. As a first approximation, the typical UHECR escape time is given by the dimension of the source.

In this work, we assume the radius of the ejecta material as the typical size of the interaction region, i.e. we adopt the ballistic approximation.

In order to link the observed information in UHECRs and neutrinos, the re-processing of the accelerated UHECRs within the merger region needs to be combined with the effects of the extragalactic propagation from the production site to Earth, consisting of interactions with the cosmic photon fields, as the cosmic microwave background (CMB) and extragalactic background light (EBL) [29, 30]. Interactions with cosmic photons will give rise to a second population of neutrinos called cosmogenic. Hereafter we will refer to neutrinos produced in the source as source neutrinos, and neutrinos produce during the propagation as cosmogenic neutrinos.

A population of BNS mergers is considered in this work; therefore, the diffuse flux will

depend on the event rate per volume of BNS mergers, \(\dot{n}\), which also affects the amount of UHECRs injected in the extragalactic space. We use these quantities to constrain the baryonic loading \(\eta\) (i.e. the ratio between the fallback luminosity and the UHECR luminosity at the acceleration) of BNS mergers.

This study is organized as follows: the modeling of the BNS merger remnant and its interaction efficiency, and numerical implementation are discussed in Sec. 2. Simulation results at the escape from the source and at Earth are shown in Sec. 3 (in particular, in Sec. 3.3 the production of high energy neutrinos is analyzed in detail). In Sec. 3.4, neutrino production is studied by integrating over the time evolution of the source environment. Discussion of the results and conclusions are given in Sec. 4.

# 2 Modeling interactions in the source environment

The production of high-energy neutrinos is here investigated as the result of interactions between the fallback material and the local photon fields of the source environment. We assume, as done in [22], that acceleration mechanisms can happen in the fallback process, so that nuclei can reach energies up to \(\sim10^{19}\) eV (see also Sec. 3.1 for more details). A brief introduction to our modeling of the source environment and interactions can be found in [31, 32]. We define the time \(t\) as the time after the coalescence, so that \(t=0\) corresponds to the merger event.

Due to the nuclear decay of the unstable species synthesized in the ejecta by the merger, a thermal photon field is produced in the source environment. Assuming that the heat from nuclear decays is homogeneously distributed, the photon emission can be modeled as a black-body (BB) photon field. The flux of energy (i.e. the energy per unit of time, area and frequency) emitted by a black body and observed at a distance \(d\) is given by

\[
F_{\nu}^{\mathrm{BB}}=\frac{2\pi h\nu}{c^{2}}\left(\frac{R}{d}\right)^{2}\frac{\nu^{2}}{\exp(h\nu/k_{\mathrm{B}}T)-1},
\]

where \(\nu\) is the photon frequency, \(T\) is the BB temperature, \(k_{B}\) is the Boltzmann constant, \(c\) is the speed of light, \(h\) is the Planck constant and \(R\) is the source radius. The temperature of the BB is given by the Stefan-Boltzmann law

\[
T=\left(\frac{3E_{\mathrm{BB}}}{4\pi a R^3}\right)^{1/4},
\]

where \(u=3E_{\mathrm{BB}}/4\pi R^{3}\) is the energy density of the BB and \(a=7.6\cdot10^{-15}\,\mathrm{erg\,cm^{-3}\,K^{-4}}\) [22, 33]. We modeled the temporal evolution of the BB photon field density as in Fig. 2 of [22]. Therefore, the time after the merger \(t\) and the BB temperature \(T\) are such that

\[
T=10^{6}\cdot\left(\frac{t}{10^{3}\mathrm{s}}\right)^{-2}\mathrm{K}.
\]

As expected, when time increases after the merger, the temperature of the BB decreases. The spectral energy density (SED, i.e. the number of photons per unit of energy and volume) is given by

\[
n_{\mathrm{BB}}(\epsilon)=\frac{1}{\pi^2(\hbar c)^3}\frac{\epsilon^2}{\exp(\epsilon/k_{\mathrm{B}}T)-1},
\]

where \(\epsilon\) is the photon energy. The number density of BB photons is given by the integral in the photon energy \(\epsilon\) of Eq. (2.4), and it corresponds to \(n_{\mathrm{BB}}\simeq20\cdot(T/1\,\mathrm{K})^{3}\,\mathrm{cm}^{-3}\).

Figure 1. Spectral energy densities of the photon fields used in this work: black body (solid lines) and non-thermal (dashed lines). Different times after the merger (corresponding to different temperatures of the black body) are shown in different colors.

![](dt=2025-08-08/ht=01/844e2a66466473463bac0bea4fd948cdef6f23ee3c0dc0ab6cc5531131855b46.jpg)

Several days after the merger, the dominance of the thermal photon field is replaced by a non-thermal (NT) component, mainly due to synchrotron emission. In order to model this background field, we consider the radio-to-X-ray emission of the merger event GW170817 described in [28], where the flux density function is modeled as \(\phi_{\mathrm{NT}}(\nu) \propto \nu^{-\beta}\), where \(\nu\) is the photon frequency and the index \(\beta\) is fixed to the value 0.6, as discussed in [28]. From [28], we obtain the flux density function<sup>1</sup> as

\[
\phi_{\mathrm{NT}}(\nu)=0.76\cdot\left(\frac{t}{1\mathrm{s}}\right)^{1.2}\left(\frac{\nu}{1\mathrm{Hz}}\right)^{-0.6}\mu\mathrm{Jy},
\]

where \(t\) is the time after the merger. The flux density in Eq. (2.5) can be converted to spectrum energy density, so that the non-thermal SED reads

\[
n_{\mathrm{NT}}(\epsilon)=1.2\cdot10^{27}\cdot\left(\frac{V}{1\mathrm{km}^3}\right)^{-1}\left(\frac{t}{1\mathrm{s}}\right)^{2.2}\left(\frac{\epsilon}{1\mathrm{eV}}\right)^{-1.6}\mathrm{eV}^{-1}\mathrm{cm}^{-3},
\]

where \(V\) is the volume of the source environment. Both the SEDs in Eqs. (2.4) and (2.6) are shown in Fig. 1, where solid lines correspond to the thermal (BB) field, while dashed lines correspond to the non-thermal (NT) component. Different colors are used for different temperatures (or times) after the merger. The BB component is the dominant background for \(\epsilon \gtrsim 0.01\,\mathrm{eV}\). For \(T \lesssim 10^{4}\,\mathrm{K}\) (i.e. \(t \gtrsim 3\,\mathrm{h}\)) the non-thermal field becomes dominant over the BB. However, we will show that after this time from the merger, the neutrino production by photohadronic interactions becomes inefficient.

The size of the production site for neutrinos can be approximated by taking into account the radius of the ejected material, in free expansion after the merger. In this work we make the simplistic assumption that the typical escape length for UHBCRs is given by the radius of the ejected material, namely

\[
\lambda_{\mathrm{esc}}(t)=\beta_{\mathrm{ej}}ct,
\]

where \(\beta_{\mathrm{ej}}\) is the speed of the ejected material in units of speed of light. The typical escape rate is thus

\[
\tau_{\mathrm{esc}}^{-1}(t)=\frac{c}{\lambda_{\mathrm{esc}}(t)}=\frac{1}{\beta_{\mathrm{ej}}t}.
\]

We assume \(\beta_{\mathrm{ej}}=0.3\), as done in [22]. This assumption may influence the neutrino production efficiency of the source. A higher value of \(\beta_{\mathrm{ej}}\) corresponds to a lower escape rate, and thus accelerated CRs may produce more neutrinos. We also note that the confinement of a nucleus is only due to the dimension of the source itself, and does not depend on its rigidity. Further details on the validity of the ballistic approximation can be found in Appendix A.

The typical source length defined in Eq. (2.7) can be used to compute the source volume in the non-thermal SED in Eq. (2.6). Therefore, the non-thermal SED \(n_{\mathrm{NT}}(\epsilon)\) can be written as

\[
n_{\mathrm{NT}}(\epsilon)=4.2\cdot10^{11}\cdot\left(\frac{\beta_{\mathrm{ej}}}{0.3}\right)^{-3}\left(\fr
ac{t}{1\mathrm{s}}\right)^{-0.8}\left(\frac{\epsilon}{1\mathrm{eV}}\right)^{-1.3}\mathrm{eV}^{-1}\mathrm{cm}^{-3}.
\]

We obtain that \(n_{\mathrm{NT}}(\epsilon)\) evolves in time as \(\propto t^{-0.8}\).

# 2.1 Interaction efficiency

In this section we compute the photohadronic interaction lengths of the two nuclear species considered in this work, namely protons (p) and iron nuclei \((^{56}\mathrm{Fe})\), at different times after the merger. Bethe-Heitler pair production is also taken into account in this work. However, as shown in Figure 3 of [22], the energy-loss length of pair production is always several orders of magnitude greater than that associated with photopion production. Therefore, Bethe-Heitler pair production will not have a major effect on the total interaction efficiency and escape conditions of the accelerated nuclei. Further details on the calculation of photohadronic interaction lengths can be found in Appendix B.

In the left panel of Fig. 2, the photopion interaction lengths (solid lines) of protons interacting with BB photon fields are shown, as a function of the proton Lorentz factor \(\Gamma\). Different colors correspond to different temperatures, as in the legend of Fig. 1. The dashed lines correspond to the source typical lengths as defined in Eq. (2.7) at the corresponding temperatures.

Two main effects can be observed: first, both the interaction lengths \(\lambda_{\mathrm{p}}\) and the source radii \(\lambda_{\mathrm{esc}}\) increase as the BB field cools; secondly, \(\lambda_{\mathrm{p}}>\lambda_{\mathrm{esc}}\) for each value of \(\Gamma\) only when \(T\sim10^{4}\,\mathrm{K}\). After the merger, the BB temperature decreases and the number of available high-energy photons decreases, making photopion production less efficient. At the same time, the expansion of the source environment is not fast enough to compensate for the field dilution.

The result is that, for \(T\gtrsim10^{4}\,\mathrm{K}\), protons no longer interact with the local field and escape freely from the source. Conversely, in the early stages after the merger, protons interact frequently with the local field, and only very high energy protons escape undisturbed. The same result applies to protons of very low energy. However, in this case the escape energy threshold for low energy protons increases rapidly as the temperature of the BB decreases, to the point where protons escape for any energy (black line in left panel of Fig. 2). In the right panel of Fig.

2 the same quantities of the left panel are shown, corresponding to the

Figure 2. Interaction lengths for protons for photopion production, corresponding to the BB photon field (solid line, left panel) and to the NT field (solid line, right panel), as a function of the Lorentz factor; the size of the source radii is indicated with dashed lines. Different colors refer to the BB temperatures as indicated in Fig. 1. Note the different ranges of the y-axes.

![](dt=2025-08-08/ht=01/f7cec1ee1f8841d24f15796611ba6e571161055f957dcbbc0745b996ce07fd36.jpg)

Figure 3. Same as in Fig. 2, for iron nuclei. Note the different ranges of the y-axes.

![](dt=2025-08-08/ht=01/3fcba11974928b497374bb0dd46ced224a55654130216363230aa8a349432652.jpg)

NT field as defined in Eq. (2.9). In this case, the photopion production is never an efficient process since the number of high-energy photons is highly suppressed, even when the source temperature is \( T=10^{8}\,\mathrm{K} \).

The propagation of iron nuclei in the source environment is affected by the photodisintegration as well as by the photopion process. In the left panel of Fig. 3 the total interaction length of an iron nucleus is shown together with the typical source radius; the low-energy minimum can be attributed to the photodisintegration while the high energy minimum to the photopion production. The general evolution of the interaction-escape dynamics with the temperature is similar to the proton scenario, but for this case interactions are possible also for \( T\sim10^{4}\,\mathrm{K} \). The total interaction length of iron nuclei with NT photons is slightly smaller than in the case of protons. However, interactions with this photon component of the source environment are still inefficient.

The evolution of the efficiency of photohadronic interactions as a function of the time after the merger can be summarized by considering the source opacity. The opacity of the source environment for a nucleus of mass \( A \) can be defined as

\[
\zeta_{A}(\Gamma,t)=\frac{\lambda_{\mathrm{esc}}(t)}{\lambda_{A}(\Gamma,t)},
\]

Figure 4. Source opacity as defined in Eq. (2.10) for Lorentz factors $\Gamma=10^{9}$ (blue), $\Gamma=10^{10}$ (red) and $\Gamma=10^{11}$ (green). Solid lines corresponds to the BB field, dashed lines to the NT field and the solid gray line represents the condition $\lambda_{A}=\lambda_{\mathrm{esc}}$. The cases for protons (left) and for iron nuclei (right panel) are shown.

![](dt=2025-08-08/ht=01/70499a9a5d46b0d01586b835208f8ecfb36c70b6f8ff63af2d201136bdd26b2c.jpg)

where \(\lambda_{A}\) is the total interaction length of a nucleus with mass \(A\) and \(\lambda_{\mathrm{esc}}\) is defined in Eq. (2.7). In particular, when \(\zeta_{A}>1\) (\(<1\)), interactions dominate (are suppressed) over the escape condition. This quantity is shown in the left panel of Fig. 4 for protons and in the right panel for iron nuclei, as a function of time, for different fixed values of the Lorentz factor. Here, solid lines refer to the opacity of the BB, and dashed lines to the opacity of the NT field.

The solid gray line represents the confinement-escape limit \(\lambda_{A}=\lambda_{\mathrm{esc}}\). The NT component is always transparent for protons and iron nuclei, while the BB opacity is greater than 1 for \(\Gamma\lesssim10^{8}\) for protons, and \(\Gamma\lesssim10^{10}\) for iron nuclei, for a considerable time after the merger. As also discussed in [27], very high energy cosmic rays (\(\Gamma\gtrsim10^{12}\)) will undergo interactions with local photon fields when \(t\lesssim10^{4}\,\mathrm{s}\).

At later times, the source opacity becomes smaller than 1, and the nuclei are free to escape from the source region without interacting.

The NT component of the source environment is, for both the nuclear species considered, not relevant for UHECR interactions, and therefore for the production of astrophysical neutrinos. For this reasons, we will consider only the interactions in the BB field in the following.

# 2.2 Numerical simulations

In this work, we modify the Monte Carlo code for the computation of the UHECR extragalactic propagation SimProp-v2r4\(^{3}\) [34], in order to simulate the interactions within the source environment, until the escape condition is met. We name this modified version SimProp-Mod. All the interaction processes present in SimProp-v2r4 involve CRs and the extragalactic background photon fields (CMB and EBL). The first modification concerns the target photon fields for photohadronic interactions. We replaced the cosmic fields with the local fields described in Sec. 2.

As discussed earlier, the non-thermal component of the source environment is not considered in the simulations. In the original version of SimProp-v2r4, particles propagate from the source until they reach the redshift condition \(z=0\). In the current version, we implemented an escape condition based on a leaky-box model, where the escape rate in Eq. (2.8) is compared to the interaction rates as defined in Eq. (B.1) (see Appendix B for

more details). The escape condition is therefore determined by the random sampling of the escape rate among all the possible processes.

# 3 Results

In this section, we make use of the simulation framework previously
described considering several scenarios for the injection of UHECRs in the source region. When the spectra of the particles at the escape are obtained, these are propagated to the Earth and compared to available data, in order to constrain the source parameters.

# 3.1 Source escape and interactions

We describe here the propagation of UHECRs in the source taking into account pair production, photopion production and photodisintegration with the BB component of the local photon fields (see Sec. 2.1). We consider post-merger times corresponding to BB temperatures between \(10^{8}\,\mathrm{K}\) and \(10^{4}\,\mathrm{K}\) (i.e. the temperatures shown in Fig. 1). Since the temperature of the BB is a fixed parameter in every simulation, the other source properties, which depend on the temperature (source radius, environment SEDs and escape rate) are also fixed. The photodisintegration of nuclei heavier than protons is simulated considering the default option of SimProp-v2r4 with the parametrization of the cross-sections adapted from [35] and [36].

In [22], it was shown that, assuming that the magnetic energy at the acceleration site is a fraction of the kinetic energy of the fallback, the only limiting process to the acceleration of nuclei is synchrotron cooling. Since hadronic and photoadronic interaction processes occur on a large time scale, we can separate the CR acceleration phase from the neutrino production phase. Furthermore, assuming an equipartition between magnetic and kinetic energy, a maximum CR acceleration energy of \(\sim10^{19}\) eV can be reached for proton and iron nuclei. In [22] we can also see that the maximum acceleration energy depends slightly on the post-merger time. Therefore, we assume a scenario in which \(E_{\mathrm{cut}}\) does not change over time.

We study the production of high energy neutrinos by injecting CRs with energy between \(10^{14}\,\mathrm{eV}\) and \(10^{20}\,\mathrm{eV}\) into the source region. The injection rate at the acceleration (i.e. the number of cosmic rays per unit of energy, time and volume) of the nuclear species with mass number \(A\) is taken as

\[
Q_{\mathrm{acc}}^{A}(E)=Q_{0,\mathrm{acc}}^{A}\left(\frac{E}{1\mathrm{GeV}}\right)^{-\gamma}\exp\left(-\frac{E}{E_{\mathrm{cut}}}\right),
\]

where \(\gamma\) is the spectral index and \(E_{\mathrm{cut}}\) is the high-energy cutoff; the normalization \(Q_{0,\mathrm{acc}}^{A}\) will be adjusted thanks to the comparison of the propagated cosmic rays to the experimental data. For the spectral parameters we consider \(\gamma=0.5,...,2.5\) with steps of \(\Delta\gamma=0.25\), and \(\log(E_{\mathrm{cut}}/1\,\mathrm{eV})=17.0,...,19.0\) with steps of \(\Delta\log(E_{\mathrm{cut}}/1\,\mathrm{eV})=0.1\).

The escaped UHECR and neutrino spectra for a pure-proton injection (for a simulation of \(10^{6}\) protons) are shown in Fig. 5 in arbitrary units. Several injection configurations are shown to appreciate the effect of different injection scenarios on the escaped spectra: \(\gamma=1.5\) and \(E_{\mathrm{cut}}=10^{18.5}\,\mathrm{eV}\) in the upper panels, \(\gamma=1.5\) and \(E_{\mathrm{cut}}=10^{17.5}\,\mathrm{eV}\) in the central panels and \(\gamma=0.75\) and \(E_{\mathrm{cut}}=10^{18.5}\,\mathrm{eV}\) in the bottom panels.

Different colors correspond to different BB temperatures while the dashed black line in the UHECR spectra corresponds to the injected energy spectrum from Eq. (3.1). As expected, photohadronic interactions systematically suppress the injected proton spectrum, and, when the BB field cools down, the escaped spectrum tends to converge to the injected one. This is due to the fact that the

Figure 5. Left panels: energy spectra at the escape from the source for pure-proton acceleration, in arbitrary units. Right panels: neutrino energy spectra at the escape from the source, in arbitrary units. The dashed black line in the CR spectra corresponds to the injected energy spectrum. The injection parameters are \(\gamma=1.5\) and \(E_{\mathrm{cut}}=10^{18.5}\,\mathrm{eV}\) (upper panels), \(\gamma=1.5\) and \(E_{\mathrm{cut}}=10^{17.5}\,\mathrm{eV}\) (central panels) and \(\gamma=0.75\) and \(E_{\mathrm{cut}}=10^{18.5}\,\mathrm{eV}\) (bottom panels). Colors refer to different BB temperatures (see Fig. 1).

![](dt=2025-08-08/ht=01/0d1f8c3c747ed1e1490efa1ba03cd3cf5f0db8dbfe789af533dfcba4df8a693c.jpg)

total interaction length \(\lambda_{p}(\Gamma,t)\) increases (decreases) faster than the escape length \(\lambda_{\mathrm{esc}}(t)\) with time (and temperature). The injected spectrum at \(E\lesssim10^{19}\,\mathrm{eV}\) is in general more suppressed than the high energy tail. This is because the condition \(\lambda_{p}\gtrsim\lambda_{\mathrm{esc}}\) is satisfied for almost all the temperatures when \(E\gtrsim10^{19}\,\mathrm{eV}\). The opposite behavior can be observed for the neutrino spectra, as shown in the right panels of Fig. 5. When the temperature of the BB is high, interactions are more efficient, and the production of high-energy neutrinos is favored, until the saturation level is reached. We can also see that the peak in the neutrino spectra is at

Figure 6. Left panel: energy spectrum at the escape from the source for pure-iron acceleration, in arbitrary units. Only the BB temperature \( T=10^{4}\,\mathrm{K} \) is shown. Colors correspond to different mass groups at the escape, as indicated in the legend. Note the different range of the y-axis from the proton scenario. Right panel: neutrino energy spectrum at the escape from the source, in arbitrary units. Colors refer to the BB temperatures in Fig. 1. The dashed black line in the UHECR spectra correspond to the injected energy spectrum. The injection parameters are \( \gamma=1.5 \) and \( E_{\mathrm{cut}}=10^{18.5}\,\mathrm{eV} \).

![](dt=2025-08-08/ht=01/0980a4a64e5f5805693aa4d2c394547afab9b61dc01fc4650c1d16ba7ec3f2df.jpg)

\( E_{\nu}\sim5\%\cdot E_{\mathrm{p}} \), where \( E_{\mathrm{p}} \) is the peak in the CR spectrum.

In Fig. 6 the escaped spectra for a pure-iron injection (corresponding to a simulation of \( 10^{5} \) iron nuclei) are shown. The injection parameters are \( \gamma=1.5 \) and \( E_{\mathrm{cut}}=10^{18.5}\,\mathrm{eV} \). Only the case with \( T=10^{4}\,\mathrm{K} \) is shown in the left panel, where different colors correspond to different mass groups.

The effect of injecting nuclei can be appreciated in two different ways: the escaped UHECRs are a mix of nuclear species, due to photodisintegration, and the production of high-energy neutrinos is reduced with respect to the proton scenario, as due to the increased threshold for the photopion production. However, the effect of the propagation within the source environment for different temperatures is similar to the case of the proton injection: high temperatures correspond to an enhanced production of neutrinos, and a reduced flux of escaped cosmic rays.

The inverse behavior of the CR escape and neutrino production while changing the target temperature characterizes the efficiency of the source in converting the accelerated baryonic material into neutrinos. We quantify this effect by computing the UHECR and neutrino emissivities. In particular, we compute the UHECR emissivity as

\[
\begin{array}{l}\displaystyle\mathcal{E}_{\mathrm{acc}}=\sum_{A}\int dEEQ_{\mathrm{acc}}^{A}(E),\\\displaystyle\mathcal{E}_{\mathrm{esc}}=\sum_{A}\int dEEQ_{\mathrm{esc}}^{A}(E),\end{array}
\]

where \( Q_{\mathrm{acc}}^{A}(E) \) and \( Q_{\mathrm{esc}}^{A}(E) \) are the rates of the nuclear species \( A \) at the acceleration (i.e. the injection into the source environment) and at the escape, respectively. The integration is performed over the injection energy range \( (10^{14}-10^{20}\,\mathrm{eV}) \). The neutrino emissivity is calculated in the same way starting from the neutrino e
scape rate \( Q_{\mathrm{esc}}^{\nu}(E_{\nu}) \), but using an energy range shifted by a factor of 5% relative to the energy range of cosmic rays, and it is denoted by \( \mathcal{E}_{\mathrm{esc}}^{\nu} \). Therefore, we define the efficiency parameters as

\[
f_{\mathrm{CR}}=\frac{\mathcal{E}_{\mathrm{acc}}-\mathcal{E}_{\mathrm{esc}}}{\mathcal{E}_{\mathrm{acc}}},
\]

Figure 7. Left panel: source environment efficiency, as defined in Eq. (3.4). Right panel: neutrino production efficiency, as defined in Eq. (3.5). Colors correspond to different injection scenarios: $\gamma = 1.5$ and $\log(E_{\mathrm{cut}}/1\,\mathrm{eV}) = 18.5$ (black), $\gamma = 0.5$ and $\log(E_{\mathrm{cut}}/1\,\mathrm{eV}) = 18.5$ (blue), $\gamma = 2.5$ and $\log(E_{\mathrm{cut}}/1\,\mathrm{eV}) = 18.5$ (green), $\gamma = 1.5$ and $\log(E_{\mathrm{cut}}/1\,\mathrm{eV}) = 17.0$ (red) and $\gamma = 1.5$ and $\log(E_{\mathrm{cut}}/1\,\mathrm{eV}) = 19.0$ (yellow). Both panels refer to a pure-proton injection.

![](dt=2025-08-08/ht=01/ceaec330154020a288884a81a6bdc835e86babaa610ed0363bbb0780cbd6043e.jpg)

Figure 8. Same as Fig. 7, corresponding to pure-iron injection.

![](dt=2025-08-08/ht=01/35f208af272f764978d64f46ee20abf97ee802f5697b8f3b2ed46bb4e43fe4e3.jpg)

\[
f_{\nu}=\frac{\mathcal{E}_{\mathrm{esc}}^{\nu}}{\mathcal{E}_{\mathrm{acc}}},
\]

where \(f_{\mathrm{CR}}\) represents the fraction of energy lost by accelerated UHECRs during the propagation within the source (source efficiency), and \(f_{\nu}\) the fraction of energy transformed into neutrino energy by in-source interactions (neutrino production efficiency). Given the definitions of the efficiency parameters in Eqs. (3.4) and (3.5), we can see that the maximum interaction efficiency corresponds to both \(f_{\mathrm{CR}}\), \(f_{\nu} \simeq 1\), i.e. total conversion of accelerated CR energy in source neutrinos.

In the left panel of Fig. 7, the source efficiency \(f_{\mathrm{CR}}\) is shown for different BB temperatures in the scenarios of a pure-proton injection. Different colors correspond to different combinations of acceleration parameters \(\gamma\) and \(E_{\mathrm{cut}}\). All configurations considered saturate at a specific value of \(f_{\mathrm{CR}}\), for \(T \gtrsim 10^{6}\,\mathrm{K}\). This demonstrates that interactions within the source environment are very efficient during the early stages after the merger.

However, scenarios with \(\gamma \gtrsim 1.5\) are characterized by a maximum fraction of energy lost of \(f_{\mathrm{CR}} \lesssim 10^{-2}\), due to the large number of low energy protons escaping the source environment almost undisturbed. In the right panel of Fig. 7, the neutrino production efficiency as defined in Eq. (3.5) is shown, for the same combinations of parameters in the left panel. The neutrino production

efficiency is clearly increasing as a function of the increasing temperature (decreasing time) of the source environment. Since protons can only produce neutrinos through photopion production, we observe that the neutrino production saturates at the same temperature as the one for protons. Most of the configurations considered saturate at \( f_{\nu} \simeq 0.1 \), corresponding to a \( \sim 10\% \) conversion of cosmic ray energy into neutrino flux.

The observed saturation of neutrino production is consistent with the known Waxman-Bahzell bound for source environments optically thin to photopion production [37, 38]. As discussed earlier, scenarios characterized by \( \gamma = 2.5 \) correspond to \( f_{\nu} \simeq 10^{-3} \), as a result of the fact that the escape from the source is favored over interactions.

In Fig. 8 the same quantities are shown, corresponding to a pure iron injection. In the left panel of Fig. 8, the evolution of \( f_{\mathrm{CR}} \) with the BB temperature is shown. In this case, all scenarios saturate to \( f_{\mathrm{CR}} \simeq 1 \), for \( T > 10^{6} \, \mathrm{K} \), as can be expected from the total interaction length in Fig. 3. We observe that the photodisintegration increases the source conversion efficiency in cosmic rays. The neutrino production is almost unchanged for scenarios characterized by a low \( \gamma \) value or a large \( E_{\mathrm{cut}} \) value.

In contrast, scenarios characterized by a large value of \( \gamma \) or a low value of \( E_{\mathrm{cut}} \) (see red and green points in the right panel of Fig. 8) show reduced neutrino production when \( T \lesssim 10^{5} \, \mathrm{K} \). In fact, as shown in Fig. 3, the presence of the low-energy minimum in \( \lambda_{\mathrm{Fe}} \), associated to photodisintegration, corresponds to a low escape rate of iron nuclei, which continue to interact with BB photons by disintegrating, but without producing neutrinos.

We note here that the large interaction efficiency in the case of iron nuclei will have an effect on the normalization of the propagated fluxes, compared to the proton scenario. In fact, in order to reproduce the observed UHECR flux, a higher normalization factor will be required with respect to the proton case, resulting in a higher injection rate at the source and an increased neutrino flux on Earth in the case of iron acceleration at the source. We will discuss these effects in the next section.

We also note here that in BNS environments, nuclei heavier than the ones belonging to the iron group should be taken into account, as due to \( r \)-processes. We plan to include their treatment, as for instance done already in [39], in future works.

# 3.2 Extragalactic propagation

The escaped spectra described in the previous section are used as an input for the extragalactic propagation, in order to compute the expected diffuse spectra at Earth, from a population of BNS mergers. We use the original version of the simulation framework SimProp-v2r4. UHECRs are propagated in the extragalactic space taking into account stochastic interactions with the CMB and the EBL from [40], nuclear decay and redshift energy loss. The production and propagation of cosmogenic neutrinos (i.e. neutrinos produced by interactions between UHECRs and cosmic photon fields) are also simulated. The propagation of high-energy neutrinos produced within the source environment is taken into account by considering the energy evolution on cosmological distances, given by

\[
\frac{dE_{\nu}}{dz}=\frac{E_{\nu}}{1+z},
\]

where \( E_{\nu} \) is the neutrino energy, \( z \) is the redshift and \( E_{\nu,0} = E_{\nu}(z = 0) \) is the energy expected at Earth. Therefore, the energy spectrum of astrophysical neutrinos at Earth is given by

\[
J_{\nu,\mathrm{source}}(E_{\nu,0})=\frac{c}{4\pi H_0}\int_0^{z_{\mathrm{max}}}dz\frac{\xi(z)}{\sqrt{\Omega_{\mathrm{m}}(1+z)^3+\Omega_{\Lambda}}}Q_{\mathrm{esc}}^{\nu}(E_{\nu}(z)),
\]

where \( z_{\mathrm{max}} \) is the maximum redshift considered for the extragalactic propagation (in this work, \( z_{\mathrm{max}}=6 \)), \( H_{0}\simeq70\,\mathrm{km/s/Mpc} \) is the Hubble constant at present time, \( \Omega_{\mathrm{m}}\simeq0.3 \) is the density matter and \( \Omega_{\Lambda}\simeq0.7 \) is the dark energy density, in the standard cosmological model (ΛCDM), and \( Q_{\mathrm{esc}}^{\nu}(E_{\nu}(z)) \) is the neutrino production rate of the source (i.e. the number of produced neutrinos per unit energy, time and volume).

The function \( \xi(z) \) describes the redshift evolution of the source distribution, and it is generically of the form \( \xi(z)\propto(1+z)^{m} \), where \( m \) is the source evolution index. In this work, we consider the star formation rate evolution (SFR) parametrisation like in [41],

\[
\xi(z)=\left\{\begin{array}{ll}(1+z)^{3.4},&0\leq z\leq1;\\2^{3.7}(1+z)^{-0.3},&1\leq z\leq4;\\2^{3.7}5^{3.2}(1+z)^{-3.5},&4\leq z\leq6,\end{array}\right.
\]

together with the case of no source cosmological evolution \( m=
0 \).

In order to compare our simulations with measured UHECR and neutrino fluxes, we scale the propagated UHECR all-particle spectrum. In particular, we require that the propagated all-particle UHECR spectrum at \( E=E_{\mathrm{cut}} \) corresponds to the UHECR flux measured by the Pierre Auger Observatory [42] at the same energy\(^{4}\) (see Appendix C). In this way, we fix the accelerated injection rate \( Q_{0,\mathrm{acc}}^{\mathrm{A}} \) in Eq. (3.1).

For our reference scenario with protons we obtain \( Q_{0,\mathrm{acc}}^{\mathrm{p}}=8.2\cdot10^{39}\,\mathrm{erg^{-1}\,Mpc^{-3}\,yr^{-1}} \), while for the same scenario with iron nuclei we obtain \( Q_{0,\mathrm{acc}}^{\mathrm{Fe}}=3.4\cdot10^{41}\,\mathrm{erg^{-1}\,Mpc^{-3}\,yr^{-1}} \). We then introduce the scale factor \( g \), such that

\[
J_{\mathrm{prop}}(E=E_{\mathrm{cut}})=g\cdot J_{\mathrm{exp}}(E=E_{\mathrm{cut}}),
\]

where \( J_{\mathrm{prop}}(E)=\sum_{A}J_{\mathrm{prop}}^{A}(E) \). In this way, the scale factor \( g \) can be used to soften the assumption on the relative contribution of BNS mergers to the observed UHECR flux in the energy range below the ankle. The following values for the scale factor \( g \) will be considered: \( \log g=-5,...,0 \) with \( \Delta\log g=0.1 \).

We define a reference scenario characterized by the following choices: pure-proton injection, source temperature of \( T=10^{5}\,\mathrm{K} \), and spectral parameters \( \gamma=1.5 \) and \( E_{\mathrm{cut}}=10^{18.5}\,\mathrm{eV} \); we also fix the normalization factor \( g=0.4 \), as roughly what is obtained from the fit of the nuclear species at Earth at energy \( 10^{18.5}\,\mathrm{eV} \) in [43], for the proton component.

In Fig. 9 the propagated source neutrino spectra for the reference scenario with source evolution \( m=0 \) (left) and SFR evolution in Eq. (3.8) are shown. In both panels, the (single flavor) source neutrino spectrum and the cosmogenic neutrino spectrum (black line) are shown. The measured neutrino flux and cosmogenic neutrino limit by IceCube [44] as well as the cosmogenic neutrino limit by the Pierre Auger Observatory [45] are also shown.

The source neutrino fluxes are compatible with the upper limits from IceCube, but only partially compatible with the observed neutrinos for \( E_{\nu}\gtrsim10^{6}\,\mathrm{GeV} \), in the case \( m=0 \). The produced cosmogenic neutrino flux is several orders of magnitude below the experimental limits. In the right panel the scenario corresponding to the SFR source evolution is shown. The production of cosmogenic neutrinos is slightly enhanced in this case, due to increased interactions with the cosmic fields corresponding to the SFR source evolution.

However, the production of source neutrinos also increases, making the spectrum of propagated neutrino higher than the observed flux.

Figure 9. Single-flavor neutrino energy spectra for a pure-proton injection, for a population of BNS mergers at a fixed time after the merger (no-evolution (left) and SFR evolution (right panel) cases): propagated source neutrinos (blue line) and cosmogenic neutrinos (black line). Observed neutrino flux and cosmogenic neutrino limit by IceCube [44] and the cosmogenic neutrino limit by the Pierre Auger Observatory [45] are also shown.

![](dt=2025-08-08/ht=01/48eb5e0887cc9e4a00e39fa1d4f12b0203a582437e6083b173ec4af8dca0a94f.jpg)

Figure 10. Same as in Fig. 9, for a pure-iron injection. Note the different range of the y-axes with respect to the proton scenarios.

![](dt=2025-08-08/ht=01/07109ae45d141a64e669885c07ec764382acfa877d3fe045f0a1478fd0561430.jpg)

In Fig. 10, the reference scenario of Fig. 9 is shown for a pure iron composition at the acceleration. In this case, the scaling factor is fixed to \( g = 0.1 \). The position of the high energy cutoff is shifted to lower energy with respect to the pure-proton case, due to the fact that most of the UHECRs that generate neutrinos are protons produced in the disintegration of iron nuclei.

The low-energy region of the spectra is enhanced with respect to the proton scenario, and this can be explained by considering the behavior of the interaction lengths for small values of the Lorentz factors in Figs. 2 and 3. Moreover, the general effect of a heavier composition injected in the source environment is an increase of the number of produced neutrinos. As discussed in Sec. 3.1, this is due to the normalization to the observed UHECR spectrum.

Photodisintegration processes within the source environment suppress more intensively the UHECRs at the escape, compared to the case of pure protons undergoing photomeson production in the source. Therefore, a larger rate of acceleration at the source is required. The result is an increased neutrino flux at the escape from the source, and then on Earth. The SFR scenario is shown in the right panel of Fig. 10 and it shows the same effects described in the proton case.

In order to quantify the compatibility of the obtained spectra of source and cosmogenic neutrinos with the available data and limits, we introduce two control quantities. We define the neutrino spectral ratio at \( E_{\nu}=10^{6}\,\mathrm{GeV} \) as

\[
R_{\nu}(E_{\nu}=10^{6}\mathrm{GeV})=\frac{J_{\nu,\mathrm{source}}(E_{\nu}=10^{6}\mathrm{GeV})}{J_{\mathrm{IceCube}}(E_{\nu}=10^{6}\mathrm{GeV})},
\]

where \( J_{\nu,\mathrm{source}} \) is the propagated source-neutrino spectrum and \( J_{\mathrm{IceCube}} \) is the observed IceCube neutrino spectrum previously introduced. Additionally, given the total neutrino exposure \( \mathcal{E}(E_{\nu}) \) of the Pierre Auger Observatory [45], we calculate the expected number of neutrinos as

\[
N_{\nu}=\int dE_{\nu}\mathcal{E}(E_{\nu})\left[J_{\nu,\mathrm{source}}(E_{\nu})+J_{\nu,\mathrm{cosmo}}(E_{\nu})\right].
\]

and we compare this value to 2.39, being this the Feldman-Cousins factor for non-observation of events in the absence of expected background [46]. We compute both these quantities in Eqs. (3.10) and (3.11) for the reference scenario and for both the source evolution parameterizations. We then vary one parameter, keeping the others unchanged, to investigate their impact on our control quantities.

In Fig. 11, we show the parametric study of the quantities defined in Eqs. (3.10) and (3.11) for a pure proton composition at the acceleration and evolution parameter \( m=0 \). In each row of Fig. 11, one parameter among \( \gamma \), \( E_{\mathrm{cut}} \) and \( g \) is varied for all the source temperatures \( T \) (indicated as colored lines). In particular, in the first row the scan over \( g \), in the second row the scan over \( \gamma \), and in the third row the scan over \( E_{\mathrm{cut}} \) are shown. In the first (second) column of Fig.

11 the neutrino spectral ratios at \( E_{\nu}=10^{6}\,\mathrm{GeV} \) (the number of neutrinos) are shown. Gray lines in the first column indicate the \( 0.01\% \), \( 1.0\% \) and \( 100.0\% \) contributions to the measured IceCube neutrino flux, while in the second column the Feldman-Cousins factor for the non observation of neutrinos is shown.

We can immediately see that the two control quantities clearly depend on the temperature of the local photon field, as already discussed above and shown in other studies investigating the neutrino production as a function of the density of photons in the source environment [47–50]. The factor \( g \) simply scales the UHECR flux at Earth and, as a consequence, the same scaling is found both in the source and cosmogenic neutrinos.

The spectral shape of the UHECR spectrum at the acceleration affects the spectral ratio at \( E_{\nu}=10^{6}\,\mathrm{GeV} \) more than the expected number of neutrinos above \( E_{\nu}=10^{8}\,\mathrm{GeV} \). This is mostly due to the fact that a softer UHECR spectrum involves a larger number of low
-energy UHECR protons, that is reflected in a larger number of neutrinos at \( E_{\nu}=10^{6}\,\mathrm{GeV} \). This is not contradictory to the results shown in Sec. 2.1: the neutrino production efficiency \( f_{\nu} \) is given by the total neutrino emissivity of the source, while in Fig.

11 we calculate the control quantity \( R_{\nu} \) at a specific energy value to compare our simulated fluxes with available experimental results. The same cannot be seen for the number of neutrinos, defined above \( 10^{8}\,\mathrm{GeV} \) due to the experimental neutrino exposure \( \mathcal{E}(E_{\nu}) \) from [45], because only the high-energy part of the expected neutrino flux (due to the highest energy region of the UHECR spectrum) could be possibly measured.

Both the spectral ratio at \( E_{\nu}=10^{6}\,\mathrm{GeV} \) and the number of neutrinos above \( 10^{8}\,\mathrm{GeV} \) increase when \( E_{\mathrm{cut}} \) decreases. This is due to the fact that for low values of \( E_{\mathrm{cut}} \) a higher normalization is required to account for the observed UHECR flux.

In Fig. 12 the parameter scan is shown for the case of pure iron injection at the acceleration in the source environment. In this case, the production of neutrinos is even more

Figure 11. Left column: neutrino spectral ratio, calculated at \( E_{\nu}=10^{6}\,\mathrm{GeV} \), as defined in Eq. (3.10). Right column: number of neutrinos, as defined in Eq. (3.11). These quantities are shown as a function of the normalization factor \( g \) (upper), the spectral index \( \gamma \) (central) and the high energy cutoff \( E_{\mathrm{cut}} \) (bottom panels). Colors correspond to different temperatures. In the right panels, gray lines indicate some reference ratios. The Feldman-Cousins factor for non-observation of events is indicated with a gray line in the left panels. The injected UHECR composition in the source environment is pure-proton, and the evolution parameter is \( m=0 \).

![](dt=2025-08-08/ht=01/df12b088a73a2bbed79286d5a7866c0fc451b7e0d6b4eb1cb43aef8acbc7a9c6.jpg)

dependent on the temperature of the BB than in the pure-proton scenario. We also find that the number of neutrinos increases rapidly as the UHECR spectral index becomes softer. A softer spectral index corresponds to a larger number of low energy iron nuclei in the source environment, compared to the high-energy ones. Therefore, due to photodisintegration, a much larger normalization is required than in the pure proton scenario. The same effect can be seen when \( E_{\mathrm{cut}} \) decreases. The cases of SFR source evolution are shown in Appendix D in

Figure 12. Same as in Fig. 11, for the case of pure-iron injection and evolution parameter \( m=0 \).

![](dt=2025-08-08/ht=01/fd6722da68731517df09ca2832f9f6135643217643d40056822f9c960d60a93d.jpg)

Fig. 22 for proton injection and Fig. 23 for iron injection. This source evolution assumption slightly increases the neutrino flux with respect to the \( m=0 \) case.

# 3.4 Source temporal evolution

In Sec. 2 we have discussed the temporal evolution of the quantities that characterize the source environment, after the merger time. In particular, the evolution of SEDs in Eqs. (2.4) and (2.9) and of the source radius in Eq. (2.7) is determined by the relation between the time after the merger and the BB temperature in Eq. (2.3). To account for the evolution of the interaction region, and thus the evolution of neutrino production, we integrate the energy spectra of the escaped particles over the time after the merger. We consider the time steps shown in Fig.

1: the first time step corresponds to \( t=10^{2}\,\mathrm{s} \) after the merger ( \( T=10^{8}\,\mathrm{K} \) ), and the last time step corresponds to \( t=10^{4}\,\mathrm{s} \) after the merger ( \( T=10^{4}\,\mathrm{K} \) ). As discussed

Figure 13. Single-flavor neutrino energy spectra for a pure-proton injection for a population of BNS mergers where the temporal evolution is taken into account (no-evolution (left) and SFR evolution (right panel) cases): propagated source neutrinos (blue line) and cosmogenic neutrinos (black line). The observed neutrino flux and cosmogenic neutrino limit by IceCube [44] and the cosmogenic neutrino limit by the Pierre Auger Observatory [45] are also shown.

![](dt=2025-08-08/ht=01/84814dbd665ef14415dc1a5c2af86510768cc42dcb715bd27910c18d22340da7.jpg)

Figure 14. Same as in Fig. 13, for a pure-iron injection. Note the different range of the y-axes with respect to the proton scenarios.

![](dt=2025-08-08/ht=01/19cd500878896afb7b0e78cb75a878e5074baaf686f5e228478824f688aa44c3.jpg)

in Sec. 3.1, the maximum acceleration energy depends slightly on the time after the merger. Therefore, as done in the previous sections, we consider \( E_{\mathrm{cut}} \) as a fixed parameter of the acceleration process. We adopt the same normalization procedure described in Sec. 3.2, and for normalizing to the observed UHECR flux at \( E_{\mathrm{cut}} \), we use the all-particle propagated spectrum integrated over the time evolution of the source. The reference scenarios are again characterized by the spectral parameter \( \gamma=1.5 \) and \( E_{\mathrm{cut}}=10^{18.5}\,\mathrm{eV} \). The same definition of the scale factor \( g \) in Eq. 3.9 is used: for the reference scenario \( g=0.4 \) in the proton injection case and \( g=0.1 \) in the iron case.

In Fig. 13 the propagated source neutrino spectra for the reference scenario for a population of BNS mergers with evolution \( m=0 \) (left) and SFR evolution (right) are shown. Experimental data and limits are the same of Fig. 9. In Fig. 14 the pure-iron injection scenarios are shown. The general results obtained in Sec. 3.2 apply also here. However, some differences are introduced when the time evolution of the source emission is considered. All the source neutrino spectra in Figs. 13 and 14 (in particular the iron scenarios) are suppressed with respect to the corresponding spectra in Figs. 9 and 10. This is due to the normalization of the propagated CRs: in particular, if we consider the latest times after the merger

Figure 15. Left column: neutrino spectral ratio, calculated at \( E_{\nu}=10^{6}\,\mathrm{GeV} \), as defined in Eq. (3.10). Right column: number of neutrinos, as defined in Eq. (3.11). These quantities are shown as a function of the spectral index \(\gamma\) (upper) and the high energy cutoff \(E_{\mathrm{cut}}\) (bottom panels), both integrating over the temporal evolution of the source. Several normalization factors \(g\) are considered in each panel: \(g=1.0\) (solid lines), \(g=10^{-1}\) (dashed lines), \(g=10^{-2}\) (dotted lines) and \(g=10^{-3}\) (dashed-dotted lines). In the right panels, gray lines indicate some reference ratios. The Feldman-Cousins factor for non-observation of events is indicated with a gray line in the left panels. The injected CR composition in the source environment is pure-proton, and the evolution parameter is \(m=0\).

![](dt=2025-08-08/ht=01/72f49368bdbca061737033c2a601639c227e39997fdbb481bde11b201862c83c.jpg)

\(T\sim10^{4}\,\mathrm{K}\) most of the injected nuclei escape from the source region without interacting; therefore, a smaller normalization factor is needed to account for the contribution to the observed CR spectrum at \(10^{18.5}\,\mathrm{eV}\), with respect to the case in which a fixed time after the merger was considered. In addition, different slopes in the low energy part of source neutrino spectra are obtained as a result of combining different neutrino spectra escaped from the source (see Figs. 5 and 6 for the energy spectra
of the escaped CRs and neutrinos from the source environment).

We evaluate the two control quantities defined in Eqs. (3.10) and (3.11) for the source neutrino spectra discussed above. As done in Sec. 3.3, we vary one parameter, keeping the other ones unchanged. In Fig. 15 the parametric study of the control quantities is shown for a pure-proton composition at the acceleration and source evolution \(m=0\). Differently from Fig. 11, in each panel of Fig. 15 different values of \(g\) are shown together.

The qualitative behaviors of \(R_{\nu}(E_{\nu}=10^{6}\,\mathrm{GeV})\) and \(N_{\nu}\) as functions of \(\gamma\) and \(E_{\mathrm{cut}}\) are the same discussed for a fixed value of the source temperature. Moreover, \(R_{\nu}(E_{\nu}=10^{6}\,\mathrm{GeV})\) and \(N_{\nu}\) are reduced when the scaling factor \(g\) decreases. It can be seen that the most stringent constraints are given by \(R_{\nu}(E_{\nu}=10^{6}\,\mathrm{GeV})\), while \(N_{\nu}\) almost always agrees with the experimental limits and

Figure 16. Same as in Fig. 15, for the case of pure-iron injection and evolution parameter \( m=0 \).

![](dt=2025-08-08/ht=01/9757e48ab04cd12a51f8ebd86d31dc1f5c7c15f5ed54721f31df72f95cc9aa49.jpg)

observations, even for the most extreme scenario \( g=1.0 \). In Fig. 16 the same results are shown for the case of pure-iron acceleration. In general, all the control quantities in Figs. 15 and 16 are in better agreement with the neutrino data and limits than in Figs. 11 and 12. This is because the integration over the source temperature has the consequence that the normalization is mostly defined by the CR flux at low temperature (i.e. the latest to leave the source). Therefore, a smaller UHECR normalization is required and smaller neutrino fluxes are obtained. The cases of SFR source evolution are shown in Appendix D in Figs. 24 and 25.

# 4 Discussion and conclusions

In this work, we have realized a source-propagation model where the considered source environment is the end-state of the merger of binary-neutron-star systems. The Monte Carlo code SimProp-v2r4 has been adapted to simulate the in-source interactions, while the original version of the code was used for the extragalactic propagation.

We have assumed, as supported by [22], that the neutrino production does not take place in relativistic jets, but in the shocks that might be generated behind the ejected material; in addition, we have taken into account cosmic rays that can reach energies possibly contributing to the region just below the ankle in the measured energy spectrum, as shown in [27].

We have considered the non-thermal and thermal spectral energy density of the local photons, generated by the synchrotron emission and the nuclear decays of the unstable nuclear species synthesized in the ejected material, respectively. For doing this, we have taken as a reference the event GW170817 [16], the only

event for which a gravitational wave and the electromagnetic counterpart have been detected nowadays.

As a result of the study of the propagation of UHECRs in the BNS remnant environment, we have shown that interactions with the non-thermal photons are not efficient enough to produce neutrinos or photodisintegrate the fallback material (see Fig. 4). For this reason, we have neglected this photon field in our in-source simulations. On the other hand, blackbody photons can trigger interactions with both injected protons and iron nuclei.

In this case, the opacity of the source is greater than one for \( T \gtrsim 10^{5} \) K (\( t \lesssim 10^{3.5} \) s) for protons, and for \( T \gtrsim 10^{4} \) K (\( t \lesssim 10^{4} \) s) for iron nuclei. In particular, for Lorentz factors \( \Gamma \simeq 10^{9} \), the fallback material interacts with the thermal photon field through photomeson production and photodisintegration. However, very low- and very high-energy nuclei can leave the source environment undisturbed (see total interaction lengths in Figs. 2 and 3).

As an outcome, we have shown that the efficiency of interactions is higher in the early stages after the merger (i.e. \( T \gtrsim 10^{6} \) K) than at later stages. In particular, scenarios of proton injection saturate at 10% conversion of cosmic ray energy into neutrinos. However, in the case of iron nuclei we observe that photodisintegration increases the source conversion efficiency in cosmic rays, while the neutrino production is almost unchanged.

In Sections. 3.3 and 3.4, we have quantified the diffuse neutrino flux at Earth (the neutrinos produced in the source environment and the ones produced in the extragalactic propagation) as a function of the temperature of the black body, as well as depending on the CR spectral parameters, for a population of identical BNS mergers. We have shown that in general the observed neutrino flux cannot be associated with cosmogenic neutrinos.

The neutrino spectral ratio \( R_{\nu}(E_{\nu} = 10^{6} \) GeV) and the number of neutrinos \( N_{\nu} \) show a dependence on the photon-field temperature, and in general very high temperatures correspond to a large neutrino flux at Earth. We calculated the propagated neutrino fluxes taking into account the time evolution of the source environment.

For both compositions considered, in order to avoid overshooting of the measured neutrino flux, the high energy cutoff must be \( \gtrsim 10^{18} \) eV, if the reference values of the scaling \( g \) of the cosmic-ray expected spectra are considered. The effect of the variation of the spectral index \( \gamma \) is almost independent of the composition considered and the scaling factor \( g \): for both protons and iron nuclei we find that all the values of gamma are acceptable for \( g \lesssim 0.4 \). These results depend weakly on the source evolution model adopted.

The efficiency in producing high energy astrophysical neutrinos can be, as a first approximation, connected to the ratio of the total interaction length of cosmic rays to the typical size of the interaction region. In this study we have assumed that the typical escape length is given by the radius of the ejected material in Eq. (2.7), and the typical escape time is then given by \( \tau_{\mathrm{esc}}(t) = \beta_{\mathrm{ej}} \), where \( \beta_{\mathrm{ej}} = 0.3 \).

This corresponds to a source size ranging from \( \lambda_{\mathrm{esc}} \approx 10^{12} \) cm immediately after the merger, to \( \lambda_{\mathrm{esc}} \approx 10^{14} \) cm in the last considered stage. Being the typical values of \( \beta_{\mathrm{ej}} = 0.1 - 0.3 \) (see [51, 52]), different values of \( \beta_{\mathrm{ej}} \) with respect to what assumed here will only marginally affect the production of neutrinos.

Another important approximation made regarding the confinement of accelerated cosmic rays is the fact that the escape time does not depend on the rigidity of the particles. The presence of a magnetic field in the post-merger environment (with strength \( \mathcal{O}(\mathrm{mG}) \lesssim B \lesssim \mathcal{O}(\mathrm{G}) \), see [22, 27]) could lead to a longer confinement time for low rigidity nuclei. This effect could be particularly important in the case of heavy nuclei: a longer confinement time would imply more interactions with local fields, and thus higher photodisintegration and photopion efficiency. Additionally, synchrotron energy losses should be evaluated in studying

the neutrino production efficiency of this class of sources. Future developments of the present work might include these details.

The comparison of the propagated diffuse UHECR spectrum to the measured flux offers the possibility of studying the source-parameters, such as the baryonic loading \(\eta\), as well as the number density of BNS mergers. The CR emissivity at acceleration is related to the number density of mergers as

\[
\mathcal{E}_{\mathrm{acc}}=E_{\mathrm{acc}}\dot{n},
\]

where \(E_{\mathrm{acc}}\
) is the total accelerated CR energy and \(\dot{n}\) the event rate per volume (i.e. the number of mergers per unit of volume and unit of time) of BNS mergers. The energy in cosmic rays is related to the fallback luminosity \(\mathcal{L}_{\mathrm{fb}}\), being this the luminosity of the outflow powered by accretion. We parameterize the fall-back luminosity as in the optimistic scenario of [22] (i.e. the ejected mass by the merger is \(10^{-4}\,M_{\odot}\) and \(\beta_{\mathrm{ej}}=0.3\)) obtaining

\[
\mathcal{L}_{\mathrm{fb}}(t)=1.3\cdot10^{43}\cdot\left(\frac{t}{10^3\mathrm{s}}\right)^{-5/3}\mathrm{ergs}^{-1},
\]

where \(t\) is the time after the merger; the time dependence is taken as in [22], corresponding to the fall-back mass dynamics (see also [51, 53]). We then define the baryonic loading \(\eta\) as the conversion coefficient of fall-back material into accelerated UHECRs, i.e. \(\mathcal{L}_{\mathrm{acc}}=\eta\cdot\mathcal{L}_{\mathrm{fb}}\). Thus, the total accelerated CR energy is given by

\[
E_{\mathrm{acc}}=\int dt\mathcal{L}_{\mathrm{acc}}(t)=\eta\int dt\mathcal{L}_{\mathrm{fb}}(t),
\]

and then

\[
\frac{\mathcal{E}_{\mathrm{acc}}}{\dot{n}\eta}=\int dt\mathcal{L}_{\mathrm{fb}}(t).
\]

The CR luminosity at acceleration is converted into the propagated CR spectrum by in-source and extragalactic interactions. As shown in Sec. 3.2, we scale the propagated CR spectrum to the observed spectrum by the coefficient \(g\), defined in Eq. (3.9). Therefore, for a given value of \(g\) corresponding to a propagated CR flux that does not overshoot the CR data, different combinations of the parameters \(\eta\) and \(\dot{n}\) are acceptable, i.e. \(g=g(\dot{n}\cdot\eta)\).

By construction, the possible values of \(g\) are limited by the condition \(g\leq1\) and, in addition, thanks to the model developed in this work, by the fact that the corresponding neutrino flux must be such that \(R_{\nu}(E_{\nu}=10^{6}\,\mathrm{GeV})\leq1\) (see Eq. (3.10)) and \(N_{\nu}\leq2.39\) (see Eq. (3.11)).

In Fig. 17 the possible values of \(g(\dot{n}\cdot\eta)\) are shown for different injection scenarios at the acceleration; in other words, the scenarios corresponding to each value of \(g\) are degenerate in terms of the product of \(\eta\) and \(\dot{n}\), as shown in Eq. (4.4). The acceleration scenarios shown in Fig. 17 are \(\gamma=1.5\) for the spectral index and \(E_{\mathrm{cut}}=10^{17.5}\,\mathrm{eV}\) (upper panels) and \(E_{\mathrm{cut}}=10^{18.5}\,\mathrm{eV}\) (lower panels).

The left panels correspond to pure-proton composition at the acceleration, and the right panels to pure-iron composition. The BNS merger event rate per volume estimated by [16, 54] is shown as an green region. We show the case of source evolution model \(m=0\); the corresponding scenarios for the SFR evolution are shown in Fig. 26. We also indicate some reference values of \(\log_{10}g\) with gray lines in each panel of Figs. 17 and 26.

In particular, the gray lines for \(\log_{10}g=0\) correspond to the value of the product \(\eta\dot{n}\), for which the CR flux is maximum at Earth in the region below the ankle (i.e. it saturates the measured flux at the fixed energy). We can immediately notice that, due to the more intense energy loss experienced by nuclei with respect to protons (in the source and

Figure 17. Scaling parameter \( g \) as a function of the baryonic loading \( \eta \) and the BNS merging event rate per volume \( \hat{n} \), as defined in Eq. (4.4). The value of \( \log_{10}g \) is indicated by the color bar. The gray lines correspond to the reference values of \( \log_{10}g \), as shown in the panels. The acceleration parameters of the sources are \( \gamma=1.5 \) for the spectral index and \( E_{\mathrm{cut}}=10^{17.5}\,\mathrm{eV} \) (upper panels) and \( E_{\mathrm{cut}}=10^{18.5}\,\mathrm{eV} \) (lower panels). The left panels correspond to pure-proton injection and the right panels to pure-iron injection. The BNS merger event rate per volume estimated by [16, 54] is shown as an green region. The cosmological source evolution model is \( m=0 \).

![](dt=2025-08-08/ht=01/ff53bb605ce0a37a4ec7968a01bd2af13261bbea5ab8aa8a70a47aab5bdd87c7.jpg)

in the extragalactic propagation) the required \( \eta\,\hat{n} \) is in general larger. For \( E_{\mathrm{cut}}=10^{17.5}\,\mathrm{eV} \), a scaling factor \( g\lesssim10^{-1} \) is required by the constraint on the number of neutrinos produced. This because a larger normalization factor is needed, corresponding to a larger neutrino flux. On the other hand, for \( E_{\mathrm{cut}}=10^{18.5}\,\mathrm{eV} \) the CR saturation scenario \( g=1.0 \) is possible and it corresponds to a baryonic loading \( \eta\simeq10^{-4}-10^{-6} \) for the values of \( \hat{n} \) in the allowed region. In the SFR scenario, due to the large number of high-redshift sources, for the same values of \( \hat{n} \) we obtain a slightly higher required baryonic.

Differently from what done in this work, in [22] a baryonic loading \( \eta\,\simeq\,0.1 \) is fixed a-priori; for an event rate per volume of \( \hat{n}\simeq1000\,\mathrm{Gpc^{-3}\,y r^{-1}} \), a contribution to the observed IceCube flux of \( \simeq10\% \) is therefore attributed to BNS mergers. With our parametric study, we are instead able to consistently constrain the neutrino production efficiency from BNS mergers by using CR data between the knee and the ankle.

In particular, we derive that corresponding to \( \hat{n}=1000\,\mathrm{Gpc^{-3}\,y r^{-1}} \), a baryonic loading of \( \eta\lesssim10^{-5} \) is required in most of the considered scenarios, and in addition we can account for the contribution of the BNS to the cosmic rays below the ankle, if a scaling factor \( g\lesssim10^{-1} \) for \( E_{\mathrm{cut}}\simeq10^{17.5}\,\mathrm{eV} \), and \( g\lesssim1 \)

for \( E_{\mathrm{cut}} \simeq 10^{18.5} \, \mathrm{eV} \) are considered respectively.

In conclusion, thanks to the source-propagation model proposed in this work, we have shown that a region of the parameter space of the baryonic loading and rate of merger events can be excluded, depending on what fraction of the cosmic ray flux below the ankle is ascribed to BNS mergers and to the constraints from neutrino measurements and upper limits. Further measurements by LIGO/Virgo, such as next-generation gravitational wave and neutrino detectors [55-57], might improve the constraining power of such a model, and future possible multimensenger observations might provide further tests of source-propagation models as the one here developed.

# Acknowledgments

S.R. and G.S. acknowledge support by the Bundesministerium für Bildung und Forschung, under grants 05A20GU2 and 05A23GU3. The authors acknowledge their participation to the Pierre Auger Collaboration.

# A Ballistic approximation

The validity of the ballistic approximation can be expressed considering the Larmor radius

\[
r_{\mathrm{B}}=3.1\cdot10^{12}\left(\frac{E/Z}{10^{15}\mathrm{eV}}\right)\left(\frac{B}{1\mathrm{G}}\right)^{-1}\mathrm{cm},
\]

where \(B\) is the magnetic field within the interaction region and \(E\) and \(Z\) are the energy and the atomic number of the CR, respectively. Given the radius of interaction region in Eq. (2.7), the ballistic approximation is defined by \(\lambda_{\mathrm{esc}}(t)\lesssim r_{g}\) . Therefore, the condition on the CR Lorentz factor \(\Gamma\) is

\[
\Gamma\gtrsim3\cdot10^{5}\left(\frac{Z}{A}\right)\left(\frac{t}{10^{2}\mathrm{s}}\right)\left(\frac{B}{1\mathrm{G}}\right),
\]

where \(A\) is the CR atomic mass. If we consider that \(Z/A\sim1/2\) and that the neutrino production is maximal for \(t\sim10^{2}\,\mathrm{s}\) , considering a limiting magnetic field strength value equal to \(B\sim1\,\mathrm{G}\) (see [22, 27]), we obtain that the ballistic approximation can be used for \(\Gamma\gtrsim10^{5}\) values of the CR Lorentz factor ( \(E\gtrsim10
^{14}\,\mathrm{eV}\) for protons and \(E\gtrsim6\cdot10^{15}\,\mathrm{eV}\) for iron nuclei).

# B UHECR interaction rate

In this appendix we report some details of the computation of UHECR interactions. Due to the very high relativistic boost of UHECRs, cosmic photons appear as high-energy gamma ray in the rest frame of the particle. Therefore, photohadronic interactions between UHECRs and cosmic photons become possible, and the corresponding interaction rate \( \tau_{ij}^{-1} \) for the process \( i \) between the cosmic ray nucleus and the background photon field \( j \) is given by

\[
\tau_{ij}^{-1}=\frac{c}{2\Gamma^2}\int_{\epsilon_{\mathrm{th}}}^{\infty}d\epsilon\sigma_i(\epsilon)\epsilon\int_{\epsilon/2\Gamma}^{\infty}d\bar{\epsilon}n_j(\bar{\epsilon})\bar{\epsilon}^{-2},
\]

where \( \bar{\epsilon} \) is the photon energy in the laboratory rest frame, \( n_{j}(\bar{\epsilon}) \) is the photon spectral energy density (SED, i.e. the number of photons per unit of volume and energy) in the laboratory

Figure 18. Energy spectra for a pure-proton injection: injected in the source (red line), escaped from the source (blue line) and propagated at Earth (black line). Observed cosmic ray flux by the Pierre Auger Observatory [60] is also shown (black dots). Left panel refers to no cosmological source evolution, right panel refers to SFR source evolution in Eq. (3.8).

![](dt=2025-08-08/ht=01/e7550688b6468e32b0c998295c8bd1ed41b29f20e33e6122c087a70ccd69d9e3.jpg)

rest frame, \(\sigma_{i}(\epsilon)\) is the total cross-section expressed as a function of the photon energy in the nucleus rest frame, \(\epsilon_{\mathrm{th}}\) is the threshold photon energy in the nucleus rest frame and \(\Gamma\) is the nucleus Lorentz factor (a complete derivation of Eq. (B.1) can be found in [58, 59]).

The interaction rate in Eq. (B.1) is a linear function in both the cross-section and photon SED. We can then compute the total interaction rate of different interaction processes as

\[
\tau_{\mathrm{tot}}^{-1}=\sum_{i,j}\tau_{ij}^{-1},
\]

where the sum is over the considered photon fields and the possible interaction processes. The corresponding probability associated with the combination \(i\) and \(j\) is

\[
p_{ij}=\frac{\tau_{ij}^{-1}}{\tau_{\mathrm{tot}}^{-1}}.
\]

The latter relation shows the fact that several interaction processes can be evaluated separately. The escape condition of a particle from the source environment can be associated with an escape rate \(\tau_{\mathrm{esc}}^{-1}\) and compared with the interaction rates to determine when the particle is free to leave the interaction region. In this work, we adopt this strategy.

# C Cosmic-ray spectra

In this appendix, the CR spectra corresponding to the predicted neutrino fluxes shown in Figs. 9 and 10 are shown. In particular, the pure proton scenario is shown in Fig. 18, and the pure iron scenario is shown in Fig. 19. In the case of protons, the accelerated (red), the escaped (blue) and the propagated (black) spectra are shown together with observed data<sup>5</sup>. For iron nuclei, the accelerated spectrum (black dotted) is shown with the total propagated spectrum (black) and the different mass components (colored lines). Observed cosmic ray flux by the Pierre Auger Observatory [60] is also shown (black dots). In Figs. 20 and 21 the CR spectra corresponding to the predicted neutrino fluxes shown in Figs. 13 and 14 are shown.

Figure 19. Same of Fig. 18, but for pure-iron injection. Injected iron nuclei are shown with black dotted line. Propagated mass groups (colored lines) and the total flux (black line) are shown. Note the different range of the y-axes with respect to the proton scenarios.

![](dt=2025-08-08/ht=01/684ca75e793dc470c9788d1602ab4b94f63f492196d886bbf8b896bb47847d72.jpg)

Figure 20. Same of Fig. 18, but integrating on the temporal evolution of the source.

![](dt=2025-08-08/ht=01/8a3199a5b00aa9278147e1c8a970cabcc3da0d9964b2b7aa1dcd7534b8d4b2fd.jpg)

Figure 21. Same as in Fig. 20, for a pure-iron injection. Note the different range of the y-axes with respect to the proton scenarios.

![](dt=2025-08-08/ht=01/3c2807951eb29529455cd50cfed14e5e6793b4d8fc87d1f245980c80cf7ea0e6.jpg)

# D Parameter study with the SFR

In this appendix, the same analysis presented in Sec. 3.3 is shown for the scenarios in which the source evolution is that of the SFR, given in Eq. (3.8). The results for proton and iron acceleration are shown in Figs. 22 and 23, respectively. No relevant differences are observed

Figure 22. Same of Fig. 11, but for the SFR source evolution in Eq. (3.8).

![](dt=2025-08-08/ht=01/fede9db546bd4590da4653a6cd94a076bb172290c0a02f82c338e238e63ad281.jpg)

with respect to the case \( m=0 \). In Figs. 24 and 25 we show the same analysis of Figs. 15 and 16 for the SFR source evolution model. In Fig. 26 the same constraints of Fig. 17 are shown for the SFR source evolution model.

Figure 23. Same of Fig. 12, but for the SFR source evolution in Eq. (3.8).

![](dt=2025-08-08/ht=01/a801f153654ccef9303107b41fb41fd44dea22f0479817d36be7e6f0d12df635.jpg)

Figure 24. Same of Fig. 15, but for the SFR source evolution in Eq. (3.8).

![](dt=2025-08-08/ht=01/6c0e187ee60d416e58b7d582b09a3b5a612489677a42fb545cffca236c2e28c6.jpg)

Figure 25. Same of Fig. 16, but for the SFR source evolution in Eq. (3.8).

![](dt=2025-08-08/ht=01/44ac27d4649ad9070cce44d6ea90c78b48aa225642cf1810f5916b21255b545e.jpg)

Figure 26. Same of Fig. 17, but for the SFR source evolution in Eq. (3.8).

![](dt=2025-08-08/ht=01/982f903277428f115f7283a7e07c74e4d6123f464612ea2d68ba54d2b8d2e1da.jpg)

# References

[1] M. G. Aartsen et al. [IceCube], "The IceCube Neutrino Observatory: Instrumentation and Online Systems", JINST 12 (2017) no.03, P03012 doi:10.1088/1748-0221/12/03/P03012 [arXiv:1612.05093 [astro-ph.IM]].  [2] M. G. Aartsen et al. [IceCube], "Evidence for High-Energy Extraterrestrial Neutrinos at the IceCube Detector", Science 342 (2013), 1242856 doi:10.1126/science.1242856 [arXiv:1311.5238 [astro-ph.HE]].  [3] R. Abbasi et al. [IceCube], "Improved Characterization of the Astrophysical Muon–neutrino Flux with 9.5 Years of IceCube Data", Astrophys. J. 928 (2022) no.

1, 50 doi:10.3847/1538-4357/928d29 [arXiv:2111.10299 [astro-ph.HE]].  [4] M. G. Aartsen et al. [IceCube], "Characteristics of the diffuse astrophysical electron and tau neutrino flux with six years of IceCube high energy cascade data", Phys. Rev. Lett. 125 (2020) no.12, 121104 doi:10.1103/PhysRevLett.125.121104 [arXiv:2001.09520 [astro-ph.HE]].  [5] M. G. Aartsen et al. [IceCube], "The contribution of Fermi-2LAC blazars to the diffuse TeV-PeV neutrino flux," Astrophys. J. 835 (2017) no.1, 45 doi:10.3847/1538-4357/835/1/45 [arXiv:1611.03874 [astro-ph.HE]].  [6] M. Ackermann, M. Ahlers, L.

Anchordoqui, M. Bustamante, A. Connolly, C. Deaconu, D. Grant, P. Gorham, F. Halzen and A. Karle, et al. "Astrophysics Uniquely Enabled by Observations of High-Energy Cosmic Neutrinos", Bull. Am. Astron. Soc. 51 (2019), 185 [arXiv:1903.04334 [astro-ph.HE]].

[7] M. G. Aartsen et al. [IceCube, Fermi-LAT, MAGIC, AGILE, ASAS-SN, HAWC, H.E.S.S., INTEGRAL, Kanata, Kiso, Kapteyn, Liverpool Telescope, Subaru, Swift NuSTAR, VERITAS and VLA/17B-403], "Multimessenger observations of a flaring blazar coincident with high-energy ne
utrino IceCube-170922A" Science 361 (2018) no.6398, eaat1378 doi:10.1126/science.aat1378 [arXiv:1807.08816 [astro-ph.HE]].[8] R. Abbasi et al. [IceCube], "Evidence for neutrino emission from the nearby active galaxy NGC 1068," Science 378 (2022) no.6619, 538-543 doi:10.1126/science.abg3395 [arXiv:2211.09972 [astro-ph.HE]].[9] R.

Stein, S. Van Velzen, M. Kowalski, A. Franckowiak, S. Gezari, J. C. A. Miller-Jones, S. Frederick, I. Sfarali, M. F. Bielemfloz and A. Horesh, et al. "A tidal disruption event coincident with a high-energy neutrino," Nature Astron. 5 (2021) no.5, 510-518 doi:10.1038/s41550-020-01295-8 [arXiv:2005.05340 [astro-ph.HE]].[10] S. Reusch, R. Stein, M. Kowalski, S. van Velzen, A. Franckowiak, C. Lunardini, K. Murase, W. Winter, J. C. A. Miller-Jones and M. M. Kasliwal, et al. "Candidate Tidal Disruption Event AT2019fdr Coincident with a High-Energy Neutrino," Phys. Rev. Lett. 128 (2022) no.

22, 221101 doi:10.1103/PhysRevLett.128.221101 [arXiv:2111.09390 [astro-ph.HE]].[11] S. van Velzen, R. Stein, M. Gilfanov, M. Kowalski, K. Hayasaki, S. Reusch, Y. Yao, S. Garrappa, A. Franckowiak and S. Gezari, et al."Establishing accretion flares from massive black holes as a major source of high-energy neutrinos", [arXiv:2111.09391 [astro-ph.HE]].[12] S. Gao, A. Fedynitch, W. Winter and M. Pohl, "Modelling the coincident observation of a high-energy neutrino and a bright blazar flare," Nature Astron. 3 (2019) no.1, 88-92 doi:10.1038/s41550-018-0410-1 [arXiv:1807.04275 [astro-ph.HE]].[13] B.

Eichmann, F. Oikonomou, S. Salvatore, R. J. Dettmar and J. Becker Tjus, "Solving the Multimessenger Puzzle of the AGN-starburst Composite Galaxy NGC 1068," Astrophys. J. 939 (2022) no.1, 43 doi:10.3847/1538-4357/ac9588 [arXiv:2207.00102 [astro-ph.HE]].[14] P. Padovani, R. Gilli, E. Resconi, C. Bellenghi and F. Henningsen, "The neutrino background from non-jetted active galactic nuclei," [arXiv:2404.05690 [astro-ph.HE]].[15] W. Winter and C. Lunardini, "Interpretation of the Observed Neutrino Emission from Three Tidal Disruption Events," Astrophys. J. 948 (2023) no.

1, 42 doi:10.3847/1538-4357/acbe9e [arXiv:2205.11538 [astro-ph.HE]].[16] B. P. Abbott et al. [LIGO Scientific and Virgo], "GW170817: Observation of Gravitational Waves from a Binary Neutron Star Inspiral," Phys. Rev. Lett. 119 (2017) no.16, 161101 doi:10.1103/PhysRevLett.119.161101 [arXiv:1710.05832 [gr-qc]].[17] B. P. Abbott et al.

[LIGO Scientific, Virgo, Fermi GBM, INTEGRAL, IceCube, AstroSat Cadmium Zinc Telluride Imager Team, IPN, Insight-Hxmt, ANTARES, Swift, AGILE Team, 1M2H Team, Dark Energy Camera GW-EM, DES, DLT40, GRAWITA, Fermi-LAT, ATCA, ASKAP, Las Cumbres Observatory Group, OzGrav, DWF (Deeper Wider Faster Program), AST3, CAASTRO, VINDOUGE, MASTER, J-GEM, GROWTH, JAGWAR, CaltechNRAO, TTU-NRAO, NuSTAR, Pan-STARRS, MAXI Team, TZAC Consortium, KU, Nordic Optical Telescope, ePESSTG, GROND, Texas Tech University, SALT Group, TOROS, BOOTES, MWA, CALET, IKI-GW Follow-up, H.E.S.S.

, LOFAR, LWA, HAWC, Pierre Auger, ALMA, Euro VLBI Team, Pi of Sky, Chandra Team at McGill University, DFN, ATLAS Telescopes, High Time Resolution Universe Survey, RIMAS, RATIR and SKA South Africa/MeerKAT], "Multi-messenger Observations of a Binary Neutron Star Merger," Astrophys. J. Lett. 848 (2017) no.2, L12 doi:10.3847/2041-8213/aa91c9 [arXiv:1710.05833 [astro-ph.HE]].[18] B. P. Abbott et al. [LIGO Scientific, Virgo, Fermi-GBM and INTEGRAL], "Gravitational Waves and Gamma-rays from a Binary Neutron Star Merger: GW170817 and GRB 170817A," Astrophys. J. Lett. 848 (2017) no.

2, L13 doi:10.3847/2041-8213/aa920c [arXiv:1710.05834 [astro-ph.HE]].

[19] M. G. Aartsen et al. [IceCube], "IceCube Search for Neutrinos Coincident with Compact Binary Mergers from LIGO-Virgo's First Gravitational-wave Transient Catalog," Astrophys. J. Lett. 898 (2020) no.1, L10 doi:10.3847/2041-8213/ab9d24 [arXiv:2004.02910 [astro-ph.HE]].[20] K. Kotera, "Ultrahigh energy cosmic ray acceleration in newly born magnetars and their associated gravitational wave signatures," Phys. Rev. D 84 (2011), 023002 doi:10.1103/PhysRevD.84.023002 [arXiv:1106.3060 [astro-ph.HE]].[21] G. R.

Farrar, "Binary neutron star mergers as the source of the highest energy cosmic rays," [arXiv:2405.12004 [astro-ph.HE]].[22] V. Decoene, C. Guépin, K. Fang, K. Kotera and B. D. Metzger, "High-energy neutrinos from fallback accretion of binary neutron star merger remnants", JCAP 04 (2020), 045 doi:10.1088/1475-7516/2020/04/045 [arXiv:1910.06578 [astro-ph.HE]].[23] K. Kotera, E. Amate and P. Blasi, "The fate of ultrahigh energy nuclei in the immediate environment of young fast-rotating pulsars", JCAP 08 (2015), 026 doi:10.1088/1475-7516/2015/08/026 [arXiv:1503.07907 [astro-ph.HE]].[24] S. S.

Kimura, K. Murase, P. Mészáros and K. Kiuchi, "High-Energy Neutrino Emission from Short Gamma-Ray Bursts: Prospects for Coincident Detection with Gravitational Waves", Astrophys. J. Lett. 848 (2017) no.1, L4 doi:10.3847/2041-8213/aad8d14 [arXiv:1708.07075 [astro-ph.HE]].[25] S. S. Kimura, K. Murase, I. Bartos, K. Ioka, I. S. Heng and P. Mészáros, "Transejecta high-energy neutrino emission from binary neutron star mergers", Phys. Rev. D 98 (2018) no.4, 043020 doi:10.1103/PhysRevD.98.043020 [arXiv:1805.11613 [astro-ph.HE]].[26] S. S. Kimura, K. Murase and P.

Mészáros, "Super-Knee Cosmic Rays from Galactic Neutron Star Merger Remnants", Astrophys. J. 866 (2018) no.1, 51 doi:10.3847/1538-4357/aadc0a [arXiv:1807.03290 [astro-ph.HE]].[27] X. Rodrigues, D. Biehl, D. Boncioli and A. M. Taylor, "Binary neutron star merger remnants as sources of cosmic rays below the "Ankle"," Astropart. Phys. 106 (2019), 10-17 doi:10.1016/j.astropartphys.2018.10.007 [arXiv:1806.01624 [astro-ph.HE]].[28] R. Margutti, K. D. Alexander, X. Xie, L. Sironi, B. D. Metzger, A. Kathirgamaraju, W. Fong, P. K. Blanchard, E. Berger and A. MacFadyen, et al.

"The Binary Neutron Star Event LIGO/Virgo GW170817 160 Days after Merger: Synchrotron Emission across the Electromagnetic Spectrum", Astrophys. J. Lett. 856 (2018) no.1, L18 doi:10.3847/2041-8213/aab2ad [arXiv:1801.03531 [astro-ph.HE]].[29] K. Greisen, "End to the cosmic ray spectrum?", Phys. Rev. Lett. 16 (1966), 748-750 doi:10.1103/PhysRevLett.16.748[30] G. T. Zatsepin and V. A. Kuzmin, "Upper limit of the spectrum of cosmic rays", JETP Lett. 4 (1966), 78-80[31] S. Rossoni, D. Boncioli and G. Sigl, "Production of high-energy neutrinos in binary-neutron-star merger events", EPJ Web Conf.

283 (2023), 04006 doi:10.1051/epjconf/202328304006[32] S. Rossoni, D. Boncioli and G. Sigl, "Study of the production of high-energy neutrinos in the environment of binary-neutron-star mergers.", PoS ICRC2021 (2021), 1004 doi:10.22323/1.395.1004[33] D. Maoz, "Astrophysics in a nutshell; 1st ed.", Princeton Univ. Press (2007)[34] R. Aloisio, D. Boncioli, A. Di Matteo, A. F. Grillo, S. Petrera and F. Salamida, "SimProp v2r4: Monte Carlo simulation code for UHECR propagation", JCAP 11 (2017), 009 doi:10.1088/1475-7516/2017/11/009 [arXiv:1705.03729 [astro-ph.HE]].

[35] J. L. Puget, F. W. Stecker and J. H. Bredekamp, "Photonuclear Interactions of Ultrahigh-Energy Cosmic Rays and their Astrophysical Consequences", Astrophys. J. 205 (1976), 638-654 doi:10.1086/154321[36] F. W. Stecker and M. H. Salamon, "Photodisintegration of ultrahigh-energy cosmic rays: A New determination", Astrophys. J. 512 (1999), 521-526 doi:10.1086/306816 [arXiv:astro-ph/9808110 [astro-ph]].[37] E. Waxman and J. N. Bahcall, "High-energy neutrinos from astrophysical sources: An Upper bound", Phys. Rev. D 59 (1999), 023002 doi:10.1103/PhysRevD.

59.023002 [arXiv:hep-ph/9807282 [hep-ph]].[38] J. N. Bahcall and E. Waxman, "High-energy astrophysical neutrinos: The Upper bound is robust", Phys. Rev. D 64 (2001), 023002 doi:10.1103/PhysRevD.64.023002[arXiv:hep-ph/9902383 [hep-ph]].[39] B. T. Zhang, K. Murase, N. Ekanger, M. Bhattacharya and S. Horiuchi, "Ultraheavy Ultrahigh-Energy Cosmic Rays," [arXiv:2405.17409 [
astro-ph.HE]].[40] F. W. Stecker, M. A. Malkan and S. T. Scully, "Intergalactic photon spectra from the far ir to the uv lyman limit for \(0 < Z < 6\) and the optical depth of the universe to high energy gamma-rays", Astrophys.

J. 648, 774-783 (2006) doi:10.1086/506188 [arXiv:astro-ph/0510449 [astro-ph]].[41] R. Alves Batista, D. Boncioli, A. di Matteo and A. van Vliet, "Secondary neutrino and gamma-ray fluxes from SimProp and CRPropa", JCAP 05, 006 (2019) doi:10.1088/1475-7516/2019/05/006 [arXiv:1901.01244 [astro-ph.HE]].[42] A. Aab et al. [Pierre Auger], "The Pierre Auger Cosmic Ray Observatory", Nucl. Instrum. Meth. A 798, 172-213 (2015) doi:10.1016/j.nima.2015.06.058 [arXiv:1502.01323 [astro-ph.IM]].[43] E. W. Mayotte et al.

[Pierre Auger], "Measurement of the mass composition of ultra-high-energy cosmic rays at the Pierre Auger Observatory," PoS ICRC2023 (2023), 365 doi:10.22323/1.444.0365[44] M. G. Aartsen et al. [IceCube], "The IceCube Neutrino Observatory - Contributions to ICRC 2017 Part II: Properties of the Atmospheric and Astrophysical Neutrino Flux", [arXiv:1710.01191 [astro-ph.HE]].[45] A. Aab et al.

[Pierre Auger], "Probing the origin of ultra-high-energy cosmic rays with neutrinos in the EeV energy range using the Pierre Auger Observatory", JCAP 10, 022 (2019) doi:10.1088/1475-7516/2019/10/022 [arXiv:1906.07422 [astro-ph.HE]].[46] G. J. Feldman and B. D. Cousins, "A Unified approach to the classical statistical analysis of small signals," Phys. Rev. D 57 (1998), 3873-3889 doi:10.1103/PhysRevD.57.3873 [arXiv:physics/9711021 [physics.data-an]].[47] D. Biehl, D. Boncioli, A. Fedynitch and W. Winter, "Cosmic-Ray and Neutrino Emission from Gamma-Ray Bursts with a Nuclear Cascade," Astron.

Astrophys. 611 (2018), A101 doi:10.1051/0004-6361/201731337 [arXiv:1705.08909 [astro-ph.HE]].[48] D. Boncioli, D. Biehl and W. Winter, "On the common origin of cosmic rays across the ankle and diffuse neutrinos at the highest energies from low-luminosity Gamma-Ray Bursts," Astrophys. J. 872 (2019) no.1, 110 doi:10.3847/1538-4357/aafda7 [arXiv:1808.07481 [astro-ph.HE]].[49] M. S. Muzio, M. Unger and G. R. Farrar, "Progress towards characterizing ultrahigh energy cosmic ray sources," Phys. Rev. D 100 (2019) no.10, 103008 doi:10.1103/PhysRevD.100.103008 [arXiv:1906.06233 [astro-ph.HE]].[50] M.

S. Muzio, G. R. Farrar and M. Unger, "Probing the environments surrounding ultrahigh

energy cosmic ray accelerators and their implications for astrophysical neutrinos," Phys. Rev. D 105 (2022) no.2, 023022 doi:10.1103/PhysRevD.105.023022 [arXiv:2108.05512 [astro-ph.HE]].

[51] B. D. Metzger, "Kilonovae", Living Rev. Rel. 23 (2020) no.1, 1 doi:10.1007/s41114-019-0024-0 [arXiv:1910.01617 [astro-ph.HE]].  [52] M. Shibata and K. Hotokezaka, "Merger and Mass Ejection of Neutron-Star Binaries", Ann. Rev. Nucl. Part. Sci. 69 (2019), 41-64 doi:10.1146/annurev-nucl-101918-023625 [arXiv:1908.02350 [astro-ph.HE]].  [53] S. Rosswog, "Fallback accretion in the aftermath of a compact binary merger", Mon. Not. Roy. Astron. Soc. 376 (2007), L48-L51 doi:10.1111/j.1745-3933.2007.00284.x [arXiv:astro-ph/0611440 [astro-ph]].  [54] R. Abbott et al.

[KAGRA, VIRGO and LIGO Scientific], "GWTC-3: Compact Binary Coalescences Observed by LIGO and Virgo during the Second Part of the Third Observing Run" Phys. Rev. X 13 (2023) no.4, 041039 doi:10.1103/PhysRevX.13.041039 [arXiv:2111.03606 [gr-qc]].  [55] M. Mukhopadhyay, S. S. Kimura and K. Murase, "Gravitational wave triggered searches for high-energy neutrinos from binary neutron star mergers: Prospects for next generation detectors", Phys. Rev. D 109 (2024) no.4, 043053 doi:10.1103/PhysRevD.109.043053 [arXiv:2310.16875 [astro-ph.HE]].  [56] M. Mukhopadhyay, K. Kotera, S. Wissel, K.

Murase and S. S. Kimura, "Ultrahigh-energy neutrino searches using next-generation gravitational wave detectors at radio neutrino detectors: GRAND, IceCube-Gen2 Radio, and RNO-G", Phys. Rev. D 110 (2024) no.6, 063004 doi:10.1103/PhysRevD.110.063004 [arXiv:2406.19440 [astro-ph.HE]].  [57] M. Mukhopadhyay, S. S. Kimura and B. D. Metzger, "High-energy neutrino signatures from pulsar remnants of binary neutron-star mergers: coincident detection prospects with gravitational waves", [arXiv:2407.04767 [astro-ph.HE]].  [58] D. Boncioli, S. Rossoni and C.

Trimarelli, "Ultra-high-energy cosmic rays: propagation and detection", PoS CORFU2021 (2022), 320 doi:10.22323/1.406.0320  [59] D. Boncioli, "Cosmic-ray propagation in extragalactic space and secondary messengers," Proc. Int. Sch. Phys. Fermi 208 (2024), 315-351 doi:10.3254/ENFI240009 [arXiv:2309.12743 [astro-ph.HE]].  [60] A. Aab et al. [Pierre Auger], "The Pierre Auger Observatory: Contributions to the 36th International Cosmic Ray Conference (ICRC 2019): Madison, Wisconsin, USA, July 24-August 1, 2019", [arXiv:1909.09073 [astro-ph.HE]].