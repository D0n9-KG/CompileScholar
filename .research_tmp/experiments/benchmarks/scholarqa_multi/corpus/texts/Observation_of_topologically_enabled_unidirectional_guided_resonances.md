# Research Article

Zixuan Zhang, Xuefan Yin, Zihao Chen, Feifan Wang, Weiwei Hu and Chao Peng*

# Observation of intensity flattened phase shifting enabled by unidirectional guided resonance

https://doi.org/10.1515/nanoph-2021-0393

Received August 25, 2021; accepted October 19, 2021; published online November 10, 2021

Abstract: Phase-only light modulation is an important functionality for many optoelectronic applications. Although modulation efficiency can be significantly improved by using optical resonances, resonance detuning is always accompanied with dramatic intensity variation that is less ideal. Here, we propose a method to achieve intensity-flattened phase shifting by utilizing the unidirectional guided resonance (UGR) - a novel class of topologically enabled guided resonance that only radiates toward a single side.

Consequently, the incident excites resonances and generates phase shifting, but it transmits to only one out-going port without other choice, which flattens the transmittance. Theory and simulation agree well and confirm our findings, in particular when nonradiative loss has been taken into account. By directly measuring the intensity and phase responses of UGR samples, a dip depth of 0.43 is observed with nonradiative $Q$ around 2500.

We further predict a dip depth of 0.13 can be achieved with a reasonable nonradiative $Q$ around 8000 in state-of-art fabrication precision, which is sufficient and useful for the applications ranging from light projection, flat metalens optics, optical phased array, to light detection and ranging.

Keywords: intensity flattened phase shifting; temporal coupled-mode theory; unidirectional guided resonance.

*Corresponding author: Chao Peng, State Key Laboratory of Advanced Optical Communication Systems and Networks, Department of Electronics & Frontiers Science Center for Nano-optoelectronics, Peking University, Beijing, China; and Peng Cheng Laboratory, Shenzhen, China, E-mail: pengchao@pku.edu.cn. https://orcid.org/0000-0002-0200-0798

Zixuan Zhang, Zihao Chen, Feifan Wang and Weiwei Hu, State Key Laboratory of Advanced Optical Communication Systems and Networks, Department of Electronics & Frontiers Science Center for Nano-optoelectronics, Peking University, Beijing, China

Xuefan Yin, Department of Electronic Science and Engineering, Kyoto University, Kyoto, Japan

# 1 Introduction

Phase shifting is an important functionality for a variety of optoelectronic applications. Examples range from phase modulators for optical communication [1-3], interferometry [4, 5], wave-front manipulation of metasurfaces [6-8] and beam steering of optical phased arrays [9-11]. Since phase accumulation relies on both the change of refractive index and structural geometry, various phase shifters are built from thermal-optically [12-14] or electro-optically [15-17] modifying refractive index, or alternatively, changing optical path in static [6, 18] or dynamic manners [10, 19].

However, achieving phase shifting in chip scale suffers from the insufficient efficiency in modifying refractive index, and limited footprint to change the light path. To overcome such shortcomings, optical resonances are usually applied [5, 20-24].

Optical resonances capture, trap the incident light, and release it in a period of photon lifetime. During this process, the phase of light dramatically changes according to the quality factor $Q$ of resonances in a modal volume $V$ [25]. By promoting the $Q$ and shrinking the $V$ [26], phase shifters are expected to be more efficient and compact. However, while resonance brings strong phase shifting, it is also accompanied with dramatic modification in intensity response at the same time [27], which makes it less ideal for many application scenarios. Therefore, it is theoretically interesting and practically important to find out whether it is possible to realize phase shifting alone with flattened intensity varying, when optical resonances are introduced.

Here, we theoretically propose that nearly perfect intensity-invariant phase shifting can be achieved by utilizing a novel class of topologically enabled unidirectional guided resonances (UGRs) [28] in a photonic crystal (PhC) slab, and experimentally observe in realistic samples that the intensity varying is indeed dramatically flattened during the phase shifting. UGRs are unique since they only radiate toward a single side of PhC slab while completely forbid the radiation on the other side. This phenomenon origins from the consequence of topological charge

DE GRUYTER

Nanophotonics 2021; 10(18): 4467-4475

a

Open Access. © 2021 Zixuan Zhang et al., published by De Gruyter. This work is licensed under the Creative Commons Attribution 4.0 International License.

evolution - the winding number of topological defects [29] in polarization vector field characterizing the radiation [30-35]. Specifically, integer topological charges correspond to a complete elimination of light escaping, known as the bound states in the continuum (BIC) [36, 37]. By breaking structural symmetries, an integer topological charge splits into two half-integer charges, carried by two circular polarized (CP) states in opposite helicities [38]. Through tuning structure parameters, the half-charges combine and restore as an integer charge at a single-side of PhC but remain separated at the other side, consequently, creating UGRs. Theory and experiment have confirmed the existence and effectiveness of UGRs [28].

The principle of the intensity flattened phase shifting relies on the UGRs for closing extra radiation channels. When an incident light goes through the PhC slab, part of the light transmit and reflect directly, while the other part couples into the resonance and radiates out. Unlike conventional guided resonances that radiate toward both upper and lower sides, the UGRs only radiate to one single side, which implies that all the light coupled into the UGR would eventually return to the same channel, resulting in phase shifting with unchanged intensity.

This feature separates the intensity and phase responses of a single resonance, which would be particularly useful for manipulating wave-fronts in a flexible way to realize flat metalens, programmable metasurfaces, and large-scale optical phase arrays.

# 2 Principle and theory

We start by considering an arbitrary guided resonance embedded in a uniform medium whose in-plane momentum is specified as $k$ . In general, the interaction between incident light and resonance can be depicted by a 4-port model of temporal coupled mode theory (TCMT) [39], as illustrated in Figure 1a.

Assuming a plane-wave incident with in-plane momentum $k$ inputs from Port 3 (left-plane), a part of the incident doesn't interact with the resonance but directly passes through to Port 2 or reflects to Port 4 (mid-plane), while the other part excites the resonance under phase-matching condition, which radiates toward Port 2 and 4, respectively (right-plane). Therefore, observed from Port 2 and 4, the transmittance and reflectance depend on how much incident energy coupled to the resonance, which can be quite different under on- or off-resonance excitation.

Such a process can be modeled by using TCMT [40]. Here, we assume weak coupling, linearity, energy conservation, and time-reversal symmetry in the system. Although the incident wave only couples to resonance and out-going waves with the same $k$ (conservation of Bloch momentum), to describe time-reversal symmetry constraints for general geometries and incident angles, we need to include the resonance at $-k$ as well, resulting in a two-resonance, four-port model:

![](dt=2026-06-02/ht=06/527518a7e58df676d533984e446d2bc5e0eb33d965a4bdcf7af8a2755a782106.jpg)

4468

Z. Zhang et al.: Observation of intensity flattened phase shifting

DE GRUYTER

$$
\frac {\mathrm {d} A}{\mathrm {d} t} = \left(j \omega_ {
0} - \frac {1}{\tau} - \frac {1}{\tau_ {n r}}\right) A + K ^ {\mathrm {T}} S _ {+}, \quad S _ {-} = C S _ {+} + D A \tag {1}
$$

in which:

$$
S _ {+} = \left( \begin{array}{c} S _ {1 +} \\ S _ {2 +} \\ S _ {3 +} \\ S _ {4 +} \end{array} \right), \quad S _ {-} = \left( \begin{array}{c} S _ {1 -} \\ S _ {2 -} \\ S _ {3 -} \\ S _ {4 -} \end{array} \right), \quad A = \left( \begin{array}{c} A _ {1} \\ A _ {2} \end{array} \right) \qquad (2)
$$

$$
K = \left( \begin{array}{c c c c} \kappa_ {1} & 0 & \kappa_ {3} & 0 \\ 0 & \kappa_ {2} & 0 & \kappa_ {4} \end{array} \right), \quad D = \left( \begin{array}{c c} 0 & d _ {1} \\ d _ {2} & 0 \\ 0 & d _ {3} \\ d _ {4} & 0 \end{array} \right), \tag {3}
$$

$$
C = \mathrm {e} ^ {j \phi} \left( \begin{array}{c c c c} 0 & r & 0 & j t \\ r & 0 & j t & 0 \\ 0 & j t & 0 & r \\ j t & 0 & r & 0 \end{array} \right)
$$

Here, $A_{1,2}$ are the amplitudes of resonances at $k$ and $-k$ , but their center frequencies are necessarily identical due to the law of reciprocity, then we denote it as $\omega_0$ ; $\tau$ and $\tau_{\mathrm{nr}}$ represent the radiative and nonradiative lifetime of resonance, respectively. $S_{\pm}$ are the amplitudes of the input and out-going waves; $K$ and $D$ are the coupling matrices that depict the strength of coupling into and out of the resonance, using subscripts $(i = 1,2,3,4)$ to denote the 4 ports as shown in Figure 1a; $C$ is the scattering matrix for the direct (nonresonant) transmission and reflection through the resonance.

Since time-reversal flips the two resonances at $k_{\pm}$ , we have $d_{i} = \kappa_{i}$ . By imposing the constraint of energy conservation, we got $|d_2|^2 +|d_4|^2 = |d_1|^2 +|d_3|^2 = \frac{2}{\tau}$ . The direct scattering matrix $C$ is caused by the structure itself and have nothing to do with the resonance, because $r$ and $t$ characterize the background reflectivity and transmissivity, and follow $r^2 +t^2 = 1$ . The initial phase $\phi$ depends on the position of the reference plane, which we set to 0 for simplicity. According to the time-reversal symmetry, we readily have $CD^{*} = -D$ . Solving the coupling equation Eq. (1) with above mentioned constraints, we obtain the reflectance and transmittance for an input from Port 3, as:

$$
R = \left| S _ {3 4} \right| ^ {2} = \left| \mathrm {e} ^ {j \phi} r + \frac {d _ {3} d _ {4}}{j \left(w - w _ {0}\right) + \frac {1}{\tau} + \frac {1}{\tau_ {\mathrm {n r}}}} \right| ^ {2} \tag {4}
$$

$$
T = \left| S _ {3 2} \right| ^ {2} = \left| \mathrm {e} ^ {j \phi} j t + \frac {d _ {2} d _ {3}}{j \left(w - w _ {0}\right) + \frac {1}{\tau} + \frac {1}{\tau_ {\mathrm {n r}}}} \right| ^ {2} \tag {5}
$$

Furthermore, we assume the guided resonance is a UGR at in-plane momentum of $k$ , namely the resonance

only radiates toward Port 2 while Port 4 is closed. A realistic design is presented in Figure 1b and c, as we elaborated in our previous work [28]. Briefly, the structure consists of a series of equally spaced one-dimensional bars in a period of $825\mathrm{nm}$ , fabricated on a silicon-on-insulator (SOI) wafer with a silicon layer of $500\mathrm{nm}$ and silica box layer of $2\mu \mathrm{m}$ . The silicon bar is trapezoidal in general, showing that the structure possesses no symmetry in $180^{\circ}$ rotation $(C_2)$ or up-down mirror $(\sigma_z)$ .

The upper surface of the trapezoid is $473\mathrm{nm}$ and the two side-walls are in angles of $75^{\circ}$ and $79^{\circ}$ , respectively. The modal pattern of $E_{y}$ field at resonance wavelength of $1551\mathrm{nm}$ is illustrated in Figure 1c, confirming the UGR is singled-sided radiative. The quality factor $Q_{\mathrm{r}}$ of the UGR is 277.

At UGRs, we have $d_4 = 0$ . Together with the relations of $|d_2|^2 + |d_4|^2 = |d_1|^2 + |d_3|^2 = \frac{2}{\tau}$ and $CD^* = -D$ , we obtain that:

$$
\left| d _ {2} \right| ^ {2} = \frac {2}{\tau}, \quad j t d _ {2} ^ {*} = - d _ {3} \tag {6}
$$

Therefore, $S_{34}$ and $S_{32}$ can also be simplified as:

$$
S _ {3 4} = r \tag {7}
$$

$$
S _ {3 2} = - \frac {\left(\omega - \omega_ {0}\right) + j _ {\tau} ^ {\frac {1}{\tau}}}{\left(\omega - \omega_ {0}\right) - j _ {\tau} ^ {\frac {1}{\tau}}} t \tag {8}
$$

As a result, under the limit that nonradiative loss is negligible, i.e., $\tau_{\mathrm{nr}}\rightarrow \infty$ , we derive that the reflectance and transmittance become:

$$
R = | r | ^ {2} \tag {9}
$$

$$
T = | t | ^ {2} \tag {10}
$$

$$
\operatorname {A r g} \left(S _ {3 2}\right) = \arctan \left[ \frac {\frac {2}{\tau} \left(\omega - \omega_ {0}\right)}{\left(\omega - \omega_ {0}\right) ^ {2} - \frac {1}{\tau^ {2}}} \right] \tag {11}
$$

From Eqs. (9) and (10), we find that the reflectance and transmittance upon UGR are pinned to the background reflection and transmission, and don't exhibit any features of the resonance. Namely from the intensity measurement, the resonance becomes invisible, which is quite different from conventional observation of guided resonances. We confirm this finding from numerical simulations by using the commercial finite-element simulation software COMSOL Multiphysics.

As presented in Figure 2a, the simulation agrees well with the theory and consistently indicates that, the intensity response is indeed invariant from on- to off-resonance. Noteworthy that, the Fabry-Perot background model that was assumed in previously reported results [39, 40] is not suitable for our case, see Supplementary Section 1 for more details and discussions.

DE GRUYTER

Z. Zhang et al.: Observation of intensity flattened phase shifting

4469

![](dt=2026-06-02/ht=06/72e936822268e8a5b0eeb0a16fe6537600a91b4bc17942c4d7d141feb4d0b631.jpg)

![](dt=2026-06-02/ht=06/ba67073eb70d35512d3ae83095422e23ea6c9c88b24e4edef1a88fc0f732960b.jpg)

![](dt=2026-06-02/ht=06/fd35a7bdbe421b33a6584bb8562b556866bd3d9c32627a8fa0c7c94c896bbdd0.jpg)

![](dt=2026-06-02/ht=06/dd0db06a2efd9633455017dbe70ff84d245ce42af241bb0ff9ccb72a128975df.jpg)

![](dt=2026-06-02/ht=06/6d80855fd584f488ea90a167a1b7b46cf8ba26fb0f86215b6b6aeee99ac4381e.jpg)

Nevertheless, the incident from Port 3 truly excites the resonance. From Eq. (8), we find that the phase upon transmission observed from Port 2 experiences a shifting of $2\pi$ when scanning through the UGR resonance in a range of $20\mathrm{nm}$ , supported by the theory and simulations presented in Figure 2b, which is exactly the behavior of intensity flattened phase shifting we desired.

In realistic samples, the lifetime of resonances also depends on those nonradiative processes such as material absorption, out-of-plane scatterings, and lateral leakage. For our UGR design presented in Figure 1b, although material absorption is negligible since silicon is transparent in telecommunication wavelength and lateral leakage is omitted since infinite periodicity is assumed, scattering loss caused by fabrication imperfection (hole fluctuations and non-uniformity, surface roughness, etc.) are still inevitable in practices.

Different from the radiations that emit toward specific directions, nonradiative energy dissipation is not directional and cannot be directly observed from the outgoing ports. When taking such nonradiative losses into account, we have the reflectance and transmittance written as:

$$
R = | r | ^ {2} \tag {12}
$$

$$
T = \left| \frac {\left(\omega - \omega_ {0}\right) + j \left(\frac {1}{\tau} - \frac {1}{\tau_ {\mathrm {n r}}}\right)}{\left(\omega - \omega_ {0}\right) - j \left(\frac {1}{\tau} + \frac {1}{\tau_ {\mathrm {n r}}}\right)} t \right| ^ {2} \tag {13}
$$

Noticed from Eq. (12) tha
t the reflectance $R$ still pins to the background reflectivity $|r|^2$ , namely no resonance feature appears on it. This is not surprising because the UGR eliminates any downward radiation to Port 4. However, the transmittance $T$ observed from Port 2 would be modified by the nonradiative lifetime $\tau_{\mathrm{nr}}$ according to Eq. (13). As a typical example shown in Figure 2c and d, the phase shifting behavior preserves under nonradiative losses, but the transmittance is no longer the same as the background $|t|^2$ and exhibits a feature of "dip".

To capture the nonradiative loss in numerical simulation, we apply an imaginary part $\alpha$ to the permittivity of silicon as $\varepsilon_{\mathrm{si}} = 12.11 + j\alpha$ . As illustrated in Figure 2c and d, the simulation result agrees well with the theory by assuming $\alpha = 0.0046$ for a nonradiative $Q_{\mathrm{nr}} = \tau_{\mathrm{nr}}\omega_0 / 2 = 3010$ for an example.

Phase shifting is expected to be perfect intensity invariant in the limit of $\tau_{\mathrm{nr}} \to \infty$ . Under the absence of nonradiative loss, we use $T_{\mathrm{bg}}$ to denote the transmittance at the center frequency of UGR. When nonradiative loss is applied, we propose a figure-of-merit called "normalized

4470

Z. Zhang et al.: Observation of intensity flattened phase shifting

DE GRUYTER

dip depth" to characterize the flatness of transmittance. Specifically, we denote the transmittance at the dip as $T_{\mathrm{dip}}$ , and define the normalized dip depth as $(T_{\mathrm{bg}} - T_{\mathrm{dip}}) / T_{\mathrm{bg}}$ . We further calculate the depths under a series of $Q_{\mathrm{nr}}$ from 10 to $10^{5}$ as presented in Figure 2e. The maximum depth appears at the critical-coupling condition of $Q_{\mathrm{r}} = Q_{\mathrm{nr}}$ [41], which implies that the transmittance reaches $T_{\mathrm{dip}} = 0$ , i.e., no light can be observed from Port 2.

Nevertheless, for $Q_{\mathrm{nr}} > Q_{\mathrm{r}}$ , the flatness of transmittance improves with increasing $Q_{\mathrm{nr}}$ , and eventually approaches to 0 for $\tau_{\mathrm{nr}} \rightarrow \infty$ . In particular, we find that a normalized dip depth of about 0.13 can be achieved for a nonradiative $Q_{\mathrm{nr}}$ around 8000. As illustrated in Figure 2c, depth of about 0.13 has been quite flat for practical usage.

Given that a $Q_{\mathrm{nr}}$ around 8000 is not quite challenging for state-of-art fabrication precision, we conclude that the proposed method of intensity flattened phase shifting is feasible.

Noteworthy that, different from those plasmonic/ dielectric metasurfaces platforms working at off-resonance region [42], the intensity-flatten behaviors presented in this work are direct consequences of the directionally emission characteristics of UGR as an eigen state in PhC slab. Besides, since UGRs are topologically robust in any 2D parameter space [28], the geometries of UGR design can be freely scaled and continuously tuned,

which is promising for the applications of different purposes.

# 3 Experiments and results

To verify our theoretical findings, we measure the intensity and phase responses of the UGR by applying an incident input from Port 3 and observe from Port 2 and 4, as schematically shown in Figure 1a. The samples used here are the same as our previous work [28], in which the characteristics of UGRs had been investigated in great details. Briefly, the UGRs are realized in an SOI wafer that consists of a layered structure of silicon/silica/silicon stacks, with the thickness of $500\mathrm{nm}$ , $2\mu \mathrm{m}$ , and $725\mu \mathrm{m}$ , respectively.

The samples are fabricated using silica hard mask, followed by reactive ion etching with wedge-shaped holder to create trapezoidal air-holes, and chemical-mechanical polished to mirror finish the bottom facet. The detailed structural parameters are shown in Figure 1b. The SEM images of our sample are presented in Figure 3b. For more details of the UGR samples, please refer to our previous work [28].

As schematically shown in Figure 3b, the measurement system is capable of observing transmission and reflection simultaneously, which is similar to our previous

![](dt=2026-06-02/ht=06/5cbf9d5daaf08272b9c78791c1d4682bc23b2845579c9b5807fd7086ed636126.jpg)

![](dt=2026-06-02/ht=06/1c43e9c25a191f7e2ac04e6b5abb4c01c017ec1702e1c3c47cb427597d6d482b.jpg)

![](dt=2026-06-02/ht=06/0cbe61d03099c032981aec9ee8bdbfd628ef8634760b06c91bdda1df0c87f742.jpg)

(a) Schematic of the experimental setup. The polarizers (dashed-box) are inserted to block direct reflected and transmitted light for characterizing the resonance itself by using cross-polarization technique [28]; while they are removed for reflectance and transmittance measurement. The reference light (dashed line) and signal light (solid line) form Michelson interferometer configurations. L, lens; Obj, objective; PD, photodetector; CCD, charge-coupled device; POL, polarizer; BS, beam splitter; $4f$ , relay $4f$ optical system. (b) Scanning electron microscope (SEM) image of the fabricated UGR sample from a side views. (c) Measured radiation spectrum of UGR (blue cross) and the fitting curve (red line), $Q_{\mathrm{tot}} = 250$ is obtained.

DE GRUYTER

Z. Zhang et al.: Observation of intensity flattened phase shifting

4471

work [28]. A tunable telecommunication laser in the $\mathrm{C} + \mathrm{L}$ band is first sent through a polarizer (POL) before it is focused by a lens (L1) onto the rear focal plane (RFP) of an infinity-corrected objective lens (Obj). The incident angle is tuned by moving L1 in the $x - y$ plane, to excite the UGR at the given in-plane momentum $k$ . The transmitted and reflected lights are collected by two identical objective lenses, followed by a $4f$ system that adjusts the magnification ratio to best fit the CCD camera and photodetector (PD). When characterizing the eigenstate of the

resonance, additional orthogonal-aligned polarizers are inserted into the transmission arm and the reflection arm to suppress the direct light but leave the resonance radiation passing through. The insertion losses of individual optical components are calibrated carefully in order to normalize the measured intensity. By scanning the wavelength, both reflectance and transmittance spectra are recorded.

In order to observe the phase shifting from the UGR, we adopt Michelson interferometer configuration on transmission and reflection arms, respectively. As shown in

![](dt=2026-06-02/ht=06/fc9b0a83e39959975986ee21f6518df50a2f18831f2ef1b893b4c7391b24d9d3.jpg)

![](dt=2026-06-02/ht=06/c01b8da9a32e6c295754a32cf707edf63008d160a6ea46c8fd22a41ca2f21fd8.jpg)

4472

Z. Zhang et al.: Observation of intensity flattened phase shifting

DE GRUYTER

Figure 3a, the incident is divided by a beam splitter (BS) into signal light (solid line), and reference light (dash line). The signal light goes the same path as in the intensity measurement, while the reference light is mirror reflected to arrive at the CCD camera and generates interference patterns on it. By fine-tuning the positions and orientations of the mirrors, the optical path difference between the signal and reference light is aligned to zero, which creates concentric interference fringes of equal inclination. Similar to standard Michelson interferometer, phase shifting is observed from accounting the fringe shifting. To precisely extract the phase shift from interference fringes, we adopt a spatial correlations method and please see Supplementary Section 2 for the details.

The UGR is excited by o
n-resonance pumping technique, accordingly, Fano line-shape is observed as shown in Figure 3c. The total quality factor $Q_{\mathrm{tot}} = 250$ is extracted by numerically fitting the radiation spectra, which is close but lower than the numerical simulation of $Q_{\mathrm{r}} = 277$ , because the surface roughness and fabrication disorder raise nonradiative scatterings losses. According to the relationship $1 / Q_{\mathrm{tot}} = 1 / Q_{\mathrm{nr}} + 1 / Q_{\mathrm{r}}$ . The nonradiative quality factor is estimated as $Q_{\mathrm{nr}} = 2500$ .

Next, we measure the reflectance and transmittance and present the results in Figure 4a. As expected, the reflectance is smooth and exhibits no feature of resonance, while a "dip" is observed from transmittance, and its normalized dip depth is read as about 0.43. The observed phenomena agree well with the theoretical predication of Figure 2c. For quantitative comparison, we apply $Q_{\mathrm{nr}} = 2000$ into the TCMT model and found out the experimental result is consistent with theory as shown in Figure 4a, proving the effectiveness of flattened intensity response when the UGR is applied.

Further, we directly observe the phase shifting upon transmitted and reflected lights. At a specified wavelength, concentric interference fringes are recorded by the CCD camera, and the phase shift is extracted by using the spatial correlations method. As illustrated in Figure 4b, the phase of reflected light barely changes, while the phase of transmitted experiences a shifting of $2\pi$ when the wavelength scans from 1540 to $1556~\mathrm{nm}$ . The theory and simulation match with each other under the assumption of $Q_{\mathrm{nr}} = 2500$ (lines vs. markers).

Moreover, the phase shifting process is evident from interference patterns. We pick three individual wavelengths, and plot the fringe patterns in Figure 4c, which are marked as $W$ , $X$ , $Y$ for the transmission and $W^{\prime}$ , $X^{\prime}$ , $Y^{\prime}$ for the reflection, respectively. Observed from the transmission, the central part of the interference pattern evolves from bright to dark, and then back to bright

through wavelength scanning, which is distinct feature of phase shifting. During the same process, the interference patterns of reflection are almost steady and remain as dark. For more details about the fringe evaluation, please see the Supplementary Video.

# 4 Conclusions

To summarize, we propose and demonstrate a new method to realize phase-only transmission in principle and observe the phase shifting with flattened intensity variation in experiments. The method utilizes the unique feature of topologically enabled UGRs, which only radiates toward a single side of the PhC slab but completely forbids the radiation toward the other side. Different from conventional guided resonances, the UGRs close one port for light going into and out while leaving the other as an opened channel. Consequently, although the incident excites the resonance and generates phase shifting, it transmits to only one out-going port without other choices, and therefore, the transmittance becomes invariant with respect to the detuning of resonance.

We present comprehensive investigation by using the TCMT theory and FEM simulation. The theory and simulation agree well and validate the proposed method. Furthermore, we take the nonradiative loss into account since it is somehow inevitable in realistic samples, and has observed an intensity flattening by a dip depth of 0.43 for a nonradiative $Q_{\mathrm{nr}}$ around 2500. We estimate that a $Q_{\mathrm{nr}}$ around 8000 is sufficient to obtain a dip depth of 0.13, which is indeed fairly flat for many applications, and such level of precision is feasible for state-of-art fabrication.

The proposed method paves the way to further improving the performance for a variety of optoelectronic applications, ranging from three-dimensional video projection, flat metalens optics, to optical phased arrays and light detection and ranging.

Acknowledgments: We thank Bo Zhen, Jicheng Jin and Hengyun Zhou for the discussions, and High-performance Computing Platform of Peking University for the support of numerical simulations.

Author contribution: All the authors have accepted responsibility for the entire content of this submitted manuscript and approved submission.

Research funding: This work was supported by National Key Research and Development Program of China (Grant No. 2020YFB1806405, 2018YFB2201704), the National Natural Science Foundation of China (Grant No. 61922004, 62135001), China Postdoctoral Science Foundation funded

DE GRUYTER

Z. Zhang et al.: Observation of intensity flattened phase shifting

4473

project (Grant No. 2021M690239), Major Key Project of PCL (Grant No. PCL2021A14), and the Open Fund of the State Key Laboratory of Integrated Optoelectronics.

Conflict of interest statement: The authors declare no conflicts of interest regarding this article.

# References

4474

Z. Zhang et al.: Observation of intensity flattened phase shifting

DE GRUYTER

Supplementary Material: The online version of this article offers supplementary Material (https://doi.org/10.1515/nanoph-2021-0393).

DE GRUYTER

Z. Zhang et al.: Observation of intensity flattened phase shifting

4475