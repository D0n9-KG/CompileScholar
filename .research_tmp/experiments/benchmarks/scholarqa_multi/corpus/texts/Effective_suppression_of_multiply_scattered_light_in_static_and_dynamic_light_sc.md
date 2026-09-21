# A new instrument for time-resolved static and dynamic light-scattering experiments in turbid media

Christian Moitzi *, Ronny Vavrin $^{1}$ , Suresh Kumar Bhat $^{2}$ , Anna Stradner, Peter Schurtenberger

Adolphe Merkle Institute, University of Fribourg, Rte. de l'Ancienne Papeterie, P.O. Box 209, 1723 Marly 1, Switzerland

# ARTICLE INFO

Article history:

Received 19 December 2008

Accepted 8 April 2009

Available online 21 April 2009

Keywords:

Dynamic light scattering

Static light scattering

Casein micelles

Turbid suspensions

Sol-gel transition

# ABSTRACT

We present a new 3D cross-correlation instrument that not only allows for static and dynamic scattering experiments with turbid samples but measures at four angles simultaneously. It thus extends the application of cross-correlation light scattering to time-resolved studies where we can, for example, efficiently investigate the temporal evolution of aggregating or phase separating turbid dispersions. The combination of multiangle 3D and on-line transmission measurements is an essential prerequisite for such studies.

This not only provides time-resolved information about the overall size and shape of the particles through measurements of the mean apparent radius of gyration and hydrodynamic radius, but also on the weight-average apparent molar mass via the absolute forward scattering intensity. We present an efficient alignment strategy based on the novel design of the instrument and then the application range of the instrument using well-defined model latex suspensions.

The effectiveness of the cross-correlation multiangle technique to monitor aggregation processes in turbid suspensions is finally shown for the acidification of skim milk during the yoghurt-making process. Due to the self-assembled nature of the casein micelles an understanding of the sol-gel process induced by the acidification is only feasible if time-resolved light-scattering experiments on an absolute scale are possible under industrially relevant conditions, where the casein solutions are highly turbid.

© 2009 Elsevier Inc. All rights reserved.

# 1. Introduction

Static (SLS) [1,2] and dynamic light scattering (DLS) [3-5] are very popular techniques for the characterization of colloidal suspensions. They provide a wealth of structural and dynamic information on mesoscopic length scales. The characteristic length scale of a light-scattering experiment $l_{\mathrm{res}}$ is determined by the wavelength $\lambda$ of the light used to probe the system, where $l_{\mathrm{res}}$ is given by the relationship $l_{\mathrm{res}} \propto 2\pi / q$ .

Here $q$ denotes the scattering vector given by $q = (4\pi n / \lambda)\sin(\theta / 2)$ , $n$ is the index of refraction of the solvent, and $\theta$ the scattering angle. DLS yields information about the dynamics of the system over an extended range of time scales. While it is routinely used for particle sizing, the technique has seen important developments that currently allow us to probe a number of dynamic features of complex fluids and viscoelastic solids such as colloidal glasses and gels [6].

SLS on the other hand is not only routinely used for molar mass determination in polymer sciences, but gives access to structure and interactions in colloidal

suspensions via the measurement of the form factor $P(q)$ and the structure factor $S(q)$ . However, most of the current applications of SLS and DLS rely on experiments with samples that exhibit single scattering only. This is quite in contrast to the situation encountered when dealing with industrially relevant systems, where multiple scattering often cannot be neglected.

While there are dynamic light-scattering schemes such as diffusing wave spectroscopy (DWS) that use the multiply scattered light for the analysis (e.g., [7-9]), they in general do not provide $q$ -dependent information and thus do not allow us to measure the full static and dynamic structure factor. There exist DLS techniques which reduce the effect of multiply scattered light, for instance, by reducing the path length of the beam in the sample [10,11], or by measuring in backscattering geometry [12,13].

The major drawbacks of these approaches are that they are not able to remove multiply scattered light completely and they are limited in $q$ range. However, there also exist several DLS schemes to suppress contributions from multiple scattering and thus extend the application range of light-scattering experiments considerably [14,15]. They mostly rely on cross-correlation strategies, where the general idea is to perform two scattering experiments at the same time in the same scattering volume.

If the scattering vector of both experiments is the same, singly scattered photons give correlated signals in both detectors. Multiply scattered light results in uncorrelated fluctuations that contribute to the background only,

Journal of Colloid and Interface Science 336 (2009) 565-574

SELFIER

Contents lists available at ScienceDirect

Journal of Colloid and Interface Science

www.elsevier.com/locate/jcis

JOURNAL OF Colloid and Interface Science

* Corresponding author. Fax: +41 26 300 9747.

E-mail address: christian.moitzi@unifr.ch (C. Moitzi).

Present address: Laboratory for Neutron Scattering, ETH Zurich & Paul Scherrer Institut, CH-5232 Villigen PSI, Switzerland.

<sup>2</sup> Present address: Complex Fluids & Polymer Engineering Group, Polymer Science & Engineering Division, National Chemical Laboratory, Pune 411008, India.

0021-9797/$ - see front matter © 2009 Elsevier Inc. All rights reserved.  
doi:10.1016/j.jcis.2009.04.043

due to the fact that it has been scattered in a succession of different scattering vectors. Performing a cross-correlation between the signal of both detectors then allows a suppression of the contributions from multiply scattered light and to determine the single scattering intensity cross-correlation function or dynamic structure factor of turbid systems with the same information content that a hypothetical measurement under single scattering conditions would have provided.

These cross-correlation experiments are not only restricted to DLS experiments, but also allow us to extend SLS experiments to turbid systems as the scattered intensity can be corrected for all contributions from multiple scattering. One possible realization of the cross-correlation method is the two-color technique [12,14,16], which, however, poses extreme technical difficulties that have limited its application for routine measurements.

Another implementation of such a cross-correlation approach is the so-called 3D cross-correlation scheme [14,17,18] that has been developed during the last few years to an extent that it can now be routinely applied for the characterization of turbid suspensions. There even exist instruments which combine the 3D cross-correlation with the echo technique and thereby give access to investigation of highly turbid and nonergodic samples [19].

The fact that only the singly scattered light is used and that the multiply scattered light results in a reduction of the signal-to-base line ratio implies that there must still be a significant contribution of light which is scattered only once to the overall scattered intensity. This generally limits the use of the cross-correlation techniques to samples with a transmission of at least $1 - 2\%$ [17].

While we can extend the applicability of the cross-correlation techniques to extremely turbid samples by reducing the path length of the light in the sample to a minimum and thereby increasing the amount of singly scattered light, the significant reduction in the signal-to-base line ratio nevertheless affects the typical measurement time.

Given the fact that for particle size determination or measurements of molar mass and radius of gyration we normally have to do experiments at different scattering angles, this results in a very long experiment duration on classical goniometer
-based 3D cross-correlation instruments [17]. While this does not really pose serious problems on stable systems, it often drastically limits our ability to investigate turbid suspensions that exhibit a time dependence such as aggregating or phase separating suspensions.

Here we therefore describe the development of a novel multiantangle light-scattering instrument that implements the 3D cross-correlation scheme [14,17,20-22] and allows for time-resolved measurements of SLS and DLS in turbid suspensions. We first describe the layout of the instrument and its calibration, and then demonstrate its superior performance with a study of the aggregation and sol-gel transition in acidified skim milk, where the high

turbidity normally excludes the use of light scattering for a quantitative description of the aggregation kinetics.

# 2. Instrument layout

The new light-scattering instrument fully implements the 3D cross-correlation scheme (Scheme 1). It allows for time-resolved measurements at four angles simultaneously. Multiply scattered light is suppressed, so that one can still detect intensities and correlation functions which originate from singly scattered light only.

A schematic drawing of the instrument is shown in Scheme 2. A Coherent Compass 415M-200 diode-pumped solid-state laser (532 nm) is used as light source. A polarization preserving single mode fiber (Dantec DAN $60 \times 30$ ) is used in order to bring the illuminating beam to the instrument. The laser beam is split into two parallel beams, which are then focused onto the scattering cell by the lens $L_{0}$ (planar convex, focal length $150~\mathrm{mm}$ , diameter $50~\mathrm{mm}$ ) with the wave vectors $k_{\mathrm{ia}}$ and $k_{\mathrm{ib}}$ .

This lens has been chosen in order to allow for a sufficiently large $\delta$ required for an efficient suppression of contributions from multiple scattering. In our current setup we have chosen values of $R \leqslant 100~\mu \mathrm{m}$ for the beam waist at the scattering volume and $\delta = 11.4^{\circ}$ , which results in a suppression of double scattering by approximately a factor of $5.3 \times 10^{-3}$ for $\lambda_{0} = 532~\mathrm{nm}$ (the efficiency for suppressing higher order contributions is much higher; for details see Ref. [14]).

Here $\delta$ is the angle between the two incident beams (see Schemes 1 and 2). The scattering cell is immersed in an index match vat, where decahydronaphthalene (decaline) is used as the index matching fluid. The vat temperature is controlled via two heat exchangers that are connected to a thermostat.

Four identical lenses $(L_{1} - L_{4})$ are used for the detection side, and the scattered light with wave vectors $k_{\mathrm{fa}}$ and $k_{\mathrm{fb}}$ is collected and guided by a combination of a GRIN lens and a single mode fiber (OZ optics) at each monitored scattering angle $\theta$ . The end of each fiber is connected via standard SMA fiber connectors to a home-built housing for 8 photomultiplier tubes (PM; H3460-54 photon counting heads), and the PM signal is processed by two amplifier/discriminators and fed into an 8-channel digital correlator (Flex01/8ch from correlator.com).

The signals from each detector pair that covers the same scattering vector $q(q_{\mathrm{a}} = q_{\mathrm{b}})$ are then cross-correlated to get four intensity correlation functions. A description of the alignment procedure of the instrument can be found in Supporting Information.

To correct for temporal fluctuations in the primary laser intensity $I_{\mathrm{Laser}}$ a part of the laser is split of and detected separately by a photodiode. The corrected scattered intensity $I_{\mathrm{c}}(q,t)$ was calculated with

$$
I _ {\mathrm {c}} (q, t) = I _ {\text {r a w}} (q, t) \frac {\overline {{I _ {\text {L a s e r}}}}}{I _ {\text {L a s e r}} (t)} - I _ {\text {d a r k}}, \tag {1}
$$

![](dt=2026-05-11/ht=06/9c931978079a28bc26bd469e386faa7c675816731c9bec215bbbcba7f92148f3.jpg)

![](dt=2026-05-11/ht=06/fe843f24961021808f0f203b2a855385c119be47648d49f96caa20d4419bb9d4.jpg)

566

C. Moitzi et al./Journal of Colloid and Interface Science 336 (2009) 565-574

![](dt=2026-05-11/ht=06/2ae1dd80cd33571fdd0e47ad0a113f417ef4cff7e0967e4f518842a01dbe1e84.jpg)

where $I_{\mathrm{Laser}}$ is the average laser intensity, $I_{\mathrm{raw}}$ is the detected raw intensity, and $I_{\mathrm{dark}}$ is the dark current of the detector.

The changes in the size of the scattering volume were taken into account by multiplying the static intensities with the sine of the scattering angle.

$$
I _ {\mathrm {d}} (\theta) = I _ {\mathrm {c}} (\theta) \sin (\theta). \tag {2}
$$

To set the static scattering intensities on absolute scale the intensities were calibrated by a toluene measurement,

$$
I (q) = (\mathrm {d} \Sigma / \mathrm {d} \Omega) _ {\text {T o l u e n e}} \frac {n _ {\mathrm {H} _ {2} \mathrm {O}} ^ {2} I _ {\mathrm {d}} (q)}{n _ {\text {T o l u e n e}} ^ {2} I _ {\mathrm {d} , \text {T o l u e n e}} (q)}. \tag {3}
$$

In Eq. (3) $I(q)$ is the scattered intensity on absolute scale, $\mathrm{d}\Sigma/\mathrm{d}\Omega$ is the Rayleigh ratio in $\mathrm{cm}^{-1}$ , and $n$ is the refractive index. Because of different sizes of the scattering volume at different scattering angles, different sensitivities of the detectors, and imperfectness of the alignment $I_{\mathrm{Toluene}}$ takes different values for the different detectors.

The ratio of the intercepts of the correlation functions of the sample $\beta_{\mathrm{ab}}$ and a singly scattering sample $\beta_{\mathrm{ab}}^{(1)}$ gives the fraction of singly scattered light within the whole signal at this particular angle. The intensities given by the two detectors are geometrically averaged. The geometrical average must be chosen because the ratio of singly scattered light and the overall intensity is determined from the intercept of the intensity cross-correlation function $g_{\mathrm{ab}}^{(2)}(q,\tau)$ which contains the product of the intensities $I_{\mathrm{a}}$ and $I_{\mathrm{b}}$ .

$$
\mathbf {g} _ {\mathrm {a b}} ^ {(2)} (\tau) = \frac {\left\langle I _ {\mathrm {a}} (t) I _ {\mathrm {b}} (t + \tau) \right\rangle}{\left\langle I _ {\mathrm {a}} \right\rangle \left\langle I _ {\mathrm {b}} \right\rangle} = \frac {\left\langle I _ {\mathrm {a}} \right\rangle \left\langle I _ {\mathrm {b}} \right\rangle + \beta_ {\mathrm {a b}} ^ {(1)} I _ {\mathrm {a}} ^ {(1)} I _ {\mathrm {b}} ^ {(1)} | S (\tau) | ^ {2}}{\left\langle I _ {\mathrm {a}} \right\rangle \left\langle I _ {\mathrm {b}} \right\rangle}. \tag {4}
$$

Here $S(\tau)$ is the dynamic structure factor. Consequently, the singly scattered intensity $I^{(1)}$ as a function of the scattering vector $q$ can be calculated from

$$
I ^ {(1)} (q) = \sqrt {I _ {\mathrm {a}} ^ {(1)} (q) I _ {\mathrm {b}} ^ {(1)} (q)} = \sqrt {\frac {\beta_ {\mathrm {a b}} (q)}{\beta_ {\mathrm {a b}} ^ {(1)} (q)} I _ {\mathrm {a}} (q) I _ {\mathrm {b}} (q)}, \tag {5}
$$

where $I_{\mathrm{a}}$ and $I_{\mathrm{b}}$ are the intensities of the two detectors on absolute scale and $I^{(1)}$ are the corresponding single scattering intensities [20].

The transmission was used to correct for the loss of singly scattered intensity in turbid samples. The singly scattered intensity $I_{T=1}^{(1)}$ without loss due to turbidity (which would in principle be measured in a cell of infinitely small diameter) can be calculated by

$$
I _ {T = 1} ^ {(1)} = I ^ {(1)} \frac {1}{T}, \tag {6}
$$

where $T$ is the transmission of the sample in a cell of diameter $d$ . The transmission was measured directly by dividing the intensity of the transmitted beam through the sample by the intensity of the transmitted beam through a cell fille
d with water. The intensities were measured with a power meter in the primary beam after the index matching vat. However, if the samples are extremely turbid the diameter of the cell must be reduced to increase the amount of single scattering (cells down to $2.4\mathrm{mm}$ inner diameter are used).

Then the large curvature of the glass surface expands the primary beam in one dimension. This makes it impossible to measure the transmission accurately. In this case the transmission was measured in a separated flat cell with a thickness of $1\mathrm{mm}$ in parallel to the scattering experiment. The transmission $T$ used for the correction in Eq. (6) was then calculated using the Lambert-Beer law

$$
T = T _ {\mathrm {f c}} ^ {d / I}. \tag {7}
$$

$T_{\mathrm{fc}}$ is the transmission which was measured in the flat cell with a thickness of $l$ and $d$ is the diameter of the cylindrical cell which was used for the light-scattering experiment.

Thus we are able to measure time-resolved static scattering intensities on absolute scale even in highly turbid samples. The angular range is not restricted to the four angles of the standard setup, as we can move the goniometer arm with the fiber of the incoming laser beam and thus simultaneously vary the scattering angles correspondingly. The accessible angles are between $10^{\circ}$ and $150^{\circ}$ , which corresponds to a $q$ range between 0.003 and $0.03\mathrm{nm}^{-1}$ .

For extremely turbid samples the cell diameter must be reduced as much as possible in order to reduce multiple scattering and obtain a measurable intercept of the correlation function. For a cell diameter of $1.6\mathrm{mm}$ , the correspondingly lower optical quality of the cell automatically limits the smallest measurable angle to about $30^{\circ}$ . The resolution limit, defined by $\pi / q_{\mathrm{min}}$ , therefore is normally approximately $1\mu \mathrm{m}$ . The typical measurement time depends strongly on the turbidity of the sample.

Although the detected count rate of turbid samples is usually relatively high, one must consider that in turbid samples only the single scattering events give rise to a correlated signal. While for a diluted sample $60\mathrm{s}$ of measurement time might be sufficient, the time needed to achieve reasonable statistics increases therefore to $600\mathrm{s}$ or

C. Moitzi et al./Journal of Colloid and Interface Science 336 (2009) 565-574

567

more when highly turbid samples are used. Under such conditions the multiangle approach then becomes essential in any attempt to perform time-resolved measurements.

# 3. Materials and sample preparation

# 3.1. Latex suspensions

Surfactant-free white sulfate latex particles with a diameter of $190\mathrm{nm}$ , a concentration of $8.1\mathrm{wt}\%$ , and a narrow size distribution $(3.1\%)$ from Interfacial Dynamics Corporation (OR, USA) were used. To adjust the concentration the suspension was diluted with distilled water. The electrostatic interactions were screened by the addition of $5\mathrm{mM}$ KCl $(\geqslant 99\%)$ Sigma).

# 3.2. Skim milk preparation

A 9.46 wt% of Nilac skimmed low-heat milk powder (NIZO, The Netherlands) was dispersed in Milli-Q water at room temperature and stirred at $40^{\circ}\mathrm{C}$ for $1\mathrm{h}$ . This leads to a protein solution with a concentration of $3.47\mathrm{wt\%}$ . After adding $0.02\mathrm{wt\%}$ of sodium azide the solution was kept for $12\mathrm{h}$ at $4^{\circ}\mathrm{C}$ . For lower casein concentrations the skim milk was diluted in its own serum. The serum was obtained by ultrafiltration through a regenerated cellulose membrane (YM10, MWCO 10,000 Da) from Millipore Corporation (MA, USA).

Prior to acidification the milk was allowed to equilibrate at least for $1\mathrm{h}$ at $25^{\circ}\mathrm{C}$ . The pH shift was induced by the addition of glucono- $\delta$ -lactone (GDL, $\geqslant 99\%$ , Sigma).

# 4. Results and discussion

To show the ability of our new instrument to measure the static scattering intensities as well as the intensity correlation function even in highly turbid samples correctly test experiments using suspensions of latex spheres of known size and narrow size distribution were measured. The concentration was increased from completely transparent samples up to the limit of the measurement. The concentration at which this limit is reached depends directly on the path length of the laser beam in the sample.

In principle it is possible to reduce this path length to a few $100\mu \mathrm{m}$ by using a squared sample cell and measuring close to the edge. A so-called $\theta -2\theta$ scheme has been implemented previously using an additional sample goniometer to move the scattering cell accordingly in order to use this approach also for angles different from $90^{\circ}$ [20]. However, by doing so it is only possible to measure at one particular angle at any given time and simultaneous multi-angle measurements are not possible.

Therefore it is impossible to measure the static scattering curve in a time-resolved way. This is only possible when all needed angles are measured at the same time and cylindrical cuvettes are used. In our experiments we have thus restricted ourselves to sample cells with an inner diameter of $2.4\mathrm{mm}$ or larger. Because of lensing effects due to the curvature of the glass surface it is difficult to further decrease the cell size.

# 4.1. Alignment procedure

The use of thin cylindrical cuvettes makes the measurement extremely sensitive to an exact alignment of the sample cell. The large curvature of the glass and the different refractive indices of the index matching liquid, the glass, and the sample cause refraction of the incident laser beam if the beam does not pass exactly through the center of the cell. The same is true for the scattered light on its path toward the detector. In the present case, the interfaces cause refraction to larger angles with respect to the normal line when the light enters the cell. The light path length within

the sample and the detected scattering vector are altered when the cells are shifted relative to the beam. These effects become more pronounced the thinner the cell is. To guarantee exact measurements attention must be paid in exactly aligning the sample cell relative to the incoming laser beam. In Fig. 1 the effects of the shift of the sample cell perpendicular to the laser axis and the axis of the cuvette on the static intensities (Fig. 1a) and the hydrodynamic radii (Fig. 1b) are shown. A suspension of latex spheres of $190\mathrm{nm}$ in diameter was used for these experiments. The concentration was $0.1\mathrm{wt}\%$ which corresponds to a transmission of $15.5\%$ for the cells used $(2.4\mathrm{mm}$ inner, $3.0\mathrm{mm}$ outer diameter).

The changes in the static scattering intensities are mostly due to changes in the detected scattering vector. If the cell is moved perpendicular to the laser beam the incident laser beam hits the surface of the cell not at a right angle and is refracted. If the cell is moved slightly to the left side a larger scattering vector compared to the perfect alignment is detected in a detector placed at the left side of the instrument. For a cell displacement to the right side, the inverse situation occurs. Such a displacement leads to an additional systematic error in $I(q)$ . This behavior was exactly found in the experiment. In turbid samples the changing light path length in the sample is amplifying the effect on the static intensity.

In DLS the changes in the scattering vector have a strong influence as well. The correlation functions decay proportional to $\exp(-Dq^2\tau)$ with the correlation time $\tau$ ( $D$ is the diffusion coefficient of the scattering particles). This means that a small error in the scattering vector can cause a relatively large error in the hydrodynamic radius which is calculated from $D$ via the Stokes-Einstein relation [23].

In contrast
to the static experiment, however, a variation of the light path length in turbid samples has no effect on the decay time whatsoever, only the intercept is affected. In Fig. 1b the dependence of the apparent hydrodynamic radius on the position of the cell is shown. Again, the detectors placed on different sides of the instrument shown different trends. In general, the radius appears to become smaller if the detected scattering vector becomes larger (the nominal scattering vector given by the position of the detector is used for the calculation).

The changes in the apparent hydrodynamic radius are striking. Obviously there is only one position of the cell where all measured radii coincide. Moreover this radius is exactly the one specified by the supplier. Therefore the measurement of the dynamics as a function of the position of the cell allows for the precise determination of the center of the cell. Admittedly this is only possible for monodisperse samples or particles with a small size that lead to a form factor $P(q) \simeq 1$ at all angles.

If the sample cell is shifted along the laser axis there is no difference between detectors placed on the left and on the right side of the instrument. At all detector positions a decrease of the singly scattered intensity is observed if the cell is moved toward the laser source (Fig. 2a). This is because scattering vectors which are too large are detected. For the same reason the apparent hydrodynamic radii are changing in the same direction (Fig. 2b). The exact alignment of the sample cell in this direction is obviously not possible by the experiment shown in Fig. 2. Therefore the goniometer was rotated as far as possible $(50^{\circ})$ after the alignment perpendicular to the beam (Fig. 1). Then the position in the other direction was optimized.

# 4.2. Test measurements with model system

In a next step we have investigated the ability of our instrument to perform multiangle dynamic and static light-scattering experiments with turbid samples and in particular demonstrate the possibility to obtain absolute intensities to give access to molar mass determination and structure factor measurements. We use

568

C. Moitzi et al./Journal of Colloid and Interface Science 336 (2009) 565-574

![](dt=2026-05-11/ht=06/b66dbf446b33fcf04c3af70704b128ab6948bf2b585a4dd248d5f3f8f2a67715.jpg)

![](dt=2026-05-11/ht=06/8b5caaf4f79f2b0a68526e790dcc10361463b1077968e83267289193a2000045.jpg)

monodisperse model suspensions with exactly known properties for this purpose. To avoid any influence of interactions $5\mathrm{mM}$ KCl was added [24]. This results in a hard sphere-like behavior, for which we expect a negligible effect of interparticle interactions at volume fractions up to $\phi \leqslant 0.003$ despite the high turbidity. An estimate using the well-known relationship for the dependence of the collective diffusion coefficient $D_{C}$ and the osmotic compressibility $\mathrm{d}\Pi/\mathrm{d}c$ of hard spheres results in deviations of less than $0.5\%$ for $D_{C}$ and $2.4\%$ for $\mathrm{d}\Pi/\mathrm{d}c$ due to interaction effects.

The reliability of the dynamic light-scattering measurement in turbid samples is demonstrated in Fig. 3. In Fig. 3a the intensity correlation functions of suspensions of latex spheres at different concentrations are shown. The intercept is decreasing with increasing concentration as expected. This is reflecting the increasing amount of multiply scattered light in the detected signal. If the

correlation functions are normalized with respect to their intercept one can see that they overlap almost perfectly. Only the highest concentration, with a transmission of $0.4\%$ only, gives a significantly different result. This is shown in Fig. 3b for one selected angle. The results of the other angles were very similar. As a consequence the hydrodynamic radii calculated by the cumulant method were very close $(+/-2\mathrm{nm})$ to the nominal value of $95\mathrm{nm}$ as specified by the supplier for all concentrations and all angles, except for the highest concentration.

There, deviations due to the very low intercept and correspondingly very low signal-to-noise ratios were observed. It must be noted that the form factor of the particles significantly decreases with increasing angle in the observed $q$ range, while the intensity of the multiply scattered light remains almost constant. As a result the ratio of singly scattered intensity to the overall intensity becomes smaller

C. Moitzi et al./Journal of Colloid and Interface Science 336 (2009) 565-574

569

![](dt=2026-05-11/ht=06/656ce199c6808de073a7dc4db8edff713e3ac0cc157bbf7bc09fb2160197854f.jpg)

![](dt=2026-05-11/ht=06/513317fccff11a7326966f1a536dd1f114a763448a0ee5833d405a8f527b3b91.jpg)

at larger angles for the most turbid samples. The resulting hydrodynamic radii are shown in Table 1.

In Fig. 4a the scattered intensities, normalized by the concentration of the suspension, are shown as a function of the scattering vector $q$ . For comparison the theoretical form factor calculated with Mie theory [25] is plotted as well. In Fig. 4b the transmission of the samples as a function of their concentration is shown. In agreement with the Lambert-Beer law there is a linear relation between the concentration and the logarithm of the turbidity. One can clearly see (Fig. 4a) that the scattering curves agree with the predicted one at all concentrations.

Also the absolute calibration is in quantitative agreement. Only at the highest concentrations the errors of the transmission measurement and the determination of the amount of multiple scattering caused by the very low intercept add up to a relatively large uncertainty of the absolute calibration of about $20\%$ .

As a rule of thumb it can be stated that both SLS and DLS are reliable as long as the transmission is above $2\%$ .

# 4.3. Time-resolved measurements of skim milk acidification—a feasibility study

To demonstrate the effectiveness of our new instrument, experiments performed during the acidification of skim milk are reported here. Casein micelles are a key protein component of milk and clearly can be looked at as a naturally occurring food colloid of very high industrial relevance. Correspondingly they have been extensively studied in the past. Under native conditions, their colloidal properties are well described by a model of an electrosterically stabilized colloid particle, where a polyelectrolyte brush formed by the so-called $\kappa$ -casein provides the necessary colloidal stability. In the yoghurt-making process [26-30] the pH of milk

570

C. Moitzi et al./Journal of Colloid and Interface Science 336 (2009) 565-574

![](dt=2026-05-11/ht=06/695ca97a7ca5c9df3cb30db32bd490fa45f4f803043db273bc014409f51d08b9.jpg)

![](dt=2026-05-11/ht=06/90798fa7048fd7684e73cf9447a4be80a5b06383e679c89b127975c822721cce.jpg)

is decreased until the $\kappa$ -casein brush becomes neutralized, collapses, and loses its stabilizing power. Aggregation of the casein micelles sets in and a gel is formed. To follow this process by light-scattering time-resolved experiments are needed. However, an additional difficulty arises from the self-assembled nature of the casein micelles which means that strong dilution of the system must be avoided. In order to mimic processing conditions the experiments must be performed at a high concentration, which means in the regime of excessive multiple scattering.

While there are studies using DW
S to overcome this problem [31], they provide only the temporal resolution of average proper

![](dt=2026-05-11/ht=06/855a94612d584e46ae62e7e1b0ead0ec03486ba74476554d52764fb66d9a1bb1.jpg)

![](dt=2026-05-11/ht=06/861841f78a52a743e427b24b6d4d39eb359d100d6e3e987585c8dedcd444a385.jpg)

ties such as the mean self-diffusion coefficient or the transport mean free path and in particular do not allow obtaining $q$ -dependent static information or absolute molar masses of the aggregates formed in the early stages of aggregation.

A flat cell static light-scattering instrument was applied by Lehner and co-workers [32] to investigate the rennet-induced casein aggregation in undiluted milk in a time-resolved way. However, the instrument gives access to the static intensities only and not to the dynamics. In addition it is, to our knowledge, not possible to achieve absolute calibration using this approach. This means that absolute values for the aggregate weight and the aggregation number are not accessible.

Table 1 Hydrodynamic radii of latex particles ( $190\mathrm{nm}$ diameter) measured at varying concentrations.

![](dt=2026-05-11/ht=06/a57988ae98243a6214d54700b780623f42319a67a46093ad6adb694907cb72b5.jpg)

<table><tr><td>Concn. [wt%]</td><td>0.003</td><td></td><td>0.03</td><td></td><td>0.1</td><td></td><td>0.2</td><td></td><td>0.3</td><td></td></tr><tr><td>Transmission</td><td>0.930</td><td></td><td>0.570</td><td></td><td>0.155</td><td></td><td>0.022</td><td></td><td>0.004</td><td></td></tr><tr><td>Angle [degree]</td><td>RH[nm]</td><td>β</td><td>RH[nm]</td><td>β</td><td>RH[nm]</td><td>β</td><td>RH[nm]</td><td>β</td><td>RH[nm]</td><td>β</td></tr><tr><td>30</td><td>94.8</td><td>0.116</td><td>95.8</td><td>0.105</td><td>95.3</td><td>0.075</td><td>93.3</td><td>0.042</td><td>99.4</td><td>0.022</td></tr><tr><td>60</td><td>94.6</td><td>0.159</td><td>95.5</td><td>0.135</td><td>95.3</td><td>0.079</td><td>93.2</td><td>0.029</td><td>91.0</td><td>0.008</td></tr><tr><td>90</td><td>96.2</td><td>0.159</td><td>95.1</td><td>0.128</td><td>96.2</td><td>0.058</td><td>93.4</td><td>0.013</td><td>91.2</td><td>0.002</td></tr><tr><td>110</td><td>96.5</td><td>0.160</td><td>96.1</td><td>0.124</td><td>96.5</td><td>0.047</td><td>93.8</td><td>0.006</td><td>2.7a</td><td>0.001</td></tr></table>

The cross-correlation functions measured at four different angles were analyzed with the cumulant method. a This unrealistic value is a result of an extremely small intercept of about $0.1\%$

C. Moitzi et al./Journal of Colloid and Interface Science 336 (2009) 565-574

571

We believe that our multiangle 3D instrument is ideal to study these aggregation processes under native conditions, because it allows for simultaneous static and dynamic light scattering and provides a complete set of data which was not possible before.

In principle we are able to measure skim milk even at the natural casein concentration. However, it turns out that at this concentration the repulsive interactions between the caseine micelles significantly alter the static scattering signal. This causes an error in the Guinier extrapolation. Quantitative values for the

forward scattering intensity and the radius of gyration can therefore only be obtained for weakly diluted samples. This, however, is a restriction due to the system under investigation and not an insufficiency of the instrumental setup. Some details about the concentration dependence can be found in Supporting Information.

The experiments were started immediately after the pH shift was initiated by the addition of glucono- $\delta$ -lactone to skim milk. The pH of the sample was recorded in a separate cell. In Fig. 5a the raw data, which were only corrected for fluctuations in the laser intensity

![](dt=2026-05-11/ht=06/70ca4224a736482b379bda6623eb2668713dd09d2d4952ba6c5a095144b14ba9.jpg)

![](dt=2026-05-11/ht=06/6d8a53b169739271c13102a4ecc17d8b4b3303eef427901e0efea0af4c10e382.jpg)

![](dt=2026-05-11/ht=06/a2e3746b5af594208c2d3f96c80815da197c9beef129678b60a210b1250d5f3e.jpg)

![](dt=2026-05-11/ht=06/6d2361a40ff2393b30d62c38cc4bde3d270d1e4c67790d123b67a20fe0a84669.jpg)

![](dt=2026-05-11/ht=06/cb0f19fe65680de33cd2a20f2fd0313cd8f26263d0d7b7611e67fa4f5c311142.jpg)

![](dt=2026-05-11/ht=06/ad3155b9ad2009d6ad2ebfb8234f440c3e7ff29bde1a76ed938ce3bc407e52d7.jpg)

572

C. Moitzi et al./Journal of Colloid and Interface Science 336 (2009) 565-574

and placed on absolute scale, are shown. No attempts to correct for the multiply scattered light have been made. There are only minor changes in the scattered intensity until the aggregation sets in after about 8000 s. Then the intensity drops, which is counterintuitive at first sight. For an aggregating system an increasing scattering intensity is expected. This is due to the formation of larger clusters, which result in a strong increase of the average molar mass of the clusters that will diverge at the gel point. In the right panel of Fig.

5 $I(0)$ and $R_{\mathrm{G}}$ as determined by a Guinier extrapolation are shown. The data, however, are distorted by multiply scattered light. Some remarks concerning the choice of the angular range for the Guinier extrapolation are given in Supporting Information.

In a next attempt we corrected for the multiply scattered light by using the reduction in intercept as described in Eq. (5). If the contribution from multiply scattered light is removed accordingly, the values are lower but the trends remain unchanged. This is shown in Fig. 5b which demonstrates that the drop of the scattered intensity after $8000\mathrm{s}$ is now even more pronounced. The reason for this behavior can be found in the strongly increasing turbidity of the sample.

The cluster formation results not only in a strongly enhanced scattering cross section but also a correspondingly enhanced contribution from multiple scattering. At the same time the dramatically increased probability for scattering induces a strong reduction of the incident laser beam as well as in a strong reduction of the scattered light on its path to and from the scattering volume. Therefore the Guinier extrapolation (results shown in the right panel of Fig. 5b) is still not giving the correct results for the forward scattering intensity.

The data thus must be corrected for the concomitant changes in the transmission that are directly related to the visibly increasing turbidity of the sample using Eq. (6). The resulting angular dependence of the singly scattered light intensity on absolute scale as a function of time is shown in Fig. 5c. Now one can clearly see the strongly enhanced intensity at low angles and the increasing angular dependence as the aggregation proceeds.

The data set shown in Fig. 5c provides very detailed information about the temporal evolution of the overall weight average molar mass of the casein micelles, the average dimension through the radius of gyration $R_{\mathrm{G}}$ (shown in the right panel of Fig. 5). The evolution of all these parameters and the hydrodynamic radius as function of the pH are summarized in Fig. 6. The forward scattering intensity $I(0)$ and the radius of gyration $R_{\mathrm{G}}$ were obtained by a Guinier extrapolation, and the hydrodynamic radii $R_{\mathrm{H}}$ were
calcu

![](dt=2026-05-11/ht=06/67f2a4dda1a63bebda63450d8a45beac8282419c222921c1ad961f88ec99fe32.jpg)

lated by a cumulant analysis from the correlation functions and extrapolated to a scattering vector on zero.

Fig. 6 provides detailed information about the structural rearrangements of the casein micelles caused by the pH shift as well as the subsequent aggregation and sol-gel transition once the stabilizing $\kappa$ -casein brush loses its stabilizing power. While a detailed analysis of the data is beyond the scope of this paper and will be reported elsewhere, it is worth pointing out a rather surprising feature that once again shows why time-resolved 3D measurements are vital to study nonequilibrium properties of this and other similar systems.

The data do not exhibit the classical features of aggregating colloids with a monotonically increasing forward scattering intensity that appears to diverge at the gel point, but instead first shows a well pronounced minimum at about $4000\mathrm{s}$ or a pH of 5.5. Casein micelles are known to dissolve partially during acidification as a result of the solubilization of the calcium phosphate clusters that occurs at lower pH [33,34].

Since these clusters are thought to be largely responsible for maintaining the integrity of the micelles, their dissociation in turn causes significant internal rearrangements and partial dissolution of the casein micelles. Due to the fact that casein micelles are self-assembled structures, this interplay between disintegration and aggregation is strongly concentration dependent.

An understanding of the sol-gel process induced by acidification of milk is thus only feasible if time-resolved light-scattering experiments on an absolute scale are possible under industrially relevant conditions, where the casein solutions are highly turbid. Fig. 6 demonstrates that this is indeed possible with our new multiangle 3D instrument, and we will present a full account of a systematic study of this process in a forthcoming paper.

# 5. Conclusion

Colloidal suspensions under industrially relevant conditions are frequently turbid and thus very difficult to characterize with optical techniques commonly used for this purpose. The implementation of cross-correlation techniques in light-scattering techniques has recently provided us with instruments that significantly extend the range of applicability of dynamic and static light scattering. This is particularly important for self-assembled structures that do not allow for a strong dilution without altering their properties.

The importance of working under industrially relevant conditions and the possibilities provided by the so-called 3D cross-correlation light-scattering technique has, for example, recently been demonstrated for milk samples [21]. While this technique allows for measurements with highly turbid samples due to the strong suppression of multiply scattered light, the individual measurements last much longer due to the reduced signal-to-noise ratio inherent to this measurement principle.

In particular for polydisperse dispersions of larger particles, where measurements at different scattering angles are required in order to obtain a complete and meaningful experimental characterization, this results in rather long measurement times. While this poses no problem for equilibrium systems that are fully stable, it clearly limits the applicability of the technique to nonequilibrium systems that exhibit a temporal evolution of the particle properties.

Here we now have described the design and implementation of a multiangle 3D instrument that allows a significant reduction in the measurement time for multiangle measurements with turbid samples.

The combination of a simultaneous measurement of static and dynamic light scattering at four angles with an on-line turbidity measurement has proven to be very powerful in determining the temporal evolution of a turbid colloidal sample undergoing restructuring and aggregation. The fact that the measurement time required for an individual measurement at four angles is reduced by a factor of 4 and that all angles and the transmission are always

C. Moitzi et al./Journal of Colloid and Interface Science 336 (2009) 565-574

573

measured at the same time is crucial for such systems. Moreover, the fact that the described alignment and normalization schemes allow for measurements on absolute scale even for samples exhibiting strong multiple scattering is a very important asset in such studies.

This not only provides time-resolved information about the overall size and shape of the particles through measurements of the mean apparent radius of gyration and hydrodynamic radius, but also on the weight-average apparent molar mass on absolute units, which is indispensable when trying to distinguish among aggregation, dissolution, and restructuration.

Our new instrument thus extends the applicability of dynamic and static light scattering to efficiently investigate the temporal evolution of aggregating or phase separating systems to turbid dispersions that are difficult to study otherwise.

# Acknowledgments

For the realization of the multiangle 3D instrument the help of the mechanic and electronic workshop of the physics department of the University of Fribourg was essential. We thank Hugo Bissig, Ben Graham Nasser, and Oswald Raetzo for their important help. The work was partially financed by the Nestlé Research Center, Lausanne, Switzerland.

# Appendix A. Supplementary data

Supplementary data associated with this article can be found, in the online version, at doi:10.1016/j.jcis.2009.04.043.

# References

[1] B. Chu, Laser Light Scattering, second ed., Academic Press, San Diego, 1991.

574

C. Moitzi et al./Journal of Colloid and Interface Science 336 (2009) 565-574