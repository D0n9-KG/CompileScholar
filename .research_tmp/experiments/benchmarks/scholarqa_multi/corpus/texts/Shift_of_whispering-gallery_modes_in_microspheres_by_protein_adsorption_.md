# Shift of whispering-gallery modes in microspheres by protein adsorption

S. Arnold, M. Khoshima, and I. Teraoka

Microparticle Photophysics Lab (MP³L), Polytechnic University, Brooklyn, New York 11201

S. Holler*

Los Gatos Research, Mountain View, California 94041

F. Vollmer

Center for Studies in Physics and Biology, Rockefeller University, New York, New York 10021

Received July 16, 2002

Biosensors based on the shift of whispering-gallery modes in microspheres accompanying protein adsorption are described by use of a perturbation theory. For random spatial adsorption, theory predicts that the shift should be inversely proportional to microsphere radius $R$ and proportional to protein surface density and excess polarizability. Measurements are found to be consistent with the theory, and the correspondence enables the average surface area occupied by a single protein to be estimated. These results are consistent with crystallographic data for bovine serum albumin.

The theoretical shift for adsorption of a single protein is found to be extremely sensitive to the target region, with adsorption in the most sensitive region varying as $1 / R^{5 / 2}$ . Specific parameters for single protein or virus particle detection are predicted. © 2003 Optical Society of America

OCIS codes: 170.4520, 300.6490.

In the recent past, the need for miniature biosensors for the detection of infectious agents, toxins, protein, and DNA has taken on added urgency as the world anticipates further bioterrorism. Recently Vollmer et al. reported specific detection of unlabeled biomolecules on a spherical surface (radius $R \cong 0.15 \mathrm{~mm}$ ) from the frequency shift of whispering-gallery modes (WGMs). The modes were stimulated in a dielectric sphere immersed in an aqueous environment by means of coupling light evanescently from an optical fiber.

The authors claimed unprecedented sensitivity for the adsorption of protein molecules with spatial uniformity. In what follows, we (1) introduce an optical theory that describes this effect in an asymptotic limit ( $2\pi R / \lambda \gg 1$ ), (2) compare the predicted size dependence with that from new experiments, and (3) calculate the effect of reducing the size while placing protein molecules at specific locations on the sphere surface. We show that, for particular locations, the sensitivity to single-protein adsorption can be enhanced by orders of magnitude.

Figure 1 illustrates the basic configuration of interest. Light from a tunable distributed feedback laser is coupled into a WGM of the sphere from an eroded optical fiber<sup>3</sup> and circulates about the equator. Resonant modes are detected from dips in the transmission through the fiber. A protein molecule diffuses to the sphere's surface from the surrounding aqueous medium and is adsorbed at position $\mathbf{r}_i$ , where it interacts with the evanescent field of the WGM. The index $i$ distinguishes each adsorbed protein molecule. This interaction polarizes the molecule, shifting the frequency of the mode.

To evaluate the shift $\delta \omega$ in angular frequency $\omega$ of a single protein molecule, it is useful to consider

the energy of interaction as a first-order perturbation to a single-photon resonant state, with semiclassical field $\mathbf{E}_0(\mathbf{r})\exp (i\omega t)$ . The evanescent tail of the field induces a dipole moment in the protein in excess of the displaced water, $\delta \mathbf{p}\exp (i\omega t)$ , causing a shift in the photon energy of the resonant state, $\hbar \delta \omega = -\delta \mathbf{p}\cdot \mathbf{E}_0^* (\mathbf{r}_i) / 2$ . The excess dipole moment can be represented in terms of the real part of an excess polarizability $\alpha_{\mathrm{ex}}$ , i.e.

, $\delta \mathbf{p} = \alpha_{\mathrm{ex}}E_0(r_i)$ . The fractional frequency shift for a protein positioned at $\mathbf{r}_i$ is given by the result of dividing the perturbation by the energy of the mode (i.e., $\hbar \omega$ ), as represented by integrating over the energy density in the interior:

![](dt=2026-02-22/ht=00/c989a89b0287ba9463c2fdece1cf43ee028247f796fb5121afe3eec2de543f24.jpg)

272

OPTICS LETTERS / Vol. 28, No. 4 / February 15, 2003

0146-9592/03/040272-03$15.00/0

© 2003 Optical Society of America

$$
\left(\frac {\delta \omega}{\omega}\right) _ {i} \cong \frac {- \alpha_ {\mathrm {e x}} | \mathbf {E} _ {0} (\mathbf {r} _ {i}) ^ {2} |}{2 \int \epsilon_ {s} | \mathbf {E} _ {0} (\mathbf {r}) | ^ {2} \mathrm {d} V}. \tag {1}
$$

The integral in the denominator is taken over the interior of the sphere, which contains the overwhelming majority of the mode energy $(>94\%)$ . This approximation simplifies the analysis by allowing the homogeneous permittivity, $\epsilon_{s}$ , of the sphere to be pulled through the integral. The factor of 2 preceding this integral results from adding equal electric and magnetic contributions.

It should be noted that for protein molecules, which are composed of a variety of amino acids, $\alpha_{\mathrm{ex}}$ is roughly proportional to the mass of the molecule, and the shift in frequency in accordance with relation (1) should behave in the same way.

Relation (1) represents the shift that is due to an individual molecule at an arbitrary position on the sphere, a point we will return to in calculating the optimal effect. However, in Ref. 1 it was reported that a large number of protein molecules are distributed over random locations on the sphere's surface. To account for all these molecules we sum the singular contribution in relation (1) over $N$ randomly located molecules and then turn this discrete sum into an integral over surface differentials, $\sum_{i}^{N}|E_{0}(\mathbf{r}_{i})|^{2}\cong \sigma_{p}\int |E_{0}(\mathbf{r})|^{2}\mathrm{d}A,$ where $\sigma_p$ , the protein surface density, is $N / (4\pi R^2)$ . With this transformation from a discrete to continuous sum, relation (1) becomes

$$
\frac {\delta \omega}{\omega} \cong - \frac {\alpha_ {\mathrm {e x}} \sigma_ {p}}{2 \epsilon_ {0} \epsilon_ {r s}} \frac {\int | \mathbf {E} _ {0} (\mathbf {r}) | ^ {2} \mathrm {d} A}{\int | \mathbf {E} _ {0} (\mathbf {r}) | ^ {2} \mathrm {d} V}, \tag {2}
$$

where $\epsilon_{s}$ is written in terms of a relative permittivity, $\epsilon_{s} = \epsilon_{0}\epsilon_{rs}$ .

We now evaluate relation (2) for a general TE mode for which the interior field at distance $r$ from the sphere center is given as $\mathbf{E}_0 = A_{\mathrm{in}}j_l(k_0r\sqrt{\epsilon_{rs}})\hat{\mathbf{L}} Y_{lm}$ , where $A_{\mathrm{in}}$ is the amplitude, $j_{l}(z)$ is a spherical Bessel function, $\hat{\mathbf{L}}$ is a dimensionless angular momentum operator ( $\hat{\mathbf{L}} = -i\mathbf{r} \times \nabla$ ), $k_0 = \omega / c$ with $c$ being the speed of light in vacuum, and $Y_{lm}$ is a spherical harmonic function. Fortunately both the surface and volume integrals in relation (2) contain precisely the same angular integrands. Consequently,

$$
\frac {\delta \omega}{\omega} \cong - \frac {\alpha_ {\mathrm {e x}} \sigma_ {s}}{2 \epsilon_ {0} \epsilon_ {\mathrm {r s}}} \frac {\left[ j _ {l} \left(k _ {0} R \sqrt {\epsilon_ {r s}}\right) \right] ^ {2} R ^ {2}}{\int_ {0} ^ {R} \left[ j _ {l} \left(k _ {0} r \sqrt {\epsilon_ {r s}}\right) \right] ^ {2} r ^ {2} \mathrm {d} r}, \tag {3}
$$

where $R$ is the radius of the sphere. On resonance, the volume integral in the denominator of relation (3) may be asymptotically $(2\pi R / \lambda >> 1)$ related to the surface value of $j_{l}^{2}$ through $\int_0^R [j_l(k_0r\sqrt{\epsilon_{rs}})]^2 r^2\mathrm{d}r\cong (R^3 /2)[j_l(k_0R\sqrt{\epsilon_{rs}})]^2[(\epsilon_{rs}- \epsilon_{rm}) / \epsilon_{rs}]$ , where $\epsilon_{rm}$ is the relative permittivity of the surrounding medium. Inserting this expression into relation (3), we find that the fractional frequency shift is given by a surprisingly simple formula:

$$
\frac {\delta \om
ega}{\omega} \cong - \frac {\alpha_ {\mathrm {e x}} \sigma_ {p}}{\epsilon_ {0} \left(\epsilon_ {r s} - \epsilon_ {r m}\right) R} = - \frac {\alpha_ {\mathrm {e x}} \sigma_ {p}}{\epsilon_ {0} \left(n _ {s} ^ {2} - n _ {m} ^ {2}\right) R}, \tag {4}
$$

where $n_s$ and $n_m$ are the refractive indices of the sphere and the aqueous medium, respectively. The analysis of $\delta \omega / \omega$ for TM modes involves changing the field in relation (2). The result produced by a similar analysis has the same $\alpha_{\mathrm{ex}} \sigma_p / R$ dependence with numerically calculated shifts that differ from the TE shifts by only a few percent for our silica-water interface.

The $1 / R$ size dependence in relation (4) is expected for a homogeneous sphere. If such a sphere accretes a layer that is $\delta R$ thick, it must preserve the product $k_{0}R$ for a given resonance, and consequently $\delta k_{0} / k_{0} = \delta \omega /\omega = -\delta R / R$ . However, the formula becomes more complicated when the sphere is optically heterogeneous, as revealed in relation (4).

Nonetheless, when the surface is saturated with protein, as revealed by no additional shift regardless of the external concentration, a plot of $-\delta \omega /\omega$ versus $1 / R$ will have slope $\delta R_{\mathrm{eff}}$ , the effective thickness of the layer. It should be noted that $\delta R_{\mathrm{eff}}$ as defined can be negative, if the absorbed material has a polarizability less than that of an equal volume of water. This odd circumstance is not the case for protein adsorption, since the optical permittivity of proteins is higher than that of water.

In fact, proteins have permittivities close to that of quartz.

We have performed experiments on the adsorption of bovine serum albumin (BSA) protein on quartz microspheres. The silica glass surface is sensitized for protein adsorption by chemical modification with vapor-phase 3-aminopropyltriethoxysilane following oxygen plasma cleaning. The resonance shifts $-\delta \omega/\omega$ measured for complete saturation by use of a current-tuned distributed feedback laser operating at a nominal wavelength of $1.34\mu \mathrm{m}$ are shown as a function of $1 / R$ in Fig.2.

Protein injection was implemented only after equilibrium was reached at $23^{\circ}\mathrm{C}$ . The system was verified to have returned to this temperature when wavelength-shift measurement was taken. The spheres ranged in radius from 88 to $232\mu \mathrm{m}$ ( $412 < 2\pi R / \lambda < 1087$ ). Within the scatter in the data over this size range, a $1 / R$ size dependence appears reasonable. The slope of the fit is $\delta R_{\mathrm{eff}} = 3.6\mathrm{nm}$ .

An effective thickness of $3.6\mathrm{nm}$ is very close to the smallest dimension of BSA as revealed through x-ray

![](dt=2026-02-22/ht=00/c99c76f0add3c6f4bfd4d79dcfdaaf1d93785381f3176128c432f5ab1e7776ec.jpg)

February 15, 2003 / Vol. 28, No. 4 / OPTICS LETTERS

273

crystallography. $^{10}$ BSA resembles a thick pancake with a heart-shaped profile. The smallest dimension is the height of the pancake. Furthermore, from the effective thickness and relation (4), it is possible to estimate the molecular surface density, $\sigma_{p} = \delta R_{\mathrm{eff}}\epsilon_{0}(n_{s}^{2} - n_{m}^{2}) / \alpha_{\mathrm{ex}}$ .

We calculate this surface density by use of the excess polarizability arrived at from differential refractive-index measurements $^{1}$ $[\alpha_{\mathrm{ex}} = 4\pi \epsilon_{0}(3.85\times 10^{-21}\mathrm{cm}^{3})]$ , and the usual refractive indices for quartz and water, with the result $\sigma_{p} = 2.9\times 10^{12}\mathrm{cm}^{-2}$ . So a BSA molecule occupies an area $\sigma_{p}^{-1} = 3.4\times 10^{-13}\mathrm{cm}^{2}$ . This agrees well again with crystallographic data, for which the area of the heart-shaped projection is $3.7\times 10^{-13}\mathrm{cm}^{2}$ .

It appears that BSA forms an extremely compact layer on the microsphere surface.

Finally, we are interested in the possibility of single-protein detection. Single-protein detection would be possible by looking at steps in the change of $\delta \omega /\omega$ with time, and this in turn provides a possible means for separately measuring $\alpha_{\mathrm{ex}}$ . Since the light within a WGM circumnavigates the equator $(\theta = \pi /2)$ in an orbit that is confined to a thin ring, molecules at polar angles outside the ring cannot influence the mode frequency. The greatest signal comes from molecules that stick at $\theta = \pi /2$ .

For a TE mode that circulates at the equator, $l = m$ , and the angular intensity is proportional to $|\hat{\mathbf{L}} Y_{ll}|^2$ , which for large $l$ is proportional to $|Y_{ll}|^{2}$ . So the ratio of the frequency shift for a protein at the equator to that averaged over random positions on the surface is enhanced by a factor $EF = 4\pi |Y_{ll}(\pi /2,\varphi)|^2$ . This spatial enhancement $EF$ can be significant. For the average size particle used in Fig. 2, $l\sim 1000$ and $EF\cong 36$ .

To obtain the average shift for an individual protein at a random position, we set the surface density in relation (4) to $\sigma = 1 / (4\pi R^2)$ with the result $(\delta \omega /\omega)_r = -\alpha_{\mathrm{ex}} / [4\pi \epsilon_0(n_s^2 -n_m^2)R^3 ]$ . The shift that is due to a single protein at the equator is $(\delta \omega /\omega)_e = EF\times (\delta \omega /\omega)_r$ , or

$$
(\delta \omega / \omega) _ {e} = - \frac {\alpha_ {\mathrm {e x}} | Y _ {l l} (\pi / 2 , \varphi) | ^ {2}}{\epsilon_ {0} (n _ {s} ^ {2} - n _ {m} ^ {2}) R ^ {3}}. \qquad (5)
$$

This single-protein shift has a large size dependence. Since $|Y_{ll}(\pi /2,\varphi)|^2$ increases roughly in proportion to $l^{1 / 2}$ or $R^{1 / 2}$ , the single-protein shift should go as $R^{-5 / 2}$ . Currently, we can detect a fractional frequency change as small as $10^{-8}$ . Since we can see a shift of one fiftieth of a linewidth, this requires that $Q$ be $2\times 10^{6}$ . This value of $Q$ is controlled by overtone vibrational absorption of water at $1.34\mu \mathrm{m}$ and the size of the microsphere. Leakage at the quartz-water interface limits the smallest radius for which this sensitivity is reasonable to approximately $50~\mu \mathrm{m}$ . For a first-order TE mode within such a particle, and for

a wavelength of $1.34\mu \mathrm{m}$ , $4\pi |Y_{ll}(\pi /2,\varphi)|^2 = 20.8$ . Under these conditions, the smallest detectable single-protein polarizability is $\alpha_{\mathrm{sd}} = 4\pi \epsilon_0(2.4\times 10^{-17}\mathrm{cm}^3)$ or 6230 times the polarizability of BSA. Protein masses seldom exceed $10^{6}$ Da, which is only 15 times the mass of BSA. Thus single-protein measurements are unrealistic from the resonance shift at $1.34\mu \mathrm{m}$ . We may overcome the problem by working in the blue, where water absorption is reduced by more than a factor of 100, and by choosing a material for the microsphere with a larger refractive index.

As an example, a microsphere of amorphous sapphire has a refractive index of 1.7 at a wavelength of $400\mathrm{nm}$ (blue diode laser with external cavity), which enables the radius to be reduced to approximately $3.6\mu \mathrm{m}$ for $Q$ of $2\times 10^{7}$ in water. Assuming that we are able to see a shift of a fiftieth of a linewidth as before, the least measurable fractional shift would be $10^{-9}$ . The minimum detectable polarizability projected from Eq. (5) is now approximately three times the polarizability of BSA, a number that is consistent with large protein molecules such as thyroglobulin, ferritin, and virus particles (e.g., lambda phage). Adsorption onto the equator may be promoted by selective silanization of the equator.

Research at the Polytechnic was supported by a National Science Foundation grant (BES-0119273). F. Vollmer was supported by a fellowship of the Boehringer Ingelheim Fonds. S. Arn
old's e-mail address is arnold@photon.poly.edu.

*Present address, Sandia National Laboratories, Albuquerque, New Mexico 87185.

# References

274

OPTICS LETTERS / Vol. 28, No. 4 / February 15, 2003