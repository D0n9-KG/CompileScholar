# Leaky-mode assisted fluorescence extraction: application to fluorescence enhancement biosensors

Nikhil Ganesh $^{1,3}$ , Ian D. Block $^{1, + }$ , Patrick C. Mathias $^{1, + }$ , Wei Zhang $^{1,3, + }$ , Edmond Chow $^{2}$ , Viktor Malyarchuk $^{3}$ and Brian T. Cunningham $^{1*}$

$^{1}$ Department of Electrical and Computer Engineering, Nano Sensors Group

University of Illinois at Urbana-Champaign

208 North Wright Street, Urbana, Illinois, 61801

$^{2}$ Micro and Nanotechnology Laboratory, University of Illinois at Urbana-Champaign, 208 North Wright Street, Urbana, Illinois, 61801

$^{3}$ Department of Materials Science and Engineering, University of Illinois at Urbana-Champaign, 1304 West Green Street, Urbana, IL 61801

+These authors contributed equally to this study

*Corresponding author: bcunning@illinois.edu

Abstract: Efficient recovery of light emitted by fluorescent molecules by employing photonic structures can result in high signal-to-noise ratio detection for biological applications including DNA microarrays, fluorescence microscopy and single molecule detection. By employing a model system comprised of colloidal quantum dots, we consider the physical basis of the extraction effect as provided by photonic crystals.

Devices with different lattice symmetry are fabricated ensuring spectral and spatial coupling of quantum dot emission with leaky eigenmodes and the emission characteristics are studied using angle-resolved and angle-integrated measurements. Comparison with numerical calculations and lifetime measurements reveals that the enhancement occurs via resonant redirection of the emitted radiation. Comparison of various lattices reveals differences in the enhancement factor with a maximum enhancement factor approaching 220.

We also demonstrate the first enhanced extraction biosensor that allows for over 20-fold enhancement of the fluorescence signal in detection of the cytokine TNF- $\alpha$ by a fluorescence sandwich immunoassay.

© 2008 Optical Society of America

OCIS codes: (050.6624) Subwavelength structures; (170.2520) Fluorescence microscopy.

# References and links

#103698 - $15.00 USD Received 7 Nov 2008; revised 11 Dec 2008; accepted 11 Dec 2008; published 15 Dec 2008

(C) 2008 OSA

22 December 2008 / Vol. 16, No. 26 / OPTICS EXPRESS 21626

# 1. Introduction

The efficient extraction of radiation emitted by light emitting structures such as semiconductor and organic light emitting diodes has been a long standing engineering issue. Amongst the various techniques that have been suggested, significant effort has been put into design and development of structures that make use of leaky photonic crystal $^{1}$ /surface plasmon $^{2}$ resonances, localized surface plasmon resonances $^{3}$ and roughness mediated random scattering $^{4}$ .

The photonic crystal (PC) approach essentially relies on the coupling of light that is trapped within the high index device layers to free-space via the creation of leaky modes by periodic structuring $^{5,6}$ . The ability to control the dispersion of these leaky modes by tailoring the PC properties also provides a powerful mechanism to redirect the emitted light into certain preferred directions, where it can be detected with greater efficiency.

Such a scheme is particularly interesting for the development of fluorescence biosensors, where efficient collection of the emitted radiation will allow for lowering the detection limits (by providing enhanced signal-to-noise ratio, SNR) in a wide variety of applications including DNA/Protein microarrays, sensitive fluorescence microscopy and potentially single molecule detection.

Here, we carry out an in-depth study of the enhanced extraction effect provided by PCs, in the context of fluorescence enhancement biosensors. A model system comprised of colloidal QD emitters spectrally and spatially coupled to the resonant modes of the PC is employed to study the effect of lattice period, resonant mode polarization and symmetry on

103698 - $15.00 USD Received 7 Nov 2008; revised 11 Dec 2008; accepted 11 Dec 2008; published 15 Dec 2008

(C) 2008 OSA

22 December 2008 / Vol. 16, No. 26 / OPTICS EXPRESS 21627

the enhancement effect. Angle-resolved and angle-integrated fluorescence, fluorescence lifetime measurements and Rigorous Coupled-Wave Analysis $^{7}$ (RCWA) simulations are employed to clarify the mechanism of enhancement. We show that the extraction enhancement results from redirection of radiation into free-space via coupling to the leaky modes of the PC. As a first demonstration of the effectiveness of this enhancement effect in fluorescence applications, we show an enhancement of over 20-fold in the fluorescence detection of the cytokine Tumor Necrosis Factor- $\alpha$ (TNF- $\alpha$ ) in a fluorescence sandwich immunoassay.

# 2. Guided-mode resonance

The study of resonant anomalies in periodic structures can be traced back to the studies by Wood $^{8}$ in 1902 and subsequent important contributions $^{9,10,11,12,13}$ that identified the Guided-mode resonance (GMR) phenomenon. Subsequently, the unique properties of the GMR effect have been successfully applied to filtering $^{14}$ , biosensing $^{15}$ , potentially enhancing energy harvesting $^{16}$ and feedback applications $^{17}$ .

The GMR effect was shown to arise in structures that comprised of a high index layer evanescently coupled to (or directly containing) a subwavelength periodic structuring, such as a PC slab $^{18}$ . The waveguided modes of such a high index layer become leaky in this scenario as the periodicity now allows coupling of these modes to free-space via phase matching into and out of the high index layer.

The response of such a device to radiation incident from free-space is well studied and is manifested as an efficient reflection resonance whose wavelength and line width can be arbitrarily tuned $^{19}$ . When light is incident upon this device, some portion is reflected and some is transmitted via the $0^{\text{th}}$ reflected and transmitted orders respectively. However, the existence of the periodicity allows phase matching of the rest of the incident energy into a leaky eigenmode, described in the one-dimensional case by $^{20}$ :

$$
\mathrm {k} _ {0} \mathrm {n} _ {\mathrm {s}} \sin (\theta) \pm \mathrm {m} 2 \pi / \Lambda = \beta \tag {1}
$$

where $\mathbf{k}_0$ is the free-space wave-vector, $\mathfrak{n}_{\mathrm{s}}$ is the index of the medium from which the external light is incident at an angle $\theta$ with the surface normal, $\mathfrak{m}$ is the diffraction order and $\Lambda$ is the period. $\beta$ is the real part of the propagation constant of the leaky eigenmode of the structure. Typically in devices utilizing the GMR effect, only the $0^{\mathrm{th}}$ orders propagate and higher evanescent orders excite the leaky modes. The leaky modes eventually lose energy in the form of waves propagating in the specular and transmitted directions, which interfere constructively and destructively with the $0^{\mathrm{th}}$ reflected and transmitted orders respectively, leading to complete reflection of the incident light[21].

Conversely in the weak coupling regime, radiation produced by point sources in close proximity to the device can not only directly emit into the substrate and superstrate regions, but can also couple to a leaky mode and be extracted along its dispersion. In such a case, (1) can be rewritten to determine the angle of escape as:

$$
\sin (\theta) = \left(\beta \pm \mathrm {m} 2 \pi / \Lambda\right) / \mathrm {k} _ {0} \mathrm {n} _ {\mathrm {s}} \tag {2}
$$

This implies that for any given wavelength of coupled light, the angle of escape can be arbitrarily chosen by appropriate choice of the photonic structure. An important point is that since $\beta$ is essentially an in-plane propagation vector, coupling to leaky modes automatically provides enhanced extraction as light that would have been lost as waveguided modes is now red
irected with specific control. Eq. (2) also suggests that the light will be extracted symmetrically (via 2 channels) about the surface normal in the superstrate and substrate regions, at different angles.

# 3. Experimental approach and results

# 3.1 Design and fabrication of PC slabs

In order to study the nuances of the enhanced extraction phenomenon, we begin by considering PCs with 3 different lattice symmetries (linear, square lattice of holes and

103698 - $15.00 USD Received 7 Nov 2008; revised 11 Dec 2008; accepted 11 Dec 2008; published 15 Dec 2008

(C) 2008 OSA

22 December 2008 / Vol. 16, No. 26 / OPTICS EXPRESS 21628

hexagonal lattice of holes) that possess leaky modes spectrally and spatially overlapping the emission spectrum of an ensemble of point sources (colloidal CdSe/ZnS QDs, chosen due to the ability to perform fluorescence measurements without significant photo-bleaching). Schematics of these structures and corresponding atomic force microscopy images of the fabricated devices are shown in Fig. 1.

The devices are comprised of a thin $\mathrm{TiO}_2$ waveguide $(\mathrm{n}_{\mathrm{wg}} = 2.13)$ layer deposited upon a flat glass substrate $(\mathrm{n}_{\mathrm{glass}} = 1.52)$ upon which resist gratings impregnated with QDs are patterned $(\mathrm{n}_{\mathrm{grating}} = 1.54)$ . By control of the period of these structures, the TE-polarized leaky modes that emit waves close to the device normal (normally coupled leaky modes) were spectrally matched to the emission maximum of the QDs $(\lambda_{qd} \sim 620 \mathrm{~nm})$ .

The rationale behind this design is to ensure that maximum fluorescence is extracted close to the device normal for efficient collection.

![](dt=2026-05-09/ht=07/9689e688cd486223674f436cecd101cf0239ae85da6d6d4c345d3e00ce8f54e7.jpg)

![](dt=2026-05-09/ht=07/bc0bb411f511e4e83dd9cce803212436396b839f32351081fde1d2377ff32027.jpg)

# 3.2 Far-field and angle-resolved fluorescence measurements

Following fabrication, we first examined the transmission properties of the structures for normally-incident white light. The response to TE-polarized illumination for the case of the linear lattice with a period $\varLambda_{lin}=340\mathrm{nm}$ is shown in Fig. 2(a) (red curve). A clear resolution-limited dip in the transmission is seen around $620\mathrm{nm}$ , which corresponds to the excitation of the GMR.

In order to verify if the fluorescence emitted by the QDs couples to this mode, the device was non-resonantly illuminated at an absorption wavelength of the QDs ( $\lambda_{ex}=457\mathrm{nm}$ ) using a continuous-wave laser (85 BLD, Melles Griot). The resulting fluorescence spectrum collected at normal incidence is shown in Fig. 2(a) (black curve). A drastic modification of the otherwise broad spectrum emission (full width at half maximum, FWHM $\sim35\mathrm{nm}$ ) from the QDs is seen exactly overlapping the leaky mode.

The fluorescence output at the peak resonant wavelength is boosted by over two orders of magnitude and the fluorescence linewidth

103698 - $15.00 USD Received 7 Nov 2008; revised 11 Dec 2008; accepted 11 Dec 2008; published 15 Dec 2008

(C) 2008 OSA

22 December 2008 / Vol. 16, No. 26 / OPTICS EXPRESS 21629

(FWHM) is reduced to less than $3\mathrm{nm}$ . The resonance provides enhanced extraction of the fluorescence emitted by the QDs in a direction engineered by control of the photonic lattice. We also fabricated structures with periods lesser and greater than $\varLambda_{lin}=340\mathrm{nm}$ and noted reduced extraction efficiency at normal incidence due to the inefficient spectral overlap between the fluorescence maximum $(\lambda_{qd})$ and the normally coupled TE leaky mode (data not shown).

By increasing the period until the normally coupled TM leaky mode spectrally overlapped the emission spectrum, we confirmed similar fluorescence enhancement (see supplementary information) and believe this to be due to the similar quality-factors for both TE and TM resonances. We also observed that strong extraction of light occurs into the substrate and escapes from the backside of the device (see supplementary materials).

![](dt=2026-05-09/ht=07/b323e867c907ecb6e73e56f2952d2dbe421d21d1e1827ebbc183e99141da9924.jpg)

![](dt=2026-05-09/ht=07/bba8b1f4240245e8b68096fdc49d4bc248b5e51d306823ed8dd4c76e91cdb464.jpg)

Insight into the coupling between the QD fluorescence and the leaky mode can be gathered by observing the electric field profile of the leaky mode, as shown in Fig. 2(b). The plot shows the spatial distribution of the electric field intensity for the normally coupled leaky mode, as calculated by RCWA. As expected, the majority of the intensity associated with the mode is confined within the waveguide layer with evanescent tails decaying into the superstrate and substrate regions.

It can be seen that the spatial overlap between the QDs (present in the outlined grating region) and the leaky mode is not particularly efficient. The coupling could be greatly enhanced by placing the QDs within the waveguide layer, which might be the strategy to employ in the case of light emitting systems such as LEDs/OLEDs.

However, in a fluorescence enhancement biosensor, the analyte will typically be bound on the surface of the device and calculations show that the mode intensity available for interaction with fluorescent molecules near the device surface is similar to that available to the QDs within the grating, making the results of our model QD system applicable to relevant biological applications. Similar results for the overlap of the leaky modes and electric field intensity were observed for the square and hexagonal lattices.

Eqs. (1), (2) indicate that GMRs can be excited over a range of wavelengths by control of the incident angle, and conversely, it follows that the various wavelength components of the emitted fluorescence will be extracted at different angles, along the dispersion of the leaky modes.

103698 - $15.00 USD Received 7 Nov 2008; revised 11 Dec 2008; accepted 11 Dec 2008; published 15 Dec 2008

(C) 2008 OSA

22 December 2008 / Vol. 16, No. 26 / OPTICS EXPRESS 21630

![](dt=2026-05-09/ht=07/a4f676f45fe6337d3a90e8e7d7251b6ea099587e648039486be8827c9dd0bc10.jpg)

![](dt=2026-05-09/ht=07/09e05105d0e030e4de5620be3a294e1fa56dfe9885e7031969092d809743126a.jpg)

![](dt=2026-05-09/ht=07/51bcd51683d66b97c1d5308a77b7b1d7bf4d0a10bff48a9b29a81ca8aa861031.jpg)

![](dt=2026-05-09/ht=07/14eb8b1248eb76ab0b9a80ae3c98d3ac8965a86a4fc788bc65c29450ee0a9bc1.jpg)

![](dt=2026-05-09/ht=07/92cffa1d9a89cfcaf7701adb8b2dad9f471c885b5eb8707e974d8d9a6e299a58.jpg)

![](dt=2026-05-09/ht=07/0b3e54aa59637dcbf982cb18ed339ac3ef6e54793ca91a6cf488c299664032a4.jpg)

To study this effect in different lattices and compare them based on their symmetry, we calculated the dispersion of the leaky modes into the superstrate using RCWA and compared these results with angle-resolved fluorescence measurements from the devices. The devices were illuminated with $\lambda_{ex} = 457\mathrm{nm}$ light and the fluorescence was collected as a function of polar angle $(\theta)$ along the directions of high-symmetry, using a small numerical
aperture (NA) fiber probe. Figs. 3(a), (b), (c) show the comparison between the RCWA calculations and the angle-resolved fluorescence for the different lattices.

In the RCWA simulations, multiple peaks arising from splitting of degenerate TE (longer wavelength) and TM (shorter wavelengths) resonances are seen and strong agreement of the numerically calculated band structure with the spectral and angular locations of the fluorescence extraction peaks is evident in the case of all the lattices. Let us first consider the

103698 - $15.00 USD Received 7 Nov 2008; revised 11 Dec 2008; accepted 11 Dec 2008; published 15 Dec 2008

(C) 2008 OSA

22 December 2008 / Vol. 16, No. 26 / OPTICS EXPRESS 21631

case of the linear lattice, Fig 3(a). For the given parameters, RCWA simulations predict the device produces a leaky TE resonance with a quality (Q) factor of $\sim 290$ at $620\mathrm{nm}$ (experimentally, we observe a Q-factor of $\sim 210$ ). The device also produces a TM resonance with similar Q-factor.

Although the TE mode of the device spectrally overlaps the QD emission (ranging from $600 - 650\mathrm{nm}$ ) for a small range of angles about normal incidence, it begins to lose this overlap for larger in-plane wave-vectors and the extraction efficiency from these resonances declines sharply. However, we also observe that for larger wave-vectors, the TM mode begins to overlap the emission range, and thus extraction of the emitted fluorescence can continue for an angular range of $\pm 15^{\circ}$ about the device normal.

An important consideration in the case of the linear lattice case is the lack of periodicity in the $y$ direction. This would imply that light emitted with non-zero wave-vectors in the $y$ direction might suffer from lowered extraction efficiency due to the lack of leaky modes with propagation constants along this direction, in contrast with structures possessing higher order symmetry. The comparison between calculated band structures and experimentally determined fluorescence for the square (Q $\sim 260$ ) and hexagonal (Q $\sim 310$ ) lattices is shown in Fig. 3(b), 3(c) respectively.

Higher peak intensity for the more symmetric lattices is seen at normal incidence compared to the linear lattice. The explanation for this may be found in the fact that the normally coupled leaky modes (leaky modes at the $\Gamma$ -point) are degenerate and coupled to an increasing number of diffraction planes (1, 2 and 3) based on the symmetry of the lattice[22] (linear, square, hexagonal). Thus, in this special case, an increasing number of leaky modes couple their energy into the same channel that results in emission of fluorescence at normal incidence.

Another interesting observation we make is that the extraction intensities for the square and hexagonal lattice structures is highest at normal incidence and reduced for bands outside this range, in comparison to the linear lattice case, where the enhancement follows uniformly for the bands overlapping the fluorescence.

We found that the coupling efficiency (defined as the degree of overlap of the resonant mode intensity with the layer containing the QDs) was higher for large in-plane wave-vectors in the case of the linear lattice than the square or hexagonal lattice resulting in high extraction efficiency at greater angles. This suggests that the choice of lattice symmetry for maximal extraction might not be immediately obvious and might depend on the application and collection mechanism of the extracted radiation.

# 3.3 Angle-integrated fluorescence measurements

From the perspective of biosensors, we believe that the requirement is to maximize the collection of emitted radiation that is coupled to the PC (spectrally and spatially matched to the fluorophore of interest) and reduce the collection of uncoupled radiation (which might comprise of background fluorescence that is neither spectrally or spatially matched), in order to maximize the SNR.

To this end, one can control both the bandwidth of collection (to concentrate on the wavelength range of interest) and the angular range of collection (to concentrate on the angular range in which the PC extracts the wavelengths of interest). Defining a 'detection window' as the wavelength and angular range over which radiation is collected, we note from Fig. 3 that within this window, the radiation coupled to photonic modes predominantly defines the resonantly extracted signal and the remaining uncoupled radiation within this window is comprised of signal and noise.

It becomes obvious that increasing SNR enhancement is achieved as the size of the window is reduced to a point, i.e. a single wavelength at normal incidence where a majority of the radiation collected is resonant. Reducing the dimensions of this window also, however, reduces the total amount of light collected, which might require extremely sensitive detection mechanisms and/or result in significant contribution of electronic and system noise to the measurement, negating the advantageous effects of the PC.

103698 - $15.00 USD Received 7 Nov 2008; revised 11 Dec 2008; accepted 11 Dec 2008; published 15 Dec 2008

(C) 2008 OSA

22 December 2008 / Vol. 16, No. 26 / OPTICS EXPRESS 21632

![](dt=2026-05-09/ht=07/c7f535023baf294180aff9178d5f1d00311f3079842b3f6728c49c3cd3d9d510.jpg)

![](dt=2026-05-09/ht=07/a89d4fd6b773cd251700e6bd2b6abc89e098bab52befab82570bd1e183698d11.jpg)

![](dt=2026-05-09/ht=07/2661d4985857f557377840d4d4e2205f55aa4ff5b587057482592f5af9b7b42a.jpg)

![](dt=2026-05-09/ht=07/c95c4444e282a66b8324efac4e175d97ab7b95d27b22c32f96fa2f6b0bebfbdd.jpg)

![](dt=2026-05-09/ht=07/094e2f0d21350c2917a2cae3e4f73a682a0829c3e3bac14a7440717e04ab5ad0.jpg)

![](dt=2026-05-09/ht=07/e3f79af6bf7861cf9d7bea4e421d487cdbfc0e03d7c5d8cc7e05174f08eeeade.jpg)

In order to estimate effectiveness of different lattices based on the size of this detection window, we measured the total emitted fluorescence from the QDs at normal incidence, by collecting the radiation over different angular and wavelength ranges. Collection half angles of $\sim 0.05^{\circ}$ , $\sim 8.6^{\circ}$ and $\sim 14.5^{\circ}$ were considered using a fiber probe, 0.15 NA and 0.25 NA objective lenses respectively. The total intensity collected from each device and an

103698 - $15.00 USD Received 7 Nov 2008; revised 11 Dec 2008; accepted 11 Dec 2008; published 15 Dec 2008

(C) 2008 OSA

22 December 2008 / Vol. 16, No. 26 / OPTICS EXPRESS 21633

unpatterned reference sample for different collection angles is shown in Fig. 4. Fig. 4a) shows the comparison between different lattices and the reference sample for radiation collected over a half angle of $\sim 0.05^{\circ}$ .

For the linear, square and hexagonal lattices, the intensity of emission at the resonance peak when compared to the maximum emission intensity from the reference sample shows an enhancement of $\sim 120$ , $\sim 202$ and $\sim 220$ times respectively in comparison to the broad featureless reference curve (black). Significant enhancement of the fluorescence is seen outside the resonance peaks owing to the resonance sidebands along with peaks at lower wavelengths arising from the coupling of the fluorescence to TM leaky modes.

The relative enhancement of fluorescence collected over various bandwidths is shown Fig. 4(b). It is clear that for small detection bandwidths (5nm, $10\mathrm{nm}$ ) centered about the emission maximum, the hexagonal lattice performs best, as is expected from the h
igh enhancement achieved at normal incidence due to radiation extracted from 3 diffraction planes. For the same reason, it is observed that the linear lattice produces the least enhancement.

However as the bandwidth of detection increases, the enhancement by the square lattice dominates, due to the high enhancement produced for TM modes ( $\sim 580\mathrm{nm}$ ). Fig. 4(c) shows the fluorescence collected using the 0.15 NA objective lens, collecting a half-angle of $\sim 8.6^{\circ}$ . It is seen that in comparison to the unpatterned reference, the enhancement factor is reduced. This occurs primarily due to the fact that a larger number of angles containing radiation that is non-resonant with the structure are collected.

This also results in the broadening and smearing out of the photonic features in the radiation that is collected from the PCs. In Fig. 4(d) the inefficient coupling efficiency between the QD emission and large wave-vector leaky modes in the hexagonal lattice becomes evident in the lowered extraction efficiency, in comparison to the linear and square lattices that perform better in this angular range. We also observe that the enhancement factor as a function of detection bandwidth is much less sensitive in this case due to the broadened resonance features. Fig.

4(e) shows the radiation as collected from the PCs using a 0.25 NA objective, collecting radiation over an angular range of $\sim 14.5^{\circ}$ . Due to an even larger range of integrated angles, the photonic features almost completely disappear.

The enhancement factor is further reduced, with the peak enhancement factor being the highest for the linear lattice, closely followed by the square lattice: The linear lattice performs best when a large range of angles is collected because of the high efficiency extraction via the TM modes up to 15 degrees, an effect that weakens with increasing symmetry. As seen in Fig. 4(f), the dependence of the enhancement factor on the wavelength of collection is greatly reduced due to the broad fluorescence peaks.

Angle-integrated measurements thus suggest that the choice of PC symmetry for maximal enhanced extraction depends on the detection window size. Highest enhancement is obtained from the hexagonal lattice for a very narrow detection window and for increasing size of the detection window, the variation between different lattices in terms of enhancement factor is reduced. For the largest detection window considered, the linear lattice performs best followed closely by the square lattice, the basis for which can be explained by inspection of the angle-resolved fluorescence measurements.

It is thus our conclusion that for most fluorescence applications, a simple linear lattice might be attractive from the standpoint of ease of simulation and fabrication and characterization.

In the weak coupling regime, the modification of the dynamics of spontaneous emission via the Purcell effect $^{23}$ is well studied for a wide range of resonator-emitter $^{24,25,26}$ systems. Essentially, in the presence of enhanced photonic mode density, the lifetime of emission from fluorescent species can be reduced, which can result in enhanced output power due to dominance of radiative pathways and a concomitant increase in quantum efficiency $^{27}$ .

Since the system under study operates in the weak coupling regime, we considered the possibility that the peaks in the fluorescence emission spectrum of the QDs might be compounded by a similar enhancement effect. We performed fluorescence lifetime studies (see supplementary information) and found an insignificant change in lifetimes for QDs under different coupling conditions which seems to indicate the absence of another form of enhancement in the fluorescence output from the QDs.

103698 - $15.00 USD Received 7 Nov 2008; revised 11 Dec 2008; accepted 11 Dec 2008; published 15 Dec 2008

(C) 2008 OSA

22 December 2008 / Vol. 16, No. 26 / OPTICS EXPRESS 21634

# 3.4 Detection of TNF- $\alpha$

In order to demonstrate the performance of the PC device as a tool for enhancement of fluorescence in life science applications, we performed a sandwich immunoassay for the detection of a cytokine on a PC fabricated with a linear lattice (details in supplementary information).

![](dt=2026-05-09/ht=07/78771a0640c42f672d8aadd90ec4ac723ce5715c85619d294550772de4c674a7.jpg)

![](dt=2026-05-09/ht=07/24d11741534b43f8bc29a68dd9da4c151f0aca1dec9638ef79220a547de35339.jpg)

Tumor Necrosis Factor- $\alpha$ (TNF- $\alpha$ ) is a cytokine that plays a prominent role in cell signaling during inflammation. Because local inflammation is implicated in the progression of cardiovascular disease and systemic inflammation during sepsis is often deadly, accurate quantification of TNF- $\alpha$ as well as other cytokines may serve as useful diagnostic and prognostic indicators. TNF- $\alpha$ (1 ng/ml) was attached to the PC surface and to a reference surface via spots of capture antibody linked to the surfaces by a silane surface chemistry. TNF- $\alpha$ was detected by adding a detection antibody conjugated to the organic fluorophore Cyanine-5 (Cy-5) whose emission maximum ( $\lambda_{\mathrm{cy - 5}}$ ) occurs at $\sim 690~\mathrm{nm}$ (see methods).

103698 - $15.00 USD Received 7 Nov 2008; revised 11 Dec 2008; accepted 11 Dec 2008; published 15 Dec 2008

(C) 2008 OSA

22 December 2008 / Vol. 16, No. 26 / OPTICS EXPRESS 21635

The fluorescence was collected using a customized fluorescence microscope that allowed illumination of the sample with a $632.8\mathrm{nm}$ HeNe laser and collection of the emitted radiation using a 2x (0.06 NA) objective and a $10\mathrm{~nm}$ bandpass filter centered at $\lambda_{\mathrm{cy-5}}$ . Imaging was performed by electron-multiplying CCD (ImageEM, Hamamatsu). The fluorescence output from the spots on the PC surface was compared to spots off the PC surface that served as a reference, and the ratio of background-subtracted intensities was defined as the enhancement factor.

Due to the small size of the PC area ( $1\mathrm{mm}^2$ ) we were able to compare only 4 spots on the device to those off the device. Fig. 5 shows the acquired fluorescence image. The red circles represent the area where the antibody-protein complex is present and where fluorescence is to be expected. Bright fluorescence is seen from the spots on the PC region, whereas the spots outside the PC region are hardly visible above the background noise. A line profile of the spots in the first row is plotted below the image and shows the variation of fluorescence intensity along the spots.

Some edge effects are seen for spots at the PC boundaries (not included in comparisons) along with spreading of the spots but the enhancement factor for the spots on the crystal versus those off is calculated to be $\sim 22\mathrm{x}$ , demonstrating a significant increase in detection sensitivity on the PC surface. Further studies will be performed on large arrays of PCs to determine the limit of detection offered by the PCs in comparison to unstructured surfaces.

In conclusion, we have demonstrated using a model system the nuances of the enhanced extraction phenomenon as provided by PCs, in the context of a fluorescence biosensor and showed that large gains in sensitivity can be achieved depending on the detection modality. We also demonstrated for the first time, the enhancement in fluorescence output using the enhanced extraction effect, with $\sim 22\mathrm{x}$ enhancement intensity of fluorescence in the detection of the cytokine TNF- $\alpha$ . We believe the results of this work can be extended to a wide range of biological systems that might benefit from the added sensitivity afforded by this approach.

# 4. Methods

# 4.1 Fabrication

The PCs in this study were fabricated by electron-
beam lithography (EBL). Across all the structures, thickness of the waveguide and grating layers were kept constant at $t_{wg} = 150$ nm and $t_g = 170$ nm respectively. The period and duty cycle of the linear, square and hexagonal lattices were $\varLambda_{lin} = 340\mathrm{nm}$ and $50\%$ , $\varLambda_{sq} = 340\mathrm{nm}$ and $50\%$ , $\varLambda_{hex} = 395\mathrm{nm}$ and $67\%$ .

The waveguide layers were deposited using electron-beam evaporation (Infinity 22, Denton Vacuum) in an oxygen environment to ensure low loss films and the grating layer was exposed using EBL (JBS6000-FS, Jeol) in poly (methyl methacrylate) (PMMA A4, Microchem) electron beam resist, into which QDs (CdSe/ZnS, Evident) had been incorporated. Refractive index measurements of the PMMA films before and after the addition of QDs were made and showed no significant difference, establishing the fact that the filling fraction of the QDs in the PMMA was low.

# 4.2 TNF- $\alpha$ detection protocol

The fabricated device was washed with acetone, isopropanol, and deionized water and then treated for 5 minutes in oxygen plasma to thoroughly clean the surface. The device was then immersed in a solution of $97.8\%$ toluene, $2\%$ (3-Glycidoxypropyl)trimethoxysilane, and $0.2\%$ triethylamine inside a nitrogen-purged glovebox for 1.5 hours (all chemicals purchased from Sigma-Aldrich). After washing with toluene and ethanol, the functionalized device was exposed to a high intensity UV lamp for 45 seconds and stored at room temperature for 24 hours.

Anti-Tumor Necrosis Factor- $\alpha$ (TNF- $\alpha$ ) capture antibody (Mab1, BioLegend) in a Phosphate Buffered Saline (PBS) solution with $0.5\%$ trehalose was spotted onto the device using a Perkin-Elmer Piezorray. The capture antibody was incubated at $4^{\circ}\mathrm{C}$ overnight. The remaining reactive epoxysilane groups were blocked by applying a solution of $1\mathrm{mg / ml}$ casein in PBS to the device for 1 hour at room temperature. After washing with $0.05\%$ Tween in PBS (PBS-T), TNF- $\alpha$ (BioLegend) was added at a concentration of $1\mathrm{ng / ml}$ in casein-PBS

103698 - $15.00 USD Received 7 Nov 2008; revised 11 Dec 2008; accepted 11 Dec 2008; published 15 Dec 2008

(C) 2008 OSA

22 December 2008 / Vol. 16, No. 26 / OPTICS EXPRESS 21636

and incubated for 2 hours at room temperature. Biotinylated Anti-TNF- $\alpha$ detection antibody (biotin-Mab-11, BioLegend) was added at a concentration of $10~\mu \mathrm{g / ml}$ in casein-PBS for 1 hour incubation after a PBS-T wash step. Cyanine-5 conjugated streptavidin (GE Healthcare) was added at a concentration of $10~\mu \mathrm{g / ml}$ in PBS-T and incubated for 30 minutes. After a final PBS-T wash, the device was dried under a stream of nitrogen and scanned in the fluorescence setup.

# Supplementary information

# S1.1 Extraction effect using the TM mode

The period of the lattices was increased to until the lower wavelength TM mode resonance overlapped with the fluorescence emission maximum of the QDs. For the linear lattice, the period was increased to $\varLambda_{lin}=370\mathrm{nm}$ . Fig. S 1 (blue curve) shows the transmission spectrum for TM-polarized white light through the device when illuminated at normal incidence, with the resonance occurring at $\sim620\mathrm{nm}$ , as targeted. Fig.

S 1 (black curve) shows the fluorescence collected at normal incidence, once again showing exact overlap with the predicted resonance, similar line-widths and enhancement factor as the TE resonances. One aspect to note however is the larger separation of the two band-edges in the TM case in contrast to the TE case indicating a larger band-gap. We also note that the TE-polarized modes are red shifted for this device ( $\sim660\mathrm{nm}$ ) and weakly extract the fluorescence due to the inefficient spectral overlap with the QD emission spectrum.

![](dt=2026-05-09/ht=07/4ee2e0dcba8ec2793ab3ac41fbde139831f275b7a4c9e2de4f68b7737da81ba3.jpg)

103698 - $15.00 USD Received 7 Nov 2008; revised 11 Dec 2008; accepted 11 Dec 2008; published 15 Dec 2008

(C) 2008 OSA

22 December 2008 / Vol. 16, No. 26 / OPTICS EXPRESS 21637

# S1.2 Extraction of radiation from the backside of the device

The data shown in this study, as mentioned before, is for radiation collected from the topside (superstrate) of the device. We also measured the fluorescence output from the backside (substrate) of the device in verify that the extraction effect occurs in the substrate region as well, as predicted by Eq. 2. Fig. S 2 shows the radiation collected from the topside of the device (red curve) compared to the radiation collected from the backside (black curve) at normal incidence. The peaks in the fluorescence occur at the same spectral location as predicted by Eq. 2. The peak intensity of fluorescence collected from the backside seems to be lower than that collected from the topside, an effect that is not clearly understood as yet.

![](dt=2026-05-09/ht=07/30cf573135629abcb60599d2249d774c28f127dd82d0333d516d77f6f1ac4dc3.jpg)

# S1.3 Fluorescence lifetime measurements

As mentioned before, it is well known that in the presence of enhanced photonic mode density, the lifetime of emission from fluorescent species can be reduced, which can result in enhanced output power due to dominance of radiative pathways and a concomitant increase in quantum efficiency. Since the system under study operates in the weak coupling regime, it is of interest to determine if the peaks in the fluorescence emission spectrum of the QDs result from a similar enhancement effect.

However, the requirements for noticeable Purcell enhancement include a small mode volume and high resonance Q-factor, both of which seem to be conditions that are not met by the PC. Since the resonances in these devices are degenerate over the whole structure, the mode volume essentially approaches the size of the device itself. Also, the leaky nature of the resonances places a practical limit on the Q-factors of these devices. In order to investigate we performed time-resolved photoluminescence measurements.

A frequency-doubled Ti:Sapphire laser operating at $400\mathrm{nm}$ was used to excite the QD impregnated PMMA film. The pulsed laser was incident such that it was nonresonant with the photonic crystal. QD emission normal to the device surface was first

103698 - $15.00 USD Received 7 Nov 2008; revised 11 Dec 2008; accepted 11 Dec 2008; published 15 Dec 2008

(C) 2008 OSA

22 December 2008 / Vol. 16, No. 26 / OPTICS EXPRESS 21638

measured as a function of emission wavelength using a 0.03 NA collection objective. Time resolution for the measurement setup is $\sim 80$ ps with an observable time window of $12.5\mathrm{ns}$ , dictated by the $80\mathrm{MHz}$ repetition rate of the pulsed laser. Figure S 3(a) shows a power sweep of the fluorescence emission from two linear lattices providing leaky resonances at normal incidence (red and green curves) and a reference sample (black curve) that was fabricated exactly as the other devices with the exception that there was no grating, and hence leaky resonances present.

Of the two linear lattices, one overlaps the QD emission at $\sim 616\mathrm{nm}$ and the other at $\sim 633\mathrm{nm}$ (the longer wavelength resonance was obtained by increasing the period of the grating). Measuring the lifetime at $616\mathrm{nm}$ allows us to ascertain the effect of the device on the life time at resonance, off resonance and completely decoupled to the leaky modes. Results of the measurements are shown in Fig. S 3(b).

The red and green curves that show the decay of the fluorescence on and off resonance clearly show there is no significant change in the lifetime of fluorescence from the QDs when coupled to the leaky modes. Comparison with QDs on the refere
nce device (no leaky modes present) shows a very slight reduction in lifetime which is not clearly understood but is thought to arise from the slightly different environment surrounding the QDs.

Nevertheless, the insignificant change in lifetimes for QDs under different coupling conditions seems to indicate the absence of another form of enhancement in the fluorescence output from the QDs.

![](dt=2026-05-09/ht=07/9a8ec01ace73f50e1e4ffd071252214d0740e260c8270ae4fe5908967de3ef34.jpg)

![](dt=2026-05-09/ht=07/cd95de6271bac62477a31cbfc32387af8e623a7d927ebef737c63d9ce8d903fd.jpg)

103698 - $15.00 USD Received 7 Nov 2008; revised 11 Dec 2008; accepted 11 Dec 2008; published 15 Dec 2008

(C) 2008 OSA

22 December 2008 / Vol. 16, No. 26 / OPTICS EXPRESS 21639

# S1.4 Fabrication of enhanced extraction biosensor for TNF- $\alpha$ assay

The device used in this study was fabricated in a manner similar to the devices fabricated for the experiments outlined in the manuscript (see methods) with the exception that the grating was structured into the $\mathrm{TiO_2}$ layer and quantum dot impregnated PMMA grating was absent. The period of the structure was $\Lambda = 435\mathrm{nm}$ and the duty cycle was $50\%$ . The thickness of the $\mathrm{TiO_2}$ layer was $\mathrm{t} = 120\mathrm{nm}$ and the height of the grating was $\mathrm{h} = 30\mathrm{nm}$ .

The device was designed such that the normal incidence resonance for the TE-mode overlapped with the emission maximum of the dye used in the detection of TNF- $\alpha$ (Cy-5, $\lambda_{\mathrm{cy - 5}}\sim 690~\mathrm{nm}$ ). The predicted quality factor for this mode was $\sim 280$ .

![](dt=2026-05-09/ht=07/9ceddf213b0882fef5c4f2260adbeb06eefbd778141f2a67746403cca7cd754c.jpg)

# Acknowledgments

This work was supported by SRU Biosystems and the National Science Foundation (CBET 07-54122). Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation. The authors would like to thank the staff of the Micro and Nanotechnology Laboratory and colleagues from the Nano Sensors Group for their suggestions and input.

Correspondence and requests for materials should be addressed to BTC.

103698 - $15.00 USD Received 7 Nov 2008; revised 11 Dec 2008; accepted 11 Dec 2008; published 15 Dec 2008

(C) 2008 OSA

22 December 2008 / Vol. 16, No. 26 / OPTICS EXPRESS 21640