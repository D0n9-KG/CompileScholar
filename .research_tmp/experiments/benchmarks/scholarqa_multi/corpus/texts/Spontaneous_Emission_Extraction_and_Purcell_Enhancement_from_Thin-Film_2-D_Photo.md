# Spontaneous Emission Extraction and Purcell Enhancement from Thin-Film 2-D Photonic Crystals

Misha Boroditsky, Rutger Vrijen, Thomas F. Krauss, Roberto Coccioli, Raj Bhat, and Eli Yablonovitch, Fellow, IEEE

Abstract—Electromagnetic band structure can produce either an enhancement or a suppression of spontaneous emission from two-dimensional (2-D) photonic crystal thin films. We believe that such effects might be important for light emitting diodes. Our experiments were based on thin-film InGaAs/InP 2-D photonic crystals at ambient temperature, but the concepts would apply equally to InGaN thin films, for example. We show that the magnitude of Purcell enhancement factor, $F_{p} \sim 2$ , for spatially extended band modes, is similar to that for a tiny mode in a three-dimensional (3-D) nanocavity. Nonetheless, light extraction enhancement that arises from Zone folding or Bragg scattering of the photonic bands is probably the more important effect, and an external quantum efficiency $>50\%$ is possible. Angle resolved photoluminescence from inside the photonic crystal gives a direct spectral readout of the internal 2-D photonic band dispersion. The tradeoffs for employing various photonic crystal structures in high efficiency light-emitting diodes are analyzed.

Index Terms— Photonic crystals, Purcell effect, spontaneous emission.

# I. INTRODUCTION

FOR a long time, spontaneous emission rates were believed to be an intrinsic property of a material, and that such light could be manipulated only after the photons were emitted. It was later understood that spontaneous emission also depends strongly on the surrounding environment through the density of states and the local strength of the electromagnetic modes [1]. We expect this to have valuable implications for the design of light-emitting diodes (LED's).

It was predicted by Purcell [2] that an atom in a wavelength-size cavity can radiate much faster than in the free space. The first work on enhancement and suppression of spontaneous emission was performed by Drexhage [3] and Kleppner [4], in the 1980's at microwave frequencies. Their measurements were performed in an atomic vapor. A similar effect can be observed in semiconductor materials. A five-fold enhancement of the spontaneous emission rate in semiconductors was recently observed by Gerard [5] at low temperatures in vertical cavity surface emitting laser type structures of small lateral dimensions.

Manuscript received August 19, 1999.   
M. Boroditsky, R. Vrijen, R. Coccioli, and E. Yablonovitch are with the Electrical Engineering Department, University of California at Los Angeles (UCLA), Los Angeles, CA 90095 USA.   
T. F. Krauss is with the Electrical Engineering Department, Glasgow University, Glasgow G12 8LT U.K.   
R. Bhat is with Corning Inc., Corning NY 14831 USA.   
Publisher Item Identifier S 0733-8724(99)08945-8.

Photonic crystals are artificially created, multidimensionally periodic structures, that are known for a forbidden electromagnetic bandgap. Such a gap is a range of frequencies in which no electromagnetic modes can exist. The following spontaneous emission rate can be given by Fermi's Golden Rule:

$$
\Gamma (\mathbf {r}) = \frac {2 \pi}{\hbar} \left\langle (\mathbf {d} \cdot \mathbf {E} (\mathbf {r})) \right\rangle^ {2} \rho (\omega) (1)
$$

where d is the electric dipole of the transition, $\mathbf{E}(\mathbf{r})$ is the local zero-point rms electric field, and $\rho(\omega)$ is the density of electromagnetic modes. In a photonic bandgap, the electromagnetic density of states is zero, inhibiting spontaneous emission [6]. In photonic crystals the dipole matrix element, and the density of states, can be engineered to enhance spontaneous emission as well.

As an example of enhancement, Fermi's Golden Rule is used to derive the spectrally integrated Purcell enhancement ratio $F_{p} \equiv \Gamma / \Gamma_{o} \equiv \tau_{o} / \tau$ , of the quantum well spontaneous emission rate in a small cavity relative to a bulk semiconductor, (in Appendix A):

$$
\frac {\Gamma}{\Gamma_ {o}} = \frac {3 Q _ {m} g (\lambda / 2 n) ^ {3}}{2 \pi V _ {\mathrm{eff}}} \tag {2}
$$

where $\Gamma$ and $\tau$ are the spontaneous emission rate and lifetime, respectively, $Q_{m}$ is the material quality factor, $V_{eff}$ is the nanocavity mode volume, n is the semiconductor refractive index, and g is the electromagnetic mode degeneracy. The volume dependence in (2) can be understood in terms of a desired concentration of zero point electric field into the smallest possible mode volume. Later in this paper we will show that enhancement can occur in a band mode with a large mode volume $V_{eff}$ occupying the whole semiconductor film, since the mode degeneracy g would also be correspondingly large.

While these quantum electrodynamic effects, represented by (1) and (2) are fascinating, the world of LED design is dominated by questions of light extraction. Most light generated inside a semiconductor tends to be trapped internally by total internal reflection. Internally, there is a serious risk of parasitic absorption, and most of that light is lost. The solid angle for light escape from a semiconductor surface is only $(1/4n^{2}) \times 4\pi$ steradians, representing only $\approx2\%$ efficiency in many cases. This problem is often resolved by introducing surface roughness [7] on the LED, allowing random scattering

![](images/14776cc478b4035bf939c9106628d115780833afcbd04b96afbd53de3056a7a3.jpg)

<details>
<summary>text_image</summary>

7053 x 284 In. membrane on
glass
26/6/58 TFK
:13031 32kV x40.5k"752m
</details>

(a)

![](images/faba1232812340c6630d286231df6441d762243338bee27eea5e217621fc6ffa.jpg)

<details>
<summary>text_image</summary>

triangular lattice of holes
dielectric slab
z
y
t
x
a
r
glass substrate
</details>

(b)   
Fig. 1. (a) A scanning electron micrograph of a triangular array of voids in a thin InGaAs/InP double hetero-structure film bonded to a glass substrate. Under the voids is the flat surface of the glass. The center to center spacing $a = 720 \, \mathrm{nm}$ , the film thickness $t = 240 \, \mathrm{nm}$ , and the void diameter is $2r = 550 \, \mathrm{nm}$ . (b) A perspective drawing of the experimental structure.

and disrupting light trapping, so that eventually over 30% of the light can escape [8] from the semiconductor surface.

In this paper, we will be introducing periodic photonic crystal surface patterning rather than random roughness. Periodic patterned semiconductor films prevent light trapping, just as randomly patterned surfaces do. The reciprocal lattice vectors associated with periodicity will Bragg scatter all internal light into the escape cone near the origin of k-space, permitting all the light to eventually escape.

Thus, we have two mechanisms for improved LED operation associated with photonic crystals:

1) The surface geometry and patterning can create band structure and nanocavities exhibiting Purcell spontaneous emission rate enhancement.   
2) At the same time, the surface patterns allow Bragg scattering and prevent light trapping, permitting the internal spontaneous emission to escape. The effectiveness of these scattering processes, in competition with parasitic absorption, is called extraction efficiency.

The effects 1) and 2) have a rather different character: the Purcell effect 1), is a rate enhancement, while the extraction effect 2), pertains to the overall LED efficiency. In the presence of nonradiative recombination, and parasitic optical absorption, the 1) and 2) effects are less easily distinguished. Much of this paper will be dedicated to photonic crystal designs that accomplish both 1) and 2), and to establish criteria for distinguishing between 1) and 2). In this way, it is hoped that this paper can guide the LED designer to optimize the desired performance. In most cases of room temperature LED's, it appears that the light extraction effect 2), will be the more important.

Unlike three-dimensional (3-D) photonic crystals, the two-dimensional (2-D) type are relatively easy to fabricate [9], making them more interesting for practical applications. Thus, the enhancement and suppression of spontaneous emission in thin-film 2-D photonic crystals at room temperature is the subject of this paper.

The performance of LED's depends on a delicate interplay between radiative and nonradiative recombination as well as light extraction and light absorption rates. Overall, LED efficiency, and modulation speed, can be significantly improved with the help of photonic crystals.

# II. ANGLE AND SPECTRALLY INTEGRATED PHOTOLUMINESCENCE MEASUREMENTS

In the photonic crystal field, calculations of electromagnetic bandgaps for 2-D photonic crystals of infinite length [10] arrived shortly after the initial demonstrations [11] of 3-D photonic bandgaps. The first calculations of a thin 2-D periodic slab were done much later, independently by Russell [12] and by Fan et al. [13]. Such thin-film photonic crystals are relatively easy to make. Vertical confinement is provided by refractive index guiding, while horizontal control comes from the photonic crystal periodicity. In this paper we will be considering 2-D photonic crystals in the form of a thin semiconductor slab on a glass substrate. A perspective view is illustrated in Fig. 1(b), and Fig. 1(a) is a top view electron micrograph.

Thin slab photonic crystals have a bandgap only for TE polarized, guided waves, but they remain nonetheless quite useful. We have computed, [14], [15] the band structure of our thin film 2-D photonic crystals using the finite-difference time-domain (FDTD) algorithm, with results plotted in Fig. 2(a). The 2-D photonic band structure is dominated by the light cone, the shaded region in Fig. 2(a) that represents those modes that can couple to external plane waves in the glass substrate. In the shaded region, free space plane waves are present at all frequencies, and they pass directly through the

![](images/473d6f2a6c38557e243595f8a9df1362e9d39da01c4e29dba90ef670c862f8ce.jpg)  
(a)

![](images/9602f5086b518aa1e3f30ec16cd6d36cb31fcfa715fcda1bd27d3795b7da3377.jpg)

<details>
<summary>line</summary>

| Phase | Frequency (c/a) |
|-------|-----------------|
| (2)   | 0.5             |
| (4)   | 0.5             |
| (4)   | 0.5             |
| (4)   | 0.5             |
| (4)   | 0.5             |
| (4)   | 0.5             |
| (4)   | 0.5             |
| (4)   | 0.5             |
| (4)   | 0.5             |
| (4)   | -               |
| (4)   | -               |
| (4)   | -               |
| (4)   | -               |
| (4)   | -               |
| (4)   | -               |
| (4)   | -               |
| (4)   | -               |
| (4)   | -               |
| (4)   | -               |
| (4)   | -               |
| (4)   | -               |
</details>

(b)   
Fig. 2. (a) The computed dispersion diagram of our 2-D triangular photonic crystal. The solid lines represent the “TE-like” bands, and the long dashed lines represent the “TM-like” bands. The TE bands have a forbidden gap, while the TM bands do not. Nevertheless the most important feature of the band structure is the grey region representing modes that couple out to plane waves in the glass substrate. It is bounded by the light line, representing the speed of light in glass. This shows that there is in fact no forbidden gap at all, since the grey modes above the light line extend all the way down to zero frequency. (b) The dispersion of TE waves in the unpatterned thin semiconductor film. This may be regarded as the “free photon dispersion.” The numbers in parentheses represent the mode degeneracy of the various dispersion branches, including a factor 2 for the TM modes that are not shown. Fig. 2(b) is useful for estimating the thin-film optical density of states in the absence of the periodic photonic crystal voids.

thin film. Thus there is no forbidden band at any frequency above the light cone. The plotted curves, (solid for TE, and dashed for TM) represent band modes that have a relatively high Q for remaining confined in the semiconductor film, but they too eventually leak away as external plane waves.

The four types of waves above the light cone are labeled A, B, C, and D, in Fig. 3. The waves labeled “A” are simply external plane waves that pass from glass directly through the thin film into the air. The waves labeled “B” are guided modes that couple to external plane waves in both air and the glass substrate. Thus the guided modes are inherently leaky. They are often call leaky conduction band modes, since they correspond to the so-called photonic conduction bands, lying at frequencies above the two valence bands that originate at zero frequency at the $\Gamma$ point.

Spontaneous emission is associated with zero-point electromagnetic fields. Thus we can imagine that incoming zero-point plane waves interact with the thin semiconductor film. These can occur at all frequencies, but those waves that spend a long time in the film before continuing on, as in point “B” of Fig. 3, will induce a considerably enhanced spontaneous emission. This idea is made more rigorous in the Appendices to this article, but the net result of the long interaction time with the thin film is that there is far more spontaneous emission into the allowed bands within the light cone, case “B,” than at other light cone frequencies, case “A” in Fig. 3.

![](images/ad7cc9793fb5d8d87a336870251303a19400d7157f07e6b9d795fdd96237d89d.jpg)

<details>
<summary>text_image</summary>

air
film
glass
D
B
air
film
glass
TE
C
TM
K
Γ
A
air
film
glass
</details>

Fig. 3. Cases A, B, C, and D are examples of the different types of modes which can exist in different points on the frequency versus wave-vector graph. Case A represents a free space mode, above the light line, that couples directly through the semiconductor film. Case B represents a frequency and wave-vector at which external plane waves couple to an allowed band in the film. Thus they propagate along the film for some distance before leaking out again. Case C represents a mode far from the center of k-space.

The “C” and “D” waves are, respectively, similar to the “A” and “B” waves in Fig. 3, but they have lateral wave vectors that are too large to emit into the air, so they produce spontaneous emission only into the glass, whose refractive index defines the shaded light cones. Case “C” consists of plane waves in glass that enter the semiconductor film, and then reflect back from the air interface by total internal reflection. Case “D” is a guided mode that couples to incoming and outgoing plane waves in the glass, but the guided mode does not leak into the air due to the large wave vector.

Spontaneous emission at “B” and “D” is very strong, while at “A” and “C” it is unenhanced. By rescaling our photonic crystal lattice constant, we can tune the leaky conduction band modes, case “B,” to overlap the material emission band, producing enhanced spontaneous emission. On the other hand, if the leaky conduction bands are made to fall above the material emission band, case “A,” then there will be the normal volume of spontaneous emission into external plane waves. Thus the external spontaneous emission can be much stronger than usual, if leaky conduction bands can take part. This pertains to the rate of spontaneous emission, that influences the external efficiency in competition with nonradiative recombination, or other loss mechanisms.

To test these ideas, we use an $In_{0.47}Ga_{0.53}As/InP$ single quantum well double hetero-structure for these photoluminescence experiments, (layer thicknesses are given in Table I), grown by metallo-organic chemical vapor deposition. The separated thin films are fabricated by a chemically selective substrate removal technique, and bonded to a glass slide with

TABLE I
SAMPLE HETEROSTRUCTURE LAYERS 

<table><tr><td>InP top cladding layer</td><td>90nm</td></tr><tr><td> $In_{0.47}Ga_{0.53}As, n=10^{18}cm^{-3}$ , active region</td><td>60nm</td></tr><tr><td>InP bottom cladding layer</td><td>90nm</td></tr><tr><td>Glass slide</td><td></td></tr></table>

![](images/060a1b841d8a6996c7fe63913c497caa1c559efb3c1d25451c17e7ee3dceae25.jpg)

<details>
<summary>text_image</summary>

Glass hemisphere
θ
0°
30°
60°
90°
Photonic crystal
Pump beam
</details>

Fig. 4. A glass hemisphere is bonded to the glass substrate of the photonic crystal. This allows the spontaneous emission to avoid total internal reflection inside the glass, and permits an accurate measurement of the angle dependent spontaneous emission. The spectral peak position versus angle directly reads out the photon band structure.

Norland 73 ultraviolet-curable optical glue. A triangular array of voids is defined by electron-beam lithography, using a LEICA EBPG-5 Beamwriter. The semiconductor slab was etched through by reactive ion etching (RIE) using SiCl $_4$ at an elevated temperature of 200 °C. The InP/InGaAs/InP double heterostructure film thickness is 240 nm, containing an InGaAs quantum well active region, of thickness 60 nm.

Each sample had numerous test areas, called “fields,” each with a triangular lattice structure. The lattice constant of different photonic crystal fields spanned a tuning range sufficient for the leaky conduction band modes to overlap with the material spontaneous emission band or to fall below that band. Since the InGaAs emission wavelength is centered at $\lambda = 1650$ nm, or 182 THz, the photonic conduction band edge was tuned approximately from 165 to 240 THz. This was accomplished by tuning the photonic crystal’s center-to-center lattice constant from a = 910 nm to a = 550 nm. Fig. 1(a) is a SEM picture of a typical structure with a = 720 nm and r/a = 0.38, where r is the radius of the voids. Since the refractive index ratio is smallest at the semiconductor glass interface, more light leaks into the glass substrate than into the air on the other side. To assist the collection of emitted light that would otherwise be trapped in the glass substrate, the sample was cemented to the center of a glass hemisphere, as in Fig. 4, utilizing an optically transparent adhesive.

Prior to the photoluminescence measurements on the thin-film photonic crystals, we calibrated the internal efficiency of an unprocessed sample resting on the original InP-growth substrate, by referencing photoluminescence to elastic light scattering [16] from a $99\%$ white surface. (We used Spectralon from Thorlabs.) The collection solid angle, and pump source for elastically scattered light, were the same as for luminescence. Thus the only corrections to be taken into account were for the detector quantum efficiency ratio at the pump and luminescence wavelengths, the Fresnel surface transmission at the semiconductor/air interface, and the escape cone at the top surface. This calibration showed that the unprocessed InGaAs double hetero-structures used in these experiments have internal quantum efficiency near $100\%$ . This level of material quality is common for the mature InGaAs/InP material system.

The internal efficiency in a thin-film can be converted into an external efficiency for a given light extraction configuration. When the thin film samples are bonded to a glass hemisphere, a 100% internally efficient, unpatterned, thin film, will have a single-sided external efficiency of $1/(4(n_{f}/n_{g})^{2}) \approx 0.055$ , where $n_{f} = 3.2$ and $n_{g} = 1.5$ are the refractive indices of the semiconductor film and glass, respectively. By using a glass hemisphere on both sides of the photonic crystal, the photon extraction efficiency would be doubled to $2 \times (1/4(n_{f}/n_{g})^{2}) \approx 0.11$ . That 11% extraction efficiency is actually achieved even with only a single hemisphere, since totally internally reflected light at the bottom air interface readily escapes from the top glass hemisphere. Our goal is to use this 11% calibration level to set the external efficiency scale for the photonic crystal samples, which had the same single hemisphere extraction configuration as in Fig. 4.

There are some further minor corrections. In Fig. 2(b) the calculated phase velocity for guided modes in the unpatterned ultra-thin films, implies an effective refractive index slightly smaller than the $n_f = 3.2$ in bulk material. Also, the high internal efficiency of the InGaAs active layer permits some photon recycling to boost the observed unpatterned thin-film external efficiency even higher than $11\%$ , via photon recycling. We conservatively estimate that the unpatterned thin film, on our glass hemisphere has at least $12\%$ external efficiency. Once we know the external quantum efficiency of the unpatterned sample, we can use that as a reference for external efficiency measurements on thin film photonic crystal samples.

Angle and spectrally integrated photoluminescence from the 2-D photonic crystals, of various lattice constants, was measured at room temperature. A 780 nm AlGaAs pump laser was focused to a 30 $\mu$ m spot. The angle integrated measurements collected light in a solid angle from normal, up to a 45° angle away from normal, within the glass hemisphere. Full solid angle coverage was obtained by confirming that the spectrally integrated angular distribution was approximately Lambertian for all samples, (i.e., it followed a $\cos\theta$ dependence versus polar angle). The pumping conditions were fixed and the results did not depend on orientation of the pump beam with respect to the samples.

The dependence of the photonic crystal photoluminescence signal (referenced to the unpatterned sample in arbitrary units),

![](images/4707c0a49d52e34b9b8a0f52fb378527a7e93832389607700deaf3464a455882.jpg)

<details>
<summary>line</summary>

| Lattice constant, nm | PI signal, a.u. |
| ------------------- | --------------- |
| 600                 | 0.6             |
| 625                 | 0.4             |
| 650                 | 0.3             |
| 675                 | 0.3             |
| 700                 | 0.4             |
| 725                 | 0.6             |
| 750                 | 0.8             |
| 775                 | 1.6             |
| 800                 | 1.9             |
| 825                 | 2.7             |
| 850                 | 2.9             |
| 875                 | 2.9             |
| 900                 | 3.5             |
</details>

(a)

![](images/6430fc615f2124d030ef8565458f4e433ea51713d7c5fd75ef7473270c264363.jpg)

<details>
<summary>line</summary>

| Lattice constant, nm | PL efficiency, % |
| ------------------- | ---------------- |
| 600                 | 12               |
| 650                 | 8                |
| 700                 | 10               |
| 750                 | 12               |
| 800                 | 24               |
| 850                 | 36               |
| 900                 | 48               |
</details>

(b)   
Fig. 5. Angle integrated (from 0 to 45°) and spectrally integrated photoluminescence (PL) signal from the thin-slab triangular photonic crystal as a function of its lattice constant. (a) The vertical scale is in arbitrary units. The horizontal line is the PL signal from an unpatterned portion of the sample, in the same units. This is raw uncalibrated data, not corrected for fractional sample absorption. (b) The same data as in (a), now absolutely calibrated with respect to a white surface, and corrected for the fractional sample absorption.

is plotted versus photonic crystal lattice constant in Fig. 5(a). The unpatterned film is plotted as a horizontal line independent of lattice constant. In Fig. 5(b), the same data is replotted referenced to the calibrated 12% external efficiency of the unpatterned thin film samples. An additional correction was taken in Fig. 5(b) for the void fraction of $\approx50\%$ in the 2-D thin-film photonic crystal. Fig. 5(b) assumes the photonic crystal sample absorbs 50% less pump light than the unpatterned thin-film.

As can be seen from Fig. 5(b), the photoluminescence efficiency from photonic crystal samples with lattice constant $a \approx 900$ nm is actually four times larger than that from the unpatterned sample. According to our band structure computations, leaky conduction band modes, at lattice constant $a \approx 900$ nm, do indeed match the 182 THz InGaAs emission frequency. The calibrated external efficiency for this photonic crystal sample is $48\%$ . Optical pumping notwithstanding, this is a promising result, since state-of-the-art electrically pumped LED's have only $30\%$ quantum efficiency. The $48\%$ efficiency number is even more surprising since the photonic crystal sample has to combat nonradiative surface recombination at exposed surfaces in the voids. Thus the overall effect of the photonic crystal in this case is a competition between the enhancement factors versus nonradiative recombination, and is net favorable for external spontaneous emission.

We need to question now whether the improvements are mostly due to Purcell type enhancement, or due to the evident light extraction improvement associated with the patterned semiconductor film. In the following section we shall discuss the angle integrated efficiency results in Fig. 5, and to compare them with some theoretical models.

# III. MODELS FOR ANGLE AND SPECTRALLY INTEGRATED EFFICIENCY

Light emission from 2-D photonic crystals can benefit from the Purcell effect, but the benefit from improved light extraction efficiency is probably far more important. The two effects are of a different character. The Purcell effect changes the rate of spontaneous emission and is helpful in competition with nonradiative recombination, while light extraction competes with parasitic light absorption to influence the overall efficiency.

Given that the internal quantum efficiency is already close to 100% in good quality materials, there is not much further internal efficiency improvement to be expected from the Purcell effect. Instead the Purcell effect could lead to an increased modulation speed for telecommunications applications, if desired.

The efficiency situation is more complicated in this experiment, since the photonic crystal voids actually expose edges of the active region that suffer from nonradiative surface recombination. Thus the Purcell enhancement would be needed to compensate the increased nonradiative recombination associated with the photonic crystal! Indeed, for that reason, our configuration might be regarded as somewhat self-defeating; the Purcell effect being needed to defeat a problem caused by the photonic crystal. We will show that in photonic crystals in which the voids don't penetrate all the way through the active region, both the Purcell enhancement and surface recombination are probably nearly absent, but the extraction enhancement remains. Thus a photonic crystal film, surface patterned by blind voids might turn out to be optimal for LED efficiency. The extraction benefit would be accompanied by no nonradiative recombination penalty.

The Purcell effect at room temperature is not particularly strong. Recalling (2), there is great advantage in having the highest possible Q, and the smallest possible mode volume. Reference [15] made a thorough investigation of the smallest possible mode volume that could reasonably be expected. It appears [15] that the smallest achievable mode volume in a dielectric cavity is $V_{\mathrm{eff}} \sim 2(\lambda/2n)^{3}$ . Due to spectral averaging, the Q factor that shows up in (2) is the material $Q_{m}$ related to the room temperature spontaneous emission linewidth. Since this amounts to only $Q_{m} \sim 10$ for InGaAs, the available room temperature Purcell Factor is only $\Gamma/\Gamma_{o} \sim 2.5$ . If there is two-fold polarization degeneracy, the nanocavity Purcell factor is

doubled to five, which is still not a large increase in the spontaneous emission rate. While a properly designed 2-D photonic crystal nanocavity can have enough cavity Q to support lasing [17], the effective Q for spontaneous emission, when averaged [15] over the broad material spontaneous emission linewidth at room temperature, is quite modest. The predicted Purcell enhancement factor is unimpressive.

The anticipated magnitude of the Purcell effect can also be determined by another approach, directly from Fermi's Golden Rule, (1), by estimating the matrix element and density of states. In the angle and spectrally integrated photoluminescence measurements, we referenced the 2-D photonic crystal spontaneous emission to that of the unpatterned film from which it was made. Calculations have shown [18] that in a thin symmetric waveguide, similar to our unpatterned thin film on glass, the spontaneous emission rate is about $20\%$ lower than in bulk InP. Therefore the unpatterned thin films form a meaningful reference case, having a spontaneous emission rate within $20\%$ of the bulk semiconductor.

Thus, we should compare the relative density of states for the 2-D patterned, and unpatterned thin films. The unpatterned film has a band structure corresponding to an asymmetric waveguide, whose frequency versus wave vector dispersion is known [19]. Such a waveguide dispersion can be collapsed into a Brillouin Zone of the corresponding 2-D photonic crystal. This may be regarded as a type of free photon dispersion relation, and is shown in Fig. 2(b) for the TE case. The numbers in parentheses represent the overall degeneracy of those branches of the dispersion curves. The curves in Fig. 2(b) have a degeneracy index $\geq 2$ , since there is no periodic perturbation to lift degeneracy; [the number of modes was doubled to account for TM modes that are not shown in Fig. 2(b)]. The mode density should be compared between the patterned film Fig. 2(a), and the unpatterned film Fig. 2(b). It is difficult to come to an exact conclusion, without integrating over the full 2-D Brillouin Zone, but the density of states is probably similar in the two cases.

The matrix element associated with the zero point electric field does indeed appear to be notably influenced by the 2-D photonic crystal. This can be seen in Fig. 6, a calculation of the electromagnetic mode in real space of a typical leaky conduction band mode. Not surprisingly, the electric field tends to be concentrated in the semiconductor backbone, and to be very weak in the void holes. Since the voids represent about half the total area, this means that the zero-point electric energy is about twice as concentrated within the semiconductor backbone, relative to the volume of the unpatterned film.

Giving no credit for increased density of states, and about a factor two credit for the volume concentration of the zero point fluctuations in the semiconductor, the result is a Purcell factor of approximately $F_{p} \equiv \Gamma/\Gamma_{o} \sim 2$ , similar to the case for a nanocavity [15]. The large effective volume of the band modes appears to be almost exactly compensated by the increased effective degeneracy number g, producing a Purcell enhancement factor similar [15] to the case ( $F_{p} \sim 2.5$ ) of a nanocavity.

Thus it appears that the dominant influence for increased spontaneous emission output in photonic crystals is extraction by Bragg scattering into the escape cone. In this respect a periodic structure might be regarded as merely an alternative to a random texture [7], [8] that also scatters light into the escape cone, improving the extraction efficiency. If there were no parasitic optical absorption, then 100% of the spontaneous emission would escape in either case. Nonetheless, a periodic structure does scatter coherently, and that might be a faster mechanism for coupling out light than a random scatterer. Extraction would compete more effectively against ever present parasitic optical absorption.

![](images/60e947b035d61f9d9b2164bf4a90a4da091e461692c2c9455334a6accd6d33bf.jpg)

<details>
<summary>natural_image</summary>

Microscopic view of hexagonal cells arranged in a grid pattern (no text or symbols)
</details>

Fig. 6. The spatial distribution of electric energy for a leaky conduction band mode near point “B” in Fig. 3, for the 2-D thin-film photonic crystal consisting of a triangular array of voids as in Fig. 1. The electric energy tends to concentrate in the dielectric backbone, and avoid the empty voids. In this case the electric energy is concentrated in the triangles between the voids. Thus the zero point electric energy tends to be spatially concentrated compared to the case of an unpatterned thin-film. This spatial or volume concentration leads to a Purcell enhancement.

To build a model for the spectrally and angle-integrated external efficiency, we begin by analyzing the fate of the photons emitted by electron/hole recombination. Some of them are emitted directly into free space modes, as in point A of Fig. 3 at a rate $1 / \tau_{fr}$ . Some are emitted into the leaky conduction band modes, as in point B of Fig. 3 at a rate $1 / \tau_{lm}$ . Some are emitted into trapped modes at a rate $1 / \tau_{tr}$ , under the escape cone, where they can never leak out, and end up being reabsorbed by the semiconductor. We have assigned to each of these processes, a partial electron/hole recombination rate, $1 / \tau_{fr}$ , $1 / \tau_{lm}$ , and $1 / \tau_{tr}$ .

The photons that are emitted can in turn have various destinies, particularly those that are emitted into leaky conduction bands modes. They may leak out of those modes at a rate determined by the modal $Q$ ; the leakage rate per photon is: $2\pi \nu / Q$ , where $\nu$ is the photon frequency. They may be absorbed in the semiconductor at a rate $\alpha(c/n)$ , where $\alpha$ is the band-to-band absorption coefficient per unit centimeter, which becomes an absorption rate per unit time when multiplied by the speed of light $c$ in a medium of refractive index $n$ . If the absorbing layer of semiconductor film has only a thickness $t$ , and the overall film thickness is $d$ , then the lifetime of a photon is $\alpha(c/n)(t/d)$ . This in turn can be augmented by a second absorption mechanism, $\alpha'$ , a totally parasitic absorption, like free carrier absorption that fails to recycle energy into electron/hole pairs. Thus the overall emission rate $1/\tau_{lm}$ of photons into leaky mode can result in a variety of

possible partial emission rates per electron/hole pair

$$
\begin{array}{l} \frac {1}{\tau_ {l m}} \left\{\left(\frac {\frac {2 \pi \nu}{Q}}{\frac {2 \pi \nu}{Q} + \alpha \frac {c}{n} \frac {t}{d} + \alpha^ {\prime} \frac {c}{n} \frac {t}{d}}\right) \right. \\ + \left(\frac {\alpha \frac {c}{n} \frac {t}{d}}{\frac {2 \pi \nu}{Q} + \alpha \frac {c}{n} \frac {t}{d} + \alpha^ {\prime} \frac {c}{n} \frac {t}{d}}\right) \\ \left. + \binom{\alpha^ {\prime} \frac {c}{n} \frac {t}{d}}{\frac {2 \pi \nu}{Q} + \alpha \frac {c}{n} \frac {t}{d} + \alpha^ {\prime} \frac {c}{n} \frac {t}{d}} \right\} \tag {3} \\ \end{array}
$$

where the first parenthesis is the probability $\chi$ of external light emission, the second is the probability Z, of reabsorption into electron/hole pairs, and the third is the probability $Z'$ , for utterly parasitic reabsorption into some other less useful light absorption mechanism. Thus, (3) can be abbreviated

$$
\frac {1}{\tau_ {l m}} = \frac {1}{\tau_ {l m}} \{\chi + Z + Z ^ {\prime} \}. \tag {4}
$$

Likewise there is a similar formula for light that is emitted directly into the escape cone. In that case the internal path length is too short for significant absorption and all the light is externally emitted, albeit at a slower rate $1/\tau_{fr}$ . In that instance $\chi = 1$ , and $Z = Z' = 0$ .

For the case of trapped light under the escape cone, $\chi = 0$ , nothing escapes, and the photons distribute themselves among the two absorption mechanisms, Z and $Z'$ . In our case, we notice that in Fig. 3, none of the light falls below the escape cone. This is largely because of the glass substrate which makes the cone wider than it would be for escape into air.

The external efficiency can now be estimated by examining the partial electron/hole recombination rates, and selecting those partial rates that contribute to external light emission. In our experimental example we need to be concerned with light that is directly externally emitted, and that which is emitted into leaky modes, and then leaks out. However, light that is reabsorbed to produce electron hole pairs, does not directly affect the overall recombination rate. The net electron-hole recombination is made up of the following partial rates:

$$
\begin{array}{l} \frac {1}{\tau_ {f r}} + \frac {1}{\tau_ {l m}} \left\{\chi + Z + Z ^ {\prime} \right\} + \frac {1}{\tau_ {n r}} - \frac {1}{\tau_ {l m}} Z \\ = \frac {1}{\tau_ {f r}} + \frac {1}{\tau_ {l m}} \{\chi + Z ^ {\prime} \} + \frac {1}{\tau_ {n r}} (5) \\ \end{array}
$$

where we have introduced the nonradiative recombination rate $1/\tau_{nr}$ . The photon recycling term containing Z, disappears from (5), since it has no net effect on the electron/hole recombination. The $\alpha$ term remains nevertheless, in the denominators of (3). The external LED efficiency $\eta_{ext}$ can be written as a ratio of partial rates

$$
\eta_ {e x t} = \frac {\frac {1}{\tau_ {f r}} + \frac {1}{\tau_ {l m}} \chi}{\frac {1}{\tau_ {f r}} + \frac {1}{\tau_ {l m}} \left\{\chi + Z ^ {\prime} \right\} + \frac {1}{\tau_ {n r}}}. \tag {6}
$$

Equation (6) is one of the central results of this paper. It applies directly to points B and D in Fig. 3, where spontaneous emission can occur into both leaky modes, as well as directly into external modes. However at points A and C of Fig. 3, the $1 / \tau_{lm}$ term is absent, and the efficiency is usually diminished. Much depends on the nonradiative recombination rate $1 / \tau_{nr}$ . If it were absent, then points A and C would have $100\%$ external efficiency. However, the direct external emission $1 / \tau_{fr}$ is a rather slow process, and it usually competes poorly with nonradiative recombination, except under the most favorable circumstances. At points A and C we observe in Fig. 5 a calibrated external efficiency of only $\approx 5\%$ . This sets the scale for the $\tau_{nr} / \tau_{fr}$ ratio.

The partial recombination rates in Appendix B, can be spectrally integrated to give the rate ratio $\tau_{fr}/\tau_{lm}$ , which represents a substantial enhancement into the band modes. This enhancement is composed of a minor Purcell effect and a substantial extraction improvement by Brillouin Zone folding, essentially Bragg scattering. The formula predicts a spectrally integrated extraction enhancement by about a factor ten, related to the material $Q_{m}$ . (When spectrally resolved this enhancement factor begins to approach the cavity Q which can be as high as 50.) This predicts well the observed external efficiency $\eta_{ext}$ , at points B and D relative to points A and C of Fig. 3. Points B and D appear experimentally at the right edge of Fig. 5 for lattice constants a = 850–900 nm, and have close to $\eta_{ext} = 50\%$ efficiency.

Thus the various efficiency measurements determine the ratios $\tau_{nr}/\tau_{fr}$ , and $\tau_{fr}/\tau_{lm}$ . At the same time we have argued that the overall $1/\tau_{lm} + 1/\tau_{fr}$ radiative recombination rate is likely to be no more than a factor two faster (by the Purcell effect) than the total radiative rate in the unpatterned film, $1/\tau_{tr} + 1/\tau_{fr}$ . In addition there have been calculations [18] showing that the total radiative rate in the unpatterned thin film $1/\tau_{tr} + 1/\tau_{fr}$ is only 20% slower than in 3-D bulk semiconductor material. Thus we have a calibration chain that links all the measured recombination rates to the fundamental spontaneous rate in the corresponding bulk semiconductor. Here, we will use the various measured efficiency ratios, observed in Fig. 5, and spectrally resolved in Fig. 7, between samples of various geometry's and between points A and B of Fig. 3, to fix the various recombination rates in terms of the bulk semiconductor spontaneous emission rate.

In the case of an unpatterned thin film, that we have used as a reference, a major role is played by photon emission at the rate $1/\tau_{tr}$ , that is permanently trapped below the light cone, $\chi = 0$ . The external efficiency is, as before, a ratio of partial rates

$$
\eta_ {\mathrm{ext}} = \frac {\frac {1}{\tau_ {f r}}}{\frac {1}{\tau_ {f r}} + \frac {1}{\tau_ {t r}} Z ^ {\prime} + \frac {1}{\tau_ {n r}}} (7)
$$

where $Z'$ is in turn the ratio of partial photon absorption rates, and is lacking the leakage term $\chi$ of (3).

Equation (7) is easy to analyze for the case of an unpatterned film, since the partial rates $1 / \tau_{fr}$ and $1 / \tau_{tr}$ are proportional to their respective solid angle fractions, $1 / (2(n_f / n_g)^2)\approx 0.11$ ,

![](images/1e8a2b027b3a9b540196575702593753f3fb742123efc86ca226cecec916cca3.jpg)

<details>
<summary>line</summary>

| Wavelength, nm | PL signal, mV |
| -------------- | ------------- |
| 1600           | 19            |
| 1650           | 17            |
</details>

Fig. 7. The angle dependent spontaneous emission spectrum from leaky conduction band modes in a 2-D photonic crystal. The reference spectrum is from an unpatterned region of the same film under the same pump intensity. The peak-to-reference ratio is $\sim$ 10–15.

and $1 - (1 / (2(n_{f} / n_{g})^{2}))$ . Since there are no exposed surfaces, and the double heterostructures are of high quality, the nonradiative rate $1 / \tau_{nr}$ , is likely to be negligible. Thus, in the worst case, at least $11\%$ of the light will escape. The external efficiency might actually be higher, by photon recycling if the trapped light is not completely parasitically absorbed, i.e., the parasitic absorption fraction $Z' < 1$ . These effects are not likely to be all that serious since there is an upper limit to the external efficiency for these unpatterned thin films, $\eta_{\mathrm{ext}} < 25\%$ , set by the photonic crystal sample which is four times more efficient. Thus we have fixed the external efficiency $\eta_{\mathrm{ext}} \equiv 12\%$ from the unpatterned thin film for the purpose of plotting Fig. 5(b). This sets a corresponding range from 48 to $100\%$ for the external efficiency of the photonic crystal sample. In extracting the Purcell factor from the experimental data, in Fig. 11, we have permitted a variation over that range.

Our experimental results in Fig. 5 and our modeling in (5)-(7), do not agree well with the pioneering numerical computations of S. Fan et al. [13]. In their calculations they took a small computational domain of $5 \times 5$ unit cells, and they put absorbing boundary conditions at the top and bottom of the photonic crystal and at the edges. The electromagnetic energy absorbed at the top and bottom was regarded as extracted, and that at the edges was regarded as lost. This is equivalent to a very short parasitic absorption length $(\alpha')^{-1} = 2.5$ unit cells. The extraction problem was compounded by the assumption of very low surrounding refractive index $n = 1$ in contrast to our use of a glass substrate. Among the leaky conduction band modes, they predict [13] a high efficiency $\eta_{\mathrm{ext}} \approx 80\%$ . It is difficult to see how that could come about, given that the leaky modes have to compete in (6) with a parasitic absorption length close to $(\alpha')^{-1} = 2.5$ unit cells. The leaky modes would have to leak out almost immediately, $Q \approx 1$ , to ensure $80\%$ external efficiency prior to parasitic absorption. Such a low $Q$ appears to be unphysical

In our experiments and our modeling we would have had poor efficiency at conduction band frequencies if we had as much parasitic absorption as assumed by Fan et al. [13]. Furthermore our experiments and modeling show poor efficiency in any case, at frequencies below the leaky conduction band modes, (points A and C).

We attribute the discrepancy between our experimental results and Fan et al. [13] to the small vertical height of their computational domain, $\approx 1$ wavelength in air. It is possible that some of the electromagnetic energy captured by their absorbing boundary conditions might not have been able to escape to infinity in the vertical direction. Thus, they may have overestimated the external efficiency, particularly given the high parasitic absorption they assumed.

# IV. ANGLE RESOLVED PHOTOLUMINESCENCE MEASUREMENTS: BAND STRUCTURE

Spontaneous emission into the leaky conduction band modes appears to lead to good overall spontaneous emission efficiency. The spontaneous emission spectra have peaks at the frequencies and wave vectors corresponding to the leaky conduction band modes. The external angle-dependent emission spectrum directly reveals the dispersion diagram of a photonic crystal's leaky conduction band modes, i.e., modes with frequencies lying above the light cone in glass. Some part of the emission leaking into the glass substrate can be trapped by total internal reflection at the glass-air interface. To avoid that trapping, we cemented the semiconductor film to the center of a glass hemisphere, as shown in Fig. 4. This allowed us to access electromagnetic modes radiating into the glass, at the points near the edges of the escape cone, points C and D of Fig. 3, that would otherwise not be able to escape into air.

The leaky, guided, conduction band modes emit external light at frequencies and external angles imposed by lateral wave vector matching. For that reason the photoluminescence spectrum in a given direction consists of sharp spectral peaks representing emission into photonic bands, on a broad background corresponding to the emission into the continuum of extended modes.

We monitor the evolution of the spontaneous emission spectral peaks versus angle to measure the band structure of the photonic crystal. There is a unique relationship between the frequency of a spectral peak at a certain direction of observation and the wave-vector of the corresponding leaky guided mode.

It is convenient to measure in-plane wave vector, $k_{\parallel}$ , and frequency, f, in units of $\pi/a$ and c/a, respectively, where a is the photonic crystal lattice constant and c is the speed of light in vacuum. Then, from the wavelength of a detected spectral peak $\lambda$ emitted in glass at an angle $\alpha$ from normal incidence, it is straightforward to obtain, based on geometrical considerations, a corresponding point on the dispersion diagram $\nu$ versus $k_{\parallel}$

$$
\nu = \frac {a}{\lambda} \left(\frac {c}{a}\right), \quad k _ {| |} = \frac {2 a n _ {g} \sin \alpha}{\lambda} \left(\frac {\pi}{a}\right) \qquad (8)
$$

where $n_{g}=1.5$ is the refractive index of glass hemisphere. By a proper choice of the azimuthal direction of observation, we could sample the $\Gamma K$ or the $\Gamma M$ edges of the Brillouin zone that are illustrated in Fig. 2. A typical sequence of spectra at observation angles of 29, 32, and $35^{\circ}$ with respect to normal,

![](images/6069c4eb7ed0f4acc973e915580373c3528a5802bf9978b474786a5d8e2f423b.jpg)

<details>
<summary>line</summary>

| Wavelength, nm | TE    | TM    | Reference |
| -------------- | ----- | ----- | --------- |
| 1300           | 0.0   | 0.0   | 0.0       |
| 1400           | 0.2   | 0.1   | 0.0       |
| 1500           | 1.0   | 0.5   | 0.1       |
| 1550           | 5.0   | 1.5   | 0.3       |
| 1600           | 1.5   | 2.5   | 0.5       |
| 1650           | 3.8   | 2.3   | 0.7       |
| 1700           | 0.5   | 0.2   | 0.1       |
</details>

Fig. 8. An example of polarization dependent spontaneous emission from a 2-D thin-film photonic crystal. The reference spectrum is from an unpatterned region of the same film, under the same optical pump intensity. Some modes have both TE and TM character.

are shown in Fig. 1. The smooth evolution of the peaks with the observation angle is clearly visible.

Thin-film photonic crystal modes cannot be classified as pure TE or pure TM. In an infinitely thick 2-D photonic crystal, modes can be divided into pure TM and TE modes, with electric and magnetic field, respectively, parallel to the direction of the voids. This is a consequence of the translational invariance in the vertical direction. While there is no translational invariance in the vertical direction in a thin slab, some of the modes will still have strongly dominating TE or TM components, while others are more mixed.

When observed outside of the photonic crystal, TE modes correspond to the s-polarized light and TM modes correspond to p-polarized light. (s and p are, respectively, polarized perpendicular and parallel to the plane of incidence.) In the vertical direction, normal to the thin film plane, TE and TM modes are not distinguishable. However, the photoluminescence at large observation angles is polarized, depending on the dominant polarization of a corresponding leaky mode. We used a polarizing beam-splitter to select the different polarizations. As can be seen from the Fig. 8, the luminescence consists of a few clearly polarized, or sometimes mixed, peaks.

The spectral peak positions versus angle can be replotted as frequency versus in-plane wave vector $k_{\parallel}$ , using (8). The experimental points are shown with circles in Fig. 9. All detectable bands above the light line correspond to leaky modes, permitting them to be measured. Radiation efficiency into the leaky modes depends on the overlap of the modal electric field with the active region of the photonic crystal, which is determined by the internal structure of the mode. Modes with electric fields concentrated in the air do not couple well to the semiconductor. As indicated by the typical spatial mode in Fig. 6, most of the electric energy in thin-film structures does seem to be concentrated in the semiconductor.

The available frequency range for spontaneous emission is the material linewidth of the active material inside the photonic crystal. InGaAs at room temperature has a full-width at half-maximum (FWHM) linewidth of about 10% of its center frequency. Since we could easily measure peaks in the tails of the material emission band, we had about a 20% dynamic range of usable emission frequencies, as can be seen from Figs. 7 and 8.

![](images/4b053cafe776edc8e0a32b1fe7e26f226b0eaf08c3fec4a28082ae706300fdb8.jpg)

<details>
<summary>line</summary>

| x    | TE polarization | TE polarization |
| ---- | --------------- | --------------- |
| 0.0  | 0.5             | 0.5             |
| 0.1  | 0.45            | 0.48            |
| 0.2  | 0.4             | 0.45            |
| 0.3  | 0.35            | 0.4             |
| 0.4  | 0.3             | 0.35            |
| 0.5  | 0.25            | 0.3             |
| 0.6  | 0.2             | 0.25            |
</details>

Fig. 9. The calculated and measured TE-like and TM-like bands in our 2-D photonic crystal thin film. The agreement as to frequency and polarization character is quite good. The theoretical dispersion was calculated by the FDTD method. The shaded region above the light lines represents light waves that can leak into the glass hemispheres.

The solid and dashed lines in Fig. 9 are the dispersion diagrams, computed using a FDTD algorithm. We see good agreement between the experimental circles and dots in Fig. 9 with the calculated curve shapes, as well as their polarizations. The triangle formed by the crossing of the lowest TE and TM conduction bands in Fig. 9 is a consistent and repeatable feature in several samples of various lattice constants. To our knowledge, this is the first demonstration of internal spontaneous emission directly into the 2-D photonic bands, and it complements the external white light scattering experiments on the thin-film photonic crystals by Astratov et al. [20].

Angle resolved spontaneous emission spectra from thin film photonic crystals reveal some relatively sharp peaks in Fig. 7 compared to the reference emission linewidth of InGaAs in the same figure. The reference luminescence comes from an unpatterned region of the light emitting film. The observed spontaneous emission Q is between 100 and 30, while the material $Q_{m}$ of InGaAs is only $Q_{m} \approx 10$ at room temperature. We define $Q_{m}$ as a ratio of the emission frequency to the full width half magnitude of the ordinary spontaneous emission spectrum.

Another important feature of Fig. 7, besides the high Q, is the very high peak-to-reference ratio, $\approx20$ . That means that the external emission from leaky guided modes at this frequency is 20 times more intense than the emission from an unpatterned film. Based on previous discussions this is mostly due to the greatly enhanced external light extraction caused by Bragg scattering in the photonic crystal, with a small contribution due to the Purcell effect in the extended band modes.

In the next section, we will show how the peak-to-reference enhancement factor can be calculated and compared with experiment.

# V. ANGLE RESOLVED SPONTANEOUS EMISSION ENHANCEMENT

According to Fermi's golden rule the spontaneous emission rate from a transition dipole $\mathbf{d}$ at a location $\mathbf{r}$ into a given

leaky photonic crystal mode is

$$
\Gamma (\mathbf {r}) = \frac {2 \pi}{\hbar} \left\langle (\mathbf {d} \cdot \mathbf {E} (\mathbf {r})) \right\rangle^ {2} \frac {\Delta \omega g}{2 \pi \hbar ((\omega - \omega_ {0}) ^ {2} + (\Delta \omega / 2) ^ {2})} (9)
$$

where d denotes the atomic moment dipole, $\Delta\omega$ is the leaky mode linewidth, and g is the degeneracy of the mode. This is the same as (1) except that the density of states has been replaced by a single-mode, Lorentzian, line-shape, properly normalized, and multiplied by the mode degeneracy g.

Fermi's Golden Rule can be rewritten as a Purcell factor, that depends on cavity $Q$ , that differs from (2), in being the enhancement factor for a quantum well at the peak of the Lorentzian, without spectral integration

$$
\frac {\Gamma}{\Gamma_ {o}} = \frac {3 Q g (\lambda / 2 n) ^ {3}}{\pi^ {2} V _ {\mathrm{eff}}}. (1 0)
$$

In the photonic crystals studied in the previous section, the measured $Q$ 's of electromagnetic conduction band modes were between 30 and 100, comparable to small defect cavities in the same structures [15]. Since these modes are large in volume $V_{\mathrm{eff}}$ , occupying the full area of the 2-D photonic crystal, it might be thought that the Purcell enhancement would me negligible. But there are many optical modes within the angular and spectral resolution of our detection system. These modes add up to a large degeneracy factor $g$ compensating the large modal volume.

By including the degeneracy factor in (2) and (10), we have slightly changed the interpretation of the Purcell concept, from that used by Gerard [5], for example. In Gerard's interpretation, the Purcell factor is an electromagnetic property of a single mode. In artificial photonic band structures, a large density of states, or degeneracy factor, can more than compensate for a large mode volume, and give a significant spontaneous emission enhancement. We find that such spontaneous emission rate enhancements are comparable to the enhancements that accrue to tiny single cavities.

The spontaneous emission rate $\Gamma$ , in (1) and (2) is generally interpreted as the total angle integrated rate. By contrast, in our angle-dependent experiments, we were in fact measuring partial rates by comparing the intensity of photonic crystal spectral peaks to the reference radiation from unpatterned samples. The enhancement factor arises from both Purcell effects and Bragg scattering extraction effects, into the escape cone. For that reason the substantial angle dependent enhancement cannot be regarded as a true Purcell effect, although the formulas look very similar to (2) and (10). We will see that they will look similar except that they have been collapsed into a single spatial dimension. This we shall simply call the angle dependent spontaneous emission enhancement factor, in a narrow frequency range $d\omega$ . Unlike the earlier sections of this paper, there will be no spectral and spatial averaging.

Suppose we select a narrow solid angle $\sin\theta d\theta d\phi$ . Then, in a manner similar to Appendix A, the angle integrated derivation, we have to compare the mode density and local fields for emission into leaky guided modes relative to the reference external spontaneous emission from an unpatterned sample into the glass hemisphere of Fig. 4. This is the enhancement factor we observe directly in Figs. 7 and 8. Spontaneous emission is proportional to the local zero point electric energy of the mode $\mathbf{E}^{2}(\mathbf{r})$ (normalized to $\hbar\omega/2$ in the mode volume) and to the number of modes $\Delta N$ in a given solid angle in a given frequency range. For the reference emission into the 3-D background continuum of modes in the glass, the average zero-point electric field fluctuations occupy whole quantization volume $V = L^{3}$ and can be regarded as propagating into the thin film from the glass

$$
\mathbf {E} ^ {2} (\mathbf {r}) = \frac {\hbar \omega}{2 n _ {f} V}. \tag {11}
$$

The semiconductor refractive index $n_{f}$ in the denominator comes from the averaging the local electric field in the thin film over incidence angles of zero-point fluctuations [21]. The number of electromagnetic modes emitting into the given solid angle in the glass is given by

$$
\Delta N _ {f r} = 2 \frac {k ^ {2} \sin \theta d k d \theta d \phi}{(2 \pi) ^ {3}} V (1 2)
$$

where $k = \omega/c$ is an external wavevector. We apply Fermi's Golden Rule to the free space reference emission into the glass, and to emission into the leaky conduction band modes, as described in Appendix B. This makes the ratio of the two partial spontaneous emission appear like a one-dimensional (1-D) Purcell factor

$$
\frac {\Gamma_ {l m} (\omega)}{\Gamma_ {f r}} = \gamma Q \cdot \frac {\lambda}{2 n _ {f} t} \cos \theta \cdot \frac {\Delta \omega g}{2 \pi \hbar ((\omega - \omega_ {0}) ^ {2} + (\Delta \omega / 2) ^ {2})} \tag {13}
$$

where the detuning of the dipole moment frequency from the mode frequency is taken into account explicitly, and $\Delta\omega$ denotes the spectral width of the leaky mode. The parameter $\gamma$ , defined by (B8) in Appendix B, characterizes the electric energy concentration factor of the photonic bands within the active region.

In semiconductors, the emission band is normally wider than the modal linewidth. For that reason the Lorentzian in (13) has to be convoluted with the semiconductor emission spectrum, as it is done in the derivation of Purcell factor for a semiconductor microcavity [15]. This would integrate out the $2 / \pi$ in the Lorentzian and substitutes $Q$ with the material $Q_{m}$ (10 for InGaAs in our case). However, in this section, we are considering spectrally resolved and angle resolved rate enhancement factors

$$
\frac {\Gamma_ {l m}}{\Gamma_ {f r}} = \frac {2}{\pi} \gamma Q \cdot \frac {\lambda}{2 n _ {f} t} \cos \theta . \tag {14}
$$

In the structure we have studied, the active region covers approximately half of the sample area. When all electric energy of the mode is concentrated in the semiconductor, the concentration parameter is simply $\gamma\approx2$ . Such a concentration of the electric energy within the semiconductor backbone can be seen from the computed energy distribution plot in Fig. 6, a mode close to the $\Gamma$ point of the Brillouin zone. The peak-to-reference intensity ratio is the same as the ratio of partial rates in (14), but it must be corrected by a factor 1/2 to allow for the fact that there is less pumping in the photonic crystal

than in the unpatterned film due to the voids:

$$
\frac {I _ {l m}}{I _ {f r}} = \frac {\Gamma_ {l m}}{\Gamma_ {f r}} \times \frac {1}{2}. \tag {15}
$$

This factor 1/2 is due to absorption, and is similar to the change from Fig. 5(a) to (b). Our thickness t = 240 nm, for a peak wavelength $\lambda = 1650$ nm, and refractive index $n_{f} = 3.2$ , happens to correspond to a half wavelength in the material. In a direction near normal, the intensity enhancement ratio computed from (15), in a direction near normal, is $(I_{lm}/I_{fr}) \approx (2/\pi)Q$ , i.e., roughly 2/3 of the modal Q, or $I_{lm}/I_{fr} \approx 20$ .

The observed peak-to-reference enhancement factor ratio is actually smaller than the predicted $(2/\pi)Q \approx 20$ ; it is more typically $\approx 10$ . We attribute this enhancement reduction to the effects of parasitic absorption. Parasitic absorption reduces the modal Q, but it also reduces the enhancement factor even more. The total modal $Q_{tot}$ becomes diminished as the modes are damped by absorption:

$$
\frac {1}{Q _ {t o t}} = \frac {1}{Q} + \frac {\alpha}{2 \pi \nu} \frac {c}{n _ {f}} \frac {t}{d} \equiv \frac {1}{Q} + \frac {1}{Q _ {a b s}}. \tag {16}
$$

This is a reduction factor of Q by the ratio $[Q_{abs}/Q + Q_{abs}]$ . But there is a further reduction in enhancement factor associated with the parasitic absorption. Only a fraction $[Q_{abs}/Q + Q_{abs}]$ of the spontaneous emission is externally extracted, the rest being absorbed. This is essentially the first parenthesis in (3). Therefore both the observed Q-factor and the extraction efficiency are reduced. The net result is that the angle dependent intensity enhancement factor is reduced by the square $[Q_{abs}/Q + Q_{abs}]^{2}$ .

What is the expected electromagnetic $Q_{abs}$ due to absorption? The InGaAs band absorption spectrum, and luminescence spectrum are given in Fig. 10. At the mid-point of the luminescence spectrum the band-to-band absorption coefficient is $\alpha \approx 9000 \, cm^{-1}$ . In our sample geometry, from Table I, the ratio of active thickness to total thickness is t/d = 1/4, and the refractive index of the semiconductor film is $n_{f} = 3.2$ . Substituting these experimental parameters, we find that at mid-spectrum $Q_{abs} \approx 50$ . This is similar to the purely electromagnetic Q due to leakage of the modes. It would explain the observed peak-to-reference intensity ratio of $\approx 10$ .

From the absorption and luminescence spectra in Fig. 10, the effect of $Q_{abs}$ would be much more severe on the blue edge of the luminescence spectrum, and the absorption effect would be virtually absent on the red edge. The data show a tendency toward lower observed $Q$ 's, and less intensity enhancement for modal peaks near the blue edge of the material spectrum, but the effect appears to be less pronounced than expected.

# VI. EXPERIMENTAL BOUNDS ON THE PURCELL ENHANCEMENT FACTOR

Having developed an understanding of the light emission in 2-D photonic crystals, as represented by (3)-(7), we can proceed to an accounting of recombination rates. We will try to separate the contributions of different effects, and to give the experimental bounds on Purcell enhancement factor. We previously argued on theoretical grounds that the Purcell effect is not likely to be large in a room temperature semiconductor, but the actual value can also be obtained experimentally. Our luminescence efficiency ratios, measured in and out of the leaky conduction band modes, provide enough experimental constraints to put bounds on the Purcell enhancement factor. We have data from the leaky conduction band modes, where there is likely to be a Purcell enhancement effect, and from free space waves that are unlikely to have a Purcell effect. Comparing those two efficiencies, we can extract the Purcell enhancement factor, $F_{p} = \tau_{o}/\tau$ .

![](images/bb2fd27f2bcf88a79f3cdda06742a53c95248cbaa3ce14da7263d58cc0e399ba.jpg)

<details>
<summary>line</summary>

| Wavelength, nm | Absorption, cm⁻¹ | Photoluminescence |
| -------------- | ---------------- | ----------------- |
| 1200           | ~1e+5            | ~1e+3             |
| 1300           | ~1e+5            | ~1e+4             |
| 1400           | ~1e+5            | ~1e+4             |
| 1500           | ~1e+4            | ~1e+4             |
| 1600           | ~1e+3            | ~1e+4             |
| 1700           | ~1e+2            | ~1e+3             |
| 1800           | ~1e+1            | ~1e+2             |
</details>

Fig. 10. The absolute absorption coefficient, $\alpha$ , (solid line) and luminescence spectrum (dashed line) of InGaAs. Luminescence at the blue edge of the emission can suffer reabsorption, photon recycling, lower Q, and lower optical extraction efficiency. The cross-over absorption coefficient appears to $\alpha \sim 9000 \, cm^{-1}$ in these experimental conditions.

A good starting point is (6) for the external efficiency from the photonic crystal in the leaky conduction band frequencies. The experimentally calibrated quantum efficiency is quite good, at least $\eta_{ext} = 48\%$ , external. Under these circumstances parasitic optical absorption is playing a minor role, and it is conservative to neglect it. (If it were included, the Purcell factor estimate would rise slightly.) Then (6) simplifies to:

$$
\eta_ {e x t} = \frac {1}{\tau} / \frac {1}{\tau} + \frac {1}{\tau_ {n r}} \tag {17}
$$

where $1/\tau = (1/\tau_{fr}) + (1/\tau_{lm})$ is the total spontaneous emission rate in the photonic crystal, and $1/\tau_{fr}$ is the partial rate into free external plane waves, and $1/\tau_{lm}$ is the partial rate into leaky modes. Equation (17) implies that:

$$
\frac {\tau_ {n r}}{\tau} = \frac {\eta_ {e x t}}{1 - \eta_ {\mathrm{ext}}}. \tag {18}
$$

At frequencies that emit only into free space plane waves in the glass hemisphere, where leaky conduction band modes are unavailable, the external photonic crystal efficiency, is measured in Fig. 5, to be ten times lower:

$$
\frac {\eta_ {e x t}}{1 0} = \frac {1}{\tau_ {f r}} / \frac {1}{\tau_ {f r}} + \frac {1}{\tau_ {n r}}. \tag {19}
$$

The slow partial rate into free space is responsible for the ten times poorer efficiency in (19), and is due to the absence

![](images/ae698cce8eb38e484e951773295fc297161d5bf0767384b1115bb5bc0347961c.jpg)

<details>
<summary>line</summary>

| Photonic Crystal External Efficiency | Purcell Factor |
| ------------------------------------- | -------------- |
| 0.5                                   | 1.8            |
| 0.7                                   | 3.0            |
</details>

Fig. 11. The experimental bounds on the Purcell enhancement factor. This curve is controlled by the ratio of external efficiency into leaky conduction band modes versus external free space modes. A range of values is given between $1.5 < F_{p} < 3$ due to the uncertainty in the absolute external efficiency of the photonic crystal sample. $F_{p} = 3$ represents an absolute cap, since that is the Purcell enhancement in the smallest possible nanocavity.

of the leaky conduction band partial rate $1/\tau_{lm}$ . Rearranging (19) implies that

$$
\frac {\tau_ {n r}}{\tau_ {f r}} = \frac {\eta_ {e x t}}{1 0 - \eta_ {e x t}}. (2 0)
$$

Now let us reference the Purcell factor to the internal spontaneous emission rate in the unpatterned thin film. That rate is $(1/\tau_{fr}) + (1/\tau_{tr})$ , but since the bidirectional escape cone represents $1/(2(n_{f}/n_{g})^{2})$ of $4\pi$ steradians, the reference spontaneous emission rate inside the unpatterned thin film can be written $(2(n_{f}/n_{g})^{2})/\tau_{fr}$ . The Purcell factor is then:

$$
\begin{array}{l} F _ {p} = \frac {\frac {1}{\tau}}{\frac {2 (n _ {f} / n _ {g}) ^ {2}}{\tau_ {f r}}} \\ = \frac {\tau_ {n r}}{\tau} \cdot \frac {\tau_ {f r}}{\tau_ {n r}} \cdot \frac {1}{2 (n _ {f} / n _ {g}) ^ {2}} \\ = \frac {\eta_ {\mathrm{ext}}}{1 - \eta_ {\mathrm{ext}}} \cdot \frac {1 0 - \eta_ {\mathrm{ext}}}{\eta_ {\mathrm{ext}}} \cdot \frac {1}{2 (n _ {f} / n _ {g}) ^ {2}} \\ = \frac {1 0 - \eta_ {\mathrm{ext}}}{\left(1 - \eta_ {\mathrm{ext}}\right) 2 \left(n _ {f} / n _ {g}\right) ^ {2}} \tag {21} \\ \end{array}
$$

where the final expression in (21) is the result of substitution by (18) and (20). The only experimental input in (21) is the factor 10, the observed ratio in the spectral and angle integrated spontaneous emission tuned in and out of the leaky conduction band modes.

Equation (21) is plotted in Fig. 11 over the measured 48–100% uncertainty range of the external efficiency $\eta_{ext}$ from the photonic crystal films. Actually we don’t regard $\eta_{ext} > 70\%$ to be realistic, since that would imply a Purcell Factor >3, larger than could be produced in the tiniest [15] optical cavities. Thus the measured uncertainty range of the Purcell Factor is $1.5 < F_{p} < 3$ , which is close to the physical estimate $F_{p} \approx 2$ we made earlier in this article, that was based on the volume concentration of electric energy in the ribs of the photonic crystal.

# VII. PURCELL ENHANCEMENT APPLIED TO LIGHT EMITTING DIODES

The form of 2-D photonic crystal that we have studied, consists of a triangular array of voids, formed in a thin semiconducting film. We have already seen in Fig. 6 that the leaky conduction band modes tend to concentrate in the semiconducting ribs of the structure, and to keep away from the voids. This electric energy concentration is responsible for a modest Purcell effect on the carrier spontaneous emission rate. At the same time the exposed inner surfaces of the voids suffer from nonradiative surface recombination, defeating much of the benefit derived from the Purcell effect. In this section we will quantify the severity of these surface recombination effects.

Nonradiative recombination in our photonic crystals is dominated by surface recombination. Since the scale of the photonic crystal is submicron, we can conservatively assume (this being the worst case scenario), that the diffusion length is longer than the period of the photonic lattice. Then the carrier concentration would be constant over the area of the photonic crystal permitting the maximum surface recombination. Total surface recombination is then proportional to the perimeter of the exposed surface, $n_{min} \times S \times 2\pi r \times t$ , where $n_{min}$ is the minority carrier density (1/cm³), S is the surface recombination velocity (cm/s), $2\pi r$ is the length of the internal perimeter of a circular void in the photonic crystal (cm), and t is the thickness of the active region (cm). This is the total number of carriers recombining nonradiatively per second. To get the corresponding nonradiative lifetime, we have to divide by the total number of carriers in the photonic crystal unit cell, $n_{min} \times \text{Area} \times \text{thickness} = n_{\text{min}} \times \{(\sqrt{3}/2)a^{2} - \pi r^{2}\} \times t$ , where the area of the photonic crystal unit cell, $(\sqrt{3}/2)a^{2}$ has been corrected by the void area $\pi r^{2}$ . Dividing the number of carriers into the number recombining per second gives the nonradiative recombination rate

$$
\frac {1}{\tau_ {n r}} = \frac {S \times 2 \pi r}{(\sqrt {3} / 2) a ^ {2} - \pi r ^ {2}}. \tag {22}
$$

In good semiconductor surfaces, such as InGaN or InGaAs, the surface recombination velocity can be as low [16] as $S \approx 10^{4}$ cm/s. Scaling to the $a = 600$ nm length scale of our photonic crystals, $\tau_{nr}$ is measured in nanoseconds. This is roughly the speed of spontaneous radiative recombination in semiconductors, $1/\tau = B \times N$ , where $B$ is [21] $\sim 5 \times 10^{-10}$ cm $^{3}$ /s and the majority carrier density is usually $N \approx 10^{18}$ /cm $^{3}$ . Thus we have to struggle against surface recombination.

The external efficiency of a photonic crystal is: $\eta_{\mathrm{ext}} = (1/\tau)/(1/\tau) + (1/\tau_{nr})$ , where we have neglected parasitic optical absorption, but kept nonradiative recombination. The radiative recombination can be enhanced by a Purcell effect $1/\tau = F_{p}/\tau_{o}$ and we can substitute the nonradiative recombination rate, (22) into the efficiency formula, and assume a typical photonic crystal aspect ratio r/a = 0.37. The external efficiency then becomes

$$
\eta_ {\mathrm{ext}} = \frac {1}{1 + 3 . 8 \frac {S \tau_ {o}}{a F _ {p}}}. \tag {23}
$$

If we set a minimum interesting external efficiency limit of $\eta_{ext} > 50\%$ , then that implies the following required inequality:

$$
\frac {S \tau_ {o}}{a F _ {p}} <   0. 2 7. \tag {24}
$$

For a given material system this inequality defines whether the use of perforated photonic crystals as active light emitters is advantageous. The formula is useful because the $S\tau_{o}$ product has been measured [16] in a number of material systems. Let us consider InGaAs. The $S\tau_{o}$ product was measured to be $S\tau_{o} = 0.45 \mu m$ . Combined with the maximally achievable Purcell number $F_{p} = 3$ and lattice constant yielding the best signal a = 900 nm, this gives $(S\tau_{o}/aF_{p}) = 0.16 < 0.27$ , consistent with our observation of at least 50% external quantum efficiency.

Another important material system is InGaN/GaN. Scaling the photonic crystal down for blue light requires the lattice constant $a = 350$ nm. We have measured [16] the surface recombination velocity in gallium nitride, in terms of the diffusion constant $D$ , to be $S \approx 3 \times 10^{4} \mathrm{~cm}^{-1} \times D$ , which, used in the inequality (23) gives

$$
\frac {S \tau_ {o}}{a F _ {p}} = \frac {3 \times 1 0 ^ {4} \mathrm{cm} ^ {- 1} D \tau_ {o}}{a F _ {p}} <   0. 2 7,
$$

or

$$
L _ {D} <   \sqrt {F _ {p}} \cdot 1 8 0 \mathrm{nm} \tag {25}
$$

where we used diffusion length definition $L_{D} = \sqrt{\tau_{o} D}$ . Doped quantum wells in GaN heterostructures are well known for relatively short ambipolar diffusion length [22] $L_{D} \sim 200$ nm due to the very fast radiative rate. Also, of since GaN has a high material Q, the maximally achievable Purcell factor is $F_{p} \sim 10$ , which makes it possible to satisfy inequality (25), and to achieve an external efficiency $<50\%$ , even in a perforated GaN photonic crystal film.

Another important visible-light optoelectronic material, In-GaAlP has been shown $[16]$ to have high nonradiative surface recombination rate, not satisfying $(24)$ . Purcell enhancement still can be beneficial for materials emitting in yellow-green part of the spectrum. These materials have low internal efficiency that can benefit from the Purcell effect. Indeed, for these alloys, advanced designs $[23]$ are used, that have about 30% extraction efficiency. The external efficiency is simply the product of internal efficiency and extraction efficiency

$$
\eta_ {\mathrm{ext}} = \frac {1 / \tau}{1 / \tau + 1 / \tau_ {n r}} \times 0. 3 (2 6)
$$

where $\tau$ and $\tau_{nr}$ are radiative and nonradiative minority carrier lifetimes, respectively. For a photonic crystal structure the external efficiency is given by (23) which can be rewritten as follows:

$$
\eta_ {\mathrm{ext}} = \frac {F _ {p}}{F _ {p} + \tau_ {o} / \tau_ {n r} + 3 . 8 \frac {S \tau_ {o}}{a}}. (2 7)
$$

In these alloys the internal efficiency is low due to bulk nonradiative recombination, the second term in the denominator. If the bulk internal efficiency is so low that surface recombination is unimportant, there can be three-fold enhancement in light extraction efficiency due to superior extraction from leaky conduction band modes, combined with the $F_{p}$ -fold improvement in the internal quantum efficiency of the material. Therefore the total improvement in the total efficiency could be as high as $3F_{p}$ . Certainly, the same reasoning would apply to a host of thin films organic LED's as well.

# VIII. BRAGG EXTRACTION ENHANCEMENT APPLIED TO LED'S

The dominant benefit of the 2-D photonic crystal appears to be the extraction advantage from Bragg scattering that allows large wave-vector photons to escape from the semiconductor film. By introducing periodicity, we fold the dispersion diagram into the Brillouin zones, so that modes at all wave vectors are, in principle, leaky. That is no matter how weak a periodic perturbation is, all modes can escape provided that they are not reabsorbed first. That extraction advantage does not require voids that cut through the active layer of the semiconductor film. Coherent photonic crystal extraction will work with blind voids as well. Thus, a photonic crystal with blind voids would not suffer from surface recombination. However for a weak modulation, the leakage rate may be small, and for that reason it may be not be competitive with randomly textured films. In this section we will analyze the extraction advantage provided by coherent scattering from a photonic crystal relative to the incoherent extraction obtained from a randomly textured surface.

If there were no parasitic absorption, then any scattering mechanism would be strong enough to ensure 100% external extraction efficiency. The challenge for us is to design a surface structure that helps to extract the internal light as quickly as possible, before it is dissipated by parasitic absorption. In the case of a photonic crystal the optical extraction efficiency was already given as the first parenthesis in (3) shown in (28) at the bottom of the page.

Correspondingly, we know that the extraction efficiency in unpatterned films is rather poor as indicated by (7), and small escape cones in general. A random pattern improves [24] the situation greatly relative to an unpatterned film. In that case, the dynamics of the light rays can be modeled [25] in a so-called “photon gas model.” Optical energy in the film leaks

$$
\text { Photonic   Crystal   Extraction   Efficiency } = \left(\frac {\frac {2 \pi \nu}{Q}}{\frac {2 \pi \nu}{Q} + \alpha^ {\prime} \frac {c}{n} \frac {t}{d}}\right). \tag {28}
$$

![](images/1dc58801194b20b3f50eae1f6c387b76ac664fde693ead094390abe2ed9f2cf5.jpg)

<details>
<summary>text_image</summary>

20µm
</details>

(a)

![](images/af3a2676bc6104c834aa721970db70268a3ebf18c1f421c923d9cab1cc01866c.jpg)

<details>
<summary>text_image</summary>

Extracted photoluminescence
Glass substrate
240nm
20nm
InP
Pump light
InGaAs
</details>

(b)   
Fig. 12. A concept for separating the light generation region from the light extraction region in an LED: (a) top view and (b) side view. Spontaneous emission is generated at the center, and then is extracted from the 2-D patterned region around the periphery. Photopumped external quantum efficiency of 70% has been demonstrated by this structure.

out through the escape cone at the rate $1/(2(n_{f}/n_{g})^{2})(c/n_{f}d)$ which the probability of escape divided by the average time to reach the film surface from the interior, where d=240 nm is the semiconductor film thickness. Therefore, in a randomly patterned film, the extraction efficiency is shown in (29) at the bottom of the page. The relative effectiveness of coherent extraction in (28), and incoherent extraction in (29) can be gauged by comparing the two extraction rates

$$
\frac {2 \pi \nu}{Q} \Leftrightarrow \frac {1}{2 (n _ {f} / n _ {g}) ^ {2}} \frac {c}{n _ {f} d}. \tag {30}
$$

Since the film thickness $d$ is close to half a wavelength in the semiconductor, therefore $c/n_{f}d=2\nu$ . Thus, the coherent extraction rate in (30) becomes superior when modal $Q<2\pi(n_{f}/n_{g})^{2}$ or $Q\leq28$ . The goal here is to have the lowest possible $Q$ , not the highest $Q$ . A low $Q$ is needed in order to compete with randomly surface textured LED's [26], [27].

Another approach to increase LED efficiency is to separate the light generation region from the extraction region, as shown in Fig. 12. When the light from the center arrives at a photonic crystal patterned region around the periphery, it couples into leaky conduction band modes, as a passive out-coupling mechanism [28]. The periodic structure is in effect an efficient, coherent scatterer of light from the semiconductor into the glass substrate, from which it can escape externally.

The two respective regions of Fig. 12 are the hexagonal area of unpatterned thin film for light generation, surrounded by a few periods of the photonic crystal for light extraction. Since the group velocity in photonic crystals tends to be rather slow, the light will not travel far within the external leakage time.

If the spontaneous emission is generated in the center region, a small fraction $1/2(n_{f}/n_{g})^{2}$ corresponding to top and bottom escape cones is emitted directly from the central part of the unpatterned hexagonal area, and the rest is trapped in the thin film waveguide. When the guided light reaches the surrounding patterned region, it scatters or reflects at the interface $[29]$ , or couples to the leaky modes of the photonic crystal, and then scatters into the air or into the glass substrate. In absolute photoluminescence efficiency measurements, that structure was shown $[28]$ to be capable of 70% external quantum efficiency, indicating its resistance to parasitic absorption effects.

# IX. SUMMARY

The results of photoluminescence measurements on thin slab InGaAs/InP photonic crystals were presented in this paper, demonstrating a possibility of spontaneous emission engineering at room temperature using 2-D periodic thin film photonic crystals. The angle dependence of the PL spectral peaks was shown to track the photonic band dispersion. This revealed the band structure of the leaky conduction bands within the optical escape cone. Up to a 15-fold emission intensity enhancement was observed and explained in terms of a combination of Purcell enhancement, and Bragg extraction of high wave vector photons. The Purcell enhancement factor was probably no more than two under our conditions, with most of the efficiency increase associated with Bragg extraction improvement. Different design concepts for improving LED performance were demonstrated.

# APPENDIX A

# PURCELL FACTOR DERIVED FROM FERMI'S GOLDEN RULE

The Purcell enhancement factor $^{2}$ should be modified when considering generation of light in semiconductor quantum-well (QW) structures.

$$
\text { Random   Texture   Extraction   Efficiency } = \left(\frac {\frac {1}{2 (n _ {f} / n _ {g}) ^ {2}} \frac {c}{n _ {f} d}}{\frac {1}{2 (n _ {f} / n _ {g}) ^ {2}} \frac {c}{n _ {f} d} + \alpha^ {\prime} \frac {c}{n _ {f}} \frac {t}{d}}\right). \tag {29}
$$

The classical electric field of the cavity mode $\mathbf{E}(\mathbf{r})$ has to be normalized $E \rightarrow \alpha E$ so that

$$
\frac {1}{4 \pi} \int \varepsilon (\mathbf {r}) (\alpha \mathbf {E} (\mathbf {r})) ^ {2} d ^ {3} \mathbf {r} = \frac {\hbar \omega_ {0}}{2} \qquad \mathrm{(A1)}
$$

where $\varepsilon$ is the material dielectric constant and $\omega_{0}$ is the resonant frequency of the cavity. This gives the normalization factor

$$
\alpha^ {2} = \frac {2 \pi \hbar \omega_ {0}}{\int \varepsilon (\mathbf {r}) \mathbf {E} ^ {2} (\mathbf {r}) d ^ {3} \mathbf {r}}. \tag {A2}
$$

In the above equations the integration extends over the quantization volume. The spontaneous emission rate into the resonant mode at a given point r follows from the Fermi golden rule and equals to

$$
\Gamma (\mathbf {r}) = \frac {2 \pi}{\hbar} \left\langle (\mathbf {d} \cdot \alpha \mathbf {E} (\mathbf {r})) \right\rangle^ {2} \frac {g}{\hbar \Delta \omega} \tag {A3}
$$

where $\mathbf{d}$ denotes the atomic moment dipole, $\Delta\omega$ is the cavity linewidth, and $g$ is the degeneracy of the cavity mode. The last term in (8) represents the density of electromagnetic modes—there are $g$ modes in the frequency range $\Delta\omega$ . The dot product $(\mathbf{d} \cdot \alpha\mathbf{E}(\mathbf{r}))$ has to be averaged over the possible orientations of the atomic dipole moment. At this point, we need to take account of the specific optical transition in semiconductor quantum wells. First, electron-heavy hole transitions are the major contributor to the spontaneous emission. Second, these transitions are only allowed if the dipole moment of the transition lies in the plane of the quantum well, so that $d_x^2 = d_y^2 = \mathrm{d}^2/2$ , $d_z^2 = 0$ . Then, if the mode's electric field is also in the QW plane, as happens for TE modes, the average $\langle(\mathbf{d} \cdot \alpha\mathbf{E}(\mathbf{r})) \rangle^2 = (1/2)\mathrm{d}^2(\alpha\mathbf{E}(\mathbf{r}))^2$ . Note that in the case of bulk semiconductor, such as our thick well, or for interaction with random modes, the prefactor above would have been $1/3$ .

If the active material is placed in the point of maximum electric field of the mode, the emission rate is

$$
\Gamma = \frac {2 \pi}{\hbar} \frac {\mathbf {d} ^ {2}}{2} (\alpha \mathbf {E} _ {\mathrm{max}}) ^ {2} \frac {g}{\hbar \Delta \omega}. \tag {A4}
$$

Introducing the value of the normalization factor $\alpha^{2}$ gives

$$
\Gamma = \frac {\omega_ {0}}{\Delta \omega} \frac {2 g \pi^ {2} \mathbf {d} ^ {2}}{\hbar} \frac {\mathbf {E} _ {\mathrm{max}} ^ {2}}{\int \varepsilon (\mathbf {r}) \mathbf {E} ^ {2} (\mathbf {r}) d ^ {3} \mathbf {r}}. \qquad \mathrm{(A5)}
$$

The spontaneous emission rate in the bulk can be calculated using the classical formula [30]

$$
\Gamma_ {0} = \frac {4 n \mathbf {d} ^ {2} \omega^ {3}}{3 \hbar c ^ {3}} = \frac {4 n \mathbf {d} ^ {2} 8 \pi^ {3}}{3 \hbar \lambda^ {3}}. \tag {A6}
$$

Thus, the overall enhancement factor, which can be used as a figure of merit for the optimization of a resonant-cavity structure becomes

$$
\frac {\Gamma}{\Gamma_ {0}} = Q g \frac {3 \lambda^ {3}}{1 6 \pi n} \frac {\mathbf {E} _ {\max} ^ {2}}{\int \varepsilon (\mathbf {r}) \mathbf {E} ^ {2} (\mathbf {r}) d ^ {3} \mathbf {r}} \tag {A7}
$$

where $Q = \omega_0 / \Delta \omega$ is the quality factor of the cavity. Introducing the mode volume $V_{\mathrm{eff}}$

$$
V _ {\mathrm{eff}} = \frac {\int \varepsilon (\mathbf {r}) \mathbf {E} ^ {2} (\mathbf {r}) d ^ {3} \mathbf {r}}{(\varepsilon (\mathbf {r}) \mathbf {E} ^ {2} (\mathbf {r})) _ {\max}} \tag {A8}
$$

the enhancement of spontaneous emission rate can be recast in the following form:

$$
\frac {\Gamma}{\Gamma_ {0}} = \frac {3 Q g (\lambda / 2 n) ^ {3}}{2 \pi V _ {\mathrm{eff}}}. \tag {A9}
$$

This expression differs from the one originally derived by Purcell by a degeneracy factor g and a factor of $\pi/4$ which follows from the peculiarities of the optical transitions is quantum well structures and Lorentzian line shapes.

# APPENDIX B ANGLE DEPENDENT SPONTANEOUS EMISSION ENHANCEMENT FACTOR

According to Fermi's Golden Rule (1), the spontaneous emission rate is proportional to the square of the local field of the mode $E^2(r)$ (normalized to $\hbar \omega / 2$ in the mode volume) and to the number of modes $\Delta N$ emitting into the given solid angle in a given frequency range.

For the reference emission into the 3-D background continuum of modes, the average zero-point electric field fluctuations propagate into the thin-film from external surroundings [31]

$$
\mathbf {E} ^ {2} (\mathbf {r}) = \frac {\hbar \omega}{2 n V} \tag {B1}
$$

and number of modes emitting into the given solid angle is given by

$$
\Delta N _ {\mathrm{ref}} = 2 \frac {k ^ {2} \sin \theta d k d \theta d \phi}{(2 \pi) ^ {3}} V \tag {B2}
$$

where $k = \omega / c$ is an external wavevector and $V = L^3$ is the macroscopic quantization volume.

For the case of emission into the leaky modes, the local field depends on the position of the emission dipole in the active region. We can assume that the electric field is concentrated in the slab waveguide volume of thickness t. Then, the average square of the electric field at a point r normalized by a half quantum is given by

$$
\begin{array}{l} \mathbf {E} ^ {2} (\mathbf {r}) = \frac {\hbar \omega}{2 \int_ {\mathrm{slab}} \varepsilon (\mathbf {r} ^ {\prime}) \mathbf {E} _ {0} ^ {2} (\mathbf {r} ^ {\prime}) d ^ {3} \mathbf {r} ^ {\prime}} \mathbf {E} _ {0} ^ {2} (\mathbf {r}) \\ \approx \frac {\hbar \omega}{2 (t / 2) \int_ {\mathrm{slab}} \varepsilon (\mathbf {r} ^ {\prime}) \mathbf {E} _ {0} ^ {2} (\mathbf {r} ^ {\prime}) d ^ {2} \mathbf {r} ^ {\prime}} \mathbf {E} _ {0} ^ {2} (\mathbf {r}) \qquad \mathrm{(B3)} \\ \end{array}
$$

where $\mathbf{E}_{0}(\mathbf{r})$ is the nonnormalized electric field distribution of the mode. Integration in the last expression may be approximated over the center plane of the slab. The number of leaky guided modes of a thin slab photonic crystal emitting at this angle is given by a projection of 3-D k-space onto the 2-D plane of guided-mode wavevectors

$$
\Delta N _ {\mathrm{ref}} = 2 \frac {k _ {| |} d k _ {| |} d \phi}{(2 \pi) ^ {2}} A _ {q a} \tag {B4}
$$

where $k_{\parallel} = k \sin \theta$ , $dk_{\parallel} = k \cos \theta d\theta$ , and $A_{qa} = L^{2}$ is the quantization area of the slab. Averaging over the active region $A_{active}$ in a unit cell would give the total radiation rate in a given direction. However, since the line width of the leaky mode $\Delta\omega$ is larger than the experimental spectral resolution $d\omega$ , only $(2/\pi) (d\omega/\Delta\omega)$ of the radiation will be in the spectral range. The $2/\pi$ prefactor comes from the Lorentzian lineshape of a leaky mode, similar to the derivation in [15]. Thus, the rate of reference continuum background radiation would be given by

$$
\begin{array}{l} \Gamma_ {\mathrm{ref}} \propto \int_ {\mathrm{active}} \mathbf {E} ^ {2} (\mathbf {r}) \Delta N _ {3 d} d ^ {2} r \\ = 2 \frac {k ^ {2} \sin \theta d k d \theta d \phi}{(2 \pi) ^ {3}} V \cdot \frac {\hbar \omega}{2 n V} \cdot A _ {\mathrm{active}}. \qquad (\mathrm{B5}) \\ \end{array}
$$

And, in the same way, emission into the leaky modes (lm) is

$$
\begin{array}{l} \Gamma_ {l m} \propto \int_ {\mathrm{active}} \mathbf {E} ^ {2} (\mathbf {r}) \Delta N _ {2 d} d ^ {2} r \\ = \frac {\hbar \omega}{2 (t / 2) \int_ {q a} \varepsilon (\mathbf {r ^ {\prime}}) \mathbf {E} _ {0} ^ {2} (\mathbf {r ^ {\prime}}) d ^ {2} \mathbf {r ^ {\prime}}} \cdot \frac {2 d \omega}{\pi \Delta \omega} \\ \cdot \frac {(k \sin \theta) (k \cos \theta d \theta) d \phi}{(2 \pi) ^ {2}} \int_ {\mathrm{active}} A \mathbf {E} _ {0} ^ {2} (\mathbf {r}) d ^ {2} r. \quad (\mathrm{B6}) \\ \end{array}
$$

Taking ratio of (B5) and (B6), and simplifying, we obtain

$$
\begin{array}{l} \frac {\Gamma_ {l m}}{\Gamma_ {\mathrm{ref}}} = \frac {1}{\pi} \frac {A _ {q a}}{\int_ {q a} \varepsilon (\mathbf {r} ^ {\prime}) \mathbf {E} _ {0} ^ {2} (\mathbf {r} ^ {\prime}) d ^ {2} \mathbf {r} ^ {\prime}} \cdot \frac {\int_ {\mathrm{active}} \varepsilon \mathbf {E} _ {0} ^ {2} (\mathbf {r}) d ^ {2} r}{A _ {\mathrm{active}}} \\ \cdot \frac {\omega}{\Delta \omega} \cdot \frac {\lambda}{n t} \cos \theta . \tag {B7} \\ \end{array}
$$

Introducing mode's $Q = \omega / \Delta \omega$ , and a parameter $\gamma$ to characterize electric energy overlap with the active region

$$
\gamma = \frac {A _ {q a}}{A _ {\mathrm{active}}} \cdot \frac {\int_ {\mathrm{active}} \varepsilon \mathbf {E} _ {0} ^ {2} (\mathbf {r}) d ^ {2} \mathbf {r}}{\int_ {q a} \varepsilon (\mathbf {r} ^ {\prime}) \mathbf {E} _ {0} ^ {2} (\mathbf {r} ^ {\prime}) d ^ {2} \mathbf {r} ^ {\prime}} \tag {B8}
$$

the enhancement of spontaneous emission rate at a given frequency can be recast into a familiar structure of a Purcell factor in 1-D

$$
\frac {\Gamma_ {l m} (\omega)}{\Gamma_ {\mathrm{ref}}} = \frac {2}{\pi} \gamma Q \cdot \frac {\lambda}{2 n t} \cos \theta . \tag {B9}
$$

# REFERENCES

[1] P. Selenyi, Phys. Rev., vol. 56, pp. 477–480, 1939.   
[2] E. M. Purcell, “Spontaneous emission probabilities at radio frequencies,” Phys. Rev., vol. 69, p. 681, 1946.   
[3] P. Goy, J. M Raimond, M. Gross, and S. Haroche, “Observation of cavity enhanced single atom spontaneous emission,” Phys. Rev. Lett., vol. 50, p. 1903, 1983; K. H. Drexhage, “Interaction of light with monomolecular dye layers,” in Progress in Optics, E. Wolf, Ed. Amsterdam, The Netherlands: North Holland, 1974, p. 165.   
[4] R. G. Hulet, E. S. Hilfer, and D. Kleppner, “Inhibited spontaneous emission by a Rydberg atom,” Phys. Rev. Lett., vol. 50, p. 1903, 1983.   
[5] J. Gérard, B. Sermage, B. Gayral, B. Legrand, E. Costard, and V. Tierry-Mieg, “Enhanced spontaneous emission by quantum boxes in a monolithic optical microcavity,” Phys. Rev. Lett., vol. 91, no. 5, pp. 1110–1113, 1998.

[6] E. Yablonovitch, “Inhibited spontaneous emission in solid-state physics and electronics,” Phys. Rev. Lett., vol. 58, no. 20, pp. 2059–2062, 1987.   
[7] W. B. Joyce, R. J. Bachrach, R. W. Dixon, and D. A. Sealer, “Geometrical properties of random particles and the extraction of photons from electroluminescent diodes,” J. Appl. Phys., vol. 45, no. 5, pp. 2229–2253, May, 1974.   
[8] I. Schnitzer, E. Yablonovitch, C. Caneau, T. J. Gmitter, and A. Scherer, "30% external quantum efficiency from surface textured, thin-film light-emitting diodes," Appl. Phys. Lett., vol. 63, p. 2174, 1993.   
[9] A. Scherer, O. Painter, B. D'Urso, R. Lee, and A. Yariv, "InGaAsP photonic band gap crystal membrane microresonators," J. Vac. Sci. and Tech., vol. B16, p. 3906, 1998.   
[10] R. D. Meade, K. D. Brommer, A. M. Rappe, and J. D. Joannopoulos, "Existence of a photonic band gap in two dimensions," Appl. Phys. Lett., vol. 61, p. 495, 1992.   
[11] P. R. Villeneuve and M. Piche, “Photonic band gaps in two-dimensional square lattices: Square and circular rods,” Phys. Rev., vol. B46, p. 4973, 1992.   
[12] E. Yablonovitch, T. J. Gmitter, and K. M. Leung, “Photonic band structure: The face-centered-cubic case employing nonspherical atoms,” Phys. Rev. Lett., vol. 67, p. 2295, 1991.   
[13] D. N. Atkin, P. S. J. Russell, T. A. Birks, and P. J. Roberts, J. Modern Opt. 43, pp. 1035–1053, 1996.   
[14] S. Fan, P. Villeneuve, J. Joannopoulos, and E. F. Schubert, Phys. Rev. Lett., vol. 78, p. 3295, 1997.   
[15] M. Boroditsky, R. Coccioli, and E. Yablonovitch, “Analysis of photonic crystals for light-emitting diodes using the finite difference time domain technique,” in Proc. Photonics West’98, San Jose, CA, Jan. 26–30, 1998, vol. 3283, pp. 184–190.   
[16] R. Coccioli, M. Boroditsky, K. Kim, Y. Rahmat-Samii, and E. Yablonovitch, “The smallest possible electromagnetic mode volume in a dielectric cavity,” in Inst. Elec. Eng. Proc., J. (Optoelectron.), 1998, vol. 145, no. 6, pp. 391–397.   
[17] M. Boroditsky, I. Gontijo, M. Jackson, R. Vrijen, T. F. Krauss, C.-C. Cheng, A. Scherer, R. Bhat, M. Krames, and E. Yablonovitch, "Surface recombination measurements on III–V candidate materials for nano-structure light-emitting diodes," to be published.   
[18] O. Painter, R. K. Lee, A. Scherer, A. Yariv, J. D. O'Brien, P. D. Dapkus, and I. Kim, "Two-dimensional photonic bandgap defect laser," Science, vol. 284, p. 1819, 1999.   
[19] Y. Xu, J. Vuckovic, R. Lee, O. Painter, A. Scherer, and A. Yariv, “Finite-difference time-domain calculation of spontaneous emission lifetime in a microcavity,” J. Opt. Soc. Amer., vol. B16, pp. 465–474, 1999.   
[20] A. Yariv and P. Yeh, Optical Waves in Crystals. New York: Wiley, 1984.   
[21] V. N. Astratov, R. M. Stevenson, M. S. Skolnick, D. M. Whittaker, S. Brand, I. Culshaw, T. F. Krauss, R. M. De La Rue, and O. Z. Karimov, “Experimental technique to determine the band structure of two-dimensional photonic lattices,” in Inst. Elec. Eng. Proc. J. (Optoelectron.), vol. 145, pp. 398–402, 1998; also V. N. Astratov, I. S. Culshaw, R. M. Stevenson, D. M. Whittaker, M. S. Skolnick, T. F. Krauss, and R. M. De La Rue, “Resonant coupling of near-infrared radiation to photonic band structure waveguides,” this issue, pp. 2050–2057.   
[22] E. Yablonovitch, T. J. Gmitter, and R. Bhat, “Inhibited and enhanced spontaneous emission from optically thin AlGaAs/GaAs double heterostructures,” Phys. Rev. Lett., vol. 61, pp. 2546–2549, 1988.   
[23] A. Vertikov, I. Ozden, and A. V. Nurmikko, “Investigation of excess carrier diffusion in nitride semiconductors with near field microscopy,” Appl. Phys. Lett., vol. 74, pp. 850–852, 1999.   
[24] N. F. Gardner, H. C. Chui, E. I. Chen, M. R. Krarnes, J.-W. Huang, F. A. Kish, S. A. Stockman, C. P. Kocot, T. S. Tan, and N. Moll, “1.4× efficiency improvement in transparent-substrate $(\mathrm{Al}_{x}\mathrm{Ga}_{1-x})_{0.5}\mathrm{In}_{0.5}\mathrm{P}$ light-emitting diodes with thin ( $\leq200nm$ ) active regions,” Appl. Phys. Lett., vol. 74, pp. 2230–2232, 1999; M. Boroditsky and E. Yablonovitch, “Light extraction efficiency from light-emitting diodes,” SPIE, vol. 3002, pp. 119–122, 1997.   
[25] M. Boroditsky, R. Ragan, and E. Yablonivitch, “Absorption enhancement in ultra-thin textured AlGaAs films,” Solar Energy Solar Cell Mater., vol. 57, pp. 1–7, 1999.   
[26] E. Yablonovitch, “Statistical ray optics,” J. Opt. Soc. Amer., vol. 72, p. 899, 1982.   
[27] I. Schnitzer, E. Yablonovitch, C. Caneau, and T. J. Gmitter, J. Appl. Phys. Lett., vol. 62, p. 131, 1993.   
[28] R. Windisch, P. Heremans, A. Knobloch, P. Kiesel, G. H. D’lhler, B. Dutta, and G. Borghs, Appl. Phys. Lett., vol. 74, p. 2256, 1999.   
[29] M. Boroditsky, T. F. Krauss, R. Coccioli, R. Vrijen, R. Bhat, and E. Yablonovitch, “Light extraction from optically pumped light-emitting

diode by thin slab photonic crystals," Appl. Phys. Lett., Aug. 23, 1999, to appear.   
[30] D. Labilloy, H. Benisty, C. Weisbuch, T. F. Krauss, D. Cassagne, R. M. De La Rue, V. Bardinal, R. Houdré, U. Oesterle, D. Cassagne, and C. Jouanin, Phys. Rev. Lett., vol. 79, pp. 4197–4150, 1997.   
[31] P. Milonni, The Quantum Vacuum: An Introduction to Quantum Electrodynamics. Boston, MA: Academic, 1994.   
[32] E. Yablonovitch, T. J. Gmitter, and R. Bhat, “Inhibited and enhanced spontaneous emission from optically thin AlGaAs/GaAs double heterostructures,” Phys. Rev. Lett., vol. 61, pp. 2546–2549, 1988.

Thomas F. Krauss, photograph and biography not available at the time of publication.

Roberto Coccioli, photograph and biography not available at the time of publication.

Misha Boroditsky, photograph and biography not available at the time of publication.

Raj Bhat, photograph and biography not available at the time of publication.

Rutger Vrijen, photograph and biography not available at the time of publication.

Eli Yablonovitch (M'75–SM'90–F'92), for a photograph and biography, see this issue, p. 1930.