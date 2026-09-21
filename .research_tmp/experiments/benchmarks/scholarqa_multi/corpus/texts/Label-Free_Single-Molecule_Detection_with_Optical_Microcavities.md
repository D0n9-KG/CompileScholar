Label-free, single-molecule all-optical sensor Andrea M. Armani<sup>1</sup>, Scott E. Fraser<sup>1,2</sup>, and Richard C. Flagan<sup>3</sup>  
<sup>1</sup>Department of Applied Physics, California Institute of Technology  
<sup>2</sup>Division of Biology, California Institute of Technology  
sion of Chemistry and Chemical Engineering, California Institute of Techn  
1200 E California Blvd, Pasadena, CA 91125  
armani@caltech.edu, sefraser@caltech.edu, flagan@caltech.edu

# ABSTRACT

Recently, quality factors greater than 100 million were demonstrated using planar arrays of silica microtoroid resonators. These high Q factors allow the toroidal resonators to perform very sensitive detection experiments. By functionalizing the silica surface of the toroid with biotin, the toroidal resonators become both specific and sensitive detectors for Streptavidin. One application of this sensor is performing detection in lysates. To mimic this type of environment, additional solutions of Streptavidin were prepared which also contained high concentrations (nM and $\mu$ M) of tryptophan.

# 1. INTRODUCTION

While single molecule experiments have made significant advances in understanding protein folding kinetics[1], molecular transport,[2, 3] and aspects of DNA replication[4], all of these breakthrough discoveries required labeling the target molecule.[5, 6] In most experiments, this label behaves as an amplifier for an otherwise undetectable single molecule signal; however, it also restricts an experiment's scope, because there must be prior knowledge of the target's presence and the target molecule must be modified to incorporate the label.

[7-12] There have been several attempts to overcome this need to label the analyte by developing label-free sensing technologies, ranging from fiber optic waveguides[13] and nanowires[14] to nanoparticle probes[15], biochips[16] and mechanical cantilevers[17]; but none has achieved single molecule sensitivity.

Optical microcavities have successfully demonstrated label-free, single-molecule detection.[18] Sensitivity is inherent to ultra-high-Q microcavities because of the long photon lifetime within the microcavity which results in an increase in sampling or amplification of the signal without a label on the target molecule.[19] Additionally, microcavity-based detection can be performed in real-time, which allows for data to be taken continuously while other biologically relevant parameters (such as temperature, pH, salt) are changed. Specificity is endowed to the microcavity through surface functionalization.[20]

Previous microcavity detection experiments have been performed using a range of geometries and materials.[21, 22] Silica resonant sensors fabricated from high-Q microspheres (Q~2 million) have demonstrated the ability to distinguish between two strands of DNA and between cis/trans isomers based on a resonant wavelength shift in real time. [23, 24] The Q in these experiments was limited by the testing wavelength and was not a fundamental limit of the cavity. Polymer devices have also performed similar biological detection experiments.

Polymer microring resonators have demonstrated detection of glucose and bacteria.[25, 26] The techniques used to fabricate these devices enable integration and multiplexing.[27] Integrated polymer resonator sensors have also demonstrated detection of avidin[28] The quality factors of the polymer devices were limited by the fabrication methods used.[29, 30]

Optical Fibers and Sensors for Medical Diagnostics and Treatment Applications VIII, edited by Israel Gannot, Proc. of SPIE Vol. 6852, 68520A, (2008) $\cdot$ 1605-7422/08/ $\$ 18$ doi: 10.1117/12.761007

Proc. of SPIE Vol. 6852 68520A-1

Downloaded From: http://proceedings.spiedigitallibrary.org/ on 07/02/2016 Terms of Use: http://spiedigitallibrary.org/ss/TermsOfUse.aspx

Single molecule experiments using ultra-high-Q resonators have been previously proposed using a variety of detection techniques such as fluorescence [31], transmission variations [19] and polarizability changes[32]. However, these mechanisms assumed that the molecule was non-absorbing. As has been shown in previous theoretical and experimental studies, an optically absorbing monolayer will have significant effects on a microcavity's behavior [33-35].

These optical losses interact with the whispering gallery mode of the microcavity and, due to the high circulating intensities present in the microcavity, are amplified. The subsequent heating of the microcavity induces a resonant wavelength red-shift which can be described by the thermo-optic effect. This effect has not been previously proposed as a detection mechanism because of the incorrect assumption that biological molecules were non-absorbing.

The resonant wavelength shift that molecule produces is dependent on the optical absorption of the molecule, which is easily determined using a commercially available spectrophotometer, and on several other parameters, such as input power, Q and mode volume. In microcavity-based detection, the microcavity directly detects the molecule. This direct detection is in contrast to the previous single molecule experiments based on fluorescent labels, where the emission of light from the label is detected, not the molecule.

From finite element modeling (FEM) of microtoroid resonators, it has been shown that the majority of the optical field intensity (over $90\%$ ) resides within the silica. Additionally, the conductivity of water and silica are similar (0.6 and $1.38~\mathrm{W / m}~^\circ \mathrm{K}$ ). Taking both of these into account, the theoretical wavelength shift produced by a single bound molecule via the thermo-optic mechanism can be shown to be given by the expression below:

$$
\left[ \frac {\delta \lambda}{\lambda} \right] _ {S M} = \frac {\sigma \lambda^ {d n / d _ {T}}}{8 \pi^ {2} n ^ {2} \kappa V} Q P \int \frac {\left| u (r) \right| ^ {2}}{| r | + \varepsilon} d r \tag {1}
$$

where $\lambda$ is the wavelength, $\sigma$ is the absorption cross section of a single molecule, $dn/dT$ is the opto-thermal constant of silica $(1.3\times 10^{-5}\mathrm{K}^{-1})$ , $\kappa$ is thermal conductivity, $n$ is the effective refractive index of the silica toroid, $V$ is the optical mode volume, $Q$ is cavity Q-factor, and $P$ is the coupled optical power. The integral in this expression accounts for the spatial overlap of the whispering gallery mode field $(u(\mathbf{r}))$ with the temperature profile created by the nearly point-like molecular heat source.

The actual form of the temperature plume in the vicinity of the molecule is likely complex and has been combined into a single empirical parameter, $\varepsilon$ . In contrasting a perfect point source of heat with a molecule, this parameter captures the essential fact that the temperature profile is not singular at the source and instead rises steadily until reaching some radius of order the molecular size.

This approximation is justified first because the thermal transport process itself rapidly smoothes nano-scale spatial variations created by molecular shape, and second because the ensuing temperature field created by the molecular hot spot is long-range (i.e., $1 / \mathrm{r}$ dependence). For this reason, the tuning shift is only a weak function of the parameter “ $\varepsilon$ ”. In fact, a variation in “ $\varepsilon$ ” of $1 \mathrm{~nm}$ to $100 \mathrm{~nm}$ induces only a $16\%$ change in resonant wavelength shift.

Therefore, the optical cross section $\sigma$ is more significant to the thermo-optic induced heating that the physical radius, $\varepsilon$ . On the other hand, the size of “ $\varepsilon$ ” strongly suggests a maximum temperature in the vicinity of the molecule.

# 2. METHODOLOGY

To verify this effect, a single-mode, tunable external cavity laser centered at $681.5\mathrm{nm}$ was coupled to a single-mode tapered optical fiber waveguide. Tapered optical fibers are very low-loss/high-effici
ency waveguides

Proc. of SPIE Vol. 6852 68520A-2

Downloaded From: http://proceedings.spiedigitallibrary.org/ on 07/02/2016 Terms of Use: http://spiedigitallibrary.org/ss/TermsOfUse.aspx

used for probing ultra-high-Q modes in microcavities (Figure 1).[36] To create the testing chamber, the ultrahigh-Q microtoroids were placed on a high-resolution translation stage and were monitored by two cameras (top and side view) simultaneously. With the taper waveguide in close proximity to the microtoroid, pure water was added and a cover slip was placed on top, forming a water-filled microaquarium.[37] Solutions were injected into the aquarium and removed from the aquarium using a series of syringes at one end.

Both the intrinsic Q and resonant wavelength were determined by monitoring the power transmission spectra. The intrinsic Q factor was determined by scanning the wavelength of the single-mode laser and measuring both the resonant power transmission and the loaded linewidth (full-width-half-maximum) in the under-coupled regime. The intrinsic modal linewidth (and hence intrinsic Q) is then computed using a resonator-waveguide coupling model.

[36, 38] The position of the resonant frequency was determined by scanning the laser over a $0.03\mathrm{nm}$ range and recording the resonance position from an oscilloscope.

![](dt=2026-06-02/ht=18/04c9aff98d49cbe709a8dd1e054de7b2b4fcc89e453c11a9fe187d27d72c8ac1.jpg)

A Biotin surface functionalization was used (Figure 2). To detect Streptavidin, the surface of the toroid was functionalized with $0.1\mu \mathrm{M}$ of Biotin. The large dissociation constant $(\mathrm{K}_{\mathrm{D}})$ of the Streptavidin-Biotin bond has increased its popularity among biologists and biochemists, and it is commonly used to functionalize sensor surfaces.[39] Additionally, because antibodies can be easily biotinylated, this technique creates a "self-passivating" surface or one where only the antibody with the Biotin-tag on it binds to the surface.

Finally, studies have shown that the Biotin-Streptavidin pair correctly align and orient antibodies on a silica surface.[20] Therefore, this pair of functionalization techniques forms a foundation for a vast array of future experiments in this field.

Proc. of SPIE Vol. 6852 68520A-3

Downloaded From: http://proceedings.spiedigitallibrary.org/ on 07/02/2016 Terms of Use: http://spiedigitallibrary.org/ss/TermsOfUse.aspx

![](dt=2026-06-02/ht=18/ce193c48decd9bc8c9b08bfe96bac980e7bc7abad1fac39e68e8b97eacf84df4.jpg)

To perform single molecule measurements, a $3 \times 10^{-16} \mathrm{M}$ (300aM) solutions of the target molecule (Streptavidin) were used. At this concentration level, only a few molecular binding events on the whispering gallery are expected. As this solution was added, the resonance position was recorded using an automated data acquisition system until the $1 \mathrm{~mL}$ syringe was empty. The solution around the toroid was then cleansed by removing the ambient solution and replacing it with fresh water. At this concentration, single molecule detection experiments could be repeated numerous times on a single microtoroid.

To demonstrate that the microtoroid sensor's single molecule detection capabilities are not negatively impacted by the presence of additional materials, a set of complementary single molecule detection experiments were performed using 300aM Streptavidin solution containing additional Tryptophan (SigmaAldrich, $99.9\%$ pure L-Tryptophan) at either $1\mathrm{nM}$ or $1\mu \mathrm{M}$ . Tryptophan (Trp) is a commonly found amino acid in lysates.

While the Biotin surface functionalization may leave binding sites open on the surface of the toroid for the Trp, the toroid can overcome this limitation because of the detection mechanism. Unlike conventional techniques, such as fluorescence which detects a single signal, the toroid is continuously detecting the resonant wavelength and is providing information about its environment. Therefore, after the Biotin was physisorbed onto the toroid surface, the microtoroid was exposed to the Trp solution (1μM Trp).

Because testing was performed at 680nm which is significantly away from the fluorescent maximum of Trp (278nm), the binding of the Trp to the surface of the toroid did not significantly impact the Q factor or change the sensitivity of the toroid. Finally, the zero point is re-set and the single molecule detection of Streptavidin is performed.

Proc. of SPIE Vol. 6852 68520A-4

Downloaded From: http://proceedings.spiedigitallibrary.org/ on 07/02/2016 Terms of Use: http://spiedigitallibrary.org/ss/TermsOfUse.aspx

# 3. RESULTS

Figure 3 shows the resonance shifts which occurred as the microtoroid was exposed to the $300\mathrm{aM}$ Streptavidin solutions and the $1\mu \mathrm{M}$ Trp solution. Because testing was performed sufficiently away from the absorption maximum for Trp, the quality factor of the microtoroid was not impacted by Trp binding during the first injection.

It is important to compare the total resonance shift for each of the different solutions (Figure 3a). The total resonance shift is approximately the same, whether the toroid is exposed to pure Streptavidin or a Streptavidin solution containing additional Trp. The second injection of Trp induced a resonance shift that is negligible in comparison with the Streptavidin induced shifts and is of the same order of magnitude as noise-induced fluctuations.

![](dt=2026-06-02/ht=18/ec78847a04b397204298c2096be4ce961f9cce0e0b31bee8f2f861703a491dc8.jpg)

![](dt=2026-06-02/ht=18/3eebe85bace4ba13c97e402f2b86d4b2a9e75dc99ee265582808dd7dc26b553a.jpg)

Proc. of SPIE Vol. 6852 68520A-5

Downloaded From: http://proceedings.spiedigitallibrary.org/ on 07/02/2016 Terms of Use: http://spiedigitallibrary.org/ss/TermsOfUse.aspx

![](dt=2026-06-02/ht=18/8664779f86d05c76b854f110b4ab04b50bc0e72140b55e374029e8b2ef090c61.jpg)

![](dt=2026-06-02/ht=18/c40835ed4029b5505b6a8904226189764aa5778cb5deb0d14cec728b7e55ff20.jpg)

![](dt=2026-06-02/ht=18/0de8d7467ee6615f766f224516b07a534b9c2f8f5912f8d5aadbebec41fd10a0.jpg)

![](dt=2026-06-02/ht=18/f5714f44c9620c5859e68d3b8c2f9046db8ae63bea83bd796b0d233d7ba0222f.jpg)

The histogram showing the resonant wavelength shifts of the single molecule binding events is contained in Figure 4. The largest shift which occurred was the same in all of the solutions, except for the pure Trp solution. This value agrees very well with the theoretically predicted value based upon the toroid's Q factor and the absorption cross section of Streptavidin. In the pure Trp solution, only noise was recorded. In the histograms containing Streptavidin, shifts below $0.001\mathrm{pm}$ were considered noise and not included. Because all of the shifts in the Trp data were below $0.001\mathrm{pm}$ , these shifts were included the Trp histogram. It is also

Proc. of SPIE Vol. 6852 68520A-6

Downloaded From: http://proceedings.spiedigitallibrary.org/ on 07/02/2016 Terms of Use: http://spiedigitallibrary.org/ss/TermsOfUse.aspx

important to note that the number of molecules that bind is approximately constant, regardless of the amount of Trp in the solution.

# 4. CONCLUSION

In the present work, ultra-high-Q toroidal resonators have demonstrated label-free, single molecule detection of Streptavidin using a Biotin surface fu
nctionalization. The proposed thermo-optic detection mechanism was also verified.

Additional experiments were performed in more complex environments to explore the microtoroid sensor's sensitivity in a more realistic environment. These experiments in the presence of high concentrations of Tryptophan, the dominant component of lysates, demonstrated that the sensor's single molecule detection capabilities are not significantly affected. The experimental resonant wavelength shifts were in excellent agreement with the thermo-optic mechanism.

Future work will focus on integration and improvements to surface functionalization.

# ACKNOWLEDGEMENTS

The authors would like to thank Dr. Rajan Kulkarni for numerous helpful discussions. A.M. Armani is supported by a Clare Boothe Luce Post-doctoral Fellowship. This work was supported by the DARPA Center for OptoFluidic Integration.

# REFERENCES

Proc. of SPIE Vol. 6852 68520A-7

Downloaded From: http://proceedings.spiedigitallibrary.org/ on 07/02/2016 Terms of Use: http://spiedigitallibrary.org/ss/TermsOfUse.aspx

Proc. of SPIE Vol. 6852 68520A-8

Downloaded From: http://proceedings.spiedigitallibrary.org/ on 07/02/2016 Terms of Use: http://spiedigitallibrary.org/ss/TermsOfUse.aspx

Proc. of SPIE Vol. 6852 68520A-9

Downloaded From: http://proceedings.spiedigitallibrary.org/ on 07/02/2016 Terms of Use: http://spiedigitallibrary.org/ss/TermsOfUse.aspx