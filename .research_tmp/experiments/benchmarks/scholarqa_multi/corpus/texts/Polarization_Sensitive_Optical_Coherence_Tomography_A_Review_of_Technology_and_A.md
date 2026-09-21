Review

# Polarization Sensitive Optical Coherence Tomography: A Review of Technology and Applications

Bernhard Baumann

Medical University of Vienna, Center for Medical Physics and Biomedical Engineering, Waehringer Guertel 18-20, 4L, 1090 Vienna, Austria; bernhard.baumann@meduniwien.ac.at; Tel.: +43-1-40400-73727

Academic Editor: Totaro Imasaka  
Received: 10 February 2017; Accepted: 25 April 2017; Published: 4 May 2017

Abstract: Polarization sensitive optical coherence tomography (PS-OCT) is an imaging technique based on light scattering. PS-OCT performs rapid two- and three-dimensional imaging of transparent and translucent samples with micrometer scale resolution. PS-OCT provides image contrast based on the polarization state of backscattered light and has been applied in many biomedical fields as well as in non-medical fields. Thereby, the polarimetric approach enabled imaging with enhanced contrast compared to standard OCT and the quantitative assessment of sample polarization properties. In this article, the basic methodological principles, the state of the art of PS-OCT technologies, and important applications of the technique are reviewed in a concise yet comprehensive way.

Keywords: optical coherence tomography; polarization sensitive devices; biomedical imaging; birefringence; scattering; depolarization

# 1. Introduction

Optical coherence tomography (OCT) is an imaging modality providing 2D and 3D images with micrometer scale resolution [1-3]. Often considered an optical analog of ultrasound imaging, OCT detects light backscattered from sample structures. However, since the speed of light is much greater than that of sound, subtle differences in time delays corresponding to optical path lengths from different scatter locations within the sample cannot easily be measured in a direct way.

For instance, the time delay corresponding to an optical path length of $10\mu \mathrm{m}$ is only on the order of $\sim 30$ fs. In order to assess such short delay times, OCT employs the interference of low coherent light [4,5]. Low-coherent light sources span a broad wavelength range, usually covering several tens of nanometers when used for OCT. Before the light interacts with the sample, it is split into two.

One portion is directed onto the sample, while the other portion—the so-called reference beam—travels a defined path length before being recombined and interfered with the light beam scattered by the sample. Now, since low-coherent light consists of a continuum of wavelengths, the interference spectrum will be subject to modulations which depend on the path length difference between the sample beam and the reference beam. From the interference signal in the time or spectral domain, the axial position of scattering structures within the sample can be reconstructed [4,5].

The respective depth profile (backscatter intensity vs. depth) is called an axial scan (A-scan) and forms the basic unit of OCT images. By scanning the beam laterally across the sample, two- and three-dimensional images can be assembled from the acquired A-scans. OCT is a rapid imaging method providing 3D data comprising up to several millions of axial scans within few seconds [6].

The axial resolution of OCT is usually in the order of a few micrometers and, despite the high imaging speeds, the interferometric approach enables detection sensitivities of reflected light signals as low as $10^{-10}$ of the input.

applied sciences

MDPI

Appl. Sci. 2017, 7, 474; doi:10.3390/app7050474

www.mdpi.com/journal/applsci

During the past 25 years since its invention, OCT experienced a multitude of technological advances leading to higher imaging speeds, improved resolution, and novel contrast mechanisms. One so-called functional extension of OCT is polarization sensitive (PS) OCT. PS-OCT adds polarization contrast to the technique. While the standard OCT is based solely on the intensity of light backscattered or reflected by the sample, PS-OCT also detects its polarization state.

Since the polarization state can be measured for every pixel in a depth scan, PS-OCT can enhance the image contrast and also enables quantitative measurements of a sample's polarization properties. These properties are often linked to the micro- or even ultrastructure of the sample which themselves are below the optical resolution limit of OCT [7].

Hence, the detection of changes of polarization properties—be they due to disordered microstructure in pathological tissue or due to stress and strain in a technical sample—may provide access to quantities and markers that are of interest for a broad variety of applications.

PS-OCT has been applied in many biomedical as well as non-medical fields. Figure 1 shows results of a search for scientific publications on PS-OCT using the free literature search engine PubMed (https://www.ncbi.nlm.nih.gov/pubmed/). In Figure 1a, the number of publications per year was plotted beginning with the first article on PS-OCT published in 1997 [8]. A rising trend of published documents per year can be observed, peaking with more than 30 publications per year for the last three years. The set of 360 publications was classified into seven medical fields and one non-medical field.

The latter included articles primarily focusing on technological aspects or on measurements of technical (i.e., non-biological) samples. Figure 1b shows a pie chart with the break-up into the eight fields. The most prominent field was ophthalmology including roughly a third of all publications, followed by the 'technical' publications, reports on dental applications, PS-OCT in bones, cartilage, muscles and tendons, and skin imaging. Further fields of application were imaging of cardiac and vascular tissue, of cancerous tissue, and of neural tissue.

The evolution of the eight groups over the past 20 years is shown in Figure 1c. Here, an increasing number of research papers on PS-OCT in the eye during the last decade can be observed. The relative share of publications grouped in the respective fields is shown in Figure 1d.

![](dt=2026-04-21/ht=00/fa142f19a9b1731a53eac0b66bd5964267e529bda0f476f956f64c653b15191d.jpg)

![](dt=2026-04-21/ht=00/dda11c6bbe7304eca0882824a829efa8b353c77bc2c6fece5152b3ce8c3a1a12.jpg)

![](dt=2026-04-21/ht=00/8935d5532cba0a355a8ec08d4a0c47ad8f51c50a749b54ba9ca4ff609390994e.jpg)

![](dt=2026-04-21/ht=00/8a132192bbcf8dc67749a555d5ec3397defb637dbcc0f955108960d5cf81189f.jpg)

Appl. Sci. 2017, 7, 474

2 of 34

This review paper strives to provide an overview of the state of the art in PS-OCT. While other review papers provided a more in-depth discussion of the physical principles of the technique [9,10], this article is particularly focusing on applications of PS-OCT. In the following, we will first introduce basic concepts for describing polarization of light as well as commonly used technical approaches for realizing PS-OCT. Then, using the eight categories mentioned above, we will discuss important applications of PS-OCT in the biomedical field and for non-biomedical use.

# 2. Principles of Light Polarization and PS-OCT

# 2.1. Polarization of Light

Polarization of light describes the geometrical orientation of the oscillations of electromagnetic waves. When considering a light beam as a transverse wave propagating in $z$ -direction, its electric field vector and magnetic field will not only be perpendicular to each other but also be perpendicular to $z$ . At any spatial coordinate and time point, such a wave can be characterized by the complex-valued field components $e_{x,y}$ describing the oscillations in $x$ - and $y$ -direction, respectively.

For a
monochromatic plane wave travelling in $z$ -direction, these two complex-valued components form the so-called Jones vector [11]. Depending on the amplitudes of $e_x$ and $e_y$ and on their respective phase delay $\delta$ , different states of polarization will be observed, as illustrated in Figure 2a. A Jones vector having a relative delay of $0^{\circ}$ or $180^{\circ}$ (i.e., half a wave) describes a linear polarization state. In case this linear state oscillates only in $x$ -direction or only in $y$ -direction, it is termed a horizontal or vertical linear state, respectively.

When the $x$ - and $y$ -amplitudes are equal, the state is linear with an orientation of $+45^{\circ}$ for $\delta = 0^{\circ}$ and linear with an orientation of $-45^{\circ}$ for $\delta = 180^{\circ}$ . If the amplitudes in $x$ - and $y$ -direction are equal but $\delta$ is a quarter of a wave (i.e., $\delta = \pm 90^{\circ}$ ), the Jones vector will describe a right- or left-hand circular polarization state. In the general case of arbitrary $\delta$ and amplitudes, the wave will be in an elliptical polarization state.

As an alternative to describing polarization states by Jones vectors, a three-dimensional space spanned by horizontal/vertical linear state, $+45^{\circ} / -45^{\circ}$ linear state, and right-/left-hand circular state can be used (Figure 2b). Four-component, real-valued vectors $[I Q U V]^T$ , so-called Stokes vectors, describe polarization states in this space [12]. Here, $I$ corresponds to the intensity of light, and $Q$ , $U$ , and $V$ are the components along the three above-mentioned axes. Unlike Jones vectors, Stokes vectors enable the characterization of light depolarization.

In case of fully polarized light, $I$ corresponds to the length of the vector $[Q U V]$ , i.e., $I = \sqrt{Q^2 + U^2 + V^2}$ . Then, the light's degree of polarization $DOP = \sqrt{Q^2 + U^2 + V^2} / I$ equals unity. In case of depolarization, DOP is less than unity, and equals zero for completely depolarized light. The unit sphere in Figure 2b is called a Poincaré sphere.

In order to describe the interaction of light with an optical element (or a sample investigated by a PS-OCT system), an operator acts on the polarization vector of the interrogating light beam. In Jones calculus, this operator is a complex-valued $2 \times 2$ matrix called Jones matrix $(J)$ [11]. The evanescent light beam is represented by $e_{out} = J e_{in}$ , where $e_{in}$ is the input Jones vector (Figure 2c).

If the light beam traverses several optical elements (or sample structures), the resulting Jones vector can be calculated by multiplying a cascade of Jones matrices to the input vector, $J_{N}J_{N-1} \cdots J_{2}J_{1}e_{in}$ where $J_{1}$ through $J_{N}$ represent the polarization properties of $N$ elements (or sample layers). Analogously, real-valued $4 \times 4$ matrices—so-called Müller matrices $(M)$ —are used to describe the interaction of Stokes vectors with optical elements by the Stokes-Müller formalism: $S_{out} = MS_{in}$ (see Figure 2d) [12].

Different approaches of PS-OCT enable the pixelwise measurement of Jones vectors, Jones matrices, Stokes vectors, or Müller matrices. Since the display and interpretation of these multidimensional quantities is often not straightforward (or even impossible), PS-OCT imagery usually displays physical polarization measures directly related to relevant sample properties.

Appl. Sci. 2017, 7, 474

3 of 34

![](dt=2026-04-21/ht=00/a1362ed371057f93f4953f2b201fbb4f2fff8e924433d4dc6d0c9c36b6838cc9.jpg)

![](dt=2026-04-21/ht=00/88c43cd4e16436b9740fc44d175911ff1246103826e78489ab3f34c747063eb3.jpg)

![](dt=2026-04-21/ht=00/51c96d3b3a79a5a62341e47b36f78d81578c0fe72ba00aa613887392d5e0de91.jpg)

![](dt=2026-04-21/ht=00/dbe01a0df8327f7fb73fe4954f4d1f9939b4c44365cda7cbd71669943ef287ba.jpg)

# 2.2. Polarization Effects

The polarization state of a light beam can be affected by interaction with optical components or sample structures [7]. In the following, four polarization effects will be described which are relevant for PS-OCT imaging.

Appl. Sci. 2017, 7, 474

4 of 34

quarter wave plate (QWP) for example amounts to $90^{\circ}$ ( $\pi/2$ rad). If aligned with its slow and fast axes at $45^{\circ}$ with respect to the $x$ - and $y$ -axes, the QWP would render a horizontal or vertical linear state into a circular polarization state (Figure 2a, center). The Jones matrix of a retarder with a retardation $\delta$ and aligned with an orientation $\vartheta = 0$ is represented by

$$
J _ {r e t} (\delta , \vartheta = 0) = \left[ \begin{array}{c c} e ^ {i \delta / 2} & 0 \\ 0 & e ^ {- i \delta / 2} \end{array} \right] \tag {1}
$$

If an optical element such as a retarder is rotated by an angle $\vartheta$ , its Jones matrix $J = J(\vartheta = 0)$ is transformed into $T(\vartheta)JT(-\vartheta)$ where $T(\vartheta)$ is a rotation matrix. In Stokes-Müller formalism, the propagation of light through birefringent tissue is represented by a circular rotation of the Stokes vector tip on the Poincaré sphere. PS-OCT approaches based on Jones calculus or Stokes-Müller formalism enable depth-resolved measurements of phase retardation and of birefringent axis orientation [8,13-16].

- Diattenuation. Diattenuation (or dichroism) refers to a polarization dependent attenuation in an optical medium. When the axes of a diattenuating optical element or structure are aligned with the $x$ - and $y$ -direction, its Jones matrix is represented by

$$
J _ {d i a t t} \left(p _ {1}, p _ {2}, \vartheta = 0\right) = \left[ \begin{array}{c c} p _ {1} & 0 \\ 0 & p _ {2} \end{array} \right] \tag {2}
$$

where $p_1$ and $p_2$ correspond to the respective signal attenuation $p_{1,2} = \exp \left(-\mu_{a_{1,2}}L\right)$ with attenuation coefficients $\mu_{a}$ and the length $L$ . Similar to the linear birefringence, the Jones matrix for a diattenuating element oriented at $\vartheta$ can be computed by sandwiching Equation (2) by rotation matrices. An extreme case of a diattenuating element with $p_1 = 1$ and $p_2 = 0$ is a linear polarizer which only transmits light along axis 1. It should be noted that diattenuation in biological tissue is usually very weak and thus is often assumed negligible for PS-OCT [16,17], although attempts have been made to quantify diattenuation using PS-OCT [15,17,18].

- Depolarization. Depolarization or polarization scrambling refers to a more or less random change of the incident polarization state at spatially adjacent sample locations. Using Stokes-Müller polarimetry, depolarization can be described by the DOP discussed in Section 2.1. However, owing to the coherent detection in OCT, DOP will always equal unity in any pixel of a PS-OCT dataset [9]. Therefore, in order to analyze polarization scrambling using PS-OCT, the randomization of polarization states among neighboring speckles is investigated [19,20].

The Stokes vectors of adjacent speckles will be more or less parallel in polarization preserving or weakly birefringent media, while they will point in different directions in depolarizing media (Figure 3a). For the purpose of depolarization assessment, the average Stokes vector can be calculated within a small kernel including several speckles (Figure 3b). Typical kernel sizes for this calculation are on the order of $\sim 100$ pixels spanning 2-3 times the axial resolution in depth and 2-3 times the transverse resolution laterally [20]. Note also that pixels with low reflectivity (i.e.

, less than several decibels above the noise floor) are usually excluded from the analysis. The length of the average normalized Stokes vector is refe
rred to as the degree of polarization uniformity (DOPU) [20]:

$$
D O P U = \sqrt {\bar {Q} ^ {2} + \bar {U} ^ {2} + \bar {V} ^ {2}} / \bar {I} \tag {3}
$$

where the overbar indicates the ensemble average and $\overline{I}$ denotes the average Stokes vector length for normalization. As shown in Figure 3b, DOPU or the average Stokes vector length will be close to unity in polarization preserving tissue, while it will be lower in the case of polarization scrambling where the orientation of the Stokes vectors is more diverse. In DOPU images, DOPU values at every spatial coordinate are color-coded. Recently, advanced depolarization measures

Appl. Sci. 2017, 7, 474

5 of 34

have been developed including DOPU with noise floor normalization [21], spectral DOPU [22], the depolarization index independent of the incident polarization state [23], and the differential depolarization index providing a larger dynamic range for depolarization mapping [24]. Different mechanisms can cause polarization scrambling, and the actual cause of depolarization observed in DOPU images has not always been completely clarified. In biological tissues, depolarization can be caused by multiple scattering or by scattering at non-spherical particles such as melanin granules [25].

Since the strength of depolarization is proportional to the concentration of polarization scrambling scatterers, depolarization measures such as DOPU may for instance enable a quantitative assessment of the melanin concentration in ocular tissues [26].

![](dt=2026-04-21/ht=00/6cab22a8cdc37f23125e42b5fa7172cbc4ab0ceef2d9f869595e9de2ead15fdb.jpg)

![](dt=2026-04-21/ht=00/c0cd0154add3c0cccb617daac948e4101568e524bccaed59c1af49f95698bf4a.jpg)

In general, polarization effects may be subject to dispersion, that is, their strength depends on wavelength. Since OCT is based on broadband light covering a wide range of wavelengths, efforts have been made to mitigate effects such as polarization mode dispersion in PS-OCT systems based on optical fibers [18,27-30].

Appl. Sci. 2017, 7, 474

6 of 34

# 2.3. Brief Basics of OCT

OCT is based on low coherence interferometry, i.e., the interference of broad band light [1,4]. A multitude of different interferometer designs have been used for OCT. A sketch of a basic Michelson interferometer is shown in Figure 4a. Here, light from a low coherent light source such as a superluminescent diode or a broadband laser is split into one beam that is directed on the sample and another beam that serves as a reference. After the beam splitter, the beam in the sample arm is directed onto the sample, whereas the reference beam is reflected by a mirror.

Light backscattered and reflected by the sample $e_{S}$ and light reflected by the reference mirror $e_{R}$ is recombined at the beam splitter. The sample and reference beam interfere and their interference signal is detected at the interferometer exit. The interference signal in the time domain can be described by

$$
I (z) = I _ {R} + I _ {S} + 2 \sqrt {I _ {R} I _ {S}} | \gamma (z - z _ {0}) | \cos [ 2 k _ {0} (z - z _ {0}) ]. \tag {4}
$$

![](dt=2026-04-21/ht=00/b18a084622e968ead767b5b0886685b9de9875fcff992426248bf63815c49b05.jpg)

![](dt=2026-04-21/ht=00/21c8597d8bccb31dff4250cea815a75b35bcdec460363dcc7e0a63458586d284.jpg)

![](dt=2026-04-21/ht=00/5c99c5e7c758a3248886d36fc6ab3a8af5e655f67d7af5a625a2064215cb968c.jpg)

![](dt=2026-04-21/ht=00/6f156e675abd1b024f729a3f2898ea254c0b7ed2542c74c6119cf93c0a729adc.jpg)

Here, $I_R \sim |e_R|^2$ is proportional to the intensity of the reference beam and $I_S \sim |e_S|^2$ is proportional to the intensity of light backscattered or reflected by the sample. The third term contains the interference information. Its first component $\sqrt{I_R I_S}$ indicates that the strength of the interference signal will scale with both the sample and the reference amplitude. The high sensitivity of OCT is based on the fact that even weak light scatter signals $I_S$ from the sample can be amplified by a strong

Appl. Sci. 2017, 7, 474

7 of 34

reference signal $I_{R}$ . The second component $|\gamma (z - z_0)|$ contains the complex degree of coherence $\gamma (z)$ which is inversely related to the spectral bandwidth of the light source (via the Fourier transform of the spectral density). For broad bandwidth sources common in OCT, $|\gamma (z)|$ will only be greater than zero for a shallow depth around every scattering sample interface along the propagation direction $z$ . Hence, the high resolution of OCT is contained in this term.

The third component $\cos [2k_0(z - z_0)]$ describes a sinusoidal modulation of the interference signal along $z$ . Here $k_{0} = 2\pi /\lambda_{0}$ is the central wavenumber of the spectrum. In order to acquire depth scans in time domain OCT, the signal intensity $I(z)$ is recorded at the interferometer exit while the reference mirror is axially translated in beam direction $z$ .

Most modern OCT systems rely on frequency (or Fourier) domain detection of the interference signal [5,31]. In such Fourier domain OCT systems, the reference mirror position is fixed and the interference spectrum is acquired. This can be achieved by using a spectrometer at the interferometer exit which disperses the interference signal into its spectral intensity components. Alternatively, a broadband wavelength-swept light source can be used to rapidly tune the spectrum with a narrow instantaneous bandwidth. In this case, the interference spectrum is recorded as a function of time by a detector at the interferometer exit. For each interface, the acquired spectral interference signal

$$
S (k, \Delta z) = S _ {R} (k) + S _ {S} (k) + 2 \sqrt {S _ {R} (k) S _ {S} (k)} \cos [ 2 \Delta z k ]. \tag {5}
$$

contains three terms, similar to Equation (4). The first two terms contain the spectral densities returning from the reference arm $(R)$ and the sample arm $(S)$ . Via $\sqrt{S_R(k)S_S(k)}$ , the last term again is proportional to the spectral densities of the reference beam and the sample beam. Note that the last term is subject to a modulation $\cos [2\Delta zk]$ across wavenumber $k$ , whose modulation frequency is proportional to the path length difference between the light path to sample interface and the light path to the reference mirror, $\Delta z = z_{S} - z_{R}$ .

The factor 2 accounts for the double pass through the interferometer arms. Since every path length difference $\Delta z$ is encoded by a different spectral modulation frequency, the interference signals from multiple depth locations can be recorded simultaneously. A frequency analysis using the Fourier transform then provides the axial depth scan similar to Equation (4).

# 2.4. Technical Approaches to PS-OCT

During the past 25 years, a great variety of PS-OCT layouts has been devised. PS-OCT schemes differ in terms of optical technology (fiber optics vs. bulk optics), number of input states, number of detected variables, and reconstruction algorithm. The use of free-space beams in bulk optics permits defined polarization states at any location within the interferometer. Fiber optics provide easier system alignment, but the polarization of light will in general be influenced by birefringence and polarization mode dispersion in optical fibers.

PS-OCT has been performed with as little as one input state and one detected intensity signal. Such settings correspond to regular OCT, however with altered reference polarization for cross-polarization imaging
[32,33] or for imaging with variable reference polarization [34]. In contrast, the most comprehensive PS-OCT approaches detected up to 16 elements of the sample's Müller matrix—in every single image pixel [35-38].

In the following, we are going to describe two major categories of PS-OCT schemes: PS-OCT with a single circular input stage and PS-OCT based on sample illumination by multiple polarization states.

# 2.4.1. PS-OCT with a Single Circular Input State

PS-OCT with a single circular input state relies on a polarization sensitive low coherence interferometer design devised by Hee et al. in 1992 [13]. The basic scheme using a Michelson interferometer is shown in Figure 4b. Light from a low coherent light source is linearly polarized before being split up into a reference arm (top) and a sample arm (right). In the sample arm, the beam passes a QWP oriented at $45^{\circ}$ which renders the original linear polarization into a circular polarization state and then illuminates the sample. Sample illumination by circular light offers sensitivity to any transverse orientation of birefringent media. If linearly polarized light were used for sample

Appl. Sci. 2017, 7, 474

8 of 34

illumination, the sample's fast or slow birefringent axis could align with the interrogating linear polarization such that no birefringence would be observed. In the case of circular sample illumination shown in Figure 4b, a birefringent sample will in general produce an elliptical state. The reflected or backscattered light will transmit the QWP again and interfere with the reference beam at the beam splitter.

Due to double passing a QWP oriented at $22.5^{\circ}$ in the reference arm, the reference beam is a linearly polarized light beam oscillating at $45^{\circ}$ which provides equal intensity components in the horizontal and vertical orientation, respectively. At the interferometer exit, the OCT light beam is split up into its horizontal (H) and vertical (V) component, which are detected by separate detection units. By the respective amplitudes $A_{H,V}$ and the relative phase difference $\Delta \Phi$ , Jones vectors are detected for every image pixel.

These Jones vectors enable the calculation of the sample's birefringent properties, namely of phase retardation $\delta$ [13,39] and fast birefringent axis orientation $\vartheta$ [14] as well as sample reflectivity $R$ :

$$
R \propto A _ {H} ^ {2} + A _ {V} ^ {2} \tag {6}
$$

$$
\delta = \arctan \left(\frac {A _ {V}}{A _ {H}}\right) \tag {7}
$$

$$
\vartheta = \frac {\pi - \Delta \Phi}{2} \tag {8}
$$

Since $\delta = \Delta n\cdot L$ accumulates as a function of the light path $L$ travelled in a birefringent material, phase retardation measurements are cumulative. However, the measurement of $\delta$ is restricted to $0 - 90^{\circ}$ due to the arctangent, which leads to cumulative retardation images with a banded structure caused by increasing and artificially decreasing $\delta$ in strongly birefringent samples (cf. Figures 9 and 10). From the detected amplitudes $A_{H,V}$ and the relative phase difference $\Delta \Phi$ , the Stokes vector elements can also be calculated for every image pixel [19]. These may then serve as the input for depolarization images, for instance based on DOPU [20].

The beauty of the above scheme lies in its simplicity. Most implementations were done using free-space optics [14,40-44], however fiber optic prototypes have also been reported based on polarization maintaining (PM) fiber optics [45-52] and regular single mode fibers [53-55].

# 2.4.2. PS-OCT Based on Multiple Input States

PS-OCT systems using multiple polarization states as an input may provide access to additional polarization quantities. The scheme described in Section 2.4.1 is based on a single circular input state and relies on the assumptions that the sample is not diattenuating (which is a valid assumption for most biological tissues [17,56]) and that the axis orientation of the birefringent structure does not change along depth [43,44].

A method to overcome the latter limitation for retinal PS-OCT has been developed to remove the impact of corneal birefringence on birefringence measurements in the back of the eye [57]. Nevertheless, for applications such as PS-OCT in samples with strongly varying birefringent fiber orientations or for many approaches based on single-mode fiber optics, implementations based on multiple polarization states can provide access to Stokes vector quantification, Jones matrix characterization, and Müller matrix measurements [15-17,35-38,58-63].

In order to provide measurements of several polarization states, different approaches have been proposed, only a few of which are described here. By adding a polarization modulator (e.g., an electrooptic modulator) in the source arm, different input states can be produced in a sequential manner. In a commonly used scheme depicted in Figure 4c, a consecutive pair of polarization states corresponding to Stokes vectors perpendicular in a Poincaré sphere representation is generated at the input of the interferometer.

From the polarization states detected at the output of the interferometer, depth-resolved Stokes vectors can be computed [64,65]. The retardation induced by birefringent tissue is related to the angle of rotation of Stokes vectors on the Poincaré sphere. By computing this angle between the Stokes vectors at the sample surface and those within the tissue, cumulative phase retardation can be computed at any sample position [66,67]. Furthermore, the direction of the optic

Appl. Sci. 2017, 7, 474

9 of 34

axis can be determined from rotations of the pair of perpendicular Stokes vectors on the Poincaré sphere [68]. In such dual-input PS-OCT systems, polarization parameters such as retardation, axis orientation, and di attenuation can also be assessed using the Jones formalism. In this approach, the measured polarization states in the sample originating from the pair of input polarization states are used to reconstruct the Jones matrix in every image pixel. An eigenvalue analysis of these measured Jones matrices enables the calculation of phase retardation and di attenuation [15,16].

The Jones matrix approach to PS-OCT also enables the measurement of the optic axis orientation [15,16]. Depth-resolved measurements of the birefringent axis orientation have recently gained interest for mapping the orientation of birefringent fibers in PS-OCT based tractography of collagenous tissue [69,70].

Alternatively, using a Mach-Zehnder type interferometer, only the polarization state in the sample arm can be varied [71,72] from one scan to the next. By multiplexing two different states using a passive polarization delay unit as shown in Figure 4d, the four elements of a Jones matrix can be measured simultaneously [18,73,74]. In that case, the sample beam is split into two orthogonal input Jones vectors which travel different path lengths in the sample arm and therefore generate signals at different depths in the OCT image.

These two input vectors provide an orthogonal system and their response—i.e., the two Jones vectors measured via multiplexing— readily provides a Jones matrix [15]. Jones matrix OCT relates the Jones matrix $J_{meas}$ measured at each sample position to a reference matrix (e.g., $J_{surface}$ at the surface of the sample), thereby yielding a unitary transformation of the sample matrix $J_{sample}$ [15,16,75]

$$
\widetilde {J} _ {\text {s a m p l e}} = J _ {\text {m e a s}} J _ {\text {s u r f a c e}} ^ {- 1}. \tag {9}
$$

Here the tilde denotes the unitary transformation. From $\widetilde{J}_{\text {sample }}$ , the polarization properties can be computed. As such, Jones matrix PS-OCT can not only measure phase retardation but also diattenuation [15], local birefringence [16], and local optic axis orientation [76]. Compared to cumulative retardation measurements, local measurements of birefringent properties provide a more intuitive a
pproach to tissue architecture and composition. For instance, in collagenous tissue such as skin, local birefringence can be used for the depth-resolved assessment of the collagen content [77,78]. Different applications of birefringence imaging of collagen in healthy and diseased tissues will be discussed in Section 3.

# 2.5. Recent Advances in PS-OCT Technology

The development of PS-OCT has greatly advanced since Hee and coworkers first presented birefringence-sensitive ranging [13]. Not only have PS-OCT devices become faster and the detection schemes become more sophisticated, as briefly described in the previous section, but also the analysis of PS-OCT images has improved a lot.

The first PS-OCT prototypes provided axial scan rates on the order of several hertz [8,13,14]. Later rapid reference scanning schemes [79,80] and advanced beam scanning approaches such as transverse-scanning PS-OCT [40] sped up the technique to several frames (B-scans) per second.

The advent of Fourier domain OCT (or: frequency domain OCT), which computes A-scan signals by a Fourier transform of the interference spectrum, provided a huge increase in detection sensitivity and the possibility to scan even faster since no more mechanical reference mirror movement was required to perform depth scanning [5,81-83]. First high-speed PS-OCT systems with spectrometer-based detection provided scan rates of several tens of A-scans per second [41,68]. These spectral domain (SD) PS-OCT prototypes employed two spectrometer cameras, one for each orthogonal polarization channel.

In order to reduce system complexity, cost, and alignment efforts, SD PS-OCT approaches based on single camera detection were developed [42,49,84-89].

Fourier domain OCT can also be performed by using a frequency-swept laser and a high-speed detector, such that interference spectra are acquired as a function of time rather than in parallel with a spectrometer [90-92]. This variant of OCT is usually called swept-source (SS) OCT and sometimes also referred to as optical frequency domain imaging (OFDI) or time-encoded frequency domain OCT.

Appl. Sci. 2017, 7, 474

10 of 34

Providing the same sensitivity and speed advantages as spectrometer-based Fourier domain OCT, SS-OCT was soon expanded by polarization sensitivity [47,71,75,93-97]. In particular the advent of commercial laser technology providing longer imaging ranges led to the development of PS-OCT at ultrahigh imaging speeds of 100,000 axial scans per second [18,54,73,74,98]. Even higher imaging speeds were achieved by experimental swept lasers operating at several hundred kilohertz [50,99-102].

While PS-OCT technology has greatly advanced, there are still some limitations to the technique. Being an optical method, its applicability is limited to imaging of superficial locations in tissues and other objects. Further, PS-OCT has been used for qualitative imaging mostly; the exploitation of quantitative measurements however bears great potential for diagnostics and other applications, as will be demonstrated in the next sections. PS-OCT was also combined with other functional OCT extensions such as Doppler OCT or OCT angiography [30,68,80,103-105].

Such combinations may not only improve the contrast for vascular tissue components but also provide additional, complementary insight into disease patterns [30,103,105,106]. In order to perform PS-OCT beneath the body surface, endoscopic and needle-based PS-OCT was developed [28,29,107-112]. To further increase the contrast and image range, PS-OCT has been combined with other technologies such as ultrasound and fluorescence imaging [113,114].

In parallel to the impressive evolution of PS-OCT hardware, PS-OCT image processing also underwent massive improvements. Real-time display of PS-OCT data was enabled by parallel computing [115]. Computational methods were devised for removing polarization artifacts in order to produce clearer PS-OCT images [21,27-29,57,116,117]. As PS-OCT is an interferometric technique based on coherent light, images are subject to speckling which sometimes obscures structural details.

The size of speckles can be kept small by using broadband light sources and optics providing high transverse resolution [46,118]. In image processing, speckle noise can be reduced by image averaging and dedicated algorithms [119,120]. PS-OCT also enables the segmentation of structures based on common polarization properties and the determination of interfaces between different tissue segments based on changing polarization properties. Segmentation and image feature assessment was developed based on depolarization [20,103,121-125] and birefringence [98,116,126-129].

Practical examples of PS-OCT applications will be shown in the following sections.

# 3. PS-OCT Applications

# 3.1. PS-OCT in the Eye

OCT is most established in ophthalmology, where it has become a standard diagnostic method in everyday clinical routine [130]. Also PS-OCT has been successfully applied for ophthalmic imaging using experimental prototypes [131]. The eye features a variety of tissues exhibiting birefringence or depolarization, which enable PS-OCT to provide additional contrast for discerning, segmenting, and quantifying ocular structures. Birefringence can be found in fibrous tissues such as the retinal nerve fiber layer (RNFL), the sclera (i.e.

, the white outer shell of the eye), the cornea, as well as in extraorbital muscles and tendons. Depolarization is pronounced in structures containing melanin pigments such as the retinal pigment epithelium (RPE), the choroid, and the pigment epithelium of the iris. Other structures such as the photoreceptor layer, conjunctive tissue, and the stroma of the iris are rather polarization preserving and do not markedly influence the polarization state of light.

The RNFL consists of the axons of the retinal ganglion cells. Since the RNFL is damaged in glaucoma—the second leading cause of blindness worldwide [132]—and since RNFL birefringence is connected to layer integrity [133,134], the polarization properties of the RNFL were investigated as potential diagnostic markers for glaucoma. PS-OCT based assessment of the RNFL's birefringent properties might be particularly interesting since it was shown that polarization changes in experimental glaucoma can be observed earlier than RNFL thickness changes [135]. Peripapillary RNFL thickness is currently a key OCT parameter for glaucoma diagnostics in state-of-the-art clinical routine [136]. In the vein of earlier scanning laser polarimetry approaches [137-139], PS-OCT was

Appl. Sci. 2017, 7, 474

11 of 34

applied to investigate the RNFL using PS-OCT. After initial experiments in the primate retina [140], RNFL birefringence was measured in vivo in the human eye by performing circular scans around the optic nerve head [79,141]. Faster Fourier domain PS-OCT later enabled 2D mapping of RNFL birefringence and retardation as well as comparisons to scanning laser polarimetry [18,73,142-145]. Exemplary PS-OCT fundus images mapping reflectivity and RNFL retardation in a human eye are shown in Figure 5.

Also in preclinical research, PS-OCT was used to investigate the birefringence properties of the RNFL and their relation to the intraocular pressure, which is an important parameter for glaucoma, in animals [135,146-148]. Aside from measuring their birefringence, PS-OCT was also demonstrated for tracing nerve fiber bundles in the RNFL [149].

PS-OCT images of the human retina exhibit strong depolarization in pigmented structures such as the RPE [150,151]. In the RPE, this depolarization is most pronounced around the fovea [152] and correlates with the pigmentation status, i.e., it is reduced or even absent in albino patients [25,153]. Comparative measurements of PS-OCT and histology in rat eyes have revealed a correlation between DOPU and the density of melanin pigments in the RPE and choroid (Figure 5e) [26].

Depolarization has proven a particular
ly useful contrast for the assessment of the RPE in clinical cases, where it is often hard to distinguish ocular structures in pathological eyes [20,103,154,155]. Based on DOPU images, algorithms were developed to assess areas and volumes of lesions quantitatively [121]. In age-related macular degeneration (AMD), PS-OCT was not only used to distinguish drusen characteristics but also to quantify the area and volume of drusen during disease progression (Figure 5f-h) [122,124,156].

In late stage non-exudative (dry) AMD, PS-OCT enables the assessment of atrophic areas lacking RPE (Figure 5a-d) [121,157,158]. In exudative diseases such as wet AMD, central serous chorioretinopathy, and diabetic macular edema, PS-OCT was demonstrated for imaging and identifying fibrotic scars, hard exudates, as well as pigment epithelial features [123,159-162]. Finally, PS-OCT also proved useful to enhance contrast for imaging pathologic structures in less common retinal diseases such as macular telangiectasia and Stargardt disease [163,164].

PS-OCT of the anterior eye markedly improves the contrast for birefringent, collagenous tissues such as the cornea, sclera, and tendons as well as for the trabecular meshwork [40,84,94,127]. The additional contrast has been exploited for automated, feature-based tissue discrimination [126]. Substantial changes in the birefringent appearance of the cornea can be observed in keratoconus as shown in Figure 6a-e, such that PS-OCT was proposed as a diagnostic method for this disease [128,165].

Since corneal birefringence depends on the microstructure, PS-OCT was also proposed for imaging changes during corneal crosslinking therapy [166]. After trabeculectomy, which is a surgical procedure for glaucoma treatment, the evolution of filtering blebs was monitored by PS-OCT (Figure 6f) [167-169]. In the sclera, PS-OCT was used to image necrotizing scleritis [170] and to study birefringence changes related to increased intraocular pressure [148,171].

Appl. Sci. 2017, 7, 474

12 of 34

![](dt=2026-04-21/ht=00/cc3ef3d608044319158ca212a643959f3184ebc09e0eff94171fcecd61f6fc88.jpg)

![](dt=2026-04-21/ht=00/78ee8cc1971b74bc7c338ffa36b440b20fa40ee0e5f4f97bc2739fc02b22cb04.jpg)

![](dt=2026-04-21/ht=00/951e501aeb43612bfbd3333318fd9c840bd946bbd92890d57ea9729fae90e130.jpg)

![](dt=2026-04-21/ht=00/8a46b76eb7fec61f21fa8aabaec31d0e9b63218ca76baf5e50cd8443d8c132a2.jpg)

![](dt=2026-04-21/ht=00/52722c51382dcdfecd7f0b42ffed36286d7edbc8253eba43bd04806d168b751b.jpg)

![](dt=2026-04-21/ht=00/9a4f79c9b8ccbb66add633edc6f1f631eef5ec902894e34e81843e6501d3a51e.jpg)

![](dt=2026-04-21/ht=00/93b354162a5170e5814865464f36e9c6b53b28d7cfd061f1f87f4ee5e0ccb8ca.jpg)

![](dt=2026-04-21/ht=00/c59d004fc7b856e649164b9b118eb92a2ef68741df7f1613808e1f5c034ea6e7.jpg)

![](dt=2026-04-21/ht=00/bf4786c64012af82e060e82a02711c6df2b9dd61f906dfc0b93232c11e96c124.jpg)

![](dt=2026-04-21/ht=00/72ce6933389c4b6e2999ee786448bf66f22196ca81f7cec048cdc49bd7175aa4.jpg)

![](dt=2026-04-21/ht=00/263020fb2b8a01e464a7c7aa5019e807343bfd9c690cdffdc749d8451073f911.jpg)

![](dt=2026-04-21/ht=00/62f84c63e86fe521b33f99f1ef8d7931774a30fb15d8b72f43bd79cebedf8ca7.jpg)

Appl. Sci. 2017, 7, 474

13 of 34

![](dt=2026-04-21/ht=00/ee9d8de61b2fd30071b44215386a53c25dc9ee62e173744e4845e18853b6ae7b.jpg)

# 3.2. PS-OCT in Skin and Oropharyngeal Tissue

Since the imaging regime of OCT is usually restricted to superficial layers of scattering structures (unless special probes such as catheters are used), skin is a preferred candidate for OCT imaging. Using PS-OCT, dermal layers with different scattering and polarization properties can be observed, including stratum corneum, dermis, and epidermis (Figure 7) [50,64,172,173]. Oral and laryngeal tissue have also been imaged by PS-OCT. In the oropharyngeal tract, PS-OCT was demonstrated for investigating the mucosa of the vocal fold and for detecting lesions in the buccal mucosa based on increased birefringence [174,175].

Appl. Sci. 2017, 7, 474

14 of 34

![](dt=2026-04-21/ht=00/4a46b5f79b20b2631979aaae04f818c336f19e10246e7dd70acd2fc480c7e990.jpg)

Via dermal birefringence, PS-OCT provides access to tissue alterations caused by deformation, scarring, and burns [106,177-179]. Figure 7j shows an example of scarred skin exhibiting significantly higher birefringence than normal skin (Figure 7i). Additionally, wound healing processes including collagen restoration can be followed with PS-OCT [180,181]. Moreover, Stokes vector based depolarization imaging can reveal multiple scattering as well as pathological conditions in skin such as cancer [19,24,58], as will be discussed in the next section.

# 3.3. PS-OCT in Cancerous Tissue

Cancer alters tissue microstructure. This alteration can change the optical properties of affected tissues. PS-OCT has been applied for imaging cancerous tissues in several organs. Altered birefringence

Appl. Sci. 2017, 7, 474

15 of 34

and depolarization characteristics enabled imaging and identification of skin lesions such as basal cell carcinoma (Figure 8a-d) [24,182,183].

![](dt=2026-04-21/ht=00/a0abf14338b901064829d9e660b9942a8ed5ddb5b3c1a41914f5c6f6bd6cd7f1.jpg)

![](dt=2026-04-21/ht=00/37e3cd6934848b9e91bdd0a856722ecb4f69e55bcdcd491e0b6832c41009ec35.jpg)

![](dt=2026-04-21/ht=00/995e3fa8a6bedbf2177e296529545c2f322846afd74a78544a310a20916a4906.jpg)

![](dt=2026-04-21/ht=00/47676a2adea5797b76ff4ea61d677c09d5f022dcd863f9f1423a56f27e7a937a.jpg)

![](dt=2026-04-21/ht=00/06b978070b34880b353f77fb3320128c0857478f2ced5ed2e24ea7a9e187deb3.jpg)

![](dt=2026-04-21/ht=00/0bc91f5dfac9039b8d96dee84138594ecd3af66dde20fbc4ace756a5b1d09638.jpg)

![](dt=2026-04-21/ht=00/caf5d9095b9abaf710e6d5d1a58742afba38a9fc496903387505505337bb2aa4.jpg)

![](dt=2026-04-21/ht=00/57d4f25c8ac2efbacd58d17de3d046bc4db97091c21165521750ea30e670dcdf.jpg)

Further promising results of PS-OCT based cancer imaging were reported in larynx, ovaries, and bladder [32,184,185]. Several groups also successfully studied PS-OCT for imaging breast cancer [112,186,187]. Figure 8 shows exciting results of PS-OCT imaging, which enabled the differentiation of tumor from surrounding tissue. Using intraoperative scanning of excis
ed tissue or in situ needle-based imaging (cf. Figure $8\mathrm{e}-\mathrm{j}$ ), PS-OCT could represent a promising method for reliably demarking malignant breast tumors, thus reducing the re-excision rate due to positive margins.

# 3.4. PS-OCT in Muscles, Tendons, Cartilage, and Bone

Tendon was the first biological tissue imaged by PS-OCT [8]. Being collagen-rich structures, tendons and muscles exhibit strong birefringence, which enables an easy discrimination from surrounding supportive tissue by PS-OCT.

Since the integrity of collagen is an indicator for structural stability and pathologic state, PS-OCT was suggested for collagen assessment in tendons and ligaments [189]. Consequently, PS-OCT was used to visualize the evolution of the collagen fiber alignment via birefringence in tissue-engineered tendons in response to varying growth environments and to investigate degenerative changes related to rupture in Achilles tendons [190,191]. Lately, the influence of proteoglycans—which are essential components of the tendon extracellular matrix associated with tendinopathies—on the optical

Appl. Sci. 2017, 7, 474

16 of 34

properties of tendons have been studied by PS-OCT [192]. In skeletal muscle of genetically-altered (mdx) mice, exercise-induced ultrastructural changes were detected by PS-OCT in in vivo animals [193]. Compared to wildtype controls, the highly birefringent properties of skeletal muscles markedly decreased in mdx mice, thus suggesting a relationship between the degree of birefringence detected using PS-OCT and the sarcomeric ultrastructure present within skeletal muscle. PS-OCT was also shown to be capable of detecting muscle necrosis in dystrophic mdx mice [194].

In cartilage, PS-OCT can detect areas of enhanced or reduced birefringence in hyaline cartilage (mostly composed of type-II collagen) and fibrocartilage (predominantly type-I collagen) related to degeneration and repair mechanisms [195]. In an in vivo study on human knee joints prior to partial or total joint replacement treatment, reduced birefringence was found in degenerated cartilage [196]. PS-OCT images of one proximal joint surface of bovine tibia are shown in Figure 9d [188]. PS-OCT using variable incidence angles was further used to investigate the 3D architecture of the collagen fiber network in cartilage [197,198]. Due to its high sensitivity to cartilage disorder, PS-OCT proved a promising tool for imaging cartilage in osteoarthritis in both humans and animal models [199,200].

![](dt=2026-04-21/ht=00/e5c70e1c1762860937edaaaaf801d9119a1e9640a57f4be61f1e63774ae97e1a.jpg)

![](dt=2026-04-21/ht=00/5c3edce35d0f047347d31a598d20b1245121bd86379a0935e8823b7231a58914.jpg)

![](dt=2026-04-21/ht=00/23cf88ce9f857ff66899e90aad0d789413d6b7e0ff92a1abb91e052644895f21.jpg)

![](dt=2026-04-21/ht=00/01f2270d3039f617456edc9c59591c4e10e90f94f5a1b6a332b88b5bc184e915.jpg)

# 3.5. PS-OCT in Vessels and Cardiac Tissue

Birefringence of vessel walls is a promising diagnostic parameter for arteriosclerotic vascular disease accessible by PS-OCT [201]. In atherosclerosis, artery walls locally thicken and may form atherosclerotic plaque lesions, which may be categorized into stable and unstable (called vulnerable) plaques. Ex vivo scanning compared to histology as well as catheter based PS-OCT imaging of atherosclerotic artery walls have revealed altered birefringence patterns in atherosclerotic

Appl. Sci. 2017, 7, 474

17 of 34

plaques [202-204]. Examples of different plaques imaged by PS-OCT as well as corresponding histologic images are shown in Figure 10 [205].

![](dt=2026-04-21/ht=00/cd0adeb085c5dbaa1feebf8ba2ad92d5249b578cbc9fd6a6dc72b7d2468830d2.jpg)

![](dt=2026-04-21/ht=00/d8c65572cbcefa776ae040667bb83d65e73d7fd1f27077dee90e7696d274ffcc.jpg)

![](dt=2026-04-21/ht=00/6cebb56d16aa898752730067fd9119ae620f9af9cc473b122776929df85dbaca.jpg)

![](dt=2026-04-21/ht=00/3e2a54c2a8ceb7a65f071856e8bb82a1a9db29d7f159cb6b9dac21de1e20382c.jpg)

![](dt=2026-04-21/ht=00/ae4f0344c34a510f571df2e3c377c47872493d78e4fda683578156eb53f56018.jpg)

![](dt=2026-04-21/ht=00/8de9c9db267dbe687de6ca9835b9e8193814f3dae79902141926098dd35313e9.jpg)

![](dt=2026-04-21/ht=00/22bacc6454ae58d21a1c9a8cad335f6b08021116a2ac915f6c815b9f66dc2d65.jpg)

![](dt=2026-04-21/ht=00/1c65c90fadc5305a29d9e870d48189e54031b6b34346c5e5f0f65e90cb05cc3e.jpg)

![](dt=2026-04-21/ht=00/0ead18e1b9dcf72d7489b82f36ed470ae8eeb24dbee5b867229eb1d393ab9a0e.jpg)

Information on the birefringent axis orientation provides access to fiber alignment in fibrous tissue (Figure 10a-c) [14]. Tractographic PS-OCT imaging was performed in the walls of blood vessels (Figure 10d-f) and in the mouse heart, thereby revealing fibrous layers with varying fiber orientations [69,70]. In rabbit hearts, the geometry of the perfusion border zone was investigated using PS-OCT and tissue clearing [206], and tissue discrimination was enabled by PS-OCT in rat hearts where decreased birefringence was observed in infarcted hearts [207].

# 3.6. PS-OCT in Teeth

PS-OCT has been used for imaging dental structures for almost 20 years [208-210]. In teeth, PS-OCT provides contrast for dentin, enamel, as well as carious lesions. Dentin is a calcified tissue and is a central component of teeth. On the crown, it is covered by enamel, a highly mineralized

Appl. Sci. 2017, 7, 474

18 of 34

substance. The apatite crystals in dental enamel are highly ordered and produce negative birefringence. In contrast, collagen makes dentin positively birefringent. Processes such as demineralization lead to birefringence changes which can be observed by PS-OCT.

PS-OCT was demonstrated for the assessment of early and advanced demineralization in dentin as well as in enamel (Figure 11d-e) [211-214]. Ablation of demineralized tooth structures was monitored by PS-OCT [215]. Demineralization was also investigated in tooth roots [216]. PS-OCT of enamel treated by $\mathrm{CO}_{2}$ laser irradiation confirmed inhibited demineralization [217]. In particular, caries—characterized by mineral breakdown of teeth due to bacterial activity—has been an interesting target for PS-OCT imaging.

Caries lesions in various conditions were investigated and their progression was followed longitudinally [209,210,218-220]. Consequently, also remineralization processes in enamel and dentin were imaged based on their birefringence [211,221]. Recently, an automated method for assessing remineralized lesions was developed based on PS-OCT [222].

![](dt=2026-04-21/ht=00/2fcabcbbd1c538fb75013c1ec2dece8ffb0080036c27c82cbb3a8710aa8272c6.jpg)

![](dt=2026-04-21/ht=00/c3a507e6128954cbe83f04f57b86130bd2991dc831fcb3003a0bd64eaa44720f.jpg)

![](image)
ge/dt=2026-04-21/ht=00//2adcb6f693870045cfa1ad1d44c06e274962daca800c78d38e7ca8606ef9749d.jpg)

![](dt=2026-04-21/ht=00/aa07827f9282a4825630095dc343a79e6048f977ba7a7c35c4da696d951b8a67.jpg)

![](dt=2026-04-21/ht=00/5b73de0498537ce344fe5c6a2ab8f4ede73156acd632dadf924e522e01981e70.jpg)

# 3.7. PS-OCT in Nerves and Brain

Nerve fibers exhibit birefringence, an optical property that has been exploited for quantitative measurements in the retinal nerve fiber layer (see Section 3.1). Also nerve fibers in cerebral white matter or in peripheral nerves may be imaged by PS-OCT based on their birefringence [223].

Aside from neural structures in the central nervous system, peripheral nerves have also been imaged by PS-OCT. Improved delineation of the sciatic nerve boundaries to muscle and adipose

Appl. Sci. 2017, 7, 474

19 of 34

tissues as well as quantitative birefringence measurements were enabled by the additional polarization contrast [224]. In an experimental nerve crush model, decreasing birefringence was observed in parallel to a loss of myelination (Figure 12a-d) [225]. The prostatic nerves—indiscernible from surrounding tissue by standard, intensity based OCT—were identified in prostates of rats and humans, thereby indicating the feasibility of PS-OCT as a method for intrasurgical imaging [226].

In the brain, PS-OCT was not only used to enhance the contrast of birefringent structures but also to trace white matter structures as shown in Figure 12e-h. PS-OCT based tractography provides images encoding the orientation of fiber tracts in different colors and can be used to verify diffusion tensor based MRI tractography images with micrometer scale resolution [51,227-229]. Lately, PS-OCT was also demonstrated for imaging a hallmark of Alzheimer's disease, namely neuritic amyloid-beta plaques as well as amyloidosis in cerebral vasculature [230]. PS-OCT images of birefringent neuritic plaques are shown in Figure 12i-k.

![](dt=2026-04-21/ht=00/bd6eb8d470ced90e04e822e399e8edb5a41d840648137731f99da1bbb93c0a9d.jpg)

![](dt=2026-04-21/ht=00/da97c53af12234a8bd83a834c0a2dfdcbe4357f33ee2afbe1b8f691ab68962bf.jpg)

![](dt=2026-04-21/ht=00/0b8fb1e7ae78af9fdae0939afc4e5fe3736286a1f0b7c94fca82d32b3aba55f0.jpg)

![](dt=2026-04-21/ht=00/479ae705722bdd9d59478bbafcc74c01d02be2584f048020149baf592edb2fc5.jpg)

![](dt=2026-04-21/ht=00/269baba36c74209f5f45aa362a59387f031af7832b3faf692dc8ac59b29d3763.jpg)

![](dt=2026-04-21/ht=00/8ab31ffb15cca25bccb524f76b0152a9fa62cd9516cd1837aac34b4a0e307936.jpg)

![](dt=2026-04-21/ht=00/d080a0554c609450ef6eacde11082283129df6606780120f6d985487788fad3f.jpg)

![](dt=2026-04-21/ht=00/5382b863c16fe4f1e0284cd211c40b1a0752b2b1e126060c83eef5792126d69a.jpg)

![](dt=2026-04-21/ht=00/721a77057e2800b31494dd1a8ef51b4c624e59fe8287548d9e7df0be8415aea3.jpg)

Appl. Sci. 2017, 7, 474

20 of 34

# 3.8. Other Applications of PS-OCT

PS-OCT also found applications in biomedical fields other than those discussed in the previous sections as well as in non-medical fields. One exciting use of PS-OCT is in imaging applications relying on small particles as exogenous contrast agents. As such, plasmon-resonant nanoparticles like gold nanostars are popular contrast agents in biophotonic imaging. In order to increase detection sensitivity for single particles, their polarization-sensitive scattering signal in the near infrared can be modulated by an external oscillating magnetic field [231].

PS-OCT can be used to detect these dynamic scattering signals and was demonstrated for depth-resolved viscosity measurements based on the diffusion of gold nanorods [232]. Based on temporal changes in polarization contrast parameters, a method for differentiating light scatterers such as cells and gold nanorods was developed [233]. Recently, PS-OCT based detection of gold nanorods was proposed for detecting nanotopological changes in 3D tissue models of mammary extracellular matrix and pulmonary mucus (Figure 13d) [234].

Having a non-invasive imaging technique like PS-OCT for studies of such tissue models may help to interpret biophysical changes associated with disease progression.

![](dt=2026-04-21/ht=00/6f21589567c280f7db2fcfbdcdff7d1436fcba385d8c88ce018841d770de49da.jpg)

![](dt=2026-04-21/ht=00/ec66fd9e54d14670ae355a9cb236dc841dd09aa13339f2c11d5309c2c4d74bc6.jpg)

![](dt=2026-04-21/ht=00/b0af038f0f3fa74503e5af92f621438926a3f7392c358bed90eab7f526fb0d6b.jpg)

![](dt=2026-04-21/ht=00/46e377d221cb6da6d61648123eb0a3df3fd56857e6a7b399f1dfed234d4fba3b.jpg)

![](dt=2026-04-21/ht=00/d1123f41661dd8fb010aca2d5c3bacccb48203db7bb4b951acee8454771bf415.jpg)

![](dt=2026-04-21/ht=00/ac2db8231f74b31ceaa4064d47a7565cbad5783b2fb300b71cd6fdf478710a05.jpg)

![](dt=2026-04-21/ht=00/37a2bf0c425df85ca6fce70e755a9d0195a753d7733fe0ac2ba66280826e048f.jpg)

![](dt=2026-04-21/ht=00/242032259e8122641fee92b5e67e758955062b5d1c069efba43f520b490472f2.jpg)

![](dt=2026-04-21/ht=00/fbfd926d22dd8bb850d20eeb8a8d29daa6bc8fc058ccddc3943d0a7c48a950e2.jpg)

![](dt=2026-04-21/ht=00/5d4836286ad269353f17049e8d8751a09897a5c910bc0185a9206873a3c4a360.jpg)

![](dt=2026-04-21/ht=00/6d807425268199beedb0aea5bd8fa639dfcabc58d81605bff69662fe133cf615.jpg)

![](dt=2026-04-21/ht=00/cdc50539f438a8b8d7593e1f0385e9bfa9265a04a8b2103d2cf2a15ae2904233.jpg)

Its sensitivity for microstructural changes affecting sample polarization properties along with its 3D imaging capabilities made PS-OCT also an interesting modality for materials science and nondestructive testing applications [236]. Using translucent glass-epoxy composite phantoms, the spatial distribution of mechanical stress was mapped by PS-OCT [237]. Strain mapping by also exploiting the optic axis orientation in PS-OCT images was demonstrated as a method for charting the directionality of strained sample areas (Figure 13a-c) [235]. Material dynamics were optically

Appl. Sci. 2017, 7, 474

21 of 34

investigated by translating measured phase retardation into stress images [238]. Thereby, dynamics could be studied from the elastic regime over the deformation phase up to fracture.

# 4. Conclusions

PS-OCT is a versatile functi
onal extension of OCT. As described in the section on the technical background, only few modifications to the standard OCT layout are necessary for polarization sensitivity. Of course, more sophisticated setups and advanced analysis methods were also developed. This review aimed to provide a concise introduction to the basic principles underlying PS-OCT and a crisp overview of advances in PS-OCT technology development.

By highlighting research on state of the art PS-OCT applications based on the published literature, the obvious potential of this powerful technique for improved qualitative and quantitative imaging was portrayed. Given the achievements of the past 20 years discussed here, we are anticipating exciting new technological developments, advances of applied biomedical imaging, and potential applications in new fields for the next 20 years.

Acknowledgments: I would like to express my gratitude to Christoph K. Hitzenberger, Michael Pircher, and Erich Gotzinger, Medical University of Vienna, Austria, who introduced me to the exciting world of PS-OCT more than 10 years ago.

Also, I would like to thank my colleagues at the Center for Medical Physics and Biomedical Engineering (in particular Marco Augustin and Christoph Hitzenberger for proof-reading), at the Department of Ophthalmology, at the Division of Biomedical Research, at the Core Facility Imaging and at the Institute of Neurology at Medical University of Vienna as well as at the VetCore Facility for Research and Technology at the University of Veterinary Medicine Vienna for their continuous collaborative support, fruitful discussions, and creative feedback.

Funding by the Austrian Science Fund (FWF grant P25823-B24) and the European Research Council (ERC Starting Grant 640396 OPTIMALZ) is gratefully acknowledged.

Conflicts of Interest: The author declares no conflict of interest.

# References

Appl. Sci. 2017, 7, 474

22 of 34

Appl. Sci. 2017, 7, 474

23 of 34

Appl. Sci. 2017, 7, 474

24 of 34

Appl. Sci. 2017, 7, 474

25 of 34

Appl. Sci. 2017, 7, 474

26 of 34

Appl. Sci. 2017, 7, 474

27 of 34

Appl. Sci. 2017, 7, 474

28 of 34

Appl. Sci. 2017, 7, 474

29 of 34

Appl. Sci. 2017, 7, 474

30 of 34

Appl. Sci. 2017, 7, 474

31 of 34

Appl. Sci. 2017, 7, 474

32 of 34

Appl. Sci. 2017, 7, 474

33 of 34

![](dt=2026-04-21/ht=00/af130906e99ccc38b3696da8696b9817fc7fa3a954f4faa054b37a6a519b7101.jpg)

© 2017 by the author. Licensee MDPI, Basel, Switzerland. This article is an open access article distributed under the terms and conditions of the Creative Commons Attribution (CC BY) license (http://creativecommons.org/licenses/by/4.0/).

Appl. Sci. 2017, 7, 474

34 of 34