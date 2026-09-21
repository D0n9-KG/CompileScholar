# ARTICLE

# Open Access

# Quantitative analysis of focal adhesion dynamics using photonic resonator outcoupler microscopy (PROM)

Yue Zhuo $^{1,2}$ , Ji Sun Choi $^{3,4}$ , Thibault Marin $^{5}$ , Hojeong Yu $^{6}$ , Brendan A. Harley $^{3,4}$ and Brian T. Cunningham $^{1,2,4,6}$

# Abstract

Focal adhesions are critical cell membrane components that regulate adhesion and migration and have cluster dimensions that correlate closely with adhesion engagement and migration speed. We utilized a label-free approach for dynamic, long-term, quantitative imaging of cell–surface interactions called photonic resonator outcoupler microscopy (PROM) in which membrane-associated protein aggregates outcoupled photons from the resonant evanescent field of a photonic crystal biosensor, resulting in a highly localized reduction of the reflected light intensity. By mapping the changes in the resonant reflected peak intensity from the biosensor surface, we demonstrate the ability of PROM to detect focal adhesion dimensions. Similar spatial distributions can be observed between PROM images and fluorescence-labeled images of focal adhesion areas in dental epithelial stem cells. In particular, we demonstrate that cell–surface contacts and focal adhesion formation can be imaged by two orthogonal label-free modalities in PROM simultaneously, providing a general-purpose tool for kinetic, high axial-resolution monitoring of cell interactions with basement membranes.

# Introduction

Focal adhesions (FAs), or cell–matrix adhesions, are large specialized proteins that are typically located at the interface between the cell membrane and extracellular matrix (ECM) (Fig. 1a, b) $^{1-24}$ . FAs are critical for supporting the cell membrane structure and regulating signal transmission between the cytoskeleton (e.g., actin) and transmembrane receptors (e.g., integrins) during adhesion and migration $^{16-24}$ . Monitoring the response of FA clusters to drugs is one important mechanism by which the action of pharmaceutical compounds may be evaluated, particularly where approaches that enable characterization to be performed with a small number of cells are especially valuable $^{22,25-28}$ . During the dynamic assembly and disassembly of a FA, the size of the FA cluster varies and is highly correlated with the level of adhesion engagement and migration speed $^{13,29}$ . For example, non-mature focal complexes (FXs) are initially formed at the leading edge of the cell (e.g., in the lamellipodia area) and are usually $<0.2\ \mu m^{2}$ . As the lamellipodia withdraws from the leading edge, many FXs disassemble and release adhesion proteins back to the inner cell body, whereas some of the FXs grow larger (typically $1-10\ \mu m^{2}$ ) and assemble into mature FA clusters by recruiting adapter proteins $^{19,29}$ . Once the remaining FAs are in place, they may form stationary attachment points by binding to the ECM, and a cell may utilize these anchors to migrate over the ECM by pushing and pulling the entire cellular body $^{18,21,23}$ . This insight into the dynamics of FA cluster formation and dissociation has been made possible by technical advances in the field of fluorescence and super resolution microscopy $^{30-36}$ . Optical modalities, including total internal reflection fluorescence microscopy, photoactivation localization microscopy (PALM),

a   
![](images/dd5b63d1d41ebe4da71e4fdd2ac7a83812e2d2a215c67f01f2e5180b4247acba.jpg)

<details>
<summary>text_image</summary>

Evanescent field
Extra cellular matrix
Photonic crystal surface
Cell non-attached
</details>

b   
![](images/3df99d17535ce62052743177a16d0c837f7a03d2f29618e123e329f6a6bc6363.jpg)

<details>
<summary>text_image</summary>

Focal adhesion
Photonic crystal surface
Cell attached
</details>

C   
![](images/19e2bb49a8e966dee93fba2d9aa4ff77d9774125d7db9d8468ef69120b61fa66.jpg)

<details>
<summary>line</summary>

| Wavelength (nm) | Reflectance (AU) - BG | Reflectance (AU) - Cell |
| --------------- | --------------------- | ----------------------- |
| 610             | ~0.0                  | ~0.0                    |
| 620             | ~0.8                  | ~0.7                    |
| 630             | ~1.0                  | ~0.9                    |
| 640             | ~0.0                  | ~0.0                    |
| 650             | ~0.0                  | ~0.0                    |

Left Chart: Reflectance (AU) vs. Wavelength (nm)
- Black Curve: Photonic crystal biosensor
- Red Curve: Objective lens
- Blue Curve: Cylindrical lens
- Light Blue Dashed: Shift
- Red Shaded: ΔI (PIS)
- Vertical Labels: Δλ (PWS), Shift, ΔI (PIS)

Right Chart: Spectrometer setup with labeled components:
- Dichroic mirror
- Tube lens
- Mirror
- Polarized beam splitter
- LED
</details>

Fig. 1 Principle of the molecular-dynamics for cell attachment on a photonic crystal (PC) biosensor in photonic resonator outcoupler microscopy (PROM). Schematic representation of the molecular mechanism a before and b after a live cell attaches to the PC biosensor surface. c The principle of PROM imaging system. Inset: spectra shift before and after the cell attaches to the PC surface

stochastic optical reconstruction microscopy, and interferometric PALM, coupled with fluorescence tagging of the element(s) of FA clusters via administration of fluorescently labeled antibodies or incorporation of fluorescent reporter genes by transfection of cells, along with progress made in single particle tracking algorithms, have allowed researchers to quantify FA-associated parameters, such as FA areas and sizes (x-y dimensions), FA architectures (x-y-z dimensions), FA turnover rates, and spatiotemporal distributions of FA complexes. Additionally, developments in traction force measurements (e.g., based on two-dimensional (2D) hydrogel substrates or micropillar substrates) $^{17,37-39}$ , mechanical probing of cells (e.g., atomic force microscopy) $^{40,41}$ , and single molecular techniques (e.g., tension sensors) $^{42}$ have allowed the quantification of molecular tension forces within FA clusters as well as FA-mediated traction and adhesion forces.

Understanding the dynamics of FA formation and changes in FA-associated parameters is beneficial not only for understanding the fundamentals of biology but also for the field of biosensor diagnostics and screening for clinical applications $^{36,43,44}$ . Changes in FA-associated parameters, such as FA sizes and traction forces, have

been linked to critical cellular processes, including metastasis, apoptosis, and chemotaxis, as well as pathologies of cancers and other diseases $^{9,29,36,45-47}$ . As such, monitoring the response of FA clusters to drugs, for example, is an important mechanism by which the action of pharmaceutical compounds may be evaluated $^{22,25-28,36}$ , and high-throughput approaches that enable the characterization of small cell populations in real time are especially valuable for these applications. Currently available techniques largely make use of fluorescence tagging to mark individual FA proteins, which entails temporal limitations imposed by photobleaching and challenges associated with accurate quantitation and long-term analysis $^{9,11,48}$ . New tools are therefore required to study the dynamic behavior of FA clusters and their interaction with the ECM to characterize changes in FA dynamics in live cells in situ. However, determining the dynamic activity of a FA cluster is challenging, especially with all of the FA proteins that are simultaneously active during the in situ assembly and disassembly processes in live cells. Although a variety of approaches have been utilized to investigate these processes, the detailed mechanism of FA assembly and disassembly in live cells, including the variability of the FA dimension, is poorly understood $^{9,11,48}$ . For instance, fluorescent tags are often used to mark individual FA proteins, but due to the temporal limitations imposed by photobleaching, accurate quantitation and long-term analysis are exceedingly difficult to perform, whereas the cytotoxicity of fluorescent tags compromises the viability of the cells under study. Here we describe a label-free optical sensing approach that combines two optical modalities to quantify the FA-associated parameters that are critical for characterizing spatiotemporal distribution patterns and the strength of FA clusters in real time. In our previous studies on cell imaging by photonic crystal enhanced microscopy (PCEM), we utilized an imaging modality in which the reflected resonant peak wavelength value (PWV) was measured over the imaging field-of-view to derive images of the peak wavelength shift (PWS) that occur when cells attach to the photonic crystal (PC) surface $^{49-52}$ . In these studies, we describe how the engagement of the cell membrane components with the surface of the PC results in highly localized shifts in the resonant reflected wavelength from the biosensor $^{52}$ , as well as the design of a modified brightfield (BF) microscope that enables visualization of cell–surface attachments with a $\sim0.6\times0.6\mu m^{2}$ pixel size. The PWS image sequences clearly show the evolution of cell attachment through the engagement of the lipid bilayer cell membrane and internal cell-associated proteins within the $\sim200nm$ deep evanescent field region of the PC. In this study, we demonstrate a novel and orthogonal imaging modality within PCEM in which we measure the resonant reflected peak intensity value (PIV) from the PC before and after live cell attachment to acquire the peak intensity shift (PIS) at each local voxel volume (Fig. 1a, b). Images of the PIS reveal highly localized and easily observable loci of protein clusters that correlate with the spatial distribution and size of FAs observed by fluorescence microscopy. We hypothesize that the observed reduction in reflected intensity from the PC is mainly caused by the outcoupling of resonant standing wave photons via scattering.

Because this imaging modality operates using an independent sensing mechanism that obtains contrast through the formation of protein clusters at the cell–ECM interface that are capable of outcoupling light from the PC biosensor surface, we name this technique photonic resonator outcoupler microscopy (PROM) (Fig. 1c). In our previous study, we report the first observation of a reduced and highly localized reflected intensity in the context of nanoparticles with optical absorption at the resonant wavelength of the PC $^{50}$ . However, these observations were made with very high contrast and highly localized metal absorbers (for plasmonic nanoparticles) or titanium dioxide (TiO $_{2}$ ) nanoparticle dielectric scatterers. Although the reduced reflected resonant intensity from the high-contrast surface-attached TiO $_{2}$ scatterers (the refractive index of TiO $_{2}$ nanoparticles ( $n_{TiO2}$ = \~2.4) is much larger than that of the surrounding water medium ( $n_{2}$ = \~1.333)) was the first observation with PROM, this study is the first to use PROM to observe scattered outcoupling from very low contrast FAs in live cell membranes (averaged cell $n_{cell}$ = 1.35–1.38) to the surrounding medium ( $n_{2}$ = \~1.333). Using dental epithelial stem cells attached to a fibronectin-coated ECM surface as a representative example, we demonstrate that PWS and PIS images of the same cells display distinct and complementary information. Although the regions with the greatest PWS are at the cell–surface interface in which uniformly distributed regions with the greatest surface engagement occur, the regions with the greatest PIS represent the formation of highly concentrated protein clusters at the cell–surface interface that are capable of scattering photons. Therefore, we introduce PROM as a quantitative, dynamic, and label-free approach to observe the formation and evolution of FA cluster areas that are otherwise challenging to observe with other available imaging modalities, particularly for repeated observations of the same cell population for extended time periods.

# Materials and methods

# PC biosensor

The PCs used in this study are subwavelength nanostructured surfaces with a periodic modulation in the refractive index that acts as a narrow bandwidth resonant optical reflector at one specific resonance wavelength (Fig. 1) $^{49,50,52-55}$ . The high reflection efficiency of the PC

a   
![](images/1ae0ba7e1946e92aaeda75b7d7c3300362b720cfe9eaccaec2d5297d56ee2e30.jpg)

<details>
<summary>natural_image</summary>

Microscopic view of a microfabricated structure with dashed outline patterns and scale bars (no text or symbols)
</details>

b   
![](images/d9463012b651b7cd448f187ad53e4af49ee8f10e2ecd70ed5d16c56c7e4dea0e.jpg)

<details>
<summary>natural_image</summary>

Microscopic view of parallel fiber structures with scale bars (no text or symbols)
</details>

C   
![](images/c90f39f1f77d7b30d2009bc753b97e55b8b82b7cd77601481bd50a0ba7d8d85e.jpg)

<details>
<summary>text_image</summary>

d=dg+ds
d=0
ds
dg
fs
ns
f9
(Medium)
θs
θg
θ1
Slab
n2
n1
n0
(UVCP)
k0
H
E
E
H
k0
TE
TM
Substrate
n'0
z
y
x
n
</details>

d

![](images/248ad7e5c34caa88f2c2d26855fb1c97f383ce6beab78e2025602ab385dc8ed8.jpg)

<details>
<summary>text_image</summary>

d=ds
d=0
n2
n1
ds
θi
n0
z
y
x
n'0
</details>

e   
![](images/8d63a9c1e5188bc79029ed42483d28f0ca6eb9678a4470a95f1ad8e0372cd5eb.jpg)

<details>
<summary>line</summary>

| Wavelength (nm) | Reflectance (AU) for n₂=1.333 | Reflectance (AU) for n₂=1.343 | Reflectance (AU) for n₂=1.353 | Reflectance (AU) for n₂=1.363 | Reflectance (AU) for n₂=1.373 |
| --------------- | ----------------------------- | ----------------------------- | ----------------------------- | ----------------------------- | ----------------------------- |
| 580             | ~0.2                          | ~0.2                          | ~0.2                          | ~0.2                          | ~0.2                          |
| 600             | ~0.1                          | ~0.1                          | ~0.1                          | ~0.1                          | ~0.1                          |
| 620             | ~0.9                          | ~0.9                          | ~0.9                          | ~0.9                          | ~0.9                          |
| 640             | ~0.1                          | ~0.1                          | ~0.1                          | ~0.1                          | ~0.1                          |
| 660             | ~0.0                          | ~0.0                          | ~0.0                          | ~0.0                          | ~0.0                          |
</details>

f   
![](images/e888a23a8b7c50f2fdf82308f57ffcec0e43506f347235afa9d6923b7b61e1e4.jpg)

<details>
<summary>line</summary>

| Refractive index (AU) | Normalized PIV (AU) | Normalized PIS (AU) |
| --------------------- | ------------------- | ------------------- |
| 1.34                  | 1.0                 | 0.0                 |
| 1.36                  | 1.0                 | 0.0                 |
| 1.37                  | 1.0                 | 0.0                 |
</details>

g   
![](images/14b701af03fca36ebc665f2c32e2eb469bb0d1ee858632ccf29db8bb07ea8140.jpg)

<details>
<summary>scatter</summary>

| Refractive index (AU) | PWS (nm) | PWV (nm) |
| --------------------- | -------- | -------- |
| 1.33                  | 0        | 625      |
| 1.34                  | 1        | 626      |
| 1.35                  | 2        | 627      |
| 1.36                  | 3        | 628      |
| 1.37                  | 4        | 629      |
</details>

h   
![](images/9f5b1586e3a57cfe1114d291c232f0a5075cbe5af6c0c89517bf4a8e7363bbad.jpg)

<details>
<summary>line</summary>

| Wavelength (nm) | Reflectance (AU) for r=50 nm | Reflectance (AU) for r=100 nm | Reflectance (AU) for r=250 nm | Reflectance (AU) for r=500 nm |
| --------------- | ---------------------------- | ----------------------------- | ----------------------------- | ----------------------------- |
| 580             | ~0.1                         | ~0.1                          | ~0.1                          | ~0.1                          |
| 600             | ~0.1                         | ~0.1                          | ~0.1                          | ~0.1                          |
| 620             | ~0.9                         | ~0.9                          | ~0.9                          | ~0.9                          |
| 640             | ~0.1                         | ~0.1                          | ~0.1                          | ~0.1                          |
| 660             | ~0.0                         | ~0.0                          | ~0.0                          | ~0.0                          |
| 624             | ~1.0                         | ~1.0                          | ~1.0                          | ~1.0                          |
| 626             | ~1.0                         | ~1.0                          | ~1.0                          | ~1.0                          |
| 628             | ~1.0                         | ~1.0                          | ~1.0                          | ~1.0                          |
| 630             | ~1.0                         | ~1.0                          | ~1.0                          | ~1.0                          |
</details>

i   
![](images/37b2095cf8306639796674ccb782d631fc45eb72bdcde43dc984cf5adc491cbf.jpg)

<details>
<summary>line</summary>

| NP radius (nm) | Normalized PIV (AU) | Normalized PIS (AU) |
| -------------- | ------------------- | ------------------- |
| 0              | 1.0                 | 0.0                 |
| 200            | 0.9                 | 0.05                |
| 400            | 0.8                 | 0.15                |
</details>

j   
![](images/1ee2983db0bba47f68548eb5d51026ada42cc4331201c49993c655cbd3adb642.jpg)

<details>
<summary>scatter</summary>

| NP radius (nm) | PWS (nm) | PWV (nm) |
| -------------- | -------- | -------- |
| 0              | 0        | 625      |
| 100            | 0.5      | 626      |
| 200            | 1.0      | 627      |
| 300            | 1.5      | 628      |
| 400            | 2.0      | 629      |
| 500            | 2.5      | 629      |
</details>

Fig. 2 Principle of peak intensity shift (PIS) and peak wavelength shift (PWS) on a PC surface. SEM images of a fabricated PC biosensor with a side views of the cross-section (inset: zoomed-in side view) and b top views (inset: zoomed-in top view). c FDTD simulation model of the PC surface (side view of the cross-section). d Simplified model as a waveguide on the PC surface (side view of the cross-section). e Normalized spectra with different background refractive indices ( $n_2 = 1.333$ , 1.343, 1.353, 1.363, 1.373) on a PC surface without dielectric nanoparticles (inset: zoomed-in peak of the reflection spectra). Corresponding f peak intensity shift (PIS) (inset: peak intensity value (PIV)) and g peak wavelength shift (PWS) (inset: peak wavelength value (PWV)). h Normalized spectra with different sizes (radius of 50, 100, 250, 500 nm) of dielectric nanoparticles on the PC surface (inset: zoomed-in peak of the reflection spectra). Corresponding i PIS (inset: PIV) and j PWS (inset: PWV). Scale bar: 200 nm

at the resonant wavelength (Fig. 1c) is the result of the formation of surface-confined electromagnetic standing waves that extend into the surrounding medium in the form of an evanescent electromagnetic field $^{53-80}$ . The photonic band gap of the PC strictly limits the lateral propagation of light. Thus the PC exhibits a strong optical confinement of incident light into an infinitesimal volume that selectively interacts with surface-adsorbed cell components while being insensitive to the components of the cell body that are not engaged with the surface. Simulations (Fig. 2) performed using the finite-difference time-domain (FDTD) method show the spatial distribution of the resonant electromagnetic field, which extends $\sim$ 200 nm into the aqueous medium at the top of the PC. Previous research has demonstrated that a specific location on the PC surface has a resonant reflected wavelength that can be independently measured from neighboring regions and that the local PWV is determined by the dielectric permittivity of the biomaterial that is adsorbed at that specific location $^{50}$ . The PC surface can therefore act as a proxy for a biological surface with a built-in capacity to detect changes in the cell membrane components of cells that attach to the PC within the evanescent field, providing a compelling platform for adhesion phenotyping of single cells (see Supplementary Materials Section S-1 for details). PC biosensor surfaces are inexpensively fabricated uniformly over large surface areas by a room temperature nanoreplica molding process, as described previously in refs. $^{54,55}$ , and are incorporated onto glass microscope slides, described in refs. $^{49,50,52,81}$ .

# Modeling the PC surface for sensor design and simulation

A numerical electromagnetics simulation package (FDTD, Lumerical Solutions, Inc., Vancouver, BC, Canada) is used to calculate the distribution of a resonant evanescent field on the PC biosensor surface. In our previous studies, the PC surface was modeled as an ideal case with a rectangular nanostructure for simplicity $^{50,52}$ . To more accurately represent the fabricated structure (Fig. 2a, b), the model used in this study incorporates a sidewall slope in a trapezoidal shape. As shown in Fig. 2c, d, the PC consists of a one-dimensional ultraviolet-curable polymer (UVCP) grating surface structure (refractive index $n_{0}=1.46$ , grating depth $d_{g}=120$ nm, period $\Lambda=400$ nm, duty cycle $f_{g}=41.6\%$ , sidewall angle $\theta_{g}=85^{\circ}$ ) coated with a thin film of $TiO_{2}$ (refractive index $n_{1}=2.4$ , slab thickness $d_{s}=61$ nm, duty cycle $f_{s}=50\%$ , sidewall angle $\theta_{s}=82^{\circ}$ ) to generate a resonant reflected narrowband mode at a wavelength near $\lambda_{0}=\sim626$ nm. The adhesion of FAs on the PC surface is also modeled in FDTD, where the FA is represented as a homogeneous and lossless sphere ( $n_{FA}=\sim1.46$ , radius range 50–500 nm) composed of many protein molecules (Fig. 2e–j).

# Fabrication and preparation of the PC surface

A room temperature replica molding approach is used to fabricate the PC on a glass substrate using a quartz mold template with a negative volume image of the desired grating structure (fabricated with e-beam lithography and reactive ion etching). First, the quartz mold template is thoroughly cleaned with a piranha solution (a mixture of sulfuric acid $\left(\mathrm{H}_{2}\mathrm{SO}_{4}\right)$ and hydrogen peroxide $\left(\mathrm{H}_{2}\mathrm{O}_{2}\right)$ , $H_{2}SO_{4}:H_{2}O_{2}=3:1$ ) for approximately 3 hours to remove organic residues from the surface of the master template. The glass substrate is cleaned in an ultrasonic bath three times with isopropyl alcohol (IPA), acetone and deionized (DI) water for 1 min in each solvent and then dried with nitrogen gas and treated with oxygen plasma. Second, the liquid UVCP is deposited between the quartz mold template and glass substrate, and a high intensity UV lamp is used to cure the liquid polymer to a solid state. After peeling the grating replica away from the quartz mold template, the nano-patterned surface is attached to a glass cover slip with an adhesive. Then PC fabrication is completed by reactive sputter deposition (PVD 75, Kurt J. Lesker, Jefferson Hills, PA, USA) of a high refractive index thin film $\left(\mathrm{TiO}_{2}\right)$ atop the grating structure. Scanning electron microscopic (SEM) images of a cross-sectional view and a top view of the structure are shown in Fig. 2a, b, respectively. Next, before cell attachment experiments, the PC is cleaned in an ultrasonic bath with IPA and DI water for 1 min each, followed by drying with nitrogen gas. The PC is then treated with oxygen plasma to facilitate attachment of a liquid containment gasket formed from polydimethylsiloxane. Finally, the PC surface is hydrated with a phosphate-buffered saline solution and coated with a layer of ECM molecules (e.g., fibronectin) to promote cellular attachment.

# Photonic resonator outcoupler microscopy

The PROM instrument is a modified BF microscope that uses a line-scanning approach to measure the spatial distribution of optical spectra across a PC surface with a submicron spatial resolution in the axial direction for label-free imaging (Fig. 1c) $^{49,51}$ . An optical fiber-coupled light-emitting diode is used as the light source, and a line-profiled (polarized perpendicular to the grating structure) light beam illuminates the PC biosensor from below through a microscope objective lens (e.g., 10×). Illumination from below eliminates the possibility of the scattering, absorption, and meniscus reflection and refraction of materials in the cell media or cell body from effecting a resonant reflected signal to the PC surface. The reflected light, containing the resonant reflected spectrum, passes through the objective lens in the opposite direction and through the entrance slit of an imaging spectrometer and is finally collected by a charge-coupled device camera,

which records the resonant reflected spectrum from each pixel across the illuminated line on the PC surface. A high spatial resolution in the axial direction is obtained due to the shallow evanescent field of the PC ( $\sim200~nm$ ). The resolution in the lateral direction is determined by the lateral propagation distance of resonant-coupled photons, resulting in the detection of distinct surface-attached objects for widely dispersed features at the micron size scale $^{50}$ . The same field of view can be re-scanned repeatedly to generate a sequence of images for the same cells. The current shortest available scan interval for our instrument is $\sim10~s$ for $\sim100\times100\mu m^{2}$ , which is limited by the exposure time and speed of the motorized scan stage. While characterizing the resolution performance through the intentional introduction of dielectric and metallic nanoparticles of a variety of sizes (30–500 nm), we observe that dielectric objects not only induce a shift in the PWV but also a reduction in the resonant peak intensity $^{50}$ . When a cell attaches to the PC surface, the peak resonant wavelength red-shifts from a lower wavelength (before cell attachment, e.g., $\lambda_{BG}=\sim626~nm$ ) to a higher wavelength (after cell attachment, e.g., $\lambda_{cell}=\sim628~nm$ ). At the same time, the resonant reflection efficiency, as measured by the peak intensity, changes from a higher PIV (before cell attachment, e.g., $\sim90\%$ normalized to a peak reflectance of 100% for the PC immersed in water) to a lower value (after cell attachment, e.g., 80% as a normalized PIV). A negative PIV shift indicates that proteins in some areas of the cell bind to transmembrane proteins (e.g., integrins) to form more substantial FA clusters. The PROM instrument and sensor structure measure the resonant reflection characteristics of the PC via the spectrum obtained from each $\sim0.6\times0.6\mu m^{2}$ pixel area, representing a total field of view region (e.g., $\sim300\times300\mu m^{2}$ ) of the PC surface.

# Stem cell culture

Murine dental epithelial stem cells (mHAT9a) were maintained in Dulbecco's Modified Eagle Medium supplemented with 10% fetal bovine serum and 1% Penicillin–Streptomycin. Stem cells were cultured at a temperature of $37^{\circ}$ C and supplied with an environment of 5% CO $_{2}$ humidified air during imaging.

# Results and Discussion

# Electromagnetic computer modeling of resonant outcoupling from a PC by FA

It is important to understand the physical mechanism that is responsible for the PIS in the context of cell attachment. Theoretical and experimental analyses suggest that a reduction in the PIV can occur by two mechanisms: (1) materials that act as efficient absorbers of the resonant wavelength (such as gold nanoparticles) locally quench the PC resonance; (2) concentrated local regions of high dielectric permittivity that can outcouple resonantly confined light by scattering $^{50}$ . Interestingly, by analyzing PROM data during cell attachment, we observed the characteristics of PIV images that differ substantially from PWS images. Although optical absorption at the PC resonant wavelength will efficiently reduce the PIV in a highly localized manner, the protein and lipid components of cells and cell membranes do not display strong absorption in the visible wavelength range. Human tissues and live cells exhibit strong absorption in the infrared wavelength range; however, these wavelengths are not utilized in our detection approach and thus are unlikely to be the dominant factors that contribute to PIV reduction. Additionally, though metallic elements comprise a small fraction of a cell's atomic constituency, metal atoms are present as ions rather than as clusters that are capable of optical absorption in the visible part of the spectrum. Scattering occurs when light is forced to deviate from its original trajectory due to localized non-uniformities in its propagation medium, which occur, for example, when light propagating through water is reflected or refracted by a particle with a greater refractive index. A highly concentrated region (e.g., FA cluster) with a greater refractive index than its surroundings (e.g., cell media) generates more localized and efficient scattering than a diffuse region with a gradual gradient in the refractive index. Scattering effects also become stronger when the size of a region with a refractive index contrast increases. Because the cross-sectional area of a FA cluster is typically $0.2–1.0 \mu m^{2}$ , we can expect to observe measurable differences in the scattering efficiency of membrane-associated protein clusters as they form, change size, and subsequently dissipate. Light scattering from internal cell components, such as organelles and mitochondria, has recently been utilized to achieve imaging contrast in the context of changes that occur in precancerous cells that express phenotypic changes due to the expression of mutant genes $^{82-84}$ . In PROM, we detect the modulation of membrane-associated scattering that occurs due to FA formation by utilizing the ability of localized high refractive index protein clusters to produce image contrast by reducing the reflection efficiency of a PC biosensor.

After analyzing the mechanism of PIV reduction, we study the useful cellular information that can be uniquely extracted from the measurement of this physical quantity. Our hypothesis is that the dominant cause for the measured PIV reduction is light scattering rather than absorption. Thus a scattering model can represent the interaction of light with a FA cluster since scattering describes the effect of an electromagnetic plane wave propagating through a dielectric particle. To predict the

effects on the resonant reflection spectrum from a PC that can be induced by a FA on its surface, we compared the computed reflection spectra with and without a small region of dielectric contrast on a PC surface. In a FDTD electromagnetic computer model of a FA on a PC surface (Fig. 2e–j), we represent the FA as a locus of a material with a designated radius (50–500 nm) and designated refractive index elevation ( $n_{FA} = \sim 1.46$ ) in contrast to the surrounding medium (estimated to be $n_{2} = \sim 1.333$ ). The simulation results demonstrate that, when the FA cluster is not present, the PWS in the resonant reflection spectrum increases (Fig. 2g) as the refractive index contrast of the surrounding media increases, whereas the PIS remains the same (Fig. 2f). However, when the FA cluster is present, as the radius of the FA increases (Fig. 2h), the maximum in the resonant reflection spectrum decreases (Fig. 2i, Inset), indicating that more energy is outcoupled from the PC resonant standing wave. Thus the PWSs to a higher wavelength (Fig. 2j), as expected, and the PIS (compared with the original PIS without a FA cluster) increases as the FA cluster size increases (Fig. 2i) due to the scattering-induced outcoupling (for more details, see Supplementary Materials S-2).

# Dynamic PIS images of live cells

Dynamic PIS images can be acquired by PROM during live cell adhesion over an extended time period to create movies of FA development, which are difficult to acquire via fluorescence imaging due to photobleaching. The PIS here is defined such that greater reductions in the reflection efficiency are displayed as higher intensity values for a simpler visual comparison with other imaging modalities. As shown in Fig. 3a, the resulting PIS image sequence reveals that the cell periphery has a greater degree of scattering than the cell center. This “ring” effect demonstrates that the PIS intensity is not homogeneously distributed throughout the cell membrane. Fig. 3b shows spectra from three sample points marked in Fig. 3a (A—red, B—black, C—green) at different times. Initially, all three points are located outside of the cell ( $\sim$ 0 min), and all of the points show high resonant PIVs as highlighted by the dashed line in the spectra (Fig. 3b). During cell adhesion, the attachment perimeter expands and surpasses the three points as the cell extends its attachment area. Once the cell firmly attaches and adheres to the PC surface ( $\sim$ 16 min), points A', B', and C' represent locations inside, near, and outside of the cell boundary, respectively. The solid lines in Fig. 3b demonstrate that the spectra of point A' shifts to a lower resonance peak intensity, point B' shifts to the lowest resonance peak intensity, and point C' remains at the original resonance peak intensity. Considering all of the pixels within the cell, we observe a ring of enhanced scattering that encompasses much of the cell periphery.

A difference of PWS and PIS images for the same cell at each time point is clearly observed in the spectral data acquired by PROM. In Fig. 3b, the dashed lines represent the background spectra for a representative pixel acquired before cell attachment (0 min), and the dotted lines with dot markers represent the spectra of the same pixel after cell attachment (\~16 min). PWS and PIS images are simultaneously extracted from the spectra data at the PWV and PIV, respectively (at every $\sim0.6\times0.6\mu m^{2}$ pixel area on the PC surface). In Fig. 3a, the regions with the greatest values in the PWS and PIS images show distinct distribution patterns, which suggests that these regions may represent two different physical mechanisms. For instance, the PWS image of the cell marked by the white arrow (e.g., \~16 min) has a high PWS on the top and bottom of the cell body, whereas the PIS image of the same cell has a “ring” of a high PIS along the cell boundary. The zoomed-in images of the PWS and PIS taken 16 min after introduction of a single cell are shown in Fig. 4a and S-Fig. 1, respectively, and the overlaid PWS and PIS images (high intensity values only, PWS—green, PIS—red, overlap—yellow) over the BF image of the same cells show the distinct distribution patterns between these two physical quantities. The cross-sectional curves (L1 and L2) along the diameter of the cell are plotted in Fig. 4b, and the corresponding statistical results are shown in Fig. 4c (N=5 cells). The high intensity of the PWS (in green at the bottom of Fig. 4a) represents a higher mass density of cellular materials associated with the cell membrane. The high intensity of the PIS (in red at the bottom of Fig. 4a) along the cell boundary represents the scattering outcoupling effect from the locally generated FA clusters. Typically, a larger cluster size of protein aggregates corresponds to a greater PIV reduction compared with the background. Therefore, these measurements of local PWS and PIS can quantify the surface-attached cellular mass density and dimension of the FA clusters dynamically and simultaneously.

# Comparison of PIS and fluorescence images

Fluorescence images were acquired for the same cells to further investigate the FA areas detected in the PIS images. In Fig. 5a, the top row shows BF, PWS, and PIS images and the bottom row displays fluorescence images of the same cell with fluorescence tags applied to three different cellular components (nucleus, actin, and vinculin). There is no obvious pattern similarity between the PIS image and actin (indicating the presence of cytoskeleton components) or nucleus images. However, the fluorescent image of vinculin (a type of FA molecule) in the bottom row of Fig. 5a indicates that filopodia reside in the FA area along the stem cell boundary. As shown in Fig. 5a in the right column, the PIS image shows a nearly identical distribution pattern along the cell peripheral

![](images/9d8637b79d6ba29e17e6bd6598ff0e28b50c5fb7377ee0f00a41078ee8df270a.jpg)

<details>
<summary>heatmap</summary>

| Time   | BF     | PIS    | PWS    |
|--------|--------|--------|--------|
| 0 min  | 0.5    | 0.5    | 0.5    |
| 2 min  | 0.5    | 0.5    | 0.5    |
| 4 min  | 0.5    | 0.5    | 0.5    |
| 6 min  | 0.5    | 0.5    | 0.5    |
| 8 min  | 0.5    | 0.5    | 0.5    |
| 10 min | 0.5    | 0.5    | 0.5    |
| 12 min | 0.5    | 0.5    | 0.5    |
| 14 min | 0.5    | 0.5    | 0.5    |
| 16 min | 0.5    | 0.5    | 0.5    |
| 18 min | 0.5    | 0.5    | 0.5    |
| 20 min | 0.5    | 0.5    | 0.5    |
| 22 min | 0.5    | 0.5    | 0.5    |
| 24 min | 0.5    | 0.5    | 0.5    |
| 26 min | 0.5    | 0.5    | 0.5    |
| 0      | 1      | 1      | 1      |
| 2      | 1      | 1      | 1      |
| 4      | 1      | 1      | 1      |
| 6      | 1      | 1      | 1      |
| 8      | 1      | 1      | 1      |
| 10      | 1      | 1      | 1      |
| 12      | 1      | 1      | 1      |
| 2        | 2      | 2      | 2      |
| 4        | 2      | 2      | 2      |
| 6        | 2      | 2      | 2      |
| 8        | 2      | 2      | 2      |
| 10       | 2      | 2      | 2      |
| 12       | 2      | 2      | 2      |
| 2        | 3      | 3      | 3      |
| 4        | 3      | 3      | 3      |
| 6        | 3      | 3      | 3      |
| 8        | 3      | 3      | 3      |
| 10       | 3      | 3      | 3      |
| 12       | 3      | 3      | 3      |
| ...    | ...    | ...    | ...    |
| ...    | ...    | ...    | ...    |
| ...    | ...    | ...    | ...    |
| ...    | ...    | ...    | ...    |
| ...    | ...    | ...    | ...    |
| ...    | ...    | ...    | ...    |
| ...    | ...    | ...    | ...    |
| ...    | ...    | ...    | ...    |
| ...    | ...    |...    | ...    |
| ...    | ...    | ...    | ...    |
| ...    | ...    | ...    | ...    |
| ...    | ...    | ...    | ...    |
| ...    | ...    | ...    | ...    |
| ...    | ...    | ...    | ...    |
| ...    | ...    | ...    | ...    |
| ...    | ...    | ...    | ...    |
| ...    = ?   | ?      | ?      | ?      |
| ?        = ?   | ?      | ?      | ?      |
| ?        = ?   | ?      | ?      | ?      |
| ?        = ?   | ?      | ?      | ?      |
| ?        = ?   | ?      | ?      | ?      |
| ?        = ?   | ?      | ?      | ?      |
| ?        = ?   | ?      | ?      | ?      |
| ?        = ?   (c)   | cB'     | A'     | A'     |
| ?        = ?   (c)   | cB'     | A'     | A'     |
| ?        = ?   (c)   | cB'     | A'     | A'     |
| ?        = ?   (c)   | cB'     | A'     | A'     |
| ?        = ?   (c)   (c) * b* (c) * b* (c) * b* (c) * b* (c) * b* (c) * b* (c) * b* (c) * b* (c) * b* (c) * b* (c) * b* (c) * b* (c) * b* (c) * b* (c) * b* (c) * b* (c) * b* (c) / [a] * b* (c) * b* (c) * b* (c) * b* (c) * b* (c) * b* (c) * b* (c) * b* (c) * b* (c) * b* (c) * b* (c) * b* (c) * b* (c) * b* (c) * b* (c) * b* (c) * b* < .5, < .5, < .5, < .5, < .5, < .5, < .5, < .5, < .5, < .5, < .5, < .5, < .5, < .5, < .5, < .5, < .5, < .5, < .5, < .5, < .5, < .5, < .5, < .5, < .5, < .5 |
PIS: C'B'              PIS: C'B'          PWS: C'B'          PWS: C'B'          PWS: C'B'          PWS: C'B'          PWS: C'B'          PWS: C'B'          PWS: C'B'          PWS: C'B'          PWS: C'B'          PWS: C'B'          PWS: C'B'          PWS: C'B'          PWS: C'B'          PWS: C'B'          PWS: cB'            pB': cB', pC': cB', pD': cB', pE': cB', pF': cB', pG': cB', pH': cB', pI': cB', pJ': cB', pK': cB', pL': cB', pM': cB', pQ': cB', pR': cB', pS': cB', pT': cB', pU': cB', pV': cB', pW': cB', pX': cB', pY': cB', pZ': cB', pW': cB', pX': cB', pY': cB', pZ': cB', pR': cB', pS': cB', pT': cB', pU': cB', pV': cB', pW': cB', pX': cB', pY': cB', pZ': cB', pW': cB', pX': cB', pY': cB', pZ': cB', pW': cB', pX': cB', pY': cB', pZ': cB', pW': cB', pX': cB', pY': cB', pZ': cB', pW': cB', pX': cB', pY': cB', pZ': cB'|
</details>

![](images/e440bc964e79a4f7ece30b5cd29697b51e7d272e4715e10d87300d1cee0d2bc2.jpg)

<details>
<summary>line</summary>

| Wavelength (nm) | Reflectance (AU) - BG-outside | Reflectance (AU) - BG-inner cell | Reflectance (AU) - BG-cell boundary | Reflectance (AU) - Outside | Reflectance (AU) - Inner cell | Reflectance (AU) - Cell boundary |
| --------------- | ----------------------------- | -------------------------------- | ---------------------------------- | -------------------------- | ---------------------------- | ------------------------------- |
| 620             | ~0.7                          | ~0.7                             | ~0.7                               | ~0.7                       | ~0.7                         | ~0.7                            |
| 626             | ~0.95                         | ~0.95                            | ~0.95                              | ~0.95                      | ~0.95                        | ~0.95                           |
| 628             | ~0.7                          | ~0.7                             | ~0.7                               | ~0.7                       | ~0.7                         | ~0.7                            |
| 630             | ~0.7                          | ~0.7                             | ~0.7                               | ~0.7                       | ~0.7                         | ~0.7                            |
</details>

Fig. 3 PIV images and spectra during live stem cell (mHAT) attachment. a Sequence of PROM-acquired images at several time points (0–26 min) during the cell adhesion process. Top row: brightfield (BF) images; middle row: peak intensity shift (PIS) images; bottom row: peak wavelength shift (PWS) images. b PROM-measured spectra at three locations on the PC surface before (A, B, C, in dashed lines) and after (A', B', C', in dotted lines) cell adhesion. Point A' (red) represents the inner cell area; point B' (black) represents the cell boundary; point C' (green) represents the outside of the cell boundary. Scale bar: 20 $\mu$ m

region to that obtained by fluorescence microscopy with a labeled vinculin where the FA areas are concentrated along the cell boundary. Fig. 5b shows two cross-sections along different radial directions (L3 and L4, as shown in Fig. 5a) sampled across the cell diameter for the PWS (blue curves), PIS (red curves), and fluorescence-tagged images (nucleus—green, actin—magenta, vinculin—black). Both the PIS (in red) and vinculin (in black) curves exhibit a similar “ring” effect along the cellular boundary

(highlighted in the light-yellow regions), which indicates that the high intensity in the PIS images are probably colocalized with the FA areas. Therefore, the dimensional change of the FA cluster can also be detected with PIS images using PROM.

Statistical analyses are shown in Fig. 5c (N = 5 cells) for the fluorescent, PIS, and PWS images along the cell boundary (marked as “Edge”) and within the nucleus area (marked as “Inner”). Fig. 5c (Left) displays fluorescence

![](images/c0dbddcc59bbd252ed8198e5fd944d7a2b4a270d8f1bf8a6485f49dba95e1b3a.jpg)  
Fig. 4 Zoomed-in PROM images and cross-section (before and after mHAT cell adhesion). a PROM-acquired images (including PWS, PIS, and BF images) and their overlap images at \~16 min. b Comparison of cross-sectional curves (L1 and L2) of the PWS and PIS images. c Bar graph of the statistical comparison between the edge and the center of the cells. Scale bar: 20 $\mu$ m

images (including actin, nucleus, and vinculin) that demonstrate three different patterns of distribution along the cell edge and center. The nucleus image only shows high intensity within the area of the nucleus because the fluorescent molecular probes only tag nucleic acid material, such as chromosomes. Actin mainly functions as a cytoskeleton molecule that is rapidly remodeled by dynamically forming microfilaments to support the cell structure or participating in many important cellular processes, including cell division or cell signaling. As a scaffold protein, the distribution of actin is relatively uniform, and thus the difference of the fluorescence intensity between the cell edge and center in the actin image is small. Vinculin is a membrane-cytoskeletal protein that is often localized in the FA area because it participates in the linkage between the transmembrane protein (e.g., integrin) and cytoskeletal protein (e.g., actin). Therefore, a fluorescence dye for vinculin is often used to visualize the locations of FAs. Fig. 5c clearly shows that the distribution of vinculin is mainly along the cell

![](images/f55bbe427c1e6c57b6f6320c82cc3375cc04d17daa3e3393eb82fed9fec12b4a.jpg)

![](images/a33a31df65fa96ce5c9cbf86eab3ab0aa00583d03f9f71f223e054652b68f979.jpg)

<details>
<summary>line</summary>

| Length (μm) | Nucleus | Actin | Vinculin |
|-------------|---------|-------|----------|
| 0           | 0.0     | 0.0   | 0.0      |
| 50          | 0.5     | 0.6   | 0.7      |
| 100         | 1.0     | 0.5   | 0.9      |
| 150         | 0.2     | 0.3   | 0.4      |
| 200         | 0.1     | 0.1   | 0.1      |
</details>

![](images/77c613fed140ddd0219bd5aeccf9a35ad52041fc3123eb637e496095a5eef6c4.jpg)

<details>
<summary>line</summary>

| Length (μm) | Nucleus | Actin | Vinculin |
|-------------|---------|-------|----------|
| 0           | 0.0     | 0.0   | 0.0      |
| 50          | 0.0     | 0.0   | 0.6      |
| 100         | 0.8     | 0.6   | 0.3      |
| 150         | 0.0     | 0.4   | 0.7      |
</details>

![](images/f4111ef10a2702767208f3bf4c64815b41c9c18cc8446979585edcf1a1e4dffe.jpg)

<details>
<summary>line</summary>

| Length (μm) | PWS       | PIS       |
|-------------|-----------|-----------|
| 0           | 0.0       | 0.0       |
| 25          | 0.8       | 0.8       |
| 50          | 0.9       | 0.4       |
| 75          | 0.8       | 0.0       |
| 100         | 0.6       | 0.1       |
| 125         | 0.9       | 0.3       |
| 150         | 0.8       | 0.9       |
| 175         | 0.7       | 0.8       |
</details>

![](images/edae7a3b4f643050de2854550949a41fab3ec3b26553c9fafc6003031bdeacba.jpg)

<details>
<summary>line</summary>

| Length (μm) | PWS       | PIS       |
|-------------|-----------|-----------|
| 50          | 0.8       | 0.9       |
| 100         | 0.9       | 0.3       |
| 150         | 0.8       | 0.9       |
</details>

![](images/57a24d03669ba492159603c049269c848d0d6384a95bcac3d2953064b715850f.jpg)

<details>
<summary>bar</summary>

| Protein   | Edge  | Inner |
| --------- | ----- | ----- |
| Actin     | 0.42  | 0.38  |
| Nucleus   | 0.10  | 0.52  |
| Vinculin  | 0.42  | 0.24  |
</details>

![](images/fce79f3d1605a8f94f2c14ab2807bffe54fc01da2e4bd3ca74b7938ef08f622f.jpg)

<details>
<summary>bar</summary>

| Region | PWS (nm) | Normalized PIS (AU) |
|--------|----------|---------------------|
| Edge   | 1.8      | 0.8                 |
| Inner  | 1.9      | 0.1                 |
</details>

Fig. 5 (See legend on next page.)

(see figure on previous page)

Fig. 5 Comparison of label-free images and fluorescence images with cross-section. a Top row: brightfield (BF), PWS, and PIS images; bottom row: fluorescence images, including dyes that selectively stain the nucleus, actin, and vinculin. b Comparison of the image cross-sections through lines (L3 and L4) for the fluorescent (nucleus—green, actin—magenta, vinculin—black), PWS (blue), and PIS (red) images. Light-yellow regions represent the regions near the cell edges, where the vinculin (black curves) and PIS (red curves) both have high intensities. c Statistical comparison among the fluorescence images (actin, nucleus, and vinculin) and label-free images (normalized PWS and PIS value comparison between cell edges and centers). Scale bar: 20 $\mu$ m

periphery for a surface-attached cell, which is highly consistent with the distribution pattern of the PIS (high in "Edge", low in "Inner") (Fig. 5c Right), which is not similar to that of the PWS (Fig. 5c Middle). There is a small difference between the PIS image and vinculin fluorescence image around (outside of) the nucleus. This is likely because the vinculin fluorescence image is a transmission image in the axial direction across the entire cell body (with a thickness of several microns to several tens of microns) including the nucleus area. By contrast, the PIS image is only measured through the evanescent field, which has a thickness that is several hundred times thinner (several tens of nanometers) in the axial direction starting from the bottom of the cell body (before reaching nucleus).

To highlight how information from PROM images complements those obtained by orthogonal imaging modalities, five selected stem cells are shown in Fig. 6a imaged by PWS images, PIS images, confocal images with fluorescence dyes (FL), phase-contrast images (PH), and SEM images. PROM images obtained using PWS and PIS information reveal different features of cell attachment, and both show clear details of the cell attachment boundary. PWS and PIS images highlight only behavior associated with the cell–ECM interface and thus do not show material in the upper cell body. Unlike SEM and fluorescence images, PWS and PIS images yield dynamic and highly quantitative information that can be visualized graphically. As shown in Fig. 6b, c, the stem cell boundary can be tracked along the local normal direction frame-by-frame, enabling the PIS to be sampled along the cell boundary spatially and temporally at the same time. Associated dynamic analyses with different sampling bands (S-Fig. 2, black for band 1—near cell boundary, white for band 2—inner region of the cell) in cells were performed on PWS images and PIS images. The resulting 2D maps shown in Fig. 6b, c represent the PWS and PIS with spatiotemporal information along the cell boundary and time frames. Comparing the different bands between both maps, the PIS increase is much higher in band 1 compared with that in band 2 (Fig. 6c), whereas the PWS shows the opposite trend (Fig. 6b). The dramatic increase of the PIS may be due to the aggregations of FAs along the cell boundaries, which is confirmed in the FL images (Fig. 6a). The mechanisms of the temporal curves for cell adhesion for the PWS and PIS are shown in Fig. 6d, e (N = 5 cells). Directly comparing the dynamics between the PWS and PIS is difficult because they have different dynamic ranges. However, different slopes indicating different increasing ratios along the temporal dimension are observed if the PWS and normalized PIS are plotted together, as shown in Fig. 6f.

The PROM images show cell borders and intra-cell features that are approximately in the order of the pixel size limit in the axial direction, which is near the diffraction limit of the resonant wavelength. It is important to put these images in the context of PCEM images that were gathered previously from high-contrast objects. In the lateral directions, although the effect of a point dielectric object on the reflected wavelength from a PC can extend to the surrounding pixels (because the electric field standing wave “samples” a greater lateral dimension than one period), the outcoupling from a surface adsorbed scatterer (or absorber) is observed to be more highly localized. The full-widths at half-maximums of the point spread function of $TiO_{2}$ nanoparticles (e.g., diameter of $\sim100$ nm) were measured as $\sim1.20-1.56\mu m$ for PWS images and as $\sim0.95\mu m$ for PIS images $^{50}$ . Our previous study shows that the spatial resolution of the dielectric objects in PWV-based PCEM images is directly correlated with the refractive index contrast of the object. Although high-contrast objects (such as a $TiO_{2}$ dielectric nanoparticle or the edge of a photoresist pattern) can extend their “influence” on the measured PWV by as much as a couple of microns (e.g., $2-3\mu m$ ) in any direction, lower-contrast objects have a much more limited perturbation. In PROM images of attached cells, there is an extremely low refractive index contrast between the attached cell membrane and surrounding cell media. Within the footprint of an attached cell, the refractive index contrast between a FA and the neighboring cell membrane regions is even lower. These hypotheses were formed by the contrast observed at the cell attachment border and the contrasting regions of attachment within a cell. Instead of observing smeared borders that extend for several microns, we observed a contrast in PIS and PWS images with micron-scale features. These observations are consistent with our earlier measurements but have been

![](images/5e4b70d06964ddc413808a9cbc8e29a2804f249424e4e98958f3befd14b65f9b.jpg)

<details>
<summary>text_image</summary>

a
PWS
PIS
FL
PH
SEM
Cell 1
Cell 2
Cell 3
Cell 4
Cell 5
0.6 1.5 (nm)
0.4 0.8 (AU)
20 µm
SEM (zoom)
5 µm
</details>

![](images/e44cdd1166644115b2ab078ecc324c6ad6b03d5af90b862ed2953a5718f4c079.jpg)

<details>
<summary>heatmap</summary>

| Sampling window (AU) | Time (min) | Band 1 (nm) | Band 2 (nm) |
|----------------------|------------|-------------|-------------|
| 50                   | 12         | 1.5         | 1.5         |
| 50                   | 40         | 0.6         | 1.5         |
</details>

![](images/af1de135e4d46dafa93bfc2f45a8ef54e413ebf7ff50a8413b9ce6fbd31d081b.jpg)

<details>
<summary>heatmap</summary>

| Sampling window (AU) | Time (min) | PIS band 1 (AU) | PIS band 2 (AU) |
|----------------------|------------|-----------------|-----------------|
| 50                   | 12         | ~0.8            | ~0.8            |
| 50                   | 40         | ~0.8            | ~0.8            |
| 1                    | 12         | ~0.8            | ~0.8            |
| 1                    | 40         | ~0.8            | ~0.8            |
</details>

![](images/3dcf9cf425f9db36f333decb9a85b379d31c3db4453b5e842d40892bc3c15d76.jpg)

<details>
<summary>line</summary>

| Time (min) | Band 1 | Band 2 |
| ---------- | ------ | ------ |
| 10         | 0.9    | 1.2    |
| 20         | 0.8    | 1.1    |
| 30         | 0.9    | 1.2    |
| 40         | 1.0    | 1.3    |
</details>

![](images/dfecb39a107899b5b3788b2d4ba084b31138f49baba5174b08aef5829e88bd3c.jpg)

<details>
<summary>line</summary>

| Time (min) | Band 1 | Band 2 |
| ---------- | ------ | ------ |
| 10         | 0.65   | 0.55   |
| 20         | 0.70   | 0.58   |
| 30         | 0.72   | 0.60   |
| 40         | 0.90   | 0.55   |
</details>

![](images/0c3ec423be1e87457feb857ff95d6cb5ea8113fe262b5ef09ae0a88ed004e8ed.jpg)

<details>
<summary>line</summary>

| Time (min) | PWS (nm) | Normalized PIS (AU) |
| ---------- | -------- | ------------------- |
| 10         | 0.9      | 0.6                 |
| 20         | 0.8      | 0.7                 |
| 30         | 0.9      | 0.8                 |
| 40         | 1.0      | 1.0                 |
</details>

Fig. 6 Image modality comparison and dynamic analysis of PROM images during cell adhesion. a Five selected cells imaged by peak wavelength shift (PWS), peak intensity shift (PIS), confocal fluorescence microscopy (FL) (red—actin, green—vinculin, blue—nucleus), phase-contrast microscopy (PH), and scanning electron microscopy (SEM). Scale bar: 20 $\mu$ m. (Top-right inset: zoom-in SEM image for cell 1. Scale bar: 5 $\mu$ m). 2D spatiotemporal maps are tracked from different bands (band 1—near cell boundary, band 2—inner region of cell) in stem cells with b PWS images and c normalized PIS images. Mean and standard deviation of the temporal curves of cell adhesion for d PWS, e normalized PIS, and f the comparison between PWS and normalized PIS

applied here for the first time in the context of attached cell images of the PIS.

# Conclusions

This study describes a label-free microscopic approach that quantitatively measures the scatter-induced changes in the reflected intensity from a PC biosensor surface to reveal the kinetic evolution and spatial features of FAs that form at the cell–surface interface. Compared to a sensing approach in which image contrast is generated by the dielectric permittivity of attached cell components, PROM provides contrast in the reflected resonant intensity that is induced by the refractive index contrast of the localized protein clusters that occur at the cell–surface interface, which comprise FA sites. Our hypothesis is supported by electromagnetic computer simulations that have modeled small and low refractive index contrast regions on a PC that induce measurable reductions in the resonant reflection efficiency. Our hypothesis is also supported by fluorescence microscopy of cells in which the patterns of FA regions are similarly distributed as patterns of reflected intensity reduction measured by PROM. We show that images of the PIS and PWS can be gathered from the same spectral information for the same cells and that the two imaging modalities have distinct spatial patterns and thus provide complementary information about cell–surface activity. Dynamic images of the PIS and PWS can be repeatedly gathered over extended time periods with a 10-s temporal resolution via a line-scanning approach to generate time-course movies of cell–surface behavior during processes that occur over several hours. As a label-free imaging approach, PROM does not suffer from the limitations of fluorescence-based microscopy, which include photobleaching and stain cytotoxicity. We expect PROM to be a highly useful tool that can reveal the mechanisms of biological processes that occur near the cell membrane when the membrane is attached to ECM materials during cell migration, division, metastasis, apoptosis, and stem cell differentiation.

# Acknowledgements

This work is supported by the National Science Foundation (NSF) Grant CBET 11-32301 and National Institutes of Health (NIH) R01 DK099528 and NIH R21 EB018481. The content is solely the responsibility of the authors and does not necessarily represent the official views of the NSF and NIH. The authors would like to thank the Nano Sensor Groups (NSG), the staff at the Beckman Institute for Advanced Science and Technology, the Micro and Nanotechnology Laboratory (MNTL), the Institute for Genomic Biology (IGB), and the Center for Innovative Instrumentation Technology (CiiT) at the University of Illinois at Urbana-Champaign for their support.

# Author details

$^{1}$ Department of Bioengineering, University of Illinois at Urbana-Champaign, Urbana, IL 61801, USA. $^{2}$ Micro and Nanotechnology Laboratory, University of Illinois at Urbana-Champaign, Urbana, IL 61801, USA. $^{3}$ Department of Chemical and Biomolecular Engineering, University of Illinois at Urbana-Champaign, Urbana, IL 61801, USA. $^{4}$ Carl R. Woese Institute for Genomic Biology, University of Illinois at Urbana-Champaign, Urbana, IL 61801, USA. $^{5}$ Atkins Building, University of Illinois Research Park, 1800 South Oak Street, Champaign, IL 61820, USA. $^{6}$ Department of Electrical and Computer Engineering, University of Illinois at Urbana-Champaign, Urbana, IL 61801, USA

# Authors' contributions

Y.Z. designed the experiments; Y.Z. and J.S.C. performed experiments and wrote the manuscript; Y.Z. and T.M. developed the analysis software and performed data analysis; Y.Z. and H.Y. fabricated the PC sensors; B.T.C. and B.A.H. provided guidance and edited the manuscript.

# Conflict of interest

The authors declare that they have no conflict of interest.

Supplementary information accompanies this paper at https://doi.org/10.1038/s41377-018-0001-5.

Received: 11 May 2017 Revised: 13 February 2018 Accepted: 14 February 2018 Accepted article preview online: 23 February 2018  
Published online: 30 May 2018

# References

1. Davies, P. F. & Tripathi, S. C. Mechanical stress mechanisms and the cell. An endothelial paradigm. Circ. Res. 72, 239–245 (1993).   
2. Schaller, M. D. & Parsons, J. T. Focal adhesion kinase and associated proteins. Curr. Opin. Cell Biol. 6, 705–710 (1994).   
3. Yamada, K. M. & Geiger, B. Molecular interactions in cell adhesion complexes. Curr. Opin. Cell Biol. 9, 76–85 (1997).   
4. Pelham, R. J. Jr. & Wang, Y. Cell locomotion and focal adhesions are regulated by substrate flexibility. Proc. Natl Acad. Sci. USA 94, 13661–13665 (1997).   
5. Turner, C. E. Paxillin and focal adhesion signalling. Nat. Cell Biol. 2, E231–E236 (2000).   
6. Gerthoffer, W. T. & Gunst, S. J. Invited review: focal adhesion and small heat shock proteins in the regulation of actin remodeling and contractility in smooth muscle. J. Appl. Physiol. 91, 963–972 (2001).   
7. Schaller, M. D. Biochemical signals and biological responses elicited by the focal adhesion kinase. Biochim. Biophys. Acta 1540, 1–21 (2001).   
8. Wehrle-Haller, B. & Imhof, B. A. The inner lives of focal adhesions. Trends Cell Biol. 12, 382–389 (2002).   
9. Chen, C. S., Alonso, J. L., Ostuni, E., Whitesides, G. M. & Ingber, D. E. Cell shape provides global control of focal adhesion assembly. Biochem. Biophys. Res Commun. 307, 355–361 (2003).   
10. Carragher, N. O. & Frame, M. C. Focal adhesion and actin dynamics: a place where kinases and proteases meet to promote invasion. Trends Cell Biol. 14, 241–249 (2004).   
11. Owen, G. R., Meredith, D. O., ap Gwynn, I. & Richards, R. G. Focal adhesion quantification - a new assay of material biocompatibility? Review. Eur. Cell Mater. 9, 85–96 (2005). discussion 85-96.   
12. Green, J. A. & Yamada, K. M. Three-dimensional microenvironments modulate fibroblast signaling responses. Adv. Drug Deliv. Rev. 59, 1293–1298 (2007).   
13. Gallant, N. D., Michael, K. E. & García, A. J. Cell adhesion strengthening: contributions of adhesive area, integrin binding, and focal adhesion assembly. Mol. Biol. Cell 16, 4329–4340 (2005).   
14. Wolfenson, H., Henis, Y. I., Geiger, B. & Bershadsky, A. D. The heel and toe of the cell's foot: a multifaceted approach for understanding the structure and dynamics of focal adhesions. Cell Motil. Cytoskelet. 66, 1017–1029 (2009).   
15. Atilgan, E. & Ovryn, B. Nucleation and growth of integrin adhesions. Biophys. J. 96, 3555–3572 (2009).   
16. Frisch, S. M., Vuori, K., Ruoslahti, E. & Chan-Hui, P. Y. Control of adhesion-dependent cell survival by focal adhesion kinase. J. Cell Biol. 134, 793–799 (1996).   
17. Schlaepfer, D. D., Hauck, C. R. & Sieg, D. J. Signaling through focal adhesion kinase. Prog. Biophys. Mol. Biol. 71, 435–478 (1999).   
18. Parsons, J. T., Martin, K. H., Slack, J. K., Taylor, J. M. & Weed, S. A. Focal adhesion kinase: a regulator of focal adhesion dynamics and cell movement. Oncogene 19, 5606–5613 (2000).   
19. Petit, V. & Thiery, J. P. Focal adhesions: structure and dynamics. Biol. Cell 92, 477–494 (2000).

20. Hauck, C. R., Hsia, D. A. & Schlaepfer, D. D. The focal adhesion kinase—a regulator of cell migration and invasion. IUBMB Life 53, 115–119 (2002).   
21. Wozniak, M. A., Modzelewska, K., Kwong, L. & Keely, P. J. Focal adhesion regulation of cell behavior. Biochim. Biophys. Acta 1692, 103–119 (2004).   
22. McLean, G. W. et al. The role of focal-adhesion kinase in cancer - a new therapeutic opportunity. Nat. Rev. Cancer 5, 505–515 (2005).   
23. Parsons, J. T., Horwitz, A. R. & Schwartz, M. A. Cell adhesion: integrating cytoskeletal dynamics and cellular tension. Nat. Rev. Mol. Cell Biol. 11, 633–643 (2010).   
24. Mitra, S. K., Hanson, D. A. & Schlaepfer, D. D. Focal adhesion kinase: in command and control of cell motility. Nat. Rev. Mol. Cell Biol. 6, 56–68 (2005).   
25. Damiano, J. S. & Dalton, W. S. Integrin-mediated drug resistance in multiple myeloma. Leuk. Lymphoma 38, 71–81 (2000).   
26. Hazlehurst, L. A., Landowski, T. H. & Dalton, W. S. Role of the tumor microenvironment in mediating de novo resistance to drugs and physiological mediators of cell death. Oncogene 22, 7396–7402 (2003).   
27. McCulloch, C. A., Downey, G. P. & El-Gabalawy, H. Signalling platforms that modulate the inflammatory response: new targets for drug development. Nat. Rev. Drug Discov. 5, 864–876 (2006).   
28. Zhao, X. & Guan, J. L. Focal adhesion kinase and its signaling pathways in cell migration and angiogenesis. Adv. Drug Deliv. Rev. 63, 610–615 (2011).   
29. Kim, D. H. & Wirtz, D. Focal adhesion size uniquely predicts cell migration. FASEB J. 27, 1351–1361 (2013).   
30. Kanchanawong, P. et al. Nanoscale architecture of integrin-based cell adhesions. Nature 468, 580–584 (2010).   
31. Geiger, B., Spatz, J. P. & Bershadsky, A. D. Environmental sensing through focal adhesions. Nat. Rev. Mol. Cell Biol. 10, 21–33 (2009).   
32. Kusumi, A., Tsunoyama, T. A., Hirosawa, K. M. & Kasai, R. S. & Fujiwara, T. K. Tracking single molecules at work in living cells. Nat. Chem. Biol. 10, 524–532 (2014).   
33. Oakes, P. W. & Gardel, M. L. Stressing the limits of focal adhesion mechanosensitivity. Curr. Opin. Cell Biol. 30, 68–73 (2014).   
34. Stehbens, S. J. & Wittmann, T. Analysis of focal adhesion turnover: a quantitative live-cell imaging example. Methods Cell Biol. 123, 335–346 (2014).   
35. Deschout, H. et al. Complementarity of PALM and SOFI for super-resolution live-cell imaging of focal adhesions. Nat. Commun. 7, 13693 (2016).   
36. Maziveyi, M. & Alahari, S. K. Cell matrix adhesions in cancer: the proteins that form the glue. Oncotarget 8, 48471–48487 (2017).   
37. Legant, W. R. et al. Multidimensional traction force microscopy reveals out-of-plane rotational moments about focal adhesions. Proc. Natl. Acad. Sci. 110, 881–886 (2013).   
38. Colin-York, H. et al. Super-resolved traction force microscopy (STFM). Nano Lett. 16, 2633–2638 (2016).   
39. Sarangi, B. R. et al. Coordination between intra- and extracellular forces regulates focal adhesion dynamics. Nano Lett. 17, 399–406 (2017).   
40. Franz, C. M. & Muller, D. J. Analyzing focal adhesion structure by atomic force microscopy. J. Cell Sci. 118, 5315–5323 (2005).   
41. von Bilderling, C., Caldarola, M., Masip, M. E., Bragas, A. V. & Pietrasanta, L. I. Monitoring in real-time focal adhesion protein dynamics in response to a discrete mechanical stimulus. Rev. Sci. Instrum. 88, 013703 (2017).   
42. Grashoff, C. et al. Measuring mechanical tension across vinculin reveals regulation of focal adhesion dynamics. Nature 466, 263–266 (2010).   
43. Figel, S. & Gelman, I. H. Focal adhesion kinase controls prostate cancer progression via intrinsic kinase and scaffolding functions. Anticancer Agents Med. Chem. 11, 607–616 (2011).   
44. Brooks, J., Watson, A. & Korcsmaros, T. Omics approaches to identify potential biomarkers of inflammatory diseases in the focal adhesion complex. Genomics Proteomics Bioinformatics 15, 101–109 (2017).   
45. Reticker-Flynn, N. E. et al. A combinatorial extracellular matrix platform identifies cell-extracellular matrix interactions that correlate with metastasis. Nat. Commun. 3, 1122 (2012).   
46. Zhou, T., Marx, K. A., Dewilde, A. H., McIntosh, D. & Braunhut, S. J. Dynamic cell adhesion and viscoelastic signatures distinguish normal from malignant human mammary cells using quartz crystal microbalance. Anal. Biochem. 421, 164–171 (2012).   
47. Smolyakov, G. et al. Elasticity, adhesion, and tether extrusion on breast cancer cells provide a signature of their invasive potential. ACS Appl. Mater. Interfaces 8, 27426–27431 (2016).

48. Berginski, M. E., Vitriol, E. A., Hahn, K. M. & Gomez, S. M. High-resolution quantification of focal adhesion spatiotemporal dynamics in living cells. PLoS ONE 6, e22025 (2011).   
49. Chen, W. et al. Photonic crystal enhanced microscopy for imaging of live cell adhesion. Analyst 138, 5886–5894 (2013).   
50. Zhuo, Y. et al. Single nanoparticle detection using photonic crystal enhanced microscopy. Analyst 139, 1007–1015 (2014).   
51. Zhuo, Y. & Cunningham, B. T. Label-free biosensor imaging on photonic crystal surfaces. Sensors 15, 21613–21635 (2015).   
52. Zhuo, Y. et al. Quantitative imaging of cell membrane-associated effective mass density using photonic crystal enhanced microscopy (PCEM). Prog. Quantum Electron. 50, 1–18 (2016).   
53. Cunningham, B. T., Li, P., Lin, B. & Pepper, J. Colorimetric resonant reflection as a direct biochemical assay technique. Sens. Actuators B Chem. 81, 316–328 (2002).   
54. Cunningham, B. T. et al. A plastic colorimetric resonant optical biosensor for multiparallel detection of label-free biochemical interactions. Sens. Actuators B Chem. 85, 219–226 (2002).   
55. Cunningham, B. T. et al. Label-free assays on the BIND system. J. Biomol. Screen 9, 481–490 (2004).   
56. Hessel, A. & Oliner, A. A. A new theory of Wood's anomalies on optical gratings. Appl. Opt. 4, 1275-1297 (1965).   
57. Yeh, P., Yariv, A. & Cho, A. Y. Optical surface waves in periodic layered media. Appl. Phys. Lett. 32, 104–105 (1978).   
58. Mashev, L. & Popov, E. Diffraction efficiency anomalies of multicoated dielectric gratings. Opt. Commun. 51, 131–136 (1984).   
59. Popov, E., Mashev, L. & Maystre, D. Theoretical study of the anomalies of coated dielectric gratings. Opt. Acta 33, 607–619 (1986).   
60. John, S. Strong localization of photons in certain disordered dielectric superlattices. Phys. Rev. Lett. 58, 2486–2489 (1987).   
61. Yablonovitch, E. Inhibited spontaneous emission in solid-state physics and electronics. Phys. Rev. Lett. 58, 2059–2062 (1987).   
62. Meade, R. D., Brommer, K. D., Rappe, A. M. & Joannopoulos, J. D. Electromagnetic Bloch waves at the surface of a photonic crystal. Phys. Rev. B 44, 10961–10964 (1991).   
63. Magnusson, R. & Wang, S. S. New principle for optical filters. Appl. Phys. Lett. 61, 1022–1024 (1992).   
64. Fan, S., Villeneuve, P. R., Joannopoulos, J. D. & Schubert, E. F. High extraction efficiency of spontaneous emission from slabs of photonic crystals. Phys. Rev. Lett. 18, 3294–3297 (1997).   
65. Joannopoulos, J. D., Villeneuve, P. R. & Fan, S. Photonic crystals: putting a new twist on light. Nature 386, 143–149 (1997).   
66. Kanskar, M. et al. Observation of leaky slab modes in an air-bridged semiconductor waveguide with a two-dimensional photonic lattice. Appl. Phys. Lett. 70, 1438–1440 (1997).   
67. Johnson, S. G., Fan, S., Villeneuve, P. R., Joannopoulos, J. D. & Kolodziejski, L. A. Guided modes in photonic crystal slabs. Phys. Rev. B 60, 5751–5758 (1999).   
68. Boroditsky, M. et al. Spontaneous emission extraction and Purcell enhancement from thin-film 2-D photonic crystals. J. Light Technol. 17, 2096–2112 (1999).   
69. Painter, O., Vuckovic, J. & Scherer, A. Defect modes of a two-dimensional photonic crystal in an optically thin dielectric slab. J. Opt. Soc. Am. B 16, 275–285 (1999).   
70. Robertson, W. M. & May, M. S. Surface electromagnetic wave excitation on one-dimensional photonic band gap arrays. Appl. Phys. Lett. 74, 1800–1802 (1999).   
71. Lin, S. Y., Chow, E., Johnson, S. G. & Joannopoulos, J. D. Demonstration of highly efficient waveguiding in a photonic crystal slab at the 1.5-um wavelength. Opt. Lett. 25, 1297–1299 (2000).   
72. Pacradouni, V. et al. Photonic band structure of dielectric membranes periodically textured in two dimensions. Phys. Rev. B 62, 4204–4207 (2000).   
73. Kuchinsky, S., Allan, D. C., Borrelli, N. F. & Cotteverte, J.-C. 3D localization in a channel waveguide in a photonic crystal with 2D periodicity. Opt. Commun. 175, 147–152 (2000).   
74. Benisty, H. et al. Radiation losses of waveguide-based two-dimensional photonic crystals: positive role of the substrate. Appl. Phys. Lett. 76, 532–534 (2000).   
75. Chutinan, A. & Noda, S. Waveguides and waveguide bends in two-dimensional photonic crystal slabs. Phys. Rev. B 62, 4488–4492 (2000).   
76. Joshi, B. et al. Phosphorylated caveolin-1 regulates Rho/ROCK-dependent focal adhesion dynamics and tumor cell migration and invasion. Cancer Res. 68, 8210–8220 (2008).

77. Liu, J. N., Schulmerich, M. V., Bhargava, R. & Cunningham, B. T. Sculpting narrowband Fano resonances inherent in the large-area mid-infrared photonic crystal microresonators for spectroscopic imaging. Opt. Express 22, 18142–18158 (2014).   
78. Chuang, S. L. Physics of Photonic Devices, 2nd edn (John Wiley & Sons Inc, New Jersey, USA, 2009).   
79. Foreman, M. Cavity Coupled Photonic Crystal Enhanced Fluorescence for High Sensitivity Biomarker Detection. MSc thesis. University of Illinois at Urbana-Champaign (2016).   
80. Joannopoulos, J. D., Johnson, S. G., Winn, J. N. & Meade, R. D. Photonic Crystals: Molding the Flow of Light, 2nd edn, (Princeton University Press, Princeton, 2008).

81. Chen, W. L. et al. Enhanced live cell imaging via photonic crystal enhanced fluorescence microscopy. Analyst 139, 5954–5963 (2014).   
82. Backman, V. et al. Polarized light scattering spectroscopy for quantitative measurement of epithelial cellular structures in situ. IEEE J. Sel. Top. Quantum Electron. 5, 1019–1026 (1999).   
83. Chandler, J. E., Cherkezyan, L., Subramanian, H. & Backman, V. Nanoscale refractive index fluctuations detected via sparse spectral microscopy. Biomed. Opt. Express 7, 883–893 (2016).   
84. Miao, Q., Derbas, J., Eid, A., Subramanian, H. & Backman, V. Automated cell selection using support vector machine for application to spectral nanocytology. Biomed. Res Int. 2016, 6090912 (2016).